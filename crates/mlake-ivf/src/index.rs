//! IVF index build and query.
//!
//! Items are partitioned across cluster files by nearest centroid. A query probes the
//! `nprobe` nearest centroids, fetches those cluster files, and re-ranks exactly over the
//! vectors it fetched.
//!
//! Exact re-rank is affordable precisely because the payload is inline: the cluster file
//! already had to be fetched to get the items, so the vectors come along for free and
//! there is nothing to gain from an approximate distance.

use std::collections::HashMap;

use mlake_core::{MemoryId, StoredMemory};
use rkyv::{Archive, Deserialize, Serialize};

use crate::kmeans::{self, nearest};

/// A cluster file: the items assigned to one centroid.
///
/// Sized to 2–8 MB (SPEC §5.1) so that fetching one is a single coalesced ranged GET.
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug, Default)]
#[archive(check_bytes)]
pub struct ClusterFile {
    pub centroid_id: u32,
    pub items: Vec<StoredMemory>,
}

/// Object header magic + version for a cluster file (`cluster-{i}.bin`).
const CLUSTER_MAGIC: &[u8; 4] = b"CLUS";
const CLUSTER_VERSION: u16 = 1;

impl ClusterFile {
    pub fn to_bytes(&self) -> Result<Vec<u8>, mlake_core::Error> {
        let payload = rkyv::to_bytes::<_, 65536>(self)
            .map_err(|e| mlake_core::Error::Encode(e.to_string()))?;
        Ok(mlake_core::envelope::wrap(CLUSTER_MAGIC, CLUSTER_VERSION, &payload))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, mlake_core::Error> {
        let (_version, payload) = mlake_core::envelope::unwrap(CLUSTER_MAGIC, bytes)
            .ok_or_else(|| mlake_core::Error::Decode("cluster file: bad header".into()))?;
        // Shared validated, alignment-tolerant rkyv read (see mlake_core::rkyv_io).
        mlake_core::rkyv_read(payload)
            .ok_or_else(|| mlake_core::Error::Decode("cluster file".into()))
    }
}

/// The centroid table, held in memory on every query node. Small — `sqrt(N)` vectors —
/// and hot, so it lives in the in-memory ARC tier rather than being re-fetched.
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug, Default)]
#[archive(check_bytes)]
pub struct Centroids {
    pub vectors: Vec<Vec<f32>>,
    /// Number of items in each cluster, used to decide splits and merges.
    pub sizes: Vec<usize>,
    pub dim: usize,
    /// Cluster radius: `max ‖v̂ − c‖` over the cluster's members, where `v̂` is the member
    /// *direction* (see [`member_radius`]). One `f32` per cluster, parallel to `vectors`.
    ///
    /// This is what turns "how far is the query from this centroid" into "what is the best
    /// score anything in this cluster could possibly have" — the geometry that lets a probe
    /// retire a cluster without reading it ([`Centroids::max_similarity`]).
    ///
    /// May be **shorter than `vectors`, or hold a non-finite entry**: a centroid table written
    /// before radii existed, or one whose radius has not been recomputed since members were
    /// added. Both mean *unknown*, and every reader must treat unknown as `+∞` (retires
    /// nothing) rather than as zero — a radius that is too small silently drops results.
    pub radii: Vec<f32>,
}

/// The distance a member contributes to its cluster's radius: `‖v̂ − c‖`, where
/// `v̂ = v/‖v‖` is the member's direction (and `0` for a zero/absent vector).
///
/// Normalising the member — not the centroid — is the whole trick. See
/// [`Centroids::max_similarity`] for why.
pub fn member_radius(centroid: &[f32], member: &[f32]) -> f32 {
    let n = mlake_core::norm(member);
    let mut acc = 0.0f32;
    for (i, ci) in centroid.iter().enumerate() {
        // A shorter/absent member is read as zeros — the same convention `dot` uses, and the
        // same one a text-only memory (no embedding) gets when its block row is zero-padded.
        let v = if n > 0.0 { member.get(i).copied().unwrap_or(0.0) / n } else { 0.0 };
        let d = v - ci;
        acc += d * d;
    }
    acc.sqrt()
}

impl Centroids {
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// The radius of cluster `i`, or `None` when it is unknown (absent, stale-shaped, or
    /// non-finite). `None` must be read as "no bound", never as zero.
    pub fn radius(&self, i: usize) -> Option<f32> {
        if self.radii.len() != self.vectors.len() {
            return None;
        }
        self.radii.get(i).copied().filter(|r| r.is_finite() && *r >= 0.0)
    }

    /// An **upper bound on the cosine similarity** between `query` and *any* member of
    /// cluster `i`. `+∞` when the radius is unknown, which retires nothing.
    ///
    /// # Derivation
    ///
    /// Let `q̂ = q/‖q‖`. Cosine ignores the length of both sides, so for a member `v` with
    /// `‖v‖ > 0`, writing `v̂ = v/‖v‖`:
    ///
    /// ```text
    ///   cos(q, v) = ⟨q̂, v̂⟩
    ///             = ⟨q̂, c⟩ + ⟨q̂, v̂ − c⟩
    ///            ≤ ⟨q̂, c⟩ + ‖q̂‖ · ‖v̂ − c‖      (Cauchy–Schwarz)
    ///             = ⟨q̂, c⟩ + ‖v̂ − c‖            (‖q̂‖ = 1)
    ///            ≤ ⟨q̂, c⟩ + R                   (R = max over members of ‖v̂ − c‖)
    /// ```
    ///
    /// **Note what is *not* assumed.** The centroid `c` is a mean, so `‖c‖ ≠ 1`; nothing above
    /// needs it to be. Only `q̂` is normalised, and it is normalised here rather than assumed.
    /// Members are not assumed unit-length either — the radius is measured against the member
    /// *direction* `v̂`, which is the only thing cosine can see. Measuring it against the raw
    /// `v` instead would be unsound the moment a caller sent an unnormalised vector (and
    /// `uniform_dim` exists precisely because callers are not trusted to send what we expect):
    /// `⟨q̂, v⟩ ≤ ⟨q̂,c⟩ + ‖v − c‖` bounds the *dot product*, and dividing by `‖v‖ < 1` to get
    /// the cosine would inflate it past the bound.
    ///
    /// A zero-length member scores exactly `0.0` everywhere in this codebase (`cosine_opt`),
    /// and contributes `‖0 − c‖ = ‖c‖` to the radius. That is sound rather than a special
    /// case: `⟨q̂,c⟩ + ‖c‖ ≥ 0` by Cauchy–Schwarz, so the bound covers its true score of 0.
    ///
    /// A second, independent bound comes from the triangle inequality — the form usually
    /// written for IVF. `q̂` and `v̂` are both unit, so `cos = 1 − ‖q̂ − v̂‖²/2`, and
    /// `‖q̂ − v̂‖ ≥ ‖q̂ − c‖ − R`, giving `cos ≤ 1 − max(0, ‖q̂ − c‖ − R)²/2`. Neither bound
    /// dominates: the triangle form is tighter near the query (it can never exceed 1, which
    /// Cauchy–Schwarz can), the Cauchy–Schwarz form is much tighter far from it. The minimum
    /// of the two is taken, which is still a valid bound because both are.
    ///
    /// A zero-length member is covered by the triangle form too: `v̂ = 0` puts `‖q̂ − v̂‖ = 1`,
    /// and `‖q̂ − c‖ − R ≤ ‖q̂ − c‖ − ‖c‖ ≤ 1`, so the inequality still holds.
    ///
    /// The bound is *absolute*, not probabilistic — unlike the RaBitQ member bounds it is
    /// compared against.
    pub fn max_similarity(&self, query: &[f32], i: usize) -> f32 {
        let Some(r) = self.radius(i) else { return f32::INFINITY };
        let Some(c) = self.vectors.get(i) else { return f32::INFINITY };
        // A dimension mismatch means the caller is not asking about this space; refuse to
        // bound rather than bound wrongly.
        if query.len() != c.len() {
            return f32::INFINITY;
        }
        let qn = mlake_core::norm(query);
        if qn <= 0.0 || qn.is_nan() {
            return f32::INFINITY;
        }
        // ⟨q̂,c⟩ and ‖q̂ − c‖² in one pass.
        let mut qc = 0.0f32;
        let mut cc = 0.0f32;
        for (j, cj) in c.iter().enumerate() {
            qc += query[j] * cj;
            cc += cj * cj;
        }
        qc /= qn;
        let cauchy_schwarz = qc + r;
        // ‖q̂ − c‖² = 1 − 2⟨q̂,c⟩ + ‖c‖², clamped: f32 error can make it slightly negative.
        let gap = (1.0 - 2.0 * qc + cc).max(0.0).sqrt() - r;
        let triangle = if gap > 0.0 { 1.0 - gap * gap / 2.0 } else { 1.0 };
        cauchy_schwarz.min(triangle)
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The `nprobe` centroids nearest a query vector, nearest first.
    ///
    /// This ordering is what bounds the work: everything outside these clusters is never
    /// read, so query cost depends on `nprobe` rather than on corpus size.
    ///
    /// Distance must be measured the same way assignment measured it. Centroids are
    /// means, so they are not unit-length even when every item is; ranking them by cosine
    /// while items were assigned by euclidean distance sends a query to a different set
    /// of clusters than its neighbours live in, which measurably costs recall.
    pub fn probe(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, c)| (i, crate::kmeans::sq_dist_pub(query, c)))
            .collect();
        // Sort by distance, breaking ties by index so probing is deterministic.
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        scored.into_iter().take(nprobe).map(|(i, _)| i).collect()
    }

    /// The centroid a vector is assigned to: its single nearest. This is the assign-only
    /// path — new items are placed against the *existing* centroids without retraining, so
    /// a fold does not reshuffle the whole corpus (SCALE.md Phase 3).
    pub fn assign(&self, v: &[f32]) -> usize {
        crate::kmeans::nearest(&self.vectors, v)
    }

    /// Append a centroid (used by a local split) and return its index.
    ///
    /// The new cluster's radius is `+∞` — unknown — until the fold recomputes it. A split
    /// that left it at `0.0` would claim the cluster contains nothing but its own centroid.
    pub fn push(&mut self, vector: Vec<f32>) -> usize {
        self.vectors.push(vector);
        self.sizes.push(0);
        self.radii.push(f32::INFINITY);
        self.vectors.len() - 1
    }

    /// Recompute every cluster's radius from its members. **Every fold must call this**, not
    /// only a retraining one: the assign-only path adds members to centroids it did not
    /// retrain, so a radius carried forward from the previous generation is too small — i.e.
    /// unsound — the moment one new member lands outside it.
    ///
    /// `members(i)` yields cluster `i`'s member vectors. An empty cluster gets radius `0.0`
    /// (nothing can be in it, so nothing can score); that is the one place a finite radius is
    /// written without seeing a member, and it is exact.
    pub fn recompute_radii<'a, F, I>(&mut self, mut members: F)
    where
        F: FnMut(usize) -> I,
        I: IntoIterator<Item = &'a [f32]>,
    {
        let mut radii = Vec::with_capacity(self.vectors.len());
        for i in 0..self.vectors.len() {
            let c = &self.vectors[i];
            let mut r = 0.0f32;
            for v in members(i) {
                let d = member_radius(c, v);
                if d > r {
                    r = d;
                }
            }
            radii.push(r);
        }
        self.radii = radii;
    }

    /// Encode as raw little-endian f32, not JSON.
    ///
    /// `[dim: u32][count: u32][vectors: f32 × dim × count][sizes: u32 × count]`
    ///
    /// A f32 costs 4 bytes here and 12–20 as decimal text, and this table is read on every
    /// snapshot open and held resident — so the encoding is pure open-path cost. Raw also
    /// parses without a tokenizer, which matters more than the bytes on a cold node.
    pub fn to_bytes(&self) -> Result<Vec<u8>, mlake_core::Error> {
        Ok(mlake_core::envelope::wrap(CENTROIDS_MAGIC, CENTROIDS_VERSION, &mlake_core::rkyv_write(self)))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, mlake_core::Error> {
        let (version, payload) = mlake_core::envelope::unwrap(CENTROIDS_MAGIC, bytes).ok_or_else(
            || mlake_core::Error::Decode(format!("centroid table: bad header ({} bytes)", bytes.len())),
        )?;
        // v1 predates `radii`, and rkyv's layout is positional — a v1 buffer read as a v2
        // struct is not a missing field, it is a misparse. Decode it as what it is and leave
        // `radii` empty, which every reader treats as "radius unknown" (no cluster retires).
        if version < CENTROIDS_VERSION {
            let old: CentroidsV1 = mlake_core::rkyv_read(payload).ok_or_else(|| {
                mlake_core::Error::Decode(format!("malformed v1 centroid table ({} bytes)", payload.len()))
            })?;
            return Ok(Centroids {
                vectors: old.vectors,
                sizes: old.sizes,
                dim: old.dim,
                radii: Vec::new(),
            });
        }
        // An empty table round-trips as the default (empty rkyv archive).
        mlake_core::rkyv_read(payload).ok_or_else(|| {
            mlake_core::Error::Decode(format!("malformed centroid table ({} bytes)", payload.len()))
        })
    }
}

/// The pre-radii centroid table, kept only so a generation written before adaptive probing
/// still opens. Never written.
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug, Default)]
#[archive(check_bytes)]
struct CentroidsV1 {
    vectors: Vec<Vec<f32>>,
    sizes: Vec<usize>,
    dim: usize,
}

/// Object header magic + version for the centroid table (`centroids.bin`).
const CENTROIDS_MAGIC: &[u8; 4] = b"CENT";
/// v2 adds `radii` (per-cluster radius) for adaptive probing.
const CENTROIDS_VERSION: u16 = 2;

/// A scored search hit.
#[derive(Clone, PartialEq, Debug)]
pub struct Hit {
    pub id: MemoryId,
    pub score: f32,
}

/// Assign items to clusters and emit the cluster files.
///
/// Deterministic given the same items and centroids: item order within a cluster follows
/// input order, so a replay reproduces byte-identical files (G-6).
pub fn build_clusters(items: Vec<StoredMemory>, centroids: &Centroids) -> Vec<ClusterFile> {
    let mut buckets: Vec<Vec<StoredMemory>> = vec![Vec::new(); centroids.len().max(1)];
    for item in items {
        let c = if centroids.is_empty() {
            0
        } else {
            nearest(&centroids.vectors, &item.vector)
        };
        buckets[c].push(item);
    }
    buckets
        .into_iter()
        .enumerate()
        .map(|(centroid_id, items)| ClusterFile {
            centroid_id: centroid_id as u32,
            items,
        })
        .collect()
}

/// Cap on the number of vectors used to *train* centroids. Beyond this, training runs on a
/// deterministic random sample (mini-batch k-means, SPEC §5.1): a few tens of thousands of
/// points fix the centroid geometry as well as millions, and it keeps a 1M+ build feasible.
/// Every vector is still *assigned* against the trained centroids.
const TRAIN_SAMPLE_CAP: usize = 50_000;

/// Train centroids over a corpus. For large corpora the training set is a deterministic
/// sample; assignment covers everything.
pub fn train_centroids(vectors: &[Vec<f32>], seed: u64) -> Centroids {
    if vectors.is_empty() {
        return Centroids::default();
    }
    let k = kmeans::centroid_count(vectors.len());

    let sampled: Vec<Vec<f32>>;
    let train_set: &[Vec<f32>] = if vectors.len() > TRAIN_SAMPLE_CAP {
        let mut rng = kmeans::Rng::seeded(seed ^ 0xA5A5_A5A5);
        // Reservoir-free stride+jitter sample: deterministic and evenly spread.
        let mut idxs: Vec<usize> = Vec::with_capacity(TRAIN_SAMPLE_CAP);
        let stride = vectors.len() / TRAIN_SAMPLE_CAP;
        let mut i = rng.below(stride.max(1));
        while i < vectors.len() && idxs.len() < TRAIN_SAMPLE_CAP {
            idxs.push(i);
            i += stride.max(1);
        }
        sampled = idxs.into_iter().map(|j| vectors[j].clone()).collect();
        &sampled
    } else {
        vectors
    };

    let iters = if vectors.len() > TRAIN_SAMPLE_CAP { 15 } else { 25 };
    let trained = kmeans::train(train_set, k, iters, seed);
    // Cluster-size histogram over *every* vector (not the sample): this O(N·k) pass is one of
    // the two dominant fold costs at scale, and each vector's nearest-centroid is independent,
    // so compute the assignments in parallel and only the (cheap) tally is serial.
    let mut sizes = vec![0usize; trained.len()];
    // The same pass yields each cluster's radius for free: it already knows every vector's
    // nearest centroid, and the radius is one more reduction over that. It is recomputed at
    // fold anyway (assign-only folds add members here), but seeding it makes the in-process
    // engine and any direct caller sound without a second O(N·k) pass.
    let mut radii = vec![0.0f32; trained.len()];
    {
        use rayon::prelude::*;
        let assigns: Vec<(usize, f32)> = vectors
            .par_iter()
            .map(|v| {
                let c = nearest(&trained, v);
                (c, member_radius(&trained[c], v))
            })
            .collect();
        for (a, r) in assigns {
            sizes[a] += 1;
            if r > radii[a] {
                radii[a] = r;
            }
        }
    }
    Centroids {
        dim: vectors[0].len(),
        vectors: trained,
        sizes,
        radii,
    }
}

/// Train centroids on a caller-provided sample with an explicit cluster count `k`.
///
/// Unlike [`train_centroids`], this never scans the full corpus: `k` comes from the *total* N
/// (√N — the caller knows it) while training runs on the bounded `sample`, and `sizes` is only
/// the sample's histogram. The external-memory fold uses this so centroid training touches at
/// most `sample.len()` vectors, and it overwrites `sizes` with the true per-cluster counts it
/// tallies during the streaming assignment pass.
pub fn train_centroids_k(sample: &[Vec<f32>], k: usize, seed: u64) -> Centroids {
    if sample.is_empty() {
        return Centroids::default();
    }
    let trained = kmeans::train(sample, k.max(1), 15, seed);
    let mut sizes = vec![0usize; trained.len()];
    for v in sample {
        sizes[nearest(&trained, v)] += 1;
    }
    // No radii: a radius derived from a *sample* is smaller than the true one, and a radius
    // that is too small silently drops results. The streaming fold overwrites this with the
    // radii it tallies over every vector during its assignment pass, exactly as it does with
    // `sizes`. Left empty here so a caller that forgets gets "unknown" (no pruning), not a
    // confidently wrong bound.
    let radii = Vec::new();
    Centroids { dim: sample[0].len(), vectors: trained, sizes, radii }
}

/// Exact search over a set of items — the re-rank step, and the ground truth the recall
/// gate measures against.
pub fn exact_search(items: &[StoredMemory], query: &[f32], k: usize) -> Vec<Hit> {
    // The query is fixed across every candidate, so compute its norm once rather than letting
    // `cosine` recompute it per item (a third of the loop's work). Result is identical.
    let qn = mlake_core::norm(query);
    let mut hits: Vec<Hit> = items
        .iter()
        .map(|item| Hit {
            id: item.id,
            // `_opt`: a text-only memory carries no embedding and scores 0 rather than
            // being an error. A genuine dimension mismatch still panics.
            score: mlake_core::cosine_opt_prenorm(query, qn, &item.vector),
        })
        .collect();
    sort_hits(&mut hits);
    hits.truncate(k);
    hits
}

/// Sort by descending score, breaking ties by id so results are stable across runs.
pub fn sort_hits(hits: &mut [Hit]) {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
}

/// Merge hits from several sources, keeping the best score per id.
pub fn merge_hits(sources: Vec<Vec<Hit>>, k: usize) -> Vec<Hit> {
    let mut best: HashMap<MemoryId, f32> = HashMap::new();
    for source in sources {
        for hit in source {
            best.entry(hit.id)
                .and_modify(|s| {
                    if hit.score > *s {
                        *s = hit.score;
                    }
                })
                .or_insert(hit.score);
        }
    }
    let mut hits: Vec<Hit> = best
        .into_iter()
        .map(|(id, score)| Hit { id, score })
        .collect();
    sort_hits(&mut hits);
    hits.truncate(k);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlake_core::memory::Timestamps;

    fn stored(key: &str, vector: Vec<f32>) -> StoredMemory {
        StoredMemory {
            id: MemoryId::from_key(key),
            vector,
            text: key.to_string(),
            index_text: String::new(),
            memory_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![],
            semantic_out: vec![],
            causal_out: vec![],
            metadata: vec![],
            write_seq: 0,
        }
    }

    #[test]
    fn cluster_file_roundtrips() {
        let cf = ClusterFile {
            centroid_id: 3,
            items: vec![stored("a", vec![1.0, 0.0]), stored("b", vec![0.0, 1.0])],
        };
        let bytes = cf.to_bytes().unwrap();
        assert_eq!(ClusterFile::from_bytes(&bytes).unwrap(), cf);
    }

    #[test]
    fn cluster_file_decodes_when_misaligned() {
        let cf = ClusterFile {
            centroid_id: 0,
            items: vec![stored("a", vec![1.0, 0.0])],
        };
        let encoded = cf.to_bytes().unwrap();
        let mut padded = vec![0u8; encoded.len() + 1];
        padded[1..].copy_from_slice(&encoded);
        assert_eq!(ClusterFile::from_bytes(&padded[1..]).unwrap(), cf);
    }

    #[test]
    fn probe_returns_nearest_centroids_first() {
        let centroids = Centroids {
            vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]],
            sizes: vec![1, 1, 1],
            radii: Vec::new(),
            dim: 2,
        };
        let probed = centroids.probe(&[0.9, 0.1], 2);
        assert_eq!(probed[0], 0, "closest centroid must be probed first");
        assert_eq!(probed.len(), 2);
    }

    #[test]
    fn probe_is_capped_by_centroid_count() {
        let centroids = Centroids {
            vectors: vec![vec![1.0, 0.0]],
            sizes: vec![1],
            radii: Vec::new(),
            dim: 2,
        };
        assert_eq!(centroids.probe(&[1.0, 0.0], 8).len(), 1);
    }

    #[test]
    fn every_item_lands_in_exactly_one_cluster() {
        let items = vec![
            stored("a", vec![1.0, 0.0]),
            stored("b", vec![0.9, 0.1]),
            stored("c", vec![0.0, 1.0]),
        ];
        let centroids = train_centroids(
            &items.iter().map(|i| i.vector.clone()).collect::<Vec<_>>(),
            42,
        );
        let clusters = build_clusters(items.clone(), &centroids);
        let total: usize = clusters.iter().map(|c| c.items.len()).sum();
        assert_eq!(total, items.len(), "no item may be dropped or duplicated");
    }

    #[test]
    fn exact_search_ranks_by_cosine() {
        let items = vec![
            stored("far", vec![0.0, 1.0]),
            stored("near", vec![1.0, 0.0]),
            stored("mid", vec![0.7, 0.7]),
        ];
        let hits = exact_search(&items, &[1.0, 0.0], 3);
        assert_eq!(hits[0].id, MemoryId::from_key("near"));
        assert_eq!(hits[1].id, MemoryId::from_key("mid"));
        assert_eq!(hits[2].id, MemoryId::from_key("far"));
    }

    #[test]
    fn merge_keeps_the_best_score_per_id() {
        // The same item can surface from both a cluster and the WAL tail; it must appear
        // once, at its best score.
        let id = MemoryId::from_key("dup");
        let merged = merge_hits(
            vec![
                vec![Hit { id, score: 0.5 }],
                vec![Hit { id, score: 0.9 }],
            ],
            10,
        );
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].score, 0.9);
    }

    #[test]
    fn hit_ordering_is_stable_for_equal_scores() {
        let a = MemoryId::from_key("a");
        let b = MemoryId::from_key("b");
        let mut first = vec![Hit { id: a, score: 1.0 }, Hit { id: b, score: 1.0 }];
        let mut second = vec![Hit { id: b, score: 1.0 }, Hit { id: a, score: 1.0 }];
        sort_hits(&mut first);
        sort_hits(&mut second);
        assert_eq!(first, second, "equal scores must order deterministically");
    }

    #[test]
    fn empty_corpus_yields_no_centroids() {
        let centroids = train_centroids(&[], 1);
        assert!(centroids.is_empty());
        assert!(centroids.probe(&[1.0], 8).is_empty());
    }
}

#[cfg(test)]
mod centroid_encoding_tests {
    use super::*;

    fn table() -> Centroids {
        Centroids {
            vectors: vec![vec![1.0, -0.5, 0.25], vec![0.0, 2.5, -3.75]],
            sizes: vec![7, 11],
            dim: 3,
            radii: vec![0.5, 1.25],
        }
    }

    /// A generation written before radii existed must still open — and must report its radii
    /// as *unknown*, not as zero. rkyv's layout is positional, so a v1 buffer read as a v2
    /// struct is a misparse, not a missing field; the version tag is what makes it safe.
    #[test]
    fn a_v1_table_decodes_with_no_radii() {
        let v1 = CentroidsV1 {
            vectors: vec![vec![1.0, -0.5, 0.25], vec![0.0, 2.5, -3.75]],
            sizes: vec![7, 11],
            dim: 3,
        };
        let bytes = mlake_core::envelope::wrap(CENTROIDS_MAGIC, 1, &mlake_core::rkyv_write(&v1));
        let back = Centroids::from_bytes(&bytes).unwrap();
        assert_eq!(back.vectors, v1.vectors);
        assert_eq!(back.sizes, v1.sizes);
        assert!(back.radii.is_empty(), "a v1 table has no radii to trust");
        assert_eq!(back.radius(0), None);
        assert_eq!(back.max_similarity(&[1.0, 0.0, 0.0], 0), f32::INFINITY);
    }

    #[test]
    fn radii_survive_the_round_trip() {
        let back = Centroids::from_bytes(&table().to_bytes().unwrap()).unwrap();
        assert_eq!(back.radii, vec![0.5, 1.25]);
    }

    #[test]
    fn centroids_round_trip_exactly() {
        let c = table();
        let back = Centroids::from_bytes(&c.to_bytes().unwrap()).unwrap();
        // Raw f32 is bit-exact, unlike the decimal text it replaced.
        assert_eq!(back, c);
    }

    #[test]
    fn an_empty_table_round_trips() {
        let c = Centroids::default();
        let back = Centroids::from_bytes(&c.to_bytes().unwrap()).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn a_truncated_table_is_an_error_not_a_panic() {
        let bytes = table().to_bytes().unwrap();
        for cut in [1, 7, 9, bytes.len() - 1] {
            assert!(
                Centroids::from_bytes(&bytes[..cut]).is_err(),
                "a {cut}-byte prefix must be rejected, never indexed into"
            );
        }
    }
}
