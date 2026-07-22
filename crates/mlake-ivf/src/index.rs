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
}

impl Centroids {
    pub fn len(&self) -> usize {
        self.vectors.len()
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
    pub fn push(&mut self, vector: Vec<f32>) -> usize {
        self.vectors.push(vector);
        self.sizes.push(0);
        self.vectors.len() - 1
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
        let (_version, payload) = mlake_core::envelope::unwrap(CENTROIDS_MAGIC, bytes).ok_or_else(
            || mlake_core::Error::Decode(format!("centroid table: bad header ({} bytes)", bytes.len())),
        )?;
        // An empty table round-trips as the default (empty rkyv archive).
        mlake_core::rkyv_read(payload).ok_or_else(|| {
            mlake_core::Error::Decode(format!("malformed centroid table ({} bytes)", payload.len()))
        })
    }
}

/// Object header magic + version for the centroid table (`centroids.bin`).
const CENTROIDS_MAGIC: &[u8; 4] = b"CENT";
const CENTROIDS_VERSION: u16 = 1;

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
    {
        use rayon::prelude::*;
        let assigns: Vec<usize> = vectors.par_iter().map(|v| nearest(&trained, v)).collect();
        for a in assigns {
            sizes[a] += 1;
        }
    }
    Centroids {
        dim: vectors[0].len(),
        vectors: trained,
        sizes,
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
    Centroids { dim: sample[0].len(), vectors: trained, sizes }
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
        }
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
