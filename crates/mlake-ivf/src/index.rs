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

use mlake_core::{cosine, ItemId, StoredItem};
use rkyv::{Archive, Deserialize, Serialize};

use crate::kmeans::{self, nearest};

/// A cluster file: the items assigned to one centroid.
///
/// Sized to 2–8 MB (SPEC §5.1) so that fetching one is a single coalesced ranged GET.
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug, Default)]
#[archive(check_bytes)]
pub struct ClusterFile {
    pub centroid_id: u32,
    pub items: Vec<StoredItem>,
}

impl ClusterFile {
    pub fn to_bytes(&self) -> Result<Vec<u8>, mlake_core::Error> {
        rkyv::to_bytes::<_, 65536>(self)
            .map(|b| b.into_vec())
            .map_err(|e| mlake_core::Error::Encode(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, mlake_core::Error> {
        // Same alignment concern as the WAL: bytes off the network are not aligned.
        if bytes.as_ptr() as usize % 8 == 0 {
            Self::from_aligned(bytes)
        } else {
            let mut aligned = rkyv::AlignedVec::with_capacity(bytes.len());
            aligned.extend_from_slice(bytes);
            Self::from_aligned(&aligned)
        }
    }

    fn from_aligned(bytes: &[u8]) -> Result<Self, mlake_core::Error> {
        let archived = rkyv::check_archived_root::<ClusterFile>(bytes)
            .map_err(|e| mlake_core::Error::Decode(e.to_string()))?;
        Deserialize::<ClusterFile, _>::deserialize(archived, &mut rkyv::Infallible)
            .map_err(|e| mlake_core::Error::Decode(format!("{e:?}")))
    }
}

/// The centroid table, held in memory on every query node. Small — `sqrt(N)` vectors —
/// and hot, so it lives in the in-memory ARC tier rather than being re-fetched.
#[derive(Clone, PartialEq, Debug, Default, serde::Serialize, serde::Deserialize)]
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

    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

/// A scored search hit.
#[derive(Clone, PartialEq, Debug)]
pub struct Hit {
    pub id: ItemId,
    pub score: f32,
}

/// Assign items to clusters and emit the cluster files.
///
/// Deterministic given the same items and centroids: item order within a cluster follows
/// input order, so a replay reproduces byte-identical files (G-6).
pub fn build_clusters(items: Vec<StoredItem>, centroids: &Centroids) -> Vec<ClusterFile> {
    let mut buckets: Vec<Vec<StoredItem>> = vec![Vec::new(); centroids.len().max(1)];
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

/// Train centroids over a corpus.
pub fn train_centroids(vectors: &[Vec<f32>], seed: u64) -> Centroids {
    if vectors.is_empty() {
        return Centroids::default();
    }
    let k = kmeans::centroid_count(vectors.len());
    let trained = kmeans::train(vectors, k, 25, seed);
    let mut sizes = vec![0usize; trained.len()];
    for v in vectors {
        sizes[nearest(&trained, v)] += 1;
    }
    Centroids {
        dim: vectors[0].len(),
        vectors: trained,
        sizes,
    }
}

/// Exact search over a set of items — the re-rank step, and the ground truth the recall
/// gate measures against.
pub fn exact_search(items: &[StoredItem], query: &[f32], k: usize) -> Vec<Hit> {
    let mut hits: Vec<Hit> = items
        .iter()
        .map(|item| Hit {
            id: item.id,
            score: cosine(query, &item.vector),
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
    let mut best: HashMap<ItemId, f32> = HashMap::new();
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
    use mlake_core::item::Timestamps;

    fn stored(key: &str, vector: Vec<f32>) -> StoredItem {
        StoredItem {
            id: ItemId::from_key(key),
            vector,
            text: key.to_string(),
            fact_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![],
            semantic_out: vec![],
            causal_out: vec![],
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
        assert_eq!(hits[0].id, ItemId::from_key("near"));
        assert_eq!(hits[1].id, ItemId::from_key("mid"));
        assert_eq!(hits[2].id, ItemId::from_key("far"));
    }

    #[test]
    fn merge_keeps_the_best_score_per_id() {
        // The same item can surface from both a cluster and the WAL tail; it must appear
        // once, at its best score.
        let id = ItemId::from_key("dup");
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
        let a = ItemId::from_key("a");
        let b = ItemId::from_key("b");
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
