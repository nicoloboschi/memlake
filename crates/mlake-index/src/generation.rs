//! Reading and writing a generation's files on object storage.
//!
//! A generation is the indexed, immutable snapshot of a namespace at a WAL cursor. Every
//! file here is write-once (INV-2): the indexer writes a whole `gen-{G}/` tree and then
//! CAS-swaps the manifest to point at it, so a reader either sees the entire old
//! generation or the entire new one, never a mixture.

use mlake_core::manifest::generation_prefix;
use mlake_core::{GenerationFiles, ItemId, StoredItem};
use mlake_fts::FtsIndex;
use mlake_graph::ReverseAdjacency;
use mlake_ivf::{Centroids, ClusterFile};
use mlake_store::{QueryMetrics, Store};
use serde::{Deserialize, Serialize};

use crate::Result;

/// The primary-key index: maps an item id to the cluster file it lives in, so a graph
/// candidate found by id can be materialized without scanning every cluster. Tombstoned
/// ids are absent, which is how dangling edges stay invisible (SPEC §7.7).
#[derive(Default, Serialize, Deserialize)]
pub struct PkIndex {
    /// id → cluster index. Sorted-map semantics via a Vec of pairs keeps the file small
    /// and deterministic.
    pub entries: Vec<(ItemId, u32)>,
}

impl PkIndex {
    pub fn lookup(&self, id: &ItemId) -> Option<u32> {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(id))
            .ok()
            .map(|i| self.entries[i].1)
    }
}

/// All the in-memory state of one generation. The query node materializes this (in full
/// for the POC; lazily per-probed-cluster for the roundtrip-budget path) and answers from
/// it plus the WAL tail.
pub struct Generation {
    pub generation: u64,
    pub centroids: Centroids,
    pub clusters: Vec<Vec<StoredItem>>,
    pub fts: FtsIndex,
    pub radj: ReverseAdjacency,
    pub pk: PkIndex,
}

/// File names within a generation prefix.
fn centroids_key(ns: &str, g: u64) -> String {
    format!("{}/centroids.json", generation_prefix(ns, g))
}
fn cluster_key(ns: &str, g: u64, i: usize) -> String {
    format!("{}/cluster-{i}.bin", generation_prefix(ns, g))
}
fn fts_key(ns: &str, g: u64) -> String {
    format!("{}/fts/split.bin", generation_prefix(ns, g))
}
fn radj_key(ns: &str, g: u64) -> String {
    format!("{}/radj.json", generation_prefix(ns, g))
}
fn pk_key(ns: &str, g: u64) -> String {
    format!("{}/pk.idx", generation_prefix(ns, g))
}
fn stats_key(ns: &str, g: u64) -> String {
    format!("{}/stats.json", generation_prefix(ns, g))
}

/// Generation-level statistics, for observability and compaction decisions.
#[derive(Serialize, Deserialize, Default)]
pub struct Stats {
    pub doc_count: usize,
    pub cluster_count: usize,
    pub edge_count: usize,
}

/// Write a generation's files to the store and return the manifest file map. Every file is
/// written unconditionally because its path is unique to this generation (INV-2).
pub async fn write_generation(
    store: &Store,
    namespace: &str,
    generation: u64,
    centroids: &Centroids,
    clusters: &[ClusterFile],
    fts: &FtsIndex,
    radj: &ReverseAdjacency,
    pk: &PkIndex,
) -> Result<GenerationFiles> {
    store
        .put(&centroids_key(namespace, generation), centroids.to_bytes()?)
        .await?;

    let mut cluster_paths = Vec::with_capacity(clusters.len());
    for (i, cluster) in clusters.iter().enumerate() {
        let key = cluster_key(namespace, generation, i);
        store.put(&key, cluster.to_bytes()?).await?;
        cluster_paths.push(key);
    }

    store
        .put(&fts_key(namespace, generation), serde_json::to_vec(fts)?)
        .await?;
    store
        .put(&radj_key(namespace, generation), radj.to_bytes()?)
        .await?;
    store
        .put(&pk_key(namespace, generation), serde_json::to_vec(pk)?)
        .await?;

    let stats = Stats {
        doc_count: pk.entries.len(),
        cluster_count: clusters.len(),
        edge_count: radj.edge_count(),
    };
    store
        .put(&stats_key(namespace, generation), serde_json::to_vec(&stats)?)
        .await?;

    Ok(GenerationFiles {
        pk: pk_key(namespace, generation),
        centroids: centroids_key(namespace, generation),
        clusters: cluster_paths,
        radj_csr: radj_key(namespace, generation),
        radj_idx: radj_key(namespace, generation),
        fts_split: fts_key(namespace, generation),
        stats: stats_key(namespace, generation),
    })
}

/// Load a whole generation from the store, counting each fetch against the budget.
///
/// This loads every cluster, which is the correct behaviour for the indexer's own reads
/// and for small namespaces. The query node's roundtrip-budget path instead loads only the
/// probed clusters; both go through the same file layout.
pub async fn read_generation(
    store: &Store,
    files: &GenerationFiles,
    generation: u64,
    metrics: Option<&QueryMetrics>,
) -> Result<Generation> {
    let centroids_bytes = store.get(&files.centroids, metrics.map(|m| (m, 2))).await?;
    let centroids = Centroids::from_bytes(&centroids_bytes.bytes)?;

    let mut clusters = Vec::with_capacity(files.clusters.len());
    for path in &files.clusters {
        let bytes = store.get(path, metrics.map(|m| (m, 3))).await?;
        clusters.push(ClusterFile::from_bytes(&bytes.bytes)?.items);
    }

    let fts_bytes = store.get(&files.fts_split, metrics.map(|m| (m, 2))).await?;
    let fts: FtsIndex = serde_json::from_slice(&fts_bytes.bytes)?;

    let radj_bytes = store.get(&files.radj_csr, metrics.map(|m| (m, 2))).await?;
    let radj = ReverseAdjacency::from_bytes(&radj_bytes.bytes)?;

    let pk_bytes = store.get(&files.pk, metrics.map(|m| (m, 2))).await?;
    let pk: PkIndex = serde_json::from_slice(&pk_bytes.bytes)?;

    Ok(Generation {
        generation,
        centroids,
        clusters,
        fts,
        radj,
        pk,
    })
}
