//! Reading and writing a generation's files on object storage.
//!
//! A generation is the indexed, immutable snapshot of a namespace at a WAL cursor. Every
//! file here is write-once (INV-2): the indexer writes a whole `gen-{G}/` tree and then
//! CAS-swaps the manifest to point at it, so a reader either sees the entire old
//! generation or the entire new one, never a mixture.

use mlake_core::manifest::generation_prefix;
use mlake_core::{GenerationFiles, ItemId, StoredItem};
use mlake_fts::{TantivyFts, Tokenizer};
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

/// The structural state of one generation loaded from storage: the vector index, the
/// items, the graph adjacency, and the pk index.
///
/// The FTS split is *not* loaded here — it is a self-contained tantivy artifact loaded on
/// demand via [`read_fts_split`], so callers that only need items (the incremental
/// indexer) do not pay to materialize it.
pub struct Generation {
    pub generation: u64,
    pub centroids: Centroids,
    pub clusters: Vec<Vec<StoredItem>>,
    pub radj: ReverseAdjacency,
    pub pk: PkIndex,
}

/// File names within a generation *attempt* prefix. The prefix is unique per index
/// attempt (`{ns}/gen-{G}-{nonce}`), so two nodes racing to build generation G write to
/// disjoint keys and can never overwrite each other's files — the winner's manifest points
/// at its own prefix, the loser's files are orphaned for GC (INV-2).
fn centroids_key(prefix: &str) -> String {
    format!("{prefix}/centroids.json")
}
fn cluster_key(prefix: &str, i: usize) -> String {
    format!("{prefix}/cluster-{i}.bin")
}
fn fts_key(prefix: &str) -> String {
    format!("{prefix}/fts/split.bin")
}
fn radj_key(prefix: &str) -> String {
    format!("{prefix}/radj.json")
}
fn pk_key(prefix: &str) -> String {
    format!("{prefix}/pk.idx")
}
fn stats_key(prefix: &str) -> String {
    format!("{prefix}/stats.json")
}

/// A unique per-attempt generation prefix. The nonce ensures two nodes building the same
/// generation number never collide on object keys.
pub fn attempt_prefix(namespace: &str, generation: u64, nonce: &str) -> String {
    format!("{}-{nonce}", generation_prefix(namespace, generation))
}

/// Generation-level statistics, for observability and compaction decisions.
#[derive(Serialize, Deserialize, Default)]
pub struct Stats {
    pub doc_count: usize,
    pub cluster_count: usize,
    pub edge_count: usize,
}

/// Write a generation's files under a unique attempt `prefix` and return the manifest file
/// map. Every file is genuinely write-once: because `prefix` is unique per attempt, no
/// other index run ever writes these keys, so the immutability invariant holds even when
/// two nodes build the same generation number concurrently (INV-2).
pub async fn write_generation(
    store: &Store,
    prefix: &str,
    centroids: &Centroids,
    clusters: &[ClusterFile],
    fts_split: &[u8],
    radj: &ReverseAdjacency,
    pk: &PkIndex,
) -> Result<GenerationFiles> {
    store
        .put(&centroids_key(prefix), centroids.to_bytes()?)
        .await?;

    let mut cluster_paths = Vec::with_capacity(clusters.len());
    for (i, cluster) in clusters.iter().enumerate() {
        let key = cluster_key(prefix, i);
        store.put(&key, cluster.to_bytes()?).await?;
        cluster_paths.push(key);
    }

    // The FTS file is the packed tantivy split — a self-contained index a query node
    // materializes into its NVMe/mmap tier (SPEC §6.1).
    store.put(&fts_key(prefix), fts_split.to_vec()).await?;
    store.put(&radj_key(prefix), radj.to_bytes()?).await?;
    store.put(&pk_key(prefix), serde_json::to_vec(pk)?).await?;

    let stats = Stats {
        doc_count: pk.entries.len(),
        cluster_count: clusters.len(),
        edge_count: radj.edge_count(),
    };
    store
        .put(&stats_key(prefix), serde_json::to_vec(&stats)?)
        .await?;

    Ok(GenerationFiles {
        pk: pk_key(prefix),
        centroids: centroids_key(prefix),
        clusters: cluster_paths,
        radj_csr: radj_key(prefix),
        radj_idx: radj_key(prefix),
        fts_split: fts_key(prefix),
        stats: stats_key(prefix),
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
    // RT2: the small metadata files — centroids, reverse adjacency, pk index — fetched
    // concurrently, so they cost one roundtrip of latency (INV-7). The FTS split is not
    // fetched here; it is loaded on demand via `read_fts_split` only when the FTS arm is
    // needed, so the incremental indexer (which only wants items) does not pay for it.
    let (centroids_bytes, radj_bytes, pk_bytes) = futures::try_join!(
        store.get(&files.centroids, metrics.map(|m| (m, 2))),
        store.get(&files.radj_csr, metrics.map(|m| (m, 2))),
        store.get(&files.pk, metrics.map(|m| (m, 2))),
    )?;
    let centroids = Centroids::from_bytes(&centroids_bytes.bytes)?;
    let radj = ReverseAdjacency::from_bytes(&radj_bytes.bytes)?;
    let pk: PkIndex = serde_json::from_slice(&pk_bytes.bytes)?;

    // RT3: the cluster files, all fetched concurrently — the spec's "parallel ranged GETs"
    // step, which is one roundtrip regardless of how many clusters are selected.
    let cluster_futures = files
        .clusters
        .iter()
        .map(|path| store.get(path, metrics.map(|m| (m, 3))));
    let cluster_objects = futures::future::try_join_all(cluster_futures).await?;
    let mut clusters = Vec::with_capacity(cluster_objects.len());
    for obj in &cluster_objects {
        clusters.push(ClusterFile::from_bytes(&obj.bytes)?.items);
    }

    Ok(Generation {
        generation,
        centroids,
        clusters,
        radj,
        pk,
    })
}

/// Load the generation's tantivy FTS split from storage and materialize it into the local
/// NVMe/mmap tier, ready to serve BM25 queries. One ranged GET (SPEC §6.1).
pub async fn read_fts_split(
    store: &Store,
    files: &GenerationFiles,
    tokenizer: Tokenizer,
    metrics: Option<&QueryMetrics>,
) -> Result<TantivyFts> {
    let bytes = store.get(&files.fts_split, metrics.map(|m| (m, 3))).await?;
    TantivyFts::from_split(&bytes.bytes, tokenizer)
        .map_err(|e| crate::Error::Core(mlake_core::Error::Decode(e.to_string())))
}
