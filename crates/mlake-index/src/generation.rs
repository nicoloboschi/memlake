//! Reading and writing a generation's files on object storage.
//!
//! A generation is the indexed, immutable snapshot of a namespace at a WAL cursor. Every
//! file here is write-once (INV-2): the indexer writes a whole `gen-{G}/` tree and then
//! CAS-swaps the manifest to point at it, so a reader either sees the entire old
//! generation or the entire new one, never a mixture.

use mlake_core::manifest::generation_prefix;
use mlake_core::{GenerationFiles, StoredMemory};
use mlake_fts::{TantivyFts, Tokenizer};
use mlake_ivf::{Centroids, ClusterFile};
use mlake_store::{QueryMetrics, Store};
use serde::{Deserialize, Serialize};

use crate::Result;

/// The items of one generation, loaded for the incremental indexer to fold the next
/// generation from. Only what the indexer needs: the centroids and the cluster items.
///
/// The FTS split, pk, and radj are *not* loaded here — they are rebuilt by the indexer and
/// range-read by the query node, so materializing them for a fold would be pure waste.
pub struct Generation {
    pub generation: u64,
    pub centroids: Centroids,
    pub clusters: Vec<Vec<StoredMemory>>,
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
fn radj_idx_key(prefix: &str) -> String {
    format!("{prefix}/radj.idx")
}
fn radj_csr_key(prefix: &str) -> String {
    format!("{prefix}/radj.csr")
}
fn pk_idx_key(prefix: &str) -> String {
    format!("{prefix}/pk.idx")
}
fn pk_data_key(prefix: &str) -> String {
    format!("{prefix}/pk.data")
}
fn stats_key(prefix: &str) -> String {
    format!("{prefix}/stats.json")
}
fn tag_summary_key(prefix: &str) -> String {
    format!("{prefix}/tags.json")
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

/// One cluster's tag summary, for pruning it before fetch (SCALE.md Phase 4b): the union of
/// all its memories' tags, and whether any memory is untagged.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct ClusterTagSummary {
    /// Distinct tags present across the cluster's memories, sorted.
    pub tags: Vec<String>,
    pub has_untagged: bool,
}

/// The per-cluster tag summaries for a generation, indexed by cluster id.
pub type TagSummary = Vec<ClusterTagSummary>;

/// Write a generation's files under a unique attempt `prefix` and return the manifest file
/// map. Every file is genuinely write-once: because `prefix` is unique per attempt, no
/// other index run ever writes these keys, so the immutability invariant holds even when
/// two nodes build the same generation number concurrently (INV-2).
/// Write one cluster file and return its object path. Used directly by the incremental
/// indexer, which writes only the *dirty* clusters and references unchanged ones by their
/// existing path (copy-forward-by-reference, SCALE.md Phase 3).
pub async fn write_cluster_file(
    store: &Store,
    prefix: &str,
    index: usize,
    cluster: &ClusterFile,
) -> Result<String> {
    let key = cluster_key(prefix, index);
    store.put(&key, cluster.to_bytes()?).await?;
    Ok(key)
}

/// Write a generation's metadata files given the (already written) `cluster_paths` — some
/// freshly written this fold, some copied forward from a previous generation. Every file
/// is write-once under the unique attempt `prefix` (INV-2).
#[allow(clippy::too_many_arguments)]
pub async fn write_generation(
    store: &Store,
    prefix: &str,
    centroids: &Centroids,
    cluster_paths: Vec<String>,
    fts_split: &[u8],
    radj_tables: SsTablePair,
    pk_tables: SsTablePair,
    tag_summary: &TagSummary,
    doc_count: usize,
) -> Result<GenerationFiles> {
    // All metadata objects are independent, immutable, and unique to this prefix, so write
    // them concurrently rather than one sequential PUT at a time.
    let stats = Stats {
        doc_count,
        cluster_count: cluster_paths.len(),
        edge_count: 0,
    };
    let centroids_bytes = centroids.to_bytes()?;
    let tag_bytes = serde_json::to_vec(tag_summary)?;
    let stats_bytes = serde_json::to_vec(&stats)?;
    let (kc, kt, kf, kri, krc, kpi, kpd, ks) = (
        centroids_key(prefix),
        tag_summary_key(prefix),
        fts_key(prefix),
        radj_idx_key(prefix),
        radj_csr_key(prefix),
        pk_idx_key(prefix),
        pk_data_key(prefix),
        stats_key(prefix),
    );
    // All metadata objects are independent, immutable, and unique to this prefix, so write
    // them concurrently rather than one sequential PUT at a time.
    futures::try_join!(
        store.put(&kc, centroids_bytes),
        store.put(&kt, tag_bytes),
        store.put(&kf, fts_split.to_vec()),
        store.put(&kri, radj_tables.idx),
        store.put(&krc, radj_tables.data),
        store.put(&kpi, pk_tables.idx),
        store.put(&kpd, pk_tables.data),
        store.put(&ks, stats_bytes),
    )?;

    Ok(GenerationFiles {
        pk: pk_idx_key(prefix),
        pk_data: pk_data_key(prefix),
        centroids: centroids_key(prefix),
        clusters: cluster_paths,
        radj_csr: radj_csr_key(prefix),
        radj_idx: radj_idx_key(prefix),
        fts_split: fts_key(prefix),
        stats: stats_key(prefix),
        tag_summary: tag_summary_key(prefix),
    })
}

/// The two byte streams of an SSTable: the small sparse index and the block data.
pub struct SsTablePair {
    pub idx: Vec<u8>,
    pub data: Vec<u8>,
}

impl From<(Vec<u8>, Vec<u8>)> for SsTablePair {
    fn from((idx, data): (Vec<u8>, Vec<u8>)) -> Self {
        Self { idx, data }
    }
}

/// Load a generation's centroids and cluster items — everything the incremental indexer
/// needs to fold the next generation. pk/radj/fts are deliberately not loaded (the indexer
/// rebuilds them), so a fold never whole-reads the range-readable secondary indexes.
pub async fn read_generation(
    store: &Store,
    files: &GenerationFiles,
    generation: u64,
    metrics: Option<&QueryMetrics>,
) -> Result<Generation> {
    use futures::stream::{StreamExt, TryStreamExt};
    const FETCH_CONCURRENCY: usize = 16;

    let centroids_bytes = store.get(&files.centroids, metrics.map(|m| (m, 2))).await?;
    let centroids = Centroids::from_bytes(&centroids_bytes.bytes)?;

    // Cluster files fetched with bounded concurrency: at 1M there are ~1000 multi-MB
    // clusters, and firing them all at once would push GBs of streamed bodies through the
    // HTTP client simultaneously (enough to truncate responses under load).
    let cluster_objects: Vec<_> = futures::stream::iter(files.clusters.iter())
        .map(|path| store.get(path, metrics.map(|m| (m, 3))))
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await?;
    let mut clusters = Vec::with_capacity(cluster_objects.len());
    for obj in &cluster_objects {
        clusters.push(ClusterFile::from_bytes(&obj.bytes)?.items);
    }

    Ok(Generation {
        generation,
        centroids,
        clusters,
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
