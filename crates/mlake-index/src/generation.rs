//! Reading and writing a generation's files on object storage.
//!
//! A generation is the indexed, immutable snapshot of a namespace at a WAL cursor. Every
//! file here is write-once (INV-2): the indexer writes a whole `gen-{G}/` tree and then
//! CAS-swaps the manifest to point at it, so a reader either sees the entire old
//! generation or the entire new one, never a mixture.

use mlake_core::manifest::segment_prefix;
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
    format!("{prefix}/centroids.bin")
}
fn cluster_key(prefix: &str, i: usize) -> String {
    format!("{prefix}/cluster-{i}.bin")
}
fn vector_key(prefix: &str, i: usize) -> String {
    format!("{prefix}/cluster-{i}.vec")
}
fn fts_key(prefix: &str) -> String {
    format!("{prefix}/fts/split.bin")
}
fn radj_idx_key(prefix: &str) -> String {
    format!("{prefix}/radj.idx")
}
fn radj_data_key(prefix: &str) -> String {
    format!("{prefix}/radj.data")
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
fn entity_idx_key(prefix: &str) -> String {
    format!("{prefix}/entity.idx")
}
fn entity_data_key(prefix: &str) -> String {
    format!("{prefix}/entity.data")
}
fn time_idx_key(prefix: &str) -> String {
    format!("{prefix}/time.idx")
}
fn time_data_key(prefix: &str) -> String {
    format!("{prefix}/time.data")
}
fn payload_idx_key(prefix: &str) -> String {
    format!("{prefix}/payload.idx")
}
fn payload_data_key(prefix: &str) -> String {
    format!("{prefix}/payload.data")
}
fn rerank_idx_key(prefix: &str) -> String {
    format!("{prefix}/rerank.idx")
}
fn rerank_data_key(prefix: &str) -> String {
    format!("{prefix}/rerank.data")
}

/// A unique per-attempt segment prefix. The `seg_id` nonce ensures two nodes building the same
/// logical segment never collide on object keys.
pub fn attempt_prefix(namespace: &str, seg_id: &str) -> String {
    segment_prefix(namespace, seg_id)
}

fn tombstones_key(seg_prefix: &str) -> String {
    format!("{seg_prefix}/tombstones.bin")
}

/// Write a segment's delete overlay, returning its path (empty string if there is nothing to
/// record, so a clean full-rebuild segment writes no object).
pub async fn write_tombstones(
    store: &Store,
    seg_prefix: &str,
    t: &mlake_core::SegmentTombstones,
) -> Result<String> {
    if t.superseded.is_empty() && t.predicates.is_empty() {
        return Ok(String::new());
    }
    let key = tombstones_key(seg_prefix);
    store.put(&key, mlake_core::rkyv_write(t)).await?;
    Ok(key)
}

/// Read a segment's delete overlay; an empty path (a clean segment) yields the default (no deletes).
pub async fn read_tombstones(
    store: &Store,
    path: &str,
    metrics: Option<&QueryMetrics>,
) -> Result<mlake_core::SegmentTombstones> {
    if path.is_empty() {
        return Ok(mlake_core::SegmentTombstones::default());
    }
    let bytes = store.get_immutable(path, metrics.map(|m| (m, 2))).await?;
    Ok(mlake_core::rkyv_read(&bytes).unwrap_or_default())
}

/// Generation-level statistics, for observability and compaction decisions.
#[derive(Serialize, Deserialize, Default)]
pub struct Stats {
    pub doc_count: usize,
    pub cluster_count: usize,
    pub edge_count: usize,
}

/// One cluster's summary, for pruning it before fetch (SCALE.md Phase 4b): the union of all
/// its memories' tags, whether any memory is untagged, and the span of their write times.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct ClusterTagSummary {
    /// Distinct tags present across the cluster's memories, sorted.
    pub tags: Vec<String>,
    pub has_untagged: bool,
    /// The `[min, max]` of the cluster's members' `updated_at` (epoch ms) — the time twin of
    /// the tag union, letting an `updated_at` window prune a cluster before it is fetched.
    ///
    /// `None` in a summary written before the field existed, which readers must treat as
    /// "unknown, admit the cluster". A cluster whose members *all* lack a write time writes
    /// an empty range (`min > max`) instead, which admits nothing bounded. The two are
    /// different, and conflating them would silently drop clusters across an upgrade.
    #[serde(default)]
    pub updated_range: Option<[i64; 2]>,
}

impl ClusterTagSummary {
    /// Whether any member of the cluster could fall inside the window. `false` means the
    /// cluster can be dropped from the probe set without being read.
    ///
    /// Necessary, not sufficient: the range says nothing about which points inside it are
    /// occupied, so a `true` still leaves the per-member check to decide. But a `false` is
    /// exact — no member of the cluster can pass — which is what makes it safe to prune on.
    pub fn admits_window(&self, from: Option<i64>, to: Option<i64>) -> bool {
        if from.is_none() && to.is_none() {
            return true;
        }
        let Some([lo, hi]) = self.updated_range else {
            return true;
        };
        if lo > hi {
            return false;
        }
        from.is_none_or(|f| hi > f) && to.is_none_or(|t| lo < t)
    }

    /// The summary of a cluster's write times, from its members' `updated_at`.
    pub fn range_of<'a>(updated: impl Iterator<Item = &'a Option<i64>>) -> Option<[i64; 2]> {
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for u in updated.flatten() {
            lo = lo.min(*u);
            hi = hi.max(*u);
        }
        Some([lo, hi])
    }
}

/// The per-cluster tag summaries for a generation, indexed by cluster id.
pub type TagSummary = Vec<ClusterTagSummary>;

/// Write a generation's files under a unique attempt `prefix` and return the manifest file
/// map. Every file is genuinely write-once: because `prefix` is unique per attempt, no
/// other index run ever writes these keys, so the immutability invariant holds even when
/// two nodes build the same generation number concurrently (INV-2).
/// Write one cluster file and return its object path.
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

/// Write one cluster's vector block and return its object path.
///
/// Paired with [`write_cluster_file`] by index: `cluster-{i}.vec` holds the embeddings for
/// exactly the members of `cluster-{i}.bin`, in the same order, so the two are joined
/// positionally with no id lookup. Split apart because the probe scores every member but
/// reads the payload of almost none.
pub async fn write_vector_block(
    store: &Store,
    prefix: &str,
    index: usize,
    bytes: Vec<u8>,
) -> Result<String> {
    let key = vector_key(prefix, index);
    store.put(&key, bytes).await?;
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
    vector_paths: Vec<String>,
    fts_split: &[u8],
    radj_tables: SsTablePair,
    pk_tables: SsTablePair,
    entity_tables: SsTablePair,
    time_tables: SsTablePair,
    payload_tables: SsTablePair,
    rerank_tables: SsTablePair,
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
    let (kc, kt, kf, kri, krc, kpi, kpd, ks, kei, ked) = (
        centroids_key(prefix),
        tag_summary_key(prefix),
        fts_key(prefix),
        radj_idx_key(prefix),
        radj_data_key(prefix),
        pk_idx_key(prefix),
        pk_data_key(prefix),
        stats_key(prefix),
        entity_idx_key(prefix),
        entity_data_key(prefix),
    );
    let (kti, ktd) = (time_idx_key(prefix), time_data_key(prefix));
    let (kpli, kpld) = (payload_idx_key(prefix), payload_data_key(prefix));
    let (kri2, krd2) = (rerank_idx_key(prefix), rerank_data_key(prefix));
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
        store.put(&kei, entity_tables.idx),
        store.put(&ked, entity_tables.data),
        store.put(&kti, time_tables.idx),
        store.put(&ktd, time_tables.data),
        store.put(&kpli, payload_tables.idx),
        store.put(&kpld, payload_tables.data),
        store.put(&kri2, rerank_tables.idx),
        store.put(&krd2, rerank_tables.data),
    )?;

    Ok(GenerationFiles {
        pk: pk_idx_key(prefix),
        pk_data: pk_data_key(prefix),
        centroids: centroids_key(prefix),
        clusters: cluster_paths,
        vectors: vector_paths,
        radj_data: radj_data_key(prefix),
        radj_idx: radj_idx_key(prefix),
        fts_split: fts_key(prefix),
        stats: stats_key(prefix),
        tag_summary: tag_summary_key(prefix),
        entity_idx: entity_idx_key(prefix),
        entity_data: entity_data_key(prefix),
        time_idx: time_idx_key(prefix),
        time_data: time_data_key(prefix),
        payload_idx: payload_idx_key(prefix),
        payload_data: payload_data_key(prefix),
        rerank_idx: rerank_idx_key(prefix),
        rerank_data: rerank_data_key(prefix),
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
    //
    // Iterate owned paths and fetch inside a `move` async block, rather than
    // `.map(|path| store.get(path, ..))` over borrowed `&String`. The borrowed form builds a
    // higher-ranked `Fn(&String) -> Future` closure that rustc fails to prove `Send + 'static`
    // when this fold runs inline under the server's Send-bounded gRPC trait — a `move` block
    // over an owned `String` gives a concrete future and sidesteps the inference limitation.
    let ctx = metrics.map(|m| (m, 3));
    let cluster_objects: Vec<_> = futures::stream::iter(files.clusters.clone())
        .map(|path| async move { store.get(&path, ctx).await })
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await?;
    // The embeddings live in the parallel `.vec` blocks, so the fold has to re-join them:
    // a generation read that returned vector-less items would silently re-cluster the whole
    // corpus against empty embeddings.
    let vector_objects: Vec<_> = futures::stream::iter(files.vectors.clone())
        .map(|path| async move { store.get(&path, ctx).await })
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await?;
    let mut clusters = Vec::with_capacity(cluster_objects.len());
    for (i, obj) in cluster_objects.iter().enumerate() {
        let mut items = ClusterFile::from_bytes(&obj.bytes)?.items;
        let Some(vobj) = vector_objects.get(i) else {
            return Err(crate::Error::Core(mlake_core::Error::Decode(format!(
                "cluster {i} has no vector block: the generation's .bin and .vec lists disagree"
            ))));
        };
        let block = mlake_ivf::VectorBlock::from_bytes(&vobj.bytes)?;
        if block.len() != items.len() {
            return Err(crate::Error::Core(mlake_core::Error::Decode(format!(
                "cluster {i} holds {} members but its vector block holds {}",
                items.len(),
                block.len()
            ))));
        }
        for (j, item) in items.iter_mut().enumerate() {
            item.vector = block.decode(j);
        }
        clusters.push(items);
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
