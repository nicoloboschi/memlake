//! The external-memory ("streaming") fold: build a generation whose peak RAM is bounded by
//! per-stage buffer budgets instead of the corpus size, so a first build scales past what fits in
//! memory (the O(N)-RAM in-RAM [`crate::indexer::index`] caps a 36 GB box at ~5–6M).
//!
//! How it stays bounded — *nothing* held in RAM scales with the number of distinct ids or edges:
//!   * **Resolution** spills every source of truth for an id — the previous generation's item and
//!     each WAL op — as a small tagged [`Event`] keyed by that id into one [`ExternalSort`]. The
//!     sort groups an id's events together; a streaming merge resolves each id in seq order (only
//!     that id's handful of events resident) and emits the surviving live item to a per-type disk
//!     [`ItemSpill`]. The naive fold's O(distinct-ids) hashmap is gone.
//!   * **Per type** it trains centroids on a reservoir sample ([`train_centroids_k`] — never scans
//!     all N), then makes ONE pass over the spill: assign each item to a cluster and feed six
//!     [`ExternalSort`]s (cluster grouping carries full item bytes; pk / payload / entity / time
//!     carry SSTable fragments; radj carries causal edges keyed by target) plus the FTS builder.
//!     Every sort spills sorted runs and k-way-merges them, so SSTables, cluster files, and the
//!     reverse-adjacency are all written from bounded memory.
//!
//! Peak RAM is set by [`FoldBudget`] (per-stage `MEMLAKE_FOLD_*_MB`), not by N. The stages run
//! mostly sequentially (resolution, then one memory_type's build at a time), so the budgets do not
//! all sum at once.
//!
//! Scope: this is the **bulk build** path. It DOES derive semantic kNN links, but *home-cluster
//! only*: the fold holds one cluster at a time, so each item's links are found among its own
//! cluster's members rather than the in-RAM fold's wider `nprobe`-cluster neighbourhood (an item
//! whose nearest neighbour sits just across a cluster boundary may miss it — the bounded-RAM
//! tradeoff). Causal edges (client-provided, inline) are preserved. It skips the local-split
//! rebalance. The result is a correct, queryable generation.

use std::collections::{BTreeMap, BTreeSet};

use mlake_core::wal::{apply_deltas, deltas_from_rkyv_bytes, deltas_to_rkyv_bytes, Delta};
use mlake_core::{Op, Predicate, StoredMemory};
use mlake_fts::{TantivyFtsBuilder, Tokenizer};
use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag};
use mlake_ivf::{ClusterFile, VectorBlock, VectorCodec};
use mlake_wal::Namespace;

use crate::generation::{write_cluster_file, write_generation, ClusterTagSummary, TagSummary};
use crate::indexer::{IndexOptions, IndexOutcome};
use crate::spill::{ExternalSort, ItemSpill, Merge};
use crate::sstable::{encode_in_edge, ts_key};
use crate::Result;

/// Log a streaming-fold phase's duration when `MEMLAKE_TIMING` is set (matches the in-RAM fold).
fn phase_log(phase: &str, start: std::time::Instant) {
    if std::env::var("MEMLAKE_TIMING").is_ok() {
        eprintln!("[stream] {phase}: {:.2}s", start.elapsed().as_secs_f64());
    }
}

/// Vectors sampled for centroid training (mini-batch k-means).
const TRAIN_SAMPLE: usize = 50_000;

/// Per-stage RAM budget for the streaming fold, read from the environment. Each field caps ONE
/// stage's in-memory buffer before it spills to disk; the fold never holds a corpus-sized
/// collection, so peak RAM is roughly the largest concurrently-live subset of these plus one open
/// cluster / WAL entry and small bookkeeping — **not** a function of N. Because resolution runs to
/// completion before any per-type build, and the types build one at a time, the fields do not all
/// sum at once; a safe over-estimate of peak is `resolve` **or** (`cluster + payload + index +
/// radj + fts`), whichever is larger.
///
/// | env var                     | stage                                   | default (MB) |
/// |-----------------------------|-----------------------------------------|--------------|
/// | `MEMLAKE_FOLD_RESOLVE_MB`    | Phase-1 id resolution (spills events)    | 128          |
/// | `MEMLAKE_FOLD_CLUSTER_MB`    | cluster grouping (full item bytes)       | 256          |
/// | `MEMLAKE_FOLD_PAYLOAD_MB`    | payload store                            | 128          |
/// | `MEMLAKE_FOLD_INDEX_MB`      | pk + entity + time (split three ways)    | 96           |
/// | `MEMLAKE_FOLD_RADJ_MB`       | reverse-adjacency (causal edges)         | 64           |
/// | `MEMLAKE_FOLD_FTS_MB`        | tantivy writer arena                     | 128          |
#[derive(Clone, Copy, Debug)]
pub struct FoldBudget {
    pub resolve_mb: usize,
    pub cluster_mb: usize,
    pub payload_mb: usize,
    pub index_mb: usize,
    pub radj_mb: usize,
    pub fts_mb: usize,
}

impl Default for FoldBudget {
    fn default() -> Self {
        Self {
            resolve_mb: 128,
            cluster_mb: 256,
            payload_mb: 128,
            index_mb: 96,
            radj_mb: 64,
            fts_mb: 128,
        }
    }
}

impl FoldBudget {
    /// Read per-stage budgets from `MEMLAKE_FOLD_*_MB`, falling back to [`Default`] for any that is
    /// unset, empty, non-numeric, or zero.
    pub fn from_env() -> Self {
        let d = Self::default();
        fn mb(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(default)
        }
        Self {
            resolve_mb: mb("MEMLAKE_FOLD_RESOLVE_MB", d.resolve_mb),
            cluster_mb: mb("MEMLAKE_FOLD_CLUSTER_MB", d.cluster_mb),
            payload_mb: mb("MEMLAKE_FOLD_PAYLOAD_MB", d.payload_mb),
            index_mb: mb("MEMLAKE_FOLD_INDEX_MB", d.index_mb),
            radj_mb: mb("MEMLAKE_FOLD_RADJ_MB", d.radj_mb),
            fts_mb: mb("MEMLAKE_FOLD_FTS_MB", d.fts_mb),
        }
    }

    fn bytes(mb: usize) -> usize {
        mb * 1024 * 1024
    }
}

/// A deterministic xorshift RNG for the reservoir sample — reproducible for a given seed (G-6),
/// without pulling in a dependency here.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ---- Resolution events -----------------------------------------------------
//
// Wire form of one spilled event: `[tag:1][seq:8 LE][body]`. `body` is the item's rkyv bytes for
// PrevGen/Upsert, a patch's rkyv-encoded deltas for Patch, and empty for Tombstone.
const EV_PREVGEN: u8 = 0;
const EV_UPSERT: u8 = 1;
const EV_TOMBSTONE: u8 = 2;
const EV_PATCH: u8 = 3;

enum EventKind {
    /// A full item (previous-generation survivor or a WAL upsert) that becomes the id's base.
    Item(StoredMemory),
    /// A delete: the id has no live item as of this seq.
    Tombstone,
    /// Deltas applied to whatever base is live at this seq.
    Patch(Vec<Delta>),
}

struct Event {
    seq: u64,
    kind: EventKind,
}

fn encode_item_event(tag: u8, seq: u64, item: &StoredMemory) -> Vec<u8> {
    let body = item.to_rkyv_bytes();
    let mut v = Vec::with_capacity(9 + body.len());
    v.push(tag);
    v.extend_from_slice(&seq.to_le_bytes());
    v.extend_from_slice(&body);
    v
}

fn encode_marker_event(tag: u8, seq: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(9);
    v.push(tag);
    v.extend_from_slice(&seq.to_le_bytes());
    v
}

fn encode_patch_event(seq: u64, deltas: &[Delta]) -> Vec<u8> {
    let body = deltas_to_rkyv_bytes(deltas);
    let mut v = Vec::with_capacity(9 + body.len());
    v.push(EV_PATCH);
    v.extend_from_slice(&seq.to_le_bytes());
    v.extend_from_slice(&body);
    v
}

fn decode_event(val: &[u8]) -> Result<Event> {
    let decode_err =
        |what: &str| crate::Error::Core(mlake_core::Error::Decode(format!("event {what}")));
    if val.len() < 9 {
        return Err(decode_err("truncated"));
    }
    let tag = val[0];
    let seq = u64::from_le_bytes(val[1..9].try_into().unwrap());
    let body = &val[9..];
    let kind = match tag {
        EV_PREVGEN | EV_UPSERT => EventKind::Item(
            StoredMemory::from_payload_bytes(body).ok_or_else(|| decode_err("item"))?,
        ),
        EV_TOMBSTONE => EventKind::Tombstone,
        EV_PATCH => {
            EventKind::Patch(deltas_from_rkyv_bytes(body).ok_or_else(|| decode_err("patch"))?)
        }
        _ => return Err(decode_err("tag")),
    };
    Ok(Event { seq, kind })
}

/// Resolve one id's grouped events to its live item (last-write-wins over upsert/tombstone, with
/// patches applied to the base live at their seq) and, if it survives predicate deletes, emit it.
/// Only this id's events are ever resident, so RAM does not scale with the corpus.
fn resolve_and_emit(
    events: &mut Vec<Event>,
    predicate_deleted: &impl Fn(&StoredMemory) -> bool,
    resolver: &mut Resolver,
) -> Result<()> {
    // Order by seq; a stable sort keeps same-seq events (never expected for one id) in spill order.
    events.sort_by_key(|e| e.seq);
    let mut base: Option<StoredMemory> = None;
    for ev in events.drain(..) {
        match ev.kind {
            // A later upsert (or the sole prev-gen item) replaces everything, dropping the effect
            // of any earlier patch — last write wins.
            EventKind::Item(item) => base = Some(item),
            EventKind::Tombstone => base = None,
            EventKind::Patch(deltas) => {
                if let Some(item) = base.as_mut() {
                    apply_deltas(item, &deltas);
                }
            }
        }
    }
    if let Some(item) = base {
        if predicate_deleted(&item) {
            return Ok(());
        }
        // Keep the item's `semantic_out`: links are derived at write time and travel in the WAL as
        // intrinsic data, so the fold carries them forward (and feeds `radj` from them) rather than
        // re-deriving. The index stays a pure speed optimization.
        resolver.emit(item)?;
    }
    Ok(())
}

/// All WAL sequences in `(after, head]`, ascending — metadata only (LISTs, nothing decoded).
async fn wal_seqs(ns: &Namespace, after: u64, head: u64) -> Result<Vec<u64>> {
    let mut seqs = Vec::new();
    let mut start = after + 1;
    loop {
        let (objs, next) = ns.list_wal(start, 100_000).await?;
        for o in &objs {
            if o.seq <= head {
                seqs.push(o.seq);
            }
        }
        match next {
            Some(n) if n <= head => start = n,
            _ => break,
        }
    }
    Ok(seqs)
}

/// Resolves live items from prev-gen + tail and spills them by memory_type, reservoir-sampling
/// vectors per type for training — all in bounded RAM.
struct Resolver {
    spills: BTreeMap<u8, ItemSpill>,
    samples: BTreeMap<u8, Vec<Vec<f32>>>,
    seen: BTreeMap<u8, usize>,
    rng: Rng,
}

impl Resolver {
    fn new(seed: u64) -> Self {
        Self {
            spills: BTreeMap::new(),
            samples: BTreeMap::new(),
            seen: BTreeMap::new(),
            rng: Rng::new(seed),
        }
    }

    fn emit(&mut self, item: StoredMemory) -> Result<()> {
        let ft = item.memory_type;
        // Reservoir-sample the vector (skip empty vectors — text-only memories aren't clustered).
        if !item.vector.is_empty() {
            let cnt = self.seen.entry(ft).or_insert(0);
            let s = self.samples.entry(ft).or_default();
            if s.len() < TRAIN_SAMPLE {
                s.push(item.vector.clone());
            } else {
                let j = self.rng.below(*cnt + 1);
                if j < TRAIN_SAMPLE {
                    s[j] = item.vector.clone();
                }
            }
            *cnt += 1;
        }
        let spill = match self.spills.get_mut(&ft) {
            Some(s) => s,
            None => self.spills.entry(ft).or_insert(ItemSpill::new()?),
        };
        spill.push(&item)?;
        Ok(())
    }
}

/// Build a generation for `ns` with bounded memory, using per-stage budgets from the environment
/// (`MEMLAKE_FOLD_*_MB`). See the module docs for scope.
pub async fn index_streaming(
    ns: &Namespace,
    tokenizer: &Tokenizer,
    opts: IndexOptions,
) -> Result<IndexOutcome> {
    index_streaming_with_budget(ns, tokenizer, opts, FoldBudget::from_env()).await
}

/// [`index_streaming`] with an explicit budget — the entry point tests use to force tiny buffers
/// (so the external sorts actually spill and merge) without racing on process-wide env vars.
pub async fn index_streaming_with_budget(
    ns: &Namespace,
    tokenizer: &Tokenizer,
    opts: IndexOptions,
    budget: FoldBudget,
) -> Result<IndexOutcome> {
    let (manifest, etag) = ns.read_manifest().await?;
    let head = ns.wal_head().await?;
    let version = manifest.version + 1;
    // Phase 1: read the single previous segment. Phase 2+ compaction reads across input segments.
    let prev_seg = manifest.segments.first();

    // ---- Phase 1: resolve live items with bounded RAM ----
    //
    // Same last-write-wins / tombstone / patch / predicate-delete semantics as `fold_entries`, but
    // fully external: every source of truth for an id — the previous generation's item and each WAL
    // op — is spilled as a small tagged event keyed by that id. One external sort groups an id's
    // events; a streaming merge then resolves each id (only its own events resident) and spills the
    // survivor by type. Predicate deletes are global, evaluated at emit time (they need the item's
    // write_seq + metadata), so they alone stay in RAM — bounded by the count of TombstoneWhere ops
    // (a bulk operation), never by the corpus.
    let seqs = wal_seqs(ns, manifest.wal_index_cursor, head).await?;

    let te = std::time::Instant::now();
    let mut res_sort = ExternalSort::new(FoldBudget::bytes(budget.resolve_mb));
    let mut preds: Vec<(u64, Predicate)> = Vec::new();

    // Previous generation: one event per live item, streamed cluster-by-cluster (one resident).
    for ft in manifest.memory_types() {
        let Some(fti) = prev_seg.and_then(|s| s.index(ft)) else { continue };
        for path in &fti.files.clusters {
            if path.is_empty() {
                continue;
            }
            let bytes = ns.store.get_immutable(path, None).await?;
            let cf = ClusterFile::from_bytes(&bytes)?;
            for item in cf.items {
                res_sort.add(item.id.0, encode_item_event(EV_PREVGEN, item.write_seq, &item))?;
            }
        }
    }
    // WAL tail: one event per op. Read ONCE (the previous two-pass fold read the whole WAL twice).
    for &seq in &seqs {
        let entry = ns.read_wal_entry(seq).await?;
        for op in entry.ops {
            match op {
                Op::Upsert(m) => {
                    let mut item = m.into_stored();
                    item.write_seq = seq;
                    res_sort.add(item.id.0, encode_item_event(EV_UPSERT, seq, &item))?;
                }
                Op::Tombstone { id } => {
                    res_sort.add(id.0, encode_marker_event(EV_TOMBSTONE, seq))?
                }
                Op::Patch { id, deltas } => {
                    res_sort.add(id.0, encode_patch_event(seq, &deltas))?
                }
                Op::TombstoneWhere { predicate } => preds.push((seq, predicate)),
                Op::Guard { .. } => {}
            }
        }
    }
    phase_log("wal_events", te);

    let predicate_deleted = |m: &StoredMemory| -> bool {
        preds.iter().any(|(seq, p)| m.write_seq < *seq && p.matches(m))
    };

    // Merge the events (grouped by id) and resolve each id to its survivor.
    let tr = std::time::Instant::now();
    let mut resolver = Resolver::new(opts.seed);
    let mut merge = res_sort.finish()?;
    let mut cur_id: Option<[u8; 16]> = None;
    let mut events: Vec<Event> = Vec::new();
    while let Some((key, val)) = merge.next()? {
        if cur_id.is_some_and(|c| c != key) {
            resolve_and_emit(&mut events, &predicate_deleted, &mut resolver)?;
        }
        cur_id = Some(key);
        events.push(decode_event(&val)?);
    }
    if cur_id.is_some() {
        resolve_and_emit(&mut events, &predicate_deleted, &mut resolver)?;
    }
    phase_log("resolve", tr);

    // ---- Phase 2: build each memory_type's index from its spill ----
    let seg_id = mlake_core::MemoryId::new_v4().as_uuid().simple().to_string();
    let mut indexes: BTreeMap<u8, mlake_core::manifest::FactTypeIndex> = BTreeMap::new();
    let mut doc_count = 0usize;

    let Resolver { mut spills, mut samples, .. } = resolver;
    let fts = std::mem::take(&mut spills);
    for (ft, spill) in fts {
        let sample = samples.remove(&ft).unwrap_or_default();
        let n = spill.len();
        doc_count += n;
        let fti = build_type_streaming(
            ns, &seg_id, ft, spill, sample, n, tokenizer, opts.seed, opts.vector_codec,
            &opts.indexed_metadata_keys, budget,
        )
        .await?;
        indexes.insert(ft, fti);
    }

    // ---- Publish: one full segment (same CAS swap as the in-RAM fold) ----
    let segment = mlake_core::Segment {
        id: seg_id,
        level: 0,
        seq_lo: 0,
        seq_hi: head,
        doc_count: doc_count as u64,
        indexes,
        tombstones: String::new(),
    };
    let mut next = manifest.clone();
    next.version = version;
    next.wal_index_cursor = head;
    next.wal_head = head;
    next.prev_wal_index_cursor = manifest.wal_index_cursor;
    next.tokenizer_config_hash = tokenizer.config_hash();
    next.prev_segments = manifest.segments.clone();
    next.segments = vec![segment];

    let published = match etag {
        Some(etag) => ns.swap_manifest(&etag, &next).await.map(|_| true).or_else(|e| {
            if e.is_conflict() {
                Ok(false)
            } else {
                Err(e)
            }
        })?,
        None => false,
    };

    Ok(IndexOutcome { generation: version, doc_count, published })
}

#[allow(clippy::too_many_arguments)]
async fn build_type_streaming(
    ns: &Namespace,
    seg_id: &str,
    memory_type: u8,
    spill: ItemSpill,
    sample: Vec<Vec<f32>>,
    n: usize,
    tokenizer: &Tokenizer,
    seed: u64,
    codec: VectorCodec,
    indexed_metadata_keys: &[String],
    budget: FoldBudget,
) -> Result<mlake_core::manifest::FactTypeIndex> {
    let prefix = format!("{}/mt{memory_type}", mlake_core::manifest::segment_prefix(&ns.name, seg_id));
    let ttr = std::time::Instant::now();
    let k = mlake_ivf::centroid_count(n);
    // The dimension the vector blocks encode against; 0 for a type with no embeddings.
    let vector_dim = sample.first().map(|v| v.len()).unwrap_or(0);
    let mut centroids = mlake_ivf::train_centroids_k(&sample, k, seed);
    drop(sample);
    let kk = centroids.len().max(1);
    phase_log("train", ttr);
    let tas = std::time::Instant::now();

    // The cluster + payload sorts carry the large values (full item bytes / payloads); pk / entity
    // / time carry small SSTable fragments (split the index budget three ways); radj carries causal
    // edges keyed by target. Each is capped by its own [`FoldBudget`] stage and spills to disk.
    let per_index = (FoldBudget::bytes(budget.index_mb) / 3).max(4096);
    let mut cluster_sort = ExternalSort::new(FoldBudget::bytes(budget.cluster_mb));
    let mut payload_sort = ExternalSort::new(FoldBudget::bytes(budget.payload_mb));
    // Full precision for stage two of the search. Spilled like every other column so the
    // fold's RAM stays bounded by its budget rather than by the corpus.
    let mut rerank_sort = ExternalSort::new(FoldBudget::bytes(budget.payload_mb));
    let mut pk_sort = ExternalSort::new(per_index);
    let mut entity_sort = ExternalSort::new(per_index);
    let mut time_sort = ExternalSort::new(per_index);
    let mut radj_sort = ExternalSort::new(FoldBudget::bytes(budget.radj_mb));
    let mut fts = TantivyFtsBuilder::new(tokenizer.clone(), FoldBudget::bytes(budget.fts_mb))
        .map_err(|e| crate::Error::Fts(e.to_string()))?;
    let mut sizes = vec![0usize; kk];
    let mut cluster_tags: Vec<(BTreeSet<String>, bool)> = vec![(BTreeSet::new(), false); kk];
    // Per-cluster write-time span, accumulated in the same pass. Starts as an empty range so
    // a cluster whose members all lack an `updated_at` stays empty and prunes correctly.
    let mut cluster_updated: Vec<[i64; 2]> = vec![[i64::MAX, i64::MIN]; kk];
    // Value-counts for the declared metadata keys, tallied in the same assign pass so the
    // streaming fold builds them without a second walk (mirrors the in-RAM build_meta_counts).
    let wanted_meta: std::collections::HashSet<&str> =
        indexed_metadata_keys.iter().map(|s| s.as_str()).collect();
    let mut meta_counts: std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>> =
        std::collections::BTreeMap::new();
    // Edge totals, tallied in the same assign pass (mirrors the in-RAM fold), summed across
    // segments and corrected for the tail by `LinkStats`.
    let mut semantic_edge_count: usize = 0;
    let mut causal_edge_count: usize = 0;

    // Single assignment pass over the spill, in batches: the per-item centroid assignment and the
    // two rkyv serializations (full item for the cluster file, payload for the store) are the
    // CPU-heavy work and are independent, so a batch does them across cores (like the in-RAM fold's
    // parallel assign). Feeding the external sorts / FTS stays serial (shared state), but is cheap.
    use rayon::prelude::*;
    const ASSIGN_BATCH: usize = 8_192;
    let mut batch: Vec<StoredMemory> = Vec::with_capacity(ASSIGN_BATCH);
    let mut reader = spill.into_reader()?;
    loop {
        batch.clear();
        while batch.len() < ASSIGN_BATCH {
            match reader.next() {
                Some(item) => batch.push(item),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        // Parallel: nearest centroid + both serializations per item.
        let prepared: Vec<(usize, Vec<u8>, Vec<u8>, Vec<u8>)> = batch
            .par_iter()
            .map(|item| {
                let c = if centroids.is_empty() { 0 } else { centroids.assign(&item.vector) };
                let mut vec_bytes = Vec::with_capacity(item.vector.len() * 4);
                for x in &item.vector {
                    vec_bytes.extend_from_slice(&x.to_le_bytes());
                }
                (c, item.to_rkyv_bytes(), item.to_payload_bytes(), vec_bytes)
            })
            .collect();
        // Serial: feed the sorts / FTS / summaries.
        for (item, (c, full, payload, vec_bytes)) in batch.drain(..).zip(prepared) {
            sizes[c] += 1;
            cluster_sort.add(cluster_key(c), full)?;
            pk_sort.add(item.id.0, (c as u32).to_le_bytes().to_vec())?;
            payload_sort.add(item.id.0, payload)?;
            // A memory with no embedding contributes no rerank row: nothing to rescore.
            if !vec_bytes.is_empty() {
                rerank_sort.add(item.id.0, vec_bytes)?;
            }
            for e in &item.entity_ids {
                entity_sort.add(e.0, item.id.0.to_vec())?;
            }
            let t = &item.timestamps;
            if let Some(ts) = t.occurred_start.or(t.mentioned_at).or(t.occurred_end) {
                time_sort.add(ts_key(ts), item.id.0.to_vec())?;
            }
            fts.add(item.id, item.fts_text(), &item.tags).map_err(|e| crate::Error::Fts(e.to_string()))?;
            semantic_edge_count += item.semantic_out.len();
            causal_edge_count += item.causal_out.len();
            for edge in &item.causal_out {
                let ie = InEdge {
                    source: item.id,
                    kind: EdgeKind::Causal(LinkTypeTag::from(edge.link_type)),
                    weight: edge.weight.to_f32(),
                };
                radj_sort.add(edge.target.0, encode_in_edge(&ie))?;
            }
            if let Some(u) = item.timestamps.updated_at {
                let r = &mut cluster_updated[c];
                r[0] = r[0].min(u);
                r[1] = r[1].max(u);
            }
            let (tset, unt) = &mut cluster_tags[c];
            if item.tags.is_empty() {
                *unt = true;
            } else {
                for tag in &item.tags {
                    tset.insert(tag.clone());
                }
            }
            if !wanted_meta.is_empty() {
                for (k, v) in &item.metadata {
                    if wanted_meta.contains(k.as_str()) {
                        *meta_counts.entry(k.clone()).or_default().entry(v.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    centroids.sizes = sizes;
    phase_log("assign", tas);

    // Write cluster files from the cluster-grouped stream.
    let tcw = std::time::Instant::now();
    let mut cluster_paths: Vec<String> = vec![String::new(); kk];
    let mut vector_paths: Vec<String> = vec![String::new(); kk];
    let mut merge = cluster_sort.finish()?;
    let mut cur_c: Option<usize> = None;
    let mut cur_items: Vec<StoredMemory> = Vec::new();
    loop {
        let rec = merge.next()?;
        match rec {
            Some((key, val)) => {
                let c = cluster_from_key(&key);
                let item = StoredMemory::from_payload_bytes(&val)
                    .ok_or_else(|| crate::Error::Core(mlake_core::Error::Decode("spilled item".into())))?;
                match cur_c {
                    Some(cc) if cc == c => cur_items.push(item),
                    Some(cc) => {
                        feed_radj(&cur_items, &mut radj_sort)?;
                        let (cp, vp) = flush_cluster(ns, &prefix, cc, std::mem::take(&mut cur_items), codec, vector_dim).await?;
                        cluster_paths[cc] = cp;
                        vector_paths[cc] = vp;
                        cur_c = Some(c);
                        cur_items.push(item);
                    }
                    None => {
                        cur_c = Some(c);
                        cur_items.push(item);
                    }
                }
            }
            None => {
                if let Some(cc) = cur_c {
                    feed_radj(&cur_items, &mut radj_sort)?;
                    let (cp, vp) = flush_cluster(ns, &prefix, cc, std::mem::take(&mut cur_items), codec, vector_dim).await?;
                    cluster_paths[cc] = cp;
                    vector_paths[cc] = vp;
                }
                break;
            }
        }
    }
    // Empty clusters still need a (empty) file so a query can address them by index.
    for c in 0..kk {
        if cluster_paths[c].is_empty() {
            let (cp, vp) = flush_cluster(ns, &prefix, c, Vec::new(), codec, vector_dim).await?;
            cluster_paths[c] = cp;
            vector_paths[c] = vp;
        }
    }

    phase_log("cluster_write", tcw);

    // Build the SSTables from the sorted merges. radj shares the SSTable grouping: its values are
    // per-target edge encodings that concatenate under the target key — the same layout
    // `RadjTable::build` produces (only the within-target edge order differs, which no reader
    // depends on).
    let tsst = std::time::Instant::now();
    let pk = build_sstable_from_merge(pk_sort.finish()?)?;
    let payload = build_sstable_from_merge(payload_sort.finish()?)?;
    let rerank = build_sstable_from_merge(rerank_sort.finish()?)?;
    let entity = build_sstable_from_merge(entity_sort.finish()?)?;
    let time = build_sstable_from_merge(time_sort.finish()?)?;
    let radj = build_sstable_from_merge(radj_sort.finish()?)?;
    let fts_split = fts.finish().map_err(|e| crate::Error::Fts(e.to_string()))?.split_bytes().to_vec();
    let tag_summary: TagSummary = cluster_tags
        .into_iter()
        .zip(cluster_updated)
        .map(|((tags, has_untagged), updated_range)| ClusterTagSummary {
            tags: tags.into_iter().collect(),
            has_untagged,
            updated_range: Some(updated_range),
        })
        .collect();
    phase_log("sstable_finalize", tsst);

    let twg = std::time::Instant::now();
    let files = write_generation(
        &ns.store,
        &prefix,
        &centroids,
        cluster_paths,
        vector_paths,
        &fts_split,
        radj.into(),
        pk.into(),
        entity.into(),
        time.into(),
        payload.into(),
        rerank.into(),
        &tag_summary,
        n,
        semantic_edge_count,
        causal_edge_count,
        meta_counts,
    )
    .await?;
    phase_log("write_gen", twg);

    Ok(mlake_core::manifest::FactTypeIndex { train_count: n as u64, files })
}

/// Feed a cluster's items' inline `semantic_out` (derived at write time, carried in the WAL) into
/// the reverse-adjacency external sort — a reorganization, not a derivation. Causal reverse edges
/// are fed elsewhere in the assign pass.
fn feed_radj(items: &[StoredMemory], radj_sort: &mut ExternalSort) -> Result<()> {
    for item in items.iter() {
        for edge in &item.semantic_out {
            let ie =
                InEdge { source: item.id, kind: EdgeKind::Semantic, weight: edge.weight.to_f32() };
            radj_sort.add(edge.target.0, encode_in_edge(&ie))?;
        }
    }
    Ok(())
}

/// Write one cluster's pair: the payload file and its vector block, in the same member
/// order so index `j` names the same memory in both. Returns `(cluster_path, vector_path)`.
async fn flush_cluster(
    ns: &Namespace,
    prefix: &str,
    c: usize,
    mut items: Vec<StoredMemory>,
    codec: VectorCodec,
    dim: usize,
) -> Result<(String, String)> {
    let ids: Vec<mlake_core::MemoryId> = items.iter().map(|m| m.id).collect();
    // Absent embeddings (text-only memories) pad to zeros rather than failing the encode.
    let vectors: Vec<Vec<f32>> = items
        .iter()
        .map(|m| if m.vector.is_empty() { vec![0.0; dim] } else { m.vector.clone() })
        .collect();
    // Tags and write times travel with the codes so the probe can filter without the payload
    // half.
    let member_tags: Vec<Vec<String>> = items.iter().map(|m| m.tags.clone()).collect();
    let member_updated: Vec<i64> = items
        .iter()
        .map(|m| m.timestamps.updated_at.unwrap_or(mlake_ivf::UPDATED_UNKNOWN))
        .collect();
    // The embedding moves to the vector block; leaving a copy inline would give back every
    // byte this split exists to save.
    for m in items.iter_mut() {
        m.vector = Vec::new();
    }
    let cf = ClusterFile { centroid_id: c as u32, items };
    let cluster = write_cluster_file(&ns.store, prefix, c, &cf).await?;
    let block = VectorBlock::encode_with_columns(
        codec,
        dim,
        &ids,
        &vectors,
        Some(&member_tags),
        &member_updated,
    )?;
    let vec_path = crate::generation::write_vector_block(&ns.store, prefix, c, block.to_bytes()).await?;
    Ok((cluster, vec_path))
}

/// Group a sorted `(key, value)` merge into an SSTable: consecutive same-key values are
/// concatenated (the encoding every SSTable value uses), and each distinct key is one record.
fn build_sstable_from_merge(mut merge: Merge) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut b = crate::sstable::SsTableBuilder::new();
    let mut cur: Option<([u8; 16], Vec<u8>)> = None;
    while let Some((key, val)) = merge.next()? {
        match &mut cur {
            Some((k, v)) if *k == key => v.extend_from_slice(&val),
            Some((k, v)) => {
                b.add(*k, v);
                cur = Some((key, val));
            }
            None => cur = Some((key, val)),
        }
    }
    if let Some((k, v)) = cur {
        b.add(k, &v);
    }
    Ok(b.finish())
}

/// Order-preserving 16-byte key for a cluster id (big-endian in the low 8 bytes), so the
/// external sort groups a cluster's items together.
fn cluster_key(c: usize) -> [u8; 16] {
    let mut k = [0u8; 16];
    k[8..].copy_from_slice(&(c as u64).to_be_bytes());
    k
}
fn cluster_from_key(k: &[u8; 16]) -> usize {
    u64::from_be_bytes(k[8..].try_into().unwrap()) as usize
}
