//! The indexer: fold a WAL slice into a new generation (SPEC §5).
//!
//! Idempotent and coordination-free: any node may run it, and two nodes racing from the
//! same input produce *equivalent* generations, so the CAS-swap that publishes the result
//! is safe to lose (INV-6). A crash mid-run leaves only unreferenced files, which GC later
//! reclaims.
//!
//! Note on determinism (G-6): the vector, pk and radj files are byte-identical across
//! replays. The tantivy FTS split is not — tantivy stamps each segment with a random id —
//! but its *retrieval results* are identical, so query behaviour is still reproducible.

use mlake_core::memory::{SemanticEdge, Weight, MAX_SEMANTIC_OUT, SEMANTIC_LINK_THRESHOLD};
use mlake_core::{MemoryId, StoredMemory};
use mlake_fts::Tokenizer;
use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag};
use mlake_ivf::{train_centroids, ClusterFile, VectorBlock, VectorCodec};
use mlake_wal::{Namespace, WalTail};

use crate::generation::write_generation;
use crate::Result;

/// Options controlling a generation build. Note there is no `derive_links` knob: semantic links are
/// derived by the server at write time and carried in the WAL, so the fold never derives — it only
/// reorganizes WAL data into fast structures. The index is a pure optimization, not a correctness or
/// quality step.
#[derive(Clone)]
pub struct IndexOptions {
    /// Deterministic seed for centroid training (G-6).
    pub seed: u64,
    /// How embeddings are encoded in the vector block. The embedding is ~84% of a stored
    /// memory, so this is the single biggest lever on both stored bytes and bytes scanned
    /// per query — and the one place where storage is traded directly against recall.
    pub vector_codec: VectorCodec,
    /// Metadata keys to tally value-counts for (`Manifest::indexed_metadata_keys`). The fold
    /// reads these off the manifest and injects them here, so every build path picks them up
    /// through the options it already carries. Empty for a namespace that declared none.
    pub indexed_metadata_keys: Vec<String>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            vector_codec: VectorCodec::Binary,
            indexed_metadata_keys: Vec::new(),
        }
    }
}

/// Result of an index run.
pub struct IndexOutcome {
    pub generation: u64,
    pub doc_count: usize,
    /// True if the manifest swap succeeded. False means another node published first — not
    /// an error, since its generation is equivalent to ours (INV-6).
    pub published: bool,
}

/// Above this many estimated live documents, the in-RAM fold's O(N) RAM gets risky (it caps a
/// 36 GB box at ~5–6M), so [`fold`] switches to the bounded external-memory (streaming) fold.
pub const DEFAULT_STREAMING_THRESHOLD_DOCS: usize = 4_000_000;

/// Rough per-document byte size (upsert incl. embedding), used only to turn cheap WAL-tail LIST
/// sizes into a document-count estimate for the fold selector. Over-estimating docs errs toward
/// the safe (bounded-RAM) streaming fold, which is the conservative direction.
const NOMINAL_DOC_BYTES: u64 = 2048;

/// The fold entry point: **auto-selects** which fold to run. Below `streaming_threshold_docs` the
/// in-RAM [`index`] fold runs — faster, and it derives semantic links in-fold.
/// At or above it, the bounded-RAM [`crate::streaming::index_streaming_with_budget`] runs so a
/// corpus too big for memory still folds (at the cost of a full rebuild and no in-fold links).
///
/// `budget` is only consulted on the streaming path. Set `streaming_threshold_docs` to `0` to
/// force streaming (e.g. to benchmark it at small scale) or `usize::MAX` to force in-RAM.
pub async fn fold(
    ns: &Namespace,
    tokenizer: &Tokenizer,
    opts: IndexOptions,
    budget: crate::streaming::FoldBudget,
    streaming_threshold_docs: usize,
) -> Result<IndexOutcome> {
    let (manifest, _etag) = ns.read_manifest().await?;
    // A full rebuild happens on the first build (no segments) and at the hard count cap
    // (`COMPACT_FANOUT`): `index`/`index_streaming` merge ALL segments + the tail into one fresh
    // segment (O(corpus)). In between, a fold flushes the tail into a new L0 segment (O(tail)) —
    // EXCEPT when small segments have accumulated: then a *minor* compaction merges just the
    // newest small-segment run + the tail (O(small), not O(corpus)), so read-time fan-out stays
    // bounded without the indexer paying a full rebuild every few small writes. See
    // docs/concurrency-findings.md.
    if manifest.segments.is_empty() || manifest.segments.len() >= COMPACT_FANOUT {
        // Auto-select in-RAM vs streaming by size — the in-RAM merge is O(N) RAM, so a corpus too
        // big for memory uses the external-memory fold.
        let corpus = estimate_corpus_docs(ns).await?;
        if corpus >= streaming_threshold_docs {
            crate::streaming::index_streaming_with_budget(ns, tokenizer, opts, budget).await
        } else {
            index(ns, tokenizer, opts).await
        }
    } else {
        // Steady state: flush the WAL tail into a new L0 segment and append it — O(tail).
        // (Size-tiered `minor_compact` exists and is correctness-proven, but firing it here
        // over-works the single indexer and regressed reads — fan-out is no longer the bottleneck,
        // the indexer is. Left dormant for the queue-based indexer; see docs/concurrency-findings.md.)
        flush(ns, tokenizer, opts).await
    }
}

/// Segment count at which a fold compacts (merges all segments into one) instead of flushing. Keeps
/// the per-query segment fan-out — and the roundtrip budget — a small constant (see INV-7).
pub const COMPACT_FANOUT: usize = 8;

/// A segment with fewer than this many docs is "small" — a fold slice from a low-write interval.
pub const SMALL_SEGMENT_DOCS: u64 = 1000;

/// Compact once this many *small* segments have accumulated, just below the hard `COMPACT_FANOUT`
/// count cap. Firing here replaces the O(corpus) full rebuild the cap would otherwise do with a
/// cheap O(small) *minor* merge of only the small segments — a strict win. Deliberately close to
/// the cap (not aggressive): each minor compaction still opens a snapshot and rebuilds the merged
/// index, so firing every few folds over-works a busy single indexer (measured — it inflates the
/// WAL tail and slows reads more than the extra fan-out costs). See docs/concurrency-findings.md.
pub const SMALL_SEGMENT_FANOUT: usize = 7;

/// Flush the un-indexed WAL tail into a NEW L0 segment and append it (the LSM flush — O(tail), not
/// O(corpus)). Deletes and re-upserts in the slice become the segment's supersede overlay, hiding
/// the older copies at query time. See docs/segmented-index.md §4.
pub async fn flush(ns: &Namespace, tokenizer: &Tokenizer, opts: IndexOptions) -> Result<IndexOutcome> {
    use std::collections::{BTreeMap, HashSet};

    let (manifest, etag) = ns.read_manifest().await?;
    let head = ns.wal_head().await?;
    if head <= manifest.wal_index_cursor {
        return Ok(IndexOutcome { generation: manifest.version, doc_count: 0, published: false });
    }
    let cursor = manifest.wal_index_cursor;
    let scan = WalTail::new(ns).scan(cursor, Some(head)).await?;

    // Resolve the slice's live items: upserts, with in-slice patches applied.
    let mut items: Vec<StoredMemory> = Vec::new();
    let mut touched: HashSet<[u8; 16]> = HashSet::new();
    for (id, mut item) in scan.upserts {
        if let Some(deltas) = scan.pending_patches.get(&id) {
            mlake_core::wal::apply_deltas(&mut item, deltas);
        }
        touched.insert(id.0);
        items.push(item);
    }
    // The pre-flush snapshot: used to re-materialize patched older items and detect re-upserts.
    let node = crate::QueryNode::open(ns, tokenizer.clone()).await?;

    // Patches to ids not upserted in this slice target an older segment: re-materialize the full
    // item (with vector), apply the patch, and re-index it here — it supersedes the old copy.
    let patch_only: Vec<MemoryId> = scan
        .pending_patches
        .keys()
        .filter(|id| !touched.contains(&id.0))
        .copied()
        .collect();
    if !patch_only.is_empty() {
        for mut item in node.get_many(&patch_only, true).await? {
            if let Some(deltas) = scan.pending_patches.get(&item.id) {
                mlake_core::wal::apply_deltas(&mut item, deltas);
            }
            touched.insert(item.id.0);
            items.push(item);
        }
    }

    // Supersede overlay: kill, in older segments, every id this flush deletes plus every touched id
    // that ALSO exists in a segment (a genuine re-upsert — a brand-new id has no older copy to hide,
    // so it is left out, keeping the overlay small).
    let touched_ids: Vec<MemoryId> = touched.iter().map(|id| MemoryId(*id)).collect();
    let reupserts = node.segment_ids(&touched_ids).await?;
    let mut superseded: Vec<MemoryId> = scan.tombstones.clone();
    superseded.extend(reupserts);
    superseded.sort_unstable();
    superseded.dedup();

    // Links are NOT derived here — they were derived by the server at write time and travel in the
    // WAL as intrinsic data (`semantic_out`). The fold just carries them forward and builds `radj`
    // from them (below), so the index stays a pure speed optimization.
    let build_opts = IndexOptions {
        indexed_metadata_keys: manifest.indexed_metadata_keys.clone(),
        ..opts
    };

    // Build the L0 segment: a fresh per-type index over the slice's items (no copy-forward).
    let seg_id = mlake_core::MemoryId::new_v4().as_uuid().simple().to_string();
    let doc_count = items.len();
    let mut items_by_ft: BTreeMap<u8, Vec<StoredMemory>> = BTreeMap::new();
    for item in items {
        items_by_ft.entry(item.memory_type).or_default().push(item);
    }
    let mut indexes: BTreeMap<u8, mlake_core::manifest::FactTypeIndex> = BTreeMap::new();
    for (ft, ft_items) in items_by_ft {
        let fti = build_memory_type_index(ns, &seg_id, ft, ft_items, build_opts.clone())
            .await?;
        indexes.insert(ft, fti);
    }

    // Write the delete overlay and publish the new L0 segment at the head of the list.
    let seg_prefix = mlake_core::manifest::segment_prefix(&ns.name, &seg_id);
    let tomb = mlake_core::SegmentTombstones { superseded, predicates: scan.predicate_tombstones };
    let tomb_path = crate::generation::write_tombstones(&ns.store, &seg_prefix, &tomb).await?;

    let segment = mlake_core::Segment {
        id: seg_id,
        level: 0,
        seq_lo: cursor + 1,
        seq_hi: head,
        doc_count: doc_count as u64,
        indexes,
        tombstones: tomb_path,
    };
    let version = manifest.version + 1;
    let mut next = manifest.clone();
    next.version = version;
    next.wal_index_cursor = head;
    next.wal_head = head;
    next.prev_wal_index_cursor = manifest.wal_index_cursor;
    next.prev_segments = Vec::new(); // a flush drops no segment, so there is no grace window
    next.segments = std::iter::once(segment).chain(manifest.segments).collect();

    let published = match etag {
        Some(etag) => ns
            .swap_manifest(&etag, &next)
            .await
            .map(|_| true)
            .or_else(|e| if e.is_conflict() { Ok(false) } else { Err(e) })?,
        None => false,
    };
    Ok(IndexOutcome { generation: version, doc_count, published })
}

/// How many of the newest (front-of-list) segments a minor compaction should merge, or `None` if
/// one is not warranted. Merges the newest contiguous run of SMALL segments (older/larger segments
/// stay put — that is what keeps it O(small), not O(corpus)).
fn minor_compact_prefix(segments: &[mlake_core::Segment]) -> Option<usize> {
    let run = segments.iter().take_while(|s| s.doc_count < SMALL_SEGMENT_DOCS).count();
    (run >= SMALL_SEGMENT_FANOUT).then_some(run)
}

/// Merge the newest `merge_len` segments together with the WAL tail into ONE new L0 segment,
/// dropping those segments and leaving the older ones — a size-tiered *minor* compaction. It reads
/// only the merged segments (O(small)); the kept segments are untouched, so the new segment carries
/// a supersede overlay to keep hiding their now-shadowed copies, exactly as a flush does. Its result
/// is identical to a fresh open over the equivalent full rebuild (covered by
/// `minor_compaction_matches_a_full_rebuild`).
pub async fn minor_compact(
    ns: &Namespace,
    tokenizer: &Tokenizer,
    opts: IndexOptions,
    merge_len: usize,
) -> Result<IndexOutcome> {
    use std::collections::{BTreeMap, HashMap, HashSet};

    let (manifest, etag) = ns.read_manifest().await?;
    let head = ns.wal_head().await?;
    let cursor = manifest.wal_index_cursor;
    // The segment list may have changed since the trigger read it; fall back to a plain flush if the
    // requested prefix no longer holds.
    if merge_len < 2 || merge_len > manifest.segments.len() {
        return flush(ns, tokenizer, opts).await;
    }
    let build_opts = IndexOptions {
        indexed_metadata_keys: manifest.indexed_metadata_keys.clone(),
        ..opts
    };
    let merged_segs = &manifest.segments[..merge_len];
    let kept_segs = manifest.segments[merge_len..].to_vec();

    // (1) Materialize the merged segments' live items (newest-first, honoring their internal
    // supersede overlays), and collect those overlays to carry forward — they may hide copies that
    // live in the KEPT segments, which must keep being hidden.
    let mut kill: HashMap<[u8; 16], u64> = HashMap::new();
    let mut carry_superseded: Vec<MemoryId> = Vec::new();
    let mut carry_predicates: Vec<(u64, mlake_core::Predicate)> = Vec::new();
    for seg in merged_segs {
        let tomb = crate::generation::read_tombstones(&ns.store, &seg.tombstones, None).await?;
        for id in &tomb.superseded {
            let e = kill.entry(id.0).or_insert(0);
            *e = (*e).max(seg.seq_hi);
        }
        carry_superseded.extend(tomb.superseded);
        carry_predicates.extend(tomb.predicates);
    }
    let mut by_id: BTreeMap<[u8; 16], StoredMemory> = BTreeMap::new();
    for seg in merged_segs {
        for ft in seg.memory_types() {
            let fti = seg.index(ft).unwrap();
            let gen =
                crate::generation::read_generation(&ns.store, &fti.files, manifest.version, None)
                    .await?;
            for cluster in &gen.clusters {
                for item in cluster {
                    if by_id.contains_key(&item.id.0)
                        || kill.get(&item.id.0).is_some_and(|&s| s > seg.seq_hi)
                    {
                        continue;
                    }
                    by_id.insert(item.id.0, item.clone());
                }
            }
        }
    }
    if !carry_predicates.is_empty() {
        by_id.retain(|_, m| !carry_predicates.iter().any(|(seq, p)| m.write_seq < *seq && p.matches(m)));
    }

    // (2) Fold the WAL tail into the merged set (identical to `index`/`flush`).
    let scan = WalTail::new(ns).scan(cursor, Some(head)).await?;
    let mut touched: HashSet<[u8; 16]> = HashSet::new();
    for id in &scan.tombstones {
        by_id.remove(&id.0);
        touched.insert(id.0);
    }
    for (id, mut item) in scan.upserts {
        if let Some(deltas) = scan.pending_patches.get(&id) {
            mlake_core::wal::apply_deltas(&mut item, deltas);
        }
        touched.insert(id.0);
        by_id.insert(id.0, item);
    }
    if !scan.predicate_tombstones.is_empty() {
        by_id.retain(|_, m| {
            !scan.predicate_tombstones.iter().any(|(seq, p)| m.write_seq < *seq && p.matches(m))
        });
    }

    // A pre-fold snapshot: for re-materializing tail patches that target an id living only in a KEPT
    // segment, and for the supersede membership check below.
    let node = crate::QueryNode::open(ns, tokenizer.clone()).await?;
    let patch_only: Vec<MemoryId> = scan
        .pending_patches
        .keys()
        .filter(|id| !touched.contains(&id.0) && !by_id.contains_key(&id.0))
        .copied()
        .collect();
    if !patch_only.is_empty() {
        for mut item in node.get_many(&patch_only, true).await? {
            if let Some(deltas) = scan.pending_patches.get(&item.id) {
                mlake_core::wal::apply_deltas(&mut item, deltas);
            }
            touched.insert(item.id.0);
            by_id.insert(item.id.0, item);
        }
    }

    // (3) The new segment's supersede overlay: the merged segments' overlays carried forward, plus
    // every tail delete, plus tail-touched ids that also exist in a segment (a re-upsert whose older
    // copy — possibly in a KEPT segment — must be shadowed). Over-inclusive is safe: an entry for an
    // id in no kept segment hides nothing, and the new segment's OWN items (seq_hi = head) are not
    // hidden by an equal seq_hi.
    let touched_ids: Vec<MemoryId> = touched.iter().map(|id| MemoryId(*id)).collect();
    let mut superseded: Vec<MemoryId> = carry_superseded;
    superseded.extend(scan.tombstones.iter().copied());
    superseded.extend(node.segment_ids(&touched_ids).await?);
    superseded.sort_unstable();
    superseded.dedup();
    let mut predicates = carry_predicates;
    predicates.extend(scan.predicate_tombstones.iter().cloned());

    // (4) Build the merged segment. Semantic links are left as-is (dangling targets are filtered at
    // read time, and a target may live in a kept segment — pruning to only the merged live set would
    // wrongly drop valid cross-segment links).
    let seg_id = mlake_core::MemoryId::new_v4().as_uuid().simple().to_string();
    let items: Vec<StoredMemory> = by_id.into_values().collect();
    let doc_count = items.len();
    let mut items_by_ft: BTreeMap<u8, Vec<StoredMemory>> = BTreeMap::new();
    for item in items {
        items_by_ft.entry(item.memory_type).or_default().push(item);
    }
    let mut indexes: BTreeMap<u8, mlake_core::manifest::FactTypeIndex> = BTreeMap::new();
    for (ft, ft_items) in items_by_ft {
        let fti = build_memory_type_index(ns, &seg_id, ft, ft_items, build_opts.clone()).await?;
        indexes.insert(ft, fti);
    }

    let seg_prefix = mlake_core::manifest::segment_prefix(&ns.name, &seg_id);
    let tomb = mlake_core::SegmentTombstones { superseded, predicates };
    let tomb_path = crate::generation::write_tombstones(&ns.store, &seg_prefix, &tomb).await?;

    let merged_seq_lo = merged_segs.iter().map(|s| s.seq_lo).min().unwrap_or(cursor + 1);
    let segment = mlake_core::Segment {
        id: seg_id,
        level: 0,
        seq_lo: merged_seq_lo,
        seq_hi: head, // includes the freshly folded tail, so it shadows the kept (older) segments
        doc_count: doc_count as u64,
        indexes,
        tombstones: tomb_path,
    };

    // (5) Publish: merged segment at the front, kept segments behind. Advance the cursor.
    let version = manifest.version + 1;
    let mut next = manifest.clone();
    next.version = version;
    next.wal_index_cursor = head;
    next.wal_head = head;
    next.prev_wal_index_cursor = manifest.wal_index_cursor;
    next.prev_segments = manifest.segments.clone();
    next.segments = std::iter::once(segment).chain(kept_segs).collect();

    let published = match etag {
        Some(etag) => ns
            .swap_manifest(&etag, &next)
            .await
            .map(|_| true)
            .or_else(|e| if e.is_conflict() { Ok(false) } else { Err(e) })?,
        None => false,
    };
    Ok(IndexOutcome { generation: version, doc_count, published })
}

/// Derive each new memory's semantic kNN links **at write time**, before the WAL commit, against
/// the current index (`node`) plus the other memories in the same write batch. This is what moves
/// link generation off the fold and onto the write path (SPEC change): the links then travel in the
/// WAL as intrinsic data, so the index is a pure speed optimization and a query over the un-indexed
/// tail is already correct. `node` is a snapshot of the *committed* corpus (index + existing tail);
/// the batch's own items are compared in RAM because they are not yet committed and so invisible to
/// `node`. Sets each memory's `semantic_out` (top-`MAX_SEMANTIC_OUT` at cosine ≥ threshold).
/// Timing split of [`derive_links_for_write`], so a trace can tell the O(new·N) corpus queries
/// apart from the O(n²) within-batch compare.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeriveStats {
    /// Wall time in the per-item corpus queries (step a) — grows with corpus size.
    pub query_ms: f64,
    /// Wall time in the within-batch cosine compare (step b) — grows with batch size squared.
    pub batch_ms: f64,
    /// Number of corpus queries issued (one per non-empty new item).
    pub queries: usize,
}

pub async fn derive_links_for_write(
    node: &crate::QueryNode,
    memories: &mut [mlake_core::Memory],
    metrics: &mlake_store::QueryMetrics,
) -> Result<DeriveStats> {
    let tags = mlake_core::TagFilter::new(Vec::new(), mlake_core::TagsMatch::Any);
    let depths = crate::ArmDepths {
        vector: MAX_SEMANTIC_OUT + 4,
        text: 0,
        graph: 0,
        nprobe: 16,
        graph_seed_min: crate::query_node::DEFAULT_GRAPH_SEED_MIN_SIMILARITY,
        // Links only need approximate neighbours: rank by the RaBitQ scan estimate and skip the
        // exact rerank (the dominant per-item cost at scale). See ArmDepths::exact_rerank.
        exact_rerank: false,
    };
    let n = memories.len();
    // Snapshot vectors/ids/types so the within-batch compare can read while we mutate semantic_out.
    let vecs: Vec<Vec<f32>> = memories.iter().map(|m| m.vector.clone()).collect();
    let ids: Vec<MemoryId> = memories.iter().map(|m| m.id).collect();
    let mts: Vec<u8> = memories.iter().map(|m| m.memory_type).collect();
    // Precompute each vector's norm ONCE, so the O(n²) within-batch compare below is a bare SIMD
    // dot per pair instead of re-deriving `|vecs[i]|²` on every one of its `n` comparisons.
    let norms: Vec<f32> =
        vecs.iter().map(|v| if v.is_empty() { 0.0 } else { mlake_core::norm(v) }).collect();
    use futures::stream::{StreamExt, TryStreamExt};
    let mut stats = DeriveStats::default();

    // (a) For each new item, query the committed index + tail for its nearest neighbours (one
    // exact-scored vector query each). These queries are independent and I/O+CPU bound, so run them
    // with bounded concurrency instead of one-await-at-a-time — this is the dominant write-path cost
    // and the batch's items don't depend on one another here. `node`/`metrics` are shared by `&`
    // (QueryNode reads are lock-free snapshots; QueryMetrics is all atomics), so this is safe.
    const DERIVE_QUERY_CONCURRENCY: usize = 8;
    let qt = std::time::Instant::now();
    let mut query_neigh: Vec<Vec<(MemoryId, f32)>> = futures::stream::iter(0..n)
        .map(|i| {
            let (vecs, ids, mts, tags) = (&vecs, &ids, &mts, &tags);
            async move {
                if vecs[i].is_empty() {
                    return Ok::<_, crate::Error>(Vec::new());
                }
                let raw = node
                    .query_raw_metered(mts[i], Some(&vecs[i]), None, tags, depths, None, Default::default(), metrics)
                    .await?;
                let mut neigh = Vec::new();
                for h in raw {
                    if let Some(d) = h.dense {
                        if h.id != ids[i] && d.score >= SEMANTIC_LINK_THRESHOLD {
                            neigh.push((h.id, d.score));
                        }
                    }
                }
                Ok(neigh)
            }
        })
        .buffered(DERIVE_QUERY_CONCURRENCY)
        .try_collect()
        .await?;
    // Wall time of the whole concurrent query phase (not the sum of per-query times).
    stats.query_ms += qt.elapsed().as_secs_f64() * 1000.0;
    stats.queries += vecs.iter().filter(|v| !v.is_empty()).count();

    // (b) Add neighbours within this same batch (not yet committed, so invisible to `node`), then
    // merge, truncate and assign. The batch is disjoint from the index, so these never duplicate the
    // query hits. Pure CPU — cheap after prenorm — so it stays a sequential pass.
    let bt = std::time::Instant::now();
    for i in 0..n {
        if vecs[i].is_empty() {
            continue;
        }
        let mut neigh = std::mem::take(&mut query_neigh[i]);
        for j in 0..n {
            if j == i || mts[j] != mts[i] || vecs[j].is_empty() {
                continue;
            }
            let sim = mlake_core::cosine_prenorm_both(&vecs[i], norms[i], &vecs[j], norms[j]);
            if sim >= SEMANTIC_LINK_THRESHOLD {
                neigh.push((ids[j], sim));
            }
        }
        neigh.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        neigh.truncate(MAX_SEMANTIC_OUT);
        memories[i].semantic_out =
            neigh.into_iter().map(|(id, sim)| SemanticEdge { target: id, weight: Weight::from_f32(sim) }).collect();
    }
    stats.batch_ms += bt.elapsed().as_secs_f64() * 1000.0;

    Ok(stats)
}

/// Cheap live-document estimate for fold selection: the previous generation's indexed count
/// (exact, from the manifest) plus the un-indexed WAL tail approximated from its object sizes
/// (a LIST, no GETs — so it stays O(1) roundtrips even when the tail is the whole first build).
async fn estimate_corpus_docs(ns: &Namespace) -> Result<usize> {
    let (manifest, _etag) = ns.read_manifest().await?;
    let prev_docs: usize = manifest.doc_count() as usize;
    let head = ns.wal_head().await?;

    let mut tail_bytes: u64 = 0;
    let mut start = manifest.wal_index_cursor + 1;
    loop {
        let (objs, next) = ns.list_wal(start, 100_000).await?;
        tail_bytes += objs.iter().filter(|o| o.seq <= head).map(|o| o.size_bytes).sum::<u64>();
        match next {
            Some(n) if n <= head => start = n,
            _ => break,
        }
    }
    Ok(prev_docs + (tail_bytes / NOMINAL_DOC_BYTES) as usize)
}

/// Build the next generation for a namespace and publish it.
/// Fold the bank's WAL into a new generation for each fact type and publish one manifest.
///
/// Fact types are fully independent indexes (no shared links/vectors/postings), so the fold
/// partitions the live item set by `memory_type` and builds a separate generation per type,
/// each a fresh build over its slice (SCALE.md Phase 4). One WAL, one
/// manifest — so a `bank + [memory_types]` query reads a single manifest and fans out.
pub async fn index(ns: &Namespace, tokenizer: &Tokenizer, opts: IndexOptions) -> Result<IndexOutcome> {
    let (manifest, etag) = ns.read_manifest().await?;
    let head = ns.wal_head().await?;
    // The declared keys ride in the build options every build path already carries.
    let opts = IndexOptions {
        indexed_metadata_keys: manifest.indexed_metadata_keys.clone(),
        ..opts
    };

    // Resolve the live set across ALL segments (this is the compaction merge; on a first build the
    // list is empty). A newer segment (higher seq_hi) shadows an older copy of a re-upserted or
    // deleted id via the segments' supersede overlays; segment predicate-deletes are materialized.
    let mut kill: std::collections::HashMap<[u8; 16], u64> = std::collections::HashMap::new();
    let mut seg_predicates: Vec<(u64, mlake_core::Predicate)> = Vec::new();
    for seg in &manifest.segments {
        let tomb = crate::generation::read_tombstones(&ns.store, &seg.tombstones, None).await?;
        for id in tomb.superseded {
            let e = kill.entry(id.0).or_insert(0);
            *e = (*e).max(seg.seq_hi);
        }
        seg_predicates.extend(tomb.predicates);
    }
    let mut by_id: std::collections::BTreeMap<[u8; 16], StoredMemory> =
        std::collections::BTreeMap::new();
    for seg in &manifest.segments {
        // Newest-first, so an already-present id (from a newer segment) wins.
        for ft in seg.memory_types() {
            let fti = seg.index(ft).unwrap();
            let gen =
                crate::generation::read_generation(&ns.store, &fti.files, manifest.version, None)
                    .await?;
            for cluster in &gen.clusters {
                for item in cluster {
                    if by_id.contains_key(&item.id.0)
                        || kill.get(&item.id.0).is_some_and(|&s| s > seg.seq_hi)
                    {
                        continue; // a newer segment provided this id, or deleted/re-upserted it
                    }
                    by_id.insert(item.id.0, item.clone());
                }
            }
        }
    }
    if !seg_predicates.is_empty() {
        by_id.retain(|_, m| {
            !seg_predicates.iter().any(|(seq, p)| m.write_seq < *seq && p.matches(m))
        });
    }

    let scan = WalTail::new(ns)
        .scan(manifest.wal_index_cursor, Some(head))
        .await?;
    for id in &scan.tombstones {
        by_id.remove(&id.0);
    }
    for (id, item) in scan.upserts {
        by_id.insert(id.0, item);
    }
    // Materialize predicate deletes: drop every memory a tail predicate tombstone matches
    // whose last write predates it (a same-entry re-ingest upsert has write_seq == the
    // predicate's seq, so it survives). This is where the eager scan becomes free — the fold
    // is already reading every cluster.
    if !scan.predicate_tombstones.is_empty() {
        by_id.retain(|_, m| {
            !scan
                .predicate_tombstones
                .iter()
                .any(|(seq, p)| m.write_seq < *seq && p.matches(m))
        });
    }
    for item in by_id.values_mut() {
        if let Some(deltas) = scan.pending_patches.get(&MemoryId(item.id.0)) {
            mlake_core::wal::apply_deltas(item, deltas);
        }
    }

    let mut items: Vec<StoredMemory> = by_id.into_values().collect();
    let live: std::collections::HashSet<[u8; 16]> = items.iter().map(|i| i.id.0).collect();
    for item in items.iter_mut() {
        item.semantic_out.retain(|e| live.contains(&e.target.0));
    }

    let version = manifest.version + 1;
    let doc_count = items.len();

    // Partition the live items by fact type (BTreeMap keeps a deterministic type order).
    let mut items_by_ft: std::collections::BTreeMap<u8, Vec<StoredMemory>> =
        std::collections::BTreeMap::new();
    for item in items {
        items_by_ft.entry(item.memory_type).or_default().push(item);
    }

    // Phase 1: the in-RAM fold produces ONE full segment (a full rebuild over the whole live set),
    // published as the sole live segment. Phase 2 switches to flushing just the tail into a new L0
    // segment and appending it.
    let seg_id = mlake_core::MemoryId::new_v4().as_uuid().simple().to_string();
    let mut indexes: std::collections::BTreeMap<u8, mlake_core::manifest::FactTypeIndex> =
        std::collections::BTreeMap::new();
    for (ft, ft_items) in items_by_ft {
        // A full rebuild of the merged live set — no copy-forward (that is the flush's job); prev
        // is None so this segment is fresh, retrained, with its centroids over the merged set.
        let fti =
            build_memory_type_index(ns, &seg_id, ft, ft_items, opts.clone())
                .await?;
        indexes.insert(ft, fti);
    }

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

    Ok(IndexOutcome {
        generation: version,
        doc_count,
        published,
    })
}


/// Build one fact type's independent generation (train centroids + local split + IVF link
/// derivation) from its slice of the live items, and return its manifest entry.
///
/// The segmented index always builds a fresh segment: a flush writes an L0 segment over its
/// slice, a compaction rebuilds the merged live set. There is no incremental in-place fold,
/// so centroids are always trained from scratch and every cluster is written — the
/// assign-only-vs-retrain and copy-forward paths that a single mutable generation needed are
/// gone with it.
async fn build_memory_type_index(
    ns: &Namespace,
    seg_id: &str,
    memory_type: u8,
    items: Vec<StoredMemory>,
    opts: IndexOptions,
) -> Result<mlake_core::manifest::FactTypeIndex> {
    let doc_count = items.len();
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    // One dimension per fact type, checked once here rather than deep inside the parallel
    // link derivation: everything downstream (centroid training, probing, semantic-edge
    // cosine) assumes it. Failing the fold with a typed error keeps a corpus that somehow
    // mixed dimensions from becoming an unexplained panic in a rayon worker.
    // Also the dimension the vector block encodes against; 0 when the type is text-only.
    let vector_dim = mlake_core::uniform_dim(vectors.iter().map(|v| v.as_slice()))?.unwrap_or(0);
    let tt = std::time::Instant::now();
    let mut centroids = train_centroids(&vectors, opts.seed);
    let train_count = doc_count as u64;
    phase_log("train", tt);

    // Assign every item to its nearest centroid. The other dominant O(N·k) fold pass; each
    // assignment is independent, so run it across cores (deterministic — nearest is pure).
    let mut assignments: Vec<usize> = if centroids.is_empty() {
        vec![0; items.len()]
    } else {
        use rayon::prelude::*;
        items.par_iter().map(|i| centroids.assign(&i.vector)).collect()
    };

    // Reshapes centroids/assignments in place; which clusters split no longer matters now
    // that every cluster is written fresh (it only guarded copy-forward before).
    local_split(&mut centroids, &items, &mut assignments, opts.seed);

    // No link derivation here: `items` already carry their `semantic_out` (derived at write time,
    // carried in the WAL and forward through folds). The fold only builds `radj` from them below.

    let k = centroids.len().max(1);
    let mut clusters: Vec<Vec<StoredMemory>> = vec![Vec::new(); k];
    for (item, &c) in items.iter().zip(assignments.iter()) {
        clusters[c].push(item.clone());
    }

    // Per-fact-type prefix so different types never collide on object keys.
    let prefix = format!("{}/mt{memory_type}", mlake_core::manifest::segment_prefix(&ns.name, seg_id));

    // Write every cluster as a pair: `cluster-{i}.bin` (payload) and `cluster-{i}.vec` (the
    // codes, tag bitmaps and update times). Members stay in one order across both.
    let tw = std::time::Instant::now();
    let mut cluster_paths: Vec<Option<String>> = vec![None; k];
    let mut vector_paths: Vec<Option<String>> = vec![None; k];
    let mut writes = Vec::new();
    for i in 0..k {
        // The embedding leaves the cluster file and lives in the vector block; the two
        // stay in member order so index `j` means the same memory in both.
        let ids: Vec<MemoryId> = clusters[i].iter().map(|m| m.id).collect();
        // A text-only memory carries no embedding. Absent is not a dimension error —
        // pad it to zeros, which scores 0 on the vector arm and so never surfaces there,
        // exactly as an empty vector did through `cosine_opt`.
        let vectors: Vec<Vec<f32>> = clusters[i]
            .iter()
            .map(|m| if m.vector.is_empty() { vec![0.0; vector_dim] } else { m.vector.clone() })
            .collect();
        let mut items = clusters[i].clone();
        for m in items.iter_mut() {
            m.vector = Vec::new();
        }
        let cf = ClusterFile { centroid_id: i as u32, items };
        // Tags and write times travel with the codes so the probe can filter without the
        // payload half.
        let member_tags: Vec<Vec<String>> = clusters[i].iter().map(|m| m.tags.clone()).collect();
        let member_updated: Vec<i64> = clusters[i]
            .iter()
            .map(|m| m.timestamps.updated_at.unwrap_or(mlake_ivf::UPDATED_UNKNOWN))
            .collect();
        let block = VectorBlock::encode_with_columns(
            opts.vector_codec,
            vector_dim,
            &ids,
            &vectors,
            Some(&member_tags),
            &member_updated,
        )?;
        let block_bytes = block.to_bytes();
        let store = ns.store.clone();
        let prefix = prefix.clone();
        writes.push(async move {
            let cluster = crate::generation::write_cluster_file(&store, &prefix, i, &cf).await?;
            let vectors =
                crate::generation::write_vector_block(&store, &prefix, i, block_bytes).await?;
            Ok::<_, crate::Error>((i, cluster, vectors))
        });
    }
    // Bounded concurrency instead of one sequential PUT at a time — the dominant cost of a
    // build over S3 (SCALE.md Phase 3 perf).
    {
        use futures::stream::{StreamExt, TryStreamExt};
        let written: Vec<(usize, String, String)> = futures::stream::iter(writes)
            .buffer_unordered(32)
            .try_collect()
            .await?;
        for (i, cluster, vectors) in written {
            cluster_paths[i] = Some(cluster);
            vector_paths[i] = Some(vectors);
        }
    }
    let cluster_paths: Vec<String> = cluster_paths.into_iter().map(|p| p.unwrap_or_default()).collect();
    let vector_paths: Vec<String> = vector_paths.into_iter().map(|p| p.unwrap_or_default()).collect();
    phase_log("cluster_write", tw);

    // pk / radj / fts, scoped to this fact type.
    let mut pk_entries: Vec<(MemoryId, u32)> = Vec::with_capacity(doc_count);
    for (ci, cluster) in clusters.iter().enumerate() {
        for item in cluster {
            pk_entries.push((item.id, ci as u32));
        }
    }
    let pk_tables = crate::sstable::PkTable::build(pk_entries);

    let tfts = std::time::Instant::now();
    let fts = mlake_fts::TantivyFts::build_with_tags(
        items.iter().map(|i| (i.id, i.fts_text(), i.tags.as_slice())),
        mlake_fts::Tokenizer::new(mlake_fts::TokenizerConfig::default()),
    )
    .map_err(|e| crate::Error::Core(mlake_core::Error::Encode(e.to_string())))?;
    let fts_split = fts.split_bytes().to_vec();
    phase_log("fts_build", tfts);

    let mut radj_pairs: Vec<(MemoryId, InEdge)> = Vec::new();
    for item in &items {
        for edge in &item.semantic_out {
            radj_pairs.push((
                edge.target,
                InEdge { source: item.id, kind: EdgeKind::Semantic, weight: edge.weight.to_f32() },
            ));
        }
        for edge in &item.causal_out {
            radj_pairs.push((
                edge.target,
                InEdge {
                    source: item.id,
                    kind: EdgeKind::Causal(LinkTypeTag::from(edge.link_type)),
                    weight: edge.weight.to_f32(),
                },
            ));
        }
    }
    let radj_tables = crate::sstable::RadjTable::build(radj_pairs);

    // Entity postings: EntityId -> the memories that carry it. This is what lets the graph
    // arm's entity expansion find sharers anywhere in the corpus (not just probed clusters).
    let mut entity_pairs: Vec<(mlake_core::EntityId, MemoryId)> = Vec::new();
    for item in &items {
        for e in &item.entity_ids {
            entity_pairs.push((*e, item.id));
        }
    }
    let entity_tables = crate::sstable::EntityTable::build(entity_pairs);

    // Time index: effective_ts = COALESCE(occurred_start, mentioned_at, occurred_end) -> id.
    // Entry-point selection for the temporal arm is one ranged scan of this key range.
    let mut time_pairs: Vec<(i64, MemoryId)> = Vec::new();
    for item in &items {
        let t = &item.timestamps;
        if let Some(ts) = t.occurred_start.or(t.mentioned_at).or(t.occurred_end) {
            time_pairs.push((ts, item.id));
        }
    }
    let time_tables = crate::sstable::TimeTable::build(time_pairs);

    // Payload store: one addressable row per memory (embedding stripped), so a point read
    // (FTS/graph hit, `get`) fetches one memory instead of its whole cluster file.
    let payload_tables = crate::sstable::PayloadTable::build(&items);
    // Full precision for stage two. Never scanned; point-fetched for the few candidates
    // whose error bound leaves them possibly in the true top-k.
    let rerank_tables = crate::sstable::RerankTable::build(&items);

    // Per-cluster summaries: the union of each cluster's tags + an untagged flag + the span of
    // its write times, so a query can prune clusters that cannot contain a matching memory
    // (SCALE.md Phase 4b).
    let tag_summary: crate::generation::TagSummary = clusters
        .iter()
        .map(|cluster| {
            let mut tags: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            let mut has_untagged = false;
            for m in cluster {
                if m.tags.is_empty() {
                    has_untagged = true;
                } else {
                    tags.extend(m.tags.iter().cloned());
                }
            }
            crate::generation::ClusterTagSummary {
                tags: tags.into_iter().collect(),
                has_untagged,
                updated_range: crate::generation::ClusterTagSummary::range_of(
                    cluster.iter().map(|m| &m.timestamps.updated_at),
                ),
            }
        })
        .collect();

    // Value-counts for the declared metadata keys, for this segment's slice. Summed across
    // segments (and corrected for the tail) by `MetadataStats`.
    let meta_counts = crate::generation::build_meta_counts(&items, &opts.indexed_metadata_keys);

    // Edge totals for this segment's slice, summed across segments (and corrected for the tail)
    // by `LinkStats` so the bank stats page reports a link count without a corpus walk.
    let semantic_edge_count: usize = items.iter().map(|m| m.semantic_out.len()).sum();
    let causal_edge_count: usize = items.iter().map(|m| m.causal_out.len()).sum();

    let twg = std::time::Instant::now();
    let files = write_generation(
        &ns.store,
        &prefix,
        &centroids,
        cluster_paths,
        vector_paths,
        &fts_split,
        radj_tables.into(),
        pk_tables.into(),
        entity_tables.into(),
        time_tables.into(),
        payload_tables.into(),
        rerank_tables.into(),
        &tag_summary,
        doc_count,
        semantic_edge_count,
        causal_edge_count,
        meta_counts,
    )
    .await?;
    phase_log("write_meta", twg);

    Ok(mlake_core::manifest::FactTypeIndex { train_count, files })
}

/// Log an index phase's duration to stderr when `MEMLAKE_TIMING` is set. Cheap enough to
/// leave in — a couple of `Instant`s per fold.
fn phase_log(phase: &str, start: std::time::Instant) {
    if std::env::var("MEMLAKE_TIMING").is_ok() {
        eprintln!("[index] {phase}: {:.2}s", start.elapsed().as_secs_f64());
    }
}

/// Split any cluster grown past 8× the average size, in place: a 2-means over just that
/// cluster's members yields two sub-centroids (one replaces the original, one is appended),
/// and only that cluster's members are reassigned between them (SPFresh-lite core, LIRE).
/// Returns the set of centroid indices that changed, which the caller marks dirty.
fn local_split(
    centroids: &mut mlake_ivf::Centroids,
    items: &[StoredMemory],
    assignments: &mut [usize],
    seed: u64,
) -> std::collections::HashSet<usize> {
    let mut split = std::collections::HashSet::new();
    let k = centroids.len();
    if k == 0 || items.is_empty() {
        return split;
    }
    let avg = items.len() as f32 / k as f32;
    let threshold = (8.0 * avg).ceil() as usize;

    let mut counts = vec![0usize; k];
    for &c in assignments.iter() {
        counts[c] += 1;
    }

    for i in 0..k {
        if counts[i] <= threshold || counts[i] < 2 {
            continue;
        }
        let members: Vec<usize> = (0..items.len()).filter(|&j| assignments[j] == i).collect();
        let vecs: Vec<Vec<f32>> = members.iter().map(|&j| items[j].vector.clone()).collect();
        let sub = mlake_ivf::kmeans::train(&vecs, 2, 10, seed);
        if sub.len() < 2 {
            continue;
        }
        // Replace centroid i with the first sub-centroid; append the second.
        let new_idx = centroids.push(sub[1].clone());
        centroids.vectors[i] = sub[0].clone();
        for &j in &members {
            let d0 = mlake_ivf::kmeans::sq_dist_pub(&items[j].vector, &sub[0]);
            let d1 = mlake_ivf::kmeans::sq_dist_pub(&items[j].vector, &sub[1]);
            assignments[j] = if d1 < d0 { new_idx } else { i };
        }
        split.insert(i);
        split.insert(new_idx);
    }
    split
}
