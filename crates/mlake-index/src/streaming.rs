//! The external-memory ("streaming") fold: build a generation whose peak RAM is bounded by a
//! fixed buffer budget instead of the corpus size, so a first build scales past what fits in
//! memory (the O(N)-RAM in-RAM [`crate::indexer::index`] caps a 36 GB box at ~5–6M).
//!
//! How it stays bounded:
//!   * **Resolution** streams the previous generation cluster-by-cluster (one cluster resident)
//!     and applies the WAL tail (bounded), spilling each resolved live item to a per-type disk
//!     [`ItemSpill`]. Nothing but the current cluster + the tail is in RAM.
//!   * **Per type** it trains centroids on a reservoir sample ([`train_centroids_k`] — never
//!     scans all N), then makes ONE pass over the spill: assign each item to a cluster and feed
//!     five [`ExternalSort`]s (the cluster grouping carries the full item bytes; pk / payload /
//!     entity / time carry their SSTable fragments) plus the FTS builder, tallying cluster sizes
//!     and tag summaries. The external sorts spill sorted runs and k-way-merge them, so the
//!     SSTables and cluster files are written from bounded memory.
//!
//! Scope: this is the **bulk build** path. It does NOT derive semantic kNN links (that pass
//! reads a 16-cluster neighbourhood per new item, incompatible with one-cluster-at-a-time
//! streaming; at scale links are derived incrementally / at query time — the same thing
//! `--no-links` models). causal edges (client-provided, inline) are preserved. It also skips
//! the local-split rebalance. The result is a correct, queryable generation equivalent to an
//! in-RAM first build with `derive_links=false`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use mlake_core::wal::{apply_deltas, Delta};
use mlake_core::{MemoryId, Op, Predicate, StoredMemory};
use mlake_fts::{TantivyFtsBuilder, Tokenizer};
use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag};
use mlake_ivf::ClusterFile;
use mlake_wal::Namespace;

use crate::generation::{write_cluster_file, write_generation, ClusterTagSummary, TagSummary};
use crate::indexer::{IndexOptions, IndexOutcome};
use crate::spill::{ExternalSort, ItemSpill, Merge};
use crate::sstable::{ts_key, RadjTable};
use crate::Result;

/// Total external-sort buffer budget per memory_type, split across the five sorts. Peak RAM is
/// roughly this plus one open cluster and the FTS writer's arena.
const SORT_BUDGET_BYTES: usize = 512 * 1024 * 1024;
/// Vectors sampled for centroid training (mini-batch k-means).
const TRAIN_SAMPLE: usize = 50_000;

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

/// Per-id resolution state built while streaming the WAL — no item bodies, so it stays small
/// (~a few dozen bytes per id) even at a corpus that would not fit in RAM as full memories.
struct WalId {
    /// Seq of the latest upsert for this id, when the latest op *is* an upsert (else `None`).
    winner: Option<u64>,
    tombstoned: bool,
    /// Deltas from patches after the latest upsert (or on a still-in-generation item), in order.
    patches: Vec<Delta>,
    /// Set once pass 2 has spilled this id, so a same-entry duplicate upsert can't double-emit.
    emitted: bool,
}
impl WalId {
    /// State for an id first seen via a patch — its item lives in the generation, not the WAL.
    fn gen_item() -> Self {
        Self { winner: None, tombstoned: false, patches: Vec::new(), emitted: false }
    }
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

/// Build a generation for `ns` with bounded memory. See the module docs for scope.
pub async fn index_streaming(
    ns: &Namespace,
    tokenizer: &Tokenizer,
    opts: IndexOptions,
) -> Result<IndexOutcome> {
    let (manifest, etag) = ns.read_manifest().await?;
    let head = ns.wal_head().await?;
    let generation = manifest.generation + 1;

    // ---- Phase 1: resolve live items, spill by type — all in bounded RAM ----
    //
    // The WAL is folded with the same last-write-wins semantics as `fold_entries`, but STREAMED:
    // pass 1 reads each WAL entry to build id-level state only (winner upsert seq, tombstone,
    // accumulated patches — no item bodies), pass 2 re-reads the WAL and spills each winning
    // upsert's resolved item. The previous generation is streamed cluster-by-cluster and
    // overlaid with that state. Predicate deletes are evaluated at emit time (they need the
    // item's metadata + write_seq). Peak RAM is the id-level map + one WAL entry / one cluster.
    let seqs = wal_seqs(ns, manifest.wal_index_cursor, head).await?;

    // Pass 1: id-level state.
    let mut state: HashMap<[u8; 16], WalId> = HashMap::new();
    let mut preds: Vec<(u64, Predicate)> = Vec::new();
    for &seq in &seqs {
        let entry = ns.read_wal_entry(seq).await?;
        for op in entry.ops {
            match op {
                Op::Upsert(m) => {
                    // A new upsert replaces everything for this id (last write wins), which is
                    // what drops patches recorded before it and revives a tombstone.
                    state.insert(m.id.0, WalId { winner: Some(seq), tombstoned: false, patches: Vec::new(), emitted: false });
                }
                Op::Tombstone { id } => {
                    state.insert(id.0, WalId { winner: None, tombstoned: true, patches: Vec::new(), emitted: false });
                }
                Op::Patch { id, deltas } => {
                    let e = state.entry(id.0).or_insert_with(WalId::gen_item);
                    if !e.tombstoned {
                        e.patches.extend(deltas);
                    }
                }
                Op::TombstoneWhere { predicate } => preds.push((seq, predicate)),
                Op::Guard { .. } => {}
            }
        }
    }

    let predicate_deleted = |m: &StoredMemory| -> bool {
        preds.iter().any(|(seq, p)| m.write_seq < *seq && p.matches(m))
    };

    let mut resolver = Resolver::new(opts.seed);

    // Overlay the previous generation, streamed cluster-by-cluster.
    for ft in manifest.memory_types() {
        let fti = manifest.index(ft).unwrap();
        for path in &fti.files.clusters {
            if path.is_empty() {
                continue;
            }
            let bytes = ns.store.get_immutable(path, None).await?;
            let cf = ClusterFile::from_bytes(&bytes)?;
            for mut item in cf.items {
                if let Some(s) = state.get(&item.id.0) {
                    if s.tombstoned || s.winner.is_some() {
                        continue; // deleted, or superseded by a WAL upsert (emitted in pass 2)
                    }
                    apply_deltas(&mut item, &s.patches); // deferred patches to a gen item
                }
                if predicate_deleted(&item) {
                    continue;
                }
                item.semantic_out.clear(); // bulk build derives no semantic links
                resolver.emit(item)?;
            }
        }
    }

    // Pass 2: re-stream the WAL and emit each winning upsert's resolved item.
    for &seq in &seqs {
        let entry = ns.read_wal_entry(seq).await?;
        for op in entry.ops {
            let Op::Upsert(m) = op else { continue };
            let Some(s) = state.get_mut(&m.id.0) else { continue };
            if s.winner != Some(seq) || s.emitted {
                continue; // not the winning upsert (or already emitted)
            }
            s.emitted = true;
            let mut item = m.into_stored();
            item.write_seq = seq;
            apply_deltas(&mut item, &s.patches);
            if predicate_deleted(&item) {
                continue;
            }
            item.semantic_out.clear();
            resolver.emit(item)?;
        }
    }

    // ---- Phase 2: build each memory_type's generation from its spill ----
    let nonce = mlake_core::MemoryId::new_v4().as_uuid().simple().to_string();
    let mut indexes: BTreeMap<u8, mlake_core::manifest::FactTypeIndex> = BTreeMap::new();
    let mut doc_count = 0usize;

    let Resolver { mut spills, mut samples, .. } = resolver;
    let fts = std::mem::take(&mut spills);
    for (ft, spill) in fts {
        let sample = samples.remove(&ft).unwrap_or_default();
        let n = spill.len();
        doc_count += n;
        let fti =
            build_type_streaming(ns, generation, &nonce, ft, spill, sample, n, tokenizer, opts.seed)
                .await?;
        indexes.insert(ft, fti);
    }

    // ---- Publish (same CAS swap as the in-RAM fold) ----
    let mut next = manifest.clone();
    next.generation = generation;
    next.wal_index_cursor = head;
    next.wal_head = head;
    next.prev_wal_index_cursor = manifest.wal_index_cursor;
    next.prev_generation = Some(manifest.generation);
    next.tokenizer_config_hash = tokenizer.config_hash();
    next.indexes = indexes;

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

    Ok(IndexOutcome { generation, doc_count, published })
}

#[allow(clippy::too_many_arguments)]
async fn build_type_streaming(
    ns: &Namespace,
    generation: u64,
    nonce: &str,
    memory_type: u8,
    spill: ItemSpill,
    sample: Vec<Vec<f32>>,
    n: usize,
    tokenizer: &Tokenizer,
    seed: u64,
) -> Result<mlake_core::manifest::FactTypeIndex> {
    let prefix = format!("{}/mt{memory_type}/gen-{generation}-{nonce}", ns.name);
    let k = mlake_ivf::centroid_count(n);
    let mut centroids = mlake_ivf::train_centroids_k(&sample, k, seed);
    drop(sample);
    let kk = centroids.len().max(1);

    // One big sort carries full item bytes grouped by cluster (avoids √N open cluster files);
    // the rest carry small SSTable fragments. Budget split: the item + payload sorts hold the
    // large values, so they get the lion's share.
    let big = SORT_BUDGET_BYTES * 2 / 5;
    let small = SORT_BUDGET_BYTES / 10;
    let mut cluster_sort = ExternalSort::new(big);
    let mut payload_sort = ExternalSort::new(big);
    let mut pk_sort = ExternalSort::new(small);
    let mut entity_sort = ExternalSort::new(small);
    let mut time_sort = ExternalSort::new(small);
    let mut fts = TantivyFtsBuilder::new(tokenizer.clone()).map_err(|e| crate::Error::Fts(e.to_string()))?;
    let mut radj_pairs: Vec<(MemoryId, InEdge)> = Vec::new();
    let mut sizes = vec![0usize; kk];
    let mut cluster_tags: Vec<(BTreeSet<String>, bool)> = vec![(BTreeSet::new(), false); kk];

    // Single assignment pass over the spill.
    for item in spill.into_reader()? {
        let c = if centroids.is_empty() { 0 } else { centroids.assign(&item.vector) };
        sizes[c] += 1;
        cluster_sort.add(cluster_key(c), item.to_rkyv_bytes())?;
        pk_sort.add(item.id.0, (c as u32).to_le_bytes().to_vec())?;
        payload_sort.add(item.id.0, item.to_payload_bytes())?;
        for e in &item.entity_ids {
            entity_sort.add(e.0, item.id.0.to_vec())?;
        }
        let t = &item.timestamps;
        if let Some(ts) = t.occurred_start.or(t.mentioned_at).or(t.occurred_end) {
            time_sort.add(ts_key(ts), item.id.0.to_vec())?;
        }
        fts.add(item.id, &item.text, &item.tags).map_err(|e| crate::Error::Fts(e.to_string()))?;
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
        let (tset, unt) = &mut cluster_tags[c];
        if item.tags.is_empty() {
            *unt = true;
        } else {
            for tag in &item.tags {
                tset.insert(tag.clone());
            }
        }
    }
    centroids.sizes = sizes;

    // Write cluster files from the cluster-grouped stream.
    let mut cluster_paths: Vec<String> = vec![String::new(); kk];
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
                        cluster_paths[cc] = flush_cluster(ns, &prefix, cc, std::mem::take(&mut cur_items)).await?;
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
                    cluster_paths[cc] = flush_cluster(ns, &prefix, cc, std::mem::take(&mut cur_items)).await?;
                }
                break;
            }
        }
    }
    // Empty clusters still need a (empty) file so a query can address them by index.
    for (c, path) in cluster_paths.iter_mut().enumerate() {
        if path.is_empty() {
            *path = flush_cluster(ns, &prefix, c, Vec::new()).await?;
        }
    }

    // Build the SSTables from the sorted merges, and the sparse radj in RAM.
    let pk = build_sstable_from_merge(pk_sort.finish()?)?;
    let payload = build_sstable_from_merge(payload_sort.finish()?)?;
    let entity = build_sstable_from_merge(entity_sort.finish()?)?;
    let time = build_sstable_from_merge(time_sort.finish()?)?;
    let radj = RadjTable::build(radj_pairs);
    let fts_split = fts.finish().map_err(|e| crate::Error::Fts(e.to_string()))?.split_bytes().to_vec();
    let tag_summary: TagSummary = cluster_tags
        .into_iter()
        .map(|(tags, has_untagged)| ClusterTagSummary { tags: tags.into_iter().collect(), has_untagged })
        .collect();

    let files = write_generation(
        &ns.store,
        &prefix,
        &centroids,
        cluster_paths,
        &fts_split,
        radj.into(),
        pk.into(),
        entity.into(),
        time.into(),
        payload.into(),
        &tag_summary,
        n,
    )
    .await?;

    Ok(mlake_core::manifest::FactTypeIndex { prev_files: None, train_count: n as u64, files })
}

async fn flush_cluster(
    ns: &Namespace,
    prefix: &str,
    c: usize,
    items: Vec<StoredMemory>,
) -> Result<String> {
    let cf = ClusterFile { centroid_id: c as u32, items };
    write_cluster_file(&ns.store, prefix, c, &cf).await
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
