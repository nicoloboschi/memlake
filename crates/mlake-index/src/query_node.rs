//! The stateless query node (SPEC §6), lazy per-probe, per-fact-type.
//!
//! A bank namespace holds several fully-independent fact-type indexes behind one WAL and
//! one manifest. `open` reads that single manifest (RT1) and loads each fact type's small
//! hot metadata — centroids, FTS split, pk/radj sparse indexes — in parallel; a query for a
//! given fact type then ranged-fetches only the clusters it probes and the exact pk/radj
//! blocks it needs. Query cost scales with `nprobe`, not the corpus (INV-7), and results
//! are returned per fact type (fact types share nothing, so they are never fused).
//!
//! The un-indexed WAL tail is a small overlay, scanned exhaustively (SPEC §6.1),
//! partitioned by fact type, and merged over each type's indexed arms — keeping an acked
//! write visible immediately (INV-5).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use mlake_core::{EntityId, MemoryId, Predicate, StoredMemory, TagFilter};
use mlake_fts::{TantivyFts, Tokenizer};
use mlake_graph::radj::InEdge;
use mlake_graph::{GraphParams, GraphSource};
use mlake_ivf::{exact_search, Centroids};
use mlake_store::{Phase, QueryMetrics};
use mlake_wal::{Namespace, WalTail};

use crate::fusion::{rrf, weighted_rrf, FusedHit, RankedArm};
use crate::generation::{read_fts_split, TagSummary};
use crate::sstable::{EntityTable, PayloadTable, PkTable, RadjTable, TimeTable};
use crate::{QueryConfig, Result};

/// One arm's contribution to a hit: its 0-based rank within that arm and its raw score
/// (dense cosine similarity, BM25 score, or graph activation).
#[derive(Clone, Copy, Debug)]
pub struct ArmScore {
    pub rank: u32,
    pub score: f32,
}

/// A query candidate carrying the **raw** signal from each arm that surfaced it (an arm that
/// did not is `None`) plus the materialized `memory`. memlake does no fusion — the client
/// combines the arm signals (RRF, weighting, re-ranking) however it likes, and gets the
/// stored memory inline so recall needs no second round trip to hydrate.
#[derive(Clone, Debug)]
pub struct RawHit {
    pub id: MemoryId,
    pub dense: Option<ArmScore>,
    pub text: Option<ArmScore>,
    pub graph: Option<ArmScore>,
    /// The temporal arm: entry points in the query's time window + their one-hop spread,
    /// scored by proximity to the window centre. `score` is the temporal score in [0, 1+].
    pub temporal: Option<ArmScore>,
    /// The stored memory, materialized server-side (already fetched to score the candidate).
    pub memory: Option<StoredMemory>,
}

impl RawHit {
    fn new(id: MemoryId) -> Self {
        Self { id, dense: None, text: None, graph: None, temporal: None, memory: None }
    }
}

/// Per-arm candidate depths for a query, plus the IVF probe width. A depth of 0 disables
/// that arm.
#[derive(Clone, Copy, Debug)]
pub struct ArmDepths {
    pub vector: usize,
    pub text: usize,
    pub graph: usize,
    pub nprobe: usize,
}

/// One fact type's loaded state: the indexed generation metadata plus its tail overlay.
struct FactTypeState {
    centroids: Centroids,
    cluster_paths: Vec<String>,
    vector_paths: Vec<String>,
    tag_summary: TagSummary,
    gen_fts: TantivyFts,
    radj: RadjTable,
    pk: PkTable,
    entity: EntityTable,
    time: TimeTable,
    payload: PayloadTable,
    tail_items: Vec<StoredMemory>,
    tail_fts: TantivyFts,
    doc_count: usize,
}

/// A loaded, queryable snapshot of a bank namespace across its fact types.
pub struct QueryNode {
    ns: Namespace,
    per_type: BTreeMap<u8, FactTypeState>,
    tombstones: HashSet<MemoryId>,
    /// Active predicate deletes from the tail: `(sequence, predicate)`. A generation memory
    /// is hidden if it matches one whose sequence exceeds the memory's `write_seq`. Evaluated
    /// lazily at read; materialized at the next fold.
    predicate_tombstones: Vec<(u64, mlake_core::Predicate)>,
    /// The WAL sequence this snapshot reflects.
    pub through_seq: u64,
    /// The generation this snapshot's indexed files belong to. Part of the snapshot's
    /// identity: a `ScanCursor` is only meaningful against the generation that issued it,
    /// since cluster paths and ordering change when the indexer publishes a new one.
    pub generation: u64,
    /// Roundtrips consumed opening this snapshot (loading the metadata), for the budget.
    pub load_roundtrips: usize,
}

/// One fact type's IVF layout, borrowed from the open snapshot: the trained centroids and
/// what k-means put in each. Everything here was already in memory.
pub struct ClusterLayout<'a> {
    pub dim: usize,
    pub centroids: &'a [Vec<f32>],
    /// Members assigned per centroid at train time, parallel to `centroids`.
    pub sizes: &'a [usize],
    /// Per-cluster tag summaries, parallel to `centroids` when present. May be shorter (or
    /// empty) for a generation built before tag summaries existed.
    pub tag_summary: &'a TagSummary,
}

/// A position in a [`QueryNode::scan`]: which cluster of a fact type, and how far into it.
/// The un-indexed WAL tail is walked as one virtual cluster just past the real ones, so a
/// scan covers exactly what a query can see.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanCursor {
    pub cluster: usize,
    pub offset: usize,
}

impl QueryNode {
    /// Open a snapshot of a bank: read the manifest, load each fact type's metadata, scan
    /// and partition the tail. Cluster and pk/radj data blocks are not fetched here.
    ///
    /// A read is always strongly consistent: it discovers the live WAL head and scans the
    /// tail past the manifest cursor, so every acked write is reflected even before the
    /// indexer folds it.
    pub async fn open(ns: &Namespace, tokenizer: Tokenizer) -> Result<Self> {
        let metrics = QueryMetrics::new();

        // RT1: manifest, then the live WAL head (the read's consistency point).
        let (manifest, _etag) = ns.read_manifest().await?;
        let head = ns.wal_head().await?;

        // RT4: the WAL tail (exhaustive overlay), partitioned by fact type.
        let scan = WalTail::new(ns)
            .scan(manifest.wal_index_cursor, Some(head))
            .await?;
        let tombstones: HashSet<MemoryId> = scan.tombstones.iter().copied().collect();
        let predicate_tombstones = scan.predicate_tombstones.clone();
        let mut tail_by_ft: BTreeMap<u8, Vec<StoredMemory>> = BTreeMap::new();
        for item in scan.upserts.into_values() {
            tail_by_ft.entry(item.memory_type).or_default().push(item);
        }

        // Fact types to load: those with an index, plus any that appear only in the tail.
        let mut memory_types: HashSet<u8> = manifest.memory_types().collect();
        memory_types.extend(tail_by_ft.keys().copied());

        let mut per_type = BTreeMap::new();
        for ft in memory_types {
            let tail_items = tail_by_ft.remove(&ft).unwrap_or_default();
            let tail_fts = TantivyFts::build_with_tags(
                tail_items.iter().map(|i| (i.id, i.text.as_str(), i.tags.as_slice())),
                tokenizer.clone(),
            )?;

            let state = match manifest.index(ft) {
                Some(fti) => {
                    let files = &fti.files;
                    // The metadata objects (centroids, tag summary, radj/pk sparse indexes, FTS
                    // split) are independent immutable reads, so fetch them in one concurrent
                    // wave instead of five sequential roundtrips. This is the cost the snapshot
                    // cache re-pays whenever a write invalidates it, so it is worth collapsing.
                    let centroids_f = async {
                        ns.store
                            .get_immutable(&files.centroids, Some((&metrics, 2)))
                            .await
                            .map_err(crate::Error::from)
                    };
                    let tag_f = async {
                        if files.tag_summary.is_empty() {
                            Ok(bytes::Bytes::new())
                        } else {
                            ns.store
                                .get_immutable(&files.tag_summary, Some((&metrics, 2)))
                                .await
                                .map_err(crate::Error::from)
                        }
                    };
                    let radj_f = async {
                        ns.store
                            .get_immutable(&files.radj_idx, Some((&metrics, 2)))
                            .await
                            .map_err(crate::Error::from)
                    };
                    let pk_f = async {
                        ns.store
                            .get_immutable(&files.pk, Some((&metrics, 2)))
                            .await
                            .map_err(crate::Error::from)
                    };
                    let entity_f = async {
                        if files.entity_idx.is_empty() {
                            Ok(bytes::Bytes::new())
                        } else {
                            ns.store
                                .get_immutable(&files.entity_idx, Some((&metrics, 2)))
                                .await
                                .map_err(crate::Error::from)
                        }
                    };
                    let time_f = async {
                        if files.time_idx.is_empty() {
                            Ok(bytes::Bytes::new())
                        } else {
                            ns.store
                                .get_immutable(&files.time_idx, Some((&metrics, 2)))
                                .await
                                .map_err(crate::Error::from)
                        }
                    };
                    let payload_f = async {
                        if files.payload_idx.is_empty() {
                            Ok(bytes::Bytes::new())
                        } else {
                            ns.store
                                .get_immutable(&files.payload_idx, Some((&metrics, 2)))
                                .await
                                .map_err(crate::Error::from)
                        }
                    };
                    let fts_f = read_fts_split(&ns.store, files, tokenizer.clone(), Some(&metrics));
                    let (centroids_bytes, tag_bytes, radj_idx, pk_idx, entity_idx, time_idx, payload_idx, gen_fts) =
                        futures::try_join!(centroids_f, tag_f, radj_f, pk_f, entity_f, time_f, payload_f, fts_f)?;

                    let tag_summary: TagSummary = if tag_bytes.is_empty() {
                        Vec::new()
                    } else {
                        serde_json::from_slice(&tag_bytes).unwrap_or_default()
                    };
                    let pk = PkTable::open(&pk_idx, files.pk_data.clone())?;
                    // Old generations have no entity/time index (back-compat): treat as empty.
                    let entity = if entity_idx.is_empty() {
                        EntityTable::open(&[0u8; 16], String::new())?
                    } else {
                        EntityTable::open(&entity_idx, files.entity_data.clone())?
                    };
                    let time = if time_idx.is_empty() {
                        TimeTable::open(&[0u8; 16], String::new())?
                    } else {
                        TimeTable::open(&time_idx, files.time_data.clone())?
                    };
                    // Old generations have no payload store: an empty table forces the fallback
                    // (materialize from clusters), so a v2 generation still reads correctly.
                    let payload = if payload_idx.is_empty() {
                        PayloadTable::open(&[0u8; 16], String::new())?
                    } else {
                        PayloadTable::open(&payload_idx, files.payload_data.clone())?
                    };

                    // Live doc count for this fact type: its pk record count minus tombstones
                    // that hit it, plus genuinely-new tail items.
                    let mut doc_count = pk.record_count() as usize;
                    for t in &tombstones {
                        if pk.lookup(&ns.store, t, None).await?.is_some() {
                            doc_count -= 1;
                        }
                    }
                    for it in &tail_items {
                        if pk.lookup(&ns.store, &it.id, None).await?.is_none() {
                            doc_count += 1;
                        }
                    }

                    FactTypeState {
                        centroids: Centroids::from_bytes(&centroids_bytes)?,
                        cluster_paths: files.clusters.clone(),
                        vector_paths: files.vectors.clone(),
                        tag_summary,
                        gen_fts,
                        radj: RadjTable::open(&radj_idx, files.radj_csr.clone())?,
                        pk,
                        entity,
                        time,
                        payload,
                        doc_count,
                        tail_items,
                        tail_fts,
                    }
                }
                None => {
                    // Fact type present only in the tail (never indexed yet).
                    let doc_count = tail_items.len();
                    FactTypeState {
                        centroids: Centroids::default(),
                        cluster_paths: Vec::new(),
                        vector_paths: Vec::new(),
                        tag_summary: Vec::new(),
                        gen_fts: TantivyFts::build(
                            std::iter::empty::<(MemoryId, &str)>(),
                            tokenizer.clone(),
                        )?,
                        radj: RadjTable::open(&[0u8; 16], String::new())?,
                        pk: PkTable::open(&[0u8; 16], String::new())?,
                        entity: EntityTable::open(&[0u8; 16], String::new())?,
                        time: TimeTable::open(&[0u8; 16], String::new())?,
                        payload: PayloadTable::open(&[0u8; 16], String::new())?,
                        doc_count,
                        tail_items,
                        tail_fts,
                    }
                }
            };
            per_type.insert(ft, state);
        }

        Ok(Self {
            ns: ns.clone(),
            per_type,
            tombstones,
            predicate_tombstones,
            through_seq: head,
            generation: manifest.generation,
            load_roundtrips: metrics.roundtrips(),
        })
    }

    /// Total live items across all fact types.
    pub fn doc_count(&self) -> usize {
        self.per_type.values().map(|s| s.doc_count).sum()
    }

    /// Live items of one fact type.
    pub fn doc_count_of(&self, memory_type: u8) -> usize {
        self.per_type.get(&memory_type).map(|s| s.doc_count).unwrap_or(0)
    }

    /// Fact types this snapshot can answer.
    pub fn memory_types(&self) -> Vec<u8> {
        self.per_type.keys().copied().collect()
    }

    /// Cluster files backing one fact type's current generation. Zero for a type that
    /// exists only in the un-indexed tail.
    pub fn cluster_count_of(&self, memory_type: u8) -> usize {
        self.per_type.get(&memory_type).map(|s| s.cluster_paths.len()).unwrap_or(0)
    }

    /// How k-means partitioned one fact type: each cluster's centroid, trained size, and
    /// tag summary. `None` for a type this snapshot does not hold.
    ///
    /// This costs no object-storage read at all — the centroids and tag summaries are
    /// already resident, loaded once when the snapshot opened, because every query probes
    /// against them.
    ///
    /// The sizes are the *trained* sizes: they count what the generation was built from and
    /// so exclude un-indexed WAL-tail writes, which is why they can disagree with
    /// [`QueryNode::doc_count_of`].
    pub fn cluster_layout(&self, memory_type: u8) -> Option<ClusterLayout<'_>> {
        let state = self.per_type.get(&memory_type)?;
        Some(ClusterLayout {
            dim: state.centroids.dim,
            centroids: &state.centroids.vectors,
            sizes: &state.centroids.sizes,
            tag_summary: &state.tag_summary,
        })
    }

    /// A bounded sample of member memories with the cluster each belongs to, for plotting
    /// the layout rather than reading it exhaustively.
    ///
    /// Clusters are sampled at an even stride across the whole layout, and the number of
    /// cluster files read is capped, so the cost stays bounded no matter how large the
    /// corpus or how big a `budget` the caller asks for.
    pub async fn sample_members(
        &self,
        memory_type: u8,
        budget: usize,
    ) -> Result<Vec<(u32, StoredMemory)>> {
        let Some(state) = self.per_type.get(&memory_type) else {
            return Ok(Vec::new());
        };
        let total = state.cluster_paths.len();
        if budget == 0 || total == 0 {
            return Ok(Vec::new());
        }

        /// Cap on cluster files read for one sample — this is a visualization, and the
        /// point is to show the shape of the layout, not to pay for the whole corpus.
        const MAX_CLUSTER_READS: usize = 64;
        let reads = total.min(MAX_CLUSTER_READS).min(budget);
        // Even stride so the sample spans the layout instead of clumping at cluster 0.
        let picks: Vec<usize> = (0..reads).map(|i| i * total / reads).collect();
        let per_cluster = (budget / reads).max(1);

        let metrics = QueryMetrics::new();
        let fetched = futures::future::try_join_all(picks.into_iter().map(|c| {
            let metrics = &metrics;
            async move {
                let blob = self.ns.store.get_immutable(&state.cluster_paths[c], Some((metrics, 3))).await?;
                Ok::<_, crate::Error>((c, mlake_ivf::ClusterFile::from_bytes(&blob)?))
            }
        }))
        .await?;

        let mut out = Vec::new();
        for (c, cf) in fetched {
            for item in cf.items.into_iter().take(per_cluster) {
                if !self.tombstones.contains(&item.id) {
                    out.push((c as u32, item));
                }
            }
        }
        Ok(out)
    }

    /// The embedding dimension a fact type's index was built with, if it has one.
    ///
    /// The centroids are authoritative — they are what the query is probed against. A type
    /// present only in the un-indexed tail has none yet, so its first embedded memory
    /// stands in. `None` means nothing in this type carries an embedding, so there is no
    /// dimension to violate.
    fn expected_dim(state: &FactTypeState) -> Option<usize> {
        if !state.centroids.is_empty() && state.centroids.dim > 0 {
            return Some(state.centroids.dim);
        }
        state.tail_items.iter().map(|m| m.vector.len()).find(|&n| n > 0)
    }

    /// Fetch memories by id, without ranking anything. Each id is resolved through its fact
    /// type's pk SSTable to a cluster, then the distinct clusters are read in one coalesced
    /// wave — so the cost is bounded by the number of *clusters* touched, not the corpus.
    ///
    /// The tail overlay is consulted first and wins: it is strictly newer than the indexed
    /// generation. Tombstoned and unknown ids are simply absent from the result — this is a
    /// lookup, not an existence assertion.
    pub async fn get_many(&self, ids: &[MemoryId], include_vector: bool) -> Result<Vec<StoredMemory>> {
        let metrics = QueryMetrics::new();
        let wanted: HashSet<MemoryId> =
            ids.iter().copied().filter(|id| !self.tombstones.contains(id)).collect();
        if wanted.is_empty() {
            return Ok(Vec::new());
        }

        let mut found: HashMap<MemoryId, StoredMemory> = HashMap::new();
        for state in self.per_type.values() {
            for item in &state.tail_items {
                if wanted.contains(&item.id) {
                    found.insert(item.id, item.clone());
                }
            }
        }

        // Anything the tail did not answer must come from an indexed generation. Each fact
        // type is a separate index, so an id is resolved against every type's index.
        for state in self.per_type.values() {
            let missing: Vec<MemoryId> =
                wanted.iter().copied().filter(|id| !found.contains_key(id)).collect();
            if missing.is_empty() {
                break;
            }
            if !include_vector {
                // Fast path: the caller does not want the embedding, so read each memory's row
                // from the payload store (one coalesced ranged GET) instead of deserializing
                // its whole cluster file. Returned memories carry an empty `vector`.
                for (id, item) in
                    state.payload.lookup_batch(&self.ns.store, &missing, Some((&metrics, 1))).await?
                {
                    if wanted.contains(&id) {
                        found.entry(id).or_insert(item);
                    }
                }
                continue;
            }
            let by_cluster = state.pk.lookup_batch(&self.ns.store, &missing, Some((&metrics, 1))).await?;
            if by_cluster.is_empty() {
                continue;
            }
            let mut clusters: Vec<usize> = by_cluster.values().map(|&c| c as usize).collect();
            clusters.sort_unstable();
            clusters.dedup();
            for item in self.fetch_clusters(state, &clusters, &metrics, 2).await? {
                if wanted.contains(&item.id) {
                    found.entry(item.id).or_insert(item);
                }
            }
        }

        // Return in the caller's requested order, skipping ids that did not resolve or that a
        // query would hide. The `wanted` set already dropped id-tombstoned ids up front (so
        // their clusters were never fetched); this additionally drops predicate-tombstoned
        // memories, which can only be judged once the `StoredMemory` (its `write_seq`) is in
        // hand — so a `get` never returns something `query`/`scan` would hide.
        Ok(ids
            .iter()
            .filter_map(|id| found.get(id).cloned())
            .filter(|m| !self.hidden(m))
            .collect())
    }

    /// Page through one fact type's stored memories in cluster order, resuming from
    /// `cursor`. Returns the page and the cursor to pass next, or `None` when exhausted.
    ///
    /// This is a full scan by construction — unlike every other read path here, its total
    /// cost DOES grow with the corpus. It exists for browsing and debugging; retrieval uses
    /// [`QueryNode::query`]. One cluster file is read per step, so a single page costs at
    /// most `limit`-bounded work, but walking the whole type reads the whole type.
    /// Live memory count per entity, across `types` (empty = every type).
    ///
    /// Reads the entity posting index rather than the corpus, then applies the same
    /// adjustment `Stats.doc_count` makes: memories hidden by a tombstone are subtracted,
    /// and entities carried by un-indexed tail writes are added. Entities with no live
    /// memories are omitted, so an id the caller asked about and does not get back is an
    /// orphan.
    pub async fn entity_counts(
        &self,
        types: &[u8],
        entities: Option<&[EntityId]>,
    ) -> Result<HashMap<EntityId, u64>> {
        let metrics = QueryMetrics::new();
        let mut out: HashMap<EntityId, u64> = HashMap::new();
        let wanted: Option<HashSet<EntityId>> = entities.map(|e| e.iter().copied().collect());

        let walk: Vec<u8> = if types.is_empty() { self.memory_types() } else { types.to_vec() };
        for ty in walk {
            let Some(state) = self.per_type.get(&ty) else { continue };

            if !state.entity.is_empty() {
                for (entity, count) in state
                    .entity
                    .counts(&self.ns.store, entities, Some((&metrics, 3)))
                    .await?
                {
                    *out.entry(entity).or_insert(0) += count;
                }
            }

            // The posting index is built from the indexed generation, so correct it for
            // what a query would actually see: tail upserts add, hidden memories subtract.
            for m in &state.tail_items {
                if self.hidden(m) {
                    continue;
                }
                for e in &m.entity_ids {
                    if wanted.as_ref().is_none_or(|w| w.contains(e)) {
                        *out.entry(*e).or_insert(0) += 1;
                    }
                }
            }
        }

        out.retain(|_, n| *n > 0);
        Ok(out)
    }

    pub async fn scan(
        &self,
        memory_type: u8,
        cursor: ScanCursor,
        limit: usize,
        filter: &Predicate,
    ) -> Result<(Vec<StoredMemory>, Option<ScanCursor>)> {
        let Some(state) = self.per_type.get(&memory_type) else {
            return Ok((Vec::new(), None));
        };
        let metrics = QueryMetrics::new();
        // The virtual tail cluster sits just past the real ones, so one walk covers both the
        // indexed generation and the un-indexed overlay a query would also see.
        let tail_cluster = state.cluster_paths.len();
        // An id can sit in both a cluster and the tail — a re-upsert of an already-indexed
        // memory. The tail version is newer and wins, exactly as it does for a query, so the
        // indexed copy is skipped and each memory is yielded once.
        let superseded: HashSet<MemoryId> = state.tail_items.iter().map(|m| m.id).collect();
        let (mut cluster, mut offset) = (cursor.cluster, cursor.offset);
        let mut out = Vec::new();

        while out.len() < limit && cluster <= tail_cluster {
            let items: Vec<StoredMemory> = if cluster == tail_cluster {
                state
                    .tail_items
                    .iter()
                    .filter(|m| !self.hidden(m))
                    .cloned()
                    .collect()
            } else {
                // Hide anything a query would hide: a tail upsert supersedes the indexed copy,
                // and a tail id-tombstone or predicate-tombstone hides it entirely. Without the
                // `hidden` check a deleted-but-still-indexed memory would surface in a scan.
                self.fetch_clusters(state, &[cluster], &metrics, 3).await?
                    .into_iter()
                    .filter(|m| !superseded.contains(&m.id) && !self.hidden(m))
                    .collect()
            };
            // Filtering is deterministic for a fixed generation, so `offset` indexes the
            // filtered list stably across calls — the cursor stays valid between pages.
            let matching: Vec<StoredMemory> =
                items.into_iter().filter(|m| filter.matches(m)).collect();

            let take = (limit - out.len()).min(matching.len().saturating_sub(offset));
            out.extend(matching.iter().skip(offset).take(take).cloned());
            offset += take;
            if offset >= matching.len() {
                cluster += 1;
                offset = 0;
            }
        }

        let next = (cluster <= tail_cluster).then_some(ScanCursor { cluster, offset });
        Ok((out, next))
    }

    /// Ids of every live memory — across `memory_types` (or all if empty) — whose metadata
    /// contains all of `metadata_equals` (AND semantics, e.g. `document_id` + `chunk_id`) and
    /// whose tags pass `tags`. Used by delete-by-predicate.
    ///
    /// A full scan by construction: metadata is not indexed, so this reads every cluster of
    /// every selected type once. Cost grows with the corpus — acceptable for a maintenance
    /// operation, not a query path.
    pub async fn ids_matching(
        &self,
        memory_types: &[u8],
        metadata_equals: &[(String, String)],
        tags: &TagFilter,
    ) -> Result<Vec<MemoryId>> {
        let types: Vec<u8> =
            if memory_types.is_empty() { self.memory_types() } else { memory_types.to_vec() };
        let metrics = QueryMetrics::new();
        let mut ids = Vec::new();
        for mt in types {
            let Some(state) = self.per_type.get(&mt) else { continue };
            // Tail overrides the indexed copy of a re-upserted id (newer wins), same as a scan.
            let superseded: HashSet<MemoryId> = state.tail_items.iter().map(|m| m.id).collect();
            let tail_cluster = state.cluster_paths.len();
            for cluster in 0..=tail_cluster {
                let items: Vec<StoredMemory> = if cluster == tail_cluster {
                    state
                        .tail_items
                        .iter()
                        .filter(|m| !self.tombstones.contains(&m.id))
                        .cloned()
                        .collect()
                } else {
                    self.fetch_clusters(state, &[cluster], &metrics, 3)
                        .await?
                        .into_iter()
                        .filter(|m| !superseded.contains(&m.id))
                        .collect()
                };
                for m in &items {
                    if tags.matches(&m.tags) && metadata_contains_all(&m.metadata, metadata_equals) {
                        ids.push(m.id);
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Convenience: run the query and fuse the arms with (weighted) RRF into a single ranked
    /// list. The gRPC server does NOT use this — it returns the raw per-arm scores and lets
    /// the client fuse — but it keeps a simple, self-contained retrieval API for Rust callers
    /// and tests. `config.arm_depth` bounds every arm; `graph_weight == 0` drops the graph arm.
    pub async fn query(
        &self,
        memory_type: u8,
        vector: Option<&[f32]>,
        text: Option<&str>,
        tags: &TagFilter,
        top_k: usize,
        config: QueryConfig,
    ) -> Result<Vec<FusedHit>> {
        let metrics = QueryMetrics::new();
        let depths = ArmDepths {
            vector: config.arm_depth,
            text: config.arm_depth,
            graph: if config.graph_weight > 0.0 { config.arm_depth } else { 0 },
            nprobe: config.nprobe,
        };
        let raw = self
            .query_raw_metered(memory_type, vector, text, tags, depths, None, &metrics)
            .await?;
        Ok(fuse_raw(&raw, top_k, config))
    }

    /// Answer a query for one memory_type, returning each candidate with its **raw per-arm
    /// scores** (dense cosine, BM25, graph activation) and per-arm ranks — no fusion. The
    /// client fuses (RRF or any weighting) with the raw signals. `*_top_k` bound each arm's
    /// candidate depth; a `*_top_k` of 0 skips that arm. Empty list if the type is unknown.
    #[allow(clippy::too_many_arguments)]
    pub async fn query_raw_metered(
        &self,
        memory_type: u8,
        vector: Option<&[f32]>,
        text: Option<&str>,
        tags: &TagFilter,
        depths: ArmDepths,
        temporal_window: Option<(i64, i64)>,
        metrics: &QueryMetrics,
    ) -> Result<Vec<RawHit>> {
        let Some(state) = self.per_type.get(&memory_type) else {
            return Ok(Vec::new());
        };
        // Reject a query vector that does not match this type's embedding dimension before
        // any arm runs. Every vector-consuming arm would otherwise compare incomparable
        // embeddings, and the failure would surface as a plausible ranking rather than an
        // error — the caller silently gets nonsense for a wrong or stale embedding model.
        if let (Some(q), Some(dim)) = (vector, Self::expected_dim(state)) {
            if !q.is_empty() && q.len() != dim {
                return Err(mlake_core::Error::DimMismatch { expected: dim, got: q.len() }.into());
            }
        }
        let (vector_scored, fts_scored, graph_scored, mut materialized) =
            self.run_arms(state, vector, text, tags, depths, metrics).await?;

        // The temporal arm: entry-point selection over the time window + one-hop spread. Needs
        // the query vector (to rank entry points) and a window; scores by proximity, ranked
        // desc so the client sees the strongest-in-window first.
        let temporal_scored: Vec<(MemoryId, f32)> = match (vector, temporal_window) {
            (Some(q), Some((from, to))) => {
                let mut scored = self.temporal_arm(state, q, from, to, tags, &mut materialized, metrics).await?;
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
                });
                scored
            }
            _ => Vec::new(),
        };

        // Merge the arms' candidates by id, recording each arm's rank + raw score.
        let mut by_id: HashMap<MemoryId, RawHit> = HashMap::new();
        let mut fill = |scored: Vec<(MemoryId, f32)>, set: fn(&mut RawHit, ArmScore)| {
            for (rank, (id, score)) in scored.into_iter().enumerate() {
                let hit = by_id.entry(id).or_insert_with(|| RawHit::new(id));
                set(hit, ArmScore { rank: rank as u32, score });
            }
        };
        fill(vector_scored, |h, s| h.dense = Some(s));
        fill(fts_scored, |h, s| h.text = Some(s));
        fill(graph_scored, |h, s| h.graph = Some(s));
        fill(temporal_scored, |h, s| h.temporal = Some(s));

        // The generation's FTS index still contains memories tombstoned in the tail (the
        // tantivy split predates the delete), so an arm can surface a tombstoned id. Drop them
        // — a deleted memory must never appear in results, regardless of which arm found it.
        by_id.retain(|id, _| !self.tombstones.contains(id));

        // Materialize any hit not already in hand — FTS/graph hits that fell outside the
        // probed clusters. One coalesced ranged GET over the payload store hydrates them all
        // by id, instead of deserializing a whole cluster file per hit (the old cost that made
        // an FTS query ~40ms warm at 1M). The payload has no embedding, which query hits never
        // return anyway. Every v3 generation has the payload store (older ones are rejected at
        // the manifest read), so an indexed hit always resolves here.
        let missing: Vec<MemoryId> =
            by_id.keys().filter(|id| !materialized.contains_key(id)).copied().collect();
        if !missing.is_empty() {
            for (id, item) in
                state.payload.lookup_batch(&self.ns.store, &missing, Some((metrics, 3))).await?
            {
                materialized.entry(id).or_insert(item);
            }
        }
        metrics.check_budget(&self.ns.name, "query");

        // Attach the materialized memory to each hit (returned inline — no second round trip).
        for (id, hit) in by_id.iter_mut() {
            hit.memory = materialized.remove(id);
        }
        // A hit with no materialized memory is one the FTS split still indexes but that is now
        // tombstoned or predicate-deleted (the split predates the delete); `fetch_clusters`
        // filtered it out, so drop it — a deleted memory must never surface via any arm.
        by_id.retain(|_, hit| hit.memory.is_some());
        Ok(by_id.into_values().collect())
    }

    /// Run the three arms for one memory_type, returning each arm's ranked candidates with
    /// their raw scores `(dense, fts, graph)` plus the memories materialized while doing so
    /// (the probed clusters + tail, keyed by id). Shared by every query path. The vector and
    /// graph arms share the probed clusters (one fetch); the graph arm seeds off the dense
    /// ranking. An arm with `top_k == 0`, or a missing input, yields an empty list.
    #[allow(clippy::type_complexity)]
    async fn run_arms(
        &self,
        state: &FactTypeState,
        vector: Option<&[f32]>,
        text: Option<&str>,
        tags: &TagFilter,
        depths: ArmDepths,
        metrics: &QueryMetrics,
    ) -> Result<(
        Vec<(MemoryId, f32)>,
        Vec<(MemoryId, f32)>,
        Vec<(MemoryId, f32)>,
        HashMap<MemoryId, StoredMemory>,
    )> {
        // Vector arm: probe + fetch the candidate clusters once (RT3); the graph arm reuses
        // the materialized memories. Tags are applied inline (memories carry their tags).
        let (probed_items, vector_scored) = match vector {
            Some(q) if depths.vector > 0 && !state.centroids.is_empty() => {
                let t = Instant::now();
                let probed = self.select_clusters(state, q, depths.nprobe, tags);
                metrics.record_phase(Phase::Probe, t.elapsed());

                // Scan: vector blocks only. Tags are a column in the block, so the filter is
                // applied exactly and *before* scoring — no oversampling, no post-filter, and
                // the payload half is never read for the members that lose.
                let t = Instant::now();
                let blocks = self.fetch_vector_blocks(state, &probed, metrics, 3).await?;
                metrics.record_phase(Phase::FetchClusters, t.elapsed());

                let t = Instant::now();
                let mut scored: Vec<(MemoryId, f32)> = Vec::new();
                for block in &blocks {
                    let mask = block.tag_mask(&tags.tags);
                    // A block whose dictionary cannot satisfy the filter is skipped whole —
                    // e.g. an ALL filter naming a tag the cluster does not contain.
                    if !tags.is_noop() && block.has_tags() && !block.any_can_pass(&mask, tags.mode)
                    {
                        continue;
                    }
                    let prepared = block.prepare(q)?;
                    for i in 0..block.len() {
                        let id = block.ids()[i];
                        if self.tombstones.contains(&id) {
                            continue;
                        }
                        if !tags.is_noop() && block.has_tags() && !block.passes(i, &mask, tags.mode)
                        {
                            continue;
                        }
                        scored.push((id, block.score(&prepared, i)));
                    }
                }
                // The un-indexed tail has no block; it is small and already resident.
                for m in &state.tail_items {
                    if tags.matches(&m.tags) && !self.hidden(m) {
                        scored.push((m.id, mlake_core::cosine_opt(q, &m.vector)));
                    }
                }
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
                });
                scored.truncate(depths.vector);
                metrics.record_phase(Phase::Rerank, t.elapsed());

                // Hydrate only the survivors. This is what the graph arm seeds from, and what
                // is returned inline — bounded by the arm depth, not by cluster size.
                let mut materialized: HashMap<MemoryId, StoredMemory> = HashMap::new();
                let winners: Vec<MemoryId> = scored.iter().map(|(id, _)| *id).collect();
                // Tail winners are already resident and are NOT in the pk index — the fold
                // has not seen them yet — so seed them directly. Leaving them to
                // `materialize_into` would silently drop every un-indexed write from the
                // results, which is exactly the visibility INV-5 promises.
                let want: HashSet<MemoryId> = winners.iter().copied().collect();
                for m in &state.tail_items {
                    if want.contains(&m.id) {
                        materialized.insert(m.id, m.clone());
                    }
                }
                self.materialize_into(state, &winners, &mut materialized, metrics).await?;
                let items: Vec<StoredMemory> = materialized.into_values().collect();
                (items, scored)
            }
            _ => {
                let mut items = state.tail_items.clone();
                items.retain(|m| tags.matches(&m.tags));
                (items, Vec::new())
            }
        };

        let tf = Instant::now();
        let fts_scored = text
            .filter(|t| !t.is_empty() && depths.text > 0)
            .map(|t| self.fts_arm(state, t, depths.text, tags))
            .unwrap_or_default();
        metrics.record_phase(Phase::Fts, tf.elapsed());

        // The graph arm needs dense seeds; it does ranged pk/radj reads, so it is skipped
        // when disabled (top_k 0) or when there is nothing to seed from.
        let graph_scored = if depths.graph > 0 && !vector_scored.is_empty() {
            let seed_ids: Vec<MemoryId> = vector_scored.iter().map(|(id, _)| *id).collect();
            self.graph_arm(state, &seed_ids, &probed_items, depths.graph, tags, metrics)
                .await?
        } else {
            Vec::new()
        };

        // The probed items (vector clusters + tail) are the base set of materialized memories
        // the caller can return inline; anything else is fetched on demand by query_raw_metered.
        let materialized: HashMap<MemoryId, StoredMemory> =
            probed_items.into_iter().map(|m| (m.id, m)).collect();
        Ok((vector_scored, fts_scored, graph_scored, materialized))
    }

    /// Choose which clusters to fetch for the vector arm.
    ///
    /// Without a tag filter (or without per-cluster tag summaries) this is the plain
    /// `nprobe`-nearest probe. With a filter, the per-cluster tag summaries prune clusters
    /// that cannot contain a matching memory, and the query probes among the *admissible*
    /// clusters — so a selective filter still finds its matches instead of them being
    /// starved out of the nprobe-nearest set (SCALE.md Phase 4b). Because a selective filter
    /// leaves few admissible clusters, fetching all of them (capped) stays within budget;
    /// a broad filter admits ~everything, degrading to the plain probe.
    fn select_clusters(
        &self,
        state: &FactTypeState,
        query: &[f32],
        nprobe: usize,
        tags: &TagFilter,
    ) -> Vec<usize> {
        if tags.is_noop() || state.tag_summary.is_empty() {
            return state.centroids.probe(query, nprobe);
        }

        // Admissible clusters: those whose tag summary could contain a matching memory.
        let admissible: Vec<usize> = (0..state.centroids.len())
            .filter(|&c| {
                state
                    .tag_summary
                    .get(c)
                    .map(|s| tags.cluster_admits(&s.tags, s.has_untagged))
                    .unwrap_or(true)
            })
            .collect();

        // Rank the admissible clusters by centroid distance and take enough to cover the
        // matches. Cap at a small multiple of nprobe so a broad filter can't blow the byte
        // budget; a selective filter's admissible set is already small.
        let cap = nprobe.saturating_mul(4).max(nprobe);
        if admissible.len() <= cap {
            return admissible;
        }
        let ranked = state.centroids.probe(query, state.centroids.len());
        ranked
            .into_iter()
            .filter(|c| {
                state
                    .tag_summary
                    .get(*c)
                    .map(|s| tags.cluster_admits(&s.tags, s.has_untagged))
                    .unwrap_or(true)
            })
            .take(cap)
            .collect()
    }

    /// Whether a memory is deleted — by id (tombstone) or by a tail predicate delete that
    /// post-dates its last write. Evaluated on the full record (predicates read metadata).
    fn hidden(&self, m: &StoredMemory) -> bool {
        self.tombstones.contains(&m.id)
            || self
                .predicate_tombstones
                .iter()
                .any(|(seq, p)| m.write_seq < *seq && p.matches(m))
    }

    /// Fetch the items in the given clusters of a fact type.
    ///
    /// Fan-out is bounded: a wide probe (large `nprobe`, or a broad tag filter that admits many
    /// clusters) would otherwise issue one GET per cluster all at once and hold every raw cluster
    /// blob in memory simultaneously — a spike proportional to `nprobe × cluster_size`. Instead at
    /// most `CLUSTER_FETCH_FANOUT` blobs are in flight; each is decoded and its live items kept as
    /// it arrives, so the transient buffering is capped by the fan-out, not the probe width. (The
    /// returned live set is still the whole probe — this bounds the fetch spike, not the rerank
    /// set; per-query rerank size is governed by `nprobe`.) Cross-cluster order is not preserved,
    /// which no caller depends on.
    /// Fetch just the vector blocks for `cluster_ids` — the scan path.
    ///
    /// This is the read the split exists for: the probe scores every member of every probed
    /// cluster, but wants the payload of almost none, and the embedding was 84% of a stored
    /// memory. Tag filtering rides along in the block, so the scan never needs the payload
    /// half to decide what is admissible.
    async fn fetch_vector_blocks(
        &self,
        state: &FactTypeState,
        cluster_ids: &[usize],
        metrics: &QueryMetrics,
        roundtrip: usize,
    ) -> Result<Vec<mlake_ivf::VectorBlock>> {
        use futures::stream::{StreamExt, TryStreamExt};
        const BLOCK_FETCH_FANOUT: usize = 8;

        let paths: Vec<String> =
            cluster_ids.iter().filter_map(|&c| state.vector_paths.get(c).cloned()).collect();
        let store = &self.ns.store;
        futures::stream::iter(paths)
            .map(|path| async move {
                let blob = store.get_immutable(&path, Some((metrics, roundtrip))).await?;
                Ok::<_, crate::Error>(mlake_ivf::VectorBlock::from_bytes(&blob)?)
            })
            .buffer_unordered(BLOCK_FETCH_FANOUT)
            .try_collect()
            .await
    }

    async fn fetch_clusters(
        &self,
        state: &FactTypeState,
        cluster_ids: &[usize],
        metrics: &QueryMetrics,
        roundtrip: usize,
    ) -> Result<Vec<StoredMemory>> {
        use futures::stream::{StreamExt, TryStreamExt};

        /// Max cluster blobs fetched + resident at once within one `fetch_clusters` call.
        const CLUSTER_FETCH_FANOUT: usize = 8;

        // Own the paths so each fetch future captures no borrow with a higher-ranked lifetime
        // (a `&String` stream item would not satisfy `buffer_unordered`'s Send bound).
        // Each cluster is a pair: `.bin` holds the payload, `.vec` the embeddings, joined
        // positionally. Both ride the same roundtrip wave, so splitting them cost no extra
        // latency — the embedding simply stopped being carried through every byte of text.
        let paths: Vec<(String, String)> = cluster_ids
            .iter()
            .filter_map(|&c| {
                Some((state.cluster_paths.get(c)?.clone(), state.vector_paths.get(c)?.clone()))
            })
            .collect();
        let store = &self.ns.store;
        let per_cluster: Vec<Vec<StoredMemory>> = futures::stream::iter(paths)
            .map(|(path, vec_path)| async move {
                let (blob, vblob) = futures::try_join!(
                    store.get_immutable(&path, Some((metrics, roundtrip))),
                    store.get_immutable(&vec_path, Some((metrics, roundtrip))),
                )?;
                let mut items = mlake_ivf::ClusterFile::from_bytes(&blob)?.items;
                let block = mlake_ivf::VectorBlock::from_bytes(&vblob)?;
                if block.len() != items.len() {
                    return Err(crate::Error::Core(mlake_core::Error::Decode(format!(
                        "cluster {path} holds {} members but its vector block holds {}",
                        items.len(),
                        block.len()
                    ))));
                }
                for (j, item) in items.iter_mut().enumerate() {
                    item.vector = block.decode(j);
                }
                Ok::<_, crate::Error>(items)
            })
            .buffer_unordered(CLUSTER_FETCH_FANOUT)
            .try_collect()
            .await?;
        // Drop tombstoned + predicate-deleted memories so no arm can surface them.
        Ok(per_cluster.into_iter().flatten().filter(|item| !self.hidden(item)).collect())
    }

    /// Materialize `ids` (those not already present) into `into`: one coalesced pk lookup +
    /// cluster fetch. Tombstoned/absent ids resolve to nothing.
    async fn materialize_into(
        &self,
        state: &FactTypeState,
        ids: &[MemoryId],
        into: &mut HashMap<MemoryId, StoredMemory>,
        metrics: &QueryMetrics,
    ) -> Result<()> {
        let missing: Vec<MemoryId> = ids
            .iter()
            .filter(|id| !into.contains_key(id) && !self.tombstones.contains(id))
            .copied()
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        // Hydrate from the payload store: one coalesced ranged read of just these rows.
        // Going via pk to whole cluster files would pull every member of every cluster a
        // winner happens to live in — the read the vector/payload split exists to avoid.
        if !state.payload.is_empty() {
            let rows = state.payload.lookup_batch(&self.ns.store, &missing, Some((metrics, 4))).await?;
            for (id, item) in rows {
                if !self.hidden(&item) {
                    into.entry(id).or_insert(item);
                }
            }
            return Ok(());
        }

        // No payload store (a generation built before it existed): fall back to the cluster
        // files, which still carry every field except the embedding.
        let clusters_map = state.pk.lookup_batch(&self.ns.store, &missing, Some((metrics, 4))).await?;
        let clusters: HashSet<usize> = clusters_map.values().map(|c| *c as usize).collect();
        if clusters.is_empty() {
            return Ok(());
        }
        let want: HashSet<MemoryId> = missing.into_iter().collect();
        let cids: Vec<usize> = clusters.into_iter().collect();
        for item in self.fetch_clusters(state, &cids, metrics, 4).await? {
            if want.contains(&item.id) {
                into.entry(item.id).or_insert(item);
            }
        }
        Ok(())
    }

    /// The temporal arm (SPEC-less; a 1:1 port of Hindsight's `retrieve_temporal_combined`
    /// with the BFS bounded to one hop per INV-7). Select entry points in the query's time
    /// window (one ranged scan of the time index), rank them by similarity, spread them across
    /// the window with `select_with_temporal_coverage`, score by proximity to the window
    /// centre, then spread one hop through links (semantic + causal). Returns `(id, score)`.
    async fn temporal_arm(
        &self,
        state: &FactTypeState,
        query: &[f32],
        from: i64,
        to: i64,
        tags: &TagFilter,
        materialized: &mut HashMap<MemoryId, StoredMemory>,
        metrics: &QueryMetrics,
    ) -> Result<Vec<(MemoryId, f32)>> {
        use crate::temporal as tmp;
        let eff = |m: &StoredMemory| {
            m.timestamps.occurred_start.or(m.timestamps.mentioned_at).or(m.timestamps.occurred_end)
        };
        let prox = |m: &StoredMemory, default: f32| {
            tmp::best_date(m.timestamps.occurred_start, m.timestamps.occurred_end, m.timestamps.mentioned_at)
                .map(|d| tmp::temporal_proximity(d, from, to))
                .unwrap_or(default)
        };

        // 1. Entry-point pool: ids whose effective_ts is in the window (one ranged scan) plus
        //    in-window tail items.
        let mut window_ids = if state.time.is_empty() {
            Vec::new()
        } else {
            state
                .time
                .in_window(&self.ns.store, from, to, tmp::TEMPORAL_WINDOW_CAP, Some((metrics, 4)))
                .await?
        };
        for m in &state.tail_items {
            if eff(m).is_some_and(|ts| ts >= from && ts <= to) {
                window_ids.push(m.id);
            }
        }
        window_ids.retain(|id| !self.tombstones.contains(id));
        if window_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.materialize_into(state, &window_ids, materialized, metrics).await?;

        // 2. Similarity-ranked, tag-filtered pool -> coverage-spread entry points.
        let mut pool: Vec<tmp::Candidate> = window_ids
            .iter()
            .filter_map(|id| materialized.get(id).map(|m| (id, m)))
            .filter(|(_, m)| tags.matches(&m.tags))
            .map(|(id, m)| tmp::Candidate {
                id: *id,
                similarity: mlake_core::cosine_opt(query, &m.vector),
                effective_ts: eff(m),
            })
            .collect();
        pool.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal).then(a.id.cmp(&b.id))
        });
        pool.truncate(tmp::TEMPORAL_POOL_SIZE);
        let entry_points =
            tmp::select_with_temporal_coverage(pool, from, to, tmp::TEMPORAL_ENTRY_POINTS, tmp::TEMPORAL_COVERAGE_BUCKETS);

        // 3. Score entry points; they seed the spread with parent propagation score 1.0.
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        let mut seeds: Vec<MemoryId> = Vec::new();
        for ep in &entry_points {
            if let Some(m) = materialized.get(&ep.id) {
                scores.insert(ep.id, prox(m, tmp::NO_DATE_ENTRY));
                seeds.push(ep.id);
            }
        }
        if seeds.is_empty() {
            return Ok(scores.into_iter().collect());
        }

        // 4. One hop through links: seeds' inline outgoing (semantic + causal) + radj incoming.
        let incoming = state.radj.incoming_batch(&self.ns.store, &seeds, Some((metrics, 4))).await?;
        // (neighbor, weight, boost)
        let mut links: Vec<(MemoryId, f32, f32)> = Vec::new();
        for seed in &seeds {
            if let Some(m) = materialized.get(seed) {
                for e in &m.semantic_out {
                    links.push((e.target, e.weight.to_f32(), 1.0));
                }
                for e in &m.causal_out {
                    links.push((e.target, e.weight.to_f32(), causal_boost_of(e.link_type)));
                }
            }
            for e in incoming.get(seed).into_iter().flatten() {
                let boost = match e.kind {
                    mlake_graph::radj::EdgeKind::Semantic => 1.0,
                    mlake_graph::radj::EdgeKind::Causal(t) => causal_boost_tag(t),
                };
                links.push((e.source, e.weight, boost));
            }
        }
        let neighbor_ids: Vec<MemoryId> = links.iter().map(|(id, _, _)| *id).collect();
        self.materialize_into(state, &neighbor_ids, materialized, metrics).await?;
        for (nid, weight, boost) in links {
            if scores.contains_key(&nid) || self.tombstones.contains(&nid) {
                continue; // an entry point, or already scored via another seed's max below
            }
            let Some(m) = materialized.get(&nid) else { continue };
            if !tags.matches(&m.tags) {
                continue;
            }
            // Parent propagation score is 1.0 for entry-point seeds (one hop).
            let combined = tmp::propagate(prox(m, tmp::NO_DATE_NEIGHBOR), 1.0, weight, boost);
            let e = scores.entry(nid).or_insert(0.0);
            *e = e.max(combined);
        }

        Ok(scores.into_iter().collect())
    }

    /// The FTS arm's ranked hits with their raw BM25 scores.
    fn fts_arm(&self, state: &FactTypeState, text: &str, depth: usize, tags: &TagFilter) -> Vec<(MemoryId, f32)> {
        let mut hits = state.gen_fts.search_filtered(text, depth, tags);
        hits.extend(
            state
                .tail_fts
                .search_filtered(text, depth, tags)
                .into_iter()
                .filter(|h| !self.tombstones.contains(&h.id)),
        );
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        hits.dedup_by_key(|h| h.id);
        hits.truncate(depth);
        hits.into_iter().map(|h| (h.id, h.score)).collect()
    }

    #[allow(clippy::too_many_arguments)]
    async fn graph_arm(
        &self,
        state: &FactTypeState,
        vector_ranking: &[MemoryId],
        probed_items: &[StoredMemory],
        depth: usize,
        tags: &TagFilter,
        metrics: &QueryMetrics,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let mut by_id: HashMap<MemoryId, StoredMemory> =
            probed_items.iter().map(|i| (i.id, i.clone())).collect();

        let seeds: Vec<StoredMemory> = vector_ranking
            .iter()
            .take(20)
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        // Two coalesced reads, issued together (same wave): the seeds' incoming edges over
        // radj.csr, and the entity postings — the memories sharing each seed entity, from the
        // persisted entity index. The postings are what make the entity arm real: it finds
        // sharers anywhere in the corpus, not just in the clusters the vector arm probed.
        let tr = Instant::now();
        let seed_ids: Vec<MemoryId> = seeds.iter().map(|s| s.id).collect();
        let mut seed_entities: HashSet<EntityId> = HashSet::new();
        for seed in &seeds {
            seed_entities.extend(seed.entity_ids.iter().copied());
        }
        let seed_entities: Vec<EntityId> = seed_entities.into_iter().collect();
        let per_entity_cap = GraphParams::default().per_entity_cap;
        let (incoming, entity_candidates) = futures::try_join!(
            state.radj.incoming_batch(&self.ns.store, &seed_ids, Some((metrics, 4))),
            state.entity.candidates_batch(&self.ns.store, &seed_entities, per_entity_cap, Some((metrics, 4))),
        )?;
        metrics.record_phase(Phase::GraphRadj, tr.elapsed());

        // Score structurally — NO candidate hydration. Activation comes entirely from the
        // graph structure already in hand: seed edges, incoming edges (radj), and how many
        // seed-entity postings each candidate appears in (its shared-entity count). Existence
        // is just "not tombstoned": semantic/causal targets were liveness-filtered at fold
        // time, so an edge only dangles if its target was deleted since. This is the change
        // that removed the whole-corpus hydration: previously every candidate (thousands, via
        // `per_entity_cap`) was read from its cluster just to be scored and then truncated.
        let te = Instant::now();
        let source = LazyGraphSource {
            entity_index: &entity_candidates,
            incoming: &incoming,
            tombstones: &self.tombstones,
        };
        let ranked = mlake_graph::retrieve(
            &source,
            &seeds,
            GraphParams { budget: depth, ..GraphParams::default() },
        );
        metrics.record_phase(Phase::GraphExpand, te.elapsed());

        // Only the ranked results (≤ budget) are hydrated, and only when a tag filter needs
        // their tags. Without a filter, nothing is fetched here — the final materialization
        // pass hydrates the surviving hits once, shared across all arms.
        if tags.is_noop() {
            return Ok(ranked.into_iter().map(|r| (r.id, r.activation)).collect());
        }
        let tf = Instant::now();
        let ranked_ids: Vec<MemoryId> =
            ranked.iter().map(|r| r.id).filter(|id| !by_id.contains_key(id)).collect();
        for (id, item) in
            state.payload.lookup_batch(&self.ns.store, &ranked_ids, Some((metrics, 4))).await?
        {
            by_id.entry(id).or_insert(item);
        }
        metrics.record_phase(Phase::GraphFetch, tf.elapsed());
        let out = ranked
            .into_iter()
            .filter(|r| by_id.get(&r.id).map(|m| tags.matches(&m.tags)).unwrap_or(false))
            .map(|r| (r.id, r.activation))
            .collect();
        Ok(out)
    }
}

/// Whether `metadata` contains every `(key, value)` in `required` (AND semantics). An empty
/// `required` matches everything.
fn metadata_contains_all(metadata: &[(String, String)], required: &[(String, String)]) -> bool {
    required
        .iter()
        .all(|(k, v)| metadata.iter().any(|(mk, mv)| mk == k && mv == v))
}

/// Spread multiplier for a causal edge's link type (temporal arm).
fn causal_boost_of(lt: mlake_core::LinkType) -> f32 {
    use mlake_core::LinkType::*;
    crate::temporal::causal_boost(matches!(lt, Causes | CausedBy), matches!(lt, Enables | Prevents))
}
fn causal_boost_tag(t: mlake_graph::radj::LinkTypeTag) -> f32 {
    use mlake_graph::radj::LinkTypeTag::*;
    crate::temporal::causal_boost(matches!(t, Causes | CausedBy), matches!(t, Enables | Prevents))
}

/// Reconstruct each arm's ranking from the raw hits and combine with (weighted) RRF. This is
/// the same fusion the client would do; kept here only for [`QueryNode::query`].
fn fuse_raw(raw: &[RawHit], top_k: usize, config: QueryConfig) -> Vec<FusedHit> {
    fn ranking(raw: &[RawHit], arm: impl Fn(&RawHit) -> Option<ArmScore>) -> Vec<MemoryId> {
        let mut v: Vec<&RawHit> = raw.iter().filter(|h| arm(h).is_some()).collect();
        v.sort_by_key(|h| arm(h).unwrap().rank);
        v.into_iter().map(|h| h.id).collect()
    }
    let vector = ranking(raw, |h| h.dense);
    let fts = ranking(raw, |h| h.text);
    let graph = ranking(raw, |h| h.graph);

    let mut arms: Vec<(RankedArm<'_>, f32)> = Vec::new();
    if !vector.is_empty() {
        arms.push((RankedArm { name: "vector", ranking: &vector }, config.vector_weight));
    }
    if !fts.is_empty() {
        arms.push((RankedArm { name: "fts", ranking: &fts }, config.fts_weight));
    }
    if !graph.is_empty() {
        arms.push((RankedArm { name: "graph", ranking: &graph }, config.graph_weight));
    }
    if arms.len() == 1 && (config.vector_weight - config.fts_weight).abs() < f32::EPSILON {
        let only = [RankedArm { name: arms[0].0.name, ranking: arms[0].0.ranking }];
        return rrf(&only, config.rrf_k, top_k);
    }
    weighted_rrf(&arms, config.rrf_k, top_k)
}

/// The graph source for structural scoring — no item hydration. The entity postings are
/// already scoped to one memory_type (the fold builds a separate entity index per type), so
/// candidates need no per-item memory_type check, and liveness is a tombstone test.
struct LazyGraphSource<'a> {
    entity_index: &'a HashMap<EntityId, Vec<MemoryId>>,
    incoming: &'a HashMap<MemoryId, Vec<InEdge>>,
    tombstones: &'a HashSet<MemoryId>,
}

impl GraphSource for LazyGraphSource<'_> {
    fn entity_candidates(&self, entity_id: EntityId, _memory_type: Option<u8>, cap: usize) -> Vec<MemoryId> {
        // The posting is already this memory_type's index, so no per-item type check is needed.
        self.entity_index
            .get(&entity_id)
            .into_iter()
            .flatten()
            .filter(|id| !self.tombstones.contains(id))
            .take(cap)
            .copied()
            .collect()
    }

    fn exists(&self, id: &MemoryId) -> bool {
        !self.tombstones.contains(id)
    }

    fn incoming(&self, target: &MemoryId) -> Vec<InEdge> {
        self.incoming.get(target).cloned().unwrap_or_default()
    }
}
