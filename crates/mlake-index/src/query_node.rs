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

use mlake_core::{EntityId, MemoryId, StoredMemory, TagFilter};
use mlake_fts::{TantivyFts, Tokenizer};
use mlake_graph::radj::InEdge;
use mlake_graph::{GraphParams, GraphSource};
use mlake_ivf::{exact_search, Centroids};
use mlake_store::{Phase, QueryMetrics};
use mlake_wal::{Namespace, WalTail};

use crate::fusion::{rrf, weighted_rrf, FusedHit, RankedArm};
use crate::generation::{read_fts_split, TagSummary};
use crate::sstable::{PkTable, RadjTable};
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
    /// The stored memory, materialized server-side (already fetched to score the candidate).
    pub memory: Option<StoredMemory>,
}

impl RawHit {
    fn new(id: MemoryId) -> Self {
        Self { id, dense: None, text: None, graph: None, memory: None }
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

/// Consistency level for a read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Consistency {
    /// Reflect every acked write: check the WAL head and scan the tail (default).
    Strong,
    /// Serve from the cached manifest without a head check; staleness bounded by the
    /// indexing interval.
    Eventual,
}

/// One fact type's loaded state: the indexed generation metadata plus its tail overlay.
struct FactTypeState {
    centroids: Centroids,
    cluster_paths: Vec<String>,
    tag_summary: TagSummary,
    gen_fts: TantivyFts,
    radj: RadjTable,
    pk: PkTable,
    tail_items: Vec<StoredMemory>,
    tail_fts: TantivyFts,
    doc_count: usize,
}

/// A loaded, queryable snapshot of a bank namespace across its fact types.
pub struct QueryNode {
    ns: Namespace,
    per_type: BTreeMap<u8, FactTypeState>,
    tombstones: HashSet<MemoryId>,
    /// The WAL sequence this snapshot reflects.
    pub through_seq: u64,
    /// Roundtrips consumed opening this snapshot (loading the metadata), for the budget.
    pub load_roundtrips: usize,
}

impl QueryNode {
    /// Open a snapshot of a bank: read the manifest, load each fact type's metadata, scan
    /// and partition the tail. Cluster and pk/radj data blocks are not fetched here.
    pub async fn open(
        ns: &Namespace,
        tokenizer: Tokenizer,
        consistency: Consistency,
    ) -> Result<Self> {
        let metrics = QueryMetrics::new();

        // RT1: manifest (+ WAL head, for strong consistency).
        let (manifest, _etag) = ns.read_manifest().await?;
        let head = if consistency == Consistency::Strong {
            ns.wal_head().await?
        } else {
            manifest.wal_head
        };

        // RT4: the WAL tail (exhaustive overlay), partitioned by fact type.
        let scan = WalTail::new(ns)
            .scan(manifest.wal_index_cursor, Some(head))
            .await?;
        let tombstones: HashSet<MemoryId> = scan.tombstones.iter().copied().collect();
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
                    let fts_f = read_fts_split(&ns.store, files, tokenizer.clone(), Some(&metrics));
                    let (centroids_bytes, tag_bytes, radj_idx, pk_idx, gen_fts) =
                        futures::try_join!(centroids_f, tag_f, radj_f, pk_f, fts_f)?;

                    let tag_summary: TagSummary = if tag_bytes.is_empty() {
                        Vec::new()
                    } else {
                        serde_json::from_slice(&tag_bytes).unwrap_or_default()
                    };
                    let pk = PkTable::open(&pk_idx, files.pk_data.clone())?;

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
                        tag_summary,
                        gen_fts,
                        radj: RadjTable::open(&radj_idx, files.radj_csr.clone())?,
                        pk,
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
                        tag_summary: Vec::new(),
                        gen_fts: TantivyFts::build(
                            std::iter::empty::<(MemoryId, &str)>(),
                            tokenizer.clone(),
                        )?,
                        radj: RadjTable::open(&[0u8; 16], String::new())?,
                        pk: PkTable::open(&[0u8; 16], String::new())?,
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
            through_seq: head,
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
            .query_raw_metered(memory_type, vector, text, tags, depths, &metrics)
            .await?;
        Ok(fuse_raw(&raw, top_k, config))
    }

    /// Answer a query for one memory_type, returning each candidate with its **raw per-arm
    /// scores** (dense cosine, BM25, graph activation) and per-arm ranks — no fusion. The
    /// client fuses (RRF or any weighting) with the raw signals. `*_top_k` bound each arm's
    /// candidate depth; a `*_top_k` of 0 skips that arm. Empty list if the type is unknown.
    pub async fn query_raw_metered(
        &self,
        memory_type: u8,
        vector: Option<&[f32]>,
        text: Option<&str>,
        tags: &TagFilter,
        depths: ArmDepths,
        metrics: &QueryMetrics,
    ) -> Result<Vec<RawHit>> {
        let Some(state) = self.per_type.get(&memory_type) else {
            return Ok(Vec::new());
        };
        let (vector_scored, fts_scored, graph_scored, mut materialized) =
            self.run_arms(state, vector, text, tags, depths, metrics).await?;

        // Merge the three arms' candidates by id, recording each arm's rank + raw score.
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

        // Materialize any hit not already in hand — FTS-only hits that fell outside the
        // probed clusters. One coalesced pk lookup + cluster fetch covers them all; the
        // graph arm's candidates are usually cache-warm from its own fetch.
        let missing: Vec<MemoryId> =
            by_id.keys().filter(|id| !materialized.contains_key(id)).copied().collect();
        if !missing.is_empty() {
            let clusters_map = state.pk.lookup_batch(&self.ns.store, &missing, Some((metrics, 3))).await?;
            let clusters: std::collections::HashSet<usize> =
                clusters_map.values().map(|c| *c as usize).collect();
            if !clusters.is_empty() {
                let cids: Vec<usize> = clusters.into_iter().collect();
                for item in self.fetch_clusters(state, &cids, metrics, 3).await? {
                    if by_id.contains_key(&item.id) {
                        materialized.entry(item.id).or_insert(item);
                    }
                }
            }
        }
        metrics.check_budget(&self.ns.name, "query");

        // Attach the materialized memory to each hit (returned inline — no second round trip).
        for (id, hit) in by_id.iter_mut() {
            hit.memory = materialized.remove(id);
        }
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
                let t = Instant::now();
                let mut items = self.fetch_clusters(state, &probed, metrics, 3).await?;
                metrics.record_phase(Phase::FetchClusters, t.elapsed());
                items.extend(state.tail_items.iter().cloned());
                items.retain(|m| tags.matches(&m.tags));
                let t = Instant::now();
                let scored: Vec<(MemoryId, f32)> = exact_search(&items, q, depths.vector)
                    .into_iter()
                    .map(|h| (h.id, h.score))
                    .collect();
                metrics.record_phase(Phase::Rerank, t.elapsed());
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

    /// Fetch the items in the given clusters of a fact type — one coalesced roundtrip.
    async fn fetch_clusters(
        &self,
        state: &FactTypeState,
        cluster_ids: &[usize],
        metrics: &QueryMetrics,
        roundtrip: usize,
    ) -> Result<Vec<StoredMemory>> {
        let futures = cluster_ids.iter().filter_map(|&c| {
            state
                .cluster_paths
                .get(c)
                .map(|path| self.ns.store.get_immutable(path, Some((metrics, roundtrip))))
        });
        let blobs = futures::future::try_join_all(futures).await?;
        let mut items = Vec::new();
        for blob in &blobs {
            let cf = mlake_ivf::ClusterFile::from_bytes(blob)?;
            for item in cf.items {
                if !self.tombstones.contains(&item.id) {
                    items.push(item);
                }
            }
        }
        Ok(items)
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

        // Seed incoming edges: one coalesced batch read over radj.csr for all seeds, rather
        // than a ranged GET per seed.
        let tr = Instant::now();
        let seed_ids: Vec<MemoryId> = seeds.iter().map(|s| s.id).collect();
        let incoming = state
            .radj
            .incoming_batch(&self.ns.store, &seed_ids, Some((metrics, 4)))
            .await?;
        metrics.record_phase(Phase::GraphRadj, tr.elapsed());

        let mut wanted: HashSet<MemoryId> = HashSet::new();
        for seed in &seeds {
            for e in &seed.semantic_out {
                wanted.insert(e.target);
            }
            for e in &seed.causal_out {
                wanted.insert(e.target);
            }
            for e in incoming.get(&seed.id).into_iter().flatten() {
                wanted.insert(e.source);
            }
        }
        wanted.retain(|id| !by_id.contains_key(id) && !self.tombstones.contains(id));

        // Resolve all candidates' clusters in one coalesced batch read over pk.data.
        let tp = Instant::now();
        let wanted_vec: Vec<MemoryId> = wanted.into_iter().collect();
        let clusters_map = state
            .pk
            .lookup_batch(&self.ns.store, &wanted_vec, Some((metrics, 4)))
            .await?;
        metrics.record_phase(Phase::GraphPk, tp.elapsed());
        let clusters_needed: HashSet<usize> = clusters_map.values().map(|c| *c as usize).collect();
        if !clusters_needed.is_empty() {
            let tf = Instant::now();
            let ids: Vec<usize> = clusters_needed.into_iter().collect();
            for item in self.fetch_clusters(state, &ids, metrics, 4).await? {
                by_id.entry(item.id).or_insert(item);
            }
            metrics.record_phase(Phase::GraphFetch, tf.elapsed());
        }

        let te = Instant::now();
        let mut entity_index: HashMap<EntityId, Vec<MemoryId>> = HashMap::new();
        for item in by_id.values() {
            for e in &item.entity_ids {
                entity_index.entry(*e).or_default().push(item.id);
            }
        }

        let source = LazyGraphSource {
            by_id: &by_id,
            entity_index: &entity_index,
            incoming: &incoming,
            tombstones: &self.tombstones,
        };
        let ranked = mlake_graph::retrieve(
            &source,
            &seeds,
            GraphParams { budget: depth, ..GraphParams::default() },
        );
        // Filter graph candidates by the tag filter using their materialized tags. Keep the
        // raw activation score so the client can fuse it however it likes.
        let out = ranked
            .into_iter()
            .filter(|r| by_id.get(&r.id).map(|m| tags.matches(&m.tags)).unwrap_or(false))
            .map(|r| (r.id, r.activation))
            .collect();
        metrics.record_phase(Phase::GraphExpand, te.elapsed());
        Ok(out)
    }
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

struct LazyGraphSource<'a> {
    by_id: &'a HashMap<MemoryId, StoredMemory>,
    entity_index: &'a HashMap<EntityId, Vec<MemoryId>>,
    incoming: &'a HashMap<MemoryId, Vec<InEdge>>,
    tombstones: &'a HashSet<MemoryId>,
}

impl GraphSource for LazyGraphSource<'_> {
    fn entity_candidates(&self, entity_id: EntityId, memory_type: Option<u8>, cap: usize) -> Vec<MemoryId> {
        self.entity_index
            .get(&entity_id)
            .into_iter()
            .flatten()
            .filter(|id| !self.tombstones.contains(id))
            .filter(|id| match (memory_type, self.by_id.get(id)) {
                (Some(ft), Some(item)) => item.memory_type == ft,
                _ => true,
            })
            .take(cap)
            .copied()
            .collect()
    }

    fn item(&self, id: &MemoryId) -> Option<StoredMemory> {
        if self.tombstones.contains(id) {
            return None;
        }
        self.by_id.get(id).cloned()
    }

    fn incoming(&self, target: &MemoryId) -> Vec<InEdge> {
        self.incoming.get(target).cloned().unwrap_or_default()
    }
}
