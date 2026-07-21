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

use mlake_core::{MemoryId, StoredMemory, TagFilter};
use mlake_fts::{TantivyFts, Tokenizer};
use mlake_graph::radj::InEdge;
use mlake_graph::{GraphParams, GraphSource};
use mlake_ivf::{exact_search, Centroids};
use mlake_store::QueryMetrics;
use mlake_wal::{Namespace, WalTail};

use crate::fusion::{rrf, weighted_rrf, FusedHit, RankedArm};
use crate::generation::read_fts_split;
use crate::sstable::{PkTable, RadjTable};
use crate::{QueryConfig, Result};

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
                    let centroids_bytes = ns
                        .store
                        .get_immutable(&files.centroids, Some((&metrics, 2)))
                        .await?;
                    let radj_idx = ns
                        .store
                        .get_immutable(&files.radj_idx, Some((&metrics, 2)))
                        .await?;
                    let pk_idx = ns.store.get_immutable(&files.pk, Some((&metrics, 2))).await?;
                    let gen_fts =
                        read_fts_split(&ns.store, files, tokenizer.clone(), Some(&metrics)).await?;
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

    /// Answer a query for a single fact type. Returns an empty list if the bank has no such
    /// fact type. (Fact types share nothing, so callers query each and keep them grouped.)
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
        self.query_metered(memory_type, vector, text, tags, top_k, config, &metrics).await
    }

    /// Like [`query`], but records object-storage usage into `metrics`.
    pub async fn query_metered(
        &self,
        memory_type: u8,
        vector: Option<&[f32]>,
        text: Option<&str>,
        tags: &TagFilter,
        top_k: usize,
        config: QueryConfig,
        metrics: &QueryMetrics,
    ) -> Result<Vec<FusedHit>> {
        let Some(state) = self.per_type.get(&memory_type) else {
            return Ok(Vec::new());
        };

        // Probe + fetch the candidate clusters once (RT3); vector and graph arms share them.
        // The tag filter is applied inline to the materialized memories — they carry their
        // tags, so filtering is free once fetched (the shared TagFilter primitive).
        let (probed_items, vector_ranking) = match vector {
            Some(q) if !state.centroids.is_empty() => {
                let probed = self.select_clusters(state, q, config.nprobe, tags);
                let mut items = self.fetch_clusters(state, &probed, metrics, 3).await?;
                items.extend(state.tail_items.iter().cloned());
                items.retain(|m| tags.matches(&m.tags));
                let ranking: Vec<MemoryId> = exact_search(&items, q, config.arm_depth)
                    .into_iter()
                    .map(|h| h.id)
                    .collect();
                (items, ranking)
            }
            _ => {
                let mut items = state.tail_items.clone();
                items.retain(|m| tags.matches(&m.tags));
                (items, Vec::new())
            }
        };

        let fts_ranking = text
            .filter(|t| !t.is_empty())
            .map(|t| self.fts_arm(state, t, config.arm_depth, tags))
            .unwrap_or_default();

        let graph_ranking = if !vector_ranking.is_empty() {
            self.graph_arm(state, &vector_ranking, &probed_items, config.arm_depth, tags, metrics)
                .await?
        } else {
            Vec::new()
        };

        metrics.check_budget(&self.ns.name, "query");
        Ok(fuse(&vector_ranking, &fts_ranking, &graph_ranking, top_k, config))
    }

    /// Choose which clusters to fetch for the vector arm.
    ///
    /// Without a tag filter this is the plain `nprobe`-nearest probe. With a selective
    /// filter, 4b will intersect the probe with a tag→cluster posting so the planner only
    /// fetches clusters that can contain a matching memory and can scale nprobe cheaply.
    /// For now (4a correctness) it is the plain probe; the tag filter is applied inline
    /// after fetch.
    fn select_clusters(
        &self,
        state: &FactTypeState,
        query: &[f32],
        nprobe: usize,
        _tags: &TagFilter,
    ) -> Vec<usize> {
        state.centroids.probe(query, nprobe)
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

    fn fts_arm(&self, state: &FactTypeState, text: &str, depth: usize, tags: &TagFilter) -> Vec<MemoryId> {
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
        hits.into_iter().map(|h| h.id).collect()
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
    ) -> Result<Vec<MemoryId>> {
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

        let mut incoming: HashMap<MemoryId, Vec<InEdge>> = HashMap::new();
        for seed in &seeds {
            let edges = state.radj.incoming(&self.ns.store, &seed.id, Some((metrics, 4))).await?;
            incoming.insert(seed.id, edges);
        }

        let mut wanted: HashSet<MemoryId> = HashSet::new();
        for seed in &seeds {
            for e in &seed.semantic_out {
                wanted.insert(e.target);
            }
            for e in &seed.causal_out {
                wanted.insert(e.target);
            }
            for e in &incoming[&seed.id] {
                wanted.insert(e.source);
            }
        }
        wanted.retain(|id| !by_id.contains_key(id) && !self.tombstones.contains(id));

        let mut clusters_needed: HashSet<usize> = HashSet::new();
        for id in &wanted {
            if let Some(c) = state.pk.lookup(&self.ns.store, id, Some((metrics, 4))).await? {
                clusters_needed.insert(c as usize);
            }
        }
        if !clusters_needed.is_empty() {
            let ids: Vec<usize> = clusters_needed.into_iter().collect();
            for item in self.fetch_clusters(state, &ids, metrics, 4).await? {
                by_id.entry(item.id).or_insert(item);
            }
        }

        let mut entity_index: HashMap<u64, Vec<MemoryId>> = HashMap::new();
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
        // Filter graph candidates by the tag filter using their materialized tags.
        Ok(ranked
            .into_iter()
            .filter(|r| by_id.get(&r.id).map(|m| tags.matches(&m.tags)).unwrap_or(false))
            .map(|r| r.id)
            .collect())
    }
}

/// Combine the arm rankings with (weighted) RRF.
fn fuse(
    vector: &[MemoryId],
    fts: &[MemoryId],
    graph: &[MemoryId],
    top_k: usize,
    config: QueryConfig,
) -> Vec<FusedHit> {
    let mut arms: Vec<(RankedArm<'_>, f32)> = Vec::new();
    if !vector.is_empty() {
        arms.push((RankedArm { name: "vector", ranking: vector }, config.vector_weight));
    }
    if !fts.is_empty() {
        arms.push((RankedArm { name: "fts", ranking: fts }, config.fts_weight));
    }
    if !graph.is_empty() {
        arms.push((RankedArm { name: "graph", ranking: graph }, config.graph_weight));
    }
    if arms.len() == 1 && (config.vector_weight - config.fts_weight).abs() < f32::EPSILON {
        let only = [RankedArm { name: arms[0].0.name, ranking: arms[0].0.ranking }];
        return rrf(&only, config.rrf_k, top_k);
    }
    weighted_rrf(&arms, config.rrf_k, top_k)
}

struct LazyGraphSource<'a> {
    by_id: &'a HashMap<MemoryId, StoredMemory>,
    entity_index: &'a HashMap<u64, Vec<MemoryId>>,
    incoming: &'a HashMap<MemoryId, Vec<InEdge>>,
    tombstones: &'a HashSet<MemoryId>,
}

impl GraphSource for LazyGraphSource<'_> {
    fn entity_candidates(&self, entity_id: u64, memory_type: Option<u8>, cap: usize) -> Vec<MemoryId> {
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
