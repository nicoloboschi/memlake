//! The stateless query node (SPEC §6), lazy per-probe.
//!
//! At 10M items a generation is 25–50 GB, so the node never loads the whole thing. It
//! loads only the small, hot metadata once — the centroid table, the FTS split, and the
//! *sparse indexes* of `pk` and `radj` (a few MB even at 10M) — and then, per query,
//! ranged-fetches only the clusters it probes and the exact `pk`/`radj` blocks it needs.
//! Query cost scales with `nprobe` and the number of graph candidates, not with the
//! corpus (INV-7), and the resident structures are the published artifacts, so the recall
//! served is the recall the indexer built.
//!
//! The un-indexed WAL tail is a small overlay, scanned exhaustively (SPEC §6.1) and merged
//! over the indexed arms, keeping an acked write visible immediately (INV-5).
//!
//! Graph materialization is exact across clusters (Phase 2): the seed's incoming edges and
//! the candidates' clusters are range-read from the `radj`/`pk` SSTables, so a neighbour in
//! an unprobed cluster is still found. Full entity expansion still needs entity postings
//! (Phase 4); today it runs over the materialized candidate set.

use std::collections::{HashMap, HashSet};

use mlake_core::{ItemId, StoredItem};
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

/// A loaded, queryable snapshot of a namespace. Holds only small metadata resident;
/// cluster bytes and pk/radj blocks are range-fetched per query, warm from the NVMe cache.
pub struct QueryNode {
    ns: Namespace,
    centroids: Centroids,
    /// Cluster file paths, indexed by centroid id (parallel to `centroids.vectors`).
    cluster_paths: Vec<String>,
    gen_fts: TantivyFts,
    /// Reverse-adjacency SSTable (sparse index resident, blocks range-read).
    radj: RadjTable,
    /// Primary-key SSTable (sparse index resident, blocks range-read).
    pk: PkTable,
    /// The un-indexed tail: live upserts (patched), and the set of tombstoned ids.
    tail_items: Vec<StoredItem>,
    tail_fts: TantivyFts,
    tombstones: HashSet<ItemId>,
    doc_count: usize,
    /// The WAL sequence this snapshot reflects.
    pub through_seq: u64,
    /// Roundtrips consumed opening this snapshot (loading the metadata), for the budget.
    pub load_roundtrips: usize,
}

impl QueryNode {
    /// Open a snapshot: load the small metadata (centroids, FTS split, pk/radj sparse
    /// indexes), scan the tail. Cluster and pk/radj *data* blocks are not fetched here.
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

        // RT2: the small hot metadata. Cluster and pk/radj data blocks are not loaded.
        let (centroids, radj, pk, gen_fts) = if manifest.is_empty() {
            (
                Centroids::default(),
                RadjTable::open(&[0u8; 16], String::new())?,
                PkTable::open(&[0u8; 16], String::new())?,
                TantivyFts::build(std::iter::empty::<(ItemId, &str)>(), tokenizer.clone())?,
            )
        } else {
            let centroids_bytes = ns
                .store
                .get_immutable(&manifest.files.centroids, Some((&metrics, 2)))
                .await?;
            let radj_idx = ns
                .store
                .get_immutable(&manifest.files.radj_idx, Some((&metrics, 2)))
                .await?;
            let pk_idx = ns
                .store
                .get_immutable(&manifest.files.pk, Some((&metrics, 2)))
                .await?;
            let gen_fts =
                read_fts_split(&ns.store, &manifest.files, tokenizer.clone(), Some(&metrics))
                    .await?;
            (
                Centroids::from_bytes(&centroids_bytes)?,
                RadjTable::open(&radj_idx, manifest.files.radj_csr.clone())?,
                PkTable::open(&pk_idx, manifest.files.pk_data.clone())?,
                gen_fts,
            )
        };

        // RT4: the WAL tail (exhaustive scan overlay).
        let scan = WalTail::new(ns)
            .scan(manifest.wal_index_cursor, Some(head))
            .await?;
        let tombstones: HashSet<ItemId> = scan.tombstones.iter().copied().collect();
        let tail_items: Vec<StoredItem> = scan.upserts.into_values().collect();
        let tail_fts = TantivyFts::build(
            tail_items.iter().map(|i| (i.id, i.text.as_str())),
            tokenizer.clone(),
        )?;

        // Live doc count: generation record count, minus tombstones that hit a generation
        // item, plus tail upserts that are genuinely new. The pk lookups here are bounded
        // by the (small) tail, not the corpus.
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

        Ok(Self {
            ns: ns.clone(),
            centroids,
            cluster_paths: manifest.files.clusters.clone(),
            gen_fts,
            radj,
            pk,
            tail_items,
            tail_fts,
            tombstones,
            doc_count,
            through_seq: head,
            load_roundtrips: metrics.roundtrips(),
        })
    }

    /// Live item count. Cheap — computed at open from the pk record count and the tail.
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Fetch the items in the given clusters — one coalesced roundtrip regardless of how
    /// many clusters are selected (SPEC §6.1 RT3). Served from the NVMe cache when warm.
    async fn fetch_clusters(
        &self,
        cluster_ids: &[usize],
        metrics: &QueryMetrics,
        roundtrip: usize,
    ) -> Result<Vec<StoredItem>> {
        let futures = cluster_ids.iter().filter_map(|&c| {
            self.cluster_paths
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

    pub async fn query(
        &self,
        vector: Option<&[f32]>,
        text: Option<&str>,
        top_k: usize,
        config: QueryConfig,
    ) -> Result<Vec<FusedHit>> {
        let metrics = QueryMetrics::new();
        self.query_metered(vector, text, top_k, config, &metrics).await
    }

    /// Like [`query`], but records object-storage usage into `metrics`.
    pub async fn query_metered(
        &self,
        vector: Option<&[f32]>,
        text: Option<&str>,
        top_k: usize,
        config: QueryConfig,
        metrics: &QueryMetrics,
    ) -> Result<Vec<FusedHit>> {
        // Probe and fetch the candidate clusters once (RT3); the vector and graph arms
        // share this materialized set.
        let (probed_items, vector_ranking) = match vector {
            Some(q) if !self.centroids.is_empty() => {
                let probed = self.centroids.probe(q, config.nprobe);
                let mut items = self.fetch_clusters(&probed, metrics, 3).await?;
                items.extend(self.tail_items.iter().cloned()); // exhaustive tail overlay
                let ranking: Vec<ItemId> = exact_search(&items, q, config.arm_depth)
                    .into_iter()
                    .map(|h| h.id)
                    .collect();
                (items, ranking)
            }
            _ => (self.tail_items.clone(), Vec::new()),
        };

        // FTS arm: the generation split plus the tail split, merged by score.
        let fts_ranking = text
            .filter(|t| !t.is_empty())
            .map(|t| self.fts_arm(t, config.arm_depth))
            .unwrap_or_default();

        // Graph arm: expand one hop from the vector-chosen seeds, fetching only the
        // radj blocks and candidate clusters the seeds actually reach (RT4).
        let graph_ranking = if !vector_ranking.is_empty() {
            self.graph_arm(&vector_ranking, &probed_items, config.arm_depth, metrics)
                .await?
        } else {
            Vec::new()
        };

        metrics.check_budget(&self.ns.name, "query");
        Ok(fuse(&vector_ranking, &fts_ranking, &graph_ranking, top_k, config))
    }

    /// The FTS arm: BM25 over the generation split and the tail split, merged by score.
    fn fts_arm(&self, text: &str, depth: usize) -> Vec<ItemId> {
        let mut hits = self.gen_fts.search(text, depth);
        hits.extend(
            self.tail_fts
                .search(text, depth)
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

    /// The graph arm: one-hop link expansion from the vector-chosen seeds.
    ///
    /// Materialization is exact: for each seed we range-read its incoming edges from the
    /// `radj` SSTable, gather the one-hop candidate ids (inline outgoing + incoming
    /// sources), resolve their clusters via the `pk` SSTable, and fetch exactly those
    /// clusters — so a neighbour in an unprobed cluster is still found (fixing the Phase 1
    /// approximation). Entity expansion runs over the materialized set; full entity
    /// postings are Phase 4.
    async fn graph_arm(
        &self,
        vector_ranking: &[ItemId],
        probed_items: &[StoredItem],
        depth: usize,
        metrics: &QueryMetrics,
    ) -> Result<Vec<ItemId>> {
        // Materialized items available so far, by id (probed clusters + tail).
        let mut by_id: HashMap<ItemId, StoredItem> =
            probed_items.iter().map(|i| (i.id, i.clone())).collect();

        let seeds: Vec<StoredItem> = vector_ranking
            .iter()
            .take(20)
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        // Range-read each seed's incoming edges from radj (one block each).
        let mut incoming: HashMap<ItemId, Vec<InEdge>> = HashMap::new();
        for seed in &seeds {
            let edges = self.radj.incoming(&self.ns.store, &seed.id, Some((metrics, 4))).await?;
            incoming.insert(seed.id, edges);
        }

        // The one-hop candidate ids: seeds' inline outgoing targets and their incoming
        // sources. Any not already materialized must be fetched.
        let mut wanted: HashSet<ItemId> = HashSet::new();
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

        // Resolve the wanted ids to clusters via pk, then fetch those clusters (one
        // coalesced roundtrip) and add their items to the materialized set.
        let mut clusters_needed: HashSet<usize> = HashSet::new();
        for id in &wanted {
            if let Some(c) = self.pk.lookup(&self.ns.store, id, Some((metrics, 4))).await? {
                clusters_needed.insert(c as usize);
            }
        }
        if !clusters_needed.is_empty() {
            let ids: Vec<usize> = clusters_needed.into_iter().collect();
            for item in self.fetch_clusters(&ids, metrics, 4).await? {
                by_id.entry(item.id).or_insert(item);
            }
        }

        // Entity index over everything materialized.
        let mut entity_index: HashMap<u64, Vec<ItemId>> = HashMap::new();
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
        Ok(mlake_graph::retrieve(
            &source,
            &seeds,
            GraphParams {
                budget: depth,
                ..GraphParams::default()
            },
        )
        .into_iter()
        .map(|r| r.id)
        .collect())
    }
}

/// Combine the arm rankings with (weighted) RRF.
fn fuse(
    vector: &[ItemId],
    fts: &[ItemId],
    graph: &[ItemId],
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

/// A graph source over the per-query materialized item set, with seed incoming edges
/// pre-fetched from `radj`. `retrieve` only calls `incoming` on seeds (one hop), so the
/// pre-fetched map is complete.
struct LazyGraphSource<'a> {
    by_id: &'a HashMap<ItemId, StoredItem>,
    entity_index: &'a HashMap<u64, Vec<ItemId>>,
    incoming: &'a HashMap<ItemId, Vec<InEdge>>,
    tombstones: &'a HashSet<ItemId>,
}

impl GraphSource for LazyGraphSource<'_> {
    fn entity_candidates(&self, entity_id: u64, fact_type: Option<u8>, cap: usize) -> Vec<ItemId> {
        self.entity_index
            .get(&entity_id)
            .into_iter()
            .flatten()
            .filter(|id| !self.tombstones.contains(id))
            .filter(|id| match (fact_type, self.by_id.get(id)) {
                (Some(ft), Some(item)) => item.fact_type == ft,
                _ => true,
            })
            .take(cap)
            .copied()
            .collect()
    }

    fn item(&self, id: &ItemId) -> Option<StoredItem> {
        if self.tombstones.contains(id) {
            return None;
        }
        self.by_id.get(id).cloned()
    }

    fn incoming(&self, target: &ItemId) -> Vec<InEdge> {
        self.incoming.get(target).cloned().unwrap_or_default()
    }
}
