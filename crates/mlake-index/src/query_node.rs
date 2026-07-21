//! The stateless query node (SPEC §6), lazy per-probe.
//!
//! At 10M items a generation is 25–50 GB, so the node must never load the whole thing.
//! Instead it loads only the small, hot metadata once — the centroid table, the FTS split
//! (materialized to the NVMe/mmap tier), and the graph adjacency — and then, *per query*,
//! ranged-fetches only the clusters the query actually probes. Query cost scales with
//! `nprobe`, not with the corpus (INV-7), and the resident structures are the published
//! artifacts, not a re-trained copy — so the recall served is the recall the indexer built.
//!
//! The un-indexed WAL tail is a small overlay, scanned exhaustively (SPEC §6.1) and merged
//! over the indexed arms, which is what keeps an acked write visible immediately (INV-5).
//!
//! Scope note (SCALE.md, Phase 1): the vector arm is fully lazy. The graph arm materializes
//! over the probed clusters plus the tail; exact cross-cluster materialization via a
//! range-readable `pk.idx` is Phase 2. `radj`/`pk` are loaded whole here and become
//! range-readable in Phase 2.

use std::collections::{HashMap, HashSet};

use mlake_core::{ItemId, StoredItem};
use mlake_fts::{TantivyFts, Tokenizer};
use mlake_graph::radj::{InEdge, ReverseAdjacency};
use mlake_graph::{GraphParams, GraphSource};
use mlake_ivf::{exact_search, Centroids};
use mlake_store::QueryMetrics;
use mlake_wal::{Namespace, WalTail};

use crate::fusion::{rrf, weighted_rrf, FusedHit, RankedArm};
use crate::generation::{read_fts_split, PkIndex};
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
/// cluster bytes are fetched per query and served from the NVMe cache when warm.
pub struct QueryNode {
    ns: Namespace,
    centroids: Centroids,
    /// Cluster file paths, indexed by centroid id (parallel to `centroids.vectors`).
    cluster_paths: Vec<String>,
    gen_fts: TantivyFts,
    radj: ReverseAdjacency,
    /// Item ids present in the generation (from the pk index), for graph presence checks.
    gen_ids: HashSet<ItemId>,
    /// The un-indexed tail: live upserts (patched), and the set of tombstoned ids.
    tail_items: Vec<StoredItem>,
    tail_fts: TantivyFts,
    tombstones: HashSet<ItemId>,
    /// The WAL sequence this snapshot reflects.
    pub through_seq: u64,
    /// Roundtrips consumed opening this snapshot (loading the metadata), for the budget.
    pub load_roundtrips: usize,
}

impl QueryNode {
    /// Open a snapshot: load the small metadata, materialize the FTS split, scan the tail.
    /// Cluster bytes are deliberately *not* fetched here.
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

        // RT2: the small hot metadata. Clusters are not loaded.
        let (centroids, radj, pk, gen_fts) = if manifest.is_empty() {
            (
                Centroids::default(),
                ReverseAdjacency::build(vec![]),
                PkIndex::default(),
                TantivyFts::build(std::iter::empty::<(ItemId, &str)>(), tokenizer.clone())?,
            )
        } else {
            let centroids_bytes = ns
                .store
                .get_immutable(&manifest.files.centroids, Some((&metrics, 2)))
                .await?;
            let radj_bytes = ns
                .store
                .get_immutable(&manifest.files.radj_csr, Some((&metrics, 2)))
                .await?;
            let pk_bytes = ns
                .store
                .get_immutable(&manifest.files.pk, Some((&metrics, 2)))
                .await?;
            let gen_fts =
                read_fts_split(&ns.store, &manifest.files, tokenizer.clone(), Some(&metrics))
                    .await?;
            (
                Centroids::from_bytes(&centroids_bytes)?,
                ReverseAdjacency::from_bytes(&radj_bytes)?,
                serde_json::from_slice::<PkIndex>(&pk_bytes)?,
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

        let gen_ids: HashSet<ItemId> = pk.entries.iter().map(|(id, _)| *id).collect();

        Ok(Self {
            ns: ns.clone(),
            centroids,
            cluster_paths: manifest.files.clusters.clone(),
            gen_fts,
            radj,
            gen_ids,
            tail_items,
            tail_fts,
            tombstones,
            through_seq: head,
            load_roundtrips: metrics.roundtrips(),
        })
    }

    /// Number of live items (generation minus tombstones, plus tail). Cheap — no fetch.
    pub fn doc_count(&self) -> usize {
        let gen_live = self.gen_ids.iter().filter(|id| !self.tombstones.contains(id)).count();
        // Tail upserts may replace generation items; count distinct.
        let tail_new = self
            .tail_items
            .iter()
            .filter(|i| !self.gen_ids.contains(&i.id))
            .count();
        gen_live + tail_new
    }

    /// Fetch the items in the given probed clusters — one coalesced roundtrip regardless of
    /// how many clusters are selected (SPEC §6.1 RT3). Served from the NVMe cache when warm.
    async fn fetch_clusters(
        &self,
        cluster_ids: &[usize],
        metrics: &QueryMetrics,
    ) -> Result<Vec<StoredItem>> {
        let futures = cluster_ids.iter().filter_map(|&c| {
            self.cluster_paths
                .get(c)
                .map(|path| self.ns.store.get_immutable(path, Some((metrics, 3))))
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

    /// Answer a fused query. Vector and graph arms ranged-fetch only the probed clusters;
    /// the FTS arm reads the materialized split.
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

    /// Like [`query`], but records object-storage usage into `metrics` so a caller can
    /// assert the per-query roundtrip and cache behaviour.
    pub async fn query_metered(
        &self,
        vector: Option<&[f32]>,
        text: Option<&str>,
        top_k: usize,
        config: QueryConfig,
        metrics: &QueryMetrics,
    ) -> Result<Vec<FusedHit>> {
        // Probe and fetch the candidate clusters once; both the vector and graph arms use
        // the same materialized set, so a query fetches each probed cluster at most once.
        let (probed_items, vector_ranking) = match vector {
            Some(q) if !self.centroids.is_empty() => {
                let probed = self.centroids.probe(q, config.nprobe);
                let mut items = self.fetch_clusters(&probed, metrics).await?;
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

        // Graph arm: expand from the vector-chosen seeds over the loaded adjacency and the
        // items materialized above.
        let graph_ranking = if self.radj.edge_count() > 0 && !vector_ranking.is_empty() {
            self.graph_arm(&vector_ranking, &probed_items, config.arm_depth)
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
        // Higher score first; ties by id for determinism. Cross-index BM25 scores are only
        // approximately comparable, but the tail is small and fusion is rank-based.
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
    fn graph_arm(&self, vector_ranking: &[ItemId], materialized: &[StoredItem], depth: usize) -> Vec<ItemId> {
        let by_id: HashMap<ItemId, StoredItem> =
            materialized.iter().map(|i| (i.id, i.clone())).collect();
        let mut entity_index: HashMap<u64, Vec<ItemId>> = HashMap::new();
        for item in materialized {
            for e in &item.entity_ids {
                entity_index.entry(*e).or_default().push(item.id);
            }
        }
        let seeds: Vec<StoredItem> = vector_ranking
            .iter()
            .take(20)
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        if seeds.is_empty() {
            return Vec::new();
        }
        let source = LazyGraphSource {
            by_id: &by_id,
            entity_index: &entity_index,
            radj: &self.radj,
            tombstones: &self.tombstones,
        };
        mlake_graph::retrieve(
            &source,
            &seeds,
            GraphParams {
                budget: depth,
                ..GraphParams::default()
            },
        )
        .into_iter()
        .map(|r| r.id)
        .collect()
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

/// A graph source over the per-query materialized item set plus the loaded reverse
/// adjacency. Candidates outside the materialized set are treated as absent (Phase 2 makes
/// cross-cluster materialization exact via a range-readable pk index).
struct LazyGraphSource<'a> {
    by_id: &'a HashMap<ItemId, StoredItem>,
    entity_index: &'a HashMap<u64, Vec<ItemId>>,
    radj: &'a ReverseAdjacency,
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
        self.radj.incoming(target).to_vec()
    }
}


