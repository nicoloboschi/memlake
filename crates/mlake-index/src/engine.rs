//! In-process query engine: the fusion of the three arms behind one `query` call.
//!
//! This holds a built generation in memory and answers fused queries over it. It is the
//! query-planner logic (SPEC §6.3) exercised without the S3 roundtrip machinery, so the
//! retrieval behaviour — arm execution and RRF — can be measured directly against the
//! Qdrant baseline on identical vectors. The storage-backed query node reuses these same
//! fusion and arm calls once it has materialized the generation's files.

use std::collections::HashMap;

use mlake_core::{EntityId, MemoryId, StoredMemory};
use mlake_fts::{TantivyFts, Tokenizer, TokenizerConfig};
use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag, ReverseAdjacency};
use mlake_graph::{GraphParams, GraphSource};
use mlake_ivf::{build_clusters, exact_search, train_centroids, Centroids};

use crate::fusion::{rrf, weighted_rrf, FusedHit, RankedArm, DEFAULT_RRF_K};

/// Which arms to run and how to combine them.
#[derive(Clone, Copy, Debug)]
pub struct QueryConfig {
    pub nprobe: usize,
    pub rrf_k: f32,
    /// Per-arm RRF weights (vector, fts, graph). Equal weights reproduce canonical RRF.
    pub vector_weight: f32,
    pub fts_weight: f32,
    pub graph_weight: f32,
    /// How many candidates each arm contributes to fusion before truncation.
    pub arm_depth: usize,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            nprobe: mlake_ivf::DEFAULT_NPROBE,
            rrf_k: DEFAULT_RRF_K,
            vector_weight: 1.0,
            fts_weight: 1.0,
            graph_weight: 1.0,
            arm_depth: 100,
        }
    }
}

/// A built, queryable generation held in memory.
pub struct Engine {
    centroids: Centroids,
    /// Cluster i's items, parallel to `centroids.vectors`.
    clusters: Vec<Vec<StoredMemory>>,
    /// All items by id, for graph materialization and result hydration.
    items: HashMap<MemoryId, StoredMemory>,
    fts: TantivyFts,
    radj: ReverseAdjacency,
    entity_index: HashMap<EntityId, Vec<MemoryId>>,
}

impl Engine {
    /// Build a generation from a set of items. Mirrors what the indexer does, minus the
    /// S3 write: train centroids, assign clusters, build the FTS index, derive nothing
    /// (semantic links, if any, are already inline on the items).
    pub fn build(items: Vec<StoredMemory>, tokenizer: Tokenizer) -> Self {
        let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
        let centroids = train_centroids(&vectors, 42);
        let clusters: Vec<Vec<StoredMemory>> = build_clusters(items.clone(), &centroids)
            .into_iter()
            .map(|c| c.items)
            .collect();

        // The FTS arm is tantivy: build the index from the same tokens the shared chain
        // produces (SPEC §5.3). Uses the default tokenizer config, matching the query side.
        let fts = TantivyFts::build(
            items.iter().map(|i| (i.id, i.text.as_str())),
            Tokenizer::new(TokenizerConfig::default()),
        )
        .expect("build tantivy FTS index");
        let _ = tokenizer;

        let mut map = HashMap::new();
        let mut entity_index: HashMap<EntityId, Vec<MemoryId>> = HashMap::new();
        let mut radj_pairs: Vec<(MemoryId, InEdge)> = Vec::new();

        for item in &items {
            for e in &item.entity_ids {
                entity_index.entry(*e).or_default().push(item.id);
            }
            // Reverse edges from each item's inline outgoing links.
            for edge in &item.semantic_out {
                radj_pairs.push((
                    edge.target,
                    InEdge {
                        source: item.id,
                        kind: EdgeKind::Semantic,
                        weight: edge.weight.to_f32(),
                    },
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
            map.insert(item.id, item.clone());
        }
        for ids in entity_index.values_mut() {
            ids.sort();
        }

        Self {
            centroids,
            clusters,
            items: map,
            fts,
            radj: ReverseAdjacency::build(radj_pairs),
            entity_index,
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The vector arm: probe clusters, re-rank exactly, return a ranked id list.
    pub fn vector_arm(&self, query: &[f32], depth: usize, nprobe: usize) -> Vec<MemoryId> {
        if self.centroids.is_empty() {
            return Vec::new();
        }
        let probed = self.centroids.probe(query, nprobe);
        let mut candidates: Vec<StoredMemory> = Vec::new();
        for c in probed {
            candidates.extend_from_slice(&self.clusters[c]);
        }
        exact_search(&candidates, query, depth)
            .into_iter()
            .map(|h| h.id)
            .collect()
    }

    /// The FTS arm: tantivy BM25 over the split (standard k1=1.2, b=0.75).
    pub fn fts_arm(&self, query_text: &str, depth: usize) -> Vec<MemoryId> {
        self.fts
            .search(query_text, depth)
            .into_iter()
            .map(|h| h.id)
            .collect()
    }

    /// The graph arm: link expansion from vector-chosen seeds.
    pub fn graph_arm(&self, query: &[f32], depth: usize, nprobe: usize) -> Vec<MemoryId> {
        // Seeds are the vector arm's top hits, materialized to full records so their
        // inline links are available (SPEC §7.1).
        let seed_ids = self.vector_arm(query, 20, nprobe);
        let seeds: Vec<StoredMemory> = seed_ids
            .iter()
            .filter_map(|id| self.items.get(id).cloned())
            .collect();
        if seeds.is_empty() {
            return Vec::new();
        }
        mlake_graph::retrieve(
            self,
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

    /// Fused query over vector + FTS (+ graph when the corpus has links).
    ///
    /// `text` may be empty for a pure-vector query, and `query` empty for pure-FTS; an arm
    /// with no input simply contributes nothing to the fusion.
    pub fn query(
        &self,
        query: Option<&[f32]>,
        text: Option<&str>,
        top_k: usize,
        config: QueryConfig,
    ) -> Vec<FusedHit> {
        let vector = query
            .map(|q| self.vector_arm(q, config.arm_depth, config.nprobe))
            .unwrap_or_default();
        let fts = text
            .filter(|t| !t.is_empty())
            .map(|t| self.fts_arm(t, config.arm_depth))
            .unwrap_or_default();
        let graph = if self.radj.edge_count() > 0 {
            query
                .map(|q| self.graph_arm(q, config.arm_depth, config.nprobe))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Weighted RRF so the accuracy-tuning lever (per-arm weight) is available; equal
        // weights collapse to canonical RRF.
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
            // A single arm: skip weighting so the output is the arm's own order exactly.
            let only = [RankedArm {
                name: arms[0].0.name,
                ranking: arms[0].0.ranking,
            }];
            return rrf(&only, config.rrf_k, top_k);
        }
        weighted_rrf(&arms, config.rrf_k, top_k)
    }

    /// Resolve an id to its item, for hydrating results.
    pub fn item(&self, id: &MemoryId) -> Option<&StoredMemory> {
        self.items.get(id)
    }
}

impl GraphSource for Engine {
    fn entity_candidates(&self, entity_id: EntityId, memory_type: Option<u8>, cap: usize) -> Vec<MemoryId> {
        self.entity_index
            .get(&entity_id)
            .into_iter()
            .flatten()
            .filter(|id| match (memory_type, self.items.get(id)) {
                (Some(ft), Some(item)) => item.memory_type == ft,
                _ => true,
            })
            .take(cap)
            .copied()
            .collect()
    }

    fn exists(&self, id: &MemoryId) -> bool {
        self.items.contains_key(id)
    }

    fn incoming(&self, target: &MemoryId) -> Vec<InEdge> {
        self.radj.incoming(target).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlake_core::memory::Timestamps;

    fn item(key: &str, vector: Vec<f32>, text: &str) -> StoredMemory {
        StoredMemory {
            id: MemoryId::from_key(key),
            vector,
            text: text.to_string(),
            index_text: String::new(),
            memory_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![],
            semantic_out: vec![],
            causal_out: vec![],
            metadata: vec![],
            write_seq: 0,
        }
    }

    fn engine() -> Engine {
        let items = vec![
            item("cats", vec![1.0, 0.0, 0.0], "cats are wonderful feline pets"),
            item("dogs", vec![0.9, 0.1, 0.0], "dogs are loyal canine companions"),
            item("fish", vec![0.0, 1.0, 0.0], "fish swim in water tanks"),
            item("cars", vec![0.0, 0.0, 1.0], "cars drive on the road"),
        ];
        Engine::build(items, Tokenizer::default())
    }

    #[test]
    fn vector_arm_ranks_by_similarity() {
        let e = engine();
        let hits = e.vector_arm(&[1.0, 0.0, 0.0], 10, 8);
        assert_eq!(hits[0], MemoryId::from_key("cats"));
    }

    #[test]
    fn fts_arm_matches_text() {
        let e = engine();
        let hits = e.fts_arm("loyal canine", 10);
        assert_eq!(hits[0], MemoryId::from_key("dogs"));
    }

    #[test]
    fn fusion_combines_vector_and_text_signals() {
        let e = engine();
        // Vector points at cats; text mentions dogs. Both should surface near the top.
        let fused = e.query(
            Some(&[1.0, 0.0, 0.0]),
            Some("loyal canine companions"),
            10,
            QueryConfig::default(),
        );
        let top2: Vec<MemoryId> = fused.iter().take(2).map(|h| h.id).collect();
        assert!(top2.contains(&MemoryId::from_key("cats")));
        assert!(top2.contains(&MemoryId::from_key("dogs")));
    }

    #[test]
    fn pure_vector_query_ignores_missing_text() {
        let e = engine();
        let fused = e.query(Some(&[0.0, 0.0, 1.0]), None, 10, QueryConfig::default());
        assert_eq!(fused[0].id, MemoryId::from_key("cars"));
    }

    #[test]
    fn pure_text_query_ignores_missing_vector() {
        let e = engine();
        let fused = e.query(None, Some("swim water tanks"), 10, QueryConfig::default());
        assert_eq!(fused[0].id, MemoryId::from_key("fish"));
    }

    #[test]
    fn empty_engine_answers_without_panicking() {
        let e = Engine::build(vec![], Tokenizer::default());
        assert!(e.is_empty());
        assert!(e
            .query(Some(&[1.0, 0.0]), Some("x"), 10, QueryConfig::default())
            .is_empty());
    }
}
