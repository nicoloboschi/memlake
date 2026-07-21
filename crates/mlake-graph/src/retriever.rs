//! Link-expansion retrieval (SPEC §7), a behavioural port of Hindsight's
//! `LinkExpansionRetriever`.
//!
//! Given seeds already chosen by vector search, expand exactly one hop through three
//! independent signals and merge their scores:
//!
//! * **entity** — candidates sharing entity ids with the seed set, scored by shared count;
//! * **semantic** — the precomputed kNN graph, both the seeds' inline outgoing links and
//!   incoming links from reverse adjacency;
//! * **causal** — the same, over causal edge types.
//!
//! The hard bounds are what keep this inside the roundtrip budget (INV-7): one hop, a cap
//! on candidates per entity, and a timeout fallback that drops the entity arm — never a
//! recursive walk whose cost depends on graph shape (SPEC §7, "Forbidden").

use std::collections::BTreeSet;

use mlake_core::{MemoryId, StoredMemory};

use crate::radj::EdgeKind;
use crate::scorer::ScoreAccumulator;

/// Where the retriever gets candidates it does not already hold. The query node
/// implements this over entity postings, the pk index, and reverse adjacency; tests
/// implement it over in-memory maps.
pub trait GraphSource {
    /// Memory ids that mention `entity_id`, filtered to `memory_type` if given, capped at
    /// `cap` candidates. The cap is the bounded posting-prefix read of SPEC §7.2 — it
    /// stops a high-fan-out entity from blowing the budget.
    fn entity_candidates(&self, entity_id: u64, memory_type: Option<u8>, cap: usize) -> Vec<MemoryId>;

    /// Materialize an item by id, or `None` if it is tombstoned or absent. This is where
    /// dangling edges become invisible without any cleanup (SPEC §7.7).
    fn item(&self, id: &MemoryId) -> Option<StoredMemory>;

    /// Incoming edges for a target, from reverse adjacency.
    fn incoming(&self, target: &MemoryId) -> Vec<crate::radj::InEdge>;
}

/// Retrieval parameters. Defaults match SPEC §7.
#[derive(Clone, Copy, Debug)]
pub struct GraphParams {
    /// Candidates read per entity before the cap applies (SPEC §7.2, default 200).
    pub per_entity_cap: usize,
    /// Result budget after merge and truncation.
    pub budget: usize,
    /// Whether to run the entity arm. Set false to model the timeout fallback, which
    /// serves semantic + causal only (SPEC §7.6).
    pub include_entity_arm: bool,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            per_entity_cap: 200,
            budget: 50,
            include_entity_arm: true,
        }
    }
}

/// A graph-expansion result: a candidate and its additive activation score in [0, 3].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GraphResult {
    pub id: MemoryId,
    pub activation: f32,
}

/// Expand links from `seeds` and return ranked candidates.
///
/// `seeds` are the full seed records (SPEC feeds these from RT3's cluster fetch, so their
/// inline links come for free). Seed ids are excluded from the output.
pub fn retrieve(
    source: &dyn GraphSource,
    seeds: &[StoredMemory],
    params: GraphParams,
) -> Vec<GraphResult> {
    if seeds.is_empty() {
        return Vec::new();
    }

    let seed_ids: Vec<MemoryId> = seeds.iter().map(|s| s.id).collect();
    let memory_type = seeds.first().map(|s| s.memory_type);
    let mut acc = ScoreAccumulator::new();

    if params.include_entity_arm {
        expand_entities(source, seeds, memory_type, params.per_entity_cap, &mut acc);
    }
    expand_semantic(source, seeds, &mut acc);
    expand_causal(source, seeds, &mut acc);

    acc.ranked(&seed_ids, params.budget)
        .into_iter()
        .map(|(id, activation)| GraphResult { id, activation })
        .collect()
}

/// Entity arm: candidates are scored by how many distinct entities they share with the
/// seed set as a whole.
fn expand_entities(
    source: &dyn GraphSource,
    seeds: &[StoredMemory],
    memory_type: Option<u8>,
    cap: usize,
    acc: &mut ScoreAccumulator,
) {
    // Union of all seed entities. Sorted (StoredMemory keeps entity_ids sorted) so shared
    // counting is a linear merge.
    let mut seed_entities: BTreeSet<u64> = BTreeSet::new();
    for seed in seeds {
        seed_entities.extend(seed.entity_ids.iter().copied());
    }
    let seed_entities: Vec<u64> = seed_entities.into_iter().collect();
    let seed_set: BTreeSet<MemoryId> = seeds.iter().map(|s| s.id).collect();

    // Gather candidate ids across every seed entity, deduplicated. The per-entity cap is
    // applied to each posting read, not to the union, matching the reference's LATERAL cap.
    let mut candidates: BTreeSet<MemoryId> = BTreeSet::new();
    for entity in &seed_entities {
        for id in source.entity_candidates(*entity, memory_type, cap) {
            if !seed_set.contains(&id) {
                candidates.insert(id);
            }
        }
    }

    for cand_id in candidates {
        let Some(cand) = source.item(&cand_id) else {
            // Tombstoned or dangling: skip, no cleanup needed.
            continue;
        };
        let shared = cand.shared_entity_count(&seed_entities) as u32;
        if shared > 0 {
            acc.add_entity(cand_id, shared);
        }
    }
}

/// Semantic arm: seeds' inline outgoing kNN links plus incoming kNN edges from reverse
/// adjacency. The two directions together mirror the reference's bidirectional check.
fn expand_semantic(source: &dyn GraphSource, seeds: &[StoredMemory], acc: &mut ScoreAccumulator) {
    for seed in seeds {
        // Outgoing: free, already inline in the seed record.
        for edge in &seed.semantic_out {
            if source.item(&edge.target).is_some() {
                acc.add_semantic(edge.target, edge.weight.to_f32());
            }
        }
        // Incoming: from reverse adjacency.
        for in_edge in source.incoming(&seed.id) {
            if in_edge.kind == EdgeKind::Semantic && source.item(&in_edge.source).is_some() {
                acc.add_semantic(in_edge.source, in_edge.weight);
            }
        }
    }
}

/// Causal arm: same mechanics as semantic, over causal edge types.
fn expand_causal(source: &dyn GraphSource, seeds: &[StoredMemory], acc: &mut ScoreAccumulator) {
    for seed in seeds {
        for edge in &seed.causal_out {
            if source.item(&edge.target).is_some() {
                acc.add_causal(edge.target, edge.weight.to_f32());
            }
        }
        for in_edge in source.incoming(&seed.id) {
            if matches!(in_edge.kind, EdgeKind::Causal(_)) && source.item(&in_edge.source).is_some()
            {
                acc.add_causal(in_edge.source, in_edge.weight);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radj::{InEdge, ReverseAdjacency};
    use mlake_core::memory::{Timestamps, Weight};
    use mlake_core::{CausalEdge, LinkType, SemanticEdge};
    use std::collections::HashMap;

    /// A simple in-memory graph source for tests.
    struct MemGraph {
        items: HashMap<MemoryId, StoredMemory>,
        tombstoned: BTreeSet<MemoryId>,
        entity_index: HashMap<u64, Vec<MemoryId>>,
        radj: ReverseAdjacency,
    }

    impl MemGraph {
        fn new(items: Vec<StoredMemory>, radj_pairs: Vec<(MemoryId, InEdge)>) -> Self {
            let mut entity_index: HashMap<u64, Vec<MemoryId>> = HashMap::new();
            let mut map = HashMap::new();
            for item in items {
                for e in &item.entity_ids {
                    entity_index.entry(*e).or_default().push(item.id);
                }
                map.insert(item.id, item);
            }
            for ids in entity_index.values_mut() {
                ids.sort();
            }
            Self {
                items: map,
                tombstoned: BTreeSet::new(),
                entity_index,
                radj: ReverseAdjacency::build(radj_pairs),
            }
        }
    }

    impl GraphSource for MemGraph {
        fn entity_candidates(&self, entity_id: u64, memory_type: Option<u8>, cap: usize) -> Vec<MemoryId> {
            self.entity_index
                .get(&entity_id)
                .into_iter()
                .flatten()
                .filter(|id| !self.tombstoned.contains(id))
                .filter(|id| match (memory_type, self.items.get(id)) {
                    (Some(ft), Some(item)) => item.memory_type == ft,
                    _ => true,
                })
                .take(cap)
                .copied()
                .collect()
        }

        fn item(&self, id: &MemoryId) -> Option<StoredMemory> {
            if self.tombstoned.contains(id) {
                return None;
            }
            self.items.get(id).cloned()
        }

        fn incoming(&self, target: &MemoryId) -> Vec<InEdge> {
            self.radj.incoming(target).to_vec()
        }
    }

    fn item(key: &str, entities: Vec<u64>) -> StoredMemory {
        StoredMemory {
            id: MemoryId::from_key(key),
            vector: vec![1.0, 0.0],
            text: key.to_string(),
            memory_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: {
                let mut e = entities;
                e.sort_unstable();
                e
            },
            semantic_out: vec![],
            causal_out: vec![],
        }
    }

    #[test]
    fn entity_arm_scores_by_shared_count() {
        // Seed shares 2 entities with A, 1 with B, 0 with C.
        let seed = item("seed", vec![1, 2, 3]);
        let a = item("a", vec![1, 2, 9]);
        let b = item("b", vec![3, 8]);
        let c = item("c", vec![7]);
        let graph = MemGraph::new(vec![seed.clone(), a, b, c], vec![]);

        let results = retrieve(&graph, &[seed], GraphParams::default());
        let by_id: HashMap<MemoryId, f32> =
            results.iter().map(|r| (r.id, r.activation)).collect();

        // A (2 shared) ranks above B (1 shared); C (0 shared) is absent.
        assert!(by_id[&MemoryId::from_key("a")] > by_id[&MemoryId::from_key("b")]);
        assert!(!by_id.contains_key(&MemoryId::from_key("c")));
        // Scores match the tanh table.
        assert!((by_id[&MemoryId::from_key("a")] - 0.762).abs() < 0.01);
        assert!((by_id[&MemoryId::from_key("b")] - 0.462).abs() < 0.01);
    }

    #[test]
    fn semantic_arm_uses_inline_outgoing_links() {
        let mut seed = item("seed", vec![]);
        seed.semantic_out = vec![SemanticEdge {
            target: MemoryId::from_key("neighbour"),
            weight: Weight::from_f32(0.85),
        }];
        let neighbour = item("neighbour", vec![]);
        let graph = MemGraph::new(vec![seed.clone(), neighbour], vec![]);

        let results = retrieve(&graph, &[seed], GraphParams::default());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, MemoryId::from_key("neighbour"));
        assert!((results[0].activation - 0.85).abs() < 0.01);
    }

    #[test]
    fn semantic_arm_uses_incoming_reverse_edges() {
        // Nobody's outgoing links point at the seed's neighbour; the edge is only
        // discoverable through reverse adjacency.
        let seed = item("seed", vec![]);
        let source = item("source", vec![]);
        let radj = vec![(
            MemoryId::from_key("seed"),
            InEdge {
                source: MemoryId::from_key("source"),
                kind: EdgeKind::Semantic,
                weight: 0.9,
            },
        )];
        let graph = MemGraph::new(vec![seed.clone(), source], radj);

        let results = retrieve(&graph, &[seed], GraphParams::default());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, MemoryId::from_key("source"));
        assert!((results[0].activation - 0.9).abs() < 0.01);
    }

    #[test]
    fn causal_arm_expands_causal_edges() {
        let mut seed = item("seed", vec![]);
        seed.causal_out = vec![CausalEdge {
            target: MemoryId::from_key("effect"),
            link_type: LinkType::Causes,
            weight: Weight::from_f32(0.75),
        }];
        let effect = item("effect", vec![]);
        let graph = MemGraph::new(vec![seed.clone(), effect], vec![]);

        let results = retrieve(&graph, &[seed], GraphParams::default());
        assert_eq!(results[0].id, MemoryId::from_key("effect"));
        assert!((results[0].activation - 0.75).abs() < 0.01);
    }

    #[test]
    fn convergent_candidate_accumulates_across_arms() {
        // The same candidate is reached by both the entity arm and the semantic arm.
        let mut seed = item("seed", vec![1, 2]);
        seed.semantic_out = vec![SemanticEdge {
            target: MemoryId::from_key("both"),
            weight: Weight::from_f32(0.8),
        }];
        let both = item("both", vec![1, 2]); // shares 2 entities too
        let graph = MemGraph::new(vec![seed.clone(), both], vec![]);

        let results = retrieve(&graph, &[seed], GraphParams::default());
        let score = results[0].activation;
        // entity(2)=0.762 + semantic=0.8
        assert!((score - (0.762 + 0.8)).abs() < 0.02, "got {score}");
    }

    #[test]
    fn seeds_are_never_returned() {
        // Two seeds that link to each other must not surface each other as results.
        let mut a = item("a", vec![1]);
        let mut b = item("b", vec![1]);
        a.semantic_out = vec![SemanticEdge {
            target: MemoryId::from_key("b"),
            weight: Weight::from_f32(0.9),
        }];
        b.semantic_out = vec![SemanticEdge {
            target: MemoryId::from_key("a"),
            weight: Weight::from_f32(0.9),
        }];
        let graph = MemGraph::new(vec![a.clone(), b.clone()], vec![]);

        let results = retrieve(&graph, &[a, b], GraphParams::default());
        assert!(results.is_empty(), "seeds must be excluded, got {results:?}");
    }

    #[test]
    fn tombstoned_candidates_are_invisible() {
        let mut seed = item("seed", vec![]);
        seed.semantic_out = vec![SemanticEdge {
            target: MemoryId::from_key("deleted"),
            weight: Weight::from_f32(0.9),
        }];
        let deleted = item("deleted", vec![]);
        let mut graph = MemGraph::new(vec![seed.clone(), deleted], vec![]);
        graph.tombstoned.insert(MemoryId::from_key("deleted"));

        let results = retrieve(&graph, &[seed], GraphParams::default());
        assert!(
            results.is_empty(),
            "a dangling edge to a tombstoned item must not surface it"
        );
    }

    #[test]
    fn timeout_fallback_drops_only_the_entity_arm() {
        // Same graph, with and without the entity arm.
        let mut seed = item("seed", vec![1, 2]);
        seed.semantic_out = vec![SemanticEdge {
            target: MemoryId::from_key("sem"),
            weight: Weight::from_f32(0.9),
        }];
        let entity_only = item("ent", vec![1, 2]);
        let sem = item("sem", vec![]);
        let graph = MemGraph::new(vec![seed.clone(), entity_only, sem], vec![]);

        let full = retrieve(&graph, &[seed.clone()], GraphParams::default());
        assert!(full.iter().any(|r| r.id == MemoryId::from_key("ent")));

        let fallback = retrieve(
            &graph,
            &[seed],
            GraphParams {
                include_entity_arm: false,
                ..GraphParams::default()
            },
        );
        // The entity-only candidate is gone; the semantic one remains.
        assert!(!fallback.iter().any(|r| r.id == MemoryId::from_key("ent")));
        assert!(fallback.iter().any(|r| r.id == MemoryId::from_key("sem")));
    }

    #[test]
    fn per_entity_cap_bounds_the_candidate_read() {
        // One very high-fan-out entity: the cap must limit how many candidates it yields.
        let seed = item("seed", vec![42]);
        let mut items = vec![seed.clone()];
        for i in 0..500 {
            items.push(item(&format!("c{i}"), vec![42]));
        }
        let graph = MemGraph::new(items, vec![]);

        let results = retrieve(
            &graph,
            &[seed],
            GraphParams {
                per_entity_cap: 10,
                ..GraphParams::default()
            },
        );
        assert!(
            results.len() <= 10,
            "cap must bound entity fan-out, got {}",
            results.len()
        );
    }

    #[test]
    fn no_seeds_yields_no_results() {
        let graph = MemGraph::new(vec![], vec![]);
        assert!(retrieve(&graph, &[], GraphParams::default()).is_empty());
    }
}
