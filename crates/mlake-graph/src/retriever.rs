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

use std::collections::{BTreeSet, HashMap};

use mlake_core::{EntityId, MemoryId, StoredMemory};

use crate::radj::EdgeKind;
use crate::scorer::ScoreAccumulator;

/// Where the retriever gets candidates it does not already hold. The query node
/// implements this over entity postings, the pk index, and reverse adjacency; tests
/// implement it over in-memory maps.
pub trait GraphSource {
    /// Memory ids that mention `entity_id`, filtered to `memory_type` if given, capped at
    /// `cap` candidates. The cap is the bounded posting-prefix read of SPEC §7.2 — it
    /// stops a high-fan-out entity from blowing the budget.
    fn entity_candidates(&self, entity_id: EntityId, memory_type: Option<u8>, cap: usize) -> Vec<MemoryId>;

    /// True if `id` is a live memory — it exists and is not tombstoned. Scoring is purely
    /// structural (edge weights + shared-entity counts), so a candidate never needs to be
    /// hydrated to be scored; this liveness check is all that is needed to drop dangling or
    /// deleted edges (SPEC §7.7). The full memories of only the *ranked* results are fetched,
    /// by the caller, after truncation to `budget`.
    fn exists(&self, id: &MemoryId) -> bool;

    /// Incoming edges for a target, from reverse adjacency.
    fn incoming(&self, target: &MemoryId) -> Vec<crate::radj::InEdge>;

    /// Memory ids written within the temporal-spread window of `seed_id` — its time-neighbours
    /// — filtered to the seed's `memory_type`, capped at `cap`. The window is the source's to
    /// define (it owns the time index); the graph arm only tallies how many seeds each
    /// candidate neighbours. Empty when the seed has no effective timestamp or the source has
    /// no time index. Like `entity_candidates`, this is a bounded posting-style read, never a
    /// hydration.
    fn temporal_candidates(&self, seed_id: MemoryId, cap: usize) -> Vec<MemoryId>;
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
    /// Whether to run the temporal arm (time-neighbour spread). Off keeps the arm out of a
    /// timeout fallback or a corpus without timestamps.
    pub include_temporal_arm: bool,
    /// Time-neighbours read per seed before the cap applies.
    pub per_temporal_cap: usize,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            per_entity_cap: 200,
            budget: 50,
            include_entity_arm: true,
            include_temporal_arm: true,
            per_temporal_cap: 200,
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
    if params.include_temporal_arm {
        expand_temporal(source, seeds, params.per_temporal_cap, &mut acc);
    }

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
    let mut seed_entities: BTreeSet<EntityId> = BTreeSet::new();
    for seed in seeds {
        seed_entities.extend(seed.entity_ids.iter().copied());
    }
    let seed_entities: Vec<EntityId> = seed_entities.into_iter().collect();
    let seed_set: BTreeSet<MemoryId> = seeds.iter().map(|s| s.id).collect();

    // A candidate's shared-entity count *is* the number of distinct seed-entity postings it
    // appears in: the posting for entity `e` is exactly the memories that carry `e`, so a
    // candidate seen in the postings of `e1` and `e3` shares both. Tallying appearances gives
    // the score directly — no candidate is hydrated to recompute `shared_entity_count`, which
    // is what made this arm read a large fraction of the corpus. The per-entity cap is applied
    // to each posting read, matching the reference's LATERAL cap.
    let mut shared: HashMap<MemoryId, u32> = HashMap::new();
    for entity in &seed_entities {
        for id in source.entity_candidates(*entity, memory_type, cap) {
            if !seed_set.contains(&id) {
                *shared.entry(id).or_insert(0) += 1;
            }
        }
    }
    for (cand_id, count) in shared {
        acc.add_entity(cand_id, count);
    }
}

/// Temporal arm: candidates written near a seed in time, scored by how many seeds each one
/// neighbours — the time analogue of the entity arm. A memory close in time to several seeds
/// is more likely part of the same episode, so it accumulates the way a shared entity does.
/// The window and effective-time cascade live in the source; here it is a pure tally.
fn expand_temporal(source: &dyn GraphSource, seeds: &[StoredMemory], cap: usize, acc: &mut ScoreAccumulator) {
    let seed_set: BTreeSet<MemoryId> = seeds.iter().map(|s| s.id).collect();
    let mut near: HashMap<MemoryId, u32> = HashMap::new();
    for seed in seeds {
        for id in source.temporal_candidates(seed.id, cap) {
            if !seed_set.contains(&id) {
                *near.entry(id).or_insert(0) += 1;
            }
        }
    }
    for (cand_id, count) in near {
        acc.add_temporal(cand_id, count);
    }
}

/// Semantic arm: seeds' inline outgoing kNN links plus incoming kNN edges from reverse
/// adjacency. The two directions together mirror the reference's bidirectional check.
fn expand_semantic(source: &dyn GraphSource, seeds: &[StoredMemory], acc: &mut ScoreAccumulator) {
    for seed in seeds {
        // Outgoing: free, already inline in the seed record.
        for edge in &seed.semantic_out {
            if source.exists(&edge.target) {
                acc.add_semantic(edge.target, edge.weight.to_f32());
            }
        }
        // Incoming: from reverse adjacency.
        for in_edge in source.incoming(&seed.id) {
            if in_edge.kind == EdgeKind::Semantic && source.exists(&in_edge.source) {
                acc.add_semantic(in_edge.source, in_edge.weight);
            }
        }
    }
}

/// Causal arm: same mechanics as semantic, over causal edge types.
fn expand_causal(source: &dyn GraphSource, seeds: &[StoredMemory], acc: &mut ScoreAccumulator) {
    for seed in seeds {
        for edge in &seed.causal_out {
            if source.exists(&edge.target) {
                acc.add_causal(edge.target, edge.weight.to_f32());
            }
        }
        for in_edge in source.incoming(&seed.id) {
            if matches!(in_edge.kind, EdgeKind::Causal(_)) && source.exists(&in_edge.source) {
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

    /// A 16-byte entity id from a small integer, for readable tests.
    fn eid(n: u64) -> EntityId {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&n.to_le_bytes());
        EntityId(b)
    }

    /// A simple in-memory graph source for tests.
    struct MemGraph {
        items: HashMap<MemoryId, StoredMemory>,
        tombstoned: BTreeSet<MemoryId>,
        entity_index: HashMap<EntityId, Vec<MemoryId>>,
        radj: ReverseAdjacency,
    }

    impl MemGraph {
        fn new(items: Vec<StoredMemory>, radj_pairs: Vec<(MemoryId, InEdge)>) -> Self {
            let mut entity_index: HashMap<EntityId, Vec<MemoryId>> = HashMap::new();
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
        fn entity_candidates(&self, entity_id: EntityId, memory_type: Option<u8>, cap: usize) -> Vec<MemoryId> {
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

        fn exists(&self, id: &MemoryId) -> bool {
            !self.tombstoned.contains(id) && self.items.contains_key(id)
        }

        fn incoming(&self, target: &MemoryId) -> Vec<InEdge> {
            self.radj.incoming(target).to_vec()
        }

        fn temporal_candidates(&self, seed_id: MemoryId, cap: usize) -> Vec<MemoryId> {
            use mlake_core::memory::{effective_ts, TEMPORAL_SPREAD_WINDOW_MS};
            let Some(seed_ts) = self.items.get(&seed_id).and_then(|m| effective_ts(&m.timestamps)) else {
                return Vec::new();
            };
            let mut out: Vec<MemoryId> = self
                .items
                .values()
                .filter(|m| m.id != seed_id && !self.tombstoned.contains(&m.id))
                .filter(|m| effective_ts(&m.timestamps).is_some_and(|ts| (ts - seed_ts).abs() <= TEMPORAL_SPREAD_WINDOW_MS))
                .map(|m| m.id)
                .collect();
            out.sort();
            out.truncate(cap);
            out
        }
    }

    fn item(key: &str, entities: Vec<u64>) -> StoredMemory {
        StoredMemory {
            id: MemoryId::from_key(key),
            vector: vec![1.0, 0.0],
            text: key.to_string(),
            index_text: String::new(),
            memory_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: {
                let mut e: Vec<EntityId> = entities.into_iter().map(eid).collect();
                e.sort_unstable();
                e
            },
            semantic_out: vec![],
            causal_out: vec![],
            metadata: vec![],
            write_seq: 0,
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

    fn item_at(key: &str, ts_ms: i64) -> StoredMemory {
        let mut m = item(key, vec![]);
        m.timestamps.occurred_start = Some(ts_ms);
        m
    }

    #[test]
    fn temporal_arm_scores_time_neighbours() {
        let day = 24 * 60 * 60 * 1000i64;
        // Seed at t=0: `near` is inside the ±24h window, `far` well outside it.
        let seed = item_at("seed", 0);
        let near = item_at("near", day / 2); // 12h away → neighbour
        let far = item_at("far", 5 * day); // 5 days away → not a neighbour
        let graph = MemGraph::new(vec![seed.clone(), near, far], vec![]);

        let results = retrieve(&graph, &[seed], GraphParams::default());
        let by_id: HashMap<MemoryId, f32> = results.iter().map(|r| (r.id, r.activation)).collect();

        assert!(by_id.contains_key(&MemoryId::from_key("near")), "time-neighbour surfaces");
        assert!(!by_id.contains_key(&MemoryId::from_key("far")), "out-of-window item absent");
        // One neighbouring seed → temporal_score(1) = 0.5 * tanh(0.5) ≈ 0.231.
        assert!((by_id[&MemoryId::from_key("near")] - 0.231).abs() < 0.01, "{by_id:?}");
    }

    #[test]
    fn temporal_arm_can_be_disabled() {
        let day = 24 * 60 * 60 * 1000i64;
        let seed = item_at("seed", 0);
        let near = item_at("near", day / 2);
        let graph = MemGraph::new(vec![seed.clone(), near], vec![]);
        // No entities, no edges, temporal off → the arm contributes nothing and nothing ranks.
        let params = GraphParams { include_temporal_arm: false, ..GraphParams::default() };
        assert!(retrieve(&graph, &[seed], params).is_empty());
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
