//! Link-expansion scoring math (SPEC §7, ported from Hindsight).
//!
//! Pure functions, isolated here so the exact arithmetic can be table-tested against
//! values captured from the reference Python implementation (gate G-3). The three arms
//! score independently and merge additively; the merged score lands in [0, 3] and becomes
//! the result's `activation`.

use std::collections::HashMap;

use mlake_core::MemoryId;

/// Entity arm: a shared-entity *count* maps to [0, 1) via `tanh(count * 0.5)`.
///
/// The saturation is deliberate — the difference between sharing three entities and four
/// matters far less than between sharing zero and one, and tanh encodes exactly that.
/// Reference values (SPEC §7.2): 1 → 0.46, 2 → 0.76, 3 → 0.91, 4 → 0.96.
pub fn entity_score(shared_count: u32) -> f32 {
    (shared_count as f32 * 0.5).tanh()
}

/// Semantic and causal arms both score a candidate by the *maximum* edge weight reaching
/// it, not a sum: two weak links are not evidence as strong as one strong link.
pub fn max_weight(current: f32, candidate: f32) -> f32 {
    current.max(candidate)
}

/// The additive merge (SPEC §7.5). Each arm contributes its own per-candidate score;
/// a candidate surfaced by several arms accumulates them, rewarding convergent evidence.
#[derive(Default, Clone, Debug)]
pub struct ScoreAccumulator {
    entity: HashMap<MemoryId, f32>,
    semantic: HashMap<MemoryId, f32>,
    causal: HashMap<MemoryId, f32>,
}

impl ScoreAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the entity arm's contribution for a candidate: `tanh(count * 0.5)`.
    pub fn add_entity(&mut self, id: MemoryId, shared_count: u32) {
        // A candidate can be reached from several seeds; the shared-entity count is
        // computed once against the whole seed set, so the last write is authoritative
        // rather than accumulated.
        self.entity.insert(id, entity_score(shared_count));
    }

    /// Record a semantic edge to a candidate, keeping the strongest seen.
    pub fn add_semantic(&mut self, id: MemoryId, weight: f32) {
        let e = self.semantic.entry(id).or_insert(0.0);
        *e = max_weight(*e, weight);
    }

    /// Record a causal edge to a candidate, keeping the strongest seen.
    pub fn add_causal(&mut self, id: MemoryId, weight: f32) {
        let e = self.causal.entry(id).or_insert(0.0);
        *e = max_weight(*e, weight);
    }

    /// Final additive score per candidate, in [0, 3].
    pub fn merged(&self) -> HashMap<MemoryId, f32> {
        let mut all: HashMap<MemoryId, f32> = HashMap::new();
        for (id, s) in &self.entity {
            *all.entry(*id).or_insert(0.0) += s;
        }
        for (id, s) in &self.semantic {
            *all.entry(*id).or_insert(0.0) += s;
        }
        for (id, s) in &self.causal {
            *all.entry(*id).or_insert(0.0) += s;
        }
        all
    }

    /// Merged scores as a ranked list, seeds excluded, truncated to `budget`.
    ///
    /// Sorted by score descending, ties broken by id so the order is deterministic and
    /// differential-testable against the reference (G-2).
    pub fn ranked(&self, seeds: &[MemoryId], budget: usize) -> Vec<(MemoryId, f32)> {
        let seed_set: std::collections::HashSet<MemoryId> = seeds.iter().copied().collect();
        let mut ranked: Vec<(MemoryId, f32)> = self
            .merged()
            .into_iter()
            .filter(|(id, _)| !seed_set.contains(id))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(budget);
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values captured from the reference implementation (SPEC §7.2 doc comment).
    #[test]
    fn entity_score_matches_reference_table() {
        let cases = [(1u32, 0.462), (2, 0.762), (3, 0.905), (4, 0.964)];
        for (count, expected) in cases {
            let got = entity_score(count);
            assert!(
                (got - expected).abs() < 0.001,
                "entity_score({count}) = {got}, reference {expected}"
            );
        }
    }

    #[test]
    fn entity_score_of_zero_is_zero() {
        assert_eq!(entity_score(0), 0.0);
    }

    #[test]
    fn entity_score_saturates_toward_one() {
        // tanh approaches 1 asymptotically. In f32 it reaches exactly 1.0 by a count of a
        // few tens, so the meaningful property is monotone saturation, not staying
        // strictly below the ceiling.
        assert!(entity_score(4) > 0.96);
        assert!(entity_score(10) >= entity_score(4));
        assert!(entity_score(100) <= 1.0);
    }

    #[test]
    fn semantic_and_causal_keep_the_maximum_weight() {
        let mut acc = ScoreAccumulator::new();
        let id = MemoryId::from_key("a");
        acc.add_semantic(id, 0.7);
        acc.add_semantic(id, 0.95);
        acc.add_semantic(id, 0.8);
        assert_eq!(acc.merged()[&id], 0.95);
    }

    #[test]
    fn arms_accumulate_for_convergent_candidates() {
        // A candidate reached by all three arms scores higher than any single arm.
        let mut acc = ScoreAccumulator::new();
        let id = MemoryId::from_key("converged");
        acc.add_entity(id, 2); // 0.762
        acc.add_semantic(id, 0.9);
        acc.add_causal(id, 0.8);
        let score = acc.merged()[&id];
        assert!((score - (0.762 + 0.9 + 0.8)).abs() < 0.01, "got {score}");
        assert!(score <= 3.0, "merged score must stay within [0, 3]");
    }

    #[test]
    fn merged_score_never_exceeds_three() {
        let mut acc = ScoreAccumulator::new();
        let id = MemoryId::from_key("max");
        acc.add_entity(id, 1000); // → ~1.0
        acc.add_semantic(id, 1.0);
        acc.add_causal(id, 1.0);
        assert!(acc.merged()[&id] <= 3.0);
    }

    #[test]
    fn ranking_excludes_seeds() {
        let mut acc = ScoreAccumulator::new();
        let seed = MemoryId::from_key("seed");
        let cand = MemoryId::from_key("cand");
        acc.add_semantic(seed, 1.0); // seed reached via a link, must still be excluded
        acc.add_semantic(cand, 0.9);
        let ranked = acc.ranked(&[seed], 10);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, cand);
    }

    #[test]
    fn ranking_is_score_descending_then_stable() {
        let mut acc = ScoreAccumulator::new();
        let a = MemoryId::from_key("a");
        let b = MemoryId::from_key("b");
        let c = MemoryId::from_key("c");
        acc.add_semantic(a, 0.7);
        acc.add_semantic(b, 0.9);
        acc.add_semantic(c, 0.9); // tie with b
        let ranked = acc.ranked(&[], 10);
        assert_eq!(ranked[0].1, 0.9);
        assert_eq!(ranked[1].1, 0.9);
        assert_eq!(ranked[2].0, a);
        // The two 0.9s are ordered by id, deterministically.
        assert!(ranked[0].0 < ranked[1].0);
    }

    #[test]
    fn budget_truncates_the_ranking() {
        let mut acc = ScoreAccumulator::new();
        for i in 0..10 {
            acc.add_semantic(MemoryId::from_key(&format!("i{i}")), 0.7 + i as f32 * 0.01);
        }
        assert_eq!(acc.ranked(&[], 3).len(), 3);
    }
}
