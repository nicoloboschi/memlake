//! Reciprocal Rank Fusion (SPEC §6.3).
//!
//! The three arms score on incomparable scales — cosine similarity, BM25, and an additive
//! graph activation. RRF sidesteps that by fusing *ranks* rather than scores: a document's
//! contribution from an arm is `1 / (k + rank)`, so only its position in each arm's list
//! matters, not the arm's raw magnitude. This is what lets a strong BM25 hit and a strong
//! vector hit be combined without either arm's scale dominating.

use std::collections::HashMap;

use mlake_core::MemoryId;

/// The RRF constant. 60 is the value from the original Cormack et al. paper and the spec
/// default; a larger k flattens the contribution of top ranks, a smaller k sharpens it.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// One arm's ranked output. Only the order matters to RRF, but the id list must already be
/// sorted best-first.
pub struct RankedArm<'a> {
    pub name: &'a str,
    pub ranking: &'a [MemoryId],
}

/// A fused result: a document and its combined RRF score, plus which arms contributed and
/// the arm scores for debugging and the API response.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedHit {
    pub id: MemoryId,
    pub score: f32,
    /// Per-arm reciprocal-rank contributions, for explainability.
    pub contributions: Vec<(String, f32)>,
}

/// Fuse several ranked arms into one ranking.
///
/// A document present in more than one arm accumulates a contribution from each, which is
/// the whole point: agreement across arms is the signal RRF rewards.
pub fn rrf(arms: &[RankedArm<'_>], k: f32, top_k: usize) -> Vec<FusedHit> {
    let mut scores: HashMap<MemoryId, f32> = HashMap::new();
    let mut contributions: HashMap<MemoryId, Vec<(String, f32)>> = HashMap::new();

    for arm in arms {
        for (rank, id) in arm.ranking.iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_insert(0.0) += contribution;
            contributions
                .entry(*id)
                .or_default()
                .push((arm.name.to_string(), contribution));
        }
    }

    let mut hits: Vec<FusedHit> = scores
        .into_iter()
        .map(|(id, score)| FusedHit {
            id,
            score,
            contributions: contributions.remove(&id).unwrap_or_default(),
        })
        .collect();

    // Descending score, ties broken by id so fusion is deterministic.
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    hits.truncate(top_k);
    hits
}

/// Weighted RRF: scales each arm's contribution by a per-arm weight before summing.
///
/// Plain RRF gives every arm equal say. Once the accuracy gate is in play, letting the
/// stronger arm on a given corpus count for more is the main tuning lever, so the weighted
/// form is provided alongside the canonical one.
pub fn weighted_rrf(
    arms: &[(RankedArm<'_>, f32)],
    k: f32,
    top_k: usize,
) -> Vec<FusedHit> {
    let mut scores: HashMap<MemoryId, f32> = HashMap::new();
    let mut contributions: HashMap<MemoryId, Vec<(String, f32)>> = HashMap::new();

    for (arm, weight) in arms {
        for (rank, id) in arm.ranking.iter().enumerate() {
            let contribution = weight / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_insert(0.0) += contribution;
            contributions
                .entry(*id)
                .or_default()
                .push((arm.name.to_string(), contribution));
        }
    }

    let mut hits: Vec<FusedHit> = scores
        .into_iter()
        .map(|(id, score)| FusedHit {
            id,
            score,
            contributions: contributions.remove(&id).unwrap_or_default(),
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    hits.truncate(top_k);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(k: &str) -> MemoryId {
        MemoryId::from_key(k)
    }

    #[test]
    fn a_document_ranked_first_everywhere_wins() {
        let ranking = [id("a"), id("b"), id("c")];
        let arms = [
            RankedArm { name: "vec", ranking: &ranking },
            RankedArm { name: "fts", ranking: &ranking },
        ];
        let fused = rrf(&arms, DEFAULT_RRF_K, 10);
        assert_eq!(fused[0].id, id("a"));
    }

    #[test]
    fn agreement_across_arms_beats_a_single_strong_hit() {
        // `shared` is 2nd in both arms; `top_v`/`top_f` are 1st in only one arm each.
        let vec = [id("top_v"), id("shared"), id("x")];
        let fts = [id("top_f"), id("shared"), id("y")];
        let fused = rrf(
            &[
                RankedArm { name: "vec", ranking: &vec },
                RankedArm { name: "fts", ranking: &fts },
            ],
            DEFAULT_RRF_K,
            10,
        );
        // Two second-places (2 × 1/62) outweigh one first-place (1/61).
        assert_eq!(fused[0].id, id("shared"), "convergent evidence should win");
    }

    #[test]
    fn rank_not_score_determines_contribution() {
        // RRF only sees order, so an arm with a huge score gap ranks the same as one with
        // a tiny gap.
        let a = [id("a"), id("b")];
        let fused = rrf(&[RankedArm { name: "x", ranking: &a }], DEFAULT_RRF_K, 10);
        assert!((fused[0].score - 1.0 / 61.0).abs() < 1e-6);
        assert!((fused[1].score - 1.0 / 62.0).abs() < 1e-6);
    }

    #[test]
    fn a_single_arm_preserves_its_order() {
        let ranking = [id("a"), id("b"), id("c")];
        let fused = rrf(&[RankedArm { name: "only", ranking: &ranking }], DEFAULT_RRF_K, 10);
        assert_eq!(
            fused.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![id("a"), id("b"), id("c")]
        );
    }

    #[test]
    fn contributions_are_recorded_per_arm() {
        let vec = [id("a")];
        let fts = [id("a")];
        let fused = rrf(
            &[
                RankedArm { name: "vec", ranking: &vec },
                RankedArm { name: "fts", ranking: &fts },
            ],
            DEFAULT_RRF_K,
            10,
        );
        assert_eq!(fused[0].contributions.len(), 2);
    }

    #[test]
    fn weighting_can_flip_the_order() {
        // Two documents, one per arm, each at rank 0. Unweighted they tie and fall back
        // to id order; a heavy vector weight must make the vector document win outright.
        let vec = [id("v")];
        let fts = [id("f")];
        let weighted = weighted_rrf(
            &[
                (RankedArm { name: "vec", ranking: &vec }, 10.0),
                (RankedArm { name: "fts", ranking: &fts }, 1.0),
            ],
            DEFAULT_RRF_K,
            10,
        );
        assert_eq!(weighted[0].id, id("v"), "the up-weighted arm's hit must lead");
        assert!(weighted[0].score > weighted[1].score);
    }

    #[test]
    fn fusion_is_deterministic_for_tied_scores() {
        // Two documents with identical scores must always order the same way. The exact
        // order is by id (which for hashed ids is not the key's lexical order), so assert
        // stability and the tie-break rule rather than a hand-picked winner.
        let vec = [id("a")];
        let fts = [id("b")];
        let arms = || {
            [
                RankedArm { name: "vec", ranking: &vec },
                RankedArm { name: "fts", ranking: &fts },
            ]
        };
        let first = rrf(&arms(), DEFAULT_RRF_K, 10);
        let second = rrf(&arms(), DEFAULT_RRF_K, 10);
        assert_eq!(first, second, "fusion must be deterministic");
        assert_eq!(first[0].score, first[1].score, "scores tie");
        assert!(first[0].id < first[1].id, "ties break by ascending id");
    }

    #[test]
    fn top_k_truncates() {
        let ranking = [id("a"), id("b"), id("c"), id("d")];
        let fused = rrf(&[RankedArm { name: "x", ranking: &ranking }], DEFAULT_RRF_K, 2);
        assert_eq!(fused.len(), 2);
    }
}
