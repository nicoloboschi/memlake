//! Temporal-arm scoring, a 1:1 port of Hindsight's `retrieve_temporal_combined`
//! (`hindsight_api/engine/search/retrieval.py`). Pure functions, isolated here so the exact
//! arithmetic can be golden-tested against the Python — the same discipline as the graph
//! scorer (gate G-3).
//!
//! The arm has three parts; this module is the scoring math. The storage flow (entry-point
//! selection over the time index, one-hop link spread) lives in the query node.
//!
//! * **entry-point coverage** — from a similarity-ranked pool of in-window candidates, pick
//!   `limit` that span the window: split it into `n_buckets`, take the best from each
//!   populated bucket round-robin (`_select_with_temporal_coverage`);
//! * **temporal proximity** — `1 - min(|best - mid| / (span/2), 1)`, so a memory at the
//!   window's centre scores 1 and one at an edge scores 0;
//! * **spread propagation** — a neighbour's score is `max(its own proximity, parent · weight
//!   · causal_boost · 0.7)`.

use mlake_core::MemoryId;

/// The date used for *scoring* a unit (distinct from the `effective_ts` used to index/window
/// it): the midpoint of an occurred interval, else its start, else its end, else the mention
/// time. Matches Hindsight's `best_date` cascade.
pub fn best_date(occurred_start: Option<i64>, occurred_end: Option<i64>, mentioned_at: Option<i64>) -> Option<i64> {
    match (occurred_start, occurred_end) {
        (Some(s), Some(e)) => Some(s + (e - s) / 2),
        (Some(s), None) => Some(s),
        (None, Some(e)) => Some(e),
        (None, None) => mentioned_at,
    }
}

/// Temporal proximity of `date` to the centre of the window `[from, to]`, in `[0, 1]`.
///
/// `1 - min(|date - mid| / (span/2), 1)` where `mid = from + (to-from)/2`. The 86400 (seconds
/// per day) that Hindsight divides by cancels between numerator and denominator, so this is
/// independent of the epoch unit (seconds vs millis) as long as all three share it. A zero-
/// width window scores 1.
pub fn temporal_proximity(date: i64, from: i64, to: i64) -> f32 {
    let mid = from + (to - from) / 2;
    let half_span = (to - from) as f64 / 2.0;
    if half_span <= 0.0 {
        return 1.0;
    }
    let ratio = ((date - mid).abs() as f64 / half_span).min(1.0);
    (1.0 - ratio) as f32
}

/// Spread multiplier per link kind (Hindsight: causes/caused_by → 2.0, enables/prevents →
/// 1.5, everything else — semantic/temporal — → 1.0).
pub fn causal_boost(is_causal_strong: bool, is_causal_weak: bool) -> f32 {
    if is_causal_strong {
        2.0
    } else if is_causal_weak {
        1.5
    } else {
        1.0
    }
}

/// A neighbour's temporal score during one-hop spread: the stronger of its own proximity and
/// the score propagated from its parent (`parent · weight · boost · 0.7`).
pub fn propagate(neighbor_proximity: f32, parent_score: f32, weight: f32, boost: f32) -> f32 {
    let propagated = parent_score * weight * boost * 0.7;
    neighbor_proximity.max(propagated)
}

/// A candidate for entry-point coverage selection: its id, its similarity to the query, and
/// its effective timestamp (`None` if it has no date — it lands in bucket 0).
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub id: MemoryId,
    pub similarity: f32,
    pub effective_ts: Option<i64>,
}

/// Pick `limit` entry points from a pool, spread across the window `[from, to]`.
///
/// A 1:1 port of `_select_with_temporal_coverage`: split the window into `n_buckets`, rank by
/// similarity, then take the best item from each populated bucket, then the second-best from
/// each, and so on — so every populated slice of the window is represented before any slice
/// contributes a second item. Within a tier, higher similarity leads. Degenerate dates (all
/// one bucket) collapse to plain similarity order.
pub fn select_with_temporal_coverage(
    mut pool: Vec<Candidate>,
    from: i64,
    to: i64,
    limit: usize,
    n_buckets: usize,
) -> Vec<Candidate> {
    if pool.len() <= limit {
        return pool;
    }
    // Rank by similarity desc (ties broken by id for determinism).
    pool.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    let span = (to - from) as f64;
    let n = n_buckets.max(1);
    let bucket_of = |c: &Candidate| -> usize {
        match c.effective_ts {
            Some(d) if span > 0.0 => {
                let frac = (d - from) as f64 / span;
                ((frac * n as f64) as isize).clamp(0, n as isize - 1) as usize
            }
            _ => 0,
        }
    };
    // BTreeMap keeps bucket iteration deterministic; each bucket inherits the sim-desc order.
    let mut buckets: std::collections::BTreeMap<usize, Vec<Candidate>> = std::collections::BTreeMap::new();
    for c in pool {
        buckets.entry(bucket_of(&c)).or_default().push(c);
    }
    let mut selected: Vec<Candidate> = Vec::with_capacity(limit);
    let mut tier = 0;
    while selected.len() < limit && buckets.values().any(|b| b.len() > tier) {
        let mut tier_rows: Vec<Candidate> = buckets.values().filter_map(|b| b.get(tier).copied()).collect();
        tier_rows.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        for row in tier_rows {
            if selected.len() < limit {
                selected.push(row);
            }
        }
        tier += 1;
    }
    selected
}

/// Pool / entry-point / bucket sizes, matching Hindsight's constants.
pub const TEMPORAL_POOL_SIZE: usize = 60;
pub const TEMPORAL_ENTRY_POINTS: usize = 10;
pub const TEMPORAL_COVERAGE_BUCKETS: usize = 8;
/// Default proximity when a unit has no date: 0.5 for entry points, 0.3 for spread neighbours.
pub const NO_DATE_ENTRY: f32 = 0.5;
pub const NO_DATE_NEIGHBOR: f32 = 0.3;

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u8) -> MemoryId {
        MemoryId::from_bytes([n; 16])
    }

    #[test]
    fn proximity_peaks_at_the_centre_and_zeros_at_the_edges() {
        // Window [0, 100]: mid = 50, half-span = 50.
        assert!((temporal_proximity(50, 0, 100) - 1.0).abs() < 1e-6);
        assert!((temporal_proximity(0, 0, 100) - 0.0).abs() < 1e-6);
        assert!((temporal_proximity(100, 0, 100) - 0.0).abs() < 1e-6);
        // Quarter in from the centre -> 0.5.
        assert!((temporal_proximity(25, 0, 100) - 0.5).abs() < 1e-6);
        // Beyond the window clamps at 0.
        assert!((temporal_proximity(-100, 0, 100) - 0.0).abs() < 1e-6);
        // Zero-width window -> 1.
        assert!((temporal_proximity(42, 10, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn proximity_is_epoch_unit_independent() {
        // Seconds vs millis for the same relative position gives the same score.
        let secs = temporal_proximity(25, 0, 100);
        let millis = temporal_proximity(25_000, 0, 100_000);
        assert!((secs - millis).abs() < 1e-6);
    }

    #[test]
    fn best_date_follows_the_cascade() {
        assert_eq!(best_date(Some(10), Some(20), Some(99)), Some(15)); // occurred midpoint
        assert_eq!(best_date(Some(10), None, Some(99)), Some(10)); // occurred_start
        assert_eq!(best_date(None, Some(20), Some(99)), Some(20)); // occurred_end
        assert_eq!(best_date(None, None, Some(99)), Some(99)); // mentioned_at
        assert_eq!(best_date(None, None, None), None);
    }

    #[test]
    fn causal_boost_matches_reference() {
        assert_eq!(causal_boost(true, false), 2.0); // causes / caused_by
        assert_eq!(causal_boost(false, true), 1.5); // enables / prevents
        assert_eq!(causal_boost(false, false), 1.0); // semantic / temporal
    }

    #[test]
    fn propagate_takes_the_stronger_of_own_and_propagated() {
        // parent 1.0 * weight 0.9 * boost 2.0 * 0.7 = 1.26 -> beats own 0.3.
        assert!((propagate(0.3, 1.0, 0.9, 2.0) - 1.26).abs() < 1e-5);
        // own proximity higher -> keep it.
        assert!((propagate(0.8, 0.5, 0.5, 1.0) - 0.8).abs() < 1e-5);
    }

    #[test]
    fn coverage_selection_spreads_across_buckets_before_doubling_up() {
        // Window [0, 80], 8 buckets of width 10. Pool: two candidates in bucket 0 (ts 0, 5)
        // and two in bucket 7 (ts 70, 75). limit 2 must take one from each bucket (coverage),
        // not the two highest-similarity ones (which are both in bucket 0).
        let pool = vec![
            Candidate { id: id(1), similarity: 0.9, effective_ts: Some(0) },
            Candidate { id: id(2), similarity: 0.8, effective_ts: Some(5) },
            Candidate { id: id(3), similarity: 0.7, effective_ts: Some(70) },
            Candidate { id: id(4), similarity: 0.6, effective_ts: Some(75) },
        ];
        let sel = select_with_temporal_coverage(pool, 0, 80, 2, 8);
        let ids: Vec<MemoryId> = sel.iter().map(|c| c.id).collect();
        assert!(ids.contains(&id(1)), "best of bucket 0");
        assert!(ids.contains(&id(3)), "best of bucket 7 (coverage beats sim-2)");
        assert!(!ids.contains(&id(2)), "second of bucket 0 must not preempt coverage");
    }

    #[test]
    fn coverage_returns_pool_when_within_limit() {
        let pool = vec![
            Candidate { id: id(1), similarity: 0.9, effective_ts: Some(0) },
            Candidate { id: id(2), similarity: 0.8, effective_ts: Some(50) },
        ];
        assert_eq!(select_with_temporal_coverage(pool.clone(), 0, 100, 5, 8).len(), 2);
    }
}
