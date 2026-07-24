//! Per-query S3 accounting.
//!
//! INV-7 says every query issues a statically bounded number of roundtrips regardless of
//! data size or graph shape. That is only enforceable if it is *measured*, so every
//! request in the critical path flows through a tracker and a query that exceeds its
//! budget is reported as the bug it is (SPEC §6.1).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The hard cold-path roundtrip budget from SPEC §6.1.
pub const COLD_ROUNDTRIP_BUDGET: usize = 4;

/// A timed phase of a query, for the diagnostics breakdown.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Probe,
    FetchClusters,
    /// Stage one of the vector arm: the RaBitQ 1-bit scan over the probed clusters, producing an
    /// error interval per candidate. This — NOT `Rerank` — is the dominant vector-arm CPU cost, and
    /// the ONLY vector cost link derivation pays (it skips stage two). Split out from `Rerank` so a
    /// trace does not misattribute the scan to reranking.
    Scan,
    Rerank,
    Fts,
    GraphRadj,
    GraphPk,
    GraphFetch,
    GraphExpand,
    Fuse,
}

impl Phase {
    pub const ALL: [Phase; 10] = [
        Phase::Probe,
        Phase::FetchClusters,
        Phase::Scan,
        Phase::Rerank,
        Phase::Fts,
        Phase::GraphRadj,
        Phase::GraphPk,
        Phase::GraphFetch,
        Phase::GraphExpand,
        Phase::Fuse,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Phase::Probe => "probe",
            Phase::FetchClusters => "fetch_clusters",
            Phase::Scan => "scan",
            Phase::Rerank => "rerank",
            Phase::Fts => "fts",
            Phase::GraphRadj => "graph_radj",
            Phase::GraphPk => "graph_pk",
            Phase::GraphFetch => "graph_fetch",
            Phase::GraphExpand => "graph_expand",
            Phase::Fuse => "fuse",
        }
    }
    fn idx(self) -> usize {
        self as usize
    }
}

/// Counters for one query's object-storage usage and phase timing.
#[derive(Debug)]
pub struct QueryMetrics {
    /// Highest roundtrip number reached. Requests issued in parallel share a number, so
    /// this counts *round trips*, not requests.
    max_roundtrip: AtomicUsize,
    requests: AtomicUsize,
    bytes: AtomicU64,
    latency_micros: AtomicU64,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    /// Accumulated wall-clock micros per phase — the diagnostics breakdown.
    phase_micros: [AtomicU64; 10],
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self {
            max_roundtrip: AtomicUsize::new(0),
            requests: AtomicUsize::new(0),
            bytes: AtomicU64::new(0),
            latency_micros: AtomicU64::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            phase_micros: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl QueryMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record `duration` against a phase. Use [`QueryMetrics::time`] for the common
    /// "time this block" pattern.
    pub fn record_phase(&self, phase: Phase, duration: Duration) {
        self.phase_micros[phase.idx()].fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Micros accumulated for a phase.
    pub fn phase_micros(&self, phase: Phase) -> u64 {
        self.phase_micros[phase.idx()].load(Ordering::Relaxed)
    }

    /// The full phase breakdown, `(name, micros)`, in a stable order.
    pub fn phase_breakdown(&self) -> Vec<(&'static str, u64)> {
        Phase::ALL
            .iter()
            .map(|p| (p.name(), self.phase_micros(*p)))
            .collect()
    }

    pub fn record_request(&self, roundtrip_no: usize, bytes: u64, latency: Duration) {
        self.max_roundtrip.fetch_max(roundtrip_no, Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.latency_micros
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn roundtrips(&self) -> usize {
        self.max_roundtrip.load(Ordering::Relaxed)
    }

    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    pub fn cache_hits(&self) -> usize {
        self.cache_hits.load(Ordering::Relaxed)
    }

    pub fn cache_misses(&self) -> usize {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// True when this query stayed inside the cold-path budget.
    pub fn within_budget(&self) -> bool {
        self.roundtrips() <= COLD_ROUNDTRIP_BUDGET
    }

    /// Emit a warning if the budget was blown. Called once at query completion; the
    /// caller decides whether to also fail the request (tests do, production does not).
    pub fn check_budget(&self, namespace: &str, query_id: &str) {
        if !self.within_budget() {
            tracing::warn!(
                metric = "roundtrip_budget_exceeded",
                namespace,
                query_id,
                roundtrips = self.roundtrips(),
                budget = COLD_ROUNDTRIP_BUDGET,
                "query exceeded the cold-path roundtrip budget"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_requests_in_one_roundtrip_count_once() {
        let m = QueryMetrics::new();
        // Three files fetched concurrently at RT3 is one roundtrip, not three.
        m.record_request(3, 100, Duration::from_millis(80));
        m.record_request(3, 200, Duration::from_millis(85));
        m.record_request(3, 300, Duration::from_millis(90));
        assert_eq!(m.roundtrips(), 3);
        assert_eq!(m.requests(), 3);
        assert_eq!(m.bytes(), 600);
    }

    #[test]
    fn budget_is_four_roundtrips() {
        let m = QueryMetrics::new();
        m.record_request(4, 0, Duration::ZERO);
        assert!(m.within_budget());
        m.record_request(5, 0, Duration::ZERO);
        assert!(!m.within_budget());
    }

    #[test]
    fn out_of_order_records_still_track_the_max() {
        let m = QueryMetrics::new();
        m.record_request(4, 0, Duration::ZERO);
        m.record_request(2, 0, Duration::ZERO);
        assert_eq!(m.roundtrips(), 4);
    }
}

/// Lifetime object-storage accounting for a `Store` handle, across every operation — the
/// basis for the cost model in the performance suite. Unlike [`QueryMetrics`] (per-query,
/// critical-path only), this counts *all* GET/PUT/LIST/DELETE calls and their bytes.
#[derive(Debug, Default)]
pub struct StoreMetrics {
    gets: AtomicU64,
    puts: AtomicU64,
    lists: AtomicU64,
    deletes: AtomicU64,
    get_bytes: AtomicU64,
    put_bytes: AtomicU64,
}

impl StoreMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn record_get(&self, bytes: u64) {
        self.gets.fetch_add(1, Ordering::Relaxed);
        self.get_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    pub fn record_put(&self, bytes: u64) {
        self.puts.fetch_add(1, Ordering::Relaxed);
        self.put_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    pub fn record_list(&self) {
        self.lists.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_delete(&self) {
        self.deletes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn gets(&self) -> u64 {
        self.gets.load(Ordering::Relaxed)
    }
    pub fn puts(&self) -> u64 {
        self.puts.load(Ordering::Relaxed)
    }
    pub fn lists(&self) -> u64 {
        self.lists.load(Ordering::Relaxed)
    }
    pub fn deletes(&self) -> u64 {
        self.deletes.load(Ordering::Relaxed)
    }
    pub fn get_bytes(&self) -> u64 {
        self.get_bytes.load(Ordering::Relaxed)
    }
    pub fn put_bytes(&self) -> u64 {
        self.put_bytes.load(Ordering::Relaxed)
    }

    /// A snapshot difference `self - base`, for measuring one phase (write, read) in
    /// isolation against a starting snapshot.
    pub fn since(&self, base: &StoreSnapshot) -> StoreSnapshot {
        StoreSnapshot {
            gets: self.gets().saturating_sub(base.gets),
            puts: self.puts().saturating_sub(base.puts),
            lists: self.lists().saturating_sub(base.lists),
            deletes: self.deletes().saturating_sub(base.deletes),
            get_bytes: self.get_bytes().saturating_sub(base.get_bytes),
            put_bytes: self.put_bytes().saturating_sub(base.put_bytes),
        }
    }

    pub fn snapshot(&self) -> StoreSnapshot {
        StoreSnapshot {
            gets: self.gets(),
            puts: self.puts(),
            lists: self.lists(),
            deletes: self.deletes(),
            get_bytes: self.get_bytes(),
            put_bytes: self.put_bytes(),
        }
    }
}

/// A point-in-time copy of [`StoreMetrics`] counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct StoreSnapshot {
    pub gets: u64,
    pub puts: u64,
    pub lists: u64,
    pub deletes: u64,
    pub get_bytes: u64,
    pub put_bytes: u64,
}
