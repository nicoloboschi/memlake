//! Per-call tracing: a per-node JSONL view of every client call, with the phase/timing/cache
//! breakdown needed to explain why one read was instant and the next took seconds.
//!
//! **On by default.** Each record is buffered in memory and a background timer flushes the buffer to
//! object storage as an **append-only, immutable batch object** (`_obs/traces/{node}/{ms}-{seq}.jsonl`)
//! plus a small overwritten **rollup** (`_obs/rollup/{node}.json`) for the fleet overview. Batching
//! keeps the S3 PUT rate low; append-only means no trace is lost within the retention window; and the
//! buffer is size-bounded (it flushes on the tick, and a hard cap drops oldest only if S3 is
//! unreachable) so memory never grows unbounded. Retention is by TIME: the indexer periodically
//! deletes trace objects older than a configured window (default 24h) — see `mlake_index::gc`.
//!
//! `MEMLAKE_TRACE_LOG=off` (or `0`/`false`/`none`) disables it; any other value, or unset, leaves it
//! on. The request path only builds a small JSON value and appends it under a short-held lock — no
//! I/O — so tracing never adds latency to, or backpressures, the call it measures.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Reserved object-store root for observability data. Trace batches live under `{OBS_TRACES_PREFIX}`
/// and per-node rollups under `{OBS_ROLLUP_PREFIX}`. Both sit OUTSIDE any namespace prefix, so
/// namespace names starting with `_` are rejected to keep them unclaimable (see the service's
/// `reject_reserved_namespace`). The admin reads these prefixes directly from S3.
pub use mlake_core::{OBS_ROLLUP_PREFIX, OBS_TRACES_PREFIX};

/// The key for one immutable trace batch. `ms`+`seq` sort lexicographically by time (13-digit ms is
/// good past year 2286), so a LIST is time-ordered and time-retention is a key/age comparison.
pub fn obs_batch_path(node_id: &str, ms: u64, seq: u64) -> String {
    format!("{OBS_TRACES_PREFIX}{node_id}/{ms:013}-{seq:012}.jsonl")
}

/// The (overwritten) rollup object for one node — the fleet-overview heartbeat + stats.
pub fn obs_rollup_path(node_id: &str) -> String {
    format!("{OBS_ROLLUP_PREFIX}{node_id}.json")
}

/// Max bytes drained into a single batch object, so one flush can't write a huge object; a burst
/// bigger than this drains over successive ticks.
pub const BATCH_MAX_BYTES: usize = 4_000_000;
/// Hard cap on the in-memory pending buffer. Only reached if the flush task can't keep up (S3
/// unreachable); past it the oldest pending records are dropped and counted, so memory is bounded.
const BUFFER_HARD_BYTES: usize = 32_000_000;
/// Unbiased recent-latency sample kept for the rollup percentiles.
const ROLLUP_LAT_SAMPLE: usize = 2048;
/// Per-namespace latency sample for the rollup.
const ROLLUP_NS_LAT_SAMPLE: usize = 128;
/// Hard cap on namespaces tracked in the rollup.
const ROLLUP_NS_CAP: usize = 4096;

#[derive(Default)]
struct NsStat {
    count: u64,
    lat: VecDeque<f32>,
}

/// In-memory trace buffer: a FIFO of pending serialized records (drained into batch objects) plus
/// running aggregates for the rollup.
pub struct TraceBuffer {
    started_ms: u64,
    pending: VecDeque<(usize, String)>, // (byte len incl. newline, line)
    pending_bytes: usize,
    dropped: u64,
    total: u64,
    recent_lat: VecDeque<f32>,
    cache_hits: u64,
    cache_misses: u64,
    by_action: BTreeMap<String, u64>,
    by_ns: HashMap<String, NsStat>,
}

impl TraceBuffer {
    fn new() -> Self {
        Self {
            started_ms: now_ms(),
            pending: VecDeque::new(),
            pending_bytes: 0,
            dropped: 0,
            total: 0,
            recent_lat: VecDeque::new(),
            cache_hits: 0,
            cache_misses: 0,
            by_action: BTreeMap::new(),
            by_ns: HashMap::new(),
        }
    }

    /// Append one record: update the rollup aggregates and queue its serialized line for the next
    /// flush. If the pending buffer is over its hard cap (S3 not draining), drop the oldest.
    fn push(&mut self, rec: &serde_json::Value) {
        let total_ms = rec.get("total_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let line = rec.to_string();
        let bytes = line.len() + 1;

        self.total += 1;
        push_capped(&mut self.recent_lat, total_ms as f32, ROLLUP_LAT_SAMPLE);
        for key in ["io", "link_io"] {
            if let Some(io) = rec.get(key) {
                self.cache_hits += io.get("cache_hits").and_then(|v| v.as_u64()).unwrap_or(0);
                self.cache_misses += io.get("cache_misses").and_then(|v| v.as_u64()).unwrap_or(0);
            }
        }
        if let Some(action) =
            rec.get("snapshot").and_then(|s| s.get("action")).and_then(|v| v.as_str())
        {
            *self.by_action.entry(action.to_string()).or_insert(0) += 1;
        }
        if let Some(ns) = rec.get("namespace").and_then(|v| v.as_str()) {
            if self.by_ns.len() < ROLLUP_NS_CAP || self.by_ns.contains_key(ns) {
                let e = self.by_ns.entry(ns.to_string()).or_default();
                e.count += 1;
                push_capped(&mut e.lat, total_ms as f32, ROLLUP_NS_LAT_SAMPLE);
            }
        }

        self.pending_bytes += bytes;
        self.pending.push_back((bytes, line));
        while self.pending_bytes > BUFFER_HARD_BYTES {
            if let Some((b, _)) = self.pending.pop_front() {
                self.pending_bytes -= b;
                self.dropped += 1;
            } else {
                break;
            }
        }
    }

    /// Drain up to [`BATCH_MAX_BYTES`] of pending records into one JSONL blob, oldest-first. `None`
    /// when nothing is pending. Aggregates are NOT reset — they are cumulative for the rollup.
    pub fn take_batch(&mut self) -> Option<Vec<u8>> {
        if self.pending.is_empty() {
            return None;
        }
        let mut out = String::new();
        while let Some((b, line)) = self.pending.front() {
            if !out.is_empty() && out.len() + b > BATCH_MAX_BYTES {
                break;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
            self.pending_bytes -= b;
            self.pending.pop_front();
        }
        Some(out.into_bytes())
    }

    /// The rollup object for the fleet overview: heartbeat, cumulative totals, action mix, and the
    /// per-namespace rollup. Shape matches what the admin's Services view expects (`kind:"header"`).
    pub fn rollup_json(&self, node_id: &str) -> Vec<u8> {
        let now = now_ms();
        let uptime_ms = now.saturating_sub(self.started_ms);
        let mut lat: Vec<f32> = self.recent_lat.iter().copied().collect();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cache_total = self.cache_hits + self.cache_misses;
        let cache_hit =
            if cache_total > 0 { self.cache_hits as f64 / cache_total as f64 } else { 0.0 };

        let mut ns: Vec<(&String, &NsStat)> = self.by_ns.iter().collect();
        ns.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        let by_namespace: Vec<serde_json::Value> = ns
            .iter()
            .map(|(name, s)| {
                let mut l: Vec<f32> = s.lat.iter().copied().collect();
                l.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                serde_json::json!({
                    "ns": name,
                    "count": s.count,
                    "p50_ms": pct(&l, 50.0),
                    "p99_ms": pct(&l, 99.0),
                })
            })
            .collect();

        serde_json::json!({
            "kind": "header",
            "node_id": node_id,
            "updated_ms": now,
            "uptime_ms": uptime_ms,
            "totals": {
                "count": self.total,
                "qps": if uptime_ms > 0 { self.total as f64 * 1000.0 / uptime_ms as f64 } else { 0.0 },
                "p50_ms": pct(&lat, 50.0),
                "p99_ms": pct(&lat, 99.0),
                "cache_hit": cache_hit,
            },
            "by_action": self.by_action,
            "by_namespace": by_namespace,
            "pending": self.pending.len(),
            "dropped": self.dropped,
        })
        .to_string()
        .into_bytes()
    }
}

fn push_capped(buf: &mut VecDeque<f32>, v: f32, cap: usize) {
    buf.push_back(v);
    while buf.len() > cap {
        buf.pop_front();
    }
}

fn pct(sorted: &[f32], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

/// A tracer sink. `emit` appends each record to the buffer; a background timer flushes batches to
/// object storage. `emit` is a no-op when tracing is disabled.
#[derive(Clone)]
pub struct Tracer {
    /// In-memory append buffer flushed to object storage. `Some` exactly when tracing is enabled.
    buffer: Option<Arc<Mutex<TraceBuffer>>>,
}

impl Tracer {
    /// Build from the environment. Tracing is ON by default. `MEMLAKE_TRACE_LOG=off` (or `0`/`false`/
    /// `none`) turns it off; any other value, or unset, leaves it on.
    pub fn from_env() -> Self {
        let disabled = matches!(
            std::env::var("MEMLAKE_TRACE_LOG").ok().as_deref().map(str::trim).map(str::to_ascii_lowercase).as_deref(),
            Some("off") | Some("0") | Some("false") | Some("none") | Some("disabled")
        );
        if disabled {
            Self { buffer: None }
        } else {
            Self { buffer: Some(Arc::new(Mutex::new(TraceBuffer::new()))) }
        }
    }

    /// Whether tracing is on — gate record-building behind this so a disabled tracer costs nothing.
    pub fn enabled(&self) -> bool {
        self.buffer.is_some()
    }

    /// The in-memory buffer flushed to object storage, if tracing is on — for the periodic uploader.
    pub fn buffer(&self) -> Option<Arc<Mutex<TraceBuffer>>> {
        self.buffer.clone()
    }

    /// Append a record to the buffer. Non-blocking: an in-memory update under a short-held lock (no
    /// I/O), so the request path is never blocked on the network.
    pub fn emit(&self, record: serde_json::Value) {
        if let Some(buffer) = &self.buffer {
            if let Ok(mut b) = buffer.lock() {
                b.push(&record);
            }
        }
    }
}

/// Unix milliseconds now, for the trace timestamp.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Milliseconds elapsed since `since`, as an `f64` (sub-millisecond precision for fast calls).
pub fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_tracer_is_a_noop() {
        let t = Tracer { buffer: None };
        assert!(!t.enabled());
        assert!(t.buffer().is_none());
        t.emit(serde_json::json!({"op": "query"})); // must not panic
    }

    #[test]
    fn disabled_via_env() {
        std::env::set_var("MEMLAKE_TRACE_LOG", "off");
        let t = Tracer::from_env();
        std::env::remove_var("MEMLAKE_TRACE_LOG");
        assert!(!t.enabled());
    }

    #[test]
    fn appends_batches_and_rolls_up() {
        let mut b = TraceBuffer::new();
        for i in 0..2000u64 {
            b.push(&serde_json::json!({
                "op": "query", "namespace": "ns-a", "ts_ms": 1000 + i, "total_ms": 5.0,
                "io": {"cache_hits": 9, "cache_misses": 1},
                "snapshot": {"action": "reuse"},
            }));
        }
        // The rollup reflects every record (nothing dropped under the hard cap).
        let rollup: serde_json::Value =
            serde_json::from_slice(&b.rollup_json("memlake-serve-0")).unwrap();
        assert_eq!(rollup["totals"]["count"], 2000);
        assert_eq!(rollup["dropped"], 0);
        assert_eq!(rollup["by_action"]["reuse"], 2000);

        // Draining yields batches until empty; every record is accounted for (append-only, no loss).
        let mut lines = 0usize;
        while let Some(batch) = b.take_batch() {
            let text = String::from_utf8(batch).unwrap();
            for l in text.lines() {
                if !l.is_empty() {
                    let _: serde_json::Value = serde_json::from_str(l).unwrap();
                    lines += 1;
                }
            }
            assert!(text.len() <= BATCH_MAX_BYTES + 1024, "each batch is bounded");
        }
        assert_eq!(lines, 2000, "every appended record is flushed, none lost");
        assert!(b.take_batch().is_none(), "buffer is empty after draining");
    }

    #[test]
    fn batch_and_rollup_paths_sort_by_time() {
        assert!(obs_batch_path("n", 100, 1) < obs_batch_path("n", 200, 0));
        assert_eq!(obs_rollup_path("serve-0"), "_obs/rollup/serve-0.json");
    }
}
