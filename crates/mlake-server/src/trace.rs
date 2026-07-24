//! Per-call tracing: an append-only JSONL audit log of every client call, with the
//! phase/timing/cache breakdown needed to explain why one read was instant and the next took
//! seconds.
//!
//! **On by default.** Every record feeds a bounded in-memory ring ([`TraceRing`]) that the serve
//! node periodically uploads to `_obs/traces/{node_id}.jsonl` in object storage — a fixed-footprint,
//! admin-visible, per-node view that needs no log scraping and is safe on an ephemeral pod. A local
//! JSONL file is written IN ADDITION only when `MEMLAKE_TRACE_LOG` is set to an explicit path (local
//! debugging / `kubectl exec`); the unset default does NOT write an unbounded working-dir file.
//! `MEMLAKE_TRACE_LOG=off` (or `0`/`false`/`none`) disables everything, ring included.
//!
//! The request path only builds a small JSON value: the ring push is an in-memory update under a
//! short lock, and the (optional) file write is handed to a background thread over an unbounded
//! channel — so tracing never adds latency to, or backpressures, the very call it is measuring.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Where traces go when `MEMLAKE_TRACE_LOG` is unset — a file in the process's working directory.
/// In the perf docker stack the compose file points `MEMLAKE_TRACE_LOG` at the mounted `/traces`
/// instead, so the log is readable on the host.
pub const DEFAULT_TRACE_LOG: &str = "memlake-trace.jsonl";

/// Reserved object-store root for observability data — one bounded trace object per serve node lives
/// under it (`{OBS_TRACES_PREFIX}{node_id}.jsonl`). It sits OUTSIDE any namespace prefix, so
/// namespace names starting with `_` are rejected to keep it unclaimable (see the service's
/// `reject_reserved_namespace`). The admin reads this prefix directly from S3.
pub const OBS_TRACES_PREFIX: &str = "_obs/traces/";

/// The object key holding `node_id`'s recent traces.
pub fn obs_traces_path(node_id: &str) -> String {
    format!("{OBS_TRACES_PREFIX}{node_id}.jsonl")
}

// --- bounded, slow-biased ring uploaded to object storage --------------------------------------
//
// Two byte-capped tiers: a roomy one for "slow" records (the tail you debug) and a small one for
// fast records (recent context). Splitting them means a burst of fast calls evicts only from the
// fast tier, so the slow traces you actually care about survive within a fixed footprint.

/// Byte budget for the slow tier (the interesting tail).
const RING_SLOW_BYTES: usize = 1_500_000;
/// Byte budget for the fast tier (recent context).
const RING_FAST_BYTES: usize = 400_000;
/// A call at/above this total latency (ms) is "slow" and goes in the roomier, longer-lived tier.
const RING_SLOW_MS: f64 = 200.0;
/// Unbiased recent-latency sample kept for the header percentiles (independent of the slow bias).
const RING_LAT_SAMPLE: usize = 2048;
/// Per-namespace latency sample kept for the header's per-namespace rollup.
const RING_NS_LAT_SAMPLE: usize = 128;
/// Hard cap on namespaces tracked in the rollup, so a node churning through many stays bounded.
const RING_NS_CAP: usize = 4096;

struct RingRec {
    ts_ms: u64,
    bytes: usize,
    line: String,
}

#[derive(Default)]
struct NsStat {
    count: u64,
    lat: VecDeque<f32>,
}

/// An in-memory, byte-bounded window of recent trace records plus light running aggregates, rendered
/// to a single JSONL object (header line + records) and periodically uploaded to `_obs/traces/`.
pub struct TraceRing {
    started_ms: u64,
    slow: VecDeque<RingRec>,
    slow_bytes: usize,
    fast: VecDeque<RingRec>,
    fast_bytes: usize,
    total: u64,
    recent_lat: VecDeque<f32>,
    cache_hits: u64,
    cache_misses: u64,
    by_action: BTreeMap<String, u64>,
    by_ns: HashMap<String, NsStat>,
}

impl TraceRing {
    fn new() -> Self {
        Self {
            started_ms: now_ms(),
            slow: VecDeque::new(),
            slow_bytes: 0,
            fast: VecDeque::new(),
            fast_bytes: 0,
            total: 0,
            recent_lat: VecDeque::new(),
            cache_hits: 0,
            cache_misses: 0,
            by_action: BTreeMap::new(),
            by_ns: HashMap::new(),
        }
    }

    /// Fold one record into the ring: retain its serialized line (slow-biased) and update aggregates.
    fn push(&mut self, rec: &serde_json::Value) {
        let total_ms = rec.get("total_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ts_ms = rec.get("ts_ms").and_then(|v| v.as_u64()).unwrap_or_else(now_ms);
        let line = rec.to_string();
        let bytes = line.len() + 1;

        self.total += 1;
        push_capped(&mut self.recent_lat, total_ms as f32, RING_LAT_SAMPLE);
        // Cache hits/misses live under `io` (query) or `link_io` (write).
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
            if self.by_ns.len() < RING_NS_CAP || self.by_ns.contains_key(ns) {
                let e = self.by_ns.entry(ns.to_string()).or_default();
                e.count += 1;
                push_capped(&mut e.lat, total_ms as f32, RING_NS_LAT_SAMPLE);
            }
        }

        let r = RingRec { ts_ms, bytes, line };
        if total_ms >= RING_SLOW_MS {
            self.slow_bytes += bytes;
            self.slow.push_back(r);
            while self.slow_bytes > RING_SLOW_BYTES {
                if let Some(old) = self.slow.pop_front() {
                    self.slow_bytes -= old.bytes;
                } else {
                    break;
                }
            }
        } else {
            self.fast_bytes += bytes;
            self.fast.push_back(r);
            while self.fast_bytes > RING_FAST_BYTES {
                if let Some(old) = self.fast.pop_front() {
                    self.fast_bytes -= old.bytes;
                } else {
                    break;
                }
            }
        }
    }

    /// Render the ring to a JSONL object: a header line (node id, heartbeat, rollups) followed by the
    /// retained records, oldest-first. Always emits at least the header, so an idle node still
    /// heartbeats.
    pub fn render(&self, node_id: &str) -> Vec<u8> {
        let now = now_ms();
        let uptime_ms = now.saturating_sub(self.started_ms);
        let mut lat: Vec<f32> = self.recent_lat.iter().copied().collect();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cache_total = self.cache_hits + self.cache_misses;
        let cache_hit =
            if cache_total > 0 { self.cache_hits as f64 / cache_total as f64 } else { 0.0 };

        // Per-namespace rollup, busiest first.
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

        let header = serde_json::json!({
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
            "records": self.slow.len() + self.fast.len(),
        });

        // Merge the two tiers, oldest-first, so the admin reads a single time-ordered stream.
        let mut recs: Vec<&RingRec> = self.slow.iter().chain(self.fast.iter()).collect();
        recs.sort_by_key(|r| r.ts_ms);

        let mut out = header.to_string();
        for r in recs {
            out.push('\n');
            out.push_str(&r.line);
        }
        out.into_bytes()
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

/// A tracer sink. `emit` is a no-op unless `MEMLAKE_TRACE_LOG` was set at startup.
#[derive(Clone)]
pub struct Tracer {
    tx: Option<Sender<serde_json::Value>>,
    /// In-memory bounded window uploaded to object storage for the admin UI (fleet-wide view without
    /// scraping individual pods). `Some` exactly when tracing is enabled.
    ring: Option<Arc<Mutex<TraceRing>>>,
}

impl Tracer {
    /// Build from the environment. Tracing is ON by default, feeding the bounded in-memory ring that
    /// is uploaded to `_obs/traces/` — a fixed-footprint, admin-visible view that is safe on an
    /// ephemeral pod. A local JSONL file is written ONLY when `MEMLAKE_TRACE_LOG` is an explicit path
    /// (for `kubectl exec` / local debugging); the default no longer writes an unbounded file to the
    /// working directory, which on k8s would grow without limit. `MEMLAKE_TRACE_LOG=off` (or
    /// `0`/`false`/`none`) turns everything off, including the ring upload.
    pub fn from_env() -> Self {
        let path = match std::env::var("MEMLAKE_TRACE_LOG") {
            // Explicit opt-out — no ring, no file, no upload.
            Ok(v)
                if matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "off" | "0" | "false" | "none" | "disabled"
                ) =>
            {
                return Self { tx: None, ring: None }
            }
            // Explicit path: ring + a local file at that path.
            Ok(v) if !v.trim().is_empty() => v,
            // Unset or empty: default ON, ring-only (bounded, uploaded to object storage) — no
            // unbounded local file.
            _ => return Self { tx: None, ring: Some(Arc::new(Mutex::new(TraceRing::new()))) },
        };
        let (tx, rx) = mpsc::channel::<serde_json::Value>();
        let spawned = std::thread::Builder::new()
            .name("memlake-trace".into())
            .spawn(move || {
                let mut file = match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("[trace] cannot open {path}: {e}; tracing disabled");
                        return;
                    }
                };
                // Ends when the last `Sender` (held by the service) drops.
                for rec in rx {
                    if let Ok(line) = serde_json::to_string(&rec) {
                        let _ = writeln!(file, "{line}");
                        let _ = file.flush();
                    }
                }
            });
        match spawned {
            Ok(_) => Self { tx: Some(tx), ring: Some(Arc::new(Mutex::new(TraceRing::new()))) },
            Err(e) => {
                eprintln!("[trace] cannot start writer thread: {e}; tracing disabled");
                Self { tx: None, ring: None }
            }
        }
    }

    /// Whether tracing is on (ring and/or local file) — gate record-building behind this so a
    /// disabled tracer costs nothing.
    pub fn enabled(&self) -> bool {
        self.ring.is_some()
    }

    /// The in-memory ring uploaded to object storage, if tracing is on — for the periodic uploader.
    pub fn ring(&self) -> Option<Arc<Mutex<TraceRing>>> {
        self.ring.clone()
    }

    /// Hand a record to the background file writer AND fold it into the object-storage ring.
    /// Non-blocking: the file path never waits on I/O (channel send); the ring push is an in-memory
    /// update under a short-held lock (no I/O), so the request path is never blocked on the network.
    pub fn emit(&self, record: serde_json::Value) {
        if let Some(ring) = &self.ring {
            if let Ok(mut r) = ring.lock() {
                r.push(&record);
            }
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(record);
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
        let t = Tracer { tx: None, ring: None };
        assert!(!t.enabled());
        assert!(t.ring().is_none());
        t.emit(serde_json::json!({"op": "query"})); // must not panic
    }

    #[test]
    fn ring_is_slow_biased_and_bounded_and_renders_header() {
        let mut ring = TraceRing::new();
        // A flood of fast records must not evict a slow one, and the object stays bounded.
        for i in 0..20_000u64 {
            ring.push(&serde_json::json!({
                "op": "query", "namespace": "ns-a", "ts_ms": 1000 + i, "total_ms": 5.0,
                "io": {"cache_hits": 9, "cache_misses": 1},
            }));
        }
        ring.push(&serde_json::json!({
            "op": "query", "namespace": "ns-b", "ts_ms": 500, "total_ms": 4200.0,
            "snapshot": {"action": "reopen_fold"},
        }));
        for i in 0..20_000u64 {
            ring.push(&serde_json::json!({
                "op": "query", "namespace": "ns-a", "ts_ms": 30_000 + i, "total_ms": 5.0,
            }));
        }

        let body = ring.render("memlake-serve-0");
        assert!(body.len() < 2_500_000, "object stays bounded to a few MB");
        let text = String::from_utf8(body).unwrap();
        let mut lines = text.lines();
        let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(header["kind"], "header");
        assert_eq!(header["node_id"], "memlake-serve-0");
        assert_eq!(header["totals"]["count"], 40_001);
        // The lone slow record survived the fast flood (slow tier is not evicted by fast records).
        assert!(text.contains("reopen_fold"), "the slow trace is retained through the flood");
        // Per-namespace rollup is present for both namespaces.
        let by_ns = header["by_namespace"].as_array().unwrap();
        assert!(by_ns.iter().any(|n| n["ns"] == "ns-a"));
        assert!(by_ns.iter().any(|n| n["ns"] == "ns-b"));
    }

    #[test]
    fn enabled_tracer_appends_one_json_line_per_record() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("memlake-trace-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        std::env::set_var("MEMLAKE_TRACE_LOG", &path);
        let t = Tracer::from_env();
        std::env::remove_var("MEMLAKE_TRACE_LOG");
        assert!(t.enabled());

        t.emit(serde_json::json!({"op": "query", "total_ms": 1.5}));
        t.emit(serde_json::json!({"op": "write", "total_ms": 2.0}));
        drop(t); // close the channel so the writer thread finishes and flushes

        // Give the background writer a moment to drain (it flushes per line).
        std::thread::sleep(std::time::Duration::from_millis(200));
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "one JSON line per record");
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["op"], "query");
        let _ = std::fs::remove_file(&path);
    }
}
