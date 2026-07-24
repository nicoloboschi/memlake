//! Per-request object-access spans: a fine-grained timeline of every read/write the store did while
//! serving one request, each tagged with the object key, the tier that answered (memory cache / disk
//! cache / S3), and start→end microseconds relative to the request. The admin renders these as a
//! waterfall so a slow request explains itself ("manifest GET on S3 142→310ms, then 8 clusters from
//! disk cache in parallel").
//!
//! Collection rides a **tokio task-local**, so the store records into it with no extra argument
//! threaded through the whole read path — and it therefore captures accesses that carry no metrics
//! context (the snapshot's head-pointer and manifest GETs). A request enters a scope with
//! [`scope`]; every same-task future under it records automatically. Work moved onto a *spawned*
//! task does not inherit the scope (a known gap, noted where it matters).
//!
//! Always on, but **capped** ([`SPAN_CAP`]) so a fan-out-heavy request can't unbound the trace
//! record; overflow is counted, not kept.

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Which tier answered a read (or, for a write, that it went to the object store).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Served from the in-memory cache ring — no syscall.
    Mem,
    /// Served from the on-disk (NVMe) cache tier — a local file read.
    Disk,
    /// Went to the object store (a real round trip).
    S3,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Mem => "mem",
            Source::Disk => "disk",
            Source::S3 => "s3",
        }
    }
}

/// One recorded span — an object access (`kind = "io"`) or a unit of compute (`kind = "compute"`,
/// e.g. a segment's rerank). Both live on the same request timeline, grouped by `group` (the main
/// phase: snapshot / recall / rerank / graph / text / commit) so the admin can draw one waterfall.
struct Span {
    key: String,
    /// Which tier answered — only meaningful for `kind = "io"`.
    source: Option<Source>,
    kind: &'static str,
    group: &'static str,
    op: &'static str,
    start_us: u64,
    end_us: u64,
    bytes: u64,
}

/// The main phase an object key belongs to, for grouping the waterfall. A heuristic on the key +
/// op — good enough to cluster the timeline; the compute spans carry their group explicitly.
fn group_for(key: &str, op: &str) -> &'static str {
    if op.starts_with("put") || op == "cas_swap" {
        return "commit";
    }
    if key.contains("rerank") {
        return "rerank";
    }
    if key.contains("radj") || key.ends_with(".pk") {
        return "graph";
    }
    // Cluster reads + payload materialization (result hydration / derive corpus reads).
    if key.contains("cluster-")
        || key.ends_with(".vec")
        || key.ends_with(".bin")
        || key.contains("payload")
    {
        return "recall";
    }
    // WAL head/tail + manifest + the per-segment metadata a snapshot open loads.
    "snapshot"
}

/// Max spans kept per request. Beyond this, accesses are counted (`dropped`) but not stored, so a
/// query that probes thousands of clusters stays bounded in the trace object.
pub const SPAN_CAP: usize = 400;

struct SpanBuf {
    start: Instant,
    spans: Vec<Span>,
    io_len: usize,
    dropped: usize,
}

impl SpanBuf {
    fn new() -> Self {
        Self { start: Instant::now(), spans: Vec::new(), io_len: 0, dropped: 0 }
    }
}

tokio::task_local! {
    static CURRENT: Arc<Mutex<SpanBuf>>;
}

/// Run `f` with a fresh span collector active for its task. Read the collected spans with
/// [`current_json`] from inside `f` (before it returns — the collector lives only for the scope).
pub async fn scope<F: Future>(f: F) -> F::Output {
    CURRENT.scope(Arc::new(Mutex::new(SpanBuf::new())), f).await
}

/// Record one object access (an `io` span) against the active collector. A no-op when no scope is
/// active (e.g. a background task) or the per-request cap is reached.
pub fn record(key: &str, source: Source, op: &'static str, start: Instant, end: Instant, bytes: u64) {
    let group = group_for(key, op);
    push(key.to_string(), Some(source), "io", group, op, start, end, bytes);
}

/// Record a unit of compute (a `compute` span) — e.g. one segment's rerank — with its main-phase
/// `group`. Same timeline as the object accesses; segments running concurrently produce overlapping
/// spans, which is exactly the parallelism worth seeing.
pub fn compute(name: &'static str, group: &'static str, start: Instant, end: Instant) {
    push(name.to_string(), None, "compute", group, "compute", start, end, 0);
}

#[allow(clippy::too_many_arguments)]
fn push(
    key: String,
    source: Option<Source>,
    kind: &'static str,
    group: &'static str,
    op: &'static str,
    start: Instant,
    end: Instant,
    bytes: u64,
) {
    let _ = CURRENT.try_with(|buf| {
        let mut b = buf.lock().unwrap();
        // The cap applies to I/O spans only. Compute spans (per-segment phases) are inherently few
        // and are the point of the breakdown, so they're never crowded out by a cold query's flood
        // of cluster reads.
        if kind == "io" {
            if b.io_len >= SPAN_CAP {
                b.dropped += 1;
                return;
            }
            b.io_len += 1;
        }
        let s0 = start.saturating_duration_since(b.start).as_micros() as u64;
        let s1 = end.saturating_duration_since(b.start).as_micros() as u64;
        b.spans.push(Span { key, source, kind, group, op, start_us: s0, end_us: s1, bytes });
    });
}

/// The active collector's spans as JSON for the trace record, or `None` if no scope is active.
/// `{ items: [{key, src, op, start_us, end_us, bytes}], dropped, count }`.
pub fn current_json() -> Option<serde_json::Value> {
    CURRENT
        .try_with(|buf| {
            let b = buf.lock().unwrap();
            let items: Vec<serde_json::Value> = b
                .spans
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "key": s.key,
                        "src": s.source.map(Source::as_str),
                        "kind": s.kind,
                        "group": s.group,
                        "op": s.op,
                        "start_us": s.start_us,
                        "end_us": s.end_us,
                        "bytes": s.bytes,
                    })
                })
                .collect();
            serde_json::json!({ "items": items, "dropped": b.dropped, "count": b.spans.len() })
        })
        .ok()
}
