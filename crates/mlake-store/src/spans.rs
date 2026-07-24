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

/// One recorded object access.
struct Span {
    key: String,
    source: Source,
    op: &'static str,
    start_us: u64,
    end_us: u64,
    bytes: u64,
}

/// Max spans kept per request. Beyond this, accesses are counted (`dropped`) but not stored, so a
/// query that probes thousands of clusters stays bounded in the trace object.
pub const SPAN_CAP: usize = 400;

struct SpanBuf {
    start: Instant,
    spans: Vec<Span>,
    dropped: usize,
}

impl SpanBuf {
    fn new() -> Self {
        Self { start: Instant::now(), spans: Vec::new(), dropped: 0 }
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

/// Record one object access against the active collector. A no-op when no scope is active (e.g. a
/// background task) or the per-request cap is reached.
pub fn record(key: &str, source: Source, op: &'static str, start: Instant, end: Instant, bytes: u64) {
    let _ = CURRENT.try_with(|buf| {
        let mut b = buf.lock().unwrap();
        if b.spans.len() >= SPAN_CAP {
            b.dropped += 1;
            return;
        }
        let s0 = start.saturating_duration_since(b.start).as_micros() as u64;
        let s1 = end.saturating_duration_since(b.start).as_micros() as u64;
        b.spans.push(Span { key: key.to_string(), source, op, start_us: s0, end_us: s1, bytes });
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
                        "src": s.source.as_str(),
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
