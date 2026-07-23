//! Optional per-call tracing: an append-only JSONL audit log of every client call, with the
//! phase/timing/cache breakdown needed to explain why one read was instant and the next took
//! seconds.
//!
//! Off unless `MEMLAKE_TRACE_LOG=<path>` is set — zero overhead when off. When on, the request
//! path only builds a small JSON value and hands it to a background writer thread over an
//! unbounded channel, so tracing never adds latency to (or backpressures) the very call it is
//! measuring. The writer flushes each line, so a trace survives even if the server then stalls —
//! which is the whole point when chasing a hang.

use std::io::Write;
use std::sync::mpsc::{self, Sender};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// A tracer sink. `emit` is a no-op unless `MEMLAKE_TRACE_LOG` was set at startup.
#[derive(Clone)]
pub struct Tracer {
    tx: Option<Sender<serde_json::Value>>,
}

impl Tracer {
    /// Build from the environment. `MEMLAKE_TRACE_LOG=<path>` turns tracing on and spawns the
    /// background writer; anything else (unset/empty) yields a disabled tracer.
    pub fn from_env() -> Self {
        let path = match std::env::var("MEMLAKE_TRACE_LOG") {
            Ok(p) if !p.trim().is_empty() => p,
            _ => return Self { tx: None },
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
            Ok(_) => Self { tx: Some(tx) },
            Err(e) => {
                eprintln!("[trace] cannot start writer thread: {e}; tracing disabled");
                Self { tx: None }
            }
        }
    }

    /// Whether tracing is on — gate record-building behind this so a disabled tracer costs nothing.
    pub fn enabled(&self) -> bool {
        self.tx.is_some()
    }

    /// Hand a record to the background writer. Non-blocking: never waits on I/O, and drops the
    /// record only if the writer thread is gone.
    pub fn emit(&self, record: serde_json::Value) {
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
        let t = Tracer { tx: None };
        assert!(!t.enabled());
        t.emit(serde_json::json!({"op": "query"})); // must not panic
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
