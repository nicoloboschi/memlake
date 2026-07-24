//! memlake server. One binary, two deployment modes (see `docs/DEPLOYMENT.md`):
//!
//!   mlake-server serve  [--addr 0.0.0.0:50051]
//!   mlake-server index  [--namespaces a,b] [--interval-secs 5]
//!
//! `serve` is the stateless gRPC API (scale to N replicas behind one k8s Service). `index`
//! runs the async, idempotent indexer loop (its own Deployment). Both read S3 config from
//! the environment and touch the same bucket.

mod convert;
mod limiter;
mod objects;
mod pb;
mod service;
mod trace;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use mlake_fts::Tokenizer;
use mlake_store::{DiskCache, S3Config, Store, StoreMetrics};
use tonic::transport::Server;

use pb::memlake_server::MemlakeServer;
use service::{run_indexer, MemlakeService};

#[tokio::main]
async fn main() -> Result<()> {
    // Load `.env` from the working dir (or any ancestor) before anything reads config. Real
    // process env always wins over the file, so a deploy can override without editing it.
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("serve");

    // A stable-ish identity for this process, stamped on every log line so two- or three-node
    // behavior is debuggable (whose fold, whose ack, whose skip). `--node-id` / MEMLAKE_NODE_ID
    // override; otherwise it is `{host}-{pid}`.
    let node = node_id(&args);
    let span = tracing::info_span!("node", id = %node);

    use tracing::Instrument;
    match mode {
        "serve" => serve(&args, node).instrument(span).await,
        "index" => indexer(&args, node).instrument(span).await,
        other => {
            eprintln!("unknown mode '{other}'. usage: mlake-server [serve|index] [flags]");
            std::process::exit(2);
        }
    }
}

/// This process's node identity: `--node-id`, else `MEMLAKE_NODE_ID`, else `{host}-{pid}`.
fn node_id(args: &[String]) -> String {
    if let Some(id) = flag(args, "--node-id") {
        return id;
    }
    if let Ok(id) = std::env::var("MEMLAKE_NODE_ID") {
        if !id.is_empty() {
            return id;
        }
    }
    let host = std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "node".into());
    format!("{host}-{}", std::process::id())
}

/// The lease holder for this process: the node id plus the pid, so two processes that happen
/// to share a `--node-id` still hold distinct leases (and each only releases its own).
fn lease_holder(node: &str) -> String {
    format!("{node}#{}", std::process::id())
}

/// Read a service's object-store config from its own `MEMLAKE_<SERVICE>_S3_*` block. Each service
/// (query vs indexer) owns a complete, self-describing config — service-prefixed and required, no
/// shared or `AWS_*` fallback. `prefix` is e.g. `MEMLAKE_QUERY` or `MEMLAKE_INDEXER`.
fn s3_config(prefix: &str) -> Result<S3Config> {
    let req = |suffix: &str| -> Result<String> {
        let name = format!("{prefix}_{suffix}");
        std::env::var(&name).ok().filter(|v| !v.is_empty()).with_context(|| format!("{name} is required"))
    };
    let opt = |suffix: &str| std::env::var(format!("{prefix}_{suffix}")).ok().filter(|v| !v.is_empty());
    Ok(S3Config {
        bucket: req("S3_BUCKET")?,
        endpoint: opt("S3_ENDPOINT"),
        access_key: req("S3_ACCESS_KEY")?,
        secret_key: req("S3_SECRET_KEY")?,
        region: opt("S3_REGION").unwrap_or_else(|| "us-east-1".into()),
    })
}

/// Build the object store for `prefix`'s service from its config block (`.env` loaded in `main`).
fn build_store(prefix: &str) -> Result<Store> {
    Store::from_s3_config(&s3_config(prefix)?)
        .with_context(|| format!("building object store (check {prefix}_S3_* env or .env)"))
}

/// The indexer's per-stage streaming-fold budget, from `MEMLAKE_INDEXER_FOLD_*` (used only when
/// `fold` picks the streaming path). Falls back to [`FoldBudget::default`] per stage.
fn indexer_fold_budget() -> mlake_index::streaming::FoldBudget {
    let d = mlake_index::streaming::FoldBudget::default();
    let mb = |suffix: &str, default: usize| {
        std::env::var(format!("MEMLAKE_INDEXER_FOLD_{suffix}"))
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(default)
    };
    mlake_index::streaming::FoldBudget {
        resolve_mb: mb("RESOLVE_MB", d.resolve_mb),
        cluster_mb: mb("CLUSTER_MB", d.cluster_mb),
        payload_mb: mb("PAYLOAD_MB", d.payload_mb),
        index_mb: mb("INDEX_MB", d.index_mb),
        radj_mb: mb("RADJ_MB", d.radj_mb),
        fts_mb: mb("FTS_MB", d.fts_mb),
    }
}

/// The corpus-size cutoff for `fold`'s in-RAM-vs-streaming choice, from
/// `MEMLAKE_INDEXER_STREAMING_THRESHOLD` (docs); defaults to [`DEFAULT_STREAMING_THRESHOLD_DOCS`].
fn indexer_streaming_threshold() -> usize {
    std::env::var("MEMLAKE_INDEXER_STREAMING_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(mlake_index::DEFAULT_STREAMING_THRESHOLD_DOCS)
}

/// Per-pass folded-doc count above which the indexer treats a namespace as "under load" and comes
/// back on a short tick to keep draining, instead of waiting the full `--interval-secs`. This is
/// the size-triggered flush that keeps a bulk write from accumulating an unbounded un-indexed tail.
/// From `MEMLAKE_INDEXER_TAIL_FLUSH_DOCS`; default 20_000.
fn indexer_tail_flush_docs() -> usize {
    std::env::var("MEMLAKE_INDEXER_TAIL_FLUSH_DOCS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(20_000)
}

/// How often the indexer runs GC per namespace to reclaim unreferenced objects (folded WAL
/// entries, compacted-away segments). Throttled below `gc`'s own min-age guard, so it stays a
/// cheap mostly-no-op LIST. From `MEMLAKE_INDEXER_GC_INTERVAL_SECS`; default 300 (5 min).
fn indexer_gc_interval() -> Duration {
    let secs = std::env::var("MEMLAKE_INDEXER_GC_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(300);
    Duration::from_secs(secs)
}

/// The min-age GC guard: an unreferenced object younger than this is kept, so GC never deletes the
/// in-flight output of a concurrent fold that has not yet published its manifest. Default 900s
/// (15 min) matches `mlake_index::DEFAULT_MIN_AGE`; lower it (e.g. for a demo) to reclaim sooner.
/// From `MEMLAKE_INDEXER_GC_MIN_AGE_SECS`.
fn indexer_gc_min_age() -> Duration {
    let secs = std::env::var("MEMLAKE_INDEXER_GC_MIN_AGE_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(900);
    Duration::from_secs(secs)
}

async fn serve(args: &[String], node: String) -> Result<()> {
    let addr = flag(args, "--addr")
        .unwrap_or_else(|| "0.0.0.0:50051".into())
        .parse()
        .context("parsing --addr")?;

    // Attach a bounded two-tier read cache so the process has predictable memory + disk
    // footprint (both capped by construction). Defaults are conservative; a benchmark or a
    // memory-constrained pod tunes them down.
    let mut store = build_store("MEMLAKE_QUERY")?;
    let mem_mb = flag(args, "--mem-mb").and_then(|s| s.parse::<u64>().ok()).unwrap_or(256);
    let disk_mb = flag(args, "--disk-mb").and_then(|s| s.parse::<u64>().ok()).unwrap_or(4096);
    let cache_dir = flag(args, "--cache-dir")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("memlake-cache"));
    let cache = Arc::new(
        DiskCache::with_budgets(cache_dir, mem_mb * 1_000_000, disk_mb * 1_000_000)
            .context("opening read cache")?,
    );
    store = store.with_cache(cache);
    // Cap concurrent retrieval requests so peak query memory is `max_queries × per-query working
    // set`, not unbounded in request concurrency. `--max-concurrent-queries` or
    // `MEMLAKE_MAX_CONCURRENT_QUERIES`; excess requests queue rather than erroring.
    let max_queries = flag(args, "--max-concurrent-queries")
        .or_else(|| std::env::var("MEMLAKE_QUERY_MAX_CONCURRENT").ok())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(service::DEFAULT_MAX_CONCURRENT_QUERIES);
    let svc = MemlakeService::new(store, Tokenizer::default()).with_max_concurrent_queries(max_queries);

    tracing::info!(
        %addr, mem_mb, disk_mb, max_queries = svc.max_concurrent_queries(),
        trace_log = svc.tracing_enabled(), node = %node,
        "memlake serving gRPC"
    );
    // Publish this node's bounded trace ring to `_obs/traces/{node}.jsonl` for the admin's
    // fleet-wide view (no-op when tracing is off). A stable node id (StatefulSet ordinal) keeps the
    // object key stable across restarts.
    svc.spawn_trace_uploader(node.clone());
    Server::builder()
        .add_service(MemlakeServer::new(svc))
        .serve(addr)
        .await
        .context("gRPC server")?;
    Ok(())
}

async fn indexer(args: &[String], node: String) -> Result<()> {
    let namespaces: Vec<String> = flag(args, "--namespaces")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_default();
    let store = build_store("MEMLAKE_INDEXER")?;

    // `--once`: run a single metered index pass, print a JSON summary, and exit. This is the
    // benchmark's index phase — the harness reads the summary for build time and cost, so it
    // never needs an in-process Rust engine.
    if args.iter().any(|a| a == "--once") {
        return index_once(store, namespaces).await;
    }

    let interval = flag(args, "--interval-secs")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5);
    tracing::info!(
        ?namespaces,
        interval_secs = interval,
        "memlake indexer loop (empty namespaces => discover all)"
    );
    run_indexer(
        store,
        Tokenizer::default(),
        namespaces,
        Duration::from_secs(interval),
        lease_holder(&node),
        indexer_fold_budget(),
        indexer_streaming_threshold(),
        indexer_tail_flush_docs(),
        indexer_gc_interval(),
        indexer_gc_min_age(),
    )
    .await
}

/// One metered index pass over the given namespaces (or all discovered), printing a single
/// JSON line the benchmark parses: elapsed, store ops, stored bytes, docs.
async fn index_once(store: Store, namespaces: Vec<String>) -> Result<()> {
    use mlake_index::{fold, IndexOptions};
    use mlake_wal::Namespace;

    let metrics = StoreMetrics::new();
    let store = store.with_store_metrics(metrics.clone());
    let base = metrics.snapshot();
    let start = std::time::Instant::now();
    let budget = indexer_fold_budget();
    let threshold = indexer_streaming_threshold();

    let targets = if namespaces.is_empty() {
        service::discover_namespaces(&store).await?
    } else {
        namespaces
    };
    let mut docs = 0usize;
    for name in &targets {
        let ns = Namespace::new(name, store.clone());
        // `fold` auto-selects the in-RAM fold (incremental, links) below the streaming threshold
        // and the bounded external-memory fold above it (see MEMLAKE_INDEXER_STREAMING_THRESHOLD).
        let outcome =
            fold(&ns, &Tokenizer::default(), IndexOptions::default(), budget, threshold).await?;
        docs += outcome.doc_count;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let s = metrics.since(&base);
    println!(
        "{{\"elapsed_s\":{:.4},\"puts\":{},\"lists\":{},\"gets\":{},\"put_bytes\":{},\"get_bytes\":{},\"docs\":{}}}",
        elapsed, s.puts, s.lists, s.gets, s.put_bytes, s.get_bytes, docs
    );
    Ok(())
}

/// Minimal `--flag value` lookup; keeps the binary dependency-light.
fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}
