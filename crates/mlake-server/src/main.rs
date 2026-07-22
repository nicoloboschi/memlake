//! memlake server. One binary, two deployment modes (see `docs/DEPLOYMENT.md`):
//!
//!   mlake-server serve  [--addr 0.0.0.0:50051]
//!   mlake-server index  [--namespaces a,b] [--interval-secs 5]
//!
//! `serve` is the stateless gRPC API (scale to N replicas behind one k8s Service). `index`
//! runs the async, idempotent indexer loop (its own Deployment). Both read S3 config from
//! the environment and touch the same bucket.

mod convert;
mod pb;
mod service;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use mlake_fts::Tokenizer;
use mlake_store::{DiskCache, Store, StoreMetrics};
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

/// Build the base object store from the environment (`.env` is loaded in `main`). See
/// [`Store::from_env`] for the full variable list; standard `AWS_*` credentials work as-is.
fn build_store() -> Result<Store> {
    Store::from_env().context("building object store (check MEMLAKE_S3_*/AWS_* env or .env)")
}

async fn serve(args: &[String], node: String) -> Result<()> {
    let addr = flag(args, "--addr")
        .unwrap_or_else(|| "0.0.0.0:50051".into())
        .parse()
        .context("parsing --addr")?;

    // Attach a bounded two-tier read cache so the process has predictable memory + disk
    // footprint (both capped by construction). Defaults are conservative; a benchmark or a
    // memory-constrained pod tunes them down.
    let mut store = build_store()?;
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
    let svc = MemlakeService::new(store, Tokenizer::default());

    tracing::info!(%addr, mem_mb, disk_mb, node = %node, "memlake serving gRPC");
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
    let store = build_store()?;

    // `--once`: run a single metered index pass, print a JSON summary, and exit. This is the
    // benchmark's index phase — the harness reads the summary for build time and cost, so it
    // never needs an in-process Rust engine.
    if args.iter().any(|a| a == "--once") {
        return index_once(store, namespaces, args.iter().any(|a| a == "--streaming")).await;
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
    )
    .await
}

/// One metered index pass over the given namespaces (or all discovered), printing a single
/// JSON line the benchmark parses: elapsed, store ops, stored bytes, docs.
async fn index_once(store: Store, namespaces: Vec<String>, streaming: bool) -> Result<()> {
    use mlake_index::{index, IndexOptions};
    use mlake_wal::Namespace;

    let metrics = StoreMetrics::new();
    let store = store.with_store_metrics(metrics.clone());
    let base = metrics.snapshot();
    let start = std::time::Instant::now();

    let targets = if namespaces.is_empty() {
        service::discover_namespaces(&store).await?
    } else {
        namespaces
    };
    let mut docs = 0usize;
    for name in &targets {
        let ns = Namespace::new(name, store.clone());
        // `--streaming`: the external-memory fold — bounded RAM (spill + external sort) for a
        // corpus too large to fold in memory, at the cost of a slower build. The default
        // in-RAM fold is faster for anything that fits.
        let outcome = if streaming {
            mlake_index::streaming::index_streaming(&ns, &Tokenizer::default(), IndexOptions::default()).await?
        } else {
            index(&ns, &Tokenizer::default(), IndexOptions::default()).await?
        };
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
