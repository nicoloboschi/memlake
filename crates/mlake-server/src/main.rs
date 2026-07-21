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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("serve");

    match mode {
        "serve" => serve(&args).await,
        "index" => indexer(&args).await,
        other => {
            eprintln!("unknown mode '{other}'. usage: mlake-server [serve|index] [flags]");
            std::process::exit(2);
        }
    }
}

/// Build the base object store from the environment. Same knobs as the perf harness so a
/// local MinIO works out of the box.
fn build_store() -> Result<Store> {
    let endpoint = std::env::var("MEMLAKE_S3_ENDPOINT").ok();
    let bucket = std::env::var("MEMLAKE_S3_BUCKET").unwrap_or_else(|_| "memlake".into());
    let access = std::env::var("MEMLAKE_S3_ACCESS_KEY").unwrap_or_else(|_| "memlake".into());
    let secret = std::env::var("MEMLAKE_S3_SECRET_KEY").unwrap_or_else(|_| "memlake123".into());
    let region = std::env::var("MEMLAKE_S3_REGION").unwrap_or_else(|_| "us-east-1".into());
    // Default to a local MinIO endpoint only when none is configured; in AWS you leave
    // MEMLAKE_S3_ENDPOINT unset and object_store talks to real S3.
    let endpoint = endpoint.or_else(|| Some("http://localhost:9000".into()));
    Store::s3(&bucket, endpoint.as_deref(), &access, &secret, &region)
        .context("building object store (check MEMLAKE_S3_* env)")
}

async fn serve(args: &[String]) -> Result<()> {
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

    tracing::info!(%addr, mem_mb, disk_mb, "memlake serving gRPC");
    Server::builder()
        .add_service(MemlakeServer::new(svc))
        .serve(addr)
        .await
        .context("gRPC server")?;
    Ok(())
}

async fn indexer(args: &[String]) -> Result<()> {
    let namespaces: Vec<String> = flag(args, "--namespaces")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_default();
    let store = build_store()?;

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
    )
    .await
}

/// One metered index pass over the given namespaces (or all discovered), printing a single
/// JSON line the benchmark parses: elapsed, store ops, stored bytes, docs.
async fn index_once(store: Store, namespaces: Vec<String>) -> Result<()> {
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
        let outcome = index(&ns, &Tokenizer::default(), IndexOptions::default()).await?;
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
