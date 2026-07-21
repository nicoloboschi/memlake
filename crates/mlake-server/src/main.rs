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

use std::time::Duration;

use anyhow::{Context, Result};
use mlake_fts::Tokenizer;
use mlake_store::Store;
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

/// Build the shared object store from the environment. Same knobs as the perf harness so a
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
    let store = build_store()?;
    let svc = MemlakeService::new(store, Tokenizer::default());

    tracing::info!(%addr, "memlake serving gRPC");
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
    let interval = flag(args, "--interval-secs")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5);
    let store = build_store()?;

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

/// Minimal `--flag value` lookup; keeps the binary dependency-light.
fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}
