//! The gRPC service: a thin, stateless orchestration layer over `Namespace`/`Writer`/
//! `QueryNode`. Any replica can serve any request — all coordination is in object storage —
//! so this holds only process-local conveniences: a shared `Store` (its two-tier cache
//! warms across requests) and a per-namespace `Writer` so sequence claims don't self-contend
//! within a process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mlake_fts::Tokenizer;
use mlake_index::{index, IndexOptions, QueryNode};
use mlake_store::Store;
use mlake_wal::{Namespace, Writer};
use tokio::sync::Mutex as AsyncMutex;
use tonic::{Request, Response, Status};

use crate::pb::memlake_server::Memlake;
use crate::{convert, pb};

pub struct MemlakeService {
    store: Store,
    tokenizer: Tokenizer,
    /// One `Writer` per namespace. The Writer caches the next WAL sequence, so serializing
    /// a namespace's writes through it avoids every request re-reading the head and racing
    /// on the same slot. Cross-namespace writes stay fully concurrent.
    writers: Mutex<HashMap<String, Arc<AsyncMutex<Writer>>>>,
}

impl MemlakeService {
    pub fn new(store: Store, tokenizer: Tokenizer) -> Self {
        Self {
            store,
            tokenizer,
            writers: Mutex::new(HashMap::new()),
        }
    }

    fn namespace(&self, name: &str) -> Namespace {
        Namespace::new(name, self.store.clone())
    }

    fn writer_for(&self, name: &str) -> Arc<AsyncMutex<Writer>> {
        let mut writers = self.writers.lock().unwrap();
        writers
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(Writer::new(self.namespace(name)))))
            .clone()
    }
}

fn internal(e: impl std::fmt::Display) -> Status {
    Status::internal(e.to_string())
}

#[tonic::async_trait]
impl Memlake for MemlakeService {
    async fn create_namespace(
        &self,
        req: Request<pb::CreateNamespaceRequest>,
    ) -> Result<Response<pb::CreateNamespaceResponse>, Status> {
        let name = req.into_inner().namespace;
        if name.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        self.namespace(&name)
            .create_if_absent(&self.tokenizer.config_hash())
            .await
            .map_err(internal)?;
        Ok(Response::new(pb::CreateNamespaceResponse {}))
    }

    async fn write(
        &self,
        req: Request<pb::WriteRequest>,
    ) -> Result<Response<pb::WriteResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let ops = req
            .ops
            .into_iter()
            .map(convert::op)
            .collect::<Result<Vec<_>, _>>()?;
        if ops.is_empty() {
            return Err(Status::invalid_argument("write needs at least one op"));
        }

        let writer = self.writer_for(&req.namespace);
        // The commit is durable (S3 conditional PUT) before this returns — that is the ack.
        let result = {
            let mut w = writer.lock().await;
            w.commit(ops).await.map_err(internal)?
        };
        Ok(Response::new(pb::WriteResponse {
            seq: result.seq,
            attempts: result.attempts as u32,
        }))
    }

    async fn query(
        &self,
        req: Request<pb::QueryRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let memory_type = convert::memory_type_u8(req.memory_type)?;
        let vector = req.vector.as_ref().map(convert::decode_vector).transpose()?;
        let tags = convert::tag_filter(req.tags);
        let config = convert::query_config(req.config);
        let consistency = convert::consistency(req.consistency);
        let top_k = if req.top_k == 0 { 10 } else { req.top_k as usize };
        // Own every value before the await so no borrow of `req` crosses the await point.
        let text: Option<String> = if req.text.is_empty() { None } else { Some(req.text) };
        let ns = self.namespace(&req.namespace);
        // Open a fresh snapshot per query. The manifest read + tail scan are the only fresh
        // roundtrips; the per-type metadata (centroids, pk/radj indexes, FTS split) is served
        // from the shared Store cache after the first query, so repeat opens are cheap.
        let node = QueryNode::open(&ns, self.tokenizer.clone(), consistency)
            .await
            .map_err(internal)?;
        let load_roundtrips = node.load_roundtrips as u32;

        let hits = node
            .query(
                memory_type,
                vector.as_deref(),
                text.as_deref(),
                &tags,
                top_k,
                config,
            )
            .await
            .map_err(internal)?;

        Ok(Response::new(pb::QueryResponse {
            hits: hits.into_iter().map(convert::hit).collect(),
            load_roundtrips,
        }))
    }
}

/// Run the indexer over the given namespaces (or all discovered ones) on a fixed interval.
/// Idempotent by construction, so running more than one replica is safe.
pub async fn run_indexer(
    store: Store,
    tokenizer: Tokenizer,
    namespaces: Vec<String>,
    interval: std::time::Duration,
) -> anyhow::Result<()> {
    loop {
        let targets = if namespaces.is_empty() {
            discover_namespaces(&store).await.unwrap_or_default()
        } else {
            namespaces.clone()
        };
        for name in &targets {
            let ns = Namespace::new(name, store.clone());
            match index(&ns, &tokenizer, IndexOptions::default()).await {
                Ok(outcome) => tracing::info!(
                    namespace = name,
                    generation = outcome.generation,
                    docs = outcome.doc_count,
                    published = outcome.published,
                    "indexed"
                ),
                Err(e) => tracing::warn!(namespace = name, error = %e, "index failed"),
            }
        }
        tokio::time::sleep(interval).await;
    }
}

/// Namespaces are top-level prefixes that own a `manifest.json`.
async fn discover_namespaces(store: &Store) -> anyhow::Result<Vec<String>> {
    let keys = store.list("").await?;
    let mut names: Vec<String> = keys
        .into_iter()
        .filter_map(|k| k.strip_suffix("/manifest.json").map(|s| s.to_string()))
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}
