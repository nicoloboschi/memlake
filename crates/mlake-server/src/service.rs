//! The gRPC service: a thin, stateless orchestration layer over `Namespace`/`Writer`/
//! `QueryNode`. Any replica can serve any request — all coordination is in object storage —
//! so this holds only process-local conveniences: a shared `Store` (its two-tier cache
//! warms across requests) and a per-namespace `Writer` so sequence claims don't self-contend
//! within a process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mlake_fts::Tokenizer;
use mlake_index::{index, Consistency, IndexOptions, QueryNode};
use mlake_store::{QueryMetrics, Store};
use mlake_wal::{Namespace, Writer};
use tokio::sync::Mutex as AsyncMutex;
use tonic::{Request, Response, Status};

use crate::pb::memlake_server::Memlake;
use crate::{convert, pb};

/// How long an EVENTUAL-consistency snapshot may be reused before re-opening. Bounds staleness
/// to roughly the indexing interval; STRONG never uses this (it checks the WAL head instead).
const EVENTUAL_TTL: Duration = Duration::from_secs(2);

/// A cached open snapshot of a namespace. Opening a `QueryNode` costs a manifest read plus WAL
/// head/tail scans (uncached, mutable) — re-doing that per query is what makes a naive server
/// slow. We keep the last snapshot and reuse it while it is still current: for STRONG, while
/// the WAL head has not advanced; for EVENTUAL, within a short TTL.
struct Snapshot {
    node: Arc<QueryNode>,
    through_seq: u64,
    opened: Instant,
}

pub struct MemlakeService {
    store: Store,
    tokenizer: Tokenizer,
    /// One `Writer` per namespace. The Writer caches the next WAL sequence, so serializing
    /// a namespace's writes through it avoids every request re-reading the head and racing
    /// on the same slot. Cross-namespace writes stay fully concurrent.
    writers: Mutex<HashMap<String, Arc<AsyncMutex<Writer>>>>,
    /// Last opened snapshot per namespace, reused across queries when still current.
    snapshots: Mutex<HashMap<String, Arc<Snapshot>>>,
}

impl MemlakeService {
    pub fn new(store: Store, tokenizer: Tokenizer) -> Self {
        Self {
            store,
            tokenizer,
            writers: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(HashMap::new()),
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

    fn cached_snapshot(&self, name: &str) -> Option<Arc<Snapshot>> {
        self.snapshots.lock().unwrap().get(name).cloned()
    }

    /// Get a current snapshot for `ns`, reusing the cached one when it is still valid.
    /// A write invalidates the cache for its namespace, so STRONG readers re-open promptly.
    async fn snapshot(
        &self,
        ns: &Namespace,
        consistency: Consistency,
    ) -> Result<Arc<Snapshot>, Status> {
        if let Some(cached) = self.cached_snapshot(&ns.name) {
            let reusable = match consistency {
                // EVENTUAL: reuse within the TTL, no storage check at all (0 roundtrips warm).
                Consistency::Eventual => cached.opened.elapsed() < EVENTUAL_TTL,
                // STRONG: reuse only if no write has landed since — one cheap head check.
                Consistency::Strong => {
                    let head = ns.wal_head().await.map_err(internal)?;
                    head == cached.through_seq
                }
            };
            if reusable {
                return Ok(cached);
            }
        }

        let node = Arc::new(
            QueryNode::open(ns, self.tokenizer.clone(), consistency)
                .await
                .map_err(internal)?,
        );
        let snap = Arc::new(Snapshot {
            through_seq: node.through_seq,
            node,
            opened: Instant::now(),
        });
        self.snapshots.lock().unwrap().insert(ns.name.clone(), snap.clone());
        Ok(snap)
    }

    fn invalidate(&self, name: &str) {
        self.snapshots.lock().unwrap().remove(name);
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
        // A new write means any cached snapshot is stale; drop it so the next STRONG read
        // re-opens and sees this write.
        self.invalidate(&req.namespace);
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
        // Reuse the cached open snapshot when it is still current (see `snapshot`): this turns
        // the common warm read into a pure in-memory fusion over the shared Store cache, with
        // 0 (EVENTUAL) or 1 (STRONG head-check) fresh roundtrips instead of a full re-open.
        let snap = self.snapshot(&ns, consistency).await?;
        let node = &snap.node;

        // Report the roundtrips this *query* consumed (the marginal read cost), not the
        // snapshot's one-time open cost — a warm cached read is 0.
        let metrics = QueryMetrics::new();
        let hits = node
            .query_metered(
                memory_type,
                vector.as_deref(),
                text.as_deref(),
                &tags,
                top_k,
                config,
                &metrics,
            )
            .await
            .map_err(internal)?;

        Ok(Response::new(pb::QueryResponse {
            hits: hits.into_iter().map(convert::hit).collect(),
            load_roundtrips: metrics.roundtrips() as u32,
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
pub async fn discover_namespaces(store: &Store) -> anyhow::Result<Vec<String>> {
    let keys = store.list("").await?;
    let mut names: Vec<String> = keys
        .into_iter()
        .filter_map(|k| k.strip_suffix("/manifest.json").map(|s| s.to_string()))
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}
