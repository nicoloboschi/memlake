//! The gRPC service: a thin, stateless orchestration layer over `Namespace`/`Writer`/
//! `QueryNode`. Any replica can serve any request — all coordination is in object storage —
//! so this holds only process-local conveniences: a shared `Store` (its two-tier cache
//! warms across requests) and a per-namespace `Writer` so sequence claims don't self-contend
//! within a process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mlake_fts::Tokenizer;
use mlake_index::streaming::FoldBudget;
use mlake_index::{fold, IndexOptions, QueryNode, ScanCursor};
use mlake_store::{QueryMetrics, Store};
use mlake_wal::{IndexQueue, Namespace, Writer};
use tokio::sync::Mutex as AsyncMutex;
use tonic::{Request, Response, Status};

use crate::limiter::QueryLimiter;
use crate::pb::memlake_server::Memlake;
use crate::trace::{ms, now_ms, Tracer};
use crate::{convert, objects, pb};

/// How long a `wait_for_index` write waits for the *background indexer* to fold its sequence into
/// a segment, and how often it re-checks the manifest cursor while waiting. A serve replica NEVER
/// folds itself — folding is the indexer Deployment's job, with its own memory budget and
/// scheduling — so this just polls the cursor. If the deadline passes the write is still durable
/// and already visible via the WAL tail; only the "it's in a segment" confirmation timed out.
const WAIT_FOR_INDEX_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const WAIT_FOR_INDEX_POLL: std::time::Duration = std::time::Duration::from_millis(200);

/// A cached open snapshot of a namespace. Opening a `QueryNode` costs a manifest read plus WAL
/// head/tail scans (uncached, mutable) — re-doing that per query is what makes a naive server
/// slow. We keep the last snapshot and reuse it while the WAL head has not advanced, so a run
/// of reads between writes is served from memory with a single cheap head check.
struct Snapshot {
    node: Arc<QueryNode>,
    through_seq: u64,
}

/// What `snapshot_traced` did, for the per-call trace.
struct SnapshotOutcome {
    /// `reuse` (cached, head unchanged), `reopen_tail` (a write; segments reused), `reopen_fold`
    /// (a fold changed the manifest — adopt the new generation, warming its cold clusters off the
    /// request path), or `full_open` (no cache).
    action: &'static str,
    /// Time spent in the snapshot step (head check + any reopen/open).
    open_ms: f64,
    /// Un-indexed WAL-tail items the resulting snapshot carries.
    tail_entries: usize,
}

impl SnapshotOutcome {
    fn new(action: &'static str, started: std::time::Instant, tail_entries: usize) -> Self {
        Self { action, open_ms: ms(started), tail_entries }
    }
}

/// Default concurrent-retrieval cap when none is configured. Sized so `permits × per-query
/// working set` stays comfortably inside a typical pod: at ≤10M a query reranks ~nprobe×√N
/// memories (tens of MB), so 32 in flight is a few hundred MB — tune via `--max-concurrent-queries`
/// / `MEMLAKE_MAX_CONCURRENT_QUERIES` to trade throughput for a tighter memory ceiling.
pub const DEFAULT_MAX_CONCURRENT_QUERIES: usize = 32;

pub struct MemlakeService {
    store: Store,
    tokenizer: Tokenizer,
    /// One `Writer` per namespace. The Writer caches the next WAL sequence, so serializing
    /// a namespace's writes through it avoids every request re-reading the head and racing
    /// on the same slot. Cross-namespace writes stay fully concurrent.
    writers: Mutex<HashMap<String, Arc<AsyncMutex<Writer>>>>,
    /// Last opened snapshot per namespace, reused across queries when still current.
    snapshots: Mutex<HashMap<String, Arc<Snapshot>>>,
    /// Admission control for the memory-heavy retrieval paths (`query`, `get`), so peak working
    /// memory is bounded by the permit count instead of by request concurrency.
    query_limiter: QueryLimiter,
    /// Per-call JSONL audit log, on only when `MEMLAKE_TRACE_LOG` is set (see `crate::trace`).
    tracer: Tracer,
    /// Live count of executing retrieval requests, recorded per trace. High in-flight + high
    /// latency is the signature of CPU contention (per-query rerank parallelism oversubscribing
    /// the cores under concurrent load) rather than a slow snapshot or cold cache.
    in_flight: std::sync::atomic::AtomicUsize,
}

/// Decrements the in-flight gauge on drop, so it is correct across every return path.
struct InFlightGuard<'a>(&'a std::sync::atomic::AtomicUsize);
impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl MemlakeService {
    pub fn new(store: Store, tokenizer: Tokenizer) -> Self {
        Self {
            store,
            tokenizer,
            writers: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(HashMap::new()),
            query_limiter: QueryLimiter::new(DEFAULT_MAX_CONCURRENT_QUERIES),
            tracer: Tracer::from_env(),
            in_flight: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Whether per-call tracing is enabled — for the startup log.
    pub fn tracing_enabled(&self) -> bool {
        self.tracer.enabled()
    }

    /// How often each node overwrites its bounded trace object in object storage.
    const TRACE_UPLOAD_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

    /// Start the background task that periodically uploads this node's bounded trace ring to
    /// `_obs/traces/{node_id}.jsonl` (overwrite → bounded footprint of one capped object per node).
    /// The admin reads that prefix directly from S3 to render a fleet-wide view without scraping
    /// individual pods (which a load-balanced Service makes unreliable anyway). No-op when tracing is
    /// off. Call once at startup.
    pub fn spawn_trace_uploader(&self, node_id: String) {
        let Some(ring) = self.tracer.ring() else { return };
        let store = self.store.clone();
        let path = crate::trace::obs_traces_path(&node_id);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Self::TRACE_UPLOAD_INTERVAL);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                // Render under the lock, then release it before the network PUT.
                let body = match ring.lock() {
                    Ok(r) => r.render(&node_id),
                    Err(_) => continue,
                };
                if let Err(e) = store.put(&path, body).await {
                    tracing::warn!(%node_id, error = %e, "trace upload failed");
                }
            }
        });
    }

    /// Set the cap on concurrently-executing retrieval requests — the knob that bounds the
    /// server's peak query memory (`max × per-query working set`).
    pub fn with_max_concurrent_queries(mut self, max: usize) -> Self {
        self.query_limiter = QueryLimiter::new(max);
        self
    }

    /// The effective concurrent-retrieval cap, for the startup log.
    pub fn max_concurrent_queries(&self) -> usize {
        self.query_limiter.permits()
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

    /// Get a current snapshot for `ns`, reusing work while no fold has landed since it was opened.
    ///
    /// Three cases, cheapest first:
    /// * head unchanged → reuse the cached snapshot wholesale (one head-pointer GET, no LIST);
    /// * head advanced but the **manifest is unchanged** (a write, not a fold — a conditional GET
    ///   returns 304) → reopen reusing the segments and rebuilding only the WAL tail;
    /// * the manifest changed (a fold) → full reopen.
    /// Reads stay strongly consistent: a reader re-derives the tail up to the live head every time
    /// it advances, so no acked write is ever missed.
    async fn snapshot(&self, ns: &Namespace) -> Result<Arc<Snapshot>, Status> {
        self.snapshot_traced(ns).await.map(|(snap, _)| snap)
    }

    /// Like [`snapshot`], but also reports what it did (`reuse` / `reopen_tail` / `full_open`),
    /// how long it took, and the resulting tail size — the inputs the tracer needs to explain a
    /// slow read (a huge `tail_entries` means the indexer is behind; a `full_open` on a cold node
    /// means metadata round-trips).
    async fn snapshot_traced(
        &self,
        ns: &Namespace,
    ) -> Result<(Arc<Snapshot>, SnapshotOutcome), Status> {
        let started = std::time::Instant::now();
        if let Some(cached) = self.cached_snapshot(&ns.name) {
            // The per-query staleness check: the head from the pointer (one GET), not a LIST.
            let head = ns.resolve_head().await.map_err(internal)?;
            if head == cached.through_seq {
                let out = SnapshotOutcome::new("reuse", started, cached.node.tail_len());
                return Ok((cached, out));
            }

            // Head advanced. If the manifest is unchanged, only the tail grew — reopen cheaply,
            // reusing the loaded segment metadata instead of re-decoding it.
            if let Some(etag) = cached.node.manifest_etag() {
                let manifest_path = mlake_core::manifest::manifest_path(&ns.name);
                match ns.store.get_if_modified(&manifest_path, etag).await {
                    Ok(None) => {
                        let node = Arc::new(
                            cached
                                .node
                                .reopen_extending_tail(head, self.tokenizer.clone())
                                .await
                                .map_err(internal)?,
                        );
                        let out = SnapshotOutcome::new("reopen_tail", started, node.tail_len());
                        return Ok((self.install_snapshot(&ns.name, node), out));
                    }
                    Ok(Some(_)) => {
                        // A fold changed the manifest. Reopen reusing the segments whose id persisted
                        // across the fold (a flush adds one L0, a compaction replaces a few with one —
                        // most persist), reloading only the new ones, and warm those cold clusters
                        // into the read cache off the request path so the next query/derive to probe
                        // them isn't the one that eats the cold object-store fetch (the dominant tail;
                        // see docs/concurrency-findings.md).
                        let node = Arc::new(
                            cached
                                .node
                                .reopen_after_fold(self.tokenizer.clone())
                                .await
                                .map_err(internal)?,
                        );
                        let out = SnapshotOutcome::new("reopen_fold", started, node.tail_len());
                        return Ok((self.install_and_warm(&ns.name, node), out));
                    }
                    Err(_) => {} // conditional GET failed — fall through to the safe full path
                }
            }
        }

        let node = Arc::new(
            QueryNode::open(ns, self.tokenizer.clone())
                .await
                .map_err(internal)?,
        );
        let out = SnapshotOutcome::new("full_open", started, node.tail_len());
        Ok((self.install_and_warm(&ns.name, node), out))
    }

    /// Cache `node` as the current snapshot for `name` and return it.
    fn install_snapshot(&self, name: &str, node: Arc<QueryNode>) -> Arc<Snapshot> {
        let snap = Arc::new(Snapshot {
            through_seq: node.through_seq,
            node,
        });
        self.snapshots.lock().unwrap().insert(name.to_string(), snap.clone());
        snap
    }

    /// Install `node`, then kick a detached background pass that pulls its cluster blobs into the
    /// read cache. Used only where the node just adopted a **new** generation (`reopen_fold`,
    /// `full_open`): the freshly folded segment is cold, so warming it off the request path keeps a
    /// fold from spiking the next query/derive with a multi-second cold `fetch_clusters` — the
    /// dominant latency tail. Persisted segments resolve as cache hits, so the warm fetches only the
    /// blobs the fold actually changed.
    fn install_and_warm(&self, name: &str, node: Arc<QueryNode>) -> Arc<Snapshot> {
        let snap = self.install_snapshot(name, node.clone());
        tokio::spawn(async move { node.warm().await });
        snap
    }

    /// Wait until the background indexer has folded `name`'s WAL up to `seq` into a segment, by
    /// polling the manifest cursor. This replica does **not** fold — a fold can be an O(corpus)
    /// rebuild or compaction, which belongs on the indexer Deployment, not on a query-serving pod.
    /// The write is already durable and visible via the WAL tail before this is called, so a
    /// timeout is a soft failure (the caller just didn't get the "in a segment" confirmation).
    async fn index_until(&self, name: &str, seq: u64) -> Result<u64, Status> {
        let ns = self.namespace(name);
        let deadline = tokio::time::Instant::now() + WAIT_FOR_INDEX_TIMEOUT;
        loop {
            let (manifest, _etag) = ns.read_manifest().await.map_err(internal)?;
            if manifest.wal_index_cursor >= seq {
                // The indexer advanced past our write; drop any snapshot opened against the older
                // generation so the next read sees the freshly-indexed one.
                self.invalidate(name);
                return Ok(manifest.version);
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(Status::deadline_exceeded(format!(
                    "write seq {seq} not folded into a segment within {WAIT_FOR_INDEX_TIMEOUT:?}; \
                     it is durable and visible via the WAL tail — is the indexer running?"
                )));
            }
            tokio::time::sleep(WAIT_FOR_INDEX_POLL).await;
        }
    }

    fn invalidate(&self, name: &str) {
        self.snapshots.lock().unwrap().remove(name);
    }
}

fn internal(e: impl std::fmt::Display) -> Status {
    Status::internal(e.to_string())
}

/// Reject namespace names in the reserved `_`-prefix. Namespaces are top-level object-store prefixes,
/// and the `_obs/` observability root (per-node trace objects) must stay unclaimable — so `_`-leading
/// names are reserved for system use. Cheap enough to run on the create path.
fn reject_reserved_namespace(name: &str) -> Result<(), Status> {
    if name.starts_with('_') {
        return Err(Status::invalid_argument(
            "namespace names starting with '_' are reserved for system use (e.g. the _obs/ trace root)",
        ));
    }
    Ok(())
}

/// Map a retrieval error onto a gRPC status. A dimension mismatch is the caller's mistake —
/// a query embedded with a different model than the index was built with — so it must come
/// back as INVALID_ARGUMENT with the two dimensions, not as an opaque server fault the
/// caller would retry forever.
fn query_error(e: mlake_index::Error) -> Status {
    match &e {
        mlake_index::Error::Core(mlake_core::Error::DimMismatch { .. }) => {
            Status::invalid_argument(e.to_string())
        }
        _ => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl Memlake for MemlakeService {
    async fn create_namespace(
        &self,
        req: Request<pb::CreateNamespaceRequest>,
    ) -> Result<Response<pb::CreateNamespaceResponse>, Status> {
        let req = req.into_inner();
        let name = req.namespace;
        if name.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        reject_reserved_namespace(&name)?;
        self.namespace(&name)
            .create_if_absent(&self.tokenizer.config_hash(), &req.indexed_metadata_keys)
            .await
            .map_err(internal)?;
        Ok(Response::new(pb::CreateNamespaceResponse {}))
    }

    async fn delete_namespace(
        &self,
        req: Request<pb::DeleteNamespaceRequest>,
    ) -> Result<Response<pb::DeleteNamespaceResponse>, Status> {
        let name = req.into_inner().namespace;
        if name.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let objects_deleted = self.namespace(&name).delete_all().await.map_err(internal)? as u64;
        // Drop this process's cached state for the namespace: its Writer's cached WAL sequence
        // is now meaningless, and the snapshot points at deleted objects.
        self.writers.lock().unwrap().remove(&name);
        self.invalidate(&name);
        Ok(Response::new(pb::DeleteNamespaceResponse { objects_deleted }))
    }

    async fn write(
        &self,
        req: Request<pb::WriteRequest>,
    ) -> Result<Response<pb::WriteResponse>, Status> {
        // Per-request object-access spans for the trace waterfall (task-local; see mlake_store::spans).
        mlake_store::spans::scope(async move {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        reject_reserved_namespace(&req.namespace)?;
        let mut ops = req
            .ops
            .into_iter()
            .map(convert::op)
            .collect::<Result<Vec<_>, _>>()?;
        if ops.is_empty() {
            return Err(Status::invalid_argument("write needs at least one op"));
        }
        let t0 = std::time::Instant::now();
        let op_count = ops.len();

        // Derive the batch's semantic kNN links HERE, before the commit, so they travel in the WAL
        // as intrinsic data — the index is then a pure speed optimization, not a correctness
        // dependency (a query over the un-indexed tail sees the links). Neighbours come from the
        // current committed snapshot plus the other memories in this batch.
        let ns = self.namespace(&req.namespace);
        let mut upserts: Vec<mlake_core::Memory> = ops
            .iter()
            .filter_map(|op| match op {
                mlake_core::Op::Upsert(m) => Some(m.clone()),
                _ => None,
            })
            .collect();
        let upsert_count = upserts.len();
        // Broken out so the trace shows whether link derivation is bound by opening the snapshot,
        // by cold S3 reads (roundtrips/cache_misses), or by CPU in the kNN arms (phase breakdown).
        let mut link_snapshot_ms = 0.0;
        let mut link_io = serde_json::Value::Null;
        let link_ms = if !upserts.is_empty() {
            let link_t = std::time::Instant::now();
            let snap_t = std::time::Instant::now();
            let snap = self.snapshot(&ns).await?;
            link_snapshot_ms = ms(snap_t);
            let metrics = QueryMetrics::new();
            let derive_stats = mlake_index::derive_links_for_write(&snap.node, &mut upserts, &metrics)
                .await
                .map_err(query_error)?;
            let mut derived = upserts.into_iter();
            for op in ops.iter_mut() {
                if let mlake_core::Op::Upsert(m) = op {
                    *m = derived.next().expect("one derived memory per upsert");
                }
            }
            if self.tracer.enabled() {
                link_io = serde_json::json!({
                    // The two costs inside link derivation, split:
                    "corpus_query_ms": derive_stats.query_ms,   // step (a): O(new·N) queries
                    "within_batch_ms": derive_stats.batch_ms,   // step (b): O(batch²) cosine
                    "queries": derive_stats.queries,
                    // I/O the corpus queries did (0 ⇒ pure CPU, warm):
                    "roundtrips": metrics.roundtrips(),
                    "cache_hits": metrics.cache_hits(),
                    "cache_misses": metrics.cache_misses(),
                    "bytes": metrics.bytes(),
                    "phases_us": metrics.phase_breakdown(),
                });
            }
            ms(link_t)
        } else {
            0.0
        };

        let writer = self.writer_for(&req.namespace);
        // The commit is durable (S3 conditional PUT) before this returns — that is the ack.
        let commit_t = std::time::Instant::now();
        let result = {
            let mut w = writer.lock().await;
            w.commit(ops).await.map_err(internal)?
        };
        let commit_ms = ms(commit_t);

        // Notify the indexer via the object-storage queue that this namespace has un-indexed WAL —
        // this is what replaces poll-every-namespace: an indexer only ever folds namespaces with a
        // job. Enqueue is idempotent (a namespace already queued or being folded is skipped, and
        // the fold re-checks the WAL head at completion, so a write mid-fold is never lost). A
        // `wait_for_index` caller enqueues inline, so the fold it waits on is actually scheduled;
        // otherwise it is best-effort and off the ack path, with the indexer's reconciliation sweep
        // as the backstop if this pod dies before the spawned task runs.
        let queue = mlake_wal::IndexQueue::new(self.store.clone());
        if req.wait_for_index {
            queue.enqueue(&req.namespace).await.map_err(internal)?;
        } else {
            let ns = req.namespace.clone();
            tokio::spawn(async move {
                if let Err(e) = queue.enqueue(&ns).await {
                    tracing::debug!(namespace = %ns, error = %e, "index enqueue failed (sweep is the backstop)");
                }
            });
        }

        // Do NOT drop the cached snapshot here. The next read's head-check (`snapshot_traced`)
        // sees the advanced head (via the head pointer this write bumped) and reopens the stale
        // snapshot cheaply — `reopen_tail` (write) or `reopen_fold` (a fold) reusing the loaded
        // segments — instead of the `full_open` that dropping it would force. Correctness is
        // unchanged: the reopen re-scans the tail to the new head, so the write is always seen.

        // Optionally fold the write into the indexed generation before returning. Done after
        // the commit so the ack still reflects durability; the extra latency is only paid by
        // callers that opt in.
        let (generation, wait_for_index_ms) = if req.wait_for_index {
            let wait_t = std::time::Instant::now();
            let g = self.index_until(&req.namespace, result.seq).await?;
            (g, Some(ms(wait_t)))
        } else {
            (0, None)
        };

        if self.tracer.enabled() {
            self.tracer.emit(serde_json::json!({
                "ts_ms": now_ms(),
                "op": "write",
                "namespace": req.namespace,
                "total_ms": ms(t0),
                "link_ms": link_ms,
                "link_snapshot_ms": link_snapshot_ms,
                "link_io": link_io,
                "commit_ms": commit_ms,
                "wait_for_index_ms": wait_for_index_ms,
                "params": {
                    "ops": op_count,
                    "upserts": upsert_count,
                    "wait_for_index": req.wait_for_index,
                },
                "seq": result.seq,
                "attempts": result.attempts,
                "objects": mlake_store::spans::current_json(),
            }));
        }

        Ok(Response::new(pb::WriteResponse {
            seq: result.seq,
            attempts: result.attempts as u32,
            generation,
        }))
        })
        .await
    }

    async fn query(
        &self,
        req: Request<pb::QueryRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        // Collect per-object access spans for the whole request (task-local; see mlake_store::spans),
        // so the trace can render an mem/disk/S3 waterfall. Body unindented — Rust ignores it.
        mlake_store::spans::scope(async move {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let t0 = std::time::Instant::now();
        let vector = req.vector.as_ref().map(convert::decode_vector).transpose()?;
        // The flat filter carries the compound `tag_groups` alongside it — the arms push down
        // the flat tags via the block/cluster masks, and the materialization pass applies the
        // groups per-memory once full tags are in hand (see `TagFilter::groups`).
        let mut tags = convert::tag_filter(req.tags);
        tags.groups = convert::tag_predicates(req.tag_groups)?;
        let depths = convert::arm_depths(
            req.vector_top_k,
            req.text_top_k,
            req.graph_top_k,
            req.nprobe,
            req.graph_seed_min_similarity,
        );
        let text: Option<String> = if req.text.is_empty() { None } else { Some(req.text) };
        // Temporal arm runs only when both window bounds are set.
        let temporal_window = match (req.temporal_from, req.temporal_to) {
            (Some(from), Some(to)) => Some((from, to)),
            _ => None,
        };

        // Admission control: hold a permit for the whole query so at most `permits` queries
        // rerank clusters at once — the server's peak query memory is bounded by the permit
        // count, not by how many requests arrive together. Under load this awaits (backpressure),
        // it does not reject.
        let permit_t = std::time::Instant::now();
        let _permit = self.query_limiter.acquire().await;
        let permit_wait_ms = ms(permit_t);

        // Concurrency at the moment this query begins executing — a high value alongside high
        // latency points at CPU contention, not a slow snapshot.
        let in_flight = self.in_flight.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let _in_flight_guard = InFlightGuard(&self.in_flight);

        let ns = self.namespace(&req.namespace);
        // Reuse the cached open snapshot when still current (see `snapshot`): a warm read is
        // pure in-memory arm evaluation over the shared Store cache, 0 fresh roundtrips.
        let (snap, snap_out) = self.snapshot_traced(&ns).await?;
        let node = &snap.node;

        // Which types to answer: the caller's list, or every type in the snapshot.
        let mut types: Vec<u8> = if req.memory_types.is_empty() {
            node.memory_types()
        } else {
            req.memory_types
                .iter()
                .map(|&t| convert::memory_type_u8(t))
                .collect::<Result<Vec<_>, _>>()?
        };
        types.sort_unstable();
        types.dedup();

        // Run every type's three arms concurrently over the one shared snapshot, sharing a
        // single metrics sink. Their storage reads are issued together, so they land in the
        // same roundtrip waves — a 3-type × 3-arm query costs the waves of one, not nine.
        let metrics = QueryMetrics::new();
        let vref = vector.as_deref();
        let tref = text.as_deref();
        let updated = mlake_index::UpdatedWindow { from: req.updated_from, to: req.updated_to };
        let per_type = futures::future::try_join_all(types.into_iter().map(|mt| {
            let metrics = &metrics;
            let tags = &tags;
            async move {
                node.query_raw_metered(
                    mt,
                    vref,
                    tref,
                    tags,
                    depths,
                    temporal_window,
                    updated,
                    metrics,
                )
                .await
                .map(|hits| (mt, hits))
            }
        }))
        .await
        .map_err(query_error)?;

        // Flatten to one hit list; each hit carries its memory_type so the client can group.
        let mut hits = Vec::new();
        for (mt, raw) in per_type {
            hits.extend(raw.into_iter().map(|h| convert::raw_hit(mt, h)));
        }

        // The dense arm has already applied the window (it rides in the vector block's
        // `updated` column, so a non-matching member never takes a top-k slot), but the FTS
        // and graph arms have not — neither index carries a write time — and a block written
        // before the column existed admits everyone. So the window is re-applied here as the
        // authority. For a dense query this now trims nothing; for a text- or graph-only one
        // it is still a post-filter, and can trim a page below `top_k`.
        if req.updated_from.is_some() || req.updated_to.is_some() {
            hits.retain(|h| {
                let Some(ts) = h.memory.as_ref().and_then(|m| m.timestamps.as_ref()) else {
                    return false;
                };
                let Some(updated) = ts.updated_at else {
                    return false;
                };
                req.updated_from.is_none_or(|from| updated > from)
                    && req.updated_to.is_none_or(|to| updated < to)
            });
        }

        if self.tracer.enabled() {
            let (rt, ch, cm) = (metrics.roundtrips(), metrics.cache_hits(), metrics.cache_misses());
            let denom = (ch + cm).max(1) as f64;
            self.tracer.emit(serde_json::json!({
                "ts_ms": now_ms(),
                "op": "query",
                "namespace": req.namespace,
                "total_ms": ms(t0),
                "permit_wait_ms": permit_wait_ms,
                "in_flight": in_flight,
                "snapshot": {
                    "action": snap_out.action,
                    "open_ms": snap_out.open_ms,
                    "tail_entries": snap_out.tail_entries,
                },
                "phases_us": metrics.phase_breakdown(),
                "io": {
                    "roundtrips": rt,
                    "cache_hits": ch,
                    "cache_misses": cm,
                    "hit_ratio": ch as f64 / denom,
                    "bytes": metrics.bytes(),
                    "tier": if rt == 0 { "warm" } else { "cold" },
                },
                "result_count": hits.len(),
                "objects": mlake_store::spans::current_json(),
                "params": {
                    "nprobe": req.nprobe,
                    "vector_top_k": req.vector_top_k,
                    "text_top_k": req.text_top_k,
                    "graph_top_k": req.graph_top_k,
                    "has_vector": vector.is_some(),
                    "has_text": text.is_some(),
                },
            }));
        }

        Ok(Response::new(pb::QueryResponse {
            hits,
            load_roundtrips: metrics.roundtrips() as u32,
        }))
        })
        .await
    }

    // ---- Admin / introspection ----

    async fn list_namespaces(
        &self,
        _req: Request<pb::ListNamespacesRequest>,
    ) -> Result<Response<pb::ListNamespacesResponse>, Status> {
        let namespaces = discover_namespaces(&self.store).await.map_err(internal)?;
        Ok(Response::new(pb::ListNamespacesResponse { namespaces }))
    }

    async fn stats(
        &self,
        req: Request<pb::StatsRequest>,
    ) -> Result<Response<pb::StatsResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let ns = self.namespace(&req.namespace);

        // The manifest gives the published index state; the snapshot gives the *live* doc
        // counts, which differ from it by the un-indexed tail and any tombstones.
        let (manifest, _etag) = ns.read_manifest().await.map_err(internal)?;
        let wal_head = ns.wal_head().await.map_err(internal)?;
        let snap = self.snapshot(&ns).await?;
        let node = &snap.node;

        let types: Vec<pb::TypeStats> = node
            .memory_types()
            .into_iter()
            .map(|mt| pb::TypeStats {
                memory_type: mt as u32,
                doc_count: node.doc_count_of(mt) as u64,
                cluster_count: node.cluster_count_of(mt) as u32,
                train_count: manifest
                    .segments
                    .iter()
                    .filter_map(|s| s.index(mt))
                    .map(|i| i.train_count)
                    .sum(),
                has_index: manifest.segments.iter().any(|s| s.index(mt).is_some()),
            })
            .collect();

        Ok(Response::new(pb::StatsResponse {
            namespace: req.namespace,
            generation: manifest.version,
            prev_generation: None,
            // The *live* head, not `manifest.wal_head`: the indexer writes the manifest's
            // head and cursor to the same value, so their difference is always zero. The
            // backlog is the number this view exists to show, so it is worth one LIST.
            wal_head,
            wal_index_cursor: manifest.wal_index_cursor,
            tokenizer_config_hash: manifest.tokenizer_config_hash,
            format_version: manifest.format_version,
            doc_count: node.doc_count() as u64,
            types,
            through_seq: snap.through_seq,
            load_roundtrips: node.load_roundtrips as u32,
        }))
    }

    async fn get(&self, req: Request<pb::GetRequest>) -> Result<Response<pb::GetResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let ids = req
            .ids
            .iter()
            .map(|b| convert::id_bytes(b))
            .collect::<Result<Vec<_>, _>>()?;
        if ids.is_empty() {
            return Ok(Response::new(pb::GetResponse { memories: Vec::new() }));
        }
        let t0 = std::time::Instant::now();

        // `get` with `include_vector` fetches whole cluster files, so it shares the query
        // admission cap; the payload-only path is cheap but is gated too for a single, simple
        // memory ceiling over the retrieval paths.
        let permit_t = std::time::Instant::now();
        let _permit = self.query_limiter.acquire().await;
        let permit_wait_ms = ms(permit_t);

        let ns = self.namespace(&req.namespace);
        let (snap, snap_out) = self.snapshot_traced(&ns).await?;
        let found = snap.node.get_many(&ids, req.include_vector).await.map_err(internal)?;
        let result_count = found.len();
        let memories: Vec<_> = found
            .into_iter()
            .map(|m| convert::stored_record_with_edges(m, req.include_vector, req.include_edges))
            .collect();

        if self.tracer.enabled() {
            self.tracer.emit(serde_json::json!({
                "ts_ms": now_ms(),
                "op": "get",
                "namespace": req.namespace,
                "total_ms": ms(t0),
                "permit_wait_ms": permit_wait_ms,
                "snapshot": {
                    "action": snap_out.action,
                    "open_ms": snap_out.open_ms,
                    "tail_entries": snap_out.tail_entries,
                },
                "result_count": result_count,
                "params": { "ids": ids.len(), "include_vector": req.include_vector },
            }));
        }

        Ok(Response::new(pb::GetResponse { memories }))
    }

    async fn entity_stats(
        &self,
        req: Request<pb::EntityStatsRequest>,
    ) -> Result<Response<pb::EntityStatsResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let mut types: Vec<u8> = req
            .memory_types
            .iter()
            .map(|&t| convert::memory_type_u8(t))
            .collect::<Result<Vec<_>, _>>()?;
        types.sort_unstable();
        types.dedup();

        let entities: Option<Vec<mlake_core::EntityId>> = if req.entity_ids.is_empty() {
            None
        } else {
            Some(convert::entity_ids_in(&req.entity_ids)?)
        };

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns).await?;
        let counts = snap
            .node
            .entity_counts(&types, entities.as_deref())
            .await
            .map_err(internal)?;

        let mut entities: Vec<pb::EntityCount> = counts
            .into_iter()
            .map(|(id, n)| pb::EntityCount { entity_id: id.0.to_vec(), memory_count: n })
            .collect();
        // Deterministic order so a paging caller sees a stable list.
        entities.sort_by(|a, b| b.memory_count.cmp(&a.memory_count).then(a.entity_id.cmp(&b.entity_id)));
        Ok(Response::new(pb::EntityStatsResponse { entities }))
    }

    async fn metadata_stats(
        &self,
        req: Request<pb::MetadataStatsRequest>,
    ) -> Result<Response<pb::MetadataStatsResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        if req.key.is_empty() {
            return Err(Status::invalid_argument("key is required"));
        }
        let mut types: Vec<u8> = req
            .memory_types
            .iter()
            .map(|&t| convert::memory_type_u8(t))
            .collect::<Result<Vec<_>, _>>()?;
        types.sort_unstable();
        types.dedup();

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns).await?;
        let counts = snap.node.metadata_counts(&req.key, &types).await.map_err(internal)?;

        let mut values: Vec<pb::MetadataValueCount> = counts
            .into_iter()
            .map(|(value, count)| pb::MetadataValueCount { value, count })
            .collect();
        // Deterministic order: most-populous first, value as tie-break.
        values.sort_by(|a, b| b.count.cmp(&a.count).then(a.value.cmp(&b.value)));
        Ok(Response::new(pb::MetadataStatsResponse { values }))
    }

    async fn link_stats(
        &self,
        req: Request<pb::LinkStatsRequest>,
    ) -> Result<Response<pb::LinkStatsResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let mut types: Vec<u8> = req
            .memory_types
            .iter()
            .map(|&t| convert::memory_type_u8(t))
            .collect::<Result<Vec<_>, _>>()?;
        types.sort_unstable();
        types.dedup();

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns).await?;
        let (semantic_edge_count, causal_edge_count, temporal_edge_count) =
            snap.node.link_counts(&types).await.map_err(internal)?;
        Ok(Response::new(pb::LinkStatsResponse {
            semantic_edge_count,
            causal_edge_count,
            temporal_edge_count,
        }))
    }

    async fn scan(
        &self,
        req: Request<pb::ScanRequest>,
    ) -> Result<Response<pb::ScanResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let t0 = std::time::Instant::now();
        let limit = match req.limit {
            0 => DEFAULT_SCAN_LIMIT,
            n => (n as usize).min(MAX_SCAN_LIMIT),
        };
        // Tags and metadata are one conjunction, the same shape `DeleteByPredicate` takes.
        // memory_types stays out of it — the walk below already visits one type at a time.
        let tag_filter = convert::tag_filter(req.tags);
        let filter = mlake_core::Predicate {
            memory_types: Vec::new(),
            metadata_equals: req.metadata_equals.into_iter().collect(),
            tags: tag_filter.tags.clone(),
            tags_mode: tag_filter.mode as u8,
            updated_from: req.updated_from,
            updated_to: req.updated_to,
        };
        // Compound tag_groups ride alongside the flat `Predicate` — kept separate because the
        // Predicate is WAL-serializable (it is also the delete-by-predicate shape) and groups are
        // a read-only filter. Applied per-member during the walk, AND-ed with `filter`.
        let groups = convert::tag_predicates(req.tag_groups)?;
        let mut skip = req.skip as usize;

        let ns = self.namespace(&req.namespace);
        let (snap, snap_out) = self.snapshot_traced(&ns).await?;
        let node = &snap.node;

        let mut types: Vec<u8> = if req.memory_types.is_empty() {
            node.memory_types()
        } else {
            req.memory_types
                .iter()
                .map(|&t| convert::memory_type_u8(t))
                .collect::<Result<Vec<_>, _>>()?
        };
        types.sort_unstable();
        types.dedup();

        // Resume where the last page stopped. A cursor is only meaningful against the
        // generation that issued it — cluster paths and ordering change when the indexer
        // publishes — so a stale token restarts the scan rather than silently skipping or
        // repeating memories.
        let start = parse_page_token(&req.page_token, node.generation)?;
        let mut pending: &[u8] = match &start {
            Some((ty, _)) => match types.iter().position(|t| t == ty) {
                Some(i) => &types[i..],
                // The cursor's type is not in this request's set: nothing left to walk.
                None => &[],
            },
            None => &types,
        };
        let mut cursor = start.map(|(_, c)| c).unwrap_or_default();

        // Walk types in order, spilling into the next one whenever a page has room left, so
        // a page is only short when the whole scan is exhausted.
        let mut out = Vec::new();
        let mut next_token = String::new();
        while let Some((&ty, rest)) = pending.split_first() {
            // While skipping, pull full pages and discard them: the walk is the same, the
            // caller just does not pay the round trips.
            let want = if skip > 0 { limit } else { limit - out.len() };
            let (items, next) = node
                .scan(ty, cursor, want, &filter, &groups)
                .await
                .map_err(internal)?;
            let mut items = items;
            if skip > 0 {
                let dropped = skip.min(items.len());
                items.drain(..dropped);
                skip -= dropped;
            }
            out.extend(
                items
                    .into_iter()
                    .map(|m| convert::stored_record_with_edges(m, req.include_vector, req.include_edges)),
            );

            match next {
                // This type still has more and the page is full: stop here. Still
                // skipping means the page is not full yet, however many we discarded.
                Some(c) if skip == 0 && out.len() >= limit => {
                    next_token = page_token(node.generation, ty, c);
                    break;
                }
                // Page has room, so continue into this same type's next cluster.
                Some(c) => cursor = c,
                // Type exhausted: move to the next one.
                None => {
                    cursor = ScanCursor::default();
                    if skip == 0 && out.len() >= limit && !rest.is_empty() {
                        next_token = page_token(node.generation, rest[0], cursor);
                        break;
                    }
                    pending = rest;
                }
            }
        }

        if self.tracer.enabled() {
            self.tracer.emit(serde_json::json!({
                "ts_ms": now_ms(),
                "op": "scan",
                "namespace": req.namespace,
                "total_ms": ms(t0),
                "snapshot": {
                    "action": snap_out.action,
                    "open_ms": snap_out.open_ms,
                    "tail_entries": snap_out.tail_entries,
                },
                "result_count": out.len(),
                "params": {
                    "limit": limit,
                    "skip": req.skip,
                    "has_filter": !filter.metadata_equals.is_empty() || !filter.tags.is_empty(),
                    "more": !next_token.is_empty(),
                },
            }));
        }

        Ok(Response::new(pb::ScanResponse { memories: out, next_page_token: next_token }))
    }

    async fn delete_by_predicate(
        &self,
        req: Request<pb::DeleteByPredicateRequest>,
    ) -> Result<Response<pb::DeleteByPredicateResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        // Build the predicate once — shared by both the lazy WAL op and the eager scan.
        let pred = convert::predicate(pb::Predicate {
            memory_types: req.memory_types.clone(),
            metadata_equals: req.metadata_equals.clone(),
            tags: req.tags.clone(),
        })?;
        // Safety: refuse an empty predicate unless the caller explicitly asked to delete all.
        if pred.is_empty() && !req.delete_all {
            return Err(Status::invalid_argument(
                "empty predicate: set metadata_equals/tags, or delete_all to remove every memory",
            ));
        }

        // Default (lazy): one atomic, race-closed TombstoneWhere WAL op. Nothing is scanned —
        // the delete is materialized at the next fold, where the indexer reads every cluster
        // anyway. This is the path re-ingest should use (ideally batched with the new upserts).
        if !req.eager {
            let writer = self.writer_for(&req.namespace);
            let seq = {
                let mut w = writer.lock().await;
                w.commit(vec![mlake_core::Op::TombstoneWhere { predicate: pred }])
                    .await
                    .map_err(internal)?
                    .seq
            };
            // snapshot left cached: the next read re-validates its head and reopens cheaply (see write()).
            return Ok(Response::new(pb::DeleteByPredicateResponse { deleted: 0, seq }));
        }

        // Eager: scan now, tombstone matches by id (O(corpus)), return the exact count.
        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns).await?;
        let metadata_equals: Vec<(String, String)> = req.metadata_equals.into_iter().collect();
        let tags = convert::tag_filter(req.tags);
        let ids = snap.node.ids_matching(&pred.memory_types, &metadata_equals, &tags).await.map_err(internal)?;
        if ids.is_empty() {
            return Ok(Response::new(pb::DeleteByPredicateResponse { deleted: 0, seq: 0 }));
        }
        const TOMBSTONE_BATCH: usize = 10_000;
        let deleted = ids.len() as u64;
        let writer = self.writer_for(&req.namespace);
        let mut last_seq = 0u64;
        {
            let mut w = writer.lock().await;
            for chunk in ids.chunks(TOMBSTONE_BATCH) {
                let ops = chunk.iter().map(|id| mlake_core::Op::Tombstone { id: *id }).collect();
                last_seq = w.commit(ops).await.map_err(internal)?.seq;
            }
        }
        // snapshot left cached: the next read re-validates its head and reopens cheaply (see write()).
        Ok(Response::new(pb::DeleteByPredicateResponse { deleted, seq: last_seq }))
    }

    async fn list_wal(
        &self,
        req: Request<pb::ListWalRequest>,
    ) -> Result<Response<pb::ListWalResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let limit = match req.limit {
            0 => DEFAULT_WAL_LIMIT,
            n => (n as usize).min(MAX_WAL_LIMIT),
        };
        let ns = self.namespace(&req.namespace);

        // The manifest gives the fold watermark, so each entry can say whether the indexer
        // has already absorbed it — the distinction the whole view exists to show. The head
        // comes from the log itself: the manifest's copy is written equal to the cursor, so
        // it can never show a backlog.
        let (manifest, _etag) = ns.read_manifest().await.map_err(internal)?;
        let wal_head = ns.wal_head().await.map_err(internal)?;
        let (objects, next_seq) = ns.list_wal(req.start_seq, limit).await.map_err(internal)?;

        let mut entries = Vec::with_capacity(objects.len());
        for o in objects {
            // Decoding is per-entry and opt-in: one entry is a whole group-commit batch.
            // A racing GC can reclaim an entry between the LIST and this read, which is
            // normal for a folded entry — report it with counts unset rather than failing
            // the whole page.
            let decoded = if req.include_ops {
                ns.read_wal_entry(o.seq).await.ok()
            } else {
                None
            };
            entries.push(convert::wal_entry(&o, manifest.wal_index_cursor, decoded.as_ref()));
        }

        Ok(Response::new(pb::ListWalResponse {
            entries,
            wal_head,
            wal_index_cursor: manifest.wal_index_cursor,
            next_seq: next_seq.unwrap_or(0),
        }))
    }

    async fn index_layout(
        &self,
        req: Request<pb::IndexLayoutRequest>,
    ) -> Result<Response<pb::IndexLayoutResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let memory_type = convert::memory_type_u8(req.memory_type)?;
        let sample = (req.member_sample as usize).min(MAX_MEMBER_SAMPLE);

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns).await?;
        let node = &snap.node;

        // Centroids are already resident from opening the snapshot, so this branch costs
        // zero object-storage reads; only member sampling touches cluster files.
        let (dim, clusters) = match node.cluster_layout(memory_type) {
            Some(layout) => (layout.dim as u32, convert::cluster_infos(&layout)),
            None => (0, Vec::new()),
        };

        let members = node
            .sample_members(memory_type, sample)
            .await
            .map_err(internal)?
            .into_iter()
            .map(|(cluster_id, m)| convert::cluster_member(cluster_id, m))
            .collect();

        Ok(Response::new(pb::IndexLayoutResponse {
            namespace: req.namespace,
            memory_type: req.memory_type,
            generation: node.generation,
            dim,
            clusters,
            members,
        }))
    }

    async fn cache_stats(
        &self,
        req: Request<pb::CacheStatsRequest>,
    ) -> Result<Response<pb::CacheStatsResponse>, Status> {
        let req = req.into_inner();
        let limit = match req.limit {
            0 => DEFAULT_CACHE_LIMIT,
            n => (n as usize).min(MAX_CACHE_LIMIT),
        };

        // No cache configured: the node reads through to object storage every time. That is
        // a valid deployment, so report it rather than erroring.
        let Some(cache) = self.store.cache() else {
            return Ok(Response::new(pb::CacheStatsResponse {
                enabled: false,
                ..Default::default()
            }));
        };

        // The cache applies the filter and reports the pre-truncation total, so a short
        // page can be labelled as a page rather than mistaken for the whole cache.
        let ns_filter = (!req.namespace.is_empty()).then_some(req.namespace.as_str());
        let (entries, total_entries) = cache.entries(ns_filter, limit);

        Ok(Response::new(pb::CacheStatsResponse {
            enabled: true,
            mem_bytes: cache.bytes(),
            mem_budget: cache.mem_budget(),
            disk_bytes: cache.disk_bytes(),
            disk_budget: cache.disk_budget(),
            mem_entries: cache.len() as u64,
            disk_entries: cache.disk_len() as u64,
            hits: cache.hits(),
            misses: cache.misses(),
            entries: entries
                .into_iter()
                .enumerate()
                .map(|(rank, e)| convert::cache_entry(rank, e))
                .collect(),
            total_entries: total_entries as u64,
        }))
    }

    async fn list_objects(
        &self,
        req: Request<pb::ListObjectsRequest>,
    ) -> Result<Response<pb::ListObjectsResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
        let limit = match req.limit {
            0 => DEFAULT_OBJECT_LIMIT,
            n => (n as usize).min(MAX_OBJECT_LIMIT),
        };
        let ns = self.namespace(&req.namespace);
        let (manifest, _etag) = ns.read_manifest().await.map_err(internal)?;
        let live = objects::live_paths(&req.namespace, &manifest);

        // One LIST answers the whole namespace, sizes included.
        let listed = self
            .store
            .list_with_size(&format!("{}/", req.namespace))
            .await
            .map_err(internal)?;

        // Totals span the namespace, not the page — a page's bytes would say nothing about
        // how much storage the namespace actually occupies.
        let mut total_bytes = 0u64;
        let mut live_bytes = 0u64;
        let mut all: Vec<(objects::Classified, bool)> = Vec::with_capacity(listed.len());
        for (path, size) in listed {
            let c = objects::classify(&req.namespace, &path, size);
            // A WAL entry is live while the indexer has not folded it past the watermark a
            // reader could still be scanning from; generation files are live iff the
            // current manifest references them.
            let is_live = match c.seq {
                Some(seq) => seq > manifest.prev_wal_index_cursor,
                None => live.contains(&c.path),
            };
            total_bytes += size;
            if is_live {
                live_bytes += size;
            }
            all.push((c, is_live));
        }
        let total_objects = all.len() as u64;

        // Group a segment's files together (by segment id), then by kind, then path — so one
        // segment's index files read as a block. The manifest and WAL entries have no segment and
        // sort first (empty string), which is where they belong: they publish the segments.
        all.sort_by(|a, b| {
            a.0.segment
                .cmp(&b.0.segment)
                .then((a.0.kind as i32).cmp(&(b.0.kind as i32)))
                .then(a.0.path.cmp(&b.0.path))
        });

        let start: usize = if req.page_token.is_empty() {
            0
        } else {
            req.page_token
                .parse()
                .map_err(|_| Status::invalid_argument("malformed page_token"))?
        };
        let page: Vec<pb::ObjectInfo> = all
            .iter()
            .skip(start)
            .take(limit)
            .map(|(c, is_live)| convert::object_info(c, *is_live))
            .collect();
        let next = start + page.len();

        Ok(Response::new(pb::ListObjectsResponse {
            objects: page,
            total_objects,
            total_bytes,
            live_bytes,
            generation: manifest.version,
            next_page_token: if (next as u64) < total_objects {
                next.to_string()
            } else {
                String::new()
            },
        }))
    }

    async fn decode_object(
        &self,
        req: Request<pb::DecodeObjectRequest>,
    ) -> Result<Response<pb::DecodeObjectResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() || req.path.is_empty() {
            return Err(Status::invalid_argument("namespace and path are required"));
        }
        // Confine reads to the namespace's own prefix: `path` comes from a client, and an
        // inspection tool must not become a way to read arbitrary bucket keys.
        if !req.path.starts_with(&format!("{}/", req.namespace)) {
            return Err(Status::invalid_argument(
                "path must be inside the namespace's prefix",
            ));
        }
        let limit = match req.limit {
            0 => DEFAULT_DECODE_ITEMS,
            n => (n as usize).min(MAX_DECODE_ITEMS),
        };

        let classified = objects::classify(&req.namespace, &req.path, 0);
        let decoded = objects::decode(&self.store, classified.kind, &req.path, limit).await?;
        let size_bytes = self.store.head(&req.path).await.unwrap_or(0);

        Ok(Response::new(pb::DecodeObjectResponse {
            kind: classified.kind as i32,
            json: serde_json::to_string_pretty(&decoded.json).map_err(internal)?,
            size_bytes,
            total_items: decoded.total_items,
            truncated: decoded.truncated,
            undecodable_reason: decoded.undecodable_reason,
        }))
    }
}

/// Object-listing page sizes, and the ceiling on items pulled out of one decoded container.
const DEFAULT_OBJECT_LIMIT: usize = 200;
const MAX_OBJECT_LIMIT: usize = 2000;
const DEFAULT_DECODE_ITEMS: usize = 50;
const MAX_DECODE_ITEMS: usize = 1000;

/// Cache listing sizes. A warm node holds thousands of blocks; this is an inspection call.
const DEFAULT_CACHE_LIMIT: usize = 100;
const MAX_CACHE_LIMIT: usize = 2000;

/// WAL page sizes, and the ceiling on a member sample. All three bound one response: the
/// log can be long, and a sample is a picture of the layout, not a bulk export.
const DEFAULT_WAL_LIMIT: usize = 50;
const MAX_WAL_LIMIT: usize = 500;
const MAX_MEMBER_SAMPLE: usize = 5000;

/// Scan page sizes. The cap keeps one response bounded regardless of what a client asks
/// for — a scan is the one read path whose cost grows with the corpus.
const DEFAULT_SCAN_LIMIT: usize = 50;
const MAX_SCAN_LIMIT: usize = 1000;

/// Encode a scan cursor as an opaque token. The generation is embedded so a token issued
/// against an older index is detected rather than silently misapplied.
fn page_token(generation: u64, memory_type: u8, c: ScanCursor) -> String {
    format!("{generation}:{memory_type}:{}:{}", c.cluster, c.offset)
}

/// Decode a page token, or `None` to start from the beginning. A token from a superseded
/// generation restarts the scan — the alternative is skipping or repeating memories.
fn parse_page_token(token: &str, generation: u64) -> Result<Option<(u8, ScanCursor)>, Status> {
    if token.is_empty() {
        return Ok(None);
    }
    let parts: Vec<&str> = token.split(':').collect();
    let [gen, ty, cluster, offset] = parts[..] else {
        return Err(Status::invalid_argument("malformed page_token"));
    };
    let bad = |_| Status::invalid_argument("malformed page_token");
    if gen.parse::<u64>().map_err(bad)? != generation {
        return Ok(None);
    }
    Ok(Some((
        ty.parse::<u8>().map_err(bad)?,
        ScanCursor {
            cluster: cluster.parse::<usize>().map_err(bad)?,
            offset: offset.parse::<usize>().map_err(bad)?,
        },
    )))
}

/// Per-namespace drain iterations per claim. One fold folds the whole tail up to the head, so a
/// second only catches writes that arrived *during* the fold; a small ceiling drains a burst
/// promptly, then the completion re-check re-queues the namespace if it is still dirty.
const MAX_DRAIN_ITERS: usize = 8;
/// How long a claimed job may go without a heartbeat before another indexer may steal it (its
/// worker is presumed crashed). Generous versus a normal fold; a long compaction is kept alive by
/// the background heartbeat below, so this only fires on an actual death.
const CLAIM_STALE_MS: u64 = 120_000;
/// How often the background task refreshes a held claim's heartbeat during a fold.
const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
/// How often the reconciliation sweep runs — the backstop that enqueues any dirty namespace the
/// queue may have missed (e.g. a serve pod that died before its async enqueue ran). Rare on
/// purpose: the queue is the hot path, this is a slow safety net, and it is the only LIST-all.
const RECONCILE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(300);

/// True when the namespace has WAL past the indexed cursor — i.e. there is work to fold.
async fn namespace_is_dirty(ns: &Namespace) -> bool {
    let Ok(head) = ns.wal_head().await else { return false };
    match ns.read_manifest().await {
        Ok((m, _)) => head > m.wal_index_cursor,
        Err(_) => true, // can't tell → assume dirty; a needless fold is cheap and safe
    }
}

/// Backstop sweep: enqueue every dirty namespace so nothing is stranded if an enqueue was lost.
async fn reconcile_queue(store: &Store, queue: &IndexQueue, namespaces: &[String]) {
    let targets = if namespaces.is_empty() {
        discover_namespaces(store).await.unwrap_or_default()
    } else {
        namespaces.to_vec()
    };
    for name in &targets {
        let ns = Namespace::new(name, store.clone());
        if namespace_is_dirty(&ns).await {
            if let Err(e) = queue.enqueue(name).await {
                tracing::debug!(namespace = name, error = %e, "reconcile enqueue failed");
            }
        }
    }
}

/// Run the indexer, pulling namespaces to fold from the object-storage queue instead of polling
/// every namespace. Serve pods enqueue on write; each indexer claims a job (CAS, exclusive), drains
/// its tail, GCs, and completes it — re-queueing if more WAL arrived. A rare reconciliation sweep is
/// the backstop. Idempotent and coordination-free, so N indexer replicas share the one queue safely.
#[allow(clippy::too_many_arguments)]
pub async fn run_indexer(
    store: Store,
    tokenizer: Tokenizer,
    namespaces: Vec<String>,
    interval: std::time::Duration,
    worker_id: String,
    budget: FoldBudget,
    streaming_threshold: usize,
    _tail_flush_docs: usize,
    gc_interval: std::time::Duration,
    gc_min_age: std::time::Duration,
) -> anyhow::Result<()> {
    let queue = IndexQueue::new(store.clone());
    // Per-namespace GC throttle: reclaiming runs on a slower cadence than folding (dead objects are
    // min-age-gated inside `gc`, so more-frequent passes would mostly be no-op LISTs).
    let mut last_gc: HashMap<String, std::time::Instant> = HashMap::new();
    // Due immediately, so startup populates the queue from any already-dirty namespaces (including
    // work that predates the queue).
    let mut last_reconcile = std::time::Instant::now()
        .checked_sub(RECONCILE_INTERVAL)
        .unwrap_or_else(std::time::Instant::now);

    loop {
        if last_reconcile.elapsed() >= RECONCILE_INTERVAL {
            reconcile_queue(&store, &queue, &namespaces).await;
            last_reconcile = std::time::Instant::now();
        }

        // Claim the next namespace with work. `None` ⇒ queue empty; idle briefly and re-check.
        let name = match queue.claim(&worker_id, CLAIM_STALE_MS).await {
            Ok(Some(n)) => n,
            Ok(None) => {
                tokio::time::sleep(interval).await;
                continue;
            }
            Err(e) => {
                tracing::warn!(error = %e, "index queue claim failed");
                tokio::time::sleep(interval).await;
                continue;
            }
        };
        let ns = Namespace::new(&name, store.clone());

        // Keep the claim alive across a long fold/compaction with a background heartbeat; it stops
        // itself once the job is no longer ours (reclaimed or completed).
        let hb = {
            let (q, n, w) = (queue.clone(), name.clone(), worker_id.clone());
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(HEARTBEAT_INTERVAL).await;
                    if !q.heartbeat(&n, &w).await.unwrap_or(false) {
                        break;
                    }
                }
            })
        };

        // Drain: keep folding until the tail is empty (a fold that consumed nothing) or the per-claim
        // ceiling is hit, so a bulk load folds promptly instead of accumulating an unbounded tail.
        let mut fold_errored = false;
        for _ in 0..MAX_DRAIN_ITERS {
            match fold(&ns, &tokenizer, IndexOptions::default(), budget, streaming_threshold).await {
                Ok(outcome) => {
                    if outcome.doc_count > 0 {
                        tracing::info!(
                            namespace = name,
                            generation = outcome.generation,
                            docs = outcome.doc_count,
                            published = outcome.published,
                            "indexed"
                        );
                    }
                    if outcome.doc_count == 0 {
                        break; // tail fully drained
                    }
                }
                Err(e) => {
                    tracing::warn!(namespace = name, error = %e, "index failed");
                    fold_errored = true;
                    break;
                }
            }
        }

        // Reclaim unreferenced objects (folded WAL entries, compacted-away segments, CAS-race
        // losers), throttled to `gc_interval` — well below `gc`'s own min-age guard, which keeps
        // anything a slow reader on the previous manifest might still be scanning. GC reads the
        // manifest fresh and only deletes what the CURRENT manifest doesn't reference, so it is safe
        // alongside a peer that just published a new generation.
        let gc_due = !fold_errored && last_gc.get(&name).map(|t| t.elapsed() >= gc_interval).unwrap_or(true);
        if gc_due {
            match mlake_index::gc_with_min_age(&ns, gc_min_age).await {
                Ok(out) if out.generation_files_deleted + out.wal_entries_deleted > 0 => {
                    tracing::info!(
                        namespace = name,
                        generation_files = out.generation_files_deleted,
                        wal_entries = out.wal_entries_deleted,
                        "gc reclaimed"
                    );
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(namespace = name, error = %e, "gc failed"),
            }
            last_gc.insert(name.clone(), std::time::Instant::now());
        }

        hb.abort();

        // Complete. On a fold ERROR, drop the job rather than re-queue it: re-queueing an
        // unfoldable namespace (e.g. a corrupt/old-format manifest) means it is claimed again
        // immediately and fails again — a tight loop that starves every other namespace. Dropping it
        // lets the reconciliation sweep re-enqueue it on its slow cadence instead (bounded retry).
        // On success, re-queue iff more WAL arrived during the fold (head past the indexed cursor) —
        // this closes the completion/write race so the job is only removed when genuinely drained.
        let still_dirty = !fold_errored && namespace_is_dirty(&ns).await;
        if let Err(e) = queue.complete(&name, &worker_id, still_dirty).await {
            tracing::warn!(namespace = name, error = %e, "index queue complete failed");
        }
        // Loop straight back to claim the next job — the queue's empty path is where we idle.
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
