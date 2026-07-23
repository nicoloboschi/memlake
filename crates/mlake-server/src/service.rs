//! The gRPC service: a thin, stateless orchestration layer over `Namespace`/`Writer`/
//! `QueryNode`. Any replica can serve any request — all coordination is in object storage —
//! so this holds only process-local conveniences: a shared `Store` (its two-tier cache
//! warms across requests) and a per-namespace `Writer` so sequence claims don't self-contend
//! within a process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mlake_fts::Tokenizer;
use mlake_index::streaming::FoldBudget;
use mlake_index::{fold, IndexOptions, QueryNode, ScanCursor, DEFAULT_STREAMING_THRESHOLD_DOCS};
use mlake_store::{QueryMetrics, Store};
use mlake_wal::{Namespace, Writer};
use tokio::sync::Mutex as AsyncMutex;
use tonic::{Request, Response, Status};

use crate::limiter::QueryLimiter;
use crate::pb::memlake_server::Memlake;
use crate::{convert, objects, pb};

/// Bounded number of inline fold attempts a `wait_for_index` write makes before giving up.
/// One fold folds the whole tail up to the current head, so a single pass normally covers the
/// write; extra passes only absorb a lost manifest-CAS race or a head that advanced under a
/// concurrent write. A ceiling keeps a pathological write-storm from spinning forever.
const MAX_INDEX_ATTEMPTS: usize = 5;

/// A cached open snapshot of a namespace. Opening a `QueryNode` costs a manifest read plus WAL
/// head/tail scans (uncached, mutable) — re-doing that per query is what makes a naive server
/// slow. We keep the last snapshot and reuse it while the WAL head has not advanced, so a run
/// of reads between writes is served from memory with a single cheap head check.
struct Snapshot {
    node: Arc<QueryNode>,
    through_seq: u64,
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
}

impl MemlakeService {
    pub fn new(store: Store, tokenizer: Tokenizer) -> Self {
        Self {
            store,
            tokenizer,
            writers: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(HashMap::new()),
            query_limiter: QueryLimiter::new(DEFAULT_MAX_CONCURRENT_QUERIES),
        }
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

    /// Get a current snapshot for `ns`, reusing the cached one while no write has landed
    /// since it was opened — one cheap WAL-head check. A write invalidates the cache for its
    /// namespace, and reads are always strongly consistent, so a reader re-opens promptly and
    /// never serves a stale view.
    async fn snapshot(&self, ns: &Namespace) -> Result<Arc<Snapshot>, Status> {
        if let Some(cached) = self.cached_snapshot(&ns.name) {
            let head = ns.wal_head().await.map_err(internal)?;
            if head == cached.through_seq {
                return Ok(cached);
            }
        }

        let node = Arc::new(
            QueryNode::open(ns, self.tokenizer.clone())
                .await
                .map_err(internal)?,
        );
        let snap = Arc::new(Snapshot {
            through_seq: node.through_seq,
            node,
        });
        self.snapshots.lock().unwrap().insert(ns.name.clone(), snap.clone());
        Ok(snap)
    }

    /// Fold `name`'s WAL tail into the indexed generation until the manifest cursor covers
    /// `seq`, then return the resulting generation. Runs the same fold the background indexer
    /// runs, inline on this replica — the manifest CAS serializes it against any other folder,
    /// so a lost race just means someone else already advanced the cursor. Bounded retries
    /// absorb that race and a head that a concurrent write pushed past our fold.
    async fn index_until(&self, name: &str, seq: u64) -> Result<u64, Status> {
        let ns = self.namespace(name);
        for _ in 0..MAX_INDEX_ATTEMPTS {
            let (manifest, _etag) = ns.read_manifest().await.map_err(internal)?;
            if manifest.wal_index_cursor >= seq {
                // A fold advanced past our write; drop any snapshot opened against the older
                // generation so the next read sees the freshly-indexed one.
                self.invalidate(name);
                return Ok(manifest.version);
            }
            // Inline fold on the write path: auto-select with default budget/threshold.
            fold(
                &ns,
                &self.tokenizer,
                IndexOptions::default(),
                FoldBudget::default(),
                DEFAULT_STREAMING_THRESHOLD_DOCS,
            )
            .await
            .map_err(internal)?;
        }
        Err(Status::deadline_exceeded(format!(
            "write seq {seq} was not indexed after {MAX_INDEX_ATTEMPTS} fold attempts"
        )))
    }

    fn invalidate(&self, name: &str) {
        self.snapshots.lock().unwrap().remove(name);
    }
}

fn internal(e: impl std::fmt::Display) -> Status {
    Status::internal(e.to_string())
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
        // A new write means any cached snapshot is stale; drop it so the next read re-opens
        // and sees this write (via the WAL tail, even before it is indexed).
        self.invalidate(&req.namespace);

        // Optionally fold the write into the indexed generation before returning. Done after
        // the commit so the ack still reflects durability; the extra latency is only paid by
        // callers that opt in.
        let generation = if req.wait_for_index {
            self.index_until(&req.namespace, result.seq).await?
        } else {
            0
        };
        Ok(Response::new(pb::WriteResponse {
            seq: result.seq,
            attempts: result.attempts as u32,
            generation,
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
        let vector = req.vector.as_ref().map(convert::decode_vector).transpose()?;
        let tags = convert::tag_filter(req.tags);
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
        let _permit = self.query_limiter.acquire().await;

        let ns = self.namespace(&req.namespace);
        // Reuse the cached open snapshot when still current (see `snapshot`): a warm read is
        // pure in-memory arm evaluation over the shared Store cache, 0 fresh roundtrips.
        let snap = self.snapshot(&ns).await?;
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

        Ok(Response::new(pb::QueryResponse {
            hits,
            load_roundtrips: metrics.roundtrips() as u32,
        }))
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

        // `get` with `include_vector` fetches whole cluster files, so it shares the query
        // admission cap; the payload-only path is cheap but is gated too for a single, simple
        // memory ceiling over the retrieval paths.
        let _permit = self.query_limiter.acquire().await;

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns).await?;
        let found = snap.node.get_many(&ids, req.include_vector).await.map_err(internal)?;
        Ok(Response::new(pb::GetResponse {
            memories: found
                .into_iter()
                .map(|m| convert::stored_record_with_edges(m, req.include_vector, req.include_edges))
                .collect(),
        }))
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

    async fn scan(
        &self,
        req: Request<pb::ScanRequest>,
    ) -> Result<Response<pb::ScanResponse>, Status> {
        let req = req.into_inner();
        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace is required"));
        }
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
        let mut skip = req.skip as usize;

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns).await?;
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
                .scan(ty, cursor, want, &filter)
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
            self.invalidate(&req.namespace);
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
        self.invalidate(&req.namespace);
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

        // Newest generation first, then by kind, so the current generation's files read as
        // a group above the superseded ones.
        all.sort_by(|a, b| {
            b.0.generation
                .cmp(&a.0.generation)
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

/// Run the indexer over the given namespaces (or all discovered ones) on a fixed interval.
/// Idempotent by construction, so running more than one replica is safe.
/// TTL for the index lease. A fold is normally seconds; this leaves comfortable headroom, and
/// a crashed holder's lease becomes stealable this long after its last acquire.
const INDEX_LEASE_TTL_SECS: u64 = 60;

#[allow(clippy::too_many_arguments)]
pub async fn run_indexer(
    store: Store,
    tokenizer: Tokenizer,
    namespaces: Vec<String>,
    interval: std::time::Duration,
    lease_holder: String,
    budget: FoldBudget,
    streaming_threshold: usize,
) -> anyhow::Result<()> {
    loop {
        let targets = if namespaces.is_empty() {
            discover_namespaces(&store).await.unwrap_or_default()
        } else {
            namespaces.clone()
        };
        for name in &targets {
            let ns = Namespace::new(name, store.clone());
            // Soft lease: skip a namespace a peer is actively folding, so N indexers don't all
            // fold every namespace (doubled compute + doubled S3 PUTs). Best-effort — it fails
            // open, so at worst two nodes fold once and one wins the manifest CAS (safe).
            if !ns.acquire_index_lease(&lease_holder, INDEX_LEASE_TTL_SECS).await {
                tracing::debug!(namespace = name, "index skipped (peer holds lease)");
                continue;
            }
            match fold(&ns, &tokenizer, IndexOptions::default(), budget, streaming_threshold).await {
                Ok(outcome) => tracing::info!(
                    namespace = name,
                    generation = outcome.generation,
                    docs = outcome.doc_count,
                    published = outcome.published,
                    "indexed"
                ),
                Err(e) => tracing::warn!(namespace = name, error = %e, "index failed"),
            }
            ns.release_index_lease(&lease_holder).await;
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
