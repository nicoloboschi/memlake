//! The gRPC service: a thin, stateless orchestration layer over `Namespace`/`Writer`/
//! `QueryNode`. Any replica can serve any request — all coordination is in object storage —
//! so this holds only process-local conveniences: a shared `Store` (its two-tier cache
//! warms across requests) and a per-namespace `Writer` so sequence claims don't self-contend
//! within a process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mlake_fts::Tokenizer;
use mlake_index::{index, Consistency, IndexOptions, QueryNode, ScanCursor};
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
        let vector = req.vector.as_ref().map(convert::decode_vector).transpose()?;
        let tags = convert::tag_filter(req.tags);
        let consistency = convert::consistency(req.consistency);
        let depths = convert::arm_depths(req.vector_top_k, req.text_top_k, req.graph_top_k, req.nprobe);
        let text: Option<String> = if req.text.is_empty() { None } else { Some(req.text) };
        // Temporal arm runs only when both window bounds are set.
        let temporal_window = match (req.temporal_from, req.temporal_to) {
            (Some(from), Some(to)) => Some((from, to)),
            _ => None,
        };

        let ns = self.namespace(&req.namespace);
        // Reuse the cached open snapshot when still current (see `snapshot`): a warm read is
        // pure in-memory arm evaluation over the shared Store cache, 0 fresh roundtrips.
        let snap = self.snapshot(&ns, consistency).await?;
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
        let per_type = futures::future::try_join_all(types.into_iter().map(|mt| {
            let metrics = &metrics;
            let tags = &tags;
            async move {
                node.query_raw_metered(mt, vref, tref, tags, depths, temporal_window, metrics)
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
        let consistency = convert::consistency(req.consistency);

        // The manifest gives the published index state; the snapshot gives the *live* doc
        // counts, which differ from it by the un-indexed tail and any tombstones.
        let (manifest, _etag) = ns.read_manifest().await.map_err(internal)?;
        let wal_head = ns.wal_head().await.map_err(internal)?;
        let snap = self.snapshot(&ns, consistency).await?;
        let node = &snap.node;

        let types: Vec<pb::TypeStats> = node
            .memory_types()
            .into_iter()
            .map(|mt| pb::TypeStats {
                memory_type: mt as u32,
                doc_count: node.doc_count_of(mt) as u64,
                cluster_count: node.cluster_count_of(mt) as u32,
                train_count: manifest.index(mt).map(|i| i.train_count).unwrap_or(0),
                has_index: manifest.index(mt).is_some(),
            })
            .collect();

        Ok(Response::new(pb::StatsResponse {
            namespace: req.namespace,
            generation: manifest.generation,
            prev_generation: manifest.prev_generation,
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

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns, convert::consistency(req.consistency)).await?;
        let found = snap.node.get_many(&ids).await.map_err(internal)?;
        Ok(Response::new(pb::GetResponse {
            memories: found
                .into_iter()
                .map(|m| convert::stored_record(m, req.include_vector))
                .collect(),
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
        let limit = match req.limit {
            0 => DEFAULT_SCAN_LIMIT,
            n => (n as usize).min(MAX_SCAN_LIMIT),
        };
        let tags = convert::tag_filter(req.tags);

        let ns = self.namespace(&req.namespace);
        let snap = self.snapshot(&ns, convert::consistency(req.consistency)).await?;
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
            let (items, next) = node
                .scan(ty, cursor, limit - out.len(), &tags)
                .await
                .map_err(internal)?;
            out.extend(items.into_iter().map(|m| convert::stored_record(m, req.include_vector)));

            match next {
                // This type still has more and the page is full: stop here.
                Some(c) if out.len() >= limit => {
                    next_token = page_token(node.generation, ty, c);
                    break;
                }
                // Page has room, so continue into this same type's next cluster.
                Some(c) => cursor = c,
                // Type exhausted: move to the next one.
                None => {
                    cursor = ScanCursor::default();
                    if out.len() >= limit && !rest.is_empty() {
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
        let snap = self.snapshot(&ns, Consistency::Strong).await?;
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
        let snap = self.snapshot(&ns, convert::consistency(req.consistency)).await?;
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
}

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
