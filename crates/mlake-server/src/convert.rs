//! Conversions between the wire types (`crate::pb`) and the core domain types. Kept in one
//! place so the service impl stays about orchestration, not field-shuffling. Every fallible
//! conversion returns `tonic::Status` so it maps straight onto a gRPC error.

use mlake_core::memory::{CausalEdge, LinkType, Timestamps, Weight};
use mlake_core::predicate::tags_mode_to_u8;
use mlake_core::{Delta, EntityId, Memory, MemoryId, Op, Predicate, StoredMemory, TagFilter, TagsMatch};
use mlake_index::{ArmDepths, ArmScore, RawHit};
use tonic::Status;

use crate::pb;

/// Decode a raw little-endian f32 blob into a vector. Length must be a multiple of 4.
pub fn decode_vector(v: &pb::Vector) -> Result<Vec<f32>, Status> {
    if v.f32le.len() % 4 != 0 {
        return Err(Status::invalid_argument("vector byte length must be a multiple of 4"));
    }
    Ok(v.f32le
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Encode a vector back to a raw little-endian f32 blob. Symmetric with `decode_vector`;
/// used by the admin reads that echo a stored embedding back (`include_vector`).
pub fn encode_vector(v: &[f32]) -> pb::Vector {
    let mut f32le = Vec::with_capacity(v.len() * 4);
    for x in v {
        f32le.extend_from_slice(&x.to_le_bytes());
    }
    pb::Vector { f32le }
}

/// A 16-byte id, or — if absent — one derived deterministically from `key`.
fn id_from(bytes: &[u8], key: &str) -> Result<MemoryId, Status> {
    if bytes.is_empty() {
        if key.is_empty() {
            return Err(Status::invalid_argument("memory needs either `id` (16 bytes) or `key`"));
        }
        return Ok(MemoryId::from_key(key));
    }
    let arr: [u8; 16] = bytes
        .try_into()
        .map_err(|_| Status::invalid_argument("id must be exactly 16 bytes"))?;
    Ok(MemoryId::from_bytes(arr))
}

/// A bare 16-byte id, for callers that address a memory directly (tombstone and patch
/// targets, and the admin `Get`) and so have no `key` fallback to derive one from.
pub fn id_bytes(bytes: &[u8]) -> Result<MemoryId, Status> {
    id_exact(bytes)
}

/// A bare 16-byte id (for tombstone / patch targets, which carry no key fallback).
fn id_exact(bytes: &[u8]) -> Result<MemoryId, Status> {
    let arr: [u8; 16] = bytes
        .try_into()
        .map_err(|_| Status::invalid_argument("id must be exactly 16 bytes"))?;
    Ok(MemoryId::from_bytes(arr))
}

pub fn entity_ids_in(raw: &[Vec<u8>]) -> Result<Vec<EntityId>, Status> {
    raw.iter()
        .map(|b| {
            let arr: [u8; 16] = b
                .as_slice()
                .try_into()
                .map_err(|_| Status::invalid_argument("entity_id must be exactly 16 bytes"))?;
            Ok(EntityId::from_bytes(arr))
        })
        .collect()
}

fn entity_ids_out(ids: &[EntityId]) -> Vec<Vec<u8>> {
    ids.iter().map(|e| e.0.to_vec()).collect()
}

pub fn memory_type_u8(v: u32) -> Result<u8, Status> {
    u8::try_from(v).map_err(|_| Status::invalid_argument("memory_type must be 0..=255"))
}

fn now_epoch_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn timestamps(t: Option<pb::Timestamps>) -> Timestamps {
    let t = t.unwrap_or_default();
    Timestamps {
        event_date: t.event_date,
        occurred_start: t.occurred_start,
        occurred_end: t.occurred_end,
        mentioned_at: t.mentioned_at,
        updated_at: t.updated_at,
    }
}

fn causal_edge(e: pb::CausalEdge) -> Result<CausalEdge, Status> {
    let link_type = match pb::LinkType::try_from(e.link_type).unwrap_or(pb::LinkType::Causes) {
        pb::LinkType::Causes => LinkType::Causes,
        pb::LinkType::CausedBy => LinkType::CausedBy,
        pb::LinkType::Enables => LinkType::Enables,
        pb::LinkType::Prevents => LinkType::Prevents,
    };
    Ok(CausalEdge {
        target: id_exact(&e.target)?,
        link_type,
        weight: Weight::from_f32(e.weight),
    })
}

pub fn memory(m: pb::Memory) -> Result<Memory, Status> {
    let id = id_from(&m.id, &m.key)?;
    let vector = m.vector.as_ref().map(decode_vector).transpose()?.unwrap_or_default();
    let causal_out = m
        .causal_out
        .into_iter()
        .map(causal_edge)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Memory {
        id,
        vector,
        text: m.text,
        index_text: m.index_text,
        memory_type: memory_type_u8(m.memory_type)?,
        tags: m.tags,
        // Default `updated_at` to now when the client omits it, so the field is
        // reliably populated and an "updated since" window never silently skips
        // memories written by a client that does not set it.
        timestamps: {
            let mut t = timestamps(m.timestamps);
            if t.updated_at.is_none() {
                t.updated_at = Some(now_epoch_ms());
            }
            t
        },
        proof_count: m.proof_count,
        entity_ids: entity_ids_in(&m.entity_ids)?,
        causal_out,
        metadata: m.metadata.into_iter().collect(),
    })
}

// ---- Read-side conversions (StoredMemory -> wire) ----

fn timestamps_out(t: &Timestamps) -> pb::Timestamps {
    pb::Timestamps {
        event_date: t.event_date,
        occurred_start: t.occurred_start,
        occurred_end: t.occurred_end,
        mentioned_at: t.mentioned_at,
        updated_at: t.updated_at,
    }
}

fn causal_edge_out(e: &CausalEdge) -> pb::CausalEdge {
    let link_type = match e.link_type {
        LinkType::Causes => pb::LinkType::Causes,
        LinkType::CausedBy => pb::LinkType::CausedBy,
        LinkType::Enables => pb::LinkType::Enables,
        LinkType::Prevents => pb::LinkType::Prevents,
    };
    pb::CausalEdge {
        target: e.target.0.to_vec(),
        link_type: link_type as i32,
        weight: e.weight.to_f32(),
    }
}

/// The stored memory as a wire payload (vector omitted — large, and the client has it).
fn payload(m: StoredMemory) -> pb::MemoryPayload {
    payload_with_edges(m, false)
}

/// `include_edges` adds the indexer-derived kNN edges. They are already materialized on the
/// memory, so this costs response bytes and nothing else — but the ranking path never uses
/// them, so it stays off by default rather than paying ~18 bytes per edge on every candidate.
fn payload_with_edges(m: StoredMemory, include_edges: bool) -> pb::MemoryPayload {
    let semantic_out = if include_edges {
        m.semantic_out.iter().map(semantic_edge_out).collect()
    } else {
        Vec::new()
    };
    pb::MemoryPayload {
        text: m.text,
        tags: m.tags,
        proof_count: m.proof_count,
        entity_ids: entity_ids_out(&m.entity_ids),
        timestamps: Some(timestamps_out(&m.timestamps)),
        causal_out: m.causal_out.iter().map(causal_edge_out).collect(),
        metadata: m.metadata.into_iter().collect(),
        semantic_out,
    }
}

fn semantic_edge_out(e: &mlake_core::SemanticEdge) -> pb::SemanticEdge {
    pb::SemanticEdge {
        target: e.target.0.to_vec(),
        weight: e.weight.to_f32(),
    }
}

/// A directly-addressed memory (Get / Scan), which carries no arm scores because nothing
/// ranked it. `include_vector` is opt-in: the embedding dominates the response size and a
/// caller browsing memories rarely wants it.
pub fn stored_record(m: StoredMemory, include_vector: bool) -> pb::StoredMemoryRecord {
    stored_record_with_edges(m, include_vector, false)
}

pub fn stored_record_with_edges(
    m: StoredMemory,
    include_vector: bool,
    include_edges: bool,
) -> pb::StoredMemoryRecord {
    pb::StoredMemoryRecord {
        id: m.id.0.to_vec(),
        memory_type: m.memory_type as u32,
        vector: include_vector.then(|| encode_vector(&m.vector)),
        memory: Some(payload_with_edges(m, include_edges)),
    }
}

// ---- Admin views (WAL log, IVF layout) ----

/// One WAL object as a log row. `folded` is the whole point of the view: an entry at or
/// below the manifest's cursor is already in a generation, everything above it is backlog
/// that every query re-scans.
pub fn wal_entry(
    o: &mlake_wal::WalObject,
    wal_index_cursor: u64,
    decoded: Option<&mlake_core::WalEntry>,
) -> pb::WalEntryInfo {
    let mut counts = pb::WalOpCounts::default();
    let mut ops = Vec::new();
    // Counts require decoding the entry, so when the caller did not ask for ops the field
    // is left *absent* rather than zeroed. Zeros would read as "this entry has no ops",
    // which is the opposite of the truth.
    let Some(entry) = decoded else {
        return pb::WalEntryInfo {
            seq: o.seq,
            size_bytes: o.size_bytes,
            counts: None,
            folded: o.seq <= wal_index_cursor,
            ops,
        };
    };
    {
        for op in &entry.ops {
            match op {
                Op::Upsert(m) => {
                    counts.upserts += 1;
                    ops.push(wal_op(pb::wal_op_detail::Kind::Upsert(pb::WalUpsert {
                        id: m.id.0.to_vec(),
                        memory_type: m.memory_type as u32,
                        text: m.text.clone(),
                        tags: m.tags.clone(),
                        vector_dim: m.vector.len() as u32,
                    })));
                }
                Op::Tombstone { id } => {
                    counts.tombstones += 1;
                    ops.push(wal_op(pb::wal_op_detail::Kind::Tombstone(id.0.to_vec())));
                }
                Op::Patch { id, deltas } => {
                    counts.patches += 1;
                    ops.push(wal_op(pb::wal_op_detail::Kind::Patch(patch_detail(id, deltas))));
                }
                Op::Guard { expect_seq_lt } => {
                    counts.guards += 1;
                    ops.push(wal_op(pb::wal_op_detail::Kind::GuardExpectSeqLt(*expect_seq_lt)));
                }
                Op::TombstoneWhere { predicate } => {
                    counts.predicate_tombstones += 1;
                    ops.push(wal_op(pb::wal_op_detail::Kind::TombstoneWhere(
                        predicate_summary(predicate),
                    )));
                }
            }
        }
    }
    pb::WalEntryInfo {
        seq: o.seq,
        size_bytes: o.size_bytes,
        counts: Some(counts),
        folded: o.seq <= wal_index_cursor,
        ops,
    }
}

fn wal_op(kind: pb::wal_op_detail::Kind) -> pb::WalOpDetail {
    pb::WalOpDetail { kind: Some(kind) }
}

/// A patch op as the log view sees it: which memory it targets and which fields it sets.
/// The deltas are the whole content of the op — reporting only the id would make every
/// patch look like a no-op.
fn patch_detail(id: &MemoryId, deltas: &[Delta]) -> pb::Patch {
    let mut p = pb::Patch { id: id.0.to_vec(), ..Default::default() };
    for d in deltas {
        match d {
            Delta::ProofCount(n) => p.proof_count_delta += n,
            Delta::SetText(t) => p.text = Some(t.clone()),
            Delta::SetVector(v) => p.vector = Some(encode_vector(v)),
            Delta::SetTags(tags) => p.tags = Some(pb::TagList { tags: tags.clone() }),
            Delta::SetTimestamps(ts) => {
                p.timestamps = Some(timestamps_out(ts));
                p.replace_timestamps = true;
            }
            Delta::MergeTimestamps(ts) => p.timestamps = Some(timestamps_out(ts)),
            Delta::MergeMetadata(kv) => p.metadata.extend(kv.iter().cloned()),
            // The server's own write-time stamp, not something a client set. It rides in
            // the reported timestamps so the log view shows when the patch landed.
            Delta::Touch(at) => {
                p.timestamps.get_or_insert_with(Default::default).updated_at = Some(*at)
            }
            // Entity ids have no field on the wire `Patch` (clients cannot set them), so
            // there is nothing to report beyond the op being a patch.
            Delta::SetEntityIds(_) => {}
        }
    }
    p
}

/// A predicate delete rendered for the log view. The entry records what the op asked to
/// match, never which memories it hit — that is decided when the fold applies it.
fn predicate_summary(p: &mlake_core::predicate::Predicate) -> String {
    if p.is_empty() {
        return "everything (unconstrained predicate)".into();
    }
    let mut parts = Vec::new();
    if !p.memory_types.is_empty() {
        let types: Vec<String> = p.memory_types.iter().map(|t| t.to_string()).collect();
        parts.push(format!("type={}", types.join(",")));
    }
    for (k, v) in &p.metadata_equals {
        parts.push(format!("meta[{k}={v}]"));
    }
    if !p.tags.is_empty() {
        let mode = match p.tags_mode {
            1 => "ALL",
            2 => "ANY_STRICT",
            3 => "ALL_STRICT",
            4 => "EXACT",
            _ => "ANY",
        };
        parts.push(format!("tags:{mode}[{}]", p.tags.join(",")));
    }
    parts.join(" ")
}

/// The IVF layout as wire clusters. Tag summaries are parallel to the centroids but may be
/// absent on a generation built before they existed, so they are looked up defensively.
pub fn cluster_infos(layout: &mlake_index::ClusterLayout<'_>) -> Vec<pb::ClusterInfo> {
    layout
        .centroids
        .iter()
        .enumerate()
        .map(|(i, centroid)| {
            let summary = layout.tag_summary.get(i);
            pb::ClusterInfo {
                cluster_id: i as u32,
                centroid: Some(encode_vector(centroid)),
                size: layout.sizes.get(i).copied().unwrap_or(0) as u64,
                tags: summary.map(|s| s.tags.clone()).unwrap_or_default(),
                has_untagged: summary.map(|s| s.has_untagged).unwrap_or(false),
            }
        })
        .collect()
}

/// One cached object. `lru_rank` is the entry's position in the returned (MRU-first)
/// order, not the raw LRU counter — the counter is an internal monotonic tick that would
/// mean nothing to a caller.
pub fn cache_entry(rank: usize, e: mlake_store::CacheEntry) -> pb::CacheEntry {
    pb::CacheEntry {
        namespace: e.namespace,
        path: e.path,
        etag: e.etag,
        bytes: e.bytes,
        in_memory: e.in_memory,
        on_disk: e.on_disk,
        lru_rank: rank as u32,
    }
}

/// A stored object as a browser row. `live` is the interesting bit: everything except the
/// manifest is immutable, so superseded generations linger as garbage until GC.
pub fn object_info(c: &crate::objects::Classified, live: bool) -> pb::ObjectInfo {
    pb::ObjectInfo {
        path: c.path.clone(),
        size_bytes: c.size_bytes,
        kind: c.kind as i32,
        generation: c.generation,
        memory_type: c.memory_type.unwrap_or(0) as u32,
        has_memory_type: c.memory_type.is_some(),
        seq: c.seq.unwrap_or(0),
        live,
    }
}

pub fn cluster_member(cluster_id: u32, m: StoredMemory) -> pb::ClusterMember {
    pb::ClusterMember {
        id: m.id.0.to_vec(),
        cluster_id,
        vector: Some(encode_vector(&m.vector)),
        text: m.text,
    }
}

pub fn op(o: pb::Op) -> Result<Op, Status> {
    let kind = o.kind.ok_or_else(|| Status::invalid_argument("op has no kind set"))?;
    Ok(match kind {
        pb::op::Kind::Upsert(m) => Op::Upsert(memory(m)?),
        pb::op::Kind::Tombstone(id) => Op::Tombstone { id: id_exact(&id)? },
        pb::op::Kind::Patch(p) => {
            let id = id_exact(&p.id)?;
            let mut deltas = Vec::new();
            if p.proof_count_delta != 0 {
                deltas.push(Delta::ProofCount(p.proof_count_delta));
            }
            if let Some(t) = p.text {
                deltas.push(Delta::SetText(t));
            }
            if let Some(v) = &p.vector {
                deltas.push(Delta::SetVector(decode_vector(v)?));
            }
            if let Some(tl) = p.tags {
                deltas.push(Delta::SetTags(tl.tags));
            }
            if let Some(ts) = p.timestamps {
                let ts = timestamps(Some(ts));
                // Merge unless the caller explicitly asked to replace: a patch that mentions
                // one timestamp has said nothing about the others, and wiping them is data
                // loss the caller never asked for.
                deltas.push(if p.replace_timestamps {
                    Delta::SetTimestamps(ts)
                } else {
                    Delta::MergeTimestamps(ts)
                });
            }
            if !p.metadata.is_empty() {
                deltas.push(Delta::MergeMetadata(p.metadata.into_iter().collect()));
            }
            // A patch is a write, so it bumps `updated_at` — the same stamp an upsert gets
            // when the client omits one. Last, so it wins over a timestamps delta in the same
            // patch: a client that sets content times has not thereby said when the write
            // happened, and "changed since X" must not be defeatable by a stale client value.
            deltas.push(Delta::Touch(now_epoch_ms()));
            Op::Patch { id, deltas }
        }
        pb::op::Kind::TombstoneWhere(p) => Op::TombstoneWhere { predicate: predicate(p)? },
        pb::op::Kind::GuardExpectSeqLt(seq) => Op::Guard { expect_seq_lt: seq },
    })
}

/// A wire predicate -> the core `Predicate` (tags-mode carried as a `u8` discriminant).
pub fn predicate(p: pb::Predicate) -> Result<Predicate, Status> {
    let memory_types = p
        .memory_types
        .iter()
        .map(|&t| memory_type_u8(t))
        .collect::<Result<Vec<_>, _>>()?;
    let (tags, tags_mode) = match p.tags {
        Some(tf) if !tf.tags.is_empty() => {
            let mode = match pb::TagsMatch::try_from(tf.mode).unwrap_or(pb::TagsMatch::Any) {
                pb::TagsMatch::Any => TagsMatch::Any,
                pb::TagsMatch::All => TagsMatch::All,
                pb::TagsMatch::AnyStrict => TagsMatch::AnyStrict,
                pb::TagsMatch::AllStrict => TagsMatch::AllStrict,
                pb::TagsMatch::Exact => TagsMatch::Exact,
            };
            (tf.tags, tags_mode_to_u8(mode))
        }
        _ => (Vec::new(), 0),
    };
    Ok(Predicate {
        // Delete predicates carry no time window; only reads range over updated_at.
        updated_from: None,
        updated_to: None,
        memory_types,
        metadata_equals: p.metadata_equals.into_iter().collect(),
        tags,
        tags_mode,
    })
}

pub fn tag_filter(f: Option<pb::TagFilter>) -> TagFilter {
    let Some(f) = f else { return TagFilter::none() };
    if f.tags.is_empty() {
        return TagFilter::none();
    }
    let mode = match pb::TagsMatch::try_from(f.mode).unwrap_or(pb::TagsMatch::Any) {
        pb::TagsMatch::Any => TagsMatch::Any,
        pb::TagsMatch::All => TagsMatch::All,
        pb::TagsMatch::AnyStrict => TagsMatch::AnyStrict,
        pb::TagsMatch::AllStrict => TagsMatch::AllStrict,
        pb::TagsMatch::Exact => TagsMatch::Exact,
    };
    TagFilter::new(f.tags, mode)
}

/// Default per-arm candidate depth when the client sends 0.
const DEFAULT_ARM_DEPTH: usize = 100;
/// `nprobe = 0` means "the index decides".
///
/// A probe width is not a client's decision: it trades recall against bytes read, and only
/// the server knows how many clusters exist to probe. A fixed constant is wrong for the
/// same reason — 8 clusters is most of a small index and a sliver of a large one. The
/// snapshot resolves it from its own cluster count (see `QueryNode::resolve_nprobe`).
const NPROBE_FROM_INDEX: usize = 0;

/// Resolve per-arm depths from the wire request, filling server defaults for zero fields.
pub fn arm_depths(vector_top_k: u32, text_top_k: u32, graph_top_k: u32, nprobe: u32) -> ArmDepths {
    let depth = |v: u32| if v == 0 { DEFAULT_ARM_DEPTH } else { v as usize };
    ArmDepths {
        vector: depth(vector_top_k),
        text: depth(text_top_k),
        graph: depth(graph_top_k),
        nprobe: if nprobe == 0 { NPROBE_FROM_INDEX } else { nprobe as usize },
    }
}

fn arm_score(a: Option<ArmScore>) -> pb::ArmScore {
    match a {
        Some(s) => pb::ArmScore { present: true, rank: s.rank, score: s.score },
        None => pb::ArmScore { present: false, rank: 0, score: 0.0 },
    }
}

pub fn raw_hit(memory_type: u8, h: RawHit) -> pb::Hit {
    pb::Hit {
        id: h.id.0.to_vec(),
        memory_type: memory_type as u32,
        dense: Some(arm_score(h.dense)),
        text: Some(arm_score(h.text)),
        graph: Some(arm_score(h.graph)),
        temporal: Some(arm_score(h.temporal)),
        memory: h.memory.map(payload),
    }
}
