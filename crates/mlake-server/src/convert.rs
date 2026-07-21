//! Conversions between the wire types (`crate::pb`) and the core domain types. Kept in one
//! place so the service impl stays about orchestration, not field-shuffling. Every fallible
//! conversion returns `tonic::Status` so it maps straight onto a gRPC error.

use mlake_core::memory::{CausalEdge, LinkType, Timestamps, Weight};
use mlake_core::{Delta, EntityId, Memory, MemoryId, Op, StoredMemory, TagFilter, TagsMatch};
use mlake_index::{ArmDepths, ArmScore, Consistency, RawHit};
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

fn entity_ids_in(raw: &[Vec<u8>]) -> Result<Vec<EntityId>, Status> {
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

fn timestamps(t: Option<pb::Timestamps>) -> Timestamps {
    let t = t.unwrap_or_default();
    Timestamps {
        event_date: t.event_date,
        occurred_start: t.occurred_start,
        occurred_end: t.occurred_end,
        mentioned_at: t.mentioned_at,
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
        memory_type: memory_type_u8(m.memory_type)?,
        tags: m.tags,
        timestamps: timestamps(m.timestamps),
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
    pb::MemoryPayload {
        text: m.text,
        tags: m.tags,
        proof_count: m.proof_count,
        entity_ids: entity_ids_out(&m.entity_ids),
        timestamps: Some(timestamps_out(&m.timestamps)),
        causal_out: m.causal_out.iter().map(causal_edge_out).collect(),
        metadata: m.metadata.into_iter().collect(),
    }
}

/// A directly-addressed memory (Get / Scan), which carries no arm scores because nothing
/// ranked it. `include_vector` is opt-in: the embedding dominates the response size and a
/// caller browsing memories rarely wants it.
pub fn stored_record(m: StoredMemory, include_vector: bool) -> pb::StoredMemoryRecord {
    pb::StoredMemoryRecord {
        id: m.id.0.to_vec(),
        memory_type: m.memory_type as u32,
        vector: include_vector.then(|| encode_vector(&m.vector)),
        memory: Some(payload(m)),
    }
}

pub fn op(o: pb::Op) -> Result<Op, Status> {
    let kind = o.kind.ok_or_else(|| Status::invalid_argument("op has no kind set"))?;
    Ok(match kind {
        pb::op::Kind::Upsert(m) => Op::Upsert(memory(m)?),
        pb::op::Kind::Tombstone(id) => Op::Tombstone { id: id_exact(&id)? },
        pb::op::Kind::Patch(p) => Op::Patch {
            id: id_exact(&p.id)?,
            deltas: vec![Delta::ProofCount(p.proof_count_delta)],
        },
        pb::op::Kind::GuardExpectSeqLt(seq) => Op::Guard { expect_seq_lt: seq },
    })
}

pub fn consistency(c: i32) -> Consistency {
    match pb::Consistency::try_from(c).unwrap_or(pb::Consistency::Strong) {
        pb::Consistency::Strong => Consistency::Strong,
        pb::Consistency::Eventual => Consistency::Eventual,
    }
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
/// Default IVF probe width (mirrors `mlake_ivf::DEFAULT_NPROBE`).
const DEFAULT_NPROBE: usize = 8;

/// Resolve per-arm depths from the wire request, filling server defaults for zero fields.
pub fn arm_depths(vector_top_k: u32, text_top_k: u32, graph_top_k: u32, nprobe: u32) -> ArmDepths {
    let depth = |v: u32| if v == 0 { DEFAULT_ARM_DEPTH } else { v as usize };
    ArmDepths {
        vector: depth(vector_top_k),
        text: depth(text_top_k),
        graph: depth(graph_top_k),
        nprobe: if nprobe == 0 { DEFAULT_NPROBE } else { nprobe as usize },
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
