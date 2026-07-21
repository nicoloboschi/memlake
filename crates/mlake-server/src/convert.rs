//! Conversions between the wire types (`crate::pb`) and the core domain types. Kept in one
//! place so the service impl stays about orchestration, not field-shuffling. Every fallible
//! conversion returns `tonic::Status` so it maps straight onto a gRPC error.

use mlake_core::memory::{CausalEdge, LinkType, Timestamps, Weight};
use mlake_core::{Delta, Memory, MemoryId, Op, TagFilter, TagsMatch};
use mlake_index::{Consistency, QueryConfig};
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
/// kept for round-trip tests and any future response that echoes vectors.
#[allow(dead_code)]
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

/// A bare 16-byte id (for tombstone / patch targets, which carry no key fallback).
fn id_exact(bytes: &[u8]) -> Result<MemoryId, Status> {
    let arr: [u8; 16] = bytes
        .try_into()
        .map_err(|_| Status::invalid_argument("id must be exactly 16 bytes"))?;
    Ok(MemoryId::from_bytes(arr))
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
        entity_ids: m.entity_ids,
        causal_out,
    })
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

/// Build a `QueryConfig` from the wire message, treating each zero field as "use the
/// server default" so a client can send an empty message and get sensible fusion.
pub fn query_config(c: Option<pb::QueryConfig>) -> QueryConfig {
    let mut out = QueryConfig::default();
    let Some(c) = c else { return out };
    if c.nprobe != 0 {
        out.nprobe = c.nprobe as usize;
    }
    if c.rrf_k != 0.0 {
        out.rrf_k = c.rrf_k;
    }
    if c.vector_weight != 0.0 {
        out.vector_weight = c.vector_weight;
    }
    if c.fts_weight != 0.0 {
        out.fts_weight = c.fts_weight;
    }
    // graph_weight is special: 0 is a meaningful value (drop the graph arm), so we only
    // keep the default when the whole config message was omitted (handled above).
    out.graph_weight = c.graph_weight;
    if c.arm_depth != 0 {
        out.arm_depth = c.arm_depth as usize;
    }
    out
}

pub fn hit(h: mlake_index::FusedHit) -> pb::Hit {
    pb::Hit {
        id: h.id.0.to_vec(),
        score: h.score,
        contributions: h
            .contributions
            .into_iter()
            .map(|(arm, score)| pb::Contribution { arm, score })
            .collect(),
    }
}
