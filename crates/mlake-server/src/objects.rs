//! The object-storage browser: classify a namespace's stored objects, and decode one.
//!
//! This is the *physical* view of a namespace — what actually sits on S3 — as opposed to
//! the logical view `Stats` gives. Two things it exists to show:
//!
//! * **What each file is.** The engine's on-disk vocabulary (clusters, SSTable index/data
//!   pairs, the FTS split, the manifest) is otherwise invisible.
//! * **What is still referenced.** Every object except the manifest is immutable. The index is
//!   segmented (LSM): a flush appends a new L0 segment and carries the unchanged ones forward by
//!   reference, so publishing does NOT orphan the previous segments — only a compaction (which
//!   merges several segments into one) makes its inputs garbage. Compacted-away segments and
//!   folded WAL entries linger until GC reclaims them.
//!
//! Decoding is deliberately best-effort and its JSON shape is not a contract: it follows
//! the on-disk formats, which change.

use std::collections::HashSet;

use mlake_core::manifest::Manifest;
use mlake_index::sstable::SsTableIndex;
use mlake_store::Store;
use serde_json::{json, Value};
use tonic::Status;

use crate::pb::ObjectKind;

/// A stored object with everything derivable from its key alone.
pub struct Classified {
    pub path: String,
    pub size_bytes: u64,
    pub kind: ObjectKind,
    /// The `seg-{id}` this file belongs to, or empty for the manifest / WAL entries.
    pub segment: String,
    pub memory_type: Option<u8>,
    pub seq: Option<u64>,
}

/// Classify an object by its key. The layout is `{ns}/manifest.json`, `{ns}/wal/{seq}.bin`, a
/// per-segment delete overlay at `{ns}/seg-{id}/tombstones.json`, and the per-memory_type index
/// files at `{ns}/seg-{id}/mt{type}/{file}`.
pub fn classify(namespace: &str, path: &str, size_bytes: u64) -> Classified {
    let rest = path.strip_prefix(namespace).unwrap_or(path).trim_start_matches('/');
    let mut out = Classified {
        path: path.to_string(),
        size_bytes,
        kind: ObjectKind::Unknown,
        segment: String::new(),
        memory_type: None,
        seq: None,
    };

    if rest == "manifest.json" {
        out.kind = ObjectKind::Manifest;
        return out;
    }
    if rest == "index-lease.json" {
        out.kind = ObjectKind::IndexLease;
        return out;
    }
    if rest == "wal-head" {
        out.kind = ObjectKind::WalHead;
        return out;
    }
    if let Some(file) = rest.strip_prefix("wal/") {
        out.kind = ObjectKind::WalEntry;
        out.seq = file.strip_suffix(".bin").and_then(|s| s.parse().ok());
        return out;
    }

    // Everything else lives under a segment: `seg-{id}/...`.
    let mut parts = rest.split('/');
    let Some(seg) = parts.next() else { return out };
    let Some(seg_id) = seg.strip_prefix("seg-") else { return out };
    out.segment = seg_id.to_string();

    // The rest is either a segment-level file (its tombstone overlay) or, under `mt{type}/`, one
    // per-memory_type index file (which may itself nest, as the FTS split does under `fts/`).
    let tail: Vec<&str> = parts.collect();
    out.kind = match tail.as_slice() {
        ["tombstones.json"] => ObjectKind::SegmentTombstones,
        [mt, rest_parts @ ..] if mt.starts_with("mt") => {
            out.memory_type = mt.strip_prefix("mt").and_then(|n| n.parse().ok());
            match rest_parts {
                ["fts", _] => ObjectKind::FtsSplit,
                [name] => match *name {
                    "centroids.bin" => ObjectKind::Centroids,
                    "stats.json" => ObjectKind::Stats,
                    "tags.json" => ObjectKind::TagSummary,
                    "pk.idx" => ObjectKind::PkIndex,
                    "pk.data" => ObjectKind::PkData,
                    "radj.idx" => ObjectKind::RadjIndex,
                    "radj.data" => ObjectKind::RadjData,
                    "entity.idx" => ObjectKind::EntityIndex,
                    "entity.data" => ObjectKind::EntityData,
                    "time.idx" => ObjectKind::TimeIndex,
                    "time.data" => ObjectKind::TimeData,
                    "payload.idx" => ObjectKind::PayloadIndex,
                    "payload.data" => ObjectKind::PayloadData,
                    "rerank.idx" => ObjectKind::RerankIndex,
                    "rerank.data" => ObjectKind::RerankData,
                    n if n.starts_with("cluster-") && n.ends_with(".bin") => ObjectKind::Cluster,
                    n if n.starts_with("cluster-") && n.ends_with(".vec") => ObjectKind::VectorBlock,
                    _ => ObjectKind::Unknown,
                },
                _ => ObjectKind::Unknown,
            }
        }
        _ => ObjectKind::Unknown,
    };
    out
}

/// Every object key the current manifest still points at, plus the manifest itself.
///
/// Anything outside this set is garbage awaiting GC: a superseded generation, or the files
/// of an indexer that lost the manifest CAS race. WAL entries are judged separately — an
/// entry is live while it is at or above the cursor a reader could still be scanning from.
pub fn live_paths(namespace: &str, manifest: &Manifest) -> HashSet<String> {
    let mut live: HashSet<String> = HashSet::new();
    live.insert(format!("{namespace}/manifest.json"));
    // All live segments plus the grace-window (prev) segments — the latter retained on purpose so
    // readers still holding the older manifest do not observe deleted files.
    live.extend(manifest.all_referenced_paths().map(str::to_string));
    live
}

// ---- decoding ---------------------------------------------------------------

/// A decoded object: JSON for display, plus how much of it is shown.
pub struct Decoded {
    pub json: Value,
    pub total_items: u64,
    pub truncated: bool,
    pub undecodable_reason: String,
}

impl Decoded {
    fn whole(json: Value) -> Self {
        Self { json, total_items: 0, truncated: false, undecodable_reason: String::new() }
    }
    fn items(json: Value, total: usize, shown: usize) -> Self {
        Self {
            json,
            total_items: total as u64,
            truncated: shown < total,
            undecodable_reason: String::new(),
        }
    }
    fn undecodable(json: Value, why: &str) -> Self {
        Self {
            json,
            total_items: 0,
            truncated: false,
            undecodable_reason: why.to_string(),
        }
    }
}

/// Decode one object into JSON for display. `limit` bounds items pulled out of a container.
pub async fn decode(
    store: &Store,
    kind: ObjectKind,
    path: &str,
    limit: usize,
) -> Result<Decoded, Status> {
    let bytes = store
        .get(path, None)
        .await
        .map_err(|e| Status::not_found(format!("reading {path}: {e}")))?
        .bytes;

    let decoded = match kind {
        // Already JSON on disk — reparse so the response is pretty-printed and validated
        // rather than echoed blindly.
        ObjectKind::Manifest
        | ObjectKind::Centroids
        | ObjectKind::Stats
        | ObjectKind::TagSummary
        | ObjectKind::SegmentTombstones
        | ObjectKind::IndexLease => match serde_json::from_slice::<Value>(&bytes) {
            Ok(v) => Decoded::whole(summarize_json(kind, v, limit)),
            Err(e) => Decoded::undecodable(json!({}), &format!("not valid JSON: {e}")),
        },

        // The head pointer is a decimal-ASCII u64: the highest committed WAL sequence.
        ObjectKind::WalHead => match std::str::from_utf8(&bytes).map(|s| s.trim().parse::<u64>()) {
            Ok(Ok(head_seq)) => Decoded::whole(json!({ "head_seq": head_seq })),
            _ => Decoded::undecodable(
                json!({ "raw": String::from_utf8_lossy(&bytes) }),
                "not a valid head pointer (expected a decimal sequence number)",
            ),
        },

        ObjectKind::WalEntry => match mlake_core::WalEntry::from_bytes(&bytes) {
            Ok(entry) => {
                let total = entry.ops.len();
                let ops: Vec<Value> = entry.ops.iter().take(limit).map(wal_op_json).collect();
                Decoded::items(
                    json!({ "seq": entry.seq, "op_count": total, "ops": ops }),
                    total,
                    ops.len(),
                )
            }
            Err(e) => Decoded::undecodable(json!({}), &format!("not a WAL entry: {e}")),
        },

        ObjectKind::Cluster => match mlake_ivf::ClusterFile::from_bytes(&bytes) {
            Ok(cf) => {
                let total = cf.items.len();
                let items: Vec<Value> = cf.items.iter().take(limit).map(memory_json).collect();
                Decoded::items(
                    json!({
                        "centroid_id": cf.centroid_id,
                        "member_count": total,
                        "members": items,
                    }),
                    total,
                    items.len(),
                )
            }
            Err(e) => Decoded::undecodable(json!({}), &format!("not a cluster file: {e}")),
        },

        // The sparse index of an SSTable pair: small, loaded whole, and the only part that
        // is self-describing — it names the block boundaries its sibling `.data` is
        // range-read against.
        ObjectKind::PkIndex
        | ObjectKind::RadjIndex
        | ObjectKind::EntityIndex
        | ObjectKind::TimeIndex
        | ObjectKind::PayloadIndex
        | ObjectKind::RerankIndex => match SsTableIndex::parse(&bytes) {
            Ok(idx) => Decoded::whole(json!({
                "record_count": idx.record_count(),
                "note": "sparse index: loaded whole, then one ranged GET per lookup into \
                         the sibling .data object",
            })),
            Err(e) => Decoded::undecodable(json!({}), &format!("not an SSTable index: {e}")),
        },

        // Block data is meaningless without its sibling index — it is addressed by byte
        // range, not read sequentially — so report shape rather than inventing a parse.
        ObjectKind::PkData
        | ObjectKind::RadjData
        | ObjectKind::EntityData
        | ObjectKind::TimeData
        | ObjectKind::PayloadData
        | ObjectKind::RerankData => Decoded::undecodable(
            json!({ "size_bytes": bytes.len() }),
            "SSTable block data: addressed by byte range through its sibling .idx, so it \
             has no standalone decoding. Open the .idx to see the table's shape.",
        ),

        ObjectKind::VectorBlock => Decoded::undecodable(
            json!({ "size_bytes": bytes.len() }),
            "the quantized vector block for one cluster (RaBitQ codes + norms), scanned as a unit \
             by the vector arm — it has no per-record standalone decoding. The exact f32 vectors \
             live in the sibling rerank.data.",
        ),

        ObjectKind::FtsSplit => Decoded::undecodable(
            json!({ "size_bytes": bytes.len() }),
            "a tantivy split — an opaque third-party index format that memlake stores and \
             hands back to tantivy without parsing",
        ),

        ObjectKind::Unknown => Decoded::undecodable(
            json!({ "size_bytes": bytes.len() }),
            "unrecognised object key — memlake does not know what this file is",
        ),
    };
    Ok(decoded)
}

/// Centroid tables are JSON but enormous (k × dim floats). Show their shape and a bounded
/// sample instead of megabytes of numbers.
fn summarize_json(kind: ObjectKind, v: Value, limit: usize) -> Value {
    if kind != ObjectKind::Centroids {
        return v;
    }
    let dim = v.get("dim").and_then(Value::as_u64).unwrap_or(0);
    let vectors = v.get("vectors").and_then(Value::as_array);
    let sizes = v.get("sizes").cloned().unwrap_or(Value::Null);
    let k = vectors.map(|a| a.len()).unwrap_or(0);
    json!({
        "dim": dim,
        "centroid_count": k,
        "sizes": sizes,
        "vectors_note": format!(
            "{k} centroids x {dim} float32 omitted; showing the first {} of the first centroid",
            limit.min(8)
        ),
        "first_centroid_head": vectors
            .and_then(|a| a.first())
            .and_then(Value::as_array)
            .map(|c| c.iter().take(limit.min(8)).cloned().collect::<Vec<_>>())
            .unwrap_or_default(),
    })
}

/// A `(String, String)` list as a JSON object.
fn str_map(kv: &[(String, String)]) -> Value {
    Value::Object(
        kv.iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect::<serde_json::Map<String, Value>>(),
    )
}

fn id_str(id: &mlake_core::MemoryId) -> String {
    id.as_uuid().to_string()
}

fn memory_json(m: &mlake_core::StoredMemory) -> Value {
    json!({
        "id": id_str(&m.id),
        "memory_type": m.memory_type,
        "text": m.text,
        "tags": m.tags,
        "proof_count": m.proof_count,
        // The embedding is the bulk of a memory; report its shape, not 384 floats per row.
        "vector_dim": m.vector.len(),
        "entity_ids": m.entity_ids.iter().map(|e| e.as_uuid().to_string()).collect::<Vec<_>>(),
        "semantic_out": m.semantic_out.len(),
        "causal_out": m.causal_out.len(),
        "metadata": str_map(&m.metadata),
    })
}

fn wal_op_json(op: &mlake_core::Op) -> Value {
    match op {
        mlake_core::Op::Upsert(m) => json!({
            "kind": "upsert",
            "id": id_str(&m.id),
            "memory_type": m.memory_type,
            "text": m.text,
            "tags": m.tags,
            "vector_dim": m.vector.len(),
        }),
        mlake_core::Op::Tombstone { id } => json!({ "kind": "tombstone", "id": id_str(id) }),
        mlake_core::Op::Patch { id, deltas } => json!({
            "kind": "patch",
            "id": id_str(id),
            "deltas": deltas.iter().map(delta_name).collect::<Vec<_>>(),
        }),
        mlake_core::Op::Guard { expect_seq_lt } => {
            json!({ "kind": "guard", "expect_seq_lt": expect_seq_lt })
        }
        mlake_core::Op::TombstoneWhere { predicate } => json!({
            "kind": "tombstone_where",
            "memory_types": predicate.memory_types,
            "metadata_equals": str_map(&predicate.metadata_equals),
            "tags": predicate.tags,
        }),
    }
}

fn delta_name(d: &mlake_core::Delta) -> &'static str {
    match d {
        mlake_core::Delta::ProofCount(_) => "proof_count",
        mlake_core::Delta::SetText(_) => "set_text",
        mlake_core::Delta::SetVector(_) => "set_vector",
        mlake_core::Delta::SetTags(_) => "set_tags",
        mlake_core::Delta::SetEntityIds(_) => "set_entity_ids",
        mlake_core::Delta::SetTimestamps(_) => "set_timestamps",
        mlake_core::Delta::MergeMetadata(_) => "merge_metadata",
        mlake_core::Delta::Touch(_) => "touch",
        mlake_core::Delta::MergeTimestamps(_) => "merge_timestamps",
    }
}
