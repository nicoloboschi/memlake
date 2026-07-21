//! WAL entry format.
//!
//! One entry is one atomic transaction: every op in it becomes visible to every reader at
//! the same instant, replacing the per-document Postgres transaction of the reference
//! implementation.

use rkyv::{Archive, Deserialize, Serialize};

use crate::id::{EntityId, MemoryId};
use crate::memory::{Memory, StoredMemory, Timestamps};

/// Alignment rkyv requires to read an archived `WalEntry` in place.
const ARCHIVE_ALIGNMENT: usize = 8;

/// A partial update to a memory. `ProofCount` is a commutative *relative* delta (read-
/// modify-write is forbidden on the write path, SPEC §4); the `Set*` variants are *absolute*
/// field replacements, applied last-write-wins in op order. Together they express a partial
/// update without re-sending the whole memory — text/embedding/tags/timestamps from
/// consolidation and curation, plus arbitrary metadata (merged, not replaced).
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug)]
#[archive(check_bytes)]
pub enum Delta {
    /// Increment (or decrement, if negative) the proof count.
    ProofCount(i32),
    SetText(String),
    SetVector(Vec<f32>),
    SetTags(Vec<String>),
    SetEntityIds(Vec<EntityId>),
    SetTimestamps(Timestamps),
    /// Upsert these metadata keys (merge — other keys are untouched).
    MergeMetadata(Vec<(String, String)>),
}

/// Apply one delta to a stored memory in place.
pub fn apply_delta(item: &mut StoredMemory, delta: &Delta) {
    match delta {
        Delta::ProofCount(n) => {
            item.proof_count = (item.proof_count as i64 + *n as i64).clamp(0, u32::MAX as i64) as u32;
        }
        Delta::SetText(t) => item.text = t.clone(),
        Delta::SetVector(v) => item.vector = v.clone(),
        Delta::SetTags(t) => item.tags = t.clone(),
        Delta::SetEntityIds(e) => {
            item.entity_ids = e.clone();
            item.entity_ids.sort_unstable();
            item.entity_ids.dedup();
        }
        Delta::SetTimestamps(ts) => item.timestamps = *ts,
        Delta::MergeMetadata(pairs) => {
            for (k, v) in pairs {
                match item.metadata.iter_mut().find(|(ek, _)| ek == k) {
                    Some(existing) => existing.1 = v.clone(),
                    None => item.metadata.push((k.clone(), v.clone())),
                }
            }
            item.metadata.sort_by(|a, b| a.0.cmp(&b.0));
        }
    }
}

/// Apply a sequence of deltas in order.
pub fn apply_deltas(item: &mut StoredMemory, deltas: &[Delta]) {
    for d in deltas {
        apply_delta(item, d);
    }
}

#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug)]
#[archive(check_bytes)]
pub enum Op {
    Upsert(Memory),
    Tombstone { id: MemoryId },
    Patch { id: MemoryId, deltas: Vec<Delta> },
    /// Delete every memory matching `predicate` whose last write is *older* than this entry's
    /// sequence. Atomic (one entry), race-closed (a concurrent or same-entry upsert with an
    /// equal/higher seq survives), and lazy: evaluated at read against the active predicates
    /// and materialized at the next fold. Put it in the same entry as the re-ingest's upserts
    /// to replace a document's facts atomically — the new upserts share this seq, so they are
    /// not deleted.
    TombstoneWhere { predicate: crate::predicate::Predicate },
    /// Optimistic precondition: the entry is only valid if the WAL head was below this
    /// sequence when it was written. Lets a client express compare-and-set without a lock.
    Guard { expect_seq_lt: u64 },
}

#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug)]
#[archive(check_bytes)]
pub struct WalEntry {
    pub seq: u64,
    pub ops: Vec<Op>,
}

impl WalEntry {
    pub fn new(seq: u64, ops: Vec<Op>) -> Self {
        Self { seq, ops }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, crate::Error> {
        rkyv::to_bytes::<_, 4096>(self)
            .map(|b| b.into_vec())
            .map_err(|e| crate::Error::Encode(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::Error> {
        // rkyv reads the buffer in place, so it must satisfy the archive's alignment.
        // Bytes handed back by an HTTP response body carry no alignment guarantee — this
        // is why the in-memory store worked while S3 failed — so realign by copying when
        // needed. The copy only happens on the WAL tail path, which is bounded and small;
        // the warm path reads mmapped generation files, which are page-aligned already.
        if (bytes.as_ptr() as usize).is_multiple_of(ARCHIVE_ALIGNMENT) {
            Self::from_aligned_bytes(bytes)
        } else {
            let mut aligned = rkyv::AlignedVec::with_capacity(bytes.len());
            aligned.extend_from_slice(bytes);
            Self::from_aligned_bytes(&aligned)
        }
    }

    fn from_aligned_bytes(bytes: &[u8]) -> Result<Self, crate::Error> {
        let archived = rkyv::check_archived_root::<WalEntry>(bytes)
            .map_err(|e| crate::Error::Decode(e.to_string()))?;
        Deserialize::<WalEntry, _>::deserialize(archived, &mut rkyv::Infallible)
            .map_err(|e| crate::Error::Decode(format!("{e:?}")))
    }
}

/// Folds the proof-count deltas in a stream into a starting value (other delta kinds are
/// ignored — they are absolute field sets, not counters).
pub fn fold_proof_count<'a>(start: u32, deltas: impl Iterator<Item = &'a Delta>) -> u32 {
    let mut acc = start as i64;
    for d in deltas {
        if let Delta::ProofCount(n) = d {
            acc += *n as i64;
        }
    }
    acc.clamp(0, u32::MAX as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::EntityId;
    use crate::memory::Timestamps;

    fn item(key: &str) -> Memory {
        Memory {
            id: MemoryId::from_key(key),
            vector: vec![0.1, 0.2, 0.3],
            text: format!("text for {key}"),
            memory_type: 1,
            tags: vec!["t".into()],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![EntityId::from_bytes([1; 16]), EntityId::from_bytes([2; 16])],
            causal_out: vec![],
            metadata: vec![],
        }
    }

    #[test]
    fn entry_roundtrips_through_rkyv() {
        let entry = WalEntry::new(
            7,
            vec![
                Op::Upsert(item("a")),
                Op::Tombstone { id: MemoryId::from_key("b") },
                Op::Patch {
                    id: MemoryId::from_key("c"),
                    deltas: vec![Delta::ProofCount(1)],
                },
                Op::Guard { expect_seq_lt: 9 },
            ],
        );
        let bytes = entry.to_bytes().unwrap();
        assert_eq!(WalEntry::from_bytes(&bytes).unwrap(), entry);
    }

    #[test]
    fn decodes_from_a_misaligned_buffer() {
        // Reproduces the S3 path: response bodies are not alignment-guaranteed, and an
        // underaligned buffer made rkyv reject an otherwise valid entry.
        let entry = WalEntry::new(1, vec![Op::Upsert(item("a"))]);
        let encoded = entry.to_bytes().unwrap();

        // Force a deliberately misaligned view of the same bytes.
        let mut padded = vec![0u8; encoded.len() + 1];
        padded[1..].copy_from_slice(&encoded);
        let misaligned = &padded[1..];
        assert_ne!(
            misaligned.as_ptr() as usize % ARCHIVE_ALIGNMENT,
            0,
            "test setup must actually produce a misaligned slice"
        );

        assert_eq!(WalEntry::from_bytes(misaligned).unwrap(), entry);
    }

    #[test]
    fn corrupt_bytes_are_rejected_not_panicking() {
        // check_archived_root validates rather than trusting the buffer, so a truncated
        // or garbage object yields an error instead of unsound zero-copy access.
        assert!(WalEntry::from_bytes(&[0xff; 32]).is_err());
        assert!(WalEntry::from_bytes(&[]).is_err());
    }

    #[test]
    fn proof_count_deltas_are_order_independent() {
        let a = fold_proof_count(5, [Delta::ProofCount(3), Delta::ProofCount(-2)].iter());
        let b = fold_proof_count(5, [Delta::ProofCount(-2), Delta::ProofCount(3)].iter());
        assert_eq!(a, b);
        assert_eq!(a, 6);
    }

    #[test]
    fn proof_count_saturates_at_zero() {
        assert_eq!(fold_proof_count(1, [Delta::ProofCount(-5)].iter()), 0);
    }
}
