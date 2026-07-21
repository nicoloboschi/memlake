//! WAL entry format.
//!
//! One entry is one atomic transaction: every op in it becomes visible to every reader at
//! the same instant, replacing the per-document Postgres transaction of the reference
//! implementation.

use rkyv::{Archive, Deserialize, Serialize};

use crate::id::MemoryId;
use crate::memory::Memory;

/// Alignment rkyv requires to read an archived `WalEntry` in place.
const ARCHIVE_ALIGNMENT: usize = 8;

/// A commutative delta. Read-modify-write is forbidden on the write path (SPEC §4), so
/// mutations that would otherwise need a read are expressed as deltas that fold to the
/// same result regardless of the order they are applied in.
#[derive(Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Debug)]
#[archive(check_bytes)]
pub enum Delta {
    /// Increment (or decrement, if negative) the proof count.
    ProofCount(i32),
}

#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug)]
#[archive(check_bytes)]
pub enum Op {
    Upsert(Memory),
    Tombstone { id: MemoryId },
    Patch { id: MemoryId, deltas: Vec<Delta> },
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

/// Folds a stream of deltas into a starting value.
pub fn fold_proof_count(start: u32, deltas: impl Iterator<Item = Delta>) -> u32 {
    let mut acc = start as i64;
    for d in deltas {
        match d {
            Delta::ProofCount(n) => acc += n as i64,
        }
    }
    acc.clamp(0, u32::MAX as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
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
            entity_ids: vec![1, 2],
            causal_out: vec![],
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
        let a = fold_proof_count(5, [Delta::ProofCount(3), Delta::ProofCount(-2)].into_iter());
        let b = fold_proof_count(5, [Delta::ProofCount(-2), Delta::ProofCount(3)].into_iter());
        assert_eq!(a, b);
        assert_eq!(a, 6);
    }

    #[test]
    fn proof_count_saturates_at_zero() {
        assert_eq!(fold_proof_count(1, [Delta::ProofCount(-5)].into_iter()), 0);
    }
}
