//! WAL entry format.
//!
//! One entry is one atomic transaction: every op in it becomes visible to every reader at
//! the same instant, replacing the per-document Postgres transaction of the reference
//! implementation.

use rkyv::{Archive, Deserialize, Serialize};

use crate::id::{EntityId, MemoryId};
use crate::memory::{Memory, StoredMemory, Timestamps};

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
    /// Stamp the write time (epoch ms), leaving the four content timestamps alone.
    ///
    /// Distinct from `SetTimestamps`, which replaces the whole struct: a patch is a write
    /// and must bump `updated_at`, but it must not thereby wipe the event/occurred/mentioned
    /// times a caller did not mention. The value is baked in at commit rather than read from
    /// a clock during the fold, so replaying the log is deterministic.
    ///
    /// Appended last so a log written before it existed still decodes.
    Touch(i64),
    /// Set the timestamps that are `Some`, leaving the rest alone.
    ///
    /// What a partial update means, and the default the wire `Patch` maps to. `SetTimestamps`
    /// remains the way to *clear* a field, since under a merge a `None` says nothing.
    MergeTimestamps(Timestamps),
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
        Delta::Touch(at) => item.timestamps.updated_at = Some(*at),
        Delta::MergeTimestamps(ts) => {
            let cur = &mut item.timestamps;
            cur.event_date = ts.event_date.or(cur.event_date);
            cur.occurred_start = ts.occurred_start.or(cur.occurred_start);
            cur.occurred_end = ts.occurred_end.or(cur.occurred_end);
            cur.mentioned_at = ts.mentioned_at.or(cur.mentioned_at);
            cur.updated_at = ts.updated_at.or(cur.updated_at);
        }
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

/// rkyv-encode a delta list for storage outside the WAL — the streaming fold spills each patch as
/// an event into an external sort, so it needs a standalone codec. Mirrors
/// [`StoredMemory::to_rkyv_bytes`].
pub fn deltas_to_rkyv_bytes(deltas: &[Delta]) -> Vec<u8> {
    crate::rkyv_write(&deltas.to_vec())
}

/// Decode a delta list written by [`deltas_to_rkyv_bytes`], tolerating an unaligned slice (it is
/// read back out of a spilled record whose start is not 8-byte aligned).
pub fn deltas_from_rkyv_bytes(bytes: &[u8]) -> Option<Vec<Delta>> {
    if bytes.is_empty() {
        return Some(Vec::new());
    }
    crate::rkyv_read(bytes)
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
        // Shared validated, alignment-tolerant rkyv read (see `rkyv_io`). Bytes from an HTTP
        // response body carry no alignment guarantee, and a corrupt object must be an error, not
        // UB — both handled there. `None` (empty or failed validation) maps to a decode error.
        crate::rkyv_read(bytes).ok_or_else(|| crate::Error::Decode("wal entry".into()))
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

    /// The same fixture as [`item`], as the fold sees it after an upsert.
    fn stored(key: &str, timestamps: Timestamps) -> StoredMemory {
        let m = item(key);
        StoredMemory {
            id: m.id,
            vector: m.vector,
            text: m.text,
            index_text: m.index_text,
            memory_type: m.memory_type,
            tags: m.tags,
            timestamps,
            proof_count: m.proof_count,
            entity_ids: m.entity_ids,
            semantic_out: vec![],
            causal_out: m.causal_out,
            metadata: m.metadata,
            write_seq: 0,
        }
    }

    fn item(key: &str) -> Memory {
        Memory {
            id: MemoryId::from_key(key),
            vector: vec![0.1, 0.2, 0.3],
            text: format!("text for {key}"),
            index_text: String::new(),
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
            misaligned.as_ptr() as usize % 8,
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

    #[test]
    fn touch_stamps_the_write_time_and_leaves_content_time_alone() {
        // The distinction the whole delta exists for. `SetTimestamps` replaces the struct, so
        // a patch could not bump `updated_at` without also wiping the event/occurred/mentioned
        // times a caller never mentioned — and "changed since X" would then be trading one
        // silent data loss for another.
        let mut item = stored(
            "a",
            Timestamps {
                event_date: Some(10),
                occurred_start: Some(20),
                occurred_end: Some(30),
                mentioned_at: Some(40),
                updated_at: Some(50),
            },
        );
        apply_delta(&mut item, &Delta::Touch(9_000));
        assert_eq!(item.timestamps.updated_at, Some(9_000));
        assert_eq!(item.timestamps.event_date, Some(10));
        assert_eq!(item.timestamps.occurred_start, Some(20));
        assert_eq!(item.timestamps.occurred_end, Some(30));
        assert_eq!(item.timestamps.mentioned_at, Some(40));
    }

    #[test]
    fn merging_timestamps_leaves_the_ones_not_mentioned_alone() {
        // The bug this delta exists to fix: revising one time must not clear the other three.
        // `SetTimestamps` replaces wholesale, and the client builds its argument from only the
        // fields the caller passed, so a patch that corrects `occurred_start` used to silently
        // null `event_date`, `occurred_end` and `mentioned_at`.
        let full = Timestamps {
            event_date: Some(10),
            occurred_start: Some(20),
            occurred_end: Some(30),
            mentioned_at: Some(40),
            updated_at: Some(50),
        };
        let one = Timestamps { occurred_start: Some(99), ..Default::default() };

        let mut merged = stored("a", full);
        apply_delta(&mut merged, &Delta::MergeTimestamps(one));
        assert_eq!(
            merged.timestamps,
            Timestamps { occurred_start: Some(99), ..full },
            "a merge overwrites what it mentions and nothing else"
        );

        // `SetTimestamps` still clears, which is what makes it the way to null a field.
        let mut replaced = stored("a", full);
        apply_delta(&mut replaced, &Delta::SetTimestamps(one));
        assert_eq!(replaced.timestamps, one);
    }

    #[test]
    fn merging_an_empty_timestamps_changes_nothing() {
        // A merge cannot express "clear", so an all-`None` argument is a no-op rather than a
        // wipe — the property that makes the default safe.
        let full = Timestamps {
            event_date: Some(10),
            occurred_start: Some(20),
            occurred_end: Some(30),
            mentioned_at: Some(40),
            updated_at: Some(50),
        };
        let mut item = stored("a", full);
        apply_delta(&mut item, &Delta::MergeTimestamps(Timestamps::default()));
        assert_eq!(item.timestamps, full);
    }

    #[test]
    fn a_touch_after_set_timestamps_wins() {
        // The order the server emits them in: a client that sets content times has not
        // thereby said when the write happened, so the server's stamp must survive.
        let mut item = stored("a", Timestamps::default());
        apply_deltas(
            &mut item,
            &[
                Delta::SetTimestamps(Timestamps { updated_at: Some(1), ..Default::default() }),
                Delta::Touch(9_000),
            ],
        );
        assert_eq!(item.timestamps.updated_at, Some(9_000));
    }

    #[test]
    fn deltas_codec_round_trips_including_unaligned() {
        let deltas = vec![
            Delta::SetText("hello".into()),
            Delta::ProofCount(3),
            Delta::SetTags(vec!["a".into(), "b".into()]),
            Delta::MergeMetadata(vec![("k".into(), "v".into())]),
            Delta::Touch(9_000),
            Delta::MergeTimestamps(Timestamps { updated_at: Some(7), ..Default::default() }),
        ];
        let bytes = deltas_to_rkyv_bytes(&deltas);
        assert_eq!(deltas_from_rkyv_bytes(&bytes).unwrap(), deltas);

        // The streaming fold reads events back at a non-8-byte-aligned offset (tag + seq header),
        // so decode must copy into an aligned buffer rather than assume alignment.
        let mut framed = vec![0u8; 9];
        framed.extend_from_slice(&bytes);
        assert_eq!(deltas_from_rkyv_bytes(&framed[9..]).unwrap(), deltas);

        // Empty and garbage inputs are handled, not panicked on.
        assert_eq!(deltas_from_rkyv_bytes(&[]).unwrap(), Vec::<Delta>::new());
        assert!(deltas_from_rkyv_bytes(&[0xff; 7]).is_none());
    }
}
