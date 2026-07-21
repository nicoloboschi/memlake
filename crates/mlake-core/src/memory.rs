//! The memory item — the unit of storage, retrieval and linking.
//!
//! Items are small (a sentence or two plus a vector), so the full payload lives inline in
//! the cluster file. Fetching a seed cluster therefore yields the seed's outgoing
//! adjacency for free: no extra roundtrip is needed to walk links forward (SPEC §3.3).

use rkyv::{Archive, Deserialize, Serialize};

use crate::id::MemoryId;

/// Classification of a memory item. Fusion may combine across fact types, so the
/// discriminants are stable and stored as `u8`.
#[derive(Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[archive(check_bytes)]
#[archive_attr(derive(PartialEq, Eq, Debug))]
#[repr(u8)]
pub enum MemoryType {
    #[default]
    Unspecified = 0,
    Semantic = 1,
    Episodic = 2,
    Procedural = 3,
    Observation = 4,
}

impl MemoryType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Semantic,
            2 => Self::Episodic,
            3 => Self::Procedural,
            4 => Self::Observation,
            _ => Self::Unspecified,
        }
    }
}

/// Kind of causal edge. Mirrors Hindsight's causal link vocabulary.
#[derive(Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[archive(check_bytes)]
#[archive_attr(derive(PartialEq, Eq, Debug))]
#[repr(u8)]
pub enum LinkType {
    Causes = 0,
    CausedBy = 1,
    Enables = 2,
    Prevents = 3,
}

/// Timestamps carried by an item. All epoch milliseconds, all optional.
#[derive(Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[archive(check_bytes)]
pub struct Timestamps {
    pub event_date: Option<i64>,
    pub occurred_start: Option<i64>,
    pub occurred_end: Option<i64>,
    pub mentioned_at: Option<i64>,
}

/// Maximum inline semantic (kNN) links per item, per SPEC §3.3.
pub const MAX_SEMANTIC_OUT: usize = 5;

/// Minimum cosine similarity for a derived semantic link, per SPEC §5.2.
pub const SEMANTIC_LINK_THRESHOLD: f32 = 0.7;

/// An edge weight, stored as the raw bits of an `f16`.
///
/// Weights live in [0.0, 1.0] where f16's ~3 decimal digits are far finer than the
/// scoring needs, so this halves the edge footprint versus `f32`. The bits are held as
/// `u16` rather than `half::f16` because rkyv 0.7 has no `Archive` impl for the latter;
/// conversion is free and total in both directions.
#[derive(Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[archive(check_bytes)]
#[archive_attr(derive(PartialEq, Eq, Debug))]
pub struct Weight(pub u16);

impl Weight {
    pub fn from_f32(v: f32) -> Self {
        Self(half::f16::from_f32(v).to_bits())
    }

    pub fn to_f32(self) -> f32 {
        half::f16::from_bits(self.0).to_f32()
    }
}

impl ArchivedWeight {
    pub fn to_f32(&self) -> f32 {
        half::f16::from_bits(self.0).to_f32()
    }
}

/// A semantic kNN edge. Derived data, written by the indexer — never by a client.
#[derive(Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[archive(check_bytes)]
#[archive_attr(derive(PartialEq, Eq, Debug))]
pub struct SemanticEdge {
    pub target: MemoryId,
    pub weight: Weight,
}

/// A causal edge. Intrinsic data, supplied by the client in the WAL.
#[derive(Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[archive(check_bytes)]
#[archive_attr(derive(PartialEq, Eq, Debug))]
pub struct CausalEdge {
    pub target: MemoryId,
    pub link_type: LinkType,
    pub weight: Weight,
}

/// An item as stored in a cluster file. Read zero-copy via rkyv; never deserialized on
/// the warm path.
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug)]
#[archive(check_bytes)]
pub struct StoredMemory {
    pub id: MemoryId,
    pub vector: Vec<f32>,
    pub text: String,
    pub memory_type: u8,
    pub tags: Vec<String>,
    pub timestamps: Timestamps,
    pub proof_count: u32,
    /// Dictionary-encoded entity ids, sorted ascending so shared-entity counting between
    /// two items is a linear merge rather than a hash-set intersection.
    pub entity_ids: Vec<u64>,
    pub semantic_out: Vec<SemanticEdge>,
    pub causal_out: Vec<CausalEdge>,
}

impl StoredMemory {
    /// Count of entity ids shared with another item. Both sides must be sorted.
    pub fn shared_entity_count(&self, other: &[u64]) -> usize {
        merge_count(&self.entity_ids, other)
    }
}

/// Count elements present in both sorted slices, ignoring duplicates within a side.
pub fn merge_count(a: &[u64], b: &[u64]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// An item as supplied by a client on the write path.
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug)]
#[archive(check_bytes)]
pub struct Memory {
    pub id: MemoryId,
    pub vector: Vec<f32>,
    pub text: String,
    pub memory_type: u8,
    pub tags: Vec<String>,
    pub timestamps: Timestamps,
    pub proof_count: u32,
    pub entity_ids: Vec<u64>,
    /// Causal edges are intrinsic and travel in the WAL. Semantic edges are absent here
    /// by construction: they are derived by the indexer (SPEC §3.2).
    pub causal_out: Vec<CausalEdge>,
}

impl Memory {
    /// Promote a client item into its stored form. `semantic_out` starts empty and is
    /// filled in by the indexer's kNN derivation pass.
    pub fn into_stored(mut self) -> StoredMemory {
        self.entity_ids.sort_unstable();
        self.entity_ids.dedup();
        StoredMemory {
            id: self.id,
            vector: self.vector,
            text: self.text,
            memory_type: self.memory_type,
            tags: self.tags,
            timestamps: self.timestamps,
            proof_count: self.proof_count,
            entity_ids: self.entity_ids,
            semantic_out: Vec::new(),
            causal_out: self.causal_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_entities_counts_intersection() {
        let item = StoredMemory {
            id: MemoryId::from_key("a"),
            vector: vec![],
            text: String::new(),
            memory_type: 0,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![1, 3, 5, 7],
            semantic_out: vec![],
            causal_out: vec![],
        };
        assert_eq!(item.shared_entity_count(&[3, 5, 9]), 2);
        assert_eq!(item.shared_entity_count(&[2, 4]), 0);
        assert_eq!(item.shared_entity_count(&[]), 0);
    }

    #[test]
    fn into_stored_sorts_and_dedups_entities() {
        let item = Memory {
            id: MemoryId::from_key("a"),
            vector: vec![1.0],
            text: "hi".into(),
            memory_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![5, 1, 5, 3],
            causal_out: vec![],
        };
        let stored = item.into_stored();
        assert_eq!(stored.entity_ids, vec![1, 3, 5]);
        // Semantic links are derived, never client-supplied.
        assert!(stored.semantic_out.is_empty());
    }
}
