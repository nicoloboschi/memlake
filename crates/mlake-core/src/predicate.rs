//! A match predicate for bulk operations (delete-by-predicate).
//!
//! Carried inside a WAL op (`Op::TombstoneWhere`), so it must be rkyv-serializable — hence
//! the tag-match mode is a plain `u8` discriminant rather than the serde-only [`TagsMatch`].
//! Evaluated at read against the few active predicates, and materialized at the next fold.

use rkyv::{Archive, Deserialize, Serialize};

use crate::memory::StoredMemory;
use crate::tags::{TagFilter, TagsMatch};

/// A conjunction (AND) of conditions on a memory: its type, its metadata, and its tags. An
/// empty predicate matches every memory (used for "delete all").
#[derive(Archive, Deserialize, Serialize, Clone, PartialEq, Debug, Default)]
#[archive(check_bytes)]
pub struct Predicate {
    /// Restrict to these memory_types; empty means any.
    pub memory_types: Vec<u8>,
    /// Metadata key=value pairs that must ALL be present (e.g. `document_id` + `chunk_id`).
    pub metadata_equals: Vec<(String, String)>,
    /// Tag filter values, interpreted per `tags_mode`. Empty means no tag condition.
    pub tags: Vec<String>,
    /// `TagsMatch` discriminant: 0 Any, 1 All, 2 AnyStrict, 3 AllStrict, 4 Exact.
    pub tags_mode: u8,
    /// Inclusive-exclusive window on `timestamps.updated_at` (epoch ms): a memory matches
    /// when its write time is strictly after `updated_from` and strictly before
    /// `updated_to`. A memory with no `updated_at` fails a bounded window rather than
    /// passing it — an unknown write time cannot be shown to fall inside one.
    pub updated_from: Option<i64>,
    pub updated_to: Option<i64>,
}

impl Predicate {
    /// True when the predicate constrains nothing — it then matches every memory.
    pub fn is_empty(&self) -> bool {
        self.memory_types.is_empty()
            && self.metadata_equals.is_empty()
            && self.tags.is_empty()
            && self.updated_from.is_none()
            && self.updated_to.is_none()
    }

    /// Whether `m` satisfies every condition.
    pub fn matches(&self, m: &StoredMemory) -> bool {
        if !self.memory_types.is_empty() && !self.memory_types.contains(&m.memory_type) {
            return false;
        }
        if !self.tags.is_empty() {
            let tf = TagFilter::new(self.tags.clone(), tags_mode_from_u8(self.tags_mode));
            if !tf.matches(&m.tags) {
                return false;
            }
        }
        if self.updated_from.is_some() || self.updated_to.is_some() {
            let Some(updated) = m.timestamps.updated_at else {
                return false;
            };
            if let Some(from) = self.updated_from {
                if updated <= from {
                    return false;
                }
            }
            if let Some(to) = self.updated_to {
                if updated >= to {
                    return false;
                }
            }
        }
        self.metadata_equals
            .iter()
            .all(|(k, v)| m.metadata.iter().any(|(mk, mv)| mk == k && mv == v))
    }
}

/// Map a `u8` discriminant back to a [`TagsMatch`].
pub fn tags_mode_from_u8(n: u8) -> TagsMatch {
    match n {
        1 => TagsMatch::All,
        2 => TagsMatch::AnyStrict,
        3 => TagsMatch::AllStrict,
        4 => TagsMatch::Exact,
        _ => TagsMatch::Any,
    }
}

/// Map a [`TagsMatch`] to its `u8` discriminant.
pub fn tags_mode_to_u8(m: TagsMatch) -> u8 {
    match m {
        TagsMatch::Any => 0,
        TagsMatch::All => 1,
        TagsMatch::AnyStrict => 2,
        TagsMatch::AllStrict => 3,
        TagsMatch::Exact => 4,
    }
}
