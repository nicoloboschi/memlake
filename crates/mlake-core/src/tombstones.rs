//! A segment's delete overlay.
//!
//! When a flush appends an L0 segment, the deletes and re-upserts in its WAL slice must keep
//! hiding OLDER segments' copies of those ids. Segments are seq-ordered (a newer segment covers
//! higher WAL sequences), so this is position-based: an item in an older segment is hidden if a
//! newer segment's `superseded` set contains its id. Predicate-deletes stay seq-scoped and are
//! applied post-hydration (they need the full item). Small — bounded by the flush slice's
//! deletes/re-upserts, not the corpus — so it loads whole at query open.

use rkyv::{Archive, Deserialize, Serialize};

use crate::{MemoryId, Predicate};

#[derive(Archive, Deserialize, Serialize, Clone, Default, Debug)]
#[archive(check_bytes)]
pub struct SegmentTombstones {
    /// Ids this segment kills in older segments (deletes + re-upserts).
    pub superseded: Vec<MemoryId>,
    /// Predicate-deletes in this segment's slice, with the seq they were issued at.
    pub predicates: Vec<(u64, Predicate)>,
}
