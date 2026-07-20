//! Reading the un-indexed tail of the log.
//!
//! A generation lags the log by design — indexing is asynchronous. A strongly-consistent
//! query therefore reads the indexed data *and* every entry past the manifest's cursor,
//! folding the two together. That fold is what makes an acked write visible immediately
//! without the write path ever waiting for the indexer (INV-5).
//!
//! The tail is small by construction (compaction triggers at 64 slices), so scanning it
//! exhaustively — brute-force cosine, linear text match, direct membership — is cheaper
//! than maintaining a second index over it.

use std::collections::HashMap;

use futures::future::try_join_all;
use mlake_core::wal::fold_proof_count;
use mlake_core::{Delta, ItemId, Op, StoredItem, WalEntry};

use crate::{parse_wal_seq, Namespace, Result};

/// The un-indexed portion of the log, resolved into current item state.
#[derive(Default, Debug)]
pub struct TailScan {
    /// Items created or replaced in the tail, in final form with patches folded in.
    pub upserts: HashMap<ItemId, StoredItem>,
    /// Items deleted in the tail. These must be subtracted from generation results even
    /// though the generation still contains them.
    pub tombstones: Vec<ItemId>,
    /// Patches for items that live in the generation rather than the tail. The query
    /// layer applies these after materializing the indexed item.
    pub pending_patches: HashMap<ItemId, Vec<Delta>>,
    /// Highest sequence included in this scan — the consistency point of the read.
    pub through_seq: u64,
    pub entries_scanned: usize,
}

impl TailScan {
    /// Whether an id was deleted in the tail.
    pub fn is_tombstoned(&self, id: &ItemId) -> bool {
        self.tombstones.contains(id)
    }

    /// Apply any tail patches to an item materialized from the generation.
    pub fn apply_patches(&self, item: &mut StoredItem) {
        if let Some(deltas) = self.pending_patches.get(&item.id) {
            item.proof_count = fold_proof_count(item.proof_count, deltas.iter().copied());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.upserts.is_empty() && self.tombstones.is_empty() && self.pending_patches.is_empty()
    }
}

/// Reads WAL entries past a cursor.
pub struct WalTail<'a> {
    namespace: &'a Namespace,
}

impl<'a> WalTail<'a> {
    pub fn new(namespace: &'a Namespace) -> Self {
        Self { namespace }
    }

    /// Fetch and fold every entry in `(after_seq, through_seq]`.
    ///
    /// Entries are fetched concurrently — they are independent objects, so the whole tail
    /// is one roundtrip's worth of latency rather than one per entry.
    pub async fn scan(&self, after_seq: u64, through_seq: Option<u64>) -> Result<TailScan> {
        let prefix = format!("{}/wal/", self.namespace.name);
        let mut paths: Vec<(u64, String)> = self
            .namespace
            .store
            .list(&prefix)
            .await?
            .into_iter()
            .filter_map(|p| parse_wal_seq(&p).map(|s| (s, p)))
            .filter(|(seq, _)| *seq > after_seq)
            .filter(|(seq, _)| through_seq.is_none_or(|t| *seq <= t))
            .collect();
        // Ops must be folded in sequence order: a later tombstone has to win over an
        // earlier upsert of the same id.
        paths.sort_by_key(|(seq, _)| *seq);

        let fetches = paths
            .iter()
            .map(|(_, path)| self.namespace.store.get(path, None));
        let objects = try_join_all(fetches).await?;

        let mut entries = Vec::with_capacity(objects.len());
        for obj in &objects {
            entries.push(WalEntry::from_bytes(&obj.bytes)?);
        }
        Ok(fold_entries(&entries))
    }

    /// Scan everything the manifest has not yet folded in.
    pub async fn scan_from_manifest(&self, wal_index_cursor: u64) -> Result<TailScan> {
        self.scan(wal_index_cursor, None).await
    }
}

/// Fold ordered WAL entries into current state.
///
/// Split out from the fetching so it can be unit-tested directly and reused by the
/// indexer, which folds the same slice when building a generation. That shared fold is
/// what keeps a query's view and the indexer's output consistent (SPEC §5).
pub fn fold_entries(entries: &[WalEntry]) -> TailScan {
    let mut scan = TailScan::default();
    for entry in entries {
        scan.through_seq = scan.through_seq.max(entry.seq);
        scan.entries_scanned += 1;
        for op in &entry.ops {
            match op {
                Op::Upsert(item) => {
                    let id = item.id;
                    // A re-upsert revives a tombstoned id: last write wins.
                    scan.tombstones.retain(|t| *t != id);
                    let mut stored = item.clone().into_stored();
                    // Patches recorded before this upsert in the same tail applied to the
                    // *old* item, which no longer exists. Dropping them is what "last
                    // write wins" means for a full-item replace.
                    scan.pending_patches.remove(&id);
                    stored.proof_count = item.proof_count;
                    scan.upserts.insert(id, stored);
                }
                Op::Tombstone { id } => {
                    scan.upserts.remove(id);
                    scan.pending_patches.remove(id);
                    if !scan.tombstones.contains(id) {
                        scan.tombstones.push(*id);
                    }
                }
                Op::Patch { id, deltas } => {
                    if scan.tombstones.contains(id) {
                        // Patching a deleted item is a no-op, not a resurrection.
                        continue;
                    }
                    if let Some(item) = scan.upserts.get_mut(id) {
                        // The item is in the tail: fold immediately.
                        item.proof_count =
                            fold_proof_count(item.proof_count, deltas.iter().copied());
                    } else {
                        // The item lives in the generation: defer until it is materialized.
                        scan.pending_patches
                            .entry(*id)
                            .or_default()
                            .extend(deltas.iter().copied());
                    }
                }
                Op::Guard { .. } => {}
            }
        }
    }
    scan
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlake_core::item::Timestamps;
    use mlake_core::Item;

    fn item(key: &str, proof: u32) -> Item {
        Item {
            id: ItemId::from_key(key),
            vector: vec![1.0, 0.0],
            text: key.to_string(),
            fact_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: proof,
            entity_ids: vec![],
            causal_out: vec![],
        }
    }

    fn entry(seq: u64, ops: Vec<Op>) -> WalEntry {
        WalEntry::new(seq, ops)
    }

    #[test]
    fn later_tombstone_beats_earlier_upsert() {
        let id = ItemId::from_key("a");
        let scan = fold_entries(&[
            entry(1, vec![Op::Upsert(item("a", 0))]),
            entry(2, vec![Op::Tombstone { id }]),
        ]);
        assert!(scan.upserts.is_empty());
        assert!(scan.is_tombstoned(&id));
    }

    #[test]
    fn later_upsert_revives_a_tombstoned_id() {
        let id = ItemId::from_key("a");
        let scan = fold_entries(&[
            entry(1, vec![Op::Upsert(item("a", 0))]),
            entry(2, vec![Op::Tombstone { id }]),
            entry(3, vec![Op::Upsert(item("a", 7))]),
        ]);
        assert!(!scan.is_tombstoned(&id));
        assert_eq!(scan.upserts[&id].proof_count, 7);
    }

    #[test]
    fn patches_fold_into_a_tail_resident_item() {
        let id = ItemId::from_key("a");
        let scan = fold_entries(&[
            entry(1, vec![Op::Upsert(item("a", 5))]),
            entry(2, vec![Op::Patch { id, deltas: vec![Delta::ProofCount(3)] }]),
            entry(3, vec![Op::Patch { id, deltas: vec![Delta::ProofCount(-1)] }]),
        ]);
        assert_eq!(scan.upserts[&id].proof_count, 7);
        // Nothing deferred: the item was fully resolved within the tail.
        assert!(scan.pending_patches.is_empty());
    }

    #[test]
    fn patches_for_indexed_items_are_deferred() {
        let id = ItemId::from_key("indexed");
        let scan = fold_entries(&[
            entry(1, vec![Op::Patch { id, deltas: vec![Delta::ProofCount(2)] }]),
            entry(2, vec![Op::Patch { id, deltas: vec![Delta::ProofCount(3)] }]),
        ]);
        assert!(scan.upserts.is_empty());

        // The query layer materializes the item from the generation, then applies these.
        let mut from_generation = item("indexed", 10).into_stored();
        scan.apply_patches(&mut from_generation);
        assert_eq!(from_generation.proof_count, 15);
    }

    #[test]
    fn an_upsert_discards_earlier_patches_for_the_same_id() {
        // The patches applied to a version of the item that no longer exists; carrying
        // them forward would double-count against the new value.
        let id = ItemId::from_key("a");
        let scan = fold_entries(&[
            entry(1, vec![Op::Patch { id, deltas: vec![Delta::ProofCount(100)] }]),
            entry(2, vec![Op::Upsert(item("a", 1))]),
        ]);
        assert_eq!(scan.upserts[&id].proof_count, 1);
        assert!(scan.pending_patches.is_empty());
    }

    #[test]
    fn patching_a_tombstoned_item_does_not_resurrect_it() {
        let id = ItemId::from_key("a");
        let scan = fold_entries(&[
            entry(1, vec![Op::Tombstone { id }]),
            entry(2, vec![Op::Patch { id, deltas: vec![Delta::ProofCount(5)] }]),
        ]);
        assert!(scan.is_tombstoned(&id));
        assert!(scan.upserts.is_empty());
        assert!(scan.pending_patches.is_empty());
    }

    #[test]
    fn through_seq_reports_the_consistency_point() {
        let scan = fold_entries(&[
            entry(7, vec![Op::Upsert(item("a", 0))]),
            entry(9, vec![Op::Upsert(item("b", 0))]),
        ]);
        assert_eq!(scan.through_seq, 9);
        assert_eq!(scan.entries_scanned, 2);
    }

    #[test]
    fn a_multi_op_entry_is_all_or_nothing() {
        // Both ops come from one entry, so no reader can observe one without the other.
        let scan = fold_entries(&[entry(
            1,
            vec![Op::Upsert(item("a", 0)), Op::Upsert(item("b", 0))],
        )]);
        assert_eq!(scan.upserts.len(), 2);
    }
}
