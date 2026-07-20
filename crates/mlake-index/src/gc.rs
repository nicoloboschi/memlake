//! Garbage collection (SPEC §5.4).
//!
//! Two kinds of file become reclaimable over time: generation files no longer referenced
//! by the manifest (an old generation superseded by a newer one), and WAL entries already
//! folded into a generation. Both are safe to delete only after they can no longer be
//! read by an in-flight query — which for generations means keeping `prev_generation` as a
//! grace window, and for the WAL means keeping everything past the manifest cursor.
//!
//! GC is idempotent and runs on any node: deleting an already-deleted file is a no-op, so
//! two nodes collecting concurrently do not conflict (INV-6).

use mlake_core::manifest::generation_prefix;
use mlake_wal::{parse_wal_seq, Namespace};

use crate::Result;

/// What a GC pass reclaimed.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct GcOutcome {
    pub generation_files_deleted: usize,
    pub wal_entries_deleted: usize,
}

/// Reclaim unreferenced files for a namespace.
///
/// Deletes:
/// * generation files under `gen-{G}/` for any `G` older than `prev_generation` — the
///   current and previous generations are both retained so a reader mid-flight on the
///   previous manifest still sees a complete tree;
/// * WAL entries at or below `wal_index_cursor`, which are durably folded into the
///   current generation and can no longer be needed by any consistent read.
pub async fn gc(ns: &Namespace) -> Result<GcOutcome> {
    let (manifest, _etag) = ns.read_manifest().await?;
    let mut outcome = GcOutcome::default();

    // The oldest generation still referenced. Keep it and everything at or above it.
    let keep_from = manifest.prev_generation.unwrap_or(manifest.generation);

    // Delete stale generation prefixes.
    for g in 0..keep_from {
        let prefix = generation_prefix(&ns.name, g);
        let paths = ns.store.list(&prefix).await?;
        for path in paths {
            ns.store.delete(&path).await?;
            outcome.generation_files_deleted += 1;
        }
    }

    // Delete folded WAL entries. Everything at or below the cursor is in the generation;
    // the query node only ever scans strictly past the cursor.
    let wal_prefix = format!("{}/wal/", ns.name);
    for path in ns.store.list(&wal_prefix).await? {
        if let Some(seq) = parse_wal_seq(&path) {
            if seq <= manifest.wal_index_cursor {
                ns.store.delete(&path).await?;
                outcome.wal_entries_deleted += 1;
            }
        }
    }

    Ok(outcome)
}
