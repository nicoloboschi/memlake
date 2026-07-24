//! Garbage collection (SPEC §5.4).
//!
//! GC is **reference-based**: it keeps exactly the objects the current manifest points at,
//! plus the previous generation's objects (retained in the manifest as the grace window),
//! and reclaims everything else — orphaned files from index attempts that lost the manifest
//! swap, and superseded generations. Because generation files live under a per-attempt
//! nonce prefix, "delete by generation number" is not enough; only the manifest knows which
//! objects are live.
//!
//! Two grace windows protect in-flight readers:
//! * **generations** — the current and previous generation's files are always kept, so a
//!   reader still holding the previous manifest sees a complete tree;
//! * **the WAL** — entries are kept back to the *previous* generation's cursor, since a
//!   strong reader on the previous manifest is scanning `(prev_cursor, head]`.
//!
//! An additional age floor (`min_age`) keeps very recently written objects regardless, so
//! an index run that is mid-flight on another node is never collected out from under it.
//! GC is idempotent and safe to run on any node (INV-6): a delete of an already-deleted or
//! still-referenced-elsewhere object is a no-op.

use std::collections::HashSet;
use std::time::Duration;

use mlake_wal::{parse_wal_seq, Namespace};

use crate::Result;

/// Default age floor: objects younger than this are never collected, covering an index
/// run in progress on another node. Matches the spec's 15-minute file grace.
pub const DEFAULT_MIN_AGE: Duration = Duration::from_secs(15 * 60);

/// What a GC pass reclaimed.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct GcOutcome {
    pub generation_files_deleted: usize,
    pub wal_entries_deleted: usize,
}

/// Reclaim unreferenced objects, keeping anything younger than [`DEFAULT_MIN_AGE`].
pub async fn gc(ns: &Namespace) -> Result<GcOutcome> {
    gc_with_min_age(ns, DEFAULT_MIN_AGE).await
}

/// Reclaim unreferenced objects, keeping anything younger than `min_age`. Tests pass
/// `Duration::ZERO` to exercise deletion deterministically without waiting.
pub async fn gc_with_min_age(ns: &Namespace, min_age: Duration) -> Result<GcOutcome> {
    let (manifest, _etag) = ns.read_manifest().await?;
    let mut outcome = GcOutcome::default();

    // The live object set: every path any fact type's current or previous generation
    // references (the manifest aggregates them).
    let referenced: HashSet<String> = manifest
        .all_referenced_paths()
        .map(|p| p.to_string())
        .collect();

    let cutoff = chrono::Utc::now() - chrono::Duration::from_std(min_age).unwrap_or_default();

    // Collect unreferenced segment objects that are old enough to be safe. Segment files live at
    // `{ns}/seg-{seg_id}/mt{ft}/…`; the WAL is `{ns}/wal/…` and the manifest is
    // `{ns}/manifest.json`, neither of which contains `/seg-`. So a segment object is any listed
    // path under the namespace containing `/seg-`.
    let ns_prefix = format!("{}/", ns.name);
    for (path, modified) in ns.store.list_with_age(&ns.name).await? {
        if !path.starts_with(&ns_prefix) || !path.contains("/seg-") {
            continue;
        }
        if referenced.contains(&path) {
            continue;
        }
        if modified > cutoff {
            continue; // too young: an index run may still be writing this attempt
        }
        ns.store.delete(&path).await?;
        outcome.generation_files_deleted += 1;
    }

    // Delete folded WAL entries, but keep everything a reader on the previous manifest
    // still needs: retain entries above the *previous* generation's cursor.
    let wal_keep_above = manifest.prev_wal_index_cursor;
    let wal_prefix = format!("{}/wal/", ns.name);
    for (path, modified) in ns.store.list_with_age(&wal_prefix).await? {
        let Some(seq) = parse_wal_seq(&path) else {
            continue;
        };
        if seq > wal_keep_above {
            continue; // still inside a previous-manifest reader's scan window
        }
        if modified > cutoff {
            continue;
        }
        ns.store.delete(&path).await?;
        outcome.wal_entries_deleted += 1;
    }

    Ok(outcome)
}

/// Default retention for observability trace batches: keep the last 24h, drop older.
pub const DEFAULT_TRACE_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);

/// How long a node's rollup may go un-refreshed before it is considered gone and reaped. Live nodes
/// rewrite their rollup every flush (sub-second), so an hour of silence is unambiguous; a node that
/// comes back republishes immediately.
pub const DEFAULT_ROLLUP_STALE: Duration = Duration::from_secs(60 * 60);

/// What an observability sweep reclaimed.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ObsGcOutcome {
    pub trace_objects_deleted: usize,
    pub hour_buckets_deleted: usize,
    pub rollups_deleted: usize,
}

/// Expire observability data: whole trace hour-buckets past `retention`, and rollups of nodes that
/// have gone silent for `rollup_stale`.
///
/// Trace batches are append-only and immutable, partitioned as
/// `_obs/traces/{node}/{YYYYMMDDHH}/{ms}-{seq}.jsonl`. That partition is the point: expiry is
/// decided from the bucket NAME (lexicographic == chronological), so this lists objects ONLY inside
/// buckets it is about to delete — live data is never enumerated, which is what keeps the sweep O(
/// expired) instead of O(retention window).
///
/// Rollups are small overwritten objects (`_obs/rollup/{node}.json`); nothing else reaps them, so a
/// scaled-down or renamed node would otherwise linger forever as a stale card in the admin.
///
/// Idempotent and safe to run concurrently on any node: deleting an already-deleted object is a
/// no-op. Global (not per-namespace) — the indexer runs it on its GC cadence.
pub async fn gc_traces(
    store: &mlake_store::Store,
    retention: Duration,
    rollup_stale: Duration,
) -> Result<ObsGcOutcome> {
    let mut outcome = ObsGcOutcome::default();
    let now = chrono::Utc::now();

    // Buckets strictly older than this one are entirely expired. Comparing bucket names means we
    // never look at an object's metadata, and never touch a live bucket.
    let cutoff_bucket = (now - chrono::Duration::from_std(retention).unwrap_or_default())
        .format("%Y%m%d%H")
        .to_string();

    for node_prefix in store.list_prefixes(mlake_core::OBS_TRACES_PREFIX).await? {
        for hour_prefix in store.list_prefixes(&node_prefix).await? {
            // `.../{node}/{YYYYMMDDHH}/` -> the bucket name.
            let bucket = hour_prefix.trim_end_matches('/').rsplit('/').next().unwrap_or("");
            if bucket.len() != 10 || bucket >= cutoff_bucket.as_str() {
                continue; // still inside the retention window (or not a bucket we wrote)
            }
            for path in store.list(&hour_prefix).await? {
                store.delete(&path).await?;
                outcome.trace_objects_deleted += 1;
            }
            outcome.hour_buckets_deleted += 1;
        }
    }

    // Reap rollups of nodes that stopped publishing.
    let rollup_cutoff = now - chrono::Duration::from_std(rollup_stale).unwrap_or_default();
    for (path, modified) in store.list_with_age(mlake_core::OBS_ROLLUP_PREFIX).await? {
        if modified > rollup_cutoff {
            continue;
        }
        store.delete(&path).await?;
        outcome.rollups_deleted += 1;
    }

    Ok(outcome)
}
