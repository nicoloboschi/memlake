//! The stateless query node (SPEC §6).
//!
//! Holds no durable state: it reads a namespace's manifest, loads the generation it points
//! at from object storage (through the etag-keyed cache), and merges the un-indexed WAL
//! tail over it. Because everything comes from S3, a freshly started node serves the same
//! results as a warm one — losing a node's disk costs latency, never correctness (INV-4).
//!
//! Strong consistency (the default) reads the WAL head at query time and scans the tail
//! past the manifest cursor, so every acked write is reflected (INV-5). Eventual
//! consistency skips the head check and serves from the cached manifest.

use std::collections::HashSet;

use mlake_core::StoredItem;
use mlake_fts::Tokenizer;
use mlake_store::QueryMetrics;
use mlake_wal::{Namespace, WalTail};

use crate::engine::Engine;
use crate::fusion::FusedHit;
use crate::generation::read_generation;
use crate::{QueryConfig, Result};

/// Consistency level for a read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Consistency {
    /// Reflect every acked write: check the WAL head and scan the tail (default).
    Strong,
    /// Serve from the cached manifest without a head check; staleness bounded by the
    /// indexing interval.
    Eventual,
}

/// A loaded, queryable snapshot of a namespace at a consistency point.
pub struct QueryNode {
    engine: Engine,
    /// The WAL sequence this snapshot reflects.
    pub through_seq: u64,
    /// Roundtrips consumed loading this snapshot, for the budget check.
    pub load_roundtrips: usize,
}

impl QueryNode {
    /// Open a snapshot of a namespace: load the generation and merge the WAL tail.
    ///
    /// The generation is loaded in full here (correct for any size, ideal for small
    /// namespaces). The strict per-probed-cluster roundtrip path is a refinement over the
    /// same file layout.
    pub async fn open(
        ns: &Namespace,
        tokenizer: Tokenizer,
        consistency: Consistency,
    ) -> Result<Self> {
        let metrics = QueryMetrics::new();

        // RT1: manifest (+ WAL head, for strong consistency).
        let (manifest, _etag) = ns.read_manifest().await?;
        let head = if consistency == Consistency::Strong {
            ns.wal_head().await?
        } else {
            manifest.wal_head
        };

        // Load the generation unless the namespace has never been indexed.
        let mut live: Vec<StoredItem> = if manifest.is_empty() {
            Vec::new()
        } else {
            let generation =
                read_generation(&ns.store, &manifest.files, manifest.generation, Some(&metrics))
                    .await?;
            generation.clusters.into_iter().flatten().collect()
        };

        // Merge the WAL tail: apply tombstones, add/replace upserts, fold patches.
        let scan = WalTail::new(ns)
            .scan(manifest.wal_index_cursor, Some(head))
            .await?;

        // Drop tombstoned generation items.
        live.retain(|item| !scan.is_tombstoned(&item.id));
        // Apply deferred patches to surviving generation items.
        for item in live.iter_mut() {
            scan.apply_patches(item);
        }
        // Replace/append tail upserts (patches already folded into them).
        let tail_ids: HashSet<_> = scan.upserts.keys().copied().collect();
        live.retain(|item| !tail_ids.contains(&item.id));
        live.extend(scan.upserts.into_values());

        let engine = Engine::build(live, tokenizer);

        Ok(Self {
            engine,
            through_seq: head,
            load_roundtrips: metrics.roundtrips(),
        })
    }

    /// Answer a fused query over the snapshot.
    pub fn query(
        &self,
        vector: Option<&[f32]>,
        text: Option<&str>,
        top_k: usize,
        config: QueryConfig,
    ) -> Vec<FusedHit> {
        self.engine.query(vector, text, top_k, config)
    }

    pub fn doc_count(&self) -> usize {
        self.engine.len()
    }
}
