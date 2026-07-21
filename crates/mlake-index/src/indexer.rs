//! The indexer: fold a WAL slice into a new generation (SPEC §5).
//!
//! Idempotent and coordination-free: any node may run it, and two nodes racing from the
//! same input produce *equivalent* generations, so the CAS-swap that publishes the result
//! is safe to lose (INV-6). A crash mid-run leaves only unreferenced files, which GC later
//! reclaims.
//!
//! Note on determinism (G-6): the vector, pk and radj files are byte-identical across
//! replays. The tantivy FTS split is not — tantivy stamps each segment with a random id —
//! but its *retrieval results* are identical, so query behaviour is still reproducible.

use std::collections::BTreeMap;

use mlake_core::item::{SemanticEdge, Weight, MAX_SEMANTIC_OUT, SEMANTIC_LINK_THRESHOLD};
use mlake_core::{ItemId, StoredItem};
use mlake_fts::Tokenizer;
use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag};
use mlake_ivf::{build_clusters, train_centroids, ClusterFile};
use mlake_wal::{Namespace, WalTail};

use crate::generation::write_generation;
use crate::Result;

/// Options controlling a generation build.
#[derive(Clone, Copy)]
pub struct IndexOptions {
    /// Derive the semantic kNN link graph (SPEC §5.2). **On by default** — the graph arm
    /// is a first-class signal, so deriving its links is core behaviour, not opt-in.
    /// Derivation is incremental (only new items are linked, against the current corpus),
    /// so the steady-state cost is `O(new · N)`, not a full `O(N²)` rebuild. Disable only
    /// for a throughput benchmark that deliberately excludes graph work.
    pub derive_links: bool,
    /// Deterministic seed for centroid training (G-6).
    pub seed: u64,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            derive_links: true,
            seed: 42,
        }
    }
}

/// Result of an index run.
pub struct IndexOutcome {
    pub generation: u64,
    pub doc_count: usize,
    /// True if the manifest swap succeeded. False means another node published first — not
    /// an error, since its generation is equivalent to ours (INV-6).
    pub published: bool,
}

/// Build the next generation for a namespace and publish it.
pub async fn index(ns: &Namespace, tokenizer: &Tokenizer, opts: IndexOptions) -> Result<IndexOutcome> {
    let (manifest, etag) = ns.read_manifest().await?;

    // Incremental build: start from the previous generation's live items, then apply only
    // the WAL slice `(cursor, head]`. This is what makes WAL GC safe — once an entry is
    // folded into a generation, the generation carries it and the entry can be reclaimed
    // (SPEC §5). The first run has no previous generation and folds from an empty base.
    let head = ns.wal_head().await?;

    let mut by_id: std::collections::BTreeMap<[u8; 16], StoredItem> =
        std::collections::BTreeMap::new();
    if !manifest.is_empty() {
        let prev = crate::generation::read_generation(
            &ns.store,
            &manifest.files,
            manifest.generation,
            None,
        )
        .await?;
        for item in prev.clusters.into_iter().flatten() {
            by_id.insert(item.id.0, item);
        }
    }

    // Apply the tail: upserts replace, tombstones remove, patches were folded by the scan.
    let scan = WalTail::new(ns)
        .scan(manifest.wal_index_cursor, Some(head))
        .await?;
    for id in &scan.tombstones {
        by_id.remove(&id.0);
    }
    // Items introduced or replaced by this WAL slice are "new": their semantic links have
    // not been derived yet. Carried-over items keep the links they were indexed with —
    // derived data is expensive, so it is recomputed incrementally for new items, never
    // wholesale-deleted (which would silently empty the graph on every plain index run).
    let mut new_ids: std::collections::HashSet<[u8; 16]> = std::collections::HashSet::new();
    for (id, item) in scan.upserts {
        new_ids.insert(id.0);
        by_id.insert(id.0, item); // an upsert arrives with empty semantic_out
    }
    // Deferred patches apply to items carried from the previous generation.
    for item in by_id.values_mut() {
        if let Some(deltas) = scan.pending_patches.get(&ItemId(item.id.0)) {
            item.proof_count =
                mlake_core::wal::fold_proof_count(item.proof_count, deltas.iter().copied());
        }
    }

    // BTreeMap iteration is id-sorted, so the item order — and thus the build — is
    // deterministic and replayable (G-6).
    let mut items: Vec<StoredItem> = by_id.into_values().collect();

    // Drop carried links whose target was tombstoned in this slice, so radj never carries
    // an edge to something no longer present (retrieval would filter it, but keeping the
    // files clean is cheaper than relying on that downstream).
    let live: std::collections::HashSet<[u8; 16]> = items.iter().map(|i| i.id.0).collect();
    for item in items.iter_mut() {
        item.semantic_out.retain(|e| live.contains(&e.target.0));
    }

    if opts.derive_links {
        derive_semantic_links(&mut items, &new_ids);
    }

    let generation = manifest.generation + 1;
    let doc_count = items.len();

    // Train centroids and assign clusters.
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let centroids = train_centroids(&vectors, opts.seed);
    let clusters: Vec<ClusterFile> = build_clusters(items.clone(), &centroids);

    // Build the pk index as an SSTable: id → cluster, range-readable at scale.
    let mut pk_entries: Vec<(ItemId, u32)> = Vec::with_capacity(doc_count);
    for (ci, cluster) in clusters.iter().enumerate() {
        for item in &cluster.items {
            pk_entries.push((item.id, ci as u32));
        }
    }
    let pk_tables = crate::sstable::PkTable::build(pk_entries);

    // Build the tantivy FTS split, packed into one object for storage (SPEC §5.3).
    let fts = mlake_fts::TantivyFts::build(
        items.iter().map(|i| (i.id, i.text.as_str())),
        mlake_fts::Tokenizer::new(mlake_fts::TokenizerConfig::default()),
    )
    .map_err(|e| crate::Error::Core(mlake_core::Error::Encode(e.to_string())))?;
    let fts_split = fts.split_bytes().to_vec();

    // Build reverse adjacency as an SSTable from inline outgoing links.
    let mut radj_pairs: Vec<(ItemId, InEdge)> = Vec::new();
    for item in &items {
        for edge in &item.semantic_out {
            radj_pairs.push((
                edge.target,
                InEdge {
                    source: item.id,
                    kind: EdgeKind::Semantic,
                    weight: edge.weight.to_f32(),
                },
            ));
        }
        for edge in &item.causal_out {
            radj_pairs.push((
                edge.target,
                InEdge {
                    source: item.id,
                    kind: EdgeKind::Causal(LinkTypeTag::from(edge.link_type)),
                    weight: edge.weight.to_f32(),
                },
            ));
        }
    }
    let radj_tables = crate::sstable::RadjTable::build(radj_pairs);

    // Write the generation files under a prefix unique to this attempt, so concurrent
    // builders of the same generation number never overwrite each other's files (INV-2).
    let nonce = mlake_core::ItemId::new_v4().as_uuid().simple().to_string();
    let prefix = crate::generation::attempt_prefix(&ns.name, generation, &nonce);
    let files = write_generation(
        &ns.store,
        &prefix,
        &centroids,
        &clusters,
        &fts_split,
        radj_tables.into(),
        pk_tables.into(),
        doc_count,
    )
    .await?;

    let mut next = manifest.clone();
    next.generation = generation;
    next.wal_index_cursor = head;
    next.wal_head = head;
    // Carry the outgoing generation's files and cursor forward as the GC grace window: a
    // reader still holding this manifest keeps working until the next generation ages out.
    next.prev_files = if manifest.is_empty() {
        None
    } else {
        Some(manifest.files.clone())
    };
    next.prev_wal_index_cursor = manifest.wal_index_cursor;
    next.files = files;
    next.tokenizer_config_hash = tokenizer.config_hash();
    next.prev_generation = Some(manifest.generation);

    let published = match etag {
        Some(etag) => ns.swap_manifest(&etag, &next).await.map(|_| true).or_else(|e| {
            if e.is_conflict() {
                // Another node published first. Its generation is equivalent; our files
                // become unreferenced garbage for GC. Not an error (INV-6).
                Ok(false)
            } else {
                Err(e)
            }
        })?,
        None => false,
    };

    Ok(IndexOutcome {
        generation,
        doc_count,
        published,
    })
}

/// Derive top-5 semantic kNN links (cosine ≥ 0.7) for the *new* items only, over the full
/// current corpus, as SPEC §5.2 specifies ("for each new item in the WAL slice").
///
/// Carried-over items keep the links they were indexed with — this is incremental
/// derivation (`O(new · N)`), not a full `O(N²)` rebuild. Deterministic (ties broken by
/// id) for G-6. The production path queries the warm IVF index for neighbours instead of
/// scanning; that is the O(N)-per-item optimization, not a behavioural change.
fn derive_semantic_links(items: &mut [StoredItem], new_ids: &std::collections::HashSet<[u8; 16]>) {
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let ids: Vec<ItemId> = items.iter().map(|i| i.id).collect();

    for (i, item) in items.iter_mut().enumerate() {
        if !new_ids.contains(&item.id.0) {
            continue; // carried item: its links are already derived and preserved
        }
        let mut scored: BTreeMap<[u8; 16], f32> = BTreeMap::new();
        for (j, v) in vectors.iter().enumerate() {
            if i == j {
                continue;
            }
            let sim = mlake_core::cosine(&vectors[i], v);
            if sim >= SEMANTIC_LINK_THRESHOLD {
                scored.insert(ids[j].0, sim);
            }
        }
        let mut neighbours: Vec<([u8; 16], f32)> = scored.into_iter().collect();
        neighbours.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        neighbours.truncate(MAX_SEMANTIC_OUT);
        item.semantic_out = neighbours
            .into_iter()
            .map(|(id, sim)| SemanticEdge {
                target: ItemId(id),
                weight: Weight::from_f32(sim),
            })
            .collect();
    }
}
