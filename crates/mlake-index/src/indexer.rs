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
use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag, ReverseAdjacency};
use mlake_ivf::{build_clusters, train_centroids, ClusterFile};
use mlake_wal::{Namespace, WalTail};

use crate::generation::{write_generation, PkIndex};
use crate::Result;

/// Options controlling a generation build.
#[derive(Clone, Copy)]
pub struct IndexOptions {
    /// Derive the semantic kNN link graph (SPEC §5.2). Off by default because it is
    /// O(N²) brute force here; the query-quality demonstration turns it on.
    pub derive_links: bool,
    /// Deterministic seed for centroid training (G-6).
    pub seed: u64,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            derive_links: false,
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
    for (id, item) in scan.upserts {
        by_id.insert(id.0, item);
    }
    // Deferred patches apply to items carried from the previous generation.
    for item in by_id.values_mut() {
        if let Some(deltas) = scan.pending_patches
            .get(&ItemId(item.id.0)) { item.proof_count =
                    mlake_core::wal::fold_proof_count(item.proof_count, deltas.iter().copied()); }
    }

    // BTreeMap iteration is id-sorted, so the item order — and thus the build — is
    // deterministic and replayable (G-6).
    let mut items: Vec<StoredItem> = by_id.into_values().collect();
    // semantic_out is derived, not carried: recompute it if requested, else clear stale
    // links from the previous generation so they cannot dangle.
    for item in items.iter_mut() {
        item.semantic_out.clear();
    }

    if opts.derive_links {
        derive_semantic_links(&mut items);
    }

    let generation = manifest.generation + 1;
    let doc_count = items.len();

    // Train centroids and assign clusters.
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let centroids = train_centroids(&vectors, opts.seed);
    let clusters: Vec<ClusterFile> = build_clusters(items.clone(), &centroids);

    // Build the pk index: id → cluster, sorted for binary search.
    let mut pk_entries: Vec<(ItemId, u32)> = Vec::with_capacity(doc_count);
    for (ci, cluster) in clusters.iter().enumerate() {
        for item in &cluster.items {
            pk_entries.push((item.id, ci as u32));
        }
    }
    pk_entries.sort_by_key(|a| a.0);
    let pk = PkIndex { entries: pk_entries };

    // Build the tantivy FTS split, packed into one object for storage (SPEC §5.3).
    let fts = mlake_fts::TantivyFts::build(
        items.iter().map(|i| (i.id, i.text.as_str())),
        mlake_fts::Tokenizer::new(mlake_fts::TokenizerConfig::default()),
    )
    .map_err(|e| crate::Error::Core(mlake_core::Error::Encode(e.to_string())))?;
    let fts_split = fts.split_bytes().to_vec();

    // Build reverse adjacency from inline outgoing links.
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
    let radj = ReverseAdjacency::build(radj_pairs);

    // Write the generation files, then publish by CAS-swapping the manifest.
    let files = write_generation(
        &ns.store, &ns.name, generation, &centroids, &clusters, &fts_split, &radj, &pk,
    )
    .await?;

    let mut next = manifest.clone();
    next.generation = generation;
    next.wal_index_cursor = head;
    next.wal_head = head;
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

/// Derive each item's top-5 semantic kNN links (cosine ≥ 0.7), as SPEC §5.2 specifies.
///
/// Brute force O(N²): acceptable at the prototype's scale; the production path queries the
/// warm IVF index for neighbours instead. Deterministic (ties broken by id) for G-6.
fn derive_semantic_links(items: &mut [StoredItem]) {
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let ids: Vec<ItemId> = items.iter().map(|i| i.id).collect();

    for (i, item) in items.iter_mut().enumerate() {
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
