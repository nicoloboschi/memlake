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

use mlake_core::memory::{SemanticEdge, Weight, MAX_SEMANTIC_OUT, SEMANTIC_LINK_THRESHOLD};
use mlake_core::{MemoryId, StoredMemory};
use mlake_fts::Tokenizer;
use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag};
use mlake_ivf::{train_centroids, ClusterFile};
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
    /// Force a full centroid retrain this fold, regardless of the 2× growth trigger. Used
    /// by the recall-vs-churn study to compare assign-only drift against a fresh retrain.
    pub force_retrain: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            derive_links: true,
            seed: 42,
            force_retrain: false,
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
/// Fold the bank's WAL into a new generation for each fact type and publish one manifest.
///
/// Fact types are fully independent indexes (no shared links/vectors/postings), so the fold
/// partitions the live item set by `memory_type` and builds a separate generation per type,
/// each with its own assign-only/copy-forward state (SCALE.md Phase 4). One WAL, one
/// manifest — so a `bank + [memory_types]` query reads a single manifest and fans out.
pub async fn index(ns: &Namespace, tokenizer: &Tokenizer, opts: IndexOptions) -> Result<IndexOutcome> {
    let (manifest, etag) = ns.read_manifest().await?;
    let head = ns.wal_head().await?;

    // Load each existing fact type's previous items (for the fold + copy-forward).
    let mut prev_by_ft: std::collections::BTreeMap<u8, mlake_index_prev::Generation> =
        std::collections::BTreeMap::new();
    for ft in manifest.memory_types() {
        let fti = manifest.index(ft).unwrap();
        let gen =
            crate::generation::read_generation(&ns.store, &fti.files, manifest.generation, None)
                .await?;
        prev_by_ft.insert(ft, gen);
    }

    // Fold the whole live item set (across all fact types), applying the tail.
    let mut by_id: std::collections::BTreeMap<[u8; 16], StoredMemory> =
        std::collections::BTreeMap::new();
    for gen in prev_by_ft.values() {
        for cluster in &gen.clusters {
            for item in cluster {
                by_id.insert(item.id.0, item.clone());
            }
        }
    }

    let scan = WalTail::new(ns)
        .scan(manifest.wal_index_cursor, Some(head))
        .await?;
    for id in &scan.tombstones {
        by_id.remove(&id.0);
    }
    let mut new_ids: std::collections::HashSet<[u8; 16]> = std::collections::HashSet::new();
    for (id, item) in scan.upserts {
        new_ids.insert(id.0);
        by_id.insert(id.0, item);
    }
    let mut touched: std::collections::HashSet<[u8; 16]> = new_ids.clone();
    for item in by_id.values_mut() {
        if let Some(deltas) = scan.pending_patches.get(&MemoryId(item.id.0)) {
            mlake_core::wal::apply_deltas(item, deltas);
            touched.insert(item.id.0);
        }
    }

    let mut items: Vec<StoredMemory> = by_id.into_values().collect();
    let live: std::collections::HashSet<[u8; 16]> = items.iter().map(|i| i.id.0).collect();
    for item in items.iter_mut() {
        item.semantic_out.retain(|e| live.contains(&e.target.0));
    }

    let generation = manifest.generation + 1;
    let doc_count = items.len();

    // Partition the live items by fact type (BTreeMap keeps a deterministic type order).
    let mut items_by_ft: std::collections::BTreeMap<u8, Vec<StoredMemory>> =
        std::collections::BTreeMap::new();
    for item in items {
        items_by_ft.entry(item.memory_type).or_default().push(item);
    }

    // Build an independent generation per fact type that still has items.
    let nonce = mlake_core::MemoryId::new_v4().as_uuid().simple().to_string();
    let mut indexes: std::collections::BTreeMap<u8, mlake_core::manifest::FactTypeIndex> =
        std::collections::BTreeMap::new();
    for (ft, ft_items) in items_by_ft {
        let fti = build_memory_type_index(
            ns,
            generation,
            &nonce,
            ft,
            ft_items,
            &new_ids,
            &touched,
            prev_by_ft.get(&ft),
            manifest.index(ft),
            opts,
        )
        .await?;
        indexes.insert(ft, fti);
    }

    let mut next = manifest.clone();
    next.generation = generation;
    next.wal_index_cursor = head;
    next.wal_head = head;
    next.prev_wal_index_cursor = manifest.wal_index_cursor;
    next.prev_generation = Some(manifest.generation);
    next.tokenizer_config_hash = tokenizer.config_hash();
    next.indexes = indexes;

    let published = match etag {
        Some(etag) => ns.swap_manifest(&etag, &next).await.map(|_| true).or_else(|e| {
            if e.is_conflict() {
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

// Alias so the module can name Generation without importing it at the top (it is only used
// in this function's type annotation).
mod mlake_index_prev {
    pub use crate::generation::Generation;
}

/// Build one fact type's independent generation (assign-only + copy-forward + local split +
/// IVF link derivation) from its slice of the live items, and return its manifest entry.
#[allow(clippy::too_many_arguments)]
async fn build_memory_type_index(
    ns: &Namespace,
    generation: u64,
    nonce: &str,
    memory_type: u8,
    mut items: Vec<StoredMemory>,
    new_ids: &std::collections::HashSet<[u8; 16]>,
    touched: &std::collections::HashSet<[u8; 16]>,
    prev_gen: Option<&crate::generation::Generation>,
    prev_index: Option<&mlake_core::manifest::FactTypeIndex>,
    opts: IndexOptions,
) -> Result<mlake_core::manifest::FactTypeIndex> {
    let doc_count = items.len();
    let prev_train_count = prev_index.map(|p| p.train_count).unwrap_or(0);

    // Assign-only vs retrain, scoped to this fact type.
    let retrain =
        prev_gen.is_none() || opts.force_retrain || doc_count as u64 > 2 * prev_train_count.max(1);
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    // One dimension per fact type, checked once here rather than deep inside the parallel
    // link derivation: everything downstream (centroid training, probing, semantic-edge
    // cosine) assumes it. Failing the fold with a typed error keeps a corpus that somehow
    // mixed dimensions from becoming an unexplained panic in a rayon worker.
    mlake_core::uniform_dim(vectors.iter().map(|v| v.as_slice()))?;
    let tt = std::time::Instant::now();
    let (mut centroids, train_count) = match (prev_gen, retrain) {
        (Some(p), false) => (p.centroids.clone(), prev_train_count),
        _ => (train_centroids(&vectors, opts.seed), doc_count as u64),
    };
    phase_log("train", tt);

    let mut assignments: Vec<usize> = if centroids.is_empty() {
        vec![0; items.len()]
    } else {
        items.iter().map(|i| centroids.assign(&i.vector)).collect()
    };

    let split_clusters = local_split(&mut centroids, &items, &mut assignments, opts.seed);

    let td = std::time::Instant::now();
    if opts.derive_links && !centroids.is_empty() {
        derive_links_ivf(&mut items, new_ids, &centroids, &assignments);
    } else if opts.derive_links {
        derive_semantic_links(&mut items, new_ids);
    }
    phase_log("derive_links", td);

    let k = centroids.len().max(1);
    let mut clusters: Vec<Vec<StoredMemory>> = vec![Vec::new(); k];
    for (item, &c) in items.iter().zip(assignments.iter()) {
        clusters[c].push(item.clone());
    }

    // Per-fact-type prefix so different types never collide on object keys.
    let prefix = format!("{}/mt{memory_type}/gen-{generation}-{nonce}", ns.name);

    // Copy-forward unchanged clusters (skip their PUT); rewrite only dirty ones.
    let empty: Vec<Vec<StoredMemory>> = Vec::new();
    let old_clusters = prev_gen.map(|p| &p.clusters).unwrap_or(&empty);
    let empty_paths: Vec<String> = Vec::new();
    let old_paths = prev_index.map(|p| &p.files.clusters).unwrap_or(&empty_paths);
    let tw = std::time::Instant::now();
    let mut cluster_paths: Vec<Option<String>> = vec![None; k];
    let mut dirty_writes = Vec::new();
    for i in 0..k {
        let can_copy_forward = !retrain
            && !split_clusters.contains(&i)
            && i < old_clusters.len()
            && i < old_paths.len()
            && !cluster_changed(&clusters[i], &old_clusters[i], touched);
        if can_copy_forward {
            cluster_paths[i] = Some(old_paths[i].clone());
        } else {
            let cf = ClusterFile { centroid_id: i as u32, items: clusters[i].clone() };
            let store = ns.store.clone();
            let prefix = prefix.clone();
            dirty_writes.push(async move {
                crate::generation::write_cluster_file(&store, &prefix, i, &cf)
                    .await
                    .map(|path| (i, path))
            });
        }
    }
    // Write the dirty clusters with bounded concurrency instead of one sequential PUT at a
    // time — the dominant cost of a full build over S3 (SCALE.md Phase 3 perf).
    {
        use futures::stream::{StreamExt, TryStreamExt};
        let written: Vec<(usize, String)> = futures::stream::iter(dirty_writes)
            .buffer_unordered(32)
            .try_collect()
            .await?;
        for (i, path) in written {
            cluster_paths[i] = Some(path);
        }
    }
    let cluster_paths: Vec<String> = cluster_paths.into_iter().map(|p| p.unwrap_or_default()).collect();
    phase_log("cluster_write", tw);

    // pk / radj / fts, scoped to this fact type.
    let mut pk_entries: Vec<(MemoryId, u32)> = Vec::with_capacity(doc_count);
    for (ci, cluster) in clusters.iter().enumerate() {
        for item in cluster {
            pk_entries.push((item.id, ci as u32));
        }
    }
    let pk_tables = crate::sstable::PkTable::build(pk_entries);

    let tfts = std::time::Instant::now();
    let fts = mlake_fts::TantivyFts::build_with_tags(
        items.iter().map(|i| (i.id, i.text.as_str(), i.tags.as_slice())),
        mlake_fts::Tokenizer::new(mlake_fts::TokenizerConfig::default()),
    )
    .map_err(|e| crate::Error::Core(mlake_core::Error::Encode(e.to_string())))?;
    let fts_split = fts.split_bytes().to_vec();
    phase_log("fts_build", tfts);

    let mut radj_pairs: Vec<(MemoryId, InEdge)> = Vec::new();
    for item in &items {
        for edge in &item.semantic_out {
            radj_pairs.push((
                edge.target,
                InEdge { source: item.id, kind: EdgeKind::Semantic, weight: edge.weight.to_f32() },
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

    // Entity postings: EntityId -> the memories that carry it. This is what lets the graph
    // arm's entity expansion find sharers anywhere in the corpus (not just probed clusters).
    let mut entity_pairs: Vec<(mlake_core::EntityId, MemoryId)> = Vec::new();
    for item in &items {
        for e in &item.entity_ids {
            entity_pairs.push((*e, item.id));
        }
    }
    let entity_tables = crate::sstable::EntityTable::build(entity_pairs);

    // Time index: effective_ts = COALESCE(occurred_start, mentioned_at, occurred_end) -> id.
    // Entry-point selection for the temporal arm is one ranged scan of this key range.
    let mut time_pairs: Vec<(i64, MemoryId)> = Vec::new();
    for item in &items {
        let t = &item.timestamps;
        if let Some(ts) = t.occurred_start.or(t.mentioned_at).or(t.occurred_end) {
            time_pairs.push((ts, item.id));
        }
    }
    let time_tables = crate::sstable::TimeTable::build(time_pairs);

    // Per-cluster tag summaries: the union of each cluster's tags + an untagged flag, so a
    // query can prune clusters that cannot contain a matching memory (SCALE.md Phase 4b).
    let tag_summary: crate::generation::TagSummary = clusters
        .iter()
        .map(|cluster| {
            let mut tags: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            let mut has_untagged = false;
            for m in cluster {
                if m.tags.is_empty() {
                    has_untagged = true;
                } else {
                    tags.extend(m.tags.iter().cloned());
                }
            }
            crate::generation::ClusterTagSummary {
                tags: tags.into_iter().collect(),
                has_untagged,
            }
        })
        .collect();

    let twg = std::time::Instant::now();
    let files = write_generation(
        &ns.store,
        &prefix,
        &centroids,
        cluster_paths,
        &fts_split,
        radj_tables.into(),
        pk_tables.into(),
        entity_tables.into(),
        time_tables.into(),
        &tag_summary,
        doc_count,
    )
    .await?;
    phase_log("write_meta", twg);

    Ok(mlake_core::manifest::FactTypeIndex {
        prev_files: prev_index.map(|p| p.files.clone()),
        train_count,
        files,
    })
}

/// Log an index phase's duration to stderr when `MEMLAKE_TIMING` is set. Cheap enough to
/// leave in — a couple of `Instant`s per fold.
fn phase_log(phase: &str, start: std::time::Instant) {
    if std::env::var("MEMLAKE_TIMING").is_ok() {
        eprintln!("[index] {phase}: {:.2}s", start.elapsed().as_secs_f64());
    }
}

/// True if a cluster's membership or content changed versus the previous generation. Both
/// slices are id-sorted, so a position-wise compare suffices; a member touched by this
/// WAL slice (re-upserted or patched) also counts as changed.
fn cluster_changed(
    new: &[StoredMemory],
    old: &[StoredMemory],
    touched: &std::collections::HashSet<[u8; 16]>,
) -> bool {
    if new.len() != old.len() {
        return true;
    }
    for (n, o) in new.iter().zip(old.iter()) {
        if n.id != o.id || touched.contains(&n.id.0) {
            return true;
        }
    }
    false
}

/// Split any cluster grown past 8× the average size, in place: a 2-means over just that
/// cluster's members yields two sub-centroids (one replaces the original, one is appended),
/// and only that cluster's members are reassigned between them (SPFresh-lite core, LIRE).
/// Returns the set of centroid indices that changed, which the caller marks dirty.
fn local_split(
    centroids: &mut mlake_ivf::Centroids,
    items: &[StoredMemory],
    assignments: &mut [usize],
    seed: u64,
) -> std::collections::HashSet<usize> {
    let mut split = std::collections::HashSet::new();
    let k = centroids.len();
    if k == 0 || items.is_empty() {
        return split;
    }
    let avg = items.len() as f32 / k as f32;
    let threshold = (8.0 * avg).ceil() as usize;

    let mut counts = vec![0usize; k];
    for &c in assignments.iter() {
        counts[c] += 1;
    }

    for i in 0..k {
        if counts[i] <= threshold || counts[i] < 2 {
            continue;
        }
        let members: Vec<usize> = (0..items.len()).filter(|&j| assignments[j] == i).collect();
        let vecs: Vec<Vec<f32>> = members.iter().map(|&j| items[j].vector.clone()).collect();
        let sub = mlake_ivf::kmeans::train(&vecs, 2, 10, seed);
        if sub.len() < 2 {
            continue;
        }
        // Replace centroid i with the first sub-centroid; append the second.
        let new_idx = centroids.push(sub[1].clone());
        centroids.vectors[i] = sub[0].clone();
        for &j in &members {
            let d0 = mlake_ivf::kmeans::sq_dist_pub(&items[j].vector, &sub[0]);
            let d1 = mlake_ivf::kmeans::sq_dist_pub(&items[j].vector, &sub[1]);
            assignments[j] = if d1 < d0 { new_idx } else { i };
        }
        split.insert(i);
        split.insert(new_idx);
    }
    split
}

/// Incremental link derivation via the IVF (SCALE.md Phase 3): each new item probes the
/// centroids and scans only the probed clusters for its top-5 neighbours, so the cost is
/// `O(new · nprobe · cluster_size)` instead of the `O(new · N)` full scan. Deterministic
/// (ties by id) for G-6.
fn derive_links_ivf(
    items: &mut [StoredMemory],
    new_ids: &std::collections::HashSet<[u8; 16]>,
    centroids: &mlake_ivf::Centroids,
    assignments: &[usize],
) {
    const DERIVE_NPROBE: usize = 16;
    // cluster -> member item indices.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); centroids.len().max(1)];
    for (j, &c) in assignments.iter().enumerate() {
        members[c].push(j);
    }
    // Snapshot vectors/ids so we can read while mutating semantic_out.
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let ids: Vec<MemoryId> = items.iter().map(|i| i.id).collect();

    // Each new item's neighbour derivation is independent (reads the shared vectors/ids
    // snapshots, writes only its own semantic_out), so derive them across all cores. This
    // is the dominant cost of a first-time index build (16 probes × cluster members ×
    // cosine, per new item), so parallelizing it is what keeps large ingests tractable.
    use rayon::prelude::*;
    let derived: Vec<(usize, Vec<SemanticEdge>)> = (0..items.len())
        .into_par_iter()
        .filter(|&j| new_ids.contains(&items[j].id.0))
        .map(|j| {
            let probed = centroids.probe(&vectors[j], DERIVE_NPROBE);
            let mut scored: BTreeMap<[u8; 16], f32> = BTreeMap::new();
            for c in probed {
                for &m in &members[c] {
                    if m == j {
                        continue;
                    }
                    let sim = mlake_core::cosine_opt(&vectors[j], &vectors[m]);
                    if sim >= SEMANTIC_LINK_THRESHOLD {
                        scored.insert(ids[m].0, sim);
                    }
                }
            }
            let mut neighbours: Vec<([u8; 16], f32)> = scored.into_iter().collect();
            neighbours.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
            });
            neighbours.truncate(MAX_SEMANTIC_OUT);
            let edges = neighbours
                .into_iter()
                .map(|(id, sim)| SemanticEdge { target: MemoryId(id), weight: Weight::from_f32(sim) })
                .collect();
            (j, edges)
        })
        .collect();
    for (j, edges) in derived {
        items[j].semantic_out = edges;
    }
}

/// Derive top-5 semantic kNN links (cosine ≥ 0.7) for the *new* items only, over the full
/// current corpus, as SPEC §5.2 specifies ("for each new item in the WAL slice").
///
/// Carried-over items keep the links they were indexed with — this is incremental
/// derivation (`O(new · N)`), not a full `O(N²)` rebuild. Deterministic (ties broken by
/// id) for G-6. The production path queries the warm IVF index for neighbours instead of
/// scanning; that is the O(N)-per-item optimization, not a behavioural change.
fn derive_semantic_links(items: &mut [StoredMemory], new_ids: &std::collections::HashSet<[u8; 16]>) {
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let ids: Vec<MemoryId> = items.iter().map(|i| i.id).collect();

    use rayon::prelude::*;
    let derived: Vec<(usize, Vec<SemanticEdge>)> = (0..items.len())
        .into_par_iter()
        .filter(|&i| new_ids.contains(&items[i].id.0))
        .map(|i| {
            let mut scored: BTreeMap<[u8; 16], f32> = BTreeMap::new();
            for (j, v) in vectors.iter().enumerate() {
                if i == j {
                    continue;
                }
                let sim = mlake_core::cosine_opt(&vectors[i], v);
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
            let edges = neighbours
                .into_iter()
                .map(|(id, sim)| SemanticEdge {
                    target: MemoryId(id),
                    weight: Weight::from_f32(sim),
                })
                .collect();
            (i, edges)
        })
        .collect();
    for (i, edges) in derived {
        items[i].semantic_out = edges;
    }
}
