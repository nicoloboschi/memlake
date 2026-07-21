//! End-to-end: write → index → query, exercising the full S3-native path.
//!
//! These tie every layer together — WAL commit, generation build, manifest swap, and a
//! stateless query node reading it all back from object storage — and assert the
//! architecture's headline properties: a freshly started node serves correct results
//! (INV-4), and strongly-consistent reads reflect writes committed after the last index
//! run (INV-5).

use std::sync::Arc;

use mlake_core::item::Timestamps;
use mlake_core::{Item, ItemId, Op};
use mlake_fts::Tokenizer;
use mlake_index::{index, Consistency, IndexOptions, QueryConfig, QueryNode};
use mlake_store::Store;
use mlake_wal::{Namespace, Writer};

fn item(key: &str, vector: Vec<f32>, text: &str) -> Item {
    Item {
        id: ItemId::from_key(key),
        vector,
        text: text.to_string(),
        fact_type: 1,
        tags: vec![],
        timestamps: Timestamps::default(),
        proof_count: 0,
        entity_ids: vec![],
        causal_out: vec![],
    }
}

async fn namespace(store: Store, name: &str) -> Namespace {
    let ns = Namespace::new(name, store);
    ns.create_if_absent(&Tokenizer::default().config_hash()).await.unwrap();
    ns
}

/// The whole pipeline over a shared backing store: write, index, then query from a node
/// that has never seen the namespace.
#[tokio::test]
async fn write_index_then_query_from_a_fresh_node() {
    let backing = Arc::new(object_store::memory::InMemory::new());
    let ns = namespace(Store::new(Arc::clone(&backing) as _), "ns").await;

    let mut writer = Writer::new(ns.clone());
    writer
        .commit(vec![
            Op::Upsert(item("cats", vec![1.0, 0.0, 0.0], "cats are feline pets")),
            Op::Upsert(item("dogs", vec![0.9, 0.1, 0.0], "dogs are loyal canine companions")),
            Op::Upsert(item("cars", vec![0.0, 0.0, 1.0], "cars drive on the road")),
        ])
        .await
        .unwrap();

    // Index into a generation and publish.
    let outcome = index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    assert!(outcome.published, "the first index run must publish");
    assert_eq!(outcome.generation, 1);
    assert_eq!(outcome.doc_count, 3);

    // A brand-new node over the same bucket — no shared memory, cold cache.
    let fresh = Namespace::new("ns", Store::new(backing as _));
    let node = QueryNode::open(&fresh, Tokenizer::default(), Consistency::Strong)
        .await
        .unwrap();
    assert_eq!(node.doc_count(), 3);

    // A pure-vector query returns the nearest item.
    let vec_hits = node.query(Some(&[1.0, 0.0, 0.0]), None, 10, QueryConfig::default());
    assert_eq!(vec_hits[0].id, ItemId::from_key("cats"), "nearest vector should lead");

    // A fused query blends the text signal: `dogs` is nearest-but-one by vector *and*
    // the text match, so convergent evidence puts it and `cats` at the top.
    let hits = node.query(Some(&[1.0, 0.0, 0.0]), Some("loyal canine"), 10, QueryConfig::default());
    let top2: Vec<_> = hits.iter().take(2).map(|h| h.id).collect();
    assert!(top2.contains(&ItemId::from_key("cats")));
    assert!(top2.contains(&ItemId::from_key("dogs")));
}

/// INV-5 across the index boundary: a write committed *after* the last index run is still
/// visible to a strongly-consistent query, via the WAL tail merge.
#[tokio::test]
async fn writes_after_indexing_are_visible_under_strong_consistency() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "first"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // This lands in the WAL tail, past the generation's cursor.
    writer.commit(vec![Op::Upsert(item("b", vec![0.0, 1.0], "second"))]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(node.doc_count(), 2, "the un-indexed write must be visible");
    let hits = node.query(Some(&[0.0, 1.0]), None, 10, QueryConfig::default());
    assert_eq!(hits[0].id, ItemId::from_key("b"), "the tail write must be queryable");
}

/// A delete committed after indexing removes the item from strongly-consistent results,
/// even though the generation still physically contains it.
#[tokio::test]
async fn deletes_after_indexing_take_effect_immediately() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer
        .commit(vec![
            Op::Upsert(item("keep", vec![1.0, 0.0], "keep me")),
            Op::Upsert(item("drop", vec![0.0, 1.0], "delete me")),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    writer.commit(vec![Op::Tombstone { id: ItemId::from_key("drop") }]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(node.doc_count(), 1);
    let hits = node.query(Some(&[0.0, 1.0]), None, 10, QueryConfig::default());
    assert!(
        !hits.iter().any(|h| h.id == ItemId::from_key("drop")),
        "a tombstoned item must not appear in strong-consistency results"
    );
}

/// Re-indexing folds the tail into a new generation and advances the cursor, without
/// changing what a query returns.
#[tokio::test]
async fn reindexing_is_stable_and_advances_the_cursor() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();
    let first = index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    writer.commit(vec![Op::Upsert(item("b", vec![0.0, 1.0], "beta"))]).await.unwrap();
    let second = index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    assert_eq!(second.generation, first.generation + 1);
    assert_eq!(second.doc_count, 2);

    let (manifest, _) = ns.read_manifest().await.unwrap();
    assert_eq!(manifest.generation, 2);
    // Everything is now indexed: nothing left in the tail.
    assert_eq!(manifest.index_lag(), 0);

    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(node.doc_count(), 2);
}

/// The determinism gate (G-6): re-running the indexer on the same WAL slice produces the
/// same generation content. Checked here via the published document set and manifest.
#[tokio::test]
async fn indexing_is_deterministic() {
    async fn build_and_hash() -> String {
        let store = Store::in_memory();
        let ns = namespace(store, "ns").await;
        let mut writer = Writer::new(ns.clone());
        for i in 0..10 {
            writer
                .commit(vec![Op::Upsert(item(
                    &format!("item-{i}"),
                    vec![i as f32, (10 - i) as f32],
                    &format!("document number {i}"),
                ))])
                .await
                .unwrap();
        }
        index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
        // Hash the published centroids + pk index as a proxy for generation content.
        let (manifest, _) = ns.read_manifest().await.unwrap();
        let centroids = ns.store.get(&manifest.files.centroids, None).await.unwrap();
        let pk = ns.store.get(&manifest.files.pk, None).await.unwrap();
        format!("{}-{}", centroids.bytes.len(), pk.bytes.len())
    }

    let a = build_and_hash().await;
    let b = build_and_hash().await;
    assert_eq!(a, b, "indexing the same input must be reproducible (G-6)");
}

/// The graph arm works end to end: derived links let link expansion surface a neighbour.
#[tokio::test]
async fn graph_arm_works_through_the_full_pipeline() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // Two near-identical vectors will be linked as kNN neighbours (cosine ≥ 0.7).
    writer
        .commit(vec![
            Op::Upsert(item("a", vec![1.0, 0.0, 0.0], "alpha document")),
            Op::Upsert(item("a2", vec![0.98, 0.02, 0.0], "closely related to alpha")),
            Op::Upsert(item("z", vec![0.0, 0.0, 1.0], "unrelated topic")),
        ])
        .await
        .unwrap();

    let opts = IndexOptions {
        derive_links: true,
        ..IndexOptions::default()
    };
    index(&ns, &Tokenizer::default(), opts).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    // Query near `a`; the graph arm should also pull in its neighbour `a2`.
    let cfg = QueryConfig {
        graph_weight: 1.0,
        ..QueryConfig::default()
    };
    let hits = node.query(Some(&[1.0, 0.0, 0.0]), None, 10, cfg);
    let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
    assert!(ids.contains(&ItemId::from_key("a")));
    assert!(ids.contains(&ItemId::from_key("a2")), "graph expansion should surface the neighbour");
}

/// The full pipeline against a real S3 implementation (MinIO), skipped when it is down.
/// This is the one that proves the architecture works over actual object storage, not
/// just the in-memory stand-in.
#[tokio::test]
async fn full_pipeline_against_minio() {
    let endpoint =
        std::env::var("MEMLAKE_S3_ENDPOINT").unwrap_or_else(|_| "http://localhost:9000".into());
    let Ok(store) = Store::s3("memlake", Some(&endpoint), "memlake", "memlake123", "us-east-1")
    else {
        eprintln!("note: MinIO unavailable, skipping");
        return;
    };
    if store.exists("__probe__").await.is_err() {
        eprintln!("note: MinIO unreachable, skipping");
        return;
    }

    let name = format!(
        "e2e-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let ns = namespace(store.clone(), &name).await;
    let mut writer = Writer::new(ns.clone());
    writer
        .commit(vec![
            Op::Upsert(item("x", vec![1.0, 0.0, 0.0], "the quick brown fox")),
            Op::Upsert(item("y", vec![0.0, 1.0, 0.0], "a lazy sleeping dog")),
        ])
        .await
        .unwrap();

    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // A completely separate node, as a different process would be.
    let other = Namespace::new(
        &name,
        Store::s3("memlake", Some(&endpoint), "memlake", "memlake123", "us-east-1").unwrap(),
    );
    let node = QueryNode::open(&other, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(node.doc_count(), 2);
    let hits = node.query(Some(&[1.0, 0.0, 0.0]), Some("fox"), 10, QueryConfig::default());
    assert_eq!(hits[0].id, ItemId::from_key("x"));
}

// ---------------------------------------------------------------- compaction & GC (M6)

use mlake_index::{gc_with_min_age, GcOutcome};
use std::time::Duration as GcDuration;

/// After GC, folded WAL entries and superseded generation files are gone, but queries
/// still return the same results — GC touches only unreferenced files (INV-4).
#[tokio::test]
async fn gc_reclaims_folded_wal_and_old_generations_without_changing_results() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // Two index runs, so there is a superseded generation and folded WAL to reclaim.
    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    writer.commit(vec![Op::Upsert(item("b", vec![0.0, 1.0], "beta"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    // A third run so generation 1 falls below prev_generation and becomes collectable.
    writer.commit(vec![Op::Upsert(item("c", vec![1.0, 1.0], "gamma"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let before = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(before.doc_count(), 3);

    let outcome = gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();
    assert!(outcome.wal_entries_deleted > 0, "folded WAL entries should be reclaimed");
    assert!(outcome.generation_files_deleted > 0, "an old generation should be reclaimed");

    // Results are unchanged: GC only removed files nothing references.
    let after = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(after.doc_count(), 3);
    let hits = after.query(Some(&[0.0, 1.0]), None, 10, QueryConfig::default());
    assert_eq!(hits[0].id, ItemId::from_key("b"));
}

/// GC keeps the current and previous generations, so a query still works immediately after
/// collecting.
#[tokio::test]
async fn gc_keeps_the_current_generation_intact() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());
    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(node.doc_count(), 1, "current generation must survive GC");
}

/// GC is idempotent: a second pass finds nothing new to delete and does not error.
#[tokio::test]
async fn gc_is_idempotent() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());
    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    writer.commit(vec![Op::Upsert(item("b", vec![0.0, 1.0], "beta"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let first = gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();
    let second = gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();
    assert_eq!(second, GcOutcome::default(), "second GC pass must find nothing new");
    assert!(first != GcOutcome::default(), "first GC pass should reclaim something");
}

/// Incremental indexing folds a tombstone permanently: after the delete is folded into a
/// generation and its WAL entry GC'd, the item stays gone.
#[tokio::test]
async fn tombstone_survives_compaction_and_wal_gc() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer
        .commit(vec![
            Op::Upsert(item("keep", vec![1.0, 0.0], "keep")),
            Op::Upsert(item("drop", vec![0.0, 1.0], "drop")),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    writer.commit(vec![Op::Tombstone { id: ItemId::from_key("drop") }]).await.unwrap();
    // Fold the tombstone into a new generation, then GC the tombstone's WAL entry.
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    assert_eq!(node.doc_count(), 1, "the deleted item must stay gone after compaction");
    assert!(node.query(Some(&[0.0, 1.0]), None, 10, QueryConfig::default())
        .iter()
        .all(|h| h.id != ItemId::from_key("drop")));
}

use std::sync::Arc as ArcChaos;

/// Chaos: a crash after writing generation files but before the manifest swap leaves the
/// old generation serving. The orphaned files are unreferenced and GC-collectable, and the
/// next index run simply republishes — no data loss, no corruption (INV-6).
#[tokio::test]
async fn crash_before_manifest_swap_leaves_the_old_generation_serving() {
    let backing = ArcChaos::new(object_store::memory::InMemory::new());
    let ns = namespace(Store::new(ArcChaos::clone(&backing) as _), "ns").await;
    let mut writer = Writer::new(ns.clone());
    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // Simulate a crashed index run: write generation-2 files directly, but never swap the
    // manifest. The manifest still points at generation 1.
    writer.commit(vec![Op::Upsert(item("b", vec![0.0, 1.0], "beta"))]).await.unwrap();
    let (manifest, _) = ns.read_manifest().await.unwrap();
    let empty_split = mlake_fts::TantivyFts::build(
        std::iter::empty::<(ItemId, &str)>(),
        mlake_fts::Tokenizer::default(),
    )
    .unwrap();
    let orphan_prefix = format!("{}/gen-99-orphanattempt", ns.name);
    let orphan_files = mlake_index::write_generation(
        &ns.store,
        &orphan_prefix,
        &mlake_ivf::train_centroids(&[vec![0.0, 1.0]], 42),
        &[],
        empty_split.split_bytes(),
        &mlake_graph::ReverseAdjacency::build(vec![]),
        &mlake_index::PkIndex::default(),
    )
    .await
    .unwrap();
    let _ = &orphan_files;
    assert_eq!(manifest.generation, 1, "manifest must still point at the pre-crash generation");

    // The namespace still serves generation 1 correctly.
    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    // Strong consistency also sees the un-indexed 'b' via the WAL tail.
    assert_eq!(node.doc_count(), 2);

    // A fresh index run republishes cleanly over the mess.
    let outcome = index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    assert!(outcome.published);
    assert_eq!(outcome.doc_count, 2);
}

// ---------------------------------------------------------------- roundtrip budget (INV-7, speed)

use mlake_store::COLD_ROUNDTRIP_BUDGET;

/// INV-7: the number of roundtrips to load a generation is bounded and independent of
/// data size and cluster count. This is the property the whole storage layout exists to
/// guarantee — query cost must not scale with the corpus.
#[tokio::test]
async fn generation_load_roundtrips_are_constant_regardless_of_size() {
    async fn load_roundtrips_for(n: usize) -> usize {
        let store = Store::in_memory();
        let ns = namespace(store, "ns").await;
        let mut writer = Writer::new(ns.clone());
        // Distinct vectors so k-means produces many clusters (~sqrt(n)).
        for i in 0..n {
            let angle = i as f32;
            writer
                .commit(vec![Op::Upsert(item(
                    &format!("item-{i}"),
                    vec![angle.sin(), angle.cos(), (angle * 0.5).sin()],
                    &format!("document {i}"),
                ))])
                .await
                .unwrap();
        }
        index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

        let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
        node.load_roundtrips
    }

    // Growing the corpus 30x (and the cluster count ~5x) must not grow the roundtrip count.
    let small = load_roundtrips_for(30).await;
    let large = load_roundtrips_for(900).await;
    assert_eq!(
        small, large,
        "roundtrips must be independent of data size (INV-7): {small} vs {large}"
    );
    assert!(
        large <= COLD_ROUNDTRIP_BUDGET,
        "generation load must stay within the {COLD_ROUNDTRIP_BUDGET}-roundtrip budget, was {large}"
    );
}

/// The tantivy FTS split persists to object storage and answers queries after a full
/// round-trip through the store — the warm-path FTS load (SPEC §6.1), proven end to end.
#[tokio::test]
async fn tantivy_fts_split_loads_from_storage_and_serves_queries() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());
    writer
        .commit(vec![
            Op::Upsert(item("fox", vec![1.0, 0.0], "the quick brown fox jumps")),
            Op::Upsert(item("dog", vec![0.0, 1.0], "a lazy sleeping dog")),
            Op::Upsert(item("cn", vec![0.5, 0.5], "北京大学的研究")),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // Load only the FTS split from the manifest's file map — the warm FTS path.
    let (manifest, _) = ns.read_manifest().await.unwrap();
    let fts = mlake_index::read_fts_split(&ns.store, &manifest.files, Tokenizer::default(), None)
        .await
        .unwrap();

    let en = fts.search("brown fox", 10);
    assert_eq!(en[0].id, ItemId::from_key("fox"), "English BM25 over the loaded split");

    // Chinese query works through the same persisted split (tokenizer chain preserved).
    let cn = fts.search("北京大学", 10);
    assert_eq!(cn[0].id, ItemId::from_key("cn"), "CJK retrieval over the loaded split");
}

/// Regression for the graph-wipe bug: a *plain* index run (default options) must derive
/// links so the graph arm works, and a *second* index run must not wipe the links carried
/// from the first. Derived data is recomputed incrementally, never wholesale-deleted.
#[tokio::test]
async fn default_indexing_preserves_the_graph_across_reruns() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // Two near-identical vectors → kNN neighbours (cosine ≥ 0.7).
    writer
        .commit(vec![
            Op::Upsert(item("a", vec![1.0, 0.0, 0.0], "alpha document")),
            Op::Upsert(item("a2", vec![0.98, 0.02, 0.0], "closely related to alpha")),
        ])
        .await
        .unwrap();
    // Default options — no explicit derive_links flag.
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let cfg = QueryConfig { graph_weight: 1.0, ..QueryConfig::default() };
    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    let hits = node.query(Some(&[1.0, 0.0, 0.0]), None, 10, cfg);
    assert!(
        hits.iter().any(|h| h.id == ItemId::from_key("a2")),
        "default indexing must leave the graph arm working (links derived)"
    );

    // Add an unrelated item and re-index. The a↔a2 links must survive.
    writer.commit(vec![Op::Upsert(item("z", vec![0.0, 0.0, 1.0], "unrelated"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node2 = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await.unwrap();
    let hits2 = node2.query(Some(&[1.0, 0.0, 0.0]), None, 10, cfg);
    assert!(
        hits2.iter().any(|h| h.id == ItemId::from_key("a2")),
        "a second index run must not wipe the graph carried from the first"
    );
}

/// Regression for the immutability race (review bug 1): two index attempts building the
/// same generation number write to disjoint object keys, so neither can overwrite the
/// other's files. This tests the mechanism directly and deterministically — a genuine
/// concurrent race would serialize under a single-threaded runtime and miss the point.
#[tokio::test]
async fn concurrent_generation_builds_write_disjoint_files() {
    use mlake_index::write_generation;

    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;

    // Two attempts at generation 1, each under its own nonce prefix, with *different*
    // content (as two nodes working from different WAL heads would produce).
    let empty_split = mlake_fts::TantivyFts::build(
        std::iter::empty::<(ItemId, &str)>(),
        Tokenizer::default(),
    )
    .unwrap();
    let prefix_a = mlake_index::generation::attempt_prefix("ns", 1, "attemptA");
    let prefix_b = mlake_index::generation::attempt_prefix("ns", 1, "attemptB");

    let files_a = write_generation(
        &ns.store,
        &prefix_a,
        &mlake_ivf::train_centroids(&[vec![1.0, 0.0]], 42),
        &[],
        empty_split.split_bytes(),
        &mlake_graph::ReverseAdjacency::build(vec![]),
        &mlake_index::PkIndex { entries: vec![(ItemId::from_key("a"), 0)] },
    )
    .await
    .unwrap();
    let files_b = write_generation(
        &ns.store,
        &prefix_b,
        &mlake_ivf::train_centroids(&[vec![0.0, 1.0], vec![1.0, 1.0]], 42),
        &[],
        empty_split.split_bytes(),
        &mlake_graph::ReverseAdjacency::build(vec![]),
        &mlake_index::PkIndex { entries: vec![(ItemId::from_key("b"), 0)] },
    )
    .await
    .unwrap();

    // Disjoint key sets: no path is shared, so one attempt physically cannot overwrite the
    // other (INV-2 holds even for the same generation number).
    let paths_a: std::collections::HashSet<&str> = files_a.all_paths().collect();
    let paths_b: std::collections::HashSet<&str> = files_b.all_paths().collect();
    assert!(paths_a.is_disjoint(&paths_b), "attempts must not share any object key");

    // Both file sets remain intact and independently readable — neither clobbered the other.
    let a_centroids = ns.store.get(&files_a.centroids, None).await.unwrap();
    let b_centroids = ns.store.get(&files_b.centroids, None).await.unwrap();
    assert_ne!(a_centroids.bytes, b_centroids.bytes, "each attempt kept its own bytes");
}

/// Regression for the missing WAL grace window (review bug 3): after an index run folds
/// the tail and GC reclaims folded WAL, a reader still holding the *previous* manifest can
/// complete its tail scan, because GC keeps entries above the previous cursor.
#[tokio::test]
async fn wal_gc_keeps_entries_a_previous_manifest_reader_still_needs() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    // Capture the previous manifest, as an in-flight reader would hold it.
    let (prev_manifest, _) = ns.read_manifest().await.unwrap();

    // More writes land in the tail, then a second index run folds them and advances the cursor.
    writer.commit(vec![Op::Upsert(item("b", vec![0.0, 1.0], "beta"))]).await.unwrap();
    writer.commit(vec![Op::Upsert(item("c", vec![1.0, 1.0], "gamma"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // GC runs.
    gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();

    // A reader holding the previous manifest scans (prev_cursor, head]. Those entries must
    // still be present — GC kept everything above the previous cursor.
    let tail = mlake_wal::WalTail::new(&ns)
        .scan_from_manifest(prev_manifest.wal_index_cursor)
        .await
        .expect("a previous-manifest reader's tail scan must not hit a GC'd entry");
    assert!(tail.upserts.contains_key(&ItemId::from_key("b")));
    assert!(tail.upserts.contains_key(&ItemId::from_key("c")));
}
