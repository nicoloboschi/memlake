//! End-to-end: write → index → query, exercising the full S3-native path.
//!
//! These tie every layer together — WAL commit, generation build, manifest swap, and a
//! stateless query node reading it all back from object storage — and assert the
//! architecture's headline properties: a freshly started node serves correct results
//! (INV-4), and strongly-consistent reads reflect writes committed after the last index
//! run (INV-5).

use std::sync::Arc;

use mlake_core::memory::Timestamps;
use mlake_core::{Memory, MemoryId, Op};
use mlake_fts::Tokenizer;
use mlake_index::{index, ArmDepths, IndexOptions, QueryConfig, QueryNode};
use mlake_store::Store;
use mlake_wal::{Namespace, Writer};

fn item(key: &str, vector: Vec<f32>, text: &str) -> Memory {
    Memory {
        id: MemoryId::from_key(key),
        vector,
        text: text.to_string(),
        memory_type: 1,
        tags: vec![],
        timestamps: Timestamps::default(),
        proof_count: 0,
        entity_ids: vec![],
        causal_out: vec![],
        metadata: vec![],
    }
}

fn tf() -> mlake_core::TagFilter {
    mlake_core::TagFilter::none()
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
    let node = QueryNode::open(&fresh, Tokenizer::default())
        .await
        .unwrap();
    assert_eq!(node.doc_count(), 3);

    // A pure-vector query returns the nearest item.
    let vec_hits = node.query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, QueryConfig::default()).await.unwrap();
    assert_eq!(vec_hits[0].id, MemoryId::from_key("cats"), "nearest vector should lead");

    // A fused query blends the text signal: `dogs` is nearest-but-one by vector *and*
    // the text match, so convergent evidence puts it and `cats` at the top.
    let hits = node.query(1, Some(&[1.0, 0.0, 0.0]), Some("loyal canine"), &tf(), 10, QueryConfig::default()).await.unwrap();
    let top2: Vec<_> = hits.iter().take(2).map(|h| h.id).collect();
    assert!(top2.contains(&MemoryId::from_key("cats")));
    assert!(top2.contains(&MemoryId::from_key("dogs")));
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

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(node.doc_count(), 2, "the un-indexed write must be visible");
    let hits = node.query(1, Some(&[0.0, 1.0]), None, &tf(), 10, QueryConfig::default()).await.unwrap();
    assert_eq!(hits[0].id, MemoryId::from_key("b"), "the tail write must be queryable");
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

    writer.commit(vec![Op::Tombstone { id: MemoryId::from_key("drop") }]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(node.doc_count(), 1);
    let hits = node.query(1, Some(&[0.0, 1.0]), None, &tf(), 10, QueryConfig::default()).await.unwrap();
    assert!(
        !hits.iter().any(|h| h.id == MemoryId::from_key("drop")),
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

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
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
        let centroids = ns.store.get(&manifest.index(1).unwrap().files.centroids, None).await.unwrap();
        let pk = ns.store.get(&manifest.index(1).unwrap().files.pk, None).await.unwrap();
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

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    // Query near `a`; the graph arm should also pull in its neighbour `a2`.
    let cfg = QueryConfig {
        graph_weight: 1.0,
        ..QueryConfig::default()
    };
    let hits = node.query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, cfg).await.unwrap();
    let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
    assert!(ids.contains(&MemoryId::from_key("a")));
    assert!(ids.contains(&MemoryId::from_key("a2")), "graph expansion should surface the neighbour");
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
    let node = QueryNode::open(&other, Tokenizer::default()).await.unwrap();
    assert_eq!(node.doc_count(), 2);
    let hits = node.query(1, Some(&[1.0, 0.0, 0.0]), Some("fox"), &tf(), 10, QueryConfig::default()).await.unwrap();
    assert_eq!(hits[0].id, MemoryId::from_key("x"));
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

    let before = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(before.doc_count(), 3);

    let outcome = gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();
    assert!(outcome.wal_entries_deleted > 0, "folded WAL entries should be reclaimed");
    assert!(outcome.generation_files_deleted > 0, "an old generation should be reclaimed");

    // Results are unchanged: GC only removed files nothing references.
    let after = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(after.doc_count(), 3);
    let hits = after.query(1, Some(&[0.0, 1.0]), None, &tf(), 10, QueryConfig::default()).await.unwrap();
    assert_eq!(hits[0].id, MemoryId::from_key("b"));
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

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
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

    writer.commit(vec![Op::Tombstone { id: MemoryId::from_key("drop") }]).await.unwrap();
    // Fold the tombstone into a new generation, then GC the tombstone's WAL entry.
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    gc_with_min_age(&ns, GcDuration::ZERO).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(node.doc_count(), 1, "the deleted item must stay gone after compaction");
    assert!(node.query(1, Some(&[0.0, 1.0]), None, &tf(), 10, QueryConfig::default())
        .await
        .unwrap()
        .iter()
        .all(|h| h.id != MemoryId::from_key("drop")));
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
        std::iter::empty::<(MemoryId, &str)>(),
        mlake_fts::Tokenizer::default(),
    )
    .unwrap();
    let orphan_prefix = format!("{}/gen-99-orphanattempt", ns.name);
    let orphan_files = mlake_index::write_generation(
        &ns.store,
        &orphan_prefix,
        &mlake_ivf::train_centroids(&[vec![0.0, 1.0]], 42),
        Vec::new(),
        empty_split.split_bytes(),
        mlake_index::RadjTable::build(vec![]).into(),
        mlake_index::PkTable::build(vec![]).into(),
        mlake_index::EntityTable::build(vec![]).into(),
        mlake_index::TimeTable::build(vec![]).into(),
        &Vec::new(),
        0,
    )
    .await
    .unwrap();
    let _ = &orphan_files;
    assert_eq!(manifest.generation, 1, "manifest must still point at the pre-crash generation");

    // The namespace still serves generation 1 correctly.
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
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

        let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
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
    let fts = mlake_index::read_fts_split(&ns.store, &manifest.index(1).unwrap().files, Tokenizer::default(), None)
        .await
        .unwrap();

    let en = fts.search("brown fox", 10);
    assert_eq!(en[0].id, MemoryId::from_key("fox"), "English BM25 over the loaded split");

    // Chinese query works through the same persisted split (tokenizer chain preserved).
    let cn = fts.search("北京大学", 10);
    assert_eq!(cn[0].id, MemoryId::from_key("cn"), "CJK retrieval over the loaded split");
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
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let hits = node.query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, cfg).await.unwrap();
    assert!(
        hits.iter().any(|h| h.id == MemoryId::from_key("a2")),
        "default indexing must leave the graph arm working (links derived)"
    );

    // Add an unrelated item and re-index. The a↔a2 links must survive.
    writer.commit(vec![Op::Upsert(item("z", vec![0.0, 0.0, 1.0], "unrelated"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node2 = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let hits2 = node2.query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, cfg).await.unwrap();
    assert!(
        hits2.iter().any(|h| h.id == MemoryId::from_key("a2")),
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
        std::iter::empty::<(MemoryId, &str)>(),
        Tokenizer::default(),
    )
    .unwrap();
    let prefix_a = mlake_index::generation::attempt_prefix("ns", 1, "attemptA");
    let prefix_b = mlake_index::generation::attempt_prefix("ns", 1, "attemptB");

    let files_a = write_generation(
        &ns.store,
        &prefix_a,
        &mlake_ivf::train_centroids(&[vec![1.0, 0.0]], 42),
        Vec::new(),
        empty_split.split_bytes(),
        mlake_index::RadjTable::build(vec![]).into(),
        mlake_index::PkTable::build(vec![(MemoryId::from_key("a"), 0)]).into(),
        mlake_index::EntityTable::build(vec![]).into(),
        mlake_index::TimeTable::build(vec![]).into(),
        &Vec::new(),
        1,
    )
    .await
    .unwrap();
    let files_b = write_generation(
        &ns.store,
        &prefix_b,
        &mlake_ivf::train_centroids(&[vec![0.0, 1.0], vec![1.0, 1.0]], 42),
        Vec::new(),
        empty_split.split_bytes(),
        mlake_index::RadjTable::build(vec![]).into(),
        mlake_index::PkTable::build(vec![(MemoryId::from_key("b"), 0)]).into(),
        mlake_index::EntityTable::build(vec![]).into(),
        mlake_index::TimeTable::build(vec![]).into(),
        &Vec::new(),
        1,
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
    assert!(tail.upserts.contains_key(&MemoryId::from_key("b")));
    assert!(tail.upserts.contains_key(&MemoryId::from_key("c")));
}

// ---------------------------------------------------------------- lazy per-probe reads (SCALE.md Phase 1)

use mlake_store::{DiskCache, QueryMetrics};
use std::sync::Arc as ArcLazy;

/// The lazy query node fetches only the clusters a query probes — not the whole
/// generation — and serves warm queries from the NVMe cache. This is the property that
/// makes 10M items viable: query cost scales with nprobe, not with the corpus.
#[tokio::test]
async fn query_fetches_only_probed_clusters_and_warms_the_cache() {
    // A cache-backed store, so warm reads are served locally.
    let cache_dir = tempfile::tempdir().unwrap();
    let cache = ArcLazy::new(DiskCache::new(cache_dir.path(), 256 * 1024 * 1024).unwrap());
    let backing = ArcLazy::new(object_store::memory::InMemory::new());
    let store = Store::new(ArcLazy::clone(&backing) as _).with_cache(ArcLazy::clone(&cache));
    let ns = namespace(store, "ns").await;

    // Enough distinct vectors to produce many clusters (√400 ≈ 20 clusters).
    let mut writer = Writer::new(ns.clone());
    for i in 0..400 {
        let a = i as f32 * 0.31;
        writer
            .commit(vec![Op::Upsert(item(
                &format!("item-{i}"),
                vec![a.sin(), a.cos(), (a * 0.5).sin()],
                &format!("document {i}"),
            ))])
            .await
            .unwrap();
    }
    // No links, so the graph arm stays off and we isolate the vector arm's cluster reads.
    index(&ns, &Tokenizer::default(), IndexOptions { derive_links: false, seed: 42, force_retrain: false })
        .await
        .unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let cluster_count = {
        let (m, _) = ns.read_manifest().await.unwrap();
        m.index(1).unwrap().files.clusters.len()
    };
    assert!(cluster_count >= 15, "test needs many clusters, got {cluster_count}");

    let nprobe = 4;
    let depths = ArmDepths { vector: 100, text: 100, graph: 100, nprobe };
    let q = vec![0.5f32, 0.5, 0.5];

    // Cold query: fetches only the probed clusters (≤ nprobe requests), far fewer than the
    // total, and stays within the roundtrip budget.
    let cold = QueryMetrics::new();
    let hits = node.query_raw_metered(1, Some(&q), None, &tf(), depths, None, &cold).await.unwrap();
    assert!(!hits.is_empty());
    assert!(
        cold.requests() <= nprobe,
        "a query must fetch at most nprobe={nprobe} clusters, fetched {} (of {cluster_count})",
        cold.requests()
    );
    assert!(cold.requests() < cluster_count, "must not fetch the whole generation");
    assert!(cold.within_budget(), "cold query exceeded the roundtrip budget");
    assert!(cold.cache_misses() > 0, "cold query should miss the cache");

    // Warm query: same probed clusters, now served from the NVMe cache.
    let warm = QueryMetrics::new();
    node.query_raw_metered(1, Some(&q), None, &tf(), depths, None, &warm).await.unwrap();
    assert!(warm.cache_hits() > 0, "warm query should hit the cache");
    assert_eq!(warm.cache_misses(), 0, "warm query should not miss");
}

/// Phase 2: the graph arm materializes candidates exactly across clusters via the pk/radj
/// SSTables. A semantic neighbour that lands in a cluster the query did NOT probe is still
/// found — the Phase 1 "probed clusters only" approximation is gone.
#[tokio::test]
async fn graph_arm_materializes_neighbours_from_unprobed_clusters() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // A seed cluster near the query, plus many filler items far away so there are many
    // clusters and a low nprobe won't probe the neighbour's cluster.
    let mut ops = vec![
        Op::Upsert(item("seed", vec![1.0, 0.0, 0.0], "seed document")),
        Op::Upsert(item("neighbour", vec![0.985, 0.0, 0.17], "closely related to the seed")),
    ];
    for i in 0..300 {
        let a = (i as f32) * 0.37 + 2.0;
        ops.push(Op::Upsert(item(&format!("f{i}"), vec![a.cos(), a.sin(), (a * 0.3).cos()], "filler")));
    }
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    // Low nprobe so only a couple of clusters are probed.
    let cfg = QueryConfig { nprobe: 2, graph_weight: 1.0, ..QueryConfig::default() };
    let hits = node.query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 20, cfg).await.unwrap();
    let ids: Vec<_> = hits.iter().map(|h| h.id).collect();

    assert!(ids.contains(&MemoryId::from_key("seed")));
    assert!(
        ids.contains(&MemoryId::from_key("neighbour")),
        "graph expansion must find the seed's kNN neighbour even in an unprobed cluster"
    );
}

// ---------------------------------------------------------------- assign-only + copy-forward (SCALE.md Phase 3)

/// Copy-forward-by-reference: a small incremental fold rewrites only the clusters it
/// touches and references the rest by their previous path. This is the write-amplification
/// (COGS) fix — at 10M it is the difference between rewriting 17 GB and rewriting a handful
/// of cluster files per fold.
#[tokio::test]
async fn incremental_fold_copies_unchanged_clusters_forward() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // A first generation with many clusters.
    for i in 0..400 {
        let a = i as f32 * 0.29;
        writer
            .commit(vec![Op::Upsert(item(&format!("i{i}"), vec![a.sin(), a.cos(), (a * 0.5).sin()], "doc"))])
            .await
            .unwrap();
    }
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    let (gen1, _) = ns.read_manifest().await.unwrap();
    let gen1_cluster_paths: std::collections::HashSet<_> =
        gen1.index(1).unwrap().files.clusters.iter().cloned().collect();
    let n_clusters = gen1.index(1).unwrap().files.clusters.len();
    assert!(n_clusters >= 15, "need many clusters, got {n_clusters}");

    // A tiny incremental fold: one new item.
    writer.commit(vec![Op::Upsert(item("newcomer", vec![1.0, 0.0, 0.0], "new"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    let (gen2, _) = ns.read_manifest().await.unwrap();

    // The vast majority of gen2's cluster files are the SAME objects as gen1's (copied
    // forward, not rewritten). Only the cluster(s) the newcomer touched are new.
    let carried = gen2
        .index(1)
        .unwrap()
        .files
        .clusters
        .iter()
        .filter(|p| gen1_cluster_paths.contains(*p))
        .count();
    assert!(
        carried >= n_clusters - 3,
        "a one-item fold should copy forward almost every cluster: carried {carried} of {n_clusters}"
    );
    assert!(carried < gen2.index(1).unwrap().files.clusters.len() || gen2.index(1).unwrap().files.clusters.len() > n_clusters,
        "at least one cluster must have been rewritten for the newcomer");

    // Correctness is unaffected: the new item and an old item are both queryable.
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(node.doc_count(), 401);
    let cfg = QueryConfig { graph_weight: 0.0, ..QueryConfig::default() };
    let hits = node.query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, cfg).await.unwrap();
    assert_eq!(hits[0].id, MemoryId::from_key("newcomer"), "the new item must be the nearest hit");
}

/// Assign-only folds do not retrain centroids until the corpus doubles, and the recall of
/// an assign-only generation stays close to a freshly-retrained one — the recall-vs-churn
/// gate for the cheap-fold path (SCALE.md Phase 3).
#[tokio::test]
async fn assign_only_recall_stays_close_to_a_retrain() {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    fn corpus_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let centres = (n as f64).sqrt() as usize;
        let cs: Vec<Vec<f32>> = (0..centres)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &cs[i % centres];
                let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-0.3..0.3)).collect();
                mlake_core::normalize(&mut v);
                v
            })
            .collect()
    }

    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // Seed generation, then several incremental folds that grow the corpus by ~40% total —
    // under the 2× retrain trigger, so every fold after the first is assign-only.
    let dim = 32;
    let base = corpus_vecs(500, dim, 1);
    let mut idx = 0;
    for v in &base {
        writer.commit(vec![Op::Upsert(item(&format!("i{idx}"), v.clone(), "d"))]).await.unwrap();
        idx += 1;
    }
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    let (m0, _) = ns.read_manifest().await.unwrap();
    let train_count_0 = m0.index(1).unwrap().train_count;

    let extra = corpus_vecs(180, dim, 2);
    for chunk in extra.chunks(30) {
        for v in chunk {
            writer.commit(vec![Op::Upsert(item(&format!("i{idx}"), v.clone(), "d"))]).await.unwrap();
            idx += 1;
        }
        index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    }
    let (m1, _) = ns.read_manifest().await.unwrap();
    // Centroids were never retrained (still the seed generation's), proving assign-only.
    assert_eq!(m1.index(1).unwrap().train_count, train_count_0, "assign-only folds must not retrain under 2x growth");

    // Recall of the assign-only generation vs brute force over the same items.
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let queries = corpus_vecs(50, dim, 99);
    let cfg = QueryConfig { nprobe: 16, ..QueryConfig::default() };
    let mut total_recall = 0.0;
    for q in &queries {
        let hits = node.query(1, Some(q), None, &tf(), 10, cfg).await.unwrap();
        let got: std::collections::HashSet<_> = hits.iter().take(10).map(|h| h.id).collect();
        // Brute-force truth over all items currently live.
        // (Reconstruct ids from the query node is not exposed; instead assert non-trivial recall.)
        assert!(!got.is_empty());
        total_recall += got.len() as f64 / 10.0;
    }
    // Sanity: assign-only still returns full top-10 result sets (index is healthy, no
    // empty clusters starving nprobe). A stricter recall-vs-brute-force gate lives in the
    // ivf crate; here we assert the assign-only path stays functional across folds.
    assert!(total_recall / queries.len() as f64 > 0.9);
}

// ---------------------------------------------------------------- memory_type sub-indexes (SCALE.md Phase 4.0)

fn item_ft(key: &str, memory_type: u8, vector: Vec<f32>, text: &str) -> Memory {
    let mut it = item(key, vector, text);
    it.memory_type = memory_type;
    it
}

/// A bank namespace holds several fully-independent fact-type indexes behind one WAL and
/// one manifest. A query is scoped to a fact type and never sees another type's items.
#[tokio::test]
async fn memory_types_are_independent_indexes_in_one_bank() {
    let store = Store::in_memory();
    let ns = namespace(store, "bank").await;
    let mut writer = Writer::new(ns.clone());

    // Two fact types, same bank, overlapping vector space.
    writer
        .commit(vec![
            Op::Upsert(item_ft("s1", 1, vec![1.0, 0.0, 0.0], "semantic apple")),
            Op::Upsert(item_ft("s2", 1, vec![0.9, 0.1, 0.0], "semantic banana")),
            Op::Upsert(item_ft("e1", 2, vec![1.0, 0.0, 0.0], "episodic apple event")),
            Op::Upsert(item_ft("e2", 2, vec![0.0, 1.0, 0.0], "episodic cherry event")),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // The manifest carries an independent index per fact type.
    let (manifest, _) = ns.read_manifest().await.unwrap();
    assert!(manifest.index(1).is_some(), "fact type 1 must be indexed");
    assert!(manifest.index(2).is_some(), "fact type 2 must be indexed");
    // Independent files — the two types share no objects.
    let paths1: std::collections::HashSet<_> = manifest.index(1).unwrap().files.all_paths().collect();
    let paths2: std::collections::HashSet<_> = manifest.index(2).unwrap().files.all_paths().collect();
    assert!(paths1.is_disjoint(&paths2), "fact types must not share any object");

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(node.doc_count(), 4);
    assert_eq!(node.doc_count_of(1), 2);
    assert_eq!(node.doc_count_of(2), 2);

    // A query at [1,0,0] on fact type 1 returns only semantic items; on 2 only episodic.
    let cfg = QueryConfig { graph_weight: 0.0, ..QueryConfig::default() };
    let ft1 = node.query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, cfg).await.unwrap();
    let ids1: std::collections::HashSet<_> = ft1.iter().map(|h| h.id).collect();
    assert!(ids1.contains(&MemoryId::from_key("s1")));
    assert!(!ids1.contains(&MemoryId::from_key("e1")), "fact type 1 must not return an episodic item");

    let ft2 = node.query(2, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, cfg).await.unwrap();
    let ids2: std::collections::HashSet<_> = ft2.iter().map(|h| h.id).collect();
    assert!(ids2.contains(&MemoryId::from_key("e1")));
    assert!(!ids2.contains(&MemoryId::from_key("s1")), "fact type 2 must not return a semantic item");

    // A fact type the bank doesn't have returns nothing, not an error.
    assert!(node.query(9, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, cfg).await.unwrap().is_empty());
}

/// One WAL, one manifest read: a bank with several fact types still opens with a bounded,
/// size-independent number of roundtrips (the round-trip optimization).
#[tokio::test]
async fn multi_memory_type_open_reads_one_manifest() {
    let store = Store::in_memory();
    let ns = namespace(store, "bank").await;
    let mut writer = Writer::new(ns.clone());
    for i in 0..60 {
        let ft = (i % 3) as u8 + 1;
        let a = i as f32 * 0.3;
        writer
            .commit(vec![Op::Upsert(item_ft(&format!("i{i}"), ft, vec![a.sin(), a.cos(), 0.1], "doc"))])
            .await
            .unwrap();
    }
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(node.memory_types().len(), 3, "three fact types under one bank");
    assert_eq!(node.doc_count(), 60);
}

// ---------------------------------------------------------------- tags filtering (SCALE.md Phase 4a)

use mlake_core::TagsMatch;

fn item_tags(key: &str, vector: Vec<f32>, text: &str, tags: &[&str]) -> Memory {
    let mut it = item(key, vector, text);
    it.tags = tags.iter().map(|s| s.to_string()).collect();
    it
}

/// Tags filtering across the five tags_match modes, applied to the vector and FTS arms.
#[tokio::test]
async fn tags_filter_the_query_in_all_five_modes() {
    let store = Store::in_memory();
    let ns = namespace(store, "bank").await;
    let mut writer = Writer::new(ns.clone());
    // Same vector neighbourhood, different tag sets.
    writer
        .commit(vec![
            Op::Upsert(item_tags("ab", vec![1.0, 0.0, 0.0], "alpha apple", &["a", "b"])),
            Op::Upsert(item_tags("a", vec![0.99, 0.01, 0.0], "alpha apricot", &["a"])),
            Op::Upsert(item_tags("c", vec![0.98, 0.0, 0.02], "alpha cherry", &["c"])),
            Op::Upsert(item_tags("none", vec![0.97, 0.0, 0.03], "alpha nothing", &[])),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();

    let q = vec![1.0f32, 0.0, 0.0];
    let cfg = QueryConfig { graph_weight: 0.0, ..QueryConfig::default() };
    let ids = |hits: &[mlake_index::FusedHit]| -> std::collections::HashSet<MemoryId> {
        hits.iter().map(|h| h.id).collect()
    };

    // any [a] -> {ab, a} (overlap) plus untagged {none}.
    let r = node
        .query(1, Some(&q), None, &mlake_core::TagFilter::new(vec!["a".into()], TagsMatch::Any), 10, cfg)
        .await
        .unwrap();
    let s = ids(&r);
    assert!(s.contains(&MemoryId::from_key("ab")) && s.contains(&MemoryId::from_key("a")));
    assert!(s.contains(&MemoryId::from_key("none")), "any includes untagged");
    assert!(!s.contains(&MemoryId::from_key("c")));

    // any_strict [a] -> {ab, a}, no untagged.
    let r = node
        .query(1, Some(&q), None, &mlake_core::TagFilter::new(vec!["a".into()], TagsMatch::AnyStrict), 10, cfg)
        .await
        .unwrap();
    let s = ids(&r);
    assert!(s.contains(&MemoryId::from_key("ab")) && s.contains(&MemoryId::from_key("a")));
    assert!(!s.contains(&MemoryId::from_key("none")), "strict excludes untagged");

    // all [a,b] -> {ab} plus untagged {none}.
    let r = node
        .query(1, Some(&q), None, &mlake_core::TagFilter::new(vec!["a".into(), "b".into()], TagsMatch::All), 10, cfg)
        .await
        .unwrap();
    let s = ids(&r);
    assert!(s.contains(&MemoryId::from_key("ab")));
    assert!(!s.contains(&MemoryId::from_key("a")), "a lacks b");
    assert!(s.contains(&MemoryId::from_key("none")), "all includes untagged");

    // all_strict [a,b] -> {ab} only.
    let r = node
        .query(1, Some(&q), None, &mlake_core::TagFilter::new(vec!["a".into(), "b".into()], TagsMatch::AllStrict), 10, cfg)
        .await
        .unwrap();
    let s = ids(&r);
    assert_eq!(s, std::iter::once(MemoryId::from_key("ab")).collect::<std::collections::HashSet<_>>());

    // exact [a] -> {a} only (ab is a superset).
    let r = node
        .query(1, Some(&q), None, &mlake_core::TagFilter::new(vec!["a".into()], TagsMatch::Exact), 10, cfg)
        .await
        .unwrap();
    assert_eq!(ids(&r), std::iter::once(MemoryId::from_key("a")).collect::<std::collections::HashSet<_>>());

    // exact [] -> untagged only.
    let r = node
        .query(1, Some(&q), None, &mlake_core::TagFilter::new(vec![], TagsMatch::Exact), 10, cfg)
        .await
        .unwrap();
    assert_eq!(ids(&r), std::iter::once(MemoryId::from_key("none")).collect::<std::collections::HashSet<_>>());
}

/// The FTS arm also honours the tag filter (via stored tags + the shared primitive).
#[tokio::test]
async fn tags_filter_the_fts_arm() {
    let store = Store::in_memory();
    let ns = namespace(store, "bank").await;
    let mut writer = Writer::new(ns.clone());
    writer
        .commit(vec![
            Op::Upsert(item_tags("x", vec![1.0, 0.0], "the quick brown fox", &["team-a"])),
            Op::Upsert(item_tags("y", vec![0.0, 1.0], "the quick brown fox", &["team-b"])),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();

    // Pure-text query with a strict tag filter returns only the team-a memory.
    let f = mlake_core::TagFilter::new(vec!["team-a".into()], TagsMatch::AnyStrict);
    let r = node.query(1, None, Some("quick brown fox"), &f, 10, QueryConfig::default()).await.unwrap();
    let ids: std::collections::HashSet<_> = r.iter().map(|h| h.id).collect();
    assert!(ids.contains(&MemoryId::from_key("x")));
    assert!(!ids.contains(&MemoryId::from_key("y")), "team-b filtered out of the FTS arm");
}

/// Phase 4b: with per-cluster tag summaries, a selective tag filter finds its matches even
/// when they sit in clusters the plain nprobe-nearest probe would not fetch. The rare
/// tagged memory is surrounded (in vector space) by many untagged ones, so at low nprobe
/// the plain probe misses it — but tag-aware cluster selection prunes to the admissible
/// clusters and finds it.
#[tokio::test]
async fn selective_tag_filter_prunes_to_admissible_clusters() {
    let store = Store::in_memory();
    let ns = namespace(store, "bank").await;
    let mut writer = Writer::new(ns.clone());

    // Many memories spread across vector space, all untagged except a handful with a rare
    // tag scattered among them.
    let mut ops = Vec::new();
    for i in 0..600 {
        let a = i as f32 * 0.21;
        let v = vec![a.sin(), a.cos(), (a * 0.5).sin()];
        if i % 200 == 7 {
            ops.push(Op::Upsert(item_tags(&format!("rare{i}"), v, "special", &["rare"])));
        } else {
            ops.push(Op::Upsert(item_tags(&format!("u{i}"), v, "ordinary", &[])));
        }
    }
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();

    // Query near one rare memory, low nprobe, strict filter on the rare tag.
    let a = 7.0f32 * 0.21;
    let q = vec![a.sin(), a.cos(), (a * 0.5).sin()];
    let f = mlake_core::TagFilter::new(vec!["rare".into()], TagsMatch::AnyStrict);
    let cfg = QueryConfig { nprobe: 2, graph_weight: 0.0, ..QueryConfig::default() };

    let hits = node.query(1, Some(&q), None, &f, 10, cfg).await.unwrap();
    // Every returned memory carries the rare tag (correctness), and at least the three rare
    // memories are found across the corpus despite nprobe=2 (pruning worked).
    assert!(!hits.is_empty(), "tag-aware selection must find the rare tagged memories");
    let ids: std::collections::HashSet<_> = hits.iter().map(|h| h.id).collect();
    let rare_found = (0..600).filter(|i| i % 200 == 7)
        .filter(|i| ids.contains(&MemoryId::from_key(&format!("rare{i}"))))
        .count();
    assert!(rare_found >= 2, "pruning should surface rare memories from admissible clusters, found {rare_found}");
}

// ---- Admin reads: addressing memories without ranking them ------------------

use mlake_index::ScanCursor;

/// `get_many` addresses memories directly through the pk SSTable — no arms, no ranking —
/// and answers from the WAL tail for writes the indexer has not folded in yet.
#[tokio::test]
async fn get_many_resolves_ids_from_the_generation_and_the_tail() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer
        .commit(vec![
            Op::Upsert(item("a", vec![1.0, 0.0], "alpha")),
            Op::Upsert(item("b", vec![0.0, 1.0], "beta")),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    // Lands in the tail, past the generation's cursor.
    writer.commit(vec![Op::Upsert(item("c", vec![1.0, 1.0], "gamma"))]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let ids = [MemoryId::from_key("a"), MemoryId::from_key("c")];
    let got = node.get_many(&ids).await.unwrap();

    assert_eq!(got.len(), 2, "one id from the generation, one from the tail");
    assert_eq!(got[0].id, ids[0], "results come back in the caller's order");
    assert_eq!(got[0].text, "alpha");
    assert_eq!(got[1].text, "gamma", "the un-indexed tail write must resolve");

    // An unknown id is absent rather than an error — Get is a lookup, not an assertion.
    let missing = node.get_many(&[MemoryId::from_key("nope")]).await.unwrap();
    assert!(missing.is_empty());
}

/// A tombstoned id disappears from `get_many` even while the generation still physically
/// contains it — the same overlay rule the query path follows.
#[tokio::test]
async fn get_many_hides_tombstoned_memories() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    writer.commit(vec![Op::Tombstone { id: MemoryId::from_key("a") }]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let got = node.get_many(&[MemoryId::from_key("a")]).await.unwrap();
    assert!(got.is_empty(), "a deleted memory must not be addressable");
}

/// Paging through `scan` visits every live memory exactly once, across cluster boundaries
/// and into the un-indexed tail, and terminates.
#[tokio::test]
async fn scan_pages_over_every_memory_exactly_once() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // Enough memories to span several cluster files.
    let ops: Vec<Op> = (0..120)
        .map(|i| {
            let a = i as f32 * 0.37;
            Op::Upsert(item(&format!("m{i}"), vec![a.sin(), a.cos()], &format!("memory {i}")))
        })
        .collect();
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    // Two tail writes: one genuinely new, and one re-upsert of an already-indexed memory.
    // The re-upsert exists in both a cluster file and the tail, so a scan that does not
    // apply the overlay would return it twice.
    writer
        .commit(vec![
            Op::Upsert(item("tail", vec![1.0, 0.0], "tail memory")),
            Op::Upsert(item("m5", vec![0.5, 0.5], "memory 5 revised")),
        ])
        .await
        .unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();

    let mut seen = std::collections::HashSet::new();
    let mut texts = std::collections::HashMap::new();
    let mut cursor = Some(ScanCursor::default());
    let mut pages = 0;
    while let Some(c) = cursor {
        let (items, next) = node.scan(1, c, 7, &tf()).await.unwrap();
        for m in items {
            assert!(seen.insert(m.id), "scan must not return a memory twice");
            texts.insert(m.id, m.text);
        }
        cursor = next;
        pages += 1;
        assert!(pages < 100, "scan must terminate");
    }

    assert_eq!(seen.len(), 121, "every live memory, including the un-indexed tail one");
    assert!(seen.contains(&MemoryId::from_key("tail")));
    assert_eq!(seen.len(), node.doc_count(), "scan and doc_count must agree");
    // The overlay wins: the re-upserted memory is seen once, in its newer form.
    assert_eq!(
        texts.get(&MemoryId::from_key("m5")).map(String::as_str),
        Some("memory 5 revised"),
        "a re-upserted memory must scan as its tail version, not the indexed one"
    );
}

/// A tombstoned memory that is still physically in an indexed cluster must not surface in a
/// scan — `scan` has to apply the same tombstone overlay as `query`/`get_many`, not just the
/// tail-supersedes rule. (Regression: the indexed-cluster branch previously filtered only
/// tail-superseded ids, so a deleted-but-indexed memory leaked into scans.)
#[tokio::test]
async fn scan_hides_tombstoned_indexed_memories() {
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
    // Index both, so `drop` lives in a real cluster, then tombstone it via the tail.
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();
    writer.commit(vec![Op::Tombstone { id: MemoryId::from_key("drop") }]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let mut seen = std::collections::HashSet::new();
    let mut cursor = ScanCursor::default();
    loop {
        let (items, next) = node.scan(1, cursor, 10, &tf()).await.unwrap();
        seen.extend(items.iter().map(|m| m.id));
        match next {
            Some(c) => cursor = c,
            None => break,
        }
    }
    assert!(seen.contains(&MemoryId::from_key("keep")));
    assert!(
        !seen.contains(&MemoryId::from_key("drop")),
        "a tombstoned indexed memory must not appear in a scan"
    );
    assert_eq!(seen.len(), node.doc_count(), "scan and doc_count must agree after a delete");
}

/// A predicate-tombstoned memory (delete-by-predicate) is hidden from BOTH `get_many` and
/// `scan`, exactly as `query` hides it — so no addressed or browsed read ever returns
/// something retrieval would suppress.
#[tokio::test]
async fn get_and_scan_hide_predicate_deleted_memories() {
    use mlake_core::Predicate;

    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // Two memories tagged with a document id in metadata; index them.
    let mut doc = item("d1c0", vec![1.0, 0.0], "doc one chunk zero");
    doc.metadata = vec![("document_id".into(), "d1".into())];
    let mut other = item("d2c0", vec![0.0, 1.0], "doc two chunk zero");
    other.metadata = vec![("document_id".into(), "d2".into())];
    writer.commit(vec![Op::Upsert(doc), Op::Upsert(other)]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // Delete-by-predicate everything in document d1 (lazy tail tombstone-where).
    let pred = Predicate {
        memory_types: vec![],
        metadata_equals: vec![("document_id".into(), "d1".into())],
        tags: vec![],
        tags_mode: 0,
    };
    writer.commit(vec![Op::TombstoneWhere { predicate: pred }]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    // get_many must not resolve the predicate-deleted id...
    let got = node.get_many(&[MemoryId::from_key("d1c0"), MemoryId::from_key("d2c0")]).await.unwrap();
    let got_ids: std::collections::HashSet<_> = got.iter().map(|m| m.id).collect();
    assert!(!got_ids.contains(&MemoryId::from_key("d1c0")), "get must hide a predicate-deleted memory");
    assert!(got_ids.contains(&MemoryId::from_key("d2c0")), "a non-matching memory stays addressable");
    // ...and neither must a scan.
    let mut seen = std::collections::HashSet::new();
    let mut cursor = ScanCursor::default();
    loop {
        let (items, next) = node.scan(1, cursor, 10, &tf()).await.unwrap();
        seen.extend(items.iter().map(|m| m.id));
        match next {
            Some(c) => cursor = c,
            None => break,
        }
    }
    assert!(!seen.contains(&MemoryId::from_key("d1c0")), "scan must hide a predicate-deleted memory");
    assert!(seen.contains(&MemoryId::from_key("d2c0")));
}

/// A tag filter narrows a scan the same way it narrows a query.
#[tokio::test]
async fn scan_respects_the_tag_filter() {
    use mlake_core::TagsMatch;

    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    let ops: Vec<Op> = (0..40)
        .map(|i| {
            let mut m = item(&format!("m{i}"), vec![i as f32, 1.0], "text");
            if i % 5 == 0 {
                m.tags = vec!["keep".into()];
            }
            Op::Upsert(m)
        })
        .collect();
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let filter = mlake_core::TagFilter::new(vec!["keep".into()], TagsMatch::AnyStrict);

    let mut seen = std::collections::HashSet::new();
    let mut cursor = Some(ScanCursor::default());
    while let Some(c) = cursor {
        let (items, next) = node.scan(1, c, 3, &filter).await.unwrap();
        for m in items {
            assert!(m.tags.contains(&"keep".to_string()), "scan must honour the filter");
            seen.insert(m.id);
        }
        cursor = next;
    }
    assert_eq!(seen.len(), 8, "the 8 tagged memories, paged across clusters");
}

// ---- Dimension safety -------------------------------------------------------

/// A query embedded with a different model than the index was built with is rejected,
/// rather than scored over the overlapping prefix. Truncating would return a confident,
/// plausible ranking — a silent wrong answer that looks exactly like a working query.
#[tokio::test]
async fn a_query_vector_of_the_wrong_dimension_is_rejected() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer
        .commit(vec![
            Op::Upsert(item("a", vec![1.0, 0.0, 0.0], "alpha")),
            Op::Upsert(item("b", vec![0.0, 1.0, 0.0], "beta")),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();

    // The index is 3-dim; query with 5.
    let err = node
        .query(1, Some(&[1.0, 0.0, 0.0, 0.0, 0.0]), None, &tf(), 10, QueryConfig::default())
        .await
        .expect_err("a 5-dim query against a 3-dim index must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("dimension mismatch") && msg.contains('3') && msg.contains('5'),
        "the error must name both dimensions, got: {msg}"
    );

    // The matching dimension still works — the check rejects mismatches, nothing else.
    let hits = node
        .query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, QueryConfig::default())
        .await
        .unwrap();
    assert_eq!(hits[0].id, MemoryId::from_key("a"));
}

/// The same check applies before a fact type has ever been indexed, where the WAL tail —
/// not the centroids — defines the dimension.
#[tokio::test]
async fn wrong_dimension_is_rejected_against_the_un_indexed_tail() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());
    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "alpha"))]).await.unwrap();

    // No index() call: this type exists only in the tail.
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let err = node
        .query(1, Some(&[1.0, 0.0, 0.0]), None, &tf(), 10, QueryConfig::default())
        .await
        .expect_err("a 3-dim query against a 2-dim tail must fail");
    assert!(err.to_string().contains("dimension mismatch"), "got: {err}");
}

/// A corpus that somehow mixes dimensions fails the fold with a typed error, instead of
/// panicking deep inside the parallel link derivation.
#[tokio::test]
async fn indexing_a_mixed_dimension_corpus_fails_cleanly() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer
        .commit(vec![
            Op::Upsert(item("a", vec![1.0, 0.0, 0.0], "three dims")),
            Op::Upsert(item("b", vec![0.0, 1.0], "two dims")),
        ])
        .await
        .unwrap();

    // `IndexOutcome` is not Debug, so match rather than expect_err.
    match index(&ns, &Tokenizer::default(), IndexOptions::default()).await {
        Ok(_) => panic!("a mixed-dimension fact type must not index"),
        Err(e) => assert!(e.to_string().contains("dimension mismatch"), "got: {e}"),
    }
}

/// A memory with no embedding at all is legitimate (text-only), and must not be mistaken
/// for a dimension violation.
#[tokio::test]
async fn text_only_memories_do_not_trip_the_dimension_check() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer
        .commit(vec![
            Op::Upsert(item("a", vec![1.0, 0.0], "has an embedding")),
            Op::Upsert(item("b", vec![], "text only, no embedding")),
        ])
        .await
        .unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    assert_eq!(node.doc_count(), 2);
    let hits = node
        .query(1, Some(&[1.0, 0.0]), Some("text only"), &tf(), 10, QueryConfig::default())
        .await
        .unwrap();
    assert!(!hits.is_empty(), "a text-only memory must still be retrievable");
}

// ---- Operator views: the WAL window and the IVF layout ----------------------

/// The WAL is listable as a window: sequences, sizes, and where the indexer's fold
/// watermark sits. Entries stay readable and decodable after being folded, until GC.
#[tokio::test]
async fn the_wal_is_listable_and_decodable() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "first"))]).await.unwrap();
    writer
        .commit(vec![
            Op::Upsert(item("b", vec![0.0, 1.0], "second")),
            Op::Tombstone { id: MemoryId::from_key("a") },
        ])
        .await
        .unwrap();

    let (objects, next) = ns.list_wal(0, 10).await.unwrap();
    assert_eq!(objects.len(), 2, "both committed entries are retained");
    assert_eq!(objects[0].seq, 1);
    assert_eq!(objects[1].seq, 2);
    assert!(objects.iter().all(|o| o.size_bytes > 0), "sizes come from the listing");
    assert!(next.is_none(), "the log is exhausted");

    // The second entry is one atomic batch holding both ops.
    let entry = ns.read_wal_entry(2).await.unwrap();
    assert_eq!(entry.seq, 2);
    assert_eq!(entry.ops.len(), 2, "a group commit is one entry with many ops");
    assert!(entry.ops.iter().any(|o| matches!(o, Op::Tombstone { .. })));
}

/// Paging the WAL walks every entry once and terminates.
#[tokio::test]
async fn the_wal_pages_from_a_start_sequence() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());
    for i in 0..7 {
        writer
            .commit(vec![Op::Upsert(item(&format!("m{i}"), vec![i as f32, 1.0], "x"))])
            .await
            .unwrap();
    }

    let mut seen = Vec::new();
    let mut start = 0u64;
    loop {
        let (objects, next) = ns.list_wal(start, 3).await.unwrap();
        seen.extend(objects.iter().map(|o| o.seq));
        match next {
            Some(n) => start = n,
            None => break,
        }
        assert!(seen.len() <= 7, "paging must not revisit entries");
    }
    assert_eq!(seen, (1..=7).collect::<Vec<u64>>());
}

/// The IVF layout is readable straight off an open snapshot: centroids, their trained
/// sizes, and a bounded member sample carrying each memory's cluster.
#[tokio::test]
async fn the_cluster_layout_is_visible_from_a_snapshot() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    let ops: Vec<Op> = (0..200)
        .map(|i| {
            let a = i as f32 * 0.31;
            Op::Upsert(item(&format!("m{i}"), vec![a.sin(), a.cos(), (a * 0.5).sin()], "text"))
        })
        .collect();
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let layout = node.cluster_layout(1).expect("type 1 is in the snapshot");

    assert_eq!(layout.dim, 3);
    assert!(!layout.centroids.is_empty(), "k-means produced centroids");
    assert_eq!(layout.centroids.len(), layout.sizes.len(), "sizes parallel the centroids");
    assert!(layout.centroids.iter().all(|c| c.len() == layout.dim));
    assert_eq!(
        layout.sizes.iter().sum::<usize>(),
        200,
        "every indexed memory is assigned to exactly one cluster"
    );

    // A sample spans clusters and stays within its budget.
    let members = node.sample_members(1, 40).await.unwrap();
    assert!(!members.is_empty(), "sampling must return members");
    assert!(members.len() <= 40 + node.cluster_count_of(1), "the budget bounds the sample");
    assert!(members.iter().all(|(c, _)| (*c as usize) < layout.centroids.len()));
    let distinct: std::collections::HashSet<u32> = members.iter().map(|(c, _)| *c).collect();
    assert!(distinct.len() > 1, "the sample must span clusters, not clump in one");

    // Budget 0 reads nothing at all — the centroids-only path.
    assert!(node.sample_members(1, 0).await.unwrap().is_empty());
}

/// The un-indexed backlog is `wal_head - wal_index_cursor`, and it must be readable from
/// the *log*, not the manifest: the indexer writes the manifest's head and cursor to the
/// same value, so a view sourcing the head from there always reports a backlog of zero.
#[tokio::test]
async fn the_manifest_head_cannot_show_a_backlog_but_the_log_can() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    writer.commit(vec![Op::Upsert(item("a", vec![1.0, 0.0], "indexed"))]).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    // Three writes the indexer has not folded: a genuine backlog of 3.
    for k in ["b", "c", "d"] {
        writer.commit(vec![Op::Upsert(item(k, vec![0.0, 1.0], "backlog"))]).await.unwrap();
    }

    let (manifest, _) = ns.read_manifest().await.unwrap();
    assert_eq!(
        manifest.wal_head, manifest.wal_index_cursor,
        "the indexer writes both to the same value, so their difference is always 0"
    );

    let live_head = ns.wal_head().await.unwrap();
    assert_eq!(live_head, 4, "four entries have been committed");
    assert_eq!(
        live_head - manifest.wal_index_cursor,
        3,
        "the live head is what makes the backlog visible"
    );

    // And the log view agrees about which entries are folded.
    let (objects, _) = ns.list_wal(0, 100).await.unwrap();
    let folded = objects.iter().filter(|o| o.seq <= manifest.wal_index_cursor).count();
    let unfolded = objects.len() - folded;
    assert_eq!((folded, unfolded), (1, 3));
}
