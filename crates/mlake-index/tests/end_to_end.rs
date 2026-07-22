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
        index_text: String::new(),
        memory_type: 1,
        tags: vec![],
        timestamps: Timestamps::default(),
        proof_count: 0,
        entity_ids: vec![],
        causal_out: vec![],
        metadata: vec![],
    }
}

/// An unconstrained predicate — `scan` filters on one of these now, not a bare TagFilter.
fn pred() -> mlake_core::Predicate {
    mlake_core::Predicate::default()
}

/// A predicate matching exactly these tags under `mode`.
fn pred_tags(tags: Vec<String>, mode: mlake_core::TagsMatch) -> mlake_core::Predicate {
    mlake_core::Predicate {
        tags,
        tags_mode: mlake_core::predicate::tags_mode_to_u8(mode),
        ..Default::default()
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
    assert_eq!(manifest.version, 2);
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
    let orphan_prefix = format!("{}/seg-99orphanattempt", ns.name);
    let orphan_files = mlake_index::write_generation(
        &ns.store,
        &orphan_prefix,
        &mlake_ivf::train_centroids(&[vec![0.0, 1.0]], 42),
        Vec::new(),
        Vec::new(),
        empty_split.split_bytes(),
        mlake_index::RadjTable::build(vec![]).into(),
        mlake_index::PkTable::build(vec![]).into(),
        mlake_index::EntityTable::build(vec![]).into(),
        mlake_index::TimeTable::build(vec![]).into(),
        mlake_index::PayloadTable::build(&[]).into(),
        mlake_index::RerankTable::build(&[]).into(),
        &Vec::new(),
        0,
    )
    .await
    .unwrap();
    let _ = &orphan_files;
    assert_eq!(manifest.version, 1, "manifest must still point at the pre-crash generation");

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
    let prefix_a = mlake_index::generation::attempt_prefix("ns", "attemptA");
    let prefix_b = mlake_index::generation::attempt_prefix("ns", "attemptB");

    let files_a = write_generation(
        &ns.store,
        &prefix_a,
        &mlake_ivf::train_centroids(&[vec![1.0, 0.0]], 42),
        Vec::new(),
        Vec::new(),
        empty_split.split_bytes(),
        mlake_index::RadjTable::build(vec![]).into(),
        mlake_index::PkTable::build(vec![(MemoryId::from_key("a"), 0)]).into(),
        mlake_index::EntityTable::build(vec![]).into(),
        mlake_index::TimeTable::build(vec![]).into(),
        mlake_index::PayloadTable::build(&[]).into(),
        mlake_index::RerankTable::build(&[]).into(),
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
        Vec::new(),
        empty_split.split_bytes(),
        mlake_index::RadjTable::build(vec![]).into(),
        mlake_index::PkTable::build(vec![(MemoryId::from_key("b"), 0)]).into(),
        mlake_index::EntityTable::build(vec![]).into(),
        mlake_index::TimeTable::build(vec![]).into(),
        mlake_index::PayloadTable::build(&[]).into(),
        mlake_index::RerankTable::build(&[]).into(),
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
    index(&ns, &Tokenizer::default(), IndexOptions { derive_links: false, ..IndexOptions::default() })
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
    let hits = node.query_raw_metered(1, Some(&q), None, &tf(), depths, None, Default::default(), &cold).await.unwrap();
    assert!(!hits.is_empty());
    // A cluster is now a *pair* of objects — `cluster-{i}.bin` (payload) and
    // `cluster-{i}.vec` (embeddings) — so probing `nprobe` clusters costs `2 * nprobe`
    // requests. They ride one wave, so the roundtrip budget below is unchanged, but the
    // request count genuinely doubled: asserted exactly, so that if the scan ever stops
    // needing the payload half, this test fails and someone has to notice the win.
    const OBJECTS_PER_CLUSTER: usize = 2;
    assert!(
        cold.requests() <= nprobe * OBJECTS_PER_CLUSTER,
        "a query must fetch at most nprobe={nprobe} clusters ({} objects), fetched {} (of {cluster_count} clusters)",
        nprobe * OBJECTS_PER_CLUSTER,
        cold.requests()
    );
    assert!(cold.requests() < cluster_count, "must not fetch the whole generation");
    assert!(cold.within_budget(), "cold query exceeded the roundtrip budget");
    assert!(cold.cache_misses() > 0, "cold query should miss the cache");

    // Warm query: same probed clusters, now served from the NVMe cache.
    let warm = QueryMetrics::new();
    node.query_raw_metered(1, Some(&q), None, &tf(), depths, None, Default::default(), &warm).await.unwrap();
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
    let got = node.get_many(&ids, true).await.unwrap();

    assert_eq!(got.len(), 2, "one id from the generation, one from the tail");
    assert_eq!(got[0].id, ids[0], "results come back in the caller's order");
    assert_eq!(got[0].text, "alpha");
    assert_eq!(got[1].text, "gamma", "the un-indexed tail write must resolve");

    // An unknown id is absent rather than an error — Get is a lookup, not an assertion.
    let missing = node.get_many(&[MemoryId::from_key("nope")], true).await.unwrap();
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
    let got = node.get_many(&[MemoryId::from_key("a")], false).await.unwrap();
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
        let (items, next) = node.scan(1, c, 7, &pred()).await.unwrap();
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
        let (items, next) = node.scan(1, cursor, 10, &pred()).await.unwrap();
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
    let doc1_predicate = Predicate {
        memory_types: vec![],
        metadata_equals: vec![("document_id".into(), "d1".into())],
        tags: vec![],
        tags_mode: 0,
        updated_from: None,
        updated_to: None,
    };
    writer.commit(vec![Op::TombstoneWhere { predicate: doc1_predicate }]).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    // get_many must not resolve the predicate-deleted id...
    let got = node.get_many(&[MemoryId::from_key("d1c0"), MemoryId::from_key("d2c0")], false).await.unwrap();
    let got_ids: std::collections::HashSet<_> = got.iter().map(|m| m.id).collect();
    assert!(!got_ids.contains(&MemoryId::from_key("d1c0")), "get must hide a predicate-deleted memory");
    assert!(got_ids.contains(&MemoryId::from_key("d2c0")), "a non-matching memory stays addressable");
    // ...and neither must a scan.
    let mut seen = std::collections::HashSet::new();
    let mut cursor = ScanCursor::default();
    loop {
        let (items, next) = node.scan(1, cursor, 10, &pred()).await.unwrap();
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
    let scan_filter = pred_tags(vec!["keep".into()], TagsMatch::AnyStrict);

    let mut seen = std::collections::HashSet::new();
    let mut cursor = Some(ScanCursor::default());
    while let Some(c) = cursor {
        let (items, next) = node.scan(1, c, 3, &scan_filter).await.unwrap();
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

// ---- Streaming (external-memory) fold ------------------------------------------------------

/// Collect every live id of `memory_type` from a node, paging the scan.
async fn scan_ids(node: &QueryNode, mt: u8) -> std::collections::BTreeSet<MemoryId> {
    let mut out = std::collections::BTreeSet::new();
    let mut cursor = mlake_index::ScanCursor::default();
    loop {
        let (items, next) = node.scan(mt, cursor, 64, &pred()).await.unwrap();
        out.extend(items.iter().map(|m| m.id));
        match next {
            Some(c) => cursor = c,
            None => break,
        }
    }
    out
}

/// A varied item: clustered vector, tags, an entity, a timestamp, some metadata.
fn rich_item(i: usize) -> Memory {
    // Ten vector "families" so k-means forms several clusters.
    let fam = i % 10;
    let mut v = vec![0.0f32; 10];
    v[fam] = 1.0;
    v[(fam + 1) % 10] = 0.15;
    Memory {
        id: MemoryId::from_key(&format!("m{i}")),
        vector: v,
        text: format!("memory number {i} family {fam} lorem ipsum"),
        index_text: String::new(),
        memory_type: if i % 3 == 0 { 2 } else { 1 },
        tags: if i % 4 == 0 { vec![] } else { vec![format!("tag-{}", i % 5)] },
        timestamps: Timestamps { occurred_start: Some(1_000 + i as i64), ..Timestamps::default() },
        proof_count: (i % 3) as u32,
        entity_ids: vec![mlake_core::EntityId::from_bytes([(i % 7) as u8; 16])],
        causal_out: vec![],
        metadata: vec![("doc".into(), format!("d{}", i % 8))],
    }
}

/// The streaming fold must produce a generation functionally equivalent to an in-RAM first
/// build with `derive_links=false`: same live set, same doc counts, correct recall, identical
/// stored content, and the same handling of tail deletes/patches.
#[tokio::test]
async fn streaming_fold_matches_in_ram_fold() {
    use mlake_core::{Predicate, TagsMatch};

    let no_links = IndexOptions { derive_links: false, ..IndexOptions::default() };

    // The exact same WAL history is written to two namespaces.
    let write_corpus = |ns: Namespace| async move {
        let mut w = Writer::new(ns.clone());
        // 120 varied items across two memory_types.
        let ups: Vec<Op> = (0..120).map(|i| Op::Upsert(rich_item(i))).collect();
        w.commit(ups).await.unwrap();
        // Tail: delete one by id, patch one, and predicate-delete everything with doc=d3.
        w.commit(vec![
            Op::Tombstone { id: MemoryId::from_key("m5") },
            Op::Patch {
                id: MemoryId::from_key("m6"),
                deltas: vec![mlake_core::wal::Delta::SetText("patched text".into())],
            },
            Op::TombstoneWhere {
                predicate: Predicate {
                    memory_types: vec![],
                    metadata_equals: vec![("doc".into(), "d3".into())],
                    tags: vec![],
                    tags_mode: TagsMatch::Any as u8,
                    updated_from: None,
                    updated_to: None,
                },
            },
        ])
        .await
        .unwrap();
    };

    let backing_a = Arc::new(object_store::memory::InMemory::new());
    let ns_a = namespace(Store::new(Arc::clone(&backing_a) as _), "a").await;
    write_corpus(ns_a.clone()).await;
    let out_a = index(&ns_a, &Tokenizer::default(), no_links).await.unwrap();

    let backing_b = Arc::new(object_store::memory::InMemory::new());
    let ns_b = namespace(Store::new(Arc::clone(&backing_b) as _), "b").await;
    write_corpus(ns_b.clone()).await;
    let out_b = mlake_index::streaming::index_streaming(&ns_b, &Tokenizer::default(), no_links)
        .await
        .unwrap();

    // Same overall doc count.
    assert_eq!(out_a.doc_count, out_b.doc_count, "streaming and in-RAM doc_count must match");
    assert!(out_b.published, "streaming fold must publish");

    let node_a = QueryNode::open(&ns_a, Tokenizer::default()).await.unwrap();
    let node_b = QueryNode::open(&ns_b, Tokenizer::default()).await.unwrap();
    assert_eq!(node_a.doc_count(), node_b.doc_count());

    // Same live set per memory_type (deletes/predicate applied identically).
    for mt in [1u8, 2u8] {
        let a = scan_ids(&node_a, mt).await;
        let b = scan_ids(&node_b, mt).await;
        assert_eq!(a, b, "live id set for memory_type {mt} must match");
    }
    // The tombstoned + predicate-deleted ids are gone from both.
    let all_b: std::collections::BTreeSet<MemoryId> =
        scan_ids(&node_b, 1).await.union(&scan_ids(&node_b, 2).await).copied().collect();
    assert!(!all_b.contains(&MemoryId::from_key("m5")), "id-tombstoned item gone");
    assert!(!all_b.contains(&MemoryId::from_key("m3")), "predicate-deleted (doc=d3) item gone");

    // Recall + equivalence: a self-query (item m1's vector) returns an exact match with score
    // ~1.0, and the streaming and in-RAM indexes agree on the top hit. (Family-1 items share a
    // vector, so which exact match wins is an id tiebreak — but it is the *same* in both.)
    let q = rich_item(1).vector;
    // Every live type-1 item in family 1 (i%10==1 and i%3!=0) — they share m1's exact vector.
    let fam1: std::collections::BTreeSet<MemoryId> = (0..120)
        .filter(|i| i % 10 == 1 && i % 3 != 0)
        .map(|i| MemoryId::from_key(&format!("m{i}")))
        .collect();
    // Pure vector recall (graph/fts off, probe every cluster): the exact match tops both, and
    // the two independently-built indexes return identical rankings.
    let vec_only = QueryConfig { nprobe: 1000, graph_weight: 0.0, fts_weight: 0.0, ..QueryConfig::default() };
    let hits_b = node_b.query(1, Some(&q), None, &tf(), 10, vec_only).await.unwrap();
    let hits_a = node_a.query(1, Some(&q), None, &tf(), 10, vec_only).await.unwrap();
    assert!(fam1.contains(&hits_b[0].id), "streaming self-query must recall an exact match");
    assert!(fam1.contains(&hits_a[0].id), "in-RAM self-query recalls an exact match too");
    let ids_b: Vec<_> = hits_b.iter().map(|h| h.id).collect();
    let ids_a: Vec<_> = hits_a.iter().map(|h| h.id).collect();
    assert_eq!(ids_a, ids_b, "streaming and in-RAM vector rankings must match");

    // Content: the patched item reads back its new text from the streaming generation.
    let got = node_b.get_many(&[MemoryId::from_key("m6")], false).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].text, "patched text", "patch must be applied in the streaming fold");

    // FTS works over the streaming split: "family 1" surfaces the family-1 item m1.
    let fts = node_b
        .query(1, None, Some("family 1"), &tf(), 10, QueryConfig::default())
        .await
        .unwrap();
    assert!(
        fts.iter().any(|h| h.id == MemoryId::from_key("m1")),
        "streaming FTS split must answer a text query"
    );
}

/// The bounded path under real spilling: fold with a deliberately tiny per-stage budget so
/// Phase-1 resolution AND the per-type cluster/payload/SSTable sorts all overflow their buffers,
/// spill sorted runs to disk, and k-way-merge them. The result must still match the in-RAM fold.
/// This guards the external-memory fold's correctness through the spill+merge path — the small
/// equivalence test above stays entirely in memory and never exercises it.
#[tokio::test]
async fn streaming_fold_bounded_budget_matches_in_ram() {
    use mlake_core::{Predicate, TagsMatch};
    use mlake_index::streaming::{index_streaming_with_budget, FoldBudget};

    let no_links = IndexOptions { derive_links: false, ..IndexOptions::default() };
    // ~8k items ≈ 2 MB of item/event bytes, well over the 1 MB-per-stage budget below, so the
    // resolution sort and the cluster/payload sorts each spill multiple runs and merge them.
    const N: usize = 8_000;
    let write_corpus = |ns: Namespace| async move {
        let mut w = Writer::new(ns.clone());
        for chunk in (0..N).collect::<Vec<_>>().chunks(2_000) {
            let ups: Vec<Op> = chunk.iter().map(|&i| Op::Upsert(rich_item(i))).collect();
            w.commit(ups).await.unwrap();
        }
        // Same tail as the small test: id delete, patch, and a predicate delete (doc=d3).
        w.commit(vec![
            Op::Tombstone { id: MemoryId::from_key("m5") },
            Op::Patch {
                id: MemoryId::from_key("m6"),
                deltas: vec![mlake_core::wal::Delta::SetText("patched text".into())],
            },
            Op::TombstoneWhere {
                predicate: Predicate {
                    memory_types: vec![],
                    metadata_equals: vec![("doc".into(), "d3".into())],
                    tags: vec![],
                    tags_mode: TagsMatch::Any as u8,
                    updated_from: None,
                    updated_to: None,
                },
            },
        ])
        .await
        .unwrap();
    };

    let ns_a = namespace(Store::in_memory(), "a").await;
    write_corpus(ns_a.clone()).await;
    let out_a = index(&ns_a, &Tokenizer::default(), no_links).await.unwrap();

    let ns_b = namespace(Store::in_memory(), "b").await;
    write_corpus(ns_b.clone()).await;
    let tiny = FoldBudget {
        resolve_mb: 1,
        cluster_mb: 1,
        payload_mb: 1,
        index_mb: 1,
        radj_mb: 1,
        fts_mb: 1,
    };
    let out_b = index_streaming_with_budget(&ns_b, &Tokenizer::default(), no_links, tiny)
        .await
        .unwrap();

    assert_eq!(out_a.doc_count, out_b.doc_count, "doc_count must match under spilling");
    assert!(out_b.published);

    let node_a = QueryNode::open(&ns_a, Tokenizer::default()).await.unwrap();
    let node_b = QueryNode::open(&ns_b, Tokenizer::default()).await.unwrap();
    // Identical live set per memory_type — the tail delete/patch/predicate resolved identically
    // through the external group-by.
    for mt in [1u8, 2u8] {
        assert_eq!(
            scan_ids(&node_a, mt).await,
            scan_ids(&node_b, mt).await,
            "live id set for memory_type {mt} must match under spilling"
        );
    }
    let all_b: std::collections::BTreeSet<MemoryId> =
        scan_ids(&node_b, 1).await.union(&scan_ids(&node_b, 2).await).copied().collect();
    assert!(!all_b.contains(&MemoryId::from_key("m5")), "id-tombstoned item gone");
    assert!(!all_b.contains(&MemoryId::from_key("m3")), "predicate-deleted (doc=d3) item gone");
    // The patch survived the external resolution (a Patch event applied to its base at merge time).
    let got = node_b.get_many(&[MemoryId::from_key("m6")], false).await.unwrap();
    assert_eq!(got[0].text, "patched text", "patch applied through the spilling resolution");
    // Recall through the spilled + merged cluster files: a self-query still tops out on an exact
    // match. (Many items share m1's vector, so assert recall of the family, not a strict ranking.)
    let q = rich_item(1).vector;
    let fam1: std::collections::BTreeSet<MemoryId> = (0..N)
        .filter(|i| i % 10 == 1 && i % 3 != 0)
        .map(|i| MemoryId::from_key(&format!("m{i}")))
        .collect();
    let vec_only =
        QueryConfig { nprobe: 1000, graph_weight: 0.0, fts_weight: 0.0, ..QueryConfig::default() };
    let hits_b = node_b.query(1, Some(&q), None, &tf(), 10, vec_only).await.unwrap();
    assert!(fam1.contains(&hits_b[0].id), "self-query recalls an exact match through spilled clusters");
}

/// The streaming fold's incremental path: fold once, then fold *again* over the previous
/// generation plus a new tail (a delete + fresh upserts). It must still match the in-RAM fold's
/// live set — this exercises streaming the prior generation's clusters and overlaying the WAL.
#[tokio::test]
async fn flush_appends_l0_and_matches_full_rebuild() {
    use mlake_index::fold;
    use mlake_index::streaming::FoldBudget;
    let opts = IndexOptions { derive_links: false, ..IndexOptions::default() };
    let budget = FoldBudget::default();
    let hi = usize::MAX; // force the in-RAM first build (not streaming)

    // The same write history, applied two ways.
    let batch1: Vec<Op> = (0..40).map(|i| Op::Upsert(rich_item(i))).collect();
    let mut reup = rich_item(3); // id 3 is memory_type 2
    reup.text = "re-upserted text".into();
    let batch2: Vec<Op> = (40..50)
        .map(|i| Op::Upsert(rich_item(i)))
        .chain(std::iter::once(Op::Upsert(reup))) // re-upsert id 3 (lives in the older segment)
        .chain(std::iter::once(Op::Tombstone { id: MemoryId::from_key("m5") })) // delete id 5 (mt 1)
        .collect();

    // A: fold after each batch → a full segment, then an appended L0 flush.
    let ns_a = namespace(Store::in_memory(), "a").await;
    let mut wa = Writer::new(ns_a.clone());
    wa.commit(batch1.clone()).await.unwrap();
    fold(&ns_a, &Tokenizer::default(), opts, budget, hi).await.unwrap();
    assert_eq!(ns_a.read_manifest().await.unwrap().0.segments.len(), 1, "first build is one segment");
    wa.commit(batch2.clone()).await.unwrap();
    fold(&ns_a, &Tokenizer::default(), opts, budget, hi).await.unwrap();
    assert_eq!(ns_a.read_manifest().await.unwrap().0.segments.len(), 2, "flush appended an L0 segment");

    // B: commit both batches, fold once → a single full-rebuild segment.
    let ns_b = namespace(Store::in_memory(), "b").await;
    let mut wb = Writer::new(ns_b.clone());
    wb.commit(batch1).await.unwrap();
    wb.commit(batch2).await.unwrap();
    fold(&ns_b, &Tokenizer::default(), opts, budget, hi).await.unwrap();
    assert_eq!(ns_b.read_manifest().await.unwrap().0.segments.len(), 1);

    let node_a = QueryNode::open(&ns_a, Tokenizer::default()).await.unwrap();
    let node_b = QueryNode::open(&ns_b, Tokenizer::default()).await.unwrap();

    // Same live set per type — the delete + re-upsert resolved identically across the segment
    // boundary (via the supersede overlay) as they do in a single full rebuild.
    for mt in [1u8, 2u8] {
        assert_eq!(scan_ids(&node_a, mt).await, scan_ids(&node_b, mt).await, "live set mt {mt}");
    }
    let all_a: std::collections::BTreeSet<MemoryId> =
        scan_ids(&node_a, 1).await.union(&scan_ids(&node_a, 2).await).copied().collect();
    assert!(!all_a.contains(&MemoryId::from_key("m5")), "deleted id gone from the flushed index");
    assert!(all_a.contains(&MemoryId::from_key("m3")), "re-upserted id present");
    assert!(all_a.contains(&MemoryId::from_key("m45")), "new id present");
    assert!(all_a.contains(&MemoryId::from_key("m0")), "old id still present");

    // The re-upsert's NEW text wins over the older segment's stale copy, in both.
    let ga = node_a.get_many(&[MemoryId::from_key("m3")], false).await.unwrap();
    let gb = node_b.get_many(&[MemoryId::from_key("m3")], false).await.unwrap();
    assert_eq!(ga[0].text, "re-upserted text", "flush: newest copy wins");
    assert_eq!(gb[0].text, "re-upserted text");

    // Vector recall identical: probe every cluster (exact), so the flushed and full indexes must agree.
    let q = rich_item(1).vector;
    let vec_only =
        QueryConfig { nprobe: 1000, graph_weight: 0.0, fts_weight: 0.0, ..QueryConfig::default() };
    let ha = node_a.query(1, Some(&q), None, &tf(), 10, vec_only).await.unwrap();
    let hb = node_b.query(1, Some(&q), None, &tf(), 10, vec_only).await.unwrap();
    let ids_a: Vec<_> = ha.iter().map(|h| h.id).collect();
    let ids_b: Vec<_> = hb.iter().map(|h| h.id).collect();
    assert_eq!(ids_a, ids_b, "vector rankings match across flush vs full rebuild");
}

/// Two similar items ingested in the SAME slice link to EACH OTHER, bidirectionally — the
/// within-batch case that matters for concurrent ingest: when many memories land together, links
/// must connect them, not just connect them to the older corpus. Mirrors Hindsight's bidirectional
/// within-batch semantic links.
#[tokio::test]
async fn flush_links_within_a_slice_are_bidirectional() {
    use mlake_index::fold;
    use mlake_index::streaming::FoldBudget;
    let opts = IndexOptions::default(); // derive_links = true
    let budget = FoldBudget::default();
    let hi = usize::MAX;

    let ns = namespace(Store::in_memory(), "ns").await;
    let mut w = Writer::new(ns.clone());
    // First build: only families 2..=8, memory_type 1 — deliberately NO family-1 items, so any
    // family-1 link a later item forms can ONLY have come from within its own slice.
    let base: Vec<Op> = [2usize, 4, 5, 7, 8, 12, 14, 15, 17, 18]
        .into_iter()
        .map(|i| Op::Upsert(rich_item(i)))
        .collect();
    w.commit(base).await.unwrap();
    fold(&ns, &Tokenizer::default(), opts, budget, hi).await.unwrap();

    // Flush a slice containing two fresh family-1 (identical-vector), memory_type-1 items together.
    let mut a = rich_item(1);
    a.id = MemoryId::from_key("wa");
    let mut b = rich_item(1);
    b.id = MemoryId::from_key("wb");
    w.commit(vec![Op::Upsert(a), Op::Upsert(b)]).await.unwrap();
    fold(&ns, &Tokenizer::default(), opts, budget, hi).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let got = node
        .get_many(&[MemoryId::from_key("wa"), MemoryId::from_key("wb")], true)
        .await
        .unwrap();
    let a_targets: std::collections::BTreeSet<MemoryId> =
        got.iter().find(|m| m.id == MemoryId::from_key("wa")).unwrap().semantic_out.iter().map(|e| e.target).collect();
    let b_targets: std::collections::BTreeSet<MemoryId> =
        got.iter().find(|m| m.id == MemoryId::from_key("wb")).unwrap().semantic_out.iter().map(|e| e.target).collect();
    assert!(a_targets.contains(&MemoryId::from_key("wb")), "wa links to its slice-mate wb, got {a_targets:?}");
    assert!(b_targets.contains(&MemoryId::from_key("wa")), "wb links to its slice-mate wa, got {b_targets:?}");
}

/// The streaming (external-memory) fold — the >4M-doc path — must NOT drop semantic links: it
/// derives them home-cluster-at-a-time so identical-vector items in the same cluster link, and
/// the reverse edges reach radj. Forced here at tiny scale via streaming_threshold_docs = 0.
#[tokio::test]
async fn streaming_fold_derives_home_cluster_links() {
    use mlake_index::fold;
    use mlake_index::streaming::FoldBudget;
    let opts = IndexOptions::default(); // derive_links = true
    let budget = FoldBudget::default();

    let ns = namespace(Store::in_memory(), "ns").await;
    let mut w = Writer::new(ns.clone());
    // Four family-1 (identical-vector) memory_type-1 items — m1, m31, m61, m91 — plus other
    // families, so a family-1 cluster forms and its members should link to one another.
    let ids = [1usize, 31, 61, 91, 2, 4, 5, 7];
    w.commit(ids.iter().map(|&i| Op::Upsert(rich_item(i))).collect()).await.unwrap();
    // threshold_docs = 0 forces the streaming fold even at this tiny scale.
    fold(&ns, &Tokenizer::default(), opts, budget, 0).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let got = node.get_many(&[MemoryId::from_key("m1")], true).await.unwrap();
    assert_eq!(got.len(), 1, "m1 survives the streaming build");
    let targets: std::collections::BTreeSet<MemoryId> =
        got[0].semantic_out.iter().map(|e| e.target).collect();
    let fam1 = [MemoryId::from_key("m31"), MemoryId::from_key("m61"), MemoryId::from_key("m91")];
    assert!(
        fam1.iter().any(|id| targets.contains(id)),
        "streaming fold derived home-cluster links (not dropped), got {targets:?}"
    );
}

/// A flush derives semantic links against the WHOLE corpus, not just its own slice: a new item
/// flushed into an L0 segment links to a same-family item that lives in the older segment.
#[tokio::test]
async fn flush_derives_links_across_segments() {
    use mlake_index::fold;
    use mlake_index::streaming::FoldBudget;
    let opts = IndexOptions::default(); // derive_links = true
    let budget = FoldBudget::default();
    let hi = usize::MAX;

    let ns = namespace(Store::in_memory(), "ns").await;
    let mut w = Writer::new(ns.clone());
    // First build: items 0..40 (family 1 = i % 10 == 1). Ids 1, 11, 31 are family 1, memory_type 1.
    w.commit((0..40).map(|i| Op::Upsert(rich_item(i))).collect()).await.unwrap();
    fold(&ns, &Tokenizer::default(), opts, budget, hi).await.unwrap();

    // Flush a NEW item with family-1's vector (a copy of item 1 under a fresh id).
    let mut newitem = rich_item(1);
    newitem.id = MemoryId::from_key("m100");
    newitem.text = "new family one".into();
    w.commit(vec![Op::Upsert(newitem)]).await.unwrap();
    fold(&ns, &Tokenizer::default(), opts, budget, hi).await.unwrap();
    assert_eq!(ns.read_manifest().await.unwrap().0.segments.len(), 2, "the new item flushed to an L0");

    // Its links reach into the OLDER segment's family-1 members.
    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let got = node.get_many(&[MemoryId::from_key("m100")], true).await.unwrap();
    assert_eq!(got.len(), 1);
    let targets: std::collections::BTreeSet<MemoryId> =
        got[0].semantic_out.iter().map(|e| e.target).collect();
    let fam1 = [MemoryId::from_key("m1"), MemoryId::from_key("m11"), MemoryId::from_key("m31")];
    assert!(
        fam1.iter().any(|id| targets.contains(id)),
        "flushed item links across the segment boundary to an older family-1 item, got {targets:?}"
    );
}

/// Phase 4: after enough flushes the segment count reaches the fan-out and a fold compacts them
/// all back into one segment — bounding the per-query fan-out — without losing or corrupting data.
#[tokio::test]
async fn compaction_merges_segments_and_preserves_data() {
    use mlake_index::streaming::FoldBudget;
    use mlake_index::{fold, COMPACT_FANOUT};
    let opts = IndexOptions { derive_links: false, ..IndexOptions::default() };
    let budget = FoldBudget::default();
    let hi = usize::MAX;

    let ns = namespace(Store::in_memory(), "ns").await;
    let mut w = Writer::new(ns.clone());
    // First build, then flush repeatedly (a fresh batch each time) until a fold compacts. Also
    // delete one id and re-upsert another mid-stream, so compaction must resolve them correctly.
    let mut reup = rich_item(2);
    reup.text = "reupserted".into();
    for b in 0..=COMPACT_FANOUT {
        let mut batch: Vec<Op> = (b * 5..b * 5 + 5).map(|i| Op::Upsert(rich_item(i))).collect();
        if b == 3 {
            batch.push(Op::Tombstone { id: MemoryId::from_key("m1") }); // delete id 1
            batch.push(Op::Upsert(reup.clone())); // re-upsert id 2
        }
        w.commit(batch).await.unwrap();
        fold(&ns, &Tokenizer::default(), opts, budget, hi).await.unwrap();
    }
    // The fold at COMPACT_FANOUT segments compacted them back down.
    let segs = ns.read_manifest().await.unwrap().0.segments.len();
    assert!(segs < COMPACT_FANOUT, "compaction bounded the segment count, got {segs}");

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let live: std::collections::BTreeSet<MemoryId> =
        scan_ids(&node, 1).await.union(&scan_ids(&node, 2).await).copied().collect();
    // Every upserted id survives compaction except the deleted one.
    let last = (COMPACT_FANOUT * 5 + 5) as usize;
    for i in 0..last {
        let present = live.contains(&MemoryId::from_key(&format!("m{i}")));
        if i == 1 {
            assert!(!present, "deleted id gone through compaction");
        } else {
            assert!(present, "id {i} survived compaction");
        }
    }
    // The re-upsert's text survived the merge.
    let got = node.get_many(&[MemoryId::from_key("m2")], false).await.unwrap();
    assert_eq!(got[0].text, "reupserted", "re-upsert resolved by compaction");
}

/// The streaming fold's incremental path: fold once, then fold *again* over the previous
/// generation plus a new tail. It must still match the in-RAM fold's live set.
#[tokio::test]
async fn streaming_fold_incremental_matches_in_ram() {
    async fn build(streaming: bool) -> std::collections::BTreeSet<MemoryId> {
        let no_links = IndexOptions { derive_links: false, ..IndexOptions::default() };
        let store = Store::in_memory();
        let ns = namespace(store, "ns").await;
        let mut w = Writer::new(ns.clone());
        // First batch, then fold (creates generation 1).
        w.commit((0..80).map(|i| Op::Upsert(rich_item(i))).collect()).await.unwrap();
        if streaming {
            mlake_index::streaming::index_streaming(&ns, &Tokenizer::default(), no_links).await.unwrap();
        } else {
            index(&ns, &Tokenizer::default(), no_links).await.unwrap();
        }
        // New tail past the cursor: delete an indexed item, add fresh ones.
        w.commit(
            std::iter::once(Op::Tombstone { id: MemoryId::from_key("m10") })
                .chain((80..110).map(|i| Op::Upsert(rich_item(i))))
                .collect(),
        )
        .await
        .unwrap();
        // Fold again — reads generation 1 (prev-gen streaming) + the new tail.
        if streaming {
            mlake_index::streaming::index_streaming(&ns, &Tokenizer::default(), no_links).await.unwrap();
        } else {
            index(&ns, &Tokenizer::default(), no_links).await.unwrap();
        }
        let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
        scan_ids(&node, 1).await.union(&scan_ids(&node, 2).await).copied().collect()
    }

    let ids_ram = build(false).await;
    let ids_stream = build(true).await;
    assert_eq!(ids_ram, ids_stream, "incremental streaming fold must match in-RAM live set");
    assert!(!ids_stream.contains(&MemoryId::from_key("m10")), "tombstoned item stays deleted across the re-fold");
    assert!(ids_stream.contains(&MemoryId::from_key("m100")), "a fresh tail item is indexed");
}

// ---- The scan/payload split -------------------------------------------------

/// A tag-filtered query must not read the payload of the members it filters out.
///
/// Tags live in the vector block as a per-cluster dictionary plus per-member bitmaps, so
/// the filter is applied exactly and *before* scoring. If tag filtering ever regresses to
/// "materialize, then retain", this test fails: the payload reads would scale with the
/// probed clusters instead of with the surviving hits.
#[tokio::test]
async fn a_tag_filtered_query_filters_before_reading_any_payload() {
    use mlake_core::TagsMatch;

    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    // 400 memories, of which only 8 carry the rare tag.
    let ops: Vec<Op> = (0..400)
        .map(|i| {
            let a = i as f32 * 0.23;
            let mut m = item(&format!("m{i}"), vec![a.sin(), a.cos(), (a * 0.7).sin()], "text");
            if i % 50 == 0 {
                m.tags = vec!["rare".into()];
            } else {
                m.tags = vec!["common".into()];
            }
            Op::Upsert(m)
        })
        .collect();
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let filter = mlake_core::TagFilter::new(vec!["rare".into()], TagsMatch::AnyStrict);
    let depths = ArmDepths { vector: 10, text: 0, graph: 0, nprobe: 8 };
    let q = vec![0.4f32, 0.6, 0.2];

    let metrics = QueryMetrics::new();
    let hits = node
        .query_raw_metered(1, Some(&q), None, &filter, depths, None, Default::default(), &metrics)
        .await
        .unwrap();

    // Correctness first: the filter is exact, so nothing untagged can slip through.
    assert!(!hits.is_empty(), "the rare-tagged memories must still be found");
    for h in &hits {
        let m = h.memory.as_ref().expect("a hit carries its memory inline");
        assert!(
            m.tags.contains(&"rare".to_string()),
            "an unfiltered memory reached the results: {:?}",
            m.tags
        );
    }
    assert!(metrics.within_budget(), "the split must not cost extra roundtrips");
}

/// The graph arm seeds off the dense ranking, and its seeds' adjacency used to come free
/// from the cluster fetch. With payload deferred to the winners, that only still holds if
/// every seed is among the hydrated hits — so assert it rather than assume it.
#[tokio::test]
async fn graph_seeds_are_covered_by_the_hydrated_winners() {
    let store = Store::in_memory();
    let ns = namespace(store, "ns").await;
    let mut writer = Writer::new(ns.clone());

    let ops: Vec<Op> = (0..120)
        .map(|i| {
            let a = i as f32 * 0.37;
            Op::Upsert(item(&format!("g{i}"), vec![a.sin(), a.cos(), (a * 0.5).sin()], "linked"))
        })
        .collect();
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let depths = ArmDepths { vector: 20, text: 0, graph: 20, nprobe: 8 };
    let q = vec![0.3f32, 0.9, 0.1];
    let metrics = QueryMetrics::new();
    let hits = node
        .query_raw_metered(1, Some(&q), None, &tf(), depths, None, Default::default(), &metrics)
        .await
        .unwrap();

    // Every hit the dense arm surfaced must carry its memory: that inline payload is what
    // the graph arm reads its seed adjacency from.
    let dense: Vec<_> = hits.iter().filter(|h| h.dense.is_some()).collect();
    assert!(!dense.is_empty(), "the dense arm must surface seeds");
    for h in &dense {
        assert!(
            h.memory.is_some(),
            "a dense hit without inline payload leaves the graph arm without its seed"
        );
    }
    assert!(metrics.within_budget());
}

/// The `updated_at` window is a real push-down in the dense arm, not a post-filter.
///
/// The distinction only shows when the window is selective and the matching memories rank
/// *below* the arm's depth: the arm truncates to `depths.vector` before anything is
/// materialized, so filtering afterwards can only remove rows from a page that already holds
/// none of them. This lays out the corpus so that is exactly the case — the nearest 30
/// memories are all outside the window and the only 5 inside it sit at ranks 31-35 — and
/// asserts the arm reaches them at a depth of 10.
///
/// This is the shape the one real caller has: a "what changed since my last refresh" query
/// against a bank whose recent writes are a thin slice of the whole.
#[tokio::test]
async fn the_updated_window_reaches_past_the_arm_depth() {
    use mlake_index::UpdatedWindow;

    const N: usize = 60;
    const IN_WINDOW: std::ops::Range<usize> = 30..35;

    let backing = Arc::new(object_store::memory::InMemory::new());
    let ns = namespace(Store::new(Arc::clone(&backing) as _), "ns").await;

    // Vectors on an arc, so similarity to `q` falls monotonically with `i` and the ranking
    // the arm produces is known ahead of time.
    let mut writer = Writer::new(ns.clone());
    let ops: Vec<Op> = (0..N)
        .map(|i| {
            let theta = i as f32 * 0.02;
            let mut m = item(
                &format!("m{i:02}"),
                vec![theta.cos(), theta.sin(), 0.0],
                &format!("memory {i}"),
            );
            // Everything is old except the slice the window selects.
            m.timestamps.updated_at =
                Some(if IN_WINDOW.contains(&i) { 5_000 } else { 1_000 });
            Op::Upsert(m)
        })
        .collect();
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await.unwrap();

    let node = QueryNode::open(&Namespace::new("ns", Store::new(backing as _)), Tokenizer::default())
        .await
        .unwrap();
    let q = [1.0f32, 0.0, 0.0];
    let depths = ArmDepths { vector: 10, text: 0, graph: 0, nprobe: 64 };
    let metrics = mlake_store::QueryMetrics::new();

    // Unfiltered, the page is the nearest 10 — none of which the window admits. This is what
    // a post-filter would have had to work with.
    let unfiltered = node
        .query_raw_metered(1, Some(&q), None, &tf(), depths, None, UpdatedWindow::default(), &metrics)
        .await
        .unwrap();
    assert_eq!(unfiltered.len(), 10);
    assert!(
        unfiltered.iter().all(|h| !IN_WINDOW.contains(&key_index(h))),
        "the fixture is wrong: an in-window memory ranked inside the top 10 on its own"
    );

    // Pushed down, the same depth reaches the matching slice instead.
    let window = UpdatedWindow { from: Some(4_000), to: Some(6_000) };
    let filtered = node
        .query_raw_metered(1, Some(&q), None, &tf(), depths, None, window, &metrics)
        .await
        .unwrap();
    let mut found: Vec<usize> = filtered.iter().map(key_index).collect();
    found.sort();
    assert_eq!(
        found,
        IN_WINDOW.collect::<Vec<_>>(),
        "the arm must return every in-window memory and nothing else"
    );

    // A narrow probe must not starve the window either: the matching memories sit in clusters
    // the plain nprobe-nearest probe would never reach, so the per-cluster write-time summary
    // has to pull them into the probe set — the same treatment a selective tag filter gets.
    let narrow = ArmDepths { nprobe: 1, ..depths };
    let mut narrow_found: Vec<usize> = node
        .query_raw_metered(1, Some(&q), None, &tf(), narrow, None, window, &metrics)
        .await
        .unwrap()
        .iter()
        .map(key_index)
        .collect();
    narrow_found.sort();
    assert_eq!(
        narrow_found,
        IN_WINDOW.collect::<Vec<_>>(),
        "cluster pruning must reach the in-window memories even at nprobe=1"
    );

    // And the window's edges are exclusive, exactly as `Predicate` defines them.
    let touching = UpdatedWindow { from: Some(5_000), to: None };
    assert!(
        node.query_raw_metered(1, Some(&q), None, &tf(), depths, None, touching, &metrics)
            .await
            .unwrap()
            .is_empty(),
        "`from` is exclusive, so a memory written exactly at it must not match"
    );
}

/// `i` back out of a hit's `m{i:02}` key, via the memory the hit carries.
fn key_index(hit: &mlake_index::RawHit) -> usize {
    let text = &hit.memory.as_ref().expect("a hit always carries its memory").text;
    text.strip_prefix("memory ").expect("fixture text").parse().expect("fixture index")
}

// ---------------------------------------------------------------- adaptive probing (cluster radii)

/// A fold that keeps embeddings exactly as written. The default `Binary` codec makes the
/// fold's carried-forward vectors lossy decodes of the previous generation's block, which
/// would blur this assertion by the codec's error rather than by the thing under test.
fn exact_codec() -> IndexOptions {
    IndexOptions { vector_codec: mlake_ivf::VectorCodec::F32, ..IndexOptions::default() }
}

/// Assert every member of every cluster lies inside the radius the fold wrote for it.
///
/// The vectors come from the caller's own map, not from the generation: the cluster `.bin`
/// no longer carries embeddings (they live in the `.vec` block), so the only copy a test can
/// compare against is the one it wrote.
async fn assert_radii_contain_every_member(
    ns: &Namespace,
    truth: &std::collections::HashMap<MemoryId, Vec<f32>>,
    what: &str,
) {
    let (manifest, _) = ns.read_manifest().await.unwrap();
    let files = &manifest.index(1).unwrap().files;
    let gen = mlake_index::read_generation(&ns.store, files, manifest.version, None).await.unwrap();
    let centroids = &gen.centroids;
    assert_eq!(
        centroids.radii.len(),
        centroids.len(),
        "{what}: one radius per centroid, or every reader must treat them all as unknown"
    );
    let mut checked = 0usize;
    for (c, cluster) in gen.clusters.iter().enumerate() {
        let r = centroids.radius(c).unwrap_or_else(|| panic!("{what}: cluster {c} has no radius"));
        for m in cluster {
            let v = &truth[&m.id];
            assert!(
                mlake_ivf::member_radius(&centroids.vectors[c], v) <= r + 1e-5,
                "{what}: member of cluster {c} lies outside the radius the fold wrote"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, truth.len(), "{what}: every member must have been checked");
}

/// Every fold must write a cluster radius per centroid, and it must actually contain that
/// cluster's members. A radius that is too small silently retires a cluster that held a
/// winner, and nothing downstream would report it — so it is checked at the source.
#[tokio::test]
async fn the_fold_writes_a_radius_that_contains_every_member() {
    let store = Store::in_memory();
    let ns = namespace(store, "radii").await;
    let mut writer = Writer::new(ns.clone());
    let mut truth = std::collections::HashMap::new();
    let mut ops = Vec::new();
    for i in 0..400 {
        let a = i as f32 * 0.31;
        let v = vec![a.cos(), a.sin(), (a * 0.7).cos()];
        truth.insert(MemoryId::from_key(&format!("m{i}")), v.clone());
        ops.push(Op::Upsert(item(&format!("m{i}"), v, "doc")));
    }
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), exact_codec()).await.unwrap();

    assert_radii_contain_every_member(&ns, &truth, "first fold").await;
}

/// The assign-only path adds members to centroids it did not retrain, and a flush appends a
/// whole segment without touching the old one. A radius carried forward from the previous
/// generation would be too small — i.e. unsound — the moment one new member lands outside it,
/// so every fold must recompute rather than copy it.
#[tokio::test]
async fn a_radius_does_not_go_stale_across_folds() {
    let store = Store::in_memory();
    let ns = namespace(store, "radii-refold").await;
    let mut writer = Writer::new(ns.clone());
    let mut truth = std::collections::HashMap::new();

    // A first fold over a tight cloud, so the radii it writes are small.
    let mut ops = Vec::new();
    for i in 0..200 {
        let a = i as f32 * 0.017;
        let v = vec![1.0, a.sin() * 0.02, a * 0.001];
        truth.insert(MemoryId::from_key(&format!("a{i}")), v.clone());
        ops.push(Op::Upsert(item(&format!("a{i}"), v, "doc")));
    }
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), exact_codec()).await.unwrap();

    // Then a slice of far-flung members, which land in those same clusters without a retrain.
    let mut ops = Vec::new();
    for i in 0..20 {
        let a = i as f32 * 0.3;
        let v = vec![a.cos(), a.sin(), 1.0];
        truth.insert(MemoryId::from_key(&format!("b{i}")), v.clone());
        ops.push(Op::Upsert(item(&format!("b{i}"), v, "doc")));
    }
    writer.commit(ops).await.unwrap();
    index(&ns, &Tokenizer::default(), exact_codec()).await.unwrap();

    assert_radii_contain_every_member(&ns, &truth, "assign-only re-fold").await;
}
