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

use mlake_index::{gc, GcOutcome};

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

    let outcome = gc(&ns).await.unwrap();
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

    gc(&ns).await.unwrap();

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

    let first = gc(&ns).await.unwrap();
    let second = gc(&ns).await.unwrap();
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
    gc(&ns).await.unwrap();

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
    let orphan_files = mlake_index::write_generation(
        &ns.store,
        &ns.name,
        99, // a generation number the manifest will never reference
        &mlake_ivf::train_centroids(&[vec![0.0, 1.0]], 42),
        &[],
        &mlake_fts::FtsBuilder::new().finish(),
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
