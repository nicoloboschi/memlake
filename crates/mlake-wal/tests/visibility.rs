//! Write durability and visibility (INV-5, INV-4).
//!
//! The contract these pin down: once `commit` returns, the data is on object storage and
//! the very next strongly-consistent read sees it — with no indexer run in between, from
//! a node that has never seen the namespace before, and with no local state of any kind.

use std::sync::Arc;

use mlake_core::item::Timestamps;
use mlake_core::{Item, ItemId, Op};
use mlake_store::Store;
use mlake_wal::{Namespace, WalTail, Writer};

fn item(key: &str, proof: u32) -> Item {
    Item {
        id: ItemId::from_key(key),
        vector: vec![1.0, 0.0, 0.0],
        text: format!("body of {key}"),
        fact_type: 1,
        tags: vec!["tag".into()],
        timestamps: Timestamps::default(),
        proof_count: proof,
        entity_ids: vec![1, 2],
        causal_out: vec![],
    }
}

async fn namespace(store: Store, name: &str) -> Namespace {
    let ns = Namespace::new(name, store);
    ns.create_if_absent("tok-hash").await.unwrap();
    ns
}

/// INV-5: an acked write is visible to the next consistent query, with no indexing.
#[tokio::test]
async fn acked_write_is_immediately_visible_without_indexing() {
    let ns = namespace(Store::in_memory(), "ns").await;
    let mut writer = Writer::new(ns.clone());
    writer.commit(vec![Op::Upsert(item("a", 1))]).await.unwrap();

    // The manifest is still empty — nothing has been indexed at all.
    let (manifest, _) = ns.read_manifest().await.unwrap();
    assert!(manifest.is_empty(), "no generation should exist yet");

    let scan = WalTail::new(&ns)
        .scan_from_manifest(manifest.wal_index_cursor)
        .await
        .unwrap();
    assert!(scan.upserts.contains_key(&ItemId::from_key("a")));
}

/// INV-4: a node with no local state returns the same answer. Data lives on S3 alone, so
/// a freshly started process — a different `Store` handle over the same bucket — sees
/// everything.
#[tokio::test]
async fn a_node_with_no_local_state_sees_all_committed_data() {
    let backing = Arc::new(object_store::memory::InMemory::new());

    // Node A writes.
    let ns_a = namespace(Store::new(Arc::clone(&backing) as _), "ns").await;
    let mut writer = Writer::new(ns_a.clone());
    for i in 0..5 {
        writer
            .commit(vec![Op::Upsert(item(&format!("item-{i}"), i))])
            .await
            .unwrap();
    }
    drop(writer);
    drop(ns_a);

    // Node B has never touched this namespace.
    let ns_b = Namespace::new("ns", Store::new(backing as _));
    let (manifest, _) = ns_b.read_manifest().await.unwrap();
    let scan = WalTail::new(&ns_b)
        .scan_from_manifest(manifest.wal_index_cursor)
        .await
        .unwrap();

    assert_eq!(scan.upserts.len(), 5);
    for i in 0..5 {
        let id = ItemId::from_key(&format!("item-{i}"));
        assert_eq!(scan.upserts[&id].proof_count, i, "item-{i} must round-trip");
    }
}

/// Every write from a burst of concurrent writers survives — no lost updates.
#[tokio::test]
async fn no_committed_write_is_lost_under_concurrency() {
    let ns = Arc::new(namespace(Store::in_memory(), "ns").await);
    const WRITERS: usize = 8;
    const PER_WRITER: usize = 5;

    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let ns = Arc::clone(&ns);
        handles.push(tokio::spawn(async move {
            let mut writer = Writer::new((*ns).clone());
            for i in 0..PER_WRITER {
                writer
                    .commit(vec![Op::Upsert(item(&format!("w{w}-i{i}"), 0))])
                    .await
                    .unwrap();
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let scan = WalTail::new(&ns).scan(0, None).await.unwrap();
    assert_eq!(
        scan.upserts.len(),
        WRITERS * PER_WRITER,
        "every acked write must be present"
    );
    for w in 0..WRITERS {
        for i in 0..PER_WRITER {
            assert!(
                scan.upserts.contains_key(&ItemId::from_key(&format!("w{w}-i{i}"))),
                "w{w}-i{i} was lost"
            );
        }
    }
}

/// A crash between committing and indexing loses nothing: the log is the source of truth
/// and the tail scan reconstructs state from it.
#[tokio::test]
async fn state_survives_a_crash_before_indexing() {
    let backing = Arc::new(object_store::memory::InMemory::new());
    let ns = namespace(Store::new(Arc::clone(&backing) as _), "ns").await;

    let mut writer = Writer::new(ns.clone());
    writer.commit(vec![Op::Upsert(item("a", 1))]).await.unwrap();
    writer.commit(vec![Op::Upsert(item("b", 2))]).await.unwrap();
    writer
        .commit(vec![Op::Tombstone { id: ItemId::from_key("a") }])
        .await
        .unwrap();

    // Simulate the process dying here: drop everything in memory, keep only the bucket.
    drop(writer);
    drop(ns);

    let recovered = Namespace::new("ns", Store::new(backing as _));
    let scan = WalTail::new(&recovered).scan(0, None).await.unwrap();
    assert!(
        scan.is_tombstoned(&ItemId::from_key("a")),
        "the delete must survive the crash"
    );
    assert!(scan.upserts.contains_key(&ItemId::from_key("b")));
}

/// Reading `through_seq` gives a stable snapshot: writes committed afterwards are
/// invisible to that read, which is what lets a query pin a consistency point.
#[tokio::test]
async fn a_bounded_scan_is_a_stable_snapshot() {
    let ns = namespace(Store::in_memory(), "ns").await;
    let mut writer = Writer::new(ns.clone());
    let first = writer.commit(vec![Op::Upsert(item("a", 0))]).await.unwrap();
    writer.commit(vec![Op::Upsert(item("b", 0))]).await.unwrap();

    let snapshot = WalTail::new(&ns).scan(0, Some(first.seq)).await.unwrap();
    assert!(snapshot.upserts.contains_key(&ItemId::from_key("a")));
    assert!(
        !snapshot.upserts.contains_key(&ItemId::from_key("b")),
        "a write past the consistency point must not leak into the snapshot"
    );
}

/// The same guarantees against a real S3 implementation, skipped when MinIO is down.
#[tokio::test]
async fn writes_are_durable_on_s3() {
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
        "durability-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let ns = namespace(store.clone(), &name).await;
    let mut writer = Writer::new(ns.clone());
    writer.commit(vec![Op::Upsert(item("s3", 42))]).await.unwrap();

    // A completely separate handle, as a different node would have.
    let other = Namespace::new(
        &name,
        Store::s3("memlake", Some(&endpoint), "memlake", "memlake123", "us-east-1").unwrap(),
    );
    let scan = WalTail::new(&other).scan(0, None).await.unwrap();
    assert_eq!(scan.upserts[&ItemId::from_key("s3")].proof_count, 42);
}
