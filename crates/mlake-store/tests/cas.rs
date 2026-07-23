//! Conditional-write semantics (INV-3).
//!
//! The whole design rests on `If-None-Match` and `If-Match` behaving correctly, so these
//! run against both the deployment target (S3, via MinIO) and the in-memory stand-in that
//! implements the same conditional-put contract. Any backend that silently degraded a
//! conditional write into an unconditional one would lose committed data without erroring,
//! so the assertions here are deliberately blunt.
//!
//! The MinIO cases are skipped unless a MinIO endpoint is reachable, so `cargo test` works
//! on a machine with no Docker; CI and the nightly runs start it via docker-compose.

use std::sync::Arc;

use mlake_store::{Error, Store};

struct Backend {
    name: &'static str,
    store: Store,
}

async fn backends() -> Vec<Backend> {
    // Only backends that faithfully implement the S3 conditional-put contract. The local
    // filesystem is deliberately absent: it cannot do `If-Match` at all, so it can never
    // host a manifest swap and is not a supported storage target.
    let mut out = vec![Backend {
        name: "memory",
        store: Store::in_memory(),
    }];

    if let Some(store) = minio().await {
        out.push(Backend { name: "minio", store });
    } else {
        eprintln!("note: MinIO unreachable, skipping the S3-compatible CAS cases");
    }
    out
}

/// A MinIO-backed store, or `None` if the dev stack is not running.
async fn minio() -> Option<Store> {
    let endpoint =
        std::env::var("MEMLAKE_S3_ENDPOINT").unwrap_or_else(|_| "http://localhost:9000".into());
    let store = Store::s3("memlake", Some(&endpoint), "memlake", "memlake123", "us-east-1").ok()?;
    // A cheap probe: if the bucket does not answer, the stack is down.
    match store.exists("__probe__").await {
        Ok(_) => Some(store),
        Err(_) => None,
    }
}

/// Unique key per test run so repeated runs against a persistent MinIO volume don't
/// collide with their own history.
fn key(test: &str) -> String {
    format!(
        "test-{test}-{}/obj",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

#[tokio::test]
async fn put_if_absent_creates_then_rejects() {
    for Backend { name, store } in backends().await {
        let path = key("create");
        store
            .put_if_absent(&path, b"first".to_vec())
            .await
            .unwrap_or_else(|e| panic!("{name}: first create should succeed: {e}"));

        let second = store.put_if_absent(&path, b"second".to_vec()).await;
        assert!(
            matches!(second, Err(Error::AlreadyExists(_))),
            "{name}: second create must be rejected, got {second:?}"
        );

        // The loser must not have overwritten the winner.
        let got = store.get(&path, None).await.unwrap();
        assert_eq!(&got.bytes[..], b"first", "{name}: winner's bytes must survive");
    }
}

#[tokio::test]
async fn cas_swap_succeeds_on_matching_etag_and_fails_on_stale() {
    for Backend { name, store } in backends().await {
        let path = key("swap");
        store.put_if_absent(&path, b"v1".to_vec()).await.unwrap();

        let v1 = store.get(&path, None).await.unwrap();
        let etag1 = v1.etag.expect("backend must expose an etag");

        store
            .cas_swap(&path, &etag1, b"v2".to_vec())
            .await
            .unwrap_or_else(|e| panic!("{name}: swap with a current etag should succeed: {e}"));

        // etag1 is now stale; a writer holding it must be told, not silently allowed.
        let stale = store.cas_swap(&path, &etag1, b"v3".to_vec()).await;
        assert!(
            matches!(stale, Err(Error::CasConflict(_))),
            "{name}: stale swap must conflict, got {stale:?}"
        );

        let got = store.get(&path, None).await.unwrap();
        assert_eq!(&got.bytes[..], b"v2", "{name}: stale write must not land");
    }
}

/// The WAL commit protocol: N writers race for one sequence number, exactly one wins.
#[tokio::test]
async fn concurrent_creates_produce_exactly_one_winner() {
    for Backend { name, store } in backends().await {
        let path = Arc::new(key("race"));
        let store = Arc::new(store);

        let mut handles = Vec::new();
        for i in 0..8 {
            let store = Arc::clone(&store);
            let path = Arc::clone(&path);
            handles.push(tokio::spawn(async move {
                store.put_if_absent(&path, format!("writer-{i}").into_bytes()).await
            }));
        }

        let mut winners = 0;
        let mut conflicts = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => winners += 1,
                Err(e) if e.is_conflict() => conflicts += 1,
                Err(e) => panic!("{name}: unexpected error racing for a WAL slot: {e}"),
            }
        }
        assert_eq!(winners, 1, "{name}: exactly one writer may claim a slot");
        assert_eq!(conflicts, 7, "{name}: every loser must see a conflict");
    }
}

/// Likewise for the manifest swap: concurrent indexers must not both believe they
/// published a generation.
#[tokio::test]
async fn concurrent_swaps_produce_exactly_one_winner() {
    for Backend { name, store } in backends().await {
        let path = Arc::new(key("swap-race"));
        let store = Arc::new(store);
        // The base content must differ from every writer's payload. An etag is derived
        // from content, so a writer that rewrites the existing bytes leaves the version
        // unchanged and a second writer's If-Match would still legitimately match — see
        // `rewriting_identical_content_has_backend_specific_etag_semantics` below.
        store.put_if_absent(&path, b"base".to_vec()).await.unwrap();
        let etag = Arc::new(store.get(&path, None).await.unwrap().etag.unwrap());

        let mut handles = Vec::new();
        for i in 0..8 {
            let store = Arc::clone(&store);
            let path = Arc::clone(&path);
            let etag = Arc::clone(&etag);
            handles.push(tokio::spawn(async move {
                store.cas_swap(&path, &etag, format!("gen-{i}").into_bytes()).await
            }));
        }

        let mut winners = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => winners += 1,
                Err(e) if e.is_conflict() => {}
                Err(e) => panic!("{name}: unexpected error swapping the manifest: {e}"),
            }
        }
        assert_eq!(
            winners, 1,
            "{name}: only one indexer may publish from a given base generation"
        );
    }
}

#[tokio::test]
async fn missing_objects_report_not_found() {
    for Backend { name, store } in backends().await {
        let missing = store.get(&key("absent"), None).await;
        assert!(
            matches!(missing, Err(Error::NotFound(_))),
            "{name}: expected NotFound, got {missing:?}"
        );
    }
}

#[tokio::test]
async fn delete_is_idempotent() {
    // GC runs on any node and may race another node doing the same work, so deleting an
    // already-deleted file must not be an error.
    for Backend { name, store } in backends().await {
        let path = key("delete");
        store.put_if_absent(&path, b"x".to_vec()).await.unwrap();
        store.delete(&path).await.unwrap();
        store
            .delete(&path)
            .await
            .unwrap_or_else(|e| panic!("{name}: repeated delete must be a no-op: {e}"));
        assert!(!store.exists(&path).await.unwrap());
    }
}


/// Etag semantics differ across backends, and memlake must not depend on either flavour.
///
/// * S3/MinIO derive the etag from content, so rewriting identical bytes leaves the
///   version *unchanged* — a concurrent writer holding the "old" etag still passes
///   `If-Match`, because that etag is in fact still current.
/// * `InMemory` uses a monotonic version counter, so the same rewrite *does* advance it.
///
/// The consequence for the design: CAS serializes *changes*, not *attempts*. Nothing may
/// infer "I was the only writer" from a successful swap, and nothing may use an etag as a
/// generation counter. Both are safe for the manifest, since two indexers can only write
/// byte-identical manifests by having done identical work.
///
/// This test documents the divergence so a future backend swap surfaces it loudly.
#[tokio::test]
async fn rewriting_identical_content_has_backend_specific_etag_semantics() {
    for Backend { name, store } in backends().await {
        let path = key("same-content");
        store.put_if_absent(&path, b"same".to_vec()).await.unwrap();
        let before = store.get(&path, None).await.unwrap().etag.unwrap();

        store.cas_swap(&path, &before, b"same".to_vec()).await.unwrap();
        let after = store.get(&path, None).await.unwrap().etag.unwrap();

        match name {
            "minio" => assert_eq!(
                before, after,
                "minio: content-derived etag must not change for identical bytes"
            ),
            "memory" => assert_ne!(
                before, after,
                "memory: counter-derived etag must advance on every write"
            ),
            other => panic!("unclassified backend {other}: pin its etag semantics here"),
        }

        // Whichever flavour applies, the object still holds exactly what was written and
        // no update was lost.
        let got = store.get(&path, None).await.unwrap();
        assert_eq!(&got.bytes[..], b"same", "{name}: content must survive the rewrite");
    }
}

// ---- Cache inspection -------------------------------------------------------

/// The two tiers OVERLAP rather than partition: a memory eviction demotes an object to
/// disk without dropping its bytes, so one object can be resident in both at once. An
/// inspector must report that as two independent flags, never as a single tier.
#[tokio::test]
async fn cache_entries_report_both_tiers_and_overlap() {
    use mlake_store::{CacheKey, DiskCache};

    let dir = tempfile::tempdir().unwrap();
    // A memory budget that fits one 100-byte blob, and ample disk.
    let cache = DiskCache::with_budgets(dir.path(), 150, 10_000).unwrap();

    let a = CacheKey::new("ns", "ns/gen-1/cluster-0.bin", "e1");
    let b = CacheKey::new("ns", "ns/gen-1/cluster-1.bin", "e2");
    cache.put(a.clone(), bytes::Bytes::from(vec![0u8; 100]));
    cache.put(b.clone(), bytes::Bytes::from(vec![1u8; 100]));

    let entries = cache.entries(None, 100).0;
    assert_eq!(entries.len(), 2, "both objects are cached somewhere");

    // `a` was pushed out of memory by `b`, but its bytes remain on disk.
    let ea = entries.iter().find(|e| e.path.ends_with("cluster-0.bin")).unwrap();
    let eb = entries.iter().find(|e| e.path.ends_with("cluster-1.bin")).unwrap();
    assert!(eb.in_memory, "the most recent object stays in memory");
    assert!(eb.on_disk, "and is resident on disk too — the tiers overlap, not partition");
    assert!(ea.on_disk, "the evicted object is demoted to disk, not dropped");
    assert!(!ea.in_memory);
    assert_eq!(ea.bytes, 100);

    // Tier counts overlap, so summing them would double-count.
    assert!(
        cache.len() + cache.disk_len() >= entries.len(),
        "mem_entries + disk_entries is not an object count"
    );

    // The demoted object is still readable — the demotion cost latency, not the bytes.
    assert!(cache.get(&a).is_some(), "the demoted object is still readable");

    // A disk-tier hit PROMOTES the object back into memory — once per key (until re-evicted), not
    // per hit — so the next read is served from RAM under the shared lock instead of re-opening +
    // mmapping the file on every lookup. That per-lookup mmap is what serialised concurrent readers
    // (near-zero CPU, multi-second latency); promotion is the fix (see `cache.rs::get`). With a
    // one-blob memory budget, promoting `a` evicts `b` from memory — its bytes stay on disk.
    let after = cache.entries(None, 100).0;
    let ea2 = after.iter().find(|e| e.path.ends_with("cluster-0.bin")).unwrap();
    assert!(ea2.in_memory, "a disk-tier hit promotes the object back into memory");
    assert!(ea2.on_disk);
    let eb2 = after.iter().find(|e| e.path.ends_with("cluster-1.bin")).unwrap();
    assert!(!eb2.in_memory, "promoting `a` evicted `b` from the one-blob memory tier");
    assert!(eb2.on_disk, "but `b`'s bytes remain on disk");

    // The just-promoted object is the newest admission, so it now leads the ring listing.
    assert!(after[0].path.ends_with("cluster-0.bin"), "the promoted object is newest, leads");
}

/// The cache is read-through: `Store::put` writes the object and leaves it out of the
/// cache, so a replica that folds a generation inline then pays a roundtrip to read back
/// bytes it had in hand. `put_admitting` is the opt-in that closes that — opt-in rather
/// than the default, because a fold writes far more than any imminent query probes and its
/// admissions arrive with the CLOCK reference bit clear, so they would lap the ring.
#[tokio::test]
async fn put_admitting_populates_the_read_cache_and_plain_put_does_not() {
    use mlake_store::{DiskCache, StoreMetrics};

    let dir = tempfile::tempdir().unwrap();
    let cache = Arc::new(DiskCache::with_budgets(dir.path(), 1 << 20, 1 << 20).unwrap());
    let metrics = StoreMetrics::new();
    let store = Store::in_memory()
        .with_cache(cache.clone())
        .with_store_metrics(metrics.clone());

    store.put("ns/gen-1/cluster-0.bin", vec![1u8; 64]).await.unwrap();
    let before = metrics.gets();
    let got = store.get_immutable("ns/gen-1/cluster-0.bin", None).await.unwrap();
    assert_eq!(got, bytes::Bytes::from(vec![1u8; 64]));
    assert_eq!(metrics.gets(), before + 1, "a plain put leaves the first read to miss");

    store.put_admitting("ns/gen-1/cluster-1.bin", vec![2u8; 64]).await.unwrap();
    let before = metrics.gets();
    let got = store.get_immutable("ns/gen-1/cluster-1.bin", None).await.unwrap();
    assert_eq!(got, bytes::Bytes::from(vec![2u8; 64]));
    assert_eq!(metrics.gets(), before, "an admitting put serves the read with no roundtrip");
    assert!(cache.hits() >= 1);
}

/// `limit` bounds the listing, and lookups are counted for the hit ratio.
#[tokio::test]
async fn cache_listing_is_bounded_and_counts_lookups() {
    use mlake_store::{CacheKey, DiskCache};

    let dir = tempfile::tempdir().unwrap();
    let cache = DiskCache::with_budgets(dir.path(), 10_000, 100_000).unwrap();
    for i in 0..20 {
        cache.put(
            CacheKey::new("ns", &format!("ns/gen-1/cluster-{i}.bin"), "e"),
            bytes::Bytes::from(vec![0u8; 50]),
        );
    }
    assert_eq!(cache.entries(None, 5).0.len(), 5, "the limit bounds the listing");
    assert_eq!(cache.entries(None, 1000).0.len(), 20, "everything else is still there");

    assert!(cache.get(&CacheKey::new("ns", "ns/gen-1/cluster-0.bin", "e")).is_some());
    assert!(cache.get(&CacheKey::new("ns", "nope", "e")).is_none());
    assert!(cache.hits() >= 1 && cache.misses() >= 1, "lookups are counted");
}
