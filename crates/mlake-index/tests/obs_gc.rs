//! Observability retention: whole trace hour-buckets expire by BUCKET NAME, and rollups of nodes
//! that stopped publishing are reaped. Both matter because `_obs/` is append-only — nothing else
//! bounds its growth.

use std::time::Duration;

use mlake_store::Store;

fn hour_bucket_now() -> String {
    chrono::Utc::now().format("%Y%m%d%H").to_string()
}

#[tokio::test]
async fn expires_old_hour_buckets_and_keeps_live_ones() {
    let store = Store::in_memory();
    let current = hour_bucket_now();

    // Two nodes, each with one long-expired bucket and one in the current hour.
    for node in ["serve-0", "serve-1"] {
        store
            .put(
                &format!("_obs/traces/{node}/2020010100/0000000000001-000000000000.jsonl"),
                b"{}".to_vec(),
            )
            .await
            .unwrap();
        store
            .put(
                &format!("_obs/traces/{node}/2020010101/0000000000002-000000000000.jsonl"),
                b"{}".to_vec(),
            )
            .await
            .unwrap();
        store
            .put(
                &format!("_obs/traces/{node}/{current}/9999999999999-000000000000.jsonl"),
                b"{}".to_vec(),
            )
            .await
            .unwrap();
    }

    // 24h retention: the 2020 buckets are expired, the current-hour bucket is not. Rollup staleness
    // is irrelevant here (none exist).
    let out = mlake_index::gc_traces(&store, Duration::from_secs(24 * 60 * 60), Duration::from_secs(3600))
        .await
        .unwrap();

    assert_eq!(out.hour_buckets_deleted, 4, "two expired buckets per node");
    assert_eq!(out.trace_objects_deleted, 4, "one object in each expired bucket");

    let left = store.list("_obs/traces/").await.unwrap();
    assert_eq!(left.len(), 2, "only the current-hour object per node survives: {left:?}");
    assert!(left.iter().all(|p| p.contains(&current)), "survivors are current-hour: {left:?}");

    // Idempotent: a second sweep finds nothing left to expire.
    let again = mlake_index::gc_traces(&store, Duration::from_secs(24 * 60 * 60), Duration::from_secs(3600))
        .await
        .unwrap();
    assert_eq!(again.hour_buckets_deleted, 0);
    assert_eq!(again.trace_objects_deleted, 0);
}

#[tokio::test]
async fn reaps_rollups_of_nodes_that_stopped_publishing() {
    let store = Store::in_memory();
    store.put("_obs/rollup/serve-0.json", b"{}".to_vec()).await.unwrap();
    store.put("_obs/rollup/serve-1.json", b"{}".to_vec()).await.unwrap();

    // A generous staleness window keeps freshly-written rollups (live nodes rewrite every flush).
    let kept = mlake_index::gc_traces(&store, Duration::from_secs(3600), Duration::from_secs(3600))
        .await
        .unwrap();
    assert_eq!(kept.rollups_deleted, 0, "live nodes' rollups are kept");
    assert_eq!(store.list("_obs/rollup/").await.unwrap().len(), 2);

    // Zero staleness treats every rollup as silent — the scaled-down/renamed-node case, which would
    // otherwise linger forever as a stale card in the admin.
    let reaped = mlake_index::gc_traces(&store, Duration::from_secs(3600), Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(reaped.rollups_deleted, 2);
    assert!(store.list("_obs/rollup/").await.unwrap().is_empty());
}
