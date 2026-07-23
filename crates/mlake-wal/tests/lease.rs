//! The index lease (best-effort fold dedup across nodes).
//!
//! The lease is an optimization, never a correctness gate — the nonce'd generation prefixes
//! already make concurrent folds safe. So these pin down exactly the property the deployment
//! relies on: a live lease held by one holder makes a *different* holder skip, while a free,
//! expired, or self-owned lease lets a holder proceed. It must never wrongly block a fold.

use mlake_store::Store;
use mlake_wal::Namespace;

async fn namespace(name: &str) -> Namespace {
    let ns = Namespace::new(name, Store::in_memory());
    ns.create_if_absent("tok-hash", &[]).await.unwrap();
    ns
}

#[tokio::test]
async fn free_lease_is_acquired() {
    let ns = namespace("ns").await;
    assert!(ns.acquire_index_lease("node-a", 60).await, "a free lease must be acquired");
}

#[tokio::test]
async fn live_lease_makes_a_peer_skip() {
    let ns = namespace("ns").await;
    assert!(ns.acquire_index_lease("node-a", 60).await);
    // A different node must not fold while node-a's lease is live.
    assert!(
        !ns.acquire_index_lease("node-b", 60).await,
        "a live peer lease must make node-b skip"
    );
}

#[tokio::test]
async fn holder_can_reacquire_its_own_live_lease() {
    let ns = namespace("ns").await;
    assert!(ns.acquire_index_lease("node-a", 60).await);
    // The same holder folding again (next tick) must not be blocked by its own lease.
    assert!(
        ns.acquire_index_lease("node-a", 60).await,
        "a holder must be able to re-acquire its own lease"
    );
}

#[tokio::test]
async fn expired_lease_is_stealable() {
    let ns = namespace("ns").await;
    // ttl = 0: the lease is already expired the instant it is written.
    assert!(ns.acquire_index_lease("node-a", 0).await);
    assert!(
        ns.acquire_index_lease("node-b", 60).await,
        "an expired lease must be stealable by a peer"
    );
}

#[tokio::test]
async fn release_frees_the_lease_for_a_peer() {
    let ns = namespace("ns").await;
    assert!(ns.acquire_index_lease("node-a", 60).await);
    ns.release_index_lease("node-a").await;
    assert!(
        ns.acquire_index_lease("node-b", 60).await,
        "after release a peer must be able to acquire immediately (no TTL wait)"
    );
}

#[tokio::test]
async fn release_by_a_non_holder_is_a_noop() {
    let ns = namespace("ns").await;
    assert!(ns.acquire_index_lease("node-a", 60).await);
    // A stray release from someone who does not hold the lease must not free node-a's lease.
    ns.release_index_lease("node-b").await;
    assert!(
        !ns.acquire_index_lease("node-c", 60).await,
        "a non-holder's release must not disturb the live lease"
    );
}

#[tokio::test]
async fn fails_open_when_lease_is_unparseable() {
    // A corrupt lease object must not wedge indexing: acquisition fails open.
    let store = Store::in_memory();
    let ns = Namespace::new("ns", store.clone());
    ns.create_if_absent("tok-hash", &[]).await.unwrap();
    store
        .put_if_absent("ns/index-lease.json", b"not json".to_vec())
        .await
        .unwrap();
    assert!(
        ns.acquire_index_lease("node-a", 60).await,
        "an unparseable lease must fail open (steal), never block folding"
    );
}
