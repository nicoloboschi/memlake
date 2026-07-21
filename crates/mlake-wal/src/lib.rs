//! The write-ahead log: the namespace's ordering and durability primitive.
//!
//! The WAL is a sequence of immutable objects at `{ns}/wal/{seq:08}.bin`. A sequence
//! number is claimed with a conditional create, so ordering emerges from S3 itself with
//! no lock service and no consensus (INV-1, INV-3).
//!
//! Everything a client writes lands here first. The indexer later folds a slice of the
//! log into a generation, but queries never wait for that: a strongly-consistent read
//! scans the entries past the manifest's cursor and merges them over the indexed data,
//! which is what makes an acked write immediately visible (INV-5).

pub mod commit;
pub mod tail;

pub use commit::{CommitResult, Writer};
pub use tail::{TailScan, WalTail};

use mlake_core::manifest::{index_lease_path, manifest_path, wal_path};
use mlake_core::{Manifest, WalEntry};
use mlake_store::{Error as StoreError, Etag, Store};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("store: {0}")]
    Store(#[from] StoreError),
    #[error("core: {0}")]
    Core(#[from] mlake_core::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("gave up claiming a WAL sequence after {0} attempts")]
    CommitRetriesExhausted(usize),
    #[error("guard failed: entry required WAL head < {expected}, but head was {actual}")]
    GuardFailed { expected: u64, actual: u64 },
    #[error("namespace {0} has no manifest")]
    NoManifest(String),
}

impl Error {
    /// True when the failure was a lost CAS race (a create or swap conflict), which the
    /// caller should treat as "someone else got there first" rather than a hard error.
    pub fn is_conflict(&self) -> bool {
        matches!(self, Error::Store(e) if e.is_conflict())
    }
}

/// A namespace handle: the manifest plus the WAL that extends it.
#[derive(Clone)]
pub struct Namespace {
    pub name: String,
    pub store: Store,
}

impl Namespace {
    pub fn new(name: impl Into<String>, store: Store) -> Self {
        Self {
            name: name.into(),
            store,
        }
    }

    /// Create the namespace if it does not exist. Safe to call concurrently from any
    /// number of nodes: the loser of the race simply finds the manifest already there.
    pub async fn create_if_absent(&self, tokenizer_config_hash: &str) -> Result<Manifest> {
        let path = manifest_path(&self.name);
        let manifest = Manifest::empty(tokenizer_config_hash);
        match self.store.put_if_absent(&path, manifest.to_bytes()?).await {
            Ok(_) => Ok(manifest),
            Err(e) if e.is_conflict() => Ok(self.read_manifest().await?.0),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete every object under this namespace's prefix — manifest, WAL, all generations —
    /// dropping the namespace entirely. Returns the number of objects removed. Irreversible.
    ///
    /// The manifest is deleted *first*, and that is what fences a concurrent indexer. A fold
    /// in flight read the manifest at some etag and publishes with a CAS swap conditional on
    /// it (INV-2); once the manifest is gone, that swap fails its `If-Match` precondition and
    /// the fold cannot publish. So after this returns the namespace is either absent (readers
    /// get `NoManifest`) or freshly recreated with its own manifest — never a *surviving*
    /// manifest that points at generation files this sweep has already deleted, which is the
    /// dangling state that made a later `Stats`/query error.
    ///
    /// This does not make the drop transactional: a fold that lands generation files after
    /// the sweep leaves them orphaned (unreferenced, GC-reclaimable), and a query that read
    /// the manifest just before the delete may still fault on a since-deleted file. Dropping
    /// a namespace under active writes is therefore still discouraged — the fence only
    /// guarantees the drop cannot leave the namespace in a corrupt, dangling-manifest state.
    pub async fn delete_all(&self) -> Result<usize> {
        use futures::stream::{StreamExt, TryStreamExt};
        // Fence the indexer before touching anything else (see the doc comment): with the
        // manifest gone, no in-flight fold can CAS-publish a generation over what we sweep.
        let manifest = manifest_path(&self.name);
        self.store.delete(&manifest).await?;

        // The trailing slash keeps a sibling namespace whose name is a prefix (e.g. `foo` vs
        // `foo-bar`) from being swept in. The manifest deleted above no longer appears here.
        let prefix = format!("{}/", self.name);
        let paths = self.store.list(&prefix).await?;
        let count = paths.len() + 1; // + the manifest deleted above
        let store = &self.store;
        futures::stream::iter(paths)
            .map(|p| async move { store.delete(&p).await })
            .buffer_unordered(32)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(count)
    }

    /// Best-effort attempt to become the folder for this namespace, so peer indexers skip it.
    ///
    /// Returns `true` if this holder may fold (the lease was free, expired, or a storage error
    /// left us unsure), `false` only when a *live* lease is held by someone else. It FAILS
    /// OPEN by design: a wasted duplicate fold is safe (nonce'd generation prefixes, INV-6),
    /// while wrongly skipping a needed fold would stall indexing — so every ambiguous outcome
    /// resolves to "go ahead and fold". `ttl_secs` should comfortably exceed one fold; if a
    /// holder crashes, the lease is stealable once it expires.
    ///
    /// Pair every `true` with [`release_index_lease`] when the fold finishes.
    pub async fn acquire_index_lease(&self, holder: &str, ttl_secs: u64) -> bool {
        let now = now_secs();
        let path = index_lease_path(&self.name);
        let bytes = lease_bytes(holder, now + ttl_secs);
        match self.store.put_if_absent(&path, bytes.clone()).await {
            Ok(_) => true, // acquired a free lease
            Err(e) if e.is_conflict() => {
                // A lease exists. Read it: skip only if it is live and someone else's.
                let Ok(cur) = self.store.get(&path, None).await else {
                    return true; // can't read it — fail open
                };
                match parse_lease(&cur.bytes) {
                    Some((cur_holder, expires_at)) if expires_at > now && cur_holder != holder => {
                        false // a peer is actively folding — skip
                    }
                    // Expired, ours, or unparseable: try to (re)claim it with a CAS so two
                    // stealers don't both proceed. Whoever wins the swap folds; the loser skips.
                    _ => match cur.etag {
                        Some(etag) => match self.store.cas_swap(&path, &etag, bytes).await {
                            Ok(_) => true,
                            Err(e) if e.is_conflict() => false, // lost the steal race
                            Err(_) => true,                     // storage error — fail open
                        },
                        None => true, // no etag to CAS against — fail open
                    },
                }
            }
            Err(_) => true, // storage error acquiring — fail open
        }
    }

    /// Release a lease acquired by `holder`, so a fresh fold need not wait out the TTL.
    /// Best-effort and holder-guarded: only deletes the lease if `holder` still owns it, so a
    /// peer that stole an expired lease is not disturbed. Any error is ignored — the lease
    /// will simply expire on its own.
    pub async fn release_index_lease(&self, holder: &str) {
        let path = index_lease_path(&self.name);
        if let Ok(cur) = self.store.get(&path, None).await {
            if let Some((cur_holder, _)) = parse_lease(&cur.bytes) {
                if cur_holder == holder {
                    let _ = self.store.delete(&path).await;
                }
            }
        }
    }

    /// Read the manifest together with its etag, which a later swap must present.
    pub async fn read_manifest(&self) -> Result<(Manifest, Option<Etag>)> {
        let path = manifest_path(&self.name);
        let versioned = self.store.get(&path, None).await.map_err(|e| match e {
            StoreError::NotFound(_) => Error::NoManifest(self.name.clone()),
            other => Error::Store(other),
        })?;
        Ok((Manifest::from_bytes(&versioned.bytes)?, versioned.etag))
    }

    /// Publish a new manifest, conditional on nobody else having published since it was
    /// read. Callers that lose must re-read and re-derive rather than retry blindly,
    /// which is why this returns the conflict instead of looping internally (SPEC §3.1).
    pub async fn swap_manifest(&self, expected: &Etag, next: &Manifest) -> Result<Etag> {
        let path = manifest_path(&self.name);
        let etag = self
            .store
            .cas_swap(&path, expected, next.to_bytes()?)
            .await?;
        Ok(etag.unwrap_or_else(|| Etag(String::new())))
    }

    /// The highest committed WAL sequence, discovered by listing.
    ///
    /// WAL keys are zero-padded, so lexicographic order is sequence order and the head is
    /// simply the last key. Returns 0 for an empty log.
    pub async fn wal_head(&self) -> Result<u64> {
        let prefix = format!("{}/wal/", self.name);
        let paths = self.store.list(&prefix).await?;
        Ok(paths.last().and_then(|p| parse_wal_seq(p)).unwrap_or(0))
    }

    /// The WAL objects currently retained, ascending by sequence, starting at `start_seq`.
    /// Returns the page and the sequence to resume from, or `None` when exhausted.
    ///
    /// This is a *window*, not a history: once the indexer folds an entry, GC is free to
    /// reclaim it, so the oldest sequence here is generally well above 0. One LIST answers
    /// it, sizes included — nothing is decoded.
    pub async fn list_wal(
        &self,
        start_seq: u64,
        limit: usize,
    ) -> Result<(Vec<WalObject>, Option<u64>)> {
        let prefix = format!("{}/wal/", self.name);
        let mut objects: Vec<WalObject> = self
            .store
            .list_with_size(&prefix)
            .await?
            .into_iter()
            .filter_map(|(path, size_bytes)| {
                parse_wal_seq(&path).map(|seq| WalObject { seq, size_bytes })
            })
            .filter(|o| o.seq >= start_seq)
            .collect();
        objects.sort_by_key(|o| o.seq);

        let next = objects.get(limit).map(|o| o.seq);
        objects.truncate(limit);
        Ok((objects, next))
    }

    /// Read and decode one WAL entry. `Err(NotFound)` if GC has already reclaimed it.
    pub async fn read_wal_entry(&self, seq: u64) -> Result<WalEntry> {
        let bytes = self.store.get(&seq_path(&self.name, seq), None).await?;
        Ok(WalEntry::from_bytes(&bytes.bytes)?)
    }
}

/// A WAL object as the log view sees it: its sequence and stored size, undecoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalObject {
    pub seq: u64,
    pub size_bytes: u64,
}

/// Extract the sequence number from a WAL object key.
pub fn parse_wal_seq(path: &str) -> Option<u64> {
    path.rsplit('/')
        .next()?
        .strip_suffix(".bin")?
        .parse::<u64>()
        .ok()
}

/// Wall-clock seconds since the Unix epoch, for lease expiry. A clock skew between nodes only
/// changes how eagerly an expired lease is stolen — never correctness — so a plain wall clock
/// is fine here.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Serialize an index lease. Hand-rolled JSON (two scalar fields) so `mlake-wal` needs no
/// `serde` derive dependency.
fn lease_bytes(holder: &str, expires_at: u64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({ "holder": holder, "expires_at": expires_at }))
        .unwrap_or_default()
}

/// Parse `(holder, expires_at)` from a lease object, or `None` if it is malformed.
fn parse_lease(bytes: &[u8]) -> Option<(String, u64)> {
    let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let holder = v.get("holder")?.as_str()?.to_string();
    let expires_at = v.get("expires_at")?.as_u64()?;
    Some((holder, expires_at))
}

/// Object key for a WAL sequence in a namespace.
pub fn seq_path(namespace: &str, seq: u64) -> String {
    wal_path(namespace, seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sequence_from_key() {
        assert_eq!(parse_wal_seq("ns/wal/00000042.bin"), Some(42));
        assert_eq!(parse_wal_seq("ns/wal/00000000.bin"), Some(0));
        assert_eq!(parse_wal_seq("ns/wal/garbage"), None);
        assert_eq!(parse_wal_seq("ns/manifest.json"), None);
    }

    #[tokio::test]
    async fn create_if_absent_is_idempotent() {
        let ns = Namespace::new("ns", Store::in_memory());
        let first = ns.create_if_absent("tok").await.unwrap();
        let second = ns.create_if_absent("tok").await.unwrap();
        assert_eq!(first, second);
        assert!(first.is_empty());
    }

    #[tokio::test]
    async fn wal_head_is_zero_for_an_empty_log() {
        let ns = Namespace::new("ns", Store::in_memory());
        ns.create_if_absent("tok").await.unwrap();
        assert_eq!(ns.wal_head().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn wal_head_finds_the_highest_sequence_not_the_last_written() {
        let ns = Namespace::new("ns", Store::in_memory());
        ns.create_if_absent("tok").await.unwrap();
        // Write out of order; zero-padding must still order these correctly.
        for seq in [3u64, 1, 12, 2] {
            ns.store
                .put_if_absent(&seq_path("ns", seq), b"x".to_vec())
                .await
                .unwrap();
        }
        assert_eq!(ns.wal_head().await.unwrap(), 12);
    }

    #[tokio::test]
    async fn missing_manifest_is_reported_as_such() {
        let ns = Namespace::new("absent", Store::in_memory());
        assert!(matches!(
            ns.read_manifest().await,
            Err(Error::NoManifest(_))
        ));
    }
}
