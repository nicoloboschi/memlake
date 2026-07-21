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

use mlake_core::manifest::{manifest_path, wal_path};
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
    /// dropping the namespace entirely. Returns the number of objects removed. Irreversible
    /// and not atomic: a concurrent write can recreate objects, so callers must not delete a
    /// namespace that is being written.
    pub async fn delete_all(&self) -> Result<usize> {
        use futures::stream::{StreamExt, TryStreamExt};
        // The trailing slash keeps a sibling namespace whose name is a prefix (e.g. `foo` vs
        // `foo-bar`) from being swept in.
        let prefix = format!("{}/", self.name);
        let paths = self.store.list(&prefix).await?;
        let count = paths.len();
        let store = &self.store;
        futures::stream::iter(paths)
            .map(|p| async move { store.delete(&p).await })
            .buffer_unordered(32)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(count)
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
