//! Content-addressed local cache.
//!
//! Keyed by `(namespace, path, etag)`. Since every object except the manifest and the WAL
//! head is immutable (INV-2), a key that matches guarantees identical bytes — so a hit
//! never needs revalidation, and deleting the cache directory can only change latency,
//! never a query result (INV-4).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use bytes::Bytes;

use crate::Result;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CacheKey {
    pub namespace: String,
    pub path: String,
    pub etag: String,
}

impl CacheKey {
    pub fn new(namespace: &str, path: &str, etag: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            path: path.to_string(),
            etag: etag.to_string(),
        }
    }

    /// Filename for this key. The etag is part of the name, so a new version of an object
    /// lands beside the old one rather than overwriting it — no torn reads for a
    /// concurrent reader still using the previous generation.
    fn filename(&self) -> String {
        let mut hash: u64 = 0xcbf29ce484222325;
        for part in [&self.namespace, &self.path, &self.etag] {
            for byte in part.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash ^= 0xff;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{hash:016x}.blob")
    }
}

struct Entry {
    bytes: Bytes,
    /// Monotonic tick of last use, for LRU eviction.
    last_used: u64,
}

/// An LRU byte cache with a local-disk spill directory.
///
/// The in-memory map is the hot tier (hotcache footers, centroids, sparse indexes — all
/// small); the directory is the NVMe tier that survives process restart.
pub struct DiskCache {
    dir: PathBuf,
    state: Mutex<CacheState>,
    capacity_bytes: u64,
}

struct CacheState {
    entries: HashMap<CacheKey, Entry>,
    bytes: u64,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl DiskCache {
    pub fn new(dir: impl Into<PathBuf>, capacity_bytes: u64) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            capacity_bytes,
            state: Mutex::new(CacheState {
                entries: HashMap::new(),
                bytes: 0,
                tick: 0,
                hits: 0,
                misses: 0,
            }),
        })
    }

    pub fn get(&self, key: &CacheKey) -> Option<Bytes> {
        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;
        if let Some(entry) = state.entries.get_mut(key) {
            entry.last_used = tick;
            let bytes = entry.bytes.clone();
            state.hits += 1;
            return Some(bytes);
        }

        // Not in memory: the disk tier may still have it from a previous process.
        let path = self.dir.join(key.filename());
        match std::fs::read(&path) {
            Ok(data) => {
                let bytes = Bytes::from(data);
                state.hits += 1;
                let len = bytes.len() as u64;
                state.entries.insert(
                    key.clone(),
                    Entry {
                        bytes: bytes.clone(),
                        last_used: tick,
                    },
                );
                state.bytes += len;
                Self::evict_to_capacity(&mut state, self.capacity_bytes, &self.dir);
                Some(bytes)
            }
            Err(_) => {
                state.misses += 1;
                None
            }
        }
    }

    /// Write-through admission on first fetch.
    pub fn put(&self, key: CacheKey, bytes: Bytes) {
        let path = self.dir.join(key.filename());
        // A failed disk write is not an error: the cache is advisory, and dropping to
        // memory-only degrades latency rather than correctness.
        if let Err(e) = std::fs::write(&path, &bytes) {
            tracing::debug!(?path, error = %e, "cache disk write failed; keeping in memory only");
        }
        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;
        let len = bytes.len() as u64;
        if let Some(old) = state.entries.insert(
            key,
            Entry {
                bytes,
                last_used: tick,
            },
        ) {
            state.bytes -= old.bytes.len() as u64;
        }
        state.bytes += len;
        Self::evict_to_capacity(&mut state, self.capacity_bytes, &self.dir);
    }

    fn evict_to_capacity(state: &mut CacheState, capacity: u64, dir: &PathBuf) {
        while state.bytes > capacity && !state.entries.is_empty() {
            let victim = state
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else { break };
            let _ = std::fs::remove_file(dir.join(victim.filename()));
            if let Some(entry) = state.entries.remove(&victim) {
                state.bytes -= entry.bytes.len() as u64;
            }
        }
    }

    pub fn bytes(&self) -> u64 {
        self.state.lock().unwrap().bytes
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hit ratio across the cache's lifetime, or `None` before any lookup.
    pub fn hit_ratio(&self) -> Option<f64> {
        let state = self.state.lock().unwrap();
        let total = state.hits + state.misses;
        (total > 0).then(|| state.hits as f64 / total as f64)
    }

    /// Drop everything, memory and disk. Used by the cold-start tests, which assert that
    /// wiping the cache changes only latency.
    pub fn wipe(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.entries.clear();
        state.bytes = 0;
        for entry in std::fs::read_dir(&self.dir)? {
            let _ = std::fs::remove_file(entry?.path());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(cap: u64) -> (DiskCache, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let c = DiskCache::new(dir.path(), cap).unwrap();
        (c, dir)
    }

    #[test]
    fn stores_and_returns_bytes() {
        let (c, _d) = cache(1024);
        let key = CacheKey::new("ns", "gen-1/pk.idx", "etag-a");
        assert!(c.get(&key).is_none());
        c.put(key.clone(), Bytes::from_static(b"hello"));
        assert_eq!(c.get(&key).unwrap(), Bytes::from_static(b"hello"));
    }

    #[test]
    fn a_new_etag_is_a_different_entry() {
        // The manifest swap gives an object a new etag; the old cached bytes must remain
        // reachable for readers still on the previous generation.
        let (c, _d) = cache(1024);
        let old = CacheKey::new("ns", "manifest.json", "etag-a");
        let new = CacheKey::new("ns", "manifest.json", "etag-b");
        c.put(old.clone(), Bytes::from_static(b"gen-1"));
        c.put(new.clone(), Bytes::from_static(b"gen-2"));
        assert_eq!(c.get(&old).unwrap(), Bytes::from_static(b"gen-1"));
        assert_eq!(c.get(&new).unwrap(), Bytes::from_static(b"gen-2"));
    }

    #[test]
    fn namespaces_do_not_collide() {
        let (c, _d) = cache(1024);
        let a = CacheKey::new("ns-a", "pk.idx", "e");
        let b = CacheKey::new("ns-b", "pk.idx", "e");
        c.put(a.clone(), Bytes::from_static(b"a"));
        c.put(b.clone(), Bytes::from_static(b"b"));
        assert_eq!(c.get(&a).unwrap(), Bytes::from_static(b"a"));
        assert_eq!(c.get(&b).unwrap(), Bytes::from_static(b"b"));
    }

    #[test]
    fn evicts_least_recently_used_when_over_capacity() {
        let (c, _d) = cache(10);
        let a = CacheKey::new("ns", "a", "e");
        let b = CacheKey::new("ns", "b", "e");
        c.put(a.clone(), Bytes::from_static(b"aaaa"));
        c.put(b.clone(), Bytes::from_static(b"bbbb"));
        // Touch `a` so `b` becomes the eviction victim.
        assert!(c.get(&a).is_some());
        let e = CacheKey::new("ns", "e", "e");
        c.put(e.clone(), Bytes::from_static(b"eeee"));
        assert!(c.bytes() <= 10);
        assert!(c.get(&a).is_some(), "recently used entry should survive");
        assert!(c.get(&e).is_some(), "newest entry should survive");
    }

    #[test]
    fn survives_process_restart_via_disk_tier() {
        let dir = tempfile::tempdir().unwrap();
        let key = CacheKey::new("ns", "clusters/0.bin", "etag-1");
        {
            let c = DiskCache::new(dir.path(), 1024).unwrap();
            c.put(key.clone(), Bytes::from_static(b"payload"));
        }
        let reopened = DiskCache::new(dir.path(), 1024).unwrap();
        assert_eq!(reopened.get(&key).unwrap(), Bytes::from_static(b"payload"));
    }

    #[test]
    fn wipe_clears_both_tiers() {
        let (c, _d) = cache(1024);
        let key = CacheKey::new("ns", "a", "e");
        c.put(key.clone(), Bytes::from_static(b"x"));
        c.wipe().unwrap();
        assert!(c.get(&key).is_none());
        assert_eq!(c.bytes(), 0);
    }
}
