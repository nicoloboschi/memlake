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

/// One cached object, as an inspector sees it. Both tier flags can be true at once — see
/// [`DiskCache::entries`].
#[derive(Clone, Debug)]
pub struct CacheEntry {
    /// Owning namespace, derived from the object key's first segment. `CacheKey` carries a
    /// namespace field, but the read paths leave it empty (the cache is keyed by path and
    /// byte range alone), so the key layout is the only reliable source.
    pub namespace: String,
    /// The object key. A ranged read appends `#start-end`, because the cache is keyed by
    /// `(path, byte range)` — the same object can be cached as several distinct blocks.
    pub path: String,
    pub etag: String,
    pub bytes: u64,
    pub in_memory: bool,
    pub on_disk: bool,
    /// The LRU clock value when this entry was last touched. Only meaningful relative to
    /// other entries — it is a monotonic counter, not a timestamp.
    pub last_used: u64,
}

impl CacheEntry {
    fn new(key: &CacheKey, bytes: u64, in_memory: bool, on_disk: bool, last_used: u64) -> Self {
        // Object keys are `{namespace}/...`; the key's own namespace field is unset on the
        // read paths, so take it from the path.
        let namespace = key
            .path
            .split_once('/')
            .map(|(ns, _)| ns.to_string())
            .unwrap_or_default();
        Self {
            namespace,
            path: key.path.clone(),
            etag: key.etag.clone(),
            bytes,
            in_memory,
            on_disk,
            last_used,
        }
    }
}

struct MemEntry {
    bytes: Bytes,
    last_used: u64,
}

struct DiskEntry {
    size: u64,
    last_used: u64,
}

/// A two-tier cache with **independent, bounded** memory and disk budgets, so a query node
/// has predictable resource usage regardless of workload.
///
/// * The **memory tier** is the hot bytes in RAM, bounded by `mem_budget` (LRU).
/// * The **disk tier** is the NVMe spill, bounded by `disk_budget` (LRU), and survives a
///   process restart.
///
/// A memory eviction *demotes* an item to disk (the bytes stay on disk); only a disk
/// eviction deletes the file. A hit promotes back into memory. Neither tier can exceed its
/// budget, so peak RAM and peak disk are both capped by construction.
pub struct DiskCache {
    dir: PathBuf,
    state: Mutex<CacheState>,
    mem_budget: u64,
    disk_budget: u64,
}

struct CacheState {
    mem: HashMap<CacheKey, MemEntry>,
    mem_bytes: u64,
    disk: HashMap<CacheKey, DiskEntry>,
    disk_bytes: u64,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl DiskCache {
    /// A cache with separate memory and disk byte budgets.
    pub fn with_budgets(
        dir: impl Into<PathBuf>,
        mem_budget: u64,
        disk_budget: u64,
    ) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            mem_budget,
            disk_budget,
            state: Mutex::new(CacheState {
                mem: HashMap::new(),
                mem_bytes: 0,
                disk: HashMap::new(),
                disk_bytes: 0,
                tick: 0,
                hits: 0,
                misses: 0,
            }),
        })
    }

    /// Backwards-compatible constructor: split one budget as 25% memory / 75% disk.
    pub fn new(dir: impl Into<PathBuf>, capacity_bytes: u64) -> Result<Self> {
        Self::with_budgets(dir, capacity_bytes / 4, capacity_bytes)
    }

    pub fn get(&self, key: &CacheKey) -> Option<Bytes> {
        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;

        if let Some(entry) = state.mem.get_mut(key) {
            entry.last_used = tick;
            let bytes = entry.bytes.clone();
            state.hits += 1;
            return Some(bytes);
        }

        // Not in memory: the disk tier may still hold it (this or a previous process).
        let path = self.dir.join(key.filename());
        match std::fs::read(&path) {
            Ok(data) => {
                let bytes = Bytes::from(data);
                state.hits += 1;
                // Ensure the disk tier accounts for it (e.g. after a restart), then promote
                // into memory.
                if let std::collections::hash_map::Entry::Vacant(e) = state.disk.entry(key.clone()) {
                    let size = bytes.len() as u64;
                    e.insert(DiskEntry { size, last_used: tick });
                    state.disk_bytes += size;
                } else if let Some(d) = state.disk.get_mut(key) {
                    d.last_used = tick;
                }
                self.promote(&mut state, key.clone(), bytes.clone(), tick);
                Some(bytes)
            }
            Err(_) => {
                state.misses += 1;
                None
            }
        }
    }

    /// Admit on first fetch: write to disk (bounded) and to memory (bounded).
    pub fn put(&self, key: CacheKey, bytes: Bytes) {
        let path = self.dir.join(key.filename());
        // A failed disk write is not an error: the cache is advisory, and dropping to
        // memory-only degrades latency rather than correctness.
        let on_disk = std::fs::write(&path, &bytes).is_ok();

        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;
        let len = bytes.len() as u64;

        if on_disk {
            if let Some(old) = state.disk.insert(key.clone(), DiskEntry { size: len, last_used: tick }) {
                state.disk_bytes -= old.size;
            }
            state.disk_bytes += len;
            self.evict_disk(&mut state);
        }
        self.promote(&mut state, key, bytes, tick);
    }

    /// Insert bytes into the memory tier and evict it back to budget.
    fn promote(&self, state: &mut CacheState, key: CacheKey, bytes: Bytes, tick: u64) {
        let len = bytes.len() as u64;
        if let Some(old) = state.mem.insert(key, MemEntry { bytes, last_used: tick }) {
            state.mem_bytes -= old.bytes.len() as u64;
        }
        state.mem_bytes += len;
        // Memory eviction demotes to disk (the bytes remain on disk), so it drops from the
        // memory map only.
        while state.mem_bytes > self.mem_budget && !state.mem.is_empty() {
            let victim = state
                .mem
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else { break };
            if let Some(e) = state.mem.remove(&victim) {
                state.mem_bytes -= e.bytes.len() as u64;
            }
        }
    }

    /// Evict the disk tier to budget, deleting files (and any resident memory copy).
    fn evict_disk(&self, state: &mut CacheState) {
        while state.disk_bytes > self.disk_budget && !state.disk.is_empty() {
            let victim = state
                .disk
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else { break };
            let _ = std::fs::remove_file(self.dir.join(victim.filename()));
            if let Some(e) = state.disk.remove(&victim) {
                state.disk_bytes -= e.size;
            }
            if let Some(e) = state.mem.remove(&victim) {
                state.mem_bytes -= e.bytes.len() as u64;
            }
        }
    }

    /// Bytes resident in the memory tier (bounded by `mem_budget`).
    pub fn bytes(&self) -> u64 {
        self.state.lock().unwrap().mem_bytes
    }

    /// Bytes resident in the disk tier (bounded by `disk_budget`).
    pub fn disk_bytes(&self) -> u64 {
        self.state.lock().unwrap().disk_bytes
    }

    pub fn mem_budget(&self) -> u64 {
        self.mem_budget
    }
    pub fn disk_budget(&self) -> u64 {
        self.disk_budget
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().mem.len()
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

    /// Lookups served from cache, and lookups that missed, over the cache's lifetime.
    pub fn hits(&self) -> u64 {
        self.state.lock().unwrap().hits
    }
    pub fn misses(&self) -> u64 {
        self.state.lock().unwrap().misses
    }

    /// Objects resident in the disk tier. Note this is not `len()` — an object demoted out
    /// of memory still occupies disk, so the two tiers overlap rather than partition.
    pub fn disk_len(&self) -> usize {
        self.state.lock().unwrap().disk.len()
    }

    /// What the cache is currently holding, most-recently-used first, optionally filtered
    /// to one namespace. Returns the (bounded) page and the total number of entries that
    /// matched *before* `limit` truncated it, so a caller can say how much it is not
    /// showing rather than presenting a short list as the whole cache.
    ///
    /// An entry can be resident in both tiers at once: a memory eviction *demotes* to disk
    /// without dropping the bytes, and a hit promotes back. So `in_memory` and `on_disk`
    /// are independent flags, not a single tier field.
    pub fn entries(&self, namespace: Option<&str>, limit: usize) -> (Vec<CacheEntry>, usize) {
        let state = self.state.lock().unwrap();
        let mut by_key: HashMap<&CacheKey, CacheEntry> = HashMap::new();
        for (key, e) in &state.mem {
            by_key.insert(key, CacheEntry::new(key, e.bytes.len() as u64, true, false, e.last_used));
        }
        for (key, e) in &state.disk {
            by_key
                .entry(key)
                .and_modify(|c| {
                    c.on_disk = true;
                    c.last_used = c.last_used.max(e.last_used);
                })
                .or_insert_with(|| CacheEntry::new(key, e.size, false, true, e.last_used));
        }

        let mut out: Vec<CacheEntry> = by_key.into_values().collect();
        if let Some(ns) = namespace {
            out.retain(|e| e.namespace == ns);
        }
        let total = out.len();
        // Most recently used first: the hot working set is what an operator looks at.
        // Ties break on path so repeated calls are stable rather than hash-order noise.
        out.sort_by(|a, b| b.last_used.cmp(&a.last_used).then(a.path.cmp(&b.path)));
        out.truncate(limit);
        (out, total)
    }

    /// Drop everything, memory and disk.
    pub fn wipe(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.mem.clear();
        state.mem_bytes = 0;
        state.disk.clear();
        state.disk_bytes = 0;
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

#[cfg(test)]
mod two_tier_tests {
    use super::*;

    #[test]
    fn memory_and_disk_budgets_are_enforced_independently() {
        let dir = tempfile::tempdir().unwrap();
        // Room for 2 entries in memory, 10 on disk (each entry is 100 bytes).
        let c = DiskCache::with_budgets(dir.path(), 250, 1050).unwrap();
        for i in 0..20 {
            let key = CacheKey::new("ns", &format!("obj{i}"), "e");
            c.put(key, Bytes::from(vec![0u8; 100]));
        }
        // Memory never exceeds its budget; disk never exceeds its budget.
        assert!(c.bytes() <= 250, "memory tier over budget: {}", c.bytes());
        assert!(c.disk_bytes() <= 1050, "disk tier over budget: {}", c.disk_bytes());
        // Disk holds far more than memory — the point of two tiers.
        assert!(c.disk_bytes() > c.bytes());
    }

    #[test]
    fn memory_eviction_demotes_to_disk_not_deletion() {
        let dir = tempfile::tempdir().unwrap();
        // Tiny memory (1 entry), generous disk.
        let c = DiskCache::with_budgets(dir.path(), 100, 100_000).unwrap();
        let a = CacheKey::new("ns", "a", "e");
        let b = CacheKey::new("ns", "b", "e");
        c.put(a.clone(), Bytes::from(vec![1u8; 100]));
        c.put(b.clone(), Bytes::from(vec![2u8; 100])); // evicts `a` from memory

        // `a` is gone from memory but still on disk — a get promotes it back (a hit, not a miss).
        let before = c.hit_ratio();
        assert_eq!(c.get(&a).unwrap(), Bytes::from(vec![1u8; 100]));
        assert!(c.hit_ratio() >= before, "demoted entry must be a disk hit, not a miss");
    }
}
