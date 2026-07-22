//! Content-addressed local cache.
//!
//! Keyed by `(namespace, path, etag)`. Since every object except the manifest and the WAL
//! head is immutable (INV-2), a key that matches guarantees identical bytes — so a hit
//! never needs revalidation, and deleting the cache directory can only change latency,
//! never a query result (INV-4).
//!
//! # Why FIFO and not LRU
//!
//! Both tiers are **FIFO rings**: an entry is ordered by when it was *admitted* and nothing
//! reorders it afterwards. LRU would keep a hot working set better, but it has to mutate
//! the entry on every hit, which means every cache *read* takes the write lock — the cache
//! becomes a contention point exactly when concurrent queries make it matter. FIFO buys:
//!
//! * a read only ever takes the lock *shared*, so concurrent queries do not serialise on
//!   it — the one exception is adopting a blob left behind by a previous process, which is
//!   restart recovery and happens at most once per key,
//! * eviction is a pointer bump — pop the oldest slot — instead of a scan for the minimum,
//! * deletes follow the order the writes went in, rather than LRU's scattered pattern —
//!   claimed to suit NVMe better, but unmeasured; the lock is what this rests on.
//!
//! The cost is hit rate under skew, and it is **not** small. Measured against the LRU this
//! replaced, over an IVF-probe-shaped trace (256 clusters, Zipf-skewed, 16 distinct probes
//! per query plus three always-hot blocks — harness in `tests/cache_skew.rs`, full table in
//! `TODOS.md` §"Read path"), FIFO gives up 4–15 points of hit ratio wherever the cache is a
//! small fraction of the corpus:
//!
//! ```text
//! cache/corpus     5%      10%     25%     50%
//! LRU  (s=1.1)   0.0781  0.4491  0.6758  0.8314
//! FIFO (s=1.1)   0.0895  0.3327  0.5937  0.7940
//! ```
//!
//! It is only better in the leftmost column, where the probe working set exceeds the cache
//! and LRU thrashes instead. Most of the loss is one specific effect: the few small objects
//! *every* query reads — centroids, footers, `pk.idx` — are admitted once and then age out,
//! because a hit no longer refreshes them, so cluster traffic laps the ring and evicts them
//! on a cycle. That is why the loss shows up even on a uniform control trace (0.2153 →
//! 0.1379 at 10%), where policy should not matter at all. It makes the small in-RAM tier
//! for those objects (SPEC §6.2's ARC layer, still unbuilt) load-bearing rather than an
//! optimisation.
//!
//! So this is a contention-over-hit-rate trade made with the number in hand, not a free
//! win. If the hit ratio turns out to matter more than the lock does, the next move is
//! CLOCK — one atomic reference bit set on hit (still no reordering, still no write lock)
//! and a second chance on eviction — which recovers most of LRU's hit rate without giving
//! the shared-lock read back.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

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
    /// Position in the ring: the value of the admission counter when this entry was last
    /// *admitted*. Only meaningful relative to other entries — a monotonic counter, not a
    /// timestamp, and under FIFO it is not touched by reads.
    pub admitted: u64,
}

impl CacheEntry {
    fn new(key: &CacheKey, bytes: u64, in_memory: bool, on_disk: bool, admitted: u64) -> Self {
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
            admitted,
        }
    }
}

struct MemEntry {
    bytes: Bytes,
    /// Admission sequence, matched against the ring slot to spot a stale slot.
    seq: u64,
}

struct DiskEntry {
    size: u64,
    seq: u64,
}

/// A two-tier cache with **independent, bounded** memory and disk budgets, so a query node
/// has predictable resource usage regardless of workload.
///
/// * The **memory tier** is the hot bytes in RAM, bounded by `mem_budget`.
/// * The **disk tier** is the NVMe spill, bounded by `disk_budget`, and survives a process
///   restart.
///
/// Both are FIFO rings (see the module docs). A memory eviction *demotes* an item to disk
/// (the bytes stay on disk, and a later read maps them back in); only a disk eviction
/// deletes the file. Neither tier can exceed its budget, so peak RAM and peak disk are both
/// capped by construction.
///
/// Unlike the LRU this replaced, a disk hit does **not** promote back into memory: the ring
/// is admission-ordered, so re-admitting on every hit would be exactly the per-hit mutation
/// FIFO exists to avoid — and since a disk hit is served by an mmap that the page cache
/// already backs, copying it onto the heap as well would cost RAM to buy nothing.
pub struct DiskCache {
    dir: PathBuf,
    state: RwLock<CacheState>,
    mem_budget: u64,
    disk_budget: u64,
    /// Counters live outside the state lock so a hit never needs to take it for writing.
    hits: AtomicU64,
    misses: AtomicU64,
    /// Disambiguates the scratch file two concurrent `put`s of the *same* key would
    /// otherwise both write to.
    tmp_nonce: AtomicU64,
}

struct CacheState {
    mem: HashMap<CacheKey, MemEntry>,
    /// The memory ring: admission order, oldest at the front. May hold stale slots for
    /// keys that were re-admitted or dropped; `seq` identifies them on the way out.
    mem_ring: VecDeque<(CacheKey, u64)>,
    mem_bytes: u64,
    disk: HashMap<CacheKey, DiskEntry>,
    disk_ring: VecDeque<(CacheKey, u64)>,
    disk_bytes: u64,
    next_seq: u64,
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
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            tmp_nonce: AtomicU64::new(0),
            state: RwLock::new(CacheState {
                mem: HashMap::new(),
                mem_ring: VecDeque::new(),
                mem_bytes: 0,
                disk: HashMap::new(),
                disk_ring: VecDeque::new(),
                disk_bytes: 0,
                next_seq: 0,
            }),
        })
    }

    /// Backwards-compatible constructor: split one budget as 25% memory / 75% disk.
    pub fn new(dir: impl Into<PathBuf>, capacity_bytes: u64) -> Result<Self> {
        Self::with_budgets(dir, capacity_bytes / 4, capacity_bytes)
    }

    pub fn get(&self, key: &CacheKey) -> Option<Bytes> {
        // Memory tier under a *shared* lock: nothing is mutated, so concurrent readers all
        // proceed. This is the whole point of the ring.
        if let Some(bytes) = self
            .state
            .read()
            .unwrap()
            .mem
            .get(key)
            .map(|e| e.bytes.clone())
        {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(bytes);
        }

        // Not in memory: the disk tier may still hold it (this or a previous process). The
        // mapping itself is taken with no lock held at all.
        let path = self.dir.join(key.filename());
        let Some(bytes) = read_blob(&path) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.hits.fetch_add(1, Ordering::Relaxed);

        // A file left by a previous process is on disk but not in the ring, so it is
        // unaccounted against the disk budget. Adopting it costs the write lock, but at most
        // once per key per process — this is restart recovery, not per-hit bookkeeping.
        // (The read guard is scoped explicitly: `RwLock` is not reentrant, and taking the
        // write lock below while still holding it would deadlock.)
        let unaccounted = { !self.state.read().unwrap().disk.contains_key(key) };
        if unaccounted {
            let mut state = self.state.write().unwrap();
            let size = bytes.len() as u64;
            let seq = state.next_seq;
            // Vacant-only: another thread may have adopted the same key in between.
            if let Entry::Vacant(slot) = state.disk.entry(key.clone()) {
                slot.insert(DiskEntry { size, seq });
                state.next_seq += 1;
                state.disk_bytes += size;
                state.disk_ring.push_back((key.clone(), seq));
                // Adoption can push the tier over budget, and the eviction that follows may
                // unlink this very file. Safe: `bytes` owns the mapping, and on Unix a
                // mapping keeps the inode alive past its last link.
                self.evict_disk(&mut state);
            }
        }
        Some(bytes)
    }

    /// Admit bytes into both tiers: the tail of the disk ring and the tail of the memory
    /// ring. Called on a read miss, and — opt in — by a writer that already has the bytes.
    pub fn put(&self, key: CacheKey, bytes: Bytes) {
        let path = self.dir.join(key.filename());
        // A failed disk write is not an error: the cache is advisory, and dropping to
        // memory-only degrades latency rather than correctness.
        let on_disk = self.write_blob(&path, &bytes).is_ok();

        let mut state = self.state.write().unwrap();
        let len = bytes.len() as u64;

        // Memory first, then disk: a disk eviction drops the memory copy too (the file is
        // gone, so keeping the tiers' contents nested is the honest bookkeeping), and doing
        // it in this order means that also covers the entry we just admitted if it alone
        // overflows the disk budget.
        let seq = state.next_seq;
        state.next_seq += 1;
        if let Some(old) = state.mem.insert(key.clone(), MemEntry { bytes, seq }) {
            state.mem_bytes -= old.bytes.len() as u64;
        }
        state.mem_bytes += len;
        state.mem_ring.push_back((key.clone(), seq));
        self.evict_mem(&mut state);

        if on_disk {
            if let Some(old) = state.disk.insert(key.clone(), DiskEntry { size: len, seq }) {
                state.disk_bytes -= old.size;
            }
            state.disk_bytes += len;
            state.disk_ring.push_back((key, seq));
            self.evict_disk(&mut state);
        }
    }

    /// Write a blob through a scratch file and rename it into place.
    ///
    /// The rename matters for more than crash-consistency: readers `mmap` these files, and
    /// writing in place would truncate a file another thread is mapping — a SIGBUS, not an
    /// error. Rename swaps the directory entry, so a live mapping keeps reading the old
    /// inode until it is dropped.
    fn write_blob(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        let nonce = self.tmp_nonce.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("tmp{nonce}"));
        match std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    /// Drop the oldest memory slots until the tier is inside its budget. The bytes stay on
    /// disk, so this is a demotion, not a deletion.
    fn evict_mem(&self, state: &mut CacheState) {
        while state.mem_bytes > self.mem_budget {
            let Some((key, seq)) = state.mem_ring.pop_front() else {
                break;
            };
            // A stale slot: the key was re-admitted (a newer slot holds it) or already
            // dropped by a disk eviction. Skipping it is the pointer bump.
            if state.mem.get(&key).map(|e| e.seq) != Some(seq) {
                continue;
            }
            if let Some(e) = state.mem.remove(&key) {
                state.mem_bytes -= e.bytes.len() as u64;
            }
        }
    }

    /// Drop the oldest disk slots until the tier is inside its budget, deleting the file
    /// and any resident memory copy with it.
    fn evict_disk(&self, state: &mut CacheState) {
        while state.disk_bytes > self.disk_budget {
            let Some((key, seq)) = state.disk_ring.pop_front() else {
                break;
            };
            if state.disk.get(&key).map(|e| e.seq) != Some(seq) {
                continue;
            }
            let _ = std::fs::remove_file(self.dir.join(key.filename()));
            if let Some(e) = state.disk.remove(&key) {
                state.disk_bytes -= e.size;
            }
            if let Some(e) = state.mem.remove(&key) {
                state.mem_bytes -= e.bytes.len() as u64;
            }
        }
    }

    /// Bytes resident in the memory tier (bounded by `mem_budget`).
    pub fn bytes(&self) -> u64 {
        self.state.read().unwrap().mem_bytes
    }

    /// Bytes resident in the disk tier (bounded by `disk_budget`).
    pub fn disk_bytes(&self) -> u64 {
        self.state.read().unwrap().disk_bytes
    }

    pub fn mem_budget(&self) -> u64 {
        self.mem_budget
    }
    pub fn disk_budget(&self) -> u64 {
        self.disk_budget
    }

    pub fn len(&self) -> usize {
        self.state.read().unwrap().mem.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hit ratio across the cache's lifetime, or `None` before any lookup.
    pub fn hit_ratio(&self) -> Option<f64> {
        let hits = self.hits();
        let total = hits + self.misses();
        (total > 0).then(|| hits as f64 / total as f64)
    }

    /// Lookups served from cache, and lookups that missed, over the cache's lifetime.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Objects resident in the disk tier. Note this is not `len()` — an object demoted out
    /// of memory still occupies disk, so the two tiers overlap rather than partition.
    pub fn disk_len(&self) -> usize {
        self.state.read().unwrap().disk.len()
    }

    /// What the cache is currently holding, most-recently-*admitted* first, optionally
    /// filtered to one namespace. Returns the (bounded) page and the total number of
    /// entries that matched *before* `limit` truncated it, so a caller can say how much it
    /// is not showing rather than presenting a short list as the whole cache.
    ///
    /// An entry can be resident in both tiers at once: a memory eviction *demotes* to disk
    /// without dropping the bytes. So `in_memory` and `on_disk` are independent flags, not
    /// a single tier field.
    pub fn entries(&self, namespace: Option<&str>, limit: usize) -> (Vec<CacheEntry>, usize) {
        let state = self.state.read().unwrap();
        let mut by_key: HashMap<&CacheKey, CacheEntry> = HashMap::new();
        for (key, e) in &state.mem {
            by_key.insert(key, CacheEntry::new(key, e.bytes.len() as u64, true, false, e.seq));
        }
        for (key, e) in &state.disk {
            by_key
                .entry(key)
                .and_modify(|c| {
                    c.on_disk = true;
                    c.admitted = c.admitted.max(e.seq);
                })
                .or_insert_with(|| CacheEntry::new(key, e.size, false, true, e.seq));
        }

        let mut out: Vec<CacheEntry> = by_key.into_values().collect();
        if let Some(ns) = namespace {
            out.retain(|e| e.namespace == ns);
        }
        let total = out.len();
        // Newest admission first: the head of the ring is what an operator looks at, and it
        // is also what has the longest to live. Ties break on path so repeated calls are
        // stable rather than hash-order noise.
        out.sort_by(|a, b| b.admitted.cmp(&a.admitted).then(a.path.cmp(&b.path)));
        out.truncate(limit);
        (out, total)
    }

    /// Drop everything, memory and disk.
    pub fn wipe(&self) -> Result<()> {
        let mut state = self.state.write().unwrap();
        state.mem.clear();
        state.mem_ring.clear();
        state.mem_bytes = 0;
        state.disk.clear();
        state.disk_ring.clear();
        state.disk_bytes = 0;
        for entry in std::fs::read_dir(&self.dir)? {
            let _ = std::fs::remove_file(entry?.path());
        }
        Ok(())
    }
}

/// Read a cached blob off disk **without copying it**: the file is mapped and the mapping
/// itself becomes the owner behind the returned `Bytes`, so a warm hit costs a page-table
/// entry and some faults into the page cache rather than a heap allocation the size of the
/// blob. On a re-read of a still-resident file, no I/O happens at all.
///
/// This removes *one* copy from the read path, not all of them. The consumers still call
/// `rkyv_read`, which validates the archive and then deserializes it into an owned graph —
/// so the process is not yet zero-copy end to end, and this function is not what makes it
/// so. See `TODOS.md` §"Read path".
///
/// `None` means "not cached" — a missing file, or any I/O error, is a miss rather than a
/// failure (INV-4).
fn read_blob(path: &Path) -> Option<Bytes> {
    let file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    // A zero-length mapping is invalid, and there is nothing to map.
    if len == 0 {
        return Some(Bytes::new());
    }
    // SAFETY: mapping a file is unsafe because the mapped bytes change if the file changes
    // underneath, and a truncation *faults* (SIGBUS) rather than returning an error.
    // Neither can happen to a cache blob:
    //   * blobs are content-addressed and written exactly once;
    //   * a write is published by rename (see `write_blob`), so a concurrent re-`put` of
    //     the same key installs a new inode instead of truncating the mapped one;
    //   * eviction only unlinks, and on Unix an existing mapping keeps the inode alive
    //     until the last reference drops — so evicting a blob mid-query cannot pull the
    //     bytes out from under the query that is reading them.
    // (On Windows unlinking a mapped file fails instead; the cache is advisory, so that
    // degrades to a blob that lingers past its eviction rather than to unsoundness.)
    let mmap = unsafe { memmap2::Mmap::map(&file) }.ok()?;
    // `from_owner` moves the mapping into the `Bytes`; the mapping — and therefore the
    // inode — lives exactly as long as the last clone of what we return. The `File` handle
    // is not needed once the mapping exists and drops here.
    Some(Bytes::from_owner(mmap))
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

    /// The ring evicts by admission order, and a read does not change that order — which is
    /// exactly what lets a read take a shared lock. (Under the LRU this replaced, the `get`
    /// below would have saved `a` and condemned `b`.)
    #[test]
    fn evicts_in_admission_order_and_reads_do_not_reorder() {
        let dir = tempfile::tempdir().unwrap();
        // Disk fits two 4-byte blobs; memory fits one.
        let c = DiskCache::with_budgets(dir.path(), 4, 8).unwrap();
        let a = CacheKey::new("ns", "a", "e");
        let b = CacheKey::new("ns", "b", "e");
        c.put(a.clone(), Bytes::from_static(b"aaaa"));
        c.put(b.clone(), Bytes::from_static(b"bbbb"));
        // Reading `a` does *not* move it to the back of the ring.
        assert!(c.get(&a).is_some());
        let e = CacheKey::new("ns", "e", "e");
        c.put(e.clone(), Bytes::from_static(b"eeee"));

        assert!(c.disk_bytes() <= 8);
        assert!(c.bytes() <= 4);
        assert!(c.get(&a).is_none(), "the oldest admission is the victim");
        assert!(c.get(&b).is_some(), "the next-oldest survives");
        assert!(c.get(&e).is_some(), "the newest survives");
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
        // Adopted into the ring, so it is now accounted against the disk budget.
        assert_eq!(reopened.disk_len(), 1);
        assert_eq!(reopened.disk_bytes(), 7);
    }

    /// A disk hit returns a *mapping*, and the query holding it may still be running when
    /// the entry is evicted. Eviction unlinks the file; on Unix the inode survives until
    /// the last mapping drops, so the bytes stay valid. If that were not true this would be
    /// a use-after-free, silently returning garbage into a query result.
    #[test]
    fn bytes_from_a_disk_hit_outlive_the_evicted_file() {
        let dir = tempfile::tempdir().unwrap();
        // Memory too small for any blob, so every read comes off the disk tier.
        let c = DiskCache::with_budgets(dir.path(), 8, 200).unwrap();
        let a = CacheKey::new("ns", "a", "e");
        c.put(a.clone(), Bytes::from(vec![0xABu8; 100]));
        let held = c.get(&a).unwrap();

        // Push `a` out of the disk ring: its file is unlinked while `held` still maps it.
        for i in 0..10 {
            c.put(
                CacheKey::new("ns", &format!("f{i}"), "e"),
                Bytes::from(vec![0u8; 50]),
            );
        }
        assert!(c.get(&a).is_none(), "the blob really was evicted");
        assert_eq!(held, Bytes::from(vec![0xABu8; 100]), "the mapping outlived the file");
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

    /// Re-admitting a key leaves stale slots behind it in the ring. Eviction walks past
    /// them; treating one as a live entry would double-subtract the byte count and delete
    /// a blob that is still current.
    #[test]
    fn stale_ring_slots_from_re_admission_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let c = DiskCache::with_budgets(dir.path(), 100, 100).unwrap();
        let a = CacheKey::new("ns", "a", "e");
        for _ in 0..5 {
            c.put(a.clone(), Bytes::from(vec![7u8; 40]));
        }
        assert_eq!(c.disk_bytes(), 40, "one live copy, not five");

        // Four stale `a` slots now sit ahead of its live one.
        let b = CacheKey::new("ns", "b", "e");
        c.put(b.clone(), Bytes::from(vec![9u8; 80]));
        assert_eq!(c.disk_bytes(), 80, "exactly one object was reclaimed");
        assert!(c.get(&a).is_none());
        assert_eq!(c.get(&b).unwrap().len(), 80);
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

    /// A memory eviction demotes rather than deletes: the bytes stay on disk and a later
    /// read still hits. Under the ring the read does *not* promote back into memory — the
    /// entry keeps its place in the disk ring and is served from there.
    #[test]
    fn memory_eviction_demotes_to_disk_not_deletion() {
        let dir = tempfile::tempdir().unwrap();
        // Tiny memory (1 entry), generous disk.
        let c = DiskCache::with_budgets(dir.path(), 100, 100_000).unwrap();
        let a = CacheKey::new("ns", "a", "e");
        let b = CacheKey::new("ns", "b", "e");
        c.put(a.clone(), Bytes::from(vec![1u8; 100]));
        c.put(b.clone(), Bytes::from(vec![2u8; 100])); // demotes `a` out of memory

        assert_eq!(c.len(), 1, "memory holds only the newest admission");
        assert_eq!(c.disk_len(), 2, "both are still on disk");

        // `a` is gone from memory but still on disk — a get serves it as a hit, not a miss.
        assert_eq!(c.get(&a).unwrap(), Bytes::from(vec![1u8; 100]));
        assert_eq!(c.hits(), 1);
        assert_eq!(c.misses(), 0);
        assert_eq!(c.len(), 1, "a disk hit does not re-admit into memory");
    }
}
