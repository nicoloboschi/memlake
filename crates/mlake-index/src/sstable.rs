//! A sorted-key block store (SSTable) for range-readable secondary indexes.
//!
//! At 10M items, `pk` is ~200 MB and `radj` is multiple GB (SCALE.md #3), so neither can
//! be a whole-read JSON blob. Each is instead split into two objects, exactly as SPEC §3.4
//! describes for `radj`:
//!
//! * a small **`.idx`** — one entry per data block: the block's first key and its byte
//!   range. Loaded whole and cached (a few MB even at 10M), it is the only part resident;
//! * a large **`.data`** — the sorted key/value records, in blocks. Never read whole: a
//!   lookup binary-searches the in-memory index to one block, then issues a single ranged
//!   GET for just that block.
//!
//! So a point lookup is "cached index binary-search + one ranged GET", independent of the
//! table's size — the same discipline as the cluster files.

use mlake_core::{EntityId, MemoryId};
use mlake_store::{QueryMetrics, Store};

use crate::{Error, Result};

/// Target uncompressed size of a data block. A block is the unit of a ranged GET, so this
/// trades read amplification (larger = fewer, bigger reads) against granularity.
const BLOCK_TARGET_BYTES: usize = 16 * 1024;

/// One entry of the sparse `.idx`: a data block's first key and byte range in `.data`.
#[derive(Clone, Copy, Debug)]
struct BlockRef {
    first_key: [u8; 16],
    offset: u64,
    len: u32,
}

/// Builds an SSTable from keys added in ascending order.
pub struct SsTableBuilder {
    data: Vec<u8>,
    blocks: Vec<BlockRef>,
    cur: Vec<u8>,
    cur_first: Option<[u8; 16]>,
    last_key: Option<[u8; 16]>,
    count: u64,
}

impl Default for SsTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SsTableBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            blocks: Vec::new(),
            cur: Vec::new(),
            cur_first: None,
            last_key: None,
            count: 0,
        }
    }

    /// Append a record. Keys MUST be added in strictly ascending order — callers sort
    /// first — which is what lets a lookup binary-search and stop.
    pub fn add(&mut self, key: [u8; 16], value: &[u8]) {
        debug_assert!(
            self.last_key.map_or(true, |k| key > k),
            "SSTable keys must be added in ascending order"
        );
        self.last_key = Some(key);
        if self.cur.is_empty() {
            self.cur_first = Some(key);
        }
        // Record: [key:16][value_len:u32][value].
        self.cur.extend_from_slice(&key);
        self.cur.extend_from_slice(&(value.len() as u32).to_le_bytes());
        self.cur.extend_from_slice(value);
        self.count += 1;
        if self.cur.len() >= BLOCK_TARGET_BYTES {
            self.flush_block();
        }
    }

    fn flush_block(&mut self) {
        if self.cur.is_empty() {
            return;
        }
        self.blocks.push(BlockRef {
            first_key: self.cur_first.take().unwrap(),
            offset: self.data.len() as u64,
            len: self.cur.len() as u32,
        });
        self.data.append(&mut self.cur);
    }

    /// Finish, returning `(idx_bytes, data_bytes)` to write as the two objects.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.flush_block();

        // Index: [count:u64][block_count:u64] then per block [first_key:16][offset:8][len:4].
        let mut idx = Vec::with_capacity(16 + self.blocks.len() * 28);
        idx.extend_from_slice(&self.count.to_le_bytes());
        idx.extend_from_slice(&(self.blocks.len() as u64).to_le_bytes());
        for b in &self.blocks {
            idx.extend_from_slice(&b.first_key);
            idx.extend_from_slice(&b.offset.to_le_bytes());
            idx.extend_from_slice(&b.len.to_le_bytes());
        }
        (idx, self.data)
    }
}

/// The parsed sparse index — the only part of an SSTable held resident.
pub struct SsTableIndex {
    blocks: Vec<BlockRef>,
    count: u64,
}

impl SsTableIndex {
    /// Parse a `.idx` object.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(Error::Core(mlake_core::Error::Decode(
                "sstable index too short".into(),
            )));
        }
        let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let block_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut blocks = Vec::with_capacity(block_count);
        let mut p = 16;
        for _ in 0..block_count {
            if p + 28 > bytes.len() {
                return Err(Error::Core(mlake_core::Error::Decode(
                    "sstable index truncated".into(),
                )));
            }
            let mut first_key = [0u8; 16];
            first_key.copy_from_slice(&bytes[p..p + 16]);
            let offset = u64::from_le_bytes(bytes[p + 16..p + 24].try_into().unwrap());
            let len = u32::from_le_bytes(bytes[p + 24..p + 28].try_into().unwrap());
            blocks.push(BlockRef { first_key, offset, len });
            p += 28;
        }
        Ok(Self { blocks, count })
    }

    pub fn record_count(&self) -> u64 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The block that could contain `key`: the last block whose first key is ≤ `key`.
    fn block_for(&self, key: &[u8; 16]) -> Option<BlockRef> {
        if self.blocks.is_empty() {
            return None;
        }
        // partition_point gives the first block with first_key > key; the one before it is
        // the candidate.
        let i = self.blocks.partition_point(|b| &b.first_key <= key);
        if i == 0 {
            None // key precedes the whole table
        } else {
            Some(self.blocks[i - 1])
        }
    }

    /// Look up `key`, issuing one ranged GET for the containing block. Returns the value
    /// bytes, or `None` if the key is absent.
    pub async fn get(
        &self,
        store: &Store,
        data_path: &str,
        key: &MemoryId,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Option<Vec<u8>>> {
        let Some(block) = self.block_for(&key.0) else {
            return Ok(None);
        };
        let start = block.offset as usize;
        let end = start + block.len as usize;
        let bytes = store.get_range(data_path, start..end, ctx).await?;
        Ok(scan_block(&bytes, &key.0))
    }

    /// Look up many keys in **one coalesced request**: the distinct blocks the keys fall
    /// into are read together via `get_ranges` (the store coalesces adjacent ranges), then
    /// scanned in memory. This turns "N point lookups = N ranged GETs" into a single
    /// roundtrip — the fix for the graph arm's per-seed `radj`/`pk` reads.
    pub async fn get_many(
        &self,
        store: &Store,
        data_path: &str,
        keys: &[MemoryId],
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Vec<(MemoryId, Vec<u8>)>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        // Distinct blocks needed, and which keys land in each.
        let mut block_keys: std::collections::BTreeMap<(usize, usize), Vec<MemoryId>> =
            std::collections::BTreeMap::new();
        for k in keys {
            if let Some(b) = self.block_for(&k.0) {
                let range = (b.offset as usize, b.offset as usize + b.len as usize);
                block_keys.entry(range).or_default().push(*k);
            }
        }
        if block_keys.is_empty() {
            return Ok(Vec::new());
        }
        let ranges: Vec<std::ops::Range<usize>> =
            block_keys.keys().map(|(s, e)| *s..*e).collect();
        let blocks = store.get_ranges(data_path, &ranges, ctx).await?;

        let mut out = Vec::new();
        for (block_bytes, ks) in blocks.iter().zip(block_keys.values()) {
            for k in ks {
                if let Some(v) = scan_block(block_bytes, &k.0) {
                    out.push((*k, v));
                }
            }
        }
        Ok(out)
    }

    /// All records with key in `[lo, hi]` (inclusive), in key order, read as one coalesced
    /// request over the covering blocks. This is the range primitive the time index needs:
    /// entry-point selection over a time window becomes one bounded ranged scan.
    pub async fn scan_range(
        &self,
        store: &Store,
        data_path: &str,
        lo: &[u8; 16],
        hi: &[u8; 16],
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Vec<([u8; 16], Vec<u8>)>> {
        if self.blocks.is_empty() || lo > hi {
            return Ok(Vec::new());
        }
        // First block that could hold `lo`: the last block with first_key <= lo (or block 0
        // if lo precedes the whole table). Last block to read: any with first_key <= hi.
        let start = self.blocks.partition_point(|b| &b.first_key <= lo).saturating_sub(1);
        let end = self.blocks.partition_point(|b| &b.first_key <= hi); // exclusive
        if start >= end {
            return Ok(Vec::new());
        }
        let ranges: Vec<std::ops::Range<usize>> = self.blocks[start..end]
            .iter()
            .map(|b| b.offset as usize..b.offset as usize + b.len as usize)
            .collect();
        let blocks = store.get_ranges(data_path, &ranges, ctx).await?;
        let mut out = Vec::new();
        for block_bytes in &blocks {
            scan_block_range(block_bytes, lo, hi, &mut out);
        }
        Ok(out)
    }
}

/// Collect every record in `block` whose key is in `[lo, hi]`. Records are sorted, so the
/// scan stops once it passes `hi`.
fn scan_block_range(block: &[u8], lo: &[u8; 16], hi: &[u8; 16], out: &mut Vec<([u8; 16], Vec<u8>)>) {
    let mut p = 0;
    while p + 20 <= block.len() {
        let mut rec_key = [0u8; 16];
        rec_key.copy_from_slice(&block[p..p + 16]);
        let vlen = u32::from_le_bytes(block[p + 16..p + 20].try_into().unwrap()) as usize;
        let vstart = p + 20;
        let vend = vstart + vlen;
        if vend > block.len() {
            break;
        }
        if &rec_key > hi {
            break; // sorted: nothing further is in range
        }
        if &rec_key >= lo {
            out.push((rec_key, block[vstart..vend].to_vec()));
        }
        p = vend;
    }
}

/// Linear scan of a block for `key`. Records are sorted, so the scan stops once it passes
/// the key. Blocks are ~16 KB, so this is a few hundred comparisons at most.
fn scan_block(block: &[u8], key: &[u8; 16]) -> Option<Vec<u8>> {
    let mut p = 0;
    while p + 20 <= block.len() {
        let rec_key = &block[p..p + 16];
        let vlen = u32::from_le_bytes(block[p + 16..p + 20].try_into().unwrap()) as usize;
        let vstart = p + 20;
        let vend = vstart + vlen;
        if vend > block.len() {
            break;
        }
        match rec_key.cmp(&key[..]) {
            std::cmp::Ordering::Equal => return Some(block[vstart..vend].to_vec()),
            std::cmp::Ordering::Greater => return None, // passed it: absent
            std::cmp::Ordering::Less => {}
        }
        p = vend;
    }
    None
}

// ---------------------------------------------------------------- typed tables

use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag};

/// The `pk` index as an SSTable: item id → cluster index. A graph candidate's cluster is
/// found with one cached-index lookup + one ranged GET, rather than a 200 MB whole read.
pub struct PkTable {
    index: SsTableIndex,
    data_path: String,
}

impl PkTable {
    /// Build the two objects from id→cluster entries (any order; sorted here).
    pub fn build(mut entries: Vec<(MemoryId, u32)>) -> (Vec<u8>, Vec<u8>) {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut b = SsTableBuilder::new();
        for (id, cluster) in entries {
            b.add(id.0, &cluster.to_le_bytes());
        }
        b.finish()
    }

    pub fn open(idx_bytes: &[u8], data_path: impl Into<String>) -> Result<Self> {
        Ok(Self {
            index: SsTableIndex::parse(idx_bytes)?,
            data_path: data_path.into(),
        })
    }

    pub fn record_count(&self) -> u64 {
        self.index.record_count()
    }

    /// The cluster an item lives in, or `None` if the id is not in this generation.
    pub async fn lookup(
        &self,
        store: &Store,
        id: &MemoryId,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Option<u32>> {
        Ok(self
            .index
            .get(store, &self.data_path, id, ctx)
            .await?
            .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]])))
    }

    /// Resolve many ids to their clusters in one coalesced request.
    pub async fn lookup_batch(
        &self,
        store: &Store,
        ids: &[MemoryId],
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<std::collections::HashMap<MemoryId, u32>> {
        let pairs = self.index.get_many(store, &self.data_path, ids, ctx).await?;
        Ok(pairs
            .into_iter()
            .map(|(id, v)| (id, u32::from_le_bytes([v[0], v[1], v[2], v[3]])))
            .collect())
    }
}

/// The reverse-adjacency index as an SSTable: target id → its incoming edges. `incoming`
/// is one cached-index lookup + one ranged GET, not a multi-GB whole read.
pub struct RadjTable {
    index: SsTableIndex,
    data_path: String,
}

impl RadjTable {
    /// Build from `(target, edge)` pairs (any order; grouped and sorted here).
    pub fn build(mut pairs: Vec<(MemoryId, InEdge)>) -> (Vec<u8>, Vec<u8>) {
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.source.cmp(&b.1.source)));
        let mut b = SsTableBuilder::new();
        let mut i = 0;
        while i < pairs.len() {
            let target = pairs[i].0;
            let mut value = Vec::new();
            while i < pairs.len() && pairs[i].0 == target {
                encode_edge(&pairs[i].1, &mut value);
                i += 1;
            }
            b.add(target.0, &value);
        }
        b.finish()
    }

    pub fn open(idx_bytes: &[u8], data_path: impl Into<String>) -> Result<Self> {
        Ok(Self {
            index: SsTableIndex::parse(idx_bytes)?,
            data_path: data_path.into(),
        })
    }

    pub fn edge_count_hint(&self) -> u64 {
        // Number of targets with incoming edges; used only as an "is the graph non-empty"
        // signal, so a target count is sufficient.
        self.index.record_count()
    }

    /// Incoming edges for a target, or empty if it has none.
    pub async fn incoming(
        &self,
        store: &Store,
        target: &MemoryId,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Vec<InEdge>> {
        match self.index.get(store, &self.data_path, target, ctx).await? {
            Some(bytes) => Ok(decode_edges(&bytes)),
            None => Ok(Vec::new()),
        }
    }

    /// Incoming edges for many targets in one coalesced request — the graph arm's seed
    /// expansion, turned from N ranged GETs into a single roundtrip.
    pub async fn incoming_batch(
        &self,
        store: &Store,
        targets: &[MemoryId],
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<std::collections::HashMap<MemoryId, Vec<InEdge>>> {
        let pairs = self
            .index
            .get_many(store, &self.data_path, targets, ctx)
            .await?;
        Ok(pairs
            .into_iter()
            .map(|(id, bytes)| (id, decode_edges(&bytes)))
            .collect())
    }
}

/// The entity posting index: `EntityId -> sorted [MemoryId]`, the memories that carry each
/// entity. Same SSTable discipline as `radj`/`pk` — one bounded ranged GET per entity, so the
/// entity arm can find sharers anywhere in the corpus, not just in the probed clusters.
///
/// This is what makes the entity arm real graph expansion rather than a re-rank of the vector
/// neighbourhood. Entities and memory ids are both 16-byte, so the value of each key is just
/// the candidate `MemoryId`s concatenated.
pub struct EntityTable {
    index: SsTableIndex,
    data_path: String,
}

impl EntityTable {
    /// Build from `(entity, memory)` pairs (any order; grouped and sorted here).
    pub fn build(mut pairs: Vec<(EntityId, MemoryId)>) -> (Vec<u8>, Vec<u8>) {
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut b = SsTableBuilder::new();
        let mut i = 0;
        while i < pairs.len() {
            let entity = pairs[i].0;
            let mut value = Vec::new();
            while i < pairs.len() && pairs[i].0 == entity {
                value.extend_from_slice(&pairs[i].1 .0);
                i += 1;
            }
            b.add(entity.0, &value);
        }
        b.finish()
    }

    pub fn open(idx_bytes: &[u8], data_path: impl Into<String>) -> Result<Self> {
        Ok(Self {
            index: SsTableIndex::parse(idx_bytes)?,
            data_path: data_path.into(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The memories carrying each of `entities`, in one coalesced request, each list capped at
    /// `cap` (the bounded posting-prefix read of SPEC §7.2 — a high-fan-out entity can't blow
    /// the budget). Entities with no postings are absent from the map.
    pub async fn candidates_batch(
        &self,
        store: &Store,
        entities: &[EntityId],
        cap: usize,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<std::collections::HashMap<EntityId, Vec<MemoryId>>> {
        // EntityId and MemoryId are both 16-byte, so reuse the MemoryId-keyed reader.
        let keys: Vec<MemoryId> = entities.iter().map(|e| MemoryId(e.0)).collect();
        let pairs = self.index.get_many(store, &self.data_path, &keys, ctx).await?;
        Ok(pairs
            .into_iter()
            .map(|(k, bytes)| (EntityId(k.0), decode_ids(&bytes, cap)))
            .collect())
    }
}

fn decode_ids(bytes: &[u8], cap: usize) -> Vec<MemoryId> {
    bytes
        .chunks_exact(16)
        .take(cap)
        .map(|c| {
            let mut a = [0u8; 16];
            a.copy_from_slice(c);
            MemoryId(a)
        })
        .collect()
}

/// The time index: `effective_ts -> [MemoryId]`, sorted by time, so entry-point selection
/// over a `[from, to]` window is one bounded ranged scan (item i of the temporal arm), and
/// the memories timeseries is the same scan bucketed. `effective_ts` is
/// `COALESCE(occurred_start, mentioned_at, occurred_end)`, an `i64` epoch.
///
/// The key is an **order-preserving** 16-byte encoding of the `i64`: the sign bit is flipped
/// and the value stored big-endian, so raw byte order equals numeric order (negatives before
/// positives). Memories at the same instant are grouped into one key's value.
pub struct TimeTable {
    index: SsTableIndex,
    data_path: String,
}

/// Encode an `i64` timestamp as an order-preserving 16-byte SSTable key (8-byte flipped
/// big-endian value + 8 zero bytes; ties within the same instant share the key).
fn ts_key(ts: i64) -> [u8; 16] {
    let mut k = [0u8; 16];
    let flipped = (ts as u64) ^ (1u64 << 63);
    k[..8].copy_from_slice(&flipped.to_be_bytes());
    k
}

impl TimeTable {
    /// Build from `(effective_ts, memory)` pairs. Memories with no effective timestamp are
    /// simply not indexed (the caller filters `None`).
    pub fn build(mut pairs: Vec<(i64, MemoryId)>) -> (Vec<u8>, Vec<u8>) {
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut b = SsTableBuilder::new();
        let mut i = 0;
        while i < pairs.len() {
            let ts = pairs[i].0;
            let mut value = Vec::new();
            while i < pairs.len() && pairs[i].0 == ts {
                value.extend_from_slice(&pairs[i].1 .0);
                i += 1;
            }
            b.add(ts_key(ts), &value);
        }
        b.finish()
    }

    pub fn open(idx_bytes: &[u8], data_path: impl Into<String>) -> Result<Self> {
        Ok(Self {
            index: SsTableIndex::parse(idx_bytes)?,
            data_path: data_path.into(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The memories whose effective timestamp falls in `[from, to]` (inclusive), one bounded
    /// ranged scan over the covering blocks.
    pub async fn in_window(
        &self,
        store: &Store,
        from: i64,
        to: i64,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Vec<MemoryId>> {
        // The key for ts=`to` is `ts_key(to)` (…+zero padding); widen the high bound to that
        // instant's whole group so it is included.
        let mut hi = ts_key(to);
        hi[8..].copy_from_slice(&[0xFF; 8]);
        let records = self.index.scan_range(store, &self.data_path, &ts_key(from), &hi, ctx).await?;
        let mut ids = Vec::new();
        for (_, value) in records {
            ids.extend(decode_ids(&value, usize::MAX));
        }
        Ok(ids)
    }
}

/// Edge wire format: `[source:16][kind:1][linktype:1][weight:f32]` = 22 bytes.
const EDGE_BYTES: usize = 22;

fn encode_edge(e: &InEdge, out: &mut Vec<u8>) {
    out.extend_from_slice(&e.source.0);
    let (kind, lt) = match e.kind {
        EdgeKind::Semantic => (0u8, 0u8),
        EdgeKind::Causal(lt) => (
            1u8,
            match lt {
                LinkTypeTag::Causes => 0,
                LinkTypeTag::CausedBy => 1,
                LinkTypeTag::Enables => 2,
                LinkTypeTag::Prevents => 3,
            },
        ),
    };
    out.push(kind);
    out.push(lt);
    out.extend_from_slice(&e.weight.to_le_bytes());
}

fn decode_edges(bytes: &[u8]) -> Vec<InEdge> {
    let mut edges = Vec::with_capacity(bytes.len() / EDGE_BYTES);
    let mut p = 0;
    while p + EDGE_BYTES <= bytes.len() {
        let mut source = [0u8; 16];
        source.copy_from_slice(&bytes[p..p + 16]);
        let kind_byte = bytes[p + 16];
        let lt_byte = bytes[p + 17];
        let weight = f32::from_le_bytes(bytes[p + 18..p + 22].try_into().unwrap());
        let kind = if kind_byte == 0 {
            EdgeKind::Semantic
        } else {
            EdgeKind::Causal(match lt_byte {
                1 => LinkTypeTag::CausedBy,
                2 => LinkTypeTag::Enables,
                3 => LinkTypeTag::Prevents,
                _ => LinkTypeTag::Causes,
            })
        };
        edges.push(InEdge {
            source: MemoryId(source),
            kind,
            weight,
        });
        p += EDGE_BYTES;
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_key_is_order_preserving_across_the_sign_boundary() {
        // Raw byte order of the key must equal numeric order of the i64, incl. negatives.
        let mut times = vec![i64::MIN, -1_000_000, -1, 0, 1, 1_700_000_000_000, i64::MAX];
        let mut shuffled = times.clone();
        shuffled.reverse();
        shuffled.sort_by_key(|&t| ts_key(t));
        times.sort();
        assert_eq!(shuffled, times, "ts_key byte order must match numeric order");
    }

    #[tokio::test]
    async fn time_window_scan_returns_ids_in_range() {
        let store = Store::in_memory();
        // Memories at times -100, -50, 0, 10, ..., 200 (mix of negative + positive epochs).
        let pairs: Vec<(i64, MemoryId)> = (-2..=20)
            .map(|i| (i * 10, MemoryId::from_key(&format!("m{i}"))))
            .collect();
        let (idx, data) = TimeTable::build(pairs);
        store.put("t/time.data", data).await.unwrap();
        let tt = TimeTable::open(&idx, "t/time.data").unwrap();

        // Window [-10, 50] should return times -10, 0, 10, 20, 30, 40, 50 -> ids m-1..m5.
        let got = tt.in_window(&store, -10, 50, None).await.unwrap();
        let mut got_keys: Vec<MemoryId> = got;
        got_keys.sort();
        let mut want: Vec<MemoryId> = (-1..=5).map(|i| MemoryId::from_key(&format!("m{i}"))).collect();
        want.sort();
        assert_eq!(got_keys, want);

        // Empty window before everything.
        assert!(tt.in_window(&store, i64::MIN, -1000, None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn point_lookups_hit_the_right_block() {
        let mut b = SsTableBuilder::new();
        // Many records so the table spans several blocks.
        let mut keys: Vec<MemoryId> = (0..5000).map(|i| MemoryId::from_key(&format!("k{i:05}"))).collect();
        keys.sort();
        for (i, k) in keys.iter().enumerate() {
            b.add(k.0, format!("value-{i}").as_bytes());
        }
        let (idx_bytes, data_bytes) = b.finish();
        assert!(idx_bytes.len() < data_bytes.len(), "index must be smaller than data");

        let store = Store::in_memory();
        store.put("t.data", data_bytes).await.unwrap();
        let idx = SsTableIndex::parse(&idx_bytes).unwrap();
        assert_eq!(idx.record_count(), 5000);

        // Every key resolves to its value.
        for (i, k) in keys.iter().enumerate() {
            let v = idx.get(&store, "t.data", k, None).await.unwrap();
            assert_eq!(v.as_deref(), Some(format!("value-{i}").as_bytes()));
        }
    }

    #[tokio::test]
    async fn absent_keys_return_none() {
        let mut b = SsTableBuilder::new();
        let mut keys: Vec<MemoryId> = (0..100).map(|i| MemoryId::from_key(&format!("k{i}"))).collect();
        keys.sort();
        for k in &keys {
            b.add(k.0, b"x");
        }
        let (idx_bytes, data_bytes) = b.finish();
        let store = Store::in_memory();
        store.put("t.data", data_bytes).await.unwrap();
        let idx = SsTableIndex::parse(&idx_bytes).unwrap();

        assert!(idx
            .get(&store, "t.data", &MemoryId::from_key("absent"), None)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn a_lookup_reads_only_one_block_not_the_whole_table() {
        let mut b = SsTableBuilder::new();
        let mut keys: Vec<MemoryId> = (0..20000).map(|i| MemoryId::from_key(&format!("k{i:06}"))).collect();
        keys.sort();
        for k in &keys {
            b.add(k.0, &[0u8; 64]); // fat values so the table is many blocks
        }
        let (idx_bytes, data_bytes) = b.finish();
        let total = data_bytes.len();
        let store = Store::in_memory();
        store.put("t.data", data_bytes).await.unwrap();
        let idx = SsTableIndex::parse(&idx_bytes).unwrap();

        let metrics = QueryMetrics::new();
        idx.get(&store, "t.data", &keys[10000], Some((&metrics, 4))).await.unwrap();
        // Exactly one ranged GET, and it read far less than the whole table.
        assert_eq!(metrics.requests(), 1);
        assert!(
            (metrics.bytes() as usize) < total / 10,
            "a lookup must read one block ({} bytes), not the whole {total}-byte table",
            metrics.bytes()
        );
    }

    #[test]
    fn empty_table_is_valid() {
        let (idx_bytes, data_bytes) = SsTableBuilder::new().finish();
        assert!(data_bytes.is_empty());
        let idx = SsTableIndex::parse(&idx_bytes).unwrap();
        assert!(idx.is_empty());
    }
}

#[cfg(test)]
mod typed_tests {
    use super::*;
    use mlake_graph::radj::{EdgeKind, InEdge, LinkTypeTag};

    #[tokio::test]
    async fn pk_table_round_trips_cluster_lookups() {
        let entries: Vec<(MemoryId, u32)> =
            (0..3000).map(|i| (MemoryId::from_key(&format!("i{i}")), (i % 50) as u32)).collect();
        let (idx, data) = PkTable::build(entries.clone());
        let store = Store::in_memory();
        store.put("pk.data", data).await.unwrap();
        let pk = PkTable::open(&idx, "pk.data").unwrap();

        for (id, cluster) in &entries {
            assert_eq!(pk.lookup(&store, id, None).await.unwrap(), Some(*cluster));
        }
        assert_eq!(pk.lookup(&store, &MemoryId::from_key("nope"), None).await.unwrap(), None);
    }

    #[tokio::test]
    async fn radj_table_round_trips_incoming_edges() {
        let t1 = MemoryId::from_key("t1");
        let t2 = MemoryId::from_key("t2");
        let pairs = vec![
            (t1, InEdge { source: MemoryId::from_key("a"), kind: EdgeKind::Semantic, weight: 0.8 }),
            (t1, InEdge { source: MemoryId::from_key("b"), kind: EdgeKind::Causal(LinkTypeTag::Prevents), weight: 0.6 }),
            (t2, InEdge { source: MemoryId::from_key("c"), kind: EdgeKind::Semantic, weight: 0.9 }),
        ];
        let (idx, data) = RadjTable::build(pairs);
        let store = Store::in_memory();
        store.put("radj.csr", data).await.unwrap();
        let radj = RadjTable::open(&idx, "radj.csr").unwrap();

        let in1 = radj.incoming(&store, &t1, None).await.unwrap();
        assert_eq!(in1.len(), 2);
        assert!(in1.iter().any(|e| e.kind == EdgeKind::Causal(LinkTypeTag::Prevents) && (e.weight - 0.6).abs() < 1e-6));
        let in2 = radj.incoming(&store, &t2, None).await.unwrap();
        assert_eq!(in2.len(), 1);
        assert_eq!(in2[0].source, MemoryId::from_key("c"));
        // Unknown target: no edges, no error.
        assert!(radj.incoming(&store, &MemoryId::from_key("t9"), None).await.unwrap().is_empty());
    }
}
