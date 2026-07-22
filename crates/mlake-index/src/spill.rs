//! Disk spill + external merge-sort — the primitives that let the streaming fold keep RAM
//! bounded regardless of corpus size.
//!
//! The in-RAM fold holds every live memory (and clones them again per cluster), so its peak
//! memory is O(N): ~4 GB/million, which caps a first build at a few million on a 36 GB box.
//! The streaming fold instead spills resolved items to a temp file ([`ItemSpill`]) and builds
//! each SSTable through an [`ExternalSort`] — an external merge-sort that buffers pairs, spills
//! sorted runs when the buffer fills, and k-way-merges them into one ascending stream. Peak RAM
//! is then the buffer cap + one open cluster, not the whole corpus.
//!
//! Temp files are `tempfile::tempfile()` handles: unnamed, auto-removed on close, written once
//! and read back once (rewound to offset 0). All I/O here is synchronous local disk — the fold
//! is throughput-bound, not latency-bound, and mixing it with the async storage reads is fine.

use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use mlake_core::StoredMemory;

/// A write-once/read-once spill of serialized `StoredMemory` (each length-prefixed). Lets the
/// fold hold resolved items on disk instead of all in RAM.
pub(crate) struct ItemSpill {
    writer: BufWriter<File>,
    count: usize,
}

impl ItemSpill {
    pub fn new() -> io::Result<Self> {
        Ok(Self { writer: BufWriter::new(tempfile::tempfile()?), count: 0 })
    }

    /// Append one item (full, embedding included — the fold needs vectors to cluster/rerank).
    pub fn push(&mut self, m: &StoredMemory) -> io::Result<()> {
        let bytes = m.to_rkyv_bytes();
        self.writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
        self.writer.write_all(&bytes)?;
        self.count += 1;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    /// Rewind and return an iterator over the spilled items. Consumes the spill.
    pub fn into_reader(self) -> io::Result<ItemSpillReader> {
        let count = self.count;
        let mut file = self.writer.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(0))?;
        Ok(ItemSpillReader { reader: BufReader::new(file), remaining: count })
    }
}

pub(crate) struct ItemSpillReader {
    reader: BufReader<File>,
    remaining: usize,
}

impl Iterator for ItemSpillReader {
    type Item = StoredMemory;
    fn next(&mut self) -> Option<StoredMemory> {
        if self.remaining == 0 {
            return None;
        }
        let mut len = [0u8; 4];
        self.reader.read_exact(&mut len).ok()?;
        let n = u32::from_le_bytes(len) as usize;
        let mut buf = vec![0u8; n];
        self.reader.read_exact(&mut buf).ok()?;
        self.remaining -= 1;
        // `from_payload_bytes` decodes any rkyv `StoredMemory`, full or vector-stripped.
        StoredMemory::from_payload_bytes(&buf)
    }
}

/// One length-prefixed `(key, value)` run file, read sequentially.
pub(crate) struct RunReader {
    reader: BufReader<File>,
}

impl RunReader {
    fn read_next(&mut self) -> io::Result<Option<([u8; 16], Vec<u8>)>> {
        let mut key = [0u8; 16];
        match self.reader.read_exact(&mut key) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let mut lb = [0u8; 4];
        self.reader.read_exact(&mut lb)?;
        let n = u32::from_le_bytes(lb) as usize;
        let mut val = vec![0u8; n];
        self.reader.read_exact(&mut val)?;
        Ok(Some((key, val)))
    }
}

/// A heap entry for the k-way merge: min-ordered by key, then run index (a total, stable
/// order so the merged stream is deterministic).
pub(crate) struct HeapItem {
    key: [u8; 16],
    value: Vec<u8>,
    run: usize,
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.run == other.run
    }
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed so `BinaryHeap` (a max-heap) yields the smallest key first.
        other.key.cmp(&self.key).then(other.run.cmp(&self.run))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// External merge-sort of `([u8;16] key, Vec<u8> value)` pairs. Buffer, spill sorted runs when
/// the buffer exceeds `cap_bytes`, then [`finish`](Self::finish) into one ascending stream.
pub(crate) struct ExternalSort {
    buf: Vec<([u8; 16], Vec<u8>)>,
    buf_bytes: usize,
    cap_bytes: usize,
    runs: Vec<File>,
}

impl ExternalSort {
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            buf: Vec::new(),
            buf_bytes: 0,
            cap_bytes: cap_bytes.max(4096),
            runs: Vec::new(),
        }
    }

    pub fn add(&mut self, key: [u8; 16], value: Vec<u8>) -> io::Result<()> {
        self.buf_bytes += 20 + value.len();
        self.buf.push((key, value));
        if self.buf_bytes >= self.cap_bytes {
            self.spill()?;
        }
        Ok(())
    }

    fn spill(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        self.buf.sort_by(|a, b| a.0.cmp(&b.0));
        let mut w = BufWriter::new(tempfile::tempfile()?);
        for (k, v) in self.buf.drain(..) {
            w.write_all(&k)?;
            w.write_all(&(v.len() as u32).to_le_bytes())?;
            w.write_all(&v)?;
        }
        let mut f = w.into_inner().map_err(|e| e.into_error())?;
        f.seek(SeekFrom::Start(0))?;
        self.runs.push(f);
        self.buf_bytes = 0;
        Ok(())
    }

    /// Merge into one ascending `(key, value)` stream. If nothing ever spilled, sorts the
    /// buffer in RAM; otherwise k-way-merges the run files.
    pub fn finish(mut self) -> io::Result<Merge> {
        if self.runs.is_empty() {
            self.buf.sort_by(|a, b| a.0.cmp(&b.0));
            return Ok(Merge::InMemory(std::mem::take(&mut self.buf).into_iter()));
        }
        self.spill()?; // flush the remaining buffer as a final run
        let mut readers: Vec<RunReader> = Vec::with_capacity(self.runs.len());
        let mut heap = BinaryHeap::new();
        for (run, file) in self.runs.into_iter().enumerate() {
            let mut r = RunReader { reader: BufReader::new(file) };
            if let Some((key, value)) = r.read_next()? {
                heap.push(HeapItem { key, value, run });
            }
            readers.push(r);
        }
        Ok(Merge::Runs { readers, heap })
    }
}

/// An ascending stream of `(key, value)` — either an in-memory sorted vec or a k-way merge of
/// run files. Yields `io::Result` because a merge reads from disk lazily.
pub(crate) enum Merge {
    InMemory(std::vec::IntoIter<([u8; 16], Vec<u8>)>),
    Runs {
        readers: Vec<RunReader>,
        heap: BinaryHeap<HeapItem>,
    },
}

impl Merge {
    /// The next `(key, value)` in ascending order, or `None` at the end.
    pub fn next(&mut self) -> io::Result<Option<([u8; 16], Vec<u8>)>> {
        match self {
            Merge::InMemory(it) => Ok(it.next()),
            Merge::Runs { readers, heap } => {
                let Some(top) = heap.pop() else { return Ok(None) };
                if let Some((key, value)) = readers[top.run].read_next()? {
                    heap.push(HeapItem { key, value, run: top.run });
                }
                Ok(Some((top.key, top.value)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_sort_orders_across_spills() {
        // 4 KiB cap + ~200-byte values over 2000 keys => hundreds of KiB => many spilled runs,
        // exercising the k-way merge (not the in-memory path).
        let mut s = ExternalSort::new(4096);
        let mut expected: Vec<[u8; 16]> = Vec::new();
        for i in (0..2000u32).rev() {
            let mut k = [0u8; 16];
            k[12..].copy_from_slice(&i.to_be_bytes());
            s.add(k, vec![7u8; 200]).unwrap();
            expected.push(k);
            s.add(k, vec![9u8; 200]).unwrap(); // duplicate key
            expected.push(k);
        }
        expected.sort();

        let mut merge = s.finish().unwrap();
        assert!(matches!(merge, Merge::Runs { .. }), "this test must exercise the spilled merge");
        let mut got: Vec<[u8; 16]> = Vec::new();
        while let Some((k, _v)) = merge.next().unwrap() {
            got.push(k);
        }
        assert_eq!(got, expected, "merge must yield every pair in ascending key order");
    }

    #[test]
    fn external_sort_in_memory_path() {
        let mut s = ExternalSort::new(1 << 30); // never spills
        for i in [5u32, 1, 3, 2, 4] {
            let mut k = [0u8; 16];
            k[12..].copy_from_slice(&i.to_be_bytes());
            s.add(k, vec![i as u8]).unwrap();
        }
        let mut merge = s.finish().unwrap();
        let mut vals = Vec::new();
        while let Some((_k, v)) = merge.next().unwrap() {
            vals.push(v[0]);
        }
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }
}
