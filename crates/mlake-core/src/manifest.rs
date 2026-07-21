//! The manifest: the single mutable pointer that defines what a namespace currently is.
//!
//! Every other file on S3 is immutable. Publishing new data means writing new files and
//! then CAS-swapping this one object (INV-2). A reader that has read the manifest holds a
//! consistent, complete view of a generation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// On-disk format version. **Bump this on any change to a serialized on-disk layout** —
/// `StoredMemory` / `Memory` (rkyv cluster + WAL records), the SSTable encodings, or the
/// manifest schema — so a generation written by an older build is rejected at the manifest
/// read with a clear error instead of failing deep in an rkyv decode ("pointer overran
/// buffer"). Bumped to 2 for the `write_seq` + opaque `metadata` + 16-byte `EntityId` changes;
/// to 3 for the payload store (`payload.idx`/`payload.data`), which a reader must find to
/// hydrate hits.
pub const FORMAT_VERSION: u32 = 3;

/// Paths to the files making up a generation. Stored as an explicit struct rather than a
/// map so a missing file is a deserialization error rather than a runtime surprise.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct GenerationFiles {
    pub pk: String,
    pub centroids: String,
    pub clusters: Vec<String>,
    /// `radj.csr`: the reverse-adjacency SSTable data blocks (range-read per lookup).
    pub radj_csr: String,
    /// `radj.idx`: the reverse-adjacency SSTable sparse index (loaded whole, small).
    pub radj_idx: String,
    pub fts_split: String,
    pub stats: String,
    /// `pk.data`: the primary-key SSTable data blocks. `pk` above is its sparse index.
    #[serde(default)]
    pub pk_data: String,
    /// Per-cluster tag summaries, for pruning clusters before fetch (SCALE.md Phase 4b).
    #[serde(default)]
    pub tag_summary: String,
    /// `entity.idx` / `entity.data`: the entity posting SSTable (EntityId -> [MemoryId]),
    /// range-read per entity by the graph arm's entity expansion.
    #[serde(default)]
    pub entity_idx: String,
    #[serde(default)]
    pub entity_data: String,
    /// `time.idx` / `time.data`: the time index SSTable (effective_ts -> [MemoryId]),
    /// range-scanned by the temporal arm's entry-point selection.
    #[serde(default)]
    pub time_idx: String,
    #[serde(default)]
    pub time_data: String,
    /// `payload.idx` / `payload.data`: the payload store (MemoryId -> memory bytes without the
    /// embedding), range-read to hydrate a hit or a `get` without deserializing its cluster.
    #[serde(default)]
    pub payload_idx: String,
    #[serde(default)]
    pub payload_data: String,
}

impl GenerationFiles {
    /// Every object path this generation references, deduplicated. GC keeps exactly these
    /// (for the current and previous generations) and reclaims everything else.
    pub fn all_paths(&self) -> impl Iterator<Item = &str> {
        [
            self.pk.as_str(),
            self.pk_data.as_str(),
            self.centroids.as_str(),
            self.radj_csr.as_str(),
            self.radj_idx.as_str(),
            self.fts_split.as_str(),
            self.stats.as_str(),
            self.tag_summary.as_str(),
            self.entity_idx.as_str(),
            self.entity_data.as_str(),
            self.time_idx.as_str(),
            self.time_data.as_str(),
            self.payload_idx.as_str(),
            self.payload_data.as_str(),
        ]
        .into_iter()
        .chain(self.clusters.iter().map(|s| s.as_str()))
        .filter(|s| !s.is_empty())
    }
}

/// One memory_type's independent index within a bank. Fact types share nothing — no links,
/// vectors, or postings — so each carries its own generation files and its own assign-only
/// retrain state (SCALE.md Phase 4).
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct FactTypeIndex {
    pub files: GenerationFiles,
    /// The previous generation's files for this fact type, retained as the GC grace window.
    #[serde(default)]
    pub prev_files: Option<GenerationFiles>,
    /// Memory count when this fact type's centroids were last trained (assign-only trigger).
    #[serde(default)]
    pub train_count: u64,
}

impl FactTypeIndex {
    fn all_paths(&self) -> impl Iterator<Item = &str> {
        self.files.all_paths().chain(
            self.prev_files
                .iter()
                .flat_map(|f| f.all_paths())
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct Manifest {
    pub format_version: u32,
    pub generation: u64,
    /// Last WAL sequence folded into the current generation. Readers scan the WAL tail past
    /// this point to satisfy strong consistency.
    pub wal_index_cursor: u64,
    /// Last committed WAL sequence as of this manifest write.
    pub wal_head: u64,
    /// Guards against querying a split built with a different tokenizer than the query
    /// parser uses — a silent, hard-to-debug recall failure otherwise.
    pub tokenizer_config_hash: String,
    /// Kept alive for the GC grace period so in-flight readers holding the previous
    /// manifest do not observe deleted files.
    pub prev_generation: Option<u64>,
    /// The previous generation's WAL cursor. WAL entries at or below the *current* cursor
    /// are folded, but a strong reader that read the previous manifest is scanning
    /// `(prev_wal_index_cursor, head]`, so GC must keep entries above this watermark.
    #[serde(default)]
    pub prev_wal_index_cursor: u64,
    /// Per-fact-type independent indexes. A bank namespace holds one WAL and this map; each
    /// entry is a fully separate generation. Empty until the first fold indexes something.
    #[serde(default)]
    pub indexes: BTreeMap<u8, FactTypeIndex>,
}

impl Manifest {
    /// The manifest for a bank namespace that has been created but never indexed.
    pub fn empty(tokenizer_config_hash: impl Into<String>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            generation: 0,
            wal_index_cursor: 0,
            wal_head: 0,
            tokenizer_config_hash: tokenizer_config_hash.into(),
            prev_generation: None,
            prev_wal_index_cursor: 0,
            indexes: BTreeMap::new(),
        }
    }

    /// This fact type's index, or `None` if the bank has never indexed that type.
    pub fn index(&self, memory_type: u8) -> Option<&FactTypeIndex> {
        self.indexes.get(&memory_type)
    }

    /// The fact types this bank currently has an index for.
    pub fn memory_types(&self) -> impl Iterator<Item = u8> + '_ {
        self.indexes.keys().copied()
    }

    /// True when nothing has been indexed yet, so all reads come from the WAL tail.
    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// Every object path any fact type's current or previous generation references. GC
    /// keeps exactly these and reclaims the rest.
    pub fn all_referenced_paths(&self) -> impl Iterator<Item = &str> {
        self.indexes.values().flat_map(|idx| idx.all_paths())
    }

    /// Number of WAL entries not yet folded into a generation.
    pub fn index_lag(&self) -> u64 {
        self.wal_head.saturating_sub(self.wal_index_cursor)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::Error> {
        let m: Manifest = serde_json::from_slice(bytes)?;
        if m.format_version != FORMAT_VERSION {
            return Err(crate::Error::FormatVersion {
                found: m.format_version,
                expected: FORMAT_VERSION,
            });
        }
        Ok(m)
    }
}

/// Object key for a namespace's manifest.
pub fn manifest_path(namespace: &str) -> String {
    format!("{namespace}/manifest.json")
}

/// Object key for a namespace's index lease — a best-effort marker that one node is currently
/// folding this namespace, so the other nodes' periodic indexers skip it and don't duplicate
/// the compute and S3 PUTs. Lives under the namespace prefix so `delete_all` reclaims it. It
/// is only an optimization: losing or ignoring it costs a wasted (but safe) duplicate fold,
/// never correctness — the nonce'd generation prefixes already make concurrent folds safe.
pub fn index_lease_path(namespace: &str) -> String {
    format!("{namespace}/index-lease.json")
}

/// Object key for a WAL entry. Zero-padded so lexicographic listing is sequence order —
/// this is what makes "find the head" a single LIST with a start-after cursor.
pub fn wal_path(namespace: &str, seq: u64) -> String {
    format!("{namespace}/wal/{seq:08}.bin")
}

/// Prefix under which a generation's files live.
pub fn generation_prefix(namespace: &str, generation: u64) -> String {
    format!("{namespace}/gen-{generation}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wal_paths_sort_in_sequence_order() {
        let mut paths: Vec<String> = [9u64, 100, 10, 1].iter().map(|s| wal_path("ns", *s)).collect();
        paths.sort();
        let seqs: Vec<&str> = paths.iter().map(|p| p.rsplit('/').next().unwrap()).collect();
        assert_eq!(
            seqs,
            vec!["00000001.bin", "00000009.bin", "00000010.bin", "00000100.bin"]
        );
    }

    #[test]
    fn roundtrip_preserves_manifest() {
        let m = Manifest::empty("tok-hash");
        let bytes = m.to_bytes().unwrap();
        assert_eq!(Manifest::from_bytes(&bytes).unwrap(), m);
    }

    #[test]
    fn rejects_unknown_format_version() {
        let mut v = serde_json::to_value(Manifest::empty("h")).unwrap();
        v["format_version"] = serde_json::json!(999);
        let bytes = serde_json::to_vec(&v).unwrap();
        assert!(matches!(
            Manifest::from_bytes(&bytes),
            Err(crate::Error::FormatVersion { found: 999, .. })
        ));
    }

    #[test]
    fn index_lag_is_head_minus_cursor() {
        let mut m = Manifest::empty("h");
        m.wal_head = 141;
        m.wal_index_cursor = 137;
        assert_eq!(m.index_lag(), 4);
        // Never negative, even if a stale manifest reports a cursor ahead of head.
        m.wal_head = 100;
        assert_eq!(m.index_lag(), 0);
    }
}
