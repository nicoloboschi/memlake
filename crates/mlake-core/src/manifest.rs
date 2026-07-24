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
/// hydrate hits; to 4 for the split vector blocks (`cluster-{i}.vec`) and the rerank tier
/// (`rerank.idx`/`rerank.data`) the two-stage search reads; to 5 for the segmented manifest
/// (a list of LSM segments instead of one generation — see docs/segmented-index.md).
pub const FORMAT_VERSION: u32 = 6;

/// Paths to the files making up a generation. Stored as an explicit struct rather than a
/// map so a missing file is a deserialization error rather than a runtime surprise.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct GenerationFiles {
    pub pk: String,
    pub centroids: String,
    pub clusters: Vec<String>,
    /// `cluster-{i}.vec`: one cluster's embeddings, split out of the cluster file and
    /// parallel to `clusters` by index. The vector arm reads only these — the embedding is
    /// ~84% of a stored memory, so scanning it alongside text and metadata meant every
    /// probe paid for bytes it scored once and discarded.
    #[serde(default)]
    pub vectors: Vec<String>,
    /// `radj.data`: the reverse-adjacency SSTable data blocks (range-read per lookup).
    pub radj_data: String,
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
    /// `rerank.idx` / `rerank.data`: full-precision embeddings, keyed by MemoryId.
    ///
    /// The second stage of the two-stage search. Never scanned — only point-fetched for the
    /// candidates whose error bound leaves them possibly in the true top-k, which is a small
    /// fraction of a probe. Keeping f32 here is what lets the scan tier be 1 bit per
    /// dimension without losing the exact ranking.
    #[serde(default)]
    pub rerank_idx: String,
    #[serde(default)]
    pub rerank_data: String,
}

impl GenerationFiles {
    /// Every object path this generation references, deduplicated. GC keeps exactly these
    /// (for the current and previous generations) and reclaims everything else.
    pub fn all_paths(&self) -> impl Iterator<Item = &str> {
        [
            self.pk.as_str(),
            self.pk_data.as_str(),
            self.centroids.as_str(),
            self.radj_data.as_str(),
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
            self.rerank_idx.as_str(),
            self.rerank_data.as_str(),
        ]
        .into_iter()
        .chain(self.clusters.iter().map(|s| s.as_str()))
        .chain(self.vectors.iter().map(|s| s.as_str()))
        .filter(|s| !s.is_empty())
    }
}

/// One memory_type's files within a single segment. Fact types share nothing — no links,
/// vectors, or postings — so each carries its own files and its own retrain count.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct FactTypeIndex {
    pub files: GenerationFiles,
    /// Memory count when this fact type's centroids were last trained (assign-only trigger).
    #[serde(default)]
    pub train_count: u64,
}

impl FactTypeIndex {
    fn all_paths(&self) -> impl Iterator<Item = &str> {
        self.files.all_paths()
    }
}

/// One immutable segment: a self-contained mini-index over a slice of the WAL, at a level in the
/// LSM stack. A flush appends a new L0 segment (its `seq` slice); compaction merges segments into a
/// higher level (see docs/segmented-index.md). Queries fan out across all live segments and merge.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct Segment {
    /// Content nonce — the segment's immutable object prefix, so it is safe to GC by identity.
    pub id: String,
    /// LSM level: 0 = newest/smallest (a flush), higher = compacted.
    pub level: u32,
    /// Inclusive WAL sequence range this segment covers.
    pub seq_lo: u64,
    pub seq_hi: u64,
    /// Live item count across all fact types in this segment.
    pub doc_count: u64,
    /// Per-fact-type files within this segment.
    pub indexes: BTreeMap<u8, FactTypeIndex>,
    /// `tombstones.json`: this segment's delete overlay — the ids it supersedes in OLDER segments
    /// (deletes + re-upserts) and its predicate-deletes. Small (bounded by the flush slice's
    /// deletes/re-upserts, not the corpus), loaded whole at query open. Empty for a full-rebuild /
    /// compacted segment, which has already materialized its deletes. See docs/segmented-index.md §6.
    #[serde(default)]
    pub tombstones: String,
}

impl Segment {
    fn all_paths(&self) -> impl Iterator<Item = &str> {
        self.indexes
            .values()
            .flat_map(|idx| idx.all_paths())
            .chain(std::iter::once(self.tombstones.as_str()))
            .filter(|s| !s.is_empty())
    }

    /// This segment's fact types.
    pub fn memory_types(&self) -> impl Iterator<Item = u8> + '_ {
        self.indexes.keys().copied()
    }

    /// This fact type's files within this segment, if present.
    pub fn index(&self, memory_type: u8) -> Option<&FactTypeIndex> {
        self.indexes.get(&memory_type)
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct Manifest {
    pub format_version: u32,
    /// Monotonic manifest version — bumped on every swap, for debugging and ordering (not a
    /// generation number anymore; the index is a set of segments, not one generation).
    pub version: u64,
    /// Last WAL sequence folded into a segment. Readers scan the WAL tail past this point to
    /// satisfy strong consistency.
    pub wal_index_cursor: u64,
    /// Last committed WAL sequence as of this manifest write.
    pub wal_head: u64,
    /// Guards against querying a split built with a different tokenizer than the query
    /// parser uses — a silent, hard-to-debug recall failure otherwise.
    pub tokenizer_config_hash: String,
    /// The previous manifest's WAL cursor. A strong reader that read the previous manifest is
    /// scanning `(prev_wal_index_cursor, head]`, so GC must keep WAL entries above this watermark.
    #[serde(default)]
    pub prev_wal_index_cursor: u64,
    /// The live segments, newest-first within a level. Empty until the first flush indexes
    /// something. Queries read across all of them.
    #[serde(default)]
    pub segments: Vec<Segment>,
    /// Segments dropped by the last swap (superseded by a flush/compaction), kept alive for the
    /// GC grace period so in-flight readers holding the previous manifest still see their files.
    #[serde(default)]
    pub prev_segments: Vec<Segment>,
    /// Metadata keys the namespace has declared for value-count aggregation (`MetadataStats`).
    /// Declared once at creation and carried across every swap. The fold tallies each item's
    /// value for exactly these keys into the per-fact-type `Stats.meta_counts`; a key not
    /// declared here is never counted, so the cost is bounded by distinct declared
    /// (key, value) pairs rather than by every metadata key in the corpus.
    #[serde(default)]
    pub indexed_metadata_keys: Vec<String>,
}

impl Manifest {
    /// The manifest for a bank namespace that has been created but never indexed.
    pub fn empty(tokenizer_config_hash: impl Into<String>, indexed_metadata_keys: Vec<String>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            version: 0,
            wal_index_cursor: 0,
            wal_head: 0,
            tokenizer_config_hash: tokenizer_config_hash.into(),
            prev_wal_index_cursor: 0,
            segments: Vec::new(),
            prev_segments: Vec::new(),
            indexed_metadata_keys,
        }
    }

    /// The newest segment's index for a fact type. **Phase-1 convenience** while there is a single
    /// segment; a multi-segment query must fan out across [`Manifest::segments`] and merge, not use
    /// this. Returns `None` if no segment has that type.
    pub fn index(&self, memory_type: u8) -> Option<&FactTypeIndex> {
        self.segments.iter().find_map(|s| s.index(memory_type))
    }

    /// The fact types any live segment has an index for (deduplicated, ascending).
    pub fn memory_types(&self) -> impl Iterator<Item = u8> + '_ {
        let set: std::collections::BTreeSet<u8> =
            self.segments.iter().flat_map(|s| s.memory_types()).collect();
        set.into_iter()
    }

    /// True when nothing has been indexed yet, so all reads come from the WAL tail.
    pub fn is_empty(&self) -> bool {
        self.segments.iter().all(|s| s.indexes.is_empty())
    }

    /// Total live doc count summed across segments (upper bound — cross-segment shadowing is
    /// resolved at query time, not counted here).
    pub fn doc_count(&self) -> u64 {
        self.segments.iter().map(|s| s.doc_count).sum()
    }

    /// Every object path any live or grace-window segment references. GC keeps exactly these and
    /// reclaims the rest.
    pub fn all_referenced_paths(&self) -> impl Iterator<Item = &str> {
        self.segments
            .iter()
            .chain(self.prev_segments.iter())
            .flat_map(|s| s.all_paths())
    }

    /// Number of WAL entries not yet folded into a segment.
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

/// Object key for a namespace's WAL head pointer — a small, monotonic record of the highest
/// sequence any writer has committed. Readers GET it (etag-cacheable) to learn the live head
/// without a LIST, which on S3 is both slower and ~12× the per-request price of a GET. A writer
/// bumps it after durably appending its entry; the pointer never decreases, so it has no
/// fold/GC race (unlike deriving the head from the live WAL objects, which GC reclaims). It is
/// only an accelerator: if it is missing or lags a crashed writer's un-acked entry, readers fall
/// back to the LIST, and the indexer (which LISTs anyway) reconciles it. Lives under the
/// namespace prefix — but NOT under `{ns}/wal/` — so `delete_all` reclaims it while the WAL GC,
/// which parses `{ns}/wal/` keys as sequences, never touches it.
pub fn wal_head_pointer_path(namespace: &str) -> String {
    format!("{namespace}/wal-head")
}

/// Object key for a WAL entry. Zero-padded so lexicographic listing is sequence order —
/// this is what makes "find the head" a single LIST with a start-after cursor.
///
/// The width is 20, the digit count of `u64::MAX`, so EVERY sequence in the u64 range formats to
/// the same length and lexicographic order equals numeric order for all of it. A narrower pad (e.g.
/// 8) is a lexicographic time bomb: the first sequence that overflows the width grows a digit and
/// sorts *before* all the shorter keys (`100000000` < `99999999`), silently reordering the WAL.
pub fn wal_path(namespace: &str, seq: u64) -> String {
    format!("{namespace}/wal/{seq:020}.bin")
}

/// Prefix under which one segment's files live. `seg_id` is a content nonce, so the prefix is
/// unique and immutable — safe to GC by identity once the manifest no longer references it.
pub fn segment_prefix(namespace: &str, seg_id: &str) -> String {
    format!("{namespace}/seg-{seg_id}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wal_paths_sort_in_sequence_order() {
        // Straddle the 10^8 boundary — where an 8-wide pad would grow a digit and sort the larger
        // sequence *before* the smaller ones — and go up to u64::MAX, to prove lexicographic key
        // order equals numeric sequence order across the whole u64 range.
        let seqs = [1u64, 9, 10, 100, 99_999_999, 100_000_000, 100_000_001, u64::MAX];
        let mut paths: Vec<String> = seqs.iter().map(|s| wal_path("ns", *s)).collect();
        paths.sort();
        let sorted: Vec<u64> = paths
            .iter()
            .map(|p| p.rsplit('/').next().unwrap().strip_suffix(".bin").unwrap().parse().unwrap())
            .collect();
        let mut expected = seqs.to_vec();
        expected.sort_unstable();
        assert_eq!(sorted, expected, "lexicographic key order must equal numeric sequence order");
        // Every key is the same length, so no sequence can ever overflow the pad and reorder.
        assert!(paths.iter().all(|p| p.len() == paths[0].len()));
    }

    #[test]
    fn roundtrip_preserves_manifest() {
        let m = Manifest::empty("tok-hash", Vec::new());
        let bytes = m.to_bytes().unwrap();
        assert_eq!(Manifest::from_bytes(&bytes).unwrap(), m);
    }

    #[test]
    fn rejects_unknown_format_version() {
        let mut v = serde_json::to_value(Manifest::empty("h", Vec::new())).unwrap();
        v["format_version"] = serde_json::json!(999);
        let bytes = serde_json::to_vec(&v).unwrap();
        assert!(matches!(
            Manifest::from_bytes(&bytes),
            Err(crate::Error::FormatVersion { found: 999, .. })
        ));
    }

    #[test]
    fn index_lag_is_head_minus_cursor() {
        let mut m = Manifest::empty("h", Vec::new());
        m.wal_head = 141;
        m.wal_index_cursor = 137;
        assert_eq!(m.index_lag(), 4);
        // Never negative, even if a stale manifest reports a cursor ahead of head.
        m.wal_head = 100;
        assert_eq!(m.index_lag(), 0);
    }
}
