//! The manifest: the single mutable pointer that defines what a namespace currently is.
//!
//! Every other file on S3 is immutable. Publishing new data means writing new files and
//! then CAS-swapping this one object (INV-2). A reader that has read the manifest holds a
//! consistent, complete view of a generation.

use serde::{Deserialize, Serialize};

pub const FORMAT_VERSION: u32 = 1;

/// Paths to the files making up a generation. Stored as an explicit struct rather than a
/// map so a missing file is a deserialization error rather than a runtime surprise.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct GenerationFiles {
    pub pk: String,
    pub centroids: String,
    pub clusters: Vec<String>,
    pub radj_csr: String,
    pub radj_idx: String,
    pub fts_split: String,
    pub stats: String,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct Manifest {
    pub format_version: u32,
    pub generation: u64,
    /// Last WAL sequence folded into `generation`. Readers scan the WAL tail past this
    /// point to satisfy strong consistency.
    pub wal_index_cursor: u64,
    /// Last committed WAL sequence as of this manifest write.
    pub wal_head: u64,
    pub files: GenerationFiles,
    /// Guards against querying a split built with a different tokenizer than the query
    /// parser uses — a silent, hard-to-debug recall failure otherwise.
    pub tokenizer_config_hash: String,
    /// Kept alive for the GC grace period so in-flight readers holding the previous
    /// manifest do not observe deleted files.
    pub prev_generation: Option<u64>,
}

impl Manifest {
    /// The manifest for a namespace that has been created but never indexed.
    pub fn empty(tokenizer_config_hash: impl Into<String>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            generation: 0,
            wal_index_cursor: 0,
            wal_head: 0,
            files: GenerationFiles {
                pk: String::new(),
                centroids: String::new(),
                clusters: Vec::new(),
                radj_csr: String::new(),
                radj_idx: String::new(),
                fts_split: String::new(),
                stats: String::new(),
            },
            tokenizer_config_hash: tokenizer_config_hash.into(),
            prev_generation: None,
        }
    }

    /// True when no generation has been built yet, so all reads come from the WAL tail.
    pub fn is_empty(&self) -> bool {
        self.generation == 0 && self.files.pk.is_empty()
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
