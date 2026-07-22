//! A self-describing header for memlake's binary object formats.
//!
//! The bespoke, whole-read binary objects (cluster files, vector blocks, centroid tables) carry a
//! 6-byte header — a 4-byte magic identifying the format plus a little-endian `u16` version — so a
//! reader rejects a wrong-format or wrong-version object *loudly* instead of misparsing raw bytes,
//! and each format gets an independent version to evolve.
//!
//! Scope: this is for objects read whole. The SSTable `.data`/`.csr` blocks are ranged-read by byte
//! offset (a prefix would shift every offset), so they stay raw and are versioned via their `.idx`
//! and the manifest's `format_version`; JSON control-plane objects are self-describing; the FTS
//! split is tantivy's own format.

/// Header length: 4-byte magic + 2-byte little-endian version.
pub const HEADER_LEN: usize = 6;

/// Prefix `payload` with `magic` and `version`.
pub fn wrap(magic: &[u8; 4], version: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Validate the magic and strip the header, returning `(version, payload)`. `None` if the input is
/// too short or the magic does not match — i.e. this is not an object of the expected format.
pub fn unwrap<'a>(magic: &[u8; 4], bytes: &'a [u8]) -> Option<(u16, &'a [u8])> {
    if bytes.len() < HEADER_LEN || &bytes[0..4] != magic {
        return None;
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    Some((version, &bytes[HEADER_LEN..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rejects_mismatch() {
        let framed = wrap(b"CENT", 1, &[1, 2, 3]);
        assert_eq!(unwrap(b"CENT", &framed), Some((1, [1, 2, 3].as_slice())));
        // Wrong magic (e.g. a cluster file read as centroids) is rejected, not misparsed.
        assert_eq!(unwrap(b"CLUS", &framed), None);
        // Truncated / empty inputs are rejected.
        assert_eq!(unwrap(b"CENT", &framed[..3]), None);
        assert_eq!(unwrap(b"CENT", &[]), None);
    }
}
