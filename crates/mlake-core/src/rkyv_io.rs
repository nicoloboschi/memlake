//! Shared rkyv (de)serialization.
//!
//! Every rkyv object memlake stores — WAL entries, cluster files, payload records, patch deltas,
//! centroids — needs the same read dance: tolerate an unaligned input slice (bytes sliced out of a
//! fetched block can start anywhere), *validate* the archive (so a corrupt or truncated object is
//! an error, not undefined behaviour), then deserialize. That dance was copy-pasted at four call
//! sites; this is the one implementation they all share.

use rkyv::validation::validators::DefaultValidator;
use rkyv::{Archive, CheckBytes, Deserialize, Infallible};

/// Serialize `value` to rkyv bytes. rkyv only fails on a broken serializer (not on data), so this
/// returns an empty `Vec` on the impossible error rather than a `Result` the callers can't act on.
pub fn rkyv_write<T>(value: &T) -> Vec<u8>
where
    T: rkyv::Serialize<rkyv::ser::serializers::AllocSerializer<4096>>,
{
    rkyv::to_bytes::<_, 4096>(value)
        .map(|b| b.into_vec())
        .unwrap_or_default()
}

/// Deserialize a `T` from rkyv bytes, validating the buffer and tolerating an unaligned slice.
/// Returns `None` on empty input or any validation / decode failure.
pub fn rkyv_read<T>(bytes: &[u8]) -> Option<T>
where
    T: Archive,
    for<'a> T::Archived: CheckBytes<DefaultValidator<'a>> + Deserialize<T, Infallible>,
{
    if bytes.is_empty() {
        return None;
    }
    if (bytes.as_ptr() as usize).is_multiple_of(8) {
        let archived = rkyv::check_archived_root::<T>(bytes).ok()?;
        Deserialize::<T, _>::deserialize(archived, &mut Infallible).ok()
    } else {
        // The slice is not 8-byte aligned; copy into an aligned buffer rkyv can read in place.
        let mut aligned = rkyv::AlignedVec::with_capacity(bytes.len());
        aligned.extend_from_slice(bytes);
        let archived = rkyv::check_archived_root::<T>(&aligned).ok()?;
        Deserialize::<T, _>::deserialize(archived, &mut Infallible).ok()
    }
}
