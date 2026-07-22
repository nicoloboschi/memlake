//! Shared vocabulary for memlake: ids, item records, the manifest, and the WAL format.
//!
//! This crate deliberately has no I/O and no async. Everything here is a type or a pure
//! function, so the storage, index and query layers can all agree on the on-disk formats
//! without depending on each other.

pub mod id;
pub mod memory;
pub mod predicate;
pub mod tags;
pub mod envelope;
pub mod manifest;
pub mod rkyv_io;
pub mod tombstones;
pub mod wal;

pub use id::{EntityId, MemoryId};
pub use rkyv_io::{rkyv_read, rkyv_write};
pub use tombstones::SegmentTombstones;
pub use memory::{
    CausalEdge, MemoryType, Memory, LinkType, SemanticEdge, StoredMemory, Timestamps,
    MAX_SEMANTIC_OUT, SEMANTIC_LINK_THRESHOLD,
};
pub use predicate::Predicate;
pub use tags::{TagFilter, TagsMatch};
pub use manifest::{FactTypeIndex, GenerationFiles, Manifest, Segment};
pub use wal::{apply_delta, apply_deltas, Delta, Op, WalEntry};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to encode: {0}")]
    Encode(String),
    #[error("failed to decode: {0}")]
    Decode(String),
    #[error("on-disk format version {found} is incompatible with this build (expects {expected}); \
             the namespace was written by a different version — delete and re-ingest it")]
    FormatVersion { found: u32, expected: u32 },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns 0.0 for a zero-magnitude vector rather than NaN, so a degenerate embedding
/// sorts to the bottom of a result list instead of poisoning comparisons.
///
/// # Panics
///
/// If the two vectors have different lengths. That is a broken invariant, not bad input:
/// callers validate dimensions at their boundary ([`uniform_dim`]) and return a typed
/// [`Error::DimMismatch`], so a mismatch reaching here means a check was skipped.
///
/// It deliberately does not fall back to comparing the overlapping prefix. Truncating to
/// the shorter side turns "you queried with the wrong embedding model" into a confident,
/// plausible-looking ranking — a silent wrong answer, which is worse than a loud failure.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine over mismatched dimensions");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine similarity where either side may be an *absent* embedding, represented by an
/// empty slice — a memory may legitimately be text-only. An absent embedding scores 0.0:
/// it is simply not retrievable by a vector arm.
///
/// This is the variant to use anywhere an operand comes from stored memories. Absent and
/// wrong-dimension are different failures and must not be conflated: the first is normal
/// and scores zero, the second is a bug and panics in [`cosine`].
pub fn cosine_opt(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    cosine(a, b)
}

/// L2 norm of a vector — for precomputing a query's norm once before a rerank loop.
pub fn norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// [`cosine_opt`] with `a`'s L2 norm precomputed. The rerank scores thousands of candidate
/// vectors against one fixed query, so recomputing the query's norm per candidate (as
/// `cosine` does) is ~a third of the loop's work wasted. `a_norm` must be `norm(a)` — then the
/// result is identical to `cosine_opt(a, b)`. Absent (empty) `b`, or a zero norm on either
/// side, scores 0.0.
///
/// # Panics
/// If `a` and `b` have different non-zero lengths (same invariant as [`cosine`]).
pub fn cosine_opt_prenorm(a: &[f32], a_norm: f32, b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    assert_eq!(a.len(), b.len(), "cosine over mismatched dimensions");
    let mut dot = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        nb += b[i] * b[i];
    }
    let denom = a_norm * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// The single embedding dimension shared by `vectors`, or `None` if none carry one.
/// Absent (empty) embeddings are skipped — a memory may legitimately be text-only.
///
/// Mixed dimensions within one index are an error rather than something to reconcile:
/// there is no correct way to compare a 384-dim query against an 8-dim memory, and every
/// way of faking it produces a confident wrong answer.
pub fn uniform_dim<'a>(
    vectors: impl IntoIterator<Item = &'a [f32]>,
) -> Result<Option<usize>, Error> {
    let mut dim: Option<usize> = None;
    for v in vectors {
        if v.is_empty() {
            continue;
        }
        match dim {
            None => dim = Some(v.len()),
            Some(d) if d == v.len() => {}
            Some(d) => return Err(Error::DimMismatch { expected: d, got: v.len() }),
        }
    }
    Ok(dim)
}

/// Dot product, for callers that maintain pre-normalized vectors and want to skip the
/// magnitude work on the hot path.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        acc += a[i] * b[i];
    }
    acc
}

/// Normalize in place to unit length. A zero vector is left untouched.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_ignores_magnitude() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_zero_vector_is_zero_not_nan() {
        let c = cosine(&[0.0, 0.0], &[1.0, 1.0]);
        assert!(!c.is_nan());
        assert_eq!(c, 0.0);
    }

    #[test]
    fn normalized_vectors_agree_with_cosine_under_dot() {
        let mut a = vec![3.0, 4.0, 0.0];
        let mut b = vec![1.0, 2.0, 2.0];
        let expected = cosine(&a, &b);
        normalize(&mut a);
        normalize(&mut b);
        assert!((dot(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn prenorm_matches_cosine_opt() {
        let cases = [
            (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            (vec![0.3, -0.7, 0.1, 0.9], vec![-0.2, 0.5, 0.5, 0.1]),
            (vec![1.0, 0.0], vec![0.0, 0.0]),   // zero-norm b
        ];
        for (a, b) in cases {
            let expected = cosine_opt(&a, &b);
            let got = cosine_opt_prenorm(&a, norm(&a), &b);
            assert_eq!(got, expected, "prenorm must equal cosine_opt bit-for-bit");
        }
        // absent b (text-only memory) scores 0
        assert_eq!(cosine_opt_prenorm(&[1.0, 2.0], norm(&[1.0, 2.0]), &[]), 0.0);
    }

    #[test]
    fn normalize_leaves_zero_vector_alone() {
        let mut v = vec![0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
