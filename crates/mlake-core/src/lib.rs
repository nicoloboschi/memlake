//! Shared vocabulary for memlake: ids, item records, the manifest, and the WAL format.
//!
//! This crate deliberately has no I/O and no async. Everything here is a type or a pure
//! function, so the storage, index and query layers can all agree on the on-disk formats
//! without depending on each other.

pub mod id;
pub mod memory;
pub mod manifest;
pub mod wal;

pub use id::MemoryId;
pub use memory::{
    CausalEdge, MemoryType, Memory, LinkType, SemanticEdge, StoredMemory, Timestamps,
    MAX_SEMANTIC_OUT, SEMANTIC_LINK_THRESHOLD,
};
pub use manifest::{GenerationFiles, Manifest};
pub use wal::{Delta, Op, WalEntry};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to encode: {0}")]
    Encode(String),
    #[error("failed to decode: {0}")]
    Decode(String),
    #[error("unsupported format version {found} (this build expects {expected})")]
    FormatVersion { found: u32, expected: u32 },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns 0.0 for a zero-magnitude vector rather than NaN, so a degenerate embedding
/// sorts to the bottom of a result list instead of poisoning comparisons.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine over mismatched dimensions");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
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
    fn normalize_leaves_zero_vector_alone() {
        let mut v = vec![0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
