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
pub use tags::{TagFilter, TagPredicate, TagsMatch};
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
/// SIMD dot product over the common prefix of `a` and `b` (8 lanes of `f32` at a time via
/// `wide`, FMA-accumulated, scalar remainder). Every cosine/norm below routes through this, so
/// they share one summation order — which keeps [`cosine_opt_prenorm`] bit-identical to
/// [`cosine`] (a covered invariant) and gets the ~8× speedup uniformly.
#[inline]
fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = wide::f32x8::ZERO;
    for k in 0..chunks {
        let off = k * 8;
        let va = wide::f32x8::from(<[f32; 8]>::try_from(&a[off..off + 8]).unwrap());
        let vb = wide::f32x8::from(<[f32; 8]>::try_from(&b[off..off + 8]).unwrap());
        acc = va.mul_add(vb, acc);
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine over mismatched dimensions");
    let denom = simd_dot(a, a).sqrt() * simd_dot(b, b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        simd_dot(a, b) / denom
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
    simd_dot(a, a).sqrt()
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
    let denom = a_norm * simd_dot(b, b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        simd_dot(a, b) / denom
    }
}

/// [`cosine_opt`] with *both* norms precomputed — for an O(n²) all-pairs loop (link derivation's
/// within-batch compare), where every vector's norm is otherwise recomputed once per row. `a_norm`
/// / `b_norm` must be `norm(a)` / `norm(b)`. Absent (empty) side or a zero norm scores 0.0.
pub fn cosine_prenorm_both(a: &[f32], a_norm: f32, b: &[f32], b_norm: f32) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    assert_eq!(a.len(), b.len(), "cosine over mismatched dimensions");
    let denom = a_norm * b_norm;
    if denom == 0.0 {
        0.0
    } else {
        simd_dot(a, b) / denom
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
    simd_dot(a, b)
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

    // Micro-benchmark: scalar cosine vs the SIMD+prenorm path, on 384-dim vectors, over an
    // all-pairs loop like link derivation's within-batch compare. Run with:
    //   cargo test -p mlake-core --release bench_cosine -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_cosine() {
        use std::time::Instant;
        const DIM: usize = 384;
        const N: usize = 2000;
        const ROWS: usize = 200; // ROWS × N pairs, to keep it quick
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(N);
        let mut s: u64 = 0x1234_5678;
        for _ in 0..N {
            let mut v = vec![0f32; DIM];
            for x in v.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *x = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
            }
            vecs.push(v);
        }
        fn scalar_cos(a: &[f32], b: &[f32]) -> f32 {
            let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
            for i in 0..a.len() {
                d += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            let den = na.sqrt() * nb.sqrt();
            if den == 0.0 { 0.0 } else { d / den }
        }
        let t = Instant::now();
        let mut acc = 0f32;
        for i in 0..ROWS {
            for j in 0..N {
                acc += scalar_cos(&vecs[i], &vecs[j]);
            }
        }
        let scalar_ms = t.elapsed().as_secs_f64() * 1000.0;

        let norms: Vec<f32> = vecs.iter().map(|v| norm(v)).collect();
        let t = Instant::now();
        let mut acc2 = 0f32;
        for i in 0..ROWS {
            for j in 0..N {
                acc2 += cosine_prenorm_both(&vecs[i], norms[i], &vecs[j], norms[j]);
            }
        }
        let simd_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "cosine {DIM}-dim, {ROWS}x{N} pairs: scalar={scalar_ms:.1}ms  simd+prenorm={simd_ms:.1}ms  speedup={:.1}x  (checksums {:.3}/{:.3})",
            scalar_ms / simd_ms,
            acc,
            acc2
        );
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
