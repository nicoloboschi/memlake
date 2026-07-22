//! Vector blocks: the scan-facing half of a cluster.
//!
//! A cluster's vectors live here, split out of the payload, so a probe reads only the bytes
//! it actually scores (docs/vector-storage.md, Phase 1). The embedding is 84% of a stored
//! memory, and the scan reads it once and discards it — pulling it into its own contiguous,
//! member-ordered block is what turns "fetch 1834 B per candidate" into "fetch 60".
//!
//! On top of the layout sits the codec (Phases 2 and 3). Quantized codes are *estimates*:
//! the block ranks candidates cheaply and the caller reranks the survivors at full
//! precision. That split is why the estimators only have to preserve the *ordering* of the
//! top region, not the exact scores — but they do have to preserve it, so every codec here
//! is measured against the f32 ranking in the tests below rather than assumed to be fine.
//!
//! Everything is pure computation: no I/O, no async. The storage layer owns where these
//! bytes live.

use mlake_core::MemoryId;

/// How one cluster's vectors are encoded on disk.
///
/// The choice is per block rather than per namespace so a rewrite can change codec one
/// cluster at a time, and so a full-precision rerank block can sit beside a quantized scan
/// block without either knowing about the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VectorCodec {
    /// Raw little-endian `f32`. Exact, and the baseline the others are measured against.
    F32,
    /// Per-vector affine scalar quantization of the block-mean-centred residual to 8 bits.
    /// ~3.9x smaller than [`VectorCodec::F32`].
    Int8,
    /// One sign bit per dimension over the block-mean-centred residual, plus a per-vector
    /// corrective term. ~25.6x smaller than [`VectorCodec::F32`] at dim 384.
    Binary,
}

impl VectorCodec {
    /// The on-disk discriminant. Fixed forever: it is in the header of every block written.
    fn tag(self) -> u8 {
        match self {
            VectorCodec::F32 => 0,
            VectorCodec::Int8 => 1,
            VectorCodec::Binary => 2,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(VectorCodec::F32),
            1 => Some(VectorCodec::Int8),
            2 => Some(VectorCodec::Binary),
            _ => None,
        }
    }
}

/// Magic prefix, so a block fetched from the wrong key fails loudly instead of decoding
/// into plausible garbage.
const MAGIC: [u8; 4] = *b"MLVB";

/// Bumped only for an incompatible layout change; `from_bytes` rejects anything else.
const FORMAT_VERSION: u8 = 1;

/// `magic | version | codec | reserved | dim | count`.
const HEADER_LEN: usize = 16;

/// A [`MemoryId`] is a raw 16-byte UUID.
const ID_LEN: usize = 16;

/// Per-vector corrective floats carried by the quantized codecs, ahead of the codes.
///
/// [`VectorCodec::Int8`] stores `offset`, `scale` and the true L2 norm;
/// [`VectorCodec::Binary`] stores the residual norm, the code's cosine to its residual, and
/// the true L2 norm. Both are three `f32`.
const CORRECTIVE_LEN: usize = 12;

/// One cluster's vectors, split out of the payload so a probe reads only what it scores.
///
/// Members are addressed by position, and that position is the contract with the sibling
/// payload block: index `i` here is index `i` there. `ids` is carried so a scan can emit
/// hits without a second fetch.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct VectorBlock {
    codec: Option<VectorCodec>,
    dim: usize,
    ids: Vec<MemoryId>,
    /// Quantized codecs only: the arithmetic mean of the block's vectors.
    ///
    /// Both quantized codecs encode the *residual* `v - mean`, not `v`. Real embeddings
    /// share a large common component, and within one IVF cluster — which is exactly what
    /// one block holds — they share more of one still. Quantizing the raw vector spends
    /// every code level describing that shared part, which is precisely the part that
    /// cannot separate two members. The mean's contribution to every score is then computed
    /// *exactly*, once per query, and only the residual is estimated.
    ///
    /// It costs `dim * 4` bytes once per block — about 3 B/vector at 500 members — and it is
    /// the single largest factor in both codecs' accuracy (see the tests).
    mean: Vec<f32>,
    /// Codes for every member, contiguous and fixed-stride: member `i` starts at
    /// `i * bytes_per_vector(codec, dim)`.
    codes: Vec<u8>,
}

/// A query vector prepared once per query for a given codec + dim.
///
/// The query is kept at full `f32` precision against 8-bit or 1-bit stored codes. This is
/// the asymmetric encoding RaBitQ and BBQ use, taken to its limit: the query is one vector,
/// so its precision is free, and every bit of it removes error from every comparison. It
/// costs nothing in speed either — against 1-bit codes the dot product degenerates to a
/// signed accumulation with no multiply, and against 8-bit codes to a `f32 * u8` product.
#[derive(Clone, PartialEq, Debug)]
pub struct PreparedQuery {
    codec: Option<VectorCodec>,
    dim: usize,
    /// The query itself, full precision.
    q: Vec<f32>,
    /// `|q|`, the cosine denominator's query half.
    norm: f32,
    /// `sum(q)`, the coefficient of the affine offset in the [`VectorCodec::Int8`] estimate.
    sum: f32,
    /// `<q, mean>` — the exact contribution of the block centroid to every member's dot
    /// product, computed once instead of estimated.
    mean_dot: f32,
    /// `|mean|`.
    mean_norm: f32,
    /// `<q, mean/|mean|>`.
    along_mean: f32,
    /// `q` with its component along the block mean removed. [`VectorCodec::Binary`] only —
    /// see [`score_binary`] for why the estimator runs on this rather than on `q`.
    perp: Vec<f32>,
    /// `|perp|`.
    perp_norm: f32,
}

impl VectorBlock {
    /// Encode `vectors` (all of length `dim`, aligned 1:1 with `ids`) under `codec`.
    ///
    /// Deterministic: the same input produces byte-identical output (G-6).
    pub fn encode(
        codec: VectorCodec,
        dim: usize,
        ids: &[MemoryId],
        vectors: &[Vec<f32>],
    ) -> Result<Self, mlake_core::Error> {
        if ids.len() != vectors.len() {
            return Err(mlake_core::Error::Encode(format!(
                "{} ids for {} vectors: a vector block is positional",
                ids.len(),
                vectors.len()
            )));
        }
        for v in vectors {
            if v.len() != dim {
                return Err(mlake_core::Error::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let mean = if codec == VectorCodec::F32 {
            Vec::new()
        } else {
            block_mean(dim, vectors)
        };

        let stride = Self::bytes_per_vector(codec, dim);
        let mut codes = Vec::with_capacity(stride * vectors.len());
        let mut residual = Vec::with_capacity(dim);
        for v in vectors {
            match codec {
                VectorCodec::F32 => {
                    for x in v {
                        codes.extend_from_slice(&x.to_le_bytes());
                    }
                }
                VectorCodec::Int8 | VectorCodec::Binary => {
                    residual.clear();
                    residual.extend(v.iter().zip(&mean).map(|(x, m)| x - m));
                    if codec == VectorCodec::Int8 {
                        encode_int8(&residual, mlake_core::norm(v), &mut codes);
                    } else {
                        encode_binary(&residual, mlake_core::norm(v), &mut codes);
                    }
                }
            }
        }
        debug_assert_eq!(codes.len(), stride * vectors.len());

        Ok(Self {
            codec: Some(codec),
            dim,
            ids: ids.to_vec(),
            mean,
            codes,
        })
    }

    /// Serialize to bytes. Self-describing: codec, dim and count are in the header.
    ///
    /// `[magic 4][version 1][codec 1][reserved 2][dim u32][count u32]`, then `count` ids,
    /// then the block mean (quantized codecs only), then the codes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let codec = self.codec();
        let stride = Self::bytes_per_vector(codec, self.dim);
        let mut out =
            Vec::with_capacity(HEADER_LEN + self.ids.len() * ID_LEN + self.codes.len() + self.mean.len() * 4);
        out.extend_from_slice(&MAGIC);
        out.push(FORMAT_VERSION);
        out.push(codec.tag());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        for id in &self.ids {
            out.extend_from_slice(&id.0);
        }
        for x in &self.mean {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&self.codes);
        debug_assert_eq!(out.len(), HEADER_LEN + self.ids.len() * (ID_LEN + stride) + self.mean.len() * 4);
        out
    }

    /// Parse. Must reject any malformed or truncated input with an error, never panic and
    /// never index out of bounds.
    ///
    /// The length is checked exactly rather than as a lower bound: a block with trailing
    /// bytes is a block whose writer disagreed with this reader about the layout, and
    /// silently ignoring the tail is how that disagreement becomes a wrong answer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, mlake_core::Error> {
        let bad = |why: &str| mlake_core::Error::Decode(format!("vector block: {why}"));

        if bytes.len() < HEADER_LEN {
            return Err(bad(&format!("{} bytes is shorter than the header", bytes.len())));
        }
        if bytes[0..4] != MAGIC {
            return Err(bad("bad magic"));
        }
        if bytes[4] != FORMAT_VERSION {
            return Err(mlake_core::Error::FormatVersion {
                found: bytes[4] as u32,
                expected: FORMAT_VERSION as u32,
            });
        }
        let codec = VectorCodec::from_tag(bytes[5])
            .ok_or_else(|| bad(&format!("unknown codec {}", bytes[5])))?;
        let dim = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

        let mean_len = if codec == VectorCodec::F32 { 0 } else { dim };
        // Every product is checked: `dim` and `count` come off the network and a 32-bit
        // pair multiplies well past usize on a 32-bit target.
        let ids_end = count
            .checked_mul(ID_LEN)
            .and_then(|n| n.checked_add(HEADER_LEN))
            .ok_or_else(|| bad("id table overflows"))?;
        let mean_end = mean_len
            .checked_mul(4)
            .and_then(|n| n.checked_add(ids_end))
            .ok_or_else(|| bad("mean overflows"))?;
        let total = count
            .checked_mul(Self::bytes_per_vector(codec, dim))
            .and_then(|n| n.checked_add(mean_end))
            .ok_or_else(|| bad("code table overflows"))?;
        if bytes.len() != total {
            return Err(bad(&format!(
                "declares {count} x dim {dim} under {codec:?} ({total} bytes), got {}",
                bytes.len()
            )));
        }

        let ids = bytes[HEADER_LEN..ids_end]
            .chunks_exact(ID_LEN)
            .map(|c| MemoryId(c.try_into().expect("chunks_exact yields 16 bytes")))
            .collect();
        let mean = bytes[ids_end..mean_end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(Self {
            codec: Some(codec),
            dim,
            ids,
            mean,
            codes: bytes[mean_end..].to_vec(),
        })
    }

    /// The codec this block's members are stored under.
    pub fn codec(&self) -> VectorCodec {
        // A default-constructed block holds nothing, so any codec describes it correctly.
        self.codec.unwrap_or(VectorCodec::F32)
    }

    /// The dimension every member has.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the block holds no members.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The member ids, in the same order as the codes.
    pub fn ids(&self) -> &[MemoryId] {
        &self.ids
    }

    /// Prepare a query once, then score many members with it.
    ///
    /// Everything that depends only on the query lives here — its norm, its sum, and its
    /// exact dot with the block mean — so the per-member loop is one pass over the codes.
    pub fn prepare(&self, query: &[f32]) -> Result<PreparedQuery, mlake_core::Error> {
        if query.len() != self.dim {
            return Err(mlake_core::Error::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mean_norm = mlake_core::norm(&self.mean);
        let mean_dot = mlake_core::dot(query, &self.mean);
        let along_mean = if mean_norm > 0.0 { mean_dot / mean_norm } else { 0.0 };
        let perp = if self.codec == Some(VectorCodec::Binary) {
            if mean_norm > 0.0 {
                let k = along_mean / mean_norm;
                query.iter().zip(&self.mean).map(|(x, m)| x - k * m).collect()
            } else {
                query.to_vec()
            }
        } else {
            Vec::new()
        };
        Ok(PreparedQuery {
            codec: self.codec,
            dim: self.dim,
            norm: mlake_core::norm(query),
            sum: query.iter().sum(),
            mean_dot,
            mean_norm,
            along_mean,
            perp_norm: mlake_core::norm(&perp),
            perp,
            q: query.to_vec(),
        })
    }

    /// Estimated cosine similarity between the prepared query and member `i`.
    /// For `F32` this is exact. For `Int8`/`Binary` it is an estimate.
    ///
    /// Returns 0.0 — the score of an absent embedding elsewhere in the codebase — for an
    /// out-of-range `i`, a degenerate (zero-norm) vector on either side, or a query
    /// prepared against a different block's codec or dim.
    pub fn score(&self, q: &PreparedQuery, i: usize) -> f32 {
        debug_assert_eq!(q.codec, self.codec, "query prepared for a different codec");
        debug_assert_eq!(q.dim, self.dim, "query prepared for a different dim");
        if i >= self.len() || q.dim != self.dim || q.codec != self.codec || q.norm == 0.0 {
            return 0.0;
        }
        let codec = self.codec();
        let stride = Self::bytes_per_vector(codec, self.dim);
        let Some(code) = self.codes.get(i * stride..(i + 1) * stride) else {
            return 0.0;
        };
        match codec {
            VectorCodec::F32 => score_f32(code, q),
            VectorCodec::Int8 => score_int8(code, q),
            VectorCodec::Binary => score_binary(code, self.dim, q),
        }
    }

    /// Top `k` members by estimated score, best first, ties broken by id so results are
    /// deterministic. Returns `(index, score)` pairs.
    pub fn top_k(&self, q: &PreparedQuery, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> =
            (0..self.len()).map(|i| (i, self.score(q, i))).collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(self.ids[a.0].cmp(&self.ids[b.0]))
        });
        scored.truncate(k);
        scored
    }

    /// Decode member `i` back to f32. Lossy for the quantized codecs; exact for `F32`.
    ///
    /// `Binary` returns the block mean plus the best reconstruction of the residual that a
    /// sign pattern admits — its projection onto the code direction — so the result has the
    /// right direction and a shrunk magnitude, not the right components.
    ///
    /// Returns an empty vector for an out-of-range `i`, matching the "absent embedding"
    /// convention of [`mlake_core::cosine_opt`].
    pub fn decode(&self, i: usize) -> Vec<f32> {
        if i >= self.len() {
            return Vec::new();
        }
        let codec = self.codec();
        let stride = Self::bytes_per_vector(codec, self.dim);
        let Some(code) = self.codes.get(i * stride..(i + 1) * stride) else {
            return Vec::new();
        };
        match codec {
            VectorCodec::F32 => code
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            VectorCodec::Int8 => {
                let (offset, scale, _) = correctives(code);
                code[CORRECTIVE_LEN..]
                    .iter()
                    .zip(&self.mean)
                    .map(|(c, m)| m + offset + scale * *c as f32)
                    .collect()
            }
            VectorCodec::Binary => {
                let (r_norm, c, _) = correctives(code);
                let bits = &code[CORRECTIVE_LEN..];
                // The residual's projection onto its own code: |r| * cos(r, u) * u, with
                // u = b / sqrt(d) the unit sign vector.
                let amp = r_norm * c / (self.dim as f32).sqrt();
                (0..self.dim)
                    .map(|j| {
                        let s = if bit_at(bits, j) { amp } else { -amp };
                        self.mean.get(j).copied().unwrap_or(0.0) + s
                    })
                    .collect()
            }
        }
    }

    /// Bytes one vector occupies under this codec, excluding the block header.
    ///
    /// "Header" here means everything that is not a member's own codes: the fixed 16-byte
    /// prelude, the id table (16 B/member — see the module tests, it dominates the binary
    /// block) and, for the quantized codecs, the one shared mean vector.
    pub fn bytes_per_vector(codec: VectorCodec, dim: usize) -> usize {
        match codec {
            VectorCodec::F32 => dim * 4,
            VectorCodec::Int8 => dim + CORRECTIVE_LEN,
            VectorCodec::Binary => dim.div_ceil(8) + CORRECTIVE_LEN,
        }
    }
}

/// The arithmetic mean of `vectors`, accumulated in f64 so a large cluster's mean does not
/// drift with member order (G-6 wants byte-identical replays).
fn block_mean(dim: usize, vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut acc = vec![0.0f64; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v) {
            *a += *x as f64;
        }
    }
    let n = vectors.len().max(1) as f64;
    acc.into_iter().map(|a| (a / n) as f32).collect()
}

/// `[offset | scale | norm]` or `[residual norm | code cosine | norm]`, per codec.
fn correctives(code: &[u8]) -> (f32, f32, f32) {
    let f = |o: usize| f32::from_le_bytes([code[o], code[o + 1], code[o + 2], code[o + 3]]);
    (f(0), f(4), f(8))
}

fn bit_at(bits: &[u8], j: usize) -> bool {
    bits[j / 8] & (1 << (j % 8)) != 0
}

/// Affine per-vector quantization of the residual: `r ~= offset + scale * code`, `code` a
/// `u8`. `v_norm` is the *original* vector's L2 norm, kept exact for the cosine denominator.
///
/// The scale comes from the residual's own min and max rather than a global constant. bge
/// embeddings are L2-normalized, which would make a shared scale behave — but `uniform_dim`
/// exists precisely because callers do not always send what we expect, and a vector that
/// arrives unnormalized must quantize as well as one that does.
fn encode_int8(r: &[f32], v_norm: f32, out: &mut Vec<u8>) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for x in r {
        min = min.min(*x);
        max = max.max(*x);
    }
    if r.is_empty() {
        min = 0.0;
        max = 0.0;
    }
    // A constant residual has no range to spread over 256 levels; scale 0 makes every code
    // dequantize back to `offset`, which is exactly right.
    let scale = if max > min { (max - min) / 255.0 } else { 0.0 };
    out.extend_from_slice(&min.to_le_bytes());
    out.extend_from_slice(&scale.to_le_bytes());
    out.extend_from_slice(&v_norm.to_le_bytes());
    for x in r {
        let code = if scale > 0.0 {
            (((*x - min) / scale).round()).clamp(0.0, 255.0) as u8
        } else {
            0
        };
        out.push(code);
    }
}

/// One sign bit per dimension of the block-mean-centred residual `r`, plus the two
/// corrective terms the estimator needs (see [`score_binary`]) and the original vector's
/// true norm.
fn encode_binary(r: &[f32], v_norm: f32, out: &mut Vec<u8>) {
    let residual = r;
    let r_norm = mlake_core::norm(residual);
    let d = residual.len() as f32;
    // cos(r, u) where u = sign(r)/sqrt(d): the fraction of the residual the sign pattern
    // captures. `sum |r_j| / sqrt(d)` is <r, u> because u's components are +-1/sqrt(d).
    let abs_sum: f32 = residual.iter().map(|x| x.abs()).sum();
    let c = if r_norm > 0.0 && d > 0.0 {
        abs_sum / (d.sqrt() * r_norm)
    } else {
        0.0
    };
    out.extend_from_slice(&r_norm.to_le_bytes());
    out.extend_from_slice(&c.to_le_bytes());
    out.extend_from_slice(&v_norm.to_le_bytes());

    let mut bits = vec![0u8; residual.len().div_ceil(8)];
    for (j, x) in residual.iter().enumerate() {
        // sign(0) = +1, so a zero residual still has a defined code.
        if *x >= 0.0 {
            bits[j / 8] |= 1 << (j % 8);
        }
    }
    out.extend_from_slice(&bits);
}

fn score_f32(code: &[u8], q: &PreparedQuery) -> f32 {
    let mut dot = 0.0f32;
    let mut nv = 0.0f32;
    for (c, qj) in code.chunks_exact(4).zip(&q.q) {
        let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        dot += x * qj;
        nv += x * x;
    }
    let denom = q.norm * nv.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// `cos(q, v) ~= (<q, mean> + offset * sum(q) + scale * <q, code>) / (|q| * |v|)`.
///
/// The numerator is the exact dot product of `q` with `mean + dequantized residual`,
/// expanded so the loop never materializes anything: `<q, mean>` and `sum(q)` are
/// precomputed per query and the remaining term is an `f32 * u8` accumulation. The
/// denominator uses the *stored true* norm, not the dequantized one, so the only error left
/// is the quantization of the residual's codes.
fn score_int8(code: &[u8], q: &PreparedQuery) -> f32 {
    let (offset, scale, v_norm) = correctives(code);
    let mut acc = 0.0f32;
    for (c, qj) in code[CORRECTIVE_LEN..].iter().zip(&q.q) {
        acc += *c as f32 * qj;
    }
    let denom = q.norm * v_norm;
    if denom == 0.0 {
        0.0
    } else {
        ((q.mean_dot + offset * q.sum + scale * acc) / denom).clamp(-1.0, 1.0)
    }
}

/// A RaBitQ-family estimator, applied to the part of the vector the 1-bit code is the only
/// evidence for — and *only* that part. Everything else in the dot product is exact.
///
/// Write `v = m + r`, with `m` the block mean and `r` the residual whose signs are stored,
/// and split both `r` and the query along `m̂ = m/|m|`:
///
/// ```text
///   r = a * m̂ + r_perp,          a  = <r, m̂>
///   q = <q, m̂> * m̂ + q_perp
///   <q, v> = <q, m> + a * <q, m̂> + <q_perp, r_perp>
/// ```
///
/// The first two terms are **exact**, for no extra bytes:
/// * `<q, m>` and `<q, m̂>` are computed once per query in [`VectorBlock::prepare`];
/// * `a` comes out of the stored norms by `|v|^2 = |m|^2 + 2*<m, r> + |r|^2`, so
///   `a = (|v|^2 - |m|^2 - |r|^2) / (2 |m|)`.
///
/// Only the last term is estimated, and that is RaBitQ's estimator with `u = b/sqrt(d)`
/// the unit vector of the stored signs:
///
/// ```text
///   <q_perp, r_perp>  ~=  |q_perp| * |r| * <q_perp/|q_perp|, u> / cos(r, u)
///                      =  |r| * <q_perp, b> / (sqrt(d) * cos(r, u))
/// ```
///
/// Its derivation: decompose the unit `q_perp` into its component along `r` and a remainder
/// `w`, giving `<q_perp/|q_perp|, u> = cos(q_perp, r) * cos(r, u) + <w, u>`; dividing by the
/// stored `cos(r, u)` recovers `cos(q_perp, r)` up to `<w, u>`, which has mean zero when the
/// code's error direction is uncorrelated with the query. RaBitQ *guarantees* that with a
/// random rotation; we *assume* it, which is the estimator's one soft spot on adversarial
/// data (see the tests for what it costs on anisotropic input).
///
/// Running the estimator on `q_perp` rather than on `q` is the single largest accuracy
/// lever in this module, and the reason is the error term, not the signal: the estimator's
/// noise scales with the norm of the query vector fed to it. `|q|` is dominated by the block
/// mean — every member is near it, that is what a cluster is — while `|q_perp|` is the part
/// of the query that can actually distinguish one member from another. Feeding it `q` makes
/// the noise proportional to the *shared* component, which is precisely the component whose
/// contribution we already know exactly. Measured, this triples recall@10 (see the tests).
///
/// The clamps are because the estimator is unbiased, not bounded: a member whose code
/// happens to align with the query can estimate a cosine above 1 and sort above a genuine
/// exact match.
fn score_binary(code: &[u8], dim: usize, q: &PreparedQuery) -> f32 {
    let (r_norm, c, v_norm) = correctives(code);
    let denom = q.norm * v_norm;
    if denom == 0.0 || dim == 0 {
        return 0.0;
    }
    // The component of the residual along the mean, recovered exactly from the norms.
    let along = if q.mean_norm > 0.0 {
        (v_norm * v_norm - q.mean_norm * q.mean_norm - r_norm * r_norm) / (2.0 * q.mean_norm)
    } else {
        0.0
    };
    let known = q.mean_dot + along * q.along_mean;
    if c <= 0.0 || q.perp_norm == 0.0 {
        // Nothing is left to estimate: either the member sits exactly on the block mean, or
        // the query has no component off it. Either way `known` is the whole answer.
        return (known / denom).clamp(-1.0, 1.0);
    }
    let bits = &code[CORRECTIVE_LEN..];
    // <q_perp, b>, a signed accumulation: a 1-bit code needs no multiply.
    let mut acc = 0.0f32;
    for (j, x) in q.perp.iter().enumerate() {
        if bit_at(bits, j) {
            acc += x;
        } else {
            acc -= x;
        }
    }
    let sqrt_d = (dim as f32).sqrt();
    let cos_est = (acc / (q.perp_norm * sqrt_d * c)).clamp(-1.0, 1.0);
    ((known + q.perp_norm * r_norm * cos_est) / denom).clamp(-1.0, 1.0)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::Rng;

    const DIM: usize = 384;
    const N: usize = 500;
    const QUERIES: usize = 60;
    /// Sub-clusters inside the block. A block is one IVF cluster, and a real corpus keeps
    /// clustering below the level k-means stopped at — topics inside a topic.
    const TOPICS: usize = 12;

    // Every threshold below is the number actually measured, rounded down. They are
    // constants rather than literals in the asserts so a regression names itself.

    /// Recall@10 of the [`VectorCodec::Int8`] ranking against the exact f32 ranking, over
    /// `N` vectors at dim `DIM`. Measured: 0.982 — Elastic's "int8 costs ~2% recall",
    /// reproduced.
    const INT8_RECALL_AT_10: f32 = 0.95;
    /// Spearman correlation of the int8 scores with the exact scores. Measured: 0.9999.
    const INT8_RANK_CORRELATION: f32 = 0.999;

    /// Recall@10 of the [`VectorCodec::Binary`] ranking, same corpus. Measured: 0.515.
    ///
    /// This is *not* a gate anyone should serve from: 1 bit/dim cannot resolve the top ten
    /// of a cluster, and no amount of estimator work will change that (the arithmetic is in
    /// the test that pins it). Binary is a candidate generator — see
    /// [`BINARY_RECALL_AT_10_OVERSAMPLED`], which is the number Phase 3 actually depends on.
    const BINARY_RECALL_AT_10: f32 = 0.48;
    /// Fraction of the true top 10 present in binary's top 40 — 4x oversampling, the size of
    /// the set Phase 3 would hand to a full-precision rerank. Measured: 1.000.
    const BINARY_RECALL_AT_10_OVERSAMPLED: f32 = 0.99;
    /// Spearman correlation of the binary scores with the exact scores. Measured: 0.866.
    const BINARY_RANK_CORRELATION: f32 = 0.85;

    /// Recall@10 of a naive Hamming ranking — sign of the *raw* vector, no centring and no
    /// corrective, i.e. SimHash — on the same corpus. Measured: 0.385.
    const NAIVE_HAMMING_RECALL_AT_10: f32 = 0.40;

    fn ids(n: usize) -> Vec<MemoryId> {
        (0..n).map(|i| MemoryId::from_key(&format!("m{i}"))).collect()
    }

    /// Approximately normal: six uniforms, Irwin-Hall. Deterministic and seeded — SPEC §10.3
    /// G-6 wants replayable builds, so no `rand` and no wall clock anywhere in this module.
    fn gauss(rng: &mut Rng) -> f32 {
        (0..6).map(|_| rng.unit_f32()).sum::<f32>() - 3.0
    }

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        mlake_core::normalize(&mut v);
        v
    }

    /// The default corpus: one block's worth of vectors with the structure a real one has.
    fn block_corpus(seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        corpus(N, DIM, TOPICS, 0.5, seed)
    }

    /// Vectors shaped like one IVF cluster of a real embedding model's output.
    ///
    /// * a shared mean of norm `global_scale`, because embedding spaces are markedly
    ///   anisotropic — two unrelated bge vectors sit at cosine ~0.3-0.8, never ~0;
    /// * `topics` sub-centres around it, because a corpus keeps clustering below the level
    ///   k-means stopped at;
    /// * per-member noise around those;
    /// * all L2-normalized, as bge output is.
    ///
    /// Queries are drawn from the same sub-centres with slightly more noise, so a query has
    /// genuine near-neighbours rather than an arbitrary ordering of equidistant points.
    /// `topics = 1` removes that structure and is the isotropic worst case; it is tested
    /// separately and deliberately.
    fn corpus(
        n: usize,
        dim: usize,
        topics: usize,
        global_scale: f32,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = Rng::seeded(seed);
        let global: Vec<f32> = (0..dim).map(|_| global_scale * gauss(&mut rng)).collect();
        let centre: Vec<f32> = global.iter().map(|g| g + 0.6 * gauss(&mut rng)).collect();
        let topics: Vec<Vec<f32>> = (0..topics.max(1))
            .map(|_| centre.iter().map(|x| x + 0.30 * gauss(&mut rng)).collect())
            .collect();

        let mut vectors = Vec::with_capacity(n);
        for i in 0..n {
            let t: &Vec<f32> = &topics[i % topics.len()];
            vectors.push(unit(t.iter().map(|x| x + 0.20 * gauss(&mut rng)).collect()));
        }
        let queries = (0..QUERIES)
            .map(|i| {
                let t: &Vec<f32> = &topics[i % topics.len()];
                unit(t.iter().map(|x| x + 0.22 * gauss(&mut rng)).collect())
            })
            .collect();
        (vectors, queries)
    }

    fn top_ids(block: &VectorBlock, query: &[f32], k: usize) -> Vec<MemoryId> {
        let prepared = block.prepare(query).unwrap();
        block
            .top_k(&prepared, k)
            .into_iter()
            .map(|(i, _)| block.ids()[i])
            .collect()
    }

    /// Mean fraction of the exact top 10 that `codec`'s top `candidates` contains.
    /// `candidates = 10` is recall@10; larger values are what an oversampling rerank sees.
    fn recall_into(
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        codec: VectorCodec,
        candidates: usize,
    ) -> f32 {
        let ids = ids(vectors.len());
        let exact = VectorBlock::encode(VectorCodec::F32, DIM, &ids, vectors).unwrap();
        let approx = VectorBlock::encode(codec, DIM, &ids, vectors).unwrap();
        let mut total = 0.0;
        for q in queries {
            let truth = top_ids(&exact, q, 10);
            let got = top_ids(&approx, q, candidates);
            total += truth.iter().filter(|id| got.contains(id)).count() as f32 / 10.0;
        }
        total / queries.len() as f32
    }

    /// SimHash: sign of the raw vector, ranked by sign agreement with the query's signs.
    /// The thing every 1-bit scheme has to beat to justify its corrective bytes.
    fn naive_hamming_recall(
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        candidates: usize,
    ) -> f32 {
        let ids = ids(vectors.len());
        let exact = VectorBlock::encode(VectorCodec::F32, DIM, &ids, vectors).unwrap();
        let signs: Vec<Vec<bool>> = vectors
            .iter()
            .map(|v| v.iter().map(|x| *x >= 0.0).collect())
            .collect();
        let mut total = 0.0;
        for q in queries {
            let qs: Vec<bool> = q.iter().map(|x| *x >= 0.0).collect();
            let mut scored: Vec<(usize, i32)> = signs
                .iter()
                .enumerate()
                .map(|(i, s)| (i, s.iter().zip(&qs).filter(|(a, b)| a == b).count() as i32))
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1).then(ids[a.0].cmp(&ids[b.0])));
            let truth = top_ids(&exact, q, 10);
            total += scored[..candidates]
                .iter()
                .filter(|(i, _)| truth.contains(&ids[*i]))
                .count() as f32
                / 10.0;
        }
        total / queries.len() as f32
    }

    /// Spearman rank correlation. The inputs are continuous scores, so exact ties are
    /// vanishingly rare and are broken by position rather than averaged.
    fn spearman(a: &[f32], b: &[f32]) -> f32 {
        fn ranks(v: &[f32]) -> Vec<f32> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|x, y| v[*x].partial_cmp(&v[*y]).unwrap());
            let mut r = vec![0.0; v.len()];
            for (rank, i) in idx.into_iter().enumerate() {
                r[i] = rank as f32;
            }
            r
        }
        let (ra, rb) = (ranks(a), ranks(b));
        let mean = (a.len() as f32 - 1.0) / 2.0;
        let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
        for i in 0..a.len() {
            let (x, y) = (ra[i] - mean, rb[i] - mean);
            num += x * y;
            da += x * x;
            db += y * y;
        }
        num / (da.sqrt() * db.sqrt())
    }

    fn rank_correlation(codec: VectorCodec) -> f32 {
        let (vectors, queries) = block_corpus(7);
        let ids = ids(vectors.len());
        let exact = VectorBlock::encode(VectorCodec::F32, DIM, &ids, &vectors).unwrap();
        let approx = VectorBlock::encode(codec, DIM, &ids, &vectors).unwrap();
        let mut total = 0.0;
        for q in &queries {
            let pe = exact.prepare(q).unwrap();
            let pa = approx.prepare(q).unwrap();
            let se: Vec<f32> = (0..exact.len()).map(|i| exact.score(&pe, i)).collect();
            let sa: Vec<f32> = (0..approx.len()).map(|i| approx.score(&pa, i)).collect();
            total += spearman(&se, &sa);
        }
        total / queries.len() as f32
    }

    // --- format -----------------------------------------------------------------------

    #[test]
    fn f32_round_trips_every_component_exactly() {
        let v = vec![vec![1.0, -0.5, 0.25], vec![0.0, 2.5e-8, -3.75e12]];
        let block = VectorBlock::encode(VectorCodec::F32, 3, &ids(2), &v).unwrap();
        let back = VectorBlock::from_bytes(&block.to_bytes()).unwrap();
        assert_eq!(back, block);
        assert_eq!(back.decode(0), v[0]);
        assert_eq!(back.decode(1), v[1]);
        assert_eq!(back.ids(), ids(2));
        assert_eq!(back.codec(), VectorCodec::F32);
        assert_eq!(back.dim(), 3);
        assert_eq!(back.len(), 2);
    }

    #[test]
    fn f32_scores_exactly_what_cosine_does() {
        let v = vec![vec![1.0, 0.0, 0.0], vec![0.7, 0.7, 0.1]];
        let block = VectorBlock::encode(VectorCodec::F32, 3, &ids(2), &v).unwrap();
        let q = vec![0.9, 0.3, 0.0];
        let prepared = block.prepare(&q).unwrap();
        for (i, vi) in v.iter().enumerate() {
            assert!(
                (block.score(&prepared, i) - mlake_core::cosine(&q, vi)).abs() < 1e-6,
                "f32 is the baseline: it must not merely approximate cosine"
            );
        }
    }

    #[test]
    fn every_codec_round_trips_through_bytes() {
        let (vectors, _) = corpus(40, DIM, TOPICS, 0.5, 3);
        let ids = ids(40);
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, DIM, &ids, &vectors).unwrap();
            let back = VectorBlock::from_bytes(&block.to_bytes()).unwrap();
            assert_eq!(back, block, "{codec:?} must survive a round trip");
        }
    }

    #[test]
    fn encode_rejects_a_vector_whose_dim_disagrees_with_the_block() {
        let err = VectorBlock::encode(
            VectorCodec::Int8,
            3,
            &ids(2),
            &[vec![1.0, 2.0, 3.0], vec![1.0, 2.0]],
        )
        .unwrap_err();
        assert!(
            matches!(err, mlake_core::Error::DimMismatch { expected: 3, got: 2 }),
            "got {err:?}"
        );
    }

    #[test]
    fn encode_rejects_an_id_count_that_disagrees_with_the_vector_count() {
        let err = VectorBlock::encode(VectorCodec::F32, 2, &ids(3), &[vec![1.0, 2.0]]).unwrap_err();
        assert!(matches!(err, mlake_core::Error::Encode(_)), "got {err:?}");
    }

    #[test]
    fn an_empty_block_round_trips_and_scores_nothing() {
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, DIM, &[], &[]).unwrap();
            assert!(block.is_empty());
            assert_eq!(block.len(), 0);
            let back = VectorBlock::from_bytes(&block.to_bytes()).unwrap();
            assert_eq!(back, block);
            let q = back.prepare(&vec![0.1f32; DIM]).unwrap();
            assert!(back.top_k(&q, 10).is_empty());
            assert_eq!(back.score(&q, 0), 0.0, "there is no member 0 to score");
            assert!(back.decode(0).is_empty());
        }
    }

    #[test]
    fn bytes_per_vector_matches_the_encoded_size() {
        let (vectors, _) = block_corpus(11);
        let ids = ids(N);
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, DIM, &ids, &vectors).unwrap();
            let mean = if codec == VectorCodec::F32 { 0 } else { DIM * 4 };
            let expected =
                HEADER_LEN + mean + N * (ID_LEN + VectorBlock::bytes_per_vector(codec, DIM));
            assert_eq!(
                block.to_bytes().len(),
                expected,
                "{codec:?} block size must be exactly what bytes_per_vector predicts"
            );
        }
    }

    #[test]
    fn the_codecs_compress_by_the_ratios_we_claim() {
        let raw = VectorBlock::bytes_per_vector(VectorCodec::F32, DIM) as f32;
        let int8 = raw / VectorBlock::bytes_per_vector(VectorCodec::Int8, DIM) as f32;
        let binary = raw / VectorBlock::bytes_per_vector(VectorCodec::Binary, DIM) as f32;
        // 1536 B -> 396 B and 1536 B -> 60 B at dim 384.
        assert!((3.8..4.0).contains(&int8), "int8 ratio {int8}");
        assert!((25.0..26.0).contains(&binary), "binary ratio {binary}");
        // The plan's "32x" is the code alone; the three corrective f32 are the difference,
        // and they are what make the estimate usable. 60 B beats the plan's own 62 B guess.
        assert_eq!(raw / DIM.div_ceil(8) as f32, 32.0);
    }

    #[test]
    fn the_id_table_is_what_caps_the_binary_block_ratio() {
        // Worth pinning, because it is the number the storage layer will actually measure:
        // at 1 bit/dim the 16 B id costs a third more than the 48 B of codes it labels, so a
        // whole binary block is ~19x smaller than a whole f32 one, not 25.6x. If the sibling
        // payload block can address members positionally, dropping the id table here is the
        // cheapest remaining win in the format.
        let (vectors, _) = block_corpus(5);
        let ids = ids(N);
        let size = |codec| {
            VectorBlock::encode(codec, DIM, &ids, &vectors)
                .unwrap()
                .to_bytes()
                .len() as f32
        };
        let ratio = size(VectorCodec::F32) / size(VectorCodec::Binary);
        assert!((19.0..20.5).contains(&ratio), "whole-block binary ratio {ratio}");
    }

    #[test]
    fn a_truncated_or_corrupt_block_is_an_error_not_a_panic() {
        let (vectors, _) = corpus(20, 16, TOPICS, 0.5, 2);
        let ids = ids(20);
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let bytes = VectorBlock::encode(codec, 16, &ids, &vectors)
                .unwrap()
                .to_bytes();
            for cut in [0, 1, HEADER_LEN - 1, HEADER_LEN, HEADER_LEN + 3, bytes.len() - 1] {
                assert!(
                    VectorBlock::from_bytes(&bytes[..cut]).is_err(),
                    "{codec:?}: a {cut}-byte prefix must be rejected, never indexed into"
                );
            }

            let mut extra = bytes.clone();
            extra.push(0);
            assert!(
                VectorBlock::from_bytes(&extra).is_err(),
                "{codec:?}: trailing bytes mean the writer disagreed about the layout"
            );

            let mut bad_magic = bytes.clone();
            bad_magic[0] = b'X';
            assert!(VectorBlock::from_bytes(&bad_magic).is_err());

            let mut bad_version = bytes.clone();
            bad_version[4] = 99;
            assert!(matches!(
                VectorBlock::from_bytes(&bad_version),
                Err(mlake_core::Error::FormatVersion { .. })
            ));

            let mut bad_codec = bytes.clone();
            bad_codec[5] = 7;
            assert!(VectorBlock::from_bytes(&bad_codec).is_err());

            // A count or dim that addresses far past the buffer must be caught by the length
            // check, not by an allocation or an index.
            for field in [8usize, 12] {
                let mut huge = bytes.clone();
                huge[field..field + 4].copy_from_slice(&u32::MAX.to_le_bytes());
                assert!(
                    VectorBlock::from_bytes(&huge).is_err(),
                    "{codec:?}: a u32::MAX at byte {field} must not be trusted"
                );
            }
        }
    }

    #[test]
    fn prepare_rejects_a_query_of_the_wrong_dim() {
        let block =
            VectorBlock::encode(VectorCodec::Int8, 3, &ids(1), &[vec![1.0, 0.0, 0.0]]).unwrap();
        assert!(matches!(
            block.prepare(&[1.0, 0.0]),
            Err(mlake_core::Error::DimMismatch { expected: 3, got: 2 })
        ));
    }

    #[test]
    fn encoding_is_deterministic() {
        // G-6: the same input must produce byte-identical output on every replay.
        let (vectors, _) = corpus(30, DIM, TOPICS, 0.5, 21);
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let bytes = || {
                VectorBlock::encode(codec, DIM, &ids(30), &vectors)
                    .unwrap()
                    .to_bytes()
            };
            assert_eq!(bytes(), bytes(), "{codec:?}");
        }
    }

    // --- the estimators ---------------------------------------------------------------

    #[test]
    fn int8_keeps_recall_at_10_against_the_f32_ranking() {
        let (vectors, queries) = block_corpus(7);
        let recall = recall_into(&vectors, &queries, VectorCodec::Int8, 10);
        assert!(
            recall >= INT8_RECALL_AT_10,
            "int8 recall@10 = {recall} over {N} vectors at dim {DIM}"
        );
    }

    #[test]
    fn binary_keeps_recall_at_10_against_the_f32_ranking() {
        let (vectors, queries) = block_corpus(7);
        let recall = recall_into(&vectors, &queries, VectorCodec::Binary, 10);
        assert!(
            recall >= BINARY_RECALL_AT_10,
            "binary recall@10 = {recall} over {N} vectors at dim {DIM}"
        );
    }

    #[test]
    fn binary_finds_the_true_top_ten_when_it_is_allowed_to_oversample() {
        // The number Phase 3 rests on. Binary scans, hands the top k*oversample to a
        // full-precision rerank, and what matters is whether the true top 10 are in there.
        let (vectors, queries) = block_corpus(7);
        let recall = recall_into(&vectors, &queries, VectorCodec::Binary, 40);
        assert!(
            recall >= BINARY_RECALL_AT_10_OVERSAMPLED,
            "binary recall@10 into a 40-candidate set = {recall}"
        );
    }

    #[test]
    fn the_binary_estimator_beats_a_naive_hamming_ranking() {
        // The bar 1 bit/dim has to clear to justify its 12 corrective bytes and its shared
        // mean: it must beat SimHash, which needs neither.
        let (vectors, queries) = block_corpus(7);
        let naive = naive_hamming_recall(&vectors, &queries, 10);
        assert!(
            naive <= NAIVE_HAMMING_RECALL_AT_10,
            "naive hamming recall@10 = {naive}; if it were this good the correctives would \
             not be earning their bytes"
        );
        let ours = recall_into(&vectors, &queries, VectorCodec::Binary, 10);
        assert!(
            ours > naive * 1.3,
            "binary {ours} must be materially better than naive hamming {naive}"
        );
    }

    #[test]
    fn the_binary_estimator_holds_up_where_naive_hamming_falls_apart() {
        // SimHash reads sign(v), so it degrades as the embedding space concentrates away
        // from the origin — and real embedding spaces are strongly concentrated. Centring on
        // the block mean is exactly the fix, and this pins that it works: over a 5x sweep of
        // the shared component our estimator is flat and SimHash is not.
        let mut ours = Vec::new();
        let mut naive = Vec::new();
        for global in [0.5f32, 2.5] {
            let (vectors, queries) = corpus(N, DIM, TOPICS, global, 7);
            ours.push(recall_into(&vectors, &queries, VectorCodec::Binary, 40));
            naive.push(naive_hamming_recall(&vectors, &queries, 40));
        }
        // Measured: ours 1.000 -> 1.000, naive 0.985 -> 0.845.
        assert!(
            ours[1] >= ours[0] - 0.02,
            "binary must not care how concentrated the space is: {ours:?}"
        );
        assert!(
            naive[1] < naive[0] - 0.05,
            "if naive hamming did not degrade here, centring would be buying nothing: \
             {naive:?}"
        );
    }

    #[test]
    fn binary_is_still_far_ahead_of_hamming_on_isotropic_data() {
        // The worst case, and the honest floor: strip the sub-topic structure and every
        // member is an equidistant random draw around the centre, so the true top 10 are
        // separated by less than 1 bit/dim can resolve. Nothing recovers this — but the
        // estimator still extracts several times what SimHash does from the same one bit.
        let (vectors, queries) = corpus(N, DIM, 1, 0.5, 7);
        let ours = recall_into(&vectors, &queries, VectorCodec::Binary, 10);
        let naive = naive_hamming_recall(&vectors, &queries, 10);
        // Measured: ours 0.448, naive 0.070 — 6.4x. Int8 is unbothered at 0.980.
        assert!(ours >= 0.40, "binary recall@10 on isotropic data = {ours}");
        assert!(ours > naive * 3.0, "binary {ours} vs naive hamming {naive}");
        let int8 = recall_into(&vectors, &queries, VectorCodec::Int8, 10);
        assert!(int8 >= 0.95, "int8 recall@10 on isotropic data = {int8}");
    }

    #[test]
    fn quantized_scores_preserve_the_exact_ordering() {
        // Top-k membership alone would hide a codec that ranks the tail arbitrarily, and the
        // tail is exactly what an oversampling rerank draws its candidates from.
        let int8 = rank_correlation(VectorCodec::Int8);
        assert!(int8 >= INT8_RANK_CORRELATION, "int8 spearman = {int8}");
        let binary = rank_correlation(VectorCodec::Binary);
        assert!(binary >= BINARY_RANK_CORRELATION, "binary spearman = {binary}");
    }

    // --- decoding and edges -----------------------------------------------------------

    #[test]
    fn int8_decodes_to_within_half_a_quantization_step() {
        let (vectors, _) = corpus(50, DIM, TOPICS, 0.5, 13);
        let block = VectorBlock::encode(VectorCodec::Int8, DIM, &ids(50), &vectors).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            let back = block.decode(i);
            let worst = back
                .iter()
                .zip(v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // Measured worst case over this corpus: 3.1e-4 absolute, cosine 0.99999.
            assert!(worst < 1e-3, "component error {worst}");
            assert!(mlake_core::cosine(&back, v) > 0.9999);
        }
    }

    #[test]
    fn binary_decodes_to_the_right_direction_if_not_the_right_components() {
        // A sign pattern cannot carry components; what it can carry is direction, and only
        // for the residual — the mean is exact. Measured worst case: cosine 0.961.
        let (vectors, _) = corpus(50, DIM, TOPICS, 0.5, 13);
        let block = VectorBlock::encode(VectorCodec::Binary, DIM, &ids(50), &vectors).unwrap();
        let worst = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| mlake_core::cosine(&block.decode(i), v))
            .fold(f32::INFINITY, f32::min);
        assert!(worst > 0.95, "worst decoded cosine {worst}");
    }

    #[test]
    fn top_k_orders_best_first_and_breaks_ties_by_id() {
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let a = MemoryId::from_key("zzz");
        let b = MemoryId::from_key("mmm");
        let c = MemoryId::from_key("aaa");
        let block = VectorBlock::encode(VectorCodec::F32, 2, &[a, b, c], &v).unwrap();
        let q = block.prepare(&[1.0, 0.0]).unwrap();
        let hits = block.top_k(&q, 3);
        let ordered: Vec<MemoryId> = hits.iter().map(|(i, _)| block.ids()[*i]).collect();
        let (lo, hi) = if a < c { (a, c) } else { (c, a) };
        assert_eq!(ordered, vec![lo, hi, b], "equal scores must order by id");
        assert!(hits[0].1 > hits[2].1);
    }

    #[test]
    fn top_k_is_capped_by_the_member_count() {
        let (vectors, queries) = corpus(5, DIM, 1, 0.5, 4);
        let block = VectorBlock::encode(VectorCodec::Binary, DIM, &ids(5), &vectors).unwrap();
        let q = block.prepare(&queries[0]).unwrap();
        assert_eq!(block.top_k(&q, 100).len(), 5);
    }

    #[test]
    fn a_zero_query_scores_zero_rather_than_nan() {
        let (vectors, _) = corpus(8, DIM, 1, 0.5, 6);
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, DIM, &ids(8), &vectors).unwrap();
            let q = block.prepare(&vec![0.0f32; DIM]).unwrap();
            for i in 0..block.len() {
                assert_eq!(block.score(&q, i), 0.0, "{codec:?} must not produce NaN");
            }
        }
    }

    #[test]
    fn a_member_sitting_exactly_on_the_block_mean_still_scores() {
        // Degenerate residual: the sign code carries nothing, and the estimator has to fall
        // back on the exact mean term instead of dividing by zero.
        let v = vec![vec![0.6, 0.8], vec![0.6, 0.8]];
        for codec in [VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, 2, &ids(2), &v).unwrap();
            let q = block.prepare(&[0.6, 0.8]).unwrap();
            for i in 0..2 {
                assert!((block.score(&q, i) - 1.0).abs() < 1e-4, "{codec:?}");
            }
        }
    }

    #[test]
    fn a_constant_vector_quantizes_without_dividing_by_zero() {
        // No range to spread over 256 levels, and a residual of exactly zero.
        let v = vec![vec![0.5; 8], vec![0.5; 8]];
        for codec in [VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, 8, &ids(2), &v).unwrap();
            let q = block.prepare(&[0.5; 8]).unwrap();
            assert!((block.score(&q, 0) - 1.0).abs() < 1e-4, "{codec:?}");
            assert!(block.decode(0).iter().all(|x| (x - 0.5).abs() < 1e-6), "{codec:?}");
        }
    }

    #[test]
    fn a_block_of_unnormalized_vectors_scores_the_same_as_a_normalized_one() {
        // The plan leans on bge being L2-normalized; `uniform_dim` exists because callers do
        // not always send what we expect. Cosine is scale-free and so is every codec here:
        // the mean, the residual and the stored norm all scale together.
        let (vectors, queries) = corpus(60, DIM, TOPICS, 0.5, 17);
        let scaled: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * 7.0).collect())
            .collect();
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let plain = VectorBlock::encode(codec, DIM, &ids(60), &vectors).unwrap();
            let big = VectorBlock::encode(codec, DIM, &ids(60), &scaled).unwrap();
            let pp = plain.prepare(&queries[0]).unwrap();
            let pb = big.prepare(&queries[0]).unwrap();
            for i in 0..plain.len() {
                let (a, b) = (plain.score(&pp, i), big.score(&pb, i));
                assert!((a - b).abs() < 1e-4, "{codec:?} member {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn binary_loses_accuracy_when_norms_vary_wildly_inside_one_block() {
        // A known soft spot, pinned rather than hidden. The binary estimator splits the dot
        // product into an exact part (along the block mean) and an estimated part (the
        // residual), so its error grows with |r|/|v| — and a member whose norm is far from
        // the block's has a residual as large as itself. Every embedding model we target
        // emits L2-normalized vectors, so |r|/|v| stays small; a caller mixing scales inside
        // one block would not get that.
        let (vectors, queries) = corpus(60, DIM, TOPICS, 0.5, 17);
        let skewed: Vec<Vec<f32>> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| v.iter().map(|x| x * (1.0 + i as f32)).collect())
            .collect();
        let exact = VectorBlock::encode(VectorCodec::F32, DIM, &ids(60), &skewed).unwrap();
        let mut worst: f32 = 0.0;
        for codec in [VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, DIM, &ids(60), &skewed).unwrap();
            let (pe, pa) = (
                exact.prepare(&queries[0]).unwrap(),
                block.prepare(&queries[0]).unwrap(),
            );
            let err = (0..60)
                .map(|i| (exact.score(&pe, i) - block.score(&pa, i)).abs())
                .fold(0.0f32, f32::max);
            if codec == VectorCodec::Int8 {
                // Int8 feels it too — its 256 levels have to span a residual as large as the
                // block — but an order of magnitude less. Measured: 6.9e-3, against 4.1e-4
                // on a normalized block.
                assert!(err < 0.02, "int8 worst score error {err}");
            } else {
                worst = err;
            }
        }
        // Measured: 0.112, against 0.019 on a normalized block.
        assert!(worst > 0.02, "if this stopped being true the caveat could go: {worst}");
        assert!(worst < 0.15, "binary worst score error {worst}");
    }
}
