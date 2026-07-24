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
//! Each estimate comes with an interval. [`VectorBlock::score`] is the point estimate and
//! [`VectorBlock::score_bounds`] is `[lo, hi]` around the true cosine, which is what lets the
//! rerank set be *derived* rather than guessed: take the top k by `lo`, rerank everything
//! whose `hi` clears the k-th best `lo`, and nothing that belongs in the true top k has been
//! dropped. For [`VectorCodec::Binary`] that interval only exists because the residual's signs
//! are taken in a random basis — see [`Rotation`], which is the whole reason a 1-bit code can
//! say anything about its own error. The interval is absolute for [`VectorCodec::F32`] and
//! [`VectorCodec::Int8`] and probabilistic for [`VectorCodec::Binary`]; the measured
//! containment rate is in the tests, and a caller narrowing on it should read them.
//!
//! Beside the codes sits the *tag column*: a per-cluster dictionary of the distinct tags its
//! members carry, plus one bitmap per member. Filtering used to be applied over materialized
//! memories, which meant a scan that avoided reading payload could not filter at all — the
//! whole point of splitting the vectors out. Making tags a scan-side column keeps filtering
//! **exact and pre-search**: [`VectorBlock::passes`] is a handful of `AND`s over bytes the
//! scan has already fetched, so a filtered probe is the same read as an unfiltered one, and
//! the k it returns is a true k rather than an oversampled guess that a post-filter thins.
//!
//! Everything is pure computation: no I/O, no async. The storage layer owns where these
//! bytes live.

use crate::kmeans::Rng;
use mlake_core::{MemoryId, TagsMatch};
use std::collections::BTreeSet;
use std::sync::Arc;

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
    /// One sign bit per dimension over the block-mean-centred residual **after a random
    /// rotation** ([`Rotation`]), plus a per-vector corrective term. ~25.6x smaller than
    /// [`VectorCodec::F32`] at dim 384. The rotation is what makes RaBitQ's error bound
    /// applicable, and so what makes [`VectorBlock::score_bounds`] mean anything here.
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
///
/// v2 added the tag column. There is no backwards compatibility: a v1 block is rejected
/// rather than read as an untagged v2 one, because "this block has no tag information" and
/// "this block's members have no tags" are different facts and only the writer knows which.
///
/// v3 rotates the residual before [`VectorCodec::Binary`] takes its signs (see [`Rotation`]).
/// The layout is byte-for-byte identical to v2 — same header, same stride, same corrective
/// triple — but the bits mean something else, so a v2 block read as v3 would score plausible
/// nonsense rather than fail. That is exactly the case a version byte exists for.
const FORMAT_VERSION: u8 = 3;

/// `magic | version | codec | flags | reserved | dim | count`.
const HEADER_LEN: usize = 16;

/// Header flag bit: the block carries a tag dictionary and per-member bitmaps.
///
/// Explicit rather than inferred from a non-empty dictionary, because a block whose members
/// are *all* untagged has an empty dictionary and still filters (the strict modes must
/// exclude its members; `Exact` with an empty request must select them).
const FLAG_HAS_TAGS: u8 = 0b0000_0001;
/// Set when the block carries a per-member `updated_at` column. See [`VectorBlock::updated`].
const FLAG_HAS_UPDATED: u8 = 0b0000_0010;

/// A member whose write time is unknown. Distinct from any real timestamp, and it fails
/// every bounded window: an unknown write time cannot be shown to fall inside one.
pub const UPDATED_UNKNOWN: i64 = i64::MIN;

/// A [`MemoryId`] is a raw 16-byte UUID.
const ID_LEN: usize = 16;

/// Per-vector corrective floats carried by the quantized codecs, ahead of the codes.
///
/// [`VectorCodec::Int8`] stores `offset`, `scale` and the true L2 norm;
/// [`VectorCodec::Binary`] stores the residual norm, the code's cosine to its residual, and
/// the true L2 norm. Both are three `f32`.
const CORRECTIVE_LEN: usize = 12;

/// Seed of the shared random rotation. Part of the format: changing it invalidates every
/// [`VectorCodec::Binary`] block ever written, exactly as changing the codec would.
const ROTATION_SEED: u64 = 0x5241_4249_5451_524F;

/// Rounds of `permute -> sign-flip -> block Hadamard` composed into the rotation.
///
/// One round decorrelates the coordinates *within* a Hadamard segment and not at all across
/// them, which at a `dim` that is not a power of two leaves the transform visibly non-random:
/// measured, a spike confined to the 256-segment comes out of one round at
/// `c = 0.90` against the isotropic 0.798, and two rounds fix it (this is what
/// `the_rotation_makes_a_spiky_residual_behave_like_a_random_direction` pins). Rounds past the
/// second buy nothing measurable here — recall over 1/2/3/5 rounds is 0.527/0.500/0.534/0.532,
/// which is noise, and the bound's width and containment do not move at all. Three is kept
/// because it is the standard `HD3 HD2 HD1` depth (Ailon–Chazelle, and every FJLT-derived
/// quantizer since) and the third round costs ~1% of encode time, not because this corpus can
/// tell it apart from two.
const ROTATION_ROUNDS: usize = 3;

/// The `epsilon` of RaBitQ's error bound: how many standard deviations of the estimator's
/// error [`VectorBlock::score_bounds`] admits.
///
/// The bound is **probabilistic**, not absolute (see [`score_binary_bounded`]), with a
/// per-member failure probability of about `2 exp(-epsilon^2 / 2)` — 7.5e-6 at 5.0. It is
/// deliberately not larger: the interval width is linear in it, and a wider interval is paid
/// for on every query in rerank volume, while the cost of a miss is one dropped candidate out
/// of an oversampled set. The empirical containment rate is pinned by
/// `the_binary_bound_contains_the_true_cosine`.
const RABITQ_EPSILON: f32 = 5.0;

/// Absolute slack folded into every quantized bound so f32 rounding in the corrective terms
/// — norms stored at 24-bit precision, and the cancellation in recovering `a` from them —
/// cannot make a bound miss by an ulp. Four orders of magnitude below the bounds themselves.
const BOUND_FP_SLACK: f32 = 1e-5;

/// A deterministic random orthogonal transform of `R^dim`, in `O(dim log dim)` and `O(dim)`
/// memory.
///
/// **Why it exists.** RaBitQ's error bound is provable because the vector's signs are taken in
/// a *random* basis: that is what makes the quantization error isotropic, so the component of
/// it that the query sees is a random projection and concentrates like one. Without the
/// rotation the error is whatever the embedding model's coordinate system makes it, there is
/// no bound to state, and a caller wanting guaranteed recall has to guess an oversampling
/// factor instead of computing one. The rotation is the whole reason
/// [`VectorBlock::score_bounds`] can exist.
///
/// **Why not a matrix.** A dense `dim x dim` rotation is 590 KB at dim 384 — ten times the
/// entire binary block it would serve — and `O(d^2)` per vector on the index-build hot path.
/// So the transform is *derived* from [`ROTATION_SEED`] and `dim` rather than stored, and it
/// is built from operations that are individually orthogonal and individually cheap:
///
/// * a permutation of the coordinates,
/// * a sign flip per coordinate,
/// * a fast Walsh–Hadamard transform, scaled by `1/sqrt(len)`.
///
/// **Why segments.** The FWHT needs a power-of-two length and `dim` is routinely not one (384
/// is not, 768 is). Zero-padding to the next power of two would work but would widen the code
/// — 384 bits become 512, a third more bytes per vector, forever. Instead `dim` is split into
/// the descending powers of two of its binary expansion (384 = 256 + 128) and the FWHT runs
/// block-diagonally. A block-diagonal orthogonal matrix is orthogonal, so the transform is
/// still a rotation of the full space; what it is not, in one round, is *mixing* between
/// segments — which is what the inter-round permutation supplies.
///
/// **What it is not.** This is not a uniformly random element of `O(dim)`. It is the standard
/// randomized-Hadamard stand-in for one, which is what the whole FJLT literature and every
/// shipping RaBitQ implementation use, and it is an approximation. The bound below is
/// therefore theory-shaped rather than proven, and is measured rather than asserted.
#[derive(Clone, PartialEq, Debug)]
struct Rotation {
    dim: usize,
    /// Lengths of the Hadamard blocks: descending powers of two summing to `dim`.
    segments: Vec<usize>,
    /// Empty for the identity, which exists so the tests can measure what the rotation buys.
    rounds: Vec<RotationRound>,
}

#[derive(Clone, PartialEq, Debug)]
struct RotationRound {
    /// `perm[i]` is the *source* coordinate of output slot `i`.
    perm: Vec<u32>,
    /// `+1.0` or `-1.0` per coordinate.
    signs: Vec<f32>,
}

impl Rotation {
    /// The rotation every block of this `dim` uses. A pure function of `dim` and
    /// [`ROTATION_SEED`]: same input, same bytes, in any process (G-6).
    fn derive(dim: usize) -> Self {
        // The seed is mixed with `dim` so two dims do not share a coordinate permutation
        // prefix, and through SplitMix64's finalizer so nearby dims start far apart.
        let mut rng = Rng::seeded(
            ROTATION_SEED ^ (dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let rounds = (0..ROTATION_ROUNDS)
            .map(|_| {
                let mut perm: Vec<u32> = (0..dim as u32).collect();
                // Fisher-Yates, descending: a fixed number of draws in a fixed order, so the
                // permutation is a pure function of the seed and not of allocator or
                // iteration order.
                for i in (1..dim).rev() {
                    perm.swap(i, rng.below(i + 1));
                }
                let signs = (0..dim)
                    .map(|_| if rng.below(2) == 0 { -1.0f32 } else { 1.0 })
                    .collect();
                RotationRound { perm, signs }
            })
            .collect();
        Self {
            dim,
            segments: hadamard_segments(dim),
            rounds,
        }
    }

    /// The transform that does nothing. Only the tests construct one — it is how
    /// `rotation_is_worth_its_cost_in_recall` measures the same codec without the rotation.
    #[cfg(test)]
    fn identity(dim: usize) -> Self {
        Self {
            dim,
            segments: hadamard_segments(dim),
            rounds: Vec::new(),
        }
    }

    /// `v <- R v`, in place. `scratch` must be at least `dim` long and its contents are
    /// garbage afterwards; it is a parameter rather than a local because `encode` runs this
    /// once per vector in the corpus and an allocation per vector is not free.
    fn apply(&self, v: &mut [f32], scratch: &mut [f32]) {
        for round in &self.rounds {
            for i in 0..self.dim {
                scratch[i] = round.signs[i] * v[round.perm[i] as usize];
            }
            v[..self.dim].copy_from_slice(&scratch[..self.dim]);
            self.hadamard(v);
        }
    }

    /// `v <- R^T v`, in place. Exact up to f32 rounding: every factor is its own inverse
    /// (`H/sqrt(n)` is a symmetric involution, a sign flip is one) bar the permutation, which
    /// is undone by scattering where the forward pass gathered.
    fn apply_inverse(&self, v: &mut [f32], scratch: &mut [f32]) {
        for round in self.rounds.iter().rev() {
            self.hadamard(v);
            for i in 0..self.dim {
                scratch[round.perm[i] as usize] = round.signs[i] * v[i];
            }
            v[..self.dim].copy_from_slice(&scratch[..self.dim]);
        }
    }

    /// Block-diagonal normalized Walsh-Hadamard transform.
    fn hadamard(&self, v: &mut [f32]) {
        let mut off = 0;
        for &len in &self.segments {
            let seg = &mut v[off..off + len];
            fwht(seg);
            let s = 1.0 / (len as f32).sqrt();
            for x in seg.iter_mut() {
                *x *= s;
            }
            off += len;
        }
    }
}

/// `dim` as descending powers of two: 384 -> [256, 128], 100 -> [64, 32, 4], 1 -> [1].
///
/// A length-1 segment is a no-op Hadamard; that coordinate is still permuted and sign-flipped,
/// so it is still mixed into the rest by the following round.
fn hadamard_segments(dim: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut rem = dim;
    while rem > 0 {
        let p = 1usize << (usize::BITS - 1 - rem.leading_zeros());
        out.push(p);
        rem -= p;
    }
    out
}

/// In-place unnormalized fast Walsh-Hadamard transform. `v.len()` must be a power of two.
fn fwht(v: &mut [f32]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (x, y) = (v[j], v[j + h]);
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

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
    /// Whether the tag column below is meaningful. See [`FLAG_HAS_TAGS`].
    has_tags: bool,
    /// The block's distinct tags, sorted and deduplicated. Sorted for two reasons: a tag's
    /// dictionary position is then a pure function of the block's contents, which is what
    /// makes encoding byte-identical across replays (G-6), and [`VectorBlock::tag_mask`] can
    /// compile a query against it by binary search instead of a linear scan.
    tag_dict: Vec<String>,
    /// One bitmap per member, `ceil(|tag_dict| / 8)` bytes each, laid out contiguously in
    /// member order: bit `t` of member `i` is set iff member `i` carries `tag_dict[t]`.
    ///
    /// **Known cost, deliberately uncapped.** The width is linear in the *block's* distinct
    /// tag count, so a cluster whose members share a small vocabulary costs almost nothing
    /// (32 distinct tags = 4 B/member, ~7% of a 60 B binary code) while a cluster where every
    /// member carries a unique tag costs `count/8` bytes per member — quadratic in the block,
    /// and at 500 members that is 63 B/member, more than the code it rides beside. There is
    /// no cap, no fallback to a sparse or roaring encoding, and no spill of a long tail to
    /// payload. That is a conscious omission, not an oversight: the fixed stride is what
    /// makes [`VectorBlock::passes`] branch-free and what lets the whole column be range-read
    /// with the codes. If a real namespace ever shows a high-cardinality per-cluster
    /// vocabulary, the fix is a sparse encoding behind the same [`TagMask`] API, and the test
    /// `the_bitmap_costs_what_we_claim_per_member` is where the number to beat is pinned.
    tag_bits: Vec<u8>,
    /// Bitwise OR of every member bitmap — the block's tag union, for [`VectorBlock::any_can_pass`].
    /// Derived, never serialized.
    tag_union: Vec<u8>,
    /// Whether any member is untagged (an all-zero bitmap). Derived, never serialized.
    any_untagged: bool,
    /// Whether the `updated` column below is meaningful. See [`FLAG_HAS_UPDATED`].
    has_updated: bool,
    /// Per-member write time (epoch ms), member order, [`UPDATED_UNKNOWN`] where absent.
    ///
    /// Fixed 8-byte stride, unlike the tag bitmap's variable width, so the column is a plain
    /// `i64` lookup. It costs 8 B/member — about 13% on top of a 60 B binary code — and buys
    /// the *only* way to apply an `updated_at` window before the top-k truncation: a segment
    /// candidate is `(id, score)` until it is materialized, which happens after the cut, so
    /// without this a window can only remove rows from the winners rather than deepen the
    /// search into the matching set.
    updated: Vec<i64>,
    /// The block's `updated` range over members with a known value. Lets a query skip a whole
    /// block whose every member falls outside the window, the way `tag_union` does for tags.
    /// Derived, never serialized.
    updated_min: i64,
    updated_max: i64,
    /// The basis [`VectorCodec::Binary`]'s signs were taken in. Derived from `dim` and
    /// [`ROTATION_SEED`], never serialized — storing it would cost more than the codes.
    /// `None` for the codecs that do not rotate. Behind an [`Arc`] so cloning a block does not
    /// clone three permutations, and built once here rather than once per `prepare` so a
    /// query pays `O(dim log dim)` for the rotation, not `O(members * dim log dim)`.
    rot: Option<Arc<Rotation>>,
}

/// A tag filter compiled against one block's dictionary.
///
/// Compiling is the only place a string comparison happens on the query path: it resolves
/// each request tag to a dictionary position once per block per query, after which
/// [`VectorBlock::passes`] is pure bitwise work. Because a mask is bound to the dictionary it
/// was compiled against, it is not portable between blocks — [`VectorBlock::tag_mask`] is
/// cheap precisely so it can be called per block.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TagMask {
    /// The request tags that exist in this block's dictionary, as a bitmap of the same width
    /// as a member's.
    bits: Vec<u8>,
    /// Width of `bits`, kept explicitly so a mask compiled against a different block is
    /// detectable rather than silently misaligned.
    width: usize,
    /// How many *distinct* request tags resolved to a dictionary entry.
    present: usize,
    /// How many *distinct* request tags are absent from this block's dictionary.
    ///
    /// This is the field that makes absent tags exact rather than approximate. An absent tag
    /// is one no member of this block carries, so:
    /// * for `Any`/`AnyStrict` it can never contribute an overlap — it simply sets no bit,
    ///   and the remaining request tags decide;
    /// * for `All`/`AllStrict`/`Exact` it makes `request ⊆ member tags` unsatisfiable for
    ///   *every* member, so `missing > 0` is a whole-block answer, which is exactly what
    ///   [`VectorBlock::any_can_pass`] exploits to skip the block without scoring it.
    missing: usize,
}

impl TagMask {
    /// Number of distinct request tags, present or absent. Zero means "no filter" for every
    /// mode except `Exact`, where it is the untagged scope.
    fn requested(&self) -> usize {
        self.present + self.missing
    }
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
    /// `sum |q|`, the coefficient of the worst-case per-coordinate error in the
    /// [`VectorCodec::Int8`] bound.
    abs_sum: f32,
    /// `q` with its component along the block mean removed, **then rotated into the basis the
    /// codes' signs were taken in**. [`VectorCodec::Binary`] only — see
    /// [`score_binary_bounded`] for why the estimator runs on this rather than on `q`, and
    /// [`Rotation`] for why it is rotated.
    perp: Vec<f32>,
    /// `|perp|`, which the rotation leaves alone.
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
        Self::encode_inner(codec, dim, ids, vectors, None, None, None)
    }

    /// Encode with a per-cluster tag dictionary and per-member bitmaps.
    ///
    /// `member_tags[i]` are the tags of member `i`; it must align 1:1 with `ids`/`vectors`.
    /// Duplicates within a member's list and duplicates across members collapse into the one
    /// dictionary entry, so the encoding does not depend on how the caller happened to order
    /// or repeat them (G-6).
    ///
    /// The result answers [`VectorBlock::has_tags`] with `true` even when every member is
    /// untagged: that is a filterable fact, not an absence of information.
    pub fn encode_with_tags(
        codec: VectorCodec,
        dim: usize,
        ids: &[MemoryId],
        vectors: &[Vec<f32>],
        member_tags: &[Vec<String>],
    ) -> Result<Self, mlake_core::Error> {
        if member_tags.len() != ids.len() {
            return Err(mlake_core::Error::Encode(format!(
                "{} tag lists for {} ids: the tag column is positional",
                member_tags.len(),
                ids.len()
            )));
        }
        Self::encode_inner(codec, dim, ids, vectors, Some(member_tags), None, None)
    }

    /// Encode with the tag column *and* a per-member `updated_at` column.
    ///
    /// `member_updated[i]` is member `i`'s write time in epoch ms, or [`UPDATED_UNKNOWN`].
    /// Both columns are positional and must align 1:1 with `ids`/`vectors`.
    pub fn encode_with_columns(
        codec: VectorCodec,
        dim: usize,
        ids: &[MemoryId],
        vectors: &[Vec<f32>],
        member_tags: Option<&[Vec<String>]>,
        member_updated: &[i64],
    ) -> Result<Self, mlake_core::Error> {
        if member_updated.len() != ids.len() {
            return Err(mlake_core::Error::Encode(format!(
                "{} updated_at values for {} ids: the column is positional",
                member_updated.len(),
                ids.len()
            )));
        }
        Self::encode_inner(codec, dim, ids, vectors, member_tags, Some(member_updated), None)
    }

    /// The same encoding with the rotation replaced — the only way to build a block whose
    /// signs are taken in the raw coordinate basis, which is the baseline the rotation is
    /// measured against.
    #[cfg(test)]
    fn encode_with_rotation(
        codec: VectorCodec,
        dim: usize,
        ids: &[MemoryId],
        vectors: &[Vec<f32>],
        rot: Rotation,
    ) -> Result<Self, mlake_core::Error> {
        Self::encode_inner(codec, dim, ids, vectors, None, None, Some(Arc::new(rot)))
    }

    fn encode_inner(
        codec: VectorCodec,
        dim: usize,
        ids: &[MemoryId],
        vectors: &[Vec<f32>],
        member_tags: Option<&[Vec<String>]>,
        member_updated: Option<&[i64]>,
        rot_override: Option<Arc<Rotation>>,
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

        let rot = rot_from(codec, dim, rot_override);

        let stride = Self::bytes_per_vector(codec, dim);
        let mut codes = Vec::with_capacity(stride * vectors.len());
        let mut residual = vec![0.0f32; dim];
        let mut scratch = vec![0.0f32; dim];
        for v in vectors {
            match codec {
                VectorCodec::F32 => {
                    for x in v {
                        codes.extend_from_slice(&x.to_le_bytes());
                    }
                }
                VectorCodec::Int8 | VectorCodec::Binary => {
                    for ((r, x), m) in residual.iter_mut().zip(v).zip(&mean) {
                        *r = x - m;
                    }
                    if codec == VectorCodec::Int8 {
                        // Deliberately *not* rotated — see `score_int8_bounded`.
                        encode_int8(&residual, mlake_core::norm(v), &mut codes);
                    } else {
                        // `|r|` is taken before the rotation and carried across it: the
                        // transform is orthogonal, so this is the same number, but computing
                        // it here keeps the norm identity `|v|^2 = |m|^2 + 2<m,r> + |r|^2`
                        // that `score_binary_bounded` inverts exact in the unrotated frame it
                        // is stated in.
                        let r_norm = mlake_core::norm(&residual);
                        if let Some(rot) = &rot {
                            rot.apply(&mut residual, &mut scratch);
                        }
                        encode_binary(&residual, r_norm, mlake_core::norm(v), &mut codes);
                    }
                }
            }
        }
        debug_assert_eq!(codes.len(), stride * vectors.len());

        let (has_tags, tag_dict, tag_bits) = match member_tags {
            None => (false, Vec::new(), Vec::new()),
            Some(per_member) => {
                // Sorted + deduplicated, so the dictionary — and therefore every bit
                // position — is a pure function of the block's contents.
                let dict: Vec<String> = per_member
                    .iter()
                    .flatten()
                    .collect::<BTreeSet<&String>>()
                    .into_iter()
                    .cloned()
                    .collect();
                let width = dict.len().div_ceil(8);
                let mut bits = vec![0u8; width * per_member.len()];
                for (i, tags) in per_member.iter().enumerate() {
                    let row = &mut bits[i * width..(i + 1) * width];
                    for t in tags {
                        // Present by construction: the dictionary is the union of these.
                        if let Ok(t) = dict.binary_search(t) {
                            row[t / 8] |= 1 << (t % 8);
                        }
                    }
                }
                (true, dict, bits)
            }
        };
        let (tag_union, any_untagged) = tag_summary(&tag_bits, tag_dict.len().div_ceil(8), ids.len());
        let updated = member_updated.map(|u| u.to_vec()).unwrap_or_default();
        let has_updated = member_updated.is_some();
        let (updated_min, updated_max) = updated_summary(&updated);

        Ok(Self {
            codec: Some(codec),
            dim,
            ids: ids.to_vec(),
            mean,
            codes,
            has_tags,
            tag_dict,
            tag_bits,
            tag_union,
            any_untagged,
            has_updated,
            updated,
            updated_min,
            updated_max,
            rot,
        })
    }

    /// Serialize to bytes. Self-describing: codec, dim, count and the tag column are all in
    /// the stream.
    ///
    /// `[magic 4][version 1][codec 1][flags 1][reserved 1][dim u32][count u32]`, then `count`
    /// ids, then the block mean (quantized codecs only), then the tag column when
    /// [`FLAG_HAS_TAGS`] is set, then the `updated` column when [`FLAG_HAS_UPDATED`] is, then
    /// the codes.
    ///
    /// The tag column is `[dict len u32]`, then that many `[len u32][utf8]` entries, then
    /// `count` bitmaps of `ceil(dict len / 8)` bytes. The `updated` column is `count` little-
    /// endian `i64`s. Both sit *before* the codes so the codes remain the tail: a reader that
    /// has parsed everything else knows the exact number of bytes that must remain, which is
    /// what makes the length check below an equality.
    pub fn to_bytes(&self) -> Vec<u8> {
        let codec = self.codec();
        let stride = Self::bytes_per_vector(codec, self.dim);
        let mut out = Vec::with_capacity(
            HEADER_LEN
                + self.ids.len() * ID_LEN
                + self.mean.len() * 4
                + self.tag_bits.len()
                + self.updated_section_len()
                + self.codes.len(),
        );
        out.extend_from_slice(&MAGIC);
        out.push(FORMAT_VERSION);
        out.push(codec.tag());
        let mut flags = 0u8;
        if self.has_tags {
            flags |= FLAG_HAS_TAGS;
        }
        if self.has_updated {
            flags |= FLAG_HAS_UPDATED;
        }
        out.push(flags);
        out.push(0);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        for id in &self.ids {
            out.extend_from_slice(&id.0);
        }
        for x in &self.mean {
            out.extend_from_slice(&x.to_le_bytes());
        }
        if self.has_tags {
            out.extend_from_slice(&(self.tag_dict.len() as u32).to_le_bytes());
            for t in &self.tag_dict {
                out.extend_from_slice(&(t.len() as u32).to_le_bytes());
                out.extend_from_slice(t.as_bytes());
            }
            out.extend_from_slice(&self.tag_bits);
        }
        // Fixed 8 bytes per member; sits after the tag column so the codes stay the tail.
        if self.has_updated {
            for u in &self.updated {
                out.extend_from_slice(&u.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.codes);
        debug_assert_eq!(
            out.len(),
            HEADER_LEN
                + self.ids.len() * (ID_LEN + stride)
                + self.mean.len() * 4
                + self.tag_section_len()
                + self.updated_section_len()
        );
        out
    }

    /// Bytes the `updated` column occupies in the serialized form.
    fn updated_section_len(&self) -> usize {
        if self.has_updated {
            self.updated.len() * 8
        } else {
            0
        }
    }

    /// Bytes the tag column occupies in the serialized form.
    fn tag_section_len(&self) -> usize {
        if !self.has_tags {
            return 0;
        }
        4 + self.tag_dict.iter().map(|t| 4 + t.len()).sum::<usize>() + self.tag_bits.len()
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
        let flags = bytes[6];
        if flags & !(FLAG_HAS_TAGS | FLAG_HAS_UPDATED) != 0 {
            return Err(bad(&format!("unknown header flags {flags:#04x}")));
        }
        let has_tags = flags & FLAG_HAS_TAGS != 0;
        let has_updated = flags & FLAG_HAS_UPDATED != 0;
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
        if bytes.len() < mean_end {
            return Err(bad(&format!(
                "declares {count} ids and a {mean_len}-dim mean ({mean_end} bytes) but has {}",
                bytes.len()
            )));
        }

        // The tag column is the one variable-length section, so it is walked with an explicit
        // cursor and every step bounds-checked before it is taken.
        let mut cursor = mean_end;
        let mut tag_dict: Vec<String> = Vec::new();
        let mut tag_bits: Vec<u8> = Vec::new();
        if has_tags {
            let u32_at = |c: &mut usize| -> Result<usize, mlake_core::Error> {
                let end = c.checked_add(4).ok_or_else(|| bad("tag column overflows"))?;
                let s = bytes.get(*c..end).ok_or_else(|| bad("tag column truncated"))?;
                *c = end;
                Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
            };
            let dict_len = u32_at(&mut cursor)?;
            // A dictionary entry costs at least 4 bytes, so a length past the buffer is a lie
            // worth rejecting before it becomes an allocation.
            if dict_len > bytes.len().saturating_sub(cursor) / 4 {
                return Err(bad(&format!(
                    "declares {dict_len} dictionary entries, more than the remaining bytes admit"
                )));
            }
            tag_dict.reserve(dict_len);
            for _ in 0..dict_len {
                let len = u32_at(&mut cursor)?;
                let end = cursor
                    .checked_add(len)
                    .ok_or_else(|| bad("tag dictionary overflows"))?;
                let raw = bytes
                    .get(cursor..end)
                    .ok_or_else(|| bad("tag dictionary truncated"))?;
                let t = std::str::from_utf8(raw)
                    .map_err(|_| bad("tag dictionary holds invalid utf-8"))?;
                tag_dict.push(t.to_string());
                cursor = end;
            }
            if tag_dict.windows(2).any(|w| w[0] >= w[1]) {
                // Not merely a tidiness check: `tag_mask` binary-searches this, and a scan
                // silently missing a filter tag is a wrong answer rather than a loud one.
                return Err(bad("tag dictionary is not strictly sorted"));
            }
            let bitmap_len = count
                .checked_mul(tag_dict.len().div_ceil(8))
                .ok_or_else(|| bad("tag bitmaps overflow"))?;
            let end = cursor
                .checked_add(bitmap_len)
                .ok_or_else(|| bad("tag bitmaps overflow"))?;
            tag_bits = bytes
                .get(cursor..end)
                .ok_or_else(|| bad("tag bitmaps truncated"))?
                .to_vec();
            cursor = end;
        }

        // Fixed-width, so one bounds check covers the whole column.
        let mut updated: Vec<i64> = Vec::new();
        if has_updated {
            let end = count
                .checked_mul(8)
                .and_then(|n| n.checked_add(cursor))
                .ok_or_else(|| bad("updated column overflows"))?;
            let raw = bytes
                .get(cursor..end)
                .ok_or_else(|| bad("updated column truncated"))?;
            updated = raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().expect("chunks_exact yields 8 bytes")))
                .collect();
            cursor = end;
        }

        let total = count
            .checked_mul(Self::bytes_per_vector(codec, dim))
            .and_then(|n| n.checked_add(cursor))
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
        let (tag_union, any_untagged) = tag_summary(&tag_bits, tag_dict.len().div_ceil(8), count);
        let (updated_min, updated_max) = updated_summary(&updated);

        Ok(Self {
            codec: Some(codec),
            dim,
            ids,
            mean,
            codes: bytes[cursor..].to_vec(),
            has_tags,
            tag_dict,
            tag_bits,
            tag_union,
            any_untagged,
            has_updated,
            updated,
            updated_min,
            updated_max,
            // Rederived, not read: `dim` is in the header and the rotation is a pure function
            // of it. `dim` is already bounded by the length checks above, so this cannot be
            // talked into a large allocation by a malformed block.
            rot: rot_from(codec, dim, None),
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

    /// The block's distinct tags, sorted. Empty when the block was encoded without tags —
    /// and also when it was encoded with tags but no member carries any. [`Self::has_tags`]
    /// is what distinguishes those two.
    pub fn tag_dictionary(&self) -> &[String] {
        &self.tag_dict
    }

    /// Whether this block carries tag bitmaps at all.
    ///
    /// `false` means the block holds no filtering information, and every tag method here
    /// degrades to "no filter": [`Self::passes`] admits every member and
    /// [`Self::any_can_pass`] admits the block. Absence of information is not a reason to
    /// drop results.
    pub fn has_tags(&self) -> bool {
        self.has_tags
    }

    /// Bytes one member's tag bitmap occupies. `0` when the block carries no tags, or when
    /// it carries tags but every member is untagged.
    pub fn tag_bitmap_width(&self) -> usize {
        self.tag_dict.len().div_ceil(8)
    }

    /// Compile a filter's tags against this block's dictionary, once per block per query.
    ///
    /// The result is bound to this block: bit positions are dictionary positions, and
    /// dictionaries differ between blocks. Request tags absent from the dictionary are
    /// counted rather than dropped — see [`TagMask::missing`] for why that distinction is
    /// what keeps the absent-tag case exact.
    pub fn tag_mask(&self, tags: &[String]) -> TagMask {
        let width = self.tag_bitmap_width();
        let mut bits = vec![0u8; width];
        let (mut present, mut missing) = (0usize, 0usize);
        // Distinct, so a request repeating a tag does not count it twice — `TagFilter` works
        // on sets and the counts here have to mean the same thing.
        for t in tags.iter().collect::<BTreeSet<&String>>() {
            match self.tag_dict.binary_search(t) {
                Ok(pos) => {
                    present += 1;
                    bits[pos / 8] |= 1 << (pos % 8);
                }
                Err(_) => missing += 1,
            }
        }
        TagMask {
            bits,
            width,
            present,
            missing,
        }
    }

    /// Member `i`'s tag bitmap, or `None` when `i` is out of range.
    fn member_bits(&self, i: usize) -> Option<&[u8]> {
        let w = self.tag_bitmap_width();
        if i >= self.ids.len() {
            return None;
        }
        // A zero-width bitmap has no bytes to slice, and every member is untagged.
        Some(if w == 0 {
            &[]
        } else {
            self.tag_bits.get(i * w..(i + 1) * w)?
        })
    }

    /// Whether member `i` satisfies the filter. Pure bitwise; no payload read.
    ///
    /// Exactly equivalent to `TagFilter::new(tags, mode).matches(&member_tags[i])` for the
    /// `tags` the mask was compiled from — including how each mode treats an untagged
    /// member, which is the part that is easy to get subtly wrong: `Any`/`All` *include*
    /// untagged members, the strict modes and `Exact` exclude them. A member is untagged iff
    /// its bitmap is all zeroes, which is exact because the dictionary is the union of the
    /// block's own member tags: every tag a member carries has a bit.
    ///
    /// Returns `true` for every member when the block carries no tags, and `false` for an
    /// `i` past the end (there is no such member to admit).
    pub fn passes(&self, i: usize, mask: &TagMask, mode: TagsMatch) -> bool {
        debug_assert_eq!(
            mask.width,
            self.tag_bitmap_width(),
            "mask compiled against a different block's dictionary"
        );
        let Some(m) = self.member_bits(i) else {
            return false;
        };
        if !self.has_tags || mask.width != self.tag_bitmap_width() {
            return true;
        }
        if mask.requested() == 0 {
            // Empty request: no filter, except Exact where it selects the untagged scope.
            return match mode {
                TagsMatch::Exact => is_zero(m),
                _ => true,
            };
        }
        let untagged = is_zero(m);
        match mode {
            TagsMatch::Any => untagged || overlaps(m, mask),
            TagsMatch::All => untagged || contains_all(m, mask),
            TagsMatch::AnyStrict => !untagged && overlaps(m, mask),
            TagsMatch::AllStrict => !untagged && contains_all(m, mask),
            TagsMatch::Exact => !untagged && set_eq(m, mask),
        }
    }

    /// Whether ANY member could satisfy the filter. `false` means the caller can skip this
    /// block entirely without scoring it.
    ///
    /// A necessary condition, evaluated against the block's tag *union* and whether it holds
    /// an untagged member — the per-member [`Self::passes`] still decides. It is the bitwise
    /// twin of `TagFilter::cluster_admits`, and its whole value is the absent-tag case: a
    /// request tag no member carries makes `All`/`AllStrict`/`Exact` unsatisfiable for the
    /// entire block, which is a whole cluster's worth of scoring skipped for one comparison.
    ///
    /// Never optimistic in the direction that matters: if it returns `false`, `passes`
    /// returns `false` for every member. It may return `true` where nothing in fact passes
    /// (`Exact` in particular, where the union admits a request no single member holds).
    pub fn any_can_pass(&self, mask: &TagMask, mode: TagsMatch) -> bool {
        if self.ids.is_empty() {
            return false;
        }
        if !self.has_tags || mask.width != self.tag_bitmap_width() {
            return true;
        }
        if mask.requested() == 0 {
            return match mode {
                TagsMatch::Exact => self.any_untagged,
                _ => true,
            };
        }
        let u = &self.tag_union;
        match mode {
            TagsMatch::Any => self.any_untagged || overlaps(u, mask),
            TagsMatch::All => self.any_untagged || contains_all(u, mask),
            TagsMatch::AnyStrict => overlaps(u, mask),
            TagsMatch::AllStrict | TagsMatch::Exact => contains_all(u, mask),
        }
    }

    /// Whether member `i` falls inside the `updated_at` window. Pure integer compare; no
    /// payload read.
    ///
    /// Exactly the window arm of `Predicate::matches`: strictly after `from`, strictly before
    /// `to`, and a member whose write time is unknown fails a *bounded* window rather than
    /// passing it by default. An unbounded window (both `None`) admits everyone, unknowns
    /// included.
    ///
    /// Returns `true` for every member when the block carries no `updated` column — the
    /// caller must then still filter on the hydrated payload — and `false` for an `i` past
    /// the end.
    pub fn passes_updated(&self, i: usize, from: Option<i64>, to: Option<i64>) -> bool {
        if i >= self.ids.len() {
            return false;
        }
        if from.is_none() && to.is_none() {
            return true;
        }
        let Some(&u) = self.updated.get(i) else {
            // No column: undecidable here, so admit and let the payload-side check rule.
            return true;
        };
        if u == UPDATED_UNKNOWN {
            return false;
        }
        from.is_none_or(|f| u > f) && to.is_none_or(|t| u < t)
    }

    /// Whether ANY member could fall inside the window. `false` means the caller can skip the
    /// whole block without scoring it.
    ///
    /// Compares the request against the block's `[min, max]` of *known* write times. A block
    /// written entirely before `from` or entirely after `to` is a whole cluster's worth of
    /// scoring skipped for two integer compares — the common shape for a time-sliced query
    /// against an index whose blocks tend to be built from contiguous writes.
    ///
    /// Never optimistic in the direction that matters: `false` implies [`Self::passes_updated`]
    /// is `false` for every member. It may return `true` where nothing in fact passes, since
    /// the range says nothing about which points inside it are occupied.
    pub fn any_can_pass_updated(&self, from: Option<i64>, to: Option<i64>) -> bool {
        if self.ids.is_empty() {
            return false;
        }
        if from.is_none() && to.is_none() {
            return true;
        }
        if !self.has_updated {
            return true;
        }
        // An empty range means every member is unknown, and unknowns fail a bounded window.
        if self.updated_min > self.updated_max {
            return false;
        }
        from.is_none_or(|f| self.updated_max > f) && to.is_none_or(|t| self.updated_min < t)
    }

    /// Member `i`'s write time in epoch ms, or `None` when it is unknown, the block carries
    /// no `updated` column, or `i` is past the end.
    pub fn member_updated(&self, i: usize) -> Option<i64> {
        match self.updated.get(i) {
            Some(&u) if u != UPDATED_UNKNOWN => Some(u),
            _ => None,
        }
    }

    /// The tags of member `i`, decoded from its bitmap. For hydrating a hit without payload.
    ///
    /// Sorted, because the dictionary is; deduplicated, because a bitmap cannot represent a
    /// repeat. Empty for an untagged member, for a block without tags, and for an `i` past
    /// the end.
    pub fn member_tags(&self, i: usize) -> Vec<String> {
        let Some(m) = self.member_bits(i) else {
            return Vec::new();
        };
        (0..self.tag_dict.len())
            .filter(|t| bit_at(m, *t))
            .map(|t| self.tag_dict[t].clone())
            .collect()
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
        let mut perp = if self.codec == Some(VectorCodec::Binary) {
            if mean_norm > 0.0 {
                let k = along_mean / mean_norm;
                query.iter().zip(&self.mean).map(|(x, m)| x - k * m).collect()
            } else {
                query.to_vec()
            }
        } else {
            Vec::new()
        };
        // Taken before the rotation, which preserves it — one fewer pass, and it keeps the
        // norm exactly consistent with the `|r|` the encoder stored the same way.
        let perp_norm = mlake_core::norm(&perp);
        if let Some(rot) = &self.rot {
            // The codes are signs in the rotated basis, so the query has to arrive there too.
            // `<R q_perp, R r> = <q_perp, r>`: the rotation changes the basis the sign
            // quantization happens in and nothing else.
            let mut scratch = vec![0.0f32; self.dim];
            rot.apply(&mut perp, &mut scratch);
        }
        Ok(PreparedQuery {
            codec: self.codec,
            dim: self.dim,
            norm: mlake_core::norm(query),
            sum: query.iter().sum(),
            abs_sum: query.iter().map(|x| x.abs()).sum(),
            mean_dot,
            mean_norm,
            along_mean,
            perp_norm,
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
        self.score_full(q, i).0
    }

    /// Lower and upper bound on the true cosine for member `i`: the true value lies in
    /// `[lo, hi]`, and [`Self::score`] lies in it too. For `F32` this is the exact value twice
    /// over — a zero-width interval.
    ///
    /// This is the method the query path narrows on. Take the top `k` by `lo`, then rerank at
    /// full precision every member whose `hi` clears the `k`-th best `lo`: nothing outside
    /// that set can belong in the true top `k`, so the rerank is *derived* rather than an
    /// oversampling factor someone guessed. `the_rerank_set_is_a_small_fraction_of_the_block`
    /// measures how large it comes out.
    ///
    /// **The interval is not equally trustworthy per codec, and the caller should know which
    /// it is holding:**
    ///
    /// * `F32` — exact.
    /// * `Int8` — an *absolute* bound. `|r_j - r̂_j| <= scale/2` holds by construction for
    ///   every coordinate, so the interval cannot be escaped, only widened by f32 rounding
    ///   (which [`BOUND_FP_SLACK`] covers).
    /// * `Binary` — a *probabilistic* bound at [`RABITQ_EPSILON`] sigma, roughly
    ///   `2 exp(-eps^2/2)` per member per query. It is the one place in this module where a
    ///   correct-looking answer can be wrong, and the measured rate is in the tests.
    ///
    /// Costs one pass over the member's code, the same pass [`Self::score`] makes, and
    /// allocates nothing.
    ///
    /// Returns `(0.0, 0.0)` wherever [`Self::score`] returns `0.0` for want of an answer — an
    /// out-of-range `i`, a degenerate vector, a query prepared against another block. That is
    /// a placeholder rather than a bound, exactly as the `0.0` score is a placeholder rather
    /// than a similarity.
    pub fn score_bounds(&self, q: &PreparedQuery, i: usize) -> (f32, f32) {
        let (_, lo, hi) = self.score_full(q, i);
        (lo, hi)
    }

    /// `(estimate, lower bound, upper bound)` — one pass, shared by both public entry points
    /// so a bound can never drift out of step with the score it brackets.
    fn score_full(&self, q: &PreparedQuery, i: usize) -> (f32, f32, f32) {
        debug_assert_eq!(q.codec, self.codec, "query prepared for a different codec");
        debug_assert_eq!(q.dim, self.dim, "query prepared for a different dim");
        if i >= self.len() || q.dim != self.dim || q.codec != self.codec || q.norm == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let codec = self.codec();
        let stride = Self::bytes_per_vector(codec, self.dim);
        let Some(code) = self.codes.get(i * stride..(i + 1) * stride) else {
            return (0.0, 0.0, 0.0);
        };
        match codec {
            VectorCodec::F32 => {
                let s = score_f32(code, q);
                (s, s, s)
            }
            VectorCodec::Int8 => score_int8_bounded(code, q),
            VectorCodec::Binary => score_binary_bounded(code, self.dim, q),
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
                // u = b / sqrt(d) the unit sign vector. In the *rotated* basis, since that is
                // where the signs were taken.
                let amp = r_norm * c / (self.dim as f32).sqrt();
                let mut r: Vec<f32> = (0..self.dim)
                    .map(|j| if bit_at(bits, j) { amp } else { -amp })
                    .collect();
                if let Some(rot) = &self.rot {
                    let mut scratch = vec![0.0f32; self.dim];
                    rot.apply_inverse(&mut r, &mut scratch);
                }
                for (x, m) in r.iter_mut().zip(&self.mean) {
                    *x += m;
                }
                r
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

/// The rotation a block of this codec and dim uses, or the override the tests supply.
///
/// Only [`VectorCodec::Binary`] rotates. [`VectorCodec::Int8`] does not, and that is a
/// decision rather than an omission — see [`score_int8_bounded`] for the reasoning.
fn rot_from(
    codec: VectorCodec,
    dim: usize,
    over: Option<Arc<Rotation>>,
) -> Option<Arc<Rotation>> {
    match (over, codec) {
        (Some(r), _) => Some(r),
        (None, VectorCodec::Binary) if dim > 0 => Some(Arc::new(Rotation::derive(dim))),
        _ => None,
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

/// The block's tag union and whether any member is untagged, derived once from the bitmaps
/// so [`VectorBlock::any_can_pass`] is `O(width)` rather than `O(members * width)`.
/// The min and max of the known values in an `updated` column. `(i64::MAX, i64::MIN)` when
/// the column is empty or every member is unknown — an empty range, which
/// [`VectorBlock::any_can_pass_updated`] reads as "nothing here can match a bounded window".
fn updated_summary(updated: &[i64]) -> (i64, i64) {
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for &u in updated {
        if u != UPDATED_UNKNOWN {
            lo = lo.min(u);
            hi = hi.max(u);
        }
    }
    (lo, hi)
}

fn tag_summary(bits: &[u8], width: usize, count: usize) -> (Vec<u8>, bool) {
    if width == 0 {
        // No dictionary: every member is untagged, and the union is empty. `count == 0` is
        // the empty block, which holds no untagged member because it holds no member.
        return (Vec::new(), count > 0);
    }
    let mut union = vec![0u8; width];
    let mut any_untagged = false;
    for row in bits.chunks_exact(width) {
        let mut zero = true;
        for (u, b) in union.iter_mut().zip(row) {
            *u |= *b;
            zero &= *b == 0;
        }
        any_untagged |= zero;
    }
    (union, any_untagged)
}

fn is_zero(bits: &[u8]) -> bool {
    bits.iter().all(|b| *b == 0)
}

/// Any request tag present in `bits`. An absent request tag sets no bit in the mask, so it
/// simply cannot contribute an overlap — which is the correct semantics, not an approximation.
fn overlaps(bits: &[u8], mask: &TagMask) -> bool {
    bits.iter().zip(&mask.bits).any(|(m, q)| m & q != 0)
}

/// Every request tag present in `bits` (request ⊆ bits).
///
/// `missing > 0` short-circuits to `false`: a request tag absent from the block's dictionary
/// is absent from every member, so no member can contain the request.
fn contains_all(bits: &[u8], mask: &TagMask) -> bool {
    mask.missing == 0 && bits.iter().zip(&mask.bits).all(|(m, q)| m & q == *q)
}

/// `bits` is exactly the request set. Requires `missing == 0` for the same reason as
/// [`contains_all`], and then plain bitmap equality: the dictionary covers every tag either
/// side can hold, so equal bitmaps mean equal sets.
fn set_eq(bits: &[u8], mask: &TagMask) -> bool {
    mask.missing == 0 && bits == mask.bits.as_slice()
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

/// One sign bit per dimension of the **rotated** block-mean-centred residual, plus the two
/// corrective terms the estimator needs (see [`score_binary_bounded`]) and the original
/// vector's true norm.
///
/// `residual` arrives already rotated; `r_norm` is `|r|`, which the rotation preserves and
/// which the caller measured before applying it.
fn encode_binary(residual: &[f32], r_norm: f32, v_norm: f32, out: &mut Vec<u8>) {
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

/// `cos(q, v) ~= (<q, mean> + offset * sum(q) + scale * <q, code>) / (|q| * |v|)`, with an
/// **absolute** error interval around it.
///
/// The numerator is the exact dot product of `q` with `mean + dequantized residual`,
/// expanded so the loop never materializes anything: `<q, mean>` and `sum(q)` are
/// precomputed per query and the remaining term is an `f32 * u8` accumulation. The
/// denominator uses the *stored true* norm, not the dequantized one, so the only error left
/// is the quantization of the residual's codes.
///
/// **The bound.** `encode_int8` picks `scale = (max - min) / 255` from the residual's own
/// range, so every coordinate lands strictly inside the quantizer's span and never clamps;
/// rounding to the nearest level therefore leaves `|r_j - r̂_j| <= scale/2` for every `j`,
/// with no distributional assumption at all. Hölder then gives
///
/// ```text
///   |<q, r - r̂>|  <=  (scale/2) * sum_j |q_j|
/// ```
///
/// and dividing by `|q| |v|` puts it on the cosine. That is a *hard* interval — the true
/// value cannot be outside it — which is why **`Int8` is deliberately left unrotated**. The
/// rotation buys RaBitQ a provable bound it otherwise has no way to state; scalar
/// quantization already has one, tighter and unconditional, straight out of its own step
/// size. Rotating would add `O(dim log dim)` to every vector encoded and every query
/// prepared, would not narrow this interval, and would cost the codec its one structural
/// advantage — that `decode` reconstructs components rather than a direction. Measured, the
/// interval is 7.3e-3 wide at dim 384, nine times tighter than the rotated binary one.
///
/// It is loose by roughly `sqrt(d)`: the per-coordinate errors are near-independent, so the
/// *typical* error is a random sum, not the aligned worst case Hölder charges for. Tightening
/// it would mean a probabilistic bound, and an absolute one is worth more here than a narrow
/// one — it is already narrower than binary's.
fn score_int8_bounded(code: &[u8], q: &PreparedQuery) -> (f32, f32, f32) {
    let (offset, scale, v_norm) = correctives(code);
    let mut acc = 0.0f32;
    for (c, qj) in code[CORRECTIVE_LEN..].iter().zip(&q.q) {
        acc += *c as f32 * qj;
    }
    let denom = q.norm * v_norm;
    if denom == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let est = (q.mean_dot + offset * q.sum + scale * acc) / denom;
    let half = (0.5 * scale * q.abs_sum) / denom + BOUND_FP_SLACK;
    (
        est.clamp(-1.0, 1.0),
        (est - half).clamp(-1.0, 1.0),
        (est + half).clamp(-1.0, 1.0),
    )
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
/// # The error bound
///
/// This is what [`VectorBlock::score_bounds`] returns, and the reason [`Rotation`] exists.
/// Write `o = r/|r|` and `u = b/sqrt(d)`, both unit, `c = <o, u>` the stored corrective, and
/// decompose the code direction against the residual it encodes:
///
/// ```text
///   u = c * o + s * w,      s = sqrt(1 - c^2),   w unit, w ⊥ o
/// ```
///
/// For the unit query direction `p = q_perp/|q_perp|`, that makes the estimator's error
/// exactly one term:
///
/// ```text
///   <p, u>/c = <p, o> + (s/c) * <p, w>
///                       ^^^^^^^^^^^^^^ all of it
/// ```
///
/// `w` is a unit vector orthogonal to `o`, and it is *entirely* determined by which orthant
/// the rotated residual fell in. Rotate the space randomly and `w` is distributed uniformly on
/// the unit sphere of `o`'s orthogonal complement — a `d-1` dimensional sphere — independently
/// of where the query sits. A coordinate of a uniform point on `S^{d-2}` concentrates:
/// `P(|<p, w>| > t) <= 2 exp(-(d-1) t^2 / 2)`. Substituting `t = eps / sqrt(d-1)`:
///
/// ```text
///   | <p, o> - <p, u>/c |  <=  eps * sqrt(1 - c^2) / (c * sqrt(d - 1))
/// ```
///
/// with probability at least `1 - 2 exp(-eps^2 / 2)`. That is RaBitQ's Theorem 3.2, and it is
/// the formula implemented below with `eps = ` [`RABITQ_EPSILON`]. Scaling back onto the
/// cosine, the half-width of the returned interval is
///
/// ```text
///   |q_perp| * |r| * eps * sqrt(1 - c^2) / (c * sqrt(d - 1) * |q| * |v|)
/// ```
///
/// Everything in it is either stored (`|r|`, `c`, `|v|`) or per-query (`|q_perp|`, `|q|`), so
/// the bound costs no extra bytes and no extra pass.
///
/// Four honest caveats, in descending order of how much they should worry a caller:
///
/// 1. **It is probabilistic.** At `eps = 5` the per-member failure probability is `2e^-12.5`,
///    about 7.5e-6 — but it is a *tail* bound on an idealized rotation, not a guarantee. The
///    measured containment rate is the number to trust, and it is pinned in the tests.
/// 2. **One rotation serves the whole block and every query.** Failures are therefore not
///    independent across members: an unlucky rotation is unlucky for a correlated set of them,
///    so the effective failure rate over a *query* is worse than `members * 7.5e-6` would
///    suggest.
/// 3. **The rotation is randomized-Hadamard, not uniform on `O(d)`.** The uniformity `w`'s
///    distribution needs is approached, not attained.
/// 4. **`|<p, w>|` is bounded using `|p| <= 1`** where `|p_perp_to_o| = sqrt(1 - <p,o>^2)`
///    would be tighter — materially so for exactly the high-scoring members that matter. It is
///    left loose because tightening it means feeding the estimate back into its own bound.
///
/// The `1 - 2 exp(-eps^2/2)` is per (member, query); the interval is clamped to `[-1, 1]`,
/// beyond which no cosine can be anyway, so the bound degrades to the trivially true one
/// rather than to a wrong one.
///
/// Its derivation: decompose the unit `q_perp` into its component along `r` and a remainder
/// `w`, giving `<q_perp/|q_perp|, u> = cos(q_perp, r) * cos(r, u) + <w, u>`; dividing by the
/// stored `cos(r, u)` recovers `cos(q_perp, r)` up to `<w, u>`, which has mean zero when the
/// code's error direction is uncorrelated with the query. RaBitQ *guarantees* that with a
/// random rotation, and since v3 so do we: the residual's signs are taken in the basis
/// [`Rotation`] picks, not in the embedding model's, which is what turns "we assume the error
/// is isotropic" into "we made it so".
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
fn score_binary_bounded(code: &[u8], dim: usize, q: &PreparedQuery) -> (f32, f32, f32) {
    let (r_norm, c, v_norm) = correctives(code);
    let denom = q.norm * v_norm;
    if denom == 0.0 || dim == 0 {
        return (0.0, 0.0, 0.0);
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
        // the query has no component off it. Either way `known` is the whole answer — and,
        // being exact, it is its own bound.
        let s = (known / denom).clamp(-1.0, 1.0);
        return (s, (s - BOUND_FP_SLACK).max(-1.0), (s + BOUND_FP_SLACK).min(1.0));
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
    let raw = acc / (q.perp_norm * sqrt_d * c);
    // eps * sqrt(1 - c^2) / (c * sqrt(d - 1)) — the whole bound, three loads and a divide.
    // At dim 1 there is no orthogonal complement for the error to live in and no
    // concentration to appeal to, so the interval opens to everything a cosine can be.
    let half = if dim > 1 {
        RABITQ_EPSILON * (1.0 - c * c).max(0.0).sqrt() / (c * ((dim - 1) as f32).sqrt())
    } else {
        2.0
    };
    // Clamping the *cosine* before scaling, not just the final score: cos(q_perp, r) cannot
    // leave [-1, 1], so the interval never widens past the trivial `known ± |q_perp| |r|`.
    let cosine = |x: f32| {
        ((known + q.perp_norm * r_norm * x.clamp(-1.0, 1.0)) / denom).clamp(-1.0, 1.0)
    };
    let est = cosine(raw);
    let lo = (cosine(raw - half) - BOUND_FP_SLACK).max(-1.0);
    let hi = (cosine(raw + half) + BOUND_FP_SLACK).min(1.0);
    (est, lo.min(est), hi.max(est))
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

    /// Recall@10 of the [`VectorCodec::Binary`] ranking, same corpus. Measured: 0.542
    /// (0.513 before the random rotation).
    ///
    /// This is *not* a gate anyone should serve from: 1 bit/dim cannot resolve the top ten
    /// of a cluster, and no amount of estimator work will change that (the arithmetic is in
    /// the test that pins it). Binary is a candidate generator — see
    /// [`BINARY_RECALL_AT_10_OVERSAMPLED`], which is the number Phase 3 actually depends on.
    const BINARY_RECALL_AT_10: f32 = 0.48;
    /// Fraction of the true top 10 present in binary's top 40 — 4x oversampling, the size of
    /// the set Phase 3 would hand to a full-precision rerank. Measured: 1.000.
    const BINARY_RECALL_AT_10_OVERSAMPLED: f32 = 0.99;
    /// Spearman correlation of the binary scores with the exact scores. Measured: 0.870.
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

    // --- tags -------------------------------------------------------------------------

    const MODES: [TagsMatch; 5] = [
        TagsMatch::Any,
        TagsMatch::All,
        TagsMatch::AnyStrict,
        TagsMatch::AllStrict,
        TagsMatch::Exact,
    ];

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Vectors nobody looks at: the tag tests are about the column, not the codec.
    fn filler(n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| vec![i as f32, 1.0]).collect()
    }

    fn tagged_block(member_tags: &[Vec<String>]) -> VectorBlock {
        let n = member_tags.len();
        VectorBlock::encode_with_tags(VectorCodec::F32, 2, &ids(n), &filler(n), member_tags)
            .unwrap()
    }

    /// The property the whole column rests on: the bitwise path is not *approximately*
    /// `TagFilter`, it is `TagFilter`.
    fn assert_agrees_with_reference(member_tags: &[Vec<String>], request: &[String]) {
        let block = tagged_block(member_tags);
        let mask = block.tag_mask(request);
        for mode in MODES {
            let reference = mlake_core::TagFilter::new(request.to_vec(), mode);
            for (i, mt) in member_tags.iter().enumerate() {
                assert_eq!(
                    block.passes(i, &mask, mode),
                    reference.matches(mt),
                    "{mode:?}: member {i} {mt:?} against request {request:?}"
                );
            }
            if !block.any_can_pass(&mask, mode) {
                assert!(
                    (0..block.len()).all(|i| !block.passes(i, &mask, mode)),
                    "{mode:?}: any_can_pass said no member could pass request {request:?}, \
                     but one does"
                );
            }
        }
    }

    #[test]
    fn the_bitwise_filter_agrees_with_tag_filter_on_every_mode() {
        // The test that matters. If the bitwise path disagrees with `TagFilter` anywhere the
        // filter is wrong and silently so, so this sweeps randomized tag sets rather than
        // hand-picked ones: overlapping, disjoint, subset, superset, duplicated and empty
        // member sets, against requests that are sometimes drawn from the same vocabulary and
        // sometimes not.
        let vocab: Vec<String> = (0..12).map(|i| format!("t{i}")).collect();
        let mut rng = Rng::seeded(0xA65);
        for _ in 0..400 {
            let members = 1 + rng.below(9);
            let member_tags: Vec<Vec<String>> = (0..members)
                .map(|_| {
                    // A quarter of members are untagged: the strict modes exist for them.
                    if rng.below(4) == 0 {
                        return Vec::new();
                    }
                    let k = 1 + rng.below(4);
                    (0..k).map(|_| vocab[rng.below(vocab.len())].clone()).collect()
                })
                .collect();
            // Requests draw from a wider vocabulary than the members do, so roughly a third
            // of request tags are absent from the block's dictionary.
            let req_len = rng.below(4);
            let request: Vec<String> = (0..req_len)
                .map(|_| {
                    let i = rng.below(vocab.len() + 6);
                    vocab.get(i).cloned().unwrap_or_else(|| format!("absent{i}"))
                })
                .collect();
            assert_agrees_with_reference(&member_tags, &request);
        }
    }

    #[test]
    fn untagged_members_are_treated_per_mode_exactly_as_the_reference_does() {
        // Pinned separately from the randomized sweep because this is the asymmetry the whole
        // five-mode design exists for, and a bug here is invisible on tagged corpora.
        let member_tags = vec![Vec::new(), tags(&["a"])];
        let block = tagged_block(&member_tags);
        let mask = block.tag_mask(&tags(&["a"]));
        // Any/All include the untagged member; the strict modes and Exact exclude it.
        assert!(block.passes(0, &mask, TagsMatch::Any));
        assert!(block.passes(0, &mask, TagsMatch::All));
        assert!(!block.passes(0, &mask, TagsMatch::AnyStrict));
        assert!(!block.passes(0, &mask, TagsMatch::AllStrict));
        assert!(!block.passes(0, &mask, TagsMatch::Exact));
        for mode in MODES {
            assert!(block.passes(1, &mask, mode), "{mode:?}: the tagged member matches");
        }

        // An empty request is "no filter" everywhere except Exact, where it is the untagged
        // scope — so it inverts which member survives.
        let empty = block.tag_mask(&[]);
        for mode in MODES {
            let want_untagged = mode == TagsMatch::Exact;
            assert!(block.passes(0, &empty, mode), "{mode:?} on untagged");
            assert_eq!(block.passes(1, &empty, mode), !want_untagged, "{mode:?} on tagged");
            assert!(block.any_can_pass(&empty, mode), "{mode:?}");
        }
        assert_agrees_with_reference(&member_tags, &[]);
        assert_agrees_with_reference(&member_tags, &tags(&["a"]));
    }

    #[test]
    fn a_block_whose_members_are_all_untagged_still_filters() {
        // Empty dictionary, zero-width bitmaps — and still a filterable fact, which is why
        // `has_tags` is a header flag rather than "the dictionary is non-empty".
        let member_tags = vec![Vec::new(), Vec::new()];
        let block = tagged_block(&member_tags);
        assert!(block.has_tags());
        assert!(block.tag_dictionary().is_empty());
        assert_eq!(block.tag_bitmap_width(), 0);
        let mask = block.tag_mask(&tags(&["a"]));
        for mode in [TagsMatch::AnyStrict, TagsMatch::AllStrict, TagsMatch::Exact] {
            assert!(!block.any_can_pass(&mask, mode), "{mode:?}");
            assert!(!block.passes(0, &mask, mode), "{mode:?}");
        }
        for mode in [TagsMatch::Any, TagsMatch::All] {
            assert!(block.passes(0, &mask, mode), "{mode:?} includes untagged");
        }
        assert_agrees_with_reference(&member_tags, &tags(&["a"]));
    }

    #[test]
    fn a_request_tag_absent_from_the_dictionary_is_exact_not_approximate() {
        let member_tags = vec![tags(&["a", "b"]), tags(&["a"])];
        let block = tagged_block(&member_tags);
        assert_eq!(block.tag_dictionary(), tags(&["a", "b"]));

        // "a" is present, "zz" is not. Any/AnyStrict: the absent tag contributes no bit and
        // the present one still decides.
        let mixed = block.tag_mask(&tags(&["a", "zz"]));
        assert!(block.passes(0, &mixed, TagsMatch::Any));
        assert!(block.passes(0, &mixed, TagsMatch::AnyStrict));
        assert!(block.any_can_pass(&mixed, TagsMatch::AnyStrict));
        // All/AllStrict/Exact need the absent tag, so nothing in the block can match — and
        // any_can_pass must say so, because skipping the block is the win.
        for mode in [TagsMatch::All, TagsMatch::AllStrict, TagsMatch::Exact] {
            assert!(!block.passes(0, &mixed, mode), "{mode:?}");
            assert!(!block.passes(1, &mixed, mode), "{mode:?}");
        }
        assert!(!block.any_can_pass(&mixed, TagsMatch::AllStrict));
        assert!(!block.any_can_pass(&mixed, TagsMatch::Exact));
        // `All` still admits the block if it holds an untagged member — and here it does not.
        assert!(!block.any_can_pass(&mixed, TagsMatch::All));

        // Every request tag absent: even the overlap modes can prune the whole block.
        let none = block.tag_mask(&tags(&["zz", "yy"]));
        for mode in MODES {
            assert!(!block.any_can_pass(&none, mode), "{mode:?}");
        }
        assert_agrees_with_reference(&member_tags, &tags(&["a", "zz"]));
        assert_agrees_with_reference(&member_tags, &tags(&["zz", "yy"]));
    }

    #[test]
    fn any_can_pass_is_never_optimistic() {
        // The one direction that has to hold: `false` is a licence to skip the block without
        // scoring it, so a false negative silently drops results.
        let vocab: Vec<String> = (0..6).map(|i| format!("t{i}")).collect();
        let mut rng = Rng::seeded(99);
        for _ in 0..300 {
            let members = 1 + rng.below(6);
            let member_tags: Vec<Vec<String>> = (0..members)
                .map(|_| {
                    if rng.below(5) == 0 {
                        return Vec::new();
                    }
                    (0..1 + rng.below(3))
                        .map(|_| vocab[rng.below(vocab.len())].clone())
                        .collect()
                })
                .collect();
            let block = tagged_block(&member_tags);
            let request: Vec<String> = (0..1 + rng.below(3))
                .map(|_| {
                    let i = rng.below(vocab.len() + 3);
                    vocab.get(i).cloned().unwrap_or_else(|| format!("absent{i}"))
                })
                .collect();
            let mask = block.tag_mask(&request);
            for mode in MODES {
                if !block.any_can_pass(&mask, mode) {
                    for (i, mt) in member_tags.iter().enumerate() {
                        assert!(
                            !block.passes(i, &mask, mode),
                            "{mode:?}: block skipped but member {i} {mt:?} passes {request:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_block_encoded_without_tags_filters_nothing() {
        // No filter information is "no filtering", not "filter everything out" — the opposite
        // choice would make an unmigrated block return zero results instead of all of them.
        let block = VectorBlock::encode(VectorCodec::F32, 2, &ids(3), &filler(3)).unwrap();
        assert!(!block.has_tags());
        assert!(block.tag_dictionary().is_empty());
        let mask = block.tag_mask(&tags(&["a", "b"]));
        for mode in MODES {
            assert!(block.any_can_pass(&mask, mode), "{mode:?}");
            for i in 0..block.len() {
                assert!(block.passes(i, &mask, mode), "{mode:?} member {i}");
            }
        }
        assert!(block.member_tags(0).is_empty());
    }

    #[test]
    fn member_tags_decode_back_to_the_set_that_was_encoded() {
        let member_tags = vec![
            tags(&["zeta", "alpha"]),
            Vec::new(),
            tags(&["alpha", "alpha", "mid"]),
        ];
        let block = tagged_block(&member_tags);
        assert_eq!(block.tag_dictionary(), tags(&["alpha", "mid", "zeta"]));
        assert_eq!(block.member_tags(0), tags(&["alpha", "zeta"]), "sorted, not input order");
        assert_eq!(block.member_tags(1), Vec::<String>::new());
        assert_eq!(block.member_tags(2), tags(&["alpha", "mid"]), "duplicates collapse");
        assert_eq!(block.member_tags(9), Vec::<String>::new(), "no such member");
    }

    #[test]
    fn out_of_range_members_do_not_pass_and_do_not_panic() {
        let block = tagged_block(&[tags(&["a"])]);
        let mask = block.tag_mask(&tags(&["a"]));
        for mode in MODES {
            assert!(!block.passes(1, &mask, mode), "{mode:?}");
            assert!(!block.passes(usize::MAX, &mask, mode), "{mode:?}");
        }
        let empty = tagged_block(&[]);
        assert!(empty.has_tags());
        for mode in MODES {
            assert!(!empty.any_can_pass(&empty.tag_mask(&tags(&["a"])), mode), "{mode:?}");
        }
    }

    #[test]
    fn the_tag_column_round_trips_through_bytes() {
        let member_tags: Vec<Vec<String>> = (0..40)
            .map(|i| match i % 4 {
                0 => Vec::new(),
                1 => tags(&["shared"]),
                2 => vec![format!("uniq{i}"), "shared".to_string()],
                _ => tags(&["shared", "other", "utf8-ünïcødé-🏷"]),
            })
            .collect();
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let (vectors, _) = corpus(40, DIM, TOPICS, 0.5, 3);
            let block =
                VectorBlock::encode_with_tags(codec, DIM, &ids(40), &vectors, &member_tags)
                    .unwrap();
            let back = VectorBlock::from_bytes(&block.to_bytes()).unwrap();
            assert_eq!(back, block, "{codec:?}");
            assert!(back.has_tags());
            assert_eq!(back.tag_dictionary(), block.tag_dictionary());
            for (i, mt) in member_tags.iter().enumerate() {
                let mut want = mt.clone();
                want.sort();
                want.dedup();
                assert_eq!(back.member_tags(i), want, "{codec:?} member {i}");
            }
            // Scoring is untouched by the tag column riding alongside.
            let plain = VectorBlock::encode(codec, DIM, &ids(40), &vectors).unwrap();
            assert_eq!(back.codes, plain.codes, "{codec:?}");
        }
    }

    #[test]
    fn a_truncated_or_corrupt_tag_column_is_an_error_not_a_panic() {
        let member_tags: Vec<Vec<String>> = (0..20)
            .map(|i| vec![format!("t{}", i % 5), "shared".to_string()])
            .collect();
        let (vectors, _) = corpus(20, 16, TOPICS, 0.5, 2);
        let bytes = VectorBlock::encode_with_tags(VectorCodec::Int8, 16, &ids(20), &vectors, &member_tags)
            .unwrap()
            .to_bytes();
        // Every prefix, not a sampled few: the tag column is the one variable-length section
        // in the format, so its cursor is the part most likely to walk off the end.
        for cut in 0..bytes.len() {
            assert!(
                VectorBlock::from_bytes(&bytes[..cut]).is_err(),
                "a {cut}-byte prefix must be rejected, never indexed into"
            );
        }
        assert!(VectorBlock::from_bytes(&bytes).is_ok());

        let mut extra = bytes.clone();
        extra.push(0);
        assert!(VectorBlock::from_bytes(&extra).is_err(), "trailing bytes");

        // A v1 block: no compatibility to preserve, so it is rejected rather than guessed at.
        let mut old = bytes.clone();
        old[4] = 1;
        assert!(matches!(
            VectorBlock::from_bytes(&old),
            Err(mlake_core::Error::FormatVersion { .. })
        ));

        // An unknown flag bit means a writer that knew something this reader does not.
        let mut future = bytes.clone();
        future[6] = 0b1000_0001;
        assert!(VectorBlock::from_bytes(&future).is_err());

        // The dictionary count is attacker-controlled in the same way `count` and `dim` are.
        let dict_at = HEADER_LEN + 20 * ID_LEN + 16 * 4;
        for value in [u32::MAX, 1 << 20, 0] {
            let mut lying = bytes.clone();
            lying[dict_at..dict_at + 4].copy_from_slice(&value.to_le_bytes());
            assert!(
                VectorBlock::from_bytes(&lying).is_err(),
                "a dictionary count of {value} must not be trusted"
            );
        }
        // An individual entry length, likewise.
        let mut long_entry = bytes.clone();
        long_entry[dict_at + 4..dict_at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(VectorBlock::from_bytes(&long_entry).is_err());

        // Invalid utf-8 in a dictionary entry.
        let mut bad_utf8 = bytes.clone();
        bad_utf8[dict_at + 8] = 0xFF;
        assert!(VectorBlock::from_bytes(&bad_utf8).is_err());

        // An out-of-order dictionary would break the binary search in `tag_mask`, which fails
        // by silently missing a filter tag — the worst possible failure mode for a filter.
        let mut unsorted = bytes.clone();
        unsorted[dict_at + 8] = b'z';
        assert!(VectorBlock::from_bytes(&unsorted).is_err());
    }

    #[test]
    fn the_tag_column_encodes_deterministically() {
        // G-6: byte-identical replays. The dictionary is a sorted set, so neither the order a
        // member lists its tags in nor the order members appear in can perturb the bytes.
        let a = vec![tags(&["b", "a", "a"]), tags(&["c"]), Vec::new()];
        let b = vec![tags(&["a", "b"]), tags(&["c"]), Vec::new()];
        assert_eq!(tagged_block(&a).to_bytes(), tagged_block(&b).to_bytes());
        assert_eq!(tagged_block(&a).to_bytes(), tagged_block(&a).to_bytes());
    }

    fn updated_block(member_updated: &[i64]) -> VectorBlock {
        let n = member_updated.len();
        VectorBlock::encode_with_columns(
            VectorCodec::F32,
            2,
            &ids(n),
            &filler(n),
            None,
            member_updated,
        )
        .unwrap()
    }

    /// A member as `Predicate` sees it, so the block can be checked against the definition
    /// rather than against a restatement of itself.
    fn reference_memory(updated: i64) -> mlake_core::StoredMemory {
        mlake_core::StoredMemory {
            id: MemoryId::from_key("m"),
            vector: vec![1.0, 0.0],
            text: String::new(),
            index_text: String::new(),
            memory_type: 1,
            tags: vec![],
            timestamps: mlake_core::memory::Timestamps {
                updated_at: (updated != UPDATED_UNKNOWN).then_some(updated),
                ..Default::default()
            },
            proof_count: 0,
            entity_ids: vec![],
            semantic_out: vec![],
            causal_out: vec![],
            metadata: vec![],
            write_seq: 0,
        }
    }

    #[test]
    fn the_window_filter_agrees_with_predicate_on_every_bound() {
        // The property the push-down rests on: the block-side window is not *approximately*
        // `Predicate`'s window, it is the same predicate evaluated earlier. Swept over open,
        // half-open and closed windows, boundary values included, with unknowns mixed in —
        // the strictness of `>` / `<` and the treatment of an unknown write time are exactly
        // the two things that would be silently wrong if they drifted.
        let members: Vec<i64> = vec![10, 20, 20, 30, UPDATED_UNKNOWN, 40];
        let block = updated_block(&members);
        let bounds = [None, Some(9), Some(10), Some(20), Some(30), Some(40), Some(41)];
        for from in bounds {
            for to in bounds {
                for (i, &u) in members.iter().enumerate() {
                    let reference = mlake_core::Predicate {
                        updated_from: from,
                        updated_to: to,
                        ..Default::default()
                    }
                    .matches(&reference_memory(u));
                    assert_eq!(
                        block.passes_updated(i, from, to),
                        reference,
                        "member {i} ({u}) against ({from:?}, {to:?})"
                    );
                }
                if !block.any_can_pass_updated(from, to) {
                    assert!(
                        (0..block.len()).all(|i| !block.passes_updated(i, from, to)),
                        "any_can_pass_updated said no member could pass ({from:?}, {to:?}), \
                         but one does"
                    );
                }
            }
        }
    }

    #[test]
    fn a_block_written_outside_the_window_is_skipped_whole() {
        // What the column is for: two integer compares retire a cluster before it is scored.
        let block = updated_block(&[100, 150, 200]);
        assert!(!block.any_can_pass_updated(Some(200), None), "entirely before `from`");
        assert!(!block.any_can_pass_updated(None, Some(100)), "entirely after `to`");
        assert!(!block.any_can_pass_updated(Some(200), Some(300)), "window past the block");
        assert!(block.any_can_pass_updated(Some(150), None), "one member is inside");
        assert!(block.any_can_pass_updated(None, None), "unbounded admits everything");
    }

    #[test]
    fn a_block_whose_members_are_all_unknown_passes_nothing_bounded() {
        // An empty known-range must not read as "the whole line": a bounded window excludes
        // an unknown write time, so the block can be retired outright.
        let block = updated_block(&[UPDATED_UNKNOWN, UPDATED_UNKNOWN]);
        assert!(!block.any_can_pass_updated(Some(0), None));
        assert!(!block.any_can_pass_updated(None, Some(i64::MAX)));
        assert!(block.any_can_pass_updated(None, None), "unbounded still admits them");
        assert!(block.passes_updated(0, None, None));
    }

    #[test]
    fn a_block_encoded_without_the_column_filters_nothing() {
        // Blocks written before the column existed are undecidable here, so they must admit
        // every member and leave the window to the payload-side check. Dropping them instead
        // would silently lose hits across a format upgrade.
        let block = VectorBlock::encode(VectorCodec::F32, 2, &ids(3), &filler(3)).unwrap();
        assert!(block.any_can_pass_updated(Some(0), Some(1)));
        assert!((0..3).all(|i| block.passes_updated(i, Some(0), Some(1))));
        assert!(block.member_updated(0).is_none());
    }

    #[test]
    fn an_index_past_the_end_is_admitted_by_nothing() {
        let block = updated_block(&[10]);
        assert!(!block.passes_updated(1, None, None));
        assert!(!block.passes_updated(1, Some(0), Some(100)));
        assert!(block.member_updated(1).is_none());
    }

    #[test]
    fn the_updated_column_round_trips_through_bytes() {
        let members = vec![10, UPDATED_UNKNOWN, 30];
        let block = updated_block(&members);
        let back = VectorBlock::from_bytes(&block.to_bytes()).unwrap();
        assert_eq!(back, block);
        for (i, &u) in members.iter().enumerate() {
            assert_eq!(back.member_updated(i), (u != UPDATED_UNKNOWN).then_some(u));
        }
        assert!(!back.any_can_pass_updated(Some(30), None));
    }

    #[test]
    fn both_columns_round_trip_together() {
        // The two columns are serialized back to back ahead of the codes; this is the case
        // that catches a reader walking them in the wrong order or off by a section.
        let member_tags = vec![tags(&["a"]), Vec::new(), tags(&["a", "b"])];
        let block = VectorBlock::encode_with_columns(
            VectorCodec::F32,
            2,
            &ids(3),
            &filler(3),
            Some(&member_tags),
            &[10, UPDATED_UNKNOWN, 30],
        )
        .unwrap();
        let back = VectorBlock::from_bytes(&block.to_bytes()).unwrap();
        assert_eq!(back, block);
        assert_eq!(back.member_tags(2), tags(&["a", "b"]));
        assert_eq!(back.member_updated(2), Some(30));
        assert_eq!(back.codes, block.codes, "the codes must still be the tail");
    }

    #[test]
    fn a_truncated_updated_column_is_an_error_not_a_panic() {
        let bytes = updated_block(&[10, 20, 30]).to_bytes();
        for cut in 1..=(3 * 8) {
            let err = VectorBlock::from_bytes(&bytes[..bytes.len() - cut]);
            assert!(err.is_err(), "truncating {cut} bytes decoded anyway");
        }
        // A header that claims the column against a body that does not carry it.
        let mut lying = updated_block(&[10]).to_bytes();
        let plain = VectorBlock::encode(VectorCodec::F32, 2, &ids(1), &filler(1))
            .unwrap()
            .to_bytes();
        lying.truncate(plain.len());
        assert!(VectorBlock::from_bytes(&lying).is_err());
    }

    #[test]
    fn the_updated_column_encodes_deterministically() {
        assert_eq!(updated_block(&[3, 1, 2]).to_bytes(), updated_block(&[3, 1, 2]).to_bytes());
    }

    #[test]
    fn encode_with_columns_rejects_a_column_that_disagrees_with_the_member_count() {
        let err = VectorBlock::encode_with_columns(
            VectorCodec::F32,
            2,
            &ids(2),
            &filler(2),
            None,
            &[1],
        )
        .unwrap_err();
        assert!(matches!(err, mlake_core::Error::Encode(_)), "got {err:?}");
    }

    #[test]
    fn the_updated_column_costs_eight_bytes_a_member() {
        // The number the size decision was made against, measured rather than argued.
        let with = updated_block(&vec![1i64; N]).to_bytes().len();
        let without = VectorBlock::encode(VectorCodec::F32, 2, &ids(N), &filler(N))
            .unwrap()
            .to_bytes()
            .len();
        assert_eq!(with - without, N * 8);
    }

    #[test]
    fn encode_with_tags_rejects_a_tag_list_that_disagrees_with_the_member_count() {
        let err = VectorBlock::encode_with_tags(
            VectorCodec::F32,
            2,
            &ids(2),
            &filler(2),
            &[tags(&["a"])],
        )
        .unwrap_err();
        assert!(matches!(err, mlake_core::Error::Encode(_)), "got {err:?}");
    }

    #[test]
    fn the_bitmap_costs_what_we_claim_per_member() {
        // The number the uncapped-width decision is being made against, measured rather than
        // argued. A realistic cluster: 500 members drawn from a 40-tag vocabulary, 2-3 tags
        // each — the shape a namespace with a few dozen topic labels actually has.
        let vocab: Vec<String> = (0..40).map(|i| format!("tag{i}")).collect();
        let mut rng = Rng::seeded(4242);
        let member_tags: Vec<Vec<String>> = (0..N)
            .map(|_| {
                (0..2 + rng.below(2))
                    .map(|_| vocab[rng.below(vocab.len())].clone())
                    .collect()
            })
            .collect();
        let (vectors, _) = block_corpus(11);
        let tagged =
            VectorBlock::encode_with_tags(VectorCodec::Binary, DIM, &ids(N), &vectors, &member_tags)
                .unwrap();
        let plain = VectorBlock::encode(VectorCodec::Binary, DIM, &ids(N), &vectors).unwrap();
        let overhead = (tagged.to_bytes().len() - plain.to_bytes().len()) as f32 / N as f32;
        assert_eq!(tagged.tag_dictionary().len(), 40);
        assert_eq!(tagged.tag_bitmap_width(), 5, "ceil(40/8)");
        // Measured: 5.708 B/member — 5 B of bitmap plus 0.708 B of amortized dictionary
        // (354 B of strings over 500 members) — against the 60 B binary code it rides
        // beside, i.e. 9.5% of the code and 3.6% of the whole 74 B/member block.
        assert!((5.0..6.0).contains(&overhead), "tag overhead {overhead} B/member");
        let ratio = overhead / VectorBlock::bytes_per_vector(VectorCodec::Binary, DIM) as f32;
        assert!(ratio < 0.10, "tag column is {ratio} of the code it rides beside");

        // The worst case the doc comment warns about, so the ceiling is a measured number
        // rather than an assertion: every member carrying a unique tag makes the width
        // count/8, which at 500 members is more than the code itself.
        let unique: Vec<Vec<String>> = (0..N).map(|i| vec![format!("u{i}")]).collect();
        let worst =
            VectorBlock::encode_with_tags(VectorCodec::Binary, DIM, &ids(N), &vectors, &unique)
                .unwrap();
        assert_eq!(worst.tag_bitmap_width(), N.div_ceil(8), "63 B/member at N=500");
        assert!(
            worst.tag_bitmap_width() > VectorBlock::bytes_per_vector(VectorCodec::Binary, DIM),
            "if this stopped being true the uncapped width would need no caveat"
        );
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
        // Measured: ours 0.477, naive 0.070 — 6.8x. Int8 is unbothered at 0.980.
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
        // for the residual — the mean is exact. Measured worst case: cosine 0.962, which the
        // rotation leaves alone: `apply_inverse` puts the reconstruction back in the original
        // basis, so this measures the code, not the frame it was taken in.
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
        // Measured: 0.077, against 0.019 on a normalized block.
        assert!(worst > 0.02, "if this stopped being true the caveat could go: {worst}");
        assert!(worst < 0.15, "binary worst score error {worst}");
    }

    // --- the rotation -------------------------------------------------------------------

    /// Every property that makes the construction a *rotation* rather than merely a shuffle:
    /// it preserves norms and inner products, and it is invertible. If any of these fails the
    /// error bound below is meaningless, because the bound is stated about an orthogonal map.
    #[test]
    fn the_rotation_is_orthogonal_and_invertible() {
        // Powers of two, a sum of two, a sum of three, an odd tail, and the degenerate ends.
        for dim in [1usize, 2, 3, 5, 8, 100, 127, 384, 768, 1000] {
            let rot = Rotation::derive(dim);
            assert_eq!(hadamard_segments(dim).iter().sum::<usize>(), dim, "dim {dim}");
            let mut rng = Rng::seeded(dim as u64 + 1);
            let a: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect();
            let b: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect();
            let mut scratch = vec![0.0f32; dim];
            let (mut ra, mut rb) = (a.clone(), b.clone());
            rot.apply(&mut ra, &mut scratch);
            rot.apply(&mut rb, &mut scratch);

            let tol = 1e-4 * (dim as f32).sqrt();
            assert!(
                (mlake_core::norm(&ra) - mlake_core::norm(&a)).abs() < tol,
                "dim {dim}: |Rv| = {} but |v| = {}",
                mlake_core::norm(&ra),
                mlake_core::norm(&a)
            );
            assert!(
                (mlake_core::dot(&ra, &rb) - mlake_core::dot(&a, &b)).abs() < tol,
                "dim {dim}: <Ra, Rb> must equal <a, b>"
            );
            let mut back = ra.clone();
            rot.apply_inverse(&mut back, &mut scratch);
            let worst = back
                .iter()
                .zip(&a)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < tol, "dim {dim}: R^T R != I, worst component {worst}");
        }
    }

    /// The rotation's whole job: make the sign quantization see an isotropic vector whatever
    /// the embedding model's coordinate system looks like.
    ///
    /// `c = cos(r, sign(r)/sqrt(d))` is the direct measure of that. For a genuinely random
    /// direction it concentrates hard at `sqrt(2/pi) = 0.798`; for a vector whose energy is
    /// concentrated in a few coordinates it collapses toward `1/sqrt(d)`, and with it the
    /// fraction of the residual one bit per dimension can carry. Real embeddings are not
    /// adversarial, but they are not isotropic either, and the bound is only as good as this.
    #[test]
    fn the_rotation_makes_a_spiky_residual_behave_like_a_random_direction() {
        let dim = 384;
        let rot = Rotation::derive(dim);
        let mut scratch = vec![0.0f32; dim];
        // The pathological case a rotation exists to defuse: all the energy in 4 coordinates.
        let mut spiky = vec![0.0f32; dim];
        for j in 0..4 {
            spiky[j * 7] = 1.0;
        }
        let c_of = |v: &[f32]| {
            let n = mlake_core::norm(v);
            v.iter().map(|x| x.abs()).sum::<f32>() / ((dim as f32).sqrt() * n)
        };
        let before = c_of(&spiky);
        rot.apply(&mut spiky, &mut scratch);
        let after = c_of(&spiky);
        // Measured: 0.102 -> 0.797, against the isotropic ideal of sqrt(2/pi) = 0.798. This
        // is the measurement that sets ROTATION_ROUNDS: at one round the same spike comes out
        // at 0.90, because a single block-diagonal Hadamard cannot move energy from the
        // 256-segment into the 128-segment. Two rounds is where it lands.
        assert!(before < 0.15, "the unrotated spike should be terrible: {before}");
        assert!(
            (after - 0.7979).abs() < 0.03,
            "rotated c = {after}, wanted sqrt(2/pi)"
        );
    }

    /// G-6 across processes, not merely across calls: the rotation is a pure function of a
    /// constant seed and `dim`, driven by this crate's own PRNG, so it is the same on every
    /// machine and every run. Pinned by digest — if this number moves, every
    /// [`VectorCodec::Binary`] block ever written has been invalidated and
    /// [`FORMAT_VERSION`] must move with it.
    #[test]
    fn the_rotation_is_pinned_across_processes() {
        fn digest(rot: &Rotation) -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut eat = |b: u8| {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            };
            for r in &rot.rounds {
                for p in &r.perm {
                    for b in p.to_le_bytes() {
                        eat(b);
                    }
                }
                for s in &r.signs {
                    eat(if *s < 0.0 { 0 } else { 1 });
                }
            }
            h
        }
        assert_eq!(digest(&Rotation::derive(384)), 0x5c07_2c39_439c_77f8);
        // Two dims must not share a rotation prefix, or a truncated-embedding corpus would
        // reuse the same basis.
        assert_ne!(digest(&Rotation::derive(384)), digest(&Rotation::derive(385)));
    }

    #[test]
    fn rotating_does_not_change_what_a_block_costs_or_how_it_round_trips() {
        // The reason the rotation is segmented rather than zero-padded to 512: padding would
        // have widened every binary code by a third, permanently, and the storage tier is
        // built against `bytes_per_vector`.
        assert_eq!(VectorBlock::bytes_per_vector(VectorCodec::Binary, DIM), 60);
        let (vectors, _) = block_corpus(3);
        let block = VectorBlock::encode(VectorCodec::Binary, DIM, &ids(N), &vectors).unwrap();
        let back = VectorBlock::from_bytes(&block.to_bytes()).unwrap();
        assert_eq!(back, block, "the rotation is rederived on read, not stored");
        let q = block.prepare(&vectors[0]).unwrap();
        let p = back.prepare(&vectors[0]).unwrap();
        for i in 0..block.len() {
            assert_eq!(block.score(&q, i), back.score(&p, i), "member {i}");
        }
    }

    /// Recall with the rotation against recall without it, same harness, same corpus, same
    /// seeds — the change measured rather than assumed.
    #[test]
    fn rotation_is_worth_its_cost_in_recall() {
        let mut with = (0.0f32, 0.0f32);
        let mut without = (0.0f32, 0.0f32);
        let seeds = [7u64, 23, 91];
        for seed in seeds {
            let (vectors, queries) = block_corpus(seed);
            let id = ids(N);
            let rotated = VectorBlock::encode(VectorCodec::Binary, DIM, &id, &vectors).unwrap();
            let plain = VectorBlock::encode_with_rotation(
                VectorCodec::Binary,
                DIM,
                &id,
                &vectors,
                Rotation::identity(DIM),
            )
            .unwrap();
            let exact = VectorBlock::encode(VectorCodec::F32, DIM, &id, &vectors).unwrap();
            for q in &queries {
                let truth = top_ids(&exact, q, 10);
                let hit = |b: &VectorBlock, k: usize| {
                    let got = top_ids(b, q, k);
                    truth.iter().filter(|t| got.contains(t)).count() as f32 / 10.0
                };
                with.0 += hit(&rotated, 10);
                with.1 += hit(&rotated, 40);
                without.0 += hit(&plain, 10);
                without.1 += hit(&plain, 40);
            }
        }
        let n = (seeds.len() * QUERIES) as f32;
        let (w10, w40) = (with.0 / n, with.1 / n);
        let (p10, p40) = (without.0 / n, without.1 / n);
        println!(
            "recall@10  rotated {w10:.4}  unrotated {p10:.4}\n\
             recall@10 into 40  rotated {w40:.4}  unrotated {p40:.4}"
        );
        // Measured over 3 seeds x 60 queries: recall@10 0.5344 rotated against 0.5133
        // unrotated (+2.1 points), and into a 40-candidate rerank set 0.9983 against 1.0000
        // (-0.17 points, i.e. one true neighbour out of 1800 moved outside the top 40).
        //
        // Read that honestly: **the rotation is not bought for recall.** It is bought for the
        // bound — without it there is no honest interval to hand the query path, and the
        // oversampling factor has to be guessed. On this corpus it is recall-neutral to
        // within noise, and that is the outcome to want; the gate is only that it is not a
        // regression, because a rotation that cost recall would be paying for the bound
        // twice.
        assert!(
            w10 >= p10 - 0.03,
            "rotation must not cost recall@10: {w10} vs {p10}"
        );
        assert!(
            w40 >= p40 - 0.01,
            "rotation must not cost oversampled recall: {w40} vs {p40}"
        );
    }

    // --- the bounds ---------------------------------------------------------------------

    #[derive(Default)]
    struct BoundStats {
        samples: usize,
        missed: usize,
        width: f64,
        worst_miss: f32,
        /// Summed over queries: the fraction of the block a lower-bound-then-rerank caller
        /// would have to fetch at full precision.
        rerank: f64,
        queries: usize,
    }

    impl BoundStats {
        fn containment(&self) -> f64 {
            1.0 - self.missed as f64 / self.samples.max(1) as f64
        }
        fn mean_width(&self) -> f64 {
            self.width / self.samples.max(1) as f64
        }
        fn mean_rerank(&self) -> f64 {
            self.rerank / self.queries.max(1) as f64
        }
    }

    /// Sweep `codec`'s bounds over several seeded corpora and report every number the query
    /// path needs: does the interval hold, how wide is it, and how much of a block would a
    /// caller reranking on it actually have to fetch.
    fn measure_bounds(codec: VectorCodec, seeds: &[u64], k: usize) -> BoundStats {
        measure_bounds_on(codec, seeds, k, TOPICS)
    }

    fn measure_bounds_on(
        codec: VectorCodec,
        seeds: &[u64],
        k: usize,
        topics: usize,
    ) -> BoundStats {
        let mut st = BoundStats::default();
        for &seed in seeds {
            let (vectors, queries) = corpus(N, DIM, topics, 0.5, seed);
            let id = ids(N);
            let exact = VectorBlock::encode(VectorCodec::F32, DIM, &id, &vectors).unwrap();
            let block = VectorBlock::encode(codec, DIM, &id, &vectors).unwrap();
            for q in &queries {
                let pe = exact.prepare(q).unwrap();
                let pa = block.prepare(q).unwrap();
                let mut bounds = Vec::with_capacity(block.len());
                for i in 0..block.len() {
                    let truth = exact.score(&pe, i);
                    let (lo, hi) = block.score_bounds(&pa, i);
                    let est = block.score(&pa, i);
                    assert!(lo <= est && est <= hi, "the estimate must lie in its own bound");
                    st.samples += 1;
                    st.width += (hi - lo) as f64;
                    if truth < lo || truth > hi {
                        st.missed += 1;
                        st.worst_miss = st.worst_miss.max((lo - truth).max(truth - hi));
                    }
                    bounds.push((lo, hi));
                }
                // Exactly the narrowing turbopuffer describes: rank by the lower bound, take
                // the k-th best of those as a floor, and rerank everything that could still
                // beat it. Nothing outside this set can be in the true top k — provided the
                // bound holds, which is what `missed` above is counting.
                let mut los: Vec<f32> = bounds.iter().map(|(lo, _)| *lo).collect();
                los.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let floor = los[k.min(los.len()) - 1];
                let need = bounds.iter().filter(|(_, hi)| *hi >= floor).count();
                st.rerank += need as f64 / block.len() as f64;
                st.queries += 1;
            }
        }
        st
    }

    /// Why stage two dominates query CPU only on *unstructured* queries, and that a top-`k*refine`
    /// contender cap by scan estimate is a safe worst-case bound on it.
    ///
    /// The bound filter reranks every candidate whose upper bound `hi >= tau` (the k-th best lower
    /// bound). With real, clustered embeddings a query has genuine near-neighbours, `tau` is high,
    /// and the filter prunes to ~k — rerank is cheap. With *random* queries (no near-neighbours,
    /// e.g. a synthetic load generator) every candidate scores alike, `tau` is unselective, and the
    /// filter necessarily reranks nearly the whole scanned set — the fleet trace's ~8k rescorings
    /// per query. So the "rerank is 90% of query CPU" figure is an artifact of unstructured queries,
    /// not a property of real workloads.
    ///
    /// Capping the contender set to the top `k*refine` by estimate is a no-op on the structured case
    /// (the bound already returns < cap) and, on the unstructured case, bounds rerank to O(k) while
    /// recall degrades gracefully — an oversampled rerank, which the `RABITQ_EPSILON` doc calls the
    /// intended model. This pins both regimes.
    #[test]
    fn a_contender_cap_bounds_worst_case_rerank_without_hurting_structured_recall() {
        const N: usize = 4000;
        const K: usize = 100; // the vector arm's over-fetch (DEFAULT_ARM_DEPTH)
        let idv = ids(N);
        let cos = |vs: &[Vec<f32>], q: &[f32], i: usize| -> f32 {
            vs[i].iter().zip(q).map(|(a, b)| a * b).sum()
        };
        // Returns (mean bound-filter contenders, mean bound recall@10, per-refine (contenders, recall@10)).
        let run = |topics: usize, refines: &[usize]| {
            let (vectors, qs) = corpus(N, DIM, topics, 0.4, 99);
            let exact = VectorBlock::encode(VectorCodec::F32, DIM, &idv, &vectors).unwrap();
            let approx = VectorBlock::encode(VectorCodec::Binary, DIM, &idv, &vectors).unwrap();
            let recall = |q: &[f32], set: &[usize], truth: &std::collections::HashSet<MemoryId>| {
                let mut s: Vec<(usize, f32)> = set.iter().map(|&i| (i, cos(&vectors, q, i))).collect();
                s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let got: std::collections::HashSet<MemoryId> =
                    s.iter().take(10).map(|(i, _)| idv[*i]).collect();
                truth.iter().filter(|t| got.contains(t)).count() as f32 / 10.0
            };
            let (mut bct, mut brc) = (0.0f32, 0.0f32);
            let mut cct = vec![0.0f32; refines.len()];
            let mut crc = vec![0.0f32; refines.len()];
            for q in &qs {
                let pe = exact.prepare(q).unwrap();
                let truth: std::collections::HashSet<MemoryId> =
                    exact.top_k(&pe, 10).into_iter().map(|(i, _)| idv[i]).collect();
                let pa = approx.prepare(q).unwrap();
                let mut cands: Vec<(usize, f32, f32, f32)> = (0..N)
                    .map(|i| {
                        let (lo, hi) = approx.score_bounds(&pa, i);
                        (i, approx.score(&pa, i), lo, hi)
                    })
                    .collect();
                let mut los: Vec<f32> = cands.iter().map(|c| c.2).collect();
                los.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let tau = los.get(K - 1).copied().unwrap_or(f32::NEG_INFINITY);
                let bound: Vec<usize> = cands.iter().filter(|c| c.3 >= tau).map(|c| c.0).collect();
                bct += bound.len() as f32;
                brc += recall(q, &bound, &truth);
                cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                for (ri, &r) in refines.iter().enumerate() {
                    let capped: Vec<usize> =
                        cands.iter().take((K * r).min(N)).map(|c| c.0).collect();
                    cct[ri] += capped.len() as f32;
                    crc[ri] += recall(q, &capped, &truth);
                }
            }
            let nq = qs.len() as f32;
            let per: Vec<(f32, f32)> =
                (0..refines.len()).map(|i| (cct[i] / nq, crc[i] / nq)).collect();
            (bct / nq, brc / nq, per)
        };

        let refines = [3usize, 8, 16];
        // Structured (real-embedding-like): bound already prunes to ~k, cap is a no-op.
        let (sb_ct, sb_rc, sper) = run(40, &refines);
        println!("structured : bound contenders {:.0}/{} recall {:.4}", sb_ct, N, sb_rc);
        for (i, &r) in refines.iter().enumerate() {
            println!("  cap x{:<2} contenders {:.0} recall {:.4}", r, sper[i].0, sper[i].1);
        }
        assert!(sb_ct < 3.0 * K as f32, "structured: bound should prune to ~k, got {sb_ct:.0}");
        assert!(sper[0].1 >= sb_rc - 0.001, "structured: cap must not cost recall");

        // Unstructured (random-query worst case): bound reranks ~everything; cap bounds it.
        let (ub_ct, ub_rc, uper) = run(1, &refines);
        println!("unstructured: bound contenders {:.0}/{} recall {:.4}", ub_ct, N, ub_rc);
        for (i, &r) in refines.iter().enumerate() {
            println!("  cap x{:<2} contenders {:.0} recall {:.4}", r, uper[i].0, uper[i].1);
        }
        assert!(ub_ct > 0.9 * N as f32, "unstructured: bound should rerank ~everything");
        // Cap x16 bounds rerank to <= 16k while keeping high recall even in this worst case.
        let i16 = refines.iter().position(|&r| r == 16).unwrap();
        assert!(uper[i16].0 <= 16.0 * K as f32 + 1.0, "cap must bound contenders");
        assert!(uper[i16].1 >= 0.99, "cap x16 recall {:.4} in worst case", uper[i16].1);
    }

    /// The number the query path is being wired to trust. If this interval does not hold, a
    /// caller narrowing on it drops real results silently — the worst failure mode in the
    /// module.
    #[test]
    fn the_binary_bound_contains_the_true_cosine() {
        let st = measure_bounds(VectorCodec::Binary, &[7, 23, 91, 404], 10);
        println!(
            "binary bound: {} samples, containment {:.6}, mean width {:.4}, worst miss {:.5}",
            st.samples,
            st.containment(),
            st.mean_width(),
            st.worst_miss
        );
        // **Probabilistic, not absolute.** Measured 1.000000 over 120k (member, query) pairs
        // at eps = 5, which is what the tail bound predicts (7.5e-6 x 120k < 1 expected
        // miss) — but it is a tail bound over an approximate rotation, so this is evidence
        // rather than proof. The gate is set at a rate a caller can reason about: below
        // 0.999 the narrowing is dropping real top-10 members often enough to see.
        assert!(
            st.containment() >= 0.999,
            "containment {:.6} over {} samples, worst miss {}",
            st.containment(),
            st.samples,
            st.worst_miss
        );
        // And when it does miss, it must miss by a hair rather than by a mile — a bound that
        // fails rarely but catastrophically is worse than one that fails often and slightly.
        assert!(st.worst_miss < 0.05, "worst miss {}", st.worst_miss);
    }

    /// A bound that always returned `[-1, 1]` would be perfectly correct and perfectly
    /// useless. This is the number that says it is not.
    #[test]
    fn the_binary_bound_is_narrow_enough_to_be_worth_having() {
        let st = measure_bounds(VectorCodec::Binary, &[7, 23, 91, 404], 10);
        let width = st.mean_width();
        println!("binary mean interval width {width:.4} (the useless bound is 2.0)");
        // Measured: 0.0639 at eps = 5 — 3% of the range a cosine can occupy, against a
        // within-block score spread of roughly 0.3 on this corpus. It is linear in
        // `RABITQ_EPSILON`: 0.0256 at eps = 2, 0.0766 at eps = 6.
        assert!(width < 0.10, "mean interval width {width}");
    }

    /// Turbopuffer quotes "less than 1% of data vectors in the narrowed search space need to
    /// be reranked". This is the same measurement on our bound. Reported honestly: it is not
    /// 1% — and the reason is not the bound.
    ///
    /// The rerank set is `{i : hi_i >= the k-th best lo}`. Its size is decided by two things,
    /// the width of the interval and how densely the block's scores pack just under the k-th
    /// best. On the corpus this module measures everything else against, the *second*
    /// dominates completely: the interval is narrower than the gap between the query's own
    /// sub-topic and the next one, so the rerank set is the sub-topic, `N/TOPICS = 41.7`
    /// members, and it barely moves when the interval width is tripled. The two controls
    /// below make that visible rather than leaving it as a coincidence:
    ///
    /// * `Int8`, whose interval is 8x narrower and absolute, needs almost the same set;
    /// * on isotropic data, where there is no sub-topic gap to hide behind, both blow up.
    ///
    /// So this number characterizes the corpus more than it characterizes the bound, and a
    /// caller wiring rerank volume to it should size for their own data, not for this.
    #[test]
    fn the_rerank_set_is_the_query_s_neighbourhood_not_the_bound_s_width() {
        let seeds = [7u64, 23, 91, 404];
        let binary = measure_bounds(VectorCodec::Binary, &seeds, 10);
        let int8 = measure_bounds(VectorCodec::Int8, &seeds, 10);
        let pct = |st: &BoundStats| st.mean_rerank() * 100.0;
        println!(
            "top-10 rerank set, {TOPICS}-topic corpus: binary {:.2}% ({:.1} of {N}), \
             int8 {:.2}% ({:.1}) — sub-topic size is {:.1}",
            pct(&binary),
            binary.mean_rerank() * N as f64,
            pct(&int8),
            int8.mean_rerank() * N as f64,
            N as f64 / TOPICS as f64
        );
        // Measured: binary 8.33% (41.7 members), int8 7.64% (38.2), sub-topic size 41.7.
        assert!(binary.mean_rerank() < 0.12, "binary rerank {:.4}", binary.mean_rerank());
        // The claim that the corpus and not the bound is setting the size: a bound eight times
        // narrower buys less than a fifth off the rerank set. If this ever stopped holding,
        // narrowing the interval would start being worth something and `RABITQ_EPSILON` would
        // become a real tuning knob rather than a safety margin to be set generously.
        assert!(
            binary.mean_width() > 5.0 * int8.mean_width(),
            "binary width {:.4} vs int8 {:.4}",
            binary.mean_width(),
            int8.mean_width()
        );
        assert!(
            binary.mean_rerank() < int8.mean_rerank() * 1.5,
            "binary {:.4} vs int8 {:.4}: 8x the interval must not mean 1.5x the rerank",
            binary.mean_rerank(),
            int8.mean_rerank()
        );

        // The pessimistic end, and the one a caller should size against: no sub-topic
        // structure, every member an equidistant draw, nothing for the bound to separate.
        let iso_b = measure_bounds_on(VectorCodec::Binary, &seeds, 10, 1);
        let iso_8 = measure_bounds_on(VectorCodec::Int8, &seeds, 10, 1);
        println!(
            "top-10 rerank set, isotropic: binary {:.2}%, int8 {:.2}%",
            pct(&iso_b),
            pct(&iso_8)
        );
        // Measured: binary 99.95%, int8 19.24%. Blunt version: on data with no structure
        // *inside* a cluster, a 1-bit bound narrows nothing at all — the interval is wider
        // than the entire spread of scores in the block, so every member could be in the top
        // 10 and the caller reranks the block. That is a correct answer and a useless one,
        // and it is the case a caller must not assume away. It is also the case in which the
        // codec itself is already at its floor (recall@10 = 0.477 there), so the bound is not
        // hiding a failure the estimator does not have.
        assert!(iso_b.mean_rerank() > binary.mean_rerank(), "isotropic must be worse");
    }

    /// Int8's interval is not a tail bound — it is `scale/2` per coordinate and Hölder, so it
    /// cannot be escaped at all. Held to a stricter standard for that reason: zero misses.
    #[test]
    fn the_int8_bound_is_absolute_not_probabilistic() {
        let st = measure_bounds(VectorCodec::Int8, &[7, 23, 91, 404], 10);
        println!(
            "int8 bound: containment {:.6}, mean width {:.5}, rerank {:.3}%",
            st.containment(),
            st.mean_width(),
            st.mean_rerank() * 100.0
        );
        assert_eq!(st.missed, 0, "worst miss {}", st.worst_miss);
        // Measured: 0.0073 — nine times tighter than binary's, from a codec 6.6x larger.
        assert!(st.mean_width() < 0.02, "int8 mean width {}", st.mean_width());
    }

    #[test]
    fn f32_bounds_are_the_exact_score_twice_over() {
        let (vectors, queries) = corpus(40, DIM, TOPICS, 0.5, 5);
        let block = VectorBlock::encode(VectorCodec::F32, DIM, &ids(40), &vectors).unwrap();
        let q = block.prepare(&queries[0]).unwrap();
        for i in 0..block.len() {
            let (lo, hi) = block.score_bounds(&q, i);
            assert_eq!(lo, block.score(&q, i), "member {i}");
            assert_eq!(hi, lo, "an exact codec has a zero-width interval");
        }
    }

    #[test]
    fn bounds_are_ordered_finite_and_inside_the_cosine_range_everywhere() {
        // Including all the degenerate corners, where returning a NaN or an inverted interval
        // would make a caller's `hi >= floor` comparison silently drop the whole block.
        let (vectors, queries) = corpus(20, DIM, 1, 0.5, 8);
        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let block = VectorBlock::encode(codec, DIM, &ids(20), &vectors).unwrap();
            for q in [&queries[0], &vec![0.0f32; DIM]] {
                let p = block.prepare(q).unwrap();
                for i in [0usize, 19, 20, usize::MAX] {
                    let (lo, hi) = block.score_bounds(&p, i);
                    assert!(lo.is_finite() && hi.is_finite(), "{codec:?} member {i}");
                    assert!(lo <= hi, "{codec:?} member {i}: [{lo}, {hi}] is inverted");
                    assert!((-1.0..=1.0).contains(&lo) && (-1.0..=1.0).contains(&hi));
                    if i >= 20 {
                        assert_eq!((lo, hi), (0.0, 0.0), "{codec:?}: no such member");
                    }
                }
            }
        }
        // dim 1 has no orthogonal complement for the binary error to concentrate in, so the
        // bound must open all the way rather than divide by zero.
        let one = VectorBlock::encode(VectorCodec::Binary, 1, &ids(2), &[vec![1.0], vec![3.0]])
            .unwrap();
        let p = one.prepare(&[2.0]).unwrap();
        for i in 0..2 {
            let (lo, hi) = one.score_bounds(&p, i);
            assert!(lo <= one.score(&p, i) && one.score(&p, i) <= hi);
        }
    }

    /// The rest of the module measures the estimator on a corpus shaped like real embeddings.
    /// The bound has to hold on data that is *not* shaped like that too, because a bound is a
    /// promise and a corpus is not.
    #[test]
    fn the_binary_bound_holds_on_isotropic_and_on_spiky_data() {
        let id = ids(N);
        // Isotropic: no sub-topic structure, every member an equidistant draw.
        let (iso, iso_q) = corpus(N, DIM, 1, 0.5, 7);
        // Spiky: energy concentrated in a handful of coordinates, the case a coordinate-basis
        // sign quantizer is worst at and the rotation is supposed to fix.
        let mut rng = Rng::seeded(1234);
        let spiky: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                let mut v = vec![0.0f32; DIM];
                for j in 0..8 {
                    v[(i * 3 + j * 41) % DIM] = gauss(&mut rng);
                }
                v[i % DIM] += 2.0;
                unit(v)
            })
            .collect();
        let spiky_q: Vec<Vec<f32>> = (0..QUERIES)
            .map(|i| {
                let mut v = spiky[i * 7 % N].clone();
                for x in v.iter_mut() {
                    *x += 0.15 * gauss(&mut rng);
                }
                unit(v)
            })
            .collect();

        for (name, vectors, queries) in [
            ("isotropic", &iso, &iso_q),
            ("spiky", &spiky, &spiky_q),
        ] {
            let exact = VectorBlock::encode(VectorCodec::F32, DIM, &id, vectors).unwrap();
            let block = VectorBlock::encode(VectorCodec::Binary, DIM, &id, vectors).unwrap();
            let (mut n, mut missed, mut width) = (0usize, 0usize, 0.0f64);
            for q in queries {
                let pe = exact.prepare(q).unwrap();
                let pa = block.prepare(q).unwrap();
                for i in 0..block.len() {
                    let truth = exact.score(&pe, i);
                    let (lo, hi) = block.score_bounds(&pa, i);
                    n += 1;
                    width += (hi - lo) as f64;
                    if truth < lo || truth > hi {
                        missed += 1;
                    }
                }
            }
            let rate = 1.0 - missed as f64 / n as f64;
            println!("{name}: containment {rate:.6}, mean width {:.4}", width / n as f64);
            // Measured: isotropic 1.000000 at width 0.0218, spiky 1.000000 at width 0.3859.
            // The spiky width is the honest cost of the case: `c` stays healthy because the
            // rotation fixes the *basis*, but those vectors sit far from the block mean, so
            // `|r|/|v|` — the fraction of the score that is estimated rather than known —
            // is close to 1 and the interval scales with it. Wide, and still correct, which
            // is the direction a bound is allowed to fail in.
            assert!(rate >= 0.999, "{name} containment {rate:.6} over {n} samples");
        }
    }

    /// Repeated decode -> re-encode across generations, the exact loop a compaction fold runs
    /// (`read_generation` decodes the previous `.vec` block, the fold re-encodes the decoded
    /// values). This pins the *measured* behaviour of that loop for every codec so the
    /// compounding-quantization worry (TODOS §Vector storage) is bounded by a test, not an
    /// argument.
    ///
    /// Measured over 8 generations at dim 384, N=500 (see the asserts for the pinned numbers):
    ///
    /// * **F32 is exactly idempotent** — byte-identical from gen 2 on. Lossless decode, lossless
    ///   re-encode.
    /// * **Int8 and Binary are NOT byte-idempotent.** Int8 is affine over the *block mean-centred
    ///   residual*, so a shifting mean and a rescaled step move the codes by an ulp each fold — it
    ///   is not the symmetric max-abs grid that would reproduce codes exactly. Binary re-encodes
    ///   the shrunk projection its own `decode` returns, which also drifts.
    /// * **The drift is immaterial to retrieval.** Both codecs are candidate generators reranked
    ///   at 4x oversampling; across all 8 generations recall@10-of-the-top-40 against the
    ///   *original* embeddings never drops below 1.000 (Int8) / 0.99 (Binary). Int8's worst
    ///   per-coordinate error grows ~2.5e-5/gen (0.0003 -> 0.0005 over 8), four orders below
    ///   anything the rerank can see. The loss happens essentially once; it does not accumulate
    ///   into a ranking change within any realistic number of folds.
    #[test]
    fn codecs_are_stable_under_repeated_decode_reencode() {
        let (vectors, queries) = block_corpus(11);
        let ids = ids(vectors.len());
        // The true ranking is against the ORIGINAL caller embeddings — not the current (already
        // once-decoded) generation, which would only measure self-consistency.
        let exact = VectorBlock::encode(VectorCodec::F32, DIM, &ids, &vectors).unwrap();
        const GENERATIONS: usize = 8;

        for codec in [VectorCodec::F32, VectorCodec::Int8, VectorCodec::Binary] {
            let mut cur = vectors.clone();
            let mut prev_bytes: Option<Vec<u8>> = None;
            let mut worst_err = 0.0f32;
            let mut min_recall = 1.0f32;
            for g in 1..=GENERATIONS {
                let b = VectorBlock::encode(codec, DIM, &ids, &cur).unwrap();
                let bytes = b.to_bytes();

                if codec == VectorCodec::F32 {
                    if let Some(prev) = &prev_bytes {
                        assert_eq!(
                            *prev, bytes,
                            "F32 must be byte-idempotent under decode->re-encode (gen {g})"
                        );
                    }
                }

                for i in 0..cur.len() {
                    let d = b.decode(i);
                    let e: f32 =
                        d.iter().zip(&vectors[i]).map(|(a, x)| (a - x).abs()).fold(0.0, f32::max);
                    worst_err = worst_err.max(e);
                }
                // Recall of THIS generation's block against the ORIGINAL exact ranking, at 4x
                // oversampling (the set Phase 3 hands the full-precision rerank).
                let mut r = 0.0;
                for q in &queries {
                    let truth = top_ids(&exact, q, 10);
                    let got: Vec<MemoryId> = b
                        .top_k(&b.prepare(q).unwrap(), 40)
                        .into_iter()
                        .map(|(i, _)| b.ids()[i])
                        .collect();
                    r += truth.iter().filter(|id| got.contains(id)).count() as f32 / 10.0;
                }
                min_recall = min_recall.min(r / queries.len() as f32);

                cur = (0..b.len()).map(|i| b.decode(i)).collect();
                prev_bytes = Some(bytes);
            }

            match codec {
                // Pinned floors, each below the measured value, so a regression names itself.
                VectorCodec::F32 => {
                    assert_eq!(worst_err, 0.0, "F32 decode is exact");
                    assert!(min_recall >= 0.999, "F32 recall {min_recall}");
                }
                VectorCodec::Int8 => {
                    // Measured 0.000472 after 8 gens; the loss is one-shot, not compounding.
                    assert!(worst_err < 2e-3, "int8 drift {worst_err} over {GENERATIONS} gens");
                    assert!(min_recall >= 0.99, "int8 recall {min_recall}");
                }
                VectorCodec::Binary => {
                    // 1 bit/dim never resolves a top-10 on its own; the rerank does. What must
                    // hold is that the *oversampled* recall does not decay across generations.
                    assert!(min_recall >= 0.98, "binary x4 recall {min_recall}");
                }
            }
        }
    }
}
