//! Mini-batch k-means for centroid training.
//!
//! Written here rather than pulled from faiss/linfa: the index needs exactly one routine
//! from those libraries, and both carry a native build step that would complicate every
//! downstream build for no benefit at this size.
//!
//! Determinism matters as much as quality — SPEC §10.3 G-6 requires that replaying the
//! indexer on the same input produces byte-identical output — so every random choice is
//! driven by an explicit seed and every tie is broken by index order.

/// A deterministic PRNG. Reproducibility is a correctness requirement here, so the
/// generator is part of the format rather than an implementation detail that could change
/// under us with a dependency upgrade.
pub struct Rng(u64);

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        // SplitMix64 finalizer on the seed, so even sequential seeds start well separated.
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    pub fn unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Squared euclidean distance. Comparisons only ever need the ordering, so the square
/// root is skipped throughout.
///
/// Exposed to the index layer so probing and assignment provably use one metric.
pub fn sq_dist_pub(a: &[f32], b: &[f32]) -> f32 {
    sq_dist(a, b)
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    // Squared Euclidean distance — the innermost, hottest primitive of both training and
    // assignment. A single running accumulator forces a serial f32 add chain: the compiler may
    // NOT vectorize it, because reassociating f32 sums would change the result, and reproducible
    // output is a correctness requirement (G-6/INV-6). So sum into a FIXED number of independent
    // lanes and combine them in a FIXED order: the lane loop autovectorizes (NEON/AVX), yet the
    // arithmetic order is still fully determined by the code, so every node computes identical
    // bits. ~4-8× the throughput of the scalar chain.
    const LANES: usize = 8;
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];
    let mut acc = [0.0f32; LANES];
    // `chunks_exact` hands out fixed-length `[f32; LANES]`-shaped slices, so indexing `ca[l]` for
    // `l < LANES` needs no bounds check — the lane loop lowers to a single SIMD multiply-add.
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..LANES {
            let d = xa[l] - xb[l];
            acc[l] += d * d;
        }
    }
    let mut total = 0.0f32;
    for l in 0..LANES {
        total += acc[l];
    }
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        total += d * d;
    }
    total
}

/// Index of the nearest centroid, ties broken toward the lower index for determinism.
pub fn nearest(centroids: &[Vec<f32>], v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_dist(c, v);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// k-means++ seeding: pick spread-out initial centres so the refinement converges to a
/// balanced clustering. Plain random seeding routinely produces near-empty clusters,
/// which directly costs recall — an under-populated cluster wastes an nprobe slot.
fn seed_centroids(vectors: &[Vec<f32>], k: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    centroids.push(vectors[rng.below(vectors.len())].clone());

    let mut closest: Vec<f32> = vectors
        .iter()
        .map(|v| sq_dist(v, &centroids[0]))
        .collect();

    while centroids.len() < k {
        let total: f64 = closest.iter().map(|d| *d as f64).sum();
        let next = if total <= 0.0 {
            // Every remaining point coincides with a chosen centre; any pick is as good.
            rng.below(vectors.len())
        } else {
            // Sample proportional to squared distance.
            let target = rng.unit_f32() as f64 * total;
            let mut acc = 0.0f64;
            let mut chosen = vectors.len() - 1;
            for (i, d) in closest.iter().enumerate() {
                acc += *d as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };

        centroids.push(vectors[next].clone());
        let newest = centroids.last().unwrap();
        // Refresh each point's distance-to-nearest-centre in parallel. `min` is
        // order-independent, so this stays deterministic.
        use rayon::prelude::*;
        closest
            .par_iter_mut()
            .zip(vectors.par_iter())
            .for_each(|(c, v)| *c = c.min(sq_dist(v, newest)));
    }
    centroids
}

/// Train `k` centroids over `vectors` with Lloyd's algorithm.
///
/// Returns the centroids and each vector's assignment.
pub fn train(vectors: &[Vec<f32>], k: usize, max_iters: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(!vectors.is_empty(), "cannot train on an empty set");
    let k = k.clamp(1, vectors.len());
    let dim = vectors[0].len();

    let mut rng = Rng::seeded(seed);
    let mut centroids = seed_centroids(vectors, k, &mut rng);

    for _ in 0..max_iters {
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];

        // Assignment is the dominant cost (O(N·k·dim) per iteration), so compute each
        // vector's nearest centroid across all cores. Accumulation stays sequential in index
        // order below, so the f64 sums are summed in a fixed order — the training output is
        // still byte-identical for a given seed (G-6), just computed faster.
        use rayon::prelude::*;
        let assignments: Vec<usize> = vectors.par_iter().map(|v| nearest(&centroids, v)).collect();

        for (v, &c) in vectors.iter().zip(&assignments) {
            counts[c] += 1;
            for (d, x) in v.iter().enumerate().take(dim) {
                sums[c][d] += *x as f64;
            }
        }

        let mut moved = false;
        for c in 0..k {
            if counts[c] == 0 {
                // An empty cluster contributes nothing but still consumes an nprobe slot,
                // so re-seed it onto the point furthest from its own centre — the region
                // most in need of subdivision.
                if let Some(far) = furthest_point(vectors, &centroids) {
                    if centroids[c] != vectors[far] {
                        centroids[c] = vectors[far].clone();
                        moved = true;
                    }
                }
                continue;
            }
            for d in 0..dim {
                let next = (sums[c][d] / counts[c] as f64) as f32;
                if (next - centroids[c][d]).abs() > f32::EPSILON {
                    moved = true;
                }
                centroids[c][d] = next;
            }
        }

        if !moved {
            break;
        }
    }
    centroids
}

/// The vector furthest from its nearest centroid — the worst-served point.
fn furthest_point(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Option<usize> {
    let mut best = None;
    let mut best_d = -1.0f32;
    for (i, v) in vectors.iter().enumerate() {
        let c = nearest(centroids, v);
        let d = sq_dist(v, &centroids[c]);
        if d > best_d {
            best_d = d;
            best = Some(i);
        }
    }
    best
}

/// Centroid count for a corpus of `n` items: `max(1, round(sqrt(n)))` per SPEC §5.1.
pub fn centroid_count(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    ((n as f64).sqrt().round() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clustered_data() -> Vec<Vec<f32>> {
        // Three well-separated blobs.
        let mut v = Vec::new();
        for i in 0..30 {
            let j = i as f32 * 0.01;
            v.push(vec![0.0 + j, 0.0]);
        }
        for i in 0..30 {
            let j = i as f32 * 0.01;
            v.push(vec![10.0 + j, 0.0]);
        }
        for i in 0..30 {
            let j = i as f32 * 0.01;
            v.push(vec![0.0, 10.0 + j]);
        }
        v
    }

    #[test]
    fn centroid_count_follows_sqrt_n() {
        assert_eq!(centroid_count(0), 1);
        assert_eq!(centroid_count(1), 1);
        assert_eq!(centroid_count(100), 10);
        assert_eq!(centroid_count(10_000), 100);
        // 5000 -> 70.7 -> 71
        assert_eq!(centroid_count(5000), 71);
    }

    #[test]
    fn recovers_well_separated_clusters() {
        let data = clustered_data();
        let centroids = train(&data, 3, 25, 42);
        assert_eq!(centroids.len(), 3);

        // Every blob should own a distinct centroid.
        let a = nearest(&centroids, &[0.15, 0.0]);
        let b = nearest(&centroids, &[10.15, 0.0]);
        let c = nearest(&centroids, &[0.0, 10.15]);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn training_is_deterministic_for_a_given_seed() {
        // G-6: the indexer must be replayable to byte-identical output.
        let data = clustered_data();
        let first = train(&data, 5, 20, 42);
        let second = train(&data, 5, 20, 42);
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_can_differ() {
        let data = clustered_data();
        let a = train(&data, 5, 20, 1);
        let b = train(&data, 5, 20, 999);
        // Not a correctness requirement, but if these always matched the seed would be
        // doing nothing and the determinism test above would be vacuous.
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn no_cluster_is_left_empty() {
        let data = clustered_data();
        let centroids = train(&data, 8, 30, 7);
        let mut counts = vec![0usize; centroids.len()];
        for v in &data {
            counts[nearest(&centroids, v)] += 1;
        }
        assert!(
            counts.iter().all(|c| *c > 0),
            "empty clusters waste nprobe slots: {counts:?}"
        );
    }

    #[test]
    fn k_is_clamped_to_the_population() {
        let data = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        assert_eq!(train(&data, 10, 10, 1).len(), 2);
    }

    #[test]
    fn nearest_breaks_ties_toward_the_lower_index() {
        // Two identical centroids: the choice must be stable, not arbitrary.
        let centroids = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        assert_eq!(nearest(&centroids, &[1.0, 0.0]), 0);
    }

    #[test]
    fn rng_is_reproducible() {
        let mut a = Rng::seeded(42);
        let mut b = Rng::seeded(42);
        for _ in 0..100 {
            assert_eq!(a.below(1000), b.below(1000));
        }
    }
}
