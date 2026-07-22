//! Soundness of the adaptive-probe stopping rule.
//!
//! Adaptive probing retires a cluster — declines to read it at all — when the best score it
//! could possibly hold, `⟨q̂,c⟩ + R`, falls below the k-th best score already in hand. If that
//! bound is ever *too low*, results are silently dropped and no test downstream would notice:
//! recall would just be a bit worse and everyone would blame the clustering. So the bound is
//! checked here directly, against brute force, rather than inferred from a recall number.

use mlake_ivf::{member_radius, train_centroids, Centroids};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Clustered, unit-length vectors — what real embeddings look like. Uniform noise would make
/// every cluster's radius identical and the bound trivially loose.
fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_centres = (n as f64).sqrt() as usize;
    let centres: Vec<Vec<f32>> = (0..n_centres)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let centre = &centres[i % n_centres];
            let mut v: Vec<f32> = centre.iter().map(|c| c + rng.gen_range(-0.35..0.35)).collect();
            mlake_core::normalize(&mut v);
            v
        })
        .collect()
}

fn queries(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            mlake_core::normalize(&mut v);
            v
        })
        .collect()
}

/// Members grouped by their assigned centroid, exactly as `build_clusters` groups them.
fn assign(centroids: &Centroids, vectors: &[Vec<f32>]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); centroids.len()];
    for (i, v) in vectors.iter().enumerate() {
        out[centroids.assign(v)].push(i);
    }
    out
}

// -------------------------------------------------------------------- the bound is sound

#[test]
fn the_bound_is_never_below_a_clusters_true_best_score() {
    let vectors = corpus(4_000, 64, 42);
    let centroids = train_centroids(&vectors, 42);
    let members = assign(&centroids, &vectors);

    let mut worst_slack = f32::INFINITY;
    for q in queries(50, 64, 7) {
        for c in 0..centroids.len() {
            let bound = centroids.max_similarity(&q, c);
            for &m in &members[c] {
                let truth = mlake_core::cosine(&q, &vectors[m]);
                assert!(
                    bound >= truth - 1e-5,
                    "cluster {c} bound {bound} is below a member's true score {truth} — \
                     the stopping rule would drop it"
                );
                worst_slack = worst_slack.min(bound - truth);
            }
        }
    }
    println!("tightest observed slack (bound - true best member): {worst_slack:.5}");
}

#[test]
fn the_bound_holds_when_members_are_not_unit_length() {
    // Centroids are means, so they are never unit-length; `uniform_dim` exists because callers
    // are not trusted to normalise their vectors either. A bound derived against raw ‖v − c‖
    // rather than the member *direction* breaks exactly here.
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let dim = 16;
    let vectors: Vec<Vec<f32>> = (0..500)
        .map(|_| {
            let scale: f32 = rng.gen_range(0.05..20.0);
            (0..dim).map(|_| rng.gen_range(-1.0..1.0) * scale).collect()
        })
        .collect();
    let centroids = train_centroids(&vectors, 5);
    let members = assign(&centroids, &vectors);

    for q in queries(20, dim, 3) {
        for c in 0..centroids.len() {
            let bound = centroids.max_similarity(&q, c);
            for &m in &members[c] {
                let truth = mlake_core::cosine(&q, &vectors[m]);
                assert!(bound >= truth - 1e-4, "cluster {c}: bound {bound} < true {truth}");
            }
        }
    }
}

#[test]
fn a_zero_length_member_is_covered_by_the_bound() {
    // A text-only memory carries no embedding; its row is zero-padded and it scores exactly
    // 0.0. Its contribution to the radius is ‖0 − c‖ = ‖c‖, and ⟨q̂,c⟩ + ‖c‖ ≥ 0 by
    // Cauchy–Schwarz, so the bound covers it without a special case.
    let mut vectors = corpus(200, 8, 1);
    vectors.push(vec![0.0; 8]);
    let centroids = train_centroids(&vectors, 1);
    let zero_cluster = centroids.assign(&vec![0.0; 8]);
    for q in queries(20, 8, 2) {
        assert!(
            centroids.max_similarity(&q, zero_cluster) >= 0.0,
            "the bound must cover a zero-vector member's score of 0.0"
        );
    }
}

// ------------------------------------------------------- the stopping rule drops nothing

/// The claim the query path relies on, stated literally: with `tau` the k-th best score among
/// candidates already in hand, a cluster whose bound is below `tau` holds no member of the
/// *global* brute-force top-k.
#[test]
fn a_retired_cluster_never_held_a_true_top_k_member() {
    const K: usize = 10;
    let vectors = corpus(4_000, 64, 42);
    let centroids = train_centroids(&vectors, 42);
    let members = assign(&centroids, &vectors);

    for q in queries(50, 64, 7) {
        // Brute force: the truth the stopping rule must not damage.
        let mut all: Vec<(usize, f32)> =
            vectors.iter().enumerate().map(|(i, v)| (i, mlake_core::cosine(&q, v))).collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let top_k: std::collections::HashSet<usize> = all[..K].iter().map(|(i, _)| *i).collect();

        // Wave one: a seed of the nearest clusters, and the k-th best score it found.
        let order = centroids.probe(&q, centroids.len());
        let seed_n = (centroids.len() / 4).max(4).min(order.len());
        let mut seen: Vec<f32> = Vec::new();
        for &c in &order[..seed_n] {
            for &m in &members[c] {
                seen.push(mlake_core::cosine(&q, &vectors[m]));
            }
        }
        seen.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let tau = seen.get(K - 1).copied().unwrap_or(f32::NEG_INFINITY);

        // Wave two: everything the bound cannot rule out. Whatever is left over is retired.
        for &c in &order[seed_n..] {
            if centroids.max_similarity(&q, c) >= tau {
                continue;
            }
            for &m in &members[c] {
                assert!(
                    !top_k.contains(&m),
                    "cluster {c} was retired but held member {m}, a true top-{K} hit"
                );
            }
        }
    }
}

#[test]
fn stopping_is_deterministic() {
    let vectors = corpus(2_000, 32, 5);
    let centroids = train_centroids(&vectors, 5);
    let q = &queries(1, 32, 13)[0];
    let decide = || -> Vec<(usize, bool)> {
        (0..centroids.len())
            .map(|c| (c, centroids.max_similarity(q, c) >= 0.5))
            .collect()
    };
    assert_eq!(decide(), decide(), "the bound is a pure function of resident data");
}

/// The measurement the whole idea lives or dies on: how often does the bound actually retire
/// anything? One outlier member inflates `R` and the cluster never retires — and one such
/// cluster keeps the whole tail alive. Prints rather than asserts a threshold: the number is
/// corpus-dependent, and a gate here would just be another fitted constant.
///
/// Also reports what a *quantile* radius would retire, since that is the obvious mitigation
/// for outliers. It is not implemented: a p95 radius makes stopping approximate (5% of each
/// cluster's members fall outside a bound the query path treats as absolute), so it would need
/// its own recall gate rather than the soundness proof this one has.
#[test]
fn how_often_the_bound_retires_a_cluster() {
    for k in [10usize, 100] {
        for (n, dim) in [(4_000usize, 64usize), (10_000, 128)] {
            let vectors = corpus(n, dim, 42);
            let centroids = train_centroids(&vectors, 42);
            let members = assign(&centroids, &vectors);
            // p95 radius per cluster, for the counterfactual only.
            let p95: Vec<f32> = (0..centroids.len())
                .map(|c| {
                    let mut ds: Vec<f32> = members[c]
                        .iter()
                        .map(|&m| member_radius(&centroids.vectors[c], &vectors[m]))
                        .collect();
                    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    ds.get((ds.len() as f64 * 0.95) as usize).or(ds.last()).copied().unwrap_or(0.0)
                })
                .collect();
            let mut quantile = centroids.clone();
            quantile.radii = p95;

            let mut considered = 0usize;
            let (mut retired, mut retired_q) = (0usize, 0usize);
            let mut queries_with_any = 0usize;
            let qs = queries(50, dim, 7);
            for q in &qs {
                let order = centroids.probe(q, centroids.len().div_ceil(2));
                let seed_n = (order.len() / 4).max(4).min(order.len());
                let mut seen: Vec<f32> = Vec::new();
                for &c in &order[..seed_n] {
                    for &m in &members[c] {
                        seen.push(mlake_core::cosine(q, &vectors[m]));
                    }
                }
                seen.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let tau = seen.get(k - 1).copied().unwrap_or(f32::NEG_INFINITY);
                let mut r = 0;
                for &c in &order[seed_n..] {
                    considered += 1;
                    if centroids.max_similarity(q, c) < tau {
                        r += 1;
                    }
                    if quantile.max_similarity(q, c) < tau {
                        retired_q += 1;
                    }
                }
                retired += r;
                if r > 0 {
                    queries_with_any += 1;
                }
            }
            println!(
                "n={n} dim={dim} k={k}: max-radius retired {retired}/{considered} of the tail \
                 ({:.1}%), {queries_with_any}/{} queries retired ≥1 | p95-radius would retire \
                 {retired_q}/{considered} ({:.1}%)",
                100.0 * retired as f64 / considered.max(1) as f64,
                qs.len(),
                100.0 * retired_q as f64 / considered.max(1) as f64,
            );
        }
    }
}

// ------------------------------------------------------------- unknown radius bounds nothing

#[test]
fn an_absent_or_stale_radius_bounds_nothing() {
    let base = Centroids {
        vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        sizes: vec![1, 1],
        dim: 2,
        radii: vec![0.1, 0.1],
    };
    assert!(base.max_similarity(&[1.0, 0.0], 0).is_finite());

    // Absent: a table written before radii existed.
    let mut absent = base.clone();
    absent.radii = Vec::new();
    assert_eq!(absent.max_similarity(&[1.0, 0.0], 0), f32::INFINITY);
    assert_eq!(absent.radius(0), None);

    // Stale shape: a centroid was appended and the radius never recomputed. The whole vector
    // is distrusted, not just the missing entry — a partial `radii` means the fold that wrote
    // it did not see this membership.
    let mut short = base.clone();
    short.radii.pop();
    assert_eq!(short.max_similarity(&[1.0, 0.0], 0), f32::INFINITY);

    // Non-finite / negative entries are unknown, not bounds.
    for bad in [f32::NAN, f32::INFINITY, -1.0] {
        let mut c = base.clone();
        c.radii[0] = bad;
        assert_eq!(c.max_similarity(&[1.0, 0.0], 0), f32::INFINITY, "radius {bad}");
    }

    // A dimension the table does not speak, or a zero-length query, bounds nothing either.
    assert_eq!(base.max_similarity(&[1.0, 0.0, 0.0], 0), f32::INFINITY);
    assert_eq!(base.max_similarity(&[0.0, 0.0], 0), f32::INFINITY);
    assert_eq!(base.max_similarity(&[1.0, 0.0], 99), f32::INFINITY);
}

#[test]
fn a_split_centroid_starts_with_an_unknown_radius() {
    // `local_split` appends a centroid mid-fold. Until the fold recomputes radii that cluster's
    // membership is unknown, and 0.0 would claim it contains nothing but the centroid itself.
    let mut c = Centroids {
        vectors: vec![vec![1.0, 0.0]],
        sizes: vec![3],
        dim: 2,
        radii: vec![0.2],
    };
    let i = c.push(vec![0.0, 1.0]);
    assert_eq!(c.radius(i), None, "a fresh split must not claim a zero radius");
    assert_eq!(c.max_similarity(&[0.0, 1.0], i), f32::INFINITY);
}

#[test]
fn recompute_radii_covers_every_member() {
    let vectors = corpus(300, 16, 4);
    let mut centroids = train_centroids(&vectors, 4);
    let members = assign(&centroids, &vectors);
    // Wipe and rebuild through the fold's entry point.
    centroids.radii = Vec::new();
    let owned: Vec<Vec<&[f32]>> = members
        .iter()
        .map(|ms| ms.iter().map(|&m| vectors[m].as_slice()).collect())
        .collect();
    centroids.recompute_radii(|i| owned[i].iter().copied());

    for (c, ms) in members.iter().enumerate() {
        let r = centroids.radius(c).expect("radius must be known after recompute");
        for &m in ms {
            assert!(
                member_radius(&centroids.vectors[c], &vectors[m]) <= r + 1e-6,
                "member {m} lies outside cluster {c}'s radius"
            );
        }
    }
}
