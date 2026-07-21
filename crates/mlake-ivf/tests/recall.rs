//! Gate G-1: IVF recall against brute force.
//!
//! Target (SPEC §10.3): recall@10 ≥ 0.95 at nprobe=8, ≥ 0.99 at nprobe=32.
//!
//! Recall here means agreement with exhaustive search over the same data, so this
//! measures only what partitioning costs — the embedding model and the corpus are held
//! identical between the two sides.

use mlake_core::memory::Timestamps;
use mlake_core::{MemoryId, StoredMemory};
use mlake_ivf::{build_clusters, exact_search, train_centroids, Centroids, Hit};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Synthetic corpus with cluster structure, which is what real embeddings look like —
/// uniform random vectors would make any partitioning look equally good and the gate
/// meaningless.
fn corpus(n: usize, dim: usize, seed: u64) -> Vec<StoredMemory> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_centres = (n as f64).sqrt() as usize;
    let centres: Vec<Vec<f32>> = (0..n_centres)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    (0..n)
        .map(|i| {
            let centre = &centres[i % n_centres];
            let mut v: Vec<f32> = centre
                .iter()
                .map(|c| c + rng.gen_range(-0.35..0.35))
                .collect();
            mlake_core::normalize(&mut v);
            StoredMemory {
                id: MemoryId::from_key(&format!("item-{i}")),
                vector: v,
                text: format!("item {i}"),
                memory_type: 1,
                tags: vec![],
                timestamps: Timestamps::default(),
                proof_count: 0,
                entity_ids: vec![],
                semantic_out: vec![],
                causal_out: vec![],
                metadata: vec![],
                write_seq: 0,
            }
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

/// Search by probing `nprobe` clusters and re-ranking exactly over what was fetched.
fn ivf_search(
    clusters: &[Vec<StoredMemory>],
    centroids: &Centroids,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> Vec<Hit> {
    let probed = centroids.probe(query, nprobe);
    let mut candidates: Vec<StoredMemory> = Vec::new();
    for c in probed {
        candidates.extend_from_slice(&clusters[c]);
    }
    exact_search(&candidates, query, k)
}

/// Fraction of the true top-k that the approximate search also returned.
fn recall_at_k(approx: &[Hit], truth: &[Hit]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let found: std::collections::HashSet<MemoryId> = approx.iter().map(|h| h.id).collect();
    let hits = truth.iter().filter(|t| found.contains(&t.id)).count();
    hits as f64 / truth.len() as f64
}

struct Measurement {
    mean_recall: f64,
    min_recall: f64,
}

fn measure(n: usize, dim: usize, nprobe: usize, k: usize) -> Measurement {
    let items = corpus(n, dim, 42);
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let centroids = train_centroids(&vectors, 42);
    let clusters: Vec<Vec<StoredMemory>> = build_clusters(items.clone(), &centroids)
        .into_iter()
        .map(|c| c.items)
        .collect();

    let mut total = 0.0;
    let mut min: f64 = 1.0;
    let qs = queries(100, dim, 7);
    for q in &qs {
        let truth = exact_search(&items, q, k);
        let approx = ivf_search(&clusters, &centroids, q, k, nprobe);
        let r = recall_at_k(&approx, &truth);
        total += r;
        min = min.min(r);
    }
    Measurement {
        mean_recall: total / qs.len() as f64,
        min_recall: min,
    }
}

#[test]
fn g1_recall_at_10_meets_the_gate_at_nprobe_8() {
    let m = measure(10_000, 128, 8, 10);
    println!("nprobe=8  recall@10 mean={:.4} min={:.4}", m.mean_recall, m.min_recall);
    assert!(
        m.mean_recall >= 0.95,
        "G-1: recall@10 at nprobe=8 was {:.4}, gate is 0.95",
        m.mean_recall
    );
}

#[test]
fn g1_recall_at_10_meets_the_gate_at_nprobe_32() {
    let m = measure(10_000, 128, 32, 10);
    println!("nprobe=32 recall@10 mean={:.4} min={:.4}", m.mean_recall, m.min_recall);
    assert!(
        m.mean_recall >= 0.99,
        "G-1: recall@10 at nprobe=32 was {:.4}, gate is 0.99",
        m.mean_recall
    );
}

#[test]
fn recall_increases_monotonically_with_nprobe() {
    // Sanity on the knob itself: if this did not hold, the nprobe parameter would not be
    // doing what the query planner assumes and tuning it would be guesswork.
    let r1 = measure(5_000, 64, 1, 10).mean_recall;
    let r4 = measure(5_000, 64, 4, 10).mean_recall;
    let r16 = measure(5_000, 64, 16, 10).mean_recall;
    println!("recall@10 by nprobe: 1={r1:.4} 4={r4:.4} 16={r16:.4}");
    assert!(r4 >= r1, "nprobe=4 ({r4:.4}) should not be worse than 1 ({r1:.4})");
    assert!(r16 >= r4, "nprobe=16 ({r16:.4}) should not be worse than 4 ({r4:.4})");
}

#[test]
fn probing_every_cluster_is_exhaustive() {
    // With nprobe covering all clusters, IVF must exactly equal brute force — any gap
    // would mean an item was lost during partitioning rather than merely unprobed.
    let items = corpus(1_000, 32, 3);
    let vectors: Vec<Vec<f32>> = items.iter().map(|i| i.vector.clone()).collect();
    let centroids = train_centroids(&vectors, 3);
    let clusters: Vec<Vec<StoredMemory>> = build_clusters(items.clone(), &centroids)
        .into_iter()
        .map(|c| c.items)
        .collect();

    for q in queries(20, 32, 11) {
        let truth = exact_search(&items, &q, 10);
        let full = ivf_search(&clusters, &centroids, &q, 10, centroids.len());
        assert_eq!(
            recall_at_k(&full, &truth),
            1.0,
            "probing all clusters must be exhaustive"
        );
    }
}
