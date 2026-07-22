//! In-memory microbenchmark for semantic-link derivation — the dominant cost of a large fold.
//!
//! Builds synthetic *clustered* vectors (so links actually form), preps the IVF once, then times
//! derivation across modes/nprobe with NO S3 and NO WAL commit — seconds instead of a ten-minute
//! write+build. Reports derive wall time and, for each approximate config, its LINK RECALL against
//! the exact f32 reference (fraction of exact top-5 targets it reproduces), so a speedup is never
//! bought with silent quality loss.
//!
//!   cargo run --release -p mlake-index --example derive_bench -- <n> <dim> <avg_cluster> <eps>
//!
//! Defaults: n=1_000_000, dim=384, avg_cluster=200, eps=0.5.

use mlake_core::memory::StoredMemory;
use mlake_core::MemoryId;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Per-item set of link target ids, for recall comparison.
fn link_targets(items: &[StoredMemory]) -> Vec<Vec<[u8; 16]>> {
    items.iter().map(|it| it.semantic_out.iter().map(|e| e.target.0).collect()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = a.first().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let dim: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(384);
    let avg_cluster: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let eps: f32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    // Prep mode: "fast" = sampled centroids (quick, but boundaries cut through clusters so low
    // nprobe under-recalls); "real" = full k-means (slow, but the fair setting for nprobe recall).
    let fast_prep = a.get(4).map(|s| s != "real").unwrap_or(true);

    let num_clusters = (n / avg_cluster).max(1);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let centers: Vec<Vec<f32>> =
        (0..num_clusters).map(|_| (0..dim).map(|_| rng.unit()).collect()).collect();

    let gen0 = std::time::Instant::now();
    let base: Vec<StoredMemory> = (0..n)
        .map(|i| {
            let c = &centers[i % num_clusters];
            let vector: Vec<f32> = (0..dim).map(|d| c[d] + eps * rng.unit()).collect();
            StoredMemory {
                id: MemoryId::from_key(&format!("k{i}")),
                vector,
                text: String::new(),
                index_text: String::new(),
                memory_type: 1,
                tags: Vec::new(),
                timestamps: Default::default(),
                proof_count: 0,
                entity_ids: Vec::new(),
                semantic_out: Vec::new(),
                causal_out: Vec::new(),
                metadata: Vec::new(),
                write_seq: 0,
            }
        })
        .collect();
    println!("n={n} dim={dim} clusters={num_clusters} eps={eps} gen={:.1}s", gen0.elapsed().as_secs_f64());

    // Prep the IVF once (fast: sampled centroids, no k-means iterations) and reuse across modes.
    let prep0 = std::time::Instant::now();
    let (centroids, assignments) = mlake_index::indexer::bench_prepare(&base, 42, fast_prep);
    println!(
        "prep={:.1}s ({} centroids, {})",
        prep0.elapsed().as_secs_f64(),
        centroids.vectors.len(),
        if fast_prep { "fast/sampled" } else { "real/kmeans" }
    );

    // Exact f32 reference at high nprobe = ground truth for recall.
    let mut exact_items = base.clone();
    let (t_exact, links_exact) =
        mlake_index::indexer::bench_derive_links(&mut exact_items, &centroids, &assignments, 16, true);
    let ref_targets = link_targets(&exact_items);
    let ref_total: usize = ref_targets.iter().map(|t| t.len()).sum();
    println!(
        "EXACT nprobe=16   derive={:>7.2}s links={links_exact} avg={:.2}",
        t_exact.as_secs_f64(),
        links_exact as f64 / n as f64
    );

    // Approximate configs: int8 two-stage across a few nprobe values.
    for &np in &[16usize, 8, 4, 2, 1] {
        let mut it = base.clone();
        let (t, links) =
            mlake_index::indexer::bench_derive_links(&mut it, &centroids, &assignments, np, false);
        // Recall = fraction of each exact item's targets reproduced, averaged over items with ≥1 link.
        let got = link_targets(&it);
        let mut hit = 0usize;
        for (r, g) in ref_targets.iter().zip(&got) {
            for t in r {
                if g.contains(t) {
                    hit += 1;
                }
            }
        }
        let recall = if ref_total == 0 { 1.0 } else { hit as f64 / ref_total as f64 };
        println!(
            "int8  nprobe={np:<2}      derive={:>7.2}s links={links} avg={:.2} recall={:.4} speedup={:.1}x",
            t.as_secs_f64(),
            links as f64 / n as f64,
            recall,
            t_exact.as_secs_f64() / t.as_secs_f64().max(1e-9),
        );
    }
}
