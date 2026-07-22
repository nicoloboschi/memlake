//! Cache hit ratio under an IVF-probe-shaped workload.
//!
//! This is the harness the FIFO-vs-LRU decision was measured with (numbers in `cache.rs`'s
//! module docs and `TODOS.md` §"Read path"). It asserts nothing — it prints — so it is
//! `#[ignore]`d and does not cost `cargo test` 80 seconds. It is kept so the eviction
//! policy can be revisited against a number rather than an argument:
//!
//! ```text
//! cargo test -p mlake-store --release --test cache_skew -- --ignored --nocapture
//! ```
//!
//! The trace models one IVF probe: every query reads three small always-hot objects
//! (centroids, `pk.idx`, `radj.idx`) and then `NPROBE` distinct cluster files drawn with
//! Zipf skew, because a probe re-reads popular clusters far more often than cold ones.
//! Object and cache sizes are scaled down together, so what is realistic is the *ratio* of
//! cache to corpus, not the absolute bytes.

use bytes::Bytes;
use mlake_store::{CacheKey, DiskCache};

/// xorshift64* — deterministic, no dev-dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Zipf CDF over `n` items with exponent `s`.
fn zipf_cdf(n: usize, s: f64) -> Vec<f64> {
    let w: Vec<f64> = (1..=n).map(|i| 1.0 / (i as f64).powf(s)).collect();
    let total: f64 = w.iter().sum();
    let mut acc = 0.0;
    w.iter()
        .map(|x| {
            acc += x / total;
            acc
        })
        .collect()
}

fn sample(cdf: &[f64], rng: &mut Rng) -> usize {
    let u = rng.unit();
    cdf.partition_point(|&c| c < u).min(cdf.len() - 1)
}

const N_CLUSTERS: usize = 256;
const CLUSTER_BYTES: usize = 16 * 1024;
const HOT_BYTES: usize = 1024;
const NPROBE: usize = 16;
const QUERIES: usize = 1500;

/// Objects every query touches: centroids, pk.idx footer, radj footer.
const HOT: [&str; 3] = ["ns/gen-1/centroids.bin", "ns/gen-1/pk.idx", "ns/gen-1/radj.idx"];

fn run(label: &str, mem_budget: u64, disk_budget: u64, s: f64) {
    let dir = tempfile::tempdir().unwrap();
    let cache = DiskCache::with_budgets(dir.path(), mem_budget, disk_budget).unwrap();
    let cdf = zipf_cdf(N_CLUSTERS, s);
    let mut rng = Rng(0x9E3779B97F4A7C15);

    let mut lookups = 0u64;
    for _ in 0..QUERIES {
        // Every query reads the small always-hot blocks...
        for p in HOT {
            let k = CacheKey::new("", p, "immutable");
            if cache.get(&k).is_none() {
                cache.put(k, Bytes::from(vec![0u8; HOT_BYTES]));
            }
            lookups += 1;
        }
        // ...then probes NPROBE distinct clusters, chosen with skew.
        let mut probed = Vec::with_capacity(NPROBE);
        while probed.len() < NPROBE {
            let c = sample(&cdf, &mut rng);
            if !probed.contains(&c) {
                probed.push(c);
            }
        }
        for c in probed {
            let k = CacheKey::new("", &format!("ns/gen-1/cluster-{c}.bin"), "immutable");
            if cache.get(&k).is_none() {
                cache.put(k, Bytes::from(vec![0u8; CLUSTER_BYTES]));
            }
            lookups += 1;
        }
    }

    let corpus = (N_CLUSTERS * CLUSTER_BYTES + HOT.len() * HOT_BYTES) as f64;
    println!(
        "{label:<28} s={s:<4} disk_budget={:>7} ({:>4.1}% of corpus)  hit_ratio={:.4}  hits={} misses={} lookups={}",
        disk_budget,
        100.0 * disk_budget as f64 / corpus,
        cache.hit_ratio().unwrap(),
        cache.hits(),
        cache.misses(),
        lookups,
    );
}

#[test]
#[ignore = "measurement, not an assertion: run explicitly with --ignored --nocapture"]
fn hit_ratio_under_ivf_probe_skew() {
    let corpus = (N_CLUSTERS * CLUSTER_BYTES + HOT.len() * HOT_BYTES) as u64;
    println!("\ncorpus = {corpus} bytes ({N_CLUSTERS} clusters x {CLUSTER_BYTES} B + 3 hot blocks)");
    for pct in [5u64, 10, 25, 50] {
        let disk = corpus * pct / 100;
        let mem = disk / 4;
        run("ivf-probe skew", mem, disk, 1.1);
    }
    println!();
    for pct in [10u64, 25] {
        let disk = corpus * pct / 100;
        run("heavier skew", disk / 4, disk, 1.5);
    }
    println!();
    for pct in [10u64, 25] {
        let disk = corpus * pct / 100;
        run("uniform (control)", disk / 4, disk, 0.0);
    }
    println!();
}
