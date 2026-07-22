//! Cache hit ratio under an IVF-probe-shaped workload, for each eviction policy.
//!
//! This is the harness the LRU/FIFO/CLOCK decision was measured with (table in `cache.rs`'s
//! module docs and `TODOS.md` §"Read path"). It asserts nothing — it prints — so it is
//! `#[ignore]`d and does not cost `cargo test` several minutes. It is kept so the eviction
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
//! cache to corpus, not the absolute bytes. Every policy sees the byte-identical trace: the
//! RNG is seeded the same and the draw does not depend on what the cache answered.
//!
//! The hit ratio is decided by the *disk* tier alone — the memory tier is a subset of it
//! (an admission writes both; a disk eviction drops the memory copy), so a lookup that
//! misses memory but hits disk is still a hit. The memory budget therefore moves latency,
//! not this number.

use bytes::Bytes;
use mlake_store::{CacheKey, DiskCache, EvictionPolicy};

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

/// Column order of every table below: LRU, FIFO, CLOCK.
const POLICIES: [EvictionPolicy; 3] = [
    EvictionPolicy::Lru,
    EvictionPolicy::Fifo,
    EvictionPolicy::Clock,
];

/// One run over the whole trace. Returns `(overall hit ratio, hit ratio on the three
/// always-hot objects)` — the second is the diagnostic: those objects are 3 of 19 lookups
/// per query and a policy that cannot hold 3 KiB of them is losing points it should not.
fn run(policy: EvictionPolicy, mem_budget: u64, disk_budget: u64, s: f64) -> (f64, f64) {
    let dir = tempfile::tempdir().unwrap();
    let cache = DiskCache::with_policy(dir.path(), mem_budget, disk_budget, policy).unwrap();
    let cdf = zipf_cdf(N_CLUSTERS, s);
    let mut rng = Rng(0x9E3779B97F4A7C15);

    let (mut hot_hits, mut hot_lookups) = (0u64, 0u64);
    for _ in 0..QUERIES {
        // Every query reads the small always-hot blocks...
        for p in HOT {
            let k = CacheKey::new("", p, "immutable");
            match cache.get(&k) {
                Some(_) => hot_hits += 1,
                None => cache.put(k, Bytes::from(vec![0u8; HOT_BYTES])),
            }
            hot_lookups += 1;
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
        }
    }

    (
        cache.hit_ratio().unwrap(),
        hot_hits as f64 / hot_lookups as f64,
    )
}

fn corpus_bytes() -> u64 {
    (N_CLUSTERS * CLUSTER_BYTES + HOT.len() * HOT_BYTES) as u64
}

/// One row of the table: the same trace under all three policies.
fn row(label: &str, s: f64, pct: u64) {
    let disk = corpus_bytes() * pct / 100;
    let mem = disk / 4;
    let r: Vec<(f64, f64)> = POLICIES.iter().map(|p| run(*p, mem, disk, s)).collect();
    let (lru, fifo, clock) = (r[0].0, r[1].0, r[2].0);
    println!(
        "{label:<18} {pct:>3}%   {lru:.4}  {fifo:.4}  {clock:.4}   {:+.4}     {:+.4}      {:.3} {:.3} {:.3}",
        clock - fifo,
        clock - lru,
        r[0].1,
        r[1].1,
        r[2].1,
    );
}

#[test]
#[ignore = "measurement, not an assertion: run explicitly with --ignored --nocapture"]
fn hit_ratio_under_ivf_probe_skew() {
    let corpus = corpus_bytes();
    println!("\ncorpus = {corpus} bytes ({N_CLUSTERS} clusters x {CLUSTER_BYTES} B + 3 hot blocks)");
    println!("{QUERIES} queries x (3 hot + {NPROBE} skewed cluster reads); mem budget = disk/4\n");
    println!(
        "{:<18} {:>4}   {:<7} {:<7} {:<7}  {:<9} {:<10} hot-object hit ratio (L/F/C)",
        "trace", "size", "LRU", "FIFO", "CLOCK", "vs FIFO", "vs LRU"
    );
    for (label, s) in [
        ("zipf s=1.1", 1.1),
        ("zipf s=1.5", 1.5),
        ("uniform (control)", 0.0),
    ] {
        for pct in [5u64, 10, 25, 50] {
            row(label, s, pct);
        }
        println!();
    }
}
