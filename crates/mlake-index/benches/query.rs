//! Warm-path latency and index-throughput benchmarks (SPEC §10.4).
//!
//! These measure the in-process query engine — the warm path once a generation is
//! materialized — against the spec's warm gates, and the indexer's build throughput. They
//! are wall-clock micro-benchmarks, not the S3 roundtrip rig; the roundtrip budget is
//! asserted separately as an integration test.

use std::time::Instant;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use mlake_core::memory::Timestamps;
use mlake_core::{MemoryId, StoredMemory};
use mlake_fts::Tokenizer;
use mlake_index::{Engine, QueryConfig};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

const DIM: usize = 128;

/// A clustered synthetic corpus, so IVF has real structure to work with.
fn corpus(n: usize, seed: u64) -> Vec<StoredMemory> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres = (n as f64).sqrt() as usize;
    let centroids: Vec<Vec<f32>> = (0..centres)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let words = ["alpha", "beta", "gamma", "delta", "search", "memory", "vector", "graph"];
    (0..n)
        .map(|i| {
            let c = &centroids[i % centres];
            let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-0.3..0.3)).collect();
            mlake_core::normalize(&mut v);
            let text: String = (0..8).map(|_| *words.choose(&mut rng).unwrap()).collect::<Vec<_>>().join(" ");
            StoredMemory {
                id: MemoryId::from_key(&format!("item-{i}")),
                vector: v,
                text,
                memory_type: 1,
                tags: vec![],
                timestamps: Timestamps::default(),
                proof_count: 0,
                entity_ids: vec![],
                semantic_out: vec![],
                causal_out: vec![],
            }
        })
        .collect()
}

fn queries(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
            mlake_core::normalize(&mut v);
            v
        })
        .collect()
}

fn bench_warm_query(c: &mut Criterion) {
    let engine = Engine::build(corpus(100_000, 42), Tokenizer::default());
    let qs = queries(256, 7);
    let config = QueryConfig::default();
    let mut qi = 0;

    let mut group = c.benchmark_group("warm_query_100k");

    group.bench_function("vector_arm", |b| {
        b.iter(|| {
            let q = &qs[qi % qs.len()];
            qi += 1;
            engine.vector_arm(q, 100, config.nprobe)
        })
    });

    group.bench_function("fused_vector_fts", |b| {
        b.iter(|| {
            let q = &qs[qi % qs.len()];
            qi += 1;
            engine.query(Some(q), Some("search memory vector"), 10, config)
        })
    });

    group.finish();
}

fn bench_index_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build");
    group.sample_size(10);
    let items = corpus(20_000, 99);

    group.bench_function("build_20k", |b| {
        b.iter_batched(
            || items.clone(),
            |items| {
                let n = items.len();
                let start = Instant::now();
                let engine = Engine::build(items, Tokenizer::default());
                let elapsed = start.elapsed();
                // Report throughput to stderr for the write-up; criterion tracks the time.
                eprintln!(
                    "index: {} items in {:.3}s = {:.0} items/s",
                    n,
                    elapsed.as_secs_f64(),
                    n as f64 / elapsed.as_secs_f64()
                );
                engine
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_warm_query, bench_index_throughput);
criterion_main!(benches);
