//! Synthetic memory generation for the performance suite.
//!
//! Produces reproducible (seeded) memories that exercise every arm: clustered vectors
//! (so IVF has real structure), text (FTS), Zipfian tags over a large vocabulary (high
//! cardinality + realistic selectivity), Zipfian entity ids and a fraction of causal edges
//! (the graph arm — semantic kNN links are derived by the indexer itself), spread across
//! several independent memory types (the real model).
//!
//! Generation is batched by id range so a 1M+ run never holds the whole corpus in RAM.

use mlake_core::memory::{CausalEdge, LinkType, Timestamps, Weight};
use mlake_core::{EntityId, Memory, MemoryId};

/// A 16-byte entity id from a small integer (synthetic corpus only).
fn entity_id(n: u64) -> EntityId {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&n.to_le_bytes());
    EntityId::from_bytes(b)
}
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Knobs for a synthetic corpus.
#[derive(Clone, Copy)]
pub struct GenConfig {
    pub scale: usize,
    pub memory_types: u8,
    pub dim: usize,
    pub tag_vocab: usize,
    pub tags_per_memory: usize,
    pub untagged_frac: f32,
    pub entity_vocab: usize,
    pub entities_per_memory: usize,
    pub causal_frac: f32,
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            scale: 10_000,
            memory_types: 3,
            dim: 384,
            tag_vocab: 2_000,
            tags_per_memory: 3,
            untagged_frac: 0.2,
            entity_vocab: 5_000,
            entities_per_memory: 2,
            causal_frac: 0.05,
            seed: 42,
        }
    }
}

/// Epoch (arbitrary, ~2023) the synthetic effective-time series starts at. Memory `i` occurs
/// ~`i` seconds after this, so the corpus spans `scale` seconds of history.
pub const TIME_EPOCH: i64 = 1_700_000_000;

const WORDS: &[&str] = &[
    "memory", "recall", "vector", "graph", "lake", "index", "cluster", "query", "tag",
    "entity", "semantic", "episodic", "signal", "search", "bank", "fold", "shard", "probe",
];

/// A generator that yields batches of memories over a fixed id space.
pub struct Generator {
    cfg: GenConfig,
    centers: Vec<Vec<f32>>,
    /// Cumulative Zipf weights for tag ranks, for O(log n) sampling.
    zipf_tag: Vec<f64>,
    zipf_entity: Vec<f64>,
}

impl Generator {
    pub fn new(cfg: GenConfig) -> Self {
        let n_centers = (cfg.scale as f64).sqrt().ceil() as usize;
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0xC0FFEE);
        let centers: Vec<Vec<f32>> = (0..n_centers.max(1))
            .map(|_| {
                let mut v: Vec<f32> = (0..cfg.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                mlake_core::normalize(&mut v);
                v
            })
            .collect();
        Self {
            zipf_tag: zipf_cumulative(cfg.tag_vocab),
            zipf_entity: zipf_cumulative(cfg.entity_vocab),
            cfg,
            centers,
        }
    }

    /// The external key for memory `i` — deterministic, so causal edges can reference it.
    fn key(i: usize) -> String {
        format!("m{i}")
    }

    /// Generate memories for the id range `[start, end)`.
    pub fn batch(&self, start: usize, end: usize) -> Vec<Memory> {
        let mut out = Vec::with_capacity(end - start);
        for i in start..end {
            // Per-memory deterministic RNG so a batch is reproducible in isolation.
            let mut rng = ChaCha8Rng::seed_from_u64(self.cfg.seed.wrapping_add(i as u64));

            let center = &self.centers[i % self.centers.len()];
            let mut vector: Vec<f32> = center
                .iter()
                .map(|c| c + rng.gen_range(-0.35..0.35))
                .collect();
            mlake_core::normalize(&mut vector);

            let text = (0..6)
                .map(|_| *WORDS.choose(&mut rng).unwrap())
                .collect::<Vec<_>>()
                .join(" ");

            let tags = if rng.gen::<f32>() < self.cfg.untagged_frac {
                Vec::new()
            } else {
                let mut ts: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                for _ in 0..self.cfg.tags_per_memory {
                    let rank = zipf_sample(&self.zipf_tag, &mut rng);
                    ts.insert(format!("tag-{rank}"));
                }
                ts.into_iter().collect()
            };

            let mut entity_ids: Vec<EntityId> = (0..self.cfg.entities_per_memory)
                .map(|_| entity_id(zipf_sample(&self.zipf_entity, &mut rng) as u64))
                .collect();
            entity_ids.sort_unstable();
            entity_ids.dedup();

            let causal_out = if i > 0 && rng.gen::<f32>() < self.cfg.causal_frac {
                let target = rng.gen_range(0..i);
                vec![CausalEdge {
                    target: MemoryId::from_key(&Self::key(target)),
                    link_type: LinkType::Causes,
                    weight: Weight::from_f32(rng.gen_range(0.5..1.0)),
                }]
            } else {
                Vec::new()
            };

            // Effective time spread across the corpus: memory `i` occurs `i` seconds after a
            // fixed epoch (plus a little jitter), so a window query [from, to] selects a
            // contiguous slice and the temporal arm has real entry points to rank and spread.
            let occurred_start = TIME_EPOCH + i as i64 + rng.gen_range(0..2);

            out.push(Memory {
                id: MemoryId::from_key(&Self::key(i)),
                vector,
                text,
                index_text: String::new(),
                memory_type: (i % self.cfg.memory_types.max(1) as usize) as u8 + 1,
                tags,
                timestamps: Timestamps { occurred_start: Some(occurred_start), ..Timestamps::default() },
                proof_count: 0,
                entity_ids,
                causal_out,
                semantic_out: Vec::new(),
                // A little opaque metadata, so the read path exercises returning it.
                metadata: vec![
                    ("doc".to_string(), format!("d{}", i % 1000)),
                    ("src".to_string(), "perf".to_string()),
                ],
            });
        }
        out
    }

    /// A query vector near center `c` (for read benches that want hits).
    pub fn query_vector(&self, c: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = &self.centers[c % self.centers.len()];
        let mut v: Vec<f32> = center.iter().map(|x| x + rng.gen_range(-0.2..0.2)).collect();
        mlake_core::normalize(&mut v);
        v
    }

    /// A time window `[from, to]` centered on memory `i`'s time, `span` seconds wide — so the
    /// temporal arm's entry-point scan returns roughly `span` in-window memories.
    pub fn time_window(&self, i: usize, span: i64) -> (i64, i64) {
        let center = TIME_EPOCH + (i % self.cfg.scale.max(1)) as i64;
        (center - span / 2, center + span / 2)
    }

    pub fn center_count(&self) -> usize {
        self.centers.len()
    }
}

/// Cumulative weights for a Zipf(1.0) distribution over `n` ranks (rank 1 = hottest).
fn zipf_cumulative(n: usize) -> Vec<f64> {
    let mut cum = Vec::with_capacity(n.max(1));
    let mut acc = 0.0;
    for r in 1..=n.max(1) {
        acc += 1.0 / r as f64;
        cum.push(acc);
    }
    cum
}

/// Sample a 1-based rank from a Zipf cumulative table.
fn zipf_sample(cum: &[f64], rng: &mut impl Rng) -> usize {
    let total = *cum.last().unwrap_or(&1.0);
    let target = rng.gen::<f64>() * total;
    match cum.binary_search_by(|c| c.partial_cmp(&target).unwrap()) {
        Ok(i) => i + 1,
        Err(i) => i + 1,
    }
}
