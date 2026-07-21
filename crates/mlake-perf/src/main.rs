//! memlake performance suite.
//!
//! Drives the *real* write → index → query path over storage (MinIO by default) at
//! configurable scale, and reports write throughput + build time, read latency +
//! roundtrips, and S3 cost — the metrics that make the "lowball Postgres" pitch concrete.
//!
//! Usage:
//!   mlake-perf write  --scale N [--types M] [--seed S]
//!   mlake-perf read   --scale N [--queries Q]          (assumes the bank was written first)
//!   mlake-perf suite  [--scales 10000,100000,1000000]  (write+read each, then a report)
//!
//! Storage: MinIO at $MEMLAKE_S3_ENDPOINT (default http://localhost:9000), bucket `memlake`.

mod cost;
mod datagen;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use datagen::{GenConfig, Generator};
use mlake_core::{Op, TagFilter, TagsMatch};
use mlake_fts::Tokenizer;
use mlake_index::{index, Consistency, IndexOptions, QueryConfig, QueryNode};
use mlake_store::{DiskCache, QueryMetrics, Store, StoreMetrics};
use mlake_wal::{Namespace, Writer};

const COMMIT_BATCH: usize = 5_000;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: mlake-perf <write|read|suite> [options]");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "write" => {
            let cfg = parse_gen(&args);
            let store = new_store(store_metrics())?;
            let r = write_bench(&store, &bank_name(cfg.scale), cfg).await?;
            println!("\n{}", r.render());
        }
        "read" => {
            let cfg = parse_gen(&args);
            let queries = flag_usize(&args, "--queries", 200);
            let store = new_store(store_metrics())?;
            let r = read_bench(&store, &bank_name(cfg.scale), cfg, queries).await?;
            println!("\n{}", r.render());
        }
        "suite" => {
            let scales = flag_str(&args, "--scales", "10000,100000,1000000");
            let mut reports = Vec::new();
            for scale in scales.split(',').filter_map(|s| s.trim().parse::<usize>().ok()) {
                let cfg = GenConfig { scale, ..parse_gen(&args) };
                let store = new_store(store_metrics())?;
                eprintln!("=== scale {scale}: write ===");
                let w = write_bench(&store, &bank_name(scale), cfg).await?;
                eprintln!("=== scale {scale}: read ===");
                let store = new_store(store_metrics())?;
                let rd = read_bench(&store, &bank_name(scale), cfg, 200).await?;
                reports.push((w, rd));
            }
            println!("\n{}", render_suite(&reports));
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            std::process::exit(2);
        }
    }
    Ok(())
}

fn bank_name(scale: usize) -> String {
    format!("perf-{scale}")
}

fn store_metrics() -> Arc<StoreMetrics> {
    StoreMetrics::new()
}

/// A MinIO-backed store with lifetime op accounting attached.
fn new_store(metrics: Arc<StoreMetrics>) -> Result<Store> {
    let endpoint =
        std::env::var("MEMLAKE_S3_ENDPOINT").unwrap_or_else(|_| "http://localhost:9000".into());
    Store::s3("memlake", Some(&endpoint), "memlake", "memlake123", "us-east-1")
        .map(|s| s.with_store_metrics(metrics))
        .context("connecting to MinIO (is `docker compose up -d` running?)")
}

// ---------------------------------------------------------------- write bench

struct WriteReport {
    scale: usize,
    commit_secs: f64,
    index_secs: f64,
    put_count: u64,
    list_count: u64,
    bytes_written: u64,
    cost: cost::WriteCost,
}

impl WriteReport {
    fn render(&self) -> String {
        format!(
            "write @ {scale}\n  \
             commit:  {csec:.1}s  ({wps:.0} memories/s)\n  \
             index:   {isec:.1}s  ({ips:.0} memories/s)\n  \
             S3:      {puts} PUTs, {lists} LISTs, {gb:.2} GB written\n  \
             cost:    ${ingest:.4} ingest requests, ${perM:.3}/1M memories, \
             ${store:.4}/GB-month stored ({sgb:.2} GB)",
            scale = self.scale,
            csec = self.commit_secs,
            wps = self.scale as f64 / self.commit_secs.max(1e-6),
            isec = self.index_secs,
            ips = self.scale as f64 / self.index_secs.max(1e-6),
            puts = self.put_count,
            lists = self.list_count,
            gb = cost::gb(self.bytes_written),
            ingest = self.cost.ingest_requests_usd,
            perM = self.cost.usd_per_million_ingested,
            store = self.cost.storage_usd_month,
            sgb = cost::gb(self.cost.stored_bytes),
        )
    }
}

async fn write_bench(store: &Store, bank: &str, cfg: GenConfig) -> Result<WriteReport> {
    let metrics = store.store_metrics().unwrap().clone();
    let ns = Namespace::new(bank, store.clone());
    ns.create_if_absent(&Tokenizer::default().config_hash()).await?;
    let gen = Generator::new(cfg);

    // Commit phase.
    let base = metrics.snapshot();
    let t0 = Instant::now();
    let mut writer = Writer::new(ns.clone());
    let mut start = 0;
    while start < cfg.scale {
        let end = (start + COMMIT_BATCH).min(cfg.scale);
        let ops: Vec<Op> = gen.batch(start, end).into_iter().map(Op::Upsert).collect();
        writer.commit(ops).await?;
        start = end;
    }
    let commit_secs = t0.elapsed().as_secs_f64();

    // Index phase (full first build).
    let ti = Instant::now();
    index(&ns, &Tokenizer::default(), IndexOptions::default()).await?;
    let index_secs = ti.elapsed().as_secs_f64();

    let phase = metrics.since(&base);
    let (manifest, _) = ns.read_manifest().await?;
    let stored_bytes = generation_bytes(store, &manifest).await?;
    let c = cost::write_cost(&cost::Pricing::default(), &phase, stored_bytes, cfg.scale);

    Ok(WriteReport {
        scale: cfg.scale,
        commit_secs,
        index_secs,
        put_count: phase.puts,
        list_count: phase.lists,
        bytes_written: phase.put_bytes,
        cost: c,
    })
}

/// Total stored footprint of the current generation across all memory types (HEAD each
/// referenced object).
async fn generation_bytes(store: &Store, manifest: &mlake_core::Manifest) -> Result<u64> {
    let mut total = 0u64;
    for path in manifest.all_referenced_paths() {
        total += store.head(path).await.unwrap_or(0);
    }
    Ok(total)
}

// ---------------------------------------------------------------- read bench

struct WorkloadResult {
    name: String,
    p50_ms: f64,
    p90_ms: f64,
    p99_ms: f64,
    roundtrips: f64,
    cost: cost::ReadCost,
    /// Per-phase micros per query (name, us/query), non-zero phases only.
    phases: Vec<(String, f64)>,
}

struct ReadReport {
    scale: usize,
    memory_types: Vec<u8>,
    cold: Vec<WorkloadResult>,
    warm: Vec<WorkloadResult>,
}

impl ReadReport {
    fn render(&self) -> String {
        let mut s = format!("read @ {} (memory_types {:?})\n", self.scale, self.memory_types);
        s.push_str("  cold:\n");
        for w in &self.cold {
            s.push_str(&fmt_workload(w));
        }
        s.push_str("  warm:\n");
        for w in &self.warm {
            s.push_str(&fmt_workload(w));
        }
        s
    }
}

fn fmt_workload(w: &WorkloadResult) -> String {
    let breakdown = if w.phases.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = w
            .phases
            .iter()
            .map(|(n, us)| format!("{n} {:.0}us", us))
            .collect();
        format!("\n                   [{}]", parts.join("  "))
    };
    format!(
        "    {name:14} p50 {p50:>6.1}ms  p90 {p90:>6.1}ms  p99 {p99:>6.1}ms  \
         rt {rt:>3.1}  {gpq:>4.1} GET/q  ${cost:.4}/1k{breakdown}\n",
        name = w.name,
        p50 = w.p50_ms,
        p90 = w.p90_ms,
        p99 = w.p99_ms,
        rt = w.roundtrips,
        gpq = w.cost.gets_per_query,
        cost = w.cost.usd_per_1k_queries,
    )
}

async fn read_bench(store: &Store, bank: &str, cfg: GenConfig, queries: usize) -> Result<ReadReport> {
    // A cache-backed store so the warm pass is served locally. Wiped up front so the cold
    // pass is genuinely cold (the cache dir persists across runs).
    let cache = Arc::new(DiskCache::new(
        std::env::temp_dir().join(format!("mlake-perf-cache-{}", cfg.scale)),
        1 << 30,
    )?);
    cache.wipe().ok();
    let store = store.clone().with_cache(cache);
    let ns = Namespace::new(bank, store.clone());
    let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await?;
    let gen = Generator::new(cfg);

    let memory_types = node.memory_types();
    let mt = *memory_types.first().unwrap_or(&1);

    // Build the query set (vectors near random centers) and the workloads.
    let qs: Vec<Vec<f32>> = (0..queries)
        .map(|i| gen.query_vector(i % gen.center_count(), 7 + i as u64))
        .collect();

    let novec = QueryConfig { graph_weight: 0.0, ..QueryConfig::default() };
    let withgraph = QueryConfig::default();
    type WB = Box<dyn Fn(&[f32]) -> (Option<Vec<f32>>, Option<String>, TagFilter, QueryConfig)>;
    let workloads: Vec<(&str, WB)> = vec![
        ("vector", Box::new(move |q: &[f32]| (Some(q.to_vec()), None, TagFilter::none(), novec))),
        ("fts", Box::new(move |_q: &[f32]| (None, Some("memory vector search".into()), TagFilter::none(), novec))),
        ("hybrid", Box::new(move |q: &[f32]| (Some(q.to_vec()), Some("memory vector".into()), TagFilter::none(), novec))),
        ("graph", Box::new(move |q: &[f32]| (Some(q.to_vec()), None, TagFilter::none(), withgraph))),
        ("tags-any", Box::new(move |q: &[f32]| (Some(q.to_vec()), None, TagFilter::new(vec!["tag-1".into(), "tag-2".into()], TagsMatch::Any), novec))),
        ("tags-strict", Box::new(move |q: &[f32]| (Some(q.to_vec()), None, TagFilter::new(vec!["tag-1".into()], TagsMatch::AnyStrict), novec))),
    ];

    // Cold pass: fresh cache-miss metrics per query. Warm pass: repeat (served from cache).
    let mut cold = Vec::new();
    let mut warm = Vec::new();
    for (name, build) in &workloads {
        cold.push(run_workload(&node, mt, &qs, build, store.store_metrics().unwrap(), name).await?);
        warm.push(run_workload(&node, mt, &qs, build, store.store_metrics().unwrap(), name).await?);
    }

    Ok(ReadReport { scale: cfg.scale, memory_types, cold, warm })
}

#[allow(clippy::type_complexity)]
async fn run_workload(
    node: &QueryNode,
    memory_type: u8,
    qs: &[Vec<f32>],
    build: &dyn Fn(&[f32]) -> (Option<Vec<f32>>, Option<String>, TagFilter, QueryConfig),
    store_metrics: &Arc<StoreMetrics>,
    name: &str,
) -> Result<WorkloadResult> {
    let base = store_metrics.snapshot();
    let mut latencies = Vec::with_capacity(qs.len());
    let mut total_rt = 0usize;
    let mut phase_us: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for q in qs {
        let (vector, text, tags, config) = build(q);
        let qm = QueryMetrics::new();
        let t = Instant::now();
        node.query_metered(
            memory_type,
            vector.as_deref(),
            text.as_deref(),
            &tags,
            10,
            config,
            &qm,
        )
        .await?;
        latencies.push(t.elapsed().as_secs_f64() * 1000.0);
        total_rt += qm.roundtrips();
        for (name, us) in qm.phase_breakdown() {
            *phase_us.entry(name.to_string()).or_default() += us;
        }
    }
    let n = qs.len().max(1) as f64;
    let phases: Vec<(String, f64)> = phase_us
        .into_iter()
        .filter(|(_, us)| *us > 0)
        .map(|(name, us)| (name, us as f64 / n))
        .collect();
    let phase = store_metrics.since(&base);
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(WorkloadResult {
        name: name.to_string(),
        p50_ms: pct(&latencies, 50.0),
        p90_ms: pct(&latencies, 90.0),
        p99_ms: pct(&latencies, 99.0),
        roundtrips: total_rt as f64 / qs.len().max(1) as f64,
        cost: cost::read_cost(&cost::Pricing::default(), &phase, qs.len()),
        phases,
    })
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn render_suite(reports: &[(WriteReport, ReadReport)]) -> String {
    let mut s = String::from("# memlake performance suite\n\n");
    for (w, r) in reports {
        s.push_str(&w.render());
        s.push('\n');
        s.push_str(&r.render());
        s.push('\n');
    }
    s
}

// ---------------------------------------------------------------- arg parsing

fn parse_gen(args: &[String]) -> GenConfig {
    GenConfig {
        scale: flag_usize(args, "--scale", 10_000),
        memory_types: flag_usize(args, "--types", 3) as u8,
        seed: flag_usize(args, "--seed", 42) as u64,
        ..GenConfig::default()
    }
}

fn flag_usize(args: &[String], flag: &str, default: usize) -> usize {
    flag_str(args, flag, "").parse().unwrap_or(default)
}

fn flag_str(args: &[String], flag: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}
