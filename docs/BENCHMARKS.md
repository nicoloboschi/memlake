# Benchmarks

Performance numbers from the `mlake-perf` harness against **real MinIO** (`docker compose up
-d`), plus the BEIR accuracy comparison from `mlake-bench`. Reproduce with the commands in
each section. Latency/throughput vary with hardware — these were taken on a 14-core machine.

Update this page when a run materially changes a number; note the date and what changed.

---

## Write & index throughput

`cargo run --release -p mlake-perf -- write --scale N` (add `MEMLAKE_TIMING=1` for the
per-phase breakdown).

### 100k, 3 memory_types — index-build optimization history

Each row is the same workload after a successive optimization; the diagnostics (per-phase
timing) drove each step.

| Version | Index time | Throughput | What changed |
|---|---:|---:|---|
| Baseline | 275.5s | 363 mem/s | sequential cluster + metadata PUTs, sequential link derivation |
| + parallel writes | 148.5s | 673 mem/s | cluster PUTs `buffer_unordered(32)`, metadata PUTs `try_join!` |
| **+ parallel link derivation** | **30.7s** | **3256 mem/s** | `derive_links_ivf` across all cores (rayon) |

**9× faster** end-to-end. Commit phase (WAL group-commit) is ~1.8s (56k mem/s) and was never
the bottleneck.

Per-phase breakdown at the current version (100k, summed across 3 types):

| Phase | Time | Notes |
|---|---:|---|
| `derive_links` | ~24s | 16-probe × cluster-member cosine per new memory; now core-scaled |
| `cluster_write` | ~1.2s | parallel PUTs |
| `fts_build` | ~1.1s | tantivy split per type |
| `write_meta` | ~0.2s | centroids/tags/radj/pk/stats, one `try_join!` |
| `train` | ~0s | mini-batch k-means (samples ≤50k) |

`derive_links` is the remaining bottleneck; further gains would be algorithmic (SIMD cosine,
fewer probes) with diminishing returns.

### Cost (100k)

594 PUTs, 3 LISTs, 0.19 GB written → **$0.0032 ingest requests, $0.032/1M memories**,
$0.0086/GB-month stored. S3-op-count based (AWS S3 Standard us-east-1 pricing).

---

## Read latency & roundtrips

`cargo run --release -p mlake-perf -- read --scale N --queries 200 --mem-mb 64 --disk-mb 512`

### 100k, 3 memory_types, bounded 64MB/512MB cache

| Workload | cold p50 | cold p99 | cold RT | cold GET/q | warm p50 | warm p99 |
|---|---:|---:|---:|---:|---:|---:|
| vector | 1.6ms | 12.3ms | 0.9 | 0.9 | 1.0ms | 1.2ms |
| fts | 0.9ms | 1.1ms | 0.0 | 0.0 | 0.9ms | 1.1ms |
| hybrid | 1.5ms | 1.7ms | 0.0 | 0.0 | 1.5ms | 6.0ms |
| graph | 2.2ms | 16.4ms | 0.8 | 0.2 | 1.9ms | 2.9ms |
| tags-any | 3.8ms | 4.9ms | 0.0 | 0.0 | 4.0ms | 8.9ms |
| tags-strict | 3.4ms | 21.4ms | 0.0 | 0.0 | 3.4ms | 7.3ms |

Observations:

* **Roundtrips are bounded** (≤0.9 GET/query cold, 0 warm) independent of corpus size — INV-7
  holds at 100k.
* The graph arm's ranged-block cache pays off warm: `graph_pk` 795µs → 2µs, `graph_radj`
  72µs → 11µs, `graph_fetch` 291µs → 57µs.
* `vector`/`tags` are dominated by `fetch_clusters` + `rerank`; `tags-*` fetch more clusters
  because Zipfian tags admit many clusters under pruning.

### Predictable resources

```
cache:   mem 63.1/64 MB   disk 63.1/512 MB
```

Both cache tiers stay under their configured budgets by construction (two-tier cache: memory
eviction demotes to disk, disk eviction deletes). To exercise the demote-to-disk eviction
path, run with a smaller mem budget (e.g. `--mem-mb 16 --disk-mb 512`).

---

## Accuracy — BEIR vs Qdrant

`mlake-bench` embeds each corpus once (shared vector cache, so the comparison isolates
retrieval) and scores every engine with identical metric code. nDCG@10:

| Dataset  | Arm    | memlake | qdrant | |
|----------|--------|--------:|-------:|-|
| scifact  | dense  | 0.7127  | 0.7127 | parity |
| scifact  | sparse | 0.6907  | 0.6830 | **memlake wins** |
| scifact  | hybrid | 0.7325  | 0.7345 | parity (−0.3%) |
| nfcorpus | dense  | 0.3429  | 0.3436 | parity |
| nfcorpus | sparse | 0.3244  | 0.3236 | **memlake wins** |
| nfcorpus | hybrid | 0.3638  | 0.3626 | **memlake wins** |
| nfcorpus | +graph | 0.3645  | 0.3626 | **memlake wins** (R@100 0.3304 > 0.3165) |

```bash
uv run --project bench memlake-bench all scifact
uv run --project bench memlake-bench all nfcorpus
uv run --project bench memlake-bench baseline memlake nfcorpus --graph
uv run --project bench memlake-bench report        # renders bench/results/report.md
```

Full analysis in [`DECISIONS.md`](DECISIONS.md).
