# Benchmarks

All performance numbers are **end-to-end through the deployed path**: the Python client →
gRPC → `mlake-server` → S3 (real MinIO, `docker compose up -d`). Nothing here is an
in-process micro-benchmark. Reproduce:

```bash
uv run --project bench memlake-bench perf --scales 10000,100000 --queries 200 --mem-mb 64 --disk-mb 512
```

The BEIR accuracy comparison (also e2e via the client-server) is at the bottom. Latency and
throughput vary with hardware — these were taken on a 14-core machine against a local MinIO.
Update this page when a run materially changes a number.

---

## Write & index throughput (100k, 3 memory_types)

| Phase | Result | Notes |
|---|---|---|
| **write** | 9.1s — **10,900 memories/s** | 196 batches via the client; group-committed to the WAL. Acks on durable PUT, not on indexing. |
| **index (first build)** | **56s** — 1,780 memories/s | k-means train + kNN link derivation + FTS + writes, one generation. Varies ~55–75s run-to-run (MinIO + scheduling). |
| **index (incremental re-index)** | ~30s | folding new WAL entries onto an existing generation reuses trained centroids (train ≈ 0). |

The indexer runs as its own process (`index --once` here; a Deployment in prod), so its cost
is off the write path entirely.

### First-build index breakdown (summed across 3 memory_types, from a `MEMLAKE_TIMING` run)

| Phase | Time | Parallelized? |
|---|---:|---|
| `train` (k-means) | ~28s | ✅ assignment + k-means++ seeding across cores |
| `derive_links` (semantic kNN) | ~35s | ✅ per-memory neighbour search across cores |
| `cluster_write` | ~1.9s | ✅ concurrent PUTs |
| `fts_build` | ~1.4s | tantivy split per type |
| `write_meta` | ~0.4s | one `try_join!` |

Both dominant phases (`train`, `derive_links`) are CPU-bound and scale with cores. They were
serial originally (~140s each ≈ 300s total); parallelizing them (rayon, determinism preserved
for G-6) is what brought a fresh 100k build down to ~55–75s. The k-means output stays
byte-identical for a given seed — only the assignment/seeding compute is parallel; the f64
accumulation order is fixed.

### Cost (100k)

Write ~197 PUTs, index 574 PUTs / 0.19 GB → **~$0.0039 ingest requests, ~$0.039/1M memories**,
$0.0043/GB-month stored. S3 Standard us-east-1, from counted store ops.

---

## Read latency & roundtrips (100k, bounded 64MB/512MB cache)

Measured through the client with `EVENTUAL` consistency (retrieval reads don't need
read-your-writes). The server caches the open snapshot per namespace, so a warm read is pure
in-memory fusion over the local cache — **0 object-storage roundtrips**.

| Workload | cold p50 | cold p99 | cold rt | warm p50 | warm p99 | warm rt |
|---|---:|---:|---:|---:|---:|---:|
| vector | 1.18ms | 14.4ms | 0.9 | 1.16ms | 1.40ms | 0.0 |
| fts | 1.05ms | 1.51ms | 0.0 | 1.13ms | 8.59ms | 0.0 |
| hybrid | 1.79ms | 3.18ms | 0.0 | 1.75ms | 6.49ms | 0.0 |
| graph | 1.96ms | 8.31ms | 0.0 | 2.05ms | 10.3ms | 0.0 |
| tags-any | 4.12ms | 5.10ms | 0.0 | 4.14ms | 15.3ms | 0.0 |
| tags-strict | 3.53ms | 4.58ms | 0.0 | 3.62ms | 22.5ms | 0.0 |

p99 tails are first-touch cluster fetches / GC pauses; p50/p90 are the steady state.

Single-digit-ms p50 across every arm, and **roundtrips are bounded** (≤1 cold, 0 warm)
independent of corpus size — INV-7, now verified through the network path. `tags-*` cost more
because Zipfian tags admit many clusters under pruning (`fetch_clusters` + `rerank`).

### 10k, same cache

Sub-millisecond p50 across all workloads (vector 0.5ms, fts 0.3ms, hybrid 0.6ms, graph 0.8ms,
tags ~0.9–1.1ms), 0 roundtrips warm. Write 11.5k memories/s, index 3.2s.

### Predictable resources

```
cache budget: mem 64 MB   disk 512 MB (enforced by construction)
```

The server's read cache is two-tier with independent, bounded memory and disk budgets, both
capped by construction (memory eviction demotes to disk, disk eviction deletes). Set via
`serve --mem-mb --disk-mb`. To exercise the demote-to-disk path, lower `--mem-mb`.

---

## Accuracy — BEIR vs Qdrant (e2e via the client-server)

The `memlake` engine writes the corpus through the client, runs the indexer, and queries each
arm over the wire; every engine is scored with identical metric code, so only the ranking
differs. nDCG@10:

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
uv run --project bench memlake-bench report
```

Full analysis in [`DECISIONS.md`](DECISIONS.md).
