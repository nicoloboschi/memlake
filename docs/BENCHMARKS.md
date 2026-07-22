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

> **Scale + per-arm profiling → [`benchmarks-scale.md`](benchmarks-scale.md).** This page is the
> 100k end-to-end (client→gRPC→S3) view. The 1M scale ladder, the per-arm latency breakdown
> (vector / BM25 / graph / temporal), the concurrent-request percentiles, and the optimization
> history (graph **644ms→~8.7ms**, FTS/hybrid **~44ms→~8ms**, the payload store, the temporal
> pool cap, the parallelized fold) all live there, measured in-process at 1M where the wins are
> visible. This page's arm timings inherit those improvements.

---

## Write & index throughput (100k, 3 memory_types)

| Phase | Result | Notes |
|---|---|---|
| **write** | ~9–10s — **~10,000 memories/s** | 196 batches via the client; group-committed to the WAL. Acks on durable PUT, not on indexing. |
| **index (first build)** | **~58s** — ~1,700 memories/s | k-means train + kNN link derivation + FTS + writes, one generation. Varies ~55–75s run-to-run (MinIO + scheduling). |
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
accumulation order is fixed. At millions of items the two remaining serial O(N·√N) passes
(the cluster-size histogram and per-item assignment) also run across cores now — the `train`
phase dropped **67s→~20s per memory_type at 1M** (see the scale doc).

The fold also builds a **payload store** (`payload.idx`/`payload.data`): one addressable row per
memory (embedding stripped) so a point read — an FTS/graph hit, a `get` — fetches one memory
via a ranged GET instead of deserializing its whole cluster file. It adds ~17% to stored bytes
(vector-stripped rows) and is what took FTS/graph query latency from ~44ms to ~8ms at 1M.

### Cost (100k)

Write ~197 PUTs, index 574 PUTs / 0.19 GB → **~$0.0039 ingest requests, ~$0.039/1M memories**,
$0.0043/GB-month stored. S3 Standard us-east-1, from counted store ops.

---

## Read latency & roundtrips (100k, bounded 64MB/512MB cache)

memlake serves exactly one query shape: **a single call across all memory_types running the
arms** (dense vector + BM25 + graph, and the temporal arm when a time window is given). It
returns the raw per-arm scores; the client fuses. Reads are always strongly consistent; the
server caches the open snapshot per namespace and reuses it after one cheap WAL-head check, so a
warm read is pure in-memory arm evaluation over the local cache — **0 object-storage
roundtrips**. `3way+tags` adds a tag filter.

| Workload | cold p50 | cold p99 | cold rt | warm p50 | warm p99 | warm rt |
|---|---:|---:|---:|---:|---:|---:|
| **3way** | **10.4ms** | 183ms | **1.3** | **9.5ms** | 146ms | **0.0** |
| 3way+tags | 27.4ms | 362ms | 0.0 | 27.6ms | 215ms | 0.0 |

One `3way` call fans out to 3 memory_types × 3 arms = 9 arm-executions and returns the full
raw candidate set (~800 hits: up to `top_k` per arm per type) yet consumes only **~1.3 shared
roundtrips cold / 0 warm** — the reads coalesce into common waves instead of multiplying. The
p50 reflects evaluating and serializing all candidates so the client has everything it needs
to fuse; p99 tails are first-touch cluster fetches / GC pauses. `3way+tags` costs more because
Zipfian tags admit many clusters under pruning. **Roundtrips are bounded** (≤~1.3 cold, 0
warm) independent of corpus size — INV-7, verified through the network path.

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

The `memlake` engine writes the corpus through the client, runs the indexer, and issues **one
query per BEIR query** — memlake returns the raw per-arm scores and the engine derives the
dense / sparse / hybrid rankings client-side (RRF over the raw ranks), exactly as Hindsight
would. Every engine is scored with identical metric code, so only the ranking differs. nDCG@10:

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
