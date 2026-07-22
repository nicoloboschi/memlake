# memlake configuration reference

Every configuration knob, by service. Config is **service-scoped**: each service reads its own
`MEMLAKE_<SERVICE>_*` variables and builds its own config object — there is no shared/unprefixed
fallback, and shared classes (the object store, the fold) never read the environment themselves.

## How config is loaded

- `mlake-server` and `mlake-perf` load a **`.env`** file (working dir or any ancestor) at startup
  via `dotenvy`. Any variable below can live there. `.env` is gitignored; [`.env.example`](../.env.example)
  is the template.
- **Precedence:** CLI flag → process environment (incl. `.env`) → built-in default.
- **Services:** `QUERY` (`mlake-server serve`), `INDEXER` (`mlake-server index`), `PERF`
  (`mlake-perf`), `BENCH` (`mlake-bench`). The admin UI is a separate app with its own env file.

---

## 1. Object store (per service, required)

Each service parses its own `MEMLAKE_<SERVICE>_S3_*` block into an `S3Config` and hands it to
`Store::from_s3_config`. No shared or `AWS_*` fallback — every service is self-describing.

| Variable (per `<SERVICE>` ∈ QUERY, INDEXER, PERF) | Default | Meaning |
|---------------------------------------------------|---------|---------|
| `MEMLAKE_<SERVICE>_S3_BUCKET` | *(required)* | Bucket holding all data. |
| `MEMLAKE_<SERVICE>_S3_ENDPOINT` | *(unset ⇒ real AWS S3)* | Set for MinIO / S3-compatible. |
| `MEMLAKE_<SERVICE>_S3_ACCESS_KEY` | *(required)* | Access key. |
| `MEMLAKE_<SERVICE>_S3_SECRET_KEY` | *(required)* | Secret key. |
| `MEMLAKE_<SERVICE>_S3_REGION` | `us-east-1` | Region. |

E.g. the indexer reads `MEMLAKE_INDEXER_S3_BUCKET`, the perf tool `MEMLAKE_PERF_S3_BUCKET`, etc.

---

## 2. QUERY service — `mlake-server serve`

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--addr` | — | `0.0.0.0:50051` | gRPC listen address. |
| `--mem-mb` | — | `256` | In-memory read-cache budget (MB). |
| `--disk-mb` | — | `4096` | NVMe read-cache budget (MB). |
| `--cache-dir` | — | `$TMPDIR/memlake-cache` | Disk cache location. |
| `--max-concurrent-queries` | `MEMLAKE_QUERY_MAX_CONCURRENT` | `32` | Max in-flight `query`/`get`. Peak query memory ≈ this × per-query working set; excess requests queue (backpressure), never rejected. |
| `--node-id` | `MEMLAKE_NODE_ID` | `{host}-{pid}` | Node identity for logs. |

---

## 3. INDEXER service — `mlake-server index`

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--namespaces` | — | *(empty ⇒ discover all)* | Comma-separated namespaces to fold. |
| `--interval-secs` | — | `5` | Loop interval between fold passes. |
| `--once` | — | *(off)* | Run one metered pass, print a JSON summary, exit. |
| `--node-id` | `MEMLAKE_NODE_ID` | `{host}-{pid}` | Node identity (also holds the index lease). |

### Fold selection & budget

There is **one fold entry point** that auto-selects the strategy — no `--streaming` flag. Below the
threshold it runs the in-RAM fold (faster, incremental copy-forward, derives semantic links); at or
above it, the bounded external-memory (streaming) fold (peak RAM set by the budget, not by corpus
size; full rebuild, no in-fold links).

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_INDEXER_STREAMING_THRESHOLD` | `4000000` | Live-doc cutoff for the choice. `0` forces streaming, a huge value forces in-RAM. |

Per-stage streaming-fold RAM budget (MB), consulted only on the streaming path. Stages run mostly
sequentially, so they don't all sum; a safe peak estimate is `resolve` **or**
(`cluster + payload + index + radj + fts`), whichever is larger.

| Variable | Default (MB) | Stage |
|----------|--------------|-------|
| `MEMLAKE_INDEXER_FOLD_RESOLVE_MB` | `128` | Phase-1 id resolution (spills events). |
| `MEMLAKE_INDEXER_FOLD_CLUSTER_MB` | `256` | Cluster grouping (full item bytes). |
| `MEMLAKE_INDEXER_FOLD_PAYLOAD_MB` | `128` | Payload store. |
| `MEMLAKE_INDEXER_FOLD_INDEX_MB` | `96` | pk + entity + time (split three ways). |
| `MEMLAKE_INDEXER_FOLD_RADJ_MB` | `64` | Reverse-adjacency (causal edges). |
| `MEMLAKE_INDEXER_FOLD_FTS_MB` | `128` | tantivy writer arena (floored at 15 MB). |

---

## 4. Diagnostics — all Rust binaries

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_TIMING` | *(off)* | If set, print per-phase index build timings to stderr. |
| `RUST_LOG` | `info` | Standard `tracing` env filter (server logging). |

---

## 5. PERF service — `mlake-perf` (dev tool)

Drives the real write → index → query path. Subcommands: `write`, `read`, `suite`. Uses the fold
selector too (`MEMLAKE_PERF_STREAMING_THRESHOLD`, `MEMLAKE_PERF_FOLD_*` — same stages as §3);
set the threshold to `0` to benchmark the streaming fold at small scale.

| Flag | Default | Meaning |
|------|---------|---------|
| `--scale` | `10000` | Number of synthetic memories. |
| `--types` | `3` | Memory types in the synthetic corpus. |
| `--seed` | `42` | Datagen RNG seed. |
| `--no-index` | *(off)* | Write only; skip the index phase (pure commit-throughput runs). |
| `--commit-concurrency` | `16` | Pipelined WAL PUT concurrency for the bulk-load commit. |
| `--queries` | `200` | Query count for `read`. |
| `--concurrency` | `1` | Concurrent query workers for `read`. |
| `--scales` | `10000,100000,1000000` | Comma-separated scales for `suite`. |
| `--mem-mb` / `--disk-mb` | `256` / `4096` | Read-cache budgets for `read`. |

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_PERF_TEMPORAL_SPAN` | `120` | Time-window span (synthetic units) for temporal queries. |
| `MEMLAKE_PERF_STREAMING_THRESHOLD` | `4000000` | Fold selector cutoff; `0` forces streaming. |
| `MEMLAKE_PERF_FOLD_*_MB` | (as §3) | Streaming-fold per-stage budget. |

---

## 6. BENCH service — `mlake-bench` (dev tool)

BEIR recall harness. Args: `mlake-bench <dataset> [testdata_dir] [out_file]`. Retrieval tuning via
env (configures the benchmark's own knobs; the production query API takes these per-request):

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_NPROBE` | `64` | IVF probe width. |
| `MEMLAKE_ARM_DEPTH` | `200` | Per-arm candidate depth. |
| `MEMLAKE_RRF_K` | `60.0` | RRF fusion constant. |
| `MEMLAKE_VEC_WEIGHT` / `MEMLAKE_FTS_WEIGHT` / `MEMLAKE_GRAPH_WEIGHT` | `1.0` / `1.0` / `0.25` | Per-arm fusion weights. |
| `MEMLAKE_GRAPH` | `0` | `1` enables the graph arm. |

---

## 7. Admin UI — `admin/` (Next.js dev tool)

Reads its own env (see [`admin/.env.local.example`](../admin/.env.local.example)), not `.env`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_ADDR` | `localhost:50051` | Address of the `mlake-server` gRPC endpoint. |
| `MEMLAKE_PROTO_PATH` | `../proto/memlake/v1/memlake.proto` | Proto file (relative to `admin/`). |
| `MEMLAKE_EMBEDDINGS` | *(on)* | `off` disables server-side embedding. |

---

## 8. Local MinIO — `docker-compose.yml`

| Setting | Value |
|---------|-------|
| endpoint | `http://localhost:9000` (console `:9001`) |
| user / password | `memlake` / `memlake123` |
| bucket | `memlake` |

For each service's S3 block, point `*_S3_ENDPOINT` at `http://localhost:9000` and use the MinIO
credentials.

---

## Notes

- **One fold, auto-selected.** The `--streaming` and `--no-links` flags are gone. Streaming is
  chosen automatically by corpus size; semantic links are derived by the in-RAM fold (small
  corpora) and skipped by the streaming fold (large ones), which is the same effect the old
  `--no-links` had at scale.
- **Config is duplicated per service** (each `MEMLAKE_<SERVICE>_S3_*` block is independent) by
  design — every service is fully self-describing, with no hidden shared state.
- The fold budget and streaming threshold are **env-only** (ops tuning, not per-invocation flags).
