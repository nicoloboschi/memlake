# memlake configuration reference

Every configuration knob: environment variables and CLI flags, with defaults. It aims to be the
single place that lists *all* of them.

## How config is loaded

- `mlake-server` and `mlake-perf` load a **`.env`** file (working dir or any ancestor) at startup
  via `dotenvy`. Any environment variable below can live there. `.env` is gitignored;
  [`.env.example`](../.env.example) is the template.
- **Precedence:** CLI flag → process environment (incl. `.env`) → built-in default. A real
  environment variable overrides the `.env` file, so a deployment overrides without editing it.
- The **object store** config additionally reads each field memlake-name-first, then the standard
  AWS name (so a plain `AWS_*` `.env` works). See `Store::from_env`.

---

## 1. Object store (required) — all components

Read by `Store::from_env()`; used by `mlake-server`, `mlake-perf`, and any tool that opens a store.

| Variable | Fallback | Default | Meaning |
|----------|----------|---------|---------|
| `MEMLAKE_S3_BUCKET` | — | *(required)* | Bucket that holds all data. |
| `MEMLAKE_S3_ENDPOINT` | — | *(unset ⇒ real AWS S3)* | Set for MinIO / any S3-compatible endpoint. |
| `MEMLAKE_S3_ACCESS_KEY` | `AWS_ACCESS_KEY_ID` | *(required)* | Access key. |
| `MEMLAKE_S3_SECRET_KEY` | `AWS_SECRET_ACCESS_KEY` | *(required)* | Secret key. |
| `MEMLAKE_S3_REGION` | `AWS_REGION` | `us-east-1` | Region. |

Leaving `MEMLAKE_S3_ENDPOINT` unset selects real AWS S3; setting it points at MinIO/other.

---

## 2. Query server — `mlake-server serve`

Stateless gRPC API. `serve` is the default mode (`mlake-server` with no subcommand).

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--addr` | — | `0.0.0.0:50051` | gRPC listen address. |
| `--mem-mb` | — | `256` | In-memory read-cache budget (MB). Capped by construction. |
| `--disk-mb` | — | `4096` | NVMe read-cache budget (MB). Capped by construction. |
| `--cache-dir` | — | `$TMPDIR/memlake-cache` | Disk cache location. |
| `--max-concurrent-queries` | `MEMLAKE_MAX_CONCURRENT_QUERIES` | `32` | Max in-flight `query`/`get` requests. Peak query memory ≈ this × per-query working set; excess requests **queue** (backpressure), never rejected. |
| `--node-id` | `MEMLAKE_NODE_ID` | `{host}-{pid}` | Node identity, stamped on log lines. |

---

## 3. Indexer — `mlake-server index`

Async, idempotent fold loop (its own deployment), or a single metered pass.

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--namespaces` | — | *(empty ⇒ discover all)* | Comma-separated namespaces to fold. |
| `--interval-secs` | — | `5` | Loop interval between fold passes. |
| `--once` | — | *(off)* | Run one metered pass, print a JSON summary, exit. |
| `--streaming` | — | *(off)* | Use the external-memory (bounded-RAM) fold instead of the in-RAM fold. |
| `--node-id` | `MEMLAKE_NODE_ID` | `{host}-{pid}` | Node identity (also holds the index lease). |

### Streaming-fold memory budget (env-only)

Per-stage RAM caps for the `--streaming` fold, in MB. Peak fold RAM is bounded by these, **not** by
corpus size. Stages run mostly sequentially (resolution, then one memory_type at a time), so a safe
peak estimate is `resolve` **or** (`cluster + payload + index + radj + fts`), whichever is larger —
they do not all sum at once. No CLI flags; env-only.

| Variable | Default (MB) | Stage |
|----------|--------------|-------|
| `MEMLAKE_FOLD_RESOLVE_MB` | `128` | Phase-1 id resolution (spills events). |
| `MEMLAKE_FOLD_CLUSTER_MB` | `256` | Cluster grouping (full item bytes). |
| `MEMLAKE_FOLD_PAYLOAD_MB` | `128` | Payload store. |
| `MEMLAKE_FOLD_INDEX_MB` | `96` | pk + entity + time (split three ways). |
| `MEMLAKE_FOLD_RADJ_MB` | `64` | Reverse-adjacency (causal edges). |
| `MEMLAKE_FOLD_FTS_MB` | `128` | tantivy writer arena (floored at 15 MB). |

The in-RAM fold (no `--streaming`) ignores these — its RAM is O(N) by design and is only for corpora
that fit in memory.

---

## 4. Diagnostics — all Rust binaries

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_TIMING` | *(off)* | If set to anything, print per-phase index build timings to stderr. |
| `RUST_LOG` | `info` | Standard `tracing` env filter (server logging). |

---

## 5. Scale/throughput benchmark — `mlake-perf` (dev tool)

Drives the real write → index → query path over the store. Not deployed. Uses the object-store env
(§1), the fold budget (§3), and `MEMLAKE_TIMING` (§4).

Subcommands: `write`, `read`, `suite`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--scale` | `10000` | Number of synthetic memories. |
| `--types` | `3` | Number of memory types in the synthetic corpus. |
| `--seed` | `42` | Datagen RNG seed (deterministic corpus). |
| `--streaming` | *(off)* | Index with the external-memory fold. |
| `--no-index` | *(off)* | Write only; skip the index phase (pure commit-throughput runs). |
| `--no-links` | *(off)* | Skip semantic kNN link derivation in the in-RAM fold. **No-op under `--streaming`** (the streaming fold never derives in-fold links). |
| `--commit-concurrency` | `16` | Pipelined WAL PUT concurrency for the bulk-load commit. |
| `--queries` | `200` | Query count for `read`. |
| `--concurrency` | `1` | Concurrent query workers for `read`. |
| `--scales` | `10000,100000,1000000` | Comma-separated scales for `suite`. |
| `--mem-mb` | `256` | Read-cache mem budget for `read`. |
| `--disk-mb` | `4096` | Read-cache disk budget for `read`. |

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_TEMPORAL_SPAN` | `120` | Time-window span (in synthetic units) for temporal benchmark queries. |

---

## 6. Recall benchmark — `mlake-bench` (dev tool)

BEIR recall harness: builds a generation over a cached corpus and scores rankings against Qdrant's.
Not deployed. Args: `mlake-bench <dataset> [testdata_dir] [out_file]`. Retrieval tuning via env:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_NPROBE` | `64` | IVF probe width. |
| `MEMLAKE_ARM_DEPTH` | `200` | Per-arm candidate depth. |
| `MEMLAKE_RRF_K` | `60.0` | RRF fusion constant. |
| `MEMLAKE_VEC_WEIGHT` | `1.0` | Vector-arm fusion weight. |
| `MEMLAKE_FTS_WEIGHT` | `1.0` | FTS-arm fusion weight. |
| `MEMLAKE_GRAPH_WEIGHT` | `0.25` | Graph-arm fusion weight. |
| `MEMLAKE_GRAPH` | `0` | `1` enables the graph arm, `0` disables. |

These configure the benchmark's own retrieval knobs; the production query API takes them per-request
(`QueryConfig`), not from the environment.

---

## 7. Admin UI — `admin/` (Next.js dev tool)

Reads its own env (see [`admin/.env.local.example`](../admin/.env.local.example)), not `.env`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMLAKE_ADDR` | `localhost:50051` | Address of the `mlake-server` gRPC endpoint. |
| `MEMLAKE_PROTO_PATH` | `../proto/memlake/v1/memlake.proto` | Proto file (resolved relative to `admin/`). |
| `MEMLAKE_EMBEDDINGS` | *(on)* | `off` disables server-side embedding (use raw-vector / text-only queries). |

---

## 8. Local MinIO — `docker-compose.yml`

The dev object store. Credentials are fixed in the compose file:

| Setting | Value |
|---------|-------|
| endpoint | `http://localhost:9000` (console `:9001`) |
| `MINIO_ROOT_USER` | `memlake` |
| `MINIO_ROOT_PASSWORD` | `memlake123` |
| bucket | `memlake` |

Point the Rust tools at it with the MinIO preset at the bottom of `.env.example`.

---

## Cleanup notes / gotchas

- **`--no-links` is a no-op under `--streaming`.** The streaming fold never derives semantic links
  in-fold (they're incremental / query-time at scale), so passing both is harmless but redundant.
- **The server `index` mode has no `--no-links` flag** (only `mlake-perf` does). The server's in-RAM
  fold always derives links; use `--streaming` to skip them.
- **Two benchmark crates, different jobs:** `mlake-perf` = scale/throughput/latency;
  `mlake-bench` = BEIR recall vs Qdrant. Both are dev-only and not part of a deployment.
- **`--mem-mb` / `--disk-mb`** appear in both `mlake-server serve` and `mlake-perf read` with the
  same meaning (read-cache budgets) — intentional, not a duplicate to remove.
- The fold-budget vars are **env-only by design** (per-stage tuning is an ops concern, not a
  per-invocation flag).
