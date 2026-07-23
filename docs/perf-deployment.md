# memlake compute-vs-SLA perf harness

Answers the one question the in-process `mlake-perf` cannot: **how much CPU does serving
queries actually take, and what latency/QPS SLA does a given compute budget buy** — measured
against the *real* two-process topology (`serve` + `index` + object storage) under *real*
container CPU/memory limits (`docker-compose.perf.yml`, whose `deploy.resources.limits.cpus`
is a CFS quota — a faithful proxy for a k8s CPU limit).

## What it measures

- **Achieved QPS** and **p50/p90/p99** over a steady-state window (a warm-up prefix is
  discarded), driven by a pool of `concurrency` in-flight gRPC queries (grpc.aio, not serial).
- The **serve container's CPU% and memory** during the window, sampled via
  `docker stats --no-stream` (mean + peak). Latency without CPU accounting cannot size compute.
- The **cache hit ratio** for the window, via the `CacheStats` RPC — p99 at 99% hit is a
  different SLA than at 10%, so every row is labelled with it.

Each run emits one JSON row:
`{scale, serve_cpus, serve_mem_mb, cache_mem_mb, concurrency, achieved_qps, p50, p90, p99,
cpu_pct_mean, cpu_pct_peak, cache_hit_ratio, ...}` appended to `perf/results/rows.jsonl`.

## Backend caveat (read before quoting any number)

The default backend is the bundled **MinIO**, which has ~zero network latency. That makes the
**CPU→QPS** relationship (the gap being filled) valid, but it **understates cold-query
latency**: the real SLA is S3-round-trip-bound. Every row is labelled `backend=minio`. To run
the latency-SLA pass against real AWS S3, point the compose at it (endpoint empty + real
creds) — see the header of `docker-compose.perf.yml`. **Do not quote the MinIO p99 as the
production SLA.** The load driver also shares host cores with the containers, so keep the serve
CPU limit well below the host core count (reported in each row as `host_cpus`).

## Run

```bash
# one point
uv run --project perf memlake-perf-harness --scale 100000 --concurrency 8

# the sweep (brings the stack up/recreates serve per cpu setting, tabulates)
uv run --project perf memlake-perf-sweep --scales 100000 --cpus 0.5,1,2 --concurrency 1,8,32,64
```

Results table + analysis: [`compute-cost.md`](compute-cost.md).
