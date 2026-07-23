# Compute COGS: what a vCPU of memlake `serve` buys, at an SLA

Companion to [`docs/cost-comparison.md`](cost-comparison.md), which prices **storage** and
explicitly defers compute ("What this leaves out → Compute"). This doc fills that gap with a
**measured** number: how much CPU serving queries actually costs, and therefore what
**sustained QPS at a stated p99 SLA** a given compute budget buys — the input to a compute
COGS line (`nodes = ceil(target_qps / per_node_qps_at_sla) × instance price`).

The in-process `mlake-perf` cannot answer this: it drives `QueryNode` in one process with no
gRPC, no serialization, no container CPU limit, no request concurrency. This number comes from
the **real** two-process topology under **real** container limits.

## How it was measured

- **Topology** — [`docker-compose.perf.yml`](../docker-compose.perf.yml): `minio` +
  `mlake-server serve` + `mlake-server index`, one image ([`Dockerfile`](../Dockerfile)).
  `serve` runs under `deploy.resources.limits.cpus`, which Docker enforces as a **CFS quota** —
  the same mechanism as a Kubernetes CPU `limit`, so `cpus: "1.0"` here is a faithful proxy for
  a 1-vCPU pod. Memory limit 3 GB, cache `--mem-mb 1536`, held constant so only CPU varies.
- **Harness** — [`perf/`](../perf/): seeds N synthetic 384-dim unit-vector memories through the
  real gRPC write path, folds them into a generation, then drives **concurrent** query load
  (grpc.aio, a pool of in-flight queries) for a fixed window after a warm-up prefix. It records
  achieved QPS + p50/p90/p99, samples the **serve container's CPU%** via `docker stats`, and
  reads the **cache hit ratio** (`CacheStats` RPC) — because p99 at a 99 % hit is a different
  SLA than at 10 %.
- **Workload** — the query runs all three arms (dense + BM25 + graph) with **server-default
  top-k**, which returns **~600 hits/query** at this corpus. That is a *heavy* query — see
  "Softer than it looks". Scale **100 k** memories, 3 memory_types.
- **Host** — Apple M-series, **14 cores** (Docker sees 14). The Python load driver shares those
  cores with the containers; `serve` is capped at ≤ 2 vCPU, leaving ≥ 12 cores for the driver +
  MinIO, so the driver is not the bottleneck. Index build (one-shot fold of 100 k) took **37 s**.

### Backend caveat — read before quoting any latency

The backend is **local MinIO**, labelled `minio` on every row. MinIO-local has ~zero network
latency, so this run measures the **CPU→QPS relationship faithfully** (the gap being filled) but
**understates cold-query latency**: a real cold query is S3-round-trip-bound (tens of ms ×
roundtrips), which MinIO does not reproduce. Moreover, at 100 k the whole generation (~230 MB)
is **RAM-resident** in the `QueryNode` after the snapshot opens, so warm queries do **0
object-storage roundtrips** (`rt` column ≈ 0, cache hit ratio ≈ 1.0) and are **pure CPU**. That
is exactly the regime where the CPU-per-QPS number is valid — and exactly why **the p50/p99 here
are CPU-queueing latencies, not the production SLA**. The compose can point at real AWS S3
(endpoint empty + real creds — see the compose header); the latency-SLA pass must be re-run
there before anyone quotes a p99 to a customer.

## Measured grid (100 k memories, warm, MinIO, backend=`minio`)

Raw rows in [`perf/results/rows.jsonl`](../perf/results/rows.jsonl). `util` = mean CPU ÷ the
container's vCPU limit (1.0 = the limit is saturated). `hit` = block-cache hit ratio for the
window; `rt` = mean object-storage roundtrips/query (0 = served from the RAM-resident snapshot).

| serve vCPU | concurrency | QPS | QPS/vCPU | p50 ms | p90 ms | p99 ms | cpu% mean | util | hit | rt |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1  | 2.2  | 4.3 | 366   | 919   | 1 536  | 50%  | 1.00 | 0.99 | 2.6 |
| 0.5 | 8  | 3.0  | 6.0 | 2 591 | 3 832 | 4 334  | 50%  | 1.00 | 1.00 | 0.0 |
| 0.5 | 32 | 2.8  | 5.5 | 9 894 | 13 706| 15 418 | 47%  | 0.95 | 1.00 | 0.0 |
| 0.5 | 64 | 3.6  | 7.3 | 17 016| 18 289| 20 220 | 48%  | 0.95 | 1.00 | 0.0 |
| 1.0 | 1  | 4.6  | 4.6 | 189   | 327   | 613    | 92%  | 0.92 | 0.99 | 1.3 |
| 1.0 | 8  | 7.7  | 7.7 | 1 023 | 1 453 | 1 776  | 100% | 0.99 | 1.00 | 0.0 |
| 1.0 | 32 | 7.0  | 7.0 | 4 578 | 6 362 | 7 264  | 100% | 1.00 | 1.00 | 0.0 |
| 2.0 | 1  | 5.8  | 2.9 | 144   | 207   | 845    | 96%  | 0.48 | 0.99 | 1.0 |
| 2.0 | 8  | 15.1 | 7.5 | 518   | 738   | 949    | 186% | 0.93 | 1.00 | 0.0 |
| 2.0 | 32 | 13.8 | 6.9 | 2 276 | 3 244 | 3 927  | 187% | 0.93 | 1.00 | 0.0 |
| 2.0 | 64 | 12.1 | 6.1 | 5 121 | 7 995 | 8 713  | 190% | 0.95 | 1.00 | 0.0 |

Reproduce: `uv run --project perf memlake-perf-sweep --scales 100000 --cpus 0.5,1,2 --concurrency 1,8,32,64`.

## The grid above is a *heavy* query — the realistic one is ~3.6× cheaper

The grid queried **all 3 memory_types with the default top-k**, returning ~600 hits/query.
That is not a Hindsight recall: a real recall targets **one** independent index (memory_type)
and asks for **top-k 10–50**. Re-measured with `--memory-type 1 --topk 20` (single type, 40
hits), same 100 k corpus, warm, MinIO:

| serve vCPU | concurrency | QPS | QPS/vCPU | p50 ms | p90 ms | p99 ms | cpu% | note |
|---|---|---|---|---|---|---|---|---|
| 1.0 | **1** | 28.5 | **28.5** | **33** | 41 | **66** | 92% | unqueued — true per-query latency |
| 1.0 | 2 | 29.5 | 29.5 | 63 | 92 | 112 | 84% | core already saturated at conc 1 |
| 1.0 | 8 | 27.2 | 27.2 | 289 | — | 548 | 97% | QPS flat; extra conc only queues |
| 2.0 | 8 | 52.5 | 26.2 | 143 | 221 | 271 | 194% | ~2× the 1-vCPU throughput (linear) |

Two things this makes clear, and they matter more than the heavy grid:

- **A realistic query costs ~33 ms of one core.** So **1 vCPU sustains ~28 QPS** and is
  saturated at **concurrency ≈ 1** — one in-flight query ≈ one busy core. Throughput scales
  linearly with vCPU (27 → 52 for 1 → 2), and per-vCPU QPS holds at **~26–28**.
- **Past one-query-per-core, adding concurrency buys no throughput, only queue.** The p99 in the
  heavy grid (0.9–15 s) and here at conc 8 (0.5 s) is *queue depth*, not query cost — the query
  itself is 33 ms p50 / 66 ms p99. Size concurrency to ≈ vCPU count and the tail stays tens of ms.

So the sizing number is **~28 QPS/vCPU at p99 < 70 ms** for a realistic recall — not the 7.6
below. The heavy grid stands as the worst-case floor for a broad, high-top-k, all-types query.

## What the numbers say (heavy-query grid)

1. **CPU→QPS is ~linear, and it is the ceiling.** At the saturation point (concurrency ≥ 8,
   where `util` ≈ 1.0) the peak QPS is **3.0 / 7.7 / 15.1** for **0.5 / 1 / 2 vCPU** — i.e.
   doubling the CPU limit ~doubles throughput. Per-query cost is stable at **~0.13 vCPU-seconds**
   (cores-used ÷ QPS across the saturated rows), so:

   > **memlake `serve` sustains ~7.6 QPS per vCPU** for this workload (100 k, 3-arm, ~600-hit
   > query, warm). That is the compute-throughput constant the COGS model needs.

2. **Concurrency must reach the core count to saturate, but past it only latency grows.** At
   `concurrency=1` a single query cannot fill 2 cores (`2.0/conc=1`: util **0.48**, half the box
   idle) — you need ≥ ~ncores in-flight to saturate. Beyond saturation, QPS is flat (even dips
   slightly from scheduling overhead) while p50/p99 climb linearly with the queue: at `0.5 vCPU,
   conc=64`, p50 is **17 s**. This is textbook CPU queueing, and it is *graceful* — **0 errors**
   at every point, throughput holds, latency degrades predictably.

3. **The CPU account ties out.** At saturation the serve container sits at its CFS limit
   (`util` 0.93–1.0); the bottleneck is genuinely serve CPU, not the driver, not MinIO, not the
   network. Where `util` < 1 (the `conc=1` rows) it is because one request can't fill the cores —
   not because something else is the bottleneck.

## Per-vCPU sustained QPS at a p99 SLA

Reading the grid for the best sustained QPS that still holds a p99 target (MinIO-warm p99 —
**CPU-queueing only**, not S3):

| p99 SLA | 0.5 vCPU | 1 vCPU | 2 vCPU | note |
|---|---|---|---|---|
| **≤ 1 s** | — (min p99 1.5 s) | 4.6 QPS @ conc 1 (4.6/vCPU) | **15.1 QPS @ conc 8 (7.5/vCPU)** | tail needs multiple cores to absorb |
| **≤ 2 s** | 2.2 QPS @ conc 1 | 7.7 QPS @ conc 8 (7.7/vCPU) | 15.1 QPS @ conc 8 (7.5/vCPU) | |
| **≤ 5 s** | 3.0 QPS @ conc 8 | 7.7 QPS @ conc 8 | 15.1 QPS @ conc 8 | |

The key non-obvious result: **to hold p99 ≤ 1 s you cannot run a fraction of a core hot** — a
single query's own service time plus queueing blows the tail. A **2-vCPU** node at concurrency 8
delivers **15 QPS at p99 < 1 s (~7.5 QPS/vCPU)**; a 1-vCPU node can only hold p99 < 1 s at
concurrency 1 (4.6 QPS). So the practical sizing unit is **~2 vCPU → ~15 QPS at a sub-second
(CPU-side) p99**.

## Turning it into a compute COGS line

`serve` nodes are stateless and identical, so:

```
serve_vcpu = ceil(target_qps / qps_per_vcpu_at_sla)
serve_cost = serve_vcpu × $/vCPU-hour × 730 h/month
```

Using the measured **7.5 QPS/vCPU at a (CPU-side) p99 < 1 s**, and an order-of-magnitude
on-demand rate of **~$0.04/vCPU-hour** (general-purpose x86; this moves with instance family and
region — treat as a model, not a quote):

| sustained target QPS | serve vCPU (@7.5 QPS/vCPU) | serve $/month | + storage @100 k (from cost-comparison) |
|---:|---:|---:|---:|
| 10 | 2 (1 node × 2 vCPU) | ~$58 | +~$0.02/mo storage (100 k × 7.2 KB) |
| 100 | 14 | ~$409 | storage still a rounding error |
| 1 000 | 134 | ~$3 900 | |
| 10 000 | 1 334 | ~$39 000 | S3 GET COGS now matters — see below |

**This inverts the storage story.** [`cost-comparison.md`](cost-comparison.md) shows storage COGS
is a rounding error below ~10 M memories ("storage is the wrong line to optimise below ~10 M").
This doc shows why: at 100 k memories the **entire** monthly bill is compute + S3 requests — the
$0.02/mo of storage is invisible next to hundreds-to-thousands of dollars of serve CPU. memlake's
storage win is real but only bends the total at scale; **below ~10 M memories the COGS lever is
`qps_per_vcpu`, not `$/GB`.** And the two are coupled through the cache: the RAM-resident-snapshot
regime that gives `rt≈0` here is also what keeps S3 GET COGS at $0 — the same
cache-hit-rate question `cost-comparison.md` flags for high QPS.

## Softer than it looks — every place the number needs an asterisk

- **MinIO ≠ S3. The latency SLA is optimistic and must be re-run against real S3 before anyone
  quotes it.** These p99s are pure CPU-queueing on a RAM-resident snapshot (`rt ≈ 0`). A cold
  query on real S3 pays bounded-but-real roundtrips (tens of ms each) that MinIO does not
  reproduce. The **CPU-per-QPS / throughput** numbers are valid on MinIO; the **latency**
  numbers are a floor. The compose is S3-ready for that pass.
- **The 7.6 QPS/vCPU is for a ~600-hit, 3-arm, default-top-k query.** That is a heavy query —
  ~0.13 vCPU-s each. A production caller asking for top-k 10–50 (Hindsight-style) would be
  markedly cheaper per query, so **7.6 QPS/vCPU is a conservative floor for this config, not a
  ceiling for all configs.** Re-run with realistic `--topk` before sizing a specific workload.
- **Only 100 k measured; the working set was RAM-resident.** At 100 k the generation fits in the
  cache/snapshot, so this is a *warm, in-RAM* number. At 1 M–10 M the working set exceeds RAM,
  probed clusters get fetched per query (the `rt` column would go positive), and both CPU (more
  candidates to score) and latency (real fetches) rise. **1 M was not completed** — see below.
- **1 M is not here, and indexing was the blocker, not querying.** The background indexer's 5 s
  loop **hangs on incremental re-folds at scale** in this environment (repeated tantivy
  commits, never republishing); the harness works around it with a one-shot `index --once` fold
  from a clean cursor (37 s for 100 k). That is a **real indexer finding worth a bug**, but it
  means the 1 M seed/fold path needs that fix (or more indexer CPU/time) before the 1 M query
  grid can run. Do not extrapolate 100 k QPS to 1 M — cluster fetches will change it.
- **The load driver shares the host.** 14 cores, serve ≤ 2, driver + MinIO on the rest — the
  driver is not the bottleneck here, but on a smaller host it would be; size the serve limit well
  below the host core count when reproducing.
- **`concurrency=1` under-reports multi-core nodes.** A 2-vCPU node at conc 1 shows 2.9 QPS/vCPU
  only because one request can't fill two cores; the node's real capacity is the conc-8 row
  (7.5/vCPU). Size by the saturated row, not the serial one.
- **Instance price is a placeholder.** $0.04/vCPU-hour is an order-of-magnitude general-purpose
  on-demand rate; the real COGS depends on instance family, reservation/spot, and region. Only
  the *ratios* and the `qps_per_vcpu` constant are measured here.
