# mlake-perf

Performance suite: drives the real **write → index → query** path over storage (MinIO) at
scale and reports write throughput + build time, read latency + roundtrips, and S3 cost.

## Prerequisites

MinIO up (`docker compose up -d` from repo root). Endpoint overridable via
`MEMLAKE_S3_ENDPOINT` (default `http://localhost:9000`).

## Usage

```bash
cargo build -p mlake-perf --release

# Write (generate + commit + index) a synthetic bank at a scale
./target/release/mlake-perf write --scale 100000 --types 3

# Read workloads against a bank that was written first
./target/release/mlake-perf read  --scale 100000 --queries 200

# Full sweep: write + read each scale, then a report
./target/release/mlake-perf suite --scales 10000,100000,1000000
```

## What it measures

- **Write**: commit throughput, index build seconds, PUTs + LISTs + bytes, and cost
  (`$/1M memories ingested`, `$/GB-month stored`).
- **Read**, per workload (vector / fts / hybrid / graph / tags-any / tags-strict), cold and
  warm: latency p50/p90/p99, roundtrips, GET/query, and `$/1k queries`.

## Data generator

Reproducible (seeded) synthetic memories exercising every arm: clustered vectors, text,
Zipfian tags over a large vocabulary (high cardinality + realistic selectivity), Zipfian
entities + causal edges (graph; semantic kNN links are derived by the indexer), across
several independent memory types.

## Cost model

AWS S3 Standard us-east-1 pricing (`src/cost.rs`), computed from counted store operations —
so an indexer change that doubles PUTs shows directly as a cost change.
