# Deployment

memlake is designed to drop in where Hindsight currently connects to Postgres: instead of a
DB connection, the caller talks gRPC to a memlake `Service`. Because **object storage is the
only stateful dependency** (INV-1), every server replica is interchangeable — there is no
leader, no sticky state, no per-pod data to lose.

## Two Deployments, one image

The binary (`mlake-server`) runs in two modes. They are deployed separately because they
scale and fail differently.

| Deployment | Command | Replicas | Role |
|---|---|---|---|
| **API** | `mlake-server serve` | N (scale with traffic) | stateless gRPC front end for reads + writes |
| **Indexer** | `mlake-server index` | 1+ (idempotent) | async loop that folds the WAL into new immutable generations |

Both are the same container image with different args, and both read S3 config from the same
environment. The indexer is deliberately *off* the request path: writes ack as soon as the
WAL object is durable (see below), and indexing catches up asynchronously. If two indexer
replicas race to build the same generation, the manifest CAS-swap picks a single winner
(INV-6), so running more than one is safe.

```
                    ┌───────────────────────── k8s Service ─────────────────────────┐
   Hindsight ──gRPC─┤  serve pod   serve pod   serve pod   ...   (N, stateless)      │
                    └───────────────────────────────────────────────────────────────┘
                                          │  (reads + writes)
                                          ▼
                                   ┌──────────────┐        ┌──────────────────────┐
                                   │  S3 / bucket │ ◀──────│ index Deployment (1+) │
                                   └──────────────┘        └──────────────────────┘
```

## What a write ack means

`Write` returns only after the batch is a durable, uniquely-sequenced WAL object in S3 (a
successful `If-None-Match` conditional PUT). It does **not** wait for indexing. The write is
immediately visible to any `STRONG`-consistency `Query`, which scans the un-indexed WAL tail
and overlays it on the last generation (INV-5). So read-after-write holds without the request
path ever blocking on the indexer.

## Protocol: gRPC (HTTP/2 + protobuf)

Chosen for an internal, east-west, service-to-service API: a typed `.proto` contract Hindsight
generates its client from, binary framing, and streaming. Vectors travel as raw
little-endian float32 (`Vector.f32le`) rather than JSON floats — ~4x smaller and zero-copy.
The RPC itself is not the latency bottleneck (S3 roundtrips dominate at milliseconds; framing
is microseconds), so the choice optimizes for schema clarity and operability.

### k8s load-balancing caveat

gRPC multiplexes all calls over one long-lived HTTP/2 connection, so a plain L4 `ClusterIP`
Service pins each client to a single pod and load ends up lopsided. Pick one:

- a **headless Service** + client-side round-robin (gRPC resolves the pod set), or
- an **L7 / gRPC-aware proxy or mesh** (Envoy, Linkerd, a gateway) in front of the pods.

### Routing

- **Correctness needs no affinity** — any replica can serve any read or write, because all
  coordination is S3 conditional writes. Scale replicas freely.
- **Cache warmth benefits from affinity** — each `serve` pod caches clusters/blocks locally
  (the bounded two-tier cache), so routing similar reads to the same replica raises the warm
  (0-roundtrip) hit rate. Optional: consistent-hash reads by `namespace`/`memory_type`. Send
  writes anywhere.

## Configuration (environment)

| Var | Default | Meaning |
|---|---|---|
| `MEMLAKE_S3_BUCKET` | `memlake` | bucket / namespace root |
| `MEMLAKE_S3_ENDPOINT` | `http://localhost:9000` | S3 endpoint; **unset for real AWS S3** |
| `MEMLAKE_S3_ACCESS_KEY` / `MEMLAKE_S3_SECRET_KEY` | `memlake` / `memlake123` | credentials |
| `MEMLAKE_S3_REGION` | `us-east-1` | region |
| `RUST_LOG` | `info` | log filter |

Flags: `serve --addr 0.0.0.0:50051`; `index --namespaces a,b --interval-secs 5` (omit
`--namespaces` to discover every namespace in the bucket).

## Local smoke test

```bash
docker compose up -d                                          # MinIO
cargo run --release -p mlake-server -- serve --addr 127.0.0.1:50051 &
cargo run --release -p mlake-server -- index --namespaces smoke-py --interval-secs 3 &
uv run --project clients/python clients/python/scripts/smoke.py
```

The API is defined in [`../proto/memlake/v1/memlake.proto`](../proto/memlake/v1/memlake.proto);
the Python client is in [`../clients/python`](../clients/python/README.md).
