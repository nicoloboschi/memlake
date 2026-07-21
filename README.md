# memlake

An S3-native retrieval engine for small memory records. One storage layer serves three
retrieval arms — **IVF vector search**, **BM25 full-text** (Chinese-capable), and **bounded
graph link-expansion**. Each hit carries the raw per-arm scores; the client fuses. Object
storage is the sole source of truth; query nodes are stateless caches that can be thrown away
and rebuilt from S3.

Built as a prototype from [`docs/SPEC.md`](docs/SPEC.md). Each arm's write and read path is
documented in detail: [`docs/arms/vector.md`](docs/arms/vector.md),
[`docs/arms/text.md`](docs/arms/text.md), [`docs/arms/graph.md`](docs/arms/graph.md),
[`docs/arms/temporal.md`](docs/arms/temporal.md).

## Model & naming

A **`Memory`** is the unit of storage: an id, an embedding `vector`, its `text`, a
`memory_type`, `tags`, causal edges, and a `proof_count`. The vocabulary is deliberately
narrow — these are the only nouns you need:

| Term            | Type                | What it is |
|-----------------|---------------------|------------|
| `Memory`        | `Memory` / `StoredMemory` | the record; `StoredMemory` is the on-disk form with derived semantic edges folded in |
| `MemoryId`      | `MemoryId` (16 bytes) | content-addressed id, derived from a caller key |
| `namespace`     | `Namespace`         | an isolated tenant/bank: its own manifest, WAL, and generations |
| `memory_type`   | `u8`                | an **independent sub-index** within a namespace |
| `tags`          | `Vec<String>` + `TagFilter` | post/pre-filter labels, matched by `TagsMatch` |

**`memory_type` is the key structural idea.** Each type is its own index — its own
centroids, clusters, FTS split, and graph. A query names one `memory_type` and gets results
from that type only; results are never fused *across* types. This lets one namespace hold,
say, `episodic`, `semantic`, and `entity` memories with different densities without one
type's statistics polluting another's ranking.

**Tags** filter within a type. `TagFilter::new(tags, mode)` supports five `TagsMatch` modes
— `Any`, `All`, `AnyStrict`, `AllStrict`, `Exact` — where the `*Strict` variants exclude
untagged memories. Tag summaries per cluster let the engine prune whole clusters before
fetching them, so filtering stays cheap at scale.

## Architecture

```
        write path                     index path                    query path
  client → any node               any node (idempotent)          any node (stateless)
     │ buffer + group commit         read WAL slice                 read manifest (RT1)
     │ PUT wal/{seq}.bin             fold → build 3 arms            load per-type meta (RT2)
     │  (If-None-Match: *)           write gen-{G}/ files           fetch probed clusters (RT3)
     ▼                               CAS-swap manifest              scan WAL tail (RT4)
   S3  ◀──────────── the only stateful dependency (INV-1) ────────────▶  fuse → results
```

The three paths are fully decoupled and each runs on any stateless node:

* **Write** — `Writer` buffers memories and group-commits a batch as one immutable
  `wal/{seq}.bin` object, claiming the sequence with `If-None-Match: *`. No node ever waits
  for the indexer.
* **Index** — an idempotent pass reads the un-indexed WAL slice, folds it onto the previous
  generation, rebuilds each `memory_type`'s arms, writes an immutable `gen-{G}/` tree, and
  publishes it by `If-Match`-swapping the manifest. Two nodes indexing the same generation
  write to disjoint prefixes and one CAS wins — no locks.
* **Query** — `QueryNode::open` loads the manifest and each type's metadata; `query` probes
  IVF centroids, range-reads only the candidate clusters, runs BM25 over the FTS split, and
  expands graph links, returning each candidate with its **raw per-arm scores** (the client
  fuses — memlake does not). The un-indexed WAL tail is scanned and overlaid so **acked
  writes are visible immediately** (INV-5).

Invariants that hold the design together:

* **Object storage is the only stateful dependency** (INV-1). All coordination is S3
  conditional writes — no locks, no etcd, no Postgres.
* **Every file except the manifest is immutable** (INV-2). Mutation = write new file +
  CAS-swap; a reader sees a whole generation or none of it.
* **Query nodes hold no durable state** (INV-4). A cold node rebuilds from S3; losing a
  node's disk costs latency, never correctness.
* **Query cost is independent of corpus size** (INV-7): a cold query is a statically bounded
  number of roundtrips, verified by test.

### Predictable resources

A query node's local footprint is capped by construction. The disk cache is two-tier with
**independent memory and disk budgets** — memory eviction demotes a block to disk, disk
eviction deletes it — so a long-running node's RAM and NVMe use both stay within their
configured ceilings regardless of workload. Immutable blocks are cached by `(path, byte
range)`, so a warm graph or cluster read costs zero roundtrips.

### Crates

| Crate          | Responsibility |
|----------------|----------------|
| `mlake-core`   | ids, `Memory`/edge records, manifest, WAL format, `TagFilter` (no I/O) |
| `mlake-store`  | instrumented object-store client, CAS, two-tier cache, op/phase metrics |
| `mlake-wal`    | write path (group commit), tail scan, manifest read/swap |
| `mlake-ivf`    | mini-batch k-means centroids, cluster files, probe-then-rerank |
| `mlake-fts`    | tokenizer chain (NFKC/OpenCC/jieba dual-emission) + tantivy BM25 |
| `mlake-graph`  | reverse-adjacency CSR, link-expansion retriever, scorer |
| `mlake-index`  | indexer, generation IO, GC, RRF fusion, `QueryNode` |
| `mlake-server` | gRPC API (`serve`) + indexer loop (`index`) over the crates above |
| `mlake-perf`   | in-process micro-benchmark (library-level; the e2e suite is Python) |
| `mlake-bench`  | BEIR accuracy runner producing per-query rankings |

## Using it

memlake is used as a **service**: a client talks gRPC to `mlake-server` (the drop-in for a
Postgres connection — point it at a k8s `Service` instead of a DB). The Python client:

```python
from memlake_client import MemlakeClient, memory, ANY_STRICT

with MemlakeClient("memlake:50051") as c:
    c.create_namespace("my-bank")

    # Write a batch of memories. Returns once the batch is durable in object storage
    # (a claimed WAL sequence) — not after indexing. `metadata` is opaque str->str: memlake
    # stores and returns it verbatim, never indexes it — stash context, document_id, etc.
    c.write("my-bank", [
        memory("s3-native retrieval engine", vector=embedding,
               memory_type=1, tags=["prod"], key="doc-42",
               metadata={"document_id": "d-42", "chunk_id": "0"}),
    ])

    # ONE query across memory_types (all, if omitted), always running the three arms —
    # dense vector + BM25 full-text + graph. `vector` drives dense + graph, `text` drives
    # full-text. The server runs them concurrently over one snapshot, so the storage reads
    # coalesce into shared roundtrips.
    hits = c.query("my-bank", vector=query_embedding, text="retrieval",
                   tags=["prod"], tags_mode=ANY_STRICT)

    # memlake does NO fusion. Each hit carries the RAW per-arm signals — dense cosine, BM25,
    # graph activation, with each arm's rank — so the caller (e.g. Hindsight) runs its own
    # RRF or weighting. The materialized memory (text, tags, timestamps, metadata) comes back
    # INLINE, so recall needs no second round trip to hydrate. Types are independent: group
    # by memory_type first.
    for h in hits:
        print(h.memory_type, h.id_uuid,
              h.dense.score, h.dense.rank,     # dense arm (present=False if it didn't retrieve h)
              h.text.score,  h.text.rank,      # BM25 arm
              h.graph.score, h.graph.rank,     # graph arm
              h.memory.text, h.memory.metadata)   # the memory, returned inline
```

The contract is [`proto/memlake/v1/memlake.proto`](proto/memlake/v1/memlake.proto); generate a
client for any language from it. See the [Python client](clients/python/README.md) and the
end-to-end demo `uv run --project clients/python clients/python/scripts/smoke.py`.

## Deployment topology

Two Deployments, one image. Everything stateful lives in object storage, so every `serve`
replica is interchangeable — no leader, no per-pod data to lose.

```
     Hindsight (or any client)
             │  gRPC (HTTP/2 + protobuf)
             ▼
   ┌──────── k8s Service ────────┐
   │  serve   serve   serve  ...  │   mlake-server serve   — stateless API, N replicas,
   │  (pod)   (pod)   (pod)       │                          bounded local read cache
   └──────────────┬──────────────┘
                  │ reads + writes (conditional PUT / ranged GET)
                  ▼
          ┌───────────────┐          mlake-server index    — async, idempotent indexer,
          │  S3 / bucket  │ ◀──────     its OWN Deployment (1+ replicas), off the request path
          └───────────────┘
```

| Component | What it is | Depends on |
|---|---|---|
| **Client** | your service (e.g. Hindsight) using the generated gRPC stubs | the `serve` Service endpoint |
| **`serve`** | stateless gRPC API; N replicas behind one k8s Service | **S3 only** (+ a local, bounded read cache) |
| **`index`** | the indexer loop; separate Deployment, idempotent | **S3 only** |
| **S3 / bucket** | the sole source of truth: WAL, generations, manifest | — |

```bash
mlake-server serve --addr 0.0.0.0:50051 --mem-mb 256 --disk-mb 4096   # API pods
mlake-server index --namespaces my-bank --interval-secs 5             # indexer Deployment
```

A write acks only once its WAL object is durable in S3 (a claimed sequence), never waiting on
the indexer; a `STRONG` query still sees it immediately by scanning the WAL tail. gRPC over a
plain L4 Service load-balances poorly (one long-lived HTTP/2 connection) — front the pods with
a headless Service + client-side LB or an L7/gRPC-aware proxy. Full rationale, env vars, and
routing/cache-affinity notes are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Running it

Prerequisites: Rust, Docker (MinIO for real S3 conditional-write semantics), and `uv` for
the accuracy harness.

```bash
docker compose up -d          # MinIO on :9000, bucket `memlake`
cargo test                    # unit + integration, including live MinIO paths
```

### Performance harness

`mlake-perf` generates clustered vectors with Zipfian tags and causal edges, then measures
write throughput, index build time, read latency, roundtrips, and S3-op cost against real
MinIO. Iterate at 10k, then sweep up.

```bash
cargo run --release -p mlake-perf -- write --scale 100000
cargo run --release -p mlake-perf -- read  --scale 100000 --mem-mb 64 --disk-mb 512
cargo run --release -p mlake-perf -- suite --scales 10000,100000,1000000    # write+read each

MEMLAKE_TIMING=1 cargo run --release -p mlake-perf -- write --scale 100000  # per-phase breakdown
```

Latest write/read/accuracy numbers live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### Accuracy comparison (BEIR vs Qdrant)

memlake reaches **accuracy parity with Qdrant hybrid search** on BEIR using identical
`bge-small` embeddings, and its graph arm — which Qdrant has no equivalent of — **wins** on
the denser-relevance corpus. nDCG@10:

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

Full analysis is in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## What's a prototype here

Deliberately deferred (recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md)): the axum HTTP
server over `QueryNode` (the retrieval substance is built; the wrapper is not); the full
differential against live Hindsight Postgres; and quantization, sharding, multi-region, and
auth — all v1 non-goals per the spec. The FTS arm **is tantivy** (SPEC §5.3), packaged the
S3-native way: a whole index packed into one `split.bin` object and materialized into the
local mmap tier to serve reads.
