# memlake

An S3-native retrieval engine for small memory records. One storage layer serves three
retrieval arms — **IVF vector search**, **BM25 full-text** (Chinese-capable), and **bounded
graph link-expansion** — fused into a single ranking. Object storage is the sole source of
truth; query nodes are stateless caches that can be thrown away and rebuilt from S3.

Built as a prototype from [`docs/SPEC.md`](docs/SPEC.md).

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
  IVF centroids, range-reads only the candidate clusters, runs BM25 over the FTS split,
  expands graph links, and fuses the arms with weighted RRF. The un-indexed WAL tail is
  scanned and overlaid so **acked writes are visible immediately** (INV-5).

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
| `mlake-perf`   | data generator + write/read performance harness (real MinIO) |
| `mlake-bench`  | BEIR accuracy runner producing per-query rankings |

## Using it

```rust
use mlake_wal::{Namespace, Writer};
use mlake_index::{index, Consistency, IndexOptions, QueryConfig, QueryNode};
use mlake_fts::Tokenizer;
use mlake_core::{Memory, Op, TagFilter, TagsMatch};

// A namespace is a bank on the shared store.
let ns = Namespace::new("my-bank", store);
ns.create_if_absent(&Tokenizer::default().config_hash()).await?;

// Write: group-commit a batch of memories to the WAL.
let mut writer = Writer::new(ns.clone());
writer.commit(memories.into_iter().map(Op::Upsert).collect()).await?;

// Index: fold the WAL tail into a new immutable generation (async, idempotent).
index(&ns, &Tokenizer::default(), IndexOptions::default()).await?;

// Query: open a stateless snapshot, then ask one memory_type at a time.
let node = QueryNode::open(&ns, Tokenizer::default(), Consistency::Strong).await?;
let hits = node.query(
    /* memory_type */ 1,
    Some(&query_vector),                 // vector arm (or None)
    Some("s3 native retrieval"),         // FTS arm (or None)
    &TagFilter::new(vec!["prod".into()], TagsMatch::AnyStrict),
    /* top_k */ 10,
    QueryConfig::default(),              // per-arm RRF weights, nprobe, arm_depth
).await?;
```

Set `Consistency::Eventual` to skip the WAL-head check and serve from the cached manifest;
set a `QueryConfig` arm weight to `0.0` to drop an arm from fusion.

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
