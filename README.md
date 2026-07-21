# memlake

An S3-native vector + full-text + graph search engine for small memory items, built as a
prototype from [`docs/SPEC.md`](docs/SPEC.md). Object storage is the sole source of truth;
query nodes are stateless caches. Three retrieval arms — IVF vector search, BM25 full-text
(Chinese-capable), and bounded graph link-expansion — fuse over one storage layer.

## Headline result

memlake reaches **accuracy parity with Qdrant hybrid search** on BEIR, using identical
`bge-small` embeddings, and its graph arm — which Qdrant has no equivalent of — **beats
Qdrant** on the denser-relevance corpus. nDCG@10:

| Dataset  | Arm    | memlake | qdrant | |
|----------|--------|--------:|-------:|-|
| scifact  | dense  | 0.7127  | 0.7127 | parity |
| scifact  | sparse | 0.6907  | 0.6830 | **memlake wins** |
| scifact  | hybrid | 0.7325  | 0.7345 | parity (−0.3%) |
| nfcorpus | dense  | 0.3429  | 0.3436 | parity |
| nfcorpus | sparse | 0.3244  | 0.3236 | **memlake wins** |
| nfcorpus | hybrid | 0.3638  | 0.3626 | **memlake wins** |
| nfcorpus | +graph | 0.3645  | 0.3626 | **memlake wins** (and R@100 0.3304 > 0.3165) |

Full analysis and the speed numbers are in [`docs/DECISIONS.md`](docs/DECISIONS.md); the
live comparison table is [`bench/results/report.md`](bench/results/report.md).

## Architecture

```
        write path                     index path                    query path
  client → any node               any node (idempotent)          any node (stateless)
     │ buffer + group commit         read WAL slice                 read manifest (RT1)
     │ PUT wal/{seq}.bin             fold → build arms              load generation (RT2/3)
     │  (If-None-Match: *)           write gen-{G}/ files           merge WAL tail (RT4)
     ▼                               CAS-swap manifest              fuse arms → results
   S3  ◀──────────── the only stateful dependency (INV-1) ────────────▶  S3
```

* **Object storage is the only stateful dependency** (INV-1). All coordination is S3
  conditional writes — a WAL sequence is claimed with `If-None-Match`, a generation is
  published by `If-Match`-swapping the manifest. No locks, no etcd, no Postgres.
* **Every file except the manifest is immutable** (INV-2). Mutation = write new file +
  CAS-swap. A reader sees a whole generation or none of it.
* **Query nodes hold no durable state** (INV-4). A freshly started node loads everything
  from S3; losing a node's disk costs latency, never correctness.
* **Acked writes are immediately visible** (INV-5) via the WAL tail scan, without waiting
  for the async indexer.
* **Query cost is independent of data size** (INV-7): a cold query is a statically bounded
  number of roundtrips, verified by test.

### Crates

| Crate          | Responsibility |
|----------------|----------------|
| `mlake-core`   | ids, item/edge records, manifest, WAL format (no I/O) |
| `mlake-store`  | instrumented object-store client, CAS, etag-keyed cache, latency shim |
| `mlake-wal`    | write path (group commit), tail scan, manifest read/swap |
| `mlake-ivf`    | k-means centroids, cluster files, probe-then-exact-rerank |
| `mlake-fts`    | tokenizer chain (NFKC/OpenCC/jieba dual-emission) + BM25 |
| `mlake-graph`  | reverse-adjacency CSR, link-expansion retriever, scorer |
| `mlake-index`  | indexer, generation IO, GC, RRF fusion, query node |
| `mlake-bench`  | BEIR runner producing per-query rankings for the harness |

## Running it

Prerequisites: Rust, `uv`, Docker (for MinIO + Qdrant).

```bash
# Bring up MinIO (real S3 conditional-write semantics)
docker compose up -d

# Full test suite — unit + integration, including live MinIO paths
cargo test

# Warm-latency and index-throughput micro-benchmarks
cargo bench -p mlake-index
```

### Reproducing the accuracy comparison

The benchmark harness ([`bench/`](bench/README.md)) is a `uv` project. It embeds BEIR
corpora once (the vector cache is shared by every engine, so the comparison isolates
retrieval), runs each engine, and scores every one with the same metric code.

```bash
# Download → embed → exact + qdrant baselines → memlake → report
uv run --project bench memlake-bench all scifact
uv run --project bench memlake-bench all nfcorpus

# memlake with the graph arm (synthesizes kNN links, adds link-expansion to fusion)
uv run --project bench memlake-bench baseline memlake nfcorpus --graph

uv run --project bench memlake-bench report   # renders bench/results/report.md
```

## What's a prototype here

Deliberately deferred (recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md)):

* **Warm-path per-cluster lazy loading** — the query node currently loads a whole
  generation; the roundtrip *budget* is honoured and verified, but the strict
  per-probed-cluster ranged-GET path is the next refinement.
* **The axum HTTP server** — `Engine::query` and the query node are the substance of the
  API; the HTTP wrapper over them is not built.
* **The full G-2 differential** against live Hindsight Postgres — the graph arm is a
  behavioural port validated arm-by-arm and against the spec's scorer goldens (G-3).
* **Quantization, sharding, multi-region, auth** — all v1 non-goals per the spec.

The FTS arm **is tantivy** (SPEC §5.3), packaged the S3-native way: a whole tantivy index
is packed into one `split.bin` object and materialized into the local NVMe/mmap tier to
serve reads. The Chinese-capable tokenizer chain (§8) drives what gets indexed via
pre-tokenized streams. See `docs/DECISIONS.md`.
