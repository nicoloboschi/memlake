# memlake-bench

Benchmark harness for [memlake](ARCHITECTURE.md). Two jobs:

1. Establish a **Qdrant hybrid-search baseline** (dense + BM25 sparse, RRF-fused) on BEIR.
2. Later, evaluate **memlake** against that baseline on *identical data and identical
   embeddings*, so any difference is attributable to the engine and not to the model.

Point 2 is why the embedding cache format below is a hard contract.

---

## Quick start

Everything runs through `uv` — never activate a venv manually.

```bash
# one dataset, end to end (download -> embed -> exact -> qdrant -> report)
uv run --project bench memlake-bench all scifact

# or step by step
uv run --project bench memlake-bench download scifact
uv run --project bench memlake-bench embed scifact
uv run --project bench memlake-bench baseline exact scifact
uv run --project bench memlake-bench baseline qdrant scifact
uv run --project bench memlake-bench report
```

Results land in `bench/results/{dataset}/{engine}.json` (committed — they're small)
and are rendered to [`bench/results/report.md`](results/report.md).

### Datasets

| name | docs | queries (test) | notes |
|---|--:|--:|---|
| `scifact` | 5,183 | 300 | default smoke-test dataset |
| `nfcorpus` | 3,633 | 323 | small, many relevant docs per query |
| `fiqa` | 57,638 | 648 | ~10x bigger; embedding takes a while |
| `arguana`, `scidocs`, `trec-covid` | — | — | supported, not routinely run |

Corpora are cached under `testdata/beir/` and are gitignored.

---

## Embedding cache format (the contract)

`embed` is the expensive step and the **single source of truth for vectors**. Every
engine — numpy, Qdrant, and the future Rust engine — reads these exact arrays. Nothing
re-embeds at query time.

Layout, `testdata/embeddings/{dataset}/`:

```
meta.json          model / dim / counts / normalization / query prefix
corpus.npy         float32, shape (n_docs, dim),     C-order, L2-normalized
corpus_ids.json    list[str], length n_docs
queries.npy        float32, shape (n_queries, dim),  C-order, L2-normalized
queries_ids.json   list[str], length n_queries
```

Guarantees:

- **Row order is the contract.** `corpus.npy[i]` belongs to document
  `corpus_ids[i]`; same for queries. Qdrant point ids are the same row index `i`,
  so ids round-trip through every engine unchanged.
- **`float32`, C-contiguous, no pickled objects.** Plain NumPy v1 `.npy` files —
  readable from Rust via the `ndarray-npy` crate, or by parsing the 128-byte
  header and `mmap`-ing the tail directly.
- **L2-normalized at write time.** Cosine similarity == dot product; no engine
  needs to renormalize, and no engine may apply its own normalization.
- **Model**: `BAAI/bge-small-en-v1.5` (384-dim) via `fastembed`.
- **Query prefix**: bge retrieval models are asymmetric, so queries are embedded
  with `"Represent this sentence for searching relevant passages: "` prepended.
  This is recorded in `meta.json` as `query_prefix`. Dropping it costs several
  nDCG points — a Rust engine embedding its own queries must apply it too.
- **Document text**: `title + " " + text` (BEIR convention), recorded in `meta.json`.

`embed` is idempotent: it skips when a cache with a matching model and document
count exists. Use `--force` to rebuild.

Reading the cache from Rust:

```rust
// ndarray-npy = "0.8"
let corpus: ndarray::Array2<f32> =
    ndarray_npy::read_npy("testdata/embeddings/scifact/corpus.npy")?;
let ids: Vec<String> =
    serde_json::from_reader(std::fs::File::open(".../corpus_ids.json")?)?;
assert_eq!(corpus.nrows(), ids.len());
```

---

## Engines

### `baseline exact` — reference ceiling

Pure numpy brute-force cosine kNN (no ANN, so it is the exact dense ceiling) plus a
dependency-free Okapi BM25 (`k1=0.9`, `b=0.4`, stopwords + light plural stemmer),
fused with RRF (`k=60`). Runs entirely in-process.

Its purpose is to **validate the harness independently of Qdrant**. If Qdrant's dense
arm drifts materially from this, Qdrant is misconfigured — not "ANN recall loss".

### `baseline qdrant` — the real baseline

- Dense: Qdrant HNSW, cosine, vectors loaded from the embedding cache.
- Sparse: fastembed `Qdrant/bm25` sparse vectors; **IDF is applied server-side**
  via `Modifier.IDF` on the sparse vector config (the model emits raw term
  frequencies, so omitting the modifier silently degrades the sparse arm).
- Fusion: Qdrant-native `FusionQuery(RRF)` over both prefetch arms, prefetch
  limit `2 * top_k`.

Each of the three arms (dense / sparse / hybrid) is queried and scored separately.

---

## Qdrant container lifecycle

`baseline qdrant` handles this for you: it checks whether Qdrant is reachable, and
starts one via docker compose if not.

```bash
docker compose -f bench/docker-compose.qdrant.yml up -d
docker compose -f bench/docker-compose.qdrant.yml down -v
```

Port selection:

1. `QDRANT_URL` if set — used as-is, and the run fails fast if unreachable.
2. Otherwise `QDRANT_HTTP_PORT` (default `6333`). If something is already serving
   Qdrant there, it is reused as-is.
3. If the port is occupied by something that isn't Qdrant, the harness picks the
   first free port from `6343` upward and starts its own container there.

This compose file is deliberately separate from the repo-root `docker-compose.yml`
(MinIO / engine services) to avoid edit conflicts.

---

## Metrics

Computed in `metrics.py` against BEIR qrels, following pytrec_eval conventions so
numbers are comparable to published leaderboards:

- **nDCG@10** — graded relevance, ideal DCG from the full qrel list truncated at k.
- **Recall@100** — binary relevance (`rel > 0`).
- **MRR@10** — reciprocal rank of the first relevant doc.
- **Latency** — mean / p50 / p90 / p99 / max, single-query and sequential.

Only queries present in the split's qrels are evaluated (BEIR convention — e.g.
scifact test is 300 of 1,109 queries).

Latency caveat: `exact` runs in-process while `qdrant` numbers include HTTP
round-trip and JSON serialization. Compare quality metrics across engines freely;
compare latency only within an engine.

---

## Sanity ranges

If a run lands far outside these, something is wrong — debug before trusting it.

| dataset | arm | expected nDCG@10 |
|---|---|--:|
| scifact | dense (bge-small) | 0.65 – 0.72 |
| scifact | sparse (BM25) | ~0.66 (Anserini reference) |
| scifact | hybrid | ≥ dense |
| nfcorpus | dense (bge-small) | 0.30 – 0.35 |
| nfcorpus | sparse (BM25) | ~0.32 |

---

## Layout

```
bench/
  pyproject.toml               uv project, console script `memlake-bench`
  docker-compose.qdrant.yml    Qdrant service for the harness
  results/{dataset}/*.json     per-engine metrics (committed)
  results/report.md            rendered comparison table (committed)
  src/memlake_bench/
    cli.py                     argparse entrypoint
    paths.py                   canonical on-disk locations
    datasets.py                BEIR download + load
    embed.py                   embedding cache (source of truth)
    bm25.py                    dependency-free Okapi BM25
    metrics.py                 nDCG / Recall / MRR / latency / RRF
    results.py                 JSON persistence
    report.py                  markdown renderer
    qdrant_docker.py           container lifecycle + port selection
    engines/exact.py           numpy + BM25 + RRF reference
    engines/qdrant_engine.py   Qdrant hybrid baseline
```

Caches (`testdata/beir/`, `testdata/embeddings/`) are gitignored and fully
reproducible from the CLI.
