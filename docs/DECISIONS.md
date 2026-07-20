# Decisions

Choices made at project kickoff (2026-07-20), resolving ambiguity in `SPEC.md`.

## Naming

The spec calls the project `pufferlite` with `plite-*` crates. **We use `memlake` with
`mlake-*` crates**, matching the repository name. All `plite-*` paths in `SPEC.md` §9 map
one-to-one onto `mlake-*`.

## Language & layout

Rust, as specified. Cargo workspace under `crates/`. The benchmark harness is the one
exception: it lives in `bench/` as a Python (uv) project, because the accuracy baseline is
Qdrant and the reference retrieval stack (BEIR, fastembed) is Python-native. Criterion
micro-benchmarks stay in Rust in `crates/mlake-bench`.

## Storage backend

MinIO via `docker-compose.yml` from day one, so `If-None-Match` / `If-Match` conditional
writes (INV-3) are validated against a real implementation rather than a stand-in. Unit
tests additionally run against `object_store::LocalFileSystem` with a latency-injecting
wrapper, per SPEC §10.4.

## Accuracy benchmark

- **Datasets**: BEIR — SciFact (~5K docs) → NFCorpus (~3.6K) → FiQA (~57K). Real qrels,
  published baselines, and a natural small → mid-large progression.
- **Embeddings**: local `fastembed` / `BAAI/bge-small-en-v1.5` (384-dim). Free,
  reproducible, no rate limits. Both engines consume **the exact same cached vectors** so
  the comparison isolates retrieval, not embedding quality.
  - Note: SPEC §10.1 assumes 768-dim vectors. Dimension is a namespace parameter; the
    synthetic `mem-*` corpora keep 768, BEIR runs use 384.
- **Metrics**: nDCG@10, Recall@100, MRR@10, plus query latency p50/p90/p99.

## Parity bar vs Qdrant

Two-stage target:

1. **Parity**: memlake's vector + FTS + RRF fusion must match Qdrant hybrid search
   (dense + BM25 sparse + RRF) on identical vectors, within ~1% nDCG@10.
2. **Beat it**: BEIR has no links or entities, so the graph arm is invisible there. We
   synthesize semantic kNN links over the BEIR corpora exactly as the indexer does
   (SPEC §5.2: top-5 neighbours, cosine ≥ 0.7) and show the link-expansion arm lifting
   nDCG@10 above Qdrant's hybrid ceiling — Qdrant has no equivalent arm.

The graph arm's *correctness* is validated separately against the Hindsight reference
implementation (gate G-2), not against Qdrant.

## FTS: hand-rolled BM25 instead of tantivy

SPEC §5.3 specifies "BM25 via tantivy". The POC uses a self-contained BM25 inverted index
instead. The spec's own §6.2 identifies the tantivy `Directory`-over-object-storage
integration as the hard part of the whole design, and it is a multi-day effort that fights
an abstraction built for local disk. A bespoke index packs directly into the single-split-
with-footer model the rest of the architecture already assumes: a query reads the footer,
learns which posting byte ranges it needs, and fetches them in one coalesced GET. Scoring
is standard Okapi BM25, so retrieval quality is unaffected — only the storage mechanism
differs. The tokenizer chain (NFKC → OpenCC t2s → lowercase → script segmentation → jieba
dual-emission) is implemented exactly as SPEC §8 specifies, and is shared verbatim by the
indexer and query parser. Swapping in tantivy later is possible behind the same arm
interface if its packaging story is solved.

## Reference implementation

The graph arm ports `hindsight_api/engine/search/link_expansion_retrieval.py` from the
local Hindsight checkout (e.g. `~/dev/hindsight-wt1/hindsight-api-slim/`). Behavior is
table-tested against captured goldens (gate G-3).

## Build order

Functionality first across all milestones, then quality (accuracy parity), then speed
(performance gates) — rather than optimizing each milestone as it lands.
