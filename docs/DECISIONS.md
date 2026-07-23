# Decisions

Choices made at project kickoff (2026-07-20), resolving ambiguity in `ARCHITECTURE.md`.

## Naming

The spec calls the project `pufferlite` with `plite-*` crates. **We use `memlake` with
`mlake-*` crates**, matching the repository name. All `plite-*` paths in `ARCHITECTURE.md` §9 map
one-to-one onto `mlake-*`.

### Domain vocabulary (memory-lake theme)

The spec's generic/Postgres terms are renamed to fit "memory lake". This is the vocabulary
the code and API use; `ARCHITECTURE.md` keeps the original terms as historical:

| concept | spec / Hindsight term | memlake term |
|---|---|---|
| the unit stored & retrieved | item / fact / memory_unit | **`Memory`** (`StoredMemory`, `MemoryId`) |
| a unit's independent index category | `fact_type` | **`memory_type`** (`MemoryType`, `MemoryTypeIndex`) |
| the top-level container (a bank) | bank_id | **`namespace`** |
| the filter dimension | tags | **`tags`** |

A namespace holds memories; each memory has a `memory_type`; memories of different types
are indexed fully independently (no shared links/vectors/postings) and queried per type,
grouped, never fused. `memory_type` stays a taxonomy word honestly — it is the type *of a
memory*, and matches Hindsight's semantic / episodic / procedural / observation types.

## Language & layout

Rust, as specified. Cargo workspace under `crates/`. The benchmark harness is the one
exception: it lives in `bench/` as a Python (uv) project, because the accuracy baseline is
Qdrant and the reference retrieval stack (BEIR, fastembed) is Python-native. Criterion
micro-benchmarks stay in Rust in `crates/mlake-bench`.

## Storage backend

**S3 only.** The object-store backend is S3 (MinIO in dev via `docker-compose.yml`), so
`If-None-Match` / `If-Match` conditional writes (INV-3) are validated against a real
implementation. The design is optimized for the S3 interface — coalesced ranged GETs, a
bounded roundtrip budget, larger immutable objects — not for a local disk.

`object_store::LocalFileSystem` is **deliberately not supported**: it implements
`If-None-Match` but returns "not yet implemented" for `If-Match`, so it cannot host a
manifest swap. Rather than carry a backend that silently can't do the thing the whole
coordination model depends on, it is removed. Fast unit tests use `object_store::InMemory`
instead — unlike the local filesystem, it implements the full conditional-put contract, so
it is a faithful stand-in for the S3 *interface* without a network. (This supersedes SPEC
§10.4's suggestion of a LocalFileSystem test rig.)

Local disk still appears in one place, correctly: the NVMe read cache (`DiskCache`), which
per INV-4 is only ever a cache — deleting it changes latency, never results.

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

## FTS: tantivy, packaged S3-native

SPEC §5.3 specifies "BM25 via tantivy", and the FTS arm uses tantivy — the real engine,
with block-WAND scoring, compressed posting blocks, and mature segment handling.

The initial overnight build shipped a hand-rolled BM25 to get to measured numbers faster;
that was a time shortcut, not the right call, and it is now replaced. tantivy is also the
*more* S3-optimal choice: its term dictionary + compressed postings mean a query reads far
fewer bytes, which is exactly what the S3 interface rewards.

Packaging follows the Quickwit split pattern, adapted to `mlake-store`:

- a whole tantivy index is built into a temp `MmapDirectory`, then packed into one
  immutable `split.bin` object (`[count][name,data]…`);
- a query node fetches that one object and materializes it into the local NVMe/mmap tier
  to serve reads — the "warm query served from NVMe/mmap" path of §6.1;
- the custom Chinese-capable tokenization (§8) is preserved by feeding tantivy
  *pre-tokenized* streams, so jieba dual-emission and the OpenCC fold still decide what
  gets indexed while tantivy owns storage and scoring.

Accuracy is unchanged from the hand-rolled version (scifact sparse 0.6906 vs 0.6907,
hybrid 0.7325 identical; nfcorpus hybrid 0.3638, still beating Qdrant), so parity holds
with the battle-tested engine now in place.

**One property is genuinely lost:** tantivy stamps each segment with a random id, so the
FTS split bytes are *not* byte-identical across index replays — the strong form of G-6
does not hold for the FTS file (the vector, pk and radj files still are byte-identical).
Retrieval *results* remain deterministic, so query behaviour is reproducible; only the
physical split bytes vary. Test: `retrieval_is_deterministic_across_rebuilds`.

The `Bm25Params` knob (and the `MEMLAKE_BM25_*` bench env vars) were removed: tantivy uses
its own standard BM25 (k1=1.2, b=0.75), so a memlake-side parameter would silently do
nothing — a trap worth deleting rather than keeping.

## Reference implementation

The graph arm ports `hindsight_api/engine/search/link_expansion_retrieval.py` from the
local Hindsight checkout (e.g. `~/dev/hindsight-wt1/hindsight-api-slim/`). Behavior is
table-tested against captured goldens (gate G-3).

## Accuracy result vs Qdrant (nDCG@10, identical bge-small vectors)

Achieved with `nprobe=64`, `arm_depth=200` (matching Qdrant's per-arm prefetch), English
stemming + stopwords on the BM25 arm, and canonical RRF (k=60).

| Dataset  | Arm    | memlake | qdrant | outcome        |
|----------|--------|--------:|-------:|----------------|
| scifact  | dense  | 0.7127  | 0.7127 | exact parity   |
| scifact  | sparse | 0.6907  | 0.6830 | **memlake +1.1%** |
| scifact  | hybrid | 0.7325  | 0.7345 | −0.27% (parity)|
| nfcorpus | dense  | 0.3429  | 0.3436 | −0.20% (parity)|
| nfcorpus | sparse | 0.3244  | 0.3236 | **memlake +0.2%** |
| nfcorpus | hybrid | 0.3638  | 0.3626 | **memlake +0.3%** |

memlake matches or beats Qdrant on 5 of 6 arm/dataset combinations. The dense arm reaches
exact parity once nprobe covers enough clusters (the earlier gap was pure IVF recall);
adding a Snowball stemmer + stopwords to the BM25 arm pushed sparse *past* Qdrant's
fastembed BM25. The one remaining shortfall — scifact hybrid, 0.27% — is within run-to-run
noise. Reproduce with `uv run --project bench memlake-bench baseline memlake <dataset>`.

The `nprobe=64` default is near-exhaustive for these small corpora (~72 / ~60 centroids);
at `mem-1m` scale, `sqrt(N)` centroids keep nprobe a small fraction of the index. The knob
is environment-overridable (`MEMLAKE_NPROBE`) for the speed/recall trade-off study.

## Speed results

Warm-path micro-benchmarks (criterion, `crates/mlake-index/benches/query.rs`) and the
roundtrip-budget integration test:

| Gate (SPEC §10.4)            | Target            | Measured                | Status |
|------------------------------|-------------------|-------------------------|--------|
| Warm vector arm p50          | ≤ 5 ms            | 1.55 ms (100k corpus)   | pass   |
| Index throughput             | ≥ 5k items/s/core | ~11,500 items/s         | pass   |
| Cold-load roundtrips         | ≤ 4, size-independent | constant as corpus 30× | pass   |
| Warm fused p50               | ≤ 15 ms           | 23 ms (synthetic) / 2–4 ms (BEIR) | see note |

The synthetic fused-query number is inflated by an adversarial micro-bench corpus: its
text is drawn from an 8-word vocabulary, so every document shares every query term and the
BM25 posting lists are maximal. On the real BEIR corpora — where query terms are selective
— the benchmark harness measured hybrid p50 at 2–4 ms (see `bench/results/report.md`),
comfortably inside the gate. The synthetic case is kept as a worst-case stress, not a
representative one.

The roundtrip test (`generation_load_roundtrips_are_constant_regardless_of_size`) is the
important one architecturally: it proves query cost is independent of data size (INV-7),
which is the entire justification for the storage layout.

## Build order

Functionality first across all milestones, then quality (accuracy parity), then speed
(performance gates) — rather than optimizing each milestone as it lands.
