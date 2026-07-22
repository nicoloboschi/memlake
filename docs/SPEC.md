# SPEC: `pufferlite` — S3-Native Vector + FTS + Graph Search Engine for Memory Items

**Status:** Draft v1 · **Language:** Rust · **License target:** Apache-2.0
**Inspiration:** turbopuffer architecture (object-storage-first LSM), Quickwit split/hotcache packaging, Hindsight link-expansion retrieval.

---

## 1. Purpose & Scope

A prototype search engine where **object storage (S3) is the sole source of truth** and query nodes are **fully stateless caches**. Optimized for **small memory items** (a sentence or two + vector + metadata + links), not documents. Serves three retrieval arms over one storage layer:

1. **Vector search** — IVF/centroid ANN (SPFresh-style, NOT graph-based ANN).
2. **Full-text search** — BM25 via tantivy, script-aware tokenization incl. Chinese.
3. **Link expansion (graph)** — bounded one-hop expansion over precomputed links, ported 1:1 from Hindsight's `LinkExpansionRetriever`.

### Goals
- Any node can serve any namespace; a node dying loses zero data and zero correctness.
- Cold query ≤ **4 S3 roundtrips**; warm query served from NVMe/mmap.
- Strong consistency for item visibility; eventual consistency for derived graph structure.
- Behavioral equivalence with Hindsight's Postgres retrieval (differential-testable).

### Non-Goals (v1)
- Sharding a namespace across multiple indexes.
- Quantization (PQ/SQ) — store raw f32 vectors; leave hooks.
- Multi-region, auth, encryption, billing.
- Recursive / multi-hop graph walks (explicitly forbidden on the cold path — see §7).

---

## 2. Architecture Invariants (MUST hold at all times)

- **INV-1** S3 is the only stateful dependency in the critical path. No etcd, no Postgres, no Redis.
- **INV-2** Every file on S3 except the WAL head and `manifest.json` is **immutable**. Mutation = write new file + CAS-swap manifest.
- **INV-3** All coordination uses S3 conditional writes (`If-None-Match: *` for create, `If-Match: <etag>` for swap). No locks.
- **INV-4** Local disk/memory is a cache keyed by `(namespace, file_path, etag)`. Deleting a node's disk MUST NOT change any query result (only latency).
- **INV-5** A successful write ack ⇒ data durable on S3 ⇒ visible to the next consistent query (via WAL tail scan).
- **INV-6** Indexing and compaction are idempotent: any node may run them; crashing mid-run leaves only unreferenced garbage files.
- **INV-7** Every query has a statically bounded number of S3 roundtrips, independent of data size and graph shape.

Each invariant MUST have at least one automated test (see §10).

---

## 3. Storage Layout

> **Direction:** the `gen-{G}/` monolith below is becoming a stack of immutable **segments**
> (`seg/{seg_id}/`, same file families) — see [segmented-index.md](segmented-index.md) §3. The
> manifest lists the live segments; a flush adds one, compaction merges several.

```
s3://{bucket}/{namespace}/
  manifest.json                  # CAS-swapped; lists the live segments
  wal/
    00000001.bin ... 000000NN.bin
  seg/{seg_id}/                  # one immutable segment (per memory_type subtree)
    pk.idx  pk.data              # sorted item_id -> cluster (last-writer-wins across segments)
    centroids.bin                # this segment's IVF centroids
    cluster-{i}.bin  cluster-{i}.vec   # item records (rkyv) + int8 vector block
    radj.data                    # reverse adjacency (incoming links) SSTable data
    radj.idx                     # sparse offset index for radj.data
    entity.* time.* payload.* rerank.*   # the other range-read SSTables
    fts/split.bin                # packed tantivy segment + hotcache footer
    tombstones.bin  stats.json   # this segment's deletes + doc counts / lengths
```

### 3.1 `manifest.json`
```json
{
  "format_version": 1,
  "generation": 42,
  "wal_index_cursor": 137,        // last WAL seq folded into gen-42
  "wal_head": 141,                // last committed WAL seq at manifest write
  "files": { "pk": "gen-42/pk.idx", "...": "..." },
  "tokenizer_config_hash": "…",   // guards against mixed-tokenizer segments
  "prev_generation": 41           // GC grace: keep prev gen until TTL
}
```
Swapped with `If-Match` on etag. Writers of the manifest MUST re-read + merge on conflict, never blind-retry.

### 3.2 WAL entry (`wal/{seq:08}.bin`)
One entry = one atomic transaction (replaces Postgres per-document transaction). Encoded with rkyv, containing:

```rust
struct WalEntry {
  seq: u64,
  ops: Vec<Op>,
}
enum Op {
  Upsert(Item),          // full item: id, text, vector, metadata, entity_ids,
                         // causal_edges, tags, fact_type, timestamps
  Tombstone { id: Uuid },
  Patch { id: Uuid, deltas: Vec<Delta> },   // commutative only, e.g. ProofCount(+1)
  Guard { expect_seq_lt: u64 },             // optional optimistic precondition
}
```
Rules:
- Writers buffer ops, group-commit ≤ 1 entry/sec/namespace.
- Commit = `PUT wal/{next}.bin` with `If-None-Match: *`; on 412 conflict, refetch head seq and retry.
- **Semantic kNN links are NOT in the WAL** — they are derived data, computed by the indexer (§5.2). Entity ids and causal edges ARE intrinsic and go in the WAL.
- `Patch` deltas must be commutative and fold-able at read time (tail scan) and index time.

### 3.3 Cluster file entry (item record)
```rust
#[derive(Archive /* rkyv */)]
struct StoredItem {
  id: Uuid,
  vector: Vec<f32>,            // f32; dim from namespace config
  text: String,
  fact_type: u8,
  tags: Vec<String>,
  event_date / occurred_start / occurred_end / mentioned_at: Option<i64>,
  proof_count: u32,
  entity_ids: Vec<u64>,        // dictionary-encoded entity ids
  semantic_out: [(Uuid, f16); MAX 5],   // kNN links, sim >= 0.7 (derived, indexer-written)
  causal_out: Vec<(Uuid, LinkType, f16)>, // causes/caused_by/enables/prevents
}
```
Items are small ⇒ full payload lives inline in the cluster file. Fetching seed clusters yields seed adjacency for free (zero extra graph roundtrips for outgoing links). Files are rkyv-archived so they CAN be read zero-copy. Today they are not: the read path validates the buffer and then fully deserializes it into an owned item graph, and the disk cache reads with `fs::read` rather than mmap — so a warm hit still costs a copy plus one allocation per field per member. The scan path no longer pays this (it reads the flat `cluster-{i}.vec` block instead), so the cost now falls only on hydrating the winners. Closing the gap is tracked in TODOS.

### 3.4 Reverse adjacency (`radj.data` + `radj.idx`)
An SSTable over incoming semantic + causal edges, sorted by target id (same `.idx`/`.data` format as
pk/entity/time/payload). In the segmented model an edge is stored in the segment that owns its
**target** id, and graph expansion unions each segment's `radj` (§ [segmented-index.md](segmented-index.md) §5):
- `radj.idx`: sparse index, every Kth target id → byte offset (K sized so idx ≤ 256 KB for 1M items).
- Lookup: binary search idx (cached) → 1 coalesced ranged GET → scan block.

### 3.5 Entity postings
Entities are indexed as a dedicated tantivy field (`entities`, raw tokenizer, one token per entity id) inside the same FTS split — entity expansion IS an inverted-index query (§7). No separate file family.

---

## 4. Write Path

```
client ──HTTP──▶ any node
  buffer ops (per namespace, in-mem)
  group commit tick (≤1/s or 4 MB):
    seq = head+1
    PUT wal/{seq}.bin  If-None-Match:*
    412 ⇒ re-list head, retry (bounded, jittered)
  ack client after successful PUT
```

- Write latency budget: p50 ≤ 250 ms, p99 ≤ 1.5 s (dominated by S3 PUT + batch wait).
- Multi-op atomicity: everything in one `WalEntry` is all-or-nothing to every reader.
- Deletes: tombstone only. Dangling edges are filtered at candidate materialization (PK index knows tombstones); physical reclamation at compaction. Never eagerly rewrite neighbors.
- Read-modify-write is forbidden. Use `Patch` deltas; use `Guard` for optimistic preconditions.

---

## 5. Indexer (async, any node, idempotent)

> **Direction (see [segmented-index.md](segmented-index.md)):** a generation is moving from one
> monolithic file set to an **LSM-style stack of immutable segments**. A *flush* turns the WAL tail
> into a new small L0 segment (O(tail)); a background *tiered compaction* merges segments (the only
> O(corpus) step). The single-generation `index(namespace)` below is the v1 fold; the sections that
> follow describe its build steps, which a per-segment flush reuses.

Runs `index(namespace)`: read manifest, read WAL entries `(wal_index_cursor, wal_head]`, produce `gen-{G+1}`, CAS-swap manifest. Determinism requirement: output depends only on (prev generation files, WAL slice) so replays are byte-stable modulo float tie-breaks.

### 5.1 Vector index build
- Centroid count: `max(1, round(sqrt(N)))` per **segment** (`N` = the segment's item count), retrained
  on the segment's slice at build; a query probes each segment's centroids and merges (§6.1). Segments
  are never appended to — incrementality comes from new segments + compaction, not from mutating an
  existing centroid set (immutable object storage rules out SPFresh-style in-place update).
- Cluster split when size > 8×avg; merge when < ⅛×avg, evaluated per segment / at compaction.
- Cluster file target size 2–8 MB (coalesces into one ranged GET each).

### 5.2 Semantic kNN link derivation
For each new item in the WAL slice: query the *current* index (in-process, warm) for top-5 neighbors with cosine ≥ 0.7; write `semantic_out` into the item's record; append reverse edges into `radj`. Incoming links of pre-existing items update only in `radj` (their cluster records are NOT rewritten) — outgoing-inline + reverse-file covers both directions, mirroring Hindsight's bidirectional check.

### 5.3 FTS build
- Build a tantivy segment for the WAL slice using the tokenizer chain in §8; merge segments per tantivy policy; pack final segment set into `fts/split.bin` with a **hotcache footer**: term-dictionary blocks, per-field metadata, posting offsets required to plan a query without extra roundtrips (Quickwit pattern).
- Custom `Directory` impl (§6.2) is the only storage interface tantivy sees.

### 5.4 Compaction & GC
- **Tiered segment compaction** (target model, [segmented-index.md](segmented-index.md) §7): when a
  level accumulates ≥ `FANOUT` segments, merge them into one segment at the next level — resolving
  last-writer-wins + tombstones and physically reclaiming deleted/shadowed items. The merge is the
  only O(corpus) step and runs the **streaming (external-memory) fold as its engine**, off the write
  path. Small L0 flushes use the fast in-RAM build.
- GC: delete segment prefixes no longer referenced by the manifest, after a ~15 min reader-grace TTL.

---

## 6. Query Node

### 6.1 Cold-path roundtrip plan (hard budget: 4)
```
RT1  GET manifest.json  +  LIST/HEAD wal head            (consistency point)
RT2  parallel: centroids.bin | fts hotcache | pk.idx | radj.idx | wal tail entries
RT3  parallel ranged GETs: selected cluster files | fts posting ranges
RT4  parallel ranged GETs: radj.data blocks | linked-candidate items (via pk.idx)
```
- Every S3 request MUST flow through one instrumented client that tags `(namespace, query_id, roundtrip_no, bytes, latency)`. A query exceeding 4 roundtrips cold is a **bug** and must emit a `roundtrip_budget_exceeded` metric + debug log.
- **Segment fan-out (target model, [segmented-index.md](segmented-index.md) §5):** each arm reads
  across *all live segments* + the WAL tail and merges — vector probes each segment's centroids;
  pk/payload lookups take the newest segment that owns the id (last-writer-wins by seq); tombstones are
  a cross-segment overlay. The roundtrip budget becomes O(levels), which tiered compaction keeps a
  small constant.
- WAL tail (entries past `wal_index_cursor`): exhaustive scan — brute-force cosine for vector arm, linear text match for FTS arm, direct entity/causal membership for graph arm; fold patches; apply tombstones. Reads are strongly consistent: RT1 includes the WAL head check.

### 6.2 Cache
- Unified NVMe cache: content-addressed by `(namespace, path, etag)`, mmap on hit, LRU by bytes with per-namespace accounting. Admission = write-through on first fetch.
- In-mem ARC layer for hotcache footers, centroids, pk/radj idx (small, hot).
- Routing: LB uses rendezvous hashing on namespace as a *preference only*; MUST degrade to any node.

### 6.3 Retrieval arms & fusion
- Vector: nprobe nearest centroids (default 8), exact re-rank on fetched vectors, top-k.
- FTS: BM25 with block-WAND via tantivy over the split.
- Graph: §7.
- Fusion: RRF over the three arms (k=60), then optional cross-fact-type combine using `activation` semantics identical to Hindsight (`result.activation` = additive graph score).

---

## 7. Graph Retrieval (port of Hindsight `LinkExpansionRetriever`)

Reference implementation: `hindsight-api-slim/hindsight_api/engine/search/link_expansion_retrieval.py` + `ops_postgresql.py` CTEs. Port the *behavior*, table-tested (§10.3):

1. **Seeds**: vector search, `limit=20`, `threshold=0.3`, filtered by fact_type / tags / time range.
2. **Entity expansion**: seed items' `entity_ids` (in hand from RT3) → query `entities` field postings, capped at **200 candidates per entity** (bounded posting-prefix read), candidates filtered to fact_type before the cap consumes budget. Score = distinct shared-entity count → `tanh(count * 0.5)`.
3. **Semantic expansion**: union of seeds' `semantic_out` (inline, free) and incoming edges from `radj` (RT4). Score = `max(weight)` per candidate, weights ∈ [0.7, 1.0].
4. **Causal expansion**: same mechanics over causal edge types (`causes|caused_by|enables|prevents`), max weight per candidate.
5. **Merge**: `score = entity + semantic + causal ∈ [0,3]`; exclude seed ids; sort desc; truncate to budget; store as `activation`.
6. **Bounds**: exactly one hop; per-entity cap; global timeout (default 800 ms cold) falls back to semantic+causal only (drop entity arm), mirroring the Postgres timeout fallback.
7. Candidate materialization via `pk.idx` filters tombstones ⇒ dangling edges are invisible without cleanup.

**Forbidden:** recursive expansion, unbounded fan-out, or any traversal whose S3 request count depends on graph shape.

---

## 8. Tokenization (Chinese-capable FTS)

Pipeline (single implementation used by both indexer and query parser; config hash stored in manifest):

1. **Normalize**: Unicode NFKC → OpenCC t2s (traditional→simplified) → lowercase.
2. **Script segmentation**: split text into runs by Unicode script (Latin / Han / Kana / Hangul / digits).
3. **Per-run tokenize**:
   - Latin: whitespace + punctuation split, light stemmer optional (config), keep code identifiers intact.
   - Han: **dual emission** — (a) `jieba-rs` `cut_for_search` word tokens into field `text_words`; (b) character bigrams into field `text_bigrams`.
   - Kana/Hangul: bigrams (v1).
4. **Query side**: same chain; query hits `text_words OR text_bigrams` (and Latin tokens hit both fields identically); BM25 scores combined by tantivy's multi-field query.

Crates: `tantivy`, `jieba-rs`, `opencc-rust` (or `character_converter`), `unicode-script`, `unicode-normalization`. Entity field uses `raw` tokenizer.

---

## 9. Crate Layout & Dependencies

```
pufferlite/
  crates/
    plite-core/        # types, ids, rkyv formats, WalEntry, manifest
    plite-store/       # object_store wrapper, instrumented client, CAS helpers, cache
    plite-wal/         # write path, group commit, tail scan
    plite-ivf/         # centroid train/assign, cluster files, ANN query
    plite-fts/         # tokenizer chain, tantivy Directory over plite-store, split packing
    plite-graph/       # radj CSR build/read, link-expansion retriever, scorer
    plite-index/       # indexer orchestration, compaction, GC
    plite-server/      # axum HTTP API, query planner, fusion; --mode query|indexer|all
    plite-bench/       # benchmark + differential harness (§10)
  testdata/            # fixed corpora (see §10.1)
```
Key deps: `tokio`, `object_store` (S3 + LocalFileSystem + conditional put), `tantivy`, `jieba-rs`, `rkyv`, `axum`, `faiss` or `linfa-clustering`, `criterion`, `proptest`, `metrics` + `metrics-exporter-prometheus`, `tracing`.
Local dev/CI: **MinIO** (supports conditional writes) via docker-compose; unit tests use `LocalFileSystem` with an injected latency shim (see §10.4).

---

## 10. Benchmark-Driven Development (this section gates all milestones)

**Rule: no optimization without a benchmark; no merge that regresses a gate.** Every milestone in §12 ships with its benchmarks green in CI.

### 10.1 Fixed corpora (checked into `testdata/` or generated deterministically, seed=42)
- `mem-10k` / `mem-100k` / `mem-1m`: synthetic memory items, 768-dim vectors, Zipfian entities (vocab 5k, avg 2.5 entities/item), semantic links derivable, 5% causal edges, 30% Chinese / 55% English / 15% mixed text.
- `hindsight-diff`: export of a real Hindsight bank (anonymized) for differential testing.
- `cjk-fts`: labeled Chinese retrieval set (query → relevant item ids) for tokenizer evaluation, incl. OOV product names and traditional-script queries.

### 10.2 Metrics (Prometheus; every metric labeled by namespace + cache state cold|warm)
| Metric | Type |
|---|---|
| `s3_requests_total{op,roundtrip_no}` | counter |
| `s3_request_bytes` / `s3_request_latency_seconds` | histogram |
| `query_roundtrips` (cold path; alert if > 4) | histogram |
| `query_latency_seconds{arm=vector\|fts\|graph\|fused, state}` | histogram |
| `write_commit_latency_seconds`, `wal_batch_ops` | histogram |
| `index_lag_entries` (wal_head − wal_index_cursor) | gauge |
| `index_build_seconds{phase=kmeans\|knn_links\|fts\|radj}` | histogram |
| `cache_hit_ratio{tier=mem\|nvme}`, `cache_bytes` | gauge |
| `cas_conflicts_total{target=wal\|manifest}` | counter |
| `graph_fallback_total` (entity-arm timeouts) | counter |
| `recall_at_k` (from shadow eval job, §10.5) | gauge |

### 10.3 Correctness gates (must pass in CI, `mem-10k` + `hindsight-diff`)
- **G-1 Vector recall**: recall@10 vs brute force ≥ 0.95 (nprobe=8), ≥ 0.99 (nprobe=32).
- **G-2 Graph differential**: link-expansion ranked ids match Hindsight Postgres output on `hindsight-diff` — Jaccard@budget ≥ 0.95 and identical top-5 on ≥ 90% of 500 fixed queries (tolerance for float ties documented per divergence).
- **G-3 Scorer unit tables**: tanh/additive/RRF math table-tested against values captured from the Python implementation (goldens in repo).
- **G-4 CJK FTS**: on `cjk-fts`, dual-field ≥ bigram-only ≥ word-only in MRR@10; dual-field MRR@10 ≥ 0.80; traditional-script queries retrieve simplified items (OpenCC path) with 100% of the labeled pairs.
- **G-5 Consistency**: proptest — arbitrary interleavings of {upsert, delete, patch, index, compact, crash-before-manifest-swap, cache wipe} ⇒ strong-consistency queries always reflect every acked write; INV-1..7 assertions embedded.
- **G-6 Determinism**: indexer replay on same (gen, WAL slice) produces identical manifest content hash.

### 10.4 Performance gates (criterion + end-to-end rig; S3 latency simulated in CI via latency-injecting `ObjectStore` wrapper: GET p50 80 ms / p90 250 ms, PUT p50 120 ms; real-S3 numbers tracked out-of-CI on `mem-1m`)
| Benchmark | Gate (CI, simulated S3, `mem-100k`) | Target (real S3, `mem-1m`) |
|---|---|---|
| Cold fused query p90 | ≤ 4 roundtrips AND ≤ 1.2 s | ≤ 900 ms |
| Warm fused query p50 / p99 | ≤ 15 ms / ≤ 60 ms | ≤ 15 ms / ≤ 75 ms |
| Warm vector arm p50 | ≤ 5 ms | ≤ 8 ms |
| Warm graph arm p50 (20 seeds) | ≤ 6 ms | ≤ 10 ms |
| Write ack p50 / p99 | ≤ 250 ms / ≤ 1.5 s | same |
| Index throughput | ≥ 5k items/s/core (excl. embedding) | same |
| Indexer kNN-link phase | ≥ 10k links/s | same |
| Cold-start after cache wipe: 2nd query warm | true | true |
| Micro: rkyv item access | ≤ 200 ns/item (zero-copy proof) | — |

Criterion benches live in `plite-bench`; results exported to JSON and compared against the last main-branch baseline; regression > 10% on any gated bench fails CI.

### 10.5 Continuous eval
`plite-bench shadow` job: replays a fixed query log against a live namespace nightly, computes recall/MRR/latency, pushes `recall_at_k` gauges, writes a markdown report artifact. Any gate drift opens an issue automatically.

---

## 11. HTTP API (v1, minimal)

```
POST /v1/ns/{ns}/write      {upserts:[Item], deletes:[id], patches:[…]} → {seq}
POST /v1/ns/{ns}/query      {vector?, text?, graph?:{…}, filters, top_k,
                             consistency: strong|eventual} → {results:[{id,scores,activation,…}]}
GET  /v1/ns/{ns}/stats      manifest + lag + cache info
POST /v1/ns/{ns}/warm       pre-flight cache warm (tpuf-style)
POST /v1/admin/index/{ns}   force index run   POST /v1/admin/compact/{ns}
```

---

## 12. Milestones (each = code + tests + benchmarks green)

1. **M1 Storage core**: `plite-store` (instrumented client, CAS, cache), `plite-wal` write path + tail scan. Gates: G-5 subset (write/visibility), write-latency bench, INV tests.
2. **M2 Vector**: IVF build + query + WAL-tail merge. Gates: G-1, cold roundtrip ≤ 3 for pure-vector query, warm vector bench.
3. **M3 FTS**: tokenizer chain, tantivy Directory, split packing + hotcache. Gates: G-4, FTS roundtrip budget, warm FTS bench.
4. **M4 Graph**: inline links, kNN derivation in indexer, radj CSR, retriever + scorer port. Gates: G-2, G-3, warm graph bench.
5. **M5 Fusion + server**: RRF, planner enforcing RT budget, API. Gates: full cold/warm fused benches, `query_roundtrips` alerting.
6. **M6 Compaction/GC + chaos**: tombstone folding, patch folding, GC, kill-and-restart chaos test. Gates: G-5 full, G-6, cold-start bench.
7. **M7 Hardening**: real-S3 `mem-1m` run, shadow eval job, tuning (nprobe, cluster size, cache sizes) documented as benchmark deltas.

## 13. Open Questions (resolve before or during M4)
- Does the Hindsight consolidation pipeline assume semantic links exist at insert time? If yes, define a `links_ready(seq)` readiness signal or WAL-direct read for consolidation.
- Observation fact-type expansion (source_memory_ids traversal) — in scope for the prototype or deferred? If in scope, model `observation_sources` as an additional postings field.
- Entity id dictionary: global per namespace (simpler) vs per generation (smaller); v1 = per namespace, rebuilt at compaction.
