# Architecture

memlake is an **S3-native retrieval engine for memory items** — small records (a
sentence or two + an embedding + metadata + links), not documents. Object storage is the
**sole source of truth**; query nodes are **stateless caches**. It serves four retrieval
arms over one storage layer — [vector](arms/vector.md), [full-text](arms/text.md),
[graph](arms/graph.md), [temporal](arms/temporal.md) — and returns each arm's *raw*
per-candidate signal, leaving fusion to the caller.

This document is the cross-cutting map: the invariants, the on-disk layout, and the write /
read / index / compaction paths. The per-arm mechanics live in the four arm docs; how to
configure and deploy it live in [CONFIGURATION.md](CONFIGURATION.md) and
[DEPLOYMENT.md](DEPLOYMENT.md).

---

## 1. Invariants

Everything else follows from these. Each has at least one automated test.

- **INV-1 — S3 is the only stateful dependency.** No etcd, Postgres, Redis, or lock
  service in the critical path. A namespace is a set of objects under one prefix.
- **INV-2 — every object except the WAL head and `manifest.json` is immutable.** A
  mutation is *write a new object + CAS-swap the manifest*, never an in-place edit.
- **INV-3 — all coordination is S3 conditional writes.** `If-None-Match: *` to create
  (WAL append), `If-Match: <etag>` to swap the manifest. No locks; a lost race re-reads
  and retries.
- **INV-4 — local disk/RAM is a pure cache** keyed by `(namespace, path, etag)`. Deleting a
  node's cache changes latency, never a result.
- **INV-5 — a write ack ⇒ durable on S3 ⇒ visible to the next consistent query.** Visibility
  comes from the WAL-tail scan (§6), so a memory is searchable the instant its commit PUT
  returns, before any fold.
- **INV-6 — indexing and compaction are idempotent and coordination-free.** Any node may
  run them; two nodes racing from the same input produce *equivalent* output, so the
  manifest CAS is safe to lose. A crash mid-run leaves only unreferenced objects, which GC
  reclaims.
- **INV-7 — every query has a statically bounded number of S3 roundtrips**, independent of
  corpus size and graph shape. The segmented index makes this `O(number of segments)`, and
  compaction keeps the segment count a small constant.

Determinism (**G-6**): a replay over the same inputs produces byte-identical vector/pk/radj
files. The tantivy FTS split is the one exception — it stamps random segment ids — but its
*retrieval results* are identical.

---

## 2. Storage layout

```
s3://{bucket}/{namespace}/
  manifest.json                 # the only mutable object; CAS-swapped; lists live segments
  wal/
    00000000000000000000.bin …  # one object per commit, key = seq zero-padded to 20 digits
  seg-{seg_id}/                 # one immutable segment (seg_id = uuid nonce)
    mt{type}/                   # a segment holds an independent index per memory_type
      centroids.bin             # this segment's IVF centroids
      cluster-{i}.bin           # item records (rkyv StoredMemory, embedding stripped)
      cluster-{i}.vec           # int8/binary/f32 vector block for cluster i (+ tag/updated columns)
      pk.idx  pk.data           # id -> cluster (sorted SSTable)
      payload.idx  payload.data # id -> full record (embedding stripped)
      rerank.idx  rerank.data   # id -> exact f32 vector (RaBitQ stage-2 rerank)
      entity.idx  entity.data   # entity_id -> memory ids (postings)
      time.idx  time.data       # timestamp -> memory ids
      radj.idx  radj.data       # target id -> incoming edges (reverse adjacency)
      fts/…                     # packed tantivy split
    tombstones.bin              # this segment's supersede overlay (see §4)
```

The WAL key is padded to 20 digits — `u64::MAX`'s width — so lexicographic listing order
equals numeric sequence order for the whole `u64` range (a narrower pad silently reorders
the WAL once a sequence grows a digit).

### The manifest

```jsonc
{
  "format_version": 5,
  "version": 88,                 // monotonic; the CAS identity
  "wal_index_cursor": 137,       // last WAL seq folded into a segment
  "wal_head": 141,               // last committed WAL seq at manifest write
  "segments":       [ … ],       // the live stack, newest-first
  "prev_segments":  [ … ],       // kept one grace window for readers; GC-able after TTL
  "tokenizer_config_hash": "…"   // guards a query against a differently-tokenized split
}
```

Each `Segment` carries `{ id, level, seq_lo, seq_hi, doc_count, indexes: {memory_type -> FactTypeIndex}, tombstones }`.
`seq_lo`/`seq_hi` are the WAL range it covers; `seq_hi` is also its **age stamp** for the
supersede overlay. There is one `manifest.json`, always CAS-swapped; a writer that loses the
swap re-reads and merges rather than blind-retrying.

### The unit of storage

A `StoredMemory` (rkyv, read without a serde round-trip) is small enough to live inline:
id, embedding, text (+ optional `index_text` for FTS), `memory_type`, tags, five timestamps
(four *content* times + `updated_at` *write* time), `proof_count`, dictionary-encoded
`entity_ids`, up to five derived `semantic_out` kNN edges, client-supplied `causal_out`
edges, an opaque `metadata` bag, and `write_seq` (the sequence of its last upsert — the
key to the supersede overlay). The embedding is ~84% of the record, so it is **split out**:
the cluster `.bin` carries the record with the vector stripped, the cluster `.vec` carries
the quantized vector block scanned by the vector arm, and `rerank.data` holds the exact f32
for the final rerank. See [arms/vector.md](arms/vector.md).

---

## 3. Write path

Writes go to the WAL, never to a segment. A commit is one object.

```
client ──gRPC──▶ any node
  derive_links_for_write(snapshot, batch)   # set each upsert's semantic_out BEFORE commit
  Writer::commit(ops)                        # ops = Vec<Op>, one entry per commit
    seq = head + 1
    PUT wal/{seq:020}.bin  If-None-Match:*
    412 conflict ⇒ re-read head, seq = head+1, retry (bounded)
  ack after the PUT succeeds  (INV-5: now durable + visible)
```

- **One `WalEntry` = one atomic transaction** to every reader — all its ops or none.
- **`Op`** is `Upsert(Memory)`, `Tombstone { id }`, `Patch { id, deltas }`,
  `TombstoneWhere { predicate }`, or `Guard { expect_seq_lt }` (optimistic CAS). Deletes are
  tombstones only — a `TombstoneWhere` deletes every memory matching a metadata/tag/type
  predicate whose last write is *older* than the entry's seq, which makes "replace all of a
  document's facts" one atomic, re-ingest-safe op.
- **Semantic kNN links are derived on the write path, before the commit** (`derive_links_for_write`,
  §5), so they travel in the WAL as intrinsic `semantic_out` data alongside entity ids and causal
  edges. This is what makes the index a pure speed optimization: a query over the un-indexed WAL
  tail already sees a memory's links, because they are in the WAL, not synthesized by the fold.
- **Pipelined commits.** `Writer::commit_many(batches, concurrency)` issues many WAL PUTs
  concurrently against a running head, so bulk ingest is bound by S3 throughput, not by
  per-commit round trips. Each batch is still one atomic entry.
- Read-modify-write is forbidden; use `Patch` deltas (commutative, fold-able at read and
  index time) and `Guard` for preconditions.

---

## 4. The segmented (LSM) index

A generation is not one monolithic file set — it is an **ordered stack of immutable
segments across levels** (L0 = newest/smallest), each a self-contained mini-index over its
own slice of items. This is the LSM model (Lucene/turbopuffer), chosen because immutable
object storage rules out SPFresh-style in-place mutation.

- A **flush** turns the un-indexed WAL tail into one new small **L0 segment** — `O(tail)`,
  never touching prior segments.
- A **query** fans each arm out across all segments + the still-un-indexed WAL tail and
  merges (§6).
- **Deletes and re-upserts are a read-time overlay**, not a rewrite: a newer segment
  shadows an older copy.
- A background **tiered compaction** merges segments to bound the count (§5.3) — the only
  `O(corpus)` step, off the write path.

This is what makes fold cost independent of corpus size: folding 100 memories into a 100M
namespace builds a 100-item L0 segment, not a 100M-entry rebuild.

### The supersede overlay

Each segment stores a a `tombstones` object (rkyv `SegmentTombstones`): the ids it supersedes
(genuine deletes **plus** ids it re-upserts) and its predicate-deletes. At query open a node
folds these into `seg_superseded: id -> max seq_hi`. A candidate from segment S is hidden if a
**newer** segment (higher `seq_hi`) supersedes its id, or a predicate-delete at seq `> write_seq`
matches it. That is the `liveDocs` overlay, resolved across segments at read time — position-based
by `seq_hi`, so the 1-bit vector-block scan never needs to decode `write_seq`.

---

## 5. Indexer / fold

`fold()` is the single entry point and **auto-selects** what to run against the manifest:

| condition | path | cost |
|---|---|---|
| no segments (first build) or `≥ COMPACT_FANOUT (8)` segments | full rebuild — `index()` (in-RAM) or `index_streaming()` (bounded-RAM) | `O(corpus)` |
| otherwise (steady state) | `flush()` — append one L0 segment | `O(tail)` |

The in-RAM vs streaming choice on the full-rebuild path is by size:
`estimate_corpus_docs ≥ MEMLAKE_INDEXER_STREAMING_THRESHOLD` (default 4M docs) uses the
external-memory fold, otherwise the in-RAM fold. Each fold reads the manifest, reads WAL
entries `(wal_index_cursor, wal_head]`, produces new segment(s), and CAS-swaps the manifest.

### What a build produces (per memory_type)

1. **Centroid training** — `√N` centroids by k-means over a deterministic ≤50k sample
   (`sq_dist` is the hot primitive, SIMD-vectorized into fixed lanes so it stays fast *and*
   byte-deterministic). Then every item is assigned to its nearest centroid, and oversized
   clusters `local_split` in two.
2. **Cluster + vector blocks** — item records (embedding stripped) to `cluster-{i}.bin`, the
   quantized vector block (with tag + `updated_at` columns) to `cluster-{i}.vec`, and the
   per-cluster tag/write-time summary used to prune clusters before fetch. See
   [arms/vector.md](arms/vector.md).
3. **Semantic links — carried, not derived.** Each item already carries its `semantic_out` from
   the write path (§3, `derive_links_for_write`), so the fold does *no* derivation: it copies the
   links forward and mirrors their reverse edges into `radj` (`feed_radj`). This holds on both the
   in-RAM and streaming paths — the index reorganizes links for fast reads, it never invents or
   drops them. See [arms/graph.md](arms/graph.md).
4. **SSTables + FTS** — `pk`, `payload`, `entity`, `time`, `rerank`, `radj` (`.idx` sparse
   offset + `.data` blocks) and the tantivy split.

### Compaction — the only O(N) step

When the segment stack reaches `COMPACT_FANOUT (8)`, the next fold **compacts**: it merges
*all* segments + the tail into one fresh segment, resolving last-writer-wins + tombstones and
physically reclaiming shadowed/deleted items (centroids are retrained over the merged
population; the items' write-time links are kept, never re-derived). A large merge is exactly the
bounded-RAM workload the **streaming fold** solves, so streaming is the compaction engine and
the fast in-RAM path builds small L0 flushes. Idempotent and coordination-free (INV-6): a lost
CAS just means a peer built an equivalent segment; GC drops unreferenced `seg-*` prefixes after
a reader-grace TTL.

---

## 6. Read path

A `QueryNode::open` loads each segment's small hot state (centroids, SSTable indexes,
tombstones) + the WAL tail. `query_raw_metered` runs the arms and returns **raw per-arm
signals** — `(id, rank, score)` per arm, no fusion — because memlake leaves fusion to the
caller (the client applies RRF / weights / re-ranking however it likes).

Every arm **fans out across all segments + the WAL tail and merges**, newest-source-wins:

- **Vector** — probe each segment's centroids, run the RaBitQ two-stage search per segment,
  merge exact-scored top-ks (globally exact: a global top-k item is top-k within its own
  segment's pool). The WAL tail is scored exhaustively.
- **FTS** — query each segment's tantivy split + a tail split built on the fly; merge hits.
- **Graph** — one-hop expansion from the vector arm's seeds over each segment's `radj` +
  entity postings + the seeds' inline links.
- **Temporal** — ranged reads of each segment's `time` index for the window, then a one-hop
  spread.
- **Supersede overlay** applies across every arm: a hit is dropped if a newer segment
  supersedes its id or a predicate-delete matches it (§4).

Winners are **materialized once** (payload lookup, bounded by arm depth), shared across arms.

### Roundtrip budget (INV-7) and cache

Cost is `O(#segments)`, which compaction keeps a small constant (`≈ fanout × levels`). A
query that exceeds its cold budget is a bug and emits a metric. The cache is a two-tier
(bounded RAM + NVMe) **CLOCK** ring — scan-resistant, a read takes the lock only shared and
sets an atomic reference bit, eviction is a hand-walk. NVMe hits are served by **mmap** (no
blob copy). Admission is read-through; a fold does not pollute the cache by default
(`Store::put_admitting` is the opt-in for bytes a query is certain to want).

### Consistency

Reads always merge the WAL tail over the indexed segments, so an acked write is visible
immediately (INV-5). `WriteRequest.wait_for_index` folds inline before returning for callers
that want the write in a segment (e.g. so the vector arm — which needs clusters to probe —
sees it). Cross-store atomicity with an external system (Postgres) is handled by an
idempotent outbox pattern; see TODOS and the deletes discussion in [arms/graph.md](arms/graph.md).

---

## 7. Crate map

| crate | responsibility |
|---|---|
| `mlake-core` | types, ids, rkyv formats (`Memory`/`StoredMemory`), `Op`/WAL entry, `Predicate`, manifest + segment model, envelope/rkyv IO |
| `mlake-store` | `object_store` wrapper, instrumented S3 client, CAS helpers, the CLOCK disk/RAM cache |
| `mlake-wal` | write path (commit / pipelined commit), WAL head discovery, tail scan, namespace lifecycle |
| `mlake-ivf` | k-means, centroids, vector block codecs (binary/int8/f32) + RaBitQ, cluster files |
| `mlake-fts` | tokenizer chain, tantivy directory over the store, split packing |
| `mlake-graph` | reverse adjacency (`radj`), the link-expansion retriever + structural scorer |
| `mlake-index` | the fold (index/flush/streaming), compaction, the `QueryNode` + all four arms, fusion helpers |
| `mlake-server` | the `serve` gRPC API and the `index` loop; service-scoped config |
| `mlake-perf` / `mlake-bench` | throughput/latency harness and the BEIR quality harness |

---

See the arm docs for the technical detail of each retrieval path:
[vector](arms/vector.md) · [text](arms/text.md) · [graph](arms/graph.md) ·
[temporal](arms/temporal.md). Configuration and deployment:
[CONFIGURATION.md](CONFIGURATION.md) · [DEPLOYMENT.md](DEPLOYMENT.md). Design rationale:
[DECISIONS.md](DECISIONS.md).
