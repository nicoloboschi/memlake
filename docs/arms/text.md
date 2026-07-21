# The full-text arm

Keyword retrieval via **BM25**, backed by a real **tantivy** index packaged the S3-native
way. Complements the vector arm: it catches exact terms, rare tokens, and names that dense
embeddings blur together.

Per **memory_type**, like every arm. Chinese-capable via a dual-emission tokenizer chain.

Relevant code: `mlake-fts` (tantivy wrapper, tokenizer), `mlake-index/indexer.rs` (the
build), `mlake-index/query_node.rs::fts_arm` (the read).

---

## Write path

1. **Tokenize.** Each memory's `text` goes through the tokenizer chain (`mlake-fts`,
   SPEC §8): Unicode **NFKC** normalization, **OpenCC** (traditional→simplified Han), then a
   **dual emission** — `jieba` word segmentation *and* CJK unigrams — so a query can hit
   either a segmented word or a single character. Latin text tokenizes conventionally. Tags
   are indexed alongside the text so the arm can filter on them.

2. **Build the tantivy index.** `TantivyFts::build_with_tags` builds a normal tantivy index
   in memory over `(id, text, tags)` for every item in the generation (`indexer.rs`,
   `fts_build` phase). tantivy computes and stores the BM25 statistics (term frequencies,
   document frequencies, field norms).

3. **Pack into one object.** The whole tantivy index is serialized into a single
   `gen-…/fts/split.bin` — a "split" — rather than left as tantivy's many small segment
   files. That one object is what the generation references (`GenerationFiles.fts_split`), so
   the FTS index is write-once and CAS-published with the rest of the generation (INV-2).

The tokenizer's configuration hash is pinned in the namespace manifest at creation, so the
same tokenization is used at index and query time (a mismatch would silently break recall).

---

## Read path

Driven by `fts_arm` in `query_node.rs`, for a query carrying non-empty `text`.

1. **Materialize the split** (at snapshot open). `read_fts_split` fetches `split.bin` and
   maps it into the local NVMe/mmap tier so tantivy can serve reads from it. One ranged read,
   `get_immutable`-cached, amortized by the snapshot cache — part of the bounded cold-open,
   not per query.

2. **Search — no object-storage roundtrips.** BM25 query execution runs entirely over the
   materialized split in local memory/mmap. `search_filtered(text, depth, tags)` tokenizes the
   query with the same chain, scores documents by BM25, applies the tag filter, and returns
   the top `text_top_k` with their **raw BM25 scores**.

3. **Overlay the WAL tail.** The un-indexed tail has its own small in-memory tantivy index
   built at open (`tail_fts`); the arm searches it too and merges. Tail hits that were
   tombstoned are dropped. The generation and tail rankings are merged, de-duplicated by id,
   re-sorted by score, and truncated to `text_top_k`.

The arm returns `(MemoryId, bm25_score)` pairs. As with every arm, memlake does **no**
fusion: the raw BM25 score and this arm's rank are returned as `hit.text = {present, rank,
score}`, and the client combines them with the dense and graph signals.

### Roundtrips & caching

- The FTS arm does **0 object-storage roundtrips at query time** — everything runs against the
  materialized split. Its only storage cost is the one-time split fetch at snapshot open,
  shared with every other query on that snapshot.
- **Warm:** the split stays materialized in the cached snapshot, so BM25 is a local operation
  throughout. This is why `fts` and `hybrid` workloads show `rt 0.0` even cold in the
  benchmarks once the snapshot is open.
- **Known gap vs Hindsight:** memlake indexes `text` only. Hindsight enriches the BM25
  document with entity names and spelled-out date tokens (`text_signals`), so its keyword arm
  is stronger on entity/date queries. Closing this means indexing `text + signals` or adding a
  second indexed-text field (TODOS §2).
