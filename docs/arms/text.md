# The full-text (FTS / BM25) arm

Keyword retrieval via **BM25**, backed by a real **tantivy** index packaged the S3-native way,
with a script-aware, Chinese-capable tokenizer. It catches exact terms, rare tokens, and names
that dense embeddings blur together. Like every arm it returns a **raw** ranking
`(id, bm25_score)` and leaves fusion to the caller — see [ARCHITECTURE.md](../ARCHITECTURE.md).

Code: `mlake-fts` (tantivy wrapper + tokenizer), `mlake-index/indexer.rs` + `streaming.rs`
(the build), `mlake-index/query_node.rs::fts_arm` (the read).

## Query time

`fts_arm` runs when the query carries non-empty `text` and `depths.text > 0`. It fans out
across **every source of the fact type** and pools raw hits:

1. The **WAL-tail** index (`state.tail_fts`, a small in-memory tantivy index built fresh at
   snapshot open) is searched first — tail-first, so a re-upserted id's tail hit wins the later
   dedup.
2. Each **segment**'s split (`seg.gen_fts`) is searched independently.
3. Each search is `TantivyFts::search_filtered(text, depth, tags)` → `[FtsHit { id, score }]`,
   where `score` is tantivy's **raw BM25**.

The pooled hits sort by score (id tiebreak), `dedup_by_key(id)` keeps the highest-scoring copy
per id, and the result truncates to `depth`. Tombstoned ids are dropped at each search and again
after the global arm merge; an FTS-only winner that fell outside the probed vector clusters is
hydrated by one coalesced payload GET.

The arm returns `(MemoryId, bm25_score)`; memlake does **no** fusion — the raw score and this
arm's rank surface as `hit.text = { present, rank, score }` for the client to combine.

**BM25 is per-segment.** Each split carries its own document frequencies, so pooling raw scores
across segments is approximate — the same tradeoff Lucene makes. It drifts most right after a
flush and resolves at compaction, when segments merge into one.

## What is stored

- **The split** — `fts/split.bin` per segment: one packed blob of the whole tantivy directory
  (`[count]` then per-file `[name_len][name][data_len][data]`, filenames sorted for stable
  framing), referenced from the segment's `FactTypeIndex`.
- Materialization is one ranged GET at snapshot open, unpacked into the local NVMe/mmap tier —
  so the **FTS arm does 0 object-storage roundtrips at query time**.
- **Schema — four fields:** `words` and `bigrams`, both indexed `WithFreqs` via the `raw`
  tokenizer (tokenization happens upstream), plus `STORED` `id` and `STORED` `tags` (JSON).
  tantivy sums the per-field BM25 contributions.
- **`index_text` / `fts_text()`** — a memory may carry an `index_text` distinct from its `text`;
  `fts_text()` returns `index_text` when non-empty, else `text`. It is what BM25 indexes and is
  **never returned** (`text` is), so a client can enrich matchable tokens — entity names,
  spelled-out dates ("May 8 2023") — without changing what a hit shows. Wired into all three
  build sites (in-RAM fold, streaming fold, WAL-tail index). *This closes the earlier gap vs
  Hindsight's `text_signals`.*

## The tokenizer

One chain, used by both indexer and query parser, its config hash pinned into the manifest so a
split and a query agree on tokenization:

1. **Normalize** — Unicode NFKC → optional traditional→simplified (via the `character_converter`
   crate) → lowercase.
2. **Script segmentation** — split into runs by Unicode script (Latin / Han / Kana / Hangul /
   digits).
3. **Per run** —
   - *Latin*: split on non-alphanumeric (keeping `_` and digits so identifiers survive), drop
     stopwords, Snowball-English stem; emitted into **both** fields.
   - *Han*: **dual emission** — jieba `cut_for_search` words into `words`, character **bigrams**
     into `bigrams` (a single char only as a length-1 fallback).
   - *Kana / Hangul*: bigrams.
4. **Query side** — same chain; a query hits `words OR bigrams`, scores combined by tantivy.

`config_hash` is FNV-1a over `"v2:t2s={}:stem={}"` (the `v2:` prefix lets a chain change force
invalidation). It is a **create-time pin** — recorded at namespace creation and echoed on every
fold — not a query-time assertion, so a mismatch degrades recall silently rather than erroring.

## Optimizations and properties

- **Tail FTS on the fly** — the un-indexed tail's split is rebuilt in memory at every snapshot
  open, so a just-written memory is text-searchable immediately (INV-5).
- **Tag filtering is one shared path** — applied inside `search_filtered` by reading each hit's
  stored `tags` and running the same `TagFilter` every arm uses (not a tantivy tag query). To
  keep a selective filter from starving `k`, it over-fetches `k*50` (clamped `[k, 10_000]`) when
  a filter is present, else exactly `k`.
- **Determinism caveat** — retrieval is reproducible, but the split bytes are **not**
  byte-identical across rebuilds: tantivy stamps each segment with a random UUID. The one place
  G-6 byte-determinism does not hold; its *results* still do.

## Caveats

- **The `updated_at` window is not honored by this arm.** The window pushes down only into the
  vector arm and cluster selection; the FTS arm ignores it, so a stale-but-matching document can
  surface from FTS outside the requested update window. The server re-applies the window over all
  arms and remains the authority.

## Build

Two builders share one `add_doc`, so documents index identically either way:

- **Batch** (`TantivyFts::build_with_tags`) — the in-RAM fold and the tail index; a 50 MB writer
  arena.
- **Streaming** (`TantivyFtsBuilder`) — the external-memory fold; arena = `MEMLAKE_FOLD_FTS_MB`
  (default 128 MB), tantivy spilling segments to disk as it indexes, bounding the FTS stage's RAM
  in a compaction.
