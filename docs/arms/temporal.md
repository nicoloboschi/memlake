# The temporal arm

Time-window retrieval: given a `[from, to]` window, find the memories whose effective time
falls inside it, spread one hop through links, and score by proximity to the window's centre.
A 1:1 port of Hindsight's `retrieve_temporal_combined`, with the BFS **bounded to one hop**
(cold) so it stays inside the roundtrip budget (INV-7) rather than Hindsight's five DB
iterations.

Per **memory_type**, like every arm. Runs only when the query carries both a `vector` (entry
points are similarity-ranked) and a time window.

Relevant code: `mlake-index/sstable.rs` (`TimeTable`, `scan_range`), `mlake-index/temporal.rs`
(the scoring math, golden-tested), `mlake-index/query_node.rs::temporal_arm` (the read).

---

## Write path

The temporal arm adds one artifact per generation: the **time index**.

1. **Effective timestamp.** For each memory the indexer computes `effective_ts =
   COALESCE(occurred_start, mentioned_at, occurred_end)` (an `i64` epoch; the unit only needs
   to match what the client wrote). Memories with no timestamp are simply not indexed.

2. **Time index — `time.idx` / `time.data`** (`TimeTable::build`). `effective_ts → [MemoryId]`,
   the same range-readable SSTable shape as radj/pk/entity, grouped by instant. The key is an
   **order-preserving** 16-byte encoding of the `i64` — the sign bit is flipped and the value
   stored big-endian, so raw byte order equals numeric order (negatives sort before positives).
   This is what lets a window query be one bounded ranged scan instead of a full sort.

Listed in `GenerationFiles`, kept live by GC (`all_paths`), and CAS-published with the rest of
the generation (INV-2). Back-compat: generations built before the time index load an empty one.

The same `TimeTable` is designed to serve the memories **timeseries** (bucketed counts over the
window) — one artifact, two consumers.

---

## Read path

Driven by `temporal_arm` in `query_node.rs`.

1. **Entry-point pool** — `TimeTable.in_window(from, to)` is one bounded ranged scan of the key
   range (RT4) returning the in-window ids, plus in-window tail items (a small filter over the
   overlay). The candidates are materialized (coalesced pk lookup + cluster fetch, reusing the
   shared materialized map) and scored by cosine similarity to the query, tag-filtered.

2. **Coverage selection** (`select_with_temporal_coverage`, ported exactly). Take the top
   `TEMPORAL_POOL_SIZE` (60) by similarity, split the window into `TEMPORAL_COVERAGE_BUCKETS`
   (8) equal buckets, then round-robin: the best-similarity item from each populated bucket,
   then the second-best from each, until `TEMPORAL_ENTRY_POINTS` (10) are chosen. So entry
   points **span the window** rather than clustering in the densest slice; degenerate dates
   collapse to plain similarity order.

3. **Score entry points** — `temporal_proximity(best_date, from, to) = 1 − min(|best − mid| /
   (span/2), 1)`, so a memory at the window centre scores 1 and one at an edge scores 0.
   `best_date` is the occurred-interval midpoint, else its start, else its end, else the
   mention time (a distinct cascade from the `effective_ts` used to *index* it). No date →
   0.5.

4. **One-hop spread** — one coalesced `radj` read gives the entry points' incoming edges;
   combined with their inline `semantic_out`/`causal_out`, that is the one-hop neighbour set,
   materialized in a coalesced fetch. Each neighbour is scored `max(its own proximity,
   parent · weight · causal_boost · 0.7)` where `causal_boost` is 2.0 for causes/caused_by,
   1.5 for enables/prevents, 1.0 for semantic (memlake's stand-in for Hindsight's `temporal`
   links). No date → 0.3. Reached from several seeds → the max is kept. **No recursion** — one
   hop only.

The arm returns `(MemoryId, temporal_score)`, surfaced as `hit.temporal = {present, rank,
score}`. As with every arm memlake does no fusion; the client combines the temporal score with
dense/text/graph however it likes.

### Roundtrips, bounds & caching

- **One hop only** keeps the arm's cost bounded by the entry-point count and the batch reads,
  independent of graph shape (INV-7). Hindsight's deeper BFS (5 iterations) can be re-enabled
  behind a config once its value on a real bank is measured; the default is one hop.
- **Cold:** the `in_window` scan, the entry-point materialization, the `radj` read, and the
  neighbour fetch all land in **RT4** (batched/concurrent), so the whole query stays within the
  4-roundtrip budget.
- **Warm:** the time index blocks and candidate clusters are immutable and cached by
  `(path, range)`, so a repeated temporal query is 0 roundtrips.

### Difference from Hindsight

memlake keeps the scoring and coverage selection identical (golden-tested, G-3). The spread
differs by design: Hindsight walks a `temporal` link type it maintains and iterates up to five
hops; memlake spreads one hop over its **derived semantic + inline causal** links (it has no
stored `temporal` links), treating semantic as the boost-1.0 equivalent.
