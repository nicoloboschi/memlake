# The temporal arm

Time-window retrieval: given a `[from, to]` window, find the memories whose effective time falls
inside it, spread one hop through links, and score by proximity to the window's centre. A port of
Hindsight's `retrieve_temporal_combined`, with the spread **bounded to one hop** so it stays
inside the roundtrip budget (INV-7) rather than Hindsight's five DB iterations.

Runs only when the query carries **both** a `vector` (entry points are similarity-ranked) and a
time window. Code: `mlake-index/sstable.rs` (`TimeTable`), `mlake-index/temporal.rs` (scoring
math, golden-tested), `mlake-index/query_node.rs::temporal_arm`.

## Query time

1. **Entry-point pool.** For each segment with a time index, one bounded ranged scan
   `TimeTable::in_window(from, to, cap)` returns the in-window ids (all in RT4); plus in-window
   tail items. Over `TEMPORAL_WINDOW_CAP (256)` ids, `in_window` takes an **even time-stride
   sample** (records arrive time-ordered, so the sample stays spread across the window). Ids are
   tombstone-filtered, materialized (coalesced pk lookup + cluster fetch, shared map), and scored
   by cosine to the query, tag-filtered.

2. **Coverage selection** (`select_with_temporal_coverage`, ported exactly). Take the top
   `TEMPORAL_POOL_SIZE (60)` by similarity, split the window into
   `TEMPORAL_COVERAGE_BUCKETS (8)`, then round-robin — best-similarity item of each populated
   bucket, then the second-best of each — until `TEMPORAL_ENTRY_POINTS (10)` are chosen. Entry
   points **span the window** rather than clustering in its densest slice; degenerate/equal dates
   collapse to plain similarity order.

3. **Score entry points** — `temporal_proximity(best_date, from, to) = 1 − min(|best − mid| /
   (span/2), 1)`, so a memory at the window centre scores 1 and one at an edge scores 0.
   `best_date` is the occurred-interval midpoint, else start, else end, else mention time — a
   deliberately *distinct* cascade from the `effective_ts` used to *index* it. No date → 0.5
   (`NO_DATE_ENTRY`).

4. **One-hop spread.** One coalesced `radj` read per segment gives the entry points' incoming
   edges; combined with their inline `semantic_out` / `causal_out`, that is the one-hop neighbour
   set, materialized in a coalesced fetch. Each neighbour is scored
   `max(own_proximity, 1.0 · weight · boost · 0.7)` — the parent term propagates from a **fixed
   1.0**, not the entry point's own proximity. `boost` is 2.0 for causes/caused_by, 1.5 for
   enables/prevents, 1.0 for semantic (memlake's stand-in for Hindsight's `temporal` links). No
   date → 0.3 (`NO_DATE_NEIGHBOR`). Reached from several seeds → the max is kept. One hop only, no
   recursion.

The arm returns `(MemoryId, temporal_score)`, surfaced as `hit.temporal = { present, rank,
score }`; memlake does no fusion.

### Bounds & caching

- **One hop only** keeps cost bounded by the entry-point count and the batch reads, independent
  of graph shape. Hindsight's deeper BFS can be re-enabled behind config once its value on a real
  bank is measured; the default is one hop.
- **Cold:** the `in_window` scan(s), entry-point materialization, `radj` read, and neighbour
  fetch all land in RT4, so the query stays within the roundtrip budget.
- **Warm:** the time-index blocks and candidate clusters are immutable and cached by
  `(path, range)`, so a repeated temporal query is ~0 roundtrips.

## Effective time vs `updated_at`

**Effective (content) time** = `COALESCE(occurred_start, mentioned_at, occurred_end)` — what the
memory is *about* or when it was mentioned. This is what the time index is keyed on and what
window membership tests. `updated_at` is the *write* time — a separate field, not in the time
index; it is carried only as a per-cluster `[min,max]` range in the tag summary so an
`updated_at` window can prune clusters before fetch. (Note the deliberate asymmetry: indexing
uses `mentioned_at` before `occurred_end`, while the *scoring* `best_date` cascade above uses
`occurred_end` before `mentioned_at`.)

## What is stored

The **time index** — `time.idx` / `time.data` (`TimeTable::build`): `effective_ts → [MemoryId]`,
the same range-readable SSTable shape as radj/pk/entity, grouped by instant. The key is an
**order-preserving** 16-byte encoding of the `i64` (sign bit flipped, stored big-endian, low 8
bytes zero), so raw byte order equals numeric order and negatives sort before positives.
`in_window` is one `scan_range` from `ts_key(from)` to `ts_key(to)` (widened to `0xFF` in the low
bytes so the whole `to` instant is inclusive), decoding all ids and applying the stride cap. A
generation built before the time index opens an empty one and skips the scan.

## Build

The `effective_ts` is computed identically in the in-RAM and streaming folds; ids are fed into
the `TimeTable` (in-RAM) or the time external-sort (streaming). The scoring math in `temporal.rs`
is pure and golden-tested (G-3) against the Hindsight values.
