# Scale benchmarks

Largest-scale performance runs of the real write → index → query path over object storage
(MinIO). Driven by `mlake-perf` (in-process: `Writer`/`index`/`QueryNode` directly against the
`Store`), which is the only path fast enough to reach millions of memories per namespace.

## Environment

- Machine: 14 cores, 36 GB RAM, local MinIO (single node), ~17–25 GB free disk.
- Corpus: synthetic, `dim=384`, 3 memory_types, clustered vectors (√N IVF centroids), Zipfian
  tags + entity ids (`mlake-perf` datagen).
- Each scale is an independent namespace, cleaned between runs so each gets the full local disk.

## Hard limits found (why 100M is not local)

- **Disk:** 100M × 384-dim f32 = ~153 GB of vectors alone; local free disk is ~20 GB.
- **RAM — the real blocker:** the first-build fold is **O(N) in RAM**. `index()` loads every
  live memory into `by_id: BTreeMap<_, StoredMemory>`, collects a `Vec<StoredMemory>`, clones
  per-cluster, and (before the fix below) ran two serial O(N·k) passes. 100M ≈ 200 GB+ RAM.
  A real 100M build needs a streaming / external-memory fold (sampled-train is already done;
  the item residency and per-cluster spill are the remaining work). Tracked separately.

## Engine changes made for scale (this work)

1. **Parallelized the two dominant O(N·√N) fold passes.** k-means *training* already sampled
   (50k cap) and ran parallel Lloyd iterations, but the **cluster-size histogram**
   (`train_centroids`) and the **per-item centroid assignment** (`build_memory_type_index`)
   were serial scans over all N. Both are embarrassingly parallel (nearest-centroid is pure),
   so they now use rayon. At 1M the `train` phase dropped **67s → ~20s per memory_type** (2.9×,
   ~8 cores busy). Deterministic — same assignments, just faster.
2. **`mlake-perf --no-links`** to skip semantic-link derivation for pure vector/FTS scale runs
   (see below).

## Known bottleneck: graph-link derivation does NOT scale

`derive_links_ivf` is O(N · avg_cluster_size) ≈ O(N·√N). At 100k it is ~8s/type; at 1M
(333k/type) it ran **>4 minutes for a single type without finishing**. So the graph arm's build
caps the *with-graph* first-build scale at roughly a few hundred k – 1M today. The scale ladder
below runs `--no-links` to characterize the vector-IVF + FTS path at its true ceiling; the graph
build is a separate scaling item (candidate fixes: cap neighbors per item, approximate kNN via
the centroids already computed, or derive links incrementally only for new items as SPEC intends
at steady state).

## Operational lesson

SIGKILLing an in-flight `mlake-perf` leaves half-open connections that later surface as
`object_store` "end of file before message length reached" on the next run. Restart MinIO after
any killed run. (Also: multi-minute runs must use a detached background task — `nohup &` inside a
tool call gets killed with the tool's process group at timeout.)

## Results — scale ladder (`--no-links`, vector IVF + FTS)

Measured on the machine above, one scale at a time (namespace dropped between scales).

| N | write | index build (--no-links) | peak RSS | stored | vector p50 warm |
|---|-------|--------------------------|----------|--------|-----------------|
| 1M | 59k mem/s (17s) | 132s | 5.4 GB | 1.95 GB | **4.8 ms** |
| 3M | 47k mem/s (65s) | 446s (7.4 min) | 12.2 GB | 5.83 GB | _(read not captured)_ |

- **Vector query stays flat and fast** (4.8ms warm at 1M) — IVF nprobe + SSTable range reads
  are independent of N, as designed.
- **Peak RSS scales linearly** (~4 GB/M): the O(N) in-RAM fold caps this machine at ~5–6M
  (36 GB). This is the streaming-fold item to lift the ceiling.
- **Build time grows ~N** (~7.5 min at 3M with `--no-links`), dominated by k-means assignment
  (now parallel) + cluster writes.

## Cumulative per-arm wins @ 1M (warm p50)

| Arm | Original | Now | Change |
|-----|----------|-----|--------|
| vector | 4.8ms | ~4.5ms | rerank query-norm hoisted |
| fts | 43ms | ~7.6ms | payload store (~5.7×) |
| hybrid | 45ms | ~8.3ms | payload store |
| graph | **644ms** | **~8.7ms** | score-then-hydrate + payload (**~74×**) |
| temporal | (unbounded/hung) | ~22ms, bounded | pool cap + payload |

All landed as committed, tested changes: graph score-then-hydrate; parallelized fold
O(N·√N) passes; payload store; temporal pool cap; rerank query-norm hoist; `get` via payload.

One remaining query-side item is a deeper refactor best done with a validation checkpoint: the
**zero-copy cluster rerank** — `fetch_clusters` fully rkyv-deserializes each cluster to score it;
scoring over the archived form and deserializing only the top-k would trim every cluster-fetching
arm, but it's a hot-path API change with rkyv-lifetime ripple.

## Write throughput — pipelined commits (vs turbopuffer's 150 MB/s)

The write path is `commit()` = one `put_if_absent(wal/{seq}.bin)`. A single writer's sequences
are contiguous and each WAL object is an independent conditional create, so the batches don't need
to be written one blocking round-trip at a time. `Writer::commit_many(batches, concurrency)`
pre-assigns the sequences and pipelines the object PUTs `concurrency` at a time (`buffer_unordered`),
so throughput is bounded by S3 parallelism rather than a serial chain of round-trips.

Measured at 1M memories (~1.8 GB of WAL) on the **local MinIO** node above. Synthetic
embedding/text generation is excluded from the timing (it's a benchmark artifact, timed
separately as `datagen`); this is pure durable-write cost. `--no-index` (index skipped):

| concurrency | commit time | throughput |
|-------------|-------------|------------|
| x1 (serial) | 41.9s | 43 MB/s |
| x8 | 21.0s | 86 MB/s |
| **x16** | **14.0s** | **129 MB/s** |
| x32 | 15.3s | 118 MB/s |

- Pipelining lifts throughput **3× (43 → 129 MB/s)**, peaking at concurrency 16 — **~86% of
  turbopuffer's 150 MB/s**, on a single local MinIO node.
- It **saturates** by x16; x32 slightly regresses (local MinIO / single-disk contention). Against
  real S3 with more front-end parallelism the knee should move right and the ceiling up.
- Correctness is unchanged: `commit_many` still claims contiguous, hole-free sequences (the log is
  totally ordered); a lost slot fails the burst and invalidates the cached head. It is the
  sole-writer bulk-load path — a contended namespace uses `commit()` with its CAS retry loop.
- Bench flags: `mlake-perf write --scale N --no-index --no-links --commit-concurrency 16`.

## Streaming (external-memory) fold — DONE

The first-build fold was O(N) in RAM (`by_id`/`items` hold every memory; the SSTable builds hold
every pair), so it capped this 36 GB box at ~5–6M. `index_streaming` (`mlake-index/src/streaming.rs`,
opt-in via `mlake-perf write --streaming` and `mlake-server index --once --streaming`) makes the
fold bounded:

- **Resolution** streams the WAL twice — pass 1 builds only id-level state (winner upsert seq,
  tombstone, accumulated patches; ~50 B/id, no bodies), pass 2 re-reads and spills each winning
  item to a per-type disk `ItemSpill`. The previous generation is streamed cluster-by-cluster and
  overlaid. Last-write-wins / tombstone / patch / predicate-delete semantics match `fold_entries`.
- **Per-type build** trains centroids on a reservoir sample (`train_centroids_k` — never scans all
  N), then makes ONE pass assigning each item and feeding five `ExternalSort`s (one carries the
  full item bytes grouped by cluster; the rest carry pk/payload/entity/time fragments) plus a
  streaming FTS builder. Each external sort spills sorted runs and k-way-merges them, so the
  cluster files and SSTables are written from bounded memory.

Scope: the **bulk build** path. The fold derives no semantic links at all (in either the in-RAM or
streaming path): links are derived on the write path before the commit and carried in the WAL as
`semantic_out`, so the fold only feeds their reverse edges into `radj`. The streaming build skips
local-split. The result is a correct, queryable generation.

**Measured @ 800k (--no-links):**

| Fold | peak RSS | index time |
|------|----------|------------|
| in-RAM | **4,631 MB** (~5.8 GB/M, caps at ~6M) | 112s |
| streaming | **~920 MB** (bounded: 512 MB sort budget + ~50 B/id) | **120s** |

~5× less RAM at 800k, and the gap widens with N — the streaming fold's memory is bounded, so it
scales to 100M+ on adequate disk, at **~parity build time** (was 256s before the assign pass was
parallelized — see below). Equivalence + incremental-refold correctness are covered by
`streaming_fold_matches_in_ram_fold` and `streaming_fold_incremental_matches_in_ram`; the
spill/merge primitives by unit tests in `spill.rs`.

**Phase profile (300k, per memory_type) + the speedups:**

| phase | before | after parallel assign |
|-------|--------|----------------------|
| train (k-means) | ~13s | ~13s (dominant; parallel + early-terminating already) |
| assign | ~10s | **~1.7s** (batched rayon: nearest-centroid + both serializations) |
| cluster_write | ~3s | ~3s |
| wal pass 1 + 2 | ~3.5s total | ~3.5s (small — the double WAL read is *not* a bottleneck) |

The parallel assign took a 300k build 85.5s → 57.6s (and 800k 256s → 120s). Remaining lifts, in
order: (1) **`train`** is now the dominant phase but it's inherent k-means, shared with the in-RAM
fold, and already parallel + convergence-terminating — cutting it means fewer iterations or a
smaller sample, a recall trade to validate against BEIR, not a free win; (2) **`cluster_write`** is
streaming-specific (it deserializes each item off the merge and re-serializes the `ClusterFile`) —
a concatenated cluster-file format would let it copy spill bytes through without the round-trip, at
the cost of a format change.

## Per-arm query profile @ 1M (warm, rt=0 → pure compute/cache)

| Arm | p50 | Breakdown | Diagnosis |
|-----|-----|-----------|-----------|
| vector | 4.8 ms | fetch_clusters 2ms + rerank 1.6ms | healthy |
| tags | ~20 ms | fetch_clusters 10ms + rerank | ok |
| fts | 43 ms | internal `fts` 8ms; **~35ms overhead** | split re-open per query (TODO) |
| graph (pre-fix) | **644 ms** | `graph_fetch` **318ms** + graph_pk 20ms | hydrated ~8k candidates to return 50 |

### Graph fix — score-then-hydrate

The graph arm gathered `per_entity_cap` (200) × up-to-40 seed-entities ≈ **8,000 candidates** and
read each one's **whole cluster file** (~500 clusters × ~1MB) just to score them, then truncated
to `budget=50`. Two changes remove that:

1. **Entity score from postings, not hydration.** A candidate's shared-entity count *is* the
   number of distinct seed-entity postings it appears in — no need to read the memory to
   recompute `shared_entity_count`. `GraphSource::item()` (returned a full `StoredMemory`) became
   `exists()` (a tombstone check); semantic/causal liveness is just "not tombstoned" (targets are
   already liveness-filtered at fold time).
2. **Hydrate only the ranked results.** With no tag filter, the graph arm reads *nothing* extra —
   the shared final materialization pass hydrates the ≤`budget` surviving hits. With a tag filter,
   only the ≤`budget` ranked ids are hydrated to read their tags.

Net: from hydrating ~8,000 candidates to ≤50 (or 0).

**Measured @ 1M (warm): graph p50 644ms → 44.5ms (~14.5×).** The `graph_fetch` (318ms) and
`graph_pk` (20ms) phases are gone; the graph arm is now `graph_radj` 0.3ms + `graph_expand`
0.9ms. Its residual ~40ms is the *shared FTS overhead* below (the graph workload also runs the
text arm) — fixing FTS drops graph toward the vector arm's ~5ms. Correctness held: recall check
still surfaces the entity-sharer at the identical 0.462 score; mlake-graph (25) + mlake-index (69)
tests pass; postings-count matches `shared_entity_count` under the LATERAL cap.

## Next: point-read materialization (#2) — the last systemic cost

`fts`, `hybrid`, and `graph` all sit at ~44ms warm while their internal search timers are single-
digit ms. The overhead is **not** tantivy (the `IndexReader` is reused, not re-opened) — it is
**materialization**. The final pass hydrates every hit not already in a probed cluster by calling
`fetch_clusters`, which **deserializes the entire cluster file** (~577 items at 1M) to extract the
handful of hit memories in it. FTS/graph hits are spread across many clusters, so a text query can
deserialize ~100 whole clusters (~110 MB warm) to return ~100 rows.

Proof from the data: the **vector** arm materializes from the clusters it already probed → 4.8ms;
every workload that materializes hits *outside* the probed set (`fts`, `hybrid`, `graph`) pays the
same ~40ms, regardless of arm. So the cost tracks "hits needing out-of-probe hydration", i.e.
whole-cluster reads for point lookups.

This is the same root cause the graph fix dodged by hydrating only ≤`budget` results. The general
fix is a **row-addressable payload** so a point read fetches one memory, not its whole cluster:
- **Option A (systemic): a payload SSTable** `id -> memory bytes`, range-read per id (mirrors
  pk/radj/entity). Fixes fts/graph/get/temporal at once; costs a second copy of the payload, or a
  storage-layout split (vectors stay in clusters for rerank, the rest moves to the payload store).
- **Option B (FTS-targeted): store the non-vector payload in the tantivy split** (text+tags are
  already there; add metadata/entity_ids/timestamps as STORED fields). FTS hits hydrate straight
  from the split — no cluster read. Smaller blast radius, doesn't help graph/get.

Recommended: Option A, since it also removes the residual ~40ms from the graph arm and speeds
`get`.

### Payload store — DONE (Option A)

Added `PayloadTable`: `id -> memory rkyv bytes with the embedding stripped`, an SSTable
range-read per id (like pk/radj/entity). The fold builds it; the final query materialization
pass and the graph tag-filter branch hydrate misses via one coalesced payload lookup instead of
pk + whole-cluster fetch. The vector is omitted because query hits return `MemoryPayload` (no
vector); the vector arm's rerank and the temporal arm's entry-point ranking still read clusters
(they need vectors), as does `get --include_vector`. `FORMAT_VERSION -> 3`. Storage grows ~17%
(1.95 → 2.28 GB at 1M) for the vector-stripped rows.

**Measured @ 1M (warm p50):** fts **44 → 7.7ms** (~5.7×), hybrid 45 → 8.3ms, graph **44.5 → 8.9ms**.
Combined with the earlier score-then-hydrate change, the graph arm went **644ms → 8.9ms (~72×)**.
Every arm is now single-digit ms warm (vector 4.7, fts 7.6, hybrid 8.3, graph 8.7), except tags
(~17-21ms, which fetch probed clusters + rerank) and temporal (see below).

## Temporal arm — enabled, and a scaling finding

The datagen now stamps `occurred_start` (memory `i` at `epoch + i`s), and `read_bench` has a
`temporal` workload (vector + time window). Enabling it surfaced a real issue: the temporal arm
**materializes the entire time window before truncating to its 60-entry pool**, and it needs the
candidates' *vectors* (it ranks entry points by similarity to the query), so it reads whole
cluster files. Because time and vector-cluster are uncorrelated, a window of W memories touches
~min(W, √N) clusters — a wide window (e.g. 2% of history ≈ 20k memories at 1M) reads most of the
corpus and the query never returns in reasonable time. Bounded to ~120-memory windows it is fine.

This is the same *materialize-before-truncate* shape the graph fix removed, but harder: the
truncation key (similarity) needs vectors, which the payload store omits.

**DONE (option 1): `in_window` now caps its result to a time-spread sample** (`TEMPORAL_WINDOW_CAP`
= 256, ~4× the pool) via an even stride over the time-ordered range, so the arm materializes at
most that many entry points regardless of window width (INV-7). Validated: a span=100000 window
(~half a 200k corpus — which hung before) now runs **32ms warm / 53ms cold**, bounded. Narrow
windows are unaffected; recall is unchanged on them.

## Concurrent-request percentiles

`read_bench --concurrency C` runs queries through `buffer_unordered(C)` so percentiles reflect
latency under parallel load (C=1 is the sequential baseline).

**Measured @ 1M, warm, C=32 (32 queries in flight):** latency barely moves vs sequential — the
read path is CPU + local cache with no cross-query contention (a stateless node scales with
cores):

| Arm | warm p50 | warm p90 | warm p99 |
|-----|----------|----------|----------|
| vector | 4.7ms | 5.5ms | 10.2ms |
| fts | 7.6ms | 8.1ms | 16.2ms |
| hybrid | 8.3ms | 9.3ms | 23.8ms |
| graph | 8.7ms | 11.4ms | 63.1ms |
| temporal | 22.6ms | 35.2ms | 163.7ms |

**Cold, C=32** spikes (p90/p99 in the 100s of ms): 32 concurrent cold queries all miss the cache
and hammer the *single local MinIO* at once, saturating it (e.g. vector cold fetch_clusters
146ms). That is a one-box test artifact, not a design limit — in production each stateless node
has its own warm cache and S3 fans out across many backends. Warm is the served-state metric.
