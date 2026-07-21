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
