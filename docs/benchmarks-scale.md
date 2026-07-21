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

## Next: FTS overhead (#2)

`fts`, `hybrid`, and `graph` workloads all sit at ~44ms warm while their internal `fts` timer is
~8ms and the vector arm is ~5ms — a fixed ~35ms per-query overhead that appears whenever the text
arm runs. Likely the tantivy split being re-opened/mmapped per query instead of reused warm.
Fixing it drops fts/hybrid/graph to single-digit ms. (Not yet done.)
