# The vector arm

Dense retrieval over embeddings, via **IVF** (inverted file / coarse-quantized search):
cluster the corpus with k-means, then at query time probe only the nearest few clusters and
re-rank them exactly. Cost scales with `nprobe`, not with the corpus (INV-7).

Everything here is per **memory_type** — each type has its own centroids and cluster files;
nothing is shared across types.

Relevant code: `mlake-ivf` (k-means, cluster files, probe/rerank), `mlake-index/indexer.rs`
(the build), `mlake-index/query_node.rs::run_arms` (the read).

---

## Write path

The indexer folds the un-indexed WAL tail onto the previous generation and rebuilds the IVF
structure into a new immutable `gen-{G}-{nonce}/` tree (`indexer.rs`).

1. **Assemble items.** New memories from the WAL slice, plus the memories carried forward
   from the previous generation's clusters. Each is a `StoredMemory` (id, `vector`, text,
   tags, timestamps, entity_ids, edges, metadata).

2. **Train centroids** (`mlake-ivf/kmeans.rs`). `k = round(sqrt(N))` (SPEC §5.1). Training
   is **mini-batch k-means++**: seed spread-out centres, then Lloyd iterations. Over
   `TRAIN_SAMPLE_CAP` (50k) items it trains on a strided sample; assignment of *all* items
   still happens afterward. The assignment step and the k-means++ seeding are parallelized
   with rayon while keeping the f64 accumulation order fixed, so the output stays
   byte-identical for a given seed (G-6). Retraining is **not** every fold: centroids are
   reused (assign-only) until the corpus grows ~2× or a retrain is forced, so steady-state
   folds are cheap.

3. **Assign + split.** Each item is assigned to its nearest centroid. Oversized clusters are
   locally subdivided (SPFresh-lite) so no single cluster file dominates a read.

4. **Write cluster files.** One object per cluster, `gen-…/cluster-{i}.bin`, holding that
   cluster's `StoredMemory` records serialized with **rkyv** (zero-copy readable). Only
   **dirty** clusters (those that changed this fold) are written; unchanged clusters are
   referenced by their existing path — *copy-forward-by-reference*, so a fold writes O(new
   clusters), not O(all). Dirty-cluster PUTs run concurrently (`buffer_unordered`).

5. **Centroids** are serialized to `gen-…/centroids.json` and listed in the manifest's
   `GenerationFiles`. The cluster paths are listed too.

The generation is published by a single `If-Match` CAS-swap of the manifest (INV-2), so a
reader sees either the whole new IVF structure or the whole old one.

**What the vector arm persists per generation:** `centroids.json` + `cluster-{i}.bin` (one
per cluster). (The `pk` index — id → cluster — is built here too but is consumed by the graph
arm and hydration; see [graph.md](graph.md).)

---

## Read path

Driven by `run_arms` in `query_node.rs`, for a query carrying a `vector`.

1. **Open the snapshot** (once per namespace, cached across queries — see below). Loading a
   memory_type reads `centroids.json` into memory (`get_immutable`, cached). This is part of
   the bounded cold-open, amortized by the snapshot cache.

2. **Probe** (`Centroids::probe`, in-memory). Compute the `nprobe` centroids nearest the query
   vector — pure CPU over the cached centroids, **0 roundtrips**. With a tag filter, per-cluster
   tag summaries prune clusters that cannot contain a match *before* probing, so a selective
   filter still finds its matches (`select_clusters`, SCALE.md Phase 4b).

3. **Fetch the probed clusters** (`fetch_clusters`). A ranged/`get_immutable` read per probed
   cluster object, issued **concurrently** (one roundtrip wave, RT3). Immutable, so cached by
   `(path)` — a warm cluster is **0 roundtrips**. The fetched `StoredMemory` records are also
   what the graph arm reuses and what the hit payload is materialized from — one fetch, three
   uses.

4. **Overlay the WAL tail** and **apply the tag filter** inline (memories carry their tags, so
   filtering is free once fetched). Tombstoned ids are dropped.

5. **Exact re-rank** (`exact_search`). Brute-force cosine of the query against every fetched
   candidate, sorted, truncated to `vector_top_k`. This is the "coarse probe, exact rerank"
   of IVF: recall comes from probing enough clusters; precision from the exact final scoring.

The arm returns `(MemoryId, cosine)` pairs. memlake does **no** fusion — the raw cosine and
the candidate's rank in this arm are returned to the client as `hit.dense = {present, rank,
score}`, and the client fuses (see [text.md](text.md), [graph.md](graph.md)).

### Roundtrips & caching

- **Cold:** RT1 manifest + RT2 metadata (open, amortized) + **RT3 = one wave** of concurrent
  cluster GETs. The whole cold query is bounded at 4 waves (`COLD_ROUNDTRIP_BUDGET`, INV-7),
  verified by test.
- **Warm:** the server caches the open `QueryNode` snapshot per namespace and the immutable
  cluster blocks by path, so a repeat query is pure in-memory probe + rerank — **0
  roundtrips**. A write invalidates the snapshot; `EVENTUAL` reads reuse it within a TTL,
  `STRONG` re-checks the WAL head.
- **Tuning:** `nprobe` trades recall for cost (more clusters fetched = higher recall, more
  bytes/roundtrip-wave width); `vector_top_k` bounds how many candidates the arm contributes.

---

## Adaptive probing — tried, measured, rejected (negative result)

**Status: implemented, measured on BEIR, and removed.** No code, no env flag, no radii on the
centroid table. This section is the record so nobody re-derives it. The stopping rule is
*sound*; it simply never fires at these dimensions, and the number below says by how much.

`nprobe` is a fixed fraction of the cluster count (half, floor 8, cap 64). The fraction is a
guess: at a quarter, nfcorpus measured `ann_recall@10` 0.8598 and scifact 0.9517 — the same
fraction nine points apart, because the right number depends on how a corpus clusters, not on
how many clusters it has. The principled replacement is a **stopping rule**: probe
nearest-first, and stop when the k-th best score already in hand beats the best score any
unprobed cluster could possibly yield.

### The bound (sound — this part was never the problem)

Give each centroid a **radius** `R = max ‖v̂ − c‖` over its members, where `v̂ = v/‖v‖` is the
member direction (one `f32` per cluster, recomputed every fold — it cannot be carried forward,
because the assign-only fold adds members to centroids it did not retrain). With
`q̂ = q/‖q‖`, no member of cluster `i` can score better than

```
  cos(q, v) = ⟨q̂, v̂⟩ = ⟨q̂, c⟩ + ⟨q̂, v̂ − c⟩ ≤ ⟨q̂, c⟩ + R        (Cauchy–Schwarz)
  cos(q, v) = 1 − ‖q̂ − v̂‖²/2               ≤ 1 − max(0, ‖q̂ − c‖ − R)²/2   (triangle)
```

taking the smaller of the two. Neither form assumes a unit-length centroid (centroids are
means, so they are not) nor a unit-length member (the radius is measured against the member
*direction*, which is all cosine can see). The bound is absolute, unlike the RaBitQ per-member
bounds it is compared against, and it was checked against brute force in a test.

Adaptivity was bounded at **two waves**, never a probe→look→decide→probe loop, which would turn
one coalesced wave into N and break INV-7: probe a seed (a quarter of the probe set), compute
`tau` from it, then use the bound — which needs only resident data — to decide the remainder in
one more wave. The chosen set is a *subset* of the fixed-fraction probe, so recall cannot fall
below the baseline.

### The measurement, and why it was dropped

Measured e2e on BEIR (bge-small, 384-d, `nprobe` = half, arm depth 100):

| dataset  | clusters | probed (fixed) | probed (adaptive) | `ann_recall@10` | mean `tau` | tightest bound in the tail |
|----------|---------:|---------------:|------------------:|----------------:|-----------:|---------------------------:|
| scifact  | 72       | 36.0           | **36.0**          | 0.9893 (both)   | 0.653      | 0.979                      |
| nfcorpus | 60       | 30.0           | **30.0**          | 0.9625 (both)   | 0.548      | 0.962                      |

**It retires nothing.** Not "rarely" — **zero clusters across 623 queries**, on both datasets,
with `ann_recall@10` bit-identical to the fixed probe. The bound sits ~0.3 above the threshold
it would have to fall below.

And it is not outliers inflating `R`, which was the expected failure mode: the mean *max*
radius is 0.62 and the mean *p95* radius is 0.58, so a quantile radius — which would cost the
soundness proof — moves the bound by only ~0.04 and still retires 0.00% (measured at p90). **The
cause is dimensional, not statistical.** At 384-d these clusters have radius ~0.62 while the
query's own k-th nearest neighbour sits ~0.77 away, so every probed cluster's ball reaches into
the query's neighbourhood and none can be ruled out. Cluster geometry offers no separation at
this scale — the classic curse-of-dimensionality failure of ball-bound pruning, arriving well
before 384 dimensions.

So it cost a second serial fetch wave and bought nothing, and the code was deleted rather than
kept behind a flag: dead code that is never exercised rots, and the expensive part to recover
is the *reasoning*, which is this page. The implementation and its brute-force soundness test
are recoverable from git history (`git log -S adaptive_probe`).

**Before re-deriving this, note what would have to change.** The verdict is a property of the
*embedding space*, not of the corpus size — a bigger corpus at 384-d fails the same way. It
would take genuinely lower intrinsic dimensionality, or much tighter clusters (many more
centroids, so `R` shrinks faster than the k-th-NN distance), for the bound to start firing.
Re-measure the two numbers — mean cluster radius vs. mean distance to the k-th nearest
neighbour — *before* writing any code; if the radius is not comfortably below that distance,
the stopping rule cannot fire and nothing else matters.

**Caveats that applied to the measurement:**

- `tau` is the k-th best *lower* bound from the scan, and `Binary`'s lower bound is
  probabilistic, not absolute. A `tau` that is optimistic makes the stopping rule slightly
  over-eager. `Int8` and `F32` bounds are absolute.
- `k` is the arm depth (100), not the reported `@10`. A `tau` at k=10 would be ~0.05 higher —
  still nowhere near the bound.
- The radius was computed from the vectors the fold holds. From the second generation onward
  those are decodes of the previous `.vec` block, not the caller's original embeddings — but
  the rerank tier the query scores against is built from the same values in the same fold, so
  the bound stayed sound with respect to what the query actually ranks. Any future attempt
  inherits this constraint.
