# The vector arm

Dense retrieval over embeddings via **IVF** (inverted-file / coarse-quantized search) with a
**two-stage RaBitQ scan**: cluster the corpus with k-means, at query time probe only the nearest
few clusters, scan them with **1-bit codes** under a provable error bound, and exact-rerank only
the candidates the bound cannot rule out. Cost scales with `nprobe`, not the corpus (INV-7).

Everything is per **memory_type** and per **segment** — each has its own centroids and cluster
files, nothing shared. Code: `mlake-ivf` (k-means, vector blocks, RaBitQ, codecs),
`mlake-index/indexer.rs` + `streaming.rs` (the build), `mlake-index/query_node.rs::run_arms` +
`vector_arm_segment` (the read).

## The vector/payload split

The embedding is ~84% of a stored memory, and the vector arm scores it once and discards it. So
it is split out of the record:

- **`cluster-{i}.bin`** — the `StoredMemory` records with the **embedding stripped** (text, tags,
  timestamps, links, metadata). Read only to hydrate a hit.
- **`cluster-{i}.vec`** — the `VectorBlock`: the quantized vectors the scan reads, ~62 B/member
  (Binary) vs ~1.8 KB for the full record.
- **`rerank.idx` / `rerank.data`** — full-precision f32 embeddings, keyed by id, one record per
  block so a point-fetch reads exactly one vector. Never scanned; fetched only for the stage-2
  rerank contenders. Keeping f32 here is what lets the scan tier be 1 bit/dimension without losing
  the exact ranking.

## The `VectorBlock` and its codecs

`cluster-{i}.vec` is a self-describing block (magic `MLVB`, `FORMAT_VERSION = 3`): a 16-byte
header (`codec`, `flags`, `dim`, `count`), then the member ids, the block **mean**, an optional
tag column and `updated_at` column (pushed down for filtering — below), and the fixed-stride
**codes** as the tail. Quantized codecs encode the *residual* `v − mean`; the mean's exact
contribution is computed once per query. Three codecs (`bytes_per_vector` at dim 384):

| codec | bytes/vec | what it stores | bound |
|---|---|---|---|
| `Binary` (**default**) | `ceil(dim/8)+12` ≈ 60 | one sign bit/dim of the **rotated** residual + a corrective triple `(‖r‖, cos(r,u), ‖v‖)` | probabilistic (RaBitQ) |
| `Int8` | `dim+12` = 396 | per-vector affine quant of the (un-rotated) residual + `(offset, scale, ‖v‖)` | absolute (`≤ scale/2`) |
| `F32` | `dim*4` = 1536 | raw f32 | exact |

**Rotation** (Binary only): a deterministic randomized-Hadamard transform (3 rounds of
permute → sign-flip → normalized fast Walsh–Hadamard, segmented into powers of two to avoid
zero-padding). It is what makes RaBitQ's probabilistic bound applicable, is derived on read, and
is **never serialized** (it would cost more than the codes). Binary is ~25× smaller than F32 with
`ann_recall@10` identical to Int8 on BEIR — because the exact rerank makes the scan codec's error
irrelevant to the final ranking.

## Query time

Per segment, `vector_arm_segment`:

1. **Probe** — the `nprobe` nearest centroids by **squared-Euclidean** distance (Euclidean, to
   match assignment). Under a tag filter or `updated_at` window it instead ranks all centroids,
   keeps only clusters whose per-cluster summary could admit a match, and caps the admissible set
   at `nprobe*4`. Push-down here is load-bearing: a match in an *unprobed* cluster is lost outright.
2. **Fetch vector blocks only** — the `.vec` blocks (not the `.bin` payload), one concurrent wave
   (RT3, fan-out 8).
3. **Stage 1 — 1-bit scan** — per block, `prepare(q)` once, then per member push `(id, est, lo,
   hi)` where `est = block.score` and `(lo,hi)` are its error-bound interval; whole-block and
   per-member tag/`updated`/supersede filters applied inline.
4. **Error bound → τ → contenders** — `τ` = the k-th best **lower** bound (k = arm depth).
   Contenders = every candidate whose **upper** bound `hi ≥ τ`. This is *derived*, not an
   oversampling guess: nothing with `hi < τ` can be in the true top-k, because k candidates already
   have a lower bound ≥ τ.
5. **Stage 2 — exact f32 rerank** — point-fetch the contenders' full-precision vectors from the
   rerank tier (RT3), rescore with exact cosine (fall back to `est` if a vector is missing), sort,
   truncate to depth.

**Cross-segment + tail merge** — the shared WAL tail is scored **exactly** first (cosine over the
tail items, so a newest copy shadows an older segment's), then each segment's two-stage top-k is
merged newest-wins and truncated. Exact overall: the global top-k is a subset of the union of
per-source top-ks, so merging exact scores and truncating is exact. Winners are hydrated once,
shared across arms.

The arm returns `(MemoryId, cosine)` as `hit.dense = { present, rank, score }`; memlake does no
fusion (a server-side weighted-RRF convenience exists but the primary API is the raw signal).

### nprobe auto-selection

`nprobe = 0` (the default) means "the index decides": **half the clusters, floor 8, cap 64**. A
fixed constant made recall depend on corpus size — 8 clusters is 11% of a small index and a
rounding error on a large one. On scifact this moved `ann_recall@10` from 0.859 (quarter) to
0.963 (half). The wire field remains as an escape hatch. (The cap binds at scale — 1M docs ≈ 1000
clusters → 64 probed ≈ 6% — which is untested territory.)

### Optimizations

- **Payload/vector split** — the scan reads ~60 B/member, not the ~1.8 KB record.
- **Asymmetric encoding** — the query stays full-precision f32 against 1-bit/8-bit codes; against
  1-bit the dot is a branch-free signed accumulation.
- **Estimator on `q_perp`** — the query with its along-mean component removed (then rotated), so
  noise scales with the *discriminating* part of the query rather than the shared mean (stated to
  ~3× recall@10).
- **Tag + `updated_at` push-down into the block** — member tags and write times ride in the
  `.vec` block, so a non-matching member never takes a top-k slot; and the per-cluster summary
  carries the tag set + write-time span so a selective filter isn't starved out of the probe set.
- **Binary bound is probabilistic** — RaBitQ Theorem 3.2, `RABITQ_EPSILON = 5.0` → per-member
  failure ≈ 7.5e-6. One rotation serves a whole block, so failures correlate within a block; a
  caller needing a hard guarantee should use Int8/F32, whose bounds are absolute.

## Build

- **Centroid training** — `k = round(√N)` k-means++ + Lloyd iterations; over `TRAIN_SAMPLE_CAP
  (50k)` it trains on a deterministic stride sample (15 iters) but **assigns all N**. Distance is
  squared-Euclidean via a **fixed-lane SIMD** accumulator kept bit-deterministic for G-6.
- **Assign + `local_split`** — every item to its nearest centroid; a 2-means splits any cluster
  > 8× the average size, so no cluster file dominates a read.
- **Cluster + vector-block writing** — per cluster, the embedding-stripped records to
  `cluster-{i}.bin` and `encode_with_columns(codec, …)` (with tag + `updated_at` columns) to
  `cluster-{i}.vec`, in the same member order; then the rerank tier and the per-cluster tag
  summary.
- **Segments, not copy-forward.** Every fold writes a **fresh** segment with centroids trained
  from scratch over its slice — there is no assign-only / copy-forward reuse (immutable object
  storage rules out in-place mutation, and the segmented model makes incrementality come from
  *new segments + compaction*, §[ARCHITECTURE](../ARCHITECTURE.md)). Codec migration is therefore
  incremental by the segment lifecycle: a flush writes L0 in the current codec, a compaction
  re-encodes what it merges, and blocks stay self-describing so a mixed-codec index reads
  correctly. The **streaming** fold trains on a reservoir sample, reads the WAL once, and feeds
  clusters/rerank through bounded external sorts.

---

## Adaptive probing — tried, measured, rejected (negative result)

**Status: implemented, measured on BEIR, and removed.** No code, no env flag, no radii on the
centroid table (`Centroids` carries only `vectors/sizes/dim`). This is the record so nobody
re-derives it. The stopping rule is *sound*; it simply never fires at these dimensions.

`nprobe` is a fixed fraction of the cluster count. The fraction is a guess — at a quarter,
nfcorpus measured `ann_recall@10` 0.86 and scifact 0.95, the same fraction nine points apart,
because the right number depends on how a corpus clusters, not how many clusters it has. The
principled replacement is a **stopping rule**: probe nearest-first, stop when the k-th best score
in hand beats the best any unprobed cluster could yield.

### The bound (sound — never the problem)

Give each centroid a radius `R = max ‖v̂ − c‖` over its members (`v̂ = v/‖v‖`). With `q̂ = q/‖q‖`,
no member of cluster `i` can score better than

```
  cos(q,v) = ⟨q̂,v̂⟩ ≤ ⟨q̂,c⟩ + R                         (Cauchy–Schwarz)
  cos(q,v) = 1 − ‖q̂−v̂‖²/2 ≤ 1 − max(0, ‖q̂−c‖ − R)²/2   (triangle)
```

taking the smaller. Absolute (unlike the RaBitQ per-member bounds), checked against brute force.
Adaptivity was bounded at **two fetch waves** (probe a seed → compute τ → decide the rest from
resident data), and the chosen set was a *subset* of the fixed probe, so recall couldn't fall
below baseline.

### The measurement, and why it was dropped

Measured e2e on BEIR (bge-small, 384-d, `nprobe` = half, depth 100):

| dataset | clusters | probed fixed | probed adaptive | `ann_recall@10` | mean τ | tightest tail bound |
|---|--:|--:|--:|--:|--:|--:|
| scifact | 72 | 36.0 | **36.0** | 0.9893 (both) | 0.653 | 0.979 |
| nfcorpus | 60 | 30.0 | **30.0** | 0.9625 (both) | 0.548 | 0.962 |

**It retires nothing** — zero clusters across 623 queries, recall bit-identical. The cause is
**dimensional, not statistical**: at 384-d these clusters have radius ~0.62 while the query's
k-th nearest neighbour sits ~0.77 away, so every probed cluster's ball reaches into the query's
neighbourhood and none can be ruled out — the classic curse-of-dimensionality failure of
ball-bound pruning. (Not outlier-driven: mean max radius 0.62 vs mean p95 0.58, so a quantile
radius moves the bound ~0.04 and still retires 0%.)

So it cost a second serial fetch wave for nothing, and was deleted rather than kept behind a flag.
The implementation and its soundness test are in git history (`git log -S adaptive_probe`).

**Before re-deriving this:** the verdict is a property of the *embedding space*, not corpus size —
a bigger corpus at 384-d fails the same way. Re-measure mean cluster radius vs mean k-th-NN
distance *first*; if the radius isn't comfortably below that distance, the stopping rule cannot
fire.
