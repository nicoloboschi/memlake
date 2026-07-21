# Scaling to 10M items / namespace

Target: **10M items in one namespace, handled comfortably.** This document records what
that target changes about the architecture and the sequenced plan to get there. It is the
authority on scale decisions; `DECISIONS.md` covers everything else.

## The numbers that drive everything

At 10M items, 384-dim f32:

| Quantity | Size |
|---|---|
| vectors | ~15 GB (30 GB at 768-dim) |
| one full generation (vectors + text + links + postings) | ~25–50 GB |
| centroids (√N ≈ 3,200) | ~5–10 MB |
| items per cluster | ~3,100 |
| one cluster file | ~5–10 MB |
| links (Hindsight ratio ~50/item) → radj | ~500M edges, multiple GB |
| pk index | ~200+ MB |

The cluster numbers are ideal for the object-storage model: nprobe=8–32 → 8–32 clusters →
80–300 MB, **one coalesced roundtrip**. The generation-scale numbers are fatal for anything
that touches the whole generation per query or per fold.

## What flips from "deferred" to "wrong" at 10M

1. **Full-load query node is impossible, not lazy-pending.** Cold open = 25–50 GB download
   + k-means retrain per snapshot. Per-probe ranged GETs are the *only* viable path, and
   the resident representation must be mmap'd cache files, not owned `HashMap<ItemId,
   StoredItem>` of cloned items. → **Phase 1.**
2. **Full-rebuild indexing is a COGS problem.** Rewriting 25–50 GB per fold at hourly folds
   ≈ 1 TB/day of S3 PUTs *per active bank*. Write amplification is gross margin. Needs
   copy-forward-by-reference (unchanged cluster files are referenced, not rewritten) +
   assign-only + SPFresh-lite local split. → **Phase 3.**
3. **`pk.idx` and `radj` as whole-read JSON blobs are wrong.** They must be sorted binary
   with sparse block indexes: a lookup = "cached ~1 MB sparse index + one ranged GET,"
   never a full fetch. Same SSTable discipline as cluster files. → **Phase 2.**

## Two gaps that only appear at this scale

4. **Filtered ANN — the biggest genuinely-missing piece.** Every real query filters (bank,
   fact_type, tags, time). Post-filtering IVF collapses at 10M with a selective filter:
   nprobe=8 → ~25k candidates → ~250 survive a 1% tag filter → filtered recall dies, and
   cranking nprobe blows the byte budget. Fix, in order:
   - (a) **partition IVF by fact_type** — retrieval is per-fact_type anyway, so this is free
     selectivity; → **Phase 4a.**
   - (b) per-generation **tag/time roaring bitmaps**, intersected *before* cluster fetch so
     the planner skips zero-match clusters and raises nprobe cheaply when selective; → **4b.**
   - (c) per-cluster filter summaries (tag bloom, time min/max) in `centroids.bin` so
     pruning costs zero extra roundtrips. → **4c.**
5. **Cost as a first-class benchmark metric.** The pitch is "lowball Postgres," so the bench
   must output **$/GB-month stored, $/1k queries (cold & warm), $/GB ingested**, computed
   from counted S3 requests × real pricing + cache amortization, gated like recall. →
   **Phase 5.**

## Smaller scale-driven notes

- O(N²) link derivation is 10¹⁴ comparisons at 10M — incremental-only via the warm IVF is
  mandatory (the derivation is already incremental as of the graph-wipe fix; the O(N)
  per-new-item scan must become a warm-IVF top-k query). → folded into Phase 3.
- **Sharding escape hatch**: keep the manifest schema from precluding a `shards[]` array
  (multiple internal indexes behind one WAL) for the bank that hits 30M. Don't build it;
  don't preclude it. → **Phase 0** (schema only).
- **Purge SLA**: GDPR erasure needs "deleted data physically gone within N days," which
  requires *scheduled* per-namespace compaction, not opportunistic LSM laziness. → **Phase 6.**

## What does NOT change

Centroid + exact-rerank is right (10M is comfortably single-index for IVF). The WAL/CAS
layer needs nothing new. The graph arm's one-hop bound is scale-independent by design. The
small-uniform-items assumption keeps FTS cheaper than the document case. **The architecture
is right for this market — the repo just has to stop cheating on lazy reads, incremental
writes, range-readable secondary structures, filtered ANN, and priced benchmarks.**

## Sequence

- **Phase 0** — manifest schema forward-compat (`shards[]`), so nothing later is blocked.
- **Phase 1** — lazy per-probe query node: load centroids only, ranged-GET the probed
  clusters, use the *published* centroid geometry (also fixes measured-recall ≠
  published-recall), WAL tail as a small brute-forced overlay. Wire `DiskCache` into
  `Store` for the warm path.
- **Phase 2** — range-readable binary `pk.idx` and `radj` (sorted + sparse block index),
  lazily read via `get_range`.
- **Phase 3** — copy-forward-by-reference + assign-only indexing + SPFresh-lite local split;
  incremental link derivation via warm IVF; recall-vs-churn benchmark at 10M synthetic.
- **Phase 4** — filtered ANN: (a) fact_type partition, (b) tag/time roaring bitmaps,
  (c) per-cluster filter summaries in centroids.
- **Phase 5** — cost metrics in the bench harness, gated.
- **Phase 6** — scheduled compaction + purge SLA.
