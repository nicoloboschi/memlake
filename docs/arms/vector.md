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
