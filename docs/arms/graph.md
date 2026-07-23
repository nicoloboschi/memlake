# The graph arm

Bounded one-hop **link expansion**: given seeds from the vector arm, reach one hop out through
three independent relations and merge their scores. A behavioural port of Hindsight's
`LinkExpansionRetriever`. The hard rule: graph retrieval must never become an unbounded chain of
object-storage reads — exactly one hop, per-entity fan-out capped, timeout fallback, never
recursion (INV-7).

The three relations:

* **entity** — memories sharing an entity id with the seed set, scored by shared count;
* **semantic** — the write-time-derived kNN graph (seeds' inline outgoing links + incoming links
  from reverse adjacency);
* **causal** — the same mechanics over client-supplied causal edges.

Code: `mlake-graph` (`retriever.rs` expansion, `scorer.rs` math, `radj.rs` reverse adjacency),
`mlake-index/sstable.rs` (`EntityTable`, `RadjTable`), `mlake-index/query_node.rs::graph_arm`
(the read), `mlake-index/indexer.rs::derive_links_for_write` (write-time derivation),
`mlake-index/streaming.rs::feed_radj` (carrying links into `radj` during the fold).

## Query time

`graph_arm` runs only when a `vector` was given (it needs dense seeds) and `graph_top_k > 0`.

1. **Seeds — with a configurable similarity floor.** The vector arm's ranking (exact cosine,
   sorted) is first **filtered to hits at cosine ≥ `graph_seed_min` (default 0.3)**, then the top
   20 are taken. Expanding from a barely-relevant seed only pulls its (equally off-query)
   neighbours into fusion, so weak seeds are dropped before the hop. The floor is
   `QueryConfig::graph_seed_min_similarity` / `ArmDepths::graph_seed_min`, exposed over gRPC as
   `graph_seed_min_similarity` (unset → server default 0.3, `0` → seed from every dense hit),
   matching Hindsight's `_find_semantic_seeds(threshold=0.3)`. Seeds come from the memories the
   vector arm already fetched, so their inline `semantic_out` / `causal_out` are free.

2. **One-hop expansion over three signals** (`mlake_graph::retrieve`):
   - **entity** — union of all seed entities; a candidate's score is how many distinct
     seed-entity postings it appears in, read via `entity.candidates_batch` (capped at
     `per_entity_cap = 200` per entity).
   - **semantic** — the seeds' inline `semantic_out` plus incoming `Semantic` edges from each
     segment's `radj` (unioned across segments — an edge lives where its target lives).
   - **causal** — the seeds' inline `causal_out` plus incoming `Causal` radj edges.

3. **Structural scoring — no query-similarity rerank.** This is the key property: candidates are
   scored purely by graph structure, never re-ranked against the query vector.
   - entity: `tanh(shared_count · 0.5)` — saturating (1→0.46, 2→0.76, 3→0.91, 4→0.96).
   - semantic / causal: the **max** edge weight reaching the candidate (two weak links are not
     as strong as one strong link).
   - additive merge across signals → activation in `[0, 3]`. Seeds excluded; ranked desc,
     tie-broken by id (deterministic, G-2/G-3); truncated to `graph_top_k`.

4. **No candidate hydration.** Scoring runs over a `LazyGraphSource` — the entity postings, radj
   edges, and a liveness (`exists`, i.e. not-tombstoned) check — with **no memory fetched to be
   scored**. Only the ≤budget *ranked* results are hydrated, and only when a tag filter needs
   their tags; without a filter the arm fetches nothing here. (This removed the old whole-corpus
   candidate hydration — previously every entity candidate, up to 200 per entity, was read from
   its cluster just to be scored and truncated.)

The arm returns `(MemoryId, activation)`, surfaced as `hit.graph = { present, rank, score }`;
memlake does no fusion.

**Note on entity-less data.** With no entities/causal edges (e.g. BEIR corpora), the arm reduces
to flat semantic-neighbour weights (all ≥ 0.7, near-tied) — its discriminating signal is entity
convergence. This is faithful to Hindsight, which is built for entity-rich production; on
entity-less data at equal fusion weight it can add noise. See DECISIONS/TODOS.

### Bounds & caching

- **One hop only.** Cost is bounded by seed count, the per-entity cap, and the batch reads —
  independent of graph shape. A timeout fallback drops the entity arm and serves semantic +
  causal only.
- **Cold:** `radj` + `entity` are coalesced batch reads in RT4; structural scoring adds none.
- **Warm:** radj/entity blocks are immutable and cached by `(path, range)`, so a warm graph
  query is effectively 0 roundtrips.

## What is stored

- **Inline `semantic_out`** on each memory — up to `MAX_SEMANTIC_OUT (5)` `SemanticEdge
  { target, weight: f16 }`, read for free from a seed record. Inline `causal_out` is
  client-supplied `CausalEdge { target, link_type, weight }`.
- **Reverse adjacency `radj.idx` / `radj.data`** — an SSTable keyed by *target* id, holding
  incoming `InEdge { source, kind, weight: f32 }` for both `Semantic` and `Causal(LinkType)`
  kinds. (Incoming weights are full-precision f32; the inline forward weight is f16.) Built from
  every item's `semantic_out` and `causal_out`, keyed by `edge.target`. Lets the read walk edges
  *backward* with one ranged read.
- **Entity postings `entity.idx` / `entity.data`** — `EntityId → sorted [MemoryId]`, every
  memory carrying each entity. This is what makes the entity arm find sharers *anywhere* in the
  corpus, not just in the probed clusters.

**memlake derives its own semantic links; it does not ingest external link tables.**

## Semantic-link derivation (write time)

Links are derived on the **write path, before the WAL commit** (`derive_links_for_write`), not by
the fold. This is the load-bearing invariant: **the index is a pure speed optimization, never a
correctness dependency.** Because the links are set on each `Memory` before it is committed, they
travel in the WAL as intrinsic data (`semantic_out`) — so a query over the un-indexed WAL tail is
*already* correct, and the fold only reorganizes the links for faster reads, it never invents them.

For each new memory in a write batch, neighbours come from two sources, merged and truncated to the
top `MAX_SEMANTIC_OUT (5)` at cosine ≥ `SEMANTIC_LINK_THRESHOLD (0.7)`:

1. **Committed corpus** — one exact-scored vector query against the current snapshot (index + tail),
   at `nprobe = 16`, depth `MAX_SEMANTIC_OUT + 4` so self and any tombstoned hit can be dropped and
   the slots still fill.
2. **Within the same batch** — an all-pairs cosine over the batch's own memories, which are not yet
   committed and so invisible to the snapshot. This is what makes a burst of concurrent ingests link
   to *each other*, not only to the older corpus (mirrors Hindsight's bidirectional within-batch
   links). Links are one-directional new→old and bidirectional within a batch; a committed item is
   never re-derived (matching Hindsight's "compute links once at ingest").

The fold — both the in-RAM and the **streaming (external-memory)** path — carries each item's
`semantic_out` forward untouched and feeds its reverse edges into `radj` (`feed_radj`). Neither
path derives or drops links; the streaming path proves this at any scale, since the >4M-doc build
resolves items one cluster at a time yet preserves the WAL's links exactly.

## Deletes and re-ingest

A `TombstoneWhere { predicate }` op deletes every memory matching a metadata/tag/type predicate
whose last write is *older* than the entry's sequence — atomic (one entry), idempotent, and
race-closed by `write_seq` (a later re-ingest with an equal/higher seq survives). Putting it in
the same entry as a re-ingest's upserts replaces a document's facts atomically. Dangling edges to
a since-deleted target simply fail the `exists` check and vanish, with no cleanup.
