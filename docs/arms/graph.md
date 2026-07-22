# The graph arm

Bounded one-hop **link expansion**: given seeds from the vector arm, reach one hop out through
three independent relations and merge their scores. A behavioural port of Hindsight's
`LinkExpansionRetriever` (SPEC §7). The hard rule is that graph retrieval must never become an
unbounded chain of object-storage reads — exactly one hop, per-entity fan-out capped, timeout
fallback, never recursion.

The three relations:

* **entity** — memories sharing an entity id with the seed set, scored by shared count;
* **semantic** — the indexer-derived kNN graph (seeds' inline outgoing links + incoming links
  from reverse adjacency);
* **causal** — the same mechanics over client-supplied causal edges.

Relevant code: `mlake-graph` (`retriever.rs` expansion, `scorer.rs` math, `radj.rs` reverse
adjacency), `mlake-index/sstable.rs` (`EntityTable`, `RadjTable`, `PkTable`),
`mlake-index/query_node.rs::graph_arm` (the read).

---

## Write path

The graph arm persists **three** range-readable SSTables per generation, each the same shape
(a small whole-loaded `.idx` + a block-structured `.data` read by bounded ranged GET). Built in
`indexer.rs` and published atomically with the generation (INV-2).

1. **Derive semantic links** (`indexer.rs::derive_links_ivf`). For each *new* memory, find its
   nearest neighbours (probe the IVF, exact cosine over probed clusters), keep the top
   `MAX_SEMANTIC_OUT` with cosine ≥ `SEMANTIC_LINK_THRESHOLD`, and store them **inline** on the
   memory as `semantic_out`. This is incremental — only new memories are linked against the
   current corpus (`O(new · neighbourhood)`), not a full O(N²) rebuild — and parallelized with
   rayon. Carried-forward memories keep the links they were indexed with. **memlake derives its
   own semantic links; it does not ingest external link tables.**

2. **Reverse adjacency — `radj.idx` / `radj.data`** (`RadjTable::build`). The inverse of the
   inline forward edges: `target → [incoming edges]`, over both semantic and causal edge kinds
   (each edge tagged with its kind + weight). Lets the read walk edges *backward* (who links to
   this seed) with one ranged read, without scanning.

3. **Entity postings — `entity.idx` / `entity.data`** (`EntityTable::build`). `EntityId →
   sorted [MemoryId]`, i.e. every memory that carries each entity. Built from every item's
   `entity_ids` (16-byte `EntityId`s). **This is what makes the entity arm real** — without it
   the arm could only reconnect memories already fetched by the vector probe; with it, the arm
   finds entity-sharers anywhere in the corpus via one bounded ranged read.

4. **Primary key — `pk.idx` / `pk.data`** (`PkTable::build`). `MemoryId → cluster index`.
   Shared infrastructure (also used for live doc counts and inline-payload hydration), but the
   graph arm is its main reader: after expansion yields candidate ids, pk maps each to the
   cluster it lives in so the candidate can be fetched.

All four (`radj`, `entity`, `pk`, plus the inline `semantic_out`/`causal_out` that ride in the
cluster files) are listed in `GenerationFiles` and kept live by GC (`all_paths`).

---

## Read path

Driven by `graph_arm` in `query_node.rs`. The graph arm runs only when a `vector` was given
(it needs dense seeds) and `graph_top_k > 0`.

1. **Seeds.** The top ~20 of the vector arm's ranking, taken from the memories the vector arm
   already fetched (`by_id`) — so the seeds' inline `semantic_out` / `causal_out` come for free,
   no extra read.

2. **Two coalesced batch reads, issued together (one wave, RT4):**
   - `radj.incoming_batch(seed_ids)` — every seed's incoming semantic/causal edges, one
     request instead of a ranged GET per seed;
   - `entity.candidates_batch(seed_entities, cap)` — the memories sharing each seed entity,
     capped per entity (SPEC §7.2's bounded posting prefix).

3. **Collect candidates (`wanted`).** The union of: seeds' inline outgoing targets (semantic +
   causal), the radj incoming sources, and the entity postings. Drop ids already materialized
   or tombstoned.

4. **Resolve + fetch.** `pk.lookup_batch(wanted)` maps the candidates to their clusters (one
   coalesced read), then `fetch_clusters` materializes the needed clusters (concurrent, cached
   by path). Now every candidate is a full `StoredMemory` in memory.

5. **Expand + score** (`mlake-graph::retrieve`, `scorer.rs`). Three arms score independently and
   merge **additively** into an activation in `[0, 3]`:
   - **entity:** `tanh(shared_entity_count · 0.5)` — saturating, so 0→1 shared matters far more
     than 3→4. Candidates come from the persisted postings (step 2), materialized in step 4.
   - **semantic / causal:** the **max** edge weight reaching the candidate (two weak links are
     not as strong as one strong link), over the seeds' inline outgoing edges and the radj
     incoming edges.
   Seeds are excluded from the output; results are ranked, tie-broken by id (deterministic,
   G-2/G-3), and truncated to `graph_top_k`. Dangling/tombstoned edges resolve to nothing and
   vanish with no cleanup (SPEC §7.7).

The arm returns `(MemoryId, activation)` pairs, surfaced as `hit.graph = {present, rank,
score}`. As with the other arms memlake does no fusion — the client combines the raw graph
activation with the dense and text signals.

### Roundtrips, bounds & caching

- **One hop only.** Expansion never recurses; the cost is bounded by the seed count, the
  per-entity cap, and the batch reads — independent of graph shape (INV-7). A timeout fallback
  drops the entity arm and serves semantic + causal only (SPEC §7.6).
- **Cold:** the graph arm's reads all land in **RT4** — `radj` + `entity` coalesced, then `pk`,
  then the candidate cluster fetch. All are batched/concurrent, keeping the whole cold query
  within the 4-roundtrip budget.
- **Warm:** radj/entity/pk blocks and candidate clusters are immutable and cached by
  `(path, range)`, so a warm graph query is **0 roundtrips** — verified in the benchmarks
  (`graph_pk` 795µs→2µs warm).

### Where this still differs from Hindsight

- **Semantic links are memlake's own** derived kNN, not Hindsight's explicit `memory_links` —
  a deliberate choice (the goal is memlake owning the memories *and* links, not mirroring PG).
  So the semantic arm won't be edge-identical; the entity + causal arms carry the parity.
- **`source_memory_ids`** (observation ↔ source-fact) has no analogue yet — Hindsight's
  `expand_observations` walks it bidirectionally. Planned as a new inline + reverse-adjacency
  relation (TODOS §3 Plan C).
