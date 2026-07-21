# TODOS — what memlake still needs to back Hindsight's memories

Context: a first slice of the integration lives in the Hindsight worktree
`hindsight-wt8`, branch `feat/memlake-provider`. It adds a `MemoriesProvider`
seam (`hindsight_api/engine/memories/`) with two implementations — `postgres`
(the historical SQL path, unchanged and still the default) and `memlake`.

The memlake provider currently plays a **narrow role**: it indexes facts and
answers the dense + full-text arms, returning ranked ids that Hindsight hydrates
from Postgres. Postgres stays the authoritative row store. Everything below is
what stands between that and memlake actually owning the memories slice.

Ordered by what blocks what.

---

## 0. ~~Blocking — the Python client cannot talk to the current server~~ ✅ DONE

The `.proto` was rewritten (single `Query` RPC, `memory_types` repeated, per-arm
`vector_top_k`/`text_top_k`/`graph_top_k`, `ArmScore`, no server-side fusion) and
`clients/python/memlake_client/client.py` is now fully migrated to it.

- [x] **`_hits()` / `Hit` reconciled.** `_hits()` builds `Hit(id, memory_type,
      dense, text, graph)` where each of `dense/text/graph` is an
      `Arm(present, rank, score)`. No more `score` / `contributions`.
- [x] **One query method, current protocol.** `query_metered` / `query_multi` /
      `query_multi_metered` are gone. `Query` sends `memory_types` (repeated) and
      per-arm `*_top_k` / `nprobe`; `QueryConfig` / `QueryMultiRequest` no longer
      exist in the proto.
- [x] **Signature settled** exactly as proposed:
      `query(namespace, *, vector=…, text=…, memory_types=[...], tags=…,
      tags_mode=…, vector_top_k=…, text_top_k=…, graph_top_k=…, nprobe=…,
      consistency=…) -> list[Hit]`. The shared roundtrips are on
      `client.last_roundtrips`.

**Hindsight follow-up:** the provider's stopgap in
`engine/memories/memlake.py:initialize` (driving the generated stub directly) can
be deleted and pointed back at the wrapper's `query()`.

Smaller client gaps (non-blocking, still open):

- [ ] `memory()` accepts no `timestamps` and no `causal_out`, so callers mutate
      the protobuf message after construction. Add both as kwargs.
- [ ] `proof_count` defaults to `0` in `memory()`; Hindsight's column defaults to
      `1`. Pick one, or make the parameter required, so the discrepancy is not
      silent.
- [ ] No `grpc.aio` client. Hindsight is fully async and currently wraps every
      call in `asyncio.to_thread`, which costs a thread hop per retain batch and
      per recall.

---

## 1. Protocol gaps — memlake cannot yet return a memory

A `Hit` is now `{id, memory_type, dense, text, graph, memory}` — the **memory is
returned inline** with every search hit (see below). Remaining gaps are the
non-search access paths (scan, counts, delete-by-predicate, partial update).

- [x] **Return the memory inline on `Hit`.** Each hit now carries a
      `MemoryPayload` (`text, tags, proof_count, entity_ids, timestamps,
      causal_out, metadata`) — the server already has it materialized to score the
      candidate, so recall gets it with **no extra object-storage read and no
      second round trip**. Combined with the new opaque `metadata` bag (see §2),
      `context / document_id / chunk_id / arbitrary JSON-as-string` ride along too,
      and `fact_type` maps from `memory_type`. The embedding vector is the only
      stored field not returned (large; the client has it).
- [ ] **`Get(namespace, ids) -> [Memory]`** — still worth adding for *non-search*
      hydration (curation opening a specific unit, export). The machinery is ready
      (pk lookup → cluster fetch → the same `MemoryPayload` conversion); it just
      needs an RPC. Lower priority now that recall hydrates inline.
- [ ] **Scan / list with pagination and ordering.** Hindsight's curation UI
      (`list_memory_units`), export, and `get_graph_data` all page through
      memory units filtered by `fact_type` / `document_id` / text ILIKE /
      `consolidated_at IS NULL`, ordered by `mentioned_at DESC, created_at DESC`.
- [ ] **Counts / aggregates.** Bank stats (`GROUP BY fact_type`), consolidation
      backlog gauges, per-document unit counts, and the memories timeseries
      (`date_trunc` buckets) are all `COUNT(*)` queries today.
- [ ] **Delete by predicate.** Only tombstone-by-id exists. Hindsight deletes
      "all units for this document", "all units for this bank", "all observations
      in this bank". The provider currently issues a `RETURNING id` on the
      Postgres delete and mirrors the ids — correct, but it means the row store
      must be consulted to delete from the index.
- [ ] **Partial update.** `Patch` carries only `proof_count_delta`. Hindsight
      mutates `text`, `embedding`, `tags`, `occurred_start/end`, `mentioned_at`,
      `consolidated_at`, `consolidation_failed_at`, `edited_at` — mostly from
      consolidation and curation. Today each would need a full re-upsert.
- [ ] **`DeleteNamespace`.** `delete_bank` has no way to drop a namespace.

---

## 2. Model gaps — fields memlake does not carry

`Memory` has `id, key, vector, text, memory_type, tags, proof_count, entity_ids,
timestamps, causal_out`. The `memory_units` row has considerably more, and some
of it participates in retrieval:

- [ ] **`text_signals`** — Hindsight enriches the BM25 document with entity names
      and spelled-out date tokens ("May 8 2023") so keyword search can hit them.
      memlake indexes `text` only, so the memlake full-text arm is strictly weaker
      than the Postgres one on entity/date queries. Either accept a separate
      indexed-text field, or index `text + signals`.
- [ ] **`source_memory_ids`** — the observation → source-fact edge. Hindsight's
      `expand_observations` walks it in both directions (find the observations
      backed by these facts; find the facts behind these observations). No
      analogue exists; `causal_out` is a different relation.
- [x] **`context`, `metadata` (arbitrary JSON), `document_id` / `chunk_id`,
      `observation_scopes`, `access_count`, `created_at` / `updated_at`** — all of
      these can now ride in the opaque **`metadata`** (str→str) bag: memlake stores
      them verbatim and returns them inline on every hit, without modelling each
      one. Caveat: metadata is **not queryable or indexable** — so the
      delete-by-document predicate (§1) and any push-down filter on these fields
      still needs either a real indexed column or a scan. Put display/hydration
      fields in metadata; promote a field to first-class only when retrieval must
      filter on it.
- [ ] **Invalidated/archive state.** Hindsight models curation state structurally:
      live rows in `memory_units`, invalidated ones moved to
      `invalidated_memory_units` (a `LIKE memory_units` clone) and revertible.
      memlake has tombstones, which are one-way.
- [x] **`entity_ids` width — DONE (§3 Plan A).** Widened to a 16-byte `EntityId`
      (mirrors `MemoryId`, on the wire as `bytes`). The lossy UUID→u64 narrowing and
      its silent collisions are gone; drop `memlake.py:_entity_id_to_u64` and pass
      `uuid.bytes`.
- [ ] **`memory_type` is a `u8`**; Hindsight's `fact_type` is a string enum
      (`world` / `experience` / `observation`). The mapping lives in
      `engine/memories/base.py:FACT_TYPE_TO_MEMORY_TYPE`. Fine for now, but it
      needs to be a shared registry before anyone adds a fourth type.

---

## 3. Retrieval gaps — arms Hindsight has that memlake does not

- [ ] **Temporal arm.** Hindsight has a whole second retrieval path
      (`retrieve_temporal_combined`): select entry points whose
      `COALESCE(occurred_start, mentioned_at, occurred_end)` falls in a window,
      spread over links with BFS, and score by temporal proximity with coverage
      buckets across the window. memlake stores `Timestamps` but exposes no
      time-window query at all. This arm still runs against Postgres.
- [ ] **Entity-expansion arm — the central graph gap.** Hindsight fans out over
      `unit_entities` (a persisted entity→unit posting, LATERAL-capped) *and*
      `memory_links`, then expands observations via `source_memory_ids`. memlake
      has **no persisted entity index at all**: `query_node.rs:580` builds the
      entity map in-memory from `by_id` (only the already-fetched memories), so
      the entity arm can only reconnect the vector-probe neighborhood, never find
      an entity-sharer in an unprobed cluster. Plan below.

  ### Plan: make the graph arm real (entities + observation edges)

  - **A. Widen entity ids to 16 bytes — DONE.** `entity_ids: Vec<u64>` → a 16-byte
        `EntityId` (mirrors `MemoryId`), killing the lossy UUID→u64 narrowing and
        its silent collisions. Touches core, graph, indexer, datagen. Format
        change (indexes rebuild). Mechanical, low-risk. *Do first.*
  - **B. Persisted entity posting SSTable — DONE.** At index time we build
        `entity.idx` + `entity.data` mapping `EntityId → sorted [MemoryId]`, the
        same SSTable shape as radj/pk. The query node's `entity_candidates` reads
        it range-bounded with the per-entity cap (SPEC §7.2's bounded prefix).
        This turns the degenerate entity arm into a true entity-expansion arm that
        finds sharers anywhere in the corpus. Adds one bounded roundtrip wave, like
        radj. Rebuilt per fold from the corpus (as radj is).
  - **C. Observation↔fact edges (`source_memory_ids`).** A new relation: carry
        `source_memory_ids` inline on the memory (forward) + reverse adjacency
        (like radj) for the backward walk, exposed as a bidirectional expansion in
        the graph arm. Needed for `expand_observations` parity.
  - **D. Semantic-link provenance + differential.** Decide whether memlake keeps
        deriving kNN links or ingests Hindsight's explicit `memory_links` (as
        client-supplied edges). Then wire the G-2 differential of memlake's graph
        arm against Hindsight's `LinkExpansionRetriever` on identical input — the
        scorer is already a G-3-verified port, so only the candidate sources need
        to converge.
- [ ] **Tag groups.** memlake has five flat modes (`ANY/ALL/ANY_STRICT/
      ALL_STRICT/EXACT`). Hindsight also supports nested AND/OR/NOT tag *groups*.
      The provider applies those in Python after hydration, which means they
      filter *after* `top_k` and can therefore return fewer rows than the SQL
      path.
- [ ] **`updated_at` range push-down.** Same problem: recall's
      `created_after`/`created_before` window is applied post-ranking
      (`memlake.py:_in_updated_range`).
- [ ] Confirm dense scores are cosine similarity on the same scale Postgres
      produces (`1 - (embedding <=> query)`). The provider applies Hindsight's
      `semantic_min_similarity` / `bm25_min_score` floors to memlake's raw arm
      scores on that assumption; if the scales differ, the floors silently cut
      the wrong things.

---

## 4. Correctness and operations

- [ ] **No cross-store atomicity.** Facts are indexed inside the retain
      transaction (`retain/fact_storage.py:index_facts_in_provider`). If that
      transaction rolls back, memlake keeps entries whose rows never existed.
      Recall skips ids it cannot hydrate, so the failure mode is a wasted slot
      rather than a wrong answer — but it accumulates. Options: index after
      commit (needs an outbox), or a compensating sweep.
- [ ] **Backfill.** No path to migrate an existing Postgres bank into a memlake
      namespace. Needed before this can be switched on for anything real, and it
      is also how a benchmark comparison would be set up.
- [ ] **Reconciliation.** Nothing detects drift between the row store and the
      index (rows with no index entry, index entries with no row).
- [ ] **Auth / TLS.** The gRPC surface is unauthenticated. The proto describes it
      as internal east-west, which is fine as a deployment assumption, but it
      should be stated as a requirement rather than left implicit.
- [ ] **Multi-tenancy.** Hindsight isolates tenants by Postgres schema, and banks
      by `bank_id`. The provider maps a bank to a namespace with an optional
      configured prefix (`HINDSIGHT_API_MEMLAKE_NAMESPACE_PREFIX`). Decide
      whether namespaces are the tenancy boundary and what isolation they
      actually guarantee.
- [ ] **Read-your-writes cost.** `STRONG` scans the WAL tail. Retain-then-recall
      in one request is a normal Hindsight pattern; measure what that costs when
      the tail is long.

---

## 5. Hindsight-side follow-ups (not memlake's job, but blocking parity)

Write paths that still touch `memory_units` directly and are **not** mirrored to
the provider. Each is a place where the index would go stale:

- [ ] Consolidation — creates, updates, merges and deletes observations with raw
      SQL (`engine/consolidation/consolidator.py`).
- [ ] Curation — `update_memory_unit` edit / invalidate / revert
      (`engine/memory_engine.py:6654`).
- [ ] `delete_memory_unit`, `delete_bank`, `delete_document`,
      `clear_observations`, `update_document` tag rewrites.
- [ ] The document importer (`engine/transfer/importer.py`).
- [ ] Graph and temporal retrieval still query Postgres regardless of provider —
      only the dense + full-text arms are routed.

The broader shape of the problem: `DataAccessOps` covers only a fraction of
`memory_units` access. Roughly 60 raw statements across `memory_engine.py`,
`retrieval.py`, `link_expansion_retrieval.py`, `consolidator.py`,
`fact_storage.py` and `export.py` hit the table directly. Full provider ownership
means pushing those behind the interface first.
