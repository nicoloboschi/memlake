# TODOS — what memlake still needs to back Hindsight's memories

Context: a first slice of the integration lives in the Hindsight worktree
`hindsight-wt8`, branch `feat/memlake-provider`. It adds a `MemoriesProvider`
seam (`hindsight_api/engine/memories/`) with two implementations — `postgres`
(the historical SQL path, unchanged and still the default) and `memlake`.

**As of the inline-payload / entity-index work, memlake mode no longer writes
anything memory-shaped to Postgres.** In that mode Hindsight skips the
`memory_units` INSERT (ids are minted client-side), skips `unit_entities` (the
unit→entity posting rides on the memory), skips all three `memory_links` writers
— temporal has no counterpart, semantic is derived by the indexer, causal rides
inline as `causal_out` — and skips the Phase-1/Phase-3 ANN passes. Recall reads
the whole result row off the inline `MemoryPayload`, so there is no hydration
query, and the graph arm comes from memlake's persisted entity index instead of
`LinkExpansionRetriever`. Postgres keeps documents, chunks, banks, operations and
the `entities` registry.

With the temporal arm and the admin RPCs (`Get` / `Scan` / `Stats`), all four
recall arms and the main read surfaces — the curation list, single-unit reads,
bank fact counts — are served by the provider too. With `DeleteNamespace`,
`DeleteByPredicate`, `Op.tombstone_where` and a full `Patch`, so are the deletes
and the curation edit: a document re-ingest now *replaces* its facts instead of
duplicating them.

What is left is **consolidation** — the one subsystem still assuming a SQL table —
plus the importer's follow-up UPDATEs and a handful of read surfaces. Ordered by
what blocks what.

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

**Hindsight follow-up — done.** The stopgap that drove the generated stub
directly is gone; the provider calls `client.query()` / `client.delete()`.

Smaller client gaps (non-blocking, still open):

- [x] `memory()` takes the timestamps as kwargs (`event_date`, `occurred_start`,
      `occurred_end`, `mentioned_at`, epoch ints). The provider passes them
      directly now.
- [ ] `memory()` still takes no `causal_out`, so the provider appends causal
      edges to the protobuf message after construction
      (`memlake.py:index_facts`). Add it as a kwarg.
- [ ] **The wrapper does not expose the admin RPCs.** `ListNamespaces`, `Stats`,
      `Get` and `Scan` are in the proto and the generated stubs, but
      `MemlakeClient` has no methods for them, so the provider builds its own
      stub off `client._channel` (`memlake.py:_stub`). Same stopgap shape as the
      pre-migration query path — worth closing the same way.
- [ ] `proof_count` defaults to `0` in `memory()`; Hindsight's column defaults to
      `1`. Pick one, or make the parameter required, so the discrepancy is not
      silent.
- [ ] No `grpc.aio` client. Hindsight is fully async and currently wraps every
      call in `asyncio.to_thread`, which costs a thread hop per retain batch and
      per recall.
- [ ] **protobuf runtime conflict — blocks running the two together.** The
      generated stubs call `ValidateProtobufRuntimeVersion(PUBLIC, 7, …)`, so they
      require protobuf ≥ 7.x. Hindsight's lockfile pins protobuf 6.33.5 (via its
      OTel deps), and importing `memlake_client` there dies with a
      `VersionError`. Installing protobuf 7 into the Hindsight venv works and the
      provider tests pass, but that is a manual override, not a resolution.
      Either regenerate the stubs against a 6.x-compatible gencode, or agree on a
      protobuf floor across both projects.

---

## 1. Protocol gaps

A `Hit` is now `{id, memory_type, dense, text, graph, temporal, memory}` — the
**memory is returned inline** with every search hit — and the admin RPCs
(`ListNamespaces`, `Stats`, `Get`, `Scan`) cover the non-search reads. What is
left here is mostly Scan's ergonomics — the write side (delete-by-predicate,
partial update, `DeleteNamespace`) is now covered.

- [x] **Return the memory inline on `Hit`.** Each hit now carries a
      `MemoryPayload` (`text, tags, proof_count, entity_ids, timestamps,
      causal_out, metadata`) — the server already has it materialized to score the
      candidate, so recall gets it with **no extra object-storage read and no
      second round trip**. Combined with the new opaque `metadata` bag (see §2),
      `context / document_id / chunk_id / arbitrary JSON-as-string` ride along too,
      and `fact_type` maps from `memory_type`. The embedding vector is the only
      stored field not returned (large; the client has it).
- [x] **`Get(namespace, ids)`** — wired to `get_memory_unit` (curation opening a
      single unit) via `provider.get_memories`.
- [x] **`Scan`** — wired to `list_memory_units` via `provider.scan_memories`. Two
      gaps remain on the Hindsight side, both from Scan being a cursor walk rather
      than a query:
  - [ ] **No push-down for `document_id`, text search, or consolidation state.**
        Hindsight filters each page in Python (`memories/reads.py:_matches`), so a
        page can come back short and `total` reports what the walk saw rather than
        a true count. `document_id` in particular is only in the un-indexed
        metadata bag — an indexed field would fix this *and* the delete-by-document
        problem below.
  - [ ] **No ordering.** The SQL path returns `ORDER BY mentioned_at DESC NULLS
        LAST, created_at DESC`; Scan walks in cluster order, so the curation list
        comes back in storage order.
  - [ ] **Offset paging costs pages.** The API takes offset/limit, Scan takes a
        cursor, so reaching an offset means walking to it. Hindsight caps the walk
        at 50 pages and logs when it truncates. Either the API moves to cursor
        paging end to end, or Scan grows a skip.
- [x] **Counts** — `Stats` wired to `_compute_bank_stats`. `TypeStats.doc_count`
      is already live (indexed generation − tombstones + WAL tail), so per-
      fact_type counts cost index metadata rather than a scan.
  - [ ] Link counts in bank stats are reported as zero: memlake derives its edges
        and does not expose them as a countable relation.
  - [ ] The memories timeseries (`date_trunc` buckets over `created_at`) has no
        equivalent and still returns empty.
- [x] **Delete by predicate — DONE, and it fixed the re-ingest bug.**
      `DeleteByPredicate` plus `Op.tombstone_where` land the whole problem. Hindsight
      deletes a document's memories by `{document_id}` from the metadata bag
      (`fact_storage.delete_document_from_provider`), used by both `delete_document`
      and the re-ingest path in `handle_document_tracking`. The lazy (`eager=false`)
      form is what makes it safe: the tombstone only removes writes older than its
      own sequence, so replacement facts written moments later survive even though
      the delete is issued first — no need to batch them into one entry.
      Verified: importing the same 19 documents three times into a live bank leaves
      exactly 239 memories each pass (it used to grow to 717).
- [x] **Partial update — DONE.** `Patch` now carries text, vector, tags,
      timestamps and a merging metadata map alongside `proof_count_delta`. Wired to
      curation's edit path (`update_memory_unit`), which reads the pre-edit memory
      through `Get` and applies the edit as one patch. Note Hindsight hands
      embeddings around as the pgvector literal `'[0.1,...]'` in places, so the
      provider parses either that or a float list.
- [x] **`DeleteNamespace` — DONE.** Wired to `delete_bank`. One caveat found in
      use: dropping a namespace that was never created returns `INTERNAL: no
      manifest`, and callers delete defensively (`delete_bank` runs before ingest to
      clear a prior run), so the provider swallows that specific error. A NOT_FOUND
      status, or treating the drop as idempotent server-side, would be cleaner than
      string-matching an INTERNAL message.
  - [ ] Dropping a namespace while the indexer is folding it is not safe — the proto
        says as much ("no snapshot atomicity across the deletes"). Observed once in a
        loop of drop-then-reimport with the indexer running at a 3s interval: the
        following `Stats` failed and that pass was lost. Either the drop should fence
        the indexer, or the docs should state the operator is responsible for stopping
        it first.

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
      analogue exists; `causal_out` is a different relation. Beyond recall, this
      also blocks *cleanup*: `delete_stale_observations_for_memories` exists to
      drop observations whose sources are being replaced, and in memlake mode it
      cannot run at all — Hindsight logs and skips it rather than querying the
      empty table and reporting a clean sweep.
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

- [x] **Temporal arm — DONE.** `temporal_from` / `temporal_to` on `QueryRequest`
      plus `Hit.temporal`: entry points whose effective time
      `COALESCE(occurred_start, mentioned_at, occurred_end)` falls in the window,
      one-hop spread, scored by proximity to the window centre — the same
      semantics as Hindsight's SQL path. `retrieve_temporal_combined` routes to it
      when the provider owns the store.
  - [ ] Hindsight's version also spreads with a multi-hop BFS and selects entry
        points across N coverage buckets so a wide window is sampled evenly rather
        than clustered at the most-similar end. memlake spreads one hop and does
        no bucketing, so results on wide windows will skew toward the query
        vector. Worth a differential before treating the two as equivalent.
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
- [ ] **IVF trains 0 clusters on real-dimension vectors — the dense arm is dead.**
      Found running LoComo end to end. A namespace of 344 Hindsight memories
      (384-dim, all-MiniLM) indexes cleanly — indexer publishes, `has_index=true`,
      `train_count` equals `doc_count` — but `cluster_count` is **0** for every
      memory_type, so the dense arm has nothing to probe and returns no hits.
      Reproduction: query with a memory's *own* vector (fetched via
      `Scan(include_vector=true)`) and `dense.present` is false for every hit;
      only the text arm returns anything. The graph arm is empty too, since it
      seeds off the dense probe.
      Suspected trigger: `n < dim`. The namespaces that do have clusters use toy
      8-dim vectors (`admin-ui-smoke`: 30 docs, dim 8 → 5 clusters); ours is 239
      docs at dim 384 → 0 clusters. Worth an assertion that training either
      produces clusters or fails loudly — silently publishing a vector index with
      no clusters looks healthy from `Stats` while serving nothing.
- [ ] Confirm dense scores are cosine similarity on the same scale Postgres
      produces (`1 - (embedding <=> query)`).
- [ ] **The default `nprobe` silently costs recall.** With clusters building
      correctly, memlake's semantic arm returned 258 of 344 candidates where the
      pgvector path returned all 344 — and LoComo accuracy dropped to 11/15.
      The cause is cluster coverage, not depth: `vector_top_k` was already 2500,
      but candidates in unprobed clusters are unreachable at any depth. Measured on
      a 344-memory bank (15 + 10 clusters):

      | nprobe | dense candidates | coverage | median query |
      |--------|-----------------|----------|--------------|
      | 0 (default) | 266 | 77% | 5.9 ms |
      | 15 | 344 | 100% | 5.2 ms |
      | 32 | 344 | 100% | 4.9 ms |

      Full coverage was *free* here (0 roundtrips either way — all cached), so the
      default is trading recall for nothing at this size. Setting
      `HINDSIGHT_API_MEMLAKE_NPROBE=32` restored both the candidate count (344) and
      accuracy (14/15, matching postgres exactly).
      Worth considering: scale the default with cluster count, or return the
      coverage actually achieved so a caller can tell a partial probe from an
      exhaustive one. Right now a 77% probe is indistinguishable from a full scan.
- [ ] **The dense arm does not see un-indexed writes.** `STRONG` is documented as
      "reflect every acked write", and the text arm does — but a query issued
      immediately after a write returns nothing from the vector arm until the
      indexer folds it (observed: benchmark imports 344 memories, queries at once,
      gets `semantic=0`; the same namespace queried minutes later returns 344).
      Retain-then-recall in one request is a normal Hindsight pattern, so this is
      the difference between "BM25-only answer" and "full answer". Either the tail
      scan should cover the dense arm, or the guarantee should be documented
      per-arm. The provider applies Hindsight's
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

## 5. Hindsight-side status

**Routed to the provider (no Postgres memory/link writes):**

- [x] Fact writes — `memory_units` INSERT skipped; ids minted by the provider
      (`retain/fact_storage.py:insert_facts_batch`).
- [x] `unit_entities` — the unit→entity posting travels on the memory as
      `entity_ids`. The `entities` registry itself stays in Postgres: it is the
      canonical name/alias store, and its UUIDs are what memlake records.
- [x] `memory_links` — all three writers skipped. Causal edges ride inline;
      semantic links are derived by the indexer; temporal links have no
      counterpart (memlake carries the timestamps instead).
- [x] Phase-1 and Phase-3 semantic ANN passes — skipped; they scanned
      `memory_units` to build links the indexer now derives.
- [x] Dense + full-text arms — served from the provider, results built from the
      inline payload, no hydration query.
- [x] Graph arm — `ProviderGraphRetriever` replaces `LinkExpansionRetriever` when
      the provider owns the links.
- [x] Temporal arm — `retrieve_temporal_combined` routes to
      `provider.temporal_search`, which sends the window as `temporal_from` /
      `temporal_to` and reads `Hit.temporal`.
- [x] `list_memory_units` — paged through `Scan` (`memories/reads.py`), with
      entity names resolved from the Postgres `entities` registry.
- [x] `get_memory_unit` — served by `Get`.
- [x] Bank fact counts (`_compute_bank_stats`) — served by `Stats`.
- [x] Deletes — `delete_bank` drops the namespace, `delete_document` and the
      retain re-ingest path predicate-delete by `document_id`, `clear_observations`
      predicate-deletes the observation type.
- [x] Curation edit — `update_memory_unit` reads the live memory through `Get` and
      applies the edit as a `Patch`.
- [x] Consolidation — refuses to run and says so, rather than reading an empty
      table and reporting a clean pass (see below).

**Still Postgres-only, so unsupported in memlake mode:**

- [ ] **Consolidation** (`engine/consolidation/consolidator.py`) — selects
      unconsolidated rows and writes observations back with raw INSERTs. `Scan`
      covers the read half now; still needs partial update (to stamp
      `consolidated_at`) and the observation↔fact relation. Currently guarded off
      with a warning, so `consolidated_at` / `consolidation_failed_at` are always
      reported as null and every unit reads as pending.
- [ ] **Curation invalidate / revert.** The *edit* case is wired. Invalidate
      moves a row to an archive table and revert moves it back; the provider has
      only one-way tombstones, so both are still unsupported and
      `list_memory_units` always reports `state: "valid"`.
- [ ] **`delete_memory_unit` needs a bank_id.** The endpoint takes only a unit id
      and resolves the bank from the `memory_units` row — with no row there is no
      bank, and a provider delete needs a namespace. Hindsight now raises a clear
      error instead of reporting "not found". Fix is an API change (thread the
      bank through), not a memlake one.
- [ ] **The document importer** (`engine/transfer/importer.py`) — writes facts
      and then patches them with follow-up UPDATEs.
- [ ] **Still-empty read surfaces**: `get_graph_data` (needs the edges as a
      readable relation), the memories timeseries, per-document unit counts, and
      document/bank export.

The shape of what remains has narrowed: retain, all four recall arms, and the
main read surfaces are behind the provider interface. What is left is
concentrated in the **mutation** paths — consolidation, curation, delete, import
— and every one of them is waiting on the same two primitives: **partial update**
and **delete by predicate**. Those two are the highest-leverage things left in
this document.
