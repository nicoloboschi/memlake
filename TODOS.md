# TODOS — what memlake still needs to back Hindsight's memories

Scope: mostly **memlake-side work**, but §0 also tracks the Hindsight-side and
operational pieces that stand between "verified end to end in dev" and "running
against real banks". The integration itself now lives in this repo,
`integrations/hindsight` (`hindsight_memlake`), tested against a live server; the
OSS seam it implements is Hindsight PR #2917 (`feat/pluggable-memories-provider`).

Current state: in memlake mode Hindsight writes **nothing** memory-shaped to
Postgres — no `memory_units`, no `memory_links`, no `unit_entities`. Running
through memlake and verified end to end:

* retain, and all four recall arms (dense, full-text, graph, temporal);
* consolidation — observations written denormalised and upserted by id, sources
  stamped, observation history recorded;
* the curation list, single-unit reads, curation edit, tag listing, bank fact
  counts, entity listing with live per-entity counts;
* deletes — bank (namespace drop), document and observation (predicate), unit
  (tombstone) — and document re-ingest that *replaces* rather than duplicates;
* document and whole-bank export, including observations;
* the graph view: entity edges, observation edges, and stored semantic edges;
* graph maintenance's orphan-entity sweep, via `EntityStats`;
* mental-model staleness.

LoComo scores identically to the Postgres path (14/15 on conv-26) with
`nprobe=32`.

Ordered by what blocks the most.

---

## 0. Path to a real Hindsight deployment

The functional surface is done and verified at LoComo parity; what is left is
mostly **operational and packaging**, not features. Grouped by how hard it blocks.

### 0a. Hard blockers — the extension cannot even load / run in a real deployment

- [ ] **The OSS seam is unreleased.** `hindsight_memlake` imports
      `hindsight_api.engine.memories.base`, which exists only on Hindsight PR #2917.
      Until that merges and ships in a `hindsight-api-slim` release, the extension
      installs only from a source checkout (which is why the integration tests wire
      it in through `HINDSIGHT_API_SLIM_PATH`).
- [ ] **protobuf runtime conflict.** `memlake_client`'s generated stubs call
      `ValidateProtobufRuntimeVersion(...)` against protobuf-7 gencode; Hindsight
      pins `protobuf>=6.33.5`. It is a floor, so 7.x *may* resolve, but OTel deps
      have historically capped it — verify against the real Hindsight lockfile, and
      if it does not resolve, regenerate the stubs against 6.x gencode. Otherwise
      `import memlake_client` raises `VersionError` and the extension never loads.
- [ ] **No deployment wiring, and the indexer is mandatory.** memlake needs two
      processes against S3 — the `serve` gRPC API *and* a continuous
      `mlake-server index` loop. Search returns nothing until a fold runs (the tests
      call `index --once` by hand). There is no docker-compose/helm standing up
      server + indexer + bucket next to Hindsight, and no operator config doc.

### 0b. Correctness gaps that fail *silently* in production

- [x] **Engine surfaces that bypassed the store — DONE.** An audit found ~13 sites
      still issuing `memory_units` / `memory_links` SQL on paths that run for *any*
      store, so under memlake they hit empty tables: the whole curation surface
      (`delete_bank` — which orphaned the namespace entirely — `delete_document`,
      `update_document`, `clear_observations{,_for_memory}`, `retry_failed_consolidation`,
      delta-retain tag propagation and chunk deletes), recall context-expansion
      (observation source-chunks, reflect's `expand`), and the residual stats
      (consolidation freshness, per-document counts, per-bank `fact_count`). Each is
      now gated on `writes_memory_rows_in_sql`: the SQL branch is byte-identical, the
      other routes through the store. `find_failed_consolidation` was added to the
      contract for the retry path. Verified 28/28 against *both* backends after
      making the fixtures seed and assert through the store rather than raw SQL —
      which is what caught the last bug (a fact-type-scoped `delete_bank` read its
      sweep ids from `memory_units`, so the stale-observation sweep silently skipped).

- [ ] **Cross-store atomicity — design decided, not yet built (see §7).** Postgres
      stays the single commit authority; memlake is a dumb idempotent target (no
      2PC). Sync retain disabled under memlake; async retain compensates a failed op
      by deleting the Postgres rows + a reliable `TombstoneWhere(document_id)` — which
      memlake already supports (atomic, idempotent, `write_seq`-race-closed). Left:
      the sync-disable flag, the compensation path, a drift-reconciliation backstop.
- [ ] **Format-version bumps silently invalidate namespaces.** A v1→v2 bump left
      existing namespaces unreadable with no migration path. Fine pre-release; needs
      a story before real data lands. (See also §7.)

### 0c. Migration prerequisites for real data

- [ ] **No backfill path** from an existing Postgres bank into a namespace. This
      can only be switched on for *new* banks until it exists, and it is also how a
      like-for-like A/B benchmark would be set up. (See also §7.)
- [ ] **Multi-tenancy undecided.** Hindsight isolates tenants by Postgres schema;
      the provider maps a bank to a flat namespace with a prefix, one shared bucket.
      Decide whether the namespace *is* the tenancy boundary and what isolation it
      guarantees. (See also §7.)
- [ ] **`DeleteNamespace` hazards** — not safe against a running indexer, and
      returns `INTERNAL: no manifest` rather than `NOT_FOUND` on a missing namespace.
      (See §7.)

### 0d. Feature-parity gaps — runs, but degraded UX

- [x] **Curation edit / invalidate / revert — DONE → §6.** All three paths of
      `update_memory_unit` now go through the store, tested live: invalidate/revert
      via the Postgres archive table (the memory is deleted from the index), and
      field edits (text/context/dates/fact_type/entities + re-embed) via an
      `apply_edit` rewrite.
- [x] **Count surfaces — DONE.** All four route through the store and return real
      numbers in memlake mode instead of zero/empty. The two keyed on a metadata
      value — `list_documents` per-doc counts (`document_id`) and
      `get_bank_freshness` pending/failed (`consolidated`) — are now **metadata-only
      reads** via `MetadataStats` (§5b, implemented); `get_memories_timeseries` and
      `list_observation_scopes` stay bounded scans (time buckets and tag sets are
      different primitives — TimeTable / a tag multiset). Bank-stats *link* counts
      now return real numbers too, off `LinkStats` / `EntityStats` rather than a
      queryable edge relation (§2).
- [ ] **`list_tags` is O(corpus) per call → §5.** No `ListTags` RPC / tag-count
      histogram yet.
- [ ] **Smaller parity gaps:** no scan ordering (curation list comes back in
      storage order, not `mentioned_at DESC` → §5); no key-*absence* predicate
      (worked around with a positive `consolidated` flag → §5). *(Nested tag groups
      were the third gap here — now pushed down and applied inline, §4.)*

### 0e. Performance / polish

- [ ] **No `grpc.aio` client → §8.** Every call is wrapped in `asyncio.to_thread`,
      a thread hop per retain batch and per recall.
- [ ] **Residual stale co-occurrences accumulate → §6b.** Inert hygiene.

---

## 1. ~~Observation ↔ fact edges~~ ✅ RESOLVED — no memlake work needed

Decided: Hindsight **denormalises** observations instead of asking memlake for an
edge relation. Every primitive it needed already existed.

An observation is now written as an ordinary memory (`memory_type=3`) that
carries the **union of its sources' entity ids** resolved at write time, so it is
searchable and renderable on its own terms with no special case. Create and
update are the same call — an upsert of a stable id — so a reinforced observation
is replaced in place and never leaves the index.

Sources ride in the metadata bag twice, because the two directions want different
shapes: a JSON list for the forward read, and one `src:<uuid>` key per source for
the backward one. The per-source keys mean "observations built on fact X" is an
equals predicate, which `DeleteByPredicate` already accepts — so stale-observation
cleanup needs no new RPC either.

Verified end to end on the 239-fact LoComo bank: 84 observations created, 20
updated in place, 83 live, each carrying its sources and inherited entities.

What this trades away, deliberately: Postgres inherits an observation's entities
from its sources on *every read*, so editing a source fact's entities changes the
observation immediately. Denormalised, they only catch up the next time
consolidation touches that observation.

One residue: the backward direction works for *deletes* (predicate) but not for
*reads* — finding observations for a source without deleting them still needs a
walk, because `Scan` takes no metadata predicate. That is §5's push-down item, not
a separate gap.

---

## 2. Edges as a readable relation — mostly unblocked

`get_graph_data` now renders. It turned out most of the graph was never read from
`memory_links` in the first place: the edges are **derived**, and from data the
provider already has.

* **Entity edges** — pair visible memories that share an entity. Built from the
  memories' own `entity_ids` plus the Postgres name registry. Verified: 200 nodes,
  2786 entity edges on the LoComo bank.
* **Observation edges** — pair observations that share a source memory, from
  `source_memory_ids` carried on each observation.
* **`get_entity_graph`** — a separate endpoint over `entity_cooccurrences`, which
  is still written and read normally (see §6b). Verified: 159 nodes, 352 edges.

What is still missing is only the *stored* edge types:

- [x] **Semantic edges on the payload — DONE and verified.**
      `MemoryPayload.semantic_out` (target + weight), behind an `include_edges`
      flag on `Scan` and `Get`. Measured on the LoComo bank: 1508 semantic edges
      across 344 memories, rendering as 924 `semantic` edges alongside 2859
      `entity` edges in a 200-node graph.

      No new RPC and no new storage: the indexer already derives these during the
      fold (`indexer.rs:590-677`), already prunes them to live targets
      (`indexer.rs:183`), and already persists them on `StoredMemory` — the graph
      arm reads them at `query_node.rs:1174`. The payload conversion was simply
      dropping them. So the cost is response bytes and nothing else: no extra
      write-time compute, no extra bytes on disk, no second round trip, because
      the memory is already materialized when the payload is built.

      Deliberately **opt-in**: `Query` is the hot path and never uses edges, and
      would otherwise pay ~18 bytes per edge across every candidate (~27 KB on a
      300-candidate recall). The graph reads through `Scan`/`Get`, where it is free.

- [x] **The streaming indexer derives semantic links — DONE.** It used to clear
      `semantic_out` on the bulk path (`streaming.rs:238`), so a namespace over
      `MEMLAKE_INDEXER_STREAMING_THRESHOLD` docs had no semantic edges at all — the
      graph silently showed entity/observation structure only, a "works in dev,
      empty in production" failure with no error. The streaming fold now re-derives
      semantic links per cluster in `build_type_streaming`, so both index paths
      produce the same edges and the size threshold no longer changes the graph.

- [ ] **Temporal and causal edge types.** Temporal links are not edges — derive
      adjacency from the timestamps already on the payload. `causal_out` is already
      on the payload but Hindsight does not yet render it in the graph.

- [x] **Bank-stats link counts — DONE.** Counted *without* first making edges a
      queryable relation: `semantic` / `causal` from the fold-time per-segment
      `LinkStats` tally plus the WAL tail, `entity` from the `EntityStats` posting
      index (the same `SUM(LEAST(n-1, cap))` Postgres computes, so the two agree),
      `temporal` derived by sweeping the time index. A metadata read, not a corpus
      walk. Routed through the store's `link_counts`, so the stats page no longer
      disagrees with the graph view about whether links exist.

---

## 3. ~~Consolidation~~ ✅ RUNNING

Consolidation creates, updates and deletes observations through the provider,
stamps its sources, and completes a full pass over a real bank (239 facts → 80
observations created, 27 updated in place, 77 live).

Its candidate query is now pushed down: memories carry a positive
`consolidated` flag (`"0"` / `"1"`) so `metadata_equals` can select the backlog,
rather than the server shipping every page for Hindsight to discard. See §5 for
why a positive flag was needed instead of matching the marker's absence.

Two behaviours worth knowing, neither blocking:

- Failed memories are stamped with the same marker as successful ones — there is
  no separate `consolidation_failed_at` — so a permanently-failing memory is
  retried once and then treated as done rather than retried forever.
- `_filter_live_source_memories` is a visibility check, not a lock: there is no
  transaction to join, so a source deleted between the check and the write leaves
  an observation citing it. The next pass rebuilds from what is still live.

---

## 4. Retrieval semantics — mostly closed

- [x] **Read-after-write — solved by `wait_for_index`.** A read always merges the
      un-indexed WAL tail over the indexed generation, and `WriteRequest.wait_for_index`
      folds inline before returning for callers that want the write in the generation.
      The earlier symptom (a query right after a write returning `semantic=0`) was the
      vector arm having no clusters to probe, not the tail being invisible.
- [x] **`nprobe` — now chosen automatically.** The default no longer silently
      probes a fraction of clusters. For the record, what it cost when it did: on a
      344-memory bank the semantic arm returned 266 of 344 candidates and LoComo
      accuracy dropped to 11/15; `nprobe=32` restored both. Coverage and depth are
      different knobs — `vector_top_k` was already 2500 and could not compensate,
      because candidates in unprobed clusters are unreachable at any depth.
- [x] **`text_signals` — DONE, as `Memory.index_text`.** Text to index for BM25 when
      it should differ from `text`; empty means index `text`. `StoredMemory::fts_text()`
      feeds all three FTS build sites (in-RAM fold, streaming fold, WAL tail).
      Hindsight sends `text + text_signals`, so entity names and spelled-out dates
      ("May 8 2023") are matchable without changing what a hit returns.
      Verified: a term present only in `index_text` matches, and the hit returns the
      clean original text.
- [x] **`updated_at` window — DONE.** A fifth field on `Timestamps` — deliberately
      separate from the other four, which are *content* time where this is *write*
      time — plus `updated_from` / `updated_to` on Query and Scan. The server defaults
      it to now when a client omits it, so a window never silently skips memories
      written by a client that does not set it.
      Verified round-trip and both window directions.
  - [x] **Push-down on the dense arm — DONE.** The window used to be a post-arm
        filter, which for a selective window meant a page of nothing: the dense arm
        truncates to `vector_top_k` *before* anything is materialized, so filtering
        afterwards removes rows rather than reaching deeper into the matching set.
        `updated_at` now rides in the vector block beside the tags (an 8-byte column
        per member, `FLAG_HAS_UPDATED`), so a non-matching member never takes a
        top-k slot, and the per-cluster summary carries the write-time span so a
        selective window is not starved out of the nprobe-nearest probe set either —
        the same treatment a selective tag filter gets.
        Verified end to end: with the only in-window memories at ranks 31-35 and a
        depth of 10, the arm returns exactly those five, at `nprobe=1` as well.
        The WAL tail is filtered in the arm too (it is scored exhaustively).
    - [ ] **The FTS and graph arms still post-filter.** Neither index carries a
          write time — tantivy would need the field, and the graph arm walks edges —
          so a text- or graph-only query can still be trimmed below `top_k`. The
          server re-applies the window over all arms and remains the authority.
    - [ ] **Blocks written before the column admit everyone.** A generation built by
          an older binary carries no `updated` column, so the arm cannot decide and
          falls back to admitting; the server's own filter catches it. Reindexing a
          bank is what turns the push-down on for its existing data.
  - [x] **A patch now bumps the write time — DONE.** `Patch` used to leave
        `updated_at` untouched, so a consolidated or edited memory stayed invisible to
        "changed since X" — the window was only ever true for upserts. The server
        appends a `Delta::Touch(now)` to every patch, last, so it wins over a client's
        `SetTimestamps`. It is a separate delta because `SetTimestamps` *replaces* the
        whole struct: stamping the write time through it would wipe the four content
        timestamps the caller never mentioned. The value is baked in at commit, so
        replaying the log stays deterministic.
  - [x] **A partial timestamp patch no longer wipes the rest — DONE.** Found while
        adding `Touch`, not caused by it: `Delta::SetTimestamps` replaces the whole
        struct and the client built that struct from *only* the fields passed, so
        `patch(id, occurred_start=X)` silently cleared `event_date`, `occurred_end`
        and `mentioned_at` — and Hindsight's `update_memories` sets whichever of the
        four it has, so it fired on any consolidation revising one time and not the
        others.
        `Delta::MergeTimestamps` now carries the patch: `Some` overwrites, `None`
        leaves alone, which is what a partial update means. `SetTimestamps` stays as
        the only way to *null* a field, reached by the new `replace_timestamps` flag
        on the wire `Patch` (`Timestamps` fields are `optional`, so presence is exact).
        The default flipped to merge — the safe direction, since the replace discards
        times the caller never mentioned.
        Verified over gRPC: a partial patch keeps the other three and still stamps
        `updated_at`; `replace_timestamps=True` clears them and stamps too.
- [x] **Nested tag groups — DONE, pushed down.** A recursive `TagPredicate`
      (leaf / and / or / not) on `Query` and `Scan`, evaluated server-side; leaves
      reuse the five flat modes, so every per-mode subtlety (including how each
      treats untagged memories) is inherited rather than reimplemented.

      They no longer trim below the limit, because each arm applies the predicate
      *inline, before it truncates to depth* — the dense arm off the block's
      per-member tag column (a bit decode, no payload read), the FTS arm off its
      stored tags (it already over-fetches internally to fill `k` passing), the
      graph/temporal arms off the hydrated record. An arm's top-k is therefore
      already the top-k that *pass*, so a match ranked below the top-k non-matches
      is still returned. `Scan` applies it per member over the whole walk, where
      there is no truncation to race at all. The Python post-filter is gone.

---

## 5. Scan ergonomics

`Scan` is a cursor walk, which is right for browsing but leaves four gaps for the
curation UI and export:

- [x] **Metadata predicate on `Scan` — DONE.** `metadata_equals` reuses
      `core::Predicate`, the same matcher `DeleteByPredicate` uses, so tags and
      metadata are one conjunction with one implementation. Hindsight now pushes
      down `document_id` in the curation list, the consolidation candidate filter,
      and the backward observation lookup.
  - [ ] **No way to match key *absence*.** The predicate is equality-only, so
        "not yet consolidated" — the absence of a marker — is inexpressible.
        Hindsight works around it by writing a positive flag (`consolidated: "0"`,
        flipped to `"1"`), which is fine but means every such state needs a
        pre-declared field. A `metadata_missing` / `metadata_not_equals` form
        would remove the workaround.
- [ ] **No ordering.** The SQL path returns `ORDER BY mentioned_at DESC NULLS
      LAST, created_at DESC`; a scan walks in cluster order, so the curation list
      comes back in storage order.
- [x] **Offset/skip paging — DONE.** `skip` discards N matching memories before
      filling the page; verified byte-identical to following the cursor. Hindsight
      uses it for offset in the curation list, falling back to the page walk only
      when a text search forces client-side filtering.
- [ ] **No tag histogram — but the index is most of the way there.**
      `list_tags` is `SELECT tag, COUNT(*) … GROUP BY tag` in SQL; against memlake
      Hindsight walks the corpus and counts in Python, which is O(corpus) per call.

      `ClusterTagSummary` (`mlake-index/src/generation.rs:95`) already stores, per
      cluster, the sorted set of distinct tags plus `has_untagged`, built at index
      time for Phase-4b pruning and referenced from the manifest. So the tag
      **vocabulary** is already derivable from index metadata alone — no cluster
      reads, cost independent of corpus size, same shape as `Stats`.

      What is missing is counts: the summary is a set, not a multiset, so a tag
      spanning ten clusters cannot be summed. Extending it to carry a count per
      tag (`Vec<(String, u32)>`, or a parallel `counts: Vec<u32>`) makes an exact
      histogram a metadata-only op, and leaves pruning unchanged — presence is
      just `count > 0`. Size cost is one u32 per (cluster, distinct tag) pair.

      Suggested surface:

      ```protobuf
      rpc ListTags(ListTagsRequest) returns (ListTagsResponse);
      message ListTagsRequest { string namespace = 1; repeated uint32 memory_types = 2; }
      message TagCount { string tag = 1; uint64 count = 2; }
      message ListTagsResponse { repeated TagCount tags = 1; uint64 untagged_count = 2; }
      ```

      Pattern matching, ordering and paging stay client-side — the vocabulary is
      small and Hindsight's `list_tags` takes an ILIKE pattern with limit/offset
      that is easier to apply over a returned list.

      Two things to decide: counts would reflect the **indexed generation, not the
      WAL tail** (fine for a UI facet, worth documenting — it is the same staleness
      the dense arm has), and **tombstones** need the same adjustment `Stats.doc_count`
      already makes, or the counts silently include deleted memories.
      A vocabulary-only `ListTags` with no counts is still a large win over the
      corpus walk if the multiset change proves awkward.

---

## 5b. Declared indexed metadata keys — the counts primitive

**Status: the four surfaces below now work via a bounded scan (routed through the
store, §0d) — this item is the *optimization* that makes them metadata-only reads
instead of O(matching) walks.** A scan is the wrong answer for a hot path like
`get_bank_freshness` (reflect calls it often); the cheap sources are:

| surface | grouped by | cheapest source |
|---|---|---|
| `get_memories_timeseries` | time buckets | `TimeTable` — already sorted; bucketing is a range walk |
| `list_observation_scopes` | tag set | `ClusterTagSummary` multiset (same work as §5's tag histogram) |
| `get_bank_freshness` pending/failed | `consolidated` metadata value | **this item** |
| `list_documents` per-document count | `document_id` metadata value | **this item** |

The last two share one primitive, and it is the highest-leverage of the four
because it also closes `document_id` push-down on `Scan` (§5) and makes
delete-by-document an index lookup rather than a walk.

**Design — mirror `tag_summary`, not a new concept.** Metadata cannot be indexed
wholesale: `context`, `created_at` and `updated_at` are effectively unique per
memory, so a blanket index is one entry per memory per key. The keys worth
indexing are low-cardinality or bounded (`document_id` ~ documents,
`chunk_id` ~ chunks, `consolidated` = 2), so the namespace declares them.

Concrete insertion points, all alongside structures that already exist:

- `Manifest.indexed_metadata_keys: Vec<String>` (`manifest.rs:167`, `#[serde(default)]`)
  — declared once for the namespace, carried across swaps.
- `FactTypeIndex.meta_counts: String` (`manifest.rs:~45`, beside `tag_summary`)
  — one small per-segment blob, `key -> [(value, count)]`, for the declared keys only.
- Built at fold from the segment's items, exactly where `tag_summary` is built.
- `MetadataStats(namespace, key) -> [(value, count)]` sums across segments and
  applies the same tail/tombstone adjustment `Stats.doc_count` and `EntityStats`
  already make.

Cost is bounded by distinct `(declared key, value)` pairs, not corpus size — the
same shape as the per-cluster tag set.

- [x] **Implemented.** `Manifest.indexed_metadata_keys` (declared once, carried
      across every swap); the fold tallies each item's value for exactly those keys
      into `Stats.meta_counts` (per fact type, per segment), built on both the
      in-RAM and streaming paths beside `tag_summary`. `MetadataStats(namespace,
      key)` sums the per-segment tallies and folds in the WAL tail, the same
      adjustment `EntityStats` makes. The stats blob is read on demand (only by
      this admin call), so the query hot path never pays for it. Verified: the fold
      counts by value, the tail is included, and an undeclared key returns empty.
      Wired into the extension — `document_memory_counts` and the freshness
      pending/failed are `MetadataStats` reads, no scan fallback (every bank
      declares the keys; backwards compat for pre-declaration namespaces is not
      handled yet — such a bank would need a reindex, or the changeable
      declaration surface below).
- [x] **Declaration surface decided: a `CreateNamespace` field**
      (`indexed_metadata_keys`), fixed at creation. Simple, and it matches how the
      keys are used — the extension declares `document_id` + `consolidated` when it
      first ensures a bank's namespace. A changeable `SetNamespaceConfig` (needing a
      re-fold to backfill a newly-declared key) is a later refinement, not needed
      for the surfaces this serves.

---

## 6. Curation archive state

- [x] **Invalidate / revert — DONE. Invalidated units leave the index entirely.**
      Invalidation is structural on both stores: Postgres moves the row to
      `invalidated_memory_units`, and the extension **deletes the memory from the
      bank's namespace** — so it never touches the IVF/FTS index again — and keeps
      it in that same Postgres archive table (cold storage, no index, which is what
      it exists for). Restore writes it back and deletes the archive row.

      An earlier cut moved the memory to a sibling `<ns>__invalidated` namespace,
      but a namespace *is* an indexed structure — folding it re-indexed the very
      units that should be inert. Deleting from memlake + parking the payload in
      Postgres is the fix: invalidated units cost nothing to keep, and
      `list_memory_units(state="invalidated")` is a plain `SELECT` on the archive
      (no fold, exact count), exactly like the SQL path.

      Everything the memory_units-shaped archive columns cannot hold — the vector,
      the memory_type, the causal edges, the raw metadata bag — rides in a reserved
      key inside the row's `metadata`, so revert rebuilds the memory faithfully.
      `get_memory_unit` reports `state: "invalidated"` with the reason/timestamp.

      Behind a five-method archive lifecycle on the store interface
      (`invalidate_memory` / `restore_memory` / `get_archived_memory` /
      `set_invalidation_reason` / `set_memory_embedding`), so no call site branches
      on which store is installed. Verified live against a real memlake server *and*
      a real Postgres: invalidate deletes from memlake and writes the archive row;
      the reason updates; the invalidated tab lists it; restore reconstructs the
      memory (tags/edges/metadata intact) and clears the archive.

      This is a deliberate, narrow relaxation of "nothing memory-shaped in
      Postgres" — only for cold, invalidated units, in the archive table, never
      `memory_units` / `memory_links` / `unit_entities`. The proper long-term fix
      is a memlake-server *archived* state the fold skips (kept in the payload
      store, out of IVF/FTS, fetchable by Get) — tracked below.

      One residue: `index_text` (entity/date BM25 enrichment) does not survive the
      round trip, so a reverted memory loses it until its next edit — the archive
      is never FTS-queried, and revert re-embeds anyway.

- [ ] **Proper fix: a memlake-server `archived` state.** The Postgres-archive
      approach above keeps invalidated units out of the index, but at the cost of
      writing them to Postgres. The clean version is a state memlake owns: the fold
      skips archived units (kept in the payload SSTable so Get/restore work), recall
      and scan exclude them, an op flips it back. Then invalidation is a single op,
      nothing memory-shaped touches Postgres, and the archive is native. Rust work
      across the fold + query path + proto; do it once the segmented-index refactor
      settles.

- [x] **Edit (`update_memory_unit` field edits) — DONE.** Correcting a memory's
      text/context/dates/fact_type/entities now routes through the store too, so it
      works in memlake mode instead of silently no-op'ing. Two new interface
      methods: `clear_unit_entities` (drop a unit's postings before re-linking — a
      no-op for a store that carries entity ids on the memory) and `apply_edit`
      (write the new fields, reset the consolidation markers, drop the derived
      links; the embedding follows via `set_memory_embedding` once the caller
      re-embeds). Postgres keeps its `UPDATE memory_units` + `DELETE memory_links`;
      memlake rewrites the memory (an edit changes the entity set and can change
      the memory_type, neither of which `patch` expresses, so it is a full upsert).
      Entity *resolution* stays Hindsight's — the engine resolves names against the
      Postgres registry and hands the ids down. Verified: the Postgres curation
      edit suite passes unchanged, and a memlake edit rewrites text/context/
      fact_type/entities while preserving the untouched fields.

---

## 6b. ~~Graph maintenance~~ ✅ — `EntityStats` added and wired

Hindsight's maintenance job has three passes. One is obsolete here, one now works
through a new RPC, one has to be skipped.

* **Relink top-up — obsolete.** `memory_links` is never written: semantic links
  are re-derived by the indexer from the whole corpus on every fold, and temporal
  links are not edges at all (the arm reads timestamps), so a delete cannot leave
  one dangling.
* **Orphan entity prune — works, via `EntityStats`.** The SQL version keys off
  `unit_entities`, which is empty by design, so its `NOT EXISTS` matches *every*
  entity — an unguarded run deleted 161 of 161 on a real bank. It now asks the
  provider which entities are actually carried.
* **Stale-cooccurrence prune — skipped, and must stay skipped.** Same shape of
  bug: `NOT EXISTS (… INTERSECT …)` over an empty `unit_entities` is always true,
  so it would delete every co-occurrence row.

- [x] **`EntityStats` — DONE and verified.** `EntityId -> live memory count`, read
      from the entity posting SSTable (`EntityTable::counts`): the value is the
      memory ids concatenated, so a count is `len / 16` — no decode, no cluster
      reads, cost scaling with entities rather than corpus. Applies the same
      tail/tombstone adjustment `Stats.doc_count` makes; entities with no live
      memories are omitted, so an id you ask about and do not get back is an orphan.

      It serves two callers, which is why it mattered more than a background sweep:
      `list_entities` surfaces a per-entity count in the **API**, where a
      scan-based recount would be O(corpus) *per request*. It is also the more
      accurate number — the `mention_count` column it replaces only ever increments,
      since nothing decrements it on delete.

      Verified on a 160-entity bank: `list_entities` returns Caroline 153,
      Melanie 126, family 28; the orphan sweep runs and correctly prunes nothing
      when every entity is live.

      Note this was **not** the same work as the tag histogram (§5): the entity
      posting already stores what is needed, whereas `ClusterTagSummary` stores a
      set and would have to become a multiset.

### Correction: `entity_cooccurrences` is *not* dead

An earlier revision of this file claimed nothing reads it. Wrong — it has two
readers, both via `fq_table()` which a naive grep misses:

1. `get_entity_graph()` (`memory_engine.py:9745`), exposed at `http.py:4408`;
2. entity resolution's disambiguation signal (`entity_resolver.py:350`, `:476`, `:582`).

It also references only `entities`, never `memory_units`, so it works unchanged
in memlake mode. Hindsight now splits the writer: the `unit_entities` insert is
skipped (FK to `memory_units`), the co-occurrence cache still runs. Verified: 352
co-occurrence rows written and a 160-node / 352-edge entity graph rendered with
`unit_entities` at zero.

- [ ] **Residual leak: stale co-occurrences accumulate.** Pairs whose witnessing
      memories have all been deleted linger, because deciding that needs
      `unit_entities`. Recomputing from the provider is possible but O(corpus).
      The rows are inert — they inflate the entity graph slightly and add weak
      disambiguation signal — so this is hygiene, not correctness. Options: accept
      it, recompute periodically, or have memlake expose entity *pair* counts the
      way it now exposes per-entity counts.

---

## 6c. Postgres constraints that provider-held memories cannot satisfy

A pattern worth naming, because it bit twice and both times failed *silently*.

Tables that reference `memory_units` assume every memory is a Postgres row. When
the provider owns the store that assumption breaks, and the failure mode is never
an error the caller sees:

* **`observation_history.observation_id → memory_units(id)`.** Every history insert
  raised a foreign-key violation, which the writer catches and logs as "a race with
  parallel consolidation". The audit trail was silently empty. Fixed on the
  Hindsight side by dropping the FK (migration `a1c9e7f3b2d8`) and replacing the
  cascade with explicit deletes.
* **`prune_orphan_entities` / `prune_stale_cooccurrences`** both key off
  `unit_entities`, which is empty by design, so their `NOT EXISTS` matched
  everything. One measured run deleted 161 of 161 entities. Both now guarded.

No memlake work — recorded so the next such table is checked before it is trusted.
The general shape: any `NOT EXISTS` or FK against `memory_units` is either
always-true or always-violated in provider mode, and both read as success.

---

## 7. Operational

- [ ] **Dropping a namespace is not safe against a running indexer.** The proto
      says as much ("no snapshot atomicity across the deletes"). Observed: a
      drop-then-reimport loop with the indexer at a 3s interval lost a pass and the
      following `Stats` errored. Either the drop fences the indexer, or the docs
      state that the operator stops it first.
- [ ] **`DeleteNamespace` on a missing namespace returns `INTERNAL: no
      manifest`.** Callers delete defensively, so Hindsight swallows that specific
      string. A `NOT_FOUND` status or an idempotent drop would beat string-matching
      an INTERNAL message.
- [ ] **Cross-store atomicity — design decided, not yet built.** Facts are written
      inside Hindsight's retain transaction; if that rolls back, memlake keeps
      entries whose Postgres document rows never existed. Recall skips ids it cannot
      resolve (wasted slots, not wrong answers), but they accumulate with no
      reconciliation.
      **Decision: memlake stays a dumb, idempotent target — no transactions, no 2PC.**
      Postgres is the single commit authority; memlake converges to it.
      - *Sync retain:* not supported when memlake is the memory store — sync-atomic
        across two stores needs 2PC. Disable it via config (fail/downgrade the sync
        path); callers that require sync-atomic don't use memlake for that write.
      - *Async retain:* doc/chunks/entities commit and are visible immediately;
        memories go to memlake via the existing at-least-once op-level retry. On op
        failure (retries exhausted) the op is marked failed and **compensated**: a new
        txn deletes the Postgres doc rows AND enqueues a reliable
        `Op::TombstoneWhere(metadata_equals=[("document_id", X)])` to memlake.
      - *memlake already has everything* — `TombstoneWhere` is atomic (one entry),
        idempotent, and race-closed by `write_seq` (a later valid re-retain survives an
        earlier compensation delete; same-entry replace lets a re-ingest swap a
        document's facts atomically). Nothing to add server-side.
      - *Hindsight-side requirements:* stamp `document_id` into memory metadata (already
        done); route the compensation delete through the reliable retry path, not
        fire-and-forget; serialize same-document ops; and note that if recall joins
        memlake hits back to Postgres, orphan memories are invisible during the
        compensation window (the tombstone is then just physical GC).
      - *Still to build:* the sync-retain-disable config flag, the compensation-delete
        path in `hindsight_memlake`, and a periodic drift/reconciliation sweep as a
        backstop for the poison-message tail.
- [ ] **No backfill path** from an existing Postgres bank into a namespace.
      Needed before this can be switched on for real data, and it is also how a
      like-for-like benchmark would be set up.
- [ ] **Multi-tenancy is undecided.** Hindsight isolates tenants by Postgres
      schema and banks by `bank_id`; the provider maps a bank to a flat namespace
      with a configurable prefix. Decide whether namespaces are the tenancy
      boundary and what isolation they guarantee.
- [ ] **Format-version bumps silently invalidate namespaces.** A v1→v2 bump left
      every existing namespace unreadable (`Stats` fails, `ListNamespaces` omits
      them) with no migration path — during this work it wiped several test banks
      mid-session. Fine pre-release; needs a story before real data lands.

---

## 8. Client / packaging

- [ ] **protobuf runtime conflict.** The generated stubs require protobuf ≥ 7
      (`ValidateProtobufRuntimeVersion(PUBLIC, 7, …)`); Hindsight's lockfile pins
      6.33.5 via its OTel deps, so importing `memlake_client` there raises
      `VersionError`. Installing protobuf 7 into the Hindsight venv works, but that
      is a manual override. Either regenerate against 6.x-compatible gencode, or
      agree a floor across both projects.
- [ ] **`memory()` takes no `causal_out`,** so the provider appends causal edges
      to the protobuf message after construction.
- [ ] **`proof_count` defaults to `0`** in `memory()`; Hindsight's column defaults
      to `1`.
- [ ] **No `grpc.aio` client.** Hindsight is fully async and wraps every call in
      `asyncio.to_thread` — a thread hop per retain batch and per recall.

## Vector storage — decided, measured, not yet shipped

Context and numbers in [`docs/vector-storage.md`](docs/vector-storage.md). The
two-stage search (1-bit scan → error bound → exact f32 rerank) is built and
green; what follows is what is left.

- [ ] **Make `Binary` the default codec again.** It was set to `Binary` in
      `8ce2617`, then back to `Int8` in `477b620` (a WIP commit). BEIR now says
      the choice is free: on scifact, Binary and Int8 give **identical**
      `ann_recall@10` on a clean rebuild, because the rerank stage makes the
      scan codec's error irrelevant to the final ranking. Binary is ~6.5× smaller
      in the scan tier, so there is no reason to pay for Int8 codes.
- [x] **A codec change does not re-encode existing clusters.** ~~Copy-forward
      reuses `.vec` blocks by reference, so flipping the codec silently leaves
      old generations in the old encoding.~~ **FIXED.** `read_generation` now
      records each cluster's stored codec from its (self-describing) block header
      (`Generation::codecs`), and the copy-forward gate requires that codec to
      equal `IndexOptions::vector_codec` — a mismatch forces a re-encode, so a
      codec change migrates. Copy-forward still fires in the common (same-codec)
      case: paths are reused, asserted by
      `unchanged_codec_copies_clusters_forward`; the migration by
      `codec_change_reencodes_copied_forward_clusters`. Migration is incremental
      (each fold/compaction re-encodes only what it visits; mixed state is safe
      because reads are per-block codec-aware) — no forced whole-corpus fold. See
      docs/vector-storage.md §"The codec is per-block". The compounding-quantization
      worry (each fold decodes then re-encodes) is pinned as one-shot, not
      accumulating, by `codecs_are_stable_under_repeated_decode_reencode`: F32 is
      byte-idempotent, Int8/Binary drift is immaterial (recall holds flat over 8
      generations). Remaining gap: a fully static corpus never folds, so it needs
      a codec-mismatch compaction trigger to migrate — left for the segmented-index
      work.
- [ ] **The rerank SSTable spends ~3117 B to hold a 1536 B vector** — roughly 2×
      overhead, unexplained. Worth understanding before this scales.
- [x] **`nprobe` is resolved by the index, not the client.** `nprobe = 0` now means
      "the index decides" and the snapshot picks a quarter of its clusters
      (floor 8, cap 64). A fixed constant made recall depend on corpus size:
      8 clusters is 11% of a small index and a rounding error on a large one. On
      scifact this moved `ann_recall@10` from 0.8590 to **0.9627**. The wire field
      remains as an escape hatch; it should eventually be removed from the client
      API entirely.
- [x] ~~**Adaptive probing via the error bounds.**~~ **DONE AND REJECTED — do not
      re-attempt without re-measuring the two numbers below first.** Built, proved
      sound against brute force, measured on BEIR, and then **removed from the tree**
      (not left behind an env flag — unexercised code rots, and the reasoning is the
      part worth keeping).
      *What was built:* each centroid carried a radius `R = max ‖v̂ − c‖`, recomputed
      every fold on both build paths, bounding the best score any unprobed cluster
      could hold at `min(⟨q̂,c⟩ + R, 1 − max(0, ‖q̂−c‖ − R)²/2)`. Bounded at two fetch
      waves so INV-7 held, and the probed set was a subset of the fixed-fraction one,
      so recall could not fall below baseline. The rule is correct; that was never the
      problem.
      *The measurement is a clean negative.* On BEIR at nprobe=half it retired **0
      clusters across 623 queries**: scifact 36.0 → 36.0 probed, `ann_recall@10` 0.9893
      either way; nfcorpus 30.0 → 30.0, 0.9625 either way. Mean `tau` is 0.55–0.65 while
      the *tightest* bound anywhere in the unprobed tail is 0.96–0.98 — a ~0.3 gap, not a
      near miss. Nor is it outliers inflating `R`: mean max radius 0.62 vs mean p95 radius
      0.58, so a p90 radius (which would cost soundness) moves the bound ~0.04 and still
      retires 0.00%.
      *The cause is dimensional, not statistical.* At 384-d the cluster radius (~0.62) is
      below the query's own k-th nearest-neighbour distance (~0.77) by less than the bound
      needs, so every ball reaches into the query's neighbourhood and nothing can be
      retired. A larger corpus at the same dimensionality fails identically. **The gate for
      any retry: measure mean cluster radius vs. mean k-th-NN distance first; if the radius
      is not comfortably smaller, stop.** Full write-up in docs/arms/vector.md; the
      implementation is in git history (`git log -S adaptive_probe`).
- [ ] **The binary bound is probabilistic, not absolute.** Measured containment
      is 1.000000 over 120k samples with a 0.999 gate and a worst-miss cap, but
      one rotation serves a whole block, so misses are correlated across members
      of the same block rather than independent. `Int8` and `F32` bounds are
      absolute. If a caller ever needs a hard guarantee, Binary is not it.
- [ ] **On isotropic data the bound narrows nothing** (99.95% of the block enters
      rerank). Correct, and useless — a caller must not assume the rerank set is
      always small.

## Read path: the zero-copy claim, and the cache

- [ ] **We deserialize; we do not read zero-copy — but the value is now small,
      and the invasive fix should wait for the segmented index to settle.**
      `rkyv_read` validates with `check_archived_root` (yielding a usable
      `&Archived<T>`) and then throws it away by `Deserialize`ing into an owned
      graph. Two things learned while scoping this:

      1. **The 6-byte envelope header makes the aligned fast path DEAD.** Every
         format wraps its payload behind `envelope::HEADER_LEN = 6`, so
         `&bytes[6..]` is never 8-aligned and *every* rkyv read takes the copy
         branch, unconditionally — the in-place branch in `rkyv_read` never runs
         for an enveloped object. Padding the header 6->8 makes a cold read from
         mmap (page-aligned, so 8-aligned) skip that copy. It is a format change
         (version bump + re-ingest) and removes only one of three copies; the
         deserialize allocations, the dominant cost, remain.
      2. **After the vector/payload split the deserialize is off the hot path.**
         The scan reads flat `VectorBlock` bytes and never deserializes. A whole
         `ClusterFile::from_bytes` now happens only in the fold (`read_generation`,
         not latency-critical) and winner-hydration is bounded by arm depth through
         the payload store. So the allocation storm that motivated this is already
         mostly gone.

      The real prize (operating on `&Archived<StoredMemory>` at the call sites,
      the accessors already exist) threads borrow lifetimes through the query and
      fold path — which is exactly the code the segmented/LSM refactor is
      restructuring. Doing a large lifetime refactor concurrently with a
      structural one is high-risk for low post-split payoff; sequence it *after*
      the segmented work lands, and gate it on a measured allocation/latency
      number rather than the principle. The header-alignment sub-fix is
      independent and smaller, but is still a core format break and should ride
      the next deliberate format bump, not a solo one.
- [x] **The disk cache now mmaps instead of `fs::read`** (`cache.rs`). A warm hit
      maps the blob and hands the mapping to `Bytes::from_owner`, so the blob is
      never copied onto the heap and a re-read of a resident file does no I/O at
      all. Sound because blobs are content-addressed and written once, published
      by rename (so a concurrent re-`put` installs a new inode instead of
      truncating a mapped one — truncation under a mapping is SIGBUS, not an
      error), and removed only by unlink, which on Unix leaves an existing mapping
      valid; a covering test evicts a blob while a reader holds it.
      **This removes one copy, not all of them** — see the item above: consumers
      still `Deserialize` into an owned graph, so the read path is not zero-copy.
- [ ] **Blobs are still not written 8-byte aligned**, so `rkyv_read` may realign
      before validating and the mmap gives that copy straight back. mmap returns
      page-aligned addresses, so a blob whose archive starts at offset 0 is already
      fine; a *ranged* block cached as `path#start-end` is not. Worth confirming
      which of the two the hot paths actually hit before doing anything.
- [x] **The disk cache is a CLOCK ring, not LRU and no longer plain FIFO**
      (`cache.rs`). Both tiers stay admission-ordered rings — a read never reorders
      them, so a memory hit takes the state lock only *shared* and a disk hit takes
      it shared too — but a hit now sets an atomic **reference bit**, and eviction
      walks a hand that clears a set bit and skips that entry (second chance) before
      taking the first entry whose bit is already clear. A memory eviction still
      demotes to disk; a disk hit still does not promote back into memory (that
      promotion *was* the per-hit write-lock mutation, and with mmap the page cache
      already backs the bytes). Measured with
      `crates/mlake-store/tests/cache_skew.rs`, which now runs all three policies
      over the byte-identical trace (256 clusters, 16 distinct Zipf-skewed probes per
      query plus three always-hot small objects; `--ignored --nocapture`, ~5 min):

      | cache/corpus | 5% | 10% | 25% | 50% |
      |---|---|---|---|---|
      | LRU, Zipf s=1.1   | 0.0780 | 0.4493 | 0.6769 | 0.8442 |
      | FIFO, Zipf s=1.1  | 0.0895 | 0.3327 | 0.5937 | 0.7940 |
      | CLOCK, Zipf s=1.1 | 0.0889 | **0.4708** | **0.6898** | **0.8500** |
      | LRU, Zipf s=1.5   | 0.0994 | 0.5948 | 0.8137 | 0.9226 |
      | FIFO, Zipf s=1.5  | 0.1450 | 0.4479 | 0.7344 | 0.8839 |
      | CLOCK, Zipf s=1.5 | 0.1047 | **0.6172** | **0.8218** | **0.9269** |
      | LRU, uniform      | 0.0165 | 0.2154 | 0.3465 | 0.5657 |
      | FIFO, uniform     | 0.0170 | 0.1379 | 0.3193 | 0.5546 |
      | CLOCK, uniform    | 0.0164 | **0.2157** | **0.3470** | **0.5666** |

      CLOCK does not just recover FIFO's 3–15 point loss: outside the 5% column it is
      0.4–2.2 points *ahead of LRU*, because an admission arrives with its bit clear
      and a one-shot cluster read dies at its FIFO position, where LRU promotes every
      cold cluster it touches to most-recently-used and evicts hot entries for it.
      (Admitting with the bit set was tried and is worse everywhere — 0.4133 vs
      0.4708 at s=1.1/10% — so the scan resistance is doing the work, not the second
      chance alone.) The 5% column is the thrash regime: 16 distinct probes per query
      already exceed the cache, every policy laps itself once per query, and no
      reference bit survives to the hand's next pass; FIFO's edge there is position,
      not policy, and everything is under 0.15.
      **The uniform-control regression is fixed** — 0.1379 → 0.2157 against LRU's
      0.2154, i.e. within noise, which is what a flat access distribution should
      show. The always-hot objects' own hit ratio says it directly: 0.999 under CLOCK
      and LRU, 0.503 under FIFO. So SPEC §6.2's in-RAM tier for
      centroids/footers/`pk.idx` goes back to being an optimisation rather than
      load-bearing.
      Caveat on the baseline: the LRU column is `EvictionPolicy::Lru`, a
      re-measurement that refreshes both rings on a hit; the LRU actually replaced
      only refreshed the memory tier on a memory hit and scored 0.8314 in the
      s=1.1/50% cell. Every other cell reproduces the old table to ±0.001, so CLOCK
      is being judged against a slightly *stronger* LRU than the one that existed.

- [ ] **Nothing calls `Store::put_admitting` yet.** The mechanism landed
      (`store.rs`): an opt-in write that admits its own bytes under exactly the key
      `get_immutable` looks up, so a `serve` replica folding a generation inline no
      longer has to re-fetch what it just wrote. `Store::put` stays read-through on
      purpose — a fold writes the whole generation, most of which no imminent query
      probes, and on a ring that laps the cache and leaves only the fold's own tail.
      What is left is the decision it was scoped to leave open: **which** inline-fold
      writes opt in. The plausible answer given the measurement above is the small
      certain-to-be-read objects (centroids, footers, `pk.idx`) and not the cluster
      or vector-block bulk, but that is untested — it could not be measured from
      inside `mlake-store`, which has no fold to run.

## WAL read path: caching + the hot-path LIST (high-QPS reads)

Strongly-consistent reads re-scan the WAL tail every query. Two costs per query: a LIST to
enumerate the tail, and a GET per tail entry. Progress and what's left:

- [x] **Tail entry bodies are now cached.** `WalTail::scan` and `read_wal_entry` read through
      `get_immutable` (the NVMe cache), not the uncached `get`. A `{ns}/wal/{seq}` object is
      write-once and — after the fix below — its sequence never repeats, so its path names one
      immutable body forever; caching by path is sound. An unchanged tail re-read across queries
      is now local hits instead of N S3 GETs. Only the enumerating LIST still hits S3.
- [x] **Sequences are monotonic for the namespace's life (correctness fix).**
      `Writer::cold_next_seq` now resumes at `max(live head, manifest.wal_head) + 1`. Before, a
      fully-folded-and-GC'd namespace had an empty `/wal/` (live head 0) and a cold writer
      restarted at seq 1 — which is **at or below `wal_index_cursor`, so the write was invisible
      to readers (`(cursor, head]` is empty) and never folded (`head <= cursor` short-circuits):
      a silently lost write.** The manifest's `wal_head` survives GC as a high-water mark, so the
      resume is always past it. Also the precondition for the caching above (no path reuse → no
      stale-body aliasing). Covered by `a_reset_wal_resumes_past_the_manifest_high_water_mark`.
- [x] **Per-query LIST eliminated via a head pointer.** `snapshot()`'s validity check and
      `QueryNode::open`'s consistency point now call `Namespace::resolve_head` — one GET of a
      monotonic `{ns}/wal-head` pointer — instead of `wal_head()`'s LIST. A writer CAS-bumps the
      pointer after durably appending and **before** acking (`Writer::bump_head_pointer`, cached
      etag → one CAS on the single-writer path), so it is `>=` every acked write; a reader trusting
      it never misses one. A missing/malformed pointer falls back to the authoritative LIST, so
      correctness never depends on it. The indexer keeps LISTing (background) and reconciles a
      crashed writer's un-acked entry. Tests: `commit_publishes_the_head_pointer`,
      `resolve_head_falls_back_to_list_when_pointer_absent`,
      `concurrent_writers_leave_the_head_pointer_at_the_max`.
- [x] **Tail-enumeration LIST eliminated too — the read path is now LIST-free.**
      `WalTail::scan_up_to(after_seq, head)` enumerates the tail by construction —
      `seq_path(after_seq+1..=head)`, `head` from the pointer — and GETs each through the immutable
      cache, tolerating a `NotFound` as a `commit_many` gap. Safe because GC reclaims only
      `seq <= prev_wal_index_cursor`, so entries in `(cursor, head]` are never GC'd — a miss there
      is only a genuine gap. `QueryNode::open` and `reopen_extending_tail` use it; the **indexer
      keeps `scan` (LIST)** on the background fold, which is also the authority that reconciles
      gaps. Validated by the full 52-test end-to-end suite (all exercise `open` → `scan_up_to`).
      So a read now touches S3 with: one pointer GET (staleness) + on reopen, constructed tail
      GETs (cached) + a conditional manifest GET — **no LIST anywhere on the read path.**
- [ ] **WAL naming: incremental seq vs random UUID — keep incremental.** Random-UUID WAL objects
      would remove the `put_if_absent` claim contention and be reset-proof, *but* they make the
      hot-path LIST **mandatory** (you cannot probe or enumerate an unpredictable name) and lose
      the total order the fold and supersede logic rely on (you'd need an explicit `write_seq`
      key). Since "avoid LIST" is the higher-value goal and incremental seq is what enables it,
      UUID naming is the wrong trade. The clash it targets is already absent within a replica
      (one `Writer` per namespace serializes claims) and is best handled across replicas by
      per-namespace ownership, not by giving up ordering and probeability.

## Snapshot reopen: reuse segment metadata across a write (etag fast-path)

- [x] **On a write (not a fold), reopen rebuilds only the tail and reuses the segments — gated on
      the manifest etag.** `snapshot()` now, when the head advanced, does a conditional GET of the
      manifest (`Store::get_if_modified`, `If-None-Match`): a **304** (a write, segments unchanged)
      → `QueryNode::reopen_extending_tail`, which reuses each fact type's `Arc<Vec<SegmentState>>`
      (no re-decoding of centroids/tables/FTS) plus the segment-derived tombstone overlay, and
      re-scans only `(wal_index_cursor, head]`; a **200** (a fold) → full `open`. The tail is
      correctly re-scanned (not reused), so no acked write is missed. `FactType.segments` became
      `Arc`-shared; the segment vs tail predicate-tombstones were split so the segment half is
      reused verbatim. Proven equivalent to a fresh open (incl. a tail delete of an already-folded
      item) by `reopen_extending_tail_matches_a_fresh_open`.

## Cache: namespace isolation (surfaced by the SLA model)

- [ ] **The read cache needs per-namespace isolation or frequency-awareness; global CLOCK
      gives neither.** CLOCK (the current policy) is scan-resistant but not isolated: under
      concurrent namespaces, one namespace's large `Scan` or write burst floods the single
      shared cache and evicts a *different* busy namespace's hot working set — a noisy
      neighbour that drops that namespace from the MEMORY tier to COLD and spikes its p99.
      The SLA model (`docs/sla-model.md`) can only promise a *per-namespace* SLA if a busy
      namespace's working set is protected. Two fixes, both stronger than CLOCK:
      **per-namespace reservation** (cache divided among active namespaces, weighted by
      traffic — simplest, makes the SLA `mem_cache / active_namespaces`), or
      **frequency-aware admission** (TinyLFU / S3-FIFO — a one-shot scan block never
      displaces a repeatedly-hit one, no per-namespace bookkeeping). Decide behind a
      multi-namespace load test that reproduces the noisy-neighbour eviction first.

## Indexer fold throughput and fairness (surfaced by the SLA model)

**The "indexer hang" is starvation, not fold cost — corrected after a clean measurement.** An
earlier draft of this section blamed `derive_links` O(N); a controlled MinIO run disproved that.

- [ ] **Single indexer folds all namespaces sequentially → starvation (the real "hang").**
      The `index` loop with empty `--namespaces` discovers *every* namespace and folds them
      round-robin in one thread. **Measured:** a 100k-memory namespace **isolated** (its own
      bucket, no other namespaces) folds to completion in **~34 s (~2 900 docs/s @ 4 vCPU)**. The
      *same* 100k in a bucket the parallel Hindsight test suite had polluted with 18 `ext-test-*`
      corpse namespaces **crawled to ~35 docs/s and never converged** — the target published its
      first 8k docs, then the indexer spent every cycle re-folding the 18 others and never came
      back. That is the "hang": per-namespace fold latency is inflated by the sum of all other
      namespaces, and a namespace under steady ingest can starve indefinitely.
      The built image's `index` subcommand also **ignored `--namespaces`/`--once`** (the flag was
      a no-op; it just ran discover-all) — verify these flags actually reach the loop, since §0a
      lists them as the operational scoping mechanism. Fix direction: per-namespace fold
      concurrency/fairness (round-robin with a per-namespace time slice, or a work queue), or one
      indexer per (group of) namespace(s). This is what blocks a *multi-namespace* write SLA.

- [ ] **`derive_links` is O(new·N) — a scaling watch-item, NOT the current bottleneck.**
      Every fold derives semantic kNN links: `IndexOptions::default()` has `derive_links: true`
      (`indexer.rs:48`), used by both the indexer loop (`service.rs:1007`) and the CLI
      (`main.rs:235`); `derive_corpus_links` runs one full-corpus vector query per new memory per
      fact type (`indexer.rs:234`, comment: "the accepted incremental O(new · N) link-derivation
      cost"). This is genuinely super-linear and *will* dominate fold cost at much larger `N`.
      **But at 100k it is already included in the 34 s bulk fold above**, so it is not what caps
      throughput today — the isolated fold is fast. Re-measure before quoting a fold SLA at
      ≫1M memories/namespace; if it bites there, the levers are: amortize link derivation off the
      synchronous path (async pass after ack — costs edge freshness), bound the kNN to the new
      items' own clusters (O(new), slightly worse graph), or make `derive_links` opt-in per
      namespace (Hindsight uses edges §2, so per-workload not a global flip). No action needed for
      the current SLA card; `CPU_S_PER_FOLDED_MEM_1T ≈ 0.45 ms` is valid at the 100k scale measured.
