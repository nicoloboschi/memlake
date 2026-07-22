# TODOS — what memlake still needs to back Hindsight's memories

Scope: **memlake-side work only.** Hindsight-side integration work is tracked in
the `hindsight-wt8` worktree (branch `feat/memlake-provider`) and is deliberately
not repeated here. Everything below is something memlake has to provide before
Hindsight can stop falling back or degrading.

Current state: in memlake mode Hindsight writes **nothing** memory-shaped to
Postgres — no `memory_units`, no `memory_links`, no `unit_entities`. Retain, all
four recall arms (dense, full-text, graph, temporal), the curation list and
single-unit reads, bank fact counts, deletes, the curation edit and document
export all run through memlake. LoComo scores identically to the Postgres path
(14/15 on conv-26) with `nprobe=32`.

Ordered by what blocks the most.

---

## 1. Observation ↔ fact edges (`source_memory_ids`)

The single highest-leverage gap: one relation unblocks five surfaces.

In Postgres an observation carries `source_memory_ids`, and Hindsight leans on it
at **runtime** rather than denormalising. Notably an observation has *no*
`unit_entities` rows of its own — `_entity_rows_for_units_sql` inherits its
entities from its source facts on every read, so editing a source fact's entities
is immediately reflected in the observation.

memlake has no equivalent, so today:

- [ ] **Observations come back with no entities.** World/experience facts resolve
      fine (the memory carries its own `entity_ids`), but observations have
      nothing to inherit from.
- [ ] **`include_source_facts` returns nothing** — recall cannot show the facts
      behind an observation.
- [ ] **`prefer_observations` dedup is inert** — it drops source facts already
      covered by a returned observation, which needs the edge.
- [ ] **Observation history has no source resolution.**
- [ ] **Stale-observation cleanup cannot run.** When a source fact is deleted or
      re-ingested its observations are stale; Hindsight logs and skips rather than
      querying an empty table and reporting a clean sweep.

What is needed: `source_memory_ids` carried on the memory (forward) plus reverse
adjacency for the backward walk, exposed as a bidirectional expansion.

**Design note worth deciding together:** the alternative is for Hindsight to
denormalise — write the union of the source facts' `entity_ids` onto the
observation at consolidation time. That needs no memlake change, but it loses the
runtime-freshness property Postgres has today: an entity edit would no longer
propagate to existing observations. Cheaper, strictly weaker.

---

## 2. Edges as a readable relation

memlake derives semantic links at index time and expands the graph internally,
but never returns edges. Hindsight can therefore rank *through* the graph but
cannot render it.

- [ ] **`get_graph_data` returns 0 nodes and 0 links** — the graph view is blank.
- [ ] **Link counts in bank stats are `{}`.**

What is needed: an RPC returning a memory's neighbours (semantic + causal, with
weights), or edges attached to a `Get`. Display and diagnostics only, so it does
not have to be on the hot path.

---

## 3. Consolidation support

Consolidation is the one Hindsight subsystem still fully disabled — it refuses to
run rather than reading an empty table and reporting a clean pass. Restoring it
needs three things, two of which already exist:

- [x] Partial update (`Patch`) — for stamping `consolidated_at`.
- [x] `Scan` — for walking candidate memories.
- [ ] **A queryable "unconsolidated" predicate.** The job selects memories where
      `consolidated_at IS NULL`. That flag lives in the metadata bag, which is not
      indexable, so the only implementation today is a full corpus walk filtered
      in Python on every consolidation cycle. Needs either an indexed/filterable
      metadata field, or a scan-side metadata predicate — the same shape
      `DeleteByPredicate` already accepts.
- [ ] §1's observation↔fact edges (consolidation is what writes them).

---

## 4. Retrieval semantics

- [ ] **The dense arm does not see un-indexed writes.** Reads are documented as
      always strong, and the text arm honours that — but a query issued
      immediately after a write returns nothing from the vector arm until the
      indexer folds it. Measured: import 344 memories, query at once →
      `semantic=0`; the same namespace minutes later → `semantic=344`.
      Retain-then-recall in one request is a normal Hindsight pattern, so this is
      the difference between a BM25-only answer and a full one. Either the tail
      scan covers the dense arm, or the guarantee is documented per-arm so callers
      can decide.
- [ ] **The default `nprobe` silently costs recall.** On a 344-memory bank
      (15 + 10 clusters) the default returned 266 of 344 candidates where pgvector
      returned all 344 — and LoComo accuracy dropped to 11/15. The cause is
      cluster coverage, not depth: `vector_top_k` was already 2500, and candidates
      in unprobed clusters are unreachable at any depth.

      | nprobe | candidates | coverage | median query |
      |--------|-----------|----------|--------------|
      | 0 (default) | 266 | 77% | 5.9 ms |
      | 15 | 344 | 100% | 5.2 ms |
      | 32 | 344 | 100% | 4.9 ms |

      Full coverage was *free* at this size (0 roundtrips either way). Setting
      `nprobe=32` restored both the candidate count and 14/15 accuracy. Suggested:
      scale the default with cluster count, and/or report the coverage a query
      actually achieved — right now a 77% probe is indistinguishable from an
      exhaustive one, which is exactly how this went unnoticed.
- [ ] **`text_signals` are not indexed.** Hindsight enriches its BM25 document
      with entity names and spelled-out dates ("May 8 2023") so keyword search can
      hit them. memlake indexes `text` only, so its full-text arm is strictly
      weaker on entity/date queries. Either accept a second indexed-text field, or
      index `text + signals`.
- [ ] **No nested tag groups.** memlake has five flat modes; Hindsight also
      supports AND/OR/NOT trees, applied in Python after the query — so they can
      trim below the requested limit.
- [ ] **No `updated_at` range push-down.** Recall's `created_after` /
      `created_before` window is applied post-query for the same reason.

---

## 5. Scan ergonomics

`Scan` is a cursor walk, which is right for browsing but leaves four gaps for the
curation UI and export:

- [ ] **No push-down for metadata predicates.** `document_id`, text search and
      consolidation state are filtered per page in Python, so a page can come back
      short and `total` reports what the walk saw rather than a true count.
      `DeleteByPredicate` already accepts a metadata predicate — the same shape on
      `Scan` would close this, §3, and the export path in one move.
- [ ] **No ordering.** The SQL path returns `ORDER BY mentioned_at DESC NULLS
      LAST, created_at DESC`; a scan walks in cluster order, so the curation list
      comes back in storage order.
- [ ] **Offset paging costs pages.** The API takes offset/limit, `Scan` takes a
      cursor, so reaching an offset means walking to it. Hindsight caps the walk at
      50 pages and logs when it truncates.
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

## 6. Curation archive state

- [ ] **Tombstones are one-way.** Hindsight's curation models invalidation
      structurally: a row moves to `invalidated_memory_units` and can be reverted.
      memlake can only tombstone, so invalidate/revert is unsupported and
      `list_memory_units` always reports `state: "valid"`. Needs either a
      soft-delete/archived state that `Scan` can filter on, or an explicit
      "restore tombstoned id" op.

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
- [ ] **No cross-store atomicity.** Facts are written inside Hindsight's retain
      transaction; if that rolls back, memlake keeps entries whose Postgres
      document rows never existed. Recall skips ids it cannot resolve, so the
      failure mode is wasted slots rather than wrong answers — but they accumulate,
      and there is no reconciliation or drift detection.
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
