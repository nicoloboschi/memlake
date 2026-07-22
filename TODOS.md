# TODOS — what memlake still needs to back Hindsight's memories

Scope: **memlake-side work only.** Hindsight-side integration work is tracked in
the `hindsight-wt8` worktree (branch `feat/memlake-provider`) and is deliberately
not repeated here. Everything below is something memlake has to provide before
Hindsight can stop falling back or degrading.

Current state: in memlake mode Hindsight writes **nothing** memory-shaped to
Postgres — no `memory_units`, no `memory_links`, no `unit_entities`. Retain, all
four recall arms (dense, full-text, graph, temporal), the curation list and
single-unit reads, bank fact counts, deletes, the curation edit, document export
and **consolidation** all run through memlake. LoComo scores identically to the
Postgres path (14/15 on conv-26) with `nprobe=32`.

Ordered by what blocks the most.

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

- [x] **Semantic edges on the payload — DONE.** `MemoryPayload.semantic_out`
      (target + weight), behind an `include_edges` flag on `Scan` and `Get`.

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

- [ ] **The streaming indexer derives no semantic links.**
      `streaming.rs:238` — `item.semantic_out.clear(); // bulk build derives no
      semantic links`. Above `MEMLAKE_INDEXER_STREAMING_THRESHOLD` docs a namespace
      takes that path and has **no semantic edges at all**, so the graph silently
      shows entity and observation structure only. Since the graph is core UI and
      this flips at a size threshold, the failure mode is "works in dev, empty in
      production" with no error. (Noted as being removed.)

- [ ] **Temporal and causal edge types.** Temporal links are not edges — derive
      adjacency from the timestamps already on the payload. `causal_out` is already
      on the payload but Hindsight does not yet render it in the graph.

- [ ] **Bank-stats link counts stay `{}`.** Countable only once edges are a
      queryable relation rather than a per-memory list; low value on its own.

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

## 6. Curation archive state

- [ ] **Tombstones are one-way.** Hindsight's curation models invalidation
      structurally: a row moves to `invalidated_memory_units` and can be reverted.
      memlake can only tombstone, so invalidate/revert is unsupported and
      `list_memory_units` always reports `state: "valid"`. Needs either a
      soft-delete/archived state that `Scan` can filter on, or an explicit
      "restore tombstoned id" op.

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
- [ ] **A codec change does not re-encode existing clusters.** Copy-forward
      reuses `.vec` blocks by reference, so flipping the codec silently leaves
      old generations in the old encoding. Blocks are self-describing so reads
      stay correct — but a migration never happens, and the first attempt at
      measuring Binary vs Int8 returned identical numbers for exactly this
      reason. Needs either a forced re-encode on codec change, or an explicit
      "the codec is per-generation, not per-index" statement in the docs.
- [ ] **The rerank SSTable spends ~3117 B to hold a 1536 B vector** — roughly 2×
      overhead, unexplained. Worth understanding before this scales.
- [x] **`nprobe` is resolved by the index, not the client.** `nprobe = 0` now means
      "the index decides" and the snapshot picks a quarter of its clusters
      (floor 8, cap 64). A fixed constant made recall depend on corpus size:
      8 clusters is 11% of a small index and a rounding error on a large one. On
      scifact this moved `ann_recall@10` from 0.8590 to **0.9627**. The wire field
      remains as an escape hatch; it should eventually be removed from the client
      API entirely.
- [ ] **Adaptive probing via the error bounds (the real answer to `nprobe`).**
      Rather than a tuned fraction, probe clusters nearest-first and stop when the
      k-th best *lower* bound already exceeds the best score any unprobed cluster
      could yield (bounded by its centroid distance). That spends probes only
      where the ranking is still contested and needs no per-corpus calibration —
      a stopping rule instead of a constant. We are unusually well placed for it
      since `score_bounds` already exists, but the centroid-distance bound itself
      is unvalidated: it needs to be proven sound before anything trusts it to
      stop early, because stopping too soon silently drops results.
- [ ] **The binary bound is probabilistic, not absolute.** Measured containment
      is 1.000000 over 120k samples with a 0.999 gate and a worst-miss cap, but
      one rotation serves a whole block, so misses are correlated across members
      of the same block rather than independent. `Int8` and `F32` bounds are
      absolute. If a caller ever needs a hard guarantee, Binary is not it.
- [ ] **On isotropic data the bound narrows nothing** (99.95% of the block enters
      rerank). Correct, and useless — a caller must not assume the rerank set is
      always small.
