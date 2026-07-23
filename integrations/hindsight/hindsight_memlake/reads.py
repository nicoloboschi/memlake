"""The non-retrieval read surfaces, rebuilt on memlake's addressed reads.

`list_memory_units`, `get_memory_unit`, the entity list and the graph view are
all `SELECT`s against `memory_units` in the Postgres implementation. That table
is empty here, so these rebuild the same response shapes from Get / Scan / Stats
instead.

Entity *names* still come from Postgres: the `entities` registry stays there
whichever store is installed, and each memory carries its entity ids, so
resolving names is a join against a table that is still populated. That is the
only reason these helpers take `conn` and `fq_table` at all.

Two honest limitations, both from Scan being a cursor walk rather than a query:

* **Filters memlake cannot push down** (text search, anything not an equality on
  the metadata bag) are applied per page in Python. A page can therefore come
  back short.
* **Offset paging costs pages.** The API takes offset/limit; Scan takes an opaque
  cursor. Reaching an offset means walking to it, so deep offsets get expensive
  and are capped.

A curation-lifecycle note: Postgres moves an *invalidated* fact into a separate
archive table and keeps `invalidation_reason` / `invalidated_at` on it. memlake
has no archive — a memory is either stored or tombstoned, and a tombstoned one
is simply gone — so ``state="invalidated"`` reads an empty set and the
invalidation fields are always null.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from hindsight_api.engine.memories.base import (
    CONSOLIDATED_NO,
    CONSOLIDATED_YES,
    META_CHUNK_ID,
    META_CONSOLIDATED_FLAG,
    META_DOCUMENT_ID,
    StoredMemory,
)

logger = logging.getLogger(__name__)

# Ceiling on pages walked to satisfy one offset+limit request. At the default
# page size this covers a few thousand units — past that the curation UI needs
# cursor paging end to end rather than a deeper walk.
_MAX_SCAN_PAGES = 50

# Cap on memory-to-memory edges the graph view materialises, matching the SQL
# path's `_GRAPH_MAX_EDGES`.
_GRAPH_MAX_EDGES = 10000


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _to_epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _matches(memory: StoredMemory, *, search_query: str | None, document_id: str | None) -> bool:
    """Apply the predicates memlake cannot filter on.

    `document_id` is pushed down as a metadata equality; it stays here as a cheap
    belt-and-braces check. Text search has no server-side equivalent — the bag is
    never indexed — so it still runs per page.
    """
    if document_id is not None and memory.document_id != document_id:
        return False
    if search_query:
        needle = search_query.lower()
        haystack = f"{memory.text or ''} {memory.context or ''}".lower()
        if needle not in haystack:
            return False
    return True


def _consolidation_flag(consolidation_state: str | None) -> str | None:
    """The `META_CONSOLIDATED_FLAG` value a consolidation_state filter selects.

    The flag is the one consolidation field that is a pushable equality: `mark_consolidated`
    writes a distinct value for each of the three states, so 'done', 'pending' and 'failed'
    each map to exactly one value and the server does the filtering. Returns ``None`` for an
    unrecognised state, so the caller can reject it the way the SQL path does.
    """
    from .provider import CONSOLIDATED_FAILED

    return {"done": CONSOLIDATED_YES, "pending": CONSOLIDATED_NO, "failed": CONSOLIDATED_FAILED}.get(
        (consolidation_state or "").lower()
    )


async def resolve_entity_names(conn, entity_ids: list[str], fq_table) -> dict[str, str]:
    """Map entity id -> canonical name from the Postgres entity registry."""
    if not entity_ids:
        return {}
    rows = await conn.fetch(
        f"SELECT id, canonical_name FROM {fq_table('entities')} WHERE id = ANY($1::uuid[])",
        list(set(entity_ids)),
    )
    return {str(r["id"]): r["canonical_name"] for r in rows}


async def entity_map_for_units(store, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> dict[str, list[dict]]:
    """`{unit_id: [{entity_id, canonical_name}]}` for the memories in `unit_ids`.

    The Postgres equivalent joins `unit_entities` and, for observations — which
    carry no rows of their own — inherits the entities of their source memories
    at read time. Here both halves resolve the same way, because every memory
    carries its own ids: an observation's were unioned from its sources when it
    was written (see `observations.resolve_source_entities`), so nothing is
    inherited at read time and nothing is missing.
    """
    if not unit_ids:
        return {}
    memories = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
    names = await resolve_entity_names(conn, [e for m in memories for e in m.entity_ids], fq_table)
    out: dict[str, list[dict]] = {}
    for memory in memories:
        rows = [{"entity_id": eid, "canonical_name": names[eid]} for eid in memory.entity_ids if eid in names]
        if rows:
            out[memory.unit_id] = rows
    return out


async def list_memory_units(
    store,
    *,
    conn,
    fq_table,
    bank_id: str,
    fact_type: str | None,
    search_query: str | None,
    consolidation_state: str | None,
    state: str | None,
    document_id: str | None,
    tags: list[str] | None,
    tags_match: str,
    limit: int,
    offset: int,
) -> dict:
    """The curation list, in the ``{"items", "total", "limit", "offset"}`` shape.

    ``total`` is the live count when no Python-side filter is in play; with one it
    reports what the walk actually saw, since the true total would mean scanning
    the whole corpus.
    """
    if state is not None and state not in ("valid", "invalidated"):
        raise ValueError(f"Invalid state '{state}': expected 'valid' or 'invalidated'.")
    if state == "invalidated":
        # Invalidated memories are deleted from the index and kept in the Postgres
        # archive table, so the "invalidated" tab is a plain SELECT there — no fold,
        # no walk — exactly what the SQL path does.
        return await _list_archived(
            conn, fq_table, bank_id, fact_type=fact_type, search_query=search_query, limit=limit, offset=offset
        )

    fact_types = [fact_type] if fact_type else None

    # Everything expressible as a metadata equality is pushed down and AND-ed by
    # the server; text search is the only predicate left for Python.
    metadata_equals: dict[str, str] = {}
    if document_id:
        metadata_equals[META_DOCUMENT_ID] = document_id
    if consolidation_state:
        flag = _consolidation_flag(consolidation_state)
        if flag is None:
            raise ValueError(
                f"Invalid consolidation_state '{consolidation_state}': expected 'failed', 'pending', or 'done'."
            )
        metadata_equals[META_CONSOLIDATED_FLAG] = flag

    needs_python_filter = bool(search_query)

    collected: list[StoredMemory] = []
    # With no client-side filter the server can skip straight to the offset; with one,
    # the skip would apply before that filter, so we walk and slice as before.
    server_skip = 0 if needs_python_filter else offset
    wanted = limit if not needs_python_filter else offset + limit
    page_token = ""
    pages = 0

    while len(collected) < wanted and pages < _MAX_SCAN_PAGES:
        page = await store.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=max(limit, 100),
            page_token=page_token,
            tags=tags,
            tags_match=tags_match,
            metadata_equals=metadata_equals or None,
            skip=server_skip if pages == 0 else 0,
        )
        pages += 1
        collected.extend(m for m in page.memories if _matches(m, search_query=search_query, document_id=document_id))
        page_token = page.next_page_token
        if not page_token:
            break

    # Stopped at the page cap with more corpus left behind it.
    if page_token and len(collected) < wanted:
        logger.warning(
            "[memories] list_memory_units for bank %s stopped after %d scan pages; "
            "results past offset %d are incomplete",
            bank_id,
            _MAX_SCAN_PAGES,
            offset,
        )

    window = collected[:limit] if server_skip else collected[offset : offset + limit]

    entity_ids = [eid for m in window for eid in m.entity_ids]
    names = await resolve_entity_names(conn, entity_ids, fq_table)

    items = [
        {
            "id": m.unit_id,
            "text": m.text,
            "context": m.context or "",
            "date": _iso(m.event_date) or "",
            "fact_type": m.fact_type,
            "document_id": m.document_id,
            "mentioned_at": _iso(m.mentioned_at),
            "occurred_start": _iso(m.occurred_start),
            "occurred_end": _iso(m.occurred_end),
            "entities": ", ".join(filter(None, (names.get(eid) for eid in m.entity_ids))),
            "chunk_id": m.chunk_id,
            "proof_count": m.proof_count if m.proof_count is not None else 1,
            "tags": list(m.tags or []),
            "metadata": m.metadata or {},
            "consolidated_at": _iso(m.consolidated_at),
            # Written to the metadata bag by `mark_consolidated(failed=True)`, but
            # StoredMemory models no field for it, so it does not survive the read.
            "consolidation_failed_at": None,
            "state": "valid",
            "invalidation_reason": None,
            "invalidated_at": None,
            "edited_at": None,
        }
        for m in window
    ]

    if needs_python_filter or metadata_equals:
        # A filtered walk cannot know the true total without scanning the rest.
        total = len(collected)
    else:
        counts = await store.count_memories(conn=conn, fq_table=fq_table, bank_id=bank_id)
        total = counts.get(fact_type, 0) if fact_type else sum(counts.values())

    return {"items": items, "total": total, "limit": limit, "offset": offset}


async def _list_archived(conn, fq_table, bank_id, *, fact_type, search_query, limit, offset) -> dict:
    """The invalidated tab: a paged SELECT of the Postgres archive table.

    A plain query, not a memlake scan — invalidated memories were deleted from the
    index on invalidation and kept only here, so this is O(archived), not O(corpus),
    and the count is exact. `fact_type` and free-text `search` are the filters the
    UI offers on this tab.
    """
    from .provider import _archived_row_to_stored

    conditions = ["bank_id = $1"]
    params: list = [bank_id]
    if fact_type:
        params.append(fact_type)
        conditions.append(f"fact_type = ${len(params)}")
    if search_query:
        params.append(f"%{search_query}%")
        conditions.append(f"(text ILIKE ${len(params)} OR context ILIKE ${len(params)})")
    where = " AND ".join(conditions)
    arch = fq_table("invalidated_memory_units")

    total_row = await conn.fetchrow(f"SELECT COUNT(*) AS total FROM {arch} WHERE {where}", *params)
    total = total_row["total"] if total_row else 0

    from .provider import _ARCHIVE_SELECT

    rows = await conn.fetch(
        f"SELECT {_ARCHIVE_SELECT} FROM {arch} WHERE {where} "
        f"ORDER BY invalidated_at DESC NULLS LAST LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}",
        *params,
        limit,
        offset,
    )
    stored = [_archived_row_to_stored(r) for r in rows]
    entity_ids = [eid for m in stored for eid in m.entity_ids]
    names = await resolve_entity_names(conn, entity_ids, fq_table)
    items = [
        {
            "id": m.unit_id,
            "text": m.text,
            "context": m.context or "",
            "date": _iso(m.event_date) or "",
            "fact_type": m.fact_type,
            "document_id": m.document_id,
            "mentioned_at": _iso(m.mentioned_at),
            "occurred_start": _iso(m.occurred_start),
            "occurred_end": _iso(m.occurred_end),
            "entities": ", ".join(filter(None, (names.get(eid) for eid in m.entity_ids))),
            "chunk_id": m.chunk_id,
            "proof_count": m.proof_count if m.proof_count is not None else 1,
            "tags": list(m.tags or []),
            "metadata": m.metadata or {},
            "consolidated_at": _iso(m.consolidated_at),
            "consolidation_failed_at": None,
            "state": "invalidated",
            "invalidation_reason": row["invalidation_reason"],
            "invalidated_at": _iso(row["invalidated_at"]),
            "edited_at": None,
        }
        for m, row in zip(stored, rows)
    ]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


async def get_memory_unit(store, *, conn, fq_table, bank_id: str, unit_id: str) -> dict | None:
    """One unit by id, in the curation detail shape.

    For an observation this also expands ``source_memories`` — the facts behind
    it — the same way the SQL detail view does, so the UI can render the chain.
    """
    from .provider import _ARCHIVE_SELECT, _archived_row_to_stored

    memories = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[unit_id])
    memory = memories[0] if memories else None
    # An invalidated memory is out of the index and in the Postgres archive table;
    # the detail view still renders it, flagged, the same way the SQL path reads
    # `invalidated_memory_units`.
    archived = memory is None
    archived_reason: str | None = None
    archived_at = None
    if memory is None:
        row = await conn.fetchrow(
            f"SELECT {_ARCHIVE_SELECT} FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2",
            unit_id,
            bank_id,
        )
        if row is None:
            return None
        memory = _archived_row_to_stored(row)
        archived_reason = row["invalidation_reason"]
        archived_at = _iso(row["invalidated_at"])
    names = await resolve_entity_names(conn, memory.entity_ids, fq_table)
    result: dict = {
        "id": memory.unit_id,
        "text": memory.text,
        "context": memory.context or "",
        "date": _iso(memory.event_date) or "",
        "type": memory.fact_type,
        "mentioned_at": _iso(memory.mentioned_at),
        "occurred_start": _iso(memory.occurred_start),
        "occurred_end": _iso(memory.occurred_end),
        "document_id": memory.document_id,
        "chunk_id": memory.chunk_id,
        "tags": list(memory.tags or []),
        "metadata": memory.metadata or {},
        "proof_count": memory.proof_count if memory.proof_count is not None else 1,
        "entities": [n for n in (names.get(eid) for eid in memory.entity_ids) if n],
        "observation_scopes": None,
        "state": "invalidated" if archived else "valid",
        "consolidated_at": _iso(memory.consolidated_at),
        "consolidation_failed_at": None,
        "edited_at": None,
        "invalidation_reason": archived_reason if archived else None,
        "invalidated_at": archived_at if archived else None,
    }

    if memory.fact_type == "observation":
        # history is deprecated on this route (use GET /memories/{id}/history).
        result["history"] = []
        if memory.source_memory_ids:
            result["source_memory_ids"] = list(memory.source_memory_ids)
            sources = await store.get_memories(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=memory.source_memory_ids
            )
            # mentioned_at DESC NULLS LAST, matching the SQL detail order.
            sources.sort(key=lambda s: (s.mentioned_at is None, s.mentioned_at), reverse=True)
            result["source_memories"] = [
                {
                    "id": s.unit_id,
                    "text": s.text,
                    "type": s.fact_type,
                    "context": s.context,
                    "occurred_start": _iso(s.occurred_start),
                    "mentioned_at": _iso(s.mentioned_at),
                }
                for s in sources
            ]

    return result


async def list_tags(store, *, conn, fq_table, bank_id: str) -> dict[str, int]:
    """Distinct tags and how many live memories carry each.

    The SQL path does `SELECT tag, COUNT(*) FROM unnest(tags) GROUP BY tag`.
    memlake filters on tags but does not aggregate over them, so this walks the
    corpus and counts in Python — O(corpus), acceptable for a UI facet on a
    bank-sized index but not something to call per request.
    """
    counts: dict[str, int] = {}
    page_token = ""
    pages = 0
    while pages < _MAX_SCAN_PAGES:
        page = await store.scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, limit=500, page_token=page_token
        )
        for memory in page.memories:
            for tag in memory.tags or []:
                counts[tag] = counts.get(tag, 0) + 1
        pages += 1
        page_token = page.next_page_token
        if not page_token:
            break
    else:
        logger.warning("[memories] list_tags for bank %s truncated at %d scan pages", bank_id, _MAX_SCAN_PAGES)
    return counts


async def _walk(store, *, conn, fq_table, bank_id, fact_types=None, metadata_equals=None):
    """Yield every live memory in a bank (optionally filtered), bounded by the page cap.

    The shared spine of the count surfaces: memlake has no GROUP BY, so an
    aggregate is a walk that tallies in Python. `metadata_equals` is pushed to the
    server, so a selective count (e.g. the consolidation backlog) walks only the
    matching pages, not the whole corpus.
    """
    page_token = ""
    pages = 0
    truncated = True
    while pages < _MAX_SCAN_PAGES:
        page = await store.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=500,
            page_token=page_token,
            metadata_equals=metadata_equals,
        )
        for memory in page.memories:
            yield memory
        pages += 1
        page_token = page.next_page_token
        if not page_token:
            truncated = False
            break
    if truncated:
        logger.warning("[memories] count walk for bank %s truncated at %d scan pages", bank_id, _MAX_SCAN_PAGES)


async def consolidation_freshness(store, *, conn, fq_table, bank_id: str) -> dict:
    """Last consolidation time + pending/failed counts, walked from memlake.

    The SQL path is one FILTER query over `memory_units`. Here each count is a
    filtered walk — pending and failed push their flag to the server, so each is
    O(matching), not O(corpus); `last_consolidated_at` is the max over the
    consolidated ones, which is the one that costs a full walk of them.

    §5b (declared indexed metadata keys → a MetadataStats RPC) is what makes this
    a metadata-only read instead. Until then it is a walk, which is why
    get_bank_freshness should not be on a hot per-request path against a huge bank.
    """
    from .provider import CONSOLIDATED_FAILED

    facts = ["experience", "world"]
    pending = 0
    async for _ in _walk(
        store, conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=facts,
        metadata_equals={META_CONSOLIDATED_FLAG: CONSOLIDATED_NO},
    ):
        pending += 1
    failed = 0
    async for _ in _walk(
        store, conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=facts,
        metadata_equals={META_CONSOLIDATED_FLAG: CONSOLIDATED_FAILED},
    ):
        failed += 1

    # The bank's last consolidation is the newest consolidated_at across its
    # observations — an observation is written when consolidation runs, so the
    # freshest one dates the last pass without walking every fact.
    last = None
    async for obs in _walk(store, conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=["observation"]):
        when = obs.consolidated_at or obs.created_at
        if when is not None and (last is None or when > last):
            last = when
    return {"last_consolidated_at": last, "pending": pending, "failed": failed}


async def document_memory_counts(store, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, int]:
    """Live memory count per document id — one walk of the bank, tallied by document.

    Per-document push-down would be one walk *each*; a single pass grouping in
    Python is cheaper for a page of documents. Only the requested ids are counted.
    """
    if not document_ids:
        return {}
    wanted = set(document_ids)
    counts: dict[str, int] = {}
    async for memory in _walk(store, conn=conn, fq_table=fq_table, bank_id=bank_id):
        doc = memory.document_id
        if doc in wanted:
            counts[doc] = counts.get(doc, 0) + 1
    return counts


def _truncate_utc(dt, trunc: str):
    """Truncate a datetime to the bucket boundary in UTC — the Python twin of
    `date_trunc(trunc, ts AT TIME ZONE 'UTC')`."""
    dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if trunc == "minute":
        return dt.replace(second=0, microsecond=0)
    if trunc == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


async def memories_timeseries(store, *, conn, fq_table, bank_id: str, time_field: str, trunc: str, since) -> list[dict]:
    """Memories bucketed by ``time_field`` (truncated to ``trunc``) and fact_type.

    Event-time fields fall back to created_at per memory, matching the SQL
    COALESCE, so a memory with no event time still lands in a bucket.
    """
    rollup: dict[tuple, int] = {}
    async for memory in _walk(store, conn=conn, fq_table=fq_table, bank_id=bank_id):
        value = getattr(memory, time_field, None)
        if value is None and time_field != "created_at":
            value = memory.created_at
        if value is None:
            continue
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        if value < since:
            continue
        bucket = _truncate_utc(value, trunc)
        rollup[(bucket, memory.fact_type)] = rollup.get((bucket, memory.fact_type), 0) + 1
    return [
        {"bucket": bucket, "fact_type": fact_type, "count": count}
        for (bucket, fact_type), count in sorted(rollup.items(), key=lambda kv: kv[0][0])
    ]


async def observation_scope_counts(store, *, conn, fq_table, bank_id: str) -> list[dict]:
    """Observations grouped by scope — their sorted tag set — most-populous first."""
    counts: dict[tuple, int] = {}
    async for obs in _walk(store, conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=["observation"]):
        scope = tuple(sorted(obs.tags or []))
        counts[scope] = counts.get(scope, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"tags": list(scope), "count": count} for scope, count in ordered]


async def list_entities(store, *, conn, fq_table, bank_id: str, search: str | None, limit: int, offset: int) -> dict:
    """The entity list: names from the registry, counts from memlake.

    memlake's live count is the more accurate of the two — the SQL path reads an
    incrementally-maintained `mention_count` that only ever grows, since nothing
    decrements it on delete.

    Ordering is by count, which the registry cannot supply, so this fetches the
    bank's entities and their counts and sorts in memory. Both are bounded by the
    entity count rather than the corpus, but a bank with very many entities would
    want the sort pushed down.
    """
    import json

    conditions = ["bank_id = $1"]
    params: list = [bank_id]
    if search:
        params.append(f"%{search}%")
        conditions.append(f"canonical_name ILIKE ${len(params)}")
    where_clause = " AND ".join(conditions)

    rows = await conn.fetch(
        f"SELECT id, canonical_name, first_seen, last_seen, metadata FROM {fq_table('entities')} WHERE {where_clause}",
        *params,
    )
    counts = await store.entity_memory_counts(conn=conn, fq_table=fq_table, bank_id=bank_id)

    merged = []
    for row in rows:
        metadata = row["metadata"]
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        entity_id = str(row["id"])
        merged.append(
            {
                "id": entity_id,
                "canonical_name": row["canonical_name"],
                # Absent from the index means no live memory carries it.
                "mention_count": counts.get(entity_id, 0),
                "first_seen": _iso(row["first_seen"]),
                "last_seen": _iso(row["last_seen"]),
                "metadata": metadata,
            }
        )

    merged.sort(key=lambda e: (-e["mention_count"], e["last_seen"] or "", e["id"]))
    return {
        "items": merged[offset : offset + limit],
        "total": len(merged),
        "limit": limit,
        "offset": offset,
    }


async def prune_orphan_entities(store, *, conn, fq_table, bank_id: str) -> int:
    """Delete registry entries no live memory references.

    The SQL sweep uses `NOT EXISTS (… unit_entities …)`, which matches *every*
    entity when that table is empty by design — it would wipe the registry. This
    asks memlake which entities are actually carried instead, which is one read
    of the entity posting rather than a corpus scan.
    """
    rows = await conn.fetch(f"SELECT id FROM {fq_table('entities')} WHERE bank_id = $1", bank_id)
    if not rows:
        return 0
    all_ids = [str(r["id"]) for r in rows]
    counts = await store.entity_memory_counts(conn=conn, fq_table=fq_table, bank_id=bank_id, entity_ids=all_ids)
    orphans = [eid for eid in all_ids if counts.get(eid, 0) == 0]
    if not orphans:
        return 0
    result = await conn.execute(
        f"DELETE FROM {fq_table('entities')} WHERE bank_id = $1 AND id = ANY($2::uuid[])",
        bank_id,
        orphans,
    )
    return int(result.split()[-1]) if isinstance(result, str) and result.startswith("DELETE") else len(orphans)


async def graph_units(
    store,
    *,
    conn,
    fq_table,
    bank_id: str,
    fact_type: str | None = None,
    search_query: str | None = None,
    document_id: str | None = None,
    chunk_id: str | None = None,
    tags: list[str] | None = None,
    tags_match: str = "all_strict",
    limit: int = 1000,
) -> dict:
    """Graph-view nodes plus the total matching count, as ``{"units", "total"}``.

    The filters that are metadata equalities — ``fact_type`` (via memory_type),
    ``document_id``, ``chunk_id`` — and ``tags`` are pushed to the server; the
    free-text ``q`` is the one predicate left for Python, matched here. Ordering
    differs from the SQL path (`mentioned_at DESC`): a scan walks in cluster
    order, and the view is a node set rather than a ranked list, so which nodes
    you get can change but the rendering does not.

    ``total`` is what the walk saw, not a true bank-wide count: a count would mean
    scanning the whole corpus, which the graph view does not need. Two limits the
    Postgres path does not have: ``document_id`` / ``chunk_id`` do not reach an
    observation through its sources (an observation carries its sources' ids, not
    their document), and the walk is capped, so a very large bank's node set is a
    sample. Both are logged rather than silent.
    """
    fact_types = [fact_type] if fact_type else None
    metadata_equals: dict[str, str] = {}
    if document_id:
        metadata_equals[META_DOCUMENT_ID] = document_id
    if chunk_id:
        metadata_equals[META_CHUNK_ID] = chunk_id

    units: list[dict] = []
    page_token = ""
    pages = 0
    while len(units) < limit and pages < _MAX_SCAN_PAGES:
        page = await store.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=max(limit, 200),
            page_token=page_token,
            tags=tags,
            tags_match=tags_match,
            metadata_equals=metadata_equals or None,
        )
        for m in page.memories:
            if search_query and not _matches(m, search_query=search_query, document_id=None):
                continue
            units.append(_graph_unit_row(m))
            if len(units) >= limit:
                break
        pages += 1
        page_token = page.next_page_token
        if not page_token:
            break

    if page_token and len(units) >= limit:
        logger.warning(
            "[memories] graph_units for bank %s hit the node cap (%d); the view is a sample of a larger bank",
            bank_id,
            limit,
        )
    return {"units": units[:limit], "total": len(units)}


def _graph_unit_row(m: StoredMemory) -> dict:
    return {
        "id": m.unit_id,
        "text": m.text,
        "event_date": m.event_date,
        "context": m.context,
        "occurred_start": m.occurred_start,
        "occurred_end": m.occurred_end,
        "mentioned_at": m.mentioned_at,
        "document_id": m.document_id,
        "chunk_id": m.chunk_id,
        "fact_type": m.fact_type,
        "tags": list(m.tags or []),
        "created_at": m.created_at,
        "proof_count": m.proof_count,
        "source_memory_ids": list(m.source_memory_ids or []),
    }


async def graph_entity_rows(store, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict]:
    """`(unit_id, entity_id, canonical_name)` rows for the graph view's entity edges.

    Replaces the `unit_entities` JOIN: ids come off the memories, names from the
    Postgres registry.
    """
    entity_map = await entity_map_for_units(store, conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
    return [
        {"unit_id": unit_id, "entity_id": row["entity_id"], "canonical_name": row["canonical_name"]}
        for unit_id, rows in entity_map.items()
        for row in rows
    ]


async def graph_direct_links(store, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict]:
    """Memory-to-memory edges with *both* endpoints in ``unit_ids``.

    memlake derives these semantic edges at index time and stores them on the
    memory, so a Get with `include_edges` brings them back with no second query.
    Edges pointing outside the visible set are dropped, matching the SQL path's
    `both endpoints visible` predicate. ``entity_name`` is NULL so the row shape
    matches the derived edges the caller mixes these with.
    """
    if not unit_ids:
        return []
    memories = await store.get_memories_with_edges(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
    visible = set(unit_ids)
    links: list[dict] = []
    for m in memories:
        for target, weight in m.semantic_edges or []:
            if target in visible:
                links.append(
                    {
                        "from_unit_id": m.unit_id,
                        "to_unit_id": target,
                        "link_type": "semantic",
                        "weight": weight,
                        "entity_name": None,
                    }
                )
                if len(links) >= _GRAPH_MAX_EDGES:
                    return links
    return links


async def any_memory_updated_since(
    store,
    *,
    conn,
    fq_table,
    bank_id: str,
    since: datetime,
    fact_types: list | None = None,
    tags: list | None = None,
    tags_match: str = "any",
    tag_groups: list | None = None,
) -> bool:
    """Whether anything in the bank changed after `since` — the staleness signal.

    The SQL form is `SELECT 1 … WHERE updated_at > $2 … LIMIT 1` on an indexed
    column. `updated_at` is a first-class memlake timestamp with a strict
    `updated_from` window, so this is the same shape: ask for one matching
    memory. The server fills a page before answering, so a single call is
    conclusive — an empty result really does mean nothing changed, rather than
    "look at the next page".

    Not free on the negative side: the window is a filter over candidates, not an
    index lookup, so proving *nothing* changed still costs a server-side walk. It
    stays one round trip either way, which is what the caller is paying for.
    The mental model's scope narrows the question: its ``fact_types`` and flat
    ``tags`` are pushed to the scan so only in-scope memories count. ``tag_groups``
    (boolean trees) memlake cannot push down, so they are not applied here — a
    model scoped by a compound expression is treated as bank-wide, which errs
    toward refreshing it, never toward missing a change.
    """
    page = await store.scan_raw(
        bank_id,
        limit=1,
        updated_from=_to_epoch_ms(since),
        fact_types=fact_types,
        tags=tags,
        tags_match=tags_match,
    )
    return bool(page.memories)
