"""Observations, written memlake's way rather than Postgres's.

Postgres models an observation as a row that *points at* its sources: it carries
`source_memory_ids`, has no `unit_entities` rows of its own, and inherits its
entities from those sources on every read. That normalisation buys freshness —
editing a source fact's entities immediately changes what the observation
reports.

Against a store that owns the memories, the shape is inverted:

* **Denormalised.** The observation carries the *union of its sources' entity
  ids* directly, resolved once at write time. It is then a memory like any
  other, so entity read-back, the graph arm and the curation list all work on it
  with no special case.
* **Upserted.** An observation has a stable id and an update is a write of that
  same id, which memlake replaces in place. No delete-then-insert: a reinforced
  observation never briefly vanishes from recall, and its id stays valid for
  anything holding a reference.

The cost is the freshness Postgres has — editing a source fact's entities no
longer propagates to observations already built from it; they catch up the next
time consolidation touches them. A deliberate trade.

Sources are recorded twice, cheaply, because the two directions want different
shapes (:meth:`FactRecord.metadata_bag` writes both):

* forward (observation -> its sources) — a JSON list, read back as
  ``StoredMemory.source_memory_ids``;
* backward (fact -> observations built on it) — one metadata key per source
  (``src:<uuid>`` -> ``"1"``), so it is matchable by an equals predicate. That is
  what makes stale-observation cleanup a predicate delete rather than a corpus
  walk.
"""

from __future__ import annotations

import logging

from hindsight_api.engine.memories.base import DeletePredicate, FactRecord, StoredMemory, source_key

logger = logging.getLogger(__name__)


async def resolve_source_entities(store, *, conn, bank_id: str, record: FactRecord) -> FactRecord:
    """Fill an observation's ``entity_ids`` from the union of its sources'.

    The caller builds the record without them, because under Postgres an
    observation has no entity rows at all — it borrows its sources' on read.
    There is no read-time join here, so the borrow happens once, now.

    Left alone if the record already carries entities (a transfer import, which
    brings its own) or has no sources to borrow from.
    """
    if record.entity_ids or not record.source_memory_ids:
        return record

    sources = await store.get_memories(conn=conn, fq_table=None, bank_id=bank_id, unit_ids=record.source_memory_ids)
    if not sources:
        logger.warning(
            "[observations] none of the %d source memories for observation %s resolved; writing it with no entities",
            len(record.source_memory_ids),
            record.unit_id,
        )
        return record

    seen: set[str] = set()
    entity_ids: list[str] = []
    for source in sources:
        for entity_id in source.entity_ids:
            if entity_id not in seen:
                seen.add(entity_id)
                entity_ids.append(entity_id)
    record.entity_ids = entity_ids
    return record


async def observations_for_sources(store, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[StoredMemory]:
    """Every observation built on any of ``unit_ids``.

    The backward walk. Each source is its own metadata key, so this is a filtered
    scan per source rather than a corpus walk — the server drops non-matching
    memories instead of shipping them here to be discarded. Results are de-duped
    because one observation typically summarises several of the ids.
    """
    found: dict[str, StoredMemory] = {}
    for unit_id in unit_ids:
        page_token = ""
        while True:
            page = await store.scan_memories(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["observation"],
                limit=200,
                page_token=page_token,
                metadata_equals={source_key(unit_id): "1"},
            )
            for memory in page.memories:
                found[memory.unit_id] = memory
            page_token = page.next_page_token
            if not page_token:
                break
    return list(found.values())


async def delete_observations_for_sources(store, bank_id: str, unit_ids: list[str]) -> None:
    """Delete every observation built on any of ``unit_ids``.

    The Postgres equivalent walks `source_memory_ids` with an array-overlap
    operator. Here each source is its own metadata key, so this is one predicate
    delete per source — no scan, and no relation for the store to understand.
    """
    for unit_id in unit_ids:
        await store.delete_where(
            bank_id,
            DeletePredicate(fact_types=["observation"], metadata_equals={source_key(unit_id): "1"}),
        )


async def delete_stale_observations(store, *, conn, fq_table, bank_id: str, fact_ids: list) -> int:
    """Drop observations built on facts that are going away, and requeue the rest.

    An observation's text is stale the moment even one of the facts behind it
    disappears, so it is deleted rather than rewritten. Its *surviving* sources
    then go back in the consolidation queue — the same ``consolidated_at = NULL``
    reset the SQL path performs — so they are folded into a fresh observation on
    the next run instead of being stranded as consolidated with nothing to show
    for it.

    Read before delete: the sources can only be recovered from observations that
    still exist.
    """
    if not fact_ids:
        return 0

    ids = [str(f) for f in fact_ids]
    stale = await observations_for_sources(store, conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=ids)
    await delete_observations_for_sources(store, bank_id, ids)
    if not stale:
        return 0

    survivors = {sid for obs in stale for sid in obs.source_memory_ids} - set(ids)
    if survivors:
        await store.mark_consolidated(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sorted(survivors), when=None
        )
    logger.info(
        "[observations] deleted %d observation(s), reset %d source memory/ies for re-consolidation in bank %s",
        len(stale),
        len(survivors),
        bank_id,
    )
    return len(stale)
