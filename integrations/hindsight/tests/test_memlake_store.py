"""The memlake store, exercised at the MemoriesExtension seam.

Every test here drives the real store against a live server: a write is a gRPC
call that lands in S3, a search is the real dense + BM25 arms over a real index.
What is faked is only the *Hindsight* side of the boundary — the fact objects the
retain pipeline would hand in, and the one Postgres read the store makes for
entity names — because standing up Hindsight to produce those would test
Hindsight, not this store.

The store is the sole home for memories: no test touches a `memory_units` table,
because for this store there isn't one. That is the property under test — that a
store owning its rows can satisfy the whole interface — so `conn` and `ops` are
passed as ``None`` wherever the store is meant to ignore them, and a real failure
would surface as an ``AttributeError`` rather than a silent fall-through.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from hindsight_api.engine.memories.base import FactRecord


def _vec(seed: float, dim: int = 8) -> list[float]:
    """A deterministic unit-ish vector; distinct seeds give distinct directions."""
    return [seed] + [seed * 0.1 * i for i in range(1, dim)]


def _entity(entity_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(entity_id=entity_id, name=name)


def make_fact(
    text: str,
    *,
    seed: float = 0.5,
    fact_type: str = "world",
    tags: list[str] | None = None,
    document_id: str | None = None,
    chunk_id: str | None = None,
    context: str | None = None,
    metadata: dict | None = None,
    entities: list[SimpleNamespace] | None = None,
    causal_relations: list | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
) -> SimpleNamespace:
    """A stand-in for the retain pipeline's ProcessedFact.

    Only the attributes ``build_fact_records`` reads are present — the store never
    sees a real ProcessedFact, only this duck-typed shape, so matching it is
    matching the contract.
    """
    return SimpleNamespace(
        fact_text=text,
        embedding=_vec(seed),
        fact_type=fact_type,
        tags=tags or [],
        context=context,
        document_id=document_id,
        chunk_id=chunk_id,
        metadata=metadata,
        observation_scopes=None,
        entities=entities or [],
        causal_relations=causal_relations or [],
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
    )


class FakeConn:
    """A Postgres connection stand-in for the one read the store still makes.

    The entities *registry* stays in Postgres, so resolving an entity id to its
    canonical name is the single place the store reaches back across the boundary.
    Every other method ignores its ``conn``, and passing this only where names are
    wanted keeps that honest: a test that hands it in is asserting a name lookup
    happened; every other test hands in ``None``.
    """

    def __init__(self, names: dict[str, str]):
        self._names = names

    async def fetch(self, _sql: str, ids):
        return [
            {"id": entity_id, "canonical_name": self._names[str(entity_id)]}
            for entity_id in ids
            if str(entity_id) in self._names
        ]


def _fq_table(name: str) -> str:
    return name


# --------------------------------------------------------------------------- writes + addressed reads


async def test_write_then_read_by_id_needs_no_index(store, bank_id):
    """A committed write is visible to a by-id read at once — strong consistency.

    No index pass runs before the read: get-by-id is served from the WAL tail, so
    this is the store proving a write is durable and addressable the moment it
    acks, which is what the retain pipeline depends on.
    """
    unit_ids = await store.insert_facts(
        conn=None,
        ops=None,
        bank_id=bank_id,
        facts=[
            make_fact("the cat sat on the mat", seed=0.1, tags=["animals"]),
            make_fact("stock prices rose today", seed=0.9, tags=["finance"]),
        ],
    )
    assert len(unit_ids) == 2

    got = await store.get_memories(conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=unit_ids)
    by_text = {m.text: m for m in got}
    assert set(by_text) == {"the cat sat on the mat", "stock prices rose today"}
    assert by_text["the cat sat on the mat"].tags == ["animals"]
    assert by_text["the cat sat on the mat"].fact_type == "world"


async def test_search_finds_by_vector_and_text(store, bank_id, index_pass):
    """After indexing, both arms return the fact — the store's whole reason to exist."""
    ns = store._namespace(bank_id)
    await store.insert_facts(
        conn=None,
        ops=None,
        bank_id=bank_id,
        facts=[
            make_fact("a golden retriever chased the ball", seed=0.2),
            make_fact("quarterly revenue beat expectations", seed=0.85),
        ],
    )
    index_pass(ns)

    results = await store.search(
        conn=None,
        bank_id=bank_id,
        fact_types=["world"],
        query_embedding=_vec(0.2),  # nearest the retriever fact
        query_text="retriever ball",
        limit=10,
        min_semantic=0.0,
        min_keyword=0.0,
    )
    semantic, keyword = results["world"]
    assert any("retriever" in r.text for r in semantic), "the dense arm should surface the nearest vector"
    assert any("retriever" in r.text for r in keyword), "the BM25 arm should surface the text match"


async def test_scan_pages_and_counts(store, bank_id, index_pass):
    """Scan walks the whole bank in pages; count is per fact_type."""
    ns = store._namespace(bank_id)
    await store.insert_facts(
        conn=None,
        ops=None,
        bank_id=bank_id,
        facts=[make_fact(f"world fact {i}", seed=0.1 + i * 0.01) for i in range(5)]
        + [make_fact("an experience", seed=0.4, fact_type="experience")],
    )
    index_pass(ns)

    seen: set[str] = set()
    page_token = ""
    while True:
        page = await store.scan_memories(conn=None, fq_table=_fq_table, bank_id=bank_id, limit=2, page_token=page_token)
        seen.update(m.text for m in page.memories)
        page_token = page.next_page_token
        if not page_token:
            break
    assert {f"world fact {i}" for i in range(5)} <= seen
    assert "an experience" in seen

    counts = await store.count_memories(conn=None, fq_table=_fq_table, bank_id=bank_id)
    assert counts.get("world") == 5
    assert counts.get("experience") == 1


async def test_delete_facts_removes_them(store, bank_id):
    unit_ids = await store.insert_facts(conn=None, ops=None, bank_id=bank_id, facts=[make_fact("ephemeral", seed=0.3)])
    await store.delete_facts(bank_id, unit_ids)
    got = await store.get_memories(conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=unit_ids)
    assert got == []


# --------------------------------------------------------------------------- curation archive


async def test_invalidate_then_restore(store, bank_id, index_pass, pg_conn):
    """Invalidate deletes a memory from the index and keeps it in the Postgres archive;
    restore brings it back.

    An invalidated unit is gone from every memlake surface (recall, scan, get) —
    it was deleted, so it never touches the index again — and lives only in
    Postgres' `invalidated_memory_units`, readable, flagged, and restorable. The
    memlake-specific bits the archive columns can't hold ride in a reserved
    metadata key, so the round trip is faithful.
    """
    ns = store._namespace(bank_id)
    unit_ids = await store.insert_facts(
        conn=pg_conn, ops=None, bank_id=bank_id, facts=[make_fact("a curatable fact", seed=0.4, tags=["x"])]
    )
    uid = unit_ids[0]
    index_pass(ns)

    # Invalidate: gone from memlake entirely, present in the Postgres archive.
    assert (
        await store.invalidate_memory(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_id=uid, reason="wrong")
        is True
    )
    assert await store.get_memories(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_ids=[uid]) == []
    archived = await store.get_archived_memory(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_id=uid)
    assert archived is not None and archived.text == "a curatable fact"
    assert archived.tags == ["x"]

    detail = await store.get_memory_unit(conn=pg_conn, ops=None, fq_table=_fq_table, bank_id=bank_id, unit_id=uid)
    assert detail["state"] == "invalidated"
    assert detail["invalidation_reason"] == "wrong"
    assert detail["invalidated_at"]

    # Updating the reason on an already-archived memory sticks.
    await store.set_invalidation_reason(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_id=uid, reason="dup")
    detail = await store.get_memory_unit(conn=pg_conn, ops=None, fq_table=_fq_table, bank_id=bank_id, unit_id=uid)
    assert detail["invalidation_reason"] == "dup"

    # The invalidated tab is a plain SELECT of the archive table — no fold needed.
    page = await store.list_memory_units(
        conn=pg_conn, ops=None, fq_table=_fq_table, bank_id=bank_id, state="invalidated"
    )
    assert uid in {item["id"] for item in page["items"]}
    assert all(item["state"] == "invalidated" for item in page["items"])

    # Restore: written back to memlake (searchable after a fold), gone from the archive.
    restored = await store.restore_memory(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_id=uid)
    assert restored is not None and restored.text == "a curatable fact"
    assert await store.get_archived_memory(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_id=uid) is None
    live = await store.get_memories(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_ids=[uid])
    assert [m.text for m in live] == ["a curatable fact"]
    assert live[0].tags == ["x"]
    detail = await store.get_memory_unit(conn=pg_conn, ops=None, fq_table=_fq_table, bank_id=bank_id, unit_id=uid)
    assert detail["state"] == "valid"


async def test_apply_edit_rewrites_fields_and_entities(store, bank_id):
    """A curation field edit rewrites the memory in place: new text, entities, fact_type.

    This exercises the store's ``apply_edit`` directly — resolving the new entity
    names is Hindsight's job, so the engine hands the resolved ids down, which is
    what a test at this seam supplies. `patch` cannot change entity ids or the
    memory_type, so an edit is a full rewrite; the fields it leaves alone (tags)
    must survive it, and the consolidation state resets so the edit re-consolidates.
    """
    old_entity = "33333333-3333-3333-3333-333333333333"
    new_entity = "44444444-4444-4444-4444-444444444444"
    unit_ids = await store.insert_facts(
        conn=None,
        ops=None,
        bank_id=bank_id,
        facts=[make_fact("original text", seed=0.4, tags=["keep"], entities=[_entity(old_entity, "Old")])],
    )
    uid = unit_ids[0]

    await store.apply_edit(
        conn=None,
        fq_table=_fq_table,
        bank_id=bank_id,
        unit_id=uid,
        text="corrected text",
        context="new context",
        fact_type="experience",
        occurred_start=None,
        occurred_end=None,
        event_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        mentioned_at=None,
        entity_ids=[new_entity],
    )

    got = await store.get_memories(conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=[uid])
    assert len(got) == 1
    edited = got[0]
    assert edited.text == "corrected text"
    assert edited.context == "new context"
    assert edited.fact_type == "experience"  # the fact_type change is a memory_type change
    assert edited.entity_ids == [new_entity]
    assert edited.tags == ["keep"], "fields not part of the edit must survive the rewrite"

    # The re-embed writes through set_memory_embedding (the engine calls it after
    # apply_edit); here it just has to land without error.
    await store.set_memory_embedding(conn=None, fq_table=_fq_table, bank_id=bank_id, unit_id=uid, embedding=_vec(0.9))


async def test_invalidate_missing_is_false(store, bank_id, pg_conn):
    """Invalidating an id that was never live reports it, rather than archiving nothing."""
    missing = store.allocate_unit_ids(1)[0]
    assert (
        await store.invalidate_memory(conn=pg_conn, fq_table=_fq_table, bank_id=bank_id, unit_id=missing, reason=None)
        is False
    )


async def test_delete_document_takes_only_its_memories(store, bank_id, index_pass):
    """A replaced document's facts go, the rest stay — no cascade to lean on."""
    ns = store._namespace(bank_id)
    doc1 = await store.insert_facts(
        conn=None, ops=None, bank_id=bank_id, facts=[make_fact("doc one fact", seed=0.2)], document_id="doc-1"
    )
    doc2 = await store.insert_facts(
        conn=None, ops=None, bank_id=bank_id, facts=[make_fact("doc two fact", seed=0.7)], document_id="doc-2"
    )
    index_pass(ns)

    await store.delete_document(conn=None, fq_table=_fq_table, bank_id=bank_id, document_id="doc-1")

    gone = await store.get_memories(conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=doc1)
    kept = await store.get_memories(conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=doc2)
    assert gone == []
    assert [m.text for m in kept] == ["doc two fact"]


# --------------------------------------------------------------------------- the updated_at push-down


async def test_updated_at_window_is_a_pushdown(store, bank_id, index_pass):
    """`created_after` / `created_before` select on write time, server-side.

    The facts are written with distinct `created_at`, which the store maps to
    memlake's `updated_at`; a windowed search returns only the ones inside it.
    This is the end-to-end view of the block-level push-down: the arm never has to
    surface an out-of-window memory for the caller to drop.
    """
    ns = store._namespace(bank_id)
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def _write(text, seed, created_at):
        # created_at rides on the FactRecord, which the store stamps as updated_at.
        record = FactRecord(
            unit_id=store.allocate_unit_ids(1)[0],
            text=text,
            embedding=_vec(seed),
            fact_type="world",
            created_at=created_at,
        )
        await store._write_records(bank_id, [record])
        return record.unit_id

    await _write("written in 2020", 0.3, old)
    await _write("written in 2026", 0.3, new)
    index_pass(ns)

    results = await store.search(
        conn=None,
        bank_id=bank_id,
        fact_types=["world"],
        query_embedding=_vec(0.3),
        query_text="",
        limit=10,
        created_after=datetime(2025, 1, 1, tzinfo=timezone.utc),
        min_semantic=0.0,
        min_keyword=0.0,
    )
    semantic, _ = results["world"]
    texts = [r.text for r in semantic]
    assert "written in 2026" in texts
    assert "written in 2020" not in texts, "the window is a push-down; the old write must not arrive"


# --------------------------------------------------------------------------- consolidation state


async def test_unconsolidated_then_marked(store, bank_id, index_pass):
    """A fresh fact is a consolidation candidate; marking it takes it out of the queue."""
    ns = store._namespace(bank_id)
    unit_ids = await store.insert_facts(
        conn=None, ops=None, bank_id=bank_id, facts=[make_fact("a consolidatable fact", seed=0.5)]
    )
    index_pass(ns)

    pending = await store.find_unconsolidated(
        conn=None, fq_table=_fq_table, bank_id=bank_id, fact_types=["world"], limit=50
    )
    assert unit_ids[0] in {m.unit_id for m in pending}

    await store.mark_consolidated(
        conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=unit_ids, when=datetime.now(timezone.utc)
    )
    index_pass(ns)

    pending_after = await store.find_unconsolidated(
        conn=None, fq_table=_fq_table, bank_id=bank_id, fact_types=["world"], limit=50
    )
    assert unit_ids[0] not in {m.unit_id for m in pending_after}


async def test_consolidation_failure_also_leaves_the_queue(store, bank_id, index_pass):
    """A fact the LLM could not summarise must stop being a candidate too."""
    ns = store._namespace(bank_id)
    unit_ids = await store.insert_facts(
        conn=None, ops=None, bank_id=bank_id, facts=[make_fact("an unsummarisable fact", seed=0.5)]
    )
    index_pass(ns)

    await store.mark_consolidated(
        conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=unit_ids, when=datetime.now(timezone.utc), failed=True
    )
    index_pass(ns)

    pending = await store.find_unconsolidated(
        conn=None, fq_table=_fq_table, bank_id=bank_id, fact_types=["world"], limit=50
    )
    assert unit_ids[0] not in {m.unit_id for m in pending}, "a failed fact is not a candidate"


# --------------------------------------------------------------------------- observations (denormalized)


async def test_observation_round_trips_by_source(store, bank_id, index_pass):
    """An observation is found by its sources, and dies when a source is deleted."""
    ns = store._namespace(bank_id)
    sources = await store.insert_facts(
        conn=None,
        ops=None,
        bank_id=bank_id,
        facts=[make_fact("source one", seed=0.2), make_fact("source two", seed=0.25)],
    )
    observation = FactRecord(
        unit_id=store.allocate_unit_ids(1)[0],
        text="a summary of one and two",
        embedding=_vec(0.22),
        fact_type="observation",
        source_memory_ids=sources,
        created_at=datetime.now(timezone.utc),
    )
    await store.upsert_observation(conn=None, bank_id=bank_id, record=observation)
    index_pass(ns)

    found = await store.observations_for_sources(
        conn=None, ops=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=[sources[0]]
    )
    assert observation.unit_id in {o.unit_id for o in found}

    # Deleting one source invalidates the observation and requeues the survivors.
    removed = await store.delete_stale_observations(
        conn=None, ops=None, fq_table=_fq_table, bank_id=bank_id, fact_ids=[sources[0]]
    )
    assert removed == 1
    index_pass(ns)

    still = await store.observations_for_sources(
        conn=None, ops=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=sources
    )
    assert observation.unit_id not in {o.unit_id for o in still}


# --------------------------------------------------------------------------- entities


async def test_entity_ids_ride_on_the_memory_and_names_come_from_postgres(store, bank_id):
    """Entity ids travel with the memory; only their names need the registry."""
    entity_id = "11111111-1111-1111-1111-111111111111"
    unit_ids = await store.insert_facts(
        conn=None,
        ops=None,
        bank_id=bank_id,
        facts=[make_fact("Ada wrote the first algorithm", seed=0.4, entities=[_entity(entity_id, "Ada Lovelace")])],
    )

    # Ids alone need no Postgres.
    ids_map = await store.entities_for_units(conn=None, fq_table=_fq_table, bank_id=bank_id, unit_ids=unit_ids)
    assert ids_map[unit_ids[0]] == [entity_id]

    # Names are the one reach back into the Postgres registry.
    named = await store.entity_map_for_units(
        conn=FakeConn({entity_id: "Ada Lovelace"}), fq_table=_fq_table, bank_id=bank_id, unit_ids=unit_ids
    )
    assert named[unit_ids[0]] == [{"entity_id": entity_id, "canonical_name": "Ada Lovelace"}]


# --------------------------------------------------------------------------- graph view


async def test_graph_units_page_filters_and_counts(store, bank_id, index_pass):
    """The graph endpoint's node read: a filtered page plus the total.

    The three reads behind ``get_graph_data`` are the store's — nodes with their
    filters, the entity rows for their names, and the memory-to-memory edges. This
    exercises the node read (page + total, filtered by fact_type and document) and
    the entity-row read that names them.
    """
    ns = store._namespace(bank_id)
    entity_id = "22222222-2222-2222-2222-222222222222"
    await store.insert_facts(
        conn=None,
        ops=None,
        bank_id=bank_id,
        facts=[
            make_fact("graph world a", seed=0.2, document_id="d1", entities=[_entity(entity_id, "Grace Hopper")]),
            make_fact("graph world b", seed=0.3, document_id="d2"),
            make_fact("a graph experience", seed=0.4, fact_type="experience", document_id="d1"),
        ],
    )
    index_pass(ns)

    # Filter to one fact_type: the page and the total both reflect the filter.
    page = await store.graph_units(conn=None, fq_table=_fq_table, bank_id=bank_id, fact_type="world", limit=100)
    texts = {u["text"] for u in page["units"]}
    assert texts == {"graph world a", "graph world b"}
    assert page["total"] == 2

    # And by document, across fact_types.
    by_doc = await store.graph_units(conn=None, fq_table=_fq_table, bank_id=bank_id, document_id="d1", limit=100)
    assert {u["text"] for u in by_doc["units"]} == {"graph world a", "a graph experience"}

    # The entity rows that put names on the nodes come from the store too.
    unit_ids = [u["id"] for u in page["units"]]
    rows = await store.graph_entity_rows(
        conn=FakeConn({entity_id: "Grace Hopper"}), fq_table=_fq_table, bank_id=bank_id, unit_ids=unit_ids
    )
    assert {(r["unit_id"], r["canonical_name"]) for r in rows} >= {
        (uid, "Grace Hopper")
        for uid in unit_ids
        if any(u["id"] == uid and u["text"] == "graph world a" for u in page["units"])
    }


# --------------------------------------------------------------------------- staleness signal


async def test_any_memory_updated_since(store, bank_id):
    """The mental-model staleness probe: cheap, bounded, and time-scoped."""
    before = datetime.now(timezone.utc) - timedelta(minutes=1)
    await store.insert_facts(conn=None, ops=None, bank_id=bank_id, facts=[make_fact("just written", seed=0.5)])

    assert await store.any_memory_updated_since(conn=None, fq_table=_fq_table, bank_id=bank_id, since=before) is True
    future = datetime.now(timezone.utc) + timedelta(days=1)
    assert await store.any_memory_updated_since(conn=None, fq_table=_fq_table, bank_id=bank_id, since=future) is False
