"""memlake as the store for Hindsight's memories and their links.

memlake is an S3-native retrieval engine that serves the dense, full-text, graph
and temporal arms over one storage layer. A bank maps to a *namespace*, a
fact_type to a *memory_type*, and every hit comes back with the stored memory
inline — so with this extension installed **nothing memory-shaped is written to
Postgres**:

* facts are not inserted into `memory_units`; ids are minted by
  :meth:`allocate_unit_ids` before the write
* the unit→entity posting rides on the memory as `entity_ids`, so `unit_entities`
  is never written (memlake keeps a persisted entity index and expands through it)
* causal edges ride on the memory as `causal_out`, and semantic links are derived
  by memlake's indexer — so `memory_links` is never written
* recall reads the whole result row off the hit payload, so there is no hydration
  query

Columns memlake has no first-class model of (context, document_id, chunk_id, the
user metadata JSON, …) travel in its opaque metadata bag, stored verbatim and
returned on every hit. Nothing in the bag is indexed and the only predicate over
it is equality, which is why every filter that needs more than that is applied in
Python here (and can therefore return short pages).

Postgres still holds documents, chunks, banks, operations and the `entities`
registry, so the handful of methods that resolve entity *names* still take a
connection; everything else ignores the `conn` / `ops` / `fq_table` handles the
interface hands them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any

import memlake_client as mc
from hindsight_api.engine.memories.base import (
    CONSOLIDATED_NO,
    CONSOLIDATED_YES,
    FACT_TYPE_TO_MEMORY_TYPE,
    MEMORY_TYPE_TO_FACT_TYPE,
    META_CHUNK_ID,
    META_CONSOLIDATED_AT,
    META_CONSOLIDATED_FLAG,
    META_CONTEXT,
    META_CREATED_AT,
    META_DOCUMENT_ID,
    META_METADATA_JSON,
    META_SOURCE_MEMORY_IDS,
    META_UPDATED_AT,
    DeletePredicate,
    FactRecord,
    MemoriesExtension,
    MemoryPatch,
    ScanPage,
    StoredMemory,
    build_fact_records,
)
from memlake_client.v1 import memlake_pb2 as pb

from . import observations, reads

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hindsight_api.engine.search.graph_retrieval import GraphRetriever

logger = logging.getLogger(__name__)

#: Default server address. Overridden with ``HINDSIGHT_API_MEMORIES_TARGET``,
#: which may name several comma-separated nodes — the client rendezvous-hashes
#: namespaces across them and fails over on its own.
DEFAULT_TARGET = "localhost:50051"

# TagsMatch spellings Hindsight uses -> memlake's enum names. Hindsight also
# supports nested tag *groups* (AND/OR/NOT trees); memlake has only these five
# flat modes, so groups are applied in Python after the query.
_TAGS_MATCH = {
    "any": "ANY",
    "all": "ALL",
    "any_strict": "ANY_STRICT",
    "all_strict": "ALL_STRICT",
    "exact": "EXACT",
}

# Hindsight causal relation names -> memlake's LinkType enum names.
_LINK_TYPE = {
    "causes": "CAUSES",
    "caused_by": "CAUSED_BY",
    "enables": "ENABLES",
    "prevents": "PREVENTS",
}

#: When consolidation gave up on a memory. Postgres has a `consolidation_failed_at`
#: column; here it is one more key in the opaque bag. It exists to be *written*:
#: nothing reads it back through the interface (:class:`StoredMemory` models no
#: such field), but stamping it alongside the consolidated flag is what keeps a
#: memory the LLM could not summarise out of the candidate query forever after.
META_CONSOLIDATION_FAILED_AT = "consolidation_failed_at"

#: A third value for :data:`META_CONSOLIDATED_FLAG`, beyond the base's "0"/"1".
#: The base flag is binary — consolidated or not — but the curation UI's
#: consolidation_state facet wants to tell "done" from "failed", and the metadata
#: bag only filters by equality. Giving failure its own value makes all three
#: states a single pushable equality. It reads as "not a candidate" to
#: :meth:`find_unconsolidated`, which selects on the flag being exactly "0", so a
#: failed memory stays out of the queue just as a succeeded one does.
CONSOLIDATED_FAILED = "2"

#: Metadata keys the namespace declares for MetadataStats value-counts, so the count
#: surfaces read a per-segment tally instead of walking the corpus. document_id backs the
#: per-document counts; the consolidated flag backs get_bank_freshness's pending/failed.
_INDEXED_METADATA_KEYS = [META_DOCUMENT_ID, META_CONSOLIDATED_FLAG]

#: Reserved key inside an archived row's `metadata` holding the full memlake
#: memory (vector, memory_type, causal edges, the raw metadata bag) — everything
#: the archive's memory_units-shaped columns cannot hold — so restore rebuilds it
#: faithfully. Stripped from the metadata the read surfaces return.
_ARCHIVE_PAYLOAD_KEY = "__memlake_archive__"

#: Columns read back from the archive table for the detail/list views and restore.
_ARCHIVE_SELECT = (
    "id, text, fact_type, context, event_date, occurred_start, occurred_end, mentioned_at, "
    "document_id, chunk_id, tags, metadata, proof_count, created_at, consolidated_at, "
    "entity_ids, invalidation_reason, invalidated_at"
)


def _to_epoch_ms(dt: datetime | None) -> int | None:
    """memlake carries all timestamps as epoch milliseconds."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _from_epoch_ms(ms: int | None) -> datetime | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _is_missing_namespace(err: Exception) -> bool:
    """Whether an error is memlake's "namespace has no manifest".

    A namespace is created lazily on a bank's first write, so a bank that exists
    (in Postgres) but has never been written to has no manifest yet. A *read*
    against it — a list, a count, a facet on a freshly-created bank — is "nothing
    there yet", not a failure, so the read paths map this to an empty result
    instead of propagating it. memlake surfaces it as an INTERNAL rpc whose
    message carries this text.
    """
    return "no manifest" in str(err)


def _parse_embedding(query_embedding: str | list[float]) -> list[float]:
    """Recall passes embeddings around as the pgvector literal '[0.1,0.2,...]'."""
    if isinstance(query_embedding, list):
        return query_embedding
    return [float(x) for x in query_embedding.strip().lstrip("[").rstrip("]").split(",") if x.strip()]


def _tags_mode(tags_match: str) -> int:
    return getattr(mc, _TAGS_MATCH.get(tags_match, "ANY"))


class MemlakeMemories(MemoriesExtension):
    """The memories slice of Hindsight's storage, served by memlake.

    Configured through the extension environment::

        HINDSIGHT_API_MEMORIES_EXTENSION=hindsight_memlake:MemlakeMemories
        HINDSIGHT_API_MEMORIES_TARGET=localhost:50051
        HINDSIGHT_API_MEMORIES_NAMESPACE_PREFIX=prod-
        HINDSIGHT_API_MEMORIES_NPROBE=16
    """

    name = "memlake"

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        target = (config.get("target") or DEFAULT_TARGET).strip()
        # A comma-separated list is a cluster: the client hashes each namespace to
        # a preferred node for cache and commit affinity, and falls over to the
        # next one if it is down.
        self._target: str | list[str] = [t.strip() for t in target.split(",") if t.strip()] if "," in target else target
        self._namespace_prefix = config.get("namespace_prefix", "")
        # Coverage, not depth: candidates in unprobed clusters are unreachable no
        # matter how large a top_k the arms ask for. 0 keeps the server default.
        self._nprobe = int(config.get("nprobe") or 0)
        self._client: Any = None
        self._ensured: set[str] = set()
        self._graph_retriever: "GraphRetriever | None" = None

    # -- lifecycle -----------------------------------------------------------

    async def initialize(self) -> None:
        self._client = mc.MemlakeClient(self._target)
        logger.info("[memories] memlake connected to %s (postgres holds no memories)", self._target)

    async def shutdown(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None

    def _namespace(self, bank_id: str) -> str:
        return f"{self._namespace_prefix}{bank_id}"

    async def ensure_namespace(self, bank_id: str) -> None:
        ns = self._namespace(bank_id)
        if ns in self._ensured:
            return
        await asyncio.to_thread(
            partial(self._client.create_namespace, ns, indexed_metadata_keys=_INDEXED_METADATA_KEYS)
        )
        self._ensured.add(ns)

    async def _metadata_stats(self, bank_id: str, key: str) -> dict:
        """`{value: count}` for a declared key: the per-value tally memlake keeps for
        each key named in ``indexed_metadata_keys`` at namespace creation.

        Empty for a bank that has never been written to (no namespace yet) or a key
        that carries no value on any live memory — both mean "no counts", so the
        callers read it the same way.
        """
        try:
            return await asyncio.to_thread(partial(self._client.metadata_stats, self._namespace(bank_id), key))
        except Exception as e:
            if _is_missing_namespace(e):
                return {}
            raise

    # -- writes --------------------------------------------------------------

    async def insert_facts(
        self,
        *,
        conn,
        ops,
        bank_id: str,
        facts: list,
        document_id: str | None = None,
        defer_index: bool = False,
    ) -> list[str]:
        """Mint ids for a batch, and write it unless the caller deferred.

        No `memory_units` row is written, so ids cannot come from a RETURNING
        clause — they are minted here. ``conn`` and ``ops`` are unused for the
        same reason.

        The deferred path exists because the retain orchestrator can only supply
        entity ids and causal edges once Phase-1 placeholders have been remapped
        onto these ids; it comes back through :meth:`index_facts` with the
        complete picture, and a memory written twice is an upsert, not a
        duplicate — but writing it twice would still cost a WAL entry and leave
        the first version briefly recallable with no entities.
        """
        unit_ids = self.allocate_unit_ids(len(facts))
        if not defer_index:
            await self._write_records(bank_id, build_fact_records(unit_ids, facts, document_id))
        return unit_ids

    async def index_facts(
        self,
        bank_id: str,
        unit_ids: list[str],
        facts: list[Any],
        document_id: str | None = None,
        unit_entity_ids: dict[str, list[str]] | None = None,
    ) -> None:
        """Write facts whose ids came from a deferred :meth:`insert_facts`.

        ``unit_entity_ids`` is the unit→entity posting that would otherwise become
        `unit_entities` rows; causal relations become the memory's causal edges.
        Both travel with the memory, which is why this is a single write rather
        than an insert followed by link inserts.
        """
        await self._write_records(bank_id, build_fact_records(unit_ids, facts, document_id, unit_entity_ids))

    async def _write_records(self, bank_id: str, records: list[FactRecord]) -> None:
        """Upsert a batch of complete records. The one write path in this module."""
        if not records:
            return
        await self.ensure_namespace(bank_id)

        memories = []
        for r in records:
            memory_type = FACT_TYPE_TO_MEMORY_TYPE.get(r.fact_type)
            if memory_type is None:
                # An unmapped fact_type would land in a memory_type nothing queries.
                logger.warning("[memories] skipping fact %s: unmapped fact_type %r", r.unit_id, r.fact_type)
                continue
            m = mc.memory(
                r.text,
                # Retain hands over a float list, consolidation the pgvector literal.
                _parse_embedding(r.embedding),
                memory_type=memory_type,
                id=uuid.UUID(r.unit_id).bytes,
                tags=r.tags,
                proof_count=r.proof_count,
                # 16-byte EntityIds — the same UUIDs the `entities` registry uses.
                entity_ids=[uuid.UUID(e).bytes for e in r.entity_ids],
                metadata=r.metadata_bag(),
                # Entity names and spelled-out dates are folded into the BM25 document
                # without changing the text a hit returns — the same enrichment the
                # postgres path puts in `text_signals`.
                index_text=(f"{r.text} {r.text_signals}" if r.text_signals else None),
                # Epoch ms; these are what the temporal arm ranks on. Effective time
                # is COALESCE(occurred_start, mentioned_at, occurred_end), matching
                # how Hindsight's own temporal path coalesces.
                event_date=_to_epoch_ms(r.event_date),
                updated_at=_to_epoch_ms(r.created_at),
                occurred_start=_to_epoch_ms(r.occurred_start),
                occurred_end=_to_epoch_ms(r.occurred_end),
                mentioned_at=_to_epoch_ms(r.mentioned_at),
            )
            # `memory()` still takes no causal edges, so those go on the message.
            for edge in r.causal_edges:
                link_type = _LINK_TYPE.get(edge.relation_type)
                if link_type is None:
                    logger.warning("[memories] dropping causal edge of unknown type %r", edge.relation_type)
                    continue
                m.causal_out.add(
                    target=uuid.UUID(edge.target_unit_id).bytes,
                    link_type=getattr(pb, link_type),
                    weight=edge.weight,
                )
            memories.append(m)

        if not memories:
            return
        seq = await asyncio.to_thread(self._client.write, self._namespace(bank_id), memories)
        logger.debug("[memories] wrote %d memory/ies to memlake ns=%s seq=%s", len(memories), bank_id, seq)

    async def delete_facts(self, bank_id: str, unit_ids: list[str]) -> None:
        if not unit_ids:
            return
        ids = [uuid.UUID(u).bytes for u in unit_ids]
        await asyncio.to_thread(self._client.delete, self._namespace(bank_id), ids)
        logger.debug("[memories] tombstoned %d memory/ies in memlake ns=%s", len(unit_ids), bank_id)

    async def delete_where(self, bank_id: str, predicate: DeletePredicate) -> int:
        if predicate.is_empty() and not predicate.delete_all:
            raise ValueError("Refusing an empty delete predicate; set delete_all to wipe the bank")

        memory_types = [
            FACT_TYPE_TO_MEMORY_TYPE[ft] for ft in (predicate.fact_types or []) if ft in FACT_TYPE_TO_MEMORY_TYPE
        ]
        deleted = await asyncio.to_thread(
            partial(
                self._client.delete_by_predicate,
                self._namespace(bank_id),
                metadata_equals=predicate.metadata_equals,
                tags=predicate.tags,
                tags_mode=_tags_mode(predicate.tags_match),
                memory_types=memory_types,
                delete_all=predicate.delete_all,
                # Lazy: one atomic WAL op, materialized at the next fold, and
                # race-closed — it only removes writes older than its own sequence,
                # so facts written after it (a re-ingest's replacements) survive.
                # The interface allows this, at the cost of returning 0 rather
                # than a scanned count.
                eager=False,
            )
        )
        logger.debug("[memories] predicate-deleted in memlake ns=%s (%s)", bank_id, predicate)
        return deleted

    async def delete_document(self, *, conn, fq_table, bank_id: str, document_id: str) -> None:
        """Predicate-delete a document's memories by the id they carry.

        This races the replacement facts a re-ingest is about to write, and is
        safe *because* the delete is lazy: the WAL op only removes memories
        written before its own sequence, so facts landing moments later are
        spared even though the delete was issued first. An eager delete would
        have to be ordered against the write instead.
        """
        await self.delete_where(bank_id, DeletePredicate(metadata_equals={META_DOCUMENT_ID: document_id}))

    async def delete_namespace(self, bank_id: str) -> None:
        namespace = self._namespace(bank_id)
        try:
            objects = await asyncio.to_thread(self._client.delete_namespace, namespace)
        except Exception as e:
            # Callers delete defensively — `delete_bank` runs before ingest to clear
            # a previous run — so dropping a namespace that was never created has to
            # be a no-op, not an error.
            if _is_missing_namespace(e):
                self._ensured.discard(namespace)
                logger.debug("[memories] memlake namespace %s does not exist; nothing to drop", namespace)
                return
            raise
        self._ensured.discard(namespace)
        logger.info("[memories] dropped memlake namespace %s (%s objects)", namespace, objects)

    async def delete_observations(self, *, conn, fq_table, bank_id: str) -> None:
        """Drop every observation, leaving the facts behind them.

        ``delete_all`` is what makes this legal with no metadata or tag condition:
        the memory_type restriction is not part of :meth:`DeletePredicate.is_empty`,
        so without the flag the guard would refuse a predicate that is in fact
        narrow.
        """
        await self.delete_where(bank_id, DeletePredicate(fact_types=["observation"], delete_all=True))

    async def update_memories(self, bank_id: str, patches: list[MemoryPatch]) -> None:
        if not patches:
            return
        ops = []
        for p in patches:
            fields: dict[str, Any] = {"proof_count_delta": p.proof_count_delta}
            if p.text is not None:
                fields["text"] = p.text
            if p.embedding is not None:
                fields["vector"] = _parse_embedding(p.embedding)
            if p.tags is not None:
                fields["tags"] = p.tags
            for attr in ("event_date", "occurred_start", "occurred_end", "mentioned_at"):
                value = _to_epoch_ms(getattr(p, attr))
                if value is not None:
                    fields[attr] = value
            if p.metadata:
                fields["metadata"] = p.metadata
            ops.append(self._client.patch(uuid.UUID(p.unit_id).bytes, **fields))

        await asyncio.to_thread(self._client.write_ops, self._namespace(bank_id), ops)
        logger.debug("[memories] patched %d memory/ies in memlake ns=%s", len(ops), bank_id)

    # -- recall arms ---------------------------------------------------------

    async def search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        query_text: str,
        limit: int,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
    ) -> dict[str, tuple[list, list]]:
        from hindsight_api.config import get_config
        from hindsight_api.engine.search.tags import filter_results_by_tag_groups

        result: dict[str, tuple[list, list]] = {ft: ([], []) for ft in fact_types}
        memory_types = [FACT_TYPE_TO_MEMORY_TYPE[ft] for ft in fact_types if ft in FACT_TYPE_TO_MEMORY_TYPE]
        if not memory_types:
            return result

        config = get_config()
        sem_min = min_semantic if min_semantic is not None else config.semantic_min_similarity
        bm25_min = min_keyword if min_keyword is not None else config.bm25_min_score

        # Over-fetch to match the SQL path's ANN compensation and to leave room for
        # the filters memlake cannot push down (tag groups).
        depth = max(limit * 5, 100)
        hits = await self._query(
            bank_id=bank_id,
            memory_types=memory_types,
            vector=_parse_embedding(query_embedding),
            text=query_text,
            tags=tags,
            tags_match=tags_match,
            vector_top_k=depth,
            text_top_k=depth,
            # The dense and full-text arms are what this call is for; the graph arm
            # is retrieved separately, per fact_type, by MemlakeGraphRetriever.
            graph_top_k=0,
            # The updated_at window is applied server-side, so the rows that arrive
            # are already inside it.
            updated_from=_to_epoch_ms(created_after),
            updated_to=_to_epoch_ms(created_before),
        )

        per_type: dict[int, tuple[list, list]] = {mt: ([], []) for mt in memory_types}
        for hit in hits:
            row = _row_from_hit(hit)
            if row is None:
                continue
            buckets = per_type.get(hit.memory_type)
            if buckets is None:
                continue
            # memlake's arm scores are the native ones — dense is cosine similarity,
            # text is BM25 — the same scales the SQL arms produce, so Hindsight's
            # score floors apply unchanged.
            if hit.dense.present and hit.dense.score >= sem_min:
                buckets[0].append((hit.dense.rank, _to_result(row, similarity=hit.dense.score)))
            if hit.text.present and hit.text.score >= bm25_min:
                buckets[1].append((hit.text.rank, _to_result(row, bm25_score=hit.text.score)))

        for memory_type, (semantic, keyword) in per_type.items():
            fact_type = MEMORY_TYPE_TO_FACT_TYPE.get(memory_type)
            if fact_type not in result:
                continue
            # Hits come back unordered; each arm's own rank restores its ordering.
            sem = [r for _, r in sorted(semantic, key=lambda p: p[0])]
            keys = [r for _, r in sorted(keyword, key=lambda p: p[0])]
            if tag_groups:
                sem = filter_results_by_tag_groups(sem, tag_groups)
                keys = filter_results_by_tag_groups(keys, tag_groups)
            result[fact_type] = (sem[:limit], keys[:limit])

        return result

    async def temporal_search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
    ) -> dict[str, list]:
        from hindsight_api.engine.search.tags import filter_results_by_tag_groups

        results: dict[str, list] = {ft: [] for ft in fact_types}
        memory_types = [FACT_TYPE_TO_MEMORY_TYPE[ft] for ft in fact_types if ft in FACT_TYPE_TO_MEMORY_TYPE]
        if not memory_types:
            return results

        hits = await self._query(
            bank_id=bank_id,
            memory_types=memory_types,
            vector=_parse_embedding(query_embedding),
            text=None,
            tags=tags,
            tags_match=tags_match,
            # The arm ranks entry points by similarity before spreading, so the
            # dense arm has to run; only temporal-surfaced candidates are kept.
            vector_top_k=limit,
            text_top_k=0,
            graph_top_k=0,
            temporal_from=_to_epoch_ms(start_date),
            temporal_to=_to_epoch_ms(end_date),
        )

        ranked: dict[str, list] = {ft: [] for ft in fact_types}
        for hit in hits:
            if not hit.temporal.present:
                continue
            fact_type = MEMORY_TYPE_TO_FACT_TYPE.get(hit.memory_type)
            if fact_type not in ranked:
                continue
            row = _row_from_hit(hit)
            if row is None:
                continue
            ranked[fact_type].append((hit.temporal.rank, _to_result(row, temporal_score=hit.temporal.score)))

        for fact_type, scored in ranked.items():
            scored.sort(key=lambda p: p[0])
            hits_for_type = [r for _, r in scored]
            # Tag groups are a boolean tree memlake's flat tag modes cannot express,
            # so they run here and can trim below `limit`.
            if tag_groups:
                hits_for_type = filter_results_by_tag_groups(hits_for_type, tag_groups)
            results[fact_type] = hits_for_type[:limit]
        return results

    async def graph_search(
        self,
        *,
        bank_id: str,
        fact_type: str,
        query_embedding: str,
        query_text: str | None,
        limit: int,
        tags: list[str] | None = None,
        tags_match: str = "any",
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list:
        """The graph arm for one fact_type — what :class:`MemlakeGraphRetriever` calls.

        Not an interface method: the graph arm reaches the store through the
        retriever :meth:`graph_retriever` hands back, so this is the store's own
        surface rather than something every implementation must have.
        """
        memory_type = FACT_TYPE_TO_MEMORY_TYPE.get(fact_type)
        if memory_type is None:
            return []

        hits = await self._query(
            bank_id=bank_id,
            memory_types=[memory_type],
            vector=_parse_embedding(query_embedding),
            text=None,
            tags=tags,
            tags_match=tags_match,
            # The graph arm expands from the dense probe, so it needs the dense arm
            # to seed it; only graph-surfaced candidates are kept below.
            vector_top_k=limit,
            text_top_k=0,
            graph_top_k=max(limit * 2, 50),
            # Recall's window is applied server-side, on the same `updated_at` the
            # dense arm filters on — so the graph arm agrees with the others
            # instead of approximating the window from a content timestamp.
            updated_from=_to_epoch_ms(created_after),
            updated_to=_to_epoch_ms(created_before),
        )

        ranked = []
        for hit in hits:
            if not hit.graph.present:
                continue
            row = _row_from_hit(hit)
            if row is None:
                continue
            ranked.append((hit.graph.rank, _to_result(row, activation=hit.graph.score)))
        ranked.sort(key=lambda p: p[0])
        return [r for _, r in ranked[:limit]]

    def graph_retriever(self) -> "GraphRetriever | None":
        """memlake owns the links, so the SQL retrievers would walk empty tables.

        Built once and cached: the recall pipeline resolves the retriever per
        request, and there is no per-call state to keep.
        """
        if self._graph_retriever is None:
            from .graph import MemlakeGraphRetriever

            self._graph_retriever = MemlakeGraphRetriever(self)
        return self._graph_retriever

    async def _query(
        self,
        *,
        bank_id: str,
        memory_types: list[int],
        vector: list[float],
        text: str | None,
        tags: list[str] | None,
        tags_match: str,
        vector_top_k: int,
        text_top_k: int,
        graph_top_k: int,
        temporal_from: int | None = None,
        temporal_to: int | None = None,
        updated_from: int | None = None,
        updated_to: int | None = None,
    ) -> list[Any]:
        """One Query RPC covering every requested memory_type.

        memlake runs each type's arms concurrently over a single snapshot, so the
        storage reads coalesce into shared roundtrip waves — one call for N types
        costs the waves of one, not N.
        """
        return await asyncio.to_thread(
            partial(
                self._client.query,
                self._namespace(bank_id),
                vector=vector,
                text=text or None,
                memory_types=memory_types,
                tags=tags,
                tags_mode=_tags_mode(tags_match),
                vector_top_k=vector_top_k,
                text_top_k=text_top_k,
                graph_top_k=graph_top_k,
                nprobe=self._nprobe,
                temporal_from=temporal_from,
                temporal_to=temporal_to,
                updated_from=updated_from,
                updated_to=updated_to,
            )
        )

    # -- addressed reads -----------------------------------------------------
    #
    # Query answers "what is relevant"; these answer "what is stored". None of
    # them touch Postgres, so the `conn` / `fq_table` handles go unused — except
    # where a *name* from the `entities` registry is needed, which is why the
    # helpers in `reads` take them.

    async def get_memories(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[StoredMemory]:
        if not unit_ids:
            return []
        try:
            records = await asyncio.to_thread(
                self._client.get, self._namespace(bank_id), [uuid.UUID(u).bytes for u in unit_ids]
            )
        except Exception as e:
            if _is_missing_namespace(e):
                return []
            raise
        return [_stored_from_record(r) for r in records]

    async def get_memories_with_edges(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[StoredMemory]:
        """Get by id *with* the derived semantic edges populated.

        Internal to this package (the graph view's `graph_direct_links` is the one
        caller). The ranking path never wants edges and would pay the bytes on
        every candidate, so `get_memories` leaves them off; here they are the
        point.
        """
        if not unit_ids:
            return []
        records = await asyncio.to_thread(
            partial(
                self._client.get, self._namespace(bank_id), [uuid.UUID(u).bytes for u in unit_ids], include_edges=True
            )
        )
        return [_stored_from_record(r) for r in records]

    async def scan_memories(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str] | None = None,
        limit: int = 100,
        page_token: str = "",
        tags: list[str] | None = None,
        tags_match: str = "any",
        document_id: str | None = None,
        metadata_equals: dict[str, str] | None = None,
        skip: int = 0,
        include_edges: bool = False,
    ) -> ScanPage:
        # `document_id` is a real column in Postgres; here it lives in the opaque
        # bag under META_DOCUMENT_ID (a declared indexed key), so the filter is
        # just one more metadata equality folded in alongside any the caller gave.
        if document_id is not None:
            metadata_equals = {**(metadata_equals or {}), META_DOCUMENT_ID: document_id}
        return await self.scan_raw(
            bank_id,
            fact_types=fact_types,
            limit=limit,
            page_token=page_token,
            tags=tags,
            tags_match=tags_match,
            metadata_equals=metadata_equals,
            skip=skip,
            include_edges=include_edges,
        )

    async def scan_raw(
        self,
        bank_id: str,
        *,
        fact_types: list[str] | None = None,
        limit: int = 100,
        page_token: str = "",
        tags: list[str] | None = None,
        tags_match: str = "any",
        metadata_equals: dict[str, str] | None = None,
        skip: int = 0,
        include_edges: bool = False,
        updated_from: int | None = None,
        updated_to: int | None = None,
    ) -> ScanPage:
        """Scan with the push-downs the interface's :meth:`scan_memories` omits.

        The `updated_at` window is one of them, and it is what turns the staleness
        check from a corpus walk in Python into a single filtered call. Internal
        to this package — call `scan_memories` from anywhere else.

        The server fills a page before returning it, so one call with a selective
        filter walks as far as it must rather than handing back empty pages: an
        empty result with no cursor means "nothing matches", not "keep going".
        """
        memory_types = [FACT_TYPE_TO_MEMORY_TYPE[ft] for ft in (fact_types or []) if ft in FACT_TYPE_TO_MEMORY_TYPE]
        try:
            response = await asyncio.to_thread(
                partial(
                    self._client.scan,
                    self._namespace(bank_id),
                    memory_types=sorted(memory_types),
                    page_token=page_token,
                    limit=limit,
                    metadata_equals=metadata_equals,
                    tags=tags,
                    tags_mode=_tags_mode(tags_match),
                    skip=skip,
                    include_edges=include_edges,
                    updated_from=updated_from,
                    updated_to=updated_to,
                )
            )
        except Exception as e:
            if _is_missing_namespace(e):
                return ScanPage(memories=[], next_page_token="")
            raise
        return ScanPage(
            memories=[_stored_from_record(r) for r in response.memories],
            next_page_token=response.next_page_token,
        )

    async def count_memories(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        try:
            response = await asyncio.to_thread(self._client.stats, self._namespace(bank_id))
        except Exception as e:
            if _is_missing_namespace(e):
                return {}
            raise
        counts: dict[str, int] = {}
        for type_stats in response.types:
            fact_type = MEMORY_TYPE_TO_FACT_TYPE.get(type_stats.memory_type)
            if fact_type is not None:
                # doc_count is already live: indexed generation, minus tombstones,
                # plus the un-indexed WAL tail.
                counts[fact_type] = type_stats.doc_count
        return counts

    async def link_counts(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        """Live link totals for the bank stats page, keyed by link type.

        In memlake the links live inside the memory — derived semantic (kNN) edges and
        intrinsic causal edges — so this is a metadata read of the fold-time per-segment
        tally plus the WAL tail (LinkStats), never a corpus walk. There is no separate
        "entity" link type: shared-entity affinity is folded into the derived semantic
        links. Zero-valued types are omitted; an un-written bank has no links.
        """
        try:
            stats = await asyncio.to_thread(self._client.link_stats, self._namespace(bank_id))
        except Exception as e:
            if _is_missing_namespace(e):
                return {}
            raise
        return {k: v for k, v in stats.items() if v}

    async def list_tags(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        return await reads.list_tags(self, conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def list_memory_units(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str,
        fact_type: str | None = None,
        search_query: str | None = None,
        consolidation_state: str | None = None,
        state: str | None = None,
        document_id: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        # `ops` is Postgres dialect ops; the memlake reads never touch it.
        return await reads.list_memory_units(
            self,
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_type=fact_type,
            search_query=search_query,
            consolidation_state=consolidation_state,
            state=state,
            document_id=document_id,
            tags=tags,
            tags_match=tags_match,
            limit=limit,
            offset=offset,
        )

    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id: str, unit_id: str) -> dict[str, Any] | None:
        return await reads.get_memory_unit(self, conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id)

    # -- curation archive ----------------------------------------------------
    #
    # Invalidation deletes the memory from the bank's namespace — so it never
    # touches the IVF/FTS index again — and stashes it in Postgres'
    # `invalidated_memory_units`, the archive table that already exists for this
    # exact purpose (cold storage, no index). Restore writes it back. Everything
    # memlake-specific that the archive columns cannot hold — the vector, the
    # memory_type, the causal edges, the raw metadata bag — rides in a reserved
    # key inside the row's `metadata`, so the round trip is faithful; the derived
    # semantic edges (re-derived on the next fold) and the write-only `index_text`
    # are the only things not carried.

    async def _get_record(self, ns: str, unit_id: str):
        try:
            recs = await asyncio.to_thread(
                partial(self._client.get, ns, [uuid.UUID(unit_id).bytes], include_vector=True, include_edges=True)
            )
        except Exception as e:
            # A namespace with no writes yet has no manifest; a get against it is
            # "nothing there", not an error — invalidating a memory that isn't in
            # the bank returns False rather than raising.
            if _is_missing_namespace(e):
                return None
            raise
        return recs[0] if recs else None

    async def get_archived_memory(self, *, conn, fq_table, bank_id: str, unit_id: str) -> StoredMemory | None:
        row = await conn.fetchrow(
            f"SELECT {_ARCHIVE_SELECT} FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2",
            unit_id,
            bank_id,
        )
        return _archived_row_to_stored(row) if row else None

    async def invalidate_memory(self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None) -> bool:
        record = await self._get_record(self._namespace(bank_id), unit_id)
        if record is None:
            return False
        stored = _stored_from_record(record)
        # The user metadata carries the full memlake memory under a reserved key,
        # so restore can rebuild it exactly; the archive columns hold what the
        # list/detail views render.
        metadata = dict(stored.metadata or {})
        metadata[_ARCHIVE_PAYLOAD_KEY] = _serialize_record(record)
        # event_date is NOT NULL on the archive (LIKE memory_units); fall back
        # through the other times, then to the write time, so the insert holds.
        event_date = stored.event_date or stored.occurred_start or stored.mentioned_at or stored.created_at
        await conn.execute(
            f"""
            INSERT INTO {fq_table("invalidated_memory_units")}
                (id, bank_id, text, fact_type, context, event_date, occurred_start, occurred_end,
                 mentioned_at, document_id, chunk_id, tags, metadata, proof_count, created_at,
                 consolidated_at, entity_ids, invalidation_reason, invalidated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb, $14, $15, $16,
                    $17::uuid[], $18, now())
            ON CONFLICT (id) DO UPDATE SET
                metadata = EXCLUDED.metadata,
                invalidation_reason = EXCLUDED.invalidation_reason,
                invalidated_at = now()
            """,
            unit_id,
            bank_id,
            stored.text,
            stored.fact_type,
            stored.context,
            event_date or datetime.now(timezone.utc),
            stored.occurred_start,
            stored.occurred_end,
            stored.mentioned_at,
            stored.document_id,
            stored.chunk_id,
            list(stored.tags or []),
            json.dumps(metadata),
            stored.proof_count,
            stored.created_at or datetime.now(timezone.utc),
            stored.consolidated_at,
            [uuid.UUID(e) for e in stored.entity_ids],
            reason,
        )
        # Only now, once the archive row is written, drop it from the index.
        await asyncio.to_thread(self._client.delete, self._namespace(bank_id), [record.id])
        return True

    async def set_invalidation_reason(self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None) -> None:
        await conn.execute(
            f"UPDATE {fq_table('invalidated_memory_units')} SET invalidation_reason = $3 WHERE id = $1 AND bank_id = $2",
            unit_id,
            bank_id,
            reason,
        )

    async def restore_memory(self, *, conn, fq_table, bank_id: str, unit_id: str) -> StoredMemory | None:
        row = await conn.fetchrow(
            f"SELECT {_ARCHIVE_SELECT} FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2",
            unit_id,
            bank_id,
        )
        if row is None:
            return None
        blob = _archive_payload(row)
        # Reconstruct the memory from the stashed payload and write it back to the
        # bank's namespace — the next fold re-indexes it and re-derives its
        # semantic edges. The caller re-embeds afterwards (set_memory_embedding).
        if blob is not None:
            await self.ensure_namespace(bank_id)
            memory = _memory_from_blob(blob, unit_id)
            await asyncio.to_thread(self._client.write, self._namespace(bank_id), [memory])
        await conn.execute(
            f"DELETE FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2", unit_id, bank_id
        )
        return _archived_row_to_stored(row)

    async def set_memory_embedding(self, *, conn, fq_table, bank_id: str, unit_id: str, embedding) -> None:
        op = self._client.patch(uuid.UUID(unit_id).bytes, vector=_parse_embedding(embedding))
        await asyncio.to_thread(self._client.write_ops, self._namespace(bank_id), [op])

    async def apply_edit(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        unit_id: str,
        text: str,
        context: str | None,
        fact_type: str,
        occurred_start,
        occurred_end,
        event_date,
        mentioned_at,
        entity_ids: list[str] | None,
    ) -> None:
        """Apply a field edit by rewriting the memory.

        `patch` cannot change the entity ids or the memory_type, and a fact-type
        edit is exactly a memory_type change, so an edit is an upsert of the whole
        memory rather than a partial update. The existing record supplies the
        fields the edit leaves alone (tags, proof_count, vector — the caller
        re-embeds and overwrites the vector next); the edit overrides the rest.
        """
        record = await self._get_record(self._namespace(bank_id), unit_id)
        if record is None:
            return
        payload = record.memory

        memory_type = FACT_TYPE_TO_MEMORY_TYPE.get(fact_type, record.memory_type)
        eids = (
            [uuid.UUID(e).bytes for e in entity_ids]
            if entity_ids is not None
            else list(payload.entity_ids)
        )

        # Rebuild the metadata bag: the edited context, and a reset consolidation
        # state — an edit re-consolidates, so the flag goes back to "not yet".
        meta = dict(payload.metadata or {})
        if context:
            meta[META_CONTEXT] = context
        else:
            meta.pop(META_CONTEXT, None)
        meta[META_CONSOLIDATED_FLAG] = CONSOLIDATED_NO
        meta[META_CONSOLIDATED_AT] = ""
        meta[META_CONSOLIDATION_FAILED_AT] = ""

        raw = record.vector.f32le if record.vector else b""
        vector = list(struct.unpack(f"<{len(raw) // 4}f", raw)) if raw else []

        memory = mc.memory(
            text,
            vector,
            memory_type=memory_type,
            id=record.id,
            tags=list(payload.tags),
            proof_count=payload.proof_count,
            entity_ids=eids,
            metadata=meta,
            # occurred window / event date follow the edit; the write time is left
            # to the server to stamp now, since the edit is a write.
            event_date=_to_epoch_ms(event_date),
            occurred_start=_to_epoch_ms(occurred_start),
            occurred_end=_to_epoch_ms(occurred_end),
            mentioned_at=_to_epoch_ms(mentioned_at),
        )
        for edge in payload.causal_out:
            memory.causal_out.add(target=edge.target, link_type=edge.link_type, weight=edge.weight)
        await asyncio.to_thread(self._client.write, self._namespace(bank_id), [memory])

    async def list_entities(
        self, *, conn, fq_table, bank_id: str, search: str | None = None, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        return await reads.list_entities(
            self, conn=conn, fq_table=fq_table, bank_id=bank_id, search=search, limit=limit, offset=offset
        )

    async def graph_units(
        self,
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
    ) -> dict[str, Any]:
        return await reads.graph_units(
            self,
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_type=fact_type,
            search_query=search_query,
            document_id=document_id,
            chunk_id=chunk_id,
            tags=tags,
            tags_match=tags_match,
            limit=limit,
        )

    async def graph_entity_rows(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict[str, Any]]:
        return await reads.graph_entity_rows(self, conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    async def graph_direct_links(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict[str, Any]]:
        return await reads.graph_direct_links(self, conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    async def find_unconsolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str],
        limit: int,
        scope_tags: list[str] | None = None,
    ) -> list[StoredMemory]:
        found: list[StoredMemory] = []
        page_token = ""
        pages = 0
        # Bounded so one consolidation cycle cannot walk an entire large corpus.
        while len(found) < limit and pages < 100:
            page = await self.scan_memories(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=fact_types,
                limit=max(limit, 200),
                page_token=page_token,
                # Pushed down: the server drops consolidated memories rather than
                # sending pages of them for us to discard. This is also what keeps
                # a memory consolidation *failed* on out of the queue — the flag
                # flips either way, see `mark_consolidated`.
                metadata_equals={META_CONSOLIDATED_FLAG: CONSOLIDATED_NO},
                # `all` is the containment the SQL `tags @> scope` expresses.
                tags=scope_tags or None,
                tags_match="all",
            )
            found.extend(page.memories)
            pages += 1
            page_token = page.next_page_token
            if not page_token:
                break
        # The SQL path orders by created_at ASC so the oldest backlog drains first.
        found.sort(key=lambda m: (m.created_at is None, m.created_at))
        return found[:limit]

    async def mark_consolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        unit_ids: list[str],
        when: datetime | None,
        failed: bool = False,
    ) -> None:
        """Stamp, or clear, the consolidation markers on a batch of sources.

        All three outcomes take the flag out of the "not-yet-consolidated" value
        the candidate query selects on, because a memory the LLM could not
        summarise must stop being a candidate just as a summarised one does. They
        differ in the value written — success "1", failure a distinct
        :data:`CONSOLIDATED_FAILED` — and in the timestamp: success sets
        `consolidated_at`, failure sets `consolidation_failed_at`. The distinct
        flag is what lets the curation facet push "done" vs "failed" down as an
        equality; clearing (``when=None``, not failed) returns it to "0".
        """
        if not unit_ids:
            return
        # An empty string clears a marker: the metadata map merges, so there is no
        # way to delete a key — the readers treat "" as unset.
        stamp = when.isoformat() if when is not None else ""
        if failed:
            metadata = {
                META_CONSOLIDATED_AT: "",
                META_CONSOLIDATION_FAILED_AT: stamp,
                META_CONSOLIDATED_FLAG: CONSOLIDATED_FAILED,
            }
        else:
            metadata = {
                META_CONSOLIDATED_AT: stamp,
                META_CONSOLIDATION_FAILED_AT: "",
                META_CONSOLIDATED_FLAG: CONSOLIDATED_YES if when is not None else CONSOLIDATED_NO,
            }
        await self.update_memories(bank_id, [MemoryPatch(unit_id=u, metadata=metadata) for u in unit_ids])

    async def entity_memory_counts(
        self, *, conn, fq_table, bank_id: str, entity_ids: list[str] | None = None
    ) -> dict[str, int]:
        ids = [uuid.UUID(e).bytes for e in entity_ids] if entity_ids else None
        raw = await asyncio.to_thread(partial(self._client.entity_stats, self._namespace(bank_id), entity_ids=ids))
        # Reads the persisted entity posting, so cost scales with the number of
        # entities rather than the corpus. An id asked about and not returned is
        # an orphan, which is exactly what the sweep needs.
        return {str(uuid.UUID(bytes=k)): v for k, v in raw.items()}

    async def entities_for_units(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> dict[str, list[str]]:
        """The `unit_entities` join, read off the memories themselves.

        Ids only, so this needs no Postgres: the registry is only involved when a
        caller wants *names* (see :func:`reads.entity_map_for_units`).
        """
        memories = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
        return {m.unit_id: list(m.entity_ids) for m in memories if m.entity_ids}

    async def entity_map_for_units(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, list[dict[str, str]]]:
        # The named form recall renders: entity ids come off the memories, the
        # `entities` registry (still in postgres) supplies the labels.
        return await reads.entity_map_for_units(self, conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    async def any_memory_updated_since(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        since: datetime,
        fact_types: list[str] | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
    ) -> bool:
        return await reads.any_memory_updated_since(
            self,
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            since=since,
            fact_types=fact_types,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )

    # -- count surfaces ------------------------------------------------------
    #
    # No GROUP BY, so each aggregate is a scan tallied in Python — O(matching),
    # bounded by the scan-page cap. §5b (declared indexed metadata keys) is what
    # turns these into metadata-only reads.

    async def consolidation_freshness(self, *, conn, fq_table, bank_id: str) -> dict[str, Any]:
        return await reads.consolidation_freshness(self, conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def document_memory_counts(self, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, int]:
        return await reads.document_memory_counts(
            self, conn=conn, fq_table=fq_table, bank_id=bank_id, document_ids=document_ids
        )

    async def memories_timeseries(
        self, *, conn, fq_table, bank_id: str, time_field: str, trunc: str, since: datetime
    ) -> list[dict[str, Any]]:
        return await reads.memories_timeseries(
            self, conn=conn, fq_table=fq_table, bank_id=bank_id, time_field=time_field, trunc=trunc, since=since
        )

    async def observation_scope_counts(self, *, conn, fq_table, bank_id: str) -> list[dict[str, Any]]:
        return await reads.observation_scope_counts(self, conn=conn, fq_table=fq_table, bank_id=bank_id)

    # -- observations --------------------------------------------------------

    async def upsert_observation(self, *, conn, bank_id: str, record: FactRecord) -> None:
        """Write an observation, creating or replacing it.

        The write upserts by id, so create and update are the same call — which is
        the point: a reinforced observation keeps its id and never leaves the
        index, so nothing holding a reference to it goes stale and recall never
        sees it briefly missing.

        The caller hands over a record with no entities, because under Postgres an
        observation borrows its sources' on every read. There is no read-time join
        here, so the borrow is resolved once, now — see `observations`.
        """
        record = await observations.resolve_source_entities(self, conn=conn, bank_id=bank_id, record=record)
        await self._write_records(bank_id, [record])

    async def observations_for_sources(
        self, *, conn, ops, fq_table, bank_id: str, unit_ids: list[str]
    ) -> list[StoredMemory]:
        return await observations.observations_for_sources(
            self, conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def delete_stale_observations(self, *, conn, ops, fq_table, bank_id: str, fact_ids: list) -> int:
        return await observations.delete_stale_observations(
            self, conn=conn, fq_table=fq_table, bank_id=bank_id, fact_ids=fact_ids
        )

    # -- maintenance ---------------------------------------------------------
    #
    # `record_unit_entities`, `enqueue_relink_victims`, `relink_pass` and
    # `prune_stale_cooccurrences` keep their base defaults on purpose: the
    # postings and links they maintain travel *inside* the memory here, so there
    # is no join table to write, no dangling pointer to queue and nothing to
    # relink. The entity registry is the one exception — it really is in
    # Postgres, so its orphan sweep really has work to do.

    async def prune_orphan_entities(self, *, conn, fq_table, bank_id: str) -> int:
        return await reads.prune_orphan_entities(self, conn=conn, fq_table=fq_table, bank_id=bank_id)


def _row_from_hit(hit: Any) -> dict | None:
    """Rebuild a `memory_units`-shaped row from a hit's inline payload.

    This is what removes the hydration query: the server already had the memory
    materialized to score it, so it rides back with the hit.
    """
    payload = hit.memory
    if payload is None:
        return None
    meta = dict(payload.metadata or {})
    ts = payload.timestamps
    return {
        "id": str(uuid.UUID(bytes=hit.id)),
        "text": payload.text,
        "context": meta.get(META_CONTEXT),
        "event_date": _from_epoch_ms(ts.event_date),
        "occurred_start": _from_epoch_ms(ts.occurred_start),
        "occurred_end": _from_epoch_ms(ts.occurred_end),
        "mentioned_at": _from_epoch_ms(ts.mentioned_at),
        "fact_type": MEMORY_TYPE_TO_FACT_TYPE.get(hit.memory_type, "world"),
        "document_id": meta.get(META_DOCUMENT_ID),
        "chunk_id": meta.get(META_CHUNK_ID),
        "tags": list(payload.tags),
        "metadata": _parse_json_object(meta.get(META_METADATA_JSON)),
        "proof_count": payload.proof_count,
        # Prefer the first-class timestamp; fall back to the metadata bag for
        # memories written before it existed.
        "updated_at": _from_epoch_ms(ts.updated_at) or meta.get(META_UPDATED_AT),
    }


def _stored_from_record(record: Any) -> StoredMemory:
    """Map a StoredMemoryRecord (Get/Scan) onto the shared addressed-read shape."""
    payload = record.memory
    meta = dict(payload.metadata or {})
    ts = payload.timestamps
    return StoredMemory(
        unit_id=str(uuid.UUID(bytes=record.id)),
        text=payload.text,
        fact_type=MEMORY_TYPE_TO_FACT_TYPE.get(record.memory_type, "world"),
        context=meta.get(META_CONTEXT),
        document_id=meta.get(META_DOCUMENT_ID),
        chunk_id=meta.get(META_CHUNK_ID),
        tags=list(payload.tags),
        metadata=_parse_json_object(meta.get(META_METADATA_JSON)),
        proof_count=payload.proof_count,
        event_date=_from_epoch_ms(ts.event_date),
        occurred_start=_from_epoch_ms(ts.occurred_start),
        occurred_end=_from_epoch_ms(ts.occurred_end),
        mentioned_at=_from_epoch_ms(ts.mentioned_at),
        created_at=_parse_iso(meta.get(META_CREATED_AT)),
        entity_ids=[str(uuid.UUID(bytes=e)) for e in payload.entity_ids],
        source_memory_ids=_parse_json_list(meta.get(META_SOURCE_MEMORY_IDS)),
        consolidated_at=_parse_iso(meta.get(META_CONSOLIDATED_AT)),
        # Empty unless the read opted into edges; the ranking path never does.
        semantic_edges=[
            (str(uuid.UUID(bytes=e.target)), float(e.weight)) for e in getattr(payload, "semantic_out", [])
        ],
    )


# --------------------------------------------------------------------- archive round-trip


_ARCHIVE_TS_FIELDS = ("event_date", "updated_at", "occurred_start", "occurred_end", "mentioned_at")


def _serialize_record(record: Any) -> dict:
    """The full memlake memory as a JSON-able dict, for the archive's reserved key.

    Carries what the archive's memory_units-shaped columns cannot: the vector, the
    memory_type, the causal edges and the raw metadata bag. Bytes become hex so it
    round-trips through JSONB.
    """
    payload = record.memory
    raw = record.vector.f32le if record.vector else b""
    ts = payload.timestamps
    return {
        "text": payload.text,
        "memory_type": record.memory_type,
        "vector": list(struct.unpack(f"<{len(raw) // 4}f", raw)) if raw else [],
        "tags": list(payload.tags),
        "proof_count": payload.proof_count,
        "entity_ids": [e.hex() for e in payload.entity_ids],
        "metadata": dict(payload.metadata or {}),
        "timestamps": {f: getattr(ts, f) for f in _ARCHIVE_TS_FIELDS if ts.HasField(f)},
        "causal_out": [
            {"target": e.target.hex(), "link_type": int(e.link_type), "weight": float(e.weight)}
            for e in payload.causal_out
        ],
    }


def _memory_from_blob(blob: dict, unit_id: str):
    """Rebuild a memlake ``Memory`` from a :func:`_serialize_record` blob."""
    ts = blob.get("timestamps", {})
    memory = mc.memory(
        blob["text"],
        blob.get("vector") or [],
        memory_type=blob["memory_type"],
        id=uuid.UUID(unit_id).bytes,
        tags=blob.get("tags", []),
        proof_count=blob.get("proof_count", 1),
        entity_ids=[bytes.fromhex(e) for e in blob.get("entity_ids", [])],
        metadata=blob.get("metadata", {}),
        event_date=ts.get("event_date"),
        updated_at=ts.get("updated_at"),
        occurred_start=ts.get("occurred_start"),
        occurred_end=ts.get("occurred_end"),
        mentioned_at=ts.get("mentioned_at"),
    )
    for edge in blob.get("causal_out", []):
        memory.causal_out.add(target=bytes.fromhex(edge["target"]), link_type=edge["link_type"], weight=edge["weight"])
    return memory


def _archived_metadata(row: Any) -> dict:
    """The archived row's `metadata` as a dict (asyncpg hands back str or dict)."""
    value = row["metadata"]
    if isinstance(value, str):
        value = _parse_json_object(value) or {}
    return dict(value or {})


def _archive_payload(row: Any) -> dict | None:
    """The stashed memlake memory from an archived row, or None if absent."""
    return _archived_metadata(row).get(_ARCHIVE_PAYLOAD_KEY)


def _archived_row_to_stored(row: Any) -> StoredMemory:
    """Map an `invalidated_memory_units` row onto :class:`StoredMemory`.

    The reserved memlake payload is stripped, so the user metadata surfaced is
    exactly what the memory carried.
    """
    meta = _archived_metadata(row)
    meta.pop(_ARCHIVE_PAYLOAD_KEY, None)
    return StoredMemory(
        unit_id=str(row["id"]),
        text=row["text"],
        fact_type=row["fact_type"],
        context=row["context"],
        document_id=row["document_id"],
        chunk_id=str(row["chunk_id"]) if row["chunk_id"] else None,
        tags=list(row["tags"] or []),
        metadata=meta or None,
        proof_count=row["proof_count"] or 1,
        event_date=row["event_date"],
        occurred_start=row["occurred_start"],
        occurred_end=row["occurred_end"],
        mentioned_at=row["mentioned_at"],
        created_at=row["created_at"],
        consolidated_at=row["consolidated_at"],
        entity_ids=[str(e) for e in (row["entity_ids"] or [])],
    )


def _parse_json_object(value: str | None) -> dict | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def _parse_iso(value: str | None) -> datetime | None:
    # An empty string is how a cleared marker is written: the metadata map merges,
    # so a key cannot be removed, only blanked.
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _to_result(row: dict, **scores) -> Any:
    from hindsight_api.engine.search.types import RetrievalResult

    return RetrievalResult.from_db_row({**row, **scores})
