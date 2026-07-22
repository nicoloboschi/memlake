"""The graph arm, expanded by memlake instead of by walking `memory_links`.

Hindsight's :class:`LinkExpansionRetriever` walks `memory_links` and
`unit_entities` in Postgres. Those tables are empty by design here — an entity
posting rides on each memory and semantic links are derived by the indexer — so
the graph arm has to come from the store, which is what
:meth:`MemlakeMemories.graph_retriever` hands the recall pipeline. Because
memlake's entity posting is persisted and global, this arm expands through the
whole corpus rather than just the probed neighbourhood.

It runs its own Query rather than reusing the one `search` issues: the arms are
answered off a cached snapshot, so a second call on a warm namespace costs no
extra object-storage roundtrips, and keeping it separate leaves the
per-fact_type parallelism in the recall pipeline exactly as it is.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from hindsight_api.engine.search.graph_retrieval import GraphRetriever
from hindsight_api.engine.search.tags import TagGroup, TagsMatch, filter_results_by_tag_groups
from hindsight_api.engine.search.types import GraphRetrievalTimings, RetrievalResult

logger = logging.getLogger(__name__)


class MemlakeGraphRetriever(GraphRetriever):
    """Delegates the graph arm to the memlake store that created it."""

    def __init__(self, store):
        self._store = store

    @property
    def name(self) -> str:
        return "memlake_graph"

    async def retrieve(
        self,
        pool,
        query_embedding_str: str,
        bank_id: str,
        fact_type: str,
        budget: int,
        query_text: str | None = None,
        adjacency=None,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> tuple[list[RetrievalResult], GraphRetrievalTimings | None]:
        started = time.time()
        # `pool` and `adjacency` are the Postgres handles: the pool would open a
        # transaction on tables this store never writes, and the adjacency is a
        # pre-loaded `memory_links` graph that is empty for the same reason.
        results = await self._store.graph_search(
            bank_id=bank_id,
            fact_type=fact_type,
            query_embedding=query_embedding_str,
            query_text=query_text,
            limit=budget,
            tags=tags,
            tags_match=tags_match,
            created_after=created_after,
            created_before=created_before,
        )

        # Tag groups are a boolean tree memlake's flat tag modes cannot express, so
        # this one filter runs here — and can therefore trim below `budget`. Same
        # caveat as the dense arms.
        if tag_groups:
            results = filter_results_by_tag_groups(results, tag_groups)

        timings = GraphRetrievalTimings(
            fact_type=fact_type,
            traverse=time.time() - started,
            result_count=len(results),
        )
        return results, timings
