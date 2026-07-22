"""memlake as Hindsight's memories store, packaged as an extension.

Point Hindsight at it and every memory unit — plus the links and entity postings
that would have become `memory_links` / `unit_entities` rows — lives in memlake
instead of Postgres::

    HINDSIGHT_API_MEMORIES_EXTENSION=hindsight_memlake:MemlakeMemories
    HINDSIGHT_API_MEMORIES_TARGET=localhost:50051

Documents, chunks, banks, operations and the `entities` registry are untouched:
they stay in Postgres whichever store is installed.
"""

from .provider import MemlakeMemories

__all__ = ["MemlakeMemories"]
