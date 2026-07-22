# hindsight-memlake

memlake as the **memories store** for [Hindsight](https://github.com/hindsight-ai/hindsight).

Hindsight keeps agent memory in Postgres: `memory_units` rows, `memory_links` and
`unit_entities` for the graph, and every recall arm — semantic, BM25, graph,
temporal — as SQL over them. That slice sits behind a single extension point,
`MemoriesExtension`. This package implements it with memlake, an S3-native
retrieval engine that serves all four arms over one storage layer.

With it installed, **nothing memory-shaped reaches Postgres**:

| Postgres | memlake |
| --- | --- |
| `memory_units` row per fact | a memory in the bank's namespace, id minted before the write |
| `unit_entities` join rows | an entity posting carried on the memory |
| `memory_links` causal rows | causal edges carried on the memory |
| `memory_links` semantic rows | kNN edges derived by the indexer at fold time |
| hydration `SELECT` after ranking | the memory rides back inline with the hit |

A bank maps to a namespace, a `fact_type` to a `memory_type`, and the columns
memlake has no first-class model of (context, document_id, chunk_id, the user
metadata JSON) travel in its opaque metadata bag — stored verbatim, returned on
every read, never indexed.

Documents, chunks, banks, operations and the `entities` registry stay in
Postgres either way. That last one is why a few methods here still take a
connection: entity *ids* ride on the memory, but their canonical *names* are
still a join against a table that is still populated.

## Install

```bash
uv pip install hindsight-memlake
```

`memlake-client` is resolved from this repo (`../../clients/python`), because the
client and the server it talks to move together.

## Enable

Point Hindsight's memories extension point at the class and tell it where the
memlake server is:

```bash
export HINDSIGHT_API_MEMORIES_EXTENSION=hindsight_memlake:MemlakeMemories
export HINDSIGHT_API_MEMORIES_TARGET=localhost:50051
```

Nothing else changes: the engine reaches every memory operation through the same
interface, so no call site knows which store is installed.

## Configuration

Every `HINDSIGHT_API_MEMORIES_*` variable becomes a config key, lowercased with
the prefix stripped.

| Variable | Default | What it does |
| --- | --- | --- |
| `HINDSIGHT_API_MEMORIES_EXTENSION` | *(unset — Postgres)* | `hindsight_memlake:MemlakeMemories` to install this store |
| `HINDSIGHT_API_MEMORIES_TARGET` | `localhost:50051` | memlake server address. Comma-separate several to run against a cluster: the client rendezvous-hashes each namespace to a preferred node for cache and commit affinity, and fails over on its own |
| `HINDSIGHT_API_MEMORIES_NAMESPACE_PREFIX` | `""` | Prepended to every bank id when forming the namespace, so several deployments can share one bucket |
| `HINDSIGHT_API_MEMORIES_NPROBE` | `0` (server default) | How many clusters the dense arm probes. Coverage, not depth — candidates in unprobed clusters are unreachable no matter how large a `top_k` an arm asks for |

## What is in here

| File | Role |
| --- | --- |
| `provider.py` | `MemlakeMemories` — the extension itself: writes, the recall arms, addressed reads, maintenance |
| `observations.py` | Observations denormalised and upserted, and the stale-observation sweep |
| `reads.py` | The curation/export read surfaces rebuilt on Get / Scan / Stats |
| `graph.py` | `MemlakeGraphRetriever` — the graph arm, expanded by memlake rather than by walking `memory_links` |

## Known trade-offs

* **The metadata bag is not indexed.** The only predicate over it is equality, so
  filters that need more (text search over stored memories, tag *groups*, which
  are a boolean tree) run in Python after the query and can return short pages.
* **Offset paging costs pages.** Hindsight's list APIs take offset/limit; a Scan
  takes an opaque cursor. Deep offsets are walked and capped.
* **Tag facets are a corpus walk.** memlake filters on tags but does not
  aggregate over them, so the tag histogram counts in Python.
* **Consolidation failure is a metadata key.** Postgres has a
  `consolidation_failed_at` column; here it is written to the bag but no
  interface field reads it back — what survives is that the memory left the
  consolidation queue.
* **Observations do not inherit entity edits.** Postgres re-derives an
  observation's entities from its sources on every read; here they are resolved
  once at write time, so an edit to a source fact catches up the next time
  consolidation touches the observation.
