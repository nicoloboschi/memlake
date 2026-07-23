# Running Hindsight against memlake, locally

This directory holds `hindsight_memlake`, the extension that plugs memlake in as
Hindsight's memories store, and a `docker-compose.yml` that stands up everything
the store needs. This guide runs the whole thing end to end on one machine.

## Topology

```
                ┌──────────────── docker compose ────────────────┐
                │                                                 │
  host          │   MinIO ─── memlake `serve` (gRPC :50051)       │
  ┌───────────┐ │     │            ▲              │               │
  │ api + pg0 │─┼─────┼────────────┘              ▼               │
  └───────────┘ │     └──── object store ──── memlake `indexer`   │
                │                               (compaction loop) │
                └─────────────────────────────────────────────────┘
```

`serve` is the `HINDSIGHT_API_MEMORIES_TARGET` the extension dials. The `indexer`
folds each namespace's WAL tail into segments on a timer — writes are queryable
from the tail the moment they land, so this only keeps retrieval fast as volume
grows; it is never on the critical path.

Postgres still holds the metadata memlake has no model of (documents, chunks,
banks, operations, the entity-name registry) — but it does not need a container.
Hindsight's default `pg0` database url starts an **embedded** Postgres inside the
API process, so the whole non-memory side lives with the host process.

Hindsight's API process runs on the **host**, not in the compose. The pluggable
`MEMORIES` seam it depends on is not in a released Hindsight image yet, so it has
to run from a checkout of that branch with this (closed-source) extension
installed. When the seam ships in a release, a `hindsight` service layered on the
official image can move into the compose.

## Prerequisites

- Docker (with BuildKit — building the memlake image needs `docker >= 23`).
- A checkout of Hindsight on the `MEMORIES`-seam branch — the same tree the tests
  point `HINDSIGHT_API_SLIM_PATH` at. Export that path:
  ```sh
  export HINDSIGHT_API_SLIM_PATH=/path/to/hindsight-api-slim
  ```
- `uv`.
- An LLM + embedding provider key (Hindsight's retain/consolidate pipeline calls
  out to one). The examples below use OpenAI; see Hindsight's own docs for
  configuring a different provider or a local model.

## 1. Start the backend

From this directory:

```sh
docker compose up -d --build
```

First run compiles the memlake image (a few minutes); later runs reuse it. This
brings up MinIO, `serve` (:50051), and `indexer`. Check it:

```sh
docker compose ps
nc -z localhost 50051 && echo "memlake up"
```

## 2. Start Hindsight on the host

`pg0` (the default `HINDSIGHT_API_DATABASE_URL`) gives an embedded Postgres — no
external database to run. Install everything into one venv, then run its console
script directly.

> Install into a real venv and launch `.venv/bin/hindsight-api` — **do not** use
> `uv run …`. `uv run` re-syncs the project venv from this package's own
> `pyproject`, which reinstalls the *released* `hindsight-api-slim` (no MEMORIES
> seam) over the editable checkout and re-resolves pinned deps. Install once,
> launch directly.

```sh
# one-time: create the venv and install the branch checkout + extension + client.
# [embedded-db] = pg0, [local-onnx] = in-process embeddings, so no external
# database or embedding service is needed.
uv venv
uv pip install --python .venv/bin/python \
  -e "$HINDSIGHT_API_SLIM_PATH[embedded-db,local-onnx]" \
  -e . \
  -e ../../clients/python

# --- select the memlake memories store ---
export HINDSIGHT_API_MEMORIES_EXTENSION=hindsight_memlake.provider:MemlakeMemories
export HINDSIGHT_API_MEMORIES_TARGET=localhost:50051
# optional: HINDSIGHT_API_MEMORIES_NAMESPACE_PREFIX, HINDSIGHT_API_MEMORIES_NPROBE

# --- embeddings + reranker that need no key/model server ---
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=onnx   # in-process, from [local-onnx]
export HINDSIGHT_API_RERANKER_PROVIDER=rrf      # rank fusion, no model

# --- LLM. Retain/consolidate wants one. Either give a key ...
export HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY
#     ... or run keyless: `none` boots and serves, and stores each item's raw
#     text verbatim (no fact extraction) so you can exercise ingest/list/recall:
# export HINDSIGHT_API_LLM_PROVIDER=none

.venv/bin/hindsight-api --host 0.0.0.0 --port 8888
```

Hindsight runs its Alembic migrations against the embedded Postgres on startup
(`HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP` defaults to true), so there is no
separate migrate step. On boot the log shows the store it picked and that it
connected before serving:

```
[memories] store=memlake (memory rows do not go to postgres)
[memories] memlake connected to localhost:50051 (postgres holds no memories)
```

Then open the interactive API (Swagger UI) at **http://localhost:8888/docs**, or:

```sh
curl -s localhost:8888/health
# {"status":"healthy","database":"connected"}

# smoke test end to end (keyless `none` LLM shown):
curl -s -X PUT  localhost:8888/v1/default/banks/demo -d '{}'
curl -s -X POST "localhost:8888/v1/default/banks/demo/memories?async=false" \
  -H 'content-type: application/json' \
  -d '{"items":[{"content":"Ada Lovelace wrote the first algorithm in 1843."}]}'
curl -s -X POST "localhost:8888/v1/default/banks/demo/memories/recall" \
  -H 'content-type: application/json' -d '{"query":"who wrote the first algorithm"}'
```

From here, use the API exactly as you would any Hindsight — create a bank, retain
some memories, recall them. Every memory read and write goes through memlake; the
`indexer` container compacts in the background; Postgres carries only the
surrounding metadata.

## Teardown

```sh
docker compose down          # keep the data
docker compose down -v       # also wipe MinIO + Postgres volumes
```

## Knobs

All optional, with the defaults this compose uses:

| Variable | Default | What it does |
| --- | --- | --- |
| `INDEXER_INTERVAL` | `5` | seconds between compaction passes |
| `SERVE_MEM_MB` / `SERVE_DISK_MB` | `1024` / `8192` | `serve`'s read-cache budgets |
| `RUST_LOG` | `info` | memlake log level |

## Notes / current limitations

- **The seam is unreleased.** Hindsight must run from the `MEMORIES`-branch
  checkout (`HINDSIGHT_API_SLIM_PATH`), not a published image — that is why step 2
  is host-run rather than another compose service.
- **Postgres is required**, by design — memlake stores the memories, Postgres
  stores the documents/chunks/banks/entities around them. Here it is the embedded
  `pg0`, so there is nothing to run; point `HINDSIGHT_API_DATABASE_URL` at a real
  server instead if you want the data to outlive the process.
- **Metadata filters are equality-only.** Any richer Hindsight filter (ranges,
  nested tag AND/OR/NOT groups) is applied in Python after the query and can
  therefore return short pages.
