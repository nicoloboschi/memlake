# memlake admin

An inspection console for a memlake namespace: index stats, a browsable scan, and
a query workbench that shows every arm's raw contribution to every hit.

It is read-only apart from `CreateNamespace`. There is deliberately no write or
ingest UI — writes belong to your service, not to an operator console.

---

## How it talks to memlake

memlake is a **gRPC** service and gRPC cannot be spoken from a browser. So every
memlake call happens **server-side**, inside a Next.js route handler under
`app/api/**`, and the React components talk to those JSON endpoints instead.

The contract is loaded **dynamically at runtime** from
`proto/memlake/v1/memlake.proto` via `@grpc/proto-loader` — there are no
generated stubs to regenerate, so the UI cannot silently drift from the `.proto`.
Two loader details matter and are handled in `lib/convert.ts`:

- `bytes` fields arrive as Node `Buffer`s. Ids are 16 raw bytes and are converted
  to canonical UUIDs before they cross into JSON.
- 64-bit ints arrive as decimal **strings** (`longs: String`). They stay strings
  all the way to the browser, and any arithmetic on them uses `BigInt` — a WAL
  sequence does not fit a JS number.

`Vector.f32le` is raw little-endian float32, not a JSON array; `lib/vector.ts` is
the only place that encoding is spelled out.

---

## Prerequisites

Everything below runs from the **repo root**, not from `admin/`.

```bash
# 1. Object storage. MinIO, because memlake depends on real S3 conditional-write
#    semantics (If-None-Match / If-Match). Creates the `memlake` bucket too.
docker compose up -d

# 2. The gRPC API. Stateless — all coordination lives in object storage.
cargo run --release -p mlake-server -- serve --addr 0.0.0.0:50051

# 3. The indexer. Separate process; without it, writes stay in the WAL tail and
#    a namespace shows has_index=false with a growing un-indexed backlog.
cargo run --release -p mlake-server -- index --namespaces my-bank --interval-secs 5
#    ...or a single metered pass:
cargo run --release -p mlake-server -- index --namespaces my-bank --once
```

The server reads its S3 config from the environment (`MEMLAKE_S3_ENDPOINT`,
`MEMLAKE_S3_BUCKET`, `MEMLAKE_S3_ACCESS_KEY`, `MEMLAKE_S3_SECRET_KEY`,
`MEMLAKE_S3_REGION`); the defaults already point at the docker-compose MinIO.

## Running the UI

```bash
cd admin
npm install
cp .env.local.example .env.local     # optional; the defaults work locally
npm run dev                          # http://localhost:3000
```

Production:

```bash
npm run build && npm start
```

> The `dev` / `build` / `start` scripts pin `NODE_ENV` explicitly. A `NODE_ENV`
> inherited from your shell (some setups export `development` globally) makes
> `next build` resolve two different React builds and the prerender step dies
> with `Cannot read properties of null`.

### Environment

All server-side only; none of it reaches the browser. See
`.env.local.example`.

| Variable              | Default                              | Meaning                                                                 |
| --------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `MEMLAKE_ADDR`        | `localhost:50051`                    | `mlake-server serve` address. Insecure credentials — it is an internal, unauthenticated service. |
| `MEMLAKE_PROTO_PATH`  | `../proto/memlake/v1/memlake.proto`  | The contract. Relative paths resolve against `admin/`.                   |
| `MEMLAKE_EMBEDDINGS`  | *(unset — on)*                       | `off` disables server-side embedding entirely: no model, no download.    |

---

## Pages

| Route                   | RPC(s)                        | What it is for                                                                     |
| ----------------------- | ----------------------------- | ---------------------------------------------------------------------------------- |
| `/`                     | `ListNamespaces`, `CreateNamespace` | Every namespace in the bucket. One LIST — an operator call, not a per-request one. |
| `/ns/[ns]`              | `Stats`                       | Generation, WAL position, the **un-indexed backlog** (`wal_head − wal_index_cursor`), and a per-`memory_type` table. STRONG/EVENTUAL toggle. |
| `/ns/[ns]/browse`       | `Scan`, `Get`                 | Cursor-paged walk of stored memories, with a detail panel. |
| `/ns/[ns]/query`        | `Query`                       | The workbench: all arms, raw scores, client-side fusion. |

### Things the UI is careful about

**`memory_type` is an independent index.** Each type has its own IVF centroids,
its own BM25 index, its own doc count. The server never fuses across types, so
neither does the UI: query results are grouped by type and each group is ranked
on its own.

**The scan cursor is opaque and generation-scoped.** `page_token` is a position,
not a snapshot — writes landing mid-walk can shift later pages, and a cursor is
only valid against the generation that produced it. So Browse keeps a *stack* of
the tokens it used and pops it to go back. It never computes page numbers,
because there is no such thing.

**An absent arm is not a zero score.** Each `ArmScore` carries `present`. When
`present` is false, the arm never surfaced that id — which is categorically
different from surfacing it with a score of 0. The results table renders those
cells as `∅` on a tinted background, never as `0.0000`.

**The fusion is the client's.** memlake returns raw per-arm signal (dense cosine,
BM25, graph activation, temporal proximity) and does no fusion at all. The
default ordering is Reciprocal Rank Fusion computed **in your browser** —
`score(d) = Σ w_arm / (k + rank_arm(d) + 1)`, summed only over arms where
`present` is true. The weights and `k` are adjustable, and you can sort by any
single arm instead. The panel says so in its title; do not mistake that column
for something the server computed.

---

## The embedding model

The dense and graph arms need a query vector. The UI computes it **server-side**
with `BAAI/bge-small-en-v1.5` through transformers.js (`lib/embed.ts`). This has
to match the corpus's vectors exactly or recall degrades silently, so it mirrors
the benchmark harness (`bench/src/memlake_bench/embed.py`) in every respect:

- dim **384**, float32, **L2-normalized** (so cosine == dot product, and nothing
  renormalizes downstream)
- **CLS** pooling, `dtype: fp32`
- queries are prefixed with exactly

  ```
  Represent this sentence for searching relevant passages:␣
  ```

### Why the prefix matters

bge-\* retrieval models are trained with an **asymmetric instruction**: queries
carry that prefix, documents carry nothing. Dropping it costs several nDCG
points — the query lands in a different region of the space than the documents
it should match. The admin query box is a query, so it always gets the prefix.
It is part of the cache contract in the benchmark harness for the same reason.

### Why CLS pooling and not mean

bge models are CLS-pooled (`pooling_mode_cls_token` in their
sentence-transformers config), and `fastembed` — what the harness uses — pools on
CLS. Measured against fastembed on identical strings:

| pooling | cosine vs. harness | max component delta |
| ------- | ------------------ | ------------------- |
| `cls`   | 0.999999           | 3e-4 (ONNX noise)   |
| `mean`  | 0.94               | 2.6e-1              |

Mean pooling yields a perfectly plausible-looking vector that is simply *not* the
one the corpus was indexed against. If you change that line, re-run the
comparison first.

### First-request latency

The ONNX weights (~90MB) are downloaded and cached by transformers.js on first
use, and the pipeline is memoized on `globalThis` so hot reloads do not reload
it. The query page reads `/api/embed` on mount to show whether the model is
loaded, offers a **warm up** button, and labels the in-flight state as
"loading embedding model (first use downloads ~90MB)" rather than looking hung.

Set `MEMLAKE_EMBEDDINGS=off` to skip all of this; the query page then offers only
the raw-vector and text-only modes.

---

## Layout

```
admin/
  app/
    layout.tsx  page.tsx           namespaces list
    error.tsx  global-error.tsx    boundaries, so a throw is never a blank screen
    ns/[namespace]/
      layout.tsx                   namespace tab bar
      page.tsx                     stats
      browse/page.tsx  query/page.tsx
    api/
      embed/route.ts               GET status / POST warm up
      namespaces/route.ts          GET ListNamespaces / POST CreateNamespace
      namespaces/[namespace]/
        stats/route.ts  scan/route.ts  get/route.ts  query/route.ts
  components/                      all "use client"
    NamespacesView  StatsView  BrowseView  QueryView
    MemoryDetail  NamespaceNav  filters  ui
  lib/
    memlake.ts   the gRPC client: proto load, memoized channel, one promisified
                 wrapper per RPC, 30s deadlines, gRPC-code-aware errors
    convert.ts   wire objects -> the JSON contract (Buffers, u64 strings)
    embed.ts     bge-small-en-v1.5 via transformers.js
    types.ts     the JSON contract, shared by route handlers and components
    ids.ts       16 bytes <-> UUID           vector.ts  f32le <-> float32
    fusion.ts    client-side RRF             http.ts    route-handler plumbing
    client.ts    browser fetch helpers       format.ts  display formatting
```

`lib/memlake.ts`, `lib/convert.ts`, `lib/embed.ts` and `lib/http.ts` are
server-only and are never imported by a `"use client"` component. `next.config.ts`
additionally marks `@grpc/*`, `@huggingface/transformers` and `onnxruntime-node`
as external so the bundler leaves them alone.

## Error handling

Every RPC failure renders inline with its gRPC code, the server's verbatim
message, and an operator hint — `UNAVAILABLE` tells you to start the server,
`UNIMPLEMENTED` tells you the running binary predates the RPC. Nothing blanks the
page. Deadlines are 30s, so a hung server surfaces as `DEADLINE_EXCEEDED` rather
than a spinner that never stops.

## Checks

```bash
npm run build   # type check + lint + production build
npm run lint
```
