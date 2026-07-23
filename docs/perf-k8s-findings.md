# k8s performance investigation (hindsight-dev, real AWS S3)

Findings from running memlake on the `hindsight-dev` GKE cluster over the real
`hindsight-memlake-test` S3 bucket, behind the Envoy consistent-hash proxy. Load driver:
`perf/k8s_load.py` (in-cluster Job → proxy). serve×3 (2 vCPU, 1 GiB mem-cache each), indexer×1.

## Numbers

| run | writers | write/s | query p50 (warm) | notes |
|---|---|---|---|---|
| pre-queue, 256 MB cache | 1 | 82 | 62 ms | small cache; un-indexed tail |
| queue build, 1 GB cache | 1 | 121 | — | indexer starved (see F4), so index empty |
| fix build, 1 GB cache | 1 | 95 | 51 ms | indexer folding concurrently; still large tail |
| queue build, 1 GB cache | 8 (1 ns) | — | — | **failed**: writes timed out (F2, F3) |
| fix build, **6 namespaces** | 6×1 | **~828 agg** | — | scales across pods (F2 upside); per-ns 106–231/s |

Query p50 stays ~50 ms because in every run a large share of the corpus is still in the
**un-indexed WAL tail** (brute-force-scanned in-memory: 0 S3 roundtrips but CPU-heavy). A truly
drained steady-state number could not be captured — see F5.

## Findings

**F1 — Bigger serve cache ≈ 1.5× write throughput (82 → 121/s).** Write-time link derivation does
a vector query per doc against the snapshot; over real S3 those index reads dominate. Raising the
in-memory cache 256 MB → 1 GB keeps them warm. *Fix: `serve.memMb` default 1024 (`--mem-mb`).* The
write path is derivation-bound; ~100/s single-writer is the current per-namespace ceiling over real
S3, and that ceiling is CPU/S3-bound in derivation (owner is iterating on this via the `in_flight`
CPU-contention work).

**F2 — A hot namespace is bottlenecked on one serve pod.** Consistent-hash routing sends all of a
namespace's traffic to one pod (cache + commit affinity, by design). Under 8 concurrent writers that
pod hit ~1.4 of its 2 vCPU while the other two sat idle. A single hot namespace cannot use the
fleet — that is turbopuffer's **sharding** case, and the motivation for namespace **pinning** to
dedicated pods. Multi-*namespace* load spreads across pods; single-namespace load does not.

*Upside, measured:* 6 namespaces at once reached **~828 writes/s aggregate** (per-namespace
106–231/s) — ~8.7× the single-namespace ~100/s — as the hash spread them across the 3 pods. So the
system scales horizontally with tenants + pods. Caveat: with *few* namespaces the Maglev hash is
lumpy (observed ~4/2/0 across 3 pods — one pod at 1.7 CPU, one idle), so aggregate is capped by the
busiest pod; many namespaces even out, few hot ones want pinning.

**F3 — Envoy's 15 s route timeout hard-failed slow writes.** A derivation-heavy write under load
exceeded it and the RPC failed as `upstream request timeout` rather than merely being slow.
*Fix: configurable `proxy.routeTimeout`, default 120 s (committed).* 

**F4 — The queue indexer tight-looped on a poison namespace, starving all others.** The shared
bucket has ~15 leftover namespaces in an old on-disk format (`missing field version` /
`format version 5 incompatible`). `fold()` errors on them; the completion re-checked
`namespace_is_dirty`, which returns *true* when it cannot read the corrupt manifest, so the job was
re-queued and **immediately re-claimed → a tight loop** that never let any other namespace fold
(the single-writer baseline's index stayed empty at gen=0). *Fix: on a fold **error**, drop the job
instead of re-queueing; the 5-min reconciliation sweep retries it. Verified: poison now fails once
per sweep, real namespaces fold.* (This fix is deployed via the image; it lives in
`service.rs::run_indexer` and is pending commit — see below.)

**F5 — The deployed compaction path deterministically stalls (owner's in-flight work).** With the
queue fixed, `k8sbench-a` still would not fully drain: the indexer folds one **partial** chunk
(`docs=4608`, up to seq 13), publishes, then the *next* fold **hangs** — indexer at 27 m CPU (idle,
an S3 await, not a busy-loop), holding the queue claim with a live heartbeat, no progress for
minutes; a restart reproduces it exactly. This is in the size-tiered **partial-compaction** path
that is being actively edited, so the deployed snapshot has a stuck state; it is *not* a queue
issue (the queue behaves correctly throughout). It blocks a fully-drained steady-state query
measurement.

## Fixes landed

- `serve.memMb`/`diskMb` cache budgets (chart) — F1.
- `proxy.routeTimeout` 120 s (chart) — F3. *(committed)*
- `run_indexer`: drop-on-fold-error instead of re-queue — F4. *(deployed via image; uncommitted
  because it shares `service.rs` with in-flight work — the change is: track `fold_errored` in the
  drain loop, skip GC when set, and set `still_dirty = !fold_errored && namespace_is_dirty(...)`.)*

## Recommended next

- **F5**: fix the partial-compaction stall so a namespace fully drains (owner's area).
- **F4 polish**: have the reconciliation sweep skip namespaces whose manifest is unreadable
  (`namespace_is_dirty` → false on manifest error), so poison namespaces are not re-enqueued every
  sweep and the indexer is not diluted across them. Or delete the old-format namespaces from the
  test bucket.
- **F2**: for a hot namespace, either pin it to a dedicated serve Deployment (chart `proxy.pins`) or
  sharded internally; the single-pod ceiling is otherwise ~100 writes/s.
- **F1**: the write path is derivation-bound over S3 — the highest-leverage throughput work is
  making write-time derivation cheaper / less CPU-contended under concurrency.
