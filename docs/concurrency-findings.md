# Concurrency findings — mixed multi-namespace read+write

Working doc for the overnight optimization loop. **Target: ~500 namespaces per serve node,
3–12 concurrent readers+writers (mixed).** Not "one node handles 6k namespaces" — the 6k run was
just a stress probe. Everything below is measured on the release binary against MinIO on the dev
box (14 cores), driven by `memlake_perf.mixed` with `MEMLAKE_TRACE_LOG` on for the per-call
breakdown.

Status legend: ✅ fixed & verified · 🔬 diagnosed, fix pending · ❓ open.

---

## TL;DR

The mixed load exposed that **the read path collapsed under concurrency** — 145 QPS single-namespace
read-only vs **2.3 QPS / p50 4.9s** at 12 concurrent readers across 6 namespaces, *fully warm*. Two
fixes this session:

1. ✅ **Cache: promote disk hits into the memory tier.** The mem tier was near-empty (3MB/512MB) so
   ~all hits re-`mmap`ed a file *per lookup* → kernel serialization (serve at <1 core). → **~150×**.
2. ✅ **Lazy tail BM25 index.** Reopen/open eagerly built a tantivy index for the tail that
   vector-only queries never use. → median reopen **2s → 52ms**.

**Cumulative (full mixed, 12 readers + 4 writers, warm): READ 2.1 → 58.6 QPS (~28×), p50 7.4s → 26ms
(~280×); WRITE p50 3.0s → 495ms.** Both fixes committed, all tests green.

**#1 next task:** the p99 (~6s) is `full_open` after a fold — reuse persisting segments across a
fold (design in the doc). Details below.

---

## Method

- `uv run --project perf python -m memlake_perf.mixed --addr … --namespaces N --readers R --writers W --duration D`
  — concurrent readers+writers across N namespaces; reports per-op/per-namespace throughput+latency.
- Server trace (`MEMLAKE_TRACE_LOG=…jsonl`) gives per-call: `snapshot {action, open_ms, tail_entries}`,
  `phases_us`, `io {roundtrips, hit_ratio, tier}`, `permit_wait_ms`, `in_flight` (added this session),
  and for writes `link_ms / corpus_query_ms / within_batch_ms`.
- **Gotcha:** don't `rm` the trace file while the server holds it open — it keeps writing to the
  orphaned inode. Truncate with `: > file`.

---

## ✅ FIX #1 — cache: promote disk-tier hits into the memory tier (the big one)

**Symptom.** 12 readers, 6 namespaces, 0 writers, warm (hit-ratio 0.998): **2.3 QPS, p50 4.9s**
(solo was ~400ms). Ruled out via trace: `permit_wait=0` (limiter fine), `snapshot=reuse`,
`hit_ratio≈1.0`, `in_flight=12` on every read. **Serve CPU during the collapse was ~60% — under one
core of 14.** So not CPU-bound: reads were *blocked*, not computing.

**Root cause.** The two-tier cache's **memory tier held only 3 MB of its 512 MB budget**, so ~99.8%
of "hits" were **disk-tier** hits, and `read_blob` does `File::open` + `stat` + `mmap` **per
lookup**. At ~3400 lookups/query × 12 readers that's tens of thousands of mmap syscalls/sec →
kernel/VFS serialization (low CPU, high latency). Disk hits were **never promoted to mem** (by
design, to avoid a per-hit write lock), so the fast tier stayed empty forever.

**Fix.** On a disk-tier hit, promote the bytes into the memory ring (`cache.rs::get`). Cost is the
write lock **once per key** (until mem-evicted), not per hit — after promotion the key answers from
`state.mem` under the *shared* lock (map lookup + `Bytes`/Arc clone, no syscall), so concurrent
readers proceed in parallel. Admitted with the CLOCK reference bit clear (scan resistance
preserved); the disk entry is marked referenced (it was just hit). All 18 cache unit tests pass
(incl. `a_hit_buys_a_second_chance`, `cache_skew` policy table).

**Result (12 readers, 6 ns, 0 writers, warm):**

| | QPS | p50 | p99 | serve CPU |
|---|---|---|---|---|
| before | 2.3 | 4.9 s | 24 s | ~60% (<1 core) |
| **after** | **283–399** | **24–28 ms** | **~230 ms** | **616%** (6+ cores) |

Mem tier now fills to ~77 MB and hit-ratio → 1.000. This is the headline result of the session.

---

## 🔬 Per-query cost scales with L0 segment count

Even solo, a query on a 6k-doc namespace was ~400ms — because the namespace had **6 L0 segments**
(100/50/100/2000/2000/2000 docs) and the vector arm **fans out across every segment** (probe + scan
+ rerank per segment). The `rerank` phase was ~811ms; everything else ~0. Segments accumulate
because each fold makes a new L0 segment and compaction only triggers at `COMPACT_FANOUT=8` — so a
write-heavy namespace sits at 4–8 segments, and every read pays N× the single-segment cost.

Small writes make it worse: the load's 50-item write batches each became their own tiny segment
(the 100/50/100). Levers (pending): lower fanout or **size-based compaction** (merge many tiny
segments sooner); or defer/merge tiny flushes so they don't each become a segment.

Note: with the cache fix, warm per-query dropped from ~400ms to ~24ms even at 6 segments — because
the cost was the mmap-per-lookup, not the arithmetic. So segment fan-out matters most *cold* and for
the rerank candidate count; revisit after the write-path work.

---

## ✅ FIX #2 — lazy tail FTS (cheap reopen)

**Symptom (after fix #1).** Full mixed (12 readers + 4 writers, warm): READ 16.8 QPS, p50 82ms but
**p90 2.6s / p99 4.5s**, WRITE p50 3s. The slow reads were all `reopen_tail` with `open_ms` **1.7–3s**
for a *50-entry* tail, `roundtrips=0` (pure CPU, not S3).

**Root cause.** `reopen_extending_tail` (runs on every write that advances a namespace's head) and
`open` **eagerly built the tail's tantivy BM25 index** (`TantivyFts::build_with_tags`) — schema +
RAMDirectory + IndexWriter + commit, a fixed per-build overhead — even though **the vast majority of
queries are vector-only and never touch it**. Under write load, reopens are frequent and each paid
the build; the write path pays it too (derive opens a snapshot).

**Fix.** `FactType.tail_fts` is now a `OnceLock<TantivyFts>`, built **lazily on first text-arm use**
(`fts_arm`) from `tail_items` + the node's tokenizer. Vector-only reads and reopens skip it. 52
end-to-end tests pass (text arm still correct).

**Result (full mixed, warm):**

| | before (fix #1 only) | after fix #2 |
|---|---|---|
| READ QPS | 16.8 | **58.6** |
| READ p50 / p90 | 82ms / 2.6s | **26.5ms / 240ms** |
| WRITE p50 / p90 | 3.0s / 5.6s | **495ms / 1.2s** |
| median reopen `open_ms` | ~2s | **52ms** |

**Cumulative (session start → now):** READ 2.1 → 58.6 QPS (~28×), p50 7.4s → 26ms (~280×).

Remaining tail: p99 still ~6s (reads) / ~7s (writes); max reopen 5.9s — a few outliers, likely
`full_open` on a fold (segments changed → can't reuse) or heavy-contention moments. Next.

## ✅ FIX #3 — segment reuse across a fold + stop invalidating the snapshot on write

Two changes that together **eliminated `full_open` on the hot path** and collapsed the p99.

**3a — reuse persisting segments across a fold.** `FactType.segments` is now `Vec<Arc<SegmentState>>`
(per-segment sharing) and `SegmentState` carries its `seg-<uuid>` id. A new `reopen_after_fold`
reuses the `Arc` of every segment whose id survived the fold and reloads only the genuinely-new
ones (a flush adds one L0; a compaction replaces a few). `snapshot_traced`'s fold (200) branch now
calls it instead of `QueryNode::open`. Fold reopen dropped from 200–1200ms (`full_open`) to a median
**131ms** (`reopen_fold`). Proven identical to a fresh open by `reopen_after_fold_matches_a_fresh_open`.

**3b — don't drop the cached snapshot on write.** The write handler used to `invalidate` (remove)
the namespace's cached snapshot, so the *next read had nothing to reopen from and `full_open`ed*.
But `snapshot_traced` already re-validates the head every read (via the head pointer the write
bumped) and reopens the stale snapshot cheaply — `reopen_tail` (write) or `reopen_fold` (fold).
Removing the invalidate is safe (the reopen re-scans the tail to the new head, so the write is
always seen — visibility tests green) and turns the post-write read from `full_open` into
`reopen_tail`. **`full_open` count over a full mixed run: 229 → 0.**

**Result (full mixed, 12 readers + 4 writers, warm):**

| | before fix #3 | after fix #3 |
|---|---|---|
| READ QPS | 254 | **315** |
| READ p50 / p90 / p99 | 22 / 94 / 455 ms | **22 / 55 / 280 ms** |
| WRITE p50 / p99 | 287ms / 3.26s | **190ms / 1.15s** |
| snapshot actions | 229 full_open | **0 full_open** (reopen_tail/fold/reuse) |

### Cumulative (session start → now, full mixed 12r+4w warm)

| | start | now | factor |
|---|---|---|---|
| READ QPS | 2.1 | **315** | **~150×** |
| READ p50 | 7.4 s | **22 ms** | **~330×** |
| READ p99 | ~24 s | **280 ms** | **~85×** |
| WRITE p50 | 3.0 s | **190 ms** | **~16×** |

Three fixes: cache promotion, lazy tail FTS, segment-reuse + no-write-invalidate. All committed, all
tests green (cache, visibility, 53 end-to-end incl. two new reopen-equivalence tests).

## 🔬 (Resolved) `full_open` after a fold — kept for history

After fixes #1 and #2 the p50/p90 are excellent (26ms / 240ms), but **p99 ~6s (reads) / ~7s
(writes)** remains. The trace pins it: of reads > 1s, **23/25 are `full_open`** with `open_ms`
200–1200ms; slow writes are dominated by **`link_snapshot_ms`** (the derive opens a snapshot) — same
cause.

**Why.** The etag-reopen fast-path (fix from earlier this session) reuses the loaded segment
metadata only when the manifest is *unchanged* (a write → 304). A **fold changes the manifest**
(200), so every fold forces a `full_open` = reload+deserialize **all** segments' metadata
(centroids, tables, FTS split) for the 6+ L0 segments. It is once-per-fold-per-namespace (the first
read after a fold; then the snapshot is cached), so it is bounded by fold frequency — but it is the
whole p99, and it hits the write path too (the writer derives against a freshly-opened snapshot).

**Fix (designed, not yet built) — reuse the segments that persist across a fold.** A flush adds one
L0 segment; a compaction replaces a few with one — either way *most* segments persist unchanged
(same content-nonce `seg-<uuid>` id). So a fold-triggered reopen should reuse the loaded
`SegmentState` for persisting ids and only load the genuinely-new ones. Concretely:
1. `FactType.segments: Arc<Vec<SegmentState>>` → `Vec<Arc<SegmentState>>` (per-segment sharing;
   ~14 `.segments.iter()` sites are mechanical — deref still works).
2. Add the seg id to `SegmentState` (or parse it from `cluster_paths[0]`/`stats_path`).
3. A `reopen_after_fold(old, new_manifest)` that keys old segments by id, reuses the `Arc` for
   matches, loads only new ones, rebuilds the tail (as `reopen_extending_tail` already does).
4. `snapshot_traced` calls it on the 200 branch instead of `QueryNode::open`.
This makes a fold as cheap as a write-reopen in the common case, collapsing the p99 for reads and
writes together. **This is the #1 next task.**

## 🔬 Writes: derive still runs inline on the request path (secondary)

`derive_links` is CPU work run **inline on the write RPC** on the tokio worker threads. With the two
fixes it is no longer the dominant write cost (p50 495ms, mostly the full_open above), but a burst of
writes still competes with reads for worker threads. If write p99 stays a problem after the
full_open fix: move derivation **off the request path** (ack after the WAL commit, derive in a later
pass) or onto a **bounded blocking pool** (`spawn_blocking` + semaphore) so it can't monopolize the
request-serving threads. Approximate derivation (`exact_rerank=false`) already cut its cost ~4×.

---

## Already landed earlier this session (context)

- SIMD+prenorm cosine (~25× on the within-batch derive) and approximate no-rerank link derivation
  (~4× on the corpus-query part); 20k seed 174s → 66s.
- Per-call JSONL tracing (query/write/get/scan) + `in_flight` gauge — what made all of the above
  diagnosable.

---

## ✅/🔬 FIX #4 — partial (size-tiered) compaction + a `get_many` correctness fix

Implemented `minor_compact`: instead of the full O(corpus) rebuild the `COMPACT_FANOUT=8` count cap
does, merge only the newest run of **small** segments + the tail into one segment (O(small)), keeping
the older/larger ones. Carries the merged segments' supersede/predicate overlays forward so kept
segments' shadowed copies stay hidden. Proven identical-to-a-fresh-open — including deletes and
re-upserts in both the *kept* and *merged* segments — by `minor_compaction_preserves_correctness`.

**A real bug this surfaced (`get_many`):** `get` only filtered the *tail* tombstones; it never
checked the cross-segment `seg_superseded` overlay. So a delete or re-upsert **folded into a
segment overlay** (by any flush or compaction) while the old copy still lived in an older segment
**leaked through `get`**, even though `query`/`scan` correctly hid it. Fixed `get_many` to apply
`superseded()` like the query path. This is a genuine guarantee fix independent of compaction.

**Honest performance finding.** In the write-heavy mixed test, *any* early compaction trigger
**regressed** (315 → 150–230 QPS): the earlier fixes already made reads fast (p50 ~22ms even at 6
segments), so segment fan-out is **no longer the bottleneck — the single indexer is**. Firing
partial compactions frequently over-works it → WAL tail grows → reads slow. So the trigger is set
**conservatively** (`SMALL_SEGMENT_FANOUT=7`, just below the count cap): it fires no more often than
the cap's full rebuild but does the *cheaper* partial merge, and it only helps namespaces with a mix
of large + small segments (an all-small namespace's "partial" merge ≈ a full rebuild). The real
lever for the indexer bottleneck is the queue-based indexer being built (`index_queue`). (Caveat:
the `mix` namespaces are degraded from dozens of runs — docs 6k→13k+ — so late absolute numbers
aren't comparable to earlier ones.)

## Open / to try next (the loop) — priority order

1. ❓ **Indexer throughput/fairness** — now the top bottleneck under write load (the queue-based
   `index_queue` in progress). Compaction and write latency both gate on it.
2. ✅ **Size-tiered compaction** — DONE (fix #4), conservatively triggered; correctness-proven.
3. ❓ **Write-path isolation** — move `derive_links` off the request path or onto a bounded blocking
   pool, if write p99 persists after #1.
4. ❓ **500-namespace run** — with promotion the mem tier fills with the hot set; at 500 busy
   namespaces the combined hot set may exceed 512 MB → noisy-neighbour / per-namespace isolation
   (TODOS "Cache: namespace isolation"). Need an actual 500-ns seed to test the working-set math.
5. ❓ **Speculative reads** — low value now that warm reads are ~26ms; would only help the cold /
   full_open tail, which #1 addresses more directly. Park until real-S3 cold-tier numbers exist.
6. ❓ **Cold-cache tail** — a cold cache showed p90 ~10s from warmup misses to S3; the promote fix
   doesn't help cold. Needs real-S3 latency numbers.

## Reproduce

```
# release binary against the running MinIO, isolated bucket, tracing on:
MEMLAKE_QUERY_S3_BUCKET=mix … MEMLAKE_TRACE_LOG=$(pwd)/mix-trace.jsonl \
  target/release/mlake-server serve --addr 127.0.0.1:50052 --mem-mb 512 --disk-mb 4096 …
# seed once, then warm, then measure:
uv run --project perf python -m memlake_perf.mixed --addr localhost:50052 \
  --namespaces 6 --scale 6000 --readers 12 --writers 4 --write-batch 50 --duration 20
# analyse: jq over mix-trace.jsonl (snapshot.action, open_ms, phases_us, in_flight, link_*)
# NB: truncate the trace with `: > file`, never `rm` (server holds the fd → orphaned inode).
```

## What did NOT pan out
- The graph arm was *not* the cost (vector-only was no faster). It's the **vector arm's rerank**.
- Random query vectors inflate rerank ~1.5× vs realistic (near-cluster) queries — a benchmark
  caveat, but realistic queries were still ~400ms cold (segment fan-out), so not just an artifact.
