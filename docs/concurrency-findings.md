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
read-only vs **2.3 QPS / p50 4.9s** at 12 concurrent readers across 6 namespaces, *fully warm*.
Root cause was **not** CPU, cache-hit-rate, snapshot, or the query limiter — it was the disk-tier
cache re-`mmap`ing a file **on every lookup** and never promoting hot blocks to the (near-empty)
memory tier. One fix — promote disk hits into mem — took it to **283–399 QPS / p50 24–28ms
(~150×)**, and serve CPU went 60% → 616% (reads finally parallelize). ✅

Remaining: writes still disrupt reads (mixed p90 ~2.6s), and per-query cost scales with L0 segment
count. Details below.

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

## 🔬 Writes still disrupt reads (next target)

Full mixed (12 readers + 4 writers, 6 ns, warm), after the cache fix: **READ 16.8 QPS, p50 82ms**
(was 2.1 QPS / 7.4s — already ~8× QPS, ~90× p50), but **p90 2.6s / p99 4.5s**, and **WRITE p50 3s**.
The writers' `derive_links` is CPU-heavy and runs **inline on the write RPC** on the tokio worker
threads, so a burst of writes still stalls reads. Candidate fixes:
- Move link derivation **off the request path** (async/background), so a write acks after the WAL
  commit and links land in a later pass. Biggest lever for write latency + read isolation.
- Or run derivation on a **bounded blocking pool** (`spawn_blocking` with a semaphore) so it can't
  monopolize the async request-serving threads.
- Approximate derivation (`exact_rerank=false`, done earlier) already cut its cost ~4×.

---

## Already landed earlier this session (context)

- SIMD+prenorm cosine (~25× on the within-batch derive) and approximate no-rerank link derivation
  (~4× on the corpus-query part); 20k seed 174s → 66s.
- Per-call JSONL tracing (query/write/get/scan) + `in_flight` gauge — what made all of the above
  diagnosable.

---

## Open / to try next (the loop)

- ❓ **Write-path isolation** (above) — the current #1 remaining read-latency disruptor.
- ❓ **Size-based compaction** to bound read-time segment fan-out.
- ❓ **Speculative reads** — value unclear now that warm reads are ~24ms; only helps the cold/reopen
  tail. Park until the cold-tier (real S3) numbers exist.
- ❓ **500-namespace working set vs 512 MB mem cache** — with promotion, mem fills with the hot set;
  at 500 busy namespaces the combined hot set may exceed the budget → the noisy-neighbour /
  per-namespace isolation question (see TODOS "Cache: namespace isolation"). Needs a 500-ns run.
- ❓ **Cold-cache tail** — the first (cold) run showed p90 ~10s from warmup misses to S3; the promote
  fix doesn't help cold. Real S3 latency numbers still needed.

## What did NOT pan out
- The graph arm was *not* the cost (vector-only was no faster). It's the **vector arm's rerank**.
- Random query vectors inflate rerank ~1.5× vs realistic (near-cluster) queries — a benchmark
  caveat, but realistic queries were still ~400ms cold (segment fan-out), so not just an artifact.
