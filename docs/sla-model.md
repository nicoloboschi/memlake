# memlake SLA model

A capacity model: given the hardware you give one `serve` node, predict the read and write
SLA. The point of memlake's bounded-cache design is that this is *predeterminable* — so this
file is the function that does it, and every constant in it is either **[MEASURED]** (from the
perf harness), **[ARCH]** (a fixed property of the design), or **[DECIDE]** (a number we pick
and must agree on).

Status: **warm/CPU constants filled from a real MinIO run** (4 vCPU, 100k corpus, D=384, 3 fact
types) — the worked card at the bottom is measured, not derived. Still needs real S3 (not MinIO)
for the COLD-tier latency and the write `commit` floor, plus `SNAPSHOT_BYTES_PER_NS` for the
multi-namespace working-set math. Two caveats gate a *per-namespace, multi-tenant* SLA: the cache
has no inter-namespace isolation and the indexer folds namespaces sequentially (both in TODOS) —
so today's numbers are a single-busy-namespace SLA.

## Inputs

| input | unit | what it bounds |
|---|---|---|
| `vcpu` | cores | query/fold CPU throughput |
| `memory` | GB | the memory cache tier + per-namespace resident snapshots + working set |
| `disk` | GB | the disk (NVMe) cache tier |
| `nic` | GB/s | bytes moved to/from S3 (cold reads, all writes) |
| `namespaces` | count | concurrently-hot namespaces sharing this node |
| `batch` | memories | how many a write folds in one `wait_for_index` call (see writes) |

## Worst-case assumptions (fixed, not inputs)

- **3 fact types always.** Every read fans out to 3 independent per-type indexes; every write
  indexes 3 types. So per-query and per-write CPU carry a **×3 fan-out** over the single-type
  measurement. `[ARCH]`
- **`wait_for_index = true` on every write.** The write ack does not return until the tail is
  folded into a segment. So **write latency = commit + synchronous fold**, and write throughput
  is fold-bound, not commit-bound. `[ARCH]`
  - **Open (Hindsight to confirm):** whether the fold is per-write or amortized over a batch.
    Hindsight writes different documents in parallel but batches a document's chunks inside a
    txn, so the effective `batch` is a document's chunk count, not 1. The model takes `batch`
    as an input: `batch=1` is the pessimal per-write SLA; a real document batches its chunks
    into one fold. **If `wait_for_index` truly fires per single write, that is a design flaw to
    fix (batch the fold), not a number to quote — flagged for investigation.**
- Reads are **strongly consistent** (always scan the WAL tail) — the only mutable read on the
  path, one LIST per read. `[ARCH]`

## Constants

| symbol | meaning | value | source |
|---|---|---|---|
| `AVG_MEM_BYTES` | avg stored size of one memory (embedding + text + metadata), on the wire for a write | **2 048 B** (384-dim f32 ≈ 1.5 KB embedding + ~0.5 KB text/meta) | `[DECIDE]` — prod at 1536-dim ≈ 6.8 KB; pick per workload |
| `CPU_S_PER_QUERY_1T` | CPU-seconds for one query, single type, top-k 20, warm | **~0.056 s @ 100k corpus** (was 0.035 s at a smaller corpus — see note) | `[MEASURED]` (65 QPS @ 4 vCPU, 91% util, MinIO, 100k/D=384) |
| `FANOUT` | fact-type multiplier on per-query and per-write CPU | 3 | `[ARCH]` |
| `BYTES_READ_COLD_1T` | S3 bytes a single-type query reads on a full cache miss (nprobe `.vec` blocks + rerank point-fetches + payload for winners) | TBD | `[MEASURED]` |
| `S3_RTT` | one object-storage round-trip latency | 0 (MinIO) / ~20 ms (real S3) | `[MEASURED]`/param |
| `READ_WAVES` | coalesced round-trip waves per cold query (INV-7 bounds this to a constant) | ≤ 4 | `[ARCH]` |
| `CPU_S_PER_FOLDED_MEM_1T` | CPU-seconds to fold one memory into a segment, single type (assign + cluster/vec write + pk/radj/fts + rerank/payload + link derivation) | ~0.45 ms (valid ≤100k; super-linear beyond, see below) | `[MEASURED]` (MinIO, 100k @ 4 vCPU, 34 s) |
| `BYTES_WRITTEN_PER_MEM_1T` | S3 bytes written per folded memory, single type (its share of the segment objects) | ~0.8 KB/type (~2.4 KB across 3 types; f32 rerank dominates) | `[MEASURED]` (MinIO, 32k folded @ D=384 Binary) |
| `SNAPSHOT_BYTES_PER_NS` | resident RAM to hold one namespace's open snapshot (centroids + sparse indexes + FTS split + tail), 3 types | TBD | `[MEASURED]` |

## The cache-tier model (this is the crux — please sanity-check the shape)

memlake's cost is predeterminable because a read lands in exactly one tier, and the tier sets
both latency and whether the NIC is touched:

```
working_set  = namespaces × hot_bytes_per_ns          # the bytes queries actually re-touch
mem_cache    = memory − namespaces × SNAPSHOT_BYTES_PER_NS   # RAM left for the block cache
                                                              # after resident snapshots

if working_set ≤ mem_cache:      tier = MEMORY   # 0 roundtrips, 0 NIC, pure CPU
elif working_set ≤ disk:         tier = DISK     # NVMe read, 0 NIC (served locally)
else:                            tier = COLD     # S3: NIC + S3_RTT × READ_WAVES
```

`hot_bytes_per_ns` = the fraction of a namespace's data that queries actually re-read — the
probed clusters' `.vec` blocks + the small resident metadata, **not** the whole corpus. This is
the number that makes object-storage-native viable: the working set is `nprobe`-bounded, not
corpus-bounded. `[ARCH]`, magnitude `[MEASURED]`.

**Decided: three hard tiers.** This is the "bounded by construction" story — a namespace's
working set either fits a tier or it doesn't, and the SLA is the tier it lands in. A blended
hit-ratio would imply partial occupancy the cache doesn't cleanly have.

### The multi-namespace requirement the tier model imposes on the cache

The tier formula uses `working_set = Σ (busy namespaces × their hot set)` and assumes each busy
namespace's hot set **stays resident**. For that to hold across concurrent namespaces the cache
must give each busy namespace its share and protect it from a neighbour — and **the current
global CLOCK does not.** CLOCK is scan-resistant (a one-pass `Scan`'s blocks are evicted before
they earn a second reference) but it is **not isolated**: one namespace's large cold scan, or a
sudden burst, floods the single shared cache and evicts a *different* busy namespace's hot
blocks for the duration — a noisy neighbour. That namespace then drops from MEMORY to COLD tier
and its p99 spikes until it re-warms, which the SLA model would not predict.

So the model's per-namespace guarantee is only real if the cache is **namespace-aware**. Two
ways to get there, both stronger than CLOCK:
- **Per-namespace reservation** — divide the cache among active namespaces (weighted by
  traffic), so each busy one is guaranteed its working set. Simplest to reason about; the SLA
  becomes `mem_cache_per_ns = mem_cache / active_namespaces` and the tier check is per-namespace.
- **Frequency-aware admission (TinyLFU / S3-FIFO)** — admit by access frequency, so a busy
  namespace's repeatedly-hit blocks always beat a cold scan's one-shot blocks, emergently,
  without per-namespace bookkeeping.

**This is a design item the SLA model surfaced** (recorded in TODOS): to give a per-namespace
SLA under concurrency, the disk/memory cache needs isolation or frequency-awareness, not the
current global recency CLOCK. Until then, the multi-namespace tier prediction holds only when
the *combined* working set of all busy namespaces fits — there is no protection between them.

## Read SLA

```
cpu_qps  = vcpu / (CPU_S_PER_QUERY_1T × FANOUT)          # CPU ceiling
nic_qps  = (tier == COLD) ? nic / (BYTES_READ_COLD_1T × FANOUT) : ∞   # NIC only bites when cold
QPS      = min(cpu_qps, nic_qps)

service  = CPU_S_PER_QUERY_1T × FANOUT                   # warm service time (CPU)
          + (tier == DISK)  ? disk_read_latency          # small, NVMe
          + (tier == COLD)  ? S3_RTT × READ_WAVES        # the S3 tail

# Latency at a right-sized concurrency (≈ vcpu in flight). Beyond that it is pure queue —
# the model reports the SLA at the *knee*, and flags queue growth past it.
p50  = service
p99  = service × P99_FACTOR                              # tail multiplier, [MEASURED] (~2× warm)
```

**Measured warm (MinIO, 100k corpus, D=384, top-k 20, 0 roundtrips = MEMORY tier):**

| | throughput ceiling (saturated) | latency at right-sized conc (≈vcpu) |
|---|---|---|
| single type | **65 QPS @ 4 vCPU (~16 QPS/vcpu)**, 91% CPU | p50 **82 ms**, p99 **190 ms** (conc=4, 48 QPS) |
| 3 types (worst case) | **17.5 QPS @ 4 vCPU (~4.4 QPS/vcpu)**, 88% CPU | p50 **238 ms**, p99 **547 ms** (conc=4) |

So `P99_FACTOR ≈ 2.3` warm, and the ×3 fan-out tax is real: worst-case 3-type recall is ~4.4
QPS/vcpu, not the single-type ~16.

**Important — the per-query cost grows with corpus size.** These are ~2× the numbers measured on
a smaller corpus (single-type was ~28 QPS/vcpu / 33 ms), because `nprobe` is index-derived (a
fraction of clusters, capped at 64) so a bigger corpus scans more per probe. `CPU_S_PER_QUERY_1T`
is therefore a function of corpus size; **0.056 s is the 100k figure** and must be re-measured at
the target namespace size. Read the QPS ceiling and latency together: at conc past the knee the
system is pure queue (single-type p50 goes 82→229 ms from conc 4→16), so QPS rises but latency
degrades — size concurrency to ≈vcpu for the latency SLA.

## Fold throughput — measured, and what the "hang" actually was

**Measured (MinIO, 4 vCPU, single isolated namespace, 3 fact types):** a 100k-memory bulk
fold completes in **~34 s ≈ ~2 900 docs/s (~725 docs/s/vCPU)**; a 40k fold in ~16 s (~2 500
docs/s) — consistent. So `CPU_S_PER_FOLDED_MEM_1T × FANOUT ≈ 34 s / 100k ≈ 0.34 ms/memory` at
4 vCPU, i.e. **`CPU_S_PER_FOLDED_MEM_1T ≈ 0.45 ms`** (single-type, ÷3 fan-out, ×4 vcpu).

**The "indexer hang" was starvation, not fold cost.** An earlier 100k run appeared to crawl at
~35 docs/s and never converge. That was **not** the fold: the shared MinIO bucket had
accumulated 18 `ext-test-*` corpse namespaces (a parallel Hindsight test writing to the same
endpoint), and the single indexer folds every discovered namespace **sequentially in one
thread**, so the target namespace starved behind the others. Isolated to one namespace, the same
100k folds in 34 s. The real defect is the indexer's lack of per-namespace fairness (TODOS
§"Single indexer folds all namespaces sequentially"), not the per-memory fold cost.

**Semantic-link derivation moved OFF the fold and onto the write path.** Links are now derived
before the commit (`derive_links_for_write`), not by the fold — so the fold no longer carries the
O(new·N) derivation cost, and the fold-breakdown numbers above (which predate the change) overstate
today's fold cost by that pass. The O(new·N) cost is real but now on the *write* path, against the
current snapshot (indexed segments + un-indexed tail). It stays bounded as long as the indexer keeps
pace so the tail a write scans is small; a lagging indexer degrades writes toward O(tail). This is a
scaling watch-item (TODOS), not a blocker. Re-measure before quoting `CPU_S_PER_FOLDED_MEM_1T` at
≫1M memories/namespace.

## Write SLA (wait_for_index = true, 3 types)

A write acks after commit **and** fold, so throughput is the fold rate and latency includes it:

```
fold_cpu_wps   = vcpu / (CPU_S_PER_FOLDED_MEM_1T × FANOUT)       # CPU ceiling on folding
nic_write_wps  = nic  / (BYTES_WRITTEN_PER_MEM_1T × FANOUT)      # NIC ceiling on writing objects
WPS_mem        = min(fold_cpu_wps, nic_write_wps)                # memories/second
WPS_bytes      = WPS_mem × AVG_MEM_BYTES                         # bytes/second (ingest)

# Latency of one write batch of B memories, wait_for_index:
commit    = S3_RTT × commit_waves            # the durable WAL PUT (the ack floor)
fold      = B × CPU_S_PER_FOLDED_MEM_1T × FANOUT / vcpu   # synchronous fold of the batch
p50_write = commit + fold
p99_write = (commit + fold) × P99_FACTOR
```

The `wait_for_index` fold dominates: committing is one durable PUT, but folding B memories
across 3 types is `B × per-memory-fold × 3`. This is why `wait_for_index` writes are the
heavy path and the model treats them as worst case.

**Open question:** is `wait_for_index` per-write, or does a real ingest batch it? If Hindsight
writes N memories then waits once, the per-memory fold latency amortizes and WPS is far higher
than 1-memory-at-a-time. The model should take a **batch size `B`** as a write input; at `B=1`
it is the pessimal per-write SLA, at `B=10 000` it is the bulk-ingest SLA. **Recommend adding
`batch` as an input.**

## The output: an SLA card (worked, from measured constants)

All numbers below are **measured end-to-end on MinIO** (100k corpus, D=384, 4 vCPU, MEMORY tier,
`S3_RTT ≈ 0`) — not derived from a formula:

```
node: 4 vCPU, 16 GB, 200 GB disk, 10 Gb/s NIC, working set in MEMORY tier, 100k memories

READ   (worst-case = all 3 fact types fan out):
   ceiling  ~17.5 QPS @ 4 vCPU  (~4.4 QPS/vcpu, 88% CPU)
   latency  p50 238 ms   p99 547 ms   (at right-sized conc ≈ 4)
   (single-type, the common case:  ceiling ~65 QPS,  p50 82 ms,  p99 190 ms)

WRITE  (wait_for_index = true, fold-bound; NIC ~500k wps so never binds at MinIO):
   throughput  ~2 900 mem/s @ 4 vCPU  ·  ~6.0 MB/s ingest (@2 KB/mem)
               [MEASURED: 100k folded in ~34 s]
   batch latency (B memories, one wait_for_index call):
     fold = B × 0.00045 × 3 / 4  →  B=1: ~0.3 ms   B=1 000: ~0.34 s   B=10 000: ~3.4 s
     p50_write ≈ commit(one WAL PUT) + fold ;  p99 ≈ ×2
```

Read the READ line as the **worst case** — every recall fans out to all 3 fact types. A typical
single-type query is ~4× faster (the parenthetical). WRITE throughput is CPU-fold-bound at MinIO;
`wait_for_index` latency is dominated by the synchronous fold of the batch, so batching is the
single biggest write-latency lever (B=1 is pessimal, a real document's chunks amortize it).

**Caveats on this card:**
- **Per-namespace under concurrency is not yet guaranteed.** These numbers assume the busy
  namespace's working set stays resident. The current global CLOCK cache gives no inter-namespace
  isolation (TODOS §"Cache: namespace isolation"), so a noisy neighbour can drop a namespace to
  COLD and break the READ p99. The card is a single-busy-namespace SLA until that lands.
- **Write throughput assumes one namespace per indexer.** The single indexer folds all namespaces
  sequentially; with N busy namespaces, per-namespace fold rate divides by N (TODOS §"Single
  indexer folds all namespaces sequentially"). The 2 900 mem/s is an isolated-namespace figure.
- **Write-path derivation re-check needed ≫1M/namespace** — link derivation is O(new·tail) on the
  write path (fine when the indexer keeps the tail small; degrades toward O(tail) if it lags).

## What still needs real S3 (not MinIO)

MinIO gives the CPU and byte constants honestly but `S3_RTT ≈ 0`, so the **COLD-tier read
latency** (`S3_RTT × READ_WAVES`) and the write `commit` floor must be re-measured against real
S3 before those rows are quoted. `SNAPSHOT_BYTES_PER_NS` (for the cache-tier working-set math) and
`P99_FACTOR` still need a dedicated pass. Warm/MEMORY-tier READ and CPU-bound WRITE numbers above
are valid from MinIO.
