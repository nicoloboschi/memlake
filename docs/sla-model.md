# memlake SLA model

A capacity model: given the hardware you give one `serve` node, predict the read and write
SLA. The point of memlake's bounded-cache design is that this is *predeterminable* — so this
file is the function that does it, and every constant in it is either **[MEASURED]** (from the
perf harness), **[ARCH]** (a fixed property of the design), or **[DECIDE]** (a number we pick
and must agree on).

Status: **draft — constants not yet filled from a real run.** The logic is here to agree on
first; then we run MinIO to populate `[MEASURED]` and print a worked card.

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
| `CPU_S_PER_QUERY_1T` | CPU-seconds for one query, single type, top-k 20, warm | 0.035 s | `[MEASURED]` (28.5 QPS @ 1 vCPU, 92% util) |
| `FANOUT` | fact-type multiplier on per-query and per-write CPU | 3 | `[ARCH]` |
| `BYTES_READ_COLD_1T` | S3 bytes a single-type query reads on a full cache miss (nprobe `.vec` blocks + rerank point-fetches + payload for winners) | TBD | `[MEASURED]` |
| `S3_RTT` | one object-storage round-trip latency | 0 (MinIO) / ~20 ms (real S3) | `[MEASURED]`/param |
| `READ_WAVES` | coalesced round-trip waves per cold query (INV-7 bounds this to a constant) | ≤ 4 | `[ARCH]` |
| `CPU_S_PER_FOLDED_MEM_1T` | CPU-seconds to fold one memory into a segment, single type (assign + cluster/vec write + pk/radj/fts + rerank/payload) | TBD | `[MEASURED]` (write_bench) |
| `BYTES_WRITTEN_PER_MEM_1T` | S3 bytes written per folded memory, single type (its share of the segment objects) | TBD | `[MEASURED]` |
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

At the measured warm point (MinIO, 3 types): `cpu_qps = vcpu / (0.035 × 3) = ~9.5 QPS/vcpu`,
`p50 ≈ 105 ms`, `p99 ≈ 200 ms`. (Single-type is ~28 QPS/vcpu / 33 ms — the ×3 is the worst-case
tax.)

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

## The output: an SLA card

Given inputs, the model prints one card:

```
node: 4 vCPU, 16 GB, 200 GB disk, 10 Gb/s NIC, 8 namespaces, 3 fact types
cache tier at this working set: MEMORY (0 roundtrips)
READ   :  38 QPS   p50  105 ms   p99  210 ms
WRITE  :  wait_for_index, batch 1 000
          1 900 mem/s   ·   3.9 MB/s   ·   p50 …   p99 …
```

## What we still need to measure (the TBD constants)

Run the harness on MinIO to fill: `BYTES_READ_COLD_1T`, `CPU_S_PER_FOLDED_MEM_1T`,
`BYTES_WRITTEN_PER_MEM_1T`, `SNAPSHOT_BYTES_PER_NS`, `P99_FACTOR`, and confirm
`CPU_S_PER_QUERY_1T`. MinIO gives the CPU and byte constants honestly; it gives `S3_RTT ≈ 0`,
so the **cold-tier latency must be re-measured against real S3** before the card's COLD numbers
are quoted. Warm/MEMORY-tier numbers are valid from MinIO.
