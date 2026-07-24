# COGS: memlake vs Postgres, for Hindsight

The goal is to capture the *real* cost of goods sold, not to make memlake win. Against a realistic
**Multi-AZ + 2-read-replica** Postgres, the short version is:

- **PG's only territory is small corpus + high QPS** (10 k @ 100 QPS, and 100 k @ 100 QPS against a
  cheap PG). It is narrow, and coalescing one S3 GET would erase most of it.
- **Wherever the corpus is large, memlake wins by one to two orders of magnitude** — and adding QPS
  does not rescue PG: at 10 M memories, 1 → 100 QPS costs memlake **+$192/mo** and it is still
  **48× cheaper** ($222 vs $10 655).
- **Storage is never the deciding line** at any scale here; it is compute + S3 requests vs
  provisioned RAM × replicas.

Two parts:

1. **[Measured total COGS](#measured-total-cogs-storage--requests--compute)** — the headline.
   Storage bytes, S3 request counts and query costs are **measured on the real AWS S3 k8s
   deployment** (not modelled); instance prices are AWS list. This supersedes the napkin math for
   decision-making.
2. **[The storage model](#the-two-architectures-at-the-storage-layer)** (everything from "The two
   architectures" down) — the original order-of-magnitude byte analysis. Still useful for *why*
   the bytes differ; its `$/month` table is storage-only and at much larger scales.

## Measured total COGS (storage + requests + compute)

### Measured inputs

From the deployment in [`sla-model.md`](sla-model.md) (GKE → AWS S3 us-east-1, serve 2 vCPU/3 GiB,
`--mem-mb 1024`), measured by `aws s3 ls --recursive` on real folded namespaces and by the fleet
traces:

| input | measured | how |
|---|---|---|
| durable index bytes/memory, D=384, **real text** (scifact abstracts) | **5 950 B** | 30.8 MB of segments+manifest ÷ 5 183 docs |
| durable index bytes/memory, D=384, tiny text (synthetic) | **2 895 B** | 57.9 MB ÷ 20 000 docs |
| same, **projected to D=1536** (4·D rerank + D/8 codes, rest fixed) | **10 702 B** / 7 647 B | scaling the measured split |
| WAL bytes/memory (transient — GC reclaims after the fold) | 3 304 B | not counted in steady state |
| PUTs for a full build | **0.034 / memory** | 173 objects ÷ 5 183 docs |
| **S3 GETs per warm query** | **1** | the `resolve_head` WAL-head pointer — *uncacheable* |
| S3 GETs per cold query | 4.3 | 1 head + 3.34 measured data waves |
| QPS per serve vCPU | 28 | `compute-cost.md` (MinIO; CPU-valid, realistic top-k query) |

The single most important line is **1 GET per query, always**. Reads are strongly consistent, so
every query re-reads the WAL head — one small, mutable, *uncacheable* object. It is a GET, not a
LIST (`resolve_head` deliberately avoids the 12.5×-priced LIST), but it does not go away when the
cache is warm.

### Prices used

AWS **us-east-1 on-demand list**: S3 Standard $0.023/GB-mo, GET $0.0004/1k, PUT $0.005/1k; RDS
PostgreSQL single-AZ $0.016/hr (t4g.micro) → $3.616/hr (r6g.8xlarge); gp3 $0.115/GB-mo (20 GB min);
EC2 Graviton $0.0336/hr (t4g.medium) → $0.308/hr (m6g.2xlarge). **Verify before quoting** — these
move, and reserved/savings plans cut the compute line ~40 % on *both* sides.

### The table (D=1536, real-text byte model, $/month)

memlake = S3 storage + S3 requests + one always-on box (serve + indexer co-located, scaling with
QPS). PG is priced in **three topologies**, because replication is most of the answer:
single-AZ (cheapest honest floor), Multi-AZ (primary + standby), and **Multi-AZ + 2 read replicas —
the realistic Hindsight prod config**. Each replica carries its own instance *and* storage.

PG instance size is also sensitivity-tested on how much of the ANN index you insist stays resident:
**all-RAM** (vectors + ivfflat's near-full copy in memory — what a latency SLA wants) vs **25 % hot**
(only the probed lists resident, accepting disk reads — cheaper and slower).

| memories | memlake @1 / @10 / @100 QPS | PG all-RAM: 1AZ / MultiAZ / **+2RR** | PG 25 % hot: **+2RR** |
|---:|---:|---:|---:|
| **10 k** | **$26 / $35 / $218** | $14 / $28 / **$56** | **$56** |
| **100 k** | **$26 / $35 / $218** | $50 / $100 / **$199** | **$56** |
| **1 M** | **$26 / $35 / $218** | $332 / $665 / **$1 329** | **$509** |
| **10 M** | **$30 / $39 / $222** | $2 664 / $5 327 / **$10 655** | **$2 736** |

memlake vs the realistic **PG Multi-AZ + 2 RR**, as a multiple (>1 = memlake cheaper):

| memories | @1 QPS | @10 QPS | @100 QPS | (25 % hot PG, @100 QPS) |
|---:|---:|---:|---:|---:|
| 10 k | 2.2× | 1.6× | **0.3× — PG wins** | 0.3× — PG wins |
| 100 k | 7.8× | 5.7× | 0.9× — parity | 0.3× — PG wins |
| 1 M | 51× | 38× | **6.1×** | 2.3× |
| 10 M | **358×** | **272×** | **48×** | **12×** |

### What the shape actually is

**The two cost curves scale on different axes.** memlake = a small fixed floor + a line that is
**linear in QPS** (one S3 GET per query) and nearly flat in corpus size. PG = a line that is
**linear in resident bytes × replica count** and flat in QPS until the box saturates. So:

- **PG only wins where the corpus is small and QPS is high** — 10 k @ 100 QPS, and 100 k @ 100 QPS
  against a cheap PG. That is the whole of PG's territory, and it is narrow.
- **Everywhere the corpus is large, memlake wins by one to two orders of magnitude**, and adding QPS
  does not rescue PG: at 10 M, going from 1 → 100 QPS costs memlake **+$192/mo** and it is *still*
  **48× cheaper** than PG Multi-AZ + 2 RR ($222 vs $10 655). The $105 request line is only
  significant when it is compared against a $14–56 instance, i.e. only at toy corpus sizes.

**Why the 10 M gap is that large** (it is the number that looks implausible, so: ) at 10 M × D=1536,
PG must keep ~139 GB — 61.5 GB of vectors, another ~61.5 GB because ivfflat stores a near-full second
copy in its lists, ~16 GB of rows/links — on **provisioned** RAM, and then pay for it **four times**
across primary + standby + 2 replicas. memlake keeps the same corpus in S3 for **$2.46/mo** and
provisions RAM only for the `nprobe`-bounded working set it actually touches. Even forcing PG down to
"25 % hot" (a real latency sacrifice) it is $2 736 vs $222. **This gap is the architecture, not a
pricing trick** — it is the same reason the design is object-storage-native in the first place.

### The head GET is the fix that widens the narrow case

memlake's marginal cost per query is *one S3 GET* (the uncacheable WAL-head read). That line only
decides anything in PG's narrow territory — small corpus, high QPS — but it is cheap to attack:
coalescing **concurrent** in-flight head reads onto a single GET costs **nothing** in consistency
(simultaneous readers may legitimately share one head observation) and cuts the line by the in-flight
factor. A short TTL would cut it further but does weaken strong consistency, so that one is a product
decision. At 100 QPS a 10× coalescing factor turns $105/mo into ~$10/mo, which flips **10 k @ 100 QPS
and 100 k @ 100 QPS back to memlake** — i.e. it removes PG's remaining territory almost entirely.

### What this table does NOT model — read before quoting the 10 M row

- **memlake's 10 M column is extrapolated, and it is the optimistic direction.** `28 QPS/vCPU` was
  measured at **100 k** on MinIO, and `compute-cost.md` explicitly warns against extrapolating it:
  per-query CPU grows with corpus (more candidates per probe). The request line assumes **warm**
  (1 GET/query); if a 10 M working set does not fit the 1 GB memory + 8 GB disk cache, queries go
  cold at **4.3 GETs** — 4.3× that line ($450/mo at 100 QPS instead of $105). Both effects push
  memlake up. Neither closes a 12–358× gap, but the 10 M row should be read as "one to two orders of
  magnitude", not as three significant figures.
- **memlake at 1 fact type.** All measurements are single-type; the stated worst case is
  `FANOUT = 3`, which triples query **CPU** (the compute column, not the request column).
- **No scale-to-zero credit.** memlake's serve is stateless and *could* scale to zero between
  queries, erasing the $25 floor and winning 10 k outright. Not implemented, so not credited.
- **PG's ANN recall is assumed adequate, not measured.** The "25 % hot" row is a cost/latency trade
  this table prices only on the cost side; a disk-resident ivfflat is materially slower, and matching
  memlake's recall may need HNSW, which is *more* memory-hungry than the ivfflat modelled here.
- **PG's vacuum/bloat headroom** is only the ×1.5 storage factor; index rebuilds are not priced.
- **Reserved instances / savings plans** cut the compute line ~40 % on both sides, so they compress
  the ratios somewhat but do not change any winner.
- **Egress, backups, ops labour** — see "What this leaves out" at the end. Backups notably favour
  memlake (the bucket *is* the durable copy; RDS snapshots are a separate billed line ×4 copies).

## The two architectures, at the storage layer

## The two architectures, at the storage layer

| | Postgres (Hindsight's historical path) | memlake |
|---|---|---|
| Medium | EBS / Aurora block storage | S3 object storage |
| Durability model | you pay per **copy**: primary + standby + read replicas | 11-nines baked into the **single** `$/GB` price (S3 replicates across ≥3 AZs internally, not billed per copy) |
| Vector representation | pgvector `vector(D)` = full f32, **plus** the ANN index (ivfflat/HNSW) stores **another** near-full copy | codes (1-bit or int8) for the scan + one f32 copy in the rerank tier; no index-side duplication of vectors |
| Replication cost | linear in replica count | zero (in the `$/GB`) |

The headline is **not** "memlake stores fewer bytes per memory." With the two-stage search
keeping a full-precision rerank tier, memlake's bytes/memory are *comparable* to one Postgres
copy of the vector. The cost gap comes almost entirely from two things: the **medium**
(`$0.023/GB` S3 vs `$0.10–0.125/GB` block) and the **replication model** (S3's copies are free,
PG's are billed 2–4×).

## Prices used (AWS us-east-1, 2026, approximate)

| item | price |
|---|---|
| S3 Standard storage | $0.023 / GB-month (0–50 TB tier) |
| S3 Standard-Infrequent Access | $0.0125 / GB-month |
| S3 PUT / COPY / POST | $0.005 / 1,000 |
| S3 GET | $0.0004 / 1,000 |
| RDS gp3 storage (per copy) | $0.115 / GB-month |
| Aurora storage (incl. its 6-way internal replication) | $0.10 / GB-month |
| EBS gp3 (self-managed PG on EC2, per copy) | $0.08 / GB-month |

These move; the *ratios* between them are what the analysis rests on, and those are stable.

## Per-memory byte model

A "memory" = short text + a little metadata + one embedding. The embedding dominates. We
parametrise on embedding dimension `D`. memlake's numbers are **measured** at `D=384`
(bge-small, via `ListObjects`/`DecodeObject` — see `docs/arms/vector.md`) and scaled to
larger `D` by holding the ~300 B non-embedding part fixed and scaling the embedding linearly.
Hindsight production likely uses a larger embedding; the base case below is `D=1536`
(e.g. OpenAI `text-embedding-3-small`), with `D=384` shown for reference.

Non-embedding logical payload assumed ~500 B text + ~200 B metadata/ids ≈ **700 B**.

### Postgres, one copy (per memory, D=1536)

| component | bytes | note |
|---|---|---|
| `memory_units` row (text, jsonb metadata, ts, ids, 23 B header, alignment) | ~1,000 | text+metadata TOAST-inlined at this size |
| pgvector `vector(1536)` value | ~6,150 | `D*4 + 8`; random floats do not compress, so TOAST does not help |
| ANN index copy of the vector (ivfflat: full vector in its lists) | ~6,150 | **HNSW is worse: ~1.3–2× the vectors + graph links** |
| btree PK on id + other supporting indexes | ~250 | |
| `memory_links` (derived edges; ~10 out-edges/memory × ~70 B + index) | ~1,000 | Hindsight derives these |
| `unit_entities` (entity postings) | ~300 | |
| **one copy total** | **≈ 14.8 KB / memory** | |

Plus a **headroom factor**: you never run block storage at 100%. Assume provisioned at 70%
utilisation → divide usable by 0.7, i.e. ×1.43 on provisioned GB. Plus WAL, bloat, and
autovacuum slack (another ~10–20%). Call it **×1.5** provisioned-over-logical.

### memlake, single copy in S3 (per memory, D=1536)

Default codec is **Binary scan + f32 rerank** (the turbopuffer-style two-stage design):

| component | bytes | note |
|---|---|---|
| payload `.bin` (text, tags, metadata, edges — no embedding) | ~300 | measured ~298 B at D=384, D-independent |
| vector `.vec` (1-bit codes + corrective + ids + tag bitmaps) | ~240 | `D/8` codes = 192 B at D=1536 + framing |
| rerank tier `.data` (full f32, point-fetched only) | ~6,180 | `D*4` + ~1.5% SSTable framing |
| pk / radj / entity / time / centroids / fts (amortised) | ~500 | small, shared |
| **total** | **≈ 7.2 KB / memory** | one copy; S3 needs no headroom factor |

Two honest sub-cases:

- **If recall allows int8-only or binary-only** (no full-precision rerank tier): drops to
  **~2.0 KB / memory** (`300 + 1536` int8) or **~0.8 KB** (binary + payload). BEIR said binary
  and int8 gave *identical* `ann_recall@10` once reranked — but dropping the rerank tier means
  giving up the exact rescore, so this is a recall decision, not free. Shown as a floor.
- **memlake keeps full precision by default**, so its per-memory bytes are ~half of PG's *one
  copy* (7.2 vs 14.8 KB) — the saving is the absence of the ivfflat/HNSW vector duplication and
  the links/entities tables, not compression.

## Effective `$/GB-month`, replicated

This is where the models diverge. "Logical GB" = one copy of the useful data.

| deployment | copies | provisioned factor | `$/logical-GB-month` |
|---|---|---|---|
| **memlake** — S3 Standard | 1 (free internal ≥3 AZ) | 1.0 | **$0.023** |
| memlake — S3 Standard-IA (cold tiers) | 1 | 1.0 | $0.0125 |
| PG — Aurora (primary; storage price includes replication) | 1 priced | 1.5 | $0.10 × 1.5 = **$0.15** |
| PG — RDS Multi-AZ (primary + standby) | 2 | 1.5 | $0.115 × 2 × 1.5 = **$0.345** |
| PG — RDS Multi-AZ + 2 read replicas | 4 | 1.5 | $0.115 × 4 × 1.5 = **$0.69** |
| PG — self-managed EC2, 3 copies (1 primary + 2 replicas) | 3 | 1.5 | $0.08 × 3 × 1.5 = **$0.36** |

So per useful GB, memlake is **~6× cheaper than Aurora**, **~15× cheaper than RDS Multi-AZ**,
and **~30× cheaper than a Multi-AZ + 2-replica cluster** — before accounting for memlake also
storing ~half the bytes/memory.

## Putting it together: `$/month` by scale

> **Storage line only — not a total.** The table below prices storage in isolation, which is why it
> shows large ratios. At every scale Hindsight actually cares about, storage is a rounding error and
> the total is decided by compute + S3 requests — see
> [Measured total COGS](#measured-total-cogs-storage--requests--compute), which is the number to use.
> Note also that the 7.2 KB/memory modelled here is close to the **measured** 7.6 KB for a tiny-text
> corpus, but real text (scifact abstracts) measured **10.7 KB** at D=1536 — the non-embedding part
> is workload-dependent and this model's 700 B assumption is low for prose.

Total storage $/month = (memories × bytes-per-memory-per-copy × copies × provisioned) × `$/GB`.

memlake: 7.2 KB/memory, ×1 copy, ×1.0, × $0.023/GB.
PG (RDS Multi-AZ, the common "real prod" default): 14.8 KB/memory, ×2 copies, ×1.5, × $0.115/GB.

| memories | logical vectors | **memlake $/mo** | **PG RDS Multi-AZ $/mo** | PG Aurora $/mo | PG MultiAZ+2RR $/mo |
|---:|---:|---:|---:|---:|---:|
| 1 M | ~6.9 GB | **$0.16** | $5.1 | $2.2 | $10.2 |
| 10 M | ~69 GB | **$1.6** | $51 | $22 | $102 |
| 100 M | ~690 GB | **$16** | $510 | $221 | $1,020 |
| 1 B | ~6.9 TB | **$162** | $5,100 | $2,210 | $10,200 |
| 10 B | ~69 TB | **$1,590*** | $51,000 | $22,100 | $102,000 |

\* at 10 B, memlake crosses the 50 TB S3 tier into $0.022/GB — negligible. PG at 10 B on a
single cluster is not realistic without sharding, which adds its own multiplier; the number is
the naive linear extrapolation.

memlake storage COGS is **roughly 30× lower than the RDS Multi-AZ baseline** and ~14× lower than
Aurora, and the gap *widens* with scale because PG's per-copy block cost is linear while S3's
tiering bends down.

## Where Postgres is actually cheaper — do not skip this

1. **Small scale, existing instance.** Below ~10 M memories the *storage* line for both is a
   rounding error against compute. If Hindsight already runs a PG instance for everything else,
   the marginal storage for memories is ~free until you outgrow the instance. memlake's $0.16/mo
   at 1 M is technically ~10× cheaper than PG's $5 — but $5 is noise. **The storage argument only
   matters at scale.**

2. **Request costs are memlake's hidden COGS.** S3 bills per request; PG does not. A memlake
   query issues a handful of GETs (probe waves + rerank point-fetches). At `$0.0004/1k` GET:
   - Say ~8 GETs/query (nprobe blocks + rerank scatter). 100 queries/sec sustained =
     ~2.6 M queries/mo × 8 = ~21 M GETs = **~$8.3/mo**. Modest.
   - But at **10,000 queries/sec** it is ~$830/mo in GETs alone — which at 10 M memories
     *exceeds the entire storage line for either system*. High-QPS-over-small-corpus is the
     regime where S3 request COGS can make memlake more expensive overall, and the two-tier NVMe
     cache exists precisely to keep those GETs off S3 (a warm hit is $0). Whether that holds is a
     cache-hit-rate question, not a storage-price one.
   - Ingest PUTs: each fold writes a bounded set of objects; amortised this is small, but a
     write-heavy workload with frequent flushes pays `$0.005/1k` PUT per object written.

3. **PG's storage buys latency memlake pays for elsewhere.** A PG vector query hits local
   NVMe/RAM; memlake's cold query is object-storage round trips (bounded, but ~tens of ms each).
   memlake trades storage cost for query latency and a caching layer. That is a COGS transfer,
   not a pure saving.

## Honest summary

> Superseded for decisions by [Measured total COGS](#measured-total-cogs-storage--requests--compute).
> In one line: against a realistic **Multi-AZ + 2-replica** PG, memlake is cheaper at every scale
> except small-corpus-high-QPS, and is **48–358× cheaper at 10 M**. The bullets below are the
> storage-only view.

- **Per useful GB, memlake storage is ~6–30× cheaper** depending on the PG replication topology,
  driven by S3's medium price and free internal replication vs PG's per-copy block storage.
- **Per memory, memlake stores ~half of one PG copy** (no ANN-index vector duplication, no
  links/entities tables) — and *could* store far less by dropping the full-precision rerank tier,
  at a stated recall cost.
- **Combined, storage COGS is ~15–30× lower at scale** (100 M+ memories), and the gap widens.
- **But storage is the wrong line to optimise below ~10 M memories**, where compute dominates and
  a PG instance's spare storage is effectively free.
- **memlake moves cost from storage to S3 requests and a cache tier.** At high QPS the request
  line is real and can invert the total-COGS comparison; that is a cache-hit-rate problem, and it
  is the number to watch before claiming an end-to-end win.

## What this leaves out (would change the total COGS)

- **Compute.** PG instance (right-sized for the vector index in RAM — pgvector wants the index
  resident, which at 1 B vectors is enormous) vs memlake query nodes (stateless, bounded RAM by
  construction, scale to zero between queries). This likely *favours memlake more* at scale (PG
  must hold the ANN index in RAM; memlake does not), but it is a compute analysis, not priced here.
- **Backup/snapshot storage** (PG: RDS snapshots to S3, billed; memlake: the bucket *is* the
  durable copy, no separate backup line).
- **Data transfer / egress**, **S3 lifecycle transitions** to IA/Glacier for cold memories
  (a lever memlake has and PG largely does not), **cross-region DR**.
- **Operational cost** (PG tuning, vacuum, failover drills vs memlake's stateless pods) — real,
  unpriced, and generally favours the object-storage model.
