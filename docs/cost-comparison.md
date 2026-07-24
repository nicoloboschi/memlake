# COGS: memlake vs Postgres, for Hindsight

The goal is to capture the *real* cost of goods sold, not to make memlake win — **at 10 k
memories Postgres is cheaper, and at high QPS it is cheaper at 100 k too.** Both are stated
plainly below.

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
QPS). PG = one RDS instance sized so vectors **and** the ANN index copy stay RAM-resident, + gp3.

| memories | QPS | ML storage | ML S3 req | ML compute | **memlake** | PG instance | **PG total** | cheaper |
|---:|---:|---:|---:|---:|---:|---|---:|---|
| 10 k | 1 | $0.00 | $1.05 | $25 | **$26** | db.t4g.micro | **$14** | **PG 1.8×** |
| 10 k | 10 | $0.00 | $10.51 | $25 | **$35** | db.t4g.micro | **$14** | **PG 2.5×** |
| 10 k | 100 | $0.00 | $105 | $112 | **$218** | db.t4g.micro | **$14** | **PG 15.6×** |
| 100 k | 1 | $0.02 | $1.07 | $25 | **$26** | db.t4g.medium | $50 | memlake 1.9× |
| 100 k | 10 | $0.02 | $10.53 | $25 | **$35** | db.t4g.medium | $50 | memlake 1.4× |
| 100 k | 100 | $0.02 | $105 | $112 | $218 | db.t4g.medium | **$50** | **PG 4.4×** |
| 1 M | 1 | $0.25 | $1.22 | $25 | **$26** | db.r6g.xlarge | $332 | memlake 12.8× |
| 1 M | 10 | $0.25 | $10.68 | $25 | **$35** | db.r6g.xlarge | $332 | memlake 9.4× |
| 1 M | 100 | $0.25 | $105 | $112 | **$218** | db.r6g.xlarge | $332 | memlake 1.5× |
| 10 M | 1 | $2.46 | $2.75 | $25 | **$30** | db.r6g.8xlarge | $2 664 | memlake 90× |
| 10 M | 10 | $2.46 | $12.21 | $25 | **$39** | db.r6g.8xlarge | $2 664 | memlake 68× |
| 10 M | 100 | $2.46 | $107 | $112 | **$222** | db.r6g.8xlarge | $2 664 | memlake 12× |

### There are two crossovers, and both matter

**1. Scale: PG wins below ~50–100 k memories.** memlake's floor is an always-on box (~$25/mo) plus
per-query requests; PG's floor at 10 k is a `db.t4g.micro` at ~$14/mo, and 10 k vectors fit in its
RAM trivially. **Storage is irrelevant on both sides at this scale** — $0.00 vs $2.30. Anyone
arguing memlake on cost at 10 k is arguing the wrong thing; the argument there is operational
(stateless pods, no vacuum/failover), not COGS.

**2. QPS: the S3 request line flips it back to PG.** At 100 QPS the uncacheable head GET alone is
**$105/mo** — more than memlake's compute, and it grows strictly linearly with QPS while PG's
instance cost is flat until the box saturates. That is why 100 k @ 100 QPS goes back to PG by 4.4×.
memlake's cost curve is *per-query*; PG's is *per-resident-byte*. Which one wins is entirely a
question of which axis your workload is long on.

**Where memlake wins decisively: many memories, moderate QPS.** At 10 M the PG instance must hold
~139 GB of vectors + index in RAM (a `db.r6g.8xlarge`, $2 664/mo) while memlake holds the corpus in
S3 at $2.46/mo and pays only for the working set it touches. That is the actual shape of the
argument — not "S3 is cheaper per GB".

### The highest-leverage cost fix is the head GET

At every scale above, memlake's marginal cost per query is *one S3 GET*. Coalescing **concurrent**
in-flight head reads onto a single GET would cut that line by the in-flight factor at no consistency
cost (simultaneous readers may legitimately share one head observation). A short TTL would cut it
further but does weaken strong consistency, so it is a product decision rather than a free win. At
100 QPS a 10× coalescing factor turns $105/mo into ~$10/mo and moves the 100 k @ 100 QPS row from
"PG 4.4× cheaper" to roughly parity. **This is the single number to attack before quoting memlake
as cheap at QPS.**

### What this table does NOT model

- **memlake at 1 fact type.** All measurements are single-type; the design's stated worst case is
  `FANOUT = 3`, which triples query CPU (so the compute column, not the request column).
- **No scale-to-zero.** memlake's serve is stateless and *could* scale to zero between queries,
  which would erase the $25 floor and win the 10 k row outright. Not implemented/measured, so not
  credited.
- **PG's index build cost and vacuum/bloat headroom** are only in the ×1.5 storage factor, and PG's
  ANN recall at these instance sizes is assumed adequate, not measured. If the index does *not* stay
  RAM-resident, PG gets much cheaper and much slower — a latency/COGS trade this table does not price.
- **Multi-AZ / replicas.** PG here is **single-AZ, no replica** — the cheapest honest configuration.
  Real prod PG is usually Multi-AZ (≈2×), which roughly doubles every PG number above and moves the
  10 k crossover close to parity.
- **Egress, backups, ops labour** — see "What this leaves out" at the end.

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
> In one line: **PG is cheaper below ~50–100 k memories and at high QPS; memlake wins from ~1 M up at
> moderate QPS, by ~10–90×.** The bullets below are the storage-only view.

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
