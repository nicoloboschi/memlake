# Storage COGS: memlake vs Postgres, for Hindsight

Napkin math. The goal is to capture the *real* cost of goods sold at storage, not to make
memlake win — where Postgres is cheaper (and at small scale it is), that is stated plainly.
Every number here is order-of-magnitude and carries its assumptions; treat it as a model to
argue with, not a quote.

Scope: **storage only.** Compute (the query nodes / the PG instance CPU+RAM) is a separate,
often larger line and is discussed but not priced — see "What this leaves out". Request costs
(S3 GET/PUT) *are* storage-adjacent COGS and are called out, because for memlake they are real
and for PG they are ~zero.

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
(bge-small, via `ListObjects`/`DecodeObject` — see `docs/vector-storage.md`) and scaled to
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
