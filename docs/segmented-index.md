# Segmented generations: making the fold O(tail), not O(corpus)

Status: **design** (not yet implemented). Supersedes the single-monolithic-generation model in
[SPEC.md](SPEC.md) §3/§5. Prior art: [turbopuffer](https://turbopuffer.com/docs/architecture),
Lucene/Elasticsearch segments, and [SPFresh (SOSP '23)](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/08/SPFresh_SOSP.pdf).

## 1. The problem

A "generation" today is **one monolithic set of files** (`centroids.bin`, `cluster-*`, and the
`pk/radj/entity/time/payload/rerank` SSTables + FTS split), and a **fold rebuilds it from the entire
live set on every run**. Clusters copy-forward if unchanged, but the secondary-index layer — every
SSTable + the FTS split + tag summaries — is regenerated over *all* items each fold, because those
are single globally-sorted files you can't append into.

So **fold cost is O(corpus), not O(new writes).** Folding 100 new memories into a 100M-doc namespace
rebuilds 100M-entry SSTables and re-indexes 100M docs in tantivy. This is exactly the antipattern
SPFresh names:

> *"Existing systems maintain a secondary index to accumulate updates, which are merged with the main
> index by globally rebuilding the entire index periodically — high fluctuations of search latency and
> accuracy, substantial resources, extremely time-consuming to rebuild."*

memlake's WAL tail is that "secondary index"; the fold is that "periodic global rebuild." It also
forces the whole in-RAM-vs-streaming fold dichotomy, because a full rebuild must hold (or stream) the
whole corpus.

## 2. Approach: LSM-style immutable segments + tiered compaction

SPFresh's *in-place* mutation needs mutable storage; we are immutable object storage (like
turbopuffer), so the fitting model is **LSM segments** (Lucene / turbopuffer's storage layer):

- A write flush becomes a **new, small, immutable segment** carrying its **own complete mini-index**.
  Existing segments are never touched. This is O(tail).
- A query **fans each arm out across all segments** (+ the still-unindexed WAL tail) and **merges**.
- **Deletes/updates** are an overlay resolved at read time (a newer segment shadows an older one);
  segments aren't rewritten.
- A background **tiered compaction** merges small segments into larger ones to bound the segment
  count. This is the *only* O(merged-size) step, amortized and off the write path.

Net: flush latency and fold frequency become independent of corpus size; O(N) work happens only at
compaction.

## 3. Storage layout

A generation becomes an **ordered stack of segments across levels** (L0 = newest/smallest). Each
segment is a self-contained mini-generation over its own slice of items:

```
s3://{bucket}/{namespace}/
  manifest.json
  wal/00000001.bin …
  seg/{seg_id}/                     # seg_id = uuid; immutable, write-once
    centroids.bin                   # this segment's IVF centroids (√n_seg)
    cluster-{i}.bin  cluster-{i}.vec
    pk.idx  pk.data                 # id -> cluster, over this segment only
    radj.idx radj.data              # incoming edges whose TARGET is in this segment
    entity.idx entity.data
    time.idx  time.data
    payload.idx payload.data
    rerank.idx rerank.data
    fts/split.bin
    tags.json  stats.json
    tombstones.bin                  # ids + predicate-deletes this segment applies (see §6)
```

`seg_id` is a content-nonce (as generation prefixes are today), so segments are immutable and
GC-able. There is no `gen-{G}/` monolith anymore; the manifest is the list of live segments.

### Manifest shape

```jsonc
{
  "format_version": 4,
  "manifest_seq": 88,              // monotonic; CAS identity
  "wal_index_cursor": 137,         // last WAL seq flushed into a segment
  "wal_head": 141,
  "segments": [                    // ordered newest-first within a level
    { "id": "…", "level": 0, "seq_lo": 130, "seq_hi": 137, "doc_count": 40,
      "memory_types": [1,2], "files": { "1": { "pk": "seg/…/pk.idx", … }, "2": { … } } },
    { "id": "…", "level": 1, "seq_lo": 64,  "seq_hi": 129, "doc_count": 9000, … },
    { "id": "…", "level": 2, "seq_lo": 0,   "seq_hi": 63,  "doc_count": 990000, … }
  ],
  "tokenizer_config_hash": "…"
}
```

Still one `manifest.json`, still CAS-swapped (`If-Match`), still the single mutable pointer (INV-2).
A flush or a compaction produces a **new segment set** and swaps the manifest; losers re-read + merge.

## 4. Flush (the new "fold") — O(tail)

`flush(namespace)`:
1. Read manifest + WAL entries `(wal_index_cursor, wal_head]`.
2. Resolve that slice's live items (last-write-wins within the slice) — **only the tail**, never the
   prior segments.
3. Build **one new L0 segment** over those items: train `√n_seg` centroids on the slice, assign,
   write its cluster files + its own SSTables + FTS split, and a `tombstones.bin` for the deletes /
   predicate-deletes in the slice.
4. CAS-swap the manifest: prepend the new segment, advance `wal_index_cursor = wal_head`.

Cost is O(slice), independent of corpus. The in-RAM fold trivially handles a small L0 flush; the
streaming (external-memory) fold is now only needed for large **compactions** (§7), which is where it
belongs — the in-RAM-vs-streaming choice moves off the write path entirely.

Semantic-link derivation for the flush slice runs against the *current* index (probe the existing
segments for each new item's neighbours) and writes `semantic_out` into the new segment — incremental
by construction, no global pass (this is what SPEC §5.2 already intended).

## 5. Query: fan out across segments, merge

Every arm reads across **all segments + the WAL tail** and merges. The merge rules per arm:

- **Vector** — probe each segment's centroids, fetch the selected clusters, collect candidates from
  every segment, then rerank globally by exact score and take top-k. `nprobe` is budgeted across
  segments (e.g. proportional to each segment's cluster count) so total clusters fetched stays
  bounded.
- **FTS** — query each segment's tantivy split; merge hits. BM25 needs corpus-global term stats, so
  merging N per-segment BM25 scores is approximate; carry per-segment `df`/`doclen` in `stats.json`
  and combine (Lucene solves the same problem with `CollectionStatistics` across segments). Open
  question §9.
- **pk / payload / entity / time** — look up each segment's SSTable; for id-keyed lookups (pk,
  payload) the **newest segment that contains the id wins** (last-writer-wins by `seq_hi`), so a
  re-upsert in L0 shadows the copy in L2.
- **graph (radj)** — an edge's target may live in a different segment than its source. Store each
  incoming edge in the segment that owns its **target** id; graph expansion queries every segment's
  `radj` for a seed set and unions. Cross-segment edges are fine because ids are global.
- **tombstones** — load each segment's `tombstones.bin` (small, cached). An item from segment S is
  hidden if a **newer** segment tombstones its id, or a predicate-delete with `seq > item.write_seq`
  matches it. This is the `liveDocs`-bitset overlay, resolved at read time across segments.

### Roundtrip budget (INV-7)

Cost is now O(#segments) instead of O(1). Tiered compaction bounds `#segments` to ≈
`fanout × levels` ≈ `fanout × log_fanout(N)` — a small constant (e.g. ≤ ~30 at any realistic scale).
The hard per-query roundtrip budget becomes **O(levels)**, and the budget metric/alert (SPEC §6.1)
tracks it. This is the LSM read-amplification tax, paid to make writes cheap; compaction is the knob
that trades it back.

## 6. Deletes, updates, and correctness

The merged view must equal today's `fold_entries` semantics (last-write-wins / tombstone / patch /
predicate-delete):

- **Upsert of an existing id** → lands in a new L0 segment; its `seq` is newer, so pk/payload
  last-writer-wins makes it shadow the old copy in every arm. The old copy is physically dropped at
  compaction.
- **Tombstone / predicate-delete** → recorded in the flushing segment's `tombstones.bin` and applied
  as the cross-segment overlay above. Compaction materializes it (the shadowed item is not carried
  forward), after which the tombstone can be dropped once no older segment can still surface the id.
- **Patch** → folded into the item when it is (re)written into a segment; a patch against an id that
  only exists in an older segment is carried in the new segment as a patch record until compaction
  merges them (or, simpler v1: a patch re-materializes the full item into L0).

## 7. Compaction — the only O(N) step

Tiered, Lucene/turbopuffer-style: when level `Lk` accumulates ≥ `FANOUT` segments (e.g. 8), merge
them into **one** segment at `L{k+1}`.

- Merge = resolve last-writer-wins + tombstones across the input segments, then build the output
  segment's clusters + SSTables + FTS over the merged live set. This is where centroids are retrained
  (or LIRE-style split/merged) over a meaningful population, and where deleted/shadowed items are
  physically reclaimed.
- A large merge is exactly the workload the **streaming external-memory fold already solves** —
  bounded RAM, spill + external sort. So the streaming fold becomes the *compaction engine*, and the
  fast in-RAM path builds small L0 flushes.
- Idempotent + coordination-free (INV-6): any node may compact a set of segments; the manifest CAS
  serializes the publish; a lost race just means a peer produced an equivalent merged segment. GC
  removes segments no longer referenced by the manifest after the reader-grace TTL.

Compaction runs on the indexer deployment on its own cadence (size/count triggers), fully decoupled
from write latency.

## 8. What this buys us

- **Flush is O(tail)** → fold frequency is independent of corpus size; a namespace can fold every few
  seconds at 100M without rebuilding 100M-entry indexes.
- **The in-RAM vs streaming fold dichotomy collapses**: L0 flushes are always small (in-RAM), and the
  only O(N) work — compaction — uses the bounded streaming fold. No silent size-threshold switch, no
  quality cliff. This is the clean resolution of the `MEMLAKE_INDEXER_STREAMING_THRESHOLD` problem.
- **Incremental semantic links** fall out naturally (derive per-flush against the current index),
  which the current design can't do at scale.
- Reuses memlake's existing machinery almost wholesale: immutable prefixes, per-cluster files,
  SSTable `.idx`/`.data`, WAL-tail exhaustive scan, manifest CAS, GC.

## 9. Open questions (resolve before/while implementing)

1. **BM25 across segments** — per-segment `df`/`doclen` merge vs a lazily-maintained global stats
   object. How much recall/scoring drift is acceptable before the first compaction?
2. **`nprobe` budgeting across segments** — split evenly, by cluster count, or by segment recency?
   Recent (L0) segments are small but hot.
3. **Segment fan-out / level count** vs the roundtrip budget — pick `FANOUT` and per-query segment cap
   so INV-7 stays a small constant; measure read amplification on `mem-1m`/`mem-10m`.
4. **Centroids at compaction** — full retrain on the merged slice (simple, deterministic) vs LIRE-style
   boundary reassignment (cheaper, closer to SPFresh). Start with retrain.
5. **Patch representation across segments** — carry patch records in L0 vs re-materialize the full item
   on patch. Re-materialize is simpler; measure the write amplification.
6. **Cross-segment graph radj** — confirm edge-in-target-segment placement keeps graph expansion within
   the roundtrip budget when a seed's neighbours are spread across levels.

## 10. Phasing

1. **Manifest v4 + segment list** — a generation is a list of segments (initially 1). Query/GC read the
   list; behaviour identical to today with a single segment.
2. **Flush = new L0 segment** (O(tail)); stop rebuilding prior segments.
3. **Per-arm cross-segment merge** in the query node (vector/fts/pk/payload/entity/time/graph +
   tombstone overlay). This is the bulk of the work.
4. **Tiered compaction** (streaming fold as the merge engine) + segment GC.
5. Delete the `MEMLAKE_INDEXER_STREAMING_THRESHOLD` selector; flush=in-RAM, compaction=streaming.

Each phase ships behind the existing benchmark/correctness gates (§10 of SPEC): the merged multi-segment
query must match the single-segment result on `mem-10k` + `hindsight-diff` before compaction lands.
