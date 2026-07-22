# Vector storage: making the embedding smaller

A plan for reducing what memlake stores and fetches per vector. Today every embedding is
raw `f32`, which is the single largest thing in the system by a wide margin.

Everything in "Where we are" is measured against the `demo` namespace (629 memories,
384-dim bge-small) via `ListObjects`/`DecodeObject`, not estimated.

## Where we are

```
live cluster files: 32     members: 629     bytes: 1,153,496
1834 B per stored memory
f32 embedding (384 x 4) = 1536 B  ->  84% of every stored memory is the embedding
```

Four facts that shape everything below:

1. **The embedding is 84% of a stored memory.** The other ~298 B is text, tags, metadata
   and edges. Any storage or bandwidth conversation is a conversation about the vector.

2. **The query path fetches whole cluster files.** `nprobe` clusters are read entire, then
   `exact_search` re-ranks with full-precision cosine over everything fetched. Bytes per
   query are therefore `nprobe × cluster_size` — and 84% of those bytes are embeddings the
   scan reads once and discards.

3. **Centroids are serialized as JSON** (`Centroids::to_bytes` → `serde_json`). A `f32`
   costs 4 bytes raw and roughly 12–20 as decimal text. The centroid table is read on every
   snapshot open and kept resident, so this is pure overhead on the open path.

4. **Payload is already being split out.** `payload.idx`/`payload.data` store
   "memory bytes without the embedding" for point hydration — but `ClusterFile` still holds
   `Vec<StoredMemory>` complete with vectors *and* text. So text/metadata currently exists
   in both places, and the cluster scan still drags the full embedding through.

## What the field does

The convergence here is striking: **everyone quantizes the scan and re-ranks a small
candidate set at higher precision.** Nobody stores raw `f32` as the scan representation.

| System | Index family | Codec | Notes |
|---|---|---|---|
| [turbopuffer](https://turbopuffer.com/docs/architecture) | SPFresh (centroid-based) | RaBitQ | Same index family as our IVF; chose centroids over HNSW/DiskANN precisely to minimise object-storage roundtrips and write amplification |
| [Elastic BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch) | HNSW / brute force | RaBitQ-derived, 1-bit | 32× + ~14 B corrective data per vector; 20–30× faster to quantize than PQ |
| [Qdrant](https://qdrant.tech/articles/binary-quantization/) | HNSW | SQ8, BQ, TurboQuant | Tradeoff chosen at **search** time, not index time; rescores top candidates with originals |
| [Milvus](https://milvus.io/docs/ivf-rabitq.md) | IVF_RABITQ | RaBitQ | IVF + RaBitQ is a shipped, named index type — the closest match to our architecture |
| [LanceDB](https://www.lancedb.com/blog/feature-rabitq-quantization) | IVF_PQ → RaBitQ | RaBitQ | |

Numbers worth anchoring on:

- **RaBitQ compresses 32×** with a provable error bound, and because it also estimates the
  error of each quantized score it needs *fewer than half* the rerank comparisons of other
  methods at equal recall.
- **Elastic measured Recall@10 = 0.994** against a full-precision float32 baseline with
  BBQ on Jina v5 embeddings — a 29× memory cut.
- **int8 costs ~2% recall** versus float32 in Elastic's comparison, and is *faster* than
  float32 because of native int8 SIMD.
- Turbopuffer's framing of the underlying economics: 1 KB of text expands to ~16 KB of
  vector after chunking and embedding. The vector is the cost centre, not the corpus.

Note that turbopuffer's public architecture page does **not** disclose its compression
ratio, bytes-per-vector, or whether it keeps full-precision copies — the RaBitQ attribution
comes from secondary write-ups, not their docs. Worth treating as directional.

## The two levers are independent

This is the part most easily conflated, and for memlake the second matters more.

**Lever 1 — the codec.** How many bytes one vector occupies.

**Lever 2 — the layout.** How many of those bytes the scan is forced to read.

Applying only the codec, leaving cluster files as they are:

| Codec | Bytes/vector | Bytes/memory | Cluster files shrink |
|---|---|---|---|
| `f32` (today) | 1536 | 1834 | — |
| `fp16` | 768 | 1066 | 1.7× |
| `int8` | 384 | 682 | 2.7× |
| 1-bit + corrective | 62 | 360 | **5.1×** |

Even 1-bit only buys 5.1×, because the ~298 B of text and metadata rides along in the same
object. But if the scan reads **only codes** — vectors split into their own block — the scan
reads 62 B instead of 1834 B per candidate: **~30×**. The codec caps out at 5×; the codec
*plus* the layout change is where the order of magnitude lives.

That layout change is also the natural completion of the payload-store work already in
flight: payload is being pulled out for point reads, and this pulls vectors out for scans,
leaving the cluster file as what it should be — a scan-optimised column, not a row store.

## Plan

Ordered by (payoff / risk). Each phase is independently shippable and independently
revertable, and each has a measurable gate before the next starts.

### Phase 0 — stop storing centroids as JSON

Swap `Centroids::to_bytes` from `serde_json` to the same raw little-endian `f32` encoding
the wire already uses. Small, self-contained, **zero recall risk** — it is a pure encoding
change to a file nothing else parses.

*Gate:* existing tests pass; centroid object size drops ~3–5×.

### Phase 1 — split vectors out of the cluster file

Give each cluster two objects rather than one:

- `cluster-{i}.vec` — the vectors alone, contiguous, in member order. The scan target.
- `cluster-{i}.bin` — everything else, in the same order, addressed by the same index.

Still `f32` at this stage: **this phase changes no numbers, only what the scan must read.**
Shipping it separately keeps the layout change and the codec change from being tangled in
one hard-to-bisect diff. It also lets a probe read vectors without paying for text, which is
a latency win on its own before any quantization.

*Gate:* G-1 recall unchanged (it must be — the maths is identical); bytes-fetched per query
drops ~6×; BEIR nDCG@10 unchanged.

*Watch:* this touches the "payload is inline, so seed adjacency is free" property in SPEC
§6 — the graph arm gets its outgoing links from the cluster fetch. Keep `.bin` on the same
fetch wave so that stays true.

### Phase 2 — int8 scalar quantization

Per-vector scale and offset, `f32 → i8`. 4× on the codec, ~2% recall cost before reranking
and recoverable by reranking the top candidates from a full-precision block.

Chosen before binary deliberately: it is a few dozen lines, has an obvious inverse, and
proves out the *rerank plumbing* (oversample → rescore) that Phase 3 depends on. If the
rerank path is wrong, it is much easier to see at 4× than at 32×.

bge embeddings are already L2-normalized, which is what makes a shared scale behave — worth
asserting rather than assuming, since `uniform_dim` already proves we cannot trust callers
to send what we expect.

*Gate:* G-1 holds (recall@10 ≥ 0.95 @ nprobe=8, ≥ 0.99 @ nprobe=32); BEIR nDCG@10 within
1% of the f32 baseline on the datasets we already run.

### Phase 3 — RaBitQ / 1-bit with reranking

The real prize: 32× on the codec, ~30× on bytes scanned. Store 1 bit per dimension plus the
small corrective term RaBitQ needs, scan with the estimated distance, then rerank the top
`k × oversample` candidates against full precision.

Two decisions this forces, and they are the substance of the phase:

1. **Where does full precision live?** Either a cold `vectors.f32` block range-read for the
   rerank set only, or nowhere at all if BEIR says the estimate suffices. Elastic's 0.994
   Recall@10 suggests "nowhere" is defensible; our own BEIR numbers should decide it, not
   someone else's benchmark on someone else's embeddings.
2. **Does the rerank cost a roundtrip?** INV-7 says query cost is a statically bounded
   number of roundtrips. A rerank read must ride the same wave as payload hydration — which
   already exists — or it breaks the invariant. This is the design constraint that most
   shapes the implementation.

*Gate:* G-1 holds at a defined oversampling factor; BEIR nDCG@10 within 1%; roundtrip count
per query unchanged (there is already a test asserting it is constant regardless of size).

### Phase 4 — optional, only if measured

- **Matryoshka truncation** — free 2× by keeping 256 of 384 dims, *but* only for models
  trained for it. bge-small-en-v1.5 is not, so this needs an embedding-model change and
  should not be assumed.
- **Asymmetric query encoding** — keep the query at higher precision than the stored codes
  (RaBitQ uses ~4 bits/dim for queries). Cheap recall for no storage.
- **Per-namespace codec choice**, the way Qdrant lets the tradeoff be picked at search time.

## Why we can trust the gates

Unusually for this kind of change, we do not have to guess:

- `crates/mlake-ivf/tests/recall.rs` is an existing **G-1 gate** measuring recall against
  brute force at fixed nprobe.
- `mlake-bench` produces **nDCG@10 / Recall@100 / MRR@10** on BEIR against identical
  embeddings, with exact-numpy and Qdrant baselines already wired up.

So each phase lands behind a number, and a regression is visible immediately rather than
six months later as "search feels worse". Any phase whose gate fails should be reverted
rather than tuned into passing.

## Recommendation

Phase 0 now — it is nearly free. Then Phase 1, because it is the structural unlock and
carries no recall risk, and its win (~6× on bytes fetched) is real on its own. Decide
Phase 2 vs jumping to Phase 3 on the strength of the Phase 1 numbers; the argument for
going through int8 first is de-risking the rerank path, not the compression itself.

The thing not to do is start with the codec while leaving the layout alone. That is the
version of this work that costs recall and returns 2.7×.

## The codec is per-block, and a codec change migrates incrementally

A `.vec` block is self-describing: its header carries the codec, so reads stay correct with
any mix of encodings in the index at once. That is what makes the codec safe to change — but
it is also what let a change *silently never take effect*. Cluster files copy forward by
reference when unchanged (`cluster_changed` is false, SCALE.md Phase 3), and copy-forward
reused the old `.vec` block verbatim. So flipping `IndexOptions::vector_codec` left every
untouched cluster in the old encoding indefinitely. This was observed directly: an
Int8-vs-Binary A/B returned byte-identical BEIR numbers because the second run was still
reading Binary blocks copied forward from the first.

**The fix.** `read_generation` records each cluster's stored codec from its block header
(`Generation::codecs`), and the copy-forward gate now also requires that codec to equal the
requested one. A cluster may copy forward only when it is genuinely unchanged *and* already in
the target codec; otherwise it is re-encoded. Copy-forward still fires in the common case (same
codec) — the object paths are reused, asserted by
`indexer::migration_tests::unchanged_codec_copies_clusters_forward`.

**One fold or incremental?** Incremental, and it already is. A codec change is *not* forced
through one enormous re-encode of the whole corpus. Each fold re-encodes only the clusters it
already visits: a `flush` writes its new L0 segment in the new codec immediately; a compaction
(`index`) re-encodes everything it merges. Old segments a compaction has not yet reached keep
their old-codec blocks and are read correctly (self-describing). The mixed state is explicit
(per-block codec in the header, per-cluster codec in `Generation`) and safe (reads already
tolerate it). The one honest gap: a completely static corpus never folds or compacts, so its
old-codec blocks never migrate until some write or a deliberate compaction triggers a rewrite.
A codec-mismatch compaction trigger would close that; it is left for the segmented-index work.

## What the rerank tier is (and is not) faithful to

From generation 2 onward, a compaction fold re-joins vectors by *decoding* the previous `.vec`
block (`read_generation` → `VectorBlock::decode`), and then re-encodes those decoded values —
both the scan block and the full-precision `rerank.data` tier are built from the same decoded
values in the same fold. Two consequences, both measured in
`vectors::tests::codecs_are_stable_under_repeated_decode_reencode`:

* **The guarantee we offer is internal consistency, not fidelity to the caller's embedding.**
  The query's exact f32 rescore (`rerank.data`) is consistent with what the scan block ranks —
  they are the same decoded values — but after a quantized generation, *neither* is the
  caller's original embedding. "Full precision" in the rerank tier means "full precision of the
  decoded value", not "the bytes the client sent".
* **The quantization loss is one-shot, not compounding.** The worry was that each fold quantizes
  an already-quantized vector, so error accumulates across generations. Measured over 8
  decode→re-encode generations at dim 384: **F32 is exactly byte-idempotent**; **Int8 and Binary
  are not** (Int8 is affine over the block-mean-centred residual, so a shifting mean and rescaled
  step move the codes by an ulp each fold — it is not a symmetric max-abs grid that would
  reproduce codes exactly), **but the drift is immaterial** — Int8's worst per-coordinate error
  grows ~2.5e-5/generation (four orders below anything the rerank can see) and both codecs'
  recall@10 at 4× oversampling against the *original* ranking holds flat across all 8 generations
  (Int8 1.000, Binary ≥ 0.99). The loss happens essentially once; it does not accumulate into a
  ranking change within any realistic number of folds.
