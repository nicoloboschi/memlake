/**
 * UI copy for `memlake.v1.ObjectKind`: what each stored object actually is and
 * why the engine writes it.
 *
 * This is prose, not data. It is kept in one module — rather than sprinkled
 * through the view — because the same sentence has to serve the table legend
 * and the decode panel, and because it has to be checked against the storage
 * layout (SPEC §3) and the per-arm docs when the formats move.
 *
 * Client-safe, pure, no dependencies beyond the enum names themselves.
 */

import { STORAGE_OBJECT_KINDS, type StorageObjectKind } from "./types";

export interface ObjectKindCopy {
  /** The file as an operator would name it — usually its key suffix. */
  label: string;
  /** One or two sentences: what it holds, and what it buys at read time. */
  blurb: string;
}

export const OBJECT_KIND_COPY: Record<StorageObjectKind, ObjectKindCopy> = {
  MANIFEST: {
    label: "manifest.json",
    blurb:
      "The single mutable object. Every other file is immutable, so publishing means writing new files then CAS-swapping this one — a reader sees a whole generation or none of it. It names the current stack of segments (an LSM: L0 flushes over older, larger levels), the previous stack (a GC grace window), and the WAL cursor. A flush usually adds one segment and reuses the rest by reference, so most files it points at are unchanged from the last generation.",
  },
  WAL_ENTRY: {
    label: "wal entry",
    blurb:
      "One group-commit batch, PUT with If-None-Match so a sequence can only ever be claimed once; everything inside one entry is all-or-nothing to every reader. Once the indexer folds it into a generation (seq ≤ wal_index_cursor) nothing reads it again and it waits for GC.",
  },
  WAL_HEAD: {
    label: "wal-head",
    blurb:
      "The cached WAL head pointer: a tiny object holding the highest sequence any writer has committed, as a decimal number. Readers GET it (etag-cacheable) to learn the live head without a LIST — on S3 both slower and ~12× the per-request price. A writer bumps it after durably appending its entry, and it only ever increases. Purely an accelerator: if it is missing or lags a crashed writer's un-acked entry, readers fall back to the authoritative LIST.",
  },
  INDEX_LEASE: {
    label: "index-lease.json",
    blurb:
      "A soft, best-effort lease that lets one node announce it is currently folding this namespace, so the others' periodic indexers skip it and don't duplicate the compute and S3 PUTs. Only an optimization — losing or ignoring it costs a wasted but safe duplicate fold, never correctness, since the randomly-suffixed segment prefixes already make concurrent folds safe.",
  },
  CLUSTER: {
    label: "cluster-{i}.bin",
    blurb:
      "An IVF cluster's memory RECORDS — everything but the embedding, which is split out into the sibling cluster-{i}.vec. The memories assigned to one centroid, sized so one probe hydrates in a single coalesced ranged GET, and their inline outgoing links give the graph arm its seed adjacency for free. A segment's clusters are written once and never rewritten; a flush carries the older segments' clusters forward by reference, so a live cluster commonly sits under an older segment's prefix.",
  },
  VECTOR_BLOCK: {
    label: "cluster-{i}.vec",
    blurb:
      "The quantized vector block for one cluster: the RaBitQ codes (1-bit per dimension) plus norms and the tag/updated_at columns, split out of the record file because the embedding is ~84% of a memory's bytes. The vector arm scans this block to rank candidates by an error-bounded estimate, then reranks only the survivors against the exact f32 in rerank.data — so a probe reads a small block, not the whole record.",
  },
  CENTROIDS: {
    label: "centroids.json",
    blurb:
      "The trained k-means centroid table for one memory_type (k ≈ √N). Small, loaded whole at snapshot open and kept resident, so choosing which nprobe clusters to read is pure CPU — zero roundtrips before the fetch.",
  },
  PK_INDEX: {
    label: "pk.idx",
    blurb:
      "The sparse half of the primary-key SSTable (id → cluster): every Kth key with its byte offset. Small enough to load whole and keep cached, so a lookup is an in-memory binary search that names the one block to range-read.",
  },
  PK_DATA: {
    label: "pk.data",
    blurb:
      "The primary-key SSTable's blocks: MemoryId → cluster index, sorted. Finding a memory's cluster costs one cached index lookup plus one ranged GET of a single block rather than a full read. The graph arm is its main reader — expansion yields ids, pk turns them into the clusters to fetch.",
  },
  RADJ_INDEX: {
    label: "radj.idx",
    blurb:
      "The sparse index over radj.data: every Kth target id → byte offset, sized to stay small enough to hold in memory. Walking edges backwards is a binary search here plus one coalesced ranged GET.",
  },
  RADJ_DATA: {
    label: "radj.data",
    blurb:
      "Reverse adjacency: target → its incoming semantic and causal edges, sorted by target. Forward edges ride inline in the cluster files, so this file is what makes the backward direction — who links to this seed — one bounded read instead of a scan of the corpus.",
  },
  ENTITY_INDEX: {
    label: "entity.idx",
    blurb:
      "The sparse index over entity.data, loaded whole. The same index/data split as pk and radj: binary-search in memory, then exactly one ranged GET of the postings block.",
  },
  ENTITY_DATA: {
    label: "entity.data",
    blurb:
      "Entity postings: EntityId → sorted [MemoryId], every memory carrying each entity. This is what makes the graph arm's entity relation real — without it the arm could only reconnect memories the vector probe had already fetched; with it, one bounded ranged read finds entity-sharers anywhere in the corpus.",
  },
  TIME_INDEX: {
    label: "time.idx",
    blurb:
      "The sparse index over time.data. Small, loaded whole, and what lets a [from, to] window resolve to a single bounded ranged scan.",
  },
  TIME_DATA: {
    label: "time.data",
    blurb:
      "The time index: effective timestamp → [MemoryId], where effective_ts = COALESCE(occurred_start, mentioned_at, occurred_end). The key is an order-preserving big-endian encoding of the i64 (sign bit flipped), so raw byte order is numeric order and a window is one contiguous range — no sorting at query time. Memories with no timestamp are simply not indexed here.",
  },
  PAYLOAD_INDEX: {
    label: "payload.idx",
    blurb:
      "The sparse half of the payload SSTable, loaded whole; a hit's id binary-searches here down to the single block of payload.data that holds it.",
  },
  PAYLOAD_DATA: {
    label: "payload.data",
    blurb:
      "The payload store: MemoryId → the memory's bytes, in SSTable blocks. Hydrating a search hit is one ranged GET of one block, instead of pulling the whole multi-megabyte cluster file the memory happens to live in.",
  },
  RERANK_INDEX: {
    label: "rerank.idx",
    blurb:
      "The sparse index over rerank.data — the same in-memory-binary-search-then-one-ranged-GET split as the other SSTable pairs. It maps a candidate id to the block holding its exact f32 embedding.",
  },
  RERANK_DATA: {
    label: "rerank.data",
    blurb:
      "MemoryId → the exact float32 embedding, in SSTable blocks. The vector arm's first stage ranks candidates from the quantized cluster-{i}.vec; this file is what the second stage reads to rescore the survivors at full precision, so the quantization only ever changes which candidates are considered, never the final score.",
  },
  FTS_SPLIT: {
    label: "fts/split.bin",
    blurb:
      "The whole tantivy index for one memory_type packed into a single object — a split, with a hotcache footer — so BM25 is published write-once with the generation instead of as tantivy's many small segment files. It is materialized once at snapshot open and searched locally, so the text arm does zero object-storage roundtrips per query. memlake treats the bytes as opaque and hands them to tantivy, which is why this is the one kind DecodeObject cannot open.",
  },
  STATS: {
    label: "stats.json",
    blurb:
      "The generation's own counters — doc_count, cluster_count, edge_count — written by the indexer so Stats can answer without reading the index itself. Tiny, read at open, never on the query path.",
  },
  TAG_SUMMARY: {
    label: "tags.json",
    blurb:
      "Per cluster: the union of its memories' tags, plus whether it holds untagged ones. Read at open so a tag-filtered query can prune clusters that cannot contain a match before probing — the filter is applied before the fetch, not after it.",
  },
  SEGMENT_TOMBSTONES: {
    label: "tombstones.json",
    blurb:
      "A segment's delete overlay: the ids it supersedes in OLDER segments, so a re-upsert or a delete hides the stale copy without rewriting the segment that holds it. Written at the segment level (not per memory_type). This is how an append-only LSM deletes — the overlay is consulted at read time and the shadowed rows are physically reclaimed only when a compaction rewrites the segments they live in.",
  },
  OBJECT_KIND_UNKNOWN: {
    label: "unknown",
    blurb:
      "A key matching none of the layouts memlake writes. Not necessarily wrong — an indexer lease, a leftover from an older format version, or something another tool left under the prefix — but memlake cannot say what it is, and therefore cannot decode it either.",
  },
};

/**
 * Copy for a kind name off the wire. A server ahead of this build can send an
 * enum member we have no prose for; fall back to the unknown copy rather than
 * rendering `undefined`.
 */
export function kindCopy(kind: string): ObjectKindCopy {
  return (
    OBJECT_KIND_COPY[kind as StorageObjectKind] ??
    OBJECT_KIND_COPY.OBJECT_KIND_UNKNOWN
  );
}

/**
 * A stable display order for kinds: the namespace-level objects first, then a
 * generation's files roughly in the order a cold query touches them, with the
 * unclassified last. Used for the legend and for the secondary sort inside a
 * generation.
 */
export function kindRank(kind: string): number {
  const i = (STORAGE_OBJECT_KINDS as readonly string[]).indexOf(kind);
  return i < 0 ? STORAGE_OBJECT_KINDS.length : i;
}
