/**
 * proto-loader wire objects -> the JSON contract in `lib/types.ts`.
 *
 * SERVER ONLY (it handles Buffers). Two wire quirks are handled here and
 * nowhere else:
 *   - `bytes` fields are Node Buffers; ids become UUID strings, vectors become
 *     summaries.
 *   - 64-bit ints are decimal strings; they stay strings (BigInt for maths).
 */

import { bytesToUuid } from "./ids";
import { f32leToFloats, summarizeF32le } from "./vector";
import type {
  ArmScoreJson,
  CacheEntryJson,
  CacheKindSummary,
  CacheStatsJson,
  CausalEdgeJson,
  ClusterJson,
  ClusterMemberJson,
  DecodeObjectJson,
  HitJson,
  IndexLayoutJson,
  LinkType,
  ListObjectsJson,
  ListWalJson,
  MemoryPayloadJson,
  ObjectInfoJson,
  ObjectKind,
  ObjectKindSummaryJson,
  StatsJson,
  StorageObjectKind,
  StoredMemoryJson,
  TimestampsJson,
  TypeStatsJson,
  VectorSummary,
  WalEntryJson,
  WalOpJson,
} from "./types";
import { LINK_TYPES, STORAGE_OBJECT_KINDS } from "./types";
import type {
  CacheStatsResponse,
  DecodeObjectResponse,
  IndexLayoutResponse,
  ListObjectsResponse,
  ListWalResponse,
  StatsResponse,
  WireObjectInfo,
  WireArmScore,
  WireCacheEntry,
  WireCausalEdge,
  WireClusterInfo,
  WireClusterMember,
  WireHit,
  WireMemoryPayload,
  WireStoredMemoryRecord,
  WireTimestamps,
  WireVector,
  WireWalEntryInfo,
  WireWalOpDetail,
} from "./memlake";

/** Buffers are Uint8Arrays, but be defensive: proto-loader can hand back either. */
function asBytes(b: Buffer | Uint8Array | null | undefined): Uint8Array {
  if (!b) return new Uint8Array(0);
  return b instanceof Uint8Array ? b : new Uint8Array(b);
}

/** Coerce a proto-loader 64-bit field (string with `longs: String`) to a string. */
function u64(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "0";
  return typeof v === "string" ? v : String(v);
}

/** `a - b` on u64 decimal strings, clamped at 0. A WAL seq exceeds Number range. */
function diffU64(a: string, b: string): string {
  try {
    const d = BigInt(a) - BigInt(b);
    return (d > 0n ? d : 0n).toString();
  } catch {
    return "0";
  }
}

function linkType(v: string | number | null | undefined): LinkType {
  if (typeof v === "string" && (LINK_TYPES as readonly string[]).includes(v)) {
    return v as LinkType;
  }
  if (typeof v === "number" && LINK_TYPES[v]) return LINK_TYPES[v];
  return "CAUSES";
}

/**
 * proto3 `optional int64` becomes a synthetic oneof. With `oneofs: true` the
 * loader sets a `_field` marker naming the member that was actually present, so
 * "unset" and "explicitly zero" stay distinguishable.
 */
function optU64(
  present: boolean,
  v: string | number | null | undefined,
): string | null {
  if (!present) return null;
  if (v === null || v === undefined) return null;
  return u64(v);
}

export function timestampsToJson(
  t: WireTimestamps | null | undefined,
): TimestampsJson | null {
  if (!t) return null;
  const json: TimestampsJson = {
    eventDate: optU64(t._eventDate !== undefined, t.eventDate),
    occurredStart: optU64(t._occurredStart !== undefined, t.occurredStart),
    occurredEnd: optU64(t._occurredEnd !== undefined, t.occurredEnd),
    mentionedAt: optU64(t._mentionedAt !== undefined, t.mentionedAt),
  };
  const anySet =
    json.eventDate !== null ||
    json.occurredStart !== null ||
    json.occurredEnd !== null ||
    json.mentionedAt !== null;
  return anySet ? json : null;
}

export function causalEdgeToJson(e: WireCausalEdge): CausalEdgeJson {
  return {
    target: bytesToUuid(asBytes(e.target)),
    linkType: linkType(e.linkType),
    weight: typeof e.weight === "number" ? e.weight : 0,
  };
}

export function memoryPayloadToJson(
  m: WireMemoryPayload | null | undefined,
): MemoryPayloadJson | null {
  if (!m) return null;
  return {
    text: m.text ?? "",
    tags: m.tags ?? [],
    proofCount: m.proofCount ?? 0,
    entityIds: (m.entityIds ?? []).map((b) => bytesToUuid(asBytes(b))),
    timestamps: timestampsToJson(m.timestamps),
    causalOut: (m.causalOut ?? []).map(causalEdgeToJson),
    metadata: m.metadata ?? {},
  };
}

export function vectorToSummary(
  v: WireVector | null | undefined,
  headLen = 8,
): VectorSummary | null {
  const bytes = asBytes(v?.f32le);
  if (bytes.length === 0) return null;
  return summarizeF32le(bytes, headLen);
}

export function armScoreToJson(a: WireArmScore | null | undefined): ArmScoreJson {
  // An absent submessage means the arm did not surface this hit at all — which
  // is exactly `present: false`, and is NOT the same as a score of 0.
  if (!a) return { present: false, rank: 0, score: 0 };
  return {
    present: Boolean(a.present),
    rank: a.rank ?? 0,
    score: typeof a.score === "number" ? a.score : 0,
  };
}

export function hitToJson(h: WireHit): HitJson {
  return {
    id: bytesToUuid(asBytes(h.id)),
    memoryType: h.memoryType ?? 0,
    dense: armScoreToJson(h.dense),
    text: armScoreToJson(h.text),
    graph: armScoreToJson(h.graph),
    temporal: armScoreToJson(h.temporal),
    memory: memoryPayloadToJson(h.memory),
  };
}

export function storedMemoryToJson(
  r: WireStoredMemoryRecord,
  headLen = 8,
): StoredMemoryJson {
  return {
    id: bytesToUuid(asBytes(r.id)),
    memoryType: r.memoryType ?? 0,
    memory: memoryPayloadToJson(r.memory),
    vector: vectorToSummary(r.vector, headLen),
  };
}

export function typeStatsToJson(t: {
  memoryType: number;
  docCount: string;
  clusterCount: number;
  trainCount: string;
  hasIndex: boolean;
}): TypeStatsJson {
  return {
    memoryType: t.memoryType ?? 0,
    docCount: u64(t.docCount),
    clusterCount: t.clusterCount ?? 0,
    trainCount: u64(t.trainCount),
    hasIndex: Boolean(t.hasIndex),
  };
}

// ---- WAL --------------------------------------------------------------------

export function walOpToJson(op: WireWalOpDetail): WalOpJson {
  // With `oneofs: true` the loader names the member that is actually set, which
  // is the only reliable discriminator: `defaults: true` would otherwise leave
  // zero-valued siblings looking present.
  switch (op.kind) {
    case "upsert": {
      const u = op.upsert;
      return {
        kind: "upsert",
        id: bytesToUuid(asBytes(u?.id)),
        memoryType: u?.memoryType ?? 0,
        text: u?.text ?? "",
        tags: u?.tags ?? [],
        vectorDim: u?.vectorDim ?? 0,
      };
    }
    case "tombstone":
      return { kind: "tombstone", id: bytesToUuid(asBytes(op.tombstone)) };
    case "patch": {
      const p = op.patch;
      const vectorBytes = asBytes(p?.vector?.f32le);
      return {
        kind: "patch",
        id: bytesToUuid(asBytes(p?.id)),
        proofCountDelta: p?.proofCountDelta ?? 0,
        // proto3 `optional string`: the synthetic-oneof marker distinguishes
        // "set to empty" from "not set".
        setsText: p?._text !== undefined,
        text: p?._text !== undefined ? (p?.text ?? "") : null,
        // Message-typed fields: present => set.
        setsVector: vectorBytes.length > 0,
        vectorDim: Math.floor(vectorBytes.length / 4),
        setsTags: p?.tags != null,
        tags: p?.tags?.tags ?? [],
        setsTimestamps: p?.timestamps != null,
        timestamps: timestampsToJson(p?.timestamps),
        metadata: p?.metadata ?? {},
      };
    }
    case "guardExpectSeqLt":
      return { kind: "guard", expectSeqLt: u64(op.guardExpectSeqLt) };
    default:
      return { kind: "unknown" };
  }
}

export function walEntryToJson(e: WireWalEntryInfo): WalEntryJson {
  return {
    seq: u64(e.seq),
    sizeBytes: u64(e.sizeBytes),
    counts: {
      upserts: e.counts?.upserts ?? 0,
      tombstones: e.counts?.tombstones ?? 0,
      patches: e.counts?.patches ?? 0,
      guards: e.counts?.guards ?? 0,
    },
    folded: Boolean(e.folded),
    ops: (e.ops ?? []).map(walOpToJson),
  };
}

export function listWalToJson(
  r: ListWalResponse,
  elapsedMs: number,
): ListWalJson {
  const walHead = u64(r.walHead);
  const walIndexCursor = u64(r.walIndexCursor);
  return {
    entries: (r.entries ?? []).map(walEntryToJson),
    walHead,
    walIndexCursor,
    nextSeq: u64(r.nextSeq),
    backlog: diffU64(walHead, walIndexCursor),
    elapsedMs,
  };
}

// ---- IVF layout -------------------------------------------------------------

/**
 * Unlike Scan/Get, the layout page ships the FULL centroid and member vectors:
 * the PCA projection runs in the browser, so it needs every component. The
 * member count is bounded by `member_sample` for exactly this reason.
 */
export function clusterToJson(c: WireClusterInfo): ClusterJson {
  const bytes = asBytes(c.centroid?.f32le);
  return {
    clusterId: c.clusterId ?? 0,
    centroid: bytes.length ? f32leToFloats(bytes) : [],
    size: u64(c.size),
    tags: c.tags ?? [],
    hasUntagged: Boolean(c.hasUntagged),
  };
}

export function clusterMemberToJson(m: WireClusterMember): ClusterMemberJson {
  const bytes = asBytes(m.vector?.f32le);
  return {
    id: bytesToUuid(asBytes(m.id)),
    clusterId: m.clusterId ?? 0,
    vector: bytes.length ? f32leToFloats(bytes) : [],
    text: m.text ?? "",
  };
}

export function indexLayoutToJson(
  r: IndexLayoutResponse,
  elapsedMs: number,
): IndexLayoutJson {
  const clusters = (r.clusters ?? [])
    .map(clusterToJson)
    .sort((a, b) => a.clusterId - b.clusterId);
  let total = 0n;
  for (const c of clusters) {
    try {
      total += BigInt(c.size);
    } catch {
      /* keep going: a malformed size should not blank the page */
    }
  }
  return {
    namespace: r.namespace ?? "",
    memoryType: r.memoryType ?? 0,
    generation: u64(r.generation),
    dim: r.dim ?? 0,
    clusters,
    members: (r.members ?? []).map(clusterMemberToJson),
    totalSize: total.toString(),
    elapsedMs,
  };
}

// ---- Read cache -------------------------------------------------------------

/**
 * Infer the object kind from the cache key. memlake never labels these on the
 * wire — the path spells it out — so this is display sugar, and anything
 * unrecognised falls through to "other" rather than being guessed at.
 */
export function objectKindOf(path: string): ObjectKind {
  // Strip the `#start-end` range first: `pk.data#0-13128` is still a pk block.
  const p = splitRange(path).object.toLowerCase();
  const base = p.slice(p.lastIndexOf("/") + 1);
  if (p.includes("/wal/") || p.startsWith("wal/")) return "wal";
  if (base.startsWith("cluster-")) return "cluster";
  if (base.startsWith("centroids")) return "centroids";
  if (base.startsWith("pk.")) return "primary key";
  if (base.startsWith("radj.")) return "reverse adjacency";
  if (base.startsWith("entity.")) return "entity index";
  if (base.startsWith("time.")) return "time index";
  if (base.startsWith("tags.")) return "tags";
  if (base.startsWith("fts.")) return "full-text";
  if (base.startsWith("manifest")) return "manifest";
  return "other";
}

/**
 * A ranged read appends `#start-end` to the key. Split it out so the range is
 * its own column and blocks of one object can be recognised as such.
 */
export function splitRange(path: string): { object: string; range: string | null } {
  const hash = path.lastIndexOf("#");
  if (hash < 0) return { object: path, range: null };
  const range = path.slice(hash + 1);
  // Only treat it as a range if it actually looks like one.
  return /^\d+-\d+$/.test(range)
    ? { object: path.slice(0, hash), range }
    : { object: path, range: null };
}

export function cacheEntryToJson(e: WireCacheEntry): CacheEntryJson {
  const path = e.path ?? "";
  const { object, range } = splitRange(path);
  return {
    namespace: e.namespace ?? "",
    path,
    object,
    range,
    etag: e.etag ?? "",
    bytes: u64(e.bytes),
    // Two independent booleans, never one tier field.
    inMemory: Boolean(e.inMemory),
    onDisk: Boolean(e.onDisk),
    lruRank: e.lruRank ?? 0,
    kind: objectKindOf(path),
  };
}

export function cacheStatsToJson(
  r: CacheStatsResponse,
  namespace: string,
  limit: number,
  elapsedMs: number,
): CacheStatsJson {
  const entries = (r.entries ?? []).map(cacheEntryToJson);
  const hits = u64(r.hits);
  const misses = u64(r.misses);
  const totalEntries = u64(r.totalEntries);

  // "No lookups yet" is a distinct state from a 0% hit ratio, and dividing by
  // zero would render NaN.
  let lookups = "0";
  let hitRatio: number | null = null;
  try {
    const h = BigInt(hits);
    const m = BigInt(misses);
    const total = h + m;
    lookups = total.toString();
    if (total > 0n) {
      // Scale before dividing: BigInt division truncates.
      hitRatio = Number((h * 1000000n) / total) / 1000000;
    }
  } catch {
    /* leave hitRatio null: better nothing than a wrong number */
  }

  // Group the RETURNED entries only. This is not a summary of the whole cache
  // whenever the list was truncated, and the UI says so.
  const kinds = new Map<ObjectKind, { count: number; bytes: bigint }>();
  for (const e of entries) {
    const cur = kinds.get(e.kind) ?? { count: 0, bytes: 0n };
    cur.count += 1;
    try {
      cur.bytes += BigInt(e.bytes);
    } catch {
      /* skip a malformed size rather than losing the whole group */
    }
    kinds.set(e.kind, cur);
  }
  // Biggest first: which object kind is actually consuming the budget is the
  // question this table answers.
  const byKind: CacheKindSummary[] = [...kinds.entries()]
    .map(([kind, v]) => ({ kind, count: v.count, bytes: v.bytes.toString() }))
    .sort((a, b) => cmpU64Desc(a.bytes, b.bytes) || a.kind.localeCompare(b.kind));

  let truncated = false;
  try {
    truncated = BigInt(totalEntries) > BigInt(entries.length);
  } catch {
    truncated = false;
  }

  return {
    enabled: Boolean(r.enabled),
    memBytes: u64(r.memBytes),
    memBudget: u64(r.memBudget),
    diskBytes: u64(r.diskBytes),
    diskBudget: u64(r.diskBudget),
    memEntries: u64(r.memEntries),
    diskEntries: u64(r.diskEntries),
    hits,
    misses,
    hitRatio,
    lookups,
    entries,
    totalEntries,
    truncated,
    limit,
    namespace,
    byKind,
    elapsedMs,
  };
}

/** Descending comparator over u64 decimal strings. */
function cmpU64Desc(a: string, b: string): number {
  try {
    const x = BigInt(a);
    const y = BigInt(b);
    return x < y ? 1 : x > y ? -1 : 0;
  } catch {
    return 0;
  }
}

// ---- Object storage (the physical view) -------------------------------------

/**
 * The server classifies each key itself, so unlike the cache page nothing is
 * inferred here. An enum member this build has no name for still round-trips as
 * a string; it is mapped to UNKNOWN so the UI never renders a bare integer.
 */
function storageKind(v: string | number | null | undefined): StorageObjectKind {
  if (typeof v === "string" && (STORAGE_OBJECT_KINDS as readonly string[]).includes(v)) {
    return v as StorageObjectKind;
  }
  return "OBJECT_KIND_UNKNOWN";
}

export function objectInfoToJson(o: WireObjectInfo): ObjectInfoJson {
  return {
    path: o.path ?? "",
    sizeBytes: u64(o.sizeBytes),
    kind: storageKind(o.kind),
    generation: u64(o.generation),
    // memory_type 0 is a legitimate type, so the flag — not the value — decides
    // whether the key carried one at all.
    memoryType: o.hasMemoryType ? (o.memoryType ?? 0) : null,
    seq: u64(o.seq),
    live: Boolean(o.live),
  };
}

export function listObjectsToJson(
  r: ListObjectsResponse,
  elapsedMs: number,
): ListObjectsJson {
  const objects = (r.objects ?? []).map(objectInfoToJson);
  const totalObjects = u64(r.totalObjects);
  const totalBytes = u64(r.totalBytes);
  const liveBytes = u64(r.liveBytes);
  const deadBytes = diffU64(totalBytes, liveBytes);

  let deadShare: number | null = null;
  try {
    const t = BigInt(totalBytes);
    // An empty namespace is "no answer", not 0% dead.
    if (t > 0n) deadShare = Number((BigInt(deadBytes) * 1000000n) / t) / 1000000;
  } catch {
    /* leave null: better nothing than a wrong share */
  }

  // Grouped over THIS PAGE only. The response's totals are namespace-wide, but
  // per-kind bytes are not reported, so the summary can only describe what was
  // listed — and the view says so whenever the listing is partial.
  let pageBytes = 0n;
  const kinds = new Map<
    StorageObjectKind,
    { count: number; bytes: bigint; deadCount: number; deadBytes: bigint }
  >();
  for (const o of objects) {
    const cur =
      kinds.get(o.kind) ?? { count: 0, bytes: 0n, deadCount: 0, deadBytes: 0n };
    cur.count += 1;
    if (!o.live) cur.deadCount += 1;
    try {
      const b = BigInt(o.sizeBytes);
      cur.bytes += b;
      if (!o.live) cur.deadBytes += b;
      pageBytes += b;
    } catch {
      /* skip a malformed size rather than losing the whole group */
    }
    kinds.set(o.kind, cur);
  }

  // Biggest first: "what is actually consuming this namespace's storage" is the
  // question the summary answers.
  const byKind: ObjectKindSummaryJson[] = [...kinds.entries()]
    .map(([kind, v]) => ({
      kind,
      count: v.count,
      bytes: v.bytes.toString(),
      deadCount: v.deadCount,
      deadBytes: v.deadBytes.toString(),
    }))
    .sort((a, b) => cmpU64Desc(a.bytes, b.bytes) || a.kind.localeCompare(b.kind));

  const nextPageToken = r.nextPageToken ?? "";
  let complete = false;
  try {
    complete = !nextPageToken && BigInt(totalObjects) <= BigInt(objects.length);
  } catch {
    complete = false;
  }

  return {
    objects,
    totalObjects,
    totalBytes,
    liveBytes,
    deadBytes,
    deadShare,
    generation: u64(r.generation),
    nextPageToken,
    pageBytes: pageBytes.toString(),
    byKind,
    complete,
    elapsedMs,
  };
}

/**
 * `DecodeObjectResponse.json` is a debugging view whose shape follows the
 * on-disk formats — deliberately not a contract. It is re-indented here purely
 * so the panel can show it, and passed through verbatim when it does not parse.
 * Nothing reads a field out of it.
 */
export function decodeObjectToJson(
  r: DecodeObjectResponse,
  path: string,
  limit: number,
  elapsedMs: number,
): DecodeObjectJson {
  const raw = r.json ?? "";
  let json = raw;
  let jsonPretty = false;
  if (raw.trim()) {
    try {
      json = JSON.stringify(JSON.parse(raw), null, 2);
      jsonPretty = true;
    } catch {
      json = raw;
    }
  }
  const reason = (r.undecodableReason ?? "").trim();
  return {
    path,
    kind: storageKind(r.kind),
    json,
    jsonPretty,
    sizeBytes: u64(r.sizeBytes),
    totalItems: u64(r.totalItems),
    truncated: Boolean(r.truncated),
    undecodableReason: reason ? reason : null,
    limit,
    elapsedMs,
  };
}

export function statsToJson(s: StatsResponse, elapsedMs: number): StatsJson {
  const walHead = u64(s.walHead);
  const walIndexCursor = u64(s.walIndexCursor);
  const backlog = diffU64(walHead, walIndexCursor);
  return {
    namespace: s.namespace ?? "",
    generation: u64(s.generation),
    prevGeneration: optU64(s._prevGeneration !== undefined, s.prevGeneration),
    walHead,
    walIndexCursor,
    backlog,
    tokenizerConfigHash: s.tokenizerConfigHash ?? "",
    formatVersion: s.formatVersion ?? 0,
    docCount: u64(s.docCount),
    throughSeq: u64(s.throughSeq),
    loadRoundtrips: s.loadRoundtrips ?? 0,
    types: (s.types ?? []).map(typeStatsToJson).sort((a, b) => a.memoryType - b.memoryType),
    elapsedMs,
  };
}
