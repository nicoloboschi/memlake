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
import { summarizeF32le } from "./vector";
import type {
  ArmScoreJson,
  CausalEdgeJson,
  HitJson,
  LinkType,
  MemoryPayloadJson,
  StatsJson,
  StoredMemoryJson,
  TimestampsJson,
  TypeStatsJson,
  VectorSummary,
} from "./types";
import { LINK_TYPES } from "./types";
import type {
  StatsResponse,
  WireArmScore,
  WireCausalEdge,
  WireHit,
  WireMemoryPayload,
  WireStoredMemoryRecord,
  WireTimestamps,
  WireVector,
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

export function statsToJson(s: StatsResponse, elapsedMs: number): StatsJson {
  const walHead = u64(s.walHead);
  const walIndexCursor = u64(s.walIndexCursor);
  // u64 subtraction: a WAL sequence can exceed Number.MAX_SAFE_INTEGER.
  let backlog = "0";
  try {
    const d = BigInt(walHead) - BigInt(walIndexCursor);
    backlog = (d > 0n ? d : 0n).toString();
  } catch {
    backlog = "0";
  }
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
