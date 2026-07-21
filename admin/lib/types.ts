/**
 * The JSON contract between the Next.js route handlers (`app/api/**`) and the
 * client components. This module is deliberately dependency-free and pure so it
 * can be imported from both sides of the boundary.
 *
 * Wire notes that shape these types:
 *  - protobuf 64-bit ints arrive from proto-loader as decimal *strings* (the
 *    loader is configured with `longs: String`). They stay strings here so no
 *    precision is lost in JSON; use BigInt for arithmetic.
 *  - protobuf `bytes` arrive as Node Buffers. Ids are 16 raw bytes and are
 *    always converted to canonical UUID strings before crossing this boundary.
 *  - a `Vector` is raw little-endian f32; it is never shipped whole to the
 *    browser, only summarised (see `VectorSummary`).
 */

// ---- enums (mirrors of the proto enums, as string names) --------------------

export const CONSISTENCIES = ["STRONG", "EVENTUAL"] as const;
export type Consistency = (typeof CONSISTENCIES)[number];

export const TAGS_MATCHES = [
  "ANY",
  "ALL",
  "ANY_STRICT",
  "ALL_STRICT",
  "EXACT",
] as const;
export type TagsMatch = (typeof TAGS_MATCHES)[number];

export const TAGS_MATCH_HELP: Record<TagsMatch, string> = {
  ANY: "at least one of the tags (untagged memories allowed)",
  ALL: "all of the tags (untagged memories allowed)",
  ANY_STRICT: "ANY, but untagged memories are excluded",
  ALL_STRICT: "ALL, but untagged memories are excluded",
  EXACT: "the memory's tag set equals the filter exactly",
};

export const LINK_TYPES = ["CAUSES", "CAUSED_BY", "ENABLES", "PREVENTS"] as const;
export type LinkType = (typeof LINK_TYPES)[number];

export const ARMS = ["dense", "text", "graph", "temporal"] as const;
export type Arm = (typeof ARMS)[number];

export const ARM_SCORE_KIND: Record<Arm, string> = {
  dense: "cosine similarity",
  text: "BM25",
  graph: "activation",
  temporal: "proximity to the window centre",
};

export const ARM_HELP: Record<Arm, string> = {
  dense: "IVF vector search; runs when a vector is supplied",
  text: "BM25 full-text; runs when text is supplied",
  graph: "link expansion; runs alongside the dense arm",
  temporal:
    "entry points inside [from, to] by effective time, spread one hop; needs both bounds AND a vector",
};

// ---- value objects ----------------------------------------------------------

export interface TimestampsJson {
  /** int64 as a decimal string, or null when the field was not set. */
  eventDate: string | null;
  occurredStart: string | null;
  occurredEnd: string | null;
  mentionedAt: string | null;
}

export interface CausalEdgeJson {
  /** 16-byte MemoryId, rendered as a UUID. */
  target: string;
  linkType: LinkType;
  weight: number;
}

export interface MemoryPayloadJson {
  text: string;
  tags: string[];
  proofCount: number;
  /** 16-byte EntityIds, rendered as UUIDs. */
  entityIds: string[];
  timestamps: TimestampsJson | null;
  causalOut: CausalEdgeJson[];
  /** Opaque client metadata, returned verbatim by the server. */
  metadata: Record<string, string>;
}

/**
 * A compact stand-in for a full embedding. 384 raw floats per row would swamp
 * both the JSON payload and the table, so the handler sends the shape plus a
 * short prefix.
 */
export interface VectorSummary {
  dim: number;
  /** The first `head.length` components, in order. */
  head: number[];
  /** L2 norm of the full vector (≈1 for a bge embedding). */
  norm: number;
  /** Total byte length of the `f32le` payload. */
  bytes: number;
}

export interface StoredMemoryJson {
  /** 16-byte MemoryId, rendered as a UUID. */
  id: string;
  memoryType: number;
  memory: MemoryPayloadJson | null;
  vector: VectorSummary | null;
}

export interface ArmScoreJson {
  /**
   * Whether this arm surfaced the hit at all. `false` is NOT the same as
   * `score === 0` — the arm simply never saw this id.
   */
  present: boolean;
  /** 0-based position within this arm. Meaningless when `present` is false. */
  rank: number;
  /** The arm's native score. Meaningless when `present` is false. */
  score: number;
}

export interface HitJson {
  id: string;
  memoryType: number;
  dense: ArmScoreJson;
  text: ArmScoreJson;
  graph: ArmScoreJson;
  temporal: ArmScoreJson;
  /** The stored memory, returned inline by the server (no hydrate roundtrip). */
  memory: MemoryPayloadJson | null;
}

// ---- API responses ----------------------------------------------------------

export interface ApiErrorBody {
  error: {
    /** Numeric gRPC status code, or -1 for a local (non-RPC) failure. */
    code: number;
    /** e.g. "UNIMPLEMENTED", "UNAVAILABLE", "LOCAL". */
    codeName: string;
    message: string;
    /** Operator-facing suggestion, when we can offer one. */
    hint?: string;
  };
}

export interface ListNamespacesJson {
  namespaces: string[];
  /** Server wall-clock for the RPC, in ms. */
  elapsedMs: number;
}

export interface CreateNamespaceJson {
  namespace: string;
  elapsedMs: number;
}

export interface TypeStatsJson {
  memoryType: number;
  /** uint64 as a decimal string. */
  docCount: string;
  clusterCount: number;
  trainCount: string;
  hasIndex: boolean;
}

export interface StatsJson {
  namespace: string;
  generation: string;
  prevGeneration: string | null;
  walHead: string;
  walIndexCursor: string;
  /** `wal_head - wal_index_cursor`, computed with BigInt. */
  backlog: string;
  tokenizerConfigHash: string;
  formatVersion: number;
  docCount: string;
  throughSeq: string;
  loadRoundtrips: number;
  types: TypeStatsJson[];
  elapsedMs: number;
}

export interface ScanJson {
  memories: StoredMemoryJson[];
  /** Opaque cursor; empty string means the scan is exhausted. */
  nextPageToken: string;
  elapsedMs: number;
}

export interface GetJson {
  memories: StoredMemoryJson[];
  elapsedMs: number;
}

export type QueryVectorSource = "embedded" | "raw" | "none";

export interface QueryJson {
  hits: HitJson[];
  loadRoundtrips: number;
  /** Wall-clock of the Query RPC alone, measured server-side. */
  rpcMs: number;
  /** Wall-clock of embedding the query text, when we embedded it. */
  embedMs: number | null;
  vectorSource: QueryVectorSource;
  vectorDim: number | null;
  /** Model id used for `vectorSource === "embedded"`. */
  embeddingModel: string | null;
  /** The exact prefix prepended to the query text before embedding. */
  queryPrefix: string | null;
  /** Whether the temporal arm was asked for (both bounds set + a vector sent). */
  temporalWindow: { from: string; to: string } | null;
}

// ---- WAL --------------------------------------------------------------------

export interface WalOpCountsJson {
  upserts: number;
  tombstones: number;
  patches: number;
  guards: number;
}

export interface WalUpsertJson {
  kind: "upsert";
  id: string;
  memoryType: number;
  text: string;
  tags: string[];
  /** 0 means the memory carries NO embedding — render that, not "0". */
  vectorDim: number;
}

export interface WalTombstoneJson {
  kind: "tombstone";
  id: string;
}

export interface WalPatchJson {
  kind: "patch";
  id: string;
  proofCountDelta: number;
  /** Which optional fields the patch actually sets (present => set). */
  setsText: boolean;
  text: string | null;
  setsVector: boolean;
  vectorDim: number;
  setsTags: boolean;
  tags: string[];
  setsTimestamps: boolean;
  timestamps: TimestampsJson | null;
  /** Merged, not replaced. */
  metadata: Record<string, string>;
}

export interface WalGuardJson {
  kind: "guard";
  /** The optimistic precondition: the write applies only if head < this seq. */
  expectSeqLt: string;
}

export type WalOpJson =
  | WalUpsertJson
  | WalTombstoneJson
  | WalPatchJson
  | WalGuardJson
  | { kind: "unknown" };

export interface WalEntryJson {
  seq: string;
  sizeBytes: string;
  counts: WalOpCountsJson;
  /**
   * True once the indexer has folded this entry into a generation
   * (seq <= wal_index_cursor). Un-folded entries are exactly what a STRONG
   * query pays to scan on every read.
   */
  folded: boolean;
  /** Only populated when include_ops was requested. */
  ops: WalOpJson[];
}

export interface ListWalJson {
  entries: WalEntryJson[];
  walHead: string;
  walIndexCursor: string;
  /** Resume point; "0" when the log is exhausted. */
  nextSeq: string;
  /** wal_head - wal_index_cursor, computed with BigInt. */
  backlog: string;
  elapsedMs: number;
}

export interface ListWalRequestBody {
  startSeq: string;
  limit: number;
  includeOps: boolean;
}

// ---- IVF layout -------------------------------------------------------------

export interface ClusterJson {
  clusterId: number;
  /** Full centroid, as float32 values — the browser needs them to run PCA. */
  centroid: number[];
  /** Trained size: excludes un-indexed WAL-tail writes. */
  size: string;
  tags: string[];
  hasUntagged: boolean;
}

export interface ClusterMemberJson {
  id: string;
  clusterId: number;
  vector: number[];
  text: string;
}

export interface IndexLayoutJson {
  namespace: string;
  memoryType: number;
  generation: string;
  /** 0 when this memory_type has no embeddings at all. */
  dim: number;
  clusters: ClusterJson[];
  members: ClusterMemberJson[];
  /** Sum of the trained cluster sizes — the denominator for "% of corpus". */
  totalSize: string;
  elapsedMs: number;
}

export interface IndexLayoutRequestBody {
  memoryType: number;
  memberSample: number;
  consistency: Consistency;
}

export type EmbedState = "disabled" | "idle" | "loading" | "ready" | "error";

export interface EmbedStatusJson {
  enabled: boolean;
  model: string;
  dim: number;
  /** "cls" — matches the benchmark harness; see lib/embed.ts. */
  pooling: string;
  queryPrefix: string;
  state: EmbedState;
  error: string | null;
}

// ---- request bodies ---------------------------------------------------------

export interface TagFilterInput {
  tags: string[];
  mode: TagsMatch;
}

export interface ScanRequestBody {
  memoryTypes: number[];
  limit: number;
  pageToken: string;
  includeVector: boolean;
  tags: TagFilterInput | null;
  consistency: Consistency;
}

export interface GetRequestBody {
  ids: string[];
  includeVector: boolean;
  consistency: Consistency;
}

export interface QueryRequestBody {
  /** Free text: drives the BM25 arm, and (unless `vector` is given) the embedding. */
  text: string;
  memoryTypes: number[];
  tags: TagFilterInput | null;
  vectorTopK: number;
  textTopK: number;
  graphTopK: number;
  nprobe: number;
  consistency: Consistency;
  /**
   * "embed"  – embed `text` server-side (default)
   * "raw"    – use `vector` verbatim
   * "none"   – skip the dense + graph arms entirely
   */
  vectorMode: "embed" | "raw" | "none";
  /** Only read when `vectorMode === "raw"`. */
  vector: number[] | null;
  /**
   * The temporal arm's window, as epoch int64 decimal strings. The arm only
   * runs when BOTH bounds are set AND a vector is supplied; either omitted
   * skips it. The unit only has to match what was written.
   */
  temporalFrom: string | null;
  temporalTo: string | null;
}

// ---- helpers ----------------------------------------------------------------

/**
 * Detect the error envelope. Note the deliberate strictness: several success
 * shapes carry their own nullable `error` field (EmbedStatusJson does), and
 * `typeof null === "object"`, so testing only for the key's presence would
 * classify a perfectly good response as a failure. Require the envelope's own
 * fields.
 */
export function isApiError(body: unknown): body is ApiErrorBody {
  if (typeof body !== "object" || body === null || !("error" in body)) {
    return false;
  }
  const err = (body as { error: unknown }).error;
  return (
    typeof err === "object" &&
    err !== null &&
    typeof (err as ApiErrorBody["error"]).codeName === "string" &&
    typeof (err as ApiErrorBody["error"]).message === "string"
  );
}
