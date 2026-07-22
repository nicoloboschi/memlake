/**
 * The single gRPC client for the whole admin UI. SERVER ONLY — gRPC cannot be
 * spoken from a browser, so every memlake call happens inside a route handler
 * and the React tree only ever sees JSON.
 *
 * The proto is loaded dynamically at runtime rather than compiled to static
 * stubs: the contract file in this repo stays the single source of truth and
 * the UI cannot drift from it silently.
 *
 * The client is memoized on `globalThis` — the standard Next.js dev pattern —
 * so hot reloads do not leak a new channel (and a new TCP connection) per edit.
 */

import path from "node:path";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";

import type { TagFilterInput, TagsMatch } from "./types";

export const DEFAULT_ADDR = "localhost:50051";
export const DEFAULT_PROTO_PATH = "../proto/memlake/v1/memlake.proto";
export const DEFAULT_DEADLINE_MS = 30_000;

// ---- wire shapes ------------------------------------------------------------
//
// What @grpc/proto-loader hands back, given the options below:
//   * `bytes`      -> Node Buffer
//   * 64-bit ints  -> decimal string   (longs: String)
//   * enums        -> string name      (enums: String)
//   * field names  -> lowerCamelCase   (keepCase: false)
//   * unset submessages -> null        (defaults: true)

export interface WireVector {
  f32le: Buffer | null;
}

export interface WireTimestamps {
  eventDate?: string | null;
  occurredStart?: string | null;
  occurredEnd?: string | null;
  mentionedAt?: string | null;
  // proto3 `optional` becomes a synthetic oneof; with `oneofs: true` the loader
  // sets these marker fields to the name of the member that was actually set.
  _eventDate?: string;
  _occurredStart?: string;
  _occurredEnd?: string;
  _mentionedAt?: string;
}

export interface WireCausalEdge {
  target: Buffer | null;
  linkType: string;
  weight: number;
}

export interface WireMemoryPayload {
  text: string;
  tags: string[];
  proofCount: number;
  entityIds: Buffer[];
  timestamps: WireTimestamps | null;
  causalOut: WireCausalEdge[];
  metadata: Record<string, string>;
}

export interface WireArmScore {
  present: boolean;
  rank: number;
  score: number;
}

export interface WireHit {
  id: Buffer | null;
  memoryType: number;
  dense: WireArmScore | null;
  text: WireArmScore | null;
  graph: WireArmScore | null;
  temporal: WireArmScore | null;
  memory: WireMemoryPayload | null;
}

export interface WireStoredMemoryRecord {
  id: Buffer | null;
  memoryType: number;
  memory: WireMemoryPayload | null;
  vector: WireVector | null;
}

export interface WireTypeStats {
  memoryType: number;
  docCount: string;
  clusterCount: number;
  trainCount: string;
  hasIndex: boolean;
}

export interface WireTagFilter {
  tags: string[];
  mode: TagsMatch;
}

// requests

export interface CreateNamespaceRequest {
  namespace: string;
}
export type CreateNamespaceResponse = Record<string, never>;

export type ListNamespacesRequest = Record<string, never>;
export interface ListNamespacesResponse {
  namespaces: string[];
}

export interface StatsRequest {
  namespace: string;
}
export interface StatsResponse {
  namespace: string;
  generation: string;
  prevGeneration?: string | null;
  _prevGeneration?: string;
  walHead: string;
  walIndexCursor: string;
  tokenizerConfigHash: string;
  formatVersion: number;
  docCount: string;
  types: WireTypeStats[];
  throughSeq: string;
  loadRoundtrips: number;
}

export interface GetRequest {
  namespace: string;
  ids: Uint8Array[];
  includeVector: boolean;
}
export interface GetResponse {
  memories: WireStoredMemoryRecord[];
}

export interface ScanRequest {
  namespace: string;
  memoryTypes: number[];
  limit: number;
  pageToken: string;
  includeVector: boolean;
  tags?: WireTagFilter;
}
export interface ScanResponse {
  memories: WireStoredMemoryRecord[];
  nextPageToken: string;
}

export interface QueryRequest {
  namespace: string;
  memoryTypes: number[];
  vector?: { f32le: Uint8Array };
  text: string;
  tags?: WireTagFilter;
  vectorTopK: number;
  textTopK: number;
  graphTopK: number;
  nprobe: number;
  // proto3 `optional int64`: leave both undefined to skip the temporal arm.
  // With `longs: String` these go over the wire as decimal strings.
  temporalFrom?: string;
  temporalTo?: string;
}
export interface QueryResponse {
  hits: WireHit[];
  loadRoundtrips: number;
}

// ---- WAL introspection ------------------------------------------------------

export interface WireWalOpCounts {
  upserts: number;
  tombstones: number;
  patches: number;
  guards: number;
}

export interface WireWalUpsert {
  id: Buffer | null;
  memoryType: number;
  text: string;
  tags: string[];
  vectorDim: number;
}

export interface WirePatch {
  id: Buffer | null;
  proofCountDelta: number;
  text?: string | null;
  _text?: string;
  vector: WireVector | null;
  tags: { tags: string[] } | null;
  timestamps: WireTimestamps | null;
  metadata: Record<string, string>;
}

export interface WireWalOpDetail {
  // With `oneofs: true` the loader sets this to the name of the member present.
  kind?: "upsert" | "tombstone" | "patch" | "guardExpectSeqLt";
  upsert?: WireWalUpsert | null;
  tombstone?: Buffer | null;
  patch?: WirePatch | null;
  guardExpectSeqLt?: string;
}

export interface WireWalEntryInfo {
  seq: string;
  sizeBytes: string;
  counts: WireWalOpCounts | null;
  folded: boolean;
  ops: WireWalOpDetail[];
}

export interface ListWalRequest {
  namespace: string;
  startSeq: string;
  limit: number;
  includeOps: boolean;
}

export interface ListWalResponse {
  entries: WireWalEntryInfo[];
  walHead: string;
  walIndexCursor: string;
  nextSeq: string;
}

// ---- IVF layout -------------------------------------------------------------

export interface WireClusterInfo {
  clusterId: number;
  centroid: WireVector | null;
  size: string;
  tags: string[];
  hasUntagged: boolean;
}

export interface WireClusterMember {
  id: Buffer | null;
  clusterId: number;
  vector: WireVector | null;
  text: string;
}

export interface IndexLayoutRequest {
  namespace: string;
  memoryType: number;
  memberSample: number;
}

export interface IndexLayoutResponse {
  namespace: string;
  memoryType: number;
  generation: string;
  dim: number;
  clusters: WireClusterInfo[];
  members: WireClusterMember[];
}

// ---- Read cache (node-local) ------------------------------------------------

export interface WireCacheEntry {
  namespace: string;
  path: string;
  etag: string;
  bytes: string;
  /** Independent of `onDisk` — the tiers overlap, they are not a partition. */
  inMemory: boolean;
  onDisk: boolean;
  lruRank: number;
}

export interface CacheStatsRequest {
  namespace: string;
  limit: number;
}

export interface CacheStatsResponse {
  enabled: boolean;
  memBytes: string;
  memBudget: string;
  diskBytes: string;
  diskBudget: string;
  memEntries: string;
  diskEntries: string;
  hits: string;
  misses: string;
  entries: WireCacheEntry[];
  totalEntries: string;
}

// ---- Object storage browser -------------------------------------------------

export interface WireObjectInfo {
  path: string;
  sizeBytes: string;
  /** `enums: String` — the enum member name, e.g. "CLUSTER". */
  kind: string;
  /** 0 when the key carries no generation (manifest, WAL entries). */
  generation: string;
  memoryType: number;
  /** memory_type is a real 0, so the flag is the only way to say "absent". */
  hasMemoryType: boolean;
  seq: string;
  /** False = no longer referenced by the current manifest, i.e. GC fodder. */
  live: boolean;
}

export interface ListObjectsRequest {
  namespace: string;
  limit: number;
  pageToken: string;
}

export interface ListObjectsResponse {
  objects: WireObjectInfo[];
  totalObjects: string;
  totalBytes: string;
  liveBytes: string;
  generation: string;
  nextPageToken: string;
}

export interface DecodeObjectRequest {
  namespace: string;
  path: string;
  limit: number;
}

export interface DecodeObjectResponse {
  kind: string;
  /** A debugging view of the object, NOT a stable contract. Never parsed for meaning. */
  json: string;
  sizeBytes: string;
  totalItems: string;
  truncated: boolean;
  /** Non-empty when the format is opaque to memlake (the tantivy split). */
  undecodableReason: string;
}

// ---- errors -----------------------------------------------------------------

/** A failure worth showing to an operator verbatim, RPC or otherwise. */
export class MemlakeError extends Error {
  readonly code: number;
  readonly codeName: string;
  readonly hint?: string;

  constructor(code: number, codeName: string, message: string, hint?: string) {
    super(message);
    this.name = "MemlakeError";
    this.code = code;
    this.codeName = codeName;
    this.hint = hint;
  }
}

function codeName(code: number): string {
  return grpc.status[code] ?? `CODE_${code}`;
}

function hintFor(code: number, rpc: string): string | undefined {
  switch (code) {
    case grpc.status.UNIMPLEMENTED:
      return `server returned UNIMPLEMENTED for ${rpc} — the mlake-server at ${memlakeAddr()} is running but does not implement this RPC yet. Rebuild and restart it (cargo run --release -p mlake-server -- serve).`;
    case grpc.status.UNAVAILABLE:
      return `could not reach mlake-server at ${memlakeAddr()} — start it with \`cargo run --release -p mlake-server -- serve\` (and \`docker compose up -d\` for MinIO), or set MEMLAKE_ADDR.`;
    case grpc.status.DEADLINE_EXCEEDED:
      return `no response within ${DEFAULT_DEADLINE_MS / 1000}s — the server may be blocked on object storage.`;
    case grpc.status.NOT_FOUND:
      return "the namespace may not exist yet; create it from the namespaces page.";
    case grpc.status.INVALID_ARGUMENT:
      return "the server rejected the request arguments — check memory_type (must fit u8) and vector dimensionality.";
    default:
      return undefined;
  }
}

function isServiceError(e: unknown): e is grpc.ServiceError {
  return (
    typeof e === "object" &&
    e !== null &&
    typeof (e as grpc.ServiceError).code === "number"
  );
}

/** Normalise anything thrown inside a route handler into a MemlakeError. */
export function toMemlakeError(e: unknown, rpc = "rpc"): MemlakeError {
  if (e instanceof MemlakeError) return e;
  if (isServiceError(e)) {
    const code = e.code;
    return new MemlakeError(
      code,
      codeName(code),
      e.details || e.message || "gRPC call failed",
      hintFor(code, rpc),
    );
  }
  return new MemlakeError(
    -1,
    "LOCAL",
    e instanceof Error ? e.message : String(e),
    undefined,
  );
}

// ---- client -----------------------------------------------------------------

export function memlakeAddr(): string {
  return process.env.MEMLAKE_ADDR || DEFAULT_ADDR;
}

export function memlakeProtoPath(): string {
  const configured = process.env.MEMLAKE_PROTO_PATH || DEFAULT_PROTO_PATH;
  if (path.isAbsolute(configured)) return configured;
  // Relative to the Next.js cwd (admin/), so the default reaches the repo's
  // proto tree. The ignore comment keeps the bundler from concluding that the
  // whole project needs tracing because of this one dynamic path.
  return path.resolve(/* turbopackIgnore: true */ process.cwd(), configured);
}

type UnaryCall<Req, Res> = (
  req: Req,
  options: grpc.CallOptions,
  cb: (err: grpc.ServiceError | null, res: Res) => void,
) => grpc.ClientUnaryCall;

type MemlakeStub = grpc.Client & {
  CreateNamespace: UnaryCall<CreateNamespaceRequest, CreateNamespaceResponse>;
  ListNamespaces: UnaryCall<ListNamespacesRequest, ListNamespacesResponse>;
  Stats: UnaryCall<StatsRequest, StatsResponse>;
  Get: UnaryCall<GetRequest, GetResponse>;
  Scan: UnaryCall<ScanRequest, ScanResponse>;
  Query: UnaryCall<QueryRequest, QueryResponse>;
  ListWal: UnaryCall<ListWalRequest, ListWalResponse>;
  IndexLayout: UnaryCall<IndexLayoutRequest, IndexLayoutResponse>;
  CacheStats: UnaryCall<CacheStatsRequest, CacheStatsResponse>;
  ListObjects: UnaryCall<ListObjectsRequest, ListObjectsResponse>;
  DecodeObject: UnaryCall<DecodeObjectRequest, DecodeObjectResponse>;
};

interface MemlakeGlobal {
  stub?: MemlakeStub;
  addr?: string;
  protoPath?: string;
}

// Memoized across hot reloads: `next dev` re-evaluates modules on every edit and
// a fresh grpc.Client each time would leak channels/sockets for the session.
const globalForMemlake = globalThis as typeof globalThis & {
  __memlake?: MemlakeGlobal;
};

function loadStub(): MemlakeStub {
  const protoPath = memlakeProtoPath();
  const packageDefinition = protoLoader.loadSync(protoPath, {
    // 64-bit ints as strings: JS numbers cannot hold a u64 WAL sequence.
    longs: String,
    // enums as their string names, so the UI never juggles magic integers.
    enums: String,
    // fill in scalar defaults, so route handlers do not branch on undefined.
    defaults: true,
    // lowerCamelCase field names on the JS side.
    keepCase: false,
    oneofs: true,
    // repeated bytes stay Buffers; no arrays-of-arrays.
    bytes: Buffer,
    // The contract has no imports today, but rooting the include path at the
    // proto tree keeps `import "memlake/v1/..."` working if one is added.
    includeDirs: [path.resolve(path.dirname(protoPath), "../..")],
  });

  const pkg = grpc.loadPackageDefinition(packageDefinition) as unknown as {
    memlake?: { v1?: { Memlake?: grpc.ServiceClientConstructor } };
  };
  const Ctor = pkg.memlake?.v1?.Memlake;
  if (!Ctor) {
    throw new MemlakeError(
      -1,
      "LOCAL",
      `service memlake.v1.Memlake not found in ${protoPath}`,
      "check MEMLAKE_PROTO_PATH — it must point at proto/memlake/v1/memlake.proto",
    );
  }

  return new Ctor(memlakeAddr(), grpc.credentials.createInsecure(), {
    // memlake is an internal, unauthenticated service; TLS is terminated (or
    // absent) at the mesh, so insecure credentials are correct here.
    "grpc.max_receive_message_length": 64 * 1024 * 1024,
    "grpc.max_send_message_length": 16 * 1024 * 1024,
    "grpc.keepalive_time_ms": 60_000,
    // The dynamic loader synthesises the six RPC methods at construction time,
    // which TypeScript cannot see; MemlakeStub is the hand-written shape of
    // what it produces.
  }) as unknown as MemlakeStub;
}

function stub(): MemlakeStub {
  const g = (globalForMemlake.__memlake ??= {});
  const addr = memlakeAddr();
  const protoPath = memlakeProtoPath();
  // Re-create if the env moved under us (e.g. .env.local edited in dev).
  if (g.stub && (g.addr !== addr || g.protoPath !== protoPath)) {
    g.stub.close();
    g.stub = undefined;
  }
  if (!g.stub) {
    try {
      g.stub = loadStub();
    } catch (e) {
      if (e instanceof MemlakeError) throw e;
      throw new MemlakeError(
        -1,
        "LOCAL",
        `failed to load ${protoPath}: ${e instanceof Error ? e.message : String(e)}`,
        "MEMLAKE_PROTO_PATH is resolved relative to the admin/ directory; the default is ../proto/memlake/v1/memlake.proto",
      );
    }
    g.addr = addr;
    g.protoPath = protoPath;
  }
  return g.stub;
}

function unary<Req, Res>(
  rpc: keyof MemlakeStub & string,
  req: Req,
  deadlineMs = DEFAULT_DEADLINE_MS,
): Promise<Res> {
  return new Promise<Res>((resolve, reject) => {
    let client: MemlakeStub;
    try {
      client = stub();
    } catch (e) {
      reject(toMemlakeError(e, rpc));
      return;
    }
    const method = client[rpc] as unknown as UnaryCall<Req, Res>;
    const options: grpc.CallOptions = {
      deadline: Date.now() + deadlineMs,
    };
    try {
      method.call(client, req, options, (err, res) => {
        if (err) reject(toMemlakeError(err, rpc));
        else resolve(res);
      });
    } catch (e) {
      reject(toMemlakeError(e, rpc));
    }
  });
}

// ---- one promisified wrapper per RPC ----------------------------------------

export const memlake = {
  createNamespace(req: CreateNamespaceRequest, deadlineMs?: number) {
    return unary<CreateNamespaceRequest, CreateNamespaceResponse>(
      "CreateNamespace",
      req,
      deadlineMs,
    );
  },
  listNamespaces(deadlineMs?: number) {
    return unary<ListNamespacesRequest, ListNamespacesResponse>(
      "ListNamespaces",
      {} as ListNamespacesRequest,
      deadlineMs,
    );
  },
  stats(req: StatsRequest, deadlineMs?: number) {
    return unary<StatsRequest, StatsResponse>("Stats", req, deadlineMs);
  },
  get(req: GetRequest, deadlineMs?: number) {
    return unary<GetRequest, GetResponse>("Get", req, deadlineMs);
  },
  scan(req: ScanRequest, deadlineMs?: number) {
    return unary<ScanRequest, ScanResponse>("Scan", req, deadlineMs);
  },
  query(req: QueryRequest, deadlineMs?: number) {
    return unary<QueryRequest, QueryResponse>("Query", req, deadlineMs);
  },
  listWal(req: ListWalRequest, deadlineMs?: number) {
    return unary<ListWalRequest, ListWalResponse>("ListWal", req, deadlineMs);
  },
  indexLayout(req: IndexLayoutRequest, deadlineMs?: number) {
    return unary<IndexLayoutRequest, IndexLayoutResponse>(
      "IndexLayout",
      req,
      deadlineMs,
    );
  },
  cacheStats(req: CacheStatsRequest, deadlineMs?: number) {
    return unary<CacheStatsRequest, CacheStatsResponse>(
      "CacheStats",
      req,
      deadlineMs,
    );
  },
  listObjects(req: ListObjectsRequest, deadlineMs?: number) {
    return unary<ListObjectsRequest, ListObjectsResponse>(
      "ListObjects",
      req,
      deadlineMs,
    );
  },
  decodeObject(req: DecodeObjectRequest, deadlineMs?: number) {
    return unary<DecodeObjectRequest, DecodeObjectResponse>(
      "DecodeObject",
      req,
      deadlineMs,
    );
  },
};

/** Build a TagFilter submessage, or `undefined` for "no tag filter". */
export function tagFilter(input: TagFilterInput | null | undefined) {
  if (!input) return undefined;
  const tags = input.tags.map((t) => t.trim()).filter(Boolean);
  if (tags.length === 0) return undefined;
  return { tags, mode: input.mode };
}
