/**
 * Observability data source. SERVER ONLY.
 *
 * Serve nodes write two things into the reserved `_obs/` root:
 *
 *   `_obs/rollup/{node}.json`                 one small OVERWRITTEN object per node — the fleet
 *                                             overview (heartbeat, totals, action mix, per-ns stats)
 *   `_obs/traces/{node}/{ms}-{seq}.jsonl`     APPEND-ONLY immutable trace batches, flushed every
 *                                             couple of seconds; the indexer expires them past the
 *                                             retention window (default 24h)
 *
 * The admin reads these DIRECTLY from S3 — no gRPC, no pod scraping. Because batches are append-only
 * and time-keyed, "recent traces" is just the newest N objects, and a trace id lookup is a
 * newest-first scan; neither needs to read the whole retention window.
 *
 * Config comes from `MEMLAKE_OBS_S3_*`, falling back to the `MEMLAKE_QUERY_S3_*` block the serve
 * pods already mount (same bucket), so a deploy needs no new secret — just bucket read access.
 */

import {
  GetObjectCommand,
  ListObjectsV2Command,
  S3Client,
} from "@aws-sdk/client-s3";

export const OBS_TRACES_PREFIX = "_obs/traces/";
export const OBS_ROLLUP_PREFIX = "_obs/rollup/";

/** Newest batch objects read for the recent-traces list (each batch is ~one flush interval). */
const LIST_BATCH_OBJECTS = 80;
/** Newest batch objects scanned when hunting a specific trace id / namespace. */
const SCAN_BATCH_OBJECTS = 400;

export interface NodeTotals {
  count: number;
  qps: number;
  p50_ms: number;
  p99_ms: number;
  cache_hit: number;
}

export interface NsRollup {
  ns: string;
  count: number;
  p50_ms: number;
  p99_ms: number;
}

export interface NodeHeader {
  kind: "header";
  node_id: string;
  updated_ms: number;
  uptime_ms: number;
  totals: NodeTotals;
  by_action: Record<string, number>;
  by_namespace: NsRollup[];
  /** Records still buffered in memory, and records dropped because S3 wasn't draining. */
  pending?: number;
  dropped?: number;
}

export type TraceRecord = {
  id?: string;
  op?: string;
  namespace?: string;
  total_ms?: number;
  ts_ms?: number;
  snapshot?: { action?: string; open_ms?: number; tail_entries?: number };
  [k: string]: unknown;
};

export interface NodeSummary {
  header: NodeHeader;
  /** Bytes of the node's rollup object (tiny) — kept for shape compatibility. */
  sizeBytes: number;
  fetchedMs: number;
}

/** A light trace summary for the explorer list — no spans, so the list payload stays small. */
export interface TraceSummary {
  id: string;
  node_id: string;
  namespace: string;
  op: string;
  total_ms: number;
  ts_ms: number;
  snapshot?: { action?: string; open_ms?: number; tail_entries?: number };
}

// ---- config + client --------------------------------------------------------

interface S3Config {
  bucket: string;
  region: string;
  endpoint?: string;
  accessKeyId?: string;
  secretAccessKey?: string;
}

function firstEnv(...keys: string[]): string | undefined {
  for (const k of keys) {
    const v = process.env[k];
    if (v && v.trim() !== "") return v.trim();
  }
  return undefined;
}

/** A configuration problem the UI should surface as a clear message, not a 500 stack. */
export class ObsConfigError extends Error {}

function s3Config(): S3Config {
  const bucket = firstEnv("MEMLAKE_OBS_S3_BUCKET", "MEMLAKE_QUERY_S3_BUCKET");
  if (!bucket) {
    throw new ObsConfigError(
      "no S3 bucket configured — set MEMLAKE_OBS_S3_BUCKET (or MEMLAKE_QUERY_S3_BUCKET) so the admin can read _obs/",
    );
  }
  return {
    bucket,
    region: firstEnv("MEMLAKE_OBS_S3_REGION", "MEMLAKE_QUERY_S3_REGION") ?? "us-east-1",
    endpoint: firstEnv("MEMLAKE_OBS_S3_ENDPOINT", "MEMLAKE_QUERY_S3_ENDPOINT"),
    accessKeyId: firstEnv("MEMLAKE_OBS_S3_ACCESS_KEY", "MEMLAKE_QUERY_S3_ACCESS_KEY"),
    secretAccessKey: firstEnv("MEMLAKE_OBS_S3_SECRET_KEY", "MEMLAKE_QUERY_S3_SECRET_KEY"),
  };
}

// Memoize on globalThis (the Next.js dev-reload pattern) so hot reloads don't leak a client.
const g = globalThis as unknown as { __obsS3?: { client: S3Client; bucket: string } };

function client(): { client: S3Client; bucket: string } {
  if (g.__obsS3) return g.__obsS3;
  const cfg = s3Config();
  const s3 = new S3Client({
    region: cfg.region,
    endpoint: cfg.endpoint,
    // MinIO and other S3-compatibles need path-style addressing.
    forcePathStyle: Boolean(cfg.endpoint),
    credentials:
      cfg.accessKeyId && cfg.secretAccessKey
        ? { accessKeyId: cfg.accessKeyId, secretAccessKey: cfg.secretAccessKey }
        : undefined,
  });
  g.__obsS3 = { client: s3, bucket: cfg.bucket };
  return g.__obsS3;
}

// ---- primitives -------------------------------------------------------------

async function listKeys(prefix: string): Promise<{ key: string; size: number }[]> {
  const { client: s3, bucket } = client();
  const out: { key: string; size: number }[] = [];
  let token: string | undefined;
  do {
    const page = await s3.send(
      new ListObjectsV2Command({ Bucket: bucket, Prefix: prefix, ContinuationToken: token }),
    );
    for (const o of page.Contents ?? []) {
      if (o.Key) out.push({ key: o.Key, size: o.Size ?? 0 });
    }
    token = page.IsTruncated ? page.NextContinuationToken : undefined;
  } while (token);
  return out;
}

async function getText(key: string): Promise<string> {
  const { client: s3, bucket } = client();
  const res = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  return (await res.Body?.transformToString()) ?? "";
}

/** `_obs/traces/{node}/{ms}-{seq}.jsonl` → the node id and the flush timestamp. */
function parseBatchKey(key: string): { node: string; ms: number } | null {
  const rest = key.slice(OBS_TRACES_PREFIX.length);
  const slash = rest.lastIndexOf("/");
  if (slash < 0) return null;
  const node = rest.slice(0, slash);
  const file = rest.slice(slash + 1);
  const ms = Number(file.split("-")[0]);
  if (!Number.isFinite(ms)) return null;
  return { node, ms };
}

/** The newest batch objects, newest-first. Bounded so we never read the whole retention window. */
async function newestBatches(
  maxObjects: number,
  nodeFilter?: string,
): Promise<{ key: string; node: string; ms: number }[]> {
  const prefix = nodeFilter ? `${OBS_TRACES_PREFIX}${nodeFilter}/` : OBS_TRACES_PREFIX;
  const keys = await listKeys(prefix);
  const parsed = keys
    .filter((k) => k.key.endsWith(".jsonl"))
    .map((k) => {
      const p = parseBatchKey(k.key);
      return p ? { key: k.key, node: p.node, ms: p.ms } : null;
    })
    .filter((x): x is { key: string; node: string; ms: number } => x !== null);
  parsed.sort((a, b) => b.ms - a.ms);
  return parsed.slice(0, maxObjects);
}

function parseLines(body: string): TraceRecord[] {
  const out: TraceRecord[] = [];
  for (const line of body.split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line) as TraceRecord);
    } catch {
      // A partially-written line; skip it.
    }
  }
  return out;
}

/** Read the newest batches and return their records, each tagged with the serving node. */
async function readRecent(
  maxObjects: number,
  nodeFilter?: string,
): Promise<(TraceRecord & { node_id: string })[]> {
  const batches = await newestBatches(maxObjects, nodeFilter);
  const chunks = await Promise.all(
    batches.map(async (b) => {
      const body = await getText(b.key).catch(() => "");
      return parseLines(body).map((r) => ({ ...r, node_id: b.node }));
    }),
  );
  const all = chunks.flat();
  all.sort((a, b) => (b.ts_ms ?? 0) - (a.ts_ms ?? 0));
  return all;
}

// ---- public reads -----------------------------------------------------------

/** Every node's rollup — the fleet overview. Newest-heartbeat first. */
export async function listNodes(): Promise<NodeSummary[]> {
  const now = Date.now();
  const keys = (await listKeys(OBS_ROLLUP_PREFIX)).filter((k) => k.key.endsWith(".json"));
  const nodes: NodeSummary[] = [];
  await Promise.all(
    keys.map(async (k) => {
      const body = await getText(k.key).catch(() => "");
      if (!body.trim()) return;
      try {
        const header = JSON.parse(body) as NodeHeader;
        if (header.node_id) nodes.push({ header, sizeBytes: k.size, fetchedMs: now });
      } catch {
        // A rollup mid-write; skip this pass.
      }
    }),
  );
  nodes.sort((a, b) => b.header.updated_ms - a.header.updated_ms);
  return nodes;
}

/** Recent trace summaries across all nodes, newest first, capped. */
export async function traceSummaries(limit: number): Promise<TraceSummary[]> {
  const recs = await readRecent(LIST_BATCH_OBJECTS);
  return recs.slice(0, limit).map((r) => ({
    id: String(r.id ?? ""),
    node_id: r.node_id,
    namespace: r.namespace ?? "",
    op: r.op ?? "",
    total_ms: r.total_ms ?? 0,
    ts_ms: r.ts_ms ?? 0,
    snapshot: r.snapshot,
  }));
}

/** The full record (with spans) for one trace id, scanning newest batches first. */
export async function traceById(id: string): Promise<(TraceRecord & { node_id: string }) | null> {
  const batches = await newestBatches(SCAN_BATCH_OBJECTS);
  for (const b of batches) {
    const body = await getText(b.key).catch(() => "");
    for (const r of parseLines(body)) {
      if (String(r.id ?? "") === id) return { ...r, node_id: b.node };
    }
  }
  return null;
}

/** One node's recent records, newest first, capped. */
export async function nodeRecords(nodeId: string, limit: number): Promise<TraceRecord[]> {
  const recs = await readRecent(LIST_BATCH_OBJECTS, nodeId);
  return recs.slice(0, limit);
}

/** Records touching `namespace` across all nodes, merged newest-first, capped. */
export async function namespaceRecords(
  namespace: string,
  limit: number,
): Promise<(TraceRecord & { node_id: string })[]> {
  const recs = await readRecent(SCAN_BATCH_OBJECTS);
  return recs.filter((r) => r.namespace === namespace).slice(0, limit);
}
