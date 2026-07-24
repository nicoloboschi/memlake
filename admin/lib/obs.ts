/**
 * Observability data source. SERVER ONLY.
 *
 * Serve pods each upload a bounded, slow-biased ring of recent traces to
 * `_obs/traces/{node_id}.jsonl` in the object store (see the server's
 * `crate::trace::TraceRing`). The admin reads that prefix DIRECTLY from S3 —
 * one object per node — so it renders a fleet-wide view without scraping
 * individual pods, which a load-balanced gRPC Service makes unreliable.
 *
 * Each object is JSONL: line 1 is a header (node id, heartbeat, rollups), the
 * rest are raw trace records (each carrying `namespace`, so the same objects
 * answer both "per node" and "per namespace" questions).
 *
 * Config comes from `MEMLAKE_OBS_S3_*`, falling back to the `MEMLAKE_QUERY_S3_*`
 * block the serve pods already mount (same bucket), so a deploy needs no new
 * secret — just grant the admin read access to the bucket.
 */

import {
  GetObjectCommand,
  ListObjectsV2Command,
  S3Client,
} from "@aws-sdk/client-s3";

export const OBS_TRACES_PREFIX = "_obs/traces/";

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
  records: number;
}

export type TraceRecord = {
  op?: string;
  namespace?: string;
  total_ms?: number;
  ts_ms?: number;
  snapshot?: { action?: string; open_ms?: number; tail_entries?: number };
  [k: string]: unknown;
};

export interface NodeSummary {
  header: NodeHeader;
  /** Bytes of the node's trace object — a rough gauge of how full its ring is. */
  sizeBytes: number;
  /** When the admin fetched this, so the UI can age the heartbeat consistently. */
  fetchedMs: number;
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

function s3Config(): S3Config {
  const bucket = firstEnv("MEMLAKE_OBS_S3_BUCKET", "MEMLAKE_QUERY_S3_BUCKET");
  if (!bucket) {
    throw new ObsConfigError(
      "no S3 bucket configured — set MEMLAKE_OBS_S3_BUCKET (or MEMLAKE_QUERY_S3_BUCKET) so the admin can read _obs/traces/",
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

/** A configuration problem the UI should surface as a clear message, not a 500 stack. */
export class ObsConfigError extends Error {}

// Memoize the client + bucket on globalThis (the Next.js dev-reload pattern) so
// hot reloads don't leak a client per edit.
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

// ---- reads ------------------------------------------------------------------

interface RawObject {
  key: string;
  size: number;
  body: string;
}

/** LIST `_obs/traces/` and GET each node object. One object per serve node. */
async function fetchNodeObjects(): Promise<RawObject[]> {
  const { client: s3, bucket } = client();
  const keys: { key: string; size: number }[] = [];
  let token: string | undefined;
  do {
    const page = await s3.send(
      new ListObjectsV2Command({
        Bucket: bucket,
        Prefix: OBS_TRACES_PREFIX,
        ContinuationToken: token,
      }),
    );
    for (const o of page.Contents ?? []) {
      if (o.Key && o.Key.endsWith(".jsonl")) {
        keys.push({ key: o.Key, size: o.Size ?? 0 });
      }
    }
    token = page.IsTruncated ? page.NextContinuationToken : undefined;
  } while (token);

  return Promise.all(
    keys.map(async ({ key, size }) => {
      const res = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
      const body = (await res.Body?.transformToString()) ?? "";
      return { key, size, body };
    }),
  );
}

function parseHeader(body: string): NodeHeader | null {
  const nl = body.indexOf("\n");
  const first = nl === -1 ? body : body.slice(0, nl);
  if (!first.trim()) return null;
  try {
    const h = JSON.parse(first) as NodeHeader;
    return h.kind === "header" ? h : null;
  } catch {
    return null;
  }
}

function parseRecords(body: string): TraceRecord[] {
  const out: TraceRecord[] = [];
  const lines = body.split("\n");
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line) as TraceRecord);
    } catch {
      // A partially-flushed line; skip it.
    }
  }
  return out;
}

/** Every node's header + object size — the fleet overview. Newest-heartbeat first. */
export async function listNodes(): Promise<NodeSummary[]> {
  const now = Date.now();
  const objs = await fetchNodeObjects();
  const nodes: NodeSummary[] = [];
  for (const o of objs) {
    const header = parseHeader(o.body);
    if (header) nodes.push({ header, sizeBytes: o.size, fetchedMs: now });
  }
  nodes.sort((a, b) => b.header.updated_ms - a.header.updated_ms);
  return nodes;
}

/** One node's recent records, newest first, capped. */
export async function nodeRecords(nodeId: string, limit: number): Promise<TraceRecord[]> {
  const { client: s3, bucket } = client();
  const key = `${OBS_TRACES_PREFIX}${nodeId}.jsonl`;
  const res = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  const body = (await res.Body?.transformToString()) ?? "";
  const recs = parseRecords(body);
  recs.sort((a, b) => (b.ts_ms ?? 0) - (a.ts_ms ?? 0));
  return recs.slice(0, limit);
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

/** Recent trace summaries across ALL nodes, newest first, capped. The Traces explorer filters these
 * by namespace/service client-side; the full record (with spans) is fetched on selection by id. */
export async function traceSummaries(limit: number): Promise<TraceSummary[]> {
  const objs = await fetchNodeObjects();
  const out: TraceSummary[] = [];
  for (const o of objs) {
    const header = parseHeader(o.body);
    const nodeId = header?.node_id ?? o.key;
    for (const r of parseRecords(o.body)) {
      out.push({
        id: String(r.id ?? ""),
        node_id: nodeId,
        namespace: r.namespace ?? "",
        op: r.op ?? "",
        total_ms: r.total_ms ?? 0,
        ts_ms: r.ts_ms ?? 0,
        snapshot: r.snapshot,
      });
    }
  }
  out.sort((a, b) => b.ts_ms - a.ts_ms);
  return out.slice(0, limit);
}

/** The full record (with spans) for one trace id, searched across every node. `null` if it has aged
 * out of every ring. Tagged with the node that served it. */
export async function traceById(id: string): Promise<(TraceRecord & { node_id: string }) | null> {
  const objs = await fetchNodeObjects();
  for (const o of objs) {
    const header = parseHeader(o.body);
    const nodeId = header?.node_id ?? o.key;
    for (const r of parseRecords(o.body)) {
      if (String(r.id ?? "") === id) return { ...r, node_id: nodeId };
    }
  }
  return null;
}

/** Records touching `namespace` across ALL nodes, merged newest-first, capped. Each record is
 * tagged with the node it came from so the debugger sees which pod served it. */
export async function namespaceRecords(
  namespace: string,
  limit: number,
): Promise<(TraceRecord & { node_id: string })[]> {
  const objs = await fetchNodeObjects();
  const merged: (TraceRecord & { node_id: string })[] = [];
  for (const o of objs) {
    const header = parseHeader(o.body);
    const nodeId = header?.node_id ?? o.key;
    for (const r of parseRecords(o.body)) {
      if (r.namespace === namespace) merged.push({ ...r, node_id: nodeId });
    }
  }
  merged.sort((a, b) => (b.ts_ms ?? 0) - (a.ts_ms ?? 0));
  return merged.slice(0, limit);
}
