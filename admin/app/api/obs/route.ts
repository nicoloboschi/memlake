import { NextResponse } from "next/server";

import { errorResponse } from "@/lib/http";
import {
  indexQueue,
  listNodes,
  namespaceRecords,
  nodeRecords,
  ObsConfigError,
  traceById,
  traceSummaries,
} from "@/lib/obs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Cap on records returned when drilling into a node or namespace, to bound the payload. */
const MAX_RECORDS = 1000;

/**
 * The fleet-wide observability read, straight from `_obs/traces/` in the object
 * store (no gRPC, no pod scraping). Three shapes from one route:
 *
 *   GET /api/obs                  -> { nodes: NodeSummary[] }        the overview
 *   GET /api/obs?node=<id>        -> { records: TraceRecord[] }      one node, drilled in
 *   GET /api/obs?namespace=<ns>   -> { records: (…&{node_id})[] }    one namespace, across nodes
 *
 * All three are bounded: each node object is a capped ring, and the drill-ins
 * slice to MAX_RECORDS.
 */
export async function GET(req: Request): Promise<NextResponse> {
  const started = Date.now();
  const url = new URL(req.url);
  const node = url.searchParams.get("node");
  const namespace = url.searchParams.get("namespace");
  const trace = url.searchParams.get("trace");
  const summaries = url.searchParams.get("summaries");
  const limit = Math.min(
    Number(url.searchParams.get("limit")) || 500,
    MAX_RECORDS,
  );

  try {
    if (url.searchParams.get("queue")) {
      const queue = await indexQueue();
      const nodes = await listNodes();
      return NextResponse.json({ queue, nodes, elapsedMs: Date.now() - started });
    }
    if (trace) {
      const record = await traceById(trace);
      return NextResponse.json({ record, elapsedMs: Date.now() - started });
    }
    if (summaries) {
      const traces = await traceSummaries(limit);
      return NextResponse.json({ traces, elapsedMs: Date.now() - started });
    }
    if (node) {
      const records = await nodeRecords(node, limit);
      return NextResponse.json({ records, elapsedMs: Date.now() - started });
    }
    if (namespace) {
      const records = await namespaceRecords(namespace, limit);
      return NextResponse.json({ records, elapsedMs: Date.now() - started });
    }
    const nodes = await listNodes();
    return NextResponse.json({ nodes, elapsedMs: Date.now() - started });
  } catch (e) {
    if (e instanceof ObsConfigError) {
      return NextResponse.json(
        {
          error: {
            code: -1,
            codeName: "FAILED_PRECONDITION",
            message: e.message,
            hint: "Grant the admin S3 read access to the bucket and set MEMLAKE_OBS_S3_* (or reuse MEMLAKE_QUERY_S3_*).",
          },
        },
        { status: 500 },
      );
    }
    return errorResponse(e, "obs");
  }
}
