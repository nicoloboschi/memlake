import { NextResponse } from "next/server";

import { cacheStatsToJson } from "@/lib/convert";
import { coerceUint32, errorResponse, readJson } from "@/lib/http";
import { memlake } from "@/lib/memlake";
import type { CacheStatsRequestBody } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** The proto's documented ceiling; the server default (100) applies at 0. */
const MAX_LIMIT = 2000;

/**
 * What THIS replica is holding in its two-tier read cache.
 *
 * Unlike every other route here, the answer is process-local: it describes
 * whichever pod the call landed on, so two calls behind a load balancer will
 * disagree. Nothing it reports affects correctness — a cold cache costs latency
 * only (INV-4). The page says both of those things out loud.
 *
 * One route serves both the global view and the per-namespace one; an empty
 * `namespace` means every namespace, which is the request field's own contract.
 */
export async function POST(req: Request): Promise<NextResponse> {
  const started = Date.now();
  try {
    const body = await readJson<Partial<CacheStatsRequestBody>>(req);

    const namespace =
      typeof body.namespace === "string" ? body.namespace.trim() : "";
    const limit = Math.min(coerceUint32(body.limit, "limit"), MAX_LIMIT);

    const res = await memlake.cacheStats({ namespace, limit });
    return NextResponse.json(
      cacheStatsToJson(res, namespace, limit, Date.now() - started),
    );
  } catch (e) {
    return errorResponse(e, "CacheStats");
  }
}
