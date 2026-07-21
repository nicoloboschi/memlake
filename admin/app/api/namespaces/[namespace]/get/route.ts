import { NextResponse } from "next/server";

import { storedMemoryToJson } from "@/lib/convert";
import { uuidToBytes } from "@/lib/ids";
import { coerceConsistency, errorResponse, readJson } from "@/lib/http";
import { MemlakeError, memlake } from "@/lib/memlake";
import type { GetJson, GetRequestBody } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Fetch memories by id. Ids arrive as UUID strings and go out as 16 raw bytes.
 * A missing or tombstoned id is simply absent from the response — Get is not an
 * existence assertion.
 */
export async function POST(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  const started = Date.now();
  try {
    const { namespace } = await ctx.params;
    const body = await readJson<Partial<GetRequestBody>>(req);

    const rawIds = Array.isArray(body.ids) ? body.ids : [];
    if (rawIds.length === 0) {
      throw new MemlakeError(3, "INVALID_ARGUMENT", "no ids supplied");
    }
    let ids: Uint8Array[];
    try {
      ids = rawIds.map((s) => uuidToBytes(String(s)));
    } catch (e) {
      throw new MemlakeError(
        3,
        "INVALID_ARGUMENT",
        e instanceof Error ? e.message : String(e),
        "ids are 16 raw bytes; paste them as UUIDs (dashes optional)",
      );
    }

    const res = await memlake.get({
      namespace: decodeURIComponent(namespace),
      ids,
      includeVector: Boolean(body.includeVector),
      consistency: coerceConsistency(body.consistency),
    });

    const out: GetJson = {
      // include_vector responses carry the full embedding; keep 16 components
      // for the detail panel and summarise the rest.
      memories: (res.memories ?? []).map((m) => storedMemoryToJson(m, 16)),
      elapsedMs: Date.now() - started,
    };
    return NextResponse.json(out);
  } catch (e) {
    return errorResponse(e, "Get");
  }
}
