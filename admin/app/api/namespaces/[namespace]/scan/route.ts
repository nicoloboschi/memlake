import { NextResponse } from "next/server";

import { storedMemoryToJson } from "@/lib/convert";
import {
  coerceMemoryTypes,
  coerceTagFilter,
  coerceUint32,
  errorResponse,
  readJson,
} from "@/lib/http";
import { memlake, tagFilter } from "@/lib/memlake";
import type { ScanJson, ScanRequestBody } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_LIMIT = 500;

/**
 * Page through a memory_type's stored memories in cluster order.
 *
 * POST rather than GET because the body carries a tag filter and an opaque
 * `page_token` that is not URL-shaped. The token is a *cursor*, valid only
 * against the generation that produced it — never derive page numbers from it.
 */
export async function POST(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  const started = Date.now();
  try {
    const { namespace } = await ctx.params;
    const body = await readJson<Partial<ScanRequestBody>>(req);

    const limit = Math.min(coerceUint32(body.limit, "limit"), MAX_LIMIT);
    const res = await memlake.scan({
      namespace: decodeURIComponent(namespace),
      memoryTypes: coerceMemoryTypes(body.memoryTypes ?? []),
      limit,
      pageToken: typeof body.pageToken === "string" ? body.pageToken : "",
      includeVector: Boolean(body.includeVector),
      tags: tagFilter(coerceTagFilter(body.tags)),
    });

    const out: ScanJson = {
      memories: (res.memories ?? []).map((m) => storedMemoryToJson(m, 8)),
      nextPageToken: res.nextPageToken ?? "",
      elapsedMs: Date.now() - started,
    };
    return NextResponse.json(out);
  } catch (e) {
    return errorResponse(e, "Scan");
  }
}
