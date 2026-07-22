import { NextResponse } from "next/server";

import { listObjectsToJson } from "@/lib/convert";
import { coerceUint32, errorResponse, readJson } from "@/lib/http";
import { memlake } from "@/lib/memlake";
import type { ListObjectsRequestBody } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_LIMIT = 2000;

/**
 * Every object the namespace owns in object storage — the PHYSICAL view, as
 * against Stats, which is the logical one.
 *
 * POST rather than GET because the page token is opaque server state that has
 * no business being URL-encoded into a query string, and because the paging
 * pattern then matches Scan and ListWal exactly.
 */
export async function POST(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  const started = Date.now();
  try {
    const { namespace } = await ctx.params;
    const body = await readJson<Partial<ListObjectsRequestBody>>(req);

    // 0 is meaningful: it asks the server for its own default.
    const limit = Math.min(coerceUint32(body.limit, "limit"), MAX_LIMIT);
    const pageToken = typeof body.pageToken === "string" ? body.pageToken : "";

    const res = await memlake.listObjects({
      namespace: decodeURIComponent(namespace),
      limit,
      pageToken,
    });

    return NextResponse.json(listObjectsToJson(res, Date.now() - started));
  } catch (e) {
    return errorResponse(e, "ListObjects");
  }
}
