import { NextResponse } from "next/server";

import { decodeObjectToJson } from "@/lib/convert";
import { badRequest, coerceUint32, errorResponse, readJson } from "@/lib/http";
import { memlake } from "@/lib/memlake";
import type { DecodeObjectRequestBody } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_LIMIT = 5000;

/**
 * Decode one stored object into readable JSON. A debugging and teaching tool,
 * not a data access path — reading memories is Get/Scan/Query.
 *
 * `limit` bounds how many items are decoded out of a container: one cluster file
 * can hold thousands of memories, and a 200 MB JSON response helps nobody.
 */
export async function POST(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  const started = Date.now();
  try {
    const { namespace } = await ctx.params;
    const body = await readJson<Partial<DecodeObjectRequestBody>>(req);

    const path = typeof body.path === "string" ? body.path.trim() : "";
    if (!path) {
      return badRequest(
        "path is required",
        "pass a path exactly as ListObjects returned it.",
      );
    }
    const limit = Math.min(coerceUint32(body.limit, "limit"), MAX_LIMIT);

    const res = await memlake.decodeObject({
      namespace: decodeURIComponent(namespace),
      path,
      limit,
    });

    return NextResponse.json(
      decodeObjectToJson(res, path, limit, Date.now() - started),
    );
  } catch (e) {
    return errorResponse(e, "DecodeObject");
  }
}
