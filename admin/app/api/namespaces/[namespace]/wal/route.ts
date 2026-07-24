import { NextResponse } from "next/server";

import { errorResponse } from "@/lib/http";
import { listWalObjects } from "@/lib/obs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX = 2000;

/**
 * The retained WAL window, newest sequence first. Sequence comes from the key and size from the
 * listing, so no entry payload is decoded — they are binary, and decoding is the engine's job.
 */
export async function GET(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  try {
    const { namespace } = await ctx.params;
    const limit = Math.min(Number(new URL(req.url).searchParams.get("limit")) || 500, MAX);
    const out = await listWalObjects(decodeURIComponent(namespace), limit);
    return NextResponse.json(out);
  } catch (e) {
    return errorResponse(e, "wal");
  }
}
