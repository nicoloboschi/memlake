import { NextResponse } from "next/server";

import { errorResponse } from "@/lib/http";
import { readNamespaceState } from "@/lib/obs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * A namespace's index state, read from S3: `manifest.json` plus the live `wal-head` pointer.
 *
 * Everything here is authoritative on-disk state, so it needs no serve node. What it deliberately
 * does NOT report is a live document count — that requires replaying the un-indexed WAL tail, which
 * is the engine's job. The indexed count and the backlog are reported separately instead.
 */
export async function GET(
  _req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  try {
    const { namespace } = await ctx.params;
    const state = await readNamespaceState(decodeURIComponent(namespace));
    return NextResponse.json(state);
  } catch (e) {
    return errorResponse(e, "namespace state");
  }
}
