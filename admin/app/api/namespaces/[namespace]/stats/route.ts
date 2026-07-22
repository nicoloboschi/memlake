import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { statsToJson } from "@/lib/convert";
import { errorResponse } from "@/lib/http";
import { memlake } from "@/lib/memlake";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Index state for one namespace. Reads the manifest and each type's metadata —
 * no cluster data — so its cost is independent of corpus size.
 */
export async function GET(
  _req: NextRequest,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  const started = Date.now();
  try {
    const { namespace } = await ctx.params;
    const res = await memlake.stats({
      namespace: decodeURIComponent(namespace),
    });
    return NextResponse.json(statsToJson(res, Date.now() - started));
  } catch (e) {
    return errorResponse(e, "Stats");
  }
}
