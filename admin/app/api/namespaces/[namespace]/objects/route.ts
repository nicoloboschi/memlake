import { NextResponse } from "next/server";

import { errorResponse } from "@/lib/http";
import { listNamespaceObjects } from "@/lib/obs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX = 2000;

/** Every object under the namespace prefix, largest first. A LIST — nothing is decoded. */
export async function GET(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  try {
    const { namespace } = await ctx.params;
    const limit = Math.min(Number(new URL(req.url).searchParams.get("limit")) || 500, MAX);
    const out = await listNamespaceObjects(decodeURIComponent(namespace), limit);
    return NextResponse.json(out);
  } catch (e) {
    return errorResponse(e, "objects");
  }
}
