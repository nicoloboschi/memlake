import { NextResponse } from "next/server";

import { errorResponse } from "@/lib/http";
import { listNamespaceNames } from "@/lib/obs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Every namespace in the bucket — read straight from S3.
 *
 * A namespace IS a `{name}/manifest.json` object, which is exactly how the server discovers them,
 * so the bucket is the catalogue and no serve endpoint is involved. This console is read-only:
 * creating a namespace is a write against the engine's own invariants, so it belongs to a client,
 * not to an inspection tool.
 */
export async function GET(): Promise<NextResponse> {
  const started = Date.now();
  try {
    const namespaces = await listNamespaceNames();
    return NextResponse.json({ namespaces, elapsedMs: Date.now() - started });
  } catch (e) {
    return errorResponse(e, "ListNamespaces");
  }
}
