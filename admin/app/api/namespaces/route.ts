import { NextResponse } from "next/server";

import { errorResponse, readJson } from "@/lib/http";
import { MemlakeError, memlake } from "@/lib/memlake";
import type { CreateNamespaceJson, ListNamespacesJson } from "@/lib/types";

// gRPC must never run at build time: this route is always evaluated per request.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Every namespace in the bucket. One LIST on the server — an operator call. */
export async function GET(): Promise<NextResponse> {
  const started = Date.now();
  try {
    const res = await memlake.listNamespaces();
    const body: ListNamespacesJson = {
      namespaces: [...(res.namespaces ?? [])].sort(),
      elapsedMs: Date.now() - started,
    };
    return NextResponse.json(body);
  } catch (e) {
    return errorResponse(e, "ListNamespaces");
  }
}

/** CreateNamespace is idempotent — the only mutation this admin tool performs. */
export async function POST(req: Request): Promise<NextResponse> {
  const started = Date.now();
  try {
    const body = await readJson<{ namespace?: unknown }>(req);
    const namespace =
      typeof body.namespace === "string" ? body.namespace.trim() : "";
    if (!namespace) {
      throw new MemlakeError(3, "INVALID_ARGUMENT", "namespace must be a non-empty string");
    }
    await memlake.createNamespace({ namespace });
    const out: CreateNamespaceJson = { namespace, elapsedMs: Date.now() - started };
    return NextResponse.json(out);
  } catch (e) {
    return errorResponse(e, "CreateNamespace");
  }
}
