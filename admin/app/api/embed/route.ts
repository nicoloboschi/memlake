import { NextResponse } from "next/server";

import { embedStatus, warmup } from "@/lib/embed";
import { errorResponse } from "@/lib/http";
import type { EmbedStatusJson } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Whether the embedding model is configured and whether it is loaded yet. */
export async function GET(): Promise<NextResponse> {
  const body: EmbedStatusJson = embedStatus();
  return NextResponse.json(body);
}

/**
 * Warm the model so the first real query does not look hung. The ONNX weights
 * are ~90MB on a cold cache, so this can take a while — the UI shows it.
 */
export async function POST(): Promise<NextResponse> {
  try {
    await warmup();
    return NextResponse.json(embedStatus());
  } catch (e) {
    return errorResponse(e, "embed warmup");
  }
}
