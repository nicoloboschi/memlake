import { NextResponse } from "next/server";

import { indexLayoutToJson } from "@/lib/convert";
import {
  coerceConsistency,
  coerceUint32,
  errorResponse,
  readJson,
} from "@/lib/http";
import { MemlakeError, memlake } from "@/lib/memlake";
import type { IndexLayoutRequestBody } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Members carry full embeddings and the browser needs every component to run
 * PCA, so the response size is linear in `member_sample * dim * ~12 bytes` of
 * JSON. 1000 x 384 is already ~5MB; cap it well below anything that would make
 * the page feel broken.
 */
const MAX_MEMBER_SAMPLE = 1000;

/** How k-means partitioned one memory_type. Types are independent indexes. */
export async function POST(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  const started = Date.now();
  try {
    const { namespace } = await ctx.params;
    const body = await readJson<Partial<IndexLayoutRequestBody>>(req);

    const memoryType = coerceUint32(body.memoryType, "memory_type");
    if (memoryType > 255) {
      throw new MemlakeError(
        3,
        "INVALID_ARGUMENT",
        `memory_type ${memoryType} is out of range — each type must fit a u8 (0-255)`,
      );
    }
    const memberSample = Math.min(
      coerceUint32(body.memberSample, "member_sample"),
      MAX_MEMBER_SAMPLE,
    );

    const res = await memlake.indexLayout({
      namespace: decodeURIComponent(namespace),
      memoryType,
      // 0 returns centroids only, which costs the server no object-storage read
      // at all — the centroids are already resident on every query node.
      memberSample,
      consistency: coerceConsistency(body.consistency),
    });

    return NextResponse.json(indexLayoutToJson(res, Date.now() - started));
  } catch (e) {
    return errorResponse(e, "IndexLayout");
  }
}
