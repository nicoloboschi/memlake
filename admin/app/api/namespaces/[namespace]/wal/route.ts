import { NextResponse } from "next/server";

import { listWalToJson } from "@/lib/convert";
import {
  coerceInt64,
  coerceUint32,
  errorResponse,
  readJson,
} from "@/lib/http";
import { memlake } from "@/lib/memlake";
import type { ListWalRequestBody } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_LIMIT = 500;

/**
 * The write-ahead log as an operator view.
 *
 * POST rather than GET so `start_seq` (a u64 that must survive as a string) and
 * the include_ops flag travel in a body instead of being stringified through a
 * query param.
 *
 * Note this is a window on the LIVE log, not a history: entries at or below
 * `wal_index_cursor` are folded and may already have been reclaimed by GC.
 */
export async function POST(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  const started = Date.now();
  try {
    const { namespace } = await ctx.params;
    const body = await readJson<Partial<ListWalRequestBody>>(req);

    const startSeq = coerceInt64(body.startSeq, "start_seq") ?? "0";
    const limit = Math.min(coerceUint32(body.limit, "limit"), MAX_LIMIT);

    const res = await memlake.listWal({
      namespace: decodeURIComponent(namespace),
      startSeq,
      limit,
      includeOps: Boolean(body.includeOps),
    });

    return NextResponse.json(listWalToJson(res, Date.now() - started));
  } catch (e) {
    return errorResponse(e, "ListWal");
  }
}
