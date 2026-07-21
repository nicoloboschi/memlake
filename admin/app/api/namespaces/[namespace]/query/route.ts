import { NextResponse } from "next/server";

import { hitToJson } from "@/lib/convert";
import { EMBED_DIM, embedQuery, embeddingsEnabled } from "@/lib/embed";
import {
  coerceConsistency,
  coerceMemoryTypes,
  coerceTagFilter,
  coerceUint32,
  errorResponse,
  readJson,
} from "@/lib/http";
import { MemlakeError, memlake, tagFilter } from "@/lib/memlake";
import type { QueryJson, QueryRequestBody, QueryVectorSource } from "@/lib/types";
import { floatsToF32le } from "@/lib/vector";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// A cold embedding model download can take a while; give the whole handler room
// but keep the RPC itself on the normal 30s deadline.
const QUERY_DEADLINE_MS = 30_000;

/**
 * Run one 3-way query. Which arms run is decided by the inputs: `vector` drives
 * dense + graph, `text` drives full-text. The server returns RAW per-arm scores
 * and does NO fusion — that happens in the browser (see lib/fusion.ts).
 */
export async function POST(
  req: Request,
  ctx: { params: Promise<{ namespace: string }> },
): Promise<NextResponse> {
  try {
    const { namespace } = await ctx.params;
    const body = await readJson<Partial<QueryRequestBody>>(req);

    const text = typeof body.text === "string" ? body.text : "";
    const vectorMode = body.vectorMode ?? "embed";

    let f32le: Uint8Array | null = null;
    let embedMs: number | null = null;
    let vectorSource: QueryVectorSource = "none";
    let vectorDim: number | null = null;
    let embeddingModel: string | null = null;
    let queryPrefix: string | null = null;

    if (vectorMode === "raw") {
      const values = Array.isArray(body.vector) ? body.vector : [];
      if (values.length === 0) {
        throw new MemlakeError(
          3,
          "INVALID_ARGUMENT",
          "raw vector mode selected but no components were supplied",
        );
      }
      if (values.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
        throw new MemlakeError(
          3,
          "INVALID_ARGUMENT",
          "raw vector must be a JSON array of finite numbers",
        );
      }
      f32le = floatsToF32le(values);
      vectorSource = "raw";
      vectorDim = values.length;
    } else if (vectorMode === "embed") {
      if (!embeddingsEnabled()) {
        throw new MemlakeError(
          3,
          "INVALID_ARGUMENT",
          "embeddings are disabled (MEMLAKE_EMBEDDINGS=off)",
          "unset MEMLAKE_EMBEDDINGS, paste a raw vector, or switch the vector mode to 'none' for a text-only query",
        );
      }
      if (!text.trim()) {
        throw new MemlakeError(
          3,
          "INVALID_ARGUMENT",
          "nothing to embed: the query text is empty",
        );
      }
      const embedded = await embedQuery(text);
      f32le = floatsToF32le(embedded.vector);
      embedMs = embedded.elapsedMs;
      vectorSource = "embedded";
      vectorDim = embedded.dim;
      embeddingModel = embedded.model;
      queryPrefix = embedded.prefix;
      if (embedded.dim !== EMBED_DIM) {
        throw new MemlakeError(
          -1,
          "LOCAL",
          `embedding dim ${embedded.dim} != expected ${EMBED_DIM}`,
        );
      }
    }

    if (!f32le && !text.trim()) {
      throw new MemlakeError(
        3,
        "INVALID_ARGUMENT",
        "a query needs at least one input: a vector (dense + graph arms) or text (BM25 arm)",
      );
    }

    const started = Date.now();
    const res = await memlake.query(
      {
        namespace: decodeURIComponent(namespace),
        memoryTypes: coerceMemoryTypes(body.memoryTypes ?? []),
        // Omit the submessage entirely to skip the dense + graph arms.
        vector: f32le ? { f32le } : undefined,
        text,
        tags: tagFilter(coerceTagFilter(body.tags)),
        vectorTopK: coerceUint32(body.vectorTopK, "vector_top_k"),
        textTopK: coerceUint32(body.textTopK, "text_top_k"),
        graphTopK: coerceUint32(body.graphTopK, "graph_top_k"),
        nprobe: coerceUint32(body.nprobe, "nprobe"),
        consistency: coerceConsistency(body.consistency),
      },
      QUERY_DEADLINE_MS,
    );
    const rpcMs = Date.now() - started;

    const out: QueryJson = {
      hits: (res.hits ?? []).map(hitToJson),
      loadRoundtrips: res.loadRoundtrips ?? 0,
      rpcMs,
      embedMs,
      vectorSource,
      vectorDim,
      embeddingModel,
      queryPrefix,
    };
    return NextResponse.json(out);
  } catch (e) {
    return errorResponse(e, "Query");
  }
}
