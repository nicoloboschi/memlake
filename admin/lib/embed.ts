/**
 * Query embedding, server-side. SERVER ONLY.
 *
 * This MUST stay bit-compatible with the benchmark harness
 * (`bench/src/memlake_bench/embed.py`) or admin-UI recall silently diverges
 * from what the corpus was indexed with:
 *
 *   model      BAAI/bge-small-en-v1.5
 *   dim        384, float32
 *   pooling    CLS  (see below)
 *   dtype      fp32 (transformers.js would otherwise pick a quantized ONNX
 *              variant, which drifts from the harness)
 *   output     L2-normalized (so cosine == dot product; nothing renormalizes)
 *   prefix     QUERIES get "Represent this sentence for searching relevant
 *              passages: "; DOCUMENTS get nothing. bge is trained with an
 *              asymmetric instruction and dropping it costs several nDCG points.
 *
 * The admin query box is a query, so it always gets the prefix.
 *
 * On pooling: bge-* are CLS-pooled models (their sentence-transformers config
 * sets pooling_mode_cls_token), and fastembed — which the harness uses — pools
 * on CLS. Measured against fastembed on the same strings:
 *
 *     pooling: "cls"    cosine 0.999999   (max component delta 3e-4, ONNX noise)
 *     pooling: "mean"   cosine 0.94       (max component delta 2.6e-1)
 *
 * Mean pooling produces a *plausible* vector that is simply not the one the
 * corpus was indexed against, so recall degrades silently — exactly the failure
 * this module exists to prevent. If you change this line, re-run that
 * comparison first.
 *
 * The ONNX weights (~90MB) are downloaded and cached by transformers.js on
 * first use. The pipeline is memoized on `globalThis` so a hot reload does not
 * re-load the model.
 */

import type { FeatureExtractionPipeline } from "@huggingface/transformers";

import type { EmbedState, EmbedStatusJson } from "./types";

export const EMBED_MODEL = "BAAI/bge-small-en-v1.5";
export const EMBED_DIM = 384;
export const EMBED_POOLING = "cls" as const;
export const EMBED_DTYPE = "fp32" as const;

/** Exactly the prefix used by the benchmark cache. Do not "tidy" the spacing. */
export const BGE_QUERY_PREFIX =
  "Represent this sentence for searching relevant passages: ";

export class EmbeddingsDisabledError extends Error {
  constructor() {
    super(
      "embeddings are disabled (MEMLAKE_EMBEDDINGS=off) — use the raw-vector mode, or query text-only",
    );
    this.name = "EmbeddingsDisabledError";
  }
}

export function embeddingsEnabled(): boolean {
  return (process.env.MEMLAKE_EMBEDDINGS ?? "").toLowerCase() !== "off";
}

interface EmbedGlobal {
  pipe?: Promise<FeatureExtractionPipeline>;
  state: EmbedState;
  error: string | null;
}

const globalForEmbed = globalThis as typeof globalThis & {
  __memlakeEmbed?: EmbedGlobal;
};

function slot(): EmbedGlobal {
  return (globalForEmbed.__memlakeEmbed ??= { state: "idle", error: null });
}

/**
 * Lazily build (and memoize) the feature-extraction pipeline. The first call
 * pays the model download; every later call is a cache hit.
 */
function getPipeline(): Promise<FeatureExtractionPipeline> {
  const g = slot();
  if (!g.pipe) {
    g.state = "loading";
    g.error = null;
    g.pipe = (async () => {
      const { pipeline } = await import("@huggingface/transformers");
      return pipeline("feature-extraction", EMBED_MODEL, { dtype: EMBED_DTYPE });
    })()
      .then((p) => {
        g.state = "ready";
        return p;
      })
      .catch((e: unknown) => {
        g.state = "error";
        g.error = e instanceof Error ? e.message : String(e);
        // Drop the rejected promise so a later request can retry the download.
        g.pipe = undefined;
        throw e;
      });
  }
  return g.pipe;
}

export interface EmbedResult {
  vector: Float32Array;
  dim: number;
  /** Wall-clock of the embed call, including the model load on a cold start. */
  elapsedMs: number;
  model: string;
  prefix: string;
}

/** Embed a *query* string: prefix, mean-pool, L2-normalize. */
export async function embedQuery(text: string): Promise<EmbedResult> {
  if (!embeddingsEnabled()) throw new EmbeddingsDisabledError();

  const started = Date.now();
  const pipe = await getPipeline();
  const output = await pipe(BGE_QUERY_PREFIX + text, {
    pooling: EMBED_POOLING,
    normalize: true,
  });

  const data = output.data as
    | Float32Array
    | Float64Array
    | Int32Array
    | number[];
  const vector = data instanceof Float32Array ? data : Float32Array.from(data);

  if (vector.length !== EMBED_DIM) {
    throw new Error(
      `embedding model returned dim ${vector.length}, expected ${EMBED_DIM} — the corpus was indexed with ${EMBED_MODEL}`,
    );
  }

  return {
    vector,
    dim: vector.length,
    elapsedMs: Date.now() - started,
    model: EMBED_MODEL,
    prefix: BGE_QUERY_PREFIX,
  };
}

/** Warm the model without running a query (used by the query page on mount). */
export async function warmup(): Promise<void> {
  if (!embeddingsEnabled()) throw new EmbeddingsDisabledError();
  await getPipeline();
}

export function embedStatus(): EmbedStatusJson {
  const enabled = embeddingsEnabled();
  const g = slot();
  return {
    enabled,
    model: EMBED_MODEL,
    dim: EMBED_DIM,
    pooling: EMBED_POOLING,
    queryPrefix: BGE_QUERY_PREFIX,
    state: enabled ? g.state : "disabled",
    error: g.error,
  };
}
