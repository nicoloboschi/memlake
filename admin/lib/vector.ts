/**
 * `Vector.f32le` is raw little-endian float32 — length = dim * 4 bytes — not a
 * JSON array. These helpers are the only place that encoding is spelled out.
 *
 * DataView rather than Buffer so the module stays isomorphic.
 */

import type { VectorSummary } from "./types";

export const F32_BYTES = 4;

export class VectorFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VectorFormatError";
  }
}

/** float32[] -> little-endian bytes. */
export function floatsToF32le(values: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(values.length * F32_BYTES);
  const view = new DataView(out.buffer);
  for (let i = 0; i < values.length; i++) {
    view.setFloat32(i * F32_BYTES, values[i], /* littleEndian */ true);
  }
  return out;
}

/** little-endian bytes -> float32[]. */
export function f32leToFloats(bytes: Uint8Array): number[] {
  if (bytes.length % F32_BYTES !== 0) {
    throw new VectorFormatError(
      `f32le payload of ${bytes.length} bytes is not a multiple of 4`,
    );
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out: number[] = new Array(bytes.length / F32_BYTES);
  for (let i = 0; i < out.length; i++) {
    out[i] = view.getFloat32(i * F32_BYTES, /* littleEndian */ true);
  }
  return out;
}

/** Number of components in an f32le payload. */
export function f32leDim(bytes: Uint8Array): number {
  return Math.floor(bytes.length / F32_BYTES);
}

/**
 * Reduce a vector to something a table can hold: dim, norm and a short prefix.
 * Never ship 384 raw floats per row to the browser.
 */
export function summarizeF32le(bytes: Uint8Array, headLen = 8): VectorSummary {
  const dim = f32leDim(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let sumsq = 0;
  for (let i = 0; i < dim; i++) {
    const v = view.getFloat32(i * F32_BYTES, true);
    sumsq += v * v;
  }
  const head: number[] = [];
  for (let i = 0; i < Math.min(headLen, dim); i++) {
    head.push(view.getFloat32(i * F32_BYTES, true));
  }
  return { dim, head, norm: Math.sqrt(sumsq), bytes: bytes.length };
}

/**
 * Parse the "paste a raw vector" input: a JSON array of numbers, or plain
 * whitespace/comma-separated numbers. Throws a message worth showing verbatim.
 */
export function parseVectorInput(raw: string): number[] {
  const trimmed = raw.trim();
  if (!trimmed) throw new VectorFormatError("vector is empty");

  let values: unknown;
  if (trimmed.startsWith("[")) {
    try {
      values = JSON.parse(trimmed);
    } catch (e) {
      throw new VectorFormatError(
        `not valid JSON: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  } else {
    values = trimmed
      .split(/[\s,]+/)
      .filter(Boolean)
      .map(Number);
  }

  if (!Array.isArray(values)) {
    throw new VectorFormatError("expected a JSON array of numbers");
  }
  const out = values.map((v, i) => {
    const n = typeof v === "number" ? v : Number(v);
    if (!Number.isFinite(n)) {
      throw new VectorFormatError(`component ${i} is not a finite number: ${String(v)}`);
    }
    return n;
  });
  if (out.length === 0) throw new VectorFormatError("vector has 0 components");
  return out;
}

/** L2 norm, for showing the user whether a pasted vector is normalized. */
export function l2Norm(values: ArrayLike<number>): number {
  let sumsq = 0;
  for (let i = 0; i < values.length; i++) sumsq += values[i] * values[i];
  return Math.sqrt(sumsq);
}
