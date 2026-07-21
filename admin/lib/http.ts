/**
 * Route-handler plumbing: turn anything thrown into the `ApiErrorBody` envelope
 * the client components know how to render. SERVER ONLY.
 */

import { NextResponse } from "next/server";

import { MemlakeError, toMemlakeError } from "./memlake";
import {
  CONSISTENCIES,
  TAGS_MATCHES,
  type ApiErrorBody,
  type Consistency,
  type TagFilterInput,
  type TagsMatch,
} from "./types";

/** gRPC status -> the closest HTTP status, so devtools/logs stay legible. */
function httpStatusFor(code: number): number {
  switch (code) {
    case 3: // INVALID_ARGUMENT
      return 400;
    case 5: // NOT_FOUND
      return 404;
    case 6: // ALREADY_EXISTS
      return 409;
    case 7: // PERMISSION_DENIED
      return 403;
    case 4: // DEADLINE_EXCEEDED
      return 504;
    case 12: // UNIMPLEMENTED
      return 501;
    case 14: // UNAVAILABLE
      return 503;
    case -1: // local failure (bad input, proto load, embedding)
      return 500;
    default:
      return 502;
  }
}

export function errorResponse(e: unknown, rpc = "rpc"): NextResponse<ApiErrorBody> {
  const err = toMemlakeError(e, rpc);
  const body: ApiErrorBody = {
    error: {
      code: err.code,
      codeName: err.codeName,
      message: err.message,
      ...(err.hint ? { hint: err.hint } : {}),
    },
  };
  return NextResponse.json(body, { status: httpStatusFor(err.code) });
}

/** A 400 for input the UI sent that we can reject without touching the server. */
export function badRequest(message: string, hint?: string): NextResponse<ApiErrorBody> {
  return errorResponse(new MemlakeError(3, "INVALID_ARGUMENT", message, hint));
}

/** Parse a JSON body, failing with a readable message instead of a 500. */
export async function readJson<T>(req: Request): Promise<T> {
  try {
    return (await req.json()) as T;
  } catch (e) {
    throw new MemlakeError(
      3,
      "INVALID_ARGUMENT",
      `malformed JSON body: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
}

// ---- input coercion ---------------------------------------------------------

/** memory_type is a uint32 on the wire but must fit a u8. */
export function coerceMemoryTypes(input: unknown): number[] {
  if (!Array.isArray(input)) return [];
  const out: number[] = [];
  for (const v of input) {
    const n = typeof v === "number" ? v : Number(v);
    if (!Number.isInteger(n) || n < 0 || n > 255) {
      throw new MemlakeError(
        3,
        "INVALID_ARGUMENT",
        `memory_type ${String(v)} is out of range — each type must fit a u8 (0-255)`,
      );
    }
    out.push(n);
  }
  return Array.from(new Set(out)).sort((a, b) => a - b);
}

export function coerceConsistency(input: unknown): Consistency {
  if (typeof input === "string") {
    const upper = input.toUpperCase() as Consistency;
    if ((CONSISTENCIES as readonly string[]).includes(upper)) return upper;
  }
  return "STRONG";
}

export function coerceTagFilter(input: unknown): TagFilterInput | null {
  if (typeof input !== "object" || input === null) return null;
  const raw = input as { tags?: unknown; mode?: unknown };
  const tags = Array.isArray(raw.tags)
    ? raw.tags.filter((t): t is string => typeof t === "string" && t.trim() !== "")
    : [];
  if (tags.length === 0) return null;
  const mode =
    typeof raw.mode === "string" &&
    (TAGS_MATCHES as readonly string[]).includes(raw.mode.toUpperCase())
      ? (raw.mode.toUpperCase() as TagsMatch)
      : "ANY";
  return { tags: tags.map((t) => t.trim()), mode };
}

/**
 * An optional int64, kept as a decimal string all the way to the wire so a
 * nanosecond epoch does not lose precision through a JS number. Returns null
 * for "not supplied".
 */
export function coerceInt64(input: unknown, name: string): string | null {
  if (input === undefined || input === null || input === "") return null;
  const s = typeof input === "string" ? input.trim() : String(input);
  if (!/^-?\d+$/.test(s)) {
    throw new MemlakeError(
      3,
      "INVALID_ARGUMENT",
      `${name} must be an integer epoch value, got ${JSON.stringify(String(input))}`,
    );
  }
  try {
    BigInt(s);
  } catch {
    throw new MemlakeError(3, "INVALID_ARGUMENT", `${name} does not fit an int64`);
  }
  return s;
}

export function coerceUint32(input: unknown, name: string): number {
  if (input === undefined || input === null || input === "") return 0;
  const n = typeof input === "number" ? input : Number(input);
  if (!Number.isInteger(n) || n < 0 || n > 0xffffffff) {
    throw new MemlakeError(
      3,
      "INVALID_ARGUMENT",
      `${name} must be a non-negative 32-bit integer, got ${String(input)}`,
    );
  }
  return n;
}
