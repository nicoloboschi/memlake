/**
 * Browser-side fetch helpers. Every route handler answers either its success
 * shape or `ApiErrorBody`, so this is the one place that distinction is made.
 *
 * Client-safe: no server imports live here.
 */

import { isApiError, type ApiErrorBody } from "./types";

export class ApiError extends Error {
  readonly code: number;
  readonly codeName: string;
  readonly hint?: string;

  constructor(body: ApiErrorBody["error"]) {
    super(body.message);
    this.name = "ApiError";
    this.code = body.code;
    this.codeName = body.codeName;
    this.hint = body.hint;
  }
}

async function parse<T>(res: Response): Promise<T> {
  let body: unknown;
  const text = await res.text();
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError({
      code: -1,
      codeName: "LOCAL",
      message: `HTTP ${res.status}: response was not JSON`,
    });
  }
  if (isApiError(body)) throw new ApiError(body.error);
  if (!res.ok) {
    throw new ApiError({
      code: -1,
      codeName: "LOCAL",
      message: `HTTP ${res.status} ${res.statusText}`,
    });
  }
  return body as T;
}

export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { cache: "no-store", signal });
  return parse<T>(res);
}

export async function postJson<T>(
  url: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
    signal,
  });
  return parse<T>(res);
}

/** Normalise a caught value into something the error banner can render. */
export function describeError(e: unknown): {
  codeName: string;
  message: string;
  hint?: string;
} {
  if (e instanceof ApiError) {
    return { codeName: e.codeName, message: e.message, hint: e.hint };
  }
  if (e instanceof Error) {
    return { codeName: "LOCAL", message: e.message };
  }
  return { codeName: "LOCAL", message: String(e) };
}

export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}
