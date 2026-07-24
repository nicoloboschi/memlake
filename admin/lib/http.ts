/**
 * Route-handler plumbing: turn anything thrown into the `ApiErrorBody` envelope the client
 * components know how to render. SERVER ONLY.
 *
 * This console reads object storage ONLY — there is no RPC layer left — so a failure is an S3 error,
 * bad input, or a configuration problem. The envelope keeps its shape (code / codeName / message /
 * hint) so `ErrorBanner` renders it unchanged.
 */

import { NextResponse } from "next/server";

import type { ApiErrorBody } from "./types";

/** A failure we can describe to the operator, carrying the HTTP status to answer with. */
export class AdminError extends Error {
  constructor(
    readonly status: number,
    readonly codeName: string,
    message: string,
    readonly hint?: string,
  ) {
    super(message);
  }
}

/** Map an S3/SDK failure onto something an operator can act on. */
function describe(e: unknown, what: string): AdminError {
  if (e instanceof AdminError) return e;

  const name = (e as { name?: string })?.name ?? "";
  const message = e instanceof Error ? e.message : String(e);

  if (name === "NoSuchBucket") {
    return new AdminError(404, "NOT_FOUND", `bucket not found while reading ${what}`, message);
  }
  if (name === "NoSuchKey" || name === "NotFound") {
    return new AdminError(404, "NOT_FOUND", `${what} not found in the bucket`, message);
  }
  if (name === "AccessDenied" || name === "InvalidAccessKeyId" || name === "SignatureDoesNotMatch") {
    return new AdminError(
      403,
      "PERMISSION_DENIED",
      `not authorized to read ${what}`,
      "the admin needs read access to the bucket (MEMLAKE_OBS_S3_* or MEMLAKE_QUERY_S3_*)",
    );
  }
  return new AdminError(502, "UNAVAILABLE", `could not read ${what}`, message);
}

export function errorResponse(e: unknown, what = "object storage"): NextResponse<ApiErrorBody> {
  const err = describe(e, what);
  const body: ApiErrorBody = {
    error: {
      code: err.status,
      codeName: err.codeName,
      message: err.message,
      ...(err.hint ? { hint: err.hint } : {}),
    },
  };
  return NextResponse.json(body, { status: err.status });
}

/** A 400 for input the UI sent that we can reject before touching the bucket. */
export function badRequest(message: string, hint?: string): NextResponse<ApiErrorBody> {
  return errorResponse(new AdminError(400, "INVALID_ARGUMENT", message, hint));
}

/** Parse a JSON body, failing with a readable message instead of a 500. */
export async function readJson<T>(req: Request): Promise<T> {
  try {
    return (await req.json()) as T;
  } catch (e) {
    throw new AdminError(
      400,
      "INVALID_ARGUMENT",
      `malformed JSON body: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
}
