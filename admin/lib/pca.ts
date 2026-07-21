/**
 * Two-component PCA, computed in the browser.
 *
 * Pure TypeScript on purpose: the top two principal components are covariance +
 * power iteration with one deflation step, which is short enough that pulling in
 * a linear-algebra dependency would cost more than it saves.
 *
 * The covariance matrix is never materialised. For 384-dimensional centroids a
 * d x d matrix is 147k floats; instead each power-iteration step applies
 * X^T X implicitly in O(n*d):
 *
 *     w = sum_i x_i (x_i . v)          // == X^T X v
 *     v = w / |w|
 *
 * which converges to the leading eigenvector of the (unnormalised) covariance.
 * Deflating the data against v1 and repeating gives v2, orthogonal by
 * construction.
 *
 * Two properties the UI depends on:
 *
 *  - **Deterministic.** The starting vector is a fixed pseudo-random sequence,
 *    not Math.random, so the same data always projects to the same picture. A
 *    plot that reshuffles on every render is unreadable.
 *  - **Sign-stable.** Eigenvectors are defined up to sign; we fix it by forcing
 *    the largest-magnitude loading positive, so the scatter does not mirror
 *    itself between refreshes.
 *
 * Client-safe: no imports, no DOM, no Node APIs.
 */

export type PcaStatus =
  | "ok"
  /** Fewer than two input rows — a projection needs at least two points. */
  | "too-few-rows"
  /** dim === 0: this memory_type stores no embeddings. */
  | "no-dimensions"
  /** Every vector is identical (or a single distinct point): variance is 0. */
  | "no-variance"
  /**
   * Rank 1: all points lie on a line, so PC2 carries no variance. The
   * projection is still drawn, but it is honestly one-dimensional.
   */
  | "rank-one";

export interface PcaResult {
  status: PcaStatus;
  /** Projected coordinates, in input row order. Empty unless status is ok/rank-one. */
  points: { x: number; y: number }[];
  /** Fraction (0..1) of total variance carried by PC1 and PC2. */
  explained: [number, number];
  /** Sum of per-dimension variances of the centered input. */
  totalVariance: number;
  /** Rows actually used (input length). */
  n: number;
  dim: number;
}

/** A tiny deterministic PRNG, so the starting vector is stable across renders. */
function seededUnitVector(d: number): Float64Array {
  const v = new Float64Array(d);
  // xorshift32 with a fixed seed; any stable sequence works.
  let s = 0x9e3779b9;
  let norm = 0;
  for (let i = 0; i < d; i++) {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;
    s >>>= 0;
    const x = s / 0xffffffff - 0.5;
    v[i] = x;
    norm += x * x;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < d; i++) v[i] /= norm;
  return v;
}

/**
 * Leading eigenvector of X^T X by power iteration, X given as centered rows.
 * Returns null when the data has no variance in any remaining direction.
 */
function leadingComponent(
  rows: Float64Array[],
  d: number,
  iterations = 128,
  tol = 1e-9,
): Float64Array | null {
  let v = seededUnitVector(d);
  const w = new Float64Array(d);

  for (let it = 0; it < iterations; it++) {
    w.fill(0);
    for (const row of rows) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += row[j] * v[j];
      if (dot === 0) continue;
      for (let j = 0; j < d; j++) w[j] += row[j] * dot;
    }

    let norm = 0;
    for (let j = 0; j < d; j++) norm += w[j] * w[j];
    norm = Math.sqrt(norm);
    if (!Number.isFinite(norm) || norm < 1e-12) return null; // no variance left

    let delta = 0;
    for (let j = 0; j < d; j++) {
      const next = w[j] / norm;
      delta += Math.abs(next - v[j]);
      w[j] = next;
    }
    v = Float64Array.from(w);
    if (delta < tol) break;
  }

  // Sign convention: largest-magnitude loading is positive.
  let maxIdx = 0;
  for (let j = 1; j < d; j++) {
    if (Math.abs(v[j]) > Math.abs(v[maxIdx])) maxIdx = j;
  }
  if (v[maxIdx] < 0) {
    for (let j = 0; j < d; j++) v[j] = -v[j];
  }
  return v;
}

/** Variance of the projection of `rows` onto unit vector `v` (rows are centered). */
function varianceAlong(rows: Float64Array[], v: Float64Array, d: number): number {
  if (rows.length < 2) return 0;
  let sumsq = 0;
  for (const row of rows) {
    let dot = 0;
    for (let j = 0; j < d; j++) dot += row[j] * v[j];
    sumsq += dot * dot;
  }
  return sumsq / (rows.length - 1);
}

/**
 * Project `vectors` (all of the same length) onto their top two principal
 * components. Rows with the wrong length are rejected up front rather than
 * silently mangling the fit.
 */
export function pca2(vectors: readonly number[][]): PcaResult {
  const n = vectors.length;
  const dim = n > 0 ? vectors[0].length : 0;
  const empty: PcaResult = {
    status: "no-dimensions",
    points: [],
    explained: [0, 0],
    totalVariance: 0,
    n,
    dim,
  };

  if (dim === 0) return empty;
  if (n < 2) return { ...empty, status: "too-few-rows" };
  if (vectors.some((v) => v.length !== dim)) {
    return { ...empty, status: "no-dimensions" };
  }

  // --- center
  const mean = new Float64Array(dim);
  for (const v of vectors) {
    for (let j = 0; j < dim; j++) mean[j] += v[j];
  }
  for (let j = 0; j < dim; j++) mean[j] /= n;

  const rows: Float64Array[] = vectors.map((v) => {
    const r = new Float64Array(dim);
    for (let j = 0; j < dim; j++) r[j] = v[j] - mean[j];
    return r;
  });

  // --- total variance = trace of the covariance matrix
  let totalVariance = 0;
  for (const row of rows) {
    for (let j = 0; j < dim; j++) totalVariance += row[j] * row[j];
  }
  totalVariance /= n - 1;

  if (!Number.isFinite(totalVariance) || totalVariance < 1e-18) {
    // Every vector is the same point: PCA is undefined, not "zero".
    return { ...empty, status: "no-variance", totalVariance: 0 };
  }

  // --- PC1
  const v1 = leadingComponent(rows, dim);
  if (!v1) return { ...empty, status: "no-variance", totalVariance };
  const var1 = varianceAlong(rows, v1, dim);

  // --- deflate, then PC2
  for (const row of rows) {
    let dot = 0;
    for (let j = 0; j < dim; j++) dot += row[j] * v1[j];
    for (let j = 0; j < dim; j++) row[j] -= dot * v1[j];
  }
  const v2 = leadingComponent(rows, dim);
  const var2 = v2 ? varianceAlong(rows, v2, dim) : 0;

  // --- project (against the ORIGINAL centered rows, which `rows` no longer is)
  const points = vectors.map((vec) => {
    let x = 0;
    let y = 0;
    for (let j = 0; j < dim; j++) {
      const c = vec[j] - mean[j];
      x += c * v1[j];
      if (v2) y += c * v2[j];
    }
    return { x, y };
  });

  const rankOne = !v2 || var2 / totalVariance < 1e-9;
  return {
    status: rankOne ? "rank-one" : "ok",
    points,
    explained: [var1 / totalVariance, var2 / totalVariance],
    totalVariance,
    n,
    dim,
  };
}

/** Human-readable explanation for a non-plottable result. */
export function pcaStatusMessage(r: PcaResult): string | null {
  switch (r.status) {
    case "no-dimensions":
      return "this memory_type stores no embeddings (dim = 0), so there is nothing to project — the dense arm does not run for it";
    case "too-few-rows":
      return `a projection needs at least two points; this layout has ${r.n}`;
    case "no-variance":
      return "every vector in this layout is identical, so PCA is undefined — there are no directions of variation to project onto";
    case "rank-one":
      return "all points lie on a single line: PC2 carries no variance, so the vertical axis is not meaningful here";
    default:
      return null;
  }
}
