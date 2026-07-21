/** Display formatting. Client-safe, pure. */

/** Group a decimal u64 string with thin separators, without going through Number. */
export function groupDigits(s: string): string {
  const neg = s.startsWith("-");
  const digits = neg ? s.slice(1) : s;
  if (!/^\d+$/.test(digits)) return s;
  const out = digits.replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  return neg ? `-${out}` : out;
}

export function fmtScore(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return "—";
  if (n !== 0 && Math.abs(n) < 1e-4) return n.toExponential(2);
  return n.toFixed(digits);
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Timestamps in memlake are opaque int64s — the engine never interprets them.
 * Render the raw value, plus a best-effort date when the magnitude reads like
 * epoch seconds/millis/micros so an operator can sanity-check ingestion.
 */
export function fmtEpochGuess(raw: string): string {
  let n: bigint;
  try {
    n = BigInt(raw);
  } catch {
    return raw;
  }
  const abs = n < 0n ? -n : n;
  let ms: number | null = null;
  if (abs > 0n) {
    if (abs < 100_000_000_000n) ms = Number(n) * 1000; // seconds
    else if (abs < 100_000_000_000_000n) ms = Number(n); // millis
    else if (abs < 100_000_000_000_000_000n) ms = Number(n) / 1000; // micros
  }
  if (ms === null || !Number.isFinite(ms)) return raw;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toISOString().replace("T", " ").replace(".000Z", "Z");
}

/** Compare two u64 decimal strings. They exceed Number range, so go via BigInt. */
export function cmpU64(a: string, b: string): number {
  try {
    const x = BigInt(a);
    const y = BigInt(b);
    return x < y ? -1 : x > y ? 1 : 0;
  } catch {
    return 0;
  }
}

/** `size / total` as a percentage string. BigInt division truncates, so scale first. */
export function sharePct(size: string, total: string): string {
  try {
    const t = BigInt(total);
    if (t === 0n) return "—";
    const share = Number((BigInt(size) * 10000n) / t) / 100;
    return `${share.toFixed(share < 10 ? 1 : 0)}%`;
  } catch {
    return "—";
  }
}

/**
 * Byte sizes arrive as u64 decimal strings (WAL objects, cache entries), so the
 * magnitude test runs on BigInt before anything touches Number.
 */
export function fmtBytes(s: string): string {
  let n: bigint;
  try {
    n = BigInt(s);
  } catch {
    return s;
  }
  if (n < 1024n) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB", "PiB"];
  let v = Number(n);
  let u = -1;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[u]}`;
}

/** `used / budget` as a 0..1 fraction, or null when the budget is 0/unknown. */
export function fraction(used: string, budget: string): number | null {
  try {
    const b = BigInt(budget);
    if (b <= 0n) return null;
    return Number((BigInt(used) * 1000000n) / b) / 1000000;
  } catch {
    return null;
  }
}

export function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}…`;
}
