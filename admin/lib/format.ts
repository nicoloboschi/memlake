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

export function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}…`;
}
