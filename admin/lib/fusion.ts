/**
 * CLIENT-SIDE fusion.
 *
 * memlake deliberately does no fusion: `Query` returns each candidate's raw
 * per-arm signal (dense cosine, BM25, graph activation) and the caller decides
 * how to combine them. This module is that decision — it runs in the browser,
 * not on the server, and the UI says so.
 *
 * Reciprocal Rank Fusion:  score(d) = Σ_arms w_arm / (k + rank_arm(d) + 1)
 * Arms where `present` is false contribute nothing at all — which is not the
 * same as contributing a zero score.
 */

import type { Arm, HitJson } from "./types";
import { ARMS } from "./types";

export interface ArmWeights {
  dense: number;
  text: number;
  graph: number;
}

export const DEFAULT_WEIGHTS: ArmWeights = { dense: 1, text: 1, graph: 1 };
export const DEFAULT_RRF_K = 60;

export interface FusedHit {
  hit: HitJson;
  /** Total RRF score. */
  score: number;
  /** Per-arm contribution, for showing where the score came from. */
  contributions: Record<Arm, number>;
  /** 1-based position after fusion, within its memory_type group. */
  rank: number;
}

export function rrfScore(
  hit: HitJson,
  weights: ArmWeights,
  k: number,
): { score: number; contributions: Record<Arm, number> } {
  const contributions = { dense: 0, text: 0, graph: 0 } as Record<Arm, number>;
  let score = 0;
  for (const arm of ARMS) {
    const a = hit[arm];
    if (!a.present) continue; // absent arm: no contribution, not a zero score
    const w = weights[arm];
    if (!w) continue;
    const c = w / (k + a.rank + 1);
    contributions[arm] = c;
    score += c;
  }
  return { score, contributions };
}

export type SortMode = "rrf" | Arm;

/**
 * Order a group of hits. `mode === "rrf"` uses the fused score; any arm name
 * sorts by that arm alone, with absent hits pushed to the bottom (they were
 * never ranked by that arm — they are not "worst", they are "not applicable").
 */
export function rankHits(
  hits: HitJson[],
  weights: ArmWeights,
  k: number,
  mode: SortMode,
): FusedHit[] {
  const scored = hits.map((hit) => {
    const { score, contributions } = rrfScore(hit, weights, k);
    return { hit, score, contributions, rank: 0 };
  });

  if (mode === "rrf") {
    scored.sort((a, b) => b.score - a.score || compareId(a.hit, b.hit));
  } else {
    scored.sort((a, b) => {
      const x = a.hit[mode];
      const y = b.hit[mode];
      if (x.present !== y.present) return x.present ? -1 : 1;
      if (!x.present) return compareId(a.hit, b.hit);
      // Rank is the arm's own ordering — trust it over the raw score, which is
      // not comparable across arms anyway.
      return x.rank - y.rank || compareId(a.hit, b.hit);
    });
  }

  scored.forEach((s, i) => {
    s.rank = i + 1;
  });
  return scored;
}

function compareId(a: HitJson, b: HitJson): number {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * Split hits by memory_type. Types are INDEPENDENT indexes and the server never
 * fuses across them, so neither do we — every group is ranked on its own.
 */
export function groupByMemoryType(hits: HitJson[]): Map<number, HitJson[]> {
  const groups = new Map<number, HitJson[]>();
  for (const h of hits) {
    const g = groups.get(h.memoryType);
    if (g) g.push(h);
    else groups.set(h.memoryType, [h]);
  }
  return new Map([...groups.entries()].sort((a, b) => a[0] - b[0]));
}
