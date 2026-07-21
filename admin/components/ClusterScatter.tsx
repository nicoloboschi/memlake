"use client";

/**
 * PCA projection of one memory_type's IVF layout.
 *
 * Colour policy (this is the part that is easy to get wrong):
 *
 * A scatter is an ALL-PAIRS chart — any two marks can end up adjacent — and on
 * that stricter test the documented categorical palette carries three slots, not
 * eight. A k=26 index therefore cannot wear 26 hues: past three they stop being
 * distinguishable, especially under CVD. So the three largest clusters take
 * slots 1-3 and everything else folds into a neutral "other".
 *
 * That is not a consolation prize. The largest clusters are the ones that make
 * probe cost uneven, so "where do the big partitions sit, and do they overlap"
 * is the question the picture should answer. Any other cluster is still
 * reachable: selecting it rings and labels it, and the table below carries every
 * value the chart shows. Selection never RE-COLOURS anything — repainting on
 * filter is the anti-pattern — it only rings, labels, and dims.
 */

import { useId, useMemo, useRef, useState } from "react";

import { groupDigits } from "@/lib/format";
import { shortId } from "@/lib/ids";
import { pca2, pcaStatusMessage, type PcaResult } from "@/lib/pca";
import type { ClusterJson, ClusterMemberJson } from "@/lib/types";

const W = 860;
const H = 520;
const PAD = { top: 16, right: 16, bottom: 46, left: 52 };

const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

/** Centroid radii. Area is proportional to size, so radius goes as sqrt. */
const R_MIN = 5;
const R_MAX = 26;
const R_MEMBER = 2.5;
/** Minimum pointer/focus target, per the interaction spec. */
const HIT_MIN = 12;

const SLOTS = [
  "var(--color-viz-1)",
  "var(--color-viz-2)",
  "var(--color-viz-3)",
] as const;
const OTHER_MEMBER = "var(--color-viz-other)";
const OTHER_CENTROID = "var(--color-viz-other-strong)";
const SURFACE = "var(--color-panel)";

export interface ScatterSelection {
  clusterId: number | null;
  onSelect: (clusterId: number | null) => void;
}

interface Mark {
  kind: "centroid" | "member";
  clusterId: number;
  x: number;
  y: number;
  r: number;
  color: string;
  /** Trained size, for centroids. */
  size?: string;
  tags?: string[];
  hasUntagged?: boolean;
  id?: string;
  text?: string;
}

export function ClusterScatter({
  clusters,
  members,
  dim,
  totalSize,
  selection,
}: {
  clusters: ClusterJson[];
  members: ClusterMemberJson[];
  dim: number;
  totalSize: string;
  selection: ScatterSelection;
}) {
  const gridId = useId();
  const [hover, setHover] = useState<Mark | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // ---- which clusters get a hue (stable: by trained size, not by selection)
  const hueOf = useMemo(() => {
    const ranked = [...clusters]
      .filter((c) => c.centroid.length > 0 || true)
      .sort((a, b) => {
        const d = cmpU64(b.size, a.size);
        return d !== 0 ? d : a.clusterId - b.clusterId;
      })
      .slice(0, SLOTS.length);
    const map = new Map<number, string>();
    ranked.forEach((c, i) => map.set(c.clusterId, SLOTS[i]));
    return map;
  }, [clusters]);

  const topIds = useMemo(() => [...hueOf.keys()], [hueOf]);

  // ---- PCA over centroids + sampled members, jointly, so both live in the
  // same projected space and their positions are comparable.
  const { pca, centroidRows, memberRows } = useMemo(() => {
    const cRows = clusters.filter((c) => c.centroid.length > 0);
    const mRows = members.filter((m) => m.vector.length > 0);
    const vectors = [
      ...cRows.map((c) => c.centroid),
      ...mRows.map((m) => m.vector),
    ];
    return {
      pca: pca2(vectors),
      centroidRows: cRows,
      memberRows: mRows,
    };
  }, [clusters, members]);

  const marks = useMemo<Mark[]>(() => {
    if (pca.points.length === 0) return [];

    const xs = pca.points.map((p) => p.x);
    const ys = pca.points.map((p) => p.y);
    const sx = makeScale(xs, PAD.left, PAD.left + PLOT_W);
    const sy = makeScale(ys, PAD.top + PLOT_H, PAD.top); // y flipped

    let maxSize = 1;
    for (const c of centroidRows) {
      const n = Number(c.size);
      if (Number.isFinite(n) && n > maxSize) maxSize = n;
    }

    const out: Mark[] = [];
    // Members first so centroids paint on top of them.
    memberRows.forEach((m, i) => {
      const p = pca.points[centroidRows.length + i];
      out.push({
        kind: "member",
        clusterId: m.clusterId,
        x: sx(p.x),
        y: sy(p.y),
        r: R_MEMBER,
        color: hueOf.get(m.clusterId) ?? OTHER_MEMBER,
        id: m.id,
        text: m.text,
      });
    });
    centroidRows.forEach((c, i) => {
      const p = pca.points[i];
      const n = Math.max(0, Number(c.size) || 0);
      // AREA proportional to size: r = rMax * sqrt(n / nMax). Scaling the radius
      // instead would exaggerate big clusters quadratically.
      const r = Math.max(R_MIN, R_MAX * Math.sqrt(n / maxSize));
      out.push({
        kind: "centroid",
        clusterId: c.clusterId,
        x: sx(p.x),
        y: sy(p.y),
        r,
        color: hueOf.get(c.clusterId) ?? OTHER_CENTROID,
        size: c.size,
        tags: c.tags,
        hasUntagged: c.hasUntagged,
      });
    });
    return out;
  }, [pca, centroidRows, memberRows, hueOf]);

  const message = pcaStatusMessage(pca);
  const plottable = pca.status === "ok" || pca.status === "rank-one";

  // ---- nearest-mark hover: the pointer only has to be closest, not on target
  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const scale = W / rect.width;
    const px = (e.clientX - rect.left) * scale;
    const py = (e.clientY - rect.top) * scale;

    let best: Mark | null = null;
    let bestD = Infinity;
    for (const m of marks) {
      const d = Math.hypot(m.x - px, m.y - py);
      // Centroids win ties within their own radius; otherwise nearest edge.
      const eff = d - (m.kind === "centroid" ? m.r : 0);
      if (eff < bestD) {
        bestD = eff;
        best = m;
      }
    }
    setHover(bestD <= 28 ? best : null);
  }

  const selected = selection.clusterId;

  return (
    <figure className="m-0">
      <div className="relative">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          role="img"
          aria-label={`PCA projection of ${centroidRows.length} IVF centroids and ${memberRows.length} sampled members`}
          className="block min-h-[320px] select-none"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
          onClick={() => {
            if (hover) {
              selection.onSelect(
                selected === hover.clusterId ? null : hover.clusterId,
              );
            }
          }}
        >
          <defs>
            <pattern
              id={gridId}
              width={PLOT_W / 6}
              height={PLOT_H / 4}
              patternUnits="userSpaceOnUse"
              x={PAD.left}
              y={PAD.top}
            >
              <path
                d={`M ${PLOT_W / 6} 0 L 0 0 0 ${PLOT_H / 4}`}
                fill="none"
                stroke="var(--color-viz-grid)"
                strokeWidth="1"
              />
            </pattern>
          </defs>

          {/* Recessive chrome: solid hairlines, one step off the surface. */}
          <rect
            x={PAD.left}
            y={PAD.top}
            width={PLOT_W}
            height={PLOT_H}
            fill={`url(#${gridId})`}
            stroke="var(--color-viz-axis)"
            strokeWidth="1"
          />

          {plottable && (
            <>
              {marks.map((m, i) => {
                const dim0 = selected !== null && m.clusterId !== selected;
                if (m.kind === "member") {
                  return (
                    <circle
                      key={`m-${i}`}
                      cx={m.x}
                      cy={m.y}
                      r={R_MEMBER}
                      fill={m.color}
                      opacity={dim0 ? 0.18 : 0.8}
                    />
                  );
                }
                const isSel = selected === m.clusterId;
                return (
                  <g key={`c-${m.clusterId}`} opacity={dim0 ? 0.28 : 1}>
                    {/* 2px surface ring so overlapping centroids stay legible */}
                    <circle
                      cx={m.x}
                      cy={m.y}
                      r={m.r}
                      fill={m.color}
                      fillOpacity={0.42}
                      stroke={SURFACE}
                      strokeWidth={2}
                    />
                    <circle
                      cx={m.x}
                      cy={m.y}
                      r={m.r}
                      fill="none"
                      stroke={m.color}
                      strokeWidth={2}
                    />
                    {isSel && (
                      <circle
                        cx={m.x}
                        cy={m.y}
                        r={m.r + 5}
                        fill="none"
                        stroke="var(--color-ink)"
                        strokeWidth={1.5}
                      />
                    )}
                  </g>
                );
              })}

              {/* Selective direct labels: the three hued clusters and the
                  current selection only — never a label on every mark. */}
              {marks
                .filter(
                  (m) =>
                    m.kind === "centroid" &&
                    (topIds.includes(m.clusterId) || m.clusterId === selected),
                )
                .map((m) => (
                  <text
                    key={`l-${m.clusterId}`}
                    x={m.x}
                    y={m.y - m.r - 6}
                    textAnchor="middle"
                    className="fill-ink-dim"
                    style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
                  >
                    {m.clusterId}
                  </text>
                ))}

              {/* Keyboard parity: every centroid is focusable and announces the
                  same thing the tooltip shows. */}
              {marks
                .filter((m) => m.kind === "centroid")
                .map((m) => (
                  <circle
                    key={`h-${m.clusterId}`}
                    cx={m.x}
                    cy={m.y}
                    r={Math.max(m.r, HIT_MIN)}
                    fill="transparent"
                    tabIndex={0}
                    role="button"
                    aria-label={`cluster ${m.clusterId}, ${m.size} members${
                      m.tags?.length ? `, tags ${m.tags.join(" ")}` : ""
                    }`}
                    onFocus={() => setHover(m)}
                    onBlur={() => setHover(null)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        selection.onSelect(
                          selected === m.clusterId ? null : m.clusterId,
                        );
                      }
                    }}
                    style={{ cursor: "pointer", outlineOffset: 2 }}
                  />
                ))}
            </>
          )}

          {/* Axis labels carry the honesty of the chart. */}
          <text
            x={PAD.left + PLOT_W / 2}
            y={H - 12}
            textAnchor="middle"
            className="fill-ink-dim"
            style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
          >
            {plottable
              ? `PC1 · ${pct(pca.explained[0])} of variance`
              : "PC1 · unavailable"}
          </text>
          <text
            transform={`translate(16 ${PAD.top + PLOT_H / 2}) rotate(-90)`}
            textAnchor="middle"
            className="fill-ink-dim"
            style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
          >
            {plottable
              ? `PC2 · ${pct(pca.explained[1])} of variance`
              : "PC2 · unavailable"}
          </text>

          {!plottable && (
            <text
              x={W / 2}
              y={H / 2}
              textAnchor="middle"
              className="fill-ink-faint"
              style={{ fontSize: 13, fontFamily: "var(--font-mono)" }}
            >
              no projection
            </text>
          )}
        </svg>

        {hover && plottable && <Tooltip mark={hover} totalSize={totalSize} />}
      </div>

      {/* Legend: identity is never colour-alone — every entry is named. */}
      <div className="mt-2 flex items-center gap-x-4 gap-y-1 flex-wrap px-1">
        {topIds.map((id, i) => (
          <LegendItem
            key={id}
            color={SLOTS[i]}
            label={`cluster ${id}${i === 0 ? " (largest)" : ""}`}
          />
        ))}
        {clusters.length > topIds.length && (
          <LegendItem
            color={OTHER_CENTROID}
            label={`other clusters (${clusters.length - topIds.length})`}
          />
        )}
        <span className="font-mono text-[10px] text-ink-faint">
          large mark = centroid, area ∝ trained size · small mark = sampled
          member
        </span>
      </div>

      <figcaption className="mt-2 px-1 text-[11px] text-ink-faint leading-relaxed">
        {message ? (
          <span className="text-warn">{message}. </span>
        ) : null}
        {plottable && (
          <>
            A lossy projection of {dim} dimensions onto 2, computed in this
            browser. PC1 and PC2 together carry{" "}
            <strong className="text-ink-dim">
              {pct(pca.explained[0] + pca.explained[1])}
            </strong>{" "}
            of the total variance
            {pca.explained[0] + pca.explained[1] < 0.3 && (
              <>
                {" "}
                — which is little, so read this as a hint of structure and
                nothing more
              </>
            )}
            . Distances on screen approximate distances in the real space; two
            centroids that look close may not be, and the IVF probe never uses
            these coordinates.
          </>
        )}
      </figcaption>
    </figure>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        aria-hidden
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ background: color }}
      />
      {/* Text wears text tokens, never the series colour. */}
      <span className="font-mono text-[11px] text-ink-dim">{label}</span>
    </span>
  );
}

function Tooltip({ mark, totalSize }: { mark: Mark; totalSize: string }) {
  // Positioned in percentage of the SVG's own box so it tracks under scaling.
  const left = `${(mark.x / W) * 100}%`;
  const top = `${(mark.y / H) * 100}%`;
  return (
    <div
      className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full
        border border-line-strong bg-panel-2 rounded-sm px-2 py-1.5 shadow-lg max-w-[18rem]"
      style={{ left, top, marginTop: -10 }}
    >
      <div className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="inline-block h-2 w-4 rounded-full"
          style={{ background: mark.color }}
        />
        {/* Value leads, label follows. */}
        <span className="font-mono text-[12px] text-ink">
          {mark.kind === "centroid"
            ? groupDigits(mark.size ?? "0")
            : shortId(mark.id ?? "")}
        </span>
        <span className="font-mono text-[10px] text-ink-faint">
          {mark.kind === "centroid" ? "members" : "member"}
        </span>
      </div>
      <div className="font-mono text-[11px] text-ink-dim mt-0.5">
        cluster {mark.clusterId}
        {mark.kind === "centroid" && (
          <span className="text-ink-faint">
            {" · "}
            {sharePct(mark.size ?? "0", totalSize)} of corpus
          </span>
        )}
      </div>
      {mark.kind === "centroid" && (mark.tags?.length || mark.hasUntagged) && (
        <div className="mt-1 text-[10px] text-ink-faint break-words">
          {mark.tags?.slice(0, 6).join(" · ")}
          {(mark.tags?.length ?? 0) > 6 && ` +${(mark.tags?.length ?? 0) - 6}`}
          {mark.hasUntagged && (
            <span className="text-ink-faint"> · has untagged</span>
          )}
        </div>
      )}
      {mark.kind === "member" && mark.text && (
        <div className="mt-1 text-[10px] text-ink-dim break-words">
          {mark.text.length > 120 ? `${mark.text.slice(0, 119)}…` : mark.text}
        </div>
      )}
    </div>
  );
}

// ---- helpers ----------------------------------------------------------------

function makeScale(values: number[], out0: number, out1: number) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo < 1e-12) {
    // A degenerate axis maps everything to the middle rather than dividing by 0.
    const mid = (out0 + out1) / 2;
    return () => mid;
  }
  // 4% breathing room so marks never sit on the frame.
  const pad = (hi - lo) * 0.04;
  lo -= pad;
  hi += pad;
  return (v: number) => out0 + ((v - lo) / (hi - lo)) * (out1 - out0);
}

function pct(fraction: number): string {
  if (!Number.isFinite(fraction)) return "—";
  const p = fraction * 100;
  if (p > 0 && p < 0.1) return "<0.1%";
  return `${p.toFixed(p < 10 ? 1 : 0)}%`;
}

export function sharePct(size: string, total: string): string {
  try {
    const t = BigInt(total);
    if (t === 0n) return "—";
    // Scale before dividing: BigInt division truncates.
    const share = Number((BigInt(size) * 10000n) / t) / 100;
    return `${share.toFixed(share < 10 ? 1 : 0)}%`;
  } catch {
    return "—";
  }
}

export function cmpU64(a: string, b: string): number {
  try {
    const x = BigInt(a);
    const y = BigInt(b);
    return x < y ? -1 : x > y ? 1 : 0;
  } catch {
    return 0;
  }
}

export type { PcaResult };
