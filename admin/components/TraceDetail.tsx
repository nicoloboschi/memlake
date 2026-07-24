"use client";

import { useState } from "react";

import { fmtBytes, fmtMs, groupDigits } from "@/lib/format";

type Phase = [string, number]; // [name, microseconds]

export interface TraceRec {
  op?: string;
  namespace?: string;
  node_id?: string;
  total_ms?: number;
  ts_ms?: number;
  permit_wait_ms?: number;
  in_flight?: number;
  result_count?: number;
  snapshot?: { action?: string; open_ms?: number; tail_entries?: number };
  phases_us?: Phase[];
  io?: {
    bytes?: number;
    cache_hits?: number;
    cache_misses?: number;
    hit_ratio?: number;
    roundtrips?: number;
    tier?: string;
  };
  params?: Record<string, unknown>;
  // write-only
  attempts?: number;
  seq?: number;
  commit_ms?: number;
  link_ms?: number;
  link_snapshot_ms?: number;
  wait_for_index_ms?: number | null;
  link_io?: {
    bytes?: number;
    cache_hits?: number;
    cache_misses?: number;
    corpus_query_ms?: number;
    within_batch_ms?: number;
    queries?: number;
    roundtrips?: number;
    phases_us?: Phase[];
  };
  // Per-object access spans — the waterfall.
  objects?: {
    items: ObjSpan[];
    dropped: number;
    count: number;
  } | null;
}

type ObjSource = "mem" | "disk" | "s3";
interface ObjSpan {
  key: string;
  src: ObjSource | null;
  kind: "io" | "compute";
  group: string;
  op: string;
  start_us: number;
  end_us: number;
  bytes: number;
}

const SRC_BAR: Record<ObjSource, string> = {
  mem: "bg-ok/80",
  disk: "bg-accent/70",
  s3: "bg-danger/80",
};
const SRC_TEXT: Record<ObjSource, string> = {
  mem: "text-ok",
  disk: "text-accent",
  s3: "text-danger",
};
// Compute (CPU) spans get their own colour, distinct from the three I/O tiers.
const CPU_BAR = "bg-warn/70";
const CPU_TEXT = "text-warn";

/** Draw order for the phase groups; unknown groups fall to the end. */
const GROUP_ORDER = ["snapshot", "recall", "text", "rerank", "graph", "commit", "other"];

/** Keep the tail of a long object key (the informative part — cluster-52.vec — is the end). */
function tailKey(k: string, max = 40): string {
  return k.length > max ? `…${k.slice(-max)}` : k;
}

/** The object-access waterfall: every mem/disk/S3 read+write on one request, positioned by time. */
function SpanRow({ s, maxEnd }: { s: ObjSpan; maxEnd: number }) {
  const dur = (s.end_us - s.start_us) / 1000;
  const left = (s.start_us / maxEnd) * 100;
  const width = Math.max(((s.end_us - s.start_us) / maxEnd) * 100, 0.4);
  const compute = s.kind === "compute";
  const bar = compute ? CPU_BAR : SRC_BAR[s.src ?? "s3"];
  const text = compute ? CPU_TEXT : SRC_TEXT[s.src ?? "s3"];
  const label = compute ? "cpu" : s.src;
  return (
    <div className="flex items-center gap-2 h-3.5 pl-2">
      <span className={`w-7 shrink-0 font-mono text-[9px] ${text}`}>{label}</span>
      <span
        className={`w-48 shrink-0 font-mono text-[10px] truncate ${
          compute ? "text-warn" : "text-ink-dim"
        }`}
        title={`${s.key} · ${s.op}`}
        dir={compute ? "ltr" : "rtl"}
      >
        {compute ? s.key : tailKey(s.key)}
      </span>
      <div className="flex-1 relative h-2.5 bg-panel-2 rounded-sm">
        <div
          className={`absolute h-full rounded-sm ${bar}`}
          style={{ left: `${left}%`, width: `${width}%` }}
          title={`${s.op} · ${compute ? "cpu" : s.src} · ${fmtMs(dur)}${
            compute ? "" : ` · ${fmtBytes(String(s.bytes))}`
          }`}
        />
      </div>
      <span className="w-14 shrink-0 text-right font-mono text-[10px] tnum text-ink-dim">
        {fmtMs(dur)}
      </span>
    </div>
  );
}

/** The unified sequence diagram: every I/O access AND compute phase on one timeline, bracketed by
 * main phase (snapshot → recall → rerank → …). Segments run concurrently, so their per-phase compute
 * spans overlap here — the parallelism is visible. */
function Waterfall({
  objects,
  totalMs,
}: {
  objects: NonNullable<TraceRec["objects"]>;
  totalMs: number;
}) {
  const items = objects.items;
  const maxEnd = Math.max(totalMs * 1000, ...items.map((s) => s.end_us), 1);

  // Legend: I/O by tier + compute.
  const io: Record<ObjSource, { n: number; ms: number }> = {
    mem: { n: 0, ms: 0 },
    disk: { n: 0, ms: 0 },
    s3: { n: 0, ms: 0 },
  };
  let cpuN = 0;
  let cpuMs = 0;
  for (const s of items) {
    const d = (s.end_us - s.start_us) / 1000;
    if (s.kind === "compute") {
      cpuN += 1;
      cpuMs += d;
    } else if (s.src) {
      io[s.src].n += 1;
      io[s.src].ms += d;
    }
  }

  // Bucket by phase group, in draw order.
  const byGroup = new Map<string, ObjSpan[]>();
  for (const s of items) {
    const g = s.group || "other";
    (byGroup.get(g) ?? byGroup.set(g, []).get(g)!).push(s);
  }
  const groups = [
    ...GROUP_ORDER.filter((g) => byGroup.has(g)),
    ...[...byGroup.keys()].filter((g) => !GROUP_ORDER.includes(g)),
  ];

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-2">
        {(["s3", "disk", "mem"] as ObjSource[]).map((src) => (
          <span key={src} className="flex items-center gap-1.5 font-mono text-[10px]">
            <span className={`inline-block w-2 h-2 rounded-sm ${SRC_BAR[src]}`} />
            <span className={SRC_TEXT[src]}>{src}</span>
            <span className="text-ink-faint">
              ×{io[src].n} · {fmtMs(io[src].ms)}
            </span>
          </span>
        ))}
        <span className="flex items-center gap-1.5 font-mono text-[10px]">
          <span className={`inline-block w-2 h-2 rounded-sm ${CPU_BAR}`} />
          <span className={CPU_TEXT}>cpu</span>
          <span className="text-ink-faint">
            ×{cpuN} · {fmtMs(cpuMs)}
          </span>
        </span>
        {objects.dropped > 0 && (
          <span className="font-mono text-[10px] text-ink-faint">
            +{groupDigits(String(objects.dropped))} more (capped)
          </span>
        )}
      </div>

      <div className="max-h-[28rem] overflow-y-auto flex flex-col gap-2 pr-1">
        {groups.map((g) => {
          const gs = byGroup.get(g)!.slice().sort((a, b) => a.start_us - b.start_us);
          const gStart = Math.min(...gs.map((s) => s.start_us));
          const gEnd = Math.max(...gs.map((s) => s.end_us));
          return (
            <div key={g}>
              {/* group bracket: the phase's overall span across the timeline */}
              <div className="flex items-center gap-2 mb-0.5">
                <span className="w-[3.75rem] shrink-0 font-mono text-[10px] uppercase tracking-wide text-ink-dim">
                  {g}
                </span>
                <div className="flex-1 relative h-1.5">
                  <div
                    className="absolute h-full bg-ink-faint/25 rounded-full"
                    style={{
                      left: `${(gStart / maxEnd) * 100}%`,
                      width: `${Math.max(((gEnd - gStart) / maxEnd) * 100, 0.4)}%`,
                    }}
                  />
                </div>
                <span className="w-14 shrink-0 text-right font-mono text-[10px] text-ink-faint tnum">
                  {fmtMs((gEnd - gStart) / 1000)}
                </span>
              </div>
              {gs.map((s, i) => (
                <SpanRow key={i} s={s} maxEnd={maxEnd} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const MS = (us: number) => us / 1000;

/** One-line "why was this slow" — the biggest wall-clock contributor. */
function diagnose(r: TraceRec): string {
  const total = r.total_ms ?? 0;
  if (r.op === "write") {
    const parts: [string, number][] = [
      ["snapshot open", r.link_snapshot_ms ?? 0],
      ["link derivation", r.link_ms ?? 0],
      ["commit", r.commit_ms ?? 0],
    ];
    parts.sort((a, b) => b[1] - a[1]);
    const [lbl, v] = parts[0];
    let extra = "";
    if (lbl === "link derivation" && r.link_io) {
      const io = r.link_io;
      extra = ` — ${io.queries ?? "?"} corpus queries, ${fmtMs(io.corpus_query_ms)} in S3`;
      if ((io.cache_misses ?? 0) > 0) extra += `, ${groupDigits(String(io.cache_misses))} cold fetches`;
    }
    return `${lbl} dominated: ${fmtMs(v)} of ${fmtMs(total)}${extra}`;
  }
  // query
  const wait = r.permit_wait_ms ?? 0;
  const open = r.snapshot?.open_ms ?? 0;
  if (wait > total * 0.3 && wait > 20) {
    return `queued ${fmtMs(wait)} waiting for an admission permit — concurrency contention, not the query itself`;
  }
  if (open > total * 0.4) {
    const cold = (r.io?.cache_misses ?? 0) > 0;
    return `snapshot open took ${fmtMs(open)} (${r.snapshot?.action ?? "?"}) — ${
      cold ? "cold object-store reads" : "head/manifest/WAL-tail round trips"
    }; the search itself was cheap`;
  }
  const top = (r.phases_us ?? []).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])[0];
  if (top) {
    return `${top[0]}-bound: ${fmtMs(MS(top[1]))} accumulated in ${top[0]} (across parallel arms)`;
  }
  return `${fmtMs(total)} total`;
}

/** Wall-clock segments that (roughly) sum to total_ms — the honest "where the elapsed time went". */
function wallSegments(r: TraceRec): { label: string; ms: number; tone: string }[] {
  const total = r.total_ms ?? 0;
  const segs: { label: string; ms: number; tone: string }[] = [];
  const add = (label: string, ms: number | undefined | null, tone: string) => {
    if (ms && ms > 0.05) segs.push({ label, ms, tone });
  };
  if (r.op === "write") {
    add("snapshot open", r.link_snapshot_ms, "bg-accent/60");
    add("link derivation", r.link_ms, "bg-warn/70");
    add("commit", r.commit_ms, "bg-ok/60");
  } else {
    add("permit wait", r.permit_wait_ms, "bg-ink-faint/50");
    add("snapshot open", r.snapshot?.open_ms, "bg-accent/60");
    const used = (r.permit_wait_ms ?? 0) + (r.snapshot?.open_ms ?? 0);
    add("execute", total - used, "bg-warn/70");
  }
  const acct = segs.reduce((s, x) => s + x.ms, 0);
  if (total - acct > total * 0.02 && total - acct > 1) {
    segs.push({ label: "other", ms: total - acct, tone: "bg-ink-faint/30" });
  }
  return segs;
}

const PHASE_TONE: Record<string, string> = {
  probe: "bg-accent/40",
  fetch_clusters: "bg-warn/70",
  rerank: "bg-danger/60",
  fts: "bg-ok/50",
  graph_radj: "bg-accent/30",
  graph_pk: "bg-accent/30",
  graph_fetch: "bg-accent/40",
  graph_expand: "bg-accent/30",
  fuse: "bg-ink-faint/50",
};

function Bar({
  label,
  ms,
  frac,
  tone,
  suffix,
}: {
  label: string;
  ms: number;
  frac: number;
  tone: string;
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-28 shrink-0 font-mono text-[11px] text-ink-dim truncate" title={label}>
        {label}
      </div>
      <div className="flex-1 h-3.5 bg-panel-2 rounded-sm overflow-hidden">
        <div
          className={`h-full ${tone}`}
          style={{ width: `${Math.max(frac * 100, frac > 0 ? 1.5 : 0)}%` }}
        />
      </div>
      <div className="w-28 shrink-0 text-right font-mono text-[11px] tnum text-ink">
        {fmtMs(ms)}
        {suffix && <span className="text-ink-faint"> {suffix}</span>}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1.5">
        {title}
      </div>
      {children}
    </div>
  );
}

function StatGrid({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1">
      {rows.map(([k, v], i) => (
        <div key={`${k}-${i}`} className="flex items-baseline justify-between gap-2 min-w-0">
          <span className="font-mono text-[10px] text-ink-faint truncate">{k}</span>
          <span className="font-mono text-[11px] tnum text-ink shrink-0">{v}</span>
        </div>
      ))}
    </div>
  );
}

export function TraceDetail({ rec }: { rec: TraceRec }) {
  const [showRaw, setShowRaw] = useState(false);
  const total = rec.total_ms ?? 0;
  const wall = wallSegments(rec);

  const profile = (rec.op === "write" ? rec.link_io?.phases_us : rec.phases_us) ?? [];
  const nonzero = profile.filter(([, v]) => v > 0);
  const profMax = Math.max(1, ...nonzero.map(([, v]) => v));

  const io = rec.op === "write" ? rec.link_io : rec.io;

  const stats: [string, React.ReactNode][] = [];
  if (rec.snapshot) {
    stats.push(["snapshot", rec.snapshot.action ?? "—"]);
    stats.push(["tail entries", groupDigits(String(rec.snapshot.tail_entries ?? 0))]);
  }
  if (rec.op === "query") {
    stats.push(["results", String(rec.result_count ?? 0)]);
    stats.push(["in-flight", String(rec.in_flight ?? 0)]);
    stats.push(["permit wait", fmtMs(rec.permit_wait_ms)]);
  }
  if (rec.op === "write") {
    stats.push(["attempts", String(rec.attempts ?? 1)]);
    stats.push(["seq", String(rec.seq ?? "—")]);
    stats.push(["corpus queries", String(rec.link_io?.queries ?? "—")]);
    stats.push(["within-batch", fmtMs(rec.link_io?.within_batch_ms)]);
    stats.push([
      "wait-for-index",
      rec.wait_for_index_ms == null ? "—" : fmtMs(rec.wait_for_index_ms),
    ]);
  }
  if (io) {
    const hits = io.cache_hits ?? 0;
    const misses = io.cache_misses ?? 0;
    stats.push(["cache hit", hits + misses > 0 ? `${((hits / (hits + misses)) * 100).toFixed(1)}%` : "—"]);
    stats.push([
      "cache h/m",
      <span key="hm" className={misses > 0 ? "text-warn" : undefined}>
        {groupDigits(String(hits))}/{groupDigits(String(misses))}
      </span>,
    ]);
    stats.push(["roundtrips", String(io.roundtrips ?? 0)]);
    stats.push(["bytes", fmtBytes(String(io.bytes ?? 0))]);
    if ("tier" in io && io.tier) stats.push(["tier", io.tier]);
  }
  const params = rec.params ?? {};
  for (const [k, v] of Object.entries(params)) {
    stats.push([k, String(v)]);
  }

  return (
    <div className="flex flex-col gap-3 p-3 bg-panel-2/40 border-l-2 border-accent/40">
      <div className="text-[12px] text-ink leading-snug">
        <span className="text-ink-faint font-mono text-[10px] uppercase tracking-wide mr-2">
          why
        </span>
        {diagnose(rec)}
      </div>

      <Section title={`wall clock · ${fmtMs(total)} total`}>
        <div className="flex flex-col gap-1">
          {wall.map((s) => (
            <Bar
              key={s.label}
              label={s.label}
              ms={s.ms}
              frac={total > 0 ? s.ms / total : 0}
              tone={s.tone}
              suffix={total > 0 ? `${((s.ms / total) * 100).toFixed(0)}%` : ""}
            />
          ))}
        </div>
      </Section>

      {rec.objects && rec.objects.items.length > 0 ? (
        <Section
          title={`sequence · ${rec.objects.count} spans (I/O + compute), by phase`}
        >
          <Waterfall objects={rec.objects} totalMs={total} />
        </Section>
      ) : (
        // Fallback for records captured before per-object spans existed: the accumulated profile.
        nonzero.length > 0 && (
          <Section title="phase profile · accumulated, arms run in parallel">
            <div className="flex flex-col gap-1">
              {nonzero
                .slice()
                .sort((a, b) => b[1] - a[1])
                .map(([name, us]) => (
                  <Bar
                    key={name}
                    label={name}
                    ms={MS(us)}
                    frac={us / profMax}
                    tone={PHASE_TONE[name] ?? "bg-ink-faint/40"}
                  />
                ))}
            </div>
          </Section>
        )
      )}

      <Section title="stats">
        <StatGrid rows={stats} />
      </Section>

      <div>
        <button
          type="button"
          onClick={() => setShowRaw((v) => !v)}
          className="font-mono text-[10px] text-ink-faint hover:text-accent"
        >
          {showRaw ? "▾ raw record" : "▸ raw record"}
        </button>
        {showRaw && (
          <pre className="mt-1 p-2 bg-bg border border-line rounded-sm overflow-x-auto text-[10px] text-ink-dim leading-snug">
            {JSON.stringify(rec, null, 1)}
          </pre>
        )}
      </div>
    </div>
  );
}
