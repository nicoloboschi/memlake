"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import { fmtBytes, fmtMs, groupDigits } from "@/lib/format";
import {
  Button,
  Empty,
  ErrorBanner,
  Loading,
  Panel,
  StatTile,
  TableShell,
  Td,
  Th,
  Toggle,
} from "@/components/ui";

interface NodeCache {
  enabled: boolean;
  mem_bytes?: number;
  mem_budget?: number;
  mem_entries?: number;
  disk_bytes?: number;
  disk_budget?: number;
  disk_entries?: number;
  hits?: number;
  misses?: number;
}
interface NsRollup {
  ns: string;
  count: number;
  p50_ms: number;
  p99_ms: number;
}
interface NodeHeader {
  node_id: string;
  updated_ms: number;
  uptime_ms: number;
  totals: { count: number; qps: number; p50_ms: number; p99_ms: number; cache_hit: number };
  by_action: Record<string, number>;
  by_namespace: NsRollup[];
  pending?: number;
  dropped?: number;
  cache?: NodeCache;
}

const STALE_MS = 30_000;
const REFRESH_MS = 5_000;

function ageLabel(ms: number): string {
  if (ms < 1000) return "just now";
  const s = Math.round(ms / 1000);
  if (s < 90) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}
const pct = (f: number) => `${(f * 100).toFixed(1)}%`;

function TierBar({
  label,
  used,
  budget,
  entries,
}: {
  label: string;
  used: number;
  budget: number;
  entries: number;
}) {
  const frac = budget > 0 ? Math.min(used / budget, 1) : 0;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 font-mono text-[11px] mb-1">
        <span className="text-ink-dim uppercase tracking-wide">{label}</span>
        <span className="text-ink-faint tnum">
          {fmtBytes(String(used))} / {fmtBytes(String(budget))} · {groupDigits(String(entries))} obj
          {budget > 0 ? ` · ${pct(frac)} full` : ""}
        </span>
      </div>
      <div className="h-3 w-full bg-panel-2 rounded-sm overflow-hidden">
        <div
          className={`h-full ${frac > 0.95 ? "bg-warn/70" : "bg-accent/60"}`}
          style={{ width: `${frac * 100}%` }}
        />
      </div>
    </div>
  );
}

function ActionBar({ actions }: { actions: Record<string, number> }) {
  const entries = Object.entries(actions);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (total === 0) return null;
  const color: Record<string, string> = {
    reuse: "bg-ok/70",
    reopen_tail: "bg-accent/60",
    reopen_fold: "bg-warn/70",
    full_open: "bg-danger/70",
  };
  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-sm bg-panel-2">
        {entries.map(([k, v]) => (
          <div
            key={k}
            className={color[k] ?? "bg-ink-faint/40"}
            style={{ width: `${(v / total) * 100}%` }}
            title={`${k}: ${v} (${((v / total) * 100).toFixed(0)}%)`}
          />
        ))}
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-ink-faint">
        {entries.map(([k, v]) => (
          <span key={k}>
            {k} {groupDigits(String(v))}
          </span>
        ))}
      </div>
    </div>
  );
}

export function ServiceDetail({ node, tab }: { node: string; tab: "stats" | "cache" }) {
  const [header, setHeader] = useState<NodeHeader | null>(null);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  const [now, setNow] = useState(() => Date.now());

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Rollups are tiny; fetching the fleet and picking one keeps a single cached API shape.
      const res = await getJson<{ nodes: { header: NodeHeader }[] }>("/api/obs");
      const found = res.nodes.find((n) => n.header.node_id === node)?.header ?? null;
      setHeader(found);
      setMissing(!found);
      setNow(Date.now());
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
    } finally {
      setLoading(false);
    }
  }, [node]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => void load(), REFRESH_MS);
    return () => clearInterval(id);
  }, [auto, load]);

  if (loading && !header && !missing) {
    return (
      <div className="p-4">
        <Loading label="reading rollup" />
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-4">
        <ErrorBanner error={error} what="service" onRetry={() => void load()} />
      </div>
    );
  }
  if (missing) {
    return (
      <div className="p-4">
        <Empty title={`no rollup for ${node}`}>
          <p>
            This node has not published to <code>_obs/rollup/</code>. It may have been scaled down or
            renamed — its rollup is reaped after an hour of silence.
          </p>
          <p className="mt-2">
            <Link href="/services" className="text-accent hover:underline">
              back to services
            </Link>
          </p>
        </Empty>
      </div>
    );
  }
  if (!header) return null;

  const age = now - header.updated_ms;
  const stale = age > STALE_MS;
  const t = header.totals;
  const c = header.cache;

  return (
    <div className="p-4 max-w-5xl mx-auto flex flex-col gap-4">
      <Panel
        title={
          <span className="flex items-center gap-2">
            <span
              className={`inline-block w-1.5 h-1.5 rounded-full ${stale ? "bg-warn" : "bg-ok"}`}
              title={stale ? "stale heartbeat" : "live"}
            />
            {tab === "cache" ? "read cache" : "stats"}
          </span>
        }
        subtitle={
          <span className={stale ? "text-warn" : undefined}>
            heartbeat {ageLabel(age)} · up {ageLabel(header.uptime_ms)}
            {header.pending ? ` · ${groupDigits(String(header.pending))} traces buffered` : ""}
            {header.dropped ? (
              <span className="text-danger">
                {" "}
                · {groupDigits(String(header.dropped))} dropped
              </span>
            ) : null}
          </span>
        }
        actions={
          <>
            <Toggle checked={auto} onChange={setAuto} label="auto" />
            <Button onClick={() => void load()} disabled={loading}>
              refresh
            </Button>
          </>
        }
        bodyClassName="p-3 flex flex-col gap-4"
      >
        {tab === "stats" ? (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <StatTile label="calls" value={groupDigits(String(t.count))} hint="since start" />
              <StatTile label="qps" value={t.qps.toFixed(1)} hint="lifetime average" />
              <StatTile label="p50" value={fmtMs(t.p50_ms)} />
              <StatTile
                label="p99"
                value={fmtMs(t.p99_ms)}
                tone={t.p99_ms >= 1000 ? "warn" : "normal"}
              />
            </div>

            <div>
              <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1.5">
                snapshot outcomes
              </div>
              <ActionBar actions={header.by_action} />
              <div className="mt-1.5 text-[11px] text-ink-faint leading-snug">
                Mostly <code>reuse</code> is healthy. A lot of <code>reopen_fold</code> means fold
                churn (each one adopts a cold generation); <code>full_open</code> means the snapshot
                cache is being missed entirely.
              </div>
            </div>
          </>
        ) : !c || !c.enabled ? (
          <Empty title="no cache configured">
            <p>This node reads through to object storage on every request.</p>
          </Empty>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <StatTile
                label="hit ratio"
                value={pct(
                  (c.hits ?? 0) + (c.misses ?? 0) > 0
                    ? (c.hits ?? 0) / ((c.hits ?? 0) + (c.misses ?? 0))
                    : 0,
                )}
                tone={
                  (c.hits ?? 0) / Math.max((c.hits ?? 0) + (c.misses ?? 0), 1) < 0.9 ? "warn" : "ok"
                }
              />
              <StatTile label="hits" value={groupDigits(String(c.hits ?? 0))} />
              <StatTile label="misses" value={groupDigits(String(c.misses ?? 0))} />
              <StatTile
                label="objects"
                value={groupDigits(String(c.mem_entries ?? 0))}
                hint="in memory"
              />
            </div>
            <TierBar
              label="memory tier"
              used={c.mem_bytes ?? 0}
              budget={c.mem_budget ?? 0}
              entries={c.mem_entries ?? 0}
            />
            <TierBar
              label="disk tier (nvme)"
              used={c.disk_bytes ?? 0}
              budget={c.disk_budget ?? 0}
              entries={c.disk_entries ?? 0}
            />
            <div className="text-[11px] text-ink-faint leading-snug">
              Both tiers are bounded independently, so peak RAM and peak disk are each capped by
              construction. They <em>overlap</em> rather than partition — a memory entry usually has a
              disk copy — so the entry counts do not sum. A cold cache costs latency only, never
              correctness.
            </div>
          </>
        )}
      </Panel>

      {tab === "stats" && header.by_namespace.length > 0 && (
        <Panel
          title="namespaces on this node"
          subtitle="What this node actually served, and how fast — its own view, not the cluster's."
          bodyClassName="p-0"
        >
          <TableShell
            head={
              <>
                <Th>namespace</Th>
                <Th className="text-right">calls</Th>
                <Th className="text-right">p50</Th>
                <Th className="text-right">p99</Th>
                <Th />
              </>
            }
          >
            {header.by_namespace.map((ns) => (
              <tr key={ns.ns} className="hover:bg-panel-2">
                <Td className="font-mono text-[12px] text-ink">{ns.ns}</Td>
                <Td className="text-right tnum">{groupDigits(String(ns.count))}</Td>
                <Td className="text-right tnum">{fmtMs(ns.p50_ms)}</Td>
                <Td
                  className={`text-right tnum ${ns.p99_ms >= 1000 ? "text-warn" : "text-ink-dim"}`}
                >
                  {fmtMs(ns.p99_ms)}
                </Td>
                <Td>
                  <Link
                    href={`/traces?node=${encodeURIComponent(node)}&namespace=${encodeURIComponent(ns.ns)}`}
                    className="font-mono text-[11px] text-ink-dim hover:text-accent"
                  >
                    traces →
                  </Link>
                </Td>
              </tr>
            ))}
          </TableShell>
        </Panel>
      )}
    </div>
  );
}
