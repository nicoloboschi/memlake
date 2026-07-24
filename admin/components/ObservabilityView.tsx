"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import { fmtBytes, fmtMs, groupDigits } from "@/lib/format";
import {
  Button,
  CopyableId,
  Empty,
  ErrorBanner,
  Loading,
  Panel,
  StatTile,
  Toggle,
} from "@/components/ui";

// Mirrors lib/obs.ts (kept local so the client bundle carries no server import).
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
  records: number;
}
interface NodeSummary {
  header: NodeHeader;
  sizeBytes: number;
  fetchedMs: number;
}

const STALE_MS = 30_000;
const REFRESH_MS = 8_000;

function ageLabel(ms: number): string {
  if (ms < 1000) return "just now";
  const s = Math.round(ms / 1000);
  if (s < 90) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}

const pct = (f: number) => `${(f * 100).toFixed(1)}%`;

export function ObservabilityView() {
  const [nodes, setNodes] = useState<NodeSummary[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  const [now, setNow] = useState(() => Date.now());

  const loadNodes = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getJson<{ nodes: NodeSummary[] }>("/api/obs");
      setNodes(res.nodes);
      setNow(Date.now());
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
      setNodes(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadNodes();
  }, [loadNodes]);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => void loadNodes(), REFRESH_MS);
    return () => clearInterval(id);
  }, [auto, loadNodes]);

  const totalCalls = useMemo(
    () => (nodes ?? []).reduce((s, n) => s + (n.header.totals.count ?? 0), 0),
    [nodes],
  );

  return (
    <div className="p-4 max-w-6xl mx-auto flex flex-col gap-4">
      <Panel
        title="services — serve fleet"
        subtitle={
          <>
            Read straight from <code>_obs/traces/</code> in the bucket — one bounded ring per serve
            node, uploaded every ~5s. No pod scraping. Per-node health; a cold node costs latency
            only, never correctness. Open the{" "}
            <Link href="/traces" className="text-accent hover:underline">
              trace explorer
            </Link>{" "}
            to drill into individual requests.
          </>
        }
        actions={
          <>
            <span className="font-mono text-[11px] text-ink-faint tnum">
              {nodes ? `${nodes.length} nodes · ${groupDigits(String(totalCalls))} calls` : "—"}
            </span>
            <Toggle checked={auto} onChange={setAuto} label="auto" />
            <Button onClick={() => void loadNodes()} disabled={loading}>
              refresh
            </Button>
          </>
        }
        bodyClassName="p-0"
      >
        {loading && !nodes && (
          <div className="p-3">
            <Loading label="reading _obs/traces/" />
          </div>
        )}
        {Boolean(error) && (
          <div className="p-3">
            <ErrorBanner error={error} what="services" onRetry={() => void loadNodes()} />
          </div>
        )}
        {nodes && nodes.length === 0 && !error && (
          <div className="p-3">
            <Empty title="no trace objects yet">
              <p>
                No <code>_obs/traces/*.jsonl</code> in this bucket. A serve node publishes its ring
                within a few seconds of starting, unless <code>MEMLAKE_TRACE_LOG=off</code>.
              </p>
            </Empty>
          </div>
        )}
        {nodes && nodes.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-3">
            {nodes.map((n) => (
              <NodeCard key={n.header.node_id} node={n} now={now} />
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

function NodeCard({ node, now }: { node: NodeSummary; now: number }) {
  const h = node.header;
  const age = now - h.updated_ms;
  const stale = age > STALE_MS;
  const t = h.totals;

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          <span
            className={`inline-block w-1.5 h-1.5 rounded-full ${stale ? "bg-warn" : "bg-ok"}`}
            title={stale ? "stale heartbeat" : "live"}
          />
          <CopyableId value={h.node_id} />
        </span>
      }
      subtitle={
        <span className={stale ? "text-warn" : undefined}>
          {ageLabel(age)} · ring {fmtBytes(String(node.sizeBytes))} · up {ageLabel(h.uptime_ms)}
        </span>
      }
      actions={
        <Link
          href={`/traces?node=${encodeURIComponent(h.node_id)}`}
          className="font-mono text-[11px] text-ink-dim hover:text-accent border border-line rounded-sm px-2 py-1"
        >
          traces →
        </Link>
      }
      bodyClassName="p-3 flex flex-col gap-3"
    >
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <StatTile label="qps" value={t.qps.toFixed(1)} />
        <StatTile label="p50" value={fmtMs(t.p50_ms)} />
        <StatTile label="p99" value={fmtMs(t.p99_ms)} tone={t.p99_ms >= 1000 ? "warn" : "normal"} />
        <StatTile
          label="cache hit"
          value={pct(t.cache_hit)}
          tone={t.cache_hit < 0.9 ? "warn" : "ok"}
        />
      </div>

      <ActionBar actions={h.by_action} />

      {h.by_namespace.length > 0 && (
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1">
            busiest namespaces
          </div>
          <div className="flex flex-col gap-0.5">
            {h.by_namespace.slice(0, 6).map((ns) => (
              <Link
                key={ns.ns}
                href={`/traces?namespace=${encodeURIComponent(ns.ns)}`}
                className="flex items-center justify-between gap-2 font-mono text-[11px] text-ink-dim hover:text-accent"
                title="open this namespace's traces"
              >
                <span className="truncate">{ns.ns}</span>
                <span className="tnum text-ink-faint shrink-0">
                  {groupDigits(String(ns.count))} · p50 {fmtMs(ns.p50_ms)} · p99 {fmtMs(ns.p99_ms)}
                </span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}

function ActionBar({ actions }: { actions: Record<string, number> }) {
  const entries = Object.entries(actions);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (total === 0) return null;
  // The snapshot outcome mix — mostly-reuse is healthy; lots of reopen_fold means fold churn.
  const color: Record<string, string> = {
    reuse: "bg-ok/70",
    reopen_tail: "bg-accent/60",
    reopen_fold: "bg-warn/70",
    full_open: "bg-danger/70",
  };
  return (
    <div>
      <div className="flex h-2 w-full overflow-hidden rounded-sm bg-panel-2">
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
