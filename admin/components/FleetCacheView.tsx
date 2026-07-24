"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

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
interface NodeHeader {
  node_id: string;
  updated_ms: number;
  totals: { p50_ms: number; p99_ms: number };
  cache?: NodeCache;
}
interface NodeSummary {
  header: NodeHeader;
}

const STALE_MS = 30_000;
const REFRESH_MS = 8_000;

function ageLabel(ms: number): string {
  if (ms < 1000) return "just now";
  const s = Math.round(ms / 1000);
  if (s < 90) return `${s}s ago`;
  return `${Math.round(s / 60)}m ago`;
}

/** A tier's fill bar: used vs budget. Both tiers are bounded independently by construction. */
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
  const tone = frac > 0.95 ? "bg-warn/70" : "bg-accent/60";
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 font-mono text-[10px] mb-0.5">
        <span className="text-ink-faint uppercase tracking-wide">{label}</span>
        <span className="text-ink-dim tnum">
          {fmtBytes(String(used))} / {fmtBytes(String(budget))} ·{" "}
          {groupDigits(String(entries))} obj
        </span>
      </div>
      <div className="h-2 w-full bg-panel-2 rounded-sm overflow-hidden">
        <div className={`h-full ${tone}`} style={{ width: `${frac * 100}%` }} />
      </div>
    </div>
  );
}

export function FleetCacheView() {
  const [nodes, setNodes] = useState<NodeSummary[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  const [now, setNow] = useState(() => Date.now());

  const load = useCallback(async () => {
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
    void load();
  }, [load]);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => void load(), REFRESH_MS);
    return () => clearInterval(id);
  }, [auto, load]);

  return (
    <div className="p-4 max-w-6xl mx-auto flex flex-col gap-4">
      <Panel
        title="cache — every serve node"
        subtitle={
          <>
            Each node publishes its two-tier cache occupancy in its rollup, so this reads from the
            bucket rather than calling a node. That matters behind a load balancer: the CacheStats
            RPC is node-LOCAL, so a direct call answers for whichever pod it happened to reach. A
            cold cache costs latency only, never correctness.
          </>
        }
        actions={
          <>
            <Toggle checked={auto} onChange={setAuto} label="auto" />
            <Button onClick={() => void load()} disabled={loading}>
              refresh
            </Button>
          </>
        }
        bodyClassName="p-0"
      >
        {loading && !nodes && (
          <div className="p-3">
            <Loading label="reading rollups" />
          </div>
        )}
        {Boolean(error) && (
          <div className="p-3">
            <ErrorBanner error={error} what="fleet cache" onRetry={() => void load()} />
          </div>
        )}
        {nodes && nodes.length === 0 && !error && (
          <div className="p-3">
            <Empty title="no nodes reporting">
              <p>
                No rollups in <code>_obs/rollup/</code> — a serve node publishes one within a second
                of starting, unless <code>MEMLAKE_TRACE_LOG=off</code>.
              </p>
            </Empty>
          </div>
        )}
        {nodes && nodes.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-3">
            {nodes.map(({ header: h }) => {
              const c = h.cache;
              const stale = now - h.updated_ms > STALE_MS;
              const hits = c?.hits ?? 0;
              const misses = c?.misses ?? 0;
              const ratio = hits + misses > 0 ? hits / (hits + misses) : 0;
              return (
                <Panel
                  key={h.node_id}
                  title={
                    <span className="flex items-center gap-2">
                      <span
                        className={`inline-block w-1.5 h-1.5 rounded-full ${stale ? "bg-warn" : "bg-ok"}`}
                      />
                      <CopyableId value={h.node_id} />
                    </span>
                  }
                  subtitle={
                    <span className={stale ? "text-warn" : undefined}>
                      {ageLabel(now - h.updated_ms)} · p50 {fmtMs(h.totals?.p50_ms)} · p99{" "}
                      {fmtMs(h.totals?.p99_ms)}
                    </span>
                  }
                  actions={
                    <Link
                      href={`/services/${encodeURIComponent(h.node_id)}/cache`}
                      className="font-mono text-[11px] text-ink-dim hover:text-accent border border-line rounded-sm px-2 py-1"
                    >
                      inspect →
                    </Link>
                  }
                  bodyClassName="p-3 flex flex-col gap-3"
                >
                  {!c || !c.enabled ? (
                    <div className="font-mono text-[11px] text-ink-faint">
                      no cache configured — this node reads through to object storage every time
                    </div>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 gap-2">
                        <StatTile
                          label="hit ratio"
                          value={`${(ratio * 100).toFixed(1)}%`}
                          tone={ratio < 0.9 ? "warn" : "ok"}
                        />
                        <StatTile
                          label="hits / misses"
                          value={`${groupDigits(String(hits))} / ${groupDigits(String(misses))}`}
                        />
                      </div>
                      <TierBar
                        label="memory"
                        used={c.mem_bytes ?? 0}
                        budget={c.mem_budget ?? 0}
                        entries={c.mem_entries ?? 0}
                      />
                      <TierBar
                        label="disk (nvme)"
                        used={c.disk_bytes ?? 0}
                        budget={c.disk_budget ?? 0}
                        entries={c.disk_entries ?? 0}
                      />
                      <div className="font-mono text-[10px] text-ink-faint">
                        tiers overlap — a memory entry usually has a disk copy, so entry counts are
                        not a partition
                      </div>
                    </>
                  )}
                </Panel>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
