"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { isAbort, getJson } from "@/lib/client";
import { fmtBytes, fmtMs, groupDigits } from "@/lib/format";
import { TraceDetail, type TraceRec } from "@/components/TraceDetail";
import {
  Button,
  CopyableId,
  Empty,
  ErrorBanner,
  Loading,
  Panel,
  StatTile,
  TableShell,
  Tag,
  Td,
  TextInput,
  Th,
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
// The record shape is owned by TraceDetail (the drill-down renders every field).
type TraceRecord = TraceRec;

/** A record at/above this is "slow" — the tail the ring is biased to keep, flagged in the UI. */
const SLOW_MS = 200;
/** Past this heartbeat age a node is treated as stale (uploads are every ~5s). */
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

function clock(ts?: number): string {
  if (!ts) return "—";
  const d = new Date(ts);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function pct(f: number): string {
  return `${(f * 100).toFixed(1)}%`;
}

export function ObservabilityView() {
  const [nodes, setNodes] = useState<NodeSummary[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  const [nsFilter, setNsFilter] = useState("");
  const [nsInput, setNsInput] = useState("");
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [records, setRecords] = useState<TraceRecord[] | null>(null);
  const [recordsLoading, setRecordsLoading] = useState(false);
  // A wall-clock kept in state (not `Date.now()` in render) so heartbeat ages recompute on each
  // poll tick without an impure read during render.
  const [now, setNow] = useState(() => Date.now());
  const abortRef = useRef<AbortController | null>(null);

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

  // Drill-in: a selected node's records, or a namespace's records across nodes.
  const loadRecords = useCallback(async () => {
    if (!nsFilter && !selectedNode) {
      setRecords(null);
      return;
    }
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setRecordsLoading(true);
    try {
      const q = nsFilter
        ? `?namespace=${encodeURIComponent(nsFilter)}`
        : `?node=${encodeURIComponent(selectedNode!)}`;
      const res = await getJson<{ records: TraceRecord[] }>(`/api/obs${q}`, ac.signal);
      setRecords(res.records);
    } catch (e) {
      if (isAbort(e)) return;
      setRecords(null);
    } finally {
      if (abortRef.current === ac) setRecordsLoading(false);
    }
  }, [nsFilter, selectedNode]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadNodes();
  }, [loadNodes]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadRecords();
  }, [loadRecords]);

  // Poll at the upload cadence so the fleet OVERVIEW tracks reality. We deliberately do NOT
  // auto-reload the drilled-in records: once you're inspecting a node's or namespace's traces they
  // must hold still (a reshuffle mid-investigation is useless). Records refresh on an explicit
  // action — selecting a node/namespace, or the manual refresh button.
  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => {
      void loadNodes();
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, [auto, loadNodes]);

  const totalCalls = useMemo(
    () => (nodes ?? []).reduce((s, n) => s + (n.header.totals.count ?? 0), 0),
    [nodes],
  );

  function applyNsFilter(v: string) {
    const trimmed = v.trim();
    setNsFilter(trimmed);
    if (trimmed) setSelectedNode(null);
  }

  return (
    <div className="p-4 max-w-6xl mx-auto flex flex-col gap-4">
      <Panel
        title="services — serve fleet"
        subtitle={
          <>
            Read straight from <code>_obs/traces/</code> in the bucket — one bounded (slow-biased)
            ring per serve node, uploaded every ~5s. No pod scraping. Click any trace to see where
            its time went. Cache/latency are per node; a cold node costs latency only, never
            correctness.
          </>
        }
        actions={
          <>
            <span className="font-mono text-[11px] text-ink-faint tnum">
              {nodes ? `${nodes.length} nodes · ${groupDigits(String(totalCalls))} calls` : "—"}
            </span>
            <Toggle checked={auto} onChange={setAuto} label="auto" />
            <Button
              onClick={() => {
                void loadNodes();
                void loadRecords();
              }}
              disabled={loading}
            >
              refresh
            </Button>
          </>
        }
        bodyClassName="p-3"
      >
        <div className="flex items-center gap-2">
          <TextInput
            value={nsInput}
            placeholder="filter by namespace (see it across every node)…"
            onChange={(e) => setNsInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") applyNsFilter(nsInput);
            }}
            className="flex-1"
          />
          <Button onClick={() => applyNsFilter(nsInput)} disabled={!nsInput.trim()}>
            filter
          </Button>
          {nsFilter && (
            <Button
              onClick={() => {
                setNsInput("");
                applyNsFilter("");
              }}
            >
              clear
            </Button>
          )}
        </div>
      </Panel>

      {loading && !nodes && <Loading label="reading _obs/traces/" />}

      {Boolean(error) && (
        <ErrorBanner error={error} what="observability" onRetry={() => void loadNodes()} />
      )}

      {nodes && nodes.length === 0 && !error && (
        <Empty title="no trace objects yet">
          <p>
            No <code>_obs/traces/*.jsonl</code> in this bucket. A serve node publishes its ring
            within a few seconds of starting, unless <code>MEMLAKE_TRACE_LOG=off</code>.
          </p>
        </Empty>
      )}

      {/* namespace drill-in takes over the main area */}
      {nsFilter ? (
        <NamespaceTraces
          namespace={nsFilter}
          nodes={nodes ?? []}
          records={records}
          loading={recordsLoading}
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {(nodes ?? []).map((n) => (
            <NodeCard
              key={n.header.node_id}
              node={n}
              now={now}
              expanded={selectedNode === n.header.node_id}
              records={selectedNode === n.header.node_id ? records : null}
              recordsLoading={selectedNode === n.header.node_id && recordsLoading}
              onToggle={() =>
                setSelectedNode((cur) =>
                  cur === n.header.node_id ? null : n.header.node_id,
                )
              }
              onPickNamespace={(ns) => {
                setNsInput(ns);
                applyNsFilter(ns);
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function NodeCard({
  node,
  now,
  expanded,
  records,
  recordsLoading,
  onToggle,
  onPickNamespace,
}: {
  node: NodeSummary;
  now: number;
  expanded: boolean;
  records: TraceRecord[] | null;
  recordsLoading: boolean;
  onToggle: () => void;
  onPickNamespace: (ns: string) => void;
}) {
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
        <Button onClick={onToggle}>{expanded ? "hide traces" : "traces"}</Button>
      }
      bodyClassName="p-3 flex flex-col gap-3"
    >
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <StatTile label="qps" value={t.qps.toFixed(1)} />
        <StatTile label="p50" value={fmtMs(t.p50_ms)} />
        <StatTile
          label="p99"
          value={fmtMs(t.p99_ms)}
          tone={t.p99_ms >= 1000 ? "warn" : "normal"}
        />
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
              <button
                key={ns.ns}
                type="button"
                onClick={() => onPickNamespace(ns.ns)}
                className="flex items-center justify-between gap-2 text-left font-mono text-[11px]
                  text-ink-dim hover:text-accent"
                title="filter to this namespace across all nodes"
              >
                <span className="truncate">{ns.ns}</span>
                <span className="tnum text-ink-faint shrink-0">
                  {groupDigits(String(ns.count))} · p50 {fmtMs(ns.p50_ms)} · p99 {fmtMs(ns.p99_ms)}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {expanded && (
        <div className="border-t border-line pt-2">
          {recordsLoading && !records ? (
            <Loading label="reading node ring" />
          ) : (
            <RecordsTable records={records ?? []} showNode={false} />
          )}
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

function NamespaceTraces({
  namespace,
  nodes,
  records,
  loading,
}: {
  namespace: string;
  nodes: NodeSummary[];
  records: TraceRecord[] | null;
  loading: boolean;
}) {
  // Per-node rollup for this namespace, straight from each node's header (covers even the fast
  // calls whose individual records the slow-biased ring dropped).
  const perNode = nodes
    .map((n) => ({
      node_id: n.header.node_id,
      roll: n.header.by_namespace.find((x) => x.ns === namespace),
    }))
    .filter((x) => x.roll);

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          namespace <Tag>{namespace}</Tag> across {perNode.length || "—"} nodes
        </span>
      }
      subtitle="Every retained trace touching this namespace, merged across nodes, newest first. Rollups come from each node's header even when raw records aged out."
      bodyClassName="p-3 flex flex-col gap-3"
    >
      {perNode.length > 0 && (
        <TableShell
          head={
            <>
              <Th>node</Th>
              <Th className="text-right">calls</Th>
              <Th className="text-right">p50</Th>
              <Th className="text-right">p99</Th>
            </>
          }
        >
          {perNode.map(({ node_id, roll }) => (
            <tr key={node_id}>
              <Td>
                <CopyableId value={node_id} />
              </Td>
              <Td className="text-right tnum">{groupDigits(String(roll!.count))}</Td>
              <Td className="text-right tnum">{fmtMs(roll!.p50_ms)}</Td>
              <Td className="text-right tnum">{fmtMs(roll!.p99_ms)}</Td>
            </tr>
          ))}
        </TableShell>
      )}

      {loading && !records ? (
        <Loading label="reading rings" />
      ) : records && records.length === 0 ? (
        <Empty title="no retained traces for this namespace">
          <p>
            The rollup above still counts its calls; individual records may have aged out of the
            slow-biased ring if the namespace has been fast and quiet.
          </p>
        </Empty>
      ) : (
        <RecordsTable records={records ?? []} showNode />
      )}
    </Panel>
  );
}

function RecordsTable({
  records,
  showNode,
}: {
  records: TraceRecord[];
  showNode: boolean;
}) {
  // Which row is drilled into. Keyed by index within this render's record list.
  const [open, setOpen] = useState<number | null>(null);
  if (records.length === 0) {
    return <Empty title="no records retained" />;
  }
  const cols = showNode ? 7 : 7; // time, [node|namespace], op, snapshot, total, open, tail
  return (
    <TableShell
      head={
        <>
          <Th>time</Th>
          {showNode && <Th>node</Th>}
          <Th>op</Th>
          {!showNode && <Th>namespace</Th>}
          <Th>snapshot</Th>
          <Th className="text-right">total</Th>
          <Th className="text-right">open</Th>
          <Th className="text-right" title="un-indexed WAL tail items carried by the snapshot">
            tail
          </Th>
        </>
      }
    >
      {records.map((r, i) => {
        const slow = (r.total_ms ?? 0) >= SLOW_MS;
        const isOpen = open === i;
        return (
          <Fragment key={`${r.ts_ms}-${i}`}>
            <tr
              onClick={() => setOpen(isOpen ? null : i)}
              className={`cursor-pointer ${
                isOpen ? "bg-accent/10" : slow ? "bg-warn/5 hover:bg-warn/10" : "hover:bg-panel-2"
              }`}
            >
              <Td className="tnum text-ink-dim">
                <span className="text-ink-faint mr-1">{isOpen ? "▾" : "▸"}</span>
                {clock(r.ts_ms)}
              </Td>
              {showNode && (
                <Td>
                  <span className="font-mono text-[11px] text-ink-dim">{r.node_id ?? "—"}</span>
                </Td>
              )}
              <Td>
                <Tag>{r.op ?? "—"}</Tag>
              </Td>
              {!showNode && (
                <Td className="font-mono text-[11px] text-ink-dim truncate max-w-[12rem]">
                  {r.namespace ?? "—"}
                </Td>
              )}
              <Td className="font-mono text-[11px] text-ink-dim">{r.snapshot?.action ?? "—"}</Td>
              <Td className={`text-right tnum ${slow ? "text-warn" : "text-ink"}`}>
                {fmtMs(r.total_ms)}
              </Td>
              <Td className="text-right tnum text-ink-dim">{fmtMs(r.snapshot?.open_ms)}</Td>
              <Td className="text-right tnum text-ink-dim">
                {r.snapshot?.tail_entries != null
                  ? groupDigits(String(r.snapshot.tail_entries))
                  : "—"}
              </Td>
            </tr>
            {isOpen && (
              <tr>
                <td colSpan={cols} className="p-0 border-b border-line">
                  <TraceDetail rec={r} />
                </td>
              </tr>
            )}
          </Fragment>
        );
      })}
    </TableShell>
  );
}
