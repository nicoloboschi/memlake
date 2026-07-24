"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

import { getJson, isAbort } from "@/lib/client";
import { fmtMs } from "@/lib/format";
import { TraceDetail, type TraceRec } from "@/components/TraceDetail";
import {
  Button,
  CopyableId,
  Empty,
  ErrorBanner,
  Field,
  Loading,
  Panel,
  Select,
  TableShell,
  Tag,
  Td,
  TextInput,
  Th,
  Toggle,
} from "@/components/ui";

interface Summary {
  id: string;
  node_id: string;
  namespace: string;
  op: string;
  total_ms: number;
  ts_ms: number;
  snapshot?: { action?: string; open_ms?: number; tail_entries?: number };
}

const SLOW_MS = 200;
const REFRESH_MS = 8_000;

function clock(ts?: number): string {
  if (!ts) return "—";
  const d = new Date(ts);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function TracesView() {
  const params = useSearchParams();
  const [summaries, setSummaries] = useState<Summary[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  const [ns, setNs] = useState(params.get("namespace") ?? "");
  const [node, setNode] = useState(params.get("node") ?? "");
  const [idInput, setIdInput] = useState("");
  const [selected, setSelected] = useState<string | null>(params.get("id"));
  const [detail, setDetail] = useState<(TraceRec & { node_id?: string }) | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const loadSummaries = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getJson<{ traces: Summary[] }>("/api/obs?summaries=1&limit=1000");
      setSummaries(r.traces);
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDetail = useCallback(async (id: string) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setDetailLoading(true);
    try {
      const r = await getJson<{ record: (TraceRec & { node_id?: string }) | null }>(
        `/api/obs?trace=${encodeURIComponent(id)}`,
        ac.signal,
      );
      setDetail(r.record);
    } catch (e) {
      if (isAbort(e)) return;
      setDetail(null);
    } finally {
      if (abortRef.current === ac) setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadSummaries();
  }, [loadSummaries]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (selected) void loadDetail(selected);
    else setDetail(null);
  }, [selected, loadDetail]);

  // Auto-refresh the LIST only (the open trace holds still — it's immutable anyway).
  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => void loadSummaries(), REFRESH_MS);
    return () => clearInterval(t);
  }, [auto, loadSummaries]);

  const namespaces = useMemo(
    () => ["", ...Array.from(new Set((summaries ?? []).map((s) => s.namespace))).sort()],
    [summaries],
  );
  const nodes = useMemo(
    () => ["", ...Array.from(new Set((summaries ?? []).map((s) => s.node_id))).sort()],
    [summaries],
  );
  const filtered = useMemo(
    () =>
      (summaries ?? []).filter((s) => (!ns || s.namespace === ns) && (!node || s.node_id === node)),
    [summaries, ns, node],
  );

  return (
    <div className="p-4 flex flex-col gap-4">
      <Panel
        title="traces — request timelines"
        subtitle="Every read/write's per-object + compute span timeline, from the serve rings. Filter by namespace or service, or open a trace by id to check it again."
        actions={
          <>
            <span className="font-mono text-[11px] text-ink-faint tnum">
              {summaries ? `${summaries.length} traces` : "—"}
            </span>
            <Toggle checked={auto} onChange={setAuto} label="auto" />
            <Button onClick={() => void loadSummaries()} disabled={loading}>
              refresh
            </Button>
          </>
        }
      >
        <div className="flex flex-wrap items-end gap-3">
          <Field label="namespace">
            <Select
              value={ns}
              onChange={setNs}
              options={namespaces.map((n) => ({ value: n, label: n || "all namespaces" }))}
            />
          </Field>
          <Field label="service">
            <Select
              value={node}
              onChange={setNode}
              options={nodes.map((n) => ({ value: n, label: n || "all services" }))}
            />
          </Field>
          <div className="flex items-end gap-1">
            <Field label="open by id">
              <TextInput
                value={idInput}
                placeholder="trace uuid…"
                onChange={(e) => setIdInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && idInput.trim()) setSelected(idInput.trim());
                }}
                className="w-72"
              />
            </Field>
            <Button onClick={() => idInput.trim() && setSelected(idInput.trim())}>open</Button>
          </div>
        </div>
      </Panel>

      {loading && !summaries && <Loading label="reading traces" />}
      {Boolean(error) && (
        <ErrorBanner error={error} what="traces" onRetry={() => void loadSummaries()} />
      )}

      {selected && (
        <Panel
          title="trace"
          actions={
            <Button
              onClick={() => {
                setSelected(null);
                setIdInput("");
              }}
            >
              close
            </Button>
          }
          bodyClassName="p-0"
        >
          {detailLoading && !detail ? (
            <div className="p-3">
              <Loading label="reading trace" />
            </div>
          ) : detail ? (
            <TraceDetail rec={detail} />
          ) : (
            <div className="p-3">
              <Empty title="trace not found">
                <p>It may have aged out of every node&apos;s bounded ring.</p>
              </Empty>
            </div>
          )}
        </Panel>
      )}

      <Panel title={`${filtered.length} shown`} bodyClassName="p-0">
        {summaries && filtered.length === 0 ? (
          <div className="p-3">
            <Empty title="no traces match" />
          </div>
        ) : (
          <TableShell
            head={
              <>
                <Th>time</Th>
                <Th>service</Th>
                <Th>namespace</Th>
                <Th>op</Th>
                <Th>snapshot</Th>
                <Th className="text-right">total</Th>
                <Th className="text-right">open</Th>
                <Th>id</Th>
              </>
            }
          >
            {filtered.slice(0, 400).map((s) => {
              const slow = s.total_ms >= SLOW_MS;
              return (
                <tr
                  key={s.id || `${s.node_id}-${s.ts_ms}`}
                  onClick={() => setSelected(s.id)}
                  className={`cursor-pointer ${
                    selected === s.id
                      ? "bg-accent/10"
                      : slow
                        ? "bg-warn/5 hover:bg-warn/10"
                        : "hover:bg-panel-2"
                  }`}
                >
                  <Td className="tnum text-ink-dim">{clock(s.ts_ms)}</Td>
                  <Td className="font-mono text-[11px] text-ink-dim">{s.node_id}</Td>
                  <Td className="font-mono text-[11px] text-ink-dim truncate max-w-[10rem]">
                    {s.namespace}
                  </Td>
                  <Td>
                    <Tag>{s.op}</Tag>
                  </Td>
                  <Td className="font-mono text-[11px] text-ink-dim">
                    {s.snapshot?.action ?? "—"}
                  </Td>
                  <Td className={`text-right tnum ${slow ? "text-warn" : "text-ink"}`}>
                    {fmtMs(s.total_ms)}
                  </Td>
                  <Td className="text-right tnum text-ink-dim">{fmtMs(s.snapshot?.open_ms)}</Td>
                  <Td>
                    <CopyableId value={s.id} display={s.id ? s.id.slice(0, 8) : "—"} />
                  </Td>
                </tr>
              );
            })}
          </TableShell>
        )}
      </Panel>
    </div>
  );
}
