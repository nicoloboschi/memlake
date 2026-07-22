"use client";

import { useCallback, useEffect, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import { fmtMs, groupDigits } from "@/lib/format";
import { type StatsJson } from "@/lib/types";
import {
  Button,
  Empty,
  ErrorBanner,
  KeyValue,
  Loading,
  Panel,
  SegmentedControl,
  StatTile,
  Td,
  TableShell,
  Th,
} from "@/components/ui";

export function StatsView({ namespace }: { namespace: string }) {
  const [data, setData] = useState<StatsJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setError(null);
      try {
        const res = await getJson<StatsJson>(
          `/api/namespaces/${encodeURIComponent(namespace)}/stats`,
          signal,
        );
        setData(res);
      } catch (e) {
        if (isAbort(e)) return;
        setError(e);
        setData(null);
      } finally {
        setLoading(false);
      }
    },
    [namespace],
  );

  useEffect(() => {
    const ac = new AbortController();
    // Fetch on mount / on consistency change: synchronising with the server is
    // exactly what this effect is for; the setState is the loading flag.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load(ac.signal);
    return () => ac.abort();
  }, [load]);

  const backlog = data ? BigInt(data.backlog) : 0n;
  const backlogTone = backlog === 0n ? "ok" : backlog > 1000n ? "danger" : "warn";

  return (
    <div className="p-4 flex flex-col gap-4 max-w-6xl">
      <Panel
        title="index state"
        subtitle="Reads the manifest and each type's metadata — no cluster data — so this call's cost is independent of corpus size."
        actions={
          <>
            <Button
              onClick={() => void load()}
              disabled={loading}
              title="re-run Stats"
            >
              {loading ? "…" : "refresh"}
            </Button>
          </>
        }
      >
        {loading && !data && <Loading label="Stats" />}

        {Boolean(error) && (
          <ErrorBanner
            error={error}
            what={`Stats(${namespace})`}
            onRetry={() => void load()}
          />
        )}

        {data && (
          <>
            <div className="grid gap-2 grid-cols-2 md:grid-cols-4">
              <StatTile
                label="un-indexed backlog"
                value={groupDigits(data.backlog)}
                tone={backlogTone}
                hint="wal_head − wal_index_cursor: what a STRONG query pays to scan"
              />
              <StatTile
                label="generation"
                value={groupDigits(data.generation)}
                hint={
                  data.prevGeneration !== null
                    ? `prev ${groupDigits(data.prevGeneration)}`
                    : "no previous generation"
                }
              />
              <StatTile
                label="wal_head"
                value={groupDigits(data.walHead)}
                hint={`cursor ${groupDigits(data.walIndexCursor)}`}
              />
              <StatTile
                label="doc_count"
                value={groupDigits(data.docCount)}
                hint={`sum over ${data.types.length} memory_type${
                  data.types.length === 1 ? "" : "s"
                }`}
              />
            </div>

            <div className="mt-3 grid gap-x-8 gap-y-1 md:grid-cols-2">
              <KeyValue
                entries={[
                  { k: "namespace", v: data.namespace || namespace },
                  {
                    k: "through_seq",
                    v: groupDigits(data.throughSeq),
                    title: "the WAL sequence this snapshot reflects",
                  },
                  {
                    k: "format_version",
                    v: String(data.formatVersion),
                  },
                ]}
              />
              <KeyValue
                entries={[
                  {
                    k: "tokenizer_hash",
                    v: data.tokenizerConfigHash || "—",
                    title: "tokenizer_config_hash",
                  },
                  {
                    k: "load_roundtrips",
                    v: String(data.loadRoundtrips),
                    title: "object-storage roundtrips this call cost",
                  },
                  { k: "rpc wall clock", v: fmtMs(data.elapsedMs) },
                ]}
              />
            </div>
          </>
        )}
      </Panel>

      <Panel
        title="per memory_type"
        subtitle="Each memory_type is an INDEPENDENT index. There is no corpus-wide equivalent of these numbers except their sum."
        bodyClassName="p-0"
      >
        {data && data.types.length === 0 && (
          <div className="p-3">
            <Empty title="no memory types in this snapshot">
              Nothing has been written to this namespace yet, or the indexer has
              not run. Write with a client, then{" "}
              <span className="font-mono text-ink-dim">
                mlake-server index --namespaces {namespace} --once
              </span>
              .
            </Empty>
          </div>
        )}

        {data && data.types.length > 0 && (
          <TableShell
            head={
              <>
                <Th className="w-24">memory_type</Th>
                <Th className="text-right" title="live memories a query can return">
                  doc_count
                </Th>
                <Th className="text-right">cluster_count</Th>
                <Th
                  className="text-right"
                  title="doc count when IVF centroids were last trained"
                >
                  train_count
                </Th>
                <Th
                  className="text-right"
                  title="doc_count − train_count: what drives an assign-only retrain"
                >
                  train drift
                </Th>
                <Th className="w-28">has_index</Th>
              </>
            }
          >
            {data.types.map((t) => {
              const drift = BigInt(t.docCount) - BigInt(t.trainCount);
              return (
                <tr key={t.memoryType} className="hover:bg-panel-2">
                  <Td className="font-mono">{t.memoryType}</Td>
                  <Td className="font-mono text-right tnum">
                    {groupDigits(t.docCount)}
                  </Td>
                  <Td className="font-mono text-right tnum text-ink-dim">
                    {t.clusterCount}
                  </Td>
                  <Td className="font-mono text-right tnum text-ink-dim">
                    {groupDigits(t.trainCount)}
                  </Td>
                  <Td
                    className={`font-mono text-right tnum ${
                      drift > 0n ? "text-warn" : "text-ink-faint"
                    }`}
                  >
                    {drift > 0n ? "+" : ""}
                    {groupDigits(drift.toString())}
                  </Td>
                  <Td>
                    {t.hasIndex ? (
                      <span className="font-mono text-[11px] text-ok">true</span>
                    ) : (
                      <span
                        className="font-mono text-[11px] text-warn"
                        title="present only in the WAL tail — never indexed"
                      >
                        false · WAL only
                      </span>
                    )}
                  </Td>
                </tr>
              );
            })}
          </TableShell>
        )}

        {!data && !error && loading && (
          <div className="p-3">
            <Loading label="Stats" />
          </div>
        )}

        {!data && Boolean(error) && (
          <div className="p-3 text-[11px] text-ink-faint font-mono">
            per-type table unavailable: the Stats call above failed.
          </div>
        )}
      </Panel>
    </div>
  );
}
