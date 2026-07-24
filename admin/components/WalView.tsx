"use client";

import { useCallback, useEffect, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import { fmtBytes, groupDigits } from "@/lib/format";
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
} from "@/components/ui";

interface WalRow {
  seq: number;
  key: string;
  size: number;
}
interface WalPayload {
  entries: WalRow[];
  total: number;
  totalBytes: number;
}

export function WalView({ namespace }: { namespace: string }) {
  const [data, setData] = useState<WalPayload | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await getJson<WalPayload>(`/api/namespaces/${encodeURIComponent(namespace)}/wal?limit=500`),
      );
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
    } finally {
      setLoading(false);
    }
  }, [namespace]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  return (
    <div className="p-4 max-w-5xl mx-auto flex flex-col gap-4">
      <Panel
        title="write-ahead log"
        subtitle="The retained WAL window, newest first — one LIST of the wal/ prefix. This is a window, not a history: once the indexer folds an entry, GC is free to reclaim it, so the oldest sequence here sits well above zero. Entry payloads are binary and are not decoded here."
        actions={
          <Button onClick={() => void load()} disabled={loading}>
            refresh
          </Button>
        }
        bodyClassName="p-3"
      >
        {loading && !data && <Loading label="listing wal/" />}
        {Boolean(error) && <ErrorBanner error={error} what="wal" onRetry={() => void load()} />}
        {data && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <StatTile label="retained entries" value={groupDigits(String(data.total))} />
            <StatTile label="bytes" value={fmtBytes(String(data.totalBytes))} />
            <StatTile
              label="newest seq"
              value={data.entries[0] ? groupDigits(String(data.entries[0].seq)) : "—"}
            />
          </div>
        )}
      </Panel>

      {data && (
        <Panel title={`${data.entries.length} shown`} bodyClassName="p-0">
          {data.entries.length === 0 ? (
            <div className="p-3">
              <Empty title="no retained WAL entries">
                <p>Everything has been folded and reclaimed.</p>
              </Empty>
            </div>
          ) : (
            <TableShell
              head={
                <>
                  <Th className="text-right">seq</Th>
                  <Th className="text-right">size</Th>
                  <Th>object</Th>
                </>
              }
            >
              {data.entries.map((e) => (
                <tr key={e.key} className="hover:bg-panel-2">
                  <Td className="text-right tnum">{groupDigits(String(e.seq))}</Td>
                  <Td className="text-right tnum text-ink-dim">{fmtBytes(String(e.size))}</Td>
                  <Td className="font-mono text-[11px] text-ink-faint truncate">{e.key}</Td>
                </tr>
              ))}
            </TableShell>
          )}
        </Panel>
      )}
    </div>
  );
}
