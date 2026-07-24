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

interface ObjectRow {
  key: string;
  size: number;
}
interface ObjectsPayload {
  objects: ObjectRow[];
  total: number;
  totalBytes: number;
}

export function ObjectsView({ namespace }: { namespace: string }) {
  const [data, setData] = useState<ObjectsPayload | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await getJson<ObjectsPayload>(
          `/api/namespaces/${encodeURIComponent(namespace)}/objects?limit=500`,
        ),
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
        title="objects"
        subtitle="Everything this namespace owns in the bucket, largest first — one LIST. Object bodies are binary (rkyv / IVF / SSTable) and are not decoded here; that needs the engine."
        actions={
          <Button onClick={() => void load()} disabled={loading}>
            refresh
          </Button>
        }
        bodyClassName="p-3"
      >
        {loading && !data && <Loading label="listing objects" />}
        {Boolean(error) && <ErrorBanner error={error} what="objects" onRetry={() => void load()} />}
        {data && (
          <div className="grid grid-cols-2 gap-2">
            <StatTile label="objects" value={groupDigits(String(data.total))} />
            <StatTile label="total bytes" value={fmtBytes(String(data.totalBytes))} />
          </div>
        )}
      </Panel>

      {data && (
        <Panel title={`${data.objects.length} largest`} bodyClassName="p-0">
          {data.objects.length === 0 ? (
            <div className="p-3">
              <Empty title="no objects under this prefix" />
            </div>
          ) : (
            <TableShell
              head={
                <>
                  <Th className="text-right">size</Th>
                  <Th>key</Th>
                </>
              }
            >
              {data.objects.map((o) => (
                <tr key={o.key} className="hover:bg-panel-2">
                  <Td className="text-right tnum">{fmtBytes(String(o.size))}</Td>
                  <Td className="font-mono text-[11px] text-ink-dim break-all">{o.key}</Td>
                </tr>
              ))}
            </TableShell>
          )}
        </Panel>
      )}
    </div>
  );
}
