"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import { fmtMs } from "@/lib/format";
import { Button, Empty, ErrorBanner, Loading, Panel } from "@/components/ui";

interface ListNamespaces {
  namespaces: string[];
  elapsedMs: number;
}

export function NamespacesView() {
  const [data, setData] = useState<ListNamespaces | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getJson<ListNamespaces>("/api/namespaces"));
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  return (
    <div className="p-4 max-w-5xl mx-auto flex flex-col gap-4">
      <Panel
        title="namespaces"
        subtitle={
          <>
            Discovered from the bucket itself — a namespace IS a{" "}
            <code>{"{name}/manifest.json"}</code> object, which is exactly how the server finds them.
            This console is read-only and depends on object storage alone.
          </>
        }
        actions={
          <>
            {data && (
              <span className="font-mono text-[11px] text-ink-faint tnum">
                {data.namespaces.length} · {fmtMs(data.elapsedMs)}
              </span>
            )}
            <Button onClick={() => void load()} disabled={loading}>
              refresh
            </Button>
          </>
        }
        bodyClassName="p-0"
      >
        {loading && !data && (
          <div className="p-3">
            <Loading label="listing namespaces" />
          </div>
        )}

        {Boolean(error) && (
          <div className="p-3">
            <ErrorBanner error={error} what="namespaces" onRetry={() => void load()} />
          </div>
        )}

        {data && data.namespaces.length === 0 && !error && (
          <div className="p-3">
            <Empty title="no namespaces in this bucket">
              <p>
                A namespace appears here once its manifest object exists — write to it from a client,
                or let the indexer fold it.
              </p>
            </Empty>
          </div>
        )}

        {data && data.namespaces.length > 0 && (
          <ul className="divide-y divide-line">
            {data.namespaces.map((ns) => (
              <li key={ns}>
                <div className="flex items-center gap-3 px-3 py-2 hover:bg-panel-2">
                  <Link
                    href={`/ns/${encodeURIComponent(ns)}`}
                    className="font-mono text-[13px] text-ink hover:text-accent min-w-0 truncate flex-1"
                  >
                    {ns}
                  </Link>
                  <nav className="flex items-center gap-3 shrink-0 font-mono text-[11px]">
                    <Link
                      href={`/ns/${encodeURIComponent(ns)}`}
                      className="text-ink-dim hover:text-accent"
                    >
                      stats
                    </Link>
                    <Link
                      href={`/ns/${encodeURIComponent(ns)}/wal`}
                      className="text-ink-dim hover:text-accent"
                    >
                      wal
                    </Link>
                    <Link
                      href={`/ns/${encodeURIComponent(ns)}/objects`}
                      className="text-ink-dim hover:text-accent"
                    >
                      objects
                    </Link>
                    <Link
                      href={`/traces?namespace=${encodeURIComponent(ns)}`}
                      className="text-ink-dim hover:text-accent"
                    >
                      traces
                    </Link>
                  </nav>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
