"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { getJson, isAbort, postJson } from "@/lib/client";
import { fmtMs } from "@/lib/format";
import type { CreateNamespaceJson, ListNamespacesJson } from "@/lib/types";
import {
  Button,
  Empty,
  ErrorBanner,
  Field,
  Loading,
  Panel,
  TextInput,
} from "@/components/ui";

export function NamespacesView() {
  const [data, setData] = useState<ListNamespacesJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<unknown>(null);
  const [created, setCreated] = useState<CreateNamespaceJson | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      setData(await getJson<ListNamespacesJson>("/api/namespaces", signal));
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    // Fetch on mount: the effect synchronises this component with an external
    // system (the memlake server) and flips the loading flag while it waits.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load(ac.signal);
    return () => ac.abort();
  }, [load]);

  async function create() {
    const ns = name.trim();
    if (!ns) return;
    setCreating(true);
    setCreateError(null);
    setCreated(null);
    try {
      const res = await postJson<CreateNamespaceJson>("/api/namespaces", {
        namespace: ns,
      });
      setCreated(res);
      setName("");
      await load();
    } catch (e) {
      setCreateError(e);
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="p-4 max-w-5xl mx-auto flex flex-col gap-4">
      <Panel
        title="namespaces"
        subtitle={
          <>
            One LIST over the bucket&apos;s manifest objects. This is an operator
            call, not something to poll per request.
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
        {loading && !data && <Loading label="ListNamespaces" />}

        {Boolean(error) && (
          <div className="p-3">
            <ErrorBanner
              error={error}
              what="ListNamespaces"
              onRetry={() => void load()}
            />
          </div>
        )}

        {data && data.namespaces.length === 0 && (
          <div className="p-3">
            <Empty title="no namespaces in this bucket">
              <p>
                Create one below, or write to it from a client — a namespace
                appears once its manifest object exists.
              </p>
              <p className="mt-2 font-mono text-ink-dim">
                mlake-server serve --addr 0.0.0.0:50051
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
                      className="text-ink-faint hover:text-accent"
                    >
                      stats
                    </Link>
                    <Link
                      href={`/ns/${encodeURIComponent(ns)}/browse`}
                      className="text-ink-faint hover:text-accent"
                    >
                      browse
                    </Link>
                    <Link
                      href={`/ns/${encodeURIComponent(ns)}/query`}
                      className="text-ink-faint hover:text-accent"
                    >
                      query
                    </Link>
                  </nav>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel
        title="create namespace"
        subtitle="CreateNamespace is idempotent — safe to call on every startup, and the only mutation this console performs."
      >
        <div className="flex items-end gap-2">
          <Field label="namespace" className="flex-1">
            <TextInput
              value={name}
              placeholder="my-bank"
              spellCheck={false}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void create();
              }}
            />
          </Field>
          <Button
            variant="primary"
            onClick={() => void create()}
            disabled={creating || !name.trim()}
            className="mb-px py-1.5"
          >
            {creating ? "creating…" : "CreateNamespace"}
          </Button>
        </div>

        {created && (
          <p className="mt-2 font-mono text-[11px] text-ok">
            created {created.namespace} in {fmtMs(created.elapsedMs)}
          </p>
        )}
        {Boolean(createError) && (
          <div className="mt-2">
            <ErrorBanner error={createError} what="CreateNamespace" />
          </div>
        )}
      </Panel>
    </div>
  );
}
