"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import { groupDigits } from "@/lib/format";
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
  Th,
  Toggle,
} from "@/components/ui";

interface IndexJob {
  state: string;
  claimed_by?: string | null;
  heartbeat_ms?: number;
  enqueued_ms?: number;
}
interface QueuePayload {
  queue: { jobs: Record<string, IndexJob>; fetchedMs: number };
}

const REFRESH_MS = 5_000;
/** Past this without a heartbeat a claim is treated as abandoned (the indexer reclaims it too). */
const STALE_CLAIM_MS = 60_000;

function age(ms: number | undefined, now: number): string {
  if (!ms) return "—";
  const d = now - ms;
  if (d < 1000) return "just now";
  const s = Math.round(d / 1000);
  if (s < 90) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}

export function IndexerView() {
  const [data, setData] = useState<QueuePayload | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(true);
  const [now, setNow] = useState(() => Date.now());

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getJson<QueuePayload>("/api/obs?queue=1");
      setData(res);
      setNow(Date.now());
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
      setData(null);
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

  const jobs = useMemo(() => {
    const entries = Object.entries(data?.queue.jobs ?? {});
    // Claimed work first (that's what's actually happening), then oldest-enqueued.
    entries.sort((a, b) => {
      const ca = a[1].claimed_by ? 0 : 1;
      const cb = b[1].claimed_by ? 0 : 1;
      if (ca !== cb) return ca - cb;
      return (a[1].enqueued_ms ?? 0) - (b[1].enqueued_ms ?? 0);
    });
    return entries;
  }, [data]);

  const claimed = jobs.filter(([, j]) => Boolean(j.claimed_by)).length;
  const stale = jobs.filter(
    ([, j]) => j.claimed_by && now - (j.heartbeat_ms ?? 0) > STALE_CLAIM_MS,
  ).length;
  const workers = new Set(
    jobs.map(([, j]) => j.claimed_by).filter((w): w is string => Boolean(w)),
  );
  const oldest = jobs.reduce(
    (acc, [, j]) => (j.enqueued_ms && (!acc || j.enqueued_ms < acc) ? j.enqueued_ms : acc),
    0,
  );

  return (
    <div className="p-4 max-w-6xl mx-auto flex flex-col gap-4">
      <Panel
        title="indexer — work queue"
        subtitle={
          <>
            Read straight from <code>_index-queue.json</code> in the bucket. The indexer coordinates
            entirely through object storage — one CAS&apos;d job per namespace that might need
            folding — so this needs no indexer endpoint and no discovery. A job disappears once its
            namespace is folded and clean.
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
      >
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <StatTile label="queued" value={groupDigits(String(jobs.length))} />
          <StatTile label="claimed" value={groupDigits(String(claimed))} />
          <StatTile
            label="workers"
            value={groupDigits(String(workers.size))}
            hint={workers.size === 0 ? "no indexer holding work" : undefined}
          />
          <StatTile
            label="stale claims"
            value={groupDigits(String(stale))}
            tone={stale > 0 ? "warn" : "normal"}
            hint={stale > 0 ? "no heartbeat — will be reclaimed" : undefined}
          />
        </div>
        {jobs.length > 0 && oldest > 0 && (
          <div className="mt-2 font-mono text-[11px] text-ink-faint">
            oldest job enqueued {age(oldest, now)} — a growing backlog means the indexer is behind
            (reads stay correct; they just carry a longer WAL tail)
          </div>
        )}
      </Panel>

      {loading && !data && <Loading label="reading _index-queue.json" />}
      {Boolean(error) && <ErrorBanner error={error} what="indexer queue" onRetry={() => void load()} />}

      <Panel title={`${jobs.length} jobs`} bodyClassName="p-0">
        {data && jobs.length === 0 ? (
          <div className="p-3">
            <Empty title="queue is empty">
              <p>
                Every namespace is folded and clean. A job appears when a write leaves un-indexed WAL
                behind, and the indexer&apos;s reconcile sweep also enqueues anything it finds dirty.
              </p>
            </Empty>
          </div>
        ) : (
          <TableShell
            head={
              <>
                <Th>namespace</Th>
                <Th>state</Th>
                <Th>claimed by</Th>
                <Th>heartbeat</Th>
                <Th>enqueued</Th>
                <Th />
              </>
            }
          >
            {jobs.map(([ns, j]) => {
              const isStale = Boolean(j.claimed_by) && now - (j.heartbeat_ms ?? 0) > STALE_CLAIM_MS;
              return (
                <tr key={ns} className={isStale ? "bg-warn/5" : undefined}>
                  <Td className="font-mono text-[12px] text-ink">{ns}</Td>
                  <Td>
                    <Tag>{j.state}</Tag>
                  </Td>
                  <Td>
                    {j.claimed_by ? (
                      <CopyableId value={j.claimed_by} />
                    ) : (
                      <span className="text-ink-faint font-mono text-[11px]">unclaimed</span>
                    )}
                  </Td>
                  <Td className={`font-mono text-[11px] ${isStale ? "text-warn" : "text-ink-dim"}`}>
                    {j.claimed_by ? age(j.heartbeat_ms, now) : "—"}
                  </Td>
                  <Td className="font-mono text-[11px] text-ink-dim">{age(j.enqueued_ms, now)}</Td>
                  <Td>
                    <Link
                      href={`/ns/${encodeURIComponent(ns)}`}
                      className="font-mono text-[11px] text-ink-dim hover:text-accent"
                    >
                      inspect →
                    </Link>
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
