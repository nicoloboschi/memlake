"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { isAbort, postJson } from "@/lib/client";
import { cmpU64, fmtBytes, fmtMs, fraction, groupDigits } from "@/lib/format";
import type { CacheEntryJson, CacheStatsJson } from "@/lib/types";
import {
  Button,
  Empty,
  ErrorBanner,
  Field,
  Loading,
  NumberInput,
  Panel,
  StatTile,
  Td,
  TableShell,
  Th,
} from "@/components/ui";

type SortKey = "lru" | "bytes";

export function CacheView({ namespace }: { namespace: string | null }) {
  const [limit, setLimit] = useState(200);
  const [data, setData] = useState<CacheStatsJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  // MRU-first is the order the server returns and the default we keep: the hot
  // working set is the interesting part of a cache listing.
  const [sortKey, setSortKey] = useState<SortKey>("lru");

  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (lim: number) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setLoading(true);
      setError(null);
      try {
        const res = await postJson<CacheStatsJson>(
          "/api/cache",
          { namespace: namespace ?? "", limit: lim },
          ac.signal,
        );
        setData(res);
      } catch (e) {
        if (isAbort(e)) return;
        setError(e);
        setData(null);
      } finally {
        if (abortRef.current === ac) setLoading(false);
      }
    },
    [namespace],
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void run(limit);
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namespace]);

  const rows = useMemo(() => {
    if (!data) return [];
    if (sortKey === "lru") return data.entries; // server order, MRU first
    return [...data.entries].sort(
      (a, b) => cmpU64(b.bytes, a.bytes) || a.lruRank - b.lruRank,
    );
  }, [data, sortKey]);

  const scope = namespace ? `namespace ${namespace}` : "every namespace";

  return (
    <div className="p-4 flex flex-col gap-4 max-w-7xl">
      <NodeLocalNotice namespace={namespace} />

      <Panel
        title="read cache"
        subtitle={
          <>
            Two tiers, each bounded independently — peak RAM and peak disk are
            capped by construction whatever the workload. Scope: {scope}.
          </>
        }
        actions={
          <>
            <Field label="limit" className="w-24">
              <NumberInput
                min={0}
                max={2000}
                step={100}
                value={limit}
                disabled={loading}
                onChange={(e) => setLimit(Number(e.target.value) || 0)}
              />
            </Field>
            <Button
              className="mt-4"
              disabled={loading}
              onClick={() => void run(limit)}
            >
              {loading ? "…" : "refresh"}
            </Button>
          </>
        }
      >
        {loading && !data && <Loading label="CacheStats" />}

        {Boolean(error) && (
          <ErrorBanner
            error={error}
            what="CacheStats"
            onRetry={() => void run(limit)}
          />
        )}

        {data && !data.enabled && (
          <Empty title="this replica has no read cache configured">
            <p>
              Every read goes straight through to object storage. That is a
              latency choice, not a correctness one — the node still answers
              exactly the same queries.
            </p>
            <p className="mt-2 font-mono text-ink-dim">
              mlake-server serve --mem-mb 256 --disk-mb 4096
            </p>
          </Empty>
        )}

        {data && data.enabled && (
          <>
            <div className="grid gap-2 grid-cols-2 md:grid-cols-4">
              <BudgetTile
                label="memory"
                used={data.memBytes}
                budget={data.memBudget}
              />
              <BudgetTile
                label="disk"
                used={data.diskBytes}
                budget={data.diskBudget}
              />
              <StatTile
                label="hit ratio"
                value={
                  data.hitRatio === null ? (
                    <span className="text-ink-faint text-[15px]">
                      no lookups yet
                    </span>
                  ) : (
                    `${(data.hitRatio * 100).toFixed(1)}%`
                  )
                }
                hint={
                  data.hitRatio === null ? (
                    "hits + misses = 0 — nothing has been looked up on this node"
                  ) : (
                    <>
                      {groupDigits(data.hits)} hits / {groupDigits(data.misses)}{" "}
                      misses over {groupDigits(data.lookups)} lookups
                    </>
                  )
                }
              />
              <StatTile
                label="cached blocks"
                value={groupDigits(data.totalEntries)}
                hint={
                  data.truncated ? (
                    <span className="text-warn">
                      showing {data.entries.length} — list truncated
                    </span>
                  ) : (
                    `all ${data.entries.length} listed below`
                  )
                }
              />
            </div>

            <ResidencySummary data={data} />
          </>
        )}
      </Panel>

      {data && data.enabled && data.byKind.length > 0 && (
        <Panel
          title="by object kind"
          subtitle={
            <>
              Inferred from the object key — memlake does not label kinds on the
              wire.{" "}
              {data.truncated
                ? "Covers only the blocks returned below, not the whole cache."
                : "Covers every cached block."}
            </>
          }
          bodyClassName="p-0"
        >
          <TableShell
            head={
              <>
                <Th className="w-48">kind</Th>
                <Th className="w-24 text-right">blocks</Th>
                <Th className="w-28 text-right">bytes</Th>
                <Th>share of listed bytes</Th>
              </>
            }
          >
            {data.byKind.map((k) => {
              const total = data.byKind.reduce((acc, x) => {
                try {
                  return acc + BigInt(x.bytes);
                } catch {
                  return acc;
                }
              }, 0n);
              const frac = total > 0n ? fraction(k.bytes, total.toString()) : 0;
              return (
                <tr key={k.kind} className="hover:bg-panel-2">
                  <Td className="font-mono">{k.kind}</Td>
                  <Td className="font-mono text-right tnum text-ink-dim">
                    {k.count}
                  </Td>
                  <Td className="font-mono text-right tnum">
                    {fmtBytes(k.bytes)}
                  </Td>
                  <Td>
                    <Meter fraction={frac ?? 0} />
                  </Td>
                </tr>
              );
            })}
          </TableShell>
        </Panel>
      )}

      {data && data.enabled && (
        <Panel
          title="cached blocks"
          subtitle={
            <>
              {sortKey === "lru"
                ? "Most-recently-used first, as the node returned them — this is the hot working set."
                : "Sorted by size. Switch back to LRU to see the working set."}{" "}
              lru_rank orders entries against each other and says nothing about
              wall-clock time. Rows are cache <em>blocks</em>: the cache is keyed
              by (path, byte range), so one object can appear more than once.
            </>
          }
          actions={
            <>
              <Button
                variant={sortKey === "lru" ? "primary" : "default"}
                onClick={() => setSortKey("lru")}
              >
                lru order
              </Button>
              <Button
                variant={sortKey === "bytes" ? "primary" : "default"}
                onClick={() => setSortKey("bytes")}
              >
                by size
              </Button>
              <span className="font-mono text-[11px] text-ink-faint">
                {fmtMs(data.elapsedMs)}
              </span>
            </>
          }
          bodyClassName="p-0"
        >
          {data.truncated && (
            <p className="px-3 py-1.5 border-b border-warn/30 bg-warn/5 font-mono text-[11px] text-warn">
              showing {data.entries.length} of{" "}
              {groupDigits(data.totalEntries)} cached blocks — the list is capped
              at limit = {data.limit || "server default"}. This is not the whole
              cache.
            </p>
          )}

          {data.entries.length === 0 ? (
            <div className="p-3">
              <Empty title={`nothing cached for ${scope}`}>
                The cache is configured but holds no objects for this scope yet.
                Run a query against the namespace and refresh — reads populate it
                on the way through.
              </Empty>
            </div>
          ) : (
            <TableShell
              head={
                <>
                  <Th className="w-16 text-right" title="0 = most recently used">
                    lru
                  </Th>
                  {!namespace && <Th className="w-40">namespace</Th>}
                  <Th>path</Th>
                  <Th
                    className="w-32"
                    title="a ranged read is keyed by (path, byte range), so one object can be resident as several independent blocks"
                  >
                    byte range
                  </Th>
                  <Th className="w-36">kind</Th>
                  <Th className="w-24 text-right">size</Th>
                  <Th
                    className="w-40"
                    title="in_memory and on_disk are independent — an object is commonly in both"
                  >
                    residency
                  </Th>
                </>
              }
            >
              {rows.map((e) => (
                <tr key={`${e.namespace}/${e.path}/${e.etag}`} className="hover:bg-panel-2">
                  <Td className="font-mono text-right tnum text-ink-faint">
                    {e.lruRank}
                  </Td>
                  {!namespace && (
                    <Td className="font-mono text-[11px] text-ink-dim truncate">
                      {e.namespace || "—"}
                    </Td>
                  )}
                  <Td
                    className="font-mono text-[11px] break-all"
                    title={e.etag ? `etag ${e.etag}` : undefined}
                  >
                    <PathCell path={e.object} />
                  </Td>
                  <Td className="font-mono text-[11px] text-ink-dim tnum">
                    {e.range ?? (
                      <span className="text-ink-faint/60" title="whole object">
                        whole
                      </span>
                    )}
                  </Td>
                  <Td className="font-mono text-[11px] text-ink-dim">{e.kind}</Td>
                  <Td className="font-mono text-right tnum">
                    {fmtBytes(e.bytes)}
                  </Td>
                  <Td>
                    <Residency entry={e} />
                  </Td>
                </tr>
              ))}
            </TableShell>
          )}
        </Panel>
      )}
    </div>
  );
}

/**
 * The caveat that makes the rest of the page safe to read. Top of the page, not
 * a footnote: an operator who takes these numbers for "the cache" will draw
 * conclusions the data cannot support.
 */
function NodeLocalNotice({ namespace }: { namespace: string | null }) {
  return (
    <div className="border border-warn/40 bg-warn/5 rounded-sm px-3 py-2.5">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-mono text-[10px] uppercase tracking-[0.1em] px-1.5 py-0.5 rounded-sm border border-warn/50 text-warn">
          node-local
        </span>
        <span className="font-mono text-[11px] text-ink-dim">
          not a cluster-wide view
        </span>
        {namespace ? (
          <Link
            href="/cache"
            className="ml-auto font-mono text-[11px] text-accent hover:underline"
          >
            all namespaces →
          </Link>
        ) : null}
      </div>
      <p className="mt-1.5 text-[11px] text-ink leading-relaxed max-w-4xl">
        Every other RPC in this API is answerable identically by any replica,
        because the answer lives in object storage. This one is not: the cache is
        process-local, so these numbers describe{" "}
        <strong>whichever pod this request happened to land on</strong>, and two
        calls behind a load balancer will disagree. Do not read them as “the
        cache”.
      </p>
      <p className="mt-1 text-[11px] text-ink-dim leading-relaxed max-w-4xl">
        Nothing here affects correctness. A cold cache costs latency and nothing
        else (INV-4) — it can never change a query result.
      </p>
    </div>
  );
}

/**
 * mem_entries and disk_entries overlap. Presenting them side by side without
 * saying so invites an operator to add them, which is exactly wrong.
 */
function ResidencySummary({ data }: { data: CacheStatsJson }) {
  const inBoth = data.entries.filter((e) => e.inMemory && e.onDisk).length;
  const memOnly = data.entries.filter((e) => e.inMemory && !e.onDisk).length;
  const diskOnly = data.entries.filter((e) => !e.inMemory && e.onDisk).length;

  return (
    <div className="mt-3 border-t border-line pt-3">
      <div className="flex items-baseline gap-x-6 gap-y-1 flex-wrap font-mono text-[11px]">
        <span className="text-ink-faint uppercase tracking-[0.1em] text-[10px]">
          residency
        </span>
        <span className="text-ink-dim">
          mem_entries <span className="text-ink tnum">{groupDigits(data.memEntries)}</span>
        </span>
        <span className="text-ink-dim">
          disk_entries{" "}
          <span className="text-ink tnum">{groupDigits(data.diskEntries)}</span>
        </span>
        <span className="text-ink-dim">
          total_entries{" "}
          <span className="text-ink tnum">{groupDigits(data.totalEntries)}</span>
        </span>
      </div>
      <p className="mt-1.5 text-[11px] text-ink-dim leading-relaxed max-w-4xl">
        The tiers <strong>overlap</strong> — they are not a partition. A memory
        eviction demotes an object to disk without dropping its bytes, and a
        later hit promotes it back, so an object is commonly resident in both.{" "}
        <span className="text-warn">
          mem_entries + disk_entries is therefore not the object count
        </span>{" "}
        — total_entries is.
      </p>
      {data.entries.length > 0 && (
        <p className="mt-1 font-mono text-[10px] text-ink-faint">
          of the {data.entries.length} listed: {inBoth} in memory+disk ·{" "}
          {memOnly} memory only · {diskOnly} disk only
        </p>
      )}
    </div>
  );
}

/** A tile whose value is a usage-against-budget pair, with its own meter. */
function BudgetTile({
  label,
  used,
  budget,
}: {
  label: string;
  used: string;
  budget: string;
}) {
  const frac = fraction(used, budget);
  return (
    <div className="border border-line bg-panel-2 rounded-sm px-3 py-2 min-w-0">
      <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint">
        {label}
      </div>
      <div className="mt-1 font-mono text-[19px] leading-6 truncate text-ink">
        {fmtBytes(used)}
      </div>
      <div className="mt-1">
        <Meter fraction={frac ?? 0} />
      </div>
      <div className="mt-1 text-[10px] text-ink-faint">
        {frac === null ? (
          "no budget reported"
        ) : (
          <>
            {(frac * 100).toFixed(frac >= 0.1 ? 0 : 1)}% of {fmtBytes(budget)}
          </>
        )}
      </div>
    </div>
  );
}

/**
 * A plain magnitude bar in one hue — deliberately NOT a severity meter.
 *
 * A bounded LRU cache at 100% is its normal steady state: it means the budget
 * is being used, which is the point of having one. Colouring "full" as a
 * warning would tell an operator to act on healthy behaviour.
 */
function Meter({ fraction: f }: { fraction: number }) {
  const pct = Math.max(0, Math.min(1, f)) * 100;
  return (
    <div
      className="h-1.5 w-full rounded-sm overflow-hidden"
      style={{ background: "var(--color-viz-track)" }}
      role="img"
      aria-label={`${pct.toFixed(0)}% of budget`}
    >
      <div
        className="h-full rounded-sm"
        style={{
          width: `${pct}%`,
          background: "var(--color-viz-1)",
        }}
      />
    </div>
  );
}

function Residency({ entry }: { entry: CacheEntryJson }) {
  return (
    <span className="inline-flex items-center gap-1">
      <TierFlag on={entry.inMemory} label="mem" title="in_memory" />
      <TierFlag on={entry.onDisk} label="disk" title="on_disk" />
      {!entry.inMemory && !entry.onDisk && (
        <span className="font-mono text-[10px] text-ink-faint">neither</span>
      )}
    </span>
  );
}

/** One tier's independent boolean. Never rendered as an exclusive choice. */
function TierFlag({
  on,
  label,
  title,
}: {
  on: boolean;
  label: string;
  title: string;
}) {
  return (
    <span
      title={`${title} = ${on}`}
      className={`font-mono text-[10px] leading-4 px-1.5 py-px rounded-sm border ${
        on
          ? "border-viz-1/60 bg-viz-1/15 text-viz-1"
          : "border-line text-ink-faint/50"
      }`}
    >
      {label}
    </span>
  );
}

/** Dim the directory prefix so the object name — the useful bit — leads. */
function PathCell({ path }: { path: string }) {
  const cut = path.lastIndexOf("/");
  if (cut < 0) return <span className="text-ink">{path}</span>;
  return (
    <>
      <span className="text-ink-faint">{path.slice(0, cut + 1)}</span>
      <span className="text-ink">{path.slice(cut + 1)}</span>
    </>
  );
}
