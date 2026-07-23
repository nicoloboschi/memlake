"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { isAbort, postJson } from "@/lib/client";
import { cmpU64, fmtBytes, fmtMs, groupDigits, sharePct } from "@/lib/format";
import { kindCopy, kindRank } from "@/lib/objectKinds";
import {
  STORAGE_OBJECT_KINDS,
  type DecodeObjectJson,
  type DecodeObjectRequestBody,
  type ListObjectsJson,
  type ListObjectsRequestBody,
  type ObjectInfoJson,
  type ObjectKindSummaryJson,
  type StorageObjectKind,
} from "@/lib/types";
import {
  Button,
  Empty,
  ErrorBanner,
  Field,
  Loading,
  NumberInput,
  Panel,
  SegmentedControl,
  Select,
  StatTile,
  Td,
  TableShell,
  Th,
} from "@/components/ui";

const COLS = 6;

type SortKey = "segment" | "size" | "path";
type LiveFilter = "all" | "live" | "dead";

const SORTS = [
  {
    value: "segment" as const,
    label: "by segment",
    title:
      "group each segment's files together, then by kind — the manifest and WAL entries (no segment) sort first",
  },
  { value: "size" as const, label: "by size", title: "largest object first" },
  { value: "path" as const, label: "by path", title: "the server's own key order" },
];

const LIVE_FILTERS = [
  { value: "all" as const, label: "all" },
  { value: "live" as const, label: "live" },
  { value: "dead" as const, label: "dead" },
];

export function ObjectsView({ namespace }: { namespace: string }) {
  const [limit, setLimit] = useState(500);

  /**
   * The page token is opaque and only meaningful to the server, so page numbers
   * do not exist. Same cursor-stack shape as Scan and the WAL: the last element
   * is the token that produced what is on screen ("" for the first page), and
   * "back" pops it.
   */
  const [stack, setStack] = useState<string[]>([""]);
  const [data, setData] = useState<ListObjectsJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  // View state over the page. These are client-side and the view says so — the
  // server pages by key, not by kind or size.
  const [kindFilter, setKindFilter] = useState<"ALL" | StorageObjectKind>("ALL");
  const [liveFilter, setLiveFilter] = useState<LiveFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("segment");
  const [selected, setSelected] = useState<ObjectInfoJson | null>(null);
  const [showAllKinds, setShowAllKinds] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (tokens: string[], lim: number) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setLoading(true);
      setError(null);

      const body: ListObjectsRequestBody = {
        limit: lim,
        pageToken: tokens[tokens.length - 1] ?? "",
      };
      try {
        const res = await postJson<ListObjectsJson>(
          `/api/namespaces/${encodeURIComponent(namespace)}/objects`,
          body,
          ac.signal,
        );
        setData(res);
        setStack(tokens);
        setSelected(null);
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
    void run([""], limit);
    return () => abortRef.current?.abort();
    // Keyed on namespace only: a limit change is an explicit re-read.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namespace]);

  // Memoized so the identity of the empty case is stable — the derived sort
  // below depends on it.
  const objects = useMemo(() => data?.objects ?? [], [data]);

  const rows = useMemo(() => {
    let out = objects;
    if (kindFilter !== "ALL") out = out.filter((o) => o.kind === kindFilter);
    if (liveFilter !== "all") {
      const wantLive = liveFilter === "live";
      out = out.filter((o) => o.live === wantLive);
    }
    const sorted = [...out];
    if (sortKey === "size") {
      sorted.sort(
        (a, b) => cmpU64(b.sizeBytes, a.sizeBytes) || a.path.localeCompare(b.path),
      );
    } else if (sortKey === "path") {
      sorted.sort((a, b) => a.path.localeCompare(b.path));
    } else {
      // Group each segment's files together, then the kind order, then path.
      // Objects with no segment in their key (the manifest, WAL entries) sort
      // first — they are what the segments are published through.
      sorted.sort(
        (a, b) =>
          groupRank(a) - groupRank(b) ||
          a.segment.localeCompare(b.segment) ||
          kindRank(a.kind) - kindRank(b.kind) ||
          a.path.localeCompare(b.path),
      );
    }
    return sorted;
  }, [objects, kindFilter, liveFilter, sortKey]);

  const kindsOnPage = useMemo(
    () => (data?.byKind ?? []).map((k) => k.kind),
    [data],
  );

  const canBack = stack.length > 1;
  const canNext = Boolean(data?.nextPageToken);
  const filtered = rows.length !== objects.length;

  return (
    <div className="flex min-h-full">
      <div className="flex-1 min-w-0 p-4 flex flex-col gap-4">
        <Panel
          title="object storage"
          subtitle={
            <>
              The <em>physical</em> view: every key this namespace owns on S3,
              versus <span className="font-mono">stats</span>, which is the
              logical one. Keys are{" "}
              <span className="font-mono">{"{ns}/manifest.json"}</span>,{" "}
              <span className="font-mono">{"{ns}/wal/{seq}.bin"}</span> and{" "}
              <span className="font-mono">
                {"{ns}/seg-{id}/mt{type}/{file}"}
              </span>
              .
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
                onClick={() => void run([""], limit)}
              >
                {loading ? "…" : "reload"}
              </Button>
            </>
          }
        >
          {loading && !data && <Loading label="ListObjects" />}

          {Boolean(error) && (
            <ErrorBanner
              error={error}
              what={`ListObjects(${namespace})`}
              onRetry={() => void run(stack, limit)}
            />
          )}

          {data && (
            <>
              <div className="grid gap-2 grid-cols-2 md:grid-cols-4">
                <StatTile
                  label="objects"
                  value={groupDigits(data.totalObjects)}
                  hint={`${objects.length} on this page · ${fmtMs(data.elapsedMs)}`}
                />
                <StatTile
                  label="total bytes"
                  value={fmtBytes(data.totalBytes)}
                  hint="everything under the namespace prefix"
                />
                <StatTile
                  label="dead bytes"
                  value={fmtBytes(data.deadBytes)}
                  hint={
                    data.deadShare === null ? (
                      "nothing stored yet"
                    ) : (
                      <>
                        {(data.deadShare * 100).toFixed(
                          data.deadShare >= 0.1 ? 0 : 1,
                        )}
                        % of the namespace · not referenced by the current
                        manifest
                      </>
                    )
                  }
                />
                <StatTile
                  label="generation"
                  value={groupDigits(data.generation)}
                  hint="what the manifest currently points at"
                />
              </div>

              <LiveDeadSplit data={data} />
            </>
          )}
        </Panel>

        {data && data.byKind.length > 0 && (
          <Panel
            title="by object kind"
            subtitle={
              data.complete ? (
                <>
                  Every object in the namespace. Shares are of{" "}
                  {fmtBytes(data.pageBytes)}.
                </>
              ) : (
                <>
                  Covers the {objects.length} objects on this page only — the
                  server pages by key, and per-kind totals are not reported
                  namespace-wide. Shares are of the{" "}
                  {fmtBytes(data.pageBytes)} listed here, not of the{" "}
                  {fmtBytes(data.totalBytes)} stored.
                </>
              )
            }
            actions={
              <Button onClick={() => setShowAllKinds((v) => !v)}>
                {showAllKinds ? "only kinds present" : "all kinds"}
              </Button>
            }
            bodyClassName="p-0"
          >
            <KindTable
              byKind={data.byKind}
              pageBytes={data.pageBytes}
              showAll={showAllKinds}
              selected={kindFilter}
              onSelect={(k) => setKindFilter(k)}
            />
          </Panel>
        )}

        <Panel
          title="objects"
          subtitle={
            data ? (
              <>
                page {stack.length} · showing {rows.length}
                {filtered ? ` of ${objects.length} listed` : ""} ·{" "}
                {data.nextPageToken ? "more to come" : "end of listing"}. Filter
                and sort apply to <em>this page</em>; paging is by key,
                server-side.
              </>
            ) : undefined
          }
          actions={
            <>
              <Button
                disabled={!canBack || loading}
                onClick={() => void run(stack.slice(0, -1), limit)}
              >
                ← back
              </Button>
              <Button
                disabled={!canNext || loading}
                onClick={() =>
                  void run([...stack, data?.nextPageToken ?? ""], limit)
                }
              >
                next →
              </Button>
            </>
          }
          bodyClassName="p-0"
        >
          {data && (
            <div className="flex items-end gap-4 flex-wrap px-3 py-2 border-b border-line">
              <Field label="kind" className="w-56">
                <Select<"ALL" | StorageObjectKind>
                  value={kindFilter}
                  disabled={loading}
                  onChange={setKindFilter}
                  options={[
                    { value: "ALL", label: `all kinds (${objects.length})` },
                    ...kindsOnPage.map((k) => ({
                      value: k,
                      label: `${kindCopy(k).label} — ${k}`,
                    })),
                  ]}
                />
              </Field>
              <div className="pb-0.5">
                <span className="block font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1">
                  live
                </span>
                <SegmentedControl
                  value={liveFilter}
                  onChange={setLiveFilter}
                  options={LIVE_FILTERS}
                  disabled={loading}
                />
              </div>
              <div className="pb-0.5">
                <span className="block font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1">
                  sort
                </span>
                <SegmentedControl
                  value={sortKey}
                  onChange={setSortKey}
                  options={SORTS}
                  disabled={loading}
                />
              </div>
            </div>
          )}

          {/*
            Paging is by key, so a segment's files land wherever its prefix sorts
            — often not on page one, and one segment can straddle a page boundary.
            Flag when nothing on this page is live, so an operator does not read the
            page as the current segment set.
          */}
          {data &&
            objects.length > 0 &&
            !objects.some((o) => o.live) && (
              <p className="px-3 py-1.5 border-b border-line bg-panel-2 text-[11px] text-ink-dim">
                Nothing on this page is live — the listing pages by key, so the
                current segments&apos; files may sort onto another page.
              </p>
            )}

          {loading && !data && (
            <div className="p-3">
              <Loading label="ListObjects" />
            </div>
          )}

          {data && rows.length === 0 && (
            <div className="p-3">
              <Empty title="no objects match">
                {objects.length === 0
                  ? "This page of the listing is empty — the namespace may hold nothing yet, or the cursor walked past the end."
                  : "The kind / live filters exclude every object on this page. Filters are page-local; try the next page or widen them."}
              </Empty>
            </div>
          )}

          {data && rows.length > 0 && (
            <TableShell
              head={
                <>
                  <Th>path</Th>
                  <Th className="w-44">kind</Th>
                  <Th className="w-24 text-right">size</Th>
                  <Th className="w-28" title="the seg-{id} prefix parsed out of the key">
                    segment
                  </Th>
                  <Th
                    className="w-20 text-right"
                    title="only keys under mt{n}/ carry a memory_type"
                  >
                    type
                  </Th>
                  <Th
                    className="w-28"
                    title="whether the CURRENT manifest still references this object"
                  >
                    live
                  </Th>
                </>
              }
            >
              {rows.map((o, i) => (
                <ObjectRow
                  key={o.path}
                  object={o}
                  groupHeader={
                    sortKey === "segment" && startsGroup(rows, i)
                      ? groupLabel(o)
                      : null
                  }
                  selected={selected?.path === o.path}
                  onSelect={() => setSelected(o)}
                />
              ))}
            </TableShell>
          )}
        </Panel>
      </div>

      {selected && (
        <DecodeDrawer
          // Remount per path: the decode state belongs to the selected object.
          key={selected.path}
          namespace={namespace}
          object={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

// ---- live vs dead -----------------------------------------------------------

/**
 * The reason this page exists.
 *
 * Every object except the manifest is immutable. The index is segmented (LSM): a
 * flush appends ONE new L0 segment and carries the rest forward by reference, so
 * publishing does not orphan the old files — only a compaction (merging several
 * segments into one) turns its inputs into garbage, alongside folded WAL entries.
 * Deliberately NOT styled as an error — accumulating garbage between GC runs is
 * correct behaviour, and colouring it red would tell an operator to act on a
 * healthy namespace.
 */
function LiveDeadSplit({ data }: { data: ListObjectsJson }) {
  const dead = data.deadShare ?? 0;
  const live = Math.max(0, Math.min(1, 1 - dead));
  return (
    <div className="mt-3 border-t border-line pt-3">
      <div
        className="h-2 w-full rounded-sm overflow-hidden flex"
        style={{ background: "var(--color-line)" }}
        role="img"
        aria-label={`${(live * 100).toFixed(0)}% live bytes, ${(dead * 100).toFixed(0)}% dead bytes`}
      >
        <div
          className="h-full"
          style={{ width: `${live * 100}%`, background: "var(--color-viz-1)" }}
        />
        <div
          className="h-full"
          style={{
            width: `${Math.max(0, Math.min(1, dead)) * 100}%`,
            background: "var(--color-viz-2)",
          }}
        />
      </div>
      <div className="mt-1.5 flex items-baseline gap-x-5 gap-y-1 flex-wrap font-mono text-[11px]">
        <Swatch color="var(--color-viz-1)">
          live {fmtBytes(data.liveBytes)}{" "}
          <span className="text-ink-faint">
            ({sharePct(data.liveBytes, data.totalBytes)})
          </span>
        </Swatch>
        <Swatch color="var(--color-viz-2)">
          dead {fmtBytes(data.deadBytes)}{" "}
          <span className="text-ink-faint">
            ({sharePct(data.deadBytes, data.totalBytes)})
          </span>
        </Swatch>
      </div>
      <p className="mt-1.5 text-[11px] text-ink-dim leading-relaxed max-w-4xl">
        Dead means <strong>no longer referenced by the current manifest</strong>,
        not broken. Nothing is rewritten in place, but a flush does NOT orphan the
        older files — it appends one new segment and carries the rest forward by
        reference. Dead bytes come from folded WAL entries and from{" "}
        <strong>compaction</strong>, which merges several segments into one and
        leaves the inputs unreferenced. GC reclaims them after a grace window (the
        manifest keeps the <span className="font-mono">prev</span> segment set
        readable for in-flight readers), so a healthy namespace always carries some
        dead bytes — a large and growing share is a signal that GC is not keeping
        up, not that anything is wrong with the data.
      </p>
      <p className="mt-1 text-[11px] text-ink-faint leading-relaxed max-w-4xl">
        Each segment lives under a randomly-suffixed{" "}
        <span className="font-mono">seg-{"{id}"}</span> prefix, which is what makes
        building safe under concurrency: two indexers may fold at once, each writing
        its own segment directory, so neither can overwrite the other. One wins the
        manifest CAS; the loser&apos;s segment is complete, correct, and dead on
        arrival.
      </p>
    </div>
  );
}

function Swatch({
  color,
  children,
}: {
  color: string;
  children: React.ReactNode;
}) {
  return (
    <span className="inline-flex items-center gap-1.5 text-ink">
      <span
        aria-hidden
        className="inline-block h-2 w-2 rounded-[1px]"
        style={{ background: color }}
      />
      {children}
    </span>
  );
}

// ---- by-kind summary (doubles as the legend) --------------------------------

function KindTable({
  byKind,
  pageBytes,
  showAll,
  selected,
  onSelect,
}: {
  byKind: ObjectKindSummaryJson[];
  pageBytes: string;
  showAll: boolean;
  selected: "ALL" | StorageObjectKind;
  onSelect: (k: "ALL" | StorageObjectKind) => void;
}) {
  const present = new Set(byKind.map((k) => k.kind));
  const missing = showAll
    ? STORAGE_OBJECT_KINDS.filter((k) => !present.has(k))
    : [];

  return (
    <TableShell
      head={
        <>
          <Th>kind</Th>
          <Th className="w-20 text-right">objects</Th>
          <Th className="w-24 text-right">bytes</Th>
          <Th className="w-24 text-right" title="bytes of this kind no longer referenced">
            dead
          </Th>
          <Th className="w-56">share of listed bytes</Th>
        </>
      }
    >
      {byKind.map((k) => {
        const copy = kindCopy(k.kind);
        const active = selected === k.kind;
        return (
          <tr
            key={k.kind}
            onClick={() => onSelect(active ? "ALL" : k.kind)}
            className={`cursor-pointer hover:bg-panel-2 ${active ? "bg-accent-dim/40" : ""}`}
            title={active ? "click to clear the filter" : "click to filter the table below"}
          >
            <Td>
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="font-mono text-[12px] text-ink">
                  {copy.label}
                </span>
                <span className="font-mono text-[10px] text-ink-faint">
                  {k.kind}
                </span>
              </div>
              <p className="mt-0.5 text-[11px] text-ink-dim leading-relaxed max-w-3xl">
                {copy.blurb}
              </p>
            </Td>
            <Td className="font-mono text-right tnum text-ink-dim">{k.count}</Td>
            <Td className="font-mono text-right tnum">{fmtBytes(k.bytes)}</Td>
            <Td className="font-mono text-right tnum text-ink-dim">
              {k.deadCount === 0 ? (
                <span className="text-ink-faint/60">—</span>
              ) : (
                <>
                  {fmtBytes(k.deadBytes)}
                  <span className="block text-[10px] text-ink-faint">
                    {k.deadCount} obj
                  </span>
                </>
              )}
            </Td>
            <Td>
              <KindMeter bytes={k.bytes} deadBytes={k.deadBytes} total={pageBytes} />
            </Td>
          </tr>
        );
      })}

      {missing.map((k) => {
        const copy = kindCopy(k);
        return (
          <tr key={k} className="opacity-55">
            <Td>
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="font-mono text-[12px] text-ink-dim">
                  {copy.label}
                </span>
                <span className="font-mono text-[10px] text-ink-faint">{k}</span>
              </div>
              <p className="mt-0.5 text-[11px] text-ink-faint leading-relaxed max-w-3xl">
                {copy.blurb}
              </p>
            </Td>
            <Td className="font-mono text-right text-ink-faint/60">—</Td>
            <Td className="font-mono text-right text-ink-faint/60">—</Td>
            <Td className="font-mono text-right text-ink-faint/60">—</Td>
            <Td className="font-mono text-[10px] text-ink-faint">
              none on this page
            </Td>
          </tr>
        );
      })}
    </TableShell>
  );
}

/**
 * One kind's share of the listed bytes, with the dead part of that share drawn
 * in the same bar. Magnitude, not severity — see LiveDeadSplit.
 */
function KindMeter({
  bytes,
  deadBytes,
  total,
}: {
  bytes: string;
  deadBytes: string;
  total: string;
}) {
  const share = ratio(bytes, total);
  const deadShare = ratio(deadBytes, total);
  const liveShare = Math.max(0, share - deadShare);
  return (
    <div className="flex items-center gap-2">
      {/*
        A neutral track, NOT the blue meter track used elsewhere: here the fill
        is itself blue (live) and the remainder means "some other kind", so a
        blue track would read as more live bytes than the row has.
      */}
      <div
        className="h-1.5 flex-1 rounded-sm overflow-hidden flex"
        style={{ background: "var(--color-line)" }}
      >
        <div
          className="h-full"
          style={{ width: `${liveShare * 100}%`, background: "var(--color-viz-1)" }}
        />
        <div
          className="h-full"
          style={{ width: `${deadShare * 100}%`, background: "var(--color-viz-2)" }}
        />
      </div>
      <span className="font-mono text-[10px] text-ink-faint tnum w-10 text-right">
        {sharePct(bytes, total)}
      </span>
    </div>
  );
}

function ratio(part: string, total: string): number {
  try {
    const t = BigInt(total);
    if (t <= 0n) return 0;
    return Number((BigInt(part) * 1000000n) / t) / 1000000;
  } catch {
    return 0;
  }
}

// ---- the object table -------------------------------------------------------

/** Objects whose key carries no segment (manifest, WAL) lead the listing; see the sort. */
function groupRank(o: ObjectInfoJson): number {
  return o.segment === "" ? 0 : 1;
}

function groupKey(o: ObjectInfoJson): string {
  return groupRank(o) === 0 ? "ns" : o.segment;
}

function startsGroup(rows: ObjectInfoJson[], i: number): boolean {
  return i === 0 || groupKey(rows[i]) !== groupKey(rows[i - 1]);
}

function groupLabel(o: ObjectInfoJson): string {
  return groupRank(o) === 0
    ? "namespace level — above any segment"
    : `segment seg-${o.segment}`;
}

function ObjectRow({
  object,
  groupHeader,
  selected,
  onSelect,
}: {
  object: ObjectInfoJson;
  groupHeader: string | null;
  selected: boolean;
  onSelect: () => void;
}) {
  const copy = kindCopy(object.kind);
  const dead = !object.live;
  return (
    <>
      {groupHeader && (
        <tr>
          <td colSpan={COLS} className="p-0">
            <div className="flex items-center gap-2 px-3 py-1 bg-panel-2 border-y border-line">
              <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-dim">
                {groupHeader}
              </span>
              <span className="h-px flex-1 bg-line" />
            </div>
          </td>
        </tr>
      )}
      <tr
        onClick={onSelect}
        className={`cursor-pointer hover:bg-panel-2 ${selected ? "bg-accent-dim/40" : ""}`}
      >
        <Td
          className={`font-mono text-[11px] break-all border-l-2 ${
            // The whole run of dead rows carries a left accent, so the split
            // reads as a region of the table rather than one badge per row.
            dead ? "border-l-viz-2/60" : "border-l-transparent"
          }`}
        >
          <PathCell path={object.path} dead={dead} />
        </Td>
        <Td title={copy.blurb}>
          <span className="font-mono text-[11px] text-ink-dim border-b border-dotted border-line-strong">
            {copy.label}
          </span>
        </Td>
        <Td className="font-mono text-right tnum">{fmtBytes(object.sizeBytes)}</Td>
        <Td className="font-mono text-left text-ink-dim">
          {object.segment === "" ? (
            <span
              className="text-ink-faint/60"
              title="this key carries no segment — the manifest and the WAL live above them"
            >
              —
            </span>
          ) : (
            <span title={`seg-${object.segment}`}>seg-{object.segment.slice(0, 8)}</span>
          )}
        </Td>
        <Td className="font-mono text-right tnum text-ink-dim">
          {object.memoryType === null ? (
            <span
              className="text-ink-faint/60"
              title="has_memory_type = false: only keys under mt{n}/ carry one"
            >
              —
            </span>
          ) : (
            object.memoryType
          )}
        </Td>
        <Td>
          <LiveFlag live={object.live} />
        </Td>
      </tr>
    </>
  );
}

/**
 * Not a status badge: `dead` is a normal, expected state between GC runs, so it
 * gets its own categorical hue rather than the danger colour.
 */
function LiveFlag({ live }: { live: boolean }) {
  if (live) {
    return (
      <span
        className="font-mono text-[10px] leading-4 px-1.5 py-px rounded-sm border border-viz-1/60 bg-viz-1/15 text-viz-1"
        title="referenced by the current manifest"
      >
        live
      </span>
    );
  }
  return (
    <span
      className="font-mono text-[10px] leading-4 px-1.5 py-px rounded-sm border border-viz-2/60 bg-viz-2/15 text-viz-2"
      title="not referenced by the current manifest — garbage awaiting GC, which is normal"
    >
      dead
    </span>
  );
}

/** Dim the prefix so the file name — the useful bit — leads. */
function PathCell({ path, dead }: { path: string; dead: boolean }) {
  const cut = path.lastIndexOf("/");
  const name = cut < 0 ? path : path.slice(cut + 1);
  const prefix = cut < 0 ? "" : path.slice(0, cut + 1);
  return (
    <>
      {prefix && <span className="text-ink-faint">{prefix}</span>}
      <span className={dead ? "text-ink-dim" : "text-ink"}>{name}</span>
    </>
  );
}

// ---- drill-in ---------------------------------------------------------------

const DEFAULT_DECODE_LIMIT = 50;

/**
 * DecodeObject for one row. The JSON it returns is a debugging view whose shape
 * follows the on-disk formats — explicitly not a contract — so it is rendered
 * and never interpreted.
 */
function DecodeDrawer({
  namespace,
  object,
  onClose,
}: {
  namespace: string;
  object: ObjectInfoJson;
  onClose: () => void;
}) {
  const [limit, setLimit] = useState(DEFAULT_DECODE_LIMIT);
  const [data, setData] = useState<DecodeObjectJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const copy = kindCopy(object.kind);

  const run = useCallback(
    async (lim: number) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setLoading(true);
      setError(null);
      const body: DecodeObjectRequestBody = { path: object.path, limit: lim };
      try {
        const res = await postJson<DecodeObjectJson>(
          `/api/namespaces/${encodeURIComponent(namespace)}/objects/decode`,
          body,
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
    [namespace, object.path],
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void run(DEFAULT_DECODE_LIMIT);
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [object.path]);

  return (
    <aside className="w-[36rem] shrink-0 border-l border-line bg-panel overflow-y-auto max-h-[calc(100vh-2.25rem)] sticky top-9">
      <header className="flex items-center gap-2 px-3 py-2 border-b border-line sticky top-0 bg-panel z-10">
        <h3 className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim flex-1">
          decode object
        </h3>
        <Field label="items" className="w-20">
          <NumberInput
            min={0}
            max={5000}
            value={limit}
            disabled={loading}
            onChange={(e) => setLimit(Number(e.target.value) || 0)}
          />
        </Field>
        <Button className="mt-4" disabled={loading} onClick={() => void run(limit)}>
          {loading ? "…" : "decode"}
        </Button>
        <Button variant="ghost" className="mt-4" onClick={onClose} title="close">
          ✕
        </Button>
      </header>

      <div className="p-3 flex flex-col gap-3">
        <div>
          <p className="font-mono text-[11px] break-all">
            <PathCell path={object.path} dead={!object.live} />
          </p>
          <div className="mt-1.5 flex items-center gap-2 flex-wrap">
            <span className="font-mono text-[11px] text-ink">{copy.label}</span>
            <span className="font-mono text-[10px] text-ink-faint">
              {object.kind}
            </span>
            <LiveFlag live={object.live} />
            <span className="font-mono text-[11px] text-ink-dim tnum">
              {fmtBytes(object.sizeBytes)}
            </span>
          </div>
        </div>

        {/* The explanation is the point of the panel, not a footnote to it. */}
        <p className="border border-line bg-panel-2 rounded-sm px-2.5 py-2 text-[11px] text-ink leading-relaxed">
          {copy.blurb}
        </p>

        {!object.live && (
          <p className="text-[11px] text-ink-dim leading-relaxed">
            This object is <span className="text-viz-2 font-mono">dead</span> — no
            query will ever read it again. It is still perfectly decodable:
            immutability means it holds exactly what it held when it was
            published, it is simply not what the current manifest points at.
          </p>
        )}

        {loading && !data && <Loading label="DecodeObject" />}

        {Boolean(error) && (
          <ErrorBanner
            error={error}
            what={`DecodeObject(${object.path})`}
            onRetry={() => void run(limit)}
          />
        )}

        {data && (
          <>
            <div className="grid grid-cols-2 gap-2">
              <StatTile label="size" value={fmtBytes(data.sizeBytes)} />
              <StatTile
                label="items"
                value={groupDigits(data.totalItems)}
                hint={
                  data.totalItems === "0"
                    ? "not a container — the whole object is shown"
                    : data.truncated
                      ? `showing ${data.limit || "the server default"}`
                      : "all shown"
                }
              />
            </div>

            {data.truncated && (
              <p className="border border-warn/40 bg-warn/5 rounded-sm px-2.5 py-1.5 text-[11px] text-warn leading-relaxed">
                showing{" "}
                {data.limit > 0
                  ? groupDigits(String(data.limit))
                  : "the server's default number"}{" "}
                of {groupDigits(data.totalItems)} items — raise the limit to see
                more. What is below is a prefix of the object, not all of it.
              </p>
            )}

            {data.undecodableReason && (
              <div className="border border-dashed border-line rounded-sm px-3 py-3">
                <p className="font-mono text-[11px] text-ink-dim">
                  memlake does not parse this format
                </p>
                <p className="mt-1.5 font-mono text-[11px] text-ink break-words">
                  {data.undecodableReason}
                </p>
                <p className="mt-2 text-[11px] text-ink-faint leading-relaxed">
                  Expected, not a failure: the object below is shown as whatever
                  metadata the server could report. Reading its contents means
                  going through the engine that owns the format.
                </p>
              </div>
            )}

            {data.json.trim() ? (
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint">
                    decoded
                  </span>
                  <span className="font-mono text-[10px] text-ink-faint">
                    {fmtMs(data.elapsedMs)}
                  </span>
                </div>
                <pre className="border border-line bg-panel-2 rounded-sm p-2.5 overflow-auto max-h-[46rem] font-mono text-[11px] leading-relaxed text-ink whitespace-pre tnum">
                  {data.json}
                </pre>
                <p className="mt-1 text-[10px] text-ink-faint leading-relaxed">
                  A debugging view of the on-disk format
                  {data.jsonPretty ? "" : " (shown verbatim — it did not parse as JSON)"}
                  . Its shape follows the storage layout and is deliberately not
                  a stable contract; read memories with Get / Scan / Query.
                </p>
              </div>
            ) : (
              !data.undecodableReason && (
                <Empty title="the server returned no decoded body">
                  DecodeObject answered without an error and without JSON — there
                  is nothing to show for this object.
                </Empty>
              )
            )}
          </>
        )}
      </div>
    </aside>
  );
}
