"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { isAbort, postJson } from "@/lib/client";
import { cmpU64, fmtEpochGuess, fmtMs, groupDigits, truncate } from "@/lib/format";
import { shortId } from "@/lib/ids";
import type {
  ListWalJson,
  ListWalRequestBody,
  WalEntryJson,
  WalOpJson,
} from "@/lib/types";
import {
  Button,
  CopyableId,
  Empty,
  ErrorBanner,
  Field,
  Loading,
  NumberInput,
  Panel,
  StatTile,
  Tag,
  Td,
  TableShell,
  Th,
  Toggle,
} from "@/components/ui";

const COLS = 8;

export function WalView({ namespace }: { namespace: string }) {
  const [limit, setLimit] = useState(50);
  const [includeOps, setIncludeOps] = useState(false);

  /**
   * `start_seq` is a resume point, not a page number: the log is a window that
   * GC trims from the front. Same cursor-stack shape as Scan — the last element
   * is the start_seq that produced what is on screen ("0" = oldest retained).
   */
  const [stack, setStack] = useState<string[]>(["0"]);
  const [data, setData] = useState<ListWalJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (tokens: string[], withOps: boolean) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setLoading(true);
      setError(null);

      const body: ListWalRequestBody = {
        startSeq: tokens[tokens.length - 1] ?? "0",
        limit,
        includeOps: withOps,
      };
      try {
        const res = await postJson<ListWalJson>(
          `/api/namespaces/${encodeURIComponent(namespace)}/wal`,
          body,
          ac.signal,
        );
        setData(res);
        setStack(tokens);
        setExpanded(new Set());
      } catch (e) {
        if (isAbort(e)) return;
        setError(e);
        setData(null);
      } finally {
        if (abortRef.current === ac) setLoading(false);
      }
    },
    [namespace, limit],
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void run(["0"], false);
    return () => abortRef.current?.abort();
    // Keyed on namespace only: limit/include_ops changes are explicit re-reads.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namespace]);

  const canBack = stack.length > 1;
  const canNext = Boolean(data && data.nextSeq !== "0");

  const backlog = data ? BigInt(data.backlog) : 0n;
  const cursor = data ? BigInt(data.walIndexCursor) : 0n;

  // Where does the folded/un-folded boundary fall on THIS page? Entries come
  // back ascending, and folded <=> seq <= wal_index_cursor, so it is a single
  // index — or off-page entirely, which is worth saying out loud.
  const entries = data?.entries ?? [];
  const firstUnfolded = entries.findIndex((e) => !e.folded);
  const allFolded = entries.length > 0 && firstUnfolded === -1;
  const allUnfolded = entries.length > 0 && firstUnfolded === 0;

  /**
   * The backlog tile is derived from `wal_head`, but the entry list is
   * authoritative about what is actually un-folded. When the two disagree —
   * an entry exists whose seq is past the reported head — say so instead of
   * rendering "backlog 0" directly above four un-folded rows.
   */
  const maxSeqSeen = entries.reduce(
    (m, e) => (cmpU64(e.seq, m) > 0 ? e.seq : m),
    "0",
  );
  const unfoldedOnPage = entries.filter((e) => !e.folded).length;
  const headLags = data ? cmpU64(maxSeqSeen, data.walHead) > 0 : false;

  /**
   * The proto bills WalOpCounts as "cheap to compute and usually all an operator
   * needs", i.e. available in the header view — but this server build only fills
   * it in when include_ops was set. Four zero columns would read as "this entry
   * has no ops", which is a different and wrong statement, so render them as
   * not-measured instead.
   */
  const countsMissing =
    !includeOps &&
    entries.length > 0 &&
    entries.every(
      (e) =>
        e.counts.upserts === 0 &&
        e.counts.tombstones === 0 &&
        e.counts.patches === 0 &&
        e.counts.guards === 0,
    );

  function toggleRow(seq: string) {
    if (!includeOps) return;
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });
  }

  return (
    <div className="p-4 flex flex-col gap-4 max-w-7xl">
      <Panel
        title="write-ahead log"
        subtitle={
          <>
            A window on the <em>live</em> log, not a history of every write:
            entries at or below <code>wal_index_cursor</code> are folded into a
            generation and may already have been reclaimed by GC.
          </>
        }
        actions={
          <>
            <Field label="limit" className="w-20">
              <NumberInput
                min={0}
                max={500}
                value={limit}
                disabled={loading}
                onChange={(e) => setLimit(Number(e.target.value) || 0)}
              />
            </Field>
            <Button
              className="mt-4"
              disabled={loading}
              onClick={() => void run(["0"], includeOps)}
            >
              {loading ? "…" : "reload"}
            </Button>
          </>
        }
      >
        {loading && !data && <Loading label="ListWal" />}

        {Boolean(error) && (
          <ErrorBanner
            error={error}
            what={`ListWal(${namespace})`}
            onRetry={() => void run(stack, includeOps)}
          />
        )}

        {data && (
          <>
            <div className="grid gap-2 grid-cols-2 md:grid-cols-4">
              <StatTile
                label="un-indexed backlog"
                value={groupDigits(data.backlog)}
                tone={backlog === 0n ? "ok" : backlog > 1000n ? "danger" : "warn"}
                hint="entries every STRONG query scans on every read"
              />
              <StatTile
                label="wal_head"
                value={groupDigits(data.walHead)}
                hint="last committed sequence"
              />
              <StatTile
                label="wal_index_cursor"
                value={groupDigits(data.walIndexCursor)}
                hint="last sequence folded into the generation"
              />
              <StatTile
                label="rpc"
                value={fmtMs(data.elapsedMs)}
                hint={`${entries.length} entr${entries.length === 1 ? "y" : "ies"} on this page`}
              />
            </div>

            {headLags && (
              <p className="mt-2 border border-warn/40 bg-warn/5 rounded-sm px-2.5 py-1.5 text-[11px] text-warn leading-relaxed">
                <span className="font-mono">wal_head = {data.walHead}</span>, but
                this page enumerated entries up to{" "}
                <span className="font-mono">seq {maxSeqSeen}</span>, of which{" "}
                <span className="font-mono">{unfoldedOnPage}</span>{" "}
                {unfoldedOnPage === 1 ? "is" : "are"} un-folded. The backlog tile
                is derived from wal_head and therefore under-reports here; the
                per-entry folded flags below are the ones to trust.
              </p>
            )}

            <div className="mt-3 flex items-center gap-4 flex-wrap">
              <Toggle
                checked={includeOps}
                onChange={(v) => {
                  setIncludeOps(v);
                  void run(stack, v);
                }}
                disabled={loading}
                label="include_ops"
              />
              <span className="text-[10px] text-ink-faint max-w-xl">
                Off by default: one entry is a whole group-commit batch and can
                hold thousands of memories, so the listing stays a header view
                unless asked. With it on, click a row to expand its ops.
                {countsMissing && (
                  <span className="text-warn">
                    {" "}
                    This build only fills in the op counts when include_ops is
                    set, so those columns read “—” rather than a misleading 0.
                  </span>
                )}
              </span>
            </div>
          </>
        )}
      </Panel>

      <Panel
        title="entries"
        subtitle={
          data ? (
            <>
              page {stack.length} · start_seq{" "}
              <span className="font-mono">{stack[stack.length - 1]}</span>
              {data.nextSeq === "0" ? " · end of log" : ""}
            </>
          ) : undefined
        }
        actions={
          <>
            <Button
              disabled={!canBack || loading}
              onClick={() => void run(stack.slice(0, -1), includeOps)}
            >
              ← back
            </Button>
            <Button
              disabled={!canNext || loading}
              onClick={() => void run([...stack, data?.nextSeq ?? "0"], includeOps)}
            >
              next →
            </Button>
          </>
        }
        bodyClassName="p-0"
      >
        {data && entries.length === 0 && (
          <div className="p-3">
            <Empty title="no WAL entries in this window">
              Either nothing has been written to this namespace, or every entry
              has been folded and reclaimed by GC.
            </Empty>
          </div>
        )}

        {data && entries.length > 0 && (
          <>
            {allFolded && (
              <BoundaryNote tone="folded">
                every entry on this page is folded (seq ≤ wal_index_cursor{" "}
                {groupDigits(data.walIndexCursor)}) — a STRONG query scans none
                of them
              </BoundaryNote>
            )}
            {allUnfolded && (
              <BoundaryNote tone="unfolded">
                every entry on this page is UN-folded (seq &gt; wal_index_cursor{" "}
                {groupDigits(data.walIndexCursor)}) — a STRONG query scans all of
                them on every read
              </BoundaryNote>
            )}

            <TableShell
              head={
                <>
                  <Th className="w-28 text-right">seq</Th>
                  <Th className="w-24 text-right">size</Th>
                  <Th className="w-20 text-right" title="Memory upserts">
                    upserts
                  </Th>
                  <Th className="w-24 text-right" title="Tombstones (deletes)">
                    tombstones
                  </Th>
                  <Th className="w-20 text-right" title="Field-level patches">
                    patches
                  </Th>
                  <Th
                    className="w-20 text-right"
                    title="Optimistic preconditions (compare-and-set)"
                  >
                    guards
                  </Th>
                  <Th className="w-32">state</Th>
                  <Th />
                </>
              }
            >
              {entries.map((e, i) => (
                <WalRow
                  key={e.seq}
                  entry={e}
                  showDivider={i === firstUnfolded && i > 0}
                  cursor={cursor}
                  includeOps={includeOps}
                  countsMissing={countsMissing}
                  expanded={expanded.has(e.seq)}
                  onToggle={() => toggleRow(e.seq)}
                />
              ))}
            </TableShell>
          </>
        )}

        {loading && !data && (
          <div className="p-3">
            <Loading label="ListWal" />
          </div>
        )}
      </Panel>
    </div>
  );
}

/**
 * The folded/un-folded boundary. This is the single most important thing on the
 * page, so it is a full-width rule with a label — not a badge you have to hunt
 * for column by column.
 */
function BoundaryNote({
  tone,
  children,
}: {
  tone: "folded" | "unfolded";
  children: React.ReactNode;
}) {
  const unfolded = tone === "unfolded";
  return (
    <p
      className={`px-3 py-1.5 font-mono text-[11px] border-b ${
        unfolded
          ? "text-warn bg-warn/5 border-warn/25"
          : "text-ink-faint bg-panel-2 border-line"
      }`}
    >
      {children}
    </p>
  );
}

function WalRow({
  entry,
  showDivider,
  cursor,
  includeOps,
  countsMissing,
  expanded,
  onToggle,
}: {
  entry: WalEntryJson;
  showDivider: boolean;
  cursor: bigint;
  includeOps: boolean;
  countsMissing: boolean;
  expanded: boolean;
  onToggle: () => void;
}) {
  const unfolded = !entry.folded;
  // Un-folded rows carry a left accent and a tint for the whole run, so the
  // boundary reads as a region rather than a per-row property.
  const rowTint = unfolded ? "bg-warn/5" : "";

  return (
    <>
      {showDivider && (
        <tr>
          <td colSpan={COLS} className="p-0">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-warn/10 border-y border-warn/40">
              <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-warn">
                wal_index_cursor = {cursor.toString()}
              </span>
              <span className="h-px flex-1 bg-warn/40" />
              <span className="font-mono text-[10px] text-warn">
                below: un-folded — scanned by every STRONG query
              </span>
            </div>
          </td>
        </tr>
      )}
      <tr
        onClick={onToggle}
        className={`${rowTint} ${includeOps ? "cursor-pointer" : ""} hover:bg-panel-2`}
      >
        <Td
          className={`font-mono text-right tnum ${
            unfolded ? "border-l-2 border-l-warn" : "border-l-2 border-l-transparent"
          }`}
        >
          {entry.seq}
        </Td>
        <Td className="font-mono text-right tnum text-ink-dim">
          {fmtBytes(entry.sizeBytes)}
        </Td>
        <OpCount n={entry.counts.upserts} missing={countsMissing} />
        <OpCount n={entry.counts.tombstones} missing={countsMissing} />
        <OpCount n={entry.counts.patches} missing={countsMissing} />
        <OpCount n={entry.counts.guards} missing={countsMissing} />
        <Td>
          {unfolded ? (
            <span className="font-mono text-[11px] text-warn">un-folded</span>
          ) : (
            <span className="font-mono text-[11px] text-ok">folded</span>
          )}
        </Td>
        <Td className="text-right">
          {includeOps && (
            <span className="font-mono text-[11px] text-ink-faint">
              {expanded ? "▾" : "▸"} {entry.ops.length} op
              {entry.ops.length === 1 ? "" : "s"}
            </span>
          )}
        </Td>
      </tr>
      {includeOps && expanded && (
        <tr className={rowTint}>
          <Td colSpan={COLS} className="px-3 py-2 border-l-2 border-l-warn/0">
            <WalOps ops={entry.ops} />
          </Td>
        </tr>
      )}
    </>
  );
}

function OpCount({ n, missing }: { n: number; missing: boolean }) {
  if (missing) {
    return (
      <Td
        className="font-mono text-right text-ink-faint/60"
        title="this server build only fills in WalOpCounts when include_ops is set — this is 'not measured', not 'zero'"
      >
        —
      </Td>
    );
  }
  return (
    <Td
      className={`font-mono text-right tnum ${n === 0 ? "text-ink-faint/50" : "text-ink"}`}
    >
      {n}
    </Td>
  );
}

function WalOps({ ops }: { ops: WalOpJson[] }) {
  if (ops.length === 0) {
    return (
      <span className="font-mono text-[11px] text-ink-faint">
        no ops decoded for this entry
      </span>
    );
  }
  return (
    <ol className="flex flex-col gap-1">
      {ops.map((op, i) => (
        <li key={i} className="flex items-start gap-2">
          <span className="font-mono text-[10px] text-ink-faint tnum w-8 shrink-0 text-right pt-px">
            {i}
          </span>
          <WalOpDetail op={op} />
        </li>
      ))}
    </ol>
  );
}

const KIND_STYLE: Record<string, string> = {
  upsert: "border-arm-dense/50 text-arm-dense",
  tombstone: "border-danger/50 text-danger",
  patch: "border-arm-temporal/50 text-arm-temporal",
  guard: "border-arm-graph/50 text-arm-graph",
  unknown: "border-line-strong text-ink-faint",
};

function KindBadge({ kind }: { kind: string }) {
  return (
    <span
      className={`shrink-0 font-mono text-[10px] uppercase tracking-[0.08em] px-1.5 py-px rounded-sm border ${
        KIND_STYLE[kind] ?? KIND_STYLE.unknown
      }`}
    >
      {kind}
    </span>
  );
}

function WalOpDetail({ op }: { op: WalOpJson }) {
  if (op.kind === "upsert") {
    return (
      <div className="flex items-start gap-2 flex-wrap min-w-0">
        <KindBadge kind="upsert" />
        <CopyableId value={op.id} display={shortId(op.id)} />
        <span className="font-mono text-[11px] text-ink-dim">
          type {op.memoryType}
        </span>
        <span className="font-mono text-[11px]">
          {op.vectorDim === 0 ? (
            // 0 means the memory carries no embedding at all — saying "0" here
            // would read as a zero-length vector, which is a different thing.
            <span className="text-ink-faint">no embedding</span>
          ) : (
            <span className="text-arm-dense">f32[{op.vectorDim}]</span>
          )}
        </span>
        {op.tags.length > 0 && (
          <span className="flex gap-1 flex-wrap">
            {op.tags.map((t) => (
              <Tag key={t}>{t}</Tag>
            ))}
          </span>
        )}
        {op.text && (
          <span className="text-[11px] text-ink min-w-0 break-words">
            {truncate(op.text.replace(/\s+/g, " "), 120)}
          </span>
        )}
      </div>
    );
  }

  if (op.kind === "tombstone") {
    return (
      <div className="flex items-center gap-2 flex-wrap">
        <KindBadge kind="tombstone" />
        <CopyableId value={op.id} display={shortId(op.id)} />
        <span className="text-[11px] text-ink-faint">delete</span>
      </div>
    );
  }

  if (op.kind === "patch") {
    const sets: string[] = [];
    if (op.setsText) sets.push("text");
    if (op.setsVector) sets.push(`vector f32[${op.vectorDim}]`);
    if (op.setsTags) sets.push(`tags[${op.tags.length}]`);
    if (op.setsTimestamps) sets.push("timestamps");
    const metaKeys = Object.keys(op.metadata);
    return (
      <div className="flex items-start gap-2 flex-wrap min-w-0">
        <KindBadge kind="patch" />
        <CopyableId value={op.id} display={shortId(op.id)} />
        {op.proofCountDelta !== 0 && (
          <span className="font-mono text-[11px] text-ink">
            proof_count {op.proofCountDelta > 0 ? "+" : ""}
            {op.proofCountDelta}
          </span>
        )}
        {sets.length > 0 && (
          <span className="font-mono text-[11px] text-ink-dim">
            sets {sets.join(", ")}
          </span>
        )}
        {metaKeys.length > 0 && (
          <span className="font-mono text-[11px] text-ink-dim">
            merges metadata {metaKeys.join(", ")}
          </span>
        )}
        {op.setsText && op.text !== null && (
          <span className="text-[11px] text-ink min-w-0 break-words">
            {truncate(op.text.replace(/\s+/g, " "), 100)}
          </span>
        )}
        {op.setsTimestamps && op.timestamps && (
          <span className="font-mono text-[10px] text-ink-faint">
            {[
              ["event", op.timestamps.eventDate],
              ["start", op.timestamps.occurredStart],
              ["end", op.timestamps.occurredEnd],
              ["mentioned", op.timestamps.mentionedAt],
            ]
              .filter(([, v]) => v !== null)
              .map(([k, v]) => `${k}=${fmtEpochGuess(String(v))}`)
              .join(" · ")}
          </span>
        )}
        {sets.length === 0 && metaKeys.length === 0 && op.proofCountDelta === 0 && (
          // Either the patch really changed nothing, or the server returned the
          // id without the field detail. Say what we know, not what we guess.
          <span
            className="text-[11px] text-ink-faint"
            title="ListWal returned this patch's id with no populated fields — either a genuine no-op or the server did not decode them"
          >
            no field detail returned
          </span>
        )}
      </div>
    );
  }

  if (op.kind === "guard") {
    return (
      <div className="flex items-center gap-2 flex-wrap">
        <KindBadge kind="guard" />
        <span className="font-mono text-[11px] text-ink">
          expect_seq_lt {op.expectSeqLt}
        </span>
        <span className="text-[11px] text-ink-faint">
          optimistic precondition: the batch applies only if the WAL head is
          below this sequence
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <KindBadge kind="unknown" />
      <span className="text-[11px] text-ink-faint">
        the server sent an op kind this build does not know about — the proto has
        moved ahead of the UI
      </span>
    </div>
  );
}

/** WAL object sizes are u64 decimal strings; format without going through Number. */
function fmtBytes(s: string): string {
  let n: bigint;
  try {
    n = BigInt(s);
  } catch {
    return s;
  }
  if (n < 1024n) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let v = Number(n);
  let u = -1;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[u]}`;
}
