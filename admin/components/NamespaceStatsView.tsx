"use client";

import { useCallback, useEffect, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import { groupDigits } from "@/lib/format";
import {
  Button,
  CopyableId,
  Empty,
  ErrorBanner,
  KeyValue,
  Loading,
  Panel,
  StatTile,
  TableShell,
  Td,
  Th,
} from "@/components/ui";

interface ManifestSegment {
  id: string;
  level: number;
  seq_lo: number;
  seq_hi: number;
  doc_count: number;
  indexes: Record<string, { train_count?: number; files?: { clusters?: string[] } }>;
}
interface Manifest {
  format_version: number;
  /** Current manifests use `version`; legacy pre-segmentation ones use `generation`. */
  version?: number;
  generation?: number;
  wal_index_cursor: number;
  wal_head: number;
  prev_wal_index_cursor?: number;
  tokenizer_config_hash: string;
  indexed_metadata_keys?: string[];
  /** Absent on legacy manifests, which predate segmentation. */
  segments?: ManifestSegment[];
}
interface NamespaceState {
  namespace: string;
  manifest: Manifest | null;
  walHead: number | null;
  elapsedMs: number;
}

export function NamespaceStatsView({ namespace }: { namespace: string }) {
  const [data, setData] = useState<NamespaceState | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getJson<NamespaceState>(
        `/api/namespaces/${encodeURIComponent(namespace)}/stats`,
      );
      setData(res);
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

  if (loading && !data) return <div className="p-4"><Loading label="reading manifest" /></div>;
  if (error)
    return (
      <div className="p-4">
        <ErrorBanner error={error} what="namespace state" onRetry={() => void load()} />
      </div>
    );
  if (!data) return null;

  const m = data.manifest;
  if (!m) {
    return (
      <div className="p-4">
        <Empty title="no manifest">
          <p>
            <code>{namespace}/manifest.json</code> is missing — the namespace exists as a prefix but
            has never been folded.
          </p>
        </Empty>
      </div>
    );
  }

  // Real buckets hold both manifest shapes; tolerate either rather than assuming the newest writer.
  const segments = m.segments ?? [];
  const legacy = m.segments === undefined;
  const generation = m.version ?? m.generation ?? 0;
  const indexedDocs = segments.reduce((s, x) => s + (x.doc_count ?? 0), 0);
  const head = data.walHead ?? m.wal_head;
  const backlog = Math.max(head - m.wal_index_cursor, 0);
  // Fact types are the union of every segment's index keys.
  const types = Array.from(
    new Set(segments.flatMap((s) => Object.keys(s.indexes ?? {}))),
  ).sort();

  return (
    <div className="p-4 max-w-5xl mx-auto flex flex-col gap-4">
      <Panel
        title="index state"
        subtitle="Read from manifest.json plus the live wal-head pointer — authoritative on-disk state, no serve node involved."
        actions={
          <Button onClick={() => void load()} disabled={loading}>
            refresh
          </Button>
        }
        bodyClassName="p-3 flex flex-col gap-4"
      >
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <StatTile label="generation" value={String(generation)} />
          <StatTile
            label="indexed docs"
            value={groupDigits(String(indexedDocs))}
            hint={legacy ? "legacy manifest" : "across segments"}
          />
          <StatTile label="wal head" value={groupDigits(String(head))} hint="live pointer" />
          <StatTile
            label="un-indexed backlog"
            value={groupDigits(String(backlog))}
            tone={backlog > 0 ? "warn" : "ok"}
            hint="WAL entries every query re-scans"
          />
        </div>
        <div className="text-[11px] text-ink-faint leading-snug">
          The <em>live</em> document count is deliberately not shown: it requires replaying the
          un-indexed WAL tail over the segments, which is the engine&apos;s job. What is shown is the
          indexed count and the backlog — together they say the same thing without guessing.
        </div>
        <KeyValue
          entries={[
            { k: "format version", v: String(m.format_version) },
            { k: "wal index cursor", v: groupDigits(String(m.wal_index_cursor)) },
            { k: "prev cursor", v: groupDigits(String(m.prev_wal_index_cursor ?? 0)) },
            { k: "tokenizer hash", v: <CopyableId value={m.tokenizer_config_hash} /> },
            { k: "fact types", v: types.length ? types.join(", ") : "—" },
            {
              k: "indexed metadata",
              v: m.indexed_metadata_keys?.length ? m.indexed_metadata_keys.join(", ") : "—",
            },
          ]}
        />
      </Panel>

      <Panel
        title={`${segments.length} segments`}
        subtitle="The LSM segments this generation points at. Level 0 is a flush; higher levels are compactions."
        bodyClassName="p-0"
      >
        {segments.length === 0 ? (
          <div className="p-3">
            <Empty title={legacy ? "legacy manifest — pre-segmentation" : "no segments — nothing folded yet"}>
              <p>
                {legacy
                  ? "This manifest predates the segmented index (it carries a flat `indexes` map). It will gain segments the next time the namespace is folded."
                  : "Nothing has been folded into a segment yet; everything live is still in the WAL tail."}
              </p>
            </Empty>
          </div>
        ) : (
          <TableShell
            head={
              <>
                <Th>segment</Th>
                <Th className="text-right">level</Th>
                <Th className="text-right">seq range</Th>
                <Th className="text-right">docs</Th>
                <Th className="text-right">clusters</Th>
              </>
            }
          >
            {segments.map((s) => {
              const clusters = Object.values(s.indexes ?? {}).reduce(
                (n, ft) => n + (ft.files?.clusters?.length ?? 0),
                0,
              );
              return (
                <tr key={s.id} className="hover:bg-panel-2">
                  <Td>
                    <CopyableId value={s.id} display={s.id.slice(0, 14)} />
                  </Td>
                  <Td className="text-right tnum">{s.level}</Td>
                  <Td className="text-right tnum text-ink-dim">
                    {groupDigits(String(s.seq_lo))}–{groupDigits(String(s.seq_hi))}
                  </Td>
                  <Td className="text-right tnum">{groupDigits(String(s.doc_count))}</Td>
                  <Td className="text-right tnum text-ink-dim">{groupDigits(String(clusters))}</Td>
                </tr>
              );
            })}
          </TableShell>
        )}
      </Panel>
    </div>
  );
}

