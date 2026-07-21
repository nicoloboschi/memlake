"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { isAbort, postJson } from "@/lib/client";
import { cmpU64, fmtMs, groupDigits, sharePct } from "@/lib/format";
import {
  CONSISTENCIES,
  type Consistency,
  type IndexLayoutJson,
  type IndexLayoutRequestBody,
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
  StatTile,
  Tag,
  Td,
  TableShell,
  Th,
} from "@/components/ui";
import { ClusterScatter } from "@/components/ClusterScatter";
import { useKnownTypes } from "@/components/filters";

const CONSISTENCY_OPTIONS = CONSISTENCIES.map((c) => ({ value: c, label: c }));

type SortKey = "clusterId" | "size" | "tags" | "hasUntagged";
type SortDir = "asc" | "desc";

export function ClustersView({ namespace }: { namespace: string }) {
  const knownTypes = useKnownTypes(namespace);

  const [memoryType, setMemoryType] = useState(0);
  const [memberSample, setMemberSample] = useState(0);
  const [consistency, setConsistency] = useState<Consistency>("EVENTUAL");

  const [data, setData] = useState<IndexLayoutJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);

  // Sorting by size is the useful default: imbalance is immediately visible,
  // and imbalance is what makes probe cost uneven.
  const [sortKey, setSortKey] = useState<SortKey>("size");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (type: number, sample: number, cons: Consistency) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setLoading(true);
      setError(null);

      const body: IndexLayoutRequestBody = {
        memoryType: type,
        memberSample: sample,
        consistency: cons,
      };
      try {
        const res = await postJson<IndexLayoutJson>(
          `/api/namespaces/${encodeURIComponent(namespace)}/layout`,
          body,
          ac.signal,
        );
        setData(res);
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
    void run(memoryType, memberSample, consistency);
    return () => abortRef.current?.abort();
    // Re-reads are explicit; a member sample is a real object-storage read.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namespace]);

  // Default to the first type that actually exists, once Stats tells us.
  useEffect(() => {
    if (knownTypes && knownTypes.length > 0 && !knownTypes.includes(memoryType)) {
      const first = knownTypes[0];
      // Stats told us which types actually exist; adopt the first real one
      // rather than leaving the page asking for a type that isn't there.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setMemoryType(first);
      void run(first, memberSample, consistency);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [knownTypes]);

  const rows = useMemo(() => {
    if (!data) return [];
    const sorted = [...data.clusters];
    const dir = sortDir === "asc" ? 1 : -1;
    sorted.sort((a, b) => {
      switch (sortKey) {
        case "size":
          return dir * cmpU64(a.size, b.size) || a.clusterId - b.clusterId;
        case "tags":
          return dir * (a.tags.length - b.tags.length) || a.clusterId - b.clusterId;
        case "hasUntagged":
          return (
            dir * (Number(a.hasUntagged) - Number(b.hasUntagged)) ||
            a.clusterId - b.clusterId
          );
        default:
          return dir * (a.clusterId - b.clusterId);
      }
    });
    return sorted;
  }, [data, sortKey, sortDir]);

  function sortBy(k: SortKey) {
    if (k === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir(k === "clusterId" ? "asc" : "desc");
    }
  }

  const imbalance = useMemo(() => {
    if (!data || data.clusters.length === 0) return null;
    const sizes = data.clusters.map((c) => Number(c.size) || 0);
    const max = Math.max(...sizes);
    const mean = sizes.reduce((s, x) => s + x, 0) / sizes.length;
    return mean > 0 ? max / mean : null;
  }, [data]);

  return (
    <div className="p-4 flex flex-col gap-4 max-w-7xl">
      {/* One filter row, above everything it scopes. */}
      <Panel
        title="ivf layout"
        subtitle="How k-means partitioned ONE memory_type. Types are independent indexes, so this is one type at a time — there is no combined view to show."
        actions={
          <SegmentedControl
            value={consistency}
            onChange={(c) => {
              setConsistency(c);
              void run(memoryType, memberSample, c);
            }}
            options={CONSISTENCY_OPTIONS}
            disabled={loading}
          />
        }
      >
        <div className="flex items-end gap-4 flex-wrap">
          <Field label="memory_type" className="w-32">
            <NumberInput
              min={0}
              max={255}
              value={memoryType}
              disabled={loading}
              onChange={(e) => setMemoryType(Number(e.target.value) || 0)}
            />
          </Field>

          {knownTypes && knownTypes.length > 0 && (
            <div className="flex flex-wrap gap-1 pb-1.5">
              {knownTypes.map((t) => (
                <button
                  key={t}
                  type="button"
                  disabled={loading}
                  onClick={() => {
                    setMemoryType(t);
                    void run(t, memberSample, consistency);
                  }}
                  className={`font-mono text-[10px] leading-4 px-1.5 py-px rounded-sm border ${
                    memoryType === t
                      ? "border-accent/60 bg-accent-dim text-accent"
                      : "border-line-strong bg-panel-2 text-ink-dim hover:text-ink"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          )}

          <Field
            label="member_sample"
            className="w-32"
            hint="0 = centroids only"
          >
            <NumberInput
              min={0}
              max={1000}
              step={50}
              value={memberSample}
              disabled={loading}
              onChange={(e) => setMemberSample(Number(e.target.value) || 0)}
            />
          </Field>

          <Button
            variant="primary"
            className="mb-1"
            disabled={loading}
            onClick={() => void run(memoryType, memberSample, consistency)}
          >
            {loading ? "loading…" : "IndexLayout"}
          </Button>

          <p className="pb-1 text-[10px] text-ink-faint max-w-md leading-relaxed">
            {memberSample === 0 ? (
              <>
                <span className="text-ok">Centroids only</span> — this costs the
                server <strong>no object-storage read at all</strong>: the
                centroids are already resident on every query node. Sampling
                members reads cluster files.
              </>
            ) : (
              <>
                Sampling <span className="text-ink-dim">{memberSample}</span>{" "}
                members reads cluster files, unlike the centroid-only call. Their
                full embeddings ship to the browser for the projection.
              </>
            )}
          </p>
        </div>
      </Panel>

      {Boolean(error) && (
        <ErrorBanner
          error={error}
          what={`IndexLayout(${namespace}, type ${memoryType})`}
          onRetry={() => void run(memoryType, memberSample, consistency)}
        />
      )}

      {loading && !data && <Loading label="IndexLayout" />}

      {data && (
        <>
          <div className="grid gap-2 grid-cols-2 md:grid-cols-4">
            <StatTile
              label="clusters"
              value={data.clusters.length}
              hint={`memory_type ${data.memoryType} · generation ${data.generation}`}
            />
            <StatTile
              label="trained members"
              value={groupDigits(data.totalSize)}
              hint="excludes un-indexed WAL-tail writes"
            />
            <StatTile
              label="dim"
              value={data.dim === 0 ? "none" : data.dim}
              hint={
                data.dim === 0
                  ? "this type stores no embeddings"
                  : "float32 per centroid"
              }
              tone={data.dim === 0 ? "warn" : "normal"}
            />
            <StatTile
              label="largest / mean"
              value={imbalance ? `${imbalance.toFixed(2)}×` : "—"}
              tone={imbalance && imbalance > 3 ? "warn" : "normal"}
              hint="cluster imbalance — uneven probe cost"
            />
          </div>

          <Panel
            title="pca projection"
            subtitle={
              <>
                Computed in this browser from the centroids
                {data.members.length > 0
                  ? ` and ${data.members.length} sampled members`
                  : ""}
                . Click a centroid to select it; the table below follows.
              </>
            }
            actions={
              <>
                <span className="font-mono text-[11px] text-ink-faint">
                  {fmtMs(data.elapsedMs)}
                </span>
                {selected !== null && (
                  <Button onClick={() => setSelected(null)}>
                    clear selection
                  </Button>
                )}
              </>
            }
          >
            {data.clusters.length === 0 ? (
              <Empty title="this memory_type has no trained clusters">
                Nothing has been indexed for type {data.memoryType} yet, so there
                are no centroids to project. Run the indexer, then reload.
              </Empty>
            ) : (
              <ClusterScatter
                clusters={data.clusters}
                members={data.members}
                dim={data.dim}
                totalSize={data.totalSize}
                selection={{ clusterId: selected, onSelect: setSelected }}
              />
            )}
          </Panel>

          <Panel
            title="clusters"
            subtitle="The chart's table-view twin: every value the scatter encodes is readable here without hovering."
            bodyClassName="p-0"
          >
            {rows.length === 0 ? (
              <div className="p-3">
                <Empty title="no clusters" />
              </div>
            ) : (
              <TableShell
                head={
                  <>
                    <SortTh
                      label="cluster_id"
                      k="clusterId"
                      cur={sortKey}
                      dir={sortDir}
                      onSort={sortBy}
                      className="w-28"
                    />
                    <SortTh
                      label="size"
                      k="size"
                      cur={sortKey}
                      dir={sortDir}
                      onSort={sortBy}
                      className="w-28 text-right"
                      title="trained size — excludes un-indexed WAL-tail writes"
                    />
                    <Th className="w-40 text-right">% of corpus</Th>
                    <SortTh
                      label="tags"
                      k="tags"
                      cur={sortKey}
                      dir={sortDir}
                      onSort={sortBy}
                      title="the cluster's tag summary (union over members)"
                    />
                    <SortTh
                      label="has_untagged"
                      k="hasUntagged"
                      cur={sortKey}
                      dir={sortDir}
                      onSort={sortBy}
                      className="w-32"
                    />
                  </>
                }
              >
                {rows.map((c) => {
                  const share = sharePct(c.size, data.totalSize);
                  const pctNum = parseFloat(share) || 0;
                  return (
                    <tr
                      key={c.clusterId}
                      onClick={() =>
                        setSelected(selected === c.clusterId ? null : c.clusterId)
                      }
                      className={`cursor-pointer hover:bg-panel-2 ${
                        selected === c.clusterId ? "bg-accent-dim/40" : ""
                      }`}
                    >
                      <Td className="font-mono tnum">{c.clusterId}</Td>
                      <Td className="font-mono text-right tnum">
                        {groupDigits(c.size)}
                      </Td>
                      <Td className="text-right">
                        <span className="inline-flex items-center gap-2 justify-end w-full">
                          <span
                            aria-hidden
                            className="h-1.5 rounded-sm bg-viz-1/60"
                            style={{
                              width: `${Math.max(2, Math.min(100, pctNum * 3))}px`,
                            }}
                          />
                          <span className="font-mono tnum text-ink-dim w-12 text-right">
                            {share}
                          </span>
                        </span>
                      </Td>
                      <Td>
                        {c.tags.length === 0 ? (
                          <span className="font-mono text-[11px] text-ink-faint">
                            none
                          </span>
                        ) : (
                          <span className="flex flex-wrap gap-1">
                            {c.tags.slice(0, 8).map((t) => (
                              <Tag key={t}>{t}</Tag>
                            ))}
                            {c.tags.length > 8 && (
                              <span className="font-mono text-[10px] text-ink-faint">
                                +{c.tags.length - 8}
                              </span>
                            )}
                          </span>
                        )}
                      </Td>
                      <Td>
                        <span
                          className={`font-mono text-[11px] ${
                            c.hasUntagged ? "text-ink" : "text-ink-faint"
                          }`}
                        >
                          {String(c.hasUntagged)}
                        </span>
                      </Td>
                    </tr>
                  );
                })}
              </TableShell>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}

function SortTh({
  label,
  k,
  cur,
  dir,
  onSort,
  className = "",
  title,
}: {
  label: string;
  k: SortKey;
  cur: SortKey;
  dir: SortDir;
  onSort: (k: SortKey) => void;
  className?: string;
  title?: string;
}) {
  const active = cur === k;
  return (
    <Th className={className} title={title}>
      <button
        type="button"
        onClick={() => onSort(k)}
        className={`inline-flex items-center gap-1 uppercase tracking-[0.08em] ${
          active ? "text-ink" : "text-ink-faint hover:text-ink-dim"
        }`}
      >
        {label}
        <span aria-hidden className="text-[9px]">
          {active ? (dir === "asc" ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </Th>
  );
}
