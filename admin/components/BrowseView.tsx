"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { isAbort, postJson } from "@/lib/client";
import { fmtMs, truncate } from "@/lib/format";
import { shortId } from "@/lib/ids";
import {
  CONSISTENCIES,
  type Consistency,
  type GetJson,
  type ScanJson,
  type ScanRequestBody,
  type StoredMemoryJson,
  type TagsMatch,
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
  Tag,
  Td,
  TableShell,
  Th,
  Toggle,
} from "@/components/ui";
import {
  MemoryTypesField,
  TagsField,
  buildTagFilter,
  parseMemoryTypes,
  useKnownTypes,
} from "@/components/filters";
import { MemoryDetail } from "@/components/MemoryDetail";

const CONSISTENCY_OPTIONS = CONSISTENCIES.map((c) => ({ value: c, label: c }));

export function BrowseView({ namespace }: { namespace: string }) {
  const knownTypes = useKnownTypes(namespace);

  // filters
  const [typesRaw, setTypesRaw] = useState("");
  const [tagsRaw, setTagsRaw] = useState("");
  const [tagsMode, setTagsMode] = useState<TagsMatch>("ANY");
  const [limit, setLimit] = useState(50);
  const [includeVector, setIncludeVector] = useState(false);
  const [consistency, setConsistency] = useState<Consistency>("EVENTUAL");

  /**
   * The scan cursor is opaque and only valid against the generation that
   * produced it, so page numbers are meaningless. We keep the stack of tokens
   * we used to get here; "back" pops it. `stack[stack.length - 1]` is the token
   * that produced what is on screen ("" for the first page).
   */
  const [stack, setStack] = useState<string[]>([""]);
  const [data, setData] = useState<ScanJson | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<StoredMemoryJson | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (tokens: string[]) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      const parsed = parseMemoryTypes(typesRaw);
      if (parsed.error) {
        setError(new Error(parsed.error));
        return;
      }

      setLoading(true);
      setError(null);
      const body: ScanRequestBody = {
        memoryTypes: parsed.types,
        limit,
        pageToken: tokens[tokens.length - 1] ?? "",
        includeVector,
        tags: buildTagFilter(tagsRaw, tagsMode),
        consistency,
      };
      try {
        const res = await postJson<ScanJson>(
          `/api/namespaces/${encodeURIComponent(namespace)}/scan`,
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
    [namespace, typesRaw, tagsRaw, tagsMode, limit, includeVector, consistency],
  );

  // First page on mount. Filter changes require an explicit re-scan, since a
  // scan is a full walk and should not fire on every keystroke.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void run([""]);
    return () => abortRef.current?.abort();
    // Deliberately keyed on `namespace` only: depending on `run` would fire a
    // full corpus walk on every filter keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namespace]);

  const canBack = stack.length > 1;
  const canNext = Boolean(data?.nextPageToken);

  return (
    <div className="flex min-h-full">
      <div className="flex-1 min-w-0 p-4 flex flex-col gap-4">
        <Panel
          title="scan filters"
          subtitle="Scan is a full walk in cluster order — its cost grows with the corpus. It exists for browsing and debugging; use Query to find things."
          actions={
            <SegmentedControl
              value={consistency}
              onChange={setConsistency}
              options={CONSISTENCY_OPTIONS}
              disabled={loading}
            />
          }
        >
          <div className="grid gap-3 md:grid-cols-2">
            <MemoryTypesField
              value={typesRaw}
              onChange={setTypesRaw}
              known={knownTypes}
              disabled={loading}
            />
            <TagsField
              tags={tagsRaw}
              onTagsChange={setTagsRaw}
              mode={tagsMode}
              onModeChange={setTagsMode}
              disabled={loading}
            />
          </div>
          <div className="mt-3 flex items-end gap-4 flex-wrap">
            <Field label="limit" className="w-24">
              <NumberInput
                min={0}
                max={500}
                value={limit}
                disabled={loading}
                onChange={(e) => setLimit(Number(e.target.value) || 0)}
              />
            </Field>
            <div className="pb-1.5">
              <Toggle
                checked={includeVector}
                onChange={setIncludeVector}
                disabled={loading}
                label="include_vector"
              />
            </div>
            <Button
              variant="primary"
              className="mb-1"
              disabled={loading}
              onClick={() => void run([""])}
            >
              {loading ? "scanning…" : "Scan"}
            </Button>
            <span className="mb-2 text-[10px] text-ink-faint">
              a new scan restarts at the first page — the cursor is only valid
              against the generation that produced it
            </span>
          </div>
        </Panel>

        <Panel
          title="memories"
          subtitle={
            data ? (
              <>
                page {stack.length} · {data.memories.length} row
                {data.memories.length === 1 ? "" : "s"} · {fmtMs(data.elapsedMs)}
                {data.nextPageToken ? "" : " · end of scan"}
              </>
            ) : undefined
          }
          actions={
            <>
              <Button
                disabled={!canBack || loading}
                onClick={() => void run(stack.slice(0, -1))}
              >
                ← back
              </Button>
              <Button
                disabled={!canNext || loading}
                onClick={() =>
                  void run([...stack, data?.nextPageToken ?? ""])
                }
              >
                next →
              </Button>
            </>
          }
          bodyClassName="p-0"
        >
          {loading && !data && <Loading label="Scan" />}

          {Boolean(error) && (
            <div className="p-3">
              <ErrorBanner
                error={error}
                what={`Scan(${namespace})`}
                onRetry={() => void run(stack)}
              />
            </div>
          )}

          {data && data.memories.length === 0 && (
            <div className="p-3">
              <Empty title="no memories on this page">
                {stack.length > 1
                  ? "The cursor walked past the end, or the filters exclude everything here."
                  : "Nothing matched. Check the memory_type and tag filters, or write some memories first."}
              </Empty>
            </div>
          )}

          {data && data.memories.length > 0 && (
            <TableShell
              head={
                <>
                  <Th className="w-40">id</Th>
                  <Th className="w-16 text-right">type</Th>
                  <Th>text</Th>
                  <Th className="w-48">tags</Th>
                  <Th className="w-16 text-right" title="proof_count">
                    proofs
                  </Th>
                  <Th className="w-14 text-right" title="entity_ids">
                    ents
                  </Th>
                  <Th className="w-14 text-right" title="causal_out edges">
                    edges
                  </Th>
                  {includeVector && <Th className="w-44">vector</Th>}
                </>
              }
            >
              {data.memories.map((m) => (
                <tr
                  key={m.id}
                  onClick={() => setSelected(m)}
                  className={`cursor-pointer hover:bg-panel-2 ${
                    selected?.id === m.id ? "bg-accent-dim/40" : ""
                  }`}
                >
                  <Td className="font-mono text-[11px] text-ink-dim" title={m.id}>
                    {shortId(m.id)}
                  </Td>
                  <Td className="font-mono text-right tnum">{m.memoryType}</Td>
                  <Td className="text-ink">
                    {m.memory?.text ? (
                      truncate(m.memory.text.replace(/\s+/g, " "), 160)
                    ) : (
                      <span className="text-ink-faint">—</span>
                    )}
                  </Td>
                  <Td>
                    <div className="flex flex-wrap gap-1">
                      {(m.memory?.tags ?? []).slice(0, 4).map((t) => (
                        <Tag key={t}>{t}</Tag>
                      ))}
                      {(m.memory?.tags.length ?? 0) > 4 && (
                        <span className="font-mono text-[10px] text-ink-faint">
                          +{(m.memory?.tags.length ?? 0) - 4}
                        </span>
                      )}
                    </div>
                  </Td>
                  <Td className="font-mono text-right tnum text-ink-dim">
                    {m.memory?.proofCount ?? 0}
                  </Td>
                  <Td className="font-mono text-right tnum text-ink-dim">
                    {m.memory?.entityIds.length ?? 0}
                  </Td>
                  <Td className="font-mono text-right tnum text-ink-dim">
                    {m.memory?.causalOut.length ?? 0}
                  </Td>
                  {includeVector && (
                    <Td className="font-mono text-[10px] text-ink-faint">
                      {m.vector ? (
                        <>
                          <span className="text-ink-dim">
                            f32[{m.vector.dim}]
                          </span>{" "}
                          {m.vector.head
                            .slice(0, 3)
                            .map((x) => x.toFixed(3))
                            .join(" ")}
                          …
                        </>
                      ) : (
                        "—"
                      )}
                    </Td>
                  )}
                </tr>
              ))}
            </TableShell>
          )}
        </Panel>
      </div>

      {selected && (
        <MemoryDrawer
          // Keying on the id resets the drawer's fetched-vector state when a
          // different row is selected, without an effect to do it.
          key={selected.id}
          namespace={namespace}
          record={selected}
          consistency={consistency}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

/**
 * Detail panel for one row. The vector is fetched on demand through `Get` —
 * scanning with include_vector on every page is wasteful when you only wanted
 * to look at one record.
 */
function MemoryDrawer({
  namespace,
  record,
  consistency,
  onClose,
}: {
  namespace: string;
  record: StoredMemoryJson;
  consistency: Consistency;
  onClose: () => void;
}) {
  // Remounted (via `key`) whenever a different row is selected, so this initial
  // state is always the row on screen.
  const [full, setFull] = useState<StoredMemoryJson>(record);
  const [fetching, setFetching] = useState(false);
  const [getError, setGetError] = useState<unknown>(null);

  async function fetchVector() {
    setFetching(true);
    setGetError(null);
    try {
      const res = await postJson<GetJson>(
        `/api/namespaces/${encodeURIComponent(namespace)}/get`,
        { ids: [record.id], includeVector: true, consistency },
      );
      if (res.memories.length === 0) {
        setGetError(
          new Error(
            "Get returned nothing for this id — it may have been tombstoned since the scan",
          ),
        );
      } else {
        setFull(res.memories[0]);
      }
    } catch (e) {
      setGetError(e);
    } finally {
      setFetching(false);
    }
  }

  return (
    <aside className="w-[26rem] shrink-0 border-l border-line bg-panel overflow-y-auto max-h-[calc(100vh-5rem)] sticky top-20">
      <header className="flex items-center gap-2 px-3 py-2 border-b border-line sticky top-0 bg-panel">
        <h3 className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim flex-1">
          memory
        </h3>
        <Button onClick={() => void fetchVector()} disabled={fetching}>
          {fetching ? "…" : "Get + vector"}
        </Button>
        <Button variant="ghost" onClick={onClose} title="close">
          ✕
        </Button>
      </header>
      <div className="p-3">
        {Boolean(getError) && (
          <div className="mb-3">
            <ErrorBanner error={getError} what={`Get(${namespace})`} />
          </div>
        )}
        <MemoryDetail
          id={full.id}
          memoryType={full.memoryType}
          memory={full.memory}
          vector={full.vector}
        />
      </div>
    </aside>
  );
}
