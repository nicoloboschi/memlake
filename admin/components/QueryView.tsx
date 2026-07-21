"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getJson, isAbort, postJson } from "@/lib/client";
import { fmtMs, fmtScore, truncate } from "@/lib/format";
import { shortId } from "@/lib/ids";
import {
  DEFAULT_RRF_K,
  DEFAULT_WEIGHTS,
  groupByMemoryType,
  rankHits,
  type ArmWeights,
  type SortMode,
} from "@/lib/fusion";
import {
  ARMS,
  ARM_HELP,
  ARM_SCORE_KIND,
  CONSISTENCIES,
  type Arm,
  type ArmScoreJson,
  type Consistency,
  type EmbedStatusJson,
  type HitJson,
  type QueryJson,
  type QueryRequestBody,
  type TagsMatch,
} from "@/lib/types";
import { l2Norm, parseVectorInput } from "@/lib/vector";
import {
  Button,
  Empty,
  ErrorBanner,
  Field,
  NumberInput,
  Panel,
  SegmentedControl,
  StatTile,
  Tag,
  Td,
  TableShell,
  TextArea,
  TextInput,
  Th,
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

const VECTOR_MODES = [
  { value: "embed" as const, label: "embed text", title: "embed the query text server-side with bge-small-en-v1.5" },
  { value: "raw" as const, label: "raw vector", title: "paste a JSON float array to send verbatim" },
  { value: "none" as const, label: "no vector", title: "skip the dense and graph arms; text/BM25 only" },
];

const ARM_COLOR: Record<Arm, string> = {
  dense: "text-arm-dense",
  text: "text-arm-text",
  graph: "text-arm-graph",
  temporal: "text-arm-temporal",
};

export function QueryView({ namespace }: { namespace: string }) {
  const knownTypes = useKnownTypes(namespace);

  // ---- inputs
  const [text, setText] = useState("");
  const [typesRaw, setTypesRaw] = useState("");
  const [tagsRaw, setTagsRaw] = useState("");
  const [tagsMode, setTagsMode] = useState<TagsMatch>("ANY");
  const [vectorTopK, setVectorTopK] = useState(50);
  const [textTopK, setTextTopK] = useState(50);
  const [graphTopK, setGraphTopK] = useState(50);
  const [nprobe, setNprobe] = useState(0);
  const [consistency, setConsistency] = useState<Consistency>("STRONG");
  const [vectorMode, setVectorMode] = useState<"embed" | "raw" | "none">("embed");
  const [rawVector, setRawVector] = useState("");
  const [temporalFrom, setTemporalFrom] = useState("");
  const [temporalTo, setTemporalTo] = useState("");

  // ---- results
  const [result, setResult] = useState<QueryJson | null>(null);
  const [wallMs, setWallMs] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState<string>("");
  const [selected, setSelected] = useState<HitJson | null>(null);

  // ---- client-side fusion controls
  const [weights, setWeights] = useState<ArmWeights>(DEFAULT_WEIGHTS);
  const [rrfK, setRrfK] = useState(DEFAULT_RRF_K);
  const [sortMode, setSortMode] = useState<SortMode>("rrf");

  const [embedStatus, setEmbedStatus] = useState<EmbedStatusJson | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refreshEmbedStatus = useCallback(async (signal?: AbortSignal) => {
    try {
      setEmbedStatus(await getJson<EmbedStatusJson>("/api/embed", signal));
    } catch {
      // Non-fatal: the strip just shows nothing.
    }
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    // Read the model's load state on mount so the query box can say whether the
    // first request will pay for a ~90MB download.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshEmbedStatus(ac.signal);
    return () => ac.abort();
  }, [refreshEmbedStatus]);

  async function warmModel() {
    setEmbedStatus((s) => (s ? { ...s, state: "loading" } : s));
    try {
      setEmbedStatus(await postJson<EmbedStatusJson>("/api/embed", {}));
    } catch (e) {
      setError(e);
      void refreshEmbedStatus();
    }
  }

  const run = useCallback(async () => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    const parsedTypes = parseMemoryTypes(typesRaw);
    if (parsedTypes.error) {
      setError(new Error(parsedTypes.error));
      return;
    }

    let vector: number[] | null = null;
    if (vectorMode === "raw") {
      try {
        vector = parseVectorInput(rawVector);
      } catch (e) {
        setError(e);
        return;
      }
    }

    setLoading(true);
    setError(null);
    setSelected(null);
    // A cold model load dominates the first request; say so rather than looking
    // hung for ninety seconds.
    setPhase(
      vectorMode === "embed" && embedStatus?.state !== "ready"
        ? "loading embedding model (first use downloads ~90MB)"
        : "Query",
    );

    const body: QueryRequestBody = {
      text,
      memoryTypes: parsedTypes.types,
      tags: buildTagFilter(tagsRaw, tagsMode),
      vectorTopK,
      textTopK,
      graphTopK,
      nprobe,
      consistency,
      vectorMode,
      vector,
      temporalFrom: temporalFrom.trim() || null,
      temporalTo: temporalTo.trim() || null,
    };

    const t0 = performance.now();
    try {
      const res = await postJson<QueryJson>(
        `/api/namespaces/${encodeURIComponent(namespace)}/query`,
        body,
        ac.signal,
      );
      setWallMs(performance.now() - t0);
      setResult(res);
      void refreshEmbedStatus();
    } catch (e) {
      if (isAbort(e)) return;
      setError(e);
      setResult(null);
      setWallMs(null);
      void refreshEmbedStatus();
    } finally {
      if (abortRef.current === ac) {
        setLoading(false);
        setPhase("");
      }
    }
  }, [
    namespace,
    text,
    typesRaw,
    tagsRaw,
    tagsMode,
    vectorTopK,
    textTopK,
    graphTopK,
    nprobe,
    consistency,
    vectorMode,
    rawVector,
    temporalFrom,
    temporalTo,
    embedStatus?.state,
    refreshEmbedStatus,
  ]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const groups = result ? groupByMemoryType(result.hits) : new Map<number, HitJson[]>();
  const rawNorm = vectorMode === "raw" && rawVector.trim() ? safeNorm(rawVector) : null;

  return (
    <div className="flex min-h-full">
      <div className="flex-1 min-w-0 p-4 flex flex-col gap-4">
        <Panel
          title="query"
          subtitle="One call runs all three arms over one shared snapshot. `vector` drives dense + graph; `text` drives full-text."
          actions={
            <SegmentedControl
              value={consistency}
              onChange={setConsistency}
              options={CONSISTENCY_OPTIONS}
              disabled={loading}
            />
          }
        >
          <Field
            label="text"
            hint="drives the BM25 arm, and — in `embed text` mode — the dense/graph query vector too"
          >
            <TextArea
              rows={3}
              value={text}
              spellCheck={false}
              placeholder="what are the effects of …"
              disabled={loading}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void run();
              }}
            />
          </Field>

          <div className="mt-3 grid gap-3 md:grid-cols-2">
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

          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
            <Field label="vector_top_k" hint="0 = server default">
              <NumberInput
                min={0}
                value={vectorTopK}
                disabled={loading}
                onChange={(e) => setVectorTopK(Number(e.target.value) || 0)}
              />
            </Field>
            <Field label="text_top_k" hint="0 = server default">
              <NumberInput
                min={0}
                value={textTopK}
                disabled={loading}
                onChange={(e) => setTextTopK(Number(e.target.value) || 0)}
              />
            </Field>
            <Field label="graph_top_k" hint="0 = server default">
              <NumberInput
                min={0}
                value={graphTopK}
                disabled={loading}
                onChange={(e) => setGraphTopK(Number(e.target.value) || 0)}
              />
            </Field>
            <Field label="nprobe" hint="IVF clusters probed; 0 = default">
              <NumberInput
                min={0}
                value={nprobe}
                disabled={loading}
                onChange={(e) => setNprobe(Number(e.target.value) || 0)}
              />
            </Field>
          </div>

          <div className="mt-3 border-t border-line pt-3">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint">
                query vector
              </span>
              <SegmentedControl
                value={vectorMode}
                onChange={setVectorMode}
                options={VECTOR_MODES}
                disabled={loading}
              />
              <EmbedStatusStrip
                status={embedStatus}
                active={vectorMode === "embed"}
                onWarm={() => void warmModel()}
              />
            </div>

            {vectorMode === "raw" && (
              <div className="mt-2">
                <Field
                  label="vector (JSON float array)"
                  hint={
                    rawNorm === null ? (
                      "sent verbatim as bytes f32le — little-endian float32, not JSON, on the wire"
                    ) : (
                      <span>
                        {rawNorm.dim} components, ‖v‖ = {rawNorm.norm.toFixed(6)}
                        {Math.abs(rawNorm.norm - 1) > 1e-3 && (
                          <span className="text-warn">
                            {" "}
                            — not unit norm; the corpus was indexed with
                            L2-normalized vectors
                          </span>
                        )}
                      </span>
                    )
                  }
                >
                  <TextArea
                    rows={4}
                    value={rawVector}
                    spellCheck={false}
                    disabled={loading}
                    placeholder="[0.0123, -0.0456, …]"
                    onChange={(e) => setRawVector(e.target.value)}
                    className="text-[11px]"
                  />
                </Field>
              </div>
            )}

            {vectorMode === "none" && (
              <p className="mt-2 text-[11px] text-ink-faint">
                No vector is sent, so the dense, graph and temporal arms do not
                run — this is a text-only BM25 query.
              </p>
            )}
          </div>

          <div className="mt-3 border-t border-line pt-3">
            <div className="grid grid-cols-2 gap-2 max-w-lg">
              <Field
                label="temporal_from"
                hint="epoch int64; unit only has to match what was written"
              >
                <TextInput
                  value={temporalFrom}
                  spellCheck={false}
                  disabled={loading}
                  placeholder="(empty = arm off)"
                  onChange={(e) => setTemporalFrom(e.target.value)}
                  className="text-right"
                />
              </Field>
              <Field label="temporal_to" hint={temporalHint(vectorMode)}>
                <TextInput
                  value={temporalTo}
                  spellCheck={false}
                  disabled={loading}
                  placeholder="(empty = arm off)"
                  onChange={(e) => setTemporalTo(e.target.value)}
                  className="text-right"
                />
              </Field>
            </div>
            <p className="mt-1 text-[10px] text-ink-faint">
              The temporal arm selects entry points whose effective time —
              COALESCE(occurred_start, mentioned_at, occurred_end) — falls in the
              window, spreads one hop through links, and scores by proximity to
              the window centre. Both bounds AND a vector are required; omit
              either and the arm is skipped.
            </p>
          </div>

          <div className="mt-3 flex items-center gap-3">
            <Button variant="primary" onClick={() => void run()} disabled={loading}>
              {loading ? "running…" : "Query"}
            </Button>
            <span className="font-mono text-[11px] text-ink-faint">
              ⌘/ctrl + enter
            </span>
            {loading && phase && (
              <span className="font-mono text-[11px] text-warn">{phase}…</span>
            )}
          </div>
        </Panel>

        {Boolean(error) && (
          <ErrorBanner error={error} what={`Query(${namespace})`} onRetry={() => void run()} />
        )}

        {result && (
          <>
            <div className="grid gap-2 grid-cols-2 md:grid-cols-4">
              <StatTile
                label="hits"
                value={result.hits.length}
                hint={`${groups.size} memory_type group${groups.size === 1 ? "" : "s"}`}
              />
              <StatTile
                label="load_roundtrips"
                value={result.loadRoundtrips}
                hint="object-storage waves, shared across arms + types"
                tone="accent"
              />
              <StatTile
                label="wall clock"
                value={fmtMs(wallMs)}
                hint={`Query RPC ${fmtMs(result.rpcMs)}${
                  result.embedMs !== null ? ` · embed ${fmtMs(result.embedMs)}` : ""
                }`}
              />
              <StatTile
                label="query vector"
                value={
                  result.vectorSource === "none"
                    ? "none"
                    : `f32[${result.vectorDim ?? "?"}]`
                }
                hint={
                  <>
                    {result.vectorSource === "embedded"
                      ? `${result.embeddingModel} (prefixed)`
                      : result.vectorSource === "raw"
                        ? "pasted verbatim"
                        : "dense + graph + temporal arms skipped"}
                    {result.temporalWindow && (
                      <span className="block text-arm-temporal">
                        temporal [{result.temporalWindow.from},{" "}
                        {result.temporalWindow.to}]
                      </span>
                    )}
                  </>
                }
              />
            </div>

            <FusionControls
              weights={weights}
              onWeights={setWeights}
              k={rrfK}
              onK={setRrfK}
              sortMode={sortMode}
              onSortMode={setSortMode}
            />

            {result.hits.length === 0 ? (
              <Empty title="no hits">
                No arm surfaced a candidate. Widen the per-arm depths, relax the
                tag filter, or check that the memory_types you asked for exist.
              </Empty>
            ) : (
              [...groups.entries()].map(([memoryType, hits]) => (
                <ResultGroup
                  key={memoryType}
                  memoryType={memoryType}
                  hits={hits}
                  weights={weights}
                  rrfK={rrfK}
                  sortMode={sortMode}
                  selectedId={selected?.id ?? null}
                  onSelect={setSelected}
                />
              ))
            )}
          </>
        )}
      </div>

      {selected && (
        <aside className="w-[26rem] shrink-0 border-l border-line bg-panel overflow-y-auto max-h-[calc(100vh-5rem)] sticky top-20">
          <header className="flex items-center gap-2 px-3 py-2 border-b border-line sticky top-0 bg-panel">
            <h3 className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim flex-1">
              hit · inline memory
            </h3>
            <Button variant="ghost" onClick={() => setSelected(null)} title="close">
              ✕
            </Button>
          </header>
          <div className="p-3">
            <div className="mb-3 grid grid-cols-2 gap-2">
              {ARMS.map((arm) => (
                <ArmBadge key={arm} arm={arm} s={selected[arm]} />
              ))}
            </div>
            <MemoryDetail
              id={selected.id}
              memoryType={selected.memoryType}
              memory={selected.memory}
            />
            <p className="mt-3 text-[10px] text-ink-faint leading-relaxed">
              The stored memory came back inline with the hit — the server had it
              materialized to score the candidate, so there was no hydrate
              roundtrip. The embedding is omitted from Query; fetch it with Get
              from the browse page.
            </p>
          </div>
        </aside>
      )}
    </div>
  );
}

// ---- fusion -----------------------------------------------------------------

function FusionControls({
  weights,
  onWeights,
  k,
  onK,
  sortMode,
  onSortMode,
}: {
  weights: ArmWeights;
  onWeights: (w: ArmWeights) => void;
  k: number;
  onK: (k: number) => void;
  sortMode: SortMode;
  onSortMode: (m: SortMode) => void;
}) {
  return (
    <Panel
      title="fusion — computed in this browser"
      subtitle="memlake returns RAW per-arm scores and does NO fusion. The ordering below is this page's Reciprocal Rank Fusion over those scores; change the weights and it changes here only."
    >
      <div className="flex items-end gap-4 flex-wrap">
        <div>
          <span className="block font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1">
            order by
          </span>
          <SegmentedControl
            value={sortMode}
            onChange={onSortMode}
            options={[
              { value: "rrf" as SortMode, label: "RRF", title: "fused ordering" },
              ...ARMS.map((a) => ({
                value: a as SortMode,
                label: a,
                title: `sort by the ${a} arm alone (${ARM_SCORE_KIND[a]})`,
              })),
            ]}
          />
        </div>

        {ARMS.map((arm) => (
          <Field key={arm} label={`w_${arm}`} className="w-20">
            <NumberInput
              min={0}
              step={0.1}
              value={weights[arm]}
              disabled={sortMode !== "rrf"}
              onChange={(e) =>
                onWeights({ ...weights, [arm]: Number(e.target.value) || 0 })
              }
            />
          </Field>
        ))}

        <Field label="RRF k" className="w-20" hint="rank damping">
          <NumberInput
            min={1}
            value={k}
            disabled={sortMode !== "rrf"}
            onChange={(e) => onK(Number(e.target.value) || 1)}
          />
        </Field>

        <p className="font-mono text-[10px] text-ink-faint pb-2 flex-1 min-w-[16rem]">
          score(d) = Σ_arms w_arm / (k + rank_arm(d) + 1), summed only over arms
          where present = true
        </p>
      </div>
    </Panel>
  );
}

// ---- results ----------------------------------------------------------------

function ResultGroup({
  memoryType,
  hits,
  weights,
  rrfK,
  sortMode,
  selectedId,
  onSelect,
}: {
  memoryType: number;
  hits: HitJson[];
  weights: ArmWeights;
  rrfK: number;
  sortMode: SortMode;
  selectedId: string | null;
  onSelect: (h: HitJson) => void;
}) {
  const ranked = rankHits(hits, weights, rrfK, sortMode);
  const counts = ARMS.map((arm) => ({
    arm,
    n: hits.filter((h) => h[arm].present).length,
  }));

  return (
    <Panel
      title={`memory_type ${memoryType}`}
      subtitle={
        <>
          {hits.length} candidate{hits.length === 1 ? "" : "s"} · surfaced by{" "}
          {counts.map((c, i) => (
            <span key={c.arm}>
              {i > 0 && ", "}
              <span className={ARM_COLOR[c.arm]}>{c.arm}</span> {c.n}
            </span>
          ))}
          . This type is an independent index — it is never fused with the
          others.
        </>
      }
      bodyClassName="p-0"
    >
      <TableShell
        head={
          <>
            <Th className="w-10 text-right">#</Th>
            <Th className="w-36">id</Th>
            <Th>text</Th>
            {ARMS.map((arm) => (
              <Th
                key={arm}
                className={`w-24 text-right ${ARM_COLOR[arm]}`}
                title={`${arm} arm — ${ARM_SCORE_KIND[arm]}. ${ARM_HELP[arm]}`}
              >
                {arm}
              </Th>
            ))}
            <Th className="w-24 text-right" title="client-side RRF score">
              rrf
            </Th>
          </>
        }
      >
        {ranked.map((r) => (
          <tr
            key={r.hit.id}
            onClick={() => onSelect(r.hit)}
            className={`cursor-pointer hover:bg-panel-2 ${
              selectedId === r.hit.id ? "bg-accent-dim/40" : ""
            }`}
          >
            <Td className="font-mono text-right tnum text-ink-faint">{r.rank}</Td>
            <Td className="font-mono text-[11px] text-ink-dim" title={r.hit.id}>
              {shortId(r.hit.id)}
            </Td>
            <Td className="text-ink">
              {r.hit.memory?.text ? (
                <>
                  {truncate(r.hit.memory.text.replace(/\s+/g, " "), 140)}
                  {r.hit.memory.tags.length > 0 && (
                    <span className="ml-2 inline-flex gap-1 align-middle">
                      {r.hit.memory.tags.slice(0, 3).map((t) => (
                        <Tag key={t}>{t}</Tag>
                      ))}
                    </span>
                  )}
                </>
              ) : (
                <span className="text-ink-faint">no inline memory</span>
              )}
            </Td>
            {ARMS.map((arm) => (
              <ArmCell key={arm} arm={arm} s={r.hit[arm]} />
            ))}
            <Td className="font-mono text-right tnum text-ink">
              {fmtScore(r.score, 5)}
            </Td>
          </tr>
        ))}
      </TableShell>
      <div className="px-3 py-2 border-t border-line text-[10px] text-ink-faint font-mono">
        <span className="text-ink-dim">∅</span> = arm absent (present = false):
        the arm never surfaced this id. That is NOT a score of 0.
      </div>
    </Panel>
  );
}

function ArmCell({ arm, s }: { arm: Arm; s: ArmScoreJson }) {
  if (!s.present) {
    return (
      <Td
        className="text-right bg-line/20"
        title={`the ${arm} arm did not surface this hit (present = false) — not a score of 0`}
      >
        <span className="font-mono text-ink-faint">∅</span>
      </Td>
    );
  }
  return (
    <Td className="text-right" title={`${arm}: ${ARM_SCORE_KIND[arm]}`}>
      <span className={`font-mono text-[10px] tnum ${ARM_COLOR[arm]}`}>
        #{s.rank}
      </span>{" "}
      <span className="font-mono text-[11px] tnum text-ink">
        {fmtScore(s.score)}
      </span>
    </Td>
  );
}

function ArmBadge({ arm, s }: { arm: Arm; s: ArmScoreJson }) {
  return (
    <div className="border border-line bg-panel-2 rounded-sm px-2 py-1">
      <div className={`font-mono text-[10px] uppercase ${ARM_COLOR[arm]}`}>
        {arm}
      </div>
      {s.present ? (
        <div className="font-mono text-[12px] tnum text-ink">
          {fmtScore(s.score)}
          <span className="ml-1 text-[10px] text-ink-faint">#{s.rank}</span>
        </div>
      ) : (
        <div className="font-mono text-[12px] text-ink-faint" title="present = false">
          ∅ absent
        </div>
      )}
    </div>
  );
}

// ---- embedding status -------------------------------------------------------

function EmbedStatusStrip({
  status,
  active,
  onWarm,
}: {
  status: EmbedStatusJson | null;
  active: boolean;
  onWarm: () => void;
}) {
  if (!status) return null;

  if (!status.enabled) {
    return (
      <span className="font-mono text-[11px] text-warn">
        embeddings disabled (MEMLAKE_EMBEDDINGS=off) — use raw vector or no vector
      </span>
    );
  }

  const tone =
    status.state === "ready"
      ? "text-ok"
      : status.state === "error"
        ? "text-danger"
        : status.state === "loading"
          ? "text-warn"
          : "text-ink-faint";

  return (
    <span className="flex items-center gap-2 font-mono text-[11px]">
      <span
        className="text-ink-faint"
        title={`dim ${status.dim}, ${status.pooling} pooling, L2-normalized, query prefix: ${status.queryPrefix}`}
      >
        {status.model}
      </span>
      <span className={tone}>
        {status.state === "idle"
          ? "model not loaded"
          : status.state === "loading"
            ? "loading model…"
            : status.state}
      </span>
      {active && status.state === "idle" && (
        <Button onClick={onWarm}>warm up</Button>
      )}
      {status.error && (
        <span className="text-danger" title={status.error}>
          {truncate(status.error, 60)}
        </span>
      )}
    </span>
  );
}

function temporalHint(vectorMode: "embed" | "raw" | "none"): string {
  return vectorMode === "none"
    ? "the arm cannot run without a vector — its entry points are similarity-ranked"
    : "set both bounds to run the arm";
}

function safeNorm(raw: string): { dim: number; norm: number } | null {
  try {
    const v = parseVectorInput(raw);
    return { dim: v.length, norm: l2Norm(v) };
  } catch {
    return null;
  }
}
