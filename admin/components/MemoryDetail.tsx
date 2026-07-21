"use client";

/**
 * The full MemoryPayload, laid out for reading. Used by the Browse drawer and
 * the Query result expander.
 */

import { fmtEpochGuess, fmtScore } from "@/lib/format";
import type { MemoryPayloadJson, VectorSummary } from "@/lib/types";
import { CopyableId, KeyValue, Tag } from "@/components/ui";

function Section({
  title,
  count,
  children,
}: {
  title: string;
  count?: number;
  children: React.ReactNode;
}) {
  return (
    <div className="border-t border-line pt-2 mt-2 first:border-t-0 first:pt-0 first:mt-0">
      <h4 className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1.5">
        {title}
        {count !== undefined && (
          <span className="ml-1.5 text-ink-faint/70">[{count}]</span>
        )}
      </h4>
      {children}
    </div>
  );
}

function None() {
  return <span className="font-mono text-[11px] text-ink-faint">none</span>;
}

export function VectorSummaryView({ v }: { v: VectorSummary }) {
  return (
    <div>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-0.5 font-mono text-[11px]">
        <span className="text-ink-dim">
          dim <span className="text-ink tnum">{v.dim}</span>
        </span>
        <span className="text-ink-dim">
          bytes <span className="text-ink tnum">{v.bytes}</span>
        </span>
        <span className="text-ink-dim">
          ‖v‖ <span className="text-ink tnum">{v.norm.toFixed(6)}</span>
          {Math.abs(v.norm - 1) < 1e-3 && (
            <span className="ml-1 text-ok">normalized</span>
          )}
        </span>
      </div>
      <div className="mt-1 font-mono text-[11px] text-ink-dim break-all">
        [{v.head.map((x) => x.toFixed(5)).join(", ")}
        {v.dim > v.head.length && (
          <span className="text-ink-faint">
            , … {v.dim - v.head.length} more
          </span>
        )}
        ]
      </div>
    </div>
  );
}

export function MemoryDetail({
  id,
  memoryType,
  memory,
  vector,
}: {
  id: string;
  memoryType: number;
  memory: MemoryPayloadJson | null;
  vector?: VectorSummary | null;
}) {
  if (!memory) {
    return (
      <div className="font-mono text-[11px] text-ink-faint">
        the server returned no MemoryPayload for {id}
      </div>
    );
  }

  const ts = memory.timestamps;
  const tsEntries = ts
    ? (
        [
          ["event_date", ts.eventDate],
          ["occurred_start", ts.occurredStart],
          ["occurred_end", ts.occurredEnd],
          ["mentioned_at", ts.mentionedAt],
        ] as const
      )
        .filter(([, v]) => v !== null)
        .map(([k, v]) => ({
          k,
          v: (
            <span>
              <span className="tnum">{v}</span>
              <span className="ml-2 text-ink-faint">{fmtEpochGuess(v!)}</span>
            </span>
          ),
        }))
    : [];

  const metaEntries = Object.entries(memory.metadata);

  return (
    <div className="flex flex-col gap-0">
      <Section title="identity">
        <KeyValue
          entries={[
            { k: "id", v: <CopyableId value={id} className="text-[12px]" /> },
            { k: "memory_type", v: <span className="tnum">{memoryType}</span> },
            {
              k: "proof_count",
              v: <span className="tnum">{memory.proofCount}</span>,
            },
          ]}
        />
      </Section>

      <Section title="text">
        {memory.text ? (
          <p className="font-mono text-[12px] text-ink whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
            {memory.text}
          </p>
        ) : (
          <None />
        )}
      </Section>

      <Section title="tags" count={memory.tags.length}>
        {memory.tags.length ? (
          <div className="flex flex-wrap gap-1">
            {memory.tags.map((t) => (
              <Tag key={t}>{t}</Tag>
            ))}
          </div>
        ) : (
          <None />
        )}
      </Section>

      <Section title="timestamps">
        {tsEntries.length ? <KeyValue entries={tsEntries} /> : <None />}
      </Section>

      <Section title="entity_ids" count={memory.entityIds.length}>
        {memory.entityIds.length ? (
          <ul className="flex flex-col gap-0.5">
            {memory.entityIds.map((e) => (
              <li key={e}>
                <CopyableId value={e} />
              </li>
            ))}
          </ul>
        ) : (
          <None />
        )}
      </Section>

      <Section title="causal_out" count={memory.causalOut.length}>
        {memory.causalOut.length ? (
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-left">
                <th className="font-mono text-[10px] font-normal text-ink-faint pb-1">
                  target
                </th>
                <th className="font-mono text-[10px] font-normal text-ink-faint pb-1">
                  link_type
                </th>
                <th className="font-mono text-[10px] font-normal text-ink-faint pb-1 text-right">
                  weight
                </th>
              </tr>
            </thead>
            <tbody>
              {memory.causalOut.map((e, i) => (
                <tr key={`${e.target}-${i}`}>
                  <td className="pr-3">
                    <CopyableId value={e.target} />
                  </td>
                  <td className="pr-3 font-mono text-[11px] text-ink-dim">
                    {e.linkType}
                  </td>
                  <td className="font-mono text-[11px] text-ink text-right tnum">
                    {fmtScore(e.weight, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <None />
        )}
      </Section>

      <Section title="metadata (opaque)" count={metaEntries.length}>
        {metaEntries.length ? (
          <KeyValue
            entries={metaEntries.map(([k, v]) => ({
              k,
              v: <span className="whitespace-pre-wrap break-all">{v}</span>,
            }))}
          />
        ) : (
          <None />
        )}
        <p className="mt-1.5 text-[10px] text-ink-faint">
          memlake stores and returns this verbatim — it never indexes or
          interprets it.
        </p>
      </Section>

      {vector && (
        <Section title="vector">
          <VectorSummaryView v={vector} />
        </Section>
      )}
    </div>
  );
}
