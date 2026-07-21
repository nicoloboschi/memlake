"use client";

/** Filter controls shared by Browse and Query. */

import { useEffect, useState } from "react";

import { getJson, isAbort } from "@/lib/client";
import {
  TAGS_MATCHES,
  TAGS_MATCH_HELP,
  type StatsJson,
  type TagFilterInput,
  type TagsMatch,
} from "@/lib/types";
import { Field, Select, TextInput } from "@/components/ui";

/** Parse "0, 1,2" -> [0,1,2]; anything out of u8 range is dropped by the caller. */
export function parseMemoryTypes(raw: string): { types: number[]; error: string | null } {
  const parts = raw
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const types: number[] = [];
  for (const p of parts) {
    const n = Number(p);
    if (!Number.isInteger(n) || n < 0 || n > 255) {
      return { types: [], error: `"${p}" is not a memory_type — must be an integer 0-255 (u8)` };
    }
    types.push(n);
  }
  return { types: Array.from(new Set(types)).sort((a, b) => a - b), error: null };
}

export function parseTags(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function buildTagFilter(raw: string, mode: TagsMatch): TagFilterInput | null {
  const tags = parseTags(raw);
  return tags.length ? { tags, mode } : null;
}

/**
 * Discover which memory_types exist, so the type selector can offer real values
 * instead of asking the operator to guess. Degrades to manual entry when Stats
 * is unavailable.
 */
export function useKnownTypes(namespace: string): number[] | null {
  const [types, setTypes] = useState<number[] | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    void (async () => {
      try {
        const s = await getJson<StatsJson>(
          `/api/namespaces/${encodeURIComponent(namespace)}/stats?consistency=EVENTUAL`,
          ac.signal,
        );
        setTypes(s.types.map((t) => t.memoryType));
      } catch (e) {
        if (isAbort(e)) return;
        // Non-fatal: the selector falls back to free-text entry.
        setTypes(null);
      }
    })();
    return () => ac.abort();
  }, [namespace]);

  return types;
}

export function MemoryTypesField({
  value,
  onChange,
  known,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  known: number[] | null;
  disabled?: boolean;
}) {
  const { types, error } = parseMemoryTypes(value);
  const selected = new Set(types);

  // Cheap enough to rebuild per render — `known` is a handful of u8s.
  function toggle(t: number) {
    const next = new Set(selected);
    if (next.has(t)) next.delete(t);
    else next.add(t);
    onChange([...next].sort((a, b) => a - b).join(","));
  }

  return (
    <Field
      label="memory_types"
      hint={
        error ? (
          <span className="text-danger">{error}</span>
        ) : (
          <>
            empty = every type in the snapshot. Types are independent indexes;
            results are never fused across them.
          </>
        )
      }
    >
      <TextInput
        value={value}
        disabled={disabled}
        spellCheck={false}
        placeholder="e.g. 0,1  (empty = all)"
        onChange={(e) => onChange(e.target.value)}
      />
      {known && known.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {known.map((t) => (
            <button
              key={t}
              type="button"
              disabled={disabled}
              onClick={() => toggle(t)}
              className={`font-mono text-[10px] leading-4 px-1.5 py-px rounded-sm border ${
                selected.has(t)
                  ? "border-accent/60 bg-accent-dim text-accent"
                  : "border-line-strong bg-panel-2 text-ink-dim hover:text-ink"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      )}
    </Field>
  );
}

export function TagsField({
  tags,
  onTagsChange,
  mode,
  onModeChange,
  disabled,
}: {
  tags: string;
  onTagsChange: (v: string) => void;
  mode: TagsMatch;
  onModeChange: (v: TagsMatch) => void;
  disabled?: boolean;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_9.5rem] gap-2">
      <Field label="tags" hint="comma-separated; empty = no tag filter">
        <TextInput
          value={tags}
          disabled={disabled}
          spellCheck={false}
          placeholder="alpha, beta"
          onChange={(e) => onTagsChange(e.target.value)}
        />
      </Field>
      <Field label="TagsMatch" hint={TAGS_MATCH_HELP[mode]}>
        <Select
          value={mode}
          onChange={onModeChange}
          options={TAGS_MATCHES}
          disabled={disabled}
        />
      </Field>
    </div>
  );
}
