"use client";

/**
 * The whole component vocabulary for the console. Tailwind only — no component
 * library — so every surface here is one of a handful of primitives.
 *
 * Client-safe by construction: nothing in this file imports a server module.
 */

import { useState, type ReactNode } from "react";

import { describeError } from "@/lib/client";

// ---- surfaces ---------------------------------------------------------------

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = "",
  bodyClassName = "p-3",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`border border-line bg-panel rounded-sm min-w-0 ${className}`}
    >
      {(title || actions) && (
        <header className="flex items-start gap-3 px-3 py-2 border-b border-line">
          <div className="min-w-0 flex-1">
            {title && (
              <h2 className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-[11px] text-ink-faint leading-snug">
                {subtitle}
              </p>
            )}
          </div>
          {actions && (
            <div className="flex items-center gap-2 shrink-0">{actions}</div>
          )}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

// ---- states -----------------------------------------------------------------

/**
 * Every RPC failure surfaces here with its gRPC code, verbatim message, and a
 * hint. The admin RPCs land in the Rust server asynchronously, so UNIMPLEMENTED
 * is a routine, expected state — not a crash.
 */
export function ErrorBanner({
  error,
  what,
  onRetry,
}: {
  error: unknown;
  what: string;
  onRetry?: () => void;
}) {
  const { codeName, message, hint } = describeError(error);
  const expected = codeName === "UNIMPLEMENTED";
  return (
    <div
      className={`border rounded-sm px-3 py-2.5 ${
        expected
          ? "border-warn/40 bg-warn/5"
          : "border-danger/40 bg-danger/5"
      }`}
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className={`font-mono text-[10px] uppercase tracking-[0.1em] px-1.5 py-0.5 rounded-sm border ${
            expected
              ? "border-warn/50 text-warn"
              : "border-danger/50 text-danger"
          }`}
        >
          {codeName}
        </span>
        <span className="font-mono text-[11px] text-ink-dim">{what}</span>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="ml-auto font-mono text-[11px] text-accent hover:underline"
          >
            retry
          </button>
        )}
      </div>
      <p className="mt-1.5 font-mono text-[12px] text-ink break-words whitespace-pre-wrap">
        {message}
      </p>
      {hint && (
        <p className="mt-1.5 text-[11px] text-ink-dim leading-relaxed break-words">
          {hint}
        </p>
      )}
    </div>
  );
}

export function Loading({ label = "loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 px-1 py-3 font-mono text-[11px] text-ink-dim">
      <Spinner />
      <span>{label}…</span>
    </div>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={`inline-block h-3 w-3 rounded-full border border-ink-faint border-t-accent animate-spin ${className}`}
    />
  );
}

export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="border border-dashed border-line rounded-sm px-4 py-6 text-center">
      <p className="font-mono text-[12px] text-ink-dim">{title}</p>
      {children && (
        <div className="mt-2 text-[11px] text-ink-faint leading-relaxed">
          {children}
        </div>
      )}
    </div>
  );
}

// ---- controls ---------------------------------------------------------------

const CONTROL =
  "bg-panel-2 border border-line rounded-sm px-2 py-1 font-mono text-[12px] text-ink " +
  "placeholder:text-ink-faint hover:border-line-strong focus:border-accent " +
  "disabled:opacity-40 disabled:cursor-not-allowed";

export function Field({
  label,
  hint,
  children,
  className = "",
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={`block min-w-0 ${className}`}>
      <span className="block font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint mb-1">
        {label}
      </span>
      {children}
      {hint && (
        <span className="block mt-1 text-[10px] text-ink-faint leading-snug">
          {hint}
        </span>
      )}
    </label>
  );
}

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  const { className = "", ...rest } = props;
  return <input {...rest} className={`${CONTROL} w-full ${className}`} />;
}

export function NumberInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  const { className = "", ...rest } = props;
  return (
    <input
      type="number"
      {...rest}
      className={`${CONTROL} w-full text-right ${className}`}
    />
  );
}

export function TextArea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const { className = "", ...rest } = props;
  return (
    <textarea {...rest} className={`${CONTROL} w-full resize-y ${className}`} />
  );
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  className = "",
  disabled,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly T[] | readonly { value: T; label: string }[];
  className?: string;
  disabled?: boolean;
}) {
  const opts = options.map((o) =>
    typeof o === "string" ? { value: o, label: o } : o,
  ) as { value: T; label: string }[];
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as T)}
      className={`${CONTROL} w-full ${className}`}
    >
      {opts.map((o) => (
        <option key={o.value} value={o.value} className="bg-panel-2">
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Button({
  children,
  variant = "default",
  className = "",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "ghost";
}) {
  const styles =
    variant === "primary"
      ? "bg-accent-dim border-accent/60 text-accent hover:bg-accent/20"
      : variant === "ghost"
        ? "bg-transparent border-transparent text-ink-dim hover:text-ink hover:border-line"
        : "bg-panel-2 border-line text-ink hover:border-line-strong";
  return (
    <button
      type="button"
      {...rest}
      className={`border rounded-sm px-2.5 py-1 font-mono text-[11px] leading-5 whitespace-nowrap
        disabled:opacity-40 disabled:cursor-not-allowed ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label
      className={`inline-flex items-center gap-1.5 font-mono text-[11px] select-none ${
        disabled ? "opacity-40" : "cursor-pointer text-ink-dim hover:text-ink"
      }`}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3 w-3"
      />
      {label}
    </label>
  );
}

/** Segmented control — used for STRONG/EVENTUAL and for sort mode. */
export function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  disabled,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly { value: T; label: string; title?: string }[];
  disabled?: boolean;
}) {
  return (
    <div className="inline-flex border border-line rounded-sm overflow-hidden">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.title}
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={`px-2 py-1 font-mono text-[11px] leading-5 border-r border-line last:border-r-0
            disabled:opacity-40 disabled:cursor-not-allowed ${
              value === o.value
                ? "bg-accent-dim text-accent"
                : "bg-panel-2 text-ink-dim hover:text-ink"
            }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ---- data display -----------------------------------------------------------

export function Mono({
  children,
  className = "",
  title,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span className={`font-mono text-[12px] ${className}`} title={title}>
      {children}
    </span>
  );
}

export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-block font-mono text-[10px] leading-4 px-1.5 py-px rounded-sm border border-line-strong bg-panel-2 text-ink-dim">
      {children}
    </span>
  );
}

/** A big number with a label — the Stats dashboard's unit of composition. */
export function StatTile({
  label,
  value,
  hint,
  tone = "normal",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "normal" | "ok" | "warn" | "danger" | "accent";
}) {
  const toneClass = {
    normal: "text-ink",
    ok: "text-ok",
    warn: "text-warn",
    danger: "text-danger",
    accent: "text-accent",
  }[tone];
  return (
    <div className="border border-line bg-panel-2 rounded-sm px-3 py-2 min-w-0">
      <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint">
        {label}
      </div>
      <div
        className={`mt-1 font-mono text-[19px] leading-6 tnum truncate ${toneClass}`}
      >
        {value}
      </div>
      {hint && (
        <div className="mt-0.5 text-[10px] text-ink-faint leading-snug">
          {hint}
        </div>
      )}
    </div>
  );
}

export function KeyValue({
  entries,
}: {
  entries: { k: string; v: ReactNode; title?: string }[];
}) {
  return (
    <dl className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)] gap-x-4 gap-y-1">
      {entries.map((e, i) => (
        <div key={`${e.k}-${i}`} className="contents">
          <dt
            className="font-mono text-[11px] text-ink-faint truncate"
            title={e.title ?? e.k}
          >
            {e.k}
          </dt>
          <dd className="font-mono text-[12px] text-ink min-w-0 break-all">
            {e.v}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** Click-to-copy for ids and tokens, which are exactly the things you paste. */
export function CopyableId({
  value,
  display,
  className = "",
}: {
  value: string;
  display?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  if (!value) return <span className="text-ink-faint font-mono">—</span>;
  return (
    <button
      type="button"
      title={`${value} (click to copy)`}
      onClick={(e) => {
        e.stopPropagation();
        void navigator.clipboard?.writeText(value).then(
          () => {
            setCopied(true);
            setTimeout(() => setCopied(false), 900);
          },
          () => undefined,
        );
      }}
      className={`font-mono text-[11px] text-ink-dim hover:text-accent text-left ${className}`}
    >
      {copied ? "copied" : (display ?? value)}
    </button>
  );
}

export function TableShell({
  head,
  children,
  className = "",
}: {
  head: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full border-collapse text-[12px]">
        <thead className="bg-panel-2">
          <tr className="text-left">{head}</tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Th({
  children,
  className = "",
  title,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <th
      title={title}
      className={`font-mono text-[10px] uppercase tracking-[0.08em] font-normal
        text-ink-faint px-2 py-1.5 border-b border-line whitespace-nowrap ${className}`}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  className = "",
  title,
  colSpan,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  colSpan?: number;
}) {
  return (
    <td
      title={title}
      colSpan={colSpan}
      className={`px-2 py-1.5 border-b border-line/70 align-top ${className}`}
    >
      {children}
    </td>
  );
}
