"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** Top-level sections. `match` decides the active state from the current path. */
const SECTIONS = [
  {
    href: "/services",
    label: "Services",
    hint: "serve fleet · health",
    match: (p: string) => p.startsWith("/services") || p.startsWith("/cache"),
  },
  {
    href: "/traces",
    label: "Traces",
    hint: "request timelines",
    match: (p: string) => p.startsWith("/traces"),
  },
  {
    href: "/",
    label: "Namespaces",
    hint: "data · indexes · WAL",
    match: (p: string) => p === "/" || p.startsWith("/ns"),
  },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-52 shrink-0 h-screen sticky top-0 border-r border-line bg-panel flex flex-col">
      <div className="px-4 h-11 flex items-center border-b border-line">
        <Link
          href="/"
          className="font-mono text-[13px] tracking-tight text-ink hover:text-accent"
        >
          memlake<span className="text-ink-faint">/</span>admin
        </Link>
      </div>

      <nav className="flex-1 p-2 flex flex-col gap-0.5">
        {SECTIONS.map((s) => {
          const active = s.match(pathname);
          return (
            <Link
              key={s.href}
              href={s.href}
              className={`px-3 py-2 rounded-sm block ${
                active
                  ? "bg-accent-dim text-accent"
                  : "text-ink-dim hover:text-ink hover:bg-panel-2"
              }`}
            >
              <div className="font-mono text-[12px] leading-4">{s.label}</div>
              <div className="text-[10px] text-ink-faint leading-4">{s.hint}</div>
            </Link>
          );
        })}

        {/* Node-local read cache — a serve-node internal, so it nests under Services. */}
        <Link
          href="/cache"
          className={`ml-3 mt-0.5 px-3 py-1 rounded-sm font-mono text-[11px] ${
            pathname.startsWith("/cache")
              ? "text-accent"
              : "text-ink-faint hover:text-ink"
          }`}
        >
          node cache
        </Link>
      </nav>

      <div className="p-3 border-t border-line text-[10px] text-ink-faint leading-snug">
        S3-native retrieval engine
      </div>
    </aside>
  );
}
