"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { slug: "", label: "stats" },
  { slug: "cache", label: "cache" },
] as const;

export function ServiceNav({ node }: { node: string }) {
  const pathname = usePathname();
  const base = `/services/${encodeURIComponent(node)}`;

  return (
    <div className="border-b border-line bg-panel">
      <div className="flex items-center gap-4 px-4 h-9">
        <Link
          href="/services"
          className="font-mono text-[11px] text-ink-faint hover:text-accent shrink-0"
        >
          ← services
        </Link>
        <span className="font-mono text-[12px] text-ink truncate min-w-0">{node}</span>
        <nav className="flex items-center gap-px ml-2">
          {TABS.map((t) => {
            const href = t.slug ? `${base}/${t.slug}` : base;
            const active = t.slug ? pathname === href : pathname === base;
            return (
              <Link
                key={t.label}
                href={href}
                className={`px-2.5 py-1 font-mono text-[11px] rounded-sm ${
                  active
                    ? "bg-accent-dim text-accent"
                    : "text-ink-dim hover:text-ink hover:bg-panel-2"
                }`}
              >
                {t.label}
              </Link>
            );
          })}
        </nav>
        <Link
          href={`/traces?node=${encodeURIComponent(node)}`}
          className="ml-auto font-mono text-[11px] text-ink-dim hover:text-accent shrink-0"
        >
          traces →
        </Link>
      </div>
    </div>
  );
}
