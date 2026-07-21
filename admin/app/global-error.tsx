"use client";

import "./globals.css";

/**
 * Root-level boundary. `error.tsx` cannot catch a throw in the root layout, so
 * this file supplies its own <html>/<body>. It is deliberately plain: it must
 * render even when everything else is broken.
 */
export default function GlobalError({
  error,
}: {
  error: Error & { digest?: string };
}) {
  return (
    <html lang="en">
      <body className="bg-bg text-ink p-4">
        <div className="max-w-3xl border border-danger/40 bg-danger/5 rounded-sm px-3 py-2.5">
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] px-1.5 py-0.5 rounded-sm border border-danger/50 text-danger">
            fatal render error
          </span>
          <p className="mt-2 font-mono text-[12px] break-words whitespace-pre-wrap">
            {error.message}
          </p>
          {error.digest && (
            <p className="mt-1 font-mono text-[11px] text-ink-faint">
              digest {error.digest}
            </p>
          )}
          <p className="mt-2 text-[11px] text-ink-dim">
            Reload the page. If this persists, check the `next dev` / `next
            start` console for the server-side stack.
          </p>
        </div>
      </body>
    </html>
  );
}
