import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "memlake admin",
  description: "Inspection console for a memlake namespace: stats, browse, query.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col bg-bg text-ink">
        <header className="sticky top-0 z-20 border-b border-line bg-panel">
          <div className="flex items-center gap-4 px-4 h-11">
            <Link
              href="/"
              className="font-mono text-[13px] tracking-tight text-ink hover:text-accent"
            >
              memlake<span className="text-ink-faint">/</span>admin
            </Link>
            <span className="text-ink-faint font-mono text-[11px]">
              S3-native retrieval engine
            </span>
          </div>
        </header>
        <main className="flex-1 min-w-0">{children}</main>
      </body>
    </html>
  );
}
