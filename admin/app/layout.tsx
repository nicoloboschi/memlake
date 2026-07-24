import type { Metadata } from "next";

import { Sidebar } from "@/components/Sidebar";
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
      <body className="h-full flex bg-bg text-ink">
        <Sidebar />
        <main className="flex-1 min-w-0 h-screen overflow-y-auto">{children}</main>
      </body>
    </html>
  );
}
