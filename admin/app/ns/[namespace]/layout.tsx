import { NamespaceNav } from "@/components/NamespaceNav";

export default async function NamespaceLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ namespace: string }>;
}) {
  const { namespace } = await params;
  const decoded = decodeURIComponent(namespace);
  return (
    <div className="flex flex-col min-h-full">
      <NamespaceNav namespace={decoded} />
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}
