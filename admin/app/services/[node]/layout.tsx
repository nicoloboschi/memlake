import { ServiceNav } from "@/components/ServiceNav";

export default async function ServiceLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ node: string }>;
}) {
  const { node } = await params;
  const decoded = decodeURIComponent(node);
  return (
    <div className="flex flex-col min-h-full">
      <ServiceNav node={decoded} />
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}
