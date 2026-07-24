import { ServiceDetail } from "@/components/ServiceDetail";

export const dynamic = "force-dynamic";

/** One serve node's stats, from its rollup. */
export default async function Page({ params }: { params: Promise<{ node: string }> }) {
  const { node } = await params;
  return <ServiceDetail node={decodeURIComponent(node)} tab="stats" />;
}
