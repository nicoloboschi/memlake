import { ServiceDetail } from "@/components/ServiceDetail";

export const dynamic = "force-dynamic";

/** One serve node's read cache — published in its rollup, so no node-local RPC is needed. */
export default async function Page({ params }: { params: Promise<{ node: string }> }) {
  const { node } = await params;
  return <ServiceDetail node={decodeURIComponent(node)} tab="cache" />;
}
