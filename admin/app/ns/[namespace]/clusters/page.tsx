import { ClustersView } from "@/components/ClustersView";

export const dynamic = "force-dynamic";

export default async function Page({
  params,
}: {
  params: Promise<{ namespace: string }>;
}) {
  const { namespace } = await params;
  return <ClustersView namespace={decodeURIComponent(namespace)} />;
}
