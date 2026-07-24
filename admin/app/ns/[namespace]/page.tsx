import { NamespaceStatsView } from "@/components/NamespaceStatsView";

export const dynamic = "force-dynamic";

export default async function Page({ params }: { params: Promise<{ namespace: string }> }) {
  const { namespace } = await params;
  return <NamespaceStatsView namespace={decodeURIComponent(namespace)} />;
}
