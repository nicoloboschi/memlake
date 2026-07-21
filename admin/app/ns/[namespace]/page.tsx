import { StatsView } from "@/components/StatsView";

export const dynamic = "force-dynamic";

export default async function Page({
  params,
}: {
  params: Promise<{ namespace: string }>;
}) {
  const { namespace } = await params;
  return <StatsView namespace={decodeURIComponent(namespace)} />;
}
