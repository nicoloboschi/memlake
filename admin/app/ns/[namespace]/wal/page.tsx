import { WalView } from "@/components/WalView";

export const dynamic = "force-dynamic";

export default async function Page({
  params,
}: {
  params: Promise<{ namespace: string }>;
}) {
  const { namespace } = await params;
  return <WalView namespace={decodeURIComponent(namespace)} />;
}
