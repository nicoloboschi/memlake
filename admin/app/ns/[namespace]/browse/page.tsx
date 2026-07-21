import { BrowseView } from "@/components/BrowseView";

export const dynamic = "force-dynamic";

export default async function Page({
  params,
}: {
  params: Promise<{ namespace: string }>;
}) {
  const { namespace } = await params;
  return <BrowseView namespace={decodeURIComponent(namespace)} />;
}
