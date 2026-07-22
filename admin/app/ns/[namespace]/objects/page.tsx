import { ObjectsView } from "@/components/ObjectsView";

export const dynamic = "force-dynamic";

export default async function Page({
  params,
}: {
  params: Promise<{ namespace: string }>;
}) {
  const { namespace } = await params;
  return <ObjectsView namespace={decodeURIComponent(namespace)} />;
}
