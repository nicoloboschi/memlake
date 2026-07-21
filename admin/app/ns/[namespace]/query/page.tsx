import { QueryView } from "@/components/QueryView";

export const dynamic = "force-dynamic";

export default async function Page({
  params,
}: {
  params: Promise<{ namespace: string }>;
}) {
  const { namespace } = await params;
  return <QueryView namespace={decodeURIComponent(namespace)} />;
}
