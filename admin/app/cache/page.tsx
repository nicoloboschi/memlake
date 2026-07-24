import { FleetCacheView } from "@/components/FleetCacheView";

export const dynamic = "force-dynamic";

/** Every serve node's cache occupancy, read from the rollups (not a node-local RPC). */
export default function Page() {
  return <FleetCacheView />;
}
