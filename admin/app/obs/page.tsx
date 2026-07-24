import { ObservabilityView } from "@/components/ObservabilityView";

export const dynamic = "force-dynamic";

/** Fleet-wide tracing: reads `_obs/traces/` from the bucket directly. */
export default function Page() {
  return <ObservabilityView />;
}
