import { Suspense } from "react";

import { TracesView } from "@/components/TracesView";

export const dynamic = "force-dynamic";

/** The trace explorer: filter by namespace/service, open one full-width. */
export default function Page() {
  return (
    <Suspense>
      <TracesView />
    </Suspense>
  );
}
