import { IndexerView } from "@/components/IndexerView";

export const dynamic = "force-dynamic";

/** Indexer work queue, read from `_index-queue.json` in the bucket. */
export default function Page() {
  return <IndexerView />;
}
