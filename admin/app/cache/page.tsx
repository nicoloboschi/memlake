import { CacheView } from "@/components/CacheView";

export const dynamic = "force-dynamic";

/** The unfiltered view: every namespace this node happens to be holding. */
export default function Page() {
  return <CacheView namespace={null} />;
}
