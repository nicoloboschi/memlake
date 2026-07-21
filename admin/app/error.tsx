"use client";

/**
 * Last-resort boundary for the page tree. Every data fetch already renders its
 * own inline error state, so reaching this means something threw during render
 * — it exists so an operator still sees the message instead of a blank screen.
 */
export default function Error({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  return (
    <div className="p-4 max-w-3xl">
      <div className="border border-danger/40 bg-danger/5 rounded-sm px-3 py-2.5">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] px-1.5 py-0.5 rounded-sm border border-danger/50 text-danger">
            render error
          </span>
          <button
            type="button"
            onClick={() => unstable_retry()}
            className="ml-auto font-mono text-[11px] text-accent hover:underline"
          >
            retry
          </button>
        </div>
        <p className="mt-1.5 font-mono text-[12px] text-ink break-words whitespace-pre-wrap">
          {error.message}
        </p>
        {error.digest && (
          <p className="mt-1 font-mono text-[11px] text-ink-faint">
            digest {error.digest}
          </p>
        )}
      </div>
    </div>
  );
}
