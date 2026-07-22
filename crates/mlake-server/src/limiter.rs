//! Admission control for the retrieval hot paths.
//!
//! A query's peak working memory is roughly `nprobe × cluster_size` (the clusters it fetches and
//! reranks). Nothing caps how many queries run at once, so aggregate server memory is
//! `in_flight × per_query` — unbounded in request concurrency. [`QueryLimiter`] caps `in_flight`
//! with a semaphore, turning peak memory into `permits × per_query`, a number an operator can size
//! against a pod's limit. Excess requests **await** a permit (natural backpressure) rather than
//! being rejected, so a burst slows down instead of OOM-ing or erroring.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Concurrent-holder counters, shared (behind an `Arc`) between the limiter and each live permit
/// so a permit can decrement `in_flight` on drop without borrowing the limiter — which lets the
/// guard be owned + `'static` + `Send`, safe to hold across the `.await`s of an async trait method.
#[derive(Default)]
struct Counts {
    in_flight: AtomicUsize,
    /// High-water mark of concurrent holders — surfaced for tests and observability.
    max_in_flight: AtomicUsize,
}

/// Bounds the number of concurrently-executing retrieval requests.
pub struct QueryLimiter {
    sem: Arc<Semaphore>,
    permits: usize,
    counts: Arc<Counts>,
}

impl QueryLimiter {
    /// A limiter admitting at most `permits` concurrent requests (floored at 1, so a
    /// misconfigured `0` never deadlocks every read).
    pub fn new(permits: usize) -> Self {
        let permits = permits.max(1);
        Self {
            sem: Arc::new(Semaphore::new(permits)),
            permits,
            counts: Arc::new(Counts::default()),
        }
    }

    /// Await a permit, then hold it for the returned guard's lifetime. When all permits are out,
    /// this suspends until one frees — the caller queues rather than piling on more work. The
    /// guard owns its permit (no borrow of the limiter), so it is safe to carry across awaits.
    pub async fn acquire(&self) -> QueryPermit {
        // acquire_owned only errors if the semaphore was closed; we never close it.
        let permit = Arc::clone(&self.sem)
            .acquire_owned()
            .await
            .expect("query limiter semaphore closed");
        let now = self.counts.in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.counts.max_in_flight.fetch_max(now, Ordering::AcqRel);
        QueryPermit { _permit: permit, counts: Arc::clone(&self.counts) }
    }

    /// Configured permit count.
    pub fn permits(&self) -> usize {
        self.permits
    }

    /// Requests currently holding a permit. Reserved for observability; exercised by the tests.
    #[allow(dead_code)]
    pub fn in_flight(&self) -> usize {
        self.counts.in_flight.load(Ordering::Acquire)
    }

    /// Highest concurrent holder count observed so far. Reserved for observability; exercised by
    /// the tests.
    #[allow(dead_code)]
    pub fn max_in_flight(&self) -> usize {
        self.counts.max_in_flight.load(Ordering::Acquire)
    }
}

/// Held for the duration of an admitted request; releases the permit and decrements the in-flight
/// count on drop. Owned + `'static`, so it can be held across a handler's awaits.
pub struct QueryPermit {
    _permit: OwnedSemaphorePermit,
    counts: Arc<Counts>,
}

impl Drop for QueryPermit {
    fn drop(&mut self) {
        self.counts.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Barrier;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn never_admits_more_than_permits_concurrently() {
        const PERMITS: usize = 3;
        /// A whole number of cohorts. The barrier below is `PERMITS` wide and only releases
        /// when that many holders reach it, so a remainder cohort waits for a task that can
        /// never arrive — at 20 the last two hang forever and the suite never finishes.
        const TASKS: usize = 21;
        let limiter = Arc::new(QueryLimiter::new(PERMITS));
        // A barrier the size of the permit count: each cohort of `PERMITS` holders must all reach
        // the barrier before any releases, forcing genuine concurrency up to the cap (and proving
        // the cap is actually reached, not just respected).
        let barrier = Arc::new(Barrier::new(PERMITS));

        let mut handles = Vec::new();
        for _ in 0..TASKS {
            let limiter = Arc::clone(&limiter);
            let barrier = Arc::clone(&barrier);
            handles.push(tokio::spawn(async move {
                let _p = limiter.acquire().await;
                assert!(limiter.in_flight() <= PERMITS, "over-admitted: {}", limiter.in_flight());
                barrier.wait().await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert!(limiter.in_flight() == 0, "all permits returned");
        assert_eq!(limiter.max_in_flight(), PERMITS, "the cap is reached, and never exceeded");
    }

    #[tokio::test]
    async fn zero_permits_is_floored_to_one() {
        let limiter = QueryLimiter::new(0);
        assert_eq!(limiter.permits(), 1);
        let _p = limiter.acquire().await; // must not deadlock
        assert_eq!(limiter.in_flight(), 1);
    }
}
