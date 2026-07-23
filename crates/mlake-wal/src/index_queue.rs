//! An object-storage work queue for the indexer — turbopuffer's `queue.json` pattern.
//!
//! A single CAS'd JSON object, [`QUEUE_PATH`], holds one job per namespace that *might* need
//! folding. It replaces the indexer's old poll-every-namespace loop: instead of LISTing every
//! manifest each tick and checking each for un-indexed WAL, a serve pod enqueues a namespace right
//! after it commits a write, and indexers only ever touch namespaces that have a job.
//!
//! Mechanics (all over object storage, no broker):
//! * **enqueue** — read the queue, add the namespace as `pending` if absent, CAS the file back.
//!   Idempotent: a namespace already queued (or being folded) is left alone.
//! * **claim** — read the queue, find a `pending` job (or a `claimed` one whose heartbeat went
//!   stale — its worker crashed), flip it to `claimed` with our id + a fresh heartbeat, CAS back.
//!   The CAS makes the claim exclusive: two indexers can never fold the same namespace at once.
//! * **heartbeat** — while folding, periodically CAS a new timestamp so peers don't reclaim us.
//! * **complete** — when the namespace is drained (WAL head folded), remove the job; if more WAL
//!   arrived meanwhile, return it to `pending` instead. Combined with enqueue-after-commit, no
//!   write is ever lost — the job either still exists (re-checked here) or a later enqueue re-adds
//!   it. This is at-least-once delivery over memlake's idempotent folds (INV-6).
//!
//! Correctness never depends on the queue: it is a notification/hint layer. A reconciliation sweep
//! (the indexer, rarely) can still enqueue any dirty namespace directly, covering a serve pod that
//! died between commit and enqueue.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use mlake_store::{Error as StoreError, Etag, Store};
use serde::{Deserialize, Serialize};

use crate::Result;

/// Bucket-root object holding the whole queue. Not under any namespace prefix, so
/// `discover_namespaces` (which matches `*/manifest.json`) never mistakes it for a bank.
pub const QUEUE_PATH: &str = "_index-queue.json";

/// Bounded retry for a read-modify-CAS cycle. Contention scales with the number of concurrently
/// enqueueing/claiming nodes, not with data, so a small burst-absorber is enough.
const MAX_CAS_ATTEMPTS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Unclaimed — any indexer may claim it.
    Pending,
    /// An indexer is folding it; `claimed_by` + `heartbeat_ms` prove liveness.
    Claimed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub state: JobState,
    #[serde(default)]
    pub claimed_by: Option<String>,
    /// Epoch ms of the last claim/heartbeat. A `claimed` job older than the stale window is
    /// reclaimable — its worker is presumed dead.
    #[serde(default)]
    pub heartbeat_ms: u64,
    #[serde(default)]
    pub enqueued_ms: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueueState {
    #[serde(default)]
    pub jobs: BTreeMap<String, Job>,
}

impl QueueState {
    pub fn pending(&self) -> impl Iterator<Item = &String> {
        self.jobs.iter().filter(|(_, j)| j.state == JobState::Pending).map(|(n, _)| n)
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The object-storage index queue, bound to a [`Store`].
#[derive(Clone)]
pub struct IndexQueue {
    store: Store,
}

impl IndexQueue {
    pub fn new(store: Store) -> Self {
        Self { store }
    }

    async fn read(&self) -> Result<(QueueState, Option<Etag>)> {
        match self.store.get(QUEUE_PATH, None).await {
            Ok(v) => {
                // A malformed queue file is treated as empty rather than wedging the indexer; the
                // reconciliation sweep re-enqueues real work.
                let state = serde_json::from_slice(&v.bytes).unwrap_or_default();
                Ok((state, v.etag))
            }
            Err(StoreError::NotFound(_)) => Ok((QueueState::default(), None)),
            Err(e) => Err(e.into()),
        }
    }

    /// CAS the queue back. `etag == None` means "create" (the queue did not exist when read).
    async fn write(&self, state: &QueueState, etag: Option<&Etag>) -> Result<()> {
        let bytes = serde_json::to_vec(state)?;
        match etag {
            Some(e) => {
                self.store.cas_swap(QUEUE_PATH, e, bytes).await?;
            }
            None => {
                self.store.put_if_absent(QUEUE_PATH, bytes).await?;
            }
        }
        Ok(())
    }

    /// Add `namespace` as a pending job if absent. Idempotent and safe to call after every write —
    /// a namespace already queued or being folded is left as-is (the fold re-checks the WAL head at
    /// completion). Returns quietly if it loses too many CAS races; the sweep is the backstop.
    pub async fn enqueue(&self, namespace: &str) -> Result<()> {
        for _ in 0..MAX_CAS_ATTEMPTS {
            let (mut state, etag) = self.read().await?;
            if state.jobs.contains_key(namespace) {
                return Ok(());
            }
            state.jobs.insert(
                namespace.to_string(),
                Job { state: JobState::Pending, claimed_by: None, heartbeat_ms: 0, enqueued_ms: now_ms() },
            );
            match self.write(&state, etag.as_ref()).await {
                Ok(()) => return Ok(()),
                Err(e) if e.is_conflict() => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Claim a job for `worker_id`: the first `pending` job, or a `claimed` one whose heartbeat is
    /// older than `stale_after_ms` (its worker crashed). Returns the namespace, or `None` when
    /// there is nothing to do. Exclusive by CAS — a lost race just retries against fresh state.
    pub async fn claim(&self, worker_id: &str, stale_after_ms: u64) -> Result<Option<String>> {
        for _ in 0..MAX_CAS_ATTEMPTS {
            let (mut state, etag) = self.read().await?;
            let now = now_ms();
            let pick = state
                .jobs
                .iter()
                .find(|(_, j)| {
                    j.state == JobState::Pending
                        || now.saturating_sub(j.heartbeat_ms) > stale_after_ms
                })
                .map(|(n, _)| n.clone());
            let Some(ns) = pick else { return Ok(None) };
            if let Some(job) = state.jobs.get_mut(&ns) {
                job.state = JobState::Claimed;
                job.claimed_by = Some(worker_id.to_string());
                job.heartbeat_ms = now;
            }
            match self.write(&state, etag.as_ref()).await {
                Ok(()) => return Ok(Some(ns)),
                Err(e) if e.is_conflict() => continue, // someone claimed first; re-read
                Err(e) => return Err(e),
            }
        }
        Ok(None)
    }

    /// Refresh the heartbeat on a job we hold. Returns `false` if the job was reclaimed by a peer
    /// or removed — the caller should stop, since another worker may now own it.
    pub async fn heartbeat(&self, namespace: &str, worker_id: &str) -> Result<bool> {
        for _ in 0..MAX_CAS_ATTEMPTS {
            let (mut state, etag) = self.read().await?;
            match state.jobs.get_mut(namespace) {
                Some(job) if job.claimed_by.as_deref() == Some(worker_id) => {
                    job.heartbeat_ms = now_ms();
                }
                _ => return Ok(false), // no longer ours
            }
            match self.write(&state, etag.as_ref()).await {
                Ok(()) => return Ok(true),
                Err(e) if e.is_conflict() => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(false)
    }

    /// Finish a job we hold. `still_dirty` = the namespace still has un-indexed WAL (a write landed
    /// during the fold): keep it as `pending` for another pass instead of removing it. Only removes
    /// when the namespace is fully drained, so no write slips through the completion window.
    pub async fn complete(&self, namespace: &str, worker_id: &str, still_dirty: bool) -> Result<()> {
        for _ in 0..MAX_CAS_ATTEMPTS {
            let (mut state, etag) = self.read().await?;
            match state.jobs.get(namespace) {
                Some(job) if job.claimed_by.as_deref() == Some(worker_id) => {}
                _ => return Ok(()), // reclaimed/removed already; nothing to do
            }
            if still_dirty {
                let job = state.jobs.get_mut(namespace).unwrap();
                job.state = JobState::Pending;
                job.claimed_by = None;
                job.heartbeat_ms = 0;
            } else {
                state.jobs.remove(namespace);
            }
            match self.write(&state, etag.as_ref()).await {
                Ok(()) => return Ok(()),
                Err(e) if e.is_conflict() => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// The current queue snapshot — for the reconciliation sweep and observability.
    pub async fn snapshot(&self) -> Result<QueueState> {
        Ok(self.read().await?.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlake_store::Store;

    fn q() -> IndexQueue {
        IndexQueue::new(Store::in_memory())
    }

    #[tokio::test]
    async fn enqueue_is_idempotent() {
        let q = q();
        q.enqueue("ns-a").await.unwrap();
        q.enqueue("ns-a").await.unwrap();
        q.enqueue("ns-b").await.unwrap();
        let s = q.snapshot().await.unwrap();
        assert_eq!(s.jobs.len(), 2, "duplicate enqueue does not add a second job");
        assert_eq!(s.pending().count(), 2);
    }

    #[tokio::test]
    async fn claim_is_exclusive_and_drains_the_pending_set() {
        let q = q();
        q.enqueue("ns-a").await.unwrap();
        q.enqueue("ns-b").await.unwrap();

        let a = q.claim("w1", 60_000).await.unwrap();
        let b = q.claim("w2", 60_000).await.unwrap();
        assert!(a.is_some() && b.is_some(), "both pending jobs get claimed");
        assert_ne!(a, b, "two workers never claim the same namespace");

        // Nothing pending and neither claim is stale → no more work.
        assert_eq!(q.claim("w3", 60_000).await.unwrap(), None);
    }

    #[tokio::test]
    async fn enqueue_skips_a_namespace_being_folded() {
        let q = q();
        q.enqueue("ns-a").await.unwrap();
        let who = q.claim("w1", 60_000).await.unwrap().unwrap();
        // A write lands during the fold: enqueue must not create a duplicate; the completion
        // re-check is what catches the new WAL.
        q.enqueue(&who).await.unwrap();
        let s = q.snapshot().await.unwrap();
        assert_eq!(s.jobs.len(), 1);
        assert_eq!(s.jobs[&who].state, JobState::Claimed);
    }

    #[tokio::test]
    async fn complete_removes_when_drained_and_requeues_when_dirty() {
        let q = q();
        q.enqueue("ns-a").await.unwrap();
        let ns = q.claim("w1", 60_000).await.unwrap().unwrap();

        // Still dirty → back to pending, reclaimable.
        q.complete(&ns, "w1", true).await.unwrap();
        assert_eq!(q.snapshot().await.unwrap().jobs[&ns].state, JobState::Pending);
        let ns2 = q.claim("w2", 60_000).await.unwrap().unwrap();
        assert_eq!(ns2, ns);

        // Drained → removed.
        q.complete(&ns, "w2", false).await.unwrap();
        assert!(q.snapshot().await.unwrap().jobs.is_empty());
    }

    #[tokio::test]
    async fn a_stale_claim_is_reclaimable_but_a_fresh_one_is_not() {
        let q = q();
        q.enqueue("ns-a").await.unwrap();
        let _ = q.claim("dead-worker", 60_000).await.unwrap().unwrap();

        // Fresh heartbeat: not reclaimable under a long stale window.
        assert_eq!(q.claim("w2", 60_000).await.unwrap(), None);

        // Let the heartbeat age past a tiny window, then it is reclaimable.
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        let reclaimed = q.claim("w2", 5).await.unwrap();
        assert_eq!(reclaimed.as_deref(), Some("ns-a"));
        assert_eq!(
            q.snapshot().await.unwrap().jobs["ns-a"].claimed_by.as_deref(),
            Some("w2"),
            "the live worker takes ownership",
        );
    }

    #[tokio::test]
    async fn heartbeat_fails_once_reclaimed() {
        let q = q();
        q.enqueue("ns-a").await.unwrap();
        q.claim("w1", 60_000).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        q.claim("w2", 5).await.unwrap(); // w2 steals it
        assert!(!q.heartbeat("ns-a", "w1").await.unwrap(), "the evicted worker's heartbeat is rejected");
        assert!(q.heartbeat("ns-a", "w2").await.unwrap(), "the new owner's heartbeat works");
    }
}
