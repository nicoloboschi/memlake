//! The write path: buffer, group-commit, claim a sequence.
//!
//! Claiming works by conditional create. Every writer computes what it believes the next
//! free sequence is and tries to create that object; the loser re-reads the head and
//! tries again. Because the create is conditional, two writers can never occupy the same
//! sequence, so the log is totally ordered without any coordinator (SPEC §4).

use std::time::Duration;

use mlake_core::{Op, WalEntry};
use mlake_store::Error as StoreError;

use crate::{seq_path, Error, Namespace, Result};

/// Bounded retry for the claim loop. Contention is proportional to the number of writers
/// on a namespace, not to data size, so this only needs to absorb a burst.
const MAX_CLAIM_ATTEMPTS: usize = 32;

/// Outcome of a successful commit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CommitResult {
    /// The sequence this batch was durably written at. After this returns, the write is
    /// on S3 and visible to the next strongly-consistent query (INV-5).
    pub seq: u64,
    /// How many claim attempts it took. Surfaced so contention is observable.
    pub attempts: usize,
}

/// Commits batches of ops to a namespace's WAL.
pub struct Writer {
    namespace: Namespace,
    /// Cached view of the head, to skip a LIST on the common uncontended path. Only ever
    /// an optimization: a wrong value costs a conflict and a re-read, never correctness.
    next_seq: Option<u64>,
}

impl Writer {
    pub fn new(namespace: Namespace) -> Self {
        Self {
            namespace,
            next_seq: None,
        }
    }

    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Commit a batch of ops as one atomic entry.
    ///
    /// Everything in `ops` becomes visible to every reader at the same moment — this is
    /// the replacement for the reference implementation's per-document transaction.
    pub async fn commit(&mut self, ops: Vec<Op>) -> Result<CommitResult> {
        if ops.is_empty() {
            // Nothing to order; claiming a sequence for an empty entry would only add
            // work for every future tail scan.
            let head = self.namespace.wal_head().await?;
            return Ok(CommitResult {
                seq: head,
                attempts: 0,
            });
        }

        // A Guard is a precondition on the log's state, so it must be evaluated against
        // the head we are actually about to write past.
        let guard = ops.iter().find_map(|op| match op {
            Op::Guard { expect_seq_lt } => Some(*expect_seq_lt),
            _ => None,
        });

        let mut seq = match self.next_seq {
            Some(s) => s,
            None => self.namespace.wal_head().await? + 1,
        };

        for attempt in 1..=MAX_CLAIM_ATTEMPTS {
            if let Some(expected) = guard {
                if seq >= expected {
                    self.next_seq = None;
                    return Err(Error::GuardFailed {
                        expected,
                        actual: seq,
                    });
                }
            }

            let entry = WalEntry::new(seq, ops.clone());
            let path = seq_path(&self.namespace.name, seq);
            match self
                .namespace
                .store
                .put_if_absent(&path, entry.to_bytes()?)
                .await
            {
                Ok(_) => {
                    self.next_seq = Some(seq + 1);
                    return Ok(CommitResult { seq, attempts: attempt });
                }
                Err(StoreError::AlreadyExists(_)) => {
                    // Someone else took this slot. Re-read the head rather than simply
                    // incrementing, so a burst of writers converges instead of each
                    // walking the whole contended range one slot at a time.
                    let head = self.namespace.wal_head().await?;
                    seq = head + 1;
                    // Jittered backoff, scaled by attempt, to spread out a thundering herd.
                    let backoff = Duration::from_millis((attempt as u64).min(10) * 2)
                        + Duration::from_micros((seq % 997) * 10);
                    tokio::time::sleep(backoff).await;
                }
                Err(e) => return Err(e.into()),
            }
        }

        self.next_seq = None;
        Err(Error::CommitRetriesExhausted(MAX_CLAIM_ATTEMPTS))
    }

    /// Forget the cached head. Used after an error of unknown effect, so the next commit
    /// re-derives the head from storage.
    pub fn reset(&mut self) {
        self.next_seq = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlake_core::memory::Timestamps;
    use mlake_core::{Memory, MemoryId};
    use mlake_store::Store;
    use std::sync::Arc;

    fn item(key: &str) -> Memory {
        Memory {
            id: MemoryId::from_key(key),
            vector: vec![0.1, 0.2],
            text: key.to_string(),
            memory_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![],
            causal_out: vec![],
            metadata: vec![],
        }
    }

    async fn namespace() -> Namespace {
        let ns = Namespace::new("ns", Store::in_memory());
        ns.create_if_absent("tok").await.unwrap();
        ns
    }

    #[tokio::test]
    async fn commits_advance_the_sequence_from_one() {
        let mut w = Writer::new(namespace().await);
        assert_eq!(w.commit(vec![Op::Upsert(item("a"))]).await.unwrap().seq, 1);
        assert_eq!(w.commit(vec![Op::Upsert(item("b"))]).await.unwrap().seq, 2);
        assert_eq!(w.commit(vec![Op::Upsert(item("c"))]).await.unwrap().seq, 3);
    }

    #[tokio::test]
    async fn an_empty_batch_does_not_consume_a_sequence() {
        let mut w = Writer::new(namespace().await);
        w.commit(vec![Op::Upsert(item("a"))]).await.unwrap();
        let empty = w.commit(vec![]).await.unwrap();
        assert_eq!(empty.attempts, 0);
        assert_eq!(w.namespace().wal_head().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn a_stale_cached_head_costs_a_retry_not_correctness() {
        let ns = namespace().await;
        let mut w = Writer::new(ns.clone());
        w.commit(vec![Op::Upsert(item("a"))]).await.unwrap();

        // Another node commits behind this writer's back, invalidating its cache.
        let mut other = Writer::new(ns.clone());
        other.commit(vec![Op::Upsert(item("b"))]).await.unwrap();

        let result = w.commit(vec![Op::Upsert(item("c"))]).await.unwrap();
        assert_eq!(result.seq, 3, "must land past the other writer's entry");
        assert!(result.attempts > 1, "should have observed the conflict");
    }

    #[tokio::test]
    async fn concurrent_writers_get_distinct_sequences() {
        let ns = Arc::new(namespace().await);
        let mut handles = Vec::new();
        for i in 0..8 {
            let ns = Arc::clone(&ns);
            handles.push(tokio::spawn(async move {
                let mut w = Writer::new((*ns).clone());
                w.commit(vec![Op::Upsert(item(&format!("item-{i}")))])
                    .await
                    .unwrap()
                    .seq
            }));
        }
        let mut seqs = Vec::new();
        for h in handles {
            seqs.push(h.await.unwrap());
        }
        seqs.sort();
        // Every writer got its own slot, and the log has no holes.
        assert_eq!(seqs, (1..=8).collect::<Vec<u64>>());
    }

    #[tokio::test]
    async fn guard_rejects_a_batch_when_the_log_moved_past_it() {
        let ns = namespace().await;
        let mut w = Writer::new(ns.clone());
        w.commit(vec![Op::Upsert(item("a"))]).await.unwrap();
        w.commit(vec![Op::Upsert(item("b"))]).await.unwrap();

        // Requires the log to still be shorter than 2; it is at 2, so this must fail
        // rather than silently apply.
        let guarded = w
            .commit(vec![
                Op::Guard { expect_seq_lt: 2 },
                Op::Upsert(item("c")),
            ])
            .await;
        assert!(matches!(guarded, Err(Error::GuardFailed { .. })), "{guarded:?}");
    }

    #[tokio::test]
    async fn guard_admits_a_batch_when_the_precondition_holds() {
        let mut w = Writer::new(namespace().await);
        let ok = w
            .commit(vec![
                Op::Guard { expect_seq_lt: 10 },
                Op::Upsert(item("a")),
            ])
            .await
            .unwrap();
        assert_eq!(ok.seq, 1);
    }
}
