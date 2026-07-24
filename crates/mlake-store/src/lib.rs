//! Object storage for memlake: the only stateful dependency in the critical path (INV-1).
//!
//! Three concerns live here:
//!
//! * [`Store`] — an instrumented client whose only coordination primitive is the
//!   conditional write (INV-3).
//! * [`cache`] — a content-addressed local cache keyed by `(namespace, path, etag)`.
//!   Because the key includes the etag and every non-manifest object is immutable, a
//!   cache hit is always correct, and wiping the cache can only cost latency (INV-4).
//! * [`latency`] — a latency-injecting wrapper so CI can measure roundtrip behaviour
//!   against realistic S3 timings without touching a network.

pub mod cache;
pub mod latency;
pub mod metrics;
pub mod spans;
pub mod store;

pub use cache::{CacheEntry, CacheKey, DiskCache, EvictionPolicy};
pub use metrics::{Phase, QueryMetrics, StoreMetrics, StoreSnapshot, COLD_ROUNDTRIP_BUDGET};
pub use store::{Etag, S3Config, Store, Versioned};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("object not found: {0}")]
    NotFound(String),
    /// A conditional create lost the race — the slot was claimed by someone else.
    #[error("object already exists: {0}")]
    AlreadyExists(String),
    /// A conditional update lost the race — the object moved on since it was read.
    #[error("CAS conflict on {0}")]
    CasConflict(String),
    #[error("object store: {0}")]
    ObjectStore(#[from] object_store::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("core: {0}")]
    Core(#[from] mlake_core::Error),
    #[error("config: {0}")]
    Config(String),
}

impl Error {
    /// True when the operation failed because another writer got there first, and the
    /// caller should re-read and retry rather than treat this as a hard failure.
    pub fn is_conflict(&self) -> bool {
        matches!(self, Error::AlreadyExists(_) | Error::CasConflict(_))
    }
}
