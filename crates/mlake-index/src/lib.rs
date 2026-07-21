//! Query orchestration: arm fusion and the in-process engine.
//!
//! [`fusion`] holds the rank-based combination of the arms (RRF); [`engine`] wires the
//! three arms together behind one query call over a built generation.

pub mod engine;
pub mod fusion;
pub mod gc;
pub mod generation;
pub mod indexer;
pub mod query_node;

pub use engine::{Engine, QueryConfig};
pub use fusion::{rrf, weighted_rrf, FusedHit, RankedArm, DEFAULT_RRF_K};
pub use gc::{gc, GcOutcome};
pub use generation::{read_fts_split, read_generation, write_generation, Generation, PkIndex};
pub use indexer::{index, IndexOptions, IndexOutcome};
pub use query_node::{Consistency, QueryNode};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("store: {0}")]
    Store(#[from] mlake_store::Error),
    #[error("wal: {0}")]
    Wal(#[from] mlake_wal::Error),
    #[error("core: {0}")]
    Core(#[from] mlake_core::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
