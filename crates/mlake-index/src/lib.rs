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
pub mod sstable;
pub mod temporal;

pub use engine::{Engine, QueryConfig};
pub use fusion::{rrf, weighted_rrf, FusedHit, RankedArm, DEFAULT_RRF_K};
pub use gc::{gc, gc_with_min_age, GcOutcome, DEFAULT_MIN_AGE};
pub use generation::{read_fts_split, read_generation, write_generation, ClusterTagSummary, Generation, SsTablePair, TagSummary};
pub use indexer::{index, IndexOptions, IndexOutcome};
pub use query_node::{ArmDepths, ArmScore, ClusterLayout, QueryNode, RawHit, ScanCursor};
pub use sstable::{EntityTable, PkTable, RadjTable, SsTableBuilder, SsTableIndex, TimeTable};

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
    #[error("fts: {0}")]
    Fts(String),
}

impl From<tantivy::TantivyError> for Error {
    fn from(e: tantivy::TantivyError) -> Self {
        Error::Fts(e.to_string())
    }
}
