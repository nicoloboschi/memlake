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
pub mod spill;
pub mod sstable;
pub mod streaming;
pub mod temporal;

pub use engine::{Engine, QueryConfig};
pub use fusion::{rrf, weighted_rrf, FusedHit, RankedArm, DEFAULT_RRF_K};
pub use gc::{
    gc, gc_traces, gc_with_min_age, GcOutcome, ObsGcOutcome, DEFAULT_MIN_AGE,
    DEFAULT_ROLLUP_STALE, DEFAULT_TRACE_RETENTION,
};
pub use generation::{read_fts_split, read_generation, write_generation, ClusterTagSummary, Generation, SsTablePair, TagSummary};
pub use indexer::{
    derive_links_for_write, flush, fold, index, minor_compact, DeriveStats, IndexOptions,
    IndexOutcome, COMPACT_FANOUT, DEFAULT_STREAMING_THRESHOLD_DOCS,
};
pub use query_node::{
    ArmDepths, ArmScore, ClusterLayout, QueryNode, RawHit, ScanCursor, UpdatedWindow,
};
pub use sstable::{
    EntityTable, PayloadTable, PkTable, RadjTable, RerankTable, SsTableBuilder, SsTableIndex,
    TimeTable,
};

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
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<tantivy::TantivyError> for Error {
    fn from(e: tantivy::TantivyError) -> Self {
        Error::Fts(e.to_string())
    }
}
