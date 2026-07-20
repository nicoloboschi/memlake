//! Query orchestration: arm fusion and the in-process engine.
//!
//! [`fusion`] holds the rank-based combination of the arms (RRF); [`engine`] wires the
//! three arms together behind one query call over a built generation.

pub mod engine;
pub mod fusion;

pub use engine::{Engine, QueryConfig};
pub use fusion::{rrf, weighted_rrf, FusedHit, RankedArm, DEFAULT_RRF_K};
