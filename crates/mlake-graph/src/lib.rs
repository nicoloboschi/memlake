//! Graph retrieval: bounded one-hop link expansion.
//!
//! A behavioural port of Hindsight's `LinkExpansionRetriever`. The design's central
//! constraint is that graph retrieval must not turn one query into an unbounded chain of
//! object-storage reads — so expansion is exactly one hop, entity fan-out is capped, and
//! there is a timeout fallback, never recursion (SPEC §7).

pub mod radj;
pub mod retriever;
pub mod scorer;

pub use radj::{EdgeKind, InEdge, LinkTypeTag, ReverseAdjacency};
pub use retriever::{retrieve, GraphParams, GraphResult, GraphSource};
pub use scorer::{entity_score, ScoreAccumulator};
