//! IVF (inverted file) vector search.
//!
//! Centroid-partitioned rather than graph-based ANN, per SPEC §1: a graph index would
//! need pointer chasing across object storage, turning one query into an unbounded chain
//! of dependent roundtrips. Partitioning makes the read set knowable up front — probe the
//! centroid table, then fetch the chosen clusters in one parallel batch — which is what
//! keeps a query inside a fixed roundtrip budget (INV-7).

pub mod index;
pub mod kmeans;
pub mod vectors;

pub use index::{
    build_clusters, exact_search, merge_hits, sort_hits, train_centroids, train_centroids_k,
    Centroids, ClusterFile, Hit,
};
pub use kmeans::centroid_count;
pub use vectors::{PreparedQuery, TagMask, VectorBlock, VectorCodec};

/// Default clusters probed per query (SPEC §6.3).
pub const DEFAULT_NPROBE: usize = 8;
