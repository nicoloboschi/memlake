//! Reverse adjacency in CSR form (SPEC §3.4).
//!
//! An item's *outgoing* links live inline in its record, so walking forward is free once
//! the item is fetched. Walking *backward* — "who links to me" — needs a separate
//! structure, because rewriting every source item when a new target appears is exactly
//! the eager neighbour-rewrite the design forbids (SPEC §4).
//!
//! CSR keeps incoming edges compact and sorted by target, so all edges into one item are
//! a contiguous slice found by binary search. A sparse offset index over the targets
//! turns that search into one small cached lookup plus one coalesced ranged read.

use mlake_core::{MemoryId, LinkType};
use serde::{Deserialize, Serialize};

/// An incoming edge: who points at the target, by what kind of link, with what weight.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct InEdge {
    pub source: MemoryId,
    pub kind: EdgeKind,
    pub weight: f32,
}

/// Edge category. Semantic edges are the derived kNN graph; causal edges carry the link
/// type so the causal arm can distinguish `causes` from `prevents`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum EdgeKind {
    Semantic,
    Causal(LinkTypeTag),
}

/// Serializable mirror of `mlake_core::LinkType` (which is an rkyv type, not serde).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum LinkTypeTag {
    Causes,
    CausedBy,
    Enables,
    Prevents,
}

impl From<LinkType> for LinkTypeTag {
    fn from(lt: LinkType) -> Self {
        match lt {
            LinkType::Causes => Self::Causes,
            LinkType::CausedBy => Self::CausedBy,
            LinkType::Enables => Self::Enables,
            LinkType::Prevents => Self::Prevents,
        }
    }
}

/// CSR reverse-adjacency: `targets[i]` owns the edge slice `edges[offsets[i]..offsets[i+1]]`.
#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub struct ReverseAdjacency {
    /// Distinct target ids, sorted ascending for binary search.
    targets: Vec<MemoryId>,
    /// `targets.len() + 1` offsets into `edges`.
    offsets: Vec<u32>,
    edges: Vec<InEdge>,
}

impl ReverseAdjacency {
    /// Build from an iterator of (target, edge) pairs.
    pub fn build(mut pairs: Vec<(MemoryId, InEdge)>) -> Self {
        // Sort by target so each target's edges are contiguous; a stable secondary sort
        // by source keeps the built file byte-deterministic (G-6).
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.source.cmp(&b.1.source)));

        let mut targets = Vec::new();
        let mut offsets = vec![0u32];
        let mut edges = Vec::with_capacity(pairs.len());

        let mut i = 0;
        while i < pairs.len() {
            let target = pairs[i].0;
            targets.push(target);
            while i < pairs.len() && pairs[i].0 == target {
                edges.push(pairs[i].1);
                i += 1;
            }
            offsets.push(edges.len() as u32);
        }

        Self {
            targets,
            offsets,
            edges,
        }
    }

    /// Incoming edges for a target, or an empty slice if it has none.
    ///
    /// One binary search into the sorted targets — the operation the sparse `radj.idx`
    /// accelerates on the cold path.
    pub fn incoming(&self, target: &MemoryId) -> &[InEdge] {
        match self.targets.binary_search(target) {
            Ok(i) => {
                let start = self.offsets[i] as usize;
                let end = self.offsets[i + 1] as usize;
                &self.edges[start..end]
            }
            Err(_) => &[],
        }
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(source: &str, weight: f32) -> InEdge {
        InEdge {
            source: MemoryId::from_key(source),
            kind: EdgeKind::Semantic,
            weight,
        }
    }

    #[test]
    fn groups_edges_by_target() {
        let t1 = MemoryId::from_key("t1");
        let t2 = MemoryId::from_key("t2");
        let radj = ReverseAdjacency::build(vec![
            (t1, edge("a", 0.8)),
            (t2, edge("b", 0.9)),
            (t1, edge("c", 0.7)),
        ]);
        assert_eq!(radj.incoming(&t1).len(), 2);
        assert_eq!(radj.incoming(&t2).len(), 1);
        assert_eq!(radj.incoming(&t2)[0].source, MemoryId::from_key("b"));
    }

    #[test]
    fn a_target_with_no_incoming_edges_is_empty_not_an_error() {
        let radj = ReverseAdjacency::build(vec![(MemoryId::from_key("t"), edge("a", 0.8))]);
        assert!(radj.incoming(&MemoryId::from_key("unknown")).is_empty());
    }

    #[test]
    fn empty_adjacency_is_valid() {
        let radj = ReverseAdjacency::build(vec![]);
        assert_eq!(radj.edge_count(), 0);
        assert!(radj.incoming(&MemoryId::from_key("x")).is_empty());
    }

    #[test]
    fn build_is_deterministic() {
        // G-6: same edges in any input order → identical structure.
        let t = MemoryId::from_key("t");
        let a = ReverseAdjacency::build(vec![
            (t, edge("z", 0.7)),
            (t, edge("a", 0.9)),
        ]);
        let b = ReverseAdjacency::build(vec![
            (t, edge("a", 0.9)),
            (t, edge("z", 0.7)),
        ]);
        assert_eq!(a, b);
    }

    #[test]
    fn roundtrips_through_bytes() {
        let radj = ReverseAdjacency::build(vec![
            (MemoryId::from_key("t"), edge("a", 0.8)),
            (MemoryId::from_key("t"), edge("b", 0.9)),
        ]);
        let bytes = radj.to_bytes().unwrap();
        assert_eq!(ReverseAdjacency::from_bytes(&bytes).unwrap(), radj);
    }

    #[test]
    fn offsets_span_every_edge_exactly_once() {
        let t1 = MemoryId::from_key("t1");
        let t2 = MemoryId::from_key("t2");
        let t3 = MemoryId::from_key("t3");
        let radj = ReverseAdjacency::build(vec![
            (t1, edge("a", 0.8)),
            (t2, edge("b", 0.9)),
            (t2, edge("c", 0.7)),
            (t3, edge("d", 0.6)),
        ]);
        let total: usize = [t1, t2, t3].iter().map(|t| radj.incoming(t).len()).sum();
        assert_eq!(total, radj.edge_count());
    }
}
