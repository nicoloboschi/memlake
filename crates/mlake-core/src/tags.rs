//! Tag filtering (ported from Hindsight's `tags_match` semantics).
//!
//! A memory carries a set of string tags. A query may restrict results to memories whose
//! tags relate to a request tag set under one of five modes. The subtle part — faithfully
//! reproduced here — is how each mode treats *untagged* memories:
//!
//! * `Any`  — OR overlap; **untagged included**.
//! * `All`  — request ⊆ memory tags; **untagged included**.
//! * `AnyStrict` — OR overlap; untagged excluded.
//! * `AllStrict` — request ⊆ memory tags; untagged excluded.
//! * `Exact` — memory tag-set *equals* request set; untagged excluded — and an *empty*
//!   request is the global/untagged scope, matching only untagged memories.
//!
//! An empty/absent request is "no filter" for every mode except `Exact`.

use serde::{Deserialize, Serialize};

/// Tag matching mode. Mirrors Hindsight's `TagsMatch`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagsMatch {
    Any,
    All,
    AnyStrict,
    AllStrict,
    Exact,
}

impl Default for TagsMatch {
    fn default() -> Self {
        Self::Any
    }
}

/// A tag filter: the request tags plus the match mode.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct TagFilter {
    pub tags: Vec<String>,
    pub mode: TagsMatch,
}

impl TagFilter {
    /// No filtering.
    pub fn none() -> Self {
        Self {
            tags: Vec::new(),
            mode: TagsMatch::Any,
        }
    }

    pub fn new(tags: Vec<String>, mode: TagsMatch) -> Self {
        Self { tags, mode }
    }

    /// True when this filter admits every memory, so callers can skip the work entirely.
    /// An empty request is a no-op for all modes except `Exact` (where it means the
    /// untagged scope).
    pub fn is_noop(&self) -> bool {
        self.tags.is_empty() && self.mode != TagsMatch::Exact
    }

    /// Whether a memory with `memory_tags` passes this filter.
    pub fn matches(&self, memory_tags: &[String]) -> bool {
        let req = &self.tags;
        if req.is_empty() {
            // Empty request: no filter, except Exact where it selects the untagged scope.
            return match self.mode {
                TagsMatch::Exact => memory_tags.is_empty(),
                _ => true,
            };
        }
        let untagged = memory_tags.is_empty();
        match self.mode {
            TagsMatch::Any => untagged || overlaps(memory_tags, req),
            TagsMatch::All => untagged || contains_all(memory_tags, req),
            TagsMatch::AnyStrict => !untagged && overlaps(memory_tags, req),
            TagsMatch::AllStrict => !untagged && contains_all(memory_tags, req),
            TagsMatch::Exact => !untagged && set_eq(memory_tags, req),
        }
    }
}

/// Any request tag present in the memory's tags.
fn overlaps(memory_tags: &[String], req: &[String]) -> bool {
    req.iter().any(|t| memory_tags.contains(t))
}

/// Every request tag present in the memory's tags (request ⊆ memory).
fn contains_all(memory_tags: &[String], req: &[String]) -> bool {
    req.iter().all(|t| memory_tags.contains(t))
}

/// The two tag sets are equal (order-independent, ignoring duplicates).
fn set_eq(memory_tags: &[String], req: &[String]) -> bool {
    use std::collections::BTreeSet;
    let a: BTreeSet<&String> = memory_tags.iter().collect();
    let b: BTreeSet<&String> = req.iter().collect();
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn empty_request_is_no_filter_except_exact() {
        let mem = tags(&["a", "b"]);
        for mode in [
            TagsMatch::Any,
            TagsMatch::All,
            TagsMatch::AnyStrict,
            TagsMatch::AllStrict,
        ] {
            assert!(TagFilter::new(vec![], mode).matches(&mem), "{mode:?}");
            assert!(TagFilter::new(vec![], mode).is_noop());
        }
        // Exact with empty request is the untagged scope.
        let exact = TagFilter::new(vec![], TagsMatch::Exact);
        assert!(!exact.is_noop());
        assert!(!exact.matches(&mem), "exact-empty must not match a tagged memory");
        assert!(exact.matches(&[]), "exact-empty must match an untagged memory");
    }

    #[test]
    fn any_is_overlap_and_includes_untagged() {
        let f = TagFilter::new(tags(&["a", "x"]), TagsMatch::Any);
        assert!(f.matches(&tags(&["a", "b"])), "overlap on a");
        assert!(!f.matches(&tags(&["b", "c"])), "no overlap");
        assert!(f.matches(&[]), "untagged included");
    }

    #[test]
    fn all_requires_subset_and_includes_untagged() {
        let f = TagFilter::new(tags(&["a", "b"]), TagsMatch::All);
        assert!(f.matches(&tags(&["a", "b", "c"])), "request subset present");
        assert!(!f.matches(&tags(&["a", "c"])), "missing b");
        assert!(f.matches(&[]), "untagged included");
    }

    #[test]
    fn strict_variants_exclude_untagged() {
        let any_s = TagFilter::new(tags(&["a"]), TagsMatch::AnyStrict);
        assert!(any_s.matches(&tags(&["a", "b"])));
        assert!(!any_s.matches(&[]), "untagged excluded in strict");

        let all_s = TagFilter::new(tags(&["a", "b"]), TagsMatch::AllStrict);
        assert!(all_s.matches(&tags(&["a", "b"])));
        assert!(!all_s.matches(&tags(&["a"])), "missing b");
        assert!(!all_s.matches(&[]), "untagged excluded in strict");
    }

    #[test]
    fn exact_is_set_equality() {
        let f = TagFilter::new(tags(&["a", "b"]), TagsMatch::Exact);
        assert!(f.matches(&tags(&["b", "a"])), "order-independent equality");
        assert!(!f.matches(&tags(&["a", "b", "c"])), "superset is not exact");
        assert!(!f.matches(&tags(&["a"])), "subset is not exact");
        assert!(!f.matches(&[]), "untagged is not exact for a non-empty request");
    }
}
