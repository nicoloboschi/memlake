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

/// A tag filter: the request tags plus the match mode, and an optional compound predicate.
///
/// `tags`/`mode` are the flat condition the retrieval arms push down efficiently (via the
/// per-cluster/-block tag masks). `groups` is the compound form — a list of boolean trees
/// AND-ed together and AND-ed onto the flat condition — for queries a single flat filter
/// cannot express. Because a boolean tree over a memory's *full* tag set cannot be evaluated
/// from the compact block masks, `groups` is NOT consulted by [`matches`], [`is_noop`], or
/// [`cluster_admits`] — those stay pure-flat so cluster/block pruning is unchanged. The
/// compound predicate is applied once, per-memory, after a hit's full record is hydrated (see
/// the query node's materialization pass). Empty `groups` is a no-op.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct TagFilter {
    pub tags: Vec<String>,
    pub mode: TagsMatch,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<TagPredicate>,
}

impl TagFilter {
    /// No filtering.
    pub fn none() -> Self {
        Self {
            tags: Vec::new(),
            mode: TagsMatch::Any,
            groups: Vec::new(),
        }
    }

    pub fn new(tags: Vec<String>, mode: TagsMatch) -> Self {
        Self { tags, mode, groups: Vec::new() }
    }

    /// Whether every top-level `groups` predicate accepts a memory with `memory_tags`. Empty
    /// `groups` accepts everything (a top-level AND of nothing). This is the compound half of
    /// the filter, evaluated separately from the flat [`matches`] because it needs the memory's
    /// complete tag set, not the pushed-down masks.
    pub fn groups_match(&self, memory_tags: &[String]) -> bool {
        self.groups.iter().all(|g| g.matches(memory_tags))
    }

    /// True when this filter admits every memory, so callers can skip the work entirely.
    /// An empty request is a no-op for all modes except `Exact` (where it means the
    /// untagged scope).
    pub fn is_noop(&self) -> bool {
        self.tags.is_empty() && self.mode != TagsMatch::Exact
    }

    /// Whether a *cluster* could contain a memory that passes this filter, given the union
    /// of all its memories' tags (`cluster_tags`) and whether any of its memories is
    /// untagged (`has_untagged`). This is a conservative superset test used to prune
    /// clusters before fetching them (SCALE.md Phase 4b): if it returns false, no memory in
    /// the cluster can match, so the cluster is skipped; if true, the cluster is fetched and
    /// the per-memory [`matches`] confirms. Necessary conditions:
    /// * overlap modes need the cluster union to intersect the request;
    /// * `all`/`all_strict`/`exact` need `request ⊆ cluster_union` (a memory carrying all
    ///   request tags contributes them all to the union).
    pub fn cluster_admits(&self, cluster_tags: &[String], has_untagged: bool) -> bool {
        let req = &self.tags;
        if req.is_empty() {
            return match self.mode {
                TagsMatch::Exact => has_untagged, // untagged scope
                _ => true,                        // no filter
            };
        }
        let untagged_ok = has_untagged;
        match self.mode {
            TagsMatch::Any => untagged_ok || overlaps(cluster_tags, req),
            TagsMatch::All => untagged_ok || contains_all(cluster_tags, req),
            TagsMatch::AnyStrict => overlaps(cluster_tags, req),
            TagsMatch::AllStrict | TagsMatch::Exact => contains_all(cluster_tags, req),
        }
    }

    /// Leaf-of-a-predicate matching: like [`matches`] but WITHOUT the flat "empty request = no
    /// filter" convention. Inside a boolean [`TagPredicate`] tree an empty leaf is a literal set
    /// condition, not a no-op — e.g. an empty `AnyStrict` leaf matches nothing, an empty `All`
    /// leaf matches everything, an empty `Exact` leaf is the untagged scope. Mirrors Hindsight's
    /// `_match_group` leaf so a pushed-down predicate agrees with the Python post-filter
    /// memory-for-memory. For a non-empty request this is identical to [`matches`].
    pub fn leaf_matches(&self, memory_tags: &[String]) -> bool {
        let req = &self.tags;
        let untagged = memory_tags.is_empty();
        if self.mode == TagsMatch::Exact && req.is_empty() {
            return untagged; // the empty Exact scope selects only untagged memories
        }
        match self.mode {
            TagsMatch::Any => untagged || overlaps(memory_tags, req),
            TagsMatch::All => untagged || contains_all(memory_tags, req),
            TagsMatch::AnyStrict => !untagged && overlaps(memory_tags, req),
            TagsMatch::AllStrict => !untagged && contains_all(memory_tags, req),
            TagsMatch::Exact => !untagged && set_eq(memory_tags, req),
        }
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

/// A boolean tree over a memory's tags — the compound form of [`TagFilter`]. Leaves are flat
/// [`TagFilter`]s (evaluated by [`TagFilter::matches`], so every per-mode subtlety — including
/// how each mode treats untagged memories — is inherited unchanged); interior nodes combine
/// children with AND / OR / NOT. Mirrors Hindsight's recursive `TagGroup`, so a pushed-down
/// predicate and Hindsight's own Python `filter_results_by_tag_groups` agree memory-for-memory.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum TagPredicate {
    /// A flat tag condition.
    Leaf(TagFilter),
    /// AND: every child must match. An empty `All` matches every memory (AND identity).
    All(Vec<TagPredicate>),
    /// OR: at least one child must match. An empty `Any` matches no memory (OR identity).
    Any(Vec<TagPredicate>),
    /// NOT: the child must not match.
    Not(Box<TagPredicate>),
}

impl TagPredicate {
    /// Whether a memory with `memory_tags` satisfies this predicate.
    pub fn matches(&self, memory_tags: &[String]) -> bool {
        match self {
            TagPredicate::Leaf(f) => f.leaf_matches(memory_tags),
            TagPredicate::All(children) => children.iter().all(|c| c.matches(memory_tags)),
            TagPredicate::Any(children) => children.iter().any(|c| c.matches(memory_tags)),
            TagPredicate::Not(child) => !child.matches(memory_tags),
        }
    }
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

    fn leaf(t: &[&str], mode: TagsMatch) -> TagPredicate {
        TagPredicate::Leaf(TagFilter::new(tags(t), mode))
    }

    #[test]
    fn predicate_leaf_delegates_to_flat_matches() {
        let p = leaf(&["a"], TagsMatch::AnyStrict);
        assert!(p.matches(&tags(&["a", "b"])));
        assert!(!p.matches(&tags(&["b"])));
        assert!(!p.matches(&[]), "any_strict excludes untagged");
    }

    #[test]
    fn predicate_and_or_not_compose() {
        // (a AND b)  — all_strict on each leaf, combined by All.
        let and = TagPredicate::All(vec![leaf(&["a"], TagsMatch::AnyStrict), leaf(&["b"], TagsMatch::AnyStrict)]);
        assert!(and.matches(&tags(&["a", "b", "c"])));
        assert!(!and.matches(&tags(&["a"])), "missing b");

        // (a OR b)
        let or = TagPredicate::Any(vec![leaf(&["a"], TagsMatch::AnyStrict), leaf(&["b"], TagsMatch::AnyStrict)]);
        assert!(or.matches(&tags(&["b"])));
        assert!(!or.matches(&tags(&["c"])));

        // NOT (a)
        let not = TagPredicate::Not(Box::new(leaf(&["a"], TagsMatch::AnyStrict)));
        assert!(not.matches(&tags(&["b"])));
        assert!(!not.matches(&tags(&["a", "b"])));

        // (a AND b) OR (NOT c) — a nested tree
        let not_c = TagPredicate::Not(Box::new(leaf(&["c"], TagsMatch::AnyStrict)));
        let tree = TagPredicate::Any(vec![and.clone(), not_c]);
        assert!(tree.matches(&tags(&["a", "b"])), "left branch: has a and b");
        assert!(tree.matches(&tags(&["x"])), "right branch: no c");
        assert!(!tree.matches(&tags(&["c"])), "neither: c present, and a&b absent");
    }

    #[test]
    fn empty_leaf_matches_hindsight_match_group() {
        // An empty leaf is a literal set condition inside a tree — NOT a no-op (unlike a flat
        // filter). Mirrors Hindsight's _match_group for empty group.tags.
        let mem = tags(&["a"]);
        // any: untagged only
        assert!(!leaf(&[], TagsMatch::Any).matches(&mem));
        assert!(leaf(&[], TagsMatch::Any).matches(&[]));
        // all: everything
        assert!(leaf(&[], TagsMatch::All).matches(&mem));
        assert!(leaf(&[], TagsMatch::All).matches(&[]));
        // any_strict: nothing
        assert!(!leaf(&[], TagsMatch::AnyStrict).matches(&mem));
        assert!(!leaf(&[], TagsMatch::AnyStrict).matches(&[]));
        // all_strict: tagged only
        assert!(leaf(&[], TagsMatch::AllStrict).matches(&mem));
        assert!(!leaf(&[], TagsMatch::AllStrict).matches(&[]));
        // exact-empty: the untagged scope
        assert!(!leaf(&[], TagsMatch::Exact).matches(&mem));
        assert!(leaf(&[], TagsMatch::Exact).matches(&[]));
    }

    #[test]
    fn empty_and_is_true_empty_or_is_false() {
        assert!(TagPredicate::All(vec![]).matches(&tags(&["a"])), "AND identity");
        assert!(!TagPredicate::Any(vec![]).matches(&tags(&["a"])), "OR identity");
    }

    #[test]
    fn groups_match_is_top_level_conjunction() {
        let mut f = TagFilter::new(tags(&["x"]), TagsMatch::Any);
        f.groups = vec![leaf(&["a"], TagsMatch::AnyStrict), leaf(&["b"], TagsMatch::AnyStrict)];
        assert!(f.groups_match(&tags(&["a", "b"])), "both groups satisfied");
        assert!(!f.groups_match(&tags(&["a"])), "one group unsatisfied");
        // groups do not touch the flat matches / is_noop paths
        assert!(!f.is_noop(), "flat filter is non-empty");
        assert!(TagFilter::none().groups_match(&tags(&["a"])), "no groups = always true");
    }
}
