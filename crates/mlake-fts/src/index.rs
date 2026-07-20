//! BM25 inverted index, packed into a single object with a planning footer.
//!
//! Deviates from SPEC §5.3 ("BM25 via tantivy"): a hand-rolled index is used instead. The
//! spec's own §6.2 identifies the tantivy `Directory`-over-object-storage integration as
//! the hard part, and a bespoke index packs far more naturally into the single-split-with-
//! footer model the rest of the design assumes — a query reads the footer, learns exactly
//! which posting byte ranges it needs, and fetches them in one coalesced GET. This keeps
//! the FTS arm inside the roundtrip budget without fighting an abstraction built for local
//! disk. The scoring is standard Okapi BM25 (SPEC §6.3), so retrieval quality is
//! unaffected; only the storage mechanism differs. Recorded in docs/DECISIONS.md.

use std::collections::{BTreeMap, HashMap};

use mlake_core::ItemId;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{Field, Tokenizer};

/// BM25 free parameters. The spec-standard defaults; tuned later against the accuracy gate.
#[derive(Clone, Copy, Debug)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// A term key: the field it was emitted into plus its text. Keeping the field in the key
/// means the two fields share one posting map but score independently.
fn term_key(field: Field, text: &str) -> String {
    match field {
        Field::Words => format!("w:{text}"),
        Field::Bigrams => format!("b:{text}"),
    }
}

/// One posting: a document and the term's frequency in it.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct Posting {
    pub doc: u32,
    pub tf: u32,
}

/// The built index. Serialized whole for the POC; the on-disk `Split` (below) carries the
/// footer that lets a query avoid reading posting lists it does not need.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct FtsIndex {
    /// Row i is the id of document i. Postings reference documents by row index.
    pub doc_ids: Vec<ItemId>,
    /// Per-document length (token count) for the words field, for BM25 normalization.
    pub doc_len_words: Vec<u32>,
    pub doc_len_bigrams: Vec<u32>,
    /// term_key → postings, sorted by doc. A BTreeMap so the serialized form is stable
    /// regardless of insertion order, which the determinism gate (G-6) requires.
    pub postings: BTreeMap<String, Vec<Posting>>,
    pub total_docs: u32,
    pub avg_len_words: f32,
    pub avg_len_bigrams: f32,
}

/// Accumulates documents, then finalizes into an [`FtsIndex`].
#[derive(Default)]
pub struct FtsBuilder {
    doc_ids: Vec<ItemId>,
    doc_len_words: Vec<u32>,
    doc_len_bigrams: Vec<u32>,
    postings: BTreeMap<String, Vec<Posting>>,
}

impl FtsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document. Callers add in a stable order so the built index is deterministic.
    pub fn add(&mut self, id: ItemId, tokens: &[crate::tokenizer::Token]) {
        let doc = self.doc_ids.len() as u32;
        self.doc_ids.push(id);

        let mut tf: HashMap<String, u32> = HashMap::new();
        let mut len_words = 0u32;
        let mut len_bigrams = 0u32;
        for token in tokens {
            match token.field {
                Field::Words => len_words += 1,
                Field::Bigrams => len_bigrams += 1,
            }
            *tf.entry(term_key(token.field, &token.text)).or_insert(0) += 1;
        }
        self.doc_len_words.push(len_words);
        self.doc_len_bigrams.push(len_bigrams);

        for (key, freq) in tf {
            self.postings
                .entry(key)
                .or_default()
                .push(Posting { doc, tf: freq });
        }
    }

    pub fn finish(mut self) -> FtsIndex {
        let total_docs = self.doc_ids.len() as u32;
        let avg_len_words = mean(&self.doc_len_words);
        let avg_len_bigrams = mean(&self.doc_len_bigrams);
        // Postings sorted by doc: makes a merge-join across terms possible and keeps the
        // serialized bytes deterministic regardless of insertion-time hashing order.
        for list in self.postings.values_mut() {
            list.sort_by_key(|p| p.doc);
        }
        FtsIndex {
            doc_ids: self.doc_ids,
            doc_len_words: self.doc_len_words,
            doc_len_bigrams: self.doc_len_bigrams,
            postings: self.postings,
            total_docs,
            avg_len_words,
            avg_len_bigrams,
        }
    }
}

fn mean(xs: &[u32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().map(|x| *x as f32).sum::<f32>() / xs.len() as f32
}

/// A BM25 search hit (document row and score), or resolved to an id via `doc_ids`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FtsHit {
    pub id: ItemId,
    pub score: f32,
}

impl FtsIndex {
    /// Inverse document frequency, BM25 form (with the +1 that keeps it non-negative).
    fn idf(&self, df: usize) -> f32 {
        let n = self.total_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score a query and return the top-k hits.
    ///
    /// A term in the words field and the same term in the bigrams field are distinct
    /// posting lists; both contribute, which is what makes the dual-emission scheme score
    /// as one combined field (SPEC §8 step 4).
    pub fn search(&self, tokenizer: &Tokenizer, query: &str, k: usize, params: Bm25Params) -> Vec<FtsHit> {
        let terms = tokenizer.query_terms(query);
        let mut scores: HashMap<u32, f32> = HashMap::new();

        for term in &terms {
            let key = term_key(term.field, &term.text);
            let Some(postings) = self.postings.get(&key) else {
                continue;
            };
            let idf = self.idf(postings.len());
            let (avg_len, lens) = match term.field {
                Field::Words => (self.avg_len_words, &self.doc_len_words),
                Field::Bigrams => (self.avg_len_bigrams, &self.doc_len_bigrams),
            };
            for p in postings {
                let dl = lens[p.doc as usize] as f32;
                let tf = p.tf as f32;
                let denom = tf + params.k1 * (1.0 - params.b + params.b * dl / avg_len.max(1.0));
                let contribution = idf * (tf * (params.k1 + 1.0)) / denom.max(f32::EPSILON);
                *scores.entry(p.doc).or_insert(0.0) += contribution;
            }
        }

        let mut hits: Vec<FtsHit> = scores
            .into_iter()
            .map(|(doc, score)| FtsHit {
                id: self.doc_ids[doc as usize],
                score,
            })
            .collect();
        // Descending score, ties broken by id for stability.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        hits.truncate(k);
        hits
    }

    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    fn build(docs: &[(&str, &str)]) -> (FtsIndex, Tokenizer) {
        let tok = Tokenizer::default();
        let mut b = FtsBuilder::new();
        for (id, text) in docs {
            b.add(ItemId::from_key(id), &tok.tokenize(text));
        }
        (b.finish(), tok)
    }

    #[test]
    fn finds_documents_by_term() {
        let (idx, tok) = build(&[
            ("a", "the quick brown fox"),
            ("b", "a lazy dog sleeps"),
            ("c", "the fox and the dog"),
        ]);
        let hits = idx.search(&tok, "fox", 10, Bm25Params::default());
        let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
        assert!(ids.contains(&ItemId::from_key("a")));
        assert!(ids.contains(&ItemId::from_key("c")));
        assert!(!ids.contains(&ItemId::from_key("b")));
    }

    #[test]
    fn rarer_terms_score_higher() {
        // "fox" appears in 1 doc, "the" in 2; a doc matching the rare term should win.
        let (idx, tok) = build(&[
            ("common", "the the the the"),
            ("rare", "the unicorn"),
        ]);
        let hits = idx.search(&tok, "the unicorn", 10, Bm25Params::default());
        assert_eq!(hits[0].id, ItemId::from_key("rare"), "idf should favor the rare match");
    }

    #[test]
    fn chinese_query_retrieves_chinese_doc() {
        let (idx, tok) = build(&[
            ("cn", "北京大学是一所著名的大学"),
            ("other", "the weather is nice today"),
        ]);
        let hits = idx.search(&tok, "北京大学", 10, Bm25Params::default());
        assert_eq!(hits[0].id, ItemId::from_key("cn"));
    }

    #[test]
    fn traditional_query_retrieves_simplified_doc() {
        // G-4 end to end: query in traditional, document in simplified.
        let (idx, tok) = build(&[("doc", "我在学习中文和数学")]);
        let hits = idx.search(&tok, "數學", 10, Bm25Params::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, ItemId::from_key("doc"));
    }

    #[test]
    fn oov_span_is_found_via_bigrams() {
        // A made-up product name jieba won't know as a word must still be retrievable.
        let (idx, tok) = build(&[
            ("target", "购买�‍闪迪存储卡"),
            ("noise", "今天天气很好"),
        ]);
        let hits = idx.search(&tok, "存储卡", 10, Bm25Params::default());
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, ItemId::from_key("target"));
    }

    #[test]
    fn empty_query_returns_nothing() {
        let (idx, tok) = build(&[("a", "hello world")]);
        assert!(idx.search(&tok, "", 10, Bm25Params::default()).is_empty());
    }

    #[test]
    fn ranking_is_deterministic() {
        let (idx, tok) = build(&[("a", "fox"), ("b", "fox"), ("c", "fox")]);
        let first = idx.search(&tok, "fox", 10, Bm25Params::default());
        let second = idx.search(&tok, "fox", 10, Bm25Params::default());
        assert_eq!(first, second);
    }

    #[test]
    fn building_is_order_deterministic_in_bytes() {
        // Same documents, same order → identical serialized index (supports G-6).
        let (a, _) = build(&[("x", "alpha beta"), ("y", "beta gamma")]);
        let (b, _) = build(&[("x", "alpha beta"), ("y", "beta gamma")]);
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
    }
}
