//! Full-text search: a Chinese-capable BM25 arm.
//!
//! Two pieces: the [`tokenizer`] chain, shared verbatim by the indexer and the query
//! parser so a query is always tokenized the way the documents were; and a
//! [`tantivy_index`] tantivy-backed BM25 arm packed into a single S3-native split.

pub mod tantivy_index;
pub mod tokenizer;

pub use tantivy_index::{FtsHit, TantivyFts};
pub use tokenizer::{Field, Token, Tokenizer, TokenizerConfig};
