//! Full-text search: a Chinese-capable BM25 arm.
//!
//! Two pieces: the [`tokenizer`] chain, shared verbatim by the indexer and the query
//! parser so a query is always tokenized the way the documents were; and a BM25
//! [`index`] packed into a single object a query can plan against from a footer.

pub mod index;
pub mod tokenizer;

pub use index::{Bm25Params, FtsBuilder, FtsHit, FtsIndex, Posting};
pub use tokenizer::{Field, Token, Tokenizer, TokenizerConfig};
