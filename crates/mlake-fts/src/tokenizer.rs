//! The tokenization chain (SPEC §8).
//!
//! One implementation serves both the indexer and the query parser — that is the whole
//! point of the config hash in the manifest: a query tokenized differently from the
//! documents it searches silently loses recall, so the two paths must be the same code.
//!
//! The chain is: normalize (NFKC → traditional-to-simplified → lowercase), segment by
//! Unicode script, then tokenize each run by its script. Han runs emit into two fields —
//! jieba words and character bigrams — so that both dictionary words and out-of-vocabulary
//! spans (product names, novel compounds) are retrievable.

use once_cell::sync::Lazy;
use unicode_normalization::UnicodeNormalization;
use unicode_script::{Script, UnicodeScript};

/// The two text fields a document is indexed into. Latin tokens go into both so a mixed
/// query matches regardless of which field a term landed in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Field {
    /// jieba word tokens and Latin/other word tokens.
    Words,
    /// Character bigrams for CJK scripts.
    Bigrams,
}

/// A produced token: its text and which field it belongs to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Token {
    pub field: Field,
    pub text: String,
}

impl Token {
    fn words(text: impl Into<String>) -> Self {
        Self {
            field: Field::Words,
            text: text.into(),
        }
    }
    fn bigrams(text: impl Into<String>) -> Self {
        Self {
            field: Field::Bigrams,
            text: text.into(),
        }
    }
}

static JIEBA: Lazy<jieba_rs::Jieba> = Lazy::new(jieba_rs::Jieba::new);

static STEMMER: Lazy<rust_stemmers::Stemmer> =
    Lazy::new(|| rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English));

/// A small English stopword set. Kept short deliberately: aggressive stopping hurts
/// phrase-like queries, but the highest-frequency function words carry almost no
/// discriminative signal and inflate document lengths, weakening BM25 normalization.
static STOPWORDS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
        "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
        "this", "these", "those", "or", "but", "not", "have", "had", "which", "we", "they",
        "their", "them", "our", "you", "your", "can", "could", "would", "should", "than",
        "then", "there", "here", "into", "over", "under", "such", "been", "being", "also",
    ]
    .into_iter()
    .collect()
});

/// Configuration that affects tokenization output. Its hash is stored in the manifest so
/// a segment built under one configuration is never queried under another.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TokenizerConfig {
    /// Whether to fold traditional Chinese to simplified. On by default so a
    /// traditional-script query retrieves simplified documents (SPEC G-4).
    pub traditional_to_simplified: bool,
    /// Apply the Snowball English stemmer and drop stopwords on the Latin path. On by
    /// default: it measurably lifts English BM25 by conflating morphological variants and
    /// keeping high-frequency function words from distorting length normalization. SPEC §8
    /// makes the light stemmer a config option.
    pub english_stemming: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            traditional_to_simplified: true,
            english_stemming: true,
        }
    }
}

impl TokenizerConfig {
    /// A stable hash of the configuration, for the manifest's `tokenizer_config_hash`.
    pub fn config_hash(&self) -> String {
        // Version prefix so a change to the chain's *code* (not just its config) can be
        // forced to invalidate segments by bumping it.
        let mut hash: u64 = 0xcbf29ce484222325;
        let repr = format!(
            "v2:t2s={}:stem={}",
            self.traditional_to_simplified, self.english_stemming
        );
        for byte in repr.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{hash:016x}")
    }
}

/// The tokenizer. Cheap to construct and clone; holds only config (jieba is a shared
/// static), so a query node can spin up matching tokenizers for several splits freely.
#[derive(Clone)]
pub struct Tokenizer {
    config: TokenizerConfig,
}

impl Tokenizer {
    /// Its configuration, for constructing a matching tokenizer elsewhere.
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(TokenizerConfig::default())
    }
}

impl Tokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Self { config }
    }

    pub fn config_hash(&self) -> String {
        self.config.config_hash()
    }

    /// Normalize: NFKC, optional traditional→simplified, lowercase.
    fn normalize(&self, text: &str) -> String {
        let nfkc: String = text.nfkc().collect();
        let converted = if self.config.traditional_to_simplified {
            character_converter::traditional_to_simplified(&nfkc).into_owned()
        } else {
            nfkc
        };
        converted.to_lowercase()
    }

    /// Tokenize text into field-tagged tokens.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let normalized = self.normalize(text);
        let mut tokens = Vec::new();
        for run in script_runs(&normalized) {
            self.tokenize_run(run, &mut tokens);
        }
        tokens
    }

    /// The distinct term set of a query, used to plan which postings to read.
    pub fn query_terms(&self, text: &str) -> Vec<Token> {
        let mut tokens = self.tokenize(text);
        tokens.sort_by(|a, b| (a.field as u8, &a.text).cmp(&(b.field as u8, &b.text)));
        tokens.dedup();
        tokens
    }

    fn tokenize_run(&self, run: Run<'_>, out: &mut Vec<Token>) {
        match run.kind {
            RunKind::Han => {
                // Dual emission: dictionary words for precision, bigrams for OOV recall.
                for word in JIEBA.cut_for_search(run.text, true) {
                    let w = word.trim();
                    if !w.is_empty() {
                        out.push(Token::words(w));
                    }
                }
                for bg in char_bigrams(run.text) {
                    out.push(Token::bigrams(bg));
                }
            }
            RunKind::CjkOther => {
                // Kana/Hangul: bigrams only in v1.
                for bg in char_bigrams(run.text) {
                    out.push(Token::bigrams(bg));
                }
            }
            RunKind::Latin => {
                for word in latin_tokens(run.text) {
                    let word = if self.config.english_stemming {
                        // Drop stopwords before stemming; the stopword list is unstemmed
                        // surface forms.
                        if STOPWORDS.contains(word.as_str()) {
                            continue;
                        }
                        STEMMER.stem(&word).into_owned()
                    } else {
                        word
                    };
                    // Latin tokens hit both fields so a query term matches whichever field
                    // a mixed-script document happened to place it in.
                    out.push(Token::words(word.clone()));
                    out.push(Token::bigrams(word));
                }
            }
        }
    }
}

/// Character bigrams over a run, plus the single character when a run is one char long so
/// a lone ideograph is still retrievable.
fn char_bigrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return Vec::new();
    }
    if chars.len() == 1 {
        return vec![chars[0].to_string()];
    }
    chars
        .windows(2)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

/// Split Latin text on whitespace and punctuation, keeping code-identifier characters
/// (`_`, digits) so identifiers survive intact.
fn latin_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunKind {
    Han,
    CjkOther,
    Latin,
}

struct Run<'a> {
    kind: RunKind,
    text: &'a str,
}

/// Classify a character into a run kind, or `None` for separators that break runs.
fn classify(c: char) -> Option<RunKind> {
    if c.is_whitespace() {
        return None;
    }
    match c.script() {
        Script::Han => Some(RunKind::Han),
        Script::Hiragana | Script::Katakana | Script::Hangul => Some(RunKind::CjkOther),
        _ => Some(RunKind::Latin),
    }
}

/// Segment text into maximal runs of one script kind. Whitespace ends a Latin run but is
/// otherwise dropped; punctuation stays inside Latin runs for `latin_tokens` to split.
fn script_runs(text: &str) -> Vec<Run<'_>> {
    let mut runs = Vec::new();
    let mut start = 0usize;
    let mut current: Option<RunKind> = None;

    for (i, c) in text.char_indices() {
        let kind = classify(c);
        match (current, kind) {
            (Some(cur), Some(k)) if cur == k => {}
            (Some(_), _) => {
                if start < i {
                    runs.push(Run {
                        kind: current.unwrap(),
                        text: &text[start..i],
                    });
                }
                start = i;
                current = kind;
                if kind.is_none() {
                    // Skip the separator itself.
                    start = i + c.len_utf8();
                }
            }
            (None, Some(_)) => {
                start = i;
                current = kind;
            }
            (None, None) => {
                start = i + c.len_utf8();
            }
        }
    }
    if let Some(k) = current {
        if start < text.len() {
            runs.push(Run {
                kind: k,
                text: &text[start..],
            });
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(tokens: &[Token]) -> Vec<String> {
        tokens
            .iter()
            .filter(|t| t.field == Field::Words)
            .map(|t| t.text.clone())
            .collect()
    }

    fn bigrams(tokens: &[Token]) -> Vec<String> {
        tokens
            .iter()
            .filter(|t| t.field == Field::Bigrams)
            .map(|t| t.text.clone())
            .collect()
    }

    #[test]
    fn latin_is_lowercased_and_split() {
        let t = Tokenizer::default();
        let tokens = t.tokenize("Hello, World!");
        assert!(words(&tokens).contains(&"hello".to_string()));
        assert!(words(&tokens).contains(&"world".to_string()));
    }

    #[test]
    fn code_identifiers_stay_intact() {
        let t = Tokenizer::default();
        let w = words(&t.tokenize("fn get_user_id()"));
        assert!(w.contains(&"get_user_id".to_string()), "got {w:?}");
    }

    #[test]
    fn han_emits_both_words_and_bigrams() {
        let t = Tokenizer::default();
        let tokens = t.tokenize("北京大学");
        // jieba should recover the compound and its parts...
        assert!(words(&tokens).contains(&"北京".to_string()), "words: {:?}", words(&tokens));
        // ...and bigrams cover the same span for OOV robustness.
        assert!(bigrams(&tokens).contains(&"北京".to_string()));
        assert!(bigrams(&tokens).contains(&"大学".to_string()));
    }

    #[test]
    fn traditional_query_matches_simplified_text() {
        // G-4: the OpenCC path must let a traditional-script query reach simplified docs.
        let t = Tokenizer::default();
        let traditional = t.tokenize("漢語");
        let simplified = t.tokenize("汉语");
        assert_eq!(
            words(&traditional),
            words(&simplified),
            "traditional and simplified must normalize to the same tokens"
        );
    }

    #[test]
    fn disabling_t2s_keeps_scripts_distinct() {
        let t = Tokenizer::new(TokenizerConfig {
            traditional_to_simplified: false,
            english_stemming: false,
        });
        assert_ne!(words(&t.tokenize("漢語")), words(&t.tokenize("汉语")));
    }

    #[test]
    fn mixed_script_text_segments_correctly() {
        let t = Tokenizer::default();
        let tokens = t.tokenize("使用 GPT 模型");
        let w = words(&tokens);
        assert!(w.contains(&"gpt".to_string()), "Latin run lost: {w:?}");
        assert!(
            w.iter().any(|x| x.contains('模') || x == "模型"),
            "Han run lost: {w:?}"
        );
    }

    #[test]
    fn single_ideograph_is_still_a_token() {
        let t = Tokenizer::default();
        let tokens = t.tokenize("水");
        assert!(
            bigrams(&tokens).contains(&"水".to_string()),
            "a lone character must be retrievable"
        );
    }

    #[test]
    fn query_terms_are_deduped() {
        let t = Tokenizer::default();
        let terms = t.query_terms("test test test");
        let count = terms.iter().filter(|t| t.text == "test").count();
        // Once per field (words + bigrams), not once per occurrence.
        assert_eq!(count, 2, "got {terms:?}");
    }

    #[test]
    fn config_hash_is_stable_and_config_sensitive() {
        let a = TokenizerConfig::default().config_hash();
        let b = TokenizerConfig::default().config_hash();
        assert_eq!(a, b);
        let c = TokenizerConfig {
            traditional_to_simplified: false,
            english_stemming: false,
        }
        .config_hash();
        assert_ne!(a, c, "different config must hash differently");
    }

    #[test]
    fn nfkc_folds_fullwidth_forms() {
        let t = Tokenizer::default();
        // Fullwidth latin should normalize to ascii.
        let full = t.tokenize("ＧＰＴ");
        assert!(words(&full).contains(&"gpt".to_string()), "got {:?}", words(&full));
    }
}
