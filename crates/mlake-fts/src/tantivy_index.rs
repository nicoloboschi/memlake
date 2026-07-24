//! tantivy-backed BM25, packaged as an S3-native split.
//!
//! This is the FTS arm the spec asks for (§5.3, "BM25 via tantivy"), implemented so it
//! fits the object-storage model: a whole tantivy index is packed into one immutable
//! `split.bin` object, and a query node materializes it into the local NVMe/mmap tier to
//! serve reads — the "warm query served from NVMe/mmap" path of §6.1.
//!
//! Reusing tantivy buys real block-WAND scoring, compressed posting blocks, and mature
//! segment handling; the custom Chinese-capable tokenization (§8) is preserved by feeding
//! tantivy *pre-tokenized* streams, so jieba dual-emission and the OpenCC fold still drive
//! what gets indexed while tantivy owns storage and scoring.

use std::path::Path;

use mlake_core::{MemoryId, TagFilter};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{
    IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, STORED,
};
use tantivy::tokenizer::{PreTokenizedString, Token as TantivyToken};
use tantivy::{doc, Index, IndexReader, ReloadPolicy, TantivyDocument, Term};
use uuid::Uuid;

use crate::tokenizer::{Field, Tokenizer};

/// A BM25 hit: the item and its score. Same shape as the rest of the FTS arm returns, so
/// fusion and the benchmark are unaffected by which backend produced it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FtsHit {
    pub id: MemoryId,
    pub score: f32,
}

/// The two indexed text fields, mirroring the dual-emission scheme: jieba/Latin word
/// tokens in one, CJK character bigrams in the other. A query term hits whichever field it
/// was emitted into, and tantivy sums the per-field BM25 contributions.
struct Fields {
    words: tantivy::schema::Field,
    bigrams: tantivy::schema::Field,
    id: tantivy::schema::Field,
    tags: tantivy::schema::Field,
}

fn build_schema() -> (Schema, Fields) {
    let mut sb = Schema::builder();
    // "raw" tokenizer: tantivy stores our tokens verbatim, since we tokenize upstream.
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("raw")
        .set_index_option(IndexRecordOption::WithFreqs);
    let opts = TextOptions::default().set_indexing_options(indexing);
    let words = sb.add_text_field("words", opts.clone());
    let bigrams = sb.add_text_field("bigrams", opts);
    // The item id, stored so a hit can be mapped back. Hyphenated UUID round-trips cleanly.
    let id = sb.add_text_field("id", STORED);
    // Tags, stored as a JSON array so a hit is filtered with the shared `TagFilter.matches`
    // primitive — one correctness path for every arm, rather than five tantivy queries.
    let tags = sb.add_text_field("tags", STORED);
    let schema = sb.build();
    (schema, Fields { words, bigrams, id, tags })
}

/// Turn a list of token texts into a tantivy pre-tokenized value. Offsets are synthetic
/// (we index with freqs, not positions), so only `position` and `text` matter.
fn pretokenized(tokens: &[String]) -> PreTokenizedString {
    let mut offset = 0;
    let toks: Vec<TantivyToken> = tokens
        .iter()
        .enumerate()
        .map(|(pos, t)| {
            let tok = TantivyToken {
                offset_from: offset,
                offset_to: offset + t.len(),
                position: pos,
                text: t.clone(),
                position_length: 1,
            };
            offset += t.len() + 1;
            tok
        })
        .collect();
    PreTokenizedString {
        text: tokens.join(" "),
        tokens: toks,
    }
}

/// Add one document to a tantivy writer. Shared by the batch [`TantivyFts::build_with_tags`]
/// and the streaming [`TantivyFtsBuilder`] so both index a document identically.
fn add_doc(
    writer: &mut tantivy::IndexWriter,
    fields: &Fields,
    tokenizer: &Tokenizer,
    id: MemoryId,
    text: &str,
    tags: &[String],
) -> tantivy::Result<()> {
    let tokens = tokenizer.tokenize(text);
    let mut words = Vec::new();
    let mut bigrams = Vec::new();
    for t in tokens {
        match t.field {
            Field::Words => words.push(t.text),
            Field::Bigrams => bigrams.push(t.text),
        }
    }
    let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
    writer.add_document(doc!(
        fields.words => pretokenized(&words),
        fields.bigrams => pretokenized(&bigrams),
        fields.id => id.as_uuid().to_string(),
        fields.tags => tags_json,
    ))?;
    Ok(())
}

/// A streaming FTS builder: `add` one document at a time, then `finish`. The batch
/// `build_with_tags` ties every document to one borrow lifetime (fine for an in-memory slice),
/// which the external-memory fold can't satisfy — it reads owned items off a disk spill — so
/// that path feeds this builder instead. tantivy itself spills to disk as it indexes.
pub struct TantivyFtsBuilder {
    dir: tempfile::TempDir,
    index: Index,
    writer: tantivy::IndexWriter,
    fields: Fields,
    tokenizer: Tokenizer,
    doc_count: usize,
}

impl TantivyFtsBuilder {
    /// `heap_bytes` caps the writer's in-memory arena before it flushes a segment to the temp dir,
    /// so the FTS stage's RAM is bounded by the caller's budget. tantivy needs a floor (~15 MB) to
    /// make progress, enforced here.
    pub fn new(tokenizer: Tokenizer, heap_bytes: usize) -> tantivy::Result<Self> {
        let (schema, fields) = build_schema();
        let dir = tempfile::tempdir().expect("create temp dir for tantivy split");
        let index = Index::create_in_dir(dir.path(), schema)?;
        let writer = index.writer(heap_bytes.max(15_000_000))?;
        Ok(Self { dir, index, writer, fields, tokenizer, doc_count: 0 })
    }

    pub fn add(&mut self, id: MemoryId, text: &str, tags: &[String]) -> tantivy::Result<()> {
        add_doc(&mut self.writer, &self.fields, &self.tokenizer, id, text, tags)?;
        self.doc_count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> tantivy::Result<TantivyFts> {
        self.writer.commit()?;
        let split_bytes = pack_dir(self.dir.path());
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(TantivyFts {
            _dir: self.dir,
            reader,
            schema_fields: self.fields,
            tokenizer: self.tokenizer,
            split_bytes,
            doc_count: self.doc_count,
        })
    }
}

/// A built, queryable tantivy FTS index.
///
/// Holds the open index plus the temp directory it was materialized into (the NVMe/mmap
/// tier). `split_bytes` is the packed form for persistence to object storage.
pub struct TantivyFts {
    _dir: tempfile::TempDir,
    reader: IndexReader,
    schema_fields: Fields,
    tokenizer: Tokenizer,
    split_bytes: Vec<u8>,
    doc_count: usize,
}

impl TantivyFts {
    /// Build an index over `(id, text)` documents with no tags. Convenience for callers
    /// (like the benchmark) whose corpus is untagged.
    pub fn build<'a>(
        docs: impl IntoIterator<Item = (MemoryId, &'a str)>,
        tokenizer: Tokenizer,
    ) -> tantivy::Result<Self> {
        Self::build_with_tags(docs.into_iter().map(|(id, text)| (id, text, &[] as &[String])), tokenizer)
    }

    /// Build an index over `(id, text, tags)` documents, tokenizing each with the shared
    /// chain and storing tags for filtering.
    pub fn build_with_tags<'a>(
        docs: impl IntoIterator<Item = (MemoryId, &'a str, &'a [String])>,
        tokenizer: Tokenizer,
    ) -> tantivy::Result<Self> {
        let (schema, fields) = build_schema();
        let dir = tempfile::tempdir().expect("create temp dir for tantivy split");
        let index = Index::create_in_dir(dir.path(), schema)?;

        let mut doc_count = 0;
        {
            let mut writer: tantivy::IndexWriter = index.writer(50_000_000)?;
            for (id, text, tags) in docs {
                add_doc(&mut writer, &fields, &tokenizer, id, text, tags)?;
                doc_count += 1;
            }
            writer.commit()?;
        }

        let split_bytes = pack_dir(dir.path());
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            _dir: dir,
            reader,
            schema_fields: fields,
            tokenizer,
            split_bytes,
            doc_count,
        })
    }

    /// Load an index from a packed split, materializing it into the local NVMe/mmap tier.
    pub fn from_split(split: &[u8], tokenizer: Tokenizer) -> tantivy::Result<Self> {
        let dir = tempfile::tempdir().expect("create temp dir for tantivy split");
        unpack_dir(split, dir.path());
        let index = Index::open_in_dir(dir.path())?;
        let schema = index.schema();
        let fields = Fields {
            words: schema.get_field("words").unwrap(),
            bigrams: schema.get_field("bigrams").unwrap(),
            id: schema.get_field("id").unwrap(),
            tags: schema.get_field("tags").unwrap(),
        };
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let doc_count = reader.searcher().num_docs() as usize;
        Ok(Self {
            _dir: dir,
            reader,
            schema_fields: fields,
            tokenizer,
            split_bytes: split.to_vec(),
            doc_count,
        })
    }

    /// The packed split bytes, for writing to object storage.
    pub fn split_bytes(&self) -> &[u8] {
        &self.split_bytes
    }

    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Unfiltered BM25 search (no tag filter).
    pub fn search(&self, query: &str, k: usize) -> Vec<FtsHit> {
        self.search_filtered(query, k, &TagFilter::none())
    }

    /// BM25 search with a tag filter. The query is tokenized with the same chain the
    /// documents were, each term a `Should` clause across both fields (dual-emission
    /// scoring, SPEC §8 step 4). A tag filter is applied by retrieving each hit's stored
    /// tags and running the shared [`TagFilter::matches`] — the same primitive every arm
    /// uses. To keep a selective filter from starving `k`, the search over-fetches before
    /// filtering.
    pub fn search_filtered(&self, query: &str, k: usize, filter: &TagFilter) -> Vec<FtsHit> {
        let terms = self.tokenizer.query_terms(query);
        if terms.is_empty() {
            return Vec::new();
        }

        let clauses: Vec<(Occur, Box<dyn Query>)> = terms
            .iter()
            .map(|t| {
                let field = match t.field {
                    Field::Words => self.schema_fields.words,
                    Field::Bigrams => self.schema_fields.bigrams,
                };
                let term = Term::from_field_text(field, &t.text);
                let q: Box<dyn Query> =
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
                (Occur::Should, q)
            })
            .collect();
        let query = BooleanQuery::new(clauses);

        // With any filter (flat tags OR a compound tag_groups predicate), over-fetch so
        // post-filtering still yields ~k. Capped so a huge corpus doesn't pull an unbounded
        // result set. This internal over-fetch is why the FTS arm needs no caller-side headroom
        // for tag filtering: it fills k *passing* hits itself.
        let limit = if filter.admits_all() {
            k
        } else {
            (k.saturating_mul(50)).clamp(k, 10_000)
        };

        let searcher = self.reader.searcher();
        let Ok(top) = searcher.search(&query, &TopDocs::with_limit(limit)) else {
            return Vec::new();
        };

        let mut hits = Vec::with_capacity(k);
        for (score, addr) in top {
            let Ok(doc) = searcher.doc::<TantivyDocument>(addr) else {
                continue;
            };
            if !filter.admits_all() {
                let tags = doc
                    .get_first(self.schema_fields.tags)
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                    .unwrap_or_default();
                // The flat condition and the compound tag_groups predicate both apply here,
                // AND-ed, over the hit's own stored tags — the same tags every arm filters on.
                if !filter.matches(&tags) || !filter.groups_match(&tags) {
                    continue;
                }
            }
            if let Some(id_str) = doc.get_first(self.schema_fields.id).and_then(|v| v.as_str()) {
                if let Ok(uuid) = Uuid::parse_str(id_str) {
                    hits.push(FtsHit { id: MemoryId::from(uuid), score });
                    if hits.len() >= k {
                        break;
                    }
                }
            }
        }
        hits
    }
}

/// Pack every file in a tantivy index directory into one blob:
/// `[u32 file_count]` then per file `[u32 name_len][name][u64 data_len][data]`.
/// File order is sorted, so the packed bytes are deterministic (supports G-6).
fn pack_dir(dir: &Path) -> Vec<u8> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("read tantivy dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    let mut out = Vec::new();
    out.extend((files.len() as u32).to_le_bytes());
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        let data = std::fs::read(&f).expect("read tantivy file");
        out.extend((name.len() as u32).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((data.len() as u64).to_le_bytes());
        out.extend(&data);
    }
    out
}

/// Reverse of [`pack_dir`]: write the split's files into a directory.
fn unpack_dir(bytes: &[u8], dir: &Path) {
    let mut p = 0usize;
    let read_u32 = |bytes: &[u8], p: &mut usize| {
        let v = u32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let read_u64 = |bytes: &[u8], p: &mut usize| {
        let v = u64::from_le_bytes(bytes[*p..*p + 8].try_into().unwrap());
        *p += 8;
        v
    };
    let count = read_u32(bytes, &mut p);
    for _ in 0..count {
        let nl = read_u32(bytes, &mut p) as usize;
        let name = std::str::from_utf8(&bytes[p..p + nl]).unwrap().to_string();
        p += nl;
        let dl = read_u64(bytes, &mut p) as usize;
        std::fs::write(dir.join(name), &bytes[p..p + dl]).expect("write tantivy file");
        p += dl;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    fn build(docs: &[(&str, &str)]) -> TantivyFts {
        let items: Vec<(MemoryId, &str)> =
            docs.iter().map(|(id, text)| (MemoryId::from_key(id), *text)).collect();
        TantivyFts::build(items, Tokenizer::default()).unwrap()
    }

    /// Where the FTS arm's per-query time actually goes.
    ///
    /// The fleet trace put the text arm at ~68% of query CPU once the load driver started sending
    /// real question text. This splits one `search_filtered` into (a) executing the BooleanQuery
    /// and (b) turning the resulting DocAddresses into `MemoryId`s — which today means a doc-store
    /// fetch per hit (`searcher.doc()`, a compressed-block decompression) plus a `Uuid::parse_str`,
    /// because `id` is a `STORED` text field with no fast/columnar field to read instead.
    ///
    /// Prints the split; not an assertion of a target, just the measurement that says which half to
    /// fix. Run with `--nocapture`.
    #[test]
    fn where_fts_query_time_goes() {
        use std::time::Instant;
        const DOCS: usize = 5_000;
        const QUERIES: usize = 100;
        const K: usize = 10;

        // Deterministic Zipf-ish corpus: common words plus a rare per-doc marker, so posting lists
        // have a realistic skew rather than every term matching everything.
        let vocab: Vec<String> = (0..800).map(|i| format!("term{i}")).collect();
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        let texts: Vec<String> = (0..DOCS)
            .map(|d| {
                let mut w: Vec<&str> = Vec::with_capacity(200);
                for _ in 0..200 {
                    // Skewed pick: squaring biases toward the front of the vocab.
                    let r = next() % vocab.len();
                    w.push(&vocab[(r * r) / vocab.len()]);
                }
                format!("doc{d} {}", w.join(" "))
            })
            .collect();
        let items: Vec<(MemoryId, &str)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (MemoryId::from_key(&format!("doc-{i}")), t.as_str()))
            .collect();
        let idx = TantivyFts::build(items, Tokenizer::default()).unwrap();

        // ~15-word questions, the shape a real query has.
        let queries: Vec<String> = (0..QUERIES)
            .map(|_| {
                let mut w: Vec<&str> = Vec::with_capacity(15);
                for _ in 0..15 {
                    let r = next() % vocab.len();
                    w.push(&vocab[(r * r) / vocab.len()]);
                }
                w.join(" ")
            })
            .collect();

        let filter = mlake_core::TagFilter::new(Vec::new(), mlake_core::TagsMatch::Any);
        for q in &queries {
            let _ = idx.search_filtered(q, K, &filter); // warm mmap/caches
        }

        let t = Instant::now();
        let mut got = 0usize;
        for q in &queries {
            got += idx.search_filtered(q, K, &filter).len();
        }
        let full = t.elapsed();

        // (a) query execution only — no doc-store access.
        let searcher = idx.reader.searcher();
        let mut addrs = Vec::new();
        let t = Instant::now();
        for q in &queries {
            let terms = idx.tokenizer.query_terms(q);
            let clauses: Vec<(Occur, Box<dyn Query>)> = terms
                .iter()
                .map(|tk| {
                    let field = match tk.field {
                        Field::Words => idx.schema_fields.words,
                        Field::Bigrams => idx.schema_fields.bigrams,
                    };
                    let q: Box<dyn Query> = Box::new(TermQuery::new(
                        Term::from_field_text(field, &tk.text),
                        IndexRecordOption::WithFreqs,
                    ));
                    (Occur::Should, q)
                })
                .collect();
            let top = searcher.search(&BooleanQuery::new(clauses), &TopDocs::with_limit(K)).unwrap();
            addrs.push(top);
        }
        let exec = t.elapsed();

        // (b) hit -> MemoryId only: the doc-store fetch + uuid parse the arm does per hit.
        let t = Instant::now();
        let mut ids = 0usize;
        for top in &addrs {
            for (_score, addr) in top {
                let doc = searcher.doc::<TantivyDocument>(*addr).unwrap();
                if let Some(s) = doc.get_first(idx.schema_fields.id).and_then(|v| v.as_str()) {
                    if Uuid::parse_str(s).is_ok() {
                        ids += 1;
                    }
                }
            }
        }
        let hydrate = t.elapsed();

        // (c0) BUILDING an index for an empty tail. `FactType::tail_fts` is a fresh OnceLock on
        // every snapshot, so before the arm learned to skip an empty tail this was paid on the
        // first text query after every fold — a temp dir, a 50 MB writer arena with indexing
        // threads, a commit and a split pack, all to index zero documents.
        let t = Instant::now();
        let empty = TantivyFts::build(Vec::<(MemoryId, &str)>::new(), Tokenizer::default()).unwrap();
        let empty_build = t.elapsed();
        for q in &queries {
            let _ = empty.search_filtered(q, K, &filter);
        }
        let t = Instant::now();
        for q in &queries {
            let _ = empty.search_filtered(q, K, &filter);
        }
        let empty_search = t.elapsed();

        // (d) tokenization alone — paid once per index searched, so once per segment + the tail.
        let t = Instant::now();
        let mut terms_total = 0usize;
        for q in &queries {
            terms_total += idx.tokenizer.query_terms(q).len();
        }
        let tokenize = t.elapsed();

        let per = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / QUERIES as f64;
        println!(
            "fts {DOCS} docs, {QUERIES} queries, k={K}, {:.1} terms/query:\n  \
             full          {:.3} ms/q\n  \
             exec          {:.3} ms/q\n  \
             id-hydrate    {:.3} ms/q  ({:.1}% of full)\n  \
             EMPTY-tail    {:.3} ms/q  (search only; the arm now skips an empty tail entirely)\n  \
             tokenize      {:.3} ms/q  (paid per index searched: tail + each segment)\n  \
             EMPTY BUILD   {:.1} ms ONE-OFF  (per snapshot, before the empty-tail skip)\n  \
             [{got} hits, {ids} ids]",
            terms_total as f64 / QUERIES as f64,
            per(full),
            per(exec),
            per(hydrate),
            100.0 * hydrate.as_secs_f64() / full.as_secs_f64().max(1e-9),
            per(empty_search),
            per(tokenize),
            empty_build.as_secs_f64() * 1000.0,
        );
    }

    /// Reading `words` only (instead of `words` + the duplicate `bigrams` posting list) must not
    /// change what a Latin query returns, nor in what order.
    ///
    /// Indexing writes each Latin token to both fields with identical content, so both fields have
    /// identical postings and lengths: every document's score is scaled by the same constant when
    /// the duplicate clause is dropped, leaving the ranking fixed. This pins that against the
    /// both-fields query the arm used to issue.
    #[test]
    fn dropping_the_duplicate_latin_field_preserves_ranking() {
        let docs: Vec<(String, String)> = (0..200)
            .map(|i| {
                (
                    format!("d{i}"),
                    format!(
                        "alpha beta gamma delta epsilon doc{i} {}",
                        vec!["zeta eta theta"; (i % 7) + 1].join(" ")
                    ),
                )
            })
            .collect();
        let items: Vec<(MemoryId, &str)> =
            docs.iter().map(|(id, t)| (MemoryId::from_key(id), t.as_str())).collect();
        let idx = TantivyFts::build(items, Tokenizer::default()).unwrap();
        let filter = mlake_core::TagFilter::new(Vec::new(), mlake_core::TagsMatch::Any);

        for q in ["alpha zeta", "beta theta eta", "gamma delta epsilon zeta"] {
            // What the arm does now: `words` only.
            let now: Vec<MemoryId> =
                idx.search_filtered(q, 20, &filter).into_iter().map(|h| h.id).collect();

            // What it used to do: the same terms against BOTH fields.
            let searcher = idx.reader.searcher();
            let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
            for tk in idx.tokenizer.query_terms(q) {
                for field in [idx.schema_fields.words, idx.schema_fields.bigrams] {
                    clauses.push((
                        Occur::Should,
                        Box::new(TermQuery::new(
                            Term::from_field_text(field, &tk.text),
                            IndexRecordOption::WithFreqs,
                        )) as Box<dyn Query>,
                    ));
                }
            }
            let top = searcher
                .search(&BooleanQuery::new(clauses), &TopDocs::with_limit(20))
                .unwrap();
            let before: Vec<MemoryId> = top
                .into_iter()
                .filter_map(|(_, addr)| {
                    let doc = searcher.doc::<TantivyDocument>(addr).ok()?;
                    let s = doc.get_first(idx.schema_fields.id)?.as_str()?;
                    Uuid::parse_str(s).ok().map(MemoryId::from)
                })
                .collect();

            assert_eq!(now, before, "ranking changed for query {q:?}");
            assert!(!now.is_empty(), "query {q:?} matched nothing");
        }
    }

    #[test]
    fn finds_documents_by_term() {
        let idx = build(&[
            ("a", "the quick brown fox"),
            ("b", "a lazy dog sleeps"),
            ("c", "the fox and the dog"),
        ]);
        let ids: Vec<_> = idx.search("fox", 10).into_iter().map(|h| h.id).collect();
        assert!(ids.contains(&MemoryId::from_key("a")));
        assert!(ids.contains(&MemoryId::from_key("c")));
        assert!(!ids.contains(&MemoryId::from_key("b")));
    }

    #[test]
    fn rarer_terms_score_higher() {
        let idx = build(&[("common", "the the the the"), ("rare", "the unicorn")]);
        let hits = idx.search("the unicorn", 10);
        assert_eq!(hits[0].id, MemoryId::from_key("rare"), "idf should favour the rare match");
    }

    #[test]
    fn chinese_query_retrieves_chinese_doc() {
        let idx = build(&[
            ("cn", "北京大学是一所著名的大学"),
            ("other", "the weather is nice today"),
        ]);
        let hits = idx.search("北京大学", 10);
        assert_eq!(hits[0].id, MemoryId::from_key("cn"));
    }

    #[test]
    fn traditional_query_retrieves_simplified_doc() {
        // G-4 end to end: query in traditional script, document in simplified.
        let idx = build(&[("doc", "我在学习中文和数学")]);
        let hits = idx.search("數學", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, MemoryId::from_key("doc"));
    }

    #[test]
    fn oov_span_found_via_bigrams() {
        let idx = build(&[("target", "购买闪迪存储卡"), ("noise", "今天天气很好")]);
        let hits = idx.search("存储卡", 10);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, MemoryId::from_key("target"));
    }

    #[test]
    fn empty_query_returns_nothing() {
        let idx = build(&[("a", "hello world")]);
        assert!(idx.search("", 10).is_empty());
    }

    #[test]
    fn split_roundtrips_through_bytes() {
        let idx = build(&[("a", "the quick brown fox"), ("b", "a lazy dog")]);
        let split = idx.split_bytes().to_vec();

        let reloaded = TantivyFts::from_split(&split, Tokenizer::default()).unwrap();
        assert_eq!(reloaded.doc_count(), 2);
        // Same query gives the same top hit after a store round-trip.
        let before = idx.search("fox", 10);
        let after = reloaded.search("fox", 10);
        assert_eq!(before[0].id, after[0].id);
    }

    #[test]
    fn retrieval_is_deterministic_across_rebuilds() {
        // tantivy stamps each segment with a random UUID, so the split *bytes* are not
        // byte-identical across builds — the strong form of G-6 does not hold for the FTS
        // split (unlike the vector/pk/radj files, which do). What must hold, and does, is
        // that retrieval *results* are identical: the same corpus answers the same query
        // the same way regardless of the random segment ids.
        let a = build(&[("x", "alpha beta gamma"), ("y", "beta gamma delta"), ("z", "gamma delta")]);
        let b = build(&[("x", "alpha beta gamma"), ("y", "beta gamma delta"), ("z", "gamma delta")]);
        let ra: Vec<_> = a.search("gamma delta", 10).into_iter().map(|h| h.id).collect();
        let rb: Vec<_> = b.search("gamma delta", 10).into_iter().map(|h| h.id).collect();
        assert_eq!(ra, rb, "retrieval order must be reproducible even if split bytes are not");
    }
}
