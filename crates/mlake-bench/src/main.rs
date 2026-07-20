//! memlake BEIR benchmark runner.
//!
//! Loads the embedding cache the Python harness produced (so the vectors are byte-for-byte
//! the ones Qdrant scored), builds a memlake generation over the corpus, runs each query
//! through the vector, FTS, and fused arms, and writes the per-query rankings to a run
//! file. The Python harness scores that run with the same metric code it uses for Qdrant,
//! so any difference in the numbers is a difference in retrieval, not in measurement.
//!
//! Usage: mlake-bench <dataset> [testdata_dir] [out_file]

mod npy;

use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use mlake_core::item::Timestamps;
use mlake_core::{ItemId, StoredItem};
use mlake_fts::{Bm25Params, Tokenizer};
use mlake_index::{Engine, QueryConfig};
use serde::Serialize;

/// Output run for one arm: per-query ranked external ids, plus per-query latency.
#[derive(Serialize, Default)]
struct ArmRun {
    run: HashMap<String, Vec<String>>,
    latencies_ms: Vec<f64>,
}

#[derive(Serialize)]
struct BenchOutput {
    engine: &'static str,
    dataset: String,
    corpus_size: usize,
    n_queries: usize,
    index_seconds: f64,
    config: BenchConfig,
    arms: HashMap<String, ArmRun>,
}

#[derive(Serialize)]
struct BenchConfig {
    nprobe: usize,
    rrf_k: f32,
    vector_weight: f32,
    fts_weight: f32,
    arm_depth: usize,
    bm25_k1: f32,
    bm25_b: f32,
    model: String,
    dim: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: mlake-bench <dataset> [testdata_dir] [out_file]");
        std::process::exit(2);
    }
    let dataset = &args[1];
    let testdata = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "testdata".into()));
    let out_file = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("bench/results/{dataset}/memlake.run.json"));

    // Tuning knobs are overridable from the environment so the accuracy sweep can drive
    // this binary without a recompile.
    let config = QueryConfig {
        nprobe: env_usize("MEMLAKE_NPROBE", 32),
        rrf_k: env_f32("MEMLAKE_RRF_K", 60.0),
        vector_weight: env_f32("MEMLAKE_VEC_WEIGHT", 1.0),
        fts_weight: env_f32("MEMLAKE_FTS_WEIGHT", 1.0),
        graph_weight: 1.0,
        arm_depth: env_usize("MEMLAKE_ARM_DEPTH", 100),
        bm25: Bm25Params {
            k1: env_f32("MEMLAKE_BM25_K1", 1.2),
            b: env_f32("MEMLAKE_BM25_B", 0.75),
        },
    };

    let emb_dir = testdata.join("embeddings").join(dataset);
    let beir_dir = testdata.join("beir").join(dataset);

    eprintln!("[mlake-bench] loading corpus vectors + text for {dataset}");
    let corpus_vecs = npy::read_f32_matrix(&emb_dir.join("corpus.npy"))?;
    let corpus_ids: Vec<String> = read_json_array(&emb_dir.join("corpus_ids.json"))?;
    let doc_text = read_corpus_text(&beir_dir.join("corpus.jsonl"))?;

    anyhow::ensure!(
        corpus_vecs.rows == corpus_ids.len(),
        "corpus vector/id count mismatch: {} vs {}",
        corpus_vecs.rows,
        corpus_ids.len()
    );

    // Assemble stored items. The ItemId is derived from the external id so results can be
    // mapped back for scoring.
    let mut items = Vec::with_capacity(corpus_vecs.rows);
    let mut id_to_ext: HashMap<ItemId, String> = HashMap::new();
    for (i, ext_id) in corpus_ids.iter().enumerate() {
        let item_id = ItemId::from_key(ext_id);
        id_to_ext.insert(item_id, ext_id.clone());
        let text = doc_text.get(ext_id).cloned().unwrap_or_default();
        items.push(StoredItem {
            id: item_id,
            vector: corpus_vecs.row(i).to_vec(),
            text,
            fact_type: 1,
            tags: vec![],
            timestamps: Timestamps::default(),
            proof_count: 0,
            entity_ids: vec![],
            semantic_out: vec![],
            causal_out: vec![],
        });
    }

    eprintln!("[mlake-bench] building index over {} docs", items.len());
    let build_start = Instant::now();
    let engine = Engine::build(items, Tokenizer::default());
    let index_seconds = build_start.elapsed().as_secs_f64();

    // Queries.
    let query_vecs = npy::read_f32_matrix(&emb_dir.join("queries.npy"))?;
    let query_ids: Vec<String> = read_json_array(&emb_dir.join("queries_ids.json"))?;
    let query_text = read_query_text(&beir_dir.join("queries.jsonl"))?;
    anyhow::ensure!(query_vecs.rows == query_ids.len(), "query vector/id count mismatch");

    eprintln!("[mlake-bench] running {} queries", query_vecs.rows);
    let mut dense = ArmRun::default();
    let mut sparse = ArmRun::default();
    let mut hybrid = ArmRun::default();

    for (i, qid) in query_ids.iter().enumerate() {
        let qvec = query_vecs.row(i);
        let qtext = query_text.get(qid).map(String::as_str).unwrap_or("");

        run_arm(&engine, Some(qvec), None, &config, qid, &id_to_ext, &mut dense);
        run_arm(&engine, None, Some(qtext), &config, qid, &id_to_ext, &mut sparse);
        run_arm(
            &engine,
            Some(qvec),
            Some(qtext),
            &config,
            qid,
            &id_to_ext,
            &mut hybrid,
        );
    }

    let mut arms = HashMap::new();
    arms.insert("dense".to_string(), dense);
    arms.insert("sparse".to_string(), sparse);
    arms.insert("hybrid".to_string(), hybrid);

    let output = BenchOutput {
        engine: "memlake",
        dataset: dataset.clone(),
        corpus_size: corpus_vecs.rows,
        n_queries: query_vecs.rows,
        index_seconds,
        config: BenchConfig {
            nprobe: config.nprobe,
            rrf_k: config.rrf_k,
            vector_weight: config.vector_weight,
            fts_weight: config.fts_weight,
            arm_depth: config.arm_depth,
            bm25_k1: config.bm25.k1,
            bm25_b: config.bm25.b,
            model: "BAAI/bge-small-en-v1.5".into(),
            dim: corpus_vecs.cols,
        },
        arms,
    };

    if let Some(parent) = Path::new(&out_file).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_file, serde_json::to_vec_pretty(&output)?)?;
    eprintln!("[mlake-bench] wrote run to {out_file}");
    Ok(())
}

/// Run one arm for one query, recording the ranked external ids and the latency.
fn run_arm(
    engine: &Engine,
    qvec: Option<&[f32]>,
    qtext: Option<&str>,
    config: &QueryConfig,
    qid: &str,
    id_to_ext: &HashMap<ItemId, String>,
    arm: &mut ArmRun,
) {
    let start = Instant::now();
    let hits = engine.query(qvec, qtext, 100, *config);
    arm.latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    let ranked: Vec<String> = hits
        .iter()
        .filter_map(|h| id_to_ext.get(&h.id).cloned())
        .collect();
    arm.run.insert(qid.to_string(), ranked);
}

fn read_json_array(path: &Path) -> Result<Vec<String>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// BEIR corpus: one JSON object per line, `_id`/`title`/`text`. Document text follows the
/// BEIR convention of "title text", matching what the embedding cache used.
fn read_corpus_text(path: &Path) -> Result<HashMap<String, String>> {
    let file = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = HashMap::new();
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let title = v["title"].as_str().unwrap_or_default();
        let text = v["text"].as_str().unwrap_or_default();
        let combined = if title.is_empty() {
            text.to_string()
        } else {
            format!("{title} {text}")
        };
        out.insert(id, combined);
    }
    Ok(out)
}

fn read_query_text(path: &Path) -> Result<HashMap<String, String>> {
    let file = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = HashMap::new();
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let text = v["text"].as_str().unwrap_or_default().to_string();
        out.insert(id, text);
    }
    Ok(out)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
