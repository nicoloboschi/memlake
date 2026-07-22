//! Adaptive probing through the real read path, with the switch on.
//!
//! `MEMLAKE_ADAPTIVE_PROBE` is read once into a `OnceLock`, so the on-state cannot be tested in
//! the same process as the off-state. This file is its own test binary and holds exactly one
//! test, which sets the variable before anything reads it.
//!
//! What it pins down is the pair of properties the stopping rule is only allowed to have:
//! it must not cost a roundtrip beyond the budget (INV-7), and it must not lose a hit the
//! fixed-fraction probe would have found. Whether it *saves* anything is a measurement, not an
//! invariant — see docs/arms/vector.md, where the answer on BEIR is "nothing".

use mlake_core::memory::Timestamps;
use mlake_core::{Memory, MemoryId, Op};
use mlake_fts::Tokenizer;
use mlake_index::{index, ArmDepths, IndexOptions, QueryNode};
use mlake_store::Store;
use mlake_wal::{Namespace, Writer};

fn item(key: &str, vector: Vec<f32>) -> Memory {
    Memory {
        id: MemoryId::from_key(key),
        vector,
        text: key.to_string(),
        index_text: String::new(),
        memory_type: 1,
        tags: vec![],
        timestamps: Timestamps::default(),
        proof_count: 0,
        entity_ids: vec![],
        causal_out: vec![],
        metadata: vec![],
    }
}

#[tokio::test]
async fn adaptive_probing_keeps_the_hits_and_the_budget() {
    std::env::set_var("MEMLAKE_ADAPTIVE_PROBE", "1");

    let store = Store::in_memory();
    let ns = Namespace::new("adaptive", store);
    ns.create_if_absent(&Tokenizer::default().config_hash()).await.unwrap();
    let mut writer = Writer::new(ns.clone());

    // A spread-out corpus with many clusters, so the probe set is big enough to have a tail
    // for the second wave to decide on.
    let n = 1_200;
    let mut vectors: Vec<(MemoryId, Vec<f32>)> = Vec::with_capacity(n);
    let mut ops = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f32 * 0.137;
        let b = i as f32 * 0.041;
        let mut v = vec![a.cos() * b.cos(), a.sin() * b.cos(), b.sin(), (a * 0.3).cos()];
        mlake_core::normalize(&mut v);
        let key = format!("m{i}");
        vectors.push((MemoryId::from_key(&key), v.clone()));
        ops.push(Op::Upsert(item(&key, v)));
    }
    writer.commit(ops).await.unwrap();
    // F32 so the scan's own error cannot be confused with the probe's.
    let opts = IndexOptions { vector_codec: mlake_ivf::VectorCodec::F32, ..IndexOptions::default() };
    index(&ns, &Tokenizer::default(), opts).await.unwrap();

    let node = QueryNode::open(&ns, Tokenizer::default()).await.unwrap();
    let (manifest, _) = ns.read_manifest().await.unwrap();
    let cluster_count = manifest.index(1).unwrap().files.clusters.len();
    assert!(cluster_count >= 20, "test needs a real probe set, got {cluster_count} clusters");

    const DEPTH: usize = 20;
    let depths = ArmDepths { vector: DEPTH, text: 0, graph: 0, nprobe: 0 };
    let tags = mlake_core::TagFilter::none();

    for qi in [0usize, 137, 601, 999] {
        let q = &vectors[qi].1;
        let metrics = mlake_store::QueryMetrics::new();
        let hits = node
            .query_raw_metered(1, Some(q), None, &tags, depths, None, Default::default(), &metrics)
            .await
            .unwrap();

        assert!(
            metrics.within_budget(),
            "adaptive probing must stay inside the {} roundtrip budget, used {}",
            mlake_store::COLD_ROUNDTRIP_BUDGET,
            metrics.roundtrips()
        );
        // Two waves at most: each cluster is a `.vec` object here, and the union of the waves
        // can never exceed the fixed-fraction probe set it is a subset of.
        let nprobe_max = cluster_count.div_ceil(2).clamp(8, 64);
        assert!(
            metrics.requests() <= nprobe_max,
            "adaptive fetched {} objects for at most {nprobe_max} clusters",
            metrics.requests()
        );

        // Brute force over the whole corpus: the query's own vector is its nearest neighbour,
        // and the top of the exact ranking must survive the stopping rule.
        let mut truth: Vec<(MemoryId, f32)> =
            vectors.iter().map(|(id, v)| (*id, mlake_core::cosine(q, v))).collect();
        truth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let found: std::collections::HashSet<MemoryId> = hits.iter().map(|h| h.id).collect();
        let recovered = truth[..10].iter().filter(|(id, _)| found.contains(id)).count();
        assert!(
            recovered >= 9,
            "adaptive probing lost {} of the true top 10 for query {qi}",
            10 - recovered
        );
        assert!(found.contains(&vectors[qi].0), "a memory must find itself");
    }
}
