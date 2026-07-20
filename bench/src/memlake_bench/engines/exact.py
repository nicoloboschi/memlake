"""Reference baseline: exact numpy dense kNN + in-python BM25 + RRF.

This is the correctness ceiling for the dense arm (brute force, no ANN) and a
harness sanity check that does not depend on Qdrant at all. If Qdrant's dense
numbers differ materially from these, the Qdrant path is misconfigured.
"""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from .. import metrics
from ..bm25 import BM25
from ..datasets import Beir
from ..embed import Embeddings

TOP_K = 100


def _dense_search(emb: Embeddings, top_k: int) -> tuple[metrics.Run, list[float]]:
    """Exact cosine kNN. Vectors are unit-norm, so cosine == dot product."""
    run: metrics.Run = {}
    lat: list[float] = []
    corpus = emb.corpus  # (N, d)
    ids = emb.corpus_ids

    for i, qid in enumerate(tqdm(emb.query_ids, desc="exact dense", unit="q")):
        t0 = time.perf_counter()
        sims = corpus @ emb.queries[i]
        k = min(top_k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        lat.append((time.perf_counter() - t0) * 1000.0)
        run[qid] = [ids[j] for j in idx]
    return run, lat


def _sparse_search(
    beir: Beir, bm25: BM25, top_k: int
) -> tuple[metrics.Run, list[float]]:
    run: metrics.Run = {}
    lat: list[float] = []
    for qid, qtext in zip(
        beir.query_ids, tqdm(beir.query_texts, desc="exact bm25", unit="q")
    ):
        t0 = time.perf_counter()
        idx, _ = bm25.top_k(qtext, top_k)
        lat.append((time.perf_counter() - t0) * 1000.0)
        run[qid] = [beir.corpus_ids[j] for j in idx]
    return run, lat


def run(beir: Beir, emb: Embeddings, top_k: int = TOP_K, rrf_k: int = 60) -> dict:
    if emb.corpus_ids != beir.corpus_ids:
        raise ValueError("embedding cache corpus ids do not match the loaded dataset")

    dense_run, dense_lat = _dense_search(emb, top_k)

    bm25 = BM25(beir.corpus_texts)
    sparse_run, sparse_lat = _sparse_search(beir, bm25, top_k)

    hybrid_run = metrics.rrf_fuse([dense_run, sparse_run], k=rrf_k, top_k=top_k)
    # Fusion is arithmetic over both arms; charge it the sum of arm latencies.
    hybrid_lat = [d + s for d, s in zip(dense_lat, sparse_lat)]

    return {
        "engine": "exact",
        "config": {
            "dense": "numpy brute-force cosine (exact)",
            "sparse": "in-python Okapi BM25 (k1=0.9, b=0.4, stopwords + s-stemmer)",
            "fusion": f"RRF k={rrf_k}",
            "top_k": top_k,
            "model": emb.meta.get("model"),
            "dim": emb.dim,
        },
        "corpus_size": beir.n_docs,
        "n_queries": beir.n_queries,
        "arms": {
            "dense": metrics.evaluate(dense_run, beir.qrels, dense_lat),
            "sparse": metrics.evaluate(sparse_run, beir.qrels, sparse_lat),
            "hybrid": metrics.evaluate(hybrid_run, beir.qrels, hybrid_lat),
        },
    }
