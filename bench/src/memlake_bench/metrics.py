"""Retrieval metrics against BEIR qrels + latency percentiles.

Definitions follow the BEIR/pytrec_eval conventions so numbers are comparable
to published leaderboards:

  nDCG@k    graded gains (2^rel - 1 is NOT used; BEIR uses linear rel via
            pytrec_eval's ndcg_cut which applies gain = rel). Ideal DCG is
            computed from the full sorted qrel list truncated at k.
  Recall@k  |retrieved_at_k ∩ relevant| / |relevant|  (binary: rel > 0)
  MRR@k     1 / rank of first relevant doc within top-k, else 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Run = dict[str, list[str]]  # query_id -> ranked doc_ids (best first)
Qrels = dict[str, dict[str, int]]


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    scores = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        gains = [float(rel.get(d, 0)) for d in ranked[:k]]
        ideal = sorted((float(v) for v in rel.values()), reverse=True)[:k]
        idcg = _dcg(ideal)
        scores.append(_dcg(gains) / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(run: Run, qrels: Qrels, k: int = 100) -> float:
    scores = []
    for qid, ranked in run.items():
        rel = {d for d, v in qrels.get(qid, {}).items() if v > 0}
        if not rel:
            continue
        hit = len(rel & set(ranked[:k]))
        scores.append(hit / len(rel))
    return float(np.mean(scores)) if scores else 0.0


def mrr_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    scores = []
    for qid, ranked in run.items():
        rel = {d for d, v in qrels.get(qid, {}).items() if v > 0}
        if not rel:
            continue
        rr = 0.0
        for i, d in enumerate(ranked[:k]):
            if d in rel:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}
    a = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "n_queries": int(a.size),
    }


def evaluate(run: Run, qrels: Qrels, latencies_ms: list[float] | None = None) -> dict:
    return {
        "ndcg@10": round(ndcg_at_k(run, qrels, 10), 5),
        "recall@100": round(recall_at_k(run, qrels, 100), 5),
        "mrr@10": round(mrr_at_k(run, qrels, 10), 5),
        "latency": {k: round(v, 3) for k, v in latency_stats(latencies_ms or []).items()},
    }


# ---------------------------------------------------------------- RRF fusion


def rrf_fuse(runs: list[Run], k: int = 60, top_k: int = 100) -> Run:
    """Reciprocal Rank Fusion over N ranked lists. score = sum 1/(k + rank)."""
    fused: Run = {}
    qids = {q for r in runs for q in r}
    for qid in qids:
        acc: dict[str, float] = {}
        for r in runs:
            for rank, doc in enumerate(r.get(qid, []), start=1):
                acc[doc] = acc.get(doc, 0.0) + 1.0 / (k + rank)
        fused[qid] = [d for d, _ in sorted(acc.items(), key=lambda x: -x[1])[:top_k]]
    return fused


@dataclass
class ArmResult:
    """Metrics for one retrieval arm (dense / sparse / hybrid)."""

    name: str
    run: Run = field(repr=False, default_factory=dict)
    latencies_ms: list[float] = field(repr=False, default_factory=list)

    def evaluate(self, qrels: Qrels) -> dict:
        return evaluate(self.run, qrels, self.latencies_ms)
