"""Retrieval metrics against BEIR qrels + latency percentiles.

Definitions follow the BEIR/pytrec_eval conventions so numbers are comparable
to published leaderboards:

  nDCG@k    graded gains (2^rel - 1 is NOT used; BEIR uses linear rel via
            pytrec_eval's ndcg_cut which applies gain = rel). Ideal DCG is
            computed from the full sorted qrel list truncated at k.
  Recall@k  |retrieved_at_k ∩ relevant| / |relevant|  (binary: rel > 0)
  MRR@k     1 / rank of first relevant doc within top-k, else 0.

There is a second, different question these cannot answer: not "did we return
relevant documents" but "did the index return what an exhaustive scan would
have". That is `ann_recall_at_k`, and it is the metric that moves when
quantization, nprobe or clustering change — a lossy codec can hold nDCG steady
while quietly failing to find what brute force finds, because the documents it
drops were never in the qrels to begin with.

turbopuffer measure exactly this continuously in production, sampling 1% of live
queries against an exhaustive search and holding recall@10 above 90-95%. We run
it over the whole query set in the benchmark instead, which is the same
measurement without the sampling.
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


def ann_recall_at_k(run: Run, truth: Run, k: int = 10) -> float:
    """Fraction of the exhaustive top-k that the index actually returned.

    `truth` is a run produced by brute force over the whole corpus, so this
    measures index fidelity alone — it is independent of whether those documents
    were relevant, and independent of the embedding model's quality. A perfect
    score means the approximation cost nothing; it does not mean the results are
    good, which is what the qrels metrics are for.

    Queries absent from `truth`, or with an empty exhaustive result, are skipped
    rather than scored 0 — they carry no information about the index.
    """
    scores = []
    for qid, ideal in truth.items():
        gold = set(ideal[:k])
        if not gold:
            continue
        got = set(run.get(qid, [])[:k])
        scores.append(len(gold & got) / len(gold))
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


def evaluate(
    run: Run,
    qrels: Qrels,
    latencies_ms: list[float] | None = None,
    truth: Run | None = None,
) -> dict:
    """Quality against the qrels, plus — when `truth` is given — fidelity against an
    exhaustive scan. The two answer different questions and can move independently."""
    out = {
        "ndcg@10": round(ndcg_at_k(run, qrels, 10), 5),
        "recall@100": round(recall_at_k(run, qrels, 100), 5),
        "mrr@10": round(mrr_at_k(run, qrels, 10), 5),
        "latency": {k: round(v, 3) for k, v in latency_stats(latencies_ms or []).items()},
    }
    if truth is not None:
        # Reported at several k: a lossy codec typically holds the deep set and loses
        # ordering at the head, so ann_recall@10 degrades well before @100 does.
        for k in (1, 10, 100):
            out[f"ann_recall@{k}"] = round(ann_recall_at_k(run, truth, k), 5)
    return out


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
