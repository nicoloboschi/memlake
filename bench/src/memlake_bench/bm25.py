"""Minimal Okapi BM25 over an in-memory inverted index.

Used by the `exact` reference baseline only. Deliberately dependency-free
(no Anserini/Lucene) so it validates the harness independently. Expect scores
a point or two below Anserini BM25 since stemming is light.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from tqdm import tqdm

_TOKEN = re.compile(r"[a-z0-9]+")

# Standard Lucene/BEIR English stopword list.
STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such that the
    their then there these they this to was will with i me my we our you your he him his
    she her they them what which who whom would could should have has had do does did been
    being from about above after again against all am any because before below between both
    can down during each few further here how more most other over own same so some than
    too under until up very were when where why""".split()
)


def s_stem(tok: str) -> str:
    """Light plural stripper (Lucene's SStemmer). Cheap, safe, most of the win."""
    if len(tok) > 3:
        if tok.endswith("ies") and not tok.endswith(("eies", "aies")):
            return tok[:-3] + "y"
        if tok.endswith("es") and not tok.endswith(("aes", "ees", "oes")):
            return tok[:-1]
        if tok.endswith("s") and not tok.endswith(("us", "ss")):
            return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [s_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    """Okapi BM25 (k1=0.9, b=0.4 — the BEIR/Anserini defaults)."""

    def __init__(self, docs: list[str], k1: float = 0.9, b: float = 0.4, progress: bool = True):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)

        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len = np.zeros(self.n_docs, dtype=np.float32)

        it = enumerate(docs)
        if progress:
            it = tqdm(it, total=len(docs), desc="bm25 index", unit="doc")
        for i, text in it:
            toks = tokenize(text)
            doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((i, tf))

        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0

        # Freeze postings into numpy arrays + precompute idf per term.
        self.doc_ids: dict[str, np.ndarray] = {}
        self.tfs: dict[str, np.ndarray] = {}
        self.idf: dict[str, float] = {}
        for term, plist in postings.items():
            ids = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            tf = np.fromiter((t for _, t in plist), dtype=np.float32, count=len(plist))
            self.doc_ids[term] = ids
            self.tfs[term] = tf
            df = len(plist)
            # Lucene/Robertson idf, always positive.
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Precompute the length-normalization denominator component.
        self._len_norm = self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))

    def score(self, query: str) -> np.ndarray:
        """Dense score vector over all docs (0 for docs with no query term)."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in tokenize(query):
            ids = self.doc_ids.get(term)
            if ids is None:
                continue
            tf = self.tfs[term]
            contrib = self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + self._len_norm[ids])
            # Doc ids are unique within a postings list, so plain fancy-index
            # accumulation is correct here and much faster than np.add.at.
            scores[ids] += contrib
        return scores

    def top_k(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score(query)
        k = min(k, self.n_docs)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return idx, scores[idx]
