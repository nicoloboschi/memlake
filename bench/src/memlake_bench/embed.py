"""Embedding cache: THE single source of truth for vectors.

Every engine (numpy exact, Qdrant, and later the Rust memlake engine) must read
the exact same vectors from here. Nothing re-embeds at query time.

On-disk layout, testdata/embeddings/{dataset}/:

    meta.json          model / dim / counts / normalization / query prefix
    corpus.npy         float32, shape (n_docs, dim),    C-order, L2-normalized
    corpus_ids.json    list[str], length n_docs   -- row i of corpus.npy
    queries.npy        float32, shape (n_queries, dim), C-order, L2-normalized
    queries_ids.json   list[str], length n_queries -- row i of queries.npy

The .npy files are plain NumPy v1 arrays with no pickled objects, so they are
readable from Rust (e.g. the `ndarray-npy` crate) or by parsing the 128-byte
header directly. Row order is the contract: `corpus_ids.json[i]` is the BEIR
document id for `corpus.npy[i]`.

Vectors are L2-normalized at write time, so cosine similarity == dot product
and no engine needs to renormalize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .datasets import Beir
from .paths import embeddings_dir

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384

# bge-* retrieval models are trained with an asymmetric query instruction.
# Omitting it costs several nDCG points, so it is part of the cache contract.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FORMAT_VERSION = 1


@dataclass
class Embeddings:
    corpus: np.ndarray  # (n_docs, dim) float32, L2-normalized
    corpus_ids: list[str]
    queries: np.ndarray  # (n_queries, dim) float32, L2-normalized
    query_ids: list[str]
    meta: dict

    @property
    def dim(self) -> int:
        return int(self.corpus.shape[1])


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (a / norms).astype(np.float32)


def _query_prefix(model: str) -> str:
    return BGE_QUERY_PREFIX if "bge" in model.lower() else ""


def is_cached(dataset: str) -> bool:
    d = embeddings_dir(dataset)
    return all(
        (d / f).exists()
        for f in ("meta.json", "corpus.npy", "corpus_ids.json", "queries.npy", "queries_ids.json")
    )


def _embed_texts(model, texts: list[str], batch_size: int, label: str) -> np.ndarray:
    out: list[np.ndarray] = []
    with tqdm(total=len(texts), desc=f"embed {label}", unit="txt") as bar:
        for vec in model.embed(texts, batch_size=batch_size):
            out.append(np.asarray(vec, dtype=np.float32))
            bar.update(1)
    return np.vstack(out)


def build(
    beir: Beir,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    force: bool = False,
) -> Embeddings:
    """Embed corpus + queries and persist the cache. Idempotent."""
    dataset = beir.name
    out = embeddings_dir(dataset)

    if is_cached(dataset) and not force:
        cached = load(dataset)
        if cached.meta.get("model") == model_name and len(cached.corpus_ids) == beir.n_docs:
            print(f"[embed] {dataset}: cache hit ({out}), skipping")
            return cached
        print(f"[embed] {dataset}: cache stale (model/count mismatch), rebuilding")

    from fastembed import TextEmbedding

    print(f"[embed] {dataset}: loading model {model_name}")
    model = TextEmbedding(model_name=model_name)

    prefix = _query_prefix(model_name)
    corpus = _embed_texts(model, beir.corpus_texts, batch_size, f"{dataset} corpus")
    queries = _embed_texts(
        model, [prefix + t for t in beir.query_texts], batch_size, f"{dataset} queries"
    )

    corpus = _normalize(corpus)
    queries = _normalize(queries)

    meta = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "split": beir.split,
        "model": model_name,
        "dim": int(corpus.shape[1]),
        "n_docs": int(corpus.shape[0]),
        "n_queries": int(queries.shape[0]),
        "dtype": "float32",
        "normalized": True,
        "similarity": "cosine (== dot product, vectors are unit norm)",
        "query_prefix": prefix,
        "doc_text": "title + ' ' + text (BEIR convention)",
        "row_order": "corpus.npy[i] <-> corpus_ids.json[i]; queries.npy[i] <-> queries_ids.json[i]",
    }

    out.mkdir(parents=True, exist_ok=True)
    _atomic_npy(out / "corpus.npy", corpus)
    _atomic_json(out / "corpus_ids.json", beir.corpus_ids)
    _atomic_npy(out / "queries.npy", queries)
    _atomic_json(out / "queries_ids.json", beir.query_ids)
    _atomic_json(out / "meta.json", meta)

    print(
        f"[embed] {dataset}: wrote {meta['n_docs']} doc + {meta['n_queries']} query "
        f"vectors (dim={meta['dim']}) to {out}"
    )
    return Embeddings(corpus, beir.corpus_ids, queries, beir.query_ids, meta)


def _atomic_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".npy.tmp")
    # Write through a file handle: np.save(path_like) would append a second
    # ".npy" to the temp name and the rename would miss it.
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    tmp.rename(path)


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    tmp.rename(path)


def load(dataset: str) -> Embeddings:
    d = embeddings_dir(dataset)
    if not is_cached(dataset):
        raise FileNotFoundError(
            f"no embedding cache for {dataset}. Run: memlake-bench embed {dataset}"
        )
    corpus = np.load(d / "corpus.npy", allow_pickle=False)
    queries = np.load(d / "queries.npy", allow_pickle=False)
    corpus_ids = json.loads((d / "corpus_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((d / "queries_ids.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

    if corpus.shape[0] != len(corpus_ids) or queries.shape[0] != len(query_ids):
        raise ValueError(f"corrupt embedding cache at {d}: row/id count mismatch")
    return Embeddings(corpus, corpus_ids, queries, query_ids, meta)
