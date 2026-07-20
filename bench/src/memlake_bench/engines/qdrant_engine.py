"""Qdrant hybrid baseline: dense HNSW + native sparse BM25, fused server-side with RRF.

Dense vectors come from the shared embedding cache (never re-embedded here), so
the dense arm is directly comparable to `exact` and to the future Rust engine.
Sparse vectors use fastembed's "Qdrant/bm25" model, whose IDF term is computed
by Qdrant itself via `Modifier.IDF` (that is why the collection declares it).

Point ids are the corpus row index, i.e. point id `i` <-> `corpus_ids[i]`, which
keeps the embedding-cache row-order contract intact end to end.
"""

from __future__ import annotations

import time

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from .. import metrics
from ..datasets import Beir
from ..embed import Embeddings

TOP_K = 100
DENSE_VEC = "dense"
SPARSE_VEC = "bm25"
SPARSE_MODEL = "Qdrant/bm25"


def collection_name(dataset: str) -> str:
    return f"memlake_bench_{dataset.replace('-', '_')}"


def _sparse_embed_docs(texts: list[str], batch_size: int):
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    yield from model.embed(texts, batch_size=batch_size)


def _sparse_embed_queries(texts: list[str]):
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return list(model.query_embed(texts))


def _index(
    client: QdrantClient,
    name: str,
    beir: Beir,
    emb: Embeddings,
    batch_size: int,
    recreate: bool,
) -> float:
    exists = client.collection_exists(name)
    if exists and not recreate:
        info = client.get_collection(name)
        if (info.points_count or 0) == beir.n_docs:
            print(f"[qdrant] collection {name} already has {beir.n_docs} points, reusing")
            return 0.0
        print(f"[qdrant] collection {name} incomplete, rebuilding")
    if exists:
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VEC: models.VectorParams(size=emb.dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            # IDF is applied server-side; the model emits raw term frequencies.
            SPARSE_VEC: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    t0 = time.perf_counter()
    buf: list[models.PointStruct] = []
    sparse_iter = _sparse_embed_docs(beir.corpus_texts, batch_size)

    for i, sparse in enumerate(
        tqdm(sparse_iter, total=beir.n_docs, desc="qdrant index", unit="doc")
    ):
        buf.append(
            models.PointStruct(
                id=i,
                vector={
                    DENSE_VEC: emb.corpus[i].tolist(),
                    SPARSE_VEC: models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload={"doc_id": beir.corpus_ids[i]},
            )
        )
        if len(buf) >= batch_size:
            client.upsert(collection_name=name, points=buf, wait=False)
            buf = []
    if buf:
        client.upsert(collection_name=name, points=buf, wait=True)

    _wait_green(client, name)
    elapsed = time.perf_counter() - t0
    print(f"[qdrant] indexed {beir.n_docs} docs in {elapsed:.1f}s")
    return elapsed


def _wait_green(client: QdrantClient, name: str, timeout_s: float = 600.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN:
            return
        time.sleep(1.0)
    raise TimeoutError(f"collection {name} did not reach GREEN")


def _to_run(points, ids: list[str]) -> list[str]:
    out = []
    for p in points:
        doc_id = (p.payload or {}).get("doc_id")
        out.append(doc_id if doc_id is not None else ids[int(p.id)])
    return out


def run(
    beir: Beir,
    emb: Embeddings,
    url: str,
    top_k: int = TOP_K,
    batch_size: int = 256,
    recreate: bool = False,
    prefetch_limit: int | None = None,
) -> dict:
    if emb.corpus_ids != beir.corpus_ids:
        raise ValueError("embedding cache corpus ids do not match the loaded dataset")

    prefetch_limit = prefetch_limit or top_k * 2
    name = collection_name(beir.name)
    client = QdrantClient(url=url, timeout=300)

    index_seconds = _index(client, name, beir, emb, batch_size, recreate)

    print("[qdrant] embedding queries (sparse)")
    q_sparse = _sparse_embed_queries(beir.query_texts)

    dense_run: metrics.Run = {}
    sparse_run: metrics.Run = {}
    hybrid_run: metrics.Run = {}
    dense_lat: list[float] = []
    sparse_lat: list[float] = []
    hybrid_lat: list[float] = []

    for i, qid in enumerate(tqdm(beir.query_ids, desc="qdrant query", unit="q")):
        dvec = emb.queries[i].tolist()
        svec = models.SparseVector(
            indices=q_sparse[i].indices.tolist(), values=q_sparse[i].values.tolist()
        )

        t0 = time.perf_counter()
        r = client.query_points(name, query=dvec, using=DENSE_VEC, limit=top_k, with_payload=True)
        dense_lat.append((time.perf_counter() - t0) * 1000.0)
        dense_run[qid] = _to_run(r.points, beir.corpus_ids)

        t0 = time.perf_counter()
        r = client.query_points(name, query=svec, using=SPARSE_VEC, limit=top_k, with_payload=True)
        sparse_lat.append((time.perf_counter() - t0) * 1000.0)
        sparse_run[qid] = _to_run(r.points, beir.corpus_ids)

        t0 = time.perf_counter()
        r = client.query_points(
            name,
            prefetch=[
                models.Prefetch(query=dvec, using=DENSE_VEC, limit=prefetch_limit),
                models.Prefetch(query=svec, using=SPARSE_VEC, limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        hybrid_lat.append((time.perf_counter() - t0) * 1000.0)
        hybrid_run[qid] = _to_run(r.points, beir.corpus_ids)

    return {
        "engine": "qdrant",
        "config": {
            "dense": "Qdrant HNSW, cosine, vectors from the shared embedding cache",
            "sparse": f"fastembed {SPARSE_MODEL} sparse vectors, IDF applied server-side",
            "fusion": "Qdrant native RRF (FusionQuery) over both prefetch arms",
            "top_k": top_k,
            "prefetch_limit": prefetch_limit,
            "model": emb.meta.get("model"),
            "dim": emb.dim,
            "url": url,
            "collection": name,
        },
        "corpus_size": beir.n_docs,
        "n_queries": beir.n_queries,
        "index_seconds": round(index_seconds, 2),
        "arms": {
            "dense": metrics.evaluate(dense_run, beir.qrels, dense_lat),
            "sparse": metrics.evaluate(sparse_run, beir.qrels, sparse_lat),
            "hybrid": metrics.evaluate(hybrid_run, beir.qrels, hybrid_lat),
        },
    }
