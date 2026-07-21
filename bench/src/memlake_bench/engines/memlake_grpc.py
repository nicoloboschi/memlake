"""memlake accuracy engine, e2e through the gRPC server.

Unlike the in-process `memlake_engine` (which drives a Rust binary directly), this writes the
corpus into a live `mlake-server` via the Python client, runs the indexer, and collects each
query's ranking over the wire — the exact deployed path. Every arm is scored with the same
`metrics` code as every other engine, so only the ranking differs, never the measurement.
"""

from __future__ import annotations

import time
import uuid

from memlake_client import EVENTUAL, MemlakeClient, memory

from .. import metrics, server
from ..datasets import Beir
from ..embed import Embeddings

NS = "bench-grpc"
ADDR = "127.0.0.1:50252"
WRITE_BATCH = 512
MT = 1  # single memory_type for a BEIR corpus


def _id_bytes(doc_id: str) -> bytes:
    return uuid.uuid5(uuid.NAMESPACE_OID, doc_id).bytes


def _rrf(rankings, k=60) -> list:
    """Reciprocal-rank fusion of several ranked doc-id lists — the client-side fusion memlake
    deliberately leaves to the caller. This is exactly what Hindsight would do with the raw
    per-arm ranks memlake returns."""
    scores: dict = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _rank(client, query_text, qvec, *, top_k, graph, id_to_doc) -> dict:
    """ONE query returns the raw per-arm signals; every arm ranking is derived client-side.
    Returns {arm_name: ranked_doc_ids}."""
    hits = client.query(
        NS, vector=qvec, text=query_text, memory_types=[MT],
        vector_top_k=top_k, text_top_k=top_k, graph_top_k=top_k, consistency=EVENTUAL,
    )

    def arm_docs(present, rank_of) -> list:
        ordered = sorted((h for h in hits if present(h)), key=rank_of)
        return [id_to_doc[h.id] for h in ordered if h.id in id_to_doc]

    dense = arm_docs(lambda h: h.dense.present, lambda h: h.dense.rank)
    sparse = arm_docs(lambda h: h.text.present, lambda h: h.text.rank)
    out = {"dense": dense, "sparse": sparse, "hybrid": _rrf([dense, sparse])}
    if graph:
        graph_docs = arm_docs(lambda h: h.graph.present, lambda h: h.graph.rank)
        out["hybrid"] = _rrf([dense, sparse, graph_docs])
    return out


def run(
    beir: Beir,
    emb: Embeddings,
    *,
    top_k: int = 100,
    graph: bool = False,
    mem_mb: int = 256,
    disk_mb: int = 4096,
    engine_name: str = "memlake",
) -> dict:
    binary = server.build_binary()
    id_to_doc = {_id_bytes(d): d for d in beir.corpus_ids}

    # -- write + index -------------------------------------------------------
    with server.Serve(binary, addr=ADDR, mem_mb=mem_mb, disk_mb=disk_mb):
        client = MemlakeClient(ADDR)
        client.create_namespace(NS)
        for start in range(0, beir.n_docs, WRITE_BATCH):
            end = min(start + WRITE_BATCH, beir.n_docs)
            client.write(
                NS,
                [
                    memory(
                        beir.corpus_texts[i],
                        vector=[float(x) for x in emb.corpus[i]],
                        memory_type=MT,
                        key=beir.corpus_ids[i],
                    )
                    for i in range(start, end)
                ],
            )
        client.close()

    summary = server.index_once(binary, NS)
    index_seconds = summary["elapsed_s"]

    # -- query every arm -----------------------------------------------------
    runs: dict[str, dict] = {"dense": {}, "sparse": {}, "hybrid": {}}
    latencies: list[float] = []
    with server.Serve(binary, addr=ADDR, mem_mb=mem_mb, disk_mb=disk_mb):
        client = MemlakeClient(ADDR)
        for qi, qid in enumerate(beir.query_ids):
            qvec = [float(x) for x in emb.queries[qi]]
            t = time.perf_counter()
            ranked = _rank(
                client, beir.query_texts[qi], qvec,
                top_k=top_k, graph=graph, id_to_doc=id_to_doc,
            )
            latencies.append((time.perf_counter() - t) * 1000.0)
            for arm, doc_ids in ranked.items():
                runs[arm][qid] = doc_ids
        client.close()

    arms_out = {
        arm: metrics.evaluate(run, beir.qrels, latencies if arm == "hybrid" else [])
        for arm, run in runs.items()
    }

    return {
        "engine": engine_name,
        "config": {
            "note": "e2e via gRPC: python client -> mlake-server -> S3 (MinIO); "
            "same cached bge-small vectors as qdrant",
            "graph": graph,
        },
        "corpus_size": beir.n_docs,
        "n_queries": beir.n_queries,
        "index_seconds": round(index_seconds, 3),
        "arms": arms_out,
    }
