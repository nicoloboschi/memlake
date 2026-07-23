"""memlake accuracy engine, e2e through the gRPC server.

Unlike the in-process `memlake_engine` (which drives a Rust binary directly), this writes the
corpus into a live `mlake-server` via the Python client, runs the indexer, and collects each
query's ranking over the wire — the exact deployed path. Every arm is scored with the same
`metrics` code as every other engine, so only the ranking differs, never the measurement.
"""

from __future__ import annotations

import os

import time
import uuid

from memlake_client import MemlakeClient, memory

from .. import metrics, server
from ..datasets import Beir
from ..embed import Embeddings

NS = "bench-grpc"
ADDR = "127.0.0.1:50252"
WRITE_BATCH = 512
# Fold the WAL tail into a segment every this many written docs, so the un-indexed tail stays
# bounded during backfill (as a continuously-running indexer keeps it in production). Without this,
# write-time link derivation brute-force-scans an ever-growing tail — O(N^2) — which is a harness
# artifact, not how the deployed system behaves.
INDEX_EVERY = 4096
MT = 1  # single memory_type for a BEIR corpus


def _id_bytes(doc_id: str) -> bytes:
    return uuid.uuid5(uuid.NAMESPACE_OID, doc_id).bytes


def _rank(client, query_text, qvec, *, top_k, graph, id_to_doc) -> dict:
    """ONE query returns the raw per-arm signals; every arm ranking is reported STANDALONE.

    memlake returns raw per-arm ranks and does no fusion — RRF/hybrid is a caller concern (e.g.
    Hindsight), not part of memlake — so we measure each arm's own retrieval quality, never a
    fused ranking. Returns {arm_name: ranked_doc_ids}."""
    hits = client.query(
        NS, vector=qvec, text=query_text, memory_types=[MT],
        vector_top_k=top_k, text_top_k=top_k, graph_top_k=top_k,
        nprobe=int(os.environ.get("MEMLAKE_BENCH_NPROBE", "0")),
    )

    def arm_docs(present, rank_of) -> list:
        ordered = sorted((h for h in hits if present(h)), key=rank_of)
        return [id_to_doc[h.id] for h in ordered if h.id in id_to_doc]

    out = {
        "dense": arm_docs(lambda h: h.dense.present, lambda h: h.dense.rank),
        "sparse": arm_docs(lambda h: h.text.present, lambda h: h.text.rank),
    }
    if graph:
        out["graph"] = arm_docs(lambda h: h.graph.present, lambda h: h.graph.rank)
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
    truth: metrics.Run | None = None,
) -> dict:
    binary = server.build_binary()
    id_to_doc = {_id_bytes(d): d for d in beir.corpus_ids}

    # -- write + index -------------------------------------------------------
    # Links are derived on the write path (server-side, before the commit), against the current
    # snapshot: the indexed segments plus the un-indexed WAL tail. A pure write-all-then-index-once
    # backfill lets that tail grow to the whole corpus, so every write brute-force-scans an O(N) tail
    # (O(N^2) overall) — which is NOT how production runs. There the indexer folds continuously, so
    # the tail stays bounded and derivation queries the index. We mirror that by folding every
    # INDEX_EVERY docs, keeping the tail bounded and the write cost representative.
    index_seconds = 0.0
    with server.Serve(binary, addr=ADDR, mem_mb=mem_mb, disk_mb=disk_mb):
        client = MemlakeClient(ADDR)
        client.create_namespace(NS)
        since_index = 0
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
            since_index += end - start
            if since_index >= INDEX_EVERY:
                index_seconds += server.index_once(binary, NS)["elapsed_s"]
                since_index = 0
        client.close()

    # Final catch-up fold: everything still in the tail lands in a segment before we query.
    index_seconds += server.index_once(binary, NS)["elapsed_s"]

    # -- query every arm -----------------------------------------------------
    # Per-arm, standalone — memlake does no fusion, so there is no hybrid row here.
    runs: dict[str, dict] = {"dense": {}, "sparse": {}}
    if graph:
        runs["graph"] = {}
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
        arm: metrics.evaluate(
            run,
            beir.qrels,
            # One query call produces every arm's ranking, so the measured latency is the
            # whole-query cost; report it on each arm rather than inventing a fused row.
            latencies,
            # Only the dense arm is comparable to an exhaustive *vector* scan; scoring the
            # text or graph arms against it would measure the wrong thing.
            truth=truth if arm == "dense" else None,
        )
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
