"""End-to-end recall regression check: write → index → recall, asserting every arm works
through the Python client against a live server (client → gRPC → server → S3).

This is the test that would have caught a broken arm or a cluster-format regression: it
writes a controlled corpus, runs the indexer, and asserts dense, full-text, graph, and
temporal retrieval each surface the memory they should — with the payload returned inline.

Run:  uv run --project bench memlake-bench recall
"""

from __future__ import annotations

import uuid

import numpy as np

from memlake_client import ANY, EVENTUAL, MemlakeClient, memory

from . import server

ADDR = "127.0.0.1:50611"
DIM = 64
NS = "recall-check"


def _unit(v) -> list[float]:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


def _axis(i: int) -> list[float]:
    """A near-one-hot unit vector on axis `i` — orthogonal to other axes, so items on
    different axes land in different IVF clusters."""
    v = np.zeros(DIM)
    v[i % DIM] = 1.0
    v[(i + 1) % DIM] = 0.05
    return _unit(v)


def _eid(name: str) -> bytes:
    return uuid.uuid5(uuid.NAMESPACE_OID, name).bytes


def _id(key: str) -> bytes:
    return uuid.uuid5(uuid.NAMESPACE_OID, key).bytes


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run(*, keep: bool = False) -> None:
    binary = server.build_binary()
    ns = NS if keep else f"{NS}-{uuid.uuid4().hex[:8]}"
    entity = _eid("shared-topic")

    # -- corpus ---------------------------------------------------------------
    # Enough random-vector filler that k-means forms many clusters, so nprobe=1 near the
    # anchor reliably does NOT probe the far entity-sharer's cluster.
    rng = np.random.default_rng(7)
    mems = []
    for i in range(200):
        mems.append(memory(f"filler note number {i}", _unit(rng.standard_normal(DIM)),
                           memory_type=1, key=f"f{i}", occurred_start=1000 + i * 5))
    # DENSE + FTS + GRAPH anchor: distinctive vector, a rare word, and the shared entity.
    mems.append(memory("the platypus lays eggs", _axis(0), memory_type=1, key="dense",
                       tags=["zoo"], entity_ids=[entity], metadata={"document_id": "d-dense"},
                       occurred_start=5000))
    # GRAPH sharer: same entity, but a far axis (a different cluster from the anchor), so only
    # entity expansion — not the vector probe — can connect them.
    mems.append(memory("submarines dive deep in the ocean", _axis(30), memory_type=1, key="graph",
                       entity_ids=[entity], metadata={"document_id": "d-graph"}, occurred_start=6000))
    # TEMPORAL: memories at a cluster of times, for a window query.
    for i, t in enumerate([20000, 20500, 21000, 21500, 22000]):
        mems.append(memory(f"temporal event {i}", _axis(2), memory_type=1, key=f"t{i}",
                           occurred_start=t))

    with server.Serve(binary, addr=ADDR, mem_mb=64, disk_mb=512):
        c = MemlakeClient(ADDR)
        c.create_namespace(ns)
        c.write(ns, mems)
        c.close()
    server.index_once(binary, ns)

    with server.Serve(binary, addr=ADDR, mem_mb=64, disk_mb=512):
        c = MemlakeClient(ADDR)
        results = {}

        # 1. DENSE: querying with the anchor's own vector must rank it first by dense score.
        hits = c.query(ns, vector=_axis(0), consistency=EVENTUAL)
        dense_ranked = sorted((h for h in hits if h.dense.present), key=lambda h: h.dense.rank)
        _check(dense_ranked and dense_ranked[0].id == _id("dense"),
               "dense arm: self-query did not rank the anchor first")
        _check(dense_ranked[0].dense.score > 0.9, "dense arm: top self-similarity should be ~1.0")
        _check(dense_ranked[0].memory is not None and "platypus" in dense_ranked[0].memory.text,
               "payload: memory not returned inline")
        results["dense"] = f"top self-hit score={dense_ranked[0].dense.score:.3f}"

        # 2. FTS: a rare word must surface the anchor via the text arm.
        hits = c.query(ns, vector=_axis(0), text="platypus", consistency=EVENTUAL)
        fts = [h for h in hits if h.id == _id("dense") and h.text.present]
        _check(bool(fts), "fts arm: rare-word query did not surface the anchor")
        results["fts"] = f"'platypus' -> anchor (text score={fts[0].text.score:.2f})"

        # 3. GRAPH: the entity-sharer on a far axis must come back via the graph arm even with
        #    nprobe=1 (its own cluster is not probed by the vector query near the anchor).
        hits = c.query(ns, vector=_axis(0), nprobe=1, consistency=EVENTUAL)
        far = {h.id: h for h in hits}.get(_id("graph"))
        _check(far is not None and far.graph.present,
               "graph arm: entity-sharer in an unprobed cluster was not surfaced")
        via = "graph only" if not far.dense.present else "graph (also dense)"
        results["graph"] = f"entity-sharer surfaced via {via} (score={far.graph.score:.3f})"

        # 4. TEMPORAL: a window over the temporal cluster returns those memories, scored by
        #    proximity, peaking nearest the window centre.
        hits = c.query(ns, vector=_axis(2), temporal_from=20000, temporal_to=22000, consistency=EVENTUAL)
        temporal = sorted((h for h in hits if h.temporal.present), key=lambda h: -h.temporal.score)
        _check(bool(temporal), "temporal arm: window query returned no temporal hits")
        top_ts = int(temporal[0].memory.metadata.get("occurred_start", "0")) if temporal[0].memory.metadata else 0
        # centre is 21000; the closest event (21000) should top the temporal ranking
        _check(temporal[0].id == _id("t2"), "temporal arm: peak proximity not at the window centre")
        results["temporal"] = f"{len(temporal)} in-window, peak at centre (score={temporal[0].temporal.score:.3f})"

        c.close()

    print("RECALL CHECK PASSED — all arms:")
    for arm, detail in results.items():
        print(f"  {arm:9} {detail}")
