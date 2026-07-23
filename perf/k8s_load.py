"""A small load driver for the k8s deployment: write a corpus, then time queries.

Runs as a k8s Job INSIDE the cluster (so latency is in-cluster over real AWS S3, not laptop→GKE),
pointed at the Envoy proxy service. Writes go through the proxy's consistent-hash so one serve pod
owns the namespace; queries then hit that warm pod. Prints a JSON summary.

Env: MEMLAKE_ADDR (proxy host:port), MEMLAKE_LOAD_NAMESPACE, N_DOCS, BATCH, N_QUERIES, DIM,
CLUSTERS, WAIT_INDEX_SECS.
"""

from __future__ import annotations

import json
import math
import os
import random
import time

from memlake_client import MemlakeClient, memory


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def main() -> int:
    addr = os.environ.get("MEMLAKE_ADDR", "localhost:50050")
    ns = os.environ.get("MEMLAKE_LOAD_NAMESPACE", "perf")
    n_docs = int(os.environ.get("N_DOCS", "20000"))
    batch = int(os.environ.get("BATCH", "512"))
    n_queries = int(os.environ.get("N_QUERIES", "200"))
    dim = int(os.environ.get("DIM", "384"))
    clusters = int(os.environ.get("CLUSTERS", "64"))
    wait_index = float(os.environ.get("WAIT_INDEX_SECS", "20"))

    rng = random.Random(7)
    centres = [_unit([rng.gauss(0, 1) for _ in range(dim)]) for _ in range(clusters)]

    def vec():
        c = rng.choice(centres)
        return _unit([x + rng.gauss(0, 0.15) for x in c])

    client = MemlakeClient(addr)
    for _ in range(60):
        try:
            client.create_namespace(ns)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[load] waiting for {addr} ({str(e).splitlines()[0] if str(e) else type(e).__name__})", flush=True)
            time.sleep(2)
    else:
        print(f"[load] cannot reach {addr}", flush=True)
        return 1

    # -- write phase --
    print(f"[load] writing {n_docs} docs (batch {batch}, dim {dim}) to '{ns}' via {addr}", flush=True)
    t0 = time.perf_counter()
    written = 0
    for start in range(0, n_docs, batch):
        end = min(start + batch, n_docs)
        mems = [
            memory(f"perf doc {i}", vector=vec(), memory_type=1, key=f"perf-{i}")
            for i in range(start, end)
        ]
        client.write(ns, mems)
        written += len(mems)
    write_secs = time.perf_counter() - t0
    print(f"[load] wrote {written} in {write_secs:.1f}s = {written / max(write_secs, 1e-6):,.0f}/s", flush=True)

    # Let the indexer fold before querying (reads are correct without it, but warm the index).
    if wait_index > 0:
        time.sleep(wait_index)

    # -- query phase (cold then warm) --
    def run_queries():
        lat, rts = [], []
        for i in range(n_queries):
            q = vec()
            t = time.perf_counter()
            client.query(ns, vector=q, memory_types=[1], vector_top_k=10, text_top_k=10, graph_top_k=10)
            lat.append((time.perf_counter() - t) * 1000.0)
            rts.append(client.last_roundtrips)
        return lat, rts

    cold_lat, cold_rts = run_queries()
    warm_lat, warm_rts = run_queries()
    client.close()

    summary = {
        "addr": addr,
        "namespace": ns,
        "n_docs": written,
        "write_secs": round(write_secs, 2),
        "write_per_s": round(written / max(write_secs, 1e-6)),
        "n_queries": n_queries,
        "cold": {
            "p50_ms": round(pct(cold_lat, 50), 2),
            "p90_ms": round(pct(cold_lat, 90), 2),
            "p99_ms": round(pct(cold_lat, 99), 2),
            "mean_roundtrips": round(sum(cold_rts) / max(len(cold_rts), 1), 2),
        },
        "warm": {
            "p50_ms": round(pct(warm_lat, 50), 2),
            "p90_ms": round(pct(warm_lat, 90), 2),
            "p99_ms": round(pct(warm_lat, 99), 2),
            "mean_roundtrips": round(sum(warm_rts) / max(len(warm_rts), 1), 2),
        },
    }
    print("PERF_SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
