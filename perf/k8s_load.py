"""Load driver for the k8s deployment: write a corpus (concurrently), wait for the fold, time
queries. Runs as an in-cluster Job pointed at the Envoy proxy, so latency is representative over
real AWS S3. Prints a JSON summary line (`PERF_SUMMARY {...}`).

Env: MEMLAKE_ADDR, MEMLAKE_LOAD_NAMESPACE, N_DOCS, BATCH, CONCURRENCY, N_QUERIES, DIM, CLUSTERS,
FOLD_WAIT_SECS.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from memlake_client import MemlakeClient, memory


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def main() -> int:
    addr = os.environ.get("MEMLAKE_ADDR", "localhost:50050")
    ns = os.environ.get("MEMLAKE_LOAD_NAMESPACE", "perf")
    n_docs = int(os.environ.get("N_DOCS", "50000"))
    batch = int(os.environ.get("BATCH", "512"))
    concurrency = int(os.environ.get("CONCURRENCY", "8"))
    n_queries = int(os.environ.get("N_QUERIES", "300"))
    dim = int(os.environ.get("DIM", "384"))
    clusters = int(os.environ.get("CLUSTERS", "128"))
    fold_wait = float(os.environ.get("FOLD_WAIT_SECS", "180"))

    base_rng = random.Random(7)
    centres = [_unit([base_rng.gauss(0, 1) for _ in range(dim)]) for _ in range(clusters)]

    def vec(rng):
        c = rng.choice(centres)
        return _unit([x + rng.gauss(0, 0.15) for x in c])

    ctrl = MemlakeClient(addr)
    for _ in range(60):
        try:
            ctrl.create_namespace(ns)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[load] waiting for {addr} ({str(e).splitlines()[0] if str(e) else type(e).__name__})", flush=True)
            time.sleep(2)
    else:
        print(f"[load] cannot reach {addr}", flush=True)
        return 1

    # -- write phase: `concurrency` threads, each its own client + rng --------
    starts = list(range(0, n_docs, batch))
    print(f"[load] writing {n_docs} docs (batch {batch}, {concurrency} writers, dim {dim}) to '{ns}'", flush=True)

    def write_chunk(worker_i, my_starts):
        client = MemlakeClient(addr)
        rng = random.Random(1000 + worker_i)
        n = 0
        for start in my_starts:
            end = min(start + batch, n_docs)
            mems = [memory(f"perf {i}", vector=vec(rng), memory_type=1, key=f"perf-{i}") for i in range(start, end)]
            client.write(ns, mems)
            n += len(mems)
        client.close()
        return n

    # Round-robin the batches across workers so each does roughly equal work.
    buckets = [starts[i::concurrency] for i in range(concurrency)]
    t0 = time.perf_counter()
    written = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(write_chunk, i, buckets[i]) for i in range(concurrency)]
        for f in as_completed(futs):
            written += f.result()
    write_secs = time.perf_counter() - t0
    print(f"[load] wrote {written} in {write_secs:.1f}s = {written / max(write_secs, 1e-6):,.0f}/s (aggregate)", flush=True)

    # -- wait for the indexer to fold (steady state) --------------------------
    fold_t = time.perf_counter()
    folded = False
    last = None
    while time.perf_counter() - fold_t < fold_wait:
        try:
            s = ctrl.stats(ns)
            backlog = s.wal_head - s.wal_index_cursor
            last = (s.generation, s.doc_count, backlog, s.wal_head)
            if backlog == 0 and s.wal_head > 0:
                folded = True
                break
        except Exception as e:  # noqa: BLE001
            print(f"[load] stats error: {e}", flush=True)
        time.sleep(3)
    fold_secs = time.perf_counter() - fold_t
    gen, doc_count, backlog, head = last or (0, 0, 0, 0)
    print(f"[load] fold wait {fold_secs:.1f}s: folded={folded} gen={gen} docs={doc_count} backlog={backlog}", flush=True)

    # -- query phase (cold then warm) -----------------------------------------
    def run_queries():
        lat, rts = [], []
        rng = random.Random(999)
        for _ in range(n_queries):
            q = vec(rng)
            t = time.perf_counter()
            ctrl.query(ns, vector=q, memory_types=[1], vector_top_k=10, text_top_k=10, graph_top_k=10)
            lat.append((time.perf_counter() - t) * 1000.0)
            rts.append(ctrl.last_roundtrips)
        return lat, rts

    cold_lat, cold_rts = run_queries()
    warm_lat, warm_rts = run_queries()
    ctrl.close()

    def arm(lat, rts):
        return {
            "p50_ms": round(pct(lat, 50), 2),
            "p90_ms": round(pct(lat, 90), 2),
            "p99_ms": round(pct(lat, 99), 2),
            "mean_roundtrips": round(sum(rts) / max(len(rts), 1), 2),
        }

    summary = {
        "addr": addr,
        "namespace": ns,
        "n_docs": written,
        "concurrency": concurrency,
        "write_secs": round(write_secs, 2),
        "write_per_s": round(written / max(write_secs, 1e-6)),
        "fold_wait_secs": round(fold_secs, 1),
        "folded": folded,
        "generation": gen,
        "doc_count": doc_count,
        "n_queries": n_queries,
        "cold": arm(cold_lat, cold_rts),
        "warm": arm(warm_lat, warm_rts),
    }
    print("PERF_SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
