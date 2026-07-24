"""Load driver for the k8s deployment: write a corpus (concurrently), wait for the fold, time
queries. Runs as an in-cluster Job pointed at the Envoy proxy, so latency is representative over
real AWS S3. Prints a JSON summary line (`PERF_SUMMARY {...}`).

Two corpus modes:

* **real dataset** (set `PERF_DATASET`): load a precomputed artifact of real embeddings + texts
  from S3 — corpus AND query vectors produced offline by `perf/prepare_dataset.py` with a
  production embedding model (e.g. jina-embeddings-v3, 1024d). Writes real docs and queries with
  the dataset's real questions, so the vector arm sees genuine near-neighbours (not the synthetic
  near-uniform structure that made stage-two rerank look like the whole scanned set). Embedding is
  done offline; the download happens BEFORE timing, so vector generation never touches the hot path.
* **synthetic** (default): a Gaussian cluster model generated in-process. Kept for a dependency-free
  smoke test; not representative of production retrieval.

Env: MEMLAKE_ADDR, MEMLAKE_LOAD_NAMESPACE, N_DOCS, BATCH, CONCURRENCY, N_QUERIES, DIM, CLUSTERS,
FOLD_WAIT_SECS; for real mode: PERF_DATASET (S3 key prefix under the perf bucket),
MEMLAKE_PERF_S3_BUCKET / _ACCESS_KEY / _SECRET_KEY / _REGION.
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


def _load_dataset_artifact(prefix: str):
    """Download the precomputed real-embedding artifact from the perf S3 bucket and return
    (corpus_vecs, corpus_texts, corpus_ids, query_vecs, query_texts, meta). Runs before the timed
    phases — the vectors were produced offline, so no embedding happens here."""
    import io

    import boto3
    import numpy as np

    bucket = os.environ["MEMLAKE_PERF_S3_BUCKET"]
    key = prefix.strip("/")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["MEMLAKE_PERF_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MEMLAKE_PERF_S3_SECRET_KEY"],
        region_name=os.environ.get("MEMLAKE_PERF_S3_REGION", "us-east-1"),
    )

    def get(name):
        return s3.get_object(Bucket=bucket, Key=f"{key}/{name}")["Body"].read()

    corpus = np.load(io.BytesIO(get("corpus.npy")), allow_pickle=False)
    queries = np.load(io.BytesIO(get("queries.npy")), allow_pickle=False)
    corpus_ids = json.loads(get("corpus_ids.json"))
    corpus_texts = json.loads(get("corpus_texts.json"))
    query_texts = json.loads(get("query_texts.json"))
    meta = json.loads(get("meta.json"))
    print(
        f"[load] dataset '{meta.get('dataset')}' model={meta.get('model')} dim={meta.get('dim')} "
        f"({corpus.shape[0]} docs, {queries.shape[0]} queries) from s3://{bucket}/{key}",
        flush=True,
    )
    return corpus, corpus_texts, corpus_ids, queries, query_texts, meta


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

    # Real-dataset mode: precomputed embeddings + texts, fetched BEFORE any timing so no
    # vector generation ever lands on the write/query hot path.
    dataset = os.environ.get("PERF_DATASET", "").strip()
    ds = None
    if dataset:
        c_vecs, c_texts, c_ids, q_vecs, q_texts, ds_meta = _load_dataset_artifact(dataset)
        dim = int(ds_meta.get("dim", c_vecs.shape[1]))
        n_docs = min(n_docs, len(c_ids))
        ds = (c_vecs, c_texts, c_ids, q_vecs, q_texts, ds_meta)

    base_rng = random.Random(7)
    centres = [] if ds else [_unit([base_rng.gauss(0, 1) for _ in range(dim)]) for _ in range(clusters)]

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
    corpus_kind = f"{ds[5].get('dataset')}/{ds[5].get('model')}" if ds else "synthetic"
    print(
        f"[load] writing {n_docs} docs (batch {batch}, {concurrency} writers, dim {dim}, "
        f"corpus={corpus_kind}) to '{ns}'",
        flush=True,
    )

    def write_chunk(worker_i, my_starts):
        client = MemlakeClient(addr)
        rng = random.Random(1000 + worker_i)
        n = 0
        for start in my_starts:
            end = min(start + batch, n_docs)
            if ds:
                # Real corpus: the dataset's own passage text and its precomputed vector, so the
                # index sees a production-shaped similarity distribution (genuine near-neighbours).
                c_vecs, c_texts, c_ids = ds[0], ds[1], ds[2]
                mems = [
                    memory(
                        c_texts[i],
                        vector=c_vecs[i].tolist(),
                        memory_type=1,
                        key=str(c_ids[i]),
                    )
                    for i in range(start, end)
                ]
            else:
                mems = [
                    memory(f"perf {i}", vector=vec(rng), memory_type=1, key=f"perf-{i}")
                    for i in range(start, end)
                ]
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
        for i in range(n_queries):
            if ds:
                # The dataset's real questions (cycled if N_QUERIES exceeds the query set), with the
                # question text too so the FTS arm is exercised as it would be in production.
                q_vecs, q_texts = ds[3], ds[4]
                j = i % len(q_texts)
                q, qtext = q_vecs[j].tolist(), q_texts[j]
            else:
                q, qtext = vec(rng), None
            t = time.perf_counter()
            ctrl.query(
                ns, vector=q, text=qtext, memory_types=[1],
                vector_top_k=10, text_top_k=10, graph_top_k=10,
            )
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
        "corpus": corpus_kind,
        "dim": dim,
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
