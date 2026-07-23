"""Load + measurement harness for one (scale, serve_cpus, concurrency) point.

Phases:
  1. seed   — write N synthetic 384-dim unit-vector memories through the REAL gRPC write
              path (batched), then wait for the indexer Deployment to fold them into a
              generation (polls Stats until wal_index_cursor catches wal_head).
  2. warmup — drive query load for a prefix window and discard it, so the two-tier read
              cache reaches a steady (warm) state before measurement.
  3. measure— drive concurrent query load at a target concurrency for a fixed window,
              recording achieved QPS + p50/p90/p99, while sampling the serve container's
              CPU%/memory via `docker stats` and the CacheStats RPC for the hit ratio.

Emits one structured JSON row (schema in `--out`), the row the sweep tabulates.

The load driver uses grpc.aio (one event loop, `concurrency` in-flight RPCs) so the driver's
own CPU cost is low relative to a thread-per-request pool — but it still shares host cores
with the containers, so keep the serve CPU limit well below the host core count (reported).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import grpc
import numpy as np

from memlake_client.v1 import memlake_pb2 as pb
from memlake_client.v1 import memlake_pb2_grpc as rpc

# Same small vocabulary the Rust datagen uses, so the FTS arm has real terms to match.
WORDS = [
    "memory", "recall", "vector", "graph", "lake", "index", "cluster", "query", "tag",
    "entity", "semantic", "episodic", "signal", "search", "bank", "fold", "shard", "probe",
]


def _pack(v) -> pb.Vector:
    return pb.Vector(f32le=np.asarray(v, dtype="<f4").tobytes())


def _unit_rows(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """n random unit vectors, clustered so IVF has real structure: draw from a modest set of
    centers plus noise, then L2-normalize."""
    m = np.ndarray
    x = rng.standard_normal((n, dim)).astype("f4")
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x


# ----------------------------------------------------------------------------- seed


def seed(addr: str, namespace: str, scale: int, dim: int, types: int, batch: int,
         seed_val: int) -> None:
    ch = grpc.insecure_channel(addr, options=[("grpc.max_send_message_length", 256 << 20)])
    stub = rpc.MemlakeStub(ch)
    stub.CreateNamespace(pb.CreateNamespaceRequest(namespace=namespace))

    # Idempotent skip: if the namespace already holds >= scale docs, don't re-seed.
    st = stub.Stats(pb.StatsRequest(namespace=namespace))
    if st.doc_count >= scale:
        print(f"[seed] namespace {namespace!r} already has {st.doc_count} docs >= {scale}; skipping seed")
        ch.close()
        return

    rng = np.random.default_rng(seed_val)
    # A pool of clustered centers so vectors aren't uniform noise (gives IVF structure).
    n_centers = max(1, int(math.sqrt(scale)))
    centers = _unit_rows(rng, n_centers, dim)

    t0 = time.monotonic()
    written = 0
    for start in range(0, scale, batch):
        end = min(start + batch, scale)
        b = end - start
        # vectors = center + noise, normalized.
        idx = (np.arange(start, end) % n_centers)
        noise = rng.standard_normal((b, dim)).astype("f4") * 0.35
        vecs = centers[idx] + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        ops = []
        for j in range(b):
            i = start + j
            text = " ".join(WORDS[(i + k * 7) % len(WORDS)] for k in range(6))
            tags = [] if (i % 5 == 0) else [f"tag-{i % 200}", f"tag-{(i * 3) % 200}"]
            m = pb.Memory(
                key=f"m{i}",
                vector=pb.Vector(f32le=vecs[j].tobytes()),
                text=text,
                memory_type=(i % max(1, types)) + 1,
                tags=tags,
                metadata={"doc": f"d{i % 1000}", "src": "perf"},
            )
            ops.append(pb.Op(upsert=m))
        stub.Write(pb.WriteRequest(namespace=namespace, ops=ops), timeout=300)
        written += b
        if start // batch % 10 == 0 or end == scale:
            rate = written / max(1e-6, time.monotonic() - t0)
            print(f"[seed] wrote {written}/{scale} ({rate:,.0f}/s)", flush=True)
    print(f"[seed] write done: {written} memories in {time.monotonic()-t0:.1f}s "
          f"(indexer will fold the tail)", flush=True)
    ch.close()


def wait_indexed(addr: str, namespace: str, scale: int, timeout_s: float) -> dict:
    """Poll Stats until the indexer has folded the whole WAL (wal_index_cursor >= wal_head)
    and doc_count reaches scale, or timeout. Returns the final Stats as a dict."""
    ch = grpc.insecure_channel(addr)
    stub = rpc.MemlakeStub(ch)
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout_s:
        st = stub.Stats(pb.StatsRequest(namespace=namespace))
        last = st
        done = st.wal_head > 0 and st.wal_index_cursor >= st.wal_head and st.doc_count >= scale
        print(f"[index] gen={st.generation} wal_head={st.wal_head} cursor={st.wal_index_cursor} "
              f"docs={st.doc_count}/{scale} {'OK' if done else '...'}", flush=True)
        if done:
            break
        time.sleep(5)
    ch.close()
    if last is None:
        return {}
    return {"generation": last.generation, "wal_head": last.wal_head,
            "wal_index_cursor": last.wal_index_cursor, "doc_count": last.doc_count,
            "indexed": bool(last.wal_head > 0 and last.wal_index_cursor >= last.wal_head)}


# ----------------------------------------------------------------------------- cache


def cache_snapshot(addr: str, namespace: str) -> dict:
    ch = grpc.insecure_channel(addr)
    stub = rpc.MemlakeStub(ch)
    r = stub.CacheStats(pb.CacheStatsRequest(namespace=namespace, limit=1))
    ch.close()
    return {"enabled": r.enabled, "hits": r.hits, "misses": r.misses,
            "mem_bytes": r.mem_bytes, "mem_budget": r.mem_budget,
            "disk_bytes": r.disk_bytes, "total_entries": r.total_entries}


# ----------------------------------------------------------------------------- docker stats sampler


class StatsSampler(threading.Thread):
    """Samples `docker stats --no-stream` for one container in a loop. `docker stats`
    itself takes ~1-2s per no-stream call, so the effective interval is that plus `interval`."""

    def __init__(self, container: str, interval: float = 0.5):
        super().__init__(daemon=True)
        self.container = container
        self.interval = interval
        self._stop = threading.Event()
        self.cpu: list[float] = []
        self.mem_mb: list[float] = []
        self.err: str | None = None

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format",
                     "{{.CPUPerc}}|{{.MemUsage}}", self.container],
                    capture_output=True, text=True, timeout=15,
                )
                line = out.stdout.strip()
                if line and "|" in line:
                    cpu_s, mem_s = line.split("|", 1)
                    self.cpu.append(float(cpu_s.strip().rstrip("%")))
                    self.mem_mb.append(_parse_mem_mb(mem_s.split("/")[0].strip()))
                elif out.returncode != 0:
                    self.err = out.stderr.strip()[:200]
            except Exception as e:  # noqa: BLE001
                self.err = str(e)[:200]
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def _parse_mem_mb(s: str) -> float:
    s = s.strip()
    units = {"B": 1 / 1e6, "KIB": 1 / 1024, "MIB": 1.048576, "GIB": 1073.741824,
             "KB": 1e-3, "MB": 1.0, "GB": 1000.0}
    for u in ("GIB", "MIB", "KIB", "GB", "MB", "KB", "B"):
        if s.upper().endswith(u):
            try:
                return float(s[: -len(u)]) * units[u]
            except ValueError:
                return 0.0
    return 0.0


# ----------------------------------------------------------------------------- load driver


@dataclass
class LoadResult:
    count: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    roundtrips: int = 0   # sum of QueryResponse.load_roundtrips (0 => served from RAM = warm)
    elapsed_s: float = 0.0


async def _worker(stub, requests, stop_at, res: LoadResult, counter: list[int], lock,
                  query_timeout: float):
    n = len(requests)
    while time.monotonic() < stop_at:
        # round-robin over the fixed query pool so warm-up populates the cache the measurement
        # then hits (a fixed pool keeps the same clusters hot => a readable warm hit ratio).
        with lock:
            k = counter[0]
            counter[0] += 1
        req = requests[k % n]
        t0 = time.monotonic()
        try:
            # A per-query deadline is essential: without it a single slow query (e.g. a cold
            # snapshot-open at a low CPU limit) hangs the worker past the window deadline.
            resp = await stub.Query(req, timeout=query_timeout)
        except Exception:  # noqa: BLE001
            res.errors += 1
            continue
        res.latencies_ms.append((time.monotonic() - t0) * 1000.0)
        res.roundtrips += resp.load_roundtrips
        res.count += 1


async def _run_load(addr: str, requests, concurrency: int, duration: float,
                    record: bool, query_timeout: float) -> LoadResult:
    res = LoadResult()
    async with grpc.aio.insecure_channel(
        addr, options=[("grpc.max_send_message_length", 256 << 20)]
    ) as ch:
        stub = rpc.MemlakeStub(ch)
        counter = [0]
        lock = threading.Lock()
        start = time.monotonic()
        stop_at = start + duration
        sink = res if record else LoadResult()
        await asyncio.gather(*[
            _worker(stub, requests, stop_at, sink, counter, lock, query_timeout)
            for _ in range(concurrency)
        ])
        sink.elapsed_s = time.monotonic() - start
        return sink


def build_query_pool(namespace: str, dim: int, pool: int, nprobe: int, topk: int,
                     with_text: bool, seed_val: int, memory_type: int = 0):
    rng = np.random.default_rng(seed_val ^ 0x5EED)
    vecs = rng.standard_normal((pool, dim)).astype("f4")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    reqs = []
    for i in range(pool):
        req = pb.QueryRequest(
            namespace=namespace,
            vector=pb.Vector(f32le=vecs[i].tobytes()),
            text=(" ".join(WORDS[(i + k) % len(WORDS)] for k in range(3)) if with_text else ""),
            vector_top_k=topk,
            text_top_k=topk,
            graph_top_k=topk,
            nprobe=nprobe,
            # 0 => every memory_type (the heavy default). A realistic recall targets one
            # independent index, which is ~3x cheaper here since the corpus has three types.
            memory_types=([memory_type] if memory_type else []),
        )
        reqs.append(req)
    return reqs


def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs), p))


# ----------------------------------------------------------------------------- main


def prime_snapshot(addr: str, req, timeout: float) -> float:
    """Force + await the (possibly slow) cold snapshot-open ONCE before the timed phases, so
    the QueryNode materialization cost (reading the generation + tantivy splits into RAM) is
    excluded from the measured window rather than skewing the first query's latency. Returns
    the open wall-time."""
    ch = grpc.insecure_channel(addr)
    stub = rpc.MemlakeStub(ch)
    t0 = time.monotonic()
    stub.Query(req, timeout=timeout)
    dt = time.monotonic() - t0
    ch.close()
    return dt


def measure(args) -> dict:
    reqs = build_query_pool(args.namespace, args.dim, args.query_pool, args.nprobe,
                            args.topk, not args.no_text, args.seed, args.memory_type)

    # A large dedicated timeout: the cold open (materializing the generation + tantivy splits
    # into RAM) can take minutes at a low CPU limit, far longer than a per-query deadline.
    open_s = prime_snapshot(args.addr, reqs[0], max(args.query_timeout, 900.0))
    print(f"[prime] cold snapshot open took {open_s:.1f}s (excluded from the window)", flush=True)

    # Warm-up (discarded).
    if args.warmup > 0:
        print(f"[warmup] {args.warmup}s at concurrency {args.concurrency}", flush=True)
        asyncio.run(_run_load(args.addr, reqs, args.concurrency, args.warmup,
                              record=False, query_timeout=args.query_timeout))

    cache_before = cache_snapshot(args.addr, args.namespace)

    sampler = StatsSampler(args.serve_container, interval=0.4)
    sampler.start()
    print(f"[measure] {args.duration}s at concurrency {args.concurrency}", flush=True)
    res = asyncio.run(_run_load(args.addr, reqs, args.concurrency, args.duration,
                                record=True, query_timeout=args.query_timeout))
    sampler.stop()
    sampler.join(timeout=20)

    cache_after = cache_snapshot(args.addr, args.namespace)

    d_hits = max(0, cache_after["hits"] - cache_before["hits"])
    d_miss = max(0, cache_after["misses"] - cache_before["misses"])
    hit_ratio = (d_hits / (d_hits + d_miss)) if (d_hits + d_miss) > 0 else float("nan")

    cpu_mean = float(np.mean(sampler.cpu)) if sampler.cpu else float("nan")
    cpu_peak = float(np.max(sampler.cpu)) if sampler.cpu else float("nan")
    mem_peak = float(np.max(sampler.mem_mb)) if sampler.mem_mb else float("nan")
    qps = res.count / res.elapsed_s if res.elapsed_s > 0 else 0.0

    row = {
        "scale": args.scale,
        "serve_cpus": args.serve_cpus,
        "serve_mem_mb": args.serve_mem_mb,
        "cache_mem_mb": args.cache_mem_mb,
        "concurrency": args.concurrency,
        "achieved_qps": round(qps, 2),
        "p50": round(pctl(res.latencies_ms, 50), 3),
        "p90": round(pctl(res.latencies_ms, 90), 3),
        "p99": round(pctl(res.latencies_ms, 99), 3),
        "cpu_pct_mean": round(cpu_mean, 1),
        "cpu_pct_peak": round(cpu_peak, 1),
        "cache_hit_ratio": round(hit_ratio, 4) if not math.isnan(hit_ratio) else None,
        # Warm/cold signal that IS meaningful at every scale: object-storage roundtrips per
        # query. 0 => the QueryNode snapshot is fully RAM-resident (fully warm); >0 => probed
        # clusters were fetched (through MinIO here) during the window.
        "mean_load_roundtrips": round(res.roundtrips / res.count, 3) if res.count else None,
        "cache_block_hits": d_hits,
        "cache_block_misses": d_miss,
        # context / provenance
        "backend": args.backend,
        "qps_per_vcpu": round(qps / args.serve_cpus, 1) if args.serve_cpus else None,
        "cpu_util_of_limit": (round(cpu_mean / (args.serve_cpus * 100.0), 3)
                              if args.serve_cpus and not math.isnan(cpu_mean) else None),
        "mem_peak_mb": round(mem_peak, 1) if not math.isnan(mem_peak) else None,
        "snapshot_open_s": round(open_s, 1),
        "errors": res.errors,
        "count": res.count,
        "duration_s": round(res.elapsed_s, 2),
        "warmup_s": args.warmup,
        "host_cpus": os.cpu_count(),
        "cache_samples": len(sampler.cpu),
        "cache_enabled": cache_before.get("enabled"),
        "stats_err": sampler.err,
        "ts": int(time.time()),
    }
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="memlake compute-vs-SLA load harness")
    ap.add_argument("--addr", default="localhost:50051")
    ap.add_argument("--namespace", default="perf")
    ap.add_argument("--scale", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--types", type=int, default=3)
    ap.add_argument("--seed-batch", dest="seed_batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--query-pool", dest="query_pool", type=int, default=256)
    ap.add_argument("--query-timeout", dest="query_timeout", type=float, default=45.0,
                    help="per-query gRPC deadline (s); a timed-out query counts as an error")
    ap.add_argument("--nprobe", type=int, default=0, help="0 = server default")
    ap.add_argument("--topk", type=int, default=0, help="0 = server default (returns ~600 hits)")
    ap.add_argument("--memory-type", dest="memory_type", type=int, default=0,
                    help="0 = query all types (heavy); N = one type (realistic recall)")
    ap.add_argument("--no-text", action="store_true", help="skip the BM25 arm (vector+graph only)")
    ap.add_argument("--index-timeout", dest="index_timeout", type=float, default=2400.0)
    # provenance / labels for the row
    ap.add_argument("--serve-container", dest="serve_container", default="memlake-serve-1")
    ap.add_argument("--serve-cpus", dest="serve_cpus", type=float,
                    default=float(os.environ.get("SERVE_CPUS", "2") or 2))
    ap.add_argument("--serve-mem-mb", dest="serve_mem_mb", type=int,
                    default=int(os.environ.get("SERVE_MEM_MB_TOTAL", "2048") or 2048))
    ap.add_argument("--cache-mem-mb", dest="cache_mem_mb", type=int,
                    default=int(os.environ.get("SERVE_MEM_MB", "1024") or 1024))
    ap.add_argument("--backend", default="minio", help="storage backend label for the row")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "perf" / "results" / "rows.jsonl"))
    ap.add_argument("--seed-only", action="store_true", help="seed + wait for index, then exit (no measure)")
    ap.add_argument("--write-only", action="store_true",
                    help="seed the WAL and exit WITHOUT waiting for indexing (the sweep folds "
                         "separately via a one-shot `index --once`, which is reliable — the 5s "
                         "indexer loop's incremental re-folds hang at scale in this environment)")
    ap.add_argument("--skip-seed", action="store_true", help="measure only; assume already seeded")
    args = ap.parse_args(argv)

    if args.write_only:
        seed(args.addr, args.namespace, args.scale, args.dim, args.types, args.seed_batch, args.seed)
        return 0

    if not args.skip_seed:
        seed(args.addr, args.namespace, args.scale, args.dim, args.types, args.seed_batch, args.seed)
        idx = wait_indexed(args.addr, args.namespace, args.scale, args.index_timeout)
        if not idx.get("indexed"):
            print(f"[warn] namespace not fully indexed within timeout: {idx}", file=sys.stderr)
    if args.seed_only:
        return 0

    row = measure(args)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print("\n=== ROW ===")
    print(json.dumps(row, indent=2))
    print(f"(appended to {outp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
