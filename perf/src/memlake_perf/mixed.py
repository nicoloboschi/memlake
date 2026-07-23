"""Mixed multi-namespace read+write load — surfaces concurrency bottlenecks that a single-
namespace, read-only run cannot: snapshot reopen storms (every write invalidates a namespace's
cached snapshot, so its readers re-derive the tail), cross-namespace cache eviction (the shared
CLOCK cache has no per-namespace isolation — a noisy neighbour), the shared query-permit limiter,
and indexer fairness across namespaces.

Reads and writes run at the same time across N namespaces. Client-side it reports read/write
throughput + latency (aggregate and per-namespace). The server's `MEMLAKE_TRACE_LOG` JSONL has
the per-call breakdown (snapshot action/open_ms/tail_entries, cache hit ratio, permit_wait_ms,
link_ms) to root-cause — analyse it with jq after the run.

    uv run --project perf python -m memlake_perf.mixed --addr localhost:50052 \
        --namespaces 6 --scale 10000 --readers 12 --writers 4 --duration 30
"""

import argparse
import asyncio
import math
import random
import time
from dataclasses import dataclass, field

import grpc
import numpy as np
from memlake_client.v1 import memlake_pb2 as pb
from memlake_client.v1 import memlake_pb2_grpc as rpc

from .harness import (
    WORDS,
    _unit_rows,
    build_query_pool,
    cache_snapshot,
    pctl,
    seed,
    wait_indexed,
)


@dataclass
class OpStats:
    lat_ms: list = field(default_factory=list)
    errors: int = 0
    per_ns: dict = field(default_factory=dict)  # ns -> list[latency_ms]

    def record(self, ns: str, ms: float):
        self.lat_ms.append(ms)
        self.per_ns.setdefault(ns, []).append(ms)


def _write_ops(rng, dim, batch, types, centers, n_centers, tag):
    idx = rng.integers(0, n_centers, size=batch)
    noise = rng.standard_normal((batch, dim)).astype("f4") * 0.35
    vecs = centers[idx] + noise
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    ops = []
    for j in range(batch):
        i = int(rng.integers(0, 1_000_000_000))
        m = pb.Memory(
            key=f"w{tag}-{i}",
            vector=pb.Vector(f32le=vecs[j].tobytes()),
            text=" ".join(WORDS[(i + k * 7) % len(WORDS)] for k in range(6)),
            memory_type=(i % max(1, types)) + 1,
            tags=[f"tag-{i % 200}"],
            metadata={"src": "mixed"},
        )
        ops.append(pb.Op(upsert=m))
    return ops


async def _reader(stub, pools, stop_at, out: OpStats, timeout: float):
    names = list(pools.keys())
    while time.monotonic() < stop_at:
        ns = random.choice(names)
        req = random.choice(pools[ns])
        t0 = time.monotonic()
        try:
            await stub.Query(req, timeout=timeout)
        except Exception:  # noqa: BLE001
            out.errors += 1
            continue
        out.record(ns, (time.monotonic() - t0) * 1000.0)


async def _writer(stub, names, stop_at, out: OpStats, timeout, dim, batch, types, centers, n_centers, wid):
    rng = np.random.default_rng(1000 + wid)
    seq = 0
    while time.monotonic() < stop_at:
        ns = random.choice(names)
        ops = _write_ops(rng, dim, batch, types, centers, n_centers, f"{wid}-{seq}")
        seq += 1
        req = pb.WriteRequest(namespace=ns, ops=ops)
        t0 = time.monotonic()
        try:
            await stub.Write(req, timeout=timeout)
        except Exception:  # noqa: BLE001
            out.errors += 1
            continue
        out.record(ns, (time.monotonic() - t0) * 1000.0)


async def _run_mixed(addr, names, pools, readers, writers, duration, dim, wbatch, types, centers, n_centers, timeout):
    reads, writes = OpStats(), OpStats()
    async with grpc.aio.insecure_channel(
        addr, options=[("grpc.max_send_message_length", 256 << 20)]
    ) as ch:
        stub = rpc.MemlakeStub(ch)
        stop_at = time.monotonic() + duration
        tasks = [_reader(stub, pools, stop_at, reads, timeout) for _ in range(readers)]
        tasks += [
            _writer(stub, names, stop_at, writes, timeout, dim, wbatch, types, centers, n_centers, w)
            for w in range(writers)
        ]
        await asyncio.gather(*tasks)
    return reads, writes


def _report(label: str, st: OpStats, dur: float):
    n = len(st.lat_ms)
    print(
        f"{label}: {n} ops  {n / dur:,.1f}/s  errors={st.errors}  "
        f"p50={pctl(st.lat_ms, 50):.1f}ms  p90={pctl(st.lat_ms, 90):.1f}ms  p99={pctl(st.lat_ms, 99):.1f}ms"
    )
    for ns in sorted(st.per_ns):
        xs = st.per_ns[ns]
        print(f"    {ns}: {len(xs)} ops  p50={pctl(xs, 50):.1f}  p99={pctl(xs, 99):.1f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="mixed multi-namespace read+write load")
    ap.add_argument("--addr", default="localhost:50052")
    ap.add_argument("--namespaces", type=int, default=6)
    ap.add_argument("--prefix", default="mix")
    ap.add_argument("--scale", type=int, default=10_000, help="memories per namespace")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--types", type=int, default=1)
    ap.add_argument("--readers", type=int, default=12)
    ap.add_argument("--writers", type=int, default=4)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--write-batch", dest="write_batch", type=int, default=50)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--nprobe", type=int, default=0)
    ap.add_argument("--memory-type", dest="memory_type", type=int, default=1)
    ap.add_argument("--query-pool", dest="query_pool", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--skip-seed", action="store_true")
    ap.add_argument("--index-timeout", dest="index_timeout", type=float, default=300.0)
    args = ap.parse_args(argv)

    names = [f"{args.prefix}{i}" for i in range(args.namespaces)]

    if not args.skip_seed:
        for ns in names:
            print(f"[setup] seeding {ns} -> {args.scale}", flush=True)
            seed(args.addr, ns, args.scale, args.dim, args.types, 2000, args.seed)
        for ns in names:
            wait_indexed(args.addr, ns, args.scale, args.index_timeout)

    pools = {
        ns: build_query_pool(ns, args.dim, args.query_pool, args.nprobe, args.topk, False, args.seed, args.memory_type)
        for ns in names
    }
    rng = np.random.default_rng(args.seed)
    n_centers = max(1, int(math.sqrt(args.scale)))
    centers = _unit_rows(rng, n_centers, args.dim)

    before = cache_snapshot(args.addr, names[0])
    print(
        f"\n[mixed] {args.readers} readers + {args.writers} writers (batch {args.write_batch}) "
        f"across {len(names)} namespaces for {args.duration}s\n",
        flush=True,
    )
    reads, writes = asyncio.run(
        _run_mixed(args.addr, names, pools, args.readers, args.writers, args.duration,
                   args.dim, args.write_batch, args.types, centers, n_centers, args.timeout)
    )
    after = cache_snapshot(args.addr, names[0])

    print("\n=== RESULTS ===")
    _report("READ ", reads, args.duration)
    _report("WRITE", writes, args.duration)
    dh, dm = after["hits"] - before["hits"], after["misses"] - before["misses"]
    hr = dh / max(1, dh + dm)
    print(
        f"cache: hit_ratio={hr:.3f}  ({dh} hits / {dm} misses)  "
        f"mem={after['mem_bytes'] / 1e6:.0f}/{after['mem_budget'] / 1e6:.0f}MB  entries={after['total_entries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
