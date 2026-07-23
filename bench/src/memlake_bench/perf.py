"""End-to-end performance benchmark: client → gRPC → server → S3 (real MinIO).

Everything is measured through the Python client against a live `mlake-server`, so the
numbers are the deployed path, not a library micro-benchmark. Three phases:

* write  — stream memories via the client; measure throughput and request cost.
* index  — run the indexer once (its own process, as in prod); measure build time + cost.
* read   — cold then warm passes over several workloads; measure latency, roundtrips, cost.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from memlake_client import ANY, MemlakeClient

from . import server
from .perf_datagen import GenConfig, Generator

ADDR = "127.0.0.1:50251"
COMMIT_BATCH = 512
# Fold the WAL tail into a segment every this many written docs during the write phase, so the
# un-indexed tail stays bounded — as a continuously-running indexer keeps it in production. Write-
# time link derivation scans that tail, so an unbounded backfill would make the write path O(N^2);
# this mirrors the deployed topology where writer and indexer run concurrently.
INDEX_EVERY = 4096

# AWS S3 Standard, us-east-1 (same model as the Rust harness).
PUT_PER_1K = 0.005
GET_PER_1K = 0.0004
STORAGE_GB_MONTH = 0.023


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(int(p / 100.0 * len(s)), len(s) - 1)
    return s[i]


@dataclass
class WorkloadResult:
    name: str
    p50: float
    p90: float
    p99: float
    mean_rt: float
    usd_per_1k: float


@dataclass
class PerfReport:
    scale: int
    memory_types: int
    # write
    commit_secs: float
    write_per_s: float
    n_batches: int
    write_requests_usd: float
    # index
    index_secs: float
    index_puts: int
    stored_gb: float
    index_requests_usd: float
    storage_usd_month: float
    # read
    cold: list[WorkloadResult] = field(default_factory=list)
    warm: list[WorkloadResult] = field(default_factory=list)
    mem_mb: int = 0
    disk_mb: int = 0

    def render(self) -> str:
        lines = [
            f"perf @ {self.scale} ({self.memory_types} memory_types)  [e2e: python client -> gRPC -> server -> S3]",
            "",
            "  write:",
            f"    commit   {self.commit_secs:6.1f}s  ({self.write_per_s:,.0f} memories/s)"
            f"  {self.n_batches} batches  ${self.write_requests_usd:.4f} requests",
            "  index:",
            f"    build    {self.index_secs:6.1f}s  ({self.scale / max(self.index_secs, 1e-6):,.0f} memories/s)"
            f"  {self.index_puts} PUTs  {self.stored_gb:.3f} GB"
            f"  ${self.index_requests_usd:.4f} requests  ${self.storage_usd_month:.4f}/GB-month",
            "  read:",
        ]

        def rows(title: str, results: list[WorkloadResult]) -> None:
            lines.append(f"    {title}:")
            for r in results:
                lines.append(
                    f"      {r.name:<12} p50 {r.p50:6.2f}ms  p90 {r.p90:6.2f}ms  p99 {r.p99:6.2f}ms"
                    f"   rt {r.mean_rt:4.1f}   ${r.usd_per_1k:.4f}/1k"
                )

        rows("cold", self.cold)
        rows("warm", self.warm)
        lines.append(f"  cache budget: mem {self.mem_mb} MB   disk {self.disk_mb} MB (enforced by construction)")
        return "\n".join(lines)


def _run_3way(client, ns, qs, name, *, tags=None, tags_mode=ANY) -> WorkloadResult:
    """The one query pattern memlake serves: a single call across ALL memory_types with all
    three arms (dense + full-text + graph). Roundtrips are shared across types and arms, not
    multiplied. Reads are strongly consistent; a run of queries between writes reuses the warm
    cached snapshot after one cheap WAL-head check (0 fetch roundtrips warm)."""
    latencies, total_rt = [], 0
    for q in qs:
        t = time.perf_counter()
        client.query(
            ns, vector=q, text="memory vector",
            memory_types=None, tags=tags, tags_mode=tags_mode,
        )
        latencies.append((time.perf_counter() - t) * 1000.0)
        total_rt += client.last_roundtrips
    n = max(len(qs), 1)
    mean_rt = total_rt / n
    return WorkloadResult(
        name=name, p50=_pct(latencies, 50), p90=_pct(latencies, 90),
        p99=_pct(latencies, 99), mean_rt=mean_rt, usd_per_1k=mean_rt * GET_PER_1K,
    )


def run(
    cfg: GenConfig,
    *,
    namespace: str | None = None,
    queries: int = 200,
    mem_mb: int = 64,
    disk_mb: int = 512,
    addr: str = ADDR,
) -> PerfReport:
    # A unique namespace per run, so the index phase always measures a true *first build*
    # (fresh k-means training), not an incremental re-index of an already-trained namespace.
    ns = namespace or f"perf-py-{cfg.scale}-{uuid.uuid4().hex[:8]}"
    binary = server.build_binary()
    gen = Generator(cfg)

    # -- write phase -----------------------------------------------------------
    # Writer and indexer run concurrently in production; the harness mirrors that by folding every
    # INDEX_EVERY docs so the un-indexed tail — which write-time link derivation scans — stays
    # bounded. The interleaved folds are metered into the index totals but kept OUT of commit_secs,
    # so write throughput reflects durable-write + derivation cost, not the fold subprocess.
    n_batches = 0
    commit_secs = 0.0
    index_secs = 0.0
    index_puts = 0
    index_lists = 0
    stored_bytes = 0
    with server.Serve(binary, addr=addr, mem_mb=mem_mb, disk_mb=disk_mb) as _srv:
        client = MemlakeClient(addr)
        client.create_namespace(ns)
        start = 0
        since_index = 0
        while start < cfg.scale:
            end = min(start + COMMIT_BATCH, cfg.scale)
            batch = gen.batch(start, end)
            t0 = time.perf_counter()
            client.write(ns, batch)
            commit_secs += time.perf_counter() - t0
            n_batches += 1
            since_index += end - start
            start = end
            if since_index >= INDEX_EVERY:
                s = server.index_once(binary, ns)
                index_secs += s["elapsed_s"]
                index_puts += s["puts"]
                index_lists += s["lists"]
                stored_bytes += s["put_bytes"]
                since_index = 0
        client.close()

    # namespace creation is one PUT; each commit batch is one WAL PUT.
    write_requests_usd = (n_batches + 1) / 1000.0 * PUT_PER_1K

    # -- final catch-up fold (separate process, as in prod) -------------------
    summary = server.index_once(binary, ns)
    index_secs += summary["elapsed_s"]
    index_puts += summary["puts"]
    index_lists += summary["lists"]
    stored_bytes += summary["put_bytes"]
    index_requests_usd = (index_puts + index_lists) / 1000.0 * PUT_PER_1K
    stored_gb = stored_bytes / 1e9
    storage_usd_month = stored_gb * STORAGE_GB_MONTH

    # -- read phase (fresh serve => cold cache) -------------------------------
    qs = [gen.query_vector(i % gen.center_count, seed=1000 + i) for i in range(queries)]
    cold, warm = [], []
    with server.Serve(binary, addr=addr, mem_mb=mem_mb, disk_mb=disk_mb) as _srv:
        client = MemlakeClient(addr)
        # Two variants of the 3-way query: plain, and with a tag filter.
        variants = [
            ("3way", dict()),
            ("3way+tags", dict(tags=["tag-1", "tag-2"], tags_mode=ANY)),
        ]
        for name, kw in variants:
            cold.append(_run_3way(client, ns, qs, name, **kw))
            warm.append(_run_3way(client, ns, qs, name, **kw))
        client.close()

    return PerfReport(
        scale=cfg.scale,
        memory_types=cfg.memory_types,
        commit_secs=commit_secs,
        write_per_s=cfg.scale / max(commit_secs, 1e-6),
        n_batches=n_batches,
        write_requests_usd=write_requests_usd,
        index_secs=index_secs,
        index_puts=index_puts,
        stored_gb=stored_gb,
        index_requests_usd=index_requests_usd,
        storage_usd_month=storage_usd_month,
        cold=cold,
        warm=warm,
        mem_mb=mem_mb,
        disk_mb=disk_mb,
    )
