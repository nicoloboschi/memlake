"""Compute-vs-SLA sweep: run the harness across a grid of (scale, serve_cpus, concurrency)
against the real compose topology, then tabulate the rows.

For each scale it seeds once (data persists in the MinIO volume), then for each serve_cpus it
RECREATES the serve container at that CPU limit (a fresh, cold cache), and for each concurrency
runs a measurement window. The first concurrency at each cpu setting pays the warm-up that
populates the cache; subsequent windows are warm (read the per-row cache_hit_ratio).

  uv run --project perf memlake-perf-sweep \
      --scales 100000 --cpus 0.5,1,2 --concurrency 1,8,32,64

Holds serve memory + cache budget constant across the CPU sweep so only compute varies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO / "docker-compose.perf.yml"
# `--env-file` REPLACES the repo-root `.env` auto-load (which may hold real AWS creds); this
# pass is MinIO-only. The S3 config is also forced explicitly in base_env() as a second guard.
MINIO_ENV_FILE = REPO / "perf" / "minio.env"
COMPOSE = ["docker", "compose", "-f", str(COMPOSE_FILE), "--env-file", str(MINIO_ENV_FILE)]

# Held constant across the CPU sweep so the cpu->QPS relationship is isolated.
SERVE_MEM = os.environ.get("SERVE_MEM", "3g")
SERVE_MEM_MB = int(os.environ.get("SERVE_MEM_MB", "1536"))      # cache --mem-mb budget
SERVE_MEM_MB_TOTAL = 3072                                        # container limit, MB (label)
SERVE_DISK_MB = int(os.environ.get("SERVE_DISK_MB", "8192"))
INDEXER_CPUS = os.environ.get("INDEXER_CPUS", "6")
INDEXER_MEM = os.environ.get("INDEXER_MEM", "8g")
# The indexer service comes up (topology completeness) but its 5s loop is effectively disabled:
# at scale its incremental re-folds thrash and hang, so we fold deterministically with a single
# one-shot `index --once` from a clean cursor (completes in ~37s for 100k).
INDEXER_INTERVAL = os.environ.get("INDEXER_INTERVAL", "86400")


def compose(*args, env=None, check=True, capture=False):
    e = {**os.environ, **(env or {})}
    return subprocess.run(COMPOSE + list(args), env=e, check=check,
                          capture_output=capture, text=True)


def base_env(serve_cpus):
    return {
        "SERVE_CPUS": str(serve_cpus),
        "SERVE_MEM": SERVE_MEM,
        "SERVE_MEM_MB": str(SERVE_MEM_MB),
        "SERVE_DISK_MB": str(SERVE_DISK_MB),
        "INDEXER_CPUS": INDEXER_CPUS,
        "INDEXER_MEM": INDEXER_MEM,
        "INDEXER_INTERVAL": INDEXER_INTERVAL,
        # Force MinIO explicitly (shell env beats both --env-file and the repo .env), so a
        # stray real-AWS value in the environment can never redirect this pass to S3.
        "MEMLAKE_QUERY_S3_BUCKET": "memlake",
        "MEMLAKE_QUERY_S3_ACCESS_KEY": "memlake",
        "MEMLAKE_QUERY_S3_SECRET_KEY": "memlake123",
        "MEMLAKE_QUERY_S3_REGION": "us-east-1",
        "MEMLAKE_QUERY_S3_ENDPOINT": "http://minio:9000",
        "MEMLAKE_INDEXER_S3_BUCKET": "memlake",
        "MEMLAKE_INDEXER_S3_ACCESS_KEY": "memlake",
        "MEMLAKE_INDEXER_S3_SECRET_KEY": "memlake123",
        "MEMLAKE_INDEXER_S3_REGION": "us-east-1",
        "MEMLAKE_INDEXER_S3_ENDPOINT": "http://minio:9000",
    }


def serve_cid() -> str:
    out = compose("ps", "-q", "serve", capture=True)
    return out.stdout.strip().splitlines()[0].strip()


def wait_healthy(timeout=120) -> str:
    t0 = time.monotonic()
    cid = ""
    while time.monotonic() - t0 < timeout:
        try:
            cid = serve_cid()
        except Exception:
            cid = ""
        if cid:
            st = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", cid],
                capture_output=True, text=True)
            status = st.stdout.strip()
            if status == "healthy":
                return cid
        time.sleep(2)
    raise RuntimeError(f"serve not healthy within {timeout}s (cid={cid!r})")


def run_harness(extra, env):
    cmd = [sys.executable, "-m", "memlake_perf.harness", *extra]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env={**os.environ, **env}, check=True)


def fold_once(ns, env):
    """One-shot, deterministic full fold of the WAL tail into the generation. This is the
    reliable index path (the 5s loop's incremental re-folds hang at scale here)."""
    print(f"\n[fold] one-shot `index --once` for {ns} ...", flush=True)
    t0 = time.monotonic()
    subprocess.run(COMPOSE + ["run", "--rm", "--no-deps", "indexer",
                              "index", "--namespaces", ns, "--once"],
                   env={**os.environ, **env}, check=True, timeout=7200)
    print(f"[fold] done in {time.monotonic()-t0:.0f}s", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="memlake compute-vs-SLA sweep")
    ap.add_argument("--scales", default="100000")
    ap.add_argument("--cpus", default="0.5,1,2")
    ap.add_argument("--concurrency", default="1,8,32,64")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--seed-batch", type=int, default=2000)
    ap.add_argument("--backend", default="minio")
    ap.add_argument("--out", default=str(REPO / "perf" / "results" / "rows.jsonl"))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--keep-up", action="store_true", help="leave the stack running at the end")
    args = ap.parse_args(argv)

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    cpus_list = [float(c) for c in args.cpus.split(",") if c.strip()]
    conc_list = [int(c) for c in args.concurrency.split(",") if c.strip()]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Initial bring-up at the max cpu setting (fast seeding + indexing).
    seed_cpus = max(cpus_list)
    up_args = ["up", "-d"] + ([] if args.no_build else ["--build"])
    compose(*up_args, env=base_env(seed_cpus))
    wait_healthy(timeout=600)

    for scale in scales:
        ns = f"perf-{scale}"
        # Seed once; data + generation persist in the MinIO volume across serve recreations.
        print(f"\n######## SEED scale={scale} ns={ns} ########", flush=True)
        run_harness(
            ["--addr", "localhost:50051", "--namespace", ns, "--scale", str(scale),
             "--seed-batch", str(args.seed_batch), "--write-only"],
            env=base_env(seed_cpus),
        )
        fold_once(ns, base_env(seed_cpus))

        for cpus in cpus_list:
            print(f"\n######## RECREATE serve cpus={cpus} (cold cache) ########", flush=True)
            env = base_env(cpus)
            compose("up", "-d", "--force-recreate", "--no-deps", "serve", env=env)
            cid = wait_healthy(timeout=180)
            time.sleep(3)

            for i, conc in enumerate(conc_list):
                # First window at this cpu setting warms the (fresh) cache; keep warmup > 0 so
                # every measured window is steady-state warm.
                run_harness(
                    ["--addr", "localhost:50051", "--namespace", ns, "--scale", str(scale),
                     "--skip-seed", "--concurrency", str(conc),
                     "--duration", str(args.duration), "--warmup", str(args.warmup),
                     "--serve-container", cid,
                     "--serve-cpus", str(cpus),
                     "--serve-mem-mb", str(SERVE_MEM_MB_TOTAL),
                     "--cache-mem-mb", str(SERVE_MEM_MB),
                     "--backend", args.backend,
                     "--out", args.out],
                    env=env,
                )

    tabulate(args.out, scales)

    if not args.keep_up:
        print("\n[sweep] leaving stack UP (use `docker compose -f docker-compose.perf.yml down -v` to stop)")
    return 0


def tabulate(out_path, scales):
    rows = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    cols = ["scale", "backend", "serve_cpus", "cache_mem_mb", "concurrency", "achieved_qps",
            "qps_per_vcpu", "p50", "p90", "p99", "cpu_pct_mean", "cpu_util_of_limit",
            "mean_load_roundtrips", "cache_hit_ratio", "errors"]
    print("\n\n=== SWEEP TABLE ===")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    print(header)
    print(sep)
    for r in sorted(rows, key=lambda r: (r.get("scale", 0), r.get("serve_cpus", 0), r.get("concurrency", 0))):
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


if __name__ == "__main__":
    raise SystemExit(main())
