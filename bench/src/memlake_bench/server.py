"""Manage the memlake server as subprocesses for e2e benchmarking.

The perf harness measures the *deployed* path — client → gRPC → server → S3 — so it drives
the real `mlake-server` binary rather than any in-process engine. Two modes are used:

* `serve`  — the stateless gRPC API, started as a long-lived child for the write/read phases.
* `index --once` — a single metered index pass; run to completion, its JSON summary parsed
  for build time and cost (mirrors how the indexer runs as its own Deployment in prod).
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from .paths import repo_root


def build_binary() -> Path:
    root = repo_root()
    subprocess.run(
        ["cargo", "build", "--release", "-p", "mlake-server"],
        cwd=root,
        check=True,
    )
    binary = root / "target" / "release" / "mlake-server"
    if not binary.exists():
        raise FileNotFoundError(f"mlake-server binary not found at {binary}")
    return binary


def _env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    # The server reads service-scoped S3 config: the query service from MEMLAKE_QUERY_S3_* and the
    # indexer from MEMLAKE_INDEXER_S3_* (no unprefixed fallback). Point both at the local MinIO.
    for prefix in ("MEMLAKE_QUERY", "MEMLAKE_INDEXER"):
        env.setdefault(f"{prefix}_S3_ENDPOINT", "http://localhost:9000")
        env.setdefault(f"{prefix}_S3_BUCKET", "memlake")
        env.setdefault(f"{prefix}_S3_ACCESS_KEY", "memlake")
        env.setdefault(f"{prefix}_S3_SECRET_KEY", "memlake123")
        env.setdefault(f"{prefix}_S3_REGION", "us-east-1")
    env.setdefault("RUST_LOG", "warn")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _wait_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.suppress(OSError):
            with socket.create_connection((host, port), timeout=1.0):
                return
        time.sleep(0.1)
    raise TimeoutError(f"server did not open {host}:{port} within {timeout}s")


class Serve:
    """A running `mlake-server serve` process, with a bounded read cache. Use as a context
    manager; restart between passes to get a genuinely cold cache."""

    def __init__(
        self,
        binary: Path,
        *,
        addr: str = "127.0.0.1:50051",
        mem_mb: int = 256,
        disk_mb: int = 4096,
        cache_dir: str | None = None,
        log_path: str | None = None,
    ):
        self.binary = binary
        self.addr = addr
        self.mem_mb = mem_mb
        self.disk_mb = disk_mb
        self.cache_dir = cache_dir or "/tmp/memlake-bench-cache"
        self.log_path = log_path or "/tmp/memlake-bench-serve.log"
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "Serve":
        # Fresh cache dir so a cold pass is genuinely cold.
        subprocess.run(["rm", "-rf", self.cache_dir], check=False)
        log = open(self.log_path, "w")
        self._log = log
        self._proc = subprocess.Popen(
            [
                str(self.binary), "serve",
                "--addr", self.addr,
                "--mem-mb", str(self.mem_mb),
                "--disk-mb", str(self.disk_mb),
                "--cache-dir", self.cache_dir,
            ],
            cwd=repo_root(),
            env=_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        host, port = self.addr.split(":")
        _wait_port(host, int(port))
        return self

    def __exit__(self, *_exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=10)
            if self._proc.poll() is None:
                self._proc.kill()
        with contextlib.suppress(Exception):
            self._log.close()


def index_once(binary: Path, namespace: str) -> dict:
    """Run one metered index pass and return the parsed JSON summary
    ({elapsed_s, puts, lists, gets, put_bytes, get_bytes, docs})."""
    proc = subprocess.run(
        [str(binary), "index", "--once", "--namespaces", namespace],
        cwd=repo_root(),
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    # The summary is the last JSON line on stdout.
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"index --once produced no summary. stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
