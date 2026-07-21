"""A local multi-node memlake cluster for chaos/correctness testing.

Spawns N real `mlake-server serve` processes and M `index` loops against the one local MinIO,
each with a distinct `--node-id`, port, and read-cache dir. Nodes can be SIGKILLed and
restarted mid-run — that is the whole point: every server is stateless (all coordination is in
object storage), so killing one must never lose an acked write or wedge the namespace.

This is deliberately process-level, not in-process: it exercises the real gRPC path, the smart
client's routing/failover, and the on-disk CAS/lease protocol exactly as production would.
"""

from __future__ import annotations

import contextlib
import signal
import socket
import subprocess
import time
from pathlib import Path

from .paths import repo_root
from .server import _env, _wait_port, build_binary


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServeNode:
    """One `mlake-server serve` process. Restartable and kill-able."""

    def __init__(self, binary: Path, node_id: str, *, addr: str, cache_dir: str,
                 mem_mb: int = 64, disk_mb: int = 512, log_path: str | None = None):
        self.binary = binary
        self.node_id = node_id
        self.addr = addr
        self.cache_dir = cache_dir
        self.mem_mb = mem_mb
        self.disk_mb = disk_mb
        self.log_path = log_path or f"/tmp/memlake-chaos-{node_id}.log"
        self._proc: subprocess.Popen | None = None
        self.starts = 0
        self.kills = 0

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, *, wait: bool = True) -> None:
        # Keep the cache across restarts (a warm node rejoining is the realistic case); MinIO is
        # the source of truth regardless.
        log = open(self.log_path, "a")
        self._log = log
        self._proc = subprocess.Popen(
            [str(self.binary), "serve",
             "--addr", self.addr,
             "--node-id", self.node_id,
             "--mem-mb", str(self.mem_mb),
             "--disk-mb", str(self.disk_mb),
             "--cache-dir", self.cache_dir],
            cwd=repo_root(), env=_env(), stdout=log, stderr=subprocess.STDOUT,
        )
        self.starts += 1
        if wait:
            host, port = self.addr.split(":")
            _wait_port(host, int(port), timeout=30.0)

    def kill(self) -> None:
        """Hard SIGKILL — the ungraceful failure mode a chaos test must survive."""
        if self.alive:
            self._proc.send_signal(signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=5)
            self.kills += 1

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=10)
            if self._proc.poll() is None:
                self._proc.send_signal(signal.SIGKILL)
        with contextlib.suppress(Exception):
            self._log.close()


class IndexNode:
    """One `mlake-server index` loop. Folds all namespaces on an interval; the soft lease keeps
    peers from duplicating work."""

    def __init__(self, binary: Path, node_id: str, *, interval_secs: int = 1,
                 namespaces: list[str] | None = None, log_path: str | None = None):
        self.binary = binary
        self.node_id = node_id
        self.interval_secs = interval_secs
        self.namespaces = namespaces or []
        self.log_path = log_path or f"/tmp/memlake-chaos-{node_id}.log"
        self._proc: subprocess.Popen | None = None
        self.starts = 0
        self.kills = 0

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        log = open(self.log_path, "a")
        self._log = log
        cmd = [str(self.binary), "index", "--node-id", self.node_id,
               "--interval-secs", str(self.interval_secs)]
        if self.namespaces:
            # Scope the fold loop to just these namespaces — faster, and it never touches
            # unrelated (possibly stale) namespaces in the shared bucket.
            cmd += ["--namespaces", ",".join(self.namespaces)]
        self._proc = subprocess.Popen(
            cmd, cwd=repo_root(), env=_env(), stdout=log, stderr=subprocess.STDOUT,
        )
        self.starts += 1

    def kill(self) -> None:
        if self.alive:
            self._proc.send_signal(signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=5)
            self.kills += 1

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=10)
            if self._proc.poll() is None:
                self._proc.send_signal(signal.SIGKILL)
        with contextlib.suppress(Exception):
            self._log.close()


class Cluster:
    """A set of serve nodes + index loops, as a context manager."""

    def __init__(self, *, nodes: int = 3, indexers: int = 2, index_interval: int = 1,
                 index_namespaces: list[str] | None = None,
                 mem_mb: int = 64, disk_mb: int = 512, cache_root: str = "/tmp/memlake-chaos"):
        self.binary = build_binary()
        self.serve_nodes: list[ServeNode] = [
            ServeNode(self.binary, f"serve-{i}", addr=f"127.0.0.1:{_free_port()}",
                      cache_dir=f"{cache_root}/serve-{i}", mem_mb=mem_mb, disk_mb=disk_mb)
            for i in range(nodes)
        ]
        self.index_nodes: list[IndexNode] = [
            IndexNode(self.binary, f"index-{i}", interval_secs=index_interval,
                      namespaces=index_namespaces)
            for i in range(indexers)
        ]

    @property
    def addrs(self) -> list[str]:
        return [n.addr for n in self.serve_nodes]

    def live_addrs(self) -> list[str]:
        return [n.addr for n in self.serve_nodes if n.alive]

    def __enter__(self) -> "Cluster":
        subprocess.run(["rm", "-rf", *[n.cache_dir for n in self.serve_nodes]], check=False)
        for n in self.serve_nodes:
            n.start()
        for n in self.index_nodes:
            n.start()
        return self

    def __exit__(self, *_exc) -> None:
        for n in self.index_nodes:
            n.stop()
        for n in self.serve_nodes:
            n.stop()

    def ensure_all_up(self, timeout: float = 40.0) -> None:
        """Bring every serve node back up and block until all their gRPC ports accept
        connections. Used to quiesce before the final correctness assertion, so a per-node
        cross-check never races a node that is still (re)starting."""
        for n in self.serve_nodes:
            if not n.alive:
                n.start(wait=False)
        for n in self.serve_nodes:
            host, port = n.addr.split(":")
            _wait_port(host, int(port), timeout=timeout)

    def kill_random_serve(self, rng) -> ServeNode | None:
        """Kill one currently-alive serve node (leaving at least one alive)."""
        alive = [n for n in self.serve_nodes if n.alive]
        if len(alive) <= 1:
            return None
        victim = rng.choice(alive)
        victim.kill()
        return victim
