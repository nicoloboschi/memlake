"""Qdrant container lifecycle: reachable? else `docker compose up -d`.

Port selection order:
  1. QDRANT_URL / QDRANT_HTTP_PORT if the user set them
  2. an already-reachable Qdrant on the default port (reuse it, don't touch it)
  3. otherwise pick the first free port from DEFAULT_PORT upward and start ours
"""

from __future__ import annotations

import os
import socket
import subprocess
import time

import requests

from .paths import bench_dir, repo_root

DEFAULT_PORT = 6333
COMPOSE_FILE = "bench/docker-compose.qdrant.yml"


def _reachable(url: str, timeout: float = 1.5) -> bool:
    try:
        r = requests.get(f"{url}/readyz", timeout=timeout)
        if r.status_code == 200:
            return True
        # Older builds have no /readyz; the root endpoint reports the version.
        return requests.get(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _compose(args: list[str], env: dict) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-f", COMPOSE_FILE, *args]
    return subprocess.run(
        cmd, cwd=repo_root(), env={**os.environ, **env},
        capture_output=True, text=True,
    )


def ensure_running(timeout_s: float = 90.0) -> str:
    """Return a base URL for a reachable Qdrant, starting one if needed."""
    explicit = os.environ.get("QDRANT_URL")
    if explicit:
        if not _reachable(explicit):
            raise RuntimeError(f"QDRANT_URL={explicit} is set but not reachable")
        print(f"[qdrant] using QDRANT_URL={explicit}")
        return explicit

    port = int(os.environ.get("QDRANT_HTTP_PORT", DEFAULT_PORT))
    url = f"http://localhost:{port}"

    if _reachable(url):
        print(f"[qdrant] already reachable at {url}")
        return url

    # Port busy but not Qdrant -> something else owns it, shift up.
    if not _port_free(port):
        for cand in range(DEFAULT_PORT + 10, DEFAULT_PORT + 40):
            if _port_free(cand):
                print(f"[qdrant] port {port} taken by another service, using {cand}")
                port = cand
                url = f"http://localhost:{port}"
                break
        else:
            raise RuntimeError("no free port found for Qdrant")

    compose_path = bench_dir() / "docker-compose.qdrant.yml"
    if not compose_path.exists():
        raise FileNotFoundError(f"missing {compose_path}")

    env = {"QDRANT_HTTP_PORT": str(port), "QDRANT_GRPC_PORT": str(port + 1)}
    print(f"[qdrant] starting container on {url} via {COMPOSE_FILE}")
    proc = _compose(["up", "-d"], env)
    if proc.returncode != 0:
        raise RuntimeError(f"docker compose up failed:\n{proc.stderr}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _reachable(url):
            print(f"[qdrant] ready at {url}")
            return url
        time.sleep(1.0)
    raise TimeoutError(f"Qdrant did not become ready at {url} within {timeout_s}s")


def stop() -> None:
    proc = _compose(["down", "-v"], {})
    print("[qdrant] stopped" if proc.returncode == 0 else f"[qdrant] stop failed:\n{proc.stderr}")
