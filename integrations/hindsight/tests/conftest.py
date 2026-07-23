"""Fixtures for the memlake store's integration tests.

These are *real* tests: they drive a live ``mlake-server`` over gRPC, against a
real MinIO, exactly as the store runs in production. What they deliberately do
NOT do is spin up Hindsight — that would pull in Postgres, embeddings, an LLM and
the whole retain/recall stack, and none of it is what these tests are about. The
store is checked at its own seam: the :class:`MemoriesExtension` methods, called
directly with the same inputs Hindsight would hand them.

Two things have to be present, and the suite skips (never fails) without either:

* **A MinIO on ``localhost:9000``** — ``docker compose up -d`` from the repo root.
* **hindsight-api-slim with the memories seam** — the interface and the
  ``RetrievalResult`` / config types the store is written against. It is
  unreleased, so run the tests with it made importable::

      uv run --with-editable /path/to/hindsight-api-slim \
             --group dev pytest integrations/hindsight

The server binary is built once per session (``cargo build --release``) and one
``serve`` process is shared across the module; indexing is a separate
``index --once`` pass, mirroring how the indexer runs as its own Deployment.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest


def _wire_hindsight_source() -> None:
    """Put a local hindsight-api-slim checkout with the memories seam on the path.

    The seam is unreleased, so the ``hindsight-api-slim`` a plain ``uv sync``
    installs (from PyPI) does not have it, and would shadow anything layered on
    top. This inserts a source checkout that *does* have it ahead of site-packages
    so its ``hindsight_api`` wins — its third-party dependencies are still the
    installed ones, which is all we need.

    The checkout is found via ``HINDSIGHT_API_SLIM_PATH``, or by looking for a
    sibling of the memlake repo. If none has the seam, nothing is inserted and the
    module skips below.
    """
    # find_spec raises (not returns None) when a parent package is importable but
    # the submodule is missing — exactly the published-without-the-seam case.
    with contextlib.suppress(ModuleNotFoundError):
        if importlib.util.find_spec("hindsight_api.engine.memories.base") is not None:
            return

    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    env_path = os.environ.get("HINDSIGHT_API_SLIM_PATH")
    if env_path:
        candidates.append(Path(env_path))
    # Sibling checkouts of the memlake repo — the usual dev layout.
    for sibling in ("hindsight", "hindsight-ext", "hindsight-api"):
        candidates.append(repo_root.parent / sibling / "hindsight-api-slim")

    for candidate in candidates:
        if (candidate / "hindsight_api" / "engine" / "memories" / "base.py").is_file():
            sys.path.insert(0, str(candidate))
            # Drop any half-imported published `hindsight_api` so the source wins.
            for name in [m for m in sys.modules if m == "hindsight_api" or m.startswith("hindsight_api.")]:
                del sys.modules[name]
            return


_wire_hindsight_source()

# The store leans on hindsight-api-slim for the interface it implements and the
# result/config types it produces. If it still isn't importable the seam isn't
# available anywhere, so skip the whole module rather than erroring at collection.
pytest.importorskip(
    "hindsight_api.engine.memories.base",
    reason="hindsight-api-slim with the MemoriesExtension seam is not importable — "
    "set HINDSIGHT_API_SLIM_PATH to a checkout that has it",
)

MINIO_ENDPOINT = os.environ.get("MEMLAKE_TEST_S3_ENDPOINT", "http://localhost:9000")
MINIO_HOST, MINIO_PORT = MINIO_ENDPOINT.split("://", 1)[1].split(":")
SERVER_ADDR = os.environ.get("MEMLAKE_TEST_ADDR", "127.0.0.1:50077")


def _repo_root() -> Path:
    # tests/ -> hindsight/ -> integrations/ -> repo root
    return Path(__file__).resolve().parents[3]


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with contextlib.suppress(OSError), socket.create_connection((host, int(port)), timeout=timeout):
        return True
    return False


def _server_env() -> dict:
    """Point both server services at the local MinIO, quiet the logs.

    These are set unconditionally, not via ``setdefault``: the memlake repo ships
    a ``.env`` with real-cloud credentials, and both the ambient shell and the
    server's own dotenv load would otherwise win and send the tests at a bucket
    they can't reach. The MinIO target is overridable through ``MEMLAKE_TEST_S3_*``
    for a non-default local setup.
    """
    env = dict(os.environ)
    for prefix in ("MEMLAKE_QUERY", "MEMLAKE_INDEXER"):
        env[f"{prefix}_S3_ENDPOINT"] = MINIO_ENDPOINT
        env[f"{prefix}_S3_BUCKET"] = os.environ.get("MEMLAKE_TEST_S3_BUCKET", "memlake")
        env[f"{prefix}_S3_ACCESS_KEY"] = os.environ.get("MEMLAKE_TEST_S3_ACCESS_KEY", "memlake")
        env[f"{prefix}_S3_SECRET_KEY"] = os.environ.get("MEMLAKE_TEST_S3_SECRET_KEY", "memlake123")
        env[f"{prefix}_S3_REGION"] = "us-east-1"
    env.setdefault("RUST_LOG", "warn")
    return env


@pytest.fixture(scope="session")
def server_binary() -> Path:
    """The ``mlake-server`` binary to test against.

    Built fresh by default. If the build fails — the workspace can be mid-refactor
    while the integration is worked on independently — it falls back to a
    previously built binary rather than blocking these tests on unrelated compile
    errors. ``MEMLAKE_TEST_SERVER_BINARY`` overrides both. Skips if MinIO is not
    reachable, since there is nothing to test against.
    """
    if not _port_open(MINIO_HOST, MINIO_PORT):
        pytest.skip(f"MinIO not reachable at {MINIO_ENDPOINT} — start it with `docker compose up -d`")

    override = os.environ.get("MEMLAKE_TEST_SERVER_BINARY")
    if override:
        binary = Path(override)
        if not binary.exists():
            pytest.skip(f"MEMLAKE_TEST_SERVER_BINARY={binary} does not exist")
        return binary

    root = _repo_root()
    built = subprocess.run(["cargo", "build", "--release", "-p", "mlake-server"], cwd=root)
    if built.returncode == 0 and (root / "target" / "release" / "mlake-server").exists():
        return root / "target" / "release" / "mlake-server"

    # Build failed (or produced nothing) — use the newest prebuilt binary if any.
    prebuilt = [
        p
        for p in (root / "target" / "release" / "mlake-server", root / "target" / "debug" / "mlake-server")
        if p.exists()
    ]
    if not prebuilt:
        pytest.skip("mlake-server did not build and no prebuilt binary was found")
    return max(prebuilt, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="session")
def index_pass(server_binary: Path):
    """Run one ``index --once`` over a namespace, making its writes searchable.

    Strong-consistency reads (get, scan-of-tail) see a write immediately; the
    dense and full-text arms only see it once it has been folded into a
    generation, which is what this does.
    """

    def _index(namespace: str) -> None:
        subprocess.run(
            [str(server_binary), "index", "--once", "--namespaces", namespace],
            cwd=_repo_root(),
            env=_server_env(),
            check=True,
            capture_output=True,
            text=True,
        )

    return _index


@pytest.fixture(scope="session")
def _serve(server_binary: Path):
    """One long-lived ``serve`` process for the session."""
    log = open("/tmp/memlake-ext-test-serve.log", "w")
    proc = subprocess.Popen(
        [str(server_binary), "serve", "--addr", SERVER_ADDR, "--cache-dir", "/tmp/memlake-ext-test-cache"],
        cwd=_repo_root(),
        env=_server_env(),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    host, port = SERVER_ADDR.split(":")
    deadline = time.time() + 30
    while time.time() < deadline and not _port_open(host, port):
        if proc.poll() is not None:
            raise RuntimeError("mlake-server exited during startup; see /tmp/memlake-ext-test-serve.log")
        time.sleep(0.1)
    try:
        yield SERVER_ADDR
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
        log.close()


@pytest.fixture
async def store(_serve: str):
    """A live :class:`MemlakeMemories` pointed at the session's server.

    Function-scoped so each test opens its own client; the server and its S3 are
    shared. Namespaces are per-test (see :func:`bank_id`), so tests never collide
    even though they share one bucket.
    """
    from hindsight_memlake import MemlakeMemories

    memories = MemlakeMemories({"target": _serve})
    await memories.initialize()
    try:
        yield memories
    finally:
        await memories.shutdown()


@pytest.fixture
def bank_id() -> str:
    """A unique bank (namespace) per test, so nothing leaks between them."""
    return f"ext-test-{uuid.uuid4().hex[:12]}"
