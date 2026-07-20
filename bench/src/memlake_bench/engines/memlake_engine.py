"""memlake engine driver.

The Rust binary (`crates/mlake-bench`) does the retrieval: it loads the same cached
embeddings Qdrant used, builds a memlake generation, and writes per-query rankings (a
"run") to JSON. This module builds and invokes that binary, then scores the run with the
*same* metrics code used for every other engine — so the only thing that differs between
memlake and Qdrant in the report is the ranking, never the measurement.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import metrics
from ..datasets import Beir
from ..paths import repo_root, results_dir


def _cargo_build() -> Path:
    """Build the release binary and return its path."""
    root = repo_root()
    subprocess.run(
        ["cargo", "build", "--release", "-p", "mlake-bench"],
        cwd=root,
        check=True,
    )
    binary = root / "target" / "release" / "mlake-bench"
    if not binary.exists():
        raise FileNotFoundError(f"mlake-bench binary not found at {binary}")
    return binary


def run(beir: Beir, *, env: dict | None = None) -> dict:
    """Run memlake over a dataset and return a scored results payload."""
    root = repo_root()
    binary = _cargo_build()
    run_path = results_dir(beir.name) / "memlake.run.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)

    proc_env = None
    if env:
        import os

        proc_env = {**os.environ, **{k: str(v) for k, v in env.items()}}

    subprocess.run(
        [str(binary), beir.name, "testdata", str(run_path)],
        cwd=root,
        check=True,
        env=proc_env,
    )

    raw = json.loads(run_path.read_text(encoding="utf-8"))

    arms_out: dict[str, dict] = {}
    for arm_name, arm in raw["arms"].items():
        arms_out[arm_name] = metrics.evaluate(
            arm["run"], beir.qrels, arm.get("latencies_ms", [])
        )

    return {
        "engine": "memlake",
        "config": {
            "note": "in-process IVF vector + hand-rolled BM25 + weighted RRF, "
            "same cached bge-small vectors as qdrant",
            **raw.get("config", {}),
        },
        "corpus_size": raw["corpus_size"],
        "n_queries": raw["n_queries"],
        "index_seconds": round(raw["index_seconds"], 3),
        "arms": arms_out,
    }
