"""Canonical on-disk locations. Everything is derived from the repo root."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repo root = parent of the bench/ package tree.

    Overridable with MEMLAKE_ROOT for out-of-tree runs.
    """
    env = os.environ.get("MEMLAKE_ROOT")
    if env:
        return Path(env).resolve()
    # .../bench/src/memlake_bench/paths.py -> .../
    return Path(__file__).resolve().parents[3]


def bench_dir() -> Path:
    return repo_root() / "bench"


def testdata_dir() -> Path:
    return repo_root() / "testdata"


def beir_dir(dataset: str | None = None) -> Path:
    base = testdata_dir() / "beir"
    return base / dataset if dataset else base


def embeddings_dir(dataset: str | None = None) -> Path:
    base = testdata_dir() / "embeddings"
    return base / dataset if dataset else base


def results_dir(dataset: str | None = None) -> Path:
    base = bench_dir() / "results"
    return base / dataset if dataset else base


def report_path() -> Path:
    return results_dir() / "report.md"
