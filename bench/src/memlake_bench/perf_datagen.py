"""Synthetic memory generation for the e2e performance suite (Python port of the Rust
`mlake-perf` datagen). Reproducible under a seed; exercises every arm: clustered vectors
(IVF structure), text (FTS), Zipfian tags (realistic selectivity), Zipfian entity ids, a
fraction of causal edges (graph), across several memory_types.

Emits `memlake_client.memory` protos directly, so the harness streams them straight to the
server via the client.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from memlake_client import memory

_WORDS = [
    "memory", "recall", "vector", "graph", "lake", "index", "cluster", "query", "tag",
    "entity", "semantic", "episodic", "signal", "search", "bank", "fold", "shard", "probe",
]


@dataclass
class GenConfig:
    scale: int = 10_000
    memory_types: int = 3
    dim: int = 384
    tag_vocab: int = 2_000
    tags_per_memory: int = 3
    untagged_frac: float = 0.2
    entity_vocab: int = 5_000
    entities_per_memory: int = 2
    causal_frac: float = 0.05
    seed: int = 42


def _key(i: int) -> str:
    return f"m{i}"


def _id_bytes(i: int) -> bytes:
    return uuid.uuid5(uuid.NAMESPACE_OID, _key(i)).bytes


def _zipf_cumulative(n: int) -> np.ndarray:
    ranks = np.arange(1, max(n, 1) + 1, dtype=np.float64)
    return np.cumsum(1.0 / ranks)


class Generator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed ^ 0xC0FFEE)
        n_centers = max(int(np.ceil(np.sqrt(cfg.scale))), 1)
        centers = rng.uniform(-1.0, 1.0, size=(n_centers, cfg.dim))
        self.centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        self._zipf_tag = _zipf_cumulative(cfg.tag_vocab)
        self._zipf_entity = _zipf_cumulative(cfg.entity_vocab)

    def _zipf_sample(self, cum: np.ndarray, rng: np.random.Generator) -> int:
        target = rng.random() * cum[-1]
        return int(np.searchsorted(cum, target)) + 1

    def batch(self, start: int, end: int) -> list:
        """memlake_client.memory protos for id range [start, end)."""
        cfg = self.cfg
        out = []
        for i in range(start, end):
            rng = np.random.default_rng(cfg.seed + i)
            center = self.centers[i % len(self.centers)]
            vec = center + rng.uniform(-0.35, 0.35, size=cfg.dim)
            vec = vec / np.linalg.norm(vec)

            text = " ".join(_WORDS[rng.integers(0, len(_WORDS))] for _ in range(6))

            if rng.random() < cfg.untagged_frac:
                tags: list[str] = []
            else:
                ranks = {self._zipf_sample(self._zipf_tag, rng) for _ in range(cfg.tags_per_memory)}
                tags = sorted(f"tag-{r}" for r in ranks)

            entity_ids = sorted({self._zipf_sample(self._zipf_entity, rng) for _ in range(cfg.entities_per_memory)})

            out.append(
                memory(
                    text,
                    vector=[float(x) for x in vec],
                    memory_type=(i % max(cfg.memory_types, 1)) + 1,
                    key=_key(i),
                    tags=tags,
                    entity_ids=entity_ids,
                )
            )
        return out

    def query_vector(self, c: int, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        center = self.centers[c % len(self.centers)]
        v = center + rng.uniform(-0.2, 0.2, size=self.cfg.dim)
        v = v / np.linalg.norm(v)
        return [float(x) for x in v]

    @property
    def center_count(self) -> int:
        return len(self.centers)
