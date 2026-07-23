"""A never-ending trickle writer for the memlake demo.

Writes a small batch of synthetic memories into one namespace every `--interval` seconds,
forever, so the admin UI (index stats, scan, query workbench) visibly moves: the doc count
climbs, the un-indexed WAL tail grows and then drains as the indexer folds it, and new
clusters/links appear in the graph arm.

It is deliberately gentle — a handful of records per tick, not a load test. Vectors are drawn
around a few fixed centres (384-dim, matching the admin's bge-small embedding size) so similar
memories cluster and the write-time semantic links have something to connect.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
import uuid

from memlake_client import MemlakeClient, memory

# A small vocabulary so the generated text is human-readable in the admin scan.
_SUBJECTS = [
    "the indexer", "a query node", "the WAL tail", "a compaction", "the graph arm",
    "a semantic link", "the manifest", "a segment flush", "the vector arm", "a tombstone",
]
_VERBS = ["folded", "served", "expanded", "reranked", "probed", "superseded", "merged", "cached"]
_OBJECTS = [
    "three clusters", "a cold read", "the memory bank", "an entity posting", "a stale generation",
    "the top-k window", "a re-upsert", "the reverse adjacency", "a bounded neighbourhood",
]
_TAGS = [f"topic-{i}" for i in range(8)]


def _unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _centres(count: int, dim: int, rng: random.Random) -> list[list[float]]:
    """A few fixed cluster centres; every written vector is one centre plus noise, so records
    naturally group and semantic kNN links form within a group."""
    return [_unit([rng.gauss(0.0, 1.0) for _ in range(dim)]) for _ in range(count)]


def _sentence(rng: random.Random) -> str:
    return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}"


def build_batch(centres, dim, size, rng, tick):
    """One batch: `size` memories, each near a random centre, with readable text and a tag."""
    batch = []
    for j in range(size):
        centre = rng.choice(centres)
        vec = _unit([c + rng.gauss(0.0, 0.15) for c in centre])
        batch.append(
            memory(
                f"tick {tick}: {_sentence(rng)}",
                vector=vec,
                memory_type=1,
                # A fresh id every time so the doc count climbs (never a re-upsert).
                key=f"demo-{tick}-{j}-{uuid.uuid4().hex[:8]}",
                tags=[rng.choice(_TAGS)],
                entity_ids=[uuid.uuid5(uuid.NAMESPACE_OID, rng.choice(_OBJECTS)).bytes],
            )
        )
    return batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Trickle synthetic memories into a memlake namespace forever.")
    p.add_argument("--addr", default=os.environ.get("MEMLAKE_ADDR", "localhost:50051"),
                   help="host:port of mlake-server serve (env: MEMLAKE_ADDR)")
    p.add_argument("--namespace", default=os.environ.get("MEMLAKE_DEMO_NAMESPACE", "demo"),
                   help="namespace to write into (env: MEMLAKE_DEMO_NAMESPACE)")
    p.add_argument("--interval", type=float, default=float(os.environ.get("MEMLAKE_DEMO_INTERVAL", "2.0")),
                   help="seconds between batches")
    p.add_argument("--batch", type=int, default=int(os.environ.get("MEMLAKE_DEMO_BATCH", "5")),
                   help="memories per batch")
    p.add_argument("--dim", type=int, default=int(os.environ.get("MEMLAKE_DEMO_DIM", "384")),
                   help="embedding dimension (384 matches the admin's bge-small)")
    p.add_argument("--clusters", type=int, default=int(os.environ.get("MEMLAKE_DEMO_CLUSTERS", "6")),
                   help="number of vector cluster centres")
    p.add_argument("--seed", type=int, default=int(os.environ.get("MEMLAKE_DEMO_SEED", "7")))
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    centres = _centres(args.clusters, args.dim, rng)

    # Retry the initial connect so `docker compose up` ordering (serve still booting) is forgiving.
    client = MemlakeClient(args.addr)
    for attempt in range(60):
        try:
            client.create_namespace(args.namespace)
            break
        except Exception as e:  # noqa: BLE001 — any transport error is worth retrying at startup
            reason = (str(e).splitlines() or [type(e).__name__])[0]
            print(f"[writer] waiting for {args.addr} ({reason}) ...", flush=True)
            time.sleep(2.0)
    else:
        print(f"[writer] could not reach {args.addr} after 60 attempts, giving up", file=sys.stderr)
        return 1

    print(f"[writer] writing {args.batch} memories every {args.interval}s into "
          f"namespace '{args.namespace}' at {args.addr} — Ctrl-C to stop", flush=True)

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    tick = 0
    total = 0
    while not stop["now"]:
        try:
            seq = client.write(args.namespace, build_batch(centres, args.dim, args.batch, rng, tick))
            total += args.batch
            print(f"[writer] tick {tick}: +{args.batch} memories (total {total}), WAL seq {seq}", flush=True)
        except Exception as e:  # noqa: BLE001 — keep trickling across transient server hiccups
            print(f"[writer] tick {tick} failed: {e}", file=sys.stderr, flush=True)
        tick += 1
        # Sleep in small slices so a stop signal is honoured promptly.
        slept = 0.0
        while slept < args.interval and not stop["now"]:
            time.sleep(min(0.25, args.interval - slept))
            slept += 0.25

    client.close()
    print(f"[writer] stopped after {total} memories", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
