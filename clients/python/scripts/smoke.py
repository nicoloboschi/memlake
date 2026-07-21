"""End-to-end smoke test against a running memlake server.

    mlake-server serve --addr 127.0.0.1:50051      # terminal 1
    mlake-server index --namespaces smoke-py       # terminal 2 (or wait for it to run)
    uv run --project clients/python clients/python/scripts/smoke.py

Writes a few memories, queries before indexing (STRONG consistency -> served from the WAL
tail), and prints the hits.
"""

import sys

from memlake_client import ANY, MemlakeClient, memory

NS = "smoke-py"
DIM = 8


def vec(seed: float) -> list[float]:
    return [seed + i * 0.01 for i in range(DIM)]


def main() -> int:
    with MemlakeClient("127.0.0.1:50051") as c:
        c.create_namespace(NS)
        seq = c.write(
            NS,
            [
                memory("the cat sat on the mat", vec(0.1), memory_type=1, key="m1", tags=["animals"]),
                memory("a dog barked loudly", vec(0.2), memory_type=1, key="m2", tags=["animals"]),
                memory("stock prices rose today", vec(0.9), memory_type=1, key="m3", tags=["finance"]),
            ],
        )
        print(f"wrote batch, WAL seq={seq}")

        # STRONG consistency: visible immediately via the WAL tail, no indexing needed.
        hits = c.query(NS, memory_type=1, text="cat", top_k=5)
        print(f"fts 'cat' -> {len(hits)} hits")
        for h in hits:
            print(f"  {h.id_uuid}  score={h.score:.4f}  {h.contributions}")

        hits = c.query(NS, memory_type=1, vector=vec(0.1), top_k=3)
        print(f"vector near m1 -> {len(hits)} hits")
        for h in hits:
            print(f"  {h.id_uuid}  score={h.score:.4f}")

        hits = c.query(NS, memory_type=1, vector=vec(0.15), tags=["finance"], tags_mode=ANY, top_k=5)
        print(f"tag=finance -> {len(hits)} hits")

        assert hits is not None
        print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
