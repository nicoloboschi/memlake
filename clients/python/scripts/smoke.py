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
                memory("the cat sat on the mat", vec(0.1), memory_type=1, key="m1",
                       tags=["animals"], metadata={"document_id": "doc-1", "context": "pets"}),
                memory("a dog barked loudly", vec(0.2), memory_type=1, key="m2",
                       tags=["animals"], metadata={"document_id": "doc-1"}),
                memory("stock prices rose today", vec(0.9), memory_type=1, key="m3",
                       tags=["finance"], metadata={"document_id": "doc-2"}),
            ],
        )
        print(f"wrote batch, WAL seq={seq}")

        # STRONG consistency (default): visible immediately via the WAL tail, no indexing.
        # ONE call, all memory_types, all three arms; each hit carries the raw per-arm signals
        # AND the materialized memory (text + metadata) inline — no second round trip.
        hits = c.query(NS, vector=vec(0.1), text="cat")
        print(f"query -> {len(hits)} hits (roundtrips={c.last_roundtrips})")
        for h in hits:
            print(
                f"  mt={h.memory_type} {h.id_uuid[:8]}  "
                f"dense={h.dense.score:.4f}@{h.dense.rank if h.dense.present else '-'}  "
                f"text={h.text.score:.4f}@{h.text.rank if h.text.present else '-'}  "
                f"graph={'y' if h.graph.present else '-'}  "
                f"text={h.memory.text!r}  metadata={h.memory.metadata}"
            )

        # With a tag filter.
        hits = c.query(NS, vector=vec(0.15), text="prices", tags=["finance"], tags_mode=ANY)
        print(f"query tag=finance -> {len(hits)} hits")

        assert hits is not None
        print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
