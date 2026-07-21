"""Python client for the memlake gRPC server.

    from memlake_client import MemlakeClient, memory

    with MemlakeClient("localhost:50051") as c:
        c.create_namespace("my-bank")
        c.write("my-bank", [memory("hello world", vector=[...], memory_type=1)])
        # one call, all memory_types, all 3 arms; each hit carries raw per-arm scores.
        hits = c.query("my-bank", vector=[...], text="hello")
        for h in hits:
            print(h.memory_type, h.dense.score, h.text.score, h.graph.score)
"""

from .client import (
    ALL,
    ALL_STRICT,
    ANY,
    ANY_STRICT,
    EVENTUAL,
    EXACT,
    STRONG,
    Arm,
    Hit,
    MemlakeClient,
    Payload,
    memory,
)

__all__ = [
    "MemlakeClient",
    "memory",
    "Hit",
    "Arm",
    "Payload",
    "ANY",
    "ALL",
    "ANY_STRICT",
    "ALL_STRICT",
    "EXACT",
    "STRONG",
    "EVENTUAL",
]
