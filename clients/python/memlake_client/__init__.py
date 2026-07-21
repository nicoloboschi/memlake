"""Python client for the memlake gRPC server.

    from memlake_client import MemlakeClient, memory

    with MemlakeClient("localhost:50051") as c:
        c.create_namespace("my-bank")
        c.write("my-bank", [memory("hello world", vector=[...], memory_type=1)])
        hits = c.query("my-bank", memory_type=1, vector=[...], text="hello", top_k=5)
"""

from .client import (
    ALL,
    ALL_STRICT,
    ANY,
    ANY_STRICT,
    EVENTUAL,
    EXACT,
    STRONG,
    Hit,
    MemlakeClient,
    memory,
)

__all__ = [
    "MemlakeClient",
    "memory",
    "Hit",
    "ANY",
    "ALL",
    "ANY_STRICT",
    "ALL_STRICT",
    "EXACT",
    "STRONG",
    "EVENTUAL",
]
