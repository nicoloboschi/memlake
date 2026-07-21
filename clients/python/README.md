# memlake-client

Python gRPC client for the memlake server. Speaks the contract in
[`proto/memlake/v1/memlake.proto`](../../proto/memlake/v1/memlake.proto).

```python
from memlake_client import MemlakeClient, memory, ANY_STRICT

with MemlakeClient("localhost:50051") as c:
    c.create_namespace("my-bank")
    c.write("my-bank", [
        memory("s3-native retrieval", vector=embedding, memory_type=1, tags=["prod"]),
    ])
    hits = c.query(
        "my-bank", memory_type=1,
        vector=query_embedding, text="retrieval",
        tags=["prod"], tags_mode=ANY_STRICT, top_k=10,
    )
    for h in hits:
        print(h.id_uuid, h.score, h.contributions)
```

Vectors travel as raw little-endian float32 (`Vector.f32le`) — the client packs/unpacks for
you. A `query` opens a fresh consistent snapshot server-side, so writes are visible
immediately under `STRONG` consistency.

## Regenerating the stubs

The generated stubs (`memlake_client/v1/memlake_pb2*.py`) are committed. Regenerate them from
the proto after a schema change (grpcio-tools bundles protoc — no system install needed):

```bash
uv run --project clients/python --extra dev clients/python/scripts/gen.sh
```
