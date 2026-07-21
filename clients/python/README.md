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

    # ONE call, across all memory_types (or a subset), always running the three arms:
    # dense vector + BM25 full-text + graph. Returns a flat list of hits.
    hits = c.query(
        "my-bank",
        vector=query_embedding, text="retrieval",
        tags=["prod"], tags_mode=ANY_STRICT,
    )

    # memlake does NO fusion. Each hit carries the RAW per-arm signals so you run your own
    # RRF / weighting. Group by memory_type first — types are independent.
    for h in hits:
        print(h.memory_type, h.id_uuid,
              (h.dense.present, h.dense.rank, h.dense.score),   # cosine
              (h.text.present,  h.text.rank,  h.text.score),    # BM25
              (h.graph.present, h.graph.rank, h.graph.score))   # graph activation

    print("server-side roundtrips:", c.last_roundtrips)
```

Vectors travel as raw little-endian float32 (`Vector.f32le`) — the client packs/unpacks for
you. `query` reuses a cached consistent snapshot server-side; under `STRONG` consistency
(default) writes are visible immediately, `EVENTUAL` serves from the cache with 0 roundtrips.

## Regenerating the stubs

The generated stubs (`memlake_client/v1/memlake_pb2*.py`) are committed. Regenerate them from
the proto after a schema change (grpcio-tools bundles protoc — no system install needed):

```bash
uv run --project clients/python --extra dev clients/python/scripts/gen.sh
```
