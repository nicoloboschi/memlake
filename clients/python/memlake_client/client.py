"""Ergonomic wrapper over the generated gRPC stubs.

Vectors go on the wire as raw little-endian float32 (`Vector.f32le`) — ~4x smaller than JSON
floats and zero-copy — so this module handles the pack/unpack. Everything else is a thin
pass-through to keep the memlake domain vocabulary (memory / memory_type / namespace / tags)
front and center.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import grpc

from .v1 import memlake_pb2 as pb
from .v1 import memlake_pb2_grpc as rpc

# Tag-match modes, re-exported so callers don't import the raw protobuf enum.
ANY = pb.ANY
ALL = pb.ALL
ANY_STRICT = pb.ANY_STRICT
ALL_STRICT = pb.ALL_STRICT
EXACT = pb.EXACT

# Consistency levels.
STRONG = pb.STRONG
EVENTUAL = pb.EVENTUAL


def _pack(values: Optional[Sequence[float]]) -> Optional[pb.Vector]:
    if not values:
        return None
    return pb.Vector(f32le=struct.pack(f"<{len(values)}f", *values))


def _unpack(v: pb.Vector) -> list[float]:
    return list(struct.unpack(f"<{len(v.f32le) // 4}f", v.f32le))


def memory(
    text: str,
    vector: Optional[Sequence[float]] = None,
    *,
    memory_type: int = 1,
    key: str = "",
    id: bytes = b"",
    tags: Optional[Sequence[str]] = None,
    proof_count: int = 0,
    entity_ids: Optional[Sequence[int]] = None,
) -> pb.Memory:
    """Build a Memory. Pass `key` (and leave `id` empty) to let the server derive a stable
    16-byte id from the key; or pass a 16-byte `id` directly."""
    return pb.Memory(
        id=id,
        key=key,
        text=text,
        vector=_pack(vector),
        memory_type=memory_type,
        tags=list(tags or []),
        proof_count=proof_count,
        entity_ids=list(entity_ids or []),
    )


@dataclass
class Hit:
    id: bytes
    score: float
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def id_uuid(self) -> str:
        return str(uuid.UUID(bytes=self.id))


class MemlakeClient:
    """A blocking gRPC client. Reuse one instance across calls (it holds a channel)."""

    def __init__(self, target: str = "localhost:50051", *, channel: Optional[grpc.Channel] = None):
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = rpc.MemlakeStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "MemlakeClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- namespace -----------------------------------------------------------

    def create_namespace(self, namespace: str) -> None:
        self._stub.CreateNamespace(pb.CreateNamespaceRequest(namespace=namespace))

    # -- write ---------------------------------------------------------------

    def write(self, namespace: str, memories: Iterable[pb.Memory]) -> int:
        """Upsert a batch of memories atomically. Returns the claimed WAL sequence; the call
        returns only once the batch is durably persisted to object storage."""
        ops = [pb.Op(upsert=m) for m in memories]
        resp = self._stub.Write(pb.WriteRequest(namespace=namespace, ops=ops))
        return resp.seq

    def write_ops(self, namespace: str, ops: Iterable[pb.Op]) -> pb.WriteResponse:
        """Lower-level write for mixed op batches (tombstones, patches, guards)."""
        return self._stub.Write(pb.WriteRequest(namespace=namespace, ops=list(ops)))

    def tombstone(self, id16: bytes) -> pb.Op:
        return pb.Op(tombstone=id16)

    def patch(self, id16: bytes, proof_count_delta: int) -> pb.Op:
        return pb.Op(patch=pb.Patch(id=id16, proof_count_delta=proof_count_delta))

    # -- query ---------------------------------------------------------------

    def query(
        self,
        namespace: str,
        memory_type: int,
        *,
        vector: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        tags_mode: int = ANY,
        top_k: int = 10,
        consistency: int = STRONG,
        nprobe: int = 0,
        vector_weight: float = 0.0,
        fts_weight: float = 0.0,
        graph_weight: float = 1.0,
        arm_depth: int = 0,
    ) -> list[Hit]:
        """Query one memory_type. Omit `vector`/`text` to drop that arm. Set `graph_weight=0`
        to drop the graph arm. Zeroed config fields fall back to server defaults."""
        req = pb.QueryRequest(
            namespace=namespace,
            memory_type=memory_type,
            vector=_pack(vector),
            text=text or "",
            top_k=top_k,
            consistency=consistency,
            config=pb.QueryConfig(
                nprobe=nprobe,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
                graph_weight=graph_weight,
                arm_depth=arm_depth,
            ),
        )
        hits, _ = self.query_metered(
            namespace, memory_type,
            vector=vector, text=text, tags=tags, tags_mode=tags_mode,
            top_k=top_k, consistency=consistency, nprobe=nprobe,
            vector_weight=vector_weight, fts_weight=fts_weight,
            graph_weight=graph_weight, arm_depth=arm_depth,
        )
        return hits

    def query_metered(
        self,
        namespace: str,
        memory_type: int,
        *,
        vector: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        tags_mode: int = ANY,
        top_k: int = 10,
        consistency: int = STRONG,
        nprobe: int = 0,
        vector_weight: float = 0.0,
        fts_weight: float = 0.0,
        graph_weight: float = 1.0,
        arm_depth: int = 0,
    ) -> tuple[list[Hit], int]:
        """Like `query`, but also returns the object-storage roundtrips the query consumed
        server-side (0 == fully served from the server's cache)."""
        req = pb.QueryRequest(
            namespace=namespace,
            memory_type=memory_type,
            vector=_pack(vector),
            text=text or "",
            top_k=top_k,
            consistency=consistency,
            config=pb.QueryConfig(
                nprobe=nprobe,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
                graph_weight=graph_weight,
                arm_depth=arm_depth,
            ),
        )
        if tags:
            req.tags.CopyFrom(pb.TagFilter(tags=list(tags), mode=tags_mode))
        resp = self._stub.Query(req)
        hits = [
            Hit(
                id=h.id,
                score=h.score,
                contributions={c.arm: c.score for c in h.contributions},
            )
            for h in resp.hits
        ]
        return hits, resp.load_roundtrips
