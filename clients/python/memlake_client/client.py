"""Ergonomic wrapper over the generated gRPC stubs.

Vectors go on the wire as raw little-endian float32 (`Vector.f32le`) — ~4x smaller than JSON
floats and zero-copy — so this module handles the pack/unpack. Everything else is a thin
pass-through to keep the memlake domain vocabulary (memory / memory_type / namespace / tags)
front and center.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
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


def _arm(a) -> "Arm":
    return Arm(present=a.present, rank=a.rank, score=a.score)


def _hits(pb_hits) -> list["Hit"]:
    return [
        Hit(
            id=h.id,
            memory_type=h.memory_type,
            dense=_arm(h.dense),
            text=_arm(h.text),
            graph=_arm(h.graph),
        )
        for h in pb_hits
    ]


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
class Arm:
    """One arm's raw signal for a hit. `present` is False if the arm did not surface it."""
    present: bool
    rank: int
    score: float


@dataclass
class Hit:
    """A retrieved candidate with the RAW per-arm signals. memlake does no fusion — combine
    these however you like (RRF over ranks, weighted sum of scores, re-ranking...)."""
    id: bytes
    memory_type: int
    dense: Arm   # vector / cosine
    text: Arm    # BM25
    graph: Arm   # graph activation

    @property
    def id_uuid(self) -> str:
        return str(uuid.UUID(bytes=self.id))


class MemlakeClient:
    """A blocking gRPC client. Reuse one instance across calls (it holds a channel)."""

    def __init__(self, target: str = "localhost:50051", *, channel: Optional[grpc.Channel] = None):
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = rpc.MemlakeStub(self._channel)
        self.last_roundtrips: int = 0  # object-storage roundtrips of the last query()

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
        *,
        vector: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
        memory_types: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
        tags_mode: int = ANY,
        vector_top_k: int = 0,
        text_top_k: int = 0,
        graph_top_k: int = 0,
        nprobe: int = 0,
        consistency: int = STRONG,
    ) -> list[Hit]:
        """Run one query across `memory_types` (or all if None) with all three arms —
        dense vector, BM25 full-text, and graph — in a single call. `vector` drives the
        dense + graph arms; `text` drives full-text. `*_top_k` bound each arm's candidate
        depth (0 = server default).

        Returns a flat list of Hit, each carrying the RAW per-arm signals (`hit.dense`,
        `hit.text`, `hit.graph`, each an `Arm(present, rank, score)`) plus `hit.memory_type`.
        memlake does NOT fuse — group by `memory_type` and apply your own RRF / weighting.
        The server-side roundtrips of the last call are available as `client.last_roundtrips`.
        """
        req = pb.QueryRequest(
            namespace=namespace,
            memory_types=list(memory_types or []),
            vector=_pack(vector),
            text=text or "",
            vector_top_k=vector_top_k,
            text_top_k=text_top_k,
            graph_top_k=graph_top_k,
            nprobe=nprobe,
            consistency=consistency,
        )
        if tags:
            req.tags.CopyFrom(pb.TagFilter(tags=list(tags), mode=tags_mode))
        resp = self._stub.Query(req)
        self.last_roundtrips = resp.load_roundtrips
        return _hits(resp.hits)
