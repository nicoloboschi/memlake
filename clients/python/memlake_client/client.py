"""Ergonomic wrapper over the generated gRPC stubs.

Vectors go on the wire as raw little-endian float32 (`Vector.f32le`) — ~4x smaller than JSON
floats and zero-copy — so this module handles the pack/unpack. Everything else is a thin
pass-through to keep the memlake domain vocabulary (memory / memory_type / namespace / tags)
front and center.
"""

from __future__ import annotations

import hashlib
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


def _pack(values: Optional[Sequence[float]]) -> Optional[pb.Vector]:
    if not values:
        return None
    return pb.Vector(f32le=struct.pack(f"<{len(values)}f", *values))


def _unpack(v: pb.Vector) -> list[float]:
    return list(struct.unpack(f"<{len(v.f32le) // 4}f", v.f32le))


def _arm(a) -> "Arm":
    return Arm(present=a.present, rank=a.rank, score=a.score)


def _edge(e):
    return SemanticEdge(target=e.target, weight=e.weight)


def _payload(p) -> "Payload":
    return Payload(
        text=p.text,
        tags=list(p.tags),
        proof_count=p.proof_count,
        entity_ids=list(p.entity_ids),
        metadata=dict(p.metadata),
        timestamps=p.timestamps,
        semantic_out=[_edge(e) for e in p.semantic_out],
    )


def _hits(pb_hits) -> list["Hit"]:
    return [
        Hit(
            id=h.id,
            memory_type=h.memory_type,
            dense=_arm(h.dense),
            text=_arm(h.text),
            graph=_arm(h.graph),
            temporal=_arm(h.temporal),
            memory=_payload(h.memory) if h.HasField("memory") else None,
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
    entity_ids: Optional[Sequence[bytes]] = None,
    metadata: Optional[dict[str, str]] = None,
    index_text: Optional[str] = None,
    updated_at: Optional[int] = None,
    occurred_start: Optional[int] = None,
    occurred_end: Optional[int] = None,
    mentioned_at: Optional[int] = None,
    event_date: Optional[int] = None,
) -> pb.Memory:
    """Build a Memory. Pass `key` (and leave `id` empty) to let the server derive a stable
    16-byte id from the key; or pass a 16-byte `id` directly. `entity_ids` are 16-byte ids
    (e.g. `uuid.UUID(...).bytes`). `metadata` is opaque str->str the server stores and returns
    verbatim (never indexed). The timestamp fields (epoch ints) drive the temporal arm; the
    effective time is COALESCE(occurred_start, mentioned_at, occurred_end). `index_text`
    replaces `text` for full-text indexing only — enrich what BM25 matches without changing
    what a hit returns. `updated_at` is write time (the server defaults it to now), and is
    what Query/Scan's `updated_from`/`updated_to` window ranges over."""
    ts = pb.Timestamps()
    if occurred_start is not None:
        ts.occurred_start = occurred_start
    if occurred_end is not None:
        ts.occurred_end = occurred_end
    if mentioned_at is not None:
        ts.mentioned_at = mentioned_at
    if event_date is not None:
        ts.event_date = event_date
    if updated_at is not None:
        ts.updated_at = updated_at
    return pb.Memory(
        id=id,
        key=key,
        text=text,
        vector=_pack(vector),
        memory_type=memory_type,
        tags=list(tags or []),
        proof_count=proof_count,
        entity_ids=list(entity_ids or []),
        metadata=dict(metadata or {}),
        timestamps=ts,
        index_text=index_text or "",
    )


@dataclass
class Arm:
    """One arm's raw signal for a hit. `present` is False if the arm did not surface it."""
    present: bool
    rank: int
    score: float


@dataclass
class SemanticEdge:
    """A derived kNN edge: the target memory and its similarity weight."""

    target: bytes
    weight: float

    @property
    def target_uuid(self) -> str:
        return str(uuid.UUID(bytes=self.target))


@dataclass
class Payload:
    """The stored memory returned inline with a hit (embedding vector omitted). `metadata` is
    the opaque str->str the caller wrote. `timestamps` is the raw protobuf Timestamps."""
    text: str
    tags: list
    proof_count: int
    entity_ids: list
    metadata: dict
    timestamps: object
    # Present only when the read asked for `include_edges`; empty otherwise.
    semantic_out: list = field(default_factory=list)


@dataclass
class Hit:
    """A retrieved candidate: the RAW per-arm signals (memlake does no fusion — combine them
    yourself: RRF over ranks, weighted scores, re-ranking...) plus the materialized `memory`,
    returned inline so recall needs no second round trip to hydrate."""
    id: bytes
    memory_type: int
    dense: Arm      # vector / cosine
    text: Arm       # BM25
    graph: Arm      # graph activation
    temporal: Arm   # temporal (entry points + one-hop spread)
    memory: object = None   # Payload | None

    @property
    def id_uuid(self) -> str:
        return str(uuid.UUID(bytes=self.id))


def _rendezvous_key(namespace: str, node: str) -> int:
    """Highest-random-weight (rendezvous) score for (namespace, node). A stable, process-
    independent hash so every client agrees on the same preferred node for a namespace, and
    adding/removing a node reshuffles only its share of namespaces (unlike modulo hashing)."""
    h = hashlib.blake2b(f"{namespace}\x00{node}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


class MemlakeClient:
    """A blocking gRPC client over one or more server nodes.

    Pass a single address (`"localhost:50051"`) or a list (`["n1:50051", "n2:50051", ...]`).
    With several nodes the client rendezvous-hashes each namespace to a *preferred* node for
    both reads and writes: reads get cache affinity (a namespace's hot data stays warm on one
    node's cache), writes get commit affinity (one node batches a namespace's writes, so
    concurrent committers don't churn the WAL-sequence CAS). Affinity is only a hint — every
    node can serve every request (all coordination is in object storage), so on a node failure
    the client transparently fails over to the next-preferred node. Correctness never depends
    on routing; it only makes the common case cheaper.
    """

    def __init__(
        self,
        target: "str | Sequence[str]" = "localhost:50051",
        *,
        channel: Optional[grpc.Channel] = None,
    ):
        if channel is not None:
            # Legacy/testing single injected channel: one synthetic node.
            self._nodes: list[str] = ["<injected>"]
            self._channels = {"<injected>": channel}
        else:
            nodes = [target] if isinstance(target, str) else list(target)
            if not nodes:
                raise ValueError("MemlakeClient needs at least one node address")
            # De-dup while preserving order.
            self._nodes = list(dict.fromkeys(nodes))
            self._channels = {n: grpc.insecure_channel(n) for n in self._nodes}
        self._stubs = {n: rpc.MemlakeStub(ch) for n, ch in self._channels.items()}
        self.last_roundtrips: int = 0  # object-storage roundtrips of the last query()
        self.last_node: str = ""       # which node served the last call (for debugging/tests)

    @property
    def nodes(self) -> list[str]:
        return list(self._nodes)

    def _ordered_nodes(self, namespace: str) -> list[str]:
        """Nodes for `namespace`, preferred first (descending rendezvous score). The tail is
        the failover order."""
        if len(self._nodes) == 1:
            return self._nodes
        return sorted(self._nodes, key=lambda n: _rendezvous_key(namespace, n), reverse=True)

    def preferred_node(self, namespace: str) -> str:
        """The node this client routes `namespace` to first. Exposed so a test/LB can assert
        the same mapping the client uses."""
        return self._ordered_nodes(namespace)[0]

    def _call(self, method: str, request):
        """Invoke `method` on the namespace's preferred node, failing over to the next node on
        an UNAVAILABLE (a node that is down or unreachable — the RPC provably did not execute,
        so retrying elsewhere cannot double-apply a write). Any other error is the server's
        considered answer and is raised as-is."""
        namespace = getattr(request, "namespace", "") or ""
        ordered = self._ordered_nodes(namespace)
        last_err: Optional[grpc.RpcError] = None
        for node in ordered:
            try:
                result = getattr(self._stubs[node], method)(request)
                self.last_node = node
                return result
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    last_err = e
                    continue  # node down/unreachable — try the next-preferred
                raise
        # Every node was unreachable.
        raise last_err  # type: ignore[misc]

    def close(self) -> None:
        for ch in self._channels.values():
            ch.close()

    def __enter__(self) -> "MemlakeClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- namespace -----------------------------------------------------------

    def create_namespace(self, namespace: str, *, indexed_metadata_keys: Optional[Sequence[str]] = None) -> None:
        """Create the namespace if absent. `indexed_metadata_keys` declares the metadata keys
        MetadataStats can count by — fixed at creation, ignored if the namespace exists."""
        self._call(
            "CreateNamespace",
            pb.CreateNamespaceRequest(
                namespace=namespace,
                indexed_metadata_keys=list(indexed_metadata_keys or []),
            ),
        )

    def delete_namespace(self, namespace: str) -> int:
        """Drop a whole namespace — delete every object under its prefix (manifest, WAL, all
        generations). Irreversible; returns the number of objects removed. Do not call while
        the namespace is being written."""
        return self._call("DeleteNamespace", 
            pb.DeleteNamespaceRequest(namespace=namespace)
        ).objects_deleted

    # -- write ---------------------------------------------------------------

    def write(
        self, namespace: str, memories: Iterable[pb.Memory], *, wait_for_index: bool = False
    ) -> int:
        """Upsert a batch of memories atomically. Returns the claimed WAL sequence; the call
        returns only once the batch is durably persisted to object storage.

        The write is immediately queryable without indexing (reads merge the WAL tail). Pass
        `wait_for_index=True` to wait until the background indexer has folded this write into a segment before
        returning — a heavier, synchronous call; use it after a bulk load or when you need the
        write in the generation before proceeding (e.g. tests, benchmarks)."""
        ops = [pb.Op(upsert=m) for m in memories]
        resp = self._call("Write", 
            pb.WriteRequest(namespace=namespace, ops=ops, wait_for_index=wait_for_index)
        )
        return resp.seq

    def write_ops(
        self, namespace: str, ops: Iterable[pb.Op], *, wait_for_index: bool = False
    ) -> pb.WriteResponse:
        """Lower-level write for mixed op batches (tombstones, patches, guards). See `write`
        for `wait_for_index`."""
        return self._call("Write", 
            pb.WriteRequest(namespace=namespace, ops=list(ops), wait_for_index=wait_for_index)
        )

    def delete(self, namespace: str, ids: Iterable[bytes]) -> int:
        """Delete memories by 16-byte id (tombstone). Returns the claimed WAL sequence. The
        tombstone hides the memory from reads immediately and removes it from the index at the
        next indexer run. One-way — there is no revert."""
        ops = [pb.Op(tombstone=i) for i in ids]
        return self._call("Write", pb.WriteRequest(namespace=namespace, ops=ops)).seq

    def _predicate(self, metadata_equals, tags, tags_mode, memory_types) -> pb.Predicate:
        p = pb.Predicate(
            memory_types=list(memory_types or []),
            metadata_equals=dict(metadata_equals or {}),
        )
        if tags:
            p.tags.CopyFrom(pb.TagFilter(tags=list(tags), mode=tags_mode))
        return p

    def delete_by_predicate(
        self,
        namespace: str,
        *,
        metadata_equals: Optional[dict[str, str]] = None,
        tags: Optional[Sequence[str]] = None,
        tags_mode: int = ANY,
        memory_types: Optional[Sequence[int]] = None,
        delete_all: bool = False,
        eager: bool = False,
    ) -> int:
        """Delete every memory matching the predicate (metadata AND tags — e.g.
        `{"document_id": "d-42", "chunk_id": "3"}`). An empty predicate is rejected unless
        `delete_all`.

        By default this writes one atomic, race-closed TombstoneWhere WAL op, materialized at
        the next fold — the right path for document re-ingest; it returns 0 (nothing scanned).
        Pass `eager=True` for an immediate O(corpus) scan that returns the exact count. To
        replace a document's facts atomically, prefer `write_ops([tombstone_where(...), *upserts])`
        so the delete and the new facts share one sequence."""
        req = pb.DeleteByPredicateRequest(
            namespace=namespace,
            memory_types=list(memory_types or []),
            metadata_equals=dict(metadata_equals or {}),
            delete_all=delete_all,
            eager=eager,
        )
        if tags:
            req.tags.CopyFrom(pb.TagFilter(tags=list(tags), mode=tags_mode))
        return self._call("DeleteByPredicate", req).deleted

    def tombstone_where(
        self,
        *,
        metadata_equals: Optional[dict[str, str]] = None,
        tags: Optional[Sequence[str]] = None,
        tags_mode: int = ANY,
        memory_types: Optional[Sequence[int]] = None,
    ) -> pb.Op:
        """A predicate-delete op for a `write_ops` batch. Put it in the SAME batch as the new
        upserts to replace a document's facts atomically — the new upserts share the entry's
        sequence, so the delete (which only removes older writes) spares them."""
        return pb.Op(tombstone_where=self._predicate(metadata_equals, tags, tags_mode, memory_types))

    def upsert(self, m: pb.Memory) -> pb.Op:
        """Wrap a `memory(...)` as an op, for mixing with tombstone_where/patch in a
        `write_ops` batch (e.g. atomic document re-ingest)."""
        return pb.Op(upsert=m)

    def tombstone(self, id16: bytes) -> pb.Op:
        return pb.Op(tombstone=id16)

    def patch(
        self,
        id16: bytes,
        *,
        proof_count_delta: int = 0,
        text: Optional[str] = None,
        vector: Optional[Sequence[float]] = None,
        tags: Optional[Sequence[str]] = None,
        occurred_start: Optional[int] = None,
        occurred_end: Optional[int] = None,
        mentioned_at: Optional[int] = None,
        event_date: Optional[int] = None,
        metadata: Optional[dict[str, str]] = None,
        replace_timestamps: bool = False,
    ) -> pb.Op:
        """Build a partial-update op: set only the fields you pass, leave the rest.
        `proof_count_delta` is relative; the rest are absolute sets; `metadata` merges (upserts
        its keys). Timestamps merge too: passing `occurred_start` alone leaves the other three
        as they were. Pass `replace_timestamps=True` to overwrite the whole Timestamps instead,
        clearing any you did not pass — the only way to null one. A patch always stamps
        `updated_at` with the server's write time, whichever mode you use. Pair with `update()`
        or include in a `write_ops` batch."""
        p = pb.Patch(id=id16, proof_count_delta=proof_count_delta)
        if text is not None:
            p.text = text
        if vector is not None:
            p.vector.CopyFrom(pb.Vector(f32le=struct.pack(f"<{len(vector)}f", *vector)))
        if tags is not None:
            p.tags.CopyFrom(pb.TagList(tags=list(tags)))
        if any(t is not None for t in (occurred_start, occurred_end, mentioned_at, event_date)):
            ts = pb.Timestamps()
            if occurred_start is not None:
                ts.occurred_start = occurred_start
            if occurred_end is not None:
                ts.occurred_end = occurred_end
            if mentioned_at is not None:
                ts.mentioned_at = mentioned_at
            if event_date is not None:
                ts.event_date = event_date
            p.timestamps.CopyFrom(ts)
            p.replace_timestamps = replace_timestamps
        if metadata:
            p.metadata.update(metadata)
        return pb.Op(patch=p)

    def update(self, namespace: str, id16: bytes, **fields) -> int:
        """Partial update of one memory (see `patch` for the fields). Returns the WAL sequence.
        Visible immediately for a still-un-indexed memory (tail); for an already-indexed one it
        takes retrieval effect at the next indexer run (the embedding/text are re-indexed then)."""
        return self._call("Write", 
            pb.WriteRequest(namespace=namespace, ops=[self.patch(id16, **fields)])
        ).seq

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
        graph_seed_min_similarity: Optional[float] = None,
        temporal_from: Optional[int] = None,
        temporal_to: Optional[int] = None,
        updated_from: Optional[int] = None,
        updated_to: Optional[int] = None,
    ) -> list[Hit]:
        """Run one query across `memory_types` (or all if None) with the retrieval arms in a
        single call. `vector` drives the dense + graph arms; `text` drives full-text. Passing
        both `temporal_from` and `temporal_to` (epoch ints) additionally runs the temporal
        arm: entry points whose effective time falls in the window, spread one hop and scored
        by proximity to the window centre (requires `vector`). `*_top_k` bound each arm's
        candidate depth (0 = server default).

        Returns a flat list of Hit, each carrying the RAW per-arm signals (`hit.dense`,
        `hit.text`, `hit.graph`, `hit.temporal`, each an `Arm(present, rank, score)`) plus
        `hit.memory_type`. memlake does NOT fuse — group by `memory_type` and apply your own
        RRF / weighting. Last call's roundtrips are on `client.last_roundtrips`.
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
        )
        if tags:
            req.tags.CopyFrom(pb.TagFilter(tags=list(tags), mode=tags_mode))
        if graph_seed_min_similarity is not None:
            req.graph_seed_min_similarity = graph_seed_min_similarity
        if temporal_from is not None:
            req.temporal_from = temporal_from
        if temporal_to is not None:
            req.temporal_to = temporal_to
        if updated_from is not None:
            req.updated_from = updated_from
        if updated_to is not None:
            req.updated_to = updated_to
        resp = self._call("Query", req)
        self.last_roundtrips = resp.load_roundtrips
        return _hits(resp.hits)

    # -- admin / introspection ----------------------------------------------
    # Thin wrappers over the operator RPCs. They route like everything else (rendezvous +
    # failover). The chaos/correctness suite leans on these to assert acked-write visibility
    # and manifest/generation state without reaching into the raw stubs.

    def stats(self, namespace: str) -> pb.StatsResponse:
        """Index state for a namespace: generation, wal_head, wal_index_cursor, live
        doc_count, through_seq, and per-memory_type counts. One manifest read + one WAL LIST;
        does not fetch cluster data."""
        return self._call("Stats", pb.StatsRequest(namespace=namespace))

    def get(
        self,
        namespace: str,
        ids: Iterable[bytes],
        *,
        include_vector: bool = False,
        include_edges: bool = False,
    ) -> list:
        """Fetch memories by 16-byte id, bypassing ranking. Returns the resolved
        `StoredMemoryRecord`s in request order; a missing or tombstoned id is simply absent (so
        `len(result) < len(ids)` means some ids are gone). This is the visibility oracle: after
        an acked write, a `get` of its ids must return them."""
        resp = self._call(
            "Get",
            pb.GetRequest(
                namespace=namespace,
                ids=list(ids),
                include_vector=include_vector,
                include_edges=include_edges,
            ),
        )
        return list(resp.memories)

    def exists(self, namespace: str, ids: Iterable[bytes]) -> set:
        """The subset of `ids` currently visible (present and not tombstoned)."""
        return {m.id for m in self.get(namespace, ids)}

    def scan(
        self,
        namespace: str,
        *,
        memory_types: Optional[Sequence[int]] = None,
        page_token: str = "",
        limit: int = 0,
        include_vector: bool = False,
        include_edges: bool = False,
        metadata_equals: Optional[dict[str, str]] = None,
        tags: Optional[Sequence[str]] = None,
        tags_mode: int = ANY,
        skip: int = 0,
        updated_from: Optional[int] = None,
        updated_to: Optional[int] = None,
    ) -> pb.ScanResponse:
        """One page of a full scan in cluster order. Follow `next_page_token` until empty. A
        scan is eventually-complete browsing, not a consistent iterator (writes mid-walk can
        shift later pages)."""
        req = pb.ScanRequest(
            namespace=namespace,
            memory_types=list(memory_types or []),
            page_token=page_token,
            limit=limit,
            include_vector=include_vector,
            include_edges=include_edges,
            metadata_equals=dict(metadata_equals or {}),
            skip=skip,
        )
        if tags:
            req.tags.CopyFrom(pb.TagFilter(tags=list(tags), mode=tags_mode))
        if updated_from is not None:
            req.updated_from = updated_from
        if updated_to is not None:
            req.updated_to = updated_to
        return self._call("Scan", req)

    def scan_all_ids(self, namespace: str, *, memory_types: Optional[Sequence[int]] = None) -> list:
        """Every visible id in a namespace, walking all scan pages. O(corpus) — for tests and
        audits, never the hot path."""
        ids, token = [], ""
        while True:
            page = self.scan(namespace, memory_types=memory_types, page_token=token)
            ids.extend(m.id for m in page.memories)
            token = page.next_page_token
            if not token:
                return ids

    def list_wal(self, namespace: str, *, start_seq: int = 0, limit: int = 0,
                 include_ops: bool = False) -> pb.ListWalResponse:
        """Operator view of the WAL: committed sequences, which are folded, optionally the ops.
        A window on the live log (folded+GC'd entries drop off), not full history."""
        return self._call(
            "ListWal",
            pb.ListWalRequest(namespace=namespace, start_seq=start_seq, limit=limit,
                              include_ops=include_ops),
        )

    def entity_stats(
        self,
        namespace: str,
        *,
        memory_types: Optional[Sequence[int]] = None,
        entity_ids: Optional[Sequence[bytes]] = None,
    ) -> dict:
        """Live memory count per entity, as `{entity_id_bytes: count}`.

        Reads the entity posting index, so cost scales with the number of entities rather
        than the corpus. Entities with no live memories are absent from the result — an id
        you asked about and do not get back is an orphan. Counts reflect the indexed
        generation plus the un-indexed WAL tail."""
        resp = self._call(
            "EntityStats",
            pb.EntityStatsRequest(
                namespace=namespace,
                memory_types=list(memory_types or []),
                entity_ids=list(entity_ids or []),
            ),
        )
        return {e.entity_id: e.memory_count for e in resp.entities}

    def metadata_stats(
        self,
        namespace: str,
        key: str,
        *,
        memory_types: Optional[Sequence[int]] = None,
    ) -> dict:
        """Live memory count per distinct value of one declared metadata `key`, as
        `{value: count}`.

        The primitive behind "count memories grouped by document_id / consolidated": read
        from the per-segment tally the fold builds, so it is a metadata read, not a corpus
        scan. `key` must be one of the namespace's `indexed_metadata_keys`; an undeclared key
        returns an empty map. Counts reflect the indexed generation plus the WAL tail."""
        resp = self._call(
            "MetadataStats",
            pb.MetadataStatsRequest(
                namespace=namespace,
                key=key,
                memory_types=list(memory_types or []),
            ),
        )
        return {v.value: v.count for v in resp.values}

    def link_stats(
        self,
        namespace: str,
        *,
        memory_types: Optional[Sequence[int]] = None,
    ) -> dict:
        """Live edge totals for a namespace, as `{"semantic": n, "causal": m}`.

        The primitive behind the bank stats page's link count: read from the per-segment tally
        the fold builds plus the WAL tail, so it is a metadata read, not a corpus scan.
        `semantic` is the derived kNN links; `causal` the intrinsic causal links."""
        resp = self._call(
            "LinkStats",
            pb.LinkStatsRequest(
                namespace=namespace,
                memory_types=list(memory_types or []),
            ),
        )
        return {"semantic": resp.semantic_edge_count, "causal": resp.causal_edge_count}

    def list_namespaces(self) -> list:
        """Every namespace in the bucket (one LIST). Not routed to a preferred node — any node
        answers identically."""
        return list(self._call("ListNamespaces", pb.ListNamespacesRequest()).namespaces)
