"""Multi-node chaos & correctness suite.

Spin up N real serve nodes + M index loops against one MinIO, hammer them with concurrent
writes/deletes/queries through the smart (routing + failover) client while SIGKILLing and
restarting random nodes, then prove the system stayed correct:

  * no acked write is lost, no acked delete resurrects (seq-replay oracle);
  * every live node agrees on the visible set (cross-node consistency);
  * an acked write is immediately visible even across a node kill (INV-5);
  * every namespace's manifest stays well-formed (G-6): stats succeeds, cursor <= head.

The oracle replays every acked (seq, op) in WAL-sequence order — the engine's own total order
— so it needs no locks-around-time reasoning: it computes exactly what a fold would compute.
Writes that *raised* are ambiguous (at-least-once) and excluded from the exact assertion.

Only idempotent-under-retry ops are used (upsert-by-id, tombstone-by-id): the client's
UNAVAILABLE failover can duplicate a WAL entry, which is harmless for those (last-seq wins /
idempotent) but would double-apply a *relative* proof_count delta — so those are not exercised
here (they get a dedicated single-node test).

Run:
  uv run --project bench memlake-bench chaos                 # CI-sized gate (~60s, 3 nodes)
  CHAOS_DOCS=200000 CHAOS_SECS=600 uv run ... memlake-bench chaos    # soak
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import grpc

from memlake_client import MemlakeClient, memory

from .cluster import Cluster


@dataclass
class ChaosConfig:
    nodes: int = 3
    indexers: int = 2
    namespaces: int = 4
    docs: int = 5000            # total write/delete ops to attempt
    secs: int = 60             # wall-clock cap
    workers: int = 8
    seed: int = 1234
    dim: int = 16
    kill_every: float = 4.0    # seconds between serve-node kills (0 disables chaos)
    restart_delay: float = 1.5  # how long a killed node stays down
    index_interval: int = 1
    wait_index_frac: float = 0.10  # fraction of upserts that also fold inline
    delete_frac: float = 0.20      # fraction of ops that delete instead of upsert
    reupsert_frac: float = 0.10    # fraction of upserts that overwrite an existing id
    visibility_check_frac: float = 0.20  # fraction of acked upserts we immediately get()

    @classmethod
    def from_env(cls) -> "ChaosConfig":
        def _i(name, default):
            return int(os.environ.get(name, default))

        def _f(name, default):
            return float(os.environ.get(name, default))

        return cls(
            nodes=_i("CHAOS_NODES", 3),
            indexers=_i("CHAOS_INDEXERS", 2),
            namespaces=_i("CHAOS_NAMESPACES", 4),
            docs=_i("CHAOS_DOCS", 5000),
            secs=_i("CHAOS_SECS", 60),
            workers=_i("CHAOS_WORKERS", 8),
            seed=_i("CHAOS_SEED", 1234),
            kill_every=_f("CHAOS_KILL_EVERY", 4.0),
            delete_frac=_f("CHAOS_DELETE_FRAC", 0.20),
            reupsert_frac=_f("CHAOS_REUPSERT_FRAC", 0.10),
            wait_index_frac=_f("CHAOS_WAIT_INDEX_FRAC", 0.10),
        )


def _unit_axis(i: int, dim: int) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    v[(i + 1) % dim] = 0.05
    return v


class Oracle:
    """Thread-safe ledger of acked ops (for seq-replay) + ambiguous ids + violation log."""

    def __init__(self):
        self._lock = threading.Lock()
        self.events: dict[str, list] = defaultdict(list)     # ns -> [(seq, kind, id)]
        self.live: dict[str, set] = defaultdict(set)          # ns -> currently-live ids (hint)
        self.ambiguous: dict[str, set] = defaultdict(set)     # ns -> ids of raised ops
        self.acked_upserts = 0
        self.acked_deletes = 0
        self.errors = 0
        self.violations: list[str] = []

    def record_upsert(self, ns: str, seq: int, mid: bytes) -> None:
        with self._lock:
            self.events[ns].append((seq, "up", mid))
            self.live[ns].add(mid)
            self.acked_upserts += 1

    def record_delete(self, ns: str, seq: int, mid: bytes) -> None:
        with self._lock:
            self.events[ns].append((seq, "del", mid))
            self.live[ns].discard(mid)
            self.acked_deletes += 1

    def mark_ambiguous(self, ns: str, mid: bytes) -> None:
        with self._lock:
            self.ambiguous[ns].add(mid)
            self.errors += 1

    def sample_live(self, ns: str, rng: random.Random) -> bytes | None:
        with self._lock:
            s = self.live[ns]
            if not s:
                return None
            return rng.choice(tuple(s))

    def violation(self, msg: str) -> None:
        with self._lock:
            self.violations.append(msg)

    def classify(self, ns: str, mid: bytes) -> str:
        """Diagnostic: how the oracle saw an id, to explain a phantom/lost result."""
        with self._lock:
            evs = sorted([e for e in self.events[ns] if e[2] == mid], key=lambda e: e[0])
        if not evs:
            return "never-acked"
        seqs = [(s, k) for s, k, _ in evs]
        return f"last={seqs[-1]} n={len(seqs)} all={seqs[:6]}"

    def expected_present(self, ns: str) -> set:
        """Replay acked events in WAL-seq order — the fold's own order — to the final live set."""
        with self._lock:
            evs = sorted(self.events[ns], key=lambda e: e[0])
            amb = set(self.ambiguous[ns])
        state: dict[bytes, bool] = {}
        for _seq, kind, mid in evs:
            state[mid] = (kind == "up")
        return {mid for mid, present in state.items() if present} - amb


def _mk_memory(mid: bytes, ns_idx: int, dim: int, tag: str):
    return memory(f"chaos {tag}", _unit_axis(ns_idx, dim), memory_type=1, id=mid)


def run(cfg: ChaosConfig | None = None) -> dict:
    cfg = cfg or ChaosConfig.from_env()
    # A run-unique namespace token so a run NEVER inherits a previous run's data (create is
    # idempotent — it does not clear). The seed still fully determines the workload RNG, so
    # runs stay reproducible; only the namespace identity is fresh. Cleaned up at the end.
    run_token = uuid.uuid4().hex[:8]
    ns_names = [f"chaos-{run_token}-{i}" for i in range(cfg.namespaces)]
    oracle = Oracle()

    print(f"[chaos] nodes={cfg.nodes} indexers={cfg.indexers} ns={cfg.namespaces} "
          f"docs={cfg.docs} secs={cfg.secs} workers={cfg.workers} kill_every={cfg.kill_every}")

    with Cluster(nodes=cfg.nodes, indexers=cfg.indexers, index_interval=cfg.index_interval,
                 index_namespaces=ns_names) as cl:
        client = MemlakeClient(cl.addrs)
        for ns in ns_names:
            client.create_namespace(ns)
        print(f"[chaos] cluster up on {cl.addrs}; namespaces created")

        import itertools
        stop = threading.Event()
        start = time.time()
        deadline = start + cfg.secs
        op_counter = itertools.count()

        def do_op(rng: random.Random) -> None:
            ns_idx = rng.randrange(cfg.namespaces)
            ns = ns_names[ns_idx]
            roll = rng.random()

            # DELETE an existing live id.
            if roll < cfg.delete_frac:
                mid = oracle.sample_live(ns, rng)
                if mid is None:
                    return
                try:
                    seq = client.delete(ns, [mid])
                    oracle.record_delete(ns, seq, mid)
                except grpc.RpcError:
                    oracle.mark_ambiguous(ns, mid)
                return

            # UPSERT: usually a fresh id, sometimes overwrite an existing one.
            reuse = rng.random() < cfg.reupsert_frac and oracle.live[ns]
            mid = oracle.sample_live(ns, rng) if reuse else uuid.uuid4().bytes
            if mid is None:
                mid = uuid.uuid4().bytes
            wait = rng.random() < cfg.wait_index_frac
            try:
                seq = client.write(ns, [_mk_memory(mid, ns_idx, cfg.dim, "v")], wait_for_index=wait)
                oracle.record_upsert(ns, seq, mid)
            except grpc.RpcError:
                oracle.mark_ambiguous(ns, mid)
                return

            # Visibility oracle: an acked write must be immediately gettable (INV-5), even if
            # its preferred node just died and the get failed over to a peer.
            if rng.random() < cfg.visibility_check_frac:
                try:
                    if mid not in client.exists(ns, [mid]):
                        oracle.violation(f"{ns}: acked write {mid.hex()[:8]} not visible via get")
                except grpc.RpcError:
                    pass  # all replicas momentarily unreachable — not a visibility bug

        def worker(wid: int) -> None:
            rng = random.Random(cfg.seed * 1_000 + wid)
            while not stop.is_set() and time.time() < deadline:
                if next(op_counter) >= cfg.docs:
                    break
                try:
                    do_op(rng)
                except Exception as e:  # never let a worker die silently
                    oracle.violation(f"worker {wid} crashed: {e!r}")

        def killer() -> None:
            rng = random.Random(cfg.seed + 999)
            while not stop.is_set():
                if stop.wait(cfg.kill_every):
                    break
                victim = cl.kill_random_serve(rng)
                if victim is not None:
                    print(f"[chaos] killed {victim.node_id}")
                    if stop.wait(cfg.restart_delay):
                        # restart before exiting so teardown is clean
                        victim.start(wait=False)
                        break
                    victim.start(wait=False)
                    print(f"[chaos] restarted {victim.node_id}")

        threads = []
        if cfg.kill_every > 0:
            kt = threading.Thread(target=killer, daemon=True)
            kt.start()
            threads.append(kt)

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = [pool.submit(worker, w) for w in range(cfg.workers)]
            for f in futures:
                f.result()
        stop.set()
        for t in threads:
            t.join(timeout=10)

        elapsed = time.time() - start
        print(f"[chaos] workload done in {elapsed:.1f}s — "
              f"upserts={oracle.acked_upserts} deletes={oracle.acked_deletes} "
              f"ambiguous={oracle.errors} violations={len(oracle.violations)}")

        # Quiesce: bring every node fully back up (port accepting) so the per-node cross-node
        # check never races a still-(re)starting node. This is a test-harness concern only —
        # during the workload the smart client's failover already handled the down nodes.
        cl.ensure_all_up()
        # Port-open is not serving-ready: probe each node with a real RPC until it answers, so
        # the cross-node assertion measures agreement, not startup timing.
        for n in cl.serve_nodes:
            probe = MemlakeClient(n.addr)
            for _ in range(40):
                try:
                    probe.list_namespaces()
                    break
                except grpc.RpcError:
                    time.sleep(0.25)
            probe.close()

        try:
            report = _assert_correct(cl, ns_names, oracle)
        except AssertionError:
            print(f"[chaos] FAILED — namespaces kept for inspection: {ns_names}")
            client.close()
            raise
        report.update(
            acked_upserts=oracle.acked_upserts,
            acked_deletes=oracle.acked_deletes,
            ambiguous=oracle.errors,
            elapsed_s=round(elapsed, 1),
            kills=sum(n.kills for n in cl.serve_nodes),
        )
        # Success: drop this run's namespaces so the bucket doesn't accumulate.
        for ns in ns_names:
            try:
                client.delete_namespace(ns)
            except Exception:
                pass
        client.close()

    print(f"[chaos] PASSED — {report}")
    return report


def _assert_correct(cl: Cluster, ns_names: list, oracle: Oracle) -> dict:
    """The correctness certificate. Raises AssertionError on any violation."""
    failures: list[str] = []

    if oracle.violations:
        failures.extend(oracle.violations[:20])

    # Per-namespace: seq-replay oracle vs. a full scan, and cross-node agreement.
    per_node_clients = {n.addr: MemlakeClient(n.addr) for n in cl.serve_nodes if n.alive}
    total_live = 0
    try:
        for ns in ns_names:
            expected = oracle.expected_present(ns)
            amb = set(oracle.ambiguous[ns])

            # Actual visible set from EACH live node — they must all agree (modulo ambiguous).
            # A node that just came back up may briefly refuse a connection; retry before
            # calling it a failure (this is liveness, not the correctness property under test).
            per_node_sets = {}
            for addr, c in per_node_clients.items():
                last_err = None
                for _attempt in range(5):
                    try:
                        per_node_sets[addr] = set(c.scan_all_ids(ns)) - amb
                        last_err = None
                        break
                    except grpc.RpcError as e:
                        last_err = e
                        time.sleep(0.5)
                if last_err is not None:
                    failures.append(f"{ns}: scan on {addr} failed after retries: {last_err.code()}")

            if not per_node_sets:
                failures.append(f"{ns}: no node could scan")
                continue

            # Cross-node consistency.
            baseline_addr, baseline = next(iter(per_node_sets.items()))
            for addr, ids in per_node_sets.items():
                if ids != baseline:
                    only_a = len(baseline - ids)
                    only_b = len(ids - baseline)
                    failures.append(
                        f"{ns}: nodes disagree ({baseline_addr} vs {addr}): "
                        f"{only_a} only-in-first, {only_b} only-in-second")

            actual = baseline
            total_live += len(actual)
            if os.environ.get("CHAOS_DEBUG"):
                recorded = {e[2] for e in oracle.events[ns]}
                max_seq = max((e[0] for e in oracle.events[ns]), default=0)
                st = per_node_clients[baseline_addr].stats(ns)
                print(f"[dbg] {ns}: recorded_ids={len(recorded)} max_recorded_seq={max_seq} "
                      f"wal_head={st.wal_head} cursor={st.wal_index_cursor} gen={st.generation} "
                      f"scanned={len(actual)} phantom={len(actual - expected)} lost={len(expected - actual)}")
                for m in list(actual - expected)[:2]:
                    got = per_node_clients[baseline_addr].get(ns, [m])
                    txt = got[0].memory.text if got else "<gone>"
                    print(f"[dbg]   phantom {m.hex()} text={txt!r}")
                # Dump the first few WAL entries: how many ops each, and their ids.
                wal = per_node_clients[baseline_addr].list_wal(ns, limit=8, include_ops=True)
                recorded_bytes = {e[2] for e in oracle.events[ns]}
                print(f"[dbg]   wal entries returned={len(wal.entries)} head={wal.wal_head}")
                for we in wal.entries[:8]:
                    op_ids = []
                    for od in we.ops:
                        kind = od.WhichOneof("kind")
                        if kind == "upsert":
                            oid = bytes(od.upsert.id)
                            known = "R" if oid in recorded_bytes else "?"
                            op_ids.append(f"up:{oid.hex()[:6]}{known}")
                        else:
                            op_ids.append(kind)
                    print(f"[dbg]     seq={we.seq} nops={len(we.ops)} {op_ids}")
            lost = expected - actual          # acked but missing → LOST WRITE (serious)
            phantom = actual - expected        # present but never acked / acked-deleted → RESURRECTION
            if lost:
                failures.append(f"{ns}: {len(lost)} LOST writes (e.g. {[m.hex()[:8] for m in list(lost)[:5]]})")
            if phantom:
                sample = list(phantom)[:5]
                detail = "; ".join(f"{m.hex()[:8]}: {oracle.classify(ns, m)}" for m in sample)
                failures.append(f"{ns}: {len(phantom)} PHANTOM/resurrected -- {detail}")

            # G-6: manifest well-formed and consistent.
            try:
                st = per_node_clients[baseline_addr].stats(ns)
                if st.wal_index_cursor > st.wal_head:
                    failures.append(f"{ns}: cursor {st.wal_index_cursor} > head {st.wal_head}")
            except grpc.RpcError as e:
                failures.append(f"{ns}: stats failed: {e.code()}")
    finally:
        for c in per_node_clients.values():
            c.close()

    if failures:
        raise AssertionError("CHAOS FAILED:\n  " + "\n  ".join(failures))

    return {"namespaces": len(ns_names), "live_ids": total_live, "nodes": len(cl.serve_nodes)}
