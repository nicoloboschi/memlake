# Multi-node chaos & correctness suite

Status notes for the horizontal-scale-out work. This is the running log for the chaos harness
and the routing/lease/identity changes that surround it. Read this first tomorrow.

## What shipped (incremental commits on `main`)

1. **Smart client — rendezvous routing + admin RPCs** (`clients/python/memlake_client/client.py`)
   - `MemlakeClient([addr, addr, ...])` rendezvous-hashes each namespace to a preferred node
     for reads (cache affinity) and writes (commit affinity). Fails over to the next-preferred
     node on `UNAVAILABLE` (RPC provably didn't execute → safe, no double-apply). Single addr
     still works. `last_node` / `preferred_node(ns)` expose the mapping.
   - Admin wrappers used by the suite: `stats`, `get`/`exists`, `scan`/`scan_all_ids`,
     `list_namespaces`. `get` is the visibility oracle.

2. **Soft fold lease** (`mlake-core::manifest::index_lease_path`, `Namespace::{acquire,release}_index_lease`,
   wired in `run_indexer`)
   - Best-effort `{ns}/index-lease.json` (holder + expiry). A peer with a *live* lease makes
     other indexers skip that namespace; free/expired/self-owned → proceed. **Fails open**:
     any storage/parse ambiguity → fold anyway (a duplicate fold is safe via nonce'd prefixes;
     wrongly skipping is not). Holder-guarded release. TTL 60s. Unit tests: `mlake-wal/tests/lease.rs`.
   - `wait_for_index` inline folds do **not** take the lease — a waiting caller must make
     progress; `index_until` re-checks the manifest each pass, so a peer fold that advances the
     cursor still satisfies it.

3. **Node identity** (`--node-id` / `MEMLAKE_NODE_ID`, else `{host}-{pid}`)
   - Stamped on every log line via a root `node` span (`instrument`). Lease holder is
     `{node_id}#{pid}` so two processes sharing a node-id still hold distinct leases.

## Invariants the chaos suite asserts

- **INV-5 acked-write visibility**: after `write` returns (durable), a `get` of its ids from
  *any* node returns them — before and across indexing, and after node kills.
- **G-6 manifest well-formedness**: at rest, `stats` on every namespace succeeds, the manifest
  parses (format version guard), `wal_index_cursor <= wal_head`, generation monotonic per ns.
- **No lost writes**: every acked upsert id is eventually visible (via `get`/`scan`).
- **No resurrection**: a tombstoned id, once acked deleted, stays gone.
- **Concurrent folds are safe**: with N indexers, queries never fault on a swept generation
  file (readers hold immutable, GC grace-windowed files); doc counts converge.
- **Routing is only a hint**: killing the preferred node for a namespace does not lose data or
  block reads/writes (failover).

## Harness shape (Python orchestrator — `bench/src/memlake_bench/chaos.py`)

- Spawns `NODES` real `mlake-server serve` processes (distinct `--node-id`, ports) + one or
  more `index` loops, all against the one local MinIO.
- Worker pool drives concurrent `write` (some `wait_for_index`), `query`, `get`, `delete`,
  `delete_by_predicate` via the smart client (multi-endpoint).
- Chaos: periodically SIGKILL a random serve node and restart it; occasionally kill an indexer.
- Oracle: an in-harness model of what *should* be visible (acked upserts minus acked deletes);
  after quiesce, assert the model matches `scan_all_ids` / `get` from every node.
- Scale knobs (env): `CHAOS_NODES` (default 3), `CHAOS_DOCS`, `CHAOS_SECS`, `CHAOS_WORKERS`,
  `CHAOS_SEED`. CI gate runs small/bounded; soak runs large.

## Open questions / blockers (fix or flag)

- _(none yet — updated as the harness lands)_
