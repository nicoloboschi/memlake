"""memlake-bench CLI.

    memlake-bench download <dataset>
    memlake-bench embed <dataset>
    memlake-bench baseline exact <dataset>
    memlake-bench baseline qdrant <dataset>
    memlake-bench all <dataset>
    memlake-bench report
"""

from __future__ import annotations

import argparse
import sys

from . import datasets, embed, report, results
from .engines import exact as exact_engine


def _load(dataset: str, split: str | None):
    beir = datasets.load(dataset, split)
    print(f"[{dataset}] {beir.n_docs} docs, {beir.n_queries} queries (split={beir.split})")
    return beir


def cmd_download(args) -> int:
    datasets.download(args.dataset, force=args.force)
    return 0


def cmd_embed(args) -> int:
    beir = _load(args.dataset, args.split)
    embed.build(beir, model_name=args.model, batch_size=args.batch_size, force=args.force)
    return 0


def cmd_baseline_exact(args) -> int:
    beir = _load(args.dataset, args.split)
    emb = embed.load(args.dataset)
    payload = exact_engine.run(beir, emb, top_k=args.top_k, rrf_k=args.rrf_k)
    results.save(args.dataset, "exact", payload)
    _print_summary(payload)
    return 0


def cmd_baseline_qdrant(args) -> int:
    from . import qdrant_docker
    from .engines import qdrant_engine

    beir = _load(args.dataset, args.split)
    emb = embed.load(args.dataset)
    url = qdrant_docker.ensure_running()
    payload = qdrant_engine.run(
        beir,
        emb,
        url=url,
        top_k=args.top_k,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )
    results.save(args.dataset, "qdrant", payload)
    _print_summary(payload)
    return 0


def cmd_baseline_memlake(args) -> int:
    from .engines import memlake_grpc

    beir = _load(args.dataset, args.split)
    # memlake reads the same cached embeddings as every other engine; only the retrieval
    # path differs. Accuracy is now measured e2e through the gRPC server (client -> server
    # -> S3), the deployed path.
    emb = embed.load(args.dataset)
    graph = bool(getattr(args, "graph", False))
    engine_name = "memlake+graph" if graph else "memlake"
    # Exhaustive ground truth for ann_recall@k: what a brute-force scan over the whole
    # corpus would have returned. This is the metric that moves when the codec, nprobe or
    # clustering change — nDCG can sit still while the index quietly stops finding things,
    # because what it drops was never in the qrels.
    truth = None
    if not getattr(args, "no_ann_recall", False):
        from .engines import exact as exact_engine

        truth = exact_engine.dense_ground_truth(emb, args.top_k)
    payload = memlake_grpc.run(
        beir, emb, top_k=args.top_k, graph=graph, engine_name=engine_name, truth=truth
    )
    results.save(args.dataset, engine_name, payload)
    _print_summary(payload)
    return 0


def cmd_all(args) -> int:
    datasets.download(args.dataset)
    beir = _load(args.dataset, args.split)
    embed.build(beir, model_name=args.model, batch_size=args.batch_size)
    cmd_baseline_exact(args)
    cmd_baseline_qdrant(args)
    cmd_baseline_memlake(args)
    report.write()
    return 0


def cmd_report(args) -> int:
    report.write()
    print()
    print(report.render())
    return 0


def cmd_recall(args) -> int:
    from . import recall_check

    recall_check.run(keep=args.keep)
    return 0


def cmd_chaos(args) -> int:
    from . import chaos

    cfg = chaos.ChaosConfig.from_env()
    # CLI flags override env for the common knobs.
    if args.nodes:
        cfg.nodes = args.nodes
    if args.docs:
        cfg.docs = args.docs
    if args.secs:
        cfg.secs = args.secs
    if args.no_kill:
        cfg.kill_every = 0.0
    try:
        chaos.run(cfg)
        return 0
    except AssertionError as e:
        print(str(e))
        return 1


def cmd_perf(args) -> int:
    from . import perf
    from .perf_datagen import GenConfig

    scales = [int(s) for s in str(args.scales).split(",") if s.strip()]
    reports = []
    for scale in scales:
        print(f"=== perf @ {scale} (e2e via python client) ===")
        cfg = GenConfig(scale=scale, memory_types=args.types, seed=args.seed)
        rep = perf.run(
            cfg,
            queries=args.queries,
            mem_mb=args.mem_mb,
            disk_mb=args.disk_mb,
        )
        print()
        print(rep.render())
        print()
        reports.append(rep)
    return 0


def _print_summary(payload: dict) -> None:
    print(f"\n=== {payload['engine']} / {payload['dataset'] if 'dataset' in payload else ''} ===")
    print(f"{'arm':8} {'nDCG@10':>9} {'R@100':>9} {'MRR@10':>9} {'p50ms':>8} {'p99ms':>8}")
    for arm, m in payload["arms"].items():
        lat = m.get("latency", {})
        print(
            f"{arm:8} {m['ndcg@10']:>9.4f} {m['recall@100']:>9.4f} {m['mrr@10']:>9.4f} "
            + (f"ann@10 {m['ann_recall@10']:>7.4f} " if "ann_recall@10" in m else "")
            + f"{lat.get('p50_ms', 0):>8.1f} {lat.get('p99_ms', 0):>8.1f}"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memlake-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, *, split=True, model=False, batch=False):
        sp.add_argument("dataset", choices=sorted(datasets.DATASETS))
        if split:
            sp.add_argument("--split", default=None, help="qrels split (default: test)")
        if model:
            sp.add_argument("--model", default=embed.DEFAULT_MODEL)
        if batch:
            sp.add_argument("--batch-size", type=int, default=256)

    sp = sub.add_parser("download", help="fetch a BEIR dataset into testdata/beir/")
    add_common(sp, split=False)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("embed", help="embed corpus+queries into testdata/embeddings/")
    add_common(sp, model=True, batch=True)
    sp.add_argument("--force", action="store_true", help="ignore the cache and re-embed")
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("baseline", help="run a retrieval baseline")
    bsub = sp.add_subparsers(dest="engine", required=True)

    e = bsub.add_parser("exact", help="numpy brute-force dense + python BM25 + RRF")
    add_common(e)
    e.add_argument("--top-k", type=int, default=100)
    e.add_argument("--rrf-k", type=int, default=60)
    e.set_defaults(func=cmd_baseline_exact)

    e = bsub.add_parser("qdrant", help="Qdrant HNSW dense + native BM25 sparse + RRF")
    add_common(e, batch=True)
    e.add_argument("--top-k", type=int, default=100)
    e.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    e.set_defaults(func=cmd_baseline_qdrant)

    def add_memlake_tuning(sp):
        sp.add_argument("--nprobe", default=None, help="IVF clusters probed per query")
        sp.add_argument("--vec-weight", default=None, help="RRF weight for the vector arm")
        sp.add_argument("--fts-weight", default=None, help="RRF weight for the FTS arm")
        sp.add_argument(
            "--graph",
            action="store_true",
            help="synthesize kNN links and add the graph-expansion arm to fusion",
        )

    e = bsub.add_parser("memlake", help="e2e memlake via gRPC server (client -> server -> S3)")
    add_common(e)
    add_memlake_tuning(e)
    e.add_argument("--top-k", type=int, default=100)
    e.set_defaults(func=cmd_baseline_memlake)

    sp = sub.add_parser("all", help="download -> embed -> exact -> qdrant -> memlake -> report")
    add_common(sp, model=True, batch=True)
    sp.add_argument("--top-k", type=int, default=100)
    sp.add_argument("--rrf-k", type=int, default=60)
    sp.add_argument("--recreate", action="store_true")
    # memlake tuning knobs default to None so `all` uses the binary's built-in defaults.
    sp.set_defaults(
        func=cmd_all,
        graph=False,
        nprobe=None,
        vec_weight=None,
        fts_weight=None,
    )

    sp = sub.add_parser("report", help="render bench/results/report.md")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser(
        "recall",
        help="e2e recall regression: write -> index -> assert every arm recalls (via the client)",
    )
    sp.add_argument("--keep", action="store_true", help="use a fixed namespace instead of a fresh one")
    sp.set_defaults(func=cmd_recall)

    sp = sub.add_parser(
        "chaos",
        help="multi-node chaos & correctness suite (spawns N nodes, kills them, asserts no data loss)",
    )
    sp.add_argument("--nodes", type=int, default=0, help="serve nodes (0 = env CHAOS_NODES or 3)")
    sp.add_argument("--docs", type=int, default=0, help="total ops (0 = env CHAOS_DOCS or 5000)")
    sp.add_argument("--secs", type=int, default=0, help="wall-clock cap (0 = env CHAOS_SECS or 60)")
    sp.add_argument("--no-kill", action="store_true", help="disable node kills (happy-path run)")
    sp.set_defaults(func=cmd_chaos)

    sp = sub.add_parser(
        "perf",
        help="e2e performance: python client -> gRPC server -> S3 (write/index/read)",
    )
    sp.add_argument("--scales", default="10000", help="comma list, e.g. 10000,100000,1000000")
    sp.add_argument("--types", type=int, default=3, help="memory_types")
    sp.add_argument("--queries", type=int, default=200)
    sp.add_argument("--mem-mb", type=int, default=64, help="server read-cache memory budget")
    sp.add_argument("--disk-mb", type=int, default=512, help="server read-cache disk budget")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_perf)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
