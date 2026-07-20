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


def cmd_all(args) -> int:
    datasets.download(args.dataset)
    beir = _load(args.dataset, args.split)
    embed.build(beir, model_name=args.model, batch_size=args.batch_size)
    cmd_baseline_exact(args)
    cmd_baseline_qdrant(args)
    report.write()
    return 0


def cmd_report(args) -> int:
    report.write()
    print()
    print(report.render())
    return 0


def _print_summary(payload: dict) -> None:
    print(f"\n=== {payload['engine']} / {payload['dataset'] if 'dataset' in payload else ''} ===")
    print(f"{'arm':8} {'nDCG@10':>9} {'R@100':>9} {'MRR@10':>9} {'p50ms':>8} {'p99ms':>8}")
    for arm, m in payload["arms"].items():
        lat = m.get("latency", {})
        print(
            f"{arm:8} {m['ndcg@10']:>9.4f} {m['recall@100']:>9.4f} {m['mrr@10']:>9.4f} "
            f"{lat.get('p50_ms', 0):>8.1f} {lat.get('p99_ms', 0):>8.1f}"
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

    sp = sub.add_parser("all", help="download -> embed -> exact -> qdrant -> report")
    add_common(sp, model=True, batch=True)
    sp.add_argument("--top-k", type=int, default=100)
    sp.add_argument("--rrf-k", type=int, default=60)
    sp.add_argument("--recreate", action="store_true")
    sp.set_defaults(func=cmd_all)

    sp = sub.add_parser("report", help="render bench/results/report.md")
    sp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
