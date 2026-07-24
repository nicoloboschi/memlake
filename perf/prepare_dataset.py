"""Offline prep for the perf runner: turn a BEIR dataset into a precomputed artifact of
real embeddings + texts, so the in-cluster load driver never embeds on the timed path.

This reuses the `bench/` BEIR download + embedding cache (the same vectors every other
engine reads), then packs a compact, self-describing artifact:

    <out>/corpus.npy          float32 (n_docs, dim), L2-normalized  (row i <-> corpus_ids[i])
    <out>/corpus_ids.json     list[str]
    <out>/corpus_texts.json   list[str]  (BEIR title + ' ' + text, row-aligned to corpus.npy)
    <out>/queries.npy         float32 (n_queries, dim), L2-normalized
    <out>/query_texts.json    list[str]  (row-aligned to queries.npy)
    <out>/meta.json           model / dim / counts / dataset

Then upload the directory to S3 and point the perf Job at it (see perf/README.md):

    uv run --project bench python perf/prepare_dataset.py --dataset scifact \\
        --model jinaai/jina-embeddings-v3 --out /tmp/perf-scifact-jina
    aws s3 sync /tmp/perf-scifact-jina s3://$BUCKET/_perf/scifact-jina-v3/

Embedding runs HERE, once, on CPU/GPU — deliberately off the runner's write/query timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the bench package (BEIR loaders + the embedding cache) without installing it.
_BENCH_SRC = Path(__file__).resolve().parents[1] / "bench" / "src"
if str(_BENCH_SRC) not in sys.path:
    sys.path.insert(0, str(_BENCH_SRC))

import numpy as np  # noqa: E402

from memlake_bench import datasets, embed  # noqa: E402


def _embed(model_name: str, texts: list[str], batch_size: int, label: str) -> np.ndarray:
    """Embed with fastembed and L2-normalize.

    Deliberately does NOT go through `embed.build`: that cache is keyed by dataset only, so
    embedding with a different model would overwrite the shared `testdata/embeddings/{dataset}`
    cache that the BEIR baselines in bench/results were computed against. The perf artifact is
    self-contained instead.
    """
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name)
    out: list[np.ndarray] = []
    done = 0
    for vec in model.embed(texts, batch_size=batch_size):
        out.append(np.asarray(vec, dtype=np.float32))
        done += 1
        if done % 1000 == 0:
            print(f"[prep] {label}: {done}/{len(texts)}", flush=True)
    a = np.ascontiguousarray(np.vstack(out), dtype=np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (a / norms).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    ap.add_argument(
        "--model",
        default="jinaai/jina-embeddings-v3",
        help="fastembed model name (dim comes from the model; the runner is dim-agnostic)",
    )
    ap.add_argument("--split", default=None, help="qrels split (default: dataset's default)")
    ap.add_argument("--out", required=True, help="output directory for the artifact")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-docs", type=int, default=0, help="subset the corpus (0 = all)")
    args = ap.parse_args()

    datasets.download(args.dataset)
    beir = datasets.load(args.dataset, args.split)
    print(f"[prep] {args.dataset}: {beir.n_docs} docs, {beir.n_queries} queries")

    corpus_ids, corpus_texts = beir.corpus_ids, beir.corpus_texts
    if args.max_docs and args.max_docs < len(corpus_ids):
        # Keep every doc some query is judged relevant to, then fill to max_docs — a subset that
        # still has real question->passage pairs rather than an arbitrary prefix.
        keep = {d for rel in beir.qrels.values() for d in rel}
        order = [i for i, d in enumerate(corpus_ids) if d in keep]
        order += [i for i, d in enumerate(corpus_ids) if d not in keep]
        order = order[: args.max_docs]
        order.sort()
        corpus_ids = [corpus_ids[i] for i in order]
        corpus_texts = [corpus_texts[i] for i in order]
        print(f"[prep] subset to {len(corpus_ids)} docs (kept all qrel-relevant docs)")

    prefix = embed._query_prefix(args.model)  # bge models need their query instruction; jina does not
    corpus_vecs = _embed(args.model, corpus_texts, args.batch_size, "corpus")
    query_vecs = _embed(args.model, [prefix + t for t in beir.query_texts], args.batch_size, "queries")
    dim = int(corpus_vecs.shape[1])
    print(f"[prep] embedded with {args.model} (dim={dim})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.npy", "wb") as f:
        np.save(f, corpus_vecs, allow_pickle=False)
    with open(out / "queries.npy", "wb") as f:
        np.save(f, query_vecs, allow_pickle=False)
    (out / "corpus_ids.json").write_text(json.dumps(corpus_ids))
    (out / "corpus_texts.json").write_text(json.dumps(corpus_texts))
    (out / "query_texts.json").write_text(json.dumps(beir.query_texts))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "model": args.model,
                "dim": dim,
                "n_docs": len(corpus_ids),
                "n_queries": len(beir.query_texts),
                "normalized": True,
                "query_prefix": prefix,
                "row_order": "corpus.npy[i] <-> corpus_ids[i] <-> corpus_texts[i]; queries.npy[i] <-> query_texts[i]",
            }
        )
    )
    print(f"[prep] wrote artifact to {out} ({len(corpus_ids)} docs, {len(beir.query_texts)} queries, dim {dim})")
    print(f"[prep] upload: aws s3 sync {out} s3://$BUCKET/_perf/{args.dataset}-<model-slug>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
