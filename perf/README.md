# perf — the k8s load driver

`k8s_load.py` runs as an in-cluster Job against the Envoy proxy: it writes a corpus concurrently,
waits for the indexer to drain, then times queries. It prints one `PERF_SUMMARY {...}` JSON line.

## Corpus modes

**Real dataset (recommended).** Set `PERF_DATASET` to the S3 key prefix of an artifact built by
`prepare_dataset.py`: real BEIR passages and real questions, embedded ahead of time. The runner
downloads it *before* any timing, so **no embedding ever happens on the write/query hot path** —
only the precomputed vectors are sent.

This matters for more than realism. With synthetic near-uniform vectors a query has no genuine
near-neighbours, so every candidate scores alike, the vector arm's stage-two bound (`hi >= tau`)
cannot prune, and rerank rescores nearly the whole scanned set — which is what made rerank look like
~90% of query CPU in earlier runs. Real embeddings have the clustered similarity structure the index
is designed for, where the bound prunes to ~k. See `docs/perf-k8s-findings.md` (F7) and
`mlake_ivf`'s `a_contender_cap_bounds_worst_case_rerank_without_hurting_structured_recall`.

**Synthetic (fallback).** Leave `PERF_DATASET` unset for a dependency-free Gaussian cluster corpus.
Fine as a smoke test; not representative of production retrieval.

## Building a dataset artifact

Embedding is done once, offline, on your machine — deliberately not in the Job:

```bash
# 1) pack an artifact. --from-cache reuses testdata/embeddings/{dataset} (free, read-only);
#    drop it to embed with --model instead.
uv run --project bench python perf/prepare_dataset.py \
    --dataset scifact --from-cache --out /tmp/perf-scifact

# 2) upload to the perf bucket (creds: MEMLAKE_PERF_S3_* in the repo .env)
aws s3 sync /tmp/perf-scifact "s3://$MEMLAKE_PERF_S3_BUCKET/_perf/scifact-bge-small/"
```

`--max-docs N` subsets the corpus while keeping every qrel-relevant doc, so the questions still have
real answers in the index. Any `fastembed` model works; the runner reads the dim from the artifact.

**On bigger models.** A 1024d model would be more production-shaped than the 384d default, but CPU
embedding is the bottleneck: measured on this repo, `BAAI/bge-large-en-v1.5` (1024d) ran at ~0.2
docs/s on scifact — ~8 h for 5183 docs — and `jinaai/jina-embeddings-v3` (1024d, 8192 ctx) was
worse. Build a 1024d artifact on a GPU box if you want one; the runner is dim-agnostic and needs no
change. Note the dim affects absolute scan/rerank cost, not the *shape* of the finding below.

It does **not** write to `testdata/embeddings/{dataset}` — that cache is keyed by dataset only, and
the BEIR baselines in `bench/results/` were computed with the model already cached there.

## Running

```bash
kubectl -n memlake-dev apply -f deploy/perf-job.yaml
kubectl -n memlake-dev logs -f job/memlake-loadgen
```

`deploy/perf-job.yaml` pulls the S3 credentials from the `memlake-s3` secret (created by
`deploy/deploy-dev.sh` from the repo `.env`) and sets `PERF_DATASET`. `N_DOCS` is capped at the
dataset's corpus size.

The loadgen image is built from `demo/Dockerfile` (repo root as context):

```bash
docker buildx build --platform linux/amd64 -f demo/Dockerfile \
    -t ghcr.io/nicoloboschi/memlake-loadgen:dev --push .
```

## Knobs

| env | meaning |
|---|---|
| `MEMLAKE_ADDR` | proxy address (`memlake-proxy:50050` in-cluster) |
| `MEMLAKE_LOAD_NAMESPACE` | namespace to write into |
| `PERF_DATASET` | S3 key prefix of the artifact; unset ⇒ synthetic |
| `N_DOCS` / `BATCH` / `CONCURRENCY` | write shape (docs capped at corpus size) |
| `N_QUERIES` | queries per pass (real questions cycle if it exceeds the query set) |
| `DIM` / `CLUSTERS` | synthetic mode only — real mode takes dim from the artifact |
| `FOLD_WAIT_SECS` | how long to poll for the fold to drain before querying |
