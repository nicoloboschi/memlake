"""BEIR dataset download + loading.

BEIR layout after extraction (testdata/beir/{name}/):
    corpus.jsonl      {"_id","title","text",...} one per line
    queries.jsonl     {"_id","text",...}
    qrels/test.tsv    query-id \t corpus-id \t score   (header row)
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from .paths import beir_dir

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# Datasets we support out of the box. approx_docs is informational only.
DATASETS: dict[str, dict] = {
    "scifact": {"approx_docs": 5_183, "split": "test"},
    "nfcorpus": {"approx_docs": 3_633, "split": "test"},
    "fiqa": {"approx_docs": 57_638, "split": "test"},
    "trec-covid": {"approx_docs": 171_332, "split": "test"},
    "arguana": {"approx_docs": 8_674, "split": "test"},
    "scidocs": {"approx_docs": 25_657, "split": "test"},
    # Mid-size: still embeddable on one machine in minutes-to-an-hour.
    "touche2020": {"approx_docs": 382_545, "split": "test"},
    "quora": {"approx_docs": 522_931, "split": "test"},
    "trec-news": {"approx_docs": 594_977, "split": "test"},
    # Large: these dominate published BEIR tables but embedding them locally is an
    # overnight job, and `ann_recall` needs an exhaustive scan per query on top — which is
    # O(queries x docs) and becomes the bottleneck well before retrieval does. Use
    # --no-ann-recall on these unless you mean it.
    "nq": {"approx_docs": 2_681_468, "split": "test"},
    "dbpedia-entity": {"approx_docs": 4_635_922, "split": "test"},
    "hotpotqa": {"approx_docs": 5_233_329, "split": "test"},
    "fever": {"approx_docs": 5_416_568, "split": "test"},
    "climate-fever": {"approx_docs": 5_416_593, "split": "test"},
    "msmarco": {"approx_docs": 8_841_823, "split": "dev"},
}


def default_split(dataset: str) -> str:
    return DATASETS.get(dataset, {}).get("split", "test")


@dataclass
class Beir:
    """A loaded BEIR dataset.

    corpus_ids / query_ids define the canonical row order used by every
    downstream artifact (embeddings .npy rows, Qdrant point ids, metrics).
    """

    name: str
    split: str
    corpus_ids: list[str]
    corpus_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]

    @property
    def n_docs(self) -> int:
        return len(self.corpus_ids)

    @property
    def n_queries(self) -> int:
        return len(self.query_ids)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def download(dataset: str, force: bool = False) -> Path:
    """Fetch + extract a BEIR dataset. Idempotent: skips if already present."""
    target = beir_dir(dataset)
    marker = target / "corpus.jsonl"
    if marker.exists() and not force:
        print(f"[download] {dataset}: already present at {target}")
        return target

    if force and target.exists():
        shutil.rmtree(target)

    url = BEIR_URL.format(name=dataset)
    zip_path = beir_dir() / f"{dataset}.zip"
    print(f"[download] {dataset}: fetching {url}")
    _download(url, zip_path)

    print(f"[download] {dataset}: extracting")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(beir_dir())
    zip_path.unlink(missing_ok=True)

    if not marker.exists():
        raise RuntimeError(f"extraction did not produce {marker}")
    print(f"[download] {dataset}: ready at {target}")
    return target


def _doc_text(rec: dict) -> str:
    """BEIR convention: title and body concatenated for retrieval."""
    title = (rec.get("title") or "").strip()
    body = (rec.get("text") or "").strip()
    return f"{title} {body}".strip() if title else body


def load(dataset: str, split: str | None = None) -> Beir:
    """Load a downloaded BEIR dataset into memory.

    Queries are filtered to those that actually appear in the split's qrels,
    which is what BEIR evaluates against (e.g. scifact test = 300 of 1109).
    """
    split = split or default_split(dataset)
    root = beir_dir(dataset)
    if not (root / "corpus.jsonl").exists():
        raise FileNotFoundError(
            f"{dataset} not downloaded. Run: memlake-bench download {dataset}"
        )

    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            corpus_ids.append(str(rec["_id"]))
            corpus_texts.append(_doc_text(rec))

    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"missing qrels split: {qrels_path}")

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        header = f.readline()  # query-id\tcorpus-id\tscore
        if "query-id" not in header:
            f.seek(0)  # no header after all
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, {})[did] = score

    all_queries: dict[str, str] = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            all_queries[str(rec["_id"])] = (rec.get("text") or "").strip()

    query_ids = [q for q in all_queries if q in qrels]
    query_ids.sort()
    query_texts = [all_queries[q] for q in query_ids]

    return Beir(
        name=dataset,
        split=split,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
    )
