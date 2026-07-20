"""Result persistence: bench/results/{dataset}/{engine}.json (committed, small)."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

from .paths import results_dir


def save(dataset: str, engine: str, payload: dict) -> str:
    out = results_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{engine}.json"
    payload = dict(payload)
    payload.setdefault("dataset", dataset)
    payload.setdefault("engine", engine)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["platform"] = f"{platform.system()} {platform.machine()} py{platform.python_version()}"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"[results] wrote {path}")
    return str(path)


def load_all() -> dict[str, dict[str, dict]]:
    """-> {dataset: {engine: payload}}"""
    base = results_dir()
    out: dict[str, dict[str, dict]] = {}
    if not base.exists():
        return out
    for ds_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(ds_dir.glob("*.json")):
            out.setdefault(ds_dir.name, {})[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out
