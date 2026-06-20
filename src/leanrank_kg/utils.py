from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["train", "val", "test"]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for path in [
        "data/sample",
        "data/processed/demo",
        "outputs/reports",
        "outputs/graph",
        "outputs/embeddings",
        "outputs/models",
        "homepage/assets",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def theorem_id(full_name: str) -> str:
    return f"thm:{full_name}"


def proof_state_id(full_name: str, tactic_idx: int, context: str) -> str:
    return f"ps:{full_name}:{int(tactic_idx)}:{stable_hash(context)}"


def premise_id(full_name: str) -> str:
    return f"premise:{full_name}"


def file_id(file_path: str) -> str:
    return f"file:{file_path}"


def technique_id(label: str) -> str:
    return f"proof_technique:{label}"


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def domain_from_path(file_path: str) -> tuple[str, str]:
    parts = [p for p in str(file_path).replace("\\", "/").split("/") if p]
    if "Mathlib" in parts:
        parts = parts[parts.index("Mathlib") + 1 :]
    domain = parts[0] if parts else "Unknown"
    subdomain = parts[1] if len(parts) > 1 else domain
    return domain, subdomain


def namespace(name: str) -> str:
    parts = str(name).split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else parts[0]


def minmax(values: pd.Series) -> pd.Series:
    values = values.fillna(0).astype(float)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return pd.Series([0.0] * len(values), index=values.index)
    return (values - lo) / (hi - lo)


def bucket(score: float) -> str:
    if score < 0.34:
        return "easy"
    if score < 0.67:
        return "medium"
    return "hard"
