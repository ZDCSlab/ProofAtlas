from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import pandas as pd

from .utils import domain_from_path, ensure_dirs, load_config, write_json, write_jsonl, write_parquet

REQUIRED_FIELDS = [
    "file_path",
    "full_name",
    "start",
    "tactic_idx",
    "context",
    "all_pos_premises",
    "pos_premise",
    "neg_premises",
]

DOMAINS = [
    ("Algebra", "Group", "mul_assoc"),
    ("Topology", "Basic", "isOpen_univ"),
    ("LinearAlgebra", "Matrix", "Matrix.mul_assoc"),
    ("NumberTheory", "Divisibility", "Nat.dvd_trans"),
    ("Data", "Nat", "Nat.add_comm"),
    ("Order", "Lattice", "le_trans"),
]


def _premise(name: str, domain: str, subdomain: str, i: int) -> dict[str, str]:
    return {
        "full_name": name,
        "file_path": f"Mathlib/{domain}/{subdomain}.lean",
        "code": f"theorem {name.split('.')[-1]} := by simp",
        "start": str(i),
    }


def synthetic_rows(total_rows: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    theorem_count = max(12, total_rows)
    rows: list[dict[str, Any]] = []
    shared_premises = [
        _premise(f"{domain}.{sub}.{base}", domain, sub, i)
        for i, (domain, sub, base) in enumerate(DOMAINS)
    ]
    tactics = ["simp", "rw", "exact", "intro", "cases", "constructor", "linarith", "ring", "induction"]
    for t in range(theorem_count):
        domain, subdomain, base = DOMAINS[t % len(DOMAINS)]
        theorem = f"{domain}.{subdomain}.demo_theorem_{t:04d}"
        file_path = f"Mathlib/{domain}/{subdomain}/Demo{t % 7}.lean"
        local = f"x{t} : Nat\ny{t} : Nat"
        for step in range(1 + (t % 4)):
            if len(rows) >= total_rows:
                break
            tactic = tactics[(t + step) % len(tactics)]
            goal = f"{base} x{t} y{t}"
            context = f"{local}\nh{step} : x{t} = y{t}\n⊢ {goal}"
            pos = shared_premises[(t + step) % len(shared_premises)]
            all_pos = [pos, shared_premises[t % len(shared_premises)]]
            negs = rng.sample(shared_premises, k=min(3, len(shared_premises)))
            rows.append(
                {
                    "file_path": file_path,
                    "full_name": theorem,
                    "start": str(step),
                    "tactic_idx": step,
                    "context": context,
                    "tactic": tactic,
                    "all_pos_premises": all_pos,
                    "pos_premise": pos,
                    "neg_premises": negs,
                }
            )
    return pd.DataFrame(rows)


def _try_huggingface(config: dict[str, Any], limit: int) -> pd.DataFrame | None:
    if not config.get("use_huggingface", False):
        return None
    try:
        from datasets import load_dataset

        ds = load_dataset(config["dataset_name"], split=f"train[:{limit}]")
        return pd.DataFrame(ds)
    except Exception as exc:
        write_json("outputs/reports/huggingface_error.json", {"error": str(exc)})
        return None


def _adapt_premise(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        name = value.get("full_name") or value.get("name") or value.get("premise") or str(value)
        return {
            "full_name": str(name),
            "file_path": str(value.get("file_path") or value.get("path") or ""),
            "code": str(value.get("code") or value.get("statement") or name),
            "start": str(value.get("start") or ""),
        }
    return {"full_name": str(value), "file_path": "", "code": str(value), "start": ""}


def adapt_rows(df: pd.DataFrame) -> pd.DataFrame:
    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            df[field] = None
    df = df.dropna(subset=["file_path", "full_name", "context", "pos_premise"]).copy()
    df["pos_premise"] = df["pos_premise"].map(_adapt_premise)
    df["all_pos_premises"] = df["all_pos_premises"].map(
        lambda xs: [_adapt_premise(x) for x in (xs if isinstance(xs, list) else [])]
    )
    df["neg_premises"] = df["neg_premises"].map(
        lambda xs: [_adapt_premise(x) for x in (xs if isinstance(xs, list) else [])]
    )
    if "tactic" not in df.columns:
        df["tactic"] = ""
    df["tactic_idx"] = df["tactic_idx"].fillna(0).astype(int)
    domains = df["file_path"].map(domain_from_path)
    df["domain_tag"] = [d for d, _ in domains]
    df["subdomain_tag"] = [s for _, s in domains]
    return df[REQUIRED_FIELDS + ["tactic", "domain_tag", "subdomain_tag"]]


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        first = value[0] if value else None
        return {"kind": "list", "length": len(value), "item_shape": _shape(first) if first is not None else None}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(map(str, value.keys()))}
    return {"kind": type(value).__name__}


def split_by_theorem(df: pd.DataFrame, seed: int, ratios: dict[str, float]) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    names = sorted(df["full_name"].unique())
    rng.shuffle(names)
    n = len(names)
    n_train = max(1, int(n * ratios["train_ratio"]))
    n_val = max(1, int(n * ratios["val_ratio"])) if n >= 3 else 0
    mapping = {}
    for i, name in enumerate(names):
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        mapping[name] = split
    write_json("data/sample/split_assignments.json", mapping)
    splits = {split: df[df["full_name"].map(mapping) == split].reset_index(drop=True) for split in ["train", "val", "test"]}
    split_names = {split: set(frame["full_name"]) for split, frame in splits.items()}
    overlaps = {
        "train_vs_val": sorted(split_names["train"] & split_names["val"]),
        "train_vs_test": sorted(split_names["train"] & split_names["test"]),
        "val_vs_test": sorted(split_names["val"] & split_names["test"]),
    }
    write_json(
        "outputs/reports/split_leakage_report.json",
        {
            "theorem_counts": {split: int(len(names)) for split, names in split_names.items()},
            "overlaps": overlaps,
            "has_leakage": any(bool(v) for v in overlaps.values()),
        },
    )
    return splits


def run(config_path: str, debug_rows: int | None = None) -> None:
    ensure_dirs()
    config = load_config(config_path)
    total_rows = int(debug_rows if debug_rows is not None else config["sample"]["total_rows"])
    seed = int(config["random_seed"])
    raw = _try_huggingface(config, total_rows)
    if raw is None:
        raw = synthetic_rows(total_rows, seed)
    raw = raw.head(total_rows).reset_index(drop=True)
    schema = {
        "columns": {col: str(dtype) for col, dtype in raw.dtypes.items()},
        "observed_shapes": {
            field: [_shape(v) for v in raw[field].head(5).tolist()] if field in raw.columns else []
            for field in ["pos_premise", "all_pos_premises", "neg_premises"]
        },
    }
    write_json("outputs/reports/raw_schema.json", schema)
    write_jsonl("outputs/reports/raw_preview.jsonl", raw.head(5).to_dict(orient="records"))
    df = adapt_rows(raw)
    write_parquet(df, "data/sample/all_rows.parquet")
    splits = split_by_theorem(df, seed, config["split"])
    domain_report: dict[str, dict[str, int]] = defaultdict(dict)
    for split, split_df in splits.items():
        write_parquet(split_df, f"data/sample/{split}_rows.parquet")
        domain_report[split] = split_df["domain_tag"].value_counts().to_dict()
    write_json("outputs/reports/domain_distribution.json", domain_report)
