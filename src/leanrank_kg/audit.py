from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import SPLITS, read_json, write_json


def _exists(path: str) -> dict[str, Any]:
    p = Path(path)
    return {"path": path, "passed": p.exists(), "detail": "exists" if p.exists() else "missing"}


def _parquet_nonempty(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": path, "passed": False, "detail": "missing"}
    rows = len(pd.read_parquet(p))
    return {"path": path, "passed": rows > 0, "detail": f"{rows} rows"}


def _json_condition(path: str, predicate, detail_key: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": path, "passed": False, "detail": "missing"}
    data = read_json(p)
    passed, detail = predicate(data)
    return {"path": path, "passed": bool(passed), "detail": detail_key.format(detail=detail)}


def build_audit() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    required_root = ["README.md", "pyproject.toml", "Makefile", "configs/sample.yaml", "homepage/index.html"]
    for path in required_root:
        checks[f"file:{path}"] = _exists(path)
    for schema in ["theorem", "proof_state", "premise", "file_module", "proof_technique"]:
        checks[f"schema:{schema}"] = _exists(f"schemas/{schema}.schema.json")
    for split in SPLITS + ["demo"]:
        for table in [
            "theorems",
            "proof_states",
            "premises",
            "file_modules",
            "positive_edges",
            "negative_edges",
            "proof_techniques",
            "proof_state_techniques",
            "premise_techniques",
            "proof_state_features",
            "theorem_features",
        ]:
            checks[f"processed:{split}:{table}"] = _exists(f"data/processed/{split}/{table}.parquet")
        checks[f"graph:{split}:nodes"] = _parquet_nonempty(f"outputs/graph/{split}/nodes_enriched.parquet")
        checks[f"graph:{split}:edges"] = _parquet_nonempty(f"outputs/graph/{split}/edges_enriched.parquet")
        checks[f"embedding:{split}:proof_state"] = _exists(f"outputs/embeddings/{split}_proof_state_embeddings.npz")
        checks[f"embedding:{split}:premise"] = _exists(f"outputs/embeddings/{split}_premise_embeddings.npz")
        checks[f"embedding:{split}:theorem"] = _exists(f"outputs/embeddings/{split}_theorem_embeddings.npz")
    for path in [
        "outputs/reports/raw_schema.json",
        "outputs/reports/domain_distribution.json",
        "outputs/reports/metrics.json",
        "outputs/reports/retrieval_examples.json",
        "outputs/reports/retrieval_examples.md",
        "outputs/reports/proof_technique_distribution.csv",
        "outputs/reports/difficulty_distribution.csv",
        "outputs/reports/homepage_summary.json",
        "outputs/models/premise_ranker.joblib",
        "notebooks/leanrank_kg_demo.ipynb",
    ]:
        checks[f"artifact:{path}"] = _exists(path)
    checks["validation:schema"] = _json_condition(
        "outputs/reports/schema_validation_summary.json",
        lambda data: (data.get("error_count") == 0, data.get("error_count")),
        "error_count={detail}",
    )
    checks["validation:split_leakage"] = _json_condition(
        "outputs/reports/split_leakage_report.json",
        lambda data: (not data.get("has_leakage", True), data.get("has_leakage")),
        "has_leakage={detail}",
    )
    checks["validation:graph_endpoints"] = _json_condition(
        "outputs/reports/graph_validation_summary.json",
        lambda data: (
            all(row.get("missing_endpoint_count") == 0 and row.get("networkx_loadable") for row in data.values()),
            {split: row.get("missing_endpoint_count") for split, row in data.items()},
        ),
        "missing_endpoint_counts={detail}",
    )
    checks["validation:retrieval_examples"] = _json_condition(
        "outputs/reports/retrieval_examples.json",
        lambda data: (len(data) >= 20, len(data)),
        "examples={detail}",
    )
    checks["validation:homepage_sections"] = {
        "path": "homepage/index.html",
        "passed": Path("homepage/index.html").exists()
        and all(
            token in Path("homepage/index.html").read_text(encoding="utf-8")
            for token in ["KG Overview", "Retrieval Examples", "Proof-Technique Labels", "Difficulty", "Evaluation", "Reproducibility"]
        ),
        "detail": "required sections present",
    }
    passed = all(check["passed"] for check in checks.values())
    audit = {
        "passed": passed,
        "total_checks": len(checks),
        "failed_checks": [name for name, check in checks.items() if not check["passed"]],
        "checks": checks,
    }
    write_json("outputs/reports/mvp_completion_audit.json", audit)
    return audit


def run(config_path: str | None = None) -> None:
    build_audit()
