from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import load_config, read_json, write_json


def _csv_records(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if limit is not None:
        df = df.head(limit)
    return df.to_dict(orient="records")


def _processed_files() -> list[str]:
    return sorted(str(p) for p in Path("data/processed/demo").glob("*.parquet"))


def _split_counts(graph_stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, stats in sorted(graph_stats.items()):
        node_counts = stats.get("node_counts_by_type", {})
        rows.append(
            {
                "split": split,
                "theorems": node_counts.get("Theorem", 0),
                "proof_states": node_counts.get("ProofState", 0),
                "premises": node_counts.get("Premise", 0),
                "proof_techniques": node_counts.get("ProofTechnique", 0),
                "nodes": stats.get("node_count", 0),
                "edges": stats.get("edge_count", 0),
            }
        )
    return rows


def _sample_row_count() -> int:
    path = Path("data/sample/all_rows.parquet")
    if not path.exists():
        return 0
    return int(len(pd.read_parquet(path)))


def build_summary(config_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path) if config_path else {}
    project_name = config.get("project_name", "ProofAtlas")
    graph_stats = read_json("outputs/reports/graph_stats_summary.json", {})
    metrics = read_json("outputs/reports/metrics.json", {})
    embedding_config = read_json("outputs/embeddings/embedding_config.json", {})
    examples = read_json("outputs/reports/retrieval_examples.json", [])
    domains = read_json("outputs/reports/domain_coverage.json", {})
    raw_schema = read_json("outputs/reports/raw_schema.json", {})
    validation = {
        "schema": read_json("outputs/reports/schema_validation_summary.json", {}),
        "split_leakage": read_json("outputs/reports/split_leakage_report.json", {}),
        "context": read_json("outputs/reports/context_parse_coverage.json", {}),
        "graph": read_json("outputs/reports/graph_validation_summary.json", {}),
    }
    summary = {
        "dataset": {
            "project_name": project_name,
            "source": "erbacher/LeanRank-data",
            "configured_rows": int(config.get("sample", {}).get("total_rows", 0) or 0),
            "sample_rows": _sample_row_count(),
            "use_huggingface": bool(config.get("use_huggingface", False)),
            "raw_columns": raw_schema.get("columns", {}),
            "premise_shape": raw_schema.get("observed_shapes", {}).get("pos_premise", [{}])[0],
            "processed_files": _processed_files(),
            "split_counts": _split_counts(graph_stats),
        },
        "embedding": embedding_config,
        "graph_stats": graph_stats,
        "domain_coverage": domains,
        "functions": [
            "retrieve_premises",
            "retrieve_similar_theorems",
            "explain_premise_match",
            "get_proof_technique_labels",
            "get_difficulty_profile",
            "get_graph_neighborhood",
        ],
        "retrieval_examples": examples,
        "proof_techniques": _csv_records("outputs/reports/proof_technique_distribution.csv"),
        "difficulty_distribution": _csv_records("outputs/reports/difficulty_distribution.csv"),
        "metrics": metrics,
        "validation": validation,
        "commands": [
            "conda activate leanrank_kg",
            "make install",
            "leanrank-kg full-pipeline --config configs/proofatlas.yaml",
            "leanrank-kg build-homepage --config configs/proofatlas.yaml",
        ],
    }
    write_json("outputs/reports/homepage_summary.json", summary)
    return summary


def run(config_path: str | None = None) -> None:
    build_summary(config_path)
