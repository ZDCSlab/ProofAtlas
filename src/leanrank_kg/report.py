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


def _overview(dataset: dict[str, Any], graph_stats: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    split_counts = dataset["split_counts"]
    train = next((row for row in split_counts if row["split"] == "train"), {})
    total_nodes = sum(int(stats.get("node_count", 0)) for stats in graph_stats.values())
    total_edges = sum(int(stats.get("edge_count", 0)) for stats in graph_stats.values())
    total_theorems = sum(int(row.get("theorems", 0)) for row in split_counts if row.get("split") in {"train", "val", "test"})
    graph_validation = validation.get("graph", {})
    missing_endpoints = sum(int(row.get("missing_endpoint_count", 0)) for row in graph_validation.values())
    networkx_ok = all(bool(row.get("networkx_loadable", False)) for row in graph_validation.values()) if graph_validation else False
    return {
        "train_theorems": int(train.get("theorems", 0)),
        "total_theorems": total_theorems,
        "train_proof_states": int(train.get("proof_states", 0)),
        "train_premises": int(train.get("premises", 0)),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "schema_errors": int(validation.get("schema", {}).get("error_count", 0) or 0),
        "split_leakage": bool(validation.get("split_leakage", {}).get("has_leakage", True)),
        "missing_graph_endpoints": missing_endpoints,
        "networkx_ok": networkx_ok,
        "max_split_nodes": max((int(row.get("nodes", 0)) for row in split_counts), default=1),
        "max_split_edges": max((int(row.get("edges", 0)) for row in split_counts), default=1),
    }


def _top_domains(domains: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for split_counts in domains.values():
        if not isinstance(split_counts, dict):
            continue
        for domain, count in split_counts.items():
            counts[str(domain)] = counts.get(str(domain), 0) + int(count)
    total = sum(counts.values()) or 1
    rows = [
        {"domain": domain, "count": count, "share": count / total}
        for domain, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
    return rows


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
    dataset = {
        "project_name": project_name,
        "source": "erbacher/LeanRank-data",
        "configured_rows": int(config.get("sample", {}).get("total_rows", 0) or 0),
        "sample_rows": _sample_row_count(),
        "use_huggingface": bool(config.get("use_huggingface", False)),
        "raw_columns": raw_schema.get("columns", {}),
        "premise_shape": raw_schema.get("observed_shapes", {}).get("pos_premise", [{}])[0],
        "processed_files": _processed_files(),
        "split_counts": _split_counts(graph_stats),
    }
    summary = {
        "dataset": {
            **dataset,
        },
        "overview": _overview(dataset, graph_stats, validation),
        "top_domains": _top_domains(domains),
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
    summary["overview"]["max_domain_count"] = max((int(row["count"]) for row in summary["top_domains"]), default=1)
    summary["overview"]["max_technique_count"] = max((int(row.get("count", 0)) for row in summary["proof_techniques"]), default=1)
    summary["overview"]["max_difficulty_count"] = max((int(row.get("count", 0)) for row in summary["difficulty_distribution"]), default=1)
    write_json("outputs/reports/homepage_summary.json", summary)
    return summary


def run(config_path: str | None = None) -> None:
    build_summary(config_path)
