from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import networkx as nx
import pandas as pd

from .utils import ROOT, SPLITS, write_json

SCHEMA_MAP = {
    "theorems": "theorem.schema.json",
    "proof_states": "proof_state.schema.json",
    "premises": "premise.schema.json",
    "file_modules": "file_module.schema.json",
    "proof_techniques": "proof_technique.schema.json",
}


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def validate_processed_schemas(splits: list[str] | None = None) -> dict[str, Any]:
    splits = splits or SPLITS + ["demo"]
    summary: dict[str, Any] = {"tables": {}, "error_count": 0, "errors": []}
    schemas = {}
    for table, schema_name in SCHEMA_MAP.items():
        schema_path = Path("schemas") / schema_name
        if not schema_path.exists():
            schema_path = ROOT / "schemas" / schema_name
        with open(schema_path, "r", encoding="utf-8") as fh:
            schemas[table] = json.load(fh)
    for split in splits:
        for table, schema in schemas.items():
            path = Path(f"data/processed/{split}/{table}.parquet")
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            key = f"{split}/{table}"
            summary["tables"][key] = {"rows": int(len(df)), "valid_rows": 0}
            for idx, row in enumerate(df.to_dict(orient="records")):
                record = {k: _json_ready(v) for k, v in row.items()}
                try:
                    jsonschema.validate(record, schema)
                    summary["tables"][key]["valid_rows"] += 1
                except jsonschema.ValidationError as exc:
                    summary["error_count"] += 1
                    if len(summary["errors"]) < 100:
                        summary["errors"].append({"split": split, "table": table, "row_index": idx, "error": exc.message})
    write_json("outputs/reports/schema_validation_summary.json", summary)
    return summary


def validate_split_leakage() -> dict[str, Any]:
    split_names = {}
    for split in SPLITS:
        path = Path(f"data/sample/{split}_rows.parquet")
        if path.exists():
            split_names[split] = set(pd.read_parquet(path)["full_name"])
    overlaps = {}
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            overlaps[f"{left}_vs_{right}"] = sorted(split_names.get(left, set()) & split_names.get(right, set()))
    report = {
        "theorem_counts": {split: len(names) for split, names in split_names.items()},
        "overlaps": overlaps,
        "has_leakage": any(bool(v) for v in overlaps.values()),
    }
    write_json("outputs/reports/split_leakage_report.json", report)
    return report


def validate_context_coverage(splits: list[str] | None = None) -> dict[str, Any]:
    splits = splits or SPLITS + ["demo"]
    report = {}
    for split in splits:
        path = Path(f"data/processed/{split}/proof_states.parquet")
        if not path.exists():
            continue
        ps = pd.read_parquet(path)
        total = int(len(ps))
        nonempty = int(ps["goal_text"].fillna("").str.strip().astype(bool).sum()) if total else 0
        report[split] = {"proof_states": total, "nonempty_goal_text": nonempty, "coverage": (nonempty / total) if total else 0.0}
    write_json("outputs/reports/context_parse_coverage.json", report)
    return report


def validate_graph(split: str) -> dict[str, Any]:
    nodes_path = Path(f"outputs/graph/{split}/nodes_enriched.parquet")
    edges_path = Path(f"outputs/graph/{split}/edges_enriched.parquet")
    if not nodes_path.exists():
        nodes_path = Path(f"outputs/graph/{split}/nodes.parquet")
        edges_path = Path(f"outputs/graph/{split}/edges.parquet")
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    node_ids = set(nodes["id"])
    missing = edges[~edges["source"].isin(node_ids) | ~edges["target"].isin(node_ids)]
    graph = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=True, create_using=nx.MultiDiGraph())
    for node_id, attrs in nodes.set_index("id").to_dict(orient="index").items():
        graph.add_node(node_id, **attrs)
    report = {
        "split": split,
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "missing_endpoint_count": int(len(missing)),
        "networkx_loadable": True,
    }
    write_json(f"outputs/graph/{split}/graph_validation.json", report)
    return report


def validate_all_graphs(splits: list[str] | None = None) -> dict[str, Any]:
    splits = splits or SPLITS + ["demo"]
    summary = {}
    for split in splits:
        try:
            summary[split] = validate_graph(split)
        except FileNotFoundError:
            continue
    write_json("outputs/reports/graph_validation_summary.json", summary)
    return summary


def run(config_path: str | None = None) -> None:
    validate_split_leakage()
    validate_processed_schemas()
    validate_context_coverage()
    validate_all_graphs()
