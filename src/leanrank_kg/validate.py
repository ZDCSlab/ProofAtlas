from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import networkx as nx
import pandas as pd
from scipy import sparse

from .query import NewTheoremQuery
from .utils import ROOT, SPLITS, load_config, read_json, stable_hash, write_json

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


def validate_processed_schemas(splits: list[str] | None = None, max_rows_per_table: int | None = None) -> dict[str, Any]:
    splits = splits or SPLITS + ["demo"]
    summary: dict[str, Any] = {"tables": {}, "error_count": 0, "errors": [], "max_rows_per_table": max_rows_per_table}
    schemas = {}
    validators = {}
    for table, schema_name in SCHEMA_MAP.items():
        schema_path = Path("schemas") / schema_name
        if not schema_path.exists():
            schema_path = ROOT / "schemas" / schema_name
        with open(schema_path, "r", encoding="utf-8") as fh:
            schemas[table] = json.load(fh)
        validators[table] = jsonschema.Draft7Validator(schemas[table])
    for split in splits:
        for table, schema in schemas.items():
            path = Path(f"data/processed/{split}/{table}.parquet")
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            sampled = df if max_rows_per_table is None or len(df) <= max_rows_per_table else df.head(max_rows_per_table)
            key = f"{split}/{table}"
            summary["tables"][key] = {
                "rows": int(len(df)),
                "sampled_rows": int(len(sampled)),
                "validated_all_rows": int(len(sampled)) == int(len(df)),
                "valid_rows": 0,
            }
            validator = validators[table]
            for idx, row in enumerate(sampled.to_dict(orient="records")):
                record = {k: _json_ready(v) for k, v in row.items()}
                try:
                    validator.validate(record)
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


def _raw_statement_lookup(split: str) -> dict[str, str]:
    path = Path(f"data/sample/{split}_rows.parquet")
    if not path.exists():
        return {}
    raw = pd.read_parquet(path)
    if "statement" not in raw.columns:
        return {}
    return {
        str(row["full_name"]): str(row.get("statement") or "")
        for row in raw[["full_name", "statement"]].dropna(subset=["full_name"]).to_dict(orient="records")
        if str(row.get("statement") or "").strip()
    }


def validate_theorem_query_parse_coverage(splits: list[str] | None = None) -> dict[str, Any]:
    splits = splits or SPLITS + ["demo"]
    report: dict[str, Any] = {}
    for split in splits:
        theorem_path = Path(f"data/processed/{split}/theorems.parquet")
        proof_state_path = Path(f"data/processed/{split}/proof_states.parquet")
        if not theorem_path.exists() or not proof_state_path.exists():
            continue
        theorems = pd.read_parquet(theorem_path)
        proof_states = pd.read_parquet(proof_state_path)
        first_goal = (
            proof_states.sort_values(["theorem_id", "tactic_idx"])
            .drop_duplicates("theorem_id")
            .set_index("theorem_id")["goal_text"]
            .to_dict()
            if not proof_states.empty
            else {}
        )
        raw_statements = _raw_statement_lookup(split)
        rows = []
        failures = []
        for row in theorems.to_dict(orient="records"):
            full_name = str(row.get("full_name", ""))
            theorem_id = str(row.get("id", ""))
            statement = raw_statements.get(full_name) or f"theorem {full_name.replace('.', '_')} : {first_goal.get(theorem_id, '')}"
            try:
                query = NewTheoremQuery.from_text(statement, full_name=full_name, domain_hint=row.get("domain_tag"), file_path=row.get("file_path"))
                rows.append(
                    {
                        "theorem_id": theorem_id,
                        "has_goal": bool(query.goal_text),
                        "binder_count": int(query.parsed_feature_summary.get("binder_count", 0)),
                        "typeclass_hint_count": len(query.typeclass_hints),
                        "conclusion_symbol_count": len(query.conclusion_symbols),
                        "operator_symbol_count": len(query.operator_symbols),
                        "sort_symbol_count": len(query.sort_symbols),
                        "namespace_hint_count": len(query.namespace_hints),
                        "has_normalized_goal": bool(query.normalized_goal_text),
                    }
                )
            except Exception as exc:
                failures.append({"theorem_id": theorem_id, "full_name": full_name, "error": str(exc)})
        frame = pd.DataFrame(rows)
        total = int(len(theorems))
        parsed = int(len(frame))
        report[split] = {
            "theorems": total,
            "parsed": parsed,
            "parse_coverage": parsed / total if total else 0.0,
            "goal_coverage": float(frame["has_goal"].mean()) if not frame.empty else 0.0,
            "normalized_goal_coverage": float(frame["has_normalized_goal"].mean()) if not frame.empty else 0.0,
            "avg_binder_count": float(frame["binder_count"].mean()) if not frame.empty else 0.0,
            "avg_typeclass_hint_count": float(frame["typeclass_hint_count"].mean()) if not frame.empty else 0.0,
            "avg_conclusion_symbol_count": float(frame["conclusion_symbol_count"].mean()) if not frame.empty else 0.0,
            "avg_operator_symbol_count": float(frame["operator_symbol_count"].mean()) if not frame.empty else 0.0,
            "avg_sort_symbol_count": float(frame["sort_symbol_count"].mean()) if not frame.empty else 0.0,
            "avg_namespace_hint_count": float(frame["namespace_hint_count"].mean()) if not frame.empty else 0.0,
            "failure_count": len(failures),
            "failures": failures[:50],
        }
    write_json("outputs/reports/theorem_query_parse_coverage.json", report)
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


def _config_hash(config_path: str | None) -> str | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    config = load_config(path)
    return stable_hash(json.dumps(config, sort_keys=True), 16)


def _processed_counts(split: str) -> dict[str, int] | None:
    theorem_path = Path(f"data/processed/{split}/theorems.parquet")
    ps_path = Path(f"data/processed/{split}/proof_states.parquet")
    premise_path = Path(f"data/processed/{split}/premises.parquet")
    if not theorem_path.exists() or not ps_path.exists() or not premise_path.exists():
        return None
    return {
        "theorems": int(len(pd.read_parquet(theorem_path))),
        "proof_states": int(len(pd.read_parquet(ps_path))),
        "premises": int(len(pd.read_parquet(premise_path))),
    }


def _embedding_counts(split: str) -> dict[str, Any]:
    meta_path = Path(f"outputs/embeddings/{split}_embedding_metadata.parquet")
    result: dict[str, Any] = {"metadata_exists": meta_path.exists(), "metadata_rows": 0, "matrices": {}, "metadata_entity_counts": {}}
    if not meta_path.exists():
        return result
    meta = pd.read_parquet(meta_path)
    result["metadata_rows"] = int(len(meta))
    result["metadata_entity_counts"] = meta["entity_type"].value_counts().to_dict() if "entity_type" in meta.columns else {}
    for entity, stem in [("ProofState", "proof_state"), ("Premise", "premise"), ("Theorem", "theorem")]:
        path = Path(f"outputs/embeddings/{split}_{stem}_embeddings.npz")
        if not path.exists():
            result["matrices"][stem] = {"exists": False}
            continue
        matrix = sparse.load_npz(path)
        expected = int(result["metadata_entity_counts"].get(entity, 0))
        result["matrices"][stem] = {
            "exists": True,
            "rows": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
            "metadata_rows": expected,
            "matches_metadata": int(matrix.shape[0]) == expected,
        }
    return result


def _index_compatibility(split: str, entity_type: str, embedding_config_hash: str | None, corpus_info: dict[str, Any] | None = None) -> dict[str, Any]:
    stem = entity_type.lower()
    manifest_path = Path(f"outputs/indexes/{split}_{stem}_index_manifest.json")
    metadata_path = Path(f"outputs/indexes/{split}_{stem}_index_metadata.parquet")
    result = {
        "manifest_exists": manifest_path.exists(),
        "metadata_exists": metadata_path.exists(),
        "valid": False,
    }
    if not manifest_path.exists() or not metadata_path.exists():
        return result
    manifest = read_json(manifest_path, {}) or {}
    corpus_info = corpus_info or {}
    metadata = pd.read_parquet(metadata_path)
    raw_index_path = manifest.get("index_path") or ""
    index_path = Path(raw_index_path) if raw_index_path else None
    index_built = manifest.get("index_built", True)
    result.update(
        {
            "backend": manifest.get("backend"),
            "index_built": index_built,
            "index_path": str(index_path) if index_path else "",
            "index_exists": bool(index_path and index_path.exists()),
            "manifest_rows": int(manifest.get("rows", -1)),
            "metadata_rows": int(len(metadata)),
            "embedding_config_hash": manifest.get("embedding_config_hash"),
            "expected_embedding_config_hash": embedding_config_hash,
            "corpus_version": manifest.get("corpus_version"),
            "expected_corpus_version": corpus_info.get("corpus_version"),
            "extraction_config_hash": manifest.get("extraction_config_hash"),
            "expected_extraction_config_hash": corpus_info.get("extraction_config_hash"),
        }
    )
    has_compatible_empty_index = (
        index_built is False
        and int(manifest.get("rows", -1)) == 0
        and int(len(metadata)) == 0
    )
    result["valid"] = (
        (bool(index_path and index_path.exists()) or has_compatible_empty_index)
        and int(manifest.get("rows", -1)) == int(len(metadata))
        and manifest.get("split") == split
        and manifest.get("entity_type") == entity_type
        and (embedding_config_hash is None or manifest.get("embedding_config_hash") == embedding_config_hash)
        and (not corpus_info.get("corpus_version") or manifest.get("corpus_version") == corpus_info.get("corpus_version"))
        and (
            not corpus_info.get("extraction_config_hash")
            or manifest.get("extraction_config_hash") == corpus_info.get("extraction_config_hash")
        )
    )
    return result


def _data_supervision_from_manifest(corpus_manifest: dict[str, Any]) -> dict[str, Any]:
    profile = corpus_manifest.get("data_supervision")
    if isinstance(profile, dict) and profile.get("kind"):
        return profile
    source_kind = corpus_manifest.get("source_kind")
    if source_kind == "huggingface":
        return {
            "kind": "leanrank_proof_state_rows",
            "proof_trace_level": "proof_state_rows",
            "has_real_mathlib_source": True,
            "has_tactic_states": True,
            "has_true_positive_premises": True,
            "has_negative_candidates": True,
            "premise_label_semantics": "leanrank_positive_and_negative_candidates",
            "suitable_for": {
                "kg_visualization": True,
                "theorem_text_retrieval_demo": True,
                "premise_ranking_training": True,
                "proof_state_pattern_training": True,
            },
        }
    return {
        "kind": "synthetic_demo_rows",
        "proof_trace_level": "synthetic_proof_state_rows",
        "has_real_mathlib_source": False,
        "has_tactic_states": True,
        "has_true_positive_premises": True,
        "has_negative_candidates": True,
        "premise_label_semantics": "synthetic_demo_labels",
        "suitable_for": {
            "kg_visualization": True,
            "theorem_text_retrieval_demo": True,
            "premise_ranking_training": False,
            "proof_state_pattern_training": False,
        },
    }


def validate_artifact_compatibility(config_path: str | None = None) -> dict[str, Any]:
    corpus_manifest = read_json("outputs/reports/corpus_manifest.json", {}) or {}
    corpus_info = corpus_manifest.get("corpus", {}) if isinstance(corpus_manifest, dict) else {}
    data_supervision = _data_supervision_from_manifest(corpus_manifest if isinstance(corpus_manifest, dict) else {})
    embedding_config = read_json("outputs/embeddings/embedding_config.json", {}) or {}
    embedding_config_hash = stable_hash(json.dumps(embedding_config, sort_keys=True), 16) if embedding_config else None
    expected_config_hash = _config_hash(config_path)
    report: dict[str, Any] = {
        "config_path": config_path,
        "expected_config_hash": expected_config_hash,
        "manifest_config_hash": corpus_manifest.get("config_hash"),
        "config_hash_matches": expected_config_hash is None or corpus_manifest.get("config_hash") == expected_config_hash,
        "corpus": corpus_manifest.get("corpus", {}),
        "source_kind": corpus_manifest.get("source_kind"),
        "data_supervision": data_supervision,
        "embedding_config_hash": embedding_config_hash,
        "splits": {},
        "passed": True,
        "failures": [],
        "warnings": [],
    }
    if not data_supervision.get("has_real_mathlib_source"):
        report["warnings"].append("corpus_is_not_real_mathlib_source")
    if not data_supervision.get("has_tactic_states"):
        report["warnings"].append("corpus_lacks_tactic_state_supervision")
    if not data_supervision.get("has_true_positive_premises"):
        report["warnings"].append("corpus_lacks_true_positive_premise_labels")
    for split in SPLITS:
        processed = _processed_counts(split)
        embeddings = _embedding_counts(split)
        manifest_split = (corpus_manifest.get("split_counts") or {}).get(split, {})
        split_report = {
            "processed_counts": processed,
            "manifest_split_counts": manifest_split,
            "embeddings": embeddings,
            "indexes": {
                "proof_state": _index_compatibility(split, "ProofState", embedding_config_hash, corpus_info),
                "premise": _index_compatibility(split, "Premise", embedding_config_hash, corpus_info),
                "theorem": _index_compatibility(split, "Theorem", embedding_config_hash, corpus_info),
            },
        }
        split_failures = []
        if processed is None:
            split_failures.append("missing_processed_tables")
        else:
            if manifest_split and int(manifest_split.get("theorems", -1)) != processed["theorems"]:
                split_failures.append("manifest_theorem_count_mismatch")
            matrix_checks = embeddings.get("matrices", {})
            if matrix_checks.get("proof_state", {}).get("rows") != processed["proof_states"]:
                split_failures.append("proof_state_embedding_count_mismatch")
            if matrix_checks.get("premise", {}).get("rows") != processed["premises"]:
                split_failures.append("premise_embedding_count_mismatch")
            if matrix_checks.get("theorem", {}).get("rows") != processed["theorems"]:
                split_failures.append("theorem_embedding_count_mismatch")
        for name, index_report in split_report["indexes"].items():
            if index_report.get("manifest_exists") and not index_report.get("valid"):
                split_failures.append(f"{name}_index_manifest_mismatch")
        split_report["passed"] = not split_failures
        split_report["failures"] = split_failures
        if split_failures:
            report["failures"].append({split: split_failures})
        report["splits"][split] = split_report
    if not report["config_hash_matches"]:
        report["failures"].append({"config": ["config_hash_mismatch"]})
    report["passed"] = not report["failures"]
    write_json("outputs/reports/artifact_compatibility_report.json", report)
    return report


def run(config_path: str | None = None) -> None:
    validate_split_leakage()
    validate_processed_schemas()
    validate_context_coverage()
    validate_theorem_query_parse_coverage()
    validate_all_graphs()
    validate_artifact_compatibility(config_path)
