from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse

from .utils import SPLITS, load_config, read_json, stable_hash, write_json


MATRIX_STEMS = {
    "proof_state": "proof_state",
    "premise": "premise",
    "theorem": "theorem",
}


def _file_info(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "bytes": int(p.stat().st_size) if p.exists() else 0,
    }


def _parquet_rows(path: str | Path) -> dict[str, Any]:
    info = _file_info(path)
    if not info["exists"]:
        info["rows"] = 0
        return info
    try:
        info["rows"] = int(len(pd.read_parquet(path)))
    except Exception as exc:
        info["rows"] = None
        info["error"] = str(exc)
    return info


def _matrix_info(path: str | Path) -> dict[str, Any]:
    info = _file_info(path)
    if not info["exists"]:
        return {**info, "rows": 0, "dimensions": 0, "nnz": 0}
    try:
        matrix = sparse.load_npz(path)
        info.update(
            {
                "rows": int(matrix.shape[0]),
                "dimensions": int(matrix.shape[1]),
                "nnz": int(matrix.nnz),
                "density": float(matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1])),
            }
        )
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _json_report(path: str | Path) -> dict[str, Any]:
    info = _file_info(path)
    data = read_json(path, None)
    if isinstance(data, dict):
        info["top_level_keys"] = sorted(data.keys())
    return {**info, "data": data}


def _config_hash(config: dict[str, Any]) -> str:
    import json

    return stable_hash(json.dumps(config, sort_keys=True), 16)


def _sample_stage(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    split_counts = manifest.get("split_counts", {}) if isinstance(manifest, dict) else {}
    total_split_rows = 0
    if isinstance(split_counts, dict):
        for value in split_counts.values():
            if isinstance(value, dict):
                total_split_rows += int(value.get("rows") or value.get("theorems") or value.get("theorem_count") or 0)
            else:
                total_split_rows += int(value or 0)
    return {
        "dataset_name": manifest.get("dataset_name") or config.get("dataset_name"),
        "source_kind": manifest.get("source_kind"),
        "sample_plan": manifest.get("sample_plan") or config.get("sample", {}),
        "split_counts": split_counts,
        "total_split_rows": total_split_rows,
        "raw_tables": {split: _parquet_rows(f"data/sample/{split}_rows.parquet") for split in SPLITS},
        "manifest": _file_info("outputs/reports/corpus_manifest.json"),
    }


def _processed_stage() -> dict[str, Any]:
    tables = [
        "theorems",
        "proof_states",
        "premises",
        "positive_edges",
        "negative_edges",
        "proof_techniques",
        "proof_state_features",
        "theorem_features",
    ]
    return {
        split: {table: _parquet_rows(f"data/processed/{split}/{table}.parquet") for table in tables}
        for split in SPLITS + ["demo"]
    }


def _graph_stage() -> dict[str, Any]:
    return {
        split: {
            "nodes": _parquet_rows(f"outputs/graph/{split}/nodes_enriched.parquet"),
            "edges": _parquet_rows(f"outputs/graph/{split}/edges_enriched.parquet"),
            "validation": _json_report(f"outputs/graph/{split}/graph_validation.json"),
        }
        for split in SPLITS + ["demo"]
    }


def _embedding_stage() -> dict[str, Any]:
    embedding_config = read_json("outputs/embeddings/embedding_config.json", {}) or {}
    return {
        "config": embedding_config,
        "splits": {
            split: {
                "metadata": _parquet_rows(f"outputs/embeddings/{split}_embedding_metadata.parquet"),
                "matrices": {
                    name: _matrix_info(f"outputs/embeddings/{split}_{stem}_embeddings.npz")
                    for name, stem in MATRIX_STEMS.items()
                },
            }
            for split in SPLITS + ["demo"]
        },
    }


def _index_stage() -> dict[str, Any]:
    summary = read_json("outputs/indexes/index_summary.json", {}) or {}
    manifests: dict[str, Any] = {}
    for split in SPLITS + ["demo"]:
        manifests[split] = {}
        for name, stem in MATRIX_STEMS.items():
            path = f"outputs/indexes/{split}_{stem}_index_manifest.json"
            report = _json_report(path)
            manifest = report.get("data") if isinstance(report.get("data"), dict) else {}
            index_path = manifest.get("index_path") if isinstance(manifest, dict) else None
            manifests[split][name] = {
                **{key: value for key, value in report.items() if key != "data"},
                "backend": manifest.get("backend"),
                "rows": manifest.get("rows"),
                "dimensions": manifest.get("dimensions"),
                "build_seconds": manifest.get("build_seconds"),
                "index_format": manifest.get("index_format"),
                "index_artifact": _file_info(index_path) if index_path else {"exists": False, "path": "", "bytes": 0},
            }
    return {"summary": summary, "manifests": manifests}


def _benchmark_stage() -> dict[str, Any]:
    data = read_json("outputs/reports/index_benchmark.json", {}) or {}
    entities = data.get("entities", {}) if isinstance(data, dict) else {}
    compact = {}
    for name, row in entities.items():
        top_k = row.get("top_k")
        compact[name] = {
            "backend": row.get("backend"),
            "rows": row.get("rows"),
            "dimensions": row.get("dimensions"),
            "query_count": row.get("query_count"),
            "indexed_available": row.get("indexed_available"),
            "exact_ms_per_query": row.get("exact_ms_per_query"),
            "indexed_ms_per_query": row.get("indexed_ms_per_query"),
            "speedup_vs_exact": row.get("speedup_vs_exact"),
            "recall_vs_exact": row.get(f"recall_at_{top_k}_vs_exact") if top_k else None,
            "index_build_seconds": row.get("index_build_seconds"),
            "indexed_error": row.get("indexed_error"),
        }
    return {"path": "outputs/reports/index_benchmark.json", "exists": Path("outputs/reports/index_benchmark.json").exists(), "entities": compact}


def _timing_stage() -> dict[str, Any]:
    data = read_json("outputs/reports/pipeline_run_timings.json", {}) or {}
    stages = data.get("stages", []) if isinstance(data, dict) else []
    return {
        "path": "outputs/reports/pipeline_run_timings.json",
        "exists": Path("outputs/reports/pipeline_run_timings.json").exists(),
        "passed": data.get("passed"),
        "total_seconds": data.get("total_seconds"),
        "stage_count": data.get("stage_count"),
        "slowest_stages": data.get("slowest_stages", []),
        "stage_seconds": {row.get("name"): row.get("seconds") for row in stages if isinstance(row, dict) and row.get("name")},
    }


def _evaluation_stage() -> dict[str, Any]:
    metrics = read_json("outputs/reports/metrics.json", {}) or {}
    test_set_evaluation = read_json("outputs/reports/test_set_evaluation.json", {}) or {}
    selected_keys = [
        key
        for key in sorted(metrics)
        if key.lower().startswith(("recall", "mrr", "map", "ndcg", "theorem_retrieval"))
    ]
    return {
        "metrics_path": "outputs/reports/metrics.json",
        "metrics_exists": Path("outputs/reports/metrics.json").exists(),
        "selected_metrics": {key: metrics.get(key) for key in selected_keys},
        "test_set_evaluation_path": "outputs/reports/test_set_evaluation.json",
        "test_set_evaluation_exists": Path("outputs/reports/test_set_evaluation.json").exists(),
        "test_set_evaluation": {
            "task": test_set_evaluation.get("task"),
            "candidate_pool": test_set_evaluation.get("candidate_pool"),
            "label_policy": test_set_evaluation.get("label_policy"),
            "evaluation_scope": test_set_evaluation.get("evaluation_scope", {}),
            "top_k": test_set_evaluation.get("top_k"),
            "test_metrics": {
                "proof_state_retrieval": test_set_evaluation.get("test", {}).get("proof_state_retrieval", {}).get("metrics", {}),
                "theorem_retrieval": test_set_evaluation.get("test", {}).get("theorem_retrieval", {}).get("metrics", {}),
            },
            "validation_metrics": {
                "proof_state_retrieval": test_set_evaluation.get("validation", {}).get("proof_state_retrieval", {}).get("metrics", {}),
                "theorem_retrieval": test_set_evaluation.get("validation", {}).get("theorem_retrieval", {}).get("metrics", {}),
            },
        },
        "ranker_validation": read_json("outputs/reports/ranker_validation_metrics.json", {}) or {},
        "difficulty_estimator": read_json("outputs/reports/difficulty_estimator_metrics.json", {}) or {},
    }


def _readiness_stage() -> dict[str, Any]:
    compatibility = read_json("outputs/reports/artifact_compatibility_report.json", {}) or {}
    premise_trace = read_json("outputs/reports/premise_trace_supervision_report.json", {}) or {}
    return {
        "artifact_compatibility": {
            "path": "outputs/reports/artifact_compatibility_report.json",
            "exists": Path("outputs/reports/artifact_compatibility_report.json").exists(),
            "passed": compatibility.get("passed"),
            "warnings": compatibility.get("warnings", []),
            "failures": compatibility.get("failures", []),
            "data_supervision": compatibility.get("data_supervision"),
        },
        "premise_trace_supervision": {
            "path": "outputs/reports/premise_trace_supervision_report.json",
            "exists": Path("outputs/reports/premise_trace_supervision_report.json").exists(),
            "current_artifact_supervision": premise_trace.get("current_artifact_supervision", {}),
            "scope": premise_trace.get("scope"),
        },
    }


def _scale_profile(config: dict[str, Any], sample: dict[str, Any], index: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    sample_plan = sample.get("sample_plan", {}) if isinstance(sample.get("sample_plan"), dict) else {}
    config_sample = config.get("sample", {}) or {}
    requested_theorems = int(
        sample_plan.get("target_theorems")
        or sample_plan.get("total_theorems")
        or config_sample.get("total_theorems")
        or 0
    )
    requested_rows = int(
        sample_plan.get("target_rows")
        or sample_plan.get("total_rows")
        or config_sample.get("total_rows")
        or 0
    )
    source_rows = int(sample_plan.get("source_rows") or config_sample.get("hf_source_rows") or requested_rows or 0)
    current_total_split_rows = int(sample.get("total_split_rows", 0) or 0)
    backend = (index.get("summary") or {}).get("backend") or (config.get("index", {}) or {}).get("backend")
    embedding = config.get("embedding", {}) or {}
    premise_supervision = readiness.get("premise_trace_supervision", {}).get("current_artifact_supervision", {})
    return {
        "dataset_name": sample.get("dataset_name"),
        "target_dataset_confirmed": sample.get("dataset_name") == "erbacher/LeanRank-data",
        "requested_theorems": requested_theorems,
        "requested_rows": requested_rows,
        "source_rows": source_rows,
        "current_total_split_rows": current_total_split_rows,
        "scale_bucket": "large" if requested_theorems >= 10000 or source_rows >= 60000 or current_total_split_rows >= 60000 else "demo",
        "index_backend": backend,
        "embedding_backend": embedding.get("backend"),
        "embedding_device": embedding.get("device"),
        "embedding_devices": embedding.get("devices") or [],
        "ann_backend_availability": {
            "hnswlib": importlib.util.find_spec("hnswlib") is not None,
            "faiss": importlib.util.find_spec("faiss") is not None,
            "lancedb": importlib.util.find_spec("lancedb") is not None,
        },
        "leanrank_premise_supervision_ready": bool(
            premise_supervision.get("has_positive_edges") and premise_supervision.get("has_negative_candidates")
        ),
    }


def _recommendations(config: dict[str, Any], sample: dict[str, Any], benchmark: dict[str, Any], readiness: dict[str, Any], scale: dict[str, Any]) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if sample.get("dataset_name") != "erbacher/LeanRank-data":
        recommendations.append(
            {
                "priority": "high",
                "area": "data_source",
                "recommendation": "Use dataset_name: erbacher/LeanRank-data for the default large-scale run.",
            }
        )
    if int(scale.get("requested_theorems") or 0) < 10000:
        recommendations.append(
            {
                "priority": "medium",
                "area": "scale",
                "recommendation": "Increase sample.total_theorems to at least 10000 for a larger LeanRank-data profile.",
            }
        )
    backend = str(scale.get("index_backend") or "")
    rows = max([int(row.get("rows") or 0) for row in benchmark.get("entities", {}).values()] or [0])
    if backend == "sklearn" and rows >= 5000:
        ann_available = scale.get("ann_backend_availability", {}) or {}
        available = [name for name, enabled in ann_available.items() if enabled]
        recommendations.append(
            {
                "priority": "high",
                "area": "indexing",
                "recommendation": (
                    f"Switch index.backend to an ANN backend before scaling substantially beyond the current sklearn baseline. "
                    f"Currently available ANN backends: {available or 'none installed'}; install `pip install -e '.[ann]'` for hnswlib or `pip install -e '.[faiss]'` for FAISS."
                ),
            }
        )
    if not any(row.get("indexed_available") for row in benchmark.get("entities", {}).values()):
        recommendations.append(
            {
                "priority": "high",
                "area": "indexing",
                "recommendation": "Rebuild and benchmark indexes; indexed retrieval is not available in the current benchmark report.",
            }
        )
    if (config.get("embedding", {}) or {}).get("backend") == "sentence_transformers" and (config.get("embedding", {}) or {}).get("device") != "cuda":
        recommendations.append(
            {
                "priority": "medium",
                "area": "embeddings",
                "recommendation": "Use embedding.device: cuda and a larger batch size for the large LeanRank-data embedding pass when GPU is available.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "priority": "low",
                "area": "monitoring",
                "recommendation": "Keep this profile report as the baseline and compare it after every larger LeanRank-data refresh.",
            }
        )
    return recommendations


def build_report(config_path: str = "configs/proofatlas.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    manifest = read_json("outputs/reports/corpus_manifest.json", {}) or {}
    sample = _sample_stage(manifest, config)
    processed = _processed_stage()
    graph = _graph_stage()
    embeddings = _embedding_stage()
    index = _index_stage()
    benchmark = _benchmark_stage()
    timings = _timing_stage()
    evaluation = _evaluation_stage()
    readiness = _readiness_stage()
    scale = _scale_profile(config, sample, index, readiness)
    recommendations = _recommendations(config, sample, benchmark, readiness, scale)
    return {
        "config_path": config_path,
        "config_hash": _config_hash(config),
        "project_name": config.get("project_name"),
        "dataset_name": sample.get("dataset_name"),
        "source_kind": sample.get("source_kind"),
        "stages": {
            "sample": sample,
            "processed": processed,
            "graph": graph,
            "embeddings": embeddings,
            "indexes": index,
            "benchmark": benchmark,
            "timings": timings,
            "evaluation": evaluation,
            "readiness": readiness,
        },
        "scale_profile": scale,
        "recommendations": recommendations,
    }


def run(config_path: str = "configs/proofatlas.yaml", output_path: str = "outputs/reports/pipeline_performance_report.json") -> dict[str, Any]:
    report = build_report(config_path)
    write_json(output_path, report)
    return report
