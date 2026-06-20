from __future__ import annotations

import importlib.util
import math
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


def _directory_size(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.exists():
        return {"path": str(root), "exists": False, "bytes": 0, "file_count": 0}
    total = 0
    file_count = 0
    for file_path in root.rglob("*"):
        if file_path.is_file():
            total += int(file_path.stat().st_size)
            file_count += 1
    return {"path": str(root), "exists": True, "bytes": total, "file_count": file_count}


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


def _timing_stage(expected_config_hash: str | None = None) -> dict[str, Any]:
    data = read_json("outputs/reports/pipeline_run_timings.json", {}) or {}
    stages = data.get("stages", []) if isinstance(data, dict) else []
    timing_config_hash = data.get("config_hash")
    config_matches = timing_config_hash == expected_config_hash if timing_config_hash and expected_config_hash else None
    executed_stage_count = sum(1 for row in stages if isinstance(row, dict) and row.get("status") == "passed")
    skipped_stage_count = sum(1 for row in stages if isinstance(row, dict) and row.get("status") == "skipped")
    return {
        "path": "outputs/reports/pipeline_run_timings.json",
        "exists": Path("outputs/reports/pipeline_run_timings.json").exists(),
        "config_path": data.get("config_path"),
        "config_hash": timing_config_hash,
        "expected_config_hash": expected_config_hash,
        "config_matches_current": config_matches,
        "generated_at": data.get("generated_at"),
        "passed": data.get("passed"),
        "total_seconds": data.get("total_seconds"),
        "stage_count": data.get("stage_count"),
        "executed_stage_count": executed_stage_count,
        "skipped_stage_count": skipped_stage_count,
        "has_skipped_stages": skipped_stage_count > 0,
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
    evaluation_scope = test_set_evaluation.get("evaluation_scope", {}) if isinstance(test_set_evaluation, dict) else {}
    substage_timings = evaluation_scope.get("substage_timings", []) if isinstance(evaluation_scope, dict) else []
    if not isinstance(substage_timings, list):
        substage_timings = []
    timed_substages = [
        row
        for row in substage_timings
        if isinstance(row, dict) and row.get("name") and row.get("seconds") is not None
    ]
    total_substage_seconds = sum(float(row.get("seconds") or 0.0) for row in timed_substages)
    proof_test_metrics = test_set_evaluation.get("test", {}).get("proof_state_retrieval", {}).get("metrics", {})
    theorem_test_metrics = test_set_evaluation.get("test", {}).get("theorem_retrieval", {}).get("metrics", {})
    reranked_proof_state = test_set_evaluation.get("test", {}).get("proof_state_reranked_retrieval", {})
    proof_state_test_total = int(_parquet_rows("data/processed/test/proof_states.parquet").get("rows") or 0)
    theorem_test_total = int(_parquet_rows("data/processed/test/theorems.parquet").get("rows") or 0)
    proof_state_test_evaluated = proof_test_metrics.get("evaluated_queries")
    theorem_test_evaluated = theorem_test_metrics.get("theorem_retrieval_evaluated_theorems")
    proof_state_test_limit = (evaluation_scope.get("proof_state_limits") or {}).get("test")
    theorem_test_limit = (evaluation_scope.get("theorem_limits") or {}).get("test")

    def _coverage(evaluated: Any, limit: Any, total: int) -> float | None:
        numerator = evaluated if evaluated is not None else limit
        if numerator is None or total <= 0:
            return None
        return min(float(numerator) / float(total), 1.0)

    evaluation_timing = {
        "total_seconds": evaluation_scope.get("total_seconds"),
        "substage_count": len(timed_substages),
        "substage_seconds_total": total_substage_seconds,
        "slowest_substages": sorted(
            timed_substages,
            key=lambda row: float(row.get("seconds") or 0.0),
            reverse=True,
        )[:8],
    }

    return {
        "metrics_path": "outputs/reports/metrics.json",
        "metrics_exists": Path("outputs/reports/metrics.json").exists(),
        "selected_metrics": {key: metrics.get(key) for key in selected_keys},
        "evaluation_timing": evaluation_timing,
        "test_set_evaluation_path": "outputs/reports/test_set_evaluation.json",
        "test_set_evaluation_exists": Path("outputs/reports/test_set_evaluation.json").exists(),
        "test_set_evaluation": {
            "task": test_set_evaluation.get("task"),
            "candidate_pool": test_set_evaluation.get("candidate_pool"),
            "label_policy": test_set_evaluation.get("label_policy"),
            "evaluation_scope": evaluation_scope,
            "top_k": test_set_evaluation.get("top_k"),
            "test_metrics": {
                "proof_state_retrieval": proof_test_metrics,
                "theorem_retrieval": theorem_test_metrics,
            },
            "test": {
                "proof_state_reranked_retrieval": reranked_proof_state,
                "proof_state_query_representation_diagnostic": test_set_evaluation.get("test", {}).get(
                    "proof_state_query_representation_diagnostic",
                    {},
                ),
            },
            "validation": {
                "proof_state_query_representation_diagnostic": test_set_evaluation.get("validation", {}).get(
                    "proof_state_query_representation_diagnostic",
                    {},
                ),
            },
            "validation_metrics": {
                "proof_state_retrieval": test_set_evaluation.get("validation", {}).get("proof_state_retrieval", {}).get("metrics", {}),
                "theorem_retrieval": test_set_evaluation.get("validation", {}).get("theorem_retrieval", {}).get("metrics", {}),
            },
        },
        "held_out_test_coverage": {
            "proof_state_total": proof_state_test_total,
            "proof_state_configured_limit": proof_state_test_limit,
            "proof_state_evaluated_queries": proof_state_test_evaluated,
            "proof_state_coverage_fraction": _coverage(proof_state_test_evaluated, proof_state_test_limit, proof_state_test_total),
            "theorem_total": theorem_test_total,
            "theorem_configured_limit": theorem_test_limit,
            "theorem_evaluated_queries": theorem_test_evaluated,
            "theorem_coverage_fraction": _coverage(theorem_test_evaluated, theorem_test_limit, theorem_test_total),
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


def _sum_matrix_rows(embeddings: dict[str, Any]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_entity: dict[str, int] = {name: 0 for name in MATRIX_STEMS}
    total_rows = 0
    total_bytes = 0
    for split, split_info in (embeddings.get("splits") or {}).items():
        split_rows = 0
        for entity, matrix_info in (split_info.get("matrices") or {}).items():
            rows = int(matrix_info.get("rows") or 0)
            bytes_ = int(matrix_info.get("bytes") or 0)
            split_rows += rows
            by_entity[entity] = by_entity.get(entity, 0) + rows
            total_bytes += bytes_
        by_split[split] = split_rows
        total_rows += split_rows
    return {
        "total_embedding_rows": total_rows,
        "embedding_rows_by_split": by_split,
        "embedding_rows_by_entity": by_entity,
        "embedding_matrix_bytes": total_bytes,
    }


def _throughput_profile(
    sample: dict[str, Any],
    embeddings: dict[str, Any],
    index: dict[str, Any],
    benchmark: dict[str, Any],
    timings: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    matrix_rows = _sum_matrix_rows(embeddings)
    total_seconds = float(timings.get("total_seconds") or 0.0)
    current_rows = int(sample.get("total_split_rows") or 0)
    embedding_rows = int(matrix_rows.get("total_embedding_rows") or 0)
    stage_seconds = timings.get("stage_seconds", {}) if isinstance(timings.get("stage_seconds"), dict) else {}
    current_evaluation_seconds = (
        (evaluation.get("evaluation_timing") or {}).get("total_seconds")
        if isinstance(evaluation, dict)
        else None
    )
    timed_evaluate_seconds = stage_seconds.get("evaluate")
    evaluation_timing_delta = {
        "timed_pipeline_evaluate_seconds": timed_evaluate_seconds,
        "current_evaluation_seconds": current_evaluation_seconds,
        "timed_to_current_ratio": (
            float(timed_evaluate_seconds) / float(current_evaluation_seconds)
            if timed_evaluate_seconds is not None and current_evaluation_seconds and float(current_evaluation_seconds) > 0
            else None
        ),
        "current_faster_than_pipeline_timing": (
            bool(float(current_evaluation_seconds) < float(timed_evaluate_seconds) * 0.75)
            if timed_evaluate_seconds is not None and current_evaluation_seconds is not None
            else None
        ),
    }
    index_build_seconds = 0.0
    for split in (index.get("manifests") or {}).values():
        for manifest in (split or {}).values():
            index_build_seconds += float(manifest.get("build_seconds") or 0.0)
    speedups = [
        float(row.get("speedup_vs_exact"))
        for row in (benchmark.get("entities") or {}).values()
        if row.get("speedup_vs_exact") is not None
    ]
    recalls = [
        float(row.get("recall_vs_exact"))
        for row in (benchmark.get("entities") or {}).values()
        if row.get("recall_vs_exact") is not None
    ]
    slowest = timings.get("slowest_stages", []) if isinstance(timings.get("slowest_stages"), list) else []
    bottleneck = slowest[0].get("name") if slowest and isinstance(slowest[0], dict) else None
    bottleneck_rows = []
    for row in slowest[:5]:
        if not isinstance(row, dict):
            continue
        seconds = float(row.get("seconds") or 0.0)
        bottleneck_rows.append(
            {
                "name": row.get("name"),
                "seconds": seconds,
                "share_of_total": (seconds / total_seconds) if total_seconds > 0 else None,
            }
        )
    top3_seconds = sum(row["seconds"] for row in bottleneck_rows[:3])
    bottleneck_profile = {
        "primary_stage": bottleneck,
        "primary_stage_seconds": bottleneck_rows[0]["seconds"] if bottleneck_rows else None,
        "primary_stage_share_of_total": bottleneck_rows[0]["share_of_total"] if bottleneck_rows else None,
        "top3_stage_seconds": top3_seconds if bottleneck_rows else None,
        "top3_stage_share_of_total": (top3_seconds / total_seconds) if total_seconds > 0 and bottleneck_rows else None,
        "top_stages": bottleneck_rows,
    }
    embed_seconds = stage_seconds.get("embed")
    embed_seconds_f = float(embed_seconds) if embed_seconds is not None else None
    embedding_bottleneck_profile = {
        "embed_stage_seconds": embed_seconds,
        "embed_stage_share_of_total": (embed_seconds_f / total_seconds) if embed_seconds_f is not None and total_seconds > 0 else None,
        "embedding_rows_per_embed_second": (embedding_rows / embed_seconds_f) if embed_seconds_f and embed_seconds_f > 0 else None,
        "embedding_rows_by_entity": matrix_rows.get("embedding_rows_by_entity", {}),
        "embedding_rows_by_split": matrix_rows.get("embedding_rows_by_split", {}),
        "embedding_matrix_bytes": matrix_rows.get("embedding_matrix_bytes"),
    }

    def _per_100k(rows: int) -> float | None:
        if total_seconds <= 0 or rows <= 0:
            return None
        return total_seconds / rows * 100000

    requested_rows = int((sample.get("sample_plan") or {}).get("source_rows") or current_rows or 0)
    scale_factor = (requested_rows / current_rows) if current_rows else None
    estimated_seconds_at_requested_source = (total_seconds * scale_factor) if scale_factor is not None else None
    has_skipped_stages = bool(timings.get("has_skipped_stages"))
    timing_config_matches = timings.get("config_matches_current")
    return {
        **matrix_rows,
        "timing_config_hash": timings.get("config_hash"),
        "timing_config_matches_current": timing_config_matches,
        "timing_generated_at": timings.get("generated_at"),
        "timing_executed_stage_count": timings.get("executed_stage_count"),
        "timing_skipped_stage_count": timings.get("skipped_stage_count"),
        "timing_has_skipped_stages": has_skipped_stages,
        "throughput_basis": "cached_or_partial_pipeline_run" if has_skipped_stages else "executed_pipeline_run",
        "scale_estimate_reliable": bool(timing_config_matches is True and not has_skipped_stages and total_seconds > 0),
        "total_pipeline_seconds": total_seconds,
        "processed_rows_per_second": (current_rows / total_seconds) if total_seconds > 0 else None,
        "embedding_rows_per_second": (embedding_rows / total_seconds) if total_seconds > 0 else None,
        "pipeline_seconds_per_100k_processed_rows": _per_100k(current_rows),
        "pipeline_seconds_per_100k_embedding_rows": _per_100k(embedding_rows),
        "timed_stage_seconds": stage_seconds,
        "evaluation_timing_delta": evaluation_timing_delta,
        "slowest_stage": bottleneck,
        "bottleneck_profile": bottleneck_profile,
        "embedding_bottleneck_profile": embedding_bottleneck_profile,
        "index_build_seconds_total": index_build_seconds,
        "mean_index_speedup_vs_exact": (sum(speedups) / len(speedups)) if speedups else None,
        "min_index_recall_vs_exact": min(recalls) if recalls else None,
        "requested_source_rows": requested_rows,
        "scale_factor_from_current_to_requested_source_rows": scale_factor,
        "estimated_seconds_at_requested_source_rows": estimated_seconds_at_requested_source,
    }


def _retrieval_bottleneck_profile(evaluation: dict[str, Any]) -> dict[str, Any]:
    test_metrics = ((evaluation.get("test_set_evaluation") or {}).get("test_metrics") or {}) if isinstance(evaluation, dict) else {}

    def _profile(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
        def _get(name: str) -> float | None:
            key = f"{prefix}{name}" if prefix else name
            value = metrics.get(key)
            return float(value) if value is not None else None

        recall10 = _get("Recall@10")
        recall100 = _get("Recall@100")
        gap = (recall100 - recall10) if recall10 is not None and recall100 is not None else None
        top10_fraction = (recall10 / recall100) if recall10 is not None and recall100 else None
        if recall100 is None:
            bottleneck = "missing_metrics"
        elif recall100 < 0.5:
            bottleneck = "candidate_generation_or_embeddings"
        elif gap is not None and gap >= 0.1:
            bottleneck = "top10_reranking_or_candidate_ordering"
        else:
            bottleneck = "monitoring"
        return {
            "recall_at_10": recall10,
            "recall_at_100": recall100,
            "top10_to_top100_gap": gap,
            "top10_fraction_of_top100": top10_fraction,
            "primary_accuracy_bottleneck": bottleneck,
        }

    return {
        "proof_state": _profile(test_metrics.get("proof_state_retrieval", {})),
        "theorem": _profile(test_metrics.get("theorem_retrieval", {}), prefix="theorem_retrieval_"),
    }


def _rapid_convergence_profile(
    evaluation: dict[str, Any],
    readiness: dict[str, Any],
    throughput: dict[str, Any],
) -> dict[str, Any]:
    test_eval = evaluation.get("test_set_evaluation", {}) if isinstance(evaluation, dict) else {}
    test = test_eval.get("test", {}) if isinstance(test_eval, dict) else {}
    proof_metrics = test_eval.get("test_metrics", {}).get("proof_state_retrieval", {})
    theorem_metrics = test_eval.get("test_metrics", {}).get("theorem_retrieval", {})
    reranked_metrics = test.get("proof_state_reranked_retrieval", {}).get("metrics", {}) if isinstance(test, dict) else {}
    candidate_k_ablation = test.get("proof_state_reranked_retrieval", {}).get("candidate_k_ablation", []) if isinstance(test, dict) else []
    validation = test_eval.get("validation", {}) if isinstance(test_eval, dict) else {}
    validation_query_representation = (
        validation.get("proof_state_query_representation_diagnostic", {}) if isinstance(validation, dict) else {}
    )
    test_query_representation = (
        test.get("proof_state_query_representation_diagnostic", {}) if isinstance(test, dict) else {}
    )
    retrieval_profile = throughput.get("retrieval_bottleneck_profile", {}) if isinstance(throughput, dict) else {}
    premise_trace = readiness.get("premise_trace_supervision", {}).get("current_artifact_supervision", {}) if isinstance(readiness, dict) else {}
    ranker_validation = evaluation.get("ranker_validation", {}) if isinstance(evaluation, dict) else {}
    feature_groups = (ranker_validation.get("feature_ablation", {}) or {}).get("groups", {}) if isinstance(ranker_validation, dict) else {}

    def _float(value: Any) -> float | None:
        return float(value) if value is not None else None

    def _best_candidate_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
        best: dict[str, Any] = {}
        best_score: float | None = None
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
            score = metrics.get("Recall@10")
            if score is None:
                continue
            score_f = float(score)
            if best_score is None or score_f > best_score:
                best_score = score_f
                best = {"candidate_k": row.get("candidate_k"), "Recall@10": score_f, "MRR": metrics.get("MRR"), "MAP": metrics.get("MAP")}
        return best

    ranked_feature_groups = []
    for name, row in feature_groups.items():
        if not isinstance(row, dict):
            continue
        ranked_feature_groups.append(
            {
                "group": name,
                "delta_without_group": row.get("delta_without_group"),
                "auc_group_only": row.get("auc_group_only"),
                "auc_without_group": row.get("auc_without_group"),
                "columns": row.get("columns", []),
            }
        )
    ranked_feature_groups.sort(key=lambda row: float(row.get("delta_without_group") or 0.0), reverse=True)

    proof_recall10 = _float(proof_metrics.get("Recall@10"))
    proof_recall100 = _float(proof_metrics.get("Recall@100"))
    theorem_recall10 = _float(theorem_metrics.get("theorem_retrieval_Recall@10"))
    theorem_recall100 = _float(theorem_metrics.get("theorem_retrieval_Recall@100"))
    reranked_recall10 = _float(reranked_metrics.get("Recall@10"))
    rerank_delta = (reranked_recall10 - proof_recall10) if reranked_recall10 is not None and proof_recall10 is not None else None

    recommended_sequence: list[dict[str, Any]] = []
    proof_bottleneck = (retrieval_profile.get("proof_state") or {}).get("primary_accuracy_bottleneck")
    theorem_bottleneck = (retrieval_profile.get("theorem") or {}).get("primary_accuracy_bottleneck")
    if proof_bottleneck == "candidate_generation_or_embeddings":
        validation_best = validation_query_representation.get("best_variant_by_recall")
        recommended_sequence.append(
            {
                "priority": 1,
                "area": "proof_state_query_and_embedding",
                "target_metric": "proof_state Recall@100",
                "current_value": proof_recall100,
                "reason": (
                    "Proof-state gold premises are often absent from the top-100 candidate pool, so top-k reranking cannot recover them."
                    + (f" Validation query diagnostic currently favors `{validation_best}`." if validation_best else "")
                ),
            }
        )
    if theorem_bottleneck == "top10_reranking_or_candidate_ordering":
        recommended_sequence.append(
            {
                "priority": 2,
                "area": "theorem_level_reranking",
                "target_metric": "theorem_retrieval Recall@10",
                "current_value": theorem_recall10,
                "reason": "Theorem-level Recall@100 is substantially higher than Recall@10, leaving useful headroom for ordering candidates already in the pool.",
            }
        )
    if ranked_feature_groups:
        strongest = ranked_feature_groups[0]
        recommended_sequence.append(
            {
                "priority": 3,
                "area": "ranker_feature_iteration",
                "target_metric": "validation/test Recall@10 and MAP",
                "current_value": strongest.get("delta_without_group"),
                "reason": f"Ranker ablation says `{strongest.get('group')}` is the strongest currently measured feature group by delta_without_group.",
            }
        )
    if premise_trace.get("has_negative_candidates") and premise_trace.get("has_positive_edges"):
        recommended_sequence.append(
            {
                "priority": 4,
                "area": "hard_negative_training",
                "target_metric": "MRR/MAP after reranking",
                "current_value": premise_trace.get("negative_to_positive_edge_ratio"),
                "reason": "LeanRank-data already provides positive premises and hard negative candidates, so training/evaluation changes can reuse existing labels without extracting new data.",
            }
        )

    return {
        "accuracy_snapshot": {
            "proof_state_recall_at_10": proof_recall10,
            "proof_state_recall_at_100": proof_recall100,
            "theorem_recall_at_10": theorem_recall10,
            "theorem_recall_at_100": theorem_recall100,
            "reranked_proof_state_recall_at_10": reranked_recall10,
            "reranked_minus_embedding_recall_at_10": rerank_delta,
        },
        "headroom": {
            "proof_state_missing_from_top100": (1.0 - proof_recall100) if proof_recall100 is not None else None,
            "proof_state_top10_to_top100_gap": (proof_recall100 - proof_recall10) if proof_recall10 is not None and proof_recall100 is not None else None,
            "theorem_missing_from_top100": (1.0 - theorem_recall100) if theorem_recall100 is not None else None,
            "theorem_top10_to_top100_gap": (theorem_recall100 - theorem_recall10) if theorem_recall10 is not None and theorem_recall100 is not None else None,
        },
        "rerank_candidate_depth": {
            "best_by_recall_at_10": _best_candidate_k(candidate_k_ablation),
            "evaluated_candidate_k_values": [row.get("candidate_k") for row in candidate_k_ablation if isinstance(row, dict)],
        },
        "query_representation_diagnostic": {
            "validation": {
                "evaluated_queries": validation_query_representation.get("evaluated_queries"),
                "selection_metric": validation_query_representation.get("selection_metric"),
                "best_variant_by_recall": validation_query_representation.get("best_variant_by_recall"),
            },
            "test": {
                "evaluated_queries": test_query_representation.get("evaluated_queries"),
                "selection_metric": test_query_representation.get("selection_metric"),
                "best_variant_by_recall": test_query_representation.get("best_variant_by_recall"),
            },
            "validation_test_best_variant_match": (
                validation_query_representation.get("best_variant_by_recall")
                == test_query_representation.get("best_variant_by_recall")
                if validation_query_representation.get("best_variant_by_recall") and test_query_representation.get("best_variant_by_recall")
                else None
            ),
        },
        "strongest_ranker_feature_groups": ranked_feature_groups[:5],
        "label_supervision": {
            "has_positive_edges": premise_trace.get("has_positive_edges"),
            "has_negative_candidates": premise_trace.get("has_negative_candidates"),
            "negative_to_positive_edge_ratio": premise_trace.get("negative_to_positive_edge_ratio"),
        },
        "recommended_sequence": recommended_sequence,
    }


def _metric_uncertainty_profile(evaluation: dict[str, Any]) -> dict[str, Any]:
    test_metrics = ((evaluation.get("test_set_evaluation") or {}).get("test_metrics") or {}) if isinstance(evaluation, dict) else {}
    proof_metrics = test_metrics.get("proof_state_retrieval", {}) if isinstance(test_metrics, dict) else {}
    theorem_metrics = test_metrics.get("theorem_retrieval", {}) if isinstance(test_metrics, dict) else {}

    def _bounded_normal_ci(value: Any, n: Any) -> dict[str, Any]:
        if value is None or n is None:
            return {"value": value, "n": n, "standard_error": None, "ci95_low": None, "ci95_high": None, "ci95_half_width": None}
        try:
            value_f = float(value)
            n_i = int(n)
        except (TypeError, ValueError):
            return {"value": value, "n": n, "standard_error": None, "ci95_low": None, "ci95_high": None, "ci95_half_width": None}
        if n_i <= 0:
            return {"value": value_f, "n": n_i, "standard_error": None, "ci95_low": None, "ci95_high": None, "ci95_half_width": None}
        bounded = min(max(value_f, 0.0), 1.0)
        standard_error = math.sqrt((bounded * (1.0 - bounded)) / n_i)
        half_width = 1.96 * standard_error
        return {
            "value": value_f,
            "n": n_i,
            "standard_error": standard_error,
            "ci95_low": max(0.0, value_f - half_width),
            "ci95_high": min(1.0, value_f + half_width),
            "ci95_half_width": half_width,
        }

    proof_n = proof_metrics.get("evaluated_queries") or proof_metrics.get("evaluated_retrievable_queries")
    theorem_n = theorem_metrics.get("theorem_retrieval_evaluated_theorems") or theorem_metrics.get("theorem_retrieval_evaluated_queries")
    return {
        "method": "bounded_normal_approximation_for_aggregate_retrieval_metrics",
        "confidence_level": 0.95,
        "note": "Intervals are approximate diagnostics for bounded aggregate retrieval metrics, not a replacement for paired bootstrap comparisons.",
        "proof_state": {
            metric: _bounded_normal_ci(proof_metrics.get(metric), proof_n)
            for metric in ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Recall@100", "MRR", "MAP", "nDCG@10"]
        },
        "theorem": {
            metric: _bounded_normal_ci(theorem_metrics.get(metric), theorem_n)
            for metric in [
                "theorem_retrieval_Recall@1",
                "theorem_retrieval_Recall@5",
                "theorem_retrieval_Recall@10",
                "theorem_retrieval_Recall@50",
                "theorem_retrieval_Recall@100",
                "theorem_retrieval_MRR",
                "theorem_retrieval_MAP",
                "theorem_retrieval_nDCG@10",
            ]
        },
    }


def _rerank_evaluation_cost_profile(evaluation: dict[str, Any]) -> dict[str, Any]:
    test_eval = evaluation.get("test_set_evaluation", {}) if isinstance(evaluation, dict) else {}
    scope = test_eval.get("evaluation_scope", {}) if isinstance(test_eval, dict) else {}
    substage_timings = scope.get("substage_timings", []) if isinstance(scope, dict) else []
    if not isinstance(substage_timings, list):
        substage_timings = []
    rerank_stage = next(
        (
            row
            for row in substage_timings
            if isinstance(row, dict) and row.get("name") == "test_reranked_proof_state_retrieval"
        ),
        {},
    )
    batched_stage = next(
        (
            row
            for row in substage_timings
            if isinstance(row, dict) and row.get("name") == "test_proof_state_retrieval"
        ),
        {},
    )
    proof_metrics = ((test_eval.get("test_metrics") or {}).get("proof_state_retrieval") or {}) if isinstance(test_eval, dict) else {}
    reranked = (((test_eval.get("test") or {}).get("proof_state_reranked_retrieval") or {}) if isinstance(test_eval, dict) else {})
    rerank_metrics = reranked.get("metrics", {}) if isinstance(reranked, dict) else {}
    backend_info = reranked.get("backend_info", {}) if isinstance(reranked, dict) else {}

    rerank_queries = int(rerank_stage.get("evaluated_queries") or backend_info.get("evaluated_queries") or rerank_metrics.get("evaluated_queries") or 0)
    full_queries = int(proof_metrics.get("evaluated_queries") or 0)
    rerank_seconds = float(rerank_stage.get("seconds") or 0.0)
    batched_seconds = float(batched_stage.get("seconds") or 0.0)
    batched_queries = int(batched_stage.get("evaluated_queries") or proof_metrics.get("evaluated_queries") or 0)
    rerank_seconds_per_query = (rerank_seconds / rerank_queries) if rerank_queries > 0 else None
    batched_seconds_per_query = (batched_seconds / batched_queries) if batched_queries > 0 else None
    projected_full_seconds = (rerank_seconds_per_query * full_queries) if rerank_seconds_per_query is not None and full_queries > 0 else None
    relative_cost = (
        rerank_seconds_per_query / batched_seconds_per_query
        if rerank_seconds_per_query is not None and batched_seconds_per_query not in (None, 0.0)
        else None
    )
    sampled_fraction = (rerank_queries / full_queries) if full_queries > 0 else None
    recall_delta = None
    if rerank_metrics.get("Recall@10") is not None and proof_metrics.get("Recall@10") is not None:
        recall_delta = float(rerank_metrics.get("Recall@10")) - float(proof_metrics.get("Recall@10"))
    return {
        "method": "project_full_rerank_cost_from_user_facing_sampled_rerank_diagnostic",
        "rerank_stage_name": rerank_stage.get("name"),
        "rerank_backend": backend_info.get("actual_backend") or rerank_stage.get("actual_backend"),
        "candidate_k": backend_info.get("candidate_k") or rerank_stage.get("candidate_k"),
        "candidate_k_values": backend_info.get("candidate_k_values"),
        "sampled_rerank_queries": rerank_queries,
        "full_proof_state_queries": full_queries,
        "sampled_fraction_of_full_proof_state_eval": sampled_fraction,
        "rerank_seconds": rerank_seconds,
        "rerank_seconds_per_query": rerank_seconds_per_query,
        "batched_embedding_seconds": batched_seconds,
        "batched_embedding_seconds_per_query": batched_seconds_per_query,
        "rerank_to_batched_seconds_per_query_ratio": relative_cost,
        "projected_full_rerank_seconds": projected_full_seconds,
        "projected_full_rerank_minutes": (projected_full_seconds / 60.0) if projected_full_seconds is not None else None,
        "reranked_recall_at_10": rerank_metrics.get("Recall@10"),
        "batched_embedding_recall_at_10": proof_metrics.get("Recall@10"),
        "sampled_rerank_recall_at_10_delta": recall_delta,
        "policy": "keep reranked proof-state evaluation sampled for development and use full batched embedding evaluation for final held-out coverage",
    }


def _refresh_reuse_profile(
    config: dict[str, Any],
    sample: dict[str, Any],
    embeddings: dict[str, Any],
    index: dict[str, Any],
) -> dict[str, Any]:
    embedding_rows = 0
    for split in (embeddings.get("splits") or {}).values():
        for matrix in ((split or {}).get("matrices") or {}).values():
            embedding_rows += int(matrix.get("rows") or 0)
    indexed_entities = []
    for split, entities in (index.get("manifests") or {}).items():
        for entity, manifest in (entities or {}).items():
            if manifest.get("exists") and manifest.get("rows"):
                indexed_entities.append(f"{split}:{entity}")
    ranker_artifact = _file_info("outputs/models/premise_ranker.joblib")
    difficulty_artifact = _file_info("outputs/models/difficulty_estimator.joblib")
    has_embedding_cache = embedding_rows > 0
    has_index_cache = bool(indexed_entities)
    has_ranker = bool(ranker_artifact.get("exists"))

    scenarios = [
        {
            "scenario": "report_or_homepage_refresh",
            "rerun_embedding": False,
            "rerun_ranker_training": False,
            "rerun_evaluation": False,
            "commands": ["leanrank-kg profile-pipeline", "leanrank-kg build-experiment-report", "leanrank-kg build-homepage", "leanrank-kg audit"],
            "reason": "Presentation-only changes can reuse the committed LeanRank-data artifacts and regenerate reports/homepage.",
        },
        {
            "scenario": "retrieval_or_ranking_code_change",
            "rerun_embedding": False,
            "rerun_ranker_training": False,
            "rerun_evaluation": True,
            "commands": ["leanrank-kg evaluate", "leanrank-kg profile-pipeline", "leanrank-kg build-experiment-report", "leanrank-kg build-homepage"],
            "reason": "Ranking logic changes require held-out metrics, but embeddings and indexes remain reusable when query/embedding configuration and data splits are unchanged.",
        },
        {
            "scenario": "ranker_feature_or_label_change",
            "rerun_embedding": False,
            "rerun_ranker_training": True,
            "rerun_evaluation": True,
            "commands": ["leanrank-kg train-ranker", "leanrank-kg evaluate", "leanrank-kg profile-pipeline", "leanrank-kg build-experiment-report"],
            "reason": "Feature or supervision changes invalidate the learned ranker; they do not require re-embedding unless the embedding text/model changed.",
        },
        {
            "scenario": "embedding_model_or_text_change",
            "rerun_embedding": True,
            "rerun_ranker_training": True,
            "rerun_evaluation": True,
            "commands": ["leanrank-kg embed", "leanrank-kg build-index", "leanrank-kg train-ranker", "leanrank-kg evaluate"],
            "reason": "Embedding model, prefix, device-independent representation text, or vector dimensions invalidate embeddings, indexes, and learned scores.",
        },
        {
            "scenario": "data_split_or_sample_change",
            "rerun_embedding": True,
            "rerun_ranker_training": True,
            "rerun_evaluation": True,
            "commands": ["leanrank-kg full-pipeline --config configs/proofatlas.yaml --force"],
            "reason": "Changing the LeanRank-data sample or theorem-disjoint split changes rows, labels, negatives, graph edges, embeddings, indexes, and test metrics.",
        },
    ]
    return {
        "reuse_by_default": bool(has_embedding_cache and has_index_cache),
        "dataset_name": sample.get("dataset_name") or config.get("dataset_name"),
        "artifact_cache": {
            "embedding_rows": embedding_rows,
            "embedding_backend": (embeddings.get("config") or {}).get("backend"),
            "embedding_model": (embeddings.get("config") or {}).get("model_name"),
            "indexed_entity_count": len(indexed_entities),
            "index_backend": (index.get("summary") or {}).get("backend") or (config.get("index") or {}).get("backend"),
            "premise_ranker_exists": has_ranker,
            "premise_ranker_bytes": ranker_artifact.get("bytes"),
            "difficulty_estimator_exists": bool(difficulty_artifact.get("exists")),
            "difficulty_estimator_bytes": difficulty_artifact.get("bytes"),
        },
        "training_repeat_policy": (
            "Do not retrain by default. Reuse embeddings, indexes, and trained models for report/homepage refreshes; "
            "rerun ranker training only after ranker feature, label, split, or relevant config changes."
        ),
        "scenarios": scenarios,
    }


def _resource_parallelism_profile(
    config: dict[str, Any],
    embeddings: dict[str, Any],
    index: dict[str, Any],
    benchmark: dict[str, Any],
    evaluation: dict[str, Any],
    throughput: dict[str, Any],
) -> dict[str, Any]:
    embedding_config = embeddings.get("config") or {}
    config_embedding = config.get("embedding", {}) or {}
    configured_devices = embedding_config.get("devices") or config_embedding.get("devices") or []
    if isinstance(configured_devices, str):
        configured_devices = [part.strip() for part in configured_devices.split(",") if part.strip()]
    requested_device = embedding_config.get("device") or config_embedding.get("device")
    if not configured_devices and requested_device:
        configured_devices = [requested_device]
    evaluation_scope = ((evaluation.get("test_set_evaluation") or {}).get("evaluation_scope") or {}) if isinstance(evaluation, dict) else {}
    actual_backend_info = evaluation_scope.get("actual_backend_info", {}) if isinstance(evaluation_scope, dict) else {}
    actual_backends = sorted(
        {
            row.get("actual_backend")
            for task in actual_backend_info.values()
            if isinstance(task, dict)
            for row in task.values()
            if isinstance(row, dict) and row.get("actual_backend")
        }
    )
    test_proof_backend = (actual_backend_info.get("proof_state") or {}).get("test", {}) if isinstance(actual_backend_info, dict) else {}
    test_theorem_backend = (actual_backend_info.get("theorem") or {}).get("test", {}) if isinstance(actual_backend_info, dict) else {}
    benchmark_entities = benchmark.get("entities", {}) if isinstance(benchmark, dict) else {}
    bottleneck_rows = (throughput.get("bottleneck_profile") or {}).get("top_stages", []) if isinstance(throughput, dict) else []
    non_gpu_stage_names = {"sample", "normalize", "build_graph", "augment_graph", "compute_difficulty", "train_ranker", "validate"}
    cpu_or_io_stages = [
        row
        for row in bottleneck_rows
        if isinstance(row, dict) and row.get("name") in non_gpu_stage_names
    ]
    return {
        "embedding_parallelism": {
            "backend": embedding_config.get("backend") or config_embedding.get("backend"),
            "model_name": embedding_config.get("model_name") or config_embedding.get("model_name"),
            "requested_device": requested_device,
            "devices": configured_devices,
            "device_count": len(configured_devices),
            "multi_process": bool(embedding_config.get("multi_process") or len(configured_devices) > 1),
            "batch_size": embedding_config.get("batch_size") or config_embedding.get("batch_size"),
            "total_embedding_rows": throughput.get("total_embedding_rows"),
            "embed_stage_seconds": (throughput.get("embedding_bottleneck_profile") or {}).get("embed_stage_seconds"),
            "embedding_rows_per_embed_second": (throughput.get("embedding_bottleneck_profile") or {}).get("embedding_rows_per_embed_second"),
        },
        "evaluation_parallelism": {
            "ranking_backend": evaluation_scope.get("ranking_backend"),
            "requested_use_gpu": evaluation_scope.get("use_gpu"),
            "requested_gpu_device": evaluation_scope.get("gpu_device"),
            "batch_size": evaluation_scope.get("batch_size"),
            "actual_backends": actual_backends,
            "test_proof_state_backend": test_proof_backend.get("actual_backend"),
            "test_theorem_backend": test_theorem_backend.get("actual_backend"),
            "test_proof_state_queries": test_proof_backend.get("query_count"),
            "test_theorem_queries": test_theorem_backend.get("query_count"),
            "candidate_count": test_proof_backend.get("candidate_count") or test_theorem_backend.get("candidate_count"),
            "fallback_reasons": sorted(
                {
                    str(row.get("fallback_reason"))
                    for task in actual_backend_info.values()
                    if isinstance(task, dict)
                    for row in task.values()
                    if isinstance(row, dict) and row.get("fallback_reason")
                }
            ),
        },
        "index_parallelism": {
            "backend": (index.get("summary") or {}).get("backend") or (config.get("index") or {}).get("backend"),
            "requested_backend": (index.get("summary") or {}).get("requested_backend") or (config.get("index") or {}).get("backend"),
            "metric": (index.get("summary") or {}).get("metric") or (config.get("index") or {}).get("metric"),
            "hnsw_M": (config.get("index") or {}).get("M"),
            "hnsw_ef_construction": (config.get("index") or {}).get("ef_construction"),
            "hnsw_ef_search": (config.get("index") or {}).get("ef_search"),
            "indexed_entities": sorted(benchmark_entities.keys()),
            "mean_speedup_vs_exact": throughput.get("mean_index_speedup_vs_exact"),
            "min_recall_vs_exact": throughput.get("min_index_recall_vs_exact"),
        },
        "cpu_or_io_heavy_stages": cpu_or_io_stages,
    }


def _execution_mode_summary(throughput: dict[str, Any]) -> dict[str, Any]:
    resources = throughput.get("resource_parallelism_profile", {}) if isinstance(throughput, dict) else {}
    embedding = resources.get("embedding_parallelism", {}) if isinstance(resources, dict) else {}
    evaluation = resources.get("evaluation_parallelism", {}) if isinstance(resources, dict) else {}
    indexing = resources.get("index_parallelism", {}) if isinstance(resources, dict) else {}
    refresh_reuse = throughput.get("refresh_reuse_profile", {}) if isinstance(throughput, dict) else {}
    bottleneck = throughput.get("bottleneck_profile", {}) if isinstance(throughput, dict) else {}
    cpu_or_io = resources.get("cpu_or_io_heavy_stages", []) if isinstance(resources, dict) else []

    embedding_device_count = int(embedding.get("device_count") or 0)
    embedding_gpu_active = embedding.get("requested_device") == "cuda" and embedding_device_count > 0
    multi_gpu_embedding = embedding_gpu_active and embedding_device_count > 1 and bool(embedding.get("multi_process"))
    evaluation_backends = set(evaluation.get("actual_backends") or [])
    evaluation_gpu_active = "torch_cuda" in evaluation_backends
    index_backend = indexing.get("backend")
    indexed_entities = indexing.get("indexed_entities") or []
    ann_index_active = index_backend in {"hnswlib", "faiss", "lancedb"} and bool(indexed_entities)
    primary_stage = bottleneck.get("primary_stage")
    cpu_stage_names = [row.get("name") for row in cpu_or_io if isinstance(row, dict) and row.get("name")]

    if multi_gpu_embedding:
        embedding_mode = "multi_gpu_sentence_transformer"
    elif embedding_gpu_active:
        embedding_mode = "single_gpu_sentence_transformer"
    else:
        embedding_mode = "cpu_or_non_neural_embedding"

    if evaluation_gpu_active:
        evaluation_mode = "batched_gpu_retrieval_evaluation"
    elif evaluation_backends:
        evaluation_mode = "non_cuda_batched_retrieval_evaluation"
    else:
        evaluation_mode = "evaluation_backend_not_recorded"

    if ann_index_active:
        index_mode = f"{index_backend}_ann_candidate_generation"
    elif indexed_entities:
        index_mode = f"{index_backend or 'unknown'}_indexed_candidate_generation"
    else:
        index_mode = "no_persistent_index_recorded"

    if primary_stage == "embed" and embedding_gpu_active:
        bottleneck_interpretation = "embedding is still the largest timed stage even with GPU encoding, so artifact reuse matters for report and reranking refreshes"
    elif primary_stage == "evaluate" and evaluation_gpu_active:
        bottleneck_interpretation = "evaluation is the largest timed stage despite batched GPU scoring, so sampled development evaluation and full final evaluation should stay separate"
    elif primary_stage:
        bottleneck_interpretation = f"{primary_stage} is the largest timed stage and should be inspected before scaling further"
    else:
        bottleneck_interpretation = "no primary timed bottleneck was recorded"

    return {
        "embedding_mode": embedding_mode,
        "embedding_gpu_active": embedding_gpu_active,
        "multi_gpu_embedding": multi_gpu_embedding,
        "embedding_device_count": embedding_device_count,
        "evaluation_mode": evaluation_mode,
        "evaluation_gpu_active": evaluation_gpu_active,
        "evaluation_actual_backends": sorted(evaluation_backends),
        "index_mode": index_mode,
        "ann_index_active": ann_index_active,
        "primary_timed_bottleneck": primary_stage,
        "cpu_or_io_heavy_stage_names": cpu_stage_names,
        "artifact_reuse_by_default": refresh_reuse.get("reuse_by_default"),
        "ranker_retrain_policy": refresh_reuse.get("training_repeat_policy"),
        "bottleneck_interpretation": bottleneck_interpretation,
    }


def _performance_acceptance_profile(
    scale: dict[str, Any],
    throughput: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    resources = throughput.get("resource_parallelism_profile", {}) if isinstance(throughput, dict) else {}
    embedding = resources.get("embedding_parallelism", {}) if isinstance(resources, dict) else {}
    evaluation_parallel = resources.get("evaluation_parallelism", {}) if isinstance(resources, dict) else {}
    refresh_reuse = throughput.get("refresh_reuse_profile", {}) if isinstance(throughput, dict) else {}
    heldout = evaluation.get("held_out_test_coverage", {}) if isinstance(evaluation, dict) else {}
    embedding_rows_per_second = embedding.get("embedding_rows_per_embed_second")
    gates = [
        {
            "name": "target_dataset",
            "severity": "required",
            "passed": scale.get("target_dataset_confirmed") is True,
            "value": scale.get("dataset_name"),
            "threshold": "erbacher/LeanRank-data",
            "evidence": "Production performance profile must use the configured LeanRank-data source.",
        },
        {
            "name": "large_scale_slice",
            "severity": "required",
            "passed": scale.get("scale_bucket") == "large" and int(scale.get("current_total_split_rows") or 0) >= 60000,
            "value": scale.get("current_total_split_rows"),
            "threshold": ">=60000 processed split rows and scale_bucket=large",
            "evidence": "Performance numbers should describe a non-demo LeanRank-data run.",
        },
        {
            "name": "full_heldout_evaluation",
            "severity": "required",
            "passed": heldout.get("proof_state_coverage_fraction") == 1.0 and heldout.get("theorem_coverage_fraction") == 1.0,
            "value": {
                "proof_state_coverage_fraction": heldout.get("proof_state_coverage_fraction"),
                "theorem_coverage_fraction": heldout.get("theorem_coverage_fraction"),
            },
            "threshold": "both coverage fractions == 1.0",
            "evidence": "Final retrieval claims should use full held-out proof-state and theorem evaluation.",
        },
        {
            "name": "fresh_pipeline_timing",
            "severity": "required",
            "passed": throughput.get("scale_estimate_reliable") is True and throughput.get("throughput_basis") == "executed_pipeline_run",
            "value": {
                "scale_estimate_reliable": throughput.get("scale_estimate_reliable"),
                "throughput_basis": throughput.get("throughput_basis"),
            },
            "threshold": "scale_estimate_reliable=true and throughput_basis=executed_pipeline_run",
            "evidence": "Bottleneck shares and throughput estimates must come from a non-cached production timing run.",
        },
        {
            "name": "ann_speedup",
            "severity": "required",
            "passed": (throughput.get("mean_index_speedup_vs_exact") or 0.0) >= 5.0,
            "value": throughput.get("mean_index_speedup_vs_exact"),
            "threshold": ">=5x mean indexed speedup vs exact cosine",
            "evidence": "Large-scale retrieval should use indexed candidate generation with measurable speedup.",
        },
        {
            "name": "ann_recall",
            "severity": "required",
            "passed": (throughput.get("min_index_recall_vs_exact") or 0.0) >= 0.95,
            "value": throughput.get("min_index_recall_vs_exact"),
            "threshold": ">=0.95 minimum Recall@10 vs exact cosine across indexed entities",
            "evidence": "ANN speedup should not substantially change nearest-neighbor candidate quality.",
        },
        {
            "name": "gpu_embedding_parallelism",
            "severity": "advisory",
            "passed": bool(embedding.get("requested_device") == "cuda" and int(embedding.get("device_count") or 0) >= 1),
            "value": {
                "requested_device": embedding.get("requested_device"),
                "device_count": embedding.get("device_count"),
                "multi_process": embedding.get("multi_process"),
            },
            "threshold": "cuda requested with at least one device",
            "evidence": "Embedding remains the primary bottleneck, so GPU encoding should stay enabled for production refreshes.",
        },
        {
            "name": "gpu_evaluation_backend",
            "severity": "advisory",
            "passed": "torch_cuda" in set(evaluation_parallel.get("actual_backends") or []),
            "value": evaluation_parallel.get("actual_backends"),
            "threshold": "actual_backends includes torch_cuda",
            "evidence": "Held-out ranking evaluation should use the batched GPU path when available.",
        },
        {
            "name": "artifact_reuse_ready",
            "severity": "advisory",
            "passed": refresh_reuse.get("reuse_by_default") is True,
            "value": refresh_reuse.get("reuse_by_default"),
            "threshold": "reuse_by_default=true",
            "evidence": "Report and homepage refreshes should reuse embeddings, indexes, and trained models unless inputs change.",
        },
        {
            "name": "embedding_throughput_recorded",
            "severity": "advisory",
            "passed": embedding_rows_per_second is not None and float(embedding_rows_per_second) > 0.0,
            "value": embedding_rows_per_second,
            "threshold": ">0 embedding rows/sec",
            "evidence": "The report should expose enough throughput data to compare future embedding optimizations.",
        },
    ]
    required = [gate for gate in gates if gate["severity"] == "required"]
    advisory = [gate for gate in gates if gate["severity"] == "advisory"]
    return {
        "summary": {
            "required_gates_passed": all(gate["passed"] for gate in required),
            "advisory_gates_passed": all(gate["passed"] for gate in advisory),
            "passed_gate_count": sum(1 for gate in gates if gate["passed"]),
            "total_gate_count": len(gates),
            "required_gate_count": len(required),
            "advisory_gate_count": len(advisory),
        },
        "gates": gates,
    }


def _scale_projection_profile(scale: dict[str, Any], throughput: dict[str, Any]) -> dict[str, Any]:
    current_rows = int(scale.get("current_total_split_rows") or 0)
    requested_source_rows = int(scale.get("source_rows") or throughput.get("requested_source_rows") or current_rows or 0)
    timed_stage_seconds = throughput.get("timed_stage_seconds", {}) if isinstance(throughput, dict) else {}
    total_seconds = throughput.get("total_pipeline_seconds")
    embed_seconds = timed_stage_seconds.get("embed")
    index_seconds = throughput.get("index_build_seconds_total")

    def _projection(label: str, target_rows: int) -> dict[str, Any]:
        factor = (float(target_rows) / float(current_rows)) if current_rows > 0 else None

        def scaled(value: Any) -> float | None:
            if value is None or factor is None:
                return None
            return float(value) * factor

        return {
            "label": label,
            "target_processed_rows": int(target_rows),
            "scale_factor_vs_current": factor,
            "estimated_total_seconds": scaled(total_seconds),
            "estimated_embed_seconds": scaled(embed_seconds),
            "estimated_index_build_seconds": scaled(index_seconds),
        }

    target_rows = []
    if current_rows > 0:
        target_rows.extend(
            [
                ("current_1x", current_rows),
                ("current_2x", current_rows * 2),
                ("current_5x", current_rows * 5),
            ]
        )
    if requested_source_rows > 0 and requested_source_rows not in {row_count for _, row_count in target_rows}:
        target_rows.append(("configured_source_rows", requested_source_rows))
    projections = [_projection(label, row_count) for label, row_count in target_rows]
    return {
        "method": "linear_projection_from_current_timed_pipeline",
        "assumptions": [
            "Projection scales current timed pipeline seconds linearly with processed row count.",
            "Embedding and index-build estimates scale their timed stage seconds linearly.",
            "This is for capacity planning and should be replaced by a fresh timing run after changing hardware, embedding model, index backend, or sample shape.",
        ],
        "current_processed_rows": current_rows,
        "configured_source_rows": requested_source_rows,
        "scale_estimate_reliable": throughput.get("scale_estimate_reliable"),
        "basis_total_seconds": total_seconds,
        "basis_embed_seconds": embed_seconds,
        "basis_index_build_seconds": index_seconds,
        "projections": projections,
    }


def _artifact_storage_profile(scale: dict[str, Any], throughput: dict[str, Any]) -> dict[str, Any]:
    artifact_roots = [
        "data/processed",
        "outputs/embeddings",
        "outputs/indexes",
        "outputs/reports",
        "homepage/assets",
    ]
    directories = {path: _directory_size(path) for path in artifact_roots}
    total_bytes = sum(int(row.get("bytes") or 0) for row in directories.values())
    current_rows = int(scale.get("current_total_split_rows") or 0)
    bytes_per_processed_row = (total_bytes / current_rows) if current_rows > 0 else None
    largest_files: list[dict[str, Any]] = []
    for root in artifact_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                largest_files.append({"path": str(file_path), "bytes": int(file_path.stat().st_size)})
    largest_files.sort(key=lambda row: int(row["bytes"]), reverse=True)
    referenced_index_paths = {str(Path("outputs/indexes/index_summary.json"))}
    index_root = Path("outputs/indexes")
    if index_root.exists():
        for manifest_path in index_root.glob("*_index_manifest.json"):
            manifest = read_json(manifest_path, {}) or {}
            referenced_index_paths.add(str(manifest_path))
            index_path = manifest.get("index_path")
            if index_path:
                referenced_index_paths.add(str(Path(index_path)))
            metadata_path = str(manifest_path).replace("_index_manifest.json", "_index_metadata.parquet")
            referenced_index_paths.add(metadata_path)
    unreferenced_index_artifacts = []
    if index_root.exists():
        for file_path in index_root.glob("*"):
            if file_path.is_file() and str(file_path) not in referenced_index_paths:
                unreferenced_index_artifacts.append({"path": str(file_path), "bytes": int(file_path.stat().st_size)})
    unreferenced_index_artifacts.sort(key=lambda row: int(row["bytes"]), reverse=True)
    unreferenced_index_bytes = sum(int(row["bytes"]) for row in unreferenced_index_artifacts)

    scale_projection = throughput.get("scale_projection_profile", {}) if isinstance(throughput, dict) else {}
    projections = []
    for row in scale_projection.get("projections", []) if isinstance(scale_projection, dict) else []:
        if not isinstance(row, dict):
            continue
        factor = row.get("scale_factor_vs_current")
        projected_bytes = float(total_bytes) * float(factor) if factor is not None else None
        projections.append(
            {
                "label": row.get("label"),
                "target_processed_rows": row.get("target_processed_rows"),
                "scale_factor_vs_current": factor,
                "estimated_artifact_bytes": projected_bytes,
                "estimated_artifact_gib": (projected_bytes / (1024**3)) if projected_bytes is not None else None,
            }
        )
    return {
        "method": "filesystem_artifact_footprint_with_linear_scale_projection",
        "artifact_roots": artifact_roots,
        "directories": directories,
        "total_artifact_bytes": total_bytes,
        "total_artifact_gib": total_bytes / (1024**3),
        "bytes_per_processed_row": bytes_per_processed_row,
        "largest_files": largest_files[:10],
        "unreferenced_index_artifacts": unreferenced_index_artifacts[:20],
        "unreferenced_index_artifact_count": len(unreferenced_index_artifacts),
        "unreferenced_index_artifact_bytes": unreferenced_index_bytes,
        "unreferenced_index_artifact_gib": unreferenced_index_bytes / (1024**3),
        "projections": projections,
        "notes": [
            "Storage projection scales current artifact bytes linearly with processed row count.",
            "Index artifacts dominate current storage and should be monitored when scaling LeanRank-data runs.",
            "Unreferenced index artifacts are files under outputs/indexes that are not referenced by the current index manifests.",
        ],
    }


def _recommendations(
    config: dict[str, Any],
    sample: dict[str, Any],
    benchmark: dict[str, Any],
    readiness: dict[str, Any],
    scale: dict[str, Any],
    throughput: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[dict[str, str]]:
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
    if throughput.get("scale_estimate_reliable") is not True:
        recommendations.append(
            {
                "priority": "high",
                "area": "performance_timing",
                "recommendation": (
                    "Refresh pipeline timings with a non-cached production run before using throughput numbers for scale-up planning. "
                    "Run `make refresh-production-report` after a full `leanrank-kg full-pipeline --config configs/proofatlas.yaml --force` timing pass, "
                    "or keep the current throughput fields marked as cached/partial diagnostics only."
                ),
            }
        )
    evaluation_delta = throughput.get("evaluation_timing_delta") or {}
    if evaluation_delta.get("current_faster_than_pipeline_timing") is True:
        recommendations.append(
            {
                "priority": "medium",
                "area": "performance_timing",
                "recommendation": (
                    "The current standalone evaluation is materially faster than the saved full-pipeline timing. "
                    f"Saved pipeline evaluate seconds: {evaluation_delta.get('timed_pipeline_evaluate_seconds')}; "
                    f"current evaluation seconds: {evaluation_delta.get('current_evaluation_seconds')}. "
                    "Rerun `make refresh-production-timing` before using full-pipeline bottleneck shares as final throughput evidence."
                ),
            }
        )
    bottleneck_profile = throughput.get("bottleneck_profile") or {}
    primary_share = bottleneck_profile.get("primary_stage_share_of_total")
    if primary_share is not None and primary_share >= 0.2:
        primary_stage = bottleneck_profile.get("primary_stage")
        if primary_stage == "evaluate":
            recommendation = (
                "Evaluation is the current largest timed bottleneck. Keep full held-out metrics for final claims, "
                "but use sampled evaluation during development and prioritize batched/vectorized scoring or parallel domain shards before scaling evaluation further."
            )
        elif primary_stage == "embed":
            recommendation = (
                "Embedding is the current largest timed bottleneck. Reuse cached embeddings when training/reranking only, "
                "and keep multi-GPU sentence-transformer encoding enabled for larger LeanRank-data refreshes."
            )
        else:
            recommendation = (
                f"`{primary_stage}` is the current largest timed bottleneck. Inspect this stage before scaling the LeanRank-data refresh further."
            )
        recommendations.append(
            {
                "priority": "medium",
                "area": "pipeline_bottleneck",
                "recommendation": recommendation,
            }
        )
    retrieval_profile = throughput.get("retrieval_bottleneck_profile") or {}
    proof_state_bottleneck = (retrieval_profile.get("proof_state") or {}).get("primary_accuracy_bottleneck")
    if proof_state_bottleneck == "candidate_generation_or_embeddings":
        recommendations.append(
            {
                "priority": "medium",
                "area": "retrieval_accuracy",
                "recommendation": (
                    "Proof-state Recall@100 is low, so proof-state premise retrieval is currently limited by candidate generation or embeddings before reranking. "
                    "Prioritize stronger proof-state/query representations, domain-aware candidate pools, and embedding model comparisons before adding heavier rerankers."
                ),
            }
        )
    evaluation_scope = (evaluation.get("test_set_evaluation") or {}).get("evaluation_scope", {})
    if evaluation_scope.get("is_sampled") is True:
        recommendations.append(
            {
                "priority": "medium",
                "area": "evaluation_scope",
                "recommendation": (
                    "Current held-out metrics are sampled because evaluation limits are configured. "
                    f"Proof-state limits: {evaluation_scope.get('proof_state_limits')}; theorem limits: {evaluation_scope.get('theorem_limits')}. "
                    "For final quantitative claims, run `make refresh-production-full-eval` or rerun evaluation with these limits removed or raised enough to cover the full held-out split."
                ),
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
    config_hash = _config_hash(config)
    manifest = read_json("outputs/reports/corpus_manifest.json", {}) or {}
    sample = _sample_stage(manifest, config)
    processed = _processed_stage()
    graph = _graph_stage()
    embeddings = _embedding_stage()
    index = _index_stage()
    benchmark = _benchmark_stage()
    timings = _timing_stage(expected_config_hash=config_hash)
    evaluation = _evaluation_stage()
    readiness = _readiness_stage()
    scale = _scale_profile(config, sample, index, readiness)
    throughput = _throughput_profile(sample, embeddings, index, benchmark, timings, evaluation)
    throughput["retrieval_bottleneck_profile"] = _retrieval_bottleneck_profile(evaluation)
    throughput["rapid_convergence_profile"] = _rapid_convergence_profile(evaluation, readiness, throughput)
    throughput["metric_uncertainty_profile"] = _metric_uncertainty_profile(evaluation)
    throughput["rerank_evaluation_cost_profile"] = _rerank_evaluation_cost_profile(evaluation)
    throughput["refresh_reuse_profile"] = _refresh_reuse_profile(config, sample, embeddings, index)
    throughput["resource_parallelism_profile"] = _resource_parallelism_profile(config, embeddings, index, benchmark, evaluation, throughput)
    throughput["execution_mode_summary"] = _execution_mode_summary(throughput)
    throughput["performance_acceptance_profile"] = _performance_acceptance_profile(scale, throughput, evaluation)
    throughput["scale_projection_profile"] = _scale_projection_profile(scale, throughput)
    throughput["artifact_storage_profile"] = _artifact_storage_profile(scale, throughput)
    recommendations = _recommendations(config, sample, benchmark, readiness, scale, throughput, evaluation)
    return {
        "config_path": config_path,
        "config_hash": config_hash,
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
        "throughput_profile": throughput,
        "recommendations": recommendations,
    }


def run(config_path: str = "configs/proofatlas.yaml", output_path: str = "outputs/reports/pipeline_performance_report.json") -> dict[str, Any]:
    report = build_report(config_path)
    write_json(output_path, report)
    return report
