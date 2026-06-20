from __future__ import annotations

from datetime import datetime, timezone
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


def _metric_snapshot(refresh_dashboard: dict[str, Any]) -> dict[str, Any]:
    scale = refresh_dashboard.get("scale", {}).get("total_train_val_test", {})
    retrieval = refresh_dashboard.get("retrieval_quality", {})
    parsing = refresh_dashboard.get("parsing", {})
    difficulty = refresh_dashboard.get("difficulty", {})
    artifact = refresh_dashboard.get("artifact_compatibility", {})
    index = refresh_dashboard.get("index_benchmark", {})
    premise_index = index.get("premise", {}) if isinstance(index, dict) else {}
    return {
        "theorems": scale.get("theorems"),
        "proof_states": scale.get("proof_states"),
        "premises": scale.get("premises"),
        "nodes": scale.get("nodes"),
        "edges": scale.get("edges"),
        "premise_recall_at_10": retrieval.get("premise_recall_at_10"),
        "theorem_recall_at_10": retrieval.get("theorem_recall_at_10"),
        "theorem_mrr": retrieval.get("theorem_mrr"),
        "minimum_context_coverage": parsing.get("minimum_context_coverage"),
        "difficulty_train_mae": difficulty.get("train_mae"),
        "premise_index_recall_vs_exact": premise_index.get("recall_vs_exact") if isinstance(premise_index, dict) else None,
        "artifact_compatible": artifact.get("passed"),
        "ready_for_refresh_comparison": refresh_dashboard.get("ready_for_refresh_comparison"),
    }


def _numeric_delta(current: Any, previous: Any) -> dict[str, float | None]:
    if current is None or previous is None:
        return {"absolute": None, "relative": None}
    try:
        current_f = float(current)
        previous_f = float(previous)
    except (TypeError, ValueError):
        return {"absolute": None, "relative": None}
    absolute = current_f - previous_f
    relative = absolute / previous_f if previous_f else None
    return {"absolute": absolute, "relative": relative}


def _refresh_trend(current_dashboard: dict[str, Any], previous_dashboard: dict[str, Any] | None) -> dict[str, Any]:
    current_snapshot = _metric_snapshot(current_dashboard)
    previous_snapshot = _metric_snapshot(previous_dashboard or {}) if previous_dashboard else {}
    deltas = {
        key: _numeric_delta(value, previous_snapshot.get(key))
        for key, value in current_snapshot.items()
        if isinstance(value, (int, float)) or isinstance(previous_snapshot.get(key), (int, float))
    }
    return {
        "has_previous": bool(previous_dashboard),
        "current_corpus": current_dashboard.get("corpus", {}),
        "previous_corpus": (previous_dashboard or {}).get("corpus", {}),
        "current": current_snapshot,
        "previous": previous_snapshot,
        "deltas": deltas,
        "quality_gate_changes": {
            key: {
                "current": value,
                "previous": (previous_dashboard or {}).get("quality_gates", {}).get(key),
            }
            for key, value in current_dashboard.get("quality_gates", {}).items()
            if value != (previous_dashboard or {}).get("quality_gates", {}).get(key)
        },
    }


def _refresh_history(
    previous_history: list[dict[str, Any]] | None,
    current_dashboard: dict[str, Any],
    refresh_trend: dict[str, Any],
    limit: int = 20,
) -> dict[str, Any]:
    history = list(previous_history or [])
    entry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": current_dashboard.get("corpus", {}),
        "metrics": refresh_trend.get("current", {}),
        "deltas_from_previous": refresh_trend.get("deltas", {}),
        "quality_gates": current_dashboard.get("quality_gates", {}),
        "quality_gate_changes": refresh_trend.get("quality_gate_changes", {}),
        "ready_for_refresh_comparison": current_dashboard.get("ready_for_refresh_comparison"),
    }
    history.append(entry)
    if limit > 0:
        history = history[-limit:]
    return {
        "history_limit": limit,
        "entry_count": len(history),
        "entries": history,
        "latest": history[-1] if history else None,
    }


def _refresh_dashboard(
    *,
    corpus_manifest: dict[str, Any],
    graph_stats: dict[str, Any],
    domains: dict[str, Any],
    metrics: dict[str, Any],
    validation: dict[str, Any],
    artifact_compatibility: dict[str, Any],
    index_benchmark: dict[str, Any],
    difficulty_estimator_metrics: dict[str, Any],
) -> dict[str, Any]:
    split_counts = _split_counts(graph_stats)
    total = {
        "theorems": sum(int(row.get("theorems", 0)) for row in split_counts if row.get("split") in {"train", "val", "test"}),
        "proof_states": sum(int(row.get("proof_states", 0)) for row in split_counts if row.get("split") in {"train", "val", "test"}),
        "premises": sum(int(row.get("premises", 0)) for row in split_counts if row.get("split") in {"train", "val", "test"}),
        "nodes": sum(int(row.get("nodes", 0)) for row in split_counts if row.get("split") in {"train", "val", "test"}),
        "edges": sum(int(row.get("edges", 0)) for row in split_counts if row.get("split") in {"train", "val", "test"}),
    }
    context = validation.get("context", {})
    theorem_query_parse = validation.get("theorem_query_parse", {})
    context_coverages = [float(row.get("coverage", 0.0)) for row in context.values() if isinstance(row, dict)]
    theorem_query_coverages = [float(row.get("parse_coverage", 0.0)) for row in theorem_query_parse.values() if isinstance(row, dict)]
    graph_validation = validation.get("graph", {})
    index_entities = index_benchmark.get("entities", {}) if isinstance(index_benchmark, dict) else {}
    index_summary = {
        name: {
            "backend": row.get("backend"),
            "indexed_available": row.get("indexed_available"),
            "exact_ms_per_query": row.get("exact_ms_per_query"),
            "indexed_ms_per_query": row.get("indexed_ms_per_query"),
            "recall_vs_exact": row.get(f"recall_at_{row.get('top_k')}_vs_exact"),
        }
        for name, row in index_entities.items()
        if isinstance(row, dict)
    }
    gates = {
        "artifact_compatible": artifact_compatibility.get("passed") is True,
        "schema_clean": int(validation.get("schema", {}).get("error_count", 1) or 0) == 0,
        "split_clean": validation.get("split_leakage", {}).get("has_leakage") is False,
        "graph_valid": bool(graph_validation)
        and all(row.get("networkx_loadable") and row.get("missing_endpoint_count") == 0 for row in graph_validation.values()),
        "parsing_coverage_present": bool(context_coverages) and min(context_coverages) > 0.0,
        "theorem_query_parse_present": bool(theorem_query_coverages) and min(theorem_query_coverages) > 0.0,
        "theorem_eval_present": int(metrics.get("theorem_retrieval_evaluated_theorems", 0) or 0) > 0,
        "index_benchmark_present": bool(index_summary),
    }
    return {
        "corpus": {
            "dataset_name": corpus_manifest.get("dataset_name"),
            "source_kind": corpus_manifest.get("source_kind"),
            "config_hash": corpus_manifest.get("config_hash"),
            "corpus_version": corpus_manifest.get("corpus", {}).get("corpus_version"),
            "lean_version": corpus_manifest.get("corpus", {}).get("lean_version"),
            "mathlib_commit": corpus_manifest.get("corpus", {}).get("mathlib_commit"),
            "sampled_rows": corpus_manifest.get("sampled_rows"),
            "sampled_theorems": corpus_manifest.get("sampled_theorems"),
            "data_supervision": corpus_manifest.get("data_supervision")
            or artifact_compatibility.get("data_supervision", {}),
        },
        "scale": {"splits": split_counts, "total_train_val_test": total},
        "domain_coverage": {"top_domains": _top_domains(domains, limit=12), "raw": domains},
        "retrieval_quality": {
            "premise_recall_at_10": metrics.get("Recall@10"),
            "premise_mrr": metrics.get("MRR"),
            "ranker_auc": metrics.get("AUC"),
            "theorem_recall_at_10": metrics.get("theorem_retrieval_Recall@10"),
            "theorem_mrr": metrics.get("theorem_retrieval_MRR"),
            "theorem_ndcg_at_10": metrics.get("theorem_retrieval_nDCG@10"),
            "evaluated_theorems": metrics.get("theorem_retrieval_evaluated_theorems"),
            "gold_premises_in_train_index": metrics.get("theorem_retrieval_gold_premises_in_train_index"),
            "gold_premises_missing_from_train_index": metrics.get("theorem_retrieval_gold_premises_missing_from_train_index"),
        },
        "parsing": {
            "context_parse_coverage": context,
            "theorem_query_parse_coverage": theorem_query_parse,
            "minimum_context_coverage": min(context_coverages) if context_coverages else 0.0,
            "minimum_theorem_query_parse_coverage": min(theorem_query_coverages) if theorem_query_coverages else 0.0,
        },
        "difficulty": {
            "train_mae": difficulty_estimator_metrics.get("train", {}).get("mae"),
            "val_mae": difficulty_estimator_metrics.get("val", {}).get("mae"),
            "train_residual_quantiles": difficulty_estimator_metrics.get("train", {}).get("residual_quantiles"),
            "train_calibration_bins": difficulty_estimator_metrics.get("train", {}).get("calibration_bins"),
        },
        "index_benchmark": index_summary,
        "artifact_compatibility": {
            "passed": artifact_compatibility.get("passed"),
            "config_hash_matches": artifact_compatibility.get("config_hash_matches"),
            "failures": artifact_compatibility.get("failures", []),
            "warnings": artifact_compatibility.get("warnings", []),
            "data_supervision": artifact_compatibility.get("data_supervision", {}),
        },
        "quality_gates": gates,
        "ready_for_refresh_comparison": all(gates.values()),
    }


def _production_evidence(
    *,
    metrics: dict[str, Any],
    premise_supervision: dict[str, Any],
    pipeline_run_timings: dict[str, Any],
    pipeline_performance: dict[str, Any],
    test_set_evaluation: dict[str, Any],
) -> dict[str, Any]:
    current_supervision = premise_supervision.get("current_artifact_supervision", {})
    train_supervision = premise_supervision.get("splits", {}).get("train", {})
    throughput = pipeline_performance.get("throughput_profile", {})
    bottleneck_profile = throughput.get("bottleneck_profile", {}) if isinstance(throughput, dict) else {}
    evaluation_timing_delta = throughput.get("evaluation_timing_delta", {}) if isinstance(throughput, dict) else {}
    evaluation_timing = (
        pipeline_performance.get("stages", {}).get("evaluation", {}).get("evaluation_timing", {})
        if isinstance(pipeline_performance, dict)
        else {}
    )
    scale = pipeline_performance.get("scale_profile", {})
    timing_summary = pipeline_performance.get("stages", {}).get("pipeline_run_timings", {})
    test_eval = test_set_evaluation.get("test", {}) if isinstance(test_set_evaluation, dict) else {}
    proof_failure = test_eval.get("proof_state_retrieval", {}).get("failure_profile", {})
    theorem_failure = test_eval.get("theorem_retrieval", {}).get("failure_profile", {})
    reranked_failure = test_eval.get("proof_state_reranked_retrieval", {}).get("failure_profile", {})
    return {
        "heldout": {
            "proof_state_evaluated_queries": metrics.get("test_proof_state_evaluated_queries"),
            "proof_state_evaluated_retrievable_queries": metrics.get("test_proof_state_evaluated_retrievable_queries"),
            "proof_state_recall_at_10": metrics.get("Recall@10"),
            "proof_state_recall_at_100": metrics.get("Recall@100"),
            "theorem_evaluated_queries": metrics.get("theorem_retrieval_evaluated_theorems"),
            "theorem_evaluated_queries_with_train_gold": metrics.get("theorem_retrieval_evaluated_theorems_with_train_gold"),
            "theorem_recall_at_10": metrics.get("theorem_retrieval_Recall@10"),
            "theorem_recall_at_100": metrics.get("theorem_retrieval_Recall@100"),
        },
        "supervision": {
            "total_positive_edges": current_supervision.get("total_positive_edges"),
            "total_negative_edges": current_supervision.get("total_negative_edges"),
            "negative_to_positive_edge_ratio": current_supervision.get("negative_to_positive_edge_ratio"),
            "train_positive_proof_state_coverage": train_supervision.get("positive_proof_state_coverage"),
            "train_negative_proof_state_coverage": train_supervision.get("negative_proof_state_coverage"),
            "train_negative_hardness_mean": train_supervision.get("negative_candidate_hardness", {}).get("mean"),
        },
        "timing": {
            "total_seconds": pipeline_run_timings.get("total_seconds") or timing_summary.get("total_seconds"),
            "executed_stage_count": pipeline_run_timings.get("executed_stage_count"),
            "skipped_stage_count": pipeline_run_timings.get("skipped_stage_count"),
            "throughput_basis": throughput.get("throughput_basis"),
            "scale_estimate_reliable": throughput.get("scale_estimate_reliable"),
            "embedding_rows_per_second": throughput.get("embedding_rows_per_second"),
            "processed_rows_per_second": throughput.get("processed_rows_per_second"),
            "embedding_device": scale.get("embedding_device"),
            "embedding_devices": scale.get("embedding_devices", []),
            "slowest_stage": throughput.get("slowest_stage"),
            "bottleneck_profile": bottleneck_profile,
            "evaluation_timing": evaluation_timing,
            "evaluation_timing_delta": evaluation_timing_delta,
        },
        "failure_profile": {
            "proof_state": proof_failure,
            "theorem": theorem_failure,
            "reranked_proof_state": reranked_failure,
        },
    }


def _short_label(row: pd.Series) -> str:
    for col in ["full_name", "label", "file_path", "id"]:
        value = str(row.get(col, "") or "")
        if value:
            return value.split("/")[-1].split(".")[-1][:42]
    return ""


def _graph_visualization_sample(preferred_split: str = "demo", limit: int = 40) -> dict[str, Any]:
    graph_dir = Path(f"outputs/graph/{preferred_split}")
    split = preferred_split
    if not (graph_dir / "nodes.parquet").exists():
        graph_dir = Path("outputs/graph/train")
        split = "train"
    if not (graph_dir / "nodes.parquet").exists():
        return {"split": split, "nodes": [], "edges": []}
    nodes = pd.read_parquet(graph_dir / "nodes.parquet")
    edge_path = graph_dir / "edges_enriched.parquet" if (graph_dir / "edges_enriched.parquet").exists() else graph_dir / "edges.parquet"
    edges = pd.read_parquet(edge_path)
    if nodes.empty or edges.empty:
        return {"split": split, "nodes": [], "edges": []}
    theorem_ids = nodes[nodes["node_type"] == "Theorem"]["id"].tolist()
    seed = theorem_ids[0] if theorem_ids else str(nodes.iloc[0]["id"])
    selected = {seed}
    for _ in range(2):
        frontier_edges = edges[edges["source"].isin(selected) | edges["target"].isin(selected)]
        selected |= set(frontier_edges["source"].head(limit * 2)) | set(frontier_edges["target"].head(limit * 2))
        if len(selected) >= limit:
            break
    selected_nodes = nodes[nodes["id"].isin(selected)].head(limit).copy()
    selected_ids = set(selected_nodes["id"])
    selected_edges = edges[edges["source"].isin(selected_ids) & edges["target"].isin(selected_ids)].head(limit * 2).copy()
    type_order = ["Theorem", "ProofState", "Premise", "ProofTechnique", "TacticStep", "FileModule"]
    rows = []
    for node_type, group in selected_nodes.groupby("node_type", sort=False):
        type_idx = type_order.index(node_type) if node_type in type_order else len(type_order)
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            rows.append(
                {
                    "id": row["id"],
                    "node_type": node_type,
                    "label": _short_label(row),
                    "x": 90 + type_idx * 145,
                    "y": 60 + idx * 42,
                }
            )
    return {
        "split": split,
        "nodes": rows,
        "edges": selected_edges[["source", "target", "edge_type"]].to_dict(orient="records"),
    }


def build_summary(config_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path) if config_path else {}
    project_name = config.get("project_name", "ProofAtlas")
    previous_refresh_dashboard = read_json("outputs/reports/refresh_dashboard.json", {})
    previous_refresh_history = read_json("outputs/reports/refresh_history.json", {}).get("entries", [])
    graph_stats = read_json("outputs/reports/graph_stats_summary.json", {})
    metrics = read_json("outputs/reports/metrics.json", {})
    difficulty_estimator_metrics = read_json("outputs/reports/difficulty_estimator_metrics.json", {})
    embedding_config = read_json("outputs/embeddings/embedding_config.json", {})
    examples = read_json("outputs/reports/retrieval_examples.json", [])
    theorem_case_studies = read_json("outputs/reports/theorem_retrieval_case_studies.json", [])
    domains = read_json("outputs/reports/domain_coverage.json", {})
    raw_schema = read_json("outputs/reports/raw_schema.json", {})
    corpus_manifest = read_json("outputs/reports/corpus_manifest.json", {})
    artifact_compatibility = read_json("outputs/reports/artifact_compatibility_report.json", {})
    index_benchmark = read_json("outputs/reports/index_benchmark.json", {})
    premise_supervision = read_json("outputs/reports/premise_trace_supervision_report.json", {})
    pipeline_run_timings = read_json("outputs/reports/pipeline_run_timings.json", {})
    pipeline_performance = read_json("outputs/reports/pipeline_performance_report.json", {})
    test_set_evaluation = read_json("outputs/reports/test_set_evaluation.json", {})
    validation = {
        "schema": read_json("outputs/reports/schema_validation_summary.json", {}),
        "split_leakage": read_json("outputs/reports/split_leakage_report.json", {}),
        "context": read_json("outputs/reports/context_parse_coverage.json", {}),
        "theorem_query_parse": read_json("outputs/reports/theorem_query_parse_coverage.json", {}),
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
        "corpus_manifest": corpus_manifest,
    }
    refresh_dashboard = _refresh_dashboard(
        corpus_manifest=corpus_manifest,
        graph_stats=graph_stats,
        domains=domains,
        metrics=metrics,
        validation=validation,
        artifact_compatibility=artifact_compatibility,
        index_benchmark=index_benchmark,
        difficulty_estimator_metrics=difficulty_estimator_metrics,
    )
    refresh_trend = _refresh_trend(refresh_dashboard, previous_refresh_dashboard)
    refresh_history = _refresh_history(previous_refresh_history, refresh_dashboard, refresh_trend)
    refresh_dashboard["trend"] = {
        "has_previous": refresh_trend["has_previous"],
        "deltas": refresh_trend["deltas"],
        "quality_gate_changes": refresh_trend["quality_gate_changes"],
        "history_entry_count": refresh_history["entry_count"],
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
        "corpus_manifest": corpus_manifest,
        "graph_visualization": _graph_visualization_sample(),
        "functions": [
            "retrieve_premises",
            "retrieve_premises_for_query",
            "retrieve_similar_theorems",
            "retrieve_similar_theorems_for_query",
            "retrieve_knowledge_for_theorem",
            "explain_premise_match",
            "get_proof_technique_labels",
            "get_difficulty_profile",
            "get_graph_neighborhood",
        ],
        "retrieval_examples": examples,
        "theorem_case_studies": theorem_case_studies,
        "proof_techniques": _csv_records("outputs/reports/proof_technique_distribution.csv"),
        "difficulty_distribution": _csv_records("outputs/reports/difficulty_distribution.csv"),
        "metrics": metrics,
        "production_evidence": _production_evidence(
            metrics=metrics,
            premise_supervision=premise_supervision,
            pipeline_run_timings=pipeline_run_timings,
            pipeline_performance=pipeline_performance,
            test_set_evaluation=test_set_evaluation,
        ),
        "difficulty_estimator_metrics": difficulty_estimator_metrics,
        "refresh_dashboard": refresh_dashboard,
        "refresh_trend": refresh_trend,
        "refresh_history": refresh_history,
        "validation": validation,
        "theorem_query_parse_coverage": validation["theorem_query_parse"],
        "commands": [
            "conda activate leanrank_kg",
            "make install",
            "leanrank-kg full-pipeline --config configs/proofatlas.yaml",
            "leanrank-kg build-index --config configs/proofatlas.yaml",
            "leanrank-kg retrieve-theorem-guidance --theorem-text 'theorem ...'",
            "leanrank-kg build-homepage --config configs/proofatlas.yaml",
        ],
    }
    summary["overview"]["max_domain_count"] = max((int(row["count"]) for row in summary["top_domains"]), default=1)
    summary["overview"]["max_technique_count"] = max((int(row.get("count", 0)) for row in summary["proof_techniques"]), default=1)
    summary["overview"]["max_difficulty_count"] = max((int(row.get("count", 0)) for row in summary["difficulty_distribution"]), default=1)
    write_json("outputs/reports/homepage_summary.json", summary)
    write_json("outputs/reports/refresh_dashboard.json", summary["refresh_dashboard"])
    write_json("outputs/reports/refresh_trend.json", summary["refresh_trend"])
    write_json("outputs/reports/refresh_history.json", summary["refresh_history"])
    return summary


def run(config_path: str | None = None) -> None:
    build_summary(config_path)
