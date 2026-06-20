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


def _index_artifact_from_manifest(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": path, "passed": False, "detail": "manifest missing"}
    manifest = read_json(p, {}) or {}
    index_path = Path(manifest.get("index_path") or "")
    return {
        "path": str(index_path),
        "passed": bool(index_path.exists()),
        "detail": f"backend={manifest.get('backend')}; format={manifest.get('index_format')}; build_seconds={manifest.get('build_seconds')}",
    }


def _homepage_contains(tokens: list[str]) -> dict[str, Any]:
    path = Path("homepage/index.html")
    if not path.exists():
        return {"path": str(path), "passed": False, "detail": "missing"}
    text = path.read_text(encoding="utf-8")
    missing = [token for token in tokens if token not in text]
    return {"path": str(path), "passed": not missing, "detail": f"missing={missing}"}


def _file_contains(path: str, tokens: list[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "passed": False, "detail": "missing"}
    text = p.read_text(encoding="utf-8")
    missing = [token for token in tokens if token not in text]
    return {"path": str(p), "passed": not missing, "detail": f"missing={missing}"}


def _resource_parallelism_condition(data: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    profile = data.get("throughput_profile", {}).get("resource_parallelism_profile", {})
    embedding = profile.get("embedding_parallelism", {})
    evaluation = profile.get("evaluation_parallelism", {})
    index = profile.get("index_parallelism", {})
    passed = (
        bool(profile)
        and bool(embedding.get("backend"))
        and "device_count" in embedding
        and "multi_process" in embedding
        and "actual_backends" in evaluation
        and bool(index.get("backend"))
        and "indexed_entities" in index
    )
    return passed, profile


def build_audit() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    required_root = ["README.md", "pyproject.toml", "Makefile", "configs/sample.yaml", "homepage/index.html"]
    for path in required_root:
        checks[f"file:{path}"] = _exists(path)
    checks["validation:readme_delivery_evidence"] = _file_contains(
        "README.md",
        [
            "erbacher/LeanRank-data",
            "Current retrieval failure diagnosis",
            "candidate_pool_miss_top_100",
            "Current resource and parallelism profile",
            "hnswlib parameters",
            "make verify-delivery",
        ],
    )
    for module in ["api", "benchmark_index", "build_index", "deployment_security", "experiment_report", "lean_check", "pipeline_profile", "pipeline_timing", "premise_trace_supervision", "query"]:
        checks[f"source:{module}"] = _exists(f"src/leanrank_kg/{module}.py")
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
        checks[f"index_manifest:{split}:proof_state"] = _exists(f"outputs/indexes/{split}_proof_state_index_manifest.json")
        checks[f"index_manifest:{split}:premise"] = _exists(f"outputs/indexes/{split}_premise_index_manifest.json")
        checks[f"index_manifest:{split}:theorem"] = _exists(f"outputs/indexes/{split}_theorem_index_manifest.json")
        checks[f"index:{split}:proof_state"] = _index_artifact_from_manifest(f"outputs/indexes/{split}_proof_state_index_manifest.json")
        checks[f"index:{split}:premise"] = _index_artifact_from_manifest(f"outputs/indexes/{split}_premise_index_manifest.json")
        checks[f"index:{split}:theorem"] = _index_artifact_from_manifest(f"outputs/indexes/{split}_theorem_index_manifest.json")
    for path in [
        "outputs/reports/raw_schema.json",
        "outputs/reports/corpus_manifest.json",
        "outputs/reports/artifact_compatibility_report.json",
        "outputs/reports/domain_distribution.json",
        "outputs/reports/metrics.json",
        "outputs/reports/test_set_evaluation.json",
        "outputs/reports/experiment_report.md",
        "outputs/reports/premise_trace_supervision_report.json",
        "outputs/reports/retrieval_examples.json",
        "outputs/reports/theorem_retrieval_case_studies.json",
        "outputs/reports/retrieval_examples.md",
        "outputs/reports/proof_technique_distribution.csv",
        "outputs/reports/difficulty_distribution.csv",
        "outputs/reports/theorem_query_parse_coverage.json",
        "outputs/reports/ranker_validation_metrics.json",
        "outputs/reports/difficulty_estimator_metrics.json",
        "outputs/reports/index_benchmark.json",
        "outputs/reports/lean_diagnostic_extraction_report.json",
        "outputs/reports/pipeline_run_timings.json",
        "outputs/reports/pipeline_performance_report.json",
        "outputs/reports/deployment_security_review.json",
        "outputs/reports/homepage_summary.json",
        "outputs/reports/refresh_dashboard.json",
        "outputs/reports/refresh_trend.json",
        "outputs/reports/refresh_history.json",
        "outputs/models/premise_ranker.joblib",
        "outputs/models/difficulty_estimator.joblib",
        "outputs/indexes/index_summary.json",
        "homepage/assets/theorem_retrieval_case_studies.json",
        "homepage/assets/graph_visualization.json",
        "homepage/assets/refresh_dashboard.json",
        "homepage/assets/refresh_trend.json",
        "homepage/assets/refresh_history.json",
        "docs/proofatlas_deployment_guide.md",
        "docs/proofatlas_project_summary_en.md",
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
    checks["validation:corpus_manifest"] = _json_condition(
        "outputs/reports/corpus_manifest.json",
        lambda data: (
            bool(data.get("dataset_name"))
            and bool(data.get("source_kind"))
            and bool(data.get("config_hash"))
            and bool(data.get("split_counts"))
            and bool(data.get("data_supervision", {}).get("kind"))
            and bool(data.get("corpus", {}).get("corpus_version"))
            and bool(data.get("corpus", {}).get("extraction_config_hash"))
            and bool(data.get("corpus", {}).get("lean_version"))
            and bool(data.get("corpus", {}).get("mathlib_commit"))
            and bool(data.get("corpus", {}).get("source_revision")),
            {
                "source_kind": data.get("source_kind"),
                "data_supervision": data.get("data_supervision", {}).get("kind"),
                "config_hash": data.get("config_hash"),
                "corpus_version": data.get("corpus", {}).get("corpus_version"),
                "extraction_config_hash": data.get("corpus", {}).get("extraction_config_hash"),
                "splits": sorted(data.get("split_counts", {}).keys()),
            },
        ),
        "corpus_manifest={detail}",
    )
    checks["validation:artifact_compatibility"] = _json_condition(
        "outputs/reports/artifact_compatibility_report.json",
        lambda data: (
            data.get("passed") is True
            and data.get("config_hash_matches") is True
            and all(split.get("passed") is True for split in data.get("splits", {}).values()),
            {
                "config_hash_matches": data.get("config_hash_matches"),
                "failures": data.get("failures"),
                "warnings": data.get("warnings"),
                "data_supervision": data.get("data_supervision", {}).get("kind"),
                "splits": sorted(data.get("splits", {}).keys()),
            },
        ),
        "artifact_compatibility={detail}",
    )
    checks["validation:graph_endpoints"] = _json_condition(
        "outputs/reports/graph_validation_summary.json",
        lambda data: (
            all(row.get("missing_endpoint_count") == 0 and row.get("networkx_loadable") for row in data.values()),
            {split: row.get("missing_endpoint_count") for split, row in data.items()},
        ),
        "missing_endpoint_counts={detail}",
    )
    checks["validation:theorem_query_parse_coverage"] = _json_condition(
        "outputs/reports/theorem_query_parse_coverage.json",
        lambda data: (
            bool(data)
            and all(row.get("parse_coverage", 0.0) > 0.0 for row in data.values())
            and all("avg_operator_symbol_count" in row for row in data.values()),
            {
                split: {
                    "parse_coverage": row.get("parse_coverage"),
                    "avg_operator_symbol_count": row.get("avg_operator_symbol_count"),
                    "failure_count": row.get("failure_count"),
                }
                for split, row in data.items()
            },
        ),
        "theorem_query_parse={detail}",
    )
    checks["validation:retrieval_examples"] = _json_condition(
        "outputs/reports/retrieval_examples.json",
        lambda data: (len(data) >= 20, len(data)),
        "examples={detail}",
    )
    checks["validation:theorem_retrieval_metrics"] = _json_condition(
        "outputs/reports/metrics.json",
        lambda data: (
            data.get("theorem_retrieval_evaluated_theorems", 0) > 0
            and data.get("theorem_retrieval_gold_premises_total", -1)
            == data.get("theorem_retrieval_gold_premises_in_train_index", 0)
            + data.get("theorem_retrieval_gold_premises_missing_from_train_index", 0),
            {
                "evaluated": data.get("theorem_retrieval_evaluated_theorems"),
                "total_gold": data.get("theorem_retrieval_gold_premises_total"),
            },
        ),
        "theorem_retrieval={detail}",
    )
    checks["validation:test_set_evaluation"] = _json_condition(
        "outputs/reports/test_set_evaluation.json",
        lambda data: (
            data.get("candidate_pool") == "train premise index"
            and "held-out test" in str(data.get("label_policy", ""))
            and data.get("test", {}).get("proof_state_retrieval", {}).get("metrics", {}).get("evaluated_queries", 0) > 0
            and data.get("test", {}).get("theorem_retrieval", {}).get("metrics", {}).get("theorem_retrieval_evaluated_theorems", 0) > 0
            and bool(data.get("test", {}).get("proof_state_retrieval", {}).get("domain_breakdown"))
            and bool(data.get("test", {}).get("theorem_retrieval", {}).get("domain_breakdown"))
            and bool(data.get("test", {}).get("proof_state_retrieval", {}).get("worst_cases"))
            and bool(data.get("test", {}).get("theorem_retrieval", {}).get("worst_cases"))
            and all(
                key in data.get("test", {}).get("proof_state_retrieval", {}).get("metrics", {})
                for key in ["Recall@1", "Recall@5", "Recall@10", "MRR", "MAP", "nDCG@10", "gold_premise_coverage"]
            )
            and all(
                key in data.get("test", {}).get("theorem_retrieval", {}).get("metrics", {})
                for key in [
                    "theorem_retrieval_Recall@1",
                    "theorem_retrieval_Recall@5",
                    "theorem_retrieval_Recall@10",
                    "theorem_retrieval_MRR",
                    "theorem_retrieval_MAP",
                    "theorem_retrieval_nDCG@10",
                    "theorem_retrieval_gold_premise_coverage",
                ]
            ),
            {
                "candidate_pool": data.get("candidate_pool"),
                "proof_state_metrics": data.get("test", {}).get("proof_state_retrieval", {}).get("metrics", {}),
                "theorem_metrics": data.get("test", {}).get("theorem_retrieval", {}).get("metrics", {}),
                "proof_state_domains": data.get("test", {}).get("proof_state_retrieval", {}).get("domain_breakdown", []),
                "theorem_domains": data.get("test", {}).get("theorem_retrieval", {}).get("domain_breakdown", []),
                "proof_state_worst_cases": data.get("test", {}).get("proof_state_retrieval", {}).get("worst_cases", []),
                "theorem_worst_cases": data.get("test", {}).get("theorem_retrieval", {}).get("worst_cases", []),
            },
        ),
        "test_set_evaluation={detail}",
    )
    checks["validation:premise_trace_supervision"] = _json_condition(
        "outputs/reports/premise_trace_supervision_report.json",
        lambda data: (
            data.get("dataset_name") == "erbacher/LeanRank-data"
            and data.get("current_artifact_supervision", {}).get("has_positive_edges") is True
            and data.get("current_artifact_supervision", {}).get("has_negative_candidates") is True
            and data.get("current_artifact_supervision", {}).get("total_positive_edges", 0) > 0
            and data.get("current_artifact_supervision", {}).get("total_negative_edges", 0) > 0
            and data.get("splits", {}).get("train", {}).get("trace_profile", {}).get("positive_trace_rows", 0) > 0
            and data.get("splits", {}).get("train", {}).get("trace_profile", {}).get("negative_candidate_rows", 0) > 0
            and bool(data.get("splits", {}).get("train", {}).get("example_traces"))
            and data.get("scope") == "erbacher/LeanRank-data normalized positive/negative premise supervision",
            {
                "dataset_name": data.get("dataset_name"),
                "current": data.get("current_artifact_supervision"),
                "train_trace_profile": data.get("splits", {}).get("train", {}).get("trace_profile"),
                "train_example_trace_count": len(data.get("splits", {}).get("train", {}).get("example_traces", [])),
                "scope": data.get("scope"),
            },
        ),
        "premise_trace_supervision={detail}",
    )
    checks["validation:lean_diagnostic_extraction"] = _json_condition(
        "outputs/reports/lean_diagnostic_extraction_report.json",
        lambda data: (
            data.get("method") == "lean_unsolved_goals_diagnostic"
            and data.get("quality_checks", {}).get("all_cases_passed") is True
            and data.get("quality_checks", {}).get("has_successful_extraction_case") is True
            and data.get("quality_checks", {}).get("has_failure_explanation_case") is True
            and data.get("total_extracted_proof_states", 0) > 0
            and "not a corpus extractor" in data.get("production_pipeline_role", ""),
            {
                "scope": data.get("scope"),
                "case_count": data.get("case_count"),
                "passed_case_count": data.get("passed_case_count"),
                "total_extracted_proof_states": data.get("total_extracted_proof_states"),
                "pipeline_role": data.get("production_pipeline_role"),
            },
        ),
        "lean_diagnostic_extraction={detail}",
    )
    checks["validation:experiment_report"] = _file_contains(
        "outputs/reports/experiment_report.md",
        [
            "ProofAtlas Experiment Report",
            "erbacher/LeanRank-data",
            "Held-Out Test Set Metrics",
            "Domain Breakdown",
            "Error Analysis",
            "failure diagnosis",
            "candidate_pool_miss",
            "reranking_headroom_after_top10",
            "Worst Proof-State Queries",
            "Worst Theorem Queries",
            "Final Artifacts",
            "ML Task Definition",
            "Proof-State-Level Premise Ranking",
            "Theorem-Level Premise Ranking",
            "Candidate pool",
            "homepage/index.html",
            "test_set_evaluation.json",
            "Data supervision",
            "Premise Trace Supervision",
            "Positive edges",
            "Negative candidates",
            "LeanRank premise supervision ready",
            "Recall@10",
            "MRR",
            "MAP",
            "nDCG@10",
        ],
    )
    checks["validation:index_summary"] = _json_condition(
        "outputs/indexes/index_summary.json",
        lambda data: (
            bool(data.get("backend"))
            and bool(data.get("metric"))
            and bool(data.get("embedding_config_hash"))
            and bool(data.get("corpus", {}).get("extraction_config_hash"))
            and data.get("build_seconds") is not None
            and all(
                all(
                    index.get("manifest_path")
                    and index.get("embedding_config_hash") == data.get("embedding_config_hash")
                    and index.get("extraction_config_hash") == data.get("corpus", {}).get("extraction_config_hash")
                    and index.get("build_seconds") is not None
                    for index in split.get("indexes", [])
                )
                for split in data.get("splits", {}).values()
            ),
            {
                "backend": data.get("backend"),
                "metric": data.get("metric"),
                "corpus": data.get("corpus"),
                "splits": sorted(data.get("splits", {}).keys()),
            },
        ),
        "index_summary={detail}",
    )
    checks["validation:index_benchmark"] = _json_condition(
        "outputs/reports/index_benchmark.json",
        lambda data: (
            bool(data.get("entities"))
            and all(
                row.get("rows", 0) > 0
                and row.get("exact_ms_per_query") is not None
                and (
                    not row.get("indexed_available")
                    or row.get(f"recall_at_{row.get('top_k')}_vs_exact") is not None
                )
                for row in data.get("entities", {}).values()
            ),
            {
                key: {
                    "backend": row.get("backend"),
                    "indexed_available": row.get("indexed_available"),
                    "exact_ms_per_query": row.get("exact_ms_per_query"),
                    "indexed_ms_per_query": row.get("indexed_ms_per_query"),
                }
                for key, row in data.get("entities", {}).items()
            },
        ),
        "index_benchmark={detail}",
    )
    checks["validation:pipeline_run_timings"] = _json_condition(
        "outputs/reports/pipeline_run_timings.json",
        lambda data: (
            data.get("passed") is True
            and data.get("total_seconds") is not None
            and data.get("stage_count", 0) >= 15
            and {"sample", "normalize", "evaluate", "pipeline_profile"}
            <= {row.get("name") for row in data.get("stages", [])},
            {
                "passed": data.get("passed"),
                "total_seconds": data.get("total_seconds"),
                "stage_count": data.get("stage_count"),
                "stages": [row.get("name") for row in data.get("stages", [])],
            },
        ),
        "pipeline_run_timings={detail}",
    )
    checks["validation:pipeline_performance_report"] = _json_condition(
        "outputs/reports/pipeline_performance_report.json",
        lambda data: (
            data.get("dataset_name") == "erbacher/LeanRank-data"
            and bool(data.get("scale_profile"))
            and data.get("scale_profile", {}).get("target_dataset_confirmed") is True
            and bool(data.get("recommendations"))
            and {"sample", "processed", "graph", "embeddings", "indexes", "benchmark", "timings", "evaluation", "readiness"}
            <= set(data.get("stages", {}).keys()),
            {
                "dataset_name": data.get("dataset_name"),
                "scale_profile": data.get("scale_profile"),
                "recommendations": data.get("recommendations"),
                "stage_keys": sorted(data.get("stages", {}).keys()),
            },
        ),
        "pipeline_performance={detail}",
    )
    checks["validation:resource_parallelism_profile"] = _json_condition(
        "outputs/reports/pipeline_performance_report.json",
        _resource_parallelism_condition,
        "resource_parallelism_profile={detail}",
    )
    checks["validation:performance_acceptance_profile"] = _json_condition(
        "outputs/reports/pipeline_performance_report.json",
        lambda data: (
            bool(data.get("throughput_profile", {}).get("performance_acceptance_profile", {}).get("gates"))
            and data.get("throughput_profile", {})
            .get("performance_acceptance_profile", {})
            .get("summary", {})
            .get("total_gate_count", 0)
            > 0
            and (
                data.get("scale_profile", {}).get("scale_bucket") != "large"
                or data.get("throughput_profile", {})
                .get("performance_acceptance_profile", {})
                .get("summary", {})
                .get("required_gates_passed")
                is True
            ),
            {
                "scale_bucket": data.get("scale_profile", {}).get("scale_bucket"),
                "profile": data.get("throughput_profile", {}).get("performance_acceptance_profile", {}),
            },
        ),
        "performance_acceptance_profile={detail}",
    )
    checks["validation:refresh_dashboard"] = _json_condition(
        "outputs/reports/refresh_dashboard.json",
        lambda data: (
            bool(data.get("scale", {}).get("total_train_val_test"))
            and bool(data.get("domain_coverage", {}).get("top_domains"))
            and bool(data.get("retrieval_quality"))
            and bool(data.get("parsing"))
            and bool(data.get("quality_gates"))
            and bool(data.get("trend"))
            and "ready_for_refresh_comparison" in data,
            {
                "ready": data.get("ready_for_refresh_comparison"),
                "gates": data.get("quality_gates"),
            },
        ),
        "refresh_dashboard={detail}",
    )
    checks["validation:homepage_summary_supervision"] = _json_condition(
        "outputs/reports/homepage_summary.json",
        lambda data: (
            (data.get("production_evidence", {}).get("supervision", {}).get("total_positive_edges", 0) > 0)
            and (data.get("production_evidence", {}).get("supervision", {}).get("total_negative_edges", 0) > 0)
            and data.get("production_evidence", {}).get("supervision", {}).get("all_positive_negative_pairs_disjoint") is True
            and data.get("production_evidence", {}).get("supervision", {}).get("total_positive_negative_overlap_removed", 0) > 0
            and data.get("production_evidence", {}).get("supervision", {}).get("train_positive_negative_pair_overlap_count") == 0,
            data.get("production_evidence", {}).get("supervision", {}),
        ),
        "homepage_supervision={detail}",
    )
    checks["validation:refresh_trend"] = _json_condition(
        "outputs/reports/refresh_trend.json",
        lambda data: (
            "has_previous" in data
            and bool(data.get("current"))
            and "deltas" in data,
            {
                "has_previous": data.get("has_previous"),
                "delta_keys": sorted(data.get("deltas", {}).keys()),
            },
        ),
        "refresh_trend={detail}",
    )
    checks["validation:refresh_history"] = _json_condition(
        "outputs/reports/refresh_history.json",
        lambda data: (
            int(data.get("entry_count", 0) or 0) > 0
            and bool(data.get("latest"))
            and bool(data.get("entries"))
            and data.get("latest") == data.get("entries", [])[-1],
            {
                "entry_count": data.get("entry_count"),
                "history_limit": data.get("history_limit"),
            },
        ),
        "refresh_history={detail}",
    )
    checks["validation:difficulty_estimator"] = _json_condition(
        "outputs/reports/difficulty_estimator_metrics.json",
        lambda data: (
            data.get("train", {}).get("rows", 0) > 0
            and data.get("train", {}).get("mae") is not None
            and bool(data.get("train", {}).get("calibration_bins"))
            and bool(data.get("train", {}).get("residual_quantiles")),
            {
                "train_rows": data.get("train", {}).get("rows"),
                "train_mae": data.get("train", {}).get("mae"),
                "residual_quantiles": data.get("train", {}).get("residual_quantiles"),
            },
        ),
        "difficulty_estimator={detail}",
    )
    checks["validation:ranker_ablation"] = _json_condition(
        "outputs/reports/ranker_validation_metrics.json",
        lambda data: (
            bool(data.get("feature_columns"))
            and bool(data.get("feature_groups"))
            and (
                "feature_ablation" not in data
                or {"embedding_similarity", "namespace_domain", "proof_technique", "difficulty", "frequency", "symbol_overlap", "graph", "theorem_neighborhood"}
                <= set(data.get("feature_ablation", {}).get("groups", {}).keys())
            ),
            {
                "validation_auc": data.get("validation_auc"),
                "groups": sorted(data.get("feature_groups", {}).keys()),
                "has_ablation": "feature_ablation" in data,
            },
        ),
        "ranker_ablation={detail}",
    )
    checks["validation:deployment_security_review"] = _json_condition(
        "outputs/reports/deployment_security_review.json",
        lambda data: (
            data.get("passed") is True
            and bool(data.get("checks"))
            and any(row.get("id") == "bounded_request_text" and row.get("passed") for row in data.get("checks", []))
            and any(row.get("id") == "bounded_result_limits" and row.get("passed") for row in data.get("checks", []))
            and any(row.get("id") == "cors_not_open_by_default" and row.get("passed") for row in data.get("checks", [])),
            {
                "passed": data.get("passed"),
                "failures": data.get("failures"),
                "warnings": data.get("warnings"),
                "public_exposure_ready": data.get("public_exposure_ready"),
            },
        ),
        "deployment_security={detail}",
    )
    checks["validation:theorem_case_studies"] = _json_condition(
        "outputs/reports/theorem_retrieval_case_studies.json",
        lambda data: (
            bool(data)
            and bool(data[0].get("guidance", {}).get("ranked_premises"))
            and bool(data[0]["guidance"]["ranked_premises"][0].get("ranking_reasons")),
            len(data),
        ),
        "case_studies={detail}",
    )
    checks["validation:graph_visualization"] = _json_condition(
        "homepage/assets/graph_visualization.json",
        lambda data: (bool(data.get("nodes")) and bool(data.get("edges")), {"nodes": len(data.get("nodes", [])), "edges": len(data.get("edges", []))}),
        "graph_sample={detail}",
    )
    checks["validation:homepage_sections"] = _homepage_contains(
        [
            "Executive Snapshot",
            "Interactive Proof Guidance Workbench",
            "Knowledge Graph Overview",
            "Edge Types",
            "Click a graph node or edge",
            "New Theorem Proof Guidance",
            "Proof Guidance Panel",
            "Why Were These Premises Recommended?",
            "Evaluation And Examples",
            "Retrieval Failure Profile",
            "Proof-state candidate miss",
            "Theorem rerank headroom",
            "Label disjoint",
            "Train label overlap",
            "Local asset fallback",
            "renderLocalFallback",
            "Query theorem",
            "Top premise",
            "Similar theorem",
            "Suggested technique",
            "shared namespace",
            "Graph evidence for",
            "Refresh Dashboard",
            "Live Retrieval Examples",
            "Proof Technique Labels",
            "Pipeline Summary",
            "Reproducibility",
        ]
    )
    checks["validation:deployment_guide"] = _file_contains(
        "docs/proofatlas_deployment_guide.md",
        [
            "Static Homepage Review",
            "Local Interactive API Demo",
            "Readiness-Gated Server Mode",
            "Notebook Demo",
            "GitHub Pages / Hosted Static Demo",
            "GET /metrics",
            "GET /metrics/prometheus",
            "--require-ready",
            "security-review",
            "deployment_security_review.json",
        ],
    )
    checks["validation:project_summary_evidence"] = _file_contains(
        "docs/proofatlas_project_summary_en.md",
        [
            "Current Production Evidence",
            "erbacher/LeanRank-data",
            "candidate_pool_miss_top_100",
            "7 CUDA devices",
            "`torch_cuda` batched top-k",
            "custom Lean server/source extractor",
        ],
    )
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
