from __future__ import annotations

import json
from pathlib import Path


def _readme_metric(readme: str, task: str, metric: str) -> str:
    prefix = f"| {task} | {metric} | "
    for line in readme.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).removesuffix(" |").strip()
    raise AssertionError(f"Missing README metric row: {task} / {metric}")


def _readme_artifact_field(readme: str, artifact: str, field: str) -> str:
    prefix = f"| {artifact} | {field} | "
    for line in readme.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).removesuffix(" |").strip()
    raise AssertionError(f"Missing README artifact row: {artifact} / {field}")


def _readme_table_row(readme: str, first_col: str, second_col: str | None = None) -> list[str]:
    prefix = f"| {first_col} |"
    for line in readme.splitlines():
        if not line.startswith(prefix):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if second_col is None or (len(cells) > 1 and cells[1] == second_col):
            return cells
    suffix = f" / {second_col}" if second_col is not None else ""
    raise AssertionError(f"Missing README row: {first_col}{suffix}")


def _metric_uncertainty_row(text: str, task: str, metric: str) -> list[str]:
    prefix = f"| {task} |"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[1] == metric:
            return cells
    raise AssertionError(f"Missing metric uncertainty row: {task} / {metric}")


def test_readme_results_snapshot_matches_committed_metrics() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    metrics = json.loads((repo / "outputs/reports/metrics.json").read_text(encoding="utf-8"))
    expected = {
        ("Proof-state premise retrieval", "Recall@10"): metrics["Recall@10"],
        ("Proof-state premise retrieval", "Recall@100"): metrics["Recall@100"],
        ("Reranked proof-state diagnostic", "Recall@10"): metrics["reranked_proof_state_Recall@10"],
        ("Theorem-level premise retrieval", "Recall@10"): metrics["theorem_retrieval_Recall@10"],
        ("Theorem-level premise retrieval", "Recall@100"): metrics["theorem_retrieval_Recall@100"],
        ("Theorem-level premise retrieval", "MRR"): metrics["theorem_retrieval_MRR"],
        ("Learned premise ranker", "validation AUC"): metrics["AUC"],
    }

    for (task, metric), value in expected.items():
        assert _readme_metric(readme, task, metric) == f"{float(value):.4f}"


def test_readme_metric_uncertainty_matches_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    uncertainty = profile["throughput_profile"]["metric_uncertainty_profile"]
    expected = {
        ("Proof-state premise retrieval", "Recall@10"): uncertainty["proof_state"]["Recall@10"],
        ("Proof-state premise retrieval", "Recall@100"): uncertainty["proof_state"]["Recall@100"],
        ("Theorem-level premise retrieval", "Recall@10"): uncertainty["theorem"]["theorem_retrieval_Recall@10"],
        ("Theorem-level premise retrieval", "Recall@100"): uncertainty["theorem"]["theorem_retrieval_Recall@100"],
    }

    for (task, metric), row in expected.items():
        cells = _metric_uncertainty_row(readme, task, metric)
        assert cells[2] == str(int(row["n"]))
        assert cells[3] == f"{float(row['ci95_low']):.4f}"
        assert cells[4] == f"{float(row['ci95_high']):.4f}"
        assert cells[5] == f"{float(row['ci95_half_width']):.4f}"


def test_readme_failure_diagnosis_matches_committed_homepage_summary() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    summary = json.loads((repo / "outputs/reports/homepage_summary.json").read_text(encoding="utf-8"))
    diagnosis = summary["production_evidence"]["failure_diagnosis"]
    expected = {
        ("Proof-state premise retrieval", "candidate_pool_miss_top_100"): diagnosis["proof_state"]["candidate_pool_miss_top_100"],
        ("Proof-state premise retrieval", "reranking_headroom_after_top10"): diagnosis["proof_state"]["reranking_headroom_after_top10"],
        ("Theorem-level premise retrieval", "candidate_pool_miss_top_100"): diagnosis["theorem"]["candidate_pool_miss_top_100"],
        ("Theorem-level premise retrieval", "reranking_headroom_after_top10"): diagnosis["theorem"]["reranking_headroom_after_top10"],
    }

    for (task, label), row in expected.items():
        cells = _readme_table_row(readme, task, label)
        assert cells[2] == f"{int(row['queries']):,}"
        assert cells[3] == f"{float(row['share_of_evaluated']):.1%}"
        assert cells[4] == f"{float(row['share_of_retrievable']):.1%}"


def test_readme_production_snapshot_matches_committed_artifacts() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    timing = json.loads((repo / "outputs/reports/pipeline_run_timings.json").read_text(encoding="utf-8"))
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    coverage = profile["stages"]["evaluation"]["held_out_test_coverage"]
    throughput = profile["throughput_profile"]

    assert _readme_artifact_field(readme, "Held-out proof-state evaluation", "coverage") == (
        f"{coverage['proof_state_evaluated_queries']} / {coverage['proof_state_total']}"
    )
    assert _readme_artifact_field(readme, "Held-out theorem evaluation", "coverage") == (
        f"{coverage['theorem_evaluated_queries']} / {coverage['theorem_total']}"
    )
    assert _readme_artifact_field(readme, "Pipeline timing", "total seconds") == f"{float(timing['total_seconds']):.4f}"
    assert _readme_artifact_field(readme, "Pipeline timing", "executed/skipped stages") == (
        f"{timing['executed_stage_count']} / {timing['skipped_stage_count']}"
    )
    assert _readme_artifact_field(readme, "Pipeline timing", "throughput basis") == throughput["throughput_basis"]
    assert _readme_artifact_field(readme, "Pipeline timing", "scale estimate reliable") == str(throughput["scale_estimate_reliable"])
    evaluation_delta = throughput["evaluation_timing_delta"]
    assert _readme_artifact_field(readme, "Pipeline timing", "saved evaluate seconds") == (
        f"{float(evaluation_delta['timed_pipeline_evaluate_seconds']):.4f}"
    )
    assert _readme_artifact_field(readme, "Pipeline timing", "current standalone evaluation seconds") == (
        f"{float(evaluation_delta['current_evaluation_seconds']):.4f}"
    )
    assert _readme_artifact_field(readme, "Pipeline timing", "timed/current evaluation ratio") == (
        f"{float(evaluation_delta['timed_to_current_ratio']):.4f}"
    )
    rerank_cost = throughput["rerank_evaluation_cost_profile"]
    assert _readme_artifact_field(readme, "Rerank diagnostic cost", "sampled/full proof-state queries") == (
        f"{int(rerank_cost['sampled_rerank_queries'])} / {int(rerank_cost['full_proof_state_queries'])}"
    )
    assert _readme_artifact_field(readme, "Rerank diagnostic cost", "sampled fraction") == (
        f"{float(rerank_cost['sampled_fraction_of_full_proof_state_eval']):.4f}"
    )
    assert _readme_artifact_field(readme, "Rerank diagnostic cost", "projected full rerank seconds") == (
        f"{float(rerank_cost['projected_full_rerank_seconds']):.4f}"
    )
    assert _readme_artifact_field(readme, "Rerank diagnostic cost", "rerank/batched seconds per query") == (
        f"{float(rerank_cost['rerank_to_batched_seconds_per_query_ratio']):.4f}"
    )


def test_readme_performance_snapshot_matches_committed_artifacts() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    benchmark = json.loads((repo / "outputs/reports/index_benchmark.json").read_text(encoding="utf-8"))
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))

    display_names = {"premise": "Premise", "proof_state": "ProofState", "theorem": "Theorem"}
    for key, label in display_names.items():
        row = benchmark["entities"][key]
        cells = _readme_table_row(readme, label)
        assert cells[2] == f"{int(row['rows']):,}"
        assert cells[3] == f"{float(row['exact_ms_per_query']):.4f}"
        assert cells[4] == f"{float(row['indexed_ms_per_query']):.4f}"
        assert cells[5] == f"{float(row['speedup_vs_exact']):.4f}"
        assert cells[6] == f"{float(row['recall_at_10_vs_exact']):.4f}"

    bottleneck = profile["throughput_profile"]["bottleneck_profile"]
    assert _readme_table_row(readme, "Primary bottleneck", "stage")[2] == bottleneck["primary_stage"]
    assert _readme_table_row(readme, "Primary bottleneck", "seconds")[2] == f"{float(bottleneck['primary_stage_seconds']):.4f}"
    assert _readme_table_row(readme, "Primary bottleneck", "share of total")[2] == f"{float(bottleneck['primary_stage_share_of_total']):.4f}"
    assert _readme_table_row(readme, "Top-3 timed stages", "share of total")[2] == f"{float(bottleneck['top3_stage_share_of_total']):.4f}"


def test_readme_resource_parallelism_profile_matches_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    resources = profile["throughput_profile"]["resource_parallelism_profile"]
    embedding = resources["embedding_parallelism"]
    evaluation = resources["evaluation_parallelism"]
    indexing = resources["index_parallelism"]

    assert _readme_table_row(readme, "Embedding", "backend/model")[2] == (
        f"{embedding['backend']} / {embedding['model_name']}"
    )
    assert _readme_table_row(readme, "Embedding", "device count")[2] == str(embedding["device_count"])
    assert _readme_table_row(readme, "Embedding", "multi-process")[2] == str(embedding["multi_process"])
    assert _readme_table_row(readme, "Embedding", "batch size")[2] == str(embedding["batch_size"])
    assert _readme_table_row(readme, "Embedding", "rows/sec during embed stage")[2] == (
        f"{float(embedding['embedding_rows_per_embed_second']):.4f}"
    )
    assert _readme_table_row(readme, "Evaluation", "actual backends")[2] == ", ".join(evaluation["actual_backends"])
    assert _readme_table_row(readme, "Evaluation", "candidate count")[2] == f"{int(evaluation['candidate_count']):,}"
    assert _readme_table_row(readme, "Indexing", "backend")[2] == indexing["backend"]
    assert _readme_table_row(readme, "Indexing", "hnswlib parameters")[2] == (
        f"M={indexing['hnsw_M']}, ef_construction={indexing['hnsw_ef_construction']}, ef_search={indexing['hnsw_ef_search']}"
    )
    assert _readme_table_row(readme, "Indexing", "min recall vs exact")[2] == f"{float(indexing['min_recall_vs_exact']):.4f}"


def test_readme_execution_mode_summary_matches_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    summary = profile["throughput_profile"]["execution_mode_summary"]

    assert _readme_table_row(readme, "Embedding mode", "value")[2] == summary["embedding_mode"]
    assert _readme_table_row(readme, "Embedding GPU", "active")[2] == str(summary["embedding_gpu_active"])
    assert _readme_table_row(readme, "Embedding GPU", "multi GPU")[2] == str(summary["multi_gpu_embedding"])
    assert _readme_table_row(readme, "Evaluation mode", "value")[2] == summary["evaluation_mode"]
    assert _readme_table_row(readme, "Evaluation GPU", "active")[2] == str(summary["evaluation_gpu_active"])
    assert _readme_table_row(readme, "Index mode", "value")[2] == summary["index_mode"]
    assert _readme_table_row(readme, "ANN index", "active")[2] == str(summary["ann_index_active"])
    assert _readme_table_row(readme, "Primary bottleneck", "stage")[2] == summary["primary_timed_bottleneck"]
    assert _readme_table_row(readme, "Artifact reuse", "default")[2] == str(summary["artifact_reuse_by_default"])
    assert summary["bottleneck_interpretation"] in readme


def test_readme_performance_acceptance_gates_match_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    summary = profile["throughput_profile"]["performance_acceptance_profile"]["summary"]

    assert _readme_table_row(readme, "Required gates", "passed")[2] == str(summary["required_gates_passed"])
    assert _readme_table_row(readme, "Advisory gates", "passed")[2] == str(summary["advisory_gates_passed"])
    assert _readme_table_row(readme, "All gates", "passed / total")[2] == (
        f"{summary['passed_gate_count']} / {summary['total_gate_count']}"
    )


def test_readme_scale_projection_matches_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    projections = profile["throughput_profile"]["scale_projection_profile"]["projections"]

    for row in projections:
        cells = _readme_table_row(readme, row["label"])
        assert cells[1] == str(int(row["target_processed_rows"]))
        assert cells[2] == f"{float(row['estimated_total_seconds']):.4f}"
        assert cells[3] == f"{float(row['estimated_embed_seconds']):.4f}"
        assert cells[4] == f"{float(row['estimated_index_build_seconds']):.4f}"


def test_readme_artifact_reuse_policy_matches_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    reuse = profile["throughput_profile"]["refresh_reuse_profile"]
    cache = reuse["artifact_cache"]

    assert _readme_artifact_field(readme, "Artifact reuse", "reuse by default") == str(reuse["reuse_by_default"])
    assert _readme_artifact_field(readme, "Embedding cache", "rows") == str(cache["embedding_rows"])
    assert _readme_artifact_field(readme, "Index cache", "entity manifests") == str(cache["indexed_entity_count"])
    assert _readme_artifact_field(readme, "Premise ranker", "exists") == str(cache["premise_ranker_exists"])
    assert _readme_artifact_field(readme, "Difficulty estimator", "exists") == str(cache["difficulty_estimator_exists"])
    assert reuse["training_repeat_policy"] in readme


def test_delivery_audit_performance_snapshot_matches_committed_artifacts() -> None:
    repo = Path(__file__).resolve().parents[1]
    audit = (repo / "docs/proofatlas_delivery_audit.md").read_text(encoding="utf-8")
    benchmark = json.loads((repo / "outputs/reports/index_benchmark.json").read_text(encoding="utf-8"))
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    timing = json.loads((repo / "outputs/reports/pipeline_run_timings.json").read_text(encoding="utf-8"))

    display_names = {"premise": "Premise", "proof_state": "ProofState", "theorem": "Theorem"}
    for key, label in display_names.items():
        row = benchmark["entities"][key]
        cells = _readme_table_row(audit, label)
        assert cells[2] == f"{int(row['rows']):,}"
        assert cells[3] == f"{float(row['exact_ms_per_query']):.4f}"
        assert cells[4] == f"{float(row['indexed_ms_per_query']):.4f}"
        assert cells[5] == f"{float(row['speedup_vs_exact']):.4f}"
        assert cells[6] == f"{float(row['recall_at_10_vs_exact']):.4f}"

    bottleneck = profile["throughput_profile"]["bottleneck_profile"]
    assert _readme_table_row(audit, "Primary bottleneck", "stage")[2] == bottleneck["primary_stage"]
    assert _readme_table_row(audit, "Primary bottleneck", "seconds")[2] == f"{float(bottleneck['primary_stage_seconds']):.4f}"
    assert _readme_table_row(audit, "Primary bottleneck", "share of total")[2] == f"{float(bottleneck['primary_stage_share_of_total']):.4f}"
    assert _readme_table_row(audit, "Top-3 timed stages", "share of total")[2] == f"{float(bottleneck['top3_stage_share_of_total']):.4f}"
    execution = profile["throughput_profile"]["execution_mode_summary"]
    assert _readme_table_row(audit, "Embedding mode", "value")[2] == execution["embedding_mode"]
    assert _readme_table_row(audit, "Embedding GPU", "active")[2] == str(execution["embedding_gpu_active"])
    assert _readme_table_row(audit, "Embedding GPU", "multi GPU")[2] == str(execution["multi_gpu_embedding"])
    assert _readme_table_row(audit, "Evaluation mode", "value")[2] == execution["evaluation_mode"]
    assert _readme_table_row(audit, "Evaluation GPU", "active")[2] == str(execution["evaluation_gpu_active"])
    assert _readme_table_row(audit, "Index mode", "value")[2] == execution["index_mode"]
    assert _readme_table_row(audit, "ANN index", "active")[2] == str(execution["ann_index_active"])
    assert _readme_table_row(audit, "Artifact reuse", "default")[2] == str(execution["artifact_reuse_by_default"])
    assert execution["bottleneck_interpretation"] in audit
    acceptance = profile["throughput_profile"]["performance_acceptance_profile"]["summary"]
    assert _readme_table_row(audit, "Required gates", "passed")[2] == str(acceptance["required_gates_passed"])
    assert _readme_table_row(audit, "Advisory gates", "passed")[2] == str(acceptance["advisory_gates_passed"])
    assert _readme_table_row(audit, "All gates", "passed / total")[2] == (
        f"{acceptance['passed_gate_count']} / {acceptance['total_gate_count']}"
    )
    for row in profile["throughput_profile"]["scale_projection_profile"]["projections"]:
        cells = _readme_table_row(audit, row["label"])
        assert cells[1] == str(int(row["target_processed_rows"]))
        assert cells[2] == f"{float(row['estimated_total_seconds']):.4f}"
        assert cells[3] == f"{float(row['estimated_embed_seconds']):.4f}"
        assert cells[4] == f"{float(row['estimated_index_build_seconds']):.4f}"
    assert f"Total seconds: {float(timing['total_seconds']):.4f}" in audit
    rerank_cost = profile["throughput_profile"]["rerank_evaluation_cost_profile"]
    assert (
        f"Reranked proof-state diagnostic: {int(rerank_cost['sampled_rerank_queries'])} / "
        f"{int(rerank_cost['full_proof_state_queries'])} sampled queries"
    ) in audit
    assert f"projected full rerank {float(rerank_cost['projected_full_rerank_seconds']):.4f} seconds" in audit
    assert f"{float(rerank_cost['rerank_to_batched_seconds_per_query_ratio']):.4f}x batched seconds/query" in audit


def test_delivery_audit_artifact_reuse_policy_matches_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    audit = (repo / "docs/proofatlas_delivery_audit.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    reuse = profile["throughput_profile"]["refresh_reuse_profile"]
    cache = reuse["artifact_cache"]

    assert _readme_artifact_field(audit, "Artifact reuse", "reuse by default") == str(reuse["reuse_by_default"])
    assert _readme_artifact_field(audit, "Embedding cache", "rows") == str(cache["embedding_rows"])
    assert _readme_artifact_field(audit, "Index cache", "entity manifests") == str(cache["indexed_entity_count"])
    assert _readme_artifact_field(audit, "Premise ranker", "exists") == str(cache["premise_ranker_exists"])
    assert _readme_artifact_field(audit, "Difficulty estimator", "exists") == str(cache["difficulty_estimator_exists"])
    assert reuse["training_repeat_policy"] in audit


def test_delivery_audit_metric_uncertainty_matches_committed_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    audit = (repo / "docs/proofatlas_delivery_audit.md").read_text(encoding="utf-8")
    profile = json.loads((repo / "outputs/reports/pipeline_performance_report.json").read_text(encoding="utf-8"))
    uncertainty = profile["throughput_profile"]["metric_uncertainty_profile"]
    expected = {
        ("Proof-state premise retrieval", "Recall@10"): uncertainty["proof_state"]["Recall@10"],
        ("Proof-state premise retrieval", "Recall@100"): uncertainty["proof_state"]["Recall@100"],
        ("Theorem premise retrieval", "Recall@10"): uncertainty["theorem"]["theorem_retrieval_Recall@10"],
        ("Theorem premise retrieval", "Recall@100"): uncertainty["theorem"]["theorem_retrieval_Recall@100"],
    }

    for (task, metric), row in expected.items():
        cells = _metric_uncertainty_row(audit, task, metric)
        assert cells[2] == str(int(row["n"]))
        assert cells[3] == f"{float(row['ci95_low']):.4f}"
        assert cells[4] == f"{float(row['ci95_high']):.4f}"
        assert cells[5] == f"{float(row['ci95_half_width']):.4f}"


def test_delivery_audit_check_count_matches_committed_audit_artifact() -> None:
    repo = Path(__file__).resolve().parents[1]
    audit_text = (repo / "docs/proofatlas_delivery_audit.md").read_text(encoding="utf-8")
    audit_json = json.loads((repo / "outputs/reports/mvp_completion_audit.json").read_text(encoding="utf-8"))
    assert audit_json["passed"] is True
    assert f"audit: {audit_json['total_checks']}/{audit_json['total_checks']} checks passed" in audit_text


def test_current_status_timing_matches_committed_artifact() -> None:
    repo = Path(__file__).resolve().parents[1]
    current = (repo / "docs/proofatlas_current_status_and_gap_to_theorem_retrieval.md").read_text(encoding="utf-8")
    timing = json.loads((repo / "outputs/reports/pipeline_run_timings.json").read_text(encoding="utf-8"))
    assert _readme_table_row(current, "Total timed pipeline run")[1] == f"{float(timing['total_seconds']):.4f} seconds"


def test_readme_premise_supervision_snapshot_matches_committed_artifacts() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    supervision = json.loads((repo / "outputs/reports/premise_trace_supervision_report.json").read_text(encoding="utf-8"))
    ranker = json.loads((repo / "outputs/reports/ranker_validation_metrics.json").read_text(encoding="utf-8"))
    current = supervision["current_artifact_supervision"]
    train = supervision["splits"]["train"]
    utilization = ranker["training_pair_utilization"]
    training_sample = utilization["training_sample_counts"]
    hardness_feature = utilization["hardness_feature"]

    assert _readme_artifact_field(readme, "Premise supervision", "total positive edges") == str(current["total_positive_edges"])
    assert _readme_artifact_field(readme, "Premise supervision", "total negative candidates") == str(current["total_negative_edges"])
    assert _readme_artifact_field(readme, "Premise supervision", "negative/positive ratio") == f"{float(current['negative_to_positive_edge_ratio']):.4f}"
    assert _readme_artifact_field(readme, "Train proof-state supervision", "positive coverage") == f"{float(train['positive_proof_state_coverage']):.4f}"
    assert _readme_artifact_field(readme, "Train proof-state supervision", "negative coverage") == f"{float(train['negative_proof_state_coverage']):.4f}"
    assert _readme_artifact_field(readme, "Train hard negatives", "hardness mean") == f"{float(train['negative_candidate_hardness']['mean']):.4f}"
    quality = train["hard_negative_quality_profile"]
    assert _readme_artifact_field(readme, "Train hard negatives", "high-hardness rows") == str(quality["high_hardness_negative_candidate_rows"])
    assert _readme_artifact_field(readme, "Train hard negatives", "high-hardness row share") == f"{float(quality['high_hardness_negative_candidate_share']):.4f}"
    assert _readme_artifact_field(readme, "Ranker training sample", "positive pairs") == str(training_sample["positive"])
    assert _readme_artifact_field(readme, "Ranker training sample", "hard negative pairs") == str(training_sample["hard_negative"])
    assert _readme_artifact_field(readme, "Ranker training sample", "hard-negative/positive ratio") == f"{float(training_sample['hard_negative_to_positive_ratio']):.4f}"
    assert _readme_artifact_field(readme, "Ranker hardness feature", "hard-negative nonzero share") == f"{float(hardness_feature['negative_pair_nonzero_share']):.4f}"


def test_delivery_audit_premise_supervision_snapshot_matches_committed_artifacts() -> None:
    repo = Path(__file__).resolve().parents[1]
    audit = (repo / "docs/proofatlas_delivery_audit.md").read_text(encoding="utf-8")
    supervision = json.loads((repo / "outputs/reports/premise_trace_supervision_report.json").read_text(encoding="utf-8"))
    ranker = json.loads((repo / "outputs/reports/ranker_validation_metrics.json").read_text(encoding="utf-8"))
    current = supervision["current_artifact_supervision"]
    conflicts = supervision["normalization_label_conflicts"]
    cells = _readme_table_row(audit, "Premise positive/negative supervision")

    assert f"{int(current['total_positive_edges']):,} positive edges" in cells[1]
    assert f"{int(current['total_negative_edges']):,} negative edges" in cells[1]
    assert f"positive/negative overlap removed: {int(conflicts['total_positive_negative_overlap_removed']):,}" in cells[1]
    quality = supervision["splits"]["train"]["hard_negative_quality_profile"]
    assert f"{int(quality['high_hardness_negative_candidate_rows']):,} high-hardness rows" in cells[1]
    training_sample = ranker["training_pair_utilization"]["training_sample_counts"]
    hardness_feature = ranker["training_pair_utilization"]["hardness_feature"]
    assert f"ranker training sample uses {int(training_sample['positive']):,} positive pairs" in cells[1]
    assert f"{int(training_sample['hard_negative']):,} hard-negative pairs" in cells[1]
    assert f"{float(hardness_feature['negative_pair_nonzero_share']):.1%} nonzero hard-negative hardness coverage" in cells[1]
    assert cells[2] == "Delivered"


def test_production_refresh_uses_configurable_pipeline_runner() -> None:
    repo = Path(__file__).resolve().parents[1]
    makefile = (repo / "Makefile").read_text(encoding="utf-8")
    readme = (repo / "README.md").read_text(encoding="utf-8")

    assert "PIPELINE_RUN ?= conda run -n leanrank_kg" in makefile
    assert "VERIFY_RUN ?= $(PIPELINE_RUN)" in makefile
    assert "$(PIPELINE_RUN) leanrank-kg evaluate --config $(PRODUCTION_CONFIG)" in makefile
    assert "$(PIPELINE_RUN) leanrank-kg audit --config $(PRODUCTION_CONFIG)" in makefile
    assert "refresh-production-timing:" in makefile
    assert "$(PIPELINE_RUN) leanrank-kg full-pipeline --config $(PRODUCTION_CONFIG) --force" in makefile
    assert '$(MAKE) refresh-production-full-eval PRODUCTION_CONFIG=$(PRODUCTION_CONFIG) PIPELINE_RUN="$(PIPELINE_RUN)"' in makefile
    assert "refresh-production-full-eval:" in makefile
    assert "$(PIPELINE_RUN) leanrank-kg evaluate --config $(PRODUCTION_CONFIG) --full-heldout" in makefile
    assert "make refresh-production-timing" in readme
    assert "make refresh-production-full-eval" in readme


def test_delivery_artifact_allowlist_includes_pipeline_timings() -> None:
    repo = Path(__file__).resolve().parents[1]
    gitignore = (repo / ".gitignore").read_text(encoding="utf-8")

    expected = [
        "!outputs/reports/context_parse_coverage.json",
        "!outputs/reports/deployment_security_review.json",
        "!outputs/reports/graph_stats_summary.json",
        "!outputs/reports/graph_validation_summary.json",
        "!outputs/reports/homepage_summary.json",
        "!outputs/reports/pipeline_run_timings.json",
        "!outputs/reports/refresh_dashboard.json",
        "!outputs/reports/refresh_history.json",
        "!outputs/reports/refresh_trend.json",
        "!outputs/reports/retrieval_examples.json",
        "!outputs/reports/schema_validation_summary.json",
        "!outputs/reports/split_leakage_report.json",
        "!outputs/reports/theorem_query_parse_coverage.json",
        "!outputs/reports/theorem_retrieval_case_studies.json",
    ]
    for pattern in expected:
        assert pattern in gitignore


def test_current_plans_keep_production_scope_on_leanrank_data() -> None:
    repo = Path(__file__).resolve().parents[1]
    plan = (repo / "docs/proofatlas_next_steps_plan.md").read_text(encoding="utf-8")
    summary = (repo / "docs/proofatlas_project_summary_en.md").read_text(encoding="utf-8")
    current = (repo / "docs/proofatlas_current_status_and_gap_to_theorem_retrieval.md").read_text(encoding="utf-8")

    for text in (plan, summary, current):
        assert "erbacher/LeanRank-data" in text
        assert "custom Lean server" in text

    assert "real proof-state extraction" not in plan
    assert "real proof-state extraction" not in summary
    assert "full mathlib data refresh" not in summary
    assert "larger LeanRank-data refreshes" in summary
    assert "improve LeanRank-data retrieval quality" in plan


def test_project_summary_includes_current_production_evidence() -> None:
    repo = Path(__file__).resolve().parents[1]
    summary = (repo / "docs/proofatlas_project_summary_en.md").read_text(encoding="utf-8")
    for token in [
        "Current Production Evidence",
        "Theorem-level premise retrieval | Recall@10 | 0.4940",
        "candidate_pool_miss_top_100 | 1,823",
        "reranking_headroom_after_top10 | 458",
        "7 CUDA devices",
        "`torch_cuda` batched top-k",
        "Latest timed run: 551.6511 seconds",
    ]:
        assert token in summary
