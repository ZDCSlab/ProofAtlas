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
    assert f"Total seconds: {float(timing['total_seconds']):.4f}" in audit


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


def test_readme_premise_supervision_snapshot_matches_committed_artifacts() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    supervision = json.loads((repo / "outputs/reports/premise_trace_supervision_report.json").read_text(encoding="utf-8"))
    current = supervision["current_artifact_supervision"]
    train = supervision["splits"]["train"]

    assert _readme_artifact_field(readme, "Premise supervision", "total positive edges") == str(current["total_positive_edges"])
    assert _readme_artifact_field(readme, "Premise supervision", "total negative candidates") == str(current["total_negative_edges"])
    assert _readme_artifact_field(readme, "Premise supervision", "negative/positive ratio") == f"{float(current['negative_to_positive_edge_ratio']):.4f}"
    assert _readme_artifact_field(readme, "Train proof-state supervision", "positive coverage") == f"{float(train['positive_proof_state_coverage']):.4f}"
    assert _readme_artifact_field(readme, "Train proof-state supervision", "negative coverage") == f"{float(train['negative_proof_state_coverage']):.4f}"
    assert _readme_artifact_field(readme, "Train hard negatives", "hardness mean") == f"{float(train['negative_candidate_hardness']['mean']):.4f}"


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

    assert "!outputs/reports/pipeline_run_timings.json" in gitignore
