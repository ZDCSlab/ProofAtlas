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
