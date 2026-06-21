from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _table_value(markdown: str, first: str, second: str) -> str:
    prefix = f"| {first} | {second} | "
    for line in markdown.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).removesuffix(" |").strip()
    raise AssertionError(f"Missing table row: {first} / {second}")


def test_readme_is_research_facing() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")

    for token in [
        "research-oriented LeanRank retrieval dataset and report pipeline",
        "Research Deliverables",
        "Theorem-level premise retrieval and reranking",
        "Similar theorem and similar proof-state retrieval",
        "Retrieve strategy facets from similar proof states",
        "Retrieve historical difficulty profiles",
        "Visual Report",
        "Engineering Extras",
    ]:
        assert token in readme

    for old_mainline in [
        "Current production run coverage and timing",
        "Current resource and parallelism profile",
        "Current retrieval failure diagnosis",
        "make refresh-production-timing",
    ]:
        assert old_mainline not in readme


def test_readme_key_metrics_match_research_prediction_artifact() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    results = json.loads((repo / "outputs/predictions/research_prediction_results.json").read_text(encoding="utf-8"))

    theorem = results["premise_prediction"]["theorem"]
    strategy = results["proof_strategy_hinting"]["retrieval_evaluation"]
    difficulty = results["difficulty_prediction"]["retrieval_evaluation"]

    expected = {
        ("Theorem-level premise retrieval", "Recall@10"): theorem["Recall@10"],
        ("Theorem-level premise retrieval", "Recall@100"): theorem["Recall@100"],
        ("Theorem-level premise retrieval", "MRR"): theorem["MRR"],
        ("Strategy-facet retrieval", "Label Recall@5"): strategy["label_recall@5"],
        ("Strategy-facet retrieval", "Any-label Hit@3"): strategy["any_label_hit@3"],
        ("Difficulty-profile retrieval", "Bucket accuracy"): difficulty["bucket_accuracy"],
        ("Difficulty-profile retrieval", "Retrieved-profile MAE"): difficulty["retrieved_profile_mae"],
    }

    for (task, metric), value in expected.items():
        assert _table_value(readme, task, metric) == f"{float(value):.4f}"


def test_research_report_and_visual_report_have_same_task_story() -> None:
    repo = Path(__file__).resolve().parents[1]
    report = (repo / "outputs/reports/research_report.md").read_text(encoding="utf-8")
    homepage = (repo / "homepage/index.html").read_text(encoding="utf-8")

    for token in [
        "Theorem-level premise retrieval",
        "Proof Pattern Retrieval",
        "Strategy Retrieval",
        "Difficulty Retrieval",
        "We do not evaluate proof-pattern neighbors directly",
        "Action.full_res",
        "Affine.Simplex.affineCombination_mem_interior_iff",
    ]:
        assert token in report

    for token in [
        "ProofAtlas Research Report",
        "Retrieval-Centered LeanRank Proof Guidance",
        "Four Research Tasks",
        "Dataset And Split Statistics",
        "Top Test Domains",
        "Theorem-Level Premise Retrieval",
        "Strategy retrieval",
        "Difficulty retrieval",
        "Case Studies",
        "Action.full_res",
        "Affine.Simplex.affineCombination_mem_interior_iff",
        "Retrieved Premises",
        "Similar Theorems",
        "Similar Proof States",
        "Strategy Facets",
    ]:
        assert token in homepage

    for old_homepage_token in [
        "Interactive Proof Guidance Workbench",
        "Get Proof Guidance",
        "API URL",
        "renderLocalFallback",
        "Production Evidence",
    ]:
        assert old_homepage_token not in homepage


def test_project_summary_and_archived_docs_mark_scope() -> None:
    repo = Path(__file__).resolve().parents[1]
    summary = (repo / "docs/proofatlas_project_summary_en.md").read_text(encoding="utf-8")
    current = (repo / "docs/proofatlas_current_status_and_gap_to_theorem_retrieval.md").read_text(encoding="utf-8")
    audit = (repo / "docs/proofatlas_delivery_audit.md").read_text(encoding="utf-8")
    deployment = (repo / "docs/proofatlas_deployment_guide.md").read_text(encoding="utf-8")
    next_steps = (repo / "docs/proofatlas_next_steps_plan.md").read_text(encoding="utf-8")

    for token in [
        "ProofAtlas Research Summary",
        "Four Retrieval Tasks",
        "Theorem-level premise retrieval",
        "Strategy-facet retrieval",
        "Difficulty-profile retrieval",
        "Visual Report",
        "Engineering Extras",
    ]:
        assert token in summary

    assert "Status: archived engineering/status note" in current
    assert "Status: archived engineering audit" in audit
    assert "Status: engineering extra" in deployment
    assert "Status: archived planning note" in next_steps


def test_research_artifacts_are_tracked() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = [
        "outputs/reports/research_report.md",
        "outputs/predictions/research_prediction_results.json",
        "outputs/reports/theorem_retrieval_case_studies.json",
        "homepage/index.html",
        "homepage/assets/homepage_summary.json",
    ]
    tracked = subprocess.check_output(["git", "ls-files", *paths], cwd=repo, text=True).splitlines()
    assert set(paths) <= set(tracked)
