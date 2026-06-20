import json

from leanrank_kg import audit
from leanrank_kg.cli import full_pipeline


def test_audit_reports_missing_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/reports").mkdir(parents=True)
    result = audit.build_audit()
    assert result["passed"] is False
    assert result["failed_checks"]
    assert (tmp_path / "outputs/reports/mvp_completion_audit.json").exists()
    json.dumps(result, allow_nan=False)


def test_audit_covers_refined_theorem_guidance_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "schemas").mkdir()
    (tmp_path / "docs").mkdir()
    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    for schema in (repo / "schemas").glob("*.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    deployment_guide = repo / "docs/proofatlas_deployment_guide.md"
    (tmp_path / "docs/proofatlas_deployment_guide.md").write_text(deployment_guide.read_text(encoding="utf-8"), encoding="utf-8")
    project_summary = repo / "docs/proofatlas_project_summary_en.md"
    (tmp_path / "docs/proofatlas_project_summary_en.md").write_text(project_summary.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "README.md").write_text((repo / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='proofatlas-test'\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("demo:\n\ttrue\n", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks/leanrank_kg_demo.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 17\nuse_huggingface: false\nsample: {total_rows: 120, small_debug_rows: 120, committed_demo_rows: 120}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\n",
        encoding="utf-8",
    )
    full_pipeline(config="configs/sample.yaml", debug_rows=120)
    result = audit.build_audit()
    assert result["checks"]["validation:readme_delivery_evidence"]["passed"] is True
    assert result["checks"]["validation:corpus_manifest"]["passed"] is True
    assert result["checks"]["validation:artifact_compatibility"]["passed"] is True
    assert result["checks"]["validation:difficulty_estimator"]["passed"] is True
    assert result["checks"]["validation:ranker_ablation"]["passed"] is True
    assert result["checks"]["validation:theorem_retrieval_metrics"]["passed"] is True
    assert result["checks"]["validation:index_summary"]["passed"] is True
    assert result["checks"]["validation:pipeline_performance_report"]["passed"] is True
    assert result["checks"]["validation:resource_parallelism_profile"]["passed"] is True
    assert result["checks"]["validation:performance_acceptance_profile"]["passed"] is True
    assert result["checks"]["validation:lean_diagnostic_acceptance_profile"]["passed"] is True
    assert result["checks"]["validation:scale_projection_profile"]["passed"] is True
    assert result["checks"]["validation:rerank_evaluation_cost_profile"]["passed"] is True
    assert result["checks"]["validation:artifact_storage_profile"]["passed"] is True
    assert result["checks"]["validation:lean_diagnostic_extraction"]["passed"] is True
    assert result["checks"]["validation:refresh_dashboard"]["passed"] is True
    assert result["checks"]["validation:homepage_summary_supervision"]["passed"] is True
    assert result["checks"]["validation:refresh_trend"]["passed"] is True
    assert result["checks"]["validation:refresh_history"]["passed"] is True
    assert result["checks"]["validation:theorem_case_studies"]["passed"] is True
    assert result["checks"]["validation:graph_visualization"]["passed"] is True
    assert result["checks"]["validation:homepage_sections"]["passed"] is True
    assert result["checks"]["validation:deployment_guide"]["passed"] is True
    assert result["checks"]["validation:project_summary_evidence"]["passed"] is True
