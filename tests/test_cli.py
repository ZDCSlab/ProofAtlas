from typer.testing import CliRunner
from pathlib import Path

from leanrank_kg.cli import _stage_artifacts, app


def test_show_neighborhood_uses_entity_id_option():
    runner = CliRunner()
    result = runner.invoke(app, ["show-neighborhood", "--help"])
    assert result.exit_code == 0
    assert "--entity-id" in result.output


def test_query_retrieval_commands_are_registered():
    runner = CliRunner()
    result = runner.invoke(app, ["build-index", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    result = runner.invoke(app, ["benchmark-index", "--help"])
    assert result.exit_code == 0
    assert "--query-count" in result.output
    result = runner.invoke(app, ["profile-pipeline", "--help"])
    assert result.exit_code == 0
    assert "--output-path" in result.output
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--full-heldout" in result.output
    result = runner.invoke(app, ["premise-trace-supervision-report", "--help"])
    assert result.exit_code == 0
    assert "--output-path" in result.output
    result = runner.invoke(app, ["lean-diagnostic-extraction-report", "--help"])
    assert result.exit_code == 0
    assert "--output-path" in result.output
    result = runner.invoke(app, ["build-experiment-report", "--help"])
    assert result.exit_code == 0
    assert "--output-path" in result.output
    result = runner.invoke(app, ["security-review", "--help"])
    assert result.exit_code == 0
    assert "--public-exposure" in result.output
    result = runner.invoke(app, ["train-difficulty", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    result = runner.invoke(app, ["retrieve-theorem-guidance", "--help"])
    assert result.exit_code == 0
    assert "--theorem-text" in result.output
    assert "--query-file" in result.output
    assert "--input-type" in result.output
    assert "--domain-hint" in result.output
    assert "--validate-lean" in result.output
    result = runner.invoke(app, ["retrieve-premises-for-query", "--help"])
    assert result.exit_code == 0
    assert "--query-text" in result.output
    result = runner.invoke(app, ["similar-proof-states-for-query", "--help"])
    assert result.exit_code == 0
    assert "--query-text" in result.output
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--require-ready" in result.output
    assert "--startup-index-split" in result.output


def test_build_index_stage_artifacts_are_backend_neutral():
    artifacts = _stage_artifacts()["build_index"]
    assert "outputs/indexes/index_summary.json" in artifacts
    assert "outputs/indexes/train_premise_index_manifest.json" in artifacts
    assert "outputs/indexes/train_proof_state_index_manifest.json" in artifacts
    assert "outputs/indexes/train_theorem_index_manifest.json" in artifacts
    assert not any(path.endswith("_neighbors.joblib") for path in artifacts)


def test_deployment_guide_documents_demo_modes():
    text = Path("docs/proofatlas_deployment_guide.md").read_text(encoding="utf-8")
    for token in [
        "Static Homepage Review",
        "Local Interactive API Demo",
        "Readiness-Gated Server Mode",
        "Notebook Demo",
        "GitHub Pages / Hosted Static Demo",
        "GET /metrics",
        "GET /metrics/prometheus",
        "security-review",
        "deployment_security_review.json",
    ]:
        assert token in text


def test_makefile_exposes_trace_readiness_target():
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "security-review:" in text
    assert "profile-pipeline:" in text
    assert "premise-trace-supervision:" in text
    assert "experiment-report:" in text
