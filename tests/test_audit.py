import json

from leanrank_kg import audit


def test_audit_reports_missing_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/reports").mkdir(parents=True)
    result = audit.build_audit()
    assert result["passed"] is False
    assert result["failed_checks"]
    assert (tmp_path / "outputs/reports/mvp_completion_audit.json").exists()
    json.dumps(result, allow_nan=False)
