import subprocess

from leanrank_kg import lean_check


def test_check_lean_syntax_reports_missing_tool(monkeypatch):
    monkeypatch.setattr(lean_check.shutil, "which", lambda name: None)
    result = lean_check.check_lean_syntax("theorem t : True := by trivial")
    assert result["checked"] is False
    assert result["available"] is False
    assert result["ok"] is None


def test_check_lean_syntax_uses_lake_when_available(monkeypatch, tmp_path):
    calls = {}

    def fake_which(name):
        return f"/usr/bin/{name}" if name == "lake" else None

    def fake_run(command, **kwargs):
        calls["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(lean_check.shutil, "which", fake_which)
    monkeypatch.setattr(lean_check.subprocess, "run", fake_run)
    result = lean_check.check_lean_syntax("theorem t : True := by trivial")
    assert result["checked"] is True
    assert result["available"] is True
    assert result["ok"] is True
    assert calls["command"][:3] == ["lake", "env", "lean"]


def test_extract_proof_states_from_unsolved_goals_diagnostics():
    stderr = """
error: unsolved goals
case h
x : Nat
h : x = x
⊢ x = x
"""
    states = lean_check.extract_proof_states_from_diagnostics(stderr=stderr)
    assert len(states) == 1
    assert states[0]["goal_text"] == "x = x"
    assert states[0]["local_hypotheses"] == ["x : Nat", "h : x = x"]
    assert "=" in states[0]["symbols"]
    assert states[0]["proof_state_id"].startswith("lean_diag:")
    assert states[0]["retrieval_text"].endswith("⊢ x = x")


def test_extract_proof_state_report_explains_failures():
    report = lean_check.extract_proof_state_report(stderr="error: unsolved goals\ncase h\nno goal marker")
    assert report["has_unsolved_goals"] is True
    assert report["raw_block_count"] == 0
    assert report["extracted_count"] == 0
    assert report["failure_reason"] == "no_goal_blocks_with_turnstile"
