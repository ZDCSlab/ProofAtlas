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


def test_check_lean_syntax_extracts_timeout_stderr_goals(monkeypatch):
    def fake_which(name):
        return f"/usr/bin/{name}" if name == "lean" else None

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command,
            timeout=1,
            output="",
            stderr="""
error: unsolved goals
case h
x : Nat
⊢ x = x
""",
        )

    monkeypatch.setattr(lean_check.shutil, "which", fake_which)
    monkeypatch.setattr(lean_check.subprocess, "run", fake_run)

    result = lean_check.check_lean_syntax("theorem t (x : Nat) : x = x := by", timeout_seconds=1)

    assert result["checked"] is True
    assert result["ok"] is False
    assert "timed out" in result["stderr"]
    assert result["summary"]["has_unsolved_goals"] is True
    assert result["summary"]["proof_state_extracted_count"] == 1
    assert result["proof_states"][0]["goal_text"] == "x = x"


def test_check_lean_syntax_falls_back_to_initial_goal_skeleton(monkeypatch):
    calls = []

    def fake_which(name):
        return f"/usr/bin/{name}" if name == "lean" else None

    def fake_run(command, **kwargs):
        source_path = command[-1]
        with open(source_path, encoding="utf-8") as fh:
            source = fh.read()
        calls.append(source)
        if source.rstrip().endswith(":= by"):
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="""
error: unsolved goals
x : Nat
⊢ x = x
""",
            )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="error: declaration uses 'sorry'")

    monkeypatch.setattr(lean_check.shutil, "which", fake_which)
    monkeypatch.setattr(lean_check.subprocess, "run", fake_run)

    result = lean_check.check_lean_syntax("theorem t (x : Nat) : x = x")

    assert len(calls) == 2
    assert calls[1].rstrip().endswith(":= by")
    assert result["source_variant"] == "initial_goal_skeleton"
    assert result["fallback_attempted"] is True
    assert result["fallback_reason"] == "original_no_proof_states"
    assert result["original_summary"]["proof_state_extracted_count"] == 0
    assert result["proof_states"][0]["goal_text"] == "x = x"


def test_check_lean_syntax_does_not_skeleton_completed_declaration(monkeypatch):
    calls = []

    def fake_which(name):
        return f"/usr/bin/{name}" if name == "lean" else None

    def fake_run(command, **kwargs):
        source_path = command[-1]
        with open(source_path, encoding="utf-8") as fh:
            calls.append(fh.read())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(lean_check.shutil, "which", fake_which)
    monkeypatch.setattr(lean_check.subprocess, "run", fake_run)

    result = lean_check.check_lean_syntax("theorem t : True := by trivial")

    assert len(calls) == 1
    assert result["source_variant"] == "original"
    assert result["fallback_attempted"] is False


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


def test_extract_proof_states_splits_adjacent_goals_without_blank_lines():
    stderr = """
error: unsolved goals
x : Nat
⊢ x = x
y : Nat
⊢ y = y
"""
    states = lean_check.extract_proof_states_from_diagnostics(stderr=stderr)
    assert [state["goal_text"] for state in states] == ["x = x", "y = y"]
    assert states[0]["local_hypotheses"] == ["x : Nat"]
    assert states[1]["local_hypotheses"] == ["y : Nat"]


def test_extract_proof_state_report_explains_failures():
    report = lean_check.extract_proof_state_report(stderr="error: unsolved goals\ncase h\nno goal marker")
    assert report["has_unsolved_goals"] is True
    assert report["raw_block_count"] == 0
    assert report["extracted_count"] == 0
    assert report["failure_reason"] == "no_goal_blocks_with_turnstile"
