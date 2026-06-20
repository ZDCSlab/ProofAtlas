from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .parse_context import parse_context
from .utils import stable_hash


def _lean_command() -> list[str] | None:
    if shutil.which("lake"):
        return ["lake", "env", "lean"]
    if shutil.which("lean"):
        return ["lean"]
    return None


def _extract_goal_blocks(text: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] = []
    in_goals = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if "unsolved goals" in stripped:
            in_goals = True
            if current:
                blocks.append(current)
                current = []
            continue
        if not in_goals:
            continue
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        if stripped.startswith("case ") and current:
            blocks.append(current)
            current = [stripped]
            continue
        if stripped.startswith(("error:", "warning:", "info:")) and current:
            blocks.append(current)
            current = []
            in_goals = "unsolved goals" in stripped
            continue
        current.append(stripped)
    if current:
        blocks.append(current)
    return ["\n".join(block) for block in blocks if any("⊢" in line for line in block)]


def _diagnostic_text(stdout: str = "", stderr: str = "") -> str:
    return "\n".join(part for part in [stderr, stdout] if part)


def _proof_state_retrieval_text(parsed: dict) -> str:
    hypotheses = "\n".join(parsed.get("local_hypotheses") or [])
    goal = parsed.get("goal_text") or ""
    return "\n".join(part for part in [hypotheses, f"⊢ {goal}" if goal else ""] if part).strip()


def extract_proof_state_report(stdout: str = "", stderr: str = "") -> dict:
    text = _diagnostic_text(stdout, stderr)
    raw_blocks = _extract_goal_blocks(text)
    states: list[dict] = []
    rejected_blocks: list[dict] = []
    seen = set()
    for block_index, block in enumerate(raw_blocks):
        lines = [line for line in block.splitlines() if not line.startswith("case ")]
        context = "\n".join(lines)
        parsed = parse_context(context)
        if not parsed["goal_text"]:
            rejected_blocks.append({"block_index": block_index, "reason": "missing_goal_text", "raw_text": context})
            continue
        key = (tuple(parsed["local_hypotheses"]), parsed["goal_text"])
        if key in seen:
            rejected_blocks.append({"block_index": block_index, "reason": "duplicate_goal_context", "raw_text": context})
            continue
        seen.add(key)
        retrieval_text = _proof_state_retrieval_text(parsed)
        states.append(
            {
                "proof_state_id": f"lean_diag:{stable_hash(retrieval_text, 16)}",
                "extraction_method": "lean_unsolved_goals_diagnostic",
                "block_index": block_index,
                "goal_text": parsed["goal_text"],
                "local_hypotheses": parsed["local_hypotheses"],
                "symbols": parsed["symbols"],
                "namespace_hints": parsed["namespace_hints"],
                "typeclass_hints": parsed["typeclass_hints"],
                "raw_text": context,
                "retrieval_text": retrieval_text,
                "hypothesis_count": len(parsed["local_hypotheses"]),
                "symbol_count": len(parsed["symbols"]),
                "context_line_count": len([line for line in context.splitlines() if line.strip()]),
            }
        )
    failure_reason = None
    if "unsolved goals" not in text:
        failure_reason = "no_unsolved_goals_diagnostic"
    elif not raw_blocks:
        failure_reason = "no_goal_blocks_with_turnstile"
    elif not states:
        failure_reason = "goal_blocks_rejected"
    return {
        "method": "lean_unsolved_goals_diagnostic",
        "source": "stderr_stdout",
        "has_unsolved_goals": "unsolved goals" in text,
        "raw_block_count": len(raw_blocks),
        "extracted_count": len(states),
        "failure_reason": failure_reason,
        "proof_states": states,
        "rejected_blocks": rejected_blocks,
    }


def extract_proof_states_from_diagnostics(stdout: str = "", stderr: str = "") -> list[dict]:
    return extract_proof_state_report(stdout=stdout, stderr=stderr)["proof_states"]


def _diagnostic_summary(stdout: str = "", stderr: str = "") -> dict:
    text = _diagnostic_text(stdout, stderr)
    extraction = extract_proof_state_report(stdout=stdout, stderr=stderr)
    return {
        "has_unsolved_goals": "unsolved goals" in text,
        "error_count": sum(1 for line in text.splitlines() if line.strip().startswith("error:")),
        "warning_count": sum(1 for line in text.splitlines() if line.strip().startswith("warning:")),
        "proof_state_raw_block_count": extraction["raw_block_count"],
        "proof_state_extracted_count": extraction["extracted_count"],
        "proof_state_extraction_failure_reason": extraction["failure_reason"],
    }


def check_lean_syntax(source: str, timeout_seconds: int = 10) -> dict:
    command = _lean_command()
    if command is None:
        return {
            "checked": False,
            "available": False,
            "ok": None,
            "command": None,
            "returncode": None,
            "stdout": "",
            "stderr": "Lean executable not found. Install Lean/Lake or run inside a Lean project environment.",
            "proof_states": [],
            "proof_state_extraction": {
                "method": "lean_unsolved_goals_diagnostic",
                "source": "stderr_stdout",
                "has_unsolved_goals": False,
                "raw_block_count": 0,
                "extracted_count": 0,
                "failure_reason": "lean_unavailable",
                "proof_states": [],
                "rejected_blocks": [],
            },
            "summary": {"has_unsolved_goals": False, "error_count": 0, "warning_count": 0},
        }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ProofAtlasQuery.lean"
        path.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [*command, str(path)],
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            extraction = extract_proof_state_report(exc.stdout or "", "")
            return {
                "checked": True,
                "available": True,
                "ok": False,
                "command": command,
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": f"Lean syntax check timed out after {timeout_seconds} seconds.",
                "proof_states": extraction["proof_states"],
                "proof_state_extraction": extraction,
                "summary": _diagnostic_summary(exc.stdout or "", ""),
            }
    stdout = proc.stdout[-4000:]
    stderr = proc.stderr[-4000:]
    extraction = extract_proof_state_report(stdout, stderr)
    return {
        "checked": True,
        "available": True,
        "ok": proc.returncode == 0,
        "command": command,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "proof_states": extraction["proof_states"],
        "proof_state_extraction": extraction,
        "summary": _diagnostic_summary(stdout, stderr),
    }
