from __future__ import annotations

from typing import Any

from .lean_check import _initial_goal_skeleton_source, build_tactic_state_trace, extract_proof_state_report
from .utils import write_json


CASES = [
    {
        "name": "single_unsolved_goal",
        "description": "One Lean unsolved-goal diagnostic with hypotheses and a goal.",
        "stderr": """
error: unsolved goals
case h
x : Nat
h : x = x
⊢ x = x
""",
        "expected_extracted_count": 1,
    },
    {
        "name": "multiple_unsolved_goals",
        "description": "Multiple case blocks emitted by Lean for one diagnostic.",
        "stderr": """
error: unsolved goals
case left
n : Nat
⊢ n = n

case right
m : Nat
h : m = m
⊢ m = m
""",
        "expected_extracted_count": 2,
    },
    {
        "name": "duplicate_goal_context",
        "description": "Repeated goal/context blocks are deduplicated for retrieval.",
        "stderr": """
error: unsolved goals
case first
x : Nat
⊢ x = x

case second
x : Nat
⊢ x = x
""",
        "expected_extracted_count": 1,
        "expected_rejected_count": 1,
    },
    {
        "name": "adjacent_goals_without_blank_lines",
        "description": "Multiple Lean goals can be split even when diagnostics omit blank-line or case separators.",
        "stderr": """
error: unsolved goals
x : Nat
⊢ x = x
y : Nat
⊢ y = y
""",
        "expected_extracted_count": 2,
    },
    {
        "name": "missing_turnstile",
        "description": "Malformed diagnostics explain why no proof state was extracted.",
        "stderr": """
error: unsolved goals
case bad
no goal marker here
""",
        "expected_extracted_count": 0,
        "expected_failure_reason": "no_goal_blocks_with_turnstile",
    },
    {
        "name": "timeout_stderr_unsolved_goal",
        "description": "A timed-out Lean check can still preserve stderr diagnostics for proof-state extraction.",
        "stderr": """
error: unsolved goals
case timeout
y : Nat
⊢ y = y
""",
        "expected_extracted_count": 1,
        "expected_timeout_preserved": True,
    },
    {
        "name": "theorem_statement_initial_goal_skeleton",
        "description": "A theorem statement without a proof body can be checked as a temporary `:= by` skeleton to get the initial goal.",
        "source": "theorem t (x : Nat) : x = x",
        "expected_skeleton_suffix": ":= by",
        "stderr": """
error: unsolved goals
x : Nat
⊢ x = x
""",
        "expected_extracted_count": 1,
        "expected_skeleton_source": True,
    },
]


def _case_result(case: dict[str, Any]) -> dict[str, Any]:
    report = extract_proof_state_report(stdout=case.get("stdout", ""), stderr=case.get("stderr", ""))
    trace = build_tactic_state_trace(report["proof_states"], source_variant=str(case.get("source_variant", "fixture")))
    expected_extracted = int(case.get("expected_extracted_count", 0))
    expected_rejected = case.get("expected_rejected_count")
    expected_failure = case.get("expected_failure_reason")
    checks = {
        "extracted_count_matches": report["extracted_count"] == expected_extracted,
        "proof_states_have_retrieval_text": all(bool(row.get("retrieval_text")) for row in report["proof_states"]),
        "proof_states_have_stable_ids": all(str(row.get("proof_state_id", "")).startswith("lean_diag:") for row in report["proof_states"]),
        "tactic_trace_count_matches": trace["state_count"] == report["extracted_count"],
        "tactic_trace_has_stable_ids": all(str(row.get("tactic_state_id", "")).startswith("lean_trace:") for row in trace["states"]),
    }
    if expected_rejected is not None:
        checks["rejected_count_matches"] = len(report["rejected_blocks"]) == int(expected_rejected)
    if expected_failure is not None:
        checks["failure_reason_matches"] = report["failure_reason"] == expected_failure
    if case.get("expected_timeout_preserved") is not None:
        checks["timeout_diagnostic_preserved"] = bool(case.get("expected_timeout_preserved")) and report["extracted_count"] > 0
    skeleton_source = None
    if case.get("expected_skeleton_source") is not None:
        skeleton_source = _initial_goal_skeleton_source(str(case.get("source", "")))
        checks["skeleton_source_created"] = bool(skeleton_source) is bool(case.get("expected_skeleton_source"))
        suffix = str(case.get("expected_skeleton_suffix", ""))
        if suffix:
            checks["skeleton_source_suffix_matches"] = bool(skeleton_source and skeleton_source.rstrip().endswith(suffix))
    return {
        "name": case["name"],
        "description": case["description"],
        "expected_extracted_count": expected_extracted,
        "extracted_count": report["extracted_count"],
        "raw_block_count": report["raw_block_count"],
        "rejected_count": len(report["rejected_blocks"]),
        "failure_reason": report["failure_reason"],
        "checks": checks,
        "passed": all(checks.values()),
        "skeleton_source": skeleton_source,
        "tactic_state_trace": trace,
        "proof_states": report["proof_states"],
        "rejected_blocks": report["rejected_blocks"],
    }


def _acceptance_profile(report: dict[str, Any]) -> dict[str, Any]:
    quality = report.get("quality_checks", {}) if isinstance(report, dict) else {}
    gates = [
        {
            "name": "diagnostic_cases_passed",
            "severity": "required",
            "passed": quality.get("all_cases_passed") is True,
            "value": {
                "passed_case_count": report.get("passed_case_count"),
                "case_count": report.get("case_count"),
            },
            "threshold": "all fixture diagnostic extraction cases pass",
        },
        {
            "name": "proof_states_extracted",
            "severity": "required",
            "passed": int(report.get("total_extracted_proof_states") or 0) > 0,
            "value": report.get("total_extracted_proof_states"),
            "threshold": ">0 extracted proof states",
        },
        {
            "name": "retrieval_text_present",
            "severity": "required",
            "passed": quality.get("all_extracted_states_have_retrieval_text") is True,
            "value": quality.get("all_extracted_states_have_retrieval_text"),
            "threshold": "all extracted proof states have retrieval text",
        },
        {
            "name": "ordered_tactic_state_trace",
            "severity": "required",
            "passed": quality.get("all_tactic_trace_counts_match") is True and quality.get("has_multi_state_tactic_trace_case") is True,
            "value": {
                "all_tactic_trace_counts_match": quality.get("all_tactic_trace_counts_match"),
                "has_multi_state_tactic_trace_case": quality.get("has_multi_state_tactic_trace_case"),
            },
            "threshold": "trace counts match and at least one multi-state trace case exists",
        },
        {
            "name": "initial_goal_skeleton",
            "severity": "required",
            "passed": quality.get("has_initial_goal_skeleton_case") is True,
            "value": quality.get("has_initial_goal_skeleton_case"),
            "threshold": "theorem/lemma/example statement can be checked as temporary := by skeleton",
        },
        {
            "name": "timeout_stderr_extraction",
            "severity": "advisory",
            "passed": quality.get("has_timeout_stderr_extraction_case") is True,
            "value": quality.get("has_timeout_stderr_extraction_case"),
            "threshold": "timeout stderr unsolved-goal diagnostics remain parseable",
        },
        {
            "name": "failure_explanation",
            "severity": "advisory",
            "passed": quality.get("has_failure_explanation_case") is True,
            "value": quality.get("has_failure_explanation_case"),
            "threshold": "malformed diagnostics report a failure reason",
        },
        {
            "name": "query_time_only_scope",
            "severity": "required",
            "passed": "not a corpus extractor" in str(report.get("production_pipeline_role", "")),
            "value": report.get("production_pipeline_role"),
            "threshold": "query-time diagnostics only; not a LeanRank-data corpus extractor",
        },
    ]
    required = [gate for gate in gates if gate["severity"] == "required"]
    advisory = [gate for gate in gates if gate["severity"] == "advisory"]
    return {
        "scope": report.get("scope"),
        "summary": {
            "required_gates_passed": all(gate["passed"] for gate in required),
            "advisory_gates_passed": all(gate["passed"] for gate in advisory),
            "passed_gate_count": sum(1 for gate in gates if gate["passed"]),
            "total_gate_count": len(gates),
            "required_gate_count": len(required),
            "advisory_gate_count": len(advisory),
        },
        "gates": gates,
    }


def build_report() -> dict[str, Any]:
    cases = [_case_result(case) for case in CASES]
    total_extracted = sum(case["extracted_count"] for case in cases)
    report = {
        "scope": "query-time Lean unsolved-goals diagnostic proof-state extraction",
        "production_pipeline_role": "optional query diagnostics only; not a corpus extractor and not part of the default LeanRank-data pipeline",
        "method": "lean_unsolved_goals_diagnostic",
        "case_count": len(cases),
        "passed_case_count": sum(1 for case in cases if case["passed"]),
        "total_extracted_proof_states": total_extracted,
        "quality_checks": {
            "all_cases_passed": all(case["passed"] for case in cases),
            "has_successful_extraction_case": any(case["extracted_count"] > 0 for case in cases),
            "has_failure_explanation_case": any(case["failure_reason"] for case in cases),
            "has_timeout_stderr_extraction_case": any(
                case["name"] == "timeout_stderr_unsolved_goal" and case["extracted_count"] > 0
                for case in cases
            ),
            "has_adjacent_goal_split_case": any(
                case["name"] == "adjacent_goals_without_blank_lines" and case["extracted_count"] == 2
                for case in cases
            ),
            "has_initial_goal_skeleton_case": any(
                case["name"] == "theorem_statement_initial_goal_skeleton"
                and case["checks"].get("skeleton_source_created")
                and case["extracted_count"] > 0
                for case in cases
            ),
            "all_extracted_states_have_retrieval_text": all(
                row.get("retrieval_text")
                for case in cases
                for row in case["proof_states"]
            ),
            "all_tactic_trace_counts_match": all(
                case["tactic_state_trace"]["state_count"] == case["extracted_count"]
                for case in cases
            ),
            "has_multi_state_tactic_trace_case": any(
                case["tactic_state_trace"]["state_count"] >= 2
                for case in cases
            ),
        },
        "cases": cases,
    }
    report["acceptance_profile"] = _acceptance_profile(report)
    return report


def run(output_path: str = "outputs/reports/lean_diagnostic_extraction_report.json") -> dict[str, Any]:
    report = build_report()
    write_json(output_path, report)
    return report
