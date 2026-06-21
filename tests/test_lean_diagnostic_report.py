import json

from leanrank_kg import lean_diagnostic_report


def test_lean_diagnostic_report_covers_success_dedup_and_failure_cases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = lean_diagnostic_report.run()

    assert report["method"] == "lean_unsolved_goals_diagnostic"
    assert report["production_pipeline_role"].endswith("not part of the default LeanRank-data pipeline")
    assert report["quality_checks"]["all_cases_passed"] is True
    assert report["quality_checks"]["has_successful_extraction_case"] is True
    assert report["quality_checks"]["has_failure_explanation_case"] is True
    assert report["quality_checks"]["has_timeout_stderr_extraction_case"] is True
    assert report["quality_checks"]["has_adjacent_goal_split_case"] is True
    assert report["quality_checks"]["has_multiline_goal_case"] is True
    assert report["quality_checks"]["has_initial_goal_skeleton_case"] is True
    assert report["quality_checks"]["all_tactic_trace_counts_match"] is True
    assert report["quality_checks"]["has_multi_state_tactic_trace_case"] is True
    assert report["total_extracted_proof_states"] > 0
    acceptance = report["acceptance_profile"]
    assert acceptance["summary"]["required_gates_passed"] is True
    assert acceptance["summary"]["advisory_gates_passed"] is True
    assert next(row for row in acceptance["gates"] if row["name"] == "ordered_tactic_state_trace")["passed"] is True
    assert next(row for row in acceptance["gates"] if row["name"] == "multiline_goal_preserved")["passed"] is True
    assert next(row for row in acceptance["gates"] if row["name"] == "query_time_only_scope")["passed"] is True
    duplicate_case = next(case for case in report["cases"] if case["name"] == "duplicate_goal_context")
    assert duplicate_case["rejected_count"] == 1
    adjacent_case = next(case for case in report["cases"] if case["name"] == "adjacent_goals_without_blank_lines")
    assert adjacent_case["extracted_count"] == 2
    assert adjacent_case["tactic_state_trace"]["state_count"] == 2
    assert [state["tactic_idx"] for state in adjacent_case["tactic_state_trace"]["states"]] == [0, 1]
    assert [state["goal_text"] for state in adjacent_case["proof_states"]] == ["x = x", "y = y"]
    multiline_case = next(case for case in report["cases"] if case["name"] == "multiline_goal")
    assert multiline_case["extracted_count"] == 1
    assert multiline_case["checks"]["multiline_goal_preserved"] is True
    assert multiline_case["proof_states"][0]["goal_line_count"] == 2
    assert multiline_case["tactic_state_trace"]["states"][0]["has_multiline_goal"] is True
    missing_case = next(case for case in report["cases"] if case["name"] == "missing_turnstile")
    assert missing_case["failure_reason"] == "no_goal_blocks_with_turnstile"
    timeout_case = next(case for case in report["cases"] if case["name"] == "timeout_stderr_unsolved_goal")
    assert timeout_case["extracted_count"] == 1
    assert timeout_case["checks"]["timeout_diagnostic_preserved"] is True
    skeleton_case = next(case for case in report["cases"] if case["name"] == "theorem_statement_initial_goal_skeleton")
    assert skeleton_case["extracted_count"] == 1
    assert skeleton_case["checks"]["skeleton_source_created"] is True
    assert skeleton_case["skeleton_source"].rstrip().endswith(":= by")
    assert (tmp_path / "outputs/reports/lean_diagnostic_extraction_report.json").exists()
    json.dumps(report, allow_nan=False)
