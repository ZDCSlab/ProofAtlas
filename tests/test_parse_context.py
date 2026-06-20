from leanrank_kg.parse_context import parse_context


def test_parse_context_normal():
    parsed = parse_context("x : Nat\nh : x = x\n⊢ x = x")
    assert parsed["goal_text"] == "x = x"
    assert len(parsed["local_hypotheses"]) == 2


def test_parse_context_missing_turnstile():
    parsed = parse_context("x = x")
    assert parsed["goal_text"] == "x = x"


def test_parse_context_empty():
    parsed = parse_context("")
    assert parsed["goal_text"] == ""
