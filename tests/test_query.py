import pytest

from leanrank_kg.query import NewProofStateQuery, NewTheoremQuery, build_query


def test_new_theorem_query_extracts_statement_fields():
    text = "import Mathlib.Data.Nat.Basic\n\ntheorem Nat.self_eq (x : Nat) (h : x = x) : x = x := by\n  simpa"
    query = NewTheoremQuery.from_text(text)
    assert query.full_name == "Nat.self_eq"
    assert query.goal_text == "x = x"
    assert "x : Nat" in query.local_hypotheses
    assert "h : x = x" in query.local_hypotheses
    assert "Mathlib.Data.Nat.Basic" in query.namespace_hints
    assert query.retrieval_text


def test_new_theorem_query_handles_implicit_and_typeclass_binders():
    text = """
import Mathlib.Algebra.Group.Basic

open scoped BigOperators
open Nat

theorem map_mul_self {α : Type u} [Monoid α] (x y : α) : x * y = x * y := by
  rfl
"""
    query = NewTheoremQuery.from_text(text)
    assert query.full_name == "map_mul_self"
    assert query.goal_text == "x * y = x * y"
    assert "α : Type u" in query.local_hypotheses
    assert "Monoid α" in query.local_hypotheses
    assert "Monoid α" in query.typeclass_hints
    assert "x y : α" in query.local_hypotheses
    assert "Mathlib.Algebra.Group.Basic" in query.namespace_hints
    assert "Nat" in query.namespace_hints
    assert query.binder_groups[0]["kind"] == "implicit"
    assert query.binder_groups[0]["names"] == ["α"]
    assert query.binder_groups[0]["type"] == "Type u"
    assert query.binder_groups[1]["kind"] == "typeclass"
    assert query.binder_groups[1]["type"] == "Monoid α"
    assert query.typeclass_symbols == ["Monoid", "α"]
    assert "*" in query.operator_symbols
    assert "Type u" in query.sort_symbols
    assert query.normalized_goal_text == "v1 * v2 = v1 * v2"
    assert query.parsed_feature_summary["binder_count"] == 3
    assert query.parsed_feature_summary["typeclass_binder_count"] == 1
    assert query.parsed_feature_summary["operator_symbol_count"] >= 2
    assert query.parsed_feature_summary["sort_symbol_count"] == 1


def test_new_theorem_query_extracts_conclusion_constants_and_alpha_normalizes():
    left = NewTheoremQuery.from_text("theorem left (x : Nat) : Nat.succ x = x + 1 := by omega")
    right = NewTheoremQuery.from_text("theorem right (n : Nat) : Nat.succ n = n + 1 := by omega")
    assert left.conclusion_symbols == ["Nat.succ"]
    assert right.conclusion_symbols == ["Nat.succ"]
    assert left.normalized_goal_text == right.normalized_goal_text
    assert left.normalized_goal_text == "Nat.succ v0 = v0 + 1"


def test_new_theorem_query_uses_top_level_decl_colon_for_goal():
    text = "theorem forall_self : ∀ x : Nat, x = x := by intro x; rfl"
    query = NewTheoremQuery.from_text(text)
    assert query.goal_text == "∀ x : Nat, x = x"


def test_build_query_supports_goal_text():
    query = build_query("x = x", input_type="goal")
    assert isinstance(query, NewProofStateQuery)
    assert query.input_type == "goal"
    assert query.goal_text == "x = x"


def test_build_query_rejects_unknown_input_type():
    with pytest.raises(ValueError, match="Unknown query input_type"):
        build_query("x = x", input_type="unknown")  # type: ignore[arg-type]
