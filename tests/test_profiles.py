from __future__ import annotations

import pandas as pd

from proofatlas.profiles import premise_profile, proof_state_profile, theorem_profiles


def test_proof_state_profile_includes_context_fields() -> None:
    text = proof_state_profile(
        {
            "full_name": "Foo.bar",
            "domain_tag": "Algebra",
            "subdomain_tag": "Group",
            "local_hypotheses": ["x : Nat", "h : x = x"],
            "goal_text": "x = x",
            "symbols": ["Nat", "="],
        }
    )
    assert "Foo.bar" in text
    assert "x : Nat" in text
    assert "x = x" in text
    assert "Nat" in text


def test_premise_profile_includes_code_and_location() -> None:
    text = premise_profile(
        {
            "full_name": "Nat.add_comm",
            "file_path": "Mathlib/Data/Nat/Basic.lean",
            "domain_tag": "Data",
            "subdomain_tag": "Nat",
            "code": "theorem add_comm ...",
        }
    )
    assert "Nat.add_comm" in text
    assert "Mathlib/Data/Nat/Basic.lean" in text
    assert "theorem add_comm" in text


def test_theorem_profiles_uses_clean_proof_state_text_without_premises() -> None:
    theorems = pd.DataFrame(
        [
            {
                "id": "thm:Foo.bar",
                "full_name": "Foo.bar",
                "file_path": "Mathlib/Foo.lean",
                "domain_tag": "Foo",
                "subdomain_tag": "Foo.lean",
            }
        ]
    )
    proof_states = pd.DataFrame(
        [
            {
                "id": "ps:1",
                "theorem_id": "thm:Foo.bar",
                "tactic_idx": 0,
                "goal_text": "x = x",
                "local_hypotheses": ["x : Nat"],
                "symbols": ["Nat", "="],
            }
        ]
    )
    profiles = theorem_profiles(theorems, proof_states, max_states=3)
    text = profiles.loc[0, "profile_text"]
    assert "Foo.bar" in text
    assert "x = x" in text
    assert "Nat" in text
    assert "premise:" not in text
