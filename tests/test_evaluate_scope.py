from __future__ import annotations

import json

import pandas as pd

from leanrank_kg import evaluate
from leanrank_kg.utils import write_json, write_parquet


def test_full_heldout_override_removes_core_evaluation_limits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/proofatlas.yaml").write_text(
        "\n".join(
            [
                "dataset_name: erbacher/LeanRank-data",
                "retrieval: {top_k: [1]}",
                "evaluation:",
                "  max_val_proof_states: 7",
                "  max_val_theorems: 5",
                "  max_test_proof_states: 11",
                "  max_test_theorems: 3",
                "  rerank_max_test_proof_states: 0",
                "  query_representation_diagnostic_examples: 0",
            ]
        ),
        encoding="utf-8",
    )
    write_parquet(pd.DataFrame({"id": ["premise:p"]}), "data/processed/train/premises.parquet")
    write_json("outputs/reports/ranker_validation_metrics.json", {})

    proof_state_limits = {}
    theorem_limits = {}

    def fake_proof_state(split, top_ks, train_premises, max_examples=None, **kwargs):
        proof_state_limits[split] = max_examples
        return {
            "metrics": {"evaluated_queries": 1, "evaluated_retrievable_queries": 1, "Recall@1": 1.0},
            "examples": [{"split": split, "proof_state_id": f"ps{i}"} for i in range(20)],
            "per_query": [],
            "backend_info": {"actual_backend": "fake"},
        }

    def fake_theorem(split, top_ks, train_premises, max_theorems=None, **kwargs):
        theorem_limits[split] = max_theorems
        return {
            "metrics": {"theorem_retrieval_evaluated_theorems": 1, "theorem_retrieval_Recall@1": 1.0},
            "case_studies": [],
            "per_query": [],
            "backend_info": {"actual_backend": "fake"},
        }

    monkeypatch.setattr(evaluate, "_evaluate_proof_state_retrieval_split", fake_proof_state)
    monkeypatch.setattr(evaluate, "_evaluate_theorem_retrieval_split", fake_theorem)
    monkeypatch.setattr(
        evaluate,
        "_evaluate_reranked_proof_state_retrieval_split",
        lambda *args, **kwargs: {"metrics": {}, "backend_info": {}, "candidate_k_ablation": [], "examples": [], "per_query": []},
    )
    monkeypatch.setattr(evaluate, "_evaluate_proof_state_query_representations", lambda *args, **kwargs: {})

    evaluate.run("configs/proofatlas.yaml", full_heldout=True)

    data = json.loads((tmp_path / "outputs/reports/test_set_evaluation.json").read_text(encoding="utf-8"))
    assert proof_state_limits == {"val": None, "test": None}
    assert theorem_limits == {"val": None, "test": None}
    assert data["evaluation_scope"]["full_heldout_override"] is True
    assert data["evaluation_scope"]["is_sampled"] is False
