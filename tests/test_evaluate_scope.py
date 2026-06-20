from __future__ import annotations

import json

import numpy as np
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
    assert data["evaluation_scope"]["total_seconds"] >= 0
    assert [row["name"] for row in data["evaluation_scope"]["substage_timings"]] == [
        "val_proof_state_retrieval",
        "test_proof_state_retrieval",
        "val_theorem_retrieval",
        "test_theorem_retrieval",
        "test_reranked_proof_state_retrieval",
        "val_proof_state_query_representation_diagnostic",
        "test_proof_state_query_representation_diagnostic",
    ]
    assert data["evaluation_scope"]["query_representation_diagnostic_splits"] == ["test", "val"]


def test_theorem_evaluation_skips_case_study_query_text_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/processed/test").mkdir(parents=True)
    write_parquet(
        pd.DataFrame(
            [
                {"id": "thm:a", "full_name": "A", "domain_tag": "Algebra", "subdomain_tag": "Group"},
                {"id": "thm:b", "full_name": "B", "domain_tag": "Algebra", "subdomain_tag": "Ring"},
            ]
        ),
        "data/processed/test/theorems.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {"id": "ps:a", "theorem_id": "thm:a", "goal_text": "a"},
                {"id": "ps:b", "theorem_id": "thm:b", "goal_text": "b"},
            ]
        ),
        "data/processed/test/proof_states.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {"proof_state_id": "ps:a", "premise_id": "premise:p"},
                {"proof_state_id": "ps:b", "premise_id": "premise:q"},
            ]
        ),
        "data/processed/test/positive_edges.parquet",
    )

    train_premises = {"premise:p", "premise:q"}
    monkeypatch.setattr(
        evaluate,
        "_embedding_ids",
        lambda split, entity_type: ["premise:p", "premise:q"] if entity_type == "Premise" else ["thm:a", "thm:b"],
    )
    monkeypatch.setattr(evaluate, "_load_embedding", lambda split, kind: np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))

    def fake_ranking_rows_from_embeddings(**kwargs):
        rows = []
        for query_id in kwargs["query_ids"]:
            rows.append(
                {
                    "theorem_id": query_id,
                    **evaluate._ranking_row(["premise:p", "premise:q"], kwargs["gold_by_query"][query_id], train_premises, kwargs["top_ks"]),
                }
            )
        return rows, {}, {"actual_backend": "fake"}

    monkeypatch.setattr(evaluate, "_ranking_rows_from_embeddings", fake_ranking_rows_from_embeddings)
    monkeypatch.setattr(
        evaluate,
        "_theorem_query_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("query text should only be built for case studies")),
    )
    monkeypatch.setattr(
        evaluate,
        "retrieve_knowledge_for_theorem",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("guidance retrieval should only run for case studies")),
    )

    result = evaluate._evaluate_theorem_retrieval_split(
        "test",
        top_ks=[1],
        train_premises=train_premises,
        case_study_limit=0,
    )

    assert result["case_studies"] == []
    assert result["metrics"]["theorem_retrieval_evaluated_theorems"] == 2
