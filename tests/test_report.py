from __future__ import annotations

from proofatlas.io import write_json
from proofatlas.report import run


def test_report_builds_from_existing_json(tmp_path) -> None:
    write_json(
        tmp_path / "t1_test_proof_state_premise_retrieval.json",
        {
            "methods": {
                "dense": {"Recall@10": 0.1, "Recall@50": 0.2, "Recall@100": 0.3, "MAP": 0.5, "gold_pool_coverage": 0.9},
                "lexical": {"Recall@10": 0.2, "Recall@50": 0.3, "Recall@100": 0.4, "MAP": 0.6, "gold_pool_coverage": 0.9},
                "dense_lexical_rrf": {"Recall@10": 0.3, "Recall@50": 0.4, "Recall@100": 0.5, "MAP": 0.7, "gold_pool_coverage": 0.9},
            }
        },
    )
    write_json(
        tmp_path / "t2_test_theorem_theorem_retrieval.json",
        {
            "premise_coverage": {"Recall@50": 0.2, "Recall@100": 0.3},
            "strategy_facets": {"Recall": 0.4, "AnyHit": 0.5},
            "difficulty_profile": {"MAE": 0.1, "bucket_accuracy": 0.6},
        },
    )
    write_json(tmp_path / "t3_test_guidance_bundles.json", {"bundle_count": 2})

    path = run(split="test", output_dir=str(tmp_path))
    text = path.read_text(encoding="utf-8")
    assert "Proof-State -> Premise Retrieval" in text
    assert "Theorem -> Theorem Pattern Retrieval" in text
    assert "Generated guidance bundles: `2`" in text
