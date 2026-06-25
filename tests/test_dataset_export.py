from __future__ import annotations

import pandas as pd

from proofatlas.dataset_export import PROCESSED_TABLES, run


def _write_minimal_split(root, split: str) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    theorem_id = f"thm:{split}.one"
    proof_state_id = f"ps:{split}.one"
    frames = {
        "theorems": pd.DataFrame(
            [
                {
                    "id": theorem_id,
                    "full_name": f"{split}.one",
                    "file_path": f"{split}.lean",
                    "domain_tag": "Algebra",
                    "subdomain_tag": "Group",
                    "split": split,
                }
            ]
        ),
        "proof_states": pd.DataFrame(
            [
                {
                    "id": proof_state_id,
                    "theorem_id": theorem_id,
                    "full_name": f"{split}.one",
                    "tactic_idx": 0,
                    "context": "",
                    "goal_text": "prove x = x",
                    "local_hypotheses": ["x : Nat"],
                    "symbols": ["Eq", "Nat"],
                    "tactic": "rfl",
                    "domain_tag": "Algebra",
                    "subdomain_tag": "Group",
                    "split": split,
                }
            ]
        ),
        "premises": pd.DataFrame([{"id": f"premise:{split}.rfl", "full_name": "rfl", "file_path": f"{split}.lean"}]),
        "positive_edges": pd.DataFrame([{"proof_state_id": proof_state_id, "premise_id": f"premise:{split}.rfl"}]),
        "negative_edges": pd.DataFrame([{"proof_state_id": proof_state_id, "premise_id": f"premise:{split}.bad"}]),
        "file_modules": pd.DataFrame([{"file_path": f"{split}.lean", "module": split}]),
        "premise_techniques": pd.DataFrame([{"premise_id": f"premise:{split}.rfl", "label": "rewrite_transport"}]),
        "proof_state_features": pd.DataFrame([{"proof_state_id": proof_state_id, "difficulty_score": 0.1}]),
        "proof_state_techniques": pd.DataFrame(
            [{"proof_state_id": proof_state_id, "technique_id": "rewrite_transport", "label": "rewrite_transport"}]
        ),
        "proof_techniques": pd.DataFrame([{"theorem_id": theorem_id, "label": "rewrite_transport"}]),
        "theorem_features": pd.DataFrame(
            [
                {
                    "theorem_id": theorem_id,
                    "mean_proof_state_difficulty": 0.1,
                    "max_proof_state_difficulty": 0.1,
                    "mean_negative_candidate_hardness": 0.2,
                    "max_tactic_idx": 0,
                    "num_proof_states": 1,
                    "num_unique_positive_premises": 1,
                    "num_failed_negative_candidates": 0,
                    "proof_length_score": 0.1,
                    "tactic_count_score": 0.1,
                    "premise_count_score": 0.1,
                    "negative_candidate_count_score": 0.0,
                    "theorem_complexity_score": 0.1,
                    "difficulty_target_source": "test",
                    "difficulty_bucket": "easy",
                }
            ]
        ),
    }
    for table in PROCESSED_TABLES:
        frames[table].to_parquet(split_dir / f"{table}.parquet", index=False)


def _write_enrichment(root, split: str) -> None:
    llm_dir = root / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "theorem_id": f"thm:{split}.one",
                "full_name": f"{split}.one",
                "split": split,
                "topic": "reflexivity of equality",
                "mathematical_objects": ["Eq", "Nat"],
                "goal_pattern": "prove a reflexive equality",
                "key_symbols": ["Eq"],
                "useful_lemma_types": ["reflexivity lemma"],
                "strategy_facets": [{"facet": "rewrite_transport", "target": "goal", "direction_or_action": "close by reflexivity"}],
                "likely_tactics": ["rfl"],
                "difficulty_reasons": ["direct equality"],
                "difficulty_bucket_hint": "easy",
                "prompt_version": "test_prompt",
                "prompt_hash": "abc",
                "model": "deepseek-chat",
            }
        ]
    ).to_parquet(llm_dir / f"theorem_enrichment_{split}.parquet", index=False)


def test_export_enriched_dataset_builds_artifact(tmp_path) -> None:
    processed = tmp_path / "processed"
    llm_output = tmp_path / "outputs"
    output_root = tmp_path / "enriched"
    for split in ["train", "val", "test"]:
        _write_minimal_split(processed, split)
        _write_enrichment(llm_output, split)

    manifest = run(processed_dir=str(processed), llm_output_dir=str(llm_output), output_root=str(output_root))
    output_dir = output_root / "v1"

    assert manifest["splits"]["train"]["llm_enriched_theorems"] == 1
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "dataset_card.md").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "hf" / "theorems" / "train.parquet").exists()
    assert (output_dir / "hf" / "theorems" / "validation.parquet").exists()
    assert (output_dir / "hf" / "theorem_profiles" / "test.parquet").exists()

    theorems = pd.read_parquet(output_dir / "train" / "theorems.parquet")
    profiles = pd.read_parquet(output_dir / "train" / "theorem_profiles.parquet")
    hf_theorems = pd.read_parquet(output_dir / "hf" / "theorems" / "train.parquet")
    assert "llm_topic" in theorems.columns
    assert theorems.loc[0, "llm_enrichment_available"]
    assert "llm_topic" in hf_theorems.columns
    assert "base_profile_text" in profiles.columns
    assert "reflexivity of equality" in profiles.loc[0, "profile_text"]
