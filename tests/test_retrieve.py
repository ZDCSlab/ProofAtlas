import json

import pandas as pd

from leanrank_kg import augment_graph, build_graph, compute_difficulty, download_or_sample, embed, normalize, weak_label_proof_technique
from leanrank_kg.retrieve import explain_premise_match, get_difficulty_profile, get_graph_neighborhood, retrieve_premises, retrieve_similar_theorems


def test_retrieve_returns_json_serializable_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 3\nuse_huggingface: false\nsample: {total_rows: 30, small_debug_rows: 30, committed_demo_rows: 30}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 30)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    ps_id = pd.read_parquet("data/processed/train/proof_states.parquet").iloc[0]["id"]
    rows = retrieve_premises(ps_id, 3, "train")
    assert rows
    assert {"premise_id", "score"} <= set(rows[0])
    assert (tmp_path / "outputs/embeddings/embedding_config.json").exists()


def test_retrieve_val_query_uses_train_premise_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 8\nuse_huggingface: false\nsample: {total_rows: 80, small_debug_rows: 80, committed_demo_rows: 80}\nsplit: {train_ratio: 0.7, val_ratio: 0.15, test_ratio: 0.15}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 80)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    ps_id = pd.read_parquet("data/processed/val/proof_states.parquet").iloc[0]["id"]
    train_premise_ids = set(pd.read_parquet("data/processed/train/premises.parquet")["id"])
    rows = retrieve_premises(ps_id, 5, split="val")
    assert rows
    assert {row["premise_id"] for row in rows} <= train_premise_ids
    assert {row["index_split"] for row in rows} == {"train"}


def test_explain_premise_match_includes_required_signals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 10\nuse_huggingface: false\nsample: {total_rows: 60, small_debug_rows: 60, committed_demo_rows: 60}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 60)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    edge = pd.read_parquet("data/processed/train/positive_edges.parquet").iloc[0]
    explanation = explain_premise_match(edge["proof_state_id"], edge["premise_id"], split="train")
    assert {"cosine_score", "namespace_match", "file_match", "domain_match", "shared_proof_techniques"} <= set(explanation)
    assert isinstance(explanation["file_match"], bool)


def test_similar_theorems_use_multiple_non_gnn_signals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 9\nuse_huggingface: false\nsample: {total_rows: 100, small_debug_rows: 100, committed_demo_rows: 100}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 100)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    augment_graph.run(cfg)
    theorem_id = pd.read_parquet("data/processed/train/theorems.parquet").iloc[0]["id"]
    rows = retrieve_similar_theorems(theorem_id, 3, "train")
    assert rows
    assert all(row["theorem_id"] != theorem_id for row in rows)
    assert {"tfidf_similarity", "shared_premise_score", "file_namespace_score", "proof_technique_overlap", "difficulty_similarity"} <= set(rows[0]["signals"])
    edges = pd.read_parquet("outputs/graph/train/edges_enriched.parquet")
    sim_edges = edges[edges["edge_type"] == "similar_to_theorem"]
    assert not sim_edges.empty
    assert {"shared_premise_score", "difficulty_similarity"} <= set(sim_edges.columns)


def test_neighborhood_and_difficulty_are_strict_json_serializable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 13\nuse_huggingface: false\nsample: {total_rows: 80, small_debug_rows: 80, committed_demo_rows: 80}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 80)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    augment_graph.run(cfg)
    proof_state_id = pd.read_parquet("data/processed/train/proof_states.parquet").iloc[0]["id"]
    json.dumps(get_graph_neighborhood(proof_state_id, 1, "train"), allow_nan=False)
    json.dumps(get_difficulty_profile(proof_state_id, "train"), allow_nan=False)
