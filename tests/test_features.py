import pandas as pd

from leanrank_kg import build_graph, compute_difficulty, download_or_sample, embed, normalize, train_ranker, weak_label_proof_technique


def _write_config(tmp_path, rows=100):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        f"dataset_name: erbacher/LeanRank-data\nrandom_seed: 11\nuse_huggingface: false\nsample: {{total_rows: {rows}, small_debug_rows: {rows}, committed_demo_rows: {rows}}}\nsplit: {{train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}}\nretrieval: {{top_k: [1, 5, 10]}}\nsimilarity: {{theorem_top_k: 3}}\nproof_techniques: {{max_labels_per_state: 5, minimum_support: 1}}\nembedding: {{backend: tfidf}}\n",
        encoding="utf-8",
    )
    return "configs/sample.yaml"


def test_difficulty_features_are_table_driven(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, 100)
    download_or_sample.run(cfg, 100)
    normalize.run(cfg)
    compute_difficulty.run(cfg)
    ps_features = pd.read_parquet("data/processed/train/proof_state_features.parquet")
    thm_features = pd.read_parquet("data/processed/train/theorem_features.parquet")
    assert ps_features["avg_positive_premise_length"].between(0, 1).all()
    assert ps_features["negative_candidate_hardness"].between(0, 1).all()
    assert thm_features["num_unique_positive_premises"].max() > 0


def test_ranker_features_use_processed_feature_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, 100)
    download_or_sample.run(cfg, 100)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    pairs = train_ranker._pairs("train")
    features = train_ranker._features(pairs, "train")
    assert features["proof_state_difficulty"].between(0, 1).all()
    assert features["negative_candidate_hardness"].between(0, 1).all()
    assert features["premise_frequency"].between(0, 1).all()
    assert features[["proof_state_difficulty", "negative_candidate_hardness", "premise_frequency"]].nunique().sum() > 3
