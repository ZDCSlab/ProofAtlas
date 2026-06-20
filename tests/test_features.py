import pandas as pd
import json

from leanrank_kg import build_graph, compute_difficulty, download_or_sample, embed, normalize, train_difficulty, train_ranker, weak_label_proof_technique


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
    assert ps_features["theorem_complexity_score"].between(0, 1).all()
    assert set(ps_features["difficulty_target_source"]) == {"proof_length_tactic_count_premise_count_negative_candidates"}
    assert thm_features["num_unique_positive_premises"].max() > 0
    assert thm_features["theorem_complexity_score"].between(0, 1).all()
    assert {"proof_length_score", "tactic_count_score", "premise_count_score", "negative_candidate_count_score"} <= set(thm_features.columns)
    target_report = json.loads((tmp_path / "outputs/reports/difficulty_target_report.json").read_text(encoding="utf-8"))
    assert target_report["target"] == "theorem_features.theorem_complexity_score"


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
    assert features["symbol_name_overlap"].between(0, 1).all()
    assert features["symbol_context_overlap"].between(0, 1).all()
    assert features["graph_premise_degree"].between(0, 1).all()
    assert features["theorem_neighborhood_premise_score"].between(0, 1).all()
    assert features[
        [
            "proof_state_difficulty",
            "negative_candidate_hardness",
            "premise_frequency",
            "symbol_context_overlap",
            "graph_premise_degree",
            "theorem_neighborhood_premise_score",
        ]
    ].nunique().sum() > 6


def test_ranker_pair_sampling_uses_configured_label_cap():
    pairs = pd.DataFrame(
        {
            "label": [1] * 12 + [0] * 30,
            "ps": [f"ps_{idx}" for idx in range(42)],
            "prem": [f"prem_{idx}" for idx in range(42)],
        }
    )

    sampled = train_ranker._sample_pairs_by_label(pairs, max_pairs_per_label=5, random_seed=7)

    assert sampled["label"].value_counts().to_dict() == {0: 5, 1: 5}
    assert sampled.equals(sampled.sort_values(["label", "ps", "prem"]).reset_index(drop=True))


def test_ranker_pair_sampling_can_keep_all_pairs():
    pairs = pd.DataFrame(
        {
            "label": [1, 1, 0],
            "ps": ["ps_b", "ps_a", "ps_c"],
            "prem": ["prem_b", "prem_a", "prem_c"],
        }
    )

    sampled = train_ranker._sample_pairs_by_label(pairs, max_pairs_per_label=0, random_seed=7)

    assert len(sampled) == 3
    assert sampled["ps"].tolist() == ["ps_c", "ps_a", "ps_b"]


def test_ranker_writes_feature_ablation_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, 120)
    download_or_sample.run(cfg, 120)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    train_ranker.run(cfg)
    metrics = json.loads((tmp_path / "outputs/reports/ranker_validation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["feature_columns"]
    assert "feature_groups" in metrics
    utilization = metrics["training_pair_utilization"]
    assert utilization["label_source"]["positive_pairs"].endswith("positive_edges.parquet label=1")
    assert utilization["label_source"]["hard_negative_pairs"].endswith("negative_edges.parquet label=0")
    assert utilization["raw_pair_counts"]["positive"] > 0
    assert utilization["raw_pair_counts"]["hard_negative"] > 0
    assert utilization["training_sample_counts"]["positive"] > 0
    assert utilization["training_sample_counts"]["hard_negative"] > 0
    assert utilization["training_sample_counts"]["hard_negative_to_positive_ratio"] > 0
    assert utilization["hardness_feature"]["column"] == "negative_candidate_hardness"
    assert "negative_candidate_hardness" in utilization["feature_nonzero_rates"]
    if "feature_ablation" in metrics:
        ablation = metrics["feature_ablation"]
        assert "full_auc" in ablation
        assert {"embedding_similarity", "namespace_domain", "proof_technique", "difficulty", "frequency", "symbol_overlap", "graph", "theorem_neighborhood"} <= set(ablation["groups"])
        assert "delta_without_group" in ablation["groups"]["embedding_similarity"]
    assert {"symbol_overlap", "graph", "theorem_neighborhood"} <= set(metrics["feature_groups"])


def test_difficulty_estimator_trains_from_feature_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, 100)
    download_or_sample.run(cfg, 100)
    normalize.run(cfg)
    compute_difficulty.run(cfg)
    train_difficulty.run(cfg)
    assert (tmp_path / "outputs/models/difficulty_estimator.joblib").exists()
    metrics = __import__("json").loads((tmp_path / "outputs/reports/difficulty_estimator_metrics.json").read_text(encoding="utf-8"))
    assert metrics["train"]["rows"] > 0
    assert metrics["train"]["mae"] >= 0
    assert "calibration_bins" in metrics["train"]
    assert "residual_quantiles" in metrics["train"]
    assert {"p50", "p80", "p95"} <= set(metrics["train"]["residual_quantiles"])
    assert "context_length_score" in metrics["feature_columns"]
    assert metrics["target"] == "proof_state_features.theorem_complexity_score"
    assert metrics["target_source"] == "proof_length_tactic_count_premise_count_negative_candidates"
