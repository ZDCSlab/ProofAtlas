import json

from leanrank_kg import build_graph, compute_difficulty, download_or_sample, normalize, premise_trace_supervision


def test_premise_trace_supervision_report_uses_leanrank_labels_without_custom_extractor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 23\nuse_huggingface: false\nsample: {total_rows: 80, small_debug_rows: 80, committed_demo_rows: 80}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 80)
    normalize.run(cfg)
    build_graph.run(cfg)
    compute_difficulty.run(cfg)

    report = premise_trace_supervision.run()

    assert report["dataset_name"] == "erbacher/LeanRank-data"
    assert report["current_artifact_supervision"]["has_positive_edges"] is True
    assert report["current_artifact_supervision"]["has_negative_candidates"] is True
    assert report["current_artifact_supervision"]["total_positive_edges"] > 0
    assert report["current_artifact_supervision"]["total_negative_edges"] > 0
    assert report["current_artifact_supervision"]["negative_to_positive_edge_ratio"] > 0
    assert "quality_checks" in report["current_artifact_supervision"]
    assert report["current_artifact_supervision"]["quality_checks"]["all_positive_negative_pairs_disjoint"] is True
    assert "normalization_label_conflicts" in report
    assert "total_positive_negative_overlap_removed" in report["normalization_label_conflicts"]
    assert "negative_candidate_hardness" in report["splits"]["train"]
    assert "hard_negative_quality_profile" in report["splits"]["train"]
    quality = report["splits"]["train"]["hard_negative_quality_profile"]
    assert quality["bucket_method"] == "negative_candidate_hardness"
    assert {row["bucket"] for row in quality["bucket_counts"]} == {"none", "low", "medium", "high"}
    assert sum(row["proof_state_count"] for row in quality["bucket_counts"]) == report["splits"]["train"]["proof_states"]
    assert quality["high_hardness_threshold"] == 0.75
    assert "trace_profile" in report["splits"]["train"]
    assert report["splits"]["train"]["trace_profile"]["positive_trace_rows"] > 0
    assert report["splits"]["train"]["trace_profile"]["negative_candidate_rows"] > 0
    assert report["splits"]["train"]["trace_profile"]["positive_trace_source"].startswith("LeanRank-data")
    assert "example_traces" in report["splits"]["train"]
    assert report["splits"]["train"]["example_traces"]
    first_trace = report["splits"]["train"]["example_traces"][0]
    assert first_trace["positive_premises"]
    assert first_trace["hard_negative_candidates"]
    assert first_trace["negative_candidate_hardness"] >= 0
    assert "positive_proof_state_coverage" in report["splits"]["train"]
    assert "negative_proof_state_coverage" in report["splits"]["train"]
    assert "positive_negative_pair_overlap_count" in report["splits"]["train"]
    assert report["splits"]["train"]["positive_negative_pair_overlap_count"] == 0
    assert "quality_checks" in report["splits"]["train"]
    assert report["scope"] == "erbacher/LeanRank-data normalized positive/negative premise supervision"
    json.dumps(report, allow_nan=False)
