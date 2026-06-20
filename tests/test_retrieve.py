import json

import pandas as pd

from leanrank_kg import augment_graph, build_graph, build_index, compute_difficulty, download_or_sample, embed, evaluate, homepage, normalize, report, train_difficulty, train_ranker, weak_label_proof_technique
import leanrank_kg.retrieve as retrieve_module
from leanrank_kg.retrieve import (
    _load_split,
    clear_retrieval_caches,
    explain_premise_match,
    get_difficulty_profile,
    get_graph_neighborhood,
    retrieve_knowledge_for_theorem,
    retrieve_premises,
    retrieve_premises_for_query,
    retrieve_similar_proof_states_for_query,
    retrieve_similar_theorems,
    retrieve_similar_theorems_for_query,
)


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
    train_difficulty.run(cfg)
    embed.run(cfg)
    ps_id = pd.read_parquet("data/processed/train/proof_states.parquet").iloc[0]["id"]
    rows = retrieve_premises(ps_id, 3, "train")
    assert rows
    assert {"premise_id", "score"} <= set(rows[0])
    assert (tmp_path / "outputs/embeddings/embedding_config.json").exists()


def test_adaptive_retrieval_policy_expands_candidate_depth_by_difficulty():
    easy = retrieve_module._adaptive_retrieval_policy(10, 100, {"score": 0.2})
    medium = retrieve_module._adaptive_retrieval_policy(10, 100, {"score": 0.5})
    hard = retrieve_module._adaptive_retrieval_policy(10, 100, {"score": 0.8})
    assert easy["difficulty_bucket"] == "easy"
    assert medium["difficulty_bucket"] == "medium"
    assert hard["difficulty_bucket"] == "hard"
    assert easy["candidate_k"] < medium["candidate_k"] < hard["candidate_k"]
    assert hard["candidate_k"] <= 500


def test_retrieval_artifact_cache_can_be_cleared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 16\nuse_huggingface: false\nsample: {total_rows: 30, small_debug_rows: 30, committed_demo_rows: 30}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 30)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    train_difficulty.run(cfg)
    embed.run(cfg)
    first = _load_split("train")
    second = _load_split("train")
    assert first is second
    clear_retrieval_caches()
    third = _load_split("train")
    assert third is not first


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
    train_difficulty.run(cfg)
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
    build_index.run(cfg)
    augment_graph.run(cfg)
    train_ranker.run(cfg)
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
    build_index.run(cfg)
    augment_graph.run(cfg)
    proof_state_id = pd.read_parquet("data/processed/train/proof_states.parquet").iloc[0]["id"]
    json.dumps(get_graph_neighborhood(proof_state_id, 1, "train"), allow_nan=False)
    json.dumps(get_difficulty_profile(proof_state_id, "train"), allow_nan=False)


def test_text_query_retrieval_and_theorem_guidance_are_json_serializable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 15\nuse_huggingface: false\nsample: {total_rows: 100, small_debug_rows: 100, committed_demo_rows: 100}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 100)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    train_difficulty.run(cfg)
    embed.run(cfg)
    build_index.run(cfg)
    augment_graph.run(cfg)
    train_ranker.run(cfg)
    proof_state = pd.read_parquet("data/processed/test/proof_states.parquet").iloc[0]
    query_text = f"{proof_state['context']}\n{proof_state['goal_text']}"
    premises = retrieve_premises_for_query(query_text, 5, index_split="train")
    theorems = retrieve_similar_theorems_for_query(query_text, 3, index_split="train")
    proof_states = retrieve_similar_proof_states_for_query(query_text, 3, index_split="train")
    guidance = retrieve_knowledge_for_theorem(query_text, full_name=str(proof_state["full_name"]), k_premises=5, k_theorems=3)
    assert premises
    assert theorems
    assert proof_states
    assert (tmp_path / "outputs/indexes/train_premise_neighbors.joblib").exists()
    assert (tmp_path / "outputs/indexes/train_proof_state_neighbors.joblib").exists()
    index_summary = json.loads((tmp_path / "outputs/indexes/index_summary.json").read_text(encoding="utf-8"))
    assert index_summary["backend"] == "sklearn"
    assert index_summary["metric"] == "cosine"
    assert index_summary["embedding_config_hash"]
    assert premises[0]["signals"]["retrieval_backend"] == "nearest_neighbors_index"
    assert premises[0]["signals"]["index_manifest_hash"] == index_summary["embedding_config_hash"]
    assert {
        "embedding_score",
        "learned_ranker_score",
        "premise_frequency_score",
        "graph_neighbor_score",
        "graph_premise_degree",
        "theorem_neighborhood_premise_score",
        "symbol_name_overlap",
        "symbol_context_overlap",
    } <= set(premises[0]["signals"])
    assert premises[0]["signals"]["learned_ranker_score"] is not None
    assert premises[0]["ranking_reasons"]
    assert theorems[0]["signals"]["retrieval_backend"] == "nearest_neighbors_index"
    assert theorems[0]["signals"]["index_manifest_hash"] == index_summary["embedding_config_hash"]
    assert proof_states[0]["signals"]["retrieval_backend"] == "nearest_neighbors_index"
    assert proof_states[0]["signals"]["index_manifest_hash"] == index_summary["embedding_config_hash"]
    assert {"ranked_premises", "similar_theorems", "similar_proof_states", "likely_proof_techniques", "difficulty_profile", "retrieval_policy", "graph_evidence"} <= set(guidance)
    assert guidance["similar_proof_states"]
    assert guidance["retrieval_policy"]["enabled"] is True
    assert guidance["retrieval_policy"]["candidate_k"] >= guidance["retrieval_policy"]["requested_k_premises"]
    assert guidance["retrieval_policy"]["policy"] == "expand_candidate_pool_by_pre_retrieval_difficulty"
    assert guidance["difficulty_profile"]["signals"]["calibrated_by"].startswith(("query_heuristic", "query_heuristic_and_similar_theorem_prior"))
    assert "historical_prior_confidence" in guidance["difficulty_profile"]["signals"]
    assert guidance["difficulty_profile"]["signals"]["trained_estimator_available"] is True
    assert guidance["difficulty_profile"]["signals"]["trained_estimator_score"] is not None
    assert guidance["difficulty_profile"]["signals"]["trained_estimator_uncertainty"] is not None
    assert guidance["difficulty_profile"]["signals"]["trained_estimator_confidence_interval"] is not None
    assert "trained_estimator_calibration_bins" in guidance["difficulty_profile"]
    assert "similar_theorem_difficulty_neighbors" in guidance["difficulty_profile"]
    assert guidance["lean_diagnostics"]["checked"] is False
    assert guidance["query"]["lean_proof_state_extraction"]["failure_reason"] == "lean_validation_not_requested"
    assert {"query_id", "input_type", "goal_text", "local_hypotheses", "retrieval_text"} <= set(guidance["query"])
    assert {"binder_groups", "conclusion_symbols", "normalized_goal_text", "parsed_feature_summary"} <= set(guidance["query"])
    assert guidance["query"]["input_type"] in {"theorem", "proof_state"}
    assert "conclusion_symbol_score" in premises[0]["signals"]
    assert "parsed_symbol_overlap_score" in premises[0]["signals"]
    monkeypatch.setattr(
        retrieve_module,
        "check_lean_syntax",
        lambda source: {
            "checked": True,
            "available": True,
            "ok": False,
            "command": ["lean"],
            "returncode": 1,
            "stdout": "",
            "stderr": "error: unsolved goals",
            "proof_states": [
                {
                    "goal_text": "x = x",
                    "local_hypotheses": ["x : Nat"],
                    "symbols": ["x", "Nat", "="],
                    "namespace_hints": [],
                    "typeclass_hints": [],
                    "raw_text": "x : Nat\n⊢ x = x",
                    "retrieval_text": "x : Nat\n⊢ x = x",
                }
            ],
            "proof_state_extraction": {
                "method": "lean_unsolved_goals_diagnostic",
                "has_unsolved_goals": True,
                "raw_block_count": 1,
                "extracted_count": 1,
                "failure_reason": None,
                "rejected_blocks": [],
            },
            "summary": {"has_unsolved_goals": True, "error_count": 1, "warning_count": 0},
        },
    )
    lean_guidance = retrieve_knowledge_for_theorem("theorem t (x : Nat) : x = x := by sorry", k_premises=3, k_theorems=2, validate_lean=True)
    assert lean_guidance["query"]["retrieval_query_source"] == "lean_diagnostics_proof_states"
    assert lean_guidance["query"]["lean_extracted_proof_state_count"] == 1
    assert lean_guidance["query"]["lean_proof_state_extraction"]["raw_block_count"] == 1
    assert lean_guidance["lean_diagnostics"]["summary"]["has_unsolved_goals"] is True
    json.dumps(guidance, allow_nan=False)
    evaluate.run(cfg)
    metrics = json.loads((tmp_path / "outputs/reports/metrics.json").read_text(encoding="utf-8"))
    test_eval = json.loads((tmp_path / "outputs/reports/test_set_evaluation.json").read_text(encoding="utf-8"))
    case_studies = json.loads((tmp_path / "outputs/reports/theorem_retrieval_case_studies.json").read_text(encoding="utf-8"))
    assert "theorem_retrieval_Recall@10" in metrics
    assert metrics["theorem_retrieval_evaluated_theorems"] > 0
    assert test_eval["candidate_pool"] == "train premise index"
    assert "Recall@10" in test_eval["test"]["proof_state_retrieval"]["metrics"]
    assert "theorem_retrieval_Recall@10" in test_eval["test"]["theorem_retrieval"]["metrics"]
    assert test_eval["test"]["proof_state_retrieval"]["domain_breakdown"]
    assert test_eval["test"]["theorem_retrieval"]["domain_breakdown"]
    assert test_eval["test"]["proof_state_retrieval"]["worst_cases"]
    assert test_eval["test"]["theorem_retrieval"]["worst_cases"]
    assert "domain_tag" in test_eval["test"]["proof_state_retrieval"]["domain_breakdown"][0]
    assert "rank_of_first_gold" in test_eval["test"]["proof_state_retrieval"]["worst_cases"][0]
    assert metrics["theorem_retrieval_gold_premises_total"] >= metrics["theorem_retrieval_gold_premises_in_train_index"]
    assert metrics["theorem_retrieval_gold_premises_total"] == metrics["theorem_retrieval_gold_premises_in_train_index"] + metrics["theorem_retrieval_gold_premises_missing_from_train_index"]
    assert case_studies
    assert "gold_premises_missing_from_train_index" in case_studies[0]
    assert case_studies[0]["guidance"]["ranked_premises"][0]["ranking_reasons"]
    homepage.run(cfg)
    assert (tmp_path / "homepage/assets/theorem_retrieval_case_studies.json").exists()
    assert (tmp_path / "homepage/assets/corpus_manifest.json").exists()
    assert (tmp_path / "homepage/assets/refresh_dashboard.json").exists()
    assert (tmp_path / "homepage/assets/refresh_trend.json").exists()
    assert (tmp_path / "homepage/assets/refresh_history.json").exists()
    homepage_summary = json.loads((tmp_path / "homepage/assets/homepage_summary.json").read_text(encoding="utf-8"))
    refresh_dashboard = json.loads((tmp_path / "outputs/reports/refresh_dashboard.json").read_text(encoding="utf-8"))
    corpus_manifest = json.loads((tmp_path / "homepage/assets/corpus_manifest.json").read_text(encoding="utf-8"))
    refresh_trend = json.loads((tmp_path / "outputs/reports/refresh_trend.json").read_text(encoding="utf-8"))
    refresh_history = json.loads((tmp_path / "outputs/reports/refresh_history.json").read_text(encoding="utf-8"))
    assert "quality_gates" in refresh_dashboard
    assert "retrieval_quality" in refresh_dashboard
    assert "parsing" in refresh_dashboard
    assert "theorem_query_parse_coverage" in refresh_dashboard["parsing"]
    assert "minimum_theorem_query_parse_coverage" in refresh_dashboard["parsing"]
    assert "data_supervision" in refresh_dashboard["corpus"]
    assert "production_evidence" in homepage_summary
    assert "heldout" in homepage_summary["production_evidence"]
    assert "supervision" in homepage_summary["production_evidence"]
    assert "timing" in homepage_summary["production_evidence"]
    assert corpus_manifest["data_supervision"]["kind"] == "synthetic_demo_rows"
    assert "trend" in refresh_dashboard
    assert "deltas" in refresh_trend
    assert refresh_history["entry_count"] >= 1
    assert refresh_history["latest"] == refresh_history["entries"][-1]
    graph_asset = json.loads((tmp_path / "homepage/assets/graph_visualization.json").read_text(encoding="utf-8"))
    assert graph_asset["nodes"]
    assert graph_asset["edges"]
    html = (tmp_path / "homepage/index.html").read_text(encoding="utf-8")
    assert all(line.rstrip() == line for line in html.splitlines())
    assert "LeanRank Proof Knowledge Graph" in html
    assert "Interactive Proof Guidance Workbench" in html
    assert "Get Proof Guidance" in html
    assert "sample-theorems" in html
    assert "New Theorem Proof Guidance" in html
    assert "Train gold premises" in html
    assert "missing from train" in html
    assert "Knowledge Graph Overview" in html
    assert "Edge Types" in html
    assert "kg-edge-legend" in html
    assert "kg-detail" in html
    assert "Click a graph node or edge" in html
    assert "edgeDescriptions" in html
    assert "theorem uses premise" in html
    assert "Proof Guidance Panel" in html
    assert "Why Were These Premises Recommended?" in html
    assert "Evaluation And Examples" in html
    assert "Local asset fallback" in html
    assert "renderLocalFallback" in html
    assert "localCaseScore" in html
    assert "API unavailable; showing nearest precomputed case study" in html
    assert "Query theorem" in html
    assert "Top premise" in html
    assert "Similar theorem" in html
    assert "Suggested technique" in html
    assert "precomputed theorem guidance examples" in html
    assert "renderExplanationPaths" in html
    assert "shared namespace" in html
    assert "Graph evidence for" in html
    assert "historical premise frequency" in html
    assert "Pipeline Summary" in html
    assert "Refresh Dashboard" in html
    assert "Production Evidence" in html
    assert "Proof-state test coverage" in html
    assert "Theorem test coverage" in html
    assert "Positive premise edges" in html
    assert "Negative candidates" in html
    assert "Scale estimate reliable" in html
    assert "Embedding throughput" in html
    assert "Trend baseline" in html
    assert "History entries" in html
    assert "kg-svg" in html
    assert "Lean check" in html
    assert "Query source" in html
    assert "Corpus source" in html
    assert "Data supervision" in html
    assert "synthetic_demo_rows" in html
    assert "Config hash" in html
    assert "retrieve-theorem-guidance" in html


def test_refresh_trend_reports_metric_deltas():
    current = {
        "corpus": {"config_hash": "new"},
        "scale": {"total_train_val_test": {"theorems": 12, "proof_states": 20, "premises": 8, "nodes": 40, "edges": 55}},
        "retrieval_quality": {"theorem_recall_at_10": 0.7, "theorem_mrr": 0.5},
        "parsing": {"minimum_context_coverage": 0.9},
        "difficulty": {"train_mae": 0.1},
        "index_benchmark": {"premise": {"recall_vs_exact": 1.0}},
        "artifact_compatibility": {"passed": True},
        "quality_gates": {"artifact_compatible": True, "schema_clean": True},
        "ready_for_refresh_comparison": True,
    }
    previous = {
        "corpus": {"config_hash": "old"},
        "scale": {"total_train_val_test": {"theorems": 10, "proof_states": 18, "premises": 7, "nodes": 35, "edges": 50}},
        "retrieval_quality": {"theorem_recall_at_10": 0.6, "theorem_mrr": 0.4},
        "parsing": {"minimum_context_coverage": 0.8},
        "difficulty": {"train_mae": 0.2},
        "index_benchmark": {"premise": {"recall_vs_exact": 0.95}},
        "artifact_compatibility": {"passed": False},
        "quality_gates": {"artifact_compatible": False, "schema_clean": True},
        "ready_for_refresh_comparison": False,
    }
    trend = report._refresh_trend(current, previous)
    assert trend["has_previous"] is True
    assert trend["deltas"]["theorems"]["absolute"] == 2
    assert round(trend["deltas"]["theorem_recall_at_10"]["absolute"], 3) == 0.1
    assert trend["quality_gate_changes"]["artifact_compatible"] == {"current": True, "previous": False}


def test_refresh_history_appends_and_truncates():
    dashboard = {
        "corpus": {"config_hash": "new"},
        "scale": {"total_train_val_test": {"theorems": 12}},
        "retrieval_quality": {"theorem_recall_at_10": 0.7},
        "parsing": {"minimum_context_coverage": 0.9},
        "difficulty": {"train_mae": 0.1},
        "index_benchmark": {"premise": {"recall_vs_exact": 1.0}},
        "artifact_compatibility": {"passed": True},
        "quality_gates": {"artifact_compatible": True},
        "ready_for_refresh_comparison": True,
    }
    trend = report._refresh_trend(dashboard, None)
    history = report._refresh_history(
        [{"generated_at": "old-1", "metrics": {"theorems": 1}}, {"generated_at": "old-2", "metrics": {"theorems": 2}}],
        dashboard,
        trend,
        limit=2,
    )
    assert history["entry_count"] == 2
    assert history["entries"][0]["generated_at"] == "old-2"
    assert history["latest"]["metrics"]["theorems"] == 12


def test_text_query_retrieval_falls_back_when_index_manifest_mismatches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 17\nuse_huggingface: false\nsample: {total_rows: 60, small_debug_rows: 60, committed_demo_rows: 60}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\nindex: {backend: sklearn, metric: cosine, n_neighbors: 25}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 60)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    build_index.run(cfg)
    manifest_path = tmp_path / "outputs/indexes/train_premise_index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embedding_config_hash"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    clear_retrieval_caches()
    rows = retrieve_premises_for_query("⊢ x = x", 3, index_split="train")
    assert rows
    assert rows[0]["signals"]["retrieval_backend"] == "direct_cosine"
    assert rows[0]["signals"]["index_manifest_hash"] == ""
