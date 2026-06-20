import json

import pandas as pd

from leanrank_kg import pipeline_profile
from leanrank_kg.utils import stable_hash, write_json, write_parquet


def test_pipeline_profile_summarizes_leanrank_data_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs/reports").mkdir(parents=True)
    (tmp_path / "configs/proofatlas.yaml").write_text(
        "\n".join(
            [
                "dataset_name: erbacher/LeanRank-data",
                "project_name: ProofAtlas",
                "sample: {total_theorems: 10000, total_rows: 60000}",
                "embedding: {backend: sentence_transformers, device: cuda, batch_size: 512}",
                "index: {backend: sklearn, metric: cosine}",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "dataset_name": "erbacher/LeanRank-data",
        "project_name": "ProofAtlas",
        "sample": {"total_theorems": 10000, "total_rows": 60000},
        "embedding": {"backend": "sentence_transformers", "device": "cuda", "batch_size": 512},
        "index": {"backend": "sklearn", "metric": "cosine"},
    }
    config_hash = stable_hash(json.dumps(config, sort_keys=True), 16)
    write_json(
        "outputs/reports/corpus_manifest.json",
        {
            "dataset_name": "erbacher/LeanRank-data",
            "source_kind": "huggingface",
            "sample_plan": {"total_theorems": 10000, "total_rows": 60000},
            "split_counts": {"train": 80, "val": 10, "test": 10},
        },
    )
    write_json(
        "outputs/reports/index_benchmark.json",
        {
            "entities": {
                "premise": {
                    "backend": "sklearn",
                    "rows": 6000,
                    "top_k": 10,
                    "indexed_available": True,
                    "exact_ms_per_query": 5.0,
                    "indexed_ms_per_query": 3.0,
                    "speedup_vs_exact": 1.6,
                    "recall_at_10_vs_exact": 1.0,
                }
            }
        },
    )
    write_json(
        "outputs/reports/test_set_evaluation.json",
        {
            "evaluation_scope": {
                "is_sampled": False,
                "proof_state_limits": {"test": None, "val": None},
                "theorem_limits": {"test": None, "val": None},
                "total_seconds": 2.5,
                "substage_timings": [
                    {"name": "test_theorem_retrieval", "seconds": 1.5, "evaluated_queries": 5, "actual_backend": "torch_cuda"},
                    {"name": "test_proof_state_retrieval", "seconds": 1.0, "evaluated_queries": 10, "actual_backend": "torch_cuda"},
                ],
            },
            "test": {
                "proof_state_retrieval": {"metrics": {"Recall@10": 0.1, "Recall@100": 0.2}},
                "theorem_retrieval": {"metrics": {"theorem_retrieval_Recall@10": 0.55, "theorem_retrieval_Recall@100": 0.7}},
                "proof_state_reranked_retrieval": {
                    "metrics": {"Recall@10": 0.15},
                    "candidate_k_ablation": [
                        {"candidate_k": 50, "metrics": {"Recall@10": 0.15, "MRR": 0.12, "MAP": 0.08}},
                        {"candidate_k": 100, "metrics": {"Recall@10": 0.12, "MRR": 0.10, "MAP": 0.07}},
                    ],
                },
            },
        },
    )
    write_json(
        "outputs/reports/ranker_validation_metrics.json",
        {
            "feature_ablation": {
                "groups": {
                    "symbol_overlap": {
                        "delta_without_group": 0.03,
                        "auc_group_only": 0.65,
                        "auc_without_group": 0.79,
                        "columns": ["symbol_name_overlap"],
                    },
                    "frequency": {
                        "delta_without_group": 0.01,
                        "auc_group_only": 0.78,
                        "auc_without_group": 0.81,
                        "columns": ["premise_frequency"],
                    },
                }
            }
        },
    )
    write_json(
        "outputs/reports/premise_trace_supervision_report.json",
        {
            "current_artifact_supervision": {
                "has_positive_edges": True,
                "has_negative_candidates": True,
                "negative_to_positive_edge_ratio": 9.0,
            }
        },
    )
    (tmp_path / "data/processed/test").mkdir(parents=True)
    write_parquet(pd.DataFrame({"proof_state_id": [f"ps{i}" for i in range(10)]}), "data/processed/test/proof_states.parquet")
    write_parquet(pd.DataFrame({"theorem_id": [f"thm{i}" for i in range(5)]}), "data/processed/test/theorems.parquet")
    write_json(
        "outputs/reports/pipeline_run_timings.json",
        {
            "config_path": "configs/proofatlas.yaml",
            "config_hash": config_hash,
            "generated_at": "2026-06-20T00:00:00+00:00",
            "passed": True,
            "total_seconds": 10.0,
            "stage_count": 2,
            "slowest_stages": [{"name": "evaluate", "seconds": 3.0}, {"name": "embed", "seconds": 2.0}],
            "stages": [{"name": "evaluate", "status": "passed", "seconds": 3.0}, {"name": "embed", "status": "passed", "seconds": 2.0}],
        },
    )
    report = pipeline_profile.run("configs/proofatlas.yaml")

    assert report["dataset_name"] == "erbacher/LeanRank-data"
    assert report["scale_profile"]["target_dataset_confirmed"] is True
    assert report["scale_profile"]["scale_bucket"] == "large"
    assert "throughput_profile" in report
    assert report["throughput_profile"]["mean_index_speedup_vs_exact"] == 1.6
    assert report["throughput_profile"]["min_index_recall_vs_exact"] == 1.0
    assert report["stages"]["timings"]["config_matches_current"] is True
    assert report["stages"]["timings"]["executed_stage_count"] == 2
    assert report["stages"]["timings"]["skipped_stage_count"] == 0
    assert report["throughput_profile"]["timing_config_matches_current"] is True
    assert report["throughput_profile"]["throughput_basis"] == "executed_pipeline_run"
    assert report["throughput_profile"]["scale_estimate_reliable"] is True
    assert report["throughput_profile"]["bottleneck_profile"]["primary_stage"] == "evaluate"
    assert report["throughput_profile"]["bottleneck_profile"]["primary_stage_share_of_total"] == 0.3
    assert report["throughput_profile"]["bottleneck_profile"]["top3_stage_share_of_total"] == 0.5
    assert report["throughput_profile"]["embedding_bottleneck_profile"]["embed_stage_seconds"] == 2.0
    assert report["throughput_profile"]["embedding_bottleneck_profile"]["embed_stage_share_of_total"] == 0.2
    assert report["throughput_profile"]["embedding_bottleneck_profile"]["embedding_rows_per_embed_second"] == 0.0
    assert report["throughput_profile"]["evaluation_timing_delta"]["timed_pipeline_evaluate_seconds"] == 3.0
    assert report["throughput_profile"]["evaluation_timing_delta"]["current_evaluation_seconds"] == 2.5
    assert report["throughput_profile"]["evaluation_timing_delta"]["timed_to_current_ratio"] == 1.2
    assert report["throughput_profile"]["retrieval_bottleneck_profile"]["proof_state"]["primary_accuracy_bottleneck"] == "candidate_generation_or_embeddings"
    assert report["throughput_profile"]["retrieval_bottleneck_profile"]["theorem"]["primary_accuracy_bottleneck"] == "top10_reranking_or_candidate_ordering"
    rapid = report["throughput_profile"]["rapid_convergence_profile"]
    assert rapid["accuracy_snapshot"]["reranked_minus_embedding_recall_at_10"] == 0.04999999999999999
    assert rapid["rerank_candidate_depth"]["best_by_recall_at_10"]["candidate_k"] == 50
    assert rapid["strongest_ranker_feature_groups"][0]["group"] == "symbol_overlap"
    assert rapid["label_supervision"]["negative_to_positive_edge_ratio"] == 9.0
    assert rapid["recommended_sequence"][0]["area"] == "proof_state_query_and_embedding"
    assert report["stages"]["evaluation"]["evaluation_timing"]["total_seconds"] == 2.5
    assert report["stages"]["evaluation"]["evaluation_timing"]["substage_count"] == 2
    assert report["stages"]["evaluation"]["evaluation_timing"]["slowest_substages"][0]["name"] == "test_theorem_retrieval"
    assert any(row["area"] == "indexing" for row in report["recommendations"])
    assert any(row["area"] == "pipeline_bottleneck" for row in report["recommendations"])
    assert any(row["area"] == "retrieval_accuracy" for row in report["recommendations"])
    assert (tmp_path / "outputs/reports/pipeline_performance_report.json").exists()


def test_pipeline_profile_flags_stale_evaluate_timing_after_speedup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs/reports").mkdir(parents=True)
    (tmp_path / "configs/proofatlas.yaml").write_text(
        "\n".join(
            [
                "dataset_name: erbacher/LeanRank-data",
                "project_name: ProofAtlas",
                "sample: {total_theorems: 10000, total_rows: 60000}",
                "embedding: {backend: sentence_transformers, device: cuda}",
                "index: {backend: hnswlib, metric: cosine}",
            ]
        ),
        encoding="utf-8",
    )
    config_hash = stable_hash(
        json.dumps(
            {
                "dataset_name": "erbacher/LeanRank-data",
                "project_name": "ProofAtlas",
                "sample": {"total_theorems": 10000, "total_rows": 60000},
                "embedding": {"backend": "sentence_transformers", "device": "cuda"},
                "index": {"backend": "hnswlib", "metric": "cosine"},
            },
            sort_keys=True,
        ),
        16,
    )
    write_json("outputs/reports/corpus_manifest.json", {"dataset_name": "erbacher/LeanRank-data", "sample_plan": {"total_theorems": 10000}, "split_counts": {"test": 10}})
    write_json("outputs/reports/index_benchmark.json", {"entities": {"premise": {"backend": "hnswlib", "indexed_available": True, "speedup_vs_exact": 5.0, "recall_at_10_vs_exact": 0.99, "top_k": 10}}})
    write_json(
        "outputs/reports/pipeline_run_timings.json",
        {
            "config_hash": config_hash,
            "passed": True,
            "total_seconds": 100.0,
            "slowest_stages": [{"name": "evaluate", "seconds": 80.0}],
            "stages": [{"name": "evaluate", "status": "passed", "seconds": 80.0}],
        },
    )
    write_json(
        "outputs/reports/test_set_evaluation.json",
        {"evaluation_scope": {"is_sampled": False, "total_seconds": 20.0}},
    )
    (tmp_path / "data/processed/test").mkdir(parents=True)
    write_parquet(pd.DataFrame({"proof_state_id": [f"ps{i}" for i in range(10)]}), "data/processed/test/proof_states.parquet")
    write_parquet(pd.DataFrame({"theorem_id": [f"thm{i}" for i in range(10)]}), "data/processed/test/theorems.parquet")

    report = pipeline_profile.run("configs/proofatlas.yaml")

    delta = report["throughput_profile"]["evaluation_timing_delta"]
    assert delta["timed_to_current_ratio"] == 4.0
    assert delta["current_faster_than_pipeline_timing"] is True
    assert any(row["area"] == "performance_timing" for row in report["recommendations"])


def test_pipeline_profile_recommends_full_timing_for_cached_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs/reports").mkdir(parents=True)
    (tmp_path / "configs/proofatlas.yaml").write_text(
        "\n".join(
            [
                "dataset_name: erbacher/LeanRank-data",
                "project_name: ProofAtlas",
                "sample: {total_theorems: 10000, total_rows: 60000}",
                "embedding: {backend: sentence_transformers, device: cuda, batch_size: 512}",
                "index: {backend: hnswlib, metric: cosine}",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "dataset_name": "erbacher/LeanRank-data",
        "project_name": "ProofAtlas",
        "sample": {"total_theorems": 10000, "total_rows": 60000},
        "embedding": {"backend": "sentence_transformers", "device": "cuda", "batch_size": 512},
        "index": {"backend": "hnswlib", "metric": "cosine"},
    }
    config_hash = stable_hash(json.dumps(config, sort_keys=True), 16)
    write_json(
        "outputs/reports/corpus_manifest.json",
        {
            "dataset_name": "erbacher/LeanRank-data",
            "source_kind": "huggingface",
            "sample_plan": {"total_theorems": 10000, "total_rows": 60000},
            "split_counts": {"train": 80, "val": 10, "test": 10},
        },
    )
    write_json(
        "outputs/reports/index_benchmark.json",
        {
            "entities": {
                "premise": {
                    "backend": "hnswlib",
                    "rows": 6000,
                    "top_k": 10,
                    "indexed_available": True,
                    "exact_ms_per_query": 5.0,
                    "indexed_ms_per_query": 1.0,
                    "speedup_vs_exact": 5.0,
                    "recall_at_10_vs_exact": 0.99,
                }
            }
        },
    )
    write_json(
        "outputs/reports/pipeline_run_timings.json",
        {
            "config_path": "configs/proofatlas.yaml",
            "config_hash": config_hash,
            "generated_at": "2026-06-20T00:00:00+00:00",
            "passed": True,
            "total_seconds": 10.0,
            "stage_count": 2,
            "slowest_stages": [{"name": "homepage", "seconds": 3.0}],
            "stages": [
                {"name": "sample", "status": "skipped", "seconds": 0.0},
                {"name": "homepage", "status": "passed", "seconds": 3.0},
            ],
        },
    )
    write_json(
        "outputs/reports/test_set_evaluation.json",
        {
            "evaluation_scope": {
                "is_sampled": True,
                "proof_state_limits": {"test": 100, "val": 100},
                "theorem_limits": {"test": 50, "val": 50},
            },
            "test": {
                "proof_state_retrieval": {"metrics": {"evaluated_queries": 100}},
                "theorem_retrieval": {"metrics": {"theorem_retrieval_evaluated_theorems": 50}},
            },
        },
    )
    (tmp_path / "data/processed/test").mkdir(parents=True)
    write_parquet(pd.DataFrame({"proof_state_id": [f"ps{i}" for i in range(400)]}), "data/processed/test/proof_states.parquet")
    write_parquet(pd.DataFrame({"theorem_id": [f"thm{i}" for i in range(200)]}), "data/processed/test/theorems.parquet")

    report = pipeline_profile.run("configs/proofatlas.yaml")

    assert report["throughput_profile"]["throughput_basis"] == "cached_or_partial_pipeline_run"
    assert report["throughput_profile"]["scale_estimate_reliable"] is False
    assert report["throughput_profile"]["bottleneck_profile"]["primary_stage"] == "homepage"
    assert report["stages"]["evaluation"]["held_out_test_coverage"]["proof_state_coverage_fraction"] == 0.25
    assert report["stages"]["evaluation"]["held_out_test_coverage"]["theorem_coverage_fraction"] == 0.25
    assert any(row["area"] == "performance_timing" and row["priority"] == "high" for row in report["recommendations"])
    assert any(row["area"] == "evaluation_scope" and row["priority"] == "medium" for row in report["recommendations"])
