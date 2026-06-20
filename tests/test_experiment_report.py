import json

from leanrank_kg import experiment_report
from leanrank_kg.utils import stable_hash, write_json


def test_experiment_report_documents_ml_task_and_final_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs/reports").mkdir(parents=True)
    (tmp_path / "configs/proofatlas.yaml").write_text(
        "\n".join(
            [
                "dataset_name: erbacher/LeanRank-data",
                "embedding: {backend: tfidf}",
                "index: {backend: sklearn}",
            ]
        ),
        encoding="utf-8",
    )
    config_hash = stable_hash(
        json.dumps(
            {
                "dataset_name": "erbacher/LeanRank-data",
                "embedding": {"backend": "tfidf"},
                "index": {"backend": "sklearn"},
            },
            sort_keys=True,
        ),
        16,
    )
    write_json(
        "outputs/reports/corpus_manifest.json",
        {
            "dataset_name": "erbacher/LeanRank-data",
            "source_kind": "huggingface",
            "split_counts": {"train": {"rows": 80}, "val": {"rows": 10}, "test": {"rows": 10}},
            "data_supervision": {
                "kind": "leanrank_trace_rows",
                "has_tactic_states": True,
                "has_true_positive_premises": True,
            },
        },
    )
    write_json(
        "outputs/reports/test_set_evaluation.json",
        {
            "candidate_pool": "train premise index",
            "label_policy": "held-out test positive_edges are used only for evaluation",
            "test": {
                "proof_state_retrieval": {
                    "metrics": {"evaluated_queries": 3, "Recall@10": 0.5, "MRR": 0.4, "MAP": 0.3, "nDCG@10": 0.45},
                    "domain_breakdown": [{"domain_tag": "Algebra", "metrics": {"evaluated_queries": 2, "Recall@10": 0.5, "MRR": 0.4, "MAP": 0.3, "nDCG@10": 0.45, "gold_premise_coverage": 1.0}}],
                    "failure_profile": {
                        "evaluated_queries": 3,
                        "retrievable_queries": 2,
                        "queries_without_train_gold": 1,
                        "queries_with_missing_gold": 1,
                        "zero_recall_at_max_k": 1,
                        "rank_buckets": {"rank_1": 1, "miss_top_100": 1, "no_train_gold": 1},
                        "gold_coverage_buckets": {"full_train_gold_coverage": 1, "partial_train_gold_coverage": 1, "no_train_gold_coverage": 1},
                        "zero_recall_domains": [{"domain_tag": "Algebra", "zero_recall_queries": 1}],
                    },
                    "worst_cases": [{"proof_state_id": "ps_bad", "rank_of_first_gold": 27, "Recall@10": 0.0, "MRR": 0.0, "gold_premises_total": 1, "gold_premises_in_train_index": 1}],
                },
                "theorem_retrieval": {
                    "metrics": {"theorem_retrieval_evaluated_theorems": 2, "theorem_retrieval_Recall@10": 0.25},
                    "domain_breakdown": [{"domain_tag": "Algebra", "metrics": {"theorem_retrieval_evaluated_queries": 1, "theorem_retrieval_Recall@10": 0.25}}],
                    "failure_profile": {
                        "evaluated_queries": 2,
                        "retrievable_queries": 1,
                        "queries_without_train_gold": 1,
                        "queries_with_missing_gold": 1,
                        "zero_recall_at_max_k": 1,
                        "rank_buckets": {"miss_top_100": 1, "no_train_gold": 1},
                        "gold_coverage_buckets": {"partial_train_gold_coverage": 1, "no_train_gold_coverage": 1},
                        "zero_recall_domains": [{"domain_tag": "Algebra", "zero_recall_queries": 1}],
                    },
                    "worst_cases": [{"full_name": "Mathlib.Bad", "rank_of_first_gold": 19, "theorem_retrieval_Recall@10": 0.0, "theorem_retrieval_MRR": 0.0, "gold_premises_total": 2, "gold_premises_in_train_index": 1}],
                },
            },
            "validation": {
                "proof_state_retrieval": {"metrics": {"evaluated_queries": 2, "Recall@10": 0.6}},
                "theorem_retrieval": {"metrics": {"theorem_retrieval_evaluated_theorems": 1, "theorem_retrieval_Recall@10": 0.2}},
            },
        },
    )
    write_json(
        "outputs/reports/pipeline_performance_report.json",
        {
            "scale_profile": {"leanrank_premise_supervision_ready": True},
            "throughput_profile": {
                "total_embedding_rows": 42,
                "timing_config_matches_current": True,
                "throughput_basis": "executed_pipeline_run",
                "scale_estimate_reliable": True,
                "embedding_rows_by_entity": {"proof_state": 10, "premise": 20, "theorem": 12},
                "embedding_bottleneck_profile": {
                    "embed_stage_seconds": 2.0,
                    "embed_stage_share_of_total": 0.2,
                    "embedding_rows_per_embed_second": 21.0,
                    "embedding_rows_by_split": {"train": 30, "test": 12},
                    "embedding_matrix_bytes": 1234,
                },
                "retrieval_bottleneck_profile": {
                    "proof_state": {
                        "recall_at_10": 0.2,
                        "recall_at_100": 0.3,
                        "top10_to_top100_gap": 0.1,
                        "top10_fraction_of_top100": 0.6666666667,
                        "primary_accuracy_bottleneck": "candidate_generation_or_embeddings",
                    },
                    "theorem": {
                        "recall_at_10": 0.5,
                        "recall_at_100": 0.7,
                        "top10_to_top100_gap": 0.2,
                        "top10_fraction_of_top100": 0.7142857143,
                        "primary_accuracy_bottleneck": "top10_reranking_or_candidate_ordering",
                    },
                },
                "rapid_convergence_profile": {
                    "accuracy_snapshot": {
                        "proof_state_recall_at_10": 0.2,
                        "proof_state_recall_at_100": 0.3,
                        "theorem_recall_at_10": 0.5,
                        "theorem_recall_at_100": 0.7,
                        "reranked_proof_state_recall_at_10": 0.25,
                        "reranked_minus_embedding_recall_at_10": 0.05,
                    },
                    "headroom": {
                        "proof_state_missing_from_top100": 0.7,
                        "proof_state_top10_to_top100_gap": 0.1,
                        "theorem_missing_from_top100": 0.3,
                        "theorem_top10_to_top100_gap": 0.2,
                    },
                    "strongest_ranker_feature_groups": [
                        {
                            "group": "symbol_overlap",
                            "delta_without_group": 0.03,
                            "auc_group_only": 0.65,
                            "columns": ["symbol_name_overlap"],
                        }
                    ],
                    "recommended_sequence": [
                        {
                            "priority": 1,
                            "area": "proof_state_query_and_embedding",
                            "target_metric": "proof_state Recall@100",
                            "current_value": 0.3,
                            "reason": "Candidate pool is weak.",
                        }
                    ],
                },
                "metric_uncertainty_profile": {
                    "method": "bounded_normal_approximation_for_aggregate_retrieval_metrics",
                    "confidence_level": 0.95,
                    "note": "Intervals are approximate diagnostics.",
                    "proof_state": {
                        "Recall@10": {"value": 0.2, "n": 10, "ci95_low": 0.0, "ci95_high": 0.4479, "ci95_half_width": 0.2479},
                        "Recall@100": {"value": 0.3, "n": 10, "ci95_low": 0.0159, "ci95_high": 0.5841, "ci95_half_width": 0.2841},
                        "MRR": {"value": 0.4, "n": 10, "ci95_low": 0.0964, "ci95_high": 0.7036, "ci95_half_width": 0.3036},
                        "MAP": {"value": 0.3, "n": 10, "ci95_low": 0.0159, "ci95_high": 0.5841, "ci95_half_width": 0.2841},
                        "nDCG@10": {"value": 0.45, "n": 10, "ci95_low": 0.1419, "ci95_high": 0.7581, "ci95_half_width": 0.3081},
                    },
                    "theorem": {
                        "theorem_retrieval_Recall@10": {"value": 0.5, "n": 8, "ci95_low": 0.1535, "ci95_high": 0.8465, "ci95_half_width": 0.3465},
                        "theorem_retrieval_Recall@100": {"value": 0.7, "n": 8, "ci95_low": 0.3821, "ci95_high": 1.0, "ci95_half_width": 0.3179},
                        "theorem_retrieval_MRR": {"value": 0.6, "n": 8, "ci95_low": 0.2605, "ci95_high": 0.9395, "ci95_half_width": 0.3395},
                        "theorem_retrieval_MAP": {"value": 0.5, "n": 8, "ci95_low": 0.1535, "ci95_high": 0.8465, "ci95_half_width": 0.3465},
                        "theorem_retrieval_nDCG@10": {"value": 0.55, "n": 8, "ci95_low": 0.2050, "ci95_high": 0.8950, "ci95_half_width": 0.3450},
                    },
                },
                "refresh_reuse_profile": {
                    "reuse_by_default": True,
                    "training_repeat_policy": "Do not retrain by default. Reuse embeddings, indexes, and trained models for report/homepage refreshes.",
                    "artifact_cache": {
                        "embedding_rows": 42,
                        "embedding_model": "tfidf",
                        "indexed_entity_count": 3,
                        "index_backend": "sklearn",
                        "premise_ranker_exists": True,
                        "difficulty_estimator_exists": True,
                    },
                    "scenarios": [
                        {
                            "scenario": "report_or_homepage_refresh",
                            "rerun_embedding": False,
                            "rerun_ranker_training": False,
                            "rerun_evaluation": False,
                            "commands": ["leanrank-kg profile-pipeline", "leanrank-kg build-homepage"],
                        },
                        {
                            "scenario": "ranker_feature_or_label_change",
                            "rerun_embedding": False,
                            "rerun_ranker_training": True,
                            "rerun_evaluation": True,
                            "commands": ["leanrank-kg train-ranker", "leanrank-kg evaluate"],
                        },
                        {
                            "scenario": "embedding_model_or_text_change",
                            "rerun_embedding": True,
                            "rerun_ranker_training": True,
                            "rerun_evaluation": True,
                            "commands": ["leanrank-kg embed", "leanrank-kg build-index"],
                        },
                    ],
                },
                "resource_parallelism_profile": {
                    "embedding_parallelism": {
                        "backend": "tfidf",
                        "model_name": "tfidf",
                        "requested_device": None,
                        "devices": [],
                        "device_count": 0,
                        "multi_process": False,
                        "batch_size": None,
                        "total_embedding_rows": 42,
                        "embedding_rows_per_embed_second": 21.0,
                    },
                    "evaluation_parallelism": {
                        "ranking_backend": "batched_embedding_topk",
                        "requested_use_gpu": True,
                        "requested_gpu_device": "cuda:0",
                        "actual_backends": ["torch_cuda"],
                        "test_proof_state_backend": "torch_cuda",
                        "test_theorem_backend": "torch_cuda",
                        "candidate_count": 1000,
                        "fallback_reasons": [],
                    },
                    "index_parallelism": {
                        "backend": "hnswlib",
                        "requested_backend": "auto",
                        "metric": "cosine",
                        "hnsw_M": 16,
                        "hnsw_ef_construction": 200,
                        "hnsw_ef_search": 100,
                        "indexed_entities": ["premise"],
                        "mean_speedup_vs_exact": 12.5,
                        "min_recall_vs_exact": 0.98,
                    },
                    "cpu_or_io_heavy_stages": [{"name": "validate", "seconds": 2.0, "share_of_total": 0.1}],
                },
                "processed_rows_per_second": 1000.0,
                "pipeline_seconds_per_100k_processed_rows": 100.0,
                "slowest_stage": "evaluate",
                "evaluation_timing_delta": {
                    "timed_pipeline_evaluate_seconds": 5.0,
                    "current_evaluation_seconds": 4.0,
                    "timed_to_current_ratio": 1.25,
                },
                "bottleneck_profile": {
                    "primary_stage": "evaluate",
                    "primary_stage_share_of_total": 0.4,
                    "top3_stage_share_of_total": 0.4,
                    "top_stages": [{"name": "evaluate", "seconds": 5.0, "share_of_total": 0.4}],
                },
                "mean_index_speedup_vs_exact": 12.5,
                "min_index_recall_vs_exact": 0.98,
                "estimated_seconds_at_requested_source_rows": 350.0,
            },
            "stages": {
                "evaluation": {
                    "evaluation_timing": {
                        "total_seconds": 4.0,
                        "substage_count": 2,
                        "slowest_substages": [
                            {"name": "test_theorem_retrieval", "seconds": 2.5, "evaluated_queries": 2, "actual_backend": "torch_cuda"},
                            {"name": "test_proof_state_retrieval", "seconds": 1.5, "evaluated_queries": 3, "actual_backend": "torch_cuda"},
                        ],
                    },
                    "held_out_test_coverage": {
                        "proof_state_evaluated_queries": 3,
                        "proof_state_total": 30,
                        "proof_state_coverage_fraction": 0.1,
                        "theorem_evaluated_queries": 2,
                        "theorem_total": 20,
                        "theorem_coverage_fraction": 0.1,
                    }
                }
            },
            "recommendations": [],
        },
    )
    write_json(
        "outputs/reports/index_benchmark.json",
        {
            "entities": {
                "premise": {
                    "backend": "hnswlib",
                    "rows": 1000,
                    "exact_ms_per_query": 12.0,
                    "indexed_ms_per_query": 1.0,
                    "speedup_vs_exact": 12.0,
                    "recall_at_1_vs_exact": 0.9,
                    "recall_at_5_vs_exact": 0.95,
                    "recall_at_10_vs_exact": 0.98,
                    "top1_match_at_10_vs_exact": 0.9,
                    "index_build_seconds": 2.0,
                    "indexed_total_seconds": 0.1,
                }
            }
        },
    )
    write_json(
        "outputs/reports/pipeline_run_timings.json",
        {
            "config_hash": config_hash,
            "generated_at": "2026-06-20T00:00:00+00:00",
            "total_seconds": 12.5,
            "stage_count": 3,
            "stages": [{"name": "embed", "status": "passed", "seconds": 5.0}],
            "slowest_stages": [{"name": "embed", "seconds": 5.0}],
        },
    )

    result = experiment_report.run("configs/proofatlas.yaml")
    text = (tmp_path / result["path"]).read_text(encoding="utf-8")

    assert "ProofAtlas Experiment Report" in text
    assert "Final Artifacts" in text
    assert "homepage/index.html" in text
    assert "ML Task Definition" in text
    assert "theorem-disjoint train/validation/test split" in text
    assert "Proof-State-Level Premise Ranking" in text
    assert "Theorem-Level Premise Ranking" in text
    assert "Domain Breakdown" in text
    assert "Test Proof-State-Level Domains" in text
    assert "Error Analysis" in text
    assert "Failure Profile Summary" in text
    assert "Proof-State Failure Profile" in text
    assert "Theorem Failure Profile" in text
    assert "zero_recall_at_max_k" in text
    assert "miss_top_100" in text
    assert "Proof-state zero-recall domains" in text
    assert "Worst Proof-State Queries" in text
    assert "Worst Theorem Queries" in text
    assert "ps_bad" in text
    assert "Mathlib.Bad" in text
    assert "Data supervision" in text
    assert "Local Lean/mathlib source extraction is out of scope" in text
    assert "Retrieval Bottleneck Profile" in text
    assert "candidate_generation_or_embeddings" in text
    assert "top10_reranking_or_candidate_ordering" in text
    assert "Rapid Convergence Plan" in text
    assert "proof_state_query_and_embedding" in text
    assert "proof_state_missing_from_top100" in text
    assert "symbol_overlap" in text
    assert "Metric Uncertainty" in text
    assert "Proof-State Metric Intervals" in text
    assert "Theorem Metric Intervals" in text
    assert "bounded_normal_approximation_for_aggregate_retrieval_metrics" in text
    assert "Pipeline Timing" in text
    assert "Timing config matches current report config" in text
    assert "Timing config matches current report config: `True`" in text
    assert "Evaluation Substage Timing" in text
    assert "test_theorem_retrieval" in text
    assert "Evaluation internal total seconds" in text
    assert "Proof-state test coverage" in text
    assert "Theorem test coverage" in text
    assert "Throughput timing basis" in text
    assert "Scale estimate reliable" in text
    assert "LeanRank premise supervision ready" in text
    assert "Total embedding rows" in text
    assert "Embedding rows by split" in text
    assert "Embedding matrix bytes" in text
    assert "Embed stage seconds" in text
    assert "Embed stage share of total" in text
    assert "Embedding rows/sec during embed stage" in text
    assert "Primary bottleneck share" in text
    assert "Top-3 timed-stage share" in text
    assert "Saved pipeline evaluate seconds" in text
    assert "Current standalone evaluation seconds" in text
    assert "Timed/current evaluation ratio" in text
    assert "Bottleneck stage" in text
    assert "ANN Index Benchmark" in text
    assert "Recall@1 vs exact" in text
    assert "Top1 match@10" in text
    assert "Indexed total seconds" in text
    assert "Mean index speedup vs exact" in text
    assert "Estimated seconds at requested source rows" in text
    assert "Refresh And Retraining Policy" in text
    assert "Training is not repeated for every report or homepage refresh" in text
    assert "Do not retrain by default" in text
    assert "report_or_homepage_refresh" in text
    assert "ranker_feature_or_label_change" in text
    assert "embedding_model_or_text_change" in text
    assert "Premise ranker artifact exists" in text
    assert "Resource And Parallelism Profile" in text
    assert "Multi-process encoding" in text
    assert "Actual backends" in text
    assert "torch_cuda" in text
    assert "hnswlib parameters" in text
    assert "CPU/IO-Heavy Stages" in text
