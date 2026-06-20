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
                "processed_rows_per_second": 1000.0,
                "pipeline_seconds_per_100k_processed_rows": 100.0,
                "slowest_stage": "evaluate",
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
    assert "Primary bottleneck share" in text
    assert "Top-3 timed-stage share" in text
    assert "Bottleneck stage" in text
    assert "ANN Index Benchmark" in text
    assert "Recall@1 vs exact" in text
    assert "Top1 match@10" in text
    assert "Indexed total seconds" in text
    assert "Mean index speedup vs exact" in text
    assert "Estimated seconds at requested source rows" in text
