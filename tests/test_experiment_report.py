from leanrank_kg import experiment_report
from leanrank_kg.utils import write_json


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
                    "worst_cases": [{"proof_state_id": "ps_bad", "rank_of_first_gold": 27, "Recall@10": 0.0, "MRR": 0.0, "gold_premises_total": 1, "gold_premises_in_train_index": 1}],
                },
                "theorem_retrieval": {
                    "metrics": {"theorem_retrieval_evaluated_theorems": 2, "theorem_retrieval_Recall@10": 0.25},
                    "domain_breakdown": [{"domain_tag": "Algebra", "metrics": {"theorem_retrieval_evaluated_queries": 1, "theorem_retrieval_Recall@10": 0.25}}],
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
                "embedding_rows_by_entity": {"proof_state": 10, "premise": 20, "theorem": 12},
                "processed_rows_per_second": 1000.0,
                "pipeline_seconds_per_100k_processed_rows": 100.0,
                "slowest_stage": "evaluate",
                "mean_index_speedup_vs_exact": 12.5,
                "min_index_recall_vs_exact": 0.98,
                "estimated_seconds_at_requested_source_rows": 350.0,
            },
            "recommendations": [],
        },
    )
    write_json("outputs/reports/pipeline_run_timings.json", {"total_seconds": 12.5, "stage_count": 3, "slowest_stages": [{"name": "embed", "seconds": 5.0}]})

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
    assert "Worst Proof-State Queries" in text
    assert "Worst Theorem Queries" in text
    assert "ps_bad" in text
    assert "Mathlib.Bad" in text
    assert "Data supervision" in text
    assert "Local Lean/mathlib source extraction is out of scope" in text
    assert "Pipeline Timing" in text
    assert "LeanRank premise supervision ready" in text
    assert "Total embedding rows" in text
    assert "Mean index speedup vs exact" in text
    assert "Estimated seconds at requested source rows" in text
