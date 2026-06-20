import json

from leanrank_kg import pipeline_profile
from leanrank_kg.utils import stable_hash, write_json


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
            "slowest_stages": [{"name": "evaluate", "seconds": 3.0}],
            "stages": [{"name": "evaluate", "status": "passed", "seconds": 3.0}],
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
    assert report["stages"]["timings"]["executed_stage_count"] == 1
    assert report["stages"]["timings"]["skipped_stage_count"] == 0
    assert report["throughput_profile"]["timing_config_matches_current"] is True
    assert report["throughput_profile"]["throughput_basis"] == "executed_pipeline_run"
    assert report["throughput_profile"]["scale_estimate_reliable"] is True
    assert any(row["area"] == "indexing" for row in report["recommendations"])
    assert (tmp_path / "outputs/reports/pipeline_performance_report.json").exists()


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
            }
        },
    )

    report = pipeline_profile.run("configs/proofatlas.yaml")

    assert report["throughput_profile"]["throughput_basis"] == "cached_or_partial_pipeline_run"
    assert report["throughput_profile"]["scale_estimate_reliable"] is False
    assert any(row["area"] == "performance_timing" and row["priority"] == "high" for row in report["recommendations"])
    assert any(row["area"] == "evaluation_scope" and row["priority"] == "medium" for row in report["recommendations"])
