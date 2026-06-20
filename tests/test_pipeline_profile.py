from leanrank_kg import pipeline_profile
from leanrank_kg.utils import write_json


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
    report = pipeline_profile.run("configs/proofatlas.yaml")

    assert report["dataset_name"] == "erbacher/LeanRank-data"
    assert report["scale_profile"]["target_dataset_confirmed"] is True
    assert report["scale_profile"]["scale_bucket"] == "large"
    assert any(row["area"] == "indexing" for row in report["recommendations"])
    assert (tmp_path / "outputs/reports/pipeline_performance_report.json").exists()
