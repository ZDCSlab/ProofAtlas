import json

from leanrank_kg.pipeline_timing import PipelineTimer
from leanrank_kg.utils import stable_hash


def test_pipeline_timer_records_config_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs/proofatlas.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: erbacher/LeanRank-data",
                "sample: {total_theorems: 10000}",
            ]
        ),
        encoding="utf-8",
    )
    expected_config = {
        "dataset_name": "erbacher/LeanRank-data",
        "sample": {"total_theorems": 10000},
    }

    timer = PipelineTimer("configs/proofatlas.yaml")
    timer.skip_stage("sample", reason="test")
    report = timer.write()

    assert report["config_path"] == "configs/proofatlas.yaml"
    assert report["config_hash"] == stable_hash(json.dumps(expected_config, sort_keys=True), 16)
    assert report["generated_at"]
    assert report["passed"] is True
    assert report["executed_stage_count"] == 0
    assert report["skipped_stage_count"] == 1
    assert report["has_skipped_stages"] is True
    assert (tmp_path / "outputs/reports/pipeline_run_timings.json").exists()
