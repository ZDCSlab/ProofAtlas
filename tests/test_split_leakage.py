import pandas as pd

from leanrank_kg.download_or_sample import _resolve_corpus_provenance, _sample_by_theorem, split_by_theorem, synthetic_rows
from leanrank_kg import download_or_sample


def test_split_by_theorem_has_no_leakage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/sample").mkdir(parents=True)
    df = synthetic_rows(80, 1)
    splits = split_by_theorem(df, 1, {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15})
    sets = [set(frame["full_name"]) for frame in splits.values()]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_sampler_uses_total_rows_unless_debug_rows_is_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 1\nuse_huggingface: false\nsample: {total_rows: 37, small_debug_rows: 5, committed_demo_rows: 10}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\n",
        encoding="utf-8",
    )
    download_or_sample.run("configs/sample.yaml")
    assert len(__import__("pandas").read_parquet("data/sample/all_rows.parquet")) == 37
    manifest = __import__("json").loads((tmp_path / "outputs/reports/corpus_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "erbacher/LeanRank-data"
    assert manifest["source_kind"] == "synthetic"
    assert manifest["data_supervision"]["kind"] == "synthetic_demo_rows"
    assert manifest["data_supervision"]["suitable_for"]["premise_ranking_training"] is False
    assert manifest["config_hash"]
    assert manifest["corpus"]["extraction_config_hash"]
    assert set(manifest["split_counts"]) == {"train", "val", "test"}
    download_or_sample.run("configs/sample.yaml", debug_rows=7)
    assert len(__import__("pandas").read_parquet("data/sample/all_rows.parquet")) == 7


def test_corpus_provenance_uses_env_for_unknown_values(monkeypatch):
    monkeypatch.setenv("PROOFATLAS_LEAN_VERSION", "Lean 4.99.0")
    monkeypatch.setenv("PROOFATLAS_MATHLIB_COMMIT", "abc123")
    monkeypatch.setenv("PROOFATLAS_SOURCE_REVISION", "dataset-rev")
    monkeypatch.setenv("PROOFATLAS_CORPUS_VERSION", "leanrank-test-v1")
    provenance = _resolve_corpus_provenance(
        {
            "dataset_name": "erbacher/LeanRank-data",
            "use_huggingface": False,
            "corpus": {
                "lean_version": "unknown",
                "mathlib_commit": "unknown",
                "source_revision": "unknown",
                "extraction_pipeline": "LeanRank synthetic-compatible sample",
            },
            "sample": {"total_rows": 10},
            "split": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        },
        "configs/sample.yaml",
    )
    assert provenance["lean_version"] == "Lean 4.99.0"
    assert provenance["mathlib_commit"] == "abc123"
    assert provenance["source_revision"] == "dataset-rev"
    assert provenance["corpus_version"] == "leanrank-test-v1"
    assert provenance["extraction_config_hash"]


def test_theorem_sampling_keeps_whole_candidate_theorems():
    df = synthetic_rows(80, 3)
    sampled = _sample_by_theorem(df, total_theorems=10, seed=3)
    selected = set(sampled["full_name"])
    assert sampled["full_name"].nunique() == 10
    for name in selected:
        assert len(sampled[sampled["full_name"] == name]) == len(df[df["full_name"] == name])


def test_theorem_mode_sampler_limits_theorem_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 2\nuse_huggingface: false\nsample: {unit: theorem, total_theorems: 9, hf_source_rows: 80, total_rows: 80, small_debug_rows: 5, committed_demo_rows: 10}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\n",
        encoding="utf-8",
    )
    download_or_sample.run("configs/sample.yaml")
    rows = pd.read_parquet("data/sample/all_rows.parquet")
    assert rows["full_name"].nunique() == 9
    assert len(rows) >= 9
