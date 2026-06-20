import pandas as pd

from leanrank_kg.download_or_sample import split_by_theorem, synthetic_rows
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
    download_or_sample.run("configs/sample.yaml", debug_rows=7)
    assert len(__import__("pandas").read_parquet("data/sample/all_rows.parquet")) == 7
