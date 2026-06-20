import pandas as pd

from leanrank_kg import build_graph, download_or_sample, normalize
from leanrank_kg import weak_label_proof_technique
from leanrank_kg.validate import validate_processed_schemas, validate_split_leakage


def test_validation_reports_are_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "schemas").mkdir()
    repo_schemas = __import__("pathlib").Path(__file__).resolve().parents[1] / "schemas"
    for schema in repo_schemas.glob("*.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 4\nuse_huggingface: false\nsample: {total_rows: 40, small_debug_rows: 40, committed_demo_rows: 40}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 40)
    normalize.run(cfg)
    leakage = validate_split_leakage()
    schemas = validate_processed_schemas(["train"])
    assert leakage["has_leakage"] is False
    assert schemas["error_count"] == 0
    assert schemas["tables"]["train/theorems"]["rows"] > 0


def test_base_graph_includes_co_occurrence_edges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "schemas").mkdir()
    repo_schemas = __import__("pathlib").Path(__file__).resolve().parents[1] / "schemas"
    for schema in repo_schemas.glob("*.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 5\nuse_huggingface: false\nsample: {total_rows: 50, small_debug_rows: 50, committed_demo_rows: 50}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 50)
    normalize.run(cfg)
    build_graph.run(cfg)
    edges = pd.read_parquet("outputs/graph/train/edges.parquet")
    assert "co_occurs_with" in set(edges["edge_type"])


def test_proof_technique_entity_records_are_validated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "schemas").mkdir()
    repo_schemas = __import__("pathlib").Path(__file__).resolve().parents[1] / "schemas"
    for schema in repo_schemas.glob("*.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 6\nuse_huggingface: false\nsample: {total_rows: 60, small_debug_rows: 60, committed_demo_rows: 60}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 60)
    normalize.run(cfg)
    weak_label_proof_technique.run(cfg)
    techniques = pd.read_parquet("data/processed/train/proof_techniques.parquet")
    assert not techniques.empty
    schemas = validate_processed_schemas(["train"])
    assert schemas["error_count"] == 0
    assert schemas["tables"]["train/proof_techniques"]["rows"] == len(techniques)
