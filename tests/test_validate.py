import pandas as pd
import json

from leanrank_kg import build_graph, download_or_sample, normalize
from leanrank_kg import build_index, compute_difficulty, embed, weak_label_proof_technique
from leanrank_kg.validate import validate_artifact_compatibility, validate_processed_schemas, validate_split_leakage, validate_theorem_query_parse_coverage


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


def test_theorem_query_parse_coverage_report_tracks_structured_features(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "schemas").mkdir()
    repo_schemas = __import__("pathlib").Path(__file__).resolve().parents[1] / "schemas"
    for schema in repo_schemas.glob("*.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 24\nuse_huggingface: false\nsample: {total_rows: 40, small_debug_rows: 40, committed_demo_rows: 40}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 40)
    normalize.run(cfg)
    report = validate_theorem_query_parse_coverage(["train"])
    assert report["train"]["theorems"] > 0
    assert report["train"]["parse_coverage"] > 0
    assert "avg_operator_symbol_count" in report["train"]
    assert (tmp_path / "outputs/reports/theorem_query_parse_coverage.json").exists()


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


def test_base_graph_uses_theorem_level_invokes_and_scoped_tactics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "schemas").mkdir()
    repo_schemas = __import__("pathlib").Path(__file__).resolve().parents[1] / "schemas"
    for schema in repo_schemas.glob("*.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 14\nuse_huggingface: false\nsample: {total_rows: 60, small_debug_rows: 60, committed_demo_rows: 60}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 60)
    normalize.run(cfg)
    build_graph.run(cfg)
    proof_state_ids = set(pd.read_parquet("data/processed/train/proof_states.parquet")["id"])
    theorem_ids = set(pd.read_parquet("data/processed/train/theorems.parquet")["id"])
    edges = pd.read_parquet("outputs/graph/train/edges.parquet")
    invokes = edges[edges["edge_type"] == "invokes_premise"]
    tactic_edges = edges[edges["edge_type"] == "at_tactic_step"]
    assert not invokes.empty
    assert set(invokes["source"]) <= theorem_ids
    assert not (set(invokes["source"]) & proof_state_ids)
    assert tactic_edges["target"].str.startswith("tactic:thm:").all()


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


def test_artifact_compatibility_report_checks_config_embeddings_and_indexes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "schemas").mkdir()
    repo_schemas = __import__("pathlib").Path(__file__).resolve().parents[1] / "schemas"
    for schema in repo_schemas.glob("*.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 21\nuse_huggingface: false\nsample: {total_rows: 80, small_debug_rows: 80, committed_demo_rows: 80}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\nindex: {backend: sklearn, metric: cosine, n_neighbors: 20}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 80)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    build_index.run(cfg)
    report = validate_artifact_compatibility(cfg)
    assert report["passed"] is True
    assert report["config_hash_matches"] is True
    assert report["data_supervision"]["kind"] == "synthetic_demo_rows"
    assert "corpus_is_not_real_mathlib_source" in report["warnings"]
    assert report["splits"]["train"]["embeddings"]["matrices"]["premise"]["matches_metadata"] is True
    assert report["splits"]["train"]["indexes"]["premise"]["valid"] is True

    index_manifest_path = tmp_path / "outputs/indexes/train_premise_index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["corpus_version"] = "stale-corpus"
    index_manifest_path.write_text(json.dumps(index_manifest), encoding="utf-8")
    stale_index = validate_artifact_compatibility(cfg)
    assert stale_index["passed"] is False
    assert stale_index["splits"]["train"]["indexes"]["premise"]["valid"] is False
    index_manifest["corpus_version"] = report["splits"]["train"]["indexes"]["premise"]["expected_corpus_version"]
    index_manifest_path.write_text(json.dumps(index_manifest), encoding="utf-8")

    manifest_path = tmp_path / "outputs/reports/corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_hash"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stale = validate_artifact_compatibility(cfg)
    assert stale["passed"] is False
    assert stale["config_hash_matches"] is False
