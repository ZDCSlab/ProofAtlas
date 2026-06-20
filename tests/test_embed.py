import importlib.util
import json

import pytest
import pandas as pd
from scipy import sparse

from leanrank_kg import benchmark_index, build_graph, build_index, compute_difficulty, download_or_sample, embed, normalize, weak_label_proof_technique


def test_unknown_embedding_backend_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 1\nuse_huggingface: false\nsample: {total_rows: 10, small_debug_rows: 10, committed_demo_rows: 10}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nembedding: {backend: unknown}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        embed.run("configs/sample.yaml")


def test_theorem_embeddings_average_proof_states_and_positive_premises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 12\nuse_huggingface: false\nsample: {total_rows: 80, small_debug_rows: 80, committed_demo_rows: 80}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 80)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    ps = pd.read_parquet("data/processed/train/proof_states.parquet")
    prem = pd.read_parquet("data/processed/train/premises.parquet")
    pos = pd.read_parquet("data/processed/train/positive_edges.parquet")
    meta = pd.read_parquet("outputs/embeddings/train_embedding_metadata.parquet")
    ps_x = sparse.load_npz("outputs/embeddings/train_proof_state_embeddings.npz")
    prem_x = sparse.load_npz("outputs/embeddings/train_premise_embeddings.npz")
    thm_x = sparse.load_npz("outputs/embeddings/train_theorem_embeddings.npz")
    theorem_id = pos["proof_state_id"].map(ps.set_index("id")["theorem_id"]).dropna().iloc[0]
    theorem_row = int(meta[(meta["entity_type"] == "Theorem") & (meta["entity_id"] == theorem_id)]["row_index"].iloc[0])
    ps_rows = ps.index[ps["theorem_id"] == theorem_id].tolist()
    premise_ids = set(pos[pos["proof_state_id"].isin(set(ps[ps["theorem_id"] == theorem_id]["id"]))]["premise_id"])
    prem_rows = prem.index[prem["id"].isin(premise_ids)].tolist()
    expected = sparse.vstack([ps_x[ps_rows], prem_x[prem_rows]]).mean(axis=0)
    diff = sparse.csr_matrix(thm_x[theorem_row] - expected)
    assert diff.nnz == 0 or abs(diff).max() < 1e-9


def test_build_index_auto_backend_records_resolved_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 18\nuse_huggingface: false\nsample: {total_rows: 50, small_debug_rows: 50, committed_demo_rows: 50}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\nindex: {backend: auto, metric: cosine, n_neighbors: 20}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 50)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    build_index.run(cfg)
    summary = json.loads((tmp_path / "outputs/indexes/index_summary.json").read_text(encoding="utf-8"))
    expected_backend = "hnswlib" if importlib.util.find_spec("hnswlib") else "sklearn"
    assert summary["requested_backend"] == "auto"
    assert summary["backend"] == expected_backend
    manifest = json.loads((tmp_path / "outputs/indexes/train_premise_index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == expected_backend
    assert manifest["index_format"] in {"joblib", "hnswlib"}
    assert manifest["corpus_version"]
    assert manifest["extraction_config_hash"]
    assert summary["corpus"]["corpus_version"] == manifest["corpus_version"]
    proof_state_manifest = json.loads((tmp_path / "outputs/indexes/train_proof_state_index_manifest.json").read_text(encoding="utf-8"))
    assert proof_state_manifest["entity_type"] == "ProofState"
    assert proof_state_manifest["backend"] == expected_backend


def test_faiss_backend_is_optional_and_manifested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 20\nuse_huggingface: false\nsample: {total_rows: 30, small_debug_rows: 30, committed_demo_rows: 30}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\nindex: {backend: faiss, metric: cosine, n_neighbors: 10}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 30)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    if importlib.util.find_spec("faiss") is None:
        with pytest.raises(RuntimeError, match="faiss index backend requires optional dependency"):
            build_index.run(cfg)
        return
    build_index.run(cfg)
    manifest = json.loads((tmp_path / "outputs/indexes/train_premise_index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "faiss"
    assert manifest["index_format"] == "faiss"
    assert manifest["faiss_index_type"] == "IndexFlatIP"
    assert manifest["stored_vector_normalization"] == "l2"


def test_lancedb_backend_is_optional_and_manifested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 21\nuse_huggingface: false\nsample: {total_rows: 30, small_debug_rows: 30, committed_demo_rows: 30}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\nindex: {backend: lancedb, metric: cosine, n_neighbors: 10, create_vector_index: false}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 30)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    if importlib.util.find_spec("lancedb") is None:
        with pytest.raises(RuntimeError, match="lancedb index backend requires optional dependency"):
            build_index.run(cfg)
        return
    build_index.run(cfg)
    manifest = json.loads((tmp_path / "outputs/indexes/train_premise_index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "lancedb"
    assert manifest["index_format"] == "lancedb"
    assert manifest["table_name"]
    assert manifest["vector_column"] == "vector"
    assert manifest["score_kind"] == "cosine_distance"


def test_benchmark_index_reports_latency_and_exact_overlap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sample.yaml").write_text(
        "dataset_name: erbacher/LeanRank-data\nrandom_seed: 19\nuse_huggingface: false\nsample: {total_rows: 60, small_debug_rows: 60, committed_demo_rows: 60}\nsplit: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1}\nretrieval: {top_k: [1, 5, 10]}\nsimilarity: {theorem_top_k: 3}\nproof_techniques: {max_labels_per_state: 5, minimum_support: 1}\nembedding: {backend: tfidf}\nindex: {backend: sklearn, metric: cosine, n_neighbors: 20}\nindex_benchmark: {split: train, top_k: 5, query_count: 6, seed: 19}\n",
        encoding="utf-8",
    )
    cfg = "configs/sample.yaml"
    download_or_sample.run(cfg, 60)
    normalize.run(cfg)
    build_graph.run(cfg)
    weak_label_proof_technique.run(cfg)
    compute_difficulty.run(cfg)
    embed.run(cfg)
    build_index.run(cfg)
    summary = benchmark_index.run(cfg)
    report = json.loads((tmp_path / "outputs/reports/index_benchmark.json").read_text(encoding="utf-8"))
    assert summary == report
    assert set(report["entities"]) == {"proof_state", "premise", "theorem"}
    for entity in report["entities"].values():
        assert entity["manifest_valid"] is True
        assert entity["indexed_available"] is True
        assert entity["index_build_seconds"] >= 0.0
        assert entity["exact_ms_per_query"] >= 0.0
        assert entity["indexed_ms_per_query"] >= 0.0
        assert entity[f"recall_at_{entity['top_k']}_vs_exact"] >= 0.99
