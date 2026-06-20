import pytest
import pandas as pd
from scipy import sparse

from leanrank_kg import build_graph, compute_difficulty, download_or_sample, embed, normalize, weak_label_proof_technique


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
