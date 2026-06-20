from leanrank_kg.download_or_sample import adapt_rows, synthetic_rows
from leanrank_kg.normalize import normalize_split


def test_normalize_endpoints_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/reports").mkdir(parents=True)
    df = adapt_rows(synthetic_rows(20, 2))
    frames = normalize_split("train", df)
    node_ids = set(frames["proof_states"]["id"]) | set(frames["premises"]["id"])
    for edge_name in ["positive_edges", "negative_edges"]:
        for row in frames[edge_name].to_dict(orient="records"):
            assert row["proof_state_id"] in node_ids
            assert row["premise_id"] in node_ids
