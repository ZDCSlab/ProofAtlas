from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .theorem_similarity import theorem_similarity_rows
from .utils import read_json


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_split(split: str = "train") -> dict[str, Any]:
    return {
        "proof_states": pd.read_parquet(f"data/processed/{split}/proof_states.parquet"),
        "premises": pd.read_parquet(f"data/processed/{split}/premises.parquet"),
        "theorems": pd.read_parquet(f"data/processed/{split}/theorems.parquet"),
        "ps_features": pd.read_parquet(f"data/processed/{split}/proof_state_features.parquet"),
        "thm_features": pd.read_parquet(f"data/processed/{split}/theorem_features.parquet"),
        "ps_tech": pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet"),
        "prem_tech": pd.read_parquet(f"data/processed/{split}/premise_techniques.parquet"),
        "edges": pd.read_parquet(f"outputs/graph/{split}/edges_enriched.parquet") if Path(f"outputs/graph/{split}/edges_enriched.parquet").exists() else pd.read_parquet(f"outputs/graph/{split}/edges.parquet"),
        "ps_x": sparse.load_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz"),
        "prem_x": sparse.load_npz(f"outputs/embeddings/{split}_premise_embeddings.npz"),
        "thm_x": sparse.load_npz(f"outputs/embeddings/{split}_theorem_embeddings.npz"),
    }


def _embedding_description() -> str:
    config = read_json("outputs/embeddings/embedding_config.json", {}) or {}
    backend = config.get("backend", "embedding")
    model = config.get("model_name")
    device = config.get("device")
    if model and device:
        return f"{model} embeddings on {device}"
    if model:
        return f"{model} embeddings"
    if backend == "tfidf":
        return "TF-IDF embeddings"
    return f"{backend} embeddings"


def retrieve_premises(proof_state_id: str, k: int = 10, split: str = "train", index_split: str = "train") -> list[dict]:
    query_data = _load_split(split)
    index_data = _load_split(index_split)
    ps = query_data["proof_states"]
    idxs = ps.index[ps["id"] == proof_state_id].tolist()
    if not idxs:
        return []
    scores = cosine_similarity(query_data["ps_x"][idxs[0]], index_data["prem_x"]).ravel()
    prem = index_data["premises"].copy()
    prem["score"] = scores
    out = prem.sort_values("score", ascending=False).head(k)
    return [
        {
            "premise_id": r["id"],
            "full_name": r["full_name"],
            "score": float(r["score"]),
            "index_split": index_split,
            "explanation": f"Cosine similarity over {_embedding_description()} for proof-state queries and train-index premise text",
        }
        for r in out.to_dict(orient="records")
    ]


def retrieve_similar_theorems(theorem_id: str, k: int = 10, split: str = "train") -> list[dict]:
    rows = [row for row in theorem_similarity_rows(split, k) if row["source"] == theorem_id]
    return [
        {
            "theorem_id": row["target"],
            "full_name": row["target_full_name"],
            "score": row["score"],
            "explanation": f"Weighted non-GNN similarity: {_embedding_description()}, shared premises, namespace/file area, proof techniques, and difficulty",
            "signals": {
                "tfidf_similarity": row["tfidf_similarity"],
                "shared_premise_score": row["shared_premise_score"],
                "file_namespace_score": row["file_namespace_score"],
                "proof_technique_overlap": row["proof_technique_overlap"],
                "difficulty_similarity": row["difficulty_similarity"],
            },
        }
        for row in rows[:k]
    ]


def explain_premise_match(proof_state_id: str, premise_id: str, split: str = "train", index_split: str = "train") -> dict:
    results = retrieve_premises(proof_state_id, k=1000, split=split, index_split=index_split)
    score = next((r["score"] for r in results if r["premise_id"] == premise_id), 0.0)
    query_data = _load_split(split)
    index_data = _load_split(index_split)
    ps = query_data["proof_states"].set_index("id").loc[proof_state_id]
    index_premises = index_data["premises"].set_index("id")
    if premise_id in index_premises.index:
        prem = index_premises.loc[premise_id]
        premise_in_index = True
        prem_tech = index_data["prem_tech"]
    else:
        query_premises = query_data["premises"].set_index("id")
        prem = query_premises.loc[premise_id] if premise_id in query_premises.index else pd.Series({"full_name": "", "domain_tag": "", "file_path": ""})
        premise_in_index = False
        prem_tech = query_data["prem_tech"]
    ps_labels = set(query_data["ps_tech"].query("proof_state_id == @proof_state_id")["label"]) if not query_data["ps_tech"].empty else set()
    prem_labels = set(prem_tech.query("premise_id == @premise_id")["label"]) if not prem_tech.empty else set()
    return {
        "proof_state_id": proof_state_id,
        "premise_id": premise_id,
        "cosine_score": float(score),
        "index_split": index_split,
        "premise_in_index": premise_in_index,
        "namespace_match": ps["full_name"].split(".")[0] == prem["full_name"].split(".")[0],
        "file_match": ps.get("file_path", "") == prem.get("file_path", ""),
        "domain_match": ps.get("domain_tag", "") == prem.get("domain_tag", ""),
        "shared_proof_techniques": sorted(ps_labels & prem_labels),
    }


def get_proof_technique_labels(proof_state_id: str, split: str = "train") -> list[dict]:
    tech = _load_split(split)["ps_tech"]
    if tech.empty:
        return []
    return tech[tech["proof_state_id"] == proof_state_id][["label", "provenance"]].to_dict(orient="records")


def get_difficulty_profile(entity_id: str, split: str = "train") -> dict:
    data = _load_split(split)
    for frame_name in ["ps_features", "thm_features"]:
        frame = data[frame_name]
        key = "id" if "id" in frame.columns else "theorem_id"
        hit = frame[frame[key] == entity_id]
        if not hit.empty:
            return _jsonable(hit.iloc[0].to_dict())
    return {}


def get_graph_neighborhood(entity_id: str, depth: int = 1, split: str = "train") -> dict:
    edges = _load_split(split)["edges"]
    frontier = {entity_id}
    seen = {entity_id}
    selected = []
    for _ in range(depth):
        mask = edges["source"].isin(frontier) | edges["target"].isin(frontier)
        batch = edges[mask]
        selected.extend(_jsonable(batch.to_dict(orient="records")))
        frontier = set(batch["source"]) | set(batch["target"])
        frontier -= seen
        seen |= frontier
    return {"entity_id": entity_id, "nodes": sorted(seen), "edges": selected}
