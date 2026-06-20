from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _namespace(name: str) -> str:
    parts = str(name).split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else str(name)


def _theorem_embedding_ids(split: str) -> list[str]:
    meta_path = Path(f"outputs/embeddings/{split}_embedding_metadata.parquet")
    if not meta_path.exists():
        return []
    meta = pd.read_parquet(meta_path)
    rows = meta[meta["entity_type"] == "Theorem"].sort_values("row_index")
    return rows["entity_id"].tolist()


def theorem_similarity_rows(split: str, top_k: int = 10) -> list[dict[str, Any]]:
    thm = pd.read_parquet(f"data/processed/{split}/theorems.parquet")
    ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    thm_features = pd.read_parquet(f"data/processed/{split}/theorem_features.parquet")
    ps_tech = pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet")
    thm_x = sparse.load_npz(f"outputs/embeddings/{split}_theorem_embeddings.npz")
    embedding_ids = _theorem_embedding_ids(split)
    if not embedding_ids:
        embedding_ids = ps["theorem_id"].drop_duplicates().tolist()
    embedding_row = {theorem_id: idx for idx, theorem_id in enumerate(embedding_ids)}
    neighbor_lookup: dict[str, list[tuple[str, float]]] = {}
    if thm_x.shape[0] > 1 and embedding_ids:
        n_neighbors = min(thm_x.shape[0], max(top_k * 8 + 1, top_k + 1))
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        neighbors.fit(thm_x)
        distances, indices = neighbors.kneighbors(thm_x)
        for src_idx, src in enumerate(embedding_ids):
            rows_for_src = []
            for distance, dst_idx in zip(distances[src_idx], indices[src_idx], strict=True):
                dst = embedding_ids[int(dst_idx)]
                if src == dst:
                    continue
                rows_for_src.append((dst, max(0.0, 1.0 - float(distance))))
            neighbor_lookup[src] = rows_for_src
    ps_to_thm = ps.set_index("id")["theorem_id"].to_dict()
    premise_sets: dict[str, set[str]] = {tid: set() for tid in thm["id"]}
    for row in pos.to_dict(orient="records"):
        tid = ps_to_thm.get(row["proof_state_id"])
        if tid:
            premise_sets.setdefault(tid, set()).add(row["premise_id"])
    ps_tech_joined = ps_tech.merge(ps[["id", "theorem_id"]], left_on="proof_state_id", right_on="id", how="left") if not ps_tech.empty else pd.DataFrame(columns=["theorem_id", "label"])
    technique_sets = {
        tid: set(group["label"].dropna())
        for tid, group in ps_tech_joined.groupby("theorem_id")
    }
    feature_lookup = thm_features.set_index("theorem_id").to_dict(orient="index")
    thm_lookup = thm.set_index("id").to_dict(orient="index")
    rows = []
    ids = thm["id"].tolist()
    for src in ids:
        src_info = thm_lookup[src]
        src_features = feature_lookup.get(src, {})
        scored = []
        candidates = neighbor_lookup.get(src, [])
        if not candidates:
            src_idx = embedding_row.get(src)
            candidates = [(dst, 0.0) for dst in ids if dst != src and (src_idx is None or dst in embedding_row)]
        for dst, tfidf_score in candidates:
            if src == dst:
                continue
            dst_info = thm_lookup[dst]
            dst_features = feature_lookup.get(dst, {})
            shared_premise_score = _jaccard(premise_sets.get(src, set()), premise_sets.get(dst, set()))
            same_domain_score = float(src_info.get("domain_tag") == dst_info.get("domain_tag"))
            same_namespace_score = float(_namespace(src_info.get("full_name", "")) == _namespace(dst_info.get("full_name", "")))
            file_namespace_score = max(same_domain_score, same_namespace_score)
            technique_score = _jaccard(technique_sets.get(src, set()), technique_sets.get(dst, set()))
            src_diff = float(src_features.get("theorem_complexity_score", src_features.get("mean_proof_state_difficulty", 0.0)))
            dst_diff = float(dst_features.get("theorem_complexity_score", dst_features.get("mean_proof_state_difficulty", 0.0)))
            difficulty_score = max(0.0, 1.0 - abs(src_diff - dst_diff))
            score = (
                0.40 * tfidf_score
                + 0.25 * shared_premise_score
                + 0.15 * file_namespace_score
                + 0.10 * technique_score
                + 0.10 * difficulty_score
            )
            scored.append(
                {
                    "source": src,
                    "target": dst,
                    "score": float(score),
                    "tfidf_similarity": float(tfidf_score),
                    "shared_premise_score": float(shared_premise_score),
                    "file_namespace_score": float(file_namespace_score),
                    "proof_technique_overlap": float(technique_score),
                    "difficulty_similarity": float(difficulty_score),
                    "target_full_name": dst_info.get("full_name", ""),
                }
            )
        rows.extend(sorted(scored, key=lambda r: r["score"], reverse=True)[:top_k])
    return rows
