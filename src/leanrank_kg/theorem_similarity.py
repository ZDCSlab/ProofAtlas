from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


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
    tfidf = cosine_similarity(thm_x) if thm_x.shape[0] else []
    tfidf_lookup = {
        (src, dst): float(tfidf[i, j])
        for i, src in enumerate(embedding_ids)
        for j, dst in enumerate(embedding_ids)
        if i != j
    }
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
        for dst in ids:
            if src == dst:
                continue
            dst_info = thm_lookup[dst]
            dst_features = feature_lookup.get(dst, {})
            tfidf_score = tfidf_lookup.get((src, dst), 0.0)
            shared_premise_score = _jaccard(premise_sets.get(src, set()), premise_sets.get(dst, set()))
            same_domain_score = float(src_info.get("domain_tag") == dst_info.get("domain_tag"))
            same_namespace_score = float(_namespace(src_info.get("full_name", "")) == _namespace(dst_info.get("full_name", "")))
            file_namespace_score = max(same_domain_score, same_namespace_score)
            technique_score = _jaccard(technique_sets.get(src, set()), technique_sets.get(dst, set()))
            src_diff = float(src_features.get("mean_proof_state_difficulty", 0.0))
            dst_diff = float(dst_features.get("mean_proof_state_difficulty", 0.0))
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
