from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .io import write_json, write_parquet
from .llm_profiles import append_enrichment_text
from .metrics import ranking_row, summarize_rows
from .pretrained_embeddings import load_embeddings
from .profiles import theorem_profiles
from .vector import dense_topk_ids, sparse_topk_ids


def _theorem_positive_premises(split: str) -> dict[str, set[str]]:
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    proof_states["id"] = proof_states["id"].astype(str)
    proof_states["theorem_id"] = proof_states["theorem_id"].astype(str)
    pos["proof_state_id"] = pos["proof_state_id"].astype(str)
    pos["premise_id"] = pos["premise_id"].astype(str)
    ps_to_theorem = proof_states.set_index("id")["theorem_id"].to_dict()
    pos = pos.assign(theorem_id=pos["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    return {theorem_id: set(group["premise_id"]) for theorem_id, group in pos.groupby("theorem_id", sort=False)}


def _theorem_strategy_labels(split: str) -> dict[str, set[str]]:
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    tech = pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet")
    proof_states["id"] = proof_states["id"].astype(str)
    proof_states["theorem_id"] = proof_states["theorem_id"].astype(str)
    tech["proof_state_id"] = tech["proof_state_id"].astype(str)
    ps_to_theorem = proof_states.set_index("id")["theorem_id"].to_dict()
    tech = tech.assign(theorem_id=tech["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    return {theorem_id: set(group["label"].astype(str)) for theorem_id, group in tech.groupby("theorem_id", sort=False)}


def _difficulty_by_theorem(split: str) -> dict[str, dict[str, Any]]:
    path = Path(f"data/processed/{split}/theorem_features.parquet")
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {
        str(row["theorem_id"]): {
            "score": float(row.get("theorem_complexity_score", 0.0) or 0.0),
            "bucket": str(row.get("difficulty_bucket", "")),
        }
        for row in df.to_dict(orient="records")
    }


def _aggregate_neighbor_premises(neighbors: list[str], theorem_premises: dict[str, set[str]], k: int = 100) -> list[str]:
    scores: dict[str, float] = {}
    for rank, theorem_id in enumerate(neighbors, start=1):
        weight = 1.0 / rank
        for premise_id in theorem_premises.get(theorem_id, set()):
            scores[premise_id] = scores.get(premise_id, 0.0) + weight
    return [premise_id for premise_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


def _strategy_row(neighbor_labels: set[str], gold_labels: set[str]) -> dict[str, Any]:
    overlap = neighbor_labels & gold_labels
    recall = len(overlap) / len(gold_labels) if gold_labels else 0.0
    precision = len(overlap) / len(neighbor_labels) if neighbor_labels else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0.0 else 0.0
    return {
        "strategy_gold_count": len(gold_labels),
        "strategy_retrieved_count": len(neighbor_labels),
        "strategy_overlap_count": len(overlap),
        "strategy_recall": recall,
        "strategy_precision": precision,
        "strategy_f1": f1,
        "strategy_any_hit": bool(overlap) if gold_labels else False,
    }


def _difficulty_row(neighbors: list[str], train_difficulty: dict[str, dict[str, Any]], gold: dict[str, Any]) -> dict[str, Any]:
    scores = [train_difficulty[theorem_id]["score"] for theorem_id in neighbors if theorem_id in train_difficulty]
    buckets = [train_difficulty[theorem_id]["bucket"] for theorem_id in neighbors if theorem_id in train_difficulty]
    prediction = float(sum(scores) / len(scores)) if scores else 0.0
    bucket = max(set(buckets), key=buckets.count) if buckets else ""
    target = float(gold.get("score", 0.0) or 0.0)
    return {
        "difficulty_prediction": prediction,
        "difficulty_target": target,
        "difficulty_absolute_error": abs(prediction - target),
        "difficulty_bucket_prediction": bucket,
        "difficulty_bucket_target": gold.get("bucket", ""),
        "difficulty_bucket_match": bool(bucket and bucket == gold.get("bucket", "")),
    }


def run(
    split: str = "test",
    *,
    neighbor_k: int = 20,
    output_dir: str = "outputs/proofatlas",
    use_llm_enrichment: bool = False,
    use_pretrained_embeddings: bool = False,
    pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_theorems = pd.read_parquet("data/processed/train/theorems.parquet")
    train_proof_states = pd.read_parquet("data/processed/train/proof_states.parquet")
    query_theorems = pd.read_parquet(f"data/processed/{split}/theorems.parquet")
    query_proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    train_profiles = theorem_profiles(train_theorems, train_proof_states, max_states=5)
    query_profiles = theorem_profiles(query_theorems, query_proof_states, max_states=5)
    method_parts = []
    if use_llm_enrichment:
        method_parts.append("llm_enriched")
    if use_pretrained_embeddings:
        method_parts.append("pretrained")
    method = "_".join(method_parts) if method_parts else "original"
    if use_llm_enrichment:
        train_profiles = append_enrichment_text(train_profiles, "train", output_dir)
        query_profiles = append_enrichment_text(query_profiles, split, output_dir)
    train_theorem_ids = train_profiles["theorem_id"].astype(str).tolist()
    query_theorem_ids = query_profiles["theorem_id"].astype(str).tolist()
    if use_pretrained_embeddings:
        train_embedding_ids, train_matrix = load_embeddings(
            split="train",
            entity_type="theorem_profile",
            model_name=pretrained_model,
            output_dir=f"{output_dir}/pretrained_embeddings",
        )
        query_embedding_ids, query_matrix = load_embeddings(
            split=split,
            entity_type="theorem_profile",
            model_name=pretrained_model,
            output_dir=f"{output_dir}/pretrained_embeddings",
        )
        query_embedding_row = {theorem_id: idx for idx, theorem_id in enumerate(query_embedding_ids)}
        query_matrix = query_matrix[[query_embedding_row[theorem_id] for theorem_id in query_theorem_ids]]
        neighbors_by_query = dict(
            zip(
                query_theorem_ids,
                dense_topk_ids(query_matrix, train_matrix, train_embedding_ids, neighbor_k, batch_size=128),
                strict=True,
            )
        )
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w'.:]+|\S",
            min_df=1,
            max_features=300000,
            sublinear_tf=True,
            norm="l2",
        )
        train_matrix = vectorizer.fit_transform(train_profiles["profile_text"].fillna("").tolist())
        query_matrix = vectorizer.transform(query_profiles["profile_text"].fillna("").tolist())
        neighbors_by_query = dict(
            zip(
                query_theorem_ids,
                sparse_topk_ids(query_matrix, train_matrix, train_theorem_ids, neighbor_k, batch_size=256),
                strict=True,
            )
        )
    train_premises = set(pd.read_parquet("data/processed/train/premises.parquet")["id"].astype(str))
    train_theorem_premises = _theorem_positive_premises("train")
    query_theorem_premises = _theorem_positive_premises(split)
    train_strategy = _theorem_strategy_labels("train")
    query_strategy = _theorem_strategy_labels(split)
    train_difficulty = _difficulty_by_theorem("train")
    query_difficulty = _difficulty_by_theorem(split)
    profile_info = query_profiles.set_index("theorem_id", drop=False).to_dict(orient="index")
    rows = []
    for theorem_id in query_theorem_ids:
        neighbors = neighbors_by_query.get(theorem_id, [])
        premise_ranked = _aggregate_neighbor_premises(neighbors, train_theorem_premises, k=100)
        labels = set().union(*(train_strategy.get(neighbor_id, set()) for neighbor_id in neighbors))
        row = {
            "split": split,
            "theorem_id": theorem_id,
            "full_name": profile_info.get(theorem_id, {}).get("full_name", ""),
            "domain_tag": profile_info.get(theorem_id, {}).get("domain_tag", ""),
            "retrieval_method": method,
            "neighbors": neighbors,
            **ranking_row(premise_ranked, query_theorem_premises.get(theorem_id, set()), train_premises, [10, 50, 100]),
            **_strategy_row(labels, query_strategy.get(theorem_id, set())),
            **_difficulty_row(neighbors, train_difficulty, query_difficulty.get(theorem_id, {})),
        }
        rows.append(row)
    premise_metrics = summarize_rows(rows, [10, 50, 100])
    strategy_rows = [row for row in rows if int(row.get("strategy_gold_count", 0) or 0) > 0]
    difficulty_rows = [row for row in rows if row.get("difficulty_bucket_target")]
    report = {
        "task": "T2_theorem_to_theorem_pattern_retrieval",
        "split": split,
        "method": method,
        "neighbor_k": int(neighbor_k),
        "query_count": len(query_theorem_ids),
        "premise_coverage": premise_metrics,
        "strategy_facets": {
            "evaluated_theorems": len(strategy_rows),
            "Recall": float(sum(row["strategy_recall"] for row in strategy_rows) / len(strategy_rows)) if strategy_rows else 0.0,
            "Precision": float(sum(row["strategy_precision"] for row in strategy_rows) / len(strategy_rows)) if strategy_rows else 0.0,
            "F1": float(sum(row["strategy_f1"] for row in strategy_rows) / len(strategy_rows)) if strategy_rows else 0.0,
            "AnyHit": float(sum(1 for row in strategy_rows if row["strategy_any_hit"]) / len(strategy_rows)) if strategy_rows else 0.0,
        },
        "difficulty_profile": {
            "evaluated_theorems": len(difficulty_rows),
            "MAE": float(sum(row["difficulty_absolute_error"] for row in difficulty_rows) / len(difficulty_rows)) if difficulty_rows else 0.0,
            "bucket_accuracy": float(sum(1 for row in difficulty_rows if row["difficulty_bucket_match"]) / len(difficulty_rows)) if difficulty_rows else 0.0,
        },
    }
    suffix = split
    if use_llm_enrichment:
        suffix += "_llm_enriched"
    if use_pretrained_embeddings:
        suffix += "_pretrained"
    write_parquet(pd.DataFrame(rows), f"{output_dir}/t2_{suffix}_theorem_neighbors.parquet")
    write_json(f"{output_dir}/t2_{suffix}_theorem_theorem_retrieval.json", report)
    return report
