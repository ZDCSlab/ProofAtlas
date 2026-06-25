from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import write_json


def _top_counts(items: list[str], limit: int) -> list[dict[str, Any]]:
    counts: dict[str, float] = {}
    for rank, item in enumerate(items, start=1):
        counts[item] = counts.get(item, 0.0) + 1.0 / rank
    return [
        {"id": item_id, "score": float(score)}
        for item_id, score in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return []


def run(split: str = "test", *, limit: int = 25, output_dir: str = "outputs/proofatlas", use_llm_enrichment: bool = False) -> dict[str, Any]:
    suffix = f"{split}_llm_enriched" if use_llm_enrichment else split
    path = Path(f"{output_dir}/t2_{suffix}_theorem_neighbors.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing T2 neighbor output: {path}")
    rows = pd.read_parquet(path).head(limit)
    train_proof_states = pd.read_parquet("data/processed/train/proof_states.parquet")
    train_pos = pd.read_parquet("data/processed/train/positive_edges.parquet")
    train_premises = pd.read_parquet("data/processed/train/premises.parquet")
    train_tech = pd.read_parquet("data/processed/train/proof_state_techniques.parquet")
    train_diff = pd.read_parquet("data/processed/train/theorem_features.parquet")
    train_proof_states["id"] = train_proof_states["id"].astype(str)
    train_proof_states["theorem_id"] = train_proof_states["theorem_id"].astype(str)
    ps_to_theorem = train_proof_states.set_index("id")["theorem_id"].to_dict()
    premise_name = dict(zip(train_premises["id"].astype(str), train_premises["full_name"].astype(str), strict=False))
    train_pos["proof_state_id"] = train_pos["proof_state_id"].astype(str)
    train_pos["premise_id"] = train_pos["premise_id"].astype(str)
    train_pos = train_pos.assign(theorem_id=train_pos["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    theorem_to_premises = {
        theorem_id: group["premise_id"].astype(str).tolist()
        for theorem_id, group in train_pos.groupby("theorem_id", sort=False)
    }
    train_tech["proof_state_id"] = train_tech["proof_state_id"].astype(str)
    train_tech = train_tech.assign(theorem_id=train_tech["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    theorem_to_strategies = {
        theorem_id: group["label"].astype(str).tolist()
        for theorem_id, group in train_tech.groupby("theorem_id", sort=False)
    }
    difficulty = train_diff.set_index("theorem_id", drop=False).to_dict(orient="index")
    bundles = []
    for row in rows.to_dict(orient="records"):
        neighbors = _as_list(row.get("neighbors", []))
        premise_votes: list[str] = []
        strategy_votes: list[str] = []
        neighbor_difficulties = []
        for neighbor in neighbors:
            premise_votes.extend(theorem_to_premises.get(str(neighbor), []))
            strategy_votes.extend(theorem_to_strategies.get(str(neighbor), []))
            if str(neighbor) in difficulty:
                neighbor_difficulties.append(difficulty[str(neighbor)])
        top_premises = _top_counts(premise_votes, 10)
        for premise in top_premises:
            premise["full_name"] = premise_name.get(premise["id"], premise["id"])
        top_strategies = _top_counts(strategy_votes, 8)
        scores = [float(item.get("theorem_complexity_score", 0.0) or 0.0) for item in neighbor_difficulties]
        buckets = [str(item.get("difficulty_bucket", "")) for item in neighbor_difficulties if item.get("difficulty_bucket")]
        bundles.append(
            {
                "theorem_id": row.get("theorem_id"),
                "full_name": row.get("full_name"),
                "domain_tag": row.get("domain_tag"),
                "neighbors": neighbors[:10],
                "premise_suggestions": top_premises,
                "strategy_facets": top_strategies,
                "difficulty_profile": {
                    "score": float(sum(scores) / len(scores)) if scores else 0.0,
                    "bucket": max(set(buckets), key=buckets.count) if buckets else "",
                    "neighbor_count": len(neighbor_difficulties),
                },
            }
        )
    report = {
        "task": "T3_similar_theorem_guidance_aggregation",
        "split": split,
        "use_llm_enrichment": bool(use_llm_enrichment),
        "bundle_count": len(bundles),
        "bundles": bundles,
    }
    write_json(f"{output_dir}/t3_{suffix}_guidance_bundles.json", report)
    return report
