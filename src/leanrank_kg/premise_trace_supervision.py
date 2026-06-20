from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import SPLITS, read_json, write_json


def _quantiles(series: pd.Series) -> dict[str, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.quantile(0.5)),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _edge_premise_names(edges: pd.DataFrame, premises: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.assign(premise_full_name=pd.Series(dtype=str))
    premise_names = premises[["id", "full_name", "domain_tag", "subdomain_tag"]].rename(
        columns={
            "id": "premise_id",
            "full_name": "premise_full_name",
            "domain_tag": "premise_domain_tag",
            "subdomain_tag": "premise_subdomain_tag",
        }
    )
    return edges.merge(premise_names, on="premise_id", how="left")


def _list_like_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and pd.isna(value):
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _proof_state_trace_examples(
    proof_states: pd.DataFrame,
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    premises: pd.DataFrame,
    features: pd.DataFrame,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if proof_states.empty or positives.empty or negatives.empty:
        return []
    positive_names = _edge_premise_names(positives, premises)
    negative_names = _edge_premise_names(negatives, premises)
    positive_counts = positives.groupby("proof_state_id").size().rename("positive_count")
    negative_counts = negatives.groupby("proof_state_id").size().rename("negative_count")
    hardness = (
        features.set_index("id")["negative_candidate_hardness"].rename("negative_candidate_hardness")
        if not features.empty and {"id", "negative_candidate_hardness"} <= set(features.columns)
        else pd.Series(dtype=float, name="negative_candidate_hardness")
    )
    scored = (
        proof_states.set_index("id")
        .join(positive_counts)
        .join(negative_counts)
        .join(hardness)
        .fillna({"positive_count": 0, "negative_count": 0, "negative_candidate_hardness": 0.0})
    )
    scored = scored[(scored["positive_count"] > 0) & (scored["negative_count"] > 0)]
    if scored.empty:
        return []
    scored = scored.sort_values(["negative_candidate_hardness", "negative_count", "positive_count"], ascending=False).head(limit)
    examples = []
    for proof_state_id, row in scored.iterrows():
        pos_rows = positive_names[positive_names["proof_state_id"] == proof_state_id].head(5)
        neg_rows = negative_names[negative_names["proof_state_id"] == proof_state_id].head(5)
        examples.append(
            {
                "proof_state_id": str(proof_state_id),
                "theorem_id": str(row.get("theorem_id", "")),
                "full_name": str(row.get("full_name", "")),
                "tactic_idx": int(row.get("tactic_idx", 0)),
                "goal_text": str(row.get("goal_text", "")),
                "local_hypothesis_count": _list_like_len(row.get("local_hypotheses")),
                "symbol_count": _list_like_len(row.get("symbols")),
                "positive_count": int(row.get("positive_count", 0)),
                "negative_count": int(row.get("negative_count", 0)),
                "negative_candidate_hardness": float(row.get("negative_candidate_hardness", 0.0)),
                "positive_premises": [
                    {
                        "premise_id": str(premise.get("premise_id", "")),
                        "full_name": str(premise.get("premise_full_name", "")),
                        "source": str(premise.get("source", "")),
                    }
                    for premise in pos_rows.to_dict(orient="records")
                ],
                "hard_negative_candidates": [
                    {
                        "premise_id": str(premise.get("premise_id", "")),
                        "full_name": str(premise.get("premise_full_name", "")),
                        "source": str(premise.get("source", "")),
                    }
                    for premise in neg_rows.to_dict(orient="records")
                ],
            }
        )
    return examples


def _hard_negative_quality_profile(features: pd.DataFrame, negative_counts: pd.Series) -> dict[str, Any]:
    if features.empty or "negative_candidate_hardness" not in features:
        return {
            "bucket_method": "negative_candidate_hardness",
            "bucket_counts": [],
            "high_hardness_threshold": 0.75,
            "high_hardness_proof_state_count": 0,
            "high_hardness_negative_candidate_rows": 0,
            "high_hardness_negative_candidate_share": 0.0,
        }
    rows = features[["id", "negative_candidate_hardness"]].copy()
    rows["negative_candidate_hardness"] = rows["negative_candidate_hardness"].fillna(0.0).astype(float).clip(0.0, 1.0)
    rows = rows.join(negative_counts.rename("negative_candidate_rows"), on="id").fillna({"negative_candidate_rows": 0})

    def bucket(value: float) -> str:
        if value >= 0.75:
            return "high"
        if value >= 0.5:
            return "medium"
        if value > 0.0:
            return "low"
        return "none"

    rows["hardness_bucket"] = rows["negative_candidate_hardness"].map(bucket)
    total_negative_rows = int(rows["negative_candidate_rows"].sum())
    bucket_counts = []
    for label in ["none", "low", "medium", "high"]:
        part = rows[rows["hardness_bucket"] == label]
        negative_rows = int(part["negative_candidate_rows"].sum())
        bucket_counts.append(
            {
                "bucket": label,
                "proof_state_count": int(len(part)),
                "negative_candidate_rows": negative_rows,
                "negative_candidate_row_share": float(negative_rows / max(1, total_negative_rows)),
                "mean_hardness": float(part["negative_candidate_hardness"].mean()) if not part.empty else 0.0,
            }
        )
    high = rows[rows["hardness_bucket"] == "high"]
    high_negative_rows = int(high["negative_candidate_rows"].sum())
    return {
        "bucket_method": "negative_candidate_hardness",
        "bucket_counts": bucket_counts,
        "high_hardness_threshold": 0.75,
        "high_hardness_proof_state_count": int(len(high)),
        "high_hardness_negative_candidate_rows": high_negative_rows,
        "high_hardness_negative_candidate_share": float(high_negative_rows / max(1, total_negative_rows)),
    }


def _split_report(split: str) -> dict[str, Any]:
    try:
        proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
        positives = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
        negatives = pd.read_parquet(f"data/processed/{split}/negative_edges.parquet")
        premises = pd.read_parquet(f"data/processed/{split}/premises.parquet")
    except FileNotFoundError:
        return {"split": split, "exists": False}
    features_path = Path(f"data/processed/{split}/proof_state_features.parquet")
    features = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    positive_counts = positives.groupby("proof_state_id").size() if not positives.empty else pd.Series(dtype=int)
    negative_counts = negatives.groupby("proof_state_id").size() if not negatives.empty else pd.Series(dtype=int)
    premise_positive_frequency = positives.groupby("premise_id").size() if not positives.empty else pd.Series(dtype=int)
    hardness = features.get("negative_candidate_hardness", pd.Series(dtype=float)) if not features.empty else pd.Series(dtype=float)
    proof_state_ids = set(proof_states["id"])
    premise_ids = set(premises["id"])
    positive_proof_states = set(positives["proof_state_id"]) if not positives.empty else set()
    negative_proof_states = set(negatives["proof_state_id"]) if not negatives.empty else set()
    positive_pairs = set(zip(positives.get("proof_state_id", []), positives.get("premise_id", []), strict=False)) if not positives.empty else set()
    negative_pairs = set(zip(negatives.get("proof_state_id", []), negatives.get("premise_id", []), strict=False)) if not negatives.empty else set()
    positive_missing_proof_states = positive_proof_states - proof_state_ids
    negative_missing_proof_states = negative_proof_states - proof_state_ids
    positive_missing_premises = set(positives["premise_id"]) - premise_ids if not positives.empty else set()
    negative_missing_premises = set(negatives["premise_id"]) - premise_ids if not negatives.empty else set()
    overlapping_pairs = positive_pairs & negative_pairs
    proof_states_with_both = (positive_proof_states & negative_proof_states) & proof_state_ids
    proof_state_count = max(1, len(proof_states))
    positive_edge_count = int(len(positives))
    negative_edge_count = int(len(negatives))
    difficulty_sources = (
        sorted(map(str, features["difficulty_target_source"].dropna().unique().tolist()))
        if not features.empty and "difficulty_target_source" in features
        else []
    )
    return {
        "split": split,
        "exists": True,
        "proof_states": int(len(proof_states)),
        "premises": int(len(premises)),
        "positive_edges": positive_edge_count,
        "negative_edges": negative_edge_count,
        "proof_states_with_positive_edges": int(len(positive_proof_states & proof_state_ids)) if not positives.empty else 0,
        "proof_states_with_negative_edges": int(len(negative_proof_states & proof_state_ids)) if not negatives.empty else 0,
        "proof_states_with_both_positive_and_negative_edges": int(len(proof_states_with_both)),
        "positive_proof_state_coverage": float(len(positive_proof_states & proof_state_ids) / proof_state_count),
        "negative_proof_state_coverage": float(len(negative_proof_states & proof_state_ids) / proof_state_count),
        "both_label_coverage": float(len(proof_states_with_both) / proof_state_count),
        "unique_positive_premises": int(positives["premise_id"].nunique()) if not positives.empty else 0,
        "unique_negative_premises": int(negatives["premise_id"].nunique()) if not negatives.empty else 0,
        "negative_to_positive_edge_ratio": float(negative_edge_count / max(1, positive_edge_count)),
        "unique_negative_to_positive_premise_ratio": float(
            (negatives["premise_id"].nunique() if not negatives.empty else 0) / max(1, positives["premise_id"].nunique() if not positives.empty else 0)
        ),
        "positive_edge_missing_proof_state_count": int(len(positive_missing_proof_states)),
        "negative_edge_missing_proof_state_count": int(len(negative_missing_proof_states)),
        "positive_edge_missing_premise_count": int(len(positive_missing_premises)),
        "negative_edge_missing_premise_count": int(len(negative_missing_premises)),
        "positive_negative_pair_overlap_count": int(len(overlapping_pairs)),
        "quality_checks": {
            "positive_edges_have_valid_endpoints": not positive_missing_proof_states and not positive_missing_premises,
            "negative_edges_have_valid_endpoints": not negative_missing_proof_states and not negative_missing_premises,
            "positive_negative_pairs_disjoint": not overlapping_pairs,
            "all_proof_states_have_positive_edges": len(positive_proof_states & proof_state_ids) == len(proof_states),
            "all_proof_states_have_negative_candidates": len(negative_proof_states & proof_state_ids) == len(proof_states),
        },
        "avg_positive_edges_per_proof_state": float(positive_counts.mean()) if not positive_counts.empty else 0.0,
        "avg_negative_edges_per_proof_state": float(negative_counts.mean()) if not negative_counts.empty else 0.0,
        "max_positive_edges_per_proof_state": int(positive_counts.max()) if not positive_counts.empty else 0,
        "max_negative_edges_per_proof_state": int(negative_counts.max()) if not negative_counts.empty else 0,
        "negative_candidate_hardness": _quantiles(hardness),
        "hard_negative_quality_profile": _hard_negative_quality_profile(features, negative_counts),
        "trace_profile": {
            "proof_state_rows": int(len(proof_states)),
            "positive_trace_rows": positive_edge_count,
            "negative_candidate_rows": negative_edge_count,
            "proof_states_with_complete_positive_negative_trace": int(len(proof_states_with_both)),
            "positive_trace_source": "LeanRank-data pos_premise and all_pos_premises normalized to positive_edges",
            "hard_negative_trace_source": "LeanRank-data neg_premises normalized to negative_edges",
            "negative_candidate_hardness_source": "computed from normalized positive and negative premise namespace/domain overlap",
            "difficulty_target_sources": difficulty_sources,
        },
        "example_traces": _proof_state_trace_examples(proof_states, positives, negatives, premises, features),
        "top_positive_premises": [
            {"premise_id": str(premise_id), "count": int(count)}
            for premise_id, count in premise_positive_frequency.sort_values(ascending=False).head(10).items()
        ],
        "edge_source_values": {
            "positive": sorted(map(str, positives["source"].dropna().unique().tolist())) if "source" in positives else [],
            "negative": sorted(map(str, negatives["source"].dropna().unique().tolist())) if "source" in negatives else [],
        },
    }


def build_report() -> dict[str, Any]:
    manifest = read_json("outputs/reports/corpus_manifest.json", {}) or {}
    normalization_conflicts = read_json("outputs/reports/normalization_label_conflicts.json", {}) or {}
    supervision = manifest.get("data_supervision", {}) if isinstance(manifest, dict) else {}
    splits = {split: _split_report(split) for split in SPLITS + ["demo"]}
    total_positive = sum(row.get("positive_edges", 0) for row in splits.values() if row.get("exists"))
    total_negative = sum(row.get("negative_edges", 0) for row in splits.values() if row.get("exists"))
    has_positive = total_positive > 0
    has_negative = total_negative > 0
    quality_rows = [row.get("quality_checks", {}) for row in splits.values() if row.get("exists")]
    return {
        "dataset_name": manifest.get("dataset_name"),
        "source_kind": manifest.get("source_kind"),
        "data_supervision": supervision,
        "label_semantics": supervision.get("premise_label_semantics", "unknown"),
        "scope": "erbacher/LeanRank-data normalized positive/negative premise supervision",
        "normalization_label_conflicts": normalization_conflicts,
        "current_artifact_supervision": {
            "has_positive_edges": has_positive,
            "has_negative_candidates": has_negative,
            "has_negative_candidate_hardness": any(
                bool(row.get("negative_candidate_hardness", {}).get("max", 0.0) > 0.0)
                for row in splits.values()
                if row.get("exists")
            ),
            "total_positive_edges": int(total_positive),
            "total_negative_edges": int(total_negative),
            "negative_to_positive_edge_ratio": float(total_negative / max(1, total_positive)),
            "quality_checks": {
                "all_positive_edges_have_valid_endpoints": all(row.get("positive_edges_have_valid_endpoints", False) for row in quality_rows),
                "all_negative_edges_have_valid_endpoints": all(row.get("negative_edges_have_valid_endpoints", False) for row in quality_rows),
                "all_positive_negative_pairs_disjoint": all(row.get("positive_negative_pairs_disjoint", False) for row in quality_rows),
            },
        },
        "splits": splits,
    }


def run(output_path: str = "outputs/reports/premise_trace_supervision_report.json") -> dict[str, Any]:
    report = build_report()
    write_json(output_path, report)
    return report
