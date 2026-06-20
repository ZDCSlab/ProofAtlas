from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from .utils import SPLITS, bucket, minmax, namespace, write_parquet


def _name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, str(left), str(right)).ratio()


def _negative_hardness(pos_rows: pd.DataFrame, neg_rows: pd.DataFrame) -> pd.Series:
    scores = {}
    for proof_state_id, neg_group in neg_rows.groupby("proof_state_id"):
        positives = pos_rows[pos_rows["proof_state_id"] == proof_state_id]
        if positives.empty or neg_group.empty:
            scores[proof_state_id] = 0.0
            continue
        neg_scores = []
        for neg in neg_group.to_dict(orient="records"):
            best = 0.0
            for pos in positives.to_dict(orient="records"):
                namespace_match = float(namespace(neg["full_name"]) == namespace(pos["full_name"]))
                domain_match = float(neg.get("domain_tag", "") == pos.get("domain_tag", ""))
                name_sim = _name_similarity(neg["full_name"], pos["full_name"])
                best = max(best, 0.45 * namespace_match + 0.25 * domain_match + 0.30 * name_sim)
            neg_scores.append(best)
        scores[proof_state_id] = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
    return pd.Series(scores, name="negative_candidate_hardness")


def run(config_path: str) -> None:
    reports = []
    for split in SPLITS + ["demo"]:
        try:
            ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
            pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
            neg = pd.read_parquet(f"data/processed/{split}/negative_edges.parquet")
            prem = pd.read_parquet(f"data/processed/{split}/premises.parquet")
        except FileNotFoundError:
            continue
        premise_info = prem[["id", "full_name", "code", "domain_tag"]].copy()
        pos_info = pos.merge(premise_info, left_on="premise_id", right_on="id", how="left", suffixes=("", "_premise"))
        neg_info = neg.merge(premise_info, left_on="premise_id", right_on="id", how="left", suffixes=("", "_premise"))
        pos_counts = pos.groupby("proof_state_id").size().rename("num_positive_premises_raw")
        avg_pos_len = pos_info.assign(code_len=pos_info["code"].fillna("").str.len()).groupby("proof_state_id")["code_len"].mean().rename("avg_positive_premise_length_raw")
        hardness = _negative_hardness(pos_info, neg_info)
        namespace_counts = pos_info["full_name"].fillna("").map(namespace).value_counts()
        max_count = float(namespace_counts.max()) if not namespace_counts.empty else 1.0
        rarity_by_ns = {ns: 1.0 - (count / max_count) for ns, count in namespace_counts.items()}
        rarity = (
            pos_info.assign(ns=pos_info["full_name"].fillna("").map(namespace))
            .assign(rarity=lambda frame: frame["ns"].map(rarity_by_ns).fillna(0.0))
            .groupby("proof_state_id")["rarity"]
            .mean()
            .rename("premise_namespace_rarity")
        )
        feat = ps[["id", "theorem_id", "context", "local_hypotheses", "tactic_idx"]].copy()
        feat["context_length_score"] = minmax(feat["context"].str.len())
        feat["num_local_hypotheses"] = minmax(feat["local_hypotheses"].map(len))
        feat = feat.join(pos_counts, on="id").join(avg_pos_len, on="id").join(hardness, on="id").join(rarity, on="id").fillna(0)
        feat["num_positive_premises"] = minmax(feat["num_positive_premises_raw"])
        feat["avg_positive_premise_length"] = minmax(feat["avg_positive_premise_length_raw"])
        feat["negative_candidate_hardness"] = feat["negative_candidate_hardness"].clip(0, 1)
        feat["premise_namespace_rarity"] = feat["premise_namespace_rarity"].clip(0, 1)
        feat["tactic_step_index_score"] = minmax(feat["tactic_idx"])
        cols = [
            "context_length_score",
            "num_local_hypotheses",
            "num_positive_premises",
            "avg_positive_premise_length",
            "premise_namespace_rarity",
            "tactic_step_index_score",
            "negative_candidate_hardness",
        ]
        feat["difficulty_score"] = feat[cols].mean(axis=1)
        feat["difficulty_bucket"] = feat["difficulty_score"].map(bucket)
        write_parquet(feat.drop(columns=["context", "local_hypotheses", "num_positive_premises_raw", "avg_positive_premise_length_raw"]), f"data/processed/{split}/proof_state_features.parquet")
        thm = feat.groupby("theorem_id").agg(
            mean_proof_state_difficulty=("difficulty_score", "mean"),
            max_proof_state_difficulty=("difficulty_score", "max"),
            num_proof_states=("id", "count"),
        ).reset_index()
        ps_to_thm = ps.set_index("id")["theorem_id"].to_dict()
        premise_counts = (
            pos.assign(theorem_id=pos["proof_state_id"].map(ps_to_thm))
            .dropna(subset=["theorem_id"])
            .groupby("theorem_id")["premise_id"]
            .nunique()
            .rename("num_unique_positive_premises")
        )
        thm = thm.join(premise_counts, on="theorem_id").fillna({"num_unique_positive_premises": 0})
        thm["num_unique_positive_premises"] = thm["num_unique_positive_premises"].astype(int)
        thm["difficulty_bucket"] = thm["mean_proof_state_difficulty"].map(bucket)
        write_parquet(thm, f"data/processed/{split}/theorem_features.parquet")
        counts = feat["difficulty_bucket"].value_counts().reset_index()
        counts.columns = ["bucket", "count"]
        counts["split"] = split
        reports.append(counts)
    if reports:
        pd.concat(reports, ignore_index=True).to_csv("outputs/reports/difficulty_distribution.csv", index=False)
