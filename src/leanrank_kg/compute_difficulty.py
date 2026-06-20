from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from .utils import SPLITS, bucket, minmax, namespace, write_json, write_parquet


def _name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, str(left), str(right)).ratio()


def _premise_records_by_proof_state(rows: pd.DataFrame) -> dict[str, list[tuple[str, str, str]]]:
    if rows.empty:
        return {}
    prepared = rows[["proof_state_id", "full_name", "domain_tag"]].copy()
    prepared["full_name"] = prepared["full_name"].fillna("").astype(str)
    prepared["domain_tag"] = prepared["domain_tag"].fillna("").astype(str)
    prepared["namespace"] = prepared["full_name"].map(namespace)
    return {
        str(proof_state_id): list(zip(group["full_name"], group["namespace"], group["domain_tag"], strict=True))
        for proof_state_id, group in prepared.groupby("proof_state_id", sort=False)
    }


def _negative_hardness(pos_rows: pd.DataFrame, neg_rows: pd.DataFrame) -> pd.Series:
    positive_by_state = _premise_records_by_proof_state(pos_rows)
    negative_by_state = _premise_records_by_proof_state(neg_rows)
    similarity_cache: dict[tuple[str, str], float] = {}
    scores = {}
    for proof_state_id, negatives in negative_by_state.items():
        positives = positive_by_state.get(proof_state_id, [])
        if not positives or not negatives:
            scores[proof_state_id] = 0.0
            continue
        neg_scores = []
        for neg_name, neg_namespace, neg_domain in negatives:
            best = 0.0
            for pos_name, pos_namespace, pos_domain in positives:
                namespace_match = float(neg_namespace == pos_namespace)
                domain_match = float(neg_domain == pos_domain)
                cache_key = (neg_name, pos_name)
                name_sim = similarity_cache.get(cache_key)
                if name_sim is None:
                    name_sim = _name_similarity(neg_name, pos_name)
                    similarity_cache[cache_key] = name_sim
                best = max(best, 0.45 * namespace_match + 0.25 * domain_match + 0.30 * name_sim)
                if best >= 1.0:
                    break
            neg_scores.append(best)
        scores[proof_state_id] = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
    return pd.Series(scores, name="negative_candidate_hardness")


def _build_theorem_complexity_features(feat: pd.DataFrame, pos: pd.DataFrame, neg: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    thm = feat.groupby("theorem_id").agg(
        mean_proof_state_difficulty=("difficulty_score", "mean"),
        max_proof_state_difficulty=("difficulty_score", "max"),
        mean_negative_candidate_hardness=("negative_candidate_hardness", "mean"),
        max_tactic_idx=("tactic_idx", "max"),
        num_proof_states=("id", "count"),
    ).reset_index()
    ps_to_thm = ps.set_index("id")["theorem_id"].to_dict()
    if pos.empty:
        premise_counts = pd.Series(dtype="int64", name="num_unique_positive_premises")
    else:
        premise_counts = (
            pos.assign(theorem_id=pos["proof_state_id"].map(ps_to_thm))
            .dropna(subset=["theorem_id"])
            .groupby("theorem_id")["premise_id"]
            .nunique()
            .rename("num_unique_positive_premises")
        )
    if neg.empty:
        negative_counts = pd.Series(dtype="int64", name="num_failed_negative_candidates")
    else:
        negative_counts = (
            neg.assign(theorem_id=neg["proof_state_id"].map(ps_to_thm))
            .dropna(subset=["theorem_id"])
            .groupby("theorem_id")
            .size()
            .rename("num_failed_negative_candidates")
        )
    thm = thm.join(premise_counts, on="theorem_id").join(negative_counts, on="theorem_id").fillna(
        {"num_unique_positive_premises": 0, "num_failed_negative_candidates": 0}
    )
    thm["num_unique_positive_premises"] = thm["num_unique_positive_premises"].astype(int)
    thm["num_failed_negative_candidates"] = thm["num_failed_negative_candidates"].astype(int)
    thm["proof_length_score"] = minmax(thm["num_proof_states"])
    thm["tactic_count_score"] = minmax(thm["max_tactic_idx"] + 1)
    thm["premise_count_score"] = minmax(thm["num_unique_positive_premises"])
    thm["negative_candidate_count_score"] = minmax(thm["num_failed_negative_candidates"])
    thm["theorem_complexity_score"] = thm[
        [
            "proof_length_score",
            "tactic_count_score",
            "premise_count_score",
            "negative_candidate_count_score",
            "mean_negative_candidate_hardness",
            "max_proof_state_difficulty",
        ]
    ].mean(axis=1)
    thm["difficulty_bucket"] = thm["theorem_complexity_score"].map(bucket)
    thm["difficulty_target_source"] = "proof_length_tactic_count_premise_count_negative_candidates"
    return thm


def run(config_path: str) -> None:
    del config_path
    reports = []
    complexity_reports = []
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
        thm = _build_theorem_complexity_features(feat, pos, neg, ps)
        feat = feat.join(thm.set_index("theorem_id")[["theorem_complexity_score", "difficulty_target_source"]], on="theorem_id")
        write_parquet(
            feat.drop(columns=["context", "local_hypotheses", "num_positive_premises_raw", "avg_positive_premise_length_raw"]),
            f"data/processed/{split}/proof_state_features.parquet",
        )
        write_parquet(thm, f"data/processed/{split}/theorem_features.parquet")
        counts = feat["difficulty_bucket"].value_counts().reset_index()
        counts.columns = ["bucket", "count"]
        counts["split"] = split
        reports.append(counts)
        complexity_reports.append(
            {
                "split": split,
                "theorems": int(len(thm)),
                "target": "theorem_features.theorem_complexity_score",
                "target_source": "proof_length_tactic_count_premise_count_negative_candidates",
                "mean_theorem_complexity_score": float(thm["theorem_complexity_score"].mean()) if not thm.empty else 0.0,
                "max_theorem_complexity_score": float(thm["theorem_complexity_score"].max()) if not thm.empty else 0.0,
                "mean_num_proof_states": float(thm["num_proof_states"].mean()) if not thm.empty else 0.0,
                "mean_num_unique_positive_premises": float(thm["num_unique_positive_premises"].mean()) if not thm.empty else 0.0,
                "mean_num_failed_negative_candidates": float(thm["num_failed_negative_candidates"].mean()) if not thm.empty else 0.0,
            }
        )
    if reports:
        pd.concat(reports, ignore_index=True).to_csv("outputs/reports/difficulty_distribution.csv", index=False)
    if complexity_reports:
        write_json(
            "outputs/reports/difficulty_target_report.json",
            {
                "target": "theorem_features.theorem_complexity_score",
                "target_source": "proof_length_tactic_count_premise_count_negative_candidates",
                "splits": complexity_reports,
            },
        )
