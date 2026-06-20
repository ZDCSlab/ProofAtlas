from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .utils import minmax, namespace, write_json

FEATURE_GROUPS = {
    "embedding_similarity": ["cosine_similarity"],
    "namespace_domain": ["same_namespace", "same_domain"],
    "proof_technique": ["proof_technique_overlap"],
    "difficulty": ["proof_state_difficulty", "negative_candidate_hardness"],
    "frequency": ["premise_frequency"],
    "symbol_overlap": ["symbol_name_overlap", "symbol_context_overlap"],
    "graph": ["graph_premise_degree"],
    "theorem_neighborhood": ["theorem_neighborhood_premise_score"],
}


def _pairs(split: str) -> pd.DataFrame:
    pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet").assign(label=1)
    neg = pd.read_parquet(f"data/processed/{split}/negative_edges.parquet").assign(label=0)
    return pd.concat([pos, neg], ignore_index=True).rename(columns={"proof_state_id": "ps", "premise_id": "prem"})


def _tokens(*parts: object) -> set[str]:
    text = " ".join(str(part or "") for part in parts).replace(".", " ").replace("_", " ")
    return {token.lower() for token in text.split() if len(token) >= 3}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _features(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    pairs = (
        pairs.groupby("label", group_keys=False)
        .head(1000)
        .sort_values(["label", "ps", "prem"])
        .reset_index(drop=True)
    )
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet").set_index("id")
    premises = pd.read_parquet(f"data/processed/{split}/premises.parquet").set_index("id")
    theorems = pd.read_parquet(f"data/processed/{split}/theorems.parquet").set_index("id")
    ps_features = pd.read_parquet(f"data/processed/{split}/proof_state_features.parquet").set_index("id")
    ps_tech = pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet")
    prem_tech = pd.read_parquet(f"data/processed/{split}/premise_techniques.parquet")
    ps_labels = {key: set(group["label"]) for key, group in ps_tech.groupby("proof_state_id")} if not ps_tech.empty else {}
    prem_labels = {key: set(group["label"]) for key, group in prem_tech.groupby("premise_id")} if not prem_tech.empty else {}
    ps_x = sparse.load_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz").tocsr()
    prem_x = sparse.load_npz(f"outputs/embeddings/{split}_premise_embeddings.npz").tocsr()
    ps_row = {pid: i for i, pid in enumerate(proof_states.index.tolist())}
    prem_row = {pid: i for i, pid in enumerate(premises.index.tolist())}
    positive_edges = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    premise_frequency = positive_edges.groupby("premise_id").size().rename("premise_frequency_raw")
    if premise_frequency.empty:
        premise_frequency_norm = premise_frequency
    else:
        premise_frequency_norm = minmax(premise_frequency).rename("premise_frequency")
    ps_to_theorem = proof_states["theorem_id"].to_dict()
    premise_theorem_edges = positive_edges.assign(theorem_id=positive_edges["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    theorem_premise_counts = premise_theorem_edges.groupby("premise_id")["theorem_id"].nunique()
    theorem_premise_degree = minmax(theorem_premise_counts) if not theorem_premise_counts.empty else theorem_premise_counts
    theorem_records = theorems.to_dict(orient="index")
    theorem_neighbor_premises: dict[tuple[str, str], float] = {}
    if not premise_theorem_edges.empty:
        theorem_to_premises = {
            theorem_id: set(group["premise_id"])
            for theorem_id, group in premise_theorem_edges.groupby("theorem_id")
        }
        theorem_ids = list(theorem_records)
        for theorem_id in theorem_ids:
            info = theorem_records.get(theorem_id, {})
            same_area = [
                other_id
                for other_id in theorem_ids
                if other_id != theorem_id
                and (
                    theorem_records.get(other_id, {}).get("domain_tag") == info.get("domain_tag")
                    or namespace(theorem_records.get(other_id, {}).get("full_name", "")) == namespace(info.get("full_name", ""))
                )
            ]
            counts: dict[str, int] = {}
            for other_id in same_area:
                for premise_id in theorem_to_premises.get(other_id, set()):
                    counts[premise_id] = counts.get(premise_id, 0) + 1
            denom = max(len(same_area), 1)
            for premise_id, count in counts.items():
                theorem_neighbor_premises[(theorem_id, premise_id)] = min(count / denom, 1.0)
    rows = []
    for row in pairs.to_dict(orient="records"):
        ps_record = proof_states.loc[row["ps"]] if row["ps"] in proof_states.index else pd.Series({"full_name": "", "domain_tag": "", "file_path": ""})
        prem_record = premises.loc[row["prem"]] if row["prem"] in premises.index else pd.Series({"full_name": "", "domain_tag": "", "file_path": ""})
        ps_feature_row = ps_features.loc[row["ps"]] if row["ps"] in ps_features.index else {}
        if row["ps"] in ps_row and row["prem"] in prem_row:
            cosine = float(ps_x[ps_row[row["ps"]]].multiply(prem_x[prem_row[row["prem"]]]).sum())
        else:
            cosine = 0.0
        shared_labels = ps_labels.get(row["ps"], set()) & prem_labels.get(row["prem"], set())
        ps_tokens = _tokens(ps_record.get("full_name", ""), ps_record.get("context", ""), ps_record.get("goal_text", ""))
        prem_name_tokens = _tokens(prem_record.get("full_name", ""))
        prem_context_tokens = _tokens(prem_record.get("full_name", ""), prem_record.get("code", ""))
        theorem_id = ps_record.get("theorem_id", "")
        rows.append(
            {
                "cosine_similarity": cosine,
                "same_namespace": float(namespace(ps_record.get("full_name", "")) == namespace(prem_record.get("full_name", ""))),
                "same_domain": float(ps_record.get("domain_tag", "") == prem_record.get("domain_tag", "")),
                "proof_technique_overlap": float(bool(shared_labels)),
                "proof_state_difficulty": float(ps_feature_row.get("difficulty_score", 0.0)),
                "negative_candidate_hardness": float(ps_feature_row.get("negative_candidate_hardness", 0.0)),
                "premise_frequency": float(premise_frequency_norm.get(row["prem"], 0.0)) if not premise_frequency_norm.empty else 0.0,
                "symbol_name_overlap": _jaccard(ps_tokens, prem_name_tokens),
                "symbol_context_overlap": _jaccard(ps_tokens, prem_context_tokens),
                "graph_premise_degree": float(theorem_premise_degree.get(row["prem"], 0.0)) if not theorem_premise_degree.empty else 0.0,
                "theorem_neighborhood_premise_score": float(theorem_neighbor_premises.get((theorem_id, row["prem"]), 0.0)),
                "label": row["label"],
            }
        )
    return pd.DataFrame(rows)


def _auc_for_columns(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]) -> float | None:
    if not columns or validation.empty or validation["label"].nunique() <= 1 or train["label"].nunique() <= 1:
        return None
    model = LogisticRegression(max_iter=500).fit(train[columns], train["label"])
    return float(roc_auc_score(validation["label"], model.predict_proba(validation[columns])[:, 1]))


def _ablation_report(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    feature_columns = [col for col in train.columns if col != "label"]
    full_auc = _auc_for_columns(train, validation, feature_columns)
    groups = {}
    for group, columns in FEATURE_GROUPS.items():
        available = [col for col in columns if col in feature_columns]
        remaining = [col for col in feature_columns if col not in available]
        removed_auc = _auc_for_columns(train, validation, remaining)
        only_auc = _auc_for_columns(train, validation, available)
        groups[group] = {
            "columns": available,
            "auc_without_group": removed_auc,
            "auc_group_only": only_auc,
            "delta_without_group": None if full_auc is None or removed_auc is None else float(full_auc - removed_auc),
        }
    return {
        "full_auc": full_auc,
        "feature_columns": feature_columns,
        "groups": groups,
    }


def run(config_path: str) -> None:
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    train = _features(_pairs("train"), "train")
    X = train.drop(columns=["label"])
    y = train["label"]
    model = LogisticRegression(max_iter=500).fit(X, y)
    joblib.dump(model, "outputs/models/premise_ranker.joblib")
    metrics = {
        "train_pairs": int(len(train)),
        "feature_columns": X.columns.tolist(),
        "feature_groups": FEATURE_GROUPS,
    }
    try:
        val = _features(_pairs("val"), "val")
        if not val.empty and val["label"].nunique() > 1:
            metrics["validation_auc"] = float(roc_auc_score(val["label"], model.predict_proba(val.drop(columns=["label"]))[:, 1]))
            metrics["validation_pairs"] = int(len(val))
            metrics["feature_ablation"] = _ablation_report(train, val)
    except Exception as exc:
        metrics["validation_error"] = str(exc)
    write_json("outputs/reports/ranker_validation_metrics.json", metrics)
