from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .utils import minmax, namespace, write_json


def _pairs(split: str) -> pd.DataFrame:
    pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet").assign(label=1)
    neg = pd.read_parquet(f"data/processed/{split}/negative_edges.parquet").assign(label=0)
    return pd.concat([pos, neg], ignore_index=True).rename(columns={"proof_state_id": "ps", "premise_id": "prem"})


def _features(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    pairs = (
        pairs.groupby("label", group_keys=False)
        .head(1000)
        .sort_values(["label", "ps", "prem"])
        .reset_index(drop=True)
    )
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet").set_index("id")
    premises = pd.read_parquet(f"data/processed/{split}/premises.parquet").set_index("id")
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
        rows.append(
            {
                "cosine_similarity": cosine,
                "same_namespace": float(namespace(ps_record.get("full_name", "")) == namespace(prem_record.get("full_name", ""))),
                "same_domain": float(ps_record.get("domain_tag", "") == prem_record.get("domain_tag", "")),
                "proof_technique_overlap": float(bool(shared_labels)),
                "proof_state_difficulty": float(ps_feature_row.get("difficulty_score", 0.0)),
                "negative_candidate_hardness": float(ps_feature_row.get("negative_candidate_hardness", 0.0)),
                "premise_frequency": float(premise_frequency_norm.get(row["prem"], 0.0)) if not premise_frequency_norm.empty else 0.0,
                "label": row["label"],
            }
        )
    return pd.DataFrame(rows)


def run(config_path: str) -> None:
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    train = _features(_pairs("train"), "train")
    X = train.drop(columns=["label"])
    y = train["label"]
    model = LogisticRegression(max_iter=500).fit(X, y)
    joblib.dump(model, "outputs/models/premise_ranker.joblib")
    metrics = {"train_pairs": int(len(train))}
    try:
        val = _features(_pairs("val"), "val")
        if not val.empty and val["label"].nunique() > 1:
            metrics["validation_auc"] = float(roc_auc_score(val["label"], model.predict_proba(val.drop(columns=["label"]))[:, 1]))
    except Exception as exc:
        metrics["validation_error"] = str(exc)
    write_json("outputs/reports/ranker_validation_metrics.json", metrics)
