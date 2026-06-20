from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from .retrieve import retrieve_premises
from .utils import SPLITS, read_json, write_json


def run(config_path: str) -> None:
    top_ks = [1, 5, 10]
    examples = []
    recalls = {k: [] for k in top_ks}
    rr = []
    coverage = []
    train_premises = set(pd.read_parquet("data/processed/train/premises.parquet")["id"])
    for split in ["val", "test"]:
        try:
            pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
            ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet").set_index("id")
        except FileNotFoundError:
            continue
        for row in pos.drop_duplicates("proof_state_id").head(50).to_dict(orient="records"):
            gold = row["premise_id"]
            in_index = gold in train_premises
            coverage.append(in_index)
            retrieved = retrieve_premises(row["proof_state_id"], 10, split=split)
            ids = [r["premise_id"] for r in retrieved]
            if in_index:
                for k in top_ks:
                    recalls[k].append(float(gold in ids[:k]))
                rr.append(1.0 / (ids.index(gold) + 1) if gold in ids else 0.0)
            if len(examples) < 20:
                pstate = ps.loc[row["proof_state_id"]]
                examples.append(
                    {
                        "split": split,
                        "proof_state_id": row["proof_state_id"],
                        "proof_state": str(pstate["goal_text"])[:300],
                        "gold_positive_premise": gold,
                        "gold_in_train_index": in_index,
                        "top_retrieved_premises": retrieved[:5],
                    }
                )
    if len(examples) < 20:
        pos = pd.read_parquet("data/processed/train/positive_edges.parquet")
        ps = pd.read_parquet("data/processed/train/proof_states.parquet").set_index("id")
        for row in pos.drop_duplicates("proof_state_id").to_dict(orient="records"):
            if len(examples) >= 20:
                break
            retrieved = retrieve_premises(row["proof_state_id"], 10, split="train")
            pstate = ps.loc[row["proof_state_id"]]
            examples.append(
                {
                    "split": "train",
                    "proof_state_id": row["proof_state_id"],
                    "proof_state": str(pstate["goal_text"])[:300],
                    "gold_positive_premise": row["premise_id"],
                    "gold_in_train_index": True,
                    "top_retrieved_premises": retrieved[:5],
                }
            )
    metrics = {f"Recall@{k}": float(sum(vals) / len(vals)) if vals else 0.0 for k, vals in recalls.items()}
    metrics["MRR"] = float(sum(rr) / len(rr)) if rr else 0.0
    metrics["AUC"] = read_json("outputs/reports/ranker_validation_metrics.json", {}).get("validation_auc", None)
    metrics["gold_premise_coverage"] = float(sum(coverage) / len(coverage)) if coverage else 0.0
    label_coverages = []
    for split in SPLITS:
        try:
            ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
            tech = pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet")
        except FileNotFoundError:
            continue
        if len(ps):
            label_coverages.append(float(ps["id"].isin(set(tech["proof_state_id"])).mean()))
    metrics["proof_technique_label_coverage"] = float(sum(label_coverages) / len(label_coverages)) if label_coverages else 0.0
    try:
        diff = pd.read_csv("outputs/reports/difficulty_distribution.csv").to_dict(orient="records")
    except FileNotFoundError:
        diff = []
    metrics["difficulty_bucket_distribution"] = diff
    write_json("outputs/reports/metrics.json", metrics)
    write_json("outputs/reports/retrieval_examples.json", examples)
    with open("outputs/reports/retrieval_examples.md", "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(f"## {ex['proof_state_id']}\nGold: `{ex['gold_positive_premise']}`\n\n")
    domain = read_json("outputs/reports/domain_distribution.json", {})
    write_json("outputs/reports/domain_coverage.json", domain)
