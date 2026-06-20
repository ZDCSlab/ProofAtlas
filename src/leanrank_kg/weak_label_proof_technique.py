from __future__ import annotations

import pandas as pd

from .utils import SPLITS, technique_id, write_json, write_parquet

RULES = {
    "simplification": ["simp", "@[simp]", "_simp"],
    "rewriting_or_coercion": ["rw", "rewrite", "Eq", "congr", "coe", "cast"],
    "typeclass_resolution": ["inferInstance", "inst", "typeclass"],
    "definition_unfolding": ["unfold", "defeq"],
    "theorem_application": ["exact", "apply", "refine"],
    "extensionality": ["ext", "extensionality"],
    "case_or_constructor_reasoning": ["cases", "constructor", "rec"],
    "logical_reasoning": ["intro", "and", "or", "iff", "forall", "exists"],
    "induction": ["induction", "rec_on", "casesOn"],
    "contradiction": ["by_contra", "contradiction", "not_not"],
    "computation": ["norm_num", "decide", "ring"],
    "automation": ["omega", "linarith", "nlinarith", "aesop"],
}


def labels_for_text(text: str, max_labels: int = 5) -> list[dict[str, str]]:
    text_l = (text or "").lower()
    found = []
    for label, needles in RULES.items():
        for needle in needles:
            if needle.lower() in text_l:
                found.append({"label": label, "provenance": needle})
                break
    return found[:max_labels]


def run(config_path: str) -> None:
    max_labels = 5
    pool = [{"label": label, "rules": needles} for label, needles in RULES.items()]
    write_json("outputs/reports/proof_technique_candidate_pool.json", pool)
    distributions = []
    provenance = []
    for split in SPLITS + ["demo"]:
        try:
            ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
            prem = pd.read_parquet(f"data/processed/{split}/premises.parquet")
        except FileNotFoundError:
            continue
        ps_rows = []
        for row in ps.to_dict(orient="records"):
            text = " ".join([row.get("context", ""), row.get("goal_text", ""), row.get("tactic", "")])
            for lab in labels_for_text(text, max_labels):
                ps_rows.append({"proof_state_id": row["id"], "technique_id": technique_id(lab["label"]), **lab})
                provenance.append({"split": split, "entity_type": "proof_state", **lab})
        prem_rows = []
        for row in prem.to_dict(orient="records"):
            for lab in labels_for_text(" ".join([row.get("full_name", ""), row.get("code", "")]), max_labels):
                prem_rows.append({"premise_id": row["id"], "technique_id": technique_id(lab["label"]), **lab})
                provenance.append({"split": split, "entity_type": "premise", **lab})
        ps_df = pd.DataFrame(ps_rows, columns=["proof_state_id", "technique_id", "label", "provenance"]).drop_duplicates()
        prem_df = pd.DataFrame(prem_rows, columns=["premise_id", "technique_id", "label", "provenance"]).drop_duplicates()
        technique_rows = []
        combined = pd.concat(
            [
                ps_df[["technique_id", "label", "provenance"]] if not ps_df.empty else pd.DataFrame(columns=["technique_id", "label", "provenance"]),
                prem_df[["technique_id", "label", "provenance"]] if not prem_df.empty else pd.DataFrame(columns=["technique_id", "label", "provenance"]),
            ],
            ignore_index=True,
        )
        for label, group in combined.groupby("label"):
            technique_rows.append(
                {
                    "id": technique_id(label),
                    "label": label,
                    "provenance": ",".join(sorted(set(group["provenance"].dropna().astype(str)))),
                }
            )
        technique_df = pd.DataFrame(technique_rows, columns=["id", "label", "provenance"]).drop_duplicates("id")
        write_parquet(ps_df, f"data/processed/{split}/proof_state_techniques.parquet")
        write_parquet(prem_df, f"data/processed/{split}/premise_techniques.parquet")
        write_parquet(technique_df, f"data/processed/{split}/proof_techniques.parquet")
        if not ps_df.empty:
            counts = ps_df["label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            counts["split"] = split
            distributions.append(counts)
    dist = pd.concat(distributions, ignore_index=True) if distributions else pd.DataFrame(columns=["label", "count", "split"])
    dist.to_csv("outputs/reports/proof_technique_distribution.csv", index=False)
    pd.DataFrame(provenance).to_csv("outputs/reports/proof_technique_label_provenance.csv", index=False)
