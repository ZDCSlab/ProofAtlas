from __future__ import annotations

import pandas as pd

from .utils import SPLITS, file_id, technique_id, write_json, write_parquet
from .validate import validate_all_graphs


def _nodes(df: pd.DataFrame, node_type: str, cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["id", "node_type"])
    keep = ["id"] + [c for c in (cols or []) if c in df.columns]
    out = df[keep].copy()
    out["node_type"] = node_type
    return out


def build_split(split: str, enriched: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = f"data/processed/{split}"
    thm = pd.read_parquet(f"{base}/theorems.parquet")
    ps = pd.read_parquet(f"{base}/proof_states.parquet")
    prem = pd.read_parquet(f"{base}/premises.parquet")
    files = pd.read_parquet(f"{base}/file_modules.parquet")
    pos = pd.read_parquet(f"{base}/positive_edges.parquet")
    neg = pd.read_parquet(f"{base}/negative_edges.parquet")
    premise_files = prem[prem["file_path"].fillna("").astype(bool)][["file_path", "domain_tag", "subdomain_tag"]].copy()
    if not premise_files.empty:
        premise_files["id"] = premise_files["file_path"].map(file_id)
        files = pd.concat([files, premise_files[["id", "file_path", "domain_tag", "subdomain_tag"]]], ignore_index=True).drop_duplicates("id")
    nodes = pd.concat(
        [
            _nodes(thm, "Theorem", ["full_name", "domain_tag", "subdomain_tag"]),
            _nodes(ps, "ProofState", ["theorem_id", "full_name", "tactic_idx", "goal_text", "domain_tag"]),
            _nodes(prem, "Premise", ["full_name", "domain_tag", "subdomain_tag"]),
            _nodes(files, "FileModule", ["file_path", "domain_tag", "subdomain_tag"]),
        ],
        ignore_index=True,
    ).drop_duplicates("id")
    edges = []
    for row in ps.to_dict(orient="records"):
        edges.append({"source": row["theorem_id"], "target": row["id"], "edge_type": "has_proof_state", "weight": 1.0})
        edges.append({"source": row["id"], "target": f"tactic:{int(row['tactic_idx'])}", "edge_type": "at_tactic_step", "weight": 1.0})
    tactic_nodes = pd.DataFrame([{"id": e["target"], "node_type": "TacticStep"} for e in edges if e["edge_type"] == "at_tactic_step"]).drop_duplicates("id")
    nodes = pd.concat([nodes, tactic_nodes], ignore_index=True).drop_duplicates("id")
    for row in thm.to_dict(orient="records"):
        edges.append({"source": row["id"], "target": file_id(row["file_path"]), "edge_type": "appears_in_file", "weight": 1.0})
    for row in prem.to_dict(orient="records"):
        if row.get("file_path"):
            edges.append({"source": row["id"], "target": file_id(row["file_path"]), "edge_type": "defined_in_file", "weight": 1.0})
    for row in pos.to_dict(orient="records"):
        edges.append({"source": row["proof_state_id"], "target": row["premise_id"], "edge_type": "positive_uses", "weight": 1.0})
        edges.append({"source": row["proof_state_id"], "target": row["premise_id"], "edge_type": "invokes_premise", "weight": 1.0})
    for proof_state_id, group in pos.groupby("proof_state_id"):
        premise_ids = sorted(set(group["premise_id"]))
        for i, left in enumerate(premise_ids):
            for right in premise_ids[i + 1 :]:
                edges.append({"source": left, "target": right, "edge_type": "co_occurs_with", "weight": 1.0})
                edges.append({"source": right, "target": left, "edge_type": "co_occurs_with", "weight": 1.0})
    for row in neg.to_dict(orient="records"):
        edges.append({"source": row["proof_state_id"], "target": row["premise_id"], "edge_type": "negative_candidate", "weight": 1.0})
    edge_df = pd.DataFrame(edges).drop_duplicates()
    stats = {
        "split": split,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edge_df)),
        "node_counts_by_type": nodes["node_type"].value_counts().to_dict(),
        "edge_counts_by_type": edge_df["edge_type"].value_counts().to_dict(),
    }
    return nodes, edge_df, stats


def run(config_path: str) -> None:
    summary = {}
    for split in SPLITS + ["demo"]:
        try:
            nodes, edges, stats = build_split(split)
        except FileNotFoundError:
            continue
        write_parquet(nodes, f"outputs/graph/{split}/nodes.parquet")
        write_parquet(edges, f"outputs/graph/{split}/edges.parquet")
        write_json(f"outputs/graph/{split}/graph_stats.json", stats)
        summary[split] = stats
    write_json("outputs/reports/graph_stats_summary.json", summary)
    validate_all_graphs()
