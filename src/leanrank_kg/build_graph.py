from __future__ import annotations

from itertools import combinations

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


def _edge_frame(source: pd.Series, target: pd.Series, edge_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": source.astype(str).to_numpy(),
            "target": target.astype(str).to_numpy(),
            "edge_type": edge_type,
            "weight": 1.0,
        }
    )


def _map_file_ids(paths: pd.Series) -> pd.Series:
    path_text = paths.fillna("").astype(str)
    id_by_path = {path: file_id(path) for path in path_text[path_text.astype(bool)].unique()}
    return path_text.map(id_by_path).fillna("")


def _co_occurrence_edges(pos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pos.empty:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight"])
    for premise_ids in pos.groupby("proof_state_id", sort=False)["premise_id"].unique():
        unique_ids = sorted(str(premise_id) for premise_id in premise_ids)
        for left, right in combinations(unique_ids, 2):
            rows.append((left, right, "co_occurs_with", 1.0))
            rows.append((right, left, "co_occurs_with", 1.0))
    return pd.DataFrame(rows, columns=["source", "target", "edge_type", "weight"])


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
        premise_files["id"] = _map_file_ids(premise_files["file_path"])
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
    tactic_targets = "tactic:" + ps["theorem_id"].astype(str) + ":" + ps["tactic_idx"].astype(int).astype(str)
    tactic_nodes = pd.DataFrame({"id": tactic_targets, "node_type": "TacticStep"}).drop_duplicates("id")
    nodes = pd.concat([nodes, tactic_nodes], ignore_index=True).drop_duplicates("id")

    edge_frames = [
        _edge_frame(ps["theorem_id"], ps["id"], "has_proof_state"),
        _edge_frame(ps["id"], tactic_targets, "at_tactic_step"),
        _edge_frame(thm["id"], _map_file_ids(thm["file_path"]), "appears_in_file"),
    ]
    premise_with_file = prem[prem["file_path"].fillna("").astype(bool)]
    if not premise_with_file.empty:
        edge_frames.append(_edge_frame(premise_with_file["id"], _map_file_ids(premise_with_file["file_path"]), "defined_in_file"))
    edge_frames.append(_edge_frame(pos["proof_state_id"], pos["premise_id"], "positive_uses"))

    ps_to_theorem = ps.set_index("id")["theorem_id"].to_dict()
    invokes = pos.assign(theorem_source=pos["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_source"])
    if not invokes.empty:
        edge_frames.append(_edge_frame(invokes["theorem_source"], invokes["premise_id"], "invokes_premise"))
    edge_frames.append(_co_occurrence_edges(pos))
    edge_frames.append(_edge_frame(neg["proof_state_id"], neg["premise_id"], "negative_candidate"))
    edge_df = pd.concat(edge_frames, ignore_index=True).drop_duplicates()
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
