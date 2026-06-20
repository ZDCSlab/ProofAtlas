from __future__ import annotations

import pandas as pd

from .theorem_similarity import theorem_similarity_rows
from .utils import SPLITS, load_config, technique_id, write_json, write_parquet
from .validate import validate_all_graphs


def run(config_path: str) -> None:
    config = load_config(config_path)
    top_k = int(config["similarity"]["theorem_top_k"])
    summary = {}
    for split in SPLITS + ["demo"]:
        try:
            nodes = pd.read_parquet(f"outputs/graph/{split}/nodes.parquet")
            edges = pd.read_parquet(f"outputs/graph/{split}/edges.parquet")
            thm = pd.read_parquet(f"data/processed/{split}/theorems.parquet")
            ps_tech = pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet")
            prem_tech = pd.read_parquet(f"data/processed/{split}/premise_techniques.parquet")
        except FileNotFoundError:
            continue
        extra_nodes = pd.DataFrame(
            [{"id": technique_id(label), "node_type": "ProofTechnique", "label": label} for label in sorted(set(ps_tech.get("label", [])) | set(prem_tech.get("label", [])))]
        )
        extra_edges = []
        for row in ps_tech.to_dict(orient="records"):
            extra_edges.append({"source": row["proof_state_id"], "target": row["technique_id"], "edge_type": "uses_proof_technique", "weight": 1.0})
        for row in theorem_similarity_rows(split, top_k):
            extra_edges.append(
                {
                    "source": row["source"],
                    "target": row["target"],
                    "edge_type": "similar_to_theorem",
                    "weight": row["score"],
                    "tfidf_similarity": row["tfidf_similarity"],
                    "shared_premise_score": row["shared_premise_score"],
                    "file_namespace_score": row["file_namespace_score"],
                    "proof_technique_overlap": row["proof_technique_overlap"],
                    "difficulty_similarity": row["difficulty_similarity"],
                }
            )
        nodes2 = pd.concat([nodes, extra_nodes], ignore_index=True).drop_duplicates("id") if not extra_nodes.empty else nodes
        edges2 = pd.concat([edges, pd.DataFrame(extra_edges)], ignore_index=True).drop_duplicates()
        stats = {
            "split": split,
            "node_count": int(len(nodes2)),
            "edge_count": int(len(edges2)),
            "node_counts_by_type": nodes2["node_type"].value_counts().to_dict(),
            "edge_counts_by_type": edges2["edge_type"].value_counts().to_dict(),
        }
        write_parquet(nodes2, f"outputs/graph/{split}/nodes_enriched.parquet")
        write_parquet(edges2, f"outputs/graph/{split}/edges_enriched.parquet")
        write_json(f"outputs/graph/{split}/graph_stats_enriched.json", stats)
        summary[split] = stats
    write_json("outputs/reports/graph_stats_summary.json", summary)
    validate_all_graphs()
