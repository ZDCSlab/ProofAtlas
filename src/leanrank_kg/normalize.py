from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .parse_context import parse_context
from .utils import (
    SPLITS,
    domain_from_path,
    file_id,
    load_config,
    premise_id,
    proof_state_id,
    theorem_id,
    write_jsonl,
    write_parquet,
)
from .validate import validate_context_coverage, validate_processed_schemas


def _premise_rows(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for item in items:
        name = str(item.get("full_name", "")).strip()
        if name:
            path = str(item.get("file_path", ""))
            domain, subdomain = domain_from_path(path)
            rows.append(
                {
                    "id": premise_id(name),
                    "full_name": name,
                    "file_path": path,
                    "code": str(item.get("code", "")),
                    "domain_tag": domain,
                    "subdomain_tag": subdomain,
                }
            )
    return rows


def normalize_split(split: str, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    theorems, proof_states, premises, files, pos_edges, neg_edges = [], [], [], [], [], []
    errors = []
    for row in df.to_dict(orient="records"):
        try:
            tid = theorem_id(row["full_name"])
            psid = proof_state_id(row["full_name"], int(row["tactic_idx"]), row["context"])
            parsed = parse_context(row["context"])
            theorems.append(
                {
                    "id": tid,
                    "full_name": row["full_name"],
                    "file_path": row["file_path"],
                    "domain_tag": row.get("domain_tag", ""),
                    "subdomain_tag": row.get("subdomain_tag", ""),
                    "split": split,
                }
            )
            proof_states.append(
                {
                    "id": psid,
                    "theorem_id": tid,
                    "full_name": row["full_name"],
                    "tactic_idx": int(row["tactic_idx"]),
                    "context": row["context"],
                    "goal_text": parsed["goal_text"],
                    "local_hypotheses": parsed["local_hypotheses"],
                    "symbols": parsed["symbols"],
                    "tactic": row.get("tactic", ""),
                    "domain_tag": row.get("domain_tag", ""),
                    "subdomain_tag": row.get("subdomain_tag", ""),
                    "split": split,
                }
            )
            files.append({"id": file_id(row["file_path"]), "file_path": row["file_path"], "domain_tag": row.get("domain_tag", ""), "subdomain_tag": row.get("subdomain_tag", "")})
            all_pos = [row["pos_premise"], *row.get("all_pos_premises", [])]
            for prem in _premise_rows(all_pos):
                premises.append(prem)
                pos_edges.append({"proof_state_id": psid, "premise_id": prem["id"], "source": "positive"})
                if prem["file_path"]:
                    files.append({"id": file_id(prem["file_path"]), "file_path": prem["file_path"], "domain_tag": prem["domain_tag"], "subdomain_tag": prem["subdomain_tag"]})
            for prem in _premise_rows(row.get("neg_premises", [])):
                premises.append(prem)
                neg_edges.append({"proof_state_id": psid, "premise_id": prem["id"], "source": "negative"})
        except Exception as exc:
            errors.append({"split": split, "row": row, "error": str(exc)})
    write_jsonl("outputs/reports/normalization_errors.jsonl", errors)
    frames = {
        "theorems": pd.DataFrame(theorems).drop_duplicates("id"),
        "proof_states": pd.DataFrame(proof_states).drop_duplicates("id"),
        "premises": pd.DataFrame(premises).drop_duplicates("id"),
        "file_modules": pd.DataFrame(files).drop_duplicates("id"),
        "positive_edges": pd.DataFrame(pos_edges).drop_duplicates(),
        "negative_edges": pd.DataFrame(neg_edges).drop_duplicates(),
    }
    return frames


def run(config_path: str) -> None:
    config = load_config(config_path)
    demo_limit = int(config["sample"]["committed_demo_rows"])
    demo_parts: dict[str, list[pd.DataFrame]] = {}
    full_parts: dict[str, list[pd.DataFrame]] = {}
    for split in SPLITS:
        path = Path(f"data/sample/{split}_rows.parquet")
        if not path.exists():
            continue
        frames = normalize_split(split, pd.read_parquet(path))
        for name, frame in frames.items():
            write_parquet(frame, f"data/processed/{split}/{name}.parquet")
            full_parts.setdefault(name, []).append(frame)
            demo_parts.setdefault(name, []).append(frame.head(max(1, demo_limit // 10)))
    demo_tables = {}
    for name, parts in demo_parts.items():
        demo = pd.concat(parts, ignore_index=True)
        demo = demo.drop_duplicates("id") if "id" in demo.columns else demo.drop_duplicates()
        demo_tables[name] = demo.head(demo_limit)
    if demo_tables:
        all_tables = {name: pd.concat(parts, ignore_index=True).drop_duplicates("id") if "id" in pd.concat(parts, ignore_index=True).columns else pd.concat(parts, ignore_index=True).drop_duplicates() for name, parts in full_parts.items()}
        proof_state_ids = set(demo_tables["proof_states"]["id"])
        for edge_name in ["positive_edges", "negative_edges"]:
            demo_tables[edge_name] = demo_tables[edge_name][demo_tables[edge_name]["proof_state_id"].isin(proof_state_ids)].drop_duplicates()
        required_premises = set(demo_tables["positive_edges"]["premise_id"]) | set(demo_tables["negative_edges"]["premise_id"])
        demo_tables["premises"] = all_tables["premises"][all_tables["premises"]["id"].isin(required_premises)].drop_duplicates("id")
        required_theorems = set(demo_tables["proof_states"]["theorem_id"])
        demo_tables["theorems"] = all_tables["theorems"][all_tables["theorems"]["id"].isin(required_theorems)].drop_duplicates("id")
        required_files = set(demo_tables["theorems"]["file_path"]) | set(demo_tables["premises"]["file_path"])
        demo_tables["file_modules"] = all_tables["file_modules"][all_tables["file_modules"]["file_path"].isin(required_files)].drop_duplicates("id")
    for name, demo in demo_tables.items():
        write_parquet(demo, f"data/processed/demo/{name}.parquet")
    validate_processed_schemas()
    validate_context_coverage()
