from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if str(item).strip())
    if hasattr(value, "tolist"):
        return " ".join(str(item) for item in value.tolist() if str(item).strip())
    return str(value or "")


def _strategy_text(value: Any) -> str:
    if not isinstance(value, list):
        if hasattr(value, "tolist"):
            value = value.tolist()
        else:
            return str(value or "")
    parts = []
    for item in value:
        if isinstance(item, dict):
            parts.extend(
                str(item.get(key, "") or "")
                for key in ["facet", "target", "direction_or_action"]
                if str(item.get(key, "") or "").strip()
            )
        else:
            parts.append(str(item))
    return " ".join(parts)


def enrichment_path(split: str, output_dir: str = "outputs/proofatlas") -> Path:
    return Path(output_dir) / "llm" / f"theorem_enrichment_{split}.parquet"


def load_theorem_enrichment(split: str, output_dir: str = "outputs/proofatlas") -> pd.DataFrame:
    path = enrichment_path(split, output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM theorem enrichment: {path}. Run llm-enrich-theorems --split {split} first.")
    df = pd.read_parquet(path)
    df["theorem_id"] = df["theorem_id"].astype(str)
    return df


def append_enrichment_text(profiles: pd.DataFrame, split: str, output_dir: str = "outputs/proofatlas") -> pd.DataFrame:
    enrichment = load_theorem_enrichment(split, output_dir)
    enrichment_by_id = enrichment.set_index("theorem_id", drop=False).to_dict(orient="index")
    rows = []
    for row in profiles.to_dict(orient="records"):
        theorem_id = str(row["theorem_id"])
        extra = enrichment_by_id.get(theorem_id, {})
        enrichment_text = "\n".join(
            part
            for part in [
                str(extra.get("topic", "") or ""),
                str(extra.get("goal_pattern", "") or ""),
                _join_list(extra.get("mathematical_objects", [])),
                _join_list(extra.get("key_symbols", [])),
                _join_list(extra.get("useful_lemma_types", [])),
                _strategy_text(extra.get("strategy_facets", [])),
                _join_list(extra.get("likely_tactics", [])),
                _join_list(extra.get("difficulty_reasons", [])),
                str(extra.get("difficulty_bucket_hint", "") or ""),
            ]
            if part
        )
        row["llm_enrichment_text"] = enrichment_text
        row["profile_text"] = "\n".join(part for part in [str(row.get("profile_text", "") or ""), enrichment_text] if part)
        rows.append(row)
    return pd.DataFrame(rows)
