from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _join_values(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, Iterable):
        return " ".join(str(value) for value in values if str(value).strip())
    return str(values)


def proof_state_profile(row: pd.Series | dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            str(row.get("full_name", "") or ""),
            " ".join(str(row.get(key, "") or "") for key in ["domain_tag", "subdomain_tag"]).strip(),
            _join_values(row.get("local_hypotheses", [])),
            str(row.get("goal_text", "") or ""),
            _join_values(row.get("symbols", [])),
        ]
        if part
    )


def premise_profile(row: pd.Series | dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            str(row.get("full_name", "") or ""),
            str(row.get("file_path", "") or ""),
            " ".join(str(row.get(key, "") or "") for key in ["domain_tag", "subdomain_tag"]).strip(),
            str(row.get("code", "") or ""),
        ]
        if part
    )


def theorem_profiles(theorems: pd.DataFrame, proof_states: pd.DataFrame, *, max_states: int = 5) -> pd.DataFrame:
    proof_states = proof_states.copy()
    proof_states["theorem_id"] = proof_states["theorem_id"].astype(str)
    grouped = {theorem_id: group.sort_values("tactic_idx") for theorem_id, group in proof_states.groupby("theorem_id", sort=False)}
    rows = []
    for theorem in theorems.to_dict(orient="records"):
        theorem_id = str(theorem["id"])
        states = grouped.get(theorem_id, pd.DataFrame()).head(max_states)
        goals = []
        symbols: list[str] = []
        local_hypotheses = []
        if not states.empty:
            goals = [str(value) for value in states["goal_text"].fillna("").tolist() if str(value).strip()]
            for values in states.get("symbols", pd.Series(dtype=object)).tolist():
                if isinstance(values, list):
                    symbols.extend(str(value) for value in values)
            for values in states.get("local_hypotheses", pd.Series(dtype=object)).tolist():
                if isinstance(values, list):
                    local_hypotheses.extend(str(value) for value in values[:4])
        profile = "\n".join(
            part
            for part in [
                str(theorem.get("full_name", "") or ""),
                str(theorem.get("file_path", "") or ""),
                " ".join(str(theorem.get(key, "") or "") for key in ["domain_tag", "subdomain_tag"]).strip(),
                "\n".join(goals),
                " ".join(sorted(set(symbols))),
                "\n".join(local_hypotheses[:12]),
            ]
            if part
        )
        rows.append(
            {
                "theorem_id": theorem_id,
                "full_name": theorem.get("full_name", ""),
                "file_path": theorem.get("file_path", ""),
                "domain_tag": theorem.get("domain_tag", ""),
                "subdomain_tag": theorem.get("subdomain_tag", ""),
                "profile_text": profile,
                "profile_proof_state_count": int(len(states)),
                "profile_max_states": int(max_states),
            }
        )
    return pd.DataFrame(rows)
