from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io import write_json, write_parquet
from .llm_client import DEFAULT_MODEL, DeepSeekClient, prompt_hash
from .profiles import theorem_profiles


PROMPT_VERSION = "theorem_semantic_strategy_v2"


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit] + "..."


def _as_list(value: Any, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        out = [str(item) for item in value]
    elif hasattr(value, "tolist"):
        out = [str(item) for item in value.tolist()]
    elif value is None:
        out = []
    else:
        out = [str(value)]
    return out[:limit] if limit is not None else out


def _profile_rows(split: str, *, max_states: int) -> list[dict[str, Any]]:
    theorems = pd.read_parquet(f"data/processed/{split}/theorems.parquet")
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    profiles = theorem_profiles(theorems, proof_states, max_states=max_states)
    proof_states["theorem_id"] = proof_states["theorem_id"].astype(str)
    grouped = {theorem_id: group.sort_values("tactic_idx") for theorem_id, group in proof_states.groupby("theorem_id", sort=False)}
    rows = []
    for profile in profiles.to_dict(orient="records"):
        theorem_id = str(profile["theorem_id"])
        states = grouped.get(theorem_id, pd.DataFrame()).head(max_states)
        rows.append(
            {
                **profile,
                "goals": [_truncate(value, 500) for value in states.get("goal_text", pd.Series(dtype=object)).fillna("").tolist()],
                "local_hypotheses": [
                    _truncate(" | ".join(_as_list(value, 4)), 500)
                    for value in states.get("local_hypotheses", pd.Series(dtype=object)).tolist()
                ],
                "symbols": sorted(set().union(*(_as_list(value) for value in states.get("symbols", pd.Series(dtype=object)).tolist())))[:80],
            }
        )
    return rows


def build_prompt(row: dict[str, Any]) -> str:
    goals = "\n".join(f"{idx + 1}. {goal}" for idx, goal in enumerate(row.get("goals", [])) if goal)
    hyps = "\n".join(f"{idx + 1}. {hyp}" for idx, hyp in enumerate(row.get("local_hypotheses", [])) if hyp)
    symbols = ", ".join(_as_list(row.get("symbols", []), 80))
    return f"""You are enriching one Lean theorem profile for retrieval.

Use only the provided theorem metadata, proof-state goals, hypotheses, and symbols.
Do not invent concrete Lean lemma names. If a useful lemma is abstract, describe its type.
Infer strategy and difficulty from the visible theorem/proof-state text only.
Return strict JSON only.

Theorem:
- theorem_id: {row.get("theorem_id")}
- full_name: {row.get("full_name")}
- file_path: {row.get("file_path")}
- domain: {row.get("domain_tag")}
- subdomain: {row.get("subdomain_tag")}

Proof-state goals:
{goals}

Local hypotheses:
{hyps}

Symbols:
{symbols}

Return JSON with this schema:
{{
  "theorem_id": "{row.get("theorem_id")}",
  "topic": "short semantic topic",
  "mathematical_objects": ["specific object or structure"],
  "goal_pattern": "short proof obligation pattern",
  "key_symbols": ["symbol or declaration from the input"],
  "useful_lemma_types": ["abstract lemma type, not invented names"],
  "strategy_facets": [
    {{
      "facet": "rewrite_transport|case_analysis|induction_recursion|typeclass_instance_resolution|order_inequality_reasoning|algebraic_computation|set_membership_reasoning|theorem_application|other",
      "target": "expression or object being transformed",
      "direction_or_action": "what the proof likely needs",
      "confidence": 0.0
    }}
  ],
  "likely_tactics": ["simp", "rw", "cases"],
  "difficulty_reasons": ["specific reason"],
  "difficulty_bucket_hint": "easy|medium|hard|unknown"
}}"""


def _cache_path(output_dir: str, split: str, theorem_id: str, *, model: str) -> Path:
    safe = theorem_id.replace("/", "_").replace(":", "_")
    model_safe = model.replace("/", "_").replace(":", "_")
    return Path(output_dir) / "llm" / "theorem_enrichment_cache" / PROMPT_VERSION / model_safe / split / f"{safe}.json"


def _run_one(row: dict[str, Any], *, client: DeepSeekClient, output_dir: str, split: str, force: bool) -> dict[str, Any]:
    prompt = build_prompt(row)
    cache_path = _cache_path(output_dir, split, str(row["theorem_id"]), model=client.model)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    parsed, raw = client.chat_json(prompt, max_tokens=1800)
    record = {
        "theorem_id": str(row["theorem_id"]),
        "full_name": str(row.get("full_name", "")),
        "split": split,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(prompt, model=client.model),
        "model": client.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parse_status": "ok",
        "enrichment": parsed,
        "raw_response": raw,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def run(
    split: str,
    *,
    limit: int | None = None,
    concurrency: int = 8,
    model: str = DEFAULT_MODEL,
    output_dir: str = "outputs/proofatlas",
    force: bool = False,
    max_states: int = 5,
) -> dict[str, Any]:
    rows = _profile_rows(split, max_states=max_states)
    if limit is not None:
        rows = rows[:limit]
    client = DeepSeekClient(model=model)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_run_one, row, client=client, output_dir=output_dir, split=split, force=force): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001
                errors.append({"theorem_id": str(row.get("theorem_id")), "error": str(exc)})
    records.sort(key=lambda item: item["theorem_id"])
    out_dir = Path(output_dir) / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{split}_{limit}" if limit is not None else split
    jsonl_path = out_dir / f"theorem_enrichment_{suffix}.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + ("\n" if records else ""), encoding="utf-8")
    flat_rows = []
    for record in records:
        enrichment = record.get("enrichment", {})
        flat_rows.append(
            {
                "theorem_id": record["theorem_id"],
                "full_name": record.get("full_name", ""),
                "split": split,
                "topic": enrichment.get("topic", ""),
                "mathematical_objects": enrichment.get("mathematical_objects", []),
                "goal_pattern": enrichment.get("goal_pattern", ""),
                "key_symbols": enrichment.get("key_symbols", []),
                "useful_lemma_types": enrichment.get("useful_lemma_types", []),
                "strategy_facets": enrichment.get("strategy_facets", []),
                "likely_tactics": enrichment.get("likely_tactics", []),
                "difficulty_reasons": enrichment.get("difficulty_reasons", []),
                "difficulty_bucket_hint": enrichment.get("difficulty_bucket_hint", ""),
                "prompt_version": record["prompt_version"],
                "prompt_hash": record["prompt_hash"],
                "model": record["model"],
            }
        )
    parquet_path = out_dir / f"theorem_enrichment_{suffix}.parquet"
    write_parquet(pd.DataFrame(flat_rows), parquet_path)
    summary = {
        "task": "llm_theorem_semantic_strategy_enrichment",
        "split": split,
        "requested": len(rows),
        "succeeded": len(records),
        "failed": len(errors),
        "jsonl_path": str(jsonl_path),
        "parquet_path": str(parquet_path),
        "errors": errors[:20],
    }
    write_json(out_dir / f"theorem_enrichment_{suffix}_summary.json", summary)
    return summary
