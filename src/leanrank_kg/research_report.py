from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from .utils import read_json, write_json
from .weak_label_proof_technique import labels_for_text


REPORT_PATH = Path("outputs/reports/research_report.md")
PREDICTION_PATH = Path("outputs/predictions/research_prediction_results.json")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _csv_records(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return pd.read_csv(p).to_dict(orient="records")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _task_spec_table(rows: list[tuple[str, str]]) -> str:
    return _table(["Field", "Description"], [[field, description] for field, description in rows])


def _top_by_split(rows: list[dict[str, Any]], split: str, key: str, n: int = 8) -> list[dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("split")) == split]
    return sorted(filtered, key=lambda row: float(row.get(key, 0)), reverse=True)[:n]


def _load_embedding(split: str, kind: str) -> np.ndarray:
    matrix = sparse.load_npz(f"outputs/embeddings/{split}_{kind}_embeddings.npz")
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _embedding_ids(split: str, entity_type: str) -> list[str]:
    meta = pd.read_parquet(f"outputs/embeddings/{split}_embedding_metadata.parquet")
    rows = meta[meta["entity_type"] == entity_type].sort_values("row_index")
    return [str(value) for value in rows["entity_id"].tolist()]


def _topk_neighbors(query: np.ndarray, candidates: np.ndarray, k: int = 10, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    all_idx = []
    all_scores = []
    top_k = min(k, len(candidates))
    for start in range(0, len(query), batch_size):
        sims = query[start : start + batch_size] @ candidates.T
        idx = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-scores, axis=1)
        all_idx.append(np.take_along_axis(idx, order, axis=1))
        all_scores.append(np.take_along_axis(scores, order, axis=1))
    return np.vstack(all_idx), np.vstack(all_scores)


def _labels_by_proof_state(split: str) -> dict[str, set[str]]:
    path = Path(f"data/processed/{split}/proof_state_techniques.parquet")
    if not path.exists():
        return {}
    rows = pd.read_parquet(path)
    if rows.empty:
        return {}
    rows["proof_state_id"] = rows["proof_state_id"].astype(str)
    rows["label"] = rows["label"].astype(str)
    return {proof_state_id: set(group["label"]) for proof_state_id, group in rows.groupby("proof_state_id")}


def _proof_state_label_coverage(split: str = "test") -> float:
    proof_states_path = Path(f"data/processed/{split}/proof_states.parquet")
    if not proof_states_path.exists():
        return 0.0
    total = pq.ParquetFile(proof_states_path).metadata.num_rows
    if total <= 0:
        return 0.0
    return len(_labels_by_proof_state(split)) / total


def _features_by_proof_state(split: str) -> pd.DataFrame:
    path = Path(f"data/processed/{split}/proof_state_features.parquet")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "id" in frame.columns:
        frame["id"] = frame["id"].astype(str)
        frame = frame.set_index("id", drop=False)
    return frame


def _strategy_retrieval_evaluation(k: int = 10) -> dict[str, Any]:
    try:
        train_ids = _embedding_ids("train", "ProofState")
        test_ids = _embedding_ids("test", "ProofState")
        train_x = _load_embedding("train", "proof_state")
        test_x = _load_embedding("test", "proof_state")
    except FileNotFoundError:
        return {"available": False, "reason": "missing_embedding_artifacts"}
    train_labels = _labels_by_proof_state("train")
    test_labels = _labels_by_proof_state("test")
    neighbor_idx, neighbor_scores = _topk_neighbors(test_x, train_x, k=k)
    label_recall_at_1 = []
    label_recall_at_3 = []
    label_recall_at_5 = []
    label_recall_at_10 = []
    any_hit_at_1 = []
    any_hit_at_3 = []
    any_hit_at_5 = []
    any_hit_at_10 = []
    for row_idx, proof_state_id in enumerate(test_ids):
        gold = test_labels.get(proof_state_id, set())
        if not gold:
            continue
        scored: dict[str, float] = {}
        for col_idx, train_row in enumerate(neighbor_idx[row_idx]):
            labels = train_labels.get(train_ids[int(train_row)], set())
            weight = max(float(neighbor_scores[row_idx, col_idx]), 0.0)
            for label in labels:
                scored[label] = scored.get(label, 0.0) + weight
        ranked = [label for label, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0]))]
        for cutoff, recalls, hits in [
            (1, label_recall_at_1, any_hit_at_1),
            (3, label_recall_at_3, any_hit_at_3),
            (5, label_recall_at_5, any_hit_at_5),
            (10, label_recall_at_10, any_hit_at_10),
        ]:
            pred = set(ranked[:cutoff])
            recalls.append(len(gold & pred) / len(gold))
            hits.append(float(bool(gold & pred)))
    evaluated = len(label_recall_at_1)
    return {
        "available": True,
        "task": "test_proof_state_to_train_proof_state_strategy_label_retrieval",
        "neighbor_k": k,
        "evaluated_queries": evaluated,
        "label_recall@1": float(np.mean(label_recall_at_1)) if evaluated else 0.0,
        "label_recall@3": float(np.mean(label_recall_at_3)) if evaluated else 0.0,
        "label_recall@5": float(np.mean(label_recall_at_5)) if evaluated else 0.0,
        "label_recall@10": float(np.mean(label_recall_at_10)) if evaluated else 0.0,
        "any_label_hit@1": float(np.mean(any_hit_at_1)) if evaluated else 0.0,
        "any_label_hit@3": float(np.mean(any_hit_at_3)) if evaluated else 0.0,
        "any_label_hit@5": float(np.mean(any_hit_at_5)) if evaluated else 0.0,
        "any_label_hit@10": float(np.mean(any_hit_at_10)) if evaluated else 0.0,
    }


def _difficulty_retrieval_evaluation(k: int = 10) -> dict[str, Any]:
    try:
        train_ids = _embedding_ids("train", "ProofState")
        test_ids = _embedding_ids("test", "ProofState")
        train_x = _load_embedding("train", "proof_state")
        test_x = _load_embedding("test", "proof_state")
    except FileNotFoundError:
        return {"available": False, "reason": "missing_embedding_artifacts"}
    train_features = _features_by_proof_state("train")
    test_features = _features_by_proof_state("test")
    if train_features.empty or test_features.empty:
        return {"available": False, "reason": "missing_difficulty_features"}
    neighbor_idx, neighbor_scores = _topk_neighbors(test_x, train_x, k=k)
    y_true = []
    y_pred = []
    bucket_hits = []
    for row_idx, proof_state_id in enumerate(test_ids):
        if proof_state_id not in test_features.index:
            continue
        neighbor_ids = [train_ids[int(i)] for i in neighbor_idx[row_idx]]
        neighbor_rows = train_features.reindex(neighbor_ids).dropna(subset=["theorem_complexity_score", "difficulty_bucket"])
        if neighbor_rows.empty:
            continue
        scores = np.maximum(neighbor_scores[row_idx, : len(neighbor_ids)], 0.0)
        scores = scores[[neighbor_id in neighbor_rows.index for neighbor_id in neighbor_ids]]
        if float(scores.sum()) <= 0:
            scores = np.ones(len(neighbor_rows), dtype=np.float32)
        pred_score = float(np.average(neighbor_rows["theorem_complexity_score"].astype(float).to_numpy(), weights=scores))
        bucket_votes: dict[str, float] = {}
        for bucket, weight in zip(neighbor_rows["difficulty_bucket"].astype(str), scores, strict=False):
            bucket_votes[bucket] = bucket_votes.get(bucket, 0.0) + float(weight)
        pred_bucket = max(bucket_votes.items(), key=lambda item: (item[1], item[0]))[0]
        test_row = test_features.loc[proof_state_id]
        y_true.append(float(test_row["theorem_complexity_score"]))
        y_pred.append(pred_score)
        bucket_hits.append(float(pred_bucket == str(test_row["difficulty_bucket"])))
    if not y_true:
        return {"available": True, "task": "test_proof_state_to_train_proof_state_difficulty_profile_retrieval", "evaluated_queries": 0}
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "available": True,
        "task": "test_proof_state_to_train_proof_state_difficulty_profile_retrieval",
        "neighbor_k": k,
        "evaluated_queries": int(len(true)),
        "retrieved_profile_mae": float(np.mean(np.abs(true - pred))),
        "retrieved_profile_rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "bucket_accuracy": float(np.mean(bucket_hits)),
        "mean_retrieved_score": float(pred.mean()),
        "mean_target_score": float(true.mean()),
    }


def _processed_counts() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test", "demo"]:
        split_counts: dict[str, int] = {}
        for name in ["theorems", "proof_states", "premises", "positive_edges", "negative_edges", "proof_state_features", "proof_state_techniques"]:
            path = Path(f"data/processed/{split}/{name}.parquet")
            if path.exists():
                split_counts[name] = int(pq.ParquetFile(path).metadata.num_rows)
        if split_counts:
            out[split] = split_counts
    return out


def _domain_counts() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        path = Path(f"data/processed/{split}/theorems.parquet")
        if not path.exists():
            continue
        rows = pd.read_parquet(path, columns=["domain_tag"])
        counts = rows["domain_tag"].fillna("Unknown").astype(str).value_counts()
        total = int(counts.sum())
        out[split] = [
            {"domain": domain, "theorems": int(count), "share": float(count / total) if total else 0.0}
            for domain, count in counts.items()
        ]
    return out


def _top_domains(domain_counts: dict[str, list[dict[str, Any]]], split: str = "test", n: int = 10) -> list[dict[str, Any]]:
    return sorted(domain_counts.get(split, []), key=lambda row: int(row["theorems"]), reverse=True)[:n]


def _theorem_domain_lookup(split: str = "test") -> dict[str, dict[str, str]]:
    path = Path(f"data/processed/{split}/theorems.parquet")
    if not path.exists():
        return {}
    rows = pd.read_parquet(path, columns=["full_name", "domain_tag", "subdomain_tag"])
    return {
        str(row["full_name"]): {
            "domain_tag": str(row.get("domain_tag") or ""),
            "subdomain_tag": str(row.get("subdomain_tag") or ""),
        }
        for row in rows.to_dict(orient="records")
    }


def _sample_guidance_cases(limit: int = 3) -> list[dict[str, Any]]:
    cases = read_json("outputs/reports/theorem_retrieval_case_studies.json", [])
    if not isinstance(cases, list):
        return []
    compact = []
    domain_lookup = _theorem_domain_lookup("test")
    preferred_domains = {
        "CategoryTheory",
        "Analysis",
        "Topology",
        "Algebra",
        "LinearAlgebra",
        "MeasureTheory",
        "RingTheory",
        "NumberTheory",
        "Geometry",
        "GroupTheory",
        "Probability",
        "Order",
        "FieldTheory",
    }

    def priority(row: dict[str, Any]) -> tuple[int, float, int, str]:
        name = str(row.get("full_name") or row.get("theorem_id") or "")
        domain = str(row.get("domain_tag") or domain_lookup.get(name, {}).get("domain_tag") or "")
        return (
            0 if domain in preferred_domains else 1,
            -float(row.get("gold_premise_train_coverage") or 0.0),
            -int(row.get("gold_positive_premise_count") or 0),
            name,
        )

    ranked_cases = sorted(
        cases,
        key=priority,
    )
    for case in ranked_cases[:limit]:
        guidance = case.get("guidance", {}) if isinstance(case, dict) else {}
        query = guidance.get("query", {}) if isinstance(guidance, dict) else {}
        query_text = "\n".join([str(query.get("full_name") or ""), str(query.get("goal_text") or ""), str(query.get("retrieval_text") or "")])
        theorem_name = str(case.get("full_name") or case.get("theorem_id") or "")
        domain_info = domain_lookup.get(theorem_name, {})
        ranked_premises = [
            row for row in (guidance.get("ranked_premises") or [])
            if isinstance(row, dict) and str(row.get("full_name") or "") != theorem_name
        ]
        compact.append(
            {
                "theorem": theorem_name,
                "split": case.get("split", ""),
                "domain": case.get("domain_tag") or domain_info.get("domain_tag", ""),
                "subdomain": case.get("subdomain_tag") or domain_info.get("subdomain_tag", ""),
                "goal_text": str(query.get("goal_text") or "")[:300],
                "gold_positive_premise_count": case.get("gold_positive_premise_count"),
                "gold_premise_train_coverage": case.get("gold_premise_train_coverage"),
                "top_premises": [
                    {
                        "full_name": row.get("full_name"),
                        "score": row.get("score"),
                        "reason": "; ".join((row.get("ranking_reasons") or [])[:2]) or str(row.get("explanation") or "")[:140],
                    }
                    for row in ranked_premises[:5]
                ],
                "similar_theorems": [
                    {
                        "full_name": row.get("full_name"),
                        "score": row.get("score"),
                    }
                    for row in (guidance.get("similar_theorems") or [])[:5]
                ],
                "similar_proof_states": [
                    {
                        "full_name": row.get("full_name"),
                        "score": row.get("score"),
                        "goal_text": str(row.get("goal_text") or "")[:140],
                    }
                    for row in (guidance.get("similar_proof_states") or [])[:4]
                ],
                "techniques": labels_for_text(query_text, max_labels=5),
                "difficulty": guidance.get("difficulty_profile", {}),
            }
        )
    return compact


def _case_overview(case: dict[str, Any]) -> str:
    theorem = str(case.get("theorem") or "")
    goal = str(case.get("goal_text") or "")
    premises = " ".join(str(row.get("full_name") or "") for row in case.get("top_premises", []))
    neighbors = " ".join(str(row.get("full_name") or "") for row in case.get("similar_theorems", []))
    text = " ".join([theorem, goal, premises, neighbors])
    if "≫" in text or "CategoryTheory" in text or ".hom" in text or "hom_" in text:
        return (
            "This query is a categorical/action morphism equality. The important proof shape is not a numeric computation; "
            "it is an equality between composed morphisms, so useful guidance should point toward category structure, hom/extensionality lemmas, "
            "and nearby commutative-diagram style proof states."
        )
    if "lookup" in text or "dlookup" in text or "toAList" in text or "AList" in text:
        return (
            "This query is a data-structure lookup equality. The useful proof context is concentrated around list/AList conversion, "
            "lookup/dlookup lemmas, and historical proofs that normalize finite-map or association-list representations."
        )
    if "Continuous" in text or "continuous" in text or "𝓝" in text or "Filter" in text:
        return (
            "This query is a topology/continuity theorem. The useful proof context is about transporting continuity through a specific equivalence or map, "
            "so the retrieved neighbors should surface continuity, neighborhood/filter, and map-composition proof patterns."
        )
    if "Simplex" in text or "affineCombination" in text or "interior" in text or "Affine" in text:
        return (
            "This query is an affine-geometry theorem. The useful proof context is about affine combinations and interior membership, "
            "so relevant neighbors should involve affine independence, convex/affine coordinates, and geometric membership conditions."
        )
    if "Etale" in text or "algEquiv" in text or "Algebra." in text:
        return (
            "This query is an algebra/ring-theory theorem about structural equivalences. The useful proof context should emphasize algebra maps, "
            "equivalences, product decompositions, and transport of algebraic properties."
        )
    return (
        "This query illustrates the general theorem-guidance workflow: retrieve candidate premises, inspect nearby historical theorem/proof-state "
        "patterns, then use strategy facets and difficulty as calibration rather than as a generated proof."
    )


def _case_takeaway(case: dict[str, Any]) -> str:
    facets = [str(row.get("label")) for row in case.get("techniques", [])[:3]]
    premise_names = [str(row.get("full_name")) for row in case.get("top_premises", [])[:3]]
    theorem_names = [str(row.get("full_name")) for row in case.get("similar_theorems", [])[:2]]
    difficulty = case.get("difficulty") or {}
    return (
        f"Takeaway: the user would first inspect `{', '.join(premise_names)}` as candidate dependencies, "
        f"then compare the query against historical neighbors such as `{', '.join(theorem_names)}`. "
        f"The top strategy facets (`{', '.join(facets)}`) summarize the likely proof mode, while the difficulty profile "
        f"(`{difficulty.get('difficulty_bucket', 'n/a')}` / `{_fmt(difficulty.get('difficulty_score'))}`) indicates that this case is expected to be relatively lightweight in the current corpus."
    )


def _case_study_markdown(cases: list[dict[str, Any]]) -> list[str]:
    if not cases:
        return []
    lines = ["## Case Studies", ""]
    for idx, case in enumerate(cases[:2], start=1):
        difficulty = case.get("difficulty") or {}
        lines.extend(
            [
                f"### Case {idx}: `{case.get('theorem', 'unknown')}`",
                "",
                "This is a held-out theorem-guidance example showing how the retrieval bundle is meant to be used: inspect candidate premises, compare nearby historical theorems/proof states, read strategy facets, and use the difficulty profile as calibration.",
                "",
                _case_overview(case),
                "",
                _table(
                    ["Field", "Value"],
                    [
                        ["Split", case.get("split", "n/a")],
                        ["Domain", f"{case.get('domain', 'n/a')} / {case.get('subdomain', 'n/a')}"],
                        ["Goal/query text", str(case.get("goal_text") or "n/a").replace("\n", " ")],
                        ["Gold premise train coverage", _fmt(case.get("gold_premise_train_coverage"))],
                        ["Gold positive premise count", case.get("gold_positive_premise_count", "n/a")],
                        ["Difficulty", f"{difficulty.get('difficulty_bucket', 'n/a')} / {_fmt(difficulty.get('difficulty_score'))}"],
                    ],
                ),
                "",
                "**Retrieved premises.**",
                "",
                _table(
                    ["Rank", "Premise", "Score", "Why it appears"],
                    [
                        [rank, row.get("full_name", "n/a"), _fmt(row.get("score")), row.get("reason", "")]
                        for rank, row in enumerate(case.get("top_premises", [])[:5], start=1)
                    ],
                ),
                "",
                "How to read this table: these are candidate dependencies a user would inspect first. The score combines embedding similarity and reranker signals; the reason column shows why the system surfaced each premise.",
                "",
                "**Historical proof neighbors.**",
                "",
                _table(
                    ["Rank", "Similar theorem", "Score"],
                    [
                        [rank, row.get("full_name", "n/a"), _fmt(row.get("score"))]
                        for rank, row in enumerate(case.get("similar_theorems", [])[:4], start=1)
                    ],
                ),
                "",
                _table(
                    ["Rank", "Similar proof state", "Score", "Neighbor goal"],
                    [
                        [rank, row.get("full_name", "n/a"), _fmt(row.get("score")), str(row.get("goal_text", "")).replace("\n", " ")]
                        for rank, row in enumerate(case.get("similar_proof_states", [])[:3], start=1)
                    ],
                ),
                "",
                "How to read these neighbors: similar theorems give theorem-level proof templates, while similar proof states show local goal shapes that appeared in historical proofs.",
                "",
                "**Strategy facets.**",
                "",
                _table(
                    ["Rank", "Facet", "Confidence", "Evidence"],
                    [
                        [rank, row.get("label", "n/a"), _fmt(row.get("confidence")), row.get("provenance", "")]
                        for rank, row in enumerate(case.get("techniques", [])[:5], start=1)
                    ],
                ),
                "",
                _case_takeaway(case),
                "",
            ]
        )
    return lines


def _prediction_bundle() -> dict[str, Any]:
    metrics = read_json("outputs/reports/metrics.json", {})
    test_eval = read_json("outputs/reports/test_set_evaluation.json", {})
    ranker = read_json("outputs/reports/ranker_validation_metrics.json", {})
    difficulty = read_json("outputs/reports/difficulty_estimator_metrics.json", {})
    difficulty_target = read_json("outputs/reports/difficulty_target_report.json", {})
    graph = read_json("outputs/reports/graph_stats_summary.json", {})
    benchmark = read_json("outputs/reports/index_benchmark.json", {})
    corpus = read_json("outputs/reports/corpus_manifest.json", {})
    split_leakage = read_json("outputs/reports/split_leakage_report.json", {})
    return {
        "corpus": corpus,
        "split_policy": split_leakage,
        "processed_data": {
            "root": "data/processed",
            "splits": _processed_counts(),
            "domain_counts": _domain_counts(),
        },
        "prediction_artifacts": {
            "embeddings": "outputs/embeddings",
            "indexes": "outputs/indexes",
            "models": ["outputs/models/premise_ranker.joblib", "outputs/models/difficulty_estimator.joblib"],
            "reports": "outputs/reports",
        },
        "premise_prediction": {
            "proof_state": {
                "evaluated_queries": metrics.get("test_proof_state_evaluated_queries"),
                "Recall@1": metrics.get("Recall@1"),
                "Recall@5": metrics.get("Recall@5"),
                "Recall@10": metrics.get("Recall@10"),
                "Recall@100": metrics.get("Recall@100"),
                "MRR": metrics.get("MRR"),
                "MAP": metrics.get("MAP"),
                "nDCG@10": metrics.get("nDCG@10"),
                "gold_premise_coverage": metrics.get("gold_premise_coverage"),
            },
            "theorem": {
                "evaluated_theorems": metrics.get("theorem_retrieval_evaluated_theorems"),
                "Recall@1": metrics.get("theorem_retrieval_Recall@1"),
                "Recall@5": metrics.get("theorem_retrieval_Recall@5"),
                "Recall@10": metrics.get("theorem_retrieval_Recall@10"),
                "Recall@100": metrics.get("theorem_retrieval_Recall@100"),
                "MRR": metrics.get("theorem_retrieval_MRR"),
                "MAP": metrics.get("theorem_retrieval_MAP"),
                "nDCG@10": metrics.get("theorem_retrieval_nDCG@10"),
                "gold_premise_coverage": metrics.get("theorem_retrieval_gold_premise_coverage"),
            },
            "reranker": {
                "validation_auc": ranker.get("validation_auc") or metrics.get("AUC"),
                "feature_ablation": ranker.get("feature_ablation", {}).get("groups", {}),
                "hybrid_reranked_Recall@10": metrics.get("hybrid_reranked_proof_state_Recall@10"),
                "reranked_Recall@10": metrics.get("reranked_proof_state_Recall@10"),
            },
            "failure_diagnosis": test_eval.get("test", {}),
        },
        "proof_pattern_prediction": {
            "similar_theorem_metric_proxy": {
                "theorem_Recall@10": metrics.get("theorem_retrieval_Recall@10"),
                "theorem_MRR": metrics.get("theorem_retrieval_MRR"),
            },
            "graph_stats": graph,
            "index_benchmark": benchmark.get("entities", {}),
        },
        "proof_strategy_hinting": {
            "method": "retrieve similar train proof states and aggregate their curated strategy facets; query rule facets are retained as evidence when available",
            "label_distribution": _csv_records("outputs/reports/proof_technique_distribution.csv"),
            "candidate_pool": read_json("outputs/reports/proof_technique_candidate_pool.json", []),
            "label_coverage": _proof_state_label_coverage("test"),
            "retrieval_evaluation": _strategy_retrieval_evaluation(),
        },
        "difficulty_prediction": {
            "method": "retrieve historical difficulty profiles from similar train proof states and calibrate them against relative complexity buckets",
            "distribution": _csv_records("outputs/reports/difficulty_distribution.csv"),
            "target_report": difficulty_target,
            "estimator_metrics": difficulty,
            "retrieval_evaluation": _difficulty_retrieval_evaluation(),
        },
        "sample_prediction_cases": _sample_guidance_cases(),
    }


def _write_markdown(bundle: dict[str, Any], output_path: str | Path) -> None:
    metrics = bundle["premise_prediction"]
    thm = metrics["theorem"]
    reranker = metrics["reranker"]
    diff_metrics = bundle["difficulty_prediction"]["estimator_metrics"]
    graph_train = bundle["proof_pattern_prediction"]["graph_stats"].get("train", {})
    index_entities = bundle["proof_pattern_prediction"]["index_benchmark"]
    technique_rows = _top_by_split(bundle["proof_strategy_hinting"]["label_distribution"], "test", "count", n=10)
    strategy_retrieval = bundle["proof_strategy_hinting"].get("retrieval_evaluation", {})
    difficulty_rows = bundle["difficulty_prediction"]["distribution"]
    difficulty_retrieval = bundle["difficulty_prediction"].get("retrieval_evaluation", {})
    processed = bundle["processed_data"]["splits"]
    domain_counts = bundle["processed_data"].get("domain_counts", {})
    corpus = bundle.get("corpus", {})
    split_policy = bundle.get("split_policy", {})

    md = [
        "# ProofAtlas Research Report",
        "",
        "## Research Framing",
        "",
        "ProofAtlas is framed as a research dataset and retrieval study for LeanRank-style formal proof guidance. The deliverable is not a production proof assistant; it is a processed theorem/proof-state/premise dataset plus retrieval-grounded artifacts for theorem-level premise retrieval, proof-pattern retrieval, strategy retrieval, and difficulty-profile retrieval.",
        "",
        "The four tasks form a layered retrieval story. Theorem-level premise retrieval is the primary supervised benchmark. Proof-pattern retrieval supplies evidence through similar theorem and similar proof-state neighborhoods. Strategy retrieval reads strategy facets from retrieved proof-state neighbors. Difficulty retrieval reads historical difficulty profiles from similar proof states and calibrates them into a score and bucket.",
        "",
        _table(
            ["Module", "Research framing", "Primary evidence"],
            [
                ["Premise prediction", "Premise retrieval and reranking", "Held-out theorem to train-premise ranking against LeanRank positive premises."],
                ["Proof pattern prediction", "Similar theorem and similar proof-state retrieval", "Embedding neighbors plus KG neighborhoods used as reusable proof-pattern evidence."],
                ["Proof strategy hinting", "Retrieve strategy facets from similar proof states", "Neighbor facets aggregated from a curated tactic/mathematical-operation taxonomy."],
                ["Difficulty prediction", "Retrieve historical difficulty profiles from similar theorem/proof-state neighborhoods", "Neighbor difficulty scores calibrated into easy/medium/hard buckets."],
            ],
        ),
        "",
        "## Local Deliverables",
        "",
        _table(
            ["Artifact", "Path"],
            [
                ["Processed dataset", "`data/processed/{train,val,test,demo}`"],
                ["Retrieval summary", "`outputs/predictions/research_prediction_results.json`"],
                ["Research report", "`outputs/reports/research_report.md`"],
                ["Embeddings and indexes", "`outputs/embeddings`, `outputs/indexes`"],
                ["Learned models", "`outputs/models/premise_ranker.joblib`, `outputs/models/difficulty_estimator.joblib`"],
            ],
        ),
        "",
        "## Dataset",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Source", corpus.get("dataset_name", "n/a")],
                ["Source kind", corpus.get("source_kind", "n/a")],
                ["Sample unit", (corpus.get("sample_plan") or {}).get("unit", "n/a")],
                ["Sampled theorems", corpus.get("sampled_theorems", "n/a")],
                ["Sampled rows", corpus.get("sampled_rows", "n/a")],
                ["Random seed", corpus.get("random_seed", "n/a")],
                ["Config hash", corpus.get("config_hash", "n/a")],
            ],
        ),
        "",
        "The processed dataset contains theorem-level, proof-state-level, premise-level, positive-premise, negative-candidate, strategy-facet, difficulty-feature, embedding, index, and KG artifacts. The split is theorem-disjoint: held-out theorem names do not appear in train, so retrieval is evaluated against unseen theorems while using train premises and train proof states as the historical retrieval corpus.",
        "",
        _table(
            ["Split policy", "Value"],
            [
                [
                    "Train/val/test theorem counts",
                    ", ".join(f"{split}={count}" for split, count in sorted((split_policy.get("theorem_counts") or {}).items())) or "n/a",
                ],
                ["Theorem leakage detected", split_policy.get("has_leakage", "n/a")],
            ],
        ),
        "",
        "### Split Statistics",
        "",
        _table(
            ["Split", "Theorems", "Proof states", "Premises", "Positive edges", "Negative edges"],
            [
                [
                    split,
                    counts.get("theorems", 0),
                    counts.get("proof_states", 0),
                    counts.get("premises", 0),
                    counts.get("positive_edges", 0),
                    counts.get("negative_edges", 0),
                ]
                for split, counts in processed.items()
            ],
        ),
        "",
        "### Domain Statistics",
        "",
        _table(
            ["Test domain", "Theorems", "Share"],
            [[row["domain"], row["theorems"], _fmt(row["share"])] for row in _top_domains(domain_counts, "test", n=12)],
        ),
        "",
        "## Evaluation Metrics",
        "",
        _table(
            ["Metric", "Meaning"],
            [
                ["Recall@k", "Fraction of retrievable gold items recovered in the top-k retrieved results."],
                ["MRR", "Mean reciprocal rank of the first retrieved gold item."],
                ["MAP", "Mean average precision over ranked retrieved items."],
                ["nDCG@k", "Rank-sensitive gain that rewards placing gold items earlier in the top-k list."],
                ["AUC", "Validation discrimination of the learned premise reranker over positive and hard-negative premise pairs."],
                ["Label Recall@k", "Average fraction of a query proof state's strategy facets recovered by the top-k aggregated retrieved facets."],
                ["Any-label Hit@k", "Fraction of labeled proof states for which at least one strategy facet is recovered in the top-k facets."],
                ["MAE/RMSE", "Absolute and squared-error summaries for retrieved difficulty-profile scores."],
                ["Bucket accuracy", "Agreement between retrieved difficulty bucket and the query proof state's relative difficulty bucket."],
            ],
        ),
        "",
        "## 1. Theorem-Level Premise Retrieval",
        "",
        _task_spec_table(
            [
                ("Task definition", "Retrieve premises that are useful for proving a held-out theorem."),
                ("Input", "A test theorem represented by its theorem-level text embedding."),
                ("Retrieval corpus", "Train-split premise embeddings and metadata."),
                ("Output", "A ranked list of premise IDs/names with retrieval scores."),
                ("Evaluation target", "Positive LeanRank premises used by any proof state of the held-out theorem, restricted to gold premises present in the train premise index."),
                ("Metrics", "Recall@1/5/10/100, MRR, MAP, and nDCG@10."),
                ("Role in report", "Headline premise-retrieval benchmark for theorem guidance."),
            ]
        ),
        "",
        _table(
            ["Task", "Queries", "Recall@1", "Recall@5", "Recall@10", "Recall@100", "MRR", "MAP", "nDCG@10"],
            [
                [
                    "Theorem-level premise retrieval",
                    thm.get("evaluated_theorems"),
                    _fmt(thm.get("Recall@1")),
                    _fmt(thm.get("Recall@5")),
                    _fmt(thm.get("Recall@10")),
                    _fmt(thm.get("Recall@100")),
                    _fmt(thm.get("MRR")),
                    _fmt(thm.get("MAP")),
                    _fmt(thm.get("nDCG@10")),
                ],
            ],
        ),
        "",
        f"The learned premise reranker reaches validation AUC `{_fmt(reranker.get('validation_auc'))}` over positive and hard-negative premise pairs.",
        "",
        "## 2. Proof Pattern Retrieval",
        "",
        _task_spec_table(
            [
                ("Task definition", "Retrieve historical neighbors that provide proof-pattern evidence for a query theorem or proof state."),
                ("Input", "A theorem embedding for theorem-neighbor retrieval, or a proof-state embedding for local proof-state-neighbor retrieval."),
                ("Retrieval corpus", "Train theorem embeddings, train proof-state embeddings, and the enriched proof KG."),
                ("Output", "Similar theorems, similar proof states, and graph-neighborhood evidence such as similar_to_theorem and premise/strategy edges."),
                ("Evaluation target", "The dataset does not include a separate proof-pattern-neighbor benchmark; theorem-neighbor quality is reflected through theorem-level premise retrieval, while proof-state-neighbor utility is evaluated through the strategy and difficulty tasks below."),
                ("Metrics", "Theorem retrieval Recall/MRR proxy plus HNSW index recall@10 versus exact cosine for theorem/proof-state/premise indexes."),
                ("Role in report", "Evidence layer that supports strategy-facet retrieval, difficulty-profile retrieval, and interpretability."),
            ]
        ),
        "",
        _table(
            ["Pattern signal", "Value"],
            [
                ["Theorem retrieval Recall@10", _fmt(thm.get("Recall@10"))],
                ["Theorem retrieval MRR", _fmt(thm.get("MRR"))],
                ["Train KG nodes", graph_train.get("node_count", "n/a")],
                ["Train KG edges", graph_train.get("edge_count", "n/a")],
                ["Train similar_to_theorem edges", graph_train.get("edge_counts_by_type", {}).get("similar_to_theorem", "n/a")],
            ],
        ),
        "",
        _table(
            ["Indexed entity", "Rows", "Backend", "Indexed ms/query", "Recall@10 vs exact"],
            [
                [
                    name,
                    entity.get("rows"),
                    entity.get("backend"),
                    _fmt(entity.get("indexed_ms_per_query")),
                    _fmt(entity.get("recall_at_10_vs_exact")),
                ]
                for name, entity in index_entities.items()
            ],
        ),
        "",
        "## 3. Strategy Retrieval",
        "",
        _task_spec_table(
            [
                ("Task definition", "Retrieve likely proof-strategy facets for a query proof state."),
                ("Input", "A test proof-state embedding."),
                ("Retrieval corpus", "Train proof-state embeddings whose proof states are annotated with curated strategy facets."),
                ("Output", "A ranked set of strategy facets, e.g. rewrite_transport, order_inequality_reasoning, algebraic_computation, typeclass_instance_resolution, case_analysis, and set_membership_reasoning."),
                ("Evaluation target", "The query proof state's strategy facets from the same taxonomy, used to measure whether retrieved neighbors recover the same proof-mode signals."),
                ("Metrics", "Label Recall@1/3/5 and Any-label Hit@1/3."),
                ("Role in report", "Auxiliary guidance task showing whether local proof-state neighbors recover useful strategy facets."),
            ]
        ),
        "",
        _table(
            ["Strategy retrieval metric", "Value"],
            [
                ["Evaluated test proof states", strategy_retrieval.get("evaluated_queries", "n/a")],
                ["Label Recall@1", _fmt(strategy_retrieval.get("label_recall@1"))],
                ["Label Recall@3", _fmt(strategy_retrieval.get("label_recall@3"))],
                ["Label Recall@5", _fmt(strategy_retrieval.get("label_recall@5"))],
                ["Any-label Hit@1", _fmt(strategy_retrieval.get("any_label_hit@1"))],
                ["Any-label Hit@3", _fmt(strategy_retrieval.get("any_label_hit@3"))],
            ],
        ),
        "",
        _table(
            ["Strategy facet", "Test count"],
            [[row.get("label"), row.get("count")] for row in technique_rows],
        ),
        "",
        f"Strategy-facet coverage is `{_fmt(bundle['proof_strategy_hinting'].get('label_coverage'))}`. These facets are curated retrieval annotations inferred from goal shape, context markers, theorem names, and statement symbols.",
        "",
        "## 4. Difficulty Retrieval",
        "",
        _task_spec_table(
            [
                ("Task definition", "Retrieve historical difficulty profiles for a query proof state and summarize them as a relative complexity score/bucket."),
                ("Input", "A test proof-state embedding."),
                ("Retrieval corpus", "Train proof-state embeddings with precomputed difficulty features and relative difficulty buckets."),
                ("Output", "A retrieved-neighbor difficulty score, easy/medium/hard bucket, and calibration diagnostics."),
                ("Evaluation target", "The query proof state's relative complexity proxy, derived from proof length, tactic index, positive-premise count, namespace rarity, and negative-candidate hardness."),
                ("Metrics", "Retrieved-profile MAE/RMSE, bucket accuracy, mean retrieved score, and mean target score."),
                ("Role in report", "Auxiliary retrieval task for estimating whether a query resembles historically easier or harder proof states."),
            ]
        ),
        "",
        "Difficulty buckets use a split-local distribution policy: easy is the lower 50%, medium is the next 35%, and hard is the top 15%.",
        "",
        _table(
            ["Difficulty retrieval metric", "Value"],
            [
                ["Evaluated test proof states", difficulty_retrieval.get("evaluated_queries", "n/a")],
                ["Retrieved-profile MAE", _fmt(difficulty_retrieval.get("retrieved_profile_mae"))],
                ["Retrieved-profile RMSE", _fmt(difficulty_retrieval.get("retrieved_profile_rmse"))],
                ["Bucket accuracy", _fmt(difficulty_retrieval.get("bucket_accuracy"))],
                ["Mean retrieved score", _fmt(difficulty_retrieval.get("mean_retrieved_score"))],
                ["Mean target score", _fmt(difficulty_retrieval.get("mean_target_score"))],
            ],
        ),
        "",
        _table(
            ["Split", "Bucket", "Count"],
            [[row.get("split"), row.get("bucket"), row.get("count")] for row in difficulty_rows],
        ),
        "",
        _table(
            ["Split", "Rows", "MAE", "R2", "Mean pred", "Mean target"],
            [
                [
                    split,
                    section.get("rows"),
                    _fmt(section.get("mae")),
                    _fmt(section.get("r2")),
                    _fmt(section.get("mean_prediction")),
                    _fmt(section.get("mean_target")),
                ]
                for split, section in diff_metrics.items()
                if isinstance(section, dict) and section.get("available")
            ],
        ),
        "",
        *_case_study_markdown(bundle.get("sample_prediction_cases", [])),
        "## Interpretation",
        "",
        "The dataset and report support a retrieval-centered research claim. Theorem-level premise retrieval is the headline benchmark. Proof-pattern retrieval is the evidence layer: theorem neighbors support premise retrieval, and proof-state neighbors support strategy-facet and difficulty-profile retrieval. Strategy and difficulty are therefore not separate standalone prediction claims; they are retrieval-grounded guidance tasks built on historical proof-state neighbors. The current theorem-disjoint train/val/test split has no theorem leakage; future split changes should be motivated by domain balance or retrieval coverage.",
        "",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(md), encoding="utf-8")


def run(config_path: str = "configs/proofatlas.yaml", output_path: str | Path = REPORT_PATH) -> dict[str, Any]:
    del config_path
    bundle = _prediction_bundle()
    write_json(PREDICTION_PATH, bundle)
    _write_markdown(bundle, output_path)
    return {"report_path": str(output_path), "prediction_path": str(PREDICTION_PATH)}
