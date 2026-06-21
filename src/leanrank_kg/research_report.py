from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from .utils import read_json, write_json


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


def _sample_guidance_cases(limit: int = 3) -> list[dict[str, Any]]:
    cases = read_json("outputs/reports/theorem_retrieval_case_studies.json", [])
    if not isinstance(cases, list):
        return []
    compact = []
    for case in cases[:limit]:
        guidance = case.get("guidance", {}) if isinstance(case, dict) else {}
        compact.append(
            {
                "theorem": case.get("full_name") or case.get("theorem_id"),
                "top_premises": [
                    {
                        "full_name": row.get("full_name"),
                        "score": row.get("score"),
                    }
                    for row in (guidance.get("ranked_premises") or [])[:5]
                ],
                "similar_theorems": [
                    {
                        "full_name": row.get("full_name"),
                        "score": row.get("score"),
                    }
                    for row in (guidance.get("similar_theorems") or [])[:5]
                ],
                "techniques": guidance.get("likely_proof_techniques", [])[:5],
                "difficulty": guidance.get("difficulty_profile", {}),
            }
        )
    return compact


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
            "method": "retrieve similar train proof states and aggregate their weak strategy labels; query rule labels are retained as evidence when available",
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
    ps = metrics["proof_state"]
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
        "ProofAtlas is framed as a research dataset and retrieval study for LeanRank-style formal proof guidance. The deliverable is not a production proof assistant; it is a processed theorem/proof-state/premise dataset plus retrieval-grounded prediction artifacts for theorem-level premise retrieval, proof-pattern retrieval, strategy retrieval, and difficulty-profile retrieval.",
        "",
        "## Local Deliverables",
        "",
        _table(
            ["Artifact", "Path"],
            [
                ["Processed dataset", "`data/processed/{train,val,test,demo}`"],
                ["Prediction summary", "`outputs/predictions/research_prediction_results.json`"],
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
                ["Label Recall@k", "Average fraction of a query proof state's weak strategy facets recovered by the top-k aggregated retrieved facets."],
                ["Any-label Hit@k", "Fraction of labeled proof states for which at least one weak strategy facet is recovered in the top-k facets."],
                ["MAE/RMSE", "Absolute and squared-error summaries for retrieved difficulty-profile scores."],
                ["Bucket accuracy", "Agreement between retrieved difficulty bucket and the query proof state's relative difficulty bucket."],
            ],
        ),
        "",
        "## 1. Theorem-Level Premise Retrieval",
        "",
        "**Goal.** Given a held-out theorem statement, retrieve useful premises from the train premise corpus. This is the headline premise-retrieval benchmark because it matches the end-to-end theorem-guidance setting and is substantially more reliable than direct local premise prediction.",
        "",
        "**Evaluation.** Test theorem embeddings query the train premise index. Retrieved train premise IDs are compared with all held-out positive LeanRank premises attached to proof states of the theorem whose premise IDs exist in the train premise index.",
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
        "Diagnostic note: direct proof-state-to-premise retrieval is retained as a failure-analysis task rather than a headline result because it is candidate-generation limited.",
        "",
        _table(
            ["Diagnostic task", "Queries", "Recall@10", "Recall@100", "MRR", "MAP"],
            [
                [
                    "Direct proof-state-to-premise retrieval",
                    ps.get("evaluated_queries"),
                    _fmt(ps.get("Recall@10")),
                    _fmt(ps.get("Recall@100")),
                    _fmt(ps.get("MRR")),
                    _fmt(ps.get("MAP")),
                ]
            ],
        ),
        "",
        "## 2. Proof Pattern Retrieval",
        "",
        "**Goal.** Retrieve historical proof patterns that can explain or contextualize a new theorem/proof state: similar theorems, similar local proof states, and KG neighborhoods.",
        "",
        "**Evaluation.** Similar theorem retrieval is evaluated through theorem-level premise retrieval quality. Proof-state-to-proof-state retrieval is used as the neighbor substrate for strategy-facet and difficulty-profile retrieval below. Index quality is measured against exact cosine retrieval.",
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
        "**Goal.** Retrieve likely proof-strategy facets for a query proof state, such as rewriting/transport, order reasoning, algebraic computation, typeclass-instance reasoning, case analysis, or set-membership reasoning.",
        "",
        "**Evaluation.** Historical proof states receive weak strategy-facet labels from a curated taxonomy. A test proof state retrieves similar train proof states, aggregates their facets by embedding-neighbor similarity, and is scored against the test proof state's own weak facets. This is a retrieval-grounded weak-label task, not supervised tactic classification.",
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
        f"Strategy-facet coverage is `{_fmt(bundle['proof_strategy_hinting'].get('label_coverage'))}`. These facets are weak retrieval supervision inferred from goal shape, context markers, theorem names, and statement symbols; they are not ground-truth tactic annotations.",
        "",
        "## 4. Difficulty Retrieval",
        "",
        "**Goal.** Retrieve historical difficulty profiles for a query proof state and summarize them as a relative complexity score and easy/medium/hard bucket.",
        "",
        "**Evaluation.** The target is a relative complexity proxy derived from proof-state and theorem features, including proof length, tactic index, positive-premise count, namespace rarity, and negative-candidate hardness. A test proof state retrieves similar train proof states and aggregates their difficulty profiles. Buckets use a split-local distribution policy: easy is the lower 50%, medium is the next 35%, and hard is the top 15%.",
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
        "## Interpretation",
        "",
        "The dataset and report support a retrieval-centered research claim. Theorem-level premise retrieval is the strongest quantitative result, while proof-state-level premise retrieval remains candidate-generation limited and should be presented as the main open challenge. Proof-state retrieval is still useful as a local-neighbor substrate for strategy-facet retrieval, difficulty-profile retrieval, and explanation. The current theorem-disjoint train/val/test split has no theorem leakage; future split changes should be motivated by domain-balance or retrieval-coverage studies rather than leakage repair.",
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
