from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

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
    return {
        "processed_data": {
            "root": "data/processed",
            "splits": _processed_counts(),
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
            "method": "query rule labels plus embedding-neighbor aggregation from retrieved similar proof states",
            "label_distribution": _csv_records("outputs/reports/proof_technique_distribution.csv"),
            "candidate_pool": read_json("outputs/reports/proof_technique_candidate_pool.json", []),
            "label_coverage": metrics.get("proof_technique_label_coverage"),
        },
        "difficulty_prediction": {
            "method": "relative complexity score from proof length, tactic index, premise counts, namespace rarity, and negative-candidate hardness",
            "distribution": _csv_records("outputs/reports/difficulty_distribution.csv"),
            "target_report": difficulty_target,
            "estimator_metrics": difficulty,
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
    difficulty_rows = bundle["difficulty_prediction"]["distribution"]
    processed = bundle["processed_data"]["splits"]

    md = [
        "# ProofAtlas Research Report",
        "",
        "## Research Framing",
        "",
        "ProofAtlas is framed as a research dataset and retrieval study for LeanRank-style formal proof guidance. The deliverable is not a production proof assistant; it is a processed theorem/proof-state/premise dataset plus prediction artifacts for premise retrieval, proof-pattern retrieval, proof-strategy hinting, and difficulty estimation.",
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
        "## Dataset Snapshot",
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
        "## 1. Premise Prediction",
        "",
        _table(
            ["Task", "Queries", "Recall@1", "Recall@5", "Recall@10", "Recall@100", "MRR", "MAP", "nDCG@10"],
            [
                [
                    "Proof-state premise retrieval",
                    ps.get("evaluated_queries"),
                    _fmt(ps.get("Recall@1")),
                    _fmt(ps.get("Recall@5")),
                    _fmt(ps.get("Recall@10")),
                    _fmt(ps.get("Recall@100")),
                    _fmt(ps.get("MRR")),
                    _fmt(ps.get("MAP")),
                    _fmt(ps.get("nDCG@10")),
                ],
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
        f"The learned premise reranker reaches validation AUC `{_fmt(reranker.get('validation_auc'))}`. On the small full-rerank diagnostic sample, reranked proof-state Recall@10 is `{_fmt(reranker.get('reranked_Recall@10'))}` and hybrid candidate reranking reaches `{_fmt(reranker.get('hybrid_reranked_Recall@10'))}`.",
        "",
        "## 2. Proof Pattern Prediction",
        "",
        "Proof-pattern prediction is represented by similar theorem retrieval, similar proof-state retrieval, and graph-neighborhood evidence rather than a discrete classifier.",
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
        "## 3. Proof Strategy Hinting",
        "",
        "Proof strategy hinting now combines deterministic rule labels with embedding-similarity evidence from retrieved similar proof states. This is intentionally reported as weak strategy hinting, not as a supervised strategy classifier.",
        "",
        _table(
            ["Technique label", "Test count"],
            [[row.get("label"), row.get("count")] for row in technique_rows],
        ),
        "",
        f"Proof-technique label coverage is `{_fmt(bundle['proof_strategy_hinting'].get('label_coverage'))}`. The labels are used as interpretable guidance and as a reranker feature; the ablation shows they are auxiliary rather than the main retrieval signal.",
        "",
        "## 4. Difficulty Prediction",
        "",
        "Difficulty is a relative research proxy derived from proof-state and theorem complexity signals. Buckets use a split-local distribution policy: easy is the lower 50%, medium is the next 35%, and hard is the top 15%.",
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
        "The strongest quantitative result is theorem-level premise retrieval. Proof-state-level premise retrieval is harder and remains candidate-generation limited, so it should be presented as a baseline and diagnostic target rather than as a solved proof-step predictor. Proof strategies and difficulty are research-facing guidance signals: useful for explanation, slicing, and retrieval policy, but not claims of verified proof synthesis.",
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
