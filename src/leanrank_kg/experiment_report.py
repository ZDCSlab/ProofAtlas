from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import load_config, read_json, stable_hash


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_table(metrics: dict[str, Any], keys: list[str]) -> str:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key in keys:
        lines.append(f"| `{key}` | {_fmt(metrics.get(key))} |")
    return "\n".join(lines)


def _domain_table(rows: list[dict[str, Any]], metric_keys: list[str]) -> str:
    lines = ["| Domain | Queries | " + " | ".join(f"`{key}`" for key in metric_keys) + " |", "| --- | ---: | " + " | ".join("---:" for _ in metric_keys) + " |"]
    for row in rows[:12]:
        metrics = row.get("metrics", {})
        query_count = metrics.get("evaluated_queries") or metrics.get("theorem_retrieval_evaluated_queries") or metrics.get("theorem_retrieval_evaluated_theorems")
        values = " | ".join(_fmt(metrics.get(key)) for key in metric_keys)
        lines.append(f"| {row.get('domain_tag', 'Unknown')} | {_fmt(query_count)} | {values} |")
    if not rows:
        lines.append("| n/a | n/a | " + " | ".join("n/a" for _ in metric_keys) + " |")
    return "\n".join(lines)


def _worst_case_table(rows: list[dict[str, Any]], id_key: str, recall_key: str = "Recall@10") -> str:
    lines = [
        "| Item | Domain | Gold in train | Missing gold | Recall@10 | MRR contribution | MAP contribution |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:10]:
        item = row.get(id_key) or row.get("full_name") or row.get("theorem_id") or "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item),
                    str(row.get("domain_tag", "Unknown")),
                    _fmt(row.get("gold_in_train_index_count")),
                    _fmt(row.get("gold_missing_from_train_index_count")),
                    _fmt(row.get(recall_key)),
                    _fmt(row.get("reciprocal_rank")),
                    _fmt(row.get("average_precision")),
                ]
            )
            + " |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines)


def _failure_profile_table(profile: dict[str, Any]) -> str:
    lines = ["| Signal | Value |", "| --- | ---: |"]
    for key in [
        "evaluated_queries",
        "retrievable_queries",
        "queries_without_train_gold",
        "queries_with_missing_gold",
        "zero_recall_at_max_k",
        "max_k",
    ]:
        lines.append(f"| `{key}` | {_fmt(profile.get(key))} |")
    return "\n".join(lines)


def _rank_bucket_table(profile: dict[str, Any]) -> str:
    buckets = profile.get("rank_buckets", {}) if isinstance(profile, dict) else {}
    lines = ["| Rank bucket | Queries |", "| --- | ---: |"]
    for key, value in buckets.items():
        lines.append(f"| `{key}` | {_fmt(value)} |")
    if len(lines) == 2:
        lines.append("| n/a | n/a |")
    return "\n".join(lines)


def _coverage_bucket_table(profile: dict[str, Any]) -> str:
    buckets = profile.get("gold_coverage_buckets", {}) if isinstance(profile, dict) else {}
    lines = ["| Gold coverage bucket | Queries |", "| --- | ---: |"]
    for key, value in buckets.items():
        lines.append(f"| `{key}` | {_fmt(value)} |")
    if len(lines) == 2:
        lines.append("| n/a | n/a |")
    return "\n".join(lines)


def _zero_recall_domain_table(profile: dict[str, Any]) -> str:
    rows = profile.get("zero_recall_domains", []) if isinstance(profile, dict) else []
    lines = ["| Domain | Zero-recall queries |", "| --- | ---: |"]
    for row in rows[:10]:
        lines.append(f"| {row.get('domain_tag', 'Unknown')} | {_fmt(row.get('zero_recall_queries'))} |")
    if len(lines) == 2:
        lines.append("| n/a | n/a |")
    return "\n".join(lines)


def _index_benchmark_table(bench_entities: dict[str, Any]) -> str:
    lines = [
        "| Entity | Backend | Rows | Exact ms/query | Indexed ms/query | Speedup | Recall@1 vs exact | Recall@5 vs exact | Recall@10 vs exact | Top1 match@10 | Build seconds | Indexed total seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entity, row in sorted(bench_entities.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entity),
                    _fmt(row.get("backend")),
                    _fmt(row.get("rows")),
                    _fmt(row.get("exact_ms_per_query")),
                    _fmt(row.get("indexed_ms_per_query")),
                    _fmt(row.get("speedup_vs_exact")),
                    _fmt(row.get("recall_at_1_vs_exact")),
                    _fmt(row.get("recall_at_5_vs_exact")),
                    _fmt(row.get("recall_at_10_vs_exact") or row.get("recall_vs_exact")),
                    _fmt(row.get("top1_match_at_10_vs_exact")),
                    _fmt(row.get("index_build_seconds")),
                    _fmt(row.get("indexed_total_seconds")),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines)


def _retrieval_bottleneck_table(profile: dict[str, Any]) -> str:
    lines = [
        "| Task | Recall@10 | Recall@100 | Gap | Top10/Top100 | Primary bottleneck |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    labels = {"proof_state": "Proof-state premise retrieval", "theorem": "Theorem-level premise retrieval"}
    for key, label in labels.items():
        row = profile.get(key, {}) if isinstance(profile, dict) else {}
        lines.append(
            f"| {label} | {_fmt(row.get('recall_at_10'))} | {_fmt(row.get('recall_at_100'))} | "
            f"{_fmt(row.get('top10_to_top100_gap'))} | {_fmt(row.get('top10_fraction_of_top100'))} | "
            f"`{row.get('primary_accuracy_bottleneck', 'n/a')}` |"
        )
    return "\n".join(lines)


def _evaluation_substage_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Evaluation substage | Seconds | Queries | Backend |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows[:8]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('name', 'n/a')}`",
                    _fmt(row.get("seconds")),
                    _fmt(row.get("evaluated_queries")),
                    _fmt(row.get("actual_backend")),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a |")
    return "\n".join(lines)


def _split_counts(manifest: dict[str, Any]) -> str:
    counts = manifest.get("split_counts", {}) if isinstance(manifest, dict) else {}
    if not counts:
        return "n/a"
    pieces = []
    for split, count in sorted(counts.items()):
        if isinstance(count, dict):
            value = count.get("rows") or count.get("theorems") or count.get("theorem_count") or count
        else:
            value = count
        pieces.append(f"{split}: {value}")
    return ", ".join(pieces)


def build_markdown(config_path: str = "configs/proofatlas.yaml") -> str:
    config = load_config(config_path)
    config_hash = stable_hash(json.dumps(config, sort_keys=True), 16)
    manifest = read_json("outputs/reports/corpus_manifest.json", {}) or {}
    test_eval = read_json("outputs/reports/test_set_evaluation.json", {}) or {}
    pipeline = read_json("outputs/reports/pipeline_performance_report.json", {}) or {}
    benchmark = read_json("outputs/reports/index_benchmark.json", {}) or {}
    premise_trace = read_json("outputs/reports/premise_trace_supervision_report.json", {}) or {}
    timings = read_json("outputs/reports/pipeline_run_timings.json", {}) or {}
    data_supervision = manifest.get("data_supervision", {}) if isinstance(manifest, dict) else {}

    proof_metrics = test_eval.get("test", {}).get("proof_state_retrieval", {}).get("metrics", {})
    reranked_proof_state = test_eval.get("test", {}).get("proof_state_reranked_retrieval", {}) if isinstance(test_eval, dict) else {}
    reranked_proof_metrics = reranked_proof_state.get("metrics", {})
    reranked_backend = reranked_proof_state.get("backend_info", {})
    reranked_candidate_k_ablation = reranked_proof_state.get("candidate_k_ablation", [])
    query_representation_diagnostic = test_eval.get("test", {}).get("proof_state_query_representation_diagnostic", {}) if isinstance(test_eval, dict) else {}
    theorem_metrics = test_eval.get("test", {}).get("theorem_retrieval", {}).get("metrics", {})
    proof_domain_breakdown = test_eval.get("test", {}).get("proof_state_retrieval", {}).get("domain_breakdown", [])
    theorem_domain_breakdown = test_eval.get("test", {}).get("theorem_retrieval", {}).get("domain_breakdown", [])
    proof_failure_profile = test_eval.get("test", {}).get("proof_state_retrieval", {}).get("failure_profile", {})
    theorem_failure_profile = test_eval.get("test", {}).get("theorem_retrieval", {}).get("failure_profile", {})
    reranked_failure_profile = reranked_proof_state.get("failure_profile", {})
    proof_worst_cases = test_eval.get("test", {}).get("proof_state_retrieval", {}).get("worst_cases", [])
    theorem_worst_cases = test_eval.get("test", {}).get("theorem_retrieval", {}).get("worst_cases", [])
    validation_proof_metrics = test_eval.get("validation", {}).get("proof_state_retrieval", {}).get("metrics", {})
    validation_theorem_metrics = test_eval.get("validation", {}).get("theorem_retrieval", {}).get("metrics", {})
    evaluation_scope = test_eval.get("evaluation_scope", {}) if isinstance(test_eval, dict) else {}
    evaluation_profile = pipeline.get("stages", {}).get("evaluation", {}) if isinstance(pipeline, dict) else {}
    held_out_test_coverage = evaluation_profile.get("held_out_test_coverage", {}) if isinstance(evaluation_profile, dict) else {}
    evaluation_timing = evaluation_profile.get("evaluation_timing", {}) if isinstance(evaluation_profile, dict) else {}
    evaluation_substages = evaluation_timing.get("slowest_substages", []) if isinstance(evaluation_timing, dict) else []
    recommendations = pipeline.get("recommendations", [])
    throughput = pipeline.get("throughput_profile", {}) if isinstance(pipeline, dict) else {}
    evaluation_timing_delta = throughput.get("evaluation_timing_delta", {}) if isinstance(throughput, dict) else {}
    bottleneck_profile = throughput.get("bottleneck_profile", {}) if isinstance(throughput, dict) else {}
    embedding_bottleneck = throughput.get("embedding_bottleneck_profile", {}) if isinstance(throughput, dict) else {}
    retrieval_bottleneck = throughput.get("retrieval_bottleneck_profile", {}) if isinstance(throughput, dict) else {}
    bench_entities = benchmark.get("entities", {}) if isinstance(benchmark, dict) else {}
    actual_backend_info = evaluation_scope.get("actual_backend_info", {}) if isinstance(evaluation_scope, dict) else {}
    actual_proof_backend = actual_backend_info.get("proof_state", {}).get("test", {}).get("actual_backend", "n/a")
    actual_theorem_backend = actual_backend_info.get("theorem", {}).get("test", {}).get("actual_backend", "n/a")
    timing_config_hash = timings.get("config_hash") if isinstance(timings, dict) else None
    timing_config_matches = timing_config_hash == config_hash if timing_config_hash else None

    proof_keys = [
        "evaluated_queries",
        "evaluated_retrievable_queries",
        "gold_premise_coverage",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR",
        "MAP",
        "nDCG@10",
    ]
    theorem_keys = [
        "theorem_retrieval_evaluated_theorems",
        "theorem_retrieval_evaluated_theorems_with_train_gold",
        "theorem_retrieval_gold_premise_coverage",
        "theorem_retrieval_Recall@1",
        "theorem_retrieval_Recall@5",
        "theorem_retrieval_Recall@10",
        "theorem_retrieval_MRR",
        "theorem_retrieval_MAP",
        "theorem_retrieval_nDCG@10",
    ]
    candidate_diagnostic_keys = ["Recall@50", "Recall@100", "nDCG@50", "nDCG@100"]
    theorem_candidate_diagnostic_keys = [
        "theorem_retrieval_Recall@50",
        "theorem_retrieval_Recall@100",
        "theorem_retrieval_nDCG@50",
        "theorem_retrieval_nDCG@100",
    ]
    query_representation_lines = [
        "| Query Representation | Recall@50 | Recall@100 | MRR | MAP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted((query_representation_diagnostic.get("variants") or {}).items()):
        metrics = row.get("metrics", {})
        query_representation_lines.append(
            f"| `{name}` | {_fmt(metrics.get('Recall@50'))} | {_fmt(metrics.get('Recall@100'))} | {_fmt(metrics.get('MRR'))} | {_fmt(metrics.get('MAP'))} |"
        )
    if len(query_representation_lines) == 2:
        query_representation_lines.append("| n/a | n/a | n/a | n/a | n/a |")
    reranked_candidate_lines = [
        "| Candidate k | Recall@1 | Recall@5 | Recall@10 | MRR | MAP |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in reranked_candidate_k_ablation:
        metrics = row.get("metrics", {})
        reranked_candidate_lines.append(
            f"| {_fmt(row.get('candidate_k'))} | {_fmt(metrics.get('Recall@1'))} | {_fmt(metrics.get('Recall@5'))} | {_fmt(metrics.get('Recall@10'))} | {_fmt(metrics.get('MRR'))} | {_fmt(metrics.get('MAP'))} |"
        )
    if len(reranked_candidate_lines) == 2:
        reranked_candidate_lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")

    lines = [
        "# ProofAtlas Experiment Report",
        "",
        "## Experiment Setup",
        "",
        f"- Dataset: `{manifest.get('dataset_name') or config.get('dataset_name')}`",
        f"- Source kind: `{manifest.get('source_kind', 'n/a')}`",
        f"- Split counts: {_split_counts(manifest)}",
        f"- Candidate pool: `{test_eval.get('candidate_pool', 'train premise index')}`",
        f"- Label policy: {test_eval.get('label_policy', 'held-out test positive_edges are used only for evaluation')}",
        f"- Evaluation scope: `{'sampled held-out splits' if evaluation_scope.get('is_sampled') else 'full held-out splits'}`",
        f"- Proof-state evaluation limits: `{evaluation_scope.get('proof_state_limits', 'n/a')}`",
        f"- Theorem evaluation limits: `{evaluation_scope.get('theorem_limits', 'n/a')}`",
        f"- Proof-state test coverage: `{held_out_test_coverage.get('proof_state_evaluated_queries', 'n/a')}` / `{held_out_test_coverage.get('proof_state_total', 'n/a')}` (`{held_out_test_coverage.get('proof_state_coverage_fraction', 'n/a')}`)",
        f"- Theorem test coverage: `{held_out_test_coverage.get('theorem_evaluated_queries', 'n/a')}` / `{held_out_test_coverage.get('theorem_total', 'n/a')}` (`{held_out_test_coverage.get('theorem_coverage_fraction', 'n/a')}`)",
        f"- Ranking backend: `{evaluation_scope.get('ranking_backend', 'n/a')}`",
        f"- Evaluation GPU: `use_gpu={evaluation_scope.get('use_gpu', 'n/a')}`, device `{evaluation_scope.get('gpu_device', 'n/a')}`, batch size `{evaluation_scope.get('batch_size', 'n/a')}`",
        f"- Actual test ranking backend: proof-state `{actual_proof_backend}`, theorem `{actual_theorem_backend}`",
        f"- Case-study regeneration during evaluation: `{evaluation_scope.get('case_study_limit', 'n/a')}` theorem guidance cases",
        f"- Data supervision: `{data_supervision.get('kind', 'n/a')}`",
        f"- Has tactic states: `{data_supervision.get('has_tactic_states', 'n/a')}`",
        f"- Has true positive premises: `{data_supervision.get('has_true_positive_premises', 'n/a')}`",
        f"- Embedding backend: `{config.get('embedding', {}).get('backend', 'n/a')}`",
        f"- Index backend: `{config.get('index', {}).get('backend', 'n/a')}`",
        "",
        "## Final Artifacts",
        "",
        "- Homepage/demo: `homepage/index.html`",
        "- Experiment report: `outputs/reports/experiment_report.md`",
        "- Machine-readable held-out evaluation: `outputs/reports/test_set_evaluation.json`",
        "- Pipeline performance profile: `outputs/reports/pipeline_performance_report.json`",
        "",
        "## ML Task Definition",
        "",
        "ProofAtlas is evaluated as a supervised ranking/retrieval system over a theorem-disjoint train/validation/test split. The train split supplies the candidate premise index and graph evidence. The held-out validation and test positive edges are never used as retrieval candidates; they are used only as gold labels for scoring.",
        "",
        "- Proof-state-level task: query with a held-out proof state and rank train-side premises.",
        "- Theorem-level task: query with a held-out theorem-level text representation and rank train-side premises for proof guidance.",
        "- Reported metrics: Recall@k, MRR, MAP, nDCG@k, and gold-premise coverage against the train premise index.",
        "- When evaluation limits are configured, metrics are computed on a deterministic prefix sample of the held-out split and reported as sampled held-out metrics.",
        "",
        "The current default experiment uses `erbacher/LeanRank-data` only. Local Lean/mathlib source extraction is out of scope for this experiment.",
        "",
        "## Candidate Pool Diagnostic",
        "",
        "These metrics test whether the embedding/index candidate pool contains the gold premise before reranking. If Recall@100 is low, the next accuracy bottleneck is candidate generation or embeddings; if Recall@100 is high but Recall@10 is low, the bottleneck is reranking.",
        "",
        "### Retrieval Bottleneck Profile",
        "",
        _retrieval_bottleneck_table(retrieval_bottleneck),
        "",
        "### Proof-State Candidate Pool",
        "",
        _metric_table(proof_metrics, candidate_diagnostic_keys),
        "",
        "### Theorem Candidate Pool",
        "",
        _metric_table(theorem_metrics, theorem_candidate_diagnostic_keys),
        "",
        "### Proof-State Query Representation Diagnostic",
        "",
        f"- Evaluated queries: `{query_representation_diagnostic.get('evaluated_queries', 'n/a')}`",
        f"- Best variant by `{query_representation_diagnostic.get('selection_metric', 'Recall@100')}`: `{query_representation_diagnostic.get('best_variant_by_recall', 'n/a')}`",
        "",
        "\n".join(query_representation_lines),
        "",
        "## Held-Out Test Set Metrics",
        "",
        "### Proof-State-Level Premise Ranking",
        "",
        "Each held-out test proof state is used as a query. Gold positive premises from the test proof trace are used only for scoring, while candidates come from the train premise index.",
        "",
        _metric_table(proof_metrics, proof_keys),
        "",
        "### Proof-State-Level Reranked Retrieval",
        "",
        "This smaller diagnostic uses the same retrieval path as the homepage/API: query encoding, hnswlib candidate retrieval, and the learned/fixed reranker. It is slower than batched embedding evaluation, but better reflects user-facing proof guidance.",
        "",
        f"- Backend: `{reranked_backend.get('actual_backend', 'n/a')}`",
        f"- Candidate k: `{reranked_backend.get('candidate_k', 'n/a')}`",
        f"- Evaluated queries: `{reranked_backend.get('evaluated_queries', 'n/a')}`",
        "",
        _metric_table(reranked_proof_metrics, proof_keys),
        "",
        "#### Rerank Candidate-Depth Ablation",
        "",
        "This ablation uses the same held-out rerank diagnostic queries and changes only the number of embedding candidates passed into the learned/fixed reranker.",
        "",
        "\n".join(reranked_candidate_lines),
        "",
        "### Theorem-Level Premise Ranking",
        "",
        "Each held-out test theorem is used as a query for proof guidance. Gold premises are all positive premises used by that theorem in the held-out split.",
        "",
        _metric_table(theorem_metrics, theorem_keys),
        "",
        "## Domain Breakdown",
        "",
        "These tables show held-out test metrics grouped by LeanRank-data domain. They help identify where ranking quality is strong or weak instead of relying only on aggregate metrics.",
        "",
        "### Test Proof-State-Level Domains",
        "",
        _domain_table(proof_domain_breakdown, ["Recall@10", "MRR", "MAP", "nDCG@10", "gold_premise_coverage"]),
        "",
        "### Test Theorem-Level Domains",
        "",
        _domain_table(
            theorem_domain_breakdown,
            [
                "theorem_retrieval_Recall@10",
                "theorem_retrieval_MRR",
                "theorem_retrieval_MAP",
                "theorem_retrieval_nDCG@10",
                "theorem_retrieval_gold_premise_coverage",
            ],
        ),
        "",
        "## Error Analysis",
        "",
        "Worst-case rows are held-out test queries with train-index gold premises but low top-k recovery. These are the first examples to inspect when improving embeddings, reranking features, or candidate depth.",
        "",
        "### Failure Profile Summary",
        "",
        "These aggregate buckets quantify where held-out retrieval fails without storing every per-query row in the committed report. `zero_recall_at_max_k` counts retrievable queries where no train-side gold premise appeared within the largest evaluated candidate pool.",
        "",
        "#### Proof-State Failure Profile",
        "",
        _failure_profile_table(proof_failure_profile),
        "",
        "Proof-state rank buckets:",
        "",
        _rank_bucket_table(proof_failure_profile),
        "",
        "Proof-state gold coverage buckets:",
        "",
        _coverage_bucket_table(proof_failure_profile),
        "",
        "Proof-state zero-recall domains:",
        "",
        _zero_recall_domain_table(proof_failure_profile),
        "",
        "#### Theorem Failure Profile",
        "",
        _failure_profile_table(theorem_failure_profile),
        "",
        "Theorem rank buckets:",
        "",
        _rank_bucket_table(theorem_failure_profile),
        "",
        "Theorem gold coverage buckets:",
        "",
        _coverage_bucket_table(theorem_failure_profile),
        "",
        "Theorem zero-recall domains:",
        "",
        _zero_recall_domain_table(theorem_failure_profile),
        "",
        "#### Reranked Proof-State Diagnostic Failure Profile",
        "",
        _failure_profile_table(reranked_failure_profile),
        "",
        "### Worst Proof-State Queries",
        "",
        _worst_case_table(proof_worst_cases, "proof_state_id"),
        "",
        "### Worst Theorem Queries",
        "",
        _worst_case_table(theorem_worst_cases, "full_name"),
        "",
        "## Validation Metrics",
        "",
        "Validation metrics are reported for model selection and sanity checking; final claims should use the held-out test metrics above.",
        "",
        "### Validation Proof-State-Level Premise Ranking",
        "",
        _metric_table(validation_proof_metrics, proof_keys),
        "",
        "### Validation Theorem-Level Premise Ranking",
        "",
        _metric_table(validation_theorem_metrics, theorem_keys),
        "",
        "## ANN Index Benchmark",
        "",
        "This benchmark compares the saved nearest-neighbor index against exact cosine search on sampled train queries. It measures whether the ANN backend is fast enough for interactive retrieval while preserving the exact top-k neighborhood used by the embedding candidate generator.",
        "",
        _index_benchmark_table(bench_entities),
    ]

    current_trace = premise_trace.get("current_artifact_supervision", {}) if isinstance(premise_trace, dict) else {}
    train_trace = premise_trace.get("splits", {}).get("train", {}) if isinstance(premise_trace, dict) else {}
    label_conflicts = premise_trace.get("normalization_label_conflicts", {}) if isinstance(premise_trace, dict) else {}
    lines.extend(
        [
            "",
            "## Premise Trace Supervision",
            "",
            "The current ranking labels come from normalized LeanRank-data premise supervision. Positive edges are treated as proof-state-to-premise positives; negative candidates are treated as hard/failed candidate premises for ranking and difficulty features.",
            "",
            f"- Positive edges: `{current_trace.get('total_positive_edges', 'n/a')}`",
            f"- Negative candidates: `{current_trace.get('total_negative_edges', 'n/a')}`",
            f"- Negative/positive edge ratio: `{current_trace.get('negative_to_positive_edge_ratio', 'n/a')}`",
            f"- Train proof states with positive edges: `{train_trace.get('proof_states_with_positive_edges', 'n/a')}`",
            f"- Train proof states with negative candidates: `{train_trace.get('proof_states_with_negative_edges', 'n/a')}`",
            f"- Train positive proof-state coverage: `{train_trace.get('positive_proof_state_coverage', 'n/a')}`",
            f"- Train negative proof-state coverage: `{train_trace.get('negative_proof_state_coverage', 'n/a')}`",
            f"- Train positive/negative pair overlap count: `{train_trace.get('positive_negative_pair_overlap_count', 'n/a')}`",
            f"- Removed positive/negative label conflicts during normalization: `{label_conflicts.get('total_positive_negative_overlap_removed', 'n/a')}`",
            f"- Train negative-candidate hardness mean: `{train_trace.get('negative_candidate_hardness', {}).get('mean', 'n/a')}`",
            f"- Supervision quality checks: `{current_trace.get('quality_checks', {})}`",
            f"- Supervision scope: `{premise_trace.get('scope', 'erbacher/LeanRank-data normalized positive/negative premise supervision')}`",
            "",
            "## Pipeline Timing",
            "",
            f"- Total seconds: `{timings.get('total_seconds', 'n/a')}`",
            f"- Stage count: `{timings.get('stage_count', 'n/a')}`",
            f"- Executed/skipped stages: `{timings.get('executed_stage_count', 'n/a')}` / `{timings.get('skipped_stage_count', 'n/a')}`",
            f"- Timing config matches current report config: `{timing_config_matches if timing_config_matches is not None else 'n/a'}`",
            f"- Timing generated at: `{timings.get('generated_at', 'n/a')}`",
            f"- Timing report: `outputs/reports/pipeline_run_timings.json`",
            f"- Evaluation internal total seconds: `{evaluation_timing.get('total_seconds', 'n/a')}`",
            f"- Evaluation timed substages: `{evaluation_timing.get('substage_count', 'n/a')}`",
            "",
            "| Stage | Seconds |",
            "| --- | ---: |",
        ]
    )
    for row in timings.get("slowest_stages", [])[:10]:
        lines.append(f"| `{row.get('name')}` | {_fmt(row.get('seconds'))} |")
    if not timings.get("slowest_stages"):
        lines.append("| n/a | n/a |")
    lines.extend(
        [
            "",
            "### Evaluation Substage Timing",
            "",
            "These timings split the `evaluate` pipeline stage into proof-state retrieval, theorem retrieval, reranked retrieval, and query-representation diagnostics so scaling work can target the slowest internal path.",
            "",
            _evaluation_substage_table(evaluation_substages),
        ]
    )
    lines.extend(
        [
            "",
            "## Pipeline Performance And Scale-Up Notes",
            "",
            f"- Pipeline profile: `outputs/reports/pipeline_performance_report.json`",
            f"- Scale bucket: `{pipeline.get('scale_profile', {}).get('scale_bucket', 'n/a')}`",
            f"- Requested theorems: `{pipeline.get('scale_profile', {}).get('requested_theorems', 'n/a')}`",
            f"- Source rows requested: `{pipeline.get('scale_profile', {}).get('source_rows', 'n/a')}`",
            f"- Current split rows: `{pipeline.get('scale_profile', {}).get('current_total_split_rows', 'n/a')}`",
            f"- Target dataset confirmed: `{pipeline.get('scale_profile', {}).get('target_dataset_confirmed', 'n/a')}`",
            f"- LeanRank premise supervision ready: `{pipeline.get('scale_profile', {}).get('leanrank_premise_supervision_ready', 'n/a')}`",
            f"- Embedding devices: `{pipeline.get('scale_profile', {}).get('embedding_devices', [])}`",
            f"- ANN backend availability: `{pipeline.get('scale_profile', {}).get('ann_backend_availability', {})}`",
            f"- Total embedding rows: `{throughput.get('total_embedding_rows', 'n/a')}`",
            f"- Timing config matches current report config: `{throughput.get('timing_config_matches_current', 'n/a')}`",
            f"- Throughput timing basis: `{throughput.get('throughput_basis', 'n/a')}`",
            f"- Scale estimate reliable: `{throughput.get('scale_estimate_reliable', 'n/a')}`",
            f"- Embedding rows by entity: `{throughput.get('embedding_rows_by_entity', {})}`",
            f"- Embedding rows by split: `{embedding_bottleneck.get('embedding_rows_by_split', {})}`",
            f"- Embedding matrix bytes: `{embedding_bottleneck.get('embedding_matrix_bytes', 'n/a')}`",
            f"- Embed stage seconds: `{embedding_bottleneck.get('embed_stage_seconds', 'n/a')}`",
            f"- Embed stage share of total: `{embedding_bottleneck.get('embed_stage_share_of_total', 'n/a')}`",
            f"- Embedding rows/sec during embed stage: `{embedding_bottleneck.get('embedding_rows_per_embed_second', 'n/a')}`",
            f"- Processed rows/sec: `{throughput.get('processed_rows_per_second', 'n/a')}`",
            f"- Pipeline seconds per 100k processed rows: `{throughput.get('pipeline_seconds_per_100k_processed_rows', 'n/a')}`",
            f"- Slowest timed stage: `{throughput.get('slowest_stage', 'n/a')}`",
            f"- Saved pipeline evaluate seconds: `{evaluation_timing_delta.get('timed_pipeline_evaluate_seconds', 'n/a')}`",
            f"- Current standalone evaluation seconds: `{evaluation_timing_delta.get('current_evaluation_seconds', 'n/a')}`",
            f"- Timed/current evaluation ratio: `{evaluation_timing_delta.get('timed_to_current_ratio', 'n/a')}`",
            f"- Primary bottleneck share: `{bottleneck_profile.get('primary_stage_share_of_total', 'n/a')}`",
            f"- Top-3 timed-stage share: `{bottleneck_profile.get('top3_stage_share_of_total', 'n/a')}`",
            f"- Mean index speedup vs exact: `{throughput.get('mean_index_speedup_vs_exact', 'n/a')}`",
            f"- Minimum index recall vs exact: `{throughput.get('min_index_recall_vs_exact', 'n/a')}`",
            f"- Estimated seconds at requested source rows: `{throughput.get('estimated_seconds_at_requested_source_rows', 'n/a')}`",
            "",
            "| Bottleneck stage | Seconds | Share of total |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in bottleneck_profile.get("top_stages", [])[:5]:
        lines.append(f"| `{row.get('name')}` | {_fmt(row.get('seconds'))} | {_fmt(row.get('share_of_total'))} |")
    if not bottleneck_profile.get("top_stages"):
        lines.append("| n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    if recommendations:
        for row in recommendations:
            lines.append(f"- `{row.get('priority', 'n/a')}` `{row.get('area', 'general')}`: {row.get('recommendation', '')}")
    else:
        lines.append("- No recommendations recorded.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report treats ProofAtlas as an ML ranking and retrieval system on `erbacher/LeanRank-data`. The main quantitative claims are the held-out test-set ranking metrics above.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: str = "configs/proofatlas.yaml", output_path: str = "outputs/reports/experiment_report.md") -> dict[str, Any]:
    markdown = build_markdown(config_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return {"path": str(path), "bytes": int(path.stat().st_size)}
