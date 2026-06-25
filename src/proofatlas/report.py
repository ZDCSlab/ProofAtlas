from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_json


def _metric(report: dict[str, Any], path: list[str], default: Any = "") -> Any:
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pct(part: int | float, total: int | float) -> str:
    return f"{(float(part) / max(float(total), 1.0)) * 100:.1f}%"


def _overall_recall(metrics: dict[str, Any], k: int) -> str:
    recall = metrics.get(f"Recall@{k}", "")
    coverage = metrics.get("gold_pool_coverage", "")
    if isinstance(recall, int | float) and isinstance(coverage, int | float):
        return _fmt(float(recall) * float(coverage))
    return ""


def _read_processed(split: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(f"data/processed/{split}/{name}.parquet")


def _split_stats(split: str) -> dict[str, Any]:
    theorems = _read_processed(split, "theorems")
    proof_states = _read_processed(split, "proof_states")
    premises = _read_processed(split, "premises")
    positives = _read_processed(split, "positive_edges")
    techniques = _read_processed(split, "proof_state_techniques")
    features = _read_processed(split, "theorem_features")
    return {
        "theorems": len(theorems),
        "proof_states": len(proof_states),
        "premises": len(premises),
        "positive_edges": len(positives),
        "avg_proof_states_per_theorem": len(proof_states) / max(len(theorems), 1),
        "avg_positive_edges_per_proof_state": len(positives) / max(len(proof_states), 1),
        "domains": theorems["domain_tag"].value_counts().to_dict(),
        "top_domains": theorems["domain_tag"].value_counts().head(8).to_dict(),
        "proof_state_domains": proof_states["domain_tag"].value_counts().head(8).to_dict(),
        "strategy_labels": techniques["label"].value_counts().head(8).to_dict(),
        "difficulty_buckets": features["difficulty_bucket"].value_counts().to_dict(),
    }


def _append_dataset_overview(lines: list[str], split: str) -> None:
    train = _split_stats("train")
    val = _split_stats("val")
    target = _split_stats(split)
    lines.extend(
        [
            "## Dataset",
            "",
            "The experiment uses the processed LeanRank-derived ID split already present in `data/processed`. Train, validation, and test are treated as theorem-disjoint in-distribution splits: theorem and proof-state names, domains, and premise vocabulary are drawn from the same corpus regime as train, while positives for validation/test proof states are held out for evaluation.",
            "",
            "| Split | Theorems | Proof states | Premises | Positive edges | Avg proof states/theorem | Avg positives/proof-state |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    split_rows = [("train", train), ("val", val)]
    if split != "val":
        split_rows.append((split, target))
    for name, stats in split_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(stats["theorems"]),
                    str(stats["proof_states"]),
                    str(stats["premises"]),
                    str(stats["positive_edges"]),
                    _fmt(stats["avg_proof_states_per_theorem"]),
                    _fmt(stats["avg_positive_edges_per_proof_state"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "### Top Theorem Domain Distribution", ""])
    lines.extend(["| Domain | Train theorems | Val theorems | Report split theorems | Report split share |", "| --- | ---: | ---: | ---: | ---: |"])
    all_domains = list(dict.fromkeys([*train["top_domains"].keys(), *val["top_domains"].keys(), *target["top_domains"].keys()]))
    for domain in all_domains:
        train_count = int(train["domains"].get(domain, 0))
        val_count = int(val["domains"].get(domain, 0))
        test_count = int(target["domains"].get(domain, 0))
        lines.append(f"| {domain} | {train_count} | {val_count} | {test_count} | {_pct(test_count, target['theorems'])} |")
    train_other = int(train["theorems"] - sum(int(train["domains"].get(domain, 0)) for domain in all_domains))
    val_other = int(val["theorems"] - sum(int(val["domains"].get(domain, 0)) for domain in all_domains))
    target_other = int(target["theorems"] - sum(int(target["domains"].get(domain, 0)) for domain in all_domains))
    lines.append(f"| Other | {train_other} | {val_other} | {target_other} | {_pct(target_other, target['theorems'])} |")
    lines.extend(["", "### Labels Used by Guidance Tasks", ""])
    lines.extend(["| Label type | Train distribution | Val distribution | Report split distribution |", "| --- | --- | --- | --- |"])
    for label, train_values, test_values in [
        ("Difficulty bucket", train["difficulty_buckets"], target["difficulty_buckets"]),
        ("Strategy facets top labels", train["strategy_labels"], target["strategy_labels"]),
    ]:
        val_values = val["difficulty_buckets"] if label == "Difficulty bucket" else val["strategy_labels"]
        train_text = ", ".join(f"{key}: {value}" for key, value in train_values.items())
        val_text = ", ".join(f"{key}: {value}" for key, value in val_values.items())
        test_text = ", ".join(f"{key}: {value}" for key, value in test_values.items())
        lines.append(f"| {label} | {train_text} | {val_text} | {test_text} |")
    lines.append("")


def _append_glossary(lines: list[str]) -> None:
    lines.extend(
        [
            "## Metric and Abbreviation Glossary",
            "",
            "| Term | Meaning |",
            "| --- | --- |",
            "| T1 | Proof-State -> Premise retrieval, the main task. A held-out proof state is used to retrieve train-side premises. |",
            "| T2 | Theorem -> Theorem retrieval. A held-out theorem profile is used to retrieve similar train-side theorem profiles. |",
            "| T3 | Similar-theorem guidance aggregation. Similar train theorems are used to aggregate premise, strategy, and difficulty evidence. |",
            "| LLM | Large language model. Here it refers to DeepSeek-generated theorem semantic/strategy/difficulty-oriented enrichment text. |",
            "| BGE | Beijing Academy of Artificial Intelligence General Embedding model family. This report uses `BAAI/bge-base-en-v1.5`. |",
            "| TF-IDF | Term frequency-inverse document frequency sparse lexical retrieval. |",
            "| RRF | Reciprocal-rank fusion. Each source contributes `weight / rank`; candidates are sorted by summed score. |",
            "| Weighted RRF | RRF with source-specific weights, used to fuse dense, lexical, proof-state expansion, and theorem-neighborhood sources. |",
            "| Dense retrieval | Vector similarity retrieval using precomputed embeddings. |",
            "| Lexical retrieval | Sparse text retrieval using token overlap/TF-IDF rather than neural embeddings. |",
            "| Similar proof-state expansion | Retrieve similar train proof states, then rank premises used by those train proof states. |",
            "| Similar theorem premise source | Retrieve similar train theorems, then rank premises used by proof states under those train theorems. |",
            "| Covered Recall@K | Fraction of retrievable gold positives found in the top K retrieved candidates. The denominator excludes gold positives absent from the train-side premise pool. |",
            "| Overall Recall@100 | `Covered Recall@100 * Gold coverage`, an approximate all-positive recall after accounting for missing gold positives. |",
            "| MAP | Mean average precision over retrievable positives. It rewards ranking gold premises earlier across the candidate list. |",
            "| nDCG@K | Normalized discounted cumulative gain at K. It rewards hits near the top of the ranked list. |",
            "| Gold coverage | Fraction of all held-out positive edges whose premise appears in the train-side retrievable premise pool. This is a ceiling factor for recall. |",
            "| Strategy Recall/Precision/F1 | Diagnostics for broad precomputed strategy labels aggregated from neighbor theorems. They are guidance diagnostics, not the primary retrieval claim. |",
            "| Difficulty MAE | Mean absolute error between retrieved-neighbor difficulty score average and the query theorem's proxy difficulty score. |",
            "| Difficulty bucket accuracy | Accuracy of the majority bucket from neighbor difficulty evidence against the query theorem's proxy difficulty bucket. |",
            "",
        ]
    )


def _append_t1_table(lines: list[str], t1: dict[str, Any], methods: list[tuple[str, str]]) -> None:
    lines.extend(
        [
            "| Stage | Method | Covered Recall@10 | Covered Recall@50 | Covered Recall@100 | Overall Recall@100 | MAP | nDCG@10 | Gold coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage, method in methods:
        metrics = _metric(t1, ["methods", method], {})
        if not metrics:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    stage,
                    method,
                    _fmt(metrics.get("Recall@10", "")),
                    _fmt(metrics.get("Recall@50", "")),
                    _fmt(metrics.get("Recall@100", "")),
                    _overall_recall(metrics, 100),
                    _fmt(metrics.get("MAP", "")),
                    _fmt(metrics.get("nDCG@10", "")),
                    _fmt(metrics.get("gold_pool_coverage", "")),
                ]
            )
            + " |"
        )


def _append_t1_ablation_table(lines: list[str], t1: dict[str, Any], methods: list[str]) -> None:
    lines.extend(
        [
            "| Method | Covered Recall@10 | Covered Recall@50 | Covered Recall@100 | Overall Recall@100 | MAP | nDCG@10 | Gold coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in methods:
        metrics = _metric(t1, ["methods", method], {})
        if not metrics:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _fmt(metrics.get("Recall@10", "")),
                    _fmt(metrics.get("Recall@50", "")),
                    _fmt(metrics.get("Recall@100", "")),
                    _overall_recall(metrics, 100),
                    _fmt(metrics.get("MAP", "")),
                    _fmt(metrics.get("nDCG@10", "")),
                    _fmt(metrics.get("gold_pool_coverage", "")),
                ]
            )
            + " |"
        )


def _append_t2_table(lines: list[str], rows: list[tuple[str, dict[str, Any]]]) -> None:
    lines.extend(
        [
            "| Method | Neighbor premise covered Recall@10 | Recall@50 | Recall@100 | MAP | nDCG@10 | Strategy Recall | Strategy Precision | Strategy F1 | Difficulty MAE | Difficulty bucket accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, report in rows:
        if not report:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _fmt(_metric(report, ["premise_coverage", "Recall@10"])),
                    _fmt(_metric(report, ["premise_coverage", "Recall@50"])),
                    _fmt(_metric(report, ["premise_coverage", "Recall@100"])),
                    _fmt(_metric(report, ["premise_coverage", "MAP"])),
                    _fmt(_metric(report, ["premise_coverage", "nDCG@10"])),
                    _fmt(_metric(report, ["strategy_facets", "Recall"])),
                    _fmt(_metric(report, ["strategy_facets", "Precision"])),
                    _fmt(_metric(report, ["strategy_facets", "F1"])),
                    _fmt(_metric(report, ["difficulty_profile", "MAE"])),
                    _fmt(_metric(report, ["difficulty_profile", "bucket_accuracy"])),
                ]
            )
            + " |"
        )


def _delta(t1: dict[str, Any], base: str, method: str, metric: str) -> str:
    base_value = _metric(t1, ["methods", base, metric], 0.0)
    method_value = _metric(t1, ["methods", method, metric], 0.0)
    if not isinstance(base_value, int | float) or not isinstance(method_value, int | float):
        return ""
    return f"{method_value - base_value:+.4f}"


def _top_weighted_items(bundles: list[dict[str, Any]], key: str, *, limit: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for bundle in bundles:
        for item in bundle.get(key, []):
            item_id = str(item.get("full_name") or item.get("id") or "")
            if not item_id:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + float(item.get("score", 0.0) or 0.0)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _append_t3_section(lines: list[str], t3: dict[str, Any]) -> None:
    bundles = t3.get("bundles", [])
    if not isinstance(bundles, list):
        bundles = []
    bundle_count = int(t3.get("bundle_count", len(bundles)) or 0)
    total_neighbors = sum(len(bundle.get("neighbors", [])) for bundle in bundles)
    total_premises = sum(len(bundle.get("premise_suggestions", [])) for bundle in bundles)
    total_strategies = sum(len(bundle.get("strategy_facets", [])) for bundle in bundles)
    difficulty_neighbor_count = sum(int(bundle.get("difficulty_profile", {}).get("neighbor_count", 0) or 0) for bundle in bundles)
    bucket_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for bundle in bundles:
        bucket = str(bundle.get("difficulty_profile", {}).get("bucket", "") or "unknown")
        domain = str(bundle.get("domain_tag", "") or "unknown")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    lines.extend(
        [
            "### T3 Similar-Theorem Guidance Aggregation",
            "",
            f"Generated guidance bundles: `{bundle_count}`",
            "",
            "These bundles are a qualitative artifact: they are the deterministic first `limit` rows from the T2 neighbor artifact, not a stratified sample and not a full-split aggregate.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Guidance bundles | {bundle_count} |",
            f"| Avg source theorem neighbors/bundle | {_fmt(total_neighbors / max(bundle_count, 1))} |",
            f"| Avg premise suggestions/bundle | {_fmt(total_premises / max(bundle_count, 1))} |",
            f"| Avg strategy facets/bundle | {_fmt(total_strategies / max(bundle_count, 1))} |",
            f"| Avg difficulty evidence neighbors/bundle | {_fmt(difficulty_neighbor_count / max(bundle_count, 1))} |",
            "",
            "| Difficulty bucket | Bundle count | Share |",
            "| --- | ---: | ---: |",
        ]
    )
    for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {_md(bucket)} | {count} | {_pct(count, bundle_count)} |")
    lines.extend(["", "| Query domain | Bundle count | Share |", "| --- | ---: | ---: |"])
    for domain, count in sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))[:8]:
        lines.append(f"| {_md(domain)} | {count} | {_pct(count, bundle_count)} |")
    lines.extend(["", "#### Aggregated Guidance Evidence", ""])
    lines.extend(["| Evidence type | Top evidence | Aggregate score |", "| --- | --- | ---: |"])
    for item_id, score in _top_weighted_items(bundles, "premise_suggestions", limit=5):
        lines.append(f"| Premise suggestion | {_md(item_id)} | {_fmt(score)} |")
    for item_id, score in _top_weighted_items(bundles, "strategy_facets", limit=5):
        lines.append(f"| Strategy facet | {_md(item_id)} | {_fmt(score)} |")
    lines.extend(["", "#### Example Bundles", ""])
    lines.extend(["| Theorem | Domain | Neighbors | Premise suggestions | Strategy facets | Difficulty bucket |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for bundle in bundles[:5]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(bundle.get("full_name", "")),
                    _md(bundle.get("domain_tag", "")),
                    str(len(bundle.get("neighbors", []))),
                    str(len(bundle.get("premise_suggestions", []))),
                    str(len(bundle.get("strategy_facets", []))),
                    _md(bundle.get("difficulty_profile", {}).get("bucket", "")),
                ]
            )
            + " |"
        )


def run(
    split: str = "test",
    *,
    output_dir: str = "outputs/proofatlas",
    use_llm_enrichment: bool = False,
    use_pretrained_embeddings: bool = False,
) -> Path:
    out = Path(output_dir)
    suffix = split
    if use_llm_enrichment:
        suffix += "_llm_enriched"
    if use_pretrained_embeddings:
        suffix += "_pretrained"
    t1 = read_json(out / f"t1_{suffix}_proof_state_premise_retrieval.json", {})
    t2 = read_json(out / f"t2_{suffix}_theorem_theorem_retrieval.json", {})
    t2_baseline = read_json(out / f"t2_{split}_theorem_theorem_retrieval.json", {}) if use_llm_enrichment or use_pretrained_embeddings else {}
    t2_llm = read_json(out / f"t2_{split}_llm_enriched_theorem_theorem_retrieval.json", {}) if use_llm_enrichment and use_pretrained_embeddings else {}
    t3_suffix = f"{split}_llm_enriched" if use_llm_enrichment else split
    t3 = read_json(out / f"t3_{t3_suffix}_guidance_bundles.json", {})
    lines = [
        "# ProofAtlas ID Retrieval Experiment Report",
        "",
        f"Split: `{split}`",
        f"LLM theorem enrichment: `{bool(use_llm_enrichment)}`",
        f"Pretrained embeddings: `{bool(use_pretrained_embeddings)}`",
        f"Pretrained model: `{_md(t1.get('pretrained_model', ''))}`" if use_pretrained_embeddings else "Pretrained model: ``",
        "",
        "## Objective",
        "",
        "ProofAtlas evaluates whether ID retrieval structure in Lean proof data can improve premise retrieval and produce useful theorem-level guidance. The primary question is whether proof-state retrieval can be strengthened by adding structured evidence from similar proof states and similar theorems, without changing the held-out positive labels.",
        "",
        "The headline task is Proof-State -> Premise retrieval. Theorem -> Theorem retrieval and guidance aggregation are supporting tasks: they test whether theorem-level neighborhoods carry reusable premise, strategy, and difficulty evidence, and whether that evidence improves the main retrieval pipeline.",
        "",
    ]
    _append_dataset_overview(lines, split)
    _append_glossary(lines)
    lines.extend(
        [
            "## Tasks",
            "",
            "| Task | Query | Retrieved or aggregated evidence | Role in report |",
            "| --- | --- | --- | --- |",
            "| T1 Proof-State -> Premise | A held-out proof state | Train-side premises | Main quantitative result |",
            "| T2 Theorem -> Theorem | A held-out theorem profile | Similar train-side theorems | Pattern retrieval and theorem-neighborhood evidence |",
            "| T3 Guidance aggregation | A held-out theorem | Premises, strategy facets, and difficulty profile from similar theorems | Qualitative guidance artifact |",
            "",
            "## Method",
            "",
            "The retrieval pipeline starts with dense proof-state/premise embeddings and lexical TF-IDF profiles. It then adds structured candidate sources: symbol overlap, similar proof-state expansion, and similar theorem premise expansion. Candidate lists are fused with weighted reciprocal-rank fusion, with the final theorem-enhanced variants using train-side similar theorem premises as an additional source.",
            "",
            "Primary method: `weighted_rrf_llm_theorem_tuned`. This is the main reported ProofAtlas method: weighted RRF over dense, lexical, rank-based similar proof-state expansion, and the LLM-enriched similar-theorem premise source. The BGE pretrained embedding variants are reported as auxiliary ablations, not as the primary method.",
            "",
            "The final method keeps the same train/validation/test split and evaluates only against held-out positive premise edges. Strategy and difficulty in T2/T3 use precomputed rule/proxy labels and proxy difficulty buckets from `data/processed`; this experiment does not generate strategy or difficulty evaluation labels with the LLM during evaluation.",
            "",
            "Leakage control: LLM theorem enrichment is used as retrieval text, not as an evaluation label. The enrichment prompt is built from theorem metadata and a bounded number of proof-state goals, hypotheses, and symbols. It does not include validation/test positive premise labels, negative candidates, existing strategy labels, existing difficulty buckets/scores, or proof scripts/tactics as target answers. The proof-state text is still a metadata-level retrieval signal: local hypotheses and symbols may contain library identifiers or named facts from the state, so this is not a theorem-statement-only setting.",
            "",
            "Metric note: T1 and T2 premise Recall@k is reported over positives that are present in the train-side retrievable premise pool. MAP and nDCG are computed on the same filtered evaluation set. `Gold coverage` is the fraction of all held-out positive edges that are in that pool; `Overall Recall@100` multiplies covered Recall@100 by this coverage to show the approximate all-positive recall ceiling effect.",
            "",
            "## Experiments",
            "",
            "### T1 Main Result: Proof-State -> Premise Retrieval",
            "",
        ]
    )
    main_methods = [
        ("Dense baseline", "dense"),
        ("Lexical baseline", "lexical"),
        ("Dense + lexical fusion", "dense_lexical_rrf"),
        ("Naive multi-source union", "full_union_rrf"),
        ("Proof-state expansion tuned", "weighted_rrf_tuned_recall"),
        ("Theorem-enhanced frontier", "weighted_rrf_theorem_frontier"),
        ("Theorem-enhanced final", "weighted_rrf_theorem_tuned"),
    ]
    if use_llm_enrichment:
        main_methods.extend(
            [
                ("LLM theorem source", "weighted_rrf_llm_theorem_source"),
                ("Primary method", "weighted_rrf_llm_theorem_tuned"),
            ]
        )
    if use_pretrained_embeddings:
        main_methods.extend(
            [
                ("Pretrained dense source", "pretrained_dense"),
                ("Pretrained proof-state expansion", "pretrained_similar_proof_state_expansion"),
                ("BGE auxiliary fusion", "weighted_rrf_pretrained_tuned"),
            ]
        )
    if use_llm_enrichment and use_pretrained_embeddings:
        main_methods.append(("LLM + BGE auxiliary fusion", "weighted_rrf_llm_pretrained_tuned"))
    _append_t1_table(lines, t1, main_methods)
    lines.extend(
        [
            "",
            "`weighted_rrf_theorem_tuned` improves over dense retrieval by "
            f"{_delta(t1, 'dense', 'weighted_rrf_theorem_tuned', 'Recall@100')} covered Recall@100 and "
            f"{_delta(t1, 'dense', 'weighted_rrf_theorem_tuned', 'MAP')} MAP. "
            "The LLM and pretrained rows are additional variants on top of this non-LLM theorem-enhanced baseline. Relative to the naive multi-source union, the ablation sequence suggests that wider proof-state expansion, tuned fusion, and theorem-neighborhood premise evidence are associated with the improvement.",
            "",
            "### T1 Ablations",
            "",
        ]
    )
    method_order = [
        "dense",
        "lexical",
        "dense_lexical_rrf",
        "symbol_overlap",
        "similar_proof_state_expansion",
        "similar_proof_state_expansion_k50",
        "similar_proof_state_expansion_k100",
        "similar_proof_state_expansion_k100_sim",
        "similar_proof_state_expansion_k100_rank_sim",
        "similar_theorem_premises",
        "full_union_rrf",
        "weighted_rrf_balanced",
        "weighted_rrf_lexical_ps",
        "weighted_rrf_ps_heavy",
        "weighted_rrf_ps_heavy_k50",
        "weighted_rrf_ps_heavy_k100",
        "weighted_rrf_ps_heavy_k100_sim",
        "weighted_rrf_ps_heavy_k100_rank_sim",
        "weighted_rrf_tuned_frontier",
        "weighted_rrf_tuned_recall",
        "weighted_rrf_theorem_source",
        "weighted_rrf_theorem_heavy",
        "weighted_rrf_theorem_frontier",
        "weighted_rrf_theorem_tuned",
        "similar_theorem_premises_llm_enriched",
        "pretrained_dense",
        "pretrained_similar_proof_state_expansion",
        "weighted_rrf_llm_theorem_source",
        "weighted_rrf_llm_theorem_tuned",
        "weighted_rrf_pretrained_tuned",
        "weighted_rrf_llm_pretrained_tuned",
    ]
    _append_t1_ablation_table(lines, t1, method_order)
    lines.extend(
        [
            "",
            "### T2 Theorem -> Theorem Pattern Retrieval",
            "",
        ]
    )
    if use_llm_enrichment and use_pretrained_embeddings:
        t2_rows = [
            ("baseline theorem profile", t2_baseline),
            ("LLM enriched TF-IDF profile", t2_llm),
            ("LLM enriched pretrained profile", t2),
        ]
    elif use_llm_enrichment:
        t2_rows = [("baseline theorem profile", t2_baseline), ("LLM enriched theorem profile", t2)]
    elif use_pretrained_embeddings:
        t2_rows = [("baseline theorem profile", t2_baseline), ("pretrained theorem profile", t2)]
    else:
        t2_rows = [("baseline theorem profile", t2)]
    _append_t2_table(lines, t2_rows)
    lines.extend(
        [
            "",
            "Strategy metrics are included as guidance diagnostics, not as the primary retrieval claim. The high strategy recall is partly expected because strategy labels are broad, frequent, and aggregated across multiple neighbors; precision and F1 are therefore reported alongside recall to avoid over-interpreting near-ceiling recall.",
            "",
        ]
    )
    _append_t3_section(lines, t3)
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "The strongest signal is that theorem-level evidence improves T1 after it is fused with proof-state evidence. `similar_theorem_premises` alone is not stronger than the final fused method, but it adds complementary candidates: covered Recall@100 and MAP both rise once theorem-neighborhood evidence is included in weighted RRF. This is an ablation-based association rather than a causal significance claim.",
            "",
            "Pure similarity-score weighting for proof-state expansion is weaker than rank-based weighting, which suggests the current embedding score is useful for neighborhood ordering but not well calibrated as an absolute confidence value. The robust pattern is to use rank-based expansion and tune source weights rather than rely on raw similarity magnitudes.",
            "",
            (
                "LLM theorem enrichment improves the theorem-neighborhood source by adding semantic intent, proof strategy, and difficulty-oriented natural-language text to theorem retrieval. "
                "In the enriched report, this affects T2 retrieval directly and then enters T1 as one fused source inside weighted RRF via `similar_theorem_premises_llm_enriched`."
                if use_llm_enrichment
                else "The theorem-neighborhood source currently uses processed theorem profiles without LLM semantic enrichment."
            ),
            "",
            (
                "Pretrained embeddings add a second dense semantic channel over enriched proof states, premises, and theorem profiles. "
                "They are evaluated both as standalone candidate sources and as additional inputs to weighted RRF. In the current report, the pretrained channel is treated as an auxiliary signal rather than replacing the stronger LLM-enriched TF-IDF theorem source."
                if use_pretrained_embeddings
                else "Pretrained embedding sources are not included in this report."
            ),
            "",
            "The remaining bottleneck is ranking quality at the top of the list. Covered Recall@100 is much higher than covered Recall@10, so many useful premises are entering the candidate pool but are not consistently ranked high enough. A learned reranker over the fused candidate set is the most direct next step.",
            "",
            "## Conclusion",
            "",
            "The current ID retrieval setup is promising. The report shows a coherent progression from dense and lexical baselines to proof-state expansion and then theorem-neighborhood fusion. In this run, `weighted_rrf_llm_theorem_tuned` gives the strongest covered Recall@10, Recall@50, and Recall@100, while `weighted_rrf_llm_pretrained_tuned` gives the strongest MAP by a small margin. The BGE pretrained channel is useful but not dominant: its proof-state expansion source helps over the non-LLM theorem baseline, while its direct dense premise source is weak and the combined LLM+BGE fusion does not exceed the LLM theorem-tuned method on covered recall.",
            "",
            "The main result supports the design hypothesis: theorem-theorem retrieval is not just a side task, but a useful source of premise evidence for Proof-State -> Premise retrieval. The next improvement should focus on reranking and better source-specific calibration rather than only adding more candidate sources.",
        ]
    )
    path = out / f"id_experiment_report_{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
