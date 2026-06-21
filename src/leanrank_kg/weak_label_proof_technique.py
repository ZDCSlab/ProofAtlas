from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .utils import SPLITS, load_config, technique_id, write_json, write_parquet


@dataclass(frozen=True)
class StrategyRule:
    label: str
    description: str
    evidence_source: str
    patterns: tuple[str, ...]
    confidence: float
    examples: tuple[str, ...]


STRATEGY_RULES: tuple[StrategyRule, ...] = (
    StrategyRule(
        label="rewrite_transport",
        description="Transform the goal by equality, equivalence, coercion, congruence, or transport across definitional wrappers.",
        evidence_source="goal_or_statement_shape",
        patterns=(r"\b(rw|rewrite|rfl|congr|congrArg|subst)\b", r"↔", r"=", r"\bcoe\b", r"\bcast\b", r"\bmap\b", r"\bcomp\b"),
        confidence=0.72,
        examples=("rw", "congr", "coercion/cast", "iff/equality goal"),
    ),
    StrategyRule(
        label="simplification_normalization",
        description="Normalize expressions by simplification, definitional unfolding, canonical simp lemmas, or local normal forms.",
        evidence_source="lemma_name_or_statement",
        patterns=(r"\bsimp(?:_|$|\b)", r"@\[simp\]", r"\bsimp_", r"_simp\b", r"\bnormalize\b", r"\bmk_\b", r"\bto[A-Z]"),
        confidence=0.70,
        examples=("simp", "@[simp]", "normal form lemma"),
    ),
    StrategyRule(
        label="algebraic_computation",
        description="Close algebraic, numeric, ring, semiring, cardinality, or arithmetic side goals by computation or algebraic normalization.",
        evidence_source="goal_symbols_and_names",
        patterns=(r"\bnorm_num\b", r"\bring\b", r"\bomega\b", r"\blinarith\b", r"\bnlinarith\b", r"\bdecide\b", r"[+*/^]", r"\bNat\b", r"\bInt\b", r"\bFinset\b"),
        confidence=0.68,
        examples=("ring/norm_num style", "arithmetic operators", "finite sums/products"),
    ),
    StrategyRule(
        label="order_inequality_reasoning",
        description="Reason with inequalities, monotonicity, order bounds, positivity, divisibility, or modular/order relations.",
        evidence_source="goal_symbols",
        patterns=(r"≤", r"≥", r"<", r">", r"∣", r"\bMOD\b", r"\bPMOD\b", r"\bMonotone\b", r"\bStrictMono\b", r"\bNonneg\b", r"\bPositive\b"),
        confidence=0.74,
        examples=("≤/< goal", "divisibility/modEq", "monotonicity"),
    ),
    StrategyRule(
        label="typeclass_instance_resolution",
        description="Resolve or exploit typeclass structure such as algebraic hierarchy, category instances, measurability spaces, or topology instances.",
        evidence_source="context_typeclass_bindings",
        patterns=(r"\binst[^\s:]*\s*:", r"\bField\b", r"\bRing\b", r"\bCategory\b", r"\bTopologicalSpace\b", r"\bMeasurableSpace\b", r"\bNormedField\b"),
        confidence=0.66,
        examples=("instance hypothesis", "algebra/category/topology class"),
    ),
    StrategyRule(
        label="case_analysis",
        description="Split on constructors, alternatives, decidability branches, quotient representatives, or named proof cases.",
        evidence_source="context_case_marker",
        patterns=(r"^case\s+", r"\bcase\s+", r"\bcases\b", r"\bby_cases\b", r"\bconstructor\b", r"\binl\b", r"\binr\b", r"\bOr\b", r"∨"),
        confidence=0.69,
        examples=("case ...", "constructor", "or-branch"),
    ),
    StrategyRule(
        label="induction_recursion",
        description="Use induction, recursion principles, structural recursion, list/tree decomposition, or recursor-generated cases.",
        evidence_source="name_context_or_statement",
        patterns=(r"\binduction\b", r"\brec(?:_on|On)?\b", r"\bcasesOn\b", r"\bNat\.rec\b", r"\bList\b", r"\bTree\b", r"\bPath\b"),
        confidence=0.66,
        examples=("induction", "rec_on", "recursive datatype"),
    ),
    StrategyRule(
        label="extensionality",
        description="Prove equality of structured objects by reducing to pointwise or componentwise equality.",
        evidence_source="goal_or_name",
        patterns=(r"\bext\b", r"\bextensionality\b", r"\bfunext\b", r"\bDFunLike\.ext\b", r"\bhom_ext\b", r"\bext_iff\b", r"\bpointwise\b"),
        confidence=0.78,
        examples=("ext", "funext", "hom_ext"),
    ),
    StrategyRule(
        label="theorem_application",
        description="Apply an existing theorem, implication, iff direction, or named lemma to reduce the current goal.",
        evidence_source="statement_connectives",
        patterns=(r"\bapply\b", r"\bexact\b", r"\brefine\b", r"→", r"\b_of_", r"\biff\b", r"_iff_", r"\bIs[A-Z]\w+"),
        confidence=0.62,
        examples=("apply/exact/refine", "_of_ lemma", "iff lemma"),
    ),
    StrategyRule(
        label="existential_construction",
        description="Construct witnesses and prove conjunction/subset/range side conditions for existential goals.",
        evidence_source="goal_shape",
        patterns=(r"∃", r"\bExists\b", r"\brange\b", r"⊆", r"∧", r"\bSubtype\b", r"\bNonempty\b"),
        confidence=0.72,
        examples=("exists goal", "witness plus side conditions", "subset/range"),
    ),
    StrategyRule(
        label="contradiction_negation",
        description="Use negation, contradiction, impossible hypotheses, non-membership, or proof by contradiction.",
        evidence_source="goal_or_context_shape",
        patterns=(r"¬", r"\bFalse\b", r"\bcontradiction\b", r"\bby_contra\b", r"\bnot_", r"\bNe\b", r"≠"),
        confidence=0.72,
        examples=("False/¬ goal", "by_contra", "not/Ne lemma"),
    ),
    StrategyRule(
        label="set_membership_reasoning",
        description="Reason about membership, subset, image/preimage, range, union/intersection, filters, or local set transformations.",
        evidence_source="goal_symbols_and_names",
        patterns=(r"∈", r"⊆", r"∪", r"∩", r"\bSet\.", r"\bmem_", r"_mem\b", r"\bimage\b", r"\bpreimage\b", r"\brange\b"),
        confidence=0.70,
        examples=("x ∈ s", "subset", "image/preimage"),
    ),
    StrategyRule(
        label="topology_filter_limit",
        description="Use continuity, neighborhoods, filters, closure, convergence, compactness, or topological local reasoning.",
        evidence_source="domain_specific_goal",
        patterns=(r"\bContinuous", r"\bTendsto\b", r"\bFilter\b", r"𝓝", r"\bclosure\b", r"\bIsClosed\b", r"\bIsOpen\b", r"\bCompact\b"),
        confidence=0.73,
        examples=("ContinuousAt", "Filter.map", "closure"),
    ),
    StrategyRule(
        label="measure_ae_reasoning",
        description="Use measurability, almost-everywhere facts, integrability, measure restrictions, or measurable selection structure.",
        evidence_source="domain_specific_goal",
        patterns=(r"\bMeasurable\b", r"\bAEMeasurable\b", r"=ᶠ", r"\bae\b", r"\bMeasure\b", r"\bIntegrable\b", r"\bNullMeasurable\b"),
        confidence=0.73,
        examples=("AEMeasurable", "ae", "measure/integrable"),
    ),
    StrategyRule(
        label="category_morphism_reasoning",
        description="Reason with categorical morphism composition, naturality, isomorphisms, functors, limits, or commutative diagrams.",
        evidence_source="domain_specific_goal",
        patterns=(r"≫", r"\bCategoryTheory\b", r"\bFunctor\b", r"\bNatural\b", r"\bIsIso\b", r"\bLimits\b", r"\bhom\b"),
        confidence=0.73,
        examples=("≫ composition", "functor/naturality", "isIso/limits"),
    ),
)


def _match_rule(rule: StrategyRule, text: str) -> dict[str, Any] | None:
    hits = []
    for pattern in rule.patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(pattern)
    if not hits:
        return None
    confidence = min(0.98, rule.confidence + 0.04 * min(len(hits) - 1, 3))
    return {
        "label": rule.label,
        "provenance": f"{rule.evidence_source}:{';'.join(hits[:3])}",
        "confidence": float(confidence),
        "evidence_source": rule.evidence_source,
    }


def labels_for_text(text: str, max_labels: int = 5) -> list[dict[str, Any]]:
    text = text or ""
    found = []
    for rule in STRATEGY_RULES:
        hit = _match_rule(rule, text)
        if hit:
            found.append(hit)
    found.sort(key=lambda row: (-float(row["confidence"]), row["label"]))
    return found[:max_labels]


def _labels_for_proof_state(row: dict[str, Any], max_labels: int) -> list[dict[str, Any]]:
    text = "\n".join(
        [
            str(row.get("full_name", "")),
            str(row.get("goal_text", "")),
            str(row.get("context", "")),
            str(row.get("tactic", "")),
        ]
    )
    return labels_for_text(text, max_labels)


def _labels_for_premise(row: dict[str, Any], max_labels: int) -> list[dict[str, Any]]:
    text = "\n".join([str(row.get("full_name", "")), str(row.get("code", "")), str(row.get("domain_tag", ""))])
    return labels_for_text(text, max_labels)


def _candidate_pool() -> list[dict[str, Any]]:
    return [
        {
            "label": rule.label,
            "description": rule.description,
            "evidence_source": rule.evidence_source,
            "confidence": rule.confidence,
            "patterns": list(rule.patterns),
            "examples": list(rule.examples),
        }
        for rule in STRATEGY_RULES
    ]


def run(config_path: str) -> None:
    config = load_config(config_path)
    max_labels = int((config.get("proof_techniques") or {}).get("max_labels_per_state", 5))
    write_json(
        "outputs/reports/proof_technique_candidate_pool.json",
        {
            "taxonomy": "proof_state_strategy_facets_v2",
            "note": "Curated strategy facets inferred from theorem names, goal shape, context markers, and statement symbols for retrieval-grounded proof guidance.",
            "labels": _candidate_pool(),
        },
    )
    distributions = []
    provenance = []
    for split in SPLITS + ["demo"]:
        try:
            ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
            prem = pd.read_parquet(f"data/processed/{split}/premises.parquet")
        except FileNotFoundError:
            continue
        ps_rows = []
        for row in ps.to_dict(orient="records"):
            for lab in _labels_for_proof_state(row, max_labels):
                ps_rows.append({"proof_state_id": row["id"], "technique_id": technique_id(lab["label"]), **lab})
                provenance.append({"split": split, "entity_type": "proof_state", **lab})
        prem_rows = []
        for row in prem.to_dict(orient="records"):
            for lab in _labels_for_premise(row, max_labels):
                prem_rows.append({"premise_id": row["id"], "technique_id": technique_id(lab["label"]), **lab})
                provenance.append({"split": split, "entity_type": "premise", **lab})
        ps_df = pd.DataFrame(
            ps_rows,
            columns=["proof_state_id", "technique_id", "label", "provenance", "confidence", "evidence_source"],
        ).drop_duplicates(["proof_state_id", "technique_id"])
        prem_df = pd.DataFrame(
            prem_rows,
            columns=["premise_id", "technique_id", "label", "provenance", "confidence", "evidence_source"],
        ).drop_duplicates(["premise_id", "technique_id"])
        combined = pd.concat(
            [
                ps_df[["technique_id", "label", "provenance", "confidence", "evidence_source"]]
                if not ps_df.empty
                else pd.DataFrame(columns=["technique_id", "label", "provenance", "confidence", "evidence_source"]),
                prem_df[["technique_id", "label", "provenance", "confidence", "evidence_source"]]
                if not prem_df.empty
                else pd.DataFrame(columns=["technique_id", "label", "provenance", "confidence", "evidence_source"]),
            ],
            ignore_index=True,
        )
        technique_rows = []
        for label, group in combined.groupby("label"):
            technique_rows.append(
                {
                    "id": technique_id(label),
                    "label": label,
                    "provenance": ",".join(sorted(set(group["evidence_source"].dropna().astype(str)))),
                }
            )
        technique_df = pd.DataFrame(technique_rows, columns=["id", "label", "provenance"]).drop_duplicates("id")
        write_parquet(ps_df, f"data/processed/{split}/proof_state_techniques.parquet")
        write_parquet(prem_df, f"data/processed/{split}/premise_techniques.parquet")
        write_parquet(technique_df, f"data/processed/{split}/proof_techniques.parquet")
        if not ps_df.empty:
            counts = ps_df["label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            counts["split"] = split
            distributions.append(counts)
    dist = pd.concat(distributions, ignore_index=True) if distributions else pd.DataFrame(columns=["label", "count", "split"])
    dist.to_csv("outputs/reports/proof_technique_distribution.csv", index=False)
    pd.DataFrame(provenance).to_csv("outputs/reports/proof_technique_label_provenance.csv", index=False)
