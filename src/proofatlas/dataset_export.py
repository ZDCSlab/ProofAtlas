from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import SPLITS, write_json, write_parquet
from .llm_profiles import append_enrichment_text, enrichment_path, load_theorem_enrichment
from .profiles import theorem_profiles


PROCESSED_TABLES = [
    "file_modules",
    "negative_edges",
    "positive_edges",
    "premise_techniques",
    "premises",
    "proof_state_features",
    "proof_state_techniques",
    "proof_states",
    "proof_techniques",
    "theorem_features",
    "theorems",
]

HF_CONFIG_TABLES = [*PROCESSED_TABLES, "theorem_profiles"]
HF_SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


def _read_processed(processed_dir: Path, split: str, table: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / split / f"{table}.parquet")


def _prefix_llm_columns(enrichment: pd.DataFrame) -> pd.DataFrame:
    rename = {column: f"llm_{column}" for column in enrichment.columns if column != "theorem_id"}
    return enrichment.rename(columns=rename)


def _copy_processed_tables(processed_dir: Path, split: str, split_dir: Path) -> dict[str, int]:
    row_counts = {}
    for table in PROCESSED_TABLES:
        df = _read_processed(processed_dir, split, table)
        write_parquet(df, split_dir / f"{table}.parquet")
        row_counts[table] = int(len(df))
    return row_counts


def _write_enriched_theorems(
    *,
    processed_dir: Path,
    llm_output_dir: str,
    split: str,
    split_dir: Path,
) -> dict[str, Any]:
    theorems = _read_processed(processed_dir, split, "theorems").copy()
    features = _read_processed(processed_dir, split, "theorem_features").copy()
    enrichment = load_theorem_enrichment(split, llm_output_dir).copy()

    theorems["id"] = theorems["id"].astype(str)
    features["theorem_id"] = features["theorem_id"].astype(str)
    enrichment["theorem_id"] = enrichment["theorem_id"].astype(str)

    enriched = theorems.merge(features, left_on="id", right_on="theorem_id", how="left")
    if "theorem_id" in enriched.columns:
        enriched = enriched.drop(columns=["theorem_id"])
    enriched = enriched.merge(_prefix_llm_columns(enrichment), left_on="id", right_on="theorem_id", how="left")
    if "theorem_id" in enriched.columns:
        enriched = enriched.drop(columns=["theorem_id"])
    enriched["llm_enrichment_available"] = enriched["llm_topic"].notna()
    write_parquet(enriched, split_dir / "theorems.parquet")

    coverage = float(enriched["llm_enrichment_available"].mean()) if len(enriched) else 0.0
    return {
        "theorems": int(len(enriched)),
        "llm_enriched_theorems": int(enriched["llm_enrichment_available"].sum()),
        "llm_enrichment_coverage": coverage,
        "llm_models": sorted(str(value) for value in enrichment.get("model", pd.Series(dtype=object)).dropna().unique()),
        "llm_prompt_versions": sorted(str(value) for value in enrichment.get("prompt_version", pd.Series(dtype=object)).dropna().unique()),
    }


def _write_theorem_profiles(
    *,
    processed_dir: Path,
    llm_output_dir: str,
    split: str,
    split_dir: Path,
    max_states: int,
) -> None:
    theorems = _read_processed(processed_dir, split, "theorems")
    proof_states = _read_processed(processed_dir, split, "proof_states")
    profiles = theorem_profiles(theorems, proof_states, max_states=max_states)
    profiles = profiles.rename(columns={"profile_text": "base_profile_text"})
    profiles["profile_text"] = profiles["base_profile_text"]
    profiles = append_enrichment_text(profiles, split, llm_output_dir)

    enrichment = _prefix_llm_columns(load_theorem_enrichment(split, llm_output_dir))
    profiles = profiles.merge(enrichment, left_on="theorem_id", right_on="theorem_id", how="left")
    write_parquet(profiles, split_dir / "theorem_profiles.parquet")


def _dataset_card(version: str, manifest: dict[str, Any]) -> str:
    split_lines = []
    for split, stats in manifest["splits"].items():
        split_lines.append(
            f"| {split} | {stats['tables']['theorems']} | {stats['tables']['proof_states']} | "
            f"{stats['tables']['premises']} | {stats['tables']['positive_edges']} | "
            f"{stats['llm_enriched_theorems']} | {stats['llm_enrichment_coverage']:.1%} |"
        )
    return "\n".join(
        [
            f"# ProofAtlas Enriched Dataset {version}",
            "",
            "ProofAtlas Enriched combines theorem-disjoint Lean retrieval splits with LLM-generated theorem semantic and strategy enrichment. It is designed for premise-retrieval research, theorem-neighborhood retrieval, and qualitative retrieval-evidence analysis.",
            "",
            "## Task Definition",
            "",
            "The main challenge task is proof-state to premise retrieval: given a held-out Lean proof state, retrieve useful premises from the train-side premise pool. For the test split, this means retrieving from 127,561 train-side candidate premises for 3,053 held-out proof states and 7,054 held-out positive premise edges.",
            "",
            "The theorem-neighborhood task retrieves similar train-side theorem profiles for a held-out theorem profile. Retrieved theorem neighbors can then be expanded into the premises used in their proof states, giving neighbor-derived premise evidence.",
            "",
            "## Contents",
            "",
            "Each split contains the processed ProofAtlas parquet tables plus:",
            "",
            "- `theorems.parquet`: original theorem metadata joined with theorem difficulty features and `llm_*` enrichment columns.",
            "- `theorem_profiles.parquet`: retrieval-ready theorem profile text with both base processed context and LLM enrichment text.",
            "- `proof_states.parquet`: proof-state goals, local hypotheses, symbols, tactics, and theorem links.",
            "- `premises.parquet`: train/validation/test-side premise declarations and code text.",
            "- `positive_edges.parquet`: proof-state to premise labels used for retrieval evaluation.",
            "- `proof_state_techniques.parquet` and `proof_techniques.parquet`: broad proof-strategy facet labels.",
            "",
            "The LLM fields are retrieval features, not evaluation labels. Held-out premise positives remain unchanged in `positive_edges.parquet`.",
            "",
            "## Split Summary",
            "",
            "| Split | Theorems | Proof states | Premises | Positive edges | LLM enriched theorems | LLM coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *split_lines,
            "",
            "## LLM Provenance",
            "",
            f"- Models: {', '.join(manifest['llm_models'])}",
            f"- Prompt versions: {', '.join(manifest['llm_prompt_versions'])}",
            "- Enrichment fields use the `llm_` prefix in merged tables.",
            "- LLM enrichment is generated from theorem metadata and bounded proof-state context. It is retrieval text only; it does not provide gold premise labels, proof scripts, or target answers.",
            "",
            "## Proof Strategy Facets",
            "",
            "The strategy fields use 15 broad, proxy proof-pattern labels: `algebraic_computation`, `case_analysis`, `category_morphism_reasoning`, `contradiction_negation`, `existential_construction`, `extensionality`, `induction_recursion`, `measure_ae_reasoning`, `order_inequality_reasoning`, `rewrite_transport`, `set_membership_reasoning`, `simplification_normalization`, `theorem_application`, `topology_filter_limit`, and `typeclass_instance_resolution`.",
            "",
            "## Intended Use",
            "",
            "- Proof-state to premise retrieval.",
            "- Theorem-neighborhood retrieval and neighbor-derived premise evidence experiments.",
            "- Retrieval model evaluation on Lean theorem/proof-state text.",
            "- Qualitative retrieval-evidence views using theorem neighbors, premise suggestions, strategy facets, and difficulty signals.",
            "",
            "## Evaluation Notes and Limitations",
            "",
            "- Splits are theorem-disjoint, but they are in-distribution; namespace, domain, symbol, and vocabulary overlap are expected.",
            "- Covered Recall is computed over gold premises that are present in the retrievable train-side premise pool.",
            "- All-positive Recall is computed over all held-out positive premise edges, including positives absent from the train-side pool.",
            "- Strategy and difficulty fields are proxy signals and should not be treated as human-verified ground truth.",
            "- LLM enrichment fields are model-generated and may contain abstraction or interpretation errors.",
            "",
        ]
    )


def _hf_readme(version: str, manifest: dict[str, Any]) -> str:
    configs = []
    for table in HF_CONFIG_TABLES:
        configs.extend(
            [
                f"- config_name: {table}",
                "  data_files:",
            ]
        )
        for split, hf_split in HF_SPLIT_NAMES.items():
            configs.extend(
                [
                    f"  - split: {hf_split}",
                    f"    path: hf/{table}/{hf_split}.parquet",
                ]
            )
    split_lines = []
    for split, stats in manifest["splits"].items():
        split_lines.append(
            f"| {split} | {stats['tables']['theorems']} | {stats['tables']['proof_states']} | "
            f"{stats['tables']['premises']} | {stats['tables']['positive_edges']} | "
            f"{stats['llm_enriched_theorems']} | {stats['llm_enrichment_coverage']:.1%} |"
        )
    return "\n".join(
        [
            "---",
            "pretty_name: ProofAtlas Enriched",
            "language:",
            "- en",
            "task_categories:",
            "- feature-extraction",
            "- text-retrieval",
            "tags:",
            "- lean",
            "- mathlib",
            "- theorem-proving",
            "- premise-retrieval",
            "- retrieval-augmented-generation",
            "size_categories:",
            "- 1M<n<10M",
            "license: apache-2.0",
            "configs:",
            *configs,
            "---",
            "",
            f"# ProofAtlas Enriched Dataset {version}",
            "",
            "ProofAtlas Enriched combines theorem-disjoint Lean retrieval splits with LLM-generated theorem semantic and strategy enrichment. It is designed for premise-retrieval research, theorem-neighborhood retrieval, and qualitative retrieval-evidence analysis.",
            "",
            "## Source Data",
            "",
            "This dataset is derived from [`erbacher/LeanRank-data`](https://huggingface.co/datasets/erbacher/LeanRank-data), which is distributed under the Apache-2.0 license. The upstream LeanRank data was extracted from [`mathlib4`](https://github.com/leanprover-community/mathlib4) using [`LeanDojo`](https://github.com/lean-dojo/LeanDojo), at mathlib4 commit `c211948581bde9846a99e32d97a03f0d5307c31e`.",
            "",
            "ProofAtlas adds processed retrieval tables, theorem-disjoint ID splits, proxy strategy/difficulty features, and DeepSeek-generated theorem semantic/strategy enrichment.",
            "",
            "## Task Definition",
            "",
            "The main challenge task is proof-state to premise retrieval: given a held-out Lean proof state, retrieve useful premises from the train-side premise pool. For the test split, this means retrieving from 127,561 train-side candidate premises for 3,053 held-out proof states and 7,054 held-out positive premise edges.",
            "",
            "The theorem-neighborhood task retrieves similar train-side theorem profiles for a held-out theorem profile. Retrieved theorem neighbors can then be expanded into the premises used in their proof states, giving neighbor-derived premise evidence.",
            "",
            "## Loading",
            "",
            "Each table is exposed as a separate HuggingFace dataset config because the tables have different schemas:",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            'theorems = load_dataset("YOUR_ORG/proofatlas-enriched", "theorems")',
            'profiles = load_dataset("YOUR_ORG/proofatlas-enriched", "theorem_profiles")',
            'proof_states = load_dataset("YOUR_ORG/proofatlas-enriched", "proof_states")',
            'positive_edges = load_dataset("YOUR_ORG/proofatlas-enriched", "positive_edges")',
            "```",
            "",
            "The available configs are:",
            "",
            ", ".join(f"`{table}`" for table in HF_CONFIG_TABLES),
            "",
            "## Dataset Structure",
            "",
            "- `theorems`: theorem metadata joined with proxy difficulty features and `llm_*` enrichment columns.",
            "- `theorem_profiles`: retrieval-ready theorem profile text with base processed context and LLM enrichment text.",
            "- `proof_states`: proof-state contexts, goals, symbols, tactics, and theorem links.",
            "- `premises`: candidate premise declarations and code text.",
            "- `positive_edges`: held-out proof-state to premise labels for evaluation.",
            "- `negative_edges`: processed negative candidates.",
            "- `proof_state_techniques` and `proof_techniques`: broad proof-strategy facet labels.",
            "- `theorem_features` and `proof_state_features`: proxy difficulty and structural features.",
            "- remaining configs contain processed metadata and feature tables.",
            "",
            "## Split Summary",
            "",
            "| Split | Theorems | Proof states | Premises | Positive edges | LLM enriched theorems | LLM coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *split_lines,
            "",
            "## LLM Provenance",
            "",
            f"- Models: {', '.join(manifest['llm_models'])}",
            f"- Prompt versions: {', '.join(manifest['llm_prompt_versions'])}",
            "- LLM fields are retrieval features, not evaluation labels.",
            "- Held-out premise positives remain unchanged in `positive_edges`.",
            "- LLM enrichment is generated from theorem metadata and bounded proof-state context. It does not provide gold premise labels, proof scripts, or target answers.",
            "",
            "## Proof Strategy Facets",
            "",
            "The strategy fields use 15 broad, proxy proof-pattern labels:",
            "",
            "- `algebraic_computation`: ring, field, arithmetic, and normalization-style algebraic calculation.",
            "- `case_analysis`: splitting the proof into constructors, alternatives, or conditional cases.",
            "- `category_morphism_reasoning`: reasoning about categorical morphisms, composition, identities, and commutative diagrams.",
            "- `contradiction_negation`: using negation, contradiction, impossible hypotheses, or proof by contradiction.",
            "- `existential_construction`: building witnesses for existential goals or packaging data into structures.",
            "- `extensionality`: proving equality by pointwise, componentwise, or structural equality.",
            "- `induction_recursion`: using induction, recursion, or structural decomposition over inductive objects.",
            "- `measure_ae_reasoning`: measure-theoretic reasoning with almost-everywhere statements and measurable sets.",
            "- `order_inequality_reasoning`: manipulating inequalities, monotonicity, bounds, lattice/order facts, or comparisons.",
            "- `rewrite_transport`: rewriting goals across equalities, equivalences, casts, coercions, or transported structures.",
            "- `set_membership_reasoning`: reasoning about set membership, subsets, intersections, unions, images, and preimages.",
            "- `simplification_normalization`: simplifying definitions, canonical forms, coercions, and routine goals.",
            "- `theorem_application`: solving a goal mainly by applying a known lemma, theorem, or hypothesis.",
            "- `topology_filter_limit`: reasoning about continuity, filters, neighborhoods, convergence, and limits.",
            "- `typeclass_instance_resolution`: using or constructing typeclass instances and inherited algebraic/order/topological structure.",
            "",
            "## Intended Uses",
            "",
            "- Proof-state to premise retrieval.",
            "- Theorem-neighborhood retrieval and neighbor-derived premise evidence experiments.",
            "- Qualitative retrieval-evidence views using theorem neighbors, premise suggestions, strategy facets, and difficulty signals.",
            "- Embedding model evaluation on Lean theorem/proof-state text.",
            "",
            "## Evaluation Notes",
            "",
            "- Covered Recall is computed over gold premises that are present in the retrievable train-side premise pool.",
            "- All-positive Recall is computed over all held-out positive premise edges, including positives absent from the train-side pool.",
            "- Recall@100 should be read as a candidate-generation metric: it measures whether useful premises enter the top 100 retrieved candidates.",
            "",
            "## Limitations",
            "",
            "- The split is theorem-disjoint but in-distribution; namespace, domain, and vocabulary overlap are expected.",
            "- Strategy and difficulty fields are proxy signals and should not be treated as human-verified ground truth.",
            "- LLM enrichment fields are model-generated and may contain abstraction or interpretation errors.",
            "",
        ]
    )


def _write_hf_layout(output_dir: Path, manifest: dict[str, Any]) -> None:
    hf_root = output_dir / "hf"
    for table in HF_CONFIG_TABLES:
        table_dir = hf_root / table
        table_dir.mkdir(parents=True, exist_ok=True)
        for split, hf_split in HF_SPLIT_NAMES.items():
            df = pd.read_parquet(output_dir / split / f"{table}.parquet")
            write_parquet(df, table_dir / f"{hf_split}.parquet")
    (output_dir / "README.md").write_text(_hf_readme(str(manifest["version"]), manifest), encoding="utf-8")


def run(
    *,
    version: str = "v1",
    processed_dir: str = "data/processed",
    llm_output_dir: str = "outputs/proofatlas",
    output_root: str = "data/proofatlas_enriched",
    max_states: int = 5,
) -> dict[str, Any]:
    processed_path = Path(processed_dir)
    output_dir = Path(output_root) / version
    manifest: dict[str, Any] = {
        "name": "proofatlas_enriched",
        "version": version,
        "processed_dir": str(processed_path),
        "llm_output_dir": str(llm_output_dir),
        "max_profile_states": int(max_states),
        "splits": {},
        "llm_models": [],
        "llm_prompt_versions": [],
    }

    for split in SPLITS:
        if not enrichment_path(split, llm_output_dir).exists():
            raise FileNotFoundError(f"Missing LLM enrichment for split `{split}`: {enrichment_path(split, llm_output_dir)}")
        split_dir = output_dir / split
        table_counts = _copy_processed_tables(processed_path, split, split_dir)
        split_stats = _write_enriched_theorems(
            processed_dir=processed_path,
            llm_output_dir=llm_output_dir,
            split=split,
            split_dir=split_dir,
        )
        _write_theorem_profiles(
            processed_dir=processed_path,
            llm_output_dir=llm_output_dir,
            split=split,
            split_dir=split_dir,
            max_states=max_states,
        )
        manifest["splits"][split] = {"tables": table_counts, **split_stats}

    manifest["llm_models"] = sorted({model for stats in manifest["splits"].values() for model in stats["llm_models"]})
    manifest["llm_prompt_versions"] = sorted(
        {version for stats in manifest["splits"].values() for version in stats["llm_prompt_versions"]}
    )
    manifest["hf_configs"] = HF_CONFIG_TABLES
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "dataset_card.md").write_text(_dataset_card(version, manifest), encoding="utf-8")
    _write_hf_layout(output_dir, manifest)
    return manifest | {"output_dir": str(output_dir)}
