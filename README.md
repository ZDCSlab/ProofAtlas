# ProofAtlas

ProofAtlas is a Lean/mathlib premise-retrieval project built on processed LeanRank artifacts. It evaluates whether theorem-neighborhood structure and LLM-enriched theorem profiles improve retrieval for held-out proof states.

## Overview

ProofAtlas focuses on three tasks:

1. **Proof-state -> Premise retrieval**: retrieve useful train-side premises for held-out proof states.
2. **Theorem -> Theorem pattern retrieval**: retrieve similar train-side theorem profiles for held-out theorem profiles.
3. **Guidance aggregation**: aggregate premise suggestions, strategy facets, and proxy difficulty evidence from similar theorems.

The split is theorem-disjoint and in-distribution. File, domain, namespace, and vocabulary overlap are expected.

## Dataset

The enriched dataset is available on HuggingFace, and the validation report is tracked in this repository:

- [ZDCSlab/proofatlas-enriched](https://huggingface.co/datasets/ZDCSlab/proofatlas-enriched)
- [ProofAtlas Project Report](outputs/reports/ProofAtlas_Project_Report.md)

Local data files should be placed under `data/`. The entire `data/` directory is ignored by git and is used for downloaded, generated, or exported dataset artifacts.

The exported enriched dataset layout is:

```text
data/proofatlas_enriched/v1/
```

It contains theorem-disjoint train/validation/test parquet tables, enriched `theorems.parquet` files, retrieval-ready `theorem_profiles.parquet` files, `manifest.json`, and `dataset_card.md`.

## Installation

ProofAtlas requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Equivalent Make target:

```bash
make install
```

## Quick Start

Run the focused in-distribution pipeline on the test split:

```bash
proofatlas id-pipeline --split test --neighbor-k 20 --guidance-limit 25
```

Equivalent module invocation:

```bash
python -m proofatlas.cli id-pipeline --split test --neighbor-k 20 --guidance-limit 25
```

Equivalent Make target:

```bash
make id-pipeline
```

## Commands

```bash
proofatlas evaluate-t1 --split test
proofatlas evaluate-t2 --split test --neighbor-k 20
proofatlas aggregate-guidance --split test --limit 25
proofatlas build-report --split test
proofatlas id-pipeline --split test
```

For validation experiments, replace `test` with `val`.

## LLM Enrichment

Optional LLM experiments use the DeepSeek chat-completions API. Put the API key in `.env`:

```bash
DEEPSEEK_API_KEY=your_key_here
```

Run a small theorem-enrichment pilot:

```bash
proofatlas llm-enrich-theorems --split train --limit 100 --concurrency 8
```

Run full theorem enrichment for train, validation, and test:

```bash
proofatlas llm-pipeline --concurrency 8
```

Run enriched retrieval:

```bash
proofatlas evaluate-t2 --split val --use-llm-enrichment
proofatlas evaluate-t1 --split val --use-llm-enrichment
proofatlas id-pipeline --split test --use-llm-enrichment
```

Export the enriched dataset artifact:

```bash
proofatlas export-enriched-dataset --version v1
```

LLM outputs and per-item caches are written under:

```text
outputs/proofatlas/llm/
```

## Outputs

Pipeline outputs are written to:

```text
outputs/proofatlas/
```

Common output files include:

- `t1_test_proof_state_premise_retrieval.json`
- `t1_test_candidate_rankings.parquet`
- `t2_test_theorem_theorem_retrieval.json`
- `t2_test_theorem_neighbors.parquet`
- `t3_test_guidance_bundles.json`
- `id_experiment_report_test.md`

The curated validation report is tracked at:

```text
outputs/reports/ProofAtlas_Project_Report.md
```

Other generated outputs under `outputs/` are local artifacts and are ignored unless explicitly whitelisted.

## Evaluation

The main retrieval metrics are Recall@10, Recall@100, MAP, and nDCG@10. Recall@100 is used as a candidate-generation metric: it measures whether useful premises enter a manageable downstream candidate pool. Recall@10, MAP, and nDCG@10 measure top-rank quality.

Covered recall is computed over positives whose premise is present in the train-side retrievable premise pool. Overall Recall@100 additionally accounts for gold coverage.

## Development

Run tests with:

```bash
pytest -q
```

or:

```bash
make test
```

The package entry point is:

```text
proofatlas = proofatlas.cli:app
```

## Notes

ProofAtlas builds theorem profiles from theorem metadata and proof-state text for clean theorem-theorem retrieval.
