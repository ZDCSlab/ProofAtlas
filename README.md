# ProofAtlas

ProofAtlas is a Lean/mathlib premise-retrieval project built on processed LeanRank artifacts. This repository is currently framed as a midterm report: it documents the data pipeline, theorem-neighborhood retrieval system, qualitative guidance views, and a hard proof-state premise retrieval challenge.

- Homepage: <https://zdcslab.github.io/ProofAtlas/>
- Enriched dataset: <https://huggingface.co/datasets/ZDCSlab/proofatlas-enriched>
- Full report: [outputs/reports/ProofAtlas_Project_Report.md](outputs/reports/ProofAtlas_Project_Report.md)

## Executive Summary

The midterm contribution is best read as a capability build-out first, then a challenge benchmark:

1. ProofAtlas builds a theorem-neighborhood retrieval pipeline over a theorem-disjoint Lean proof split.
2. LLM-enriched theorem profiles improve theorem-neighbor premise evidence.
3. ProofAtlas exposes qualitative theorem-neighborhood guidance views with premise suggestions and strategy/difficulty facets.
4. On the harder proof-state -> premise challenge, candidate recall roughly doubles over dense retrieval, while top-rank precision remains the next bottleneck.

## Midterm Milestones

### Milestone 1: Data Pipeline

- theorem-disjoint Lean proof split
- 127,561 train-side premise pool
- 3,053 held-out proof states
- 7,054 gold premise edges

### Milestone 2: Retrieval System

- dense + lexical baseline
- similar proof-state expansion
- LLM-enriched theorem-neighborhood evidence
- weighted RRF fusion

### Milestone 3: Midterm Result

- Covered Recall@100 +0.2384
- All-positive Recall@100 +0.2223
- nDCG@10 +0.0539

### Milestone 4: Interpretability and Guidance

- theorem neighborhood views
- premise suggestions
- strategy/difficulty facets

## Terms and Abbreviations

- Lean: a theorem prover and proof assistant used to formalize mathematics.
- mathlib: the main mathematical library for Lean.
- LeanRank: the processed Lean proof dataset source used by this experiment.
- Proof state: the current goal, hypotheses, and local context at a point in a Lean proof.
- Premise: a theorem, lemma, definition, or fact that may help prove a proof state.
- Gold premise edge: a held-out labeled link between a proof state and a premise used in the reference proof.
- Theorem-disjoint split: a train/test split where test theorems do not appear in training.
- Train-side premise pool: the candidate premises available for retrieval from the training side.
- Theorem neighborhood: similar train theorems retrieved for a held-out theorem or proof state.
- LLM: large language model; in ProofAtlas it produces retrieval text, not evaluation labels.
- DeepSeek: the chat-completions provider used by the optional LLM enrichment commands.
- API: application programming interface; here it refers to the service interface used for optional LLM enrichment.
- TF-IDF: term frequency-inverse document frequency, a lexical text retrieval representation.
- BGE: an external pretrained embedding model used as an auxiliary variant; the exact model name is recorded in the report header.
- Dense retrieval: nearest-neighbor retrieval using vector embeddings.
- Lexical retrieval: retrieval based on token overlap or weighted text features such as TF-IDF.
- RRF: reciprocal-rank fusion, a method for combining ranked candidate lists.
- Recall@k: the fraction of relevant gold items found within the top `k` retrieved candidates.
- R@k: short notation for Recall@k, such as R@10 or R@100.
- Covered Recall: recall over gold premises that are present in the retrievable train-side premise pool.
- All-positive Recall: recall over all held-out positive premise edges, including non-retrievable positives.
- MAP: mean average precision, a ranking-quality metric over the retrieved list.
- nDCG@10: normalized discounted cumulative gain at rank 10, a top-rank quality metric.
- T1: Proof-State -> Premise retrieval, used here as the challenge evaluation.
- T2: Theorem -> Theorem neighborhood retrieval, used here as a mechanism check.
- T3: similar-theorem guidance aggregation, used here as qualitative interpretability output.

## Dataset and Evaluation Scope

The enriched dataset is available on HuggingFace, and the validation report is tracked in this repository:

- [ZDCSlab/proofatlas-enriched](https://huggingface.co/datasets/ZDCSlab/proofatlas-enriched)
- [ProofAtlas Project Report](outputs/reports/ProofAtlas_Project_Report.md)

The challenge task is: given a held-out proof state, retrieve useful premises from 127,561 train-side candidate premises. Recall@100 measures whether gold premises are included in the top 100 retrieved candidates out of that 127,561-premise train-side pool.

Local data files should be placed under `data/`. The entire `data/` directory is ignored by git and is used for downloaded, generated, or exported dataset artifacts.

The exported enriched dataset layout is:

```text
data/proofatlas_enriched/v1/
```

It contains theorem-disjoint train/validation/test parquet tables, enriched `theorems.parquet` files, retrieval-ready `theorem_profiles.parquet` files, `manifest.json`, and `dataset_card.md`.

## Capability 1: LLM-Enriched Theorem Neighborhood Retrieval

T2 is the mechanism check for whether LLM-enriched theorem profiles improve theorem-neighbor premise evidence.

| Method | Neighbor premise Recall@100 | nDCG@10 |
| --- | ---: | ---: |
| Baseline theorem profile | 0.3185 | 0.1499 |
| LLM-enriched TF-IDF profile | 0.3774 | 0.1717 |

LLM enrichment converts theorem metadata plus proof-state goals, hypotheses, and symbols into semantic, strategy-oriented, and difficulty-oriented theorem profiles. These profiles are retrieval text only: they do not use gold premise labels or proof scripts as target answers.

## Capability 2: ProofAtlas Guidance Views

T3 turns theorem-neighborhood evidence into qualitative guidance views. A guidance bundle can show:

- query theorem or proof-state context;
- similar theorem neighbors;
- premise suggestions drawn from neighbor proofs;
- strategy facets such as rewrite transport, order/inequality reasoning, typeclass instance resolution, algebraic computation, and case analysis;
- proxy difficulty evidence from similar theorems.

These examples illustrate the evidence exposed by ProofAtlas and are not aggregate performance claims.

## Challenge Evaluation: Proof-State -> Premise Retrieval

T1 is the hard benchmark: given a held-out proof state, retrieve useful premises from 127,561 train-side candidate premises.

| Metric | Dense baseline | Primary method | Absolute gain |
| --- | ---: | ---: | ---: |
| Covered Recall@100 | 0.2362 | 0.4746 | +0.2384 |
| All-positive Recall@100 | 0.1851 | 0.4074 | +0.2223 |
| Covered Recall@50 | 0.1961 | 0.3894 | +0.1933 |
| nDCG@10 | 0.0697 | 0.1236 | +0.0539 |

The primary method is `weighted_rrf_llm_theorem_tuned`, the strongest non-BGE LLM theorem-neighborhood variant. BGE fusion is reported as an auxiliary recall-maximizing variant. The challenge result shows that ProofAtlas improves recall-oriented candidate generation, but top-rank precision remains a next-stage reranking problem.

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

The curated midterm report is tracked at:

```text
outputs/reports/ProofAtlas_Project_Report.md
```

Other generated outputs under `outputs/` are local artifacts and are ignored unless explicitly whitelisted.

## Next Phase

- learned reranker
- downstream prover evaluation
- leakage and stress-test ablations, including statement-only and no-symbol variants
- larger theorem-neighborhood evaluation
- improved top-10 ranking quality

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
