# ProofAtlas

ProofAtlas is a research-oriented LeanRank retrieval dataset and report pipeline for Lean/mathlib proof guidance. The repo builds processed theorem, proof-state, premise, strategy-facet, difficulty-feature, embedding, index, and graph artifacts, then evaluates retrieval-grounded proof guidance tasks on a theorem-disjoint held-out split.

The primary deliverable is a dataset plus report, not a production proof assistant. The static homepage is a visual companion to the report: it presents the same processed data, retrieval metrics, graph evidence, and case studies in a more inspectable format.

## Research Deliverables

The current report package is:

- Processed dataset: `data/processed/{train,val,test,demo}`
- Embeddings and indexes: `outputs/embeddings`, `outputs/indexes`
- Retrieval results: `outputs/predictions/research_prediction_results.json`
- Research report: `outputs/reports/research_report.md`
- Visual report page: `homepage/index.html`
- Report assets: `homepage/assets/`

The report frames ProofAtlas around four retrieval tasks:

| Module | Research framing | Evidence |
| --- | --- | --- |
| Premise prediction | Theorem-level premise retrieval and reranking | Held-out theorem to train-premise ranking against LeanRank positive premises. |
| Proof pattern prediction | Similar theorem and similar proof-state retrieval | Neighbor and KG evidence used as reusable proof-pattern context. |
| Proof strategy hinting | Retrieve strategy facets from similar proof states | Curated strategy facets aggregated from proof-state neighbors. |
| Difficulty prediction | Retrieve historical difficulty profiles | Neighbor difficulty profiles calibrated into score and easy/medium/hard buckets. |

Proof-pattern neighbors are not evaluated as a standalone label-prediction task. Their utility is measured indirectly through downstream strategy-facet recovery and difficulty-profile recovery; theorem-neighbor quality is also reflected through theorem-level premise retrieval.

## Dataset

The main experiment uses `erbacher/LeanRank-data`, configured by `configs/proofatlas.yaml`. It uses theorem-first sampling from the Hugging Face dataset so that all rows for a sampled theorem stay together.

`configs/sample.yaml` remains the lightweight smoke-test config. It uses a deterministic synthetic LeanRank-shaped sample by default, so tests and local demos can run without a large external download. Set `use_huggingface: true` in a sample config for smaller real-data exploratory runs.

The split is theorem-disjoint: held-out theorem names do not appear in train, so retrieval is evaluated on unseen theorems while using train premises and train proof states as the historical retrieval corpus.

## Current Results

The current `configs/proofatlas.yaml` held-out report summarizes:

| Task | Metric | Value |
| --- | --- | ---: |
| Theorem-level premise retrieval | Recall@10 | 0.4940 |
| Theorem-level premise retrieval | Recall@100 | 0.6889 |
| Theorem-level premise retrieval | MRR | 0.5609 |
| Strategy-facet retrieval | Label Recall@5 | 0.7441 |
| Strategy-facet retrieval | Any-label Hit@3 | 0.9872 |
| Difficulty-profile retrieval | Bucket accuracy | 0.5231 |
| Difficulty-profile retrieval | Retrieved-profile MAE | 0.0819 |

The full tables, dataset statistics, domain statistics, metric definitions, and case studies are in `outputs/reports/research_report.md`.

## Regenerate The Report

Use the conda environment created for this project:

```bash
conda activate leanrank_kg
make install
```

To regenerate only the research report from existing processed artifacts:

```bash
leanrank-kg build-research-report --config configs/proofatlas.yaml
```

This writes:

```text
outputs/reports/research_report.md
outputs/predictions/research_prediction_results.json
```

To refresh the static visual report page:

```bash
leanrank-kg build-homepage --config configs/proofatlas.yaml
```

The homepage is generated at `homepage/index.html`. It is a static visual report, not a deployed service requirement.

## Rebuild Core Artifacts

For a lightweight smoke test:

```bash
conda activate leanrank_kg
make smoke
```

For a small local run:

```bash
make demo
```

For the full research artifact refresh on `configs/proofatlas.yaml`:

```bash
leanrank-kg full-pipeline --config configs/proofatlas.yaml
leanrank-kg build-research-report --config configs/proofatlas.yaml
leanrank-kg build-homepage --config configs/proofatlas.yaml
```

The core pipeline stages are:

```bash
leanrank-kg sample --config configs/proofatlas.yaml
leanrank-kg process --config configs/proofatlas.yaml
leanrank-kg build-graph --config configs/proofatlas.yaml
leanrank-kg label-techniques --config configs/proofatlas.yaml
leanrank-kg compute-difficulty --config configs/proofatlas.yaml
leanrank-kg embed --config configs/proofatlas.yaml
leanrank-kg build-index --config configs/proofatlas.yaml
leanrank-kg benchmark-index --config configs/proofatlas.yaml
leanrank-kg train-ranker --config configs/proofatlas.yaml
leanrank-kg train-difficulty --config configs/proofatlas.yaml
leanrank-kg evaluate --config configs/proofatlas.yaml --full-heldout
leanrank-kg build-research-report --config configs/proofatlas.yaml
```

## Knowledge Graph Schema

Node types:

- `Theorem`
- `ProofState`
- `Premise`
- `FileModule`
- `TacticStep`
- `ProofTechnique`

Base edge types:

- `has_proof_state`
- `appears_in_file`
- `positive_uses`
- `negative_candidate`
- `invokes_premise`
- `defined_in_file`
- `at_tactic_step`

`invokes_premise` is theorem-level: `Theorem -> Premise`, aggregated from positive proof-state premise usage. `positive_uses` and `negative_candidate` remain proof-state-level evidence. `TacticStep` nodes are scoped by theorem and tactic index instead of being shared globally by index.

Enriched edge types include `uses_proof_technique` and `similar_to_theorem`. JSON schemas live in `schemas/`.

## Retrieval Functions

The retrieval library in `src/leanrank_kg/retrieve.py` provides:

- `retrieve_premises`
- `retrieve_similar_theorems`
- `retrieve_similar_proof_states_for_query`
- `explain_premise_match`
- `get_proof_technique_labels`
- `get_difficulty_profile`
- `get_graph_neighborhood`

The command-line equivalents useful for inspection are:

```bash
leanrank-kg retrieve-premises --proof-state-id "<id>"
leanrank-kg retrieve-premises-for-query --query-text "⊢ ..."
leanrank-kg similar-theorems --theorem-id "<id>"
leanrank-kg similar-theorems-for-query --query-text "theorem ..."
leanrank-kg similar-proof-states-for-query --query-text "⊢ ..."
leanrank-kg explain-premise-match --proof-state-id "<id>" --premise-id "<id>"
leanrank-kg show-difficulty --entity-id "<id>"
leanrank-kg show-techniques --proof-state-id "<id>"
leanrank-kg show-neighborhood --entity-id "<id>"
```

## Visual Report

`homepage/index.html` is kept as a visual research report. It uses files under `homepage/assets/` copied from real pipeline outputs, including corpus metadata, metrics, graph statistics, domain coverage, retrieval examples, and theorem case studies.

The page should be read as a static companion to `outputs/reports/research_report.md`. It is useful for browsing results, but the research claims should be cited from the generated markdown report and JSON artifacts.

## Engineering Extras

The repo also contains optional engineering components that are not part of the main research narrative:

- `src/leanrank_kg/api.py`: local FastAPI wrapper for interactive retrieval inspection.
- `src/leanrank_kg/deployment_security.py`: deployment-readiness checks for service-style demos.
- `src/leanrank_kg/pipeline_profile.py` and `src/leanrank_kg/pipeline_timing.py`: scale and timing diagnostics.
- `src/leanrank_kg/audit.py`: MVP-style artifact audit.
- `docs/proofatlas_deployment_guide.md`: local/server demo notes.

These components are useful for reproducibility, debugging, and demos, but they should not be read as the central contribution. The central contribution is the processed dataset, retrieval results, evaluation metrics, case studies, and visual report.

## Limitations

ProofAtlas is retrieval-centered. It does not claim to solve proof automation or proof difficulty prediction on full mathlib. Strategy facets are deterministic retrieval annotations inferred from theorem names, goal shape, context markers, and statement symbols. Difficulty buckets are relative to the processed split distribution.

GNN models, richer Lean interaction, external proof-complexity labels, and larger multi-corpus evaluation are future work.
