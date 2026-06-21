# ProofAtlas Research Summary

Date: 2026-06-20

## One-Sentence Summary

ProofAtlas is a LeanRank-based research dataset and retrieval study for Lean/mathlib proof guidance. It packages processed proof artifacts, embeddings, retrieval indexes, evaluation results, case studies, and a visual report page.

## Research Scope

The current project narrative is intentionally narrow:

- Build a processed theorem/proof-state/premise dataset from `erbacher/LeanRank-data`.
- Keep train/validation/test theorem-disjoint.
- Encode theorem, proof-state, and premise text into reusable embedding artifacts.
- Evaluate retrieval-grounded proof guidance tasks.
- Deliver a markdown research report and a static visual report page.

The main deliverables are:

- `data/processed/{train,val,test,demo}`
- `outputs/embeddings`
- `outputs/indexes`
- `outputs/predictions/research_prediction_results.json`
- `outputs/reports/research_report.md`
- `homepage/index.html`

## Four Retrieval Tasks

| Module | Research framing | Evaluation evidence |
| --- | --- | --- |
| Premise prediction | Theorem-level premise retrieval and reranking | Held-out theorem to train-premise ranking against LeanRank positive premises. |
| Proof pattern prediction | Similar theorem and similar proof-state retrieval | Not evaluated as a direct neighbor-label task; utility is measured through downstream strategy-facet and difficulty-profile recovery. |
| Proof strategy hinting | Retrieve strategy facets from similar proof states | Label Recall@k and Any-label Hit@k over curated strategy facets. |
| Difficulty prediction | Retrieve historical difficulty profiles | Retrieved-profile MAE/RMSE and easy/medium/hard bucket accuracy. |

Theorem-level premise retrieval is the headline benchmark. Proof-pattern retrieval is the evidence layer: theorem neighbors support premise retrieval, and proof-state neighbors support strategy-facet and difficulty-profile retrieval.

## Current Evidence

Current `configs/proofatlas.yaml` held-out results:

| Task | Metric | Value |
| --- | --- | ---: |
| Theorem-level premise retrieval | Recall@10 | 0.4940 |
| Theorem-level premise retrieval | Recall@100 | 0.6889 |
| Theorem-level premise retrieval | MRR | 0.5609 |
| Strategy-facet retrieval | Label Recall@5 | 0.7441 |
| Strategy-facet retrieval | Any-label Hit@3 | 0.9872 |
| Difficulty-profile retrieval | Bucket accuracy | 0.5231 |
| Difficulty-profile retrieval | Retrieved-profile MAE | 0.0819 |

The full dataset statistics, split/domain counts, task definitions, metric explanations, experiment tables, and case studies are in `outputs/reports/research_report.md`.

## Visual Report

`homepage/index.html` is a static visual companion to the markdown report. It should be read as a more inspectable presentation of the same dataset statistics, retrieval evidence, graph context, and theorem case studies, not as a production proof-assistant interface.

## Engineering Extras

The repo also contains API, deployment, profiling, timing, and audit modules. These are useful for local inspection and reproducibility, but they are not the central contribution. The central contribution is the processed dataset plus retrieval-centered experimental report.

## Future Work

Future work can scale the LeanRank slice, improve theorem/proof-state representations, train stronger rerankers, add richer graph-neighborhood features, validate difficulty against external proof-complexity labels, and evaluate graph neural retrieval models.
