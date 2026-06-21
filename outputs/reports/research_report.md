# ProofAtlas Research Report

## Research Framing

ProofAtlas is framed as a research dataset and retrieval study for LeanRank-style formal proof guidance. The deliverable is not a production proof assistant; it is a processed theorem/proof-state/premise dataset plus retrieval-grounded prediction artifacts for premise retrieval, proof-pattern retrieval, strategy retrieval, and difficulty-profile retrieval.

## Local Deliverables

| Artifact | Path |
| --- | --- |
| Processed dataset | `data/processed/{train,val,test,demo}` |
| Prediction summary | `outputs/predictions/research_prediction_results.json` |
| Research report | `outputs/reports/research_report.md` |
| Embeddings and indexes | `outputs/embeddings`, `outputs/indexes` |
| Learned models | `outputs/models/premise_ranker.joblib`, `outputs/models/difficulty_estimator.joblib` |

## Dataset Snapshot

| Split | Theorems | Proof states | Premises | Positive edges | Negative edges |
| --- | --- | --- | --- | --- | --- |
| train | 8000 | 23723 | 127561 | 54897 | 530413 |
| val | 1000 | 2822 | 36292 | 6610 | 63753 |
| test | 1000 | 3053 | 38332 | 7054 | 68132 |
| demo | 292 | 900 | 1416 | 900 | 900 |

## 1. Premise Retrieval

| Task | Queries | Recall@1 | Recall@5 | Recall@10 | Recall@100 | MRR | MAP | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proof-state premise retrieval | 3053 | 0.0148 | 0.0798 | 0.1162 | 0.2362 | 0.0783 | 0.0494 | 0.0697 |
| Theorem-level premise retrieval | 1000 | 0.2271 | 0.4284 | 0.4940 | 0.6889 | 0.5609 | 0.3741 | 0.4452 |

The learned premise reranker reaches validation AUC `0.8157`. On the small full-rerank diagnostic sample, reranked proof-state Recall@10 is `0.1250` and hybrid candidate reranking reaches `0.1557`.

## 2. Proof Pattern Retrieval

Proof-pattern prediction is represented as retrieval: theorem-to-theorem neighbors, proof-state-to-proof-state neighbors, and graph-neighborhood evidence rather than a discrete classifier.

| Pattern signal | Value |
| --- | --- |
| Theorem retrieval Recall@10 | 0.4940 |
| Theorem retrieval MRR | 0.5609 |
| Train KG nodes | 187971 |
| Train KG edges | 1073855 |
| Train similar_to_theorem edges | 80000 |

| Indexed entity | Rows | Backend | Indexed ms/query | Recall@10 vs exact |
| --- | --- | --- | --- | --- |
| premise | 127561 | hnswlib | 3.3943 | 0.9920 |
| proof_state | 23723 | hnswlib | 0.7790 | 0.9520 |
| theorem | 8000 | hnswlib | 0.2248 | 0.9900 |

## 3. Strategy Retrieval

Strategy is treated as retrieval-grounded hinting. Historical proof states receive weak strategy-facet labels from a curated taxonomy, and a query retrieves similar train proof states before aggregating their facets. This avoids claiming a supervised tactic classifier while still making the strategy signal measurable.

| Strategy retrieval metric | Value |
| --- | --- |
| Evaluated test proof states | 3052 |
| Label Recall@1 | 0.2149 |
| Label Recall@3 | 0.5479 |
| Label Recall@5 | 0.7441 |
| Any-label Hit@1 | 0.8414 |
| Any-label Hit@3 | 0.9872 |

| Strategy facet | Test count |
| --- | --- |
| rewrite_transport | 2217 |
| typeclass_instance_resolution | 1699 |
| order_inequality_reasoning | 1674 |
| case_analysis | 1330 |
| algebraic_computation | 1079 |
| set_membership_reasoning | 975 |
| theorem_application | 926 |
| existential_construction | 796 |
| contradiction_negation | 721 |
| topology_filter_limit | 477 |

Strategy-facet coverage is `0.9997`. These facets are weak retrieval supervision inferred from goal shape, context markers, theorem names, and statement symbols; they are not ground-truth tactic annotations.

## 4. Difficulty Retrieval

Difficulty is treated as historical profile retrieval. A query proof state retrieves similar train proof states, aggregates their complexity profiles, and reports a calibrated relative score/bucket. Buckets use a split-local distribution policy: easy is the lower 50%, medium is the next 35%, and hard is the top 15%.

| Difficulty retrieval metric | Value |
| --- | --- |
| Evaluated test proof states | 3053 |
| Retrieved-profile MAE | 0.0819 |
| Retrieved-profile RMSE | 0.1278 |
| Bucket accuracy | 0.5231 |
| Mean retrieved score | 0.2050 |
| Mean target score | 0.2518 |

| Split | Bucket | Count |
| --- | --- | --- |
| train | easy | 11862 |
| train | medium | 8302 |
| train | hard | 3559 |
| val | easy | 1411 |
| val | medium | 987 |
| val | hard | 424 |
| test | easy | 1527 |
| test | medium | 1068 |
| test | hard | 458 |
| demo | easy | 450 |
| demo | medium | 315 |
| demo | hard | 135 |

| Split | Rows | MAE | R2 | Mean pred | Mean target |
| --- | --- | --- | --- | --- | --- |
| test | 3053 | 0.0670 | 0.5443 | 0.2642 | 0.2518 |
| train | 23723 | 0.0389 | 0.6388 | 0.2188 | 0.2189 |
| val | 2822 | 0.0638 | 0.5119 | 0.2512 | 0.2303 |

## Interpretation

The strongest quantitative result is theorem-level premise retrieval. Proof-state-level premise retrieval is harder and remains candidate-generation limited, but it is still useful as the local-neighbor substrate for strategy retrieval, difficulty-profile retrieval, and explanation. The current theorem-disjoint train/val/test split has no theorem leakage, so the split is suitable for this research framing; future split changes should be motivated by domain-balance or retrieval-coverage studies rather than by leakage repair.
