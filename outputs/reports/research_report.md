# ProofAtlas Research Report

## Research Framing

ProofAtlas is framed as a research dataset and retrieval study for LeanRank-style formal proof guidance. The deliverable is not a production proof assistant; it is a processed theorem/proof-state/premise dataset plus prediction artifacts for premise retrieval, proof-pattern retrieval, proof-strategy hinting, and difficulty estimation.

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

## 1. Premise Prediction

| Task | Queries | Recall@1 | Recall@5 | Recall@10 | Recall@100 | MRR | MAP | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proof-state premise retrieval | 3053 | 0.0148 | 0.0798 | 0.1162 | 0.2362 | 0.0783 | 0.0494 | 0.0697 |
| Theorem-level premise retrieval | 1000 | 0.2271 | 0.4284 | 0.4940 | 0.6889 | 0.5609 | 0.3741 | 0.4452 |

The learned premise reranker reaches validation AUC `0.8237`. On the small full-rerank diagnostic sample, reranked proof-state Recall@10 is `0.1250` and hybrid candidate reranking reaches `0.1557`.

## 2. Proof Pattern Prediction

Proof-pattern prediction is represented by similar theorem retrieval, similar proof-state retrieval, and graph-neighborhood evidence rather than a discrete classifier.

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

## 3. Proof Strategy Hinting

Proof strategy hinting now combines deterministic rule labels with embedding-similarity evidence from retrieved similar proof states. This is intentionally reported as weak strategy hinting, not as a supervised strategy classifier.

| Technique label | Test count |
| --- | --- |
| typeclass_resolution | 2288 |
| logical_reasoning | 1648 |
| computation | 722 |
| rewriting_or_coercion | 590 |
| theorem_application | 146 |
| simplification | 63 |
| case_or_constructor_reasoning | 58 |
| extensionality | 47 |
| induction | 9 |
| automation | 3 |

Proof-technique label coverage is `0.8546`. The labels are used as interpretable guidance and as a reranker feature; the ablation shows they are auxiliary rather than the main retrieval signal.

## 4. Difficulty Prediction

Difficulty is a relative research proxy derived from proof-state and theorem complexity signals. Buckets use a split-local distribution policy: easy is the lower 50%, medium is the next 35%, and hard is the top 15%.

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

The strongest quantitative result is theorem-level premise retrieval. Proof-state-level premise retrieval is harder and remains candidate-generation limited, so it should be presented as a baseline and diagnostic target rather than as a solved proof-step predictor. Proof strategies and difficulty are research-facing guidance signals: useful for explanation, slicing, and retrieval policy, but not claims of verified proof synthesis.
