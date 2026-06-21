# ProofAtlas Research Report

## Research Framing

ProofAtlas is framed as a research dataset and retrieval study for LeanRank-style formal proof guidance. The deliverable is not a production proof assistant; it is a processed theorem/proof-state/premise dataset plus retrieval-grounded prediction artifacts for theorem-level premise retrieval, proof-pattern retrieval, strategy retrieval, and difficulty-profile retrieval.

## Local Deliverables

| Artifact | Path |
| --- | --- |
| Processed dataset | `data/processed/{train,val,test,demo}` |
| Prediction summary | `outputs/predictions/research_prediction_results.json` |
| Research report | `outputs/reports/research_report.md` |
| Embeddings and indexes | `outputs/embeddings`, `outputs/indexes` |
| Learned models | `outputs/models/premise_ranker.joblib`, `outputs/models/difficulty_estimator.joblib` |

## Dataset

| Field | Value |
| --- | --- |
| Source | erbacher/LeanRank-data |
| Source kind | huggingface |
| Sample unit | theorem |
| Sampled theorems | 10000 |
| Sampled rows | 292012 |
| Random seed | 42 |
| Config hash | 614a66ee7ff568bc |

The processed dataset contains theorem-level, proof-state-level, premise-level, positive-premise, negative-candidate, strategy-facet, difficulty-feature, embedding, index, and KG artifacts. The split is theorem-disjoint: held-out theorem names do not appear in train, so retrieval is evaluated against unseen theorems while using train premises and train proof states as the historical retrieval corpus.

| Split policy | Value |
| --- | --- |
| Train/val/test theorem counts | test=1000, train=8000, val=1000 |
| Theorem leakage detected | False |

### Split Statistics

| Split | Theorems | Proof states | Premises | Positive edges | Negative edges |
| --- | --- | --- | --- | --- | --- |
| train | 8000 | 23723 | 127561 | 54897 | 530413 |
| val | 1000 | 2822 | 36292 | 6610 | 63753 |
| test | 1000 | 3053 | 38332 | 7054 | 68132 |
| demo | 292 | 900 | 1416 | 900 | 900 |

### Domain Statistics

| Test domain | Theorems | Share |
| --- | --- | --- |
| Algebra | 149 | 0.1490 |
| Data | 146 | 0.1460 |
| Analysis | 116 | 0.1160 |
| Topology | 78 | 0.0780 |
| RingTheory | 70 | 0.0700 |
| MeasureTheory | 66 | 0.0660 |
| CategoryTheory | 54 | 0.0540 |
| LinearAlgebra | 50 | 0.0500 |
| Order | 46 | 0.0460 |
| NumberTheory | 35 | 0.0350 |
| GroupTheory | 27 | 0.0270 |
| Probability | 26 | 0.0260 |

## Evaluation Metrics

| Metric | Meaning |
| --- | --- |
| Recall@k | Fraction of retrievable gold items recovered in the top-k retrieved results. |
| MRR | Mean reciprocal rank of the first retrieved gold item. |
| MAP | Mean average precision over ranked retrieved items. |
| nDCG@k | Rank-sensitive gain that rewards placing gold items earlier in the top-k list. |
| AUC | Validation discrimination of the learned premise reranker over positive and hard-negative premise pairs. |
| Label Recall@k | Average fraction of a query proof state's weak strategy facets recovered by the top-k aggregated retrieved facets. |
| Any-label Hit@k | Fraction of labeled proof states for which at least one weak strategy facet is recovered in the top-k facets. |
| MAE/RMSE | Absolute and squared-error summaries for retrieved difficulty-profile scores. |
| Bucket accuracy | Agreement between retrieved difficulty bucket and the query proof state's relative difficulty bucket. |

## 1. Theorem-Level Premise Retrieval

| Field | Description |
| --- | --- |
| Task definition | Retrieve premises that are useful for proving a held-out theorem. |
| Input | A test theorem represented by its theorem-level text embedding. |
| Retrieval corpus | Train-split premise embeddings and metadata. |
| Output | A ranked list of premise IDs/names with retrieval scores. |
| Evaluation target | Positive LeanRank premises used by any proof state of the held-out theorem, restricted to gold premises present in the train premise index. |
| Metrics | Recall@1/5/10/100, MRR, MAP, and nDCG@10. |
| Role in report | Headline premise-retrieval benchmark for theorem guidance. |

| Task | Queries | Recall@1 | Recall@5 | Recall@10 | Recall@100 | MRR | MAP | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Theorem-level premise retrieval | 1000 | 0.2271 | 0.4284 | 0.4940 | 0.6889 | 0.5609 | 0.3741 | 0.4452 |

The learned premise reranker reaches validation AUC `0.8157` over positive and hard-negative premise pairs.

## 2. Proof Pattern Retrieval

| Field | Description |
| --- | --- |
| Task definition | Retrieve historical neighbors that provide proof-pattern evidence for a query theorem or proof state. |
| Input | A theorem embedding for theorem-neighbor retrieval, or a proof-state embedding for local proof-state-neighbor retrieval. |
| Retrieval corpus | Train theorem embeddings, train proof-state embeddings, and the enriched proof KG. |
| Output | Similar theorems, similar proof states, and graph-neighborhood evidence such as similar_to_theorem and premise/strategy edges. |
| Evaluation target | No standalone human-labeled proof-pattern pairs are available; theorem-neighbor quality is proxied by theorem-level premise retrieval, while proof-state-neighbor quality is evaluated through the strategy and difficulty tasks below. |
| Metrics | Theorem retrieval Recall/MRR proxy plus HNSW index recall@10 versus exact cosine for theorem/proof-state/premise indexes. |
| Role in report | Evidence layer that supports strategy-facet retrieval, difficulty-profile retrieval, and interpretability. |

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

| Field | Description |
| --- | --- |
| Task definition | Retrieve likely proof-strategy facets for a query proof state. |
| Input | A test proof-state embedding. |
| Retrieval corpus | Train proof-state embeddings whose proof states are weakly labeled with curated strategy facets. |
| Output | A ranked set of strategy facets, e.g. rewrite_transport, order_inequality_reasoning, algebraic_computation, typeclass_instance_resolution, case_analysis, and set_membership_reasoning. |
| Evaluation target | The query proof state's own weak strategy facets from the same taxonomy. This is a retrieval-grounded weak-label evaluation, not supervised tactic classification. |
| Metrics | Label Recall@1/3/5 and Any-label Hit@1/3. |
| Role in report | Auxiliary guidance task showing whether local proof-state neighbors recover useful strategy facets. |

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

| Field | Description |
| --- | --- |
| Task definition | Retrieve historical difficulty profiles for a query proof state and summarize them as a relative complexity score/bucket. |
| Input | A test proof-state embedding. |
| Retrieval corpus | Train proof-state embeddings with precomputed difficulty features and relative difficulty buckets. |
| Output | A retrieved-neighbor difficulty score, easy/medium/hard bucket, and calibration diagnostics. |
| Evaluation target | The query proof state's relative complexity proxy, derived from proof length, tactic index, positive-premise count, namespace rarity, and negative-candidate hardness. |
| Metrics | Retrieved-profile MAE/RMSE, bucket accuracy, mean retrieved score, and mean target score. |
| Role in report | Auxiliary retrieval task for estimating whether a query resembles historically easier or harder proof states. |

Difficulty buckets use a split-local distribution policy: easy is the lower 50%, medium is the next 35%, and hard is the top 15%.

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

The dataset and report support a retrieval-centered research claim. Theorem-level premise retrieval is the strongest quantitative result, while proof-state-level premise retrieval remains candidate-generation limited and should be presented as the main open challenge. Proof-state retrieval is still useful as a local-neighbor substrate for strategy-facet retrieval, difficulty-profile retrieval, and explanation. The current theorem-disjoint train/val/test split has no theorem leakage; future split changes should be motivated by domain-balance or retrieval-coverage studies rather than leakage repair.
