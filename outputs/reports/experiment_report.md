# ProofAtlas Experiment Report

## Experiment Setup

- Dataset: `erbacher/LeanRank-data`
- Source kind: `huggingface`
- Split counts: test: 28910, train: 234520, val: 28582
- Candidate pool: `train premise index`
- Label policy: held-out test positive_edges are used only for evaluation
- Evaluation scope: `full held-out splits`
- Proof-state evaluation limits: `{'test': None, 'val': None}`
- Theorem evaluation limits: `{'test': None, 'val': None}`
- Proof-state test coverage: `100` / `3053` (`0.03275466754012447`)
- Theorem test coverage: `50` / `1000` (`0.05`)
- Ranking backend: `batched_embedding_topk`
- Proof-state query representation: `stored_embedding`
- Evaluation GPU: `use_gpu=True`, device `cuda:0`, batch size `256`
- Actual test ranking backend: proof-state `torch_cuda`, theorem `torch_cuda`
- Case-study regeneration during evaluation: `0` theorem guidance cases
- Data supervision: `leanrank_proof_state_rows`
- Has tactic states: `True`
- Has true positive premises: `True`
- Embedding backend: `sentence_transformers`
- Index backend: `auto`

## Final Artifacts

- Homepage/demo: `homepage/index.html`
- Experiment report: `outputs/reports/experiment_report.md`
- Machine-readable held-out evaluation: `outputs/reports/test_set_evaluation.json`
- Pipeline performance profile: `outputs/reports/pipeline_performance_report.json`

## ML Task Definition

ProofAtlas is evaluated as a supervised ranking/retrieval system over a theorem-disjoint train/validation/test split. The train split supplies the candidate premise index and graph evidence. The held-out validation and test positive edges are never used as retrieval candidates; they are used only as gold labels for scoring.

- Proof-state-level task: query with a held-out proof state and rank train-side premises.
- Theorem-level task: query with a held-out theorem-level text representation and rank train-side premises for proof guidance.
- Reported metrics: Recall@k, MRR, MAP, nDCG@k, and gold-premise coverage against the train premise index.
- When evaluation limits are configured, metrics are computed on a deterministic prefix sample of the held-out split and reported as sampled held-out metrics.

The current default experiment uses `erbacher/LeanRank-data` only. Local Lean/mathlib source extraction is out of scope for this experiment.

## Candidate Pool Diagnostic

These metrics test whether the embedding/index candidate pool contains the gold premise before reranking. If Recall@100 is low, the next accuracy bottleneck is candidate generation or embeddings; if Recall@100 is high but Recall@10 is low, the bottleneck is reranking.

### Retrieval Bottleneck Profile

| Task | Recall@10 | Recall@100 | Gap | Top10/Top100 | Primary bottleneck |
| --- | ---: | ---: | ---: | ---: | --- |
| Proof-state premise retrieval | 0.1279 | 0.3090 | 0.1810 | 0.4141 | `candidate_generation_or_embeddings` |
| Theorem-level premise retrieval | 0.4233 | 0.6642 | 0.2409 | 0.6373 | `top10_reranking_or_candidate_ordering` |

### Proof-State Candidate Pool

| Metric | Value |
| --- | ---: |
| `Recall@50` | 0.1961 |
| `Recall@100` | 0.2362 |
| `nDCG@50` | 0.0908 |
| `nDCG@100` | 0.0987 |

### Theorem Candidate Pool

| Metric | Value |
| --- | ---: |
| `theorem_retrieval_Recall@50` | 0.6314 |
| `theorem_retrieval_Recall@100` | 0.6889 |
| `theorem_retrieval_nDCG@50` | 0.4907 |
| `theorem_retrieval_nDCG@100` | 0.5060 |

### Rapid Convergence Plan

This plan connects the held-out retrieval metrics, reranked diagnostic, ranker ablation, and LeanRank-data supervision into the next experiments most likely to improve retrieval accuracy.

| Priority | Area | Target metric | Current value | Reason |
| ---: | --- | --- | ---: | --- |
| 1 | `proof_state_query_and_embedding` | proof_state Recall@100 | 0.3090 | Proof-state gold premises are often absent from the top-100 candidate pool, so top-k reranking cannot recover them. Validation query diagnostic currently favors `full_name_goal`. |
| 2 | `theorem_level_reranking` | theorem_retrieval Recall@10 | 0.4233 | Theorem-level Recall@100 is substantially higher than Recall@10, leaving useful headroom for ordering candidates already in the pool. |
| 3 | `ranker_feature_iteration` | validation/test Recall@10 and MAP | 0.0383 | Ranker ablation says `frequency` is the strongest currently measured feature group by delta_without_group. |
| 4 | `hard_negative_training` | MRR/MAP after reranking | 9.5478 | LeanRank-data already provides positive premises and hard negative candidates, so training/evaluation changes can reuse existing labels without extracting new data. |

Accuracy snapshot:

| Metric | Value |
| --- | ---: |
| `proof_state_recall_at_10` | 0.1279 |
| `proof_state_recall_at_100` | 0.3090 |
| `theorem_recall_at_10` | 0.4233 |
| `theorem_recall_at_100` | 0.6642 |
| `reranked_proof_state_recall_at_10` | 0.1689 |
| `reranked_minus_embedding_recall_at_10` | 0.0409 |

Headroom:

| Metric | Value |
| --- | ---: |
| `proof_state_missing_from_top100` | 0.6910 |
| `proof_state_top10_to_top100_gap` | 0.1810 |
| `theorem_missing_from_top100` | 0.3358 |
| `theorem_top10_to_top100_gap` | 0.2409 |

Query representation diagnostic summary:

| Metric | Value |
| --- | ---: |
| `validation` | {'best_variant_by_recall': 'full_name_goal', 'evaluated_queries': 50, 'selection_metric': 'Recall@100'} |
| `test` | {'best_variant_by_recall': 'full_name_goal', 'evaluated_queries': 50, 'selection_metric': 'Recall@100'} |
| `validation_test_best_variant_match` | True |

Strongest ranker feature groups:

| Feature group | Delta without group | Group-only AUC | Columns |
| --- | ---: | ---: | --- |
| `frequency` | 0.0383 | 0.7842 | `premise_frequency` |
| `symbol_overlap` | 0.0137 | 0.5993 | `symbol_name_overlap`, `symbol_context_overlap` |
| `embedding_similarity` | 0.0023 | 0.5995 | `cosine_similarity` |
| `namespace_domain` | 0.0021 | 0.5163 | `same_namespace`, `same_domain` |
| `theorem_neighborhood` | 0.0018 | 0.5810 | `theorem_neighborhood_premise_score` |

### Proof-State Query Representation Diagnostic

Validation diagnostics are the safer signal for choosing proof-state query text variants; test diagnostics are reported to show whether that choice generalizes. The main proof-state retrieval metric remains the committed production evaluation path.

Validation split:

- Evaluated queries: `50`
- Best variant by `Recall@100`: `full_name_goal`

| Query Representation | Recall@50 | Recall@100 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: |
| `context_goal` | 0.0677 | 0.0979 | 0.0386 | 0.0270 |
| `full_name_context_goal` | 0.1719 | 0.1927 | 0.0707 | 0.0476 |
| `full_name_goal` | 0.2396 | 0.2604 | 0.0745 | 0.0481 |
| `goal_only` | 0.1589 | 0.1797 | 0.0727 | 0.0589 |
| `theorem_id_goal` | 0.2344 | 0.2604 | 0.0764 | 0.0504 |

Test split:

- Evaluated queries: `50`
- Best variant by `Recall@100`: `full_name_goal`

| Query Representation | Recall@50 | Recall@100 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: |
| `context_goal` | 0.1094 | 0.1302 | 0.0695 | 0.0378 |
| `full_name_context_goal` | 0.2075 | 0.2378 | 0.0887 | 0.0528 |
| `full_name_goal` | 0.2491 | 0.2700 | 0.0777 | 0.0491 |
| `goal_only` | 0.1806 | 0.1806 | 0.0588 | 0.0265 |
| `theorem_id_goal` | 0.2387 | 0.2700 | 0.0755 | 0.0484 |

## Held-Out Test Set Metrics

### Proof-State-Level Premise Ranking

Each held-out test proof state is used as a query. Gold positive premises from the test proof trace are used only for scoring, while candidates come from the train premise index.

| Metric | Value |
| --- | ---: |
| `evaluated_queries` | 3053 |
| `evaluated_retrievable_queries` | 2832 |
| `gold_premise_coverage` | 0.9076 |
| `Recall@1` | 0.0148 |
| `Recall@5` | 0.0798 |
| `Recall@10` | 0.1162 |
| `MRR` | 0.0783 |
| `MAP` | 0.0494 |
| `nDCG@10` | 0.0697 |

### Proof-State-Level Reranked Retrieval

This smaller diagnostic uses the same retrieval path as the homepage/API: query encoding, hnswlib candidate retrieval, and the learned/fixed reranker. It is slower than batched embedding evaluation, but better reflects user-facing proof guidance.

- Backend: `batched_torch_cuda_then_rerank`
- Candidate k: `50`
- Evaluated queries: `20`

| Metric | Value |
| --- | ---: |
| `evaluated_queries` | 20 |
| `evaluated_retrievable_queries` | 19 |
| `gold_premise_coverage` | 0.8772 |
| `Recall@1` | 0.0329 |
| `Recall@5` | 0.1338 |
| `Recall@10` | 0.1689 |
| `MRR` | 0.1798 |
| `MAP` | 0.0765 |
| `nDCG@10` | 0.1238 |

#### Rerank Candidate-Depth Ablation

This ablation uses the same held-out rerank diagnostic queries and changes only the number of embedding candidates passed into the learned/fixed reranker.

| Candidate k | Recall@1 | Recall@5 | Recall@10 | MRR | MAP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.0329 | 0.1338 | 0.1689 | 0.1798 | 0.0765 |
| 100 | 0.0329 | 0.1338 | 0.1689 | 0.1798 | 0.0765 |

### Theorem-Level Premise Ranking

Each held-out test theorem is used as a query for proof guidance. Gold premises are all positive premises used by that theorem in the held-out split.

| Metric | Value |
| --- | ---: |
| `theorem_retrieval_evaluated_theorems` | 1000 |
| `theorem_retrieval_evaluated_theorems_with_train_gold` | 955 |
| `theorem_retrieval_gold_premise_coverage` | 0.8994 |
| `theorem_retrieval_Recall@1` | 0.2271 |
| `theorem_retrieval_Recall@5` | 0.4284 |
| `theorem_retrieval_Recall@10` | 0.4940 |
| `theorem_retrieval_MRR` | 0.5609 |
| `theorem_retrieval_MAP` | 0.3741 |
| `theorem_retrieval_nDCG@10` | 0.4452 |

## Metric Uncertainty

Confidence level: `0.95`. Method: `bounded_normal_approximation_for_aggregate_retrieval_metrics`.

Intervals are approximate diagnostics for bounded aggregate retrieval metrics, not a replacement for paired bootstrap comparisons.

### Proof-State Metric Intervals

| Metric | Value | n | 95% CI low | 95% CI high | Half-width |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Recall@10` | 0.1279 | 100 | 0.0625 | 0.1934 | 0.0655 |
| `Recall@100` | 0.3090 | 100 | 0.2184 | 0.3995 | 0.0906 |
| `MRR` | 0.0754 | 100 | 0.0237 | 0.1272 | 0.0518 |
| `MAP` | 0.0460 | 100 | 0.0049 | 0.0871 | 0.0411 |
| `nDCG@10` | 0.0680 | 100 | 0.0187 | 0.1174 | 0.0494 |

### Theorem Metric Intervals

| Metric | Value | n | 95% CI low | 95% CI high | Half-width |
| --- | ---: | ---: | ---: | ---: | ---: |
| `theorem_retrieval_Recall@10` | 0.4233 | 50 | 0.2863 | 0.5602 | 0.1370 |
| `theorem_retrieval_Recall@100` | 0.6642 | 50 | 0.5333 | 0.7951 | 0.1309 |
| `theorem_retrieval_MRR` | 0.5473 | 50 | 0.4093 | 0.6853 | 0.1380 |
| `theorem_retrieval_MAP` | 0.3560 | 50 | 0.2233 | 0.4887 | 0.1327 |
| `theorem_retrieval_nDCG@10` | 0.4139 | 50 | 0.2774 | 0.5505 | 0.1365 |

## Domain Breakdown

These tables show held-out test metrics grouped by LeanRank-data domain. They help identify where ranking quality is strong or weak instead of relying only on aggregate metrics.

### Test Proof-State-Level Domains

| Domain | Queries | `Recall@10` | `MRR` | `MAP` | `nDCG@10` | `gold_premise_coverage` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Analysis | 391 | 0.0927 | 0.0594 | 0.0349 | 0.0521 | 0.9054 |
| Data | 346 | 0.1741 | 0.1205 | 0.0802 | 0.1087 | 0.9161 |
| MeasureTheory | 323 | 0.0890 | 0.0411 | 0.0309 | 0.0458 | 0.9508 |
| Algebra | 307 | 0.1325 | 0.0851 | 0.0481 | 0.0747 | 0.9187 |
| Topology | 293 | 0.0671 | 0.0473 | 0.0334 | 0.0430 | 0.9392 |
| RingTheory | 271 | 0.0913 | 0.0757 | 0.0435 | 0.0603 | 0.9259 |
| CategoryTheory | 168 | 0.0997 | 0.0935 | 0.0493 | 0.0683 | 0.8224 |
| LinearAlgebra | 141 | 0.0913 | 0.1009 | 0.0456 | 0.0636 | 0.9434 |
| NumberTheory | 133 | 0.1394 | 0.0817 | 0.0578 | 0.0837 | 0.9012 |
| Probability | 116 | 0.0602 | 0.0326 | 0.0279 | 0.0311 | 0.9041 |
| GroupTheory | 80 | 0.2006 | 0.1189 | 0.0795 | 0.1209 | 0.8894 |
| Order | 79 | 0.1406 | 0.0829 | 0.0579 | 0.0804 | 0.8802 |

### Test Theorem-Level Domains

| Domain | Queries | `theorem_retrieval_Recall@10` | `theorem_retrieval_MRR` | `theorem_retrieval_MAP` | `theorem_retrieval_nDCG@10` | `theorem_retrieval_gold_premise_coverage` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Algebra | 149 | 0.5369 | 0.5949 | 0.4113 | 0.4843 | 0.9048 |
| Data | 146 | 0.6199 | 0.6702 | 0.4696 | 0.5555 | 0.9150 |
| Analysis | 116 | 0.3724 | 0.4733 | 0.2810 | 0.3387 | 0.9051 |
| Topology | 78 | 0.4723 | 0.4962 | 0.3615 | 0.4184 | 0.9168 |
| RingTheory | 70 | 0.4352 | 0.5314 | 0.3293 | 0.4004 | 0.9223 |
| MeasureTheory | 66 | 0.4028 | 0.4788 | 0.3101 | 0.3668 | 0.9324 |
| CategoryTheory | 54 | 0.4712 | 0.5981 | 0.3771 | 0.4489 | 0.7963 |
| LinearAlgebra | 50 | 0.4775 | 0.6330 | 0.3518 | 0.4401 | 0.9412 |
| Order | 46 | 0.5710 | 0.6238 | 0.4549 | 0.5196 | 0.8874 |
| NumberTheory | 35 | 0.4287 | 0.5241 | 0.2986 | 0.3829 | 0.8824 |
| GroupTheory | 27 | 0.4982 | 0.5080 | 0.3388 | 0.4165 | 0.8630 |
| Probability | 26 | 0.5799 | 0.5168 | 0.4213 | 0.4869 | 0.9167 |

## Error Analysis

Worst-case rows are held-out test queries with train-index gold premises but low top-k recovery. These are the first examples to inspect when improving embeddings, reranking features, or candidate depth.

### Failure Profile Summary

These aggregate buckets quantify where held-out retrieval fails without storing every per-query row in the committed report. `zero_recall_at_max_k` counts retrievable queries where no train-side gold premise appeared within the largest evaluated candidate pool.

#### Proof-State Failure Profile

| Signal | Value |
| --- | ---: |
| `evaluated_queries` | 3053 |
| `retrievable_queries` | 2832 |
| `queries_without_train_gold` | 221 |
| `queries_with_missing_gold` | 562 |
| `zero_recall_at_max_k` | 1823 |
| `max_k` | 100 |

Proof-state failure diagnosis:

This table converts the aggregate buckets into actionable causes. Overlap is intentional for train-gold coverage: a query can have partial gold coverage and still be a candidate-pool miss.

| Diagnosis | Queries | Share of evaluated | Share of retrievable | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `no_train_gold` | 221 | 7.2% | 7.8% | Held-out positives have no matching premise in the train candidate index; retrieval cannot score these as hits. |
| `partial_train_gold_coverage` | 341 | 11.2% | 12.0% | At least one gold premise is available, but some held-out gold premises are absent from the train candidate index. |
| `candidate_pool_miss_top_100` | 1823 | 59.7% | 64.4% | Train-side gold exists, but no gold premise appears in the top-100 embedding candidate pool. |
| `reranking_headroom_after_top10` | 458 | 15.0% | 16.2% | A gold premise appears after rank 10, so better ordering could improve top-10 metrics without changing candidate generation. |
| `top10_hit` | 551 | 18.0% | 19.5% | At least one train-side gold premise already appears in the top 10. |

Proof-state rank buckets:

| Rank bucket | Queries |
| --- | ---: |
| `miss_top_100` | 1823 |
| `no_train_gold` | 221 |
| `rank_1` | 68 |
| `rank_11_to_50` | 317 |
| `rank_2_to_5` | 319 |
| `rank_51_to_100` | 141 |
| `rank_6_to_10` | 164 |

Proof-state gold coverage buckets:

| Gold coverage bucket | Queries |
| --- | ---: |
| `full_train_gold_coverage` | 2491 |
| `no_train_gold_coverage` | 221 |
| `partial_train_gold_coverage` | 341 |

Proof-state zero-recall domains:

| Domain | Zero-recall queries |
| --- | ---: |
| Analysis | 255 |
| MeasureTheory | 230 |
| Topology | 229 |
| RingTheory | 177 |
| Data | 154 |
| Algebra | 153 |
| CategoryTheory | 89 |
| NumberTheory | 83 |
| Probability | 82 |
| LinearAlgebra | 77 |

#### Theorem Failure Profile

| Signal | Value |
| --- | ---: |
| `evaluated_queries` | 1000 |
| `retrievable_queries` | 955 |
| `queries_without_train_gold` | 45 |
| `queries_with_missing_gold` | 358 |
| `zero_recall_at_max_k` | 55 |
| `max_k` | 100 |

Theorem failure diagnosis:

| Diagnosis | Queries | Share of evaluated | Share of retrievable | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `no_train_gold` | 45 | 4.5% | 4.7% | Held-out positives have no matching premise in the train candidate index; retrieval cannot score these as hits. |
| `partial_train_gold_coverage` | 313 | 31.3% | 32.8% | At least one gold premise is available, but some held-out gold premises are absent from the train candidate index. |
| `candidate_pool_miss_top_100` | 55 | 5.5% | 5.8% | Train-side gold exists, but no gold premise appears in the top-100 embedding candidate pool. |
| `reranking_headroom_after_top10` | 133 | 13.3% | 13.9% | A gold premise appears after rank 10, so better ordering could improve top-10 metrics without changing candidate generation. |
| `top10_hit` | 767 | 76.7% | 80.3% | At least one train-side gold premise already appears in the top 10. |

Theorem rank buckets:

| Rank bucket | Queries |
| --- | ---: |
| `miss_top_100` | 55 |
| `no_train_gold` | 45 |
| `rank_1` | 403 |
| `rank_11_to_50` | 100 |
| `rank_2_to_5` | 294 |
| `rank_51_to_100` | 33 |
| `rank_6_to_10` | 70 |

Theorem gold coverage buckets:

| Gold coverage bucket | Queries |
| --- | ---: |
| `full_train_gold_coverage` | 642 |
| `no_train_gold_coverage` | 45 |
| `partial_train_gold_coverage` | 313 |

Theorem zero-recall domains:

| Domain | Zero-recall queries |
| --- | ---: |
| Analysis | 8 |
| MeasureTheory | 7 |
| NumberTheory | 6 |
| CategoryTheory | 5 |
| Algebra | 5 |
| Topology | 5 |
| Data | 4 |
| RingTheory | 3 |
| GroupTheory | 2 |
| Order | 2 |

#### Reranked Proof-State Diagnostic Failure Profile

| Signal | Value |
| --- | ---: |
| `evaluated_queries` | 20 |
| `retrievable_queries` | 19 |
| `queries_without_train_gold` | 1 |
| `queries_with_missing_gold` | 5 |
| `zero_recall_at_max_k` | 13 |
| `max_k` | 10 |

### Worst Proof-State Queries

| Item | Domain | Gold in train | Missing gold | Recall@10 | MRR contribution | MAP contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ps:exists_nat_nat_continuous_surjective_of_completeSpace:8:a5e4a0a389a2 | Topology | 24 | 0 | n/a | 0.0000 | 0.0000 |
| ps:IsAdicComplete.le_jacobson_bot:5:535863f3cc8e | RingTheory | 16 | 0 | n/a | 0.0000 | 0.0000 |
| ps:Profinite.epi_iff_surjective:10:229ef3f110bd | Topology | 15 | 1 | n/a | 0.0000 | 0.0000 |
| ps:MeasureTheory.LevyProkhorov.continuous_equiv_symm_probabilityMeasure:7:9f2d803eb2a1 | MeasureTheory | 14 | 0 | n/a | 0.0000 | 0.0000 |
| ps:exists_nat_nat_continuous_surjective_of_completeSpace:7:fd09f2736ab9 | Topology | 14 | 0 | n/a | 0.0000 | 0.0000 |
| ps:MeasureTheory.LevyProkhorov.continuous_equiv_symm_probabilityMeasure:35:a4f0a1dbd5a7 | MeasureTheory | 13 | 0 | n/a | 0.0000 | 0.0000 |
| ps:iteratedDerivWithin_vcomp_three:2:f2f332f59da8 | Analysis | 12 | 2 | n/a | 0.0000 | 0.0000 |
| ps:iteratedDerivWithin_vcomp_three:5:f2f332f59da8 | Analysis | 12 | 2 | n/a | 0.0000 | 0.0000 |
| ps:Complex.two_pi_I_inv_smul_circleIntegral_sub_inv_smul_of_differentiable_on_off_countable:6:6b2e175ca8ee | Analysis | 11 | 0 | n/a | 0.0000 | 0.0000 |
| ps:MulChar.IsQuadratic.gaussSum_frob_iter:0:427502f78e8d | NumberTheory | 11 | 0 | n/a | 0.0000 | 0.0000 |

### Worst Theorem Queries

| Item | Domain | Gold in train | Missing gold | Recall@10 | MRR contribution | MAP contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ProbabilityTheory.strong_law_aux1 | Probability | 69 | 4 | n/a | 0.0000 | 0.0000 |
| exists_nat_nat_continuous_surjective_of_completeSpace | Topology | 58 | 0 | n/a | 0.0000 | 0.0000 |
| IsAdicComplete.le_jacobson_bot | RingTheory | 34 | 1 | n/a | 0.0000 | 0.0000 |
| MeasureTheory.StronglyMeasurable.finStronglyMeasurable_of_set_sigmaFinite | MeasureTheory | 34 | 0 | n/a | 0.0000 | 0.0000 |
| MvPowerSeries.X_pow_dvd_iff | RingTheory | 21 | 2 | n/a | 0.0000 | 0.0000 |
| SimpleGraph.even_card_odd_degree_vertices | Combinatorics | 20 | 0 | n/a | 0.0000 | 0.0000 |
| LSeries_eventually_eq_zero_iff' | NumberTheory | 19 | 2 | n/a | 0.0000 | 0.0000 |
| MeasureTheory.llr_tilted_left | MeasureTheory | 18 | 0 | n/a | 0.0000 | 0.0000 |
| TopCat.isTopologicalBasis_cofiltered_limit | Topology | 16 | 2 | n/a | 0.0000 | 0.0000 |
| ArithmeticFunction.vonMangoldt.abscissaOfAbsConv_residueClass_le_one | NumberTheory | 15 | 0 | n/a | 0.0000 | 0.0000 |

## Validation Metrics

Validation metrics are reported for model selection and sanity checking; final claims should use the held-out test metrics above.

### Validation Proof-State-Level Premise Ranking

| Metric | Value |
| --- | ---: |
| `evaluated_queries` | 2822 |
| `evaluated_retrievable_queries` | 2643 |
| `gold_premise_coverage` | 0.9098 |
| `Recall@1` | 0.0163 |
| `Recall@5` | 0.0828 |
| `Recall@10` | 0.1142 |
| `MRR` | 0.0801 |
| `MAP` | 0.0511 |
| `nDCG@10` | 0.0698 |

### Validation Theorem-Level Premise Ranking

| Metric | Value |
| --- | ---: |
| `theorem_retrieval_evaluated_theorems` | 1000 |
| `theorem_retrieval_evaluated_theorems_with_train_gold` | 961 |
| `theorem_retrieval_gold_premise_coverage` | 0.8982 |
| `theorem_retrieval_Recall@1` | 0.2556 |
| `theorem_retrieval_Recall@5` | 0.4532 |
| `theorem_retrieval_Recall@10` | 0.5100 |
| `theorem_retrieval_MRR` | 0.5744 |
| `theorem_retrieval_MAP` | 0.3949 |
| `theorem_retrieval_nDCG@10` | 0.4626 |

## ANN Index Benchmark

This benchmark compares the saved nearest-neighbor index against exact cosine search on sampled train queries. It measures whether the ANN backend is fast enough for interactive retrieval while preserving the exact top-k neighborhood used by the embedding candidate generator.

| Entity | Backend | Rows | Exact ms/query | Indexed ms/query | Speedup | Recall@1 vs exact | Recall@5 vs exact | Recall@10 vs exact | Top1 match@10 | Build seconds | Indexed total seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| premise | hnswlib | 127561 | 69.1912 | 3.6822 | 18.7909 | 1.0000 | 0.9940 | 0.9930 | 1.0000 | 3.6046 | 0.3682 |
| proof_state | hnswlib | 23723 | 12.8202 | 0.8267 | 15.5081 | 0.7800 | 0.9420 | 0.9500 | 0.7800 | 0.4239 | 0.0827 |
| theorem | hnswlib | 8000 | 4.0994 | 0.2423 | 16.9156 | 1.0000 | 0.9980 | 0.9960 | 1.0000 | 0.1380 | 0.0242 |

## Premise Trace Supervision

The current ranking labels come from normalized LeanRank-data premise supervision. Positive edges are treated as proof-state-to-premise positives; negative candidates are treated as hard/failed candidate premises for ranking and difficulty features.

- Positive edges: `69461`
- Negative candidates: `663198`
- Negative/positive edge ratio: `9.547775010437512`
- Train proof states with positive edges: `23723`
- Train proof states with negative candidates: `23723`
- Train positive proof-state coverage: `1.0`
- Train negative proof-state coverage: `1.0`
- Train positive/negative pair overlap count: `0`
- Removed positive/negative label conflicts during normalization: `4088`
- Train negative-candidate hardness mean: `0.6030019312643399`
- Train high-hardness negative rows: `128855`
- Train high-hardness negative row share: `0.24293333685260354`
- Supervision quality checks: `{'all_negative_edges_have_valid_endpoints': True, 'all_positive_edges_have_valid_endpoints': True, 'all_positive_negative_pairs_disjoint': True}`
- Supervision scope: `erbacher/LeanRank-data normalized positive/negative premise supervision`

### Hard-Negative Quality Profile

Hardness buckets are derived from the normalized positive/negative premise features. This table shows whether the train split contains enough non-trivial negative candidates for ranking and difficulty experiments.

| Hardness bucket | Proof states | Negative rows | Row share | Mean hardness |
| --- | ---: | ---: | ---: | ---: |
| `none` | 0 | 0 | 0.0000 | 0.0000 |
| `low` | 8251 | 145935 | 0.2751 | 0.3731 |
| `medium` | 8470 | 255623 | 0.4819 | 0.6249 |
| `high` | 7002 | 128855 | 0.2429 | 0.8474 |

### Ranker Training Pair Utilization

This profile verifies that the learned premise ranker uses normalized LeanRank-data positive premise edges and hard-negative candidate edges directly, rather than relying only on theorem text similarity.

- Positive label source: `data/processed/train/positive_edges.parquet label=1`
- Hard-negative label source: `data/processed/train/negative_edges.parquet label=0`
- Raw train positive pairs: `54897`
- Raw train hard-negative pairs: `530413`
- Training sample positive pairs: `10000`
- Training sample hard-negative pairs: `10000`
- Training hard-negative/positive ratio: `1.0`
- Hardness feature column: `negative_candidate_hardness`
- Hard-negative pairs with nonzero hardness: `10000`
- Hard-negative nonzero hardness share: `1.0`
- Hard-negative mean hardness in ranker sample: `0.6125832683406032`
- Train feature construction seconds: `12.822568715084344`
- Validation feature construction seconds: `3.6532230398152024`
- Model fit seconds: `0.03800319111905992`
- Feature ablation seconds: `0.44068691600114107`
- Total ranker training seconds: `17.046396038029343`

## Pipeline Timing

- Total seconds: `499.3380202429835`
- Stage count: `20`
- Executed/skipped stages: `20` / `0`
- Timing config matches current report config: `True`
- Timing generated at: `2026-06-20T22:46:12.435080+00:00`
- Timing report: `outputs/reports/pipeline_run_timings.json`
- Evaluation internal total seconds: `19.37662909203209`
- Evaluation timed substages: `7`

| Stage | Seconds |
| --- | ---: |
| `embed` | 148.6884 |
| `sample` | 80.5672 |
| `augment_graph` | 50.8827 |
| `normalize` | 43.7230 |
| `validate` | 38.6715 |
| `compute_difficulty` | 36.4229 |
| `evaluate` | 19.4681 |
| `build_graph` | 17.3059 |
| `train_ranker` | 17.0510 |
| `benchmark_index` | 11.3741 |

### Evaluation Substage Timing

These timings split the `evaluate` pipeline stage into proof-state retrieval, theorem retrieval, reranked retrieval, and query-representation diagnostics so scaling work can target the slowest internal path.

| Evaluation substage | Seconds | Queries | Backend |
| --- | ---: | ---: | --- |
| `test_reranked_proof_state_retrieval` | 11.6008 | 20 | batched_torch_cuda_then_rerank |
| `val_proof_state_retrieval` | 4.9990 | 100 | torch_cuda |
| `val_proof_state_query_representation_diagnostic` | 1.1239 | 50 | n/a |
| `test_proof_state_query_representation_diagnostic` | 0.4401 | 50 | n/a |
| `test_proof_state_retrieval` | 0.3841 | 100 | torch_cuda |
| `test_theorem_retrieval` | 0.2961 | 50 | torch_cuda |
| `val_theorem_retrieval` | 0.1092 | 50 | torch_cuda |

### Rerank Evaluation Cost Profile

The reranked proof-state diagnostic follows the slower homepage/API-style path. This profile projects what full held-out reranking would cost and explains why the report keeps full coverage on batched embedding retrieval while sampling the reranked diagnostic.

- Method: `project_full_rerank_cost_from_user_facing_sampled_rerank_diagnostic`
- Backend: `batched_torch_cuda_then_rerank`
- Candidate k: `50`
- Sampled rerank queries: `20`
- Full proof-state queries: `100`
- Sampled fraction of full proof-state eval: `0.2`
- Rerank seconds/query: `0.5800416535465047`
- Batched embedding seconds/query: `0.0038407186302356422`
- Rerank/batched seconds per query: `151.02425076916322`
- Projected full rerank seconds: `58.004165354650475`
- Projected full rerank minutes: `0.9667360892441745`
- Sampled rerank Recall@10 delta: `0.0409341172079134`
- Policy: keep reranked proof-state evaluation sampled for development and use full batched embedding evaluation for final held-out coverage

## Resource And Parallelism Profile

This profile records the resource choices used by the committed LeanRank-data run. It is intended to explain which stages use GPU/vectorized paths and which stages remain CPU/IO-heavy.

### Execution Mode Summary

- Embedding mode: `multi_gpu_sentence_transformer`
- Multi-GPU embedding active: `True`
- Evaluation mode: `batched_gpu_retrieval_evaluation`
- Evaluation GPU active: `True`
- Index mode: `hnswlib_ann_candidate_generation`
- ANN index active: `True`
- Primary timed bottleneck: `embed`
- CPU/IO-heavy stages: `['sample', 'augment_graph', 'normalize', 'validate']`
- Artifact reuse by default: `True`
- Bottleneck interpretation: embedding is still the largest timed stage even with GPU encoding, so artifact reuse matters for report and reranking refreshes

### Embedding

- Backend/model: `sentence_transformers` / `BAAI/bge-base-en-v1.5`
- Requested device: `cuda`
- Devices: `['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6']`
- Device count: `7`
- Multi-process encoding: `True`
- Batch size: `512`
- Total embedding rows: `244391`
- Embedding rows/sec during embed stage: `1643.6453073837883`

### Evaluation

- Ranking backend: `batched_embedding_topk`
- Requested GPU: `use_gpu=True`, device `cuda:0`
- Actual backends: `['torch_cuda']`
- Test proof-state backend: `torch_cuda`
- Test theorem backend: `torch_cuda`
- Candidate count: `127561`
- Fallback reasons: `[]`

### Indexing

- Backend/requested backend: `hnswlib` / `auto`
- Metric: `cosine`
- hnswlib parameters: `M=16`, `ef_construction=200`, `ef_search=100`
- Indexed entities: `['premise', 'proof_state', 'theorem']`
- Mean speedup vs exact: `17.07151721843473`
- Minimum recall vs exact: `0.9500000000000002`

### CPU/IO-Heavy Stages

| Stage | Seconds | Share of total |
| --- | ---: | ---: |
| `sample` | 80.5672 | 0.1646 |
| `augment_graph` | 50.8827 | 0.1039 |
| `normalize` | 43.7230 | 0.0893 |
| `validate` | 38.6715 | 0.0790 |

### Performance Acceptance Gates

These gates summarize whether the committed performance evidence is strong enough for the current LeanRank-data retrieval report. Required gates cover data scale, held-out evaluation scope, timing freshness, and ANN quality; advisory gates cover GPU/resource usage and artifact reuse.

- Required gates passed: `False`
- Advisory gates passed: `True`
- Passed gates: `9` / `10`

| Gate | Severity | Passed | Value | Threshold |
| --- | --- | ---: | --- | --- |
| `target_dataset` | required | True | erbacher/LeanRank-data | erbacher/LeanRank-data |
| `large_scale_slice` | required | True | 292012 | >=60000 processed split rows and scale_bucket=large |
| `full_heldout_evaluation` | required | False | {'proof_state_coverage_fraction': 0.03275466754012447, 'theorem_coverage_fraction': 0.05} | both coverage fractions == 1.0 |
| `fresh_pipeline_timing` | required | True | {'scale_estimate_reliable': True, 'throughput_basis': 'executed_pipeline_run'} | scale_estimate_reliable=true and throughput_basis=executed_pipeline_run |
| `ann_speedup` | required | True | 17.0715 | >=5x mean indexed speedup vs exact cosine |
| `ann_recall` | required | True | 0.9500 | >=0.95 minimum Recall@10 vs exact cosine across indexed entities |
| `gpu_embedding_parallelism` | advisory | True | {'device_count': 7, 'multi_process': True, 'requested_device': 'cuda'} | cuda requested with at least one device |
| `gpu_evaluation_backend` | advisory | True | ['torch_cuda'] | actual_backends includes torch_cuda |
| `artifact_reuse_ready` | advisory | True | True | reuse_by_default=true |
| `embedding_throughput_recorded` | advisory | True | 1643.6453 | >0 embedding rows/sec |

## Pipeline Performance And Scale-Up Notes

- Pipeline profile: `outputs/reports/pipeline_performance_report.json`
- Scale bucket: `large`
- Requested theorems: `10000`
- Source rows requested: `350000`
- Current split rows: `292012`
- Target dataset confirmed: `True`
- LeanRank premise supervision ready: `True`
- Embedding devices: `['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6']`
- ANN backend availability: `{'faiss': False, 'hnswlib': True, 'lancedb': False}`
- Total embedding rows: `244391`
- Timing config matches current report config: `True`
- Throughput timing basis: `executed_pipeline_run`
- Scale estimate reliable: `True`
- Embedding rows by entity: `{'premise': 203601, 'proof_state': 30498, 'theorem': 10292}`
- Embedding rows by split: `{'demo': 2608, 'test': 42385, 'train': 159284, 'val': 40114}`
- Embedding matrix bytes: `684695627`
- Embed stage seconds: `148.68840552284382`
- Embed stage share of total: `0.3036933009277265`
- Embedding rows/sec during embed stage: `1643.6453073837883`
- Processed rows/sec: `596.4290751432031`
- Pipeline seconds per 100k processed rows: `167.66452905735676`
- Slowest timed stage: `embed`
- Saved pipeline evaluate seconds: `19.468068956863135`
- Current standalone evaluation seconds: `19.37662909203209`
- Timed/current evaluation ratio: `1.004719080104013`
- Primary bottleneck share: `0.3036933009277265`
- Top-3 timed-stage share: `0.5721772752516981`
- Mean index speedup vs exact: `17.07151721843473`
- Minimum index recall vs exact: `0.9500000000000002`
- Estimated seconds at requested source rows: `586.8258517007487`

| Bottleneck stage | Seconds | Share of total |
| --- | ---: | ---: |
| `embed` | 148.6884 | 0.3037 |
| `sample` | 80.5672 | 0.1646 |
| `augment_graph` | 50.8827 | 0.1039 |
| `normalize` | 43.7230 | 0.0893 |
| `validate` | 38.6715 | 0.0790 |

### Scale Projection

These linear projections use the current timed pipeline as a capacity-planning baseline. They are not substitutes for fresh timing runs after changing the sample shape, hardware, embedding model, or index backend.

- Projection method: `linear_projection_from_current_timed_pipeline`
- Projection reliable: `True`
- Current processed rows: `292012`
- Configured source rows: `350000`

| Projection | Target rows | Scale factor | Total seconds | Embed seconds | Index build seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_1x` | 292012 | 1.0000 | 489.6005 | 148.6884 | 6.3293 |
| `current_2x` | 584024 | 2.0000 | 979.2011 | 297.3768 | 12.6587 |
| `current_5x` | 1460060 | 5.0000 | 2448.0027 | 743.4420 | 31.6467 |
| `configured_source_rows` | 350000 | 1.1986 | 586.8259 | 178.2151 | 7.5862 |

### Artifact Storage Footprint

This profile records the local footprint of generated LeanRank-data artifacts. It is a practical scale-up signal because embeddings and ANN indexes can dominate disk usage before model training becomes the bottleneck.

- Method: `filesystem_artifact_footprint_with_linear_scale_projection`
- Total artifact bytes: `3052524471`
- Total artifact GiB: `2.8428849494084716`
- Bytes per processed row: `10453.421335424571`
- Unreferenced index artifact bytes: `1502501178`
- Unreferenced index artifact count: `12`

| Projection | Target rows | Scale factor | Artifact GiB |
| --- | ---: | ---: | ---: |
| `current_1x` | 292012 | 1.0000 | 2.8429 |
| `current_2x` | 584024 | 2.0000 | 5.6858 |
| `current_5x` | 1460060 | 5.0000 | 14.2144 |
| `configured_source_rows` | 350000 | 1.1986 | 3.4074 |

Largest generated artifact files:

| File | Bytes |
| --- | ---: |
| `outputs/indexes/train_premise_neighbors.joblib` | 784245864 |
| `outputs/indexes/train_premise_neighbors.bin` | 410816312 |
| `outputs/embeddings/train_premise_embeddings.npz` | 366556307 |
| `outputs/indexes/test_premise_neighbors.joblib` | 235665958 |
| `outputs/indexes/val_premise_neighbors.joblib` | 223124038 |
| `outputs/indexes/train_proof_state_neighbors.joblib` | 145849830 |
| `outputs/indexes/test_premise_neighbors.bin` | 123456824 |
| `outputs/indexes/val_premise_neighbors.bin` | 116887140 |

Unreferenced index artifacts not pointed to by current manifests:

| File | Bytes |
| --- | ---: |
| `outputs/indexes/train_premise_neighbors.joblib` | 784245864 |
| `outputs/indexes/test_premise_neighbors.joblib` | 235665958 |
| `outputs/indexes/val_premise_neighbors.joblib` | 223124038 |
| `outputs/indexes/train_proof_state_neighbors.joblib` | 145849830 |
| `outputs/indexes/train_theorem_neighbors.joblib` | 49184822 |
| `outputs/indexes/test_proof_state_neighbors.joblib` | 18770678 |
| `outputs/indexes/val_proof_state_neighbors.joblib` | 17350486 |
| `outputs/indexes/demo_premise_neighbors.joblib` | 8681798 |

## Refresh And Retraining Policy

Training is not repeated for every report or homepage refresh. The default workflow reuses LeanRank-data artifacts unless the data split, embedding representation, labels, or ranker features changed.

- Reuse by default: `True`
- Policy: Do not retrain by default. Reuse embeddings, indexes, and trained models for report/homepage refreshes; rerun ranker training only after ranker feature, label, split, or relevant config changes.
- Cached embedding rows: `244391`
- Cached embedding model: `BAAI/bge-base-en-v1.5`
- Indexed entity manifests: `12`
- Index backend: `hnswlib`
- Premise ranker artifact exists: `True`
- Difficulty estimator artifact exists: `True`

| Scenario | Re-embed | Retrain ranker | Re-evaluate | Commands |
| --- | ---: | ---: | ---: | --- |
| `report_or_homepage_refresh` | False | False | False | `leanrank-kg profile-pipeline`<br>`leanrank-kg build-experiment-report`<br>`leanrank-kg build-homepage`<br>`leanrank-kg audit` |
| `retrieval_or_ranking_code_change` | False | False | True | `leanrank-kg evaluate`<br>`leanrank-kg profile-pipeline`<br>`leanrank-kg build-experiment-report`<br>`leanrank-kg build-homepage` |
| `ranker_feature_or_label_change` | False | True | True | `leanrank-kg train-ranker`<br>`leanrank-kg evaluate`<br>`leanrank-kg profile-pipeline`<br>`leanrank-kg build-experiment-report` |
| `embedding_model_or_text_change` | True | True | True | `leanrank-kg embed`<br>`leanrank-kg build-index`<br>`leanrank-kg train-ranker`<br>`leanrank-kg evaluate` |
| `data_split_or_sample_change` | True | True | True | `leanrank-kg full-pipeline --config configs/proofatlas.yaml --force` |

## Recommendations

- `medium` `pipeline_bottleneck`: Embedding is the current largest timed bottleneck. Reuse cached embeddings when training/reranking only, and keep multi-GPU sentence-transformer encoding enabled for larger LeanRank-data refreshes.
- `medium` `retrieval_accuracy`: Proof-state Recall@100 is low, so proof-state premise retrieval is currently limited by candidate generation or embeddings before reranking. Prioritize stronger proof-state/query representations, domain-aware candidate pools, and embedding model comparisons before adding heavier rerankers.
- `medium` `evaluation_scope`: Current held-out metrics are sampled because evaluation limits are configured. Proof-state limits: {'test': 100, 'val': 100}; theorem limits: {'test': 50, 'val': 50}. For final quantitative claims, run `make refresh-production-full-eval` or rerun evaluation with these limits removed or raised enough to cover the full held-out split.

## Interpretation

This report treats ProofAtlas as an ML ranking and retrieval system on `erbacher/LeanRank-data`. The main quantitative claims are the held-out test-set ranking metrics above.
