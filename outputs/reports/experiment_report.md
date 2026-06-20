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
- Proof-state test coverage: `3053` / `3053` (`1.0`)
- Theorem test coverage: `1000` / `1000` (`1.0`)
- Ranking backend: `batched_embedding_topk`
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

### Proof-State Query Representation Diagnostic

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
| `Recall@1` | 0.0263 |
| `Recall@5` | 0.1206 |
| `Recall@10` | 0.1513 |
| `MRR` | 0.1456 |
| `MAP` | 0.0766 |
| `nDCG@10` | 0.1148 |

#### Rerank Candidate-Depth Ablation

This ablation uses the same held-out rerank diagnostic queries and changes only the number of embedding candidates passed into the learned/fixed reranker.

| Candidate k | Recall@1 | Recall@5 | Recall@10 | MRR | MAP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.0263 | 0.1206 | 0.1513 | 0.1456 | 0.0766 |
| 100 | 0.0263 | 0.1140 | 0.1382 | 0.1316 | 0.0742 |

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
| premise | hnswlib | 127561 | 69.2717 | 3.4585 | 20.0292 | 0.9900 | 0.9960 | 0.9930 | 0.9900 | 3.7882 | 0.3459 |
| proof_state | hnswlib | 23723 | 12.7351 | 0.8292 | 15.3578 | 0.7900 | 0.9600 | 0.9640 | 0.7900 | 0.4042 | 0.0829 |
| theorem | hnswlib | 8000 | 4.2275 | 0.2369 | 17.8457 | 0.9900 | 0.9900 | 0.9930 | 0.9900 | 0.1338 | 0.0237 |

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
- Supervision quality checks: `{'all_negative_edges_have_valid_endpoints': True, 'all_positive_edges_have_valid_endpoints': True, 'all_positive_negative_pairs_disjoint': True}`
- Supervision scope: `erbacher/LeanRank-data normalized positive/negative premise supervision`

## Pipeline Timing

- Total seconds: `740.1227513239719`
- Stage count: `19`
- Executed/skipped stages: `19` / `0`
- Timing config matches current report config: `True`
- Timing generated at: `2026-06-20T17:20:52.729952+00:00`
- Timing report: `outputs/reports/pipeline_run_timings.json`
- Evaluation internal total seconds: `24.223363942001015`
- Evaluation timed substages: `6`

| Stage | Seconds |
| --- | ---: |
| `evaluate` | 211.3259 |
| `embed` | 148.8000 |
| `sample` | 81.1717 |
| `train_ranker` | 56.3446 |
| `augment_graph` | 50.8757 |
| `compute_difficulty` | 48.2069 |
| `normalize` | 43.3543 |
| `validate` | 37.5526 |
| `build_graph` | 17.3007 |
| `benchmark_index` | 11.3732 |

### Evaluation Substage Timing

These timings split the `evaluate` pipeline stage into proof-state retrieval, theorem retrieval, reranked retrieval, and query-representation diagnostics so scaling work can target the slowest internal path.

| Evaluation substage | Seconds | Queries | Backend |
| --- | ---: | ---: | --- |
| `test_reranked_proof_state_retrieval` | 14.5914 | 20 | batched_torch_cuda_then_rerank |
| `val_proof_state_retrieval` | 6.9131 | 2822 | torch_cuda |
| `test_proof_state_retrieval` | 0.9482 | 3053 | torch_cuda |
| `test_proof_state_query_representation_diagnostic` | 0.5439 | 50 | n/a |
| `test_theorem_retrieval` | 0.4424 | 1000 | torch_cuda |
| `val_theorem_retrieval` | 0.3381 | 1000 | torch_cuda |

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
- Processed rows/sec: `394.5453635598055`
- Pipeline seconds per 100k processed rows: `253.45627964740214`
- Slowest timed stage: `evaluate`
- Primary bottleneck share: `0.28552813073522093`
- Top-3 timed-stage share: `0.5962491304310673`
- Mean index speedup vs exact: `17.744239287034144`
- Minimum index recall vs exact: `0.9640000000000002`
- Estimated seconds at requested source rows: `887.0969787659075`

| Bottleneck stage | Seconds | Share of total |
| --- | ---: | ---: |
| `evaluate` | 211.3259 | 0.2855 |
| `embed` | 148.8000 | 0.2010 |
| `sample` | 81.1717 | 0.1097 |
| `train_ranker` | 56.3446 | 0.0761 |
| `augment_graph` | 50.8757 | 0.0687 |

## Recommendations

- `medium` `pipeline_bottleneck`: Evaluation is the current largest timed bottleneck. Keep full held-out metrics for final claims, but use sampled evaluation during development and prioritize batched/vectorized scoring or parallel domain shards before scaling evaluation further.

## Interpretation

This report treats ProofAtlas as an ML ranking and retrieval system on `erbacher/LeanRank-data`. The main quantitative claims are the held-out test-set ranking metrics above.
