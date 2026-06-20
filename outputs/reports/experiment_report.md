# ProofAtlas Experiment Report

## Experiment Setup

- Dataset: `erbacher/LeanRank-data`
- Source kind: `huggingface`
- Split counts: test: 28910, train: 234520, val: 28582
- Candidate pool: `train premise index`
- Label policy: held-out test positive_edges are used only for evaluation
- Evaluation scope: `sampled held-out splits`
- Proof-state evaluation limits: `{'test': 100, 'val': 100}`
- Theorem evaluation limits: `{'test': 50, 'val': 50}`
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
| `Recall@50` | 0.2795 |
| `Recall@100` | 0.3090 |
| `nDCG@50` | 0.1059 |
| `nDCG@100` | 0.1123 |

### Theorem Candidate Pool

| Metric | Value |
| --- | ---: |
| `theorem_retrieval_Recall@50` | 0.5942 |
| `theorem_retrieval_Recall@100` | 0.6642 |
| `theorem_retrieval_nDCG@50` | 0.4693 |
| `theorem_retrieval_nDCG@100` | 0.4879 |

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
| `evaluated_queries` | 100 |
| `evaluated_retrievable_queries` | 94 |
| `gold_premise_coverage` | 0.8634 |
| `Recall@1` | 0.0013 |
| `Recall@5` | 0.0962 |
| `Recall@10` | 0.1279 |
| `MRR` | 0.0754 |
| `MAP` | 0.0460 |
| `nDCG@10` | 0.0680 |

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
| `theorem_retrieval_evaluated_theorems` | 50 |
| `theorem_retrieval_evaluated_theorems_with_train_gold` | 49 |
| `theorem_retrieval_gold_premise_coverage` | 0.8679 |
| `theorem_retrieval_Recall@1` | 0.2434 |
| `theorem_retrieval_Recall@5` | 0.3850 |
| `theorem_retrieval_Recall@10` | 0.4233 |
| `theorem_retrieval_MRR` | 0.5473 |
| `theorem_retrieval_MAP` | 0.3560 |
| `theorem_retrieval_nDCG@10` | 0.4139 |

## Domain Breakdown

These tables show held-out test metrics grouped by LeanRank-data domain. They help identify where ranking quality is strong or weak instead of relying only on aggregate metrics.

### Test Proof-State-Level Domains

| Domain | Queries | `Recall@10` | `MRR` | `MAP` | `nDCG@10` | `gold_premise_coverage` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AlgebraicGeometry | 37 | 0.1071 | 0.0658 | 0.0394 | 0.0586 | 0.8272 |
| RingTheory | 24 | 0.0978 | 0.0368 | 0.0297 | 0.0435 | 0.8913 |
| LinearAlgebra | 9 | 0.0000 | 0.0127 | 0.0137 | 0.0000 | 1.0000 |
| NumberTheory | 9 | 0.3000 | 0.1262 | 0.0814 | 0.1336 | 0.8148 |
| Analysis | 5 | 0.2500 | 0.0357 | 0.0357 | 0.0833 | 0.7143 |
| Algebra | 4 | 0.1562 | 0.3750 | 0.0989 | 0.1599 | 0.8667 |
| FieldTheory | 4 | 0.2500 | 0.1385 | 0.1317 | 0.1577 | 0.8750 |
| Topology | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8333 |
| CategoryTheory | 1 | 0.2500 | 0.1250 | 0.0312 | 0.1232 | 1.0000 |
| Computability | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Data | 1 | 0.6667 | 0.5000 | 0.3889 | 0.5307 | 1.0000 |
| Geometry | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7500 |

### Test Theorem-Level Domains

| Domain | Queries | `theorem_retrieval_Recall@10` | `theorem_retrieval_MRR` | `theorem_retrieval_MAP` | `theorem_retrieval_nDCG@10` | `theorem_retrieval_gold_premise_coverage` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AlgebraicGeometry | 9 | 0.3988 | 0.5362 | 0.3404 | 0.4040 | 0.8056 |
| Analysis | 7 | 0.2778 | 0.4017 | 0.2619 | 0.2750 | 0.8500 |
| RingTheory | 7 | 0.2500 | 0.2319 | 0.1230 | 0.1919 | 0.9091 |
| LinearAlgebra | 6 | 0.5611 | 0.7244 | 0.4506 | 0.5347 | 1.0000 |
| .lake | 5 | 0.6286 | 0.6500 | 0.6193 | 0.6237 | 0.7059 |
| Algebra | 4 | 0.5312 | 0.7857 | 0.5264 | 0.5777 | 0.8667 |
| NumberTheory | 4 | 0.3352 | 0.5625 | 0.2959 | 0.3654 | 0.8889 |
| CategoryTheory | 1 | 0.5000 | 1.0000 | 0.3750 | 0.5585 | 1.0000 |
| Computability | 1 | 0.0000 | 0.0769 | 0.0562 | 0.0000 | 1.0000 |
| Data | 1 | 1.0000 | 1.0000 | 0.7556 | 0.8855 | 1.0000 |
| FieldTheory | 1 | 0.1429 | 0.1429 | 0.0243 | 0.0916 | 0.8750 |
| Geometry | 1 | 0.6667 | 1.0000 | 0.4444 | 0.6364 | 0.7500 |

## Error Analysis

Worst-case rows are held-out test queries with train-index gold premises but low top-k recovery. These are the first examples to inspect when improving embeddings, reranking features, or candidate depth.

### Worst Proof-State Queries

| Item | Domain | Gold in train | Missing gold | Recall@10 | MRR contribution | MAP contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ps:Algebra.trace_trace_of_basis:6:41dd1d089d19 | RingTheory | 7 | 0 | n/a | 0.0000 | 0.0000 |
| ps:AffineBasis.surjective_coord:0:4a0c7c385303 | LinearAlgebra | 6 | 0 | n/a | 0.0000 | 0.0000 |
| ps:Algebra.Etale.iff_exists_algEquiv_prod:4:7f4836d51cf7 | RingTheory | 5 | 0 | n/a | 0.0000 | 0.0000 |
| ps:AffineBasis.surjective_coord:10:9f704bb84c46 | LinearAlgebra | 4 | 0 | n/a | 0.0000 | 0.0000 |
| ps:AffineBasis.surjective_coord:6:8c422bb1b7a4 | LinearAlgebra | 4 | 0 | n/a | 0.0000 | 0.0000 |
| ps:AkraBazziRecurrence.growsPolynomially_id:4:7c914e3bc2e5 | Computability | 4 | 0 | n/a | 0.0000 | 0.0000 |
| ps:AddCircle.continuousAt_equivIoc:1:8e4f460e0671 | Topology | 3 | 1 | n/a | 0.0000 | 0.0000 |
| ps:Affine.Simplex.ne_altitudeFoot:1:2761c2382797 | Geometry | 3 | 1 | n/a | 0.0000 | 0.0000 |
| ps:AddLECancellable.lt_add_of_tsub_lt_left:0:a8fa4afcd330 | Algebra | 2 | 0 | n/a | 0.0000 | 0.0000 |
| ps:AffineSubspace.isClosed_direction_iff:1:4fe62edbe790 | Analysis | 2 | 0 | n/a | 0.0000 | 0.0000 |

### Worst Theorem Queries

| Item | Domain | Gold in train | Missing gold | Recall@10 | MRR contribution | MAP contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ArithmeticFunction.vonMangoldt.abscissaOfAbsConv_residueClass_le_one | NumberTheory | 15 | 0 | n/a | 0.0000 | 0.0000 |
| Algebra.Etale.iff_exists_algEquiv_prod | RingTheory | 7 | 2 | n/a | 0.0000 | 0.0000 |
| Array.pairwise_extract | .lake | 1 | 3 | n/a | 0.0000 | 0.0000 |
| AlgebraicGeometry.isFinite_iff_locallyOfFiniteType_of_jacobsonSpace | AlgebraicGeometry | 19 | 7 | n/a | 0.2000 | 0.0506 |
| Algebra.trace_trace_of_basis | RingTheory | 15 | 0 | n/a | 0.1667 | 0.0322 |
| Besicovitch.SatelliteConfig.inter' | MeasureTheory | 4 | 0 | n/a | 0.2500 | 0.0625 |
| AlgEquiv.restrictNormalHom_id | FieldTheory | 7 | 1 | n/a | 0.1429 | 0.0243 |
| AffineBasis.surjective_coord | LinearAlgebra | 6 | 0 | n/a | 0.0133 | 0.0058 |
| Asymptotics.IsBigOWith.add | Analysis | 3 | 0 | n/a | 0.0500 | 0.0167 |
| AlgebraicGeometry.HasRingHomProperty.stalkwise | AlgebraicGeometry | 3 | 4 | n/a | 0.0588 | 0.0196 |

## Validation Metrics

Validation metrics are reported for model selection and sanity checking; final claims should use the held-out test metrics above.

### Validation Proof-State-Level Premise Ranking

| Metric | Value |
| --- | ---: |
| `evaluated_queries` | 100 |
| `evaluated_retrievable_queries` | 89 |
| `gold_premise_coverage` | 0.8549 |
| `Recall@1` | 0.0232 |
| `Recall@5` | 0.0640 |
| `Recall@10` | 0.1146 |
| `MRR` | 0.0782 |
| `MAP` | 0.0526 |
| `nDCG@10` | 0.0706 |

### Validation Theorem-Level Premise Ranking

| Metric | Value |
| --- | ---: |
| `theorem_retrieval_evaluated_theorems` | 50 |
| `theorem_retrieval_evaluated_theorems_with_train_gold` | 44 |
| `theorem_retrieval_gold_premise_coverage` | 0.8781 |
| `theorem_retrieval_Recall@1` | 0.1255 |
| `theorem_retrieval_Recall@5` | 0.3323 |
| `theorem_retrieval_Recall@10` | 0.3712 |
| `theorem_retrieval_MRR` | 0.4833 |
| `theorem_retrieval_MAP` | 0.2506 |
| `theorem_retrieval_nDCG@10` | 0.3299 |

## Index Benchmark

| Entity | Backend | Rows | Exact ms/query | Indexed ms/query | Speedup | Recall vs exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| premise | hnswlib | 127561 | 69.1207 | 3.7731 | 18.3195 | 0.9890 |
| proof_state | hnswlib | 23723 | 12.7723 | 0.8401 | 15.2030 | 0.9910 |
| theorem | hnswlib | 8000 | 4.0895 | 0.3204 | 12.7647 | 0.9940 |

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

- Total seconds: `4.03193892003037`
- Stage count: `19`
- Timing report: `outputs/reports/pipeline_run_timings.json`

| Stage | Seconds |
| --- | ---: |
| `evaluate` | 0.7713 |
| `validate` | 0.7056 |
| `normalize` | 0.6338 |
| `compute_difficulty` | 0.5078 |
| `train_ranker` | 0.2962 |
| `augment_graph` | 0.2610 |
| `build_graph` | 0.1665 |
| `homepage` | 0.1395 |
| `embed` | 0.1111 |
| `pipeline_profile` | 0.1068 |

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

## Recommendations

- `low` `monitoring`: Keep this profile report as the baseline and compare it after every larger LeanRank-data refresh.

## Interpretation

This report treats ProofAtlas as an ML ranking and retrieval system on `erbacher/LeanRank-data`. The main quantitative claims are the held-out test-set ranking metrics above.
