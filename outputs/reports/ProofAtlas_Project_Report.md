# ProofAtlas Project Report

Split: `test`  
LLM theorem enrichment: `True`  
BGE pretrained embeddings: `True`  
BGE model: `BAAI/bge-base-en-v1.5`

## Midterm Milestones

### Milestone 1: Data pipeline

- theorem-disjoint Lean proof split
- 127,561 train-side premise pool
- 3,053 held-out proof states
- 7,054 gold premise edges

### Milestone 2: Retrieval system

- dense + lexical baseline
- similar proof-state expansion
- LLM-enriched theorem-neighborhood evidence
- weighted RRF fusion

### Milestone 3: Midterm result

- Covered Recall@100 +0.2384
- All-positive Recall@100 +0.2223
- nDCG@10 +0.0539

### Milestone 4: Interpretability/guidance

- theorem neighborhood views
- premise suggestions
- strategy/difficulty facets

## Terms and Abbreviations

- Lean: a theorem prover and proof assistant used to formalize mathematics.
- LeanRank: the processed Lean proof dataset source used by this experiment.
- Proof state: the current goal, hypotheses, and local context at a point in a Lean proof.
- Premise: a theorem, lemma, definition, or fact that may help prove a proof state.
- Gold premise edge: a held-out labeled link between a proof state and a premise used in the reference proof.
- Theorem-disjoint split: a train/test split where test theorems do not appear in training.
- Train-side premise pool: the candidate premises available for retrieval from the training side.
- Theorem neighborhood: similar train theorems retrieved for a held-out theorem or proof state.
- LLM: large language model; in ProofAtlas it produces retrieval text, not evaluation labels.
- TF-IDF: term frequency-inverse document frequency, a lexical text retrieval representation.
- BGE: an external pretrained embedding model used as an auxiliary variant; the exact model name is recorded in the report header.
- Dense retrieval: nearest-neighbor retrieval using vector embeddings.
- Lexical retrieval: retrieval based on token overlap or weighted text features such as TF-IDF.
- RRF: reciprocal-rank fusion, a method for combining ranked candidate lists.
- Recall@k: the fraction of relevant gold items found within the top `k` retrieved candidates.
- R@k: short chart/report notation for Recall@k, such as R@10 or R@100.
- Covered Recall: recall over gold premises that are present in the retrievable train-side premise pool.
- All-positive Recall: recall over all held-out positive premise edges, including non-retrievable positives.
- MAP: mean average precision, a ranking-quality metric over the retrieved list.
- nDCG@10: normalized discounted cumulative gain at rank 10, a top-rank quality metric.
- Challenge evaluation: proof-state to premise retrieval from the full train-side candidate pool.
- Mechanism check: theorem to theorem-neighborhood retrieval, used to test whether enriched theorem profiles expose reusable premise evidence.
- Retrieval evidence view: an explanation layer showing theorem neighbors, premise suggestions, and strategy facets behind retrieved candidates.

## Executive Summary

ProofAtlas is a midterm research system for theorem-neighborhood retrieval in Lean premise selection. The midterm contribution is threefold:

- built a theorem-neighborhood retrieval pipeline over a theorem-disjoint Lean proof split;
- validated that LLM-enriched theorem profiles improve theorem-neighbor premise evidence;
- tested the resulting evidence on the harder Proof-State -> Premise retrieval challenge.

On the theorem-neighborhood retrieval task, LLM-enriched TF-IDF theorem profiles improve neighbor-premise Recall@100 from `0.3185` to `0.3774` and nDCG@10 from `0.1499` to `0.1717`.

On the harder proof-state premise retrieval challenge, the primary non-BGE method, `weighted_rrf_llm_theorem_tuned`, improves top-100 candidate generation over dense retrieval:

| Metric | Dense baseline | Primary method | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Covered Recall@100 | 0.2362 | 0.4746 | +0.2384 | +101.0% |
| All-positive Recall@100 | 0.1851 | 0.4074 | +0.2223 | +120.1% |
| MAP | 0.0494 | 0.0981 | +0.0487 | +98.6% |
| nDCG@10 | 0.0697 | 0.1236 | +0.0539 | +77.4% |

The challenge result shows that the pipeline improves hard candidate generation from a 127,561-premise train-side pool, but it also identifies the next bottleneck: top-rank precision remains a reranking problem. If the objective is maximum candidate-pool recall, the auxiliary BGE fusion is the best reported proof-state premise retrieval variant on the test split; the primary method remains the main non-BGE result.

## Dataset

The experiment uses the processed LeanRank-derived in-distribution split in `data/processed`. Train, validation, and test are theorem-disjoint. Validation/test positives are held out for evaluation.

| Split | Theorems | Proof states | Premises | Positive edges | Avg proof states/theorem | Avg positives/proof-state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 8000 | 23723 | 127561 | 54897 | 2.9654 | 2.3141 |
| test | 1000 | 3053 | 38332 | 7054 | 3.0530 | 2.3105 |

### Top Theorem Domains

| Domain | Train theorems | Test theorems | Report split share |
| --- | ---: | ---: | ---: |
| Algebra | 1193 | 149 | 14.9% |
| Data | 1039 | 146 | 14.6% |
| Analysis | 1010 | 116 | 11.6% |
| Topology | 596 | 78 | 7.8% |
| RingTheory | 630 | 70 | 7.0% |
| MeasureTheory | 505 | 66 | 6.6% |
| CategoryTheory | 394 | 54 | 5.4% |
| LinearAlgebra | 439 | 50 | 5.0% |
| Order | 367 | 46 | 4.6% |
| Other | 1827 | 225 | 22.5% |

## Method

The retrieval pipeline starts with dense proof-state/premise embeddings and lexical TF-IDF profiles. It then adds structured candidate sources:

- similar proof-state expansion: retrieve similar train proof states, then reuse their premises;
- similar theorem premise expansion: retrieve similar train theorems, then reuse premises from their proof states;
- LLM-enriched theorem profiles: add semantic, strategy-oriented, and difficulty-oriented natural-language theorem profile text for theorem-neighborhood retrieval.

The primary method is:

```text
weighted_rrf_llm_theorem_tuned
```

It uses weighted RRF over dense, lexical, rank-based proof-state expansion, and the LLM-enriched similar-theorem premise source. BGE pretrained variants are reported as auxiliary ablations, not as the primary method, because they add a separate pretrained embedding dependency beyond the selected LLM-theorem fusion configuration. For recall-oriented deployment, `weighted_rrf_llm_pretrained_tuned` is the stronger test-split variant; for the paper's main non-BGE claim and ranking-quality comparison, `weighted_rrf_llm_theorem_tuned` is the primary method.

### Evaluation Scope

Covered Recall@k is macro-averaged over proof states after filtering each proof state to positives that are present in the train-side retrievable premise pool. `Gold coverage` is the fraction of all held-out positive edges whose premise appears in that retrievable pool.

All-positive Recall@100 is computed directly as edge-level hits@100 divided by all held-out positive edges, including positives absent from the train-side pool. It is not derived by multiplying covered Recall@100 by gold coverage.

Recall@100 is used as a candidate-generation metric: it measures whether the retrieval system can place useful premises into a manageable downstream candidate pool. We also report Recall@10, MAP, and nDCG@10 to measure top-rank quality.

MAP and nDCG are computed on the same filtered evaluation set as covered recall.

### Leakage Control

LLM theorem enrichment is used as retrieval text, not as an evaluation label. The enrichment prompt is built from theorem metadata and a bounded number of proof-state goals, hypotheses, and symbols. It does not include validation/test positive premise labels, negative candidates, existing strategy labels, existing difficulty buckets/scores, or proof scripts/tactics as target answers.

The proof-state text is still a metadata-level retrieval signal: local hypotheses and symbols may contain library identifiers or named facts from the state. This is therefore not a theorem-statement-only setting.

### Test Status

This report is a held-out test-split result generated after fixing the selected validation configuration. Methods with `tuned` in the name still reflect hyperparameters selected during validation/model selection, while the metrics here are computed on the held-out test split.

## Capability 1: LLM-Enriched Theorem Neighborhood Retrieval

Theorem retrieval is used to test whether theorem-neighborhood profiles carry reusable premise evidence. Strategy coverage is included as an auxiliary guidance diagnostic.

| Method | Neighbor premise Recall@10 | Recall@50 | Recall@100 | MAP | nDCG@10 | Strategy coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline theorem profile | 0.1728 | 0.2841 | 0.3185 | 0.1102 | 0.1499 | 0.9952 |
| LLM-enriched TF-IDF profile | **0.1985** | **0.3279** | **0.3774** | **0.1257** | **0.1717** | **0.9978** |
| LLM-enriched BGE profile | 0.1757 | 0.2875 | 0.3369 | 0.1082 | 0.1480 | 0.9965 |

LLM-enriched TF-IDF profiles give the strongest theorem-neighborhood premise retrieval. The strategy signal should be read as broad coverage from neighbor labels, not as a standalone proof-strategy prediction claim.

## Capability 2: Interpretable Retrieval Evidence

This section is not a separate performance benchmark. It describes the qualitative explanation layer produced by ProofAtlas: for a query theorem, the system exposes similar train theorems, neighbor-derived premise suggestions, strategy facets, difficulty signals, and retrieval methods that contributed evidence.

Generated evidence bundles: `25`

These bundles are qualitative artifacts: they are the deterministic first `limit` rows from the theorem-neighbor artifact, not a stratified sample and not a full-split aggregate.

| Metric | Value |
| --- | ---: |
| Evidence bundles | 25 |
| Avg source theorem neighbors/bundle | 10.0000 |
| Avg premise suggestions/bundle | 10.0000 |
| Avg strategy facets/bundle | 7.8800 |
| Avg difficulty evidence neighbors/bundle | 20.0000 |

### Aggregated Guidance Evidence

| Evidence type | Top evidence | Aggregate score |
| --- | --- | ---: |
| Premise suggestion | mul_comm | 1.9984 |
| Premise suggestion | RingHom.toMorphismProperty | 1.4401 |
| Premise suggestion | eq_comm | 1.4070 |
| Premise suggestion | tsub_le_iff_right | 1.3333 |
| Premise suggestion | Algebra.norm_apply | 1.2500 |
| Strategy facet | rewrite_transport | 31.6108 |
| Strategy facet | order_inequality_reasoning | 21.2267 |
| Strategy facet | typeclass_instance_resolution | 19.7442 |
| Strategy facet | algebraic_computation | 17.3210 |
| Strategy facet | case_analysis | 14.0014 |

### Example Bundles

| Theorem | Domain | Neighbors | Premise suggestions | Strategy facets | Difficulty bucket |
| --- | --- | ---: | ---: | ---: | --- |
| AList.lookup_to_alist | Data | 10 | 10 | 8 | medium |
| AbsoluteValue.Completion.extensionEmbedding_of_comp_coe | Analysis | 10 | 10 | 8 | easy |
| Action.full_res | CategoryTheory | 10 | 10 | 5 | easy |
| AddChar.to_mulShift_inj_of_isPrimitive | NumberTheory | 10 | 10 | 8 | easy |
| AddCircle.continuousAt_equivIoc | Topology | 10 | 10 | 8 | easy |

## Challenge Evaluation: Proof-State -> Premise Retrieval

This is the hard benchmark: given a held-out proof state, retrieve useful premises from 127,561 train-side candidate premises. The main table emphasizes the progression from simple baselines to theorem-neighborhood fusion.

| Stage | Method | Covered Recall@10 | Covered Recall@50 | Covered Recall@100 | All-positive Recall@100 | MAP | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | `dense` | 0.1162 | 0.1961 | 0.2362 | 0.1851 | 0.0494 | 0.0697 |
| Lexical baseline | `lexical` | 0.1245 | 0.2255 | 0.2713 | 0.2182 | 0.0523 | 0.0736 |
| Dense + lexical | `dense_lexical_rrf` | 0.1407 | 0.2476 | 0.2915 | 0.2324 | 0.0581 | 0.0825 |
| Multi-source fusion | `full_union_rrf` | 0.1640 | 0.3197 | 0.3888 | 0.3177 | 0.0705 | 0.0949 |
| Proof-state expansion tuned | `weighted_rrf_tuned_recall` | 0.1854 | 0.3562 | 0.4408 | 0.3699 | 0.0861 | 0.1122 |
| Theorem-neighborhood fusion | `weighted_rrf_theorem_tuned` | 0.1878 | 0.3791 | 0.4618 | 0.3947 | 0.0929 | 0.1180 |
| Primary method | `weighted_rrf_llm_theorem_tuned` | 0.1943 | 0.3894 | 0.4746 | 0.4074 | **0.0981** | **0.1236** |
| Auxiliary BGE fusion | `weighted_rrf_llm_pretrained_tuned` | **0.2000** | **0.3919** | **0.4813** | **0.4124** | 0.0964 | 0.1230 |

Gold coverage is constant at `0.9076` for these runs.

The primary method improves covered Recall@100 from `0.2362` to `0.4746`, direct all-positive Recall@100 from `0.1851` to `0.4074`, and nDCG@10 from `0.0697` to `0.1236` over dense retrieval. The challenge result shows that the pipeline improves hard candidate generation, but top-rank precision remains the next-stage bottleneck.

## Analysis

The midterm result should be read in capability-first order. The theorem-neighborhood mechanism check shows that LLM-enriched theorem profiles improve theorem-neighborhood premise retrieval. The interpretable retrieval evidence view shows that the system exposes theorem-neighbor, premise, strategy, and difficulty evidence behind retrieved candidates. The proof-state premise retrieval challenge then tests whether those capabilities help on the harder retrieval task.

LLM theorem enrichment improves theorem-neighborhood retrieval by adding semantic, strategy-oriented, and difficulty-oriented natural-language profile text. This enters the proof-state premise retrieval challenge system as one fused source inside weighted RRF, not as an evaluation label.

BGE pretrained embeddings provide a useful auxiliary channel but are not dominant across all ranking metrics. On the test split, the combined LLM+BGE fusion gives the strongest covered and all-positive Recall@100, so it is the best recall-oriented candidate generator. The primary LLM theorem-tuned method remains the main non-BGE result and gives stronger MAP and nDCG@10.

Single-source expansion methods are best interpreted as complementary candidate sources rather than standalone final retrievers. Their value is clearest after weighted fusion.

The remaining bottleneck is ranking quality near the top of the list. Covered Recall@100 is much higher than covered Recall@10, so many useful premises enter the candidate pool but are not consistently ranked high enough. A learned reranker over the fused candidate set is the most direct next step.

## Conclusion

The held-out result suggests theorem-theorem retrieval can provide useful auxiliary premise evidence when fused with proof-state retrieval signals.

At midterm, the stronger claim is not that final high-precision premise ranking is solved. The stronger claim is that ProofAtlas has built the theorem-neighborhood retrieval/guidance capability, validated the LLM-enriched theorem-neighborhood mechanism, and shown a meaningful improvement on the hard candidate-generation challenge. Top-rank precision remains the next-stage bottleneck.

## Appendix: Full Challenge Ablation Table

| Method | Covered Recall@10 | Covered Recall@50 | Covered Recall@100 | All-positive Recall@100 | MAP | nDCG@10 | Gold coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.1162 | 0.1961 | 0.2362 | 0.1851 | 0.0494 | 0.0697 | 0.9076 |
| lexical | 0.1245 | 0.2255 | 0.2713 | 0.2182 | 0.0523 | 0.0736 | 0.9076 |
| dense_lexical_rrf | 0.1407 | 0.2476 | 0.2915 | 0.2324 | 0.0581 | 0.0825 | 0.9076 |
| symbol_overlap | 0.0561 | 0.1064 | 0.1339 | 0.1001 | 0.0287 | 0.0373 | 0.9076 |
| similar_proof_state_expansion | 0.1344 | 0.2305 | 0.2335 | 0.2064 | 0.0723 | 0.0920 | 0.9076 |
| similar_proof_state_expansion_k50 | 0.1363 | 0.2475 | 0.2900 | 0.2580 | 0.0742 | 0.0932 | 0.9076 |
| similar_proof_state_expansion_k100 | 0.1389 | 0.2528 | 0.3052 | 0.2737 | 0.0752 | 0.0945 | 0.9076 |
| similar_proof_state_expansion_k100_sim | 0.1128 | 0.2482 | 0.3061 | 0.2743 | 0.0563 | 0.0734 | 0.9076 |
| similar_proof_state_expansion_k100_rank_sim | 0.1387 | 0.2532 | 0.3049 | 0.2737 | 0.0753 | 0.0945 | 0.9076 |
| similar_theorem_premises | 0.1379 | 0.2583 | 0.3041 | 0.2742 | 0.0722 | 0.0925 | 0.9076 |
| full_union_rrf | 0.1640 | 0.3197 | 0.3888 | 0.3177 | 0.0705 | 0.0949 | 0.9076 |
| weighted_rrf_balanced | 0.1686 | 0.3275 | 0.3963 | 0.3248 | 0.0731 | 0.0982 | 0.9076 |
| weighted_rrf_lexical_ps | 0.1803 | 0.3394 | 0.4119 | 0.3412 | 0.0791 | 0.1060 | 0.9076 |
| weighted_rrf_ps_heavy | 0.1859 | 0.3426 | 0.4179 | 0.3487 | 0.0844 | 0.1115 | 0.9076 |
| weighted_rrf_ps_heavy_k50 | 0.1861 | 0.3475 | 0.4302 | 0.3587 | 0.0852 | 0.1120 | 0.9076 |
| weighted_rrf_ps_heavy_k100 | 0.1856 | 0.3538 | 0.4361 | 0.3652 | 0.0857 | 0.1121 | 0.9076 |
| weighted_rrf_ps_heavy_k100_sim | 0.1684 | 0.3483 | 0.4359 | 0.3638 | 0.0704 | 0.0958 | 0.9076 |
| weighted_rrf_ps_heavy_k100_rank_sim | 0.1856 | 0.3530 | 0.4368 | 0.3659 | 0.0857 | 0.1121 | 0.9076 |
| weighted_rrf_tuned_frontier | 0.1866 | 0.3554 | 0.4380 | 0.3679 | 0.0865 | 0.1129 | 0.9076 |
| weighted_rrf_tuned_recall | 0.1854 | 0.3562 | 0.4408 | 0.3699 | 0.0861 | 0.1122 | 0.9076 |
| weighted_rrf_theorem_source | 0.1930 | 0.3773 | 0.4557 | 0.3877 | 0.0903 | 0.1171 | 0.9076 |
| weighted_rrf_theorem_heavy | 0.1891 | 0.3781 | 0.4612 | 0.3937 | 0.0898 | 0.1155 | 0.9076 |
| weighted_rrf_theorem_frontier | 0.1940 | 0.3768 | 0.4558 | 0.3866 | 0.0898 | 0.1170 | 0.9076 |
| weighted_rrf_theorem_tuned | 0.1878 | 0.3791 | 0.4618 | 0.3947 | 0.0929 | 0.1180 | 0.9076 |
| similar_theorem_premises_llm_enriched | 0.1570 | 0.2868 | 0.3448 | 0.3088 | 0.0835 | 0.1065 | 0.9076 |
| pretrained_dense | 0.1122 | 0.2005 | 0.2332 | 0.1807 | 0.0496 | 0.0684 | 0.9076 |
| pretrained_similar_proof_state_expansion | 0.1340 | 0.2594 | 0.3118 | 0.2771 | 0.0730 | 0.0907 | 0.9076 |
| weighted_rrf_llm_theorem_source | 0.1969 | 0.3838 | 0.4661 | 0.3965 | 0.0924 | 0.1194 | 0.9076 |
| weighted_rrf_llm_theorem_tuned | 0.1943 | 0.3894 | 0.4746 | 0.4074 | 0.0981 | 0.1236 | 0.9076 |
| weighted_rrf_pretrained_tuned | 0.1917 | 0.3837 | 0.4703 | 0.4010 | 0.0917 | 0.1169 | 0.9076 |
| weighted_rrf_llm_pretrained_tuned | 0.2000 | 0.3919 | 0.4813 | 0.4124 | 0.0964 | 0.1230 | 0.9076 |

## Next Phase

- learned reranker
- prover-in-the-loop evaluation
- statement-only/no-symbol leakage ablations
- larger theorem-neighborhood evaluation
- improve top-10 ranking quality
