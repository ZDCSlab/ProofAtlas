# ProofAtlas Experiment Report

Split: `val`  
LLM theorem enrichment: `True`  
BGE pretrained embeddings: `True`  
BGE model: `BAAI/bge-base-en-v1.5`

## Summary

ProofAtlas evaluates whether theorem-neighborhood structure in Lean proof data can improve premise retrieval. The main task is Proof-State -> Premise retrieval: given a held-out proof state, retrieve useful train-side premises.

The primary method, `weighted_rrf_llm_theorem_tuned`, combines dense retrieval, lexical retrieval, rank-based similar proof-state expansion, and LLM-enriched similar-theorem premise evidence using weighted reciprocal-rank fusion.

On the validation split, the primary method nearly doubles both covered Recall@100 and MAP over dense retrieval:

| Metric | Dense baseline | Primary method | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Covered Recall@100 | 0.2426 | 0.4810 | +0.2384 | +98.3% |
| Overall Recall@100 | 0.2208 | 0.4376 | +0.2168 | +98.2% |
| MAP | 0.0511 | 0.1006 | +0.0495 | +96.9% |
| nDCG@10 | 0.0698 | 0.1285 | +0.0587 | +84.1% |

The strongest supported conclusion is that theorem-neighborhood evidence substantially improves premise retrieval when fused with proof-state retrieval signals.

## Dataset

The experiment uses the processed LeanRank-derived in-distribution split in `data/processed`. Train, validation, and test are theorem-disjoint. Validation/test positives are held out for evaluation.

| Split | Theorems | Proof states | Premises | Positive edges | Avg proof states/theorem | Avg positives/proof-state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 8000 | 23723 | 127561 | 54897 | 2.9654 | 2.3141 |
| val | 1000 | 2822 | 36292 | 6610 | 2.8220 | 2.3423 |

### Top Theorem Domains

| Domain | Train theorems | Val theorems | Report split share |
| --- | ---: | ---: | ---: |
| Algebra | 1193 | 146 | 14.6% |
| Analysis | 1010 | 131 | 13.1% |
| Data | 1039 | 118 | 11.8% |
| Topology | 596 | 75 | 7.5% |
| RingTheory | 630 | 71 | 7.1% |
| MeasureTheory | 505 | 65 | 6.5% |
| Order | 367 | 62 | 6.2% |
| CategoryTheory | 394 | 55 | 5.5% |
| LinearAlgebra | 439 | 53 | 5.3% |
| Other | 1827 | 224 | 22.4% |

## Method

The retrieval pipeline starts with dense proof-state/premise embeddings and lexical TF-IDF profiles. It then adds structured candidate sources:

- similar proof-state expansion: retrieve similar train proof states, then reuse their premises;
- similar theorem premise expansion: retrieve similar train theorems, then reuse premises from their proof states;
- LLM-enriched theorem profiles: add semantic, strategy-oriented, and difficulty-oriented natural-language theorem profile text for theorem-neighborhood retrieval.

The primary method is:

```text
weighted_rrf_llm_theorem_tuned
```

It uses weighted RRF over dense, lexical, rank-based proof-state expansion, and the LLM-enriched similar-theorem premise source. BGE pretrained variants are reported as auxiliary ablations, not as the primary method.

### Evaluation Scope

Recall@k is reported over positives that are present in the train-side retrievable premise pool. `Gold coverage` is the fraction of all held-out positive edges whose premise appears in that retrievable pool. `Overall Recall@100` multiplies covered Recall@100 by gold coverage to show the approximate all-positive recall after accounting for missing gold positives.

Recall@100 is used as a candidate-generation metric: it measures whether the retrieval system can place useful premises into a manageable downstream candidate pool. We also report Recall@10, MAP, and nDCG@10 to measure top-rank quality.

MAP and nDCG are computed on the same filtered evaluation set as covered recall.

### Leakage Control

LLM theorem enrichment is used as retrieval text, not as an evaluation label. The enrichment prompt is built from theorem metadata and a bounded number of proof-state goals, hypotheses, and symbols. It does not include validation/test positive premise labels, negative candidates, existing strategy labels, existing difficulty buckets/scores, or proof scripts/tactics as target answers.

The proof-state text is still a metadata-level retrieval signal: local hypotheses and symbols may contain library identifiers or named facts from the state. This is therefore not a theorem-statement-only setting.

### Validation Status

This report is a validation-split result. Methods with `tuned` in the name should be interpreted as validation/model-selection evidence unless their hyperparameters were selected on a separate development split. Final generalization should be confirmed on the held-out test split after fixing the selected configuration.

## T1 Main Result: Proof-State -> Premise Retrieval

The main result table emphasizes the progression from simple baselines to theorem-neighborhood fusion.

| Stage | Method | Covered Recall@10 | Covered Recall@50 | Covered Recall@100 | Overall Recall@100 | MAP | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | `dense` | 0.1142 | 0.1987 | 0.2426 | 0.2208 | 0.0511 | 0.0698 |
| Lexical baseline | `lexical` | 0.1195 | 0.2179 | 0.2667 | 0.2426 | 0.0533 | 0.0720 |
| Dense + lexical | `dense_lexical_rrf` | 0.1424 | 0.2509 | 0.2962 | 0.2694 | 0.0598 | 0.0832 |
| Multi-source fusion | `full_union_rrf` | 0.1665 | 0.3190 | 0.3869 | 0.3520 | 0.0705 | 0.0952 |
| Proof-state expansion tuned | `weighted_rrf_tuned_recall` | 0.1815 | 0.3560 | 0.4300 | 0.3912 | 0.0869 | 0.1117 |
| Theorem-neighborhood fusion | `weighted_rrf_theorem_tuned` | 0.1966 | 0.3713 | 0.4591 | 0.4177 | 0.0954 | 0.1233 |
| Primary method | `weighted_rrf_llm_theorem_tuned` | **0.2012** | **0.3853** | **0.4810** | **0.4376** | 0.1006 | **0.1285** |
| Auxiliary BGE fusion | `weighted_rrf_llm_pretrained_tuned` | 0.1997 | 0.3780 | 0.4774 | 0.4344 | **0.1012** | **0.1285** |

Gold coverage is constant at `0.9098` for these runs.

The primary method improves covered Recall@100 from `0.2426` to `0.4810` and MAP from `0.0511` to `0.1006` over dense retrieval. The non-LLM theorem-neighborhood fusion also improves strongly over dense retrieval, reaching `0.4591` covered Recall@100 and `0.0954` MAP.

## T2 Theorem -> Theorem Pattern Retrieval

Theorem retrieval is used to test whether theorem-neighborhood profiles carry reusable premise evidence. Strategy coverage is included as an auxiliary guidance diagnostic.

| Method | Neighbor premise Recall@10 | Recall@50 | Recall@100 | MAP | nDCG@10 | Strategy coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline theorem profile | 0.1915 | 0.3012 | 0.3366 | 0.1232 | 0.1653 | 0.9965 |
| LLM-enriched TF-IDF profile | **0.2092** | **0.3480** | **0.3962** | **0.1419** | **0.1859** | 0.9968 |
| LLM-enriched BGE profile | 0.1937 | 0.3163 | 0.3593 | 0.1252 | 0.1662 | **0.9973** |

LLM-enriched TF-IDF profiles give the strongest theorem-neighborhood premise retrieval. The strategy signal should be read as broad coverage from neighbor labels, not as a standalone proof-strategy prediction claim.

## T3 Similar-Theorem Guidance Aggregation

Generated guidance bundles: `25`

These bundles are qualitative artifacts: they are the deterministic first `limit` rows from the T2 neighbor artifact, not a stratified sample and not a full-split aggregate.

| Metric | Value |
| --- | ---: |
| Guidance bundles | 25 |
| Avg source theorem neighbors/bundle | 10.0000 |
| Avg premise suggestions/bundle | 10.0000 |
| Avg strategy facets/bundle | 8.0000 |
| Avg difficulty evidence neighbors/bundle | 20.0000 |

### Aggregated Guidance Evidence

| Evidence type | Top evidence | Aggregate score |
| --- | --- | ---: |
| Premise suggestion | CategoryTheory.Category | 2.2577 |
| Premise suggestion | AffineSubspace.mem_direction_iff_eq_vsub_left | 2.0000 |
| Premise suggestion | AffineSubspace.mem_direction_iff_eq_vsub | 1.6429 |
| Premise suggestion | ringChar.spec | 1.3333 |
| Premise suggestion | Localization.Away | 1.2880 |
| Strategy facet | rewrite_transport | 36.9410 |
| Strategy facet | typeclass_instance_resolution | 22.4533 |
| Strategy facet | theorem_application | 14.9036 |
| Strategy facet | algebraic_computation | 13.8823 |
| Strategy facet | category_morphism_reasoning | 13.8090 |

### Example Bundles

| Theorem | Domain | Neighbors | Premise suggestions | Strategy facets | Difficulty bucket |
| --- | --- | ---: | ---: | ---: | --- |
| AddChar.sum_eq_zero_of_ne_one | NumberTheory | 10 | 10 | 8 | easy |
| AddConstMapClass.map_const_add | Algebra | 10 | 10 | 8 | easy |
| AddMonoidAlgebra.mul_of'_divOf | Algebra | 10 | 10 | 8 | medium |
| AdjoinRoot.algHom_subsingleton | RingTheory | 10 | 10 | 8 | easy |
| Affine.Simplex.sum_pointWeightsWithCircumcenter | Geometry | 10 | 10 | 8 | easy |

## Analysis

The strongest signal is that theorem-neighborhood evidence improves T1 after it is fused with proof-state evidence. The ablation sequence suggests that proof-state expansion, tuned weighted fusion, and theorem-neighborhood premise evidence are associated with the improvement.

LLM theorem enrichment improves theorem-neighborhood retrieval by adding semantic, strategy-oriented, and difficulty-oriented natural-language profile text. This enters the main T1 system as one fused source inside weighted RRF, not as an evaluation label.

BGE pretrained embeddings provide a useful auxiliary channel but are not dominant. The combined LLM+BGE fusion gives the best MAP by a small margin, while the primary LLM theorem-tuned method gives the strongest covered Recall@10, Recall@50, and Recall@100.

Single-source expansion methods are best interpreted as complementary candidate sources rather than standalone final retrievers. Their value is clearest after weighted fusion.

The remaining bottleneck is ranking quality near the top of the list. Covered Recall@100 is much higher than covered Recall@10, so many useful premises enter the candidate pool but are not consistently ranked high enough. A learned reranker over the fused candidate set is the most direct next step.

## Conclusion

The current validation result supports the design hypothesis: theorem-theorem retrieval is not just a side task, but a useful source of premise evidence for Proof-State -> Premise retrieval.

The primary method, `weighted_rrf_llm_theorem_tuned`, nearly doubles covered Recall@100 and MAP over dense retrieval. The strongest claim is premise retrieval improvement; strategy and difficulty evidence are presented as auxiliary guidance signals rather than primary success metrics.

## Appendix: Full T1 Ablation Table

| Method | Covered Recall@10 | Covered Recall@50 | Covered Recall@100 | Overall Recall@100 | MAP | nDCG@10 | Gold coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.1142 | 0.1987 | 0.2426 | 0.2208 | 0.0511 | 0.0698 | 0.9098 |
| lexical | 0.1195 | 0.2179 | 0.2667 | 0.2426 | 0.0533 | 0.0720 | 0.9098 |
| dense_lexical_rrf | 0.1424 | 0.2509 | 0.2962 | 0.2694 | 0.0598 | 0.0832 | 0.9098 |
| symbol_overlap | 0.0510 | 0.0972 | 0.1296 | 0.1179 | 0.0233 | 0.0311 | 0.9098 |
| similar_proof_state_expansion | 0.1268 | 0.2119 | 0.2166 | 0.1971 | 0.0711 | 0.0899 | 0.9098 |
| similar_proof_state_expansion_k50 | 0.1292 | 0.2301 | 0.2712 | 0.2467 | 0.0730 | 0.0911 | 0.9098 |
| similar_proof_state_expansion_k100 | 0.1310 | 0.2365 | 0.2897 | 0.2636 | 0.0743 | 0.0924 | 0.9098 |
| similar_proof_state_expansion_k100_sim | 0.1014 | 0.2366 | 0.2913 | 0.2650 | 0.0545 | 0.0703 | 0.9098 |
| similar_proof_state_expansion_k100_rank_sim | 0.1312 | 0.2365 | 0.2897 | 0.2636 | 0.0742 | 0.0923 | 0.9098 |
| similar_theorem_premises | 0.1451 | 0.2525 | 0.2977 | 0.2708 | 0.0797 | 0.1021 | 0.9098 |
| full_union_rrf | 0.1665 | 0.3190 | 0.3869 | 0.3520 | 0.0705 | 0.0952 | 0.9098 |
| weighted_rrf_balanced | 0.1709 | 0.3273 | 0.3914 | 0.3561 | 0.0742 | 0.0993 | 0.9098 |
| weighted_rrf_lexical_ps | 0.1801 | 0.3368 | 0.4048 | 0.3683 | 0.0781 | 0.1047 | 0.9098 |
| weighted_rrf_ps_heavy | 0.1816 | 0.3410 | 0.4052 | 0.3687 | 0.0846 | 0.1107 | 0.9098 |
| weighted_rrf_ps_heavy_k50 | 0.1819 | 0.3482 | 0.4178 | 0.3801 | 0.0851 | 0.1109 | 0.9098 |
| weighted_rrf_ps_heavy_k100 | 0.1837 | 0.3516 | 0.4267 | 0.3882 | 0.0858 | 0.1119 | 0.9098 |
| weighted_rrf_ps_heavy_k100_sim | 0.1633 | 0.3430 | 0.4323 | 0.3933 | 0.0708 | 0.0952 | 0.9098 |
| weighted_rrf_ps_heavy_k100_rank_sim | 0.1836 | 0.3512 | 0.4265 | 0.3880 | 0.0856 | 0.1117 | 0.9098 |
| weighted_rrf_tuned_frontier | 0.1844 | 0.3531 | 0.4277 | 0.3892 | 0.0865 | 0.1123 | 0.9098 |
| weighted_rrf_tuned_recall | 0.1815 | 0.3560 | 0.4300 | 0.3912 | 0.0869 | 0.1117 | 0.9098 |
| weighted_rrf_theorem_source | 0.1950 | 0.3729 | 0.4558 | 0.4147 | 0.0909 | 0.1184 | 0.9098 |
| weighted_rrf_theorem_heavy | 0.1992 | 0.3739 | 0.4603 | 0.4188 | 0.0919 | 0.1210 | 0.9098 |
| weighted_rrf_theorem_frontier | 0.1935 | 0.3708 | 0.4552 | 0.4142 | 0.0903 | 0.1174 | 0.9098 |
| weighted_rrf_theorem_tuned | 0.1966 | 0.3713 | 0.4591 | 0.4177 | 0.0954 | 0.1233 | 0.9098 |
| similar_theorem_premises_llm_enriched | 0.1564 | 0.2917 | 0.3493 | 0.3178 | 0.0909 | 0.1131 | 0.9098 |
| pretrained_dense | 0.1041 | 0.1895 | 0.2321 | 0.2112 | 0.0447 | 0.0621 | 0.9098 |
| pretrained_similar_proof_state_expansion | 0.1386 | 0.2474 | 0.3062 | 0.2786 | 0.0830 | 0.1013 | 0.9098 |
| weighted_rrf_llm_theorem_source | 0.1967 | 0.3788 | 0.4654 | 0.4234 | 0.0920 | 0.1196 | 0.9098 |
| weighted_rrf_llm_theorem_tuned | 0.2012 | 0.3853 | 0.4810 | 0.4376 | 0.1006 | 0.1285 | 0.9098 |
| weighted_rrf_pretrained_tuned | 0.1974 | 0.3684 | 0.4698 | 0.4274 | 0.0965 | 0.1239 | 0.9098 |
| weighted_rrf_llm_pretrained_tuned | 0.1997 | 0.3780 | 0.4774 | 0.4344 | 0.1012 | 0.1285 | 0.9098 |
