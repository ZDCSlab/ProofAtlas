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

## Case Studies

### Case 1: `Action.full_res`

This is a held-out theorem-guidance example showing how the retrieval bundle is meant to be used: inspect candidate premises, compare nearby historical theorems/proof states, read strategy facets, and use the difficulty profile as calibration.

This query is a categorical/action morphism equality. The important proof shape is not a numeric computation; it is an equality between composed morphisms, so useful guidance should point toward category structure, hom/extensionality lemmas, and nearby commutative-diagram style proof states.

| Field | Value |
| --- | --- |
| Split | test |
| Domain | CategoryTheory / Action |
| Goal/query text | X.ρ (f a) ≫ g.hom = g.hom ≫ Y.ρ (f a) X.ρ (f a) ≫ g.hom = g.hom ≫ Y.ρ (f a) |
| Gold premise train coverage | 1.0000 |
| Gold positive premise count | 4 |
| Difficulty | easy / 0.2686 |

**Retrieved premises.**

| Rank | Premise | Score | Why it appears |
| --- | --- | --- | --- |
| 1 | CategoryTheory.Category | 0.8885 | high query-premise embedding similarity (0.803); learned premise ranker score 0.999 |
| 2 | CategoryTheory.MonoidalCategory | 0.8378 | high query-premise embedding similarity (0.808); learned premise ranker score 0.951 |
| 3 | CategoryTheory.rightAdjointMate | 0.8233 | high query-premise embedding similarity (0.793); learned premise ranker score 0.936 |
| 4 | Action.resEquiv | 0.8172 | high query-premise embedding similarity (0.857); learned premise ranker score 0.941 |
| 5 | Monoid.PushoutI.hom_ext | 0.8130 | high query-premise embedding similarity (0.817); learned premise ranker score 0.954 |

How to read this table: these are candidate dependencies a user would inspect first. The score combines embedding similarity and reranker signals; the reason column shows why the system surfaced each premise.

**Historical proof neighbors.**

| Rank | Similar theorem | Score |
| --- | --- | --- |
| 1 | CategoryTheory.comp_rightAdjointMate | 0.9088 |
| 2 | CategoryTheory.comp_leftAdjointMate | 0.9080 |
| 3 | CategoryTheory.MonoOver.isIso_left_iff_subobjectMk_eq | 0.8998 |
| 4 | Mathlib.Tactic.Bicategory.of_normalize_eq | 0.8975 |

| Rank | Similar proof state | Score | Neighbor goal |
| --- | --- | --- | --- |
| 1 | groupCohomology.resolution.d_comp_ε | 0.9286 | ((resolution k G).d 1 0 ≫ ε k G).hom = Action.Hom.hom 0 |
| 2 | groupCohomology.resolution.d_comp_ε | 0.9241 | (ModuleCat.Hom.hom ((resolution k G).d 1 0 ≫ ε k G).hom) x = (ModuleCat.Hom.hom (Action.Hom.hom 0)) x |
| 3 | groupCohomology.resolution.d_comp_ε | 0.9201 | (ModuleCat.Hom.hom ((resolution k G).d 1 0 ≫ ε k G).hom) x = (ModuleCat.Hom.hom (Action.Hom.hom 0)) x |

How to read these neighbors: similar theorems give theorem-level proof templates, while similar proof states show local goal shapes that appeared in historical proofs.

**Strategy facets.**

| Rank | Facet | Confidence | Evidence |
| --- | --- | --- | --- |
| 1 | category_morphism_reasoning | 0.7700 | domain_specific_goal:≫;\bhom\b |
| 2 | case_analysis | 0.7300 | context_case_marker:^case\s+;\bcase\s+ |
| 3 | rewrite_transport | 0.7200 | goal_or_statement_shape:= |
| 4 | typeclass_instance_resolution | 0.7000 | context_typeclass_bindings:\binst[^\s:]*\s*:;\bCategory\b |
| 5 | algebraic_computation | 0.6800 | goal_symbols_and_names:[+*/^] |

Takeaway: the user would first inspect `CategoryTheory.Category, CategoryTheory.MonoidalCategory, CategoryTheory.rightAdjointMate` as candidate dependencies, then compare the query against historical neighbors such as `CategoryTheory.comp_rightAdjointMate, CategoryTheory.comp_leftAdjointMate`. The top strategy facets (`category_morphism_reasoning, case_analysis, rewrite_transport`) summarize the likely proof mode, while the difficulty profile (`easy` / `0.2686`) indicates that this case is expected to be relatively lightweight in the current corpus.

### Case 2: `Affine.Simplex.affineCombination_mem_interior_iff`

This is a held-out theorem-guidance example showing how the retrieval bundle is meant to be used: inspect candidate premises, compare nearby historical theorems/proof states, read strategy facets, and use the difficulty profile as calibration.

This query is an affine-geometry theorem. The useful proof context is about affine combinations and interior membership, so relevant neighbors should involve affine independence, convex/affine coordinates, and geometric membership conditions.

| Field | Value |
| --- | --- |
| Split | test |
| Domain | LinearAlgebra / AffineSpace |
| Goal/query text | (affineCombination k univ s.points) w ∈ s.interior ↔ ∀ (i : Fin (n + 1)), w i ∈ Set.Ioo 0 1 (affineCombination k univ s.points) w ∈ s.interior ↔ ∀ (i : Fin (n + 1)), w i ∈ Set.Ioo 0 1 k : Type u_1 V : Type u_2 P : Type u_3 inst✝⁴ : Ring k inst✝³ : PartialOrder k inst✝² : AddCommGroup V inst✝¹ : Modu |
| Gold premise train coverage | 1.0000 |
| Gold positive premise count | 2 |
| Difficulty | easy / 0.3251 |

**Retrieved premises.**

| Rank | Premise | Score | Why it appears |
| --- | --- | --- | --- |
| 1 | Affine.Simplex.centroid_eq_affineCombination_of_pointsWithCircumcenter | 0.4887 | high query-premise embedding similarity (0.799); learned premise ranker score 0.373 |
| 2 | Affine.Simplex.eq_mongePoint_of_forall_mem_mongePlane | 0.4682 | high query-premise embedding similarity (0.820); learned premise ranker score 0.382 |
| 3 | Affine.Simplex.face_centroid_eq_iff | 0.4660 | high query-premise embedding similarity (0.820); learned premise ranker score 0.378 |
| 4 | Affine.Simplex.reflection_circumcenter_eq_affineCombination_of_pointsWithCircumcenter | 0.4643 | high query-premise embedding similarity (0.812); learned premise ranker score 0.377 |
| 5 | Affine.Simplex.point_eq_affineCombination_of_pointsWithCircumcenter | 0.4642 | high query-premise embedding similarity (0.801); learned premise ranker score 0.380 |

How to read this table: these are candidate dependencies a user would inspect first. The score combines embedding similarity and reranker signals; the reason column shows why the system surfaced each premise.

**Historical proof neighbors.**

| Rank | Similar theorem | Score |
| --- | --- | --- |
| 1 | affineCombination_mem_affineSpan | 0.8675 |
| 2 | sbtw_of_sbtw_of_sbtw_of_mem_affineSpan_pair | 0.8665 |
| 3 | Affine.Simplex.reflection_circumcenter_eq_affineCombination_of_pointsWithCircumcenter | 0.8599 |
| 4 | Affine.Simplex.eq_mongePoint_of_forall_mem_mongePlane | 0.8581 |

| Rank | Similar proof state | Score | Neighbor goal |
| --- | --- | --- | --- |
| 1 | sbtw_of_sbtw_of_sbtw_of_mem_affineSpan_pair | 0.8806 | Sbtw R ((Finset.affineCombination R Finset.univ t.points) (Finset.affineCombinationSingleWeights R i₁))     ((Finset.affineCombination R Fin |
| 2 | mem_affineSpan_iff_eq_weightedVSubOfPoint_vadd | 0.8715 | (Finset.affineCombination k (insert j s) p) w' ∈ affineSpan k (Set.range p) |
| 3 | affineCombination_mem_affineSpan | 0.8703 | (Finset.affineCombination k s p) w -ᵥ p i1 ∈ (affineSpan k (Set.range p)).direction |

How to read these neighbors: similar theorems give theorem-level proof templates, while similar proof states show local goal shapes that appeared in historical proofs.

**Strategy facets.**

| Rank | Facet | Confidence | Evidence |
| --- | --- | --- | --- |
| 1 | rewrite_transport | 0.7600 | goal_or_statement_shape:↔;= |
| 2 | set_membership_reasoning | 0.7400 | goal_symbols_and_names:∈;\bSet\. |
| 3 | algebraic_computation | 0.7200 | goal_symbols_and_names:\bring\b;[+*/^] |
| 4 | typeclass_instance_resolution | 0.7000 | context_typeclass_bindings:\binst[^\s:]*\s*:;\bRing\b |
| 5 | theorem_application | 0.6200 | statement_connectives:→ |

Takeaway: the user would first inspect `Affine.Simplex.centroid_eq_affineCombination_of_pointsWithCircumcenter, Affine.Simplex.eq_mongePoint_of_forall_mem_mongePlane, Affine.Simplex.face_centroid_eq_iff` as candidate dependencies, then compare the query against historical neighbors such as `affineCombination_mem_affineSpan, sbtw_of_sbtw_of_sbtw_of_mem_affineSpan_pair`. The top strategy facets (`rewrite_transport, set_membership_reasoning, algebraic_computation`) summarize the likely proof mode, while the difficulty profile (`easy` / `0.3251`) indicates that this case is expected to be relatively lightweight in the current corpus.

## Interpretation

The dataset and report support a retrieval-centered research claim. Theorem-level premise retrieval is the strongest quantitative result, while proof-state-level premise retrieval remains candidate-generation limited and should be presented as the main open challenge. Proof-state retrieval is still useful as a local-neighbor substrate for strategy-facet retrieval, difficulty-profile retrieval, and explanation. The current theorem-disjoint train/val/test split has no theorem leakage; future split changes should be motivated by domain-balance or retrieval-coverage studies rather than leakage repair.
