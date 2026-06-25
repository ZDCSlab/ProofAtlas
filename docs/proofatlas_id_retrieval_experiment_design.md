# ProofAtlas ID Retrieval Experiment Design

Date: 2026-06-25

Status: focused experiment plan for the in-distribution LeanRank retrieval setting.

## Core Claim

ProofAtlas studies whether historical train-side LeanRank/mathlib proof data can provide useful proof guidance for theorem-disjoint held-out examples from the same distribution.

The experiment has three tasks:

1. **Main result: Proof-state -> Premise retrieval.**
2. **Pattern retrieval: Theorem -> Theorem retrieval.**
3. **Guidance aggregation: aggregate premise, strategy, and difficulty evidence from similar theorems.**

The split assumption is in-distribution. Train, validation, and test are theorem-disjoint, while file/domain/namespace overlap is allowed and expected.

## Task Overview

| Task | Role | Query | Candidate / evidence pool | Output |
| --- | --- | --- | --- | --- |
| T1 Proof-state -> Premise | Headline result | Held-out proof state | Train premises | Ranked premises |
| T2 Theorem -> Theorem | Supporting pattern retrieval | Held-out theorem profile | Train theorem profiles | Similar historical theorems |
| T3 Similar-theorem guidance aggregation | Downstream utility | T2 neighbors | Neighbor premises, strategy facets, proxy difficulty | Guidance bundle |

This keeps the story tight:

```text
Current proof state retrieves premises directly.
Current theorem retrieves similar historical theorems.
Similar historical theorems provide proof-pattern evidence: premises, strategies, and difficulty profiles.
```

## Dataset And Split

The split should be named:

```text
theorem_disjoint_in_distribution
```

Current sampled data:

| Split | Rows | Theorems | Files | Domains |
| --- | ---: | ---: | ---: | ---: |
| all | 292,012 | 10,000 | 3,294 | 30 |
| train | 234,520 | 8,000 | 3,002 | 30 |
| val | 28,582 | 1,000 | 815 | 26 |
| test | 28,910 | 1,000 | 815 | 26 |

Theorem-disjoint property:

- No theorem-name overlap across train/val/test.
- All proof-state rows for one theorem stay in one split.

ID overlap is expected:

| Diagnostic | Val | Test | Interpretation |
| --- | ---: | ---: | --- |
| Files also present in train | 0.8147 | 0.8012 | Expected under ID split |
| Domains also present in train | 1.0000 | 1.0000 | Expected under ID split |
| Unique gold positive premises in train premise pool | 0.8638 | 0.8658 | Most gold premises are retrievable |
| Proof states with at least one train-side gold premise | 0.9366 | 0.9276 | Most proof-state queries are evaluable |

This split supports an ID retrieval claim. It should not be described as file/domain OOD generalization.

## Shared Query-Time Contract

Allowed for held-out queries:

- held-out theorem `full_name`,
- held-out theorem `file_path`, `domain_tag`, `subdomain_tag`,
- held-out proof-state `context`,
- parsed `goal_text`,
- parsed `local_hypotheses`,
- parsed `symbols`,
- train-side premises,
- train-side proof states,
- train-side theorem profiles,
- train-side graph edges and train-side labels.

Forbidden:

- held-out positive premise IDs in query construction,
- held-out negative candidates in query construction,
- held-out theorem-level `invokes_premise` edges,
- val/test graph edges derived from held-out labels,
- theorem embeddings that average held-out positive premise embeddings,
- any feature depending on held-out proof evidence beyond the current query text/profile.

The current theorem embedding that averages proof-state embeddings with positive premise embeddings is label-aware. Keep it only as an oracle/profile diagnostic, not as a clean T2 representation.

## T1: Proof-State -> Premise Retrieval

### Goal

Given a held-out proof state, rank train-side premises that are useful for the next proof step or local proof obligation.

This is the headline task because it directly matches LeanRank proof-state supervision.

### Query

Use a clean proof-state representation:

```text
full_name
domain_tag subdomain_tag
local_hypotheses
goal_text
symbols
```

The current stored embedding uses only:

```text
full_name + goal_text
```

The first improvement is to include local hypotheses and parsed symbols in both dense and lexical query representations.

### Candidate Corpus

Only train-side premises:

```text
data/processed/train/premises.parquet
```

Premise representation:

```text
full_name
file_path
domain_tag subdomain_tag
code
parsed symbols where available
```

### Gold Labels

Held-out positive premise edges:

```text
data/processed/{val,test}/positive_edges.parquet
```

Report train-gold coverage because not every held-out positive premise exists in the train premise pool.

### Candidate Generation

Candidate generation is the main improvement target.

Current test evidence:

| Method / diagnostic | Metric | Value |
| --- | --- | ---: |
| Dense proof-state to premise | Recall@10 | 0.1162 |
| Dense proof-state to premise | Recall@100 | 0.2362 |
| Lexical candidate diagnostic | Candidate Recall@100 | 0.2494 |
| Dense + lexical union diagnostic | Candidate Recall@100 | 0.3266 |
| Dense + lexical union diagnostic | Hit-query share | 0.4859 |
| Lexical added-gold query share | Share | 0.1296 |

This shows that lexical retrieval already recovers gold premises missed by dense retrieval.

Use these candidate sources:

| Source | Query side | Premise side | Purpose |
| --- | --- | --- | --- |
| Dense | proof-state embedding | premise embedding | Semantic baseline |
| Lexical | local hypotheses + goal + symbols | premise name + code + file path | Lean token/name overlap |
| Symbol overlap | parsed symbols/operators | premise name/code symbols | Cheap structural signal |
| Similar proof-state expansion | nearest train proof states | positive premises of neighbors | Historical proof-pattern evidence |

Initial pool:

```text
union(top100_dense, top100_lexical, top50_symbol, top50_similar_proof_state_expansion)
```

Keep candidate metadata:

- dense rank and score,
- lexical rank and score,
- symbol-overlap score,
- source flags,
- source overlap count,
- same domain,
- same namespace prefix,
- proof-state-neighbor expansion score.

### Reranking

Reranking is secondary. It should run only after candidate generation is measured.

Recommended reranker:

```text
logistic regression or gradient boosted trees
```

Training data:

- all train positive edges,
- LeanRank negative candidates,
- candidate-generated hard negatives,
- no global downsampling to the smallest positive bucket,
- cap negatives per query instead.

Features:

- dense score,
- lexical score,
- symbol overlap,
- same domain,
- namespace prefix match,
- source flags,
- source overlap count,
- premise frequency,
- premise strategy facet labels,
- proof-state local-hypothesis count.

### T1 Metrics

Headline metrics:

- Recall@10,
- Recall@50,
- Recall@100,
- MAP,
- nDCG@10,
- train-gold premise coverage,
- retrievable query count.

Candidate diagnostics:

- candidate Recall@50/100/200,
- candidate hit-query share,
- source-specific recall,
- marginal added hits by source,
- per-domain candidate miss breakdown.

Promising target:

| Layer | Target |
| --- | --- |
| Dense baseline | Preserve or improve current Recall@100 = 0.2362 |
| Dense + lexical | Match or exceed current candidate Recall@100 = 0.3266 |
| Full union | Reach roughly 0.35-0.40 candidate Recall@100 |
| Reranked top-10 | Improve over dense Recall@10 = 0.1162 on full test |

## T2: Theorem -> Theorem Pattern Retrieval

### Goal

Given a held-out theorem profile, retrieve similar train theorems that represent reusable proof patterns.

This is a supporting task, not a replacement for T1. It answers:

```text
Which historical theorems look structurally or semantically similar to this theorem?
```

### Clean Theorem Profile

The current processed theorem table does not contain complete theorem statements. Therefore the clean LeanRank-derived theorem profile should be built from query-time-visible fields:

```text
full_name
file_path
domain_tag subdomain_tag
first proof-state goal
first N proof-state goals
aggregated local symbols from first N proof states
```

Use a small fixed N, for example:

```text
N = 3 or N = 5
```

Do not include:

- positive premise IDs,
- premise embeddings,
- theorem `invokes_premise` edges from val/test,
- proof states beyond the chosen query profile if the experiment defines an early-theorem setting.

For train theorem candidates, build the same clean profile from train proof-state text only.

### Candidate Corpus

Only train theorem profiles:

```text
data/processed/train/theorems.parquet
data/processed/train/proof_states.parquet
```

### Retrieval Sources

Use a small set of interpretable sources:

| Source | Description |
| --- | --- |
| Dense theorem-profile embedding | Embeds clean theorem profile text |
| Lexical theorem-profile retrieval | TF-IDF/BM25 over theorem name, domain, goals, symbols |
| Namespace/domain prior | Valid ID prior for same-distribution theorem families |
| Goal-shape/symbol overlap | Equality/order/membership/existential/operator overlap |

Initial theorem-neighbor pool:

```text
union(top50_dense_theorems, top50_lexical_theorems, top50_symbol_theorems)
```

Final T2 output:

```text
topK similar train theorems, K in {5, 10, 20}
```

### T2 Metrics

Theorem-theorem retrieval does not have a direct "same theorem" label. Evaluate downstream usefulness and sanity:

1. **Neighbor premise coverage**
   - Aggregate train positive premises from topK neighbor theorems.
   - Compare against held-out theorem's positive premises that exist in train.
   - Report Recall@K-neighbor-premises.

2. **Neighbor strategy overlap**
   - Use current strategy facet labels from `weak_label_proof_technique.py`.
   - Aggregate strategy labels from neighbor theorem proof states.
   - Compare with held-out theorem's strategy facet labels.
   - Report label Recall@K and any-label Hit@K.

3. **Neighbor difficulty profile**
   - Use current proxy difficulty features from `compute_difficulty.py`.
   - Aggregate neighbor theorem difficulty scores/buckets.
   - Compare with held-out theorem proxy difficulty.
   - Report bucket accuracy and MAE.

4. **Retrieval sanity diagnostics**
   - same-domain rate,
   - same-subdomain rate,
   - namespace-prefix overlap,
   - qualitative case studies.

The main T2 result should be framed as pattern retrieval utility, not theorem identity classification.

## T3: Similar-Theorem Guidance Aggregation

### Goal

Use T2 theorem neighbors to build an inspectable guidance bundle:

```text
similar theorem evidence
-> premise suggestions
-> strategy facet hints
-> proxy difficulty profile
```

T3 is the bridge from retrieval to proof guidance.

### Inputs

For each held-out theorem:

- topK similar train theorems from T2,
- train proof states belonging to those neighbors,
- train positive premises used by those neighbors,
- current strategy facet labels,
- current proxy difficulty features.

### Premise Aggregation

Aggregate candidate premises from neighbor theorems:

```text
score(premise) =
  sum over neighbor theorems:
    theorem_similarity_score
    * premise_frequency_within_neighbor
    * optional recency/rank discount
```

Recommended simple scoring:

```text
neighbor_weight = 1 / theorem_neighbor_rank
premise_score = sum(neighbor_weight for neighbors that use premise)
```

Report:

- top premise suggestions,
- whether each premise appears in train,
- source theorem evidence,
- neighbor ranks that contributed the premise.

Evaluation:

- theorem-level premise Recall@10/50/100 from neighbor aggregation,
- train-gold coverage,
- per-domain breakdown.

### Strategy Aggregation

Use the existing strategy facet labels.

Current source:

```text
data/processed/{split}/proof_state_techniques.parquet
data/processed/{split}/premise_techniques.parquet
```

No new strategy labeling is required for this experiment.

Aggregation:

```text
strategy_score(label) =
  sum over neighbor proof states:
    theorem_neighbor_weight
    * label_confidence
```

Output:

- top strategy labels,
- label confidence,
- supporting neighbor theorem/proof-state examples,
- provenance pattern where available.

Evaluation:

- strategy label Recall@5,
- any-label Hit@3,
- label distribution by domain.

Language requirement:

- Call this `strategy facet retrieval`.
- Describe the labels as rule-derived strategy facets when methodological detail is needed.

### Difficulty Aggregation

Use the existing proxy difficulty features.

Current source:

```text
data/processed/{split}/proof_state_features.parquet
data/processed/{split}/theorem_features.parquet
```

No new difficulty labels are required for this experiment.

Aggregation:

```text
difficulty_score =
  weighted mean of neighbor theorem_complexity_score

difficulty_bucket =
  bucket of aggregated score or majority bucket among neighbors
```

Output:

- easy/medium/hard bucket,
- numerical difficulty score,
- neighbor difficulty distribution,
- supporting neighbor theorems.

Evaluation:

- proxy bucket accuracy,
- proxy MAE,
- calibration by bucket.

Language requirement:

- Call this `proxy difficulty profile retrieval`.
- Do not call it true proof difficulty prediction.

## Report Structure

The final report should have three result blocks.

### Block 1: Main Result

Title:

```text
Proof-State Premise Retrieval
```

Contents:

- dense baseline,
- lexical baseline,
- dense+lexical union,
- full multi-source candidate generator,
- optional full-test reranker if ready.

Headline table:

| Method | Recall@10 | Recall@100 | MAP | nDCG@10 | Candidate Recall@100 | Train-gold coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

### Block 2: Pattern Retrieval

Title:

```text
Theorem-Theorem Pattern Retrieval
```

Contents:

- clean theorem profile representation,
- topK theorem-neighbor retrieval,
- neighbor premise coverage,
- strategy overlap,
- difficulty profile agreement,
- case studies.

Headline table:

| Method | Neighbor premise Recall@50 | Strategy Recall@5 | Difficulty MAE | Same-domain rate |
| --- | ---: | ---: | ---: | ---: |

### Block 3: Guidance Aggregation

Title:

```text
Aggregated Proof Guidance from Similar Theorems
```

Contents:

- example guidance bundles,
- top premises with source theorem evidence,
- strategy facets,
- proxy difficulty profile,
- explanation text.

This block is primarily qualitative plus supporting quantitative diagnostics.

## What Not To Headline

Do not headline:

- theorem -> premise retrieval from current label-aware theorem embeddings,
- strategy labels as human-verified proof strategies,
- proxy difficulty as external proof difficulty,
- 20-query reranker diagnostics,
- file/domain OOD generalization.

Allowed diagnostic labels:

- `oracle theorem profile`,
- `strategy facet retrieval`,
- `proxy difficulty profile retrieval`,
- `engineering diagnostic`.

## Implementation Order

1. **T1 query representation**
   - Add proof-state embedding text with local hypotheses and symbols.
   - Keep the dense result as the baseline.

2. **T1 lexical and symbol candidate sources**
   - Persist lexical train-premise index.
   - Add symbol-overlap candidate source.
   - Recompute candidate recall on full val/test.

3. **T2 clean theorem profiles**
   - Build theorem profile text from name/domain/first N proof-state goals/symbols.
   - Build train theorem index with the same representation.
   - Evaluate theorem-neighbor downstream utility.

4. **T3 aggregation**
   - Aggregate neighbor premises.
   - Aggregate current strategy facet labels.
   - Aggregate current proxy difficulty features.

5. **Optional reranker**
   - Train only after candidate generation improves.
   - Evaluate on full val/test.

## Target Outcome

A strong, coherent result would be:

| Layer | Target |
| --- | --- |
| T1 dense baseline | Preserve or improve current Recall@100 = 0.2362 |
| T1 dense+lexical | Match or exceed current candidate Recall@100 = 0.3266 |
| T1 full candidate union | Reach roughly 0.35-0.40 candidate Recall@100 |
| T1 reranked top-10 | Improve over dense Recall@10 = 0.1162 on full test |
| T2 theorem-neighbor retrieval | Show useful neighbor premise/strategy/difficulty recovery |
| T3 guidance aggregation | Produce inspectable guidance bundles backed by train-side evidence |

The final story should be:

```text
Under a valid theorem-disjoint ID split, ProofAtlas combines proof-state premise retrieval with theorem-level pattern retrieval. Similar historical theorems provide premise, strategy facet, and proxy difficulty evidence, producing an inspectable proof-guidance package over train-side LeanRank/mathlib proof data.
```
