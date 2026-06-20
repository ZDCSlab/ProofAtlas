# LeanRank-Based Knowledge Graph Execution Plan

## 1. Goal

Build a runnable formal proof knowledge graph prototype based on [`erbacher/LeanRank-data`](https://huggingface.co/datasets/erbacher/LeanRank-data). The prototype should sample Hugging Face parquet data, normalize records, construct a theorem-proofstate-premise graph, compute difficulty features, train a premise-ranking baseline, and expose lemma retrieval and theorem similarity queries.

## 2. Recommended Project Structure

```text
leanrank_kg/
  data/
    raw/
    sample/
    processed/
  schemas/
    theorem.schema.json
    proof_state.schema.json
    premise.schema.json
    file_module.schema.json
    proof_technique.schema.json
  src/
    download_or_sample.py
    normalize.py
    parse_context.py
    build_graph.py
    compute_difficulty.py
    weak_label_proof_technique.py
    embed.py
    train_ranker.py
    retrieve.py
    evaluate.py
  outputs/
    graph/
    embeddings/
    models/
    reports/
  notebooks/
    leanrank_kg_demo.ipynb
  README.md
```

## 3. Implementation Phases

The implementation should proceed in two explicit phases.

### 3.1 Phase 1: KG + Embedding Index + Ranker

Phase 1 is the midterm baseline and does not require a graph neural model. The KG is used as a structured data layer for extracting nodes, edges, labels, features, and retrieval supervision.

Phase 1 components:

```text
theorem-proofstate-premise KG
proof-technique weak labeler
difficulty feature computation
proof state and premise embeddings
premise embedding index
cosine retrieval baseline
supervised premise ranker
retrieval and evaluation API
```

Application to test data:

```text
1. Build the training KG using train split only.
2. Fit vectorizers, normalizers, weak labelers, and rankers on train data.
3. Freeze proof-technique candidate pool, labeling rules, LLM prompt if used, thresholds, and ranker parameters.
4. Treat each test proof state as a query node.
5. Encode the test proof state with train-fitted encoders.
6. Retrieve/rank candidate premises from the selected premise index.
7. Reveal test positive premises only for evaluation.
```

Evaluation should use the closed-index setting only:

```text
Closed-index setting: candidate premises = train premises only
```

If a test gold premise does not appear in the train premise index, the example should be marked as `out_of_index_gold` and excluded from Recall@k/MRR numerator-denominator calculations that require retrievability. The report should include gold premise coverage:

```text
gold_premise_coverage = test examples whose gold premise appears in train index / all test examples
```

Phase 1 success criteria:

```text
Recall@k and MRR are computed on test proof states
proof-technique labels are produced using frozen rules or frozen hybrid rules
retrieval examples include scores and explanations
gold premise coverage is reported for the closed-index setting
```

### 3.2 Phase 2: Graph Model

Phase 2 adds a graph representation model over the heterogeneous KG. This is a post-baseline enhancement rather than a prerequisite for applying the KG to test data.

Candidate models:

```text
GraphSAGE
R-GCN
Heterogeneous Graph Transformer
bi-encoder with graph-based reranker
```

Training objectives:

```text
ProofState-Premise link prediction
premise ranking with graph-enhanced embeddings
Theorem-Theorem similarity prediction
ProofState-ProofTechnique prediction
```

Phase 2 should reuse the same theorem-level train/validation/test split from Phase 1. Test proof states remain query nodes, and their gold `positive_uses` edges are used only for final evaluation.

Phase 2 success criteria:

```text
graph model improves Recall@k or MRR over Phase 1 baseline
ablation compares text-only, feature-only, KG baseline, and graph model
model handles unseen test theorem nodes without using test positive edges for training
```

## 4. Load LeanRank-data

Use Hugging Face `datasets` or direct parquet loading.

Start with a sample rather than the full 2M+ rows:

```text
train sample: 50k rows
val sample: 5k rows
```

Sampling strategy:

```text
stratify by file_path
preserve multiple tactic_idx rows per theorem full_name
prioritize rows where all_pos_premises is non-empty
```

### 4.1 Train/Validation/Test Split

Use a theorem-level split, not a row-level split. Multiple rows may correspond to the same theorem because `LeanRank-data` can contain several proof states and several positive premises for the same `full_name`. Splitting by row would leak the same theorem across train and evaluation.

Default split:

```text
train: 80% of theorem full_name groups
validation: 10% of theorem full_name groups
test: 10% of theorem full_name groups
```

Grouping key:

```text
theorem_group_key = full_name
```

Recommended split procedure:

```text
1. Group all sampled rows by full_name.
2. Assign each theorem group to train, validation, or test.
3. Keep all proof states, positive premises, and negative premises for a theorem in the same split.
4. Stratify groups by top-level domain_tag derived from file_path.
5. Use a fixed random seed and write split assignment files.
```

Output files:

```text
data/sample/train_rows.parquet
data/sample/val_rows.parquet
data/sample/test_rows.parquet
data/sample/split_assignments.json
```

Leakage checks:

```text
no full_name appears in more than one split
no proof_state_id appears in more than one split
train/val/test domain distributions are reported
```

For very small samples, use:

```text
train: 70%
validation: 15%
test: 15%
```

The validation split should be used for hyperparameter choices, weak-label rule tuning, and threshold selection. The test split should be used only for final reporting.

Acceptance criteria:

- The loader can read train and validation splits.
- The sampler writes deterministic samples.
- Output fields include `file_path`, `full_name`, `start`, `tactic_idx`, `context`, `all_pos_premises`, `neg_premises`, and `pos_premise`.
- Split assignment is theorem-level and deterministic.
- No theorem `full_name` appears in more than one split.
- Domain distribution statistics are written for train, validation, and test.

## 5. Define Normalized Schemas

Define JSON schemas for the following entities.

### 5.1 Theorem

```json
{
  "id": "thm:Submodule.mem_span_set",
  "full_name": "Submodule.mem_span_set",
  "file_path": "Mathlib/LinearAlgebra/Finsupp/LinearCombination.lean",
  "start": [450, 1],
  "namespace": "Submodule",
  "domain_tag": "LinearAlgebra"
}
```

### 5.2 ProofState

```json
{
  "id": "ps:Submodule.mem_span_set:0",
  "theorem_id": "thm:Submodule.mem_span_set",
  "tactic_idx": 0,
  "context": "R : Type ... ⊢ m ∈ span R s ↔ ...",
  "goal_text": "m ∈ span R s ↔ ...",
  "local_hypotheses": ["R : Type", "M : Type"]
}
```

### 5.3 Premise

```json
{
  "id": "premise:Set.image_id",
  "full_name": "Set.image_id",
  "code": "theorem image_id (s : Set α) : id '' s = s",
  "path": "Mathlib/Data/Set/Image.lean",
  "pid": 52652
}
```

### 5.4 ProofTechnique

```json
{
  "id": "proof_technique:simplification",
  "name": "simplification",
  "source": "weak_rule",
  "description": "Uses simplification lemmas or simp-tagged declarations.",
  "candidate_patterns": ["@[simp]", ".simp", "_simp"],
  "priority": 10
}
```

Acceptance criteria:

- All normalized records validate against schemas.
- ID construction is deterministic.
- Missing premise fields are logged or skipped without crashing the pipeline.

## 6. Normalize LeanRank Records

Implement `normalize.py`.

Input: sampled LeanRank rows.
Output:

```text
theorems.jsonl
proof_states.jsonl
premises.jsonl
positive_edges.jsonl
negative_edges.jsonl
file_modules.jsonl
```

Processing logic:

- Use `full_name` to create theorem nodes.
- Use `(full_name, tactic_idx, context hash)` to create proof state nodes.
- Use `pos_premise` and `all_pos_premises` to create premise nodes.
- Use `neg_premises` to create candidate premise nodes.
- Use `file_path` and premise `path` to create file module nodes.

Acceptance criteria:

- Each theorem is represented by one theorem node.
- Each premise `full_name` is represented by one premise node.
- Every proof state connects to its theorem.
- Positive and negative edges preserve their labels.

## 7. Parse Context

Implement `parse_context.py`.

Extract from Lean context:

```text
local hypotheses
goal text
symbols
namespace hints
typeclass hints
```

Simple rules:

- Text after `⊢` becomes `goal_text`.
- Text before `⊢` is split into local hypotheses.
- Namespace/domain hints are extracted from theorem `full_name` and `file_path`.

Acceptance criteria:

- At least 95% of sampled rows produce a `goal_text`.
- Parsing failures use a fallback and do not interrupt the pipeline.

## 8. Build The Graph

Implement `build_graph.py`.

Node types:

```text
Theorem
ProofState
Premise
FileModule
TacticStep
ProofTechnique
```

Edge types:

```text
Theorem --has_proof_state--> ProofState
Theorem --appears_in_file--> FileModule
ProofState --positive_uses--> Premise
ProofState --negative_candidate--> Premise
Theorem --invokes_premise--> Premise
Premise --defined_in_file--> FileModule
ProofState --at_tactic_step--> TacticStep
ProofState --uses_proof_technique--> ProofTechnique
Premise --co_occurs_with--> Premise
Theorem --similar_to_theorem--> Theorem
```

Output:

```text
nodes.parquet
edges.parquet
graph_stats.json
```

Acceptance criteria:

- All edge endpoints exist.
- Graph stats include counts by node and edge type.
- Graph construction is deterministic.

## 9. Build Theorem Similarity Edges

Similarity formula:

```text
similarity(T1, T2) =
  w1 * shared_premise_jaccard
+ w2 * file_or_namespace_overlap
+ w3 * proof_state_text_similarity
+ w4 * proof_technique_overlap
+ w5 * tactic_index_profile_similarity
```

For the midterm prototype, keep top-k similar theorem edges for each theorem.

Acceptance criteria:

- No self edges.
- Each theorem has at most `k` similarity edges.
- Similarity scores are stored as edge attributes.

## 10. Weakly Label Proof Techniques

Implement `weak_label_proof_technique.py`.

Infer weak proof-technique labels from premise name, premise code, theorem name, and context.

### 10.1 Proof-Technique Candidate Pool

The proof-technique candidate pool should be generated before assigning labels. It should be a controlled vocabulary, not a free-form list produced independently for every proof state.

Initial candidate pool:

```text
simplification
rewriting_or_coercion
typeclass_resolution
definition_unfolding
theorem_application
extensionality
case_or_constructor_reasoning
logical_reasoning
induction
contradiction
computation
automation
```

Generate the pool from three sources:

```text
manual_seed_proof_techniques
rule_mined_proof_techniques_from_names
frequency_filtered_proof_techniques
```

`manual_seed_proof_techniques` are the fixed technique labels above. `rule_mined_proof_techniques_from_names` are created by mapping common Lean declaration names, annotations, and code patterns to proof-technique families. `frequency_filtered_proof_techniques` are optional technique labels that appear often enough in the sampled data.

Name/code-to-technique examples:

```text
@[simp], .simp, _simp -> simplification
Eq, congr, coe, cast, Subtype.coe -> rewriting_or_coercion
ext, extensionality -> extensionality
cases, constructor, rec -> case_or_constructor_reasoning
induction, rec_on, casesOn -> induction
by_contra, contradiction, not_not -> contradiction
norm_num, decide, omega, ring, linarith, nlinarith -> computation or automation
inferInstance, inst, typeclass -> typeclass_resolution
```

Frequency filtering:

```text
minimum_support = 50 proof states in train
maximum_proof_technique_count = 20 labels for the midterm prototype
```

If a candidate label appears below `minimum_support`, map it to a broader parent label such as `theorem_application`, `automation`, or `logical_reasoning`.

The candidate pool should be saved as:

```text
outputs/reports/proof_technique_candidate_pool.json
```

Each proof-technique entry should contain:

```json
{
  "id": "proof_technique:rewriting_or_coercion",
  "name": "rewriting_or_coercion",
  "source": "manual_seed+name_rule",
  "patterns": ["Eq", "congr", "coe", "cast", "Subtype.coe"],
  "minimum_support": 50,
  "parent": "theorem_application"
}
```

### 10.2 Weak Label Assignment

Assign proof-technique labels using deterministic rules. A proof state can receive multiple proof-technique labels.

Initial assignment rules:

```text
[simp] or .simp or theorem tagged simp -> simplification
rw, Eq, congr, coe -> rewriting_or_coercion
ext, extensionality -> extensionality
cases, rec, constructor -> case_or_constructor_reasoning
induction, rec_on, casesOn -> induction
by_contra, contradiction, not_not -> contradiction
norm_num, decide, omega, ring, linarith, nlinarith -> computation_or_automation
```

Rule inputs:

```text
positive premise full_name
positive premise code
negative premise full_name and code, used only for hardness features
proof state context
theorem full_name
file_path
```

Conflict handling:

```text
allow multi-label assignment
store rule provenance for every assigned label
if more than 5 labels fire, keep the top 5 by rule priority and support
if no rule fires, assign no label rather than guessing
```

Rule priority:

```text
explicit Lean annotation or tactic-like name > declaration name pattern > premise code pattern > context keyword
```

Do not tune weak-label rules on the test split. Rule thresholds and priority choices should be finalized using only train and validation data.

Domain-oriented signals such as `LinearAlgebra`, `SetTheory`, `Topology`, or `CategoryTheory` should be stored as `domain_tag` or `subdomain_tag`, not as proof-technique labels.

### 10.3 LLM-Assisted Labeling

An LLM may be used as an optional second-stage annotator for proof states that are not confidently labeled by deterministic rules. The LLM must not invent new labels. It must choose only from the frozen proof-technique candidate pool produced in Section 10.1.

Use LLM-assisted labeling for:

```text
proof states with no rule-based label
proof states with conflicting low-priority labels
proof states selected for human-readable demo examples
small validation samples used to refine rule coverage
```

Do not use the LLM to create labels for the test split after seeing test performance. If LLM labeling is used on test examples, the prompt, candidate pool, and decision rules must already be frozen.

LLM input should include:

```text
proof_state_id
theorem full_name
proof state context
goal_text
positive premise full_name
positive premise code
candidate proof-technique pool
rule-based labels, if any
```

LLM output must be strict JSON:

```json
{
  "proof_state_id": "ps:Submodule.mem_span_set:0",
  "labels": [
    {
      "name": "rewriting_or_coercion",
      "confidence": 0.78,
      "rationale": "The positive premise involves coercion/coercion-normalization patterns."
    }
  ],
  "abstain": false
}
```

Decision policy:

```text
accept LLM label only if confidence >= 0.70
accept at most 3 labels per proof state
reject labels not in frozen candidate pool
prefer deterministic rule labels over LLM labels when confidence is high
store provenance as llm_assisted
```

Recommended hybrid pipeline:

```text
1. Apply deterministic rules to all proof states.
2. Send only unlabeled or ambiguous train/validation examples to the LLM.
3. Optionally use a small human-reviewed sample to calibrate the LLM prompt.
4. Freeze the prompt, candidate pool, thresholds, and conflict policy.
5. Apply the frozen hybrid labeler to train, validation, and test.
```

Outputs:

```text
outputs/reports/proof_technique_llm_labels.jsonl
outputs/reports/proof_technique_label_provenance.csv
outputs/reports/llm_label_abstention_rate.json
```

Acceptance criteria:

- Each proof state may receive zero or more proof-technique labels.
- Each label includes rule provenance.
- Proof-technique distribution is written to a report.
- The proof-technique candidate pool is saved with patterns, source, support, and parent labels.
- Weak-label coverage is reported for train, validation, and test.
- Test labels are generated by frozen rules, not test-set tuning.
- If LLM-assisted labeling is enabled, prompts, candidate pool, thresholds, and conflict policy are frozen before test labeling.
- LLM labels include confidence, rationale, and provenance.
- Labels outside the frozen proof-technique pool are rejected automatically.

## 11. Compute Difficulty Features

Implement `compute_difficulty.py`.

For each proof state compute:

```text
context_length_score
num_local_hypotheses
num_positive_premises
avg_positive_premise_length
premise_namespace_rarity
tactic_step_index_score
negative_candidate_hardness
```

For each theorem aggregate:

```text
mean_proof_state_difficulty
max_proof_state_difficulty
num_proof_states
num_unique_positive_premises
```

Acceptance criteria:

- ProofState and Theorem nodes have difficulty vectors.
- Values are normalized to `[0, 1]`.
- Each node receives an `easy`, `medium`, or `hard` bucket.

## 12. Generate Embeddings

Implement `embed.py`.

Baseline embeddings:

```text
ProofState embedding = text_embedding(full_name + goal_text)
Premise embedding = text_embedding(full_name + code + path)
Theorem embedding = average(proof_state embeddings + positive premise embeddings)
```

Optional features:

```text
structured features
namespace one-hot
proof-technique multi-hot
difficulty vector
degree features
```

Acceptance criteria:

- ProofState, Premise, and Theorem nodes all receive embeddings.
- Embedding rows map back to node IDs through metadata.
- Outputs are saved to `outputs/embeddings/`.

## 13. Train Premise Ranker

Implement `train_ranker.py`.

Training examples:

```text
(proof_state, pos_premise, label=1)
(proof_state, neg_premise, label=0)
```

Baseline models:

```text
cosine similarity baseline
logistic regression on pair features
small MLP reranker
```

Pair features:

```text
cosine(context_embedding, premise_embedding)
same_namespace
same_file_area
premise_frequency
proof_technique_overlap
difficulty_features
```

Acceptance criteria:

- Report Recall@k, MRR, AUC, or accuracy.
- Model artifacts can be saved and reloaded.
- Validation evaluation runs end to end.

## 14. Implement Retrieval API

Implement `retrieve.py`.

Required functions:

```python
retrieve_premises(proof_state_id: str, k: int = 10) -> list[dict]
retrieve_similar_theorems(theorem_id: str, k: int = 10) -> list[dict]
explain_premise_match(proof_state_id: str, premise_id: str) -> dict
get_proof_technique_labels(proof_state_id: str) -> list[dict]
get_difficulty_profile(entity_id: str) -> dict
```

Acceptance criteria:

- Given a proof state, return top-k premises.
- Given a theorem, return similar theorems.
- Outputs include scores and short explanations.

## 15. Evaluate

Implement `evaluate.py`.

Metrics:

```text
Recall@1, Recall@5, Recall@10
MRR
AUC for positive/negative premise ranking
proof-technique label coverage
difficulty distribution
theorem similarity sanity checks
```

Reports:

```text
metrics.json
retrieval_examples.md
proof_technique_distribution.csv
difficulty_distribution.csv
graph_stats.json
```

Acceptance criteria:

- Include at least 20 retrieval examples.
- Each example includes proof state, gold positive premise, top retrieved premises, and scores.
- Reports are suitable for midterm presentation.

## 16. Demo

Create a notebook or CLI demo.

Demo flow:

```text
Load sampled LeanRank data
Build KG
Show graph stats
Select one theorem
Show its proof states
Retrieve top-k premises for one proof state
Show positive premise rank
Show similar theorems
Show weak proof-technique labels
Show difficulty vector
```

Acceptance criteria:

- Demo runs end to end.
- Demo does not depend on external services.
- Demo finishes within five minutes on a small sample.

## 17. Midterm Definition of Done

The midterm implementation is complete when:

- A sampled LeanRank dataset has been processed.
- A theorem-proofstate-premise heterogeneous KG has been constructed.
- Positive and negative premise edges have been generated.
- Weak proof-technique labels have been produced.
- Proof state and theorem difficulty vectors have been computed.
- Baseline embeddings have been generated.
- A premise-ranking baseline has been trained.
- Premise retrieval and theorem similarity queries work.
- Evaluation reports and a demo are available.

## 18. Future Extensions

After the midterm milestone:

- Scale to the full LeanRank-data dataset.
- Train heterogeneous GNNs or graph transformers.
- Add real tactic text if it can be recovered from LeanDojo or mathlib source.
- Merge with ProofNet for natural-language theorem/proof examples.
- Upgrade weak proof-technique labels with human-reviewed annotations.
- Use graph retrieval as a premise-selection module for a Lean prover.
