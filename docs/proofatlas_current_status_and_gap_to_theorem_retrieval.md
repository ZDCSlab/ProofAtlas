# ProofAtlas Current Status And Gap To Full KG Theorem Retrieval

Date: 2026-06-20

## Purpose

This note records the current ProofAtlas project state and what is still missing before it becomes a complete knowledge graph system that can take a new theorem, such as a theorem from the test set or a user-provided Lean theorem, and retrieve relevant knowledge.

The target end state is:

```text
Given a new Lean theorem or proof state
-> parse/encode it
-> retrieve relevant premises, similar theorems, proof techniques, graph neighbors, and difficulty signals
-> return ranked, explainable knowledge from the ProofAtlas KG
```

## Current Project State

ProofAtlas is currently a strong MVP built from real LeanRank data.

### Data

Current full ProofAtlas config:

```text
config: configs/proofatlas.yaml
dataset: erbacher/LeanRank-data
use_huggingface: true
raw LeanRank rows sampled: 60,000
current improved sampling plan: theorem-first sampling
target theorem budget in configs/proofatlas.yaml: 10,000 theorems
Hugging Face candidate rows for theorem sampling: 350,000
split: train / val / test
demo split: lightweight homepage/demo subset
```

Because the KG is organized around theorem retrieval, the primary scale should be reported in theorem units. Proof states and premises are secondary evidence attached to those theorem nodes.

Current theorem-level processed scale:

```text
train: 1667 theorems
val:    208 theorems
test:   209 theorems
demo:   285 theorems
total train/val/test: 2084 theorem-disjoint theorem nodes
```

Supporting scale under those theorem nodes:

```text
train: 4722 proof states, 53668 premises, avg 2.83 proof states/theorem, max 59
val:    583 proof states, 10824 premises, avg 2.80 proof states/theorem, max 22
test:   696 proof states, 12443 premises, avg 3.33 proof states/theorem, max 54
demo:   900 proof states,  1439 premises, avg 3.16 proof states/theorem, max 54
```

The train/val/test split is theorem-level, so the same theorem should not appear across train, validation, and test. This is important because the target retrieval task is theorem-centric: given a held-out theorem or a new theorem, retrieve useful knowledge from the train-side KG.

Important sampling correction: the original 60k run effectively started from a row-level Hugging Face prefix and only split by theorem afterward. That is not ideal for theorem retrieval, because it can under-cover the theorem space and lower train-index premise coverage for held-out theorem evaluation. The sampler should start from theorem units: gather a larger candidate pool, select theorem IDs, keep all candidate rows for selected theorems, then split by theorem.

### Graph Schema Implemented

Current node types:

```text
Theorem
ProofState
Premise
FileModule
TacticStep
ProofTechnique
```

Current edge types:

```text
Theorem -> ProofState: has_proof_state
Theorem -> FileModule: appears_in_file
ProofState -> Premise: positive_uses
ProofState -> Premise: negative_candidate
ProofState -> Premise: invokes_premise
Premise -> FileModule: defined_in_file
ProofState -> TacticStep: at_tactic_step
ProofState -> ProofTechnique: uses_proof_technique
Theorem -> Theorem: similar_to_theorem
Premise -> Premise: co_occurs_with
```

Important caveat: `has_proof_state` is correctly one theorem to many proof states. However, the current `invokes_premise` edge is implemented as `ProofState -> Premise`, while the design plan intended `Theorem -> Premise` aggregated over proof states. Also, `TacticStep` nodes are currently keyed only by tactic index, not by theorem plus tactic index.

### Embeddings And Retrieval

Current embedding config:

```text
backend: sentence_transformers
model: BAAI/bge-base-en-v1.5
device: cuda
batch size: 512
```

Current embedding artifacts exist for:

```text
proof states
premises
theorems
```

Current retrieval APIs:

```text
retrieve_premises(proof_state_id, k, split, index_split)
retrieve_similar_theorems(theorem_id, k, split)
explain_premise_match(proof_state_id, premise_id, split, index_split)
get_proof_technique_labels(proof_state_id, split)
get_difficulty_profile(entity_id, split)
get_graph_neighborhood(entity_id, depth, split)
```

These APIs work well for entities already present in processed tables. For example, given a `proof_state_id` from `val` or `test`, the system can retrieve premises from the train index.

### Demo And Homepage

The homepage is published from `homepage/` and currently emphasizes:

```text
real 60k LeanRank data
BGE embeddings on CUDA
KG scale, currently displayed with edge/node charts but should be narrated around theorem counts
domain coverage charts
retrieval examples
proof-technique labels
validation status
```

Current public page:

```text
https://zdcslab.github.io/ProofAtlas/
```

## What Works Today For Test-Set Retrieval

The current system already supports this narrower workflow:

```text
Given a proof_state_id from val/test
-> load that proof state's embedding
-> compare against train premise embeddings
-> return top-k train premises
-> provide score and basic explanation
```

This is the most important currently working retrieval path.

Example conceptual call:

```bash
leanrank-kg retrieve-premises \
  --proof-state-id "<test_or_val_proof_state_id>" \
  --split test \
  --index-split train \
  --k 10
```

The current evaluation also uses val/test positive edges to compute retrieval metrics and examples.

## Main Gap: New Theorem Retrieval Is Not Yet End-To-End

The project does not yet fully support this broader workflow:

```text
Given a new theorem statement
-> parse it into theorem/proof-state-like query objects
-> generate query embeddings
-> retrieve relevant premises/theorems/techniques/graph evidence
-> return a unified answer
```

Today, retrieval expects an existing `proof_state_id` or `theorem_id` already present in processed tables. It does not yet accept arbitrary theorem text, Lean source code, or a theorem full name that is not already normalized into the KG.

## Missing Steps Toward A Complete Knowledge Graph Retrieval System

### 1. Fix KG Semantics To Match The Design Plan

Needed:

```text
Theorem -> Premise: invokes_premise
```

Current implementation:

```text
ProofState -> Premise: invokes_premise
```

Recommended fix:

```text
For each positive_uses edge:
  proof_state_id -> theorem_id
  add theorem_id -> premise_id as invokes_premise
```

Also needed:

```text
TacticStep node id should be theorem-scoped or proof-state-scoped
```

Current:

```text
tactic:0
tactic:1
...
```

Better:

```text
tactic:<theorem_id>:<tactic_idx>
```

or:

```text
tactic:<proof_state_id>
```

Why this matters: without this fix, all theorems at the same tactic index share the same tactic-step node, which weakens graph semantics.

### 2. Add A Query Object For New Theorems

Current retrieval operates on existing rows. A complete system needs a query layer:

```text
NewTheoremQuery
NewProofStateQuery
```

These should support input forms such as:

```text
Lean theorem statement
theorem full_name if it exists in mathlib
goal text
local hypotheses + goal
raw Lean proof state
```

Minimum useful schema:

```text
query_id
input_type
full_name optional
file_path optional
context optional
goal_text
local_hypotheses
domain_hint optional
```

### 3. Add Out-Of-KG Embedding Support

Current embeddings are precomputed for existing processed proof states, premises, and theorems.

For a new theorem, the system needs to encode text at query time:

```text
encode_new_proof_state_query(text)
encode_new_theorem_query(text)
```

This should reuse the same BGE model and prefixes already recorded in:

```text
outputs/embeddings/embedding_config.json
```

The key requirement is embedding compatibility:

```text
new query embedding must live in the same vector space as train premise embeddings
```

### 4. Build A Persistent Retrieval Index

Current retrieval loads parquet and `.npz` embeddings directly, then computes cosine similarity.

That is acceptable for current MVP theorem scale, but a complete system should build a reusable index:

```text
premise index over train premises
theorem index over train theorem embeddings
optional proof-state index
```

Candidate implementations:

```text
FAISS
ScaNN
Annoy
sklearn NearestNeighbors for small local use
```

Expected artifacts:

```text
outputs/indexes/train_premise.faiss
outputs/indexes/train_theorem.faiss
outputs/indexes/index_metadata.parquet
```

### 5. Add Theorem-Level Knowledge Retrieval

For a new theorem, relevant knowledge is broader than just premises. A complete result should include:

```text
top premises
similar existing theorems
related proof states
likely proof techniques
difficulty estimate
nearby graph neighborhood
domain/subdomain hints
explanations
```

Recommended API:

```python
retrieve_knowledge_for_theorem(
    theorem_text: str,
    full_name: str | None = None,
    k_premises: int = 20,
    k_theorems: int = 10,
) -> dict
```

Expected output:

```text
query summary
ranked premises
ranked similar theorems
predicted proof techniques
difficulty profile
graph evidence
explanations
```

### 6. Add Test-Set Theorem Retrieval Evaluation

Current evaluation is mostly proof-state to premise retrieval.

For theorem-level retrieval, evaluate:

```text
Given a held-out test theorem
-> aggregate its proof states or theorem text into a query
-> retrieve train premises/theorems
-> compare against known positive premises from its proof states
```

Metrics:

```text
number of held-out test theorems evaluated
Recall@k over all positive premises for the test theorem
MRR
MAP
nDCG
gold premise coverage in train index
similar theorem qualitative examples
```

Reporting rule: the headline data scale for this evaluation should be theorem count first, then average proof states per theorem and premise coverage as supporting evidence.

Important evaluation detail: because train/val/test are theorem-disjoint, many exact positive premises from a test theorem may or may not exist in the train premise index. Metrics must separately report:

```text
gold premise exists in train index
gold premise missing from train index
```

### 7. Improve Graph-Aware Ranking

Current ranking is mostly embedding/cosine plus simple ranker features.

A fuller KG retriever should combine:

```text
BGE similarity
shared namespace / domain
file/module proximity
premise frequency
proof-technique compatibility
difficulty compatibility
graph proximity
co-occurrence
theorem similarity
```

This can be done first with feature-weighted reranking before adding a GNN.

Recommended next MVP step:

```text
retrieve top 100 by BGE
rerank with graph features
return top 20 with explanations
```

### 8. Add Better Lean Parsing Or Lean Server Integration

Current parsing is heuristic over LeanRank `context`.

For arbitrary new theorem input, a robust system needs one of:

```text
Lean server / lake environment integration
tree-sitter Lean parser
structured parser for theorem statement and proof state text
```

This is necessary if the system should accept actual Lean code and not only text copied from LeanRank contexts.

### 9. Add A User-Facing Demo Input

Current homepage is static and shows precomputed examples. A complete demo should include an interactive or scripted path:

```text
paste theorem / proof state
click retrieve
show relevant premises and KG evidence
```

Possible implementation options:

```text
static page with precomputed test theorem examples
Streamlit demo
FastAPI endpoint + static frontend
Jupyter notebook walkthrough
CLI command for theorem retrieval
```

For near-term delivery, the easiest credible demo is:

```text
precompute 10 test theorem retrieval case studies
publish them on homepage as interactive-looking cards
include theorem query, retrieved premises, similar theorems, proof techniques, and explanation signals
```

## Recommended Roadmap

### Phase 1: Correct KG Semantics

Tasks:

```text
use theorem-first sampling for real runs
fix invokes_premise to Theorem -> Premise
make TacticStep nodes theorem/proof-state scoped
update tests
rerun graph build / augment / validate / homepage
```

Outcome:

```text
KG edges match the design plan more closely.
```

### Phase 2: Support Out-Of-KG Query Embedding

Tasks:

```text
add query encoder using BGE model
add retrieve_premises_for_text(query_text)
add CLI command retrieve-premises-for-query
test against held-out val/test proof states
```

Outcome:

```text
The system can retrieve relevant premises for text not already assigned a proof_state_id.
```

### Phase 3: Theorem-Level Retrieval

Tasks:

```text
aggregate theorem query from statement and optional proof context
retrieve premises
retrieve similar theorems
predict proof techniques
estimate difficulty
return unified JSON result
```

Outcome:

```text
Given a test theorem, return a useful knowledge bundle.
```

### Phase 4: Precomputed Test Theorem Case Studies

Tasks:

```text
select representative test theorems
generate retrieval result cards
include gold premise coverage and top retrieved premises
add similar theorem and graph evidence
publish on homepage
```

Outcome:

```text
Reviewers can see theorem-level retrieval, not just proof-state-level retrieval.
```

### Phase 5: Interactive Service

Tasks:

```text
build FastAPI or Streamlit endpoint
load BGE model/index once at startup
accept theorem/proof-state query text
return ranked knowledge
```

Outcome:

```text
ProofAtlas becomes a usable retrieval tool rather than only a static demo.
```

## Near-Term Priority

The most valuable next step is not training a GNN. It is:

```text
1. Fix graph edge semantics.
2. Add out-of-KG text query embedding.
3. Add theorem-level retrieval over test theorems.
4. Publish 5-10 theorem retrieval case studies on the homepage.
```

This would make the project visibly closer to the target: a knowledge graph that helps retrieve relevant mathematical knowledge for a new theorem.
