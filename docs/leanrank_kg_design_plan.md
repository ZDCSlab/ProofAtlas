# LeanRank-Based Knowledge Graph Design Plan

## 1. Objective

This plan starts from [`erbacher/LeanRank-data`](https://huggingface.co/datasets/erbacher/LeanRank-data) and constructs a theorem-proofstate-premise knowledge graph for formal theorem proving.

The goal is to organize Lean/mathlib theorems, proof states, positive premises, negative premises, files/modules, tactic-step metadata, and weak proof-technique labels into a trainable and queryable graph. The resulting knowledge graph will support premise ranking, lemma retrieval, proof dependency analysis, theorem similarity learning, difficulty estimation, and proof-technique abstraction.

## 2. About `erbacher/LeanRank-data`

`erbacher/LeanRank-data` is a Hugging Face dataset derived from Lean/mathlib proof-search data. It is distributed in parquet format and contains approximately 2.09 million rows, with a large training split and a smaller validation split.

The dataset is organized around premise selection for Lean proof states. Each record contains metadata about the current theorem and proof location, a formal proof context, one or more useful premises, and a set of negative candidate premises. The key fields include:

```text
file_path
full_name
start
tactic_idx
context
all_pos_premises
pos_premise
neg_premises
```

These fields make the dataset especially suitable for building a knowledge graph. `full_name` and `file_path` identify the current theorem and module; `context` represents the current proof state; `pos_premise` and `all_pos_premises` identify useful lemmas or declarations; and `neg_premises` provide contrastive candidates for supervised ranking.

For this project, `LeanRank-data` provides three important forms of supervision:

- Proof dependency supervision: which premises are useful for a proof state.
- Retrieval supervision: which premises should be ranked above negative candidates.
- Structural supervision: how theorems, files, proof states, and premises are connected inside mathlib.

### 2.1 Mathematical Domain Coverage

Because `LeanRank-data` is derived from Lean/mathlib, its theorem coverage follows the organization of mathlib rather than a single mathematical topic. The dataset can include theorems, lemmas, definitions, and declarations from broad formalized mathematics areas such as:

```text
Algebra
LinearAlgebra
RingTheory / FieldTheory
GroupTheory
ModuleTheory
SetTheory
Logic / Foundations
OrderTheory
Topology
Analysis
MeasureTheory
Probability
NumberTheory
Combinatorics
CategoryTheory
Data / Basic Structures
```

The dataset does not provide a clean explicit `domain` label. Domain information should therefore be inferred from `file_path`.

Examples:

```text
Mathlib/LinearAlgebra/Finsupp/LinearCombination.lean
-> domain_tag = LinearAlgebra
-> subdomain_tag = Finsupp

Mathlib/SetTheory/ZFC/Basic.lean
-> domain_tag = SetTheory
-> subdomain_tag = ZFC

Mathlib/Algebra/Module/Injective.lean
-> domain_tag = Algebra
-> subdomain_tag = Module

Mathlib/Data/Set/Image.lean
-> domain_tag = Data
-> subdomain_tag = Set
```

The implementation should produce a domain coverage report from sampled or full data:

```text
top-level domain counts
second-level subdomain counts
example theorem names per domain
example file paths per domain
premise usage frequency by domain
```

This report will help characterize the mathematical scope of the KG and guide later sampling, evaluation, and presentation choices.

## 3. Data Framing

`LeanRank-data` is best understood as a formal premise-ranking dataset. Each row approximately represents:

```text
current theorem / proof state
+ one positive premise
+ several negative premises
+ metadata such as file path, theorem name, and tactic-step index
```

The knowledge graph therefore uses formal proof dependency as its organizing principle, with theorem, proof state, and premise retrieval relationships as the central structure.

## 4. Core Node Types

### 4.1 Theorem

Represents a Lean/mathlib theorem, lemma, definition, or declaration.

Primary fields:

```text
full_name
file_path
start
namespace
statement_or_context
domain_tag
embedding
```

### 4.2 ProofState

Represents the proof context and goal associated with a theorem at a tactic step.

Primary fields:

```text
proof_state_id
theorem_full_name
tactic_idx
context
goal_text
local_hypotheses
embedding
```

### 4.3 Premise

Represents a premise that may help prove a proof state. A premise may be a theorem, lemma, definition, simp rule, or other Lean declaration.

Primary fields:

```text
premise_id
full_name
code
path
pid
premise_type
embedding
```

### 4.4 FileModule

Represents a Lean file or mathlib module.

Primary fields:

```text
file_path
module_name
top_level_area
namespace
```

### 4.5 TacticStep

Represents the position of a proof state in a theorem's proof progression. `LeanRank-data` does not directly provide tactic text, but `tactic_idx` can still be used as proof-step metadata.

Primary fields:

```text
tactic_step_id
theorem_full_name
tactic_idx
context_before
positive_premises
negative_premises
```

### 4.6 ProofTechnique

Represents weakly inferred proof techniques derived from premise names, theorem names, premise code, or context patterns.

In this plan, `Strategy` is used strictly in the sense of `ProofTechnique`. Domain-oriented labels such as `linear_algebra_reasoning`, `set_reasoning`, or `topology_reasoning` should not be stored as strategy labels. They should be represented separately as `domain_tag`, `subdomain_tag`, or reasoning-domain metadata derived from `file_path` and namespaces.

Initial weak proof-technique labels may include:

```text
simplification
rewriting_or_coercion
typeclass_resolution
definition_unfolding
theorem_application
case_or_constructor_reasoning
extensionality
induction
contradiction
computation
automation
```

## 5. Core Edge Types

```text
Theorem --has_proof_state--> ProofState
Theorem --appears_in_file--> FileModule
ProofState --positive_uses--> Premise
ProofState --negative_candidate--> Premise
Theorem --invokes_premise--> Premise
Premise --defined_in_file--> FileModule
ProofState --at_tactic_step--> TacticStep
ProofState --uses_proof_technique--> ProofTechnique
Theorem --similar_to_theorem--> Theorem
Premise --co_occurs_with--> Premise
```

## 6. Edge Source And Labeling Requirements

The proposed edges fall into three categories: directly observed from `LeanRank-data`, automatically derived by aggregation, and weakly labeled or defined by project-specific rules.

### 6.1 Directly Observed Edges

These edges can be constructed directly from LeanRank fields and do not require manual labeling:

```text
Theorem --has_proof_state--> ProofState
```

Source:

```text
full_name + tactic_idx + context
```

```text
Theorem --appears_in_file--> FileModule
```

Source:

```text
file_path
```

```text
ProofState --positive_uses--> Premise
```

Source:

```text
pos_premise
all_pos_premises
```

```text
ProofState --negative_candidate--> Premise
```

Source:

```text
neg_premises
```

```text
Premise --defined_in_file--> FileModule
```

Source:

```text
pos_premise.path
neg_premise.path
```

### 6.2 Automatically Derived Edges

These edges do not require manual labels, but the implementation must define deterministic aggregation rules.

```text
Theorem --invokes_premise--> Premise
```

Derived by aggregating:

```text
Theorem --has_proof_state--> ProofState
ProofState --positive_uses--> Premise
```

If any proof state of a theorem uses a premise as a positive premise, then the theorem invokes that premise.

```text
ProofState --at_tactic_step--> TacticStep
```

Derived from:

```text
tactic_idx
```

The `TacticStep` node is a modeling abstraction introduced by the KG pipeline.

```text
Premise --co_occurs_with--> Premise
```

Derived by co-occurrence rules, for example:

```text
two premises appear in the same all_pos_premises list
two premises are used by proof states of the same theorem
```

### 6.3 Weakly Labeled Or Rule-Defined Edges

These edges require project-specific labeling rules or similarity definitions.

```text
ProofState --uses_proof_technique--> ProofTechnique
```

`LeanRank-data` does not provide explicit proof-technique labels. These edges should be created through weak-labeling rules based on premise names, premise code, theorem names, and context patterns.

Examples:

```text
premise name contains simp / @[simp] -> simplification
Eq, congr, coe, cast -> rewriting_or_coercion
ext, extensionality -> extensionality
induction, rec_on -> induction
by_contra, not_not, contradiction -> contradiction
norm_num, decide, omega, ring, linarith -> computation_or_automation
```

```text
Theorem --similar_to_theorem--> Theorem
```

`LeanRank-data` does not provide explicit theorem similarity labels. Similarity edges should be generated by a defined similarity function using signals such as:

```text
shared positive premises
same file_path or namespace
proof state context embedding similarity
shared weak proof-technique labels
similar difficulty vectors
```

The two main project-defined labeling components are therefore:

```text
proof-technique weak labels
theorem similarity function
```

## 7. Mapping LeanRank Fields To KG Components

| LeanRank Field | KG Usage |
|---|---|
| `file_path` | `FileModule` node and `Theorem appears_in_file` edge |
| `full_name` | Current `Theorem` node ID |
| `start` | Source-location metadata for the theorem |
| `tactic_idx` | `TacticStep` node or proof-progression metadata |
| `context` | `ProofState` text and embedding input |
| `all_pos_premises` | `ProofState positive_uses Premise` edges |
| `pos_premise` | Positive example for premise ranking |
| `neg_premises` | Negative examples for premise ranking |
| `pos_premise.code` | `Premise` node text and embedding input |
| `pos_premise.full_name` | `Premise` node ID |
| `pos_premise.path` | `Premise defined_in_file FileModule` edge |

## 8. Main Learning Tasks

### 8.1 Task 1: Proof-Technique Abstraction

Weak proof-technique labels can be inferred from formal premise usage. This task defines the proof-technique vocabulary and assigns interpretable labels to proof states.

Examples:

```text
[simp] theorem / simp namespace -> simplification
Nat / Int arithmetic lemmas, norm_num, ring -> computation_or_automation
coe / subtype / ext lemmas -> coercion_or_extensionality
induction / rec_on lemmas -> induction
by_contra / contradiction lemmas -> contradiction
```

These labels are weak labels rather than gold annotations, but they are useful for interpretable midterm demonstrations and downstream retrieval.

### 8.2 Task 2: Proof-Technique Retrieval

Given a proof state, retrieve or rank likely proof techniques from the frozen proof-technique candidate pool.

Input:

```text
ProofState context
goal_text
positive premise names and code, for training only
candidate proof-technique pool
```

Output:

```text
top-k proof techniques
confidence scores
rule or model explanation
```

This task can be implemented as rule-based weak labeling, LLM-assisted labeling, or a supervised multi-label classifier trained from weak labels. The label space remains closed: the system must choose only from the frozen `ProofTechnique` pool.

### 8.3 Task 3: Premise Ranking

Given a proof state, rank candidate premises by relevance.

Input:

```text
ProofState context
Candidate Premise code/name/path
Graph neighborhood features
```

Output:

```text
score(ProofState, Premise)
```

This is the most direct supervised task supported by `LeanRank-data`.

### 8.4 Task 4: Lemma Retrieval

Given a theorem or proof state, retrieve top-k relevant lemmas/premises.

Output:

```text
top-k positive premises
relevance scores
explanation features
```

### 8.5 Task 5: Theorem Similarity

Two theorems are considered similar if they use similar premises, appear in nearby files/modules, have similar proof states, or share proof-technique labels.

Similarity signals:

```text
shared positive premises
shared premise namespaces
context embedding similarity
file/module proximity
proof-technique label overlap
```

## 9. Difficulty Representation

In the LeanRank setting, difficulty primarily measures proof search complexity: how difficult a proof state is for premise retrieval and proof continuation.

Suggested difficulty vector:

```text
context_length_score
num_local_hypotheses
num_positive_premises
avg_positive_premise_length
premise_namespace_rarity
tactic_step_index_score
negative_candidate_hardness
retrieval_entropy
```

Interpretation:

- `context_length_score`: longer proof states are usually harder to process.
- `num_local_hypotheses`: larger local contexts make retrieval more difficult.
- `num_positive_premises`: proof states requiring multiple useful premises may be harder.
- `premise_namespace_rarity`: rare premise namespaces may indicate harder retrieval.
- `negative_candidate_hardness`: ranking is harder when negative candidates are close to positives.
- `retrieval_entropy`: model uncertainty over candidate premises can be used as a difficulty signal.

## 10. Representation Learning

### 10.1 Baseline Representation

First construct hybrid embeddings:

```text
ProofState embedding = text_embedding(context) + metadata features
Premise embedding = text_embedding(code + full_name + path)
Theorem embedding = aggregate(proof state embeddings + positive premise embeddings)
```

### 10.2 Graph Representation

Later train graph-based models:

```text
R-GCN
GraphSAGE
HGT
bi-encoder + graph reranker
```

For the midterm prototype, prioritize:

```text
TF-IDF / sentence embedding baseline
+ graph features
+ nearest-neighbor retrieval
+ supervised premise-ranking classifier
```

## 11. Midterm Deliverables

The midterm deliverables should include:

- LeanRank subset loader
- Normalized theorem/proofstate/premise schema
- Heterogeneous KG construction pipeline
- Positive and negative premise edges
- Baseline embeddings
- Premise-ranking baseline
- Proof-technique retrieval demo
- Lemma retrieval demo
- Theorem similarity demo
- Weak proof-technique labeling demo
- Difficulty feature report

## 12. Success Criteria

By the midterm milestone, the system should support:

```text
Input: a Lean proof state context
Output: top-k relevant premises

Input: a Lean proof state context
Output: top-k proof techniques from the frozen ProofTechnique pool

Input: a theorem full_name
Output: similar theorems based on shared premises/proof states

Input: a proof state
Output: difficulty vector and proof-search complexity bucket
```

In one sentence: this KG is built around Lean formal proof search. It uses proof states, positive premises, and negative premises from `LeanRank-data` to construct a theorem-premise dependency graph and train representation models for premise ranking and lemma retrieval.
