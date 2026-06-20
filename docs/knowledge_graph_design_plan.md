# Knowledge Graph Design Plan

## Objective

The midterm objective is to construct a difficulty-aware theorem/proof/strategy knowledge graph for a selected mathematical domain. The graph will organize mathematical statements, proofs, lemmas, proof strategies, mathematical objects, and conceptual annotations into a unified representation. It will support theorem retrieval, proof strategy recommendation, difficulty estimation, and downstream prover training.

The knowledge graph is intended to serve as the structural bridge between the curated dataset in RO1 and the prover training pipeline in RO2. It should make mathematical reasoning patterns explicitly available to the learning system, rather than relying only on raw theorem and proof text.

## Scope

The initial graph should focus on a controlled mathematical domain, preferably number theory, with emphasis on:

- Number fields
- Monogenicity
- Primality certification
- Modular square root
- Related LMFDB objects and metadata

This domain is narrow enough to support careful annotation and broad enough to connect to the later Self-Proving Model demonstrations in RO3.

## Graph Schema

The graph should be modeled as a heterogeneous knowledge graph with multiple node and edge types.

### Node Types

- `Theorem`: A mathematical statement, proposition, lemma, or challenge problem.
- `Proof`: A complete proof, proof sketch, certificate, or verification transcript.
- `Lemma`: An intermediate reusable result used inside a proof.
- `ProofStrategy`: A proof method or reasoning pattern.
- `MathObject`: A mathematical object such as a number field, polynomial, ideal, prime, or elliptic curve.
- `Concept`: A higher-level mathematical concept such as monogenicity, local obstruction, quadratic residue, or discriminant.

### Edge Types

- `theorem_has_proof`: Connects a theorem to one or more proofs.
- `proof_uses_strategy`: Connects a proof to the strategies it uses.
- `proof_invokes_lemma`: Connects a proof to lemmas or intermediate claims it depends on.
- `lemma_supports_theorem`: Connects a lemma to the theorem it helps prove.
- `theorem_about_object`: Connects a theorem to the mathematical objects it concerns.
- `object_has_property`: Connects a mathematical object to known properties.
- `theorem_related_to_concept`: Connects a theorem to relevant mathematical concepts.
- `theorem_similar_to_theorem`: Connects mathematically similar theorems.
- `strategy_transfers_to`: Indicates that a proof strategy may transfer across theorem classes or domains.

## Node Attributes

Each node should use a stable schema so that the graph can be queried, trained on, and updated.

### Theorem Attributes

```text
id
statement_text
formal_statement_optional
domain
subdomain
source
status
difficulty_vector
strategy_labels
embedding
```

### Proof Attributes

```text
id
theorem_id
proof_text
formal_proof_optional
proof_length
proof_dag_features
strategy_labels
difficulty_vector
verification_status
embedding
```

### Strategy Attributes

```text
id
strategy_name
description
applicable_domains
example_theorems
embedding
```

### Math Object Attributes

```text
id
object_type
canonical_representation
LMFDB_id_optional
known_properties
certificate_status
embedding
```

## Difficulty Representation

The graph should include a difficulty vector for theorem and proof nodes. The initial version can use interpretable proxy features:

```text
proof_length_score
dependency_depth_score
number_of_invoked_lemmas
number_of_distinct_strategies
strategy_rarity_score
formal_verification_cost_proxy
search_complexity_proxy
```

The difficulty vector can also be discretized into buckets such as `easy`, `medium`, and `hard` for curriculum learning and evaluation.

## Similarity Representation

The graph should represent theorem and proof similarity through both explicit edges and learned embeddings.

Similarity should combine:

- Textual similarity between theorem statements and proof text
- Shared mathematical objects
- Shared concepts
- Shared proof strategies
- Similar difficulty profiles
- Overlapping dependency structure
- Graph-neighborhood similarity

The initial `theorem_similar_to_theorem` edges may be rule-based. Later versions can update these edges using learned graph embeddings.

## Representation Learning

The representation learning pipeline should proceed in two stages.

### Stage 1: Feature-Based Baseline

The first version should generate hybrid embeddings using:

- Text embeddings from theorem and proof text
- One-hot or multi-hot strategy labels
- Difficulty vectors
- Object and concept features
- Local graph-neighborhood statistics

This baseline provides a fast way to validate whether the graph schema and annotations support useful retrieval.

### Stage 2: Graph Neural Representation Learning

The second version should train a graph representation model over the heterogeneous graph. Candidate models include:

- R-GCN
- GraphSAGE
- Heterogeneous Graph Transformer
- Other heterogeneous GNN variants

Potential training objectives include:

- Link prediction
- Proof strategy prediction
- Similar theorem retrieval
- Difficulty prediction
- Lemma recommendation

The midterm demonstration should prioritize:

- Given a theorem, retrieve top-k similar theorems.
- Given a theorem, recommend likely proof strategies.
- Given a theorem, estimate a difficulty profile.

## Query Interface

The knowledge graph should expose a minimal query interface for downstream prover training.

Required queries:

```text
Input: theorem statement
Output: top-k similar theorems

Input: theorem statement
Output: recommended proof strategies

Input: theorem or math object
Output: relevant lemmas and concepts

Input: theorem statement
Output: estimated difficulty vector
```

This interface may initially be implemented as a notebook, command-line script, or lightweight Python API.

## Continual Update

The graph should support updates as new proof attempts, lemmas, certificates, or verifier transcripts are generated.

The midterm version can use batch updates:

- Add new theorem, proof, lemma, and object nodes.
- Add new proof-strategy and proof-dependency edges.
- Recompute similarity edges.
- Recompute or fine-tune node embeddings.
- Update difficulty estimates.

Later versions may use continual learning methods for dynamic graph embeddings.

## Midterm Deliverables

By the midterm milestone, the project should deliver:

- `KG schema v1`
- A curated seed dataset in the selected mathematical domain
- A constructed heterogeneous theorem/proof/strategy knowledge graph
- Initial theorem, proof, strategy, object, and concept embeddings
- A retrieval demo for similar theorems
- A strategy recommendation demo
- A difficulty estimation demo
- An evaluation report with automatic metrics and expert validation

## Success Criteria

By the midterm milestone, we will deliver a difficulty-aware theorem/proof/strategy knowledge graph for a selected mathematical domain, together with trained graph embeddings and retrieval functions that can map a new theorem to similar theorems, likely proof strategies, relevant lemmas, and an estimated difficulty profile.
