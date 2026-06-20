# Knowledge Graph Execution Plan

## Goal

Build an executable prototype of a difficulty-aware theorem/proof/strategy knowledge graph. The prototype should ingest a curated seed dataset, normalize it into a typed graph schema, generate graph edges and features, train baseline representations, and expose retrieval functions for similar theorems, proof strategy recommendation, and difficulty estimation.

This document should be treated as the concrete implementation plan.

## Expected Repository Outputs

Create the following project structure:

```text
kg/
  data/
    raw/
    processed/
    seed/
  schemas/
    theorem.schema.json
    proof.schema.json
    lemma.schema.json
    strategy.schema.json
    math_object.schema.json
    concept.schema.json
  src/
    ingest.py
    normalize.py
    extract_objects.py
    annotate_strategies.py
    build_graph.py
    compute_difficulty.py
    embed.py
    train_baseline.py
    retrieve.py
    evaluate.py
  notebooks/
    kg_demo.ipynb
  outputs/
    graph/
    embeddings/
    reports/
  README.md
```

If an existing repository already has a preferred structure, adapt the paths while preserving the same logical modules.

## Data Inputs

The prototype should support at least three input types.

### Theorem Records

Use JSONL as the primary interchange format:

```json
{
  "id": "thm_0001",
  "statement_text": "Let K be a number field defined by ...",
  "domain": "number_theory",
  "subdomain": "monogenicity",
  "source": "curated",
  "status": "proved",
  "objects": ["obj_0001"],
  "concepts": ["monogenicity", "number_field"]
}
```

### Proof Records

```json
{
  "id": "proof_0001",
  "theorem_id": "thm_0001",
  "proof_text": "The proof proceeds by checking local obstructions ...",
  "lemmas": ["lem_0001", "lem_0002"],
  "strategy_labels": ["local_to_global", "case_analysis"],
  "verification_status": "informal_checked",
  "source": "curated"
}
```

### Math Object Records

```json
{
  "id": "obj_0001",
  "object_type": "number_field",
  "canonical_representation": "x^3 - 3x + 1",
  "lmfdb_id": null,
  "known_properties": {
    "degree": 3,
    "monogenicity_status": "unknown"
  }
}
```

## Step 1: Define JSON Schemas

Implement JSON schemas for:

- `Theorem`
- `Proof`
- `Lemma`
- `ProofStrategy`
- `MathObject`
- `Concept`

Each schema should validate required fields, field types, and allowed enum values where applicable.

Acceptance criteria:

- Invalid records fail validation with useful error messages.
- Valid seed records pass validation.
- Schema tests cover at least one valid and one invalid record per node type.

## Step 2: Build Seed Dataset

Create a small but coherent seed dataset under `kg/data/seed/`.

Minimum target size:

- 20 theorem records
- 20 proof records
- 20 lemma records
- 10 proof strategy records
- 20 math object records
- 10 concept records

The seed dataset should focus on number theory and include examples related to monogenicity, primality certification, and modular square root when possible.

Acceptance criteria:

- Every proof references an existing theorem.
- Every referenced lemma exists.
- Every theorem references at least one concept or math object.
- Every strategy label maps to a known `ProofStrategy` node.

## Step 3: Normalize Input Data

Implement `normalize.py`.

Responsibilities:

- Load raw JSONL files.
- Validate records against schemas.
- Canonicalize IDs and labels.
- Normalize strategy names, domains, and subdomains.
- Remove duplicate records.
- Write normalized records to `kg/data/processed/`.

Acceptance criteria:

- Running normalization produces deterministic outputs.
- Duplicate IDs are detected.
- Missing foreign keys are reported.
- The script can be run repeatedly without changing output order.

## Step 4: Extract Objects and Concepts

Implement `extract_objects.py`.

Initial implementation can be rule-based. It should detect and normalize common number-theoretic objects and concepts from theorem and proof text.

Examples:

- Polynomial expressions such as `x^3 - 3x + 1`
- Number field mentions
- Prime mentions
- Discriminant mentions
- Monogenicity mentions
- Quadratic residue mentions
- Modular square root mentions

Acceptance criteria:

- The extractor produces candidate `MathObject` and `Concept` references.
- Extracted references are attached to theorem and proof records.
- Manual annotations take precedence over extracted annotations.

## Step 5: Annotate Proof Strategies

Implement `annotate_strategies.py`.

The initial implementation should support two modes:

- Read manually provided `strategy_labels`.
- Add weak labels using keyword or pattern rules.

Example weak-label rules:

```text
assume for contradiction -> contradiction
induct on -> induction
case -> case_analysis
local obstruction -> local_to_global
certificate -> certificate_verification
random sample / probability -> probabilistic_checking
interactive -> interactive_proof
```

Acceptance criteria:

- Every proof has at least zero or more strategy labels from the strategy vocabulary.
- Weak labels are marked with provenance, such as `manual` or `rule_based`.
- Unknown strategy labels are rejected during validation.

## Step 6: Compute Difficulty Features

Implement `compute_difficulty.py`.

For each theorem and proof, compute an interpretable difficulty vector:

```text
proof_length_score
dependency_depth_score
number_of_invoked_lemmas
number_of_distinct_strategies
strategy_rarity_score
formal_verification_cost_proxy
search_complexity_proxy
```

Initial proxy definitions:

- `proof_length_score`: normalized token count of proof text.
- `dependency_depth_score`: longest path through lemma dependencies, or 1 if unavailable.
- `number_of_invoked_lemmas`: count of referenced lemmas.
- `number_of_distinct_strategies`: count of strategy labels.
- `strategy_rarity_score`: inverse frequency of proof strategies in the seed dataset.
- `formal_verification_cost_proxy`: heuristic based on proof length, number of dependencies, and verification status.
- `search_complexity_proxy`: heuristic based on strategy rarity and theorem similarity sparsity.

Acceptance criteria:

- Every theorem and proof receives a numeric difficulty vector.
- Difficulty values are normalized to a stable range, preferably `[0, 1]`.
- Each record receives a difficulty bucket: `easy`, `medium`, or `hard`.

## Step 7: Build the Heterogeneous Graph

Implement `build_graph.py`.

The graph should include these node types:

- `Theorem`
- `Proof`
- `Lemma`
- `ProofStrategy`
- `MathObject`
- `Concept`

The graph should include these edge types:

- `theorem_has_proof`
- `proof_uses_strategy`
- `proof_invokes_lemma`
- `lemma_supports_theorem`
- `theorem_about_object`
- `object_has_property`
- `theorem_related_to_concept`
- `theorem_similar_to_theorem`
- `strategy_transfers_to`

Preferred storage format for the prototype:

- Nodes: CSV or Parquet table with `node_id`, `node_type`, and attributes.
- Edges: CSV or Parquet table with `src`, `dst`, `edge_type`, and attributes.
- Optional graph object: NetworkX pickle or PyTorch Geometric object if used by the training code.

Acceptance criteria:

- Graph construction is deterministic.
- All edge endpoints exist.
- Node and edge counts are reported.
- Graph artifacts are written to `kg/outputs/graph/`.

## Step 8: Create Similarity Edges

Implement initial rule-based theorem similarity.

Similarity score should combine:

- Shared domain or subdomain
- Shared concepts
- Shared math objects
- Shared proof strategies
- Similar difficulty vector
- Text similarity if embeddings are available

Initial formula:

```text
similarity = w1 * domain_overlap
           + w2 * concept_jaccard
           + w3 * object_jaccard
           + w4 * strategy_jaccard
           + w5 * difficulty_similarity
           + w6 * text_similarity
```

Acceptance criteria:

- Each theorem has up to `k` similarity edges to other theorems.
- Self-edges are excluded.
- Similarity scores are stored as edge attributes.
- The formula weights are configurable.

## Step 9: Generate Baseline Embeddings

Implement `embed.py`.

Generate hybrid node embeddings using:

- Text features from theorem/proof text
- Multi-hot strategy features
- Multi-hot concept features
- Difficulty vectors
- Local graph degree features

Use a simple, reproducible baseline first:

- TF-IDF or local sentence embedding if available
- Concatenated structured features
- Dimensionality reduction with PCA or truncated SVD if useful

Acceptance criteria:

- Every theorem, proof, strategy, object, and concept receives an embedding vector.
- Embedding dimension is configurable.
- Embeddings are saved to `kg/outputs/embeddings/`.
- A metadata file maps node IDs to embedding rows.

## Step 10: Train Baseline Predictive Tasks

Implement `train_baseline.py`.

Train at least two baseline models:

- Theorem-to-strategy prediction
- Difficulty bucket prediction

Optional additional task:

- Link prediction for theorem-proof or theorem-strategy edges

Simple models are acceptable for the midterm prototype:

- Logistic regression
- Random forest
- Shallow MLP
- Cosine nearest-neighbor retrieval

Acceptance criteria:

- The script reports train/dev metrics.
- Results are saved to `kg/outputs/reports/`.
- The model artifacts are saved and reloadable.

## Step 11: Implement Retrieval Functions

Implement `retrieve.py`.

Required functions:

```python
retrieve_similar_theorems(theorem_id: str, k: int = 5) -> list[dict]
recommend_strategies(theorem_id: str, k: int = 5) -> list[dict]
recommend_lemmas(theorem_id: str, k: int = 5) -> list[dict]
estimate_difficulty(theorem_id: str) -> dict
```

Each output should include:

- Returned entity ID
- Score
- Human-readable label or text snippet
- Explanation features when available

Acceptance criteria:

- Functions work from saved graph and embedding artifacts.
- Retrieval results are deterministic for a fixed seed.
- The demo can run on the seed dataset without external services.

## Step 12: Evaluate Graph Quality

Implement `evaluate.py`.

Automatic metrics:

- Top-k precision for strategy recommendation
- Top-k recall for lemma recommendation, if labels exist
- Accuracy or macro-F1 for difficulty bucket prediction
- AUC for link prediction, if implemented
- Retrieval sanity checks based on shared concepts and strategies

Expert validation output:

- Export a CSV or Markdown report of sample retrievals.
- Include theorem, retrieved theorem, score, shared concepts, shared strategies, and difficulty comparison.

Acceptance criteria:

- Evaluation produces a machine-readable metrics file.
- Evaluation produces a human-readable report.
- At least 10 example retrieval cases are included.

## Step 13: Build a Demo

Create `kg/notebooks/kg_demo.ipynb` or an equivalent CLI demo.

The demo should show:

- Loading the graph
- Displaying graph statistics
- Querying a theorem
- Returning similar theorems
- Returning recommended proof strategies
- Returning relevant lemmas
- Showing the estimated difficulty vector

Acceptance criteria:

- The demo runs end to end from processed data.
- The demo does not require manual edits.
- The demo output is clear enough for a midterm presentation.

## Step 14: Add Batch Update Support

Implement a simple update path for new records.

Required behavior:

- Accept new theorem/proof/lemma/object records.
- Validate them.
- Add them to processed data.
- Rebuild graph edges.
- Recompute similarity edges.
- Regenerate embeddings.

Acceptance criteria:

- A small new batch can be added without breaking existing IDs.
- Updated graph statistics reflect the new records.
- Retrieval works after update.

## Step 15: Documentation

Write `kg/README.md`.

The README should include:

- Project purpose
- Data format
- Graph schema
- Setup instructions
- Commands for each pipeline step
- Example retrieval query
- Evaluation command
- Known limitations

Acceptance criteria:

- A new developer can run the pipeline using only the README.
- Every command needed for the midterm demo is documented.

## Recommended Pipeline Commands

The final prototype should support a flow similar to:

```bash
python kg/src/normalize.py --input kg/data/seed --output kg/data/processed
python kg/src/extract_objects.py --input kg/data/processed --output kg/data/processed
python kg/src/annotate_strategies.py --input kg/data/processed --output kg/data/processed
python kg/src/compute_difficulty.py --input kg/data/processed --output kg/data/processed
python kg/src/build_graph.py --input kg/data/processed --output kg/outputs/graph
python kg/src/embed.py --graph kg/outputs/graph --output kg/outputs/embeddings
python kg/src/train_baseline.py --graph kg/outputs/graph --embeddings kg/outputs/embeddings --output kg/outputs/reports
python kg/src/evaluate.py --graph kg/outputs/graph --embeddings kg/outputs/embeddings --output kg/outputs/reports
```

## Midterm Definition of Done

The implementation is considered complete for the midterm milestone when:

- The seed dataset exists and passes schema validation.
- A heterogeneous graph is built from the seed data.
- Difficulty vectors are computed for theorem and proof nodes.
- Similarity edges are generated between theorems.
- Baseline embeddings are generated for graph nodes.
- At least two downstream tasks are evaluated.
- Retrieval functions return similar theorems, strategies, lemmas, and difficulty estimates.
- A demo and README are available.

## Future Extensions

After the midterm milestone, extend the prototype with:

- Heterogeneous GNN training
- Continual learning for graph updates
- Integration with LMFDB-scale data
- Integration with Lean or Isabelle proof dependency graphs
- Integration with Self-Proving Model verifier transcripts
- Active learning for expert annotation
