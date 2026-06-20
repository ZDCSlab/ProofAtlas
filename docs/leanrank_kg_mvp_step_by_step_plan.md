# LeanRank KG MVP Step-by-Step Execution Plan

## 0. MVP Goal

Build a real, runnable repository for a Lean/mathlib proof knowledge graph based on `erbacher/LeanRank-data`.

This MVP explicitly does not include GNN / graph neural models. The goal is to ship:

1. A reproducible repo.
2. One processed LeanRank sample dataset.
3. A theorem-proofstate-premise knowledge graph.
4. A small set of working retrieval and analysis functions.
5. A homepage/demo page showing the processed dataset, graph statistics, examples, and available functions.

The final project should answer:

- What data did we process?
- What nodes and edges are in the math proof KG?
- Given a proof state, what premises/lemmas can we retrieve?
- Given a theorem, what similar theorems can we find?
- What weak proof-technique labels and difficulty features can we show?

## 1. Final Repository Shape

Create a repo named `leanrank_kg`.

```text
leanrank_kg/
  README.md
  pyproject.toml
  Makefile
  .gitignore
  configs/
    sample.yaml
  data/
    raw/
    sample/
    processed/
      demo/
  schemas/
    theorem.schema.json
    proof_state.schema.json
    premise.schema.json
    file_module.schema.json
    proof_technique.schema.json
  src/
    leanrank_kg/
      __init__.py
      download_or_sample.py
      normalize.py
      parse_context.py
      build_graph.py
      weak_label_proof_technique.py
      compute_difficulty.py
      embed.py
      augment_graph.py
      train_ranker.py
      retrieve.py
      evaluate.py
      report.py
      homepage.py
      cli.py
  notebooks/
    leanrank_kg_demo.ipynb
  homepage/
    index.html
    assets/
      graph_stats.json
      metrics.json
      retrieval_examples.json
      domain_coverage.json
  outputs/
    graph/
    embeddings/
    models/
    reports/
  tests/
    test_parse_context.py
    test_normalize.py
    test_retrieve.py
    test_split_leakage.py
```

Recommended dependencies:

```text
datasets
pyarrow
pandas
numpy
scikit-learn
networkx
jsonschema
pyyaml
typer
rich
jinja2
pytest
```

Optional, only if needed:

```text
duckdb
fastapi
uvicorn
```

## 2. MVP Scope

Keep these components:

- LeanRank sample loader.
- Raw dataset schema inspection.
- Theorem-level train/validation/test split.
- Normalized schemas.
- KG node and edge tables.
- Weak proof-technique labels using deterministic rules.
- Difficulty features.
- TF-IDF embeddings.
- Cosine retrieval baseline.
- Logistic regression premise ranker.
- Similar theorem retrieval using non-GNN similarity signals.
- CLI demo.
- Static homepage.

Do not build these in the MVP:

- GNN, GraphSAGE, R-GCN, HGT, or graph transformer.
- Full 2M-row production pipeline.
- External service dependency for the demo.
- LLM calls or LLM labeling.
- Live Lean prover integration.

## 3. Step 1 - Initialize Repo

Create the project skeleton, Python package, and reproducible commands.

Files to create:

```text
README.md
pyproject.toml
Makefile
configs/sample.yaml
src/leanrank_kg/__init__.py
src/leanrank_kg/cli.py
```

Minimum commands:

```bash
make install
make sample
make process
make evaluate
make homepage
make demo
```

Acceptance criteria:

- `pip install -e .` works.
- `python -m leanrank_kg.cli --help` works.
- `make help` lists the main commands.

## 4. Step 2 - Define Config

Create `configs/sample.yaml`.

Example:

```yaml
dataset_name: erbacher/LeanRank-data
random_seed: 42
sample:
  total_rows: 60000
  small_debug_rows: 1000
  committed_demo_rows: 3000
split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
retrieval:
  top_k: [1, 5, 10]
similarity:
  theorem_top_k: 10
proof_techniques:
  max_labels_per_state: 5
  minimum_support: 50
```

Acceptance criteria:

- All scripts accept `--config configs/sample.yaml`.
- The sample size can be reduced for local smoke tests.
- The config uses one sampling policy: sample `total_rows`, then split theorem groups by ratio.
- Only the small committed demo dataset is intended to live in git; larger generated outputs can be ignored.

## 5. Step 3 - Inspect, Download, And Sample LeanRank Data

Implement `download_or_sample.py`.

Input:

```text
Hugging Face dataset: erbacher/LeanRank-data
```

Output:

```text
outputs/reports/raw_schema.json
outputs/reports/raw_preview.jsonl
data/sample/all_rows.parquet
data/sample/train_rows.parquet
data/sample/val_rows.parquet
data/sample/test_rows.parquet
data/sample/split_assignments.json
outputs/reports/domain_distribution.json
```

Rules:

- First inspect a small raw batch and write the observed parquet schema and example row shapes.
- Implement field adapters for nested premise objects before full sampling.
- Sample rows deterministically.
- Keep only rows with required fields:
  - `file_path`
  - `full_name`
  - `start`
  - `tactic_idx`
  - `context`
  - `all_pos_premises`
  - `pos_premise`
  - `neg_premises`
- Split by theorem `full_name`, not by row.
- Derive `domain_tag` and `subdomain_tag` from `file_path`.
- Preserve all proof states for the same theorem in the same split.
- Write a small committed demo subset to `data/processed/demo/` later in the pipeline.

Acceptance criteria:

- `raw_schema.json` confirms the real shapes of `pos_premise`, `all_pos_premises`, and `neg_premises`.
- No `full_name` appears in more than one split.
- Domain counts are written for train, validation, and test.
- A debug sample can finish in under one minute.

## 6. Step 4 - Define Normalized Schemas

Create JSON schemas for:

```text
Theorem
ProofState
Premise
FileModule
ProofTechnique
```

Required IDs:

```text
thm:{full_name}
ps:{full_name}:{tactic_idx}:{context_hash}
premise:{premise_full_name}
file:{file_path}
proof_technique:{label}
```

Acceptance criteria:

- Every normalized record validates against its schema.
- ID creation is deterministic.
- Invalid rows are logged to `outputs/reports/normalization_errors.jsonl`.

## 7. Step 5 - Parse Lean Proof Context

Implement `parse_context.py`.

Extract:

```text
goal_text
local_hypotheses
symbols
namespace_hints
typeclass_hints
```

Simple parser rules:

- Text after `⊢` is `goal_text`.
- Text before `⊢` is split into local hypotheses.
- If `⊢` is missing, keep the full context as fallback `goal_text`.

Acceptance criteria:

- At least 95% of sampled proof states get non-empty `goal_text`.
- Parser failures do not stop the pipeline.
- Unit tests cover normal context, missing turnstile, and empty context.

## 8. Step 6 - Normalize Data

Implement `normalize.py`.

Input:

```text
data/sample/{train,val,test}_rows.parquet
```

Output:

```text
data/processed/{split}/theorems.parquet
data/processed/{split}/proof_states.parquet
data/processed/{split}/premises.parquet
data/processed/{split}/file_modules.parquet
data/processed/{split}/positive_edges.parquet
data/processed/{split}/negative_edges.parquet
data/processed/demo/*.parquet
```

Normalization logic:

- One theorem node per `full_name`.
- One proof state per `(full_name, tactic_idx, context_hash)`.
- One premise node per premise `full_name`.
- Positive edges from `pos_premise` and `all_pos_premises`.
- Negative edges from `neg_premises`.
- File module nodes from theorem and premise paths.
- Build `data/processed/demo/` from the configured small demo sample so the repo can include one processed dataset without committing large artifacts.

Acceptance criteria:

- Every proof state connects to one theorem.
- Every positive/negative edge endpoint exists.
- Duplicate nodes are deduplicated deterministically.
- The demo processed dataset is small enough for git and homepage generation.

## 9. Step 7 - Build Base KG Tables

Implement `build_graph.py`.

Output:

```text
outputs/graph/{split}/nodes.parquet
outputs/graph/{split}/edges.parquet
outputs/graph/{split}/graph_stats.json
```

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
has_proof_state
appears_in_file
positive_uses
negative_candidate
invokes_premise
defined_in_file
at_tactic_step
co_occurs_with
```

Do not create `uses_proof_technique` or `similar_to_theorem` in this step. Those are enriched graph edges added after weak labels, features, and embeddings exist.

Acceptance criteria:

- `graph_stats.json` reports counts by node type and edge type.
- All edge endpoints exist.
- The graph can be loaded by `networkx` for inspection.

## 10. Step 8 - Weakly Label Proof Techniques

Implement `weak_label_proof_technique.py`.

Use only deterministic rules in the MVP.

Frozen candidate labels:

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

Rule examples:

```text
simp, @[simp], _simp -> simplification
Eq, congr, coe, cast -> rewriting_or_coercion
inferInstance, inst, typeclass -> typeclass_resolution
unfold, defeq -> definition_unfolding
ext, extensionality -> extensionality
cases, constructor, rec -> case_or_constructor_reasoning
induction, rec_on, casesOn -> induction
by_contra, contradiction, not_not -> contradiction
norm_num, decide, omega, ring, linarith, nlinarith -> computation or automation
```

Output:

```text
outputs/reports/proof_technique_candidate_pool.json
outputs/reports/proof_technique_distribution.csv
outputs/reports/proof_technique_label_provenance.csv
data/processed/{split}/proof_state_techniques.parquet
data/processed/{split}/premise_techniques.parquet
```

Acceptance criteria:

- Each label includes provenance.
- A proof state can have zero or more labels.
- Premises also receive weak technique hints from premise name/code so ranker features can compare proof-state and premise signals.
- Domain labels such as `LinearAlgebra` are not stored as proof techniques.

## 11. Step 9 - Compute Difficulty Features

Implement `compute_difficulty.py`.

ProofState features:

```text
context_length_score
num_local_hypotheses
num_positive_premises
avg_positive_premise_length
premise_namespace_rarity
tactic_step_index_score
negative_candidate_hardness
```

For the MVP, `negative_candidate_hardness` should be computed with cheap non-embedding signals first, such as namespace overlap, domain overlap, and string similarity between positive and negative premise names. Embedding-based hardness can be added later after Step 10.

Theorem aggregate features:

```text
mean_proof_state_difficulty
max_proof_state_difficulty
num_proof_states
num_unique_positive_premises
```

Output:

```text
data/processed/{split}/proof_state_features.parquet
data/processed/{split}/theorem_features.parquet
outputs/reports/difficulty_distribution.csv
```

Acceptance criteria:

- Values are normalized to `[0, 1]`.
- Each proof state and theorem gets an `easy`, `medium`, or `hard` bucket.

## 12. Step 10 - Generate Embeddings

Implement `embed.py`.

Use TF-IDF for MVP.

Embeddings:

```text
ProofState = TF-IDF(context + goal_text)
Premise = TF-IDF(full_name + code + path)
Theorem = average(proof state vectors + positive premise vectors)
```

Use one shared TF-IDF vectorizer for proof-state text and premise text so cosine retrieval compares vectors in the same space. Fit the vectorizer on train proof-state texts plus train premise texts only.

Output:

```text
outputs/embeddings/tfidf_vectorizer.joblib
outputs/embeddings/{split}_proof_state_embeddings.npz
outputs/embeddings/{split}_premise_embeddings.npz
outputs/embeddings/{split}_theorem_embeddings.npz
outputs/embeddings/{split}_embedding_metadata.parquet
```

Acceptance criteria:

- Vectorizer is fit on train only.
- Validation and test are transformed using the train-fitted vectorizer.
- Every embedding row maps back to a node ID.

## 13. Step 11 - Add Enriched Graph Edges

Implement `augment_graph.py`.

Input:

```text
outputs/graph/{split}/nodes.parquet
outputs/graph/{split}/edges.parquet
data/processed/{split}/proof_state_techniques.parquet
data/processed/{split}/premise_techniques.parquet
data/processed/{split}/proof_state_features.parquet
data/processed/{split}/theorem_features.parquet
outputs/embeddings/{split}_*.npz
```

Add enriched edge types:

```text
uses_proof_technique
similar_to_theorem
```

Theorem similarity should use non-GNN signals:

```text
shared positive premises
same namespace or file area
proof-state TF-IDF similarity
proof-technique overlap
difficulty-vector similarity
```

Output:

```text
outputs/graph/{split}/nodes_enriched.parquet
outputs/graph/{split}/edges_enriched.parquet
outputs/graph/{split}/graph_stats_enriched.json
outputs/reports/graph_stats_summary.json
```

Acceptance criteria:

- Enriched edges are added only after their required labels/features/embeddings exist.
- No theorem has a self-similarity edge.
- Each theorem has at most configured `theorem_top_k` similarity edges.
- `graph_stats_summary.json` aggregates train, validation, test, and demo graph stats.

## 14. Step 12 - Build Retrieval Functions

Implement `retrieve.py`.

Required functions:

```python
retrieve_premises(proof_state_id: str, k: int = 10) -> list[dict]
retrieve_similar_theorems(theorem_id: str, k: int = 10) -> list[dict]
explain_premise_match(proof_state_id: str, premise_id: str) -> dict
get_proof_technique_labels(proof_state_id: str) -> list[dict]
get_difficulty_profile(entity_id: str) -> dict
get_graph_neighborhood(entity_id: str, depth: int = 1) -> dict
```

MVP retrieval behavior:

- `retrieve_premises` searches over train premises only.
- `retrieve_similar_theorems` uses shared premises, namespace/file overlap, proof-technique overlap, difficulty distance, and TF-IDF similarity.
- `explain_premise_match` returns cosine score, namespace match, file/domain match, and any shared proof-technique signal.

Acceptance criteria:

- Each function works from Python and CLI.
- Returned objects are JSON serializable.
- Results include scores and short explanations.

## 15. Step 13 - Train Baseline Premise Ranker

Implement `train_ranker.py`.

Training pairs:

```text
(proof_state, positive_premise, label=1)
(proof_state, negative_premise, label=0)
```

Model:

```text
LogisticRegression
```

Pair features:

```text
cosine_similarity
same_namespace
same_domain
premise_frequency
proof_technique_overlap
proof_state_difficulty
negative_candidate_hardness
```

Output:

```text
outputs/models/premise_ranker.joblib
outputs/reports/ranker_validation_metrics.json
```

Acceptance criteria:

- Model can be saved and reloaded.
- Validation metrics are reported.
- Cosine-only retrieval is kept as a baseline for comparison.
- `proof_technique_overlap` compares proof-state weak labels with premise weak technique hints generated in Step 8.

## 16. Step 14 - Evaluate

Implement `evaluate.py`.

Use closed-index evaluation:

```text
candidate premises = train premises only
```

Metrics:

```text
Recall@1
Recall@5
Recall@10
MRR
AUC
gold_premise_coverage
proof_technique_label_coverage
difficulty_bucket_distribution
```

Output:

```text
outputs/reports/metrics.json
outputs/reports/retrieval_examples.json
outputs/reports/retrieval_examples.md
outputs/reports/domain_coverage.json
outputs/reports/graph_stats_summary.json
```

Acceptance criteria:

- Include at least 20 retrieval examples.
- Each example shows:
  - proof state
  - gold positive premise
  - top retrieved premises
  - scores
  - whether the gold premise is in the train index
- Out-of-index gold premises are counted in coverage but excluded from retrievable Recall/MRR denominators.

## 17. Step 15 - Build CLI Demo

Implement CLI commands in `cli.py`.

Commands:

```bash
leanrank-kg sample --config configs/sample.yaml
leanrank-kg process --config configs/sample.yaml
leanrank-kg build-graph --config configs/sample.yaml
leanrank-kg embed --config configs/sample.yaml
leanrank-kg augment-graph --config configs/sample.yaml
leanrank-kg train-ranker --config configs/sample.yaml
leanrank-kg evaluate --config configs/sample.yaml
leanrank-kg retrieve-premises --proof-state-id "..."
leanrank-kg similar-theorems --theorem-id "..."
leanrank-kg show-difficulty --entity-id "..."
leanrank-kg show-techniques --proof-state-id "..."
leanrank-kg build-homepage --config configs/sample.yaml
```

Acceptance criteria:

- A user can run the full MVP from the README.
- Demo commands print readable tables.
- The small debug pipeline finishes in under five minutes.

## 18. Step 16 - Build Homepage

Implement `homepage.py` and generate `homepage/index.html`.

The homepage is a static page. It should not require a backend server.

Homepage sections:

1. Project title:
   - `LeanRank Math Proof Knowledge Graph`
2. Dataset card:
   - dataset source
   - sample size
   - train/validation/test theorem counts
   - processed file list
3. KG overview:
   - node counts by type
   - edge counts by type
   - graph construction date
4. Domain coverage:
   - top mathlib domains
   - example theorem names
   - example file paths
5. Function showcase:
   - `retrieve_premises`
   - `retrieve_similar_theorems`
   - `explain_premise_match`
   - `get_proof_technique_labels`
   - `get_difficulty_profile`
   - `get_graph_neighborhood`
6. Retrieval examples:
   - proof state snippet
   - gold premise
   - top retrieved premises
   - scores
7. Proof-technique labels:
   - label distribution
   - example proof states
8. Difficulty:
   - easy/medium/hard distribution
   - example hard proof states
9. Evaluation:
   - Recall@k
   - MRR
   - gold premise coverage
10. Reproducibility:
   - exact commands to rebuild the processed dataset and homepage

Homepage input files:

```text
outputs/reports/metrics.json
outputs/reports/retrieval_examples.json
outputs/reports/domain_coverage.json
outputs/reports/proof_technique_distribution.csv
outputs/reports/difficulty_distribution.csv
outputs/reports/graph_stats_summary.json
```

Homepage output files:

```text
homepage/index.html
homepage/assets/metrics.json
homepage/assets/retrieval_examples.json
homepage/assets/domain_coverage.json
homepage/assets/graph_stats.json
```

Acceptance criteria:

- Opening `homepage/index.html` shows the processed dataset and available functions.
- The page includes real numbers generated by the pipeline, not placeholder text.
- The page includes at least three concrete retrieval examples.
- The page can be published with GitHub Pages.

## 19. Step 17 - Write README

The README should be written for reviewers who want to understand and run the project quickly.

Required sections:

```text
What this repo is
Dataset source
What processed dataset is included
Knowledge graph schema
Available functions
Quickstart
Pipeline commands
Evaluation results
Homepage/demo
Limitations
No-GNN MVP scope
No-LLM MVP scope
Future work
```

Acceptance criteria:

- README includes one command path for debug mode.
- README includes one command path for the main sample.
- README links to `homepage/index.html`.
- README states that GNN models are future work, not part of the MVP.
- README states that the MVP does not require LLM calls.

## 20. Step 18 - Add Tests And Smoke Checks

Minimum tests:

```text
test_parse_context.py
test_normalize.py
test_retrieve.py
test_split_leakage.py
```

Smoke commands:

```bash
make test
make smoke
```

Acceptance criteria:

- Tests pass locally.
- Smoke test builds a tiny dataset, graph, embeddings, evaluation report, and homepage.

## 21. Recommended Build Order

Use this exact implementation order:

```text
1. repo skeleton + config + CLI
2. raw schema inspection + sampler + theorem-level split
3. schemas + context parser
4. normalization + committed demo dataset
5. base KG tables + graph stats
6. proof-technique weak labeler for proof states and premises
7. difficulty features
8. shared TF-IDF embeddings
9. enriched graph edges
10. retrieval functions
11. ranker
12. evaluation reports
13. CLI demo
14. homepage
15. README
16. tests + smoke script
```

This order keeps the project demonstrable early: after step 5, the repo already has a processed dataset and graph stats; after step 10, it has usable KG functions; after step 14, it has the reviewer-facing homepage.

## 22. Final Definition Of Done

The MVP is done when the repo contains:

- A deterministic sampled LeanRank processed dataset.
- A small committed demo processed dataset.
- Raw schema inspection report.
- Train/validation/test split files with theorem-level leakage checks.
- Normalized theorem, proof state, premise, file module, and proof-technique records.
- KG node and edge tables.
- Graph statistics.
- Weak proof-technique labels.
- Difficulty features.
- TF-IDF embeddings.
- Premise retrieval.
- Similar theorem retrieval.
- Premise match explanation.
- Baseline premise ranker.
- Evaluation metrics and at least 20 retrieval examples.
- Static homepage generated from real outputs.
- README with reproducible commands.
- Smoke test proving the pipeline can run end to end.

## 23. Suggested Milestones

Milestone 1 - Processed Dataset:

- Steps 1-6 complete.
- Deliverable: `data/processed/` plus schema validation and split reports.

Milestone 2 - Knowledge Graph:

- Steps 7-11 complete.
- Deliverable: base graph, enriched graph, graph stats, weak labels, difficulty reports.

Milestone 3 - Retrieval Functions:

- Steps 12-15 complete.
- Deliverable: embeddings, retrieval API, ranker, evaluation reports, CLI demo.

Milestone 4 - Public Demo:

- Steps 16-18 complete.
- Deliverable: homepage, README, tests, smoke command.

## 24. One-Sentence Project Description

This repo builds a non-GNN LeanRank-based math proof knowledge graph that processes Lean proof-state data into theorem/proof-state/premise graph tables and exposes premise retrieval, theorem similarity, proof-technique labeling, and difficulty analysis through a reproducible pipeline and static homepage.
