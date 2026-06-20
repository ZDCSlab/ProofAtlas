# ProofAtlas

ProofAtlas builds a non-GNN LeanRank-style math proof knowledge graph. It processes theorem proof-state rows into theorem, proof-state, premise, file, proof-technique, and feature tables, then exposes premise retrieval, similar theorem retrieval, weak proof-technique labeling, difficulty analysis, evaluation reports, and a static homepage.

## Dataset Source

The target source is `erbacher/LeanRank-data`. The default config uses a deterministic synthetic LeanRank-shaped sample so the MVP can run quickly and reproducibly without relying on a large external download. Set `use_huggingface: true` in `configs/sample.yaml` to attempt loading the real Hugging Face dataset.

## What Processed Dataset Is Included

The pipeline writes sampled rows to `data/sample/`, normalized split tables to `data/processed/{train,val,test}/`, and a small reviewer demo dataset to `data/processed/demo/`.

## Knowledge Graph Schema

Node types: `Theorem`, `ProofState`, `Premise`, `FileModule`, `TacticStep`, and `ProofTechnique`.

Base edge types: `has_proof_state`, `appears_in_file`, `positive_uses`, `negative_candidate`, `invokes_premise`, `defined_in_file`, and `at_tactic_step`.

Enriched edge types: `uses_proof_technique` and `similar_to_theorem`.

JSON schemas live in `schemas/`.

## Available Functions

The retrieval API in `src/leanrank_kg/retrieve.py` provides:

- `retrieve_premises`
- `retrieve_similar_theorems`
- `explain_premise_match`
- `get_proof_technique_labels`
- `get_difficulty_profile`
- `get_graph_neighborhood`

## Quickstart

Use the conda environment created for this project:

```bash
conda activate leanrank_kg
make install
make smoke
```

For a larger local demo using `configs/sample.yaml`:

```bash
conda activate leanrank_kg
make install
make demo
```

To run against a real LeanRank sample instead of the synthetic fallback:

```bash
conda activate leanrank_kg
make demo CONFIG=configs/leanrank_real_sample.yaml
```

For the full ProofAtlas run using the original 60,000-row plan, real LeanRank data, GPU-backed Hugging Face embeddings, and the ProofAtlas homepage:

```bash
conda activate leanrank_kg
pip install -e ".[hf]"
leanrank-kg full-pipeline --config configs/proofatlas.yaml
```

TF-IDF is the default embedding backend because it is fast and reproducible. To experiment with Hugging Face sentence embeddings:

```bash
conda activate leanrank_kg
pip install -e ".[hf]"
leanrank-kg full-pipeline --config configs/leanrank_hf_embeddings.yaml --debug-rows 120
```

The optional configs use `BAAI/bge-base-en-v1.5` and write the same embedding filenames as the TF-IDF backend, so retrieval and graph augmentation continue to work.

## Pipeline Commands

```bash
make sample
make process
make build-graph
make label
make difficulty
make embed
make augment-graph
make train-ranker
make evaluate
make validate
make report
make homepage
make audit
```

The CLI also supports direct commands such as:

```bash
leanrank-kg sample --config configs/sample.yaml
leanrank-kg full-pipeline --config configs/sample.yaml --debug-rows 120
leanrank-kg retrieve-premises --proof-state-id "<id>"
leanrank-kg similar-theorems --theorem-id "<id>"
leanrank-kg explain-premise-match --proof-state-id "<id>" --premise-id "<id>"
leanrank-kg build-report --config configs/sample.yaml
leanrank-kg audit --config configs/sample.yaml
```

## Evaluation Results

After `make demo` or `make smoke`, metrics are written to `outputs/reports/metrics.json`, retrieval examples to `outputs/reports/retrieval_examples.json`, and graph summaries to `outputs/reports/graph_stats_summary.json`.

Validation reports are written to `outputs/reports/schema_validation_summary.json`, `outputs/reports/split_leakage_report.json`, `outputs/reports/context_parse_coverage.json`, and `outputs/reports/graph_validation_summary.json`.

The MVP completion audit is written to `outputs/reports/mvp_completion_audit.json`.

## Homepage/Demo

The static homepage is generated at `homepage/index.html`. It uses real pipeline outputs copied into `homepage/assets/` and can be published with GitHub Pages.

## Limitations

This is an MVP. The default local dataset is synthetic unless `use_huggingface` is enabled. The parser and weak labels are deterministic heuristics. The default retrieval backend uses TF-IDF/cosine features rather than prover-aware semantics; Hugging Face sentence embeddings are available as an optional experiment.

## No-GNN MVP Scope

GNN models such as GraphSAGE, R-GCN, HGT, and graph transformers are future work and are not part of this MVP.

## No-LLM MVP Scope

The MVP does not require LLM calls. Labels, features, retrieval, ranking, reports, and homepage generation are deterministic local computations.

## Future Work

Future iterations can replace the synthetic default with a committed real LeanRank sample, scale to larger LeanRank shards, add richer Lean parsing, train graph neural retrieval models, and integrate proof assistant feedback.
