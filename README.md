# ProofAtlas

ProofAtlas builds a non-GNN LeanRank-style math proof knowledge graph. It processes theorem proof-state rows into theorem, proof-state, premise, file, proof-technique, and feature tables, then exposes premise retrieval, similar theorem retrieval, weak proof-technique labeling, difficulty analysis, evaluation reports, and a static homepage.

## Dataset Source

The production experiment source is `erbacher/LeanRank-data`, configured by `configs/proofatlas.yaml`. It uses theorem-first sampling from the Hugging Face dataset and is the source for the committed experiment report and homepage assets.

`configs/sample.yaml` remains the lightweight smoke-test config. It uses a deterministic synthetic LeanRank-shaped sample by default so tests and local demos can run quickly without a large external download. Set `use_huggingface: true` in a sample config to load real Hugging Face rows for smaller exploratory runs.

## What Processed Dataset Is Included

The pipeline writes sampled rows to `data/sample/`, normalized split tables to `data/processed/{train,val,test}/`, and a small reviewer demo dataset to `data/processed/demo/`.

## Knowledge Graph Schema

Node types: `Theorem`, `ProofState`, `Premise`, `FileModule`, `TacticStep`, and `ProofTechnique`.

Base edge types: `has_proof_state`, `appears_in_file`, `positive_uses`, `negative_candidate`, `invokes_premise`, `defined_in_file`, and `at_tactic_step`.

`invokes_premise` is theorem-level: `Theorem -> Premise`, aggregated from positive proof-state premise usage. `positive_uses` and `negative_candidate` remain proof-state-level evidence. `TacticStep` nodes are scoped by theorem and tactic index instead of being shared globally by index.

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

`configs/proofatlas.yaml` now uses theorem-first sampling for the real run. It pulls a larger Hugging Face candidate pool, samples theorem IDs, and keeps all candidate rows for those theorems. This makes the KG scale and evaluation theorem-centric instead of row-centric, which is important for held-out theorem retrieval and recall.

`configs/sample.yaml` keeps TF-IDF available for fast local smoke tests. The production ProofAtlas config uses `BAAI/bge-base-en-v1.5` sentence embeddings on GPU, with proof-state text encoded as `full_name + goal_text` after the held-out query-representation diagnostic showed that this representation gives the best proof-state candidate recall.

```bash
conda activate leanrank_kg
pip install -e ".[hf]"
leanrank-kg full-pipeline --config configs/leanrank_hf_embeddings.yaml --debug-rows 120
```

The embedding backends write the same embedding filenames, so retrieval and graph augmentation continue to work across TF-IDF smoke tests and BGE production runs.

## Pipeline Commands

```bash
make sample
make process
make build-graph
make label
make difficulty
make train-difficulty
make embed
make build-index
make benchmark-index
make security-review
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
leanrank-kg build-index --config configs/sample.yaml
leanrank-kg benchmark-index --config configs/sample.yaml
leanrank-kg profile-pipeline --config configs/proofatlas.yaml
leanrank-kg premise-trace-supervision-report
leanrank-kg security-review --host 127.0.0.1
leanrank-kg train-difficulty --config configs/sample.yaml
leanrank-kg retrieve-premises --proof-state-id "<id>"
leanrank-kg retrieve-premises-for-query --query-text "⊢ ..."
leanrank-kg similar-theorems --theorem-id "<id>"
leanrank-kg similar-theorems-for-query --query-text "theorem ..."
leanrank-kg similar-proof-states-for-query --query-text "⊢ ..."
leanrank-kg retrieve-theorem-guidance --theorem-text "theorem ..."
leanrank-kg retrieve-theorem-guidance --query-file theorem.lean --input-type theorem --domain-hint Algebra
leanrank-kg retrieve-theorem-guidance --query-file theorem.lean --validate-lean
leanrank-kg serve --host 127.0.0.1 --port 8000
leanrank-kg serve --host 127.0.0.1 --port 8000 --require-ready --startup-index-split train
leanrank-kg explain-premise-match --proof-state-id "<id>" --premise-id "<id>"
leanrank-kg build-report --config configs/sample.yaml
leanrank-kg build-experiment-report --config configs/proofatlas.yaml
leanrank-kg audit --config configs/sample.yaml
```

The text-query commands normalize input through `NewTheoremQuery` / `NewProofStateQuery` objects before retrieval. The query layer accepts Lean theorem declarations, raw proof states, standalone goals, optional full names, file paths, and domain hints, then exposes parsed goal text, local hypotheses, symbols, namespace hints, structured binders, parsed conclusion symbols, operator symbols, sort symbols, typeclass symbols, alpha-normalized goal text, and retrieval text in the guidance JSON. The theorem parser handles explicit, implicit, and typeclass binders, top-level declaration goals, imports, namespaces, and `open` namespace hints. Validation writes `outputs/reports/theorem_query_parse_coverage.json` so parse coverage can be reviewed across processed theorem rows.

Pass `--validate-lean` to `retrieve-theorem-guidance` to run an optional Lean syntax check. ProofAtlas uses `lake env lean` when `lake` is available, falls back to `lean`, and reports JSON-safe diagnostics without making Lean a required dependency. When Lean reports `unsolved goals`, ProofAtlas extracts structured proof states from the diagnostics and uses those goal/context blocks as the retrieval query. The response includes `lean_diagnostics.proof_state_extraction` with extraction method, raw block count, extracted count, rejected blocks, and failure reason so users can see whether proof-state extraction succeeded or why it failed.

The text-query commands reuse the saved embedding artifacts in `outputs/embeddings/`. For TF-IDF runs, they load the saved vectorizer. For Hugging Face runs, they load the configured SentenceTransformer model and apply the recorded query prefix. If `outputs/indexes/` exists, text-query retrieval uses the saved nearest-neighbor indexes for proof states, premises, and theorems, then reranks premise candidates with embedding similarity, premise frequency, proof-technique overlap, graph-neighbor evidence, and shared name-token signals. Index artifacts include a manifest with backend, metric, entity type, row count, dimension, embedding-config hash, corpus version, extraction-config hash, backend-specific metadata, and build time; retrieval uses the index only when the manifest matches the current embedding artifacts, otherwise it falls back to direct cosine scoring. The default index backend is sklearn; installing `pip install -e ".[ann]"` enables the optional `hnswlib` ANN backend, installing `pip install -e ".[faiss]"` enables the optional FAISS backend, installing `pip install -e ".[lancedb]"` enables the optional LanceDB vector-store backend, and `index.backend: auto` chooses hnswlib when available. If `outputs/models/premise_ranker.joblib` exists, the learned premise ranker also contributes a `learned_ranker_score` to the final premise ranking. Ranker validation writes `outputs/reports/ranker_validation_metrics.json` with feature columns, feature groups, validation AUC when available, and leave-one-feature-group-out ablations for embedding similarity, namespace/domain, proof technique, difficulty, and frequency signals.

`leanrank-kg benchmark-index` writes `outputs/reports/index_benchmark.json`. The report samples premise and theorem embedding rows, compares saved-index top-k neighbors with exact cosine top-k neighbors, and records exact latency, indexed latency, speedup, index build seconds, and recall/top-1 overlap against exact cosine. This makes backend tradeoffs measurable when switching from sklearn to `hnswlib`, FAISS, or LanceDB.

`leanrank-kg profile-pipeline --config configs/proofatlas.yaml` writes `outputs/reports/pipeline_performance_report.json`. This is the scale-up baseline for the default `erbacher/LeanRank-data` run: it summarizes sample size, processed tables, graph rows, embedding matrix shapes, index manifests/build times, index benchmark latency/recall, pipeline timings, test-set retrieval metrics, artifact compatibility, LeanRank-data premise supervision, and concrete recommendations for larger runs.

`leanrank-kg premise-trace-supervision-report` writes `outputs/reports/premise_trace_supervision_report.json`. It summarizes positive premise edges, negative candidates, negative-candidate hardness, and split-level supervision coverage from the normalized LeanRank-data artifacts.

`leanrank-kg build-experiment-report --config configs/proofatlas.yaml` writes `outputs/reports/experiment_report.md`. This is the quantitative ML report deliverable: it treats ProofAtlas as a ranking/retrieval task on `erbacher/LeanRank-data`, uses the held-out test split for final metrics, and reports proof-state-level and theorem-level Recall@k, MRR, MAP, nDCG, gold-premise coverage, domain-level metric breakdowns, index benchmark results, pipeline timings, and scale-up notes.

The default index backend remains sklearn for lightweight local demos and tests. Installing `pip install -e ".[ann]"` enables `hnswlib`, `pip install -e ".[faiss]"` enables FAISS, and `pip install -e ".[lancedb]"` enables an optional LanceDB vector-store backend. Set `index.backend: lancedb` to write LanceDB tables under `outputs/indexes/`; set `index.create_vector_index: true` for LanceDB to build a vector index when the installed LanceDB version supports it.

Ranked premise results include both numeric `signals` and human-readable `ranking_reasons`, such as embedding similarity, learned ranker score, namespace/domain match, proof-technique compatibility, graph-neighbor evidence from similar theorems, theorem-level premise degree, theorem-neighborhood premise evidence, symbol/context overlap, premise frequency, parsed conclusion-symbol overlap, and shared name tokens. The learned ranker trains on the same feature families and reports ablations for embedding, namespace/domain, proof-technique, difficulty, frequency, symbol-overlap, graph, and theorem-neighborhood signals. The theorem-guidance response also includes `similar_proof_states`, retrieved directly from historical proof-state embeddings, so users can inspect nearby goal/context patterns in addition to similar theorem statements.

The theorem guidance `difficulty_profile` combines query-level signals, a historical prior from retrieved similar theorems, and a trained proof-state difficulty estimator when `outputs/models/difficulty_estimator.joblib` is available. Difficulty feature generation writes `outputs/reports/difficulty_target_report.json` and now includes a theorem-complexity target derived from proof length, tactic count, premise count, negative-candidate count, negative-candidate hardness, and proof-state difficulty. Guidance also includes a `retrieval_policy` block that uses a pre-retrieval difficulty estimate to expand the premise candidate pool for harder queries before reranking. The response reports the calibrated score, bucket, calibration source, prior confidence, trained estimator score, residual-based uncertainty interval, calibration bins, and the similar theorem difficulty neighbors used as evidence. These diagnostics are meant to show how reliable the estimate is; they are not a claim that proof difficulty prediction is solved on full mathlib.

Sampling writes `outputs/reports/corpus_manifest.json` with dataset source, source kind, config hash, sample plan, split counts, corpus version, LeanRank provenance, and data-supervision profile. Validation writes `outputs/reports/artifact_compatibility_report.json`, which checks the current config hash, corpus split counts, processed row counts, embedding metadata/matrix rows, and index manifests. Report generation also writes `outputs/reports/refresh_dashboard.json`, a compact dashboard for KG scale, domain coverage, retrieval quality, parsing coverage, index benchmarks, difficulty calibration, data-supervision status, and artifact compatibility. If a previous dashboard exists, `outputs/reports/refresh_trend.json` compares the new run against that snapshot and records metric deltas and quality-gate changes; `outputs/reports/refresh_history.json` keeps a bounded history. The homepage copies the corpus manifest, refresh dashboard, trend report, and history report into `homepage/assets/` and shows the source/config hash in the Reproducibility section.

## Evaluation Results

After `make demo` or `make smoke`, metrics are written to `outputs/reports/metrics.json`, held-out test-set metrics to `outputs/reports/test_set_evaluation.json`, the experiment report to `outputs/reports/experiment_report.md`, premise supervision statistics to `outputs/reports/premise_trace_supervision_report.json`, retrieval examples to `outputs/reports/retrieval_examples.json`, index benchmarks to `outputs/reports/index_benchmark.json`, pipeline performance profile data to `outputs/reports/pipeline_performance_report.json`, refresh dashboard data to `outputs/reports/refresh_dashboard.json`, refresh trend/history data to `outputs/reports/refresh_trend.json` and `outputs/reports/refresh_history.json`, and graph summaries to `outputs/reports/graph_stats_summary.json`.

Current `configs/proofatlas.yaml` sampled held-out test results:

| Task | Metric | Value |
| --- | --- | ---: |
| Proof-state premise retrieval | Recall@10 | 0.1279 |
| Proof-state premise retrieval | Recall@100 | 0.3090 |
| Reranked proof-state diagnostic | Recall@10 | 0.1513 |
| Theorem-level premise retrieval | Recall@10 | 0.4233 |
| Theorem-level premise retrieval | Recall@100 | 0.6642 |
| Theorem-level premise retrieval | MRR | 0.5473 |
| Learned premise ranker | validation AUC | 0.8254 |

To refresh the production evaluation artifacts after changing retrieval code or config:

```bash
conda activate leanrank_kg
make refresh-production-report
```

This runs `evaluate`, `profile-pipeline`, `build-experiment-report`, `build-homepage`, and `audit` with `configs/proofatlas.yaml`. Override the production config with `make refresh-production-report PRODUCTION_CONFIG=<path>`.
By default it executes Python commands through `conda run -n leanrank_kg`; override that with `make refresh-production-report PIPELINE_RUN=` if your environment is already activated.

To refresh a reliable end-to-end production timing baseline for scale-up planning:

```bash
make refresh-production-timing
```

This runs `leanrank-kg full-pipeline --config configs/proofatlas.yaml --force`, then refreshes the evaluation report, homepage, and audit against the newly written timing artifact.

To refresh full held-out production evaluation metrics instead of the sampled test-set metrics:

```bash
make refresh-production-full-eval
```

This runs `leanrank-kg evaluate --config configs/proofatlas.yaml --full-heldout`, then refreshes the profile, experiment report, homepage, and audit. The reranked proof-state diagnostic and case-study limits still follow `configs/proofatlas.yaml`; the override applies to the core proof-state-level and theorem-level held-out retrieval metrics.

To verify the delivery after refreshing artifacts:

```bash
make verify-delivery
```

This runs unit tests, the production audit, and `git diff --check`.
By default it executes Python commands through `conda run -n leanrank_kg`; override that with `make verify-delivery VERIFY_RUN=` if your environment is already activated. `VERIFY_RUN` defaults to `PIPELINE_RUN`, so `make verify-delivery PIPELINE_RUN=` is also valid.

The proof-state-level and theorem-level retrieval metrics separately report held-out gold premise counts that exist in the train premise index and counts missing from the train index, so Recall/MRR/MAP/nDCG are interpreted against the actually retrievable gold set.

Validation reports are written to `outputs/reports/schema_validation_summary.json`, `outputs/reports/split_leakage_report.json`, `outputs/reports/context_parse_coverage.json`, and `outputs/reports/graph_validation_summary.json`.
Artifact compatibility is written to `outputs/reports/artifact_compatibility_report.json` and should pass before comparing retrieval quality across refreshed mathlib or LeanRank artifacts.

The MVP completion audit is written to `outputs/reports/mvp_completion_audit.json`. It checks processed KG artifacts, embeddings, retrieval indexes, held-out test-set evaluation, experiment report, pipeline performance profile, theorem guidance case studies, theorem-level evaluation coverage, graph visualization assets, validation reports, and homepage sections.

## Homepage/Demo

The static homepage is generated at `homepage/index.html`. It uses real pipeline outputs copied into `homepage/assets/` and can be published with GitHub Pages.

The homepage includes a compact SVG knowledge graph sample backed by `homepage/assets/graph_visualization.json`, precomputed theorem guidance case studies, and a local API demo form.

The two final presentation artifacts are:

- `homepage/index.html` for the knowledge graph visualization and new-theorem proof guidance demo.
- `outputs/reports/experiment_report.md` for the held-out test-set ML ranking/retrieval results.
- `docs/proofatlas_delivery_audit.md` for the artifact/evidence audit and remaining gaps.

For a local interactive retrieval endpoint:

```bash
conda activate leanrank_kg
pip install -e ".[api]"
leanrank-kg serve --host 127.0.0.1 --port 8000
```

After `leanrank-kg build-homepage`, the API server also serves the demo page at `http://127.0.0.1:8000/`. The page includes a local retrieval form that posts to `/retrieve-theorem-guidance`.

Retrieval artifacts are cached inside the running Python process using file timestamp keys, so the API server reuses loaded processed tables, embeddings, nearest-neighbor indexes, vectorizers, and rankers across requests while still seeing regenerated artifacts after pipeline reruns.

The API exposes:

```text
GET  /health
GET  /metrics
GET  /metrics/prometheus
POST /retrieve-theorem-guidance
```

`GET /health` reports whether required processed tables, embeddings, graph artifacts, and optional indexes/rankers are present for an `index_split`. `GET /metrics` reports in-process request counts, success/failure counts, average latency, and the last guidance request summary; `GET /metrics/prometheus` exposes the same counters and latency gauges as Prometheus-compatible text. The guidance endpoint validates `input_type`, `index_split`, text length, and requested result limits before running retrieval; missing required artifacts return a clear service error. Each successful guidance response includes a `service.request_id` and `service.duration_ms`.

Use `--require-ready` to fail API startup when required artifacts for `--startup-index-split` are missing. The same behavior can be controlled with `PROOFATLAS_REQUIRE_READY=1` and `PROOFATLAS_STARTUP_INDEX_SPLIT=train`.

Run `leanrank-kg security-review --host 127.0.0.1` before an interactive demo. It writes `outputs/reports/deployment_security_review.json` with request-limit, readiness, host-binding, metrics, and public-exposure checks. Public API exposure requires `--public-exposure --require-ready` plus explicit deployment assertions through `PROOFATLAS_ACCESS_CONTROL_CONFIRMED=1`, `PROOFATLAS_NETWORK_CONTROLS_CONFIRMED=1`, and `PROOFATLAS_MONITORING_CONFIRMED=1` after those external controls have been implemented.

For reproducible local, server, notebook, and static-hosted demo modes, see [docs/proofatlas_deployment_guide.md](docs/proofatlas_deployment_guide.md).

Example request body:

```json
{
  "theorem_text": "theorem Nat.self_eq (x : Nat) : x = x := by simpa",
  "input_type": "theorem",
  "k_premises": 10,
  "k_theorems": 5,
  "index_split": "train",
  "validate_lean": false
}
```

## Limitations

This is an MVP. The lightweight local smoke-test dataset is synthetic unless `use_huggingface` is enabled. The production experiment uses `erbacher/LeanRank-data`, BGE sentence embeddings, ANN candidate retrieval, and a learned/fixed reranker. The parser and weak proof-technique labels are deterministic heuristics. New theorem guidance currently uses embedding retrieval plus heuristic proof-technique and difficulty signals; graph-neural reranking remains future work.

## No-GNN MVP Scope

GNN models such as GraphSAGE, R-GCN, HGT, and graph transformers are future work and are not part of this MVP.

## No-LLM MVP Scope

The MVP does not require LLM calls. Labels, features, retrieval, ranking, reports, and homepage generation are deterministic local computations.

## Future Work

Future iterations can replace the synthetic default with a committed real LeanRank sample, scale to larger LeanRank shards, add richer Lean parsing, train graph neural retrieval models, and integrate proof assistant feedback.
