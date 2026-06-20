# ProofAtlas Current Status and Next Steps

Date: 2026-06-20

## Project Goal

ProofAtlas has two core goals:

1. Build and present a Lean/mathlib proof knowledge graph that organizes theorems, lemmas, proof states, premises, files/modules, tactic steps, and proof techniques.
2. Use this knowledge graph as the retrieval and recommendation engine for a proof assistance system.

The target workflow is:

```text
Given a new theorem or proof state
-> parse and encode the query
-> retrieve relevant premises, similar theorems, proof techniques, proof patterns, and difficulty signals
-> return ranked and explainable proof guidance
```

The final demo should show both:

- Knowledge graph visualization for Lean/mathlib proof knowledge.
- New theorem proof guidance, including relevant premises, similar theorems, likely proof techniques, related proof patterns, difficulty profile, and ranking explanations.

## Current Implementation Status

ProofAtlas has moved beyond the initial proof-state retrieval MVP. The current implementation now supports an end-to-end theorem-guidance path over the processed LeanRank/mathlib artifacts.

Completed components include:

- Knowledge graph nodes for `Theorem`, `ProofState`, `Premise`, `FileModule`, `TacticStep`, and `ProofTechnique`.
- Corrected theorem-centric KG semantics:
  - `Theorem -> Premise` edges for aggregated premise invocation.
  - `ProofState -> Premise` edges for local positive and negative premise candidates.
  - theorem-scoped tactic-step nodes.
- Embeddings for proof states, premises, and theorems.
- Persistent nearest-neighbor indexes for train-side premise and theorem retrieval.
- Persistent nearest-neighbor indexes for historical proof-state retrieval.
- Pluggable index backend support with sklearn default, optional hnswlib ANN backend, optional FAISS backend, optional LanceDB vector-store backend, and `auto` backend resolution.
- Index benchmark report comparing saved-index latency and top-k overlap against exact cosine retrieval.
- Out-of-KG query retrieval for theorem text or proof-state text.
- Structured query parsing for Lean theorem/lemma/example snippets, explicit/implicit/typeclass binders, imports, namespaces, `open` hints, goal text, parsed conclusion symbols, operator symbols, sort/typeclass symbols, and alpha-normalized goal text.
- Theorem query parse coverage report for processed theorem rows, included in refresh dashboard parsing diagnostics.
- Unified theorem guidance retrieval that returns:
  - relevant premises,
  - similar theorems,
  - likely proof techniques,
  - related proof states and proof patterns,
  - graph evidence,
  - difficulty profile,
  - ranking explanations.
- Optional Lean syntax diagnostics through `lake env lean` or `lean`.
- Structured extraction of proof states from Lean `unsolved goals` diagnostics, with extracted goals used as the retrieval query when available and extraction reports explaining raw blocks, rejected blocks, counts, provenance, and failure reasons.
- Learned premise-ranker integration when a trained ranker artifact is available.
- Learned-ranker feature group metadata and ablation report for embedding, namespace/domain, proof-technique, difficulty, frequency, symbol-overlap, graph, and theorem-neighborhood signals.
- Graph-aware and text-aware ranking reasons for returned premises.
- Theorem-level evaluation with Recall@k, MRR, MAP, nDCG, coverage counts, and case-study output.
- Query difficulty profiles that combine heuristic query features with historical difficulty priors from retrieved similar theorems.
- Adaptive retrieval policy that expands premise candidate depth for harder queries before reranking.
- Initial trained proof-state difficulty estimator artifact with MAE metrics, guidance-time estimator signal, and theorem-complexity targets derived from proof length, tactic count, premise count, and negative-candidate signals.
- Difficulty estimator calibration bins, residual quantiles, and guidance-time uncertainty intervals.
- Static homepage sections for knowledge graph visualization and new theorem proof guidance.
- Optional FastAPI service for interactive proof-guidance queries.
- API request validation, bounded retrieval limits, and `/health` artifact readiness reporting.
- API request metrics, request IDs, structured logging hooks, optional startup readiness enforcement, and a deployment security/readiness review report.
- Deployment guide for static homepage, local API, readiness-gated server, notebook, and hosted demo modes.
- Corpus provenance manifest with dataset source, source kind, config hash, sample plan, split counts, corpus version, Lean/mathlib version provenance, extraction config hash, and env/git/Lean fallback resolution.
- Corpus data-supervision profile that distinguishes synthetic demo rows and LeanRank proof-state rows, including whether the corpus is suitable for ranking experiments.
- Artifact compatibility report tying config hash, corpus split counts, processed tables, embeddings, and index manifests together.
- Artifact compatibility warnings for synthetic corpora, so internally consistent demo artifacts are not mistaken for prediction-quality results.
- Refresh dashboard report for KG scale, domain coverage, retrieval quality, parsing coverage, index benchmark, difficulty calibration, and artifact compatibility.
- Refresh trend report comparing the current dashboard with the previous dashboard snapshot, including scale, retrieval, parsing, difficulty, index, and quality-gate deltas.
- Bounded refresh history report preserving recent refresh snapshots for longer trend review.
- Premise-supervision report for LeanRank-data positive edges, negative candidates, hard-negative statistics, and split-level supervision coverage.
- Expanded audit checks covering indexes, theorem case studies, homepage assets, API/query modules, and ranking explanations.
- Runtime artifact caching for repeated retrieval calls.
- Index build-time metadata and deterministic index benchmark reports.

The strongest working retrieval path is now:

```text
Given a new theorem statement or proof goal
-> build a structured query
-> optionally validate with Lean and extract unsolved-goal proof states
-> embed the query
-> retrieve and rerank train-side premises and similar theorems
-> attach graph evidence, proof-technique hints, proof-pattern evidence, difficulty signals, and explanations
-> return JSON guidance through CLI, Python API, or the optional web service
```

## Current Production Experiment Snapshot

The current production experiment uses `configs/proofatlas.yaml` over `erbacher/LeanRank-data`, BGE embeddings, and hnswlib ANN indexes. The detailed artifact/evidence audit is in `docs/proofatlas_delivery_audit.md`.

Sampled held-out test metrics:

| Task | Metric | Value |
| --- | --- | ---: |
| Proof-state premise retrieval | Recall@10 | 0.1279 |
| Proof-state premise retrieval | Recall@100 | 0.3090 |
| Reranked proof-state diagnostic | Recall@10 | 0.1513 |
| Theorem-level premise retrieval | Recall@10 | 0.4233 |
| Theorem-level premise retrieval | Recall@100 | 0.6642 |
| Theorem-level premise retrieval | MRR | 0.5473 |
| Learned premise ranker | validation AUC | 0.8254 |

Performance snapshot:

- Current split rows: 292,012
- Train premise index rows: 127,561
- Train proof-state index rows: 23,723
- Train theorem index rows: 8,000
- Premise hnswlib speedup vs exact cosine: 18.3x at Recall@10 vs exact 0.989
- Proof-state hnswlib speedup vs exact cosine: 15.2x at Recall@10 vs exact 0.991
- Theorem hnswlib speedup vs exact cosine: 12.8x at Recall@10 vs exact 0.994

## Completed Roadmap Items

### Phase 1: KG Semantics

Status: completed.

Implemented changes:

- `invokes_premise` now represents theorem-level premise usage.
- proof-state-level premise labels remain available for positive and negative examples.
- tactic steps are scoped to their theorem context.
- graph validation, report generation, tests, README notes, and homepage summaries were updated around the corrected semantics.

### Phase 2: Out-of-KG Query Embedding

Status: completed.

Implemented changes:

- Added retrieval for raw theorem/proof-state text that is not already present in KG tables.
- Reused the configured embedding model and embedding prefixes.
- Added text-query APIs and CLI support.
- Added persistent local indexes so retrieval does not need to rebuild neighbors on every call.

### Phase 3: Theorem-Level Retrieval

Status: completed.

Implemented changes:

- Added `retrieve_knowledge_for_theorem`.
- Added structured theorem query construction.
- Added structured binder records, parsed conclusion symbols, and alpha-normalized goal text to theorem query objects.
- Added premise retrieval, similar theorem retrieval, proof-technique hints, related proof-state evidence, graph evidence, and difficulty profiling in one result object.
- Added direct similar-proof-state retrieval from historical proof-state embeddings and index manifests.
- Added similar-theorem historical difficulty priors and a trained difficulty-estimator signal to query difficulty profiles.
- Added graph-aware and learned-ranker-aware premise scoring.
- Added human-readable ranking explanations.
- Added learned-ranker feature ablations to make signal contribution measurable.

### Phase 4: Theorem-Level Evaluation

Status: completed.

Implemented changes:

- Added theorem-level retrieval metrics.
- Reported held-out theorem counts.
- Separated gold premises available in the train retrieval index from gold premises missing from the train index.
- Added theorem retrieval case-study JSON output.

### Phase 5: Homepage Case Studies and KG Visualization

Status: completed.

Implemented changes:

- Added precomputed theorem-guidance case studies to homepage assets.
- Added proof-guidance cards showing premises, similar theorems, likely techniques, difficulty signals, and ranking reasons.
- Added a compact knowledge graph visualization asset for theorem, premise, proof-state, tactic, technique, and module relations.

### Phase 6: Interactive Service

Status: initial version completed.

Implemented changes:

- Added optional FastAPI service.
- Added health and theorem-guidance endpoints.
- Added static homepage serving when generated assets exist.
- Added a homepage form that can call the proof-guidance endpoint.
- Added a Lean validation toggle and result display for query source and extracted proof-state count.
- Added request validation, retrieval result limits, and health checks for required/optional artifacts.
- Added `/metrics`, per-request service metadata, and optional `--require-ready` startup policy for deployed services.

### Phase 7: Index Benchmarking

Status: initial version completed.

Implemented changes:

- Added index build-time metadata to saved index manifests.
- Added `leanrank-kg benchmark-index`.
- Added `outputs/reports/index_benchmark.json` with exact-cosine latency, saved-index latency, speedup, and top-k overlap/recall against exact cosine for premise and theorem indexes.
- Added Makefile, config, README, and audit coverage for the benchmark artifact.

### Phase 8: API Operations Hardening

Status: initial version completed.

Implemented changes:

- Added in-process request metrics for total, successful, failed, validation-error, and missing-artifact request counts.
- Added average guidance latency and last-request metadata through `GET /metrics` and `GET /health`.
- Added request IDs and duration metadata to guidance responses.
- Added structured logging hooks for guidance requests.
- Added `leanrank-kg serve --require-ready --startup-index-split <split>` to fail startup when required artifacts are missing.
- Added homepage demo support for slow-query status, service request metadata, and live API metrics.
- Added `docs/proofatlas_deployment_guide.md` for reproducible local, server, notebook, and hosted static demos.

### Phase 9: Data Refresh Compatibility

Status: initial version completed.

Implemented changes:

- Added `outputs/reports/artifact_compatibility_report.json`.
- Validates current config hash against the corpus manifest.
- Checks corpus manifest split counts against processed theorem/proof-state tables.
- Checks embedding metadata counts against embedding matrix row counts.
- Checks index manifests against index metadata, saved artifact paths, entity type, split, and embedding-config hash.
- Added audit and test coverage so stale refresh artifacts are detected.
- Added `outputs/reports/refresh_dashboard.json` and homepage refresh dashboard section for post-refresh health checks.
- Added `outputs/reports/refresh_trend.json` and homepage trend indicators for comparisons against the previous generated dashboard snapshot.
- Added `outputs/reports/refresh_history.json` and homepage history-entry indicators for bounded multi-refresh trend tracking.

## Remaining Gaps

The project now has the main theorem-guidance loop, but several important gaps remain before it becomes a robust proof-assistance system.

### 1. Richer LeanRank-Data-Aware Parsing

Current query parsing handles common declaration structure, including explicit, implicit, and typeclass binders, parsed conclusion symbols, operator symbols, sort/typeclass symbols, and alpha-normalized goal text. It also writes theorem query parse coverage diagnostics. It is still lightweight and should be upgraded with deeper Lean-aware extraction.

Next steps:

- Keep the production pipeline based directly on `erbacher/LeanRank-data`; do not add a custom Lean server/source extractor to the default workflow.
- Improve lightweight parsing over theorem text and LeanRank proof-state rows, including namespaces, constants, binders, hypotheses, typeclass constraints, and conclusion structure.
- Normalize theorem statements more completely so alpha-renaming, notation expansion, coercions, and local binder names do not dominate retrieval.
- Expand parsed query features in ranking and explanations beyond the current binder/conclusion-symbol signals.

### 2. LeanRank Proof-State Utilization

Current proof-pattern retrieval uses both direct historical proof-state retrieval and proof states from similar theorems. The production data source is `erbacher/LeanRank-data`, which already provides proof-state rows, positive premises, negative candidates, and hardness features. The optional Lean `unsolved goals` diagnostic parser is only for user-query diagnostics and is not a corpus extractor.

Next steps:

- Scale evaluation over larger LeanRank-data slices and keep train/val/test retrieval strictly split-aware.
- Improve proof-pattern retrieval from theorem-level text, LeanRank proof-state goal/context structure, and similar-theorem neighborhoods.
- Keep diagnostics explicit when optional Lean syntax checking is unavailable or cannot extract an unsolved-goal block.

### 3. Scalable ANN Retrieval

The current persistent nearest-neighbor indexes are manifest-checked and now support a sklearn default plus optional hnswlib ANN, FAISS, and LanceDB vector-store backends. A deterministic benchmark report now measures saved-index latency and top-k overlap against exact cosine. Index manifests include embedding compatibility, corpus provenance, and backend-specific metadata. Larger mathlib-scale or multi-corpus deployments still need broader validation and benchmark coverage.

Next steps:

- Validate the optional LanceDB backend on larger mathlib-scale corpora and document operational storage choices.
- Keep the sklearn backend as a lightweight default for tests and small demos.
- Extend benchmark coverage across larger mathlib-scale corpora, more query distributions, and future LanceDB-style backends.

### 4. Stronger Learned Ranking

The current learned ranker is an optional feature-based reranker with feature-group ablation reporting. It now learns from embedding, namespace/domain, proof-technique, difficulty, frequency, symbol-overlap, graph premise-degree, and theorem-neighborhood premise evidence. A stronger system should still compare this feature-based model with richer neural and graph-aware rankers.

Next steps:

- Train and evaluate richer premise-ranking models beyond the current logistic feature reranker.
- Add deeper theorem-similarity, dependency-path, and graph-neighborhood features directly to the training set.
- Compare fixed scoring, logistic/GBDT reranking, cross-encoder reranking, and graph neural reranking.
- Extend ablations to richer graph features, symbol-overlap features, theorem-neighborhood features, and cross-encoder/GNN rerankers.

### 5. Better Difficulty Modeling

The current difficulty profile combines query heuristics, a historical prior from retrieved similar theorem feature tables, and an initial trained proof-state difficulty estimator with residual-based uncertainty and calibration bins. Difficulty feature generation now also writes a theorem-complexity target from proof length, tactic count, premise count, failed negative-candidate count, negative-candidate hardness, and proof-state difficulty. Guidance uses a pre-retrieval difficulty estimate to expand premise candidate depth for harder queries before reranking. It is more informative and better diagnosed than the initial heuristic, but it is not yet validated against external proof-complexity labels.

Next steps:

- Validate the current theorem-complexity label against external or stronger proof-complexity labels.
- Train richer difficulty estimators over theorem, proof-state, dependency-depth, and failed-proof-attempt features.
- Validate calibrated confidence and uncertainty against external or stronger proof-complexity labels.
- Validate and tune adaptive retrieval-depth policy against real proof-complexity labels, then extend difficulty estimates to proof-strategy recommendations.

### 6. Production-Ready API and Demo

The current FastAPI service and homepage demo are useful for local inspection and now include basic request validation, artifact readiness checks, JSON request metrics, Prometheus-compatible metrics text export, request IDs, structured logging hooks, optional startup readiness enforcement, and a generated deployment security/readiness review report. They are not yet a fully hardened production deployment.

Next steps:

- Connect the in-process JSON and Prometheus-compatible metrics endpoints to persistent production monitoring or deployment logs.
- Add deployment-specific startup policies for server, notebook, and hosted demos.
- Continue improving frontend loading, empty, error, and slow-query states with richer user-facing recovery actions.
- Expand deployment documentation with production-specific infrastructure examples once a target host is selected.
- Use the generated security review as a public-exposure gate, and replace deployment assertion flags with concrete host-specific auth, rate-limit, network, TLS, and monitoring integration once a target host is selected.

### 7. LeanRank-data Scale-Up

The current project scope uses `erbacher/LeanRank-data` directly. The pipeline writes a corpus manifest, artifact compatibility report, refresh dashboard, refresh trend report, bounded refresh history, held-out test-set evaluation, premise-supervision report, pipeline timing report, and experiment report. The manifest records dataset/config provenance and a data-supervision profile for LeanRank-data versus synthetic demo rows.

Next steps:

- Scale the LeanRank-data run beyond the current sample plan.
- Tune index and embedding backends using timing and benchmark reports.
- Improve test-set ranking metrics and report deltas across runs.
- Expand artifact compatibility tests across real mathlib refreshes and historical artifact versions.
- Validate refresh trend/history comparisons across repeated real mathlib refreshes and choose a durable storage location for long-running refresh history.

## Recommended Next Work

The highest-value next tasks are:

1. Add robust Lean-aware theorem parsing and proof-state extraction.
2. Replace or supplement the local nearest-neighbor index with a scalable ANN backend.
3. Train a stronger learned reranker using graph, theorem, proof-state, and symbol-overlap features.
4. Improve difficulty estimation from heuristics into an evaluated prediction task.
5. Harden the FastAPI and homepage demo for reproducible external review.
6. Document and automate full mathlib data refreshes.

## One-Sentence Summary

ProofAtlas now has an end-to-end theorem-guidance MVP; the next stage is to make it more Lean-aware, more scalable, better learned, and robust enough for repeatable mathlib-scale proof assistance demos.
