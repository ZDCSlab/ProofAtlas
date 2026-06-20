# ProofAtlas Delivery Audit

Date: 2026-06-20

This audit maps the current ProofAtlas deliverables to concrete artifacts and verification evidence. It follows the current project scope: the production experiment uses `erbacher/LeanRank-data` directly, without reintroducing a custom Lean server/source extractor.

## Scope

Current delivered scope:

- Static homepage/demo for knowledge graph visualization and theorem proof guidance.
- Quantitative ML experiment report over `erbacher/LeanRank-data`.
- Larger LeanRank-data pipeline using theorem-first sampling and BGE embeddings.
- Proof-state-level and theorem-level premise retrieval evaluation on held-out splits.
- Premise trace supervision from LeanRank positive and negative candidate labels.
- Performance reporting for embeddings, ANN indexes, evaluation, and pipeline scale.

Out of current production scope:

- Full Lean server/session tactic-state stream extraction.
- Custom mathlib source extraction independent of `erbacher/LeanRank-data`.
- Claims that proof difficulty prediction or premise retrieval is solved at full mathlib scale.

## Evidence Table

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Knowledge graph homepage | `homepage/index.html`, `homepage/assets/graph_visualization.json`, `homepage/assets/homepage_summary.json` | Delivered |
| New theorem proof guidance demo | `homepage/index.html`, `homepage/assets/theorem_retrieval_case_studies.json`, `src/leanrank_kg/retrieve.py`, `src/leanrank_kg/api.py` | Delivered |
| Experiment report | `outputs/reports/experiment_report.md` | Delivered |
| Held-out quantitative evaluation | `outputs/reports/test_set_evaluation.json`, `outputs/reports/metrics.json` | Delivered |
| LeanRank-data source provenance | `outputs/reports/corpus_manifest.json`, `outputs/reports/premise_trace_supervision_report.json` | Delivered |
| Premise positive/negative supervision | 69,461 positive edges, 663,198 negative edges, positive/negative overlap removed: 4,088 | Delivered |
| Larger real-data pipeline | `configs/proofatlas.yaml`, 292,012 current split rows, 10,000 requested theorems, 350,000 source rows | Delivered |
| GPU BGE embeddings | `outputs/embeddings/embedding_config.json`: `BAAI/bge-base-en-v1.5`, devices `cuda:0` to `cuda:6`, proof-state template `full_name + goal_text` | Delivered |
| ANN performance | `outputs/reports/index_benchmark.json`: hnswlib premise retrieval 19.0x faster than exact cosine at Recall@10 vs exact 0.994 | Delivered |
| API readiness/security review | `outputs/reports/deployment_security_review.json`, `docs/proofatlas_deployment_guide.md` | MVP delivered |
| Real Lean proof-state extraction for new queries | `src/leanrank_kg/lean_check.py` extracts proof states from Lean unsolved-goal diagnostics when validation is requested | Partial |
| Full Lean server/session extraction | Explicitly out of current LeanRank-data production scope | Out of scope |

## Current Retrieval Results

Full held-out test metrics from `configs/proofatlas.yaml`:

| Task | Metric | Value |
| --- | --- | ---: |
| Proof-state premise retrieval | Recall@10 | 0.1162 |
| Proof-state premise retrieval | Recall@100 | 0.2362 |
| Reranked proof-state retrieval | Recall@10 | 0.1513 |
| Theorem premise retrieval | Recall@10 | 0.4940 |
| Theorem premise retrieval | Recall@100 | 0.6889 |
| Theorem premise retrieval | MRR | 0.5609 |
| Premise ranker validation | AUC | 0.8254 |

Interpretation:

- The theorem-level retrieval result is the strongest demo/report result.
- Proof-state candidate retrieval improved after switching proof-state embeddings to `full_name + goal_text`, but remains the main accuracy bottleneck.
- Candidate-depth ablation supports `rerank_candidate_k=50` for the current diagnostic: Recall@10 0.1513 at k=50 versus 0.1382 at k=100.

## Performance Snapshot

From `outputs/reports/index_benchmark.json` and `outputs/reports/pipeline_performance_report.json`:

| Entity | Backend | Rows | Exact ms/query | Indexed ms/query | Speedup | Recall@10 vs exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Premise | hnswlib | 127,561 | 69.0399 | 3.6328 | 19.0044 | 0.9940 |
| ProofState | hnswlib | 23,723 | 12.7668 | 0.8265 | 15.4460 | 0.9580 |
| Theorem | hnswlib | 8,000 | 4.0973 | 0.2318 | 17.6785 | 0.9910 |

Pipeline bottleneck profile:

| Stage group | Field | Value |
| --- | --- | ---: |
| Primary bottleneck | stage | embed |
| Primary bottleneck | seconds | 149.1632 |
| Primary bottleneck | share of total | 0.2704 |
| Top-3 timed stages | share of total | 0.5195 |

Pipeline timing:

- Total seconds: 551.6511
- Saved full-pipeline evaluate stage: 19.1237 seconds
- Current standalone full-heldout evaluation: 23.8288 seconds
- Timing freshness: current; full-pipeline evaluate timing and standalone evaluation timing are aligned.

Pipeline scale profile:

- Dataset: `erbacher/LeanRank-data`
- Source kind: Hugging Face
- Current split rows: 292,012
- Embedding backend: `sentence_transformers`
- Embedding devices: `cuda:0` to `cuda:6`
- Index backend: `hnswlib`
- LeanRank premise supervision ready: true

## Verification Commands

Most recent verified checks:

```bash
make refresh-production-report
make refresh-production-timing
make refresh-production-full-eval
make verify-delivery
```

`make refresh-production-report` is the standard production artifact refresh entrypoint. It runs evaluation, pipeline profiling, experiment-report generation, homepage generation, and audit with `configs/proofatlas.yaml`. By default it executes Python commands through `conda run -n leanrank_kg`; override `PIPELINE_RUN` if the environment is already activated.
`make refresh-production-timing` runs a forced production `full-pipeline` first, then refreshes the full held-out reports from that timing artifact. Use it when the scale-up report needs reliable end-to-end timing instead of cached/partial timing diagnostics.
`make refresh-production-full-eval` runs full held-out proof-state-level and theorem-level evaluation before refreshing the reports. Use it when final quantitative claims need full test-set coverage instead of sampled held-out metrics.
`make verify-delivery` runs the unit tests, production audit, and `git diff --check`. By default `VERIFY_RUN` inherits `PIPELINE_RUN`; override either variable if the environment is already activated.

Recent passing result:

```text
pytest: 74 passed, 4 skipped
audit: 167/167 checks passed
git diff --check: passed
```

## Remaining Gaps

The current deliverable is suitable as a LeanRank-data retrieval demo and experiment report. The remaining gaps for the current ML retrieval deliverable are:

1. Train stronger rerankers beyond the current feature-based logistic/fixed reranker.
2. Expand the held-out evaluation and case-study set when runtime permits.
3. Validate retrieval quality across repeated LeanRank-data refreshes and track metric trends.
4. Harden public API deployment with real auth, rate limiting, TLS, persistent monitoring, and operational logs.
5. Keep full Lean server/session corpus extraction out of the default pipeline unless the project scope explicitly changes away from `erbacher/LeanRank-data`.
