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
| Premise positive/negative supervision | 69,461 positive edges, 663,198 negative edges, positive/negative overlap removed: 4,088; train trace profile includes 54,897 positive trace rows, 530,413 hard-negative rows, 128,855 high-hardness rows, and 3 example proof-state traces; hard-negative pair evidence compares 530,413 train negative candidates to same-proof-state positive premises, with 45.5% same namespace, 98.1% same domain, 97.7% same subdomain, and 69.1% nonzero name-token overlap; ranker training sample uses 10,000 positive pairs and 10,000 hard-negative pairs with 100.0% nonzero hard-negative hardness coverage | Delivered |
| Larger real-data pipeline | `configs/proofatlas.yaml`, 292,012 current split rows, 10,000 requested theorems, 350,000 source rows | Delivered |
| GPU BGE embeddings | `outputs/embeddings/embedding_config.json`: `BAAI/bge-base-en-v1.5`, devices `cuda:0` to `cuda:6`, proof-state template `full_name + goal_text` | Delivered |
| ANN performance | `outputs/reports/index_benchmark.json`: hnswlib premise retrieval 19.0x faster than exact cosine at Recall@10 vs exact 0.994 | Delivered |
| API readiness/security review | `outputs/reports/deployment_security_review.json`, `docs/proofatlas_deployment_guide.md` | MVP delivered |
| Real Lean proof-state extraction for new queries | `src/leanrank_kg/lean_check.py` extracts proof states from Lean unsolved-goal diagnostics when validation is requested; theorem/lemma/example statements without proof bodies can be checked as temporary `:= by` initial-goal skeletons; extracted diagnostics include ordered tactic-state trace metadata with stable trace IDs and tactic indices; `outputs/reports/lean_diagnostic_extraction_report.json` audits success, adjacent-goal splitting, initial-goal skeleton fallback, tactic-state trace counts, deduplication, timeout-stderr extraction, and failure-explanation fixtures | Partial |
| Full Lean server/session extraction | Explicitly out of current LeanRank-data production scope | Out of scope |

## Current Retrieval Results

Full held-out test metrics from `configs/proofatlas.yaml`:

| Task | Metric | Value |
| --- | --- | ---: |
| Proof-state premise retrieval | Recall@10 | 0.1162 |
| Proof-state premise retrieval | Recall@100 | 0.2362 |
| Reranked proof-state retrieval | Recall@10 | 0.1689 |
| Theorem premise retrieval | Recall@10 | 0.4940 |
| Theorem premise retrieval | Recall@100 | 0.6889 |
| Theorem premise retrieval | MRR | 0.5609 |
| Premise ranker validation | AUC | 0.8214 |

Failure diagnosis from `outputs/reports/experiment_report.md`:

| Task | Diagnosis | Queries | Share of evaluated | Share of retrievable |
| --- | --- | ---: | ---: | ---: |
| Proof-state premise retrieval | candidate_pool_miss_top_100 | 1,823 | 59.7% | 64.4% |
| Proof-state premise retrieval | reranking_headroom_after_top10 | 458 | 15.0% | 16.2% |
| Theorem premise retrieval | candidate_pool_miss_top_100 | 55 | 5.5% | 5.8% |
| Theorem premise retrieval | reranking_headroom_after_top10 | 133 | 13.3% | 13.9% |

Held-out metric uncertainty:

| Task | Metric | n | 95% CI low | 95% CI high | Half-width |
| --- | --- | ---: | ---: | ---: | ---: |
| Proof-state premise retrieval | Recall@10 | 3053 | 0.1049 | 0.1276 | 0.0114 |
| Proof-state premise retrieval | Recall@100 | 3053 | 0.2211 | 0.2512 | 0.0151 |
| Theorem premise retrieval | Recall@10 | 1000 | 0.4630 | 0.5250 | 0.0310 |
| Theorem premise retrieval | Recall@100 | 1000 | 0.6602 | 0.7176 | 0.0287 |

Interpretation:

- The theorem-level retrieval result is the strongest demo/report result.
- Proof-state candidate retrieval improved after switching proof-state embeddings to `full_name + goal_text`, but remains the main accuracy bottleneck.
- Candidate-depth ablation supports `rerank_candidate_k=50` for the current diagnostic: Recall@10 0.1689 at k=50 versus 0.1382 at k=100.

## Performance Snapshot

From `outputs/reports/index_benchmark.json` and `outputs/reports/pipeline_performance_report.json`:

| Entity | Backend | Rows | Exact ms/query | Indexed ms/query | Speedup | Recall@10 vs exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Premise | hnswlib | 127,561 | 69.1912 | 3.6822 | 18.7909 | 0.9930 |
| ProofState | hnswlib | 23,723 | 12.8202 | 0.8267 | 15.5081 | 0.9500 |
| Theorem | hnswlib | 8,000 | 4.0994 | 0.2423 | 16.9156 | 0.9960 |

Pipeline bottleneck profile:

| Stage group | Field | Value |
| --- | --- | ---: |
| Primary bottleneck | stage | embed |
| Primary bottleneck | seconds | 148.6884 |
| Primary bottleneck | share of total | 0.2978 |
| Top-3 timed stages | share of total | 0.5610 |

Performance acceptance gates:

| Gate group | Field | Value |
| --- | --- | ---: |
| Required gates | passed | True |
| Advisory gates | passed | True |
| All gates | passed / total | 10 / 10 |

Scale projection:

| Projection | Target rows | Total seconds | Embed seconds | Index build seconds |
| --- | ---: | ---: | ---: | ---: |
| current_1x | 292012 | 499.3380 | 148.6884 | 6.3293 |
| current_2x | 584024 | 998.6760 | 297.3768 | 12.6587 |
| current_5x | 1460060 | 2496.6901 | 743.4420 | 31.6467 |
| configured_source_rows | 350000 | 598.4970 | 178.2151 | 7.5862 |

Execution mode summary:

| Execution | Field | Value |
| --- | --- | ---: |
| Embedding mode | value | multi_gpu_sentence_transformer |
| Embedding GPU | active | True |
| Embedding GPU | multi GPU | True |
| Evaluation mode | value | batched_gpu_retrieval_evaluation |
| Evaluation GPU | active | True |
| Index mode | value | hnswlib_ann_candidate_generation |
| ANN index | active | True |
| Primary bottleneck | stage | embed |
| Artifact reuse | default | True |

embedding is still the largest timed stage even with GPU encoding, so artifact reuse matters for report and reranking refreshes

Artifact reuse and retraining policy:

| Artifact | Field | Value |
| --- | --- | ---: |
| Artifact reuse | reuse by default | True |
| Embedding cache | rows | 244391 |
| Index cache | entity manifests | 12 |
| Premise ranker | exists | True |
| Difficulty estimator | exists | True |

Do not retrain by default. Reuse embeddings, indexes, and trained models for report/homepage refreshes; rerun ranker training only after ranker feature, label, split, or relevant config changes.

Pipeline timing:

- Total seconds: 499.3380
- Saved full-pipeline evaluate stage: 19.4681 seconds
- Current standalone full-heldout evaluation: 26.3841 seconds
- Reranked proof-state diagnostic: 20 / 3053 sampled queries; projected full rerank 2206.3825 seconds; 2211.1546x batched seconds/query
- Timing freshness: current; full-pipeline evaluate timing and standalone evaluation timing are aligned.

Pipeline scale profile:

- Dataset: `erbacher/LeanRank-data`
- Source kind: Hugging Face
- Current split rows: 292,012
- Embedding backend: `sentence_transformers`
- Embedding devices: `cuda:0` to `cuda:6`
- Index backend: `hnswlib`
- LeanRank premise supervision ready: true
- Artifact storage: 2.8431 GiB total, 10,454.1847 bytes per processed row
- Largest storage component: `outputs/indexes`, 2,295,002,043 bytes
- Unreferenced index artifacts: 1,502,501,178 bytes (1.3993 GiB) not pointed to by current manifests
- Projected storage at current_5x: 14.2155 GiB

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
pytest: 100 passed, 4 skipped
audit: 181/181 checks passed
git diff --check: passed
```

## Remaining Gaps

The current deliverable is suitable as a LeanRank-data retrieval demo and experiment report. The remaining gaps for the current ML retrieval deliverable are:

1. Train stronger rerankers beyond the current feature-based logistic/fixed reranker.
2. Expand the held-out evaluation and case-study set when runtime permits.
3. Validate retrieval quality across repeated LeanRank-data refreshes and track metric trends.
4. Harden public API deployment with real auth, rate limiting, TLS, persistent monitoring, and operational logs.
5. Keep full Lean server/session corpus extraction out of the default pipeline unless the project scope explicitly changes away from `erbacher/LeanRank-data`.
