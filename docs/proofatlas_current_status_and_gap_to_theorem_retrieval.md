# ProofAtlas Current Status And Remaining Gap

Date: 2026-06-20

## Current Goal

ProofAtlas is now scoped as an ML ranking/retrieval project built directly on
`erbacher/LeanRank-data`. The final deliverables are:

1. A homepage/demo showing the Lean proof knowledge graph and new theorem proof
   guidance.
2. A quantitative experiment report with held-out proof-state-level and
   theorem-level retrieval metrics.

Local Lean/mathlib source extraction is not part of the default experiment.
The optional Lean check path may parse Lean diagnostics for an input theorem, but
the production data, premise labels, tactic states, positive premises, and hard
negative candidates come from normalized LeanRank-data artifacts.

## Data And Scale

Current production config:

```text
config: configs/proofatlas.yaml
dataset: erbacher/LeanRank-data
source kind: huggingface
source rows considered: 350,000
sampled rows: 292,012
sampled theorem budget: 10,000
split policy: theorem-disjoint train / val / test
demo split: small homepage subset
```

Current processed KG scale:

| Split | Theorems | Proof States | Premises | Nodes | Edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 8,000 | 23,723 | 127,561 | 187,971 | 1,073,855 |
| val | 1,000 | 2,822 | 36,292 | 44,902 | 151,856 |
| test | 1,000 | 3,053 | 38,332 | 47,549 | 159,705 |
| demo | 292 | 900 | 1,416 | 3,952 | 13,802 |

Current LeanRank-data supervision snapshot:

| Signal | Value |
| --- | ---: |
| Positive premise edges | 69,461 |
| Negative candidate edges | 663,198 |
| Negative/positive ratio | 9.5478 |
| Train positive proof-state coverage | 1.0000 |
| Train negative proof-state coverage | 1.0000 |
| Train hard-negative hardness mean | 0.6030 |

## Implemented KG And Retrieval Capabilities

Implemented node types:

```text
Theorem
ProofState
Premise
FileModule
TacticStep
ProofTechnique
```

Implemented edge types:

```text
Theorem -> ProofState: has_proof_state
Theorem -> FileModule: appears_in_file
Theorem -> Premise: invokes_premise
ProofState -> Premise: positive_uses
ProofState -> Premise: negative_candidate
Premise -> FileModule: defined_in_file
ProofState -> TacticStep: at_tactic_step
ProofState -> ProofTechnique: uses_proof_technique
Theorem -> Theorem: similar_to_theorem
Premise -> Premise: co_occurs_with
```

Important current status:

- `invokes_premise` is theorem-level and aggregated from positive
  proof-state premise usage.
- `positive_uses` and `negative_candidate` preserve proof-state-level
  supervision.
- `TacticStep` nodes are scoped by theorem/proof-state evidence instead of
  being shared only by raw tactic index.
- `NewTheoremQuery` and `NewProofStateQuery` support out-of-KG theorem or goal
  text.
- The guidance path can retrieve ranked premises, similar theorems, proof
  techniques, difficulty signals, graph evidence, and ranking explanations.
- Persistent indexes exist for train-side premise, proof-state, and theorem
  embeddings; the production run uses `hnswlib`.
- The FastAPI endpoint and homepage form use `/retrieve-theorem-guidance`.

## Current Held-Out Performance

The current report treats ProofAtlas as a supervised ranking/retrieval system.
Train-side premises are the candidate pool; held-out validation/test positives
are used only as gold labels for scoring.

Proof-state-level premise ranking on full test split:

| Metric | Value |
| --- | ---: |
| Evaluated proof-state queries | 3,053 / 3,053 |
| Retrievable proof-state queries | 2,832 |
| Recall@10 | 0.1162 |
| Recall@100 | 0.2362 |
| MRR | 0.0783 |
| MAP | 0.0494 |
| Ranker AUC | 0.8254 |

Theorem-level premise ranking on full test split:

| Metric | Value |
| --- | ---: |
| Evaluated theorem queries | 1,000 / 1,000 |
| Theorems with train-side gold premises | 955 |
| Recall@10 | 0.4940 |
| Recall@100 | 0.6889 |
| MRR | 0.5609 |
| MAP | 0.3741 |

Interpretation:

- Theorem-level retrieval is already a credible demo result: roughly half of
  held-out theorem queries recover a gold premise in the top 10, and Recall@100
  is close to 0.69.
- Proof-state-level retrieval is harder. Recall@100 at 0.2362 shows candidate
  generation and query representation are still the main bottlenecks for exact
  proof-state premise recovery.
- The failure diagnosis makes the proof-state bottleneck concrete: 1,823
  retrievable proof-state queries miss all gold premises in the top-100
  candidate pool, while 458 have a gold premise after rank 10 and are therefore
  plausible reranking wins.
- The candidate-miss diagnosis classifies proof-state failures as
  `candidate_generation_or_embedding_miss`: 64.37% of retrievable proof-state
  queries miss all train-side gold premises in the top-100 candidate pool,
  16.17% have ordering/reranking headroom after top 10, and 19.46% already
  have a top-10 hit.
- Theorem-level candidate generation is much stronger: 80.31% of retrievable
  theorem queries already have a top-10 hit, 13.93% have ordering/reranking
  headroom after top 10, and only 5.76% miss all train-side gold premises in the
  top-100 candidate pool.
- Reranking helps user-facing examples, but the broad accuracy ceiling is still
  controlled by embedding candidate generation and gold-premise availability in
  the train premise index.

## Performance And Pipeline Status

Current production timing evidence:

| Signal | Value |
| --- | ---: |
| Total timed pipeline run | 499.3380 seconds |
| Executed stages | 20 |
| Skipped stages | 0 |
| Throughput basis | executed_pipeline_run |
| Scale estimate reliable | True |
| Embedding rows/sec during embed stage | 1643.6 |
| Processed rows/sec | 584.8 |
| Slowest stage | embed |
| Evaluation backend | torch_cuda |
| Embedding device count | 7 |
| ANN index backend | hnswlib |

Current acceleration choices:

- Embeddings use `sentence_transformers` with BGE on CUDA.
- Production embedding config records multiple CUDA devices.
- Evaluation uses batched GPU top-k ranking (`torch_cuda`) for full held-out
  proof-state and theorem metrics.
- ANN retrieval uses `hnswlib` for indexed demo/API retrieval and index
  benchmarking.

## Delivered Artifacts

Homepage/demo:

```text
homepage/index.html
homepage/assets/
```

The homepage now shows:

- Knowledge graph visualization.
- New theorem proof guidance examples.
- Retrieval quality metrics.
- Production evidence: held-out coverage, Recall@100, premise supervision,
  pipeline timing, and throughput.
- Reproducibility metadata for `erbacher/LeanRank-data`.

Experiment report and machine-readable results:

```text
outputs/reports/experiment_report.md
outputs/reports/test_set_evaluation.json
outputs/reports/metrics.json
outputs/reports/pipeline_performance_report.json
outputs/reports/pipeline_run_timings.json
outputs/reports/premise_trace_supervision_report.json
```

Primary refresh commands:

```bash
make refresh-production-full-eval
make refresh-production-timing
make verify-delivery
```

## Remaining Gap

The project is no longer blocked on data extraction. The remaining work is
quality and presentation around the ML retrieval task:

1. Improve proof-state-level candidate generation.
   Current proof-state Recall@100 is the weakest headline metric. The next
   useful experiments are query representation ablations, larger candidate
   depth, domain-specific retrieval, and hybrid lexical plus embedding
   retrieval. The new candidate-miss diagnosis shows this is the primary
   accuracy gap: 1,823 retrievable proof-state queries still miss all train-side
   gold premises at top 100.

2. Broaden rerank evaluation.
   The user-facing reranker diagnostic is intentionally small because it follows
   the slower homepage/API path. A stronger report would evaluate reranking on a
   larger proof-state subset while keeping runtime reasonable.

3. Add clearer negative-result analysis.
   The report already includes worst cases. The next step is to categorize
   failures by missing train-side gold premises, domain mismatch, overloaded
   common premises, and embedding candidate miss.

4. Keep homepage and report synchronized.
   Any refreshed production run should regenerate both the experiment report and
   homepage assets so the demo and quantitative claims show the same run.

5. Avoid reintroducing custom extraction scope.
   `erbacher/LeanRank-data` already provides tactic-state rows, positive
   premises, negative candidates, and hardness features. A custom Lean server
   extractor would be a separate research task and should not be part of the
   current deliverable unless the project scope changes explicitly.

## Near-Term Plan

The best next technical step is a fast retrieval-quality pass, not another full
pipeline rewrite:

```text
1. Run targeted proof-state query/candidate-generation ablations.
2. Pick the best cheap improvement that preserves the current pipeline shape.
3. Refresh full held-out evaluation.
4. Regenerate experiment_report.md and homepage/index.html.
5. Commit/push a clean report snapshot.
```

Success should be judged by held-out retrieval metrics, especially:

```text
proof-state Recall@10 / Recall@100
theorem Recall@10 / Recall@100
MRR / MAP / nDCG
gold-premise coverage
runtime and throughput
```
