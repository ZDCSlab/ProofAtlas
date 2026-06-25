# ProofAtlas LLM Theorem-Enrichment Plan

Date: 2026-06-25

Status: executable plan for DeepSeek theorem semantic + strategy enrichment.

## Objective

This plan keeps the current ProofAtlas train/validation/test split and gold labels fixed. The LLM is used only to enrich theorem profiles, not to inspect held-out positive premise IDs.

The goal is to strengthen the existing retrieval chain:

```text
LLM theorem profiles
-> enriched T2 theorem retrieval
-> enriched similar_theorem_premises
-> T1 weighted RRF
```

The previous batch-100 T1 rerank idea is no longer part of the main plan.

## Call Budget

| Component | Scope | Calls |
| --- | --- | ---: |
| Theorem semantic + strategy enrichment pilot | first 100 train theorems | 100 |
| Full theorem semantic + strategy enrichment | train 8,000 + val 1,000 + test 1,000 theorems | 10,000 |

The pilot is a subset of the train split and is cached. After the successful 100-call pilot, the remaining full run needs:

```text
7,900 train + 1,000 val + 1,000 test = 9,900 additional calls
```

## Stage A: Theorem Semantic + Strategy Enrichment

Each theorem receives one DeepSeek call. The prompt includes:

- theorem ID and full name,
- file path,
- domain and subdomain,
- first proof-state goals,
- local hypotheses summaries,
- symbols.

The prompt does **not** include existing processed strategy labels, proxy difficulty buckets, gold premise IDs, or held-out proof evidence. Strategy and difficulty fields are inferred from visible theorem/proof-state text only.

The model returns strict JSON:

```json
{
  "theorem_id": "thm:...",
  "topic": "short semantic topic",
  "mathematical_objects": ["object or structure"],
  "goal_pattern": "short proof obligation pattern",
  "key_symbols": ["symbol or declaration from the input"],
  "useful_lemma_types": ["abstract lemma type, not invented names"],
  "strategy_facets": [
    {
      "facet": "rewrite_transport",
      "target": "expression or object being transformed",
      "direction_or_action": "what the proof likely needs",
      "confidence": 0.8
    }
  ],
  "likely_tactics": ["simp", "rw", "cases"],
  "difficulty_reasons": ["specific reason"],
  "difficulty_bucket_hint": "easy"
}
```

Outputs:

```text
outputs/proofatlas/llm/theorem_enrichment_train.parquet
outputs/proofatlas/llm/theorem_enrichment_val.parquet
outputs/proofatlas/llm/theorem_enrichment_test.parquet
```

Per-theorem raw API responses are cached under:

```text
outputs/proofatlas/llm/theorem_enrichment_cache/
```

## Stage B: Enriched T2 Theorem Retrieval

The enriched theorem profile is:

```text
original theorem profile
+ LLM topic
+ mathematical_objects
+ goal_pattern
+ key_symbols
+ useful_lemma_types
+ strategy_facets
+ likely_tactics
+ difficulty_reasons
```

Run:

```bash
proofatlas evaluate-t2 --split val --use-llm-enrichment
proofatlas evaluate-t2 --split test --use-llm-enrichment
```

Outputs:

```text
outputs/proofatlas/t2_val_llm_enriched_theorem_neighbors.parquet
outputs/proofatlas/t2_test_llm_enriched_theorem_neighbors.parquet
outputs/proofatlas/t2_val_llm_enriched_theorem_theorem_retrieval.json
outputs/proofatlas/t2_test_llm_enriched_theorem_theorem_retrieval.json
```

Metrics:

- Neighbor premise Recall@50,
- Neighbor premise Recall@100,
- Strategy facet Recall,
- Strategy facet AnyHit,
- Difficulty MAE,
- Difficulty bucket accuracy.

## Stage C: Enriched T2 -> T1 Source

The enriched theorem neighbors provide a new T1 candidate source:

```text
similar_theorem_premises_llm_enriched
```

This source is fused with the existing sources:

- dense proof-state -> premise,
- lexical proof-state -> premise,
- symbol overlap,
- similar proof-state expansion,
- original theorem-premise source,
- LLM-enriched theorem-premise source.

Run:

```bash
proofatlas evaluate-t1 --split val --use-llm-enrichment
proofatlas evaluate-t1 --split test --use-llm-enrichment
```

Outputs:

```text
outputs/proofatlas/t1_val_llm_enriched_proof_state_premise_retrieval.json
outputs/proofatlas/t1_test_llm_enriched_proof_state_premise_retrieval.json
outputs/proofatlas/t1_val_llm_enriched_candidate_rankings.parquet
outputs/proofatlas/t1_test_llm_enriched_candidate_rankings.parquet
```

New T1 methods:

```text
similar_theorem_premises_llm_enriched
weighted_rrf_llm_theorem_source
weighted_rrf_llm_theorem_tuned
```

Primary T1 metrics:

- Recall@10,
- Recall@50,
- Recall@100,
- MAP,
- nDCG@10,
- Gold coverage.

## Stage D: Enriched Guidance And Report

After enriched T2/T1, run:

```bash
proofatlas aggregate-guidance --split test --use-llm-enrichment
proofatlas build-report --split test --use-llm-enrichment
```

or:

```bash
proofatlas id-pipeline --split test --use-llm-enrichment
```

The report is written to:

```text
outputs/proofatlas/id_experiment_report_test_llm_enriched.md
```

## Execution Order

1. Confirm `.env` contains `DEEPSEEK_API_KEY`.
2. Run the 100-theorem pilot:

```bash
proofatlas llm-enrich-theorems --split train --limit 100 --concurrency 16
```

3. Run full theorem enrichment:

```bash
proofatlas llm-enrich-theorems --split train --concurrency 16
proofatlas llm-enrich-theorems --split val --concurrency 16
proofatlas llm-enrich-theorems --split test --concurrency 16
```

4. Run validation enriched retrieval:

```bash
proofatlas evaluate-t2 --split val --use-llm-enrichment
proofatlas evaluate-t1 --split val --use-llm-enrichment
```

5. Run final test enriched pipeline:

```bash
proofatlas id-pipeline --split test --use-llm-enrichment
```

## Risk Controls

- Cache every theorem API response.
- Do not use `--force` unless intentionally rerunning paid calls.
- Tune only on validation.
- Run test once after the enrichment profile format and source weights are fixed.
- Keep non-LLM baselines in the report.
- Do not expose held-out positive premise IDs to the LLM.
- Do not allow the LLM to invent concrete Lean declaration names; invented needs must be represented as abstract lemma types.

## Current Status

The 100-theorem pilot already succeeded:

```text
requested: 100
succeeded: 100
failed: 0
```

The next step is the remaining theorem enrichment calls:

```text
7,900 train + 1,000 val + 1,000 test = 9,900 calls
```
