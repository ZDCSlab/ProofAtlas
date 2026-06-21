# ProofAtlas Deployment Guide

Date: 2026-06-20

Status: engineering extra. The current project is framed as a research dataset plus retrieval report; `homepage/index.html` is a static visual report. The API/server material below is kept for local inspection and reproducibility, not as the central research deliverable.

This guide describes reproducible ways to run the ProofAtlas demo after the pipeline has generated artifacts. It covers static homepage review, local interactive API review, server deployment, and notebook use.

## Prerequisites

Use the project conda environment:

```bash
conda activate leanrank_kg
pip install -e ".[dev]"
```

For the optional API server:

```bash
pip install -e ".[api]"
```

For Hugging Face sentence-transformer embeddings:

```bash
pip install -e ".[hf]"
```

For optional hnswlib ANN indexes:

```bash
pip install -e ".[ann]"
```

For optional FAISS indexes:

```bash
pip install -e ".[faiss]"
```

## Build Artifacts

Small local demo:

```bash
leanrank-kg full-pipeline --config configs/sample.yaml --debug-rows 120
```

Full ProofAtlas-style run:

```bash
leanrank-kg full-pipeline --config configs/proofatlas.yaml
```

The full pipeline writes the core deployable artifacts:

- `homepage/index.html`
- `homepage/assets/*.json`
- `outputs/reports/metrics.json`
- `outputs/reports/refresh_dashboard.json`
- `outputs/reports/refresh_trend.json`
- `outputs/reports/refresh_history.json`
- `outputs/reports/artifact_compatibility_report.json`
- `outputs/indexes/*_index_manifest.json`
- `outputs/models/*.joblib`

## Static Homepage Review

Use this mode for GitHub Pages, local file review, and non-interactive demos.

```bash
leanrank-kg build-homepage --config configs/sample.yaml
```

Open:

```text
homepage/index.html
```

The static page uses only generated assets under `homepage/assets/`. It does not require a running API server. The live theorem-guidance form will only work when the optional API server is running.

## Local Interactive API Demo

Use this mode when reviewers should type a theorem or goal and retrieve live proof guidance.

```bash
leanrank-kg serve --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

Useful endpoints:

```text
GET  /health
GET  /metrics
GET  /metrics/prometheus
POST /retrieve-theorem-guidance
```

Example:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

```bash
curl -s http://127.0.0.1:8000/retrieve-theorem-guidance \
  -H 'Content-Type: application/json' \
  -d '{"theorem_text":"theorem Nat.self_eq (x : Nat) : x = x := by simpa","input_type":"theorem","k_premises":10,"k_theorems":5,"index_split":"train","validate_lean":false}' \
  | python -m json.tool
```

## Readiness-Gated Server Mode

Use readiness gating when deploying a server where missing artifacts should fail startup.

```bash
leanrank-kg serve \
  --host 0.0.0.0 \
  --port 8000 \
  --require-ready \
  --startup-index-split train
```

Equivalent environment variables:

```bash
export PROOFATLAS_REQUIRE_READY=1
export PROOFATLAS_STARTUP_INDEX_SPLIT=train
leanrank-kg serve --host 0.0.0.0 --port 8000
```

Before exposing the service, check:

```bash
leanrank-kg validate --config configs/proofatlas.yaml
leanrank-kg security-review --host 127.0.0.1
leanrank-kg audit --config configs/proofatlas.yaml
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

The service is still a research demo. Put it behind trusted network controls before any public deployment.

## Security Review Gate

Generate the deployment security/readiness report before sharing an interactive API demo:

```bash
leanrank-kg security-review \
  --host 127.0.0.1 \
  --startup-index-split train
```

The report is written to:

```text
outputs/reports/deployment_security_review.json
```

For public exposure, run the review in public mode after adding external controls:

```bash
export PROOFATLAS_ACCESS_CONTROL_CONFIRMED=1
export PROOFATLAS_NETWORK_CONTROLS_CONFIRMED=1
export PROOFATLAS_MONITORING_CONFIRMED=1
leanrank-kg security-review \
  --host 0.0.0.0 \
  --require-ready \
  --startup-index-split train \
  --public-exposure
```

Public mode fails unless readiness-gated startup is enabled and access control, network restriction, and monitoring have been explicitly confirmed. These environment flags are deployment assertions; they do not implement authentication, rate limiting, firewall policy, TLS, or persistent monitoring by themselves.

## Notebook Demo

Use notebook mode for walkthroughs where users inspect artifacts and call Python APIs directly.

Recommended sequence:

```bash
leanrank-kg full-pipeline --config configs/sample.yaml --debug-rows 120
jupyter notebook notebooks/leanrank_kg_demo.ipynb
```

The notebook should read generated reports from `outputs/reports/` and call retrieval APIs from `leanrank_kg.retrieve`. It should not require a separate API server unless the walkthrough explicitly tests HTTP endpoints.

## GitHub Pages / Hosted Static Demo

Use this mode for a public, non-interactive research page.

Build locally:

```bash
leanrank-kg full-pipeline --config configs/proofatlas.yaml
leanrank-kg build-homepage --config configs/proofatlas.yaml
```

Publish:

```text
homepage/index.html
homepage/assets/
```

The published page should be treated as a snapshot of generated artifacts. Rebuild and republish after data refreshes, index rebuilds, or model retraining.

## Refresh Checklist

After each LeanRank/mathlib refresh:

1. Run the full pipeline.
2. Run `leanrank-kg validate --config <config>`.
3. Confirm `outputs/reports/artifact_compatibility_report.json` has `"passed": true`.
4. Confirm `outputs/reports/refresh_dashboard.json` has the expected quality gates.
5. Review `outputs/reports/refresh_trend.json` and `outputs/reports/refresh_history.json` for scale, retrieval, parsing, difficulty, index, and quality-gate changes across refreshes.
6. Run `leanrank-kg audit --config <config>`.
7. Run `leanrank-kg security-review --host 127.0.0.1`.
8. Rebuild and republish `homepage/`.

The refresh manifest records LeanRank-data and config provenance in `outputs/reports/corpus_manifest.json`. Set `corpus.corpus_version`, `corpus.lean_version`, `corpus.mathlib_commit`, and `corpus.source_revision` in the config for explicit values when needed.

## Operational Notes

- `GET /metrics` is in-process only. It resets when the server restarts.
- `GET /metrics/prometheus` exposes the in-process counters and latency gauges as Prometheus-compatible text for local scraping or deployment integration.
- `GET /health` checks required artifacts for a selected split.
- Retrieval artifacts are cached with file timestamp signatures and refresh after pipeline outputs change.
- `--validate-lean` uses `lake env lean` or `lean` when available, and should be treated as optional in demos.
- Public exposure requires an additional security review, network restrictions, and deployment-specific logging/monitoring.
- `leanrank-kg security-review` writes `outputs/reports/deployment_security_review.json`; public API exposure requires explicit external access-control, network-control, and monitoring confirmation.
