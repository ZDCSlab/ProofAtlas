.PHONY: help install sample process build-graph label difficulty premise-trace-supervision train-difficulty embed build-index benchmark-index profile-pipeline security-review augment-graph train-ranker evaluate validate report experiment-report homepage audit demo test smoke clean

CONFIG ?= configs/sample.yaml
SECURITY_REVIEW_HOST ?= 127.0.0.1

help:
	@echo "Main commands:"
	@echo "  make install        Install package in editable mode"
	@echo "  make sample         Build deterministic raw sample"
	@echo "  make process        Normalize sample rows"
	@echo "  make evaluate       Run evaluation reports"
	@echo "  make premise-trace-supervision Summarize positive/negative premise supervision"
	@echo "  make train-difficulty Train proof-state difficulty estimator"
	@echo "  make build-index    Build local nearest-neighbor retrieval indexes"
	@echo "  make benchmark-index Benchmark saved index latency and exact-overlap"
	@echo "  make profile-pipeline Summarize pipeline scale, performance, and blockers"
	@echo "  make security-review Generate deployment security/readiness review"
	@echo "  make validate       Run schema, split, and graph validation reports"
	@echo "  make report         Build homepage summary report"
	@echo "  make experiment-report Build held-out test-set experiment report"
	@echo "  make homepage       Generate static homepage"
	@echo "  make audit          Run MVP completion audit"
	@echo "  make demo           Run full demo pipeline"
	@echo "  make test           Run unit tests"
	@echo "  make smoke          Run tiny end-to-end smoke test"

install:
	pip install -e ".[dev]"

sample:
	leanrank-kg sample --config $(CONFIG)

process:
	leanrank-kg process --config $(CONFIG)

build-graph:
	leanrank-kg build-graph --config $(CONFIG)

label:
	leanrank-kg label-techniques --config $(CONFIG)

difficulty:
	leanrank-kg compute-difficulty --config $(CONFIG)

premise-trace-supervision:
	leanrank-kg premise-trace-supervision-report

train-difficulty:
	leanrank-kg train-difficulty --config $(CONFIG)

embed:
	leanrank-kg embed --config $(CONFIG)

build-index:
	leanrank-kg build-index --config $(CONFIG)

benchmark-index:
	leanrank-kg benchmark-index --config $(CONFIG)

profile-pipeline:
	leanrank-kg profile-pipeline --config $(CONFIG)

security-review:
	leanrank-kg security-review --host $(SECURITY_REVIEW_HOST)

augment-graph:
	leanrank-kg augment-graph --config $(CONFIG)

train-ranker:
	leanrank-kg train-ranker --config $(CONFIG)

evaluate:
	leanrank-kg evaluate --config $(CONFIG)

validate:
	leanrank-kg validate --config $(CONFIG)

report:
	leanrank-kg build-report --config $(CONFIG)

experiment-report:
	leanrank-kg build-experiment-report --config $(CONFIG)

homepage:
	leanrank-kg build-homepage --config $(CONFIG)

audit:
	leanrank-kg audit --config $(CONFIG)

demo:
	leanrank-kg full-pipeline --config $(CONFIG)

test:
	pytest

smoke:
	leanrank-kg full-pipeline --config $(CONFIG) --debug-rows 120
	pytest

clean:
	rm -rf data/sample data/processed outputs/graph outputs/embeddings outputs/indexes outputs/models outputs/reports homepage/assets homepage/index.html
