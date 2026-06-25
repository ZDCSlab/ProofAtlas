.PHONY: help install test id-pipeline evaluate-t1 evaluate-t2 aggregate-guidance report

SPLIT ?= test
NEIGHBOR_K ?= 20
GUIDANCE_LIMIT ?= 25

help:
	@echo "ProofAtlas commands:"
	@echo "  make install             Install package in editable mode"
	@echo "  make id-pipeline         Run T1/T2/T3 focused ID pipeline"
	@echo "  make evaluate-t1         Run proof-state -> premise retrieval"
	@echo "  make evaluate-t2         Run theorem -> theorem pattern retrieval"
	@echo "  make aggregate-guidance  Build similar-theorem guidance bundles"
	@echo "  make report              Build markdown experiment report"
	@echo "  make test                Run tests"

install:
	pip install -e ".[dev]"

id-pipeline:
	proofatlas id-pipeline --split $(SPLIT) --neighbor-k $(NEIGHBOR_K) --guidance-limit $(GUIDANCE_LIMIT)

evaluate-t1:
	proofatlas evaluate-t1 --split $(SPLIT)

evaluate-t2:
	proofatlas evaluate-t2 --split $(SPLIT) --neighbor-k $(NEIGHBOR_K)

aggregate-guidance:
	proofatlas aggregate-guidance --split $(SPLIT) --limit $(GUIDANCE_LIMIT)

report:
	proofatlas build-report --split $(SPLIT)

test:
	pytest -q
