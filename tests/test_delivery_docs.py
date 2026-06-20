from __future__ import annotations

import json
from pathlib import Path


def _readme_metric(readme: str, task: str, metric: str) -> str:
    prefix = f"| {task} | {metric} | "
    for line in readme.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).removesuffix(" |").strip()
    raise AssertionError(f"Missing README metric row: {task} / {metric}")


def test_readme_results_snapshot_matches_committed_metrics() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    metrics = json.loads((repo / "outputs/reports/metrics.json").read_text(encoding="utf-8"))
    expected = {
        ("Proof-state premise retrieval", "Recall@10"): metrics["Recall@10"],
        ("Proof-state premise retrieval", "Recall@100"): metrics["Recall@100"],
        ("Reranked proof-state diagnostic", "Recall@10"): metrics["reranked_proof_state_Recall@10"],
        ("Theorem-level premise retrieval", "Recall@10"): metrics["theorem_retrieval_Recall@10"],
        ("Theorem-level premise retrieval", "Recall@100"): metrics["theorem_retrieval_Recall@100"],
        ("Theorem-level premise retrieval", "MRR"): metrics["theorem_retrieval_MRR"],
        ("Learned premise ranker", "validation AUC"): metrics["AUC"],
    }

    for (task, metric), value in expected.items():
        assert _readme_metric(readme, task, metric) == f"{float(value):.4f}"
