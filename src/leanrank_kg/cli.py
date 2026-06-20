from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import audit, augment_graph, build_graph, compute_difficulty, download_or_sample, embed, evaluate, homepage, normalize, report, train_ranker, validate, weak_label_proof_technique
from .retrieve import (
    explain_premise_match,
    get_difficulty_profile,
    get_graph_neighborhood,
    get_proof_technique_labels,
    retrieve_premises,
    retrieve_similar_theorems,
)

app = typer.Typer(help="LeanRank proof knowledge graph MVP")
console = Console()


@app.command()
def sample(config: str = "configs/sample.yaml", debug_rows: int | None = None) -> None:
    download_or_sample.run(config, debug_rows)


@app.command()
def process(config: str = "configs/sample.yaml") -> None:
    normalize.run(config)


@app.command("build-graph")
def build_graph_cmd(config: str = "configs/sample.yaml") -> None:
    build_graph.run(config)


@app.command("label-techniques")
def label_techniques(config: str = "configs/sample.yaml") -> None:
    weak_label_proof_technique.run(config)


@app.command("compute-difficulty")
def difficulty(config: str = "configs/sample.yaml") -> None:
    compute_difficulty.run(config)


@app.command("embed")
def embed_cmd(config: str = "configs/sample.yaml") -> None:
    embed.run(config)


@app.command("augment-graph")
def augment_graph_cmd(config: str = "configs/sample.yaml") -> None:
    augment_graph.run(config)


@app.command("train-ranker")
def train_ranker_cmd(config: str = "configs/sample.yaml") -> None:
    train_ranker.run(config)


@app.command("evaluate")
def evaluate_cmd(config: str = "configs/sample.yaml") -> None:
    evaluate.run(config)


@app.command("build-homepage")
def build_homepage(config: str = "configs/sample.yaml") -> None:
    homepage.run(config)


@app.command("build-report")
def build_report(config: str = "configs/sample.yaml") -> None:
    report.run(config)


@app.command("validate")
def validate_cmd(config: str = "configs/sample.yaml") -> None:
    validate.run(config)


@app.command("audit")
def audit_cmd(config: str = "configs/sample.yaml") -> None:
    result = audit.build_audit()
    console.print(result)


@app.command("full-pipeline")
def full_pipeline(config: str = "configs/sample.yaml", debug_rows: int | None = None) -> None:
    download_or_sample.run(config, debug_rows)
    normalize.run(config)
    build_graph.run(config)
    weak_label_proof_technique.run(config)
    compute_difficulty.run(config)
    embed.run(config)
    augment_graph.run(config)
    train_ranker.run(config)
    evaluate.run(config)
    validate.run(config)
    homepage.run(config)
    audit.run(config)
    console.print("Full pipeline complete. Open homepage/index.html for the static demo.")


@app.command("retrieve-premises")
def retrieve_premises_cmd(
    proof_state_id: Annotated[str, typer.Option("--proof-state-id")],
    k: int = 10,
    split: str = "train",
    index_split: str = "train",
) -> None:
    table = Table("Premise", "Score", "Explanation")
    for row in retrieve_premises(proof_state_id, k, split, index_split):
        table.add_row(row["full_name"], f"{row['score']:.3f}", row["explanation"])
    console.print(table)


@app.command("similar-theorems")
def similar_theorems_cmd(theorem_id: Annotated[str, typer.Option("--theorem-id")], k: int = 10, split: str = "train") -> None:
    table = Table("Theorem", "Score", "Explanation")
    for row in retrieve_similar_theorems(theorem_id, k, split):
        table.add_row(row["full_name"], f"{row['score']:.3f}", row["explanation"])
    console.print(table)


@app.command("explain-premise-match")
def explain_premise_match_cmd(
    proof_state_id: Annotated[str, typer.Option("--proof-state-id")],
    premise_id: Annotated[str, typer.Option("--premise-id")],
    split: str = "train",
    index_split: str = "train",
) -> None:
    console.print(explain_premise_match(proof_state_id, premise_id, split=split, index_split=index_split))


@app.command("show-difficulty")
def show_difficulty(entity_id: Annotated[str, typer.Option("--entity-id")], split: str = "train") -> None:
    console.print(get_difficulty_profile(entity_id, split))


@app.command("show-techniques")
def show_techniques(proof_state_id: Annotated[str, typer.Option("--proof-state-id")], split: str = "train") -> None:
    console.print(get_proof_technique_labels(proof_state_id, split))


@app.command("show-neighborhood")
def show_neighborhood(entity_id: Annotated[str, typer.Option("--entity-id")], depth: int = 1, split: str = "train") -> None:
    console.print(get_graph_neighborhood(entity_id, depth, split))


if __name__ == "__main__":
    app()
