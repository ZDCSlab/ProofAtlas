from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import audit, augment_graph, benchmark_index, build_graph, build_index, compute_difficulty, deployment_security, download_or_sample, embed, evaluate, experiment_report, homepage, lean_diagnostic_report, normalize, pipeline_profile, premise_trace_supervision, report, research_report, train_difficulty, train_ranker, validate, weak_label_proof_technique
from .pipeline_timing import PipelineTimer
from .retrieve import (
    explain_premise_match,
    get_difficulty_profile,
    get_graph_neighborhood,
    get_proof_technique_labels,
    retrieve_knowledge_for_theorem,
    retrieve_premises,
    retrieve_premises_for_query,
    retrieve_similar_proof_states_for_query,
    retrieve_similar_theorems,
    retrieve_similar_theorems_for_query,
)

app = typer.Typer(help="LeanRank proof knowledge graph MVP")
console = Console()


SPLIT_NAMES = ["train", "val", "test", "demo"]


def _all_exist(paths: list[str]) -> bool:
    return all(Path(path).exists() for path in paths)


def _stage_artifacts() -> dict[str, list[str]]:
    processed_tables = ["theorems", "proof_states", "premises", "file_modules", "positive_edges", "negative_edges"]
    technique_tables = ["proof_state_techniques", "premise_techniques", "proof_techniques"]
    return {
        "sample": [
            "data/sample/train_rows.parquet",
            "data/sample/val_rows.parquet",
            "data/sample/test_rows.parquet",
            "outputs/reports/corpus_manifest.json",
        ],
        "normalize": [f"data/processed/{split}/{table}.parquet" for split in SPLIT_NAMES for table in processed_tables],
        "build_graph": [f"outputs/graph/{split}/{name}.parquet" for split in SPLIT_NAMES for name in ["nodes", "edges"]],
        "label_techniques": [f"data/processed/{split}/{table}.parquet" for split in SPLIT_NAMES for table in technique_tables],
        "compute_difficulty": [f"data/processed/{split}/{table}.parquet" for split in SPLIT_NAMES for table in ["proof_state_features", "theorem_features"]],
        "premise_trace_supervision": ["outputs/reports/premise_trace_supervision_report.json"],
        "lean_diagnostic_extraction": ["outputs/reports/lean_diagnostic_extraction_report.json"],
        "train_difficulty": ["outputs/models/difficulty_estimator.joblib", "outputs/reports/difficulty_estimator_metrics.json"],
        "embed": [
            *[f"outputs/embeddings/{split}_{kind}_embeddings.npz" for split in SPLIT_NAMES for kind in ["proof_state", "premise", "theorem"]],
            *[f"outputs/embeddings/{split}_embedding_metadata.parquet" for split in SPLIT_NAMES],
            "outputs/embeddings/embedding_config.json",
        ],
        "build_index": [
            "outputs/indexes/index_summary.json",
            *[f"outputs/indexes/{split}_{kind}_index_manifest.json" for split in SPLIT_NAMES for kind in ["proof_state", "premise", "theorem"]],
        ],
        "benchmark_index": ["outputs/reports/index_benchmark.json"],
        "augment_graph": [f"outputs/graph/{split}/{name}_enriched.parquet" for split in SPLIT_NAMES for name in ["nodes", "edges"]],
        "train_ranker": ["outputs/models/premise_ranker.joblib", "outputs/reports/ranker_validation_metrics.json"],
        "evaluate": ["outputs/reports/test_set_evaluation.json", "outputs/reports/metrics.json", "outputs/reports/theorem_retrieval_case_studies.json"],
        "validate": ["outputs/reports/schema_validation_summary.json", "outputs/reports/graph_validation_summary.json", "outputs/reports/artifact_compatibility_report.json"],
        "homepage": ["homepage/index.html", "outputs/reports/homepage_summary.json"],
        "deployment_security": ["outputs/reports/deployment_security_review.json"],
        "pipeline_profile": ["outputs/reports/pipeline_performance_report.json"],
        "experiment_report": ["outputs/reports/experiment_report.md"],
        "research_report": ["outputs/reports/research_report.md", "outputs/predictions/research_prediction_results.json"],
        "audit": ["outputs/reports/mvp_completion_audit.json"],
    }


def _force_stage_set(force_stages: str) -> set[str]:
    return {stage.strip() for stage in force_stages.split(",") if stage.strip()}


def _run_or_skip(timer: PipelineTimer, name: str, fn, *args, force: bool = False, force_stages: set[str] | None = None) -> None:
    artifacts = _stage_artifacts().get(name, [])
    if not force and name not in (force_stages or set()) and artifacts and _all_exist(artifacts):
        timer.skip_stage(name)
        return
    timer.run_stage(name, fn, *args)


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


@app.command("build-index")
def build_index_cmd(config: str = "configs/sample.yaml") -> None:
    build_index.run(config)


@app.command("benchmark-index")
def benchmark_index_cmd(config: str = "configs/sample.yaml", split: str | None = None, top_k: int | None = None, query_count: int | None = None, seed: int | None = None) -> None:
    result = benchmark_index.run(config, split=split, top_k=top_k, query_count=query_count, seed=seed)
    console.print_json(data=result)


@app.command("security-review")
def security_review_cmd(
    host: str = "127.0.0.1",
    require_ready: bool = False,
    startup_index_split: str = "train",
    reload: bool = False,
    public_exposure: bool = False,
    output_path: str = "outputs/reports/deployment_security_review.json",
) -> None:
    result = deployment_security.run(
        host=host,
        require_ready=require_ready,
        startup_index_split=startup_index_split,
        reload=reload,
        public_exposure=public_exposure,
        output_path=output_path,
    )
    console.print_json(data=result)


@app.command("profile-pipeline")
def profile_pipeline_cmd(
    config: str = "configs/proofatlas.yaml",
    output_path: str = "outputs/reports/pipeline_performance_report.json",
) -> None:
    result = pipeline_profile.run(config, output_path=output_path)
    console.print_json(data=result)


@app.command("premise-trace-supervision-report")
def premise_trace_supervision_report_cmd(
    output_path: str = "outputs/reports/premise_trace_supervision_report.json",
) -> None:
    result = premise_trace_supervision.run(output_path=output_path)
    console.print_json(data=result)


@app.command("lean-diagnostic-extraction-report")
def lean_diagnostic_extraction_report_cmd(
    output_path: str = "outputs/reports/lean_diagnostic_extraction_report.json",
) -> None:
    result = lean_diagnostic_report.run(output_path=output_path)
    console.print_json(data=result)


@app.command("augment-graph")
def augment_graph_cmd(config: str = "configs/sample.yaml") -> None:
    augment_graph.run(config)


@app.command("train-ranker")
def train_ranker_cmd(config: str = "configs/sample.yaml") -> None:
    train_ranker.run(config)


@app.command("train-difficulty")
def train_difficulty_cmd(config: str = "configs/sample.yaml") -> None:
    train_difficulty.run(config)


@app.command("evaluate")
def evaluate_cmd(config: str = "configs/sample.yaml", full_heldout: bool = False) -> None:
    evaluate.run(config, full_heldout=full_heldout)


@app.command("build-homepage")
def build_homepage(config: str = "configs/sample.yaml") -> None:
    homepage.run(config)


@app.command("build-report")
def build_report(config: str = "configs/sample.yaml") -> None:
    report.run(config)


@app.command("build-experiment-report")
def build_experiment_report_cmd(
    config: str = "configs/proofatlas.yaml",
    output_path: str = "outputs/reports/experiment_report.md",
) -> None:
    result = experiment_report.run(config, output_path=output_path)
    console.print_json(data=result)


@app.command("build-research-report")
def build_research_report_cmd(
    config: str = "configs/proofatlas.yaml",
    output_path: str = "outputs/reports/research_report.md",
) -> None:
    result = research_report.run(config, output_path=output_path)
    console.print_json(data=result)


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, require_ready: bool = False, startup_index_split: str = "train") -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("Install API dependencies first: pip install -e '.[api]'") from exc
    os.environ["PROOFATLAS_REQUIRE_READY"] = "1" if require_ready else os.getenv("PROOFATLAS_REQUIRE_READY", "0")
    os.environ["PROOFATLAS_STARTUP_INDEX_SPLIT"] = startup_index_split
    uvicorn.run("leanrank_kg.api:create_app", factory=True, host=host, port=port, reload=reload)


@app.command("validate")
def validate_cmd(config: str = "configs/sample.yaml") -> None:
    validate.run(config)


@app.command("audit")
def audit_cmd(config: str = "configs/sample.yaml") -> None:
    result = audit.build_audit()
    console.print(result)


@app.command("full-pipeline")
def full_pipeline(
    config: str = "configs/sample.yaml",
    debug_rows: int | None = None,
    force: bool = False,
    force_stages: str = "",
) -> None:
    timer = PipelineTimer(config)
    forced = _force_stage_set(force_stages)
    try:
        _run_or_skip(timer, "sample", download_or_sample.run, config, debug_rows, force=force, force_stages=forced)
        _run_or_skip(timer, "normalize", normalize.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "build_graph", build_graph.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "label_techniques", weak_label_proof_technique.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "compute_difficulty", compute_difficulty.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "premise_trace_supervision", premise_trace_supervision.run, force=force, force_stages=forced)
        _run_or_skip(timer, "lean_diagnostic_extraction", lean_diagnostic_report.run, force=force, force_stages=forced)
        _run_or_skip(timer, "train_difficulty", train_difficulty.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "embed", embed.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "build_index", build_index.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "benchmark_index", benchmark_index.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "augment_graph", augment_graph.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "train_ranker", train_ranker.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "evaluate", evaluate.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "validate", validate.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "homepage", homepage.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "deployment_security", deployment_security.run, force=force, force_stages=forced)
        timer.write()
        _run_or_skip(timer, "pipeline_profile", pipeline_profile.run, config, force=force, force_stages=forced)
        timer.write()
        _run_or_skip(timer, "experiment_report", experiment_report.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "research_report", research_report.run, config, force=force, force_stages=forced)
        _run_or_skip(timer, "audit", audit.run, config, force=force, force_stages=forced)
    finally:
        timer.write()
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


@app.command("retrieve-premises-for-query")
def retrieve_premises_for_query_cmd(
    query_text: Annotated[str, typer.Option("--query-text")],
    k: int = 10,
    index_split: str = "train",
) -> None:
    table = Table("Premise", "Score", "Signals")
    for row in retrieve_premises_for_query(query_text, k=k, index_split=index_split):
        signals = row.get("signals", {})
        table.add_row(row["full_name"], f"{row['score']:.3f}", f"domain={signals.get('domain_hint', '')}; tokens={signals.get('shared_name_tokens', [])}")
    console.print(table)


@app.command("similar-theorems")
def similar_theorems_cmd(theorem_id: Annotated[str, typer.Option("--theorem-id")], k: int = 10, split: str = "train") -> None:
    table = Table("Theorem", "Score", "Explanation")
    for row in retrieve_similar_theorems(theorem_id, k, split):
        table.add_row(row["full_name"], f"{row['score']:.3f}", row["explanation"])
    console.print(table)


@app.command("similar-theorems-for-query")
def similar_theorems_for_query_cmd(
    query_text: Annotated[str, typer.Option("--query-text")],
    k: int = 10,
    index_split: str = "train",
) -> None:
    table = Table("Theorem", "Score", "Explanation")
    for row in retrieve_similar_theorems_for_query(query_text, k=k, index_split=index_split):
        table.add_row(row["full_name"], f"{row['score']:.3f}", row["explanation"])
    console.print(table)


@app.command("similar-proof-states-for-query")
def similar_proof_states_for_query_cmd(
    query_text: Annotated[str, typer.Option("--query-text")],
    k: int = 10,
    index_split: str = "train",
) -> None:
    table = Table("ProofState", "Theorem", "Score", "Goal")
    for row in retrieve_similar_proof_states_for_query(query_text, k=k, index_split=index_split):
        table.add_row(row["proof_state_id"], row["full_name"], f"{row['score']:.3f}", str(row.get("goal_text", ""))[:90])
    console.print(table)


@app.command("retrieve-theorem-guidance")
def retrieve_theorem_guidance_cmd(
    theorem_text: Annotated[str | None, typer.Option("--theorem-text")] = None,
    query_file: Annotated[Path | None, typer.Option("--query-file", exists=True, dir_okay=False)] = None,
    full_name: Annotated[str | None, typer.Option("--full-name")] = None,
    input_type: Annotated[str, typer.Option("--input-type")] = "lean",
    domain_hint: Annotated[str | None, typer.Option("--domain-hint")] = None,
    file_path: Annotated[str | None, typer.Option("--file-path")] = None,
    k_premises: int = 20,
    k_theorems: int = 10,
    index_split: str = "train",
    validate_lean: bool = False,
) -> None:
    if query_file is not None:
        theorem_text = query_file.read_text(encoding="utf-8")
    if not theorem_text:
        raise typer.BadParameter("Provide --theorem-text or --query-file.")
    result = retrieve_knowledge_for_theorem(
        theorem_text=theorem_text,
        full_name=full_name,
        input_type=input_type,  # type: ignore[arg-type]
        domain_hint=domain_hint,
        file_path=file_path,
        k_premises=k_premises,
        k_theorems=k_theorems,
        index_split=index_split,
        validate_lean=validate_lean,
    )
    console.print_json(data=result)


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
