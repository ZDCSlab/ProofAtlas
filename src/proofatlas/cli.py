from __future__ import annotations

import typer
from rich.console import Console

from . import dataset_export, llm_enrichment, pretrained_embeddings, report, t1, t2, t3


app = typer.Typer(help="ProofAtlas focused ID retrieval experiments")
console = Console()


@app.command("evaluate-t1")
def evaluate_t1(
    split: str = "test",
    output_dir: str = "outputs/proofatlas",
    use_llm_enrichment: bool = False,
    use_pretrained_embeddings: bool = False,
    pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    result = t1.run(
        split=split,
        output_dir=output_dir,
        use_llm_enrichment=use_llm_enrichment,
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_model=pretrained_model,
    )
    console.print_json(data=result)


@app.command("evaluate-t2")
def evaluate_t2(
    split: str = "test",
    neighbor_k: int = 20,
    output_dir: str = "outputs/proofatlas",
    use_llm_enrichment: bool = False,
    use_pretrained_embeddings: bool = False,
    pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    result = t2.run(
        split=split,
        neighbor_k=neighbor_k,
        output_dir=output_dir,
        use_llm_enrichment=use_llm_enrichment,
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_model=pretrained_model,
    )
    console.print_json(data=result)


@app.command("aggregate-guidance")
def aggregate_guidance(split: str = "test", limit: int = 25, output_dir: str = "outputs/proofatlas", use_llm_enrichment: bool = False) -> None:
    result = t3.run(split=split, limit=limit, output_dir=output_dir, use_llm_enrichment=use_llm_enrichment)
    console.print_json(data=result)


@app.command("build-report")
def build_report(
    split: str = "test",
    output_dir: str = "outputs/proofatlas",
    use_llm_enrichment: bool = False,
    use_pretrained_embeddings: bool = False,
) -> None:
    path = report.run(
        split=split,
        output_dir=output_dir,
        use_llm_enrichment=use_llm_enrichment,
        use_pretrained_embeddings=use_pretrained_embeddings,
    )
    console.print(str(path))


@app.command("id-pipeline")
def id_pipeline(
    split: str = "test",
    neighbor_k: int = 20,
    guidance_limit: int = 25,
    output_dir: str = "outputs/proofatlas",
    use_llm_enrichment: bool = False,
    use_pretrained_embeddings: bool = False,
    pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    t1.run(
        split=split,
        output_dir=output_dir,
        use_llm_enrichment=use_llm_enrichment,
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_model=pretrained_model,
    )
    t2.run(
        split=split,
        neighbor_k=neighbor_k,
        output_dir=output_dir,
        use_llm_enrichment=use_llm_enrichment,
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_model=pretrained_model,
    )
    t3.run(split=split, limit=guidance_limit, output_dir=output_dir, use_llm_enrichment=use_llm_enrichment)
    path = report.run(
        split=split,
        output_dir=output_dir,
        use_llm_enrichment=use_llm_enrichment,
        use_pretrained_embeddings=use_pretrained_embeddings,
    )
    console.print(f"ProofAtlas ID pipeline complete: {path}")


@app.command("llm-enrich-theorems")
def llm_enrich_theorems(
    split: str = "test",
    limit: int | None = None,
    concurrency: int = 8,
    model: str = "deepseek-chat",
    output_dir: str = "outputs/proofatlas",
    force: bool = False,
    max_states: int = 5,
) -> None:
    result = llm_enrichment.run(
        split=split,
        limit=limit,
        concurrency=concurrency,
        model=model,
        output_dir=output_dir,
        force=force,
        max_states=max_states,
    )
    console.print_json(data=result)


@app.command("llm-pipeline")
def llm_pipeline(
    concurrency: int = 8,
    model: str = "deepseek-chat",
    output_dir: str = "outputs/proofatlas",
    theorem_pilot_limit: int | None = None,
    force: bool = False,
) -> None:
    results = {"theorem_enrichment": []}
    for split in ["train", "val", "test"]:
        results["theorem_enrichment"].append(
            llm_enrichment.run(
                split=split,
                limit=theorem_pilot_limit,
                concurrency=concurrency,
                model=model,
                output_dir=output_dir,
                force=force,
            )
        )
    console.print_json(data=results)


@app.command("export-enriched-dataset")
def export_enriched_dataset(
    version: str = "v1",
    processed_dir: str = "data/processed",
    llm_output_dir: str = "outputs/proofatlas",
    output_root: str = "data/proofatlas_enriched",
    max_states: int = 5,
) -> None:
    result = dataset_export.run(
        version=version,
        processed_dir=processed_dir,
        llm_output_dir=llm_output_dir,
        output_root=output_root,
        max_states=max_states,
    )
    console.print_json(data=result)


@app.command("build-pretrained-embeddings")
def build_pretrained_embeddings(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_dir: str = "data/proofatlas_enriched/v1",
    output_dir: str = "outputs/proofatlas/pretrained_embeddings",
    splits: str = "train,val,test",
    entity_types: str = "premise,proof_state,theorem_profile",
    batch_size: int = 256,
) -> None:
    result = pretrained_embeddings.run(
        model_name=model,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        splits=[part.strip() for part in splits.split(",") if part.strip()],
        entity_types=[part.strip() for part in entity_types.split(",") if part.strip()],
        batch_size=batch_size,
    )
    console.print_json(data=result)


if __name__ == "__main__":
    app()
