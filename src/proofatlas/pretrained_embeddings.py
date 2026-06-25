from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import SPLITS, write_json
from .profiles import premise_profile, proof_state_profile


ENTITY_TYPES = ["premise", "proof_state", "theorem_profile"]


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip())
    return slug.strip("_") or "model"


def embedding_dir(output_dir: str, model_name: str) -> Path:
    return Path(output_dir) / model_slug(model_name)


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for pretrained embedding experiments.") from exc
    return SentenceTransformer(model_name)


def _texts_and_ids(dataset_dir: Path, split: str, entity_type: str) -> tuple[list[str], list[str]]:
    split_dir = dataset_dir / split
    if entity_type == "premise":
        df = pd.read_parquet(split_dir / "premises.parquet")
        df["id"] = df["id"].astype(str)
        return df["id"].tolist(), [premise_profile(row) for row in df.to_dict(orient="records")]
    if entity_type == "proof_state":
        df = pd.read_parquet(split_dir / "proof_states.parquet")
        df["id"] = df["id"].astype(str)
        return df["id"].tolist(), [proof_state_profile(row) for row in df.to_dict(orient="records")]
    if entity_type == "theorem_profile":
        df = pd.read_parquet(split_dir / "theorem_profiles.parquet")
        df["theorem_id"] = df["theorem_id"].astype(str)
        return df["theorem_id"].tolist(), df["profile_text"].fillna("").astype(str).tolist()
    raise ValueError(f"Unknown entity type: {entity_type}")


def _write_entity_embeddings(
    *,
    model,
    model_name: str,
    dataset_dir: Path,
    output_root: Path,
    split: str,
    entity_type: str,
    batch_size: int,
) -> dict[str, Any]:
    ids, texts = _texts_and_ids(dataset_dir, split, entity_type)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    path_prefix = output_root / f"{split}_{entity_type}"
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_prefix.with_suffix(".npy"), embeddings)
    pd.DataFrame({"entity_id": ids, "row_index": range(len(ids))}).to_parquet(path_prefix.with_name(path_prefix.name + "_metadata.parquet"), index=False)
    return {
        "split": split,
        "entity_type": entity_type,
        "rows": len(ids),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "embedding_path": str(path_prefix.with_suffix(".npy")),
        "metadata_path": str(path_prefix.with_name(path_prefix.name + "_metadata.parquet")),
        "model": model_name,
    }


def load_embeddings(
    *,
    split: str,
    entity_type: str,
    model_name: str,
    output_dir: str = "outputs/proofatlas/pretrained_embeddings",
) -> tuple[list[str], np.ndarray]:
    root = embedding_dir(output_dir, model_name)
    prefix = root / f"{split}_{entity_type}"
    emb_path = prefix.with_suffix(".npy")
    meta_path = prefix.with_name(prefix.name + "_metadata.parquet")
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing pretrained embeddings for {split}/{entity_type}/{model_name}: {emb_path}")
    metadata = pd.read_parquet(meta_path).sort_values("row_index")
    embeddings = np.load(emb_path, mmap_mode="r")
    return metadata["entity_id"].astype(str).tolist(), embeddings


def run(
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_dir: str = "data/proofatlas_enriched/v1",
    output_dir: str = "outputs/proofatlas/pretrained_embeddings",
    splits: list[str] | None = None,
    entity_types: list[str] | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    selected_splits = splits or SPLITS
    selected_entities = entity_types or ENTITY_TYPES
    invalid_splits = sorted(set(selected_splits) - set(SPLITS))
    invalid_entities = sorted(set(selected_entities) - set(ENTITY_TYPES))
    if invalid_splits:
        raise ValueError(f"Unknown splits: {invalid_splits}")
    if invalid_entities:
        raise ValueError(f"Unknown entity types: {invalid_entities}")

    dataset_path = Path(dataset_dir)
    out = embedding_dir(output_dir, model_name)
    model = _load_sentence_transformer(model_name)
    results = []
    for split in selected_splits:
        for entity_type in selected_entities:
            results.append(
                _write_entity_embeddings(
                    model=model,
                    model_name=model_name,
                    dataset_dir=dataset_path,
                    output_root=out,
                    split=split,
                    entity_type=entity_type,
                    batch_size=batch_size,
                )
            )
    manifest = {
        "model": model_name,
        "model_slug": model_slug(model_name),
        "dataset_dir": str(dataset_path),
        "output_dir": str(out),
        "splits": selected_splits,
        "entity_types": selected_entities,
        "artifacts": results,
    }
    write_json(out / "manifest.json", manifest)
    return manifest
