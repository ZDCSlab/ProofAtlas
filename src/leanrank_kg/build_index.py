from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .utils import SPLITS, load_config, read_json, stable_hash, write_json, write_parquet

ENTITY_STEMS = {
    "ProofState": "proof_state",
    "Premise": "premise",
    "Theorem": "theorem",
}


def _embedding_config_hash(embedding_config: dict[str, Any]) -> str:
    return stable_hash(json.dumps(embedding_config, sort_keys=True), 16)


def _build_neighbors(matrix, n_neighbors: int, metric: str = "cosine") -> NearestNeighbors:
    index = NearestNeighbors(n_neighbors=max(1, min(n_neighbors, matrix.shape[0])), metric=metric, algorithm="brute")
    index.fit(matrix)
    return index


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        try:
            import hnswlib  # noqa: F401

            return "hnswlib"
        except ImportError:
            return "sklearn"
    return backend


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _write_sklearn_index(index_path: str, matrix, n_neighbors: int, metric: str) -> dict[str, Any]:
    index = _build_neighbors(matrix, n_neighbors=n_neighbors, metric=metric)
    joblib.dump(index, index_path)
    return {"index_path": index_path, "index_format": "joblib"}


def _write_hnswlib_index(index_path: str, matrix, n_neighbors: int, metric: str, index_config: dict[str, Any]) -> dict[str, Any]:
    if metric != "cosine":
        raise ValueError("hnswlib backend currently supports only cosine metric.")
    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError("hnswlib index backend requires optional dependency: pip install -e '.[ann]'") from exc
    dense = _dense_float32(matrix)
    index = hnswlib.Index(space="cosine", dim=int(dense.shape[1]))
    index.init_index(
        max_elements=int(dense.shape[0]),
        ef_construction=int(index_config.get("ef_construction", 200)),
        M=int(index_config.get("M", 16)),
        random_seed=int(index_config.get("random_seed", 42)),
    )
    index.add_items(dense, np.arange(dense.shape[0], dtype=np.int64))
    index.set_ef(int(index_config.get("ef_search", max(n_neighbors, 50))))
    index.save_index(index_path)
    return {
        "index_path": index_path,
        "index_format": "hnswlib",
        "ef_construction": int(index_config.get("ef_construction", 200)),
        "M": int(index_config.get("M", 16)),
        "ef_search": int(index_config.get("ef_search", max(n_neighbors, 50))),
    }


def _write_faiss_index(index_path: str, matrix, n_neighbors: int, metric: str, index_config: dict[str, Any]) -> dict[str, Any]:
    if metric != "cosine":
        raise ValueError("faiss backend currently supports only cosine metric.")
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss index backend requires optional dependency: pip install -e '.[faiss]'") from exc
    dense = _l2_normalize(_dense_float32(matrix))
    index = faiss.IndexFlatIP(int(dense.shape[1]))
    index.add(dense)
    faiss.write_index(index, index_path)
    return {
        "index_path": index_path,
        "index_format": "faiss",
        "faiss_index_type": "IndexFlatIP",
        "stored_vector_normalization": "l2",
        "query_vector_normalization": "l2",
        "score_kind": "cosine_similarity",
        "faiss_threads": index_config.get("faiss_threads"),
    }


def _write_lancedb_index(index_path: str, matrix, rows: pd.DataFrame, n_neighbors: int, metric: str, index_config: dict[str, Any]) -> dict[str, Any]:
    if metric != "cosine":
        raise ValueError("lancedb backend currently supports only cosine metric.")
    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError("lancedb index backend requires optional dependency: pip install -e '.[lancedb]'") from exc
    dense = _dense_float32(matrix)
    table_name = str(index_config.get("table_name") or Path(index_path).name.replace(".", "_").replace("-", "_"))
    uri = str(index_config.get("uri") or Path(index_path).parent)
    db = lancedb.connect(uri)
    records = [
        {
            "row_index": int(i),
            "entity_id": str(rows.iloc[i]["entity_id"]),
            "vector": dense[i].tolist(),
        }
        for i in range(dense.shape[0])
    ]
    table = db.create_table(table_name, data=records, mode="overwrite")
    if bool(index_config.get("create_vector_index", False)):
        try:
            table.create_index(
                vector_column_name="vector",
                metric="cosine",
                num_partitions=int(index_config.get("num_partitions", 256)),
                num_sub_vectors=int(index_config.get("num_sub_vectors", 96)),
            )
        except TypeError:
            table.create_index(metric="cosine")
    return {
        "index_path": uri,
        "index_format": "lancedb",
        "table_name": table_name,
        "vector_column": "vector",
        "score_kind": "cosine_distance",
        "neighbors": int(max(1, min(n_neighbors, matrix.shape[0]))),
    }


def _write_entity_index(
    split: str,
    entity_type: str,
    matrix,
    metadata: pd.DataFrame,
    n_neighbors: int,
    *,
    backend: str,
    metric: str,
    embedding_config: dict[str, Any],
    index_config: dict[str, Any] | None = None,
) -> dict:
    index_config = index_config or {}
    backend = _resolve_backend(backend)
    if backend not in {"sklearn", "hnswlib", "faiss", "lancedb"}:
        raise ValueError(f"Unknown index backend: {backend}")
    rows = metadata[metadata["entity_type"] == entity_type].sort_values("row_index").reset_index(drop=True)
    if matrix.shape[0] != len(rows):
        raise ValueError(f"{split} {entity_type} embedding rows do not match metadata rows")
    stem = ENTITY_STEMS.get(entity_type, entity_type.lower())
    suffix = "joblib" if backend == "sklearn" else "faiss" if backend == "faiss" else "lancedb" if backend == "lancedb" else "bin"
    index_path = f"outputs/indexes/{split}_{stem}_neighbors.{suffix}"
    metadata_path = f"outputs/indexes/{split}_{stem}_index_metadata.parquet"
    manifest_path = f"outputs/indexes/{split}_{stem}_index_manifest.json"
    corpus_manifest = read_json("outputs/reports/corpus_manifest.json", {}) or {}
    corpus_info = corpus_manifest.get("corpus", {}) if isinstance(corpus_manifest, dict) else {}
    write_parquet(rows, metadata_path)
    if matrix.shape[0] == 0:
        manifest = {
            "backend": backend,
            "metric": metric,
            "split": split,
            "entity_type": entity_type,
            "rows": 0,
            "dimensions": int(matrix.shape[1]),
            "neighbors": 0,
            "index_built": False,
            "index_path": "",
            "index_format": None,
            "embedding_config_hash": _embedding_config_hash(embedding_config),
            "embedding_backend": embedding_config.get("backend"),
            "embedding_model_name": embedding_config.get("model_name"),
            "corpus_version": corpus_info.get("corpus_version"),
            "mathlib_commit": corpus_info.get("mathlib_commit"),
            "source_revision": corpus_info.get("source_revision"),
            "extraction_config_hash": corpus_info.get("extraction_config_hash"),
            "build_seconds": 0.0,
        }
        write_json(manifest_path, manifest)
        return {
            "entity_type": entity_type,
            **manifest,
            "metadata_path": metadata_path,
            "manifest_path": manifest_path,
        }
    start = perf_counter()
    if backend == "sklearn":
        index_info = _write_sklearn_index(index_path, matrix, n_neighbors=n_neighbors, metric=metric)
    elif backend == "hnswlib":
        index_info = _write_hnswlib_index(index_path, matrix, n_neighbors=n_neighbors, metric=metric, index_config=index_config)
    elif backend == "faiss":
        index_info = _write_faiss_index(index_path, matrix, n_neighbors=n_neighbors, metric=metric, index_config=index_config)
    else:
        index_info = _write_lancedb_index(index_path, matrix, rows, n_neighbors=n_neighbors, metric=metric, index_config=index_config)
    build_seconds = perf_counter() - start
    manifest = {
        "backend": backend,
        "metric": metric,
        "split": split,
        "entity_type": entity_type,
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "neighbors": int(max(1, min(n_neighbors, matrix.shape[0]))),
        "index_built": True,
        "embedding_config_hash": _embedding_config_hash(embedding_config),
        "embedding_backend": embedding_config.get("backend"),
        "embedding_model_name": embedding_config.get("model_name"),
        "corpus_version": corpus_info.get("corpus_version"),
        "mathlib_commit": corpus_info.get("mathlib_commit"),
        "source_revision": corpus_info.get("source_revision"),
        "extraction_config_hash": corpus_info.get("extraction_config_hash"),
        "build_seconds": float(build_seconds),
        **index_info,
    }
    write_json(manifest_path, manifest)
    return {
        "entity_type": entity_type,
        **manifest,
        "index_path": index_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
    }


def build_split(
    split: str,
    n_neighbors: int = 100,
    *,
    backend: str = "sklearn",
    metric: str = "cosine",
    embedding_config: dict[str, Any] | None = None,
    index_config: dict[str, Any] | None = None,
) -> dict:
    metadata_path = Path(f"outputs/embeddings/{split}_embedding_metadata.parquet")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata for split {split}. Run `leanrank-kg embed` first.")
    embedding_config = embedding_config or {}
    resolved_backend = _resolve_backend(backend)
    metadata = pd.read_parquet(metadata_path)
    proof_state_x = sparse.load_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz")
    premise_x = sparse.load_npz(f"outputs/embeddings/{split}_premise_embeddings.npz")
    theorem_x = sparse.load_npz(f"outputs/embeddings/{split}_theorem_embeddings.npz")
    start = perf_counter()
    proof_state_index = _write_entity_index(split, "ProofState", proof_state_x, metadata, n_neighbors, backend=backend, metric=metric, embedding_config=embedding_config, index_config=index_config)
    premise_index = _write_entity_index(split, "Premise", premise_x, metadata, n_neighbors, backend=backend, metric=metric, embedding_config=embedding_config, index_config=index_config)
    theorem_index = _write_entity_index(split, "Theorem", theorem_x, metadata, n_neighbors, backend=backend, metric=metric, embedding_config=embedding_config, index_config=index_config)
    return {
        "split": split,
        "backend": resolved_backend,
        "requested_backend": backend,
        "metric": metric,
        "neighbors": int(n_neighbors),
        "build_seconds": float(perf_counter() - start),
        "indexes": [
            proof_state_index,
            premise_index,
            theorem_index,
        ],
    }


def run(config_path: str | None = None) -> None:
    Path("outputs/indexes").mkdir(parents=True, exist_ok=True)
    config = load_config(config_path) if config_path else {}
    index_config = config.get("index", {})
    requested_backend = index_config.get("backend", "sklearn")
    backend = _resolve_backend(requested_backend)
    metric = index_config.get("metric", "cosine")
    n_neighbors = int(index_config.get("n_neighbors", 100))
    embedding_config = read_json("outputs/embeddings/embedding_config.json", {})
    corpus_manifest = read_json("outputs/reports/corpus_manifest.json", {}) or {}
    corpus_info = corpus_manifest.get("corpus", {}) if isinstance(corpus_manifest, dict) else {}
    summary = {
        "backend": backend,
        "requested_backend": requested_backend,
        "metric": metric,
        "neighbors": n_neighbors,
        "embedding": embedding_config,
        "embedding_config_hash": _embedding_config_hash(embedding_config),
        "corpus": {
            "corpus_version": corpus_info.get("corpus_version"),
            "mathlib_commit": corpus_info.get("mathlib_commit"),
            "source_revision": corpus_info.get("source_revision"),
            "extraction_config_hash": corpus_info.get("extraction_config_hash"),
        },
        "splits": {},
    }
    start = perf_counter()
    for split in SPLITS + ["demo"]:
        try:
            summary["splits"][split] = build_split(split, n_neighbors=n_neighbors, backend=requested_backend, metric=metric, embedding_config=embedding_config, index_config=index_config)
        except FileNotFoundError:
            continue
    summary["build_seconds"] = float(perf_counter() - start)
    write_json("outputs/indexes/index_summary.json", summary)
