from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .build_index import _embedding_config_hash
from .utils import load_config, read_json, write_json


ENTITY_MATRIX_FILES = {
    "ProofState": "proof_state",
    "Premise": "premise",
    "Theorem": "theorem",
}


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _load_matrix(split: str, entity_type: str):
    stem = ENTITY_MATRIX_FILES[entity_type]
    return sparse.load_npz(f"outputs/embeddings/{split}_{stem}_embeddings.npz")


def _load_ids(split: str, entity_type: str) -> list[str]:
    stem = ENTITY_MATRIX_FILES[entity_type]
    metadata_path = Path(f"outputs/indexes/{split}_{stem}_index_metadata.parquet")
    if metadata_path.exists():
        return pd.read_parquet(metadata_path).sort_values("row_index")["entity_id"].astype(str).tolist()
    metadata = pd.read_parquet(f"outputs/embeddings/{split}_embedding_metadata.parquet")
    rows = metadata[metadata["entity_type"] == entity_type].sort_values("row_index")
    return rows["entity_id"].astype(str).tolist()


def _manifest_matches(manifest: dict[str, Any], *, split: str, entity_type: str, matrix, ids: list[str], embedding_config: dict[str, Any]) -> bool:
    return (
        bool(manifest)
        and manifest.get("backend") in {"sklearn", "hnswlib", "faiss", "lancedb"}
        and manifest.get("metric") == "cosine"
        and manifest.get("split") == split
        and manifest.get("entity_type") == entity_type
        and int(manifest.get("rows", -1)) == int(matrix.shape[0])
        and int(manifest.get("dimensions", -1)) == int(matrix.shape[1])
        and int(manifest.get("rows", -1)) == len(ids)
        and manifest.get("embedding_config_hash") == _embedding_config_hash(embedding_config)
    )


def _sample_query_rows(row_count: int, query_count: int, seed: int) -> np.ndarray:
    if row_count <= 0:
        return np.array([], dtype=np.int64)
    count = max(1, min(int(query_count), row_count))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_count, size=count, replace=False))


def _exact_neighbors(query_x, candidate_x, top_k: int) -> tuple[np.ndarray, float]:
    start = perf_counter()
    scores = cosine_similarity(query_x, candidate_x)
    order = np.argsort(-scores, axis=1)[:, :top_k]
    return order, perf_counter() - start


def _indexed_neighbors(query_x, matrix, manifest: dict[str, Any], top_k: int) -> tuple[np.ndarray | None, float, str | None]:
    index_path = Path(manifest.get("index_path") or "")
    if not index_path.exists():
        return None, 0.0, f"missing index artifact: {index_path}"
    backend = manifest.get("backend")
    start = perf_counter()
    if backend == "sklearn":
        index = joblib.load(index_path)
        _, rows = index.kneighbors(query_x, n_neighbors=top_k)
    elif backend == "hnswlib":
        try:
            import hnswlib
        except ImportError:
            return None, 0.0, "hnswlib backend is not installed"
        index = hnswlib.Index(space="cosine", dim=int(matrix.shape[1]))
        index.load_index(str(index_path), max_elements=int(matrix.shape[0]))
        index.set_ef(int(manifest.get("ef_search", max(top_k, 50))))
        rows, _ = index.knn_query(_dense_float32(query_x), k=top_k)
    elif backend == "faiss":
        try:
            import faiss
        except ImportError:
            return None, 0.0, "faiss backend is not installed"
        index = faiss.read_index(str(index_path))
        if int(index.d) != int(matrix.shape[1]):
            return None, 0.0, "faiss index dimensions do not match current embeddings"
        _, rows = index.search(_l2_normalize(_dense_float32(query_x)), top_k)
    elif backend == "lancedb":
        try:
            import lancedb
        except ImportError:
            return None, 0.0, "lancedb backend is not installed"
        uri = manifest.get("index_path")
        table_name = manifest.get("table_name")
        if not uri or not table_name or not Path(str(uri)).exists():
            return None, 0.0, "lancedb table artifact is missing"
        db = lancedb.connect(str(uri))
        try:
            table = db.open_table(str(table_name))
        except Exception:
            return None, 0.0, "lancedb table cannot be opened"
        query_vectors = _dense_float32(query_x)
        row_lists = []
        for query_vector in query_vectors:
            try:
                result = table.search(query_vector.tolist(), vector_column_name=manifest.get("vector_column", "vector")).metric("cosine").limit(top_k).to_pandas()
            except TypeError:
                result = table.search(query_vector.tolist()).metric("cosine").limit(top_k).to_pandas()
            if "row_index" not in result.columns:
                return None, 0.0, "lancedb result is missing row_index"
            row_lists.append(result["row_index"].astype(int).to_numpy())
        rows = np.vstack(row_lists) if row_lists else np.empty((0, top_k), dtype=np.int64)
    else:
        return None, 0.0, f"unsupported index backend: {backend}"
    return np.asarray(rows, dtype=np.int64), perf_counter() - start, None


def _overlap_metrics(exact_rows: np.ndarray, indexed_rows: np.ndarray, k_values: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        overlaps = []
        top1_matches = []
        for exact, indexed in zip(exact_rows[:, :k], indexed_rows[:, :k], strict=False):
            exact_set = set(map(int, exact))
            indexed_set = set(map(int, indexed))
            overlaps.append(len(exact_set & indexed_set) / max(1, len(exact_set)))
            top1_matches.append(float(int(exact[0]) == int(indexed[0])))
        metrics[f"recall_at_{k}_vs_exact"] = float(np.mean(overlaps)) if overlaps else 0.0
        metrics[f"top1_match_at_{k}_vs_exact"] = float(np.mean(top1_matches)) if top1_matches else 0.0
    return metrics


def benchmark_entity(split: str, entity_type: str, *, top_k: int = 10, query_count: int = 25, seed: int = 0) -> dict[str, Any]:
    if entity_type not in ENTITY_MATRIX_FILES:
        raise ValueError(f"entity_type must be one of {sorted(ENTITY_MATRIX_FILES)}")
    matrix = _load_matrix(split, entity_type)
    ids = _load_ids(split, entity_type)
    stem = ENTITY_MATRIX_FILES[entity_type]
    manifest_path = Path(f"outputs/indexes/{split}_{stem}_index_manifest.json")
    manifest = read_json(manifest_path, {}) or {}
    embedding_config = read_json("outputs/embeddings/embedding_config.json", {}) or {}
    top_k = max(1, min(int(top_k), int(matrix.shape[0]))) if matrix.shape[0] else int(top_k)
    result: dict[str, Any] = {
        "split": split,
        "entity_type": entity_type,
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "query_count": 0,
        "top_k": int(top_k),
        "query_entity_ids": [],
        "manifest_path": str(manifest_path),
        "manifest_valid": _manifest_matches(manifest, split=split, entity_type=entity_type, matrix=matrix, ids=ids, embedding_config=embedding_config),
        "backend": manifest.get("backend", ""),
        "requested_neighbors": int(manifest.get("neighbors", 0) or 0),
        "index_build_seconds": float(manifest.get("build_seconds", 0.0) or 0.0),
    }
    if matrix.shape[0] == 0:
        result["indexed_available"] = False
        result["indexed_error"] = "no embedding rows available for benchmark"
        return result
    query_rows = _sample_query_rows(int(matrix.shape[0]), query_count, seed)
    query_x = matrix[query_rows]
    result["query_count"] = int(len(query_rows))
    result["query_entity_ids"] = [ids[int(i)] for i in query_rows[:10]]
    exact_rows, exact_seconds = _exact_neighbors(query_x, matrix, top_k)
    result["exact_total_seconds"] = float(exact_seconds)
    result["exact_ms_per_query"] = float(exact_seconds * 1000.0 / max(1, len(query_rows)))
    if not result["manifest_valid"]:
        result["indexed_available"] = False
        result["indexed_error"] = "index manifest is missing or does not match current embeddings"
        return result
    indexed_rows, indexed_seconds, error = _indexed_neighbors(query_x, matrix, manifest, top_k)
    if indexed_rows is None:
        result["indexed_available"] = False
        result["indexed_error"] = error or "index query failed"
        return result
    result["indexed_available"] = True
    result["indexed_total_seconds"] = float(indexed_seconds)
    result["indexed_ms_per_query"] = float(indexed_seconds * 1000.0 / max(1, len(query_rows)))
    result["speedup_vs_exact"] = float(exact_seconds / indexed_seconds) if indexed_seconds > 0 else 0.0
    k_values = sorted({1, min(5, top_k), top_k})
    result.update(_overlap_metrics(exact_rows, indexed_rows, k_values))
    return result


def run(config_path: str | None = None, *, split: str | None = None, top_k: int | None = None, query_count: int | None = None, seed: int | None = None) -> dict[str, Any]:
    config = load_config(config_path) if config_path else {}
    bench_config = config.get("index_benchmark", {})
    split = split or bench_config.get("split", "train")
    top_k = int(top_k or bench_config.get("top_k", 10))
    query_count = int(query_count or bench_config.get("query_count", 25))
    seed = int(seed if seed is not None else bench_config.get("seed", 0))
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    summary = {
        "split": split,
        "top_k": top_k,
        "query_count": query_count,
        "seed": seed,
        "entities": {
            ENTITY_MATRIX_FILES[entity_type]: benchmark_entity(split, entity_type, top_k=top_k, query_count=query_count, seed=seed)
            for entity_type in ENTITY_MATRIX_FILES
        },
    }
    write_json("outputs/reports/index_benchmark.json", summary)
    return summary
