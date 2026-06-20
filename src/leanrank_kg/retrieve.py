from __future__ import annotations

import json
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .lean_check import check_lean_syntax
from .query import InputType, build_query
from .theorem_similarity import theorem_similarity_rows
from .utils import namespace, read_json, stable_hash
from .weak_label_proof_technique import labels_for_text

ENTITY_STEMS = {
    "ProofState": "proof_state",
    "Premise": "premise",
    "Theorem": "theorem",
    "proof_state": "proof_state",
    "premise": "premise",
    "theorem": "theorem",
}
ENTITY_TYPE_NAMES = {
    "proof_state": "ProofState",
    "premise": "Premise",
    "theorem": "Theorem",
}


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _file_signature(path: str | Path) -> tuple[str, bool, int, int]:
    p = Path(path)
    if not p.exists():
        return (str(p), False, 0, 0)
    stat = p.stat()
    return (str(p), True, int(stat.st_mtime_ns), int(stat.st_size))


def _split_signature(split: str) -> tuple[tuple[str, bool, int, int], ...]:
    edge_path = Path(f"outputs/graph/{split}/edges_enriched.parquet")
    if not edge_path.exists():
        edge_path = Path(f"outputs/graph/{split}/edges.parquet")
    paths = [
        f"data/processed/{split}/proof_states.parquet",
        f"data/processed/{split}/premises.parquet",
        f"data/processed/{split}/theorems.parquet",
        f"data/processed/{split}/proof_state_features.parquet",
        f"data/processed/{split}/theorem_features.parquet",
        f"data/processed/{split}/proof_state_techniques.parquet",
        f"data/processed/{split}/premise_techniques.parquet",
        edge_path,
        f"outputs/embeddings/{split}_proof_state_embeddings.npz",
        f"outputs/embeddings/{split}_premise_embeddings.npz",
        f"outputs/embeddings/{split}_theorem_embeddings.npz",
    ]
    return tuple(_file_signature(path) for path in paths)


def _cwd() -> str:
    return os.getcwd()


def _load_split(split: str = "train") -> dict[str, Any]:
    return _load_split_cached(_cwd(), split, _split_signature(split))


@lru_cache(maxsize=16)
def _load_split_cached(cwd: str, split: str, signature: tuple[tuple[str, bool, int, int], ...]) -> dict[str, Any]:
    del cwd, signature
    return {
        "proof_states": pd.read_parquet(f"data/processed/{split}/proof_states.parquet"),
        "premises": pd.read_parquet(f"data/processed/{split}/premises.parquet"),
        "theorems": pd.read_parquet(f"data/processed/{split}/theorems.parquet"),
        "ps_features": pd.read_parquet(f"data/processed/{split}/proof_state_features.parquet"),
        "thm_features": pd.read_parquet(f"data/processed/{split}/theorem_features.parquet"),
        "ps_tech": pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet"),
        "prem_tech": pd.read_parquet(f"data/processed/{split}/premise_techniques.parquet"),
        "edges": pd.read_parquet(f"outputs/graph/{split}/edges_enriched.parquet") if Path(f"outputs/graph/{split}/edges_enriched.parquet").exists() else pd.read_parquet(f"outputs/graph/{split}/edges.parquet"),
        "ps_x": sparse.load_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz"),
        "prem_x": sparse.load_npz(f"outputs/embeddings/{split}_premise_embeddings.npz"),
        "thm_x": sparse.load_npz(f"outputs/embeddings/{split}_theorem_embeddings.npz"),
    }


def _load_embedding_config() -> dict[str, Any]:
    path = "outputs/embeddings/embedding_config.json"
    return _load_embedding_config_cached(_cwd(), _file_signature(path))


@lru_cache(maxsize=8)
def _load_embedding_config_cached(cwd: str, signature: tuple[str, bool, int, int]) -> dict[str, Any]:
    del cwd, signature
    return read_json("outputs/embeddings/embedding_config.json", {}) or {}


def _embedding_description() -> str:
    config = _load_embedding_config()
    backend = config.get("backend", "embedding")
    model = config.get("model_name")
    device = config.get("device")
    if model and device:
        return f"{model} embeddings on {device}"
    if model:
        return f"{model} embeddings"
    if backend == "tfidf":
        return "TF-IDF embeddings"
    return f"{backend} embeddings"


def _encode_query_text(query_text: str, query_kind: str = "proof_state"):
    config = _load_embedding_config()
    backend = config.get("backend", "tfidf")
    if backend == "tfidf":
        vectorizer_path = Path("outputs/embeddings/tfidf_vectorizer.joblib")
        if not vectorizer_path.exists():
            raise FileNotFoundError("Missing TF-IDF vectorizer. Run `leanrank-kg embed` first.")
        vectorizer = _load_joblib_cached(_cwd(), str(vectorizer_path), _file_signature(vectorizer_path))
        return vectorizer.transform([query_text])
    if backend in {"sentence_transformers", "sentence-transformer", "hf"}:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Query-time SentenceTransformer embeddings require the optional dependency: "
                "pip install -e '.[hf]'"
            ) from exc
        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("Missing model_name in outputs/embeddings/embedding_config.json")
        prefix = config.get("query_prefix") or ""
        if query_kind == "passage":
            prefix = config.get("passage_prefix") or ""
        model = _load_sentence_transformer(model_name, config.get("device"))
        vector = model.encode([prefix + query_text], normalize_embeddings=True, show_progress_bar=False, batch_size=int(config.get("batch_size") or 128))
        return sparse.csr_matrix(vector)
    raise ValueError(f"Unknown embedding backend: {backend}")


@lru_cache(maxsize=16)
def _load_joblib_cached(cwd: str, path: str, signature: tuple[str, bool, int, int]):
    del cwd, signature
    return joblib.load(path)


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str | None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _rank_frame_by_query(query_x, candidate_x, frame: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    scores = cosine_similarity(query_x, candidate_x).ravel()
    out = frame.copy()
    out[score_col] = scores
    return out.sort_values(score_col, ascending=False)


def _embedding_config_hash() -> str:
    return stable_hash(json.dumps(_load_embedding_config(), sort_keys=True), 16)


def _index_manifest(index_split: str, stem: str) -> dict[str, Any]:
    path = Path(f"outputs/indexes/{index_split}_{stem}_index_manifest.json")
    return _load_index_manifest_cached(_cwd(), str(path), _file_signature(path))


@lru_cache(maxsize=16)
def _load_index_manifest_cached(cwd: str, path: str, signature: tuple[str, bool, int, int]) -> dict[str, Any]:
    del cwd, signature
    return read_json(path, {}) or {}


def _index_manifest_matches(manifest: dict[str, Any], *, index_split: str, entity_type: str, candidate_x, frame: pd.DataFrame) -> bool:
    if not manifest:
        return False
    return (
        manifest.get("backend") in {"sklearn", "hnswlib", "faiss", "lancedb"}
        and manifest.get("metric") == "cosine"
        and manifest.get("split") == index_split
        and manifest.get("entity_type") == entity_type
        and int(manifest.get("rows", -1)) == int(candidate_x.shape[0])
        and int(manifest.get("dimensions", -1)) == int(candidate_x.shape[1])
        and int(manifest.get("rows", -1)) == int(len(frame))
        and manifest.get("embedding_config_hash") == _embedding_config_hash()
    )


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _rank_with_sklearn_index(query_x, index_path: Path, candidate_x, candidate_k: int):
    index = _load_joblib_cached(_cwd(), str(index_path), _file_signature(index_path))
    if getattr(index, "n_features_in_", candidate_x.shape[1]) != candidate_x.shape[1]:
        return None
    n_neighbors = max(1, min(candidate_k, candidate_x.shape[0]))
    distances, neighbor_rows = index.kneighbors(query_x, n_neighbors=n_neighbors)
    return distances.ravel(), neighbor_rows.ravel()


def _rank_with_hnswlib_index(query_x, manifest: dict[str, Any], candidate_x, candidate_k: int):
    index_path = Path(manifest.get("index_path", ""))
    if not index_path.exists():
        return None
    index = _load_hnswlib_index_cached(
        _cwd(),
        str(index_path),
        _file_signature(index_path),
        int(candidate_x.shape[1]),
        int(candidate_x.shape[0]),
    )
    if index is None:
        return None
    index.set_ef(int(manifest.get("ef_search", max(candidate_k, 50))))
    labels, distances = index.knn_query(_dense_float32(query_x), k=max(1, min(candidate_k, candidate_x.shape[0])))
    return distances.ravel(), labels.ravel()


@lru_cache(maxsize=12)
def _load_hnswlib_index_cached(cwd: str, path: str, signature: tuple[str, bool, int, int], dim: int, rows: int):
    del cwd, signature
    try:
        import hnswlib
    except ImportError:
        return None
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(path, max_elements=rows)
    return index


def _rank_with_faiss_index(query_x, manifest: dict[str, Any], candidate_x, candidate_k: int):
    index_path = Path(manifest.get("index_path", ""))
    if not index_path.exists():
        return None
    index = _load_faiss_index_cached(_cwd(), str(index_path), _file_signature(index_path))
    if index is None:
        return None
    if int(index.d) != int(candidate_x.shape[1]):
        return None
    scores, labels = index.search(_l2_normalize(_dense_float32(query_x)), max(1, min(candidate_k, candidate_x.shape[0])))
    return (1.0 - scores.ravel()), labels.ravel()


@lru_cache(maxsize=12)
def _load_faiss_index_cached(cwd: str, path: str, signature: tuple[str, bool, int, int]):
    del cwd, signature
    try:
        import faiss
    except ImportError:
        return None
    return faiss.read_index(path)


def _rank_with_lancedb_index(query_x, manifest: dict[str, Any], candidate_x, candidate_k: int):
    try:
        import lancedb
    except ImportError:
        return None
    uri = manifest.get("index_path")
    table_name = manifest.get("table_name")
    if not uri or not table_name or not Path(str(uri)).exists():
        return None
    db = lancedb.connect(str(uri))
    try:
        table = db.open_table(str(table_name))
    except Exception:
        return None
    query_vector = _dense_float32(query_x).reshape(1, -1)[0].tolist()
    try:
        result = table.search(query_vector, vector_column_name=manifest.get("vector_column", "vector")).metric("cosine").limit(max(1, min(candidate_k, candidate_x.shape[0]))).to_pandas()
    except TypeError:
        result = table.search(query_vector).metric("cosine").limit(max(1, min(candidate_k, candidate_x.shape[0]))).to_pandas()
    if "row_index" not in result.columns:
        return None
    distance_col = "_distance" if "_distance" in result.columns else "distance" if "distance" in result.columns else None
    if distance_col:
        distances = result[distance_col].astype(float).to_numpy()
    elif "score" in result.columns:
        distances = 1.0 - result["score"].astype(float).to_numpy()
    else:
        return None
    rows = result["row_index"].astype(int).to_numpy()
    return distances, rows


def _rank_frame_with_index(
    query_x,
    candidate_x,
    frame: pd.DataFrame,
    index_split: str,
    entity_type: str,
    candidate_k: int,
    score_col: str = "score",
) -> pd.DataFrame:
    stem = ENTITY_STEMS.get(entity_type, entity_type.lower())
    manifest_entity_type = ENTITY_TYPE_NAMES.get(stem, entity_type.title())
    metadata_path = Path(f"outputs/indexes/{index_split}_{stem}_index_metadata.parquet")
    if metadata_path.exists():
        manifest = _index_manifest(index_split, stem)
        if _index_manifest_matches(manifest, index_split=index_split, entity_type=manifest_entity_type, candidate_x=candidate_x, frame=frame):
            backend = manifest.get("backend")
            index_path = Path(manifest.get("index_path") or f"outputs/indexes/{index_split}_{stem}_neighbors.joblib")
            ranked = None
            if backend == "sklearn" and index_path.exists():
                ranked = _rank_with_sklearn_index(query_x, index_path, candidate_x, candidate_k)
            elif backend == "hnswlib":
                ranked = _rank_with_hnswlib_index(query_x, manifest, candidate_x, candidate_k)
            elif backend == "faiss":
                ranked = _rank_with_faiss_index(query_x, manifest, candidate_x, candidate_k)
            elif backend == "lancedb":
                ranked = _rank_with_lancedb_index(query_x, manifest, candidate_x, candidate_k)
            if ranked is None:
                out = _rank_frame_by_query(query_x, candidate_x, frame, score_col=score_col).head(candidate_k)
                out["retrieval_backend"] = "direct_cosine"
                return out
            distances, neighbor_rows = ranked
            metadata = _load_index_metadata(index_split, stem).sort_values("row_index").reset_index(drop=True)
            if len(metadata) == candidate_x.shape[0]:
                ids = metadata.iloc[neighbor_rows.ravel()]["entity_id"].tolist()
                scores = 1.0 - distances.ravel()
                out = frame.set_index("id").loc[ids].reset_index()
                out[score_col] = scores
                out["retrieval_backend"] = "nearest_neighbors_index" if backend == "sklearn" else f"{backend}_index"
                out["index_manifest_hash"] = manifest.get("embedding_config_hash", "")
                return out.sort_values(score_col, ascending=False)
    out = _rank_frame_by_query(query_x, candidate_x, frame, score_col=score_col).head(candidate_k)
    out["retrieval_backend"] = "direct_cosine"
    return out


def _namespace_prefix(full_name: str) -> str:
    text = str(full_name or "")
    return text.split(".")[0] if text else ""


def _premise_explanation(query_text: str, premise: dict[str, Any], score: float) -> dict[str, Any]:
    query_l = query_text.lower()
    full_name = str(premise.get("full_name", ""))
    namespace = _namespace_prefix(full_name)
    symbols = [token for token in full_name.replace(".", " ").replace("_", " ").split() if len(token) >= 3]
    shared_name_tokens = sorted({token for token in symbols if token.lower() in query_l})[:8]
    return {
        "embedding_similarity": float(score),
        "shared_name_tokens": shared_name_tokens,
        "namespace_hint": namespace,
        "domain_hint": premise.get("domain_tag", ""),
        "file_path": premise.get("file_path", ""),
        "reason": f"Ranked by cosine similarity over {_embedding_description()} with namespace, domain, and name-token evidence exposed for review.",
    }


def _premise_frequency(index_split: str) -> pd.Series:
    return _premise_frequency_cached(_cwd(), index_split, _file_signature(f"data/processed/{index_split}/positive_edges.parquet"))


@lru_cache(maxsize=8)
def _premise_frequency_cached(cwd: str, index_split: str, signature: tuple[str, bool, int, int]) -> pd.Series:
    del cwd, signature
    try:
        pos = pd.read_parquet(f"data/processed/{index_split}/positive_edges.parquet")
    except FileNotFoundError:
        return pd.Series(dtype=float)
    counts = pos.groupby("premise_id").size().astype(float)
    if counts.empty or counts.max() <= 0:
        return pd.Series(dtype=float)
    return counts / counts.max()


def _token_set(*parts: object) -> set[str]:
    text = " ".join(str(part or "") for part in parts).replace(".", " ").replace("_", " ")
    return {token.lower() for token in text.split() if len(token) >= 3}


def _jaccard_score(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _rerank_premise_candidates(
    query_text: str,
    candidates: pd.DataFrame,
    index_data: dict[str, Any],
    index_split: str,
    similar_theorem_ids: list[str] | None = None,
    query_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    query_labels = {row["label"] for row in labels_for_text(query_text)}
    prem_tech = index_data["prem_tech"]
    premise_labels = {key: set(group["label"]) for key, group in prem_tech.groupby("premise_id")} if not prem_tech.empty else {}
    frequency = _premise_frequency(index_split)
    edges = index_data["edges"]
    theorem_ids = set(similar_theorem_ids or [])
    graph_premises = set()
    theorem_neighbor_counts: dict[str, int] = {}
    if theorem_ids and not edges.empty:
        neighbor_edges = edges[(edges["edge_type"] == "invokes_premise") & (edges["source"].isin(theorem_ids))]
        graph_premises = set(neighbor_edges["target"])
        theorem_neighbor_counts = neighbor_edges.groupby("target").size().astype(int).to_dict()
    all_invokes = edges[edges["edge_type"] == "invokes_premise"] if not edges.empty else pd.DataFrame()
    premise_degree = all_invokes.groupby("target").size().astype(float) if not all_invokes.empty else pd.Series(dtype=float)
    if not premise_degree.empty and float(premise_degree.max()) > 0:
        premise_degree = premise_degree / float(premise_degree.max())
    query_l = query_text.lower()
    query_context = query_context or {}
    query_namespace = namespace(query_context.get("full_name", ""))
    query_namespaces = set(query_context.get("namespace_hints", []) or [])
    query_domain = str(query_context.get("domain_hint") or "")
    conclusion_symbols = {str(symbol).lower() for symbol in query_context.get("conclusion_symbols", []) or []}
    query_tokens = _token_set(query_text, query_context.get("full_name", ""), " ".join(map(str, query_context.get("local_hypotheses", []) or [])))
    proof_state_difficulty = _query_difficulty_score(query_text, query_context)
    ranker_model = _load_premise_ranker()
    rows = []
    for row in candidates.to_dict(orient="records"):
        full_name = str(row.get("full_name", ""))
        premise_namespace = namespace(full_name)
        tokens = {token.lower() for token in full_name.replace(".", " ").replace("_", " ").split() if len(token) >= 3}
        premise_context_tokens = _token_set(full_name, row.get("code", ""))
        shared_token_score = min(sum(1 for token in tokens if token in query_l) / 4.0, 1.0)
        conclusion_symbol_score = min(sum(1 for token in tokens if token in conclusion_symbols) / 4.0, 1.0)
        technique_score = 1.0 if query_labels & premise_labels.get(row["id"], set()) else 0.0
        frequency_score = float(frequency.get(row["id"], 0.0)) if not frequency.empty else 0.0
        graph_score = 1.0 if row["id"] in graph_premises else 0.0
        theorem_neighborhood_score = min(theorem_neighbor_counts.get(row["id"], 0) / max(len(theorem_ids), 1), 1.0) if theorem_ids else 0.0
        graph_premise_degree = float(premise_degree.get(row["id"], 0.0)) if not premise_degree.empty else 0.0
        symbol_name_overlap = _jaccard_score(query_tokens, tokens)
        symbol_context_overlap = _jaccard_score(query_tokens, premise_context_tokens)
        embedding_score = float(row["score"])
        same_namespace = float(bool(query_namespace and query_namespace == premise_namespace) or premise_namespace in query_namespaces)
        same_domain = float(bool(query_domain and query_domain == str(row.get("domain_tag", ""))))
        learned_ranker_score = _score_with_premise_ranker(
            ranker_model,
            {
                "cosine_similarity": embedding_score,
                "same_namespace": same_namespace,
                "same_domain": same_domain,
                "proof_technique_overlap": technique_score,
                "proof_state_difficulty": proof_state_difficulty,
                "negative_candidate_hardness": 0.0,
                "premise_frequency": frequency_score,
                "symbol_name_overlap": symbol_name_overlap,
                "symbol_context_overlap": symbol_context_overlap,
                "graph_premise_degree": graph_premise_degree,
                "theorem_neighborhood_premise_score": theorem_neighborhood_score,
            },
        )
        fixed_score = (
            0.68 * embedding_score
            + 0.10 * frequency_score
            + 0.08 * technique_score
            + 0.06 * graph_score
            + 0.03 * theorem_neighborhood_score
            + 0.02 * graph_premise_degree
            + 0.01 * max(shared_token_score, symbol_name_overlap)
            + 0.01 * max(conclusion_symbol_score, symbol_context_overlap)
        )
        rerank_score = 0.55 * learned_ranker_score + 0.45 * fixed_score if learned_ranker_score is not None else fixed_score
        enriched = dict(row)
        enriched.update(
            {
                "embedding_score": embedding_score,
                "learned_ranker_score": learned_ranker_score,
                "same_namespace_score": same_namespace,
                "same_domain_score": same_domain,
                "premise_frequency_score": frequency_score,
                "proof_technique_overlap_score": technique_score,
                "graph_neighbor_score": graph_score,
                "graph_premise_degree": graph_premise_degree,
                "theorem_neighborhood_premise_score": theorem_neighborhood_score,
                "shared_name_token_score": shared_token_score,
                "symbol_name_overlap": symbol_name_overlap,
                "symbol_context_overlap": symbol_context_overlap,
                "conclusion_symbol_score": conclusion_symbol_score,
                "fixed_rerank_score": fixed_score,
                "score": rerank_score,
            }
        )
        rows.append(enriched)
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def _query_difficulty_score(query_text: str, query_context: dict[str, Any]) -> float:
    hypotheses = query_context.get("local_hypotheses") or []
    symbols = query_context.get("symbols") or []
    length_score = min(len(query_text) / 1200.0, 1.0)
    hypothesis_score = min(len(hypotheses) / 12.0, 1.0)
    symbol_score = min(len(symbols) / 120.0, 1.0)
    return float(np.mean([length_score, hypothesis_score, symbol_score]))


def _load_premise_ranker():
    path = Path("outputs/models/premise_ranker.joblib")
    if not path.exists():
        return None
    return _load_joblib_cached(_cwd(), str(path), _file_signature(path))


def _load_difficulty_estimator():
    path = Path("outputs/models/difficulty_estimator.joblib")
    if not path.exists():
        return None
    return _load_joblib_cached(_cwd(), str(path), _file_signature(path))


def _load_index_metadata(index_split: str, stem: str) -> pd.DataFrame:
    path = Path(f"outputs/indexes/{index_split}_{stem}_index_metadata.parquet")
    return _load_index_metadata_cached(_cwd(), str(path), _file_signature(path))


@lru_cache(maxsize=16)
def _load_index_metadata_cached(cwd: str, path: str, signature: tuple[str, bool, int, int]) -> pd.DataFrame:
    del cwd, signature
    return pd.read_parquet(path)


def clear_retrieval_caches() -> None:
    _load_split_cached.cache_clear()
    _load_embedding_config_cached.cache_clear()
    _load_joblib_cached.cache_clear()
    _load_sentence_transformer.cache_clear()
    _load_hnswlib_index_cached.cache_clear()
    _load_faiss_index_cached.cache_clear()
    _premise_frequency_cached.cache_clear()
    _load_index_metadata_cached.cache_clear()
    _load_index_manifest_cached.cache_clear()


def _score_with_premise_ranker(model, features: dict[str, float]) -> float | None:
    if model is None:
        return None
    columns = list(getattr(model, "feature_names_in_", features.keys()))
    frame = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in columns}])
    try:
        return float(model.predict_proba(frame)[:, 1][0])
    except Exception:
        return None


def _ranking_reasons(row: dict[str, Any], explanation: dict[str, Any]) -> list[str]:
    reasons = []
    if float(row.get("embedding_score", 0.0)) >= 0.25:
        reasons.append(f"high query-premise embedding similarity ({float(row['embedding_score']):.3f})")
    learned = row.get("learned_ranker_score")
    if learned is not None and not pd.isna(learned):
        reasons.append(f"learned premise ranker score {float(learned):.3f}")
    if float(row.get("same_namespace_score", 0.0)) > 0:
        reasons.append("same or related namespace")
    if float(row.get("same_domain_score", 0.0)) > 0:
        reasons.append("same domain hint")
    if float(row.get("proof_technique_overlap_score", 0.0)) > 0:
        reasons.append("compatible proof-technique labels")
    if float(row.get("graph_neighbor_score", 0.0)) > 0:
        reasons.append("used by a similar theorem in the KG")
    if float(row.get("theorem_neighborhood_premise_score", 0.0)) > 0:
        reasons.append("appears in the similar-theorem neighborhood")
    if float(row.get("graph_premise_degree", 0.0)) >= 0.5:
        reasons.append("high theorem-level premise degree in the KG")
    if float(row.get("symbol_context_overlap", 0.0)) > 0:
        reasons.append("symbol overlap with query context")
    if float(row.get("conclusion_symbol_score", 0.0)) > 0:
        reasons.append("shares parsed conclusion symbols")
    if float(row.get("premise_frequency_score", 0.0)) >= 0.5:
        reasons.append("frequent positive premise in the train graph")
    shared_tokens = explanation.get("shared_name_tokens", [])
    if shared_tokens:
        reasons.append(f"shared name tokens: {', '.join(shared_tokens[:4])}")
    if not reasons:
        reasons.append("nearest candidate by embedding similarity")
    return reasons[:5]


def retrieve_premises(proof_state_id: str, k: int = 10, split: str = "train", index_split: str = "train") -> list[dict]:
    query_data = _load_split(split)
    index_data = _load_split(index_split)
    ps = query_data["proof_states"]
    idxs = ps.index[ps["id"] == proof_state_id].tolist()
    if not idxs:
        return []
    scores = cosine_similarity(query_data["ps_x"][idxs[0]], index_data["prem_x"]).ravel()
    prem = index_data["premises"].copy()
    prem["score"] = scores
    out = prem.sort_values("score", ascending=False).head(k)
    return [
        {
            "premise_id": r["id"],
            "full_name": r["full_name"],
            "score": float(r["score"]),
            "index_split": index_split,
            "explanation": f"Cosine similarity over {_embedding_description()} for proof-state queries and train-index premise text",
        }
        for r in out.to_dict(orient="records")
    ]


def retrieve_premises_for_query(
    query_text: str,
    k: int = 10,
    index_split: str = "train",
    query_kind: str = "proof_state",
    candidate_k: int = 100,
    similar_theorem_ids: list[str] | None = None,
    query_context: dict[str, Any] | None = None,
) -> list[dict]:
    index_data = _load_split(index_split)
    query_x = _encode_query_text(query_text, query_kind=query_kind)
    ranked = _rank_frame_with_index(query_x, index_data["prem_x"], index_data["premises"], index_split, "premise", max(k, candidate_k))
    ranked = _rerank_premise_candidates(query_text, ranked, index_data, index_split, similar_theorem_ids=similar_theorem_ids, query_context=query_context).head(k)
    rows = []
    for row in ranked.to_dict(orient="records"):
        explanation = _premise_explanation(query_text, row, row["embedding_score"])
        reasons = _ranking_reasons(row, explanation)
        rows.append(
            {
                "premise_id": row["id"],
                "full_name": row["full_name"],
                "score": float(row["score"]),
                "embedding_score": float(row["embedding_score"]),
                "index_split": index_split,
                "explanation": "; ".join(reasons),
                "ranking_reasons": reasons,
                "signals": {
                    **explanation,
                    "embedding_score": float(row["embedding_score"]),
                    "retrieval_backend": row.get("retrieval_backend", "direct_cosine"),
                    "index_manifest_hash": row.get("index_manifest_hash", ""),
                    "rerank_score": float(row["score"]),
                    "fixed_rerank_score": float(row.get("fixed_rerank_score", row["score"])),
                    "learned_ranker_score": None if pd.isna(row.get("learned_ranker_score")) else float(row.get("learned_ranker_score")),
                    "same_namespace_score": float(row.get("same_namespace_score", 0.0)),
                    "same_domain_score": float(row.get("same_domain_score", 0.0)),
                    "premise_frequency_score": float(row.get("premise_frequency_score", 0.0)),
                    "proof_technique_overlap_score": float(row.get("proof_technique_overlap_score", 0.0)),
                    "graph_neighbor_score": float(row.get("graph_neighbor_score", 0.0)),
                    "graph_premise_degree": float(row.get("graph_premise_degree", 0.0)),
                    "theorem_neighborhood_premise_score": float(row.get("theorem_neighborhood_premise_score", 0.0)),
                    "shared_name_token_score": float(row.get("shared_name_token_score", 0.0)),
                    "symbol_name_overlap": float(row.get("symbol_name_overlap", 0.0)),
                    "symbol_context_overlap": float(row.get("symbol_context_overlap", 0.0)),
                    "conclusion_symbol_score": float(row.get("conclusion_symbol_score", 0.0)),
                    "parsed_symbol_overlap_score": float(row.get("conclusion_symbol_score", 0.0)),
                },
            }
        )
    return rows


def retrieve_similar_theorems(theorem_id: str, k: int = 10, split: str = "train") -> list[dict]:
    rows = [row for row in theorem_similarity_rows(split, k) if row["source"] == theorem_id]
    return [
        {
            "theorem_id": row["target"],
            "full_name": row["target_full_name"],
            "score": row["score"],
            "explanation": f"Weighted non-GNN similarity: {_embedding_description()}, shared premises, namespace/file area, proof techniques, and difficulty",
            "signals": {
                "tfidf_similarity": row["tfidf_similarity"],
                "shared_premise_score": row["shared_premise_score"],
                "file_namespace_score": row["file_namespace_score"],
                "proof_technique_overlap": row["proof_technique_overlap"],
                "difficulty_similarity": row["difficulty_similarity"],
            },
        }
        for row in rows[:k]
    ]


def retrieve_similar_theorems_for_query(query_text: str, k: int = 10, index_split: str = "train") -> list[dict]:
    index_data = _load_split(index_split)
    query_x = _encode_query_text(query_text, query_kind="proof_state")
    thm_meta = pd.read_parquet(f"outputs/embeddings/{index_split}_embedding_metadata.parquet")
    thm_meta = thm_meta[thm_meta["entity_type"] == "Theorem"].sort_values("row_index")
    theorem_rows = index_data["theorems"].set_index("id")
    ranked = _rank_frame_with_index(query_x, index_data["thm_x"], thm_meta.rename(columns={"entity_id": "id"}), index_split, "theorem", k).head(k)
    rows = []
    for row in ranked.to_dict(orient="records"):
        thm = theorem_rows.loc[row["id"]] if row["id"] in theorem_rows.index else pd.Series({"full_name": row["id"], "domain_tag": "", "subdomain_tag": "", "file_path": ""})
        rows.append(
            {
                "theorem_id": row["id"],
                "full_name": thm["full_name"],
                "score": float(row["score"]),
                "index_split": index_split,
                "explanation": f"Cosine similarity over {_embedding_description()} between the query text and theorem-level embeddings aggregated from proof states and positive premises.",
                "signals": {
                    "embedding_similarity": float(row["score"]),
                    "retrieval_backend": row.get("retrieval_backend", "direct_cosine"),
                    "index_manifest_hash": row.get("index_manifest_hash", ""),
                    "domain_hint": thm.get("domain_tag", ""),
                    "subdomain_hint": thm.get("subdomain_tag", ""),
                    "file_path": thm.get("file_path", ""),
                },
            }
        )
    return rows


def retrieve_similar_proof_states_for_query(query_text: str, k: int = 10, index_split: str = "train") -> list[dict]:
    index_data = _load_split(index_split)
    query_x = _encode_query_text(query_text, query_kind="proof_state")
    ranked = _rank_frame_with_index(query_x, index_data["ps_x"], index_data["proof_states"], index_split, "proof_state", k).head(k)
    tech = index_data["ps_tech"]
    rows = []
    for row in ranked.to_dict(orient="records"):
        labels = []
        if not tech.empty:
            labels = tech[tech["proof_state_id"] == row["id"]][["label", "provenance"]].to_dict(orient="records")
        rows.append(
            {
                "proof_state_id": row["id"],
                "theorem_id": row.get("theorem_id", ""),
                "full_name": row.get("full_name", ""),
                "score": float(row["score"]),
                "goal_text": str(row.get("goal_text", ""))[:500],
                "tactic_idx": int(row.get("tactic_idx", 0)),
                "proof_techniques": labels,
                "explanation": f"Retrieved by proof-state embedding similarity over {_embedding_description()} and indexed historical LeanRank proof states.",
                "signals": {
                    "retrieval_backend": row.get("retrieval_backend", "direct_cosine"),
                    "index_manifest_hash": row.get("index_manifest_hash", ""),
                    "embedding_score": float(row["score"]),
                    "domain_hint": row.get("domain_tag", ""),
                },
            }
        )
    return rows


def explain_premise_match(proof_state_id: str, premise_id: str, split: str = "train", index_split: str = "train") -> dict:
    results = retrieve_premises(proof_state_id, k=1000, split=split, index_split=index_split)
    score = next((r["score"] for r in results if r["premise_id"] == premise_id), 0.0)
    query_data = _load_split(split)
    index_data = _load_split(index_split)
    ps = query_data["proof_states"].set_index("id").loc[proof_state_id]
    index_premises = index_data["premises"].set_index("id")
    if premise_id in index_premises.index:
        prem = index_premises.loc[premise_id]
        premise_in_index = True
        prem_tech = index_data["prem_tech"]
    else:
        query_premises = query_data["premises"].set_index("id")
        prem = query_premises.loc[premise_id] if premise_id in query_premises.index else pd.Series({"full_name": "", "domain_tag": "", "file_path": ""})
        premise_in_index = False
        prem_tech = query_data["prem_tech"]
    ps_labels = set(query_data["ps_tech"].query("proof_state_id == @proof_state_id")["label"]) if not query_data["ps_tech"].empty else set()
    prem_labels = set(prem_tech.query("premise_id == @premise_id")["label"]) if not prem_tech.empty else set()
    return {
        "proof_state_id": proof_state_id,
        "premise_id": premise_id,
        "cosine_score": float(score),
        "index_split": index_split,
        "premise_in_index": premise_in_index,
        "namespace_match": ps["full_name"].split(".")[0] == prem["full_name"].split(".")[0],
        "file_match": ps.get("file_path", "") == prem.get("file_path", ""),
        "domain_match": ps.get("domain_tag", "") == prem.get("domain_tag", ""),
        "shared_proof_techniques": sorted(ps_labels & prem_labels),
    }


def get_proof_technique_labels(proof_state_id: str, split: str = "train") -> list[dict]:
    tech = _load_split(split)["ps_tech"]
    if tech.empty:
        return []
    return tech[tech["proof_state_id"] == proof_state_id][["label", "provenance"]].to_dict(orient="records")


def get_difficulty_profile(entity_id: str, split: str = "train") -> dict:
    data = _load_split(split)
    for frame_name in ["ps_features", "thm_features"]:
        frame = data[frame_name]
        key = "id" if "id" in frame.columns else "theorem_id"
        hit = frame[frame[key] == entity_id]
        if not hit.empty:
            return _jsonable(hit.iloc[0].to_dict())
    return {}


def get_graph_neighborhood(entity_id: str, depth: int = 1, split: str = "train") -> dict:
    edges = _load_split(split)["edges"]
    frontier = {entity_id}
    seen = {entity_id}
    selected = []
    for _ in range(depth):
        mask = edges["source"].isin(frontier) | edges["target"].isin(frontier)
        batch = edges[mask]
        selected.extend(_jsonable(batch.to_dict(orient="records")))
        frontier = set(batch["source"]) | set(batch["target"])
        frontier -= seen
        seen |= frontier
    return {"entity_id": entity_id, "nodes": sorted(seen), "edges": selected}


def _historical_difficulty_prior(similar_theorems: list[dict], index_split: str) -> dict[str, Any]:
    if not similar_theorems:
        return {"score": None, "confidence": 0.0, "neighbors": []}
    try:
        features = _load_split(index_split)["thm_features"]
    except FileNotFoundError:
        return {"score": None, "confidence": 0.0, "neighbors": []}
    key = "theorem_id" if "theorem_id" in features.columns else "id"
    feature_rows = features.set_index(key)
    rows = []
    for theorem in similar_theorems[:10]:
        theorem_id = theorem.get("theorem_id")
        if theorem_id not in feature_rows.index:
            continue
        feature = feature_rows.loc[theorem_id]
        score = float(feature.get("theorem_complexity_score", feature.get("mean_proof_state_difficulty", feature.get("difficulty_score", 0.0))))
        weight = max(float(theorem.get("score", 0.0)), 0.0)
        rows.append(
            {
                "theorem_id": theorem_id,
                "full_name": theorem.get("full_name", theorem_id),
                "similarity": weight,
                "difficulty_score": score,
                "difficulty_bucket": feature.get("difficulty_bucket", ""),
                "difficulty_target_source": feature.get("difficulty_target_source", ""),
            }
        )
    if not rows:
        return {"score": None, "confidence": 0.0, "neighbors": []}
    weights = np.array([row["similarity"] for row in rows], dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(rows), dtype=float)
    scores = np.array([row["difficulty_score"] for row in rows], dtype=float)
    prior = float(np.average(scores, weights=weights))
    confidence = float(min(len(rows) / 5.0, 1.0) * min(float(weights.max()), 1.0))
    return {"score": prior, "confidence": confidence, "neighbors": rows[:5]}


def _difficulty_for_query(query_text: str, retrieved_premises: list[dict], similar_theorems: list[dict], index_split: str = "train") -> dict:
    lines = [line for line in query_text.splitlines() if line.strip()]
    hypothesis_like = sum(1 for line in lines if ":" in line and "⊢" not in line)
    symbol_count = sum(1 for char in query_text if not char.isalnum() and not char.isspace())
    length_score = min(len(query_text) / 1200.0, 1.0)
    hypothesis_score = min(hypothesis_like / 12.0, 1.0)
    symbol_score = min(symbol_count / 120.0, 1.0)
    retrieval_uncertainty = 1.0 - max([row["score"] for row in retrieved_premises[:1]] or [0.0])
    theorem_uncertainty = 1.0 - max([row["score"] for row in similar_theorems[:1]] or [0.0])
    heuristic_score = float(np.mean([length_score, hypothesis_score, symbol_score, retrieval_uncertainty, theorem_uncertainty]))
    historical_prior = _historical_difficulty_prior(similar_theorems, index_split)
    model_signal = _model_difficulty_signal(
        {
            "context_length_score": length_score,
            "num_local_hypotheses": hypothesis_score,
            "num_positive_premises": min(len(retrieved_premises) / 20.0, 1.0),
            "avg_positive_premise_length": min(
                (sum(len(str(row.get("full_name", ""))) for row in retrieved_premises) / max(len(retrieved_premises), 1)) / 120.0,
                1.0,
            )
            if retrieved_premises
            else 0.0,
            "premise_namespace_rarity": 1.0 - max([row.get("signals", {}).get("premise_frequency_score", 0.0) for row in retrieved_premises[:5]] or [0.0]),
            "tactic_step_index_score": 0.0,
            "negative_candidate_hardness": float(np.mean([retrieval_uncertainty, theorem_uncertainty])),
        }
    )
    if historical_prior["score"] is None:
        score = heuristic_score
        calibrated_by = "query_heuristic"
    else:
        prior_weight = min(0.45, 0.15 + 0.30 * historical_prior["confidence"])
        score = float((1.0 - prior_weight) * heuristic_score + prior_weight * float(historical_prior["score"]))
        calibrated_by = "query_heuristic_and_similar_theorem_prior"
    if model_signal["score"] is not None:
        score = float(0.75 * score + 0.25 * float(model_signal["score"]))
        calibrated_by = f"{calibrated_by}_and_trained_estimator"
    if score < 0.34:
        bucket = "easy"
    elif score < 0.67:
        bucket = "medium"
    else:
        bucket = "hard"
    return {
        "difficulty_score": score,
        "difficulty_bucket": bucket,
        "signals": {
            "query_length_score": length_score,
            "hypothesis_count_score": hypothesis_score,
            "symbol_density_score": symbol_score,
            "premise_retrieval_uncertainty": retrieval_uncertainty,
            "similar_theorem_uncertainty": theorem_uncertainty,
            "heuristic_difficulty_score": heuristic_score,
            "historical_prior_score": historical_prior["score"],
            "historical_prior_confidence": historical_prior["confidence"],
            "trained_estimator_score": model_signal["score"],
            "trained_estimator_available": model_signal["available"],
            "trained_estimator_uncertainty": model_signal.get("uncertainty"),
            "trained_estimator_confidence_interval": model_signal.get("confidence_interval"),
            "calibrated_by": calibrated_by,
        },
        "trained_estimator_calibration_bins": model_signal.get("calibration_bins", []),
        "similar_theorem_difficulty_neighbors": historical_prior["neighbors"],
    }


def _pre_retrieval_difficulty_estimate(query_text: str, query_context: dict[str, Any], similar_theorems: list[dict], index_split: str) -> dict[str, Any]:
    lines = [line for line in query_text.splitlines() if line.strip()]
    hypothesis_like = sum(1 for line in lines if ":" in line and "⊢" not in line)
    symbol_count = sum(1 for char in query_text if not char.isalnum() and not char.isspace())
    length_score = min(len(query_text) / 1200.0, 1.0)
    hypothesis_score = min(max(hypothesis_like, len(query_context.get("local_hypotheses") or [])) / 12.0, 1.0)
    symbol_score = min(max(symbol_count, len(query_context.get("symbols") or [])) / 120.0, 1.0)
    structural_score = _query_difficulty_score(query_text, query_context)
    heuristic = float(np.mean([length_score, hypothesis_score, symbol_score, structural_score]))
    historical_prior = _historical_difficulty_prior(similar_theorems, index_split)
    if historical_prior["score"] is None:
        score = heuristic
        calibrated_by = "pre_retrieval_query_heuristic"
    else:
        prior_weight = min(0.40, 0.12 + 0.28 * historical_prior["confidence"])
        score = float((1.0 - prior_weight) * heuristic + prior_weight * float(historical_prior["score"]))
        calibrated_by = "pre_retrieval_query_heuristic_and_similar_theorem_prior"
    return {
        "score": max(0.0, min(1.0, score)),
        "calibrated_by": calibrated_by,
        "signals": {
            "query_length_score": length_score,
            "hypothesis_count_score": hypothesis_score,
            "symbol_density_score": symbol_score,
            "structural_query_score": structural_score,
            "historical_prior_score": historical_prior["score"],
            "historical_prior_confidence": historical_prior["confidence"],
        },
    }


def _adaptive_retrieval_policy(k_premises: int, requested_candidate_k: int, difficulty_estimate: dict[str, Any]) -> dict[str, Any]:
    score = float(difficulty_estimate.get("score", 0.0) or 0.0)
    base = max(int(requested_candidate_k), int(k_premises) * 4, 40)
    if score >= 0.67:
        multiplier = 3.0
        bucket = "hard"
    elif score >= 0.34:
        multiplier = 2.0
        bucket = "medium"
    else:
        multiplier = 1.25
        bucket = "easy"
    candidate_k = int(min(max(base * multiplier, k_premises), 500))
    return {
        "enabled": True,
        "difficulty_score": score,
        "difficulty_bucket": bucket,
        "requested_k_premises": int(k_premises),
        "base_candidate_k": int(base),
        "candidate_k": candidate_k,
        "policy": "expand_candidate_pool_by_pre_retrieval_difficulty",
        "calibrated_by": difficulty_estimate.get("calibrated_by", ""),
        "signals": difficulty_estimate.get("signals", {}),
    }


def _model_difficulty_signal(features: dict[str, float]) -> dict[str, Any]:
    artifact = _load_difficulty_estimator()
    if artifact is None:
        return {"available": False, "score": None, "uncertainty": None, "confidence_interval": None}
    columns = artifact.get("feature_columns", [])
    model = artifact.get("model")
    if not columns or model is None:
        return {"available": False, "score": None, "uncertainty": None, "confidence_interval": None}
    frame = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in columns}])
    try:
        score = float(model.predict(frame)[0])
    except Exception:
        return {"available": False, "score": None, "uncertainty": None, "confidence_interval": None}
    score = max(0.0, min(1.0, score))
    residuals = artifact.get("residual_quantiles", {}) or {}
    uncertainty = residuals.get("p80", residuals.get("p50"))
    interval = None
    if uncertainty is not None:
        uncertainty = float(uncertainty)
        interval = {
            "low": max(0.0, score - uncertainty),
            "high": min(1.0, score + uncertainty),
            "residual_quantile": "p80" if "p80" in residuals else "p50",
        }
    return {
        "available": True,
        "score": score,
        "uncertainty": uncertainty,
        "confidence_interval": interval,
        "calibration_bins": artifact.get("calibration_bins", []),
    }


def _related_proof_patterns(similar_theorems: list[dict], index_split: str, limit: int = 5) -> list[dict]:
    if not similar_theorems:
        return []
    data = _load_split(index_split)
    theorem_ids = [row["theorem_id"] for row in similar_theorems]
    ps = data["proof_states"][data["proof_states"]["theorem_id"].isin(theorem_ids)].head(limit)
    tech = data["ps_tech"]
    rows = []
    for row in ps.to_dict(orient="records"):
        labels = []
        if not tech.empty:
            labels = tech[tech["proof_state_id"] == row["id"]][["label", "provenance"]].to_dict(orient="records")
        rows.append(
            {
                "proof_state_id": row["id"],
                "theorem_id": row["theorem_id"],
                "full_name": row["full_name"],
                "goal_text": str(row.get("goal_text", ""))[:500],
                "tactic_idx": int(row.get("tactic_idx", 0)),
                "proof_techniques": labels,
            }
        )
    return rows


def _diagnostic_proof_state_text(lean_diagnostics: dict[str, Any]) -> str:
    states = lean_diagnostics.get("proof_states") or []
    pieces = []
    for state in states:
        retrieval_text = state.get("retrieval_text")
        if retrieval_text:
            pieces.append(str(retrieval_text))
            continue
        raw = state.get("raw_text")
        if raw:
            pieces.append(str(raw))
            continue
        hypotheses = "\n".join(state.get("local_hypotheses") or [])
        goal = state.get("goal_text") or ""
        pieces.append("\n".join(part for part in [hypotheses, f"⊢ {goal}" if goal else ""] if part))
    return "\n\n".join(piece for piece in pieces if piece).strip()


def _lean_extraction_metadata(lean_diagnostics: dict[str, Any]) -> dict[str, Any]:
    extraction = lean_diagnostics.get("proof_state_extraction") or {}
    states = lean_diagnostics.get("proof_states") or []
    return {
        "method": extraction.get("method", "lean_unsolved_goals_diagnostic"),
        "has_unsolved_goals": extraction.get("has_unsolved_goals", lean_diagnostics.get("summary", {}).get("has_unsolved_goals", False)),
        "raw_block_count": int(extraction.get("raw_block_count", len(states)) or 0),
        "extracted_count": int(extraction.get("extracted_count", len(states)) or 0),
        "failure_reason": extraction.get("failure_reason"),
        "rejected_block_count": len(extraction.get("rejected_blocks") or []),
    }


def retrieve_knowledge_for_theorem(
    theorem_text: str,
    full_name: str | None = None,
    k_premises: int = 20,
    k_theorems: int = 10,
    index_split: str = "train",
    input_type: InputType = "lean",
    domain_hint: str | None = None,
    file_path: str | None = None,
    validate_lean: bool = False,
) -> dict:
    query = build_query(
        theorem_text,
        input_type=input_type,
        full_name=full_name,
        domain_hint=domain_hint,
        file_path=file_path,
    )
    lean_diagnostics = (
        check_lean_syntax(theorem_text)
        if validate_lean
        else {
            "checked": False,
            "available": None,
            "ok": None,
            "proof_states": [],
            "proof_state_extraction": {
                "method": "lean_unsolved_goals_diagnostic",
                "has_unsolved_goals": False,
                "raw_block_count": 0,
                "extracted_count": 0,
                "failure_reason": "lean_validation_not_requested",
                "rejected_blocks": [],
            },
            "summary": {"has_unsolved_goals": False},
        }
    )
    diagnostic_query_text = _diagnostic_proof_state_text(lean_diagnostics)
    extraction_metadata = _lean_extraction_metadata(lean_diagnostics)
    query_text = diagnostic_query_text or query.retrieval_text
    query_source = "lean_diagnostics_proof_states" if diagnostic_query_text else "parsed_theorem_text"
    similar_theorems = retrieve_similar_theorems_for_query(query_text, k=k_theorems, index_split=index_split)
    similar_proof_states = retrieve_similar_proof_states_for_query(query_text, k=min(8, max(3, k_theorems)), index_split=index_split)
    pre_difficulty = _pre_retrieval_difficulty_estimate(query_text, query.to_dict(), similar_theorems, index_split)
    retrieval_policy = _adaptive_retrieval_policy(k_premises, requested_candidate_k=max(100, k_premises * 8), difficulty_estimate=pre_difficulty)
    premises = retrieve_premises_for_query(
        query_text,
        k=k_premises,
        index_split=index_split,
        query_kind="proof_state",
        candidate_k=retrieval_policy["candidate_k"],
        similar_theorem_ids=[row["theorem_id"] for row in similar_theorems],
        query_context=query.to_dict(),
    )
    techniques = labels_for_text(query_text)
    proof_patterns = _related_proof_patterns(similar_theorems, index_split=index_split)
    top_entities = [row["theorem_id"] for row in similar_theorems[:2]] + [row["premise_id"] for row in premises[:2]]
    graph_evidence = [get_graph_neighborhood(entity_id, depth=1, split=index_split) for entity_id in top_entities]
    return {
        "query": {
            **query.to_dict(),
            "theorem_text": theorem_text,
            "index_split": index_split,
            "embedding": _embedding_description(),
            "retrieval_query_source": query_source,
            "lean_extracted_proof_state_count": len(lean_diagnostics.get("proof_states") or []),
            "lean_proof_state_extraction": extraction_metadata,
        },
        "ranked_premises": premises,
        "similar_theorems": similar_theorems,
        "similar_proof_states": similar_proof_states,
        "likely_proof_techniques": techniques,
        "related_proof_patterns": proof_patterns,
        "difficulty_profile": _difficulty_for_query(query_text, premises, similar_theorems, index_split=index_split),
        "retrieval_policy": retrieval_policy,
        "graph_evidence": graph_evidence,
        "lean_diagnostics": lean_diagnostics,
    }
