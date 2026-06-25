from __future__ import annotations

import numpy as np
from scipy import sparse


def sparse_topk_ids(query_matrix, candidate_matrix, candidate_ids: list[str], k: int, *, batch_size: int = 256) -> list[list[str]]:
    if query_matrix.shape[0] == 0 or candidate_matrix.shape[0] == 0:
        return []
    k = max(1, min(k, candidate_matrix.shape[0]))
    candidate_t = candidate_matrix.T.tocsr()
    out: list[list[str]] = []
    for start in range(0, query_matrix.shape[0], batch_size):
        scores = (query_matrix[start : start + batch_size] @ candidate_t).tocsr()
        for row_idx in range(scores.shape[0]):
            row = scores.getrow(row_idx)
            if row.nnz == 0:
                out.append([])
                continue
            if row.nnz <= k:
                order = np.argsort(-row.data)
            else:
                top = np.argpartition(-row.data, kth=k - 1)[:k]
                order = top[np.argsort(-row.data[top])]
            out.append([candidate_ids[int(idx)] for idx in row.indices[order]])
    return out


def dense_topk_ids(query_matrix, candidate_matrix, candidate_ids: list[str], k: int, *, batch_size: int = 256) -> list[list[str]]:
    if sparse.issparse(query_matrix):
        query_matrix = query_matrix.toarray()
    if sparse.issparse(candidate_matrix):
        candidate_matrix = candidate_matrix.toarray()
    query_matrix = np.asarray(query_matrix, dtype=np.float32)
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.float32)
    if query_matrix.shape[0] == 0 or candidate_matrix.shape[0] == 0:
        return []
    k = max(1, min(k, candidate_matrix.shape[0]))
    candidate_t = candidate_matrix.T
    out: list[list[str]] = []
    for start in range(0, query_matrix.shape[0], batch_size):
        scores = query_matrix[start : start + batch_size] @ candidate_t
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        rows = np.take_along_axis(top, order, axis=1)
        out.extend([[candidate_ids[int(idx)] for idx in row] for row in rows])
    return out


def dense_topk_items(query_matrix, candidate_matrix, candidate_ids: list[str], k: int, *, batch_size: int = 256) -> list[list[tuple[str, float]]]:
    if sparse.issparse(query_matrix):
        query_matrix = query_matrix.toarray()
    if sparse.issparse(candidate_matrix):
        candidate_matrix = candidate_matrix.toarray()
    query_matrix = np.asarray(query_matrix, dtype=np.float32)
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.float32)
    if query_matrix.shape[0] == 0 or candidate_matrix.shape[0] == 0:
        return []
    k = max(1, min(k, candidate_matrix.shape[0]))
    candidate_t = candidate_matrix.T
    out: list[list[tuple[str, float]]] = []
    for start in range(0, query_matrix.shape[0], batch_size):
        scores = query_matrix[start : start + batch_size] @ candidate_t
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        rows = np.take_along_axis(top, order, axis=1)
        row_scores = np.take_along_axis(top_scores, order, axis=1)
        out.extend(
            [
                [(candidate_ids[int(idx)], float(score)) for idx, score in zip(row, score_row, strict=True)]
                for row, score_row in zip(rows, row_scores, strict=True)
            ]
        )
    return out


def reciprocal_rank_union(*ranked_lists: list[str], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / rank
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


def weighted_reciprocal_rank_union(ranked_lists: dict[str, list[str]], weights: dict[str, float], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for source, ranked in ranked_lists.items():
        weight = float(weights.get(source, 0.0))
        if weight <= 0.0:
            continue
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / rank
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]
