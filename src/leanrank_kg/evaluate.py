from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .retrieve import _load_split, _rerank_premise_candidates, retrieve_knowledge_for_theorem, retrieve_premises
from .utils import SPLITS, load_config, read_json, write_json


def _average_precision(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    hits = 0
    precisions = []
    for rank, premise_id in enumerate(retrieved_ids, start=1):
        if premise_id in gold_ids:
            hits += 1
            precisions.append(hits / rank)
    return float(sum(precisions) / len(gold_ids)) if precisions else 0.0


def _ndcg(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for idx, premise_id in enumerate(retrieved_ids[:k], start=1):
        if premise_id in gold_ids:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg else 0.0


def _metric_summary(rows: list[dict], top_ks: list[int]) -> dict:
    retrievable_rows = [row for row in rows if row.get("gold_in_train_index_count", 0) > 0]
    metrics = {
        "evaluated_queries": len(rows),
        "evaluated_retrievable_queries": len(retrievable_rows),
        "gold_premises_total": int(sum(row.get("gold_total_count", 0) for row in rows)),
        "gold_premises_in_train_index": int(sum(row.get("gold_in_train_index_count", 0) for row in rows)),
        "gold_premises_missing_from_train_index": int(sum(row.get("gold_missing_from_train_index_count", 0) for row in rows)),
        "gold_premise_coverage": float(
            sum(row.get("gold_in_train_index_count", 0) for row in rows) / max(1, sum(row.get("gold_total_count", 0) for row in rows))
        )
        if rows
        else 0.0,
        "MRR": float(sum(row.get("reciprocal_rank", 0.0) for row in retrievable_rows) / len(retrievable_rows)) if retrievable_rows else 0.0,
        "MAP": float(sum(row.get("average_precision", 0.0) for row in retrievable_rows) / len(retrievable_rows)) if retrievable_rows else 0.0,
    }
    for k in top_ks:
        metrics[f"Recall@{k}"] = float(sum(row.get(f"Recall@{k}", 0.0) for row in retrievable_rows) / len(retrievable_rows)) if retrievable_rows else 0.0
        metrics[f"nDCG@{k}"] = float(sum(row.get(f"nDCG@{k}", 0.0) for row in retrievable_rows) / len(retrievable_rows)) if retrievable_rows else 0.0
    return metrics


def _domain_breakdown(rows: list[dict], top_ks: list[int], *, metric_prefix: str = "") -> list[dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if "domain_tag" not in frame.columns:
        return []
    breakdown = []
    for domain, group in frame.fillna({"domain_tag": "Unknown"}).groupby("domain_tag"):
        metrics = _metric_summary(group.to_dict(orient="records"), top_ks)
        if metric_prefix:
            metrics = {f"{metric_prefix}{key}": value for key, value in metrics.items()}
        subdomains = sorted(str(value) for value in group.get("subdomain_tag", pd.Series(dtype=str)).dropna().unique().tolist())
        breakdown.append(
            {
                "domain_tag": str(domain),
                "subdomain_tags": subdomains[:12],
                "metrics": metrics,
            }
        )
    query_key = f"{metric_prefix}evaluated_queries" if metric_prefix else "evaluated_queries"
    return sorted(breakdown, key=lambda row: int(row["metrics"].get(query_key, 0) or 0), reverse=True)


def _worst_cases(rows: list[dict], top_ks: list[int], *, id_keys: list[str], limit: int = 10) -> list[dict]:
    if not rows:
        return []
    k = max(top_ks)
    candidates = [row for row in rows if int(row.get("gold_in_train_index_count", 0) or 0) > 0]
    candidates = sorted(
        candidates,
        key=lambda row: (
            float(row.get(f"Recall@{k}", 0.0) or 0.0),
            float(row.get("reciprocal_rank", 0.0) or 0.0),
            float(row.get("average_precision", 0.0) or 0.0),
            -int(row.get("gold_in_train_index_count", 0) or 0),
        ),
    )
    fields = [
        "domain_tag",
        "subdomain_tag",
        "gold_total_count",
        "gold_in_train_index_count",
        "gold_missing_from_train_index_count",
        "rank_of_first_gold",
        f"Recall@{k}",
        "reciprocal_rank",
        "average_precision",
        f"nDCG@{k}",
    ]
    out = []
    for row in candidates[:limit]:
        item = {key: row.get(key) for key in id_keys + fields if key in row}
        out.append(item)
    return out


def _failure_profile(rows: list[dict], top_ks: list[int]) -> dict[str, Any]:
    if not rows:
        return {
            "evaluated_queries": 0,
            "retrievable_queries": 0,
            "queries_without_train_gold": 0,
            "queries_with_missing_gold": 0,
            "zero_recall_at_max_k": 0,
            "rank_buckets": {},
            "gold_coverage_buckets": {},
            "zero_recall_domains": [],
        }
    observed_top_ks = [k for k in top_ks if any(f"Recall@{k}" in row for row in rows)]
    max_k = max(observed_top_ks or top_ks)
    retrievable_rows = [row for row in rows if int(row.get("gold_in_train_index_count", 0) or 0) > 0]
    rank_buckets = {"rank_1": 0}
    if max_k >= 5:
        rank_buckets["rank_2_to_5"] = 0
    if max_k >= 10:
        rank_buckets["rank_6_to_10"] = 0
    if max_k >= 50:
        rank_buckets["rank_11_to_50"] = 0
    if max_k >= 100:
        rank_buckets["rank_51_to_100"] = 0
    rank_buckets[f"miss_top_{max_k}"] = 0
    rank_buckets["no_train_gold"] = 0
    coverage_buckets = {"full_train_gold_coverage": 0, "partial_train_gold_coverage": 0, "no_train_gold_coverage": 0}
    zero_recall_domain_counts: dict[str, int] = {}
    for row in rows:
        gold_total = int(row.get("gold_total_count", 0) or 0)
        gold_in_train = int(row.get("gold_in_train_index_count", 0) or 0)
        coverage = gold_in_train / gold_total if gold_total else 0.0
        if gold_in_train <= 0:
            rank_buckets["no_train_gold"] += 1
            coverage_buckets["no_train_gold_coverage"] += 1
        elif coverage >= 1.0:
            coverage_buckets["full_train_gold_coverage"] += 1
        else:
            coverage_buckets["partial_train_gold_coverage"] += 1
        rank = row.get("rank_of_first_gold")
        if gold_in_train > 0:
            if rank == 1:
                rank_buckets["rank_1"] += 1
            elif isinstance(rank, int) and 2 <= rank <= 5 and "rank_2_to_5" in rank_buckets:
                rank_buckets["rank_2_to_5"] += 1
            elif isinstance(rank, int) and 6 <= rank <= 10 and "rank_6_to_10" in rank_buckets:
                rank_buckets["rank_6_to_10"] += 1
            elif isinstance(rank, int) and 11 <= rank <= 50 and "rank_11_to_50" in rank_buckets:
                rank_buckets["rank_11_to_50"] += 1
            elif isinstance(rank, int) and 51 <= rank <= 100 and "rank_51_to_100" in rank_buckets:
                rank_buckets["rank_51_to_100"] += 1
            else:
                rank_buckets[f"miss_top_{max_k}"] += 1
            if float(row.get(f"Recall@{max_k}", 0.0) or 0.0) == 0.0:
                domain = str(row.get("domain_tag", "Unknown") or "Unknown")
                zero_recall_domain_counts[domain] = zero_recall_domain_counts.get(domain, 0) + 1
    return {
        "evaluated_queries": len(rows),
        "retrievable_queries": len(retrievable_rows),
        "queries_without_train_gold": len(rows) - len(retrievable_rows),
        "queries_with_missing_gold": sum(1 for row in rows if int(row.get("gold_missing_from_train_index_count", 0) or 0) > 0),
        "zero_recall_at_max_k": sum(1 for row in retrievable_rows if float(row.get(f"Recall@{max_k}", 0.0) or 0.0) == 0.0),
        "max_k": max_k,
        "rank_buckets": rank_buckets,
        "gold_coverage_buckets": coverage_buckets,
        "zero_recall_domains": [
            {"domain_tag": domain, "zero_recall_queries": count}
            for domain, count in sorted(zero_recall_domain_counts.items(), key=lambda item: item[1], reverse=True)[:12]
        ],
    }


def _candidate_miss_diagnosis(rows: list[dict], top_ks: list[int], *, ordering_k: int = 10) -> dict[str, Any]:
    if not rows:
        return {
            "method": "classify_heldout_queries_by_train_gold_availability_candidate_recall_and_topk_ordering",
            "evaluated_queries": 0,
            "retrievable_queries": 0,
            "max_k": max(top_ks) if top_ks else 0,
            "ordering_k": ordering_k,
            "bucket_counts": [],
            "top_candidate_miss_domains": [],
            "primary_failure_mode": "no_rows",
        }
    observed_top_ks = [k for k in top_ks if any(f"Recall@{k}" in row for row in rows)]
    max_k = max(observed_top_ks or top_ks)
    ordering_k = min(ordering_k, max_k)
    buckets = {
        "topk_hit": 0,
        "ordering_miss_after_topk": 0,
        "candidate_miss_at_max_k": 0,
        "no_train_gold": 0,
    }
    missing_train_gold_partial = 0
    candidate_miss_domains: dict[str, int] = {}
    total_gold = 0
    train_gold = 0
    missing_gold = 0
    retrievable = 0
    for row in rows:
        gold_total = int(row.get("gold_total_count", 0) or 0)
        gold_in_train = int(row.get("gold_in_train_index_count", 0) or 0)
        gold_missing = int(row.get("gold_missing_from_train_index_count", 0) or 0)
        total_gold += gold_total
        train_gold += gold_in_train
        missing_gold += gold_missing
        if gold_missing > 0 and gold_in_train > 0:
            missing_train_gold_partial += 1
        if gold_in_train <= 0:
            buckets["no_train_gold"] += 1
            continue
        retrievable += 1
        rank = row.get("rank_of_first_gold")
        recall_at_max = float(row.get(f"Recall@{max_k}", 0.0) or 0.0)
        if isinstance(rank, int) and rank <= ordering_k:
            buckets["topk_hit"] += 1
        elif recall_at_max > 0.0:
            buckets["ordering_miss_after_topk"] += 1
        else:
            buckets["candidate_miss_at_max_k"] += 1
            domain = str(row.get("domain_tag", "Unknown") or "Unknown")
            candidate_miss_domains[domain] = candidate_miss_domains.get(domain, 0) + 1
    denominator = max(1, len(rows))
    retrievable_denominator = max(1, retrievable)
    bucket_counts = [
        {
            "bucket": bucket,
            "query_count": int(count),
            "query_share": float(count / denominator),
            "retrievable_query_share": float(count / retrievable_denominator)
            if bucket != "no_train_gold"
            else None,
        }
        for bucket, count in buckets.items()
    ]
    if buckets["candidate_miss_at_max_k"] / retrievable_denominator >= 0.5:
        primary = "candidate_generation_or_embedding_miss"
    elif buckets["ordering_miss_after_topk"] / retrievable_denominator >= 0.25:
        primary = "candidate_ordering_after_recall"
    elif buckets["no_train_gold"] / denominator >= 0.25:
        primary = "gold_unavailable_in_train_index"
    else:
        primary = "mixed_or_monitor"
    return {
        "method": "classify_heldout_queries_by_train_gold_availability_candidate_recall_and_topk_ordering",
        "evaluated_queries": len(rows),
        "retrievable_queries": retrievable,
        "max_k": max_k,
        "ordering_k": ordering_k,
        "bucket_counts": bucket_counts,
        "missing_train_gold_partial_queries": missing_train_gold_partial,
        "gold_premises_total": int(total_gold),
        "gold_premises_in_train_index": int(train_gold),
        "gold_premises_missing_from_train_index": int(missing_gold),
        "candidate_miss_query_share_of_retrievable": float(buckets["candidate_miss_at_max_k"] / retrievable_denominator),
        "ordering_miss_query_share_of_retrievable": float(buckets["ordering_miss_after_topk"] / retrievable_denominator),
        "topk_hit_query_share_of_retrievable": float(buckets["topk_hit"] / retrievable_denominator),
        "primary_failure_mode": primary,
        "top_candidate_miss_domains": [
            {"domain_tag": domain, "query_count": count}
            for domain, count in sorted(candidate_miss_domains.items(), key=lambda item: item[1], reverse=True)[:12]
        ],
    }


def _lexical_text(*parts: Any) -> str:
    pieces = []
    for part in parts:
        if isinstance(part, (list, tuple, set, np.ndarray)):
            pieces.extend(str(value) for value in part)
        else:
            pieces.append(str(part) if part is not None else "")
    return " ".join(piece for piece in pieces if piece)


def _sparse_topk_indices(query_matrix, candidate_matrix, k: int, *, batch_size: int = 256) -> list[list[int]]:
    if query_matrix.shape[0] == 0 or candidate_matrix.shape[0] == 0:
        return []
    k = max(1, min(k, candidate_matrix.shape[0]))
    candidate_t = candidate_matrix.T.tocsr()
    out: list[list[int]] = []
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
            out.append(row.indices[order].astype(int).tolist())
    return out


def _candidate_pool_generation_diagnostic(
    *,
    query_ids: list[str],
    proof_states: pd.DataFrame,
    train_premises_frame: pd.DataFrame,
    train_premise_ids: list[str],
    embedding_retrieved_by_query: dict[str, list[str]],
    gold_by_query: dict[str, set[str]],
    train_premises: set[str],
    top_k: int,
    batch_size: int,
) -> dict[str, Any]:
    if not query_ids:
        return {
            "method": "embedding_topk_vs_lexical_topk_candidate_union",
            "evaluated_queries": 0,
            "top_k": top_k,
            "lexical_top_k": top_k,
            "metrics": {},
            "added_gold_queries": 0,
            "top_added_gold_domains": [],
        }
    premise_by_id = train_premises_frame.copy()
    premise_by_id["id"] = premise_by_id["id"].astype(str)
    premise_by_id = premise_by_id.set_index("id", drop=False)
    ordered_premises = premise_by_id.reindex(train_premise_ids).dropna(subset=["id"])
    if ordered_premises.empty:
        return {
            "method": "embedding_topk_vs_lexical_topk_candidate_union",
            "evaluated_queries": len(query_ids),
            "top_k": top_k,
            "lexical_top_k": top_k,
            "metrics": {},
            "added_gold_queries": 0,
            "top_added_gold_domains": [],
            "failure_reason": "empty_train_premise_frame",
        }
    ordered_premise_ids = [str(value) for value in ordered_premises["id"].tolist()]
    premise_texts = [
        _lexical_text(row.get("full_name"), row.get("code"), row.get("domain_tag"), row.get("subdomain_tag"), row.get("file_path"))
        for row in ordered_premises.to_dict(orient="records")
    ]
    query_texts = []
    query_info = {}
    for query_id in query_ids:
        row = proof_states.loc[query_id]
        query_texts.append(_lexical_text(row.get("full_name"), row.get("context"), row.get("goal_text"), row.get("symbols"), row.get("local_hypotheses")))
        query_info[query_id] = {
            "domain_tag": str(row.get("domain_tag", "Unknown") or "Unknown"),
            "subdomain_tag": str(row.get("subdomain_tag", "Unknown") or "Unknown"),
        }
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\w'.:]+|\S",
        min_df=1,
        max_features=300000,
        sublinear_tf=True,
        norm="l2",
    )
    premise_matrix = vectorizer.fit_transform(premise_texts)
    query_matrix = vectorizer.transform(query_texts)
    lexical_neighbor_rows = _sparse_topk_indices(query_matrix, premise_matrix, top_k, batch_size=batch_size)
    lexical_by_query = {
        query_id: [ordered_premise_ids[idx] for idx in indices]
        for query_id, indices in zip(query_ids, lexical_neighbor_rows, strict=True)
    }

    def _recall(candidate_ids: list[str], gold: set[str]) -> float:
        gold_in_index = gold & train_premises
        if not gold_in_index:
            return 0.0
        return float(len(set(candidate_ids) & gold_in_index) / len(gold_in_index))

    rows = []
    added_gold_domains: dict[str, int] = {}
    for query_id in query_ids:
        gold = gold_by_query.get(query_id, set())
        gold_in_index = gold & train_premises
        embedding_ids = embedding_retrieved_by_query.get(query_id, [])[:top_k]
        lexical_ids = lexical_by_query.get(query_id, [])[:top_k]
        union_ids = list(dict.fromkeys([*embedding_ids, *lexical_ids]))
        embedding_recall = _recall(embedding_ids, gold)
        lexical_recall = _recall(lexical_ids, gold)
        union_recall = _recall(union_ids, gold)
        added_gold = bool(gold_in_index and embedding_recall == 0.0 and union_recall > 0.0)
        if added_gold:
            domain = query_info.get(query_id, {}).get("domain_tag", "Unknown")
            added_gold_domains[domain] = added_gold_domains.get(domain, 0) + 1
        rows.append(
            {
                "proof_state_id": query_id,
                "gold_in_train_index_count": len(gold_in_index),
                "embedding_recall": embedding_recall,
                "lexical_recall": lexical_recall,
                "union_recall": union_recall,
                "embedding_hit": embedding_recall > 0.0,
                "lexical_hit": lexical_recall > 0.0,
                "union_hit": union_recall > 0.0,
                "lexical_added_gold_after_embedding_miss": added_gold,
                **query_info.get(query_id, {}),
            }
        )
    retrievable_rows = [row for row in rows if int(row["gold_in_train_index_count"]) > 0]
    denominator = max(1, len(retrievable_rows))
    metrics = {
        "retrievable_queries": len(retrievable_rows),
        "embedding_candidate_recall": float(sum(row["embedding_recall"] for row in retrievable_rows) / denominator),
        "lexical_candidate_recall": float(sum(row["lexical_recall"] for row in retrievable_rows) / denominator),
        "embedding_lexical_union_candidate_recall": float(sum(row["union_recall"] for row in retrievable_rows) / denominator),
        "embedding_hit_query_share": float(sum(1 for row in retrievable_rows if row["embedding_hit"]) / denominator),
        "lexical_hit_query_share": float(sum(1 for row in retrievable_rows if row["lexical_hit"]) / denominator),
        "union_hit_query_share": float(sum(1 for row in retrievable_rows if row["union_hit"]) / denominator),
        "lexical_added_gold_query_share": float(sum(1 for row in retrievable_rows if row["lexical_added_gold_after_embedding_miss"]) / denominator),
    }
    return {
        "method": "embedding_topk_vs_lexical_topk_candidate_union",
        "evaluated_queries": len(query_ids),
        "top_k": top_k,
        "lexical_top_k": top_k,
        "metrics": metrics,
        "added_gold_queries": int(sum(1 for row in retrievable_rows if row["lexical_added_gold_after_embedding_miss"])),
        "top_added_gold_domains": [
            {"domain_tag": domain, "query_count": count}
            for domain, count in sorted(added_gold_domains.items(), key=lambda item: item[1], reverse=True)[:12]
        ],
        "recommendation": (
            "evaluate_hybrid_lexical_embedding_candidate_union"
            if metrics["lexical_added_gold_query_share"] > 0.05
            else "lexical_union_not_primary"
        ),
    }


def _ranking_row(retrieved_ids: list[str], gold_all: set[str], train_premises: set[str], top_ks: list[int]) -> dict:
    gold_in_index = gold_all & train_premises
    gold_missing = gold_all - train_premises
    rank_of_first_gold = next((idx + 1 for idx, pid in enumerate(retrieved_ids) if pid in gold_in_index), None)
    row = {
        "gold_total_count": len(gold_all),
        "gold_in_train_index_count": len(gold_in_index),
        "gold_missing_from_train_index_count": len(gold_missing),
        "rank_of_first_gold": rank_of_first_gold,
        "reciprocal_rank": (1.0 / rank_of_first_gold) if rank_of_first_gold else 0.0,
        "average_precision": _average_precision(retrieved_ids, gold_in_index) if gold_in_index else 0.0,
    }
    for k in top_ks:
        row[f"Recall@{k}"] = len(set(retrieved_ids[:k]) & gold_in_index) / len(gold_in_index) if gold_in_index else 0.0
        row[f"nDCG@{k}"] = _ndcg(retrieved_ids, gold_in_index, k) if gold_in_index else 0.0
    return row


def _theorem_query_text(theorem: pd.Series, proof_states: pd.DataFrame) -> str:
    pieces = [str(theorem.get("full_name", ""))]
    for row in proof_states.sort_values("tactic_idx").head(6).to_dict(orient="records"):
        pieces.extend([str(row.get("context", "")), str(row.get("goal_text", ""))])
    return "\n".join(piece for piece in pieces if piece)


@lru_cache(maxsize=16)
def _embedding_ids(split: str, entity_type: str) -> list[str]:
    meta = pd.read_parquet(f"outputs/embeddings/{split}_embedding_metadata.parquet")
    rows = meta[meta["entity_type"] == entity_type].sort_values("row_index")
    return [str(value) for value in rows["entity_id"].tolist()]


@lru_cache(maxsize=16)
def _load_embedding(split: str, kind: str) -> np.ndarray:
    matrix = sparse.load_npz(f"outputs/embeddings/{split}_{kind}_embeddings.npz")
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _encode_diagnostic_queries(texts: list[str]) -> np.ndarray:
    embedding_config = read_json("outputs/embeddings/embedding_config.json", {}) or {}
    backend = str(embedding_config.get("backend", "tfidf"))
    if backend == "tfidf":
        vectorizer = joblib.load("outputs/embeddings/tfidf_vectorizer.joblib")
        matrix = vectorizer.transform(texts)
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)
    if backend in {"sentence_transformers", "sentence-transformer", "hf"}:
        model_name = embedding_config.get("model_name")
        if not model_name:
            raise ValueError("Missing model_name in outputs/embeddings/embedding_config.json")
        model = _load_diagnostic_sentence_transformer(str(model_name), str(embedding_config.get("device") or ""))
        prefix = str(embedding_config.get("query_prefix") or "")
        matrix = model.encode(
            [prefix + text for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=int(embedding_config.get("batch_size") or 128),
        )
        return np.asarray(matrix, dtype=np.float32)
    raise ValueError(f"Unknown embedding backend for diagnostics: {backend}")


@lru_cache(maxsize=4)
def _load_diagnostic_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("SentenceTransformer query representation diagnostics require sentence-transformers.") from exc
    return SentenceTransformer(model_name, device=device or None)


def _proof_state_query_variants(row: pd.Series) -> dict[str, str]:
    full_name = str(row.get("full_name", "") or "")
    context = str(row.get("context", "") or "")
    goal = str(row.get("goal_text", "") or "")
    theorem_id = str(row.get("theorem_id", "") or "")
    pieces = {
        "context_goal": "\n".join(piece for piece in [context, goal] if piece),
        "goal_only": goal,
        "full_name_goal": "\n".join(piece for piece in [full_name, goal] if piece),
        "full_name_context_goal": "\n".join(piece for piece in [full_name, context, goal] if piece),
        "theorem_id_goal": "\n".join(piece for piece in [theorem_id, goal] if piece),
    }
    return {name: text for name, text in pieces.items() if text.strip()}


def _batched_topk(
    query_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    k: int,
    *,
    batch_size: int = 256,
    use_gpu: bool = False,
    gpu_device: str = "cuda:0",
) -> tuple[list[list[int]], dict]:
    backend_info = {
        "requested_use_gpu": bool(use_gpu),
        "requested_gpu_device": gpu_device,
        "actual_backend": "none",
        "fallback_reason": None,
        "query_count": int(query_matrix.shape[0]),
        "candidate_count": int(candidate_matrix.shape[0]),
    }
    if query_matrix.shape[0] == 0 or candidate_matrix.shape[0] == 0:
        backend_info["actual_backend"] = "empty"
        return [], backend_info
    k = max(1, min(k, candidate_matrix.shape[0]))
    backend_info["top_k"] = int(k)
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device(gpu_device)
                candidates = torch.from_numpy(candidate_matrix).to(device=device, dtype=torch.float32)
                out: list[list[int]] = []
                with torch.no_grad():
                    for start in range(0, query_matrix.shape[0], batch_size):
                        queries = torch.from_numpy(query_matrix[start : start + batch_size]).to(device=device, dtype=torch.float32)
                        scores = queries @ candidates.T
                        _, indices = torch.topk(scores, k=k, dim=1)
                        out.extend(indices.cpu().numpy().astype(int).tolist())
                backend_info["actual_backend"] = "torch_cuda"
                backend_info["actual_gpu_device"] = str(device)
                return out, backend_info
            backend_info["fallback_reason"] = "torch_cuda_unavailable"
        except Exception as exc:
            backend_info["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    out = []
    candidate_t = candidate_matrix.T
    for start in range(0, query_matrix.shape[0], batch_size):
        scores = query_matrix[start : start + batch_size] @ candidate_t
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        out.extend(np.take_along_axis(top, order, axis=1).astype(int).tolist())
    backend_info["actual_backend"] = "numpy_cpu"
    return out, backend_info


def _ranking_rows_from_embeddings(
    *,
    query_ids: list[str],
    query_matrix: np.ndarray,
    gold_by_query: dict[str, set[str]],
    query_info: dict[str, dict],
    train_premise_ids: list[str],
    train_premises: set[str],
    top_ks: list[int],
    batch_size: int,
    use_gpu: bool,
    gpu_device: str,
) -> tuple[list[dict], dict[str, list[str]], dict]:
    neighbor_rows, backend_info = _batched_topk(
        query_matrix,
        _load_embedding("train", "premise"),
        max(top_ks),
        batch_size=batch_size,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    rows = []
    retrieved_by_query = {}
    for query_id, neighbor_idx in zip(query_ids, neighbor_rows, strict=True):
        retrieved_ids = [train_premise_ids[idx] for idx in neighbor_idx]
        retrieved_by_query[query_id] = retrieved_ids
        info = query_info.get(query_id, {})
        rows.append(
            {
                **info,
                **_ranking_row(retrieved_ids, gold_by_query.get(query_id, set()), train_premises, top_ks),
            }
        )
    return rows, retrieved_by_query, backend_info


def _fuse_neighbor_rows(neighbor_lists: list[list[int]], train_premise_ids: list[str], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for neighbors in neighbor_lists:
        for rank, idx in enumerate(neighbors, start=1):
            premise_id = train_premise_ids[idx]
            scores[premise_id] = scores.get(premise_id, 0.0) + 1.0 / rank
    return [premise_id for premise_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]


def _evaluate_proof_state_retrieval_split(
    split: str,
    top_ks: list[int],
    train_premises: set[str],
    *,
    max_examples: int | None = None,
    batch_size: int = 256,
    use_gpu: bool = False,
    gpu_device: str = "cuda:0",
    query_representation: str = "stored_embedding",
) -> dict:
    try:
        positive_edges = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
        proof_states_frame = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    except FileNotFoundError:
        return {"split": split, "metrics": _metric_summary([], top_ks), "examples": []}
    positive_edges["proof_state_id"] = positive_edges["proof_state_id"].astype(str)
    positive_edges["premise_id"] = positive_edges["premise_id"].astype(str)
    proof_states_frame["id"] = proof_states_frame["id"].astype(str)
    proof_states = proof_states_frame.set_index("id")
    train_premise_names = pd.read_parquet("data/processed/train/premises.parquet")
    train_premise_names["id"] = train_premise_names["id"].astype(str)
    premise_full_name = dict(zip(train_premise_names["id"], train_premise_names["full_name"], strict=False))
    train_premise_ids = _embedding_ids("train", "Premise")
    proof_state_ids = _embedding_ids(split, "ProofState")
    proof_state_row = {proof_state_id: idx for idx, proof_state_id in enumerate(proof_state_ids)}
    grouped_items = list(positive_edges.groupby("proof_state_id"))
    if max_examples is not None:
        grouped_items = grouped_items[:max_examples]
    if query_representation == "stored_embedding":
        query_ids = [str(proof_state_id) for proof_state_id, _ in grouped_items if str(proof_state_id) in proof_state_row]
    elif query_representation.startswith("stored_plus_"):
        query_ids = [str(proof_state_id) for proof_state_id, _ in grouped_items if str(proof_state_id) in proof_state_row and str(proof_state_id) in proof_states.index]
    else:
        query_ids = [str(proof_state_id) for proof_state_id, _ in grouped_items if str(proof_state_id) in proof_states.index]
    gold_by_query = {str(proof_state_id): set(group["premise_id"]) for proof_state_id, group in grouped_items}
    query_info = {
        proof_state_id: {
            "split": split,
            "proof_state_id": proof_state_id,
            "domain_tag": str(proof_states.loc[proof_state_id].get("domain_tag", "Unknown")) if proof_state_id in proof_states.index else "Unknown",
            "subdomain_tag": str(proof_states.loc[proof_state_id].get("subdomain_tag", "Unknown")) if proof_state_id in proof_states.index else "Unknown",
        }
        for proof_state_id in query_ids
    }
    if query_representation == "stored_embedding":
        query_matrix_all = _load_embedding(split, "proof_state")
        query_matrix = query_matrix_all[[proof_state_row[query_id] for query_id in query_ids]] if query_ids else query_matrix_all[:0]
    elif query_representation.startswith("stored_plus_"):
        variant_name = query_representation.removeprefix("stored_plus_")
        stored_matrix_all = _load_embedding(split, "proof_state")
        stored_query_matrix = stored_matrix_all[[proof_state_row[query_id] for query_id in query_ids]] if query_ids else stored_matrix_all[:0]
        query_texts = []
        missing_variant_count = 0
        filtered_query_ids = []
        filtered_stored_rows = []
        for query_id in query_ids:
            variants = _proof_state_query_variants(proof_states.loc[query_id])
            text = variants.get(variant_name)
            if text is None:
                missing_variant_count += 1
                continue
            filtered_query_ids.append(query_id)
            filtered_stored_rows.append(proof_state_row[query_id])
            query_texts.append(text)
        query_ids = filtered_query_ids
        stored_query_matrix = stored_matrix_all[filtered_stored_rows] if filtered_stored_rows else stored_matrix_all[:0]
        variant_query_matrix = _encode_diagnostic_queries(query_texts) if query_texts else np.zeros((0, _load_embedding("train", "premise").shape[1]), dtype=np.float32)
        train_premise_matrix = _load_embedding("train", "premise")
        stored_neighbors, stored_backend = _batched_topk(
            stored_query_matrix,
            train_premise_matrix,
            max(top_ks),
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        variant_neighbors, variant_backend = _batched_topk(
            variant_query_matrix,
            train_premise_matrix,
            max(top_ks),
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        rows = []
        retrieved_by_query = {}
        for query_id, stored_neighbor_idx, variant_neighbor_idx in zip(query_ids, stored_neighbors, variant_neighbors, strict=True):
            retrieved_ids = _fuse_neighbor_rows([stored_neighbor_idx, variant_neighbor_idx], train_premise_ids, max(top_ks))
            retrieved_by_query[query_id] = retrieved_ids
            rows.append(
                {
                    **query_info.get(query_id, {}),
                    **_ranking_row(retrieved_ids, gold_by_query.get(query_id, set()), train_premises, top_ks),
                }
            )
        backend_info = {
            **variant_backend,
            "actual_backend": f"fused_{stored_backend.get('actual_backend', 'stored')}_{variant_backend.get('actual_backend', 'query')}",
            "query_representation": query_representation,
            "fused_representations": ["stored_embedding", variant_name],
            "encoded_query_count": len(query_ids),
            "missing_query_representation_count": missing_variant_count,
            "stored_backend": stored_backend,
            "variant_backend": variant_backend,
        }
        candidate_generation_diagnostic = _candidate_pool_generation_diagnostic(
            query_ids=query_ids,
            proof_states=proof_states,
            train_premises_frame=train_premise_names,
            train_premise_ids=train_premise_ids,
            embedding_retrieved_by_query=retrieved_by_query,
            gold_by_query=gold_by_query,
            train_premises=train_premises,
            top_k=max(top_ks),
            batch_size=batch_size,
        )
        examples = []
        for proof_state_id, group in grouped_items:
            if len(examples) >= 20:
                break
            if len(examples) < 20 and proof_state_id in proof_states.index:
                pstate = proof_states.loc[proof_state_id]
                retrieved_ids = retrieved_by_query.get(str(proof_state_id), [])[:5]
                examples.append(
                    {
                        "split": split,
                        "proof_state_id": proof_state_id,
                        "proof_state": str(pstate["goal_text"])[:300],
                        "gold_positive_premises": sorted(set(group["premise_id"]))[:20],
                        "top_retrieved_premises": [
                            {
                                "premise_id": premise_id,
                                "full_name": premise_full_name.get(premise_id, premise_id),
                                "rank": idx + 1,
                                "score": 1.0 / (idx + 1),
                            }
                            for idx, premise_id in enumerate(retrieved_ids)
                        ],
                    }
                )
        return {
            "split": split,
            "metrics": _metric_summary(rows, top_ks),
            "examples": examples,
            "per_query": rows,
            "backend_info": backend_info,
            "candidate_generation_diagnostic": candidate_generation_diagnostic,
        }
    else:
        query_texts = []
        missing_variant_count = 0
        filtered_query_ids = []
        for query_id in query_ids:
            variants = _proof_state_query_variants(proof_states.loc[query_id])
            text = variants.get(query_representation)
            if text is None:
                missing_variant_count += 1
                continue
            filtered_query_ids.append(query_id)
            query_texts.append(text)
        query_ids = filtered_query_ids
        query_matrix = _encode_diagnostic_queries(query_texts) if query_texts else np.zeros((0, _load_embedding("train", "premise").shape[1]), dtype=np.float32)
    rows, retrieved_by_query, backend_info = _ranking_rows_from_embeddings(
        query_ids=query_ids,
        query_matrix=query_matrix,
        gold_by_query=gold_by_query,
        query_info=query_info,
        train_premise_ids=train_premise_ids,
        train_premises=train_premises,
        top_ks=top_ks,
        batch_size=batch_size,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    backend_info["query_representation"] = query_representation
    if query_representation != "stored_embedding":
        backend_info["encoded_query_count"] = len(query_ids)
        backend_info["missing_query_representation_count"] = missing_variant_count
    candidate_generation_diagnostic = _candidate_pool_generation_diagnostic(
        query_ids=query_ids,
        proof_states=proof_states,
        train_premises_frame=train_premise_names,
        train_premise_ids=train_premise_ids,
        embedding_retrieved_by_query=retrieved_by_query,
        gold_by_query=gold_by_query,
        train_premises=train_premises,
        top_k=max(top_ks),
        batch_size=batch_size,
    )
    examples = []
    for proof_state_id, group in grouped_items:
        if len(examples) >= 20:
            break
        if len(examples) < 20 and proof_state_id in proof_states.index:
            pstate = proof_states.loc[proof_state_id]
            retrieved_ids = retrieved_by_query.get(str(proof_state_id), [])[:5]
            examples.append(
                {
                    "split": split,
                    "proof_state_id": proof_state_id,
                    "proof_state": str(pstate["goal_text"])[:300],
                    "gold_positive_premises": sorted(set(group["premise_id"]))[:20],
                    "top_retrieved_premises": [
                        {
                            "premise_id": premise_id,
                            "full_name": premise_full_name.get(premise_id, premise_id),
                            "rank": idx + 1,
                            "score": 1.0 / (idx + 1),
                        }
                        for idx, premise_id in enumerate(retrieved_ids)
                    ],
                }
            )
    return {
        "split": split,
        "metrics": _metric_summary(rows, top_ks),
        "examples": examples,
        "per_query": rows,
        "backend_info": backend_info,
        "candidate_generation_diagnostic": candidate_generation_diagnostic,
    }


def _query_context_from_proof_state(row: pd.Series) -> dict:
    local_hypotheses = row.get("local_hypotheses", [])
    symbols = row.get("symbols", [])
    if not isinstance(local_hypotheses, list):
        local_hypotheses = []
    if not isinstance(symbols, list):
        symbols = []
    return {
        "full_name": str(row.get("full_name", "")),
        "domain_hint": str(row.get("domain_tag", "")),
        "subdomain_hint": str(row.get("subdomain_tag", "")),
        "local_hypotheses": local_hypotheses,
        "symbols": symbols,
    }


def _evaluate_reranked_proof_state_retrieval_split(
    split: str,
    top_ks: list[int],
    train_premises: set[str],
    *,
    max_examples: int,
    candidate_k: int = 100,
    candidate_k_values: list[int] | None = None,
    batch_size: int = 256,
    use_gpu: bool = False,
    gpu_device: str = "cuda:0",
) -> dict:
    if max_examples <= 0:
        return {"split": split, "metrics": _metric_summary([], top_ks), "examples": [], "per_query": [], "backend_info": {"actual_backend": "disabled"}}
    try:
        positive_edges = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
        proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    except FileNotFoundError:
        return {"split": split, "metrics": _metric_summary([], top_ks), "examples": [], "per_query": [], "backend_info": {"actual_backend": "missing_artifacts"}}
    positive_edges["proof_state_id"] = positive_edges["proof_state_id"].astype(str)
    positive_edges["premise_id"] = positive_edges["premise_id"].astype(str)
    proof_states["id"] = proof_states["id"].astype(str)
    proof_states_by_id = proof_states.set_index("id")
    grouped_items = [(str(proof_state_id), group) for proof_state_id, group in positive_edges.groupby("proof_state_id") if str(proof_state_id) in proof_states_by_id.index]
    grouped_items = grouped_items[:max_examples]
    query_ids = [proof_state_id for proof_state_id, _ in grouped_items]
    query_texts = []
    for proof_state_id in query_ids:
        pstate = proof_states_by_id.loc[proof_state_id]
        query_texts.append("\n".join(part for part in [str(pstate.get("full_name", "")), str(pstate.get("goal_text", ""))] if part))
    query_matrix = _encode_diagnostic_queries(query_texts)
    train_premise_matrix = _load_embedding("train", "premise")
    train_premise_ids = _embedding_ids("train", "Premise")
    candidate_k_values = sorted({int(value) for value in (candidate_k_values or [candidate_k]) if int(value) > 0})
    if candidate_k not in candidate_k_values:
        candidate_k_values.append(candidate_k)
        candidate_k_values = sorted(candidate_k_values)
    max_candidate_k = max(max(candidate_k_values), max(top_ks))
    neighbor_rows, backend_info = _batched_topk(
        query_matrix,
        train_premise_matrix,
        max_candidate_k,
        batch_size=batch_size,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    index_data = _load_split("train")
    premise_frame_by_id = index_data["premises"].copy()
    premise_frame_by_id["id"] = premise_frame_by_id["id"].astype(str)
    premise_frame_by_id = premise_frame_by_id.set_index("id", drop=False)
    evaluations: dict[int, dict[str, Any]] = {value: {"rows": [], "examples": []} for value in candidate_k_values}
    for query_idx, (proof_state_id, group) in enumerate(grouped_items):
        pstate = proof_states_by_id.loc[proof_state_id]
        query_text = query_texts[query_idx]
        all_neighbor_idx = neighbor_rows[query_idx] if query_idx < len(neighbor_rows) else []
        for current_candidate_k in candidate_k_values:
            neighbor_idx = all_neighbor_idx[: max(current_candidate_k, max(top_ks))]
            candidate_ids = [train_premise_ids[idx] for idx in neighbor_idx]
            candidates = premise_frame_by_id.reindex(candidate_ids).dropna(subset=["id"]).copy()
            if not candidates.empty:
                candidate_scores = query_matrix[query_idx] @ train_premise_matrix[neighbor_idx].T
                candidates["score"] = np.asarray(candidate_scores, dtype=np.float32).ravel()[: len(candidates)]
                candidates["embedding_score"] = candidates["score"]
                candidates["retrieval_backend"] = backend_info.get("actual_backend", "batched_topk")
            reranked = _rerank_premise_candidates(
                query_text,
                candidates,
                index_data,
                "train",
                query_context=_query_context_from_proof_state(pstate),
            ).head(max(top_ks))
            retrieved_ids = [str(row["id"]) for row in reranked.to_dict(orient="records")]
            evaluations[current_candidate_k]["rows"].append(
                {
                    "split": split,
                    "proof_state_id": proof_state_id,
                    "domain_tag": str(pstate.get("domain_tag", "Unknown")),
                    "subdomain_tag": str(pstate.get("subdomain_tag", "Unknown")),
                    **_ranking_row(retrieved_ids, set(group["premise_id"]), train_premises, top_ks),
                }
            )
            if current_candidate_k == candidate_k and len(evaluations[current_candidate_k]["examples"]) < 10:
                evaluations[current_candidate_k]["examples"].append(
                    {
                        "split": split,
                        "proof_state_id": proof_state_id,
                        "gold_positive_premises": sorted(set(group["premise_id"]))[:20],
                        "top_retrieved_premises": [
                            {
                                "premise_id": str(row["id"]),
                                "full_name": str(row.get("full_name", "")),
                                "score": float(row.get("score", 0.0)),
                                "embedding_score": float(row.get("embedding_score", 0.0)),
                                "learned_ranker_score": None
                                if pd.isna(row.get("learned_ranker_score"))
                                else float(row.get("learned_ranker_score")),
                            }
                            for row in reranked.head(5).to_dict(orient="records")
                        ],
                    }
                )
    rows = evaluations[candidate_k]["rows"]
    examples = evaluations[candidate_k]["examples"]
    candidate_k_ablation = [
        {
            "candidate_k": int(value),
            "metrics": _metric_summary(payload["rows"], top_ks),
        }
        for value, payload in sorted(evaluations.items())
    ]
    return {
        "split": split,
        "metrics": _metric_summary(rows, top_ks),
        "examples": examples,
        "per_query": rows,
        "backend_info": {
            **backend_info,
            "actual_backend": f"batched_{backend_info.get('actual_backend', 'topk')}_then_rerank",
            "candidate_k": int(candidate_k),
            "candidate_k_values": candidate_k_values,
            "evaluated_queries": len(rows),
        },
        "candidate_k_ablation": candidate_k_ablation,
    }


def _evaluate_proof_state_query_representations(
    split: str,
    top_ks: list[int],
    train_premises: set[str],
    *,
    max_examples: int,
    batch_size: int,
    use_gpu: bool,
    gpu_device: str,
) -> dict:
    if max_examples <= 0:
        return {"split": split, "variants": {}, "best_variant_by_recall": None, "evaluated_queries": 0}
    try:
        positive_edges = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
        proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    except FileNotFoundError:
        return {"split": split, "variants": {}, "best_variant_by_recall": None, "evaluated_queries": 0}
    positive_edges["proof_state_id"] = positive_edges["proof_state_id"].astype(str)
    positive_edges["premise_id"] = positive_edges["premise_id"].astype(str)
    proof_states["id"] = proof_states["id"].astype(str)
    proof_states_by_id = proof_states.set_index("id")
    grouped_items = [(str(proof_state_id), group) for proof_state_id, group in positive_edges.groupby("proof_state_id") if str(proof_state_id) in proof_states_by_id.index]
    grouped_items = grouped_items[:max_examples]
    query_ids = [proof_state_id for proof_state_id, _ in grouped_items]
    gold_by_query = {proof_state_id: set(group["premise_id"]) for proof_state_id, group in grouped_items}
    query_info = {
        proof_state_id: {
            "split": split,
            "proof_state_id": proof_state_id,
            "domain_tag": str(proof_states_by_id.loc[proof_state_id].get("domain_tag", "Unknown")),
            "subdomain_tag": str(proof_states_by_id.loc[proof_state_id].get("subdomain_tag", "Unknown")),
        }
        for proof_state_id in query_ids
    }
    train_premise_ids = _embedding_ids("train", "Premise")
    train_premise_matrix = _load_embedding("train", "premise")
    proof_state_embedding_ids = _embedding_ids(split, "ProofState")
    proof_state_row = {proof_state_id: idx for idx, proof_state_id in enumerate(proof_state_embedding_ids)}
    variants_by_query = {
        proof_state_id: _proof_state_query_variants(proof_states_by_id.loc[proof_state_id])
        for proof_state_id in query_ids
    }
    variant_texts: dict[str, list[str]] = {}
    for variants in variants_by_query.values():
        for variant_name, text in variants.items():
            variant_texts.setdefault(variant_name, []).append(text)
    results = {}
    stored_query_ids = [proof_state_id for proof_state_id in query_ids if proof_state_id in proof_state_row]
    stored_matrix_all = _load_embedding(split, "proof_state")
    stored_query_matrix = (
        stored_matrix_all[[proof_state_row[proof_state_id] for proof_state_id in stored_query_ids]]
        if stored_query_ids
        else np.zeros((0, train_premise_matrix.shape[1]), dtype=np.float32)
    )
    if stored_query_ids:
        stored_rows, _, stored_backend_info = _ranking_rows_from_embeddings(
            query_ids=stored_query_ids,
            query_matrix=stored_query_matrix,
            gold_by_query=gold_by_query,
            query_info=query_info,
            train_premise_ids=train_premise_ids,
            train_premises=train_premises,
            top_ks=top_ks,
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        stored_backend_info["query_representation"] = "stored_embedding"
        results["stored_embedding"] = {
            "metrics": _metric_summary(stored_rows, top_ks),
            "backend_info": stored_backend_info,
        }
    for variant_name, texts in variant_texts.items():
        if len(texts) != len(query_ids):
            continue
        query_matrix = _encode_diagnostic_queries(texts)
        rows, _, backend_info = _ranking_rows_from_embeddings(
            query_ids=query_ids,
            query_matrix=query_matrix,
            gold_by_query=gold_by_query,
            query_info=query_info,
            train_premise_ids=train_premise_ids,
            train_premises=train_premises,
            top_ks=top_ks,
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        results[variant_name] = {
            "metrics": _metric_summary(rows, top_ks),
            "backend_info": backend_info,
        }
        fused_query_ids = [
            proof_state_id
            for proof_state_id in stored_query_ids
            if variant_name in variants_by_query[proof_state_id]
        ]
        if not fused_query_ids:
            continue
        fused_texts = [variants_by_query[proof_state_id][variant_name] for proof_state_id in fused_query_ids]
        fused_stored_matrix = stored_matrix_all[[proof_state_row[proof_state_id] for proof_state_id in fused_query_ids]]
        fused_variant_matrix = _encode_diagnostic_queries(fused_texts)
        stored_neighbors, stored_backend = _batched_topk(
            fused_stored_matrix,
            train_premise_matrix,
            max(top_ks),
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        variant_neighbors, variant_backend = _batched_topk(
            fused_variant_matrix,
            train_premise_matrix,
            max(top_ks),
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        fused_rows = []
        for proof_state_id, stored_neighbor_idx, variant_neighbor_idx in zip(
            fused_query_ids,
            stored_neighbors,
            variant_neighbors,
            strict=True,
        ):
            retrieved_ids = _fuse_neighbor_rows([stored_neighbor_idx, variant_neighbor_idx], train_premise_ids, max(top_ks))
            fused_rows.append(
                {
                    **query_info.get(proof_state_id, {}),
                    **_ranking_row(retrieved_ids, gold_by_query.get(proof_state_id, set()), train_premises, top_ks),
                }
            )
        results[f"stored_plus_{variant_name}"] = {
            "metrics": _metric_summary(fused_rows, top_ks),
            "backend_info": {
                **variant_backend,
                "actual_backend": f"fused_{stored_backend.get('actual_backend', 'stored')}_{variant_backend.get('actual_backend', 'query')}",
                "query_representation": f"stored_plus_{variant_name}",
                "fused_representations": ["stored_embedding", variant_name],
                "stored_backend": stored_backend,
                "variant_backend": variant_backend,
            },
        }
    recall_key = f"Recall@{max(top_ks)}"
    best_variant = max(results, key=lambda name: float(results[name]["metrics"].get(recall_key, 0.0))) if results else None
    return {
        "split": split,
        "evaluated_queries": len(query_ids),
        "top_k": top_ks,
        "best_variant_by_recall": best_variant,
        "selection_metric": recall_key,
        "variants": results,
    }


def run(config_path: str, full_heldout: bool = False) -> None:
    evaluation_started = time.perf_counter()
    substage_timings: list[dict[str, Any]] = []

    def _record_substage(name: str, started: float, **extra: Any) -> None:
        substage_timings.append(
            {
                "name": name,
                "seconds": time.perf_counter() - started,
                **extra,
            }
        )

    _embedding_ids.cache_clear()
    _load_embedding.cache_clear()
    _load_diagnostic_sentence_transformer.cache_clear()
    config = load_config(config_path)
    report_top_ks = sorted({int(k) for k in config.get("retrieval", {}).get("top_k", [1, 5, 10])})
    train_premises = set(pd.read_parquet("data/processed/train/premises.parquet")["id"].astype(str))
    eval_config = config.get("evaluation", {}) or {}
    diagnostic_top_ks = sorted({int(k) for k in eval_config.get("candidate_pool_top_k", [50, 100])})
    top_ks = sorted(set(report_top_ks + diagnostic_top_ks))
    max_val_proof_states = eval_config.get("max_val_proof_states")
    max_val_theorems = eval_config.get("max_val_theorems")
    max_test_proof_states = eval_config.get("max_test_proof_states")
    max_test_theorems = eval_config.get("max_test_theorems")
    if full_heldout:
        max_val_proof_states = None
        max_val_theorems = None
        max_test_proof_states = None
        max_test_theorems = None
    batch_size = int(eval_config.get("batch_size", 256))
    use_gpu = bool(eval_config.get("use_gpu", False))
    gpu_device = str(eval_config.get("gpu_device", "cuda:0"))
    case_study_limit = int(eval_config.get("case_study_limit", 10))
    rerank_max_test_proof_states = int(eval_config.get("rerank_max_test_proof_states", 0) or 0)
    rerank_candidate_k = int(eval_config.get("rerank_candidate_k", 100))
    rerank_candidate_k_values = [
        int(value)
        for value in eval_config.get("rerank_candidate_k_values", [rerank_candidate_k])
        if int(value) > 0
    ]
    query_representation_diagnostic_examples = int(eval_config.get("query_representation_diagnostic_examples", 0) or 0)
    proof_state_query_representation = str(eval_config.get("proof_state_query_representation", "stored_embedding"))
    proof_state_limits = {
        "val": int(max_val_proof_states) if max_val_proof_states is not None else None,
        "test": int(max_test_proof_states) if max_test_proof_states is not None else None,
    }
    theorem_limits = {
        "val": int(max_val_theorems) if max_val_theorems is not None else None,
        "test": int(max_test_theorems) if max_test_theorems is not None else None,
    }
    proof_state_by_split = {}
    for split in ["val", "test"]:
        stage_started = time.perf_counter()
        proof_state_by_split[split] = _evaluate_proof_state_retrieval_split(
            split,
            top_ks,
            train_premises,
            max_examples=proof_state_limits.get(split),
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            query_representation=proof_state_query_representation,
        )
        _record_substage(
            f"{split}_proof_state_retrieval",
            stage_started,
            split=split,
            evaluated_queries=proof_state_by_split[split].get("metrics", {}).get("evaluated_queries"),
            actual_backend=proof_state_by_split[split].get("backend_info", {}).get("actual_backend"),
        )
    examples = list(proof_state_by_split.get("test", {}).get("examples", []))
    if len(examples) < 20:
        examples.extend(proof_state_by_split.get("val", {}).get("examples", [])[: 20 - len(examples)])
    if len(examples) < 20:
        pos = pd.read_parquet("data/processed/train/positive_edges.parquet")
        ps = pd.read_parquet("data/processed/train/proof_states.parquet").set_index("id")
        for row in pos.drop_duplicates("proof_state_id").to_dict(orient="records"):
            if len(examples) >= 20:
                break
            retrieved = retrieve_premises(row["proof_state_id"], 10, split="train")
            pstate = ps.loc[row["proof_state_id"]]
            examples.append(
                {
                    "split": "train",
                    "proof_state_id": row["proof_state_id"],
                    "proof_state": str(pstate["goal_text"])[:300],
                    "gold_positive_premise": row["premise_id"],
                    "gold_in_train_index": True,
                    "top_retrieved_premises": retrieved[:5],
                }
            )
    test_ps_metrics = proof_state_by_split.get("test", {}).get("metrics", {})
    metrics = {f"Recall@{k}": test_ps_metrics.get(f"Recall@{k}", 0.0) for k in top_ks}
    for k in top_ks:
        metrics[f"nDCG@{k}"] = test_ps_metrics.get(f"nDCG@{k}", 0.0)
    metrics["MRR"] = test_ps_metrics.get("MRR", 0.0)
    metrics["MAP"] = test_ps_metrics.get("MAP", 0.0)
    metrics["test_proof_state_evaluated_queries"] = test_ps_metrics.get("evaluated_queries", 0)
    metrics["test_proof_state_evaluated_retrievable_queries"] = test_ps_metrics.get("evaluated_retrievable_queries", 0)
    metrics["AUC"] = read_json("outputs/reports/ranker_validation_metrics.json", {}).get("validation_auc", None)
    difficulty_estimator = read_json("outputs/reports/difficulty_estimator_metrics.json", {})
    if difficulty_estimator:
        metrics["difficulty_estimator_train_mae"] = difficulty_estimator.get("train", {}).get("mae")
        metrics["difficulty_estimator_val_mae"] = difficulty_estimator.get("val", {}).get("mae")
    metrics["gold_premise_coverage"] = test_ps_metrics.get("gold_premise_coverage", 0.0)
    label_coverages = []
    for split in SPLITS:
        try:
            ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
            tech = pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet")
        except FileNotFoundError:
            continue
        if len(ps):
            label_coverages.append(float(ps["id"].isin(set(tech["proof_state_id"])).mean()))
    metrics["proof_technique_label_coverage"] = float(sum(label_coverages) / len(label_coverages)) if label_coverages else 0.0
    theorem_by_split = {}
    for split in ["val", "test"]:
        stage_started = time.perf_counter()
        theorem_by_split[split] = _evaluate_theorem_retrieval_split(
            split,
            top_ks=top_ks,
            train_premises=train_premises,
            max_theorems=theorem_limits.get(split),
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            case_study_limit=case_study_limit if split == "test" else 0,
        )
        _record_substage(
            f"{split}_theorem_retrieval",
            stage_started,
            split=split,
            evaluated_queries=theorem_by_split[split].get("metrics", {}).get("theorem_retrieval_evaluated_theorems"),
            actual_backend=theorem_by_split[split].get("backend_info", {}).get("actual_backend"),
            case_study_limit=case_study_limit if split == "test" else 0,
        )
    theorem_eval = theorem_by_split["test"]
    metrics.update(theorem_eval["metrics"])
    stage_started = time.perf_counter()
    reranked_proof_state_eval = _evaluate_reranked_proof_state_retrieval_split(
        "test",
        report_top_ks,
        train_premises,
        max_examples=rerank_max_test_proof_states,
        candidate_k=rerank_candidate_k,
        candidate_k_values=rerank_candidate_k_values,
        batch_size=batch_size,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    _record_substage(
        "test_reranked_proof_state_retrieval",
        stage_started,
        split="test",
        evaluated_queries=reranked_proof_state_eval.get("metrics", {}).get("evaluated_queries"),
        actual_backend=reranked_proof_state_eval.get("backend_info", {}).get("actual_backend"),
        candidate_k=reranked_proof_state_eval.get("backend_info", {}).get("candidate_k"),
    )
    query_representation_diagnostics = {}
    for split in ["val", "test"]:
        stage_started = time.perf_counter()
        query_representation_diagnostics[split] = _evaluate_proof_state_query_representations(
            split,
            diagnostic_top_ks,
            train_premises,
            max_examples=query_representation_diagnostic_examples,
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        _record_substage(
            f"{split}_proof_state_query_representation_diagnostic",
            stage_started,
            split=split,
            evaluated_queries=query_representation_diagnostics[split].get("evaluated_queries"),
            best_variant_by_recall=query_representation_diagnostics[split].get("best_variant_by_recall"),
        )
    for key, value in reranked_proof_state_eval.get("metrics", {}).items():
        metrics[f"reranked_proof_state_{key}"] = value
    case_studies = theorem_eval["case_studies"]
    if not case_studies and case_study_limit == 0:
        case_studies = read_json("outputs/reports/theorem_retrieval_case_studies.json", [])
    write_json("outputs/reports/theorem_retrieval_case_studies.json", case_studies)
    backend_info = {
        "proof_state": {
            split: proof_state_by_split.get(split, {}).get("backend_info", {})
            for split in ["val", "test"]
        },
        "theorem": {
            split: theorem_by_split.get(split, {}).get("backend_info", {})
            for split in ["val", "test"]
        },
    }
    test_set_evaluation = {
        "task": "premise_ranking_and_theorem_guidance",
        "candidate_pool": "train premise index",
        "label_policy": "held-out test positive_edges are used only for evaluation",
        "evaluation_scope": {
            "proof_state_limits": proof_state_limits,
            "theorem_limits": theorem_limits,
            "full_heldout_override": bool(full_heldout),
            "is_sampled": any(value is not None for value in [*proof_state_limits.values(), *theorem_limits.values()]),
            "ranking_backend": "batched_embedding_topk",
            "batch_size": batch_size,
            "use_gpu": use_gpu,
            "gpu_device": gpu_device,
            "actual_backend_info": backend_info,
            "case_study_limit": case_study_limit,
            "rerank_max_test_proof_states": rerank_max_test_proof_states,
            "rerank_candidate_k": rerank_candidate_k,
            "rerank_candidate_k_values": rerank_candidate_k_values,
            "query_representation_diagnostic_examples": query_representation_diagnostic_examples,
            "query_representation_diagnostic_splits": sorted(query_representation_diagnostics),
            "proof_state_query_representation": proof_state_query_representation,
            "total_seconds": time.perf_counter() - evaluation_started,
            "substage_timings": substage_timings,
        },
        "top_k": top_ks,
        "reported_top_k": report_top_ks,
        "candidate_pool_diagnostic_top_k": diagnostic_top_ks,
        "test": {
            "proof_state_retrieval": {
                "metrics": proof_state_by_split["test"]["metrics"],
                "domain_breakdown": _domain_breakdown(proof_state_by_split["test"].get("per_query", []), top_ks),
                "failure_profile": _failure_profile(proof_state_by_split["test"].get("per_query", []), top_ks),
                "candidate_miss_diagnosis": _candidate_miss_diagnosis(proof_state_by_split["test"].get("per_query", []), top_ks),
                "candidate_generation_diagnostic": proof_state_by_split["test"].get("candidate_generation_diagnostic", {}),
                "worst_cases": _worst_cases(proof_state_by_split["test"].get("per_query", []), top_ks, id_keys=["proof_state_id"]),
                "examples": proof_state_by_split["test"]["examples"],
            },
            "proof_state_reranked_retrieval": {
                "metrics": reranked_proof_state_eval["metrics"],
                "backend_info": reranked_proof_state_eval["backend_info"],
                "candidate_k_ablation": reranked_proof_state_eval.get("candidate_k_ablation", []),
                "domain_breakdown": _domain_breakdown(reranked_proof_state_eval.get("per_query", []), top_ks),
                "failure_profile": _failure_profile(reranked_proof_state_eval.get("per_query", []), top_ks),
                "candidate_miss_diagnosis": _candidate_miss_diagnosis(reranked_proof_state_eval.get("per_query", []), top_ks),
                "worst_cases": _worst_cases(reranked_proof_state_eval.get("per_query", []), top_ks, id_keys=["proof_state_id"]),
                "examples": reranked_proof_state_eval["examples"],
            },
            "proof_state_query_representation_diagnostic": query_representation_diagnostics.get("test", {}),
            "theorem_retrieval": {
                "metrics": theorem_by_split["test"]["metrics"],
                "domain_breakdown": _domain_breakdown(theorem_by_split["test"].get("per_query", []), top_ks, metric_prefix="theorem_retrieval_"),
                "failure_profile": _failure_profile(theorem_by_split["test"].get("per_query", []), top_ks),
                "candidate_miss_diagnosis": _candidate_miss_diagnosis(theorem_by_split["test"].get("per_query", []), top_ks),
                "worst_cases": _worst_cases(theorem_by_split["test"].get("per_query", []), top_ks, id_keys=["theorem_id", "full_name"]),
                "case_studies": case_studies,
            },
        },
        "validation": {
            "proof_state_retrieval": {
                "metrics": proof_state_by_split["val"]["metrics"],
                "domain_breakdown": _domain_breakdown(proof_state_by_split["val"].get("per_query", []), top_ks),
                "failure_profile": _failure_profile(proof_state_by_split["val"].get("per_query", []), top_ks),
                "candidate_miss_diagnosis": _candidate_miss_diagnosis(proof_state_by_split["val"].get("per_query", []), top_ks),
                "candidate_generation_diagnostic": proof_state_by_split["val"].get("candidate_generation_diagnostic", {}),
            },
            "proof_state_query_representation_diagnostic": query_representation_diagnostics.get("val", {}),
            "theorem_retrieval": {
                "metrics": theorem_by_split["val"]["metrics"],
                "domain_breakdown": _domain_breakdown(theorem_by_split["val"].get("per_query", []), top_ks, metric_prefix="theorem_retrieval_"),
                "failure_profile": _failure_profile(theorem_by_split["val"].get("per_query", []), top_ks),
                "candidate_miss_diagnosis": _candidate_miss_diagnosis(theorem_by_split["val"].get("per_query", []), top_ks),
            },
        },
    }
    write_json("outputs/reports/test_set_evaluation.json", test_set_evaluation)
    try:
        diff = pd.read_csv("outputs/reports/difficulty_distribution.csv").to_dict(orient="records")
    except FileNotFoundError:
        diff = []
    metrics["difficulty_bucket_distribution"] = diff
    write_json("outputs/reports/metrics.json", metrics)
    write_json("outputs/reports/retrieval_examples.json", examples)
    with open("outputs/reports/retrieval_examples.md", "w", encoding="utf-8") as fh:
        for ex in examples:
            gold = ex.get("gold_positive_premise") or ", ".join(ex.get("gold_positive_premises", [])[:5])
            fh.write(f"## {ex['proof_state_id']}\nGold: `{gold}`\n\n")
    domain = read_json("outputs/reports/domain_distribution.json", {})
    write_json("outputs/reports/domain_coverage.json", domain)


def _evaluate_theorem_retrieval_split(
    split: str,
    top_ks: list[int],
    train_premises: set[str],
    max_theorems: int | None = None,
    batch_size: int = 256,
    use_gpu: bool = False,
    gpu_device: str = "cuda:0",
    case_study_limit: int = 10,
) -> dict:
    total_gold_premises = 0
    total_gold_in_index = 0
    total_gold_missing = 0
    evaluated = 0
    rows = []
    case_studies = []
    try:
        theorems_frame = pd.read_parquet(f"data/processed/{split}/theorems.parquet")
        proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
        positive_edges = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    except FileNotFoundError:
        metrics = _metric_summary([], top_ks)
        return {"metrics": {f"theorem_retrieval_{key}": value for key, value in metrics.items()}, "case_studies": [], "per_query": []}
    theorems_frame["id"] = theorems_frame["id"].astype(str)
    proof_states["id"] = proof_states["id"].astype(str)
    proof_states["theorem_id"] = proof_states["theorem_id"].astype(str)
    positive_edges["proof_state_id"] = positive_edges["proof_state_id"].astype(str)
    positive_edges["premise_id"] = positive_edges["premise_id"].astype(str)
    theorems = theorems_frame.set_index("id")
    ps_to_theorem = proof_states.set_index("id")["theorem_id"].to_dict()
    pos = positive_edges.assign(theorem_id=positive_edges["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    train_premise_ids = _embedding_ids("train", "Premise")
    theorem_ids = _embedding_ids(split, "Theorem")
    theorem_row = {theorem_id: idx for idx, theorem_id in enumerate(theorem_ids)}
    grouped_items = [(str(theorem_id), group) for theorem_id, group in pos.groupby("theorem_id") if str(theorem_id) in theorem_row and str(theorem_id) in theorems.index]
    if max_theorems is not None:
        grouped_items = grouped_items[:max_theorems]
    query_ids = [theorem_id for theorem_id, _ in grouped_items]
    gold_by_query = {theorem_id: set(group["premise_id"]) for theorem_id, group in grouped_items}
    query_info = {
        theorem_id: {
            "split": split,
            "theorem_id": theorem_id,
            "full_name": theorems.loc[theorem_id].get("full_name", ""),
            "domain_tag": str(theorems.loc[theorem_id].get("domain_tag", "Unknown")),
            "subdomain_tag": str(theorems.loc[theorem_id].get("subdomain_tag", "Unknown")),
        }
        for theorem_id in query_ids
    }
    theorem_matrix_all = _load_embedding(split, "theorem")
    query_matrix = theorem_matrix_all[[theorem_row[query_id] for query_id in query_ids]] if query_ids else theorem_matrix_all[:0]
    rows, retrieved_by_query, backend_info = _ranking_rows_from_embeddings(
        query_ids=query_ids,
        query_matrix=query_matrix,
        gold_by_query=gold_by_query,
        query_info=query_info,
        train_premise_ids=train_premise_ids,
        train_premises=train_premises,
        top_ks=top_ks,
        batch_size=batch_size,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    for theorem_id, group in grouped_items:
        if theorem_id not in theorems.index:
            continue
        theorem = theorems.loc[theorem_id]
        gold_all = set(group["premise_id"])
        gold_in_index = gold_all & train_premises
        gold_missing = gold_all - train_premises
        total_gold_premises += len(gold_all)
        total_gold_in_index += len(gold_in_index)
        total_gold_missing += len(gold_missing)
        if len(case_studies) < case_study_limit:
            theorem_ps = proof_states[proof_states["theorem_id"] == theorem_id]
            query_text = _theorem_query_text(theorem, theorem_ps)
            guidance = retrieve_knowledge_for_theorem(
                theorem_text=query_text,
                full_name=str(theorem.get("full_name", "")),
                k_premises=10,
                k_theorems=5,
                index_split="train",
            )
            case_studies.append(
                {
                    "split": split,
                    "theorem_id": theorem_id,
                    "full_name": theorem.get("full_name", ""),
                    "gold_positive_premise_count": len(gold_all),
                    "gold_premises_in_train_index": len(gold_in_index),
                    "gold_premises_missing_from_train_index": len(gold_missing),
                    "gold_premise_train_coverage": len(gold_in_index) / len(gold_all) if gold_all else 0.0,
                    "guidance": guidance,
                }
            )
        evaluated += 1
    base_metrics = _metric_summary(rows, top_ks)
    metrics = {
        "theorem_retrieval_evaluated_theorems": evaluated,
        "theorem_retrieval_evaluated_theorems_with_train_gold": base_metrics["evaluated_retrievable_queries"],
        "theorem_retrieval_gold_premises_total": total_gold_premises,
        "theorem_retrieval_gold_premises_in_train_index": total_gold_in_index,
        "theorem_retrieval_gold_premises_missing_from_train_index": total_gold_missing,
        "theorem_retrieval_gold_premise_coverage": base_metrics["gold_premise_coverage"],
        "theorem_retrieval_MRR": base_metrics["MRR"],
        "theorem_retrieval_MAP": base_metrics["MAP"],
    }
    for k in top_ks:
        metrics[f"theorem_retrieval_Recall@{k}"] = base_metrics[f"Recall@{k}"]
        metrics[f"theorem_retrieval_nDCG@{k}"] = base_metrics[f"nDCG@{k}"]
    return {"metrics": metrics, "case_studies": case_studies, "per_query": rows, "backend_info": backend_info}
