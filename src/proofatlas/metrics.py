from __future__ import annotations

import math


def average_precision(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    hits = 0
    precisions = []
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in gold_ids:
            hits += 1
            precisions.append(hits / rank)
    return float(sum(precisions) / len(gold_ids)) if precisions else 0.0


def ndcg(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for idx, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in gold_ids:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg else 0.0


def ranking_row(retrieved_ids: list[str], gold_all: set[str], retrievable_ids: set[str], top_ks: list[int]) -> dict:
    gold_in_pool = gold_all & retrievable_ids
    gold_missing = gold_all - retrievable_ids
    rank_of_first_gold = next((idx + 1 for idx, item_id in enumerate(retrieved_ids) if item_id in gold_in_pool), None)
    row = {
        "gold_total_count": len(gold_all),
        "gold_in_pool_count": len(gold_in_pool),
        "gold_missing_from_pool_count": len(gold_missing),
        "rank_of_first_gold": rank_of_first_gold,
        "average_precision": average_precision(retrieved_ids, gold_in_pool) if gold_in_pool else 0.0,
    }
    for k in top_ks:
        hits_at_k = len(set(retrieved_ids[:k]) & gold_in_pool)
        row[f"hits@{k}"] = hits_at_k
        row[f"Recall@{k}"] = hits_at_k / len(gold_in_pool) if gold_in_pool else 0.0
        row[f"nDCG@{k}"] = ndcg(retrieved_ids, gold_in_pool, k) if gold_in_pool else 0.0
    return row


def summarize_rows(rows: list[dict], top_ks: list[int]) -> dict:
    retrievable = [row for row in rows if int(row.get("gold_in_pool_count", 0) or 0) > 0]
    out = {
        "evaluated_queries": len(rows),
        "evaluated_retrievable_queries": len(retrievable),
        "gold_total": int(sum(row.get("gold_total_count", 0) for row in rows)),
        "gold_in_pool": int(sum(row.get("gold_in_pool_count", 0) for row in rows)),
        "gold_missing_from_pool": int(sum(row.get("gold_missing_from_pool_count", 0) for row in rows)),
        "gold_pool_coverage": float(sum(row.get("gold_in_pool_count", 0) for row in rows) / max(1, sum(row.get("gold_total_count", 0) for row in rows))),
        "MAP": float(sum(row.get("average_precision", 0.0) for row in retrievable) / len(retrievable)) if retrievable else 0.0,
    }
    for k in top_ks:
        out[f"hits@{k}"] = int(sum(row.get(f"hits@{k}", 0) for row in rows))
        out[f"Recall@{k}"] = float(sum(row.get(f"Recall@{k}", 0.0) for row in retrievable) / len(retrievable)) if retrievable else 0.0
        out[f"AllPositiveRecall@{k}"] = float(out[f"hits@{k}"] / max(1, out["gold_total"]))
        out[f"nDCG@{k}"] = float(sum(row.get(f"nDCG@{k}", 0.0) for row in retrievable) / len(retrievable)) if retrievable else 0.0
    return out
