from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .utils import load_config, minmax, namespace, write_json

FEATURE_GROUPS = {
    "embedding_similarity": ["cosine_similarity"],
    "namespace_domain": ["same_namespace", "same_domain"],
    "proof_technique": ["proof_technique_overlap"],
    "difficulty": ["proof_state_difficulty", "negative_candidate_hardness"],
    "frequency": ["premise_frequency"],
    "symbol_overlap": ["symbol_name_overlap", "symbol_context_overlap"],
    "graph": ["graph_premise_degree"],
    "theorem_neighborhood": ["theorem_neighborhood_premise_score"],
    "candidate_source": [
        "embedding_candidate_rank_score",
        "lexical_candidate_rank_score",
        "candidate_source_overlap",
        "lexical_only_candidate",
    ],
}


def _pairs(split: str) -> pd.DataFrame:
    pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet").assign(label=1)
    neg = pd.read_parquet(f"data/processed/{split}/negative_edges.parquet").assign(label=0)
    return pd.concat([pos, neg], ignore_index=True).rename(columns={"proof_state_id": "ps", "premise_id": "prem"})


def _tokens(*parts: object) -> set[str]:
    text = " ".join(str(part or "") for part in parts).replace(".", " ").replace("_", " ")
    return {token.lower() for token in text.split() if len(token) >= 3}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _sample_pairs_by_label(pairs: pd.DataFrame, *, max_pairs_per_label: int | None, random_seed: int) -> pd.DataFrame:
    if max_pairs_per_label is None or max_pairs_per_label <= 0:
        sampled = pairs
    else:
        sampled = pairs.groupby("label", group_keys=False).sample(
            n=min(max_pairs_per_label, int(pairs.groupby("label").size().min())),
            random_state=random_seed,
        )
    return sampled.sort_values(["label", "ps", "prem"]).reset_index(drop=True)


def _lexical_text(*parts: object) -> str:
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


def _embedding_topk_indices(query_matrix, candidate_matrix, k: int, *, batch_size: int = 256) -> list[list[int]]:
    if query_matrix.shape[0] == 0 or candidate_matrix.shape[0] == 0:
        return []
    k = max(1, min(k, candidate_matrix.shape[0]))
    candidate_t = candidate_matrix.T.tocsr()
    out: list[list[int]] = []
    for start in range(0, query_matrix.shape[0], batch_size):
        scores = (query_matrix[start : start + batch_size] @ candidate_t).toarray()
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        out.extend(np.take_along_axis(top, order, axis=1).astype(int).tolist())
    return out


def _candidate_generated_pairs(
    split: str,
    *,
    max_queries: int,
    embedding_candidate_k: int,
    lexical_candidate_k: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    positive_edges = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    premises = pd.read_parquet(f"data/processed/{split}/premises.parquet")
    positive_edges["proof_state_id"] = positive_edges["proof_state_id"].astype(str)
    positive_edges["premise_id"] = positive_edges["premise_id"].astype(str)
    proof_states["id"] = proof_states["id"].astype(str)
    premises["id"] = premises["id"].astype(str)
    proof_states_by_id = proof_states.set_index("id", drop=False)
    premises_by_id = premises.set_index("id", drop=False)
    grouped = [
        (str(proof_state_id), group)
        for proof_state_id, group in positive_edges.groupby("proof_state_id")
        if str(proof_state_id) in proof_states_by_id.index
    ]
    if max_queries > 0:
        grouped = grouped[:max_queries]
    query_ids = [proof_state_id for proof_state_id, _ in grouped]
    if not query_ids:
        return pd.DataFrame(columns=["ps", "prem", "label"]), {"enabled": True, "split": split, "query_count": 0}

    ps_x = sparse.load_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz").tocsr()
    prem_x = sparse.load_npz(f"outputs/embeddings/{split}_premise_embeddings.npz").tocsr()
    ps_ids = proof_states["id"].astype(str).tolist()
    prem_ids = premises["id"].astype(str).tolist()
    ps_row = {proof_state_id: idx for idx, proof_state_id in enumerate(ps_ids)}
    query_rows = [ps_row[proof_state_id] for proof_state_id in query_ids if proof_state_id in ps_row]
    query_ids = [proof_state_id for proof_state_id in query_ids if proof_state_id in ps_row]
    query_matrix = ps_x[query_rows] if query_rows else ps_x[:0]
    embedding_neighbors = _embedding_topk_indices(query_matrix, prem_x, embedding_candidate_k, batch_size=batch_size)

    premise_texts = [
        _lexical_text(row.get("full_name"), row.get("code"), row.get("domain_tag"), row.get("subdomain_tag"), row.get("file_path"))
        for row in premises.to_dict(orient="records")
    ]
    query_texts = [
        _lexical_text(row.get("full_name"), row.get("context"), row.get("goal_text"), row.get("symbols"), row.get("local_hypotheses"))
        for row in (proof_states_by_id.loc[proof_state_id] for proof_state_id in query_ids)
    ]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\w'.:]+|\S",
        min_df=1,
        max_features=300000,
        sublinear_tf=True,
        norm="l2",
    )
    premise_lexical = vectorizer.fit_transform(premise_texts)
    query_lexical = vectorizer.transform(query_texts)
    lexical_neighbors = _sparse_topk_indices(query_lexical, premise_lexical, lexical_candidate_k, batch_size=batch_size)

    gold_by_query = {str(proof_state_id): set(group["premise_id"].astype(str)) for proof_state_id, group in grouped}
    rows = []
    lexical_added_gold_queries = 0
    union_hit_queries = 0
    for proof_state_id, embedding_idx, lexical_idx in zip(query_ids, embedding_neighbors, lexical_neighbors, strict=True):
        embedding_ids = [prem_ids[idx] for idx in embedding_idx[:embedding_candidate_k]]
        lexical_ids = [prem_ids[idx] for idx in lexical_idx[:lexical_candidate_k]]
        embedding_rank_score = {premise_id: 1.0 / rank for rank, premise_id in enumerate(embedding_ids, start=1)}
        lexical_rank_score = {premise_id: 1.0 / rank for rank, premise_id in enumerate(lexical_ids, start=1)}
        candidate_ids = list(dict.fromkeys([*embedding_ids, *lexical_ids]))
        gold = gold_by_query.get(proof_state_id, set())
        embedding_hit = bool(set(embedding_ids) & gold)
        union_hit = bool(set(candidate_ids) & gold)
        if union_hit:
            union_hit_queries += 1
        if gold and not embedding_hit and union_hit:
            lexical_added_gold_queries += 1
        for premise_id in candidate_ids:
            if premise_id not in premises_by_id.index:
                continue
            from_embedding = premise_id in embedding_rank_score
            from_lexical = premise_id in lexical_rank_score
            rows.append(
                {
                    "ps": proof_state_id,
                    "prem": premise_id,
                    "label": int(premise_id in gold),
                    "embedding_candidate_rank_score": float(embedding_rank_score.get(premise_id, 0.0)),
                    "lexical_candidate_rank_score": float(lexical_rank_score.get(premise_id, 0.0)),
                    "candidate_source_overlap": float(from_embedding and from_lexical),
                    "lexical_only_candidate": float(from_lexical and not from_embedding),
                }
            )
    frame = pd.DataFrame(rows)
    label_counts = frame["label"].value_counts().to_dict() if "label" in frame else {}
    profile = {
        "enabled": True,
        "split": split,
        "query_count": len(query_ids),
        "embedding_candidate_k": int(embedding_candidate_k),
        "lexical_candidate_k": int(lexical_candidate_k),
        "pair_count": int(len(frame)),
        "positive_pairs": int(label_counts.get(1, 0)),
        "negative_pairs": int(label_counts.get(0, 0)),
        "union_hit_query_share": float(union_hit_queries / max(1, len(query_ids))),
        "lexical_added_gold_queries": int(lexical_added_gold_queries),
        "lexical_only_pair_share": float(frame["lexical_only_candidate"].mean()) if not frame.empty else 0.0,
        "candidate_source_overlap_pair_share": float(frame["candidate_source_overlap"].mean()) if not frame.empty else 0.0,
    }
    return frame, profile


def _cosine_scores_for_pairs(
    pairs: pd.DataFrame,
    ps_x: sparse.csr_matrix,
    prem_x: sparse.csr_matrix,
    ps_row: dict[str, int],
    prem_row: dict[str, int],
) -> list[float]:
    scores = [0.0] * len(pairs)
    valid_positions = []
    valid_ps_rows = []
    valid_prem_rows = []
    for position, row in enumerate(pairs[["ps", "prem"]].itertuples(index=False)):
        ps_idx = ps_row.get(row.ps)
        prem_idx = prem_row.get(row.prem)
        if ps_idx is None or prem_idx is None:
            continue
        valid_positions.append(position)
        valid_ps_rows.append(ps_idx)
        valid_prem_rows.append(prem_idx)
    if not valid_positions:
        return scores
    raw_scores = ps_x[valid_ps_rows].multiply(prem_x[valid_prem_rows]).sum(axis=1)
    flat_scores = raw_scores.A1 if hasattr(raw_scores, "A1") else [float(value) for value in raw_scores]
    for position, score in zip(valid_positions, flat_scores, strict=True):
        scores[position] = float(score)
    return scores


def _theorem_neighbor_scores_for_pairs(
    pairs: pd.DataFrame,
    theorem_ids_by_proof_state: pd.Series,
    theorems: pd.DataFrame,
    positive_edges: pd.DataFrame,
    proof_state_to_theorem: dict[str, str],
) -> list[float]:
    if pairs.empty or positive_edges.empty:
        return [0.0] * len(pairs)

    theorem_to_premises = {
        theorem_id: set(group["premise_id"])
        for theorem_id, group in positive_edges.assign(theorem_id=positive_edges["proof_state_id"].map(proof_state_to_theorem))
        .dropna(subset=["theorem_id"])
        .groupby("theorem_id")
    }
    if not theorem_to_premises:
        return [0.0] * len(pairs)

    theorem_names = theorems["full_name"].fillna("").to_dict() if "full_name" in theorems else {}
    theorem_domains = theorems["domain_tag"].fillna("").to_dict() if "domain_tag" in theorems else {}
    domain_to_theorems: dict[str, set[str]] = defaultdict(set)
    namespace_to_theorems: dict[str, set[str]] = defaultdict(set)
    for theorem_id in theorems.index:
        theorem_id = str(theorem_id)
        domain_to_theorems[str(theorem_domains.get(theorem_id, ""))].add(theorem_id)
        namespace_to_theorems[namespace(theorem_names.get(theorem_id, ""))].add(theorem_id)

    pair_theorems = theorem_ids_by_proof_state.reindex(pairs["ps"]).fillna("").astype(str).tolist()
    candidate_premises_by_theorem: dict[str, set[str]] = defaultdict(set)
    for theorem_id, premise_id in zip(pair_theorems, pairs["prem"].astype(str), strict=True):
        if theorem_id:
            candidate_premises_by_theorem[theorem_id].add(premise_id)

    scores_by_pair: dict[tuple[str, str], float] = {}
    for theorem_id, candidate_premises in candidate_premises_by_theorem.items():
        same_area = set(domain_to_theorems.get(str(theorem_domains.get(theorem_id, "")), set()))
        same_area.update(namespace_to_theorems.get(namespace(theorem_names.get(theorem_id, "")), set()))
        same_area.discard(theorem_id)
        if not same_area:
            continue
        counts = dict.fromkeys(candidate_premises, 0)
        for other_id in same_area:
            for premise_id in theorem_to_premises.get(other_id, set()) & candidate_premises:
                counts[premise_id] += 1
        denom = len(same_area)
        for premise_id, count in counts.items():
            if count:
                scores_by_pair[(theorem_id, premise_id)] = min(count / denom, 1.0)

    return [float(scores_by_pair.get((theorem_id, str(premise_id)), 0.0)) for theorem_id, premise_id in zip(pair_theorems, pairs["prem"], strict=True)]


def _features(pairs: pd.DataFrame, split: str, *, max_pairs_per_label: int | None = 1000, random_seed: int = 0) -> pd.DataFrame:
    pairs = _sample_pairs_by_label(pairs, max_pairs_per_label=max_pairs_per_label, random_seed=random_seed)
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet").set_index("id")
    premises = pd.read_parquet(f"data/processed/{split}/premises.parquet").set_index("id")
    theorems = pd.read_parquet(f"data/processed/{split}/theorems.parquet").set_index("id")
    ps_features = pd.read_parquet(f"data/processed/{split}/proof_state_features.parquet").set_index("id")
    ps_tech = pd.read_parquet(f"data/processed/{split}/proof_state_techniques.parquet")
    prem_tech = pd.read_parquet(f"data/processed/{split}/premise_techniques.parquet")
    ps_labels = {key: set(group["label"]) for key, group in ps_tech.groupby("proof_state_id")} if not ps_tech.empty else {}
    prem_labels = {key: set(group["label"]) for key, group in prem_tech.groupby("premise_id")} if not prem_tech.empty else {}
    ps_x = sparse.load_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz").tocsr()
    prem_x = sparse.load_npz(f"outputs/embeddings/{split}_premise_embeddings.npz").tocsr()
    ps_row = {pid: i for i, pid in enumerate(proof_states.index.tolist())}
    prem_row = {pid: i for i, pid in enumerate(premises.index.tolist())}
    positive_edges = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    premise_frequency = positive_edges.groupby("premise_id").size().rename("premise_frequency_raw")
    if premise_frequency.empty:
        premise_frequency_norm = premise_frequency
    else:
        premise_frequency_norm = minmax(premise_frequency).rename("premise_frequency")
    ps_to_theorem = proof_states["theorem_id"].to_dict()
    premise_theorem_edges = positive_edges.assign(theorem_id=positive_edges["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    theorem_premise_counts = premise_theorem_edges.groupby("premise_id")["theorem_id"].nunique()
    theorem_premise_degree = minmax(theorem_premise_counts) if not theorem_premise_counts.empty else theorem_premise_counts
    ps_defaults = {"full_name": "", "domain_tag": "", "context": "", "goal_text": "", "theorem_id": ""}
    prem_defaults = {"full_name": "", "domain_tag": "", "code": ""}
    unique_ps = sorted(set(pairs["ps"].astype(str)))
    unique_premises = sorted(set(pairs["prem"].astype(str)))
    ps_cache = {
        proof_state_id: (proof_states.loc[proof_state_id] if proof_state_id in proof_states.index else pd.Series(ps_defaults))
        for proof_state_id in unique_ps
    }
    prem_cache = {
        premise_id: (premises.loc[premise_id] if premise_id in premises.index else pd.Series(prem_defaults))
        for premise_id in unique_premises
    }
    ps_tokens = {
        proof_state_id: _tokens(row.get("full_name", ""), row.get("context", ""), row.get("goal_text", ""))
        for proof_state_id, row in ps_cache.items()
    }
    premise_name_tokens = {premise_id: _tokens(row.get("full_name", "")) for premise_id, row in prem_cache.items()}
    premise_context_tokens = {
        premise_id: _tokens(row.get("full_name", ""), row.get("code", ""))
        for premise_id, row in prem_cache.items()
    }
    cosine_scores = _cosine_scores_for_pairs(pairs, ps_x, prem_x, ps_row, prem_row)
    theorem_neighbor_scores = _theorem_neighbor_scores_for_pairs(
        pairs,
        proof_states["theorem_id"],
        theorems,
        positive_edges,
        ps_to_theorem,
    )
    rows = []
    for idx, row in enumerate(pairs.to_dict(orient="records")):
        proof_state_id = str(row["ps"])
        premise_id = str(row["prem"])
        ps_record = ps_cache[proof_state_id]
        prem_record = prem_cache[premise_id]
        ps_feature_row = ps_features.loc[proof_state_id] if proof_state_id in ps_features.index else {}
        shared_labels = ps_labels.get(proof_state_id, set()) & prem_labels.get(premise_id, set())
        rows.append(
            {
                "cosine_similarity": cosine_scores[idx],
                "same_namespace": float(namespace(ps_record.get("full_name", "")) == namespace(prem_record.get("full_name", ""))),
                "same_domain": float(ps_record.get("domain_tag", "") == prem_record.get("domain_tag", "")),
                "proof_technique_overlap": float(bool(shared_labels)),
                "proof_state_difficulty": float(ps_feature_row.get("difficulty_score", 0.0)),
                "negative_candidate_hardness": float(ps_feature_row.get("negative_candidate_hardness", 0.0)),
                "premise_frequency": float(premise_frequency_norm.get(premise_id, 0.0)) if not premise_frequency_norm.empty else 0.0,
                "symbol_name_overlap": _jaccard(ps_tokens[proof_state_id], premise_name_tokens[premise_id]),
                "symbol_context_overlap": _jaccard(ps_tokens[proof_state_id], premise_context_tokens[premise_id]),
                "graph_premise_degree": float(theorem_premise_degree.get(premise_id, 0.0)) if not theorem_premise_degree.empty else 0.0,
                "theorem_neighborhood_premise_score": theorem_neighbor_scores[idx],
                "embedding_candidate_rank_score": float(row.get("embedding_candidate_rank_score", 0.0)),
                "lexical_candidate_rank_score": float(row.get("lexical_candidate_rank_score", 0.0)),
                "candidate_source_overlap": float(row.get("candidate_source_overlap", 0.0)),
                "lexical_only_candidate": float(row.get("lexical_only_candidate", 0.0)),
                "label": row["label"],
            }
        )
    return pd.DataFrame(rows)


def _auc_for_columns(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]) -> float | None:
    if not columns or validation.empty or validation["label"].nunique() <= 1 or train["label"].nunique() <= 1:
        return None
    model = LogisticRegression(max_iter=500).fit(train[columns], train["label"])
    return float(roc_auc_score(validation["label"], model.predict_proba(validation[columns])[:, 1]))


def _ablation_report(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    feature_columns = [col for col in train.columns if col != "label"]
    full_auc = _auc_for_columns(train, validation, feature_columns)
    groups = {}
    for group, columns in FEATURE_GROUPS.items():
        available = [col for col in columns if col in feature_columns]
        remaining = [col for col in feature_columns if col not in available]
        removed_auc = _auc_for_columns(train, validation, remaining)
        only_auc = _auc_for_columns(train, validation, available)
        groups[group] = {
            "columns": available,
            "auc_without_group": removed_auc,
            "auc_group_only": only_auc,
            "delta_without_group": None if full_auc is None or removed_auc is None else float(full_auc - removed_auc),
        }
    return {
        "full_auc": full_auc,
        "feature_columns": feature_columns,
        "groups": groups,
    }


def _pair_utilization_profile(raw_pairs: pd.DataFrame, train_features: pd.DataFrame) -> dict:
    raw_label_counts = raw_pairs["label"].value_counts().to_dict() if "label" in raw_pairs else {}
    train_label_counts = train_features["label"].value_counts().to_dict() if "label" in train_features else {}
    positive_pairs = int(train_label_counts.get(1, 0))
    negative_pairs = int(train_label_counts.get(0, 0))
    hardness = (
        train_features.get("negative_candidate_hardness", pd.Series(dtype=float))
        .fillna(0.0)
        .astype(float)
        .clip(0.0, 1.0)
        if not train_features.empty
        else pd.Series(dtype=float)
    )
    negative_mask = train_features["label"].eq(0) if "label" in train_features else pd.Series(dtype=bool)
    negative_hardness = hardness[negative_mask] if len(hardness) else pd.Series(dtype=float)
    feature_columns = [col for col in train_features.columns if col != "label"]
    nonzero_rates = {
        col: float((train_features[col].fillna(0.0).astype(float) != 0.0).mean())
        for col in feature_columns
        if col in train_features
    }
    return {
        "label_source": {
            "positive_pairs": "data/processed/train/positive_edges.parquet label=1",
            "hard_negative_pairs": "data/processed/train/negative_edges.parquet label=0",
        },
        "raw_pair_counts": {
            "positive": int(raw_label_counts.get(1, 0)),
            "hard_negative": int(raw_label_counts.get(0, 0)),
            "total": int(len(raw_pairs)),
        },
        "training_sample_counts": {
            "positive": positive_pairs,
            "hard_negative": negative_pairs,
            "total": int(len(train_features)),
            "hard_negative_to_positive_ratio": float(negative_pairs / max(1, positive_pairs)),
        },
        "hardness_feature": {
            "column": "negative_candidate_hardness",
            "all_pair_nonzero_count": int((hardness > 0.0).sum()) if len(hardness) else 0,
            "all_pair_nonzero_share": float((hardness > 0.0).mean()) if len(hardness) else 0.0,
            "negative_pair_nonzero_count": int((negative_hardness > 0.0).sum()) if len(negative_hardness) else 0,
            "negative_pair_nonzero_share": float((negative_hardness > 0.0).mean()) if len(negative_hardness) else 0.0,
            "negative_pair_mean_hardness": float(negative_hardness.mean()) if len(negative_hardness) else 0.0,
        },
        "feature_nonzero_rates": nonzero_rates,
    }


def run(config_path: str) -> None:
    config = load_config(config_path)
    ranker_config = config.get("ranker", {}) or {}
    random_seed = int(config.get("random_seed", 0))
    max_train_pairs_per_label = int(ranker_config.get("max_train_pairs_per_label", 1000) or 0)
    max_validation_pairs_per_label = int(
        ranker_config.get("max_validation_pairs_per_label", max_train_pairs_per_label) or 0
    )
    use_candidate_generated_pairs = bool(ranker_config.get("use_candidate_generated_pairs", False))
    candidate_query_limit = int(ranker_config.get("candidate_generated_query_limit", 0) or 0)
    candidate_embedding_k = int(ranker_config.get("candidate_generated_embedding_k", 50) or 50)
    candidate_lexical_k = int(ranker_config.get("candidate_generated_lexical_k", candidate_embedding_k) or candidate_embedding_k)
    candidate_batch_size = int(ranker_config.get("candidate_generated_batch_size", 256) or 256)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_train_pairs = _pairs("train")
    candidate_pair_profiles = {}
    train_pair_source = "raw_positive_negative_edges"
    if use_candidate_generated_pairs:
        candidate_started = time.perf_counter()
        generated_train, candidate_pair_profiles["train"] = _candidate_generated_pairs(
            "train",
            max_queries=candidate_query_limit,
            embedding_candidate_k=candidate_embedding_k,
            lexical_candidate_k=candidate_lexical_k,
            batch_size=candidate_batch_size,
        )
        candidate_pair_profiles["train"]["seconds"] = time.perf_counter() - candidate_started
        if not generated_train.empty and generated_train["label"].nunique() > 1:
            raw_train_pairs = generated_train
            train_pair_source = "candidate_generated_embedding_lexical_union"
    train_feature_started = time.perf_counter()
    train = _features(
        raw_train_pairs,
        "train",
        max_pairs_per_label=max_train_pairs_per_label,
        random_seed=random_seed,
    )
    train_feature_seconds = time.perf_counter() - train_feature_started
    X = train.drop(columns=["label"])
    y = train["label"]
    model_fit_started = time.perf_counter()
    model = LogisticRegression(max_iter=500).fit(X, y)
    model_fit_seconds = time.perf_counter() - model_fit_started
    joblib.dump(model, "outputs/models/premise_ranker.joblib")
    metrics = {
        "train_pairs": int(len(train)),
        "ranker_config": {
            "max_train_pairs_per_label": max_train_pairs_per_label,
            "max_validation_pairs_per_label": max_validation_pairs_per_label,
            "random_seed": random_seed,
            "use_candidate_generated_pairs": use_candidate_generated_pairs,
            "candidate_generated_query_limit": candidate_query_limit,
            "candidate_generated_embedding_k": candidate_embedding_k,
            "candidate_generated_lexical_k": candidate_lexical_k,
            "candidate_generated_batch_size": candidate_batch_size,
        },
        "training_pair_source": train_pair_source,
        "candidate_generated_pair_profile": candidate_pair_profiles,
        "feature_columns": X.columns.tolist(),
        "feature_groups": FEATURE_GROUPS,
        "training_pair_utilization": _pair_utilization_profile(raw_train_pairs, train),
        "timing_profile": {
            "train_feature_seconds": train_feature_seconds,
            "model_fit_seconds": model_fit_seconds,
        },
    }
    try:
        raw_val_pairs = _pairs("val")
        if use_candidate_generated_pairs:
            candidate_val_started = time.perf_counter()
            generated_val, candidate_pair_profiles["val"] = _candidate_generated_pairs(
                "val",
                max_queries=max(1, candidate_query_limit // 4) if candidate_query_limit > 0 else 0,
                embedding_candidate_k=candidate_embedding_k,
                lexical_candidate_k=candidate_lexical_k,
                batch_size=candidate_batch_size,
            )
            candidate_pair_profiles["val"]["seconds"] = time.perf_counter() - candidate_val_started
            if not generated_val.empty and generated_val["label"].nunique() > 1:
                raw_val_pairs = generated_val
        validation_feature_started = time.perf_counter()
        val = _features(
            raw_val_pairs,
            "val",
            max_pairs_per_label=max_validation_pairs_per_label,
            random_seed=random_seed,
        )
        metrics["timing_profile"]["validation_feature_seconds"] = time.perf_counter() - validation_feature_started
        if not val.empty and val["label"].nunique() > 1:
            metrics["validation_auc"] = float(roc_auc_score(val["label"], model.predict_proba(val.drop(columns=["label"]))[:, 1]))
            metrics["validation_pairs"] = int(len(val))
            ablation_started = time.perf_counter()
            metrics["feature_ablation"] = _ablation_report(train, val)
            metrics["timing_profile"]["feature_ablation_seconds"] = time.perf_counter() - ablation_started
    except Exception as exc:
        metrics["validation_error"] = str(exc)
    metrics["timing_profile"]["total_seconds"] = time.perf_counter() - started
    write_json("outputs/reports/ranker_validation_metrics.json", metrics)
