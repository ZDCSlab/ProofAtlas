from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .io import write_json, write_parquet
from .llm_profiles import append_enrichment_text
from .metrics import ranking_row, summarize_rows
from .pretrained_embeddings import load_embeddings
from .profiles import premise_profile, proof_state_profile, theorem_profiles
from .vector import dense_topk_ids, dense_topk_items, reciprocal_rank_union, sparse_topk_ids, weighted_reciprocal_rank_union


TOP_KS = [10, 50, 100]
TOKEN_PATTERN = r"(?u)\b[\w'.:]+|\S"
WEIGHT_PRESETS = {
    "weighted_rrf_balanced": {"dense": 1.0, "lexical": 1.0, "symbol": 0.7, "similar_ps": 1.0},
    "weighted_rrf_lexical_ps": {"dense": 0.8, "lexical": 1.2, "symbol": 0.4, "similar_ps": 1.4},
    "weighted_rrf_ps_heavy": {"dense": 0.7, "lexical": 1.0, "symbol": 0.3, "similar_ps": 2.0},
    "weighted_rrf_ps_heavy_k50": {"dense": 0.7, "lexical": 1.0, "symbol": 0.3, "similar_ps_k50": 2.0},
    "weighted_rrf_ps_heavy_k100": {"dense": 0.7, "lexical": 1.0, "symbol": 0.3, "similar_ps_k100": 2.0},
    "weighted_rrf_ps_heavy_k100_sim": {"dense": 0.7, "lexical": 1.0, "symbol": 0.3, "similar_ps_k100_sim": 2.0},
    "weighted_rrf_ps_heavy_k100_rank_sim": {"dense": 0.7, "lexical": 1.0, "symbol": 0.3, "similar_ps_k100_rank_sim": 2.0},
    "weighted_rrf_tuned_frontier": {"dense": 1.0, "lexical": 1.2, "symbol": 0.2, "similar_ps_k100_rank_sim": 2.5},
    "weighted_rrf_tuned_recall": {"dense": 1.0, "lexical": 1.2, "symbol": 0.0, "similar_ps_k100_rank_sim": 2.5},
    "weighted_rrf_theorem_source": {"dense": 1.0, "lexical": 1.2, "symbol": 0.2, "similar_ps_k100_rank_sim": 2.5, "similar_theorem": 1.0},
    "weighted_rrf_theorem_heavy": {"dense": 1.0, "lexical": 1.2, "symbol": 0.2, "similar_ps_k100_rank_sim": 2.5, "similar_theorem": 2.0},
    "weighted_rrf_theorem_frontier": {"dense": 1.0, "lexical": 1.2, "symbol": 0.2, "similar_ps_k100_rank_sim": 2.5, "similar_theorem": 0.75},
    "weighted_rrf_theorem_tuned": {"dense": 1.0, "lexical": 1.2, "symbol": 0.2, "similar_ps_k100_rank_sim": 2.5, "similar_theorem": 2.5},
    "weighted_rrf_llm_theorem_source": {"dense": 1.0, "lexical": 1.2, "symbol": 0.2, "similar_ps_k100_rank_sim": 2.5, "similar_theorem_llm": 1.0},
    "weighted_rrf_llm_theorem_tuned": {"dense": 1.0, "lexical": 1.2, "symbol": 0.2, "similar_ps_k100_rank_sim": 2.5, "similar_theorem_llm": 2.5},
    "weighted_rrf_pretrained_tuned": {"dense": 0.6, "lexical": 1.0, "pretrained_dense": 1.4, "pretrained_similar_ps": 1.8, "similar_ps_k100_rank_sim": 1.8, "similar_theorem": 1.2},
    "weighted_rrf_llm_pretrained_tuned": {"dense": 0.6, "lexical": 1.0, "pretrained_dense": 1.4, "pretrained_similar_ps": 1.8, "similar_ps_k100_rank_sim": 1.8, "similar_theorem_llm": 2.0},
}


def _embedding_ids(split: str, entity_type: str) -> list[str]:
    meta = pd.read_parquet(f"outputs/embeddings/{split}_embedding_metadata.parquet")
    rows = meta[meta["entity_type"] == entity_type].sort_values("row_index")
    return [str(value) for value in rows["entity_id"].tolist()]


def _positive_by_query(split: str) -> dict[str, set[str]]:
    pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    pos["proof_state_id"] = pos["proof_state_id"].astype(str)
    pos["premise_id"] = pos["premise_id"].astype(str)
    return {proof_state_id: set(group["premise_id"]) for proof_state_id, group in pos.groupby("proof_state_id", sort=False)}


def _positive_by_theorem(split: str) -> dict[str, set[str]]:
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    proof_states["id"] = proof_states["id"].astype(str)
    proof_states["theorem_id"] = proof_states["theorem_id"].astype(str)
    pos["proof_state_id"] = pos["proof_state_id"].astype(str)
    pos["premise_id"] = pos["premise_id"].astype(str)
    ps_to_theorem = proof_states.set_index("id")["theorem_id"].to_dict()
    pos = pos.assign(theorem_id=pos["proof_state_id"].map(ps_to_theorem)).dropna(subset=["theorem_id"])
    return {theorem_id: set(group["premise_id"]) for theorem_id, group in pos.groupby("theorem_id", sort=False)}


def _tokens_from_values(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            text = " ".join(str(item) for item in value)
        else:
            text = str(value)
        normalized = text.replace(".", " ").replace("_", " ").replace("'", " ")
        for token in normalized.split():
            token = token.strip().lower()
            if len(token) >= 2:
                tokens.add(token)
    return tokens


def _symbol_overlap_candidates(
    query_ids: list[str],
    proof_states_by_id: pd.DataFrame,
    train_premises: pd.DataFrame,
    *,
    k: int,
) -> dict[str, list[str]]:
    premise_ids = train_premises["id"].astype(str).tolist()
    premise_texts = [
        " ".join(sorted(_tokens_from_values(row.get("full_name"), row.get("code"), row.get("file_path"))))
        for row in train_premises.to_dict(orient="records")
    ]
    query_texts = [
        " ".join(
            sorted(
                _tokens_from_values(
                    proof_states_by_id.loc[query_id].get("symbols"),
                    proof_states_by_id.loc[query_id].get("goal_text"),
                    proof_states_by_id.loc[query_id].get("local_hypotheses"),
                    proof_states_by_id.loc[query_id].get("full_name"),
                )
            )
        )
        for query_id in query_ids
    ]
    vectorizer = CountVectorizer(binary=True, lowercase=False, token_pattern=r"(?u)\b\S+\b", min_df=1, max_features=250000)
    premise_matrix = vectorizer.fit_transform(premise_texts)
    query_matrix = vectorizer.transform(query_texts)
    ranked = sparse_topk_ids(query_matrix, premise_matrix, premise_ids, k, batch_size=256)
    return dict(zip(query_ids, ranked, strict=True))


def _similar_proof_state_expansion(
    query_ids: list[str],
    query_x,
    *,
    k_states: int = 25,
    k_premises: int = 100,
    score_mode: str = "rank",
) -> dict[str, list[str]]:
    train_ps_ids = _embedding_ids("train", "ProofState")
    train_ps_x = sparse.load_npz("outputs/embeddings/train_proof_state_embeddings.npz")
    neighbor_states = dense_topk_items(query_x, train_ps_x, train_ps_ids, k_states, batch_size=256)
    train_pos = _positive_by_query("train")
    out: dict[str, list[str]] = {}
    for query_id, states in zip(query_ids, neighbor_states, strict=True):
        scores: dict[str, float] = {}
        for state_rank, (state_id, similarity) in enumerate(states, start=1):
            similarity = max(0.0, float(similarity))
            if score_mode == "rank":
                weight = 1.0 / state_rank
            elif score_mode == "sim":
                weight = similarity
            elif score_mode == "rank_sim":
                weight = similarity / state_rank
            else:
                raise ValueError(f"Unknown proof-state expansion score_mode: {score_mode}")
            for premise_id in train_pos.get(state_id, set()):
                scores[premise_id] = scores.get(premise_id, 0.0) + weight
        out[query_id] = [premise_id for premise_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k_premises]]
    return out


def _pretrained_similar_proof_state_expansion(
    query_ids: list[str],
    query_x,
    *,
    model_name: str,
    output_dir: str,
    k_states: int = 100,
    k_premises: int = 100,
) -> dict[str, list[str]]:
    train_ps_ids, train_ps_x = load_embeddings(
        split="train",
        entity_type="proof_state",
        model_name=model_name,
        output_dir=f"{output_dir}/pretrained_embeddings",
    )
    neighbor_states = dense_topk_items(query_x, train_ps_x, train_ps_ids, k_states, batch_size=128)
    train_pos = _positive_by_query("train")
    out: dict[str, list[str]] = {}
    for query_id, states in zip(query_ids, neighbor_states, strict=True):
        scores: dict[str, float] = {}
        for state_rank, (state_id, similarity) in enumerate(states, start=1):
            weight = max(0.0, float(similarity)) / state_rank
            for premise_id in train_pos.get(state_id, set()):
                scores[premise_id] = scores.get(premise_id, 0.0) + weight
        out[query_id] = [premise_id for premise_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k_premises]]
    return out


def _similar_theorem_premises(
    query_ids: list[str],
    proof_states_by_id: pd.DataFrame,
    *,
    split: str,
    k_theorems: int = 20,
    k_premises: int = 100,
    use_llm_enrichment: bool = False,
    output_dir: str = "outputs/proofatlas",
) -> dict[str, list[str]]:
    train_theorems = pd.read_parquet("data/processed/train/theorems.parquet")
    train_proof_states = pd.read_parquet("data/processed/train/proof_states.parquet")
    query_theorems = pd.read_parquet(f"data/processed/{split}/theorems.parquet")
    query_proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    train_profiles = theorem_profiles(train_theorems, train_proof_states, max_states=5)
    query_profiles = theorem_profiles(query_theorems, query_proof_states, max_states=5)
    if use_llm_enrichment:
        train_profiles = append_enrichment_text(train_profiles, "train", output_dir)
        query_profiles = append_enrichment_text(query_profiles, split, output_dir)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        min_df=1,
        max_features=300000,
        sublinear_tf=True,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(train_profiles["profile_text"].fillna("").tolist())
    query_matrix = vectorizer.transform(query_profiles["profile_text"].fillna("").tolist())
    train_theorem_ids = train_profiles["theorem_id"].astype(str).tolist()
    query_theorem_ids = query_profiles["theorem_id"].astype(str).tolist()
    neighbors_by_theorem = dict(
        zip(
            query_theorem_ids,
            sparse_topk_ids(query_matrix, train_matrix, train_theorem_ids, k_theorems, batch_size=256),
            strict=True,
        )
    )
    train_theorem_premises = _positive_by_theorem("train")
    ranked_by_theorem: dict[str, list[str]] = {}
    for theorem_id, neighbors in neighbors_by_theorem.items():
        scores: dict[str, float] = {}
        for rank, neighbor_theorem_id in enumerate(neighbors, start=1):
            weight = 1.0 / rank
            for premise_id in train_theorem_premises.get(neighbor_theorem_id, set()):
                scores[premise_id] = scores.get(premise_id, 0.0) + weight
        ranked_by_theorem[theorem_id] = [premise_id for premise_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k_premises]]
    return {query_id: ranked_by_theorem.get(str(proof_states_by_id.loc[query_id].get("theorem_id", "")), []) for query_id in query_ids}


def _evaluate_rankings(
    query_ids: list[str],
    ranked_by_query: dict[str, list[str]],
    gold_by_query: dict[str, set[str]],
    train_premises: set[str],
    query_info: dict[str, dict[str, Any]],
) -> tuple[list[dict], dict[str, Any]]:
    rows = []
    for query_id in query_ids:
        row = {
            **query_info.get(query_id, {}),
            **ranking_row(ranked_by_query.get(query_id, []), gold_by_query.get(query_id, set()), train_premises, TOP_KS),
        }
        rows.append(row)
    return rows, summarize_rows(rows, TOP_KS)


def run(
    split: str = "test",
    *,
    output_dir: str = "outputs/proofatlas",
    use_llm_enrichment: bool = False,
    use_pretrained_embeddings: bool = False,
    pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_premises = pd.read_parquet("data/processed/train/premises.parquet")
    proof_states = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    train_premises["id"] = train_premises["id"].astype(str)
    proof_states["id"] = proof_states["id"].astype(str)
    proof_states_by_id = proof_states.set_index("id", drop=False)
    train_premise_ids = _embedding_ids("train", "Premise")
    train_premise_set = set(train_premise_ids)
    gold_by_query = _positive_by_query(split)
    proof_state_embedding_ids = _embedding_ids(split, "ProofState")
    query_ids = [query_id for query_id in proof_state_embedding_ids if query_id in gold_by_query and query_id in proof_states_by_id.index]
    query_info = {
        query_id: {
            "split": split,
            "proof_state_id": query_id,
            "theorem_id": str(proof_states_by_id.loc[query_id].get("theorem_id", "")),
            "full_name": str(proof_states_by_id.loc[query_id].get("full_name", "")),
            "domain_tag": str(proof_states_by_id.loc[query_id].get("domain_tag", "")),
            "subdomain_tag": str(proof_states_by_id.loc[query_id].get("subdomain_tag", "")),
        }
        for query_id in query_ids
    }

    premise_x = sparse.load_npz("outputs/embeddings/train_premise_embeddings.npz")
    proof_state_x = sparse.load_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz")
    proof_state_row = {proof_state_id: idx for idx, proof_state_id in enumerate(proof_state_embedding_ids)}
    query_x = proof_state_x[[proof_state_row[query_id] for query_id in query_ids]]
    dense_ranked_lists = dense_topk_ids(query_x, premise_x, train_premise_ids, max(TOP_KS), batch_size=256)
    dense_by_query = dict(zip(query_ids, dense_ranked_lists, strict=True))
    if use_pretrained_embeddings:
        pretrained_premise_ids, pretrained_premise_x = load_embeddings(
            split="train",
            entity_type="premise",
            model_name=pretrained_model,
            output_dir=f"{output_dir}/pretrained_embeddings",
        )
        pretrained_proof_state_ids, pretrained_proof_state_x = load_embeddings(
            split=split,
            entity_type="proof_state",
            model_name=pretrained_model,
            output_dir=f"{output_dir}/pretrained_embeddings",
        )
        pretrained_row = {proof_state_id: idx for idx, proof_state_id in enumerate(pretrained_proof_state_ids)}
        pretrained_query_x = pretrained_proof_state_x[[pretrained_row[query_id] for query_id in query_ids]]
        pretrained_dense_lists = dense_topk_ids(
            pretrained_query_x,
            pretrained_premise_x,
            pretrained_premise_ids,
            max(TOP_KS),
            batch_size=128,
        )
        pretrained_dense_by_query = dict(zip(query_ids, pretrained_dense_lists, strict=True))
        pretrained_similar_ps_by_query = _pretrained_similar_proof_state_expansion(
            query_ids,
            pretrained_query_x,
            model_name=pretrained_model,
            output_dir=output_dir,
            k_states=100,
            k_premises=max(TOP_KS),
        )
    else:
        pretrained_dense_by_query = {query_id: [] for query_id in query_ids}
        pretrained_similar_ps_by_query = {query_id: [] for query_id in query_ids}

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        min_df=1,
        max_features=300000,
        sublinear_tf=True,
        norm="l2",
    )
    premise_texts = [premise_profile(row) for row in train_premises.to_dict(orient="records")]
    premise_lexical = vectorizer.fit_transform(premise_texts)
    query_texts = [proof_state_profile(proof_states_by_id.loc[query_id]) for query_id in query_ids]
    query_lexical = vectorizer.transform(query_texts)
    lexical_ranked_lists = sparse_topk_ids(query_lexical, premise_lexical, train_premise_ids, max(TOP_KS), batch_size=256)
    lexical_by_query = dict(zip(query_ids, lexical_ranked_lists, strict=True))
    dense_lexical_by_query = {
        query_id: reciprocal_rank_union(dense_by_query.get(query_id, []), lexical_by_query.get(query_id, []), k=max(TOP_KS))
        for query_id in query_ids
    }
    symbol_by_query = _symbol_overlap_candidates(query_ids, proof_states_by_id, train_premises, k=max(TOP_KS))
    similar_ps_by_query = _similar_proof_state_expansion(query_ids, query_x, k_states=25, k_premises=max(TOP_KS))
    similar_ps_k50_by_query = _similar_proof_state_expansion(query_ids, query_x, k_states=50, k_premises=max(TOP_KS))
    similar_ps_k100_by_query = _similar_proof_state_expansion(query_ids, query_x, k_states=100, k_premises=max(TOP_KS))
    similar_ps_k100_sim_by_query = _similar_proof_state_expansion(query_ids, query_x, k_states=100, k_premises=max(TOP_KS), score_mode="sim")
    similar_ps_k100_rank_sim_by_query = _similar_proof_state_expansion(query_ids, query_x, k_states=100, k_premises=max(TOP_KS), score_mode="rank_sim")
    similar_theorem_by_query = _similar_theorem_premises(query_ids, proof_states_by_id, split=split, k_theorems=20, k_premises=max(TOP_KS))
    similar_theorem_llm_by_query = (
        _similar_theorem_premises(
            query_ids,
            proof_states_by_id,
            split=split,
            k_theorems=20,
            k_premises=max(TOP_KS),
            use_llm_enrichment=True,
            output_dir=output_dir,
        )
        if use_llm_enrichment
        else {query_id: [] for query_id in query_ids}
    )
    full_union_by_query = {
        query_id: reciprocal_rank_union(
            dense_by_query.get(query_id, []),
            lexical_by_query.get(query_id, []),
            symbol_by_query.get(query_id, []),
            similar_ps_by_query.get(query_id, []),
            k=max(TOP_KS),
        )
        for query_id in query_ids
    }
    source_rankings_by_query = {
        query_id: {
            "dense": dense_by_query.get(query_id, []),
            "lexical": lexical_by_query.get(query_id, []),
            "symbol": symbol_by_query.get(query_id, []),
            "similar_ps": similar_ps_by_query.get(query_id, []),
            "similar_ps_k50": similar_ps_k50_by_query.get(query_id, []),
            "similar_ps_k100": similar_ps_k100_by_query.get(query_id, []),
            "similar_ps_k100_sim": similar_ps_k100_sim_by_query.get(query_id, []),
            "similar_ps_k100_rank_sim": similar_ps_k100_rank_sim_by_query.get(query_id, []),
            "similar_theorem": similar_theorem_by_query.get(query_id, []),
            "similar_theorem_llm": similar_theorem_llm_by_query.get(query_id, []),
            "pretrained_dense": pretrained_dense_by_query.get(query_id, []),
            "pretrained_similar_ps": pretrained_similar_ps_by_query.get(query_id, []),
        }
        for query_id in query_ids
    }
    active_weight_presets = {
        method: weights
        for method, weights in WEIGHT_PRESETS.items()
        if (use_llm_enrichment or not any(source.endswith("_llm") for source in weights))
        and (use_pretrained_embeddings or not any(source.startswith("pretrained_") for source in weights))
    }
    weighted_by_method = {
        method: {
            query_id: weighted_reciprocal_rank_union(source_rankings_by_query[query_id], weights, max(TOP_KS))
            for query_id in query_ids
        }
        for method, weights in active_weight_presets.items()
    }

    dense_rows, dense_metrics = _evaluate_rankings(query_ids, dense_by_query, gold_by_query, train_premise_set, query_info)
    lexical_rows, lexical_metrics = _evaluate_rankings(query_ids, lexical_by_query, gold_by_query, train_premise_set, query_info)
    dense_lexical_rows, dense_lexical_metrics = _evaluate_rankings(query_ids, dense_lexical_by_query, gold_by_query, train_premise_set, query_info)
    symbol_rows, symbol_metrics = _evaluate_rankings(query_ids, symbol_by_query, gold_by_query, train_premise_set, query_info)
    similar_ps_rows, similar_ps_metrics = _evaluate_rankings(query_ids, similar_ps_by_query, gold_by_query, train_premise_set, query_info)
    similar_ps_k50_rows, similar_ps_k50_metrics = _evaluate_rankings(query_ids, similar_ps_k50_by_query, gold_by_query, train_premise_set, query_info)
    similar_ps_k100_rows, similar_ps_k100_metrics = _evaluate_rankings(query_ids, similar_ps_k100_by_query, gold_by_query, train_premise_set, query_info)
    similar_ps_k100_sim_rows, similar_ps_k100_sim_metrics = _evaluate_rankings(query_ids, similar_ps_k100_sim_by_query, gold_by_query, train_premise_set, query_info)
    similar_ps_k100_rank_sim_rows, similar_ps_k100_rank_sim_metrics = _evaluate_rankings(query_ids, similar_ps_k100_rank_sim_by_query, gold_by_query, train_premise_set, query_info)
    similar_theorem_rows, similar_theorem_metrics = _evaluate_rankings(query_ids, similar_theorem_by_query, gold_by_query, train_premise_set, query_info)
    similar_theorem_llm_metrics = {}
    if use_llm_enrichment:
        _, similar_theorem_llm_metrics = _evaluate_rankings(query_ids, similar_theorem_llm_by_query, gold_by_query, train_premise_set, query_info)
    pretrained_dense_metrics = {}
    pretrained_similar_ps_metrics = {}
    if use_pretrained_embeddings:
        _, pretrained_dense_metrics = _evaluate_rankings(query_ids, pretrained_dense_by_query, gold_by_query, train_premise_set, query_info)
        _, pretrained_similar_ps_metrics = _evaluate_rankings(query_ids, pretrained_similar_ps_by_query, gold_by_query, train_premise_set, query_info)
    full_union_rows, full_union_metrics = _evaluate_rankings(query_ids, full_union_by_query, gold_by_query, train_premise_set, query_info)
    weighted_metrics = {}
    for method, rankings in weighted_by_method.items():
        _, weighted_metrics[method] = _evaluate_rankings(query_ids, rankings, gold_by_query, train_premise_set, query_info)
    candidate_rows = []
    for query_id in query_ids:
        candidate_rows.append(
            {
                "proof_state_id": query_id,
                "dense_top100": dense_by_query.get(query_id, []),
                "lexical_top100": lexical_by_query.get(query_id, []),
                "symbol_top100": symbol_by_query.get(query_id, []),
                "similar_proof_state_top100": similar_ps_by_query.get(query_id, []),
                "similar_proof_state_k50_top100": similar_ps_k50_by_query.get(query_id, []),
                "similar_proof_state_k100_top100": similar_ps_k100_by_query.get(query_id, []),
                "similar_proof_state_k100_sim_top100": similar_ps_k100_sim_by_query.get(query_id, []),
                "similar_proof_state_k100_rank_sim_top100": similar_ps_k100_rank_sim_by_query.get(query_id, []),
                "similar_theorem_top100": similar_theorem_by_query.get(query_id, []),
                "similar_theorem_llm_top100": similar_theorem_llm_by_query.get(query_id, []),
                "pretrained_dense_top100": pretrained_dense_by_query.get(query_id, []),
                "pretrained_similar_proof_state_top100": pretrained_similar_ps_by_query.get(query_id, []),
                "dense_lexical_rrf_top100": dense_lexical_by_query.get(query_id, []),
                "full_union_rrf_top100": full_union_by_query.get(query_id, []),
                "weighted_rrf_lexical_ps_top100": weighted_by_method["weighted_rrf_lexical_ps"].get(query_id, []),
                "weighted_rrf_tuned_frontier_top100": weighted_by_method["weighted_rrf_tuned_frontier"].get(query_id, []),
                "weighted_rrf_tuned_recall_top100": weighted_by_method["weighted_rrf_tuned_recall"].get(query_id, []),
                "weighted_rrf_theorem_frontier_top100": weighted_by_method["weighted_rrf_theorem_frontier"].get(query_id, []),
                "weighted_rrf_theorem_tuned_top100": weighted_by_method["weighted_rrf_theorem_tuned"].get(query_id, []),
                **(
                    {"weighted_rrf_llm_theorem_tuned_top100": weighted_by_method["weighted_rrf_llm_theorem_tuned"].get(query_id, [])}
                    if use_llm_enrichment
                    else {}
                ),
                **(
                    {"weighted_rrf_llm_pretrained_tuned_top100": weighted_by_method["weighted_rrf_llm_pretrained_tuned"].get(query_id, [])}
                    if use_llm_enrichment and use_pretrained_embeddings
                    else {}
                ),
                "gold_positive_premises": sorted(gold_by_query.get(query_id, set())),
                **query_info.get(query_id, {}),
            }
        )
    suffix = split
    if use_llm_enrichment:
        suffix += "_llm_enriched"
    if use_pretrained_embeddings:
        suffix += "_pretrained"
    write_parquet(pd.DataFrame(candidate_rows), f"{output_dir}/t1_{suffix}_candidate_rankings.parquet")
    report = {
        "task": "T1_proof_state_to_premise",
        "split": split,
        "query_count": len(query_ids),
        "methods": {
            "dense": dense_metrics,
            "lexical": lexical_metrics,
            "dense_lexical_rrf": dense_lexical_metrics,
            "symbol_overlap": symbol_metrics,
            "similar_proof_state_expansion": similar_ps_metrics,
            "similar_proof_state_expansion_k50": similar_ps_k50_metrics,
            "similar_proof_state_expansion_k100": similar_ps_k100_metrics,
            "similar_proof_state_expansion_k100_sim": similar_ps_k100_sim_metrics,
            "similar_proof_state_expansion_k100_rank_sim": similar_ps_k100_rank_sim_metrics,
            "similar_theorem_premises": similar_theorem_metrics,
            **({"similar_theorem_premises_llm_enriched": similar_theorem_llm_metrics} if use_llm_enrichment else {}),
            **(
                {
                    "pretrained_dense": pretrained_dense_metrics,
                    "pretrained_similar_proof_state_expansion": pretrained_similar_ps_metrics,
                }
                if use_pretrained_embeddings
                else {}
            ),
            "full_union_rrf": full_union_metrics,
            **weighted_metrics,
        },
        "use_llm_enrichment": bool(use_llm_enrichment),
        "use_pretrained_embeddings": bool(use_pretrained_embeddings),
        "pretrained_model": pretrained_model if use_pretrained_embeddings else "",
        "weight_presets": active_weight_presets,
    }
    write_json(f"{output_dir}/t1_{suffix}_proof_state_premise_retrieval.json", report)
    return report
