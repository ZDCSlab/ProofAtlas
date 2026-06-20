from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import SPLITS, load_config, write_json, write_parquet


def _texts(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ps = pd.read_parquet(f"data/processed/{split}/proof_states.parquet")
    prem = pd.read_parquet(f"data/processed/{split}/premises.parquet")
    ps_text = ps["context"].fillna("") + " " + ps["goal_text"].fillna("")
    prem_text = prem["full_name"].fillna("") + " " + prem["code"].fillna("") + " " + prem["file_path"].fillna("")
    return ps.assign(_text=ps_text), prem.assign(_text=prem_text)


def _write_embeddings(split: str, ps: pd.DataFrame, prem: pd.DataFrame, ps_x, prem_x) -> None:
    sparse.save_npz(f"outputs/embeddings/{split}_proof_state_embeddings.npz", sparse.csr_matrix(ps_x))
    sparse.save_npz(f"outputs/embeddings/{split}_premise_embeddings.npz", sparse.csr_matrix(prem_x))
    ps_x = sparse.csr_matrix(ps_x)
    prem_x = sparse.csr_matrix(prem_x)
    theorem_ids = ps["theorem_id"].drop_duplicates().tolist()
    ps_to_thm = ps.set_index("id")["theorem_id"].to_dict()
    prem_row = {pid: i for i, pid in enumerate(prem["id"].tolist())}
    try:
        pos = pd.read_parquet(f"data/processed/{split}/positive_edges.parquet")
    except FileNotFoundError:
        pos = pd.DataFrame(columns=["proof_state_id", "premise_id"])
    theorem_positive_premise_rows: dict[str, list[int]] = {}
    for row in pos.to_dict(orient="records"):
        tid = ps_to_thm.get(row["proof_state_id"])
        pidx = prem_row.get(row["premise_id"])
        if tid is not None and pidx is not None:
            theorem_positive_premise_rows.setdefault(tid, []).append(pidx)
    thm_x = []
    for tid in theorem_ids:
        ps_idx = ps.index[ps["theorem_id"] == tid].tolist()
        pieces = [ps_x[ps_idx]]
        prem_idx = sorted(set(theorem_positive_premise_rows.get(tid, [])))
        if prem_idx:
            pieces.append(prem_x[prem_idx])
        thm_x.append(sparse.csr_matrix(sparse.vstack(pieces).mean(axis=0)))
    sparse.save_npz(f"outputs/embeddings/{split}_theorem_embeddings.npz", sparse.vstack(thm_x) if thm_x else sparse.csr_matrix((0, ps_x.shape[1])))
    meta = pd.concat(
        [
            pd.DataFrame({"entity_id": ps["id"], "entity_type": "ProofState", "row_index": range(len(ps))}),
            pd.DataFrame({"entity_id": prem["id"], "entity_type": "Premise", "row_index": range(len(prem))}),
            pd.DataFrame({"entity_id": theorem_ids, "entity_type": "Theorem", "row_index": range(len(theorem_ids))}),
        ],
        ignore_index=True,
    )
    write_parquet(meta, f"outputs/embeddings/{split}_embedding_metadata.parquet")


def _run_tfidf() -> None:
    Path("outputs/embeddings").mkdir(parents=True, exist_ok=True)
    train_ps, train_prem = _texts("train")
    vectorizer = TfidfVectorizer(max_features=2048, ngram_range=(1, 2), min_df=1)
    vectorizer.fit(pd.concat([train_ps["_text"], train_prem["_text"]], ignore_index=True))
    joblib.dump(vectorizer, "outputs/embeddings/tfidf_vectorizer.joblib")
    for split in SPLITS + ["demo"]:
        try:
            ps, prem = _texts(split)
        except FileNotFoundError:
            continue
        ps_x = vectorizer.transform(ps["_text"])
        prem_x = vectorizer.transform(prem["_text"])
        _write_embeddings(split, ps, prem, ps_x, prem_x)


def _run_sentence_transformers(
    model_name: str,
    device: str | None = None,
    batch_size: int = 128,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformer embeddings require the optional dependency: "
            "pip install -e '.[hf]'"
        ) from exc
    Path("outputs/embeddings").mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    for split in SPLITS + ["demo"]:
        try:
            ps, prem = _texts(split)
        except FileNotFoundError:
            continue
        ps_texts = [query_prefix + text for text in ps["_text"].tolist()]
        prem_texts = [passage_prefix + text for text in prem["_text"].tolist()]
        ps_x = model.encode(ps_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
        prem_x = model.encode(prem_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
        _write_embeddings(split, ps, prem, ps_x, prem_x)


def run(config_path: str) -> None:
    config = load_config(config_path)
    embedding_config = config.get("embedding", {})
    backend = embedding_config.get("backend", "tfidf")
    model_name = embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    device = embedding_config.get("device")
    batch_size = int(embedding_config.get("batch_size", 128))
    query_prefix = embedding_config.get("query_prefix", "")
    passage_prefix = embedding_config.get("passage_prefix", "")
    if backend == "tfidf":
        _run_tfidf()
    elif backend in {"sentence_transformers", "sentence-transformer", "hf"}:
        _run_sentence_transformers(model_name, device=device, batch_size=batch_size, query_prefix=query_prefix, passage_prefix=passage_prefix)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
    write_json(
        "outputs/embeddings/embedding_config.json",
        {
            "backend": backend,
            "model_name": model_name if backend != "tfidf" else None,
            "device": device if backend != "tfidf" else None,
            "batch_size": batch_size if backend != "tfidf" else None,
            "query_prefix": query_prefix if backend != "tfidf" else None,
            "passage_prefix": passage_prefix if backend != "tfidf" else None,
        },
    )
