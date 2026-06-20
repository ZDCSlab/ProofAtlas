from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .utils import SPLITS, write_json

FEATURE_COLUMNS = [
    "context_length_score",
    "num_local_hypotheses",
    "num_positive_premises",
    "avg_positive_premise_length",
    "premise_namespace_rarity",
    "tactic_step_index_score",
    "negative_candidate_hardness",
]
TARGET_COLUMN = "theorem_complexity_score"
FALLBACK_TARGET_COLUMN = "difficulty_score"


def _load_features(split: str) -> pd.DataFrame:
    frame = pd.read_parquet(f"data/processed/{split}/proof_state_features.parquet")
    target_col = TARGET_COLUMN if TARGET_COLUMN in frame.columns else FALLBACK_TARGET_COLUMN
    missing = [col for col in FEATURE_COLUMNS + [target_col] if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing difficulty feature columns for {split}: {missing}")
    out = frame[FEATURE_COLUMNS + [target_col]].copy()
    out["difficulty_target"] = out[target_col].astype(float)
    return out[FEATURE_COLUMNS + ["difficulty_target"]].fillna(0.0)


def _evaluate(model: RandomForestRegressor, split: str) -> dict[str, Any]:
    try:
        frame = _load_features(split)
    except FileNotFoundError:
        return {"available": False, "rows": 0}
    if frame.empty:
        return {"available": True, "rows": 0}
    y_true = frame["difficulty_target"].astype(float)
    y_pred = model.predict(frame[FEATURE_COLUMNS])
    abs_error = (y_true - y_pred).abs()
    out = {
        "available": True,
        "rows": int(len(frame)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mean_prediction": float(pd.Series(y_pred).mean()),
        "mean_target": float(y_true.mean()),
        "calibration_bins": _calibration_bins(y_true, y_pred),
        "residual_quantiles": _residual_quantiles(abs_error),
    }
    if len(frame) >= 2 and y_true.nunique() > 1:
        out["r2"] = float(r2_score(y_true, y_pred))
    return out


def _calibration_bins(y_true: pd.Series, y_pred) -> list[dict[str, Any]]:
    pred = pd.Series(y_pred, index=y_true.index).clip(0.0, 1.0)
    bins = pd.cut(pred, bins=[0.0, 0.34, 0.67, 1.0], labels=["easy", "medium", "hard"], include_lowest=True)
    rows = []
    for bucket in ["easy", "medium", "hard"]:
        mask = bins == bucket
        count = int(mask.sum())
        rows.append(
            {
                "bucket": bucket,
                "count": count,
                "mean_prediction": float(pred[mask].mean()) if count else None,
                "mean_target": float(y_true[mask].mean()) if count else None,
                "mae": float((y_true[mask] - pred[mask]).abs().mean()) if count else None,
            }
        )
    return rows


def _residual_quantiles(abs_error: pd.Series) -> dict[str, float]:
    if abs_error.empty:
        return {}
    return {
        "p50": float(abs_error.quantile(0.50)),
        "p80": float(abs_error.quantile(0.80)),
        "p95": float(abs_error.quantile(0.95)),
    }


def run(config_path: str | None = None) -> None:
    del config_path
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    train = _load_features("train")
    X = train[FEATURE_COLUMNS]
    y = train["difficulty_target"].astype(float)
    model = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, min_samples_leaf=2)
    model.fit(X, y)
    train_pred = pd.Series(model.predict(X), index=y.index).clip(0.0, 1.0)
    train_abs_error = (y - train_pred).abs()
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target": f"proof_state_features.{TARGET_COLUMN}",
        "fallback_target": f"proof_state_features.{FALLBACK_TARGET_COLUMN}",
        "target_source": "proof_length_tactic_count_premise_count_negative_candidates",
        "residual_quantiles": _residual_quantiles(train_abs_error),
        "calibration_bins": _calibration_bins(y, train_pred),
    }
    joblib.dump(artifact, "outputs/models/difficulty_estimator.joblib")
    metrics = {"train": _evaluate(model, "train")}
    for split in SPLITS:
        if split != "train":
            metrics[split] = _evaluate(model, split)
    metrics["feature_columns"] = FEATURE_COLUMNS
    metrics["target"] = f"proof_state_features.{TARGET_COLUMN}"
    metrics["fallback_target"] = f"proof_state_features.{FALLBACK_TARGET_COLUMN}"
    metrics["target_source"] = "proof_length_tactic_count_premise_count_negative_candidates"
    write_json("outputs/reports/difficulty_estimator_metrics.json", metrics)
