from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from .retrieve import retrieve_knowledge_for_theorem
from .utils import read_json

ALLOWED_INPUT_TYPES = {"lean", "theorem", "proof_state", "goal"}
ALLOWED_SPLITS = {"train", "val", "test", "demo"}
MAX_THEOREM_TEXT_CHARS = 20000
MAX_PREMISES = 50
MAX_THEOREMS = 25

LOGGER = logging.getLogger("proofatlas.api")
REQUEST_METRICS: dict[str, Any] = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "validation_errors": 0,
    "missing_artifact_errors": 0,
    "total_duration_ms": 0.0,
    "last_request": None,
}
WARMUP_STATUS: dict[str, Any] = {
    "attempted": False,
    "ok": None,
    "index_split": None,
    "duration_ms": 0.0,
    "error": None,
    "retrieval_backend": None,
}


def _index_artifact_exists(index_split: str, stem: str) -> bool:
    manifest_path = Path(f"outputs/indexes/{index_split}_{stem}_index_manifest.json")
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path, {}) or {}
    index_path = Path(manifest.get("index_path") or "")
    return index_path.exists()


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _record_request(*, endpoint: str, ok: bool, duration_ms: float, status_code: int, request_id: str, error_type: str | None = None) -> None:
    REQUEST_METRICS["total_requests"] += 1
    REQUEST_METRICS["total_duration_ms"] += float(duration_ms)
    if ok:
        REQUEST_METRICS["successful_requests"] += 1
    else:
        REQUEST_METRICS["failed_requests"] += 1
    if error_type == "validation":
        REQUEST_METRICS["validation_errors"] += 1
    elif error_type == "missing_artifact":
        REQUEST_METRICS["missing_artifact_errors"] += 1
    REQUEST_METRICS["last_request"] = {
        "endpoint": endpoint,
        "ok": ok,
        "status_code": status_code,
        "duration_ms": float(duration_ms),
        "request_id": request_id,
        "error_type": error_type,
    }
    LOGGER.info(
        "proofatlas_api_request",
        extra={
            "event": "proofatlas_api_request",
            "endpoint": endpoint,
            "ok": ok,
            "status_code": status_code,
            "duration_ms": float(duration_ms),
            "request_id": request_id,
            "error_type": error_type,
        },
    )


def api_metrics() -> dict[str, Any]:
    total = int(REQUEST_METRICS["total_requests"])
    avg = float(REQUEST_METRICS["total_duration_ms"]) / total if total else 0.0
    return {
        **REQUEST_METRICS,
        "average_duration_ms": avg,
        "warmup": WARMUP_STATUS,
    }


def reset_api_metrics() -> None:
    REQUEST_METRICS.update(
        {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "validation_errors": 0,
            "missing_artifact_errors": 0,
            "total_duration_ms": 0.0,
            "last_request": None,
        }
    )
    WARMUP_STATUS.update(
        {
            "attempted": False,
            "ok": None,
            "index_split": None,
            "duration_ms": 0.0,
            "error": None,
            "retrieval_backend": None,
        }
    )


def prometheus_metrics_text() -> str:
    metrics = api_metrics()
    last = metrics.get("last_request") or {}
    lines = [
        "# HELP proofatlas_api_requests_total Total theorem-guidance API requests.",
        "# TYPE proofatlas_api_requests_total counter",
        f"proofatlas_api_requests_total {int(metrics['total_requests'])}",
        "# HELP proofatlas_api_successful_requests_total Successful theorem-guidance API requests.",
        "# TYPE proofatlas_api_successful_requests_total counter",
        f"proofatlas_api_successful_requests_total {int(metrics['successful_requests'])}",
        "# HELP proofatlas_api_failed_requests_total Failed theorem-guidance API requests.",
        "# TYPE proofatlas_api_failed_requests_total counter",
        f"proofatlas_api_failed_requests_total {int(metrics['failed_requests'])}",
        "# HELP proofatlas_api_validation_errors_total Request validation errors.",
        "# TYPE proofatlas_api_validation_errors_total counter",
        f"proofatlas_api_validation_errors_total {int(metrics['validation_errors'])}",
        "# HELP proofatlas_api_missing_artifact_errors_total Missing artifact errors.",
        "# TYPE proofatlas_api_missing_artifact_errors_total counter",
        f"proofatlas_api_missing_artifact_errors_total {int(metrics['missing_artifact_errors'])}",
        "# HELP proofatlas_api_request_duration_ms_sum Total theorem-guidance request duration in milliseconds.",
        "# TYPE proofatlas_api_request_duration_ms_sum counter",
        f"proofatlas_api_request_duration_ms_sum {float(metrics['total_duration_ms']):.6f}",
        "# HELP proofatlas_api_request_duration_ms_avg Average theorem-guidance request duration in milliseconds.",
        "# TYPE proofatlas_api_request_duration_ms_avg gauge",
        f"proofatlas_api_request_duration_ms_avg {float(metrics['average_duration_ms']):.6f}",
        "# HELP proofatlas_api_last_request_duration_ms Last theorem-guidance request duration in milliseconds.",
        "# TYPE proofatlas_api_last_request_duration_ms gauge",
        f"proofatlas_api_last_request_duration_ms {float(last.get('duration_ms') or 0.0):.6f}",
    ]
    return "\n".join(lines) + "\n"


def artifact_status(index_split: str = "train") -> dict[str, Any]:
    required = {
        "proof_states": Path(f"data/processed/{index_split}/proof_states.parquet"),
        "premises": Path(f"data/processed/{index_split}/premises.parquet"),
        "theorems": Path(f"data/processed/{index_split}/theorems.parquet"),
        "embedding_config": Path("outputs/embeddings/embedding_config.json"),
        "premise_embeddings": Path(f"outputs/embeddings/{index_split}_premise_embeddings.npz"),
        "theorem_embeddings": Path(f"outputs/embeddings/{index_split}_theorem_embeddings.npz"),
        "graph_edges": Path(f"outputs/graph/{index_split}/edges_enriched.parquet"),
    }
    optional = {
        "proof_state_index": _index_artifact_exists(index_split, "proof_state"),
        "premise_index": _index_artifact_exists(index_split, "premise"),
        "theorem_index": _index_artifact_exists(index_split, "theorem"),
        "proof_state_index_manifest": Path(f"outputs/indexes/{index_split}_proof_state_index_manifest.json"),
        "premise_index_manifest": Path(f"outputs/indexes/{index_split}_premise_index_manifest.json"),
        "theorem_index_manifest": Path(f"outputs/indexes/{index_split}_theorem_index_manifest.json"),
        "premise_ranker": Path("outputs/models/premise_ranker.joblib"),
        "corpus_manifest": Path("outputs/reports/corpus_manifest.json"),
        "homepage": Path("homepage/index.html"),
    }
    required_status = {name: path.exists() for name, path in required.items()}
    optional_status = {name: path if isinstance(path, bool) else path.exists() for name, path in optional.items()}
    missing_required = [name for name, exists in required_status.items() if not exists]
    return {
        "index_split": index_split,
        "ready": not missing_required,
        "missing_required": missing_required,
        "required": required_status,
        "optional": optional_status,
    }


def validate_guidance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    theorem_text = str(payload.get("theorem_text") or payload.get("query_text") or "").strip()
    if not theorem_text:
        raise ValueError("Request must include theorem_text or query_text.")
    if len(theorem_text) > MAX_THEOREM_TEXT_CHARS:
        raise ValueError(f"theorem_text is too long; limit is {MAX_THEOREM_TEXT_CHARS} characters.")
    input_type = str(payload.get("input_type", "lean"))
    if input_type not in ALLOWED_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {sorted(ALLOWED_INPUT_TYPES)}.")
    index_split = str(payload.get("index_split", "train"))
    if index_split not in ALLOWED_SPLITS:
        raise ValueError(f"index_split must be one of {sorted(ALLOWED_SPLITS)}.")
    try:
        k_premises = int(payload.get("k_premises", 20))
        k_theorems = int(payload.get("k_theorems", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("k_premises and k_theorems must be integers.") from exc
    if not 1 <= k_premises <= MAX_PREMISES:
        raise ValueError(f"k_premises must be between 1 and {MAX_PREMISES}.")
    if not 1 <= k_theorems <= MAX_THEOREMS:
        raise ValueError(f"k_theorems must be between 1 and {MAX_THEOREMS}.")
    return {
        "theorem_text": theorem_text,
        "full_name": payload.get("full_name"),
        "input_type": input_type,
        "domain_hint": payload.get("domain_hint"),
        "file_path": payload.get("file_path"),
        "k_premises": k_premises,
        "k_theorems": k_theorems,
        "index_split": index_split,
        "validate_lean": bool(payload.get("validate_lean", False)),
    }


def guidance_from_payload(payload: dict[str, Any]) -> dict:
    return retrieve_knowledge_for_theorem(**validate_guidance_payload(payload))


def warmup_retrieval_cache(index_split: str = "train") -> dict[str, Any]:
    start = perf_counter()
    WARMUP_STATUS.update(
        {
            "attempted": True,
            "ok": False,
            "index_split": index_split,
            "duration_ms": 0.0,
            "error": None,
            "retrieval_backend": None,
        }
    )
    try:
        import pandas as pd

        proof_states = pd.read_parquet(f"data/processed/{index_split}/proof_states.parquet")
        if proof_states.empty:
            raise FileNotFoundError(f"No proof states available for warmup split: {index_split}")
        row = proof_states.iloc[0]
        theorem_text = "\n".join(
            part
            for part in [
                str(row.get("context", "")),
                f"⊢ {row.get('goal_text', '')}" if row.get("goal_text") else "",
            ]
            if part
        )
        guidance = retrieve_knowledge_for_theorem(
            theorem_text=theorem_text,
            full_name=str(row.get("full_name", "")),
            input_type="proof_state",
            k_premises=3,
            k_theorems=2,
            index_split=index_split,
            validate_lean=False,
        )
        backend = None
        ranked = guidance.get("ranked_premises") or []
        if ranked:
            backend = ranked[0].get("signals", {}).get("retrieval_backend")
        WARMUP_STATUS.update(
            {
                "ok": True,
                "duration_ms": (perf_counter() - start) * 1000.0,
                "retrieval_backend": backend,
            }
        )
    except Exception as exc:
        WARMUP_STATUS.update(
            {
                "ok": False,
                "duration_ms": (perf_counter() - start) * 1000.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    return dict(WARMUP_STATUS)


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(level=(level or os.getenv("PROOFATLAS_LOG_LEVEL", "INFO")).upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")


def create_app(require_ready: bool | None = None, startup_index_split: str | None = None, warmup_retrieval: bool | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, PlainTextResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("ProofAtlas API requires optional dependencies: pip install -e '.[api]'") from exc

    class GuidanceRequest(BaseModel):
        theorem_text: str = Field(..., max_length=MAX_THEOREM_TEXT_CHARS, description="Lean theorem declaration, raw proof state, or goal text.")
        full_name: str | None = None
        input_type: str = "lean"
        domain_hint: str | None = None
        file_path: str | None = None
        k_premises: int = Field(20, ge=1, le=MAX_PREMISES)
        k_theorems: int = Field(10, ge=1, le=MAX_THEOREMS)
        index_split: str = "train"
        validate_lean: bool = False

    configure_logging()
    startup_index_split = startup_index_split or os.getenv("PROOFATLAS_STARTUP_INDEX_SPLIT", "train")
    require_ready = _truthy_env("PROOFATLAS_REQUIRE_READY") if require_ready is None else require_ready
    warmup_retrieval = _truthy_env("PROOFATLAS_WARMUP_RETRIEVAL") if warmup_retrieval is None else warmup_retrieval
    if require_ready:
        status = artifact_status(startup_index_split)
        if not status["ready"]:
            missing = ", ".join(status["missing_required"])
            raise RuntimeError(f"ProofAtlas API startup readiness check failed for split {startup_index_split}: missing {missing}")
    if warmup_retrieval:
        try:
            warmup_retrieval_cache(startup_index_split)
        except Exception as exc:
            raise RuntimeError(f"ProofAtlas API retrieval warmup failed for split {startup_index_split}: {exc}") from exc

    app = FastAPI(title="ProofAtlas API", version="0.1.0")
    homepage_path = Path("homepage/index.html")
    assets_path = Path("homepage/assets")
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    @app.get("/")
    def homepage():
        if homepage_path.exists():
            return FileResponse(homepage_path)
        return {"name": "ProofAtlas API", "homepage": "Run `leanrank-kg build-homepage` to generate homepage/index.html."}

    @app.get("/health")
    def health(index_split: str = "train") -> dict[str, Any]:
        try:
            status = artifact_status(index_split)
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
        return {"status": "ready" if status["ready"] else "degraded", "artifacts": status, "metrics": api_metrics()}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return api_metrics()

    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        return prometheus_metrics_text()

    @app.post("/retrieve-theorem-guidance")
    def retrieve_theorem_guidance(request: GuidanceRequest) -> dict:
        request_id = str(uuid4())
        start = perf_counter()
        try:
            payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            result = guidance_from_payload(payload)
            duration_ms = (perf_counter() - start) * 1000.0
            _record_request(endpoint="/retrieve-theorem-guidance", ok=True, duration_ms=duration_ms, status_code=200, request_id=request_id)
            result.setdefault("service", {})["request_id"] = request_id
            result["service"]["duration_ms"] = duration_ms
            return result
        except ValueError as exc:
            _record_request(endpoint="/retrieve-theorem-guidance", ok=False, duration_ms=(perf_counter() - start) * 1000.0, status_code=400, request_id=request_id, error_type="validation")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            _record_request(endpoint="/retrieve-theorem-guidance", ok=False, duration_ms=(perf_counter() - start) * 1000.0, status_code=503, request_id=request_id, error_type="missing_artifact")
            raise HTTPException(status_code=503, detail=f"Required ProofAtlas artifact is missing: {exc}") from exc

    return app
