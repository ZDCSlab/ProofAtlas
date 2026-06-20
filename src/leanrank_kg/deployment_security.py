from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .api import ALLOWED_INPUT_TYPES, ALLOWED_SPLITS, MAX_PREMISES, MAX_THEOREM_TEXT_CHARS, MAX_THEOREMS, artifact_status
from .utils import write_json


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _check(check_id: str, passed: bool, severity: str, detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "severity": severity,
        "status": "pass" if passed else severity,
        "detail": detail,
    }


def review(
    *,
    host: str = "127.0.0.1",
    require_ready: bool = False,
    startup_index_split: str = "train",
    reload: bool = False,
    public_exposure: bool = False,
) -> dict[str, Any]:
    status = artifact_status(startup_index_split)
    host_is_loopback = host in {"127.0.0.1", "localhost", "::1"}
    warmup_enabled = _truthy_env("PROOFATLAS_WARMUP_RETRIEVAL")
    checks = [
        _check(
            "bounded_request_text",
            MAX_THEOREM_TEXT_CHARS <= 20000,
            "fail",
            f"MAX_THEOREM_TEXT_CHARS={MAX_THEOREM_TEXT_CHARS}",
        ),
        _check(
            "bounded_result_limits",
            MAX_PREMISES <= 50 and MAX_THEOREMS <= 25,
            "fail",
            f"MAX_PREMISES={MAX_PREMISES}; MAX_THEOREMS={MAX_THEOREMS}",
        ),
        _check(
            "input_type_allowlist",
            bool(ALLOWED_INPUT_TYPES) and "lean" in ALLOWED_INPUT_TYPES,
            "fail",
            f"allowed={sorted(ALLOWED_INPUT_TYPES)}",
        ),
        _check(
            "split_allowlist",
            bool(ALLOWED_SPLITS) and "train" in ALLOWED_SPLITS,
            "fail",
            f"allowed={sorted(ALLOWED_SPLITS)}",
        ),
        _check(
            "artifact_readiness",
            bool(status.get("ready")),
            "fail" if public_exposure else "warn",
            f"split={startup_index_split}; missing_required={status.get('missing_required', [])}",
        ),
        _check(
            "readiness_gated_startup",
            bool(require_ready),
            "fail" if public_exposure else "warn",
            "--require-ready or PROOFATLAS_REQUIRE_READY=1 should be enabled for deployed servers",
        ),
        _check(
            "local_bind_for_local_demo",
            public_exposure or host_is_loopback,
            "warn",
            f"host={host}; local demos should bind to 127.0.0.1 or localhost",
        ),
        _check(
            "reload_disabled",
            not reload,
            "fail" if public_exposure else "warn",
            "uvicorn reload should be disabled outside development",
        ),
        _check(
            "cors_not_open_by_default",
            True,
            "fail",
            "API does not install permissive CORS middleware by default",
        ),
        _check(
            "metrics_endpoints_available",
            True,
            "warn",
            "GET /metrics and GET /metrics/prometheus expose in-process counters",
        ),
        _check(
            "retrieval_warmup_enabled",
            bool(warmup_enabled),
            "warn",
            "set PROOFATLAS_WARMUP_RETRIEVAL=1 for deployed demos to load the embedding model and ANN indexes before the first user request",
        ),
        _check(
            "access_control_confirmed",
            (not public_exposure) or _truthy_env("PROOFATLAS_ACCESS_CONTROL_CONFIRMED"),
            "fail" if public_exposure else "warn",
            "set PROOFATLAS_ACCESS_CONTROL_CONFIRMED=1 after adding auth/rate-limit/proxy controls",
        ),
        _check(
            "network_controls_confirmed",
            (not public_exposure) or _truthy_env("PROOFATLAS_NETWORK_CONTROLS_CONFIRMED"),
            "fail" if public_exposure else "warn",
            "set PROOFATLAS_NETWORK_CONTROLS_CONFIRMED=1 after restricting public network access",
        ),
        _check(
            "persistent_monitoring_confirmed",
            (not public_exposure) or _truthy_env("PROOFATLAS_MONITORING_CONFIRMED"),
            "fail" if public_exposure else "warn",
            "set PROOFATLAS_MONITORING_CONFIRMED=1 after wiring metrics/logs to deployment monitoring",
        ),
        _check(
            "static_homepage_present",
            Path("homepage/index.html").exists(),
            "warn",
            "homepage/index.html should exist for demo review",
        ),
    ]
    failures = [row["id"] for row in checks if row["status"] == "fail"]
    warnings = [row["id"] for row in checks if row["status"] == "warn"]
    return {
        "public_exposure": bool(public_exposure),
        "host": host,
        "reload": bool(reload),
        "require_ready": bool(require_ready),
        "warmup_retrieval": bool(warmup_enabled),
        "startup_index_split": startup_index_split,
        "artifact_status": status,
        "passed": not failures,
        "public_exposure_ready": bool(public_exposure and not failures),
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "next_steps": [
            "Use --require-ready for deployed API servers.",
            "Set PROOFATLAS_WARMUP_RETRIEVAL=1 before live demos to avoid cold-start retrieval latency.",
            "Keep local demos bound to 127.0.0.1 unless protected by external access controls.",
            "Confirm auth/rate-limit/proxy, network, and monitoring controls before public exposure.",
            "Regenerate this review after each deployment configuration change.",
        ],
    }


def run(
    *,
    host: str = "127.0.0.1",
    require_ready: bool = False,
    startup_index_split: str = "train",
    reload: bool = False,
    public_exposure: bool = False,
    output_path: str = "outputs/reports/deployment_security_review.json",
) -> dict[str, Any]:
    result = review(
        host=host,
        require_ready=require_ready,
        startup_index_split=startup_index_split,
        reload=reload,
        public_exposure=public_exposure,
    )
    write_json(output_path, result)
    return result
