import importlib.util

import pytest

from leanrank_kg import api, deployment_security


def test_guidance_from_payload_validates_required_text():
    with pytest.raises(ValueError, match="theorem_text or query_text"):
        api.guidance_from_payload({})


def test_validate_guidance_payload_rejects_bad_limits_and_types():
    with pytest.raises(ValueError, match="input_type"):
        api.validate_guidance_payload({"theorem_text": "⊢ x = x", "input_type": "bad"})
    with pytest.raises(ValueError, match="index_split"):
        api.validate_guidance_payload({"theorem_text": "⊢ x = x", "index_split": "dev"})
    with pytest.raises(ValueError, match="k_premises"):
        api.validate_guidance_payload({"theorem_text": "⊢ x = x", "k_premises": 500})
    with pytest.raises(ValueError, match="too long"):
        api.validate_guidance_payload({"theorem_text": "x" * (api.MAX_THEOREM_TEXT_CHARS + 1)})


def test_artifact_status_reports_missing_required_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    status = api.artifact_status("train")
    assert status["ready"] is False
    assert "proof_states" in status["missing_required"]
    assert status["required"]["embedding_config"] is False
    assert status["optional"]["premise_index"] is False


def test_deployment_security_review_is_conservative_for_public_exposure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = deployment_security.run(host="0.0.0.0", public_exposure=True, require_ready=False)
    assert result["passed"] is False
    assert "readiness_gated_startup" in result["failures"]
    assert "access_control_confirmed" in result["failures"]
    assert "network_controls_confirmed" in result["failures"]
    assert (tmp_path / "outputs/reports/deployment_security_review.json").exists()


def test_deployment_security_review_accepts_confirmed_public_controls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROOFATLAS_ACCESS_CONTROL_CONFIRMED", "1")
    monkeypatch.setenv("PROOFATLAS_NETWORK_CONTROLS_CONFIRMED", "1")
    monkeypatch.setenv("PROOFATLAS_MONITORING_CONFIRMED", "1")
    monkeypatch.setattr(deployment_security, "artifact_status", lambda index_split: {"ready": True, "missing_required": [], "index_split": index_split})
    result = deployment_security.run(host="0.0.0.0", public_exposure=True, require_ready=True)
    assert result["passed"] is True
    assert result["public_exposure_ready"] is True
    assert not result["failures"]


def test_guidance_from_payload_passes_request_fields(monkeypatch):
    captured = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "query": {"theorem_text": kwargs["theorem_text"]}}

    monkeypatch.setattr(api, "retrieve_knowledge_for_theorem", fake_retrieve)
    result = api.guidance_from_payload(
        {
            "theorem_text": "theorem Nat.self_eq (x : Nat) : x = x := by simpa",
            "full_name": "Nat.self_eq",
            "input_type": "theorem",
            "domain_hint": "Data",
            "file_path": "Mathlib/Data/Nat/Basic.lean",
            "k_premises": 7,
            "k_theorems": 3,
            "index_split": "train",
            "validate_lean": True,
        }
    )
    assert result["ok"] is True
    assert captured["full_name"] == "Nat.self_eq"
    assert captured["input_type"] == "theorem"
    assert captured["k_premises"] == 7
    assert captured["k_theorems"] == 3
    assert captured["validate_lean"] is True


def test_prometheus_metrics_text_exports_counters():
    api.reset_api_metrics()
    api._record_request(
        endpoint="/retrieve-theorem-guidance",
        ok=True,
        duration_ms=12.5,
        status_code=200,
        request_id="req-test",
    )
    text = api.prometheus_metrics_text()
    assert "proofatlas_api_requests_total 1" in text
    assert "proofatlas_api_successful_requests_total 1" in text
    assert "proofatlas_api_request_duration_ms_sum 12.500000" in text
    assert "proofatlas_api_last_request_duration_ms 12.500000" in text


@pytest.mark.skipif(not importlib.util.find_spec("fastapi"), reason="FastAPI optional dependency is not installed")
def test_create_app_routes(monkeypatch):
    from fastapi.testclient import TestClient

    api.reset_api_metrics()
    monkeypatch.setattr(api, "guidance_from_payload", lambda payload: {"query": payload, "ranked_premises": []})
    client = TestClient(api.create_app())
    assert client.get("/").status_code == 200
    health = client.get("/health").json()
    assert health["status"] in {"ready", "degraded"}
    assert "artifacts" in health
    assert "metrics" in health
    assert client.get("/metrics").json()["total_requests"] == 0
    prometheus = client.get("/metrics/prometheus")
    assert prometheus.status_code == 200
    assert "proofatlas_api_requests_total" in prometheus.text
    response = client.post("/retrieve-theorem-guidance", json={"theorem_text": "⊢ x = x"})
    assert response.status_code == 200
    assert response.json()["query"]["theorem_text"] == "⊢ x = x"
    assert "request_id" in response.json()["service"]
    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 1
    assert metrics["successful_requests"] == 1
    assert metrics["average_duration_ms"] >= 0.0
    bad_response = client.post("/retrieve-theorem-guidance", json={"theorem_text": "⊢ x = x", "k_premises": 500})
    assert bad_response.status_code in {400, 422}


@pytest.mark.skipif(not importlib.util.find_spec("fastapi"), reason="FastAPI optional dependency is not installed")
def test_create_app_require_ready_rejects_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="startup readiness check failed"):
        api.create_app(require_ready=True, startup_index_split="train")


@pytest.mark.skipif(not importlib.util.find_spec("fastapi"), reason="FastAPI optional dependency is not installed")
def test_create_app_can_warm_retrieval_cache(monkeypatch):
    calls = []

    def fake_warmup(index_split):
        calls.append(index_split)
        return {"ok": True, "index_split": index_split}

    monkeypatch.setattr(api, "warmup_retrieval_cache", fake_warmup)
    api.create_app(warmup_retrieval=True, startup_index_split="train")
    assert calls == ["train"]


@pytest.mark.skipif(not importlib.util.find_spec("fastapi"), reason="FastAPI optional dependency is not installed")
def test_create_app_rejects_failed_retrieval_warmup(monkeypatch):
    def fake_warmup(index_split):
        raise FileNotFoundError(index_split)

    monkeypatch.setattr(api, "warmup_retrieval_cache", fake_warmup)
    with pytest.raises(RuntimeError, match="retrieval warmup failed"):
        api.create_app(warmup_retrieval=True, startup_index_split="train")
