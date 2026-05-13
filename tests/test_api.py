"""Smoke test for the FastAPI app's digital-twin contract.

We don't require a trained model artefact - we monkeypatch ``Predictor``
with a fake one so the test can run in CI without first running training.
The goal is to lock in the JSON shape the React client depends on.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from fastapi.testclient import TestClient


class _FakePredictor:
    model_name = "fake"
    threshold = 0.3

    def predict_row(self, machineID: int, timestamp: pd.Timestamp) -> dict[str, Any]:
        return {
            "machineID": int(machineID),
            "timestamp": pd.Timestamp(timestamp),
            "probability": 0.72,
            "threshold": self.threshold,
            "feature_row": {
                "datetime": pd.Timestamp(timestamp),
                "machineID": int(machineID),
                "hours_since_maint_comp1": 720.0,
                "hours_since_maint_comp2": 2400.0,
                "hours_since_maint_comp3": 100.0,
                "hours_since_maint_comp4": 50.0,
                "count_error1_24h": 0,
                "count_error2_24h": 3,
                "count_error3_24h": 0,
                "count_error4_24h": 0,
                "count_error5_24h": 0,
                "count_error1_72h": 0,
                "count_error2_72h": 3,
                "count_error3_72h": 0,
                "count_error4_72h": 0,
                "count_error5_72h": 0,
                "vibration_std_24h": 12.5,
                "pressure_std_24h": 5.0,
            },
        }


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    import api.server as server_mod

    monkeypatch.setattr(
        server_mod.Predictor, "from_artifacts",
        classmethod(lambda cls, cfg=None: _FakePredictor()),
    )
    # The analysis endpoints require an active session; write a fake one
    # into a tmp artifacts dir and point the config at it.
    monkeypatch.setattr(
        server_mod, "load_config",
        lambda *a, **kw: _fake_cfg(tmp_path),
    )
    app = server_mod.create_app()
    return TestClient(app)


def _fake_cfg(tmp_path) -> dict:
    """Real cfg with paths rebased into tmp_path and a session sentinel written."""
    import json
    from datetime import datetime, timezone

    from pdm.config import load_config

    cfg = load_config()
    cfg["_repo_root"] = str(tmp_path)
    cfg["api"] = {"cors_origins": []}
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / ".session.json").write_text(json.dumps({
        "loaded": True,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "files": {k: f"{k}.csv" for k in
                  ("telemetry", "errors", "failures", "machines", "maint")},
        "pipeline": {"best_run": "fake"},
    }))
    return cfg


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_digital_twin_schema(client: TestClient) -> None:
    r = client.post("/predict", json={
        "machineID": 1,
        "timestamp": "2015-06-01T06:00:00",
    })
    assert r.status_code == 200
    body = r.json()
    expected_keys = {
        "machineID", "timestamp", "failure_risk_24h", "health_state",
        "likely_component", "confidence", "main_evidence", "prescription",
    }
    assert expected_keys.issubset(body.keys())
    assert body["health_state"] in {"healthy", "watch", "degraded", "critical"}
    assert body["prescription"] in {
        "continue", "monitor", "inspect", "schedule_maintenance", "urgent_maintenance",
    }
    assert body["confidence"] in {"low", "medium", "high"}
    assert 0.0 <= body["failure_risk_24h"] <= 1.0
    assert isinstance(body["main_evidence"], list)
    assert len(body["main_evidence"]) <= 3
