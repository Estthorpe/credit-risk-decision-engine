from fastapi.testclient import TestClient
from credit_risk_decision_engine.serving.app import app

import warnings
warnings.filterwarnings(
    "ignore",
    message = "X does not have valid feature names"
)
                        


client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_score_contract_shape():
    payload = {
        "request_id": "test-123",
        "features": {
            "AMT_INCOME_TOTAL": 150000,
            "DAYS_BIRTH": -12000,
            "EXT_SOURCE_2": 0.2,
        },
    }

    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()

    # Required response fields
    assert body["request_id"] == "test-123"
    assert 0.0 <= body["pd"] <= 1.0
    assert body["decision"] in {"approve", "decline", "manual_review"}
    assert isinstance(body["reason_codes"], list)
    assert "model_version" in body
    assert "input_schema_version" in body
    assert "latency_ms" in body
