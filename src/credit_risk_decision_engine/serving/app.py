from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from credit_risk_decision_engine.serving.model_loader import load_latest_bundle
from credit_risk_decision_engine.serving.schemas import ScoreRequest, ScoreResponse
from credit_risk_decision_engine.serving.reasons import reason_codes_from_features

app = FastAPI(title="Credit Risk Decision Engine", version="0.1.0")

# Prometheus metrics
REQ_COUNT = Counter("score_requests_total", "Total /score requests", ["status"])
LATENCY = Histogram("score_request_latency_ms", "Latency for /score in ms")

# Load model bundle at startup
BUNDLE_DIR = Path("bundle")
BUNDLE_PATH, BUNDLE = load_latest_bundle(BUNDLE_DIR)


def decision_from_pd(pd: float, approve_th: float, decline_th: float) -> str:
    if pd < approve_th:
        return "approve"
    if pd >= decline_th:
        return "decline"
    return "manual_review"


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "bundle_path": str(BUNDLE_PATH),
        "model_version": BUNDLE.meta.get("model_version", "unknown"),
        "input_schema_version": BUNDLE.meta.get("input_schema_version", "unknown"),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    request_id = req.request_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        # Build one-row DataFrame aligned to training feature columns
        row = {c: req.features.get(c, None) for c in BUNDLE.feature_columns}
        X = pd.DataFrame([row])

        # --- Step 1: base model probability ---
        base_pd = float(BUNDLE.model.predict_proba(X)[:, 1][0])

        # --- Step 2: calibrate base probability to final PD ---
        # Calibrator expects shape (n_samples, 1) because it was trained on predicted probs
        pd_hat = float(BUNDLE.calibrator.predict_proba(X)[:, 1][0])

        policy = BUNDLE.meta.get("decision_policy", {})
        approve_th = float(policy.get("approve_threshold", 0.15))
        decline_th = float(policy.get("decline_threshold", 0.4))

        decision = decision_from_pd(pd_hat, approve_th, decline_th)
        reasons = reason_codes_from_features(req.features, pd_hat)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        LATENCY.observe(latency_ms)
        REQ_COUNT.labels(status="ok").inc()

        # Structured JSON log
        log_obj = {
            "event": "score",
            "request_id": request_id,
            "model_version": BUNDLE.meta.get("model_version", "unknown"),
            "input_schema_version": BUNDLE.meta.get("input_schema_version", "unknown"),
            "latency_ms": round(latency_ms, 3),
        }
        print(json.dumps(log_obj))

        return ScoreResponse(
            request_id=request_id,
            pd=pd_hat,
            decision=decision,
            reason_codes=reasons,
            model_version=BUNDLE.meta.get("model_version", "unknown"),
            input_schema_version=BUNDLE.meta.get("input_schema_version", "unknown"),
            latency_ms=latency_ms,
        )

    except Exception as e:
        REQ_COUNT.labels(status="error").inc()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        log_obj = {
            "event": "score_error",
            "request_id": request_id,
            "model_version": BUNDLE.meta.get("model_version", "unknown"),
            "input_schema_version": BUNDLE.meta.get("input_schema_version", "unknown"),
            "latency_ms": round(latency_ms, 3),
            "error": str(e),
        }
        print(json.dumps(log_obj))

        # Return a clean API error response (instead of raw traceback)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
