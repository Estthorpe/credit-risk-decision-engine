# Ops Runbook — Credit Risk Decision Engine

## 1) What this service does
Scores a single application record and returns:
- PD (probability of default)
- decision (approve / manual_review / decline)
- reason codes
- model + schema versions

## 2) How to run locally (containerless)
### Prereqs
- Python 3.11+
- Install deps:
  - pip install -e ".[dev]"

### Start API
python -m uvicorn credit_risk_decision_engine.serving.app:app --reload --host 0.0.0.0 --port 8000

### Health check
curl http://localhost:8000/health

### Metrics
curl http://localhost:8000/metrics

## 3) Monitoring: what we watch
### 3.1 PSI drift (feature distribution drift)
Key features:
- EXT_SOURCE_2
- AMT_INCOME_TOTAL
- DAYS_BIRTH

Thresholds:
- PSI < 0.10 => OK
- 0.10 <= PSI < 0.25 => WARN (investigate)
- PSI >= 0.25 => HIGH (retrain recommended)

### 3.2 Calibration drift (Brier score)
Only available when TARGET labels exist in batch.

Rule:
- If batch_brier - baseline_brier >= 0.01 => retrain recommended

### 3.3 API performance
From /metrics:
- score_requests_total (by status)
- score_request_latency_ms

Triggers:
- Elevated error rate => investigate immediately
- Latency spike => investigate infra / model load / input payloads

## 4) How to run monitoring
Example:
python scripts/monitor.py data/fixtures/train_sample.parquet

Outputs:
- Console JSON summary
- artifacts/monitoring/monitor_report_<timestamp>.json

## 5) Decision playbook
If recommended_action == "ok":
- Do nothing. Continue monitoring.

If recommended_action == "investigate":
- Check which feature PSI is WARN
- Validate upstream data pipeline / schema changes
- Run spot checks on batch inputs

If recommended_action == "retrain_recommended":
- Launch retraining (Step 6 below)
- Compare against baseline gates (AUROC/PR-AUC/Brier)
- If passed, promote new bundle to `bundle/latest`

## 6) Retraining process
1) Run ingest/build pipeline (if needed)
   - python scripts/ingest.py

2) Train a new model version
   - update SETTINGS.model_version (e.g., 0.1.1)
   - python scripts/train.py

3) Validate evaluation-as-tests
   - python -m pytest -q

4) Smoke test API
   - start uvicorn
   - curl /health
   - run inference contract test

## 7) Rollback procedure
Rollback = point `bundle/latest/PATH.txt` back to the previous bundle.

Steps:
1) Identify previous model dir:
   - dir bundle
2) Edit bundle/latest/PATH.txt to previous model path:
   - e.g. bundle/model_0.1.0
3) Restart API service
4) Confirm /health shows correct model_version
