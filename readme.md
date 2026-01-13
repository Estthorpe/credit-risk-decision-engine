# Credit Risk Decision Engine (Production-Grade ML System)

**Probability of Default (PD) scoring + decision policy + audit-friendly reason codes**  
Built as a **production-grade ML system**: data contracts, evaluation-as-tests, versioned model bundle, FastAPI scoring service, Streamlit UI, monitoring + retraining runbook, and CI.

> ✅ This project is intentionally engineered like a real ML service you’d ship:  
> **feature contract + reproducible artifacts + tests + monitoring + UI** (not a notebook demo).

---

## 🖼️ Demo

### Streamlit UI (calls FastAPI `/score`)
![Streamlit UI](docs/diagrams/screenshots/ui.png)



---

## 🚀 What This System Does

Given a minimal set of applicant inputs (8–12 fields), the system returns:

- **PD** (calibrated probability of default)
- **Decision**: `approve` / `manual_review` / `decline` (policy thresholds stored in model metadata)
- **Reason codes**: lightweight, audit-friendly explanations (rule-based mapping from key features)
- **Model version + schema version + latency**
- **Prometheus metrics** at `/metrics`

---

## 🧠 Why This Is “Production-Grade”

This repo implements the engineering standards recruiters expect for real ML systems:

### ✅ Data & Schema Governance
- **Data contract tests** ensure:
  - required columns exist
  - primary key uniqueness
  - target validity (binary)
- Contract tests run in **CI** using a committed fixture sample (`data/fixtures/train_sample.parquet`)

### ✅ Leakage-Safe Modeling
- **Stratified split** with **ID disjointness checks** to prevent leakage across train/valid/test.

### ✅ Reproducible Model Bundle + Feature Contract
A trained model is packaged into a **versioned bundle** that includes:
- `model.joblib` (pipeline: preprocessing + classifier)
- `calibrator.joblib` (probability calibration)
- `metadata.json` (model version, schema version, thresholds, etc.)
- `reference_stats.json` (baseline stats used for monitoring)
- `feature_columns.json` (**the inference feature contract**)

This makes serving deterministic and prevents “train/inference mismatch”.

### ✅ Serving Layer (FastAPI)
- `/health` for readiness + bundle identity
- `/score` for inference (PD + decision + reason codes)
- `/metrics` for Prometheus scraping

### ✅ Monitoring + Retraining Runbook
Monitoring script produces a JSON report with:
- **PSI drift** on key features (EXT_SOURCE_2, AMT_INCOME_TOTAL, DAYS_BIRTH)
- **Calibration drift** via Brier score (when labels available)
- `recommended_action`: `ok` / `investigate` / `retrain_recommended`

### ✅ CI / Automation
GitHub Actions runs:
- install (via `pyproject.toml`)
- tests (including contract tests)
- inference contract tests

---

## 🏗️ Architecture Overview

**Streamlit UI → FastAPI API → Bundle Loader → Model + Calibrator**

- The UI sends user inputs to the API (recommended)
- The API owns inference via the bundle:
  - uses `feature_columns.json` to align inference schema
  - uses `calibrator.joblib` to output calibrated PD
  - uses decision policy stored in `metadata.json`

---

## 📁 Repository Structure

```text
.
├── .github/workflows/ci.yml
├── artifacts/
│   ├── latest_eval.json
│   └── monitoring/  (generated monitor reports)
├── bundle/
│   ├── latest/PATH.txt
│   └── model_0.1.0/ (trained bundle artifacts)
├── data/
│   ├── fixtures/train_sample.parquet   (CI fixture sample)
│   └── processed/train_table.parquet   (local training input; typically not committed)
├── docs/
│   ├── architecture.md
│   ├── model_card.md
│   ├── evaluation_report.md
│   ├── ops_runbook.md
│   └── diagrams/screenshots/ui.png
├── scripts/
│   └── monitor.py
├── src/credit_risk_decision_engine/
│   ├── serving/ (FastAPI app + loader + schemas)
│   ├── modeling/ (bundle save/load, training)
│   ├── monitoring/ (PSI utilities)
│   └── ...
├── tests/
│   ├── test_data_contract.py
│   └── test_inference_contract.py
├── ui/
│   └── streamlit_app.py
├── pyproject.toml
├── Makefile
└── README.md


✅ Quickstart (Local)
0) Create environment + install

python -m venv .venv
# Windows
.venv\Scripts\activate

python -m pip install -U pip
pip install -e ".[dev]"


🧪 Run Tests (Local)
python -m pytest -q
Note: tests that read local training data expect data/processed/train_table.parquet to exist locally.
CI uses data/fixtures/train_sample.parquet so it works without your full dataset.

🏋️ Train + Build Bundle (Local)
Training produces:

artifacts/latest_eval.json

bundle/model_<version>/...

bundle/latest/PATH.txt


# Example (your project may already have a dedicated CLI/entrypoint)
python -c "from credit_risk_decision_engine.modeling.train import train_all; import pandas as pd; from credit_risk_decision_engine.config import SETTINGS; df=pd.read_parquet(SETTINGS.processed_data_dir/'train_table.parquet'); train_all(df)"
🌐 Start the API (FastAPI)

python -m uvicorn credit_risk_decision_engine.serving.app:app --reload --host 0.0.0.0 --port 8000
Health check:



Invoke-WebRequest http://localhost:8000/health -UseBasicParsing
🖥️ Start the UI (Streamlit)
In a new terminal (keep API running):


streamlit run ui/streamlit_app.py
The UI will call:

http://localhost:8000/score

🔎 Monitoring (PSI + Brier Drift)
Run monitoring on a batch file:


python scripts/monitor.py data/fixtures/train_sample.parquet
Output includes PSI + Brier drift checks and a recommended action:


{
  "psi": {
    "EXT_SOURCE_2": {"psi": 0.0016, "status": "ok"},
    "AMT_INCOME_TOTAL": {"psi": 0.0021, "status": "ok"}
  },
  "brier": {
    "batch_brier": 0.0494,
    "baseline_brier": 0.0683,
    "brier_drift_flag": false
  },
  "recommended_action": "ok"
}
Monitoring reports are saved to:

artifacts/monitoring/monitor_report_<timestamp>.json

📊 API Endpoints
GET /health → bundle identity + schema version

POST /score → PD + decision + reason codes

GET /metrics → Prometheus metrics

Example request:


curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-1",
    "features": {
      "EXT_SOURCE_1": 0.85,
      "EXT_SOURCE_2": 0.80,
      "EXT_SOURCE_3": 0.78,
      "AMT_INCOME_TOTAL": 150000,
      "AMT_CREDIT": 300000,
      "AMT_ANNUITY": 18000,
      "DAYS_BIRTH": -16000,
      "DAYS_EMPLOYED": -5000,
      "CODE_GENDER": "F",
      "FLAG_OWN_CAR": "Y",
      "NAME_EDUCATION_TYPE": "Higher education",
      "NAME_INCOME_TYPE": "Working"
    }
  }'
🧾 Documentation
Architecture: docs/architecture.md

Model Card: docs/model_card.md

Evaluation Report: docs/evaluation_report.md

Ops Runbook (Monitoring + Retraining + Rollback): docs/ops_runbook.md

⚠️ Common Pitfalls (and How This Repo Avoids Them)
1) Train/Inference Schema Mismatch
✅ Solved via feature_columns.json (the feature contract) and API-driven inference.

2) Data Leakage in Splits
✅ Solved via ID disjointness checks + stratified split logic.

3) Uncalibrated Probabilities
✅ Solved via probability calibration (stable PD output).

4) CI failures due to missing local datasets
✅ Solved via committed fixture parquet used by contract tests in CI.

🧭 Roadmap (Optional Enhancements)
Dockerize API + UI with docker-compose

Add model registry workflow (MLflow → stage promotion → bundle export)

Add scheduled monitoring runs + alerting thresholds

Add rollback command: “switch bundle/latest/PATH.txt to previous bundle”

👤 Author
Built by Esther Uzor — AI/ML Engineer focused on production engineering, MLOps, and responsible deployment patterns.

📜 License
MIT 







# 












