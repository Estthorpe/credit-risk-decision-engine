# Architecture — Credit Risk Decision Engine

## Overview
The Credit Risk Decision Engine is a containerless, production-style ML system for
predicting probability of default (PD) and returning a decision with audit-ready
reason codes.

The system is designed around:
- reproducibility
- leakage-safe training
- evaluation-as-tests
- explicit monitoring and retraining triggers

## High-Level Flow

Raw Data
→ Ingestion
→ Data Contract Validation
→ Feature Engineering
→ Leakage-Safe Split
→ Model Training + Calibration
→ Evaluation Artifacts
→ Model Bundle
→ FastAPI Service
→ Monitoring + Retraining Loop

## Components

### Data Ingestion
- Source: Home Credit Default Risk dataset
- Script: `scripts/ingest.py`
- Output: `data/processed/train_table.parquet`

### Data Contracts
- Implemented with Pandera-style rules
- Enforced via:
  - `tests/test_data_contract.py`
- Guarantees:
  - schema stability
  - target correctness
  - primary key uniqueness

### Feature Engineering
- Numeric + categorical separation
- Median / mode imputation
- One-hot encoding
- Preprocessing pipeline reused in training and serving

### Leakage-Safe Splitting
- Stratified split on target
- Explicit ID disjointness checks
- Train / validation / test separation

### Model Training
- Baseline: Logistic Regression
- Production: LightGBM
- Calibration: Platt scaling (sigmoid)
- Tracking: MLflow (local filesystem backend)

### Artifacts
- `artifacts/latest_eval.json` — evaluation-as-tests contract
- `bundle/model_<version>/`:
  - model.joblib
  - calibrator.joblib
  - metadata.json
  - reference_stats.json
  - feature_columns.json

### Serving Layer
- FastAPI (containerless)
- Endpoints:
  - `POST /score`
  - `GET /health`
  - `GET /metrics`
- Structured JSON logs
- Prometheus-compatible metrics

### Monitoring
- PSI drift on key features
- Calibration drift via Brier score
- Actionable recommendations (ok / investigate / retrain)

## Design Principles
- Explicit contracts over implicit assumptions
- Metrics as gates, not dashboards
- Bundles as deployable units
- Rollback via pointer swap, not redeploy
