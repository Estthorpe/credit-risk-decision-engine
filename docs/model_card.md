# Model Card — Credit Risk Decision Engine

## Model Overview
- Task: Binary classification (default vs non-default)
- Output: Probability of Default (PD)
- Decision policy layered on top of PD

## Intended Use
- Educational / portfolio demonstration
- Risk modeling experimentation
- Decisioning system architecture illustration

⚠️ Not intended for real-world lending decisions without full regulatory governance.

## Data
- Dataset: Home Credit Default Risk (public)
- Target: TARGET (1 = default, 0 = non-default)
- Records: ~300k applications

## Features
- Mix of numeric and categorical features
- Key drivers include:
  - EXT_SOURCE_*
  - income and credit amounts
  - age and employment duration

## Model Details
- Algorithm: LightGBM
- Calibration: Sigmoid (Platt scaling)
- Baseline: Logistic Regression

## Performance (Test Set)
- AUROC: ~0.76
- PR-AUC: ~0.25
- Brier Score: ~0.068

## Decision Policy
- Approve if PD < 0.15
- Decline if PD ≥ 0.40
- Otherwise: manual review

Thresholds chosen for demonstration and are not business-validated.

## Limitations
- No fairness constraints applied
- No temporal validation
- No reject inference
- Synthetic decision thresholds

## Monitoring
- Feature drift: PSI
- Calibration drift: Brier score
- API metrics: latency and error rate
