# Evaluation Report — Credit Risk Decision Engine

## Evaluation Strategy
Evaluation is treated as a **contract**, not an exploratory exercise.

Metrics are:
- computed offline
- serialized to JSON
- enforced in CI via tests

## Metrics Used
- AUROC — ranking quality
- PR-AUC — minority class performance
- Brier score — calibration quality

## Baseline vs Production Model

### Baseline (Logistic Regression — Validation)
- AUROC ≈ 0.61
- Brier ≈ 0.074

### Production (LightGBM + Calibration)

Validation:
- AUROC ≈ 0.75
- Brier ≈ 0.069

Test:
- AUROC ≈ 0.76
- Brier ≈ 0.068

## Calibration
- Calibration applied post-training
- Improves PD reliability for decision thresholds
- Evaluated via Brier score

## Evaluation-as-Tests
The following are enforced in CI:
- AUROC must exceed baseline floor
- Brier must not regress beyond threshold
- API inference schema must remain stable

Any regression fails the pipeline.
