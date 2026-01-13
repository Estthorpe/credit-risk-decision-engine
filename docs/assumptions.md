# Assumptions & Constraints

## Data Assumptions
- TARGET is correctly labeled
- No future information leakage
- Feature distributions are stationary unless detected by monitoring

## Modeling Assumptions
- Stratified splits approximate production class balance
- Calibration remains stable within monitored drift thresholds
- Feature availability at inference matches training schema

## Decision Policy Assumptions
- Thresholds are illustrative
- Business risk tolerance not modeled
- Manual review is treated as a neutral state

## System Constraints
- Containerless deployment
- Local MLflow backend
- No orchestration layer (Airflow/K8s)

## Ethical & Regulatory Notes
- This system does not implement fairness constraints
- No explainability guarantees beyond heuristic reason codes
- Not compliant with lending regulations out-of-the-box
