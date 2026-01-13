from pathlib import Path
from credit_risk_decision_engine.monitoring.psi import psi_status 

def test_psi_status_buckets():
    assert psi_status(0.05) == "ok"
    assert psi_status(0.15) == "warn"
    assert psi_status(0.30) == "high"

def test_fixture_exists():
    assert Path("data/fixtures/monitoring_batch.parquet").exists()