from pathlib import Path

import pandas as pd

from credit_risk_decision_engine.validation.validate import validate_training_df
from credit_risk_decision_engine.config import SETTINGS

FIXTURE = Path("data/fixtures/train_sample.parquet")


def _load_fixture() -> pd.DataFrame:
    assert FIXTURE.exists(), f"Missing fixture: {FIXTURE}. Commit it to the repo."
    return pd.read_parquet(FIXTURE).head(5000)


def test_data_contract_passes_sample():
    df = _load_fixture()
    result = validate_training_df(df)
    assert result.passed is True, result.error


def test_primary_key_unique():
    df = _load_fixture()
    assert df["SK_ID_CURR"].is_unique


def test_target_binary():
    df = _load_fixture()
    assert set(df["TARGET"].dropna().unique()).issubset({0, 1})
