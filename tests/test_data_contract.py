import pandas as pd

from credit_risk_decision_engine.config import SETTINGS
from credit_risk_decision_engine.validation.validate import validate_training_df


def test_data_contract_passes_sample():
    path = SETTINGS.processed_data_dir / "train_table.parquet"
    df = pd.read_parquet(path).head(5000)

    result = validate_training_df(df)
    assert result.passed is True, result.error


def test_primary_key_unique():
    path = SETTINGS.processed_data_dir / "train_table.parquet"
    df = pd.read_parquet(path).head(5000)

    assert df["SK_ID_CURR"].is_unique


def test_target_binary():
    path = SETTINGS.processed_data_dir / "train_table.parquet"
    df = pd.read_parquet(path).head(5000)

    assert set(df["TARGET"].dropna().unique()).issubset({0, 1})
