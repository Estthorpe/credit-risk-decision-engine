from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "./data/raw"))
    processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk-decision-engine")

    input_schema_version: str = os.getenv("INPUT_SCHEMA_VERSION", "1.0")
    model_version: str = os.getenv("MODEL_VERSION", "0.1.0")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


SETTINGS = Settings()
