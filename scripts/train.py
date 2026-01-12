import pandas as pd

from credit_risk_decision_engine.config import SETTINGS
from credit_risk_decision_engine.modeling.train import train_all
from credit_risk_decision_engine.validation.validate import validate_or_raise

def main() -> None:
    print("Starting training pipeline...")
    
    path = SETTINGS.processed_data_dir / "train_table.parquet"
    df = pd.read_parquet(path)

    #Contract gate
    validate_or_raise(df)

    train_all(df)
    print("Training complete. Wrote artifacts/latest_eval.json and bundle/...")


if __name__ == "__main__":
    main()
