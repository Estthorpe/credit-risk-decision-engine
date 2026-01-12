from pathlib import Path
import pandas as pd
from credit_risk_decision_engine.config import SETTINGS 

def main() -> None:
    raw_path = SETTINGS.raw_data_dir / "home_credit" / "application_train.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at: {raw_path}\n"
        )
    
    df = pd.read_csv(raw_path)
    SETTINGS.processed_data_dir.mkdir(parents=True, exist_ok=True)

    out_path = SETTINGS.processed_data_dir / "train_table.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved processed table: {out_path} rows={len(df)} cols={len(df.columns)}")

if __name__ == "__main__":
    main()
