from __future__ import annotations

from dataclasses  import dataclass
from typing import Tuple

import pandas as pd 
from sklearn.model_selection import train_test_split



@dataclass(frozen=True)
class SplitConfig:
    target_col: str = "TARGET"
    id_col: str = "SK_ID_CURR"
    test_size: float = 0.20
    valid_size: float = 0.20
    random_state: int = 42


def stratefied_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[cfg.target_col].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    y_train = train_df[cfg.target_col].astype(int)
    train_df, valid_df = train_test_split(
        train_df, test_size=cfg.valid_size, random_state=cfg.random_state, stratify=y_train
    )

    return train_df, valid_df, test_df
