from __future__  import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

@dataclass(frozen=True)
class LeakageSpec:
    target_col: str = "TARGET"
    id_col: str = "SK_ID_CURR"


def split_xy(df: pd.DataFrame, spec: LeakageSpec) -> Tuple[pd.DataFrame, pd.Series]:
    assert spec.target_col in df.columns, "TARGET column missing"
    assert spec.id_col in df.columns, "ID column missing"
    y = df[spec.target_col].astype(int)
    X = df.drop(columns=[spec.target_col])
    return X, y


def assert_id_disjoint(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, id_col: str) -> None:
    tr = set(train[id_col].tolist())
    va = set(valid[id_col].tolist())
    te = set(test[id_col].tolist())
    assert tr.isdisjoint(va), "Leakage: train/valid share IDs"
    assert tr.isdisjoint(te), "Leakage: train/test share IDs"
    assert va.isdisjoint(te), "Leakage: valid/test share IDs"
           