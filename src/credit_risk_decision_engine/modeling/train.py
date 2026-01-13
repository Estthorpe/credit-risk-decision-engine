from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import  CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk_decision_engine.config import SETTINGS
from credit_risk_decision_engine.evaluation.metrics import compute_classification_metrics, as_dict
from credit_risk_decision_engine.features.build import FeatureSpec, split_columns, build_preprocessor
from credit_risk_decision_engine.features.leakage import LeakageSpec, split_xy, assert_id_disjoint
from credit_risk_decision_engine.features.split import SplitConfig, stratefied_split
from credit_risk_decision_engine.modeling.bundle import save_bundle


@dataclass(frozen=True)
class TrainConfig:
    target_col: str = "TARGET"
    id_col: str = "SK_ID_CURR"
    random_state: int = 42

    #LightGBM defaults tuned for a fast + strong baseline
    lgbm_n_estimators: int = 800
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 64
    lgbm_min_child_samples: int = 50

    #Decision policy 
    approve_threshold: float = 0.15
    decline_threshold: float = 0.40


def _hash_dataframe_head(df: pd.DataFrame, n: int = 2000) -> str:
    sample = df.head(n).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(sample).hexdigest()[:16]


def _make_reference_stats(train_df: pd.DataFrame, key_features: Tuple[str, ...]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"features": {}}

    quantiles = [i / 10 for i in range(0, 11)]  # 0.0..1.0 (10 bins)

    for c in key_features:
        if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c]):
            s = train_df[c].dropna()

            base = {
                "mean": float(s.mean()) if not s.empty else None,
                "std": float(s.std()) if not s.empty else None,
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
                "missing_pct": float(train_df[c].isna().mean()),
            }

            # PSI reference: quantile bin edges + expected proportions
            if not s.empty:
                edges = s.quantile(quantiles).to_list()
                # Ensure edges are strictly increasing (quantiles can repeat in discrete columns)
                edges = sorted(set(float(x) for x in edges))
                if len(edges) >= 3:
                    counts, bin_edges = np.histogram(s.to_numpy(), bins=edges)
                    expected = (counts / max(counts.sum(), 1)).tolist()
                    base["psi"] = {
                        "bin_edges": [float(x) for x in bin_edges.tolist()],
                        "expected": [float(p) for p in expected],
                    }

            stats["features"][c] = base

        elif c in train_df.columns:
            stats["features"][c] = {
                "n_unique": int(train_df[c].nunique(dropna=True)),
                "missing_pct": float(train_df[c].isna().mean()),
            }

    return stats



def train_all(df: pd.DataFrame, cfg: TrainConfig = TrainConfig()) -> Dict[str, Any]:
    #1) Split(leakage-safe)
    split_cfg = SplitConfig(target_col=cfg.target_col, id_col=cfg.id_col, random_state=cfg.random_state)
    train_df,  valid_df, test_df = stratefied_split(df, split_cfg)
    assert_id_disjoint(train_df, valid_df, test_df, id_col=cfg.id_col)

    #2) Build preprocessing
    feat_spec = FeatureSpec(target_col=cfg.target_col, id_col=cfg.id_col)
    numeric_cols, categorical_cols = split_columns(train_df, feat_spec)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    #3) X/y
    leak_spec = LeakageSpec(target_col=cfg.target_col, id_col=cfg.id_col)
    X_train, y_train = split_xy(train_df, leak_spec)
    X_valid, y_valid = split_xy(valid_df, leak_spec)
    X_test, y_test = split_xy(test_df, leak_spec)

    #Drop ID from feaures before modelling
    X_train = X_train.drop(columns=[cfg.id_col])
    X_valid = X_valid.drop(columns=[cfg.id_col])
    X_test = X_test.drop(columns=[cfg.id_col])

    #4) MLflow setup
    mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
    mlflow.set_experiment(SETTINGS.mlflow_experiment_name)

    data_hash = _hash_dataframe_head(df)
    input_schema_version = SETTINGS.input_schema_version
    model_version = SETTINGS.model_version

    #-------Baseline: Logistic Regression-------
    baseline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=5000, solver="saga", n_jobs=-1)),
        ]
    )
    baseline.fit(X_train, y_train)
    base_valid_prob = baseline.predict_proba(X_valid)[:, 1]
    base_metrics = compute_classification_metrics(y_valid.to_numpy(), base_valid_prob)


    #----------Production: LightGBM + Calibration-------------
    lgbm = LGBMClassifier(
        n_estimators=cfg.lgbm_n_estimators,
        learning_rate=cfg.lgbm_learning_rate,
        num_leaves=cfg.lgbm_num_leaves,
        min_child_samples=cfg.lgbm_min_child_samples,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", lgbm),
        ]
    )
    model.fit(X_train, y_train)

    # Calibrate on validation set (sigmoid/Platt) for stable PD
    # CalibratedClassifierCV expects estimator with predict_proba; pipeline provides that.
    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calibrator.fit(X_train, y_train)

    valid_prob = calibrator.predict_proba(X_valid)[:, 1]
    test_prob = calibrator.predict_proba(X_test)[:, 1]

    valid_metrics = compute_classification_metrics(y_valid.to_numpy(), valid_prob)
    test_metrics = compute_classification_metrics(y_test.to_numpy(), test_prob)

    # 5) Save evaluation artifact (for tests)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    eval_payload = {
        "model_version": model_version,
        "input_schema_version": input_schema_version,
        "data_hash_head": data_hash,
        "split": {"random_state": cfg.random_state, "test_size": 0.20, "valid_size": 0.20},
        "baseline_logreg_valid": as_dict(base_metrics),
        "lgbm_calibrated_valid": as_dict(valid_metrics),
        "lgbm_calibrated_test": as_dict(test_metrics),
        "decision_policy": {
            "approve_threshold": cfg.approve_threshold,
            "decline_threshold": cfg.decline_threshold,
            "notes": "approve if pd < approve_threshold; decline if pd >= decline_threshold; else manual_review",
        },
    }
    (artifacts_dir / "latest_eval.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    # 6) Save bundle (for serving + monitoring)
    key_features = (
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED"
    )
    reference_stats = _make_reference_stats(train_df, key_features)

    bundle_dir = Path("bundle") / f"model_{model_version}"
    metadata = {
        "model_version": model_version,
        "input_schema_version": input_schema_version,
        "data_hash_head": data_hash,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "id_col": cfg.id_col,
        "target_col": cfg.target_col,
        "decision_policy": eval_payload["decision_policy"],
    }

    save_bundle(bundle_dir=bundle_dir, model=model, calibrator=calibrator, metadata=metadata, reference_stats=reference_stats, feature_columns=list(X_train.columns),
                )

    #Save feature column contract for serving 
    (bundle_dir / "feature_columns.json").write_text(
        json.dumps(numeric_cols + categorical_cols, indent=2),
        encoding="utf-8"
    )
    # Convenience symlink-ish pointer for “latest”
    latest = Path("bundle") / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "PATH.txt").write_text(str(bundle_dir), encoding="utf-8")

    # 7) MLflow logging (evidence)
    with mlflow.start_run(run_name=f"train_{model_version}") as run:
        mlflow.log_params({
            "random_state": cfg.random_state,
            "lgbm_n_estimators": cfg.lgbm_n_estimators,
            "lgbm_learning_rate": cfg.lgbm_learning_rate,
            "lgbm_num_leaves": cfg.lgbm_num_leaves,
            "lgbm_min_child_samples": cfg.lgbm_min_child_samples,
            "approve_threshold": cfg.approve_threshold,
            "decline_threshold": cfg.decline_threshold,
            "data_hash_head": data_hash,
            "input_schema_version": input_schema_version,
        })
        mlflow.log_metrics({
            "baseline_valid_auroc": base_metrics.auroc,
            "baseline_valid_brier": base_metrics.brier,
            "lgbm_valid_auroc": valid_metrics.auroc,
            "lgbm_valid_brier": valid_metrics.brier,
            "lgbm_test_auroc": test_metrics.auroc,
            "lgbm_test_brier": test_metrics.brier,
        })
        mlflow.log_artifact(str(artifacts_dir / "latest_eval.json"))
        mlflow.log_artifact(str(bundle_dir / "metadata.json"))
        mlflow.log_artifact(str(bundle_dir / "reference_stats.json"))

    return eval_payload


