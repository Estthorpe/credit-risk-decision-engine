from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from credit_risk_decision_engine.modeling.bundle import load_bundle
from credit_risk_decision_engine.monitoring.psi import compute_psi_from_edges, psi_status


KEY_FEATURES = ("EXT_SOURCE_2", "AMT_INCOME_TOTAL", "DAYS_BIRTH")


@dataclass(frozen=True)
class MonitorConfig:
    bundle_dir: Path = Path("bundle")
    report_dir: Path = Path("artifacts") / "monitoring"
    baseline_eval_path: Path = Path("artifacts") / "latest_eval.json"


def _load_latest_bundle_path(bundle_root: Path) -> Path:
    latest_path_file = bundle_root / "latest" / "PATH.txt"
    if latest_path_file.exists():
        return Path(latest_path_file.read_text(encoding="utf-8").strip())

    # fallback: scan for model_* directories
    candidates = sorted([p for p in bundle_root.glob("model_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No bundle found under bundle/. Run training first.")
    return candidates[-1]


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def _load_batch(batch_path: Path) -> pd.DataFrame:
    suf = batch_path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(batch_path)
    if suf == ".csv":
        return pd.read_csv(batch_path)
    raise ValueError("batch_path must be .parquet or .csv")


def monitor_batch(batch_path: Path, cfg: MonitorConfig = MonitorConfig()) -> Dict[str, Any]:
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # Use one timezone-aware timestamp for the whole run (fixes utcnow deprecation warnings)
    now = datetime.now(timezone.utc)

    # 1) Load latest bundle (+ reference stats + metadata)
    bundle_path = _load_latest_bundle_path(cfg.bundle_dir)
    b = load_bundle(bundle_path)
    ref_stats = b["reference_stats"]
    meta = b["metadata"]

    # 2) Load batch data
    df = _load_batch(batch_path)

    # 3) PSI drift for key features
    psi_results: Dict[str, Any] = {}
    worst_status = "ok"
    worst_psi = 0.0

    for feat in KEY_FEATURES:
        feat_ref = ref_stats.get("features", {}).get(feat, {})
        psi_ref = feat_ref.get("psi")

        if psi_ref is None:
            psi_results[feat] = {
                "status": "missing_reference",
                "psi": None,
                "note": "Reference PSI bins not found. Re-run training after adding psi bins to reference_stats.",
            }
            worst_status = "warn"
            continue

        edges = psi_ref["bin_edges"]
        expected = psi_ref["expected"]

        if feat not in df.columns:
            psi_results[feat] = {"status": "missing_in_batch", "psi": None}
            worst_status = "warn"
            continue

        psi_val = compute_psi_from_edges(expected, edges, df[feat])
        status = psi_status(psi_val)
        psi_results[feat] = {"psi": psi_val, "status": status}

        if psi_val > worst_psi:
            worst_psi = psi_val
        if status == "high":
            worst_status = "high"
        elif status == "warn" and worst_status != "high":
            worst_status = "warn"

    # 4) Calibration drift (Brier), only if labels exist
    brier_result: Optional[Dict[str, Any]] = None
    if "TARGET" in df.columns:
        feature_cols_path = bundle_path / "feature_columns.json"
        if not feature_cols_path.exists():
            raise FileNotFoundError(
                f"Missing {feature_cols_path}. Re-run training to write feature_columns.json into the bundle."
            )

        feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

        # IMPORTANT: force a real DataFrame with the exact column order.
        # This reduces LightGBM feature-name warnings and makes input stable.
        X = df.reindex(columns=feature_cols).copy()
        X = pd.DataFrame(X, columns=feature_cols)

        probs = b["calibrator"].predict_proba(X)[:, 1]
        y_true = df["TARGET"].to_numpy()

        batch_brier = _brier(y_true, probs)

        baseline_brier = None
        if cfg.baseline_eval_path.exists():
            payload = json.loads(cfg.baseline_eval_path.read_text(encoding="utf-8"))
            baseline_brier = float(payload["lgbm_calibrated_test"]["brier"])

        # Drift rule:
        # retrain if batch brier is worse than baseline by >= 0.01
        drift_flag = False
        if baseline_brier is not None and (batch_brier - baseline_brier) >= 0.01:
            drift_flag = True

        brier_result = {
            "batch_brier": batch_brier,
            "baseline_brier": baseline_brier,
            "brier_drift_flag": drift_flag,
        }

    # 5) Recommend action
    action = "ok"
    if worst_status == "high":
        action = "retrain_recommended"
    elif worst_status == "warn":
        action = "investigate"

    if brier_result and brier_result.get("brier_drift_flag"):
        action = "retrain_recommended"

    # 6) Report
    report = {
        "timestamp_utc": now.isoformat(),
        "batch_path": str(batch_path),
        "model_version": meta.get("model_version"),
        "input_schema_version": meta.get("input_schema_version"),
        "psi": psi_results,
        "brier": brier_result,
        "recommended_action": action,
    }

    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_path = cfg.report_dir / f"monitor_report_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    # Example usage:
    # python scripts/monitor.py data/fixtures/train_sample.parquet
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/monitor.py <batch.parquet|batch.csv>")

    batch = Path(sys.argv[1])
    result = monitor_batch(batch)
    print(json.dumps(result, indent=2))
