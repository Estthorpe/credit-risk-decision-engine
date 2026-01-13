from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List

from credit_risk_decision_engine.modeling.bundle import load_bundle


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    calibrator: Any
    meta: Dict[str, Any]
    reference_stats: Dict[str, Any]
    feature_columns: List[str]


def load_latest_bundle(bundle_dir: Path) -> Tuple[Path, ModelBundle]:
    latest_ptr = bundle_dir / "latest" / "PATH.txt"
    if not latest_ptr.exists():
        raise FileNotFoundError("Missing bundle/latest/PATH.txt. Run: python scripts/train.py")

    target_path = Path(latest_ptr.read_text(encoding="utf-8").strip())
    if not target_path.exists():
        raise FileNotFoundError(f"Bundle path in PATH.txt does not exist: {target_path}")

    payload = load_bundle(target_path)

    return target_path, ModelBundle(
        model=payload["model"],
        calibrator=payload["calibrator"],
        meta=payload["metadata"],
        reference_stats=payload["reference_stats"],
        feature_columns=payload["feature_columns"],
    )
