from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib

@dataclass(frozen=True)
class BundlePaths:
    root: Path

    @property
    def model_path(self) -> Path:
        return self.root / "model.joblib"
    
    @property
    def calibrator_path(self) -> Path:
        return self.root / "calibrator.joblib"
    
    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"
    
    @property
    def reference_stats_path(self) -> Path:
        return self.root / "reference_stats.json"
    
def save_bundle(
    bundle_dir: Path,
    model: Any,
    calibrator: Any,
    metadata: Dict[str, Any],
    reference_stats: Dict[str, Any],
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    p = BundlePaths(bundle_dir)
    joblib.dump(model, p.model_path)
    joblib.dump(calibrator, p.calibrator_path)
    p.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    p.reference_stats_path.write_text(json.dumps(reference_stats, indent=2), encoding="utf-8")

        
def load_bundle(bundle_dir: Path) -> Dict[str, Any]:
    p = BundlePaths(bundle_dir)
    return {
        "model": joblib.load(p.model_path),
        "calibrator": joblib.load(p.calibrator_path),
        "metadata": json.loads(p.metadata_path.read_text(encoding="utf-8")),
        "reference_stats": json.loads(p.reference_stats_path.read_text(encoding="utf-8")),
    }