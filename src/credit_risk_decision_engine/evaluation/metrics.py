from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


@dataclass(frozen=True)
class MetricResult:
    auroc: float
    pr_auc: float
    brier: float


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> MetricResult:
    return MetricResult(
        auroc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
    )


def as_dict(m: MetricResult)  -> Dict[str, float]:
    return { "auroc": m.auroc, "pr_auc": m.pr_auc, "brier": m.brier}