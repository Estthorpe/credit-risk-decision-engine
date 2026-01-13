from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

def _safe_clip(p: float, eps: float = 1e-6) -> float:
    return float(max(p, eps))

def compute_psi_from_edges(
        ref_expected: List[float],
        bin_edges: List[float],
        current: pd.Series,
) -> float:
    """
    PSI - sum( (act - exp) * In(act/exp) ) over bins
    -bin_edges: len = nbins+1
    -ref_expected: len = nbins
    """

    s = current.dropna().to_numpy()
    if s.size == 0:
        return 0.0

    counts, _ = np.histogram(s, bins=np.array(bin_edges, dtype=float))
    actual = counts / max(counts.sum(), 1)

    psi = 0.0
    for act, exp in zip(actual.tolist(), ref_expected):
        a = _safe_clip(float(act))
        e = _safe_clip(float(exp))
        psi += (a - e) * math.log(a / e)
    return float(psi)


def psi_status(psi: float) -> str:
    if psi < 0.1:
        return "ok"
    elif psi < 0.25:
        return "warn"
    else:
        return "high"