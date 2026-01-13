from __future__ import annotations

from typing import Any, Dict, List


def reason_codes_from_features(features: Dict[str, Any], pd: float) -> List[str]:
    """
    Phase 1: lightweight, deterministic reasons (auditable).
    Later: replace with SHAP-based explanations.
    """
    reasons: List[str] = []

    ext2 = features.get("EXT_SOURCE_2")
    if ext2 is not None:
        try:
            if float(ext2) < 0.3:
                reasons.append("EXT_SOURCE_2_HIGH_RISK")
        except Exception:
            pass

    days_birth = features.get("DAYS_BIRTH")
    if days_birth is not None:
        try:
            # DAYS_BIRTH is negative days; closer to 0 => younger
            if float(days_birth) > -12000:
                reasons.append("DAYS_BIRTH_YOUNG")
        except Exception:
            pass

    income = features.get("AMT_INCOME_TOTAL")
    if income is not None:
        try:
            if float(income) < 90000:
                reasons.append("INCOME_LOW")
        except Exception:
            pass

    if pd >= 0.4:
        reasons.append("PD_HIGH")

    return reasons[:3] if reasons else ["NO_STRONG_REASON_CODES"]
