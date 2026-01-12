import json
from pathlib import Path


def test_metrics_regression_gate():
    eval_path = Path("artifacts") / "latest_eval.json"
    assert eval_path.exists(), "Missing artifacts/latest_eval.json. Run: python scripts/train.py"

    payload = json.loads(eval_path.read_text(encoding="utf-8"))

    m = payload["lgbm_calibrated_valid"]
    auroc = float(m["auroc"])
    pr_auc = float(m["pr_auc"])
    brier = float(m["brier"])

    # Phase 1 locked gates (based on your current baseline)
    AUROC_FLOOR = 0.74
    PR_AUC_FLOOR = 0.22
    BRIER_CEILING = 0.075

    assert auroc >= AUROC_FLOOR, f"AUROC regression: {auroc:.4f} < {AUROC_FLOOR}"
    assert pr_auc >= PR_AUC_FLOOR, f"PR-AUC regression: {pr_auc:.4f} < {PR_AUC_FLOOR}"
    assert brier <= BRIER_CEILING, f"Brier regression: {brier:.4f} > {BRIER_CEILING}"

    # Secondary guard: LGBM must beat baseline logistic regression by margin
    base = payload["baseline_logreg_valid"]
    base_auroc = float(base["auroc"])

    assert (auroc - base_auroc) >= 0.10, (
        f"Model improvement regression: AUROC lift {auroc - base_auroc:.4f} < 0.10"
    )
