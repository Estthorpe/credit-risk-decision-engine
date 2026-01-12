"""Standard EDA (artifact-producing) script.

Outputs (created under artifacts/eda/):
- dataset_overview.json
- columns.csv
- dtypes.csv
- missingness.csv
- numeric_summary.csv
- categorical_summary.csv
- correlations_numeric.csv
- eda_report.md
- plots/*.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from credit_risk_decision_engine.config import SETTINGS

ARTIFACTS_DIR = Path("artifacts") / "eda"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

def _safe_mkdir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


def load_processed_table() -> pd.DataFrame:
    path = SETTINGS.processed_data_dir / "train_table.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run ingestion first:\n"
            "  python scripts\\ingest.py"
        )
    return pd.read_parquet(path)


def dataset_overview(df: pd.DataFrame, target_col: str = "TARGET") -> Dict[str, Any]:
    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "n_duplicates": int(df.duplicated().sum()),
        "n_missing_total": int(df.isna().sum().sum()),
        "missing_pct_total": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])),
    }
    if target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False).to_dict()
        overview["target_value_counts"] = {str(k): int(v) for k, v in vc.items()}
        if df[target_col].notna().any():
            overview["target_positive_rate"] = float(df[target_col].mean())
    return overview


def write_core_tables(df: pd.DataFrame) -> None:
    pd.Series(df.columns, name="column").to_csv(ARTIFACTS_DIR / "columns.csv", index=False)

    #Dtypes
    dtypes = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    dtypes.to_csv(ARTIFACTS_DIR / "dtypes.csv", index=False)


    #missingness
    miss = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100.0),
    }).sort_values("missing_pct", ascending=False)
    miss.to_csv(ARTIFACTS_DIR / "missingness.csv", index=False)

    #numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        numeric_summary =  df[num_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]) . T
        numeric_summary.to_csv(ARTIFACTS_DIR / "numeric_summary.csv")

    #Categorical summary
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        rows = []
        for c in cat_cols:
            s = df[c]
            rows.append({
                "column": c,
                "n_unique": int(s.nunique(dropna=True)),
                "missing_pct": float(s.isna().mean() * 100.0),
                "top_value": None if s.dropna().empty else str(s.dropna().mode().iloc[0]),
                "top_freq": 0 if s.dropna().empty else int((s == s.dropna().mode().iloc[0]).sum()),
            })
        pd.DataFrame(rows).sort_values("n_unique", ascending=False).to_csv(
            ARTIFACTS_DIR / "categorical_summary.csv", index=False
        )


def plot_missingness(missingness_csv: Path) -> None:
    miss = pd.read_csv(missingness_csv)
    top = miss.head(30).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["column"], top["missing_pct"])
    plt.title("Top 30 columns by missingness (%)")
    plt.xlabel("Missing %")
    _savefig(PLOTS_DIR / "missingness_top30.png")


def plot_target_distribution(df: pd.DataFrame, target_col: str = "TARGET") -> None:
    if target_col not in df.columns:
        return
    vc = df[target_col].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar([str(i) for i in vc.index], vc.values)
    plt.title(f"Target distribution: {target_col}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    _savefig(PLOTS_DIR / "target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 12) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop in ["SK_ID_CURR", "TARGET"]:
        if drop in num_cols:
            num_cols.remove(drop)

    num_cols = sorted(num_cols, key=lambda c: df[c].isna().mean())[:max_cols]

    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(s.values, bins=50)
        plt.title(f"Histogram: {c}")
        _savefig(PLOTS_DIR / f"hist_{c}.png")


def compute_correlations(df: pd.DataFrame, target_col: str = "TARGET") -> None:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2:
        return

    corr = num.corr(numeric_only=True)
    corr.to_csv(ARTIFACTS_DIR / "correlations_numeric.csv")

    # Also produce “top correlations with target” if available
    if target_col in corr.columns:
        top = corr[target_col].drop(index=[target_col], errors="ignore").sort_values(key=np.abs, ascending=False)
        top.head(30).to_csv(ARTIFACTS_DIR / "top_corr_with_target.csv", header=[f"corr_with_{target_col}"])


def write_markdown_report(overview: Dict[str, Any]) -> None:
    md = []
    md.append("# EDA Report (Standard)\n")
    md.append("## Dataset Overview\n")
    md.append(f"- Rows: **{overview['rows']}**\n")
    md.append(f"- Columns: **{overview['columns']}**\n")
    md.append(f"- Memory: **{overview['memory_mb']:.2f} MB**\n")
    md.append(f"- Duplicated rows: **{overview['n_duplicates']}**\n")
    md.append(f"- Total missing cells: **{overview['n_missing_total']}** "
              f"({overview['missing_pct_total']*100:.2f}%)\n")

    if "target_value_counts" in overview:
        md.append("\n## Target\n")
        md.append(f"- TARGET value counts: `{overview['target_value_counts']}`\n")
        if "target_positive_rate" in overview:
            md.append(f"- TARGET positive rate: **{overview['target_positive_rate']:.4f}**\n")

    md.append("\n## Artifacts Generated\n")
    md.append("- `columns.csv`, `dtypes.csv`\n")
    md.append("- `missingness.csv`, `numeric_summary.csv`, `categorical_summary.csv`\n")
    md.append("- `correlations_numeric.csv`, `top_corr_with_target.csv`\n")
    md.append("- Plots in `plots/`\n")

    (ARTIFACTS_DIR / "eda_report.md").write_text("".join(md), encoding="utf-8")


def main() -> None:
    _safe_mkdir()
    df = load_processed_table()

    overview = dataset_overview(df)
    (ARTIFACTS_DIR / "dataset_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    write_core_tables(df)

    plot_target_distribution(df)
    plot_missingness(ARTIFACTS_DIR / "missingness.csv")
    plot_numeric_distributions(df, max_cols=12)

    compute_correlations(df)

    write_markdown_report(overview)

    print(f"EDA artifacts written to: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()



