"""
Inter-annotator agreement between annotator 1 and annotator 2 (simulated) sheets.

Loads completed annotation CSVs, aligns on ``id``, and reports percent agreement,
Cohen's Kappa, confusion matrix, and per-class agreement statistics.

Usage (from project root):

    python calculate_kappa.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATOR1_PATH = PROJECT_ROOT / "dataset" / "annotations" / "annotator1_completed_final.csv"
ANNOTATOR2_PATH = PROJECT_ROOT / "dataset" / "annotations" / "annotator2_completed.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "kappa_confusion_matrix.png"

LABELS = ["Pratyaksha", "Anumana", "Upamana", "Shabda"]


def interpret_kappa(kappa: float) -> str:
    if kappa < 0.4:
        return "weak agreement"
    if kappa < 0.6:
        return "moderate agreement"
    if kappa < 0.8:
        return "substantial agreement"
    return "strong agreement"


def per_class_agreement(a1: pd.Series, a2: pd.Series) -> pd.DataFrame:
    """Per-label counts and agreement rates."""
    rows = []
    n = len(a1)
    for label in LABELS:
        m1 = a1 == label
        m2 = a2 == label
        both = (m1 & m2).sum()
        c1 = int(m1.sum())
        c2 = int(m2.sum())
        either = int((m1 | m2).sum())
        rows.append(
            {
                "pramana_label": label,
                "both_agree": int(both),
                "annotator1_count": c1,
                "annotator2_count": c2,
                "pct_of_total_both_agree": round(100.0 * both / n, 2) if n else 0.0,
                "agreement_when_annotator1_label": round(100.0 * both / c1, 2) if c1 else None,
                "agreement_when_annotator2_label": round(100.0 * both / c2, 2) if c2 else None,
                "agreement_when_either_label": round(100.0 * both / either, 2) if either else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    for path in (ANNOTATOR1_PATH, ANNOTATOR2_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    df1 = pd.read_csv(ANNOTATOR1_PATH)
    df2 = pd.read_csv(ANNOTATOR2_PATH)

    for name, df in ("annotator1", df1), ("annotator2", df2):
        if "id" not in df.columns or "pramana_label" not in df.columns:
            raise ValueError(f"{name} CSV must contain id and pramana_label")

    merged = df1[["id", "pramana_label"]].merge(
        df2[["id", "pramana_label"]],
        on="id",
        how="inner",
        suffixes=("_a1", "_a2"),
    )

    if len(merged) != len(df1) or len(merged) != len(df2):
        raise ValueError(
            "ID mismatch between annotators: "
            f"a1={len(df1)}, a2={len(df2)}, merged={len(merged)}"
        )

    y1 = merged["pramana_label_a1"].astype(str).str.strip()
    y2 = merged["pramana_label_a2"].astype(str).str.strip()

    invalid = set(y1.unique()) | set(y2.unique())
    invalid -= set(LABELS)
    if invalid:
        raise ValueError(f"Unexpected labels: {invalid}")

    n = len(merged)
    agreements = int((y1 == y2).sum())
    disagreements = n - agreements
    agreement_pct = 100.0 * agreements / n if n else 0.0
    kappa = float(cohen_kappa_score(y1, y2, labels=LABELS))

    cm = confusion_matrix(y1, y2, labels=LABELS)

    print("=" * 60)
    print("Inter-annotator agreement (NyayaXAI / InferAI)")
    print("=" * 60)
    print(f"Total examples (matched on id): {n}")
    print(f"Agreements:   {agreements}")
    print(f"Disagreements: {disagreements}")
    print(f"Agreement percentage: {agreement_pct:.2f}%")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Interpretation: {interpret_kappa(kappa)}")
    print()

    print("Confusion matrix (rows = annotator 1, columns = annotator 2):")
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print(cm_df.to_string())
    print()

    print("Per-class agreement statistics:")
    stats = per_class_agreement(y1, y2)
    print(stats.to_string(index=False))
    print()

    # Plot
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Annotator 2")
    plt.ylabel("Annotator 1")
    plt.title(f"Confusion matrix (κ = {kappa:.3f}, agreement = {agreement_pct:.1f}%)")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print(f"Figure saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
