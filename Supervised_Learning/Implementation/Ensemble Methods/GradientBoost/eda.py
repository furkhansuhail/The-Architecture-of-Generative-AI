"""
=============================================================
  Module 2 : Exploratory Data Analysis (EDA)
  ─────────────────────────────────────────────────────────
  Generates & saves:
    • dataset_overview.png  – shape, dtypes, missing heatmap
    • target_distribution.png
    • numeric_distributions.png
    • correlation_heatmap.png
    • categorical_vs_target.png
    • numeric_vs_target.png
=============================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (REPORTS_DIR, TARGET_COL, NUMERIC_COLS,
                    CATEGORICAL_COLS, FIG_DPI, PALETTE)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette=PALETTE)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Public API ───────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> dict:
    """
    Run full EDA pipeline and save all figures to REPORTS_DIR.
    Returns a summary dict with key statistics.
    """
    print("\n" + "═"*60)
    print("  MODULE 2 — Exploratory Data Analysis")
    print("═"*60)

    summary = _print_overview(df)
    _plot_missing(df)
    _plot_target(df)
    _plot_numeric_dists(df)
    _plot_correlation(df)
    _plot_categoricals_vs_target(df)
    _plot_numerics_vs_target(df)

    print(f"\n[EDA] All figures saved to → {REPORTS_DIR}/")
    return summary


# ── Helpers ──────────────────────────────────────────────────

def _print_overview(df: pd.DataFrame) -> dict:
    target_counts = df[TARGET_COL].value_counts()
    imbalance_ratio = round(target_counts[0] / target_counts[1], 2)

    print(f"\n  Shape          : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Numeric cols   : {len(NUMERIC_COLS)}")
    print(f"  Categorical cols: {len(CATEGORICAL_COLS)}")
    print(f"  Missing values : {df.isnull().sum().sum():,}")
    print(f"  Class balance  : <=50K={target_counts[0]:,}  |  >50K={target_counts[1]:,}  "
          f"(imbalance ratio {imbalance_ratio}:1)")

    return {
        "n_rows": df.shape[0], "n_cols": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
        "class_0": int(target_counts[0]),
        "class_1": int(target_counts[1]),
        "imbalance_ratio": imbalance_ratio,
    }


def _plot_missing(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview — Missing Values", fontsize=14, fontweight="bold")

    # Bar chart of missing per column
    if len(missing):
        missing.plot(kind="bar", ax=axes[0], color="#E74C3C", edgecolor="white")
        axes[0].set_title("Missing Values per Column")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=45)
    else:
        axes[0].text(0.5, 0.5, "No Missing Values ✓",
                     ha="center", va="center", fontsize=14,
                     color="green", transform=axes[0].transAxes)
        axes[0].set_title("Missing Values")

    # Heatmap
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False,
                cmap="viridis", ax=axes[1])
    axes[1].set_title("Missing Value Heatmap")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "01_missing_overview.png"), dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✓ 01_missing_overview.png")


def _plot_target(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Distribution — Income", fontsize=14, fontweight="bold")

    counts = df[TARGET_COL].value_counts()
    labels = ["≤50K", ">50K"]
    colors = ["#3498DB", "#E74C3C"]

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Class Counts")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Proportions")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "02_target_distribution.png"), dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✓ 02_target_distribution.png")


def _plot_numeric_dists(df: pd.DataFrame):
    n = len(NUMERIC_COLS)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold")

    for i, col in enumerate(NUMERIC_COLS):
        # Histogram + KDE
        sns.histplot(df[col].dropna(), ax=axes[0, i], kde=True,
                     color="#2ECC71", edgecolor="white", bins=30)
        axes[0, i].set_title(col.replace("_", " ").title())
        axes[0, i].set_xlabel("")

        # Box split by target
        sns.boxplot(x=TARGET_COL, y=col, data=df,
                    ax=axes[1, i], palette=["#3498DB", "#E74C3C"])
        axes[1, i].set_xlabel("Income (0=≤50K, 1=>50K)")
        axes[1, i].set_title(f"{col.replace('_', ' ').title()} vs Target")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "03_numeric_distributions.png"), dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✓ 03_numeric_distributions.png")


def _plot_correlation(df: pd.DataFrame):
    num_df = df[NUMERIC_COLS + [TARGET_COL]].dropna()
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Numeric Features + Target)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "04_correlation_heatmap.png"), dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✓ 04_correlation_heatmap.png")


def _plot_categoricals_vs_target(df: pd.DataFrame):
    cat_cols = [c for c in CATEGORICAL_COLS
                if c in df.columns and df[c].nunique() <= 20][:6]
    n = len(cat_cols)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("Categorical Features vs Target Income", fontsize=14, fontweight="bold")

    for i, col in enumerate(cat_cols):
        ratio = (df.groupby(col)[TARGET_COL].mean() * 100).sort_values(ascending=False)
        ratio.plot(kind="bar", ax=axes[i], color="#9B59B6", edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_ylabel(">50K rate (%)")
        axes[i].tick_params(axis="x", rotation=40)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "05_categorical_vs_target.png"), dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✓ 05_categorical_vs_target.png")


def _plot_numerics_vs_target(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle("Numeric Features — KDE split by Target", fontsize=14, fontweight="bold")

    for i, col in enumerate(NUMERIC_COLS):
        for cls, color, lbl in [(0, "#3498DB", "≤50K"), (1, "#E74C3C", ">50K")]:
            sns.kdeplot(df.loc[df[TARGET_COL]==cls, col].dropna(),
                        ax=axes[i], fill=True, alpha=0.4,
                        color=color, label=lbl)
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].legend()
        axes[i].set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "06_numeric_vs_target_kde.png"), dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✓ 06_numeric_vs_target_kde.png")


if __name__ == "__main__":
    from modules.data_loader import load_data
    df = load_data()
    run_eda(df)
