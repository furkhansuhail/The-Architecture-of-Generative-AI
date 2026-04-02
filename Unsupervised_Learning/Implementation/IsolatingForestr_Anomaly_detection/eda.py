"""
eda.py - Exploratory Data Analysis module.

Generates and saves:
  1. Class distribution bar chart
  2. Feature distributions (normal vs fraud overlay)
  3. Correlation heat-map (top N features)
  4. Amount & Time distributions split by class
  5. Printed statistical summary
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import (OUTPUT_DIR, CLASS_COLUMN, CORR_TOP_N,
                    FIGURE_DPI, PALETTE)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=0.9)


def run_eda(df: pd.DataFrame) -> None:
    """Entry-point: run all EDA steps and persist plots to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("──── EDA ────────────────────────────────────────")

    _print_summary(df)
    _plot_class_distribution(df)
    _plot_amount_time(df)
    _plot_feature_distributions(df)
    _plot_correlation_heatmap(df)

    logger.info("EDA plots saved to %s", OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    """Log a concise statistical summary."""
    n_normal  = (df[CLASS_COLUMN] == 0).sum()
    n_fraud   = (df[CLASS_COLUMN] == 1).sum()
    fraud_pct = 100 * n_fraud / len(df)

    logger.info("\n%s", "=" * 55)
    logger.info("  Dataset Summary")
    logger.info("  Total samples : %d", len(df))
    logger.info("  Normal        : %d  (%.2f %%)", n_normal, 100 - fraud_pct)
    logger.info("  Anomaly/Fraud : %d  (%.2f %%)", n_fraud, fraud_pct)
    logger.info("  Features      : %d", df.shape[1] - 1)
    logger.info("  Missing vals  : %d", df.isnull().sum().sum())
    logger.info("\n  Numeric stats:\n%s", df.describe().to_string())
    logger.info("%s\n", "=" * 55)


def _plot_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df[CLASS_COLUMN].value_counts()
    colors = [PALETTE["normal"], PALETTE["anomaly"]]

    # Bar chart
    axes[0].bar(["Normal", "Anomaly"], counts.values, color=colors, edgecolor="k", linewidth=0.6)
    axes[0].set_title("Class Distribution (count)")
    axes[0].set_ylabel("Number of Samples")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=9)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=["Normal", "Anomaly"],
        autopct="%1.2f%%",
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor="w", linewidth=1.5),
    )
    axes[1].set_title("Class Distribution (proportion)")

    plt.suptitle("Class Balance", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "01_class_distribution.png")


def _plot_amount_time(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    classes   = {0: ("Normal", PALETTE["normal"]), 1: ("Anomaly", PALETTE["anomaly"])}

    for cls_val, (label, color) in classes.items():
        subset = df[df[CLASS_COLUMN] == cls_val]
        col    = 0 if cls_val == 0 else 1

        axes[0, col].hist(subset["Amount"], bins=60, color=color,
                          edgecolor="none", alpha=0.85)
        axes[0, col].set_title(f"Amount Distribution — {label}")
        axes[0, col].set_xlabel("Transaction Amount")
        axes[0, col].set_ylabel("Frequency")
        axes[0, col].set_yscale("log")

        axes[1, col].hist(subset["Time"], bins=60, color=color,
                          edgecolor="none", alpha=0.85)
        axes[1, col].set_title(f"Time Distribution — {label}")
        axes[1, col].set_xlabel("Time (seconds)")
        axes[1, col].set_ylabel("Frequency")

    plt.suptitle("Amount & Time Distributions by Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_amount_time.png")


def _plot_feature_distributions(df: pd.DataFrame, n_features: int = 12) -> None:
    """Overlay KDE plots for each V-feature (normal vs fraud)."""
    v_cols  = [c for c in df.columns if c.startswith("V")][:n_features]
    ncols   = 4
    nrows   = int(np.ceil(len(v_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes_flat = axes.flatten()

    normal = df[df[CLASS_COLUMN] == 0]
    fraud  = df[df[CLASS_COLUMN] == 1]

    for i, col in enumerate(v_cols):
        ax = axes_flat[i]
        ax.hist(normal[col], bins=50, color=PALETTE["normal"],
                alpha=0.6, density=True, label="Normal")
        ax.hist(fraud[col],  bins=50, color=PALETTE["anomaly"],
                alpha=0.6, density=True, label="Anomaly")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Feature Distributions — Normal vs Anomaly",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_feature_distributions.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heat-map of the top-N most correlated features with the class label."""
    num_df   = df.select_dtypes(include=[np.number])
    corr_mat = num_df.corr()

    # Pick top N features by absolute correlation with CLASS_COLUMN
    if CLASS_COLUMN in corr_mat.columns:
        top_feats = (corr_mat[CLASS_COLUMN]
                     .abs()
                     .sort_values(ascending=False)
                     .head(CORR_TOP_N)
                     .index.tolist())
        sub_corr  = corr_mat.loc[top_feats, top_feats]
    else:
        sub_corr  = corr_mat.iloc[:CORR_TOP_N, :CORR_TOP_N]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask    = np.triu(np.ones_like(sub_corr, dtype=bool))
    sns.heatmap(
        sub_corr, mask=mask, ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cmap="RdBu_r", center=0, linewidths=0.4,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(f"Correlation Heat-map (top {CORR_TOP_N} features by |corr| with Class)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_correlation_heatmap.png")


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", name)
