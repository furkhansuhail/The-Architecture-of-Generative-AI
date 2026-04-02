"""
eda.py — Exploratory Data Analysis with automated plot generation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from config import TARGET_COLUMN, PLOTS_DIR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    """Print dataset overview to console."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  DATASET SUMMARY")
    print(sep)
    print(f"  Shape      : {df.shape}")
    print(f"  Memory     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\n  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
    print(f"\n  Class balance:\n{df[TARGET_COLUMN].value_counts().to_string()}")
    print(sep)
    print(df.describe().round(2).to_string())
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class Distribution — Diabetes Outcome", fontsize=14, fontweight="bold")

    counts = df[TARGET_COLUMN].value_counts()
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    colors = ["#4C72B0", "#DD8452"]

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Count")

    # Pie chart
    axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    axes[1].set_title("Proportion")

    plt.tight_layout()
    path = f"{PLOTS_DIR}/01_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame) -> str:
    features = [c for c in df.columns if c != TARGET_COLUMN]
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        for cls, color, label in [(0, "#4C72B0", "Non-Diabetic"), (1, "#DD8452", "Diabetic")]:
            subset = df[df[TARGET_COLUMN] == cls][feat].dropna()
            axes[i].hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor="none")
        axes[i].set_title(feat, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/02_feature_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/03_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> str:
    features = [c for c in df.columns if c != TARGET_COLUMN]
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.suptitle("Feature Boxplots by Class", fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        data_to_plot = [
            df[df[TARGET_COLUMN] == 0][feat].dropna(),
            df[df[TARGET_COLUMN] == 1][feat].dropna(),
        ]
        bp = axes[i].boxplot(
            data_to_plot, patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        colors = ["#4C72B0", "#DD8452"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_xticklabels(["Non-Diabetic", "Diabetic"])
        axes[i].set_title(feat, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/04_boxplots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA pipeline."""
    print_summary(df)
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    print("[EDA] All plots saved.\n")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_data
    run_eda(load_data())
