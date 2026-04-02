"""
========================================================
  Module 2: Exploratory Data Analysis (EDA)
========================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────
PALETTE   = {0: "#E74C3C", 1: "#2ECC71"}   # red=Malignant, green=Benign
CLASS_MAP = {0: "Malignant", 1: "Benign"}


def summary_stats(df: pd.DataFrame) -> None:
    """Print descriptive statistics."""
    print("\n" + "="*60)
    print("  DATASET OVERVIEW")
    print("="*60)
    print(f"  Rows : {df.shape[0]}   |   Columns : {df.shape[1]}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print("\n  Class Distribution:")
    counts = df["target"].value_counts().rename(CLASS_MAP)
    for cls, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"    {cls:<12}: {cnt:>4}  ({pct:.1f}%)")
    print("\n  Descriptive Statistics (first 5 features):")
    print(df.iloc[:, :5].describe().round(3).to_string())
    print("="*60 + "\n")


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar + pie chart of class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold", y=1.01)

    counts = df["target"].value_counts().rename(CLASS_MAP)

    # Bar
    axes[0].bar(counts.index, counts.values,
                color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
    for i, (label, val) in enumerate(counts.items()):
        axes[0].text(i, val + 3, str(val), ha="center", fontweight="bold")
    axes[0].set_title("Count per Class")
    axes[0].set_xlabel("Diagnosis")
    axes[0].set_ylabel("Count")

    # Pie
    axes[1].pie(counts.values, labels=counts.index,
                colors=[PALETTE[0], PALETTE[1]],
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_class_distribution.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def plot_feature_distributions(df: pd.DataFrame, n_features: int = 10) -> None:
    """Histogram of top-N features coloured by class."""
    feature_cols = [c for c in df.columns if c != "target"][:n_features]
    cols  = 5
    rows  = (len(feature_cols) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    fig.suptitle(f"Feature Distributions by Class (first {n_features})",
                 fontsize=14, fontweight="bold")

    for idx, feat in enumerate(feature_cols):
        ax = axes[idx // cols][idx % cols]
        for cls in [0, 1]:
            subset = df[df["target"] == cls][feat]
            ax.hist(subset, bins=25, alpha=0.65,
                    color=PALETTE[cls], label=CLASS_MAP[cls], edgecolor="none")
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide empty subplots
    for j in range(len(feature_cols), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_feature_distributions.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of feature correlations."""
    feature_cols = [c for c in df.columns if c != "target"]
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_correlation_heatmap.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def plot_boxplots(df: pd.DataFrame, n_features: int = 10) -> None:
    """Box-plots of top features, split by class."""
    feature_cols = [c for c in df.columns if c != "target"][:n_features]
    fig, axes    = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Feature Box-Plots by Class", fontsize=14, fontweight="bold")

    melted = df[feature_cols + ["target"]].copy()
    melted["target"] = melted["target"].map(CLASS_MAP)

    for idx, feat in enumerate(feature_cols):
        ax = axes[idx // 5][idx % 5]
        sns.boxplot(data=melted, x="target", y=feat,
                    palette=[PALETTE[0], PALETTE[1]],
                    width=0.5, ax=ax)
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_boxplots.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA pipeline."""
    print("\n[EDA] Starting Exploratory Data Analysis …")
    summary_stats(df)
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    print("[EDA] All plots saved to outputs/\n")


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    run_eda(df)
