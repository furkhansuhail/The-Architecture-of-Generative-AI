"""
Module 2: Exploratory Data Analysis (EDA)
==========================================
Generates descriptive statistics and publication-quality plots.
All figures are saved to the /outputs directory.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from data_loader import get_feature_descriptions

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = {"0": "#2ecc71", "1": "#e74c3c"}
BG_COLOR  = "#F8F9FA"
GRID_COLOR = "#DEE2E6"
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "font.family":      "DejaVu Sans",
})


def _save(fig: plt.Figure, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [EDA] Saved → {path}")
    return path


# ── 1. Summary statistics ────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  DATASET SUMMARY")
    print("═"*60)
    print(f"  Rows        : {df.shape[0]}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Missing     : {df.isnull().sum().sum()}")
    vc = df["target"].value_counts()
    print(f"  Class dist  : No Disease={vc.get(0,0)}  Disease={vc.get(1,0)}")
    pct = vc.get(1,0) / len(df) * 100
    print(f"  Positive %  : {pct:.1f}%")
    print("═"*60)
    print("\nDescriptive Statistics:")
    print(df.describe().round(2).to_string())


# ── 2. Class balance bar ─────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Class Distribution", fontsize=16, fontweight="bold", y=1.02)

    vc   = df["target"].value_counts().sort_index()
    lbls = ["No Disease", "Heart Disease"]
    clrs = ["#2ecc71", "#e74c3c"]

    # Bar chart
    bars = axes[0].bar(lbls, vc.values, color=clrs, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vc.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(v), ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Count per Class")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, vc.max() * 1.15)

    # Pie chart
    axes[1].pie(vc.values, labels=lbls, autopct="%1.1f%%",
                colors=clrs, startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Proportion")

    fig.tight_layout()
    return _save(fig, "01_class_distribution.png")


# ── 3. Numeric feature distributions ─────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]

    n_cols = 3
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5, n_rows * 4))
    fig.suptitle("Feature Distributions by Target Class",
                 fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for cls, color in zip([0, 1], ["#2ecc71", "#e74c3c"]):
            subset = df[df["target"] == cls][col].dropna()
            axes[i].hist(subset, bins=20, alpha=0.65, color=color,
                         label=["No Disease", "Heart Disease"][cls],
                         edgecolor="white")
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return _save(fig, "02_feature_distributions.png")


# ── 4. Correlation heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
        vmin=-1, vmax=1,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    return _save(fig, "03_correlation_heatmap.png")


# ── 5. Boxplots by class ─────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> str:
    cont = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(cont), figsize=(18, 5))
    fig.suptitle("Key Feature Boxplots by Disease Status",
                 fontsize=16, fontweight="bold")

    for ax, col in zip(axes, cont):
        data = [df[df["target"] == 0][col].dropna(),
                df[df["target"] == 1][col].dropna()]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No Disease", "Heart Disease"], fontsize=9)
        ax.set_title(col, fontweight="bold")

    fig.tight_layout()
    return _save(fig, "04_boxplots.png")


# ── 6. Missing values ────────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame) -> str:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  [EDA] No missing values to plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    missing.sort_values(ascending=True).plot.barh(ax=ax, color="#e67e22")
    ax.set_title("Missing Values per Feature", fontweight="bold")
    ax.set_xlabel("Count")
    fig.tight_layout()
    return _save(fig, "05_missing_values.png")


# ── Master runner ─────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> dict:
    print("\n[EDA] Starting exploratory data analysis …")
    print_summary(df)
    paths = {}
    paths["class_dist"]    = plot_class_distribution(df)
    paths["distributions"] = plot_feature_distributions(df)
    paths["heatmap"]       = plot_correlation_heatmap(df)
    paths["boxplots"]      = plot_boxplots(df)
    paths["missing"]       = plot_missing_values(df)
    print("[EDA] Done.\n")
    return paths


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data
    run_eda(load_data())
