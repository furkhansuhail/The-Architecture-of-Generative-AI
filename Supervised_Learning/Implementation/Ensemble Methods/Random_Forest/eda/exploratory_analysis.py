"""
=============================================================
  eda/exploratory_analysis.py  –  Visual & statistical EDA
=============================================================
Works with both Titanic and Breast Cancer Wisconsin datasets.
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

PALETTE   = "Set2"
OUT_DIR   = config.OUTPUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)


class EDA:
    """
    Runs a suite of exploratory analyses on the raw and clean dataframes.

    Methods (all return self for chaining):
      summary()             – shape, dtypes, missing-value report
      class_balance()       – target distribution bar chart
      numeric_dist()        – histograms for all numeric features
      correlation()         – Pearson correlation heatmap
      target_by_feature()   – class breakdown by key features
      run()                 – execute all of the above
    """

    def __init__(self, raw_df: pd.DataFrame, clean_df: pd.DataFrame):
        self.raw   = raw_df
        self.clean = clean_df

    def _save(self, fig, name: str):
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [EDA] saved → {path}")

    def summary(self) -> "EDA":
        print("\n" + "=" * 60)
        print("  EDA SUMMARY")
        print("=" * 60)
        print(f"\n  Raw shape   : {self.raw.shape}")
        print(f"  Clean shape : {self.clean.shape}")
        miss = self.raw.isnull().sum()
        miss = miss[miss > 0]
        print("\n  Missing values (raw):")
        print(miss.to_string() if not miss.empty else "  None")
        print("\n  Descriptive statistics:")
        desc = self.clean.describe().T
        print(desc[["mean", "std", "min", "50%", "max"]].to_string())
        return self

    def class_balance(self) -> "EDA":
        target = config.TARGET_COLUMN
        counts = self.clean[target].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = sns.color_palette(PALETTE, len(counts))
        bars   = ax.bar([str(v) for v in counts.index], counts.values,
                        color=colors, edgecolor="white", width=0.5)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.02,
                    f"{v}  ({v/counts.sum():.1%})", ha="center", fontsize=11)
        ax.set(title=f"Class Balance — {target}", ylabel="Count",
               ylim=(0, counts.max() * 1.2))
        sns.despine()
        self._save(fig, "01_class_balance.png")
        return self

    def numeric_dist(self) -> "EDA":
        num_cols = [c for c in self.clean.select_dtypes(include=np.number).columns
                    if c != config.TARGET_COLUMN]
        n = len(num_cols)
        ncols = 5
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
        axes = axes.flatten()
        palette = sns.color_palette(PALETTE, 6)
        for i, col in enumerate(num_cols):
            axes[i].hist(self.clean[col], bins=25,
                         color=palette[i % 6], edgecolor="white", alpha=0.85)
            axes[i].set_title(col[:20], fontsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Feature Distributions", fontsize=14, y=1.01)
        fig.tight_layout()
        self._save(fig, "02_numeric_distributions.png")
        return self

    def correlation(self) -> "EDA":
        num_df = self.clean.select_dtypes(include=np.number)
        corr   = num_df.corr()
        size   = max(8, min(16, len(corr) // 2))
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        mask   = np.triu(np.ones_like(corr, dtype=bool))
        annot  = len(corr) <= 20
        sns.heatmap(corr, mask=mask, annot=annot,
                    fmt=".2f" if annot else "",
                    cmap="coolwarm", center=0, linewidths=0.3, ax=ax)
        ax.set_title("Pearson Correlation Heatmap", fontsize=13)
        self._save(fig, "03_correlation_heatmap.png")
        return self

    def target_by_feature(self) -> "EDA":
        target = config.TARGET_COLUMN
        cats   = [c for c in ["Sex", "Pclass", "Embarked", "AgeGroup"]
                  if c in self.clean.columns]
        if cats:
            fig, axes = plt.subplots(1, len(cats), figsize=(16, 5))
            for ax, col in zip(axes, cats):
                srv = self.clean.groupby(col)[target].mean().reset_index()
                ax.bar(srv[col].astype(str), srv[target],
                       color=sns.color_palette(PALETTE, len(srv)), edgecolor="white")
                ax.set(title=f"Rate by {col}", ylabel="Rate", ylim=(0, 1))
                ax.axhline(self.clean[target].mean(), ls="--", color="grey")
                sns.despine(ax=ax)
            fig.suptitle("Positive-Class Rate by Categorical Feature", fontsize=13)
        else:
            num_cols = [c for c in self.clean.columns if c != target]
            means_diff = (
                self.clean.groupby(target)[num_cols].mean()
                .diff().abs().iloc[-1].sort_values(ascending=False)
            )
            top_n = min(5, len(num_cols))
            top   = means_diff.head(top_n).index.tolist()
            fig, axes = plt.subplots(1, top_n, figsize=(18, 5))
            palette   = sns.color_palette(PALETTE, 2)
            for ax, col in zip(axes, top):
                for cls_val, color, label in zip(
                    sorted(self.clean[target].unique()), palette,
                    ["Malignant (0)", "Benign (1)"],
                ):
                    data = self.clean.loc[self.clean[target] == cls_val, col]
                    ax.hist(data, bins=20, alpha=0.65, color=color,
                            label=label, edgecolor="white")
                ax.set(title=col[:22], xlabel="value")
                ax.legend(fontsize=7)
                sns.despine(ax=ax)
            fig.suptitle("Top-5 Most Discriminating Features by Class", fontsize=13)
        fig.tight_layout()
        self._save(fig, "04_target_by_feature.png")
        return self

    def run(self) -> "EDA":
        print("\n[EDA] Running full exploratory analysis …")
        self.summary().class_balance().numeric_dist().correlation().target_by_feature()
        print("[EDA] Done.\n")
        return self
