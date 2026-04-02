"""
eda.py - Exploratory Data Analysis module.

Generates:
  - Statistical summary
  - Class distribution plot
  - Feature correlation heatmap
  - Amount distribution by class
  - Missing-value report
  - k-distance plot (for DBSCAN eps selection)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config


# ──────────────────────────────────────────────────────────────────────────────
class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature_cols = [c for c in df.columns if c != config.TARGET_COLUMN]

    # ── public API ────────────────────────────────────────────────────────────
    def run(self):
        """Run full EDA pipeline and save all plots."""
        print("=" * 60)
        print("  EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        self._summary_stats()
        self._missing_values()
        self._class_distribution()
        self._amount_distribution()
        self._correlation_heatmap()
        self._feature_boxplots()
        print(f"\n[EDA] All plots saved to → {config.PLOT_DIR}\n")

    def kdistance_plot(self, X_scaled: np.ndarray, k: int = None):
        """
        k-distance graph — the standard method for choosing DBSCAN epsilon.
        The 'elbow' point in the sorted k-distance curve is the ideal eps.
        """
        if k is None:
            k = config.DBSCAN_MIN_SAMPLES

        print(f"[EDA] Computing {k}-NN distances for k-distance plot …")
        nbrs = NearestNeighbors(n_neighbors=k, metric=config.DBSCAN_METRIC)
        nbrs.fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])[::-1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k_distances, color="#E63946", linewidth=1.5, label=f"{k}-NN distance")
        ax.axhline(y=config.DBSCAN_EPS, color="#457B9D", linestyle="--",
                   linewidth=1.5, label=f"Selected ε = {config.DBSCAN_EPS}")
        ax.set_title(f"k-Distance Graph  (k = {k})\n"
                     f"→ The 'elbow' indicates the optimal ε value",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Points sorted by distance (descending)", fontsize=11)
        ax.set_ylabel(f"{k}-NN Distance", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{config.PLOT_DIR}/eda_kdistance.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)
        print(f"[EDA] k-distance plot saved → {path}")

    # ── private helpers ───────────────────────────────────────────────────────
    def _summary_stats(self):
        print("\n[EDA] Dataset Info:")
        print(f"  Rows      : {len(self.df):,}")
        print(f"  Columns   : {len(self.df.columns)}")
        print(f"  Features  : {len(self.feature_cols)}")
        fraud = self.df[config.TARGET_COLUMN].sum()
        total = len(self.df)
        print(f"  Fraud     : {fraud:,}  ({fraud/total*100:.3f}%)")
        print(f"  Normal    : {total - fraud:,}  ({(total-fraud)/total*100:.3f}%)")
        print("\n[EDA] Statistical Summary (Amount):")
        print(self.df["Amount"].describe().to_string())

    def _missing_values(self):
        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            print("\n[EDA] ✅  No missing values detected.")
        else:
            print(f"\n[EDA] ⚠️  Missing values found:\n{missing[missing > 0]}")

    def _class_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Class Distribution — Normal vs Fraud",
                     fontsize=14, fontweight="bold")

        counts = self.df[config.TARGET_COLUMN].value_counts()
        labels = ["Normal (0)", "Fraud (1)"]
        colors = ["#2A9D8F", "#E76F51"]

        # Bar chart
        axes[0].bar(labels, counts.values, color=colors, edgecolor="black",
                    linewidth=0.8, width=0.5)
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 20, f"{v:,}", ha="center", fontsize=11,
                         fontweight="bold")
        axes[0].set_title("Absolute Count", fontsize=12)
        axes[0].set_ylabel("Number of Transactions")
        axes[0].set_ylim(0, counts.max() * 1.15)

        # Pie chart
        axes[1].pie(counts.values, labels=labels, autopct="%1.3f%%",
                    colors=colors, startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 11})
        axes[1].set_title("Proportion", fontsize=12)

        plt.tight_layout()
        path = f"{config.PLOT_DIR}/eda_class_distribution.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)

    def _amount_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Transaction Amount Distribution by Class",
                     fontsize=14, fontweight="bold")

        normal = self.df[self.df[config.TARGET_COLUMN] == 0]["Amount"]
        fraud  = self.df[self.df[config.TARGET_COLUMN] == 1]["Amount"]

        # Histogram (log-scale x)
        axes[0].hist(np.log1p(normal), bins=60, alpha=0.7, color="#2A9D8F",
                     label="Normal", density=True)
        axes[0].hist(np.log1p(fraud),  bins=30, alpha=0.7, color="#E76F51",
                     label="Fraud",  density=True)
        axes[0].set_title("log(Amount+1) Distribution")
        axes[0].set_xlabel("log(Amount + 1)")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # Box plot
        data_to_plot = [np.log1p(normal.values), np.log1p(fraud.values)]
        bp = axes[1].boxplot(data_to_plot, patch_artist=True,
                             labels=["Normal", "Fraud"],
                             medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], ["#2A9D8F", "#E76F51"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title("Box Plot: log(Amount+1)")
        axes[1].set_ylabel("log(Amount + 1)")

        plt.tight_layout()
        path = f"{config.PLOT_DIR}/eda_amount_distribution.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)

    def _correlation_heatmap(self):
        # Show correlation of V1-V10 + Amount with Class
        cols_to_show = [f"V{i}" for i in range(1, 11)] + ["Amount", config.TARGET_COLUMN]
        available = [c for c in cols_to_show if c in self.df.columns]
        corr = self.df[available].corr()

        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, annot_kws={"size": 8},
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Heatmap (V1–V10 + Amount vs Class)",
                     fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        path = f"{config.PLOT_DIR}/eda_correlation_heatmap.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)

    def _feature_boxplots(self):
        """Top 6 features most correlated with fraud."""
        v_cols = [f"V{i}" for i in range(1, 29)]
        available_v = [c for c in v_cols if c in self.df.columns]

        corrs = (self.df[available_v + [config.TARGET_COLUMN]]
                 .corr()[config.TARGET_COLUMN]
                 .drop(config.TARGET_COLUMN)
                 .abs()
                 .sort_values(ascending=False))
        top6 = corrs.head(6).index.tolist()

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Top 6 Features by Correlation with Fraud Class",
                     fontsize=14, fontweight="bold")

        for ax, feat in zip(axes.flat, top6):
            normal = self.df[self.df[config.TARGET_COLUMN] == 0][feat]
            fraud  = self.df[self.df[config.TARGET_COLUMN] == 1][feat]
            bp = ax.boxplot([normal.values, fraud.values],
                            patch_artist=True, labels=["Normal", "Fraud"],
                            medianprops=dict(color="black", linewidth=2))
            for patch, color in zip(bp["boxes"], ["#2A9D8F", "#E76F51"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            corr_val = corrs[feat]
            ax.set_title(f"{feat}  (|corr| = {corr_val:.3f})", fontsize=11)
            ax.set_ylabel("Feature Value")

        plt.tight_layout()
        path = f"{config.PLOT_DIR}/eda_feature_boxplots.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)
