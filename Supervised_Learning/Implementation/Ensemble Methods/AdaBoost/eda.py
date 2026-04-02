"""
eda.py
------
Exploratory Data Analysis for the Heart Disease dataset.
Generates and saves a multi-panel EDA report as a PNG figure.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

PALETTE = {"0": "#2196F3", "1": "#F44336", 0: "#2196F3", 1: "#F44336"}
OUTPUT_DIR = "outputs/eda"


class EDA:
    """
    Performs and visualises exploratory data analysis.

    Usage
    -----
    >>> eda = EDA(df)
    >>> report = eda.run()          # prints summary + saves figure
    >>> fig   = eda.figure          # matplotlib Figure
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
        self.df = df.copy()
        self.output_dir = output_dir
        self.figure: plt.Figure | None = None
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self) -> dict:
        """Run full EDA: print summary stats and create the figure."""
        report = self._compute_report()
        self._print_report(report)
        self.figure = self._build_figure()
        path = os.path.join(self.output_dir, "eda_report.png")
        self.figure.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[EDA] Figure saved → {path}")
        return report

    # ── Report helpers ────────────────────────────────────────────────────────
    def _compute_report(self) -> dict:
        df = self.df
        return {
            "shape": df.shape,
            "missing": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "class_distribution": df["target"].value_counts().to_dict(),
            "numeric_summary": df.describe().round(2),
        }

    def _print_report(self, report: dict):
        sep = "=" * 60
        print(f"\n{sep}")
        print("  EDA SUMMARY REPORT")
        print(sep)
        print(f"  Shape            : {report['shape']}")
        print(f"  Duplicate rows   : {report['duplicates']}")
        print(f"  Missing values   : {sum(report['missing'].values())}")
        print(f"  Class balance    : {report['class_distribution']}")
        print(f"\n{'─'*60}")
        print("  Numeric Summary")
        print("─"*60)
        print(report["numeric_summary"].to_string())
        print(sep + "\n")

    # ── Figure ────────────────────────────────────────────────────────────────
    def _build_figure(self) -> plt.Figure:
        df = self.df
        fig = plt.figure(figsize=(20, 22), facecolor="#F8F9FA")
        fig.suptitle(
            "Heart Disease Dataset — Exploratory Data Analysis",
            fontsize=20, fontweight="bold", y=0.98, color="#1A237E"
        )

        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

        # 1. Class distribution (pie)
        ax0 = fig.add_subplot(gs[0, 0])
        counts = df["target"].value_counts()
        colors = ["#2196F3", "#F44336"]
        ax0.pie(counts, labels=["No Disease", "Disease"],
                autopct="%1.1f%%", colors=colors,
                startangle=90, textprops={"fontsize": 11})
        ax0.set_title("Target Class Distribution", fontweight="bold")

        # 2. Age distribution by target
        ax1 = fig.add_subplot(gs[0, 1])
        for tgt, grp in df.groupby("target"):
            ax1.hist(grp["age"], bins=18, alpha=0.65, label=f"Target={tgt}",
                     color=PALETTE[tgt], edgecolor="white")
        ax1.set_xlabel("Age"); ax1.set_ylabel("Count")
        ax1.set_title("Age Distribution by Target", fontweight="bold")
        ax1.legend()

        # 3. Max heart rate vs age scatter
        ax2 = fig.add_subplot(gs[0, 2])
        for tgt, grp in df.groupby("target"):
            ax2.scatter(grp["age"], grp["thalach"], alpha=0.5,
                        color=PALETTE[tgt], label=f"Target={tgt}", s=25)
        ax2.set_xlabel("Age"); ax2.set_ylabel("Max Heart Rate (thalach)")
        ax2.set_title("Age vs Max Heart Rate", fontweight="bold")
        ax2.legend()

        # 4. Correlation heatmap
        ax3 = fig.add_subplot(gs[1, :])
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax3,
                    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
        ax3.set_title("Feature Correlation Heatmap", fontweight="bold")

        # 5. Chest pain type vs target
        ax4 = fig.add_subplot(gs[2, 0])
        cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
        cp_counts.plot(kind="bar", ax=ax4, color=["#2196F3", "#F44336"],
                       edgecolor="white", width=0.7)
        ax4.set_xlabel("Chest Pain Type"); ax4.set_ylabel("Count")
        ax4.set_title("Chest Pain Type vs Target", fontweight="bold")
        ax4.legend(["No Disease", "Disease"])
        ax4.tick_params(axis="x", rotation=0)

        # 6. Cholesterol distribution
        ax5 = fig.add_subplot(gs[2, 1])
        for tgt, grp in df.groupby("target"):
            sns.kdeplot(grp["chol"], ax=ax5, fill=True, alpha=0.4,
                        color=PALETTE[tgt], label=f"Target={tgt}")
        ax5.set_xlabel("Cholesterol"); ax5.set_ylabel("Density")
        ax5.set_title("Cholesterol Distribution by Target", fontweight="bold")
        ax5.legend()

        # 7. Resting blood pressure boxplot
        ax6 = fig.add_subplot(gs[2, 2])
        df_bp = df[["trestbps", "target"]].copy()
        df_bp["target"] = df_bp["target"].map({0: "No Disease", 1: "Disease"})
        sns.boxplot(data=df_bp, x="target", y="trestbps", ax=ax6,
                    palette={"No Disease": "#2196F3", "Disease": "#F44336"})
        ax6.set_xlabel(""); ax6.set_ylabel("Resting BP (mm Hg)")
        ax6.set_title("Resting Blood Pressure by Target", fontweight="bold")

        # 8. Feature importance (correlation with target)
        ax7 = fig.add_subplot(gs[3, :])
        target_corr = (
            df.corr(numeric_only=True)["target"]
            .drop("target")
            .sort_values(key=abs, ascending=False)
        )
        colors_bar = ["#F44336" if v > 0 else "#2196F3" for v in target_corr]
        ax7.bar(target_corr.index, target_corr.values,
                color=colors_bar, edgecolor="white")
        ax7.axhline(0, color="black", linewidth=0.8)
        ax7.set_xlabel("Feature"); ax7.set_ylabel("Pearson Correlation")
        ax7.set_title("Feature Correlation with Target (red = positive, blue = negative)",
                      fontweight="bold")
        ax7.tick_params(axis="x", rotation=30)

        return fig
