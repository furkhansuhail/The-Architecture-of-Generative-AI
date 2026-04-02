"""
=============================================================
  results/visualizer.py  –  All post-training plots & reports
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.tree import plot_tree
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

PALETTE  = "Set2"
OUT_DIR  = config.OUTPUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
class Visualizer:
    """
    Produces all evaluation artefacts:

    • confusion_matrix_plot()  – colour-coded CM
    • roc_curve_plot()         – AUC-ROC for RF vs. Bagging
    • feature_importance_plot()– horizontal bar chart
    • oob_error_curve()        – OOB error vs n_estimators
    • single_tree_plot()       – one decision tree from the forest
    • metrics_summary()        – console + text file report
    • run()                    – all of the above
    """

    def __init__(self, rf_model_obj, X_test, y_test):
        self.obj    = rf_model_obj          # RandomForestModel instance
        self.rf     = rf_model_obj.rf_model
        self.bag    = rf_model_obj.bagging_model
        self.X_test = X_test
        self.y_test = y_test

        self.y_pred      = self.rf.predict(X_test)
        self.y_prob      = self.rf.predict_proba(X_test)[:, 1]
        self.bag_y_pred  = self.bag.predict(X_test)
        self.bag_y_prob  = self.bag.predict_proba(X_test)[:, 1]

    def _save(self, fig, name):
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] saved → {path}")

    # ── 1. Confusion Matrix ───────────────────────────────────────────────────
    def confusion_matrix_plot(self) -> "Visualizer":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, preds, title in zip(
            axes,
            [self.y_pred, self.bag_y_pred],
            ["Random Forest", "Bagging (explicit)"],
        ):
            cm = confusion_matrix(self.y_test, preds)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Not survived", "Survived"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(title, fontsize=13)
        fig.suptitle("Confusion Matrices", fontsize=14)
        fig.tight_layout()
        self._save(fig, "05_confusion_matrices.png")
        return self

    # ── 2. ROC Curve ──────────────────────────────────────────────────────────
    def roc_curve_plot(self) -> "Visualizer":
        fig, ax = plt.subplots(figsize=(7, 6))
        for probs, label, color in [
            (self.y_prob,     "Random Forest",     "#2196F3"),
            (self.bag_y_prob, "Bagging (explicit)", "#FF9800"),
        ]:
            fpr, tpr, _ = roc_curve(self.y_test, probs)
            auc = roc_auc_score(self.y_test, probs)
            ax.plot(fpr, tpr, label=f"{label}  (AUC = {auc:.3f})", lw=2, color=color)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curve – RF vs Bagging")
        ax.legend(loc="lower right")
        sns.despine()
        self._save(fig, "06_roc_curve.png")
        return self

    # ── 3. Feature Importance ─────────────────────────────────────────────────
    def feature_importance_plot(self) -> "Visualizer":
        fi = self.obj.feature_importance(self.X_test.columns)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = sns.color_palette(PALETTE, len(fi))
        ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
        ax.set(xlabel="Mean Decrease in Impurity", title="Feature Importances (Random Forest)")
        sns.despine()
        self._save(fig, "07_feature_importance.png")
        return self

    # ── 4. OOB Error Curve ────────────────────────────────────────────────────
    def oob_error_curve(self, X_train, y_train, max_trees: int = 150) -> "Visualizer":
        """
        Show how OOB error drops as more trees are added.
        Trains a fresh estimator with warm_start to accumulate trees.
        """
        print("[Visualizer] Computing OOB error curve …")
        from sklearn.ensemble import RandomForestClassifier as RFC
        errors = []
        n_range = range(10, max_trees + 1, 10)
        for n in n_range:
            m = RFC(
                n_estimators = n,
                oob_score    = True,
                bootstrap    = True,
                random_state = config.RANDOM_STATE,
                n_jobs       = -1,
            )
            m.fit(X_train, y_train)
            errors.append(1 - m.oob_score_)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(n_range), errors, marker="o", ms=4, color="#2196F3", lw=2)
        ax.set(
            xlabel = "Number of Trees",
            ylabel = "OOB Error",
            title  = "OOB Error vs Number of Trees",
        )
        ax.grid(True, alpha=0.3)
        sns.despine()
        self._save(fig, "08_oob_error_curve.png")
        return self

    # ── 5. Single Tree Visualisation ──────────────────────────────────────────
    def single_tree_plot(self, max_depth: int = 3) -> "Visualizer":
        """Plot the first decision tree in the forest (depth-limited for readability)."""
        tree = self.rf.estimators_[0]
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            tree,
            feature_names = list(self.X_test.columns),
            class_names   = ["Not survived", "Survived"],
            filled        = True,
            max_depth     = max_depth,
            ax            = ax,
            fontsize      = 8,
        )
        ax.set_title(
            f"One Decision Tree from the Forest (max_depth={max_depth} shown)",
            fontsize=13,
        )
        self._save(fig, "09_single_tree.png")
        return self

    # ── 6. Metrics Summary ────────────────────────────────────────────────────
    def metrics_summary(self, cv_result: dict = None) -> "Visualizer":
        lines = []
        sep   = "=" * 58

        def add(txt=""):
            lines.append(txt)
            print(txt)

        add(sep)
        add("  MODEL EVALUATION REPORT  –  Titanic Survival (RF)")
        add(sep)

        for preds, probs, name in [
            (self.y_pred,     self.y_prob,     "Random Forest"),
            (self.bag_y_pred, self.bag_y_prob, "Bagging (explicit)"),
        ]:
            add(f"\n  ── {name} ──")
            add(f"  Accuracy  : {accuracy_score(self.y_test, preds):.4f}")
            add(f"  Precision : {precision_score(self.y_test, preds):.4f}")
            add(f"  Recall    : {recall_score(self.y_test, preds):.4f}")
            add(f"  F1-Score  : {f1_score(self.y_test, preds):.4f}")
            add(f"  ROC-AUC   : {roc_auc_score(self.y_test, probs):.4f}")
            add(f"\n  Classification Report:\n")
            rep = classification_report(
                self.y_test, preds,
                target_names=["Not survived", "Survived"],
            )
            for line in rep.splitlines():
                add("    " + line)

        if cv_result:
            add(f"\n  ── {config.CV_FOLDS}-Fold Cross Validation (RF) ──")
            add(f"  Fold scores: {np.round(cv_result['scores'], 4)}")
            add(f"  Mean ± Std : {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

        if self.obj.best_params:
            add(f"\n  ── Best Hyper-parameters (Grid Search) ──")
            for k, v in self.obj.best_params.items():
                add(f"  {k:20s}: {v}")

        add("\n" + sep)

        # save to file
        report_path = os.path.join(OUT_DIR, config.REPORT_FILENAME)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n  [Visualizer] Report saved → {report_path}")
        return self

    # ── run all ───────────────────────────────────────────────────────────────
    def run(self, X_train, y_train, cv_result: dict = None) -> "Visualizer":
        print("\n[Visualizer] Generating all result artefacts …")
        (self
            .confusion_matrix_plot()
            .roc_curve_plot()
            .feature_importance_plot()
            .oob_error_curve(X_train, y_train)
            .single_tree_plot()
            .metrics_summary(cv_result)
        )
        print("[Visualizer] Done.\n")
        return self
