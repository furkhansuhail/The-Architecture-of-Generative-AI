"""
evaluator.py
------------
Computes and visualises all evaluation metrics:
  - Accuracy, Precision, Recall, F1
  - Confusion Matrix
  - ROC / AUC curve
  - Learning curve (iterations vs. error)
  - Feature importances
  - Model comparison bar chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)

OUTPUT_DIR = "outputs/results"


class Evaluator:
    """
    Evaluates all trained models and builds a comprehensive results figure.

    Usage
    -----
    >>> ev = Evaluator(models, X_test, y_test, feature_names, cv_scores)
    >>> metrics = ev.evaluate()
    >>> ev.figure   # matplotlib Figure
    """

    def __init__(
        self,
        models: dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        cv_scores: dict,
        best_model_name: str,
        output_dir: str = OUTPUT_DIR,
    ):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.cv_scores = cv_scores
        self.best_model_name = best_model_name
        self.output_dir = output_dir
        self.metrics: dict = {}
        self.figure: plt.Figure | None = None
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def evaluate(self) -> dict:
        """Run evaluation, print report, save figure."""
        self._compute_metrics()
        self._print_report()
        self.figure = self._build_figure()
        path = os.path.join(self.output_dir, "results.png")
        self.figure.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Evaluator] Results figure saved → {path}")
        return self.metrics

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _compute_metrics(self):
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = (
                model.predict_proba(self.X_test)[:, 1]
                if hasattr(model, "predict_proba") else None
            )
            fpr, tpr, roc_auc = None, None, None
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                roc_auc = auc(fpr, tpr)

            self.metrics[name] = {
                "accuracy":  accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred, zero_division=0),
                "recall":    recall_score(self.y_test, y_pred, zero_division=0),
                "f1":        f1_score(self.y_test, y_pred, zero_division=0),
                "cm":        confusion_matrix(self.y_test, y_pred),
                "y_pred":    y_pred,
                "y_prob":    y_prob,
                "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            }

    def _print_report(self):
        sep = "=" * 65
        print(f"\n{sep}")
        print("  EVALUATION RESULTS")
        print(sep)
        header = f"{'Model':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
        print(header)
        print("─" * 65)
        for name, m in self.metrics.items():
            auc_val = f"{m['roc_auc']:.4f}" if m["roc_auc"] else "  N/A "
            marker = " ◀ BEST" if name == self.best_model_name else ""
            print(
                f"{name:<35} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} {m['f1']:>7.4f} {auc_val}{marker}"
            )
        print(sep)

        # Detailed report for best model
        best = self.models[self.best_model_name]
        y_pred_best = self.metrics[self.best_model_name]["y_pred"]
        print(f"\n  Detailed report → {self.best_model_name}")
        print("─" * 65)
        print(classification_report(
            self.y_test, y_pred_best,
            target_names=["No Disease", "Disease"]
        ))
        print(sep + "\n")

    # ── Figure ────────────────────────────────────────────────────────────────
    def _build_figure(self) -> plt.Figure:
        fig = plt.figure(figsize=(20, 22), facecolor="#F8F9FA")
        fig.suptitle(
            "AdaBoost — Evaluation Results",
            fontsize=20, fontweight="bold", y=0.98, color="#1A237E"
        )
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

        # ── Row 0: confusion matrices (best model + baseline) ─────────────────
        for col_idx, name in enumerate(
            ["Baseline (Decision Tree)", self.best_model_name]
        ):
            if name not in self.metrics:
                continue
            ax = fig.add_subplot(gs[0, col_idx])
            cm = self.metrics[name]["cm"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["No Disease", "Disease"],
                        yticklabels=["No Disease", "Disease"], ax=ax,
                        linewidths=1, cbar=False)
            ax.set_title(f"Confusion Matrix\n{name}", fontweight="bold")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        # ── Row 0 col 2: Model comparison bar chart ────────────────────────────
        ax_cmp = fig.add_subplot(gs[0, 2])
        metric_keys = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(metric_keys))
        width = 0.18
        colors = ["#1565C0", "#F44336", "#388E3C", "#F57C00"]
        for i, (name, m) in enumerate(self.metrics.items()):
            vals = [m[k] for k in metric_keys]
            offset = (i - len(self.metrics) / 2) * width
            ax_cmp.bar(x + offset, vals, width, label=name,
                       color=colors[i % len(colors)], alpha=0.85)
        ax_cmp.set_xticks(x); ax_cmp.set_xticklabels(metric_keys)
        ax_cmp.set_ylim(0.5, 1.05); ax_cmp.set_ylabel("Score")
        ax_cmp.set_title("Model Comparison (Test Set)", fontweight="bold")
        ax_cmp.legend(fontsize=7, loc="lower right")

        # ── Row 1: ROC curves ─────────────────────────────────────────────────
        ax_roc = fig.add_subplot(gs[1, :])
        line_styles = ["-", "--", "-.", ":"]
        roc_colors = ["#1565C0", "#F44336", "#388E3C", "#F57C00"]
        for i, (name, m) in enumerate(self.metrics.items()):
            if m["fpr"] is not None:
                ax_roc.plot(
                    m["fpr"], m["tpr"],
                    lw=2.5, linestyle=line_styles[i % 4],
                    color=roc_colors[i % 4],
                    label=f"{name}  (AUC = {m['roc_auc']:.3f})"
                )
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.500)")
        ax_roc.fill_between(
            self.metrics[self.best_model_name]["fpr"],
            self.metrics[self.best_model_name]["tpr"],
            alpha=0.08, color="#F44336"
        )
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves — All Models", fontweight="bold")
        ax_roc.legend(fontsize=9)
        ax_roc.grid(alpha=0.3)

        # ── Row 2: Learning curve (AdaBoost staged score) ─────────────────────
        ax_lc = fig.add_subplot(gs[2, :2])
        ada_name = "AdaBoost (Stump)"
        if ada_name in self.models and hasattr(self.models[ada_name], "staged_predict"):
            ada_model = self.models[ada_name]
            train_errors, test_errors = [], []
            for y_pred_stage in ada_model.staged_predict(
                self._get_X_train_approx()
            ):
                train_errors.append(1 - accuracy_score(self._get_y_train_approx(), y_pred_stage))
            for y_pred_stage in ada_model.staged_predict(self.X_test):
                test_errors.append(1 - accuracy_score(self.y_test, y_pred_stage))
            iters = range(1, len(train_errors) + 1)
            ax_lc.plot(iters, train_errors, label="Train Error", color="#1565C0", lw=2)
            ax_lc.plot(iters, test_errors,  label="Test Error",  color="#F44336", lw=2)
            ax_lc.set_xlabel("Number of Estimators")
            ax_lc.set_ylabel("Classification Error")
            ax_lc.set_title("AdaBoost Learning Curve (Staged Error)", fontweight="bold")
            ax_lc.legend(); ax_lc.grid(alpha=0.3)

        # ── Row 2 col 2: CV Score comparison ─────────────────────────────────
        ax_cv = fig.add_subplot(gs[2, 2])
        cv_names = list(self.cv_scores.keys())
        cv_means = [self.cv_scores[n]["mean"] for n in cv_names]
        cv_stds  = [self.cv_scores[n].get("std", 0) for n in cv_names]
        bar_colors = ["#F44336" if n == self.best_model_name else "#90A4AE"
                      for n in cv_names]
        bars = ax_cv.barh(cv_names, cv_means, xerr=cv_stds,
                          color=bar_colors, edgecolor="white", height=0.5)
        ax_cv.set_xlabel("CV Accuracy (5-fold)")
        ax_cv.set_xlim(0.5, 1.0)
        ax_cv.set_title("Cross-Validation Accuracy", fontweight="bold")
        for bar, val in zip(bars, cv_means):
            ax_cv.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                       f"{val:.4f}", va="center", fontsize=9)

        # ── Row 3: Feature importances ────────────────────────────────────────
        ax_fi = fig.add_subplot(gs[3, :])
        ada_name_stump = "AdaBoost (Stump)"
        if ada_name_stump in self.models:
            ada = self.models[ada_name_stump]
            importances = ada.feature_importances_
            idx = np.argsort(importances)[::-1]
            sorted_features = [self.feature_names[i] for i in idx]
            sorted_imp = importances[idx]
            gradient_colors = plt.cm.RdYlGn(
                np.linspace(0.3, 0.9, len(sorted_features))
            )[::-1]
            ax_fi.bar(sorted_features, sorted_imp,
                      color=gradient_colors, edgecolor="white")
            ax_fi.set_xlabel("Feature")
            ax_fi.set_ylabel("Importance Score")
            ax_fi.set_title(
                f"Feature Importances — {ada_name_stump}",
                fontweight="bold"
            )
            ax_fi.tick_params(axis="x", rotation=30)

        return fig

    # ── Helpers (staged predict needs training data) ──────────────────────────
    _X_train_cache: np.ndarray | None = None
    _y_train_cache: np.ndarray | None = None

    def set_train_data(self, X_train, y_train):
        """Call this so learning curve can use training data."""
        Evaluator._X_train_cache = X_train
        Evaluator._y_train_cache = y_train

    def _get_X_train_approx(self):
        return (Evaluator._X_train_cache
                if Evaluator._X_train_cache is not None else self.X_test)

    def _get_y_train_approx(self):
        return (Evaluator._y_train_cache
                if Evaluator._y_train_cache is not None else self.y_test)
