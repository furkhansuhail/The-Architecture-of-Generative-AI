"""
visualizer.py - Rich visualisation of model results.

Plots generated
───────────────
  05_anomaly_score_distribution.png  – score histograms by true class
  06_roc_curve.png                   – ROC + PR curves side-by-side
  07_confusion_matrix.png            – annotated confusion matrix heat-map
  08_top_anomalies_scatter.png       – Amount vs anomaly score scatter
  09_feature_importance_proxy.png    – mean |score| change per feature (proxy)
  10_tsne_projection.png             – 2-D t-SNE coloured by label & score
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from config import OUTPUT_DIR, FIGURE_DPI, PALETTE

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=0.9)

_NORMAL  = PALETTE["normal"]
_ANOMALY = PALETTE["anomaly"]


def display_results(model: IsolationForest,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    test_scores: np.ndarray,
                    report: dict,
                    feature_names: list[str]) -> None:
    """Master function – render all result visualisations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("──── Visualising Results ────────────────────────")

    y_true = y_test.values

    _plot_score_distribution(test_scores, y_true)
    _plot_roc_pr_curves(y_true, test_scores)
    _plot_confusion_matrix(report)
    _plot_top_anomalies_scatter(X_test, y_true, test_scores)
    _plot_feature_importance_proxy(model, X_test, test_scores, feature_names)
    _plot_tsne(X_test, y_true, test_scores)

    logger.info("All result plots saved to %s", OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────────────

def _plot_score_distribution(scores: np.ndarray, y_true: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls, (label, color) in {0: ("Normal", _NORMAL),
                                  1: ("Anomaly", _ANOMALY)}.items():
        ax.hist(scores[y_true == cls], bins=80, color=color,
                alpha=0.7, density=True, label=label, edgecolor="none")

    # Decision threshold (where IsolationForest draws the line)
    threshold = -model_threshold_from_scores(scores)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Decision threshold ≈ {threshold:.3f}")

    ax.set_xlabel("Anomaly Score  (higher → more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution — Normal vs Anomaly", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "05_anomaly_score_distribution.png")


def _plot_roc_pr_curves(y_true, scores) -> None:
    fpr, tpr, _  = roc_curve(y_true, scores)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    pr_auc       = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── ROC ──────────────────────────────────────────────────────────────────
    axes[0].plot(fpr, tpr, color=_ANOMALY, lw=2,
                 label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color=_ANOMALY)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right")

    # ── PR ───────────────────────────────────────────────────────────────────
    baseline = y_true.mean()
    axes[1].plot(rec, prec, color=_NORMAL, lw=2,
                 label=f"PR-AUC = {pr_auc:.4f}")
    axes[1].axhline(baseline, color="k", linestyle="--", lw=1, alpha=0.5,
                    label=f"Random baseline ({baseline:.3f})")
    axes[1].fill_between(rec, prec, alpha=0.1, color=_NORMAL)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="upper right")

    plt.suptitle("Model Performance Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "06_roc_pr_curves.png")


def _plot_confusion_matrix(report: dict) -> None:
    cm   = np.array([[report["TN"], report["FP"]],
                     [report["FN"], report["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Normal", "Pred Anomaly"],
                yticklabels=["True Normal", "True Anomaly"],
                linewidths=0.5, linecolor="gray", ax=ax,
                cbar_kws={"shrink": 0.7})
    ax.set_title(
        f"Confusion Matrix\n"
        f"Accuracy={report['Accuracy']:.3f}  "
        f"F1={report['F1']:.3f}  "
        f"MCC={report['MCC']:.3f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "07_confusion_matrix.png")


def _plot_top_anomalies_scatter(X_test, y_true, scores) -> None:
    if "Amount" not in X_test.columns:
        return

    df = X_test[["Amount"]].copy()
    df["score"]  = scores
    df["true"]   = y_true
    df["label"]  = df["true"].map({0: "Normal", 1: "Anomaly"})

    fig, ax = plt.subplots(figsize=(11, 5))
    for lbl, color in {"Normal": _NORMAL, "Anomaly": _ANOMALY}.items():
        sub = df[df["label"] == lbl]
        ax.scatter(sub["Amount"], sub["score"],
                   c=color, alpha=0.35, s=10,
                   label=f"{lbl} (n={len(sub):,})")

    ax.set_xlabel("Transaction Amount (scaled)")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Transaction Amount vs Anomaly Score", fontweight="bold")
    ax.legend(markerscale=3)
    plt.tight_layout()
    _save(fig, "08_top_anomalies_scatter.png")


def _plot_feature_importance_proxy(model, X_test, scores, feature_names,
                                   n_top=20) -> None:
    """
    Proxy importance: for each feature, shuffle its values and measure
    the mean change in anomaly score  (permutation-style, quick version).
    """
    logger.info("  Computing feature importance proxy …")
    rng      = np.random.default_rng(42)
    baseline = scores.mean()
    X_arr    = X_test.values.copy()
    importances = []

    for i in range(X_arr.shape[1]):
        X_perm      = X_arr.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_scores = -model.decision_function(X_perm)
        importances.append(abs(perm_scores.mean() - baseline))

    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top        = imp_series.head(n_top)

    fig, ax = plt.subplots(figsize=(9, max(5, n_top * 0.35)))
    colors  = [_ANOMALY if v > top.median() else _NORMAL for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Mean |Δ Anomaly Score| after permutation")
    ax.set_title(f"Feature Importance (Permutation Proxy) — Top {n_top}",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "09_feature_importance_proxy.png")


def _plot_tsne(X_test, y_true, scores, sample_n=3000) -> None:
    """2-D t-SNE coloured by (a) true label and (b) anomaly score."""
    n = min(sample_n, len(X_test))
    idx  = np.random.default_rng(42).choice(len(X_test), n, replace=False)
    X_s  = X_test.values[idx]
    y_s  = y_true[idx]
    sc_s = scores[idx]

    logger.info("  Running t-SNE on %d samples …", n)
    embedding = TSNE(n_components=2, random_state=42,
                     perplexity=30, max_iter=500).fit_transform(X_s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: coloured by true label
    for cls, (label, color) in {0: ("Normal", _NORMAL),
                                  1: ("Anomaly", _ANOMALY)}.items():
        mask = y_s == cls
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=color, s=8, alpha=0.5, label=label)
    axes[0].set_title("t-SNE — True Labels", fontweight="bold")
    axes[0].legend(markerscale=3)
    axes[0].axis("off")

    # Right: coloured by anomaly score (continuous)
    sc = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                         c=sc_s, cmap="plasma", s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[1], label="Anomaly Score")
    axes[1].set_title("t-SNE — Anomaly Score", fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("2-D t-SNE Projection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_tsne_projection.png")


# ──────────────────────────────────────────────────────────────────────────────

def model_threshold_from_scores(scores: np.ndarray) -> float:
    """Approximate the decision threshold from score percentiles."""
    return float(np.percentile(scores, 98))


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", name)
