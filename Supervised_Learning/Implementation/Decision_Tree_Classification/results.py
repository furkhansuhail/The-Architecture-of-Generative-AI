"""
Module 5: Results & Visualiser
================================
Computes all evaluation metrics and renders publication-quality plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score, roc_auc_score,
)
import warnings; warnings.filterwarnings("ignore")

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
BG_COLOR = "#F8F9FA"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR})


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Results] Saved → {path}")
    return path


# ── 1. Metrics summary ────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, label="Tuned Tree") -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = dict(
        accuracy = accuracy_score(y_test, y_pred),
        f1       = f1_score(y_test, y_pred),
        roc_auc  = roc_auc_score(y_test, y_proba),
    )

    print(f"\n{'═'*60}")
    print(f"  EVALUATION RESULTS — {label}")
    print(f"{'═'*60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Disease','Heart Disease'])}")
    print(f"  Tree Depth  : {model.get_depth()}")
    print(f"  Tree Leaves : {model.get_n_leaves()}")
    return metrics


# ── 2. Confusion matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test) -> str:
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Heart Disease"],
                yticklabels=["No Disease", "Heart Disease"],
                linewidths=1, ax=ax, annot_kws={"size": 14})
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Actual Label",    fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    fig.tight_layout()
    return _save(fig, "06_confusion_matrix.png")


# ── 3. ROC + PR curves ────────────────────────────────────────────────────────
def plot_roc_pr_curves(model, X_test, y_test) -> str:
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_val   = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ROC and Precision-Recall Curves",
                 fontsize=15, fontweight="bold")

    # ROC
    axes[0].plot(fpr, tpr, color="#e74c3c", lw=2,
                 label=f"AUC = {roc_auc_val:.3f}")
    axes[0].plot([0,1],[0,1],"--", color="grey", lw=1, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.12, color="#e74c3c")
    axes[0].set(title="ROC Curve", xlabel="False Positive Rate",
                ylabel="True Positive Rate")
    axes[0].legend(loc="lower right")

    # PR
    axes[1].plot(rec, prec, color="#3498db", lw=2,
                 label=f"AUC = {pr_auc_val:.3f}")
    axes[1].fill_between(rec, prec, alpha=0.12, color="#3498db")
    axes[1].set(title="Precision-Recall Curve",
                xlabel="Recall", ylabel="Precision")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, "07_roc_pr_curves.png")


# ── 4. Feature importance ────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names) -> str:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp)))
    bars    = ax.barh(imp.index, imp.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, imp.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_title("Decision Tree Feature Importance (Gini)",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, imp.max() * 1.18)
    fig.tight_layout()
    return _save(fig, "08_feature_importance.png")


# ── 5. Decision tree visualisation ──────────────────────────────────────────
def plot_decision_tree(model, feature_names, max_depth=3) -> str:
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No Disease", "Heart Disease"],
        filled=True,
        rounded=True,
        max_depth=max_depth,
        ax=ax,
        fontsize=9,
        impurity=True,
        proportion=False,
    )
    ax.set_title(f"Decision Tree (first {max_depth} levels shown)",
                 fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    return _save(fig, "09_decision_tree_viz.png")


# ── 6. CV score distribution ─────────────────────────────────────────────────
def plot_cv_scores(cv_scores: np.ndarray, label="Tuned Model") -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, len(cv_scores) + 1))
    ax.plot(x, cv_scores, "o-", color="#3498db", lw=2, ms=8)
    ax.axhline(cv_scores.mean(), color="#e74c3c", linestyle="--",
               label=f"Mean = {cv_scores.mean():.4f}")
    ax.fill_between(x,
                    cv_scores.mean() - cv_scores.std(),
                    cv_scores.mean() + cv_scores.std(),
                    alpha=0.15, color="#3498db", label="±1 std")
    ax.set(title=f"Cross-Validation Accuracy — {label}",
           xlabel="Fold", ylabel="Accuracy")
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    return _save(fig, "10_cv_scores.png")


# ── 7. Model comparison bar ─────────────────────────────────────────────────
def plot_model_comparison(models_metrics: dict) -> str:
    """
    models_metrics = {'Baseline Tree': {...}, 'Tuned Tree': {...}, ...}
    """
    names   = list(models_metrics.keys())
    metrics = ["accuracy", "f1", "roc_auc"]
    labels  = ["Accuracy", "F1 Score", "ROC AUC"]
    colors  = ["#3498db", "#e74c3c", "#2ecc71"]

    x     = np.arange(len(names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [models_metrics[n].get(m, 0) for n in names]
        bars = ax.bar(x + i * width, vals, width,
                      label=lbl, color=col, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "11_model_comparison.png")


# ── 8. Depth vs accuracy curve ───────────────────────────────────────────────
def plot_depth_vs_accuracy(X_train, y_train, X_test, y_test) -> str:
    depths = list(range(1, 16))
    train_scores, test_scores = [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, train_scores, "o-", color="#3498db", label="Train Accuracy", lw=2)
    ax.plot(depths, test_scores,  "s-", color="#e74c3c", label="Test Accuracy",  lw=2)
    best_d = depths[np.argmax(test_scores)]
    ax.axvline(best_d, color="#2ecc71", linestyle="--",
               label=f"Best depth = {best_d}")
    ax.set(title="Decision Tree Depth vs Accuracy",
           xlabel="Max Depth", ylabel="Accuracy")
    ax.set_xticks(depths)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    return _save(fig, "12_depth_vs_accuracy.png")


# ── Master runner ─────────────────────────────────────────────────────────────
def run_results(
    best_model, base_model, pruned_model,
    X_train, X_test, y_train, y_test,
    feature_names, cv_scores,
) -> list:
    print("\n[Results] Generating all evaluation plots …")
    saved = []

    # Metrics
    m_best   = evaluate(best_model,   X_test, y_test, "Tuned Tree")
    m_base   = evaluate(base_model,   X_test, y_test, "Baseline Tree")
    m_pruned = evaluate(pruned_model, X_test, y_test, "Pruned Tree")

    # Plots
    saved += [
        plot_confusion_matrix(best_model, X_test, y_test),
        plot_roc_pr_curves(best_model, X_test, y_test),
        plot_feature_importance(best_model, feature_names),
        plot_decision_tree(best_model, feature_names, max_depth=3),
        plot_cv_scores(cv_scores),
        plot_model_comparison({
            "Baseline": m_base,
            "Tuned": m_best,
            "Pruned": m_pruned,
        }),
        plot_depth_vs_accuracy(X_train, y_train, X_test, y_test),
    ]

    print(f"\n[Results] {len(saved)} plots saved to {OUT_DIR}/")
    return saved
