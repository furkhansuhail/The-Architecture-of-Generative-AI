"""
========================================================
  Module 4: Results & Evaluation Display
  - Classification Report
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Decision Boundary (PCA 2D projection)
  - Feature Importance via permutation
  - GridSearch heatmap
========================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Malignant", "Benign"]


# ──────────────────────────────────────────────────────
# 4.1  Classification Report (text)
# ──────────────────────────────────────────────────────
def print_classification_report(y_true, y_pred, y_prob=None) -> dict:
    print("\n" + "="*60)
    print("  CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES,
                                   output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        roc = roc_auc_score(y_true, y_prob[:, 1])
        print(f"  ROC-AUC Score : {roc:.4f}")
        report["roc_auc"] = roc

    print("="*60 + "\n")
    return report


# ──────────────────────────────────────────────────────
# 4.2  Confusion Matrix
# ──────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=1, ax=axes[0])
    axes[0].set_title("Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalised
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=1, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Normalised (row %)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "results_confusion_matrix.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")


# ──────────────────────────────────────────────────────
# 4.3  ROC Curve
# ──────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3498DB", lw=2.5,
            label=f"SVM (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#3498DB")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "results_roc_curve.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")


# ──────────────────────────────────────────────────────
# 4.4  Precision-Recall Curve
# ──────────────────────────────────────────────────────
def plot_precision_recall_curve(y_true, y_prob) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#E67E22", lw=2.5,
            label=f"SVM (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.12, color="#E67E22")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "results_precision_recall.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")


# ──────────────────────────────────────────────────────
# 4.5  Decision Boundary (PCA 2D)
# ──────────────────────────────────────────────────────
def plot_decision_boundary_2d(model, X_test, y_test) -> None:
    """Project test set to 2D via PCA and visualise SVM boundary."""
    pca   = PCA(n_components=2, random_state=42)
    X_2d  = pca.fit_transform(X_test)

    # Fit a new SVM on 2D projection for boundary drawing
    from sklearn.svm import SVC
    svm_2d = SVC(kernel="rbf", C=10, gamma="scale", probability=False)
    svm_2d.fit(X_2d, y_test)

    h  = 0.05
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy       = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap_bg = plt.cm.RdYlGn
    palette = {0: "#E74C3C", 1: "#2ECC71"}

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_bg)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.8, linestyles="--")

    for cls, label in {0: "Malignant", 1: "Benign"}.items():
        mask = y_test == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=palette[cls], label=label, edgecolors="k",
                   linewidths=0.4, s=55, alpha=0.85)

    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title("SVM Decision Boundary (PCA 2-D Projection)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "results_decision_boundary.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")


# ──────────────────────────────────────────────────────
# 4.6  Permutation Feature Importance
# ──────────────────────────────────────────────────────
def plot_feature_importance(model, X_test, y_test, feature_names, top_n=15) -> None:
    result = permutation_importance(model, X_test, y_test,
                                    n_repeats=15, random_state=42,
                                    scoring="f1")
    idx     = np.argsort(result.importances_mean)[::-1][:top_n]
    means   = result.importances_mean[idx]
    stds    = result.importances_std[idx]
    labels  = [feature_names[i] for i in idx]

    colors  = ["#3498DB" if m > 0 else "#E74C3C" for m in means]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), means[::-1], xerr=stds[::-1],
                   align="center", color=colors[::-1],
                   edgecolor="white", height=0.65, capsize=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Mean decrease in F1 score", fontsize=11)
    ax.set_title(f"Permutation Feature Importance (top {top_n})",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "results_feature_importance.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")


# ──────────────────────────────────────────────────────
# 4.7  GridSearch CV Results Heatmap
# ──────────────────────────────────────────────────────
def plot_gridsearch_heatmap(gs) -> None:
    """Visualise GridSearch mean test scores for RBF kernel."""
    import pandas as pd
    results = pd.DataFrame(gs.cv_results_)

    rbf = results[results["param_svc__kernel"] == "rbf"].copy()
    if rbf.empty:
        return

    rbf["C"]     = rbf["param_svc__C"].astype(float)
    rbf["gamma"] = rbf["param_svc__gamma"].astype(str)
    pivot = rbf.pivot_table(index="C", columns="gamma",
                            values="mean_test_score")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean CV F1"})
    ax.set_title("GridSearchCV — RBF Kernel (C vs γ)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("C")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "results_gridsearch_heatmap.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")


# ──────────────────────────────────────────────────────
# 4.8  Baseline vs Tuned Comparison Bar Chart
# ──────────────────────────────────────────────────────
def plot_model_comparison(baseline_metrics: dict, tuned_metrics: dict) -> None:
    metrics = ["accuracy", "f1", "roc_auc"]
    labels  = ["Accuracy", "F1", "ROC-AUC"]
    x       = np.arange(len(metrics))
    width   = 0.35

    base_means  = [baseline_metrics[m]["mean"] for m in metrics]
    base_stds   = [baseline_metrics[m]["std"]  for m in metrics]
    tuned_means = [tuned_metrics[m]["mean"]    for m in metrics]
    tuned_stds  = [tuned_metrics[m]["std"]     for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, base_means, width, yerr=base_stds,
           label="Baseline SVM", color="#95A5A6", capsize=5, edgecolor="white")
    ax.bar(x + width/2, tuned_means, width, yerr=tuned_stds,
           label="Tuned SVM",    color="#3498DB", capsize=5, edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim([0.88, 1.0])
    ax.set_ylabel("CV Score", fontsize=12)
    ax.set_title("Baseline vs Tuned SVM (Cross-Validation)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.35)

    for i, (bm, tm) in enumerate(zip(base_means, tuned_means)):
        ax.text(i - width/2, bm + 0.001, f"{bm:.3f}", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(i + width/2, tm + 0.001, f"{tm:.3f}", ha="center",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "results_model_comparison.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")


# ──────────────────────────────────────────────────────
# Master display runner
# ──────────────────────────────────────────────────────
def display_all_results(baseline_model, tuned_model, gs,
                         X_train, X_test, y_train, y_test,
                         feature_names,
                         baseline_cv, tuned_cv) -> None:
    from trainer import cross_validate_model

    print("\n[Results] Generating all evaluation plots …\n")

    # Predictions
    y_pred_base  = baseline_model.predict(X_test)
    y_prob_base  = baseline_model.predict_proba(X_test)
    y_pred_tuned = tuned_model.predict(X_test)
    y_prob_tuned = tuned_model.predict_proba(X_test)

    print("── Baseline SVM ──")
    print_classification_report(y_test, y_pred_base, y_prob_base)

    print("── Tuned SVM ──")
    print_classification_report(y_test, y_pred_tuned, y_prob_tuned)

    # Plots
    plot_confusion_matrix(y_test, y_pred_tuned)
    plot_roc_curve(y_test, y_prob_tuned)
    plot_precision_recall_curve(y_test, y_prob_tuned)
    plot_decision_boundary_2d(tuned_model, X_test, y_test)
    plot_feature_importance(tuned_model, X_test, y_test, feature_names)
    plot_gridsearch_heatmap(gs)
    plot_model_comparison(baseline_cv, tuned_cv)

    print("\n[Results] All evaluation outputs saved to outputs/\n")
