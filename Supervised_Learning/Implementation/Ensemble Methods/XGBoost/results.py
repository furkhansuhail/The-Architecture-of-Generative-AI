"""
results.py — Comprehensive visualisation of model performance.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config import PLOTS_DIR, THRESHOLD, TARGET_COLUMN

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test) -> str:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)
    cm    = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Diabetic", "Diabetic"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/05_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(model, X_test, y_test) -> str:
    proba      = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc    = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2.5,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/06_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_precision_recall(model, X_test, y_test) -> str:
    proba = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#DD8452", lw=2.5,
            label=f"PR curve (AP = {ap:.3f})")
    ax.fill_between(rec, prec, alpha=0.08, color="#DD8452")
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", lw=1.2, ls="--", alpha=0.5,
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = f"{PLOTS_DIR}/07_precision_recall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names) -> str:
    importance = model.feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values("Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names) * 0.45)))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                   color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlabel("XGBoost Importance Score (gain)")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/08_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_probability_distribution(model, X_test, y_test) -> str:
    proba = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, color, label in [(0, "#4C72B0", "Non-Diabetic"), (1, "#DD8452", "Diabetic")]:
        subset = proba[y_test == cls]
        ax.hist(subset, bins=30, alpha=0.65, color=color,
                label=f"{label} (n={len(subset)})", edgecolor="none")

    ax.axvline(THRESHOLD, color="red", lw=2, ls="--",
               label=f"Threshold = {THRESHOLD}")
    ax.set_xlabel("Predicted Probability of Diabetes")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = f"{PLOTS_DIR}/09_probability_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_shap_summary(model, X_test, feature_names) -> str | None:
    if not SHAP_AVAILABLE:
        print("[Results] SHAP not available — skipping SHAP plot.")
        return None

    # Fix for XGBoost >= 2.0 + SHAP version incompatibility.
    # XGBoost 2.x serialises base_score as "[5E-1]" (a string), which older
    # SHAP builds cannot parse. We reset it to a plain float on the booster.
    # We also use interventional perturbation which does NOT require the
    # background data to cover every leaf.
    try:
        booster = model.get_booster()
        booster.set_param("base_score", 0.5)
        explainer   = shap.TreeExplainer(
            booster,
            feature_perturbation="interventional",
            data=X_test,
        )
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        print(f"[Results] SHAP TreeExplainer failed: {e}\n          Trying model-agnostic Explainer …")
        try:
            explainer   = shap.Explainer(model, X_test)
            shap_values = explainer(X_test).values
        except Exception as e2:
            print(f"[Results] SHAP fallback failed: {e2} — skipping SHAP plot.")
            return None

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False, plot_size=None,
    )
    plt.title("SHAP Feature Impact (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/10_shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_metrics_dashboard(metrics: dict, cv_summary: dict) -> str:
    """Single-page dashboard: bar charts for test metrics + CV comparison."""
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # — Test-set metrics bar chart —
    ax1 = fig.add_subplot(gs[0])
    display_metrics = {k: v for k, v in metrics.items()
                       if k not in ("log_loss",)}   # log_loss has different scale
    names  = list(display_metrics.keys())
    values = list(display_metrics.values())
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(names)))
    bars   = ax1.bar(names, values, color=colors, edgecolor="white", linewidth=1)
    ax1.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Score")
    ax1.set_title("Test Set Metrics", fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)

    # — CV mean ± std dot plot —
    ax2 = fig.add_subplot(gs[1])
    cv_names  = list(cv_summary.keys())
    cv_means  = [cv_summary[k]["mean"] for k in cv_names]
    cv_stds   = [cv_summary[k]["std"]  for k in cv_names]
    y_pos     = np.arange(len(cv_names))
    ax2.barh(y_pos, cv_means, xerr=cv_stds, color="#4C72B0", alpha=0.75,
             edgecolor="none", capsize=5, error_kw=dict(elinewidth=1.5, ecolor="navy"))
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cv_names)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Score")
    ax2.set_title(f"Cross-Validation (mean ± std)", fontsize=13, fontweight="bold")
    for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
        ax2.text(m + s + 0.01, i, f"{m:.3f}", va="center", fontsize=9, fontweight="bold")

    fig.suptitle("XGBoost Model Performance Dashboard", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/11_metrics_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
def run_results(model, X_test, y_test, feature_names, metrics, cv_summary) -> None:
    """Generate all result plots."""
    print("\n[Results] Generating visualisations …")
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall(model, X_test, y_test)
    plot_feature_importance(model, feature_names)
    plot_probability_distribution(model, X_test, y_test)
    plot_shap_summary(model, X_test, feature_names)
    plot_metrics_dashboard(metrics, cv_summary)
    print("[Results] All visualisations complete.\n")