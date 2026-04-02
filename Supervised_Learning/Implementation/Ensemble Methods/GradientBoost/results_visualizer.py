"""
=============================================================
  Module 5 : Results Visualizer
  ─────────────────────────────────────────────────────────
  Generates & saves:
    07_roc_curves.png          – all models on test set
    08_confusion_matrices.png  – heatmaps for each model
    09_classification_report.png
    10_feature_importance.png  – XGB + LGB native importance
    11_shap_summary.png        – SHAP beeswarm (best model)
    12_shap_dependence.png     – top-3 SHAP dependence plots
    13_optuna_history.png      – optimization history
    14_cv_scores.png           – cross-validation boxplot
    15_pr_curves.png           – Precision-Recall curves
=============================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score, RocCurveDisplay,
)
import shap

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORTS_DIR, FIG_DPI

sns.set_theme(style="whitegrid")
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_COLORS = {
    "sklearn_gb": "#3498DB",
    "xgboost":    "#E74C3C",
    "lightgbm":   "#2ECC71",
    "ensemble":   "#9B59B6",
}


# ── Public API ───────────────────────────────────────────────

def display_results(models: dict, val_scores: dict, cv_scores,
                    best_name: str, X_test, y_test,
                    feature_names: list, studies: dict = None):
    """
    Generate and save all result visualizations.
    """
    print("\n" + "═"*60)
    print("  MODULE 5 — Displaying Results")
    print("═"*60)

    _plot_roc_curves(models, X_test, y_test)
    _plot_confusion_matrices(models, X_test, y_test)
    _plot_classification_report(models[best_name], X_test, y_test, best_name)
    _plot_feature_importance(models, feature_names)
    _plot_shap_summary(models[best_name], X_test, feature_names, best_name)
    _plot_shap_dependence(models[best_name], X_test, feature_names, best_name)
    _plot_pr_curves(models, X_test, y_test)
    _plot_cv_scores(cv_scores, best_name)

    if studies:
        _plot_optuna_history(studies)

    _print_final_report(models, X_test, y_test, val_scores, best_name)
    print(f"\n[Results] All figures saved to → {REPORTS_DIR}/")


# ── Helpers ──────────────────────────────────────────────────

def _plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2,
                color=MODEL_COLORS.get(name, "gray"),
                label=f"{name}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — Test Set", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "07_roc_curves.png"), dpi=FIG_DPI)
    plt.close()
    print("[Results] ✓ 07_roc_curves.png")


def _plot_confusion_matrices(models, X_test, y_test):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="Blues", linewidths=0.5,
                    xticklabels=["≤50K", ">50K"],
                    yticklabels=["≤50K", ">50K"])
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "08_confusion_matrices.png"), dpi=FIG_DPI)
    plt.close()
    print("[Results] ✓ 08_confusion_matrices.png")


def _plot_classification_report(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred,
                                   target_names=["≤50K", ">50K"],
                                   output_dict=True)
    df_report = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
    df_report = df_report.iloc[:4]  # just classes + macro avg

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(df_report.values.astype(float), cmap="YlOrRd",
                   vmin=0.7, vmax=1.0, aspect="auto")
    ax.set_xticks(range(df_report.shape[1]))
    ax.set_xticklabels(df_report.columns, fontsize=12)
    ax.set_yticks(range(df_report.shape[0]))
    ax.set_yticklabels(df_report.index, fontsize=12)

    for i in range(df_report.shape[0]):
        for j in range(df_report.shape[1]):
            ax.text(j, i, f"{df_report.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Classification Report — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "09_classification_report.png"), dpi=FIG_DPI)
    plt.close()
    print("[Results] ✓ 09_classification_report.png")


def _plot_feature_importance(models, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")

    for ax, model_name, model_key in zip(
        axes, ["XGBoost", "LightGBM"], ["xgboost", "lightgbm"]
    ):
        model = models[model_key]
        fi = model.feature_importances_
        idx = np.argsort(fi)[-20:]  # top 20

        names_arr = np.array(feature_names) if len(feature_names) == len(fi) else np.arange(len(fi)).astype(str)
        ax.barh(range(len(idx)), fi[idx],
                color=MODEL_COLORS[model_key], edgecolor="white")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([str(names_arr[i]) for i in idx], fontsize=9)
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "10_feature_importance.png"), dpi=FIG_DPI)
    plt.close()
    print("[Results] ✓ 10_feature_importance.png")


def _plot_shap_summary(model, X_test, feature_names, name):
    print("    Computing SHAP values …")
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test[:2000])

        if isinstance(shap_vals, list):   # binary classification → class 1
            shap_vals = shap_vals[1]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test[:2000],
                          feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title(f"SHAP Summary — {name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "11_shap_summary.png"),
                    dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print("[Results] ✓ 11_shap_summary.png")
        return shap_vals
    except Exception as e:
        print(f"    [SHAP] Warning: {e}")
        return None


def _plot_shap_dependence(model, X_test, feature_names, name):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test[:2000])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        # Top 3 features by mean |SHAP|
        mean_abs = np.abs(shap_vals).mean(0)
        top3_idx = np.argsort(mean_abs)[-3:][::-1]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"SHAP Dependence Plots — {name}", fontsize=13, fontweight="bold")

        for ax, fi in zip(axes, top3_idx):
            fname = feature_names[fi] if fi < len(feature_names) else str(fi)
            ax.scatter(X_test[:2000, fi], shap_vals[:, fi],
                       c=shap_vals[:, fi], cmap="coolwarm",
                       alpha=0.5, s=10)
            ax.set_xlabel(fname, fontsize=10)
            ax.set_ylabel(f"SHAP({fname})", fontsize=10)
            ax.axhline(0, color="gray", lw=0.8, ls="--")

        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "12_shap_dependence.png"),
                    dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print("[Results] ✓ 12_shap_dependence.png")
    except Exception as e:
        print(f"    [SHAP Dependence] Warning: {e}")


def _plot_optuna_history(studies: dict):
    valid = {k: v for k, v in studies.items() if v is not None}
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(8 * len(valid), 5))
    if len(valid) == 1:
        axes = [axes]
    fig.suptitle("Optuna Optimization History", fontsize=14, fontweight="bold")

    for ax, (name, study) in zip(axes, valid.items()):
        vals = [t.value for t in study.trials if t.value is not None]
        best_so_far = np.maximum.accumulate(vals)
        ax.plot(vals, alpha=0.5, color=MODEL_COLORS.get(name, "blue"),
                label="Trial AUC")
        ax.plot(best_so_far, color="black", lw=2, label="Best so far")
        ax.set(title=name, xlabel="Trial", ylabel="Val AUC")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "13_optuna_history.png"), dpi=FIG_DPI)
    plt.close()
    print("[Results] ✓ 13_optuna_history.png")


def _plot_cv_scores(cv_scores, best_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(cv_scores, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#3498DB", alpha=0.7),
               medianprops=dict(color="red", lw=2))
    for i, s in enumerate(cv_scores, 1):
        ax.scatter(1, s, color="#E74C3C", zorder=5, s=60)

    ax.set_xticks([1])
    ax.set_xticklabels([best_name])
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Cross-Validation AUC ({len(cv_scores)}-fold) — {best_name}",
                 fontweight="bold")
    ax.text(1.3, np.mean(cv_scores),
            f"μ={np.mean(cv_scores):.4f}\nσ={np.std(cv_scores):.4f}",
            fontsize=11, va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "14_cv_scores.png"), dpi=FIG_DPI)
    plt.close()
    print("[Results] ✓ 14_cv_scores.png")


def _plot_pr_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, lw=2.2,
                color=MODEL_COLORS.get(name, "gray"),
                label=f"{name}  (AP = {ap:.4f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", ls="--", lw=1,
               label=f"Baseline (AP = {baseline:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves — Test Set",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "15_pr_curves.png"), dpi=FIG_DPI)
    plt.close()
    print("[Results] ✓ 15_pr_curves.png")


def _print_final_report(models, X_test, y_test, val_scores, best_name):
    print("\n" + "═"*60)
    print("  FINAL TEST-SET PERFORMANCE REPORT")
    print("═"*60)

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":    name,
            "AUC":      round(roc_auc_score(y_test, y_proba), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 (>50K)": round(f1_score(y_test, y_pred), 4),
        })

    df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print(df.to_string(index=False))
    print("═"*60)
    print(f"  ★  Best model: {best_name}")
