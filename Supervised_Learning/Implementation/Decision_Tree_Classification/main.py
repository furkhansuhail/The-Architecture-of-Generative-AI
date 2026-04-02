"""
Main Pipeline
=============
Orchestrates all modules end-to-end:
  1. Data Loader
  2. EDA
  3. Preprocessor
  4. Trainer
  5. Results
  6. Report Generator
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from data_loader       import load_data
from eda               import run_eda
from preprocessor      import preprocess
from trainer           import train_decision_tree, cost_complexity_prune
from results           import (
    evaluate, plot_confusion_matrix, plot_roc_pr_curves,
    plot_feature_importance, plot_decision_tree, plot_cv_scores,
    plot_model_comparison, plot_depth_vs_accuracy,
)
from report_generator  import generate_html_report

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    print("=" * 60)
    print("  HEART DISEASE CLASSIFICATION — DECISION TREE PIPELINE")
    print("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    df = load_data()

    dataset_info = dict(
        rows    = len(df),
        cols    = df.shape[1] - 1,
        missing = df.isnull().sum().sum(),
        pos_pct = df["target"].mean() * 100,
    )

    # ── 2. EDA ────────────────────────────────────────────────────────────
    eda_paths = run_eda(df)

    # ── 3. Preprocess ─────────────────────────────────────────────────────
    proc = preprocess(df, test_size=0.20, random_state=42)
    X_train, X_test   = proc["X_train"], proc["X_test"]
    y_train, y_test   = proc["y_train"], proc["y_test"]
    feature_names     = proc["feature_names"]

    # ── 4. Train ──────────────────────────────────────────────────────────
    train_result = train_decision_tree(X_train, y_train, tune=True)
    best_model   = train_result["model"]
    base_model   = train_result["base_model"]
    cv_scores    = train_result["cv_scores"]
    best_params  = train_result["best_params"]

    pruned_model = cost_complexity_prune(X_train, y_train, X_test, y_test)

    # ── 5. Evaluate & Plot ────────────────────────────────────────────────
    m_best   = evaluate(best_model,   X_test, y_test, "Tuned Tree")
    m_base   = evaluate(base_model,   X_test, y_test, "Baseline Tree")
    m_pruned = evaluate(pruned_model, X_test, y_test, "Pruned Tree")

    print("\n[Main] Generating plots …")
    os.makedirs(OUT_DIR, exist_ok=True)

    p_cm    = plot_confusion_matrix(best_model, X_test, y_test)
    p_roc   = plot_roc_pr_curves(best_model, X_test, y_test)
    p_imp   = plot_feature_importance(best_model, feature_names)
    p_tree  = plot_decision_tree(best_model, feature_names, max_depth=3)
    p_cv    = plot_cv_scores(cv_scores)
    p_comp  = plot_model_comparison({"Baseline": m_base,
                                      "Tuned": m_best,
                                      "Pruned": m_pruned})
    p_depth = plot_depth_vs_accuracy(X_train, y_train, X_test, y_test)

    # ── 6. Generate HTML Report ───────────────────────────────────────────
    report_path = generate_html_report(
        metrics      = {"Baseline": m_base, "Tuned": m_best, "Pruned": m_pruned},
        tree_info    = {
            "depth":  best_model.get_depth(),
            "leaves": best_model.get_n_leaves(),
        },
        plot_paths   = {
            "class_dist":    eda_paths.get("class_dist"),
            "distributions": eda_paths.get("distributions"),
            "heatmap":       eda_paths.get("heatmap"),
            "boxplots":      eda_paths.get("boxplots"),
            "confusion":     p_cm,
            "roc_pr":        p_roc,
            "importance":    p_imp,
            "tree_viz":      p_tree,
            "cv_scores":     p_cv,
            "comparison":    p_comp,
            "depth_acc":     p_depth,
        },
        feature_names = feature_names,
        dataset_info  = dataset_info,
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best Model Accuracy : {m_best['accuracy']:.4f}")
    print(f"  Best Model F1       : {m_best['f1']:.4f}")
    print(f"  Best Model ROC AUC  : {m_best['roc_auc']:.4f}")
    print(f"  HTML Report         : {report_path}")
    print("=" * 60)

    return report_path


if __name__ == "__main__":
    main()
