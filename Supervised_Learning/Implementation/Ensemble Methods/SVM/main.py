"""
========================================================
  main.py  ─  SVM Classification Pipeline
  Dataset : Breast Cancer Wisconsin (Diagnostic)
  Source  : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
            (mirrors sklearn's built-in load_breast_cancer)

  Run:
      python main.py

  Outputs:
      outputs/   — All EDA & results plots (PNG)
      models/    — Serialised best model  (pkl)
========================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

from data_loader import load_dataset, preprocess
from eda         import run_eda
from trainer     import (train_baseline, train_with_tuning,
                          cross_validate_model, save_model)
from results     import display_all_results

TUNE_MODEL = True          # Set False for speed (skips GridSearchCV)


def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  SVM Classification Pipeline — Breast Cancer Dataset   ║")
    print("╚" + "═"*58 + "╝\n")

    # ── 1. Data Loading ─────────────────────────────────────
    print("▶ STEP 1 — Data Loading")
    df = load_dataset()

    # ── 2. EDA ──────────────────────────────────────────────
    print("▶ STEP 2 — Exploratory Data Analysis")
    run_eda(df)

    # ── 3. Preprocessing ────────────────────────────────────
    print("▶ STEP 3 — Preprocessing & Train/Test Split")
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)

    # ── 4. Training ──────────────────────────────────────────
    print("▶ STEP 4 — Model Training")

    baseline_model = train_baseline(X_train, y_train)
    baseline_cv    = cross_validate_model(baseline_model, X_train, y_train)

    if TUNE_MODEL:
        gs          = train_with_tuning(X_train, y_train)
        tuned_model = gs.best_estimator_
        tuned_cv    = cross_validate_model(tuned_model, X_train, y_train)
    else:
        gs          = None
        tuned_model = baseline_model
        tuned_cv    = baseline_cv

    save_model(tuned_model)

    # ── 5. Results ───────────────────────────────────────────
    print("▶ STEP 5 — Displaying Results")
    display_all_results(
        baseline_model = baseline_model,
        tuned_model    = tuned_model,
        gs             = gs,
        X_train        = X_train,
        X_test         = X_test,
        y_train        = y_train,
        y_test         = y_test,
        feature_names  = features,
        baseline_cv    = baseline_cv,
        tuned_cv       = tuned_cv,
    )

    print("╔" + "═"*58 + "╗")
    print("║  Pipeline complete! See outputs/ for all plots.        ║")
    print("╚" + "═"*58 + "╝\n")


if __name__ == "__main__":
    main()
