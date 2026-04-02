"""
main.py — Orchestrates the full XGBoost pipeline.

Usage
-----
    python main.py              # standard run (no hyperparameter tuning)
    python main.py --tune       # run RandomizedSearchCV before final fit
    python main.py --no-eda     # skip EDA plots (faster)
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Make sure imports resolve when running from the project folder
sys.path.insert(0, os.path.dirname(__file__))

from data_loader   import load_data
from eda           import run_eda
from preprocessing import run_preprocessing
from model         import (
    cross_validate_model,
    tune_hyperparameters,
    train_model,
    evaluate_model,
)
from results       import run_results
from report        import generate_report


# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost Diabetes Prediction Pipeline")
    parser.add_argument("--tune",   action="store_true",
                        help="Run hyperparameter search before final fit")
    parser.add_argument("--no-eda", action="store_true",
                        help="Skip EDA plots")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    banner = """
╔══════════════════════════════════════════════════════════════╗
║          XGBoost  •  Pima Diabetes Prediction                ║
║          Intermediate ML Project  •  Full Pipeline           ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1 — DATA LOADING")
    print("=" * 60)
    df = load_data()

    # ── 2. EDA ───────────────────────────────────────────────────────────────
    if not args.no_eda:
        print("\n" + "=" * 60)
        print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        run_eda(df)

    # ── 3. Preprocessing ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — PREPROCESSING")
    print("=" * 60)
    X_train, X_test, y_train, y_test, feature_names, scaler = run_preprocessing(df)

    # ── 4. Cross-validation ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — CROSS-VALIDATION")
    print("=" * 60)
    cv_summary = cross_validate_model(X_train, y_train)

    # ── 5. (Optional) Hyperparameter tuning ─────────────────────────────────
    best_params = None
    if args.tune:
        print("\n" + "=" * 60)
        print("  STEP 5 — HYPERPARAMETER TUNING")
        print("=" * 60)
        best_params = tune_hyperparameters(X_train, y_train)

    # ── 6. Final model training ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 — FINAL MODEL TRAINING")
    print("=" * 60)
    model = train_model(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        best_params=best_params,
    )

    # ── 7. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7 — EVALUATION")
    print("=" * 60)
    metrics = evaluate_model(model, X_test, y_test)

    # ── 8. Results / visualisations ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 8 — RESULTS & VISUALISATIONS")
    print("=" * 60)
    run_results(model, X_test, y_test, feature_names, metrics, cv_summary)

    # ── 9. Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 9 — REPORT GENERATION")
    print("=" * 60)
    generate_report(metrics, cv_summary, feature_names)

    print("\n✅  Pipeline complete!")
    print(f"   Plots  → plots/")
    print(f"   Report → outputs/report.md\n")


if __name__ == "__main__":
    main()
