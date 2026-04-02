"""
╔══════════════════════════════════════════════════════════════╗
║        ADVANCED GRADIENT BOOSTING PROJECT                    ║
║  ─────────────────────────────────────────────────────────   ║
║  Dataset : Adult Income (UCI ML Repository)                  ║
║  Task    : Binary Classification (>50K / ≤50K)               ║
║  ─────────────────────────────────────────────────────────   ║
║  Architecture:                                               ║
║    1. Data Loader         → download + cache                 ║
║    2. EDA                 → 6 diagnostic plots               ║
║    3. Feature Engineering → pipelines, new features          ║
║    4. Model Training      → XGB | LGB | Sklearn + Optuna     ║
║    5. Results             → SHAP, ROC, CM, PR, CV            ║
║    6. Utilities           → timer, logger, persistence       ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#
# from modules import (
#     load_data, run_eda, build_features,
#     train_all_models, display_results,
#     seed_everything, Timer, get_logger, save_model,
# )

from data_loader import load_data
from eda import run_eda
from feature_engineering import build_features
from model_trainer import train_all_models
from results_visualizer import display_results
from utils import seed_everything, Timer, get_logger, save_model




from config import N_TRIALS

logger = get_logger("Main")


def main():
    seed_everything()

    # ── 1. Data Loading ────────────────────────────────────────
    with Timer("Data Loading"):
        df = load_data()

    # ── 2. EDA ────────────────────────────────────────────────
    with Timer("EDA"):
        eda_summary = run_eda(df)

    # ── 3. Feature Engineering ────────────────────────────────
    with Timer("Feature Engineering"):
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         feature_names, preprocessor) = build_features(df)

    # ── 4. Model Training ─────────────────────────────────────
    with Timer("Model Training"):
        models, val_scores, studies, cv_scores, best_name = train_all_models(
            X_train, X_val, y_train, y_val, n_trials=N_TRIALS
        )

    # ── 5. Results ────────────────────────────────────────────
    with Timer("Results Visualization"):
        display_results(
            models       = models,
            val_scores   = val_scores,
            cv_scores    = cv_scores,
            best_name    = best_name,
            X_test       = X_test,
            y_test       = y_test,
            feature_names= feature_names,
            studies      = studies,
        )

    # ── 6. Persist best model ─────────────────────────────────
    save_model(models[best_name], f"best_model_{best_name}")
    save_model(preprocessor, "preprocessor")

    logger.info("Pipeline complete. All outputs in ./reports/")


if __name__ == "__main__":
    main()
