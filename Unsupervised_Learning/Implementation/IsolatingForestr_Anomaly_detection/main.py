"""
main.py - Pipeline orchestrator.

Run:
    python main.py

Pipeline stages
───────────────
  1. Data Loader      → load_data()
  2. EDA              → run_eda()
  3. Pre-processing   → preprocess()
  4. Model Training   → train_model()
  5. Evaluation       → evaluate()
  6. Results Display  → display_results()
  7. Report           → generate_report()
"""

import os
import sys
import time
import logging
import io

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
# CP1252 (default Windows console encoding) cannot print box-drawing chars,
# arrows (→), stars (★), or emoji. Wrap stdout/stderr in a UTF-8 writer.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reports/run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Local modules ─────────────────────────────────────────────────────────────
from data_loader  import load_data, get_feature_target
from eda          import run_eda
from preprocessor import preprocess
from model        import train_model
from evaluator    import evaluate
from visualizer   import display_results
from reporter     import generate_report


def main():
    t_start = time.time()
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Isolation Forest — Anomaly Detection        ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── 1. Data Loading ───────────────────────────────────────────────────────
    df = load_data()

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    run_eda(df)

    # ── 3. Pre-processing ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    # ── 4. Model Training (+ optional grid-search) ────────────────────────────
    model, train_scores, test_scores, best_params = train_model(
        X_train, y_train, X_test, y_test
    )

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    report = evaluate(model, X_test, y_test, test_scores)

    # ── 6. Results Visualisation ──────────────────────────────────────────────
    display_results(model, X_test, y_test, test_scores, report, feature_names)

    # ── 7. Report generation ──────────────────────────────────────────────────
    html_path = generate_report(report, best_params, feature_names)

    elapsed = time.time() - t_start
    logger.info("═" * 52)
    logger.info("  Pipeline complete in %.1f s", elapsed)
    logger.info("  HTML report : %s", html_path)
    logger.info("  Plots       : outputs/")
    logger.info("  Model       : models/isolation_forest.pkl")
    logger.info("═" * 52)


if __name__ == "__main__":
    main()