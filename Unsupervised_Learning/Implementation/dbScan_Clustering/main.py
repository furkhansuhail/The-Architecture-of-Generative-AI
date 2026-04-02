"""
main.py - Master pipeline orchestrator.

Runs the full DBSCAN Anomaly Detection workflow:

  [1] Load Data
  [2] EDA
  [3] Preprocess (scale + PCA)
  [4] k-Distance Plot  (eps selection guide)
  [5] Hyperparameter Tuning  (optional — set RUN_TUNING = True)
  [6] Train DBSCAN
  [7] Evaluate
  [8] Visualise Results
  [9] Generate Report

Usage:
    python main.py
    python main.py --tune       # enable hyperparameter grid search
    python main.py --no-tsne    # skip t-SNE (faster)
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the directory containing this file is always on sys.path so that
# sibling modules (config, data_loader, eda, …) can be imported regardless
# of which working directory Python is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from data_loader   import DataLoader
from eda           import EDA
from preprocessor  import Preprocessor
from model         import DBSCANModel
from visualizer    import Visualizer
from reporter      import Reporter


# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DBSCAN Fraud Anomaly Detection")
    p.add_argument("--tune",    action="store_true",
                   help="Run hyperparameter grid search before final model")
    p.add_argument("--no-tsne", action="store_true",
                   help="Skip t-SNE visualisation (saves time)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    t0 = time.time()

    banner = """
╔══════════════════════════════════════════════════════╗
║   DBSCAN Anomaly Detection — Credit Card Fraud       ║
║   Architecture: Data → EDA → Preprocess →            ║
║                 Tune → Train → Evaluate → Visualise  ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── [1] DATA LOADING ─────────────────────────────────────────────────────
    print("━" * 56)
    print("  STEP 1 / 7 : DATA LOADING")
    print("━" * 56)
    loader = DataLoader()
    df     = loader.load()

    # ── [2] EDA ───────────────────────────────────────────────────────────────
    print("━" * 56)
    print("  STEP 2 / 7 : EXPLORATORY DATA ANALYSIS")
    print("━" * 56)
    eda = EDA(df)
    eda.run()

    # ── [3] PREPROCESSING ────────────────────────────────────────────────────
    print("━" * 56)
    print("  STEP 3 / 7 : PREPROCESSING")
    print("━" * 56)
    preprocessor = Preprocessor()
    X_scaled     = preprocessor.fit_transform(df)
    y_true       = preprocessor.labels
    pca_report   = preprocessor.pca_variance_report()

    # PCA variance plot
    viz = Visualizer()
    viz.plot_pca_variance(pca_report)

    # ── [4] k-DISTANCE PLOT (EPS SELECTION) ──────────────────────────────────
    print("━" * 56)
    print("  STEP 4 / 7 : k-DISTANCE PLOT (eps selection)")
    print("━" * 56)
    eda.kdistance_plot(X_scaled)

    # ── [5] HYPERPARAMETER TUNING (optional) ─────────────────────────────────
    model      = DBSCANModel()
    tuning_df  = None
    if args.tune:
        print("━" * 56)
        print("  STEP 5 / 7 : HYPERPARAMETER TUNING")
        print("━" * 56)
        tuning_df = model.tune(X_scaled, y_true)
        viz.plot_tuning_heatmaps(tuning_df)
    else:
        print(f"[Main] Skipping tuning — using config defaults "
              f"eps={config.DBSCAN_EPS}  min_samples={config.DBSCAN_MIN_SAMPLES}")
        print("[Main] Run with --tune to enable grid search.\n")

    # ── [6] TRAIN DBSCAN ─────────────────────────────────────────────────────
    print("━" * 56)
    print("  STEP 6 / 7 : TRAINING DBSCAN MODEL")
    print("━" * 56)
    cluster_labels = model.train(X_scaled)

    # ── [7] EVALUATE ─────────────────────────────────────────────────────────
    print("━" * 56)
    print("  STEP 7 / 7 : EVALUATION & VISUALISATION")
    print("━" * 56)
    results = model.evaluate(X_scaled, y_true)

    # ── [8] VISUALISE RESULTS ─────────────────────────────────────────────────
    viz.plot_clusters_2d(X_scaled, cluster_labels, y_true, method="pca")

    if not args.no_tsne:
        viz.plot_clusters_2d(X_scaled, cluster_labels, y_true, method="tsne")
    else:
        print("[Visualizer] t-SNE skipped (--no-tsne flag).")

    viz.plot_confusion_matrix(results["confusion_matrix"])
    viz.plot_cluster_sizes(cluster_labels)
    viz.plot_summary_dashboard(results, y_true, model.anomaly_preds)

    # ── [9] REPORT ────────────────────────────────────────────────────────────
    reporter = Reporter()
    reporter.build(
        data_info={
            "source"    : loader.source,
            "rows"      : len(df),
            "features"  : len(loader.get_feature_names()),
            "fraud_rate": f"{y_true.mean()*100:.3f}%",
        },
        pca_report  = pca_report,
        eval_results= results,
        tuning_df   = tuning_df,
    )
    reporter.save()
    reporter.print_summary()

    elapsed = time.time() - t0
    print(f"[Main] ✅  Pipeline complete in {elapsed:.1f}s")
    print(f"[Main] 📂  All outputs → {config.OUTPUT_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

