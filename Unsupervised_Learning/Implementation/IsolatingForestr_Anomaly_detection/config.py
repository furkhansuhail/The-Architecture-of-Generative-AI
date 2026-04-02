"""
config.py - Central configuration for the Isolation Forest Anomaly Detection project.
All hyperparameters, paths, and flags are controlled from here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
REPORT_DIR      = os.path.join(BASE_DIR, "reports")

# ── Dataset ────────────────────────────────────────────────────────────────────
# Primary: Kaggle Credit Card Fraud  (place CSV here after downloading)
KAGGLE_CSV      = os.path.join(DATA_DIR, "creditcard.csv")

# Fallback: synthetic dataset generated automatically
USE_SYNTHETIC   = True           # Set False if you have the Kaggle CSV
N_SAMPLES       = 10_000         # synthetic dataset size
CONTAMINATION   = 0.02           # expected anomaly fraction (2 %)

# ── Pre-processing ─────────────────────────────────────────────────────────────
TEST_SIZE       = 0.20           # 80/20 train-test split
RANDOM_STATE    = 42
SCALE_FEATURES  = True           # StandardScaler on Amount + Time

# ── Isolation Forest hyper-params ─────────────────────────────────────────────
IF_PARAMS = {
    "n_estimators"     : 200,          # number of trees
    "max_samples"      : "auto",       # subsample size per tree
    "contamination"    : CONTAMINATION,
    "max_features"     : 1.0,
    "bootstrap"        : False,
    "n_jobs"           : -1,
    "random_state"     : RANDOM_STATE,
    "verbose"          : 0,
}

# ── Advanced tuning (GridSearch / RandomSearch) ────────────────────────────────
TUNE_MODEL      = True
PARAM_GRID = {
    "n_estimators"  : [100, 200, 300],
    "max_samples"   : [0.5, 0.75, "auto"],
    "contamination" : [0.01, 0.02, 0.05],
    "max_features"  : [0.8, 1.0],
}

# ── EDA ────────────────────────────────────────────────────────────────────────
CORR_TOP_N      = 15             # top N features in correlation heat-map
CLASS_COLUMN    = "Class"        # 0 = normal, 1 = fraud / anomaly

# ── Visualisation ──────────────────────────────────────────────────────────────
FIGURE_DPI      = 150
PALETTE         = {"normal": "#4C72B0", "anomaly": "#DD4444"}
