"""
config.py - Central configuration for the DBSCAN Anomaly Detection Project
Dataset: Credit Card Fraud Detection (Kaggle)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [DATA_DIR, OUTPUT_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
DATASET_FILENAME = "creditcard.csv"
DATASET_PATH     = os.path.join(DATA_DIR, DATASET_FILENAME)

# How many rows to sample for demo (None = full dataset)
SAMPLE_SIZE      = 10_000
RANDOM_STATE     = 42

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
TARGET_COLUMN    = "Class"          # 0 = normal, 1 = fraud
DROP_COLUMNS     = ["Time"]         # 'Time' is noisy for clustering
SCALE_AMOUNT     = True             # Log-scale + StandardScale 'Amount'

# PCA for dimensionality reduction before DBSCAN
USE_PCA          = True
PCA_COMPONENTS   = 10               # Reduce V1-V28 + Amount to N dims

# ─────────────────────────────────────────────
# DBSCAN HYPERPARAMETERS
# ─────────────────────────────────────────────
DBSCAN_EPS       = 1.8              # Neighbourhood radius
DBSCAN_MIN_SAMPLES = 10             # Min points to form a core point
DBSCAN_METRIC    = "euclidean"

# Hyper-parameter search grid (used by tuner)
EPS_RANGE        = (0.5, 3.0)
MIN_SAMPLES_RANGE = (5, 30)

# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
TSNE_PERPLEXITY  = 30
TSNE_N_ITER      = 1000
FIGURE_DPI       = 150
COLORMAP         = "tab10"
