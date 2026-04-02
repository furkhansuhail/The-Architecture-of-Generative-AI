"""
config.py — Central configuration for the XGBoost Breast Cancer Prediction Project
"""

# ─── Dataset ───────────────────────────────────────────────────────────────────
# Breast Cancer Wisconsin (Diagnostic) — sklearn built-in, no download needed
# 569 samples | 30 features | Binary target: 0 = Malignant, 1 = Benign
TARGET_COLUMN  = "Outcome"          # renamed from sklearn's "target"
DATA_PATH      = "data/breast_cancer.csv"

# ─── Preprocessing ─────────────────────────────────────────────────────────────
ZERO_AS_NAN_COLS = []               # this dataset has no physiological zeros
TEST_SIZE        = 0.20
RANDOM_STATE     = 42

# ─── Model ─────────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        4,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,          # auto-tuned in model.py
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

CV_FOLDS        = 5
EARLY_STOPPING  = 20
THRESHOLD       = 0.50              # decision threshold for binary classification

# ─── Paths ─────────────────────────────────────────────────────────────────────
PLOTS_DIR   = "plots"
OUTPUTS_DIR = "outputs"
