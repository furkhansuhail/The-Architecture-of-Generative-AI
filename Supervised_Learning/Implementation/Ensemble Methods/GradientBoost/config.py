"""
=============================================================
  Gradient Boosting Project - Configuration
=============================================================
  Dataset : Adult Income (UCI / OpenML)
  Task    : Binary Classification (Income > 50K or ≤ 50K)
  Models  : XGBoost | LightGBM | Sklearn GradientBoosting
=============================================================
"""

import os

# ── Paths ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
DATA_FILE    = os.path.join(DATA_DIR, "adult_income.csv")

# ── Dataset ────────────────────────────────────────────────
DATASET_URL  = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
TARGET_COL   = "income"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15      # fraction of train set

# ── Feature groups ─────────────────────────────────────────
NUMERIC_COLS = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week",
]
CATEGORICAL_COLS = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex", "native_country",
]

# ── Optuna / CV ────────────────────────────────────────────
N_TRIALS     = 40
CV_FOLDS     = 5
SCORING      = "roc_auc"

# ── Model defaults (fallback if Optuna is skipped) ─────────
XGB_DEFAULTS = dict(
    n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, use_label_encoder=False,
    eval_metric="logloss", random_state=RANDOM_STATE,
    n_jobs=-1,
)
LGB_DEFAULTS = dict(
    n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=RANDOM_STATE,
    n_jobs=-1, verbose=-1,
)
SKL_DEFAULTS = dict(
    max_iter=300, learning_rate=0.05,
    max_depth=5, random_state=RANDOM_STATE,
)

# ── Plot settings ──────────────────────────────────────────
FIG_DPI      = 150
PALETTE      = "husl"
