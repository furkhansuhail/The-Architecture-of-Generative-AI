"""
=============================================================
  config.py  –  Central configuration for the RF project
=============================================================
"""

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets"
    "/master/titanic.csv"
)
TARGET_COLUMN = "Survived"
FEATURES_TO_DROP = ["PassengerId", "Name", "Ticket", "Cabin"]

# ── Train / Test split ────────────────────────────────────────────────────────
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# ── Random Forest hyper-parameters ───────────────────────────────────────────
RF_PARAMS = {
    "n_estimators"   : 200,   # number of trees (bags)
    "max_depth"      : None,  # None = grow until pure leaves
    "max_features"   : "sqrt",
    "bootstrap"      : True,  # bagging = sample with replacement
    "oob_score"      : True,  # out-of-bag evaluation
    "random_state"   : RANDOM_STATE,
    "n_jobs"         : -1,
}

# ── Grid-search space (optional fine-tuning) ─────────────────────────────────
GRID_PARAMS = {
    "n_estimators" : [100, 200, 300],
    "max_depth"    : [None, 5, 10, 15],
    "max_features" : ["sqrt", "log2"],
}
CV_FOLDS = 5

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR      = "outputs"
REPORT_FILENAME = "model_report.txt"
