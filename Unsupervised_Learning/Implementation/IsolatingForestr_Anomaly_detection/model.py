"""
model.py - Isolation Forest training with optional hyper-parameter search.

IsolationForest recap
─────────────────────
  • Builds an ensemble of random binary trees (iTrees).
  • Anomalies are isolated in fewer splits → shorter path lengths.
  • anomaly_score = − average_path_length (lower = more anomalous).
  • `predict` returns +1 (normal) or -1 (anomaly).

Advanced module features
────────────────────────
  • GridSearchCV-compatible scorer built on ROC-AUC (semi-supervised style).
  • Saves best model as pickle.
  • Exposes raw anomaly scores for downstream visualisation.
"""

import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score
from config import (MODEL_DIR, IF_PARAMS, TUNE_MODEL,
                    PARAM_GRID, RANDOM_STATE, CLASS_COLUMN)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                X_test: pd.DataFrame,
                y_test: pd.Series):
    """
    Train (and optionally tune) an Isolation Forest.

    Returns
    -------
    model        : fitted IsolationForest
    train_scores : anomaly scores for X_train  (lower = more anomalous)
    test_scores  : anomaly scores for X_test
    best_params  : dict of hyper-params used
    """
    logger.info("──── Model Training ─────────────────────────────")
    os.makedirs(MODEL_DIR, exist_ok=True)

    if TUNE_MODEL:
        model, best_params = _tune_isolation_forest(X_train, y_train, X_test, y_test)
    else:
        model       = _fit(X_train, IF_PARAMS)
        best_params = IF_PARAMS

    # ── Anomaly scores  (decision_function: higher = more normal) ────────────
    #    We negate so that higher score ↔ more anomalous (intuitive)
    train_scores = -model.decision_function(X_train)
    test_scores  = -model.decision_function(X_test)

    # ── Quick eval log ───────────────────────────────────────────────────────
    auc = roc_auc_score(y_test, test_scores)
    ap  = average_precision_score(y_test, test_scores)
    logger.info("Test ROC-AUC             : %.4f", auc)
    logger.info("Test Avg Precision (AP)  : %.4f", ap)

    # ── Persist model ────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", model_path)

    return model, train_scores, test_scores, best_params


def load_model() -> IsolationForest:
    """Load a previously saved model."""
    model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fit(X: pd.DataFrame, params: dict) -> IsolationForest:
    """Fit a single IsolationForest with given params."""
    t0    = time.time()
    model = IsolationForest(**params)
    model.fit(X)
    logger.info("IsolationForest trained in %.2f s", time.time() - t0)
    return model


def _tune_isolation_forest(X_train, y_train, X_val, y_val):
    """
    Manual grid search maximising ROC-AUC on the validation set.
    (IsolationForest is unsupervised at fit-time, but we can evaluate
    anomaly scores against known labels to pick the best combo.)
    """
    logger.info("Starting hyper-parameter grid search …")
    grid         = list(ParameterGrid(PARAM_GRID))
    logger.info("  Grid size : %d combinations", len(grid))

    best_auc, best_params, best_model = -1, None, None

    for i, params in enumerate(grid):
        p = {**params, "random_state": RANDOM_STATE, "n_jobs": -1}
        try:
            clf    = _fit(X_train, p)
            scores = -clf.decision_function(X_val)
            auc    = roc_auc_score(y_val, scores)
        except Exception as exc:
            logger.debug("  combo %d failed: %s", i, exc)
            continue

        if auc > best_auc:
            best_auc, best_params, best_model = auc, p, clf
            logger.info("  [%d/%d] ★ new best ROC-AUC=%.4f | %s",
                        i + 1, len(grid), best_auc, params)

    logger.info("Best params : %s", best_params)
    logger.info("Best val ROC-AUC : %.4f", best_auc)
    return best_model, best_params
