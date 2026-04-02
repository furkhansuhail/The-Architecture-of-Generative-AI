"""
data_loader.py - Handles loading from Kaggle CSV *or* auto-generating a
                 realistic synthetic fraud dataset as a fallback.

Dataset:  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - 284 807 transactions, 492 frauds (~0.172 %)
  - Features V1-V28 are PCA-transformed; Amount & Time are raw.
  - Class: 0 = normal, 1 = fraud
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from config import (KAGGLE_CSV, USE_SYNTHETIC, N_SAMPLES,
                    CONTAMINATION, DATA_DIR, CLASS_COLUMN, RANDOM_STATE)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Returns a DataFrame with the same schema as the Kaggle credit-card CSV.
    Priority:
      1. Kaggle CSV  (if USE_SYNTHETIC=False and file exists)
      2. Synthetic   (always available – no download needed)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not USE_SYNTHETIC and os.path.exists(KAGGLE_CSV):
        logger.info("Loading Kaggle Credit Card Fraud dataset …")
        df = _load_kaggle()
    else:
        logger.info("Generating synthetic dataset (USE_SYNTHETIC=True or CSV missing) …")
        df = _generate_synthetic()

    logger.info("Dataset shape : %s", df.shape)
    logger.info("Class balance :\n%s", df[CLASS_COLUMN].value_counts(normalize=True).to_string())
    return df


def get_feature_target(df: pd.DataFrame):
    """Split DataFrame into features X and binary label y."""
    X = df.drop(columns=[CLASS_COLUMN])
    y = df[CLASS_COLUMN]
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_kaggle() -> pd.DataFrame:
    """Load & lightly validate the Kaggle CSV."""
    df = pd.read_csv(KAGGLE_CSV)
    required = {"Time", "Amount", CLASS_COLUMN}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Kaggle CSV is missing columns: {missing}")
    logger.info("Kaggle CSV loaded  |  rows=%d  fraud=%d",
                len(df), df[CLASS_COLUMN].sum())
    return df


def _generate_synthetic() -> pd.DataFrame:
    """
    Build a synthetic dataset that mimics the Kaggle schema:
      - V1 … V28 : PCA-like numeric features
      - Time      : seconds since first transaction
      - Amount    : transaction amount (log-normal)
      - Class     : 0 = normal, 1 = fraud
    """
    rng            = np.random.default_rng(RANDOM_STATE)
    n_anomalies    = int(N_SAMPLES * CONTAMINATION)
    n_normal       = N_SAMPLES - n_anomalies
    n_features     = 28      # V1 … V28

    # ── Normal transactions ──────────────────────────────────────────────────
    X_normal = rng.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal,
    )

    # ── Fraudulent transactions (shifted + noisy) ────────────────────────────
    shift        = rng.uniform(-3, 3, n_features)   # hidden pattern
    X_fraud      = rng.multivariate_normal(
        mean=shift,
        cov=np.eye(n_features) * 2,
        size=n_anomalies,
    )

    X_all  = np.vstack([X_normal, X_fraud])
    labels = np.array([0] * n_normal + [1] * n_anomalies)

    # ── Shuffle ──────────────────────────────────────────────────────────────
    idx    = rng.permutation(N_SAMPLES)
    X_all, labels = X_all[idx], labels[idx]

    # ── Build DataFrame ───────────────────────────────────────────────────────
    v_cols = [f"V{i}" for i in range(1, n_features + 1)]
    df     = pd.DataFrame(X_all, columns=v_cols)

    df["Time"]       = np.sort(rng.uniform(0, 172_800, N_SAMPLES)[idx])   # 48 h window
    df["Amount"]     = np.exp(rng.normal(3.5, 1.5, N_SAMPLES))            # log-normal
    df[CLASS_COLUMN] = labels

    # Inject stronger signal into Amount for fraud rows
    fraud_mask              = df[CLASS_COLUMN] == 1
    df.loc[fraud_mask, "Amount"] *= rng.uniform(5, 20, fraud_mask.sum())

    logger.info("Synthetic dataset  |  total=%d  fraud=%d", N_SAMPLES, n_anomalies)
    return df
