"""
preprocessor.py - Feature engineering & train/test split.

Steps:
  1. Drop constant / near-zero-variance columns (optional)
  2. Scale 'Amount' and 'Time' with RobustScaler   (resistant to outliers)
  3. Train/test split (stratified)
  4. Persist fitted scaler for inference re-use
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from config import (MODEL_DIR, CLASS_COLUMN, TEST_SIZE,
                    RANDOM_STATE, SCALE_FEATURES)

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame):
    """
    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    feature_names   : list[str]
    scaler          : fitted RobustScaler (or None)
    """
    logger.info("──── Pre-processing ──────────────────────────────")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = df.copy()
    df = _drop_low_variance(df)

    if SCALE_FEATURES:
        df, scaler = _scale_columns(df, ["Amount", "Time"])
    else:
        scaler = None

    X = df.drop(columns=[CLASS_COLUMN])
    y = df[CLASS_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    feature_names = X.columns.tolist()
    logger.info("Train size : %d  |  Test size : %d", len(X_train), len(X_test))
    logger.info("Features   : %d  →  %s … ", len(feature_names), feature_names[:5])

    return X_train, X_test, y_train, y_test, feature_names, scaler


# ──────────────────────────────────────────────────────────────────────────────

def _drop_low_variance(df: pd.DataFrame, threshold: float = 1e-8) -> pd.DataFrame:
    """Remove columns whose variance is effectively zero."""
    num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_always = [CLASS_COLUMN]   # never drop the label
    low_var    = [c for c in num_cols
                  if c not in keep_always and df[c].var() < threshold]
    if low_var:
        logger.info("Dropping low-variance cols: %s", low_var)
        df = df.drop(columns=low_var)
    return df


def _scale_columns(df: pd.DataFrame, cols: list[str]):
    """RobustScaler on specified columns (creates *_scaled variants)."""
    cols_present = [c for c in cols if c in df.columns]
    scaler       = RobustScaler()
    scaled       = scaler.fit_transform(df[cols_present])

    for i, c in enumerate(cols_present):
        df[c] = scaled[:, i]     # overwrite in-place

    # Persist scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved → %s", scaler_path)
    return df, scaler
