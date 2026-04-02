"""
preprocessing.py — Cleaning, imputation, scaling, and train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from config import (
    TARGET_COLUMN, ZERO_AS_NAN_COLS,
    TEST_SIZE, RANDOM_STATE,
)


# ──────────────────────────────────────────────────────────────────────────────
def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace physiologically impossible zero values with NaN so they can be
    properly imputed rather than biasing the model.
    """
    df = df.copy()
    for col in ZERO_AS_NAN_COLS:
        n_zeros = (df[col] == 0).sum()
        df[col] = df[col].replace(0, np.nan)
        print(f"[Preprocessing] {col:30s}: replaced {n_zeros} zeros with NaN")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """KNN imputation on the feature columns (target excluded)."""
    features = [c for c in df.columns if c != TARGET_COLUMN]
    target   = df[TARGET_COLUMN].values

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df[features]),
        columns=features,
    )
    df_imputed[TARGET_COLUMN] = target
    n_missing_before = df[features].isnull().sum().sum()
    print(f"[Preprocessing] KNN imputed {n_missing_before} missing values.")
    return df_imputed


# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction and ratio features for the breast cancer domain.
    Worst / Mean ratios capture how extreme the worst measurement is relative
    to the average cell, which is clinically meaningful.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    # Ratio of worst to mean for the three core measurements
    for measurement in ["radius", "texture", "perimeter", "area", "smoothness"]:
        worst_col = f"worst {measurement}"
        mean_col  = f"mean {measurement}"
        if worst_col in df.columns and mean_col in df.columns:
            df[f"{measurement}_worst_mean_ratio"] = (
                df[worst_col] / (df[mean_col] + 1e-9)
            )

    added = df.shape[1] - 1 - len(feature_cols)
    print(f"[Preprocessing] Feature engineering added {added} new features. "
          f"Total features: {df.shape[1] - 1}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def split_and_scale(df: pd.DataFrame):
    """
    Returns
    -------
    X_train, X_test, y_train, y_test  (scaled NumPy arrays)
    feature_names                      (list[str])
    scaler                             (fitted StandardScaler)
    """
    features = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[features].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[Preprocessing] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[Preprocessing] Class split — train 1s: "
          f"{y_train.sum()}/{len(y_train)}  "
          f"test 1s: {y_test.sum()}/{len(y_test)}")

    return X_train, X_test, y_train, y_test, features, scaler


# ──────────────────────────────────────────────────────────────────────────────
def run_preprocessing(df: pd.DataFrame):
    """Full preprocessing pipeline."""
    from config import ZERO_AS_NAN_COLS
    if ZERO_AS_NAN_COLS:
        df = replace_zeros_with_nan(df)
        df = impute_missing(df)
    else:
        missing = df.isnull().sum().sum()
        print(f"[Preprocessing] No zero-imputation needed. Missing values: {missing}")
    df = engineer_features(df)
    return split_and_scale(df)
