"""
=============================================================
  Module 3 : Feature Engineering
  ─────────────────────────────────────────────────────────
  • Imputation  (median for numeric, most-frequent for cat)
  • Encoding    (Ordinal + Target Encoding for high-card)
  • Scaling     (RobustScaler for numeric)
  • New features (interaction terms, log transforms)
  • Splits into X_train / X_val / X_test / y_*
=============================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, LabelEncoder

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS,
                    TEST_SIZE, VAL_SIZE, RANDOM_STATE)


# ── Public API ───────────────────────────────────────────────

def build_features(df: pd.DataFrame):
    """
    Full feature engineering pipeline.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    feature_names          : list[str]
    preprocessor           : fitted ColumnTransformer
    """
    print("\n" + "═"*60)
    print("  MODULE 3 — Feature Engineering")
    print("═"*60)

    df = _add_features(df.copy())

    # All numeric cols (original + engineered)
    num_cols = [c for c in df.columns
                if c not in CATEGORICAL_COLS + [TARGET_COL]
                and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    # Train / (val+test) split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    # Val / test split from the remaining portion
    val_fraction = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - val_fraction,
        random_state=RANDOM_STATE, stratify=y_tmp
    )

    # Build sklearn preprocessor
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  RobustScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe,      num_cols),
        ("cat", categorical_pipe,  cat_cols),
    ], remainder="drop")

    X_train = preprocessor.fit_transform(X_train)
    X_val   = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)

    # Feature names
    cat_names = preprocessor.named_transformers_["cat"]["encode"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_names

    print(f"  Train : {X_train.shape[0]:,}  Val : {X_val.shape[0]:,}  Test : {X_test.shape[0]:,}")
    print(f"  Total features after engineering: {X_train.shape[1]}")
    print(f"  Engineered (new) features: {len(num_cols) - len(NUMERIC_COLS)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor


# ── Private helpers ──────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven feature engineering for Adult Income dataset."""

    # Log transform highly skewed numerics
    for col in ["capital_gain", "capital_loss", "fnlwgt"]:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    # Interaction: capital net
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]

    # Hours bucket
    df["hours_bucket"] = pd.cut(
        df["hours_per_week"],
        bins=[0, 25, 40, 50, 99],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # Age bucket
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # Education × Age interaction
    df["edu_age"] = df["education_num"] * df["age"]

    # Overworked flag
    df["overworked"] = (df["hours_per_week"] > 50).astype(int)

    # High capital flag
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)

    print(f"[FeatEng] Added {8} engineered features.")
    return df
