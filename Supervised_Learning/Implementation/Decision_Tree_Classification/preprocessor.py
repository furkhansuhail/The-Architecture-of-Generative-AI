"""
Module 3: Preprocessor
=======================
Handles missing values, feature engineering, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    scale_features: bool = False,   # Decision Trees don't need scaling;
                                    # kept as option for comparison models
) -> dict:
    """
    Returns a dict with keys:
        X_train, X_test, y_train, y_test,
        feature_names, imputer, scaler (may be None)
    """
    print("[Preprocessor] Preprocessing data …")

    # ── Separate features / target ────────────────────────────────────────
    X = df.drop(columns=["target"])
    y = df["target"]
    feature_names = list(X.columns)

    # ── Impute missing values (median strategy) ───────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)
    X       = pd.DataFrame(X_imp, columns=feature_names)

    # ── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── Optional scaling ──────────────────────────────────────────────────
    scaler = None
    if scale_features:
        scaler  = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
        X_test  = pd.DataFrame(scaler.transform(X_test),     columns=feature_names)

    print(f"[Preprocessor] Train size : {len(X_train):,}  ({(1-test_size)*100:.0f}%)")
    print(f"[Preprocessor] Test size  : {len(X_test):,}  ({test_size*100:.0f}%)")
    print(f"[Preprocessor] Features   : {feature_names}")

    return dict(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        imputer=imputer,
        scaler=scaler,
    )
