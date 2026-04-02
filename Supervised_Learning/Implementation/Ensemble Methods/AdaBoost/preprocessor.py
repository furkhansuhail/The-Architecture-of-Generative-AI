"""
preprocessor.py
---------------
Handles all data preprocessing steps:
  - Missing value handling
  - Feature / target split
  - Train / test split
  - Feature scaling
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Transforms raw DataFrame into ML-ready arrays.

    Parameters
    ----------
    test_size   : float  — fraction held out for testing (default 0.20)
    random_state: int    — reproducibility seed

    Usage
    -----
    >>> prep = Preprocessor()
    >>> X_train, X_test, y_train, y_test = prep.fit_transform(df)
    >>> feature_names = prep.feature_names
    """

    TARGET_COL = "target"

    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────
    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full preprocessing pipeline."""
        df = self._handle_missing(df)
        X, y = self._split_features_target(df)
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)
        X_train, X_test = self._scale(X_train, X_test)
        self._print_summary(y_train, y_test)
        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data (after fit_transform has been called)."""
        df = self._handle_missing(df)
        X = df[self.feature_names].values
        return self.scaler.transform(X)

    # ── Internals ─────────────────────────────────────────────────────────────
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isnull().sum().sum()
        if missing:
            print(f"[Preprocessor] Filling {missing} missing values with column medians.")
            df = df.fillna(df.median(numeric_only=True))
        return df

    def _split_features_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        self.feature_names = [c for c in df.columns if c != self.TARGET_COL]
        X = df[self.feature_names]
        y = df[self.TARGET_COL]
        return X, y

    def _train_test_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple:
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def _scale(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)
        return X_train_sc, X_test_sc

    def _print_summary(self, y_train, y_test):
        print(f"\n[Preprocessor] Split summary")
        print(f"  Train size : {len(y_train)} samples")
        print(f"  Test  size : {len(y_test)}  samples")
        print(f"  Features   : {len(self.feature_names)} → {self.feature_names}")
