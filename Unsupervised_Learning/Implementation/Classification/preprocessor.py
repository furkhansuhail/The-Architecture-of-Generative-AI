"""
preprocessor.py - Feature engineering & scaling pipeline.

Steps:
  1. Drop noisy columns (e.g. Time)
  2. Log-transform and standard-scale 'Amount'
  3. Standard-scale PCA features (V1–V28)
  4. Optional PCA dimensionality reduction
  5. Returns train-ready numpy arrays + metadata
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config


# ──────────────────────────────────────────────────────────────────────────────
class Preprocessor:
    def __init__(self):
        self.scaler      = StandardScaler()
        self.pca         = PCA(n_components=config.PCA_COMPONENTS,
                               random_state=config.RANDOM_STATE) if config.USE_PCA else None
        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []
        self._fitted = False

    # ── public API ────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit all transformations on df and return scaled feature matrix."""
        X, y = self._prepare(df)
        self.feature_names_in_ = list(X.columns)

        # ── Scale ─────────────────────────────────────────────────────────
        X_scaled = self.scaler.fit_transform(X)

        # ── PCA ───────────────────────────────────────────────────────────
        if self.pca is not None:
            X_scaled = self.pca.fit_transform(X_scaled)
            self.feature_names_out_ = [f"PC{i+1}" for i in range(X_scaled.shape[1])]
            var_explained = self.pca.explained_variance_ratio_.cumsum()[-1]
            print(f"[Preprocessor] PCA: {config.PCA_COMPONENTS} components "
                  f"explain {var_explained*100:.1f}% of variance.")
        else:
            self.feature_names_out_ = self.feature_names_in_

        self.y = y.values
        self._fitted = True
        print(f"[Preprocessor] Output shape: {X_scaled.shape}")
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations to new data (no refit)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        X, _ = self._prepare(df)
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return X_scaled

    @property
    def labels(self) -> np.ndarray:
        return self.y

    def pca_variance_report(self) -> dict:
        if self.pca is None:
            return {}
        return {
            "n_components"       : self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": self.pca.explained_variance_ratio_.cumsum().tolist(),
        }

    # ── private helpers ───────────────────────────────────────────────────────
    def _prepare(self, df: pd.DataFrame):
        df = df.copy()

        # Drop noisy columns
        drop_cols = [c for c in config.DROP_COLUMNS if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)

        # Log-transform Amount
        if config.SCALE_AMOUNT and "Amount" in df.columns:
            df["Amount"] = np.log1p(df["Amount"])

        # Separate features and target
        y = df[config.TARGET_COLUMN] if config.TARGET_COLUMN in df.columns else None
        X = df.drop(columns=[config.TARGET_COLUMN], errors="ignore")

        return X, y
