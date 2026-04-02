"""
data_loader.py - Handles dataset acquisition and loading.

Supports:
  1. Loading real Kaggle creditcard.csv if present in /data
  2. Generating a realistic synthetic dataset for demo purposes
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import config


# ──────────────────────────────────────────────────────────────────────────────
class DataLoader:
    """
    Loads the Credit Card Fraud dataset.

    If the Kaggle CSV is present in config.DATA_DIR it will be used.
    Otherwise a synthetic dataset is generated that mirrors the real
    dataset's structure (28 PCA features + Amount + Class).
    """

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.source: str = ""

    # ── public API ────────────────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        if os.path.exists(config.DATASET_PATH):
            print(f"[DataLoader] ✅  Found real dataset → {config.DATASET_PATH}")
            self.df = self._load_kaggle_csv()
            self.source = "kaggle"
        else:
            print("[DataLoader] ⚠️  Kaggle CSV not found.  Generating synthetic data …")
            print(f"[DataLoader]    To use the real data, download it from:")
            print(f"[DataLoader]    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print(f"[DataLoader]    and place it at: {config.DATASET_PATH}\n")
            self.df = self._generate_synthetic()
            self.source = "synthetic"

        if config.SAMPLE_SIZE and len(self.df) > config.SAMPLE_SIZE:
            self.df = self.df.sample(
                n=config.SAMPLE_SIZE, random_state=config.RANDOM_STATE
            ).reset_index(drop=True)
            print(f"[DataLoader] Sampled {config.SAMPLE_SIZE:,} rows for faster processing.")

        print(f"[DataLoader] Dataset shape: {self.df.shape}")
        print(f"[DataLoader] Fraud rate   : "
              f"{self.df[config.TARGET_COLUMN].mean() * 100:.3f}%\n")
        return self.df

    def get_feature_names(self) -> list[str]:
        """Return feature column names (exclude target)."""
        if self.df is None:
            raise RuntimeError("Call load() first.")
        return [c for c in self.df.columns if c != config.TARGET_COLUMN]

    # ── private helpers ───────────────────────────────────────────────────────
    def _load_kaggle_csv(self) -> pd.DataFrame:
        df = pd.read_csv(config.DATASET_PATH)
        required = {"Amount", "Class"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing expected columns: {required - set(df.columns)}")
        return df

    def _generate_synthetic(self) -> pd.DataFrame:
        """
        Generates a synthetic dataset that resembles the real credit card
        fraud dataset:
          - 28 'V' features (simulated PCA components – multivariate Gaussian)
          - 'Amount' column (log-normal)
          - 'Time' column (uniform)
          - 'Class' label (0=normal, 1=fraud) at ~0.17% fraud rate
        """
        rng = np.random.default_rng(config.RANDOM_STATE)
        n_total  = 284_807          # mirror real dataset size
        n_fraud  = 492
        n_normal = n_total - n_fraud

        # ── Normal transactions: clustered in PCA space ─────────────────────
        centers_normal = np.zeros((5, 28))   # 5 normal behaviour clusters
        for i, c in enumerate(centers_normal):
            c[i * 5 : i * 5 + 5] = rng.uniform(-1, 1, 5)

        X_normal, _ = make_blobs(
            n_samples=n_normal, centers=centers_normal,
            cluster_std=1.2, random_state=config.RANDOM_STATE
        )
        y_normal = np.zeros(n_normal, dtype=int)

        # ── Fraudulent transactions: sparse outlier clusters ────────────────
        X_fraud = rng.normal(loc=0, scale=3.5, size=(n_fraud, 28))
        X_fraud[:, 0] += rng.choice([-8, 8], size=n_fraud)   # bimodal shift
        y_fraud = np.ones(n_fraud, dtype=int)

        # ── Combine & add Amount / Time ─────────────────────────────────────
        X = np.vstack([X_normal, X_fraud])
        y = np.concatenate([y_normal, y_fraud])

        col_names = [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(X, columns=col_names)

        df["Amount"] = np.where(
            y == 0,
            rng.lognormal(mean=3.0, sigma=1.5, size=len(y)),
            rng.lognormal(mean=4.5, sigma=1.0, size=len(y)),
        )
        df["Time"]  = rng.uniform(0, 172_800, size=len(y))
        df["Class"] = y

        # Shuffle
        df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        return df
