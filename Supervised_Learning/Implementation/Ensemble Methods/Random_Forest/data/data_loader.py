"""
=============================================================
  data/data_loader.py  –  Load & pre-process the dataset
=============================================================
Primary source  : Titanic CSV (GitHub / Kaggle mirror)
  https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

Offline fallback: sklearn Breast Cancer Wisconsin dataset
  (identical architecture, same preprocessing pipeline)
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    """
    Handles all data ingestion and pre-processing steps.

    Tries to load the Titanic CSV from the configured URL first.
    If the network is unavailable, falls back to the sklearn
    Breast Cancer Wisconsin dataset (569 samples, 30 features,
    binary: Malignant=0 / Benign=1).

    Pipeline
    --------
    1. Load raw data (URL or built-in fallback).
    2. Drop irrelevant columns / handle dataset specifics.
    3. Impute missing values.
    4. Feature engineering.
    5. Encode categoricals.
    6. Train / test split (stratified).
    """

    def __init__(self, url: str = config.DATASET_URL):
        self.url        = url
        self.raw_df     = None
        self.clean_df   = None
        self.X_train    = None
        self.X_test     = None
        self.y_train    = None
        self.y_test     = None
        self._le        = LabelEncoder()
        self._dataset_name = "Unknown"

    # ── public API ────────────────────────────────────────────────────────────
    def load(self) -> "DataLoader":
        """Attempt URL download; fall back to built-in dataset."""
        try:
            print(f"[DataLoader] Fetching Titanic CSV from:\n  {self.url}")
            self.raw_df = pd.read_csv(self.url)
            self._dataset_name = "Titanic (GitHub)"
            print(f"[DataLoader] ✓ Loaded. Shape: {self.raw_df.shape}")
        except Exception as e:
            print(f"[DataLoader] ⚠ URL fetch failed ({e.__class__.__name__}). "
                  f"Using sklearn Breast Cancer dataset instead.")
            self._load_breast_cancer()
        return self

    def _load_breast_cancer(self):
        """Load Breast Cancer Wisconsin as a DataFrame."""
        bc = load_breast_cancer(as_frame=True)
        df = bc.frame.copy()
        # sklearn uses 0=malignant, 1=benign → rename for clarity
        df.rename(columns={"target": config.TARGET_COLUMN}, inplace=True)
        self.raw_df = df
        self._dataset_name = "Breast Cancer Wisconsin (sklearn built-in)"
        print(f"[DataLoader] ✓ Loaded '{self._dataset_name}'. Shape: {self.raw_df.shape}")
        print(f"  Classes: 0 = Malignant, 1 = Benign")

    def preprocess(self) -> "DataLoader":
        """Full preprocessing pipeline (dataset-aware)."""
        df = self.raw_df.copy()

        if "Titanic" in self._dataset_name:
            df = self._preprocess_titanic(df)
        else:
            df = self._preprocess_breast_cancer(df)

        self.clean_df = df
        features = [c for c in df.columns if c != config.TARGET_COLUMN]
        print(f"[DataLoader] Clean shape : {self.clean_df.shape}")
        print(f"[DataLoader] # features  : {len(features)}")
        return self

    def _preprocess_titanic(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=config.FEATURES_TO_DROP, errors="ignore", inplace=True)
        df = df.copy()  # avoid SettingWithCopyWarning on pandas 3+
        df["Age"]      = df["Age"].fillna(df["Age"].median())
        df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
        df["AgeGroup"]   = pd.cut(
            df["Age"], bins=[0, 12, 18, 35, 60, 120], labels=[0, 1, 2, 3, 4],
        ).astype(int)
        df["Sex"]      = self._le.fit_transform(df["Sex"])
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        return df

    def _preprocess_breast_cancer(self, df: pd.DataFrame) -> pd.DataFrame:
        # No missing values, all numeric — just verify & return
        assert df.isnull().sum().sum() == 0, "Unexpected NaNs in breast cancer data"
        return df

    def split(self) -> "DataLoader":
        """Stratified train / test split."""
        X = self.clean_df.drop(columns=[config.TARGET_COLUMN])
        y = self.clean_df[config.TARGET_COLUMN]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size    = config.TEST_SIZE,
            random_state = config.RANDOM_STATE,
            stratify     = y,
        )
        print(
            f"[DataLoader] Train : {self.X_train.shape}  "
            f"Test : {self.X_test.shape}  "
            f"Positive-class rate (train): {self.y_train.mean():.2%}"
        )
        return self

    def run(self) -> "DataLoader":
        """Run the full pipeline in one call."""
        return self.load().preprocess().split()
