"""
========================================================
  Module 1: Data Loader
  Dataset: Breast Cancer Wisconsin (Diagnostic)
  Source:  sklearn / Kaggle / UCI ML Repository
  Link:    https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
========================================================
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


# ──────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_dataset(from_csv: bool = False) -> pd.DataFrame:
    """
    Load the Breast Cancer Wisconsin dataset.

    Priority:
        1. CSV file inside  data/breast_cancer.csv  (if user downloaded from Kaggle)
        2. sklearn's built-in version (identical data)

    Returns
    -------
    df : pd.DataFrame
        Raw dataframe with features + 'target' column (0 = malignant, 1 = benign).
    """
    csv_path = os.path.join(DATA_DIR, "breast_cancer.csv")

    if from_csv and os.path.exists(csv_path):
        print(f"[DataLoader] Loading from CSV → {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
        df["target"] = df["diagnosis"].map({"M": 0, "B": 1})
        df = df.drop(columns=["diagnosis"])
    else:
        print("[DataLoader] Loading from sklearn (mirrors Kaggle/UCI dataset)")
        raw    = load_breast_cancer()
        df     = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target          # 0 = malignant, 1 = benign

    print(f"[DataLoader] Dataset shape : {df.shape}")
    print(f"[DataLoader] Class distribution :\n{df['target'].value_counts().rename({0:'Malignant', 1:'Benign'})}\n")
    return df


def preprocess(df: pd.DataFrame):
    """
    Split into train / test and scale features with StandardScaler.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    scaler                            : fitted StandardScaler
    feature_names                     : list[str]
    """
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[DataLoader] Train size : {X_train.shape[0]} samples")
    print(f"[DataLoader] Test  size : {X_test.shape[0]}  samples")
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ──────────────────────────────────────────────────────
# Quick sanity-check when run directly
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    print(f"Feature count : {len(features)}")
