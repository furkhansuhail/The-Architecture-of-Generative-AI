"""
Module 1: Data Loader
=====================
Loads the Heart Disease dataset from UCI ML Repository.
Handles downloading, caching, and basic validation.
"""

import os
import pandas as pd
import numpy as np


# ── Column metadata ──────────────────────────────────────────────────────────
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

FEATURE_DESCRIPTIONS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl (1=True)",
    "restecg":  "Resting ECG Results (0-2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise Induced Angina (1=Yes)",
    "oldpeak":  "ST Depression Induced by Exercise",
    "slope":    "Slope of Peak Exercise ST Segment",
    "ca":       "Number of Major Vessels (0-3)",
    "thal":     "Thalassemia (1=Normal, 2=Fixed, 3=Reversible)",
    "target":   "Heart Disease (1=Yes, 0=No)"
}

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")


# ── Loader ───────────────────────────────────────────────────────────────────
def load_data(force_download: bool = False) -> pd.DataFrame:
    """
    Returns a clean DataFrame of the Heart Disease dataset.

    Steps
    -----
    1. Load from cache (CSV) if available, else download from UCI.
    2. Replace '?' missing markers with NaN.
    3. Convert all columns to numeric.
    4. Binarise the target column (0 = no disease, 1 = disease).
    5. Cache to disk for future runs.
    """
    cache = os.path.abspath(CACHE_PATH)

    if os.path.exists(cache) and not force_download:
        print(f"[DataLoader] Loading from cache: {cache}")
        df = pd.read_csv(cache)
    else:
        print(f"[DataLoader] Downloading from UCI repository …")
        df = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values="?")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"[DataLoader] Saved to cache: {cache}")

    # Ensure numeric types
    df = df.apply(pd.to_numeric, errors="coerce")

    # Binarise target (original values 0-4; 0 = no disease, 1-4 = disease)
    df["target"] = (df["target"] > 0).astype(int)

    print(f"[DataLoader] Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"[DataLoader] Missing values: {df.isnull().sum().sum()}")
    return df


def get_feature_names(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != "target"]


def get_feature_descriptions() -> dict:
    return FEATURE_DESCRIPTIONS


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.dtypes)
