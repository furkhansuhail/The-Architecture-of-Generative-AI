"""
data_loader.py — Loads the Breast Cancer Wisconsin dataset (sklearn built-in).

About the dataset
─────────────────
• 569 samples, 30 numeric features derived from digitized images of fine-needle
  aspirate (FNA) biopsies of breast masses.
• Binary target: 0 = Malignant, 1 = Benign
• No missing values; a realistic, medium-complexity classification problem.
• Source: UCI ML Repository / sklearn.datasets.load_breast_cancer
"""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from config import DATA_PATH, TARGET_COLUMN


# ──────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load the Breast Cancer dataset from sklearn and return a single DataFrame.
    Caches a CSV copy to DATA_PATH for reproducibility and offline re-use.
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if os.path.exists(DATA_PATH):
        print(f"[DataLoader] Cache hit — loading from '{DATA_PATH}'")
        df = pd.read_csv(DATA_PATH)
    else:
        print("[DataLoader] Loading Breast Cancer Wisconsin dataset from sklearn …")
        raw    = load_breast_cancer(as_frame=True)
        df     = raw.frame.copy()
        # sklearn uses 'target' (0=malignant,1=benign); rename to our config key
        df     = df.rename(columns={"target": TARGET_COLUMN})
        df.to_csv(DATA_PATH, index=False)
        print(f"[DataLoader] Cached to '{DATA_PATH}'")

    print(f"[DataLoader] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    print(data.head())
