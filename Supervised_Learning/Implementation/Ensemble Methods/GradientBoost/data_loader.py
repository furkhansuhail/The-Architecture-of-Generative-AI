"""
=============================================================
  Module 1 : Data Loader
  ─────────────────────────────────────────────────────────
  Generates a realistic synthetic "Adult Income" dataset
  (32,561 rows) mirroring the UCI Adult / Census dataset.
=============================================================
"""

import os
import pandas as pd
import numpy as np

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DATA_FILE, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS

_RNG = np.random.default_rng(42)
_N   = 32_561


def load_data(force_generate: bool = False) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if force_generate or not os.path.exists(DATA_FILE):
        df = _generate()
        df.to_csv(DATA_FILE, index=False)
        print(f"[DataLoader] Synthetic dataset saved → {DATA_FILE}")
    else:
        df = pd.read_csv(DATA_FILE)
        print(f"[DataLoader] Loaded cached dataset  → {DATA_FILE}")
    print(f"[DataLoader] Shape: {df.shape[0]:,} rows × {df.shape[1]} cols  "
          f"| Missing: {df.isnull().sum().sum():,} cells")
    return df


def _generate() -> pd.DataFrame:
    print("[DataLoader] Generating synthetic Adult-Income dataset …")
    rng = _RNG
    n   = _N

    age = rng.integers(17, 91, n).astype(float)
    sex = rng.choice(["Male", "Female"], n, p=[0.67, 0.33])
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        n, p=[0.856, 0.096, 0.031, 0.010, 0.007]
    )
    native_country = rng.choice(
        ["United-States", "Mexico", "Philippines", "Germany",
         "Canada", "India", "Cuba", "South", "Other"],
        n, p=[0.897, 0.019, 0.007, 0.006, 0.005, 0.005, 0.005, 0.005, 0.051]
    )

    edu_levels = [
        ("HS-grad", 9, 0.322), ("Some-college", 10, 0.224),
        ("Bachelors", 13, 0.164), ("Masters", 14, 0.053),
        ("Assoc-voc", 11, 0.042), ("11th", 7, 0.038),
        ("Assoc-acdm", 12, 0.032), ("10th", 6, 0.028),
        ("7th-8th", 4, 0.020), ("Prof-school", 15, 0.018),
        ("9th", 5, 0.015), ("12th", 8, 0.013),
        ("Doctorate", 16, 0.013), ("5th-6th", 3, 0.009),
        ("1st-4th", 2, 0.005), ("Preschool", 1, 0.002),
    ]
    edu_probs = np.array([e[2] for e in edu_levels]); edu_probs /= edu_probs.sum()
    edu_idx   = rng.choice(len(edu_levels), n, p=edu_probs)
    education     = np.array([e[0] for e in edu_levels])[edu_idx]
    education_num = np.array([e[1] for e in edu_levels])[edu_idx].astype(float)

    wc_vals  = ["Private", "Self-emp-not-inc", "Local-gov",
                "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay"]
    wc_probs = np.array([0.694, 0.078, 0.064, 0.040, 0.034, 0.030, 0.004])
    wc_probs /= wc_probs.sum()
    wc_base  = rng.choice(wc_vals, n, p=wc_probs)
    wc_nan   = rng.random(n) < 0.056
    workclass = wc_base.astype(object)
    workclass[wc_nan] = None

    occ_vals  = ["Prof-specialty", "Craft-repair", "Exec-managerial",
                 "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct",
                 "Transport-moving", "Handlers-cleaners", "Tech-support",
                 "Farming-fishing", "Protective-serv", "Priv-house-serv", "Armed-Forces"]
    occ_probs = np.array([0.127, 0.125, 0.123, 0.116, 0.108, 0.107,
                          0.077, 0.049, 0.045, 0.033, 0.032, 0.020, 0.004, 0.001])
    occ_probs /= occ_probs.sum()
    occ_base  = rng.choice(occ_vals, n, p=occ_probs)
    occ_nan   = rng.random(n) < 0.056
    occupation = occ_base.astype(object)
    occupation[occ_nan] = None

    marital_status = rng.choice(
        ["Married-civ-spouse", "Never-married", "Divorced",
         "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
        n, p=[0.459, 0.330, 0.136, 0.034, 0.025, 0.014, 0.002]
    )
    relationship = rng.choice(
        ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"],
        n, p=[0.402, 0.256, 0.155, 0.104, 0.054, 0.029]
    )

    hours_per_week = np.clip(rng.normal(40, 12, n).astype(int), 1, 99).astype(float)
    fnlwgt         = rng.integers(12_285, 1_490_400, n).astype(float)

    gain_mask    = rng.random(n) < 0.092
    capital_gain = np.where(gain_mask,
        rng.choice([7298,14084,5013,3103,2964,99999,25236,13550,4934,2407], n), 0
    ).astype(float)
    loss_mask    = rng.random(n) < 0.048
    capital_loss = np.where(loss_mask,
        rng.choice([1902,2042,1590,1977,2205,1887,1848,1740,2267,2824], n), 0
    ).astype(float)

    edu_score     = (education_num - 9) * 0.18
    age_score     = (age - 38) * 0.02
    cap_score     = np.log1p(capital_gain) * 0.07
    hrs_score     = (hours_per_week - 40) * 0.015
    occ_bonus     = np.where(np.isin(occupation,
        ["Exec-managerial","Prof-specialty","Tech-support","Protective-serv"]), 0.9, 0.0)
    married_bonus = np.where(marital_status == "Married-civ-spouse", 0.4, 0.0)
    sex_penalty   = np.where(sex == "Female", -0.3, 0.0)

    logit  = (-1.8 + edu_score + age_score + cap_score + hrs_score
              + occ_bonus + married_bonus + sex_penalty + rng.normal(0, 0.5, n))
    income = (rng.random(n) < (1 / (1 + np.exp(-logit)))).astype(int)

    df = pd.DataFrame({
        "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
        "education": education, "education_num": education_num,
        "marital_status": marital_status, "occupation": occupation,
        "relationship": relationship, "race": race, "sex": sex,
        "capital_gain": capital_gain, "capital_loss": capital_loss,
        "hours_per_week": hours_per_week, "native_country": native_country,
        "income": income,
    })
    print(f"[DataLoader] Generated {n:,} rows | >50K rate: {income.mean():.1%}")
    return df

if __name__ == "__main__":
    df = load_data(force_generate=True)
    print(df.head())
