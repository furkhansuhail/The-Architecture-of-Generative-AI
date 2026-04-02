"""
=============================================================
  utils/helpers.py  –  Shared utility functions
=============================================================
"""

import os
import time
import functools
import numpy as np
import pandas as pd
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
def timer(fn: Callable) -> Callable:
    """Decorator that prints execution time of any function."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  ⏱  {fn.__qualname__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def print_banner(title: str, width: int = 60):
    """Print a section header banner."""
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝\n")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def describe_bagging():
    """Print a textual explanation of how RF = Bagging + DT."""
    text = """
  ┌─────────────────────────────────────────────────────────┐
  │           BAGGING  vs  RANDOM FOREST                    │
  ├─────────────────────────────────────────────────────────┤
  │  Pure Bagging                                           │
  │  • Bootstrap sample of training rows                    │
  │  • Train a full Decision Tree on each sample            │
  │  • Aggregate predictions by majority vote               │
  │  • Reduces VARIANCE but trees can be correlated         │
  │                                                         │
  │  Random Forest  (= Bagging + feature randomisation)     │
  │  • Same bootstrap sampling as Bagging                   │
  │  • At every split, consider only a RANDOM SUBSET        │
  │    of sqrt(p) features   (key differentiator!)          │
  │  • De-correlates trees → stronger ensemble              │
  │  • OOB rows (~36.8 %) give a free validation estimate   │
  │                                                         │
  │  Why it works: Bias–Variance decomposition              │
  │    Error = Bias² + Variance + Noise                     │
  │    Each deep tree ≈ low bias, high variance             │
  │    Averaging B trees → variance ↓ by ~1/B               │
  │    Feature subsampling → further de-correlates trees    │
  └─────────────────────────────────────────────────────────┘
  """
    print(text)


def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Proportion"] = counts["Count"] / counts["Count"].sum()
    return counts
