"""
=============================================================
  Module 6 : Utilities
  ─────────────────────────────────────────────────────────
  • Timer context manager
  • Logger setup
  • Save / load model helpers
  • Reproducibility seed setter
=============================================================
"""

import os, time, pickle, logging
import numpy as np
import random

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORTS_DIR, RANDOM_STATE


# ── Reproducibility ──────────────────────────────────────────

def seed_everything(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass


# ── Timer ────────────────────────────────────────────────────

class Timer:
    """Context manager that prints elapsed wall-clock time."""
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self._start
        mins, secs = divmod(int(elapsed), 60)
        print(f"  ⏱  {self.label} — {mins}m {secs}s")


# ── Logger ───────────────────────────────────────────────────

def get_logger(name: str = "GBProject") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(asctime)s %(name)s — %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Model persistence ────────────────────────────────────────

def save_model(model, name: str):
    path = os.path.join(REPORTS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [Utils] Model saved → {path}")


def load_model(name: str):
    path = os.path.join(REPORTS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Metrics pretty-printer ───────────────────────────────────

def print_banner(text: str, width: int = 60, char: str = "═"):
    border = char * width
    pad    = (width - len(text) - 2) // 2
    print(f"\n{border}")
    print(f"{char}{' ' * pad}{text}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{border}\n")
