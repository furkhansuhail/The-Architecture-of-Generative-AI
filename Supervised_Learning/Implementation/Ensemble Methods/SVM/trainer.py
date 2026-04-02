"""
========================================================
  Module 3: SVM Model Training
  - Baseline SVM with default params
  - GridSearchCV hyperparameter tuning
  - Cross-validation
  - Model persistence (joblib)
========================================================
"""

import numpy as np
import joblib
import os
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────
# Hyperparameter Grid
# ──────────────────────────────────────────────────────
PARAM_GRID = {
    "svc__C"      : [0.01, 0.1, 1, 10, 100],
    "svc__gamma"  : ["scale", "auto", 0.001, 0.01],
    "svc__kernel" : ["rbf", "linear", "poly"],
}

CV_FOLDS    = 5
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Return a bare SVC pipeline (data already scaled)."""
    return Pipeline([("svc", SVC(probability=True, random_state=RANDOM_STATE))])


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """Train a baseline SVM with default hyperparameters."""
    print("[Trainer] Training baseline SVM (C=1, RBF kernel) …")
    model = SVC(kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("[Trainer] Baseline training complete.\n")
    return model


def train_with_tuning(X_train: np.ndarray,
                      y_train: np.ndarray,
                      verbose: bool = True) -> GridSearchCV:
    """
    Run GridSearchCV to find the best SVM hyperparameters.

    Returns
    -------
    gs : fitted GridSearchCV object
    """
    print("[Trainer] Starting GridSearchCV …")
    print(f"          Param grid size : {np.prod([len(v) for v in PARAM_GRID.values()])} combos")
    print(f"          CV folds         : {CV_FOLDS}")

    pipe = build_pipeline()
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator  = pipe,
        param_grid = PARAM_GRID,
        cv         = cv,
        scoring    = "f1",          # optimise for F1
        n_jobs     = -1,
        verbose    = 1 if verbose else 0,
        refit      = True,
    )

    t0 = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n[Trainer] Grid search done in {elapsed:.1f}s")
    print(f"[Trainer] Best params : {gs.best_params_}")
    print(f"[Trainer] Best CV F1  : {gs.best_score_:.4f}\n")
    return gs


def cross_validate_model(model,
                          X_train: np.ndarray,
                          y_train: np.ndarray) -> dict:
    """
    Evaluate model stability with stratified cross-validation.

    Returns dict with mean/std for accuracy, f1, roc_auc.
    """
    print("[Trainer] Running cross-validation …")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    metrics = {}
    for metric in ["accuracy", "f1", "roc_auc"]:
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring=metric, n_jobs=-1)
        metrics[metric] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {metric:<12}: {scores.mean():.4f} ± {scores.std():.4f}")

    print()
    return metrics


def save_model(model, name: str = "best_svm.pkl") -> str:
    """Persist trained model to disk."""
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, path)
    print(f"[Trainer] Model saved → {path}")
    return path


def load_model(name: str = "best_svm.pkl"):
    """Load model from disk."""
    path = os.path.join(MODEL_DIR, name)
    model = joblib.load(path)
    print(f"[Trainer] Model loaded ← {path}")
    return model


# ──────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_dataset, preprocess

    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)

    # Baseline
    baseline = train_baseline(X_train, y_train)

    # Tuned
    gs  = train_with_tuning(X_train, y_train)
    cv  = cross_validate_model(gs.best_estimator_, X_train, y_train)

    save_model(gs.best_estimator_)
