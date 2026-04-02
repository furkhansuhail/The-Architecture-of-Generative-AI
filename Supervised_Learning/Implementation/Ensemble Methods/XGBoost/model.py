"""
model.py — XGBoost training with cross-validation and optional hyperparameter search.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    log_loss, matthews_corrcoef,
)
from config import (
    XGBOOST_PARAMS, CV_FOLDS, EARLY_STOPPING,
    THRESHOLD, RANDOM_STATE,
)


# ──────────────────────────────────────────────────────────────────────────────
def _build_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
    return XGBClassifier(**params)


# ──────────────────────────────────────────────────────────────────────────────
def cross_validate_model(X_train, y_train) -> dict:
    """Run stratified K-fold CV and return mean / std metrics."""
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = _build_model(scale_pos_weight=ratio)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]

    print(f"\n[Model] Running {CV_FOLDS}-fold cross-validation …")
    results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    cv_summary = {}
    for metric in scoring:
        key   = f"test_{metric}"
        mean_ = results[key].mean()
        std_  = results[key].std()
        cv_summary[metric] = {"mean": mean_, "std": std_}
        print(f"  {metric:12s}: {mean_:.4f} ± {std_:.4f}")

    return cv_summary


# ──────────────────────────────────────────────────────────────────────────────
def tune_hyperparameters(X_train, y_train, n_iter: int = 30) -> dict:
    """
    Light RandomizedSearchCV over the most impactful XGBoost knobs.
    Returns the best parameter dictionary.
    """
    param_dist = {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [3, 4, 5, 6],
        "learning_rate":     [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight":  [1, 3, 5, 7],
        "gamma":             [0, 0.1, 0.2, 0.3],
        "reg_alpha":         [0, 0.01, 0.1, 1],
        "reg_lambda":        [0.5, 1, 1.5, 2],
    }

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    base  = _build_model(scale_pos_weight=ratio)
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[Model] Hyperparameter search ({n_iter} iterations) …")
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"[Model] Best ROC-AUC (CV): {search.best_score_:.4f}")
    print(f"[Model] Best params: {search.best_params_}")
    return search.best_params_


# ──────────────────────────────────────────────────────────────────────────────
def train_model(
    X_train, y_train,
    X_val=None, y_val=None,
    best_params: dict | None = None,
) -> XGBClassifier:
    """
    Train the final XGBoost model. If best_params supplied, those override
    the defaults in config.py.
    """
    ratio  = (y_train == 0).sum() / (y_train == 1).sum()
    params = {**XGBOOST_PARAMS, "scale_pos_weight": ratio}
    if best_params:
        params.update(best_params)

    model = XGBClassifier(**params)

    eval_set = [(X_train, y_train)]
    if X_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    print("[Model] Training complete.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model: XGBClassifier, X_test, y_test) -> dict:
    """Compute a comprehensive set of test-set metrics."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)

    metrics = {
        "accuracy":          accuracy_score(y_test, preds),
        "roc_auc":           roc_auc_score(y_test, proba),
        "f1":                f1_score(y_test, preds),
        "precision":         precision_score(y_test, preds),
        "recall":            recall_score(y_test, preds),
        "log_loss":          log_loss(y_test, proba),
        "mcc":               matthews_corrcoef(y_test, preds),
    }

    sep = "─" * 50
    print(f"\n{sep}")
    print("  TEST SET METRICS")
    print(sep)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(sep + "\n")

    return metrics
