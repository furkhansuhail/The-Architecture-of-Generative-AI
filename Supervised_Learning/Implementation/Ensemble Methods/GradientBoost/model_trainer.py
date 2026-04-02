"""
=============================================================
  Module 4 : Model Training
  ─────────────────────────────────────────────────────────
  1. Baseline  – Sklearn HistGradientBoosting
  2. XGBoost   – tuned with Optuna (Bayesian search)
  3. LightGBM  – tuned with Optuna
  4. Ensemble  – soft-voting of all three
  5. Cross-validation summary for best model
=============================================================
"""

import os, warnings, time
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_TRIALS, CV_FOLDS, RANDOM_STATE, SCORING,
                    XGB_DEFAULTS, LGB_DEFAULTS, SKL_DEFAULTS)


# ── Public API ───────────────────────────────────────────────

def train_all_models(X_train, X_val, y_train, y_val,
                     n_trials: int = N_TRIALS):
    """
    Train three GB variants + ensemble.

    Returns
    -------
    dict {model_name: fitted_model}
    dict {model_name: val_roc_auc}
    dict {model_name: optuna.Study or None}
    """
    print("\n" + "═"*60)
    print("  MODULE 4 — Model Training")
    print("═"*60)

    results   = {}
    val_scores = {}
    studies   = {}

    # 1. Sklearn baseline (fast, no tuning)
    print("\n  [1/4] Sklearn HistGradientBoosting (baseline) …")
    skl_model = _train_sklearn(X_train, y_train, X_val, y_val)
    results["sklearn_gb"]   = skl_model
    val_scores["sklearn_gb"] = _eval(skl_model, X_val, y_val, "Sklearn GB")

    # 2. XGBoost + Optuna
    print(f"\n  [2/4] XGBoost — Optuna tuning ({n_trials} trials) …")
    xgb_model, xgb_study = _train_xgb_optuna(X_train, y_train, X_val, y_val, n_trials)
    results["xgboost"]   = xgb_model
    val_scores["xgboost"] = _eval(xgb_model, X_val, y_val, "XGBoost")
    studies["xgboost"]   = xgb_study

    # 3. LightGBM + Optuna
    print(f"\n  [3/4] LightGBM — Optuna tuning ({n_trials} trials) …")
    lgb_model, lgb_study = _train_lgb_optuna(X_train, y_train, X_val, y_val, n_trials)
    results["lightgbm"]   = lgb_model
    val_scores["lightgbm"] = _eval(lgb_model, X_val, y_val, "LightGBM")
    studies["lightgbm"]   = lgb_study

    # 4. Soft-voting ensemble
    print("\n  [4/4] Ensemble (soft-voting) …")
    ensemble = _build_ensemble(results)
    ensemble.fit(X_train, y_train)
    results["ensemble"]   = ensemble
    val_scores["ensemble"] = _eval(ensemble, X_val, y_val, "Ensemble")

    # Cross-validation on best single model
    best_name = max(
        {k: v for k, v in val_scores.items() if k != "ensemble"},
        key=val_scores.get
    )
    print(f"\n  ▸ Running {CV_FOLDS}-fold CV on best model ({best_name}) …")
    cv_scores = _cross_validate(results[best_name], X_train, y_train)
    print(f"    CV AUC : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    _print_leaderboard(val_scores)
    return results, val_scores, studies, cv_scores, best_name


# ── Private: trainers ────────────────────────────────────────

def _train_sklearn(X_train, y_train, X_val, y_val):
    model = HistGradientBoostingClassifier(**SKL_DEFAULTS)
    model.fit(X_train, y_train)
    return model


def _train_xgb_optuna(X_train, y_train, X_val, y_val, n_trials):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 800),
            max_depth         = trial.suggest_int("max_depth", 3, 9),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
            gamma             = trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            eval_metric       = "logloss",
            use_label_encoder = False,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
        )
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"eval_metric": "logloss",
                         "use_label_encoder": False,
                         "random_state": RANDOM_STATE, "n_jobs": -1})
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"    Best XGB AUC (val): {study.best_value:.4f}")
    return model, study


def _train_lgb_optuna(X_train, y_train, X_val, y_val, n_trials):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 800),
            max_depth         = trial.suggest_int("max_depth", 3, 9),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            num_leaves        = trial.suggest_int("num_leaves", 20, 150),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbose           = -1,
        )
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1})
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
    print(f"    Best LGB AUC (val): {study.best_value:.4f}")
    return model, study


def _build_ensemble(models: dict):
    estimators = [
        ("sklearn_gb", models["sklearn_gb"]),
        ("xgboost",    models["xgboost"]),
        ("lightgbm",   models["lightgbm"]),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def _cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return scores


def _eval(model, X_val, y_val, name: str) -> float:
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"    {name:<22} Val AUC = {auc:.4f}")
    return auc


def _print_leaderboard(val_scores: dict):
    print("\n  ── Leaderboard ────────────────────────────────")
    sorted_scores = sorted(val_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_scores, 1):
        bar = "█" * int(score * 40)
        print(f"  #{rank}  {name:<22} {score:.4f}  {bar}")
    print("  " + "─"*50)
