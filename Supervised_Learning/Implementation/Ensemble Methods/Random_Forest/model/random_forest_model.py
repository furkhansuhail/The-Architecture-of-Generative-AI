"""
=============================================================
  model/random_forest_model.py  –  RF = Bagging + Decision Trees
=============================================================

  How Random Forest is Bagging with Decision Trees
  -------------------------------------------------
  ┌──────────────────────────────────────────────────────────┐
  │  Random Forest ≡ Bagging Ensemble of Decision Trees      │
  │                                                          │
  │  For each of the B trees:                                │
  │   1. Draw a bootstrap sample  (sampling WITH replacement)│
  │      ↳ ~63.2 % unique rows per bag; ~36.8 % OOB rows    │
  │   2. Grow a full Decision Tree on the bootstrap sample   │
  │      but at each split consider only  √p  random features│
  │      (this de-correlates the trees – key RF innovation)  │
  │   3. No pruning → low-bias / high-variance per tree      │
  │                                                          │
  │  Prediction (classification):                            │
  │   Majority vote across all B trees                       │
  │                                                          │
  │  OOB Error:                                              │
  │   Each sample is OOB for ~1/3 of trees → free validation │
  └──────────────────────────────────────────────────────────┘
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─────────────────────────────────────────────────────────────────────────────
class RandomForestModel:
    """
    Wraps sklearn's RandomForestClassifier.

    Also exposes an explicit BaggingClassifier(base_estimator=DecisionTree)
    for side-by-side educational comparison.
    """

    def __init__(self):
        self.rf_model      = None   # sklearn RandomForestClassifier
        self.bagging_model = None   # explicit BaggingClassifier (for comparison)
        self.best_params   = None
        self._trained      = False

    # ── build ─────────────────────────────────────────────────────────────────
    def build(self, params: dict = None) -> "RandomForestModel":
        p = params or config.RF_PARAMS
        self.rf_model = RandomForestClassifier(**p)

        # Explicit bagging variant (same idea, no feature subsampling at splits)
        self.bagging_model = BaggingClassifier(
            estimator         = DecisionTreeClassifier(max_features="sqrt"),
            n_estimators      = p.get("n_estimators", 200),
            max_samples       = 1.0,          # use full bootstrap sample
            bootstrap         = True,
            oob_score         = True,
            random_state      = config.RANDOM_STATE,
            n_jobs            = -1,
        )
        print("[Model] RandomForestClassifier and BaggingClassifier built.")
        return self

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, X_train, y_train) -> "RandomForestModel":
        print("[Model] Training RandomForestClassifier …")
        self.rf_model.fit(X_train, y_train)

        print("[Model] Training BaggingClassifier (for comparison) …")
        self.bagging_model.fit(X_train, y_train)

        self._trained = True
        print(f"[Model] RF  OOB accuracy : {self.rf_model.oob_score_:.4f}")
        print(f"[Model] BAG OOB accuracy : {self.bagging_model.oob_score_:.4f}")
        return self

    # ── cross-validation ──────────────────────────────────────────────────────
    def cross_validate(self, X, y, cv: int = config.CV_FOLDS) -> dict:
        print(f"\n[Model] {cv}-fold Cross Validation …")
        scores = cross_val_score(
            self.rf_model, X, y,
            cv=cv, scoring="accuracy", n_jobs=-1,
        )
        result = {
            "scores" : scores,
            "mean"   : scores.mean(),
            "std"    : scores.std(),
        }
        print(f"  CV Accuracy: {result['mean']:.4f} ± {result['std']:.4f}")
        return result

    # ── grid search ───────────────────────────────────────────────────────────
    def tune(self, X_train, y_train) -> "RandomForestModel":
        print(f"\n[Model] Grid-search over {config.GRID_PARAMS} …")
        gs = GridSearchCV(
            RandomForestClassifier(
                bootstrap    = True,
                oob_score    = False,
                random_state = config.RANDOM_STATE,
                n_jobs       = -1,
            ),
            param_grid = config.GRID_PARAMS,
            cv         = config.CV_FOLDS,
            scoring    = "accuracy",
            n_jobs     = -1,
            verbose    = 1,
        )
        gs.fit(X_train, y_train)
        self.best_params = gs.best_params_
        print(f"  Best params : {self.best_params}")
        print(f"  Best CV acc : {gs.best_score_:.4f}")
        # Rebuild with best params + oob_score=True (GridSearch used oob_score=False)
        self.rf_model = RandomForestClassifier(
            **self.best_params,
            bootstrap    = True,
            oob_score    = True,
            random_state = config.RANDOM_STATE,
            n_jobs       = -1,
        )
        return self

    # ── feature importance ────────────────────────────────────────────────────
    def feature_importance(self, feature_names) -> pd.DataFrame:
        imp = pd.DataFrame({
            "feature"   : feature_names,
            "importance": self.rf_model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp
