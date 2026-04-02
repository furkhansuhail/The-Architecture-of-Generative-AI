"""
model.py
--------
AdaBoost training module.

Trains three variants for comparison:
  1. AdaBoost (Decision Stump base)
  2. AdaBoost (Deeper tree base)
  3. Baseline: single Decision Tree

Also runs GridSearchCV to find optimal hyperparameters.
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


class AdaBoostTrainer:
    """
    Trains and tunes AdaBoost models.

    Usage
    -----
    >>> trainer = AdaBoostTrainer()
    >>> results = trainer.train(X_train, y_train)
    >>> best    = trainer.best_model
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict = {}
        self.cv_scores: dict = {}
        self.best_model = None
        self.best_model_name: str = ""
        self.grid_search_result: GridSearchCV | None = None

    # ── Public API ────────────────────────────────────────────────────────────
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Train all variants + run hyperparameter search.
        Returns dict of {name: fitted_model}.
        """
        print("\n" + "="*60)
        print("  MODEL TRAINING")
        print("="*60)

        self._train_baseline(X_train, y_train)
        self._train_adaboost_stump(X_train, y_train)
        self._train_adaboost_deep(X_train, y_train)
        self._grid_search(X_train, y_train)
        self._select_best()

        print(f"\n[Model] Best model selected: '{self.best_model_name}'")
        return self.models

    def get_feature_importances(self, feature_names: list[str]) -> dict:
        """Return feature importances of the best AdaBoost model."""
        model = self.models.get("AdaBoost (Stump)")
        if model is None or not hasattr(model, "feature_importances_"):
            return {}
        return dict(zip(feature_names, model.feature_importances_))

    # ── Internals ─────────────────────────────────────────────────────────────
    def _cv_score(self, model, X, y, label: str) -> float:
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean, std = scores.mean(), scores.std()
        print(f"  [{label:30s}] CV Accuracy: {mean:.4f} ± {std:.4f}")
        self.cv_scores[label] = {"mean": mean, "std": std, "scores": scores}
        return mean

    def _train_baseline(self, X, y):
        dt = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
        dt.fit(X, y)
        self._cv_score(dt, X, y, "Baseline Decision Tree")
        self.models["Baseline (Decision Tree)"] = dt

    def _train_adaboost_stump(self, X, y):
        stump = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
        ada = AdaBoostClassifier(
            estimator=stump,
            n_estimators=100,
            learning_rate=0.5,
            random_state=self.random_state,
        )
        ada.fit(X, y)
        self._cv_score(ada, X, y, "AdaBoost (Stump)")
        self.models["AdaBoost (Stump)"] = ada

    def _train_adaboost_deep(self, X, y):
        tree = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
        ada = AdaBoostClassifier(
            estimator=tree,
            n_estimators=100,
            learning_rate=0.1,
            random_state=self.random_state,
        )
        ada.fit(X, y)
        self._cv_score(ada, X, y, "AdaBoost (Deep Tree, depth=3)")
        self.models["AdaBoost (Deep Tree)"] = ada

    def _grid_search(self, X, y):
        print("\n[Model] Running GridSearchCV on AdaBoost (Stump)…")
        param_grid = {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
        }
        stump = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
        base  = AdaBoostClassifier(
            estimator=stump,
            random_state=self.random_state,
        )
        gs = GridSearchCV(base, param_grid, cv=5,
                          scoring="accuracy", n_jobs=-1, verbose=0)
        gs.fit(X, y)
        self.grid_search_result = gs
        print(f"  Best params : {gs.best_params_}")
        print(f"  Best CV Acc : {gs.best_score_:.4f}")
        self.models["AdaBoost (Tuned)"] = gs.best_estimator_
        self.cv_scores["AdaBoost (Tuned)"] = {
            "mean": gs.best_score_, "std": 0.0, "scores": []
        }

    def _select_best(self):
        best_name = max(
            self.cv_scores,
            key=lambda k: self.cv_scores[k]["mean"]
        )
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
