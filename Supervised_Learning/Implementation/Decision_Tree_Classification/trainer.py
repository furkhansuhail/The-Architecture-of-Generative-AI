"""
Module 4: Model Trainer
========================
Trains a Decision Tree classifier with optional GridSearchCV tuning.
Also supports comparing against a pruned (cost-complexity) variant.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, StratifiedKFold
)
import warnings; warnings.filterwarnings("ignore")


# ── Default hyper-parameter grid ─────────────────────────────────────────────
DEFAULT_PARAM_GRID = {
    "criterion":        ["gini", "entropy"],
    "max_depth":        [3, 4, 5, 6, 7, None],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     [None, "sqrt", "log2"],
}


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    param_grid: dict = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Trains and (optionally) tunes a Decision Tree.

    Returns
    -------
    dict with keys:
        model          – best fitted estimator
        best_params    – parameter dict of best model
        cv_results     – full GridSearchCV cv_results_ (or None)
        cv_scores      – cross-validation accuracy scores on train set
        base_model     – simple un-tuned tree (for comparison)
    """
    print("\n[Trainer] Training Decision Tree …")

    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                            random_state=random_state)

    # ── Baseline (un-tuned) model ─────────────────────────────────────────
    base_model = DecisionTreeClassifier(random_state=random_state)
    base_model.fit(X_train, y_train)
    base_cv = cross_val_score(base_model, X_train, y_train,
                              cv=kfold, scoring="accuracy")
    print(f"[Trainer] Baseline CV Accuracy : "
          f"{base_cv.mean():.4f} ± {base_cv.std():.4f}")

    if not tune:
        return dict(model=base_model, best_params=base_model.get_params(),
                    cv_results=None, cv_scores=base_cv, base_model=base_model)

    # ── Hyper-parameter search ────────────────────────────────────────────
    grid = param_grid or DEFAULT_PARAM_GRID
    print(f"[Trainer] Running GridSearchCV "
          f"({np.prod([len(v) for v in grid.values()])} combinations, "
          f"{cv_folds}-fold CV) …")

    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid=grid,
        cv=kfold,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    best_cv    = cross_val_score(best_model, X_train, y_train,
                                 cv=kfold, scoring="accuracy")

    print(f"[Trainer] Best CV Accuracy    : "
          f"{best_cv.mean():.4f} ± {best_cv.std():.4f}")
    print(f"[Trainer] Best Parameters     : {gs.best_params_}")

    return dict(
        model=best_model,
        best_params=gs.best_params_,
        cv_results=pd.DataFrame(gs.cv_results_),
        cv_scores=best_cv,
        base_model=base_model,
    )


def cost_complexity_prune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> DecisionTreeClassifier:
    """
    Uses cost-complexity pruning path to select optimal alpha.
    Returns the best-performing pruned tree.
    """
    print("\n[Trainer] Running cost-complexity pruning …")
    path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(
        X_train, y_train
    )
    alphas = path.ccp_alphas[:-1]   # remove trivial last node

    best_score, best_tree = 0, None
    for alpha in alphas:
        clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if score > best_score:
            best_score, best_tree = score, clf

    print(f"[Trainer] Pruned tree test accuracy : {best_score:.4f}  "
          f"(depth={best_tree.get_depth()}, leaves={best_tree.get_n_leaves()})")
    return best_tree
