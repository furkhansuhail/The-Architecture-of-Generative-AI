"""
================================================================================
  LOGISTIC REGRESSION MODULE
  A complete, plug-and-play module for Binary & Multiclass Classification
================================================================================

WHAT IS LOGISTIC REGRESSION?
-----------------------------
Despite its name, Logistic Regression is a CLASSIFICATION algorithm, not
regression. It models the PROBABILITY that a sample belongs to a class.

HOW IT DIFFERS FROM LINEAR REGRESSION:
  Linear Regression  → predicts a continuous value  (e.g., house price)
  Logistic Regression → predicts a class probability  (e.g., spam? yes/no)

THE CORE IDEA — THE SIGMOID FUNCTION:
  Instead of predicting y directly, we predict P(y=1|X):

      P(y=1|X) = σ(z)  where  z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

  The sigmoid function σ(z) = 1 / (1 + e^(-z)) squashes any real number
  into the range (0, 1), giving us a valid probability.

  z → -∞  ⟹  σ(z) → 0  (confident: class 0)
  z =  0  ⟹  σ(z) = 0.5 (uncertain: decision boundary)
  z → +∞  ⟹  σ(z) → 1  (confident: class 1)

CLASSIFICATION DECISION:
  if P(y=1|X) ≥ threshold (default 0.5) → predict class 1
  if P(y=1|X)  < threshold              → predict class 0

TRAINING — HOW COEFFICIENTS ARE LEARNED:
  Unlike Linear Regression (which uses Least Squares), Logistic Regression
  is trained by MAXIMIZING the Log-Likelihood via gradient-based optimization:

      Log-Loss = -1/n · Σ [ yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ) ]

  A lower log-loss = better model. The solver iteratively adjusts β to
  minimize this loss using algorithms like:
    - 'lbfgs'   : quasi-Newton method, good default for most tasks
    - 'saga'    : stochastic gradient, handles L1 + large datasets
    - 'liblinear': coordinate descent, good for small datasets + L1

REGULARIZATION (Preventing Overfitting):
  Just like Elastic Net, Logistic Regression supports regularization:
    - L2 (default) : shrinks coefficients, handles correlated features
    - L1           : zeros out features (feature selection)
    - ElasticNet   : combines both (requires solver='saga')

  Note: sklearn's `C` parameter = 1/λ  (inverse of regularization strength)
        Larger C → less regularization | Smaller C → stronger regularization

BINARY vs MULTICLASS:
  Binary     : one sigmoid output, threshold at 0.5
  Multiclass : sklearn auto-selects One-vs-Rest (OvR) or Softmax (multinomial)

HOW TO USE THIS MODULE
----------------------
  from logistic_regression_module import LogisticModel

  model = LogisticModel()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  probabilities = model.predict_proba(X_test)
  model.summary()

CONTENTS
--------
  1. LogisticModel          — main class with fit/predict/summary
  2. LogisticTuner          — hyperparameter search (C, penalty, solver)
  3. LogisticEvaluator      — accuracy, precision, recall, F1, AUC, confusion matrix
  4. LogisticVisualizer     — decision boundary, ROC curve, confusion matrix, etc.
  5. run_full_pipeline()    — one-call: tune → fit → evaluate → visualize

================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, log_loss, average_precision_score,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 ── CORE MODEL CLASS
#  Wraps sklearn LogisticRegression with auto-scaling, label handling,
#  probability outputs, and a human-readable classification summary.
# ══════════════════════════════════════════════════════════════════════════════

class LogisticModel:
    """
    A production-ready Logistic Regression classifier.

    CLASSIFICATION WORKFLOW:
    ─────────────────────────
    1. Features (X) enter the model
    2. Linear combination computed: z = β₀ + β₁x₁ + ... + βₙxₙ
    3. Sigmoid applied: P(y=1) = 1 / (1 + e^(-z))
    4. Threshold applied: class = 1 if P ≥ 0.5 else 0

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength. Smaller = stronger regularization.
        Think of it as: C = 1 / λ  (opposite of Elastic Net's alpha)

    penalty : str, default='l2'
        Regularization type:
          'l2'          → Ridge-style (shrink all, keep all features)
          'l1'          → Lasso-style (zeros out features, requires saga/liblinear)
          'elasticnet'  → Combined L1+L2 (requires solver='saga')
          None          → No regularization

    solver : str, default='lbfgs'
        Optimization algorithm:
          'lbfgs'      → Best default for L2 + small/medium datasets
          'saga'       → Needed for L1, ElasticNet, large datasets
          'liblinear'  → Good for small datasets + L1/L2
          'newton-cg'  → Good for L2 multiclass

    l1_ratio : float, default=0.5
        Only used when penalty='elasticnet'. Mix of L1 vs L2.
        0.0 = pure L2, 1.0 = pure L1.

    multi_class : str, default='auto'
        'auto'        → sklearn decides (OvR for binary, softmax otherwise)
        'ovr'         → One-vs-Rest: trains one classifier per class
        'multinomial' → Softmax: models all classes jointly

    threshold : float, default=0.5
        Decision threshold for binary classification.
        Lower threshold → more positive predictions (higher recall, lower precision)
        Higher threshold → fewer positive predictions (higher precision, lower recall)

    scale_features : bool, default=True
        Standardize features before fitting. Strongly recommended.

    Example
    -------
    >>> model = LogisticModel(C=0.1, penalty='l2')  # doctest: +SKIP
    >>> model.fit(X_train, y_train)                  # doctest: +SKIP
    >>> preds = model.predict(X_test)                # doctest: +SKIP
    >>> model.summary()                              # doctest: +SKIP
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        l1_ratio: float = 0.5,
        multi_class: str = "auto",
        threshold: float = 0.5,
        scale_features: bool = True,
        max_iter: int = 10_000,
        random_state: int = 42,
    ):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.multi_class = multi_class
        self.threshold = threshold
        self.scale_features = scale_features
        self.max_iter = max_iter
        self.random_state = random_state

        # Auto-select solver when penalty requires it
        if penalty == "l1" and solver not in ("saga", "liblinear"):
            self.solver = "saga"
        if penalty == "elasticnet":
            self.solver = "saga"

        # Build the core sklearn model
        kw = dict(
            C=C, penalty=penalty, solver=self.solver,
            max_iter=max_iter, random_state=random_state,
        )
        if penalty == "elasticnet":
            kw["l1_ratio"] = l1_ratio

        self._model = LogisticRegression(**kw)
        self._scaler = StandardScaler() if scale_features else None
        self._label_encoder = LabelEncoder()

        self.feature_names_ = None
        self.classes_ = None
        self.is_binary_ = None
        self.is_fitted_ = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        """
        Train the logistic regression classifier.

        WHAT HAPPENS STEP-BY-STEP:
        ───────────────────────────
        1. Convert inputs to numpy arrays, encode string labels if needed.
        2. Standardize features (mean=0, std=1) so regularization is fair.
        3. Initialize coefficients β to zero (or random).
        4. Compute log-loss for current β.
        5. Use solver (e.g., lbfgs) to compute gradient of loss w.r.t. β.
        6. Update β in the direction that decreases loss.
        7. Repeat steps 4-6 until convergence (loss stops decreasing).

        The result: β coefficients that define the decision boundary.

        Parameters
        ----------
        X            : array-like (n_samples, n_features)
        y            : array-like (n_samples,) — class labels
        feature_names: optional list of column names
        """
        X = np.array(X)
        y = np.array(y).ravel()

        # Store feature names
        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"x{i}" for i in range(X.shape[1])]
        )

        # Encode labels
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.is_binary_ = len(self.classes_) == 2

        # Scale
        X_ready = self._scaler.fit_transform(X) if self.scale_features else X

        # Fit
        self._model.fit(X_ready, y_enc)
        self.is_fitted_ = True

        mode = "Binary" if self.is_binary_ else f"Multiclass ({len(self.classes_)} classes)"
        print(
            f"[LogisticModel] Fitted — {mode} | "
            f"{X.shape[0]} samples, {X.shape[1]} features\n"
            f"  C={self.C}, penalty='{self.penalty}', solver='{self.solver}'\n"
            f"  Classes: {list(self.classes_)}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict class labels.

        For binary classification, applies the custom threshold:
          class = 1  if P(y=1|X) ≥ threshold
          class = 0  otherwise

        For multiclass, returns the class with the highest probability.

        Returns
        -------
        y_pred : array of original class labels (decoded from integers)
        """
        self._check_fitted()
        X_ready = self._scaler.transform(np.array(X)) if self.scale_features else np.array(X)

        if self.is_binary_ and self.threshold != 0.5:
            # Apply custom threshold on positive-class probabilities
            proba = self._model.predict_proba(X_ready)[:, 1]
            y_enc = (proba >= self.threshold).astype(int)
        else:
            y_enc = self._model.predict(X_ready)

        return self._label_encoder.inverse_transform(y_enc)

    # ── predict_proba ─────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Return class probabilities for each sample.

        WHY PROBABILITIES MATTER:
        ──────────────────────────
        Hard predictions (class 0 or 1) discard useful information.
        Probabilities let you:
          - Rank predictions by confidence
          - Adjust threshold for business needs (e.g., high-recall fraud detection)
          - Plot ROC/PR curves
          - Calibrate the model

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
                Each row sums to 1.0
        """
        self._check_fitted()
        X_ready = self._scaler.transform(np.array(X)) if self.scale_features else np.array(X)
        return self._model.predict_proba(X_ready)

    # ── predict_proba_positive ────────────────────────────────────────────────

    def predict_proba_positive(self, X):
        """
        Shortcut: return only P(positive class) for binary problems.
        Returns ndarray of shape (n_samples,)
        """
        if not self.is_binary_:
            raise ValueError("predict_proba_positive is only for binary classification.")
        return self.predict_proba(X)[:, 1]

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print a detailed, structured summary of the fitted model including:
          - Task type (binary / multiclass)
          - Hyperparameters
          - Coefficients per class with odds ratios
          - Interpretation guide for odds ratios
        """
        self._check_fitted()
        sep = "=" * 65

        print(f"\n{sep}")
        print("  LOGISTIC REGRESSION MODEL SUMMARY")
        print(sep)
        print(f"  Task        : {'Binary Classification' if self.is_binary_ else 'Multiclass Classification'}")
        print(f"  Classes     : {list(self.classes_)}")
        print(f"  C (1/λ)     : {self.C}  "
              f"({'Weaker' if self.C > 1 else 'Stronger'} regularization)")
        print(f"  Penalty     : {self.penalty}")
        print(f"  Solver      : {self.solver}")
        if self.penalty == "elasticnet":
            print(f"  l1_ratio    : {self.l1_ratio}")
        print(f"  Threshold   : {self.threshold}")

        # Coefficients
        coefs = self._model.coef_  # shape: (n_classes, n_features) or (1, n_features)
        intercepts = self._model.intercept_

        for i, cls in enumerate(self.classes_):
            if self.is_binary_ and i == 0:
                continue    # sklearn only stores one coef vector for binary
            c = coefs[0] if self.is_binary_ else coefs[i]
            intc = intercepts[0] if self.is_binary_ else intercepts[i]

            df = pd.DataFrame({
                "Feature":    self.feature_names_,
                "Coefficient": c,
                "Odds Ratio": np.exp(c),          # e^β: how much odds multiply per unit increase
                "Status":     ["ACTIVE" if v != 0 else "zeroed" for v in c],
            }).sort_values("Coefficient", key=abs, ascending=False)

            lbl = f"Class: {cls}" if not self.is_binary_ else "Positive Class"
            print(f"\n  {lbl}  |  Intercept: {intc:.4f}")
            print(f"  {'Feature':<22} {'Coefficient':>12}  {'Odds Ratio':>11}  Status")
            print(f"  {'-'*22} {'-'*12}  {'-'*11}  {'-'*8}")
            for _, row in df.iterrows():
                print(f"  {row['Feature']:<22} {row['Coefficient']:>12.5f}  "
                      f"{row['Odds Ratio']:>11.4f}  {row['Status']}")

        print(f"\n  ODDS RATIO GUIDE:")
        print(f"    > 1.0  → feature increases  probability of positive class")
        print(f"    = 1.0  → feature has no effect")
        print(f"    < 1.0  → feature decreases probability of positive class")
        print(sep + "\n")

    # ── get_coefficients ──────────────────────────────────────────────────────

    def get_coefficients(self, class_idx: int = 0) -> pd.DataFrame:
        """Return a DataFrame of coefficients and odds ratios for a given class."""
        self._check_fitted()
        coef = self._model.coef_[0] if self.is_binary_ else self._model.coef_[class_idx]
        return pd.DataFrame({
            "feature":     self.feature_names_,
            "coefficient": coef,
            "odds_ratio":  np.exp(coef),
        }).sort_values("coefficient", key=abs, ascending=False)

    # ── set_threshold ─────────────────────────────────────────────────────────

    def set_threshold(self, threshold: float):
        """
        Change the decision threshold after fitting.

        USE CASES:
          - Medical diagnosis: lower threshold → catch more positives (high recall)
          - Spam filter: higher threshold → fewer false positives (high precision)
          - Fraud detection: very low threshold → minimize missed fraud cases
        """
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold
        print(f"[LogisticModel] Threshold updated to {threshold}")
        return self

    # ── private ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 ── HYPERPARAMETER TUNER
#  Searches over C (regularization strength) and penalty type via
#  stratified cross-validation (stratified ensures class balance in each fold).
# ══════════════════════════════════════════════════════════════════════════════

class LogisticTuner:
    """
    Cross-validation based hyperparameter tuner for Logistic Regression.

    WHY TUNE C?
    ───────────
    C is the most critical hyperparameter:
      - Very small C (e.g., 0.001) → over-regularized → underfitting
      - Very large C (e.g., 1000)  → under-regularized → overfitting
      - The sweet spot balances bias vs variance on unseen data

    WHY STRATIFIED CV?
    ──────────────────
    In classification, class imbalance is common (e.g., 95% negative, 5% positive).
    Regular k-fold might put almost no positive samples in a fold.
    Stratified k-fold ensures each fold has the same class ratio as the full dataset.

    Example
    -------
    >>> tuner = LogisticTuner(cv=5)                    # doctest: +SKIP
    >>> best_C, best_penalty = tuner.fit(X_train, y_train)  # doctest: +SKIP
    >>> tuner.results_summary()                        # doctest: +SKIP
    """

    def __init__(self, cv: int = 5, scoring: str = "roc_auc", random_state: int = 42):
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.cv_results_ = None
        self._grid_search = None

    def fit(self, X, y, param_grid: dict = None):
        """
        Run stratified grid search over C and penalty.

        STEP-BY-STEP:
        ─────────────
        1. Define candidate C values (log-spaced from 0.001 to 100).
        2. For each (C, penalty) combination, run k-fold stratified CV.
        3. Score each fold using the specified metric (default: roc_auc).
        4. Select the combination with the highest mean CV score.

        Parameters
        ----------
        X          : feature matrix
        y          : target labels
        param_grid : custom dict for GridSearchCV (optional)

        Returns
        -------
        (best_C, best_penalty) : tuple
        """
        X = np.array(X)
        y = np.array(y).ravel()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if param_grid is None:
            param_grid = [
                {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
                 "penalty": ["l2"],
                 "solver": ["lbfgs"]},
                {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
                 "penalty": ["l1"],
                 "solver": ["saga"]},
            ]

        base_model = LogisticRegression(max_iter=10_000, random_state=self.random_state)

        cv_strat = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        # Use roc_auc_ovr for multiclass, roc_auc for binary
        n_classes = len(np.unique(y))
        scoring = "roc_auc_ovr" if n_classes > 2 else self.scoring

        print(f"[LogisticTuner] Grid search over {sum(len(p['C']) * len(p['penalty']) for p in param_grid)} "
              f"combinations with {self.cv}-fold stratified CV ...")

        self._grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv_strat, scoring=scoring,
            n_jobs=-1, refit=True,
        )
        self._grid_search.fit(X_scaled, y)

        self.best_params_ = self._grid_search.best_params_
        self.cv_results_  = pd.DataFrame(self._grid_search.cv_results_)

        print(f"[LogisticTuner] ✓ Best params: {self.best_params_}  |  "
              f"Best {scoring}: {self._grid_search.best_score_:.4f}")

        return self.best_params_.get("C"), self.best_params_.get("penalty")

    def results_summary(self):
        """Print a table of all tried combinations sorted by mean test score."""
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        print("\n" + "=" * 55)
        print("  TUNING RESULTS")
        print("=" * 55)
        print(f"  Best C       : {self.best_params_.get('C')}")
        print(f"  Best penalty : {self.best_params_.get('penalty')}")
        print(f"  Best solver  : {self.best_params_.get('solver')}")
        print(f"  Best score   : {self._grid_search.best_score_:.4f}")
        print("=" * 55 + "\n")

    def get_best_model(self) -> "LogisticModel":
        """Return a pre-configured LogisticModel with the best hyperparameters."""
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        return LogisticModel(
            C=self.best_params_.get("C", 1.0),
            penalty=self.best_params_.get("penalty", "l2"),
            solver=self.best_params_.get("solver", "lbfgs"),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 ── EVALUATOR
#  Classification requires different metrics than regression.
#  A single accuracy score is often misleading (especially with imbalanced classes).
# ══════════════════════════════════════════════════════════════════════════════

class LogisticEvaluator:
    """
    Comprehensive classification metrics for Logistic Regression.

    METRICS EXPLAINED:
    ──────────────────
    Accuracy     — % of all predictions correct. Misleading with imbalanced data.
                   E.g., if 95% are class 0, predicting 0 always = 95% accuracy.

    Precision    — Of all predicted positives, how many were truly positive?
                   High precision → fewer false alarms.
                   Precision = TP / (TP + FP)

    Recall       — Of all actual positives, how many did we catch?
    (Sensitivity)  High recall → fewer missed positives.
                   Recall = TP / (TP + FN)

    F1 Score     — Harmonic mean of Precision and Recall.
                   Balances both. Use when you care equally about both.
                   F1 = 2 · (Precision · Recall) / (Precision + Recall)

    ROC-AUC      — Area Under the ROC Curve.
                   Measures model's ability to distinguish classes at ALL thresholds.
                   0.5 = random, 1.0 = perfect. Threshold-independent.

    Log Loss     — How well-calibrated are the probability estimates?
                   Lower = better. Penalizes confident wrong predictions heavily.

    Confusion Matrix:
                   Predicted →     0           1
                   Actual ↓   ┌──────────┬──────────┐
                         0    │    TN    │    FP    │  ← False Positives (Type I error)
                         1    │    FN    │    TP    │  ← False Negatives (Type II error)
                              └──────────┴──────────┘

    Example
    -------
    >>> ev = LogisticEvaluator(model)          # doctest: +SKIP
    >>> ev.evaluate(X_test, y_test)            # doctest: +SKIP
    >>> ev.cross_validate(X, y, cv=5)          # doctest: +SKIP
    """

    def __init__(self, model: LogisticModel):
        self.model = model
        self.metrics_ = {}
        self._y_true = None
        self._y_pred = None
        self._y_proba = None

    def evaluate(self, X, y, label: str = "Test Set") -> dict:
        """
        Compute all classification metrics on a given (X, y) split.

        Parameters
        ----------
        X     : feature matrix
        y     : true labels
        label : display label

        Returns
        -------
        dict of metric → value
        """
        y = np.array(y).ravel()
        y_pred  = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        self._y_true  = y
        self._y_pred  = y_pred
        self._y_proba = y_proba

        avg = "weighted"

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y, y_pred, average=avg, zero_division=0)
        ll   = log_loss(y, y_proba)

        # AUC: binary uses positive class proba; multiclass uses all probas
        try:
            if self.model.is_binary_:
                auc = roc_auc_score(y, y_proba[:, 1])
            else:
                le = self.model._label_encoder
                auc = roc_auc_score(
                    le.transform(y), y_proba,
                    multi_class="ovr", average="weighted"
                )
        except Exception:
            auc = float("nan")

        self.metrics_ = dict(
            Accuracy=acc, Precision=prec, Recall=rec,
            F1=f1, ROC_AUC=auc, Log_Loss=ll
        )

        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  EVALUATION — {label}")
        print(sep)
        print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}% correct)")
        print(f"  Precision  : {prec:.4f}  (of predicted positives, {prec*100:.1f}% were correct)")
        print(f"  Recall     : {rec:.4f}  (of actual positives, {rec*100:.1f}% were caught)")
        print(f"  F1 Score   : {f1:.4f}  (harmonic mean of precision & recall)")
        print(f"  ROC-AUC    : {auc:.4f}  (1.0 = perfect, 0.5 = random)")
        print(f"  Log Loss   : {ll:.4f}  (lower = better calibrated probabilities)")
        report = classification_report(y, y_pred, zero_division=0)
        indented = "\n".join("    " + line for line in report.splitlines())
        print(f"\n  Classification Report:\n{indented}")
        print(sep + "\n")
        return self.metrics_

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Stratified k-fold cross-validation for robust performance estimates.

        WHY STRATIFIED FOR CLASSIFICATION:
        ────────────────────────────────────
        Each fold preserves the original class distribution.
        This prevents folds where one class is absent, which would
        give meaningless metrics.

        Returns
        -------
        dict with mean and std of accuracy, F1, and ROC-AUC
        """
        X = np.array(X)
        y = np.array(y).ravel()

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=self.model.C, penalty=self.model.penalty,
                solver=self.model.solver, max_iter=10_000,
            ))
        ])

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        f1_scores  = cross_val_score(pipe, X, y, cv=skf, scoring="f1_weighted")

        try:
            auc_scores = cross_val_score(pipe, X, y, cv=skf,
                                         scoring="roc_auc" if len(np.unique(y)) == 2
                                         else "roc_auc_ovr_weighted")
        except Exception:
            auc_scores = np.array([float("nan")] * cv)

        results = {
            "Accuracy_mean": acc_scores.mean(),  "Accuracy_std": acc_scores.std(),
            "F1_mean":       f1_scores.mean(),   "F1_std":       f1_scores.std(),
            "AUC_mean":      auc_scores.mean(),  "AUC_std":      auc_scores.std(),
        }

        print(f"\n{'='*55}")
        print(f"  {cv}-FOLD STRATIFIED CROSS-VALIDATION")
        print(f"{'='*55}")
        print(f"  Accuracy  mean ± std : {results['Accuracy_mean']:.4f} ± {results['Accuracy_std']:.4f}")
        print(f"  F1        mean ± std : {results['F1_mean']:.4f} ± {results['F1_std']:.4f}")
        print(f"  ROC-AUC   mean ± std : {results['AUC_mean']:.4f} ± {results['AUC_std']:.4f}")
        print(f"{'='*55}\n")
        return results

    def threshold_analysis(self, X, y, thresholds=None):
        """
        Show how Precision, Recall, and F1 change as the decision threshold varies.

        This is crucial for:
          - Choosing the right threshold for your business need
          - Understanding the precision-recall tradeoff
          - Finding the threshold that maximizes F1

        Returns
        -------
        pd.DataFrame of threshold → precision, recall, F1
        """
        if not self.model.is_binary_:
            print("[LogisticEvaluator] Threshold analysis is only for binary classification.")
            return None

        y = np.array(y).ravel()
        y_proba_pos = self.model.predict_proba_positive(X)

        if thresholds is None:
            thresholds = np.arange(0.05, 1.0, 0.05)

        rows = []
        for t in thresholds:
            y_pred_t = (y_proba_pos >= t).astype(int)
            y_pred_labels = self.model._label_encoder.inverse_transform(y_pred_t)
            rows.append({
                "Threshold": round(t, 2),
                "Precision": precision_score(y, y_pred_labels, zero_division=0, average="weighted"),
                "Recall":    recall_score(y, y_pred_labels, zero_division=0, average="weighted"),
                "F1":        f1_score(y, y_pred_labels, zero_division=0, average="weighted"),
            })

        df = pd.DataFrame(rows)
        best_row = df.loc[df["F1"].idxmax()]
        print(f"\n  Threshold with best F1: {best_row['Threshold']:.2f}  "
              f"(F1={best_row['F1']:.4f}, "
              f"Precision={best_row['Precision']:.4f}, "
              f"Recall={best_row['Recall']:.4f})")
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 ── VISUALIZER
#  Six classification-specific diagnostic plots
# ══════════════════════════════════════════════════════════════════════════════

class LogisticVisualizer:
    """
    Diagnostic plots tailored for logistic regression classification.

    Plots available:
    ────────────────
    1. plot_sigmoid         — the sigmoid function and how z maps to probability
    2. plot_coefficients    — coefficient/odds ratio bar chart
    3. plot_confusion_matrix— heatmap of TP/FP/TN/FN
    4. plot_roc_curve       — ROC curve with AUC (binary only)
    5. plot_precision_recall— Precision-Recall curve (binary only)
    6. plot_probability_dist— distribution of predicted probabilities per class
    7. plot_threshold_analysis — precision/recall/F1 vs threshold
    8. plot_all             — 2×3 dashboard of the most useful plots

    Example
    -------
    >>> viz = LogisticVisualizer(model, evaluator)  # doctest: +SKIP
    >>> viz.plot_all(X_test, y_test)                # doctest: +SKIP
    """

    def __init__(self, model: LogisticModel, evaluator: LogisticEvaluator = None):
        self.model = model
        self.evaluator = evaluator
        self._palette = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
                         "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]

    # ── 1. Sigmoid function ───────────────────────────────────────────────────

    def plot_sigmoid(self, ax=None):
        """
        Visualize the sigmoid function — the heart of logistic regression.

        WHAT IT SHOWS:
          - How the linear score z is converted to a probability
          - The decision boundary at z=0 (P=0.5)
          - How confident the model becomes as |z| grows
        """
        z = np.linspace(-8, 8, 300)
        sigmoid = 1 / (1 + np.exp(-z))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(z, sigmoid, color="#3498DB", linewidth=2.5, label="σ(z) = 1/(1+e⁻ᶻ)")
        ax.axhline(0.5, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.8,
                   label="Decision boundary (P=0.5)")
        ax.axvline(0, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.8)

        # Shade regions
        ax.fill_between(z, 0, sigmoid, where=(z < 0), alpha=0.08, color="#E74C3C",
                        label="Predicts class 0")
        ax.fill_between(z, sigmoid, 1, where=(z >= 0), alpha=0.08, color="#2ECC71",
                        label="Predicts class 1")

        # Annotate key points
        for zv, pv, lbl in [(-4, 0.018, "z=-4\nP≈0.02"), (0, 0.5, "z=0\nP=0.5"), (4, 0.982, "z=4\nP≈0.98")]:
            ax.annotate(lbl, xy=(zv, pv), xytext=(zv, pv + 0.18),
                        ha="center", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

        ax.set_xlabel("Linear Score  z = β₀ + β₁x₁ + ... + βₙxₙ")
        ax.set_ylabel("P(y=1 | X)")
        ax.set_title("The Sigmoid Function — How Scores Become Probabilities")
        ax.legend(fontsize=8, loc="center left")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-8, 8)
        ax.grid(True, alpha=0.3)

    # ── 2. Coefficients / Odds Ratios ─────────────────────────────────────────

    def plot_coefficients(self, class_idx: int = 0, ax=None):
        """
        Horizontal bar chart of coefficients with odds ratios annotated.

        ODDS RATIO INTERPRETATION:
          Odds Ratio = e^β
          OR = 2.0 → one unit increase in feature doubles the odds of positive class
          OR = 0.5 → one unit increase halves the odds
          OR = 1.0 → feature has no effect
        """
        self.model._check_fitted()
        df = self.model.get_coefficients(class_idx)

        colors = ["#E74C3C" if c < 0 else "#2ECC71" for c in df["coefficient"]]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))

        bars = ax.barh(df["feature"], df["coefficient"], color=colors, edgecolor="white", height=0.7)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        # Annotate odds ratios on bars
        for bar, or_val in zip(bars, df["odds_ratio"]):
            x = bar.get_width()
            ax.text(
                x + (0.03 if x >= 0 else -0.03),
                bar.get_y() + bar.get_height() / 2,
                f"OR={or_val:.2f}",
                va="center", ha="left" if x >= 0 else "right",
                fontsize=7.5, color="dimgray"
            )

        ax.set_xlabel("Coefficient Value  (β)")
        cls_lbl = list(self.model.classes_)[class_idx] if not self.model.is_binary_ else "Positive Class"
        ax.set_title(f"Logistic Regression Coefficients & Odds Ratios\n({cls_lbl})")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

    # ── 3. Confusion Matrix ───────────────────────────────────────────────────

    def plot_confusion_matrix(self, X, y, normalize: bool = False, ax=None):
        """
        Heatmap of the confusion matrix.

        READING THE MATRIX:
        ────────────────────
        Rows = actual classes    Columns = predicted classes

        For binary (2×2):
          Top-left  = TN  (correctly predicted negative)
          Top-right = FP  (predicted positive, actually negative) ← Type I error
          Bot-left  = FN  (predicted negative, actually positive) ← Type II error
          Bot-right = TP  (correctly predicted positive)

        normalize=True shows proportions instead of raw counts,
        making it easier to compare across classes of different sizes.
        """
        y = np.array(y).ravel()
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=self.model.classes_)

        if normalize:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt, title_suffix = ".2f", " (Normalized)"
        else:
            cm_display = cm
            fmt, title_suffix = "d", ""

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        thresh = cm_display.max() / 2.0
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                val = f"{cm_display[i, j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if cm_display[i, j] > thresh else "black",
                        fontsize=13, fontweight="bold")

        ax.set_xticks(range(len(self.model.classes_)))
        ax.set_yticks(range(len(self.model.classes_)))
        ax.set_xticklabels(self.model.classes_, fontsize=9)
        ax.set_yticklabels(self.model.classes_, fontsize=9)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix{title_suffix}")

        # Add TP/FP/TN/FN labels for binary
        if self.model.is_binary_:
            labels = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i - 0.35, labels[i][j],
                            ha="center", va="center", fontsize=8,
                            color="grey", style="italic")

    # ── 4. ROC Curve ──────────────────────────────────────────────────────────

    def plot_roc_curve(self, X, y, ax=None):
        """
        ROC (Receiver Operating Characteristic) Curve for binary classification.

        WHAT IT SHOWS:
        ───────────────
        X-axis: False Positive Rate (FPR) = FP / (FP + TN)  [cost of false alarms]
        Y-axis: True Positive Rate (TPR)  = TP / (TP + FN)  [benefit of true detections]

        The curve shows ALL possible thresholds simultaneously.

        AUC (Area Under Curve):
          = 0.5  → random guessing (diagonal line)
          = 1.0  → perfect classifier (top-left corner)
          = 0.7  → good; = 0.8 → very good; = 0.9+ → excellent

        The AUC is THRESHOLD-INDEPENDENT — it measures the model's
        inherent discriminative ability regardless of where you set the cutoff.
        """
        if not self.model.is_binary_:
            print("[LogisticVisualizer] ROC curve is for binary classification only.")
            return

        y = np.array(y).ravel()
        y_enc = self.model._label_encoder.transform(y)
        y_proba_pos = self.model.predict_proba_positive(X)

        fpr, tpr, thresholds = roc_curve(y_enc, y_proba_pos)
        auc = roc_auc_score(y_enc, y_proba_pos)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(fpr, tpr, color="#3498DB", linewidth=2.5,
                label=f"Logistic Regression  (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
                linewidth=1.2, label="Random Classifier (AUC = 0.50)")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#3498DB")

        # Mark current threshold
        cur_thresh = self.model.threshold
        idx = np.argmin(np.abs(thresholds - cur_thresh))
        ax.scatter(fpr[idx], tpr[idx], color="#E74C3C", zorder=5, s=80,
                   label=f"Current threshold ({cur_thresh})")

        ax.set_xlabel("False Positive Rate  (FPR = FP / N)")
        ax.set_ylabel("True Positive Rate  (TPR = TP / P)")
        ax.set_title("ROC Curve")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    # ── 5. Precision-Recall Curve ─────────────────────────────────────────────

    def plot_precision_recall(self, X, y, ax=None):
        """
        Precision-Recall Curve for binary classification.

        WHEN TO USE THIS OVER ROC:
        ───────────────────────────
        With imbalanced datasets (e.g., 99% negative), the ROC curve can
        look optimistic because TN dominates FPR. The PR curve focuses
        only on the POSITIVE class and reveals true performance.

        A good model should have high precision AND high recall simultaneously.
        The area under the PR curve (AP score) summarizes this tradeoff.

        Perfect PR curve: goes through top-right corner (P=1, R=1).
        Baseline (random): horizontal line at prevalence = P/(P+N).
        """
        if not self.model.is_binary_:
            print("[LogisticVisualizer] PR curve is for binary classification only.")
            return

        y = np.array(y).ravel()
        y_enc = self.model._label_encoder.transform(y)
        y_proba_pos = self.model.predict_proba_positive(X)

        precision, recall, thresholds = precision_recall_curve(y_enc, y_proba_pos)
        ap = average_precision_score(y_enc, y_proba_pos)
        baseline = y_enc.mean()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(recall, precision, color="#9B59B6", linewidth=2.5,
                label=f"Logistic Regression  (AP = {ap:.4f})")
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Random Baseline (AP = {baseline:.4f})")
        ax.fill_between(recall, precision, alpha=0.1, color="#9B59B6")

        ax.set_xlabel("Recall  (= Sensitivity = TP / (TP + FN))")
        ax.set_ylabel("Precision  (= TP / (TP + FP))")
        ax.set_title("Precision-Recall Curve")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    # ── 6. Probability Distribution ───────────────────────────────────────────

    def plot_probability_dist(self, X, y, ax=None):
        """
        Histogram of predicted P(positive) separated by true class.

        WHAT TO LOOK FOR:
          - Class 0 distribution peaks near 0 → model is confident about negatives
          - Class 1 distribution peaks near 1 → model is confident about positives
          - Large overlap → model struggles to separate the classes
          - Well-separated distributions → strong discriminating power
        """
        y = np.array(y).ravel()
        y_proba = self.model.predict_proba(X)

        classes = self.model.classes_

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        for i, cls in enumerate(classes):
            mask = y == cls
            proba_col = min(i, y_proba.shape[1] - 1)
            ax.hist(
                y_proba[mask, proba_col if not self.model.is_binary_ else 1],
                bins=30, alpha=0.6,
                color=self._palette[i % len(self._palette)],
                label=f"True class: {cls}", edgecolor="white"
            )

        if self.model.is_binary_:
            ax.axvline(self.model.threshold, color="black", linestyle="--",
                       linewidth=1.5, label=f"Threshold = {self.model.threshold}")

        ax.set_xlabel("Predicted Probability of Positive Class")
        ax.set_ylabel("Count")
        ax.set_title("Predicted Probability Distribution by True Class")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 7. Threshold Analysis ─────────────────────────────────────────────────

    def plot_threshold_analysis(self, X, y, ax=None):
        """
        Line chart showing Precision, Recall, F1 vs decision threshold.

        USE THIS TO:
          - Find the threshold that maximizes F1 score
          - Choose a threshold based on your use case:
              * High recall needed  → pick low threshold (catch all positives)
              * High precision needed → pick high threshold (only confident positives)
        """
        if not self.model.is_binary_:
            return

        ev = LogisticEvaluator(self.model)
        df = ev.threshold_analysis(X, y)
        if df is None:
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(df["Threshold"], df["Precision"], color="#E74C3C",
                linewidth=2, label="Precision", marker="o", markersize=3)
        ax.plot(df["Threshold"], df["Recall"],    color="#2ECC71",
                linewidth=2, label="Recall",    marker="o", markersize=3)
        ax.plot(df["Threshold"], df["F1"],        color="#3498DB",
                linewidth=2.5, label="F1 Score", marker="o", markersize=3)

        best_row = df.loc[df["F1"].idxmax()]
        ax.axvline(best_row["Threshold"], color="#3498DB", linestyle="--",
                   linewidth=1.2, label=f"Best F1 threshold ({best_row['Threshold']:.2f})")
        ax.axvline(self.model.threshold, color="black", linestyle=":",
                   linewidth=1.2, label=f"Current threshold ({self.model.threshold})")

        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1  vs  Decision Threshold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    # ── 8. Full Dashboard ─────────────────────────────────────────────────────

    def plot_all(self, X, y, save_path: str = None):
        """
        Render a comprehensive 2×3 diagnostic dashboard.

        Layout:
          ┌──────────────┬──────────────┬──────────────┐
          │   Sigmoid    │ Coefficients │  Confusion   │
          │   Function   │  & Odds Ratio│   Matrix     │
          ├──────────────┼──────────────┼──────────────┤
          │  ROC Curve   │  PR Curve    │  Threshold   │
          │              │              │  Analysis    │
          └──────────────┴──────────────┴──────────────┘
        """
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(
            f"Logistic Regression  —  Classification Diagnostic Dashboard\n"
            f"C={self.model.C}  |  penalty='{self.model.penalty}'  |  "
            f"threshold={self.model.threshold}  |  "
            f"task={'Binary' if self.model.is_binary_ else 'Multiclass'}",
            fontsize=13, fontweight="bold"
        )

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])   # Sigmoid
        ax2 = fig.add_subplot(gs[0, 1])   # Coefficients
        ax3 = fig.add_subplot(gs[0, 2])   # Confusion Matrix
        ax4 = fig.add_subplot(gs[1, 0])   # ROC Curve
        ax5 = fig.add_subplot(gs[1, 1])   # Precision-Recall
        ax6 = fig.add_subplot(gs[1, 2])   # Threshold Analysis

        self.plot_sigmoid(ax=ax1)
        self.plot_coefficients(ax=ax2)
        self.plot_confusion_matrix(X, y, ax=ax3)
        self.plot_roc_curve(X, y, ax=ax4)
        self.plot_precision_recall(X, y, ax=ax5)
        self.plot_threshold_analysis(X, y, ax=ax6)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[LogisticVisualizer] Plot saved to: {save_path}")

        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 ── FULL PIPELINE
#  One function call: split → tune → fit → evaluate → visualize
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X,
    y,
    feature_names=None,
    test_size: float = 0.2,
    auto_tune: bool = True,
    C: float = 1.0,
    penalty: str = "l2",
    threshold: float = 0.5,
    cv: int = 5,
    plot: bool = True,
    save_plot: str = None,
    random_state: int = 42,
) -> dict:
    """
    End-to-end Logistic Regression classification pipeline.

    PIPELINE STEPS:
    ───────────────
    1.  Split data with stratification (preserves class ratios in train/test)
    2.  (Optional) Auto-tune C and penalty via stratified grid search
    3.  Fit the model on the training set
    4.  Print full model summary (coefficients + odds ratios)
    5.  Evaluate on train AND test sets (accuracy, F1, AUC, log-loss)
    6.  Run stratified k-fold CV for robust performance estimate
    7.  Run threshold analysis to find optimal decision boundary
    8.  (Optional) Generate 6-panel diagnostic dashboard

    Parameters
    ----------
    X            : feature matrix
    y            : class labels
    feature_names: column names (auto-detected from DataFrame)
    test_size    : fraction held out for testing
    auto_tune    : search for best C and penalty via GridSearchCV
    C            : regularization inverse (used if auto_tune=False)
    penalty      : 'l2', 'l1', or 'elasticnet' (used if auto_tune=False)
    threshold    : decision boundary (default 0.5)
    cv           : number of CV folds
    plot         : display diagnostic plots
    save_plot    : path to save the plot image
    random_state : reproducibility seed

    Returns
    -------
    dict: model, evaluator, visualizer, train_metrics, test_metrics, cv_metrics
    """
    print("\n" + "█" * 62)
    print("  LOGISTIC REGRESSION — FULL CLASSIFICATION PIPELINE")
    print("█" * 62)

    # ── 1. Prepare inputs
    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X = np.array(X)
    y = np.array(y).ravel()
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    n_classes = len(np.unique(y))
    print(f"\n[Pipeline] Dataset — {X.shape[0]} samples, {X.shape[1]} features, "
          f"{n_classes} classes: {np.unique(y)}")

    # ── 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        stratify=y,                    # preserve class ratios in both splits
        random_state=random_state
    )
    print(f"[Pipeline] Stratified split — Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Tune or use provided hyperparameters
    if auto_tune:
        print("\n[Pipeline] Step 1/4 — Hyperparameter Tuning ...")
        tuner = LogisticTuner(cv=cv, random_state=random_state)
        best_C, best_penalty = tuner.fit(X_train, y_train)
        tuner.results_summary()
        C, penalty = best_C, best_penalty
    else:
        print(f"\n[Pipeline] Step 1/4 — Using provided C={C}, penalty='{penalty}'")
        if penalty == "elasticnet":
            print("  → solver auto-set to 'saga' for elasticnet")

    # ── 4. Fit
    print("\n[Pipeline] Step 2/4 — Fitting model ...")
    model = LogisticModel(C=C, penalty=penalty, threshold=threshold)
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    # ── 5. Evaluate
    print("\n[Pipeline] Step 3/4 — Evaluating ...")
    evaluator = LogisticEvaluator(model)
    train_metrics = evaluator.evaluate(X_train, y_train, label="Train Set")
    test_metrics  = evaluator.evaluate(X_test,  y_test,  label="Test Set")
    cv_metrics    = evaluator.cross_validate(X, y, cv=cv)

    if model.is_binary_:
        print("\n[Pipeline] Threshold Analysis ...")
        evaluator.threshold_analysis(X_test, y_test)

    # ── 6. Plots
    visualizer = LogisticVisualizer(model, evaluator)
    if plot:
        print("\n[Pipeline] Step 4/4 — Generating diagnostic plots ...")
        visualizer.plot_all(X_test, y_test, save_path=save_plot)

    print("\n" + "█" * 62)
    print("  PIPELINE COMPLETE")
    print("█" * 62 + "\n")

    return {
        "model":         model,
        "evaluator":     evaluator,
        "visualizer":    visualizer,
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "cv_metrics":    cv_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO — runs when you execute this file directly
#  python logistic_regression_module.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  LOGISTIC REGRESSION MODULE — DEMO")
    print("  Using a synthetic binary classification dataset")
    print("=" * 62 + "\n")

    # ── Generate synthetic classification data
    # Two informative clusters, some noise, slight class imbalance
    from sklearn.datasets import make_classification

    X_demo, y_demo = make_classification(
        n_samples     = 800,
        n_features    = 15,
        n_informative = 8,
        n_redundant   = 3,
        n_classes     = 2,
        weights       = [0.6, 0.4],   # slight imbalance (60% class 0)
        flip_y        = 0.03,         # 3% label noise
        random_state  = 42,
    )
    y_demo = np.where(y_demo == 1, "Positive", "Negative")
    feature_names_demo = [f"feature_{i:02d}" for i in range(15)]

    # ── Run the full pipeline
    results = run_full_pipeline(
        X             = X_demo,
        y             = y_demo,
        feature_names = feature_names_demo,
        test_size     = 0.2,
        auto_tune     = True,
        cv            = 5,
        threshold     = 0.5,
        plot          = True,
        save_plot     = "logistic_regression_diagnostics.png",
    )

    # ── Access specific results
    model = results["model"]
    print("\nTop 5 most influential features:")
    print(model.get_coefficients().head(5).to_string(index=False))

    print("\nYou can adjust the threshold after fitting:")
    print("  model.set_threshold(0.35)   # Increase recall (catch more positives)")
    print("  model.set_threshold(0.65)   # Increase precision (fewer false alarms)")