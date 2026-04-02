"""
================================================================================
  K-NEAREST NEIGHBORS REGRESSION MODULE
  Built from your original KNNRegression implementation — expanded into a
  complete, plug-and-play module with detailed step-by-step explanations.
================================================================================

YOUR ORIGINAL CODE — WHAT IT DID:
───────────────────────────────────
  Your implementation had three core ideas that are exactly right:

    1. find_neighbors(k, X_train, new_point)
       → Computed Euclidean distance from a new point to every training point
       → Sorted and returned the K closest

    2. regressor(neighbor_arr)
       → Averaged the Y values of those K neighbors
       → THIS is what makes it regression (not classification)

    3. plot_predictions(new_points)
       → Visualized training data and predictions on a 3D scatter plot

  This module keeps every one of those ideas, adds feature scaling
  (the one missing piece), and wraps everything in a structured,
  reusable format consistent with your Elastic Net, Logistic Regression,
  and KNN Classification modules.

WHAT IS KNN REGRESSION?
────────────────────────
  KNN Regression predicts a CONTINUOUS value for a new point by:
    1. Finding the K most similar (nearest) training examples
    2. Averaging their known Y values → that average IS the prediction

  Contrast with KNN Classification, which takes a MAJORITY VOTE instead.

  The key aggregation step:
    Regression     → prediction = mean(Y of K neighbors)        ← your code
    Classification → prediction = majority_vote(labels of K neighbors)

THE MATH YOUR CODE IMPLEMENTS:
────────────────────────────────
  Step 1 — Euclidean Distance (your find_neighbors):
      d(X_new, X_train[i]) = √ Σ (X_new[j] - X_train[i][j])²

  Step 2 — Sort all distances, pick K smallest

  Step 3 — Predict (your regressor):
      ŷ = (1/K) · Σ Y_train[neighbor_i]

  Optional weighted version (closer neighbors count more):
      ŷ = Σ (Y_train[i] / d_i) / Σ (1 / d_i)

WHY FEATURE SCALING MATTERS:
──────────────────────────────
  Your original code used raw feature values in distance computation.
  This works when features are on similar scales (as in make_regression).
  In real data, a feature with range [0, 10000] would completely drown
  out one with range [0, 1] — making distance meaningless.
  This module adds StandardScaler to fix that.

HOW TO USE THIS MODULE:
───────────────────────
  from knn_regression_module import KNNRegressionModel

  model = KNNRegressionModel(n_neighbors=3)
  model.fit(X_train, Y_train)
  predictions = model.predict(new_points)
  model.summary()

CONTENTS:
─────────
  1. KNNRegressionModel  — your original logic, structured + scaled
  2. KNNRegressionTuner  — optimal K search via cross-validation + elbow
  3. KNNRegressionEvaluator — MSE, RMSE, MAE, R² with detailed explanation
  4. KNNRegressionVisualizer— 3D scatter, K-elbow, residuals, distance dist
  5. run_full_pipeline()  — one-call: tune → fit → evaluate → visualize

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
from mpl_toolkits.mplot3d import Axes3D                     # noqa: F401

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 ── CORE MODEL CLASS
#  This is a direct evolution of your original KNNRegression class.
#
#  YOUR CODE:                        THIS MODULE:
#  ─────────────────────────────     ─────────────────────────────────────────
#  find_neighbors()  ─────────────→  built into .predict() via sklearn + scaler
#  regressor()       ─────────────→  .predict() returns np.mean of K neighbors
#  ImportData()      ─────────────→  .fit(X, y) — separate from data loading
#  plot_predictions()─────────────→  KNNRegressionVisualizer.plot_3d_predictions
# ══════════════════════════════════════════════════════════════════════════════

class KNNRegressionModel:
    """
    K-Nearest Neighbors Regression — a structured, scalable version of
    your original KNNRegression implementation.

    PREDICTION LOGIC (directly from your regressor() method):
    ──────────────────────────────────────────────────────────
    For a new point X_new:

      1. Compute Euclidean distance to every training point:
             d = √ Σ (X_new[j] - X_train[i][j])²
             (your find_neighbors does this manually — we use sklearn
              which computes the same thing, just much faster)

      2. Sort distances, select the K smallest (nearest neighbors)

      3. Average their Y values — this IS your regressor() method:
             ŷ = mean( Y_train[neighbor_1], ..., Y_train[neighbor_K] )

      4. (Optional) Weighted average — closer neighbors contribute more:
             ŷ = Σ (Y_i / d_i) / Σ (1 / d_i)

    Parameters
    ----------
    n_neighbors : int, default=3
        Your original code used K=3. This is the most important
        hyperparameter — see KNNRegressionTuner for how to find the best K.

    weights : str, default='uniform'
        'uniform'  → your original regressor() — all K neighbors count equally
        'distance' → closer neighbors are weighted more (1/distance weighting)
                     Generally gives smoother predictions.

    metric : str, default='minkowski'
        Distance function. With p=2, this IS your original Euclidean formula:
            d = √ Σ (X_new[j] - X_train[i][j])²

    p : int, default=2
        Power for Minkowski. p=2 → Euclidean (your original), p=1 → Manhattan.

    scale_features : bool, default=True
        The one addition beyond your original code.
        Standardizes features to mean=0, std=1 before computing distances.
        Your original code skipped this — it worked because make_regression
        produces features on similar scales. Always enable for real data.

    Example
    -------
    >>> model = KNNRegressionModel(n_neighbors=3)   # doctest: +SKIP
    >>> model.fit(X_train, Y_train)                 # doctest: +SKIP
    >>> preds = model.predict(new_points)           # doctest: +SKIP
    >>> model.summary()                             # doctest: +SKIP
    """

    def __init__(
        self,
        n_neighbors: int = 3,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: int = 2,
        scale_features: bool = True,
        algorithm: str = "auto",
    ):
        self.n_neighbors   = n_neighbors
        self.weights       = weights
        self.metric        = metric
        self.p             = p
        self.scale_features = scale_features
        self.algorithm     = algorithm

        # sklearn's KNeighborsRegressor implements your find_neighbors()
        # + regressor() logic internally, with efficient indexing
        self._model  = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            algorithm=algorithm,
        )
        self._scaler = StandardScaler() if scale_features else None

        # Populated during fit()
        self.feature_names_  = None
        self.X_train_scaled_ = None
        self.y_train_        = None
        self.n_features_     = None
        self.is_fitted_      = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        """
        'Fit' the KNN Regression model — equivalent to your ImportData()
        method, but separated from data loading for reusability.

        WHAT HAPPENS HERE (expanding on your original design):
        ───────────────────────────────────────────────────────
        Step 1 — Validate and convert inputs to numpy arrays.
                 Your original code assumed numpy arrays from make_regression.
                 This handles DataFrames, lists, and any array-like input.

        Step 2 — Store feature names for reporting and plot axis labels.
                 Auto-detected from DataFrames, or defaulted to 'x0', 'x1', ...

        Step 3 — Standardize features (the addition beyond your code):
                 scaler.fit_transform(X) computes mean and std per feature
                 and transforms X so each feature has mean=0, std=1.
                 This prevents large-scale features from dominating distances.

        Step 4 — Store the scaled training data.
                 KNN is a LAZY LEARNER — there is no optimization loop.
                 The model IS the training data, stored in memory.
                 sklearn also builds a KD-tree or Ball-tree index here
                 for faster neighbor lookups at prediction time.

        Note: Your original find_neighbors() scanned all 300 points with
        a Python loop. sklearn's indexing does the same thing but much faster,
        especially for large datasets.

        Parameters
        ----------
        X            : array-like (n_samples, n_features) — your X_train
        y            : array-like (n_samples,)            — your Y_train
        feature_names: optional list of column names
        """
        # Step 1 — Convert to numpy
        X = np.array(X)
        y = np.array(y).ravel()

        # Step 2 — Feature names
        self.n_features_    = X.shape[1]
        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"X_{i+1}" for i in range(X.shape[1])]
        )

        # Step 3 — Scale features
        X_scaled = self._scaler.fit_transform(X) if self.scale_features else X

        # Step 4 — Store training data and build index (the "fit" for KNN)
        self._model.fit(X_scaled, y)
        self.X_train_scaled_ = X_scaled
        self.y_train_        = y
        self.is_fitted_      = True

        metric_label = (
            "Euclidean" if self.p == 2 and self.metric == "minkowski"
            else "Manhattan" if self.p == 1
            else self.metric
        )
        print(
            f"[KNNRegressionModel] Fitted\n"
            f"  Samples memorized : {X.shape[0]}  (your training set)\n"
            f"  Features          : {X.shape[1]}\n"
            f"  K (neighbors)     : {self.n_neighbors}  (your original K=3)\n"
            f"  Distance metric   : {metric_label}  (your Euclidean formula)\n"
            f"  Voting strategy   : {self.weights}  "
            f"({'equal weight — your original regressor()' if self.weights == 'uniform' else '1/distance weighting'})\n"
            f"  Feature scaling   : {'Yes (StandardScaler)' if self.scale_features else 'No (raw features — your original approach)'}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict continuous Y values for new points.

        THIS IS YOUR ORIGINAL FLOW, automated for any number of points:
        ─────────────────────────────────────────────────────────────────
        Your original code looped over new_points manually:
            for point in new_points:
                knn = self.find_neighbors(3, self.X_train, point)
                result = self.regressor(knn)

        This method does the same thing for all points at once:

        Step 1 — Scale new points using the SAME scaler fitted on training data.
                 Critical: we must not re-fit the scaler on new data —
                 that would use different mean/std and corrupt the distances.

        Step 2 — Find K nearest neighbors for each new point.
                 Equivalent to your find_neighbors() — computes Euclidean
                 distance from each new point to all training points,
                 sorts them, selects K closest.

        Step 3 — Average neighbor Y values (your regressor() logic):
                 ŷ = mean( Y_train[neighbor_1], ..., Y_train[neighbor_K] )
                 With weights='distance', uses weighted mean instead.

        Parameters
        ----------
        X : array-like (n_samples, n_features) — your new_points

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) — continuous float predictions
        """
        self._check_fitted()
        X = np.array(X)
        # Reshape single point: [-1, 1] → [[-1, 1]]
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self._scaler.transform(X) if self.scale_features else X
        return self._model.predict(X_scaled)

    # ── predict_single ────────────────────────────────────────────────────────

    def predict_single(self, point):
        """
        Predict a single point — mirrors your original loop body:
            knn = self.find_neighbors(3, self.X_train, point)
            result = self.regressor(knn)

        Parameters
        ----------
        point : 1D array of shape (n_features,)

        Returns
        -------
        float — the predicted Y value
        """
        return float(self.predict(np.array(point).reshape(1, -1))[0])

    # ── get_neighbors ─────────────────────────────────────────────────────────

    def get_neighbors(self, X, n_neighbors: int = None):
        """
        Return the K nearest neighbors — exposes your find_neighbors() logic.

        YOUR ORIGINAL find_neighbors():
        ────────────────────────────────
            distances = []
            for i in range(len(X_train)):
                dist = np.sqrt(np.sum((X_train[i] - new_point) ** 2))
                distances.append((i, dist))
            distances.sort(key=lambda x: x[1])
            return distances[:k]

        This method does the same thing but returns the actual neighbor
        X and Y values alongside the indices and distances, making it
        easier to inspect WHY a prediction was made.

        Returns
        -------
        distances  : ndarray (n_queries, K) — distance to each neighbor
        indices    : ndarray (n_queries, K) — training index of each neighbor
        X_neighbors: ndarray (n_queries, K, n_features) — neighbor feature values
        Y_neighbors: ndarray (n_queries, K) — neighbor Y values (used in mean)
        """
        self._check_fitted()
        k = n_neighbors or self.n_neighbors
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self._scaler.transform(X) if self.scale_features else X
        distances, indices = self._model.kneighbors(X_scaled, n_neighbors=k)

        X_neighbors = np.array([
            self.X_train_scaled_[idx] for idx in indices
        ])
        Y_neighbors = np.array([
            self.y_train_[idx] for idx in indices
        ])
        return distances, indices, X_neighbors, Y_neighbors

    # ── explain_prediction ────────────────────────────────────────────────────

    def explain_prediction(self, point):
        """
        Print a detailed trace of a single prediction — showing exactly
        what your find_neighbors() + regressor() did step by step.

        OUTPUT FORMAT:
        ─────────────
          For each of the K neighbors:
            - Rank (1 = closest)
            - Euclidean distance (your dist formula)
            - The neighbor's Y value (used in the average)
            - That neighbor's weight in the final average

          Then shows the final prediction = mean of neighbor Y values.

        Parameters
        ----------
        point : 1D array (n_features,) — matches your new_points[i]
        """
        self._check_fitted()
        point = np.array(point).reshape(1, -1)
        dists, idxs, _, Y_neighbors = self.get_neighbors(point)
        prediction = self.predict_single(point)

        print("\n" + "=" * 60)
        print("  KNN REGRESSION — PREDICTION TRACE")
        print(f"  Replicating: find_neighbors(k={self.n_neighbors}) + regressor()")
        print("=" * 60)
        print(f"  New point features : {point.flatten()}")
        print(f"\n  {'Rank':<6} {'Distance':>10}  {'Y_neighbor':>12}  {'Weight'}")
        print(f"  {'-'*5} {'-'*10}  {'-'*12}  {'-'*10}")

        y_vals  = Y_neighbors[0]
        d_vals  = dists[0]
        weights = (
            (1.0 / d_vals) / (1.0 / d_vals).sum()
            if self.weights == "distance"
            else np.ones(len(y_vals)) / len(y_vals)
        )

        for rank, (d, y, w) in enumerate(zip(d_vals, y_vals, weights), 1):
            print(f"  {rank:<6} {d:>10.4f}  {y:>12.4f}  {w:.4f}")

        print(f"\n  Prediction formula  : mean({[f'{v:.2f}' for v in y_vals]})")
        print(f"  = {' + '.join([f'{v:.2f}' for v in y_vals])} / {len(y_vals)}")
        print(f"  = {prediction:.4f}")
        print("=" * 60 + "\n")

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print a structured summary of the fitted model.

        Because KNN has no coefficients or learned parameters, the summary
        focuses on configuration, stored dataset stats, and complexity.
        This mirrors your original print statements, but structured.
        """
        self._check_fitted()
        n_train = len(self.y_train_)
        metric_label = (
            "Euclidean (p=2) — √Σ(xᵢ-yᵢ)²  ← your original formula"
            if self.p == 2 and self.metric == "minkowski"
            else "Manhattan (p=1) — Σ|xᵢ-yᵢ|"
            if self.p == 1
            else self.metric
        )

        sep = "=" * 62
        print(f"\n{sep}")
        print("  KNN REGRESSION MODEL SUMMARY")
        print(sep)
        print(f"  Task               : Regression (continuous output)")
        print(f"  Prediction method  : Average Y of K nearest neighbors")
        print(f"                       (your original regressor() method)")
        print(f"\n  ── Configuration ────────────────────────────────────")
        print(f"  K (n_neighbors)    : {self.n_neighbors}")
        print(f"  Distance metric    : {metric_label}")
        print(f"  Aggregation        : {'Uniform mean — np.mean(Y_neighbors)' if self.weights == 'uniform' else 'Weighted mean — 1/distance'}")
        print(f"  Feature scaling    : {'Yes (StandardScaler)' if self.scale_features else 'No — raw features'}")
        print(f"\n  ── Stored Training Data ─────────────────────────────")
        print(f"  Training samples   : {n_train}")
        print(f"  Features           : {self.n_features_}")
        print(f"  Feature names      : {self.feature_names_}")
        print(f"  Y range            : [{self.y_train_.min():.2f}, {self.y_train_.max():.2f}]")
        print(f"  Y mean             : {self.y_train_.mean():.4f}")
        print(f"\n  ── Complexity ───────────────────────────────────────")
        print(f"  Prediction cost    : O(n × d) = O({n_train} × {self.n_features_}) per query")
        print(f"  ≈ {n_train * self.n_features_:,} operations per prediction")
        print(f"\n  ── No Learnable Parameters ──────────────────────────")
        print(f"  KNN has no β coefficients, no weights to optimize.")
        print(f"  The model IS the training data (lazy learner).")
        print(f"  Use .explain_prediction(point) to trace any prediction")
        print(f"  back to its K neighbors and their Y values.")
        print(sep + "\n")

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(X, y) first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 ── HYPERPARAMETER TUNER
#  Your original code hardcoded K=3. This class finds the optimal K
#  via cross-validation and generates the elbow curve.
#
#  WHY K=3 MIGHT NOT ALWAYS BE BEST:
#  ────────────────────────────────────
#  K=1   → predicts exactly a training neighbor's Y → overfits (memorizes noise)
#  K=3   → your choice — a reasonable starting point
#  K=n   → predicts the global mean for everything → underfits
#
#  The right K balances local sensitivity vs. global smoothness.
# ══════════════════════════════════════════════════════════════════════════════

class KNNRegressionTuner:
    """
    Cross-validation based K tuner for KNN Regression.

    Searches over a range of K values (and optionally weights and p)
    using k-fold CV, then identifies the elbow — the K where adding
    more neighbors stops meaningfully improving RMSE.

    Example
    -------
    >>> tuner = KNNRegressionTuner(cv=5)                      # doctest: +SKIP
    >>> best_k = tuner.fit(X_train, y_train)                  # doctest: +SKIP
    >>> tuner.results_summary()                               # doctest: +SKIP
    """

    def __init__(
        self,
        cv: int = 5,
        k_range: range = None,
        random_state: int = 42,
    ):
        self.cv           = cv
        self.k_range      = k_range or range(1, 31)
        self.random_state = random_state

        self.best_k_      = None
        self.best_params_ = None
        self.k_rmse_      = {}       # k → mean CV RMSE
        self.k_stds_      = {}       # k → std of CV RMSE
        self.k_r2_        = {}       # k → mean CV R²

    def fit(self, X, y, search_weights: bool = True):
        """
        Systematically test each K value using k-fold cross-validation.

        STEP-BY-STEP:
        ─────────────
        Step 1 — Scale features once on all training data.
                 (For tuning comparison purposes, this is acceptable.)

        Step 2 — For each K in k_range:
                   a. Build a KNN Regressor with that K
                   b. Run k-fold CV: split data into k folds,
                      train on k-1 folds, evaluate on the held-out fold
                   c. Record mean and std of RMSE across folds

        Step 3 — Select K with the lowest mean CV RMSE.

        Step 4 — Optionally also search over weights='uniform' vs 'distance'.

        Parameters
        ----------
        X              : feature matrix
        y              : continuous targets
        search_weights : also compare uniform vs distance weighting

        Returns
        -------
        best_k : int
        """
        X = np.array(X)
        y = np.array(y).ravel()

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        weights_list = ["uniform", "distance"] if search_weights else ["uniform"]
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        n_combos = len(self.k_range) * len(weights_list)
        print(f"[KNNRegressionTuner] Testing {n_combos} combinations "
              f"(K={self.k_range.start}–{self.k_range.stop-1}, "
              f"weights={weights_list}) with {self.cv}-fold CV ...")

        best_rmse   = float("inf")
        best_k      = self.k_range.start
        best_weights = "uniform"

        for w in weights_list:
            for k in self.k_range:
                model  = KNeighborsRegressor(n_neighbors=k, weights=w)
                # neg_mean_squared_error: negate so higher = better
                scores = cross_val_score(
                    model, X_scaled, y,
                    cv=kf, scoring="neg_mean_squared_error"
                )
                r2_scores = cross_val_score(
                    model, X_scaled, y, cv=kf, scoring="r2"
                )
                rmse_scores = np.sqrt(-scores)
                key = (k, w)
                self.k_rmse_[key] = rmse_scores.mean()
                self.k_stds_[key] = rmse_scores.std()
                self.k_r2_[key]   = r2_scores.mean()

                if rmse_scores.mean() < best_rmse:
                    best_rmse    = rmse_scores.mean()
                    best_k       = k
                    best_weights = w

        self.best_k_      = best_k
        self.best_params_ = {
            "n_neighbors": best_k,
            "weights":     best_weights,
        }

        print(f"[KNNRegressionTuner] ✓ Best K={best_k}, "
              f"weights='{best_weights}'\n"
              f"  Best CV RMSE: {best_rmse:.4f}")
        return best_k

    def results_summary(self):
        """Print best hyperparameters and top-5 K values."""
        if self.best_k_ is None:
            raise RuntimeError("Call .fit() first.")

        print("\n" + "=" * 55)
        print("  KNN REGRESSION TUNING RESULTS")
        print("=" * 55)
        print(f"  Best K           : {self.best_k_}")
        print(f"  Best weights     : {self.best_params_['weights']}")
        print(f"\n  Top 5 K values by CV RMSE:")
        sorted_keys = sorted(self.k_rmse_.items(), key=lambda x: x[1])[:5]
        for (k, w), rmse in sorted_keys:
            marker = " ← BEST" if k == self.best_k_ and w == self.best_params_["weights"] else ""
            r2 = self.k_r2_[(k, w)]
            print(f"    K={k:<3}  weights={w:<10}  "
                  f"RMSE={rmse:.4f} ± {self.k_stds_[(k,w)]:.4f}  "
                  f"R²={r2:.4f}{marker}")
        print("=" * 55 + "\n")

    def get_best_model(self) -> "KNNRegressionModel":
        """Return a pre-configured KNNRegressionModel with best params."""
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        return KNNRegressionModel(
            n_neighbors=self.best_params_["n_neighbors"],
            weights=self.best_params_["weights"],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 ── EVALUATOR
#  Regression metrics to measure how well your averaged predictions match
#  the true continuous Y values.
#
#  METRICS YOUR ORIGINAL CODE DID NOT INCLUDE:
#  ─────────────────────────────────────────────
#  Your code printed predictions but didn't measure accuracy against truth.
#  These metrics quantify exactly how far predictions are from reality.
# ══════════════════════════════════════════════════════════════════════════════

class KNNRegressionEvaluator:
    """
    Regression metrics for KNN predictions.

    METRICS EXPLAINED:
    ──────────────────
    MSE (Mean Squared Error):
        Average of squared differences between predictions and true values.
        Penalizes large errors heavily (squared).
        MSE = (1/n) · Σ (yᵢ - ŷᵢ)²
        Unit: target² — hard to interpret directly.

    RMSE (Root MSE):
        Square root of MSE. Same unit as the target (e.g., dollars, °C).
        Think: "on average, predictions are off by RMSE units."
        Lower K → lower training RMSE, higher test RMSE (overfit).
        Higher K → higher training RMSE, may improve test RMSE (smooth).

    MAE (Mean Absolute Error):
        Average absolute difference. Less sensitive to outliers than RMSE.
        MAE = (1/n) · Σ |yᵢ - ŷᵢ|

    R² (Coefficient of Determination):
        Fraction of variance in Y explained by the model.
        R² = 1.0 → perfect (every prediction exact)
        R² = 0.0 → model predicts the mean for everything
        R² < 0.0 → worse than predicting the mean (bad model or wrong K)

    Example
    -------
    >>> ev = KNNRegressionEvaluator(model)          # doctest: +SKIP
    >>> ev.evaluate(X_test, y_test)                 # doctest: +SKIP
    >>> ev.cross_validate(X, y, cv=5)               # doctest: +SKIP
    """

    def __init__(self, model: KNNRegressionModel):
        self.model     = model
        self.metrics_  = {}
        self._residuals = None
        self._y_pred   = None
        self._y_true   = None

    def evaluate(self, X, y, label: str = "Test Set") -> dict:
        """
        Compute MSE, RMSE, MAE, and R² for a given (X, y) split.

        Parameters
        ----------
        X     : feature matrix
        y     : true continuous targets
        label : display label (e.g., "Train Set", "Test Set")

        Returns
        -------
        dict of metric name → value
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)

        self._y_true   = y
        self._y_pred   = y_pred
        self._residuals = y - y_pred

        mse  = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y, y_pred)
        r2   = r2_score(y, y_pred)

        n, p       = np.array(X).shape
        adj_r2     = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

        self.metrics_ = dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2, Adj_R2=adj_r2)

        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  EVALUATION — {label}  (K={self.model.n_neighbors})")
        print(sep)
        print(f"  MSE       : {mse:.4f}   (avg squared error — unit: Y²)")
        print(f"  RMSE      : {rmse:.4f}   (avg error in Y units — most interpretable)")
        print(f"  MAE       : {mae:.4f}   (avg absolute error — outlier-robust)")
        print(f"  R²        : {r2:.4f}   ({r2*100:.1f}% of Y variance explained)")
        print(f"  Adj. R²   : {adj_r2:.4f}   (R² penalized for number of features)")
        print(f"\n  Interpretation:")
        if r2 >= 0.9:
            interp = "Excellent — strong predictive power"
        elif r2 >= 0.7:
            interp = "Good — solid predictions"
        elif r2 >= 0.5:
            interp = "Moderate — consider tuning K or adding features"
        else:
            interp = "Weak — model may be underfitting or overfitting"
        print(f"    R² = {r2:.4f} → {interp}")
        print(sep + "\n")
        return self.metrics_

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        K-fold cross-validation for robust performance estimates.

        WHY THIS MATTERS FOR KNN REGRESSION:
        ──────────────────────────────────────
        KNN with a very small K (like K=1 or K=3 from your original code)
        can perfectly memorize training data — training RMSE near zero.
        CV reveals the TRUE generalization performance on unseen data.

        Uses a pipeline so the StandardScaler is fitted only on training
        folds — preventing data leakage between folds.

        Returns
        -------
        dict of mean/std for RMSE and R²
        """
        X = np.array(X)
        y = np.array(y).ravel()

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn",    KNeighborsRegressor(
                n_neighbors=self.model.n_neighbors,
                weights=self.model.weights,
                p=self.model.p,
            ))
        ])

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        mse_scores = -cross_val_score(
            pipe, X, y, cv=kf, scoring="neg_mean_squared_error"
        )
        r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        rmse_scores = np.sqrt(mse_scores)

        results = {
            "RMSE_mean": rmse_scores.mean(), "RMSE_std": rmse_scores.std(),
            "R2_mean":   r2_scores.mean(),   "R2_std":   r2_scores.std(),
            "MSE_mean":  mse_scores.mean(),
        }

        print(f"\n{'='*55}")
        print(f"  {cv}-FOLD CROSS-VALIDATION  (K={self.model.n_neighbors})")
        print(f"{'='*55}")
        print(f"  RMSE  mean ± std : {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
        print(f"  R²    mean ± std : {results['R2_mean']:.4f} ± {results['R2_std']:.4f}")
        print(f"{'='*55}\n")
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 ── VISUALIZER
#  Extends your original two plots (training scatter + prediction scatter)
#  with regression-specific diagnostics.
#
#  YOUR ORIGINAL PLOTS:
#    plot 1: ImportData()         → training data 3D scatter   ✓ kept
#    plot 2: plot_predictions()   → predictions on 3D scatter  ✓ kept + enhanced
#
#  NEW PLOTS ADDED:
#    plot 3: K elbow curve        → find optimal K visually
#    plot 4: Actual vs Predicted  → how close are predictions to truth?
#    plot 5: Residuals plot       → detect systematic errors
#    plot 6: Distance distribution→ how far are the K neighbors?
# ══════════════════════════════════════════════════════════════════════════════

class KNNRegressionVisualizer:
    """
    Diagnostic plots for KNN Regression, extending your original visualization.

    YOUR ORIGINAL PLOTS (preserved and enhanced):
    ───────────────────────────────────────────────
    plot_3d_training()     ← your ImportData() 3D scatter
    plot_3d_predictions()  ← your plot_predictions() with prediction points

    ADDITIONAL DIAGNOSTICS:
    ────────────────────────
    plot_k_elbow()         ← CV RMSE vs K — find optimal K
    plot_actual_vs_pred()  ← scatter of y_true vs y_pred
    plot_residuals()       ← residuals vs fitted values
    plot_distance_dist()   ← distribution of K-th neighbor distances
    plot_all()             ← 2×3 dashboard combining all plots

    Example
    -------
    >>> viz = KNNRegressionVisualizer(model, tuner)  # doctest: +SKIP
    >>> viz.plot_all(X_test, y_test)                 # doctest: +SKIP
    """

    def __init__(
        self,
        model: KNNRegressionModel,
        tuner: KNNRegressionTuner = None,
    ):
        self.model = model
        self.tuner = tuner

    # ── 1. Your original training scatter (from ImportData) ───────────────────

    def plot_3d_training(self, ax=None, title: str = "Training Data"):
        """
        3D scatter of training data — directly from your ImportData() method.

        YOUR ORIGINAL CODE:
            ax.scatter(X_train[:, 0], X_train[:, 1], Y_train,
                       c="red", alpha=0.5, marker='o', label="Training Data")

        This replicates that exactly, but uses the stored training arrays
        and works with any number of features (projecting to first two for 3D).
        """
        self.model._check_fitted()
        X = self.model.X_train_scaled_
        y = self.model.y_train_
        fn = self.model.feature_names_

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax  = fig.add_subplot(111, projection="3d")

        # Your original red scatter
        ax.scatter(
            X[:, 0], X[:, 1], y,
            c="red", alpha=0.5, marker="o", label="Training Data", s=20
        )
        ax.set_xlabel(fn[0] + " (scaled)" if self.model.scale_features else fn[0])
        ax.set_ylabel(fn[1] + " (scaled)" if self.model.scale_features else fn[1])
        ax.set_zlabel("Y")
        ax.set_title(title)
        ax.legend()

    # ── 2. Your original predictions scatter (from plot_predictions) ──────────

    def plot_3d_predictions(self, new_points, ax=None):
        """
        3D scatter of training data + predicted new points.

        YOUR ORIGINAL CODE:
            for point in new_points:
                knn = self.find_neighbors(3, self.X_train, point)
                prediction = self.regressor(knn)
                ax.scatter(point[0], point[1], prediction,
                           c="blue", marker='^', s=100)

        This replicates that loop exactly, adding annotations showing
        the predicted Y value for each new point.

        Parameters
        ----------
        new_points : array (n_points, n_features) — your new_points array
        """
        self.model._check_fitted()
        new_points = np.array(new_points)
        predictions = self.model.predict(new_points)
        X = self.model.X_train_scaled_
        y = self.model.y_train_
        fn = self.model.feature_names_

        if ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax  = fig.add_subplot(111, projection="3d")

        # Training data — your original red scatter
        ax.scatter(X[:, 0], X[:, 1], y,
                   c="red", alpha=0.35, marker="o", s=15, label="Training Data")

        # New points — your original blue triangles
        for i, (point, pred) in enumerate(zip(new_points, predictions)):
            # Scale the point the same way the model does
            pt_scaled = (
                self.model._scaler.transform(point.reshape(1, -1))[0]
                if self.model.scale_features else point
            )
            ax.scatter(
                pt_scaled[0], pt_scaled[1], pred,
                c="blue", marker="^", s=120, zorder=5,
                label="Predicted Point" if i == 0 else ""
            )
            ax.text(pt_scaled[0], pt_scaled[1], pred + 5,
                    f"ŷ={pred:.1f}", fontsize=7.5, color="navy")

        ax.set_xlabel(fn[0] + " (scaled)")
        ax.set_ylabel(fn[1] + " (scaled)")
        ax.set_zlabel("Predicted Y")
        ax.set_title(f"KNN Regression Predictions  (K={self.model.n_neighbors})\n"
                     f"Blue ▲ = predicted points (your new_points)")
        ax.legend(loc="best")

    # ── 3. K Elbow Curve ──────────────────────────────────────────────────────

    def plot_k_elbow(self, ax=None):
        """
        Cross-validated RMSE vs K — the regression elbow curve.

        WHAT TO LOOK FOR:
        ──────────────────
        - K=1 → very low training RMSE but high test RMSE (overfits)
        - K=3 (your original) → likely somewhere near the elbow
        - The elbow → where RMSE stops improving meaningfully
        - Best K (vertical line) → lowest mean CV RMSE
        """
        if self.tuner is None or not self.tuner.k_rmse_:
            print("[KNNRegressionVisualizer] No tuner results available.")
            return

        best_w  = self.tuner.best_params_["weights"]
        ks      = [k for (k, w) in self.tuner.k_rmse_ if w == best_w]
        rmses   = [self.tuner.k_rmse_[(k, best_w)] for k in ks]
        stds    = [self.tuner.k_stds_[(k, best_w)]  for k in ks]
        rmse_arr = np.array(rmses)
        std_arr  = np.array(stds)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(ks, rmse_arr, color="#E74C3C", linewidth=2.5,
                marker="o", markersize=5, label=f"CV RMSE  (weights={best_w})")
        ax.fill_between(ks, rmse_arr - std_arr, rmse_arr + std_arr,
                        alpha=0.15, color="#E74C3C", label="±1 std")

        # Mark your original K=3 choice
        if 3 in ks:
            idx3 = ks.index(3)
            ax.scatter([3], [rmse_arr[idx3]], color="#F39C12",
                       zorder=6, s=100, label=f"Your original K=3  (RMSE={rmse_arr[idx3]:.2f})")

        # Mark best K
        best_k = self.tuner.best_k_
        if best_k in ks:
            idx_b = ks.index(best_k)
            ax.axvline(best_k, color="#2ECC71", linestyle="--",
                       linewidth=1.8, label=f"Best K={best_k}  (RMSE={rmse_arr[idx_b]:.2f})")
            ax.scatter([best_k], [rmse_arr[idx_b]], color="#2ECC71",
                       zorder=7, s=100)

        ax.set_xlabel("K  (number of neighbors)")
        ax.set_ylabel("Cross-validated RMSE  (lower = better)")
        ax.set_title("KNN Regression Elbow Curve\n"
                     "Orange = your original K=3  |  Green = tuned best K")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── 4. Actual vs Predicted ────────────────────────────────────────────────

    def plot_actual_vs_pred(self, X, y, ax=None):
        """
        Scatter plot of true Y vs predicted ŷ.

        WHAT TO LOOK FOR:
        ──────────────────
        - Points on the diagonal → perfect predictions
        - Scatter around diagonal → prediction error (normal)
        - Systematic curve above/below → model bias
        - Small K → tighter scatter on training data, looser on test data
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)
        r2     = r2_score(y, y_pred)
        rmse   = np.sqrt(mean_squared_error(y, y_pred))

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(y, y_pred, alpha=0.55, color="#3498DB",
                   edgecolors="white", s=35)

        mn = min(y.min(), y_pred.min())
        mx = max(y.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="#E74C3C", linestyle="--",
                linewidth=1.5, label="Perfect prediction  (y = ŷ)")

        ax.set_xlabel("Actual Y  (true values)")
        ax.set_ylabel("Predicted ŷ  (KNN average of K neighbors)")
        ax.set_title(f"Actual vs Predicted\n"
                     f"R²={r2:.4f}  |  RMSE={rmse:.4f}  |  K={self.model.n_neighbors}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── 5. Residuals Plot ─────────────────────────────────────────────────────

    def plot_residuals(self, X, y, ax=None):
        """
        Residuals (y - ŷ) vs fitted values.

        FOR REGRESSION SPECIFICALLY:
        ──────────────────────────────
        A well-behaved regression model shows residuals randomly scattered
        around zero with no pattern. KNN-specific patterns to look for:

        - Fan shape      → variance increases with Y (heteroscedasticity)
        - Curved pattern → KNN is underfitting a non-linear relationship
                           with the current K (try smaller K)
        - Clusters       → data may have distinct groups that KNN treats together

        Residual = y_true - y_predicted
          Positive residual → model under-predicted
          Negative residual → model over-predicted
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)
        resid  = y - y_pred

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(y_pred, resid, alpha=0.55, color="#9B59B6",
                   edgecolors="white", s=35)
        ax.axhline(0, color="#E74C3C", linestyle="--", linewidth=1.5)

        # Smooth trend line
        try:
            from scipy.ndimage import uniform_filter1d
            idx    = np.argsort(y_pred)
            smooth = uniform_filter1d(resid[idx], size=max(3, len(y) // 20))
            ax.plot(y_pred[idx], smooth, color="orange", linewidth=1.5,
                    label="Trend (should hug 0)")
            ax.legend(fontsize=8)
        except ImportError:
            pass

        ax.set_xlabel("Fitted Values  ŷ")
        ax.set_ylabel("Residuals  (y − ŷ)")
        ax.set_title(f"Residuals vs Fitted  (K={self.model.n_neighbors})\n"
                     f"Random scatter around 0 = good fit")
        ax.grid(True, alpha=0.3)

    # ── 6. Distance Distribution ──────────────────────────────────────────────

    def plot_distance_dist(self, X, y, ax=None):
        """
        Distribution of K-th neighbor distances, split by error magnitude.

        WHAT THIS REVEALS:
        ───────────────────
        Points with SHORT K-th neighbor distances are deep inside a dense
        cluster → predictions are averages of very similar training points
        → generally accurate.

        Points with LONG K-th neighbor distances are in sparse regions
        → K neighbors may be far and dissimilar → less reliable predictions.

        Low residual + short distance → confident, accurate prediction
        High residual + long distance → uncertain, inaccurate prediction
        """
        self.model._check_fitted()
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)
        resid  = np.abs(y - y_pred)

        dists, _, _, _ = self.model.get_neighbors(X)
        kth_dist = dists[:, -1]    # distance to the farthest of K neighbors

        # Split into low-error and high-error halves
        median_err  = np.median(resid)
        low_err_mask  = resid <= median_err
        high_err_mask = resid > median_err

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        bins = np.histogram_bin_edges(kth_dist, bins=30)
        ax.hist(kth_dist[low_err_mask],  bins=bins, alpha=0.6, color="#2ECC71",
                label=f"Low error  |resid| ≤ {median_err:.2f}", edgecolor="white")
        ax.hist(kth_dist[high_err_mask], bins=bins, alpha=0.6, color="#E74C3C",
                label=f"High error |resid| >  {median_err:.2f}", edgecolor="white")

        ax.axvline(np.median(kth_dist[low_err_mask]),  color="#2ECC71",
                   linestyle="--", linewidth=1.5)
        ax.axvline(np.median(kth_dist[high_err_mask]), color="#E74C3C",
                   linestyle="--", linewidth=1.5)

        ax.set_xlabel(f"Distance to K-th Neighbor  (K={self.model.n_neighbors})")
        ax.set_ylabel("Count")
        ax.set_title("K-th Neighbor Distance vs Prediction Error\n"
                     "Farther neighbors → higher prediction uncertainty")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── 7. Full Dashboard ─────────────────────────────────────────────────────

    def plot_all(self, X_test, y_test, new_points=None, save_path: str = None):
        """
        Render the full 2×3 diagnostic dashboard.

        Layout:
          ┌──────────────┬──────────────┬──────────────┐
          │  K Elbow     │  3D Training │ Actual vs    │
          │  Curve       │  + Preds     │ Predicted    │
          ├──────────────┼──────────────┼──────────────┤
          │  Residuals   │  Distance    │  (reserved   │
          │  Plot        │  Distribution│   for future)│
          └──────────────┴──────────────┴──────────────┘

        Parameters
        ----------
        X_test     : test feature matrix
        y_test     : test targets
        new_points : array of custom points to show on 3D plot (optional)
        save_path  : file path to save the figure
        """
        metric_label = (
            "Euclidean" if self.model.p == 2 else
            "Manhattan" if self.model.p == 1 else self.model.metric
        )
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(
            f"KNN Regression — Diagnostic Dashboard\n"
            f"K={self.model.n_neighbors}  |  weights='{self.model.weights}'  |  "
            f"metric={metric_label}  |  scaling={'on' if self.model.scale_features else 'off'}",
            fontsize=13, fontweight="bold"
        )

        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection="3d")
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        self.plot_k_elbow(ax=ax1)

        # 3D plot: show training data + new_points or test data
        pts_to_show = (
            new_points if new_points is not None
            else np.array(X_test)[:min(10, len(X_test))]
        )
        self.plot_3d_predictions(pts_to_show, ax=ax2)

        self.plot_actual_vs_pred(X_test, y_test, ax=ax3)
        self.plot_residuals(X_test, y_test, ax=ax4)
        self.plot_distance_dist(X_test, y_test, ax=ax5)

        # Panel 6 — summary stats text box
        model     = self.model
        y_pred    = model.predict(X_test)
        y_test_np = np.array(y_test).ravel()
        rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
        r2   = r2_score(y_test_np, y_pred)
        mae  = mean_absolute_error(y_test_np, y_pred)

        ax6.axis("off")
        stats_text = (
            f"  MODEL CONFIGURATION\n"
            f"  {'─'*30}\n"
            f"  K (neighbors)  :  {model.n_neighbors}\n"
            f"  Weights        :  {model.weights}\n"
            f"  Metric         :  {metric_label}\n"
            f"  Scaled         :  {model.scale_features}\n\n"
            f"  TEST SET PERFORMANCE\n"
            f"  {'─'*30}\n"
            f"  RMSE           :  {rmse:.4f}\n"
            f"  MAE            :  {mae:.4f}\n"
            f"  R²             :  {r2:.4f}\n\n"
            f"  DATASET\n"
            f"  {'─'*30}\n"
            f"  Train samples  :  {len(model.y_train_)}\n"
            f"  Test samples   :  {len(y_test_np)}\n"
            f"  Features       :  {model.n_features_}\n"
        )
        ax6.text(
            0.05, 0.95, stats_text,
            transform=ax6.transAxes, fontsize=9.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F0F4F8",
                      edgecolor="#BDC3C7", linewidth=1.2)
        )
        ax6.set_title("Summary", fontsize=10, pad=8)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[KNNRegressionVisualizer] Dashboard saved to: {save_path}")

        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 ── FULL PIPELINE
#  Assembles all four components into one function call —
#  matching the structure of your Elastic Net, Logistic, and KNN modules.
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X,
    y,
    feature_names=None,
    new_points=None,
    test_size: float = 0.2,
    auto_tune: bool = True,
    n_neighbors: int = 3,
    weights: str = "uniform",
    k_range: range = None,
    cv: int = 5,
    plot: bool = True,
    save_plot: str = None,
    explain_samples: int = 2,
    random_state: int = 42,
) -> dict:
    """
    End-to-end KNN Regression pipeline in a single function call.

    PIPELINE STEPS:
    ───────────────
    1.  Train/test split (no stratification needed — continuous target)
    2.  (Optional) Tune K and weights via cross-validated grid search
    3.  Fit the KNN model (memorize training data + build distance index)
    4.  Print model summary
    5.  Evaluate on train AND test sets (MSE, RMSE, MAE, R²)
    6.  K-fold cross-validation for robust estimate
    7.  Explain N sample predictions (neighbor-level trace)
    8.  Predict custom new_points (your original new_points array)
    9.  (Optional) Generate 2×3 diagnostic dashboard

    Parameters
    ----------
    X            : feature matrix (array or DataFrame)
    y            : continuous target vector
    feature_names: column names (auto-detected from DataFrame)
    new_points   : your original new_points array to predict and visualize
    test_size    : fraction held out for testing (default 0.2)
    auto_tune    : search for optimal K via cross-validation
    n_neighbors  : K to use if auto_tune=False (default=3, your original)
    weights      : 'uniform' or 'distance' (if auto_tune=False)
    k_range      : range of K values to search (default range(1, 31))
    cv           : number of CV folds
    plot         : display diagnostic plots
    save_plot    : path to save the dashboard image
    explain_samples: number of test points to trace neighbor-by-neighbor
    random_state : seed for reproducibility

    Returns
    -------
    dict: model, tuner, evaluator, visualizer,
          train_metrics, test_metrics, cv_metrics, new_point_predictions
    """
    print("\n" + "█" * 62)
    print("  K-NEAREST NEIGHBORS REGRESSION — FULL PIPELINE")
    print("  (Extended from your original KNNRegression implementation)")
    print("█" * 62)

    # ── 1. Prepare inputs
    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X = np.array(X)
    y = np.array(y).ravel()
    if feature_names is None:
        feature_names = [f"X_{i+1}" for i in range(X.shape[1])]

    suggested_k = max(1, int(np.sqrt(len(y))))
    print(f"\n[Pipeline] Dataset — {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[Pipeline] Y range  — [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.2f}")
    print(f"[Pipeline] Rule-of-thumb K = √n = {suggested_k}  "
          f"(your original used K=3)")

    # ── 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[Pipeline] Split — Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Tune or use provided K
    k_range = k_range or range(1, min(31, len(X_train) // 2))
    tuner   = KNNRegressionTuner(cv=cv, k_range=k_range,
                                  random_state=random_state)

    if auto_tune:
        print("\n[Pipeline] Step 1/5 — Hyperparameter Tuning ...")
        best_k = tuner.fit(X_train, y_train, search_weights=True)
        tuner.results_summary()
        n_neighbors = tuner.best_params_["n_neighbors"]
        weights     = tuner.best_params_["weights"]
    else:
        print(f"\n[Pipeline] Step 1/5 — Using K={n_neighbors} "
              f"(your original), weights='{weights}'")
        # Populate tuner k_scores for the elbow plot
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        kf       = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        tuner.best_k_      = n_neighbors
        tuner.best_params_ = {"n_neighbors": n_neighbors, "weights": weights}
        for k in k_range:
            scores = -cross_val_score(
                KNeighborsRegressor(n_neighbors=k, weights=weights),
                X_tr_sc, y_train, cv=kf,
                scoring="neg_mean_squared_error"
            )
            rmse_scores = np.sqrt(scores)
            tuner.k_rmse_[(k, weights)] = rmse_scores.mean()
            tuner.k_stds_[(k, weights)] = rmse_scores.std()
            r2s = cross_val_score(
                KNeighborsRegressor(n_neighbors=k, weights=weights),
                X_tr_sc, y_train, cv=kf, scoring="r2"
            )
            tuner.k_r2_[(k, weights)] = r2s.mean()

    # ── 4. Fit
    print("\n[Pipeline] Step 2/5 — Fitting KNN Regression model ...")
    model = KNNRegressionModel(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    # ── 5. Evaluate
    print("\n[Pipeline] Step 3/5 — Evaluating ...")
    evaluator     = KNNRegressionEvaluator(model)
    train_metrics = evaluator.evaluate(X_train, y_train, label="Train Set")
    test_metrics  = evaluator.evaluate(X_test,  y_test,  label="Test Set")
    cv_metrics    = evaluator.cross_validate(X, y, cv=cv)

    # ── 6. Explain sample predictions (your original print loop, enhanced)
    if explain_samples > 0:
        print(f"\n[Pipeline] Step 4/5 — Explaining {explain_samples} "
              f"sample predictions ...")
        for i in range(min(explain_samples, len(X_test))):
            model.explain_prediction(X_test[i])

    # ── 7. Predict new_points (your original new_points array)
    new_point_predictions = None
    if new_points is not None:
        new_points = np.array(new_points)
        new_point_predictions = model.predict(new_points)
        print("\n[Pipeline] Predictions for your new_points:")
        print(f"  {'Point':<30}  {'Prediction':>12}")
        print(f"  {'-'*30}  {'-'*12}")
        for pt, pred in zip(new_points, new_point_predictions):
            print(f"  {str(pt):<30}  {pred:>12.4f}")

    # ── 8. Plots
    visualizer = KNNRegressionVisualizer(model, tuner)
    if plot:
        print("\n[Pipeline] Step 5/5 — Generating diagnostic dashboard ...")
        visualizer.plot_all(
            X_test, y_test,
            new_points=new_points,
            save_path=save_plot
        )

    print("\n" + "█" * 62)
    print("  PIPELINE COMPLETE")
    print("█" * 62 + "\n")

    return {
        "model":                   model,
        "tuner":                   tuner,
        "evaluator":               evaluator,
        "visualizer":              visualizer,
        "train_metrics":           train_metrics,
        "test_metrics":            test_metrics,
        "cv_metrics":              cv_metrics,
        "new_point_predictions":   new_point_predictions,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO — replicates your original KNNRegression class, then runs the
#  full extended pipeline
#  python knn_regression_module.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  KNN REGRESSION MODULE — DEMO")
    print("  Replicating and extending your original KNNRegression class")
    print("=" * 62 + "\n")

    from sklearn.datasets import make_regression

    # ── Your original dataset (exact same parameters)
    X_demo, y_demo = make_regression(
        n_samples=300, n_features=2, n_informative=2,
        noise=5, bias=30, random_state=200
    )

    # ── Your original new_points
    new_points = np.array([
        [-1,  1],
        [ 0,  2],
        [-3, -2],
        [ 3, -3],
    ])

    # ── Run the full pipeline
    results = run_full_pipeline(
        X             = X_demo,
        y             = y_demo,
        feature_names = ["X_1", "X_2"],
        new_points    = new_points,
        test_size     = 0.2,
        auto_tune     = True,
        k_range       = range(1, 25),
        cv            = 5,
        plot          = True,
        save_plot     = "knn_regression_diagnostics.png",
        explain_samples = 2,
    )

    model = results["model"]

    # ── Replicate your original output exactly
    print("\n── Replicating your original KNNRegression output ──")
    print("Predictions for new points:")
    for point in new_points:
        pred = model.predict_single(point)
        print(f"  Point: {point}, Prediction: {pred:.2f}")