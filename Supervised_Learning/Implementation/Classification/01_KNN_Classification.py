"""
================================================================================
  K-NEAREST NEIGHBORS (KNN) CLASSIFICATION MODULE
  A complete, plug-and-play module for KNN-based classification
================================================================================

WHAT IS K-NEAREST NEIGHBORS?
------------------------------
KNN is a non-parametric, instance-based supervised learning algorithm.
It makes predictions by looking at the K most similar training examples
to a new data point and taking a majority vote of their class labels.

  "Tell me who your neighbors are, and I'll tell you who you are."

HOW KNN CLASSIFICATION WORKS — STEP BY STEP:
─────────────────────────────────────────────
  Given a new point X_new:

  1. MEASURE DISTANCE to every training point:
         d(X_new, X_train[i])  using Euclidean, Manhattan, or Minkowski

  2. SORT all training points by distance (closest first)

  3. SELECT the K nearest neighbors

  4. MAJORITY VOTE among those K neighbors' class labels:
         prediction = argmax( count(class_j) for j in neighbors )

  5. PROBABILITY ESTIMATE:
         P(class_j | X_new) = count(class_j in K neighbors) / K

HOW KNN DIFFERS FROM LOGISTIC REGRESSION:
──────────────────────────────────────────
  Logistic Regression       KNN
  ─────────────────────     ─────────────────────────────────
  Parametric (learns β)     Non-parametric (no parameters)
  Eager learner (trains)    Lazy learner (memorizes dataset)
  Linear decision boundary  Non-linear, flexible boundary
  Fast prediction           Slow prediction (O(n·d) per query)
  Interpretable             Black-box
  Handles large data well   Struggles with large datasets

KEY HYPERPARAMETER — K:
────────────────────────
  K = 1   → memorizes training data perfectly, overfits heavily
  K = n   → predicts majority class for everything, underfits
  Sweet spot: typically K = sqrt(n_samples), odd number to break ties

  Low K  → jagged, complex boundary → low bias, high variance
  High K → smooth, simple boundary  → high bias, low variance

DISTANCE METRICS:
─────────────────
  Euclidean (p=2) : √Σ(xᵢ - yᵢ)²   Standard straight-line distance
  Manhattan (p=1) : Σ|xᵢ - yᵢ|     Grid/city-block distance
  Minkowski (p=n) : (Σ|xᵢ - yᵢ|ⁿ)^(1/n)  Generalization of both

  ⚠️  CRITICAL: KNN is extremely sensitive to feature scale.
      A feature with range [0, 10000] will dominate one with range [0, 1].
      Always standardize features before using KNN.

WEIGHTING STRATEGIES:
─────────────────────
  'uniform'  → all K neighbors vote equally
  'distance' → closer neighbors get more weight (1/distance)
               Use when you expect smooth class transitions

HOW TO USE THIS MODULE:
-----------------------
  from knn_module import KNNModel

  model = KNNModel(n_neighbors=5)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  model.summary()

CONTENTS:
---------
  1. KNNModel           — main classifier with fit/predict/summary
  2. KNNTuner           — optimal K search via cross-validation + elbow method
  3. KNNEvaluator       — accuracy, F1, AUC, confusion matrix, full report
  4. KNNVisualizer      — decision boundary, K-elbow, distance plots, dashboard
  5. run_full_pipeline()— one-call: tune → fit → evaluate → visualize

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
from matplotlib.patches import Patch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, log_loss,
    average_precision_score, precision_recall_curve
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 ── CORE MODEL CLASS
#  Wraps sklearn KNeighborsClassifier with auto-scaling, label encoding,
#  neighbor inspection, and a human-readable summary.
# ══════════════════════════════════════════════════════════════════════════════

class KNNModel:
    """
    A production-ready KNN Classifier.

    PREDICTION PROCESS (what happens on every .predict() call):
    ────────────────────────────────────────────────────────────
    1. Scale X_new using the same scaler fitted on training data.
    2. Compute distance from X_new to EVERY training point.
    3. Sort training points by distance — pick the K closest.
    4. Count class labels among those K neighbors.
    5. Return the class with the highest count (majority vote).

    Unlike Logistic Regression, there is NO training phase beyond
    memorizing the training data. All computation happens at prediction time.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider.
        Rule of thumb: start with sqrt(n_training_samples), use odd numbers.

    metric : str, default='minkowski'
        Distance metric.
          'minkowski' with p=2  → Euclidean (standard straight-line)
          'minkowski' with p=1  → Manhattan (city-block / grid)
          'cosine'              → angle between vectors (good for text)
          'euclidean'           → explicit Euclidean

    p : int, default=2
        Power for the Minkowski metric. 2 = Euclidean, 1 = Manhattan.

    weights : str, default='uniform'
        Voting strategy:
          'uniform'  → each neighbor counts equally
          'distance' → closer neighbors count more (weight = 1/distance)

    algorithm : str, default='auto'
        Internal data structure for neighbor search:
          'auto'      → sklearn chooses the best automatically
          'ball_tree' → efficient for high dimensions
          'kd_tree'   → efficient for low dimensions (< 20 features)
          'brute'     → exact, good for small datasets

    scale_features : bool, default=True
        Standardize features. CRITICAL for KNN — never disable unless
        features are already on identical scales.

    Example
    -------
    >>> model = KNNModel(n_neighbors=7, weights='distance')  # doctest: +SKIP
    >>> model.fit(X_train, y_train)                          # doctest: +SKIP
    >>> preds = model.predict(X_test)                        # doctest: +SKIP
    >>> model.summary()                                      # doctest: +SKIP
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        p: int = 2,
        weights: str = "uniform",
        algorithm: str = "auto",
        scale_features: bool = True,
        random_state: int = 42,
    ):
        self.n_neighbors   = n_neighbors
        self.metric        = metric
        self.p             = p
        self.weights       = weights
        self.algorithm     = algorithm
        self.scale_features = scale_features
        self.random_state  = random_state

        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p,
            weights=weights,
            algorithm=algorithm,
        )
        self._scaler        = StandardScaler() if scale_features else None
        self._label_encoder = LabelEncoder()

        # Populated during fit()
        self.feature_names_  = None
        self.classes_        = None
        self.is_binary_      = None
        self.X_train_scaled_ = None   # stored for neighbor inspection
        self.y_train_enc_    = None
        self.is_fitted_      = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        """
        'Train' the KNN model — which means memorizing the dataset.

        WHAT ACTUALLY HAPPENS HERE:
        ────────────────────────────
        1. Standardize features so all dimensions are on equal footing.
        2. Encode string/categorical labels to integers.
        3. Store the scaled training data in memory.
        4. Build the internal index structure (KD-tree or Ball-tree)
           that allows fast approximate neighbor lookups at prediction time.

        There is NO optimization loop, NO gradient descent, NO learned β.
        The model IS the training data.

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

        # Encode labels to integers
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_   = self._label_encoder.classes_
        self.is_binary_ = len(self.classes_) == 2

        # Scale features — absolutely critical for KNN
        X_scaled = self._scaler.fit_transform(X) if self.scale_features else X

        # "Fit" = store training data + build index
        self._model.fit(X_scaled, y_enc)
        self.X_train_scaled_ = X_scaled
        self.y_train_enc_    = y_enc
        self.is_fitted_      = True

        metric_desc = f"{'Euclidean' if self.p == 2 else 'Manhattan'}" \
                      if self.metric == "minkowski" else self.metric

        mode = "Binary" if self.is_binary_ else f"Multiclass ({len(self.classes_)} classes)"
        print(
            f"[KNNModel] Fitted — {mode}\n"
            f"  Samples memorized : {X.shape[0]} | Features: {X.shape[1]}\n"
            f"  K (neighbors)     : {self.n_neighbors}\n"
            f"  Distance metric   : {metric_desc}\n"
            f"  Voting strategy   : {self.weights}\n"
            f"  Classes           : {list(self.classes_)}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict class labels for new samples.

        RUNTIME COST:
        ─────────────
        For each query point, KNN must compute distance to ALL training
        points: O(n_train × n_features). With 10k samples and 20 features,
        that's 200k distance computations per prediction.

        This is why KNN is called a 'lazy learner' — all work deferred
        to prediction time.

        Returns
        -------
        y_pred : array of original class labels (not integers)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(np.array(X)) if self.scale_features else np.array(X)
        y_enc = self._model.predict(X_scaled)
        return self._label_encoder.inverse_transform(y_enc)

    # ── predict_proba ─────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Return class probability estimates.

        HOW PROBABILITIES ARE COMPUTED IN KNN:
        ────────────────────────────────────────
        Unlike logistic regression (which uses a sigmoid), KNN estimates
        probabilities by counting:

            P(class_j | X_new) = (neighbors belonging to class_j) / K

        Example: If K=5 and 3 neighbors are "Positive", 2 are "Negative":
            P(Positive) = 3/5 = 0.60
            P(Negative) = 2/5 = 0.40

        With weights='distance', probabilities are weighted by 1/distance.

        Note: KNN probabilities are coarse (multiples of 1/K), unlike
        logistic regression's continuous sigmoid outputs.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(np.array(X)) if self.scale_features else np.array(X)
        return self._model.predict_proba(X_scaled)

    # ── get_neighbors ─────────────────────────────────────────────────────────

    def get_neighbors(self, X, n_neighbors: int = None):
        """
        Return the K nearest neighbors for each query point.

        This is one of KNN's unique advantages over parametric models —
        you can inspect exactly WHICH training examples influenced a
        prediction and WHY.

        Parameters
        ----------
        X           : query points, shape (n_queries, n_features)
        n_neighbors : how many neighbors to return (defaults to model's K)

        Returns
        -------
        distances : ndarray (n_queries, K) — distance to each neighbor
        indices   : ndarray (n_queries, K) — training set index of each neighbor
        labels    : list of arrays           — class label of each neighbor
        """
        self._check_fitted()
        k = n_neighbors or self.n_neighbors
        X_scaled = self._scaler.transform(np.array(X)) if self.scale_features else np.array(X)
        distances, indices = self._model.kneighbors(X_scaled, n_neighbors=k)
        labels = [
            self._label_encoder.inverse_transform(self.y_train_enc_[idx])
            for idx in indices
        ]
        return distances, indices, labels

    # ── explain_prediction ────────────────────────────────────────────────────

    def explain_prediction(self, X_single, feature_names: list = None):
        """
        Human-readable explanation of a single prediction.

        Shows:
          - The predicted class and confidence
          - Each of the K neighbors: distance + class label
          - Vote tally and winning class

        This is KNN's killer feature for interpretability —
        you can always trace a prediction back to real training examples.

        Parameters
        ----------
        X_single : 1D array of shape (n_features,)
        """
        self._check_fitted()
        X_single = np.array(X_single).reshape(1, -1)
        pred     = self.predict(X_single)[0]
        proba    = self.predict_proba(X_single)[0]
        dists, idxs, lbls = self.get_neighbors(X_single)

        print("\n" + "=" * 58)
        print("  KNN PREDICTION EXPLANATION")
        print("=" * 58)
        print(f"  Predicted class : {pred}")
        print(f"  Confidence      : {proba.max():.2%}  "
              f"({int(proba.max() * self.n_neighbors)}/{self.n_neighbors} neighbors)")

        print(f"\n  {'Rank':<6} {'Distance':>10}  {'Class':<15}  {'Vote'}")
        print(f"  {'-'*5} {'-'*10}  {'-'*15}  {'-'*6}")

        vote_tally = {}
        for rank, (dist, lbl) in enumerate(zip(dists[0], lbls[0]), 1):
            vote = "✓" if lbl == pred else " "
            print(f"  {rank:<6} {dist:>10.4f}  {str(lbl):<15}  {vote}")
            vote_tally[lbl] = vote_tally.get(lbl, 0) + 1

        print(f"\n  Vote tally: " + " | ".join(
            f"{cls}: {cnt}" for cls, cnt in sorted(vote_tally.items(),
                                                    key=lambda x: -x[1])
        ))
        print("=" * 58 + "\n")

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print a structured summary of the fitted model.

        Unlike logistic regression, KNN has no coefficients or odds ratios.
        The summary instead focuses on:
          - Configuration (K, metric, weights)
          - Dataset characteristics (n_samples, n_features, class distribution)
          - Memory usage (the entire training set is stored)
          - Prediction complexity estimate
        """
        self._check_fitted()
        n_train = len(self.y_train_enc_)
        _, class_counts = np.unique(self.y_train_enc_, return_counts=True)
        class_dist = {
            cls: f"{cnt} ({cnt/n_train*100:.1f}%)"
            for cls, cnt in zip(self.classes_, class_counts)
        }
        metric_full = (
            f"Euclidean (p=2)" if self.p == 2 and self.metric == "minkowski"
            else f"Manhattan (p=1)" if self.p == 1 and self.metric == "minkowski"
            else self.metric
        )
        n_features  = self.X_train_scaled_.shape[1]
        mem_mb       = self.X_train_scaled_.nbytes / 1e6

        sep = "=" * 60
        print(f"\n{sep}")
        print("  KNN CLASSIFICATION MODEL SUMMARY")
        print(sep)
        print(f"  Task              : {'Binary' if self.is_binary_ else 'Multiclass'} Classification")
        print(f"  Classes           : {list(self.classes_)}")
        print(f"\n  ── Configuration ──────────────────────────────────")
        print(f"  K (n_neighbors)   : {self.n_neighbors}")
        print(f"  Distance metric   : {metric_full}")
        print(f"  Voting strategy   : {self.weights}")
        print(f"  Algorithm         : {self.algorithm}")
        print(f"  Feature scaling   : {'Yes (StandardScaler)' if self.scale_features else 'No'}")
        print(f"\n  ── Stored Dataset ─────────────────────────────────")
        print(f"  Training samples  : {n_train}")
        print(f"  Features          : {n_features}")
        print(f"  Memory footprint  : {mem_mb:.2f} MB")
        print(f"  Class distribution:")
        for cls, desc in class_dist.items():
            print(f"    {str(cls):<20} → {desc}")
        print(f"\n  ── Complexity ─────────────────────────────────────")
        print(f"  Prediction cost   : O(n × d) = O({n_train} × {n_features}) per query")
        print(f"  ≈ {n_train * n_features:,} distance computations per prediction")
        print(f"\n  ── No Coefficients ────────────────────────────────")
        print(f"  KNN is non-parametric — there are no β coefficients,")
        print(f"  no decision function, no odds ratios. The model IS")
        print(f"  the training data. Use .explain_prediction() to")
        print(f"  trace any prediction back to specific training examples.")
        print(sep + "\n")

    # ── private ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 ── HYPERPARAMETER TUNER
#  Finding the optimal K is the single most important hyperparameter decision.
#  Too small → overfit. Too large → underfit.
#  This class searches systematically and also generates the elbow curve.
# ══════════════════════════════════════════════════════════════════════════════

class KNNTuner:
    """
    Cross-validation based tuner for KNN hyperparameters.

    WHAT GETS TUNED:
    ─────────────────
    n_neighbors (K) : The most critical parameter.
                      Searched over a range using stratified CV.
    weights         : 'uniform' vs 'distance' voting
    metric          : distance function (Euclidean vs Manhattan)

    THE ELBOW METHOD FOR K:
    ───────────────────────
    Plot cross-validated accuracy vs K. The accuracy typically:
      1. Starts high at K=1 (overfitting to training data)
      2. Dips (validation is noisy)
      3. Rises to an optimal point
      4. Gradually declines as K grows too large (underfitting)

    The 'elbow' — where improvement slows — is the sweet spot.

    WHY STRATIFIED CV:
    ───────────────────
    Same reason as logistic regression — ensures each fold has a
    representative class distribution, especially important with
    imbalanced datasets.

    Example
    -------
    >>> tuner = KNNTuner(cv=5)                               # doctest: +SKIP
    >>> best_k = tuner.fit(X_train, y_train)                 # doctest: +SKIP
    >>> tuner.results_summary()                              # doctest: +SKIP
    """

    def __init__(
        self,
        cv: int = 5,
        scoring: str = "accuracy",
        k_range: range = None,
        random_state: int = 42,
    ):
        self.cv           = cv
        self.scoring      = scoring
        self.k_range      = k_range or range(1, 31)
        self.random_state = random_state

        self.best_k_       = None
        self.best_params_  = None
        self.k_scores_     = {}     # k → mean CV score
        self.k_stds_       = {}     # k → std of CV scores
        self._grid_search  = None

    def fit(self, X, y, search_weights: bool = True, search_metric: bool = True):
        """
        Run stratified grid search over K, weights, and optionally metric.

        STEP-BY-STEP:
        ─────────────
        1. Scale features (using all data — fine for tuning purposes).
        2. Build parameter grid: all combinations of K × weights × metric.
        3. For each combination, run stratified K-fold CV.
        4. Record mean and std of scores across folds.
        5. Select the K with the highest mean CV score.
        6. Also record per-K scores for the elbow plot.

        Parameters
        ----------
        X              : feature matrix
        y              : class labels
        search_weights : include 'uniform' vs 'distance' in grid
        search_metric  : include Euclidean vs Manhattan in grid

        Returns
        -------
        best_k : int
        """
        X = np.array(X)
        y = np.array(y).ravel()

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        weights_options = ["uniform", "distance"] if search_weights else ["uniform"]
        metric_options  = [1, 2] if search_metric else [2]   # p=1 Manhattan, p=2 Euclidean

        param_grid = {
            "n_neighbors": list(self.k_range),
            "weights":     weights_options,
            "p":           metric_options,
        }

        total = len(self.k_range) * len(weights_options) * len(metric_options)
        print(f"[KNNTuner] Searching {total} combinations "
              f"(K={self.k_range.start}–{self.k_range.stop - 1}, "
              f"weights={weights_options}, p={metric_options}) "
              f"with {self.cv}-fold stratified CV ...")

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                              random_state=self.random_state)

        self._grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=skf,
            scoring=self.scoring,
            n_jobs=-1,
            return_train_score=True,
        )
        self._grid_search.fit(X_scaled, y)

        self.best_params_ = self._grid_search.best_params_
        self.best_k_      = self.best_params_["n_neighbors"]

        # Build per-K accuracy curve (for elbow plot) using best weights/p
        best_w = self.best_params_["weights"]
        best_p = self.best_params_["p"]
        for k in self.k_range:
            scores = cross_val_score(
                KNeighborsClassifier(n_neighbors=k, weights=best_w, p=best_p),
                X_scaled, y, cv=skf, scoring=self.scoring
            )
            self.k_scores_[k] = scores.mean()
            self.k_stds_[k]   = scores.std()

        print(f"[KNNTuner] ✓ Best K={self.best_k_}, "
              f"weights='{best_w}', metric={'Euclidean' if best_p == 2 else 'Manhattan'}\n"
              f"  Best {self.scoring}: {self._grid_search.best_score_:.4f}")

        return self.best_k_

    def results_summary(self):
        """Print the best hyperparameters and top-5 K values."""
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")

        print("\n" + "=" * 55)
        print("  KNN TUNING RESULTS")
        print("=" * 55)
        print(f"  Best K           : {self.best_k_}")
        print(f"  Best weights     : {self.best_params_['weights']}")
        print(f"  Best metric      : {'Euclidean' if self.best_params_['p'] == 2 else 'Manhattan'}")
        print(f"  Best CV score    : {self._grid_search.best_score_:.4f}")
        print(f"\n  Top 5 K values (by {self.scoring}):")
        sorted_k = sorted(self.k_scores_.items(), key=lambda x: -x[1])[:5]
        for k, score in sorted_k:
            marker = " ← BEST" if k == self.best_k_ else ""
            print(f"    K={k:<4}  {self.scoring}={score:.4f} ± {self.k_stds_[k]:.4f}{marker}")
        print("=" * 55 + "\n")

    def get_best_model(self) -> "KNNModel":
        """Return a pre-configured KNNModel with the best hyperparameters."""
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        return KNNModel(
            n_neighbors=self.best_params_["n_neighbors"],
            weights=self.best_params_["weights"],
            p=self.best_params_["p"],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 ── EVALUATOR
#  Same rich classification metrics as logistic regression, plus
#  KNN-specific diagnostics: neighbor purity and distance distributions.
# ══════════════════════════════════════════════════════════════════════════════

class KNNEvaluator:
    """
    Comprehensive classification evaluator for KNN.

    METRICS (same as logistic regression, repeated here for completeness):
    ────────────────────────────────────────────────────────────────────────
    Accuracy    : % all predictions correct
    Precision   : of predicted positives, how many were truly positive
    Recall      : of actual positives, how many did we catch
    F1 Score    : harmonic mean of precision and recall
    ROC-AUC     : area under the ROC curve (threshold-independent)
    Log Loss    : how well-calibrated are KNN's coarse probability estimates

    KNN-SPECIFIC DIAGNOSTICS:
    ──────────────────────────
    Neighbor Purity : for each test sample, what % of its K neighbors
                      share the same class as the true label?
                      High purity → data is locally well-clustered.

    Example
    -------
    >>> ev = KNNEvaluator(model)             # doctest: +SKIP
    >>> ev.evaluate(X_test, y_test)          # doctest: +SKIP
    >>> ev.cross_validate(X, y, cv=5)        # doctest: +SKIP
    """

    def __init__(self, model: KNNModel):
        self.model    = model
        self.metrics_ = {}
        self._y_true  = None
        self._y_pred  = None
        self._y_proba = None

    def evaluate(self, X, y, label: str = "Test Set") -> dict:
        """
        Compute all classification metrics for a given (X, y) split.

        Parameters
        ----------
        X     : feature matrix
        y     : true class labels
        label : display label (e.g., "Train Set", "Test Set")

        Returns
        -------
        dict of metric name → value
        """
        y = np.array(y).ravel()
        y_pred  = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        self._y_true  = y
        self._y_pred  = y_pred
        self._y_proba = y_proba

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y, y_pred, average="weighted", zero_division=0)
        ll   = log_loss(y, y_proba)

        try:
            if self.model.is_binary_:
                auc = roc_auc_score(
                    self.model._label_encoder.transform(y),
                    y_proba[:, 1]
                )
            else:
                auc = roc_auc_score(
                    self.model._label_encoder.transform(y),
                    y_proba, multi_class="ovr", average="weighted"
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
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  ROC-AUC    : {auc:.4f}")
        print(f"  Log Loss   : {ll:.4f}")
        report = classification_report(y, y_pred, zero_division=0)
        indented = "\n".join("    " + line for line in report.splitlines())
        print(f"\n  Classification Report:\n{indented}")
        print(sep + "\n")
        return self.metrics_

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Stratified k-fold cross-validation for robust performance estimates.

        Note: uses a pipeline so scaling is fitted only on training folds,
        preventing data leakage between folds.

        Returns
        -------
        dict of mean/std for accuracy, F1, ROC-AUC
        """
        X = np.array(X)
        y = np.array(y).ravel()

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=self.model.n_neighbors,
                weights=self.model.weights,
                p=self.model.p,
                metric=self.model.metric,
            ))
        ])

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        f1_scores  = cross_val_score(pipe, X, y, cv=skf, scoring="f1_weighted")

        try:
            auc_scores = cross_val_score(
                pipe, X, y, cv=skf,
                scoring="roc_auc" if len(np.unique(y)) == 2
                else "roc_auc_ovr_weighted"
            )
        except Exception:
            auc_scores = np.array([float("nan")] * cv)

        results = {
            "Accuracy_mean": acc_scores.mean(), "Accuracy_std": acc_scores.std(),
            "F1_mean":       f1_scores.mean(),  "F1_std":       f1_scores.std(),
            "AUC_mean":      auc_scores.mean(), "AUC_std":      auc_scores.std(),
        }

        print(f"\n{'='*55}")
        print(f"  {cv}-FOLD STRATIFIED CROSS-VALIDATION  (K={self.model.n_neighbors})")
        print(f"{'='*55}")
        print(f"  Accuracy  mean ± std : {results['Accuracy_mean']:.4f} ± {results['Accuracy_std']:.4f}")
        print(f"  F1        mean ± std : {results['F1_mean']:.4f} ± {results['F1_std']:.4f}")
        print(f"  ROC-AUC   mean ± std : {results['AUC_mean']:.4f} ± {results['AUC_std']:.4f}")
        print(f"{'='*55}\n")
        return results

    def neighbor_purity(self, X, y) -> dict:
        """
        Compute neighbor purity — a KNN-specific diagnostic.

        WHAT IS NEIGHBOR PURITY?
        ─────────────────────────
        For each test sample, look at its K neighbors in the training set.
        Purity = fraction of those K neighbors that share the TRUE class label.

          Purity = 1.0 → all K neighbors agree with the true label
                         (the point is in a clean, well-separated region)
          Purity = 0.0 → no neighbors match the true label
                         (the point is deep in the wrong cluster)

        High average purity → data is well-clustered, KNN should work well.
        Low average purity  → classes overlap heavily, consider other models.

        Returns
        -------
        dict: mean_purity, std_purity, per_class_purity
        """
        y = np.array(y).ravel()
        y_enc = self.model._label_encoder.transform(y)
        _, indices, neighbor_labels = self.model.get_neighbors(X)

        purities = []
        for true_label, n_labels in zip(y, neighbor_labels):
            purity = np.mean(n_labels == true_label)
            purities.append(purity)

        purities = np.array(purities)
        per_class = {}
        for cls in self.model.classes_:
            mask = y == cls
            per_class[str(cls)] = purities[mask].mean() if mask.sum() > 0 else 0.0

        result = {
            "mean_purity": purities.mean(),
            "std_purity":  purities.std(),
            "per_class":   per_class,
        }

        print(f"\n{'='*50}")
        print(f"  NEIGHBOR PURITY ANALYSIS  (K={self.model.n_neighbors})")
        print(f"{'='*50}")
        print(f"  Mean purity   : {result['mean_purity']:.4f}")
        print(f"  Std  purity   : {result['std_purity']:.4f}")
        print(f"\n  Purity by class:")
        for cls, p in per_class.items():
            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
            print(f"    {str(cls):<15} {bar}  {p:.4f}")
        interpretation = (
            "Excellent — well-separated clusters" if result['mean_purity'] > 0.85
            else "Good — some boundary overlap" if result['mean_purity'] > 0.70
            else "Moderate — significant class overlap"
        )
        print(f"\n  Interpretation: {interpretation}")
        print(f"{'='*50}\n")
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 ── VISUALIZER
#  Seven KNN-specific diagnostic plots including the decision boundary —
#  the most intuitive visualization of what KNN actually does.
# ══════════════════════════════════════════════════════════════════════════════

class KNNVisualizer:
    """
    Diagnostic plots specifically designed for KNN classification.

    PLOTS:
    ──────
    1. plot_k_elbow          — CV accuracy vs K (find optimal K)
    2. plot_decision_boundary— how KNN carves up feature space (2D/PCA)
    3. plot_confusion_matrix — TP/FP/TN/FN heatmap
    4. plot_roc_curve        — ROC curve with AUC (binary)
    5. plot_distance_dist    — distribution of K-th neighbor distances
    6. plot_neighbor_purity  — purity bar chart per class
    7. plot_all              — 2×3 full dashboard

    Example
    -------
    >>> viz = KNNVisualizer(model, tuner, evaluator)  # doctest: +SKIP
    >>> viz.plot_all(X_test, y_test)                  # doctest: +SKIP
    """

    def __init__(
        self,
        model: KNNModel,
        tuner: KNNTuner = None,
        evaluator: KNNEvaluator = None,
    ):
        self.model     = model
        self.tuner     = tuner
        self.evaluator = evaluator
        self._palette  = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
                          "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]

    # ── 1. K Elbow Plot ───────────────────────────────────────────────────────

    def plot_k_elbow(self, ax=None):
        """
        Cross-validated accuracy vs K — the KNN elbow curve.

        WHAT TO LOOK FOR:
        ──────────────────
        - K=1 often has high training accuracy but lower CV accuracy (overfit)
        - Score typically peaks at the optimal K, then gradually declines
        - The 'elbow' — where the curve flattens — is the best K
        - Shaded band = ±1 std across folds (wider = more variance)

        Best K is marked with a vertical dashed line.
        """
        if self.tuner is None or not self.tuner.k_scores_:
            print("[KNNVisualizer] No tuner results. Run KNNTuner.fit() first.")
            return

        ks     = list(self.tuner.k_scores_.keys())
        scores = [self.tuner.k_scores_[k] for k in ks]
        stds   = [self.tuner.k_stds_[k]   for k in ks]
        scores_arr = np.array(scores)
        stds_arr   = np.array(stds)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(ks, scores_arr, color="#3498DB", linewidth=2.5,
                marker="o", markersize=5, label="CV Accuracy (mean)")
        ax.fill_between(ks,
                        scores_arr - stds_arr,
                        scores_arr + stds_arr,
                        alpha=0.15, color="#3498DB", label="±1 std")

        # Mark best K
        best_k = self.tuner.best_k_
        best_s = self.tuner.k_scores_[best_k]
        ax.axvline(best_k, color="#E74C3C", linestyle="--",
                   linewidth=1.8, label=f"Best K = {best_k}")
        ax.scatter([best_k], [best_s], color="#E74C3C", zorder=6, s=100)
        ax.annotate(f"K={best_k}\n{best_s:.3f}",
                    xy=(best_k, best_s), xytext=(best_k + 1.2, best_s - 0.015),
                    fontsize=8.5, color="#E74C3C",
                    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=0.8))

        ax.set_xlabel("K  (number of neighbors)")
        ax.set_ylabel(f"Cross-validated {self.tuner.scoring.capitalize()}")
        ax.set_title("KNN Elbow Curve — Finding the Optimal K")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ks[::2])

    # ── 2. Decision Boundary ──────────────────────────────────────────────────

    def plot_decision_boundary(self, X, y, resolution: float = 0.03, ax=None):
        """
        Visualize how KNN partitions the feature space into class regions.

        THE VORONOI-LIKE BOUNDARY:
        ───────────────────────────
        KNN's decision boundary is inherently non-linear. Each region of
        space is "owned" by the majority class among the K nearest training
        points. The result looks like a smooth version of Voronoi cells.

        Low K  → jagged, irregular boundary (complex, overfits)
        High K → smooth, rounded boundary  (simpler, more general)

        If features > 2: PCA is used to project to 2D for visualization.
        The boundary is approximate in that case.

        Parameters
        ----------
        resolution : grid resolution for the mesh (smaller = sharper, slower)
        """
        self.model._check_fitted()
        X = np.array(X)
        y = np.array(y).ravel()

        # Project to 2D if needed
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(
                self.model._scaler.transform(X) if self.model.scale_features else X
            )
            title_note = " (PCA projection — 2 components)"
            xlabel, ylabel = "PC 1", "PC 2"
            # Build a 2D KNN in PCA space for boundary visualization
            vis_model = KNeighborsClassifier(
                n_neighbors=self.model.n_neighbors,
                weights=self.model.weights,
                p=self.model.p,
            )
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            vis_model.fit(X_2d, y_enc)
            classes = le.classes_
        else:
            X_2d = (self.model._scaler.transform(X)
                    if self.model.scale_features else X)
            title_note = " (standardized)"
            xlabel = self.model.feature_names_[0]
            ylabel = self.model.feature_names_[1]
            vis_model = self.model._model
            classes = self.model.classes_
            y_enc = self.model._label_encoder.transform(y)

        n_classes = len(classes)
        colors_bg = [(*plt.cm.tab10(i / max(n_classes, 2))[:3], 0.25)
                     for i in range(n_classes)]
        colors_fg = [plt.cm.tab10(i / max(n_classes, 2)) for i in range(n_classes)]

        # Build mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, resolution),
            np.arange(y_min, y_max, resolution)
        )
        Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Draw filled regions
        cmap_light = ListedColormap([c for c in colors_bg])
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_light,
                    levels=np.arange(-0.5, n_classes + 0.5, 1))

        # Draw decision boundary lines
        ax.contour(xx, yy, Z, colors="white", linewidths=0.8, alpha=0.6,
                   levels=np.arange(-0.5, n_classes + 0.5, 1))

        # Scatter training points
        for i, cls in enumerate(classes):
            mask = y_enc == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       color=colors_fg[i], edgecolors="white",
                       s=40, linewidth=0.5, label=str(cls), alpha=0.85)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"KNN Decision Boundary  (K={self.model.n_neighbors}){title_note}")
        ax.legend(title="Classes", fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # ── 3. Confusion Matrix ───────────────────────────────────────────────────

    def plot_confusion_matrix(self, X, y, normalize: bool = False, ax=None):
        """
        Heatmap of the confusion matrix showing correct vs incorrect predictions.

        READING THE MATRIX:
          Perfect diagonal → perfect classifier
          Off-diagonal cells → where the model makes mistakes
          normalize=True → shows rates instead of counts, easier to compare
                           across classes of different sizes
        """
        y = np.array(y).ravel()
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=self.model.classes_)

        if normalize:
            cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            cm_show = cm
            fmt = "d"

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(cm_show, interpolation="nearest", cmap="YlOrRd")
        plt.colorbar(im, ax=ax)

        thresh = cm_show.max() / 2.0
        for i in range(cm_show.shape[0]):
            for j in range(cm_show.shape[1]):
                ax.text(j, i, f"{cm_show[i, j]:{fmt}}",
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if cm_show[i, j] > thresh else "black")

        ax.set_xticks(range(len(self.model.classes_)))
        ax.set_yticks(range(len(self.model.classes_)))
        ax.set_xticklabels(self.model.classes_, fontsize=9)
        ax.set_yticklabels(self.model.classes_, fontsize=9)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix  (K={self.model.n_neighbors})"
                     + (" — Normalized" if normalize else ""))

        if self.model.is_binary_:
            labels = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i - 0.35, labels[i][j],
                            ha="center", fontsize=8, color="grey", style="italic")

    # ── 4. ROC Curve ──────────────────────────────────────────────────────────

    def plot_roc_curve(self, X, y, ax=None):
        """
        ROC curve for binary KNN classification.

        NOTE ON KNN ROC CURVES:
        ───────────────────────
        KNN probabilities are COARSE — they can only take values
        k/K for k ∈ {0, 1, ..., K}. With K=5, probabilities are
        only 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.

        This makes the ROC curve stepped rather than smooth.
        Use weights='distance' for smoother probability estimates.
        """
        if not self.model.is_binary_:
            print("[KNNVisualizer] ROC curve shown for binary classification only.")
            return

        y = np.array(y).ravel()
        y_enc = self.model._label_encoder.transform(y)
        y_proba = self.model.predict_proba(X)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_enc, y_proba)
        auc = roc_auc_score(y_enc, y_proba)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.step(fpr, tpr, color="#E74C3C", linewidth=2.5,
                label=f"KNN  (K={self.model.n_neighbors}, AUC = {auc:.4f})",
                where="post")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
                linewidth=1.2, label="Random (AUC = 0.50)")
        ax.fill_between(fpr, tpr, step="post", alpha=0.1, color="#E74C3C")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve  (K={self.model.n_neighbors})\n"
                     f"Note: KNN ROC is stepped — probabilities are multiples of 1/K")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    # ── 5. Distance Distribution ──────────────────────────────────────────────

    def plot_distance_dist(self, X, y, ax=None):
        """
        Distribution of K-th nearest neighbor distances, split by
        whether the prediction was correct or incorrect.

        WHAT THIS TELLS YOU:
        ──────────────────────
        Correct predictions tend to have SHORTER K-th neighbor distances
        (the query point is well inside a cluster).

        Incorrect predictions tend to have LONGER distances
        (the query point is near a decision boundary or in sparse territory).

        If the two distributions heavily overlap → the distance threshold
        for "confident" vs "uncertain" predictions is blurry.
        """
        self.model._check_fitted()
        y = np.array(y).ravel()
        y_pred = self.model.predict(X)
        dists, _, _ = self.model.get_neighbors(X)
        kth_distances = dists[:, -1]   # distance to the K-th (farthest) neighbor

        correct   = kth_distances[y == y_pred]
        incorrect = kth_distances[y != y_pred]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        bins = np.histogram_bin_edges(kth_distances, bins=30)
        ax.hist(correct, bins=bins, alpha=0.6, color="#2ECC71",
                label=f"Correct  (n={len(correct)})", edgecolor="white")
        if len(incorrect) > 0:
            ax.hist(incorrect, bins=bins, alpha=0.6, color="#E74C3C",
                    label=f"Incorrect  (n={len(incorrect)})", edgecolor="white")

        ax.axvline(np.median(correct), color="#2ECC71", linestyle="--",
                   linewidth=1.5, label=f"Correct median = {np.median(correct):.3f}")
        if len(incorrect) > 0:
            ax.axvline(np.median(incorrect), color="#E74C3C", linestyle="--",
                       linewidth=1.5, label=f"Incorrect median = {np.median(incorrect):.3f}")

        ax.set_xlabel(f"Distance to K-th Neighbor  (K={self.model.n_neighbors})")
        ax.set_ylabel("Count")
        ax.set_title("K-th Neighbor Distance Distribution\n"
                     "Shorter distance → point is deep in a cluster → confident prediction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 6. Neighbor Purity Bar Chart ──────────────────────────────────────────

    def plot_neighbor_purity(self, X, y, ax=None):
        """
        Bar chart of neighbor purity per class.

        WHAT PURITY MEANS VISUALLY:
        ────────────────────────────
        A bar at 1.0 → every neighbor of every sample in that class
                       belongs to the same class. Perfect local separation.

        A bar at 0.5 → on average, only half the neighbors agree.
                       Heavy boundary overlap for that class.
        """
        ev = KNNEvaluator(self.model)
        purity_results = ev.neighbor_purity(X, y)

        classes  = list(purity_results["per_class"].keys())
        purities = [purity_results["per_class"][c] for c in classes]
        colors   = [self._palette[i % len(self._palette)] for i in range(len(classes))]

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(5, len(classes) * 1.5), 4))

        bars = ax.bar(classes, purities, color=colors, edgecolor="white", width=0.55)
        ax.axhline(purity_results["mean_purity"], color="black",
                   linestyle="--", linewidth=1.5,
                   label=f"Overall mean = {purity_results['mean_purity']:.4f}")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)

        for bar, p in zip(bars, purities):
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.015,
                    f"{p:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylim(0, 1.12)
        ax.set_xlabel("Class")
        ax.set_ylabel("Neighbor Purity")
        ax.set_title(f"Neighbor Purity by Class  (K={self.model.n_neighbors})\n"
                     f"Fraction of K neighbors sharing the same true class label")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    # ── 7. Full Dashboard ─────────────────────────────────────────────────────

    def plot_all(self, X, y, save_path: str = None):
        """
        Render a 2×3 KNN diagnostic dashboard.

        Layout:
          ┌──────────────┬──────────────┬──────────────┐
          │  K Elbow     │  Decision    │  Confusion   │
          │  Curve       │  Boundary    │  Matrix      │
          ├──────────────┼──────────────┼──────────────┤
          │  ROC Curve   │  Distance    │  Neighbor    │
          │              │  Distribution│  Purity      │
          └──────────────┴──────────────┴──────────────┘
        """
        metric_label = (
            "Euclidean" if self.model.p == 2
            else "Manhattan" if self.model.p == 1
            else self.model.metric
        )

        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(
            f"KNN Classification  —  Diagnostic Dashboard\n"
            f"K={self.model.n_neighbors}  |  weights='{self.model.weights}'  |  "
            f"metric={metric_label}  |  "
            f"task={'Binary' if self.model.is_binary_ else 'Multiclass'}",
            fontsize=13, fontweight="bold"
        )

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        self.plot_k_elbow(ax=ax1)
        self.plot_decision_boundary(X, y, ax=ax2)
        self.plot_confusion_matrix(X, y, ax=ax3)
        self.plot_roc_curve(X, y, ax=ax4)
        self.plot_distance_dist(X, y, ax=ax5)
        self.plot_neighbor_purity(X, y, ax=ax6)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[KNNVisualizer] Dashboard saved to: {save_path}")

        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 ── FULL PIPELINE
#  One function call: split → tune → fit → evaluate → explain → visualize
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X,
    y,
    feature_names=None,
    test_size: float = 0.2,
    auto_tune: bool = True,
    n_neighbors: int = 5,
    weights: str = "uniform",
    k_range: range = None,
    cv: int = 5,
    plot: bool = True,
    save_plot: str = None,
    explain_sample: bool = True,
    random_state: int = 42,
) -> dict:
    """
    End-to-end KNN classification pipeline in a single function call.

    PIPELINE STEPS:
    ───────────────
    1.  Stratified train/test split (preserves class ratios)
    2.  (Optional) Tune K, weights, and metric via stratified grid search
    3.  Fit the KNN model (memorize training data + build index)
    4.  Print model summary (config, dataset stats, complexity estimate)
    5.  Neighbor purity analysis (how clean are the local neighborhoods?)
    6.  Evaluate on train AND test sets (accuracy, F1, AUC, log-loss)
    7.  Stratified k-fold CV for robust performance estimate
    8.  Explain a sample prediction (trace to K nearest neighbors)
    9.  (Optional) Generate 6-panel diagnostic dashboard

    Parameters
    ----------
    X            : feature matrix (array or DataFrame)
    y            : class labels
    feature_names: column names (auto-detected from DataFrame)
    test_size    : fraction held out for testing (default 0.2)
    auto_tune    : search for optimal K, weights, and metric via GridSearchCV
    n_neighbors  : K to use if auto_tune=False
    weights      : 'uniform' or 'distance' (if auto_tune=False)
    k_range      : range of K values to search (default range(1, 31))
    cv           : number of folds for cross-validation
    plot         : display diagnostic plots
    save_plot    : path to save the plot image
    explain_sample: print a neighbor-level explanation for one test sample
    random_state : reproducibility seed

    Returns
    -------
    dict: model, tuner, evaluator, visualizer, train_metrics,
          test_metrics, cv_metrics, purity
    """
    print("\n" + "█" * 62)
    print("  K-NEAREST NEIGHBORS — FULL CLASSIFICATION PIPELINE")
    print("█" * 62)

    # ── 1. Prepare inputs
    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X = np.array(X)
    y = np.array(y).ravel()
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    n_classes = len(np.unique(y))
    default_k = max(1, int(np.sqrt(len(y))))
    if default_k % 2 == 0:
        default_k += 1

    print(f"\n[Pipeline] Dataset — {X.shape[0]} samples, {X.shape[1]} features, "
          f"{n_classes} classes: {np.unique(y)}")
    print(f"[Pipeline] Rule-of-thumb K = √n = {default_k}")

    # ── 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[Pipeline] Stratified split — Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Tune or use provided K
    k_range = k_range or range(1, min(31, len(X_train) // 2))
    tuner = KNNTuner(cv=cv, k_range=k_range, random_state=random_state)

    if auto_tune:
        print("\n[Pipeline] Step 1/5 — Hyperparameter Tuning ...")
        best_k = tuner.fit(X_train, y_train,
                           search_weights=True, search_metric=True)
        tuner.results_summary()
        n_neighbors = tuner.best_params_["n_neighbors"]
        weights     = tuner.best_params_["weights"]
        p           = tuner.best_params_["p"]
    else:
        print(f"\n[Pipeline] Step 1/5 — Using K={n_neighbors}, weights='{weights}'")
        p = 2
        # Still populate tuner k_scores for the elbow plot
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        tuner.best_k_ = n_neighbors
        tuner.best_params_ = {"n_neighbors": n_neighbors, "weights": weights, "p": p}
        for k in k_range:
            scores = cross_val_score(
                KNeighborsClassifier(n_neighbors=k, weights=weights),
                X_tr_sc, y_train, cv=skf, scoring="accuracy"
            )
            tuner.k_scores_[k] = scores.mean()
            tuner.k_stds_[k]   = scores.std()

    # ── 4. Fit
    print("\n[Pipeline] Step 2/5 — Fitting KNN model ...")
    model = KNNModel(n_neighbors=n_neighbors, weights=weights, p=p)
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    # ── 5. Neighbor purity
    print("\n[Pipeline] Step 3/5 — Neighbor Purity Analysis ...")
    evaluator = KNNEvaluator(model)
    purity = evaluator.neighbor_purity(X_test, y_test)

    # ── 6. Evaluate
    print("\n[Pipeline] Step 4/5 — Evaluating ...")
    train_metrics = evaluator.evaluate(X_train, y_train, label="Train Set")
    test_metrics  = evaluator.evaluate(X_test,  y_test,  label="Test Set")
    cv_metrics    = evaluator.cross_validate(X, y, cv=cv)

    # ── 7. Explain one prediction
    if explain_sample and len(X_test) > 0:
        print("\n[Pipeline] Sample Prediction Explanation (first test point):")
        model.explain_prediction(X_test[0], feature_names=feature_names)

    # ── 8. Plots
    visualizer = KNNVisualizer(model, tuner, evaluator)
    if plot:
        print("\n[Pipeline] Step 5/5 — Generating diagnostic plots ...")
        visualizer.plot_all(X_test, y_test, save_path=save_plot)

    print("\n" + "█" * 62)
    print("  PIPELINE COMPLETE")
    print("█" * 62 + "\n")

    return {
        "model":         model,
        "tuner":         tuner,
        "evaluator":     evaluator,
        "visualizer":    visualizer,
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "cv_metrics":    cv_metrics,
        "purity":        purity,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO — runs when you execute this file directly
#  python knn_module.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  KNN CLASSIFICATION MODULE — DEMO")
    print("  Using a synthetic binary classification dataset")
    print("=" * 62 + "\n")

    from sklearn.datasets import make_classification

    # Synthetic dataset: some natural clusters + noise, slight imbalance
    X_demo, y_demo = make_classification(
        n_samples      = 600,
        n_features     = 12,
        n_informative  = 6,
        n_redundant    = 2,
        n_classes      = 2,
        n_clusters_per_class = 2,   # two sub-clusters per class (realistic)
        weights        = [0.55, 0.45],
        flip_y         = 0.03,
        random_state   = 42,
    )
    y_demo = np.where(y_demo == 1, "Malignant", "Benign")
    feature_names_demo = [f"feature_{i:02d}" for i in range(12)]

    results = run_full_pipeline(
        X             = X_demo,
        y             = y_demo,
        feature_names = feature_names_demo,
        test_size     = 0.2,
        auto_tune     = True,
        k_range       = range(1, 25),
        cv            = 5,
        plot          = True,
        save_plot     = "knn_diagnostics.png",
        explain_sample= True,
    )

    model = results["model"]
    print("\n── How to use the fitted model in your application ──")
    print("  preds  = model.predict(X_new)")
    print("  probas = model.predict_proba(X_new)")
    print("  model.explain_prediction(X_new[0])  # trace any prediction to neighbors")
    print("  model.set_threshold is not available — adjust K or weights instead.")