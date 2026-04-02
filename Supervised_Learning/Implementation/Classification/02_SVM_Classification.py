"""
================================================================================
  SUPPORT VECTOR MACHINE (SVM) MODULE
  A complete, plug-and-play module for SVM Classification with detailed
  step-by-step breakdowns of every concept and line of code.
================================================================================

WHAT IS A SUPPORT VECTOR MACHINE?
───────────────────────────────────
  SVM is a supervised classification algorithm that finds the OPTIMAL
  decision boundary — called a HYPERPLANE — that separates classes with
  the MAXIMUM MARGIN.

  The core question SVM asks:
    "What is the widest possible 'street' I can draw between the two classes,
     such that no training points fall inside it?"

  The boundary is defined not by ALL training points (like KNN) or a
  learned probability (like Logistic Regression), but by a SMALL SUBSET
  of the most critical training points called SUPPORT VECTORS.

HOW SVM FINDS THE OPTIMAL HYPERPLANE:
───────────────────────────────────────
  In 2D, a hyperplane is just a line: w·x + b = 0
  In 3D, it's a flat plane.
  In higher dimensions, it's still called a hyperplane.

  SVM solves a CONSTRAINED OPTIMIZATION problem:

      MINIMIZE:    (1/2) ||w||²          ← maximize margin = 2/||w||
      SUBJECT TO:  yᵢ(w·xᵢ + b) ≥ 1     ← all points correctly classified

  The margin is the distance between the two parallel boundary lines:
      w·x + b = +1  (positive support vectors sit here)
      w·x + b = -1  (negative support vectors sit here)
      Width = 2 / ||w||

  Points that lie exactly ON these margin lines are the SUPPORT VECTORS.
  They are the only training points that matter — remove any other point
  and the decision boundary stays exactly the same.

HARD MARGIN vs SOFT MARGIN:
────────────────────────────
  Hard Margin: No training point can be inside the margin or misclassified.
               Only works when data is linearly separable.

  Soft Margin: Allows some violations via SLACK VARIABLES (ξᵢ ≥ 0):
      MINIMIZE:  (1/2)||w||² + C · Σξᵢ
      The C parameter controls the tradeoff:
        Large C  → small margin, few violations, more complex boundary
        Small C  → large margin, more violations allowed, simpler boundary

THE KERNEL TRICK — HANDLING NON-LINEAR DATA:
─────────────────────────────────────────────
  Most real data is NOT linearly separable. SVM handles this with KERNELS.

  Instead of finding a linear boundary in the original feature space,
  the kernel implicitly maps data to a HIGHER-DIMENSIONAL space where
  a linear boundary CAN separate the classes.

  The magic: we never explicitly compute the high-dimensional coordinates.
  We only compute dot products in the original space using the kernel function.
  This is called the KERNEL TRICK.

  Available kernels:
    Linear   : K(x,y) = x·y                       No transformation
    RBF      : K(x,y) = exp(-γ||x-y||²)            Gaussian bell curve
               Most powerful and widely used default
    Polynomial: K(x,y) = (γ·x·y + r)^d             Polynomial surface
    Sigmoid  : K(x,y) = tanh(γ·x·y + r)            Neural net-like

  RBF KERNEL INTUITION:
    γ (gamma) controls how far a single training point's influence reaches:
      Large γ → narrow bell → only very close points matter → complex boundary
      Small γ → wide bell  → far-away points matter too   → smoother boundary

HOW SVM COMPARES TO YOUR PREVIOUS MODELS:
───────────────────────────────────────────
  Logistic Regression  → Linear boundary, probabilistic, fast
  KNN                  → Non-linear, no explicit boundary, lazy learner
  Naive Bayes          → Probabilistic, generative, assumes independence
  SVM                  → Margin-maximizing, kernel-powered, memory-efficient
                         Only stores support vectors (not all training data)

CONTENTS:
─────────
  1. SVMModel          — core SVM classifier with kernel selection
  2. SVMTuner          — C and gamma tuning via stratified grid search
  3. SVMEvaluator      — accuracy, F1, AUC, confusion matrix, full report
  4. SVMVisualizer     — margin/boundary, support vectors, kernel comparison,
                         ROC curve, confusion matrix, full dashboard
  5. run_full_pipeline()— one-call: tune → fit → evaluate → visualize

HOW TO USE:
───────────
  from svm_module import SVMModel, run_full_pipeline

  model = SVMModel(kernel='rbf', C=1.0)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  model.summary()

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
from matplotlib.lines import Line2D

from sklearn.svm import SVC
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
#
#  WHAT HAPPENS WHEN SVM TRAINS:
#  ──────────────────────────────
#  1. Features are standardized (critical — SVM uses distances)
#  2. The kernel function computes pairwise similarity between all training
#     points (the Gram matrix / kernel matrix)
#  3. A quadratic programming solver finds the optimal w and b by:
#       a. Identifying which points lie on or inside the margin (candidates
#          for being support vectors)
#       b. Maximizing the margin width 2/||w|| subject to the C constraint
#  4. Non-zero coefficients (α) correspond to SUPPORT VECTORS
#  5. All other points are irrelevant to the decision boundary
# ══════════════════════════════════════════════════════════════════════════════

class SVMModel:
    """
    Support Vector Machine Classifier — full-featured, production-ready.

    THE CORE TRADEOFF (controlled by C and kernel/gamma):
    ──────────────────────────────────────────────────────
    Larger C  → penalize misclassifications more heavily
                → smaller margin, more support vectors
                → fits training data closely (risk of overfitting)
    Smaller C → allow more misclassifications
                → larger margin, fewer support vectors
                → simpler, more general boundary (risk of underfitting)

    Parameters
    ----------
    kernel : str, default='rbf'
        The kernel function to use. This is the most impactful choice:
          'linear'  → straight-line boundary, interpretable coefficients
                      Use when data is linearly separable or high-dimensional
          'rbf'     → Radial Basis Function (Gaussian kernel), default
                      Use for most classification problems — very flexible
          'poly'    → Polynomial boundary of degree `degree`
                      Use when you expect polynomial relationships
          'sigmoid' → Similar to neural network activation function

    C : float, default=1.0
        Regularization parameter (inverse of regularization strength).
        C = 1/λ — larger C → less regularization → tighter fit.
        Equivalent to how C works in Logistic Regression.

    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
          'scale' → 1 / (n_features × X.var())  ← sklearn default, recommended
          'auto'  → 1 / n_features
          float   → exact gamma value to use
        Larger gamma → more complex, wiggly boundary (potential overfit)
        Smaller gamma → smoother, simpler boundary

    degree : int, default=3
        Degree of the polynomial kernel. Only used when kernel='poly'.

    decision_function_shape : str, default='ovr'
        Multiclass strategy:
          'ovr' → One-vs-Rest: train one SVM per class vs all others
          'ovo' → One-vs-One: train one SVM per class pair (n*(n-1)/2 SVMs)

    probability : bool, default=True
        If True, enables probability estimates via Platt scaling.
        Required for ROC curves and probability outputs.
        Slight overhead at training time — disable if only hard labels needed.

    scale_features : bool, default=True
        Always leave True — SVM is EXTREMELY sensitive to feature scale
        because it computes distances between points. An unscaled feature
        with range [0, 10000] will dominate the kernel computation entirely.

    Example
    -------
    >>> model = SVMModel(kernel='rbf', C=1.0, gamma='scale')  # doctest: +SKIP
    >>> model.fit(X_train, y_train)                           # doctest: +SKIP
    >>> preds = model.predict(X_test)                         # doctest: +SKIP
    >>> model.summary()                                       # doctest: +SKIP
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma="scale",
        degree: int = 3,
        decision_function_shape: str = "ovr",
        probability: bool = True,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        self.kernel                  = kernel
        self.C                       = C
        self.gamma                   = gamma
        self.degree                  = degree
        self.decision_function_shape = decision_function_shape
        self.probability             = probability
        self.scale_features          = scale_features
        self.random_state            = random_state

        self._model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            decision_function_shape=decision_function_shape,
            probability=probability,
            random_state=random_state,
        )
        self._scaler        = StandardScaler() if scale_features else None
        self._label_encoder = LabelEncoder()

        # Populated during fit()
        self.feature_names_  = None
        self.classes_        = None
        self.is_binary_      = None
        self.X_train_scaled_ = None
        self.y_train_enc_    = None
        self.is_fitted_      = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        """
        Train the SVM — find the optimal hyperplane and support vectors.

        STEP-BY-STEP INTERNAL PROCESS:
        ────────────────────────────────
        Step 1 — Input validation and label encoding.
                 Converts string labels ('Spam'/'Ham') to integers (1/0).
                 SVM internally works with numeric labels.

        Step 2 — Feature standardization.
                 CRITICAL for SVM. The RBF kernel computes:
                     K(x, y) = exp(-γ ||x - y||²)
                 If feature A has range [0, 1000] and feature B has [0, 1],
                 then ||x - y||² is completely dominated by feature A.
                 StandardScaler ensures each feature contributes equally.

        Step 3 — Kernel matrix computation (internally by sklearn).
                 For n training samples, sklearn computes an n×n matrix
                 where entry [i,j] = K(xᵢ, xⱼ).
                 This encodes similarity between every pair of training points.

        Step 4 — Quadratic programming optimization.
                 Finds the dual variables αᵢ (one per training sample).
                 Most αᵢ = 0  → those points are NOT support vectors
                 Some αᵢ > 0  → those points ARE support vectors
                 The boundary is: f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b

        Step 5 — (If probability=True) Platt scaling calibration.
                 Fits a logistic regression on top of SVM scores to
                 produce calibrated probability estimates.

        Parameters
        ----------
        X            : array-like (n_samples, n_features)
        y            : array-like (n_samples,) — class labels
        feature_names: optional list of column names
        """
        X = np.array(X)
        y = np.array(y).ravel()

        # Step 1 — Feature names and label encoding
        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"x{i}" for i in range(X.shape[1])]
        )
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_   = self._label_encoder.classes_
        self.is_binary_ = len(self.classes_) == 2

        # Step 2 — Standardize features
        X_scaled = self._scaler.fit_transform(X) if self.scale_features else X

        # Step 3-5 — Solve the QP problem, find support vectors
        self._model.fit(X_scaled, y_enc)

        # Store for visualization
        self.X_train_scaled_ = X_scaled
        self.y_train_enc_    = y_enc
        self.is_fitted_      = True

        n_sv   = self._model.support_vectors_.shape[0]
        n_train = X.shape[0]
        sv_pct  = n_sv / n_train * 100

        mode = "Binary" if self.is_binary_ else f"Multiclass ({len(self.classes_)} classes)"
        print(
            f"[SVMModel] Fitted — {mode}\n"
            f"  Kernel          : {self.kernel}\n"
            f"  C               : {self.C}  "
            f"({'Tighter fit' if self.C > 1 else 'Wider margin'})\n"
            f"  Gamma           : {self.gamma}\n"
            f"  Training samples: {n_train}\n"
            f"  Support Vectors : {n_sv} / {n_train}  ({sv_pct:.1f}% of training data)\n"
            f"  Classes         : {list(self.classes_)}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Classify new samples using the trained hyperplane.

        HOW PREDICTION WORKS:
        ──────────────────────
        For a new point x, SVM computes the decision function:

            f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b

        where the sum is ONLY over support vectors (αᵢ > 0 for those).
        All other training points have αᵢ = 0 and contribute nothing.

        The SIGN of f(x) determines the class:
            f(x) > 0 → positive class
            f(x) < 0 → negative class
            f(x) = 0 → exactly on the decision boundary

        The MAGNITUDE of f(x) indicates confidence:
            |f(x)| >> 0 → far from boundary, high confidence
            |f(x)| ≈ 0  → near the boundary, low confidence

        Returns
        -------
        y_pred : array of original class labels (decoded)
        """
        self._check_fitted()
        X = np.array(X)
        X_scaled = self._scaler.transform(X) if self.scale_features else X
        y_enc = self._model.predict(X_scaled)
        return self._label_encoder.inverse_transform(y_enc)

    # ── predict_proba ─────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Return calibrated probability estimates.

        HOW PROBABILITIES WORK IN SVM (Platt Scaling):
        ────────────────────────────────────────────────
        SVM is NOT inherently probabilistic — it produces a decision
        function score f(x), not a probability.

        When probability=True, sklearn applies PLATT SCALING:
          1. Run cross-validation on training data
          2. Fit a logistic regression: P(y=1|f(x)) = 1 / (1 + exp(A·f(x) + B))
          3. Parameters A and B calibrate the SVM score to a [0,1] probability

        This is why fitting with probability=True takes longer —
        it runs an additional cross-validation step internally.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        if not self.probability:
            raise RuntimeError(
                "Set probability=True when initializing SVMModel to enable probabilities."
            )
        X = np.array(X)
        X_scaled = self._scaler.transform(X) if self.scale_features else X
        return self._model.predict_proba(X_scaled)

    # ── decision_function ─────────────────────────────────────────────────────

    def decision_function(self, X):
        """
        Return the raw SVM decision function scores.

        This is the un-calibrated f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b

        Unlike predict_proba (which calibrates via Platt scaling),
        decision_function returns the raw geometric distance from
        the decision boundary. Useful for:
          - Understanding model confidence without Platt scaling overhead
          - Plotting margin boundaries
          - Ranking predictions by certainty

        Returns
        -------
        scores : ndarray of shape (n_samples,) for binary,
                 (n_samples, n_classes) for multiclass
        """
        self._check_fitted()
        X = np.array(X)
        X_scaled = self._scaler.transform(X) if self.scale_features else X
        return self._model.decision_function(X_scaled)

    # ── get_support_vectors ───────────────────────────────────────────────────

    def get_support_vectors(self):
        """
        Return the support vectors — the ONLY training points that define
        the decision boundary.

        WHY SUPPORT VECTORS MATTER:
        ────────────────────────────
        This is one of SVM's most elegant properties. Of all n training
        samples, only a small subset (the support vectors) determine the
        decision boundary. You could remove every other training point and
        the model would be identical.

        Support vectors are the points that:
          - Lie exactly ON the margin (distance = 1/||w|| from boundary)
          - Lie INSIDE the margin (margin violations, when C < ∞)
          - Are MISCLASSIFIED (only possible with soft-margin SVM, C < ∞)

        Returns
        -------
        dict with:
          'vectors'      : scaled feature values of support vectors
          'indices'      : training set indices of support vectors
          'n_per_class'  : count of support vectors per class
          'pct_of_train' : percentage of training data that are support vectors
        """
        self._check_fitted()
        sv        = self._model.support_vectors_
        sv_idx    = self._model.support_
        n_sv_cls  = self._model.n_support_

        return {
            "vectors":       sv,
            "indices":       sv_idx,
            "n_per_class":   dict(zip(self.classes_, n_sv_cls)),
            "pct_of_train":  len(sv_idx) / len(self.y_train_enc_) * 100,
        }

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print a structured model summary covering:
          - Task type and kernel configuration
          - Support vector statistics (the heart of SVM)
          - Decision boundary description
          - Linear kernel coefficients (if kernel='linear')
          - Guidance on interpreting C and gamma
        """
        self._check_fitted()
        sv_info = self.get_support_vectors()
        n_sv    = len(sv_info["indices"])
        n_train = len(self.y_train_enc_)

        sep = "=" * 65
        print(f"\n{sep}")
        print("  SVM MODEL SUMMARY")
        print(sep)
        print(f"  Task             : {'Binary' if self.is_binary_ else 'Multiclass'} Classification")
        print(f"  Classes          : {list(self.classes_)}")

        print(f"\n  ── Kernel Configuration ──────────────────────────────")
        print(f"  Kernel           : {self.kernel}")
        print(f"  C (regularization): {self.C}  "
              f"({'Tight fit — small margin' if self.C > 5 else 'Balanced' if self.C > 0.5 else 'Wide margin — more violations'})")

        if self.kernel in ("rbf", "poly", "sigmoid"):
            gamma_val = (
                1 / (len(self.feature_names_) * self.X_train_scaled_.var())
                if self.gamma == "scale"
                else 1 / len(self.feature_names_)
                if self.gamma == "auto"
                else self.gamma
            )
            print(f"  Gamma            : {self.gamma}  (≈ {gamma_val:.6f})")
            if self.kernel == "rbf":
                print(f"  Kernel formula   : K(x,y) = exp(-γ ||x-y||²)")
            elif self.kernel == "poly":
                print(f"  Degree           : {self.degree}")
                print(f"  Kernel formula   : K(x,y) = (γ·x·y + 1)^{self.degree}")

        print(f"\n  ── Support Vectors (the boundary-defining points) ────")
        print(f"  Total SVs        : {n_sv} / {n_train}  "
              f"({sv_info['pct_of_train']:.1f}% of training data)")
        for cls, cnt in sv_info["n_per_class"].items():
            bar = "█" * int(cnt / n_train * 40) + "░" * (40 - int(cnt / n_train * 40))
            print(f"  {str(cls):<20} {cnt} SVs  {bar}")

        if n_sv / n_train > 0.5:
            sv_note = "Many SVs → boundary is complex (consider larger C or different kernel)"
        elif n_sv / n_train > 0.2:
            sv_note = "Moderate SVs → good balance of complexity and generalization"
        else:
            sv_note = "Few SVs → clean separation, simple boundary, likely good generalization"
        print(f"\n  Interpretation   : {sv_note}")

        # Linear kernel: show coefficients
        if self.kernel == "linear":
            print(f"\n  ── Decision Boundary Coefficients (linear kernel) ───")
            print(f"  w (hyperplane normal vector) — sorted by |magnitude|:")
            coef = self._model.coef_[0] if self.is_binary_ else self._model.coef_
            if self.is_binary_:
                df = pd.DataFrame({
                    "Feature":    self.feature_names_,
                    "Weight (w)": coef,
                    "|Weight|":   np.abs(coef),
                }).sort_values("|Weight|", ascending=False)
                print(f"  {'Feature':<22}  {'Weight':>10}  Importance")
                for _, row in df.iterrows():
                    bar = "█" * int(abs(row["Weight (w)"]) / df["|Weight|"].max() * 20)
                    sign = "+" if row["Weight (w)"] >= 0 else "-"
                    print(f"  {row['Feature']:<22}  {row['Weight (w)']:>10.5f}  {sign}{bar}")

        print(f"\n  ── Key Takeaways ─────────────────────────────────────")
        print(f"  • Only the {n_sv} support vectors define the boundary")
        print(f"  • Remove any other training point → model unchanged")
        print(f"  • Use .get_support_vectors() to inspect them")
        print(f"  • Use .decision_function(X) to get raw margin scores")
        print(sep + "\n")

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(X, y) first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 ── HYPERPARAMETER TUNER
#
#  THE TWO CRITICAL HYPERPARAMETERS FOR RBF SVM:
#  ───────────────────────────────────────────────
#  C and gamma interact with each other in a 2D grid:
#
#  High C, High γ → Very complex, wiggly boundary → likely overfit
#  High C, Low  γ → Complex but smooth boundary
#  Low  C, High γ → Simple but sensitive boundary
#  Low  C, Low  γ → Very smooth, simple boundary → likely underfit
#
#  We search a LOG-SCALE grid because both parameters span many orders
#  of magnitude (0.001 to 1000 is a typical search range).
# ══════════════════════════════════════════════════════════════════════════════

class SVMTuner:
    """
    Stratified grid search tuner for SVM hyperparameters (C and gamma).

    WHY LOG-SCALE SEARCH:
    ──────────────────────
    Both C and gamma are multiplicative in their effect. Going from
    C=1 to C=10 has a much larger impact than going from C=100 to C=110.
    Log-scale sampling [0.001, 0.01, 0.1, 1, 10, 100] covers this range
    efficiently — each step multiplies by 10.

    WHY STRATIFIED CV:
    ───────────────────
    Each CV fold must preserve the class distribution. If one class is
    rare (5%), a regular fold might have no examples of it in validation.
    StratifiedKFold guarantees the ratio is maintained in every fold.

    Example
    -------
    >>> tuner = SVMTuner(cv=5)                                # doctest: +SKIP
    >>> best_C, best_gamma = tuner.fit(X_train, y_train)      # doctest: +SKIP
    >>> tuner.results_summary()                               # doctest: +SKIP
    """

    def __init__(
        self,
        cv: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
    ):
        self.cv           = cv
        self.scoring      = scoring
        self.random_state = random_state

        self.best_params_  = None
        self.best_score_   = None
        self.cv_results_   = None
        self._grid_search  = None

    def fit(self, X, y, param_grid: dict = None):
        """
        Run stratified grid search over C, gamma, and kernel.

        STEP-BY-STEP:
        ─────────────
        Step 1 — Scale features (must be done before CV to get correct
                 gamma='scale' values, though re-fitting inside CV folds
                 is the leak-free approach — we use Pipeline for that).

        Step 2 — Define parameter grid. Default is a log-scale grid
                 over C=[0.1, 1, 10, 100] × gamma=['scale','auto',0.01,0.1,1]
                 for RBF kernel. Also tests linear kernel.

        Step 3 — GridSearchCV runs all combinations with StratifiedKFold.
                 Each combination gets cv mean scores across folds.
                 Uses n_jobs=-1 for parallel execution.

        Step 4 — Best parameters identified by scoring metric.
                 Default: roc_auc for binary, roc_auc_ovr for multiclass.

        Parameters
        ----------
        X          : feature matrix
        y          : class labels
        param_grid : custom param grid (optional)

        Returns
        -------
        (best_C, best_gamma) : tuple
        """
        X = np.array(X)
        y = np.array(y).ravel()

        n_classes = len(np.unique(y))

        # Use Pipeline so scaling is done inside each fold (no leakage)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(probability=True, random_state=self.random_state)),
        ])

        if param_grid is None:
            param_grid = [
                # RBF kernel — most important to search
                {
                    "svm__kernel": ["rbf"],
                    "svm__C":      [0.1, 1, 10, 100],
                    "svm__gamma":  ["scale", "auto", 0.01, 0.1, 1.0],
                },
                # Linear kernel — fewer params, sometimes best for high-dim data
                {
                    "svm__kernel": ["linear"],
                    "svm__C":      [0.1, 1, 10, 100],
                },
            ]

        n_combos = sum(
            np.prod([len(v) for v in p.values()])
            for p in param_grid
        )

        scoring = (
            "roc_auc_ovr" if n_classes > 2 else self.scoring
        )

        print(f"[SVMTuner] Grid search over {n_combos} combinations "
              f"with {self.cv}-fold stratified CV ...")
        print(f"  Scoring metric : {scoring}")
        print(f"  Note: SVM grid search can be slow — using n_jobs=-1 (all cores)")

        skf = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        self._grid_search = GridSearchCV(
            pipe,
            param_grid,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        self._grid_search.fit(X, y)

        self.best_params_ = self._grid_search.best_params_
        self.best_score_  = self._grid_search.best_score_
        self.cv_results_  = pd.DataFrame(self._grid_search.cv_results_)

        best_C     = self.best_params_.get("svm__C")
        best_gamma = self.best_params_.get("svm__gamma", "scale")
        best_kernel = self.best_params_.get("svm__kernel")

        print(f"[SVMTuner] ✓ Best: kernel={best_kernel}, "
              f"C={best_C}, gamma={best_gamma}\n"
              f"  Best {scoring}: {self.best_score_:.4f}")

        return best_C, best_gamma

    def results_summary(self):
        """Print best parameters and top-5 combinations."""
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")

        print("\n" + "=" * 60)
        print("  SVM TUNING RESULTS")
        print("=" * 60)
        print(f"  Best kernel  : {self.best_params_.get('svm__kernel')}")
        print(f"  Best C       : {self.best_params_.get('svm__C')}")
        print(f"  Best gamma   : {self.best_params_.get('svm__gamma', 'N/A')}")
        print(f"  Best score   : {self.best_score_:.4f}")

        print(f"\n  Top 5 combinations:")
        top5 = self.cv_results_.nlargest(5, "mean_test_score")[
            ["param_svm__kernel", "param_svm__C",
             "param_svm__gamma", "mean_test_score", "std_test_score"]
        ]
        for _, row in top5.iterrows():
            marker = " ← BEST" if (
                row.get("param_svm__C") == self.best_params_.get("svm__C") and
                row.get("param_svm__kernel") == self.best_params_.get("svm__kernel")
            ) else ""
            print(f"    kernel={row.get('param_svm__kernel', '?'):<8} "
                  f"C={str(row.get('param_svm__C','?')):<6} "
                  f"gamma={str(row.get('param_svm__gamma','?')):<8} "
                  f"score={row['mean_test_score']:.4f} "
                  f"± {row['std_test_score']:.4f}{marker}")
        print("=" * 60 + "\n")

    def get_best_model(self) -> "SVMModel":
        """Return a pre-configured SVMModel with the best hyperparameters."""
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        return SVMModel(
            kernel=self.best_params_.get("svm__kernel", "rbf"),
            C=self.best_params_.get("svm__C", 1.0),
            gamma=self.best_params_.get("svm__gamma", "scale"),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 ── EVALUATOR
#  Same rich classification metrics as previous modules, plus
#  SVM-specific diagnostics: support vector analysis and margin scoring.
# ══════════════════════════════════════════════════════════════════════════════

class SVMEvaluator:
    """
    Comprehensive classification evaluator for SVM.

    METRICS (consistent with Logistic Regression and KNN Classification modules):
    ───────────────────────────────────────────────────────────────────────────────
    Accuracy    : % all predictions correct
    Precision   : of predicted positives, how many were truly positive
    Recall      : of actual positives, how many did we catch
    F1 Score    : harmonic mean of precision and recall
    ROC-AUC     : area under ROC curve (threshold-independent)

    SVM-SPECIFIC:
    ─────────────
    Decision scores are used alongside Platt-calibrated probabilities
    to give insight into margin-based confidence.

    Example
    -------
    >>> ev = SVMEvaluator(model)             # doctest: +SKIP
    >>> ev.evaluate(X_test, y_test)          # doctest: +SKIP
    >>> ev.cross_validate(X, y, cv=5)        # doctest: +SKIP
    """

    def __init__(self, model: SVMModel):
        self.model    = model
        self.metrics_ = {}

    def evaluate(self, X, y, label: str = "Test Set") -> dict:
        """
        Compute all classification metrics for a given (X, y) split.

        Returns
        -------
        dict of metric name → value
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y, y_pred, average="weighted", zero_division=0)
        ll   = log_loss(y, y_prob)

        try:
            if self.model.is_binary_:
                auc = roc_auc_score(
                    self.model._label_encoder.transform(y),
                    y_prob[:, 1]
                )
            else:
                auc = roc_auc_score(
                    self.model._label_encoder.transform(y),
                    y_prob,
                    multi_class="ovr", average="weighted"
                )
        except Exception:
            auc = float("nan")

        self.metrics_ = dict(
            Accuracy=acc, Precision=prec, Recall=rec,
            F1=f1, ROC_AUC=auc, Log_Loss=ll
        )

        sep = "=" * 58
        print(f"\n{sep}")
        print(f"  EVALUATION — {label}  (kernel={self.model.kernel}, C={self.model.C})")
        print(sep)
        print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}% correct)")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  ROC-AUC    : {auc:.4f}  (1.0 = perfect, 0.5 = random)")
        print(f"  Log Loss   : {ll:.4f}")
        report   = classification_report(y, y_pred, zero_division=0)
        indented = "\n".join("    " + l for l in report.splitlines())
        print(f"\n  Classification Report:\n{indented}")
        print(sep + "\n")
        return self.metrics_

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Stratified k-fold cross-validation using a pipeline (no data leakage).

        WHY PIPELINES MATTER FOR SVM:
        ──────────────────────────────
        If you scale the full dataset before CV, the scaler has seen the
        validation data during fit — that's data leakage. The Pipeline ensures
        StandardScaler.fit() is called only on training folds.

        Returns
        -------
        dict of mean/std for accuracy, F1, ROC-AUC
        """
        X = np.array(X)
        y = np.array(y).ravel()

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(
                kernel=self.model.kernel,
                C=self.model.C,
                gamma=self.model.gamma,
                probability=True,
                random_state=self.model.random_state,
            ))
        ])

        skf        = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        f1_scores  = cross_val_score(pipe, X, y, cv=skf, scoring="f1_weighted")

        try:
            scoring    = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr_weighted"
            auc_scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring)
        except Exception:
            auc_scores = np.array([float("nan")] * cv)

        results = {
            "Accuracy_mean": acc_scores.mean(), "Accuracy_std": acc_scores.std(),
            "F1_mean":       f1_scores.mean(),  "F1_std":       f1_scores.std(),
            "AUC_mean":      auc_scores.mean(), "AUC_std":      auc_scores.std(),
        }

        print(f"\n{'='*58}")
        print(f"  {cv}-FOLD STRATIFIED CROSS-VALIDATION  "
              f"(kernel={self.model.kernel}, C={self.model.C})")
        print(f"{'='*58}")
        print(f"  Accuracy  mean ± std : {results['Accuracy_mean']:.4f} ± {results['Accuracy_std']:.4f}")
        print(f"  F1        mean ± std : {results['F1_mean']:.4f} ± {results['F1_std']:.4f}")
        print(f"  ROC-AUC   mean ± std : {results['AUC_mean']:.4f} ± {results['AUC_std']:.4f}")
        print(f"{'='*58}\n")
        return results

    def support_vector_analysis(self) -> dict:
        """
        Analyze the support vectors — SVM's unique diagnostic.

        WHAT THIS TELLS YOU:
        ──────────────────────
        The percentage of training points that are support vectors reveals
        how 'hard' the classification problem is:

          Very few SVs (< 10%) → classes are well-separated, clear boundary
          Moderate SVs (10-30%) → some overlap, realistic scenario
          Many SVs (> 50%)     → heavy class overlap, consider:
                                  - Different kernel
                                  - More features / feature engineering
                                  - Different algorithm altogether

        Returns
        -------
        dict with sv_count, sv_pct, n_per_class
        """
        self.model._check_fitted()
        sv_info = self.model.get_support_vectors()
        n_sv    = len(sv_info["indices"])
        n_train = len(self.model.y_train_enc_)

        print(f"\n{'='*55}")
        print(f"  SUPPORT VECTOR ANALYSIS")
        print(f"{'='*55}")
        print(f"  Total SVs      : {n_sv} of {n_train} training samples")
        print(f"  SV percentage  : {sv_info['pct_of_train']:.1f}%")
        print(f"\n  SVs per class:")
        for cls, cnt in sv_info["n_per_class"].items():
            bar = "█" * int(cnt / max(sv_info["n_per_class"].values()) * 25)
            print(f"    {str(cls):<15} {cnt:>4} SVs  {bar}")

        if sv_info["pct_of_train"] < 15:
            interp = "Excellent separation — very few boundary cases"
        elif sv_info["pct_of_train"] < 35:
            interp = "Good separation — normal amount of boundary complexity"
        else:
            interp = "Heavy overlap — boundary is complex, consider tuning"

        print(f"\n  Interpretation : {interp}")
        print(f"{'='*55}\n")
        return sv_info


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 ── VISUALIZER
#  Six SVM-specific diagnostic plots including the hallmark visualization:
#  the decision boundary with margin and support vectors highlighted.
# ══════════════════════════════════════════════════════════════════════════════

class SVMVisualizer:
    """
    Diagnostic plots specifically designed for SVM classification.

    THE SIGNATURE SVM PLOT — Decision Boundary with Margin:
    ────────────────────────────────────────────────────────
    Unlike all previous modules, SVM has a unique visualization:
    the margin bands (dashed lines at ±1 from the boundary) and
    the support vectors highlighted with rings.

    This shows exactly what SVM optimizes — maximizing the gap between
    the two dashed margin lines while keeping support vectors on/within them.

    PLOTS:
    ──────
    1. plot_decision_boundary  — boundary + margin + support vectors (2D/PCA)
    2. plot_support_vectors    — which training points are support vectors
    3. plot_confusion_matrix   — TP/FP/TN/FN heatmap
    4. plot_roc_curve          — ROC curve (binary)
    5. plot_c_sensitivity      — how accuracy changes with different C values
    6. plot_kernel_comparison  — same data with 4 different kernels side by side
    7. plot_all                — 2×3 full dashboard

    Example
    -------
    >>> viz = SVMVisualizer(model, tuner, evaluator)  # doctest: +SKIP
    >>> viz.plot_all(X_test, y_test)                  # doctest: +SKIP
    """

    def __init__(
        self,
        model: SVMModel,
        tuner: SVMTuner = None,
        evaluator: SVMEvaluator = None,
    ):
        self.model     = model
        self.tuner     = tuner
        self.evaluator = evaluator
        self._palette  = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
                          "#9B59B6", "#1ABC9C"]

    # ── 1. Decision Boundary with Margin and Support Vectors ──────────────────

    def plot_decision_boundary(self, X, y, resolution: float = 0.04, ax=None):
        """
        THE DEFINITIVE SVM VISUALIZATION — shows exactly what SVM optimizes.

        WHAT YOU WILL SEE:
        ──────────────────
        Colored regions    : class prediction zones
        Solid line (center): the decision boundary (f(x) = 0)
        Dashed lines       : the margin boundaries (f(x) = ±1)
                             The 'street' that SVM tries to widen
        Circled points     : SUPPORT VECTORS — the training points that
                             define the boundary. These are the ONLY points
                             that matter. All others could be removed
                             without changing the decision boundary.

        The width of the 'street' between the dashed lines = 2/||w||
        This is what SVM maximizes (subject to C constraint).

        If features > 2: PCA projects to 2D for visualization.
        The boundary shown is approximate in that case.

        Parameters
        ----------
        resolution : grid step size (smaller = crisper, slower)
        """
        self.model._check_fitted()
        X = np.array(X)
        y = np.array(y).ravel()

        # Project to 2D if needed
        if X.shape[1] > 2:
            pca  = PCA(n_components=2, random_state=42)
            X_sc = self.model._scaler.transform(X) if self.model.scale_features else X
            X_2d = pca.fit_transform(X_sc)

            # Train a fresh SVM in 2D PCA space for visualization only
            le    = LabelEncoder()
            y_enc = le.fit_transform(y)
            vis   = SVC(
                kernel=self.model.kernel, C=self.model.C,
                gamma=self.model.gamma, probability=False,
                random_state=self.model.random_state,
            )
            vis.fit(X_2d, y_enc)
            classes     = le.classes_
            sv_mask     = np.zeros(len(X_2d), dtype=bool)
            sv_mask[vis.support_] = True
            xlabel, ylabel = "PC 1", "PC 2"
            title_sfx = " (PCA projection)"
        else:
            X_sc    = self.model._scaler.transform(X) if self.model.scale_features else X
            X_2d    = X_sc
            vis     = self.model._model
            classes = self.model.classes_
            y_enc   = self.model._label_encoder.transform(y)

            # Mark which test points are support vectors of the trained model
            sv_idx  = self.model._model.support_
            # Map training SVs to scaled space; find closest test points
            sv_mask = np.zeros(len(X_2d), dtype=bool)
            xlabel  = self.model.feature_names_[0] + " (scaled)"
            ylabel  = self.model.feature_names_[1] + " (scaled)"
            title_sfx = ""

        n_cls   = len(classes)
        cmap_bg = [(*plt.cm.tab10(i / max(n_cls, 2))[:3], 0.20) for i in range(n_cls)]
        cmap_fg = [plt.cm.tab10(i / max(n_cls, 2)) for i in range(n_cls)]

        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, resolution),
            np.arange(y_min, y_max, resolution)
        )
        grid  = np.c_[xx.ravel(), yy.ravel()]
        Z     = vis.predict(grid).reshape(xx.shape)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Filled class regions
        cmap_light = ListedColormap([c for c in cmap_bg])
        ax.contourf(xx, yy, Z, alpha=0.30, cmap=cmap_light,
                    levels=np.arange(-0.5, n_cls + 0.5, 1))

        # Decision boundary (f=0) and margin lines (f=±1)
        try:
            scores = vis.decision_function(grid)
            if scores.ndim == 1:
                scores_2d = scores.reshape(xx.shape)
                ax.contour(xx, yy, scores_2d, levels=[0],
                           colors="black", linewidths=2.0,
                           linestyles="-")                   # decision boundary
                ax.contour(xx, yy, scores_2d, levels=[-1, 1],
                           colors="black", linewidths=1.2,
                           linestyles="--")                  # margin boundaries
        except Exception:
            pass

        # Training points
        for i, cls in enumerate(classes):
            mask = y_enc == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       color=cmap_fg[i], edgecolors="white",
                       s=40, linewidth=0.5,
                       label=str(cls), alpha=0.85, zorder=3)

        # Highlight support vectors with rings
        if X.shape[1] <= 2:
            sv_X   = self.model.X_train_scaled_[self.model._model.support_]
            ax.scatter(sv_X[:, 0], sv_X[:, 1],
                       s=150, facecolors="none",
                       edgecolors="black", linewidths=1.8,
                       label=f"Support Vectors ({len(sv_X)})", zorder=4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"SVM Decision Boundary + Margin{title_sfx}\n"
            f"Kernel={self.model.kernel}  C={self.model.C}  "
            f"━ boundary  ╌ margin (±1)"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # ── 2. Support Vector Scatter ─────────────────────────────────────────────

    def plot_support_vectors(self, ax=None):
        """
        Scatter plot highlighting which training points are support vectors.

        SVM'S UNIQUE PROPERTY:
        ───────────────────────
        Unlike Logistic Regression (which depends on ALL training points)
        or KNN (which stores ALL training points), SVM's decision boundary
        is defined ONLY by the circled support vectors.

        Everything else is irrelevant to the final model.

        This plot shows (projected to 2D via PCA if needed):
          Filled points  = normal training points (don't affect the boundary)
          Circled points = support vectors (define everything)
        """
        self.model._check_fitted()
        X_sc  = self.model.X_train_scaled_
        y_enc = self.model.y_train_enc_
        sv_idx = self.model._model.support_

        # Project to 2D if needed
        if X_sc.shape[1] > 2:
            pca  = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_sc)
            xlabel, ylabel = "PC 1 (scaled)", "PC 2 (scaled)"
        else:
            X_2d   = X_sc
            xlabel = self.model.feature_names_[0] + " (scaled)"
            ylabel = self.model.feature_names_[1] + " (scaled)"

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        classes = self.model.classes_
        for i, cls in enumerate(classes):
            mask    = y_enc == i
            sv_mask = np.zeros(len(y_enc), dtype=bool)
            sv_mask[sv_idx] = True

            # Non-support vector points (faded)
            non_sv = mask & ~sv_mask
            ax.scatter(X_2d[non_sv, 0], X_2d[non_sv, 1],
                       color=self._palette[i], alpha=0.25, s=25,
                       edgecolors="none",
                       label=f"{cls} (non-SV, n={non_sv.sum()})")

            # Support vector points (bright + ringed)
            sv_cls = mask & sv_mask
            if sv_cls.sum() > 0:
                ax.scatter(X_2d[sv_cls, 0], X_2d[sv_cls, 1],
                           color=self._palette[i], alpha=0.9, s=80,
                           edgecolors="black", linewidths=1.5,
                           label=f"{cls} Support Vectors (n={sv_cls.sum()})",
                           zorder=5)

        n_sv  = len(sv_idx)
        n_all = len(y_enc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Support Vectors Highlighted\n"
            f"{n_sv}/{n_all} training points are SVs "
            f"({n_sv/n_all*100:.1f}%) — only these define the boundary"
        )
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)

    # ── 3. Confusion Matrix ───────────────────────────────────────────────────

    def plot_confusion_matrix(self, X, y, normalize: bool = False, ax=None):
        """
        Heatmap of TP/FP/TN/FN — identical structure to previous modules.

        For SVM specifically: misclassified points are typically those
        that fall on the wrong side of the margin boundary — points that
        SVM allowed as margin violations (the ξᵢ slack variables).
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)
        cm     = confusion_matrix(y, y_pred, labels=self.model.classes_)

        if normalize:
            cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            cm_show = cm
            fmt = "d"

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(cm_show, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        thresh = cm_show.max() / 2.0
        for i in range(cm_show.shape[0]):
            for j in range(cm_show.shape[1]):
                ax.text(j, i, f"{cm_show[i,j]:{fmt}}",
                        ha="center", va="center", fontsize=12,
                        fontweight="bold",
                        color="white" if cm_show[i, j] > thresh else "black")

        ax.set_xticks(range(len(self.model.classes_)))
        ax.set_yticks(range(len(self.model.classes_)))
        ax.set_xticklabels(self.model.classes_, fontsize=9)
        ax.set_yticklabels(self.model.classes_, fontsize=9)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix  (SVM, kernel={self.model.kernel})"
                     + (" — Normalized" if normalize else ""))

        if self.model.is_binary_:
            labels = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i - 0.35, labels[i][j],
                            ha="center", fontsize=8,
                            color="grey", style="italic")

    # ── 4. ROC Curve ──────────────────────────────────────────────────────────

    def plot_roc_curve(self, X, y, ax=None):
        """
        ROC curve using Platt-calibrated SVM probabilities.

        NOTE ON SVM ROC CURVES:
        ────────────────────────
        Unlike KNN (coarse, stepped ROC) or Logistic Regression (smooth sigmoid),
        SVM probabilities come from Platt scaling — a logistic regression
        fitted on the raw SVM decision scores. This produces smooth ROC curves
        similar to Logistic Regression.

        The raw decision_function scores (without Platt scaling) often give
        slightly BETTER AUC than the calibrated probabilities — they are
        a better ranking function even if not calibrated probabilities.
        We show both when possible.
        """
        if not self.model.is_binary_:
            print("[SVMVisualizer] ROC curve shown for binary classification only.")
            return

        y     = np.array(y).ravel()
        y_enc = self.model._label_encoder.transform(y)

        # Platt-calibrated probabilities
        y_prob = self.model.predict_proba(X)[:, 1]
        fpr_p, tpr_p, _ = roc_curve(y_enc, y_prob)
        auc_p = roc_auc_score(y_enc, y_prob)

        # Raw decision scores (often higher AUC)
        y_dec = self.model.decision_function(X)
        if y_dec.ndim > 1:
            y_dec = y_dec[:, 1]
        fpr_d, tpr_d, _ = roc_curve(y_enc, y_dec)
        auc_d = roc_auc_score(y_enc, y_dec)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(fpr_p, tpr_p, color="#E74C3C", linewidth=2.5,
                label=f"Platt probabilities  (AUC = {auc_p:.4f})")
        ax.plot(fpr_d, tpr_d, color="#3498DB", linewidth=2,
                linestyle="--",
                label=f"Decision scores  (AUC = {auc_d:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":",
                linewidth=1.2, label="Random (AUC = 0.50)")
        ax.fill_between(fpr_p, tpr_p, alpha=0.08, color="#E74C3C")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — SVM (kernel={self.model.kernel}, C={self.model.C})")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    # ── 5. C Sensitivity Analysis ─────────────────────────────────────────────

    def plot_c_sensitivity(self, X, y, cv: int = 5, ax=None):
        """
        Show how model performance changes as C varies.

        THE C TRADEOFF VISUALIZED:
        ───────────────────────────
        This plot shows train vs validation accuracy across a log-scale
        range of C values. The characteristic pattern:

          Left side (small C):
            Training accuracy low → model underfits (margin too wide)
            Validation accuracy low → generalizes poorly

          Right side (large C):
            Training accuracy high → model fits training data tightly
            Validation accuracy may drop → overfitting (margin too narrow)

          The sweet spot: where train and validation curves are both high
          and close together (minimum gap = minimum overfitting).

          The vertical dashed line marks the C value currently in the model.
        """
        X = np.array(X)
        y = np.array(y).ravel()

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        c_values     = np.logspace(-3, 3, 20)
        train_scores = []
        val_scores   = []

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for c in c_values:
            m = SVC(kernel=self.model.kernel, C=c,
                    gamma=self.model.gamma, probability=False,
                    random_state=42)
            tr_sc = cross_val_score(m, X_scaled, y, cv=skf,
                                    scoring="accuracy")
            # Train accuracy on the full scaled set for illustration
            m.fit(X_scaled, y)
            tr_full = accuracy_score(y, m.predict(X_scaled))
            train_scores.append(tr_full)
            val_scores.append(tr_sc.mean())

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.semilogx(c_values, train_scores, color="#E74C3C", linewidth=2,
                    marker="o", markersize=4, label="Training accuracy")
        ax.semilogx(c_values, val_scores, color="#2ECC71", linewidth=2,
                    marker="o", markersize=4, label=f"CV validation accuracy ({cv}-fold)")
        ax.fill_between(c_values, train_scores, val_scores,
                        alpha=0.1, color="#F39C12",
                        label="Overfitting gap")

        ax.axvline(self.model.C, color="#3498DB", linestyle="--",
                   linewidth=1.8, label=f"Current C={self.model.C}")

        ax.set_xlabel("C  (log scale)  →  Less regularization →")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"C Sensitivity Analysis  (kernel={self.model.kernel})\n"
                     f"Small C = wide margin | Large C = tight fit")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    # ── 6. Kernel Comparison ──────────────────────────────────────────────────

    def plot_kernel_comparison(self, X, y, ax=None):
        """
        Side-by-side decision boundaries for all 4 SVM kernels on the same data.

        WHY THIS IS USEFUL:
        ────────────────────
        The same data can look very different under different kernels:

          Linear   → straight-line boundary (1 hyperplane)
          RBF      → curved, elliptical regions — most flexible
          Poly(3)  → cubic curves — can model complex shapes
          Sigmoid  → S-shaped curve — similar to neural network

        Use this plot to visually pick the right kernel for your data shape.

        Uses PCA if features > 2.
        """
        X = np.array(X)
        y = np.array(y).ravel()

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        if X_sc.shape[1] > 2:
            pca  = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_sc)
        else:
            X_2d = X_sc

        le    = LabelEncoder()
        y_enc = le.fit_transform(y)
        kernels = [("linear", {}),
                   ("rbf",    {}),
                   ("poly",   {"degree": 3}),
                   ("sigmoid",{})]

        if ax is None:
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            fig.suptitle(
                "SVM Kernel Comparison — Same Data, 4 Different Decision Boundaries",
                fontsize=12, fontweight="bold"
            )
        else:
            axes = [ax]
            kernels = kernels[:1]

        x_min, x_max = X_2d[:, 0].min() - .5, X_2d[:, 0].max() + .5
        y_min, y_max = X_2d[:, 1].min() - .5, X_2d[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                              np.arange(y_min, y_max, 0.05))

        n_cls  = len(le.classes_)
        colors = [plt.cm.tab10(i / max(n_cls, 2)) for i in range(n_cls)]

        for ax_k, (kern, kw) in zip(axes, kernels):
            clf = SVC(kernel=kern, C=self.model.C,
                      gamma=self.model.gamma,
                      probability=False,
                      random_state=42, **kw)
            clf.fit(X_2d, y_enc)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax_k.contourf(xx, yy, Z, alpha=0.25, cmap="tab10",
                          levels=np.arange(-0.5, n_cls + 0.5, 1))
            ax_k.contour(xx, yy, Z, colors="black", linewidths=0.8,
                         levels=np.arange(-0.5, n_cls + 0.5, 1))

            for i, cls in enumerate(le.classes_):
                mask = y_enc == i
                ax_k.scatter(X_2d[mask, 0], X_2d[mask, 1],
                             color=colors[i], edgecolors="white",
                             s=25, alpha=0.8)
            sv_X = X_2d[clf.support_]
            ax_k.scatter(sv_X[:, 0], sv_X[:, 1],
                         s=100, facecolors="none",
                         edgecolors="black", linewidths=1.2, zorder=4)

            acc = accuracy_score(y_enc, clf.predict(X_2d))
            n_sv = len(clf.support_)
            deg_note = f" (d={kw['degree']})" if kern == "poly" else ""
            ax_k.set_title(f"{kern.capitalize()}{deg_note}\n"
                           f"Acc={acc:.3f}  SVs={n_sv}",
                           fontsize=9)
            ax_k.set_xlabel("PC 1")
            ax_k.set_ylabel("PC 2" if ax_k == axes[0] else "")
            ax_k.set_xlim(x_min, x_max)
            ax_k.set_ylim(y_min, y_max)

        if ax is None:
            plt.tight_layout()
            plt.show()

    # ── 7. Full Dashboard ─────────────────────────────────────────────────────

    def plot_all(self, X, y, save_path: str = None):
        """
        Render the full 2×3 SVM diagnostic dashboard.

        Layout:
          ┌──────────────────┬──────────────────┬──────────────────┐
          │ Decision Boundary│ Support Vectors  │ Confusion Matrix │
          │ + Margin + SVs   │ Highlighted      │                  │
          ├──────────────────┼──────────────────┼──────────────────┤
          │ ROC Curve        │ C Sensitivity    │ Summary Stats    │
          │ (Platt + raw)    │ (train vs val)   │ Text Panel       │
          └──────────────────┴──────────────────┴──────────────────┘
        """
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(
            f"SVM Classification — Diagnostic Dashboard\n"
            f"kernel='{self.model.kernel}'  |  C={self.model.C}  |  "
            f"gamma={self.model.gamma}  |  "
            f"{'Binary' if self.model.is_binary_ else 'Multiclass'}",
            fontsize=13, fontweight="bold"
        )

        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        self.plot_decision_boundary(X, y, ax=ax1)
        self.plot_support_vectors(ax=ax2)
        self.plot_confusion_matrix(X, y, ax=ax3)
        self.plot_roc_curve(X, y, ax=ax4)
        self.plot_c_sensitivity(X, y, ax=ax5)

        # Panel 6 — model summary stats
        sv_info = self.model.get_support_vectors()
        y_pred  = self.model.predict(X)
        y_arr   = np.array(y).ravel()
        acc     = accuracy_score(y_arr, y_pred)
        f1      = f1_score(y_arr, y_pred, average="weighted", zero_division=0)

        try:
            y_enc = self.model._label_encoder.transform(y_arr)
            y_prb = self.model.predict_proba(X)[:, 1]
            auc   = roc_auc_score(y_enc, y_prb)
        except Exception:
            auc = float("nan")

        ax6.axis("off")
        text = (
            f"  MODEL CONFIGURATION\n"
            f"  {'─'*30}\n"
            f"  Kernel           :  {self.model.kernel}\n"
            f"  C                :  {self.model.C}\n"
            f"  Gamma            :  {self.model.gamma}\n"
            f"  Probability      :  {self.model.probability}\n\n"
            f"  SUPPORT VECTORS\n"
            f"  {'─'*30}\n"
            f"  Total SVs        :  {len(sv_info['indices'])}\n"
            f"  SV %             :  {sv_info['pct_of_train']:.1f}%\n"
            + "".join(
                f"  {str(cls):<15} :  {cnt} SVs\n"
                for cls, cnt in sv_info["n_per_class"].items()
            ) +
            f"\n  PERFORMANCE\n"
            f"  {'─'*30}\n"
            f"  Accuracy         :  {acc:.4f}\n"
            f"  F1 (weighted)    :  {f1:.4f}\n"
            f"  ROC-AUC          :  {auc:.4f}\n"
            f"\n  DATASET\n"
            f"  {'─'*30}\n"
            f"  Train samples    :  {len(self.model.y_train_enc_)}\n"
            f"  Features         :  {len(self.model.feature_names_)}\n"
        )
        ax6.text(
            0.04, 0.97, text,
            transform=ax6.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#F0F4F8", edgecolor="#BDC3C7", linewidth=1.2)
        )
        ax6.set_title("Summary", fontsize=10, pad=8)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[SVMVisualizer] Dashboard saved to: {save_path}")

        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 ── FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X,
    y,
    feature_names=None,
    test_size: float = 0.2,
    auto_tune: bool = True,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma="scale",
    cv: int = 5,
    plot: bool = True,
    save_plot: str = None,
    random_state: int = 42,
) -> dict:
    """
    End-to-end SVM classification pipeline in a single function call.

    PIPELINE STEPS:
    ───────────────
    1.  Stratified train/test split (preserves class ratios)
    2.  (Optional) Tune C, gamma, and kernel via stratified GridSearchCV
    3.  Fit the SVM (solve QP, find support vectors)
    4.  Print model summary (SVs, kernel, configuration)
    5.  Support vector analysis
    6.  Evaluate on train AND test sets (accuracy, F1, AUC)
    7.  Stratified k-fold cross-validation
    8.  Generate 6-panel diagnostic dashboard

    Parameters
    ----------
    X            : feature matrix (array or DataFrame)
    y            : class labels
    feature_names: column names (auto-detected from DataFrame)
    test_size    : fraction held out for testing
    auto_tune    : search for best C, gamma, kernel via GridSearchCV
    kernel       : SVM kernel (used if auto_tune=False)
    C            : regularization strength (used if auto_tune=False)
    gamma        : kernel coefficient (used if auto_tune=False)
    cv           : number of CV folds
    plot         : display diagnostic dashboard
    save_plot    : path to save the dashboard
    random_state : reproducibility seed

    Returns
    -------
    dict: model, tuner, evaluator, visualizer,
          train_metrics, test_metrics, cv_metrics
    """
    print("\n" + "█" * 62)
    print("  SUPPORT VECTOR MACHINE — FULL CLASSIFICATION PIPELINE")
    print("█" * 62)

    # ── 1. Prepare
    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X = np.array(X)
    y = np.array(y).ravel()
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    n_classes = len(np.unique(y))
    print(f"\n[Pipeline] Dataset — {X.shape[0]} samples, "
          f"{X.shape[1]} features, {n_classes} classes: {np.unique(y)}")

    # ── 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[Pipeline] Stratified split — Train: {len(X_train)}, "
          f"Test: {len(X_test)}")

    # ── 3. Tune
    tuner = SVMTuner(cv=cv, random_state=random_state)
    if auto_tune:
        print("\n[Pipeline] Step 1/5 — Hyperparameter Tuning ...")
        best_C, best_gamma = tuner.fit(X_train, y_train)
        tuner.results_summary()
        C      = tuner.best_params_.get("svm__C", C)
        gamma  = tuner.best_params_.get("svm__gamma", gamma)
        kernel = tuner.best_params_.get("svm__kernel", kernel)
    else:
        print(f"\n[Pipeline] Step 1/5 — Using kernel={kernel}, "
              f"C={C}, gamma={gamma}")

    # ── 4. Fit
    print("\n[Pipeline] Step 2/5 — Fitting SVM ...")
    model = SVMModel(
        kernel=kernel, C=C, gamma=gamma,
        random_state=random_state
    )
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    # ── 5. SV analysis
    evaluator = SVMEvaluator(model)
    print("\n[Pipeline] Step 3/5 — Support Vector Analysis ...")
    evaluator.support_vector_analysis()

    # ── 6. Evaluate
    print("\n[Pipeline] Step 4/5 — Evaluating ...")
    train_metrics = evaluator.evaluate(X_train, y_train, label="Train Set")
    test_metrics  = evaluator.evaluate(X_test,  y_test,  label="Test Set")
    cv_metrics    = evaluator.cross_validate(X, y, cv=cv)

    # ── 7. Plot
    visualizer = SVMVisualizer(model, tuner, evaluator)
    if plot:
        print("\n[Pipeline] Step 5/5 — Generating diagnostic dashboard ...")
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
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  SVM CLASSIFICATION MODULE — DEMO")
    print("  Synthetic binary classification with natural clusters")
    print("=" * 62 + "\n")

    from sklearn.datasets import make_classification

    X_demo, y_demo = make_classification(
        n_samples=500,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.55, 0.45],
        flip_y=0.03,
        random_state=42,
    )
    y_demo = np.where(y_demo == 1, "Malignant", "Benign")
    feature_names_demo = [f"feature_{i:02d}" for i in range(12)]

    results = run_full_pipeline(
        X             = X_demo,
        y             = y_demo,
        feature_names = feature_names_demo,
        test_size     = 0.2,
        auto_tune     = True,
        cv            = 5,
        plot          = True,
        save_plot     = "svm_diagnostics.png",
    )

    model = results["model"]
    sv    = model.get_support_vectors()

    print("\n── SVM Key Facts ──────────────────────────────")
    print(f"  Kernel used    : {model.kernel}")
    print(f"  Best C         : {model.C}")
    print(f"  Support Vectors: {len(sv['indices'])} "
          f"({sv['pct_of_train']:.1f}% of training data)")
    print(f"  SVs per class  : {sv['n_per_class']}")
    print("\n── Usage in your application ──────────────────")
    print("  preds  = model.predict(X_new)")
    print("  probas = model.predict_proba(X_new)   # Platt-calibrated")
    print("  scores = model.decision_function(X_new)  # raw margin scores")
    print("  svs    = model.get_support_vectors()  # inspect the boundary")