"""
================================================================================
  ELASTIC NET REGRESSION MODULE
  A complete, plug-and-play module for Elastic Net regularization
================================================================================

WHAT IS ELASTIC NET?
--------------------
Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties into one cost function:

    Loss = RSS + λ [ α·Σ|βᵢ|  +  (1-α)/2 · Σβᵢ² ]
                    ↑ L1 term       ↑ L2 term

  - α (l1_ratio): controls the blend. 1.0 = pure Lasso, 0.0 = pure Ridge
  - λ (alpha):    controls overall regularization strength

HOW TO USE THIS MODULE
----------------------
  from elastic_net_module import ElasticNetModel

  model = ElasticNetModel()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  model.summary()

CONTENTS
--------
  1. ElasticNetModel         — main class wrapping sklearn ElasticNet
  2. ElasticNetTuner         — hyperparameter search via cross-validation
  3. ElasticNetEvaluator     — evaluation metrics + diagnostics
  4. ElasticNetVisualizer    — plotting helpers (coefficients, residuals, etc.)
  5. run_full_pipeline()     — convenience function: fit, tune, evaluate, plot

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

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 ── CORE MODEL CLASS
#  Wraps sklearn's ElasticNet with auto-scaling, feature names, and a
#  human-readable summary so you can drop it into any project easily.
# ══════════════════════════════════════════════════════════════════════════════

class ElasticNetModel:
    """
    A ready-to-use Elastic Net Regression model.

    Parameters
    ----------
    alpha : float, default=1.0
        Overall regularization strength (λ).
        Larger → stronger regularization → smaller / sparser coefficients.

    l1_ratio : float, default=0.5
        Mix between L1 and L2 penalty.
          0.0  →  pure Ridge  (no feature selection)
          1.0  →  pure Lasso  (aggressive feature selection)
          0.5  →  equal blend (recommended starting point)

    scale_features : bool, default=True
        Whether to standardize features before fitting.
        STRONGLY recommended — both L1 and L2 penalties are sensitive to scale.

    max_iter : int, default=10_000
        Max coordinate-descent iterations.

    random_state : int, default=42
        Seed for reproducibility.

    Example
    -------
    >>> model = ElasticNetModel(alpha=0.1, l1_ratio=0.7)  # doctest: +SKIP
    >>> model.fit(X_train, y_train)                        # doctest: +SKIP
    >>> preds = model.predict(X_test)                      # doctest: +SKIP
    >>> model.summary()                                    # doctest: +SKIP
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        scale_features: bool = True,
        max_iter: int = 10_000,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scale_features = scale_features
        self.max_iter = max_iter
        self.random_state = random_state

        # Internals — populated during fit()
        self._scaler = StandardScaler() if scale_features else None
        self._model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.feature_names_ = None
        self.is_fitted_ = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        """
        Train the model on (X, y).

        STEP-BY-STEP WHAT HAPPENS INTERNALLY:
        ──────────────────────────────────────
        1. Convert input to numpy arrays for uniform handling.
        2. (Optional) Standardize X so each feature has mean=0, std=1.
           This is critical: the penalty treats all coefficients equally only
           when features are on the same scale.
        3. Run coordinate descent to minimize the Elastic Net loss:
               RSS + λ [ α·Σ|βᵢ| + (1-α)/2·Σβᵢ² ]
           Coordinate descent cycles through each coefficient one at a time,
           finding its optimal value while holding all others fixed.
        4. Store feature names for reporting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
        """
        # -- 1. Convert to arrays
        X = np.array(X)
        y = np.array(y).ravel()

        # -- 2. Store feature names
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        elif hasattr(X, "columns"):                   # DataFrame passed
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]

        # -- 3. Scale
        X_ready = self._scaler.fit_transform(X) if self.scale_features else X

        # -- 4. Fit the Elastic Net model
        self._model.fit(X_ready, y)
        self.is_fitted_ = True

        n_zero = np.sum(self._model.coef_ == 0)
        print(
            f"[ElasticNetModel] Fitted on {X.shape[0]} samples, "
            f"{X.shape[1]} features.\n"
            f"  alpha={self.alpha}, l1_ratio={self.l1_ratio}\n"
            f"  Zeroed-out coefficients: {n_zero}/{X.shape[1]}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Generate predictions for new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.array(X)
        X_ready = self._scaler.transform(X) if self.scale_features else X
        return self._model.predict(X_ready)

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print a structured summary of the fitted model including:
          - Hyperparameters
          - Intercept
          - All coefficients (sorted by absolute value)
          - Count of active (non-zero) vs eliminated features
        """
        self._check_fitted()
        coefs = self._model.coef_
        intercept = self._model.intercept_

        df = pd.DataFrame({
            "Feature":    self.feature_names_,
            "Coefficient": coefs,
            "Abs Value":  np.abs(coefs),
            "Status":     ["ACTIVE" if c != 0 else "zeroed-out" for c in coefs],
        }).sort_values("Abs Value", ascending=False)

        sep = "=" * 60
        print(f"\n{sep}")
        print("  ELASTIC NET MODEL SUMMARY")
        print(sep)
        print(f"  alpha (λ)  : {self.alpha}")
        print(f"  l1_ratio   : {self.l1_ratio}  "
              f"({'Lasso-like' if self.l1_ratio > 0.7 else 'Ridge-like' if self.l1_ratio < 0.3 else 'Balanced'})")
        print(f"  Intercept  : {intercept:.6f}")
        print(f"  Features   : {len(coefs)} total | "
              f"{(coefs != 0).sum()} active | "
              f"{(coefs == 0).sum()} zeroed-out")
        print(f"\n  {'Feature':<25} {'Coefficient':>15}  Status")
        print(f"  {'-'*25} {'-'*15}  {'-'*10}")
        for _, row in df.iterrows():
            print(f"  {row['Feature']:<25} {row['Coefficient']:>15.6f}  {row['Status']}")
        print(sep + "\n")

    # ── get_coefficients ──────────────────────────────────────────────────────

    def get_coefficients(self) -> pd.DataFrame:
        """Return a DataFrame of feature names and their coefficients."""
        self._check_fitted()
        return pd.DataFrame({
            "feature":     self.feature_names_,
            "coefficient": self._model.coef_,
        }).sort_values("coefficient", key=abs, ascending=False)

    # ── private helpers ───────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 ── HYPERPARAMETER TUNER
#  Finding the right alpha and l1_ratio is crucial.
#  This class uses cross-validated search to find the best combination.
# ══════════════════════════════════════════════════════════════════════════════

class ElasticNetTuner:
    """
    Cross-validation based hyperparameter tuner for Elastic Net.

    WHY TUNE?
    ---------
    Both alpha and l1_ratio dramatically affect model performance:
      - Too large alpha  → underfitting (over-penalized)
      - Too small alpha  → overfitting  (under-penalized)
      - Wrong l1_ratio   → wrong balance between sparsity and grouping

    HOW IT WORKS:
    -------------
    1. Build a grid of (alpha, l1_ratio) combinations.
    2. For each combination, run k-fold cross-validation.
    3. Pick the pair with the best average validation score (lowest MSE).

    Alternatively, use ElasticNetCV from sklearn which does this efficiently
    along a regularization path (much faster than brute-force grid search).

    Example
    -------
    >>> tuner = ElasticNetTuner(cv=5)                              # doctest: +SKIP
    >>> best_alpha, best_l1 = tuner.fit(X_train, y_train)         # doctest: +SKIP
    >>> tuner.results_summary()                                    # doctest: +SKIP
    """

    def __init__(self, cv: int = 5, n_alphas: int = 50, random_state: int = 42):
        self.cv = cv
        self.n_alphas = n_alphas
        self.random_state = random_state
        self.best_alpha_ = None
        self.best_l1_ratio_ = None
        self._cv_model = None

    def fit(self, X, y, l1_ratios=None):
        """
        Run cross-validated search over alpha and l1_ratio.

        STEP-BY-STEP:
        ─────────────
        1. Define candidate l1_ratio values to try.
        2. sklearn's ElasticNetCV automatically searches over alpha values
           (using the 'regularization path' — efficient vs brute force).
        3. For each l1_ratio, n_alphas alpha values are tested.
        4. The combination with the lowest cross-validated MSE is selected.

        Parameters
        ----------
        X : array-like
        y : array-like
        l1_ratios : list of float, optional
            Defaults to [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

        Returns
        -------
        (best_alpha, best_l1_ratio) : tuple
        """
        if l1_ratios is None:
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

        X = np.array(X)
        y = np.array(y).ravel()

        # Standardize before CV (fit scaler only on training folds inside CV
        # would be more rigorous — but for tuning purposes this is standard)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"[ElasticNetTuner] Searching {self.n_alphas} alphas × "
              f"{len(l1_ratios)} l1_ratios with {self.cv}-fold CV ...")

        self._cv_model = ElasticNetCV(
            l1_ratio=l1_ratios,
            n_alphas=self.n_alphas,
            cv=KFold(n_splits=self.cv, shuffle=True,
                     random_state=self.random_state),
            max_iter=10_000,
            n_jobs=-1,              # use all CPU cores
        )
        self._cv_model.fit(X_scaled, y)

        self.best_alpha_    = self._cv_model.alpha_
        self.best_l1_ratio_ = self._cv_model.l1_ratio_

        print(f"[ElasticNetTuner] ✓ Best alpha={self.best_alpha_:.6f}, "
              f"l1_ratio={self.best_l1_ratio_:.2f}")

        return self.best_alpha_, self.best_l1_ratio_

    def results_summary(self):
        """Print the best hyperparameters found."""
        if self.best_alpha_ is None:
            raise RuntimeError("Call .fit() first.")
        print("\n" + "=" * 50)
        print("  TUNING RESULTS")
        print("=" * 50)
        print(f"  Best alpha (λ)  : {self.best_alpha_:.6f}")
        print(f"  Best l1_ratio   : {self.best_l1_ratio_:.4f}")
        print(f"  Interpretation  : "
              f"{'Lasso-dominant' if self.best_l1_ratio_ > 0.7 else 'Ridge-dominant' if self.best_l1_ratio_ < 0.3 else 'Balanced blend'}")
        print("=" * 50 + "\n")

    def get_best_model(self) -> "ElasticNetModel":
        """Return a pre-configured ElasticNetModel with the best hyperparameters."""
        if self.best_alpha_ is None:
            raise RuntimeError("Call .fit() first.")
        return ElasticNetModel(alpha=self.best_alpha_, l1_ratio=self.best_l1_ratio_)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 ── EVALUATOR
#  A consistent set of metrics to understand how well the model performs.
# ══════════════════════════════════════════════════════════════════════════════

class ElasticNetEvaluator:
    """
    Computes and reports a comprehensive set of regression metrics.

    METRICS EXPLAINED:
    ──────────────────
    MSE   (Mean Squared Error)      — average squared residual. Penalizes
                                      large errors heavily. Unit: target²
    RMSE  (Root MSE)                — same as MSE but in original units.
                                      Easy to compare to target's std dev.
    MAE   (Mean Absolute Error)     — average absolute residual. Robust to
                                      outliers. Same unit as target.
    R²    (Coefficient of Det.)     — % variance explained. 1.0 = perfect,
                                      0.0 = as good as predicting the mean,
                                      <0 = worse than the mean.
    Adj.R²                          — R² penalized for extra features to
                                      prevent inflated scores.

    Example
    -------
    >>> ev = ElasticNetEvaluator(model)          # doctest: +SKIP
    >>> ev.evaluate(X_test, y_test)              # doctest: +SKIP
    >>> ev.cross_validate(X, y, cv=5)            # doctest: +SKIP
    """

    def __init__(self, model: ElasticNetModel):
        self.model = model
        self.metrics_ = {}

    def evaluate(self, X, y, label: str = "Test Set") -> dict:
        """
        Compute all metrics for a given (X, y) split.

        Parameters
        ----------
        X     : features
        y     : true targets
        label : display label (e.g., "Train Set", "Test Set")

        Returns
        -------
        dict of metric name → value
        """
        y = np.array(y).ravel()
        y_pred = self.model.predict(X)
        n, p = np.array(X).shape

        mse  = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y, y_pred)
        r2   = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        self.metrics_ = dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2, Adj_R2=adj_r2)
        self._residuals = y - y_pred
        self._y_pred = y_pred
        self._y_true = y

        print(f"\n{'='*50}")
        print(f"  EVALUATION — {label}")
        print(f"{'='*50}")
        print(f"  MSE       : {mse:.4f}")
        print(f"  RMSE      : {rmse:.4f}")
        print(f"  MAE       : {mae:.4f}")
        print(f"  R²        : {r2:.4f}  ({r2*100:.1f}% variance explained)")
        print(f"  Adj. R²   : {adj_r2:.4f}")
        print(f"{'='*50}\n")
        return self.metrics_

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Evaluate with k-fold cross-validation to get robust performance estimates.

        WHY CV?
        ───────
        A single train/test split can be lucky or unlucky depending on how the
        data is divided. Cross-validation averages over k different splits,
        giving a much more reliable picture of real-world performance.

        Parameters
        ----------
        X  : full feature matrix
        y  : full target vector
        cv : number of folds

        Returns
        -------
        dict with mean and std of each metric across folds
        """
        X = np.array(X)
        y = np.array(y).ravel()

        # Build a pipeline so scaling is done correctly inside each fold
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=self.model.alpha,
                l1_ratio=self.model.l1_ratio,
                max_iter=10_000,
            ))
        ])

        r2_scores  = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        mse_scores = -cross_val_score(pipe, X, y, cv=cv,
                                      scoring="neg_mean_squared_error")

        cv_results = {
            "R2_mean":   r2_scores.mean(),
            "R2_std":    r2_scores.std(),
            "MSE_mean":  mse_scores.mean(),
            "MSE_std":   mse_scores.std(),
            "RMSE_mean": np.sqrt(mse_scores.mean()),
        }

        print(f"\n{'='*50}")
        print(f"  {cv}-FOLD CROSS-VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"  R²   mean ± std : {cv_results['R2_mean']:.4f} ± {cv_results['R2_std']:.4f}")
        print(f"  MSE  mean ± std : {cv_results['MSE_mean']:.4f} ± {cv_results['MSE_std']:.4f}")
        print(f"  RMSE mean       : {cv_results['RMSE_mean']:.4f}")
        print(f"{'='*50}\n")
        return cv_results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 ── VISUALIZER
#  Diagnostic plots that help you understand the model behavior.
# ══════════════════════════════════════════════════════════════════════════════

class ElasticNetVisualizer:
    """
    Plotting utilities for Elastic Net model diagnostics.

    Plots available
    ───────────────
    1. plot_coefficients   — bar chart of coefficients (shows sparsity)
    2. plot_residuals      — residuals vs fitted values (detect patterns)
    3. plot_predictions    — actual vs predicted scatter
    4. plot_regularization_path — how coefficients change with alpha
    5. plot_all            — all four plots in a single figure

    Example
    -------
    >>> viz = ElasticNetVisualizer(model, evaluator)  # doctest: +SKIP
    >>> viz.plot_all(X_test, y_test)                  # doctest: +SKIP
    """

    def __init__(self, model: ElasticNetModel, evaluator: ElasticNetEvaluator = None):
        self.model = model
        self.evaluator = evaluator

    # ── 1. Coefficient bar chart ──────────────────────────────────────────────

    def plot_coefficients(self, top_n: int = None, ax=None):
        """
        Bar chart showing all (or top N) feature coefficients.

        What to look for:
          - Bars at 0 → Lasso penalty zeroed out those features
          - Largest bars → most influential features
          - Negative bars → feature has inverse relationship with target
        """
        self.model._check_fitted()
        coef_df = self.model.get_coefficients()
        if top_n:
            coef_df = coef_df.head(top_n)

        colors = ["#E74C3C" if c < 0 else "#2ECC71" for c in coef_df["coefficient"]]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, len(coef_df) * 0.35)))

        ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Coefficient Value")
        ax.set_title(
            f"Elastic Net Coefficients\n"
            f"(α={self.model.alpha}, l1_ratio={self.model.l1_ratio})"
        )
        ax.invert_yaxis()

        # Annotate zero coefficients
        zero_count = (self.model._model.coef_ == 0).sum()
        ax.text(
            0.98, 0.02,
            f"{zero_count} features zeroed-out",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="grey"
        )
        if ax is None:
            plt.tight_layout()
            plt.show()

    # ── 2. Residuals vs Fitted ────────────────────────────────────────────────

    def plot_residuals(self, X, y, ax=None):
        """
        Residuals vs Fitted values plot.

        What to look for:
          - Random scatter around 0 → good model (no systematic pattern)
          - Fan shape (heteroscedasticity) → variance increases with fitted value
          - Curve pattern → the model is missing a non-linear relationship
        """
        y = np.array(y).ravel()
        y_pred = self.model.predict(X)
        residuals = y - y_pred

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(y_pred, residuals, alpha=0.6, color="#3498DB", edgecolors="white", s=40)
        ax.axhline(0, color="#E74C3C", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals  (y − ŷ)")
        ax.set_title("Residuals vs Fitted")

        # Add a smoothed trend line to spot patterns
        try:
            from scipy.ndimage import uniform_filter1d
            idx = np.argsort(y_pred)
            smooth = uniform_filter1d(residuals[idx], size=max(3, len(y) // 20))
            ax.plot(y_pred[idx], smooth, color="orange", linewidth=1.5,
                    linestyle="-", label="Trend")
            ax.legend(fontsize=8)
        except ImportError:
            pass

        if ax is None:
            plt.tight_layout()
            plt.show()

    # ── 3. Actual vs Predicted ────────────────────────────────────────────────

    def plot_predictions(self, X, y, ax=None):
        """
        Scatter plot of actual vs predicted values.

        What to look for:
          - Points close to the diagonal → accurate predictions
          - Systematic above/below diagonal → model bias
          - Spread around diagonal → variance / noise
        """
        y = np.array(y).ravel()
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(y, y_pred, alpha=0.6, color="#9B59B6", edgecolors="white", s=40)

        # Perfect prediction line
        mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="#E74C3C", linestyle="--",
                linewidth=1.5, label="Perfect prediction")

        ax.set_xlabel("Actual Values  (y)")
        ax.set_ylabel("Predicted Values  (ŷ)")
        ax.set_title(f"Actual vs Predicted  (R² = {r2:.4f})")
        ax.legend(fontsize=8)

        if ax is None:
            plt.tight_layout()
            plt.show()

    # ── 4. Regularization path ────────────────────────────────────────────────

    def plot_regularization_path(self, X, y, l1_ratio: float = None, n_alphas: int = 50, ax=None):
        """
        Shows how coefficients shrink as the regularization strength (alpha) increases.

        WHAT THIS REVEALS:
        ──────────────────
        - Each line = one feature's coefficient value across alpha values
        - Lines that reach 0 early → less important features (penalized first)
        - Lines that persist → most important / robust features
        - The vertical dashed line = the alpha used in your current model
        """
        self.model._check_fitted()
        X_arr = np.array(X)
        y_arr = np.array(y).ravel()

        if self.model.scale_features:
            X_arr = self.model._scaler.transform(X_arr)

        if l1_ratio is None:
            l1_ratio = self.model.l1_ratio

        alphas = np.logspace(-4, 1, n_alphas)
        coefs  = []
        for a in alphas:
            m = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10_000)
            m.fit(X_arr, y_arr)
            coefs.append(m.coef_.copy())
        coefs = np.array(coefs)  # shape: (n_alphas, n_features)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        cmap = plt.cm.tab20
        for i, name in enumerate(self.model.feature_names_):
            ax.plot(alphas, coefs[:, i],
                    label=name if len(self.model.feature_names_) <= 15 else None,
                    color=cmap(i % 20), linewidth=1.4)

        ax.axvline(self.model.alpha, color="black", linestyle="--",
                   linewidth=1.5, label=f"Current α={self.model.alpha}")
        ax.set_xscale("log")
        ax.set_xlabel("Alpha  (log scale)  →  Stronger regularization")
        ax.set_ylabel("Coefficient Value")
        ax.set_title(f"Regularization Path  (l1_ratio={l1_ratio})")
        if len(self.model.feature_names_) <= 15:
            ax.legend(fontsize=7, loc="upper right")

        if ax is None:
            plt.tight_layout()
            plt.show()

    # ── 5. All plots ──────────────────────────────────────────────────────────

    def plot_all(self, X, y, save_path: str = None):
        """
        Render a 2×2 diagnostic dashboard in a single figure.

        Layout:
          ┌─────────────────────┬──────────────────────┐
          │  Coefficient Chart  │  Actual vs Predicted  │
          ├─────────────────────┼──────────────────────┤
          │  Residuals Plot     │  Regularization Path  │
          └─────────────────────┴──────────────────────┘
        """
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle(
            f"Elastic Net Diagnostic Dashboard\n"
            f"α={self.model.alpha}  |  l1_ratio={self.model.l1_ratio}",
            fontsize=14, fontweight="bold", y=1.01
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        self.plot_coefficients(ax=ax1)
        self.plot_predictions(X, y, ax=ax2)
        self.plot_residuals(X, y, ax=ax3)
        self.plot_regularization_path(X, y, ax=ax4)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[ElasticNetVisualizer] Plot saved to: {save_path}")

        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 ── FULL PIPELINE CONVENIENCE FUNCTION
#  One function that does everything: tune → fit → evaluate → visualize
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X,
    y,
    feature_names=None,
    test_size: float = 0.2,
    auto_tune: bool = True,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    cv: int = 5,
    plot: bool = True,
    save_plot: str = None,
    random_state: int = 42,
) -> dict:
    """
    End-to-end Elastic Net pipeline in a single function call.

    PIPELINE STEPS:
    ───────────────
    1. Split data into train / test sets
    2. (Optional) Auto-tune alpha and l1_ratio via cross-validation
    3. Fit the model on the training set
    4. Print model summary (coefficients)
    5. Evaluate on both train and test sets
    6. Run k-fold cross-validation for robust performance estimate
    7. (Optional) Generate diagnostic plots

    Parameters
    ----------
    X            : feature matrix (array or DataFrame)
    y            : target vector
    feature_names: list of column names (optional if X is a DataFrame)
    test_size    : fraction of data to hold out for testing (default 0.2)
    auto_tune    : if True, search for the best alpha and l1_ratio via CV
    alpha        : regularization strength (used only if auto_tune=False)
    l1_ratio     : L1/L2 mix (used only if auto_tune=False)
    cv           : number of folds for cross-validation
    plot         : whether to display diagnostic plots
    save_plot    : file path to save the plot image (e.g., "output.png")
    random_state : random seed

    Returns
    -------
    dict with keys: model, evaluator, visualizer, train_metrics, test_metrics, cv_metrics
    """

    print("\n" + "█" * 60)
    print("  ELASTIC NET — FULL PIPELINE")
    print("█" * 60)

    # ── 1. Convert inputs
    X = np.array(X)
    y = np.array(y).ravel()
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    # ── 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n[Pipeline] Data split — Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Tune or use provided hyperparameters
    if auto_tune:
        print("\n[Pipeline] Step 1/4 — Hyperparameter Tuning ...")
        tuner = ElasticNetTuner(cv=cv, random_state=random_state)
        alpha, l1_ratio = tuner.fit(X_train, y_train)
        tuner.results_summary()
    else:
        print(f"\n[Pipeline] Step 1/4 — Using provided alpha={alpha}, l1_ratio={l1_ratio}")

    # ── 4. Fit
    print("\n[Pipeline] Step 2/4 — Fitting model ...")
    model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    # ── 5. Evaluate
    print("\n[Pipeline] Step 3/4 — Evaluating ...")
    evaluator = ElasticNetEvaluator(model)
    train_metrics = evaluator.evaluate(X_train, y_train, label="Train Set")
    test_metrics  = evaluator.evaluate(X_test,  y_test,  label="Test Set")
    cv_metrics    = evaluator.cross_validate(X, y, cv=cv)

    # ── 6. Plots
    visualizer = ElasticNetVisualizer(model, evaluator)
    if plot:
        print("\n[Pipeline] Step 4/4 — Generating diagnostic plots ...")
        visualizer.plot_all(X_test, y_test, save_path=save_plot)

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print("█" * 60 + "\n")

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
#  python elastic_net_module.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  ELASTIC NET MODULE — DEMO")
    print("  Using a synthetic regression dataset")
    print("=" * 60 + "\n")

    # ── Generate synthetic data
    from sklearn.datasets import make_regression
    rng = np.random.RandomState(42)

    # 500 samples, 20 features, only 8 are truly informative
    # This is an ideal scenario for Elastic Net: sparse signal + some correlation
    X_demo, y_demo = make_regression(
        n_samples=500, n_features=20, n_informative=8,
        noise=25, random_state=42
    )
    feature_names_demo = [f"feature_{i:02d}" for i in range(20)]

    # ── Run the full pipeline
    results = run_full_pipeline(
        X             = X_demo,
        y             = y_demo,
        feature_names = feature_names_demo,
        test_size     = 0.2,
        auto_tune     = True,
        cv            = 5,
        plot          = True,
        save_plot     = "elastic_net_diagnostics.png",
    )

    # ── Access the fitted model for later use
    model = results["model"]
    print("\nTop 5 most influential features:")
    print(model.get_coefficients().head(5).to_string(index=False))