"""
================================================================================
  DECISION TREE REGRESSION MODULE
  A complete, plug-and-play module for Decision Tree Regression with
  detailed step-by-step breakdowns of every concept and line of code.
================================================================================

WHAT IS DECISION TREE REGRESSION?
───────────────────────────────────
  Decision Tree Regression uses the same tree-building algorithm as
  Classification, but instead of predicting a CLASS LABEL at each leaf,
  it predicts a CONTINUOUS VALUE — the mean of all training samples
  that reached that leaf.

  Classification leaf  → majority vote       → class label  (e.g., "Benign")
  Regression leaf      → average of Y values → float        (e.g., 47.3)

  This is the ONLY difference between the two algorithms. The split-
  finding process is completely identical in structure but uses a
  different impurity measure suited for continuous targets.

HOW REGRESSION SPLITS ARE CHOSEN (MSE criterion):
───────────────────────────────────────────────────
  For regression, "impurity" = VARIANCE of Y values in a node.
  A pure regression node = all Y values are identical (variance = 0).

  MEAN SQUARED ERROR (MSE) at a node:
      MSE(node) = (1/n) · Σ (yᵢ - ȳ)²
      where ȳ = mean of Y values in this node

  For each candidate split (feature f, threshold t):
      LEFT  ← samples where feature_f ≤ t
      RIGHT ← samples where feature_f > t

      Score = (|LEFT|/n) · MSE(LEFT) + (|RIGHT|/n) · MSE(RIGHT)

  The split minimizing this WEIGHTED MSE is chosen.
  This is equivalent to maximizing the variance REDUCTION from the split.

  VARIANCE REDUCTION = MSE(parent) - [weighted MSE(children)]
  The higher the variance reduction, the more informative the split.

  ALTERNATIVE CRITERION — MAE (Mean Absolute Error):
      MAE(node) = (1/n) · Σ |yᵢ - median(Y)|
      More robust to outliers than MSE but slower to compute.
      sklearn supports criterion='friedman_mse' as an improved variant.

WHAT THE TREE PREDICTS:
────────────────────────
  Training phase: For each leaf, record the MEAN of all Y values that
                  land there:  ŷ_leaf = mean(Y_train in this leaf)

  Prediction:     Follow the tree from root to leaf for a new sample.
                  Return ŷ_leaf as the prediction.

  This means the tree produces STEP-FUNCTION predictions — the output
  can only take values that are means of training subsets.
  The more leaves, the finer the steps, the closer to the true function.

HOW DEPTH CONTROLS SMOOTHNESS:
────────────────────────────────
  Shallow tree (depth=2):  Few large leaves, each averaging many Y values
                            → Smooth, stepped output, possibly underfitting
  Deep tree (depth=∞):     Many small leaves, some with 1 sample
                            → Jagged, noisy output, overfitting training data

  Unlike KNN (global K controls smoothness) or polynomial regression
  (degree controls flexibility), tree depth controls LOCAL smoothness —
  some regions can be finely split while others are left broad.

HOW REGRESSION TREE DIFFERS FROM REGRESSION TREE CLASSIFICATION:
──────────────────────────────────────────────────────────────────
  Feature               Classification             Regression
  ─────────────────     ───────────────────        ──────────────────────
  Target variable       Discrete class label        Continuous float
  Impurity measure      Gini / Entropy              MSE / MAE / Friedman
  Leaf prediction       Majority vote               Mean of Y values
  Evaluation metrics    Accuracy, F1, AUC           MSE, RMSE, MAE, R²
  Probability output    Class proportions           N/A (direct prediction)
  Overfitting signal    Train acc >> Test acc       Train R² >> Test R²

HOW REGRESSION TREES COMPARE TO OTHER REGRESSION MODELS:
──────────────────────────────────────────────────────────
  Elastic Net (linear)   → assumes linear relationship, smooth boundary
  KNN Regression         → local averages of neighbors, no feature selection
  Decision Tree Reg      → axis-aligned rectangular regions, step function
                           handles non-linear AND discontinuous relationships
                           naturally — no kernel trick needed

CONTENTS:
─────────
  1. DecisionTreeRegressionModel      — core model with tree extraction
  2. DecisionTreeRegressionTuner      — CV tuning of depth, leaf size, alpha
  3. DecisionTreeRegressionEvaluator  — MSE, RMSE, MAE, R², residual analysis
  4. DecisionTreeRegressionVisualizer — tree diagram, 1D fit, residuals,
                                        feature importance, depth analysis
  5. run_full_pipeline()              — one-call end-to-end pipeline

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

from sklearn.tree import (
    DecisionTreeRegressor,
    export_text,
    plot_tree,
)
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, GridSearchCV
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — CORE MODEL CLASS
#
#  KEY INSIGHT — WHY NO SCALING IS NEEDED:
#  ─────────────────────────────────────────
#  Each split compares feature_f to a threshold:   "Is X₃ ≤ 4.72?"
#  Whether X₃ is raw or standardized, the SAME samples satisfy the
#  inequality — the tree structure is invariant to monotonic transforms.
#  This is true for both classification and regression trees.
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeRegressionModel:
    """
    Decision Tree Regressor — predicts continuous values via recursive
    partitioning and leaf-level averaging.

    PREDICTION FORMULA AT EACH LEAF:
    ──────────────────────────────────
        ŷ_leaf = (1/n_leaf) · Σᵢ yᵢ   for all training samples i in the leaf

    The tree partitions the feature space into rectangular regions, and
    each region predicts the mean Y of its training samples.

    Parameters
    ----------
    criterion : str, default='squared_error'
        Impurity measure to minimize at each split:
          'squared_error'  → minimize MSE = Σ(yᵢ - ȳ)²/n
                             Standard choice, sensitive to outliers
          'friedman_mse'   → MSE with Friedman's improvement score
                             Often better in practice — recommended
          'absolute_error' → minimize MAE = Σ|yᵢ - median(Y)|/n
                             Robust to outliers but much slower
          'poisson'        → for count/rate targets (non-negative Y)

    max_depth : int or None, default=None
        Maximum depth of the tree.
        None  → grow until all leaves have min_samples_leaf samples
                Likely overfits. Start with 3–6 for regression.
        Shallower → smoother step function, better generalization

    min_samples_split : int, default=2
        A node must have at least this many samples to be split further.
        Prevents splits on tiny groups of samples.

    min_samples_leaf : int, default=1
        Each leaf must contain at least this many training samples.
        The leaf prediction is the mean of those samples — more samples
        per leaf = more stable (less noisy) leaf predictions.
        Good starting range: 5–20 for most regression problems.

    min_impurity_decrease : float, default=0.0
        A split is performed only if it decreases impurity by at least
        this amount. Prevents trivially small splits.
        Equivalent to a minimum variance reduction threshold.

    ccp_alpha : float, default=0.0
        Complexity parameter for Minimal Cost-Complexity Pruning.
        0.0    → no pruning (full tree)
        Larger → prune branches whose benefit doesn't justify complexity
        Use SVMTuner / DecisionTreeRegressionTuner to find the right alpha.

    Example
    -------
    >>> model = DecisionTreeRegressionModel(max_depth=4)        # doctest: +SKIP
    >>> model.fit(X_train, y_train)                             # doctest: +SKIP
    >>> preds = model.predict(X_test)                           # doctest: +SKIP
    >>> model.summary()                                         # doctest: +SKIP
    >>> model.print_rules()                                     # doctest: +SKIP
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features=None,
        ccp_alpha: float = 0.0,
        random_state: int = 42,
    ):
        self.criterion              = criterion
        self.max_depth              = max_depth
        self.min_samples_split      = min_samples_split
        self.min_samples_leaf       = min_samples_leaf
        self.min_impurity_decrease  = min_impurity_decrease
        self.max_features           = max_features
        self.ccp_alpha              = ccp_alpha
        self.random_state           = random_state

        self._model = DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )

        self.feature_names_ = None
        self.X_train_       = None
        self.y_train_       = None
        self.is_fitted_     = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        """
        Build the regression tree — find the optimal splits to minimize MSE.

        STEP-BY-STEP INTERNAL PROCESS:
        ────────────────────────────────
        Step 1 — Convert inputs, store feature names.
                 NO label encoding (target is already continuous float).
                 NO feature scaling (threshold splits are scale-invariant).

        Step 2 — ROOT NODE: compute MSE of all Y values.
                     MSE_root = (1/n) · Σ(yᵢ - ȳ)²
                 This represents our "error" before any splits.

        Step 3 — GREEDY SPLIT SEARCH at the root (and every subsequent node):
                 For each feature f in {0 ... n_features}:
                   For each candidate threshold t (midpoints of sorted values):
                     LEFT  = samples where X[f] ≤ t → compute MSE_left
                     RIGHT = samples where X[f] > t → compute MSE_right
                     score = (|LEFT|/n)·MSE_left + (|RIGHT|/n)·MSE_right

                 Find (f*, t*) that MINIMIZES this weighted MSE score.
                 This is the split that most reduces prediction variance.

        Step 4 — APPLY the best split: divide samples into LEFT and RIGHT.
                 Record in the node:
                   node.feature   = f*
                   node.threshold = t*

        Step 5 — RECURSE on LEFT and RIGHT child nodes with their subsets.
                 Repeat Steps 2-4 for each child.

        Step 6 — STOPPING CONDITION: create a LEAF when any of:
                   - Only min_samples_leaf samples remain
                   - Node is already pure (all Y identical, MSE = 0)
                   - max_depth reached
                   - min_impurity_decrease not met

        Step 7 — LEAF PREDICTION: each leaf stores ȳ = mean(Y in leaf).
                 This is the predicted value for any sample reaching this leaf.

        Parameters
        ----------
        X            : array-like (n_samples, n_features)
        y            : array-like (n_samples,) — CONTINUOUS target values
        feature_names: optional list of column names
        """
        X = np.array(X)
        y = np.array(y).ravel()

        # Step 1 — Store names and raw data (for 1D visualization)
        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"feature_{i}" for i in range(X.shape[1])]
        )
        self.X_train_ = X
        self.y_train_ = y

        # Steps 2-7 — Build tree (no scaling applied)
        self._model.fit(X, y)
        self.is_fitted_ = True

        n_nodes   = self._model.tree_.node_count
        n_leaves  = self._model.get_n_leaves()
        depth     = self._model.get_depth()
        leaf_mean = self._model.tree_.n_node_samples[
            self._model.tree_.children_left == -1
        ].mean()

        print(
            f"[DecisionTreeRegressor] Fitted\n"
            f"  Criterion        : {self.criterion}\n"
            f"    → Each split minimizes: "
            f"{'MSE = (1/n)·Σ(yᵢ-ȳ)²' if 'squared' in self.criterion else 'MAE'}\n"
            f"  Max depth set    : {self.max_depth}  (actual depth = {depth})\n"
            f"  Total nodes      : {n_nodes}  "
            f"({n_leaves} leaves + {n_nodes - n_leaves} internal)\n"
            f"  Avg samples/leaf : {leaf_mean:.1f}\n"
            f"  Y range          : [{y.min():.3f}, {y.max():.3f}]  "
            f"mean={y.mean():.3f}\n"
            f"  Features         : {X.shape[1]}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict continuous Y values for new samples.

        HOW REGRESSION TREE PREDICTION WORKS:
        ───────────────────────────────────────
        For each new sample x:

          1. Start at the ROOT node.
          2. Read the split rule: "Is feature_k ≤ threshold?"
             YES → go to LEFT child
             NO  → go to RIGHT child
          3. Repeat step 2 until a LEAF is reached.
          4. Return the leaf's stored ȳ (mean of training Y in that leaf).

        PREDICTION TIME COMPLEXITY: O(depth) per sample.
        A depth-5 tree makes at most 5 comparisons per prediction.

        THE STEP-FUNCTION NATURE:
        ──────────────────────────
        Every sample in the same leaf gets the same prediction (ȳ_leaf).
        The output is therefore a PIECEWISE CONSTANT (step function):
          - More leaves → more steps → finer approximation of true function
          - Fewer leaves → coarser steps → smoother but less flexible

        This contrasts with Elastic Net (smooth line) or KNN (local averages
        that smoothly transition). The step-function output can sometimes
        cause large errors for samples exactly on leaf boundaries.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) — continuous float predictions
        """
        self._check_fitted()
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._model.predict(X)

    # ── predict_single ────────────────────────────────────────────────────────

    def predict_single(self, x):
        """Predict for a single sample. Returns float."""
        return float(self.predict(np.array(x).reshape(1, -1))[0])

    # ── print_rules ───────────────────────────────────────────────────────────

    def print_rules(self, max_depth: int = None):
        """
        Print the tree as human-readable IF-THEN rules for regression.

        REGRESSION RULES SHOW:
        ───────────────────────
        At each internal node:   |--- feature_k <= threshold
        At each leaf:            |--- value: [mean_Y]

        where mean_Y is the predicted continuous value for that region.

        Example output for a house price tree:
          |--- sqft <= 1500
          |   |--- bedrooms <= 2
          |   |   |--- value: [185000.00]    ← small 2BR house
          |   |--- bedrooms > 2
          |   |   |--- value: [245000.00]    ← small 3BR house
          |--- sqft > 1500
          |   |--- value: [380000.00]        ← large house

        Parameters
        ----------
        max_depth : limit the depth of rules printed
        """
        self._check_fitted()
        rules = export_text(
            self._model,
            feature_names=self.feature_names_,
            max_depth=max_depth or self._model.get_depth(),
        )
        print("\n" + "=" * 62)
        print("  DECISION TREE REGRESSION RULES")
        print("  Each leaf 'value:' is the predicted Y for that region")
        print("  Each |--- indicates one level deeper in the tree")
        print("=" * 62)
        print(rules)
        print("=" * 62 + "\n")

    # ── explain_prediction ────────────────────────────────────────────────────

    def explain_prediction(self, x):
        """
        Trace a single prediction step-by-step through the tree.

        SHOWS:
          - Each split decision made at every node on the path
          - The leaf reached and its mean Y value
          - The actual prediction returned

        This is Decision Trees' interpretability advantage — you can
        trace exactly WHY a specific prediction was made, showing the
        exact sequence of feature comparisons.

        Parameters
        ----------
        x : 1D array of shape (n_features,)
        """
        self._check_fitted()
        x = np.array(x).ravel()

        tree     = self._model.tree_
        node     = 0
        path     = []

        while tree.children_left[node] != -1:   # while not leaf
            feat  = tree.feature[node]
            thresh= tree.threshold[node]
            fname = self.feature_names_[feat]
            val   = x[feat]

            if val <= thresh:
                decision = f"YES → LEFT   ({fname}={val:.4f} ≤ {thresh:.4f})"
                node     = tree.children_left[node]
            else:
                decision = f"NO  → RIGHT  ({fname}={val:.4f} > {thresh:.4f})"
                node     = tree.children_right[node]

            depth = len(path) + 1
            path.append((depth, fname, thresh, val, decision))

        leaf_val   = tree.value[node][0][0]
        n_in_leaf  = tree.n_node_samples[node]

        print("\n" + "=" * 62)
        print("  REGRESSION TREE — PREDICTION TRACE")
        print("=" * 62)
        print(f"  Input: {dict(zip(self.feature_names_, x))}\n")
        print(f"  {'Depth':<6} {'Decision'}")
        print(f"  {'─'*5}  {'─'*50}")
        for d, fname, thresh, val, decision in path:
            print(f"  {d:<6} {decision}")
        print(f"\n  ► Reached LEAF (depth={len(path)})")
        print(f"    Training samples in leaf  : {n_in_leaf}")
        print(f"    Mean Y of those samples   : {leaf_val:.4f}")
        print(f"    ═══════════════════════════════════════")
        print(f"    PREDICTION = {leaf_val:.4f}")
        print("=" * 62 + "\n")

    # ── get_feature_importance ────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Feature importances based on total variance reduction.

        FOR REGRESSION TREES specifically:
        ────────────────────────────────────
        importance(f) = Σ_nodes_using_f [
            (n_node / n_total) × MSE_reduction_at_node
        ]

        MSE_reduction = MSE(parent) - (|LEFT|/n)·MSE(LEFT) - (|RIGHT|/n)·MSE(RIGHT)

        Features that reduce variance heavily across many samples
        get high importance scores. A feature with zero importance
        was never used in any split — the tree found no useful threshold.

        Returns
        -------
        pd.DataFrame sorted by importance descending
        """
        self._check_fitted()
        return pd.DataFrame({
            "feature":    self.feature_names_,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print structured model summary: tree stats, leaf analysis,
        feature importances, and regression-specific notes.
        """
        self._check_fitted()
        tree     = self._model.tree_
        depth    = self._model.get_depth()
        n_leaves = self._model.get_n_leaves()
        n_nodes  = tree.node_count
        fi_df    = self.get_feature_importance()

        leaf_mask   = tree.children_left == -1
        leaf_vals   = tree.value[leaf_mask].ravel()
        leaf_counts = tree.n_node_samples[leaf_mask]

        sep = "=" * 65
        print(f"\n{sep}")
        print("  DECISION TREE REGRESSION — MODEL SUMMARY")
        print(sep)
        print(f"  Task             : Regression (continuous output)")
        print(f"  Criterion        : {self.criterion}")
        print(f"  Prediction rule  : ŷ_leaf = mean(Y_train in leaf)")

        print(f"\n  ── Tree Structure ───────────────────────────────────────")
        print(f"  Actual depth     : {depth}  (max_depth set = {self.max_depth})")
        print(f"  Total nodes      : {n_nodes}")
        print(f"  Leaf nodes       : {n_leaves}   (prediction regions)")
        print(f"  Internal nodes   : {n_nodes - n_leaves}  (split decisions)")

        print(f"\n  ── Leaf Predictions Distribution ────────────────────────")
        print(f"  Leaf Y range     : [{leaf_vals.min():.3f}, {leaf_vals.max():.3f}]")
        print(f"  Leaf Y mean      : {leaf_vals.mean():.3f}")
        print(f"  Avg samples/leaf : {leaf_counts.mean():.1f}")
        print(f"  Min samples/leaf : {leaf_counts.min()}   "
              f"(1 → that leaf memorizes a single training point)")
        print(f"\n  NOTE: Each of the {n_leaves} leaves predicts a distinct Y value.")
        print(f"  The model output is a step function with {n_leaves} possible values.")

        print(f"\n  ── Top 10 Feature Importances ───────────────────────────")
        print(f"  (Total MSE reduction each feature contributed across all splits)")
        print(f"  {'Feature':<25}  {'Importance':>10}  Bar")
        print(f"  {'-'*25}  {'-'*10}  {'-'*25}")
        top10   = fi_df.head(10)
        max_imp = top10["importance"].max()
        for _, row in top10.iterrows():
            bar = "█" * int(row["importance"] / max_imp * 25) if max_imp > 0 else ""
            print(f"  {row['feature']:<25}  {row['importance']:>10.5f}  {bar}")

        unused = (fi_df["importance"] == 0).sum()
        if unused > 0:
            print(f"\n  {unused} feature(s) never used in any split (importance = 0)")

        print(f"\n  ── Key Regression Notes ─────────────────────────────────")
        print(f"  • Tree predicts ȳ per leaf — output is piecewise constant")
        print(f"  • Cannot extrapolate beyond training Y range [{self.y_train_.min():.2f}, {self.y_train_.max():.2f}]")
        print(f"  • Use .print_rules()          → see all IF-THEN regression rules")
        print(f"  • Use .explain_prediction(x)  → trace a single prediction")
        print(sep + "\n")

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(X, y) first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TUNER
#
#  UNIQUE CONSIDERATION FOR REGRESSION TREES:
#  ────────────────────────────────────────────
#  Regression trees overfit differently from classifiers.
#  A deep regression tree memorizes individual Y values, creating leaves
#  with single samples that predict their exact training value.
#  This gives train R²=1.0 but test R² can be poor.
#
#  The cost-complexity pruning path is especially valuable here —
#  it shows exactly how much accuracy we trade for simplicity.
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeRegressionTuner:
    """
    K-fold cross-validation tuner for regression tree hyperparameters.

    WHAT WE TUNE:
    ─────────────
    max_depth           : primary lever — controls overfitting
    min_samples_leaf    : secondary — prevents single-sample leaves
    criterion           : mse vs friedman_mse vs mae
    ccp_alpha           : post-pruning strength (via pruning path)

    COST-COMPLEXITY PRUNING PATH:
    ──────────────────────────────
    sklearn can compute the ENTIRE pruning path — the sequence of trees
    produced by removing branches one at a time, from least useful to most.

    Each step in the path corresponds to an alpha value:
      alpha=0   → full unpruned tree
      alpha=0.1 → removed a few low-benefit branches
      alpha=1.0 → highly pruned, near-stump

    We CV over all alphas to find the one that gives the best test RMSE.

    Example
    -------
    >>> tuner = DecisionTreeRegressionTuner(cv=5)             # doctest: +SKIP
    >>> best_depth = tuner.fit(X_train, y_train)              # doctest: +SKIP
    >>> tuner.results_summary()                               # doctest: +SKIP
    """

    def __init__(self, cv: int = 5, scoring: str = "neg_mean_squared_error",
                 random_state: int = 42):
        self.cv           = cv
        self.scoring      = scoring
        self.random_state = random_state

        self.best_params_   = None
        self.best_score_    = None
        self.depth_rmse_    = {}    # depth → (mean_rmse, std_rmse)
        self.depth_r2_      = {}    # depth → mean_r2
        self.ccp_alphas_    = None
        self.ccp_rmse_      = None
        self._grid_search   = None

    def fit(self, X, y, param_grid: dict = None):
        """
        Grid search over max_depth, min_samples_leaf, and criterion.
        Also computes the cost-complexity pruning path for visualization.

        STEP-BY-STEP:
        ─────────────
        Step 1 — Build param grid: depths × leaf sizes × criteria.
                 Regression trees typically need deeper max_depth than
                 classifiers because continuous targets need finer splits.

        Step 2 — KFold CV (no stratification — continuous target).
                 For each (max_depth, min_samples_leaf, criterion):
                   Train on k-1 folds, evaluate RMSE on held-out fold.
                   Record mean and std across folds.

        Step 3 — Select best params by lowest mean CV RMSE.

        Step 4 — Compute Cost-Complexity Pruning path on the full tree.
                 This gives the alpha values for the pruning visualization.

        Returns
        -------
        best_depth : int or None
        """
        X = np.array(X)
        y = np.array(y).ravel()

        if param_grid is None:
            param_grid = {
                "max_depth":        [2, 3, 4, 5, 6, 8, 10, 12, None],
                "min_samples_leaf": [1, 2, 5, 10, 20],
                "criterion":        ["squared_error", "friedman_mse"],
            }

        kf = KFold(n_splits=self.cv, shuffle=True,
                   random_state=self.random_state)

        n_combos = np.prod([len(v) for v in param_grid.values()])
        print(f"[DecisionTreeRegressionTuner] Searching {n_combos} "
              f"combinations with {self.cv}-fold CV ...")

        base = DecisionTreeRegressor(random_state=self.random_state)
        self._grid_search = GridSearchCV(
            base, param_grid, cv=kf,
            scoring=self.scoring, n_jobs=-1, refit=True,
        )
        self._grid_search.fit(X, y)

        self.best_params_ = self._grid_search.best_params_
        best_mse          = -self._grid_search.best_score_
        self.best_score_  = np.sqrt(best_mse)

        # Per-depth RMSE and R² curves (for overfitting / elbow plot)
        depths = list(range(1, 16)) + [None]
        for d in depths:
            m = DecisionTreeRegressor(max_depth=d,
                                      random_state=self.random_state)
            mse_sc = -cross_val_score(m, X, y, cv=kf,
                                       scoring="neg_mean_squared_error")
            r2_sc  = cross_val_score(m, X, y, cv=kf, scoring="r2")
            self.depth_rmse_[str(d)] = (np.sqrt(mse_sc).mean(),
                                        np.sqrt(mse_sc).std())
            self.depth_r2_[str(d)]   = r2_sc.mean()

        # Cost-Complexity Pruning path
        clf_full = DecisionTreeRegressor(random_state=self.random_state)
        clf_full.fit(X, y)
        path   = clf_full.cost_complexity_pruning_path(X, y)
        alphas = path.ccp_alphas[:-1]
        ccp_sc = []
        for alpha in alphas:
            m  = DecisionTreeRegressor(ccp_alpha=alpha,
                                       random_state=self.random_state)
            sc = -cross_val_score(m, X, y, cv=kf,
                                  scoring="neg_mean_squared_error")
            ccp_sc.append(np.sqrt(sc).mean())
        self.ccp_alphas_ = alphas
        self.ccp_rmse_   = np.array(ccp_sc)

        bd = self.best_params_["max_depth"]
        bl = self.best_params_["min_samples_leaf"]
        bc = self.best_params_["criterion"]
        print(f"[DecisionTreeRegressionTuner] ✓ "
              f"max_depth={bd}, min_samples_leaf={bl}, criterion={bc}\n"
              f"  Best CV RMSE: {self.best_score_:.4f}")
        return bd

    def results_summary(self):
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        print("\n" + "=" * 55)
        print("  DECISION TREE REGRESSION TUNING RESULTS")
        print("=" * 55)
        for k, v in self.best_params_.items():
            print(f"  {k:<25} : {v}")
        print(f"  Best CV RMSE             : {self.best_score_:.4f}")
        print("=" * 55 + "\n")

    def get_best_model(self) -> "DecisionTreeRegressionModel":
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        valid_keys = DecisionTreeRegressionModel.__init__.__code__.co_varnames
        return DecisionTreeRegressionModel(**{
            k: v for k, v in self.best_params_.items()
            if k in valid_keys
        })


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — EVALUATOR
#
#  REGRESSION-SPECIFIC METRICS:
#  ─────────────────────────────
#  RMSE — Root Mean Squared Error
#    Most widely reported. Same units as Y. Penalizes large errors heavily.
#    "On average, predictions are off by RMSE units."
#
#  MAE — Mean Absolute Error
#    Robust to outliers (doesn't square the errors).
#    "Typical prediction error in Y units."
#
#  R² — Coefficient of Determination
#    Fraction of Y variance explained. 1.0 = perfect, 0.0 = predict-mean baseline.
#    CRITICAL for trees: train R² near 1.0 with test R² much lower = overfit.
#
#  MAPE — Mean Absolute Percentage Error
#    Relative error in %. Useful when Y has meaningful scale (prices, rates).
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeRegressionEvaluator:
    """
    Comprehensive regression evaluator — MSE, RMSE, MAE, R², MAPE,
    plus residual analysis and tree-specific depth/variance tradeoff.

    Example
    -------
    >>> ev = DecisionTreeRegressionEvaluator(model)      # doctest: +SKIP
    >>> ev.evaluate(X_test, y_test)                      # doctest: +SKIP
    >>> ev.cross_validate(X, y, cv=5)                    # doctest: +SKIP
    """

    def __init__(self, model: DecisionTreeRegressionModel):
        self.model      = model
        self.metrics_   = {}
        self._residuals = None
        self._y_pred    = None
        self._y_true    = None

    def evaluate(self, X, y, label: str = "Test Set") -> dict:
        """
        Compute all regression metrics and print a detailed report.

        THE OVERFITTING SIGNAL FOR REGRESSION TREES:
        ─────────────────────────────────────────────
        Train R²  ≈ 1.0  and  Test R² < 0.7  → severe overfitting
        Compare train vs test metrics to diagnose.

        Returns dict: MSE, RMSE, MAE, R², Adj_R2, MAPE
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)

        self._y_true    = y
        self._y_pred    = y_pred
        self._residuals = y - y_pred

        mse    = mean_squared_error(y, y_pred)
        rmse   = np.sqrt(mse)
        mae    = mean_absolute_error(y, y_pred)
        r2     = r2_score(y, y_pred)
        n, p   = np.array(X).shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

        # MAPE — avoid division by zero
        mask   = y != 0
        mape   = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100

        self.metrics_ = dict(
            MSE=mse, RMSE=rmse, MAE=mae, R2=r2, Adj_R2=adj_r2, MAPE=mape
        )

        d   = self.model._model.get_depth()
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  EVALUATION — {label}  (depth={d}, criterion={self.model.criterion})")
        print(sep)
        print(f"  MSE       : {mse:.4f}   (mean squared error — unit: Y²)")
        print(f"  RMSE      : {rmse:.4f}   (avg error in Y units — most interpretable)")
        print(f"  MAE       : {mae:.4f}   (avg absolute error — outlier-robust)")
        print(f"  R²        : {r2:.4f}   ({r2*100:.1f}% of Y variance explained)")
        print(f"  Adj. R²   : {adj_r2:.4f}   (R² penalized for number of features)")
        print(f"  MAPE      : {mape:.2f}%   (mean absolute % error)")

        if r2 >= 0.90:
            interp = "Excellent"
        elif r2 >= 0.75:
            interp = "Good"
        elif r2 >= 0.50:
            interp = "Moderate — check for overfitting (compare to train R²)"
        else:
            interp = "Weak — tree may be too shallow or data lacks structure"
        print(f"\n  Interpretation  : R² = {r2:.4f} → {interp}")

        # Residual statistics
        resid = self._residuals
        print(f"\n  Residual stats  :")
        print(f"    Mean   = {resid.mean():.4f}  (near 0 = no systematic bias)")
        print(f"    Std    = {resid.std():.4f}")
        print(f"    Max    = {resid.max():.4f}  (worst over-prediction)")
        print(f"    Min    = {resid.min():.4f}  (worst under-prediction)")
        print(sep + "\n")
        return self.metrics_

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        K-fold CV — no Pipeline needed (trees don't need scaling).

        THE TRAIN/TEST GAP FOR REGRESSION TREES:
        ─────────────────────────────────────────
        CV RMSE = realistic error on unseen data.
        If CV RMSE >> training RMSE → the tree is memorizing noise.
        Solution: reduce max_depth, increase min_samples_leaf, or use ccp_alpha.

        Returns dict of mean/std for RMSE and R²
        """
        X = np.array(X)
        y = np.array(y).ravel()

        m  = self.model._model
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        mse_scores = -cross_val_score(m, X, y, cv=kf,
                                       scoring="neg_mean_squared_error")
        r2_scores  = cross_val_score(m, X, y, cv=kf, scoring="r2")
        rmse_scores = np.sqrt(mse_scores)

        results = {
            "RMSE_mean": rmse_scores.mean(), "RMSE_std": rmse_scores.std(),
            "R2_mean":   r2_scores.mean(),   "R2_std":   r2_scores.std(),
        }
        d = self.model._model.get_depth()
        print(f"\n{'='*60}")
        print(f"  {cv}-FOLD CROSS-VALIDATION  (depth={d})")
        print(f"{'='*60}")
        print(f"  RMSE  mean ± std : {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
        print(f"  R²    mean ± std : {results['R2_mean']:.4f} ± {results['R2_std']:.4f}")
        print(f"{'='*60}\n")
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — VISUALIZER
#
#  REGRESSION-SPECIFIC PLOTS (beyond classification visualizer):
#  ─────────────────────────────────────────────────────────────
#  plot_1d_fit            — show the step-function predictions overlaid
#                           on the actual data (works for 1D feature)
#  plot_tree_predictions  — 3D scatter for 2-feature data (like KNN module)
#  plot_residuals         — detect bias, heteroscedasticity in residuals
#  plot_actual_vs_pred    — how close are predictions to truth?
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeRegressionVisualizer:
    """
    Six diagnostic plots for Decision Tree Regression.

    PLOTS:
    ──────
    1. plot_tree_diagram       — the actual tree with Y values at leaves
    2. plot_1d_fit             — step-function predictions vs true curve (1D)
    3. plot_actual_vs_pred     — scatter of y_true vs y_pred
    4. plot_residuals          — residuals vs fitted values
    5. plot_feature_importance — bar chart of MSE-reduction importances
    6. plot_depth_analysis     — CV RMSE vs depth (overfitting curve)
    7. plot_all                — 2×3 full dashboard

    Example
    -------
    >>> viz = DecisionTreeRegressionVisualizer(model, tuner)   # doctest: +SKIP
    >>> viz.plot_all(X_test, y_test)                           # doctest: +SKIP
    """

    def __init__(self, model: DecisionTreeRegressionModel,
                 tuner: DecisionTreeRegressionTuner = None):
        self.model = model
        self.tuner = tuner

    # ── 1. Tree Diagram ───────────────────────────────────────────────────────

    def plot_tree_diagram(self, max_depth: int = 4, ax=None):
        """
        Visualize the regression tree structure.

        WHAT EACH NODE SHOWS (regression version):
        ────────────────────────────────────────────
        feature_k ≤ threshold  — split rule
        squared_error = X.XX   — MSE at this node (lower = purer)
        samples = NNN          — training samples reaching this node
        value = X.XX           — mean Y of samples at this node
                                 For leaves, this IS the prediction.

        Darker node color = lower MSE (purer node in regression terms).
        Leaf nodes show the final predicted value (mean Y).
        """
        self.model._check_fitted()

        if ax is None:
            depth  = min(max_depth, self.model._model.get_depth())
            width  = min(20, 3 * 2**depth)
            height = max(6, 2 ** depth)
            fig, ax = plt.subplots(figsize=(width, height))

        plot_tree(
            self.model._model,
            feature_names=self.model.feature_names_,
            filled=True,
            rounded=True,
            max_depth=max_depth,
            ax=ax,
            fontsize=8,
            impurity=True,
            proportion=False,
            precision=3,
        )
        d = self.model._model.get_depth()
        ax.set_title(
            f"Decision Tree Regression Structure  (depth={d})\n"
            f"Leaf 'value' = predicted ŷ  |  "
            f"Darker = lower MSE (purer node)",
            fontsize=10
        )

    # ── 2. 1D Step-Function Fit ───────────────────────────────────────────────

    def plot_1d_fit(self, feature_idx: int = 0, ax=None):
        """
        Visualize the step-function output of the regression tree.

        THIS IS THE SIGNATURE REGRESSION TREE VISUALIZATION:
        ──────────────────────────────────────────────────────
        For a single feature (or the first PCA component), this shows:
          - Scatter of actual (X, Y) training points
          - The tree's step-function predictions (horizontal plateaus)
          - Each plateau = one leaf region, height = mean Y of that leaf
          - Vertical dashed lines = split boundaries

        THE DEEPER THE TREE:
          More plateaus → finer steps → closer to true function
          But each plateau only uses its training samples → can be noisy

        THIS IS UNIQUE TO DECISION TREES — no other model in this series
        produces a step-function output.

        Uses PCA first component if n_features > 1.
        """
        self.model._check_fitted()
        X = self.model.X_train_
        y = self.model.y_train_

        if X.shape[1] > 1:
            pca  = PCA(n_components=1, random_state=42)
            X_1d = pca.fit_transform(X).ravel()
            xlabel = "PC 1 (first principal component)"
        else:
            X_1d   = X[:, feature_idx].ravel()
            xlabel = self.model.feature_names_[feature_idx]

        # Generate dense predictions along the 1D axis to show the step function
        x_range = np.linspace(X_1d.min(), X_1d.max(), 1000).reshape(-1, 1)
        if X.shape[1] > 1:
            x_full  = pca.inverse_transform(x_range)
            y_range = self.model.predict(x_full)
        else:
            x_full  = np.zeros((1000, X.shape[1]))
            x_full[:, feature_idx] = x_range.ravel()
            y_range = self.model.predict(x_full)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(X_1d, y, color="#7F8C8D", alpha=0.45, s=20,
                   label="Training data")
        ax.plot(x_range.ravel(), y_range, color="#E74C3C", linewidth=2.5,
                label=f"Tree predictions (step function, depth={self.model._model.get_depth()})")

        # Mark split boundaries (vertical lines)
        unique_preds = np.unique(y_range)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Y  (target)")
        ax.set_title(
            f"Decision Tree Regression — Step-Function Output\n"
            f"Each horizontal plateau = one leaf's mean Y prediction\n"
            f"depth={self.model._model.get_depth()}  |  "
            f"{self.model._model.get_n_leaves()} leaves  |  "
            f"{len(unique_preds)} distinct prediction values"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 3. Actual vs Predicted ────────────────────────────────────────────────

    def plot_actual_vs_pred(self, X, y, ax=None):
        """
        Scatter of y_true vs y_pred — the standard regression diagnostic.

        FOR REGRESSION TREES SPECIFICALLY:
        ────────────────────────────────────
        Unlike smooth models (Elastic Net, KNN), the predicted values
        are DISCRETE — each unique prediction corresponds to one leaf.
        You'll see vertical columns of points with the same x-coordinate
        (all samples in the same leaf get the same prediction).

        Perfect model → all points on the diagonal.
        Vertical clusters → multiple true Y values mapped to same leaf prediction.
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)
        r2     = r2_score(y, y_pred)
        rmse   = np.sqrt(mean_squared_error(y, y_pred))

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(y, y_pred, alpha=0.55, color="#8E44AD",
                   edgecolors="white", s=35)
        mn = min(y.min(), y_pred.min())
        mx = max(y.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="#E74C3C", linestyle="--",
                linewidth=1.5, label="Perfect prediction (y = ŷ)")

        ax.set_xlabel("Actual Y  (true values)")
        ax.set_ylabel("Predicted ŷ  (leaf mean)")
        ax.set_title(
            f"Actual vs Predicted\n"
            f"R²={r2:.4f}  |  RMSE={rmse:.4f}  |  "
            f"depth={self.model._model.get_depth()}\n"
            f"Vertical clusters = multiple true Y in same leaf"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── 4. Residuals Plot ─────────────────────────────────────────────────────

    def plot_residuals(self, X, y, ax=None):
        """
        Residuals (y - ŷ) vs fitted values — bias and spread diagnostic.

        REGRESSION TREE RESIDUAL PATTERNS:
        ────────────────────────────────────
        Because tree predictions are step functions, residuals show
        STRUCTURED patterns — each vertical cluster of residuals
        corresponds to one leaf.

        WHAT TO LOOK FOR:
          Symmetric around 0    → no systematic bias
          Funnel shape          → heteroscedasticity (variance grows with ŷ)
          Clusters at same x    → leaf regions (expected for trees)
          Outliers far from 0   → samples the tree struggled with
        """
        y      = np.array(y).ravel()
        y_pred = self.model.predict(X)
        resid  = y - y_pred

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(y_pred, resid, alpha=0.55, color="#16A085",
                   edgecolors="white", s=35)
        ax.axhline(0, color="#E74C3C", linestyle="--", linewidth=1.5)

        # Optional trend line
        try:
            from scipy.ndimage import uniform_filter1d
            idx    = np.argsort(y_pred)
            smooth = uniform_filter1d(resid[idx], size=max(3, len(y) // 20))
            ax.plot(y_pred[idx], smooth, color="orange", linewidth=1.5,
                    label="Trend (should hug 0)")
            ax.legend(fontsize=8)
        except ImportError:
            pass

        ax.set_xlabel("Fitted Values  ŷ  (leaf mean predictions)")
        ax.set_ylabel("Residuals  (y − ŷ)")
        ax.set_title(
            f"Residuals vs Fitted  (depth={self.model._model.get_depth()})\n"
            f"Vertical columns = samples in the same leaf"
        )
        ax.grid(True, alpha=0.3)

    # ── 5. Feature Importance ────────────────────────────────────────────────

    def plot_feature_importance(self, top_n: int = 20, ax=None):
        """
        Bar chart of feature importances by total MSE reduction.

        REGRESSION INTERPRETATION:
        ───────────────────────────
        Unlike classification (Gini reduction), regression importances
        measure how much VARIANCE (MSE) each feature reduced across all splits.

        The most important feature is the one that, when split on,
        most consistently reduces the spread of Y values in child nodes.
        """
        self.model._check_fitted()
        fi = self.model.get_feature_importance().head(top_n)

        colors = ["#E74C3C" if i == 0 else "#1ABC9C" for i in range(len(fi))]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.38)))

        ax.barh(fi["feature"], fi["importance"],
                color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Feature Importance  (total MSE reduction)")
        ax.set_title(
            f"Decision Tree Regression Feature Importances\n"
            f"(criterion={self.model.criterion}, depth={self.model._model.get_depth()})"
        )
        ax.invert_yaxis()

        unused = (self.model._model.feature_importances_ == 0).sum()
        if unused > 0:
            ax.text(0.98, 0.02,
                    f"{unused} features with 0 importance (never used)",
                    transform=ax.transAxes, ha="right",
                    fontsize=8, color="grey")

    # ── 6. Depth Analysis (Overfitting Curve) ────────────────────────────────

    def plot_depth_analysis(self, ax=None):
        """
        CV RMSE vs tree depth — the regression overfitting curve.

        THE REGRESSION OVERFITTING PATTERN:
        ─────────────────────────────────────
        Depth 1 (stump): One split → two leaf means → high RMSE (underfit)
        Optimal depth  : Captures the main structure without fitting noise
        Deep tree      : Every training sample may get its own leaf → RMSE ≈ 0
                         on training data but high on unseen data

        NOTE: Unlike classification (which can achieve 100% train accuracy),
        regression trees can achieve RMSE → 0 on training data when every
        sample has its own leaf (depth grows until min_samples_leaf is met).

        Dashed red line = best depth from tuner.
        Shaded band     = ±1 std across CV folds.
        """
        if self.tuner is None or not self.tuner.depth_rmse_:
            print("[Visualizer] No tuner results. Run tuner.fit() first.")
            return

        depths = []
        means  = []
        stds   = []
        for k, (m, s) in self.tuner.depth_rmse_.items():
            depths.append(k)
            means.append(m)
            stds.append(s)

        labels    = [str(d) for d in depths]
        x         = np.arange(len(labels))
        means_arr = np.array(means)
        stds_arr  = np.array(stds)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(x, means_arr, color="#E74C3C", linewidth=2.5,
                marker="o", markersize=5, label="CV RMSE (mean)")
        ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.15, color="#E74C3C", label="±1 std")

        best_d = str(self.tuner.best_params_.get("max_depth"))
        if best_d in labels:
            bi = labels.index(best_d)
            ax.axvline(bi, color="#2ECC71", linestyle="--", linewidth=1.8,
                       label=f"Best depth = {best_d}")
            ax.scatter([bi], [means_arr[bi]], color="#2ECC71", s=100, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("max_depth  (None = unlimited)")
        ax.set_ylabel("Cross-validated RMSE  (lower = better)")
        ax.set_title(
            "Decision Tree Regression — Depth vs CV RMSE\n"
            "Depth too small = underfit  |  Depth too large = overfit"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 7. Full Dashboard ─────────────────────────────────────────────────────

    def plot_all(self, X_test, y_test, save_path: str = None):
        """
        Render the full 2×3 regression diagnostic dashboard.

        Layout:
          ┌──────────────┬──────────────┬──────────────┐
          │  Tree        │  Step-       │  Feature     │
          │  Diagram     │  Function    │  Importance  │
          ├──────────────┼──────────────┼──────────────┤
          │  Depth       │  Actual vs   │  Residuals   │
          │  Analysis    │  Predicted   │  vs Fitted   │
          └──────────────┴──────────────┴──────────────┘
        """
        d = self.model._model.get_depth()
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(
            f"Decision Tree Regression — Diagnostic Dashboard\n"
            f"criterion='{self.model.criterion}'  |  "
            f"depth={d}  |  "
            f"leaves={self.model._model.get_n_leaves()}  |  "
            f"n_features={len(self.model.feature_names_)}",
            fontsize=13, fontweight="bold"
        )

        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        self.plot_tree_diagram(max_depth=4, ax=ax1)
        self.plot_1d_fit(ax=ax2)
        self.plot_feature_importance(ax=ax3)
        self.plot_depth_analysis(ax=ax4)
        self.plot_actual_vs_pred(X_test, y_test, ax=ax5)
        self.plot_residuals(X_test, y_test, ax=ax6)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[Visualizer] Dashboard saved to: {save_path}")
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X, y,
    feature_names=None,
    test_size: float = 0.2,
    auto_tune: bool = True,
    max_depth=None,
    criterion: str = "squared_error",
    min_samples_leaf: int = 1,
    cv: int = 5,
    plot: bool = True,
    save_plot: str = None,
    print_rules: bool = True,
    explain_samples: int = 2,
    random_state: int = 42,
) -> dict:
    """
    End-to-end Decision Tree Regression pipeline in a single function call.

    PIPELINE STEPS:
    ───────────────
    1.  Train/test split (no stratification — continuous target)
    2.  (Optional) Tune max_depth, min_samples_leaf, criterion via CV
    3.  Fit the Decision Tree Regressor
    4.  Print model summary (tree stats, leaf distribution, importances)
    5.  Print IF-THEN regression rules
    6.  Explain N sample predictions (path trace through tree)
    7.  Evaluate on train AND test (MSE, RMSE, MAE, R²)
    8.  K-fold CV for robust performance estimate
    9.  Generate 6-panel diagnostic dashboard

    Parameters
    ----------
    X              : feature matrix (array or DataFrame)
    y              : continuous target
    feature_names  : column names (auto-detected from DataFrame)
    test_size      : fraction held out for testing
    auto_tune      : search for best max_depth, criterion, min_samples_leaf
    max_depth      : used if auto_tune=False
    criterion      : used if auto_tune=False
    min_samples_leaf: used if auto_tune=False
    cv             : number of CV folds
    plot           : display diagnostic dashboard
    save_plot      : path to save the dashboard image
    print_rules    : print IF-THEN decision rules after fitting
    explain_samples: number of training points to trace through the tree
    random_state   : seed for reproducibility

    Returns
    -------
    dict: model, tuner, evaluator, visualizer,
          train_metrics, test_metrics, cv_metrics
    """
    print("\n" + "█" * 62)
    print("  DECISION TREE REGRESSION — FULL PIPELINE")
    print("█" * 62)

    # ── 1. Prepare
    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X = np.array(X)
    y = np.array(y).ravel()
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    print(f"\n[Pipeline] {X.shape[0]} samples | {X.shape[1]} features")
    print(f"[Pipeline] Y range: [{y.min():.2f}, {y.max():.2f}]  "
          f"mean={y.mean():.2f}  std={y.std():.2f}")
    print(f"[Pipeline] NOTE: Decision Trees do NOT require feature scaling.")

    # ── 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[Pipeline] Split — Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Tune
    tuner = DecisionTreeRegressionTuner(cv=cv, random_state=random_state)
    if auto_tune:
        print("\n[Pipeline] Step 1/6 — Hyperparameter Tuning ...")
        tuner.fit(X_train, y_train)
        tuner.results_summary()
        max_depth        = tuner.best_params_.get("max_depth")
        criterion        = tuner.best_params_.get("criterion", criterion)
        min_samples_leaf = tuner.best_params_.get("min_samples_leaf", min_samples_leaf)
    else:
        print(f"\n[Pipeline] Step 1/6 — Using max_depth={max_depth}, "
              f"criterion={criterion}")
        kf    = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        for d in list(range(1, 16)) + [None]:
            m  = DecisionTreeRegressor(max_depth=d, random_state=random_state)
            sc = -cross_val_score(m, X_train, y_train, cv=kf,
                                   scoring="neg_mean_squared_error")
            r2 = cross_val_score(m, X_train, y_train, cv=kf, scoring="r2")
            tuner.depth_rmse_[str(d)] = (np.sqrt(sc).mean(), np.sqrt(sc).std())
            tuner.depth_r2_[str(d)]   = r2.mean()
        tuner.best_params_ = {
            "max_depth": max_depth,
            "criterion": criterion,
            "min_samples_leaf": min_samples_leaf,
        }

    # ── 4. Fit
    print("\n[Pipeline] Step 2/6 — Fitting Decision Tree Regressor ...")
    model = DecisionTreeRegressionModel(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    # ── 5. Print rules
    if print_rules:
        model.print_rules(max_depth=4)

    # ── 6. Explain sample predictions
    if explain_samples > 0:
        print(f"[Pipeline] Step 3/6 — Prediction traces "
              f"({explain_samples} samples) ...")
        for i in range(min(explain_samples, len(X_train))):
            model.explain_prediction(X_train[i])

    # ── 7. Evaluate
    evaluator = DecisionTreeRegressionEvaluator(model)
    print("\n[Pipeline] Step 4/6 — Evaluating ...")
    train_metrics = evaluator.evaluate(X_train, y_train, label="Train Set")
    test_metrics  = evaluator.evaluate(X_test,  y_test,  label="Test Set")

    print("[Pipeline] Step 5/6 — Cross-Validation ...")
    cv_metrics = evaluator.cross_validate(X, y, cv=cv)

    # ── 8. Plot
    visualizer = DecisionTreeRegressionVisualizer(model, tuner)
    if plot:
        print("\n[Pipeline] Step 6/6 — Generating dashboard ...")
        visualizer.plot_all(X_test, y_test, save_path=save_plot)

    print("\n" + "█" * 62)
    print("  PIPELINE COMPLETE")
    print("█" * 62 + "\n")

    return dict(
        model=model, tuner=tuner, evaluator=evaluator,
        visualizer=visualizer, train_metrics=train_metrics,
        test_metrics=test_metrics, cv_metrics=cv_metrics
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    print("\n" + "=" * 62)
    print("  DECISION TREE REGRESSION MODULE — DEMO")
    print("  Synthetic regression dataset (300 samples, 8 features)")
    print("=" * 62 + "\n")

    X_demo, y_demo = make_regression(
        n_samples=300, n_features=8, n_informative=5,
        noise=15, bias=50, random_state=42
    )
    names = [f"feature_{i:02d}" for i in range(8)]

    results = run_full_pipeline(
        X=X_demo, y=y_demo, feature_names=names,
        test_size=0.2, auto_tune=True, cv=5,
        plot=True, save_plot="dt_regression_diagnostics.png",
        print_rules=True, explain_samples=2,
    )

    model = results["model"]
    print("\n── Quick reference ──────────────────────────────")
    print("  model.predict(X_new)          → continuous Y predictions")
    print("  model.predict_single(x)       → float prediction")
    print("  model.explain_prediction(x)   → step-by-step trace")
    print("  model.print_rules()           → full IF-THEN rules")
    print("  model.get_feature_importance()→ MSE-reduction ranking")