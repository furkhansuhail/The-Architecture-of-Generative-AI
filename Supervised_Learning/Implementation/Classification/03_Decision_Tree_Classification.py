"""
================================================================================
  DECISION TREE CLASSIFICATION MODULE
  A complete, plug-and-play module for Decision Tree Classification with
  detailed step-by-step breakdowns of every concept and line of code.
================================================================================

WHAT IS A DECISION TREE?
─────────────────────────
  A Decision Tree is a flowchart-like model that learns a sequence of
  IF-THEN-ELSE rules from training data to classify new samples.

  Think of it as 20 Questions — the model asks a series of yes/no
  questions about the features, and each answer narrows down the class.

  Example tree for cancer diagnosis:
    Is tumor_size > 2.5?
    ├── YES → Is cell_uniformity > 3?
    │         ├── YES → MALIGNANT (85% confidence)
    │         └── NO  → BENIGN (72% confidence)
    └── NO  → BENIGN (91% confidence)

  Each internal node  = a question about one feature
  Each branch         = the answer (True / False, or a range)
  Each leaf node      = a final class prediction + confidence

HOW A DECISION TREE IS BUILT (The CART Algorithm):
────────────────────────────────────────────────────
  CART = Classification And Regression Trees (sklearn uses CART)

  The tree is built TOP-DOWN, GREEDY:

  At each node, the algorithm:
    1. Tries every feature and every possible split threshold
    2. Picks the split that produces the PUREST child nodes
       (measured by Gini Impurity or Entropy)
    3. Recurses on each child until a stopping condition is met

  WHAT IS PURITY?
  ───────────────
  A node is PURE if all samples in it belong to the same class.
  A pure leaf = zero uncertainty = perfect local classification.

  GINI IMPURITY (default in sklearn):
  ─────────────────────────────────────
      Gini(node) = 1 - Σ pᵢ²

  where pᵢ = fraction of class i in that node.

    Gini = 0.0  → perfectly pure (all samples same class) ← best
    Gini = 0.5  → maximally impure (50/50 split)          ← worst (binary)

  Example:
    Node with [70 Benign, 30 Malignant]:
    p₁ = 0.7, p₂ = 0.3
    Gini = 1 - (0.7² + 0.3²) = 1 - (0.49 + 0.09) = 0.42

  INFORMATION GAIN / ENTROPY (alternative):
  ───────────────────────────────────────────
      Entropy(node) = -Σ pᵢ · log₂(pᵢ)

    Entropy = 0   → perfectly pure
    Entropy = 1   → maximally impure (binary case)

  INFORMATION GAIN of a split:
      IG = Entropy(parent) - [weighted avg Entropy(children)]

  The split chosen maximizes IG (equivalently, minimizes weighted child entropy).

HOW EACH SPLIT IS CHOSEN:
───────────────────────────
  For each feature f and threshold t:
    LEFT child  ← samples where feature_f ≤ t
    RIGHT child ← samples where feature_f > t

  Score = Gini(LEFT) × |LEFT|/|PARENT|  +  Gini(RIGHT) × |RIGHT|/|PARENT|

  The (feature, threshold) pair with the LOWEST weighted Gini is chosen.
  This is repeated recursively until:
    - All leaves are pure, OR
    - Max depth reached (max_depth), OR
    - Minimum samples per leaf (min_samples_leaf), OR
    - Minimum impurity decrease (min_impurity_decrease)

WHY DECISION TREES ARE UNIQUE vs PREVIOUS MODELS:
───────────────────────────────────────────────────
  Model              Decision Mechanism
  ─────────────────  ──────────────────────────────────────────────
  Logistic Reg       Linear boundary: P = σ(w·x + b)
  KNN                Distance to K nearest neighbors → majority vote
  Naive Bayes        Bayes' theorem: P(class) × P(features|class)
  SVM                Maximum margin hyperplane
  Decision Tree      Axis-aligned rectangular regions via binary splits
                     → produces non-linear but interpretable boundaries

KEY STRENGTHS:
  ✅ Fully interpretable — you can print/visualize the exact rules
  ✅ Handles mixed types — numeric and categorical features
  ✅ No feature scaling needed (splits are threshold-based, not distance-based)
  ✅ Captures non-linear relationships without kernels
  ✅ Feature importance — measures which features reduce impurity most

KEY WEAKNESSES:
  ❌ High variance — small changes in data → very different trees
  ❌ Prone to overfitting without pruning (max_depth, min_samples_leaf)
  ❌ Axis-aligned boundaries — struggles with diagonal decision boundaries
  ❌ Biased toward features with many split points (high cardinality)

CONTENTS:
─────────
  1. DecisionTreeClassificationModel  — core model with tree extraction
  2. DecisionTreeClassificationTuner  — depth/leaf tuning via CV + complexity
  3. DecisionTreeClassificationEvaluator — full metrics + feature importance
  4. DecisionTreeClassificationVisualizer — tree plot, boundary, importance,
                                            complexity analysis, full dashboard
  5. run_full_pipeline()              — one-call: tune → fit → evaluate → plot

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

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, log_loss
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — CORE MODEL CLASS
#
#  NOTE ON FEATURE SCALING:
#  ─────────────────────────
#  Unlike SVM, KNN, or Logistic Regression, Decision Trees do NOT need
#  feature scaling. Each split is based on a THRESHOLD comparison:
#      "Is feature_3 > 4.7?"
#  The answer is the same whether feature_3 is raw or scaled.
#  This is one of the practical advantages of tree-based models.
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeClassificationModel:
    """
    Decision Tree Classifier — interpretable, non-parametric, no scaling needed.

    THE CORE BUILDING PROCESS (what happens during .fit()):
    ────────────────────────────────────────────────────────
    At each node (starting from the root with ALL training data):

      1. For every feature f in {0 ... n_features-1}:
           For every threshold t in sorted unique values of f:
             Compute weighted Gini of the proposed split
      2. Pick (f*, t*) that minimizes weighted Gini
      3. Split data into LEFT (feature_f ≤ t*) and RIGHT (feature_f > t*)
      4. Recurse on LEFT and RIGHT children
      5. Stop if: pure leaf | max_depth reached | min_samples_leaf too small

    Parameters
    ----------
    criterion : str, default='gini'
        Impurity measure used to evaluate splits:
          'gini'    → Gini Impurity = 1 - Σpᵢ²
                      Faster to compute, preferred default
          'entropy' → Information Gain = -Σpᵢlog₂pᵢ
                      Tends to produce more balanced trees

    max_depth : int or None, default=None
        Maximum depth of the tree.
        None → grow until all leaves are pure (likely overfits)
        3-5  → good starting range for most problems
        Larger depth → more complex rules → higher variance

    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
        Increase to prevent splits that affect very few samples.
        Equivalent to "don't split unless there's enough data to be meaningful."

    min_samples_leaf : int, default=1
        Minimum samples required to be at a leaf node.
        This is a regularization parameter — larger values → simpler tree.
        Setting to 5-20 effectively prunes small, noisy leaves.

    max_features : int, float, str, or None, default=None
        Number of features considered at each split.
        None  → all features (standard Decision Tree)
        'sqrt'→ sqrt(n_features) — used in Random Forest
        Restricting features reduces correlation between trees in ensembles.

    max_leaf_nodes : int or None, default=None
        Limit total number of leaf nodes. Alternative to max_depth.
        Grows the tree best-first (most impurity-reducing splits first).

    ccp_alpha : float, default=0.0
        Complexity parameter for Minimal Cost-Complexity Pruning.
        0.0    → no pruning
        Larger → more aggressive pruning (removes branches with low benefit)
        This is POST-TRAINING pruning — the tree grows fully first, then prunes.

    Example
    -------
    >>> model = DecisionTreeClassificationModel(max_depth=4, criterion='gini')  # doctest: +SKIP
    >>> model.fit(X_train, y_train)                                              # doctest: +SKIP
    >>> preds = model.predict(X_test)                                            # doctest: +SKIP
    >>> model.summary()                                                          # doctest: +SKIP
    >>> model.print_rules()                                                      # doctest: +SKIP
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        max_leaf_nodes=None,
        ccp_alpha: float = 0.0,
        random_state: int = 42,
    ):
        self.criterion         = criterion
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.max_leaf_nodes    = max_leaf_nodes
        self.ccp_alpha         = ccp_alpha
        self.random_state      = random_state

        self._model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )
        self._label_encoder = LabelEncoder()

        self.feature_names_ = None
        self.classes_       = None
        self.is_binary_     = None
        self.is_fitted_     = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        """
        Build the decision tree by recursively finding the best splits.

        STEP-BY-STEP INTERNAL PROCESS:
        ────────────────────────────────
        Step 1 — Convert inputs, encode labels, store feature names.
                 Unlike SVM/KNN/LogReg, NO scaling is applied.
                 Decision trees are invariant to monotonic transformations
                 of features — the best split threshold changes, but the
                 same feature and relative ordering is always chosen.

        Step 2 — Start at the ROOT node with ALL n training samples.
                 Compute the Gini impurity of the root node:
                     Gini = 1 - Σ(class_count / n)²

        Step 3 — GREEDY SPLIT SEARCH: try every (feature, threshold) pair.
                 For a numeric feature with values [1, 2, 3, 5, 8]:
                   Candidate thresholds = midpoints = [1.5, 2.5, 4.0, 6.5]
                   Each threshold creates a LEFT/RIGHT partition.
                   Score = weighted average Gini of left + right.

        Step 4 — Choose the split with the MINIMUM weighted Gini.
                 This split is "recorded" in the node:
                   node.feature   = which feature to split on
                   node.threshold = the threshold value

        Step 5 — Recurse: apply Step 2-4 to LEFT and RIGHT subsets.
                 Continue until stopping conditions are met.

        Step 6 — Build leaf nodes: each leaf stores:
                   leaf.value         = class counts [n_class0, n_class1, ...]
                   predicted class    = argmax(counts)
                   probability        = counts / sum(counts)

        Parameters
        ----------
        X            : array-like (n_samples, n_features) — NO scaling needed
        y            : array-like (n_samples,) — class labels
        feature_names: optional list of column names
        """
        X = np.array(X)
        y = np.array(y).ravel()

        # Step 1 — Store names and encode labels
        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"feature_{i}" for i in range(X.shape[1])]
        )
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_   = self._label_encoder.classes_
        self.is_binary_ = len(self.classes_) == 2

        # Steps 2-6 — Build the tree (no scaling)
        self._model.fit(X, y_enc)
        self.is_fitted_ = True

        n_nodes  = self._model.tree_.node_count
        n_leaves = self._model.get_n_leaves()
        depth    = self._model.get_depth()

        print(
            f"[DecisionTreeClassifier] Fitted\n"
            f"  Criterion       : {self.criterion}  "
            f"({'Gini Impurity: 1-Σpᵢ²' if self.criterion == 'gini' else 'Entropy: -Σpᵢlog₂pᵢ'})\n"
            f"  Max depth set   : {self.max_depth}  "
            f"(actual depth = {depth})\n"
            f"  Total nodes     : {n_nodes}  "
            f"({n_leaves} leaves + {n_nodes - n_leaves} internal)\n"
            f"  Training samples: {X.shape[0]}\n"
            f"  Features        : {X.shape[1]}\n"
            f"  Classes         : {list(self.classes_)}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Classify new samples by traversing the tree from root to leaf.

        HOW PREDICTION WORKS AT RUNTIME:
        ──────────────────────────────────
        For each new sample x:

          1. Start at the ROOT node
          2. Read the node's split rule: "Is feature_k ≤ threshold?"
          3. If YES → go LEFT child | If NO → go RIGHT child
          4. Repeat until a LEAF node is reached
          5. The leaf's majority class = the prediction

        This is O(depth) per sample — extremely fast prediction.
        A tree of depth 5 takes at most 5 comparisons per prediction,
        regardless of how many training samples there are.

        Contrast with KNN (O(n_train)) or SVM (O(n_support_vectors)):
        Trees are much faster at prediction time.

        Returns
        -------
        y_pred : array of original class labels
        """
        self._check_fitted()
        y_enc = self._model.predict(np.array(X))
        return self._label_encoder.inverse_transform(y_enc)

    # ── predict_proba ─────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Return class probability estimates.

        HOW TREE PROBABILITIES WORK:
        ─────────────────────────────
        At each leaf, the training samples that ended up there define
        the class distribution:

            P(class_j | leaf) = count(class_j in leaf) / total_in_leaf

        Example: If a leaf contains [40 Benign, 10 Malignant]:
            P(Benign)    = 40/50 = 0.80
            P(Malignant) = 10/50 = 0.20

        NOTE: Tree probabilities are COARSE — like KNN, they can only take
        values that are ratios of training sample counts in each leaf.
        A leaf with 1 Malignant sample = P(Malignant) = 1.0 regardless of
        how far from the true boundary that leaf really is.
        This is why trees often need calibration for probability tasks.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        return self._model.predict_proba(np.array(X))

    # ── print_rules ───────────────────────────────────────────────────────────

    def print_rules(self, max_depth: int = None):
        """
        Print the complete decision tree as human-readable IF-THEN rules.

        THIS IS DECISION TREES' KILLER FEATURE:
        ─────────────────────────────────────────
        No other model in this module series can do this. Every other
        model (SVM, KNN, Logistic Regression) requires domain knowledge
        or indirect methods to explain predictions.

        A Decision Tree's rules ARE the model. You can:
          - Copy-paste them into a business process
          - Validate them with domain experts
          - Spot spurious correlations directly
          - Implement them in any language without sklearn

        Parameters
        ----------
        max_depth : limit how many levels of rules to print
        """
        self._check_fitted()
        rules = export_text(
            self._model,
            feature_names=self.feature_names_,
            max_depth=max_depth or self._model.get_depth(),
        )
        # Decode class indices back to original labels
        for i, cls in enumerate(self.classes_):
            rules = rules.replace(f"class: {i}", f"class: {cls}")

        print("\n" + "=" * 60)
        print("  DECISION TREE RULES  (Human-readable IF-THEN format)")
        print("  Each |--- indicates one level deeper in the tree")
        print("=" * 60)
        print(rules)
        print("=" * 60 + "\n")

    # ── get_feature_importance ────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances — ranked by total impurity reduction.

        HOW FEATURE IMPORTANCE IS COMPUTED:
        ─────────────────────────────────────
        For each feature f, sum up all impurity reductions it caused
        across ALL nodes where it was used for splitting, weighted by
        the number of samples reaching that node:

            importance(f) = Σ_nodes_using_f  [
                (n_samples_at_node / n_total) × impurity_reduction
            ]

        Then normalize so all importances sum to 1.0.

        INTERPRETATION:
          importance = 0.0 → feature was never used for splitting
          importance = 1.0 → this single feature explains everything
          Higher importance → feature reduces class uncertainty more

        NOTE: This is NODE impurity importance, not the same as
        permutation importance or SHAP values. For correlated features,
        it can be misleading — use with caution.

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
        Print a structured summary: tree structure, depth analysis,
        feature importances, and leaf statistics.
        """
        self._check_fitted()
        tree    = self._model.tree_
        depth   = self._model.get_depth()
        n_leaves= self._model.get_n_leaves()
        n_nodes = tree.node_count
        fi_df   = self.get_feature_importance()

        sep = "=" * 65
        print(f"\n{sep}")
        print("  DECISION TREE CLASSIFICATION — MODEL SUMMARY")
        print(sep)
        print(f"  Task             : {'Binary' if self.is_binary_ else 'Multiclass'} Classification")
        print(f"  Classes          : {list(self.classes_)}")
        print(f"  Criterion        : {self.criterion}")
        print(f"\n  ── Tree Structure ───────────────────────────────────────")
        print(f"  Actual depth     : {depth}  (max_depth set = {self.max_depth})")
        print(f"  Total nodes      : {n_nodes}")
        print(f"  Leaf nodes       : {n_leaves}   (prediction endpoints)")
        print(f"  Internal nodes   : {n_nodes - n_leaves}  (decision nodes)")
        print(f"  Avg samples/leaf : {tree.n_node_samples[tree.children_left == -1].mean():.1f}")

        print(f"\n  ── Top 10 Feature Importances ───────────────────────────")
        print(f"  (Total impurity reduction each feature contributed)")
        print(f"  {'Feature':<25}  {'Importance':>10}  Bar")
        print(f"  {'-'*25}  {'-'*10}  {'-'*25}")
        top10 = fi_df.head(10)
        max_imp = top10["importance"].max()
        for _, row in top10.iterrows():
            bar = "█" * int(row["importance"] / max_imp * 25) if max_imp > 0 else ""
            used = "" if row["importance"] > 0 else "  (never used)"
            print(f"  {row['feature']:<25}  {row['importance']:>10.5f}  {bar}{used}")

        unused = (fi_df["importance"] == 0).sum()
        if unused > 0:
            print(f"\n  {unused} feature(s) with zero importance — never used in any split")

        print(f"\n  ── Interpretability Note ────────────────────────────────")
        print(f"  Call .print_rules() to see the full IF-THEN decision rules.")
        print(f"  This is the COMPLETE model — every prediction traces back")
        print(f"  to a specific sequence of feature comparisons.")
        print(sep + "\n")

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(X, y) first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TUNER
#
#  THE OVERFITTING PROBLEM:
#  ─────────────────────────
#  An unconstrained Decision Tree (max_depth=None) will grow until every
#  leaf is pure. On training data: 100% accuracy. On test data: often terrible.
#
#  WHY?
#  The tree memorizes the training data including noise. It learns rules like:
#    "If feature_3 = 4.71832... AND feature_7 = 2.9912... → Malignant"
#  This perfectly fits one noisy training point but generalizes poorly.
#
#  THE SOLUTION: Limit complexity via max_depth, min_samples_leaf, ccp_alpha
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeClassificationTuner:
    """
    Stratified cross-validation tuner for Decision Tree hyperparameters.

    WHAT WE TUNE:
    ─────────────
    max_depth       : limits tree growth — most impactful parameter
    min_samples_leaf: minimum leaf size — prevents over-specific rules
    criterion       : gini vs entropy
    ccp_alpha       : post-training pruning aggressiveness

    THE COST-COMPLEXITY PRUNING PATH (ccp_alpha):
    ──────────────────────────────────────────────
    sklearn can compute the full pruning path — a sequence of (alpha, tree)
    pairs where increasing alpha progressively prunes more branches.
    We visualize this alongside standard CV to give two views of optimal
    complexity.

    Example
    -------
    >>> tuner = DecisionTreeClassificationTuner(cv=5)           # doctest: +SKIP
    >>> best_depth = tuner.fit(X_train, y_train)                # doctest: +SKIP
    >>> tuner.results_summary()                                 # doctest: +SKIP
    """

    def __init__(self, cv: int = 5, scoring: str = "accuracy",
                 random_state: int = 42):
        self.cv           = cv
        self.scoring      = scoring
        self.random_state = random_state

        self.best_params_  = None
        self.best_score_   = None
        self.depth_scores_ = {}    # depth → (mean, std) CV score
        self.ccp_alphas_   = None  # pruning path alphas
        self.ccp_scores_   = None  # CV scores per alpha
        self._grid_search  = None

    def fit(self, X, y, param_grid: dict = None):
        """
        Run stratified GridSearchCV over max_depth, min_samples_leaf,
        criterion, and optionally ccp_alpha.

        STEP-BY-STEP:
        ─────────────
        Step 1 — Define the parameter grid.
                 max_depth range: 1 (stump) to ~15 (complex)
                 min_samples_leaf: 1 (no constraint) to 20 (pruned)

        Step 2 — StratifiedKFold CV: preserves class distribution in folds.
                 For each combination, train and evaluate on k folds.

        Step 3 — Record per-depth scores for the overfitting analysis plot.

        Step 4 — Also compute the Cost-Complexity Pruning path (bonus):
                 sklearn's cost_complexity_pruning_path() returns the
                 sequence of (alpha, impurities) as the tree is pruned.
                 CV over these alphas gives another view of optimal complexity.

        Returns
        -------
        best_depth : int
        """
        X = np.array(X)
        y = np.array(y).ravel()

        if param_grid is None:
            param_grid = {
                "max_depth":        [2, 3, 4, 5, 6, 8, 10, 12, None],
                "min_samples_leaf": [1, 2, 5, 10, 20],
                "criterion":        ["gini", "entropy"],
            }

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                              random_state=self.random_state)

        n_combos = np.prod([len(v) for v in param_grid.values()])
        print(f"[DecisionTreeClassificationTuner] Searching {n_combos} "
              f"combinations with {self.cv}-fold stratified CV ...")

        base = DecisionTreeClassifier(random_state=self.random_state)
        self._grid_search = GridSearchCV(
            base, param_grid, cv=skf,
            scoring=self.scoring, n_jobs=-1, refit=True,
        )
        self._grid_search.fit(X, LabelEncoder().fit_transform(y))

        self.best_params_ = self._grid_search.best_params_
        self.best_score_  = self._grid_search.best_score_

        # Per-depth scores (for elbow / overfitting plot)
        le    = LabelEncoder()
        y_enc = le.fit_transform(y)
        depths = list(range(1, 16)) + [None]
        for d in depths:
            m = DecisionTreeClassifier(
                max_depth=d, random_state=self.random_state
            )
            sc = cross_val_score(m, X, y_enc, cv=skf, scoring=self.scoring)
            self.depth_scores_[str(d)] = (sc.mean(), sc.std())

        # Cost-Complexity Pruning path
        clf_full = DecisionTreeClassifier(random_state=self.random_state)
        clf_full.fit(X, y_enc)
        path = clf_full.cost_complexity_pruning_path(X, y_enc)
        alphas = path.ccp_alphas[:-1]     # skip last (trivial tree)
        cv_sc  = []
        for alpha in alphas:
            m  = DecisionTreeClassifier(
                ccp_alpha=alpha, random_state=self.random_state
            )
            sc = cross_val_score(m, X, y_enc, cv=skf, scoring=self.scoring)
            cv_sc.append(sc.mean())
        self.ccp_alphas_  = alphas
        self.ccp_scores_  = np.array(cv_sc)

        best_d = self.best_params_["max_depth"]
        best_l = self.best_params_["min_samples_leaf"]
        best_c = self.best_params_["criterion"]
        print(f"[DecisionTreeClassificationTuner] ✓ "
              f"max_depth={best_d}, min_samples_leaf={best_l}, "
              f"criterion={best_c}\n"
              f"  Best {self.scoring}: {self.best_score_:.4f}")
        return best_d

    def results_summary(self):
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        print("\n" + "=" * 55)
        print("  DECISION TREE TUNING RESULTS")
        print("=" * 55)
        for k, v in self.best_params_.items():
            print(f"  {k:<25} : {v}")
        print(f"  Best {self.scoring:<20} : {self.best_score_:.4f}")
        print("=" * 55 + "\n")

    def get_best_model(self) -> "DecisionTreeClassificationModel":
        if self.best_params_ is None:
            raise RuntimeError("Call .fit() first.")
        return DecisionTreeClassificationModel(**{
            k: v for k, v in self.best_params_.items()
            if k in DecisionTreeClassificationModel.__init__.__code__.co_varnames
        })


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeClassificationEvaluator:
    """
    Comprehensive classification evaluator — same metrics as previous modules
    plus tree-specific diagnostics: feature importance and tree structure stats.

    Example
    -------
    >>> ev = DecisionTreeClassificationEvaluator(model)   # doctest: +SKIP
    >>> ev.evaluate(X_test, y_test)                       # doctest: +SKIP
    >>> ev.cross_validate(X, y, cv=5)                     # doctest: +SKIP
    """

    def __init__(self, model: DecisionTreeClassificationModel):
        self.model    = model
        self.metrics_ = {}

    def evaluate(self, X, y, label: str = "Test Set") -> dict:
        """
        Compute all classification metrics.

        Returns dict: Accuracy, Precision, Recall, F1, ROC_AUC, Log_Loss
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
                    self.model._label_encoder.transform(y), y_prob[:, 1]
                )
            else:
                auc = roc_auc_score(
                    self.model._label_encoder.transform(y),
                    y_prob, multi_class="ovr", average="weighted"
                )
        except Exception:
            auc = float("nan")

        self.metrics_ = dict(
            Accuracy=acc, Precision=prec, Recall=rec,
            F1=f1, ROC_AUC=auc, Log_Loss=ll
        )

        depth = self._model_depth()
        sep   = "=" * 58
        print(f"\n{sep}")
        print(f"  EVALUATION — {label}  (depth={depth}, criterion={self.model.criterion})")
        print(sep)
        print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}% correct)")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  ROC-AUC    : {auc:.4f}")
        print(f"  Log Loss   : {ll:.4f}")
        report   = classification_report(y, y_pred, zero_division=0)
        indented = "\n".join("    " + l for l in report.splitlines())
        print(f"\n  Classification Report:\n{indented}")
        print(sep + "\n")
        return self.metrics_

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Stratified k-fold CV — no Pipeline needed (trees don't need scaling).
        """
        X = np.array(X)
        y = np.array(y).ravel()

        le    = LabelEncoder()
        y_enc = le.fit_transform(y)

        m   = self.model._model
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        acc = cross_val_score(m, X, y_enc, cv=skf, scoring="accuracy")
        f1  = cross_val_score(m, X, y_enc, cv=skf, scoring="f1_weighted")
        try:
            sc = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr_weighted"
            auc = cross_val_score(m, X, y_enc, cv=skf, scoring=sc)
        except Exception:
            auc = np.array([float("nan")] * cv)

        results = {
            "Accuracy_mean": acc.mean(), "Accuracy_std": acc.std(),
            "F1_mean":       f1.mean(),  "F1_std":       f1.std(),
            "AUC_mean":      auc.mean(), "AUC_std":      auc.std(),
        }
        print(f"\n{'='*58}")
        print(f"  {cv}-FOLD STRATIFIED CROSS-VALIDATION  (depth={self._model_depth()})")
        print(f"{'='*58}")
        print(f"  Accuracy  mean ± std : {results['Accuracy_mean']:.4f} ± {results['Accuracy_std']:.4f}")
        print(f"  F1        mean ± std : {results['F1_mean']:.4f} ± {results['F1_std']:.4f}")
        print(f"  ROC-AUC   mean ± std : {results['AUC_mean']:.4f} ± {results['AUC_std']:.4f}")
        print(f"{'='*58}\n")
        return results

    def _model_depth(self):
        return self.model._model.get_depth() if self.model.is_fitted_ else "?"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeClassificationVisualizer:
    """
    Six diagnostic plots for Decision Tree Classification.

    PLOTS:
    ──────
    1. plot_tree_diagram       — the actual tree (sklearn's plot_tree)
    2. plot_decision_boundary  — axis-aligned rectangular class regions
    3. plot_feature_importance — bar chart of impurity reductions
    4. plot_depth_analysis     — CV accuracy vs depth (overfitting curve)
    5. plot_confusion_matrix   — TP/FP/TN/FN heatmap
    6. plot_roc_curve          — ROC curve
    7. plot_all                — 2×3 full dashboard

    Example
    -------
    >>> viz = DecisionTreeClassificationVisualizer(model, tuner)  # doctest: +SKIP
    >>> viz.plot_all(X_test, y_test)                              # doctest: +SKIP
    """

    def __init__(self, model: DecisionTreeClassificationModel,
                 tuner: DecisionTreeClassificationTuner = None):
        self.model  = model
        self.tuner  = tuner
        self._pal   = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6"]

    # ── 1. Tree Diagram ───────────────────────────────────────────────────────

    def plot_tree_diagram(self, max_depth: int = 4, ax=None):
        """
        Visualize the actual decision tree structure.

        WHAT EACH NODE SHOWS:
        ──────────────────────
        feature_k ≤ threshold  — the split rule for this node
        gini = X.XXX           — impurity before the split (lower = purer)
        samples = NNN          — how many training samples reach this node
        value = [n0, n1, ...]  — count per class in this node
        class = LABEL          — majority class prediction at this node

        Leaf nodes show the final prediction with no split rule.
        Node color = majority class (intensity = purity, darker = purer).
        """
        self.model._check_fitted()

        if ax is None:
            depth  = min(max_depth, self.model._model.get_depth())
            height = max(6, 2 ** depth)
            fig, ax = plt.subplots(figsize=(min(20, 3 * 2**depth), height))

        plot_tree(
            self.model._model,
            feature_names=self.model.feature_names_,
            class_names=[str(c) for c in self.model.classes_],
            filled=True,
            rounded=True,
            max_depth=max_depth,
            ax=ax,
            fontsize=8,
            impurity=True,
            proportion=False,
        )
        d = self.model._model.get_depth()
        ax.set_title(
            f"Decision Tree Structure  (depth={d})\n"
            f"Blue/orange = class majority  |  Darker = purer node",
            fontsize=10
        )

    # ── 2. Decision Boundary ─────────────────────────────────────────────────

    def plot_decision_boundary(self, X, y, resolution=0.04, ax=None):
        """
        Visualize the AXIS-ALIGNED rectangular decision regions.

        DECISION TREE BOUNDARIES ARE UNIQUE:
        ──────────────────────────────────────
        Unlike SVM (smooth curves) or Logistic Regression (straight line),
        Decision Tree boundaries are AXIS-ALIGNED RECTANGLES formed by the
        sequence of threshold splits.

        Each vertical/horizontal line corresponds to one split node.
        The rectangular regions = the prediction zones for each class.

        You can literally trace any point back to its prediction by
        following the horizontal and vertical boundary lines.

        Uses PCA projection if features > 2.
        """
        self.model._check_fitted()
        X = np.array(X)
        y = np.array(y).ravel()

        if X.shape[1] > 2:
            pca  = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            le   = LabelEncoder()
            y_enc = le.fit_transform(y)
            vis   = DecisionTreeClassifier(
                max_depth=self.model.max_depth,
                criterion=self.model.criterion,
                random_state=self.model.random_state,
            )
            vis.fit(X_2d, y_enc)
            classes = le.classes_
            xlabel, ylabel = "PC 1", "PC 2"
            title_sfx = " (PCA projection — boundaries are approximate)"
        else:
            X_2d   = X
            y_enc  = self.model._label_encoder.transform(y)
            vis    = self.model._model
            classes = self.model.classes_
            xlabel = self.model.feature_names_[0]
            ylabel = self.model.feature_names_[1]
            title_sfx = "  (axis-aligned rectangles = tree splits)"

        n_cls  = len(classes)
        cmap_bg = [(*plt.cm.tab10(i / max(n_cls, 2))[:3], 0.22)
                   for i in range(n_cls)]
        cmap_fg = [plt.cm.tab10(i / max(n_cls, 2)) for i in range(n_cls)]

        x_min, x_max = X_2d[:, 0].min() - .5, X_2d[:, 0].max() + .5
        y_min, y_max = X_2d[:, 1].min() - .5, X_2d[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                              np.arange(y_min, y_max, resolution))
        Z = vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        ax.contourf(xx, yy, Z, alpha=0.30,
                    cmap=ListedColormap([c for c in cmap_bg]),
                    levels=np.arange(-0.5, n_cls + 0.5, 1))
        # Draw the axis-aligned boundary lines (the tree splits)
        ax.contour(xx, yy, Z, colors="black", linewidths=0.8,
                   levels=np.arange(-0.5, n_cls + 0.5, 1))

        for i, cls in enumerate(classes):
            mask = y_enc == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       color=cmap_fg[i], edgecolors="white",
                       s=35, alpha=0.85, label=str(cls))

        depth = self.model._model.get_depth()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Decision Tree Boundary (depth={depth}){title_sfx}")
        ax.legend(fontsize=8)

    # ── 3. Feature Importance ────────────────────────────────────────────────

    def plot_feature_importance(self, top_n: int = 20, ax=None):
        """
        Horizontal bar chart of feature importances (Gini/Entropy reduction).

        KEY INSIGHT FOR DECISION TREES:
        ─────────────────────────────────
        Importance = 0 → feature was NEVER used in any split
                         (the tree completely ignored this feature)
        Importance > 0 → proportional to how much it reduced impurity
                         across all nodes it appeared in

        This is UNIQUE to tree models — you know exactly which features
        the model actually uses vs ignores. Logistic Regression has
        coefficients, but they don't tell you which splits happened.
        """
        self.model._check_fitted()
        fi = self.model.get_feature_importance().head(top_n)

        colors = ["#E74C3C" if i == 0 else "#3498DB"
                  for i in range(len(fi))]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.38)))

        ax.barh(fi["feature"], fi["importance"],
                color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Feature Importance  (total Gini/Entropy reduction)")
        ax.set_title(
            f"Decision Tree Feature Importances\n"
            f"(criterion={self.model.criterion}, depth={self.model._model.get_depth()})"
        )
        ax.invert_yaxis()

        zero_count = (self.model._model.feature_importances_ == 0).sum()
        if zero_count > 0:
            ax.text(0.98, 0.02, f"{zero_count} features with 0 importance (unused)",
                    transform=ax.transAxes, ha="right", fontsize=8, color="grey")

    # ── 4. Depth Analysis (Overfitting Curve) ────────────────────────────────

    def plot_depth_analysis(self, ax=None):
        """
        CV accuracy vs tree depth — reveals the overfitting sweet spot.

        THE CLASSICAL TREE OVERFITTING PATTERN:
        ─────────────────────────────────────────
        Depth 1 (stump): underfits — only one decision rule
        Optimal depth  : balances fit and generalization
        Deep tree      : overfits training data, poor on test

        Training accuracy monotonically increases with depth (can reach 1.0).
        CV accuracy peaks at the optimal depth and then either plateaus or drops.

        The gap between training and CV accuracy = overfitting gap.
        Best depth = where CV accuracy is highest (dashed vertical line).
        """
        if self.tuner is None or not self.tuner.depth_scores_:
            print("[Visualizer] No tuner results. Run tuner.fit() first.")
            return

        depths  = []
        means   = []
        stds    = []
        for k, (m, s) in self.tuner.depth_scores_.items():
            depths.append(k)
            means.append(m)
            stds.append(s)

        labels = [str(d) for d in depths]
        x      = np.arange(len(labels))
        means  = np.array(means)
        stds   = np.array(stds)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(x, means, color="#3498DB", linewidth=2.5,
                marker="o", markersize=5, label="CV Accuracy (mean)")
        ax.fill_between(x, means - stds, means + stds,
                        alpha=0.15, color="#3498DB", label="±1 std")

        best_d = str(self.tuner.best_params_.get("max_depth"))
        if best_d in labels:
            bi = labels.index(best_d)
            ax.axvline(bi, color="#E74C3C", linestyle="--",
                       linewidth=1.8, label=f"Best depth = {best_d}")
            ax.scatter([bi], [means[bi]], color="#E74C3C", s=100, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("max_depth  (None = unlimited)")
        ax.set_ylabel(f"Cross-validated Accuracy")
        ax.set_title("Decision Tree Depth Analysis\n"
                     "Find the depth where CV accuracy peaks (avoid underfitting & overfitting)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 5. Confusion Matrix ───────────────────────────────────────────────────

    def plot_confusion_matrix(self, X, y, normalize: bool = False, ax=None):
        """
        Confusion matrix heatmap — TP/FP/TN/FN.
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

        im = ax.imshow(cm_show, interpolation="nearest", cmap="Greens")
        plt.colorbar(im, ax=ax)
        thresh = cm_show.max() / 2.0
        for i in range(cm_show.shape[0]):
            for j in range(cm_show.shape[1]):
                ax.text(j, i, f"{cm_show[i,j]:{fmt}}",
                        ha="center", va="center", fontsize=12,
                        fontweight="bold",
                        color="white" if cm_show[i,j] > thresh else "black")
        ax.set_xticks(range(len(self.model.classes_)))
        ax.set_yticks(range(len(self.model.classes_)))
        ax.set_xticklabels(self.model.classes_, fontsize=9)
        ax.set_yticklabels(self.model.classes_, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix  (depth={self.model._model.get_depth()})")

        if self.model.is_binary_:
            for i in range(2):
                for j in range(2):
                    ax.text(j, i - 0.35,
                            [["TN","FP"],["FN","TP"]][i][j],
                            ha="center", fontsize=8,
                            color="grey", style="italic")

    # ── 6. ROC Curve ─────────────────────────────────────────────────────────

    def plot_roc_curve(self, X, y, ax=None):
        """
        ROC curve for binary Decision Tree classification.

        NOTE ON TREE ROC CURVES:
        ─────────────────────────
        Like KNN, tree probabilities are COARSE — limited to the set of
        class ratios that appear in leaf nodes. The ROC curve will be stepped.
        A deeper tree → more leaves → more distinct probability levels → smoother ROC.
        """
        if not self.model.is_binary_:
            print("[Visualizer] ROC curve for binary classification only.")
            return

        y     = np.array(y).ravel()
        y_enc = self.model._label_encoder.transform(y)
        y_prob = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_enc, y_prob)
        auc = roc_auc_score(y_enc, y_prob)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.step(fpr, tpr, color="#27AE60", linewidth=2.5, where="post",
                label=f"Decision Tree  (AUC = {auc:.4f})")
        ax.plot([0,1],[0,1], color="grey", linestyle="--",
                linewidth=1.2, label="Random (AUC = 0.50)")
        ax.fill_between(fpr, tpr, step="post", alpha=0.1, color="#27AE60")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve  (depth={self.model._model.get_depth()})\n"
                     f"Stepped shape = coarse leaf probabilities")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    # ── 7. Full Dashboard ─────────────────────────────────────────────────────

    def plot_all(self, X, y, save_path: str = None):
        """
        2×3 diagnostic dashboard for Decision Tree Classification.
        """
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(
            f"Decision Tree Classification — Diagnostic Dashboard\n"
            f"criterion='{self.model.criterion}'  |  "
            f"depth={self.model._model.get_depth()}  |  "
            f"leaves={self.model._model.get_n_leaves()}",
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
        self.plot_decision_boundary(X, y, ax=ax2)
        self.plot_feature_importance(ax=ax3)
        self.plot_depth_analysis(ax=ax4)
        self.plot_confusion_matrix(X, y, ax=ax5)
        self.plot_roc_curve(X, y, ax=ax6)

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
    criterion: str = "gini",
    min_samples_leaf: int = 1,
    cv: int = 5,
    plot: bool = True,
    save_plot: str = None,
    print_rules: bool = True,
    random_state: int = 42,
) -> dict:
    """
    End-to-end Decision Tree Classification pipeline.

    PIPELINE STEPS:
    ───────────────
    1.  Stratified train/test split
    2.  (Optional) Tune max_depth, min_samples_leaf, criterion via CV
    3.  Fit Decision Tree — build split hierarchy
    4.  Print model summary (tree structure, feature importances)
    5.  Print IF-THEN rules (unique to trees!)
    6.  Evaluate on train AND test sets
    7.  Stratified k-fold CV
    8.  Generate 6-panel diagnostic dashboard

    Returns dict: model, tuner, evaluator, visualizer,
                  train_metrics, test_metrics, cv_metrics
    """
    print("\n" + "█" * 62)
    print("  DECISION TREE CLASSIFICATION — FULL PIPELINE")
    print("█" * 62)

    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X = np.array(X)
    y = np.array(y).ravel()
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    n_classes = len(np.unique(y))
    print(f"\n[Pipeline] {X.shape[0]} samples | {X.shape[1]} features | "
          f"{n_classes} classes: {np.unique(y)}")
    print(f"[Pipeline] NOTE: Decision Trees do NOT require feature scaling.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[Pipeline] Split — Train: {len(X_train)}, Test: {len(X_test)}")

    tuner = DecisionTreeClassificationTuner(cv=cv, random_state=random_state)
    if auto_tune:
        print("\n[Pipeline] Step 1/5 — Hyperparameter Tuning ...")
        tuner.fit(X_train, y_train)
        tuner.results_summary()
        max_depth        = tuner.best_params_.get("max_depth")
        criterion        = tuner.best_params_.get("criterion", criterion)
        min_samples_leaf = tuner.best_params_.get("min_samples_leaf", min_samples_leaf)
    else:
        print(f"\n[Pipeline] Step 1/5 — Using max_depth={max_depth}, "
              f"criterion={criterion}")
        # Still populate depth scores for elbow plot
        le    = LabelEncoder()
        y_enc = le.fit_transform(y_train)
        skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for d in list(range(1, 16)) + [None]:
            m  = DecisionTreeClassifier(max_depth=d, random_state=random_state)
            sc = cross_val_score(m, X_train, y_enc, cv=skf, scoring="accuracy")
            tuner.depth_scores_[str(d)] = (sc.mean(), sc.std())
        tuner.best_params_ = {
            "max_depth": max_depth,
            "criterion": criterion,
            "min_samples_leaf": min_samples_leaf,
        }

    print("\n[Pipeline] Step 2/5 — Fitting Decision Tree ...")
    model = DecisionTreeClassificationModel(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    if print_rules:
        model.print_rules(max_depth=5)

    evaluator = DecisionTreeClassificationEvaluator(model)
    print("\n[Pipeline] Step 3/5 — Evaluating ...")
    train_metrics = evaluator.evaluate(X_train, y_train, label="Train Set")
    test_metrics  = evaluator.evaluate(X_test,  y_test,  label="Test Set")
    cv_metrics    = evaluator.cross_validate(X, y, cv=cv)

    visualizer = DecisionTreeClassificationVisualizer(model, tuner)
    if plot:
        print("\n[Pipeline] Step 4/5 — Generating dashboard ...")
        visualizer.plot_all(X_test, y_test, save_path=save_plot)

    print("\n" + "█" * 62)
    print("  PIPELINE COMPLETE")
    print("█" * 62 + "\n")

    return dict(model=model, tuner=tuner, evaluator=evaluator,
                visualizer=visualizer, train_metrics=train_metrics,
                test_metrics=test_metrics, cv_metrics=cv_metrics)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X_demo, y_demo = make_classification(
        n_samples=600, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42
    )
    y_demo = np.where(y_demo == 1, "Malignant", "Benign")
    names  = [f"feature_{i:02d}" for i in range(10)]

    results = run_full_pipeline(
        X=X_demo, y=y_demo, feature_names=names,
        test_size=0.2, auto_tune=True, cv=5,
        plot=True, save_plot="dt_classification_diagnostics.png",
        print_rules=True,
    )