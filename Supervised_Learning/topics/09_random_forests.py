"""
Random Forests — Wisdom of a Crowd of Trees
============================================

A single decision tree is fast, interpretable, and non-linear — but it suffers
from one crippling weakness: high variance. Small changes in training data produce
dramatically different trees. Random Forests solve this by training hundreds of
trees on randomised subsets of the data and features, then aggregating their
predictions. The result is a model that is more accurate, far more stable, and
surprisingly robust — all without a single hyperparameter that requires deep
domain knowledge to tune.

Random Forests are one of the most widely deployed machine learning algorithms
in industry. They require minimal preprocessing, handle mixed feature types, give
reliable out-of-the-box performance, and produce built-in estimates of their own
generalisation error.

"""

import base64
import os
import textwrap
import re

DISPLAY_NAME = "09 · Random Forest"
ICON         = "⚔️"
SUBTITLE     = "Random Forest Breakdown and Explanation"
TOPIC_NAME = "Random Forests"

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg",
                "gif":"image/gif","svg":"image/svg+xml"}.get(ext,"image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is a Random Forest?

A Random Forest is an ensemble of decision trees where each tree is trained on
a different randomised view of the data. Predictions are made by aggregating
the votes (classification) or averaging the outputs (regression) of all trees.

The key insight comes from statistics: the average of many noisy-but-unbiased
estimators is less noisy than any single estimator. If we could train many trees
that each made independent errors, the errors would cancel out on averaging. The
challenge is that trees trained on the same data are not independent — they make
correlated errors. Random Forests break this correlation through two sources of
randomness:

    1. Bootstrap sampling     — each tree sees a different random subset of rows
    2. Feature subsampling    — each split considers only a random subset of columns

Together, these make each tree different enough that their errors partially cancel,
producing an ensemble that is dramatically more stable and accurate than any single
tree — without adding bias.

Think of it like polling. Asking one expert gives you a noisy estimate. Asking 500
independent experts and averaging their answers gives a much more reliable estimate.
The key word is independent — if all experts read the same newspaper, they'll make
the same errors and the average won't help. Random Forests are a way of engineering
independence between trees trained on the same dataset.

    Things that exist inside the model (learned during training):
        - The full set of decision trees (each with its own structure and splits)
        - OOB (out-of-bag) error estimate — a free validation metric

    Things you control before training (hyperparameters):
        - n_estimators           — number of trees (more = better, diminishing returns)
        - max_features           — features considered per split (key randomness knob)
        - max_depth              — depth limit per tree (None = grow fully)
        - min_samples_leaf       — minimum examples per leaf
        - bootstrap              — whether to use bootstrap sampling (default True)


### Random Forests as Empirical Risk Minimisation (ERM)

Random Forests extend the decision tree ERM formulation with an ensemble layer:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Hypothesis class:  H = { majority-vote / average of B      │
    │                           randomised decision trees }       │
    │                                                             │
    │  Loss function:     0-1 loss (classification) or            │
    │                     MSE (regression) — evaluated on OOB     │
    │                                                             │
    │  Training objective:                                        │
    │      For each tree b = 1...B:                               │
    │        Sample n rows with replacement (bootstrap)           │
    │        At each split: consider only m < p features          │
    │        Grow a deep tree (low bias)                          │
    │      Aggregate: ŷ = majority_vote({T_b(x)})                 │
    │                                                             │
    │  Optimiser:  No global optimisation — each tree uses        │
    │              greedy CART splitting independently            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

There is no gradient descent, no loss function being minimised jointly across trees.
Each tree is trained independently. The "learning" happens through the averaging
operation: the ensemble's bias matches a single tree's bias, but its variance is
reduced by a factor of roughly B (number of trees).


### The Inductive Bias of Random Forests

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Random Forests inherit the decision tree inductive bias:   │
    │    - Axis-aligned splits (no linearity assumption)          │
    │    - Piecewise-constant output per region                   │
    │    - Non-parametric (can approximate any function)          │
    │                                                             │
    │  And add two ensemble beliefs:                              │
    │    - DIVERSITY is valuable — trees that disagree and are    │
    │      both partially right produce better aggregates         │
    │    - LOCAL MAJORITY — the prediction for any input should   │
    │      be determined by the consensus of trees that each      │
    │      "saw" that region of feature space differently         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The inductive bias is weaker than a single tree because the ensemble can express
more complex boundaries (each tree votes on boundary regions differently). But the
features still must be relevant through axis-aligned comparisons.


---


### Part 1: Bagging — Bootstrap Aggregating


### The Variance Problem:

From the decision tree module, we know that a single tree has high variance. Formally,
if a learning algorithm A has prediction variance σ² at any test point, and we train B
independent models and average them, the variance of the average is:

                        Var(ŷ_avg) = σ² / B

The variance shrinks by the number of models. With B=100 trees, variance shrinks
100-fold — in principle. The catch: the trees must be independent.

If all trees are identical (trained on the same data, same algorithm), averaging
does nothing: Var(ŷ_avg) = σ² (no improvement). We need diversity.


### Bootstrap Sampling:

For each tree b out of B:
    1. Draw n samples FROM THE TRAINING SET WITH REPLACEMENT (bootstrap sample)
    2. Train a full decision tree on this bootstrap sample

Each bootstrap sample contains approximately 63.2% unique training examples
(some are duplicated, some are absent). The remaining ~36.8% are the "out-of-bag"
(OOB) examples — training points not seen by tree b.

**Why 63.2%? The Poisson Approximation:**

The probability that a specific example is NOT selected in any of n draws is:
    (1 − 1/n)^n  → e^(-1) ≈ 0.368  as n → ∞

So each bootstrap sample includes ~63.2% of unique examples, leaving ~36.8% OOB.


    # =======================================================================================# 
    **Diagram 1 — Bootstrap Sampling for 3 Trees:**

    FULL TRAINING SET (n=10 examples):  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Tree 1 bootstrap: [2, 2, 4, 7, 7, 1, 9, 3, 5, 6]  ← 8 unique, 2 duplicated
    Tree 1 OOB:       [8, 10]                           ← never seen by Tree 1

    Tree 2 bootstrap: [1, 3, 3, 6, 8, 9, 2, 4, 10, 5] ← 9 unique, 1 duplicated
    Tree 2 OOB:       [7]

    Tree 3 bootstrap: [5, 6, 6, 2, 9, 1, 4, 3, 8, 10] ← 8 unique, 2 duplicated
    Tree 3 OOB:       [7, 11]... wait, only 10 examples, some just unused

    Each tree sees a slightly different training distribution.
    The differences in data → trees make different errors → errors partially cancel.

    PREDICTION (new example x):
        Tree 1 predicts: Class A
        Tree 2 predicts: Class A
        Tree 3 predicts: Class B
        Final: majority vote → Class A  (2/3 trees agree)
    # =======================================================================================# 


### The Out-of-Bag (OOB) Error Estimate:

The OOB examples for tree b are examples the tree never trained on — they function
as a validation set. For each training example i, we can get a prediction from
all trees for which example i was OOB. Aggregating these OOB predictions gives
an estimate of generalisation error without ever holding out a separate validation set.

    OOB prediction for example i = majority vote of {T_b(xᵢ) : i ∈ OOB_b}
    OOB error = fraction of training examples where OOB prediction ≠ true label

**This is remarkable:** a free, unbiased estimate of generalisation error is obtained
as a byproduct of the training process. No cross-validation needed.

The OOB error estimate is nearly identical to k-fold cross-validation with k ≈ 1/0.368 ≈ 3
folds, but costs nothing extra because the OOB sets are created automatically during training.


---


### Part 2: Feature Subsampling — Creating Tree Diversity


### The Correlation Problem with Pure Bagging:

If we only use bootstrap sampling (without feature randomness), all trees will tend
to split on the same few highly predictive features at their roots. This makes the
trees correlated — they make similar errors, and averaging correlated errors gives
less reduction in variance than averaging independent errors.

Consider a dataset where Feature A is very predictive and features B–Z are less so.
Every bootstrap-sampled tree will use Feature A at or near the root. Despite being
different trees, they all heavily rely on Feature A in the same way. Their predictions
are strongly correlated, and the variance reduction from averaging is small.


### Random Feature Subsets at Each Split:

At each split, instead of searching all p features for the best threshold, Random
Forests randomly select m < p features and only consider those as candidates:

    Classification:  m = sqrt(p)   (sklearn default)
    Regression:      m = p/3       (sklearn default)

This means even if Feature A is the globally best feature, it won't be considered
at every split of every tree — sometimes other features get their chance. This
forces the trees to use different features, decorrelating their predictions.


    # =======================================================================================# 
    **Diagram 2 — Feature Subsampling: Decorrelating Trees:**

    DATASET: p=6 features  [A, B, C, D, E, F]
    Feature A is the most predictive.  m = sqrt(6) ≈ 2 features per split.

    Tree 1, root split:  candidates [A, D]  → best is A  → split on A
    Tree 2, root split:  candidates [B, E]  → best is B  → split on B  ← DIFFERENT!
    Tree 3, root split:  candidates [A, C]  → best is A  → split on A
    Tree 4, root split:  candidates [D, F]  → best is D  → split on D  ← DIFFERENT!

    Without feature subsampling (bagging only):
        All 4 trees would pick A at the root (it's always the best).
        Trees are correlated → averaging reduces variance by < 4x.

    With feature subsampling:
        Trees pick different features → less correlated.
        Averaging reduces variance much closer to 4x.

    EFFECT:
    ─────────────────────────────────────────────────────────────
    m = p (no feature randomness):   trees correlated, small variance gain
    m = sqrt(p):                     trees decorrelated, large variance gain
    m = 1:                           maximum diversity, high bias per tree
    ─────────────────────────────────────────────────────────────
    # =======================================================================================# 


### The Bias-Variance Decomposition of Ensembles:

Let ρ = correlation between predictions of any two trees. Then for B trees:

            Var(ensemble) = ρ · σ² + (1 − ρ) · σ²/B

Where σ² is the variance of a single tree. Two components:
    - ρ · σ²:         irreducible component — adding more trees doesn't help here
    - (1 − ρ) · σ²/B: reducible component — shrinks as B grows

As B → ∞:   Var → ρ · σ²   (a lower bound — can't go below this)

**To minimise ensemble variance, we need both:**
    1. Small ρ (low correlation between trees) — achieved by feature subsampling
    2. Large B (many trees) — but diminishing returns after some point

Feature subsampling directly reduces ρ. Without it, ρ is close to 1 and the
lower bound ρ·σ² ≈ σ² — no improvement over a single tree.


---


### Part 3: The Aggregation Step


### Classification — Majority Voting:

For a new example x, each of the B trees outputs a class label prediction.
The forest predicts the class with the most votes:

            ŷ = argmax_c  Σ_b  1[T_b(x) = c]

Soft voting (probability averaging) is often better: average the per-class
probabilities from each tree, then predict the class with the highest mean probability.

    Hard voting:    each tree votes once, majority wins
    Soft voting:    each tree outputs [P(c=0), P(c=1), ...], average probabilities

Soft voting is preferred because it takes into account the confidence of each tree,
not just its top prediction.


### Regression — Mean Averaging:

For regression, each tree outputs a continuous value. The forest averages:

                    ŷ = (1/B) Σ_b  T_b(x)

This is equivalent to the mean of the piecewise-constant functions from each tree.
The ensemble produces a smoother approximation of the true function than any single tree.


'''

**Diagram 3 — How Averaging Smooths the Regression:**

TRUE FUNCTION: y = sin(x)   (smooth curve)

SINGLE TREE:              RANDOM FOREST (B=100 trees):

y ↑                        y ↑
  │    ┌──┐  ┌──┐            │     /‾‾\
  │    │  │  │  │            │    /    \
  │────┘  └──┘  └────        │   /      \
  │                          │  /        \u005f___
  │staircase (step fn)       │ smooth curve
  └──────────────→ x         └──────────────→ x

    Each tree is a staircase. 100 different staircases, slightly offset
    from each other due to bootstrap randomness, average into a smooth curve.

'''


### Convergence with Number of Trees:

Adding more trees always reduces variance (or keeps it the same). The test error
of a Random Forest converges as B → ∞ — it never increases with more trees.

This is unlike other hyperparameters (max_depth, C) where there is a sweet spot.
For n_estimators, more is always better, but with strongly diminishing returns.
Common practice: use 100–500 trees; rarely need more than 1000.

    B = 10:   high variance — ensemble still heavily influenced by individual trees
    B = 100:  good — most of the variance reduction achieved
    B = 500:  very stable — error has converged
    B = 5000: marginal improvement — 10× the compute for <1% gain


---


### Part 4: Out-of-Bag Error — Free Validation


### Using OOB Predictions:

For each training example xᵢ, we collect predictions only from the trees in whose
bootstrap sample xᵢ was absent (OOB trees for xᵢ):

    OOB prediction for xᵢ = majority_vote  or  mean  of {T_b(xᵢ) : i ∉ bootstrap_b}

On average, each example has approximately B × 0.368 ≈ 37% of B trees available
as OOB predictors. For B=100, that is ~37 trees per example.

OOB error = fraction of training examples incorrectly predicted by their OOB ensemble.


### OOB Error vs Cross-Validation:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  OOB error:                                                 │
    │    ✓ Free — computed during training at no extra cost       │
    │    ✓ No data wasted — all examples are in the training set  │
    │    ✓ Approximately equivalent to 3-fold CV accuracy         │
    │    ✗ Slightly optimistic for small B (few OOB trees)        │
    │    ✗ Can't tune hyperparameters fairly (OOB uses all data)  │
    │                                                             │
    │  Cross-validation:                                          │
    │    ✓ More reliable for hyperparameter tuning                │
    │    ✓ Exact fold-level control                               │
    │    ✗ Expensive — must retrain k times                       │
    │    ✗ Wastes 1/k of data for each fold                       │
    │                                                             │
    │  Recommendation: use OOB for quick sanity check and         │
    │  approximate error estimate; use CV for hyperparameter      │
    │  tuning and final model selection.                          │
    └─────────────────────────────────────────────────────────────┘


---


### Part 5: Feature Importance in Random Forests


### Mean Decrease in Impurity (MDI):

Exactly as in single trees, but averaged across all B trees:

    MDI(feature f) = (1/B) Σ_b  Σ_{nodes in T_b where f is used}  (n_node/n) × gain

Features used consistently across many trees and at high-impact nodes receive
large MDI scores. Features that rarely appear or only in deep, low-impact nodes
receive small scores.

**Inherited caveat:** MDI is biased toward high-cardinality features. A feature
with 100 unique values has more thresholds to consider than a binary feature, so
it's more likely to produce a high-gain split by chance.


### Mean Decrease in Accuracy (MDA) / Permutation Importance:

For each tree b and each feature f:
    1. Compute prediction accuracy on the OOB examples of tree b
    2. Randomly permute the values of feature f in the OOB examples
    3. Recompute OOB accuracy
    4. The difference is tree b's importance estimate for feature f

Average across all B trees:

    MDA(feature f) = mean over trees of  [Acc_OOB(b) − Acc_OOB_permuted(b, f)]

MDA is more reliable than MDI because:
    - It measures actual predictive contribution, not a proxy (impurity)
    - It's evaluated on OOB examples (unseen data), not training data
    - It is not biased toward high-cardinality features


    # =======================================================================================# 
    **Diagram 4 — MDI vs MDA/Permutation Importance:**

    MDI: measured during training (impurity reduction in the tree)
    MDA: measured after training (performance drop from feature shuffling)

    Example: A feature correlated with the label by chance in training data

    MDI  →  HIGH (it reduces training impurity, even if it's noise)
    MDA  →  LOW  (shuffling it doesn't hurt OOB accuracy — it IS noise)

    MDI can be fooled by noise features with many unique values.
    MDA catches this because OOB examples reveal the true predictive signal.
    Always compare both — large MDI + small MDA = suspicious (likely noise).
    # =======================================================================================# 


---


### Part 6: Extremely Randomised Trees (Extra-Trees)


### One Step Further in Randomness:

Extra-Trees (sklearn's `ExtraTreesClassifier`) push randomisation one step further:
at each split, instead of finding the best threshold for each candidate feature,
they draw a RANDOM threshold for each feature and pick the best among random splits.

    Random Forest:      random features + best threshold per feature
    Extra-Trees:        random features + random threshold per feature

This makes individual trees even weaker learners — more biased, higher training
error per tree. But it also makes the trees more decorrelated (lower ρ), which
means averaging gives even larger variance reduction.

    ┌────────────────────────────────────────────────────────────────────┐
    │                                Random Forest     Extra-Trees       │
    │────────────────────────────────────────────────────────────────────│
    │  Threshold search     Best per feature    Random per feature       │
    │  Per-tree bias        Moderate            Higher                   │
    │  Per-tree variance    Moderate            Lower                    │
    │  Tree correlation ρ   Moderate            Lower                    │
    │  Training speed       Slower (search)     Faster (no search)       │
    │  Ensemble bias        ≈ same as RF        Slightly higher          │
    │  Ensemble variance    Moderate            Lower                    │
    │  Typical accuracy     Slightly better     Slightly worse or equal  │
    └────────────────────────────────────────────────────────────────────┘

In practice, Random Forest and Extra-Trees often perform similarly. Extra-Trees
train significantly faster and can win when the dataset is very noisy.


---


### Part 7: Random Forests vs Other Models — When to Use Each


    ──────────────────────────────────────────────────────────────────────────────
    Property                   RF          LR/SVM      DT          Neural Net
    ──────────────────────────────────────────────────────────────────────────────
    Interpretability           Low         High        Very high   Very low
    Handles non-linearity      Yes         No (lin)    Yes         Yes
    Feature scaling needed     No          Yes         No          Yes
    Missing values             Partial     No          No          No
    Calibrated probabilities   Partial     Yes (LR)    No          Yes
    Out-of-the-box performance Excellent   Moderate    Poor        Moderate
    Scalability (large n)      Good        Excellent   Good        Excellent
    Scalability (large p)      Good        Excellent   Good        Good
    Training speed             Moderate    Fast        Fast        Slow
    Extrapolation              No          Yes         No          Partial
    Hyperparameter sensitivity Low         Moderate    High        Very high
    ──────────────────────────────────────────────────────────────────────────────

**Random Forests excel when:**
    - Tabular data with mixed feature types
    - You need good performance with minimal preprocessing
    - Interpretability is secondary to accuracy
    - Dataset is medium-sized (n = 1k–1M)
    - You want reliable uncertainty estimates (OOB, prediction variance)

**Random Forests struggle when:**
    - You need a human-readable model (use a single tree)
    - Data has strong temporal structure (use sequence models)
    - Extrapolation is required (all tree-based models fail here)
    - Extremely high-dimensional sparse data (n << p, use LR or SVM)
    - Very large n (> 10M): gradient boosting or neural networks scale better


### The Bridge to Gradient Boosting:

Random Forests reduce variance by averaging. But what if we want to reduce bias too?
Gradient Boosted Trees (the next module) do this by training trees sequentially,
each one fitting the residual errors of the previous ensemble:

    Random Forest:          B trees in PARALLEL, each independent
                            → reduces variance, bias unchanged

    Gradient Boosting:      B trees SEQUENTIALLY, each corrects previous errors
                            → reduces bias AND variance, but more hyperparameters

    The full progression from this module:

    Single Tree     →   high variance, interpretable
    Random Forest   →   low variance (averaging), less interpretable
    Gradient Boost  →   low bias + low variance (boosting), least interpretable,
                        but typically highest accuracy on tabular data

    """


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
──────────────────────────────────────────────────────────────────────────
Operation                   Complexity              Notes
──────────────────────────────────────────────────────────────────────────
Training (B trees)          O(B · n · m · log n)    m = max_features
Prediction                  O(B · depth)            fast; parallelisable
OOB error computation       O(B · n_oob · depth)    free during training
MDI importance              O(B · n · m · log n)    during training
MDA / permutation imp.      O(B · n_oob · p)        post-training
Memory                      O(B · 2^depth)          B full trees stored
──────────────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Random Forest from Scratch": {
        "description": "Bootstrap sampling, feature subsampling, majority vote — built with only NumPy + our DecisionTree",
        "runnable": True,
        "code": '''
"""
================================================================================
RANDOM FOREST FROM SCRATCH
================================================================================

We build a Random Forest on top of the DecisionTreeClassifier from Module 04.
This illustrates the exact mechanism without any sklearn magic:

    1. Bootstrap sample the training data (with replacement)
    2. For each split, randomly restrict which features can be considered
    3. Train a full decision tree on the bootstrap sample
    4. Repeat B times, storing all trees
    5. Predict by majority vote

We also compute the OOB error manually.

================================================================================
"""

import numpy as np
from collections import Counter


# =============================================================================
# DECISION TREE (from Module 04 — reproduced here for self-containment)
# =============================================================================

def gini_impurity(y):
    if len(y) == 0: return 0.0
    counts = np.bincount(y)
    probs  = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

def find_best_split(X, y, max_features=None):
    """
    Find the best (feature, threshold) pair, considering only max_features
    randomly selected features at each call.
    """
    parent_gini = gini_impurity(y)
    n, p        = X.shape

    # Randomly select which features to consider at this split
    if max_features is None or max_features >= p:
        feature_indices = np.arange(p)
    else:
        feature_indices = np.random.choice(p, max_features, replace=False)

    best_feat, best_thresh, best_gain = None, None, -np.inf

    for feat in feature_indices:
        vals       = np.unique(X[:, feat])
        thresholds = (vals[:-1] + vals[1:]) / 2

        for thresh in thresholds:
            left  = y[X[:, feat] <= thresh]
            right = y[X[:, feat] >  thresh]
            if len(left) == 0 or len(right) == 0:
                continue
            n_l, n_r = len(left), len(right)
            w_gini   = (n_l / n) * gini_impurity(left) + (n_r / n) * gini_impurity(right)
            gain     = parent_gini - w_gini
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh

    return best_feat, best_thresh, best_gain


class Node:
    def __init__(self):
        self.feature  = None; self.threshold = None
        self.left     = None; self.right     = None
        self.is_leaf  = False; self.pred     = None

def _grow(X, y, max_features, max_depth, min_samples_leaf, depth):
    node = Node()
    if (len(np.unique(y)) == 1 or len(y) < 2 or
            (max_depth is not None and depth >= max_depth)):
        node.is_leaf = True
        node.pred    = Counter(y).most_common(1)[0][0]
        return node
    feat, thresh, gain = find_best_split(X, y, max_features)
    if feat is None or gain <= 0:
        node.is_leaf = True; node.pred = Counter(y).most_common(1)[0][0]
        return node
    lm = X[:, feat] <= thresh; rm = ~lm
    if lm.sum() < min_samples_leaf or rm.sum() < min_samples_leaf:
        node.is_leaf = True; node.pred = Counter(y).most_common(1)[0][0]
        return node
    node.feature = feat; node.threshold = thresh
    node.left    = _grow(X[lm], y[lm], max_features, max_depth, min_samples_leaf, depth+1)
    node.right   = _grow(X[rm], y[rm], max_features, max_depth, min_samples_leaf, depth+1)
    return node

def _predict_one(x, node):
    if node.is_leaf: return node.pred
    return _predict_one(x, node.left) if x[node.feature] <= node.threshold else _predict_one(x, node.right)


# =============================================================================
# RANDOM FOREST CLASS
# =============================================================================

class RandomForestClassifier:
    """
    Random Forest for binary and multi-class classification.

    Key mechanisms implemented:
        - Bootstrap sampling (sample n rows with replacement)
        - Random feature subsampling at each split (max_features)
        - Majority vote aggregation
        - OOB error estimation (free validation without a held-out set)

    Parameters (hyperparameters — not learned):
        n_estimators  (int):   number of trees in the forest
        max_features  (int or 'sqrt'): features considered per split
        max_depth     (int):   max depth per tree (None = grow fully)
        min_samples_leaf (int): min examples required in each leaf
        random_state  (int):   random seed for reproducibility
    """

    def __init__(self, n_estimators=100, max_features="sqrt",
                 max_depth=None, min_samples_leaf=1, random_state=None):
        self.n_estimators     = n_estimators
        self.max_features     = max_features
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.trees_           = []      # list of (root_node, oob_indices)
        self.n_classes_       = None
        self.oob_score_       = None
        self.feature_importances_ = None

    def fit(self, X, y, verbose=True):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n, p = X.shape
        self.n_classes_ = len(np.unique(y))
        self.trees_     = []

        # OOB accumulation: for each training example, collect predictions
        # from all trees where that example was OOB
        oob_predictions = [[] for _ in range(n)]   # list of lists

        # Resolve max_features
        if self.max_features == "sqrt":
            m = max(1, int(np.sqrt(p)))
        elif self.max_features == "log2":
            m = max(1, int(np.log2(p)))
        elif isinstance(self.max_features, int):
            m = self.max_features
        else:
            m = p   # use all features (bagging only, no feature subsampling)

        if verbose:
            print(f"  Training {self.n_estimators} trees "
                  f"(max_features={m} of {p}, max_depth={self.max_depth})")

        for b in range(self.n_estimators):
            # ── STEP 1: Bootstrap sample ──────────────────────────────────────
            # Draw n indices with replacement
            boot_idx = np.random.choice(n, n, replace=True)
            oob_idx  = np.array([i for i in range(n) if i not in set(boot_idx)])

            X_boot   = X[boot_idx]
            y_boot   = y[boot_idx]

            # ── STEP 2: Grow a full tree on the bootstrap sample ──────────────
            root = _grow(X_boot, y_boot, max_features=m,
                         max_depth=self.max_depth,
                         min_samples_leaf=self.min_samples_leaf,
                         depth=0)
            self.trees_.append((root, oob_idx))

            # ── STEP 3: Record OOB predictions ────────────────────────────────
            for i in oob_idx:
                pred = _predict_one(X[i], root)
                oob_predictions[i].append(pred)

            if verbose and (b + 1) % 20 == 0:
                print(f"    Tree {b+1:>3}/{self.n_estimators} grown")

        # ── STEP 4: Compute OOB error ─────────────────────────────────────────
        oob_preds_final = []
        oob_true        = []
        for i in range(n):
            if len(oob_predictions[i]) > 0:
                vote = Counter(oob_predictions[i]).most_common(1)[0][0]
                oob_preds_final.append(vote)
                oob_true.append(y[i])

        if len(oob_true) > 0:
            oob_acc        = np.mean(np.array(oob_preds_final) == np.array(oob_true))
            self.oob_score_ = oob_acc
        else:
            self.oob_score_ = None

        return self

    def predict(self, X):
        """Majority vote across all trees."""
        all_preds = np.array([
            [_predict_one(x, root) for x in X]
            for root, _ in self.trees_
        ])  # shape: (n_estimators, n_test)
        # Majority vote per example
        return np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])


# =============================================================================
# DEMONSTRATION
# =============================================================================

np.random.seed(42)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

print("=" * 65)
print("  RANDOM FOREST FROM SCRATCH")
print("=" * 65)
print(f"\\n  Dataset: n=300, p=10 features (5 informative, 2 redundant)")

# --- Single tree baseline ---
from sklearn.tree import DecisionTreeClassifier as SklearnDT
single = SklearnDT(random_state=42)
single.fit(X_tr, y_tr)
single_train = accuracy_score(y_tr, single.predict(X_tr))
single_test  = accuracy_score(y_te, single.predict(X_te))

print(f"\\n  Single Decision Tree:")
print(f"    Train accuracy: {single_train:.4f}")
print(f"    Test accuracy:  {single_test:.4f}")

# --- Our Random Forest ---
rf = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                             max_depth=None, random_state=42)
rf.fit(X_tr, y_tr)
rf_train = accuracy_score(y_tr, rf.predict(X_tr))
rf_test  = accuracy_score(y_te, rf.predict(X_te))

print(f"\\n  Random Forest (n_estimators=100, max_features=sqrt(p)):")
print(f"    Train accuracy: {rf_train:.4f}")
print(f"    Test accuracy:  {rf_test:.4f}")
print(f"    OOB accuracy:   {rf.oob_score_:.4f}  ← free estimate, no test set needed")


# =============================================================================
# STEP-BY-STEP: ONE BOOTSTRAP SAMPLE
# =============================================================================

print("\\n" + "=" * 65)
print("  STEP-BY-STEP: BOOTSTRAP SAMPLING MECHANICS")
print("=" * 65)

np.random.seed(7)
n_demo = 10
demo_idx = np.arange(n_demo)
boot     = np.random.choice(n_demo, n_demo, replace=True)
oob      = np.array(sorted(set(demo_idx) - set(boot)))
unique_boot = len(set(boot))

print(f"""
  Full training set indices: {demo_idx.tolist()}  (n={n_demo})
  Bootstrap sample:          {boot.tolist()}
  Unique in bootstrap:       {sorted(set(boot))}  ({unique_boot}/{n_demo} = {100*unique_boot/n_demo:.0f}%)
  Out-of-bag (OOB):          {oob.tolist()}  ({len(oob)} examples)

  Expected: ~{100*(1 - (1 - 1/n_demo)**n_demo):.1f}% unique,  ~{100*((1 - 1/n_demo)**n_demo):.1f}% OOB
  (For large n: ~63.2% unique, ~36.8% OOB)

  The OOB examples were never seen by this tree during training.
  They act as a held-out validation set — free of charge.
""")


# =============================================================================
# CONVERGENCE WITH NUMBER OF TREES
# =============================================================================

print("=" * 65)
print("  CONVERGENCE: TEST ACCURACY vs NUMBER OF TREES")
print("=" * 65)

print(f"\\n  {'B (trees)':>10} | {'Test Acc':>10} | {'OOB Acc':>10}")
print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}")

from sklearn.ensemble import RandomForestClassifier as SklearnRF

for B in [1, 5, 10, 25, 50, 100, 200, 500]:
    rf_b = SklearnRF(n_estimators=B, random_state=42, oob_score=True, n_jobs=-1)
    rf_b.fit(X_tr, y_tr)
    te_acc  = accuracy_score(y_te, rf_b.predict(X_te))
    oob_acc = rf_b.oob_score_
    print(f"  {B:>10} | {te_acc:>10.4f} | {oob_acc:>10.4f}")

print(f"""
  Observations:
    B=1:   single bootstrap tree — similar to a single decision tree
    B=10:  large improvement — averaging 10 trees cuts most of the variance
    B=100: most of the gain achieved — error has nearly converged
    B=500: marginal improvement over 100 — 5× the compute for <1% gain

  Rule of thumb: B=100–200 is sufficient for most datasets.
  The OOB accuracy closely tracks the test accuracy — a reliable free estimate.
""")
''',
    },

    "Variance Reduction — The Math": {
        "description": "Why averaging works: bias-variance decomposition, correlation effect, OOB derivation",
        "runnable": True,
        "code": '''
"""
================================================================================
VARIANCE REDUCTION — THE MATHEMATICS OF ENSEMBLE AVERAGING
================================================================================

Why does averaging many noisy trees produce a better estimator?

This script demonstrates:
    1. The bias-variance decomposition of ensemble predictions
    2. How tree correlation limits variance reduction
    3. Why feature subsampling (low correlation) matters more than n_estimators
    4. Empirical verification of the ρ·σ² + (1-ρ)·σ²/B formula

================================================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: SINGLE TREE VARIANCE — DEMONSTRATING THE INSTABILITY PROBLEM
# =============================================================================

print("=" * 65)
print("  PART 1: SINGLE TREE VARIANCE ACROSS BOOTSTRAP SAMPLES")
print("=" * 65)

X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

# Train many single trees on different bootstrap samples — simulate RF instability
n_trials = 50
single_accs = []

for seed in range(n_trials):
    rng = np.random.RandomState(seed)
    idx  = rng.choice(len(X_tr), len(X_tr), replace=True)
    clf  = DecisionTreeClassifier(random_state=seed)
    clf.fit(X_tr[idx], y_tr[idx])
    acc  = accuracy_score(y_te, clf.predict(X_te))
    single_accs.append(acc)

single_accs = np.array(single_accs)
print(f"""
  Trained {n_trials} single trees, each on a different bootstrap sample.
  All trees trained on the SAME training set (just different bootstrap draws).

  Single tree accuracy distribution:
    Mean:   {single_accs.mean():.4f}
    Std:    {single_accs.std():.4f}    ← HIGH VARIANCE
    Min:    {single_accs.min():.4f}
    Max:    {single_accs.max():.4f}
    Range:  {single_accs.max()-single_accs.min():.4f}

  A single tree can give very different results depending on which bootstrap
  sample it happened to draw. This variance is what the Random Forest eliminates.
""")


# =============================================================================
# PART 2: THE ENSEMBLE AVERAGE REDUCES VARIANCE
# =============================================================================

print("=" * 65)
print("  PART 2: VARIANCE REDUCTION WITH ENSEMBLE SIZE")
print("=" * 65)

print(f"""
  THEORY: If B models have individual variance σ² and pairwise correlation ρ:

    Var(ensemble) = ρ·σ²  +  (1−ρ)·σ²/B

  Two components:
    ρ·σ²:         floor — irreducible regardless of how many trees we add
    (1−ρ)·σ²/B:  reducible — shrinks linearly with B

  As B → ∞:   Var → ρ·σ²

  If ρ = 0 (uncorrelated):  Var → 0 with B=∞   (perfect averaging)
  If ρ = 1 (identical):     Var → σ²            (no improvement at all)
""")

# Empirical demonstration: simulate many ensemble predictions
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

print(f"  Ensemble accuracy mean ± std across 30 seeds:")
print(f"  {'Method':>30} | {'Mean Acc':>10} | {'Std Acc':>10} | {'Var Acc':>10}")
print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

methods = [
    ("Single DT",               lambda s: DecisionTreeClassifier(random_state=s)),
    ("RF B=5  (sqrt features)",  lambda s: RandomForestClassifier(5,   random_state=s, n_jobs=-1)),
    ("RF B=20 (sqrt features)",  lambda s: RandomForestClassifier(20,  random_state=s, n_jobs=-1)),
    ("RF B=100 (sqrt features)", lambda s: RandomForestClassifier(100, random_state=s, n_jobs=-1)),
    ("Bagging B=100 (all feats)",lambda s: BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=s, n_jobs=-1)),
]

for name, clf_fn in methods:
    trial_accs = []
    for s in range(30):
        clf = clf_fn(s)
        clf.fit(X_tr, y_tr)
        trial_accs.append(accuracy_score(y_te, clf.predict(X_te)))
    arr = np.array(trial_accs)
    print(f"  {name:>30} | {arr.mean():>10.4f} | {arr.std():>10.4f} | {arr.var():>10.6f}")

print(f"""
  Key observations:
    Single DT:     highest variance — accuracy fluctuates a lot across seeds
    RF B=5:        large variance reduction even with only 5 trees
    RF B=100:      much lower variance — stable predictions
    Bagging B=100: no feature subsampling → trees more correlated → higher variance
                   vs RF B=100 with feature subsampling

  The variance reduction from feature subsampling (RF) vs no subsampling
  (Bagging) demonstrates the importance of decorrelating the trees.
""")


# =============================================================================
# PART 3: FEATURE SUBSAMPLING — THE CORRELATION EFFECT
# =============================================================================

print("=" * 65)
print("  PART 3: EFFECT OF max_features ON TREE CORRELATION AND ACCURACY")
print("=" * 65)

from sklearn.ensemble import RandomForestClassifier

print(f"  Fixed B=100, varying max_features:")
print(f"  {'max_features':>16} | {'Test Acc':>10} | {'OOB Acc':>10} | Interpretation")
print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*35}")

n_features = X_tr.shape[1]
configs = [
    (1,             "maximum diversity (1 feature)"),
    (2,             "very diverse"),
    (int(np.sqrt(n_features)), f"sqrt(p) = {int(np.sqrt(n_features))} (default)"),
    (n_features//2, f"p/2 = {n_features//2}"),
    (n_features,    "bagging only (all features)"),
]

for mf, label in configs:
    rf_mf = RandomForestClassifier(n_estimators=100, max_features=mf,
                                    oob_score=True, random_state=42, n_jobs=-1)
    rf_mf.fit(X_tr, y_tr)
    te_acc  = accuracy_score(y_te, rf_mf.predict(X_te))
    oob_acc = rf_mf.oob_score_
    print(f"  {mf:>16} | {te_acc:>10.4f} | {oob_acc:>10.4f} | {label}")

print(f"""
  Observations:
    max_features=1:    trees very different, but each tree is very weak → high bias
    max_features=sqrt: sweet spot — trees diverse enough AND each tree reasonable
    max_features=p:    all features considered → trees correlated → higher variance

  The default sqrt(p) for classification is a well-calibrated default across
  many datasets. Always try it first; adjust if performance is poor.
""")


# =============================================================================
# PART 4: BIAS OF ENSEMBLE = BIAS OF SINGLE TREE
# =============================================================================

print("=" * 65)
print("  PART 4: ENSEMBLE BIAS EQUALS SINGLE TREE BIAS")
print("=" * 65)

print(f"""
  An important theoretical result:

    Bias(RF) = Bias(single tree)

  Averaging does not reduce bias. If each tree is underfitting, the average
  of underfitting trees is still underfitting.

  This means: make each tree a LOW BIAS learner (grow deep, don't prune much).
  Then average to reduce the high variance that deep trees produce.

  Contrast with boosting: each tree in a boosted ensemble is a HIGH BIAS
  learner (shallow, max_depth=1–3). Boosting reduces bias sequentially.

  Demonstration:
""")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print(f"  {'Model':>35} | {'Train Acc':>10} | {'Test Acc':>10}")
print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}")

models = [
    ("Single DT (depth=2, high bias)",    DecisionTreeClassifier(max_depth=2, random_state=42)),
    ("RF depth=2 (same bias, B=100)",     RandomForestClassifier(100, max_depth=2, random_state=42, n_jobs=-1)),
    ("Single DT (full depth, low bias)",  DecisionTreeClassifier(random_state=42)),
    ("RF full depth (low bias, B=100)",   RandomForestClassifier(100, random_state=42, n_jobs=-1)),
]
for name, clf in models:
    clf.fit(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
    te_acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  {name:>35} | {tr_acc:>10.4f} | {te_acc:>10.4f}")

print(f"""
  A Random Forest of depth-2 trees (high bias) performs similarly to a
  single depth-2 tree on the test set — the bias is not reduced by averaging.

  A Random Forest of full-depth trees (low bias): the test accuracy
  improves significantly over a single full-depth tree because averaging
  reduces the high variance of individual deep trees.

  Rule: grow trees deep (low bias), then let averaging handle the variance.
""")
''',
    },

    "OOB Error and Feature Importance": {
        "description": "Out-of-bag error as free cross-validation, MDI vs permutation importance",
        "runnable": True,
        "code": '''
"""
================================================================================
OOB ERROR AND FEATURE IMPORTANCE
================================================================================

Two of Random Forests' most useful built-in diagnostics:
    1. OOB error: a free estimate of generalisation error (no test set needed)
    2. Feature importance: which features drive predictions

This script compares:
    - OOB error vs cross-validation vs test set error
    - MDI (Mean Decrease in Impurity) vs permutation importance
    - Detecting noise features using both methods

================================================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: OOB ERROR vs CROSS-VALIDATION vs TEST ERROR
# =============================================================================

print("=" * 65)
print("  PART 1: OOB ERROR vs CV vs HELD-OUT TEST ERROR")
print("=" * 65)

# Several datasets to compare
print(f"\\n  {'n':>6} | {'p':>4} | {'Test Acc':>10} | {'OOB Acc':>10} | {'5-fold CV':>10} | {'Diff (OOB-Test)':>16}")
print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*16}")

for n, p, sep in [(200, 5, 1.5), (500, 10, 1.0), (1000, 20, 0.8), (2000, 50, 0.6)]:
    X, y = make_classification(n_samples=n, n_features=p, n_informative=max(2, p//3),
                                class_sep=sep, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, oob_score=True,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    te_acc  = accuracy_score(y_te, rf.predict(X_te))
    oob_acc = rf.oob_score_
    cv_acc  = cross_val_score(RandomForestClassifier(200, random_state=42, n_jobs=-1),
                               X_tr, y_tr, cv=5).mean()
    diff    = oob_acc - te_acc
    print(f"  {n:>6} | {p:>4} | {te_acc:>10.4f} | {oob_acc:>10.4f} | {cv_acc:>10.4f} | {diff:>+16.4f}")

print(f"""
  OOB error closely tracks both cross-validation and test set error.
  The slight optimistic bias (OOB > Test) is normal and small.

  When to use OOB vs CV:
    OOB:    quick sanity check, model monitoring, n_estimators selection
    CV:     hyperparameter tuning, final model comparison, rigorous evaluation

  OOB requires no extra data splitting and adds zero computational cost.
""")


# =============================================================================
# PART 2: DETECTING NOISE FEATURES — MDI vs PERMUTATION IMPORTANCE
# =============================================================================

print("=" * 65)
print("  PART 2: MDI vs PERMUTATION IMPORTANCE — DETECTING NOISE")
print("=" * 65)

print("""
  We create a dataset with:
    - 5 truly informative features (features 0-4)
    - 5 pure noise features      (features 5-9, random numbers)
    - 5 high-cardinality noise   (features 10-14, many unique values)

  True features should have high importance.
  Noise features should have near-zero importance.
  High-cardinality noise may fool MDI but NOT permutation importance.
""")

np.random.seed(99)
n = 1000
X_real = np.random.randn(n, 5)                                       # 5 informative
X_noise_low  = np.random.randint(0, 3, (n, 5)).astype(float)         # 5 low-cardinality noise
X_noise_high = np.random.uniform(0, 100, (n, 5))                     # 5 high-cardinality noise
X_all = np.hstack([X_real, X_noise_low, X_noise_high])

# True target depends only on the first 5 features
y_all = (X_real[:, 0] + X_real[:, 1] * 0.5 - X_real[:, 2] > 0).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

# MDI importances
mdi = rf.feature_importances_

# Permutation importances on test set
perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=42)
perm_mean = perm.importances_mean

feature_names = (
    [f"Informative_{i}" for i in range(5)] +
    [f"Noise_LowCard_{i}" for i in range(5)] +
    [f"Noise_HighCard_{i}" for i in range(5)]
)

print(f"  {'Feature':>20} | {'Type':>15} | {'MDI':>8} | {'Perm Imp':>10} | MDI reliable?")
print(f"  {'-'*20}-+-{'-'*15}-+-{'-'*8}-+-{'-'*10}-+-{'-'*15}")

for i in range(15):
    feat_type = ("Informative" if i < 5
                 else "Noise-LowCard" if i < 10
                 else "Noise-HighCard")
    reliable = "✓" if i < 5 else ("✓" if mdi[i] < 0.02 else "✗ BIASED")
    print(f"  {feature_names[i]:>20} | {feat_type:>15} | {mdi[i]:>8.4f} | "
          f"{perm_mean[i]:>10.4f} | {reliable}")

print(f"""
  Analysis:
    Informative features: both MDI and permutation importance are high ✓
    Low-cardinality noise: both are near zero ✓
    High-cardinality noise: MDI may be inflated (many thresholds = lucky splits)
                            Permutation importance stays near zero ✓

  Conclusion: permutation importance is more trustworthy for noise detection.
  MDI is fast and useful for a quick view; always validate with permutation
  importance when feature selection decisions matter.
""")


# =============================================================================
# PART 3: HOW MANY OOB TREES PER EXAMPLE?
# =============================================================================

print("=" * 65)
print("  PART 3: OOB COVERAGE — HOW MANY TREES PREDICT EACH EXAMPLE?")
print("=" * 65)

print(f"""
  For n=100 training examples and B=200 trees:
  Each example is OOB in approximately 200 × 0.368 ≈ 74 trees.
  That is 74 independent predictions for each training example.

  Verify empirically:
""")

n_small = 100
B_small = 200
X_s, y_s = make_classification(n_samples=n_small, n_features=5, random_state=42)
X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

n_tr = len(X_s_tr)
oob_counts = np.zeros(n_tr, dtype=int)

np.random.seed(0)
for _ in range(B_small):
    boot = np.random.choice(n_tr, n_tr, replace=True)
    oob  = set(range(n_tr)) - set(boot)
    for i in oob:
        oob_counts[i] += 1

print(f"  Training examples (n={n_tr}), B={B_small} trees:")
print(f"    Mean OOB trees per example: {oob_counts.mean():.1f}  (expected: {B_small * 0.368:.1f})")
print(f"    Min OOB trees:              {oob_counts.min()}")
print(f"    Max OOB trees:              {oob_counts.max()}")
print(f"    Examples with OOB < 10:     {(oob_counts < 10).sum()}")

print(f"""
  OOB coverage histogram:
  Trees   | Count of training examples having that many OOB trees
  --------+---------------------------------------------------------""")

hist, edges = np.histogram(oob_counts, bins=8)
for count, (lo, hi) in zip(hist, zip(edges, edges[1:])):
    bar = "█" * (count // 2)
    print(f"  {int(lo):>3}–{int(hi):<3}  | {bar}  ({count})")

print(f"""
  Every example has many OOB trees → OOB estimates are reliable.
  With very small B (e.g., B=5), some examples might have 0 OOB trees
  and wouldn't be covered by the OOB estimate. Use B≥100 for reliable OOB.
""")
''',
    },

    "RF vs Single Tree vs Gradient Boosting": {
        "description": "Comprehensive comparison across datasets, hyperparameter sensitivity, and when to use each",
        "runnable": True,
        "code": '''
"""
================================================================================
RANDOM FOREST vs SINGLE TREE vs GRADIENT BOOSTING — FULL COMPARISON
================================================================================

This script compares decision trees, random forests, and gradient boosted trees
across:
    1. Accuracy on multiple dataset types
    2. Variance/instability (repeated training)
    3. Hyperparameter sensitivity
    4. Extrapolation failure (shared limitation of all tree-based methods)
    5. Training time vs accuracy tradeoff

================================================================================
"""

import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# =============================================================================
# PART 1: ACCURACY COMPARISON ACROSS DATASETS
# =============================================================================

print("=" * 70)
print("  PART 1: ACCURACY ACROSS DATASET TYPES")
print("=" * 70)

datasets = {
    "Linear (easy)":      make_classification(500, 10, n_informative=5, class_sep=2.0, random_state=42),
    "Linear (noisy)":     make_classification(500, 10, n_informative=3, class_sep=0.7, random_state=42),
    "Circles":            make_circles(500, noise=0.1, factor=0.5, random_state=42),
    "Moons":              make_moons(500, noise=0.2, random_state=42),
    "High-dim (p=50)":    make_classification(500, 50, n_informative=10, random_state=42),
}

models = {
    "Single DT (deep)":  DecisionTreeClassifier(random_state=42),
    "Random Forest":      RandomForestClassifier(100, random_state=42, n_jobs=-1),
    "Extra-Trees":        ExtraTreesClassifier(100, random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingClassifier(100, max_depth=3, random_state=42),
    "Logistic Reg.":      LogisticRegression(C=1.0, max_iter=1000),
}

print(f"\\n  {'Dataset':>20}", end="")
for name in models:
    print(f" | {name:>16}", end="")
print()
print("  " + "-"*20, end="")
for _ in models:
    print(f"-+-{'-'*16}", end="")
print()

for ds_name, (X, y) in datasets.items():
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42)
    print(f"  {ds_name:>20}", end="")
    for clf in models.values():
        clf_c = type(clf)(**clf.get_params())
        clf_c.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf_c.predict(X_te))
        print(f" | {acc:>16.4f}", end="")
    print()

print(f"""
  Takeaways:
    Linear data:   Logistic Regression is competitive or best (correct inductive bias)
    Non-linear:    RF/Extra-Trees/GB all outperform LR
    High-dim:      RF handles p=50 well; gradient boosting often best
    Noisy data:    RF is robust; single DT often overfits noise
""")


# =============================================================================
# PART 2: VARIANCE COMPARISON — STABILITY ACROSS RANDOM SEEDS
# =============================================================================

print("=" * 70)
print("  PART 2: VARIANCE (INSTABILITY) ACROSS 20 RANDOM SEEDS")
print("=" * 70)

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

print(f"\\n  {'Model':>30} | {'Mean Acc':>10} | {'Std Acc':>10} | {'Min Acc':>10} | {'Max Acc':>10}")
print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

eval_models = {
    "Single DT (full depth)": lambda s: DecisionTreeClassifier(random_state=s),
    "RF B=10":                lambda s: RandomForestClassifier(10, random_state=s, n_jobs=-1),
    "RF B=100":               lambda s: RandomForestClassifier(100, random_state=s, n_jobs=-1),
    "Gradient Boosting":      lambda s: GradientBoostingClassifier(100, random_state=s),
}

for name, clf_fn in eval_models.items():
    accs = []
    for s in range(20):
        clf = clf_fn(s)
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))
    arr = np.array(accs)
    print(f"  {name:>30} | {arr.mean():>10.4f} | {arr.std():>10.4f} | "
          f"{arr.min():>10.4f} | {arr.max():>10.4f}")

print(f"""
  Single DT: highest standard deviation — very sensitive to random seed.
  RF B=10:   large variance reduction even with only 10 trees.
  RF B=100:  very stable — the mean and std are tight.
  GB:        even more stable than RF (boosting further reduces variance).
""")


# =============================================================================
# PART 3: HYPERPARAMETER SENSITIVITY
# =============================================================================

print("=" * 70)
print("  PART 3: HYPERPARAMETER SENSITIVITY")
print("=" * 70)

X, y = make_classification(n_samples=500, n_features=20, n_informative=8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"""
  Random Forest is famously robust to hyperparameter choices.
  Effect of n_estimators (B): (other params = sklearn defaults)
""")
print(f"  {'n_estimators':>14} | {'Test Acc':>10} | {'OOB Acc':>10}")
print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
for B in [10, 50, 100, 200, 500]:
    rf = RandomForestClassifier(B, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    print(f"  {B:>14} | {accuracy_score(y_te, rf.predict(X_te)):>10.4f} | {rf.oob_score_:>10.4f}")

print(f"""
  Effect of max_features (feature subsampling):
""")
print(f"  {'max_features':>14} | {'Test Acc':>10} | {'OOB Acc':>10}")
print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
p = X_tr.shape[1]
for mf in [1, 2, int(np.sqrt(p)), p//3, p//2, p]:
    rf = RandomForestClassifier(100, max_features=mf, oob_score=True,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    print(f"  {mf:>14} | {accuracy_score(y_te, rf.predict(X_te)):>10.4f} | {rf.oob_score_:>10.4f}")

print(f"""
  Random Forest is quite robust — performance doesn't collapse with
  "wrong" hyperparameter choices. This is unlike gradient boosting
  or neural networks which require careful tuning.
""")


# =============================================================================
# PART 4: EXTRAPOLATION FAILURE (SHARED WEAKNESS OF ALL TREE METHODS)
# =============================================================================

print("=" * 70)
print("  PART 4: EXTRAPOLATION FAILURE — TREE METHODS vs LINEAR MODELS")
print("=" * 70)

print(f"""
  All tree-based models (DT, RF, GB) predict values within the range of
  training targets. They CANNOT extrapolate beyond what they have seen.

  Example: train on y = 2x for x ∈ [0, 5], predict for x ∈ [6, 10].
""")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Training data: y = 2x for x in [0,5]
X_train_reg = np.linspace(0, 5, 100).reshape(-1, 1)
y_train_reg = 2 * X_train_reg.ravel() + np.random.randn(100) * 0.2

# Test data: EXTRAPOLATION x in [6, 10]
X_test_reg = np.linspace(6, 10, 20).reshape(-1, 1)
y_test_reg  = 2 * X_test_reg.ravel()   # true values

lr_reg  = LinearRegression().fit(X_train_reg, y_train_reg)
rf_reg  = RandomForestRegressor(100, random_state=42).fit(X_train_reg, y_train_reg)

lr_preds  = lr_reg.predict(X_test_reg)
rf_preds  = rf_reg.predict(X_test_reg)

print(f"  True function: y = 2x")
print(f"  Training range: x ∈ [0, 5].    Extrapolation range: x ∈ [6, 10].")
print(f"")
print(f"  {'x':>6} | {'y_true':>8} | {'LR pred':>8} | {'RF pred':>8} | RF error")
print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")
for i in range(0, 20, 4):
    x    = X_test_reg[i, 0]
    yt   = y_test_reg[i]
    lrp  = lr_preds[i]
    rfp  = rf_preds[i]
    err  = abs(rfp - yt)
    print(f"  {x:>6.1f} | {yt:>8.2f} | {lrp:>8.2f} | {rfp:>8.2f} | {err:>8.2f}  {'✗ plateau' if err > 2 else '≈ ok'}")

max_train_y = y_train_reg.max()
print(f"""
  RF predictions plateau at ~{max_train_y:.2f} (max training target).
  For x > 5, the RF predicts values within [0, {max_train_y:.1f}] — the training range.
  Linear Regression correctly extrapolates beyond the training range.

  This is a fundamental limitation of all tree-based models:
    → Use tree models for INTERPOLATION tasks (test distribution ≈ train distribution)
    → Use linear models or neural networks for EXTRAPOLATION tasks
""")


# =============================================================================
# PART 5: TRAINING TIME vs ACCURACY
# =============================================================================

print("=" * 70)
print("  PART 5: TRAINING TIME vs ACCURACY")
print("=" * 70)

X, y = make_classification(1000, 20, n_informative=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"\\n  {'Model':>35} | {'Time (ms)':>10} | {'Test Acc':>10}")
print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}")

timed_models = [
    ("Single DT",               DecisionTreeClassifier(random_state=42)),
    ("RF B=10 (1 job)",         RandomForestClassifier(10, random_state=42, n_jobs=1)),
    ("RF B=100 (1 job)",        RandomForestClassifier(100, random_state=42, n_jobs=1)),
    ("RF B=100 (parallel)",     RandomForestClassifier(100, random_state=42, n_jobs=-1)),
    ("Extra-Trees B=100",       ExtraTreesClassifier(100, random_state=42, n_jobs=-1)),
    ("Gradient Boosting B=100", GradientBoostingClassifier(100, random_state=42)),
    ("Logistic Regression",     LogisticRegression(C=1.0, max_iter=1000)),
]

for name, clf in timed_models:
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    acc = accuracy_score(y_te, clf.predict(X_te))
    ms  = (t1 - t0) * 1000
    print(f"  {name:>35} | {ms:>10.1f} | {acc:>10.4f}")

print(f"""
  Extra-Trees train faster than Random Forests because they skip the
  exhaustive threshold search — random thresholds are drawn directly.
  Gradient Boosting is sequential (no parallelism) → slower than RF.
  RF is embarrassingly parallel — n_jobs=-1 uses all available cores.
""")
''',
    },
}

VISUAL_HTML  = ""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 65)
    print("  RANDOM FORESTS: WISDOM OF A CROWD OF TREES")
    print("=" * 65)
    print("""
  Key concepts demonstrated:
    • Bootstrap sampling → diverse training sets per tree
    • Feature subsampling at each split → decorrelated trees
    • Majority vote / mean averaging → variance reduction
    • OOB error → free generalisation estimate
    • Bias unchanged, variance shrinks as 1/B (approximately)
    • Robust to hyperparameters; excellent out-of-the-box performance
    """)

    np.random.seed(42)

    X, y = make_classification(n_samples=600, n_features=15, n_informative=7,
                                n_redundant=3, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    print("=" * 65)
    print("  SINGLE TREE vs RANDOM FOREST — HEAD TO HEAD")
    print("=" * 65)

    single = DecisionTreeClassifier(random_state=42)
    single.fit(X_tr, y_tr)

    rf = RandomForestClassifier(n_estimators=200, oob_score=True,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    single_train = accuracy_score(y_tr, single.predict(X_tr))
    single_test  = accuracy_score(y_te, single.predict(X_te))
    rf_train     = accuracy_score(y_tr, rf.predict(X_tr))
    rf_test      = accuracy_score(y_te, rf.predict(X_te))

    print(f"\n  {'Metric':>30}   {'Single Tree':>12}   {'Random Forest':>13}")
    print(f"  {'-'*30}   {'-'*12}   {'-'*13}")
    print(f"  {'Train accuracy':>30}   {single_train:>12.4f}   {rf_train:>13.4f}")
    print(f"  {'Test accuracy':>30}   {single_test:>12.4f}   {rf_test:>13.4f}")
    print(f"  {'Generalisation gap':>30}   {single_train-single_test:>12.4f}   {rf_train-rf_test:>13.4f}")
    print(f"  {'OOB accuracy (RF only)':>30}   {'—':>12}   {rf.oob_score_:>13.4f}")
    print(f"  {'Number of trees':>30}   {'1':>12}   {rf.n_estimators:>13}")
    print(f"  {'Number of leaves':>30}   {single.get_n_leaves():>12}   {'many':>13}")

    print(f"\n  Feature importances (top 5 of {X_tr.shape[1]}):")
    print(f"  {'Feature':>12}   {'RF MDI':>10}   {'Single DT MDI':>14}")
    print(f"  {'-'*12}   {'-'*10}   {'-'*14}")
    order = np.argsort(rf.feature_importances_)[::-1]
    for i in order[:5]:
        print(f"  {'feature_'+str(i):>12}   {rf.feature_importances_[i]:>10.4f}   "
              f"{single.feature_importances_[i]:>14.4f}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1.  Bootstrap + feature subsampling → diverse, decorrelated trees
  2.  Averaging B trees: Var(ensemble) = ρ·σ² + (1-ρ)·σ²/B
  3.  Bias of ensemble = bias of single tree (averaging doesn't reduce bias)
  4.  OOB error ≈ 3-fold CV accuracy — a free generalisation estimate
  5.  MDI importance is fast but biased toward high-cardinality features
  6.  Permutation importance is slower but unbiased and more reliable
  7.  More trees always helps (or is neutral) — no overfitting with n_estimators
  8.  Feature subsampling (max_features) is the key hyperparameter — use sqrt(p)
  9.  Extra-Trees: more random thresholds → faster training, similar accuracy
  10. All tree methods fail at extrapolation — plateau at training target range
  11. Next step: Gradient Boosting trains trees sequentially to reduce bias too
    """)


# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has ~20 leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None # _MAIN_SCRIPT

    scripts_available = main_script.exists()

    if "tok_step_status"  not in st.session_state:
        st.session_state.tok_step_status  = {}
    if "tok_step_outputs" not in st.session_state:
        st.session_state.tok_step_outputs = {}

    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**{op_data['description']}**")
            st.markdown("---")
            st.code(op_data["code"], language=op_data.get("language", "python"))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────


def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    try:
        from Supervised_Learning.visuals.random_forest_visual import (   # ← match your exact folder casing
            RF_VISUAL_HTML,
            RF_VISUAL_HEIGHT,
        )
        visual_html   = RF_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = RF_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[03_svm.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }