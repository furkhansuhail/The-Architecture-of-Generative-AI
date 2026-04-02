"""
Decision Trees — Learning by Asking Questions
==============================================

A decision tree learns a sequence of yes/no questions about the input features
and uses the answers to arrive at a prediction. It is the only model in this
series that makes no linearity assumption whatsoever — it can represent any
decision boundary composed of axis-aligned cuts, and it does so through a
process that mirrors how humans actually reason about classification problems.

Decision trees are also the building block for Random Forests and Gradient
Boosted Trees — two of the most powerful and widely deployed algorithms in
all of applied machine learning.

"""

import base64
import os
import textwrap
import re

TOPIC_NAME = "Decision Trees"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_NAME = "04 · Decision Trees"
ICON         = "⚔️"
SUBTITLE     = "Decision Tree Breakdown and Explanation"
THEORY = """

### What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that learns a hierarchy of
if/else questions about input features and uses the answers to make predictions.
Unlike every model studied so far — linear regression, logistic regression, SVM —
a decision tree makes no linearity assumption. Its decision boundaries are composed
entirely of axis-aligned splits, which can approximate any shape given sufficient
depth.

The analogy is a game of 20 Questions. Each internal node of the tree asks one
question about one feature ("Is age > 40?", "Is income > 50k?"). The answer routes
the example left or right. The leaf at the bottom gives the prediction — either a
class label (classification) or a numeric value (regression).

What makes decision trees remarkable is their interpretability. Unlike a neural
network or SVM, you can read out the learned rules in plain English. A doctor can
follow the chain of questions and understand exactly why a diagnosis was made.

    Things that exist inside the model (learned during training):
        - The tree structure          — which feature to split at each node
        - The split thresholds        — the cutoff value for each split
        - The leaf predictions        — class label or mean value at each leaf

    Things you control before training (hyperparameters):
        - max_depth                   — maximum depth of the tree
        - min_samples_split           — minimum examples required to split a node
        - min_samples_leaf            — minimum examples required in a leaf
        - criterion                   — impurity measure: "gini" or "entropy"


### Decision Trees as Empirical Risk Minimisation (ERM)

Decision trees fit the ERM framework, though with an important caveat:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Hypothesis class:  H = { axis-aligned recursive partitions │
    │                           of the feature space }            │
    │                     (piecewise-constant functions)          │
    │                                                             │
    │  Loss function:     Gini impurity or Information Gain       │
    │                     (local, greedy proxy for 0-1 loss)      │
    │                                                             │
    │  Training objective:                                        │
    │      min_tree  Σ_leaves  impurity(leaf) × |leaf| / n        │
    │                                                             │
    │  Optimiser:  Greedy recursive splitting (CART algorithm)    │
    │              NOT global optimisation — finding the globally │
    │              optimal tree is NP-hard                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The critical difference from all previous models: tree learning is GREEDY.
At each node we find the locally best split without backtracking. This means
we are not guaranteed to find the globally optimal tree — but the greedy
solution is fast and works well in practice.

Comparing ERM formulations across the series:
    Linear Regression:    MSE loss,       gradient descent (global optimum)
    Logistic Regression:  BCE loss,       gradient descent (global optimum)
    SVM:                  Hinge loss,     quadratic programming (global optimum)
    Decision Tree:        Gini / Entropy, greedy splitting (local, not global)


### The Inductive Bias of Decision Trees

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Decision trees encode the belief that:                     │
    │                                                             │
    │  1. AXIS-ALIGNED SPLITS — the world can be described by     │
    │     thresholds on individual features, one at a time.       │
    │     Interactions between features are captured by           │
    │     sequential splits, not a single linear combination.     │
    │                                                             │
    │  2. LOCAL STRUCTURE — the output function is piecewise      │
    │     constant: each region of input space has its own        │
    │     prediction, independent of all other regions.           │
    │                                                             │
    │  3. FEATURE RELEVANCE ORDERING — the most informative       │
    │     feature should be asked first (root), less informative  │
    │     features appear deeper or not at all.                   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

This contrasts sharply with the linear models:
    Linear/Logistic/SVM:  f(x) = w·x + b — features contribute ADDITIVELY
    Decision Tree:        f(x) = look up the region x falls into — features
                          contribute CONDITIONALLY (each split depends on
                          which branch was taken above it)


---


### Part 1: The Splitting Criterion — How to Choose the Best Question


The central challenge in training a decision tree is: at each node, which feature
and which threshold produces the most informative split? We measure this using
an impurity function.


### What is Impurity?

Impurity quantifies how mixed the class labels are in a set of examples. A perfectly
pure node contains only examples of one class (impurity = 0). A maximally impure node
has equal numbers of every class (impurity is maximum).

A good split takes an impure parent node and produces children that are more pure
than the parent. The reduction in impurity is the "gain" from that split.


### Gini Impurity:

For a node containing examples from k classes with proportions p₁, p₂, ..., pₖ:

                    Gini = 1 − Σⱼ pⱼ²

Intuition: if you randomly pick two examples from the node, Gini measures the
probability that they belong to different classes. Zero means all examples are the
same class (pure). Maximum is (1 − 1/k) when all classes are equally represented.

For binary classification (k=2, proportions p and 1−p):

                    Gini = 1 − p² − (1−p)² = 2p(1−p)

This is maximised at p = 0.5 (equal mix) and is 0 at p = 0 or p = 1 (pure).


### Entropy (Information Gain):

From information theory (Shannon, 1948), entropy measures the average number of bits
needed to encode a randomly drawn label from the node:

                    H = − Σⱼ pⱼ log₂(pⱼ)       (convention: 0 · log 0 = 0)

For binary classification:

                    H = −p log₂(p) − (1−p) log₂(1−p)

Entropy is maximised at p = 0.5 (H = 1 bit) and is 0 at p = 0 or p = 1 (pure).
Information Gain is the reduction in entropy from a split:

                    IG = H(parent) − Σ_children  |child|/|parent| · H(child)


```

**Diagram 1 — Gini vs Entropy as Functions of Class Proportion p:**

IMPURITY vs p (fraction of class 1), binary case
══════════════════════════════════════════════════════════════


  Impurity
  ↑
  1.0 │                    ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪                    ◄── Entropy (bits, scaled to 1)
  0.9 │               ▪▪▪▪▪               ▪▪▪▪▪
  0.8 │            ▪▪▪                         ▪▪▪
  0.7 │         ▪▪▪                               ▪▪▪
  0.6 │       ▪▪                                     ▪▪
  0.5 │      ▪            ·················            ▪       ◄── Gini = 2p(1−p)
  0.4 │    ▪▪       ······                 ······       ▪▪
  0.3 │   ▪    ·····                             ·····    ▪
  0.2 │  ▪  ···                                       ···  ▪
  0.1 │ ▪···                                             ···▪
  0.0 │█·                                                   ·█
  0.0 └───────────────────────────────────────────────────────→ p
       0         0.2        0.4  0.5  0.6        0.8         1.0

  ▪ = Entropy curve    · = Gini curve    █ = overlap (both = 0 at endpoints)


Key values (binary classification):
────────────────────────────────────────────────────────
p     Gini = 2p(1-p)   Entropy = -p·log₂p - (1-p)·log₂(1-p)
────────────────────────────────────────────────────────
0.0   0.000            0.000   (pure: only class 0)
0.1   0.180            0.469
0.2   0.320            0.722
0.3   0.420            0.881
0.5   0.500            1.000   (maximally impure)
0.7   0.420            0.881
0.9   0.180            0.469
1.0   0.000            0.000   (pure: only class 1)
────────────────────────────────────────────────────────

Both peak at p=0.5 and vanish at p=0 or p=1.
Entropy is slightly more sensitive to near-pure nodes.
In practice, both produce very similar trees.

```


### Finding the Best Split:

For each candidate split (feature f, threshold t), we compute the weighted impurity
of the resulting children:

    Impurity_split = (|left|/|parent|) · Impurity(left)
                   + (|right|/|parent|) · Impurity(right)

    Gain = Impurity(parent) − Impurity_split

The best split is the one with the largest Gain (equivalently, the smallest
weighted child impurity).

For a dataset with n examples and p features, finding the best split requires
checking O(n·p) candidates (one threshold per unique feature value, per feature).


### Concrete Calculation:

Let's manually compute the best split for a small dataset.

    Dataset (n=10, predicting if customer buys):
    ─────────────────────────────────────────────────────────────
    Age    Income    Buys?
    ─────────────────────────────────────────────────────────────
    25     Low       No
    30     High      Yes
    35     High      Yes
    40     Low       No
    45     High      Yes
    50     High      Yes
    55     Low       No
    60     High      Yes
    65     Low       No
    70     High      Yes
    ─────────────────────────────────────────────────────────────

    Parent node: 6 Yes, 4 No  →  p = 0.6
    Gini(parent) = 1 − 0.6² − 0.4² = 1 − 0.36 − 0.16 = 0.48

    Candidate Split 1: Age ≤ 42
        Left  (Age ≤ 42): [No, Yes, Yes, No, Yes]  → 3 Yes, 2 No  → p=0.6
            Gini(left)  = 2(0.6)(0.4) = 0.48
        Right (Age > 42): [Yes, Yes, No, Yes, No, Yes] → 4 Yes, 2 No → p=0.67
            Gini(right) = 2(0.67)(0.33) = 0.44
        Weighted Gini = (5/10)(0.48) + (5/10)(0.44) = 0.46
        Gain = 0.48 − 0.46 = 0.02   ← small gain

    Candidate Split 2: Income = High
        Left  (Income = High): [Yes, Yes, Yes, Yes, Yes, Yes]  → 6 Yes, 0 No → p=1.0
            Gini(left)  = 0.0  (PURE!)
        Right (Income = Low):  [No, No, No, No]                → 0 Yes, 4 No → p=0.0
            Gini(right) = 0.0  (PURE!)
        Weighted Gini = 0
        Gain = 0.48 − 0.00 = 0.48   ← maximum possible gain!

    Decision: Split on Income = High  (perfectly separates the classes)

---

### Part 2: The CART Algorithm — Growing the Tree


### Recursive Binary Splitting:

The Classification and Regression Trees (CART) algorithm grows a tree by:

    1. Starting with all training examples at the root
    2. Finding the best split (feature + threshold maximising the gain)
    3. Partitioning examples into left and right child nodes
    4. Recursively repeating steps 2–3 on each child
    5. Stopping when a stopping criterion is met (node is pure, depth limit
       reached, or too few examples to split)

The result is a binary tree where each internal node tests one feature against
one threshold, and each leaf holds a prediction.


**Diagram 2 — The CART Growing Process:**

```
STEP 0: All data at root (mixed classes ●○)        STEP 1: Best split found
┌────────────────────────┐                        ┌────────────────────────┐
│  ● ○ ● ● ○ ○ ● ○ ● ●   │                        │    Feature 1 ≤ 3.0?    │
│  Root: Gini = 0.48     │  ──────────────────►   └──────────┬─────────────┘
└────────────────────────┘                                  ╱│╲
                                                     YES  ╱     ╲  NO
                                                        ╱         ╲
                                                ┌──────┐           ┌──────┐
                                                │ ●●●● │           │ ○○○○ │
                                                │Pure! │           │Pure! │
                                                └──────┘           └──────┘
                                                Predict ●          Predict ○


GROWING STOPS WHEN ANY OF THESE IS TRUE:
─────────────────────────────────────────────────────────────────────────
Condition                    Effect
─────────────────────────────────────────────────────────────────────────
Node is pure (Gini = 0)      Leaf: predict the single remaining class
max_depth reached            Leaf: predict majority class
Fewer examples than          Leaf: can't split further
  min_samples_split
Best split gives zero gain   Leaf: no useful information in any split
─────────────────────────────────────────────────────────────────────────
```


### Making Predictions:

Once the tree is built, predictions are made by routing a new example from the root
downward, following the branch that matches each split condition, until reaching a leaf.

    For classification: predict the majority class in the leaf
    For regression: predict the mean target value in the leaf

Runtime complexity: O(depth) per prediction — extremely fast at inference time,
regardless of how large the training set was.


### Regression Trees:

Decision trees handle regression (continuous outputs) by replacing the impurity
criterion with variance reduction:

                Variance(node) = (1/n) Σ (yᵢ − ȳ)²

The best split is the one that maximises the reduction in total weighted variance.
Each leaf predicts the mean y of the training examples that fall there.


---


### Part 3: Overfitting and Pruning


### The Bias-Variance Tradeoff for Trees:

An unconstrained decision tree will grow until every leaf contains exactly one
training example — perfectly classifying all training data (training error = 0) but
generalising poorly to new data (high variance).

    Shallow tree (depth 1–3):
        High bias  — may not capture the true pattern
        Low variance — predictions stable across different training sets

    Deep tree (depth = n):
        Low bias  — memorises training data perfectly
        High variance — tiny changes in training data → very different trees

**Diagram 3 — Depth vs Training/Test Error:**

```
Error
↑
│╲ Training error
│  ╲
│    ╲____
│         ‾‾‾─────────────────────── → 0 (perfect training fit)
│
│         ___
│       ╱     ╲
│     ╱         ╲_____
│   ╱                  ‾‾‾──────────  Test error (U-curve)
│                       ↑
│                 optimal depth
└──────────────────────────────────→ Tree depth

Training error monotonically decreases as depth grows.
Test error is U-shaped — there is a sweet spot.
```


### Pre-Pruning (Early Stopping):

Stop growing the tree early by imposing constraints:

    max_depth:          Maximum depth from root to any leaf.
                        Simplest and most interpretable control.

    min_samples_split:  A node must have at least this many examples to be split.
                        Prevents splitting tiny subsets (likely noise).

    min_samples_leaf:   Each leaf must contain at least this many examples.
                        Ensures each prediction is based on enough data.

    min_impurity_decrease: Only split if the gain exceeds this threshold.
                        Ignores splits that contribute almost nothing.


### Post-Pruning (Cost-Complexity Pruning):

Grow the full tree first, then remove subtrees that don't sufficiently reduce error.

The cost-complexity criterion adds a penalty for tree size:

                    Rα(T) = R(T) + α · |T|

Where R(T) is the training error of tree T, |T| is the number of leaves, and α is
a penalty per leaf (higher α → simpler tree).

For each α value, there is a unique optimal pruned tree. As α increases from 0 to ∞,
the sequence of optimal trees goes from the full tree down to just the root.
The optimal α is found by cross-validation.

sklearn implements this as `ccp_alpha` (cost-complexity pruning alpha).


**Diagram 4 — Pruning: Removing a Subtree:**

```
BEFORE PRUNING:                        AFTER PRUNING:
(two leaves with near-identical        (subtree replaced by single leaf)
 class distributions)

     [Feature A ≤ 5?]                      [Feature A ≤ 5?]
      ╱           ╲                          ╱           ╲
[Feat B ≤ 2?]    [Predict ○]          [Feat B ≤ 2?]    [Predict ○]
  ╱       ╲                    ──►      ╱       ╲
[●: 8, ○: 2] [●: 7, ○: 3]           [Predict ●]    ← merged leaf
(80% ●)      (70% ●)

The subtree [Feat B ≤ 2?] splits into two leaves that both predict ●.
This split is contributing almost nothing — prune it away.
```

---

### Part 4: Feature Importance


### Mean Decrease in Impurity (MDI):

The most common importance measure: for each feature, sum the impurity reduction
it causes across all splits in the tree, weighted by the number of examples passing
through each split:

    Importance(feature f) = Σ_{nodes where f is used}  (|node|/n) · Gain(node)

Features that produce large gains early in the tree (closer to the root, affecting
more examples) receive higher importance scores. Importances are normalised to sum to 1.

**Caveat:** MDI is biased toward high-cardinality features (features with many unique
values have more thresholds to try, so more chances to get a high gain by chance).
For unbiased importance, use permutation importance (sklearn's permutation_importance).


### What Feature Importance Tells You — and What It Doesn't:

    Feature importance DOES tell you:
        Which features the tree uses most for splitting
        Relative influence of features on the training data

    Feature importance DOES NOT tell you:
        Direction of the relationship (does feature X increase or decrease prediction?)
        Whether the feature is causally important
        Whether the importance is stable (try permutation importance for robustness)


**Diagram 5 — Feature Importance: High vs Low:**

```
HIGH IMPORTANCE (feature used at root and high nodes):

depth 0:     [Feature A ≤ 5?]          ← splits 100% of data, large gain
              ╱             ╲
depth 1: [Feat B ≤ 2?]   [Feat C ≤ 7?] ← splits 50% each, moderate gain
          ╱       ╲         ╱     ╲
depth 2: ... leaves ...  ... leaves ...

Feature A importance ≈ (100/100) × 0.4 = 0.40
Feature B importance ≈ (50/100)  × 0.2 = 0.10
Feature C importance ≈ (50/100)  × 0.2 = 0.10

A low-depth feature affecting many examples gets much higher importance
than an equally discriminative feature used only in small leaf subtrees.
```

---

### Part 5: Decision Tree for Regression


Decision tree regression works identically to classification but uses:
    Splitting criterion: Variance reduction instead of Gini/Entropy
    Leaf prediction:    Mean of target values in that leaf (not majority class)

The resulting model is a step function — piecewise constant over input space.

                    Variance(node) = (1/n) Σ (yᵢ − ȳ)²

Best split = the (feature, threshold) pair that maximises:

    Variance_reduction = Variance(parent) − (|left|/n) · Var(left)
                                          − (|right|/n) · Var(right)


**Diagram 6 — Regression Tree: Piecewise Constant Approximation:**

```
TRUE FUNCTION: y = sin(x) + noise        REGRESSION TREE APPROXIMATION:

y ↑                                       y ↑
  │        •  •  •                          │     ┌──────┐
  │      •        •                         │     │      │
  │    •            •                       │─────┘      └───────────
  │  •                •                     │                      └──
  │ •                  •  •  •              │
  └──────────────────────────→ x            └──────────────────────────→ x
                                              ↑  ↑  ↑
                                              split thresholds

Each flat segment is one leaf. Deeper tree → more segments → better approximation.
```


---


### Part 6: Advantages, Limitations, and the Path to Ensembles


### Advantages of Decision Trees:

1. Interpretability — you can literally read the learned rules in English.
   No other model we've studied matches this. A doctor can audit every decision.

2. No feature scaling required — splits are threshold comparisons, not dot products.
   Standardising or normalising features changes nothing.

3. Handles mixed feature types natively — categorical and numerical features can
   sit side by side with no preprocessing.

4. Naturally handles non-linear interactions — a split at depth 2 already represents
   a feature interaction (the split condition depends on which branch you took above).

5. Fast inference — O(depth) predictions, even for millions of features.


### Limitations of Decision Trees:

1. High variance (instability) — small changes in training data can produce
   completely different trees. This is the single biggest weakness.

2. Greedy splits are not globally optimal — a locally suboptimal split early in the
   tree may prevent better splits later. No backtracking.

3. Axis-aligned boundaries — smooth diagonal decision boundaries require many splits.
   A linear boundary y = x₁ + x₂ that logistic regression fits perfectly in one
   step requires many axis-aligned steps in a decision tree.

4. Bias toward high-cardinality features — MDI importance is inflated for features
   with many unique values.

5. Poor extrapolation — regression trees can only predict values within the range of
   training targets. They cannot extrapolate beyond what they've seen.


### The Bridge to Ensembles:

The high variance of decision trees motivated the two most important ensemble methods
in applied machine learning:

    Random Forests (Breiman, 2001):
        Train many trees, each on a random subset of features and data.
        Average their predictions (classification: majority vote).
        Averaging reduces variance while keeping low bias.
        The randomness decorrelates the trees — each tree makes different errors.

    Gradient Boosted Trees (Friedman, 2001):
        Train trees sequentially, each correcting the errors of the previous.
        Each tree is a shallow learner (weak learner).
        Boosting reduces bias while controlling variance via learning rate.
        XGBoost, LightGBM, CatBoost are all this idea at scale.

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Single Tree: High variance, interpretable, fast to train   │
    │                                                             │
    │  Random Forest: Lower variance, less interpretable,         │
    │  training is parallel (trees are independent)               │
    │                                                             │
    │  Gradient Boosting: Lowest bias+variance, least interpret., │
    │  training is sequential (each tree needs previous results)  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


### Connecting to the Full Learning Progression:

    Perceptron (1958)              ← hard linear boundary, no gradient
          │
          ↓
    Logistic Regression            ← soft linear boundary, probabilistic
          │
          ↓
    SVM                            ← maximum-margin linear/kernel boundary
          │
          ↓
    Decision Tree (1984, CART)     ← non-linear, axis-aligned, no gradient needed
          │
          │  (many trees + randomness)
          ↓
    Random Forest (2001)           ← ensemble, reduced variance
          │
          │  (many trees + sequential boosting)
          ↓
    Gradient Boosted Trees (2001)  ← state-of-the-art for tabular data
          │
          │  (replace axis-aligned splits with learned representations)
          ↓
    Neural Networks                ← learn their own feature representation

    """

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ──────────────────────────────────────────────────────────────────────────
    Operation                   Complexity          Notes
    ──────────────────────────────────────────────────────────────────────────
    Training (best split search) O(n·p·log n)       n=samples, p=features
    Prediction                   O(depth)            extremely fast
    Memory                       O(2^depth)          nodes stored
    Feature importance (MDI)     O(n·p·log n)        computed during training
    Post-pruning (ccp_alpha)     O(n·p·log n · k)    k = pruning candidates
    ──────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Decision Tree from Scratch": {
        "description": "CART algorithm built from first principles — Gini, splitting, and prediction",
        "runnable": True,
        "code": '''
"""
================================================================================
DECISION TREE FROM SCRATCH — CART ALGORITHM
================================================================================

We implement the full CART (Classification and Regression Trees) algorithm:
    1. Compute Gini impurity for any node
    2. Search all features and thresholds for the best split
    3. Recursively build the tree
    4. Predict by routing examples from root to leaf

No sklearn, no black boxes. Every step is explained.

================================================================================
"""

import numpy as np
from collections import Counter


# =============================================================================
# IMPURITY FUNCTIONS
# =============================================================================

def gini_impurity(y):
    """
    Gini impurity = 1 − Σⱼ pⱼ²

    Measures how mixed the class labels are in y.
        0.0  = perfectly pure (all same class)
        ~0.5 = maximally impure (equal mix of 2 classes)

    Args:
        y: array of class labels

    Returns:
        float in [0, 0.5] for binary, [0, 1-1/k] for k classes
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs  = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def entropy(y):
    """
    Shannon entropy = − Σⱼ pⱼ log₂(pⱼ)

    Alternative impurity measure — slightly more sensitive to near-pure nodes.
    Produces very similar trees to Gini in practice.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs  = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))


def weighted_impurity(y_left, y_right, criterion="gini"):
    """
    Weighted average impurity of two child nodes.

    weighted = (|left| / |total|) · impurity(left)
             + (|right| / |total|) · impurity(right)
    """
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    fn = gini_impurity if criterion == "gini" else entropy
    return (len(y_left)/n) * fn(y_left) + (len(y_right)/n) * fn(y_right)


# =============================================================================
# FINDING THE BEST SPLIT
# =============================================================================

def find_best_split(X, y, criterion="gini"):
    """
    Search all features and thresholds for the split that maximises gain.

    For each feature:
        For each unique threshold value:
            Split examples into left (≤ threshold) and right (> threshold)
            Compute weighted impurity of the split
            Track the split with minimum weighted impurity

    Returns:
        best_feature  (int)   : index of the best feature to split on
        best_threshold (float): threshold value for the split
        best_gain     (float) : impurity reduction from the best split
    """
    fn           = gini_impurity if criterion == "gini" else entropy
    parent_impurity = fn(y)

    best_feature   = None
    best_threshold = None
    best_gain      = -np.inf

    n, p = X.shape

    for feature_idx in range(p):
        # Get all unique values of this feature as candidate thresholds
        # (threshold between consecutive unique values)
        values     = np.unique(X[:, feature_idx])
        thresholds = (values[:-1] + values[1:]) / 2   # midpoints

        for threshold in thresholds:
            # Split examples
            left_mask  = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            y_left  = y[left_mask]
            y_right = y[right_mask]

            # Skip degenerate splits (all examples go one way)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Compute gain
            w_imp = weighted_impurity(y_left, y_right, criterion)
            gain  = parent_impurity - w_imp

            if gain > best_gain:
                best_gain      = gain
                best_feature   = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


# =============================================================================
# TREE NODE
# =============================================================================

class TreeNode:
    """
    A single node in the decision tree.

    Internal node: has a split condition and two children.
    Leaf node:     has a prediction (majority class).
    """
    def __init__(self):
        # Split information (internal nodes)
        self.feature_idx  = None   # which feature to split on
        self.threshold    = None   # split threshold: left ≤ threshold, right > threshold
        self.left         = None   # left child TreeNode
        self.right        = None   # right child TreeNode

        # Leaf information
        self.is_leaf      = False
        self.prediction   = None   # majority class (classification)
        self.class_counts = None   # {class: count} at this node

        # Diagnostics
        self.depth        = 0
        self.n_samples    = 0
        self.impurity     = 0.0


# =============================================================================
# DECISION TREE CLASSIFIER
# =============================================================================

class DecisionTreeClassifier:
    """
    Binary and multi-class decision tree using the CART algorithm.

    Parameters (hyperparameters — not learned):
        max_depth (int):         maximum tree depth (None = no limit)
        min_samples_split (int): minimum examples needed to attempt a split
        min_samples_leaf (int):  minimum examples required in each leaf
        criterion (str):         "gini" or "entropy"
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini"):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.criterion         = criterion
        self.root              = None
        self.n_features_       = None
        self.classes_          = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """Build the decision tree from training data."""
        self.n_features_ = X.shape[1]
        self.classes_    = np.unique(y)
        self._importance = np.zeros(self.n_features_)

        self.root = self._grow(X, y, depth=0)

        # Normalise feature importances to sum to 1
        total = self._importance.sum()
        self.feature_importances_ = (self._importance / total
                                     if total > 0 else self._importance)
        return self

    def _grow(self, X, y, depth):
        """Recursively grow the tree."""
        node          = TreeNode()
        node.depth    = depth
        node.n_samples = len(y)

        fn = gini_impurity if self.criterion == "gini" else entropy
        node.impurity = fn(y)

        # ── Leaf conditions ──────────────────────────────────────────────────
        is_pure       = (len(np.unique(y)) == 1)
        too_small     = (len(y) < self.min_samples_split)
        depth_limit   = (self.max_depth is not None and depth >= self.max_depth)

        if is_pure or too_small or depth_limit:
            node.is_leaf      = True
            node.prediction   = Counter(y).most_common(1)[0][0]
            node.class_counts = dict(Counter(y))
            return node

        # ── Find best split ───────────────────────────────────────────────────
        feat, thresh, gain = find_best_split(X, y, self.criterion)

        if feat is None or gain <= 0:
            node.is_leaf      = True
            node.prediction   = Counter(y).most_common(1)[0][0]
            node.class_counts = dict(Counter(y))
            return node

        # ── Check min_samples_leaf ────────────────────────────────────────────
        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.is_leaf      = True
            node.prediction   = Counter(y).most_common(1)[0][0]
            node.class_counts = dict(Counter(y))
            return node

        # ── Record feature importance ─────────────────────────────────────────
        # Weight gain by the fraction of examples passing through this node
        self._importance[feat] += (len(y) / self._importance.shape[0]) * gain

        # ── Store split and recurse ───────────────────────────────────────────
        node.feature_idx  = feat
        node.threshold    = thresh
        node.class_counts = dict(Counter(y))

        node.left  = self._grow(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)

        return node

    def predict(self, X):
        """Predict class labels for all examples in X."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        """Route one example from root to leaf."""
        if node.is_leaf:
            return node.prediction
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def print_tree(self, node=None, indent="", feature_names=None):
        """Print a human-readable representation of the tree."""
        if node is None:
            node = self.root
        if node.is_leaf:
            print(f"{indent}Leaf → Predict: {node.prediction}  "
                  f"(counts: {node.class_counts}, n={node.n_samples})")
            return
        feat_label = (feature_names[node.feature_idx] if feature_names
                      else f"Feature[{node.feature_idx}]")
        print(f"{indent}{feat_label} ≤ {node.threshold:.4f}  "
              f"(Gini={node.impurity:.4f}, n={node.n_samples})")
        print(f"{indent}├─ YES:")
        self.print_tree(node.left,  indent + "│  ", feature_names)
        print(f"{indent}└─ NO:")
        self.print_tree(node.right, indent + "   ", feature_names)


# =============================================================================
# DEMO 1: XOR — Non-Linear Problem (Linear Models Fail!)
# =============================================================================

print("=" * 65)
print("  DECISION TREE FROM SCRATCH — CART ALGORITHM")
print("=" * 65)

np.random.seed(42)

# XOR: class = (x₁ > 0) XOR (x₂ > 0)
n   = 200
X   = np.random.randn(n, 2)
y   = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

print(f"\\n  DATASET: XOR Classification ({n} points)")
print(f"  Class 1: x₁>0 XOR x₂>0   Class 0: same sign")
print(f"  Class distribution: {dict(Counter(y))}")

tree = DecisionTreeClassifier(max_depth=4, criterion="gini")
tree.fit(X, y)
acc = (tree.predict(X) == y).mean()

print(f"\\n  Tree (max_depth=4):")
tree.print_tree(feature_names=["x₁", "x₂"])

print(f"\\n  Training accuracy: {acc:.4f}")
print(f"  (Linear regression/logistic on XOR: ~50%  — decision tree wins!)")


# =============================================================================
# DEMO 2: MANUAL SPLIT TRACE
# =============================================================================

print("\\n" + "=" * 65)
print("  STEP-BY-STEP: BEST SPLIT SEARCH")
print("=" * 65)

# Tiny dataset matching the theory walkthrough
X_tiny = np.array([[25, 0],   # Low income (0)
                    [30, 1],   # High income (1)
                    [35, 1],
                    [40, 0],
                    [45, 1],
                    [50, 1],
                    [55, 0],
                    [60, 1],
                    [65, 0],
                    [70, 1]], dtype=float)
y_tiny = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])

print(f"""
  Dataset (n=10):
    Feature 0: Age  [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    Feature 1: Income  [Low=0, High=1, ...]
    Labels (Buys?):    {y_tiny.tolist()}

  Parent Gini: {gini_impurity(y_tiny):.4f}
""")

feat, thresh, gain = find_best_split(X_tiny, y_tiny, "gini")
print(f"  Best split found:")
print(f"    Feature:   {'Age' if feat == 0 else 'Income'} (index {feat})")
print(f"    Threshold: {thresh}")
print(f"    Gain:      {gain:.4f}")

left_mask  = X_tiny[:, feat] <= thresh
y_left     = y_tiny[left_mask]
y_right    = y_tiny[~left_mask]
print(f"    Left  node (≤ {thresh}): y={y_left.tolist()} → Gini={gini_impurity(y_left):.4f}")
print(f"    Right node (> {thresh}): y={y_right.tolist()} → Gini={gini_impurity(y_right):.4f}")


# =============================================================================
# DEMO 3: FEATURE IMPORTANCE
# =============================================================================

print("\\n" + "=" * 65)
print("  FEATURE IMPORTANCE")
print("=" * 65)

from sklearn.datasets import load_iris
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
fn = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

tree_iris = DecisionTreeClassifier(max_depth=4, criterion="gini")
tree_iris.fit(X_iris, y_iris)
acc_iris = (tree_iris.predict(X_iris) == y_iris).mean()

print(f"\\n  Iris dataset (3 classes, 4 features, n=150)")
print(f"  Training accuracy: {acc_iris:.4f}")
print(f"\\n  Feature importances (MDI — Mean Decrease in Impurity):")
for name, imp in sorted(zip(fn, tree_iris.feature_importances_),
                         key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"    {name:>12}  {imp:.4f}  {bar}")

print(f"\\n  Top feature: petal length — it is the most discriminative feature")
print(f"  for separating Iris species, used at or near the root.")
''',
    },

    "Impurity Functions Deep Dive": {
        "description": "Gini vs Entropy — derivations, numerical comparison, and split calculations",
        "runnable": True,
        "code": '''
"""
================================================================================
IMPURITY FUNCTIONS — GINI vs ENTROPY
================================================================================

Two questions this script answers:
    1. What do Gini and Entropy actually measure?
    2. Do they produce different trees?

================================================================================
"""

import numpy as np
import math


# =============================================================================
# PART 1: ANALYTICAL COMPARISON
# =============================================================================

print("=" * 65)
print("  PART 1: GINI vs ENTROPY — NUMERICAL COMPARISON")
print("=" * 65)

def gini(p):
    """Binary Gini: 1 - p² - (1-p)² = 2p(1-p)"""
    return 2 * p * (1 - p)

def entropy_binary(p):
    """Binary entropy: -p·log₂(p) - (1-p)·log₂(1-p)"""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1-p) * math.log2(1-p)

print(f"""
  For binary classification, both are functions of p = P(class 1):

    Gini(p)   = 1 − p² − (1−p)² = 2p(1−p)
    Entropy(p) = −p·log₂(p) − (1−p)·log₂(1−p)

  {'p':>6} | {'Gini':>8} | {'Entropy':>10} | {'Gini/Entropy':>14}
  {'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*14}""")

for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    g  = gini(p)
    h  = entropy_binary(p)
    ratio = f"{g/h:.4f}" if h > 0 else "  N/A  "
    print(f"  {p:>6.1f} | {g:>8.4f} | {h:>10.4f} | {ratio:>14}")

print(f"""
  Key observations:
    • Both peak at p=0.5 (maximally impure) and are zero at p=0 or p=1
    • Entropy is always ≥ Gini (it's larger in magnitude)
    • Their ratio is roughly constant (~0.5) → they produce very similar splits
    • Gini is computationally cheaper (no logarithm)
    • Entropy is slightly more "sensitive" to near-pure nodes (penalises more)

  In practice: use Gini for speed, entropy if you want slightly more balanced trees.
""")


# =============================================================================
# PART 2: INFORMATION GAIN CALCULATION
# =============================================================================

print("=" * 65)
print("  PART 2: INFORMATION GAIN — WORKED EXAMPLES")
print("=" * 65)

def information_gain(y_parent, y_left, y_right):
    """
    IG = H(parent) - weighted_entropy(children)
    """
    n  = len(y_parent)
    nl = len(y_left)
    nr = len(y_right)

    def h(y):
        if len(y) == 0: return 0.0
        counts = np.bincount(y, minlength=2)
        probs  = counts / len(y)
        probs  = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    return h(y_parent) - (nl/n) * h(y_left) - (nr/n) * h(y_right)

print("""
  Dataset: Predict if customer buys (1=Yes, 0=No)
  ─────────────────────────────────────────────────────
  Feature: Weather   (Sunny, Overcast, Rainy)
  Labels:            [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
  n=14 examples, 9 buys (1), 5 no-buys (0)
""")

y_all = np.array([1,0,1,1,0,1,0,0,1,1,0,1,1,0])
p_pos = y_all.mean()
H_parent = entropy_binary(p_pos)
print(f"  Parent entropy H(root) = -{p_pos:.3f}·log₂({p_pos:.3f}) - {1-p_pos:.3f}·log₂({1-p_pos:.3f})")
print(f"                        = {H_parent:.4f} bits")

# Candidate split: Sunny (indices 0-4), Overcast (5-9), Rainy (10-13)
y_sunny    = np.array([1, 0, 0, 0, 1])   # 2 yes, 3 no
y_overcast = np.array([1, 1, 1, 1])      # 4 yes, 0 no
y_rainy    = np.array([0, 1, 1, 0, 0])   # 2 yes, 3 no

print(f"""
  Split on Weather:
    Sunny    (n=5): {y_sunny.tolist()}    → {y_sunny.sum()} yes, {(~y_sunny.astype(bool)).sum()} no
    Overcast (n=4): {y_overcast.tolist()}       → {y_overcast.sum()} yes, 0 no  ← PURE!
    Rainy    (n=5): {y_rainy.tolist()}    → {y_rainy.sum()} yes, {(~y_rainy.astype(bool)).sum()} no
""")

H_sunny    = entropy_binary(y_sunny.mean())
H_overcast = entropy_binary(y_overcast.mean())
H_rainy    = entropy_binary(y_rainy.mean())
n = len(y_all)
weighted_H = (5/n)*H_sunny + (4/n)*H_overcast + (5/n)*H_rainy
IG_weather = H_parent - weighted_H

print(f"  H(Sunny)    = {H_sunny:.4f}")
print(f"  H(Overcast) = {H_overcast:.4f}  (pure!)")
print(f"  H(Rainy)    = {H_rainy:.4f}")
print(f"")
print(f"  Weighted H = (5/14)×{H_sunny:.4f} + (4/14)×{H_overcast:.4f} + (5/14)×{H_rainy:.4f}")
print(f"             = {weighted_H:.4f}")
print(f"")
print(f"  IG(Weather) = {H_parent:.4f} - {weighted_H:.4f} = {IG_weather:.4f} bits")

# Compare with a less informative split
y_left2  = y_all[:7]   # first 7 examples
y_right2 = y_all[7:]   # last 7 examples
IG_random = information_gain(y_all, y_left2, y_right2)
print(f"  IG(random 50/50 split) = {IG_random:.4f} bits  ← much worse")
print(f"""
  The Weather split gains {IG_weather:.4f} bits — good because Overcast is pure.
  The random 50/50 split gains only {IG_random:.4f} bits.
  CART picks the split with the HIGHEST information gain.
""")


# =============================================================================
# PART 3: MULTICLASS GINI
# =============================================================================

print("=" * 65)
print("  PART 3: MULTI-CLASS GINI")
print("=" * 65)

def gini_multiclass(y, n_classes=None):
    """Gini for k classes: 1 - Σ pⱼ²"""
    if len(y) == 0:
        return 0.0
    if n_classes is None:
        n_classes = len(np.unique(y))
    counts = np.bincount(y, minlength=n_classes)
    probs  = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

print(f"""
  Gini impurity for k=3 classes (Iris example):
  Formula: Gini = 1 − p₀² − p₁² − p₂²

  Maximum Gini (equal mix):  p₀=p₁=p₂=1/3
    Gini = 1 - 3×(1/3)² = 1 - 3×(1/9) = 1 - 1/3 = {1 - 1/3:.4f}

  Compare with 2-class maximum (p=0.5):
    Gini = 1 - 2×(0.5)² = 0.5000

  The maximum Gini scales as (1 - 1/k):
    k=2:  max = 0.500
    k=3:  max = 0.667
    k=10: max = 0.900
""")

print(f"  Numerical examples (3 classes):")
print(f"  {'Distribution':>30} | {'Gini':>8}")
print(f"  {'-'*30}-+-{'-'*8}")
cases = [
    ([10, 0, 0], "all class 0 (pure)"),
    ([7, 2, 1],  "mostly class 0"),
    ([5, 3, 2],  "mixed"),
    ([4, 3, 3],  "near uniform"),
    ([3, 4, 3],  "near uniform (shuffled)"),
]
for counts, label in cases:
    y_case = np.array([i for i, c in enumerate(counts) for _ in range(c)])
    g = gini_multiclass(y_case, n_classes=3)
    print(f"  {str(counts)+' — '+label:>30} | {g:>8.4f}")
''',
    },

    "Overfitting, Pruning, and Depth Control": {
        "description": "How tree depth controls bias-variance tradeoff, and cost-complexity pruning",
        "runnable": True,
        "code": '''
"""
================================================================================
OVERFITTING AND PRUNING IN DECISION TREES
================================================================================

An unconstrained decision tree perfectly memorises training data (zero training error).
This script demonstrates:
    1. How depth controls the bias-variance tradeoff
    2. Pre-pruning (max_depth, min_samples_leaf)
    3. Post-pruning via cost-complexity pruning (ccp_alpha in sklearn)
    4. Cross-validation to find the optimal depth

================================================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: DEPTH vs TRAINING/TEST ACCURACY (BIAS-VARIANCE)
# =============================================================================

print("=" * 65)
print("  PART 1: DEPTH AND THE BIAS-VARIANCE TRADEOFF")
print("=" * 65)

X, y = make_moons(n_samples=400, noise=0.3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"""
  Dataset: make_moons (n=400, noise=0.3)
  Task: binary classification of two interleaved crescents.
  A linear model gets ~85% — the true boundary is non-linear.

  {'Depth':>7} | {'Train Acc':>10} | {'Test Acc':>10} | {'# Leaves':>10} | {'Gen Gap':>10}
  {'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}""")

for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_tr, y_tr)
    tr_acc  = accuracy_score(y_tr, clf.predict(X_tr))
    te_acc  = accuracy_score(y_te, clf.predict(X_te))
    n_leaves = clf.get_n_leaves()
    gap     = tr_acc - te_acc
    depth_str = str(depth) if depth else "None"
    print(f"  {depth_str:>7} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {n_leaves:>10} | {gap:>10.4f}")

print(f"""
  Pattern:
    depth=1   → underfit (high bias): too simple to capture the moon shape
    depth=3–5 → sweet spot: captures the pattern, generalises well
    depth=None → overfit (high variance): perfect training, poor test

  The optimal depth is found by cross-validation, not by looking at training accuracy.
""")


# =============================================================================
# PART 2: PRE-PRUNING HYPERPARAMETERS
# =============================================================================

print("=" * 65)
print("  PART 2: PRE-PRUNING PARAMETERS")
print("=" * 65)

print(f"""
  Three main pre-pruning controls:
    max_depth:          stops growing at this depth
    min_samples_split:  node must have ≥ this many examples to split
    min_samples_leaf:   each leaf must have ≥ this many examples

  The effect of min_samples_leaf (fixing max_depth=None):
""")

print(f"  {'min_samples_leaf':>18} | {'Train Acc':>10} | {'Test Acc':>10} | {'# Leaves':>10}")
print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for msl in [1, 2, 5, 10, 20, 50]:
    clf = DecisionTreeClassifier(min_samples_leaf=msl, random_state=42)
    clf.fit(X_tr, y_tr)
    tr_acc   = accuracy_score(y_tr, clf.predict(X_tr))
    te_acc   = accuracy_score(y_te, clf.predict(X_te))
    n_leaves = clf.get_n_leaves()
    print(f"  {msl:>18} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {n_leaves:>10}")

print(f"""
  Larger min_samples_leaf:
    → fewer, larger leaves (simpler tree)
    → reduces overfitting at the cost of some training accuracy
    → acts as a regulariser analogous to λ in linear models
""")


# =============================================================================
# PART 3: COST-COMPLEXITY PRUNING (POST-PRUNING)
# =============================================================================

print("=" * 65)
print("  PART 3: COST-COMPLEXITY PRUNING (ccp_alpha)")
print("=" * 65)

print(f"""
  Post-pruning: grow the full tree, then prune subtrees.
  Criterion: Rα(T) = R(T) + α × |leaves(T)|

  sklearn's ccp_alpha is the α value — higher α = more aggressive pruning.

  Step 1: Find the optimal alpha using cross-validation on the full tree path.
""")

# Compute the full pruning path
clf_full = DecisionTreeClassifier(random_state=42)
path = clf_full.cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas[::5]   # sample every 5th alpha for readability

print(f"  {'ccp_alpha':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'# Leaves':>10}")
print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

best_alpha, best_te_acc = 0.0, 0.0
for alpha in alphas[:12]:
    clf_p = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf_p.fit(X_tr, y_tr)
    tr_acc   = accuracy_score(y_tr, clf_p.predict(X_tr))
    te_acc   = accuracy_score(y_te, clf_p.predict(X_te))
    n_leaves = clf_p.get_n_leaves()
    print(f"  {alpha:>10.6f} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {n_leaves:>10}")
    if te_acc > best_te_acc:
        best_te_acc = te_acc
        best_alpha  = alpha

print(f"""
  Best alpha (highest test accuracy): {best_alpha:.6f}
  → Use 5-fold cross-validation to select alpha robustly.
""")

# Cross-validation to pick best alpha
print(f"  Cross-validation over alpha values:")
print(f"  {'ccp_alpha':>10} | {'CV mean acc':>12} | {'CV std':>8}")
print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*8}")

best_cv_alpha = 0.0
best_cv_acc   = 0.0
for alpha in alphas[:8]:
    clf_cv = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf_cv, X_tr, y_tr, cv=5)
    if scores.mean() > best_cv_acc:
        best_cv_acc   = scores.mean()
        best_cv_alpha = alpha
    print(f"  {alpha:>10.6f} | {scores.mean():>12.4f} | {scores.std():>8.4f}")

print(f"\\n  CV-selected alpha: {best_cv_alpha:.6f}  →  mean CV accuracy: {best_cv_acc:.4f}")


# =============================================================================
# PART 4: DECISION TREE FOR REGRESSION — PIECEWISE CONSTANT APPROXIMATION
# =============================================================================

print("\\n" + "=" * 65)
print("  PART 4: REGRESSION TREE — PIECEWISE CONSTANT")
print("=" * 65)

# True function: y = sin(2πx) + noise
np.random.seed(0)
X_reg = np.sort(np.random.uniform(0, 1, 100)).reshape(-1, 1)
y_reg = np.sin(2 * np.pi * X_reg.ravel()) + np.random.normal(0, 0.2, 100)

print(f"""
  Regression task: approximate y = sin(2πx) + noise
  x ∈ [0, 1],  n=100 examples

  A regression tree approximates any function as a step function.
  More leaves → finer steps → better approximation (but risk of overfit).

  {'Depth':>8} | {'Train MSE':>10} | {'Test MSE':>10} | {'# Leaves':>10}
  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}""")

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=0)

for depth in [1, 2, 3, 5, 8, None]:
    reg = DecisionTreeRegressor(max_depth=depth, random_state=0)
    reg.fit(X_tr_r, y_tr_r)
    tr_mse  = np.mean((reg.predict(X_tr_r) - y_tr_r) ** 2)
    te_mse  = np.mean((reg.predict(X_te_r) - y_te_r) ** 2)
    n_leaves = reg.get_n_leaves()
    depth_str = str(depth) if depth else "None"
    print(f"  {depth_str:>8} | {tr_mse:>10.4f} | {te_mse:>10.4f} | {n_leaves:>10}")

print(f"""
  Depth=1 (stump): two flat segments — extreme underfitting
  Depth=3-5: reasonable approximation of the sine curve
  Depth=None: fits training noise perfectly — high test MSE

  How a depth-3 tree approximates y=sin(2πx):
    Leaf 1: x ∈ [0.00, 0.25)  → predict mean(y for x in this range) ≈ 0.8
    Leaf 2: x ∈ [0.25, 0.50)  → predict ≈ 0.4
    Leaf 3: x ∈ [0.50, 0.75)  → predict ≈ -0.8
    Leaf 4: x ∈ [0.75, 1.00)  → predict ≈ -0.4
    ... (piecewise constant step function)
""")
''',
    },

    "Feature Importance and Interpretability": {
        "description": "MDI importance, permutation importance, reading tree rules, and comparison with linear models",
        "runnable": True,
        "code": '''
"""
================================================================================
FEATURE IMPORTANCE AND INTERPRETABILITY
================================================================================

Decision trees are the most interpretable model we have studied.
This script explores:
    1. Mean Decrease in Impurity (MDI) — the built-in feature importance
    2. Permutation Importance — a more reliable alternative
    3. Reading the tree structure as human-readable rules
    4. Comparing importance measures with linear model coefficients

================================================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: MDI vs PERMUTATION IMPORTANCE
# =============================================================================

print("=" * 65)
print("  PART 1: MDI vs PERMUTATION IMPORTANCE")
print("=" * 65)

print("""
  Two importance measures:

  MDI (Mean Decrease in Impurity):
    Sum of impurity gain × fraction of samples across all nodes
    where a feature is used.
    Fast (computed during training).
    BIASED toward high-cardinality features (many unique values).

  Permutation Importance:
    Measure test accuracy. Then randomly shuffle one feature column.
    The drop in accuracy is that feature's importance.
    Model-agnostic, unbiased, but slower (needs test set).
    More reliable for comparing features of different types.
""")

# Use Breast Cancer dataset (30 features, binary classification)
cancer = load_breast_cancer()
X, y   = cancer.data, cancer.target
feat_names = cancer.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_tr, y_tr)

# MDI importances
mdi_imp = clf.feature_importances_

# Permutation importances (on test set)
perm_result = permutation_importance(clf, X_te, y_te, n_repeats=20, random_state=42)
perm_imp    = perm_result.importances_mean

# Show top 10 by MDI
mdi_order = np.argsort(mdi_imp)[::-1]

print(f"  Top 10 features by MDI (train) vs Permutation importance (test):")
print(f"  {'Rank':>5} | {'Feature':>28} | {'MDI':>8} | {'Perm Imp':>10} | {'Perm Std':>10}")
print(f"  {'-'*5}-+-{'-'*28}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

for rank, idx in enumerate(mdi_order[:10], 1):
    print(f"  {rank:>5} | {feat_names[idx]:>28} | {mdi_imp[idx]:>8.4f} | "
          f"{perm_imp[idx]:>10.4f} | {perm_result.importances_std[idx]:>10.4f}")

print(f"""
  Note: MDI and permutation importance may rank features differently.
  When rankings disagree, permutation importance is generally more trustworthy
  because it directly measures the feature's effect on held-out test performance.
""")


# =============================================================================
# PART 2: READING THE TREE AS HUMAN-READABLE RULES
# =============================================================================

print("=" * 65)
print("  PART 2: DECISION TREE AS HUMAN-READABLE RULES")
print("=" * 65)

# Fit a shallow tree on Iris for readability
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
fn = iris.feature_names
cn = iris.target_names

clf_iris = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_iris.fit(X_iris, y_iris)
acc_iris = (clf_iris.predict(X_iris) == y_iris).mean()

print(f"""
  Iris dataset (3 classes: Setosa, Versicolor, Virginica)
  max_depth=3, training accuracy: {acc_iris:.4f}

  Decision rules (sklearn export_text):
""")
print(export_text(clf_iris, feature_names=list(fn)))

print(f"""
  Reading the rules:
    IF petal length ≤ 2.45  →  Setosa (pure leaf: 50/50 training examples)
    ELIF petal width ≤ 1.75
          AND petal length ≤ 4.95  →  Versicolor (47/3 split)
          AND petal length > 4.95  →  Versicolor (but less pure: 2/4)
    ELIF petal width > 1.75  →  Virginica (43/1 split)

  This is the power of interpretability: a doctor/business analyst can read,
  audit, and challenge each decision. No linear model or neural network
  offers this level of transparency.
""")


# =============================================================================
# PART 3: FEATURE IMPORTANCE vs LINEAR MODEL COEFFICIENTS
# =============================================================================

print("=" * 65)
print("  PART 3: DT IMPORTANCE vs LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 65)

print("""
  A common question: do decision tree importances agree with logistic
  regression coefficients about which features matter?

  They measure different things:
    DT importance:   how much a feature reduces impurity across the tree
    LR coefficient:  the additive contribution of a unit increase in a feature
                     to the log-odds (assuming linearity)

  When the relationship is linear: they should roughly agree.
  When the relationship is non-linear: they may disagree significantly.
""")

# Breast Cancer: both models
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = LogisticRegression(C=1.0, max_iter=5000).fit(X_tr_sc, y_tr)
lr_coef_abs = np.abs(lr.coef_[0])

# Normalise LR coefficients to [0,1] for comparison
lr_coef_norm = lr_coef_abs / lr_coef_abs.sum()

print(f"  Top 8 features:")
print(f"  {'Feature':>28} | {'DT MDI':>8} | {'|LR coef|':>10} | {'Agree?':>8}")
print(f"  {'-'*28}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")

dt_order_full = np.argsort(mdi_imp)[::-1]
for idx in dt_order_full[:8]:
    agree = "✓" if lr_coef_norm[idx] > lr_coef_norm.mean() else "~"
    print(f"  {feat_names[idx]:>28} | {mdi_imp[idx]:>8.4f} | {lr_coef_norm[idx]:>10.4f} | {agree:>8}")

print(f"""
  Where they agree (✓): the feature has both high tree importance and
    a large logistic regression coefficient — it's genuinely predictive.

  Where they disagree (~): the feature is important to the tree
    (non-linear relationship it can capture) but weak in LR
    (the linear assumption doesn't hold for that feature).

  Advice:
    • If LR and DT agree: high confidence the feature is truly important
    • If only DT marks it important: the feature may have non-linear effects
    • If only LR marks it important: the feature may have linear effects that
      the tree didn't isolate (e.g., masked by correlated features)
""")


# =============================================================================
# PART 4: THE INSTABILITY PROBLEM — SAME DATA, DIFFERENT SEEDS
# =============================================================================

print("=" * 65)
print("  PART 4: HIGH VARIANCE — DIFFERENT SEEDS, DIFFERENT TREES")
print("=" * 65)

print(f"""
  Decision trees have HIGH VARIANCE: small changes in training data can
  produce structurally different trees. This is the key weakness.

  Demonstration: same data, 5 different random seeds → root splits vary.
""")

X_demo, y_demo = load_breast_cancer(return_X_y=True)

print(f"  {'Seed':>6} | {'Root feature':>28} | {'Root threshold':>16} | {'Test Acc':>10}")
print(f"  {'-'*6}-+-{'-'*28}-+-{'-'*16}-+-{'-'*10}")

X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
    X_demo, y_demo, test_size=0.3, random_state=99)

for seed in range(5):
    clf_s = DecisionTreeClassifier(max_depth=4, random_state=seed)
    # Bootstrap sample to simulate different training data
    n_tr = len(X_d_tr)
    idx  = np.random.RandomState(seed).choice(n_tr, n_tr, replace=True)
    X_boot, y_boot = X_d_tr[idx], y_d_tr[idx]
    clf_s.fit(X_boot, y_boot)
    te_acc = accuracy_score(y_d_te, clf_s.predict(X_d_te))
    root_feat = cancer.feature_names[clf_s.tree_.feature[0]]
    root_thresh = clf_s.tree_.threshold[0]
    print(f"  {seed:>6} | {root_feat:>28} | {root_thresh:>16.4f} | {te_acc:>10.4f}")

print(f"""
  Different bootstrap samples → different root features and thresholds.
  The accuracy is similar, but the TREES are structurally different.

  This is exactly why Random Forests work: averaging many such unstable trees
  cancels out the individual errors, producing a stable, accurate ensemble.
  The price: we lose the interpretability of individual tree rules.
""")
''',
    },

# ── Trigger: Logistic Regression ─────────────────────────────────
    "▶ Run: Decision Tree Classification": {
        "description": (
            "Runs 03_Decision_Tree_Classification.py from the Implementation folder. "
            "Decision Tree for Classification "
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Classification"
    / "03_Decision_Tree_Classification.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/supervised/Implementation/Classification/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },

# ── Trigger: Logistic Regression ─────────────────────────────────
    "▶ Run: Decision Tree Regression": {
        "description": (
            "Runs 07_Decision_Tree_Regression.py from the Implementation folder. "
            "Decision Tree for Regression "
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Regression"
    / "07_Decision_Tree_Regression.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/supervised/Implementation/Regression/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },


# ── Trigger: Logistic Regression ─────────────────────────────────
    "▶ Run: Decision Tree Classification Project": {
        "description": (
            "Runs main.py from the Implementation/Decision_Tree_Classificaiton folder. "
            "Decision Tree for Classificaiton Use Case"
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Decision_Tree_Classification"
    / "main.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/supervised/Implementation/Decision_Tree_Classificaiton/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    from collections import Counter

    print("\n" + "=" * 65)
    print("  DECISION TREES: LEARNING BY ASKING QUESTIONS")
    print("=" * 65)
    print("""
  Key concepts demonstrated:
    • Gini impurity and entropy — measuring node purity
    • CART: greedy recursive best-split search
    • Axis-aligned splits can represent any non-linear boundary
    • Depth controls bias-variance tradeoff
    • Feature importance: impurity-weighted contribution at each node
    • Bridge to Random Forests and Gradient Boosting
    """)

    np.random.seed(42)


    # ── Impurity functions ───────────────────────────────────────────────────
    def gini(y):
        if len(y) == 0: return 0.0
        c = np.bincount(y);
        p = c / len(y)
        return 1.0 - np.sum(p ** 2)


    def ent(y):
        if len(y) == 0: return 0.0
        c = np.bincount(y);
        p = c[c > 0] / len(y)
        return -np.sum(p * np.log2(p))


    # ── Tiny dataset ─────────────────────────────────────────────────────────
    y_parent = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1])  # 6 yes, 4 no
    y_left = np.array([1, 1, 1, 0, 0])  # 3 yes, 2 no
    y_right = np.array([1, 0, 1, 0, 1])  # 3 yes, 2 no

    print("=" * 65)
    print("  IMPURITY CALCULATIONS ON EXAMPLE NODE")
    print("=" * 65)
    print(f"\n  Parent: {y_parent.tolist()}  (n={len(y_parent)}, "
          f"{y_parent.sum()} positives)")
    print(f"    Gini    = {gini(y_parent):.4f}")
    print(f"    Entropy = {ent(y_parent):.4f} bits")

    print(f"\n  Left child:  {y_left.tolist()}")
    print(f"    Gini    = {gini(y_left):.4f}")
    print(f"    Entropy = {ent(y_left):.4f} bits")

    print(f"\n  Right child: {y_right.tolist()}")
    print(f"    Gini    = {gini(y_right):.4f}")
    print(f"    Entropy = {ent(y_right):.4f} bits")

    n = len(y_parent)
    nl, nr = len(y_left), len(y_right)
    wg = (nl / n) * gini(y_left) + (nr / n) * gini(y_right)
    we = (nl / n) * ent(y_left) + (nr / n) * ent(y_right)
    print(f"\n  Weighted Gini after split:    {wg:.4f}")
    print(f"  Gini Gain:                     {gini(y_parent) - wg:.4f}")
    print(f"\n  Weighted Entropy after split: {we:.4f}")
    print(f"  Information Gain:              {ent(y_parent) - we:.4f} bits")
    print(f"  (Gain = 0 here — this was a useless split)")

    # ── sklearn tree on Iris ─────────────────────────────────────────────────
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.datasets import load_iris

    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(iris.data, iris.target)
    acc = (clf.predict(iris.data) == iris.target).mean()

    print(f"\n{'=' * 65}")
    print(f"  SKLEARN TREE ON IRIS (max_depth=3, accuracy={acc:.4f})")
    print(f"{'=' * 65}")
    print(export_text(clf, feature_names=list(iris.feature_names)))

    print(f"  Feature importances (MDI):")
    for name, imp in sorted(zip(iris.feature_names, clf.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {name:>25}  {imp:.4f}  {bar}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1. CART grows trees greedily: at each node, find the split maximising gain
  2. Gini and Entropy are nearly equivalent — Gini is faster (no log)
  3. Decision boundaries are axis-aligned step functions — no linearity assumed
  4. Unconstrained trees memorise training data (zero train error, high test error)
  5. max_depth and min_samples_leaf are the key regularisation hyperparameters
  6. Feature importance = impurity gain × fraction of samples, summed over nodes
  7. Trees have high variance — small data changes → different structure
  8. Random Forests fix this: average many diverse trees (next module)
  9. Gradient Boosting builds trees sequentially to correct residual errors
  10. Decision trees are the only fully interpretable model in this series
    """)

VISUAL_HTML  = ""



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
        from Supervised_Learning.visuals.decision_tree_visual import (   # ← match your exact folder casing
            DT_VISUAL_HTML,
            DT_VISUAL_HEIGHT,
        )
        visual_html   = DT_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = DT_VISUAL_HEIGHT
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
