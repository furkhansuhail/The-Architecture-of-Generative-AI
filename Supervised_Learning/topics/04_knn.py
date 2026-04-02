"""
K-Nearest Neighbours — Learning by Analogy
===========================================

K-Nearest Neighbours is the most literal implementation of the intuition that
"similar inputs should have similar outputs." To classify a new point, KNN simply
finds the K training examples closest to it and takes a majority vote. To predict
a continuous value, it averages those K neighbours' targets.

There is no training phase. No parameters are fitted. No loss function is minimised.
KNN stores the entire training set and defers all computation to prediction time.
This makes it unique among every algorithm in this series — it is the only
instance-based, non-parametric, lazy learner we will study.

Its simplicity is both its strength and its weakness: KNN requires no modelling
assumptions and adapts perfectly to any decision boundary shape, but it scales
poorly, suffers in high dimensions, and is sensitive to irrelevant features.

"""

import base64
import os
import textwrap
import re


DISPLAY_NAME = "04 · K-Nearest Neighbours"
ICON         = "📍"
SUBTITLE     = "Non-parametric — classify by looking at who is nearby"
TOPIC_NAME   = "K-Nearest Neighbours"
VISUAL_HTML  = ""

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is K-Nearest Neighbours?

K-Nearest Neighbours (KNN) is a non-parametric learning algorithm that makes
predictions by finding the K most similar training examples to a query point and
aggregating their labels or values.

It is the algorithmic embodiment of reasoning by analogy: "This patient has
similar age, blood pressure, and cholesterol to three patients who had heart
attacks — so we should be worried about this one too."

KNN is radically different from every other algorithm in this series:

    Every other model:  trains a compact model (weights, trees, support vectors)
                        → discards training data after fitting
                        → prediction is fast (evaluate the compact model)

    KNN:                stores the ENTIRE training set
                        → no fitting phase at all
                        → prediction requires searching ALL training points

This "lazy" learning strategy is why KNN is called a lazy learner or
instance-based learner: it defers all computation to query time.

    Things that exist inside the model (stored, not learned):
        - The complete training dataset {(xᵢ, yᵢ)}   — memorised verbatim

    Things you control before querying (hyperparameters):
        - K                     — number of neighbours to consider
        - Distance metric       — how to measure "similarity"
        - Weighting scheme      — equal votes vs inverse-distance weights


### KNN in the ERM Framework — A Special Case

KNN does not fit naturally into the ERM framework because there is no explicit
optimisation over any parameters. It is a memorisation algorithm, not a
minimisation algorithm. We can still describe what it implicitly does:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Hypothesis class:  H = { f(x) = aggregate labels of the    │
    │                           K nearest training examples }     │
    │                     (implicitly partitions space into       │
    │                      Voronoi cells for K=1)                 │
    │                                                             │
    │  Loss function:     0-1 loss (classification) or            │
    │                     MSE (regression) — evaluated at query   │
    │                     time, not minimised during training     │
    │                                                             │
    │  Training:          Store (X, y) — O(1) time, O(n) space    │
    │                                                             │
    │  Prediction:        Find K nearest neighbours of x,         │
    │                     aggregate their labels — O(n·p) time    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The complete ERM comparison across all modules:

    Linear Regression:    MSE loss, gradient descent in weight space
    Logistic Regression:  BCE loss, gradient descent in weight space
    SVM:                  Hinge loss, quadratic programming
    Decision Tree:        Gini, greedy split search
    Random Forest:        0-1, averaging independent trees
    Gradient Boosting:    any loss, gradient descent in function space
    KNN:                  no optimisation — pure memorisation


### The Inductive Bias of KNN

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  KNN encodes a single fundamental belief:                   │
    │                                                             │
    │  SMOOTHNESS IN INPUT SPACE — examples that are close        │
    │  in the feature space are likely to have similar labels.    │
    │                                                             │
    │  This is the locality assumption. It is very weak —         │
    │  KNN makes almost no other assumptions about the data.      │
    │                                                             │
    │  Consequences:                                              │
    │    - Any decision boundary shape is representable (given    │
    │      enough data)                                           │
    │    - All features are implicitly weighted equally           │
    │    - The metric defines what "close" means — this is        │
    │      the single most impactful design choice in KNN         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The locality assumption is the weakest possible inductive bias — KNN makes fewer
assumptions than any other model we've studied. This is both its strength (nothing
is assumed away) and its weakness (high data requirements to compensate).


---


### Part 1: The Algorithm


### Classification:

For a query point x_q:
    1. Compute distance d(x_q, xᵢ) for every training point xᵢ
    2. Sort training points by distance to x_q
    3. Take the K closest points: {x₁, x₂, ..., x_K} (nearest first)
    4. Return majority class: ŷ = argmax_c  Σᵢ 1[yᵢ = c]

**Tie-breaking** (when two classes tie for the majority vote):
    - If K is even, ties are possible — sklearn picks the lower class index
    - Choose K to be odd for binary classification to avoid ties


### Regression:

For a query point x_q:
    1–3. Same: find K nearest neighbours
    4. Return mean of their target values:  ŷ = (1/K) Σᵢ yᵢ

**Weighted KNN:** give closer neighbours more influence:
    ŷ = Σᵢ (1/d(x_q, xᵢ)) · yᵢ  /  Σᵢ (1/d(x_q, xᵢ))

Weighting by inverse distance is usually better than equal voting because a very
close neighbour should carry more information than a distant one.


    # ======================================================================================= # 
    **Diagram 1 — KNN Classification (K=3 and K=7):**

    Dataset: ● = Class A,  ○ = Class B,  ? = query point

    K=3:                                    K=7:
    x₂ ↑                                    x₂ ↑
       │    ●   ○                               │    ●   ○
       │  ●   ●   ○                             │  ●   ●   ○
       │      (?)  ○                            │      (?)  ○
       │        ○                               │        ○
       │  ●                                     │  ●
       └──────────────────→ x₁                  └──────────────────→ x₁

    K=3 nearest: ●, ●, ○  → majority = ●  → predict Class A
    K=7 nearest: ●, ●, ○, ○, ○, ●, ○  → majority = ○  → predict Class B

    DIFFERENT K → DIFFERENT ANSWER!
    Small K: complex, jagged boundary (low bias, high variance)
    Large K: smooth, stable boundary (high bias, low variance)
    # ======================================================================================= # 


### The Decision Boundary of KNN:

For K=1, KNN creates a Voronoi diagram: the decision boundary runs exactly midway
between training points of different classes. Every test point is assigned to the
class of the single nearest training point.

For K > 1, the boundary is smoothed — a larger neighbourhood is consulted,
and the boundary becomes less sensitive to individual training points.


    # ======================================================================================= # 
    **Diagram 2 — K=1 Voronoi Tessellation:**

    Every point in space "belongs" to its nearest training example.
    The decision boundary between regions is always midway between
    training points of different classes.

    x₂ ↑
       │ ●│○│ ○ │
       │──┼─┼───│  ← decision boundary (equidistant lines)
       │ ●│●│ ○ │
       │──┼─┼───│
       │ ●│●│ ○ │
       └──────────→ x₁

    Each cell contains one training point. The cell boundary is the
    set of points equidistant to two adjacent training points.

    K=1: extremely jagged boundary (every training point matters)
    K=5: smoother (5 training points vote, local noise averages out)
    K=n: predict the global majority class (completely smooth, maximum bias)
    # ======================================================================================= # 


### Choosing K — The Bias-Variance Tradeoff:

    K = 1:     Memorises training set exactly — 0 training error
               High variance — boundary changes drastically with any new point
               Low bias — can represent any boundary shape

    K = n:     Always predicts the global majority class
               Zero variance — same prediction regardless of training set
               High bias — ignores all local structure

    K optimal: found by cross-validation
               Rule of thumb: start with K = sqrt(n), tune from there
               Odd values of K avoid ties in binary classification


---


### Part 2: Distance Metrics


### Why the Metric Matters:

"Close in feature space" depends entirely on what metric you use. Different metrics
create different neighbourhood shapes, which changes which training points are
selected as neighbours, which changes the prediction.

The choice of metric encodes domain knowledge about which differences between
feature values are meaningful.


### Common Distance Metrics:

**Euclidean Distance (L2):**
    d(x, z) = sqrt( Σⱼ (xⱼ − zⱼ)² )

    The standard straight-line distance in feature space.
    Sensitive to scale — features with large numerical range dominate.
    Assumes all features are equally important and on comparable scales.
    Use when: features are continuous and scaled similarly.

**Manhattan Distance (L1):**
    d(x, z) = Σⱼ |xⱼ − zⱼ|

    Sum of absolute differences along each axis.
    Less sensitive to outliers in individual features than Euclidean.
    Preferred when features have different scales or when outliers are common.
    Use when: high-dimensional data, or features represent independent counts.

**Minkowski Distance (Lp generalisation):**
    d(x, z) = ( Σⱼ |xⱼ − zⱼ|^p )^(1/p)

    p=1: Manhattan,  p=2: Euclidean,  p=∞: Chebyshev (max-axis distance)

**Cosine Similarity:**
    cos(x, z) = (x · z) / (||x|| · ||z||)

    Measures the angle between vectors, not their magnitude.
    Two vectors pointing in the same direction have similarity 1.
    Use when: magnitude is uninformative (document length, image brightness).
    Common in: text classification, recommender systems.

**Hamming Distance:**
    d(x, z) = fraction of positions where xⱼ ≠ zⱼ

    For categorical features — counts mismatches.
    Use when: features are categorical (one-hot encoded or integer-coded).


    # ======================================================================================= # 
    **Diagram 3 — L2 vs L1 Neighbourhoods (Unit Balls):**

    Which points count as "within distance 1" of the origin?

    L2 (Euclidean):             L1 (Manhattan):
    x₂ ↑                        x₂ ↑
    1  │  ·····                 1  │     *
       │ ·     ·                   │   *   *
    0  ├·       ·─→ x₁          0  ├*       *─→ x₁
       │ ·     ·                   │   *   *
   -1  │  ·····                -1  │     *

    L2: circular boundary (rotational symmetry)
    L1: diamond boundary (axis-aligned symmetry)
    L∞: square boundary

    The shape of the "unit ball" determines which points are neighbours.
    In high dimensions, L1 and cosine often outperform L2.
    # ======================================================================================= # 


### The Importance of Feature Scaling:

Without scaling, KNN is dominated by the feature with the largest numerical range.

    Example:
        Feature 1: Age      (range 20–80)
        Feature 2: Income   (range 20,000–200,000)
        Feature 3: Children (range 0–5)

    Without scaling, Income differences of $10,000 completely swamp Age
    differences of 10 years — KNN essentially ignores Age and Children.

    With standardisation (z-score scaling):
        All features contribute equally to the Euclidean distance.

    Always scale features before applying KNN (StandardScaler).

    ┌─────────────────────────────────────────────────────────────┐
    │  Rule: ALWAYS apply StandardScaler (or MinMaxScaler)        │
    │        before KNN. Without scaling, KNN is unreliable.      │
    │                                                             │
    │  Exception: if all features are naturally on the same scale │
    │  (e.g., image pixels 0–255, or normalised text features).   │
    └─────────────────────────────────────────────────────────────┘


---


### Part 3: The Curse of Dimensionality


### What Goes Wrong in High Dimensions?

KNN relies on the assumption that nearby points in feature space have similar labels.
In high dimensions, this assumption breaks down catastrophically — a phenomenon
known as the curse of dimensionality.

**Problem 1 — Distance concentration:**
In high dimensions, all points become approximately equidistant from any query point.
The ratio (max distance) / (min distance) → 1 as p → ∞.
When all points are equally far away, "nearest" neighbours are no longer meaningfully closer.

    Example: with p=1, a query point at 0.5 has neighbours at 0.4 and 0.6 (clearly close).
    With p=1000, almost all training points are roughly the same distance away.
    The "nearest" neighbours are only marginally closer than the "farthest" ones.

**Problem 2 — Data sparsity:**
The volume of a p-dimensional unit hypercube grows as 1^p = 1, but to cover
it with training points, we need exponentially more points.

    To have 10 neighbours within a region covering r fraction of each axis:
        n ≥ 10 / r^p

    For p=1, r=0.1: need n ≥ 100
    For p=10, r=0.1: need n ≥ 10^11   (infeasible)

The K nearest neighbours are not actually close — they might be drawn from
completely different parts of the feature space, providing no useful local signal.

**Problem 3 — Irrelevant features:**
Each irrelevant feature adds noise to the distance calculation, masking the signal
from relevant features. With 100 features but only 5 relevant ones, the distance
is mostly noise.


    # ======================================================================================= # 
    **Diagram 4 — The Curse of Dimensionality: Volume vs Edge Length:**

    To capture 10% of the data range in EACH dimension of a p-dim space,
    the edge length r of the hypercube needed grows as:

    n points in unit cube → need edge r such that r^p × n ≥ K neighbours

    p=1:  r = 0.10  (capture 10% of the range in 1D)
    p=2:  r = 0.32  (must expand to 32% in each direction)
    p=5:  r = 0.63  (must expand to 63% to still find neighbours)
    p=10: r = 0.79  (covers 79% of each feature range!)
    p=100: r ≈ 1.0  (must include almost entire feature space!)

    At p=100, "local neighbourhood" = "entire dataset" → KNN fails.

    MITIGATION STRATEGIES:
    ─────────────────────────────────────────────────────────────────
    1. Dimensionality reduction (PCA, t-SNE) before applying KNN
    2. Feature selection — remove irrelevant features
    3. Use larger K (more neighbours) to compensate for sparsity
    4. Consider other models (trees, linear models) for p > 20–50
    ─────────────────────────────────────────────────────────────────
    # ======================================================================================= # 


---


### Part 4: Approximate Nearest Neighbours — Scaling KNN


### The Computational Bottleneck:

Exact KNN prediction requires computing distances to all n training points: O(n·p).
For n=1,000,000 training examples and p=100 features, each prediction requires
100,000,000 multiplications. For real-time applications, this is prohibitive.

Three strategies scale KNN to large datasets:


### KD-Trees (for low dimensions):

A KD-tree is a binary tree that partitions the feature space along alternating axes.
Each node splits the data on one feature at the median value.

    Building:    O(n log n)  — sort and split recursively
    Query:       O(log n) average case   —  traverse the tree
                 O(n) worst case         —  pathological data distributions

    Works well for p ≤ 20. Degrades to O(n) for p > 20–30.
    This is the default in sklearn for p ≤ 30.


### Ball Trees:

Like KD-trees but partition space into nested hyperspheres (balls) rather than
axis-aligned rectangles. Can prune entire spheres when they're farther from the
query than the current K-th nearest neighbour.

    Building:    O(n log n)
    Query:       O(log n) average case
    Better than KD-trees for p > 20 and for non-Euclidean metrics.
    This is sklearn's default for p > 30.


### Approximate Nearest Neighbours (ANN):

For large n or large p, exact search is abandoned in favour of approximate
search that finds points within a guaranteed distance factor of the true nearest
neighbour, dramatically faster:

    FAISS (Facebook AI Similarity Search):
        Uses product quantisation and inverted file index.
        Can search 1 billion vectors in milliseconds on GPU.
        Used for: image search, recommendation systems, semantic search.

    Annoy (Approximate Nearest Neighbours Oh Yeah):
        Random projection trees.
        Memory-mapped index — can query without loading into RAM.
        Used for: Spotify music recommendation.

    HNSW (Hierarchical Navigable Small World graphs):
        Graph-based: build a multi-layer proximity graph.
        State-of-the-art for ANN in many benchmarks.

    ┌─────────────────────────────────────────────────────────────┐
    │  When to use which data structure:                          │
    │    n < 1,000:      brute force (always exact, fast)         │
    │    n < 1,000,000, p < 30:    KD-tree (sklearn default)      │
    │    n < 1,000,000, p > 30:    Ball tree                      │
    │    n > 1,000,000:            FAISS or HNSW                  │
    └─────────────────────────────────────────────────────────────┘


---


### Part 5: KNN for Regression — Piecewise Local Averaging


### How KNN Regression Works:

KNN regression predicts the target for x_q as the (possibly weighted) mean of
the K nearest neighbours' target values. This creates a locally-adaptive
regression function — not a global parametric form.

Properties of KNN regression:
    - As K → 1: interpolates training data exactly (high variance)
    - As K → n: predicts the global mean of all targets (high bias)
    - As n → ∞, K → ∞, K/n → 0: converges to the true conditional mean E[y|x]
      (this is the theoretical guarantee — given infinite data, KNN is optimal)


    # ======================================================================================= # 
    **Diagram 5 — KNN Regression at Different K:**

    TRUE FUNCTION: y = sin(x) + noise     (smooth curve)

    K=1:                          K=5:                     K=20:
    y ↑                            y ↑                      y ↑
      │ ·   · ·                      │ ·   · ·                │ ·   · ·
      │· ·  ·  ·                     │  ̃  ̃  ̃  ̃               │
      │    .    ·                     │  smoother              │ flat ─────
      │           ·                   │                        │ (mean of all)
      └────────────→ x                └────────────→ x         └────────────→ x
    Noisy/wiggly                    Good fit                 Underfit

    K=1 perfectly passes through each training point (overfit to noise)
    K=5 approximates the smooth function well
    K=20 over-smooths — begins to ignore local structure
    
    # ======================================================================================= # 


---


### Part 6: Advantages, Limitations, and When to Use KNN


### Advantages:

1. **Zero training time** — simply store the training data.
   Useful when training data updates frequently (no retraining needed).

2. **No model assumptions** — can represent any decision boundary.
   The decision boundary adapts perfectly to the data distribution.

3. **Naturally multi-class** — voting naturally handles any number of classes
   with no special modification (unlike SVM, which requires one-vs-rest).

4. **Interpretable predictions** — "These 5 similar patients all recovered,
   so you are likely to recover" is an intuitive and auditable explanation.

5. **Non-parametric** — no assumptions about the functional form of f(x).


### Limitations:

1. **Slow prediction** — O(n·p) per query with brute force.
   Not suitable for real-time prediction on large training sets.

2. **High memory** — stores the entire training set.
   For n=10M examples and p=100 features (float32), that's 4GB of RAM just for X.

3. **Curse of dimensionality** — performance degrades rapidly with p.
   Essentially fails for p > 50–100 without dimensionality reduction.

4. **Feature scaling required** — must standardise before applying KNN.

5. **No feature importance** — cannot tell you which features are relevant.
   All features contribute equally to the distance (unless you hand-weight them).

6. **Sensitive to irrelevant features** — each irrelevant feature adds noise
   to the distance, diluting the signal from relevant features.


### When KNN is the Right Choice:

    ┌─────────────────────────────────────────────────────────────┐
    │  Use KNN when:                                              │
    │    • Dataset is small to medium (n < 100k) and p is small   │
    │    • Decision boundary is highly irregular/non-parametric   │
    │    • Training data updates frequently (no retraining cost)  │
    │    • Interpretability ("here are your similar cases")       │
    │    • Quick baseline for comparison                          │
    │    • Recommendation systems (collaborative filtering)       │
    │                                                             │
    │  Avoid KNN when:                                            │
    │    • n is large (> 100k) — use tree-based or linear models  │
    │    • p is large (> 50) — curse of dimensionality            │
    │    • Features have very different scales (must scale first) │
    │    • Real-time prediction is needed (O(n·p) per query)      │
    │    • Features are categorical (use decision trees instead)  │
    └─────────────────────────────────────────────────────────────┘


### KNN in the Full Series Context:

KNN is in many ways the opposite of every other model:

    All other models:  parameterised, generalise beyond training data, fast to predict
    KNN:               non-parametric, memorises training data, slow to predict

The lesson KNN teaches is that the local structure of the data is fundamentally
the information we want to exploit. Every other algorithm tries to capture this
structure in a compressed parametric form. KNN skips the compression entirely —
and in doing so reveals exactly what is lost and what is gained by that choice.

    """


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ──────────────────────────────────────────────────────────────────────────
    Operation                   Complexity              Notes
    ──────────────────────────────────────────────────────────────────────────
    Training (store data)       O(1) time, O(n·p) space   just copy data
    Prediction (brute force)    O(n·p) per query        compute all distances
    Prediction (KD-tree)        O(p·log n) average      works for p ≤ 30
    Prediction (Ball tree)      O(p·log n) average      better for p > 30
    KD-tree building            O(n·p·log n)            done once at fit()
    Ball tree building          O(n·p·log n)            done once at fit()
    ──────────────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "KNN from Scratch": {
        "description": "Distance computation, neighbour search, and majority voting — built with NumPy only",
        "runnable": True,
        "code": '''
"""
================================================================================
KNN FROM SCRATCH — DISTANCE, SEARCH, VOTE
================================================================================

We implement KNN classification and regression from first principles:
    - Multiple distance metrics (Euclidean, Manhattan, Cosine)
    - Brute-force neighbour search
    - Majority vote (classification) and mean prediction (regression)
    - Weighted KNN (inverse distance weighting)

No sklearn involved until the comparison at the end.

================================================================================
"""

import numpy as np
from collections import Counter


# =============================================================================
# DISTANCE METRICS
# =============================================================================

def euclidean_distance(x, X_train):
    """
    Euclidean (L2) distance from query x to all training points.
    d(x, z) = sqrt(Σⱼ (xⱼ − zⱼ)²)

    Vectorised: compute all n distances at once.
    """
    diff = X_train - x          # shape (n, p)
    return np.sqrt((diff ** 2).sum(axis=1))   # shape (n,)


def manhattan_distance(x, X_train):
    """
    Manhattan (L1) distance: d(x, z) = Σⱼ |xⱼ − zⱼ|
    """
    return np.abs(X_train - x).sum(axis=1)


def cosine_distance(x, X_train):
    """
    Cosine distance: 1 - cosine_similarity
    Measures angle between vectors, not magnitude.
    d = 1 - (x·z) / (||x|| · ||z||)
    """
    dot       = X_train @ x                              # (n,)
    norm_x    = np.linalg.norm(x)
    norm_z    = np.linalg.norm(X_train, axis=1)          # (n,)
    denom     = norm_x * norm_z
    denom     = np.where(denom == 0, 1e-10, denom)
    return 1.0 - dot / denom


# =============================================================================
# KNN CLASSIFIER
# =============================================================================

class KNNClassifier:
    """
    K-Nearest Neighbours for multi-class classification.

    Parameters:
        k (int):       number of neighbours to vote
        metric (str):  distance function: 'euclidean', 'manhattan', 'cosine'
        weights (str): 'uniform' (equal votes) or 'distance' (inverse-dist weighted)
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k       = k
        self.metric  = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training set — KNN has no other training step."""
        self.X_train = X.astype(float)
        self.y_train = y
        return self

    def _distance(self, x):
        """Compute distances from x to all training points."""
        if self.metric == "euclidean":
            return euclidean_distance(x, self.X_train)
        elif self.metric == "manhattan":
            return manhattan_distance(x, self.X_train)
        elif self.metric == "cosine":
            return cosine_distance(x, self.X_train)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict_one(self, x):
        """Classify a single query point."""
        distances = self._distance(x)

        # Sort and take K nearest
        nn_idx    = np.argsort(distances)[:self.k]
        nn_dists  = distances[nn_idx]
        nn_labels = self.y_train[nn_idx]

        if self.weights == "uniform":
            # Simple majority vote
            return Counter(nn_labels).most_common(1)[0][0]
        else:
            # Inverse-distance weighted vote
            # Handle exact matches (distance = 0): give infinite weight → predict that class
            if (nn_dists == 0).any():
                zero_mask = nn_dists == 0
                return Counter(nn_labels[zero_mask]).most_common(1)[0][0]
            weights = 1.0 / nn_dists
            # Weighted vote: for each class, sum weights of neighbours with that class
            class_weights = {}
            for label, w in zip(nn_labels, weights):
                class_weights[label] = class_weights.get(label, 0.0) + w
            return max(class_weights, key=class_weights.get)

    def predict(self, X):
        """Classify all query points."""
        return np.array([self.predict_one(x) for x in X])

    def predict_proba(self, X):
        """Return class probability estimates (fraction of K neighbours per class)."""
        classes   = np.unique(self.y_train)
        all_probs = []
        for x in X:
            distances = self._distance(x)
            nn_idx    = np.argsort(distances)[:self.k]
            nn_labels = self.y_train[nn_idx]
            counts    = Counter(nn_labels)
            probs     = np.array([counts.get(c, 0) / self.k for c in classes])
            all_probs.append(probs)
        return np.array(all_probs)


# =============================================================================
# KNN REGRESSOR
# =============================================================================

class KNNRegressor:
    """
    K-Nearest Neighbours for regression.

    Prediction: mean (or inverse-distance weighted mean) of K neighbours.
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k       = k
        self.metric  = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.astype(float)
        self.y_train = y.astype(float)
        return self

    def predict_one(self, x):
        if self.metric == "euclidean":
            distances = euclidean_distance(x, self.X_train)
        else:
            distances = manhattan_distance(x, self.X_train)

        nn_idx   = np.argsort(distances)[:self.k]
        nn_dists = distances[nn_idx]
        nn_vals  = self.y_train[nn_idx]

        if self.weights == "uniform":
            return nn_vals.mean()
        else:
            if (nn_dists == 0).any():
                return nn_vals[nn_dists == 0].mean()
            weights = 1.0 / nn_dists
            return np.dot(weights, nn_vals) / weights.sum()

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# =============================================================================
# DEMO 1: STEP-BY-STEP ON A TINY DATASET
# =============================================================================

print("=" * 65)
print("  KNN FROM SCRATCH — STEP-BY-STEP")
print("=" * 65)

# 2D toy dataset
X_train = np.array([[1.0, 2.0],
                     [1.5, 1.8],
                     [5.0, 8.0],
                     [8.0, 8.0],
                     [1.0, 0.6],
                     [9.0, 11.0]])
y_train = np.array([0, 0, 1, 1, 0, 1])

x_query = np.array([4.0, 5.0])   # the point to classify

print(f"""
  Training set (6 points, 2 features, 2 classes):
  {'Index':>7} | {'x₁':>6} | {'x₂':>6} | {'Class':>6}
  {'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}""")
for i, (xi, yi) in enumerate(zip(X_train, y_train)):
    print(f"  {i:>7} | {xi[0]:>6.1f} | {xi[1]:>6.1f} | {int(yi):>6}")

print(f"\n  Query point: x_q = {x_query}")

# Compute all distances
dists = euclidean_distance(x_query, X_train)

print(f"\n  Euclidean distances from x_q to each training point:")
print(f"  {'Index':>7} | {'x₁':>6} | {'x₂':>6} | {'Distance':>10} | {'Class':>6}")
print(f"  {'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}")
for i, (xi, yi, d) in enumerate(zip(X_train, y_train, dists)):
    print(f"  {i:>7} | {xi[0]:>6.1f} | {xi[1]:>6.1f} | {d:>10.4f} | {int(yi):>6}")

for k in [1, 3, 5]:
    nn_idx    = np.argsort(dists)[:k]
    nn_labels = y_train[nn_idx]
    vote      = Counter(nn_labels).most_common(1)[0][0]
    print(f"\n  K={k}: nearest neighbours = {nn_idx.tolist()}, labels = {nn_labels.tolist()}")
    print(f"        Majority vote = Class {vote}")


# =============================================================================
# DEMO 2: EFFECT OF K ON ACCURACY
# =============================================================================

print("\n" + "=" * 65)
print("  DEMO 2: EFFECT OF K — BIAS-VARIANCE TRADEOFF")
print("=" * 65)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_classification(n_samples=400, n_features=2, n_informative=2,
                            n_redundant=0, class_sep=1.0, random_state=42)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42)

print(f"\n  Dataset: n=400, p=2 (scaled), binary classification")
print(f"\n  {'K':>5} | {'Train Acc':>10} | {'Test Acc':>10} | Interpretation")
print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}")

for k in [1, 3, 5, 10, 20, 50, 100, len(X_tr)]:
    clf = KNNClassifier(k=k)
    clf.fit(X_tr, y_tr)
    tr_acc = (clf.predict(X_tr) == y_tr).mean()
    te_acc = (clf.predict(X_te) == y_te).mean()
    if k == 1:
        note = "memorises train (0 train error)"
    elif k == len(X_tr):
        note = "always predicts majority class"
    elif te_acc > 0.87:
        note = "good generalisation"
    else:
        note = ""
    print(f"  {k:>5} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {note}")


# =============================================================================
# DEMO 3: DISTANCE METRIC COMPARISON
# =============================================================================

print("\n" + "=" * 65)
print("  DEMO 3: DISTANCE METRIC COMPARISON")
print("=" * 65)

print(f"\n  K=5, same dataset (n=400, 2D, scaled):")
print(f"  {'Metric':>12} | {'Train Acc':>10} | {'Test Acc':>10}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")

for metric in ["euclidean", "manhattan", "cosine"]:
    clf = KNNClassifier(k=5, metric=metric)
    clf.fit(X_tr, y_tr)
    tr_acc = (clf.predict(X_tr) == y_tr).mean()
    te_acc = (clf.predict(X_te) == y_te).mean()
    print(f"  {metric:>12} | {tr_acc:>10.4f} | {te_acc:>10.4f}")

print(f"""
  On 2D scaled data, Euclidean and Manhattan are usually similar.
  Cosine is more appropriate for high-dimensional text/embedding data.
  For this 2D continuous feature dataset, Euclidean is the natural choice.
""")


# =============================================================================
# DEMO 4: UNIFORM vs DISTANCE WEIGHTING
# =============================================================================

print("=" * 65)
print("  DEMO 4: UNIFORM vs INVERSE-DISTANCE WEIGHTING")
print("=" * 65)

print(f"\n  When a very close neighbour exists, distance weighting helps:")
print(f"  {'K':>5} | {'Uniform Acc':>12} | {'Distance-weighted Acc':>22}")
print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*22}")

for k in [3, 5, 10, 20]:
    clf_u = KNNClassifier(k=k, weights="uniform").fit(X_tr, y_tr)
    clf_d = KNNClassifier(k=k, weights="distance").fit(X_tr, y_tr)
    acc_u = (clf_u.predict(X_te) == y_te).mean()
    acc_d = (clf_d.predict(X_te) == y_te).mean()
    diff  = acc_d - acc_u
    note  = f"  (distance better by {diff:+.4f})" if abs(diff) > 0.005 else ""
    print(f"  {k:>5} | {acc_u:>12.4f} | {acc_d:>22.4f}{note}")

print(f"""
  Distance weighting helps most when K is large (more distant neighbours
  are down-weighted so they don't dominate the vote).
  For small K, the nearest neighbours are already close — weighting helps less.
""")
''',
    },

    "Curse of Dimensionality": {
        "description": "Why KNN fails in high dimensions — distance concentration, sparsity, irrelevant features",
        "runnable": True,
        "code": '''
"""
================================================================================
THE CURSE OF DIMENSIONALITY — EMPIRICAL DEMONSTRATION
================================================================================

This script demonstrates three failure modes of KNN in high dimensions:
    1. Distance concentration: all distances become equal
    2. Data sparsity: K nearest neighbours are far away
    3. Irrelevant features: noise dimensions overwhelm signal

================================================================================
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: DISTANCE CONCENTRATION
# =============================================================================

print("=" * 65)
print("  PART 1: DISTANCE CONCENTRATION IN HIGH DIMENSIONS")
print("=" * 65)

print(f"""
  In high dimensions, the ratio (max_dist - min_dist) / min_dist → 0.
  All pairwise distances become approximately equal.
  The "nearest" neighbour is barely closer than the "farthest" one.

  Demonstration: 1000 random points uniformly in p-dimensional unit cube.
""")

print(f"  {'p (dims)':>10} | {'Mean dist':>10} | {'Std dist':>10} | {'Max/Min ratio':>14} | {'Relative spread':>16}")
print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*16}")

for p in [1, 2, 5, 10, 20, 50, 100, 500, 1000]:
    n = 1000
    X = np.random.uniform(0, 1, (n, p))
    q = np.random.uniform(0, 1, (1, p))

    dists = np.sqrt(((X - q) ** 2).sum(axis=1))
    mean_d = dists.mean()
    std_d  = dists.std()
    ratio  = dists.max() / (dists.min() + 1e-10)
    spread = std_d / (mean_d + 1e-10)   # relative spread
    print(f"  {p:>10} | {mean_d:>10.4f} | {std_d:>10.4f} | {ratio:>14.3f} | {spread:>16.4f}")

print(f"""
  As p increases:
    Mean distance grows (points spread out)
    Std deviation grows but SLOWER than mean
    Max/Min ratio → 1 (all distances similar)
    Relative spread decreases → distances concentrate around the mean

  At p=1000, the nearest neighbour is only marginally closer than the
  farthest one — the concept of "nearest" loses meaning.
""")


# =============================================================================
# PART 2: KNN ACCURACY DEGRADES WITH DIMENSION
# =============================================================================

print("=" * 65)
print("  PART 2: KNN ACCURACY vs DIMENSIONALITY")
print("=" * 65)

print(f"""
  We keep the number of INFORMATIVE features fixed at 5,
  but add increasing numbers of noise (uninformative) features.
""")

print(f"  {'Total p':>8} | {'Noise feats':>12} | {'KNN(K=5)':>10} | {'RandomForest':>14}")
print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*14}")

from sklearn.ensemble import RandomForestClassifier

for n_noise in [0, 5, 10, 20, 50, 95]:
    n_total = 5 + n_noise
    X, y = make_classification(
        n_samples=500, n_features=n_total, n_informative=5,
        n_redundant=0, n_repeated=0, n_clusters_per_class=1,
        random_state=42
    )
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    rf  = RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

    knn_acc = accuracy_score(y_te, knn.predict(X_te))
    rf_acc  = accuracy_score(y_te, rf.predict(X_te))
    print(f"  {n_total:>8} | {n_noise:>12} | {knn_acc:>10.4f} | {rf_acc:>14.4f}")

print(f"""
  KNN accuracy collapses as noise dimensions are added.
  Random Forest remains robust — feature importance lets it ignore noise.

  This is the practical implication of the curse of dimensionality:
    → KNN needs feature selection or dimensionality reduction before application
    → Tree-based models handle irrelevant features naturally
""")


# =============================================================================
# PART 3: SAMPLE COMPLEXITY — HOW MUCH DATA DOES KNN NEED?
# =============================================================================

print("=" * 65)
print("  PART 3: SAMPLE COMPLEXITY — DATA NEEDED VS DIMENSION")
print("=" * 65)

print(f"""
  To have K=5 neighbours within a neighbourhood covering r=10% of each
  feature axis, we need n ≥ K / r^p training points.

  n_needed = 5 / (0.1)^p   to have local neighbours
""")

print(f"  {'p':>5} | {'n needed for local K=5':>25} | Note")
print(f"  {'-'*5}-+-{'-'*25}-+-{'-'*30}")

for p in [1, 2, 3, 5, 10, 20, 50]:
    n_needed = 5 / (0.1 ** p)
    if n_needed < 1e6:
        note = f"{n_needed:.0f}"
    elif n_needed < 1e9:
        note = f"{n_needed:.2e}"
    else:
        note = f"{n_needed:.2e}  (astronomical)"
    print(f"  {p:>5} | {note:>25} | {'feasible' if n_needed < 1e7 else 'INFEASIBLE'}")

print(f"""
  For p=10: need 50 BILLION training examples for local neighbourhoods.
  KNN only works in low dimensions or with massive datasets.
""")


# =============================================================================
# PART 4: DIMENSIONALITY REDUCTION HELPS KNN
# =============================================================================

print("=" * 65)
print("  PART 4: PCA + KNN — DIMENSIONALITY REDUCTION SAVES KNN")
print("=" * 65)

from sklearn.decomposition import PCA

X_high, y_high = make_classification(
    n_samples=500, n_features=50, n_informative=5,
    n_redundant=0, n_repeated=0, n_clusters_per_class=1,
    random_state=42
)

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_high)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_high, test_size=0.25, random_state=42)

print(f"\n  Dataset: n=500, p=50 (5 informative + 45 noise)")
print(f"\n  {'Model':>30} | {'Test Accuracy':>14}")
print(f"  {'-'*30}-+-{'-'*14}")

# KNN on full 50D
knn_full = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
acc_full = accuracy_score(y_te, knn_full.predict(X_te))
print(f"  {'KNN (K=5, full 50D)':>30} | {acc_full:>14.4f}")

# KNN + PCA (reduce to 5D first)
for n_comp in [2, 5, 10, 20]:
    pca = PCA(n_components=n_comp)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    explained = pca.explained_variance_ratio_.sum()

    knn_pca = KNeighborsClassifier(n_neighbors=5).fit(X_tr_pca, y_tr)
    acc_pca = accuracy_score(y_te, knn_pca.predict(X_te_pca))
    print(f"  {'KNN (K=5, PCA→'+str(n_comp)+'D)':>30} | {acc_pca:>14.4f}  "
          f"(var explained: {explained:.1%})")

# RF baseline
rf_base = RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
acc_rf  = accuracy_score(y_te, rf_base.predict(X_te))
print(f"  {'Random Forest (50D, no PCA)':>30} | {acc_rf:>14.4f}")

print(f"""
  PCA before KNN recovers much of the lost accuracy by discarding noise dimensions.
  The number of PCA components is itself a hyperparameter — tune via CV.
  RandomForest handles the high-dimensional noise naturally (no PCA needed).
""")
''',
    },

    "KNN for Regression and Hyperparameter Tuning": {
        "description": "KNN regression, K selection via cross-validation, distance weighting, scaling impact",
        "runnable": True,
        "code": '''
"""
================================================================================
KNN REGRESSION AND HYPERPARAMETER TUNING
================================================================================

This script demonstrates:
    1. KNN regression — piecewise local averaging
    2. Cross-validation for K selection
    3. The critical importance of feature scaling
    4. Weighted vs unweighted KNN for regression
    5. KNN vs other regression models

================================================================================
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.datasets import make_regression, make_classification

np.random.seed(42)


# =============================================================================
# PART 1: KNN REGRESSION — PIECEWISE LOCAL AVERAGING
# =============================================================================

print("=" * 65)
print("  PART 1: KNN REGRESSION — LOCAL AVERAGING")
print("=" * 65)

# True function: y = sin(2πx) + noise
X_1d = np.linspace(0, 1, 100).reshape(-1, 1)
y_1d = np.sin(2 * np.pi * X_1d.ravel()) + np.random.normal(0, 0.15, 100)

X_tr_1d, X_te_1d, y_tr_1d, y_te_1d = train_test_split(
    X_1d, y_1d, test_size=0.25, random_state=42)

print(f"\n  Task: approximate y = sin(2πx) + noise  (n=100)")
print(f"\n  {'K':>5} | {'Train MSE':>10} | {'Test MSE':>10} | Behaviour")
print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}")

for k in [1, 2, 5, 10, 20, 50, 75]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_tr_1d, y_tr_1d)
    tr_mse = mean_squared_error(y_tr_1d, knn.predict(X_tr_1d))
    te_mse = mean_squared_error(y_te_1d, knn.predict(X_te_1d))
    if k == 1:
        note = "interpolates training exactly (overfit)"
    elif 5 <= k <= 15:
        note = "good fit"
    elif k > 50:
        note = "over-smoothed (high bias)"
    else:
        note = ""
    print(f"  {k:>5} | {tr_mse:>10.4f} | {te_mse:>10.4f} | {note}")


# =============================================================================
# PART 2: CROSS-VALIDATION FOR K SELECTION
# =============================================================================

print("\n" + "=" * 65)
print("  PART 2: SELECTING K WITH CROSS-VALIDATION")
print("=" * 65)

X_r, y_r = make_regression(n_samples=400, n_features=5, noise=15, random_state=42)
scaler_r  = StandardScaler()
X_r_sc    = scaler_r.fit_transform(X_r)
X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(X_r_sc, y_r, test_size=0.25, random_state=42)

print(\n  5-fold cross-validation across K values (n=400, p=5):")
print(  {'K':>5} | {'CV MSE (mean)':>14} | {'CV MSE (std)':>13} | Best?")
print(  {'-'*5}-+-{'-'*14}-+-{'-'*13}-+-{'-'*6}")

best_k, best_cv_mse = 1, float("inf")
cv_results = []
for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = -cross_val_score(knn, X_r_tr, y_r_tr, cv=5,
                               scoring="neg_mean_squared_error")
    cv_mse = scores.mean()
    cv_results.append((k, cv_mse, scores.std()))
    if cv_mse < best_cv_mse:
        best_cv_mse = cv_mse
        best_k = k

for k, mse, std in cv_results:
    mark = "← BEST" if k == best_k else ""
    print(f"  {k:>5} | {mse:>14.2f} | {std:>13.2f} | {mark}")

# Evaluate best K on test set
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_r_tr, y_r_tr)
te_mse = mean_squared_error(y_r_te, knn_best.predict(X_r_te))
print(f"\n  Best K={best_k} (CV), test MSE = {te_mse:.2f}")


# =============================================================================
# PART 3: FEATURE SCALING IS CRITICAL
# =============================================================================

print("\n" + "=" * 65)
print("  PART 3: FEATURE SCALING — CRITICAL FOR KNN")
print("=" * 65)

print("""
  KNN uses distances — features on larger scales dominate the distance.
  Without scaling, the algorithm ignores small-scale features.
""")

# Dataset with very different feature scales
np.random.seed(42)
n = 500
age    = np.random.uniform(20, 80, n)        # range 60
income = np.random.uniform(20000, 200000, n)  # range 180,000
score  = np.random.uniform(0, 1, n)          # range 1

# True class: based on all three features
y_scale = ((age > 45) & (income > 100000) | (score > 0.7)).astype(int)

X_scale = np.column_stack([age, income, score])
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_scale, y_scale, test_size=0.25, random_state=42)

# Without scaling
knn_no_scale = KNeighborsClassifier(n_neighbors=5)
knn_no_scale.fit(X_tr_s, y_tr_s)
acc_no_scale = accuracy_score(y_te_s, knn_no_scale.predict(X_te_s))

# With StandardScaler
scaler_s  = StandardScaler()
X_tr_s_sc = scaler_s.fit_transform(X_tr_s)
X_te_s_sc = scaler_s.transform(X_te_s)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_tr_s_sc, y_tr_s)
acc_scaled = accuracy_score(y_te_s, knn_scaled.predict(X_te_s_sc))

print(f"  Feature ranges: age=[20,80], income=[20k,200k], score=[0,1]")
print(f"\n  {'KNN (K=5, no scaling)':>30}: test accuracy = {acc_no_scale:.4f}")
print(f"  {'KNN (K=5, StandardScaler)':>30}: test accuracy = {acc_scaled:.4f}")

print(f"""
  Without scaling: income range is ~3000× larger than score range.
  Distance is almost entirely determined by income.
  Age and score have almost no influence on which neighbours are found.

  With scaling: all features contribute equally to Euclidean distance.

  Distance to neighbour without scaling:
    age diff = 30  → contributes 30²   = 900
    income diff = 50,000 → contributes 50000² = 2,500,000,000  ← dominates completely
    score diff = 0.5 → contributes 0.5² = 0.25

  With StandardScaler, all three contribute roughly equally.
""")


# =============================================================================
# PART 4: KNN vs OTHER MODELS — WHERE DOES KNN WIN?
# =============================================================================

print("=" * 65)
print("  PART 4: KNN vs OTHER MODELS — ACCURACY COMPARISON")
print("=" * 65)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

datasets = {
    "Moons (non-linear)":        __import__("sklearn.datasets", fromlist=["make_moons"]).make_moons(300, noise=0.2, random_state=42),
    "Circles (concentric)":      __import__("sklearn.datasets", fromlist=["make_circles"]).make_circles(300, noise=0.1, factor=0.4, random_state=42),
    "Linear (easy)":             make_classification(300, 2, n_informative=2, n_redundant=0, class_sep=2.0, random_state=42),
    "High-dim (p=30)":           make_classification(500, 30, n_informative=8, random_state=42),
}

models = {
    "KNN K=5":         KNeighborsClassifier(n_neighbors=5),
    "KNN K=15":        KNeighborsClassifier(n_neighbors=15),
    "Logistic Reg":    LogisticRegression(C=1.0, max_iter=1000),
    "SVM (RBF)":       SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Random Forest":   RandomForestClassifier(100, random_state=42, n_jobs=-1),
}

print(f"\n  {'Dataset':>22}", end="")
for name in models:
    print(f" | {name:>14}", end="")
print()
print("  " + "-"*22, end="")
for _ in models:
    print(f"-+-{'-'*14}", end="")
print()

for ds_name, (X_ds, y_ds) in datasets.items():
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_ds)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_ds, test_size=0.25, random_state=42)
    print(f"  {ds_name:>22}", end="")
    for clf in models.values():
        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf_copy.predict(X_te))
        print(f" | {acc:>14.4f}", end="")
    print()

print(f"""
  KNN shines on: non-linear datasets (moons, circles) with small p
  KNN struggles on: high-dimensional data (p=30)
  SVM and RF are competitive or better across the board
  Linear problems: Logistic Regression wins (correct model for linear data)
""")
''',
    },

    "KNN as a Recommendation System": {
        "description": "Collaborative filtering via KNN — the algorithm behind 'users like you also bought'",
        "runnable": True,
        "code": '''
"""
================================================================================
KNN FOR RECOMMENDATION — COLLABORATIVE FILTERING
================================================================================

KNN's "find similar examples" logic maps naturally to collaborative filtering:
    User-based CF: find K users similar to the target user, recommend items
                   those users liked that the target hasn't seen yet.
    Item-based CF: find K items similar to an item the user liked, recommend
                   those similar items.

This script implements user-based collaborative filtering from scratch,
using cosine similarity (the standard choice for sparse rating vectors).

================================================================================
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# TOY RATING MATRIX
# =============================================================================

print("=" * 65)
print("  KNN-BASED RECOMMENDATION — USER-BASED COLLABORATIVE FILTERING")
print("=" * 65)

# Rows = users, Columns = items, Values = ratings (0 = not rated)
# Items: Python, JavaScript, Rust, Java, Go, Ruby, C++, Swift
items = ["Python", "JavaScript", "Rust", "Java", "Go", "Ruby", "C++", "Swift"]
users = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"]

# Rating matrix (0 = not seen/not rated)
R = np.array([
    # Py   JS  Rust Java   Go  Ruby  C++  Swift
    [5,    4,   0,   2,    0,   3,   0,   0],  # Alice
    [4,    5,   1,   3,    2,   4,   0,   0],  # Bob
    [0,    0,   5,   1,    5,   0,   5,   4],  # Carol
    [1,    2,   4,   0,    4,   0,   5,   5],  # Dave
    [5,    4,   2,   4,    3,   5,   0,   0],  # Eve
    [0,    0,   5,   2,    5,   0,   5,   3],  # Frank
    [4,    5,   0,   5,    1,   4,   0,   1],  # Grace
], dtype=float)

print(f"\n  Rating matrix (rows=users, cols=items, 0=not rated):")
print(f"  {'':>8}", end="")
for item in items:
    print(f"  {item:>12}", end="")
print()
print(f"  {'-'*8}", end="")
for _ in items:
    print(f"--{'-'*12}", end="")
print()
for user, row in zip(users, R):
    print(f"  {user:>8}", end="")
    for val in row:
        cell = f"{int(val)}" if val > 0 else "—"
        print(f"  {cell:>12}", end="")
    print()


# =============================================================================
# USER-BASED COLLABORATIVE FILTERING WITH COSINE SIMILARITY
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  STEP 1: COMPUTE USER-USER COSINE SIMILARITY")
print(f"{'=' * 65}")

# Cosine similarity between all pairs of users
# (based only on rated items — treat 0 as "not rated", not "rated 0")
sim_matrix = cosine_similarity(R)  # shape (n_users, n_users)

print(f"\n  Cosine similarity matrix (1.0 = identical taste):")
print(f"  {'':>8}", end="")
for user in users:
    print(f"  {user:>8}", end="")
print()
for i, user_i in enumerate(users):
    print(f"  {user_i:>8}", end="")
    for j in range(len(users)):
        print(f"  {sim_matrix[i, j]:>8.3f}", end="")
    print()

print(f"\n  Interpretation:")
print(f"    Alice-Bob:   {sim_matrix[0,1]:.3f} — both like Python, JS, Ruby  ← very similar")
print(f"    Alice-Carol: {sim_matrix[0,2]:.3f} — opposite tastes (Carol: Rust/Go/C++)")
print(f"    Carol-Frank: {sim_matrix[2,5]:.3f} — both like Rust, Go, C++     ← very similar")


# =============================================================================
# MAKE RECOMMENDATIONS FOR A TARGET USER
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  STEP 2: RECOMMENDATIONS FOR 'Carol'")
print(f"{'=' * 65}")

target_user = "Carol"
target_idx  = users.index(target_user)
K_neighbours = 3

print(f"\n  Target user: {target_user}")
print(f"  Carol has rated: {[items[j] for j in range(len(items)) if R[target_idx,j] > 0]}")
print(f"  Carol has NOT rated: {[items[j] for j in range(len(items)) if R[target_idx,j] == 0]}")
print(f"  Finding K={K_neighbours} nearest neighbours (most similar users)...")

# Exclude the target user themselves
sims      = sim_matrix[target_idx].copy()
sims[target_idx] = -1   # exclude self
nn_idx    = np.argsort(sims)[::-1][:K_neighbours]

print(f"\n  {K_neighbours} most similar users:")
for rank, idx in enumerate(nn_idx, 1):
    shared  = [items[j] for j in range(len(items))
                if R[target_idx,j] > 0 and R[idx,j] > 0]
    print(f"    {rank}. {users[idx]:>8} (similarity={sims[idx]:.3f}), "
          f"shared ratings: {shared}")

# Predict ratings for unrated items
print(f"\n  Predicted ratings for {target_user}'s unrated items:")
print(f"  (weighted average of similar users' ratings, weighted by similarity)")
print(f"  {'Item':>14} | {'Predicted Rating':>18} | {'Based on'}")
print(f"  {'-'*14}-+-{'-'*18}-+-{'-'*30}")

unrated_items = [j for j in range(len(items)) if R[target_idx, j] == 0]

predictions = {}
for j in unrated_items:
    total_weight = 0.0
    weighted_sum = 0.0
    sources = []
    for idx in nn_idx:
        if R[idx, j] > 0:   # only neighbours who rated this item
            w = sims[idx]
            weighted_sum += w * R[idx, j]
            total_weight += w
            sources.append(f"{users[idx]}={int(R[idx,j])}")
    if total_weight > 0:
        pred = weighted_sum / total_weight
        predictions[j] = pred
        src_str = ", ".join(sources)
        print(f"  {items[j]:>14} | {pred:>18.3f} | {src_str}")
    else:
        print(f"  {items[j]:>14} | {'No neighbours rated':>18} | —")

# Top recommendations
if predictions:
    top_recs = sorted(predictions.items(), key=lambda x: -x[1])
    print(f"\n  TOP RECOMMENDATIONS for {target_user}:")
    for rank, (j, score) in enumerate(top_recs, 1):
        print(f"    {rank}. {items[j]}  (predicted rating: {score:.2f})")


# =============================================================================
# ITEM-BASED COLLABORATIVE FILTERING
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  ITEM-BASED FILTERING: 'Items similar to Python'")
print(f"{'=' * 65}")

# Transpose: each item is now a vector of user ratings
item_sim = cosine_similarity(R.T)   # shape (n_items, n_items)

python_idx = items.index("Python")
sims_items = item_sim[python_idx].copy()
sims_items[python_idx] = -1
top_similar = np.argsort(sims_items)[::-1][:4]

print(f"\n  Items most similar to Python (based on rating patterns):")
for rank, idx in enumerate(top_similar, 1):
    print(f"    {rank}. {items[idx]:>14} (cosine similarity = {sims_items[idx]:.3f})")

print(f"""
  Item-based CF is more stable than user-based CF because:
    - Items change less frequently than user preferences
    - Item-item similarity matrix can be precomputed offline
    - Scales better: similarity computed once, lookups at query time

  This is exactly how Amazon's original recommendation engine worked (2003):
    "Customers who bought X also bought..." = item-based KNN CF
""")
''',
    },

    # ── Trigger: Logistic Regression ─────────────────────────────────
    "▶ Run Module: KNN Classification": {
        "description": (
            "Runs 01_KNN_Classification.py from the Implementation folder. "
            "KNN for Classification "
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "supervised"
    / "Implementation"
    / "Classification"
    / "01_KNN_Classification.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Implementation/Supervised Model/Classification/")
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
    "▶ Run Module: KNN Regression": {
        "description": (
            "Runs 01_KNN_Regression.py from the Implementation folder. "
            "KNN for Regression "
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
    / "06_KNN_Regression.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Supervised_Learning/Implementation/Regression/")
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
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_classification, make_moons
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 65)
    print("  K-NEAREST NEIGHBOURS: LEARNING BY ANALOGY")
    print("=" * 65)
    print("""
  Key concepts demonstrated:
    • No training — stores entire dataset, defers all work to query time
    • Distance metric defines "similarity" — always scale features first
    • K controls bias-variance tradeoff (K=1: high variance, K=n: high bias)
    • Curse of dimensionality: fails for p > 20-50 without PCA first
    • Distance weighting: closer neighbours vote more strongly
    • OOB equivalent: cross-validation to select K
    """)

    np.random.seed(42)

    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42)

    print("=" * 65)
    print("  EFFECT OF K ON MAKE_MOONS CLASSIFICATION")
    print("=" * 65)

    print(f"\n  Dataset: make_moons n=500, noise=0.25 (scaled)")
    print(f"\n  {'K':>6} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gen Gap':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    best_k, best_acc = 1, 0.0
    for k in [1, 3, 5, 10, 20, 50, 100, 375]:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        if te_acc > best_acc:
            best_acc, best_k = te_acc, k
        k_label = str(k) if k < len(X_tr) else f"{k} (=n)"
        print(f"  {k_label:>6} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {tr_acc-te_acc:>10.4f}")

    print(f"\n  Best K={best_k} with test accuracy={best_acc:.4f}")

    # 5-fold CV to find optimal K
    print(f"\n  5-fold CV to find best K:")
    cv_best_k, cv_best_score = 1, 0.0
    for k in [1, 3, 5, 7, 10, 15, 20, 30]:
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X_tr, y_tr, cv=5).mean()
        flag = " ← best CV so far" if cv > cv_best_score else ""
        if cv > cv_best_score:
            cv_best_score, cv_best_k = cv, k
        print(f"    K={k:>3}: CV accuracy = {cv:.4f}{flag}")

    # Final evaluation
    final_clf = KNeighborsClassifier(n_neighbors=cv_best_k)
    final_clf.fit(X_tr, y_tr)
    final_acc = accuracy_score(y_te, final_clf.predict(X_te))

    print(f"\n  CV-selected K={cv_best_k}, final test accuracy = {final_acc:.4f}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1.  KNN stores all training data — O(1) train time, O(n·p) prediction time
  2.  Inductive bias: nearby points in feature space have similar labels
  3.  Always apply StandardScaler before KNN (distance is scale-sensitive)
  4.  K=1: perfectly memorises train set (0 train error, high variance)
  5.  Optimal K found by cross-validation (not by looking at train accuracy)
  6.  Distance weighting (1/d) usually outperforms uniform voting for large K
  7.  Curse of dimensionality: KNN fails for p > 20–50 without PCA
  8.  KD-tree/Ball tree reduce prediction cost from O(n·p) to O(p·log n)
  9.  For n > 1M: use FAISS or HNSW (approximate nearest neighbours)
  10. Collaborative filtering (Spotify, Amazon) = KNN in user/item space
  11. KNN is the only model in the series that makes no parameter assumptions
  12. Interpretable: "here are your 5 most similar training examples"
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
        from Supervised_Learning.visuals.knn_visual import (   # ← match your exact folder casing
            KNN_VISUAL_HTML,
            KNN_VISUAL_HEIGHT,
        )
        visual_html   = KNN_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = KNN_VISUAL_HEIGHT
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

