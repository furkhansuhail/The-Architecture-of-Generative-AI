"""
LightGBM - Light Gradient Boosting Machine
===========================================

LightGBM is a highly efficient gradient boosting framework that builds
an ensemble of decision trees sequentially, where each tree corrects the
errors of its predecessors. It is one of the most powerful and widely-used
algorithms in modern machine learning, especially in structured/tabular data.

"""

import base64
import os
import textwrap
import re

TOPIC_NAME = "LightGBM: Light Gradient Boosting Machine"
DISPLAY_NAME = "03 . LightGBM · Gradient Boosting"
ICON = "🌲"
SUBTITLE = "Ensemble Learning with Sequential Decision Trees"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER — converts local images to base64 HTML for st.markdown()
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    """Convert a local image file to an HTML <img> tag with base64 data.
    This allows images to render inside st.markdown() with unsafe_allow_html=True.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is LightGBM?

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework 
developed by Microsoft Research in 2017. It is designed to be fast, memory-efficient, 
and highly accurate — making it the go-to algorithm for tabular data competitions 
and production ML systems.

LightGBM doesn't just run gradient boosting — it reimagines how gradient boosting 
is implemented. It introduces two key algorithmic innovations (GOSS and EFB) and a 
novel tree-growing strategy (leaf-wise growth) that together make it dramatically 
faster than earlier methods like XGBoost, while matching or exceeding their accuracy.

To understand LightGBM deeply, you need to understand three layers:

    Layer 1: What is a Decision Tree? (the base learner)
    Layer 2: What is Gradient Boosting? (the ensemble strategy)
    Layer 3: What does LightGBM do differently? (the innovations)


##### PART 1 — THE BASE LEARNER: DECISION TREES

### What is a Decision Tree?

A decision tree is a model that makes predictions by asking a series of 
yes/no questions about the input features. Think of it like a game of 
20 Questions — each question narrows down the answer space.

    THE ANATOMY OF A DECISION TREE
    ════════════════════════════════════════════════════════════

                          ┌──────────────────────┐
                          │    Age > 30?         │   ← ROOT NODE (first split)
                          │  (Split on feature)  │     Asks: which feature, which value?
                          └────────┬────────┬────┘
                               YES │        │ NO
                    ┌──────────────┘        └─────────────┐
                    ▼                                     ▼
          ┌──────────────────┐                  ┌──────────────────┐
          │  Income > 50k?   │                  │   Student?       │  ← INTERNAL NODES
          └────────┬────┬────┘                  └─────────┬────┬───┘
               YES │    │ NO                          YES │    │ NO
          ┌────────┘    └──────┐               ┌──────────┘    └──────┐
          ▼                    ▼               ▼                      ▼
    ┌──────────┐         ┌──────────┐     ┌──────────┐            ┌──────────┐
    │ Predict: │         │ Predict: │     │ Predict: │            │ Predict: │
    │ BUY ✅   │         │ NO BUY ❌│     │ BUY ✅   │            │ NO BUY ❌│  ← LEAF NODES
    │ val=0.8  │         │ val=0.2  │     │ val=0.7  │            │ val=0.1  │     (predictions)
    └──────────┘         └──────────┘     └──────────┘            └──────────┘

    KEY TERMS:
    • Root Node    — the very first split (top of the tree)
    • Internal Node — an intermediate split question
    • Leaf Node    — a terminal node that holds the prediction value
    • Depth        — how many splits from root to leaf
    • num_leaves   — total number of leaf nodes


### How does a Decision Tree find the best split?

At each node, the tree must decide: "Which feature, and which value of that 
feature, gives the best split?" It does this by minimizing a loss criterion.

For regression, the most common criterion is Variance Reduction:

    Gain = Variance(parent) - [weighted average of Variance(left) + Variance(right)]

For classification, the criterion is often Gini Impurity:

    Gini(node) = 1 - Σ p_k²

    Where p_k is the fraction of samples belonging to class k in that node.

The split that maximizes the Gain is chosen. This is an exhaustive search over 
every feature and every possible threshold value — which is where LightGBM 
introduces its most important optimization (the Histogram trick, covered in Part 3).


### Decision Trees: Strengths and Limitations

    STRENGTHS                              LIMITATIONS
    ─────────────────────────────          ─────────────────────────────
    ✓ Naturally handles non-linearity      ✗ High variance (overfitting)
    ✓ No feature scaling required          ✗ Unstable: small data changes
    ✓ Handles mixed feature types            cause very different trees
    ✓ Interpretable (for shallow trees)    ✗ Not competitive on its own
    ✓ Handles missing values natively      ✗ Poor extrapolation

The key insight of Gradient Boosting is that a single tree's weaknesses 
(high variance, instability) become strengths when you build MANY small trees 
sequentially, each one correcting the errors of the previous ensemble.


##### PART 2 — THE ENSEMBLE STRATEGY: GRADIENT BOOSTING

### What is Boosting?

Boosting is an ensemble strategy that builds models SEQUENTIALLY. Unlike 
Random Forest (which trains trees in parallel, independently), boosting 
trains each new tree to correct the residual errors of the current ensemble.

    RANDOM FOREST (Bagging)          GRADIENT BOOSTING
    ───────────────────────          ─────────────────
    Tree 1  Tree 2  Tree 3           Tree 1 → Tree 2 → Tree 3 → ...
       ↓       ↓       ↓                ↑ learns from  ↑ learns from
       └───────┴───────┘               │ errors of T1  │ errors of T1+T2
             ↓                         │               │
        Vote / Average             Fit residuals   Fit residuals
                                   of T1           of T1+T2

    Trees are INDEPENDENT                Trees are SEQUENTIAL
    (parallel, can run together)         (each depends on the last)


### The Gradient Boosting Algorithm

Gradient Boosting frames the problem as function optimization in functional space.
We are trying to find a function F(x) that minimizes a loss function L(y, F(x)).

Instead of updating the parameters of a single model (like neural networks do 
with gradient descent on weights), Gradient Boosting adds a new small tree at 
each step that points in the direction of the negative gradient of the loss.

    THE GRADIENT BOOSTING ALGORITHM (Step by Step)
    ════════════════════════════════════════════════════════════

    INITIALIZATION:
        F₀(x) = argmin_γ Σ L(yᵢ, γ)
        (e.g., for MSE loss, this is just the mean of all y values)

    FOR m = 1 to M (building tree m):

        STEP 1 — Compute Pseudo-Residuals (Negative Gradients):

            rᵢₘ = -[ ∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ) ]  evaluated at F = F_{m-1}

            For MSE loss L = (y - F)²/2:
                rᵢₘ = yᵢ - F_{m-1}(xᵢ)     ← these are just the residuals!

            For Log-Loss (binary classification):
                rᵢₘ = yᵢ - σ(F_{m-1}(xᵢ))  ← residuals in probability space

        STEP 2 — Fit a Decision Tree hₘ(x) to the pseudo-residuals {rᵢₘ}:

            The tree doesn't predict y directly.
            It predicts the "direction we need to nudge" the ensemble.

        STEP 3 — Find the Optimal Leaf Values γⱼₘ:

            For each leaf j in tree m:
                γⱼₘ = argmin_γ Σ_{xᵢ∈leaf_j} L(yᵢ, F_{m-1}(xᵢ) + γ)

        STEP 4 — Update the Ensemble:

            F_m(x) = F_{m-1}(x) + η · hₘ(x)

            Where η (eta) is the learning rate (shrinkage factor, typically 0.01–0.3)

    FINAL PREDICTION:
        F_M(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)


    WORKED EXAMPLE — MSE Regression:
    ════════════════════════════════════════════════════════════

    Training data:   x = [1, 2, 3, 4]
                     y = [2, 4, 5, 4]

    STEP 0: F₀ = mean(y) = (2+4+5+4)/4 = 3.75

    STEP 1 — Compute residuals (pseudo-gradients for MSE):
        r = y - F₀ = [2-3.75, 4-3.75, 5-3.75, 4-3.75]
                   = [-1.75,  0.25,   1.25,   0.25]

    STEP 2 — Fit Tree 1 to residuals r:
        (Suppose tree splits at x <= 1.5 vs x > 1.5)
        Left leaf (x=1):  predicts -1.75
        Right leaf (x=2,3,4): predicts mean(0.25, 1.25, 0.25) = 0.583

    STEP 3 — Update (with learning rate η = 0.1):
        F₁ = F₀ + 0.1 × Tree1 predictions
        F₁(x=1) = 3.75 + 0.1 × (-1.75) = 3.575
        F₁(x=2) = 3.75 + 0.1 × (0.583) = 3.808
        F₁(x=3) = 3.75 + 0.1 × (0.583) = 3.808
        F₁(x=4) = 3.75 + 0.1 × (0.583) = 3.808

    STEP 4 — Compute NEW residuals for Tree 2:
        r₂ = y - F₁ = [-1.575, 0.192, 1.192, 0.192]

    Each iteration: residuals shrink, prediction improves, trees learn fine detail.


### Why Does the Learning Rate Matter?

The learning rate η is one of the most important hyperparameters in gradient boosting.

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  HIGH η (e.g., 1.0)   → Each tree has full influence           │
    │                          → Fast convergence                    │
    │                          → High risk of OVERFITTING            │
    │                          → Each step may "overshoot"           │
    │                                                                │
    │  LOW η  (e.g., 0.01)  → Each tree has small influence          │
    │                          → Needs many more trees (n_estimators)│
    │                          → More regularization                 │
    │                          → Generally better generalization     │
    │                                                                │
    │  RULE OF THUMB: lower η + more trees > higher η + few trees    │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    The learning rate is a SHRINKAGE factor — it prevents any single tree from 
    dominating, forcing the ensemble to spread learning across many trees.
    This is the same intuition as weight decay in neural networks.


### Loss Functions in LightGBM

LightGBM supports many loss functions — the gradient computed in Step 1 
changes depending on the task:

    TASK                  LOSS FUNCTION L(y, F)         GRADIENT (pseudo-residuals)
    ─────────────────     ───────────────────────────   ──────────────────────────────
    Regression (MSE)      (y - F)² / 2                  r = y - F
    Regression (MAE)      |y - F|                        r = sign(y - F)
    Binary Classification  -[y log σ(F) + (1-y) log(1-σ(F))]  r = y - σ(F)
    Multiclass            -Σ yₖ log p̂ₖ                  r = y_k - p̂_k (one-vs-rest)
    Ranking (LambdaRank)  Pairwise log loss              gradient from pairwise comparisons

    σ(F) = 1 / (1 + e^(-F))    ← Sigmoid/logistic function


##### PART 3 — THE LIGHTGBM INNOVATIONS

### Why LightGBM, Not Just Gradient Boosting?

Standard gradient boosting (and even early XGBoost) becomes slow when:
  1. The dataset has millions of rows (scanning all samples for splits is O(n))
  2. The dataset has thousands of features (scanning all features is O(d))

LightGBM solves both problems with two algorithmic innovations:

    Problem 1 (too many rows)     → GOSS: Gradient-based One-Side Sampling
    Problem 2 (too many features) → EFB:  Exclusive Feature Bundling

And it replaces the brute-force split-finding algorithm with:
    Slow pre-sorted split finding → Histogram-based split finding


### Innovation 1 — Histogram-Based Split Finding

The classic method for finding the best split is the pre-sorted algorithm:
  For each feature, sort all data points by that feature value, then scan 
  through every adjacent pair as a candidate split threshold.

  Cost: O(n × d × log n) per tree — extremely expensive for large n.

LightGBM replaces this with a histogram approach:

    HISTOGRAM-BASED SPLIT FINDING
    ════════════════════════════════════════════════════════════

    STEP 1: Discretize each continuous feature into bins (256 bins by default)

    Continuous values:  [0.1, 0.15, 0.23, 0.45, 0.67, 0.71, 0.89, 0.92]
                              ↓
    Binned values:         [  0,    0,    1,    2,    3,    3,    4,    4 ]
    (5 bins shown)

    STEP 2: For each feature, build a histogram counting:
            - sum of gradients per bin
            - count of samples per bin

         Bin 0    Bin 1    Bin 2    Bin 3    Bin 4
        ┌──────┬──────┬──────┬──────┬──────┐
        │ Σg=  │ Σg=  │ Σg=  │ Σg=  │ Σg=  │   (gradient sum per bin)
        │-0.3  │ 0.1  │ 0.4  │-0.2  │ 0.6  │
        ├──────┼──────┼──────┼──────┼──────┤
        │ n=2  │ n=1  │ n=1  │ n=2  │ n=2  │   (sample count per bin)
        └──────┴──────┴──────┴──────┴──────┘

    STEP 3: Scan the histogram (256 bins, not n samples!) to find best split

    COST COMPARISON:
    ┌──────────────────────────────────────────────────────────────┐
    │  Pre-sorted:    O(n × d)   per node  — slow for large n      │
    │  Histogram:     O(B × d)   per node  — B = 256 (constant!)   │
    │                                                              │
    │  With n=1,000,000 rows, B=256:                               │
    │  Pre-sorted: 1,000,000 × d operations                        │
    │  Histogram:        256 × d operations   ← 4,000x faster!     │
    └──────────────────────────────────────────────────────────────┘

    BONUS TRICK — Histogram Subtraction:
    For a binary tree split, you only need to compute the histogram 
    for ONE child. The other child's histogram is:
        hist(right) = hist(parent) - hist(left)
    This halves the work at every internal node!


### Innovation 2 — GOSS: Gradient-Based One-Side Sampling

Key observation: In gradient boosting, data points with SMALL gradients 
are already well-predicted by the current ensemble. Data points with LARGE 
gradients are the ones the model is currently getting wrong — these are 
the most informative for the next tree.

GOSS exploits this by keeping all high-gradient samples but randomly 
dropping most low-gradient samples during tree construction:

    GOSS ALGORITHM
    ════════════════════════════════════════════════════════════

    INPUT: Current ensemble F_{m-1}, training data, 
           top_rate a (e.g. 20%), other_rate b (e.g. 10%)

    STEP 1: Compute gradients gᵢ = |∂L/∂F| for each sample i

    STEP 2: Sort samples by |gᵢ| in descending order

    STEP 3: Keep the top a × 100% of samples (large gradient)
            These are the "hard" examples — keep ALL of them

    STEP 4: Randomly sample b × 100% from the rest (small gradient)
            These are the "easy" examples — keep only a fraction

    STEP 5: When building the tree, apply an amplification factor 
            to the sampled small-gradient instances:

                weight = (1 - a) / b

            This compensates for the under-sampling, ensuring the 
            histogram gradient sums remain statistically consistent.

    RESULT: Fewer samples to process, information nearly preserved.

    VISUAL INTUITION:
    ────────────────────────────────────────────────────────────

    ALL SAMPLES sorted by gradient magnitude:

    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
    │◄─── LARGE GRADIENT ─────►│◄──── SMALL GRADIENT ──────►│
    │    Hard examples         │    Easy examples           │

    GOSS KEEPS:
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          ░░░░░░   (random 10%)        │
    │    ALL kept              │   Sampled + amplified      │

    Speedup: from n samples → (a + b) × n samples per tree
    With a=0.2, b=0.1: only 30% of data per tree — 3x speedup!


### Innovation 3 — EFB: Exclusive Feature Bundling

Key observation: In high-dimensional sparse datasets (e.g., one-hot encoded 
features), many features are MUTUALLY EXCLUSIVE — they rarely take non-zero 
values simultaneously. For example, in a one-hot encoding of "color":

    is_red    is_blue    is_green
       1          0          0       ← Only one is ever non-zero at a time
       0          1          0
       0          0          1

These exclusive features can be BUNDLED into a single feature without 
losing information, reducing the effective feature count d.

    EFB ALGORITHM
    ════════════════════════════════════════════════════════════

    STEP 1: Build a conflict graph
            Edge between feature i and feature j if they take 
            non-zero values simultaneously for at least one sample

    STEP 2: Greedy graph coloring
            Group features that don't conflict (are exclusive) 
            into the same bundle (same "color" in graph coloring)

    STEP 3: Encode bundles
            Offset values so they occupy different bin ranges:

            Feature A (range 0-10) + Feature B (range 0-20) → 
            Bundle: A stays 0-10, B becomes 11-31
                    └──────────────┬───────────────┘
                                   │ single feature, no information lost!

    EXAMPLE:
    ─────────────────────────────────────────────────────────
    Before EFB:    4 exclusive features  → 4 histogram builds
    After EFB:     1 bundle             → 1 histogram build
    Speedup: 4x on feature processing!

    In practice (NLP, rec-sys, ad-click):
        - Dataset may have 10,000 features, but only 1,000 are 
          ever non-zero at once → ~10x effective feature reduction


### Leaf-Wise vs Level-Wise Tree Growth

This is another key LightGBM innovation. Traditional tree-growing algorithms 
(including CART, used by scikit-learn and original XGBoost) grow trees LEVEL-BY-LEVEL.

LightGBM grows trees LEAF-BY-LEAF:

    LEVEL-WISE GROWTH (Traditional)      LEAF-WISE GROWTH (LightGBM)
    ═══════════════════════════════      ═══════════════════════════════

    Split all leaves at current depth    Always split the leaf with
    before going deeper:                 the MAXIMUM gain:

          [Root]                               [Root]
           /  \                                 /  \
        [L1] [L2]  ← Split both!           [L1] [L2]
         / \   / \                                /  \
       [3][4] [5][6] ← Split all!             [L3] [L4] ← Only split L2 (higher gain)
                                                /  \
                                             [L5] [L6]   ← Then split L3 (next highest gain)

    Grows balanced, bushy trees.         Grows DEEPER, more asymmetric trees.
    More iterations needed for           Finds complex patterns faster,
    complex patterns.                    but can overfit without regularization.

    ACCURACY COMPARISON (fixed num_leaves=31):
    Level-wise needs ~5 levels (2⁵=32 leaves)
    Leaf-wise reaches 31 leaves in fewer iterations with HIGHER GAIN per step.

    REGULARIZATION — controlling leaf-wise growth:
    • min_child_samples: minimum samples required in a leaf (prevents tiny leaves)
    • min_child_weight: minimum sum of hessians in a leaf
    • max_depth: hard cap on tree depth (use with leaf-wise to prevent explosion)
    • num_leaves: total maximum number of leaves (KEY parameter, replaces max_depth)

    IMPORTANT: For leaf-wise trees, num_leaves is more important than max_depth.
    Typical guidance: num_leaves < 2^(max_depth)


##### PART 4 — THE MATH: GRADIENT AND HESSIAN

### Second-Order Optimization (Newton Boosting)

Standard Gradient Boosting uses only the FIRST derivative (gradient) of the 
loss function. LightGBM (like XGBoost) uses BOTH the first and second derivatives:

    First derivative (gradient):   gᵢ = ∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)
    Second derivative (hessian):   hᵢ = ∂²L(yᵢ, F(xᵢ)) / ∂²F(xᵢ)

Using a second-order Taylor expansion of the loss function:

    L(yᵢ, F + Δ) ≈ L(yᵢ, F) + gᵢ·Δ + ½ hᵢ·Δ²

The optimal leaf value for leaf j is:

    γⱼ* = - Σᵢ∈j gᵢ / (Σᵢ∈j hᵢ + λ)

    Where λ is the L2 regularization term (reg_lambda in LightGBM).

And the SPLIT GAIN used to evaluate a potential split is:

    Gain = ½ [ (Σ_L gᵢ)² / (Σ_L hᵢ + λ) 
             + (Σ_R gᵢ)² / (Σ_R hᵢ + λ)
             - (Σ_P gᵢ)² / (Σ_P hᵢ + λ) ] - γ

    Where:  L = left child,  R = right child,  P = parent
            γ = minimum gain threshold (min_split_gain)
            λ = L2 regularization (reg_lambda)

    INTUITION:
    ─────────────────────────────────────────────────────────
    The hessian hᵢ acts as a WEIGHT — samples with high hessian 
    (high curvature of loss) are treated as more important.

    For MSE: hᵢ = 1 for all i (flat curvature → uniform weights)
    For Log-loss: hᵢ = σ(F)(1-σ(F)) → high near decision boundary,
                  low for confidently predicted samples


    GRADIENTS AND HESSIANS FOR COMMON LOSSES:
    ════════════════════════════════════════════════════════════

    ┌──────────────────────┬──────────────────────┬──────────────────────┐
    │ Loss Function        │ Gradient gᵢ          │ Hessian hᵢ           │
    ├──────────────────────┼──────────────────────┼──────────────────────┤
    │ MSE: (y-F)²/2        │ F - y                │ 1                    │
    │ MAE: |y - F|         │ sign(F - y)          │ 0 (not used)         │
    │ Binary Log-Loss      │ σ(F) - y             │ σ(F)(1 - σ(F))       │
    │ Multi-class Softmax  │ p̂_k - y_k            │ p̂_k(1 - p̂_k)         │
    └──────────────────────┴──────────────────────┴──────────────────────┘


##### PART 5 — REGULARIZATION AND PREVENTING OVERFITTING

### Regularization in LightGBM

LightGBM provides multiple regularization mechanisms that operate at 
different levels of the model:

    LEVEL 1: TREE STRUCTURE REGULARIZATION
    ──────────────────────────────────────────────────────────────────
    Parameter           What it controls                Effect
    ─────────────────   ────────────────────────────    ──────────────
    num_leaves          Max leaf nodes per tree          Lower → simpler trees
    max_depth           Max depth of each tree           Lower → simpler trees
    min_child_samples   Min samples in each leaf         Higher → smoother fit
    min_child_weight    Min hessian sum in each leaf     Higher → more conservative
    min_split_gain      Min gain required to split       Higher → fewer splits

    LEVEL 2: WEIGHT REGULARIZATION (in the Gain formula)
    ──────────────────────────────────────────────────────────────────
    reg_lambda (λ)   L2 regularization on leaf weights  Shrinks leaf values toward 0
    reg_alpha  (α)   L1 regularization on leaf weights  Induces sparsity (some leaves → 0)

    LEVEL 3: SAMPLING-BASED REGULARIZATION
    ──────────────────────────────────────────────────────────────────
    subsample          Fraction of rows per tree         Acts like Dropout
    colsample_bytree   Fraction of features per tree     Random subspace method
    colsample_bylevel  Fraction of features per level    More fine-grained control
    bagging_freq       How often to subsample            Controls subsample frequency

    LEVEL 4: ENSEMBLE-LEVEL REGULARIZATION
    ──────────────────────────────────────────────────────────────────
    learning_rate (η)  Shrinks each tree's contribution  Lower → stronger regularization
    n_estimators       Number of trees (with early stop) More trees + lower η = better
    early_stopping      Stop when validation loss stops   Most important in practice!

    DIAGRAM — The Regularization Web:
    ════════════════════════════════════════════════════════════

    Underfitting ◄──────────────────────────────────► Overfitting
         │                                                  │
    Too few trees      SWEET SPOT                    Too many trees
    High learning rate ────────►◄──────── Low learning rate
    max_depth = 2    ────────►◄────────   max_depth = 100
    num_leaves = 4   ────────►◄────────   num_leaves = 512
    min_child_samples=100 ───►◄─── min_child_samples=1

    RULE OF THUMB:
    Start with n_estimators=1000, learning_rate=0.05, num_leaves=31
    Then tune num_leaves, min_child_samples, reg_lambda to reduce variance.


### Early Stopping — The Most Important Regularization Technique

Early stopping monitors a validation metric and stops training when 
performance stops improving, even if n_estimators hasn't been reached.

    ┌─────────────────────────────────────────────────────────────┐
    │  TRAINING LOSS    ── (keeps falling forever)                │
    │  ──────────────────────────────────────────────────────     │
    │                                                             │
    │  VALIDATION LOSS  ── (U-shaped curve)                       │
    │       ╲                                                     │
    │        ╲               Best point!                          │
    │         ╲_______________|_______________                    │
    │                         ↑              ╲                    │
    │                 Early stopping fires    ╲ (overfitting)     │
    │                 here (patience=50)                          │
    └─────────────────────────────────────────────────────────────┘

    Usage:
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50)])

    • stopping_rounds: how many trees of no improvement before stopping
    • The final model uses the best iteration, not the last one.


##### PART 6 — FEATURE IMPORTANCE AND INTERPRETABILITY

### Feature Importance in LightGBM

LightGBM computes three types of feature importance — each tells a different story:

    TYPE 1 — Split Importance (importance_type='split')
    ────────────────────────────────────────────────────────────────────
    Counts how many times each feature is used to split a node across 
    all trees. Easy to compute but can be biased toward high-cardinality 
    features (features with many unique values get more split opportunities).

        Feature A: used in 120 splits
        Feature B: used in 87 splits    ← B is less used
        Feature C: used in 3 splits     ← C is rarely used


    TYPE 2 — Gain Importance (importance_type='gain')
    ────────────────────────────────────────────────────────────────────
    Sums the total gain (improvement in loss) contributed by each feature 
    across all splits. More meaningful than split count — a feature used 
    once in a very informative split is ranked higher than one used many 
    times with trivial gain.

        Feature A: total gain = 0.34   (many splits, moderate each)
        Feature B: total gain = 0.51   ← B is actually more useful despite fewer splits
        Feature C: total gain = 0.02


    TYPE 3 — SHAP Values (model-agnostic, most reliable)
    ────────────────────────────────────────────────────────────────────
    SHAP (SHapley Additive exPlanations) assigns each feature a value 
    representing its contribution to the prediction for a SPECIFIC SAMPLE.

    Based on cooperative game theory (Shapley values):

    For sample i:
        prediction_i = base_value + Σⱼ SHAP_value(feature j, sample i)

    Where base_value is the average prediction across all training data.

    SHAP values tell you: "Feature j pushed this specific prediction up/down 
    by this amount." Unlike split/gain importance, SHAP works per-sample.

    LightGBM has native SHAP support:
        shap_values = model.predict(X, pred_contrib=True)

    VISUAL — SHAP Force Plot (single sample):
    ────────────────────────────────────────────────────────────────────
    base value = 0.35

    age=45  +0.15 ──►
    income=80k +0.20 ──►
                                         prediction = 0.72
    ◄── -0.08  student=no
    ◄── -0.05  region=rural
    ◄── -0.15  credit_score=600
    ────────────────────────────────────────────────────────────────────
    Features pushing prediction UP   →    positive SHAP values
    Features pushing prediction DOWN ←    negative SHAP values


##### PART 7 — KEY HYPERPARAMETERS REFERENCE

### The Complete LightGBM Hyperparameter Map

Things that exist INSIDE the model (learned):
    - Leaf values (γⱼ): the actual prediction at each leaf, updated each tree
    - Tree structure: which features and thresholds to split on

Things set BEFORE training (hyperparameters):

    CORE ENSEMBLE PARAMETERS:
    ─────────────────────────────────────────────────────────────────────
    Parameter            Default    Effect
    ─────────────────    ────────   ────────────────────────────────────
    n_estimators         100        Number of trees (boost rounds)
    learning_rate        0.1        Shrinkage factor η (0.01–0.3 typical)
    num_leaves           31         Max leaves per tree (most important!)
    max_depth            -1         Max tree depth (-1 = unlimited)

    SAMPLING PARAMETERS (prevent overfitting via randomization):
    ─────────────────────────────────────────────────────────────────────
    subsample            1.0        Row sampling fraction per tree
    subsample_freq       0          Apply subsample every k iterations
    colsample_bytree     1.0        Feature sampling fraction per tree
    min_child_samples    20         Min data points per leaf
    min_child_weight     1e-3       Min hessian sum per leaf

    REGULARIZATION PARAMETERS:
    ─────────────────────────────────────────────────────────────────────
    reg_alpha            0.0        L1 regularization coefficient
    reg_lambda           0.0        L2 regularization coefficient
    min_split_gain       0.0        Min gain to make a split

    LIGHTGBM-SPECIFIC SPEED PARAMETERS:
    ─────────────────────────────────────────────────────────────────────
    max_bin              255        Histogram bins (higher = more accurate, slower)
    min_data_in_bin      3          Min data per bin
    boosting             'gbdt'     'gbdt', 'dart', 'goss', 'rf'
    tree_learner         'serial'   'serial', 'feature', 'data', 'voting' (parallel)
    num_threads          0          CPU threads (0 = all available)
    device_type          'cpu'      'cpu' or 'gpu'

    TASK PARAMETERS:
    ─────────────────────────────────────────────────────────────────────
    objective            'regression' / 'binary' / 'multiclass' / 'rank_xendcg'
    metric               'rmse' / 'mae' / 'binary_logloss' / 'multi_logloss' / 'auc'
    num_class            1          Number of classes (for multiclass)
    is_unbalance         False      For imbalanced binary classification


    RECOMMENDED STARTING POINT:
    ════════════════════════════════════════════════════════════

    params = {
        'objective': 'binary',          # or 'regression'
        'metric': 'binary_logloss',     # or 'rmse'
        'n_estimators': 1000,           # high value — rely on early stopping
        'learning_rate': 0.05,          # low learning rate
        'num_leaves': 31,               # start here, tune up carefully
        'min_child_samples': 20,        # reduce overfitting
        'subsample': 0.8,               # row sampling
        'colsample_bytree': 0.8,        # feature sampling
        'reg_lambda': 1.0,              # L2 regularization
        'random_state': 42,
    }
    # Always use early_stopping with this configuration!


##### PART 8 — HOW LIGHTGBM HANDLES SPECIAL CASES

### Missing Values

LightGBM handles missing values natively — you do NOT need to impute them.

During split finding, for each candidate split, LightGBM tries assigning 
the missing-value samples to both the left AND right child, then picks 
whichever direction produces higher gain. This learned direction is stored.

    Split on: age > 30

    Try 1: NaN → LEFT  child  → Gain = 0.42
    Try 2: NaN → RIGHT child  → Gain = 0.51  ← WINS

    Stored: "for this split, NaN goes RIGHT"

At inference, the stored direction is used for each split.


### Categorical Features

LightGBM can handle categorical features directly WITHOUT one-hot encoding.
Declare them as categorical and LightGBM uses an optimal split-finding 
algorithm for categories (grouping categories rather than creating 
binary splits).

    # Native categorical handling:
    model.fit(X_train, y_train, categorical_feature=['color', 'city'])

    This finds splits like: "city ∈ {NYC, LA, Chicago}" vs. rest
    Rather than: "is_NYC" vs. rest, "is_LA" vs. rest, etc.

    For k categories, naive search = O(2^k) possible groupings
    LightGBM uses an O(k log k) approximation based on gradient sorting.


### Class Imbalance

For highly imbalanced datasets:

    Option 1: is_unbalance=True
        Automatically adjusts weights so the positive class is weighted 
        by (n_negative / n_positive). Simple and often effective.

    Option 2: scale_pos_weight = n_negative / n_positive
        Manually set the weight of the positive class.

    Option 3: Use AUC or PR-AUC as metric instead of logloss
        These metrics are robust to class imbalance.


═══════════════════════════════════════════════════════════════════════════════
PART 9 — THE BIG PICTURE: ALGORITHM COMPARISON
═══════════════════════════════════════════════════════════════════════════════

### LightGBM vs. XGBoost vs. CatBoost vs. Random Forest

    ┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
    │ Property        │ LightGBM     │ XGBoost      │ CatBoost     │ RandomForest │
    ├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
    │ Speed           │ ★★★★★       │ ★★★★☆      │ ★★★☆☆       │ ★★★★☆      │
    │ Memory          │ ★★★★★       │ ★★★☆☆      │ ★★★★☆       │ ★★★☆☆      │
    │ Accuracy        │ ★★★★★       │ ★★★★★      │ ★★★★★       │ ★★★★☆      │
    │ Categorical     │ ★★★★☆       │ ★★☆☆☆      │ ★★★★★       │ ★★★☆☆      │
    │ Interpretability│ ★★★☆☆       │ ★★★☆☆      │ ★★★☆☆       │ ★★★★☆      │
    │ Overfit risk    │ Medium       │ Medium       │ Low          │ Low          │
    │ Tree growth     │ Leaf-wise    │ Level-wise   │ Level-wise   │ Level-wise   │
    │ Split finding   │ Histogram    │ Histogram    │ Ordered      │ Random       │
    │ Missing values  │ Native       │ Native       │ Native       │ Impute first │
    └─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

    WHEN TO CHOOSE LIGHTGBM:
    • Large datasets (100k+ rows) — speed advantage is most significant
    • Tabular data competitions — consistently top performer
    • Need fast iteration / hyperparameter search
    • Memory is constrained

    WHEN TO PREFER ALTERNATIVES:
    • CatBoost: heavy use of high-cardinality categoricals, smaller datasets
    • XGBoost: well-established, very slightly more conservative (level-wise)
    • Random Forest: need fast baseline, no hyperparameter tuning


    THE JOURNEY:
    ════════════════════════════════════════════════════════════

    1984  CART (Breiman et al.)            — Decision trees formalized
    1996  AdaBoost (Freund & Schapire)     — First boosting algorithm
    1999  Gradient Boosting (Friedman)     — Unified gradient framework
    2001  Random Forest (Breiman)          — Parallel ensemble
    2014  XGBoost (Chen & Guestrin)        — Scalable gradient boosting
    2016  LightGBM (Ke et al., Microsoft)  — GOSS + EFB + leaf-wise growth
    2017  CatBoost (Yandex)                — Ordered boosting for categoricals
    2019+ GPU-accelerated LightGBM         — GPU histogram computation

    Today, LightGBM, XGBoost, and CatBoost are the dominant algorithms 
    for structured/tabular data. Deep learning rarely beats them on 
    tabular tasks without architecture-specific inductive biases.
    """

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """

Time Complexity:
    Training:   O(M × B × d × n_GOSS)
                M  = number of trees (n_estimators)
                B  = number of histogram bins (max_bin, default 255)
                d  = number of features (reduced by EFB)
                n_GOSS = (a + b) × n, subset of rows via GOSS
    Prediction: O(M × log2(num_leaves)) — traversing M trees

Space Complexity:
    O(B × d) — storing histograms (independent of n! This is the key advantage)
    O(n × d) — storing the dataset itself
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Gradient Boosting From Scratch": {
        "description": "Manual implementation of gradient boosting to show the core algorithm without any library",
        "runnable": True,
        "pipeline_cmd": "lgbm",
        "code": '''
"""
================================================================================
GRADIENT BOOSTING FROM SCRATCH (MSE Regression)
================================================================================

This implements the core gradient boosting loop using only Python and a simple
decision stump (1-level tree) as the base learner. No scikit-learn, no LightGBM.

The goal is to show the CONCEPTUAL ENGINE that LightGBM runs underneath,
before the histogram tricks, GOSS, and EFB optimizations.

Algorithm:
    1. Initialize F₀ = mean(y)
    2. For m = 1..M:
        a. Compute residuals r = y - F (for MSE, this IS the negative gradient)
        b. Fit a decision stump to r
        c. Update: F = F + η × stump prediction
    3. Final prediction: sum of all contributions

================================================================================
"""

import math
import random


# =============================================================================
# DECISION STUMP (1-level decision tree)
# =============================================================================
# The "base learner" — the simplest tree possible.
# Finds the best single threshold on one feature that minimizes variance.
# In real LightGBM, this is a full decision tree with num_leaves leaves.

class DecisionStump:
    """
    A depth-1 decision tree (a single split).

    Finds the best feature + threshold to split the data,
    minimizing the residual variance in both halves.
    """

    def __init__(self):
        self.feature_idx = None   # Which feature to split on
        self.threshold   = None   # Split value: left if x <= threshold
        self.left_value  = None   # Prediction for left child (x <= threshold)
        self.right_value = None   # Prediction for right child (x > threshold)

    def fit(self, X, residuals):
        """
        Find the best split over all features and all thresholds.

        For each feature f and threshold t:
            left_samples  = {i : X[i][f] <= t}
            right_samples = {i : X[i][f] >  t}
            Gain = Variance(residuals) - weighted_var(left) - weighted_var(right)
        """

        n = len(X)
        best_gain = -float('inf')
        best_feature = 0
        best_threshold = 0
        best_left_val = 0
        best_right_val = 0

        def variance(values):
            if len(values) == 0:
                return 0
            mean_v = sum(values) / len(values)
            return sum((v - mean_v) ** 2 for v in values) / len(values)

        total_var = variance(residuals)

        # Try every feature
        for f in range(len(X[0])):
            # Collect unique values for this feature as candidate thresholds
            feature_values = sorted(set(X[i][f] for i in range(n)))

            for t_idx in range(len(feature_values) - 1):
                threshold = (feature_values[t_idx] + feature_values[t_idx + 1]) / 2

                left_res  = [residuals[i] for i in range(n) if X[i][f] <= threshold]
                right_res = [residuals[i] for i in range(n) if X[i][f] >  threshold]

                if len(left_res) == 0 or len(right_res) == 0:
                    continue

                # Weighted variance reduction
                gain = (total_var
                        - (len(left_res)  / n) * variance(left_res)
                        - (len(right_res) / n) * variance(right_res))

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = f
                    best_threshold = threshold
                    best_left_val  = sum(left_res)  / len(left_res)
                    best_right_val = sum(right_res) / len(right_res)

        self.feature_idx = best_feature
        self.threshold   = best_threshold
        self.left_value  = best_left_val
        self.right_value = best_right_val

    def predict(self, X):
        """Return leaf prediction for each sample."""
        preds = []
        for x in X:
            if x[self.feature_idx] <= self.threshold:
                preds.append(self.left_value)
            else:
                preds.append(self.right_value)
        return preds


# =============================================================================
# GRADIENT BOOSTING REGRESSOR
# =============================================================================

class GradientBoostingFromScratch:
    """
    Gradient Boosting for regression (MSE loss).

    Each tree is a DecisionStump fitted to the RESIDUALS of the current ensemble.
    For MSE, residuals == negative gradients, so this is true gradient boosting.

    The final model is:
        F(x) = F₀ + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)
    """

    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.F0            = None    # Initial prediction (mean of y)
        self.stumps        = []      # Fitted base learners
        self.loss_history  = []      # Track MSE per round

    def _mse(self, y_true, y_pred):
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def fit(self, X, y):
        n = len(y)

        # ─── INITIALIZATION ───────────────────────────────────────────────
        # F₀ = mean(y) — the best constant predictor for MSE
        self.F0 = sum(y) / n
        F = [self.F0] * n   # Current ensemble predictions

        print(f"  F₀ (initial prediction) = {self.F0:.4f}  (mean of y)")
        print(f"  Initial MSE = {self._mse(y, F):.4f}")
        print()

        # ─── BOOSTING LOOP ────────────────────────────────────────────────
        for m in range(self.n_estimators):

            # STEP 1: Compute pseudo-residuals (negative gradient for MSE)
            # r_i = y_i - F(x_i)    ← this IS -∂L/∂F for MSE loss
            residuals = [y[i] - F[i] for i in range(n)]

            # STEP 2: Fit a stump to the residuals
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.stumps.append(stump)

            # STEP 3: Get stump predictions
            stump_preds = stump.predict(X)

            # STEP 4: Update ensemble with shrinkage
            # F_m = F_{m-1} + η × h_m(x)
            F = [F[i] + self.learning_rate * stump_preds[i] for i in range(n)]

            # Track loss every 10 rounds
            mse = self._mse(y, F)
            self.loss_history.append(mse)

            if (m + 1) % 10 == 0 or m < 3:
                print(f"  Tree {m+1:>3} | Split: feature[{stump.feature_idx}] <= {stump.threshold:.2f}"
                      f" | Left={stump.left_value:+.4f}  Right={stump.right_value:+.4f}"
                      f" | MSE = {mse:.4f}")

        print()
        print(f"  ✓ Training complete. Final MSE = {self.loss_history[-1]:.4f}")

    def predict(self, X):
        """Sum contributions of all trees."""
        F = [self.F0] * len(X)
        for stump in self.stumps:
            preds = stump.predict(X)
            F = [F[i] + self.learning_rate * preds[i] for i in range(len(X))]
        return F


# =============================================================================
# DEMONSTRATION
# =============================================================================

random.seed(42)

# Generate a simple regression dataset: y = 2x + x² - 0.5 + noise
def generate_data(n=80):
    X = [[random.uniform(0, 3)] for _ in range(n)]
    y = [2 * x[0] + x[0] ** 2 - 0.5 + random.gauss(0, 0.5) for x in X]
    return X, y

X_train, y_train = generate_data(80)
X_test,  y_test  = generate_data(20)

print("=" * 65)
print("  GRADIENT BOOSTING FROM SCRATCH — REGRESSION DEMO")
print("=" * 65)
print(f"""
  Task: Fit y ≈ 2x + x² - 0.5 + noise  (non-linear regression)
  Base learner: Decision Stump (depth=1 tree)
  Loss: Mean Squared Error (MSE)
  Algorithm: residuals = y - F, fit stump to residuals, update F

  Why residuals = negative gradient for MSE?
    L(y, F) = (y - F)²/2
    ∂L/∂F  = F - y
    -∂L/∂F = y - F  = residuals  ✓
""")

print("  --- Training ---")
model = GradientBoostingFromScratch(n_estimators=50, learning_rate=0.2)
model.fit(X_train, y_train)

# Evaluate on test set
test_preds = model.predict(X_test)
test_mse = sum((y_test[i] - test_preds[i]) ** 2 for i in range(len(y_test))) / len(y_test)
test_rmse = test_mse ** 0.5

print()
print(f"  --- Test Results ---")
print(f"  Test MSE  = {test_mse:.4f}")
print(f"  Test RMSE = {test_rmse:.4f}")

print()
print(f"  --- Sample Predictions vs Actuals ---")
print(f"  {'x':<8} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
print(f"  {'-' * 45}")
for i in range(min(8, len(X_test))):
    x_val = X_test[i][0]
    actual = y_test[i]
    pred = test_preds[i]
    err = actual - pred
    print(f"  {x_val:<8.3f} {actual:<12.3f} {pred:<12.3f} {err:+.3f}")

print(f"""
  KEY OBSERVATIONS:
    1. Each tree fits RESIDUALS, not raw y values.
    2. The learning rate η shrinks each tree's contribution.
    3. MSE drops quickly at first, then more slowly — diminishing returns.
    4. More trees + smaller η = better generalization (try n=200, lr=0.05!)
    5. This is the core loop that LightGBM runs — with GOSS, histograms, 
       and full trees replacing stumps for much greater accuracy.
""")
''',
    },

    "LightGBM Scikit-Learn API": {
        "description": "Full LightGBM pipeline using the sklearn-compatible API with train/validation split and early stopping",
        "runnable": True,
        "pipeline_cmd": "lgbm",
        "code": '''
"""
================================================================================
LIGHTGBM: SKLEARN API — CLASSIFICATION AND REGRESSION
================================================================================

Demonstrates the full LightGBM workflow:
  1. Data loading and splitting
  2. Model creation with key hyperparameters
  3. Training with early stopping and validation monitoring
  4. Evaluation: AUC, classification report
  5. Feature importance (split and gain)
  6. Prediction interpretation

Uses the sklearn-compatible API (LGBMClassifier / LGBMRegressor).

================================================================================
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, mean_squared_error)
import lightgbm as lgb


# =============================================================================
# PART 1: BINARY CLASSIFICATION
# =============================================================================

print("=" * 65)
print("  PART 1: BINARY CLASSIFICATION")
print("=" * 65)

# ── Generate a synthetic classification dataset ──────────────────────────────
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,    # Only 10 of 20 features actually carry signal
    n_redundant=5,       # 5 features are linear combos of informative ones
    n_clusters_per_class=1,
    random_state=42
)

# ── Split: train / validation / test ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print()
print(f"  Dataset:  {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Train:    {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Class balance: {y.mean():.2%} positive")

# ── Define the model ─────────────────────────────────────────────────────────
clf = lgb.LGBMClassifier(
    # Core ensemble
    n_estimators=500,           # Max trees; early stopping will find optimal
    learning_rate=0.05,         # Low learning rate → more trees, better generalization

    # Tree structure
    num_leaves=31,              # KEY parameter for LightGBM (replaces max_depth)
    max_depth=-1,               # Unlimited depth, controlled via num_leaves
    min_child_samples=20,       # Min samples per leaf — prevents tiny, noisy leaves

    # Sampling (randomization → reduce overfitting)
    subsample=0.8,              # Use 80% of rows per tree (row subsampling)
    subsample_freq=1,           # Apply subsampling every tree
    colsample_bytree=0.8,       # Use 80% of features per tree

    # Regularization
    reg_lambda=1.0,             # L2 regularization on leaf weights
    reg_alpha=0.1,              # L1 regularization

    # Task
    objective='binary',
    metric='binary_logloss',

    random_state=42,
    verbose=-1,                 # Suppress LightGBM verbose output
)

# ── Train with early stopping ─────────────────────────────────────────────────
print()
print("  Training with early stopping (patience=50 rounds)...")

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50),    # Print metrics every 50 trees
    ]
)

print()
print(f"  Best iteration: {clf.best_iteration_} trees")

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred_proba = clf.predict_proba(X_test)[:, 1]   # Probability of class 1
y_pred       = clf.predict(X_test)

print()
print(f"  --- Test Results ---")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")

print()
print(f"  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

# ── Feature Importance ────────────────────────────────────────────────────────
print()
print("  --- Feature Importance ---")
print(f"  {'Feature':<12} {'Split Count':>14} {'Gain':>14}")
print(f"  {'-' * 42}")

split_importance = clf.booster_.feature_importance(importance_type='split')
gain_importance  = clf.booster_.feature_importance(importance_type='gain')

# Sort by gain importance
sorted_idx = np.argsort(gain_importance)[::-1]
for rank, i in enumerate(sorted_idx[:10]):   # Top 10
    bar = "▓" * int(gain_importance[i] / gain_importance.max() * 15)
    print(f"  feature_{i:<5}  splits: {split_importance[i]:>5}   gain: {gain_importance[i]:>8.1f}  {bar}")

print("""
  INTERPRETING FEATURE IMPORTANCE:
    • Split count: how many times a feature was used to split a node
    • Gain: total loss reduction attributed to this feature (more meaningful)
    • High split count + low gain → feature splits often but with small effect
    • Low split count + high gain → feature is very informative when used
""")


# =============================================================================
# PART 2: REGRESSION
# =============================================================================

print("=" * 65)
print("  PART 2: REGRESSION")
print("=" * 65)

X_r, y_r = make_regression(n_samples=3000, n_features=15, n_informative=8,
                            noise=0.5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
X_tr, X_v, y_tr, y_v   = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

reg = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective='regression',
    metric='rmse',
    random_state=42,
    verbose=-1,
)

reg.fit(X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)])

y_hat = reg.predict(X_te)
rmse  = np.sqrt(mean_squared_error(y_te, y_hat))
r2    = 1 - sum((y_te - y_hat) ** 2) / sum((y_te - y_te.mean()) ** 2)

print()
print(f"  Regression Results:")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}  (1.0 = perfect, 0.0 = predicts mean)")
print(f"  Best iteration: {reg.best_iteration_} trees")

print("""
  KEY TAKEAWAYS:
    1. Use early stopping instead of hand-tuning n_estimators.
    2. Keep learning_rate low (0.03-0.05) and let early stopping find optimal n.
    3. num_leaves (not max_depth) is LightGBM's main complexity parameter.
    4. Subsample + colsample_bytree add healthy stochasticity → reduce variance.
    5. Monitor validation metric to catch overfitting before it hurts test perf.
""")
''',
    },

    "LightGBM Native API (Train/Predict)": {
        "description": "LightGBM's native dataset API with Dataset objects, custom callbacks, and full training control",
        "runnable": True,
        "pipeline_cmd": "lgbm",
        "code": '''
"""
================================================================================
LIGHTGBM NATIVE API
================================================================================

The native LightGBM API (lgb.Dataset + lgb.train) offers more control than
the sklearn API, including:
  - Custom evaluation functions
  - Custom loss functions (first and second derivatives)
  - Verbose training logs and callbacks
  - CV-friendly training

================================================================================
"""

import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

# ── Data ─────────────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=8000, n_features=25,
                            n_informative=12, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# ── Create lgb.Dataset objects ────────────────────────────────────────────────
# lgb.Dataset is LightGBM's optimized internal data structure.
# It bins continuous features into histograms during creation (one-time cost).
# The reference= argument links val to train so feature binning is consistent.

dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, free_raw_data=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
params = {
    # Task
    "objective":        "binary",
    "metric":           "binary_logloss",

    # Ensemble
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        -1,

    # Sampling
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.8,

    # Regularization
    "min_child_samples": 20,
    "reg_lambda":        1.0,
    "reg_alpha":         0.1,

    # Speed
    "num_threads":      4,
    "verbosity":        -1,
    "seed":             42,
}

print("=" * 65)
print("  LIGHTGBM NATIVE API — BINARY CLASSIFICATION")
print("=" * 65)
print()
print(f"  Training data:    {X_train.shape}")
print(f"  Validation data:  {X_val.shape}")
print(f"  Test data:        {X_test.shape}")

# ── Custom Evaluation Function ────────────────────────────────────────────────
def custom_auc(preds, eval_data):
    """
    Custom eval function. LightGBM will call this at each boosting round.

    Args:
        preds:     raw model scores (before sigmoid for binary)
        eval_data: lgb.Dataset with true labels

    Returns:
        (eval_name, eval_result, is_higher_better)
    """
    labels = eval_data.get_label()
    # Convert raw scores to probabilities
    proba = 1 / (1 + np.exp(-preds))
    auc = roc_auc_score(labels, proba)
    return ("custom_auc", auc, True)   # True = higher is better


# ── Train ──────────────────────────────────────────────────────────────────────
print()
print("  Training with native API...")
evals_result = {}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=500,           # Max trees
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    feval=custom_auc,              # Use our custom AUC evaluator
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(evals_result),   # Save all metrics to dict
    ],
)

print()
print(f"  Best iteration: {model.best_iteration}")
print(f"  Best val logloss: {model.best_score['val']['binary_logloss']:.4f}")
print(f"  Best val AUC:     {model.best_score['val']['custom_auc']:.4f}")

# ── Predict ────────────────────────────────────────────────────────────────────
# num_iteration=model.best_iteration uses only the best trees (not all 500)
raw_scores = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_proba = 1 / (1 + np.exp(-raw_scores))   # Apply sigmoid manually
y_pred       = (y_pred_proba >= 0.5).astype(int)

test_auc = roc_auc_score(y_test, y_pred_proba)
print()
print(f"  Test AUC: {test_auc:.4f}")

# ── Cross-Validation ────────────────────────────────────────────────────────────
print()
print("  --- 5-Fold Cross-Validation ---")
cv_result = lgb.cv(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=300,
    nfold=5,
    stratified=True,
    shuffle=True,
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
    seed=42,
)

best_rounds = len(cv_result["valid binary_logloss-mean"])
best_logloss = min(cv_result["valid binary_logloss-mean"])
best_std     = cv_result["valid binary_logloss-stdv"][
    cv_result["valid binary_logloss-mean"].index(best_logloss)
]
print(f"  CV Best rounds:   {best_rounds}")
print(f"  CV Logloss:       {best_logloss:.4f} ± {best_std:.4f}")

print("""
  NATIVE API vs SKLEARN API:
  ─────────────────────────────────────────────────────────────
  Native (lgb.train):          Sklearn (LGBMClassifier):
  ✓ Custom loss/eval           ✓ Familiar .fit/.predict API
  ✓ More callback control      ✓ Works in sklearn Pipelines
  ✓ lgb.cv() built in          ✓ GridSearchCV compatible
  ✓ Access to raw scores       ✓ Less boilerplate

  Both ultimately call the same underlying C++ engine.
""")
''',
    },

    "Hyperparameter Tuning with Optuna": {
        "description": "Bayesian hyperparameter optimization using Optuna to find the best LightGBM configuration",
        "runnable": True,
        "pipeline_cmd": "lgbm",
        "code": '''
"""
================================================================================
LIGHTGBM HYPERPARAMETER TUNING WITH OPTUNA
================================================================================

Demonstrates Bayesian optimization with Optuna — a far better strategy than
grid search or random search for LightGBM's high-dimensional parameter space.

WHY OPTUNA OVER GRID/RANDOM SEARCH?
  - Grid search: exponential in number of params, tests bad combos
  - Random search: no learning from past trials
  - Optuna: learns which regions of parameter space are promising,
    uses Tree-structured Parzen Estimator (TPE) to focus future trials

STRATEGY:
  1. Define a parameter search space (ranges for each hyperparameter)
  2. Optuna proposes a trial (specific param values)
  3. Train LightGBM with those params + early stopping on validation set
  4. Return the validation AUC to Optuna
  5. Repeat for n_trials — Optuna builds a model of AUC vs. param space
  6. Use the best trial's params to train the final model

================================================================================
"""

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)   # Suppress Optuna logs

# ── Data ─────────────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=6000, n_features=20,
                            n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

dtrain = lgb.Dataset(X_train, label=y_train)
dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)


# ── Objective function ────────────────────────────────────────────────────────
def objective(trial):
    """
    The function Optuna minimizes (we return -AUC so minimization = maximizing AUC).

    trial.suggest_*() samples a value from the specified search space.
    Optuna will call this function n_trials times, intelligently exploring
    the parameter space based on past results.

    NOTE: lgb.Dataset is recreated inside each trial rather than shared
    globally. Reusing a single Dataset across trials where min_child_samples
    varies causes a LightGBMError because the feature pre-filter cache was
    built with a different min_data_in_leaf value.
    """

    params = {
        "objective":          "binary",
        "metric":             "binary_logloss",
        "verbosity":          -1,
        "seed":               42,
        "feature_pre_filter": False,   # required when min_data_in_leaf varies across trials

        # ── Parameters to tune ──────────────────────────────────────────
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # log=True → samples in log space: most trials near 0.01-0.05, fewer near 0.3

        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        # Main LightGBM complexity control. Higher → more complex, more overfit risk.

        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        # Hard depth limit. Use with num_leaves to bound leaf-wise growth.

        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # Min data in each leaf. Higher = smoother model.

        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":    trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),

        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    # Recreate Dataset each trial — avoids feature_pre_filter cache mismatch
    # when min_child_samples changes between trials
    trial_dtrain = lgb.Dataset(X_train, label=y_train)
    trial_dval   = lgb.Dataset(X_val,   label=y_val, reference=trial_dtrain)

    # Train with early stopping — avoids needing to tune n_estimators separately
    model = lgb.train(
        params,
        trial_dtrain,
        num_boost_round=300,
        valid_sets=[trial_dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    preds  = model.predict(X_val, num_iteration=model.best_iteration)
    auc    = roc_auc_score(y_val, preds)
    return -auc    # Optuna minimizes, so return -AUC


# ── Run optimization ──────────────────────────────────────────────────────────
print("=" * 65)
print("  OPTUNA HYPERPARAMETER SEARCH FOR LIGHTGBM")
print("=" * 65)

N_TRIALS = 30   # Increase to 100+ for production
print()
print(f"  Running {N_TRIALS} trials...")

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),   # Bayesian TPE sampler
    pruner=optuna.pruners.MedianPruner(),           # Prune unpromising trials early
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Results ────────────────────────────────────────────────────────────────────
print()
print(f"  --- Optimization Results ({N_TRIALS} trials) ---")
print(f"  Best AUC (validation): {-study.best_value:.4f}")
print()
print(f"  Best hyperparameters:")
for param, value in study.best_params.items():
    if isinstance(value, float):
        print(f"    {param:<25} {value:.6f}")
    else:
        print(f"    {param:<25} {value}")

# ── Train final model with best params ────────────────────────────────────────
print()
print(f"  Training final model with best params...")

best_params = {
    "objective":          "binary",
    "metric":             "binary_logloss",
    "verbosity":          -1,
    "seed":               42,
    "feature_pre_filter": False,
    **study.best_params,
}

final_model = lgb.train(
    best_params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_auc   = roc_auc_score(y_test, test_preds)
print(f"  Final Test AUC: {test_auc:.4f}")

# ── Parameter importance ──────────────────────────────────────────────────────
try:
    importance = optuna.importance.get_param_importances(study)
    print()
    print(f"  --- Parameter Importance (which params matter most?) ---")
    for param, imp in list(importance.items())[:8]:
        bar = "▓" * int(imp * 30)
        print(f"  {param:<25} {imp:.4f}  {bar}")
except Exception:
    pass   # Skip if Optuna version doesn't support this

print("""
  TUNING STRATEGY GUIDE:
  ─────────────────────────────────────────────────────────────
  Phase 1 — Fix learning_rate=0.05, tune structural params:
    → num_leaves, max_depth, min_child_samples
    These control the bias-variance tradeoff of individual trees.

  Phase 2 — Fix structure, tune regularization:
    → reg_alpha, reg_lambda, min_child_samples
    These control how aggressively the model is constrained.

  Phase 3 — Tune learning rate + use early stopping for n_estimators:
    → Lower learning_rate + higher n_estimators (with early stopping)
    Always the final fine-tuning step.

  Phase 4 — Tune sampling:
    → subsample, colsample_bytree
    Add randomization to reduce correlation between trees.
""")
''',
    },

    "SHAP Value Analysis": {
        "description": "Deep interpretability with SHAP values — understand per-sample and global feature contributions",
        "runnable": True,
        "pipeline_cmd": "lgbm",
        "code": '''
"""
================================================================================
SHAP VALUE ANALYSIS FOR LIGHTGBM
================================================================================

SHAP (SHapley Additive exPlanations) provides model-agnostic feature 
attributions based on cooperative game theory.

For every prediction, SHAP decomposes it as:
    prediction = base_value + Σⱼ SHAP_value(feature_j, this_sample)

Where:
    • base_value = average prediction over all training data (E[f(x)])
    • SHAP_value(j, i) = how much feature j pushed prediction for sample i
                         up (positive) or down (negative) from base_value

SHAP is preferred over split/gain importance because:
  ✓ Per-sample (local) explanations
  ✓ Signed values (positive = pushes up, negative = pushes down)
  ✓ Additive: the shap values sum exactly to (prediction - base_value)
  ✓ Consistent: if a feature has more impact, its SHAP value increases
  ✗ Split/gain importance can be misleading for correlated features

LightGBM computes exact TreeSHAP values in O(TLD²) time, 
where T=trees, L=leaves, D=depth — efficient and exact.

================================================================================
"""

import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ── Load a real dataset ───────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train a model ─────────────────────────────────────────────────────────────
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

# ── Compute SHAP values ────────────────────────────────────────────────────────
# LightGBM's native TreeSHAP: pred_contrib=True returns
# shape (n_samples, n_features + 1) — last column is the base value

raw_shap = model.predict(X_test, pred_contrib=True)
shap_values = raw_shap[:, :-1]     # (n_samples, n_features) — per-feature contributions
base_value  = raw_shap[0, -1]      # Scalar — same for all samples (log-odds scale)

print("=" * 65)
print("  SHAP VALUE ANALYSIS — BREAST CANCER DATASET")
print("=" * 65)
print()
print(f"  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Task: Predict malignant (1) vs benign (0)")
print()
print(f"  Base value (log-odds): {base_value:.4f}")
print(f"  (= average model output before seeing any feature values)")

# ── Global Feature Importance via SHAP ────────────────────────────────────────
# Mean absolute SHAP value = global importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx    = np.argsort(mean_abs_shap)[::-1]

print()
print(f"  --- Global SHAP Importance (Top 10 Features) ---")
print(f"  {'Feature':<35} {'Mean |SHAP|':>12}  Bar")
print(f"  {'-' * 65}")
for rank, i in enumerate(sorted_idx[:10]):
    bar = "▓" * int(mean_abs_shap[i] / mean_abs_shap[sorted_idx[0]] * 20)
    print(f"  {feature_names[i]:<35} {mean_abs_shap[i]:>10.4f}  {bar}")

# ── Local Explanation for a Single Sample ─────────────────────────────────────
sample_idx = 0
sample_shap     = shap_values[sample_idx]
sample_pred_raw = base_value + sample_shap.sum()

print()
print(f"  --- Local Explanation: Sample #{sample_idx} ---")
print(f"  True label:    {'Malignant' if y_test[sample_idx] == 1 else 'Benign'}")
print(f"  Predicted prob: {model.predict_proba(X_test[[sample_idx]])[0, 1]:.4f}")
print(f"  Base value:    {base_value:.4f}")
print(f"  Sum of SHAPs:  {sample_shap.sum():.4f}")
print(f"  Prediction:    {sample_pred_raw:.4f} (base + SHAP sum)")

print()
print(f"  Feature contributions (sorted by absolute impact):")
print(f"  {'Feature':<35} {'SHAP Value':>12}  Direction")
print(f"  {'-' * 65}")
sorted_local = np.argsort(np.abs(sample_shap))[::-1]
for i in sorted_local[:12]:
    direction = "→ INCREASES prediction" if sample_shap[i] > 0 else "→ DECREASES prediction"
    print(f"  {feature_names[i]:<35} {sample_shap[i]:>+10.4f}  {direction}")

# ── SHAP Verification (key property: additivity) ──────────────────────────────
print(f"""
  --- SHAP Additivity Verification ---
  base_value + SHAP_sum = {base_value:.4f} + {sample_shap.sum():.4f} = {sample_pred_raw:.4f}
  This MUST equal the raw model output (before sigmoid).

  Sigmoid({sample_pred_raw:.4f}) = {1 / (1 + np.exp(-sample_pred_raw)):.4f}
  Model predict_proba:           {model.predict_proba(X_test[[sample_idx]])[0, 1]:.4f}
  {"✓ Match!" if abs(1/(1+np.exp(-sample_pred_raw)) - model.predict_proba(X_test[[sample_idx]])[0, 1]) < 0.001 else "✗ Mismatch — check version"}
""")

print("""
  SHAP USE CASES:
  ─────────────────────────────────────────────────────────────
  ✓ Debugging: which features cause unexpected predictions?
  ✓ Feature selection: prune features with near-zero mean |SHAP|
  ✓ Model auditing: detect bias (e.g., race/gender has high SHAP)
  ✓ Customer-facing explanations: "your loan was denied because..."
  ✓ A/B testing: verify that model changes affect expected features

  LIBRARIES:
  • pip install shap     → shap.TreeExplainer(model) for full plots
  • LightGBM built-in:   model.predict(X, pred_contrib=True)  ← used above
""")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
# ─────────────────────────────────────────────────────────────────────────────
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
        main_script = None

    if "tok_step_status" not in st.session_state:
        st.session_state.tok_step_status = {}
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

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html = ""
    visual_height = 400
    # try:
    #     from ml.visuals.lightgbm import (
    #         LIGHTGBM_VISUAL_HTML,
    #         LIGHTGBM_VISUAL_HEIGHT,
    #     )
    #     visual_html = LIGHTGBM_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = LIGHTGBM_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[lightgbm_module.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": visual_html,
        "visual_height": visual_height,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }