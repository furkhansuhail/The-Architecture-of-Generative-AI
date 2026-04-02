"""
Intrinsic Models — Interpretability by Design
==============================================

The models in this folder do not need an external explanation tool.
They are their own explanation. This module covers the theory behind
the three most important families of intrinsically interpretable models:
linear models, decision trees, and rule lists.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Intrinsic Models — Interpretability by Design"
DISPLAY_NAME = "02 · Intrinsic Models"
ICON = "🔍"
SUBTITLE = "Linear Models, Decision Trees, and Rule Lists"


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
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What Does "Intrinsically Interpretable" Actually Mean?

A model is intrinsically interpretable if its structure, by itself, constitutes
a human-understandable explanation — no external tool required. You do not need
SHAP or LIME to explain a logistic regression. You read the coefficients.
You do not need GradCAM to understand a decision tree. You follow the branches.

This sounds simple, but it carries a precise technical meaning. Researchers
distinguish three levels of interpretability:

    SIMULATABILITY
    ──────────────
    A model is simulatable if a human can step through its entire computation
    for a given input by hand, in reasonable time.

    A linear model with 5 features is simulatable: you multiply, sum, compare.
    A neural network with 50 million parameters is not simulatable at all.
    A decision tree with depth 3 is simulatable.
    A decision tree with depth 20 is NOT — too many paths to trace.

    Simulatability depends on model SIZE, not just model type.


    DECOMPOSABILITY
    ───────────────
    A model is decomposable if each component (weight, node, rule) has a
    standalone explanation that makes sense to a human.

    Linear regression is decomposable: each weight wᵢ means "a one-unit
    increase in feature xᵢ changes the output by wᵢ, all else held equal."
    Each weight can be interpreted in isolation.

    A neural network's individual neuron weights cannot be decomposed this way
    — a single weight means nothing without the context of all other weights
    in all other layers.


    ALGORITHMIC TRANSPARENCY
    ────────────────────────
    A model is algorithmically transparent if you can understand HOW the
    learning algorithm works — what it is optimizing and why the result
    looks like it does.

    Linear regression is transparent: it minimizes sum of squared residuals.
    You know exactly what objective shaped the weights.
    Random forests aggregate 500 trees with random subsets — the aggregation
    process itself becomes hard to reason about.


    ┌────────────────────────────────────────────────────────────────────┐
    │               INTERPRETABILITY PROPERTIES BY MODEL                 │
    │                                                                    │
    │  Model                Simulatable   Decomposable   Transparent     │
    │  ───────────────────────────────────────────────────────────────   │
    │  Linear Regression    ✓ (small)     ✓              ✓               │
    │  Logistic Regression  ✓ (small)     ✓              ✓               │
    │  Decision Tree (sm.)  ✓             ✓              ✓               │
    │  Decision Tree (lg.)  ✗             ✓              ✓               │
    │  Rule List            ✓             ✓              ✓               │
    │  GAM                  ✗ (complex)   ✓              ✓               │
    │  Random Forest        ✗             ✗              ✗               │
    │  Neural Network       ✗             ✗              ✓               │
    └────────────────────────────────────────────────────────────────────┘

    Key insight: the same model TYPE can be interpretable or not
    depending on its SIZE. A decision tree with 3 levels is interpretable.
    A decision tree with 50 levels is an opaque black box — you cannot
    hold it in your head, even though it is technically "white box."

    The practical rule: if a human cannot reasonably hold the model in
    working memory, it is not interpretable regardless of its family.
    A good benchmark: depth ≤ 5 for trees, ≤ 10 features for linear
    models, ≤ 15 rules for rule lists.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART I: LINEAR MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Linear Model Family

Linear models share one fundamental structure: the output is a weighted sum
of the input features plus a bias term.

    ŷ = w₁x₁ + w₂x₂ + ⋯ + wₙxₙ + b
       = W · X + b

Depending on what you do with ŷ afterwards, you get different models:

    ŷ directly                 → Linear Regression (continuous output)
    sign(ŷ)                    → Perceptron (hard binary decision)
    σ(ŷ) = 1 / (1 + e⁻ŷ)      → Logistic Regression (probability)
    softmax(ŷ)                 → Multinomial Logistic Regression
    identity (ŷ) per class      → Linear Discriminant Analysis

The interpretability of ALL of them flows from the same source: the
coefficient vector W. Each wᵢ is the marginal effect of feature i on the
output, holding all other features fixed.


### Linear Regression — Deep Dive

**The Objective Function**

Linear regression finds the weight vector W that minimizes the Residual Sum
of Squares (RSS) over the training data:

    RSS(W) = Σᵢ (yᵢ - ŷᵢ)²
           = Σᵢ (yᵢ - (W · Xᵢ + b))²

In matrix form:    RSS = ‖y - XW‖²

This has a closed-form solution (the Normal Equation):

    W* = (XᵀX)⁻¹ Xᵀy

This is not just a formula — it has a beautiful geometric interpretation:
W* is the projection of y onto the column space of X. The fitted values ŷ
are the point in the span of the features that is closest (in Euclidean
distance) to the true target vector y.


**Geometric Interpretation of Coefficients**

Each coefficient wᵢ tells you the slope of the output surface along the
direction of feature i. Visualise it in 2D:

    OUTPUT
    ▲
    │            /
    │           /  ← slope = w₁
    │          /
    │         /
    │        /
    └────────────────────► FEATURE x₁

    • Positive wᵢ: output rises as xᵢ increases (positive slope)
    • Negative wᵢ: output falls as xᵢ increases (negative slope)
    • wᵢ = 0:      feature xᵢ has no effect (flat — held constant)
    • Large |wᵢ|:  feature xᵢ is sensitive / strongly influential
    • Small |wᵢ|:  feature xᵢ barely moves the needle

In n dimensions, the fitted surface is a hyperplane. The weight vector
W = [w₁, w₂, ..., wₙ] is the gradient of this hyperplane — it points in
the direction of steepest ascent. The bias b shifts the hyperplane up or
down without changing its orientation.


**The Critical Assumption: Everything Must Be Standardised**

⚠️  Raw coefficients are NOT directly comparable unless features are on the
same scale. If x₁ is income in dollars ($50,000) and x₂ is age in years
(35), then w₁ = 0.001 and w₂ = 2.5 does NOT mean age is 2500× more
important than income — it means a 1-dollar increase in income raises output
by 0.001, and a 1-year increase in age raises it by 2.5.

To compare feature importance, you must standardise features first:

    x̃ᵢ = (xᵢ - μᵢ) / σᵢ

After standardisation, coefficients represent "a one standard deviation
increase in xᵢ changes output by wᵢ standard deviations" — now directly
comparable across features.

    BEFORE STANDARDISATION (not comparable):
    ─────────────────────────────────────────────────────────────
    Feature         Raw value    Coefficient    Apparent effect
    income          $50,000      0.00003        1.50
    age_years       35           2.10           2.10
    debt_ratio      0.45         -8.20          -8.20
    ─────────────────────────────────────────────────────────────
    ← Is debt_ratio really the most important? Only because its
      scale (0-1) makes the coefficient look large.

    AFTER STANDARDISATION (comparable):
    ─────────────────────────────────────────────────────────────
    Feature         Std coef    Interpretation
    income          +0.52       +1 SD income → +0.52 SD output
    age_years       +0.18       +1 SD age    → +0.18 SD output
    debt_ratio      -0.61       +1 SD debt   → -0.61 SD output
    ─────────────────────────────────────────────────────────────
    ← Now we can see: debt_ratio is the most impactful feature.


**Regularisation — Controlling Complexity and Improving Interpretability**

When features are correlated or there are many of them, ordinary least squares
can produce wildly large coefficients that overfit the training data. Two
regularisation methods add a penalty to the objective that keeps weights small:

    RIDGE REGRESSION (L2 regularisation):
    ────────────────────────────────────
    Objective: RSS + λ Σᵢ wᵢ²

    Adds the sum of squared weights to the loss. Large weights are penalised
    quadratically. The result: all weights shrink towards zero, but NONE
    become exactly zero.

    Ridge is best when you believe all features are relevant, but their
    effects are small. It handles correlated features well because it
    distributes weight across the correlated group.

    Interpretability consequence: you still get n non-zero coefficients —
    every feature contributes something to the prediction.


    LASSO REGRESSION (L1 regularisation):
    ────────────────────────────────────
    Objective: RSS + λ Σᵢ |wᵢ|

    Adds the sum of absolute weights. The L1 penalty has a crucial geometric
    property: its "penalty surface" has sharp corners at the axes, which means
    the optimal solution tends to land EXACTLY at zero for many coefficients.

    Lasso performs automatic feature selection — it produces SPARSE models
    where many weights are exactly zero. Only the most important features
    survive.

    Interpretability consequence: Lasso gives you a simpler model.
    If you have 100 features but only 8 are truly relevant, Lasso will
    zero out the other 92 and give you a clean 8-feature explanation.

    WHY DOES L1 PRODUCE SPARSITY?
    ─────────────────────────────
    RSS contours are ellipses. The L1 penalty region is a diamond (in 2D).
    The optimal point is where the smallest ellipse touches the diamond.
    The diamond's corners lie exactly on the axes (where one weight = 0),
    and the ellipse almost always touches a corner first.

    L2's penalty region is a circle — no corners — so the ellipse typically
    touches it along the edge, not at a zero-weight point.

    # ============================================================= #
    **Diagram — Geometry of L1 vs L2 Regularisation:**

    L1 (Lasso) — Diamond constraint:    L2 (Ridge) — Circle constraint:

        w₂                                  w₂
        ↑                                   ↑
        |      ╱  ← RSS contour             |      ╱
        |    ╱  ╲                           |    ╱   ╲
        |  ╱     ╲           ◆ ← optimal   |  ╱      )  ← circle
        | ◆ optimal↗         hits corner   | (        )
        |╱  ←diamond                       | ╲      ╱
        ╲   constraint                     |  ╲    ╱
         ╲                                 |   ╲  ╱
          ╲                                |
    ─────────────────► w₁            ──────────────────► w₁

    The diamond's corner forces w₁ = 0 or w₂ = 0 at the optimum.
    The circle has no corners → both weights stay non-zero.
    # ============================================================= #

    ELASTIC NET:
    ────────────
    Objective: RSS + λ₁ Σ|wᵢ| + λ₂ Σwᵢ²

    Combines L1 and L2. Gets both sparsity (L1) and stability under
    correlated features (L2). A practical compromise for real datasets
    where both properties are desirable.


### Logistic Regression — The Probability Engine

For binary classification, we need outputs between 0 and 1. The solution:
wrap the linear combination in the logistic (sigmoid) function.

    P(y=1 | X) = σ(W · X + b) = 1 / (1 + e^(-(W·X + b)))

The sigmoid maps any real number to (0, 1), which we interpret as probability.
But the weights wᵢ no longer mean "unit increase changes output by wᵢ."
The relationship is now non-linear. To recover interpretability, we use
the log-odds transformation.

**Log-Odds and Odds Ratios**

Define the odds of y=1:

    odds = P(y=1) / P(y=0) = P(y=1) / (1 - P(y=1))

Take the natural log:

    log(odds) = log(P(y=1) / (1 - P(y=1))) = W · X + b

This is called the logit (log-odds). The linear model lives in log-odds space.
Now wᵢ has a clean interpretation again:

    A one-unit increase in xᵢ multiplies the ODDS by e^(wᵢ)

This quantity e^(wᵢ) is called the ODDS RATIO.

    # ============================================================= #
    **Concrete Example — Odds Ratios in a Loan Model:**

    Feature          Coefficient (w)   Odds Ratio (e^w)
    ─────────────────────────────────────────────────────
    income_std       +0.82             2.27   ← income 1 SD above mean
                                              → 2.27× more likely approved
    debt_ratio_std   -0.61             0.54   ← debt 1 SD above mean
                                              → 0.54× as likely (46% LESS)
    credit_score_std +1.14             3.13   ← score 1 SD above mean
                                              → 3.13× more likely approved
    bias             -0.30             —

    Reading the table:
      • OR > 1 → feature increases the odds of approval
      • OR < 1 → feature decreases the odds of approval
      • OR = 1 → feature has no effect (wᵢ = 0)
      • OR = 2 → twice as likely; OR = 0.5 → half as likely
    # ============================================================= #


**The Decision Boundary of Logistic Regression**

The decision boundary (P(y=1) = 0.5) is where log-odds = 0:

    W · X + b = 0    ←  this is still a LINEAR equation

So logistic regression, like linear regression, produces a LINEAR decision
boundary — a line in 2D, a plane in 3D, a hyperplane in nD. This is its
fundamental limitation: it can only separate classes with a straight surface.

    x₂
    ↑
    |   ● ●  |  ○ ○
    |  ● ●   |   ○
    |    ●   |  ○ ○
    |    ●   |   ○
    └────────────────────────→ x₁
              ↑
         decision boundary
         W·X + b = 0
    (only one straight line — linear boundary)

Logistic regression CANNOT solve XOR, or any problem where classes wrap
around each other, or any non-linearly separable distribution.


**Multicollinearity — The Hidden Trap**

When two features are highly correlated (e.g., income and wealth), their
coefficients become unstable and uninterpretable. The model needs to
distribute a single real signal across two correlated proxies, and any
split is equally valid statistically. Small changes in training data
produce completely different coefficient allocations.

    Scenario: true underlying driver is wealth.
    You measure income (corr. with wealth = 0.95) and assets (corr. = 0.93)

    Run 1: w_income = +1.4, w_assets = +0.2
    Run 2: w_income = +0.1, w_assets = +1.5  ← same predictive power!
    Run 3: w_income = +0.8, w_assets = +0.8

    You cannot interpret either coefficient individually.
    Diagnosis: compute Variance Inflation Factor (VIF) for each feature.
    VIF > 10 indicates severe multicollinearity.
    Fix: remove one correlated feature, or apply Ridge regularisation.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART II: DECISION TREES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Idea — Recursive Partitioning

A decision tree partitions the feature space into rectangular regions by
asking a sequence of YES/NO questions about individual features. Each
internal node tests one feature against one threshold. Each leaf assigns
a prediction (the majority class, or mean output for regression).

    # ============================================================= #
    **A Simple Decision Tree (Loan Approval):**

                  ┌───────────────────────────────┐
                  │   credit_score ≥ 650?         │
                  └──────────────┬────────────────┘
                       YES ↙          ↘ NO
              ┌──────────────┐    ┌──────────────────────┐
              │ income ≥ 40k?│    │  debt_ratio ≤ 0.40?  │
              └──────┬───────┘    └──────────┬───────────┘
                YES↙   ↘NO            YES↙      ↘NO
           ┌───────┐  ┌──────┐    ┌──────────┐  ┌──────────┐
           │APPROVE│  │DENY  │    │ APPROVE  │  │  DENY    │
           │ (87%) │  │(71%) │    │  (63%)   │  │  (89%)   │
           └───────┘  └──────┘    └──────────┘  └──────────┘

    Reading this tree for a specific applicant:
      credit_score = 720 → YES → income = $35k → NO → DENY (71% confident)
    # ============================================================= #

The depth of the tree is the length of the longest root-to-leaf path.
Depth = 2 in the example above (two questions for any path).


**Geometric View — Axis-Aligned Hyperplane Cuts**

Each split in a decision tree is a cut perpendicular to one feature axis.
The entire tree is a sequence of these cuts, carving the feature space
into rectangular "boxes" — one box per leaf.

    x₂ (income)
    ↑
    │                     │
    │ income < 40k        │ income ≥ 40k
    │ credit < 650        │ credit ≥ 650
    │ → DENY              │ → APPROVE
    ───────────────────────────────────────
    │ income < 40k        │ income ≥ 40k
    │ credit ≥ 650        │ credit < 650
    │ → DENY              │ → DENY
    └─────────────────────────────────────→ x₁ (credit score)
                          ↑
                        credit_score = 650 (vertical split)
    ──────────────────── income = 40k (horizontal split)

    Each leaf = one rectangular region = one prediction.

This explains a crucial limitation: decision trees cannot naturally learn
diagonal boundaries. A boundary like "x₁ + x₂ > 5" requires many small
axis-aligned rectangles to approximate. A linear model learns it exactly
with one coefficient. Trees pay a complexity tax for non-axis-aligned patterns.


### The Splitting Algorithm — CART (Classification and Regression Trees)

CART (Breiman et al., 1984) is the algorithm behind almost all modern
decision trees (sklearn, XGBoost, LightGBM). It builds trees by greedy,
top-down, recursive splitting.

**At each node, CART asks: which (feature, threshold) pair produces the
best split?**

"Best" is measured by an impurity function. The goal is to produce child
nodes that are as "pure" as possible — ideally, every leaf contains only
one class.


**Splitting Criterion 1 — Gini Impurity**

Gini impurity measures the probability that a randomly chosen element from
a node would be incorrectly classified if randomly labelled according to
the class distribution at that node.

    Gini(node) = 1 - Σₖ pₖ²

    where pₖ = fraction of samples in the node belonging to class k.

    Pure node (all one class):   p₁=1, p₂=0  → Gini = 1 - (1² + 0²) = 0
    Maximally impure (50/50):    p₁=0.5       → Gini = 1 - (0.5² + 0.5²) = 0.5

The weighted Gini after a split is:

    Gini_split = (n_left/n) × Gini(left) + (n_right/n) × Gini(right)

CART chooses the split that MINIMISES Gini_split (equivalently, that
MAXIMISES the Gini reduction = Gini_parent - Gini_split).


    # ============================================================= #
    **Worked Example — Gini Impurity Calculation:**

    Dataset at a node (8 samples):
      [●, ●, ●, ●, ○, ○, ○, ○]   (4 class 1, 4 class 0)

    Parent Gini:
      p₁ = 4/8 = 0.5,   p₀ = 4/8 = 0.5
      Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.50   ← maximally impure

    Candidate Split A:  [●●●●|○○○○]   (perfect split)
      Left:  p₁=1.0, p₀=0.0  → Gini = 1 - (1.0² + 0.0²) = 0.00
      Right: p₁=0.0, p₀=1.0  → Gini = 1 - (0.0² + 1.0²) = 0.00
      Weighted: (4/8)×0.00 + (4/8)×0.00 = 0.00
      Gini Reduction: 0.50 - 0.00 = 0.50 ✅ MAXIMUM

    Candidate Split B:  [●●●○|●○○○]   (imperfect split)
      Left (4):  p₁=3/4, p₀=1/4  → Gini = 1 - (0.75² + 0.25²) = 0.375
      Right (4): p₁=1/4, p₀=3/4  → Gini = 1 - (0.25² + 0.75²) = 0.375
      Weighted: (4/8)×0.375 + (4/8)×0.375 = 0.375
      Gini Reduction: 0.50 - 0.375 = 0.125 ← less than Split A

    Candidate Split C:  [●●●●●○|○○]   (lopsided)
      Left (6):  p₁=5/6, p₀=1/6  → Gini = 1 - (0.833²+0.167²) = 0.278
      Right (2): p₁=0/2, p₀=2/2  → Gini = 0.000
      Weighted: (6/8)×0.278 + (2/8)×0.000 = 0.208
      Gini Reduction: 0.50 - 0.208 = 0.292 ← decent but not perfect

    CART selects Split A. It maximises the Gini Reduction (0.50).
    # ============================================================= #


**Splitting Criterion 2 — Information Gain (Entropy)**

Entropy, from information theory, measures the average number of bits
needed to encode the class of a randomly drawn sample:

    Entropy(node) = - Σₖ pₖ log₂(pₖ)

    Pure node:    p₁=1         → H = -(1·log₂(1)) = 0 bits
    50/50 split:  p₁=p₂=0.5   → H = -(0.5·log₂(0.5) + 0.5·log₂(0.5))
                                   = -(0.5·(-1) + 0.5·(-1)) = 1 bit

Information Gain = reduction in entropy after the split:

    IG = Entropy(parent) - weighted average Entropy(children)

The ID3 and C4.5 algorithms use Information Gain. Entropy and Gini are
nearly identical in practice — both are concave functions peaked at the
uniform distribution. The main difference: entropy more strongly penalises
extreme class imbalances.

    # ============================================================= #
    **Gini vs Entropy — the mathematical shapes compared:**

                               Impurity
                               ↑
                          0.5  |     ╭──── Gini: 1 - p² - (1-p)²
                               |    ╭╯╮
                               |   ╭╯  ╮
                          0.25 |  ╭╯    ╮
                               | ╭╯   ← Entropy/2 (scaled)
                               |╭╯
                               └────────────────── p (class 1 fraction)
                              0     0.5     1.0

    Both peak at p = 0.5 (maximum uncertainty), touch 0 at p=0 and p=1
    (perfect purity). Entropy is slightly more curved — it more aggressively
    penalises impure nodes. In practice, they select the same split ~95% of
    the time. Gini is faster (no log calculation needed).
    # ============================================================= #


**Pruning — Managing Complexity**

Unpruned trees overfit: they memorise training data by creating leaves with
single samples. Pruning controls this.

    PRE-PRUNING (Early Stopping):
    ─────────────────────────────
    Stop growing the tree early by imposing hard constraints:
      • max_depth: maximum number of levels (typically 3–10)
      • min_samples_split: minimum samples needed to split a node
      • min_samples_leaf: minimum samples required in any leaf
      • min_impurity_decrease: only split if Gini reduction ≥ threshold

    Pre-pruning is fast but greedy — stopping early can miss useful splits
    that first require a "bad" split to reveal a very good one later.

    POST-PRUNING (Cost-Complexity Pruning):
    ─────────────────────────────────────────
    Grow the full tree, then prune backwards.

    The key idea: define a complexity penalty α (alpha) per leaf node.
    The cost of a subtree T is:

        Cost(T, α) = Error(T) + α × |leaves(T)|

    For each value of α, find the optimal subtree by collapsing leaves
    that don't pay for themselves. As α increases, the tree shrinks.
    Select α by cross-validation.

    This is theoretically superior to pre-pruning but computationally
    more expensive.

    # ============================================================= #
    **Visualising the Effect of max_depth:**

    max_depth = 1 (too simple)        max_depth = 3 (reasonable)
    ─────────────────────────         ─────────────────────────────
    Captures only one pattern.        Captures key interactions.
    High bias, low variance.          Balanced bias-variance.
    Easy to interpret (1 rule).       Still interpretable (8 leaves).

    max_depth = 10 (too complex)      max_depth = 20 (overfit)
    ─────────────────────────         ─────────────────────────────
    Memorises training data.          Pure memorisation.
    Low bias, very high variance.     Training acc ≈ 100%.
    Hard to interpret (1024 leaves).  Test acc << training acc.
    # ============================================================= #


**The Instability Problem**

Decision trees have a well-known pathology: they are high-variance learners.
A tiny change in the training data (one outlier, 5% of samples shuffled)
can produce a completely different tree structure.

    Why? The greedy, top-down splitting means the first split dominates
    everything. If the best first split changes slightly, the entire tree
    restructures. Different root = different everything downstream.

    Example:
    Training set A → root split: "income ≥ 50k?"
    Training set B (98% identical) → root split: "credit_score ≥ 640?"
    → completely different tree structure, different explanations

    This is why Random Forests work: they average 500 unstable trees,
    and the variance washes out. But each individual tree is volatile.

    Practical implication for interpretability: if you use a single
    decision tree for explanation, check its stability. Re-train on
    bootstrapped samples and verify that key splits are consistent.
    If the top-level split changes across bootstrap samples, the tree
    is telling you the model is uncertain, not certain.


**Why Trees Cannot Efficiently Learn Linear Relationships**

If the true decision boundary is "x₁ + x₂ > 5" (a diagonal line),
a decision tree must approximate it with a staircase of rectangular splits:

    x₂
    ↑
    │● ● ● ○○○      Each step in the staircase = one split.
    │● ● ○○○○       A diagonal boundary needs O(1/ε²) splits
    │● ●○○○○        to achieve ε accuracy — exponentially many.
    │● ○○○○○
    │○○○○○○
    └─────────────────→ x₁

    A logistic regression fits the true diagonal in 2 coefficients.
    A tree needs 100+ splits to match it, and will STILL not match
    perfectly — only approximate.

    The reverse also holds: linear models can't learn step functions
    efficiently. Trees excel where discrete thresholds are real
    (e.g., "credit score < 650 → fundamentally different risk tier").


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART III: RULE LISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What Are Rule Lists?

A rule list (also called a decision list) is an ordered sequence of
IF-THEN-ELSE conditions. Rules are tested top-to-bottom, and the
FIRST rule that matches determines the prediction. If no rule matches,
a default rule at the bottom fires.

    IF   credit_score ≥ 700 AND debt_ratio ≤ 0.30  THEN  APPROVE   (conf: 94%)
    ELSE IF  income ≥ 90k                           THEN  APPROVE   (conf: 89%)
    ELSE IF  credit_score < 550                     THEN  DENY      (conf: 92%)
    ELSE IF  debt_ratio > 0.55                      THEN  DENY      (conf: 87%)
    ELSE                                                  REVIEW    (conf: 61%)
    ↑
    default rule — catches everything not covered above

This is an ordered rule list. The ORDER matters: the first matching rule
wins. A sample with credit_score=720 AND income=$95k is classified by the
first rule (APPROVE), even though the second rule would also match.


**Rule Lists vs Rule Sets**

    RULE LISTS (ordered, decision lists):
    ──────────────────────────────────────
    • Order matters — rules are mutually exclusive by construction
      because the first match wins.
    • Simpler to interpret: "test rules top-to-bottom; stop when one matches"
    • Easy to modify: swap rule order to change priority
    • The model IS the list — no need to check all rules for one prediction
    • Examples: RIPPER, Bayesian Rule Lists, OneR

    RULE SETS (unordered):
    ──────────────────────
    • Order does NOT matter — each rule fires independently
    • Multiple rules can match simultaneously → need a voting or
      conflict-resolution scheme
    • Better for multi-class problems where a single example belongs to
      multiple categories
    • Harder to interpret when rules conflict
    • Examples: CN2, PART, association rule mining


**The Falling Rule List Structure**

A "falling rule list" (Wang, Rudin et al., 2015) is a special ordered list
where:

    1. The first few rules capture high-confidence "sure approvals" or
       "sure denials" (the easy cases at the extremes of the distribution)
    2. Later rules handle less certain, borderline cases
    3. The default rule handles genuine uncertainty

This creates a natural flow from certain to uncertain:

    FALLING RULE LIST for loan approval:
    ────────────────────────────────────────────────────────────────
    Rule 1:  IF credit ≥ 750 AND income ≥ 80k        → APPROVE (99%)
    Rule 2:  IF credit ≥ 700                         → APPROVE (91%)
    Rule 3:  IF credit < 550                         → DENY    (95%)
    Rule 4:  IF debt_ratio > 0.50 AND income < 30k   → DENY    (88%)
    Rule 5:  IF income ≥ 100k                        → APPROVE (82%)
    DEFAULT:                                         → REVIEW  (51%)
    ────────────────────────────────────────────────────────────────

    The confidence scores fall as you go down the list — hence "falling."
    Rules at the top are the most certain, cleanest decisions.


**Bayesian Rule Lists (BRL)**

Bayesian Rule Lists (Letham, Rudin et al., 2015) are a principled
approach to learning rule lists. Instead of greedily searching for the
best split (like CART), BRL frames the problem as Bayesian inference:

    Posterior ∝ Likelihood × Prior

    Prior encodes our preference for SHORT, SIMPLE rule lists.
    Likelihood measures how well the rules fit the training data.

    BRL uses a Markov Chain Monte Carlo (MCMC) sampler to explore the
    space of possible rule lists, balancing accuracy against simplicity.

The result: BRL doesn't just find A rule list — it finds the rule list
that best balances fit and complexity, according to a principled probabilistic
objective. The prior can encode:
    • Preference for fewer rules (shorter list)
    • Preference for rules with fewer conditions (simpler antecedents)
    • Preference for high-confidence rules

This is fundamentally different from greedy tree induction, which makes
locally optimal decisions that may not be globally optimal.


**The RIPPER Algorithm**

RIPPER (Repeated Incremental Pruning to Produce Error Reduction,
Cohen 1995) is a fast, practical algorithm for learning rule lists.

Step 1: For each class (in order of increasing frequency):
    a. Grow a rule by greedily adding conditions that improve accuracy
       on the positive examples (the class currently being learned)
    b. Prune the rule using a held-out pruning set to remove overfitting
    c. Add the rule to the list
    d. Remove all examples covered by this rule from the training set
    e. Repeat until error rate exceeds a threshold

Step 2: Optimise the rule list by replacing or revising each rule using
        a global optimisation pass.

    WHY RIPPER IS IMPORTANT:
    ─────────────────────────
    It's fast (near-linear in data size), handles noise robustly, and
    produces compact rule lists. It is still one of the best rule
    learners for structured tabular data.


**Rule Complexity and Cognitive Load**

Rules have a natural measure of complexity: their length (number of
conditions in the antecedent).

    Short rule (easy to remember, high priority):
    IF credit_score < 550 THEN DENY

    Long rule (harder to evaluate, likely overfitting):
    IF credit_score ∈ [551, 649] AND income ∈ [25k, 40k]
    AND debt_ratio > 0.38 AND employment_years < 3
    AND has_prior_default = True THEN DENY

    The second rule has 5 conditions. Each additional condition roughly
    halves the fraction of the training set it covers — a rule with 10
    conditions may cover only 0.1% of cases, making it unreliable and
    impossible to audit in production.

    Practical guideline: most practitioners cap rule antecedents at 3–4
    conditions, and total rule list length at 10–15 rules. Beyond this,
    cognitive load exceeds the benefit over a black-box model.

    RULE LENGTH COMPLEXITY:
    ────────────────────────────────────────────────────────────────────
    Conditions    Coverage (typical)    Human memory load    Recommended
    ────────────────────────────────────────────────────────────────────
    1             High (30–60%)         Very easy            ✓ ideal
    2             Medium (10–30%)       Easy                 ✓ ideal
    3             Lower (5–15%)         Moderate             ✓ ok
    4             Low (2–8%)            Hard                 ⚠️ borderline
    5+            Very low (< 3%)       Very hard            ✗ avoid
    ────────────────────────────────────────────────────────────────────


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART IV: THE HIERARCHY — CHOOSING THE RIGHT INTRINSIC MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Bias-Variance Framing

Every model in machine learning makes the same fundamental trade-off:

    HIGH BIAS (underfitting)               HIGH VARIANCE (overfitting)
    ────────────────────────               ──────────────────────────
    Model is too simple.                   Model is too complex.
    Doesn't capture true signal.           Captures noise as signal.
    Consistently wrong.                    Right on training data,
    Low training AND test error            wrong on new data.
    difference — both are high.            High train accuracy,
                                           low test accuracy.

    Linear models have HIGH BIAS (they assume linearity), LOW VARIANCE.
    Deep trees have LOW BIAS, but HIGH VARIANCE (unstable, overfit).
    Rule lists are typically LOW BIAS (can model any boolean function
    with enough rules), with CONTROLLABLE VARIANCE (via rule count limits).

The right model depends on the signal type and the need for trustworthy
explanations:

    ┌─────────────────────────────────────────────────────────────────────┐
    │              WHEN TO USE WHICH INTRINSIC MODEL                      │
    │                                                                     │
    │  True relationship is LINEAR:                                       │
    │  ─────────────────────────────                                      │
    │  → Linear/Logistic Regression                                       │
    │     Best accuracy, most reliable coefficients, odds ratios          │
    │     give clear stakeholder communication.                           │
    │                                                                     │
    │  True relationship has THRESHOLDS / STEP CHANGES:                   │
    │  ─────────────────────────────────────────────────                  │
    │  → Decision Tree                                                    │
    │     "Credit score below 650 is a hard cutoff" → tree splits there   │
    │     perfectly. Linear model can't capture hard thresholds.          │
    │                                                                     │
    │  Audience is NON-TECHNICAL / REGULATORY:                            │
    │  ─────────────────────────────────────────                          │
    │  → Rule List                                                        │
    │     "IF income < $30k AND no prior history → DENY" is a             │
    │     sentence anyone can read, audit, and challenge.                 │
    │     Judges, doctors, and loan officers operate this way naturally.  │
    │                                                                     │
    │  FAIRNESS AUDIT is required:                                        │
    │  ────────────────────────────                                       │
    │  → All three are appropriate; rule lists are easiest to audit.      │
    │     A rule list explicitly exposes every path to every decision.    │
    │     Bias detection: check if protected attributes appear in rules.  │
    │                                                                     │
    │  Need ACTIONABLE RECOURSE for affected individuals:                 │
    │  ──────────────────────────────────────────────────                 │
    │  → Rule List or Decision Tree                                       │
    │     "You were denied because credit_score < 650.                    │
    │     Raise it above 650 for approval." Direct recourse.              │
    └─────────────────────────────────────────────────────────────────────┘


### The Rashomon Set Revisited — A Practical Implication

The Rashomon Set (Breiman 2001; Semenova & Rudin 2019) is the set of all
models whose test accuracy is within ε of the best model found. For many
real-world datasets — especially structured tabular data — this set is
LARGE. Many different model structures achieve nearly identical predictive
accuracy.

The key implication: if you are within the Rashomon set of a black-box model,
you have already found an optimal model. There is no accuracy to be gained
by adding complexity.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  The Variable Importance Cloud (Semenova & Rudin, 2019)             │
    │                                                                     │
    │  If the Rashomon set is large, different models in it will assign   │
    │  VERY DIFFERENT importances to the same features. This means:       │
    │                                                                     │
    │  "Feature A is the most important for this model" does NOT mean     │
    │  "Feature A is the most important for the underlying phenomenon."   │
    │                                                                     │
    │  Multiple, equally good models may disagree on feature importance.  │
    │  This is a fundamental epistemic limitation — not a bug in any      │
    │  specific method.                                                   │
    │                                                                     │
    │  The only safe claim: "Feature A is important for THIS model."      │
    │  Not: "Feature A is important for the world."                       │
    └─────────────────────────────────────────────────────────────────────┘


### The Three Models Side-by-Side

    ══════════════════════════════════════════════════════════════════════
    Property                Linear Model        Decision Tree       Rule List
    ──────────────────────────────────────────────────────────────────────
    Explanation form        Coefficients        Tree paths          IF-THEN rules
    Decision boundary       Linear (flat)       Axis-aligned        Axis-aligned
                                                rectangles          (rectangular)
    Captures non-linear     No (unless          Yes                 Yes
    patterns                features added)
    Captures interactions   With feat.eng.      Yes (each path      Yes (multi-cond.
                                                is one)             antecedents)
    Feature scale needed    Yes                 No                  No
    Probabilistic output    Yes (logit)         Yes (leaf freq)     Yes (leaf conf)
    Stable to data change   Yes (for OLS)       No (high var)       Moderate
    Handles missing vals    With imputation     Built-in (surr.)    With default rule
    Regulatory suitability  High                High                Very high
    Audience                Technical           Mixed               Any
    Max interpretable       ~15 features        Depth ≤ 5           ≤ 15 rules
    size                                        (~32 leaves)        (≤ 4 cond each)
    Best for                Continuous          Threshold-          Non-technical
                            targets,            based decisions,    audiences,
                            linear signals      industry cutoffs    compliance
    ══════════════════════════════════════════════════════════════════════

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔═══════════════════╦══════════════════╦══════════════════╦══════════════════╗
    ║                   ║  Linear Model    ║  Decision Tree   ║  Rule List       ║
    ╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
    ║ Training          ║ O(n·d²) OLS      ║ O(n·d·log n)     ║ O(n·d·r) RIPPER  ║
    ║ complexity        ║ O(n·d·iters) GD  ║ per node         ║ r = rules        ║
    ╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
    ║ Prediction        ║ O(d) dot product ║ O(depth)         ║ O(r·c) r=rules   ║
    ║ complexity        ║                  ║                  ║ c=conditions     ║
    ╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
    ║ Space             ║ O(d) weights     ║ O(2^depth) nodes ║ O(r·c) rules     ║
    ╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
    ║ Explanation       ║ One number per   ║ One path per     ║ One rule per     ║
    ║ output            ║ feature (global) ║ prediction       ║ prediction       ║
    ╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
    ║ Regularisation    ║ L1, L2, ElasticN ║ Pruning, depth   ║ Rule count,      ║
    ║ mechanism         ║                  ║ limits           ║ condition count  ║
    ╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
    ║ Interpretable     ║ Yes (small d)    ║ Depth ≤ 5        ║ ≤ 15 rules       ║
    ║ regime            ║ d ≤ 15 features  ║ ≤ 32 leaves      ║ ≤ 4 cond/rule    ║
    ╚═══════════════════╩══════════════════╩══════════════════╩══════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ──────────────────────────────────────────────────────────────────────
    "Three Models, One Dataset — Explanations Compared": {
        "description": (
            "Trains all three intrinsic model types (logistic regression, decision tree, "
            "rule list) on the same loan-approval dataset from scratch — zero external "
            "dependencies. Then extracts and prints the EXPLANATION each model produces "
            "for (a) the overall model and (b) one specific denied applicant. "
            "Makes the structural difference between coefficient-based, path-based, "
            "and rule-based explanation concrete and side-by-side."
        ),
        "runnable": True,
        "pipeline_cmd": "intrinsic",
        "code": '''
"""
================================================================================
THREE INTRINSIC MODELS — ONE DATASET, THREE EXPLANATION STYLES
================================================================================

Same 400-sample loan dataset. Same question: "why was applicant #7 denied?"
Three completely different explanation structures.

Features (standardised for logistic regression):
  x0 = credit_score    (300–850)
  x1 = income          ($k/year)
  x2 = debt_ratio      (0–1)
  x3 = employment_yrs  (0–25)

True rule: APPROVED if (credit ≥ 650 AND debt ≤ 0.40) OR income ≥ 85k
================================================================================
"""

import random
import math


random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(n=400):
    X, y = [], []
    for _ in range(n):
        cs  = random.gauss(640, 80)     # credit score
        inc = random.gauss(55, 25)      # income $k
        dr  = random.gauss(0.38, 0.12)  # debt ratio
        emp = random.gauss(6, 4)        # employment years
        cs  = max(300, min(850, cs))
        inc = max(10,  min(150, inc))
        dr  = max(0.05, min(0.85, dr))
        emp = max(0,   min(25, emp))
        label = int(
            (cs >= 650 and dr <= 0.40) or inc >= 85
        )
        X.append([cs, inc, dr, emp])
        y.append(label)
    return X, y

X, y = make_dataset(400)
feat_names = ["credit_score", "income", "debt_ratio", "employment_yrs"]
n = len(X)

# Standardise for logistic regression
def standardise(X):
    d = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(d)]
    stds  = [math.sqrt(sum((X[i][j]-means[j])**2 for i in range(len(X)))/len(X))
             for j in range(d)]
    return [[( X[i][j] - means[j]) / (stds[j] + 1e-9)
             for j in range(d)] for i in range(len(X))], means, stds

Xs, means, stds = standardise(X)


# ─────────────────────────────────────────────────────────────────────────────
# APPLICANT #7 (denied by the true rule)
# ─────────────────────────────────────────────────────────────────────────────
# High income but terrible credit score and high debt ratio
applicant_raw = [580.0, 50.0, 0.52, 4.0]  # denied: credit<650 AND debt>0.40
applicant_std = [(applicant_raw[j] - means[j]) / (stds[j] + 1e-9)
                 for j in range(4)]
true_label = 0  # DENIED

print("=" * 70)
print("  DATASET SUMMARY")
print("=" * 70)
print(f"  Samples: {n}   |   Approvals: {sum(y)}   |   Denials: {n-sum(y)}")
print()
print("  Applicant #7 (the case we explain):")
for i, f in enumerate(feat_names):
    print(f"    {f:<18}: {applicant_raw[i]}")
print(f"  True decision: {'DENIED' if true_label==0 else 'APPROVED'}")


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 — LOGISTIC REGRESSION
# ═════════════════════════════════════════════════════════════════════════════

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-500, min(500, z))))

# Train with gradient ascent on log-likelihood
w = [0.0, 0.0, 0.0, 0.0]
b = 0.0
lr = 0.3
for _ in range(600):
    probs = [sigmoid(sum(w[j]*Xs[i][j] for j in range(4)) + b) for i in range(n)]
    for j in range(4):
        grad = sum((y[i] - probs[i]) * Xs[i][j] for i in range(n)) / n
        w[j] += lr * grad
    b += lr * sum(y[i] - probs[i] for i in range(n)) / n

preds_lr = [1 if sigmoid(sum(w[j]*Xs[i][j] for j in range(4)) + b) > 0.5
            else 0 for i in range(n)]
acc_lr = sum(preds_lr[i] == y[i] for i in range(n)) / n

# Predict applicant #7
z7 = sum(w[j] * applicant_std[j] for j in range(4)) + b
p7 = sigmoid(z7)

print()
print("=" * 70)
print("  MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)
print(f"  Training accuracy: {acc_lr:.1%}")
print()
print("  ── GLOBAL EXPLANATION — Standardised Coefficients ──────────────────")
print(f"  {'Feature':<18}  {'Coef (std)':>11}  {'Odds Ratio':>11}  {'Direction'}")
print(f"  {'-'*62}")
sorted_idx = sorted(range(4), key=lambda j: -abs(w[j]))
for j in sorted_idx:
    odds_ratio = math.exp(w[j])
    direction = "↑ increases approval" if w[j] > 0 else "↓ decreases approval"
    bar = ("+" if w[j] > 0 else "-") * min(20, int(abs(w[j]) * 5))
    print(f"  {feat_names[j]:<18}  {w[j]:>+11.4f}  {odds_ratio:>11.4f}  {bar}")
print(f"  bias             {b:>+11.4f}")
print()
print(f"  Reading: a 1 SD increase in credit_score multiplies approval odds")
print(f"  by e^({w[0]:.3f}) = {math.exp(w[0]):.2f}×")
print()
print("  ── LOCAL EXPLANATION — Applicant #7 ────────────────────────────────")
print(f"  P(approve | applicant #7) = sigmoid({z7:.4f}) = {p7:.4f}")
print(f"  Prediction: {'APPROVED' if p7 > 0.5 else 'DENIED'}")
print()
print("  Feature contribution breakdown (wᵢ × xᵢ_standardised):")
total_contrib = 0
for j in sorted_idx:
    contrib = w[j] * applicant_std[j]
    total_contrib += contrib
    sign = "↑" if contrib > 0 else "↓"
    print(f"    {feat_names[j]:<18}: {w[j]:+.3f} × {applicant_std[j]:+.3f} = "
          f"{contrib:+.4f}  {sign}")
print(f"    bias             :                     {b:+.4f}")
print(f"    ─────────────────────────────────────────────────")
print(f"    log-odds (z)     :                     {z7:+.4f}")
print(f"    P(approve)       :                     {p7:.4f}")
print()
print(f"  Interpretation:")
neg = [(feat_names[j], w[j]*applicant_std[j]) for j in range(4)
       if w[j]*applicant_std[j] < 0]
neg.sort(key=lambda x: x[1])
print(f"  The biggest NEGATIVE contributors to this denial:")
for name, c in neg:
    print(f"    • {name}: {c:+.4f} (pushed log-odds DOWN)")


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 — DECISION TREE (depth-limited CART, from scratch)
# ═════════════════════════════════════════════════════════════════════════════

def gini(labels):
    if not labels:
        return 0.0
    n = len(labels)
    p1 = sum(labels) / n
    return 1 - (p1**2 + (1-p1)**2)

def best_split(X_sub, y_sub, feat_indices):
    best_g, best_f, best_t = float("inf"), None, None
    n = len(y_sub)
    for fi in feat_indices:
        vals = sorted(set(X_sub[i][fi] for i in range(n)))
        thresholds = [(vals[k]+vals[k+1])/2 for k in range(len(vals)-1)]
        for t in thresholds:
            left_y  = [y_sub[i] for i in range(n) if X_sub[i][fi] <= t]
            right_y = [y_sub[i] for i in range(n) if X_sub[i][fi] >  t]
            if not left_y or not right_y:
                continue
            g = (len(left_y)/n)*gini(left_y) + (len(right_y)/n)*gini(right_y)
            if g < best_g:
                best_g, best_f, best_t = g, fi, t
    return best_f, best_t, best_g

def build_tree(X_sub, y_sub, depth=0, max_depth=3, min_samples=10):
    if (depth >= max_depth or len(y_sub) < min_samples
            or len(set(y_sub)) == 1):
        majority = int(sum(y_sub) / len(y_sub) >= 0.5)
        conf = sum(y_sub)/len(y_sub) if majority==1 else 1-sum(y_sub)/len(y_sub)
        return {"leaf": True, "pred": majority,
                "conf": conf, "n": len(y_sub)}
    fi, t, g = best_split(X_sub, y_sub, list(range(len(X_sub[0]))))
    if fi is None:
        majority = int(sum(y_sub) / len(y_sub) >= 0.5)
        conf = sum(y_sub)/len(y_sub) if majority==1 else 1-sum(y_sub)/len(y_sub)
        return {"leaf": True, "pred": majority,
                "conf": conf, "n": len(y_sub)}
    mask_l = [X_sub[i][fi] <= t for i in range(len(X_sub))]
    Xl = [X_sub[i] for i in range(len(X_sub)) if mask_l[i]]
    yl = [y_sub[i] for i in range(len(y_sub)) if mask_l[i]]
    Xr = [X_sub[i] for i in range(len(X_sub)) if not mask_l[i]]
    yr = [y_sub[i] for i in range(len(y_sub)) if not mask_l[i]]
    return {
        "leaf": False, "feature": fi, "threshold": t,
        "gini_reduction": gini(y_sub) - g,
        "n": len(y_sub),
        "left":  build_tree(Xl, yl, depth+1, max_depth, min_samples),
        "right": build_tree(Xr, yr, depth+1, max_depth, min_samples),
    }

def predict_tree(node, x):
    if node["leaf"]:
        return node["pred"], node["conf"]
    if x[node["feature"]] <= node["threshold"]:
        return predict_tree(node["left"], x)
    else:
        return predict_tree(node["right"], x)

def predict_tree_with_path(node, x, path=None):
    if path is None:
        path = []
    if node["leaf"]:
        return node["pred"], node["conf"], path
    fi, t = node["feature"], node["threshold"]
    if x[fi] <= t:
        cond = f"{feat_names[fi]} ≤ {t:.1f}"
        return predict_tree_with_path(node["left"], x, path + [("YES", cond)])
    else:
        cond = f"{feat_names[fi]} > {t:.1f}"
        return predict_tree_with_path(node["right"], x, path + [("NO→", cond)])

def tree_accuracy(tree, X, y):
    preds = [predict_tree(tree, X[i])[0] for i in range(len(X))]
    return sum(preds[i] == y[i] for i in range(len(X))) / len(X)

def print_tree(node, depth=0, prefix="Root"):
    indent = "    " * depth
    if node["leaf"]:
        label = "APPROVE" if node["pred"] == 1 else "DENY"
        print(f"{indent}[{prefix}] → {label} "
              f"(conf: {node['conf']:.0%}, n={node['n']})")
    else:
        fn = feat_names[node["feature"]]
        t  = node["threshold"]
        gr = node["gini_reduction"]
        print(f"{indent}[{prefix}] {fn} ≤ {t:.1f}?  "
              f"(Gini↓ {gr:.4f}, n={node['n']})")
        print_tree(node["left"],  depth+1, "YES")
        print_tree(node["right"], depth+1, " NO")

tree = build_tree(X, y, max_depth=3, min_samples=12)
acc_tree = tree_accuracy(tree, X, y)

print()
print("=" * 70)
print("  MODEL 2: DECISION TREE  (max_depth=3, CART — Gini criterion)")
print("=" * 70)
print(f"  Training accuracy: {acc_tree:.1%}")
print()
print("  ── GLOBAL EXPLANATION — Full Tree Structure ─────────────────────────")
print_tree(tree)
print()
print("  ── LOCAL EXPLANATION — Applicant #7 ────────────────────────────────")
pred7, conf7, path7 = predict_tree_with_path(tree, applicant_raw)
print(f"  Decision path (root → leaf):")
for step, (direction, condition) in enumerate(path7, 1):
    print(f"    Step {step}: {condition}  [{direction}]")
print(f"  Leaf prediction: {'APPROVED' if pred7==1 else 'DENIED'} "
      f"(confidence: {conf7:.0%})")
print()
print(f"  Interpretation in plain English:")
conds = [cond for _, cond in path7]
print(f"  Applicant #7 was DENIED because: {' AND '.join(conds)}")


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3 — RULE LIST (Greedy RIPPER-style, from scratch)
# ═════════════════════════════════════════════════════════════════════════════

# Generate candidate conditions (feature, direction, threshold)
def candidate_conditions(X, feat_names):
    conds = []
    d = len(X[0])
    for fi in range(d):
        vals = sorted(set(X[i][fi] for i in range(len(X))))
        thresholds = [(vals[k]+vals[k+1])/2 for k in range(len(vals)-1)]
        # Use quartile thresholds to keep it manageable
        q_idx = [int(len(thresholds)*q) for q in [0.25, 0.50, 0.75]]
        for qi in q_idx:
            if qi < len(thresholds):
                t = thresholds[qi]
                conds.append((fi, "<=", t))
                conds.append((fi, ">",  t))
    return conds

def condition_matches(x, fi, op, t):
    return x[fi] <= t if op == "<=" else x[fi] > t

def evaluate_condition(X_sub, y_sub, fi, op, t):
    """Coverage and precision of condition on current data."""
    covered = [i for i in range(len(X_sub))
               if condition_matches(X_sub[i], fi, op, t)]
    if not covered:
        return 0, 0, 0
    positives = sum(y_sub[i] for i in covered)
    precision = positives / len(covered)
    coverage  = len(covered) / len(X_sub)
    return len(covered), precision, coverage

def learn_one_rule(X_sub, y_sub, target_class, all_conds,
                   max_conditions=3, min_coverage=0.05):
    """
    Greedily add conditions that maximise FOIL gain (information gain
    variant used by RIPPER) for the target class.
    FOIL gain = p · [log2(p/(p+n)) - log2(p0/(p0+n0))]
    where p = positives covered, n = negatives covered, p0/n0 = before split.
    """
    rule_conds = []
    available_conds = list(all_conds)
    current_X, current_y = X_sub[:], y_sub[:]

    for _ in range(max_conditions):
        best_score, best_cond = -1, None
        p0 = sum(1 for yv in current_y if yv == target_class)
        n0 = len(current_y) - p0
        if p0 == 0:
            break

        for (fi, op, t) in available_conds:
            covered_idx = [i for i in range(len(current_X))
                           if condition_matches(current_X[i], fi, op, t)]
            if not covered_idx:
                continue
            p = sum(1 for i in covered_idx if current_y[i] == target_class)
            n = len(covered_idx) - p
            if p == 0:
                continue
            cov_rate = len(covered_idx) / len(current_X)
            if cov_rate < min_coverage:
                continue
            # FOIL gain
            gain = p * (math.log2(p/(p+n+1e-9)) - math.log2(p0/(p0+n0+1e-9)))
            if gain > best_score:
                best_score, best_cond = gain, (fi, op, t)

        if best_cond is None:
            break

        rule_conds.append(best_cond)
        available_conds = [c for c in available_conds if c != best_cond]
        # Keep only examples covered by this condition
        covered_mask = [condition_matches(current_X[i],
                                          best_cond[0], best_cond[1], best_cond[2])
                        for i in range(len(current_X))]
        current_X = [current_X[i] for i in range(len(current_X)) if covered_mask[i]]
        current_y = [current_y[i] for i in range(len(current_y)) if covered_mask[i]]

    if not rule_conds:
        return None

    # Compute rule statistics on original data
    covered_idx = [i for i in range(len(X_sub)) if all(
        condition_matches(X_sub[i], fi, op, t) for fi, op, t in rule_conds)]
    if not covered_idx:
        return None
    precision = sum(y_sub[i] == target_class for i in covered_idx) / len(covered_idx)
    return rule_conds, precision, len(covered_idx)

def format_rule(rule_conds, target_class, precision, n_covered, feat_names):
    parts = []
    for fi, op, t in rule_conds:
        parts.append(f"{feat_names[fi]} {op} {t:.1f}")
    antecedent = "  AND  ".join(parts)
    label = "APPROVE" if target_class == 1 else "DENY"
    return f"IF  {antecedent}  →  {label}  (conf: {precision:.0%}, covers: {n_covered})"

def learn_rule_list(X, y, max_rules=6):
    """
    Simple greedy RIPPER-style rule learner.
    Alternate between learning APPROVE rules and DENY rules until
    error rate exceeds threshold or max_rules reached.
    """
    all_conds = candidate_conditions(X, feat_names)
    rules = []   # list of (rule_conds, target_class, precision)
    remaining_X, remaining_y = X[:], y[:]

    for rule_idx in range(max_rules):
        if len(remaining_X) < 20:
            break
        # Alternate target class: APPROVE first (typically minority)
        # Find current class distribution
        pos = sum(remaining_y)
        neg = len(remaining_y) - pos
        # Target the minority class (generally produces tighter rules)
        target = 1 if pos <= neg else 0
        # Try both; pick the one with better FOIL gain
        best_rule = None
        for tc in [1, 0]:
            result = learn_one_rule(remaining_X, remaining_y, tc,
                                    all_conds, max_conditions=2)
            if result is not None:
                conds, prec, ncov = result
                if prec >= 0.65 and ncov >= 10:
                    if best_rule is None or prec > best_rule[1]:
                        best_rule = (conds, prec, ncov, tc)

        if best_rule is None:
            break

        rule_conds, prec, ncov, tc = best_rule
        rules.append((rule_conds, tc, prec, ncov))

        # Remove covered examples from remaining pool
        mask = [not all(condition_matches(remaining_X[i], fi, op, t)
                        for fi, op, t in rule_conds)
                for i in range(len(remaining_X))]
        remaining_X = [remaining_X[i] for i in range(len(remaining_X)) if mask[i]]
        remaining_y = [remaining_y[i] for i in range(len(remaining_y)) if mask[i]]

    # Default rule: majority of remaining
    default_pred  = int(sum(remaining_y) / (len(remaining_y) + 1e-9) >= 0.5)
    default_conf  = (sum(remaining_y) / len(remaining_y)
                     if default_pred == 1
                     else 1 - sum(remaining_y) / len(remaining_y)
                     if remaining_y else 0.5)
    return rules, default_pred, default_conf

def predict_rule_list(rules, default_pred, x):
    for rule_conds, pred, prec, _ in rules:
        if all(condition_matches(x, fi, op, t) for fi, op, t in rule_conds):
            return pred, prec, rule_conds
    return default_pred, None, None  # default rule

rules, default_pred, default_conf = learn_rule_list(X, y, max_rules=7)

preds_rl = [predict_rule_list(rules, default_pred, X[i])[0] for i in range(n)]
acc_rl   = sum(preds_rl[i] == y[i] for i in range(n)) / n

print()
print("=" * 70)
print("  MODEL 3: RULE LIST  (greedy RIPPER-style)")
print("=" * 70)
print(f"  Training accuracy: {acc_rl:.1%}")
print()
print("  ── GLOBAL EXPLANATION — The Full Rule List ──────────────────────────")
for idx, (rule_conds, tc, prec, ncov) in enumerate(rules, 1):
    rule_str = format_rule(rule_conds, tc, prec, ncov, feat_names)
    print(f"  Rule {idx}:  {rule_str}")
default_label = "APPROVE" if default_pred == 1 else "DENY"
print(f"  DEFAULT: (no rule matched) → {default_label}  "
      f"(conf: {default_conf:.0%})")
print()
print("  ── LOCAL EXPLANATION — Applicant #7 ────────────────────────────────")
pred7_rl, conf7_rl, matched_conds = predict_rule_list(rules, default_pred,
                                                       applicant_raw)
print(f"  Testing rules top-to-bottom:")
fired = False
for idx, (rule_conds, tc, prec, ncov) in enumerate(rules, 1):
    matches = all(condition_matches(applicant_raw, fi, op, t)
                  for fi, op, t in rule_conds)
    status = "✓ MATCHED — STOP HERE" if matches else "✗ no match"
    cond_str = " AND ".join(f"{feat_names[fi]}{op}{t:.1f}"
                             for fi, op, t in rule_conds)
    print(f"    Rule {idx}: [{cond_str}]  →  {status}")
    if matches:
        label = "APPROVE" if tc == 1 else "DENY"
        print(f"    Decision: {label} (confidence: {prec:.0%})")
        fired = True
        break
if not fired:
    print(f"    (No rule matched) → DEFAULT: {default_label}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  COMPARISON SUMMARY — SAME DENIAL, THREE EXPLANATIONS")
print("=" * 70)
print()
print(f"  Applicant #7:  credit={applicant_raw[0]:.0f}, income=${applicant_raw[1]:.0f}k, "
      f"debt={applicant_raw[2]:.2f}, emp={applicant_raw[3]:.0f}yrs")
print()
print(f"  ┌────────────────────────────────────────────────────────────────┐")
print(f"  │ Model                Prediction   Accuracy   Explanation type │")
print(f"  ├────────────────────────────────────────────────────────────────┤")
lr_label = "DENIED" if p7 < 0.5 else "APPROVED"
dt_label = "DENIED" if pred7 == 0 else "APPROVED"
rl_label = "DENIED" if pred7_rl == 0 else "APPROVED"
print(f"  │ Logistic Regression  {lr_label:<12} {acc_lr:.1%}      Coefficients    │")
print(f"  │ Decision Tree        {dt_label:<12} {acc_tree:.1%}      Tree path       │")
print(f"  │ Rule List            {rl_label:<12} {acc_rl:.1%}      IF-THEN rule    │")
print(f"  └────────────────────────────────────────────────────────────────┘")
print()
print("  LOGISTIC REGRESSION says:")
top_neg = sorted([(feat_names[j], w[j]*applicant_std[j]) for j in range(4)],
                 key=lambda x: x[1])[:2]
for name, c in top_neg:
    print(f"    '{name}' contributed {c:+.3f} to log-odds (negative = pushes toward denial)")
print()
print("  DECISION TREE says:")
print(f"    Path: {' → '.join(cond for _, cond in path7)}")
print()
print("  RULE LIST says:")
if matched_conds:
    print(f"    Matched rule: " +
          " AND ".join(f"{feat_names[fi]}{op}{t:.1f}"
                       for fi, op, t in matched_conds))
else:
    print(f"    No rule matched → default prediction")
print()
print("  KEY OBSERVATION:")
print("  All three reach the same decision but for structurally different reasons.")
print("  The regression gives a score. The tree gives a path. The rule list gives")
print("  a sentence. Choosing between them is a choice about WHO reads the explanation.")
''',
    },

    # ── 2 ──────────────────────────────────────────────────────────────────────
    "Splitting Criteria Deep Dive — Gini vs Entropy vs Variance": {
        "description": (
            "An in-depth exploration of all three CART splitting criteria: Gini impurity "
            "(classification), Information Gain / Entropy (classification), and Variance "
            "Reduction (regression trees). Derives each formula from first principles, "
            "shows concrete step-by-step calculations, compares them on identical data, "
            "and explains when they produce different splits."
        ),
        "runnable": True,
        "pipeline_cmd": "intrinsic",
        "code": '''
"""
================================================================================
SPLITTING CRITERIA — GINI, ENTROPY, AND VARIANCE REDUCTION
================================================================================

Decision trees live or die by their splitting criterion. This script derives
and computes all three from scratch, side by side, on the same data, so you
can see exactly how each one measures "goodness" of a split.

The three criteria:
  1. GINI IMPURITY        — used by CART (sklearn default for classification)
  2. INFORMATION GAIN     — used by ID3 and C4.5
  3. VARIANCE REDUCTION   — used by CART for regression trees

All three answer the same question:
  "Does this split produce children that are more predictable than the parent?"
They just measure "predictable" differently.
================================================================================
"""

import math


# ─────────────────────────────────────────────────────────────────────────────
# PART A: MATHEMATICAL DEFINITIONS FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  PART A: MATHEMATICAL DEFINITIONS")
print("=" * 70)

print("""
  Given a node containing n samples with class distribution [p₁, p₂, ..., pₖ]:
  (For binary classification: p₁ = fraction of class 1, p₀ = 1 - p₁)

  ─────────────────────────────────────────────────────────────────────
  1. GINI IMPURITY
  ─────────────────────────────────────────────────────────────────────
  Gini(node) = 1 - Σₖ pₖ²

  Intuition: probability that two randomly drawn samples from the node
  are from DIFFERENT classes. Zero when pure; 0.5 maximum for binary.

  Derivation:
    P(both same class) = Σₖ pₖ² = p₁² + p₂² + ...
    P(different class) = 1 - Σₖ pₖ²

  Binary case:
    p₁ = p,   p₀ = 1-p
    Gini = 1 - p² - (1-p)² = 2p(1-p)

  Properties:
    • Range: [0, 0.5] for binary, [0, 1-1/k] for k classes
    • Zero iff pure (all one class)
    • Maximum at uniform distribution
    • Computationally cheap: no log function needed

  ─────────────────────────────────────────────────────────────────────
  2. ENTROPY (Information Gain)
  ─────────────────────────────────────────────────────────────────────
  Entropy(node) = -Σₖ pₖ log₂(pₖ)    (by convention, 0·log₂(0) = 0)

  Intuition: average number of bits needed to encode the class of a
  randomly drawn sample, given the class distribution at this node.
  If a node is pure, you already know the class — 0 bits needed.
  If 50/50, you need exactly 1 bit (one yes/no question).

  Information Gain of a split:
    IG = Entropy(parent) - Σ (nᵢ/n) × Entropy(childᵢ)

  Properties:
    • Range: [0, 1] for binary (or [0, log₂(k)] for k classes)
    • Zero iff pure
    • Maximum at uniform: -0.5·log₂(0.5) - 0.5·log₂(0.5) = 1 bit
    • More curved than Gini → penalises impurity more aggressively

  ─────────────────────────────────────────────────────────────────────
  3. VARIANCE REDUCTION (Regression Trees)
  ─────────────────────────────────────────────────────────────────────
  Variance(node) = (1/n) Σᵢ (yᵢ - ȳ)²   where ȳ = mean of y in node

  Variance Reduction of a split:
    VR = Var(parent) - Σ (nᵢ/n) × Var(childᵢ)

  Intuition: how much does the split reduce the spread of target values?
  A perfect split separates high-y examples from low-y examples completely.

  Properties:
    • Works only for continuous targets (regression, not classification)
    • Equivalent to choosing the split that maximises R² of the partition
    • The leaf prediction is the mean of y values in each leaf
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART B: IMPLEMENT ALL THREE FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

def gini_impurity(labels):
    """Gini impurity for a list of binary class labels."""
    if not labels:
        return 0.0
    n = len(labels)
    p1 = sum(labels) / n
    p0 = 1 - p1
    return 1 - (p1**2 + p0**2)

def entropy(labels):
    """Shannon entropy for a list of binary class labels."""
    if not labels:
        return 0.0
    n = len(labels)
    p1 = sum(labels) / n
    p0 = 1 - p1
    h = 0.0
    if p1 > 0:
        h -= p1 * math.log2(p1)
    if p0 > 0:
        h -= p0 * math.log2(p0)
    return h

def variance(values):
    """Variance of a list of continuous values."""
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean)**2 for v in values) / n

def gini_split(left_y, right_y):
    n = len(left_y) + len(right_y)
    return (len(left_y)/n)*gini_impurity(left_y) + (len(right_y)/n)*gini_impurity(right_y)

def entropy_split(left_y, right_y):
    n = len(left_y) + len(right_y)
    return (len(left_y)/n)*entropy(left_y) + (len(right_y)/n)*entropy(right_y)

def variance_split(left_y, right_y):
    n = len(left_y) + len(right_y)
    return (len(left_y)/n)*variance(left_y) + (len(right_y)/n)*variance(right_y)


# ─────────────────────────────────────────────────────────────────────────────
# PART C: WORKED EXAMPLE — ONE NODE, THREE CRITERIA
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  PART C: WORKED EXAMPLE — ONE NODE, THREE CANDIDATE SPLITS")
print("=" * 70)

# 12-sample node: 8 class-1 (APPROVE), 4 class-0 (DENY)
# One continuous feature (credit score)
# True split: credit ≥ 650

credit_scores = [580, 610, 625, 640, 645, 650, 660, 680, 700, 720, 750, 790]
class_labels  = [  0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1]
income_values = [ 32,  38,  45,  41,  58,  63,  72,  68,  80,  90,  95, 110]

print()
print("  Node data (12 samples, sorted by credit score):")
print(f"  {'Credit':>8}  {'Class':>6}  {'Income':>8}")
print(f"  {'-'*28}")
for i in range(len(credit_scores)):
    print(f"  {credit_scores[i]:>8}  {'APPROVE' if class_labels[i]==1 else 'DENY':>6}  "
          f"${income_values[i]:>6}k")

print()
parent_gini    = gini_impurity(class_labels)
parent_entropy = entropy(class_labels)
parent_var_inc = variance(income_values)

print(f"  Parent node statistics:")
print(f"    N = {len(class_labels)},  "
      f"APPROVE: {sum(class_labels)},  "
      f"DENY: {len(class_labels)-sum(class_labels)}")
print(f"    p(APPROVE) = {sum(class_labels)/len(class_labels):.4f}")
print(f"    Gini impurity:    {parent_gini:.6f}")
print(f"    Entropy:          {parent_entropy:.6f} bits")
print(f"    Income variance:  {parent_var_inc:.4f} (for regression demo)")


# Three candidate split thresholds
candidates = [
    ("Split A: credit ≤ 637  (4|8)",   637),
    ("Split B: credit ≤ 647  (5|7)",   647),
    ("Split C: credit ≤ 670  (7|5)",   670),
]

print()
print(f"  {'Split':<35}  {'Gini Δ':>8}  {'IG (bits)':>10}  {'Income VR':>10}  {'Best?'}")
print(f"  {'-'*75}")

results = []
for label, thresh in candidates:
    left_cl  = [class_labels[i]  for i in range(12) if credit_scores[i] <= thresh]
    right_cl = [class_labels[i]  for i in range(12) if credit_scores[i] >  thresh]
    left_inc = [income_values[i] for i in range(12) if credit_scores[i] <= thresh]
    right_inc= [income_values[i] for i in range(12) if credit_scores[i] >  thresh]

    g_after  = gini_split(left_cl, right_cl)
    e_after  = entropy_split(left_cl, right_cl)
    v_after  = variance_split(left_inc, right_inc)

    g_delta  = parent_gini    - g_after
    ig       = parent_entropy - e_after
    vr       = parent_var_inc - v_after

    results.append((label, thresh, g_delta, ig, vr, left_cl, right_cl,
                    left_inc, right_inc))

# Find best for each criterion
best_g = max(results, key=lambda r: r[2])
best_ig = max(results, key=lambda r: r[3])
best_vr = max(results, key=lambda r: r[4])

for r in results:
    label, thresh, g_delta, ig, vr = r[:5]
    is_best_g  = "G✓" if r is best_g  else "  "
    is_best_ig = "I✓" if r is best_ig else "  "
    is_best_vr = "V✓" if r is best_vr else "  "
    best_marks = f"{is_best_g}{is_best_ig}{is_best_vr}"
    print(f"  {label:<35}  {g_delta:>8.6f}  {ig:>10.6f}  {vr:>10.4f}  {best_marks}")

print()
print("  G✓ = best by Gini   |  I✓ = best by Information Gain   |  V✓ = best by Variance")


# ─────────────────────────────────────────────────────────────────────────────
# PART D: DETAILED BREAKDOWN OF THE WINNING SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  PART D: STEP-BY-STEP CALCULATION — Best Split (Split B: credit ≤ 647)")
print("=" * 70)

_, thresh, g_delta, ig, vr, left_cl, right_cl, left_inc, right_inc = results[1]

print()
print(f"  Split: credit_score ≤ {thresh}")
print(f"  Left  child: {len(left_cl)} samples  → credit ≤ {thresh}")
print(f"  Right child: {len(right_cl)} samples → credit >  {thresh}")

print()
print("  LEFT child breakdown:")
print(f"    Samples: {left_cl}")
print(f"    APPROVE: {sum(left_cl)},  DENY: {len(left_cl)-sum(left_cl)}")
print(f"    p₁ (APPROVE) = {sum(left_cl)}/{len(left_cl)} = {sum(left_cl)/len(left_cl):.4f}")

p1_l = sum(left_cl)/len(left_cl)
p0_l = 1 - p1_l
g_l = 1 - (p1_l**2 + p0_l**2)
h_l = -(p1_l*math.log2(p1_l+1e-12) + p0_l*math.log2(p0_l+1e-12)) if p1_l not in [0,1] else 0
print(f"    Gini(left)    = 1 - ({p1_l:.4f}² + {p0_l:.4f}²)")
print(f"                  = 1 - ({p1_l**2:.4f} + {p0_l**2:.4f})")
print(f"                  = 1 - {p1_l**2 + p0_l**2:.4f} = {g_l:.6f}")
print(f"    Entropy(left) = -({p1_l:.4f}·log₂{p1_l:.4f}) - ({p0_l:.4f}·log₂{p0_l:.4f})")
print(f"                  = {h_l:.6f} bits")

print()
print("  RIGHT child breakdown:")
print(f"    Samples: {right_cl}")
print(f"    APPROVE: {sum(right_cl)},  DENY: {len(right_cl)-sum(right_cl)}")
print(f"    p₁ (APPROVE) = {sum(right_cl)}/{len(right_cl)} = {sum(right_cl)/len(right_cl):.4f}")

p1_r = sum(right_cl)/len(right_cl)
p0_r = 1 - p1_r
g_r = 1 - (p1_r**2 + p0_r**2)
h_r = -(p1_r*math.log2(p1_r+1e-12) + p0_r*math.log2(p0_r+1e-12)) if p1_r not in [0,1] else 0
print(f"    Gini(right)    = 1 - ({p1_r:.4f}² + {p0_r:.4f}²) = {g_r:.6f}")
print(f"    Entropy(right) = {h_r:.6f} bits")

nl, nr, n = len(left_cl), len(right_cl), 12
print()
print("  Weighted averages:")
g_weighted = (nl/n)*g_l + (nr/n)*g_r
e_weighted = (nl/n)*h_l + (nr/n)*h_r
print(f"    Gini_after    = ({nl}/{n})×{g_l:.6f} + ({nr}/{n})×{g_r:.6f}")
print(f"                  = {(nl/n)*g_l:.6f} + {(nr/n)*g_r:.6f}")
print(f"                  = {g_weighted:.6f}")
print(f"    Entropy_after = ({nl}/{n})×{h_l:.6f} + ({nr}/{n})×{h_r:.6f}")
print(f"                  = {e_weighted:.6f} bits")

print()
print("  Gain measures (higher = better):")
print(f"    Gini Reduction         = {parent_gini:.6f} - {g_weighted:.6f}")
print(f"                           = {parent_gini - g_weighted:.6f}")
print(f"    Information Gain       = {parent_entropy:.6f} - {e_weighted:.6f}")
print(f"                           = {parent_entropy - e_weighted:.6f} bits")
print()
print(f"    Parent Gini was {parent_gini:.4f}. After split: {g_weighted:.4f}.")
print(f"    This split reduced impurity by {(parent_gini-g_weighted)/parent_gini*100:.1f}%.")


# ─────────────────────────────────────────────────────────────────────────────
# PART E: WHEN DO GINI AND ENTROPY DISAGREE?
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  PART E: WHEN DO GINI AND ENTROPY DISAGREE?")
print("=" * 70)
print("""
  Gini and Entropy are very similar mathematically. Both are:
    • Zero at purity (p = 0 or p = 1)
    • Maximum at uniform distribution (p = 0.5 for binary)
    • Concave functions of the class probability

  They disagree most when comparing:
    • A BALANCED impure split vs an UNBALANCED impure split

  Classic example where they differ:
""")

# Case: large pure + small impure vs. two medium impure children
# Split X: [20 pure class 1] + [10 impure 50/50]
# Split Y: [15 impure 2/3-1/3] + [15 impure 2/3-1/3]

cases = [
    ("Split X (big pure + small impure)",
     [1]*20, [1]*5 + [0]*5,      # 20 pure left, 10 impure right
     "Pure large chunk on one side"),
    ("Split Y (two medium impure nodes)",
     [1]*10 + [0]*5, [1]*10 + [0]*5,  # 15 each, 2/3 impure
     "Both children equally impure"),
]

parent_labels = [1]*30 + [0]*10  # 30 of class 1, 10 of class 0
pg = gini_impurity(parent_labels)
pe = entropy(parent_labels)
print(f"  Parent node: {len(parent_labels)} samples, "
      f"class 1: {sum(parent_labels)}, class 0: {len(parent_labels)-sum(parent_labels)}")
print(f"  Parent Gini: {pg:.6f}   Parent Entropy: {pe:.6f} bits")
print()

for (label, left_y, right_y, note) in cases:
    gs = gini_split(left_y, right_y)
    es = entropy_split(left_y, right_y)
    gd = pg - gs
    ig = pe - es
    print(f"  {label}")
    print(f"    Left:  n={len(left_y)}, class1={sum(left_y)}, "
          f"class0={len(left_y)-sum(left_y)}")
    print(f"    Right: n={len(right_y)}, class1={sum(right_y)}, "
          f"class0={len(right_y)-sum(right_y)}")
    print(f"    Note: {note}")
    print(f"    Gini Reduction:  {gd:.6f}")
    print(f"    Information Gain:{ig:.6f} bits")
    print()

print("""  INTERPRETATION:
  ─────────────────────────────────────────────────────────────────────────
  Gini and Entropy nearly always agree on which split is best. In practice,
  less than 5% of splits are ranked differently between them. When they do
  disagree, it is typically on lopsided splits where one child is pure
  but very small — Entropy gives MORE credit to producing a pure child,
  even a small one. Gini is more conservative.

  For practical purposes: use Gini for speed, Entropy if you want to be
  more principled about uncertainty. The accuracy difference is negligible.
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART F: VARIANCE REDUCTION — REGRESSION TREES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  PART F: VARIANCE REDUCTION — REGRESSION TREES")
print("=" * 70)
print("""
  For regression (continuous y), we cannot use Gini or Entropy —
  those require class labels. Instead, CART uses Variance Reduction.

  At each node, CART picks the split that maximally reduces the variance
  of the target values in the children. The leaf prediction is the MEAN
  of y values in that leaf.

  This is equivalent to finding the axis-aligned split that maximises the
  explained sum of squares — same as fitting a step function to the data.
""")

# A small regression example
feature_x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
target_y   = [ 5,  8, 12, 11, 13, 28, 31, 29, 32,  35]  # clear break at ~55

print("  Example dataset (house age → price, in $k):")
print(f"  {'Age':>6}  {'Price ($k)':>12}")
print(f"  {'-'*22}")
for x, y_val in zip(feature_x, target_y):
    print(f"  {x:>6}  {y_val:>12}")

parent_var = variance(target_y)
print()
print(f"  Parent node variance: {parent_var:.4f}")
print(f"  Parent mean:          {sum(target_y)/len(target_y):.4f}")

print()
print(f"  {'Threshold':>10}  {'Left mean':>10}  {'Right mean':>11}  "
      f"{'VR':>10}  {'Best?'}")
print(f"  {'-'*56}")

best_vr_val, best_thresh_r = -1, None
for thresh in [25, 35, 45, 55, 65, 75]:
    left_y  = [target_y[i] for i in range(len(feature_x))
               if feature_x[i] <= thresh]
    right_y = [target_y[i] for i in range(len(feature_x))
               if feature_x[i] >  thresh]
    if not left_y or not right_y:
        continue
    vr = parent_var - variance_split(left_y, right_y)
    mean_l = sum(left_y) / len(left_y)
    mean_r = sum(right_y) / len(right_y)
    is_best = ""
    if vr > best_vr_val:
        best_vr_val = vr
        best_thresh_r = thresh
        is_best = "← BEST"
    print(f"  age ≤ {thresh:>3}  {mean_l:>10.2f}  {mean_r:>11.2f}  {vr:>10.4f}  {is_best}")

print()
print(f"  Best split: age ≤ {best_thresh_r}  (VR = {best_vr_val:.4f})")
left_y  = [target_y[i] for i in range(len(feature_x))
           if feature_x[i] <= best_thresh_r]
right_y = [target_y[i] for i in range(len(feature_x))
           if feature_x[i] >  best_thresh_r]
print(f"  Left  leaf prediction: mean = ${sum(left_y)/len(left_y):.2f}k")
print(f"  Right leaf prediction: mean = ${sum(right_y)/len(right_y):.2f}k")
print()
print(f"  Interpretation:")
print(f"  For any house with age ≤ {best_thresh_r} years → predict ${sum(left_y)/len(left_y):.0f}k")
print(f"  For any house with age >  {best_thresh_r} years → predict ${sum(right_y)/len(right_y):.0f}k")
print(f"  The split captures the structural break in housing prices at age={best_thresh_r}.")
print()
print("  KEY TAKEAWAY:")
print("  ─────────────────────────────────────────────────────────────────────")
print("  • Gini Impurity:     measures class label mixedness (classification)")
print("  • Information Gain:  measures uncertainty reduction (classification)")
print("  • Variance Reduction: measures target value spread (regression)")
print("  • All three are maximised by the same type of split: one that puts")
print("    similar outputs on the same side of the threshold.")
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
    import streamlit as st

    st.markdown("---")
    st.subheader("⚙️ Operations")

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

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    # try:
    #     from interpretability.visuals.intrinsic_models import (
    #         INTRINSIC_MODELS_VISUAL_HTML,
    #         INTRINSIC_MODELS_VISUAL_HEIGHT,
    #     )
    #     visual_html   = INTRINSIC_MODELS_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = INTRINSIC_MODELS_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[intrinsic_models.py] Could not load visual: {e}", stacklevel=2)

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