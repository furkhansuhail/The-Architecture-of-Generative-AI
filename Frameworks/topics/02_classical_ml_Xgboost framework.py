"""
XGBoost — Extreme Gradient Boosting
=====================================

XGBoost is the most consistently dominant algorithm for structured/tabular data.
It combines gradient boosting with a suite of engineering innovations — second-order
gradients, built-in regularization, sparsity awareness, and cache-optimized tree
building — that make it both faster and more accurate than its predecessors.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "XGBoost: Extreme Gradient Boosting"
DISPLAY_NAME = "02 . XGBoost · Extreme Gradient Boosting"
ICON         = "⚡"
SUBTITLE     = "The Algorithm That Dominates Structured Data"


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

### What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is a gradient boosting framework created by
Tianqi Chen in 2014 and open-sourced in 2016. It has since become one of the most
widely used and cited machine learning algorithms in history — dominating Kaggle
competitions, powering production systems at scale, and winning the majority of
structured-data ML benchmarks.

Its name is not marketing. It is literally extreme gradient boosting: it takes the
classical gradient boosting algorithm and pushes it to the limit with a set of
carefully engineered innovations that make it faster, more accurate, and more
robust than anything that came before it.

If you are working with tabular data — rows and columns, structured features — and
you need the best performance, XGBoost (or its successors LightGBM and CatBoost)
is almost always where you start.

---

### Part 1: The Foundation — Decision Trees as Base Learners

XGBoost builds an ensemble of decision trees. Before understanding how they are
combined, you must understand what a single decision tree does.

A decision tree partitions the feature space into rectangular regions by asking
a sequence of binary questions: "Is feature X greater than threshold t?"
Each internal node asks a question; each leaf node makes a prediction.


    **Diagram 1 — Anatomy of a Decision Tree:**

    DECISION TREE (Depth = 3)
    ══════════════════════════════════════════════════════════

                      [Root Node]
                   Age > 30?
                  /           \\
               YES               NO
               /                   \\
        [Internal Node]        [Internal Node]
        Salary > 50k?          Education > BS?
        /         \\             /           \\
      YES          NO         YES            NO
       /            \\          /               \\
    [Leaf]        [Leaf]   [Leaf]           [Leaf]
    ŷ = 1        ŷ = 0    ŷ = 1           ŷ = 0
    (Buy)       (Skip)    (Buy)           (Skip)

    ─────────────────────────────────────────────────────────────
    TERMINOLOGY:
        Root Node     — first split (top of tree)
        Internal Node — a split on a feature threshold
        Leaf Node     — terminal node, makes a prediction
        Depth         — number of edges from root to deepest leaf
        n_estimators  — number of trees in the ensemble
    ──────────────────────────────────────────────────────────────

    FEATURE SPACE VIEW (2 features):

    Salary ($)
        ↑
     80k│  ■ ■   │  ● ●
        │  ■     │     ●
     50k├────────┤
        │  ○ ○   │  ○
     20k│  ○     │  ○
        └──────────────→ Age
              30        60

    Each split creates a new axis-aligned partition.
    Leaves are rectangles in feature space.
    Deeper trees → finer rectangles → more complex boundaries.


**Why Trees as Base Learners?**

Trees have three properties that make them ideal for boosting:

    1. Non-linear: they capture interactions between features without
       manual feature engineering. "If age > 30 AND salary > 50k" is
       a natural interaction that a tree learns automatically.

    2. Invariant to monotone transformations: you don't need to scale
       features. A tree splits on "feature > threshold", so multiplying
       a feature by 1000 doesn't change which split is best.

    3. Fast to fit: greedy top-down tree growing with efficient
       histogram-based split finding runs in O(n log n) per feature.

---

### Part 2: Ensemble Methods — Bagging vs. Boosting

XGBoost uses **boosting**, but it's important to understand the contrast
with **bagging** (the other major ensemble strategy, used by Random Forests).
They are fundamentally different philosophies.


    **Diagram 2 — Bagging vs. Boosting:**

    BAGGING (Random Forest)             BOOSTING (XGBoost)
    ═══════════════════════             ═════════════════════════════

    All trees trained in PARALLEL       Trees trained SEQUENTIALLY
    on random subsets of data           Each tree corrects previous errors

    Data ─────────────────────         Round 1: Tree₁ fits data
          │       │       │                     ↓ compute residuals
         T₁      T₂      T₃            Round 2: Tree₂ fits residuals
          │       │       │                     ↓ compute new residuals
          └───────┼───────┘            Round 3: Tree₃ fits residuals
                  │                             ↓ ...
              Vote/Average              Final: Sum all tree predictions
              (majority or mean)

    Trees are INDEPENDENT              Trees are DEPENDENT
    Reduces VARIANCE                   Reduces BIAS
    Hard to overfit (more trees        Can overfit (need regularization
    → always better or same)           and early stopping)

    Slow to converge on hard cases     Focuses on hard cases explicitly
    Final prediction: average          Final prediction: weighted sum

    ───────────────────────────────────────────────────────────────────
    KEY INSIGHT:
        Bagging asks: "How can we reduce disagreement between models?"
        Boosting asks: "How can we learn from our mistakes?"
    ───────────────────────────────────────────────────────────────────

---

### Part 3: Gradient Boosting — The Mathematical Core

Gradient boosting is gradient descent in function space. Instead of
optimizing parameters of a single model, it iteratively fits new models
to the negative gradient of the loss function.

**Step back: What is Gradient Descent?**

In standard gradient descent for a neural network, you adjust parameters θ:

        θ ← θ - η × ∂L/∂θ

You move θ in the direction that reduces loss L.

**Gradient Boosting Analogy:**

In gradient boosting, instead of adjusting parameters, you add a new function
(tree) that points in the direction of reducing loss. The "gradient" tells
you what correction the next tree should make.

        F₀(x) = initial prediction (usually mean of y)

        For each round t = 1, 2, ..., T:
            1. Compute pseudo-residuals (negative gradient of loss):
               rᵢ = -∂L(yᵢ, F_{t-1}(xᵢ)) / ∂F_{t-1}(xᵢ)

            2. Fit a tree hₜ to the residuals rᵢ

            3. Add the tree to the ensemble:
               Fₜ(x) = F_{t-1}(x) + η × hₜ(x)

        Final prediction: F_T(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·h_T(x)


    **Diagram 3 — Gradient Boosting Step by Step (Regression):**

    TARGET y = [3, 5, 2, 8, 6]

    ── Round 0 ──────────────────────────────────────────────
    F₀(x) = mean(y) = 4.8 for all samples

    Predictions:     [4.8, 4.8, 4.8, 4.8, 4.8]
    Residuals r = y - F₀:  [-1.8, 0.2, -2.8, 3.2, 1.2]

    ── Round 1 ──────────────────────────────────────────────
    Tree₁ fits the residuals [-1.8, 0.2, -2.8, 3.2, 1.2]

        "Learn what F₀ got wrong"

    Tree₁ predictions: [-1.5, 0.2, -2.5, 3.0, 1.0]  (approximate)

    F₁ = F₀ + η × Tree₁
       = [4.8, 4.8, 4.8, 4.8, 4.8] + 0.1 × [-1.5, 0.2, -2.5, 3.0, 1.0]
       = [4.65, 4.82, 4.55, 5.10, 4.90]

    New residuals:   [-1.65, 0.18, -2.55, 2.90, 1.10]  (smaller!)

    ── Round 2 ──────────────────────────────────────────────
    Tree₂ fits the new (smaller) residuals

    F₂ = F₁ + η × Tree₂  ...

    ── After T rounds ───────────────────────────────────────
    Each tree corrects what all previous trees got wrong.
    Residuals shrink each round → predictions converge toward true y.

    ─────────────────────────────────────────────────────────
    LEARNING RATE η (shrinkage):
        η = 1.0  → full correction each step → fast but fragile
        η = 0.1  → 10% correction each step  → robust, needs more trees
        η = 0.01 → slow but very robust       → needs many trees (1000+)

    TRADEOFF: smaller η → lower η × higher T → better generalization
    ─────────────────────────────────────────────────────────

**Loss Functions — What are we minimizing?**

The gradient changes depending on the loss function you choose:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Task             Loss Function     Gradient (pseudo-residual)       │
    │──────────────────────────────────────────────────────────────────────│
    │  Regression       MSE: ½(y-ŷ)²     y - ŷ  (simple residual)          │
    │  Regression       MAE: |y-ŷ|        sign(y - ŷ)                      │
    │  Classification   Log Loss         y - σ(ŷ)  (true - predicted prob) │
    │  Ranking          LambdaRank        custom pairwise gradient         │
    │  Survival         Cox PH Loss       custom survival gradient         │
    └──────────────────────────────────────────────────────────────────────┘

For binary classification:
    Loss = -[y·log(p) + (1-y)·log(1-p)]   (log loss / cross-entropy)
    Gradient = y - p   (true label minus predicted probability)
    → Tree fits how much each prediction over- or under-estimated the probability

---

### Part 4: XGBoost's Core Innovations

Classical gradient boosting (Friedman 2001) is already powerful.
XGBoost adds six critical innovations that make it the dominant algorithm.


    **Diagram 4 — Classical GBM vs. XGBoost:**

    CLASSICAL GRADIENT BOOSTING        XGBOOST
    ═══════════════════════════════    ══════════════════════════════════════
    First-order gradients only          Second-order gradients (Newton step)
    No regularization                   L1 (α) and L2 (λ) regularization
    Greedy exact split finding          Weighted quantile sketch (approx.)
    Dense data only                     Native sparse/missing value handling
    Single-threaded tree building       Parallelized column block structure
    No early stopping built-in          Built-in early stopping with eval set
    ═══════════════════════════════    ══════════════════════════════════════

**Innovation 1 — Second-Order Gradients (Newton Boosting):**

Classical gradient boosting uses only the first derivative (gradient) of
the loss to guide the next tree. XGBoost uses BOTH first and second derivatives.

    First derivative  gᵢ = ∂L(yᵢ, ŷᵢ) / ∂ŷᵢ     ← gradient (direction)
    Second derivative hᵢ = ∂²L(yᵢ, ŷᵢ) / ∂ŷᵢ²    ← hessian (curvature)

    XGBoost leaf value:   w* = -Σgᵢ / (Σhᵢ + λ)

This is the Newton-Raphson step — it accounts for the curvature of the loss
surface, not just its slope. The result is faster convergence: each tree
makes a more accurate correction than a first-order step would.

    Analogy:
        First-order only  → "the hill goes downward to the left, so step left"
        Second-order      → "the hill curves sharply, so take a precise step
                             scaled to the curvature to land near the bottom"


**Innovation 2 — Regularization (L1 + L2 + Tree Complexity):**

XGBoost's objective function has THREE regularization terms:

    Obj = Σ L(yᵢ, ŷᵢ)  +  γ·T  +  ½λ·Σwⱼ²  +  α·Σ|wⱼ|

    Where:
        L(yᵢ, ŷᵢ)  = data fit term (loss on training examples)
        γ·T         = penalty for the NUMBER of leaves (tree complexity)
        ½λ·Σwⱼ²    = L2 penalty on leaf weights (ridge, prevents large values)
        α·Σ|wⱼ|    = L1 penalty on leaf weights (lasso, promotes sparsity)

    ─────────────────────────────────────────────────────────
    γ (gamma / min_split_loss):
        Minimum gain required to make a split.
        γ = 0    → split freely (can overfit)
        γ = 5    → only split if gain > 5 (conservative tree)

    λ (reg_lambda, default=1):
        L2 regularization. Larger → smaller leaf weights.
        Prevents individual trees from making extreme predictions.

    α (reg_alpha, default=0):
        L1 regularization. Non-zero → some leaf weights become exactly zero.
        Useful in very high-dimensional feature spaces.
    ─────────────────────────────────────────────────────────


**Innovation 3 — The Split Gain Formula:**

When deciding whether to split a node, XGBoost computes:

    Gain = ½ [ GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ) ] - γ

    Where:
        GL, GR  = sum of gradients in left and right child
        HL, HR  = sum of hessians in left and right child
        G, H    = sum of gradients/hessians in the parent
        λ       = L2 regularization term
        γ       = minimum split gain (prune if Gain < 0)

    This formula computes:
        (quality of left leaf) + (quality of right leaf) - (quality of parent)
        minus the complexity penalty γ.

    If Gain < 0, the split makes things worse (or the improvement
    is too small to justify the added complexity) → PRUNE the split.

    ─────────────────────────────────────────────────────────────────────
    PRUNING STRATEGY: XGBoost grows trees to max_depth first ("max-depth
    first"), THEN prunes all splits with negative gain bottom-up. This is
    different from classical CART which stops growing immediately when
    gain < 0. Max-depth-first finds splits that are locally bad but
    globally good — it avoids greedy local optima.
    ─────────────────────────────────────────────────────────────────────


**Innovation 4 — Sparsity-Aware Split Finding (Missing Values):**

XGBoost handles missing values natively without imputation.

For each feature, when computing the best split:
    - XGBoost tries routing all missing values to the LEFT child
    - XGBoost tries routing all missing values to the RIGHT child
    - Whichever direction yields higher Gain is chosen → "default direction"
    - This default direction is LEARNED from data and stored in the tree

    ┌──────────────────────────────────────────────────────────────────┐
    │  At each node that uses feature f:                               │
    │                                                                  │
    │  Known values: use the split threshold as normal                 │
    │  Missing values: automatically route to learned default side     │
    │                                                                  │
    │  Result: no preprocessing needed for NaN values in XGBoost!      │
    └──────────────────────────────────────────────────────────────────┘

    This matters enormously in practice: real datasets always have missing
    values, and imputation choices can subtly bias your model. XGBoost
    learns the optimal way to handle missingness from the data itself.


**Innovation 5 — Column and Row Subsampling:**

Like Random Forests, XGBoost supports subsampling for variance reduction:

    subsample (default=1.0):
        Fraction of training rows used per tree.
        subsample=0.8 → each tree uses a random 80% of training rows.
        Prevents overfitting, adds variance reduction.

    colsample_bytree (default=1.0):
        Fraction of features considered per tree.

    colsample_bylevel (default=1.0):
        Fraction of features considered per tree LEVEL (depth).

    colsample_bynode (default=1.0):
        Fraction of features considered per NODE (split).

    INTUITION: These subsampling parameters work like Random Forest's
    random feature selection — they introduce diversity among trees,
    preventing any single feature from dominating and reducing variance.

---

### Part 5: The XGBoost Hyperparameter Map

XGBoost has ~35 hyperparameters. Most can be ignored. These are the ones
that actually matter, organized by what they control.


    **Diagram 5 — Hyperparameter Relationships:**

    ┌──────────────────────────────────────────────────────────────────────┐
    │                   XGBOOST HYPERPARAMETER MAP                         │
    │                                                                      │
    │  ENSEMBLE SIZE                   TREE COMPLEXITY                     │
    │  ─────────────────────           ────────────────────                │
    │  n_estimators     [100]          max_depth         [6]               │
    │  learning_rate    [0.1]          min_child_weight  [1]               │
    │  early_stopping_rounds [50]      gamma             [0]               │
    │                                  max_leaves        [0=unlimited]     │
    │  ↕ Tradeoff:                                                         │
    │  Low η + High T = best           ↕ Tradeoff:                         │
    │  but slow to train               Deeper = more complex = overfit     │
    │                                                                      │
    │  REGULARIZATION                  SUBSAMPLING                         │
    │  ─────────────────────           ────────────────────                │
    │  reg_lambda (L2)  [1]            subsample         [1.0]             │
    │  reg_alpha  (L1)  [0]            colsample_bytree  [1.0]             │
    │                                  colsample_bylevel [1.0]             │
    │                                  colsample_bynode  [1.0]             │
    │                                                                      │
    │  OBJECTIVE & EVALUATION                                              │
    │  ─────────────────────────────────────────                           │
    │  objective: 'reg:squarederror', 'binary:logistic', 'multi:softmax'   │
    │  eval_metric: 'rmse', 'logloss', 'auc', 'error', 'merror'            │
    └──────────────────────────────────────────────────────────────────────┘


**The Most Important Hyperparameters — Priority Order:**

    TIER 1 — Almost always tune these:
    ┌───────────────────────────────────────────────────────────────────┐
    │  n_estimators + learning_rate                                     │
    │    These are inseparable. Low lr + high n = better generalization │
    │    Use early stopping to find the right n_estimators automatically│
    │    Start: lr=0.1, n=500 with early stopping                       │
    │                                                                   │
    │  max_depth                                                        │
    │    Controls tree complexity. Range: 3–10                          │
    │    Start at 6. Shallow (3–5) for noisy data; deeper for complex.  │
    └───────────────────────────────────────────────────────────────────┘

    TIER 2 — Often helpful:
    ┌───────────────────────────────────────────────────────────────────┐
    │  min_child_weight                                                 │
    │    Minimum sum of hessians in a leaf (instance weight).           │
    │    Higher = more conservative, prevents splits on tiny groups.    │
    │    Default=1; try 1–10.                                           │
    │                                                                   │
    │  subsample + colsample_bytree                                     │
    │    Row and column subsampling. Both default to 1.0.               │
    │    Values 0.6–0.9 often improve generalization.                   │
    │                                                                   │
    │  gamma (min_split_loss)                                           │
    │    Minimum gain for a split. Default=0; try 0–5 for regularizing  │
    └───────────────────────────────────────────────────────────────────┘

    TIER 3 — For fine-tuning:
    ┌───────────────────────────────────────────────────────────────────┐
    │  reg_alpha, reg_lambda                                            │
    │    L1 and L2 regularization. Lambda=1 is already a good default.  │
    │                                                                   │
    │  scale_pos_weight                                                 │
    │    For imbalanced datasets: set to sum(neg)/sum(pos)              │
    └───────────────────────────────────────────────────────────────────┘


---

### Part 6: Feature Importance — Four Different Views

XGBoost provides four distinct ways to measure feature importance.
They measure different things and can give different rankings.


    **Diagram 6 — Feature Importance Types:**

    ┌──────────────────────────────────────────────────────────────────────┐
    │  TYPE          HOW IT'S COMPUTED           WHAT IT MEANS             │
    │──────────────────────────────────────────────────────────────────────│
    │  weight        Count how many times         How often the feature    │
    │  (frequency)   the feature is used as       appears in splits.       │
    │                a split across all trees.    Fast but biased toward   │
    │                                             high-cardinality feats.  │
    │──────────────────────────────────────────────────────────────────────│
    │  gain          Average improvement in       How much does using      │
    │  (default)     the loss (Gain formula)      this feature improve     │
    │                across all splits using       the model? Most         │
    │                this feature.                meaningful metric.       │
    │──────────────────────────────────────────────────────────────────────│
    │  cover         Average number of            How many training        │
    │                samples (hessian sum)         samples are affected    │
    │                covered by splits using       by splits on this       │
    │                this feature.                feature. Coverage.       │
    │──────────────────────────────────────────────────────────────────────│
    │  permutation   Drop in score when the       Gold standard for        │
    │  (SHAP-based)  feature values are            real-world importance.  │
    │                randomly shuffled.            Accounts for feature    │
    │                                             correlations.            │
    └──────────────────────────────────────────────────────────────────────┘

    WHICH TO USE?
        Quick analysis:          gain (default in XGBoost)
        Production/audit:        SHAP values (most reliable)
        Correlated features:     permutation importance
        Feature selection speed: weight or cover

    SHAP (SHapley Additive exPlanations):
        SHAP values explain each individual prediction, not just global
        importance. For each prediction, SHAP shows how much each feature
        pushed the prediction above or below the base value.

        Base value (average prediction)
        + SHAP(age)       = +0.12  ← age pushed prediction up
        + SHAP(salary)    = -0.05  ← salary pushed prediction down
        + SHAP(education) = +0.08  ← education pushed prediction up
        = Final prediction 0.75

        XGBoost has native SHAP support: model.predict(X, pred_contribs=True)


---

### Part 7: Early Stopping and Learning Curves

Early stopping automatically finds the optimal number of trees by monitoring
performance on a validation set during training and stopping when it stops improving.

Without early stopping, you'd have to guess n_estimators or do a slow
grid search. With early stopping, you set n_estimators high (1000+) and
let the algorithm find the right number automatically.


    **Diagram 7 — Early Stopping in Action:**

    Training Loss                        Validation Loss
        ↓ (always decreases)                 ↓ then ↑ (overfitting begins)
        │
    0.30│╲
    0.25│  ╲
    0.20│    ╲_                         0.30│╲
    0.15│      ╲──────                  0.25│  ╲
    0.10│           ╲──────             0.20│    ╲──────────────────────
    0.05│                 ╲──           0.15│                   ↑
        └─────────────────────→         0.10│              BEST: round 120
          0   50  100 150 200 rounds    0.05│                    ↑
                                            │           overfitting starts here
                                            └──────────────────────→
                                              0    50   100  150  200

    BEST MODEL = round 120  (not round 200)

    early_stopping_rounds=50:
        Stop if validation score hasn't improved in 50 consecutive rounds.
        This prevents wasting compute AND finds the optimal n_estimators.
        XGBoost stores the best model automatically.

    Usage:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        # model.best_ntree_limit = the optimal round

---

### Part 8: XGBoost's sklearn API

XGBoost provides sklearn-compatible wrappers that plug directly into
sklearn Pipelines, GridSearchCV, and cross_val_score:

    XGBClassifier    — binary and multiclass classification
    XGBRegressor     — regression
    XGBRanker        — learning to rank
    XGBRFClassifier  — XGBoost-style Random Forest (bagging, not boosting)

    All follow the exact same fit() / predict() / predict_proba() API as sklearn.
    All accept sample_weight in fit() for imbalanced data.
    All support early stopping via fit(..., eval_set=..., early_stopping_rounds=...).

    Pipeline integration:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(n_estimators=500, learning_rate=0.05))
        ])
        pipe.fit(X_train, y_train,
                 model__eval_set=[(X_val, y_val)],
                 model__early_stopping_rounds=50)


---

### Part 9: XGBoost vs. LightGBM vs. CatBoost — The Gradient Boosting Family

XGBoost spawned two successors that improve on it in specific ways:


    **Diagram 8 — Gradient Boosting Framework Comparison:**

    ┌──────────────────────────────────────────────────────────────────────┐
    │             XGBoost        LightGBM          CatBoost                │
    │──────────────────────────────────────────────────────────────────────│
    │  Released   2014/2016       2017 (Microsoft)  2017 (Yandex)          │
    │                                                                      │
    │  Tree       Level-wise      Leaf-wise         Symmetric              │
    │  Growth     (breadth-first) (best-leaf-first) (oblivious trees)      │
    │                                                                      │
    │  Speed      Fast            Very fast         Fast                   │
    │                                                                      │
    │  Categoricals  Manual OHE   Manual OHE        Native handling        │
    │                                                                      │
    │  Memory     Moderate        Low               Moderate               │
    │                                                                      │
    │  Best for   General purpose Large datasets    Data with many         │
    │             Proven baseline GPU training      categoricals           │
    │                             Memory constrained                       │
    │──────────────────────────────────────────────────────────────────────│
    │  sklearn API  ✅            ✅                ✅                     │
    │  SHAP support ✅            ✅                ✅                     │
    │  GPU support  ✅            ✅                ✅                     │
    └──────────────────────────────────────────────────────────────────────┘

    TREE GROWTH STRATEGY COMPARISON:

    Level-wise (XGBoost):        Leaf-wise (LightGBM):
    ───────────────────────      ─────────────────────
         [R]                          [R]
        /   \\                        /   \\
      [L1] [L2]                    [L1] [L2]
      / \\   / \\                         / \\
    [.][.][.][.]                       [L3][L4]

    Grows all nodes at same         Grows the leaf with highest
    depth before going deeper.      gain first, regardless of level.
    Balanced, safer.                Deeper on one side, often better
                                    loss reduction but can overfit.


**When to Choose Each:**

    Start with XGBoost when:
        - You want the battle-tested baseline
        - Your dataset is small to medium (< 1M rows)
        - You need extensive community/documentation support
        - Integration with sklearn pipelines is important

    Switch to LightGBM when:
        - Dataset is large (> 1M rows) and training speed matters
        - Memory is constrained
        - You're using categorical features (with proper encoding)

    Switch to CatBoost when:
        - You have many high-cardinality categorical features
        - You want to avoid manual OHE and target encoding
        - Training directly from raw string categories


---

### Part 10: When to Use XGBoost vs. Neural Networks

The single most important skill in applied ML is knowing which tool to use.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  USE XGBOOST WHEN:           USE NEURAL NETWORKS WHEN:              │
    │  ──────────────────────────  ────────────────────────────────────── │
    │  • Tabular/structured data   • Images, audio, video                 │
    │  • < a few million rows      • Raw text (without hand-crafted NLP)  │
    │  • Need interpretability     • Sequences (time series with complex  │
    │    (SHAP, feature importance)    temporal patterns)                 │
    │  • Limited compute / GPU     • Transfer learning from large models  │
    │  • Fast iteration needed     • Dataset is enormous (100M+ rows)     │
    │  • Mixed numeric+categorical • Custom differentiable architectures  │
    │  • Missing values present    • Multi-modal data (text + image)      │
    │  • Baseline model first                                             │
    │                                                                     │
    │  THE EMPIRICAL REALITY:                                             │
    │  On tabular data competitions (Kaggle, OpenML benchmarks),          │
    │  gradient boosting (XGBoost / LightGBM) wins 70–80% of the time     │
    │  against neural networks. The gap has narrowed with TabNet and      │
    │  FT-Transformer, but GBMs remain the default first choice for       │
    │  structured data. Don't reach for a neural network until a well-    │
    │  tuned GBM has set the bar.                                         │
    └─────────────────────────────────────────────────────────────────────┘

---

### Summary: The XGBoost Mental Model

    1. XGBoost builds trees sequentially — each tree corrects the residuals
       (errors) of all previous trees. This is gradient descent in function space.

    2. The objective function = loss (data fit) + regularization (complexity penalty).
       Both terms are optimized simultaneously at every split.

    3. Second-order gradients (Hessian) make each correction more precise than
       first-order gradient boosting — this is the Newton step.

    4. Four regularization knobs (γ, λ, α, and subsampling) prevent overfitting.
       Unlike Random Forests, XGBoost CAN overfit, so regularization is mandatory.

    5. XGBoost handles missing values natively by learning a default direction
       for each split — no imputation needed.

    6. Early stopping finds the optimal n_estimators automatically.
       Always use it: set n_estimators=1000+, early_stopping_rounds=50.

    7. SHAP values provide the most reliable feature importance and explain
       individual predictions — essential for model auditing and debugging.

    8. sklearn-compatible API means XGBoost plugs directly into Pipelines,
       GridSearchCV, and cross_val_score without any adapter code.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
None
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ──────────────────────────────────────────────────────────────────────────
    "1. XGBoost Core API — Fit, Predict, Evaluate": {
        "description": "Full XGBoost workflow: load data, train, evaluate with both native and sklearn APIs",
        "runnable": True,
        "pipeline_cmd": "xgboost",
        "code": '''
"""
================================================================================
XGBOOST CORE API — FIT / PREDICT / EVALUATE
================================================================================

XGBoost provides two APIs:
    1. Native API  — xgb.train() with DMatrix, maximum control
    2. sklearn API — XGBClassifier/XGBRegressor, drop-in sklearn replacement

Both are demonstrated here side by side.
The sklearn API is recommended for most use cases (Pipeline compatibility).
================================================================================
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# PART A: BINARY CLASSIFICATION (sklearn API)
# ═════════════════════════════════════════════════════════════════════════════

data = load_breast_cancer()
X, y = data.data, data.target     # 569 samples, 30 features, binary target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("=" * 65)
print("  XGBOOST CORE API")
print("=" * 65)
print(f"""
Dataset: Breast Cancer Wisconsin
  Samples:  {X.shape[0]}
  Features: {X.shape[1]}
  Classes:  {len(data.target_names)} ({', '.join(data.target_names)})
  Class balance: {np.bincount(y)}
""")


# ─────────────────────────────────────────────────────────────────────────────
# sklearn API — XGBClassifier
# ─────────────────────────────────────────────────────────────────────────────
# XGBClassifier follows the exact same fit/predict/predict_proba interface
# as any sklearn classifier — no special knowledge needed to use it.
#
# Key parameters explained:
#   n_estimators    — number of trees (boosting rounds)
#   learning_rate   — shrinkage applied to each tree's contribution (η)
#   max_depth       — maximum depth of each tree (complexity control)
#   subsample       — fraction of rows sampled per tree
#   colsample_bytree— fraction of features sampled per tree
#   use_label_encoder=False + eval_metric='logloss' → silences deprecation
# ─────────────────────────────────────────────────────────────────────────────

clf = xgb.XGBClassifier(
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 4,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_lambda        = 1.0,     # L2 regularization (default)
    reg_alpha         = 0.0,     # L1 regularization (default)
    gamma             = 0,       # min gain to split (default)
    objective         = 'binary:logistic',
    eval_metric       = 'logloss',
    random_state      = 42,
    verbosity         = 0
)

clf.fit(X_train, y_train)

y_pred       = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("─" * 65)
print("Part A: Binary Classification (sklearn API)")
print("─" * 65)
print(f"  Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
print(f"  AUC-ROC:   {auc:.4f}")
print()
print(classification_report(y_test, y_pred,
                             target_names=data.target_names))

# ── Inspect model attributes ──
print(f"  Model attributes after training:")
print(f"    n_estimators fitted:  {clf.n_estimators}")
print(f"    Best iteration:       {clf.best_iteration if hasattr(clf, 'best_iteration') else 'N/A (no early stopping)'}")
print(f"    Number of features:   {clf.n_features_in_}")
print(f"    Feature names:        {data.feature_names[:3].tolist()} ...")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Native API — xgb.DMatrix + xgb.train()
# ─────────────────────────────────────────────────────────────────────────────
# The native API uses DMatrix — XGBoost's internal data structure.
# DMatrix is more memory-efficient than numpy arrays and supports:
#   - Missing value encoding (NaN → internal missing marker)
#   - Sample weights
#   - Feature names embedded in the data object
#
# When to prefer the native API:
#   - You need full control over eval sets and callbacks
#   - You're using custom objectives or eval metrics
#   - You're doing research and need low-level access
# ─────────────────────────────────────────────────────────────────────────────

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(data.feature_names))
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=list(data.feature_names))

params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'max_depth':        4,
    'eta':              0.05,          # same as learning_rate
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'lambda':           1.0,
    'alpha':            0.0,
    'seed':             42,
    'verbosity':        0,
}

evals_result = {}
native_model = xgb.train(
    params,
    dtrain,
    num_boost_round = 300,
    evals           = [(dtrain, 'train'), (dtest, 'test')],
    evals_result    = evals_result,
    verbose_eval    = False,
)

native_proba = native_model.predict(dtest)          # returns probabilities directly
native_pred  = (native_proba >= 0.5).astype(int)
native_acc   = accuracy_score(y_test, native_pred)
native_auc   = roc_auc_score(y_test, native_proba)

print("─" * 65)
print("Part A (continued): Native API (xgb.train)")
print("─" * 65)
print(f"  Accuracy: {native_acc:.4f}   AUC-ROC: {native_auc:.4f}")
print(f"  (sklearn API:  Acc={acc:.4f}, AUC={auc:.4f}  — results should be identical)")
print()

# Show the learning curve from evals_result
train_auc_curve = evals_result['train']['auc']
test_auc_curve  = evals_result['test']['auc']
print("  Learning curve (AUC by round):")
print(f"  {'Round':>6}  {'Train AUC':>10}  {'Test AUC':>10}  {'Overfit Gap':>12}")
print(f"  {'-' * 44}")
for round_n in [1, 10, 50, 100, 150, 200, 250, 300]:
    if round_n - 1 < len(train_auc_curve):
        tr = train_auc_curve[round_n - 1]
        te = test_auc_curve[round_n - 1]
        gap = tr - te
        flag = " ← overfitting!" if gap > 0.03 else ""
        print(f"  {round_n:>6}  {tr:>10.4f}  {te:>10.4f}  {gap:>12.4f}{flag}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# PART B: REGRESSION
# ═════════════════════════════════════════════════════════════════════════════

diabetes = load_diabetes()
Xr, yr = diabetes.data, diabetes.target

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

reg = xgb.XGBRegressor(
    n_estimators     = 300,
    learning_rate    = 0.05,
    max_depth        = 3,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = 'reg:squarederror',
    random_state     = 42,
    verbosity        = 0
)
reg.fit(Xr_train, yr_train)
yr_pred = reg.predict(Xr_test)

rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2   = r2_score(yr_test, yr_pred)

print("─" * 65)
print("Part B: Regression (XGBRegressor)")
print("─" * 65)
print(f"""
Dataset: Diabetes progression (sklearn built-in)
  RMSE: {rmse:.3f}
  R²:   {r2:.4f}  ({r2*100:.1f}% of variance explained)

  Sample predictions (first 5 test examples):
  {'True':>10}  {'Predicted':>10}  {'Error':>8}
  {'-' * 35}""")
for true, pred in zip(yr_test[:5], yr_pred[:5]):
    err = pred - true
    print(f"  {true:>10.1f}  {pred:>10.1f}  {err:>+8.1f}")

print()
print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  1. sklearn API (XGBClassifier/XGBRegressor) → fit/predict/predict_proba
     same as any sklearn model. Use for Pipelines and GridSearchCV.

  2. Native API (DMatrix + xgb.train) → more control, callbacks,
     custom objectives. Use for research or advanced workflows.

  3. DMatrix is XGBoost's internal format — faster than numpy arrays
     for large datasets and enables native missing value handling.

  4. Learning curve (evals_result) shows if/when overfitting begins.
     Use it to diagnose and set early_stopping_rounds.

  5. objective= controls the loss function:
       'binary:logistic'    → binary classification
       'multi:softmax'      → multiclass
       'reg:squarederror'   → regression (MSE)
       'reg:absoluteerror'  → regression (MAE)
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "2. The Boosting Process — Visualizing Residual Correction": {
        "description": "Step-by-step trace of how each XGBoost tree corrects the previous ensemble's errors",
        "runnable": True,
        "pipeline_cmd": "xgboost",
        "code": '''
"""
================================================================================
THE BOOSTING PROCESS — RESIDUAL CORRECTION VISUALIZED
================================================================================

This module manually traces gradient boosting round by round on a tiny dataset,
so you can see exactly what each tree is learning and how predictions improve.

The key insight: each tree does NOT fit the original target y.
It fits the RESIDUALS (errors) left by all previous trees combined.
================================================================================
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny dataset — easy to trace by hand
# ─────────────────────────────────────────────────────────────────────────────

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2.5, 4.0, 3.5, 6.0, 7.5, 5.0, 8.0, 9.5])

print("=" * 65)
print("  THE BOOSTING PROCESS — RESIDUAL CORRECTION")
print("=" * 65)
print(f"""
Dataset (8 samples, 1 feature):
  x:  {X.flatten().tolist()}
  y:  {y.tolist()}
""")


# ─────────────────────────────────────────────────────────────────────────────
# Manual boosting walkthrough (pure numpy — no XGBoost here)
# This shows EXACTLY what gradient boosting does internally.
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("MANUAL GRADIENT BOOSTING (MSE loss, shallow stumps)")
print("─" * 65)

learning_rate = 0.5

# Round 0: initial prediction = mean(y)
F = np.full(len(y), np.mean(y))
print()
print(f"Round 0 — Initial prediction:")
print(f"  F₀(x) = mean(y) = {np.mean(y):.4f} for all samples")
print(f"  Predictions: {F.round(3).tolist()}")
print(f"  Residuals:   {(y - F).round(3).tolist()}")
print(f"  RMSE:        {np.sqrt(mean_squared_error(y, F)):.4f}")

# Function to fit a 1-level stump: split on the median of x
def fit_stump(X_1d, residuals):
    """Fit a depth-1 decision tree (stump) to residuals.
    Returns (threshold, left_value, right_value).
    """
    best_gain   = -np.inf
    best_thresh = None
    best_left   = None
    best_right  = None

    thresholds = np.unique(X_1d)
    for t in thresholds[:-1]:
        left_mask  = X_1d <= t
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        left_val  = residuals[left_mask].mean()
        right_val = residuals[right_mask].mean()

        # MSE gain: variance reduction
        parent_mse = residuals.var()
        left_mse   = residuals[left_mask].var()  if left_mask.sum()  > 1 else 0
        right_mse  = residuals[right_mask].var() if right_mask.sum() > 1 else 0
        gain = (parent_mse
                - (left_mask.sum() / len(X_1d))  * left_mse
                - (right_mask.sum() / len(X_1d)) * right_mse)

        if gain > best_gain:
            best_gain   = gain
            best_thresh = t
            best_left   = left_val
            best_right  = right_val

    return best_thresh, best_left, best_right

def predict_stump(X_1d, thresh, left_val, right_val):
    return np.where(X_1d <= thresh, left_val, right_val)


# Boosting rounds
n_rounds = 5
X_1d = X.flatten()

for t in range(1, n_rounds + 1):
    residuals = y - F
    thresh, lv, rv = fit_stump(X_1d, residuals)
    stump_pred = predict_stump(X_1d, thresh, lv, rv)
    F = F + learning_rate * stump_pred
    new_residuals = y - F
    rmse = np.sqrt(mean_squared_error(y, F))

    print()
    print(f"Round {t} — Tree fits residuals:")
    print(f"  Stump: x <= {thresh:.1f} → predict {lv:.4f}, else → predict {rv:.4f}")
    print(f"  Stump predictions: {stump_pred.round(3).tolist()}")
    print(f"  Update:  F_{t} = F_{t-1} + {learning_rate} × stump")
    print(f"  New predictions:   {F.round(3).tolist()}")
    print(f"  New residuals:     {new_residuals.round(3).tolist()}")
    print(f"  RMSE:              {rmse:.4f}")
    bar_old = "░" * 30
    bar_new = "█" * int((1 - rmse / 2.5) * 30)
    print(f"  Progress:  [{bar_new:<30}] {(1-rmse/2.5)*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Now fit the same data with real XGBoost and compare
# ─────────────────────────────────────────────────────────────────────────────

print()
print(f"{'─' * 65}")
print("Real XGBoost — comparing n_estimators effect")
print("─" * 65)
print(f"{'n_trees':>8}  {'Train RMSE':>12}  {'Ensemble size':>14}  {'Progress bar'}")
print(f"{'-' * 55}")

for n in [1, 3, 5, 10, 20, 50, 100, 200]:
    model = xgb.XGBRegressor(
        n_estimators  = n,
        learning_rate = 0.3,
        max_depth     = 2,
        reg_lambda    = 1.0,
        verbosity     = 0,
        random_state  = 42
    )
    model.fit(X, y)
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    bar  = "█" * min(30, int((1 - rmse / 3.0) * 30))
    print(f"  {n:>6}  {rmse:>12.4f}  {'trees:' + str(n):>14}  [{bar:<30}]")

print(f"""
  Observations:
    1. More trees → lower training RMSE (boosting always improves training fit)
    2. At some point, additional trees stop helping test performance (overfitting)
    3. Early stopping finds the sweet spot automatically
    4. With learning_rate=0.3 and max_depth=2: good balance
""")


# ─────────────────────────────────────────────────────────────────────────────
# Visualize the learned ensemble on the training data
# ─────────────────────────────────────────────────────────────────────────────

model_final = xgb.XGBRegressor(
    n_estimators=50, learning_rate=0.3, max_depth=2,
    verbosity=0, random_state=42
)
model_final.fit(X, y)
preds_final = model_final.predict(X)

print("─" * 65)
print("Final ensemble predictions vs. true values:")
print("─" * 65)
print(f"  {'x':>4}  {'y_true':>8}  {'y_pred':>8}  {'error':>8}  {'bar'}")
print(f"  {'-' * 50}")
for xi, yi, pi in zip(X_1d, y, preds_final):
    err = pi - yi
    bar = "+" * int(abs(err) * 3) if err > 0 else "-" * int(abs(err) * 3)
    print(f"  {xi:>4}  {yi:>8.2f}  {pi:>8.3f}  {err:>+8.3f}  {bar}")

final_rmse = np.sqrt(mean_squared_error(y, preds_final))
print()
print(f"  Final RMSE: {final_rmse:.4f}")

print()
print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  1. Round 0 starts with the global mean — the simplest possible prediction
  2. Each tree is fit to RESIDUALS (y - F), not the original target y
  3. learning_rate scales each tree's contribution — smaller = safer
  4. Residuals shrink each round: the ensemble converges toward y
  5. Training RMSE always decreases — but test RMSE will eventually rise
  6. The 'correction chain' is the core of gradient boosting:
     F₀ → F₁ = F₀ + η·h₁ → F₂ = F₁ + η·h₂ → ...
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "3. Early Stopping and Learning Curves": {
        "description": "Use eval_set + early_stopping_rounds to automatically find optimal n_estimators",
        "runnable": True,
        "pipeline_cmd": "xgboost",
        "code": '''
"""
================================================================================
EARLY STOPPING AND LEARNING CURVES
================================================================================

Early stopping is one of the most practically important XGBoost techniques.
It:
    1. Finds the optimal n_estimators automatically (no grid search needed)
    2. Prevents overfitting by stopping when validation score plateaus
    3. Saves training time by not building unnecessary trees
    4. Works for both classification and regression

Pattern:
    1. Set n_estimators HIGH (1000 or more)
    2. Pass eval_set with a held-out validation set
    3. Set early_stopping_rounds (typically 30–100)
    4. XGBoost stores the best round → use model.best_ntree_limit
================================================================================
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# Create a dataset with some noise (realistic overfitting conditions)
X, y = make_classification(
    n_samples    = 2000,
    n_features   = 20,
    n_informative= 10,
    n_redundant  = 5,
    flip_y       = 0.05,      # 5% label noise → creates overfitting pressure
    random_state = 42
)

# Three-way split: train / validation (for early stopping) / test (final eval)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42
)

print("=" * 65)
print("  EARLY STOPPING AND LEARNING CURVES")
print("=" * 65)
print(f"""
Dataset: Synthetic classification with 5% label noise
  Train:      {X_train.shape[0]} samples
  Validation: {X_val.shape[0]} samples  ← used for early stopping
  Test:       {X_test.shape[0]} samples  ← held out until the end
""")


# ─────────────────────────────────────────────────────────────────────────────
# WITHOUT early stopping — manually observe overfitting
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Scenario A: WITHOUT early stopping (fixed n_estimators=500)")
print("─" * 65)

model_no_es = xgb.XGBClassifier(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = 'binary:logistic',
    eval_metric      = 'auc',
    random_state     = 42,
    verbosity        = 0
)

evals_result_no_es = {}
model_no_es.fit(
    X_train, y_train,
    eval_set       = [(X_train, y_train), (X_val, y_val)],
    verbose        = False
)

# Since verbose=False blocks evals_result capture in sklearn API,
# use native API to get the full learning curve
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'max_depth':        5,
    'eta':              0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'seed':             42,
    'verbosity':        0,
}

evals_result = {}
model_native = xgb.train(
    params,
    dtrain,
    num_boost_round = 500,
    evals           = [(dtrain, 'train'), (dval, 'val')],
    evals_result    = evals_result,
    verbose_eval    = False,
)

train_auc = evals_result['train']['auc']
val_auc   = evals_result['val']['auc']
best_round = int(np.argmax(val_auc)) + 1
best_val   = max(val_auc)

print()
print(f"Learning curve (selected rounds):")
print(f"  {'Round':>6}  {'Train AUC':>10}  {'Val AUC':>10}  {'Gap':>8}  {'Note'}")
print(f"  {'-' * 55}")
checkpoints = [1, 5, 10, 25, 50, 100, best_round, 200, 350, 500]
for r in sorted(set(checkpoints)):
    if r - 1 < len(train_auc):
        tr  = train_auc[r - 1]
        v   = val_auc[r - 1]
        gap = tr - v
        note = ""
        if r == best_round:   note = " ← BEST VAL ROUND"
        elif r > best_round:  note = " ← overfitting"
        print(f"  {r:>6}  {tr:>10.4f}  {v:>10.4f}  {gap:>8.4f}  {note}")

print(f"""
  Best validation round: {best_round}  (Val AUC = {best_val:.4f})
  At round 500:          Train AUC = {train_auc[-1]:.4f},  Val AUC = {val_auc[-1]:.4f}
  Overfit gap at 500:    {train_auc[-1] - val_auc[-1]:.4f}

  → Without early stopping, we train {500 - best_round} unnecessary trees
    AND end up with a worse model than round {best_round}!
""")


# ─────────────────────────────────────────────────────────────────────────────
# WITH early stopping — correct approach
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Scenario B: WITH early stopping (n_estimators=1000, stop after 50)")
print("─" * 65)

evals_result_es = {}
model_es_native = xgb.train(
    params,
    dtrain,
    num_boost_round    = 1000,
    evals              = [(dtrain, 'train'), (dval, 'val')],
    evals_result       = evals_result_es,
    early_stopping_rounds = 50,
    verbose_eval       = False,
)

print(f"""
  Training stopped at round: {model_es_native.best_iteration + 1}
  Best validation AUC:       {model_es_native.best_score:.4f}
  Rounds saved vs 1000:      {1000 - (model_es_native.best_iteration + 1)}

  Final test evaluation:""")

# Evaluate best model on test set
test_proba = model_es_native.predict(dtest)
test_pred  = (test_proba >= 0.5).astype(int)
test_auc   = roc_auc_score(y_test, test_proba)
test_acc   = accuracy_score(y_test, test_pred)
print(f"    Test AUC:      {test_auc:.4f}")
print(f"    Test Accuracy: {test_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Effect of learning_rate on early stopping round
# ─────────────────────────────────────────────────────────────────────────────

print()
print(f"{'─' * 65}")
print("Effect of learning_rate on optimal round (fixed max_depth=5):")
print("─" * 65)
print(f"  {'lr':>6}  {'Best Round':>11}  {'Best Val AUC':>13}  {'Interpretation'}")
print(f"  {'-' * 60}")

for lr in [0.3, 0.1, 0.05, 0.01]:
    p = dict(params)
    p['eta'] = lr
    er = {}
    m = xgb.train(
        p, dtrain, num_boost_round=2000,
        evals=[(dval, 'val')], evals_result=er,
        early_stopping_rounds=50, verbose_eval=False
    )
    br   = m.best_iteration + 1
    bauc = m.best_score
    note = {0.3: "fast convergence, fewer trees",
            0.1: "balanced — good default",
            0.05: "slower, often better final model",
            0.01: "very slow, needs many trees"}[lr]
    print(f"  {lr:>6}  {br:>11}  {bauc:>13.4f}  {note}")

print(f"""
  KEY INSIGHT:
    Lower learning_rate → needs more rounds to converge
    BUT: the final model is often better (more regularized path)
    Trade-off: lower lr × higher n_trees = better generalization,
               but more training time.

    Rule of thumb:
      - Start: lr=0.1, n_estimators=1000, early_stopping_rounds=50
      - Fine-tune: lr=0.05 + more trees for final model
""")

print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  1. ALWAYS use early stopping — it's free regularization
  2. Three-way split: train / validation (ES) / test (final eval)
  3. Use eval_set=[(X_val, y_val)] + early_stopping_rounds in fit()
  4. model.best_iteration + 1 = optimal n_estimators for retraining
  5. Lower learning_rate → more rounds needed → often better model
  6. Early stopping is not a substitute for tuning max_depth / subsample
  7. In production: retrain with best_ntree_limit on full train+val set
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "4. Feature Importance — All Four Types + SHAP": {
        "description": "Compare weight, gain, cover importance types and interpret individual predictions with SHAP",
        "runnable": True,
        "pipeline_cmd": "xgboost",
        "code": '''
"""
================================================================================
FEATURE IMPORTANCE — WEIGHT, GAIN, COVER, AND SHAP
================================================================================

XGBoost provides four built-in feature importance metrics.
They can give different rankings — understanding which to use is critical
for model interpretability and feature selection.

    weight   — how many splits use this feature (biased, not recommended alone)
    gain     — average improvement in loss per split (most informative default)
    cover    — average samples affected per split (complementary to gain)
    SHAP     — contribution of each feature to each individual prediction
               (the gold standard for explainability)
================================================================================
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = xgb.XGBClassifier(
    n_estimators     = 200,
    learning_rate    = 0.05,
    max_depth        = 4,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbosity        = 0
)
model.fit(X_train, y_train)

print("=" * 70)
print("  FEATURE IMPORTANCE — FOUR TYPES + SHAP")
print("=" * 70)
print()
print(f"Model: XGBClassifier on Breast Cancer dataset")
print(f"AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
print(f"Features: {len(feature_names)}")


# ─────────────────────────────────────────────────────────────────────────────
# Built-in importance types
# ─────────────────────────────────────────────────────────────────────────────

booster = model.get_booster()

# Get importance scores for each type
scores_weight = booster.get_score(importance_type='weight')
scores_gain   = booster.get_score(importance_type='gain')
scores_cover  = booster.get_score(importance_type='cover')

# Sort by gain (most informative)
top_features_gain = sorted(scores_gain.items(), key=lambda x: x[1], reverse=True)[:10]

print()
print(f"{'─' * 70}")
print(f"Top 10 Features by GAIN (average loss improvement per split):")
print(f"{'─' * 70}")
print(f"  {'Feature':<32}  {'Gain':>8}  {'Weight':>8}  {'Cover':>8}")
print(f"  {'-' * 62}")
for fname, g in top_features_gain:
    w = scores_weight.get(fname, 0)
    c = scores_cover.get(fname, 0)
    g_bar = "█" * min(30, int(g / max(scores_gain.values()) * 30))
    print(f"  {fname:<32}  {g:>8.2f}  {w:>8.0f}  {c:>8.1f}  {g_bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Show cases where weight and gain disagree
# ─────────────────────────────────────────────────────────────────────────────

print()
print(f"{'─' * 70}")
print(f"Weight vs. Gain — where they disagree (reveals weight's bias):")
print(f"{'─' * 70}")
print(f"""
  Weight = how often a feature is USED in splits
  Gain   = how USEFUL a feature is per split (average improvement)

  A feature can be used frequently (high weight) but with small gain
  → it's a "cheap" feature that just barely makes splits.

  A feature can be used rarely (low weight) but with large gain
  → each time it splits, it captures something very important.

  RECOMMENDATION: Use 'gain' as your default importance metric.
  Use 'weight' only as a secondary view to understand split patterns.
""")

# Rank features by weight and gain separately, show differences
all_features = list(scores_gain.keys())
rank_by_gain   = sorted(all_features, key=lambda f: scores_gain.get(f, 0),   reverse=True)
rank_by_weight = sorted(all_features, key=lambda f: scores_weight.get(f, 0), reverse=True)

print(f"  {'Feature':<32}  {'Gain Rank':>10}  {'Weight Rank':>12}  {'Rank Diff':>10}")
print(f"  {'-' * 70}")
for feat in rank_by_gain[:8]:
    gr = rank_by_gain.index(feat) + 1
    wr = rank_by_weight.index(feat) + 1 if feat in rank_by_weight else len(all_features)
    diff = wr - gr
    flag = "  ← big disagreement!" if abs(diff) >= 3 else ""
    print(f"  {feat:<32}  {gr:>10}  {wr:>12}  {diff:>+10}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP values — the gold standard
# ─────────────────────────────────────────────────────────────────────────────

print()
print(f"{'─' * 70}")
print(f"SHAP Values — Explaining Individual Predictions")
print(f"{'─' * 70}")
print(f"""
  SHAP (SHapley Additive exPlanations) answers:
  "How much did each feature contribute to THIS specific prediction?"

  Unlike gain/weight (global averages), SHAP works per-sample:
    Base value + SHAP(feat_1) + SHAP(feat_2) + ... = log-odds of prediction

  Positive SHAP → feature pushed prediction TOWARD positive class
  Negative SHAP → feature pushed prediction TOWARD negative class
""")

# Get SHAP values using XGBoost's native SHAP support
dtest = xgb.DMatrix(X_test, feature_names=feature_names)
shap_values = booster.predict(dtest, pred_contribs=True)
# shap_values shape: (n_samples, n_features + 1)
# Last column = base value (bias term)

base_value   = shap_values[0, -1]    # same for all samples in this model
shap_contribs = shap_values[:, :-1]  # (n_samples, n_features)

# Show SHAP breakdown for a sample predicted as malignant (class 1)
malignant_idx = np.where(y_test == 1)[0][0]
benign_idx    = np.where(y_test == 0)[0][0]

def show_shap_breakdown(sample_idx, sample_label, y_test, X_test, shap_contribs,
                         feature_names, base_value, booster):
    dmat   = xgb.DMatrix(X_test[sample_idx:sample_idx+1], feature_names=feature_names)
    pred_log_odds = booster.predict(dmat, output_margin=True)[0]
    pred_prob     = 1 / (1 + np.exp(-pred_log_odds))

    print(f"  Sample #{sample_idx} — True label: {'malignant' if y_test[sample_idx] == 1 else 'benign'}")
    print(f"  Predicted probability (malignant): {pred_prob:.4f}")
    print(f"  Base value (log-odds): {base_value:.4f}")
    print()
    print(f"  SHAP decomposition (top 6 contributors):")
    print(f"  {'Feature':<32}  {'Value':>8}  {'SHAP':>8}  {'Direction'}")
    print(f"  {'-' * 62}")

    contribs = shap_contribs[sample_idx]
    top_idx  = np.argsort(np.abs(contribs))[::-1][:6]
    running  = base_value
    for i in top_idx:
        feat_name  = feature_names[i]
        feat_value = X_test[sample_idx, i]
        shap_val   = contribs[i]
        direction  = "→ malignant ↑" if shap_val > 0 else "→ benign ↓  "
        bar_val    = int(abs(shap_val) * 15)
        bar        = ("+" if shap_val > 0 else "-") * min(bar_val, 20)
        running   += shap_val
        print(f"  {feat_name:<32}  {feat_value:>8.3f}  {shap_val:>+8.4f}  {direction}  {bar}")
    print(f"  {'─' * 62}")
    print(f"  Base value:                                    {base_value:>+8.4f}")
    print(f"  Sum of all SHAPs + base = log-odds:            {pred_log_odds:>+8.4f}")
    print(f"  → Probability = sigmoid({pred_log_odds:.4f}) = {pred_prob:.4f}")
    print()

show_shap_breakdown(malignant_idx, "malignant", y_test, X_test, shap_contribs,
                     feature_names, base_value, booster)
show_shap_breakdown(benign_idx, "benign", y_test, X_test, shap_contribs,
                     feature_names, base_value, booster)

# Global SHAP importance: mean(|SHAP|) per feature
global_shap = np.abs(shap_contribs).mean(axis=0)
top_shap    = np.argsort(global_shap)[::-1][:8]

print(f"{'─' * 70}")
print(f"Global SHAP importance — mean(|SHAP|) per feature:")
print(f"{'─' * 70}")
print(f"  {'Rank':<5}  {'Feature':<32}  {'Mean |SHAP|':>12}  {'Bar'}")
print(f"  {'-' * 65}")
for rank, i in enumerate(top_shap):
    imp = global_shap[i]
    bar = "█" * int(imp / global_shap[top_shap[0]] * 30)
    print(f"  {rank+1:<5}  {feature_names[i]:<32}  {imp:>12.4f}  {bar}")

print()
print("=" * 70)
print("  Key Takeaways")
print("=" * 70)
print("""
  1. 'gain' is the best default importance type — use it over 'weight'
  2. 'weight' is biased: features used in many small splits look important
  3. SHAP is the gold standard: explains individual predictions, not averages
  4. SHAP base value = model's average prediction (log-odds space)
  5. sum(SHAP values) + base_value = model's log-odds for that sample
  6. pred_contribs=True in booster.predict() returns native SHAP values
  7. For production: use SHAP to audit model decisions and detect bias
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "5. Hyperparameter Tuning — Systematic Grid and Bayesian Search": {
        "description": "Tune XGBoost with sklearn GridSearchCV and a staged tuning strategy",
        "runnable": True,
        "pipeline_cmd": "xgboost",
        "code": '''
"""
================================================================================
XGBOOST HYPERPARAMETER TUNING
================================================================================

XGBoost has ~35 hyperparameters. The key to efficient tuning is:
    1. Understand which parameters interact and which are independent
    2. Tune in stages: find optimal trees first, then tree structure, then regularization
    3. Use early stopping to remove n_estimators from the search space
    4. Use RandomizedSearchCV for large spaces, GridSearchCV for final refinement

Stage-by-stage strategy:
    Stage 1: Fix learning_rate=0.1, use early stopping → find best n_estimators
    Stage 2: Tune max_depth + min_child_weight  (tree structure)
    Stage 3: Tune subsample + colsample_bytree  (variance reduction)
    Stage 4: Tune reg_alpha + reg_lambda         (regularization)
    Stage 5: Lower learning_rate, retrain with more trees
================================================================================
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import roc_auc_score, make_scorer
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


X, y = make_classification(
    n_samples    = 1000,
    n_features   = 15,
    n_informative= 8,
    n_redundant  = 4,
    flip_y       = 0.05,
    random_state = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

print("=" * 65)
print("  XGBOOST HYPERPARAMETER TUNING")
print("=" * 65)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Baseline — default parameters
# ─────────────────────────────────────────────────────────────────────────────

baseline = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    random_state=42, verbosity=0, eval_metric='logloss'
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = cross_val_score(baseline, X_train, y_train, cv=cv, scoring='roc_auc')

print("─" * 65)
print("Stage 1 — Baseline (all defaults):")
print("─" * 65)
print(f"  CV AUC: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1b: Use early stopping to find optimal n_estimators
# ─────────────────────────────────────────────────────────────────────────────

es_model = xgb.XGBClassifier(
    n_estimators          = 2000,
    learning_rate         = 0.05,
    max_depth             = 6,
    random_state          = 42,
    verbosity             = 0,
    eval_metric           = 'auc',
    early_stopping_rounds = 50       # XGBoost 2.x: must be in constructor, not fit()
)
es_model.fit(
    X_tr, y_tr,
    eval_set = [(X_val, y_val)],
    verbose  = False
)

optimal_n = es_model.best_iteration + 1
print(f"  Early stopping → optimal n_estimators: {optimal_n}")
print(f"  Best validation AUC: {es_model.best_score:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Tune tree structure — max_depth + min_child_weight
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Stage 2 — Tree structure: max_depth + min_child_weight")
print("─" * 65)

param_grid_tree = {
    'max_depth':        [3, 4, 5, 6, 7],
    'min_child_weight': [1, 3, 5, 7],
}

tree_search = GridSearchCV(
    xgb.XGBClassifier(
        n_estimators=optimal_n, learning_rate=0.05,
        random_state=42, verbosity=0, eval_metric='logloss'
    ),
    param_grid_tree,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
tree_search.fit(X_train, y_train)

best_depth = tree_search.best_params_['max_depth']
best_mcw   = tree_search.best_params_['min_child_weight']

print(f"  Grid: max_depth × min_child_weight = {len(param_grid_tree['max_depth'])} × {len(param_grid_tree['min_child_weight'])} = {len(param_grid_tree['max_depth']) * len(param_grid_tree['min_child_weight'])} combinations")
print()
print(f"  Results table:")
print(f"  {'max_depth':>10}  {'min_child_weight':>18}  {'CV AUC':>8}")
print(f"  {'-' * 42}")
import pandas as pd
res = pd.DataFrame(tree_search.cv_results_)
for _, row in res.nlargest(5, 'mean_test_score').iterrows():
    flag = " ← best" if (row['param_max_depth'] == best_depth and
                          row['param_min_child_weight'] == best_mcw) else ""
    print(f"  {int(row['param_max_depth']):>10}  {int(row['param_min_child_weight']):>18}  "
          f"{row['mean_test_score']:>8.4f}{flag}")
print()
print(f"  Best: max_depth={best_depth}, min_child_weight={best_mcw}")
print(f"  Score: {tree_search.best_score_:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: Tune subsampling
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Stage 3 — Subsampling: subsample + colsample_bytree")
print("─" * 65)

param_grid_sub = {
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
}

sub_search = GridSearchCV(
    xgb.XGBClassifier(
        n_estimators=optimal_n, learning_rate=0.05,
        max_depth=best_depth, min_child_weight=best_mcw,
        random_state=42, verbosity=0, eval_metric='logloss'
    ),
    param_grid_sub,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
sub_search.fit(X_train, y_train)

best_ss  = sub_search.best_params_['subsample']
best_cbt = sub_search.best_params_['colsample_bytree']
print(f"  Best: subsample={best_ss}, colsample_bytree={best_cbt}")
print(f"  Score: {sub_search.best_score_:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Tune regularization
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Stage 4 — Regularization: gamma + reg_alpha + reg_lambda")
print("─" * 65)

param_grid_reg = {
    'gamma':      [0, 0.1, 0.3, 0.5, 1.0],
    'reg_alpha':  [0, 0.01, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0],
}
total_reg = (len(param_grid_reg['gamma']) *
             len(param_grid_reg['reg_alpha']) *
             len(param_grid_reg['reg_lambda']))

from sklearn.model_selection import RandomizedSearchCV
reg_search = RandomizedSearchCV(
    xgb.XGBClassifier(
        n_estimators=optimal_n, learning_rate=0.05,
        max_depth=best_depth, min_child_weight=best_mcw,
        subsample=best_ss, colsample_bytree=best_cbt,
        random_state=42, verbosity=0, eval_metric='logloss'
    ),
    param_grid_reg,
    n_iter=20, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42
)
reg_search.fit(X_train, y_train)

best_gamma  = reg_search.best_params_['gamma']
best_alpha  = reg_search.best_params_['reg_alpha']
best_lambda = reg_search.best_params_['reg_lambda']
print(f"  Searched: {total_reg} combinations, tried: 20")
print(f"  Best: gamma={best_gamma}, reg_alpha={best_alpha}, reg_lambda={best_lambda}")
print(f"  Score: {reg_search.best_score_:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: Final model — lower lr, more trees, final evaluation
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Stage 5 — Final model: lower lr=0.02, retrain with more trees")
print("─" * 65)

final_model = xgb.XGBClassifier(
    n_estimators          = optimal_n * 5,   # more trees for lower lr
    learning_rate         = 0.02,
    max_depth             = best_depth,
    min_child_weight      = best_mcw,
    subsample             = best_ss,
    colsample_bytree      = best_cbt,
    gamma                 = best_gamma,
    reg_alpha             = best_alpha,
    reg_lambda            = best_lambda,
    random_state          = 42,
    verbosity             = 0,
    eval_metric           = 'auc',
    early_stopping_rounds = 100      # XGBoost 2.x: must be in constructor, not fit()
)
final_model.fit(
    X_tr, y_tr,
    eval_set = [(X_val, y_val)],
    verbose  = False
)

test_auc_final = roc_auc_score(
    y_test, final_model.predict_proba(X_test)[:, 1]
)
test_auc_base = roc_auc_score(
    y_test, baseline.fit(X_train, y_train).predict_proba(X_test)[:, 1]
)

print()
print(f"  Hyperparameter summary:")
print(f"    n_estimators (early stopped): {final_model.best_iteration + 1}")
print(f"    learning_rate:    0.02")
print(f"    max_depth:        {best_depth}")
print(f"    min_child_weight: {best_mcw}")
print(f"    subsample:        {best_ss}")
print(f"    colsample_bytree: {best_cbt}")
print(f"    gamma:            {best_gamma}")
print(f"    reg_alpha:        {best_alpha}")
print(f"    reg_lambda:       {best_lambda}")
print()
print(f"  ─────────────────────────────────────────")
print(f"  Baseline (default)  Test AUC: {test_auc_base:.4f}")
print(f"  Tuned final model   Test AUC: {test_auc_final:.4f}")
print(f"  Improvement:               +{test_auc_final - test_auc_base:.4f}")
print()
print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  1. Stage-by-stage tuning is more efficient than one giant grid search
  2. Use early stopping to remove n_estimators from the search entirely
  3. max_depth + min_child_weight control tree complexity — tune first
  4. subsample + colsample_bytree add variance reduction — tune second
  5. Regularization (gamma, alpha, lambda) fine-tunes overfitting control
  6. Final step: lower lr × more trees often improves the final model
  7. In practice, start with RandomizedSearchCV for exploration, then
     GridSearchCV to refine the most promising region
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "6. XGBoost in a sklearn Pipeline — Production Pattern": {
        "description": "Plug XGBoost into a full sklearn Pipeline with preprocessing, ColumnTransformer, and cross-validation",
        "runnable": True,
        "pipeline_cmd": "xgboost",
        "code": '''
"""
================================================================================
XGBOOST IN A SKLEARN PIPELINE — PRODUCTION PATTERN
================================================================================

XGBoost's sklearn API makes it a drop-in replacement for any sklearn classifier.
This module shows the full production pattern:

    1. Mixed numeric + categorical data (ColumnTransformer)
    2. XGBoost as the final estimator in a Pipeline
    3. Cross-validated evaluation
    4. Handling class imbalance (scale_pos_weight)
    5. Saving and loading the pipeline

This is the recommended pattern for taking XGBoost from notebook to production.
================================================================================
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Create a realistic mixed-type tabular dataset (loan default prediction)
# ─────────────────────────────────────────────────────────────────────────────

n = 800
df = pd.DataFrame({
    # Numeric features
    'age':           np.random.randint(22, 68, n),
    'income':        np.random.normal(55000, 20000, n).clip(10000),
    'loan_amount':   np.random.normal(15000, 8000, n).clip(1000),
    'credit_score':  np.random.normal(680, 80, n).clip(300, 850),
    'debt_ratio':    np.random.uniform(0.1, 0.8, n).round(3),
    'months_employed': np.random.randint(0, 240, n),

    # Categorical features
    'employment_type': np.random.choice(
        ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n,
        p=[0.55, 0.20, 0.15, 0.10]
    ),
    'loan_purpose': np.random.choice(
        ['Home', 'Auto', 'Education', 'Medical', 'Personal'], n
    ),
    'credit_history': np.random.choice(
        ['Excellent', 'Good', 'Fair', 'Poor'], n,
        p=[0.25, 0.40, 0.25, 0.10]
    ),
})

# Introduce missing values (realistic)
for col in ['income', 'credit_score', 'employment_type']:
    mask = np.random.rand(n) < 0.06
    df.loc[mask, col] = np.nan

# Target: loan default (imbalanced — ~15% default rate)
default_prob = (
    0.10
    + 0.15 * (df['debt_ratio'] > 0.6).astype(float)
    + 0.10 * (df['credit_score'].fillna(680) < 600).astype(float)
    - 0.05 * (df['credit_history'] == 'Excellent').astype(float)
    + 0.08 * (df['employment_type'].isin(['Unemployed', 'Part-time'])).astype(float)
)
y = (np.random.rand(n) < default_prob.clip(0.02, 0.90)).astype(int)
X = df.copy()

pos_rate = y.mean()
print("=" * 65)
print("  XGBOOST PRODUCTION PIPELINE")
print("=" * 65)
print(f"""
Dataset: Loan Default Prediction
  Samples: {n}
  Default rate: {pos_rate*100:.1f}% (imbalanced — {y.sum()} defaults, {(1-y).sum()} non-defaults)
  Missing values: {df.isnull().sum().sum()} cells
""")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ─────────────────────────────────────────────────────────────────────────────
# Define feature groups
# ─────────────────────────────────────────────────────────────────────────────

numeric_features     = ['age', 'income', 'loan_amount', 'credit_score',
                         'debt_ratio', 'months_employed']
categorical_features = ['employment_type', 'loan_purpose', 'credit_history']

# NOTE: XGBoost does NOT require scaling (trees are scale-invariant).
# We include StandardScaler here only to show the general pipeline pattern
# and for sklearn API consistency. In practice you can omit it for XGBoost.

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),   # optional for XGBoost, required for SVM/LR
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer,  numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Build the full pipeline
# ─────────────────────────────────────────────────────────────────────────────
# scale_pos_weight: handles class imbalance
#   Set to: sum(negative examples) / sum(positive examples)
#   This tells XGBoost to weight positive (default) examples more heavily.

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw       = neg_count / pos_count

print(f"Class imbalance: {neg_count} non-defaults, {pos_count} defaults")
print(f"scale_pos_weight = {neg_count}/{pos_count} = {spw:.2f}")
print()

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(
        n_estimators     = 300,
        learning_rate    = 0.05,
        max_depth        = 4,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = spw,       # handle class imbalance
        objective        = 'binary:logistic',
        eval_metric      = 'auc',
        random_state     = 42,
        verbosity        = 0,
    ))
])

print("─" * 65)
print("Pipeline structure:")
print("─" * 65)
print(f"""
  [preprocessor]
  ├── [num]  → SimpleImputer(median) → StandardScaler
  │   Cols: {numeric_features[:3]}...
  └── [cat]  → SimpleImputer(most_freq) → OneHotEncoder
      Cols: {categorical_features}
  ↓
  [model] XGBClassifier(scale_pos_weight={spw:.2f})
""")


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate
# ─────────────────────────────────────────────────────────────────────────────

pipeline.fit(X_train, y_train)

y_pred       = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("─" * 65)
print("Test Set Evaluation:")
print("─" * 65)
print(classification_report(y_test, y_pred,
                              target_names=['No Default', 'Default']))
print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc  = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
cv_f1   = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')

print()
print(f"  5-Fold Cross-Validation:")
print(f"  AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  F1:   {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Compare: with vs. without scale_pos_weight
# ─────────────────────────────────────────────────────────────────────────────

print()
print(f"{'─' * 65}")
print("Effect of scale_pos_weight on class imbalance handling:")
print("─" * 65)

for spw_val, label in [(1.0, 'Without (spw=1.0, default)'),
                        (spw,  f'With    (spw={spw:.1f})')]:
    pipe_temp = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            scale_pos_weight=spw_val, random_state=42,
            verbosity=0, eval_metric='logloss'
        ))
    ])
    pipe_temp.fit(X_train, y_train)
    preds  = pipe_temp.predict(X_test)
    probas = pipe_temp.predict_proba(X_test)[:, 1]
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    recall_default = (preds[y_test == 1] == 1).mean()
    print(f"  {label}")
    print(f"    F1={f1:.4f}  AUC={auc:.4f}  Recall(default)={recall_default:.4f}")

print(f"""
  scale_pos_weight improves recall on the minority class (actual defaults).
  This is critical in credit risk: missing a default is more costly than
  a false alarm.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Save and load pipeline
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Saving and loading the pipeline (joblib):")
print("─" * 65)

import joblib
import tempfile, os

with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
    model_path = f.name

joblib.dump(pipeline, model_path)
file_size_kb = os.path.getsize(model_path) / 1024

loaded_pipeline = joblib.load(model_path)
loaded_pred = loaded_pipeline.predict(X_test)
loaded_auc  = roc_auc_score(y_test, loaded_pipeline.predict_proba(X_test)[:, 1])

print(f"""
  Saved to:  {model_path}
  File size: {file_size_kb:.1f} KB
  Loaded AUC (should match original): {loaded_auc:.4f}
  Predictions match: {np.all(loaded_pred == y_pred)}

  In production:
    joblib.dump(pipeline, 'loan_default_model_v1.pkl')   # save
    model = joblib.load('loan_default_model_v1.pkl')      # load
    predictions = model.predict(new_data)                 # inference
""")

os.unlink(model_path)

print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  1. XGBClassifier is a drop-in sklearn estimator — Pipelines just work
  2. ColumnTransformer handles numeric + categorical in one Pipeline step
  3. scale_pos_weight = sum(neg)/sum(pos) — use for imbalanced datasets
  4. XGBoost does NOT require feature scaling (trees are scale-invariant)
     but scaling is harmless and keeps the pipeline general-purpose
  5. cross_val_score on a Pipeline is automatically leak-free per fold
  6. joblib.dump/load persists the ENTIRE pipeline (preprocessor + model)
  7. The same pipeline handles NaN at inference without any imputer refit
""")
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has leading spaces. textwrap.dedent removes the common indent,
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
    visual_html   = ""
    visual_height = 400
    # try:
    #     from frameworks.visuals.xgboost_visual import (
    #         XGBOOST_VISUAL_HTML,
    #         XGBOOST_VISUAL_HEIGHT,
    #     )
    #     visual_html   = XGBOOST_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = XGBOOST_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[xgboost_framework.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    OPERATIONS,
    }