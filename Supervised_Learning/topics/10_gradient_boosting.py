"""
Gradient Boosting — Sequential Error Correction
================================================

Random Forests reduce variance by averaging many independent trees in parallel.
Gradient Boosting takes a fundamentally different approach: it builds trees
sequentially, with each new tree trained to correct the mistakes of the entire
ensemble built so far.

Where Random Forests are a democracy — every tree votes equally — Gradient
Boosting is a committee that learns from its own failures. The result is that
Gradient Boosting achieves state-of-the-art accuracy on virtually every tabular
dataset benchmark, and forms the engine behind XGBoost, LightGBM, and CatBoost —
three of the most successful machine learning libraries in competitive data science.

"""
import base64
import os
import textwrap
import re

TOPIC_NAME = "Gradient Boosting"
DISPLAY_NAME = "10 · Gradient Boosting"
ICON         = "🚀"
SUBTITLE     = "Sequential weak learners correcting each others mistakes — XGBoost, LightGBM"
# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is Gradient Boosting?

Gradient Boosting is an ensemble method that builds an additive model sequentially:
at each step, it fits a new weak learner (almost always a shallow decision tree)
to the negative gradient of the loss function with respect to the current ensemble
prediction. In plain English: each new tree tries to fix what the current ensemble
gets wrong.

Unlike Random Forests — where each tree is independent and parallel — Gradient
Boosting trees are dependent. Tree b cannot be trained until trees 1 through b-1
have been trained, because it needs to know their combined prediction error.

The idea traces to AdaBoost (Freund & Schapire, 1995) and was generalised to the
full gradient framework by Friedman (1999, 2001) in the landmark paper "Greedy
Function Approximation: A Gradient Boosting Machine." Modern implementations
(XGBoost 2014, LightGBM 2017, CatBoost 2017) extend this framework with
second-order approximations, histogram-based splits, and sparse data handling.

    Things that exist inside the model (learned during training):
        - B decision trees (the weak learners), each shallow (depth 1–5)
        - The initial prediction F₀(x) — baseline before any tree
        - Tree weights / learning rate scaling each tree's contribution

    Things you control before training (hyperparameters):
        - n_estimators (B):       number of boosting rounds (trees)
        - learning_rate (α):      shrinkage — scales each tree's contribution
        - max_depth:              depth of each weak learner (usually 1–5)
        - subsample:              fraction of training rows per tree (stochastic GB)
        - min_samples_leaf:       minimum leaf size
        - loss:                   loss function (MSE, MAE, deviance, etc.)


### Gradient Boosting as Empirical Risk Minimisation (ERM)

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Hypothesis class:  H = { additive models F(x) =            │
    │                           F₀ + Σ_b αₜ · hₜ(x) }              │
    │                     where each hₜ is a shallow tree          │
    │                                                             │
    │  Loss function:     L(y, F(x))  — any differentiable loss   │
    │                     MSE, MAE, log loss, Huber, quantile...  │
    │                                                             │
    │  Training objective:                                        │
    │      For each step t = 1...B:                               │
    │        Compute negative gradient rᵢ = −∂L/∂F(xᵢ)            │
    │        Fit a new tree hₜ to predict rᵢ                       │
    │        Update: F_t(x) = F_{t-1}(x) + α · hₜ(x)               │
    │                                                             │
    │  Optimiser:  Gradient descent in FUNCTION SPACE             │
    │              (not weight space — we update predictions      │
    │               directly, not model parameters)               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The key insight: minimising Σ L(yᵢ, F(xᵢ)) is gradient descent where the
"parameters" being updated are the predicted values {F(xᵢ)} themselves, not
the weights of a neural network or the coefficients of a linear model.

Comparing ERM formulations across the series:
    Linear Regression:    MSE,   gradient descent in WEIGHT space (exact)
    Logistic Regression:  BCE,   gradient descent in WEIGHT space
    SVM:                  Hinge, quadratic programming
    Decision Tree:        Gini,  greedy split search (local)
    Random Forest:        0-1,   averaging B independent trees
    Gradient Boosting:    any L, gradient descent in FUNCTION space (sequential)


### The Inductive Bias of Gradient Boosting

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Gradient Boosting encodes the belief that:                 │
    │                                                             │
    │  1. ADDITIVE STRUCTURE — the output is a sum of simple      │
    │     functions (trees). Complex patterns are decomposed      │
    │     into a sequence of simple corrections.                  │
    │                                                             │
    │  2. SEQUENTIAL ERROR REDUCTION — each step specifically     │
    │     targets the current errors of the ensemble; no effort   │
    │     is "wasted" on already-correct examples.                │
    │                                                             │
    │  3. AXIS-ALIGNED LOCAL INTERACTIONS — inherits from         │
    │     decision trees: features interact through conditional   │
    │     splits, not linear combinations.                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


---


### Part 1: The Core Algorithm — Gradient Descent in Function Space


### Why "Gradient" Boosting?

Ordinary gradient descent minimises L(w) by updating parameters:
    wₜ = wₜ₋₁ − α · ∇_w L(w)

Gradient boosting minimises L(F) by updating the prediction FUNCTION:
    Fₜ(x) = Fₜ₋₁(x) − α · ∇_F L(F)

The "gradient" of the loss with respect to the predicted values at each training
point is:
                    rᵢ = −∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)

This is called the pseudo-residual. The new tree hₜ is trained to predict rᵢ —
meaning it learns to correct the errors of the current ensemble in the direction
that most steeply decreases the loss.


### The Pseudo-Residual — What the Next Tree is Fitting:

The specific form of rᵢ depends on the loss function:

    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  Loss:  L(y, F) = (1/2)(y − F)²          (MSE)               │
    │                                                              │
    │  Pseudo-residual:  rᵢ = −∂L/∂F = yᵢ − F(xᵢ)                  │
    │                                                              │
    │  Interpretation: the actual residual — how far off are we?   │
    │  The next tree literally fits the prediction errors.         │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  Loss:  L(y, F) = −[y·log(σ(F)) + (1−y)·log(1−σ(F))]         │
    │                   (Binary Cross-Entropy; F is log-odds)      │
    │                                                              │
    │  Pseudo-residual:  rᵢ = yᵢ − σ(F(xᵢ)) = yᵢ − ŷᵢ              │
    │                                                              │
    │  Interpretation: the difference between the true label and   │
    │  the current predicted probability — same form as MSE!       │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

For MSE loss, pseudo-residuals are exact residuals. For BCE loss, they are the
difference between the true label and the current predicted probability. In both
cases, the new tree is fitting "how much the current ensemble is wrong."


### The Full Algorithm (Friedman 2001):

    ═══════════════════════════════════════════════════════════════
    GRADIENT BOOSTING FOR REGRESSION (MSE loss)
    ═══════════════════════════════════════════════════════════════

    INPUT:  training data {(xᵢ, yᵢ)}, learning rate α, n_estimators B

    Step 1. Initialise with the constant prediction:
            F₀(x) = argmin_γ Σᵢ L(yᵢ, γ) = mean(y)  [for MSE]

    For t = 1 to B:
        Step 2. Compute pseudo-residuals (negative gradient):
                rᵢ = yᵢ − Fₜ₋₁(xᵢ)     [for MSE: actual residuals]

        Step 3. Fit a regression tree hₜ to the pseudo-residuals:
                hₜ = DecisionTree().fit(X, r)

        Step 4. Update the ensemble:
                Fₜ(x) = Fₜ₋₁(x) + α · hₜ(x)

    OUTPUT: F_B(x) = F₀ + α Σ_{t=1}^{B} hₜ(x)

    ═══════════════════════════════════════════════════════════════


    # =======================================================================================# 
    **Diagram 1 — Gradient Boosting Step-by-Step on 1D Regression:**

    STEP 0: Initial prediction = mean(y) = 3.0
    ─────────────────────────────────────────────────────────────
    True values:    y = [1, 2, 3, 4, 5]
    Prediction F₀:     [3, 3, 3, 3, 3]  ← constant
    Residuals r:       [-2, -1, 0, 1, 2] ← what we need to fix

    STEP 1: Tree 1 fits the residuals [-2, -1, 0, 1, 2]
    ─────────────────────────────────────────────────────────────
    Tree 1 learns: if x < 2.5 → predict -1.5,  if x >= 2.5 → predict +1.5
    F₁(x) = F₀ + 0.1 × Tree1(x)
    New prediction: [3 + 0.1×(-1.5), 3 + 0.1×(-1.5), 3 + 0.1×(+1.5), ...]
                  = [2.85, 2.85, 3.15, 3.15, 3.15]
    New residuals: [-1.85, -0.85, -0.15, 0.85, 1.85]  ← smaller!

    STEP 2: Tree 2 fits the NEW residuals [-1.85, -0.85, -0.15, 0.85, 1.85]
    ─────────────────────────────────────────────────────────────
    Each successive tree makes the predictions closer to the true values.
    After B trees, the ensemble approximates the true function well.

    KEY DIFFERENCE FROM RANDOM FOREST:
    RF:  Tree 2 is trained on a RANDOM BOOTSTRAP of the original (x, y) data
    GB:  Tree 2 is trained on the RESIDUALS of Tree 1  ← sequential, not parallel
    # =======================================================================================# 


### Concrete 5-Point Walkthrough:

Let's trace gradient boosting on a tiny regression dataset.

    Dataset (n=5, 1D regression):
    ─────────────────────────────────────────────────────────────
    x        1.0   2.0   3.0   4.0   5.0
    y_true   1.5   3.0   2.5   4.0   3.5
    ─────────────────────────────────────────────────────────────

    Settings: α = 0.5 (learning rate), max_depth = 1 (decision stump)

    ── Initial prediction ──────────────────────────────────────────────
    F₀(x) = mean(y) = (1.5 + 3.0 + 2.5 + 4.0 + 3.5) / 5 = 2.9

    Residuals r = y - F₀:
        x=1: 1.5 − 2.9 = -1.4
        x=2: 3.0 − 2.9 = +0.1
        x=3: 2.5 − 2.9 = -0.4
        x=4: 4.0 − 2.9 = +1.1
        x=5: 3.5 − 2.9 = +0.6
    MSE(F₀) = mean([-1.4², 0.1², -0.4², 1.1², 0.6²]) = 0.710

    ── Iteration 1 ─────────────────────────────────────────────────────
    Fit a stump to (x, r): best split at x ≤ 2.5
        Left  (x ≤ 2.5): r = [-1.4, +0.1] → leaf prediction = -0.65
        Right (x > 2.5):  r = [-0.4, +1.1, +0.6] → leaf prediction = +0.433

    h₁(x) = -0.65 if x ≤ 2.5 else +0.433

    F₁(x) = F₀ + 0.5 × h₁(x):
        x=1: 2.9 + 0.5 × (-0.65) = 2.575   (was 2.9, true = 1.5)
        x=2: 2.9 + 0.5 × (-0.65) = 2.575   (was 2.9, true = 3.0)
        x=3: 2.9 + 0.5 × (+0.433) = 3.117  (was 2.9, true = 2.5)
        x=4: 2.9 + 0.5 × (+0.433) = 3.117  (was 2.9, true = 4.0)
        x=5: 2.9 + 0.5 × (+0.433) = 3.117  (was 2.9, true = 3.5)

    New residuals:
        x=1: 1.5 − 2.575 = -1.075
        x=2: 3.0 − 2.575 = +0.425
        x=3: 2.5 − 3.117 = -0.617
        x=4: 4.0 − 3.117 = +0.883
        x=5: 3.5 − 3.117 = +0.383
    MSE(F₁) = mean([1.075², 0.425², 0.617², 0.883², 0.383²]) = 0.430  ← decreased!

    ── Iteration 2 ─────────────────────────────────────────────────────
    Fit a stump to the NEW residuals (the x=1 example still has large error)...
    [continued for B iterations]

    After 100 iterations, the ensemble closely approximates all 5 true values.


---


### Part 2: The Learning Rate and the Shrinkage-Trees Tradeoff


### Why the Learning Rate Matters:

Adding each tree's full prediction would be aggressive — the ensemble would
overfit to the first few trees before the later ones can contribute. The learning
rate α (also called shrinkage) scales each tree's contribution:

                    Fₜ(x) = Fₜ₋₁(x) + α · hₜ(x)

Small α means each tree contributes a small amount — the model takes many small
steps rather than a few large ones. This is gradient descent with small step size.

'''
**Diagram 2 — Learning Rate: Small Steps vs Large Steps:**

HIGH α (e.g., 0.9):                LOW α (e.g., 0.01):

Loss ↑                              Loss ↑
   │╲                                  │╲
   │  ╲                                │  ╲
   │    ╲___                           │    ╲
   │        ‾‾─                        │      ╲____
   │           ‾─ (converged fast      │           ╲_____
   │              but oscillating)     │                  ‾‾──── (stable)
   └──────────────→ Iterations         └──────────────────────→ Iterations
   Fewer trees needed,                 More trees needed,
   worse generalisation                better generalisation

The Shrinkage-Trees Tradeoff:
─────────────────────────────────────────────────────────────
Low α + High B → slow but reliable convergence, better generalisation
High α + Low B → fast but coarse, risk of overfitting early trees
─────────────────────────────────────────────────────────────
RULE: set α small (0.01–0.1), then find optimal B via early stopping.

'''

### Early Stopping:

Unlike Random Forests (where more trees always help), Gradient Boosting can
overfit with too many trees. The validation loss follows a U-curve:

    B = 10:    underfitting — ensemble hasn't captured the pattern
    B = 100:   sweet spot — good bias-variance balance
    B = 1000:  overfitting — later trees fit training noise

Early stopping monitors validation loss during training and stops when it starts
increasing. This elegantly combines hyperparameter selection (B) with regularisation.

    ┌─────────────────────────────────────────────────────────────┐
    │  Early stopping procedure:                                  │
    │    1. Split data: train set + validation set                │
    │    2. Add trees one by one, monitoring val_loss             │
    │    3. Stop when val_loss has not improved in k rounds       │
    │       (k = "n_iter_no_change" in sklearn)                   │
    │    4. Use the tree count at the best val_loss point         │
    └─────────────────────────────────────────────────────────────┘


---


### Part 3: Loss Functions — Gradient Boosting is Loss-Agnostic


One of Gradient Boosting's great strengths is that any differentiable loss function
can be plugged in — the algorithm only needs the gradient (pseudo-residuals), not
a closed form solution.

    ────────────────────────────────────────────────────────────────────────────────
    Task                Loss L(y, F)                 Pseudo-residual rᵢ
    ────────────────────────────────────────────────────────────────────────────────
    Regression (MSE)    (1/2)(y − F)²               yᵢ − F(xᵢ)
    Regression (MAE)    |y − F|                      sign(yᵢ − F(xᵢ))
    Regression (Huber)  Huber(y − F, δ)              clipped residual
    Regression (Quant.) ρ_τ(y − F)                   τ − 1[yᵢ < F(xᵢ)]
    Classification      BCE: −[y·log(σ(F)) + ...]   yᵢ − σ(F(xᵢ))
    Multiclass          Categorical cross-entropy     yᵢⱼ − softmax(F)ⱼ
    Ranking             LambdaMART, pairwise loss     gradient of NDCG
    ────────────────────────────────────────────────────────────────────────────────

**MSE vs MAE for Gradient Boosting:**
    MSE pseudo-residual = exact residual → tree fits toward outliers
    MAE pseudo-residual = sign(residual) → tree fits toward median, ignores outlier scale
    Huber loss = MSE for small residuals, MAE for large → robust to outliers, smooth near zero


    **Diagram 3 — Pseudo-Residuals Under Different Losses:**

    TRUE FUNCTION: y = sin(x) + OUTLIER at x=3 (y=10, true ≈ 0.14)
    Current prediction F(x): a flat line at 0

    RESIDUALS AT x=3:
    ────────────────────────────────────────────────────────────────────────
    MSE:    rᵢ = y − F = 10 − 0 = 10.0   ← tree will STRONGLY chase outlier
    MAE:    rᵢ = sign(10 − 0) = +1.0     ← tree knows direction, not magnitude
    Huber:  rᵢ = δ × sign(10) = 1.0      ← capped at δ, ignores scale
    ────────────────────────────────────────────────────────────────────────

    With MSE: the outlier at x=3 will "pull" many trees toward it,
    distorting the ensemble for all other predictions.

    With MAE/Huber: the outlier only gets a small gradient signal — it's
    treated as just one more point to be improved, not a priority.

    Use Huber or MAE when you have outliers in the target variable.

---

### Part 4: Regularisation in Gradient Boosting


Gradient Boosting has more regularisation levers than any other model we've studied:

**1. Learning Rate (α):** Small α shrinks each tree's contribution. Combined with
more trees (high B), this produces a smoother, less overfit model.

**2. Tree depth (max_depth):** Shallow trees (depth 1–3) are weak learners — high
bias, low variance. GB is supposed to use many weak learners and let boosting do
the heavy lifting. Deep trees risk fitting residuals too aggressively.

**3. Subsampling (stochastic gradient boosting):** Train each tree on a random
subset of training rows (analogous to bootstrap in RF, but without replacement).
    - Reduces correlation between trees (like RF's feature subsampling)
    - Acts as regularisation
    - Speeds up training
    - subsample=0.8 means each tree uses 80% of training rows

**4. Column subsampling (XGBoost feature):** Randomly sample features at each tree
or split level — directly borrowed from Random Forests.

**5. L1/L2 regularisation on leaf weights (XGBoost):** Adds λ·w² + α·|w| penalty
to each leaf prediction, shrinking small leaves toward zero. This is tree-level
regularisation beyond what depth controls.

**6. min_samples_leaf:** Prevents trees from fitting tiny subsets of residuals
(which are almost certainly noise).


### The Regularisation Hierarchy:

    Most impact                                                    Least impact
    ────────────────────────────────────────────────────────────────────────────
    learning_rate  →  n_estimators  →  max_depth  →  subsample  →  min_samples_leaf
    ────────────────────────────────────────────────────────────────────────────

The learning_rate × n_estimators interaction is the dominant regularisation force.
Always tune these two together (small α, find optimal B with early stopping).


---


### Part 5: XGBoost, LightGBM, CatBoost — Modern Implementations


### Why the Speedups Were Needed:

Sklearn's `GradientBoostingClassifier` is correct but slow — it processes splits
one by one and builds trees sequentially with no parallelism inside each tree.
For large datasets (n > 100k), training times were prohibitive.

The three major modern implementations each solve this with different approaches:


### XGBoost (Chen & Guestrin, 2016):

The breakthrough paper introduced second-order gradient information:

    ┌─────────────────────────────────────────────────────────────┐
    │  XGBoost uses both gradient (gᵢ) AND Hessian (hᵢ):          │
    │                                                             │
    │  gᵢ = ∂L/∂F(xᵢ)    (first derivative — the gradient)        │
    │  hᵢ = ∂²L/∂F(xᵢ)²  (second derivative — the curvature)      │
    │                                                             │
    │  Leaf value:  w* = −Σgᵢ / (Σhᵢ + λ)                         │
    │  Split gain:  ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]     │
    │               − γ   (γ = leaf penalty term)                 │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Using the Hessian allows XGBoost to compute optimal leaf values analytically
and score splits more precisely than sklearn's first-order gradient alone.
Additional features: sparse-aware splits, column sampling, cache-efficient tree
building, built-in cross-validation, and early stopping.


### LightGBM (Ke et al., 2017):

Two algorithmic innovations make LightGBM significantly faster than XGBoost:

    GOSS (Gradient-based One-Side Sampling):
        Keep all training examples with large gradients (high loss, hard to predict).
        Randomly sample a fraction of examples with small gradients (easy examples).
        Weight the sampled examples to preserve gradient statistics.
        Effect: focus computation on the hardest examples, skip the easy ones.

    EFB (Exclusive Feature Bundling):
        Many real-world datasets are sparse (especially one-hot encoded categoricals).
        Features that rarely take nonzero values simultaneously can be merged into
        one "bundle" without loss of information.
        Effect: reduce the effective number of features → faster split search.

    Histogram-based splits: instead of sorting feature values to find thresholds,
    bin continuous features into histograms (256 bins default). Find the best split
    bin instead of best exact threshold. Dramatically faster for large n.


### CatBoost (Prokhorenkova et al., 2018):

CatBoost's key innovation is Ordered Boosting — a way to prevent target leakage
when computing statistics for categorical features:

    Problem: using the target variable to encode categorical features (e.g.,
    mean target per category) introduces leakage — features become correlated with
    the target, inflating performance estimates.

    CatBoost fix: for each training example, compute target statistics using only
    examples seen earlier in a random permutation of the training data. This
    eliminates the leakage while preserving the useful target encoding signal.

    CatBoost also natively handles categorical features without preprocessing
    and uses symmetric (oblivious) trees — the same split threshold is applied
    at all nodes of the same depth level — which are faster to evaluate and
    tend to be less overfit.


**Diagram 4 — Library Comparison:**

────────────────────────────────────────────────────────────────────────────
Feature                 sklearn GB    XGBoost       LightGBM    CatBoost
────────────────────────────────────────────────────────────────────────────
Year                    2007          2014          2017        2017
Second-order gradient   No            Yes           Yes         Yes
Histogram splits        No            Approx.       Yes         Yes
Sparse support          No            Yes           Yes         Yes
Native categoricals     No            No            Partial     Yes (key!)
Column subsampling      No            Yes           Yes         Yes
Row subsampling         Yes           Yes           GOSS        Yes
Early stopping          Yes           Yes           Yes         Yes
GPU support             No            Yes           Yes         Yes
Training speed          Slow          Fast          Fastest     Fast
Accuracy (typical)      Good          Excellent     Excellent   Excellent
API compatibility       sklearn       sklearn-like  sklearn-like sklearn-like
────────────────────────────────────────────────────────────────────────────

---

### Part 6: Gradient Boosting vs Random Forest — The Full Picture


Both are ensembles of decision trees. The fundamental difference is in HOW they combine trees:

    Random Forest: PARALLEL, independent trees → average → reduces VARIANCE
    Gradient Boosting: SEQUENTIAL, dependent trees → sum → reduces BIAS + VARIANCE

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  RF:    Train B trees independently on bootstrap samples    │
    │         Each tree sees random data, random features         │
    │         Prediction = average (regression) / vote (class)    │
    │         Bias = single tree bias                             │
    │         Variance = ρ·σ² + (1-ρ)·σ²/B → shrinks with B       │
    │                                                             │
    │  GB:    Train B trees sequentially on pseudo-residuals      │
    │         Each tree sees ALL data (or subsampled rows)        │
    │         Prediction = sum of α·hₜ(x)                          │
    │         Bias decreases with B (bias reduction is the goal!) │
    │         Variance can increase with B (→ need regularisation)│
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


### When to Use Which:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Use Random Forest when:                                             │
    │    • Fast training and prediction required                           │
    │    • Minimal hyperparameter tuning budget                            │
    │    • Need OOB error estimate (no separate validation set)            │
    │    • Dataset is noisy (RF more robust with default settings)         │
    │    • Parallel training available (RF is embarrassingly parallel)     │
    │                                                                      │
    │  Use Gradient Boosting (XGBoost/LightGBM) when:                      │
    │    • Maximum accuracy is the priority (Kaggle competitions)          │
    │    • Dataset is medium-to-large (LightGBM scales beautifully)        │
    │    • You can afford hyperparameter tuning time                       │
    │    • Custom loss functions needed (ranking, quantile, etc.)          │
    │    • Tabular data benchmark: GBT almost always wins                  │
    └──────────────────────────────────────────────────────────────────────┘


### Where Gradient Boosting Sits in the Full Progression:

    Perceptron            ← hard linear, no probability
          │
          ↓
    Logistic Regression   ← soft linear, probabilistic, BCE loss
          │
          ↓
    SVM                   ← max-margin, kernel trick
          │
          ↓
    Decision Tree         ← non-linear, axis-aligned, interpretable
          │
          ↓
    Random Forest         ← parallel ensemble, reduces variance
          │
          ↓
    Gradient Boosting     ← sequential ensemble, reduces bias + variance
          │                 state-of-the-art on tabular data
          │
          ↓
    Neural Networks       ← learns feature representations,
                            state-of-the-art on images/text/audio

    """

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ──────────────────────────────────────────────────────────────────────────
    Operation                   Complexity             Notes
    ──────────────────────────────────────────────────────────────────────────
    sklearn GB training         O(B · n · p · log n)   sequential, no parallelism
    XGBoost training            O(B · n · p · log n)   parallelised within each tree
    LightGBM training           O(B · n · K · log n)   K=256 bins (histogram)
    Prediction (all GB)         O(B · depth)            fast; B trees evaluated
    Memory (sklearn GB)         O(B · 2^depth)          B full trees
    Memory (LightGBM)           O(n · K)                histogram-based
    ──────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Gradient Boosting from Scratch": {
        "description": "Implement GB for regression (MSE) — pseudo-residuals, stumps, sequential updates",
        "runnable": True,
        "code": '''
"""
================================================================================
GRADIENT BOOSTING FROM SCRATCH — MSE REGRESSION
================================================================================

We implement Gradient Boosting Regression using:
    - MSE loss: L(y, F) = (1/2)(y - F)²
    - Pseudo-residuals: rᵢ = yᵢ - F(xᵢ)  (exact residuals for MSE)
    - Weak learners: decision stumps (max_depth=1)
    - Learning rate: α (shrinkage)

The algorithm:
    F₀ = mean(y)
    For t = 1 to B:
        r = y - F_{t-1}(x)         # compute residuals
        h_t = fit_stump(X, r)      # fit stump to residuals
        F_t = F_{t-1} + α * h_t   # update ensemble

================================================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


# =============================================================================
# GRADIENT BOOSTING REGRESSOR
# =============================================================================

class GradientBoostingRegressorScratch:
    """
    Gradient Boosting for regression with MSE loss.

    For MSE, pseudo-residuals = exact residuals = y - F(x).
    Each tree fits these residuals directly.

    Parameters:
        n_estimators (int):   number of boosting rounds
        learning_rate (float): shrinkage — scales each tree's contribution
        max_depth (int):       depth of each weak learner (1 = stumps)
        min_samples_leaf (int): minimum leaf size (regularisation)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=1, min_samples_leaf=1):
        self.B            = n_estimators
        self.alpha        = learning_rate
        self.max_depth    = max_depth
        self.min_sl       = min_samples_leaf
        self.trees_       = []
        self.F0_          = None     # initial prediction (mean of y)
        self.train_mses_  = []

    def fit(self, X, y, verbose=True):
        """
        Train the gradient boosting ensemble on (X, y).

        Tracks MSE at each iteration for learning curve visualisation.
        """
        n = len(y)

        # ── Step 0: Initial prediction ────────────────────────────────────
        # For MSE, the constant minimiser is the mean
        self.F0_ = y.mean()
        F = np.full(n, self.F0_)   # current ensemble prediction, shape (n,)

        mse_init = np.mean((y - F) ** 2)
        self.train_mses_.append(mse_init)

        if verbose:
            print(f"  Initial (mean prediction = {self.F0_:.4f}):  MSE = {mse_init:.4f}")

        # ── Boosting iterations ───────────────────────────────────────────
        for t in range(self.B):

            # Step 1: Compute pseudo-residuals (negative gradient of MSE)
            # For MSE: r = y - F  (the exact residuals)
            r = y - F

            # Step 2: Fit a regression tree to the pseudo-residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_sl
            )
            tree.fit(X, r)
            self.trees_.append(tree)

            # Step 3: Update ensemble prediction
            F = F + self.alpha * tree.predict(X)

            # Track MSE
            mse = np.mean((y - F) ** 2)
            self.train_mses_.append(mse)

            if verbose and (t + 1) % 20 == 0:
                print(f"  Iter {t+1:>4}:  MSE = {mse:.4f}  |residual|_max = {np.abs(r).max():.4f}")

        return self

    def predict(self, X):
        """Predict by summing contributions from all trees."""
        F = np.full(X.shape[0], self.F0_)
        for tree in self.trees_:
            F = F + self.alpha * tree.predict(X)
        return F

    def staged_predict(self, X):
        """Yield predictions after each tree (for learning curves)."""
        F = np.full(X.shape[0], self.F0_)
        yield F.copy()
        for tree in self.trees_:
            F = F + self.alpha * tree.predict(X)
            yield F.copy()


# =============================================================================
# DEMO 1: TOY 1D REGRESSION — VISUALISING RESIDUALS
# =============================================================================

np.random.seed(42)

print("=" * 65)
print("  GRADIENT BOOSTING FROM SCRATCH")
print("=" * 65)

# True function: y = sin(x)
X_1d = np.sort(np.random.uniform(0, 6, 80)).reshape(-1, 1)
y_1d = np.sin(X_1d.ravel()) + np.random.normal(0, 0.15, 80)

print(f"\\n  Dataset: y = sin(x) + noise  (n=80, 1D)")

model = GradientBoostingRegressorScratch(
    n_estimators=100, learning_rate=0.1, max_depth=2
)
model.fit(X_1d, y_1d)

test_mse = np.mean((y_1d - model.predict(X_1d)) ** 2)
print(f"\\n  Final train MSE: {test_mse:.4f}")
print(f"  Initial MSE:     {model.train_mses_[0]:.4f}")
print(f"  MSE reduction:   {100*(1 - test_mse/model.train_mses_[0]):.1f}%")


# =============================================================================
# DEMO 2: STEP-BY-STEP RESIDUAL TRACE (TINY DATASET)
# =============================================================================

print("\\n" + "=" * 65)
print("  STEP-BY-STEP: RESIDUALS AFTER EACH TREE")
print("=" * 65)

# Match the 5-point example from the THEORY section
X_tiny = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_tiny = np.array([1.5,   3.0,   2.5,   4.0,   3.5])

print(f"""
  Dataset: x=[1,2,3,4,5], y=[1.5, 3.0, 2.5, 4.0, 3.5]
  Settings: α=0.5, max_depth=1 (decision stumps)
""")

gb_tiny = GradientBoostingRegressorScratch(
    n_estimators=5, learning_rate=0.5, max_depth=1, min_samples_leaf=1
)
gb_tiny.fit(X_tiny, y_tiny, verbose=False)

# Print iteration-by-iteration
F = np.full(5, y_tiny.mean())
print(f"  F₀ = mean(y) = {y_tiny.mean():.3f}")
print(f"")
print(f"  {'x':>4}  {'y':>6}  {'F₀':>6}  {'r₀':>6}")
for i, (xi, yi) in enumerate(zip(X_tiny.ravel(), y_tiny)):
    print(f"  {xi:>4.1f}  {yi:>6.2f}  {F[i]:>6.3f}  {yi-F[i]:>+6.3f}")

for t, tree in enumerate(gb_tiny.trees_):
    r = y_tiny - F
    h_pred = tree.predict(X_tiny)
    F_new  = F + 0.5 * h_pred
    mse    = np.mean((y_tiny - F_new) ** 2)
    print(f"\\n  Iteration {t+1}: fit stump to residuals → update F")
    print(f"  {'x':>4}  {'r (residual)':>13}  {'stump pred':>11}  {'F_new':>7}  {'new_r':>8}")
    for i in range(5):
        print(f"  {X_tiny[i,0]:>4.1f}  {r[i]:>+13.3f}  {h_pred[i]:>+11.3f}  {F_new[i]:>7.3f}  {y_tiny[i]-F_new[i]:>+8.3f}")
    print(f"  MSE = {mse:.4f}")
    F = F_new


# =============================================================================
# DEMO 3: LEARNING RATE COMPARISON
# =============================================================================

print("\\n" + "=" * 65)
print("  EFFECT OF LEARNING RATE ON CONVERGENCE")
print("=" * 65)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_r, y_r = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.25, random_state=42)

print(f"\\n  After 100 trees:")
print(f"  {'Learning Rate':>15} | {'Train MSE':>10} | {'Test MSE':>10}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")

for lr in [0.5, 0.2, 0.1, 0.05, 0.01]:
    gb = GradientBoostingRegressorScratch(100, learning_rate=lr, max_depth=2)
    gb.fit(X_tr, y_tr, verbose=False)
    tr_mse = np.mean((y_tr - gb.predict(X_tr)) ** 2)
    te_mse = np.mean((y_te - gb.predict(X_te)) ** 2)
    print(f"  {lr:>15.2f} | {tr_mse:>10.2f} | {te_mse:>10.2f}")

print(f"""
  With α=0.5: few trees converge quickly but to a coarser solution
  With α=0.01: need many more trees but each step is more controlled

  In practice: use small α (0.05–0.1) and increase n_estimators to compensate.
  The "optimal" α is typically found via cross-validation or early stopping.
""")
''',
    },

    "Loss Functions and Pseudo-Residuals": {
        "description": "MSE, MAE, Huber, and log-loss — how gradient changes with each loss function",
        "runnable": True,
        "code": '''
"""
================================================================================
LOSS FUNCTIONS IN GRADIENT BOOSTING
================================================================================

Gradient Boosting's power lies in its flexibility: swap the loss function,
and you change what the pseudo-residuals look like — and therefore what each
tree is trying to predict.

This script demonstrates:
    1. Pseudo-residual forms under MSE, MAE, and Huber loss
    2. How outliers affect GB with different losses
    3. Quantile regression via the pinball loss
    4. Classification with log loss

================================================================================
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: PSEUDO-RESIDUALS UNDER DIFFERENT LOSSES
# =============================================================================

print("=" * 65)
print("  PART 1: PSEUDO-RESIDUALS — WHAT EACH TREE IS FITTING")
print("=" * 65)

print("""
  For a single training example with true label y and current prediction F:

  Loss L(y, F)          │  Pseudo-residual rᵢ = −∂L/∂F
  ──────────────────────┼──────────────────────────────────────────
  MSE: (y−F)²/2         │  rᵢ = y − F        (exact residual)
  MAE: |y−F|            │  rᵢ = sign(y − F)  (only direction, not magnitude)
  Huber(δ=1):           │  rᵢ = y−F  if |y−F| ≤ δ
                        │  rᵢ = δ·sign(y−F)  if |y−F| > δ
  BCE: cross-entropy    │  rᵢ = y − σ(F)     (label minus probability)
  ──────────────────────┴──────────────────────────────────────────
""")

print("  Numerical example: y = 2.0, various F values")
print(f"  {'F (prediction)':>17} | {'MSE residual':>14} | {'MAE residual':>14} | {'Huber(δ=1)':>12}")
print(f"  {'-'*17}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")

delta = 1.0
for F in [5.0, 3.0, 2.5, 2.0, 1.5, 1.0, -1.0]:
    y  = 2.0
    r_mse   = y - F
    r_mae   = np.sign(y - F)
    diff    = y - F
    r_huber = diff if abs(diff) <= delta else delta * np.sign(diff)
    print(f"  {F:>17.1f} | {r_mse:>+14.3f} | {r_mae:>+14.3f} | {r_huber:>+12.3f}")

print(f"""
  Key insight:
    MSE residual grows linearly with error — outliers dominate gradient
    MAE residual is always ±1 — outliers get same weight as small errors
    Huber residual = MSE for small errors, capped at δ for large errors
    (Huber is the best of both worlds)
""")


# =============================================================================
# PART 2: OUTLIER ROBUSTNESS — MSE vs HUBER vs MAE
# =============================================================================

print("=" * 65)
print("  PART 2: OUTLIER ROBUSTNESS")
print("=" * 65)

# Clean dataset: y = 2x + noise
n = 200
X_clean = np.random.uniform(0, 5, (n, 1))
y_clean = 2 * X_clean.ravel() + np.random.normal(0, 0.5, n)

# Add 10 severe outliers
X_out = X_clean.copy()
y_out = y_clean.copy()
outlier_idx = np.random.choice(n, 10, replace=False)
y_out[outlier_idx] += np.random.uniform(20, 30, 10)  # extreme positive outliers

X_tr, X_te, y_tr, y_te = train_test_split(X_out, y_out, test_size=0.25, random_state=42)

# True test values (without outliers)
y_te_clean = 2 * X_te.ravel()   # the true function

print(f"  Training set: n={len(X_tr)}, ~10 outliers with y_offset = 20-30")
print(f"  Evaluating MSE and MAE on CLEAN test values (no outliers):")
print(f"")
print(f"  {'Loss function':>20} | {'Train MSE':>10} | {'Test MAE':>10} | {'Note'}")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}")

for loss_name, loss_str in [
    ("MSE (squared)",  "squared_error"),
    ("MAE (absolute)", "absolute_error"),
    ("Huber",          "huber"),
]:
    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=2,
        loss=loss_str, random_state=42
    )
    gb.fit(X_tr, y_tr)
    tr_mse = mean_squared_error(y_tr, gb.predict(X_tr))
    te_mae = mean_absolute_error(y_te_clean, gb.predict(X_te))
    note   = "outliers dominate" if loss_name.startswith("MSE") else "robust to outliers"
    print(f"  {loss_name:>20} | {tr_mse:>10.2f} | {te_mae:>10.4f} | {note}")

print(f"""
  MSE: outliers have huge gradients → many trees chase them → worse overall fit
  MAE: outliers have gradient = ±1 → treated like any other error → better fit
  Huber: clips large gradients → robust like MAE but smooth near zero (stable)

  Rule: use Huber or MAE when target variable has outliers.
  Use MSE when the residual distribution is roughly Gaussian.
""")


# =============================================================================
# PART 3: QUANTILE REGRESSION — PREDICTING INTERVALS
# =============================================================================

print("=" * 65)
print("  PART 3: QUANTILE REGRESSION")
print("=" * 65)

print("""
  GB with the pinball (quantile) loss can predict arbitrary quantiles:

    Loss(y, F, τ) = τ · max(0, y − F) + (1−τ) · max(0, F − y)

  For τ=0.5:  median regression
  For τ=0.1:  predict the 10th percentile (lower bound)
  For τ=0.9:  predict the 90th percentile (upper bound)

  This gives a prediction INTERVAL [Q₀.₁, Q₀.₉] — no parametric assumption!
""")

from sklearn.ensemble import GradientBoostingRegressor

# Dataset with heteroscedastic noise (variance grows with x)
X_q = np.sort(np.random.uniform(0, 5, 300)).reshape(-1, 1)
y_q = np.sin(X_q.ravel()) + np.random.normal(0, 0.1 + 0.3 * X_q.ravel())

X_q_tr, X_q_te, y_q_tr, y_q_te = train_test_split(X_q, y_q, test_size=0.3, random_state=42)

median_gb = GradientBoostingRegressor(loss="quantile", alpha=0.5,
                                       n_estimators=100, learning_rate=0.1,
                                       max_depth=2, random_state=42)
lower_gb  = GradientBoostingRegressor(loss="quantile", alpha=0.1,
                                       n_estimators=100, learning_rate=0.1,
                                       max_depth=2, random_state=42)
upper_gb  = GradientBoostingRegressor(loss="quantile", alpha=0.9,
                                       n_estimators=100, learning_rate=0.1,
                                       max_depth=2, random_state=42)

median_gb.fit(X_q_tr, y_q_tr)
lower_gb.fit(X_q_tr, y_q_tr)
upper_gb.fit(X_q_tr, y_q_tr)

pred_median = median_gb.predict(X_q_te)
pred_lower  = lower_gb.predict(X_q_te)
pred_upper  = upper_gb.predict(X_q_te)

coverage = np.mean((y_q_te >= pred_lower) & (y_q_te <= pred_upper))
avg_width = np.mean(pred_upper - pred_lower)
median_mae = mean_absolute_error(y_q_te, pred_median)

print(f"  Results on test set:")
print(f"    Median (τ=0.5) MAE:           {median_mae:.4f}")
print(f"    80% prediction interval:")
print(f"      Coverage (should be ~80%):  {coverage:.1%}")
print(f"      Average interval width:     {avg_width:.4f}")
print(f"""
  The 80% interval [Q₀.₁, Q₀.₉] covers {coverage:.0%} of test examples.
  A perfectly calibrated interval covers exactly 80%.

  This is prediction interval (uncertainty quantification) without ANY 
  assumption about the noise distribution — purely data-driven.

  Other frameworks (linear regression) require Gaussian noise for intervals.
  Gradient Boosting with quantile loss makes NO such assumption.
""")


# =============================================================================
# PART 4: CLASSIFICATION — LOG LOSS AND PSEUDO-RESIDUALS
# =============================================================================

print("=" * 65)
print("  PART 4: CLASSIFICATION — LOG LOSS PSEUDO-RESIDUALS")
print("=" * 65)

print("""
  For binary classification, GB uses:
    Loss: BCE = -[y·log(σ(F)) + (1-y)·log(1-σ(F))]
    Pseudo-residual: rᵢ = yᵢ - σ(F(xᵢ)) = yᵢ - ŷᵢ

  F(x) is the log-odds (not probability). σ(F) is the probability.
  The residual is (true label) - (current predicted probability).
  This is identical in form to the BCE gradient in logistic regression!

  Demonstration: sklearn GradientBoostingClassifier
""")

from sklearn.datasets import make_classification

X_cls, y_cls = make_classification(n_samples=500, n_features=10,
                                    n_informative=5, random_state=42)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cls, y_cls,
                                                     test_size=0.25, random_state=42)

print(f"  Training accuracy and test accuracy vs. n_estimators:")
print(f"  {'B':>6} | {'Train Acc':>10} | {'Test Acc':>10} | {'Train Loss':>12} | Overfitting?")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*13}")

for B in [1, 5, 20, 50, 100, 200, 500]:
    gb_cls = GradientBoostingClassifier(
        n_estimators=B, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb_cls.fit(X_tr_c, y_tr_c)
    tr_acc = accuracy_score(y_tr_c, gb_cls.predict(X_tr_c))
    te_acc = accuracy_score(y_te_c, gb_cls.predict(X_te_c))
    tr_loss = gb_cls.train_score_[-1]
    overfit = "YES" if (tr_acc - te_acc) > 0.08 else "no"
    print(f"  {B:>6} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {tr_loss:>12.4f} | {overfit}")

print(f"""
  Pattern:
    B too small (1-20): underfitting — ensemble hasn't learned enough
    B around 100-200:   sweet spot — good test accuracy
    B too large (500+): overfitting begins — training accuracy → 1.0 but test drops

  Solution: early stopping or cross-validate B.
  Unlike Random Forests, more trees in GB can hurt!
""")
''',
    },

    "XGBoost and Early Stopping": {
        "description": "XGBoost second-order gradients, regularisation, and early stopping for optimal B",
        "runnable": True,
        "code": '''
"""
================================================================================
XGBOOST, EARLY STOPPING, AND HYPERPARAMETER TUNING
================================================================================

This script demonstrates:
    1. XGBoost vs sklearn GradientBoosting — speed and accuracy comparison
    2. Early stopping — finding optimal n_estimators automatically
    3. The learning rate × n_estimators interaction
    4. Key hyperparameters and their effects

================================================================================
"""

import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, log_loss

np.random.seed(42)

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            n_redundant=4, class_sep=0.8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


# =============================================================================
# PART 1: SKLEARN GRADIENT BOOSTING — LEARNING CURVE
# =============================================================================

print("=" * 65)
print("  PART 1: LEARNING CURVE — SKLEARN GRADIENT BOOSTING")
print("=" * 65)

# Use staged_predict to get predictions at each iteration
gb_full = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=3,
    subsample=0.8, random_state=42
)
gb_full.fit(X_tr, y_tr)

print(f"\\n  Learning curve (test log-loss at each boosting round):")
print(f"  {'B':>6} | {'Train LogLoss':>14} | {'Test LogLoss':>14} | Overfitting?")
print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*13}")

# Collect staged scores
stages = list(range(0, 300, 30)) + [299]
probs_tr_staged = list(gb_full.staged_predict_proba(X_tr))
probs_te_staged = list(gb_full.staged_predict_proba(X_te))

best_te_logloss = float("inf")
best_B = 0
for B in stages:
    tr_ll = log_loss(y_tr, probs_tr_staged[B])
    te_ll = log_loss(y_te, probs_te_staged[B])
    if te_ll < best_te_logloss:
        best_te_logloss = te_ll
        best_B = B + 1
    flag = "YES" if te_ll > best_te_logloss + 0.02 else "no"
    print(f"  {B+1:>6} | {tr_ll:>14.4f} | {te_ll:>14.4f} | {flag}")

print(f"\\n  Best test log-loss at B={best_B} trees.")
print(f"  Training more trees beyond this point increases test loss (overfitting).")


# =============================================================================
# PART 2: EARLY STOPPING
# =============================================================================

print("\\n" + "=" * 65)
print("  PART 2: EARLY STOPPING")
print("=" * 65)

print(f"""
  Early stopping monitors a validation set and stops when the loss stops improving.
  sklearn GradientBoosting supports this via n_iter_no_change.
""")

from sklearn.model_selection import train_test_split

X_tr_main, X_val, y_tr_main, y_val = train_test_split(
    X_tr, y_tr, test_size=0.15, random_state=0
)

# Without early stopping — runs all 500 rounds
gb_no_es = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=3,
    subsample=0.8, random_state=42
)
t0 = time.perf_counter()
gb_no_es.fit(X_tr_main, y_tr_main)
t_no_es = time.perf_counter() - t0

# With early stopping
gb_es = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=3,
    subsample=0.8,
    n_iter_no_change=15,     # stop if no improvement for 15 rounds
    validation_fraction=0.1, # fraction of training data for early stopping
    tol=1e-4,                # minimum improvement to count as progress
    random_state=42
)
t0 = time.perf_counter()
gb_es.fit(X_tr_main, y_tr_main)
t_es = time.perf_counter() - t0

acc_no_es = accuracy_score(y_te, gb_no_es.predict(X_te))
acc_es    = accuracy_score(y_te, gb_es.predict(X_te))

print(f"  {'':>30} | {'n_estimators used':>18} | {'Test Acc':>10} | {'Time (ms)':>10}")
print(f"  {'-'*30}-+-{'-'*18}-+-{'-'*10}-+-{'-'*10}")
print(f"  {'Without early stopping':>30} | {gb_no_es.n_estimators_:>18} | {acc_no_es:>10.4f} | {t_no_es*1000:>10.1f}")
print(f"  {'With early stopping':>30} | {gb_es.n_estimators_:>18} | {acc_es:>10.4f} | {t_es*1000:>10.1f}")

print(f"""
  Early stopping used only {gb_es.n_estimators_} trees instead of 500,
  while achieving comparable accuracy. Significant speed improvement.

  n_iter_no_change=15: stop if test loss doesn't improve for 15 rounds.
  This is the most practical way to set n_estimators — don't guess, let
  early stopping find it automatically.
""")


# =============================================================================
# PART 3: THE α × B INTERACTION (SHRINKAGE-TREES TRADEOFF)
# =============================================================================

print("=" * 65)
print("  PART 3: LEARNING RATE × N_ESTIMATORS INTERACTION")
print("=" * 65)

print(f"""
  Low α + High B = slow steps, many corrections = good regularisation
  High α + Low B = big steps, few trees = fast but coarse

  Both can achieve similar accuracy — the tradeoff is training cost vs quality.
  In Kaggle competitions: α=0.01 with B=5000 often outperforms α=0.1 with B=500.
""")

print(f"  {'α':>6} | {'B':>6} | {'α×B product':>12} | {'Test Acc':>10} | {'Train Time (ms)':>16}")
print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*16}")

configs = [
    (0.5,  20),
    (0.2,  50),
    (0.1, 100),
    (0.05, 200),
    (0.02, 500),
    (0.01,1000),
]

for alpha, B in configs:
    t0 = time.perf_counter()
    gb_ab = GradientBoostingClassifier(
        n_estimators=B, learning_rate=alpha, max_depth=3,
        subsample=0.8, random_state=42
    )
    gb_ab.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    acc = accuracy_score(y_te, gb_ab.predict(X_te))
    ms  = (t1 - t0) * 1000
    print(f"  {alpha:>6.2f} | {B:>6} | {alpha*B:>12.1f} | {acc:>10.4f} | {ms:>16.1f}")

print(f"""
  The α×B product roughly controls the "total learning" done.
  Similar products → similar accuracy (given fixed dataset and depth).
  The difference: small α+large B explores the landscape more carefully → often better.
""")


# =============================================================================
# PART 4: ALL MODELS HEAD-TO-HEAD
# =============================================================================

print("=" * 65)
print("  PART 4: FULL MODEL COMPARISON")
print("=" * 65)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

models = [
    ("Logistic Regression",        LogisticRegression(C=1.0, max_iter=1000)),
    ("Decision Tree (deep)",       DecisionTreeClassifier(random_state=42)),
    ("Random Forest (B=200)",      RandomForestClassifier(200, random_state=42, n_jobs=-1)),
    ("Extra-Trees (B=200)",        ExtraTreesClassifier(200, random_state=42, n_jobs=-1)),
    ("Gradient Boosting (B=200)",  GradientBoostingClassifier(200, learning_rate=0.1, max_depth=3, random_state=42)),
    ("Gradient Boosting (B=500, α=0.02)",
                                   GradientBoostingClassifier(500, learning_rate=0.02, max_depth=3, subsample=0.8, random_state=42)),
]

print(f"  {'Model':>36} | {'Train Acc':>10} | {'Test Acc':>10} | {'Time (ms)':>10}")
print(f"  {'-'*36}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for name, clf in models:
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
    te_acc = accuracy_score(y_te, clf.predict(X_te))
    ms = (t1 - t0) * 1000
    print(f"  {name:>36} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {ms:>10.1f}")

print(f"""
  Key observations:
    Single DT: perfect training accuracy → severe overfitting
    RF and Extra-Trees: robust, good test accuracy, fast and parallel
    GB (α=0.1, B=200): competitive; single sequential pass
    GB (α=0.02, B=500): often best — slow convergence + more iterations

  For competition-level performance, combine:
    • Lower learning rate (0.01–0.05)
    • More trees (500–3000)
    • Row subsampling (subsample=0.8)
    • Column subsampling (max_features=0.7)
    • Early stopping to find optimal B
""")
''',
    },

    "GB vs RF — Bias, Variance, and Practical Guide": {
        "description": "Side-by-side comparison of Random Forest vs Gradient Boosting across every dimension",
        "runnable": True,
        "code": '''
"""
================================================================================
GRADIENT BOOSTING vs RANDOM FOREST — COMPREHENSIVE COMPARISON
================================================================================

The two dominant tree ensemble methods. This script compares them across:
    1. Bias and variance decomposition (the fundamental difference)
    2. Sensitivity to noisy data (outliers and label noise)
    3. Convergence behaviour (more trees: always good vs U-curve)
    4. Hyperparameter sensitivity
    5. Practical decision guide

================================================================================
"""

import numpy as np
import time
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

np.random.seed(42)


# =============================================================================
# PART 1: BIAS AND VARIANCE — THE CORE DIFFERENCE
# =============================================================================

print("=" * 65)
print("  PART 1: BIAS vs VARIANCE — DECOMPOSING THE ERROR")
print("=" * 65)

print("""
  THEORY:
    Total Error = Bias² + Variance + Irreducible Noise

    Random Forest:        Bias = single tree bias (unchanged)
                          Variance → 0 as B → ∞ (averaging)
                          Strategy: grow deep trees (low bias), average to reduce variance

    Gradient Boosting:    Bias decreases as B grows (each tree corrects errors)
                          Variance can increase (risk of overfit)
                          Strategy: shallow trees (weak learners) + many boosting steps

  We verify this empirically by comparing model accuracy vs tree complexity:
""")

X, y = make_classification(n_samples=1000, n_features=15, n_informative=8,
                            class_sep=0.8, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

print(f"  Effect of single tree depth on RF vs GB accuracy:")
print(f"  {'max_depth':>11} | {'RF Test Acc':>12} | {'GB Test Acc':>12} | Winner")
print(f"  {'-'*11}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

for depth in [1, 2, 3, 5, 8, None]:
    rf_d = RandomForestClassifier(100, max_depth=depth, random_state=42, n_jobs=-1)
    gb_d = GradientBoostingClassifier(100, learning_rate=0.1, max_depth=depth if depth else 10,
                                       random_state=42)
    rf_d.fit(X_tr, y_tr);  rf_acc = accuracy_score(y_te, rf_d.predict(X_te))
    gb_d.fit(X_tr, y_tr);  gb_acc = accuracy_score(y_te, gb_d.predict(X_te))
    winner = "RF" if rf_acc > gb_acc + 0.005 else ("GB" if gb_acc > rf_acc + 0.005 else "tie")
    depth_str = str(depth) if depth else "None"
    print(f"  {depth_str:>11} | {rf_acc:>12.4f} | {gb_acc:>12.4f} | {winner}")

print(f"""
  RF: performance peaks at full depth (None) — deep trees + averaging works best
  GB: peaks at shallow depth (2-3) — shallow weak learners are the right building block

  RF with max_depth=1 (stumps): terrible — stumps can't capture patterns, averaging doesn't help
  GB with max_depth=1 (stumps): reasonable — boosting can combine many stumps into complex model
""")


# =============================================================================
# PART 2: MORE TREES — ALWAYS GOOD (RF) vs U-CURVE (GB)
# =============================================================================

print("=" * 65)
print("  PART 2: CONVERGENCE — MORE TREES ALWAYS GOOD (RF) vs U-CURVE (GB)")
print("=" * 65)

print(f"  {'B (trees)':>10} | {'RF Test Acc':>12} | {'GB Test Acc':>12} | GB overfit?")
print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*11}")

prev_gb_acc = 0
for B in [1, 5, 10, 25, 50, 100, 200, 400, 800]:
    rf_b = RandomForestClassifier(B, random_state=42, n_jobs=-1)
    gb_b = GradientBoostingClassifier(B, learning_rate=0.1, max_depth=3, random_state=42)
    rf_b.fit(X_tr, y_tr); rf_acc = accuracy_score(y_te, rf_b.predict(X_te))
    gb_b.fit(X_tr, y_tr); gb_acc = accuracy_score(y_te, gb_b.predict(X_te))
    overfit_flag = "OVERFITTING" if gb_acc < prev_gb_acc - 0.01 else "no"
    prev_gb_acc = gb_acc if gb_acc > prev_gb_acc else prev_gb_acc
    print(f"  {B:>10} | {rf_acc:>12.4f} | {gb_acc:>12.4f} | {overfit_flag}")

print(f"""
  RF accuracy: monotonically improves or stays flat with more trees.
  GB accuracy: improves then degrades (overfits) with too many trees.

  This is the single most important practical difference:
    RF: set n_estimators high (200+), it won't hurt you
    GB: must tune n_estimators carefully (use early stopping)
""")


# =============================================================================
# PART 3: NOISE ROBUSTNESS
# =============================================================================

print("=" * 65)
print("  PART 3: ROBUSTNESS TO LABEL NOISE")
print("=" * 65)

print(f"""
  We add label noise by randomly flipping y from 0→1 or 1→0.
  GB is generally more sensitive to noise than RF.
""")

X_n, y_n = make_classification(n_samples=800, n_features=15, n_informative=8,
                                random_state=42)
X_n_tr, X_n_te, y_n_tr, y_n_te = train_test_split(X_n, y_n, test_size=0.25, random_state=42)

print(f"  {'Noise %':>8} | {'RF Test Acc':>12} | {'GB Test Acc':>12} | {'RF degradation':>15} | {'GB degradation':>15}")
print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}")

# Clean baseline
rf_clean = RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(X_n_tr, y_n_tr)
gb_clean = GradientBoostingClassifier(100, learning_rate=0.1, max_depth=3, random_state=42).fit(X_n_tr, y_n_tr)
base_rf = accuracy_score(y_n_te, rf_clean.predict(X_n_te))
base_gb = accuracy_score(y_n_te, gb_clean.predict(X_n_te))

print(f"  {'0%':>8} | {base_rf:>12.4f} | {base_gb:>12.4f} | {'—':>15} | {'—':>15}")

for noise_pct in [5, 10, 20, 30, 40]:
    y_noisy = y_n_tr.copy()
    n_flip = int(len(y_noisy) * noise_pct / 100)
    flip_idx = np.random.choice(len(y_noisy), n_flip, replace=False)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

    rf_n = RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(X_n_tr, y_noisy)
    gb_n = GradientBoostingClassifier(100, learning_rate=0.1, max_depth=3, random_state=42).fit(X_n_tr, y_noisy)
    rf_acc = accuracy_score(y_n_te, rf_n.predict(X_n_te))
    gb_acc = accuracy_score(y_n_te, gb_n.predict(X_n_te))
    rf_deg = base_rf - rf_acc
    gb_deg = base_gb - gb_acc
    print(f"  {noise_pct:>7}% | {rf_acc:>12.4f} | {gb_acc:>12.4f} | {rf_deg:>+15.4f} | {gb_deg:>+15.4f}")

print(f"""
  Both degrade with noise, but the pattern differs:
    RF: robustly averages over noise — degradation is gradual and modest
    GB: chases residuals aggressively — noisy labels create misleading residuals
    → GB can overfit noise more readily than RF

  Mitigation for GB with noisy data:
    • Reduce learning_rate (smaller corrections per step)
    • Reduce max_depth (weaker learners, harder to memorise noise)
    • Use subsample < 1.0 (stochastic GB, fewer examples per tree)
    • Increase min_samples_leaf (ignore tiny subsets)
""")


# =============================================================================
# PART 4: DECISION GUIDE — WHICH TO USE?
# =============================================================================

print("=" * 65)
print("  PART 4: PRACTICAL DECISION GUIDE")
print("=" * 65)

print(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │  START WITH RANDOM FOREST IF:                                            │
  │    • You have little time for hyperparameter tuning                      │
  │    • You want reliable performance with zero preprocessing               │
  │    • You need OOB error estimates (no separate validation set)           │
  │    • Training speed matters (RF is parallelisable)                       │
  │    • Data is noisy or has label errors                                   │
  │    • Quick prototype before investing in more complex models             │
  │                                                                          │
  │  USE GRADIENT BOOSTING (XGBoost/LightGBM) WHEN:                          │
  │    • Maximum accuracy is the priority                                    │
  │    • You have a validation set and will use early stopping               │
  │    • Dataset is large (LightGBM is extremely fast)                       │
  │    • Custom loss function needed (quantile, ranking, survival)           │
  │    • You can tune hyperparameters via cross-validation or Bayesian opt.  │
  │    • Tabular data benchmark: GB almost always wins with tuning           │
  │                                                                          │
  │  NEITHER IS APPROPRIATE WHEN:                                            │
  │    • Extrapolation is needed (both plateau at training range)            │
  │    • Full interpretability is required (use a single decision tree)      │
  │    • Data is images / text / audio (use neural networks)                 │
  │    • Very sparse data n << p (use logistic regression / SVM)             │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
""")

# Final timing comparison
print(f"  Timing on n={len(X_tr)} train, p=15 features:")
print(f"  {'Model':>30} | {'Time (ms)':>10} | {'Test Acc':>10}")
print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}")
timing_models = [
    ("RF (B=100, n_jobs=-1)",      RandomForestClassifier(100, n_jobs=-1, random_state=42)),
    ("RF (B=200, n_jobs=-1)",      RandomForestClassifier(200, n_jobs=-1, random_state=42)),
    ("GB sklearn (B=100)",         GradientBoostingClassifier(100, learning_rate=0.1, max_depth=3, random_state=42)),
    ("GB sklearn (B=200)",         GradientBoostingClassifier(200, learning_rate=0.05, max_depth=3, random_state=42)),
]
for name, clf in timing_models:
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    ms = (time.perf_counter() - t0) * 1000
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  {name:>30} | {ms:>10.1f} | {acc:>10.4f}")

print(f"""
  GB sklearn is sequential (no parallelism within tree building).
  XGBoost/LightGBM are significantly faster for large n via histogram splits
  and parallelisation — for those, GB training time is competitive with RF.
""")
''',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 65)
    print("  GRADIENT BOOSTING: SEQUENTIAL ERROR CORRECTION")
    print("=" * 65)
    print("""
  Key concepts demonstrated:
    • F₀ = mean(y); each tree hₜ fits residuals rᵢ = yᵢ - F_{t-1}(xᵢ)
    • Learning rate α shrinks each tree's contribution (regularisation)
    • Unlike RF: more trees CAN overfit → use early stopping
    • Any differentiable loss = any pseudo-residual = versatile framework
    • Bias reduced by boosting, variance controlled by depth + α + subsample
    • XGBoost/LightGBM extend with 2nd-order gradients, histogram splits
    """)

    np.random.seed(42)

    X, y = make_classification(n_samples=600, n_features=15, n_informative=8,
                               class_sep=0.9, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    print("=" * 65)
    print("  SINGLE TREE vs RANDOM FOREST vs GRADIENT BOOSTING")
    print("=" * 65)

    models = [
        ("Single Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest (B=200)", RandomForestClassifier(200, random_state=42, n_jobs=-1)),
        ("GB (α=0.1, B=100, depth=3)",
         GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ("GB (α=0.05, B=200, depth=3)",
         GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)),
    ]

    print(f"\n  {'Model':>35}  {'Train Acc':>10}  {'Test Acc':>10}  {'Gen Gap':>10}")
    print(f"  {'-' * 35}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for name, clf in models:
        clf.fit(X_tr, y_tr)
        tr = accuracy_score(y_tr, clf.predict(X_tr))
        te = accuracy_score(y_te, clf.predict(X_te))
        print(f"  {name:>35}  {tr:>10.4f}  {te:>10.4f}  {tr - te:>10.4f}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1.  GB = gradient descent in FUNCTION space (not weight space)
  2.  Each tree fits pseudo-residuals = negative gradient of the loss
  3.  For MSE: pseudo-residuals = exact residuals (yᵢ - ŷᵢ)
  4.  For BCE: pseudo-residuals = yᵢ - σ(F(xᵢ)) = yᵢ - ŷᵢ (same form!)
  5.  Learning rate α shrinks each tree: small α + large B = better regularisation
  6.  MORE TREES CAN OVERFIT in GB — unlike RF which always improves
  7.  Use early stopping to find optimal n_estimators automatically
  8.  Loss function is pluggable: MSE, MAE, Huber, BCE, quantile, ranking...
  9.  Shallow trees (depth 1–3) are the correct weak learners for GB
  10. XGBoost: adds second-order gradients + regularisation terms
  11. LightGBM: GOSS + EFB + histograms = fastest for large n
  12. CatBoost: ordered boosting = best native categorical handling
  13. On tabular data: GB + tuning usually beats RF and neural networks
    """)


# # ─────────────────────────────────────────────────────────────────────────────
# # CONTENT EXPORT
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_content():
#     return {
#         "theory": THEORY,
#         "theory_raw": THEORY,
#         "complexity": COMPLEXITY,
#         "operations": OPERATIONS,
#         "interactive_components": [],
#     }


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
        from Supervised_Learning.visuals.gradient_boosting_visual import (   # ← match your exact folder casing
            GB_VISUAL_HTML,
            GB_VISUAL_HEIGHT,
        )
        visual_html   = GB_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = GB_VISUAL_HEIGHT
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

