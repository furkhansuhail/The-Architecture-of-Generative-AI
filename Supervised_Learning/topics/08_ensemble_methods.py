"""
Ensemble Methods — Wisdom of Crowds in Machine Learning
========================================================

Ensemble methods are a family of meta-learning strategies that combine multiple
"weak" or "base" learners to produce a single, stronger predictor. The core
intuition is simple: many models that each make different mistakes can, when
aggregated, cancel out individual errors and produce dramatically better
generalisation than any single model alone.

This module covers Bagging, Random Forests, AdaBoost, Gradient Boosting,
and Stacking — the full ensemble toolkit.

"""

import re
import textwrap

DISPLAY_NAME = "08 · Ensemble Methods"
ICON         = "🌲"
SUBTITLE     = "Combining weak learners — crowds, forests, and sequential boosting"

THEORY = """

## 08 · Ensemble Methods

Ensemble methods answer a deceptively simple question: if one model is good,
can many models be better? The answer is yes — but only if those models make
**different** errors. Diversity is the engine of ensemble learning.

### What is an Ensemble Method?

An ensemble is a combination of multiple models (called base learners or weak
learners) whose individual predictions are aggregated into a single final
prediction. The aggregation may be averaging, majority voting, or a learned
combination.

The three canonical strategies are:

    Bagging:    train base learners INDEPENDENTLY on random subsets of data,
                then AVERAGE or VOTE on predictions.
                Goal: reduce variance.

    Boosting:   train base learners SEQUENTIALLY, each one correcting the
                errors of the previous. Weight the ensemble by learner quality.
                Goal: reduce bias (and variance).

    Stacking:   train base learners independently, then train a meta-learner
                on their predictions as features.
                Goal: learn the optimal combination of base learner outputs.

    Things that exist inside the model (learnable parameters):
        - Base learner parameters   — weights, splits, coefficients of each tree/model
        - Ensemble weights          — αₘ in Boosting, meta-model weights in Stacking
        - Sample weights (Boosting) — wᵢ, emphasising misclassified examples

    Things that exist only at setup time (hyperparameters):
        - n_estimators   — number of base learners
        - max_depth      — depth of each base tree (controls complexity)
        - learning_rate  — step size for boosting updates
        - max_features   — feature subsetting (Random Forest)
        - subsample      — row subsetting (Gradient Boosting)

### Ensemble Methods as Empirical Risk Minimisation (ERM)

Ensemble methods fit cleanly into the ERM framework:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Bagging (Random Forest):                                   │
    │    Hypothesis class:  H = { (1/M) Σₘ hₘ(x) }                 │ 
    │    Objective: average over M independently trained trees    │
    │    No explicit loss minimisation at ensemble level          │
    │                                                             │
    │  AdaBoost:                                                  │
    │    Hypothesis class:  H = { Σₘ αₘ hₘ(x) }                    │
    │    Loss function:     Exponential loss  exp(-y·F(x))        │
    │    Greedy, stage-wise minimisation of exp loss              │
    │                                                             │
    │  Gradient Boosting:                                         │
    │    Hypothesis class:  H = { F₀ + Σₘ γₘ hₘ(x) }               │
    │    Loss function:     ANY differentiable loss L(y, F(x))    │
    │    Greedy function-space gradient descent                   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Comparing across the series:
    Linear Regression:    MSE loss,         single linear model
    Logistic Regression:  BCE loss,         single linear model
    SVM:                  Hinge loss,       single maximum-margin model
    Random Forest:        No ensemble loss, average of independent trees
    AdaBoost:             Exponential loss, greedy stage-wise optimisation
    Gradient Boosting:    Any loss,         gradient descent in function space


### The Inductive Bias of Ensemble Methods

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Ensemble methods encode the belief that:                   │
    │                                                             │
    │  1. DIVERSITY — independent errors cancel. Models that      │
    │     are slightly wrong in different places average out      │
    │     to something much closer to correct.                    │
    │                                                             │
    │  2. WISDOM OF CROWDS — aggregated judgement from many       │
    │     mediocre predictors can surpass a single expert,        │
    │     provided the crowd disagrees on their errors.           │
    │                                                             │
    │  3. SMOOTHNESS (Bagging) — averaging smooths the            │
    │     jagged, high-variance boundary of a single deep tree.   │
    │                                                             │
    │  4. SEQUENTIAL CORRECTION (Boosting) — each new model       │
    │     should concentrate on the hardest examples, the         │
    │     ones that previous models got wrong.                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

---

### Part 1: The Bias–Variance Decomposition — Why Ensembles Work

The expected test error of any predictor F decomposes as:

    E[(y − F(x))²]  =  Bias²[F(x)]  +  Var[F(x)]  +  σ²(noise)
                        ↑               ↑               ↑
                        systematic      sensitivity     irreducible
                        error           to training     noise
                                        data

This decomposition reveals the fundamental tradeoff:

    High-bias models (underfitting): simple, too rigid, systematically wrong.
        Lowering bias usually raises variance.

    High-variance models (overfitting): complex, sensitive to training noise.
        Lowering variance usually raises bias.

Ensemble methods attack specific parts of this tradeoff:

    # =======================================================================================# 
    **Diagram 1 — Where Ensembles Live in Bias–Variance Space:**

    MODEL COMPLEXITY →

    Low Complexity                    Medium                    High Complexity
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                                                                           │
    │  Bias:   HIGH ────────────────────────────────────────────────────► LOW   │
    │  Var:    LOW  ◄─────────────────────────────────────────────────── HIGH   │
    │                                                                           │
    │  Single linear  →   Shallow tree   →   Deep tree  →   Fully-grown tree    │
    │      model                                                                │
    │                                                                           │
    │  BAGGING applies here:        ↓                                           │
    │    Deep tree (high var) → AVERAGE 100 deep trees → still low bias,        │
    │    but variance reduced by factor of M (roughly)                          │
    │                                                                           │
    │  BOOSTING applies here:  ↓                                                │
    │    Shallow tree (high bias) → ADD sequentially → accumulate complexity,   │
    │    drive bias toward zero while controlling variance via learning rate    │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘

    BAGGING:   starts with high-variance, low-bias model → reduces variance
    BOOSTING:  starts with high-bias, low-variance model → reduces bias

    This is the fundamental difference between the two paradigms.
    # =======================================================================================# 

### Variance Reduction by Averaging (Why Bagging Works):

If M models each have variance σ² and are perfectly uncorrelated:

                    Var[average] = σ² / M

With M = 100 trees, variance drops to 1/100 of a single tree's variance.
In practice, trees are correlated (same training data), so the reduction is:

                    Var[average] = ρσ² + (1-ρ)σ²/M

Where ρ is the average pairwise correlation between trees. As M → ∞:

                    Var[average] → ρσ²   (irreducible floor)

This is why Random Forest adds feature randomness: it reduces ρ, pushing
the floor lower. More diverse trees → lower residual variance.

---

### Part 2: Bagging — Bootstrap Aggregating

Bagging (Breiman, 1996) creates an ensemble by training each base learner
on a different bootstrap sample of the training data.

A bootstrap sample of size n is drawn WITH REPLACEMENT from the training set:
    - ~63.2% of original examples appear at least once
    - ~36.8% are NOT sampled (called out-of-bag, or OOB, examples)
    - Some examples appear 2, 3, or more times

    # =======================================================================================# 
    **Diagram 2 — The Bagging Process:**

    TRAINING DATA: [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]

    Bootstrap 1: [x₂, x₁, x₅, x₂, x₈, x₃, x₁, x₆]  → Train Tree 1
    Bootstrap 2: [x₄, x₇, x₃, x₄, x₁, x₆, x₂, x₄]  → Train Tree 2
    Bootstrap 3: [x₁, x₃, x₈, x₅, x₇, x₃, x₆, x₂]  → Train Tree 3
          ⋮
    Bootstrap M: [x₃, x₆, x₄, x₇, x₁, x₅, x₈, x₄]  → Train Tree M

    ──────────────────────────────────────────────────────────

    PREDICTION (new x):
        Tree 1 → ŷ₁ = 0.72
        Tree 2 → ŷ₂ = 0.68
        Tree 3 → ŷ₃ = 0.81
          ⋮
        Tree M → ŷₘ = 0.75

        Regression:     ŷ = (1/M) Σₘ ŷₘ  =  average
        Classification: ŷ = mode{ class₁, class₂, ..., classₘ }  =  majority vote

    ──────────────────────────────────────────────────────────
    # =======================================================================================# 

### Out-of-Bag (OOB) Evaluation:

For each training example xᵢ, roughly 36.8% of trees were trained without it.
These OOB trees form a natural validation set for xᵢ. Averaging predictions
from OOB trees over all training points gives the OOB score — a nearly unbiased
estimate of test error, without holding out a separate validation set.

    OOB Score ≈ Cross-validation score, but free — no extra computation needed.

---

### Part 3: Random Forests — Bagging with Feature Randomness

Random Forests (Breiman, 2001) extend Bagging by introducing a second source
of randomness: at each split in each tree, only a random subset of features
is considered as split candidates.

    Standard decision tree: split on the BEST feature among ALL p features
    Random Forest tree:     split on the BEST feature among RANDOM √p features

This feature randomness decorrelates the trees. Two trees that would both have
split on the same dominant feature (say, "income") are now forced to explore
different features at different splits. The resulting trees are more diverse,
which reduces the ρ in the variance formula and improves ensemble generalisation.

    # =======================================================================================# 
    **Diagram 3 — Feature Randomness in Random Forests:**

    ALL FEATURES: [f₁, f₂, f₃, f₄, f₅, f₆, f₇, f₈]

    Node in Tree 1, Split 1: randomly select [f₂, f₅, f₇]  → split on f₅
    Node in Tree 1, Split 2: randomly select [f₁, f₃, f₈]  → split on f₁
    Node in Tree 2, Split 1: randomly select [f₄, f₆, f₇]  → split on f₇
          ⋮

    WITHOUT feature randomness: every tree would likely split on f₃ first
    (the most informative feature). Trees would be highly correlated → ρ ≈ 1
    → averaging does very little. This is why simple Bagging of deep trees
    is less effective than Random Forest.

    WITH feature randomness: different trees explore different feature subsets,
    capturing different aspects of the data. Lower ρ → better ensemble.

    ──────────────────────────────────────────────────────────────────────
    Hyperparameter    Default           Effect
    ──────────────────────────────────────────────────────────────────────
    max_features      √p (classif.)     lower = more diversity, weaker trees
                      p/3 (regress.)    higher = stronger trees, more correlation
    n_estimators      100               more = lower variance, diminishing returns
    max_depth         None (full)       limit = more bias, less variance per tree
    min_samples_leaf  1                 higher = smoother, more regularised
    ──────────────────────────────────────────────────────────────────────
    # =======================================================================================# 

### Feature Importance in Random Forests:

Random Forests naturally produce feature importance scores — a by-product of
the ensemble construction, not a post-hoc analysis:

    Mean Decrease in Impurity (MDI):
        For each feature fⱼ, sum the total impurity decrease (Gini or MSE)
        across all splits on fⱼ, weighted by the number of samples reaching
        that node, averaged over all trees.

    Mean Decrease in Accuracy (MDA / Permutation Importance):
        For each feature fⱼ, randomly shuffle its values in the OOB set.
        Record the drop in OOB accuracy. Larger drop = more important feature.

    MDA is more reliable than MDI for high-cardinality or correlated features,
    because MDI can be biased toward features with many possible split values.

---

### Part 4: Boosting — Sequential Error Correction

Boosting builds an ensemble SEQUENTIALLY: each new base learner focuses on
the examples that previous learners got wrong. The final prediction is a
weighted vote, with more accurate learners getting higher weights.

The intuition: imagine a student who takes a test, reviews their mistakes,
then takes the test again focusing on weak areas. Each attempt corrects the
systematic errors of the last.

### AdaBoost — Adaptive Boosting (Freund & Schapire, 1997):

AdaBoost maintains a weight distribution over training examples. Initially
all weights are equal (1/n). At each round m:

    1. Train a weak learner hₘ on the weighted training set
    2. Compute weighted error: εₘ = Σᵢ wᵢ · 1[hₘ(xᵢ) ≠ yᵢ]
    3. Compute learner weight: αₘ = (1/2) log((1 − εₘ) / εₘ)
    4. Update sample weights:
           wᵢ ← wᵢ · exp(−αₘ · yᵢ · hₘ(xᵢ))
       then normalise so Σᵢ wᵢ = 1

    Correctly classified:   wᵢ decreases  (model already handles them)
    Misclassified:          wᵢ increases  (next model must focus here)

    Final prediction: F(x) = sign( Σₘ αₘ · hₘ(x) )

    # =======================================================================================# 
    **Diagram 4 — AdaBoost Weight Evolution:**

    Round 0: All weights equal
    ●●●●●●●●●●●●●○○○○○○○○  (● class+1, ○ class-1, size = weight)

    Round 1: h₁ misclassifies 3 ○ points
    ●●●●●●●●●●●●●○○○○○○○○  → increase weight on 3 misclassified ○
    ●●●●●●●●●●●●●○○○○⊙⊙⊙  (⊙ = upweighted point)

    Round 2: h₂ focuses on the upweighted points, now misclassifies different ones
    ...weights shift again to the NEW hard examples

    Round M: final ensemble αₘ = large for accurate hₘ, small for weak hₘ

    εₘ → 0:  hₘ is nearly perfect → αₘ → ∞ (dominates the vote)
    εₘ = 0.5: hₘ is random → αₘ = 0  (ignored entirely)
    εₘ > 0.5: hₘ is worse than random → αₘ < 0 (reversed!)
    # =======================================================================================# 

### AdaBoost as Exponential Loss Minimisation:

AdaBoost minimises the exponential loss:

    L(y, F(x)) = exp(−y · F(x))    where y ∈ {−1, +1}

The stage-wise algorithm (fit hₘ, compute αₘ, update weights) is exactly
coordinate descent on the exponential loss in function space. This connection,
established by Friedman et al. (2000), placed AdaBoost in the ERM framework
and revealed why it works.

The exponential loss strongly penalises confident wrong predictions:
    y·F(x) = −3  (very wrong): loss = e³ ≈ 20
    y·F(x) = −1  (wrong):      loss = e¹ ≈ 2.7
    y·F(x) = +1  (correct):    loss = e⁻¹ ≈ 0.37

This heavy tail means AdaBoost is sensitive to outliers and mislabelled data.

---

### Part 5: Gradient Boosting — Gradient Descent in Function Space

Gradient Boosting (Friedman, 2001) is the most powerful and general boosting
framework. Instead of deriving a specific weight-update rule for a specific
loss (as AdaBoost does for exponential loss), Gradient Boosting:

    1. Computes the NEGATIVE GRADIENT of any differentiable loss L(yᵢ, F(xᵢ))
       with respect to the current prediction F(xᵢ)

    2. Fits a new base learner hₘ to those negative gradients (pseudo-residuals)

    3. Adds the new learner with a small step size (learning rate η):
                F(x) ← F(x) + η · γₘ · hₘ(x)

The pseudo-residuals rᵢₘ = −∂L/∂F(xᵢ) tell us the direction each prediction
should move to decrease the loss. Fitting a tree to them finds the structure in
the data that predicts which way to nudge each observation.

For MSE loss (regression), the pseudo-residuals are just the residuals:
    L = (1/2)(y − F)²   →   −∂L/∂F = y − F(x)

This is why Gradient Boosting with MSE loss is literally "fit a tree to the
residuals of the current model", repeated many times.

For log-loss (classification):
    L = −y·log(p) − (1−y)·log(1−p)   →   pseudo-residuals = y − p(x)

The algorithm is identical; only the residuals change.

    # =======================================================================================# 
    **Diagram 5 — Gradient Boosting Stage by Stage:**

    TARGET:  y
    INITIAL: F₀(x) = mean(y)   (constant prediction)

    Residuals₁ = y − F₀(x)
    Fit Tree₁ to residuals₁  →  h₁(x)
    F₁(x) = F₀(x) + η · h₁(x)

    Residuals₂ = y − F₁(x)   (smaller now — F₁ corrected some errors)
    Fit Tree₂ to residuals₂  →  h₂(x)
    F₂(x) = F₁(x) + η · h₂(x)

         ⋮

    Residualsₘ = y − Fₘ₋₁(x)
    Fit Treeₘ to Residualsₘ   →  hₘ(x)
    Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)

    As M → ∞ and η → 0: Fₘ → perfect fit to training data (overfitting!)
    Solution: early stopping, small η, or explicit regularisation (XGBoost)

    KEY: Each tree is SHALLOW (max_depth=1,2,3) — not a forest of deep trees.
    Depth controls the interaction order: depth-1 trees = no interactions,
    depth-2 trees = pairwise interactions, etc.
    # =======================================================================================# 

### XGBoost, LightGBM, and CatBoost:

Modern implementations of Gradient Boosting add extensive regularisation
and engineering optimisations:

    ┌────────────────────────────────────────────────────────────────────────┐
    │  XGBoost (Chen & Guestrin, 2016):                                      │
    │    • Second-order (Newton) approximation of loss — more accurate steps │
    │    • Explicit L1 + L2 regularisation on tree leaf weights              │
    │    • Sparsity-aware split finding (handles missing values natively)    │
    │    • Column subsampling (like Random Forest feature randomness)        │
    │    • Parallelised tree construction                                    │
    │                                                                        │
    │  LightGBM (Microsoft, 2017):                                           │
    │    • Leaf-wise tree growth (vs level-wise in XGBoost) — faster/deeper  │
    │    • GOSS: Gradient-based One-Side Sampling (downsample easy examples) │
    │    • EFB: Exclusive Feature Bundling (compress sparse features)        │
    │    • Histogram-based binning — much faster than exact split search     │
    │                                                                        │
    │  CatBoost (Yandex, 2018):                                              │
    │    • Native categorical feature handling (ordered boosting)            │
    │    • Oblivious trees (same split used at every node of a level)        │
    │    • Reduces prediction shift (target leakage in gradient computation) │
    └────────────────────────────────────────────────────────────────────────┘

---

### Part 6: Stacking — Learning to Combine

Stacking (Wolpert, 1992) trains a meta-learner (level-1 model) to optimally
combine the predictions of base learners (level-0 models).

The critical insight: base learner predictions on training data will be
overfit (the learner has "seen" those examples). To get honest predictions
for the meta-learner to train on, use k-fold cross-validation:

    1. Split training data into k folds
    2. For each fold i:
           Train all base learners on folds ≠ i
           Predict on fold i → out-of-fold predictions
    3. Stack out-of-fold predictions as new features → Z_train
    4. Retrain all base learners on FULL training data
    5. Train meta-learner on Z_train (with y as target)
    6. At test time: base learners predict → meta-learner combines

    # =======================================================================================# 
    **Diagram 6 — The Stacking Pipeline:**

    TRAINING DATA:
    ┌─────────────────────────────────────────────────────────────┐
    │  X_train, y_train                                           │
    └──────┬──────────────────────────────────────────────────────┘
           │ k-fold CV on each base learner
           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Base Learner 1 (Random Forest)  ─► OOF predictions₁        │
    │  Base Learner 2 (XGBoost)        ─► OOF predictions₂        │  ─► Z_train
    │  Base Learner 3 (Logistic Reg.)  ─► OOF predictions₃        │
    └─────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Meta-learner (Logistic Regression or Ridge)                │
    │  trains on Z_train to learn: "when to trust which model"    │
    └─────────────────────────────────────────────────────────────┘

    TEST TIME:
    X_test → [RF, XGB, LR] → test predictions → meta-learner → final ŷ

    # =======================================================================================# 

---

### Part 7: Ensemble Methods vs Single Models — A Deep Comparison

    ──────────────────────────────────────────────────────────────────────────
    Property               Single Tree    Random Forest   Gradient Boosting
    ──────────────────────────────────────────────────────────────────────────
    Bias                   Low (deep)     Low             Very low
    Variance               High           Low             Low (with reg.)
    Interpretability       High           Low             Low
    Training time          Fast           Moderate        Slow (sequential)
    Prediction time        Fast           Slow (M trees)  Slow (M trees)
    Handles missing vals   Yes (CART)     Partially       Yes (XGBoost)
    Handles categoricals   Yes            Partially       Yes (CatBoost)
    Outlier sensitivity    Moderate       Low             High (AdaBoost)
    Feature importance     Built-in       MDI / MDA       Gain / shap
    Probabilistic output   No             Yes (fraction)  Yes (calibrated)
    Hyperparameter tuning  Minimal        Moderate        Extensive
    ──────────────────────────────────────────────────────────────────────────

### When to Use Which:

    ┌──────────────────────────────────────────────────────────────┐
    │  Random Forest:                                              │
    │    • Fast to train and tune                                  │
    │    • Robust baseline — works well out of the box             │
    │    • Noisy data or outliers present                          │
    │    • Feature importance needed without tuning fuss           │
    │    • n < ~500k (scales well but not infinitely)              │
    │                                                              │
    │  Gradient Boosting (XGBoost/LightGBM):                       │
    │    • Highest predictive accuracy on tabular data             │
    │    • Willing to invest time in hyperparameter tuning         │
    │    • Kaggle competitions, production ML systems              │
    │    • Large n (LightGBM handles millions of rows)             │
    │    • Data has complex interactions between features          │
    │                                                              │
    │  Stacking:                                                   │
    │    • Squeezing out the last 0.1% of performance              │
    │    • Multiple heterogeneous models available                 │
    │    • Sufficient data for k-fold cross-validation             │
    │    • Competitions; usually overkill for production           │
    │                                                              │
    │  Single Tree:                                                │
    │    • Interpretability is the primary requirement             │
    │    • Dataset is tiny (n < 100)                               │
    │    • Decision rules must be human-readable                   │
    └──────────────────────────────────────────────────────────────┘


### From Ensembles to Deep Learning:

The same ideas that power ensemble methods appear throughout deep learning:

    Dropout:          approximate ensemble of 2^p neural networks
    Multi-head attention: ensemble of independent attention patterns
    Model averaging:  classic ensembling of neural network checkpoints
    Mixture of Experts (MoE): gated ensemble — each expert handles a region

Gradient Boosting and neural networks are converging: XGBoost now uses
GPU acceleration and neural-style regularisation, while TabNet and NODE
implement differentiable tree-like operations in neural architectures.

For tabular data (the bread-and-butter of industry ML), Gradient Boosting
still outperforms deep learning on most benchmarks as of 2024 — one of the
few areas where classical ML holds its ground.

"""


VISUAL_HTML = ""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Runnable code demonstrations
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Bagging and Random Forest from Scratch": {
        "description": "Bootstrap sampling, OOB evaluation, and feature importance — building a forest manually",
        "runnable": True,
        "timeout": None,
        "code": '''
"""
================================================================================
BAGGING & RANDOM FOREST FROM SCRATCH
================================================================================

We implement:
    1. Bootstrap sampling
    2. A BaggingClassifier from scratch (majority vote over independent trees)
    3. Out-of-Bag (OOB) score computation
    4. Feature importance via Mean Decrease in Impurity

The core ensemble prediction:
    F(x) = mode{ h₁(x), h₂(x), ..., hₘ(x) }   (classification)
    F(x) = (1/M) Σₘ hₘ(x)                       (regression)

================================================================================
"""

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


np.random.seed(42)


# =============================================================================
# PART 1: BOOTSTRAP SAMPLING
# =============================================================================

print("=" * 65)
print("  PART 1: BOOTSTRAP SAMPLING — THE FOUNDATION OF BAGGING")
print("=" * 65)

def bootstrap_sample(X, y, seed=None):
    """
    Draw a bootstrap sample (with replacement) of size n from (X, y).

    Returns:
        X_boot, y_boot: bootstrap sample
        oob_mask:       boolean mask — True for out-of-bag examples
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.integers(0, n, size=n)   # sample with replacement
    oob_mask = ~np.isin(np.arange(n), indices)
    return X[indices], y[indices], oob_mask

# Demonstrate on small dataset
n_demo = 10
X_demo = np.arange(n_demo).reshape(-1, 1).astype(float)
y_demo = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])

print(f"\\n  Original indices:  {list(range(n_demo))}")
print(f"  Labels:            {list(y_demo)}\\n")

for trial in range(5):
    _, _, oob = bootstrap_sample(X_demo, y_demo, seed=trial)
    sampled = [i for i in range(n_demo) if not oob[i]]
    oob_idx  = [i for i in range(n_demo) if oob[i]]
    print(f"  Bootstrap {trial+1}:  sampled = {sampled}  |  OOB = {oob_idx}  ({len(oob_idx)}/10 OOB)")

print(f"""
  Key observation: On average, ~{100*(1 - (1 - 1/n_demo)**n_demo):.1f}% of
  examples appear in each bootstrap sample (for n={n_demo}).
  As n → ∞, this fraction → 1 − 1/e ≈ 63.2%.
  The remaining ~36.8% are always out-of-bag.
""")


# =============================================================================
# PART 2: BAGGING CLASSIFIER FROM SCRATCH
# =============================================================================

print("=" * 65)
print("  PART 2: BAGGING CLASSIFIER FROM SCRATCH")
print("=" * 65)


class BaggingClassifierScratch:
    """
    Bagging ensemble: majority vote over M independently trained trees,
    each trained on a different bootstrap sample.

    The ensemble prediction:
        F(x) = argmax_c Σₘ 1[hₘ(x) = c]   (majority vote)

    OOB predictions are accumulated during training:
        For each training example xᵢ, only trees that did NOT see xᵢ
        (the OOB trees) contribute to the OOB prediction of xᵢ.
    """

    def __init__(self, n_estimators=50, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self.estimators_  = []
        self.oob_score_   = None

    def fit(self, X, y):
        n = len(X)
        rng = np.random.default_rng(self.random_state)

        # OOB accumulator: for each sample, collect predictions from OOB trees
        oob_predictions = [[] for _ in range(n)]

        self.estimators_ = []
        for m in range(self.n_estimators):
            seed = rng.integers(0, 1_000_000)
            X_b, y_b, oob_mask = bootstrap_sample(X, y, seed=seed)

            # Train a tree on the bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=int(seed))
            tree.fit(X_b, y_b)
            self.estimators_.append(tree)

            # Collect OOB predictions
            if oob_mask.any():
                oob_preds = tree.predict(X[oob_mask])
                for idx, pred in zip(np.where(oob_mask)[0], oob_preds):
                    oob_predictions[idx].append(pred)

        # Compute OOB score
        oob_correct = 0
        oob_counted = 0
        for i, preds in enumerate(oob_predictions):
            if preds:
                majority = Counter(preds).most_common(1)[0][0]
                if majority == y[i]:
                    oob_correct += 1
                oob_counted += 1
        self.oob_score_ = oob_correct / oob_counted if oob_counted > 0 else None
        return self

    def predict(self, X):
        """Majority vote across all estimators."""
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        # all_preds shape: (n_estimators, n_samples)
        result = []
        for sample_preds in all_preds.T:
            result.append(Counter(sample_preds).most_common(1)[0][0])
        return np.array(result)

    def predict_proba(self, X):
        """Fraction of trees voting for each class."""
        classes = np.unique([tree.classes_ for tree in self.estimators_])
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        proba = np.zeros((len(X), len(classes)))
        for j, c in enumerate(classes):
            proba[:, j] = (all_preds == c).mean(axis=0)
        return proba


# Train on breast cancer dataset
data = load_breast_cancer()
X_bc, y_bc = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X_bc, y_bc, test_size=0.25,
                                            random_state=42)

print(f"\\n  Dataset: Breast Cancer ({len(X_bc)} samples, {X_bc.shape[1]} features)")
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")
print()

print(f"  Training ensemble (watching OOB score converge)...")
print(f"  {'M':>6} | {'Test Acc':>10} | {'OOB Score':>10}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")

for M in [1, 5, 10, 25, 50, 100]:
    bag = BaggingClassifierScratch(n_estimators=M, max_depth=None, random_state=42)
    bag.fit(X_tr, y_tr)
    test_acc = accuracy_score(y_te, bag.predict(X_te))
    print(f"  {M:>6} | {test_acc:>10.4f} | {bag.oob_score_:>10.4f}")

print()
# Compare with sklearn
sk_single = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
sk_bag    = BaggingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
sk_rf     = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

print(f"  Comparison (n_estimators=100):")
print(f"    Single Decision Tree:       {accuracy_score(y_te, sk_single.predict(X_te)):.4f}")
print(f"    sklearn BaggingClassifier:  {accuracy_score(y_te, sk_bag.predict(X_te)):.4f}")
print(f"    sklearn RandomForest:       {accuracy_score(y_te, sk_rf.predict(X_te)):.4f}")
print(f"""
  Random Forest beats pure Bagging because feature randomness at each
  split decorrelates the trees, reducing the ρ in Var[average] = ρσ².
  A forest of diverse-but-weak trees beats correlated-but-strong trees.
""")


# =============================================================================
# PART 3: FEATURE IMPORTANCE (MEAN DECREASE IN IMPURITY)
# =============================================================================

print("=" * 65)
print("  PART 3: FEATURE IMPORTANCE — MEAN DECREASE IN IMPURITY")
print("=" * 65)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)

importances = rf.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
feat_names  = data.feature_names

print(f"\\n  Top 10 features by MDI (out of {len(feat_names)}):")
print(f"  {'Rank':>5} | {'Feature Name':>30} | {'Importance':>12} | Bar")
print(f"  {'-'*5}-+-{'-'*30}-+-{'-'*12}-+-{'-'*30}")

for rank, idx in enumerate(sorted_idx[:10], 1):
    bar = "█" * int(importances[idx] * 150)
    print(f"  {rank:>5} | {feat_names[idx]:>30} | {importances[idx]:>12.4f} | {bar}")

print(f"""
  Interpretation:
    • High importance → feature drives many impurity-reducing splits
    • MDI can favour high-cardinality features (many distinct split values)
    • Use permutation importance (sklearn's permutation_importance) for a
      more robust measure, especially when features are correlated.

  OOB Score (100 trees): {sk_rf.oob_score_ if hasattr(sk_rf, 'oob_score_') else '(use oob_score=True)'}
  Note: set oob_score=True when creating RandomForestClassifier to get this.
""")
''',
    },

    "AdaBoost from Scratch": {
        "description": "Sample reweighting, weak learner training, and exponential loss minimisation — step by step",
        "runnable": True,
        "timeout": None,
        "code": '''
"""
================================================================================
ADABOOST FROM SCRATCH — ADAPTIVE BOOSTING
================================================================================

AdaBoost Algorithm:
    Initialise: wᵢ = 1/n  for all i

    For m = 1, ..., M:
        1. Train weak learner hₘ on weighted training set
        2. Weighted error: εₘ = Σᵢ wᵢ · 1[hₘ(xᵢ) ≠ yᵢ]
        3. Learner weight:  αₘ = (1/2) log((1 − εₘ) / εₘ)
        4. Update weights:  wᵢ ← wᵢ · exp(−αₘ · yᵢ · hₘ(xᵢ))
        5. Normalise:       wᵢ ← wᵢ / Σᵢ wᵢ

    Predict: F(x) = sign( Σₘ αₘ · hₘ(x) )

Labels must be in {−1, +1} (not {0, 1}).

================================================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


np.random.seed(42)


# =============================================================================
# ADABOOST IMPLEMENTATION
# =============================================================================

class AdaBoostScratch:
    """
    Binary AdaBoost with decision stumps (depth-1 trees) as weak learners.

    The core insight:
        αₘ = 0.5 * log((1 − εₘ) / εₘ)

        εₘ < 0.5: hₘ better than random → αₘ > 0  (contributes positively)
        εₘ = 0.5: hₘ is random          → αₘ = 0  (has no effect)
        εₘ > 0.5: hₘ worse than random  → αₘ < 0  (reversed — still useful!)
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.estimators_   = []
        self.alphas_       = []
        self.errors_       = []
        self.train_errors_ = []  # track ensemble train error per round

    def fit(self, X, y):
        """
        y must be in {-1, +1}.
        """
        n = len(X)
        w = np.full(n, 1.0 / n)   # uniform initial weights

        self.estimators_ = []
        self.alphas_     = []
        self.errors_     = []

        for m in range(self.n_estimators):
            # ── Step 1: Train weak learner on weighted data ───────────────────
            # DecisionTreeClassifier supports sample_weight directly
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)

            # ── Step 2: Weighted error ─────────────────────────────────────────
            preds = stump.predict(X)
            incorrect = (preds != y).astype(float)
            eps_m = np.dot(w, incorrect)           # weighted error rate

            # Clip to avoid log(0): if eps_m = 0 or 1, algorithm is degenerate
            eps_m = np.clip(eps_m, 1e-10, 1 - 1e-10)

            # ── Step 3: Learner weight ─────────────────────────────────────────
            alpha_m = 0.5 * np.log((1 - eps_m) / eps_m) * self.learning_rate

            # ── Step 4: Update weights ─────────────────────────────────────────
            # exp(-α·y·h(x)):
            #   correctly classified (y·h = +1): weight decreases by e^{-α}
            #   misclassified        (y·h = -1): weight increases by e^{+α}
            w = w * np.exp(-alpha_m * y * preds)

            # ── Step 5: Normalise ──────────────────────────────────────────────
            w /= w.sum()

            self.estimators_.append(stump)
            self.alphas_.append(alpha_m)
            self.errors_.append(eps_m)

            # Track running ensemble train error
            train_preds = self.predict(X)
            self.train_errors_.append((train_preds != y).mean())

        return self

    def decision_function(self, X):
        """Weighted sum of weak learner outputs."""
        scores = np.zeros(len(X))
        for stump, alpha in zip(self.estimators_, self.alphas_):
            scores += alpha * stump.predict(X)
        return scores

    def predict(self, X):
        """Sign of weighted sum."""
        return np.sign(self.decision_function(X)).astype(int)

    def staged_accuracy(self, X, y):
        """Accuracy at each boosting round (for diagnosing convergence)."""
        scores = np.zeros(len(X))
        accs   = []
        for stump, alpha in zip(self.estimators_, self.alphas_):
            scores += alpha * stump.predict(X)
            preds   = np.sign(scores).astype(int)
            accs.append((preds == y).mean())
        return accs


# =============================================================================
# DATASET: Make Moons (non-linear)
# =============================================================================

X, y_bin = make_moons(n_samples=500, noise=0.3, random_state=42)
y = 2 * y_bin - 1   # convert to {-1, +1}

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

print("=" * 65)
print("  ADABOOST FROM SCRATCH — MOONS DATASET")
print("=" * 65)
print(f"  Dataset: make_moons (n=500, noise=0.3)")
print(f"  Train: {len(X_tr)},  Test: {len(X_te)}")
print(f"  Labels: {{-1, +1}}")


# =============================================================================
# PART 1: STEP-BY-STEP FIRST 5 ROUNDS
# =============================================================================

print("\\n  PART 1: FIRST 5 ROUNDS IN DETAIL")
print("  " + "-" * 55)

ada_demo = AdaBoostScratch(n_estimators=5)
ada_demo.fit(X_tr, y_tr)

print(f"  {'Round':>6} | {'ε (error)':>10} | {'α (weight)':>12} | {'Max Sample W':>13} | Interpretation")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*13}-+-{'-'*30}")

# Rerun step-by-step to inspect weights
n = len(X_tr)
w = np.full(n, 1.0 / n)
for m in range(5):
    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X_tr, y_tr, sample_weight=w)
    preds   = stump.predict(X_tr)
    eps_m   = np.clip(np.dot(w, (preds != y_tr).astype(float)), 1e-10, 1-1e-10)
    alpha_m = 0.5 * np.log((1 - eps_m) / eps_m)
    w       = w * np.exp(-alpha_m * y_tr * preds)
    w      /= w.sum()
    interp  = ("better than random ✓" if eps_m < 0.5 else
               "random — skipped" if eps_m > 0.499 else "worse than random!")
    print(f"  {m+1:>6} | {eps_m:>10.4f} | {alpha_m:>12.4f} | {w.max():>13.6f} | {interp}")

print(f"""
  Notice:
    • ε < 0.5 for all rounds → α > 0 (all stumps contribute positively)
    • max sample weight grows: hardest examples concentrate more weight
    • Later rounds must fit those upweighted points
""")


# =============================================================================
# PART 2: CONVERGENCE AS ROUNDS INCREASE
# =============================================================================

print("  PART 2: CONVERGENCE — ACCURACY vs BOOSTING ROUNDS")
print("  " + "-" * 55)

ada_full = AdaBoostScratch(n_estimators=200, learning_rate=1.0)
ada_full.fit(X_tr, y_tr)

staged_tr = ada_full.train_errors_
staged_te = []
scores_running = np.zeros(len(X_te))
for stump, alpha in zip(ada_full.estimators_, ada_full.alphas_):
    scores_running += alpha * stump.predict(X_te)
    staged_te.append((np.sign(scores_running).astype(int) != y_te).mean())

print(f"\\n  {'M':>6} | {'Train Error':>12} | {'Test Error':>12}")
print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}")
for M in [1, 5, 10, 25, 50, 100, 150, 200]:
    print(f"  {M:>6} | {staged_tr[M-1]:>12.4f} | {staged_te[M-1]:>12.4f}")

min_test_err = min(staged_te)
best_M       = staged_te.index(min_test_err) + 1
print(f"""
  Best test error: {min_test_err:.4f} at M = {best_M}
  Train error → 0 with enough rounds (AdaBoost can perfectly memorise training)
  Test error has a U-shape: improves then overfits with too many rounds.

  COMPARISON with sklearn:
""")

sk_ada    = AdaBoostClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
sk_single = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_tr, y_tr)

print(f"  Single stump (depth=1):     train={accuracy_score(y_tr, sk_single.predict(X_tr)):.4f}  "
      f"test={accuracy_score(y_te, sk_single.predict(X_te)):.4f}")
print(f"  AdaBoost (M=200):           train={1-staged_tr[-1]:.4f}  "
      f"test={accuracy_score(y_te, sk_ada.predict(X_te)):.4f}")


# =============================================================================
# PART 3: ALPHA VALUES AND THEIR MEANING
# =============================================================================

print("\\n" + "=" * 65)
print("  PART 3: THE αₘ VALUES — LEARNER WEIGHTS")
print("=" * 65)

print(f"""
  αₘ = 0.5 * log((1 − εₘ) / εₘ)

  {'εₘ (error)':>12} | {'αₘ (weight)':>12} | Interpretation
  {'-'*12}-+-{'-'*12}-+-{'-'*35}""")

for eps in [0.01, 0.1, 0.2, 0.3, 0.4, 0.49, 0.50, 0.51, 0.6, 0.7]:
    if eps > 0.5:
        alpha = 0.5 * np.log((1 - eps + 1e-9) / (eps + 1e-9))
    else:
        alpha = 0.5 * np.log((1 - eps) / eps)
    if eps < 0.2:
        interp = "strong learner → large positive α"
    elif eps < 0.45:
        interp = "decent learner → moderate positive α"
    elif eps < 0.5:
        interp = "barely better than random → small α"
    elif abs(eps - 0.5) < 0.01:
        interp = "random → α ≈ 0, no contribution"
    else:
        interp = "worse than random → NEGATIVE α (flip prediction)"
    print(f"  {eps:>12.2f} | {alpha:>12.4f} | {interp}")

print(f"""
  KEY INSIGHT: Even a classifier that is WORSE than random (ε > 0.5) is
  useful — it gets a negative weight, which effectively flips its predictions.
  AdaBoost fails only when the weak learner is exactly random (ε = 0.5).
""")


# =============================================================================
# PART 4: SENSITIVITY TO OUTLIERS
# =============================================================================

print("=" * 65)
print("  PART 4: ADABOOST SENSITIVITY TO OUTLIERS (EXPONENTIAL LOSS)")
print("=" * 65)

# Clean dataset
X_clean, y_clean_bin = make_classification(200, 2, n_informative=2,
                                            n_redundant=0, class_sep=2.0,
                                            random_state=42)
y_clean = 2 * y_clean_bin - 1

# Noisy dataset: 10 mislabelled examples
y_noisy = y_clean.copy()
noise_idx = np.random.choice(len(y_noisy), 10, replace=False)
y_noisy[noise_idx] *= -1   # flip labels

Xcl_tr, Xcl_te, ycl_tr, ycl_te = train_test_split(X_clean, y_clean, test_size=0.3)
Xn_tr, Xn_te, yn_tr, yn_te     = train_test_split(X_clean, y_noisy, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

models = {
    "Single Tree (depth=3)": DecisionTreeClassifier(max_depth=3),
    "Random Forest (n=100)": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost (n=200)":      AdaBoostClassifier(n_estimators=200, random_state=42),
}

print(f"  {'Model':>30} | {'Clean Test Acc':>15} | {'Noisy Test Acc':>15} | Drop")
print(f"  {'-'*30}-+-{'-'*15}-+-{'-'*15}-+-{'-'*8}")

for name, model in models.items():
    model_clean = type(model)(**model.get_params())
    model_noisy = type(model)(**model.get_params())
    model_clean.fit(Xcl_tr, ycl_tr)
    model_noisy.fit(Xn_tr, yn_tr)
    acc_clean = accuracy_score(ycl_te, model_clean.predict(Xcl_te))
    acc_noisy = accuracy_score(yn_te, model_noisy.predict(Xn_te))
    drop      = acc_clean - acc_noisy
    print(f"  {name:>30} | {acc_clean:>15.4f} | {acc_noisy:>15.4f} | {drop:>+.4f}")

print(f"""
  AdaBoost is most sensitive to noise because the exponential loss
  e^{{-y·F(x)}} grows unboundedly for large errors.
  Mislabelled points get very high weights → the ensemble distorts around them.

  Use Gradient Boosting with log-loss or MSE for noise-robust boosting.
  The loss function choice is critical when data quality is uncertain.
""")
''',
    },

    "Gradient Boosting In Depth": {
        "description": "Gradient Boosting from scratch with MSE and log-loss, XGBoost vs LightGBM comparison",
        "runnable": True,
        "timeout": None,
        "code": '''
"""
================================================================================
GRADIENT BOOSTING FROM SCRATCH — GRADIENT DESCENT IN FUNCTION SPACE
================================================================================

We implement the core Gradient Boosting loop manually:

    F₀(x) = mean(y)
    For m = 1, ..., M:
        rᵢₘ = − ∂L/∂F(xᵢ)   (negative gradient = pseudo-residuals)
        Fit tree hₘ to {(xᵢ, rᵢₘ)}ᵢ
        γₘ = argmin_γ Σᵢ L(yᵢ, Fₘ₋₁(xᵢ) + γ hₘ(xᵢ))   (line search)
        Fₘ(x) = Fₘ₋₁(x) + η · γₘ · hₘ(x)

For MSE loss: rᵢₘ = yᵢ − Fₘ₋₁(xᵢ)   (the residuals literally)
For log-loss: rᵢₘ = yᵢ − pᵢ           (residuals in probability space)

================================================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

np.random.seed(42)


# =============================================================================
# GRADIENT BOOSTING REGRESSOR FROM SCRATCH (MSE LOSS)
# =============================================================================

print("=" * 65)
print("  PART 1: GRADIENT BOOSTING REGRESSOR (MSE LOSS) FROM SCRATCH")
print("=" * 65)


class GradientBoostingRegressorScratch:
    """
    Gradient Boosting for regression using MSE loss.

    With MSE loss  L(y, F) = (1/2)(y-F)²:
        pseudo-residual = -∂L/∂F = y - F(x)

    This is literally fitting each tree to the residuals of the current model.
    The γₘ (leaf output values) are computed as the mean of residuals in each leaf.

    Key hyperparameters:
        n_estimators:   number of trees M
        learning_rate:  η — shrinks each tree's contribution
        max_depth:      tree depth (controls interaction order)
        subsample:      row sampling fraction (stochastic GBM)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, subsample=1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.subsample     = subsample
        self.F0_           = None
        self.trees_        = []
        self.train_mse_    = []

    def fit(self, X, y):
        n = len(X)
        rng = np.random.default_rng(42)

        # F₀: initial prediction = mean(y)
        self.F0_ = y.mean()
        F = np.full(n, self.F0_)

        self.trees_ = []
        for m in range(self.n_estimators):
            # ── Pseudo-residuals (negative gradient of MSE) ───────────────────
            residuals = y - F    # for MSE: rᵢ = yᵢ − F(xᵢ)

            # ── Optional row subsampling (stochastic GBM) ─────────────────────
            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(n * self.subsample), replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals

            # ── Fit tree to pseudo-residuals ───────────────────────────────────
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sub, r_sub)

            # ── Update the ensemble prediction ─────────────────────────────────
            F += self.learning_rate * tree.predict(X)

            self.trees_.append(tree)
            self.train_mse_.append(mean_squared_error(y, F))

        return self

    def predict(self, X):
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        """Yield prediction after each stage (for convergence plots)."""
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
            yield F.copy()


# --- Dataset ---
X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
X_rt, X_rv, y_rt, y_rv = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print(f"\\n  Dataset: make_regression (n=500, p=10, noise=20)")
print(f"  Baseline (predict mean): RMSE = {np.sqrt(mean_squared_error(y_rv, [y_rt.mean()]*len(y_rv))):.2f}\\n")

print(f"  {'M':>6} | {'Train RMSE':>12} | {'Test RMSE':>12} | {'Reduction':>10}")
print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

gbr = GradientBoostingRegressorScratch(n_estimators=200, learning_rate=0.1, max_depth=3)
gbr.fit(X_rt, y_rt)

baseline_rmse = np.sqrt(mean_squared_error(y_rv, [y_rt.mean()]*len(y_rv)))

for M in [1, 5, 10, 25, 50, 100, 150, 200]:
    stage_preds = list(gbr.staged_predict(X_rv))
    test_rmse = np.sqrt(mean_squared_error(y_rv, stage_preds[M-1]))
    train_rmse = np.sqrt(gbr.train_mse_[M-1])
    reduction  = (baseline_rmse - test_rmse) / baseline_rmse * 100
    print(f"  {M:>6} | {train_rmse:>12.2f} | {test_rmse:>12.2f} | {reduction:>9.1f}%")


# =============================================================================
# PART 2: LEARNING RATE AND N_ESTIMATORS TRADEOFF
# =============================================================================

print("\\n" + "=" * 65)
print("  PART 2: LEARNING RATE × N_ESTIMATORS TRADEOFF")
print("=" * 65)

print(f"""
  Shrinkage (learning_rate η) scales down each tree's contribution.
  Smaller η → need more trees, but better generalisation:

    F(x) = F₀ + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)

  This slows down the function-space gradient descent → less overfitting.
  Rule of thumb: halve the learning rate → double n_estimators for same accuracy.
""")

print(f"  {'lr':>6} | {'n_est':>7} | {'Test RMSE':>10} | Bar (relative quality)")
print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*10}-+-{'-'*30}")

configs = [
    (1.0, 10), (0.5, 20), (0.2, 50), (0.1, 100), (0.05, 200), (0.01, 500)
]

rmses = []
for lr, n_est in configs:
    sk_gbr = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr,
                                        max_depth=3, random_state=42)
    sk_gbr.fit(X_rt, y_rt)
    rmse = np.sqrt(mean_squared_error(y_rv, sk_gbr.predict(X_rv)))
    rmses.append(rmse)

best_rmse = min(rmses)
for (lr, n_est), rmse in zip(configs, rmses):
    bar = "█" * int(30 * best_rmse / rmse)
    print(f"  {lr:>6.2f} | {n_est:>7} | {rmse:>10.2f} | {bar}")

print(f"""
  Conclusion: small learning_rate + many trees consistently outperforms
  large learning_rate + few trees for the same total "work" done.
  The best performing config is usually the smallest lr you can afford.
""")


# =============================================================================
# PART 3: GRADIENT BOOSTING CLASSIFIER — LOG LOSS
# =============================================================================

print("=" * 65)
print("  PART 3: GRADIENT BOOSTING CLASSIFIER — LOG LOSS RESIDUALS")
print("=" * 65)

print(f"""
  For log-loss (binary classification):
    F(x) = log-odds:  p(x) = sigmoid(F(x)) = 1 / (1 + exp(-F(x)))
    L(y, F) = -y·F + log(1 + exp(F))
    Pseudo-residuals: rᵢ = yᵢ - p(xᵢ)    (true label - predicted probability)

  This is identical to the MSE algorithm — only the residuals change.
  The tree still fits residuals; it's just probability residuals now.
""")

X_cls, y_cls = make_classification(n_samples=500, n_features=15,
                                    n_informative=8, n_redundant=4,
                                    class_sep=1.5, random_state=42)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cls, y_cls, test_size=0.25)

models = {
    "GradBoost (lr=0.1, n=100)":  GradientBoostingClassifier(n_estimators=100,  learning_rate=0.1),
    "GradBoost (lr=0.05, n=200)": GradientBoostingClassifier(n_estimators=200,  learning_rate=0.05),
    "GradBoost (lr=0.01, n=500)": GradientBoostingClassifier(n_estimators=500,  learning_rate=0.01),
}

print(f"  {'Model':>35} | {'Train Acc':>10} | {'Test Acc':>10} | {'Log-Loss':>10}")
print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for name, model in models.items():
    model.fit(Xc_tr, yc_tr)
    tr_acc = accuracy_score(yc_tr, model.predict(Xc_tr))
    te_acc = accuracy_score(yc_te, model.predict(Xc_te))
    ll     = log_loss(yc_te, model.predict_proba(Xc_te))
    print(f"  {name:>35} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {ll:>10.4f}")


# =============================================================================
# PART 4: FEATURE IMPORTANCE IN GRADIENT BOOSTING
# =============================================================================

print("\\n" + "=" * 65)
print("  PART 4: FEATURE IMPORTANCE — GAIN vs SPLIT COUNT")
print("=" * 65)

gb_fi = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=3, random_state=42)
gb_fi.fit(Xc_tr, yc_tr)

importances  = gb_fi.feature_importances_
sorted_idx   = np.argsort(importances)[::-1]

print(f"\\n  Top 10 features (out of {X_cls.shape[1]}):")
print(f"  {'Rank':>5} | {'Feature':>12} | {'Importance':>12} | Bar")
print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*30}")

for rank, idx in enumerate(sorted_idx[:10], 1):
    bar = "█" * int(importances[idx] * 200)
    print(f"  {rank:>5} | {'feature_'+str(idx):>12} | {importances[idx]:>12.4f} | {bar}")

print(f"""
  Gradient Boosting importance = total reduction in loss attributable to
  splits on each feature, summed across all trees (similar to MDI in RF).

  For a more reliable measure, use SHAP values (pip install shap):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

  SHAP decomposes each prediction into feature contributions — it tells
  you not just which features matter, but HOW and in which direction.
""")
''',
    },

    "Stacking and Meta-Learning": {
        "description": "Out-of-fold predictions, meta-learner training, and comparing ensemble strategies",
        "runnable": True,
        "timeout": None,
        "code": '''
"""
================================================================================
STACKING — TRAINING A META-LEARNER ON BASE LEARNER PREDICTIONS
================================================================================

Stacking pipeline:
    Level 0 (base learners): Random Forest, Gradient Boosting, Logistic Reg.
    Level 1 (meta-learner):  Logistic Regression trained on OOF predictions

The critical step: out-of-fold (OOF) predictions ensure the meta-learner
trains on HONEST predictions (base learners predicting unseen data).

Without OOF:
    Base learners trained on train set → predict train set → overfit scores
    Meta-learner trains on these inflated predictions → learns wrong thing

With OOF (k-fold):
    For each fold i:
        Train base learners on folds ≠ i
        Predict fold i → honest, OOF prediction
    Stack OOF predictions as meta-features → honest training for meta-learner

================================================================================
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer

np.random.seed(42)


# =============================================================================
# STACKING FROM SCRATCH
# =============================================================================

class StackingClassifierScratch:
    """
    Two-level stacking:
        Level 0: list of base learners, trained with k-fold OOF
        Level 1: meta-learner trained on OOF predictions

    At test time:
        All base learners (retrained on full train) predict test
        Meta-learner combines those predictions into final output
    """

    def __init__(self, base_learners, meta_learner, n_folds=5,
                 use_proba=True):
        """
        base_learners:  list of (name, sklearn_estimator) pairs
        meta_learner:   sklearn estimator
        n_folds:        number of CV folds for OOF
        use_proba:      if True, pass class probabilities to meta-learner
                        (more information than hard labels)
        """
        self.base_learners = base_learners
        self.meta_learner  = meta_learner
        self.n_folds       = n_folds
        self.use_proba     = use_proba
        self.fitted_base_  = []

    def _get_preds(self, model, X):
        if self.use_proba and hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]   # P(class=1)
        else:
            return model.predict(X).astype(float)

    def fit(self, X, y):
        n = len(X)
        n_base = len(self.base_learners)
        Z_train = np.zeros((n, n_base))   # meta-features for training

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        print(f"  Generating out-of-fold predictions ({self.n_folds}-fold CV)...")
        for j, (name, learner) in enumerate(self.base_learners):
            oof_preds = np.zeros(n)
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
                clone = type(learner)(**learner.get_params())
                clone.fit(X[tr_idx], y[tr_idx])
                oof_preds[val_idx] = self._get_preds(clone, X[val_idx])
            Z_train[:, j] = oof_preds
            print(f"    {name:>30}: OOF AUC = {roc_auc_score(y, oof_preds):.4f}")

        # Train meta-learner on OOF predictions
        self.meta_learner.fit(Z_train, y)

        # Retrain base learners on FULL training data for test time
        self.fitted_base_ = []
        for name, learner in self.base_learners:
            full_model = type(learner)(**learner.get_params())
            full_model.fit(X, y)
            self.fitted_base_.append(full_model)

        return self

    def predict(self, X):
        Z_test = np.column_stack([
            self._get_preds(model, X) for model in self.fitted_base_
        ])
        return self.meta_learner.predict(Z_test)

    def predict_proba(self, X):
        Z_test = np.column_stack([
            self._get_preds(model, X) for model in self.fitted_base_
        ])
        return self.meta_learner.predict_proba(Z_test)


# =============================================================================
# DATASET
# =============================================================================

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2,
                                            stratify=y, random_state=42)

print("=" * 65)
print("  STACKING FROM SCRATCH — BREAST CANCER DATASET")
print("=" * 65)
print(f"  n = {len(X)}, p = {X.shape[1]}")
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")


# =============================================================================
# PART 1: BASE LEARNER PERFORMANCE
# =============================================================================

print("\\n  PART 1: BASE LEARNER PERFORMANCE (STANDALONE)")
print("  " + "-" * 50)

base_learners = [
    ("Random Forest",        RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting",    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ("Logistic Regression",  LogisticRegression(C=1.0, max_iter=1000)),
    ("SVM (RBF)",            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
]

print(f"\\n  {'Model':>30} | {'Train Acc':>10} | {'Test Acc':>10} | {'Test AUC':>10}")
print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for name, model in base_learners:
    model.fit(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    te_acc = accuracy_score(y_te, model.predict(X_te))
    te_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"  {name:>30} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {te_auc:>10.4f}")


# =============================================================================
# PART 2: STACKING
# =============================================================================

print("\\n  PART 2: STACKING (OOF META-FEATURES)")
print("  " + "-" * 50)

# Re-create base learners (fresh, unfitted)
base_learners_fresh = [
    ("Random Forest",        RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting",    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ("Logistic Regression",  LogisticRegression(C=1.0, max_iter=1000)),
    ("SVM (RBF)",            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
]

meta_learner = LogisticRegression(C=1.0, max_iter=1000)

stacker = StackingClassifierScratch(
    base_learners=base_learners_fresh,
    meta_learner=meta_learner,
    n_folds=5,
    use_proba=True
)

print()
stacker.fit(X_tr, y_tr)

stack_acc = accuracy_score(y_te, stacker.predict(X_te))
stack_auc = roc_auc_score(y_te, stacker.predict_proba(X_te)[:, 1])
print(f"\\n  Stacking Ensemble:")
print(f"    Test Accuracy: {stack_acc:.4f}")
print(f"    Test AUC:      {stack_auc:.4f}")

# Compare with sklearn StackingClassifier
print(f"\\n  Comparison — sklearn StackingClassifier:")
sk_stack = StackingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb",  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000)),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=5,
    passthrough=False
)
sk_stack.fit(X_tr, y_tr)
sk_acc = accuracy_score(y_te, sk_stack.predict(X_te))
sk_auc = roc_auc_score(y_te, sk_stack.predict_proba(X_te)[:, 1])
print(f"    Test Accuracy: {sk_acc:.4f}")
print(f"    Test AUC:      {sk_auc:.4f}")


# =============================================================================
# PART 3: META-LEARNER COEFFICIENTS — WHAT DID IT LEARN?
# =============================================================================

print("\\n  PART 3: META-LEARNER COEFFICIENTS — TRUST IN EACH BASE LEARNER")
print("  " + "-" * 50)

meta_coefs = meta_learner.coef_[0]
meta_names = [name for name, _ in base_learners_fresh]

print(f"\\n  Meta-learner (Logistic Regression) feature weights:")
print(f"  Each weight = how much the meta-learner trusts each base model's")
print(f"  probability estimate.\\n")
print(f"  {'Base Learner':>30} | {'Meta-weight':>12} | {'Direction'}")
print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*30}")

for name, coef in zip(meta_names, meta_coefs):
    direction = "↑ class 1 prediction" if coef > 0 else "↓ class 1 prediction"
    bar = "+" * int(abs(coef) * 5) if coef > 0 else "-" * int(abs(coef) * 5)
    print(f"  {name:>30} | {coef:>+12.4f} | {bar}")

print(f"""
  The meta-learner has learned to weight base models by their reliability.
  Negative weights can occur when a base model's probability estimates are
  poorly calibrated (it may be overconfident in the wrong direction).

  KEY STACKING TIPS:
    1. Use OOF predictions — always. Direct predictions on training data leak.
    2. Add original features to meta-learner inputs (passthrough=True) for
       cases where base models disagree and original features resolve ambiguity.
    3. Keep the meta-learner SIMPLE (Logistic Regression, Ridge).
       A complex meta-learner will overfit the OOF predictions.
    4. Stacking gains most when base learners are DIVERSE (different algorithms,
       different feature subsets, different hyperparameter regimes).
""")


# =============================================================================
# PART 4: FULL COMPARISON TABLE
# =============================================================================

print("=" * 65)
print("  PART 4: FULL ENSEMBLE COMPARISON SUMMARY")
print("=" * 65)

from sklearn.ensemble import AdaBoostClassifier, VotingClassifier

all_models = {
    "Single Tree (depth=5)":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest (n=100)":       RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost (n=200)":            AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting (n=100)":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Voting (hard vote)":          VotingClassifier(voting="hard", estimators=[
                                       ("rf", RandomForestClassifier(100, random_state=42)),
                                       ("gb", GradientBoostingClassifier(100, random_state=42)),
                                       ("lr", LogisticRegression(max_iter=1000)),
                                   ]),
    "Stacking (scratch)":          None,   # already computed
}

from sklearn.tree import DecisionTreeClassifier

print(f"\\n  {'Model':>35} | {'Test Acc':>10} | {'Test AUC':>10}")
print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}")

for name, model in all_models.items():
    if model is None:
        print(f"  {'Stacking (scratch)':>35} | {stack_acc:>10.4f} | {stack_auc:>10.4f}")
        continue
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    else:
        auc = float("nan")
    print(f"  {name:>35} | {acc:>10.4f} | {auc:>10.4f}")
''',
    },

    "▶ Run: Ensemble Classification - Adaboost Project": {
        "description": (
            "Runs main.py from the Implementation folder, "
            "if present in your project structure."
        ),
        "runnable": True,
        "timeout": None,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Ensemble Methods"
    / "AdaBoost"
    / "main.py"
)

if not _impl.exists():
    print(f"[INFO] Implementation file not found at: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected path: <project_root>/Implementation/Ensemble Methods/AdaBoost/")
    print("  Create main.py there to use this launcher.")
    sys.exit(0)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },

    "▶ Run: Ensemble Classification - GradientBoost Project": {
        "description": (
            "Runs main.py from the Implementation folder, "
            "if present in your project structure."
        ),
        "runnable": True,
        "timeout": None,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Ensemble Methods"
    / "GradientBoost"
    / "main.py"
)

if not _impl.exists():
    print(f"[INFO] Implementation file not found at: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected path: <project_root>/Implementation/Ensemble Methods/GradientBoost/")
    print("  Create main.py there to use this launcher.")
    sys.exit(0)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },

    "▶ Run: Ensemble Classification - Random_Forest Project": {
        "description": (
            "Runs main.py from the Implementation folder, "
            "if present in your project structure."
        ),
        "runnable": True,
        "timeout": None,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Ensemble Methods"
    / "Random_Forest"
    / "main.py"
)

if not _impl.exists():
    print(f"[INFO] Implementation file not found at: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected path: <project_root>/Implementation/Ensemble Methods/Random_Forest/")
    print("  Create main.py there to use this launcher.")
    sys.exit(0)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },

    "▶ Run: Ensemble Classification - SVM Project": {
        "description": (
            "Runs main.py from the Implementation folder, "
            "if present in your project structure."
        ),
        "runnable": True,
        "timeout": None,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Ensemble Methods"
    / "SVM"
    / "main.py"
)

if not _impl.exists():
    print(f"[INFO] Implementation file not found at: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected path: <project_root>/Implementation/Ensemble Methods/SVM/")
    print("  Create main.py there to use this launcher.")
    sys.exit(0)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },


    "▶ Run: Ensemble Classification - XGBoost": {
        "description": (
            "Runs main.py from the Implementation folder, "
            "if present in your project structure."
        ),
        "runnable": True,
        "timeout": None,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Ensemble Methods"
    / "XGBoost"
    / "main.py"
)

if not _impl.exists():
    print(f"[INFO] Implementation file not found at: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected path: <project_root>/Implementation/Ensemble Methods/XGBoost/")
    print("  Create main.py there to use this launcher.")
    sys.exit(0)

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
# MAIN — Standalone demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    from sklearn.ensemble import (RandomForestClassifier,
                                   GradientBoostingClassifier,
                                   AdaBoostClassifier)
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 65)
    print("  ENSEMBLE METHODS: BAGGING, BOOSTING, STACKING")
    print("=" * 65)
    print("""
  This script demonstrates ensemble methods from the ground up.

  Key Concepts:
    • Ensembles = aggregated weak learners → lower total error
    • Bagging reduces VARIANCE (parallel, independent training)
    • Boosting reduces BIAS (sequential, error-correcting training)
    • Stacking learns the optimal combination of base learner outputs
    • Random Forest = Bagging + feature randomness → decorrelated trees
    • Gradient Boosting = gradient descent in function space
    """)

    np.random.seed(42)

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25,
                                                stratify=y, random_state=42)

    print("=" * 65)
    print("  DATASET: Breast Cancer Wisconsin")
    print("=" * 65)
    print(f"  n={len(X)} samples, {X.shape[1]} features")
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

    from sklearn.tree import DecisionTreeClassifier
    models = {
        "Single Decision Tree":     DecisionTreeClassifier(random_state=42),
        "Random Forest (n=100)":    RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost (n=200)":         AdaBoostClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting (n=100)":GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    print(f"\n  {'Model':>30} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gen Gap':>10}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, model.predict(X_tr))
        te_acc = accuracy_score(y_te, model.predict(X_te))
        gap    = tr_acc - te_acc
        print(f"  {name:>30} | {tr_acc:>10.4f} | {te_acc:>10.4f} | {gap:>+10.4f}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1. Single tree has high variance — test accuracy much lower than train
  2. Random Forest reduces variance via bagging + feature randomness
  3. AdaBoost reduces bias by sequential error correction (watch overfitting with noisy data)
  4. Gradient Boosting = most powerful tabular method, needs careful tuning
  5. Bagging: parallel, robust, good baseline. Boosting: sequential, best accuracy
  6. Feature importance is free from all tree ensembles (MDI or permutation)
  7. OOB score ≈ cross-validation score — use it for free validation
  8. Learning rate × n_estimators are coupled: small lr + many trees wins
  9. Stacking adds modest gains at significant complexity cost
  10. For tabular data: GBM (XGBoost/LightGBM) still beats deep learning (2024)
    """)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    try:
        from Supervised_Learning.visuals.ensemble_methods import (
            ENSEMBLE_VISUAL_HTML,
            ENSEMBLE_VISUAL_HEIGHT,
        )
        visual_html   = ENSEMBLE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = ENSEMBLE_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[08_ensemble_methods.py] Could not load visual: {e}", stacklevel=2)

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


# Dedent all operation code strings
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