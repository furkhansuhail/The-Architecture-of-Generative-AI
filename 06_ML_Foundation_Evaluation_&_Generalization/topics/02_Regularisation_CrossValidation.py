"""
Regularisation, Cross-Validation & Model Evaluation
====================================================

The toolkit for controlling overfitting and honestly measuring how well
a model will perform on data it has never seen.

"""

import textwrap
import re

TOPIC_NAME   = "Regularisation, Cross-Validation & Model Evaluation"
DISPLAY_NAME = "02 · Regularisation & Evaluation"
ICON         = "🔧"
SUBTITLE     = "Controlling Overfitting & Measuring True Performance"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — REGULARISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What is Regularisation?

Regularisation is any technique that deliberately constrains or penalises
a model to prevent it from overfitting. The core idea is to modify the
training objective so the model doesn't just minimise training error — it
also pays a cost for being too complex.

Without regularisation, a powerful model minimises:

    Loss = 1/N × Σ L(yᵢ, ŷᵢ)          ← just fit the data

With regularisation, it minimises:

    Loss = 1/N × Σ L(yᵢ, ŷᵢ)  +  λ × Penalty(weights)

    where λ (lambda) controls how strongly complexity is penalised.
    λ is a hyperparameter YOU choose. Larger λ = stronger regularisation.


──────────────────────────────────────────────────────────────────────────────
### L2 Regularisation (Ridge / Weight Decay)

The penalty is the SUM OF SQUARED WEIGHTS:

    Penalty = Σ wᵢ²      →    Loss = data_loss  +  λ Σ wᵢ²

**What it does:**
    - Pushes ALL weights toward zero, but rarely to exactly zero.
    - Prefers many small weights over a few large ones.
    - Produces smooth, diffuse solutions.
    - Effect on the update rule:
        wᵢ ← wᵢ × (1 - λ·lr)  –  lr × ∂(data_loss)/∂wᵢ
        The factor (1 - λ·lr) < 1 shrinks the weight every step
        → also called "weight decay" in deep learning.

**Geometric intuition:**
    In 2D weight space, the L2 penalty defines a circle of "allowed"
    weights. The optimum lies where the data loss contours touch the circle.
    Since circles have no corners, the solution is usually not sparse —
    most weights are small but non-zero.

    Weight w₂
        ↑
        │      L2 ball (circle)     data loss contours
        │      ╭───────╮            (ellipses)
        │    ╭─┤       ├─╮    ╭──────────────
        │   ╱  │   ★   │  ╲ ╱
        │  │   │ (opti)│   X           ╲
        │   ╲  │       │  ╱╲            ───────╮
        │    ╰─┤       ├─╯             ╭──────
        │      ╰───────╯
        └─────────────────────────── Weight w₁ →
                Solution lands INSIDE the circle — never at a corner.


──────────────────────────────────────────────────────────────────────────────
### L1 Regularisation (Lasso)

The penalty is the SUM OF ABSOLUTE VALUES of weights:

    Penalty = Σ |wᵢ|      →    Loss = data_loss  +  λ Σ |wᵢ|

**What it does:**
    - Pushes weights toward zero, AND often drives some weights to
      EXACTLY zero → automatic feature selection.
    - Produces SPARSE solutions (many weights are exactly 0).
    - This is extremely valuable when you have many features but suspect
      most are irrelevant.

**Why does L1 produce sparsity but L2 doesn't?**
    It's a geometric consequence. The L1 ball is a DIAMOND — it has
    sharp corners at the axes. Loss contours tend to first touch the
    diamond at one of these corners, where one coordinate IS zero.
    The L2 circle has no corners, so the touch point is rarely on an axis.

    Weight w₂
        ↑
        │         L1 ball (diamond)
        │              ★ ← corner on w₂ axis
        │             ╱╲          data loss contours
        │            ╱  ╲        ╭──────────────
        │           ╱    ╲      ╱
        │          ╱      ╲   ╱
        │         ╲        ╲╱  ← tangent at corner (w₁ = 0 here)
        │          ╲       ╱╲
        │           ╲     ╱  ╲─────────────╮
        │            ╲   ╱
        │             ╲ ╱
        │              ★ ← corner on w₁ axis
        └─────────────────────────── Weight w₁ →
                Solution often lands at a CORNER → one weight is exactly 0.


──────────────────────────────────────────────────────────────────────────────
### L1 vs L2 — Side-by-Side Comparison

    ┌──────────────────────┬────────────────────────┬──────────────────────┐
    │ Property             │ L1 (Lasso)             │ L2 (Ridge)           │
    ├──────────────────────┼────────────────────────┼──────────────────────┤
    │ Penalty term         │ λ Σ |wᵢ|               │ λ Σ wᵢ²              │
    │ Solution sparsity    │ YES — exact zeros      │ NO — near-zero only  │
    │ Feature selection    │ Implicit (via zeros)   │ No                   │
    │ Sensitivity to noise │ High (non-smooth)      │ Low (smooth)         │
    │ When to use          │ Suspect irrelevant     │ All features matter  │
    │                      │ features present       │ but too large        │
    │ Differentiable?      │ No (at wᵢ = 0)         │ Yes everywhere       │
    │ Also known as        │ Lasso regression       │ Ridge / weight decay │
    └──────────────────────┴────────────────────────┴──────────────────────┘

**Elastic Net** combines both: λ₁ Σ |wᵢ| + λ₂ Σ wᵢ²
    → sparsity of L1 + stability of L2. Often best in practice.


──────────────────────────────────────────────────────────────────────────────
### Dropout (Neural Network Specific)

Dropout is a regularisation technique specific to neural networks, introduced
by Srivastava et al. in 2014. During TRAINING ONLY, each neuron is randomly
"dropped" (set to zero) with probability p (typically p = 0.2 to 0.5).

    Diagram — Dropout in a Network:

    TRAINING (dropout rate = 0.5)

    Input  ──► [H1] ──► [H2] ──► Output
               ×  (dropped) ×
          ──► [H3] ──────────► [H5] ──►
                        ×  (dropped)
          ──► [H4] ──► [H6] ──────────►

    A different random subset is dropped each forward pass.
    Outputs of surviving neurons are scaled by 1/(1-p) to keep
    the expected activation magnitude the same.

    INFERENCE (no dropout):
    All neurons are active. No scaling needed (handled at training time).

**Why does dropout work?**

    1. Ensemble effect: Each forward pass uses a different sub-network.
       Training hundreds of overlapping networks simultaneously and
       averaging their predictions at test time reduces variance.

    2. Co-adaptation prevention: Neurons can't rely on specific other
       neurons always being present. Each neuron learns more robust,
       independent features.

    3. It approximates a geometric mean of exponentially many models.

    Where to apply it: After dense (fully-connected) layers. Less common
    after convolutional layers (use spatial dropout there). NEVER during
    inference — only during training.


──────────────────────────────────────────────────────────────────────────────
### Batch Normalisation (BN) — Also Has a Regularising Effect

Batch Norm normalises each mini-batch's activations to have mean 0 and
variance 1 before the activation function:

    x̂ᵢ = (xᵢ - μ_batch) / √(σ²_batch + ε)
    yᵢ  = γ x̂ᵢ + β      (γ and β are learned scale and shift parameters)

    Primary purpose: stabilises training (gradient flow, larger lr allowed).
    Secondary benefit: slight regularisation due to mini-batch noise.
    Often REPLACES dropout in modern architectures (ResNet, Transformers).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — TRAIN / VALIDATION / TEST SPLIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Three-Way Data Split

To get an honest estimate of generalisation performance, data must be split
into THREE non-overlapping sets:

    ┌─────────────────────────────────────────────────────────────┐
    │ ALL DATA                                                    │
    │ ┌──────────────────────┬──────────────┬───────────────────┐ │
    │ │   TRAINING SET       │ VALIDATION   │    TEST SET       │ │
    │ │   (60-70%)           │ SET (10-20%) │    (10-20%)       │ │
    │ │                      │              │                   │ │
    │ │  Used to update      │ Used to tune │  Used ONCE, at    │ │
    │ │  model weights       │ hyperparams  │  the very end     │ │
    │ └──────────────────────┴──────────────┴───────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

**Training Set** — the model sees this data and updates its parameters.
**Validation Set** — used during development to:
    - Compare different models
    - Tune hyperparameters (λ, learning rate, depth, dropout rate)
    - Detect overfitting via learning curves
**Test Set** — the final, unbiased performance estimate. Rules:
    - Touch it ONCE, after ALL development decisions are final.
    - Never use test performance to make design decisions.
    - If you do, the test set becomes a validation set (data leakage).


### Data Leakage — The Silent Killer

Data leakage occurs when information from the validation or test set
"leaks" into the training process. This produces an overly optimistic
performance estimate that evaporates in production.

Common leakage sources:
    1. Normalising/scaling using statistics from the full dataset
       (compute mean/std on training set ONLY, then apply to val/test)
    2. Feature selection using the full dataset before splitting
    3. Reusing the test set to select models → test becomes validation
    4. Temporal data with future information leaking backward
    5. Duplicate examples split across train and test sets


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — CROSS-VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Cross-Validation?

A single train/validation split is NOISY. If you happen to get an easy
validation set, your estimate is too optimistic. If you get a hard one, too
pessimistic. With small datasets this variance is especially large.

k-Fold Cross-Validation solves this by using EVERY example for both
training and validation, and averaging the results.


### k-Fold Cross-Validation

    Algorithm:
    1. Shuffle the data (unless temporal/ordered)
    2. Split into k equal "folds" (typically k = 5 or 10)
    3. For each fold i from 1 to k:
       - Use fold i as the validation set
       - Use all other k-1 folds as the training set
       - Train a fresh model, record validation performance
    4. Final estimate = mean of k validation scores


    Diagram — 5-Fold Cross-Validation:

    Fold 1:  [VAL] [tr ] [tr ] [tr ] [tr ]    score₁
    Fold 2:  [tr ] [VAL] [tr ] [tr ] [tr ]    score₂
    Fold 3:  [tr ] [tr ] [VAL] [tr ] [tr ]    score₃
    Fold 4:  [tr ] [tr ] [tr ] [VAL] [tr ]    score₄
    Fold 5:  [tr ] [tr ] [tr ] [tr ] [VAL]    score₅
                                              ───────
                               Final score = mean(score₁ … score₅)
                               Uncertainty =  std(score₁ … score₅)

    - Every example is validated exactly ONCE.
    - Every example is used for training k-1 = 4 times.
    - k=10 gives lower variance but costs 10× the compute.


### Stratified k-Fold

For classification, plain k-fold can produce folds with different class
distributions by chance. Stratified k-fold ensures each fold has the
same class proportions as the full dataset.

Always prefer stratified k-fold for classification tasks.


### Leave-One-Out Cross-Validation (LOOCV)

Extreme case: k = N (each fold is a single example).

    - Lowest bias estimate (each model trained on almost all data)
    - Highest variance (each model is nearly identical)
    - Computationally expensive: N model fits instead of k
    - Only practical for very small datasets or cheap models


### Time-Series Cross-Validation

For ordered data (time series), you CANNOT shuffle! A future observation
cannot be used to predict the past. Use expanding or sliding window CV:

    Expanding window (walk-forward):
    Train → [────]                Validate → [■]
    Train → [─────]               Validate → [■]
    Train → [──────]              Validate → [■]
    Train → [───────]             Validate → [■]

    Each fold expands the training window and tests the NEXT time step.
    This mirrors real-world deployment where the model always predicts future.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — EVALUATION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Confusion Matrix (Foundation for Classification Metrics)

    Predicted  →       Positive         Negative
    Actual ↓
    Positive      True Positive (TP)   False Negative (FN)   ← missed positives
    Negative      False Positive (FP)  True Negative (TN)    ← false alarms

    TP = model said positive, was correct
    TN = model said negative, was correct
    FP = model said positive, was wrong (Type I error — false alarm)
    FN = model said negative, was wrong (Type II error — missed detection)


### Classification Metrics

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)
    - Overall fraction correct.
    - MISLEADING on imbalanced datasets.
    - Example: 99% negative class → predict all negative → 99% accuracy,
      zero useful predictions. Always check class balance first.

**Precision** = TP / (TP + FP)
    "Of all the positive predictions, how many were actually positive?"
    - High precision → few false alarms.
    - Critical when false positives are costly (spam filter: don't delete real email).

**Recall (Sensitivity)** = TP / (TP + FN)
    "Of all the actual positives, how many did we catch?"
    - High recall → few missed detections.
    - Critical when false negatives are costly (cancer screening: don't miss disease).

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
    - Harmonic mean of precision and recall.
    - Best single number when you care about both and classes are imbalanced.
    - Use F1 for imbalanced datasets instead of accuracy.

**ROC-AUC**
    - ROC curve: plots True Positive Rate vs False Positive Rate at every
      threshold. AUC (Area Under Curve) = probability that the model ranks
      a random positive example higher than a random negative one.
    - AUC = 0.5 → random chance. AUC = 1.0 → perfect. AUC = 0.0 → inverse.
    - Threshold-independent → good for comparing models regardless of cutoff.

**Precision-Recall AUC (PR-AUC)**
    - Better than ROC-AUC for highly imbalanced datasets where negatives
      dominate. ROC-AUC can be misleadingly high even if the model fails
      on the rare positive class.


    Metric Decision Guide:
    ┌───────────────────────────────────────────────────────────────┐
    │ Scenario                         │ Preferred Metric           │
    ├───────────────────────────────────────────────────────────────┤
    │ Balanced classes, simple task    │ Accuracy                   │
    │ Cost of false positive > FN      │ Precision                  │
    │ Cost of false negative > FP      │ Recall                     │
    │ Both costs matter, imbalanced    │ F1 Score                   │
    │ Threshold selection matters      │ ROC-AUC                    │
    │ Very imbalanced (rare events)    │ PR-AUC                     │
    └───────────────────────────────────────────────────────────────┘


### Regression Metrics

**MSE** = 1/N × Σ (yᵢ - ŷᵢ)²
    - Mean Squared Error. Penalises LARGE errors heavily (squared).
    - Same units as y² → harder to interpret directly.
    - Sensitive to outliers.

**RMSE** = √MSE
    - Root Mean Squared Error. Same units as y → interpretable.
    - Still sensitive to outliers due to squaring before root.

**MAE** = 1/N × Σ |yᵢ - ŷᵢ|
    - Mean Absolute Error. Treats all errors equally.
    - More robust to outliers than MSE.
    - Not differentiable at zero (minor issue for gradient methods).

**R² (Coefficient of Determination)** = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
    - "What fraction of the variance in y does the model explain?"
    - R² = 1 → perfect prediction.
    - R² = 0 → model is as good as predicting the mean.
    - R² < 0 → model is WORSE than predicting the mean (very bad).
    - NOT always a reliable metric (can be gamed, insensitive to bias).

**MAPE** = 100/N × Σ |yᵢ - ŷᵢ| / |yᵢ|
    - Mean Absolute Percentage Error. Scale-independent → easy to explain.
    - Undefined when yᵢ = 0. Asymmetric (penalises under-prediction more).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — HYPERPARAMETER TUNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What are Hyperparameters?

Parameters you set BEFORE training — they are not learned from data:
    - λ (regularisation strength)
    - Learning rate, batch size, number of epochs
    - Number of layers / neurons per layer
    - Dropout rate, kernel size in CNNs
    - k in k-NN, C and γ in SVM, max_depth in decision trees

The process of finding good hyperparameters is called *hyperparameter tuning*
or *hyperparameter optimisation (HPO)*.

CRITICAL RULE: always tune hyperparameters using the VALIDATION set.
Never use test set performance to select hyperparameters.


### Tuning Strategies

**Grid Search**: Exhaustively try every combination in a pre-defined grid.
    - Simple, interpretable, reproducible.
    - Scales EXPONENTIALLY with number of hyperparameters.
    - For 4 hyperparameters × 5 values each = 5⁴ = 625 experiments.

**Random Search**: Sample hyperparameter combinations uniformly at random.
    - Surprisingly effective. Bergstra & Bengio (2012) showed random
      search often outperforms grid search with fewer evaluations.
    - Why? Most hyperparameter landscapes have only a few that matter.
      Random search doesn't waste budget on unimportant dimensions.

    Grid vs Random — why random often wins:
    ┌────────────────┬───────────────────────────────────────────┐
    │                │    λ₁ (important)                         │
    │  GRID SEARCH   │   ●   ●   ●   ●   ●                       │
    │  5×5 = 25 pts  │   ●   ●   ●   ●   ●                       │
    │                │   ●   ●   ●   ●   ●    ← only 5 unique    │
    │  λ₂            │   ●   ●   ●   ●   ●      values for λ₁    │
    │  (unimportant) │   ●   ●   ●   ●   ●                       │
    ├────────────────┼───────────────────────────────────────────┤
    │                │    λ₁ (important)                         │
    │  RANDOM SEARCH │   · ·   ·       ·                         │
    │  25 pts        │ ·   · ·   ·  ·   ·                        │
    │                │  ·   ·  ·     ·  · ←  ~25 unique values   │
    │  λ₂            │    ·   ·   ·    ·      for λ₁             │
    │  (unimportant) │  ·    ·     · ·                           │
    └────────────────┴───────────────────────────────────────────┘
    Random covers the important dimension much more thoroughly.

**Bayesian Optimisation**: Fit a probabilistic surrogate model of the
    objective function. Use it to pick the next hyperparameter configuration
    that is most likely to improve performance. Much more sample-efficient
    than grid or random search for expensive models.
    Tools: Optuna, Hyperopt, Scikit-Optimize, Ray Tune.

**Population-Based Training (PBT)**: Evolve a population of models in
    parallel, exploiting good configurations and exploring new ones.
    Used by DeepMind for large-scale training.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · L1 vs L2 Regularisation Effect on Weights": {
        "description": (
            "Train linear regression with L1 (Lasso) and L2 (Ridge) regularisation "
            "across different λ values. Observe how L1 drives weights to exactly zero "
            "while L2 merely shrinks them."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _sp_opt
from scipy.stats import loguniform  # kept from scipy (not affected by sklearn issue)

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── StandardScaler ────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── make_regression ───────────────────────────────────────────────────────────
def make_regression(n_samples=100,n_features=100,n_informative=10,noise=0.0,
                    coef=False,random_state=None):
    rng=np.random.default_rng(random_state)
    ground_truth=np.zeros(n_features)
    idx=rng.choice(n_features,n_informative,replace=False)
    ground_truth[idx]=rng.uniform(10,100,n_informative)*rng.choice([-1,1],n_informative)
    X=rng.standard_normal((n_samples,n_features))
    y=X@ground_truth+rng.normal(0,noise,n_samples)
    return (X,y,ground_truth) if coef else (X,y)

# ── make_classification ───────────────────────────────────────────────────────
def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── make_moons ────────────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

# ── load_breast_cancer (synthetic substitute) ─────────────────────────────────
def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf))
    X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

# ── train_test_split ──────────────────────────────────────────────────────────
def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

# ── StratifiedKFold ───────────────────────────────────────────────────────────
class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,class_weight=None,solver="lbfgs",
                 penalty="l2",random_state=None):
        self.C=C; self.max_iter=max_iter; self.class_weight=class_weight
        self.penalty=penalty
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y):
        n,d=X.shape; sw=self._sw(y); lam=1./self.C; w0=np.zeros(d+1)
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso (coordinate descent) ────────────────────────────────────────────────
class Lasso:
    def __init__(self,alpha=1.0,max_iter=1000): self.alpha=alpha; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=y-X@w+X[:,j]*w[j]; rho=X[:,j]@r; z=np.sum(X[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w; self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Ridge ─────────────────────────────────────────────────────────────────────
class Ridge:
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        n,d=X.shape
        self.coef_=np.linalg.solve(X.T@X+self.alpha*np.eye(d),X.T@y)
        self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Metrics ───────────────────────────────────────────────────────────────────
def confusion_matrix(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return np.array([[tn,fp],[fn,tp]])

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

def precision_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)

def recall_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fn=((yp==0)&(yt==1)).sum()
    return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)

def f1_score(yt,yp,zero_division=0):
    p=precision_score(yt,yp,zero_division); r=recall_score(yt,yp,zero_division)
    return float(2*p*r/(p+r)) if (p+r)>0 else float(zero_division)

def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))

def roc_curve(yt,ys):
    thrs=np.concatenate([[1.0+1e-9],np.sort(np.unique(ys))[::-1]])
    tprs,fprs=[0.],[0.]
    pos=yt.sum(); neg=len(yt)-pos
    for t in thrs[1:]:
        yp=(ys>=t).astype(int)
        tprs.append(((yp==1)&(yt==1)).sum()/pos if pos>0 else 0)
        fprs.append(((yp==1)&(yt==0)).sum()/neg if neg>0 else 0)
    return np.array(fprs),np.array(tprs),thrs

def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── GridSearchCV / RandomizedSearchCV ─────────────────────────────────────────
class GridSearchCV:
    def __init__(self,estimator,param_grid,cv=5,scoring="accuracy",
                 n_jobs=None,verbose=0):
        self.estimator=estimator; self.param_grid=param_grid; self.cv=cv
        self.scoring=scoring
    def _configs(self):
        keys=list(self.param_grid.keys()); vals=list(self.param_grid.values())
        import itertools
        for combo in itertools.product(*vals):
            yield dict(zip(keys,combo))
    def fit(self,X,y):
        configs=list(self._configs())
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for cfg in configs:
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

class RandomizedSearchCV:
    def __init__(self,estimator,param_distributions,n_iter=10,cv=5,
                 scoring="accuracy",n_jobs=None,random_state=None,verbose=0):
        self.estimator=estimator; self.param_distributions=param_distributions
        self.n_iter=n_iter; self.cv=cv; self.scoring=scoring
        self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for _ in range(self.n_iter):
            cfg={}
            for k,dist in self.param_distributions.items():
                if hasattr(dist,"rvs"): cfg[k]=float(dist.rvs(random_state=int(rng.integers(0,10**9))))
                else: cfg[k]=dist[rng.integers(0,len(dist))]
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

np.random.seed(42)

# ── Dataset: 200 samples, 20 features, only 5 are truly relevant ──────────
X, y, true_coefs = make_regression(
    n_samples=200, n_features=20, n_informative=5,
    noise=15, coef=True, random_state=42
)

# Standardise features (required for fair regularisation comparison)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── Sweep λ values ────────────────────────────────────────────────────────
lambdas = np.logspace(-3, 2, 60)    # 0.001 → 100

lasso_coefs = []
ridge_coefs = []

for lam in lambdas:
    lasso = Lasso(alpha=lam, max_iter=10_000)
    ridge = Ridge(alpha=lam)
    lasso.fit(X, y)
    ridge.fit(X, y)
    lasso_coefs.append(lasso.coef_.copy())
    ridge_coefs.append(ridge.coef_.copy())

lasso_coefs = np.array(lasso_coefs)
ridge_coefs = np.array(ridge_coefs)

# ── Plot regularisation paths ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("L1 vs L2 Regularisation Paths  (each line = one feature weight)",
             fontsize=13, fontweight="bold")

for ax, coefs, title, color in zip(
        axes,
        [lasso_coefs, ridge_coefs],
        ["L1 / Lasso — Sparse Solution (weights hit zero)",
         "L2 / Ridge — Dense Solution (weights shrink toward zero)"],
        ["tomato", "steelblue"]):
    for i in range(coefs.shape[1]):
        ax.semilogx(lambdas, coefs[:, i], alpha=0.6, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Regularisation strength λ (log scale)", fontsize=11)
    ax.set_ylabel("Weight value", fontsize=11)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("l1_vs_l2_paths.png", dpi=120)

# ── Print numerical comparison at specific λ values ───────────────────────
print("=" * 65)
print("  L1 vs L2 REGULARISATION: WEIGHT ANALYSIS")
print("=" * 65)
print()
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Truly relevant features: 5 out of 20")
print()

for lam in [0.01, 0.5, 5.0, 50.0]:
    lasso = Lasso(alpha=lam, max_iter=10_000).fit(X, y)
    ridge = Ridge(alpha=lam).fit(X, y)

    lasso_zeros = np.sum(np.abs(lasso.coef_) < 1e-4)
    ridge_zeros = np.sum(np.abs(ridge.coef_) < 1e-4)

    print(f"  λ = {lam:.2f}")
    print(f"    L1 (Lasso)  — weights exactly zero : {lasso_zeros:2d}/20")
    print(f"    L2 (Ridge)  — weights exactly zero : {ridge_zeros:2d}/20")
    print(f"    L2 max |w|  : {np.max(np.abs(ridge.coef_)):.3f}   "
          f"L1 max |w| : {np.max(np.abs(lasso.coef_)):.3f}")
    print()

print("  KEY INSIGHT:")
print("  L1 achieves feature selection by zeroing out irrelevant weights.")
print("  L2 shrinks all weights proportionally but keeps all features.")
print("  As λ increases, BOTH methods reduce model complexity — but")
print("  L1 does it by elimination while L2 does it by uniform shrinkage.")
print()
print("  Plot saved → l1_vs_l2_paths.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · k-Fold Cross-Validation From Scratch": {
        "description": (
            "Implement k-fold cross-validation manually to demystify it, "
            "then compare its variance against a single train/val split."
        ),
        "language": "python",
        "code": '''
import numpy as np
import copy as _copy
import scipy.optimize as _sp_opt
from scipy.stats import loguniform  # kept from scipy (not affected by sklearn issue)

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── StandardScaler ────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── make_regression ───────────────────────────────────────────────────────────
def make_regression(n_samples=100,n_features=100,n_informative=10,noise=0.0,
                    coef=False,random_state=None):
    rng=np.random.default_rng(random_state)
    ground_truth=np.zeros(n_features)
    idx=rng.choice(n_features,n_informative,replace=False)
    ground_truth[idx]=rng.uniform(10,100,n_informative)*rng.choice([-1,1],n_informative)
    X=rng.standard_normal((n_samples,n_features))
    y=X@ground_truth+rng.normal(0,noise,n_samples)
    return (X,y,ground_truth) if coef else (X,y)

# ── make_classification ───────────────────────────────────────────────────────
def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── make_moons ────────────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

# ── load_breast_cancer (synthetic substitute) ─────────────────────────────────
def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf))
    X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

# ── train_test_split ──────────────────────────────────────────────────────────
def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

# ── StratifiedKFold ───────────────────────────────────────────────────────────
class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,class_weight=None,solver="lbfgs",
                 penalty="l2",random_state=None):
        self.C=C; self.max_iter=max_iter; self.class_weight=class_weight
        self.penalty=penalty
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y):
        n,d=X.shape; sw=self._sw(y); lam=1./self.C; w0=np.zeros(d+1)
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso (coordinate descent) ────────────────────────────────────────────────
class Lasso:
    def __init__(self,alpha=1.0,max_iter=1000): self.alpha=alpha; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=y-X@w+X[:,j]*w[j]; rho=X[:,j]@r; z=np.sum(X[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w; self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Ridge ─────────────────────────────────────────────────────────────────────
class Ridge:
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        n,d=X.shape
        self.coef_=np.linalg.solve(X.T@X+self.alpha*np.eye(d),X.T@y)
        self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Metrics ───────────────────────────────────────────────────────────────────
def confusion_matrix(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return np.array([[tn,fp],[fn,tp]])

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

def precision_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)

def recall_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fn=((yp==0)&(yt==1)).sum()
    return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)

def f1_score(yt,yp,zero_division=0):
    p=precision_score(yt,yp,zero_division); r=recall_score(yt,yp,zero_division)
    return float(2*p*r/(p+r)) if (p+r)>0 else float(zero_division)

def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))

def roc_curve(yt,ys):
    thrs=np.concatenate([[1.0+1e-9],np.sort(np.unique(ys))[::-1]])
    tprs,fprs=[0.],[0.]
    pos=yt.sum(); neg=len(yt)-pos
    for t in thrs[1:]:
        yp=(ys>=t).astype(int)
        tprs.append(((yp==1)&(yt==1)).sum()/pos if pos>0 else 0)
        fprs.append(((yp==1)&(yt==0)).sum()/neg if neg>0 else 0)
    return np.array(fprs),np.array(tprs),thrs

def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── GridSearchCV / RandomizedSearchCV ─────────────────────────────────────────
class GridSearchCV:
    def __init__(self,estimator,param_grid,cv=5,scoring="accuracy",
                 n_jobs=None,verbose=0):
        self.estimator=estimator; self.param_grid=param_grid; self.cv=cv
        self.scoring=scoring
    def _configs(self):
        keys=list(self.param_grid.keys()); vals=list(self.param_grid.values())
        import itertools
        for combo in itertools.product(*vals):
            yield dict(zip(keys,combo))
    def fit(self,X,y):
        configs=list(self._configs())
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for cfg in configs:
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

class RandomizedSearchCV:
    def __init__(self,estimator,param_distributions,n_iter=10,cv=5,
                 scoring="accuracy",n_jobs=None,random_state=None,verbose=0):
        self.estimator=estimator; self.param_distributions=param_distributions
        self.n_iter=n_iter; self.cv=cv; self.scoring=scoring
        self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for _ in range(self.n_iter):
            cfg={}
            for k,dist in self.param_distributions.items():
                if hasattr(dist,"rvs"): cfg[k]=float(dist.rvs(random_state=int(rng.integers(0,10**9))))
                else: cfg[k]=dist[rng.integers(0,len(dist))]
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

np.random.seed(0)

# ── Data ──────────────────────────────────────────────────────────────────
data   = load_breast_cancer()
X, y   = data.data, data.target
scaler = StandardScaler()
X      = scaler.fit_transform(X)

N = len(y)
K = 10                   # number of folds

print("=" * 65)
print("  k-FOLD CROSS-VALIDATION DEMO (Logistic Regression)")
print("=" * 65)
print(f"  Dataset      : Breast Cancer Wisconsin")
print(f"  Samples      : {N}  |  Features : {X.shape[1]}")
print(f"  Class balance: {y.mean()*100:.1f}% positive")
print(f"  k            : {K}")
print()

# ── Manual k-Fold implementation ──────────────────────────────────────────
def manual_kfold(X, y, k):
    """Pure-numpy stratified-ish k-fold cross-validation."""
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    scores = []
    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx],   y[val_idx]
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_tr, y_tr)
        scores.append(model.score(X_vl, y_vl))
    return np.array(scores)

cv_scores = manual_kfold(X, y, K)

print(f"  Per-Fold Accuracy Scores:")
print(f"  {'Fold':>5} | {'Score':>8}")
print(f"  ──────────────────")
for i, s in enumerate(cv_scores, 1):
    bar = "█" * int(s * 30)
    print(f"  {i:5d} | {s:.4f}  {bar}")

print(f"  ──────────────────")
print(f"  Mean  : {cv_scores.mean():.4f}")
print(f"  Std   : {cv_scores.std():.4f}")
print(f"  95% CI: ({cv_scores.mean() - 2*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 2*cv_scores.std():.4f})")
print()

# ── Compare with single split (repeated many times to show variance) ──────
print("  SINGLE SPLIT VARIANCE ANALYSIS (100 random splits):")
single_scores = []
for seed in range(100):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    cut = int(0.8 * N)
    tr_i, vl_i = idx[:cut], idx[cut:]
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X[tr_i], y[tr_i])
    single_scores.append(model.score(X[vl_i], y[vl_i]))

single_scores = np.array(single_scores)
print(f"  Single-split Mean : {single_scores.mean():.4f}")
print(f"  Single-split Std  : {single_scores.std():.4f}   ← much higher!")
print(f"  Single-split range: {single_scores.min():.4f} – {single_scores.max():.4f}")
print()
print(f"  k-Fold Std        : {cv_scores.std():.4f}")
print(f"  Variance reduction: {single_scores.std() / cv_scores.std():.1f}× more "
      f"stable with {K}-fold CV")
print()
print("  WHY THIS MATTERS:")
print("  A single 80/20 split gives you ONE noisy estimate.")
print("  k-fold gives you k estimates and averages them →")
print("  much more reliable, with a principled uncertainty band.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Confusion Matrix & Classification Metrics Deep Dive": {
        "description": (
            "Build a complete classification evaluation: confusion matrix, "
            "accuracy, precision, recall, F1, and ROC-AUC. "
            "Then show why accuracy fails on imbalanced data and F1/AUC doesn't."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _sp_opt
from scipy.stats import loguniform  # kept from scipy (not affected by sklearn issue)

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── StandardScaler ────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── make_regression ───────────────────────────────────────────────────────────
def make_regression(n_samples=100,n_features=100,n_informative=10,noise=0.0,
                    coef=False,random_state=None):
    rng=np.random.default_rng(random_state)
    ground_truth=np.zeros(n_features)
    idx=rng.choice(n_features,n_informative,replace=False)
    ground_truth[idx]=rng.uniform(10,100,n_informative)*rng.choice([-1,1],n_informative)
    X=rng.standard_normal((n_samples,n_features))
    y=X@ground_truth+rng.normal(0,noise,n_samples)
    return (X,y,ground_truth) if coef else (X,y)

# ── make_classification ───────────────────────────────────────────────────────
def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── make_moons ────────────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

# ── load_breast_cancer (synthetic substitute) ─────────────────────────────────
def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf))
    X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

# ── train_test_split ──────────────────────────────────────────────────────────
def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

# ── StratifiedKFold ───────────────────────────────────────────────────────────
class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,class_weight=None,solver="lbfgs",
                 penalty="l2",random_state=None):
        self.C=C; self.max_iter=max_iter; self.class_weight=class_weight
        self.penalty=penalty
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y):
        n,d=X.shape; sw=self._sw(y); lam=1./self.C; w0=np.zeros(d+1)
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso (coordinate descent) ────────────────────────────────────────────────
class Lasso:
    def __init__(self,alpha=1.0,max_iter=1000): self.alpha=alpha; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=y-X@w+X[:,j]*w[j]; rho=X[:,j]@r; z=np.sum(X[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w; self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Ridge ─────────────────────────────────────────────────────────────────────
class Ridge:
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        n,d=X.shape
        self.coef_=np.linalg.solve(X.T@X+self.alpha*np.eye(d),X.T@y)
        self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Metrics ───────────────────────────────────────────────────────────────────
def confusion_matrix(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return np.array([[tn,fp],[fn,tp]])

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

def precision_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)

def recall_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fn=((yp==0)&(yt==1)).sum()
    return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)

def f1_score(yt,yp,zero_division=0):
    p=precision_score(yt,yp,zero_division); r=recall_score(yt,yp,zero_division)
    return float(2*p*r/(p+r)) if (p+r)>0 else float(zero_division)

def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))

def roc_curve(yt,ys):
    thrs=np.concatenate([[1.0+1e-9],np.sort(np.unique(ys))[::-1]])
    tprs,fprs=[0.],[0.]
    pos=yt.sum(); neg=len(yt)-pos
    for t in thrs[1:]:
        yp=(ys>=t).astype(int)
        tprs.append(((yp==1)&(yt==1)).sum()/pos if pos>0 else 0)
        fprs.append(((yp==1)&(yt==0)).sum()/neg if neg>0 else 0)
    return np.array(fprs),np.array(tprs),thrs

def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── GridSearchCV / RandomizedSearchCV ─────────────────────────────────────────
class GridSearchCV:
    def __init__(self,estimator,param_grid,cv=5,scoring="accuracy",
                 n_jobs=None,verbose=0):
        self.estimator=estimator; self.param_grid=param_grid; self.cv=cv
        self.scoring=scoring
    def _configs(self):
        keys=list(self.param_grid.keys()); vals=list(self.param_grid.values())
        import itertools
        for combo in itertools.product(*vals):
            yield dict(zip(keys,combo))
    def fit(self,X,y):
        configs=list(self._configs())
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for cfg in configs:
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

class RandomizedSearchCV:
    def __init__(self,estimator,param_distributions,n_iter=10,cv=5,
                 scoring="accuracy",n_jobs=None,random_state=None,verbose=0):
        self.estimator=estimator; self.param_distributions=param_distributions
        self.n_iter=n_iter; self.cv=cv; self.scoring=scoring
        self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for _ in range(self.n_iter):
            cfg={}
            for k,dist in self.param_distributions.items():
                if hasattr(dist,"rvs"): cfg[k]=float(dist.rvs(random_state=int(rng.integers(0,10**9))))
                else: cfg[k]=dist[rng.integers(0,len(dist))]
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

np.random.seed(7)

# ── Helper ────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    TP, FP = cm[1,1], cm[0,1]
    FN, TN = cm[1,0], cm[0,0]
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"  Confusion Matrix:")
    print(f"                 Pred POS   Pred NEG")
    print(f"    Actual POS:   {TP:5d}      {FN:5d}   (TP, FN)")
    print(f"    Actual NEG:   {FP:5d}      {TN:5d}   (FP, TN)")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}  "
          f"(of predicted pos, how many correct?)")
    print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}  "
          f"(of all actual pos, how many caught?)")
    print(f"  F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")
    print()

# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  SCENARIO 1 — BALANCED DATASET  (50% / 50%)")
print("=" * 65)

X_bal, y_bal = make_classification(
    n_samples=2000, n_features=10, weights=[0.5, 0.5], random_state=42)
X_bal = StandardScaler().fit_transform(X_bal)
X_tr, X_te, y_tr, y_te = train_test_split(X_bal, y_bal, test_size=0.3, random_state=1)

model_bal = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
y_pred_bal = model_bal.predict(X_te)
y_prob_bal = model_bal.predict_proba(X_te)[:, 1]
evaluate("Logistic Regression — Balanced", y_te, y_pred_bal, y_prob_bal)

# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  SCENARIO 2 — IMBALANCED DATASET  (95% Negative / 5% Positive)")
print("=" * 65)

X_imb, y_imb = make_classification(
    n_samples=2000, n_features=10, weights=[0.95, 0.05], flip_y=0.02,
    random_state=42)
X_imb = StandardScaler().fit_transform(X_imb)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_imb, y_imb, test_size=0.3, stratify=y_imb, random_state=1)

# Model A: Logistic regression (tries to learn)
model_imb = LogisticRegression(max_iter=500, class_weight="balanced").fit(
    X_tr2, y_tr2)
y_pred_imb  = model_imb.predict(X_te2)
y_prob_imb  = model_imb.predict_proba(X_te2)[:, 1]

# Model B: Naive "predict everything negative" baseline
y_pred_naive = np.zeros_like(y_te2)
y_prob_naive = np.zeros_like(y_te2, dtype=float)

evaluate("Logistic Regression — Imbalanced", y_te2, y_pred_imb, y_prob_imb)
evaluate("NAIVE baseline (predict all negative)", y_te2, y_pred_naive,
         y_prob_naive + 1e-9)

print("  INSIGHT:")
print("  The naive baseline achieves ~95% ACCURACY by doing NOTHING.")
print("  But its Recall = 0.0 → it catches zero actual positives.")
print("  F1 and ROC-AUC correctly expose this as a useless model.")
print("  → Never use accuracy alone on imbalanced datasets!")
print()

# ── ROC curve plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ROC Curves: Balanced vs Imbalanced Dataset", fontsize=13,
             fontweight="bold")

for ax, (y_t, y_p, y_prob, title) in zip(axes, [
        (y_te, y_pred_bal, y_prob_bal, "Balanced (50/50)"),
        (y_te2, y_pred_imb, y_prob_imb, "Imbalanced (95/5)")]):
    fpr, tpr, _ = roc_curve(y_t, y_prob)
    auc = roc_auc_score(y_t, y_prob)
    ax.plot(fpr, tpr, "steelblue", lw=2, label=f"Model (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "gray", linestyle="--", label="Random (AUC=0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("roc_curves.png", dpi=120)
print("  ROC curve plot saved → roc_curves.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Dropout Regularisation in a Neural Network": {
        "description": (
            "Train a simple neural network with and without dropout. "
            "Compare the train/validation gap to see dropout's anti-overfitting effect."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Tiny neural network from scratch (NumPy only) ─────────────────────────

def relu(x):         return np.maximum(0, x)
def relu_grad(x):    return (x > 0).astype(float)
def sigmoid(x):      return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def bce_loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

class TinyNet:
    """2-layer fully-connected network with optional dropout."""

    def __init__(self, n_in, n_h, dropout_rate=0.0, lr=0.01):
        self.W1 = np.random.randn(n_in, n_h) * np.sqrt(2 / n_in)
        self.b1 = np.zeros(n_h)
        self.W2 = np.random.randn(n_h, 1) * np.sqrt(2 / n_h)
        self.b2 = np.zeros(1)
        self.dr = dropout_rate
        self.lr = lr

    def forward(self, X, training=True):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        if training and self.dr > 0:
            self.mask = (np.random.rand(*self.a1.shape) > self.dr) / (1 - self.dr)
            self.a1 = self.a1 * self.mask
        else:
            self.mask = np.ones_like(self.a1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return sigmoid(self.z2).ravel()

    def backward(self, X, y, preds):
        m   = len(y)
        d2  = (preds - y).reshape(-1, 1) / m
        dW2 = self.a1.T @ d2
        db2 = d2.sum(axis=0)
        d1  = (d2 @ self.W2.T) * relu_grad(self.z1) * self.mask
        dW1 = X.T @ d1
        db1 = d1.sum(axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

# ── Make deliberately small dataset so overfitting is visible ─────────────
import copy as _copy
import scipy.optimize as _sp_opt
from scipy.stats import loguniform  # kept from scipy (not affected by sklearn issue)

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── StandardScaler ────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── make_regression ───────────────────────────────────────────────────────────
def make_regression(n_samples=100,n_features=100,n_informative=10,noise=0.0,
                    coef=False,random_state=None):
    rng=np.random.default_rng(random_state)
    ground_truth=np.zeros(n_features)
    idx=rng.choice(n_features,n_informative,replace=False)
    ground_truth[idx]=rng.uniform(10,100,n_informative)*rng.choice([-1,1],n_informative)
    X=rng.standard_normal((n_samples,n_features))
    y=X@ground_truth+rng.normal(0,noise,n_samples)
    return (X,y,ground_truth) if coef else (X,y)

# ── make_classification ───────────────────────────────────────────────────────
def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── make_moons ────────────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

# ── load_breast_cancer (synthetic substitute) ─────────────────────────────────
def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf))
    X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

# ── train_test_split ──────────────────────────────────────────────────────────
def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

# ── StratifiedKFold ───────────────────────────────────────────────────────────
class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,class_weight=None,solver="lbfgs",
                 penalty="l2",random_state=None):
        self.C=C; self.max_iter=max_iter; self.class_weight=class_weight
        self.penalty=penalty
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y):
        n,d=X.shape; sw=self._sw(y); lam=1./self.C; w0=np.zeros(d+1)
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso (coordinate descent) ────────────────────────────────────────────────
class Lasso:
    def __init__(self,alpha=1.0,max_iter=1000): self.alpha=alpha; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=y-X@w+X[:,j]*w[j]; rho=X[:,j]@r; z=np.sum(X[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w; self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Ridge ─────────────────────────────────────────────────────────────────────
class Ridge:
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        n,d=X.shape
        self.coef_=np.linalg.solve(X.T@X+self.alpha*np.eye(d),X.T@y)
        self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Metrics ───────────────────────────────────────────────────────────────────
def confusion_matrix(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return np.array([[tn,fp],[fn,tp]])

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

def precision_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)

def recall_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fn=((yp==0)&(yt==1)).sum()
    return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)

def f1_score(yt,yp,zero_division=0):
    p=precision_score(yt,yp,zero_division); r=recall_score(yt,yp,zero_division)
    return float(2*p*r/(p+r)) if (p+r)>0 else float(zero_division)

def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))

def roc_curve(yt,ys):
    thrs=np.concatenate([[1.0+1e-9],np.sort(np.unique(ys))[::-1]])
    tprs,fprs=[0.],[0.]
    pos=yt.sum(); neg=len(yt)-pos
    for t in thrs[1:]:
        yp=(ys>=t).astype(int)
        tprs.append(((yp==1)&(yt==1)).sum()/pos if pos>0 else 0)
        fprs.append(((yp==1)&(yt==0)).sum()/neg if neg>0 else 0)
    return np.array(fprs),np.array(tprs),thrs

def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── GridSearchCV / RandomizedSearchCV ─────────────────────────────────────────
class GridSearchCV:
    def __init__(self,estimator,param_grid,cv=5,scoring="accuracy",
                 n_jobs=None,verbose=0):
        self.estimator=estimator; self.param_grid=param_grid; self.cv=cv
        self.scoring=scoring
    def _configs(self):
        keys=list(self.param_grid.keys()); vals=list(self.param_grid.values())
        import itertools
        for combo in itertools.product(*vals):
            yield dict(zip(keys,combo))
    def fit(self,X,y):
        configs=list(self._configs())
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for cfg in configs:
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

class RandomizedSearchCV:
    def __init__(self,estimator,param_distributions,n_iter=10,cv=5,
                 scoring="accuracy",n_jobs=None,random_state=None,verbose=0):
        self.estimator=estimator; self.param_distributions=param_distributions
        self.n_iter=n_iter; self.cv=cv; self.scoring=scoring
        self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for _ in range(self.n_iter):
            cfg={}
            for k,dist in self.param_distributions.items():
                if hasattr(dist,"rvs"): cfg[k]=float(dist.rvs(random_state=int(rng.integers(0,10**9))))
                else: cfg[k]=dist[rng.integers(0,len(dist))]
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

X_all, y_all = make_moons(n_samples=300, noise=0.25, random_state=5)
X_all = StandardScaler().fit_transform(X_all)
X_tr, X_vl, y_tr, y_vl = train_test_split(X_all, y_all, test_size=0.4,
                                            random_state=1)

# ── Train two networks: no dropout vs dropout=0.5 ────────────────────────
N_EPOCHS  = 500
BATCH     = 32
N_HIDDEN  = 128          # large network → lots of capacity to overfit

nets = {
    "No Dropout    (overfit-prone)": TinyNet(2, N_HIDDEN, dropout_rate=0.0, lr=0.02),
    "Dropout p=0.5 (regularised)":  TinyNet(2, N_HIDDEN, dropout_rate=0.5, lr=0.02),
}

print("=" * 65)
print("  DROPOUT REGULARISATION — TRAIN vs VALIDATION LOSS")
print("=" * 65)
print(f"  Network: 2 → {N_HIDDEN} → 1   (sigmoid output, ReLU hidden)")
print(f"  Dataset: make_moons  |  Train={len(y_tr)}  Val={len(y_vl)}")
print()

history = {name: {"tr": [], "vl": []} for name in nets}

for name, net in nets.items():
    for ep in range(N_EPOCHS):
        idx = np.random.permutation(len(y_tr))
        for start in range(0, len(y_tr), BATCH):
            batch = idx[start:start + BATCH]
            Xb, yb = X_tr[batch], y_tr[batch]
            preds = net.forward(Xb, training=True)
            net.backward(Xb, yb, preds)

        tr_preds  = net.forward(X_tr, training=False)
        vl_preds  = net.forward(X_vl, training=False)
        history[name]["tr"].append(bce_loss(y_tr, tr_preds))
        history[name]["vl"].append(bce_loss(y_vl, vl_preds))

# ── Print final numbers ───────────────────────────────────────────────────
for name, hist in history.items():
    final_tr = hist["tr"][-1]
    final_vl = hist["vl"][-1]
    gap      = final_vl - final_tr
    print(f"  {name}")
    print(f"    Train loss : {final_tr:.4f}")
    print(f"    Val   loss : {final_vl:.4f}")
    print(f"    Gap        : {gap:+.4f}")
    print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Dropout Regularisation Effect on Overfitting", fontsize=13,
             fontweight="bold")

colours = {"tr": "steelblue", "vl": "tomato"}
ep_arr  = np.arange(1, N_EPOCHS + 1)

for ax, (name, hist) in zip(axes, history.items()):
    ax.plot(ep_arr, hist["tr"], colour := "steelblue", lw=2, label="Train loss")
    ax.plot(ep_arr, hist["vl"], "tomato",              lw=2, label="Val loss")
    gap = hist["vl"][-1] - hist["tr"][-1]
    ax.set_title(f"{name}\\nFinal gap = {gap:+.4f}", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.9)

plt.tight_layout()
plt.savefig("dropout_comparison.png", dpi=120)
print("  Plot saved → dropout_comparison.png")
print()
print("  WHAT TO OBSERVE IN THE PLOT:")
print("  - No Dropout : train loss dives to near-zero, val loss diverges")
print("    → large gap = overfitting")
print("  - Dropout 0.5: both curves converge closer together")
print("    → dropout acts as an ensemble of many smaller networks")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Hyperparameter Tuning: Grid vs Random Search": {
        "description": (
            "Compare grid search and random search for tuning a regularised "
            "logistic regression. Show that random search finds equally good or "
            "better results using fewer evaluations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import copy as _copy
import scipy.optimize as _sp_opt
from scipy.stats import loguniform  # kept from scipy (not affected by sklearn issue)

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── StandardScaler ────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── make_regression ───────────────────────────────────────────────────────────
def make_regression(n_samples=100,n_features=100,n_informative=10,noise=0.0,
                    coef=False,random_state=None):
    rng=np.random.default_rng(random_state)
    ground_truth=np.zeros(n_features)
    idx=rng.choice(n_features,n_informative,replace=False)
    ground_truth[idx]=rng.uniform(10,100,n_informative)*rng.choice([-1,1],n_informative)
    X=rng.standard_normal((n_samples,n_features))
    y=X@ground_truth+rng.normal(0,noise,n_samples)
    return (X,y,ground_truth) if coef else (X,y)

# ── make_classification ───────────────────────────────────────────────────────
def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── make_moons ────────────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

# ── load_breast_cancer (synthetic substitute) ─────────────────────────────────
def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf))
    X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

# ── train_test_split ──────────────────────────────────────────────────────────
def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

# ── StratifiedKFold ───────────────────────────────────────────────────────────
class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,class_weight=None,solver="lbfgs",
                 penalty="l2",random_state=None):
        self.C=C; self.max_iter=max_iter; self.class_weight=class_weight
        self.penalty=penalty
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y):
        n,d=X.shape; sw=self._sw(y); lam=1./self.C; w0=np.zeros(d+1)
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso (coordinate descent) ────────────────────────────────────────────────
class Lasso:
    def __init__(self,alpha=1.0,max_iter=1000): self.alpha=alpha; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=y-X@w+X[:,j]*w[j]; rho=X[:,j]@r; z=np.sum(X[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w; self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Ridge ─────────────────────────────────────────────────────────────────────
class Ridge:
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        n,d=X.shape
        self.coef_=np.linalg.solve(X.T@X+self.alpha*np.eye(d),X.T@y)
        self.intercept_=0.; return self
    def predict(self,X): return X@self.coef_+self.intercept_

# ── Metrics ───────────────────────────────────────────────────────────────────
def confusion_matrix(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return np.array([[tn,fp],[fn,tp]])

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

def precision_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)

def recall_score(yt,yp,zero_division=0):
    tp=((yp==1)&(yt==1)).sum(); fn=((yp==0)&(yt==1)).sum()
    return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)

def f1_score(yt,yp,zero_division=0):
    p=precision_score(yt,yp,zero_division); r=recall_score(yt,yp,zero_division)
    return float(2*p*r/(p+r)) if (p+r)>0 else float(zero_division)

def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))

def roc_curve(yt,ys):
    thrs=np.concatenate([[1.0+1e-9],np.sort(np.unique(ys))[::-1]])
    tprs,fprs=[0.],[0.]
    pos=yt.sum(); neg=len(yt)-pos
    for t in thrs[1:]:
        yp=(ys>=t).astype(int)
        tprs.append(((yp==1)&(yt==1)).sum()/pos if pos>0 else 0)
        fprs.append(((yp==1)&(yt==0)).sum()/neg if neg>0 else 0)
    return np.array(fprs),np.array(tprs),thrs

def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── GridSearchCV / RandomizedSearchCV ─────────────────────────────────────────
class GridSearchCV:
    def __init__(self,estimator,param_grid,cv=5,scoring="accuracy",
                 n_jobs=None,verbose=0):
        self.estimator=estimator; self.param_grid=param_grid; self.cv=cv
        self.scoring=scoring
    def _configs(self):
        keys=list(self.param_grid.keys()); vals=list(self.param_grid.values())
        import itertools
        for combo in itertools.product(*vals):
            yield dict(zip(keys,combo))
    def fit(self,X,y):
        configs=list(self._configs())
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for cfg in configs:
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self

class RandomizedSearchCV:
    def __init__(self,estimator,param_distributions,n_iter=10,cv=5,
                 scoring="accuracy",n_jobs=None,random_state=None,verbose=0):
        self.estimator=estimator; self.param_distributions=param_distributions
        self.n_iter=n_iter; self.cv=cv; self.scoring=scoring
        self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        if hasattr(self.cv,"split"): splits=list(self.cv.split(X,y))
        else:
            skf=StratifiedKFold(n_splits=self.cv,shuffle=True,random_state=0)
            splits=list(skf.split(X,y))
        scores=[]; params_list=[]
        for _ in range(self.n_iter):
            cfg={}
            for k,dist in self.param_distributions.items():
                if hasattr(dist,"rvs"): cfg[k]=float(dist.rvs(random_state=int(rng.integers(0,10**9))))
                else: cfg[k]=dist[rng.integers(0,len(dist))]
            fold_scores=[]
            for tr,te in splits:
                est=_copy.deepcopy(self.estimator)
                for k,v in cfg.items(): setattr(est,k,v)
                try: est.fit(X[tr],y[tr]); fold_scores.append(est.score(X[te],y[te]))
                except: fold_scores.append(0.)
            scores.append(np.mean(fold_scores)); params_list.append(cfg)
        best=np.argmax(scores)
        self.best_score_=scores[best]; self.best_params_=params_list[best]
        self.cv_results_={"mean_test_score":np.array(scores),"params":params_list}
        est=_copy.deepcopy(self.estimator)
        for k,v in self.best_params_.items(): setattr(est,k,v)
        self.best_estimator_=est.fit(X,y)
        return self
from scipy.stats import loguniform

np.random.seed(42)

# ── Dataset ───────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=1500, n_features=20, n_informative=10,
                            random_state=42)
X = StandardScaler().fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
model = LogisticRegression(max_iter=1000, solver="saga")

# ── Grid Search ───────────────────────────────────────────────────────────
grid_params = {
    "C":         [0.001, 0.01, 0.1, 1, 10, 100],          # 6 values
    "penalty":   ["l1", "l2"],                              # 2 values
}
# Total: 6 × 2 × 5-fold = 60 model fits

t0 = time.time()
grid_search = GridSearchCV(model, grid_params, cv=cv, scoring="accuracy",
                            n_jobs=-1, verbose=0)
grid_search.fit(X, y)
grid_time = time.time() - t0

# ── Random Search ─────────────────────────────────────────────────────────
# Same number of evaluations as grid (12 configs × 5 folds = 60 fits)
rand_params = {
    "C":       loguniform(1e-3, 1e2),     # continuous log-uniform distribution
    "penalty": ["l1", "l2"],
}

t0 = time.time()
rand_search = RandomizedSearchCV(model, rand_params, n_iter=12, cv=cv,
                                  scoring="accuracy", n_jobs=-1,
                                  random_state=7, verbose=0)
rand_search.fit(X, y)
rand_time = time.time() - t0

# ── Results ───────────────────────────────────────────────────────────────
print("=" * 65)
print("  HYPERPARAMETER TUNING: GRID SEARCH vs RANDOM SEARCH")
print("=" * 65)
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Model: Logistic Regression  |  CV: 5-fold")
print()

print(f"  GRID SEARCH")
print(f"    Parameter grid : C ∈ {{6 values}} × penalty ∈ {{l1,l2}}")
print(f"    Total configs  : {len(grid_search.cv_results_['mean_test_score'])}")
print(f"    Best params    : {grid_search.best_params_}")
print(f"    Best CV acc    : {grid_search.best_score_:.4f}")
print(f"    Time           : {grid_time:.2f}s")
print()

print(f"  RANDOM SEARCH")
print(f"    C sampled from : log-uniform(0.001, 100)")
print(f"    Total configs  : 12  (same compute as grid)")
print(f"    Best params    : {rand_search.best_params_}")
print(f"    Best params C  : {rand_search.best_params_['C']:.5f}")
print(f"    Best CV acc    : {rand_search.best_score_:.4f}")
print(f"    Time           : {rand_time:.2f}s")
print()

delta = rand_search.best_score_ - grid_search.best_score_
print(f"  COMPARISON:")
print(f"    Score difference : {delta:+.4f}")
print(f"    Winner           : {'Random Search' if delta >= 0 else 'Grid Search'}")
print()
print("  KEY LESSON:")
print("  Grid search fixes values at discrete points and wastes budget")
print("  on coarse coverage. Random search samples the continuous space,")
print("  finding better C values between the grid's fixed points.")
print("  For expensive models, Bayesian optimisation (Optuna, Hyperopt)")
print("  does even better by learning which regions to search next.")
print()

# ── Show top-5 configs for each ───────────────────────────────────────────
print("  TOP-5 Grid Search configs (sorted by CV accuracy):")
grid_results = sorted(
    zip(grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["params"]),
    key=lambda x: x[0], reverse=True
)[:5]
for rank, (score, params) in enumerate(grid_results, 1):
    print(f"    #{rank}: acc={score:.4f}  {params}")

print()
print("  TOP-5 Random Search configs:")
rand_results = sorted(
    zip(rand_search.cv_results_["mean_test_score"],
        rand_search.cv_results_["params"]),
    key=lambda x: x[0], reverse=True
)[:5]
for rank, (score, params) in enumerate(rand_results, 1):
    c_val = f"{params['C']:.5f}"
    print(f"    #{rank}: acc={score:.4f}  C={c_val}  penalty={params['penalty']}")
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


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
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }