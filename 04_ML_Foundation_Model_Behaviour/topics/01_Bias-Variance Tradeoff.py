"""
Bias-Variance Tradeoff
======================

The single most important conceptual framework in supervised machine learning.
Every modelling decision — complexity, regularisation, ensembling, early stopping —
is ultimately a negotiation between these two sources of generalisation error.

"""

import textwrap
import re

TOPIC_NAME   = "Bias-Variance Tradeoff"
DISPLAY_NAME = "01 · Bias-Variance Tradeoff"
ICON         = "⚖️"
SUBTITLE     = "The Fundamental Decomposition of Generalisation Error"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """


### PART 1 — THE PROBLEM: WHY DOES A MODEL FAIL ON NEW DATA?

### The Generalisation Gap

The goal of supervised learning is not to fit training data — it is to
predict accurately on unseen data drawn from the same distribution.
The gap between training performance and test performance is the
generalisation gap, and it has exactly three causes:

    Total expected test error  =  Bias²  +  Variance  +  Irreducible Noise

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Source             │  Meaning                     │  Fixable?      │
    ├─────────────────────┼──────────────────────────────┼────────────────┤
    │  Bias²              │  Systematic wrong assumptions│  Yes           │
    │  Variance           │  Sensitivity to training set │  Yes           │
    │  Irreducible noise  │  Inherent randomness in y    │  Never         │
    └─────────────────────┴──────────────────────────────┴────────────────┘

This decomposition is exact, not approximate. It tells us that even a
perfectly trained model cannot beat the noise floor — and that every
reduction in bias comes at a price paid in variance, and vice versa.


──────────────────────────────────────────────────────────────────────────────
### The Thought Experiment Behind the Decomposition

Imagine the true data-generating process:
    y = f(x) + ε,   where  ε ~ N(0, σ²_ε)   (irreducible noise)

We train a model f̂ on a dataset D of n examples drawn from this process.
Because D is finite and random, f̂ is itself a random variable — train it on a
different draw of D and you get a different f̂.

Now ask: across all possible training sets of size n, what is the expected
squared prediction error at a fixed test point x?

    E_D[(y − f̂(x))²]
    = [E_D[f̂(x)] − f(x)]²   +   E_D[(f̂(x) − E_D[f̂(x)])²]   +   σ²_ε
    =         Bias²(x)        +             Variance(x)          +  Noise

The expectation is over both the randomness of ε AND the randomness of
the training set D. This is the exact bias-variance decomposition.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — BIAS AND VARIANCE DEFINED PRECISELY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Bias

    Bias(x) = E_D[f̂(x)] − f(x)

Bias is the gap between the average prediction of your model (averaged
over all possible training sets of size n) and the true function.

A biased model is wrong on average — even with infinite data it will not
converge to the right answer, because its hypothesis class is too restricted
to represent the true function.

    High-bias scenario (underfitting):
    True function:  f(x) = sin(x)  — a smooth curve
    Model class:    f̂(x) = ax + b  — linear models only

    No matter how much data you give a linear model, it can never
    fit a sinusoid. The best line through a sine wave is permanently
    wrong. That permanent wrongness is bias.

    Low-bias scenario:
    Model class:  high-degree polynomials, deep neural networks
    These are expressive enough to approximate virtually any function,
    so their average prediction can sit very close to f(x).


──────────────────────────────────────────────────────────────────────────────
### Variance

    Variance(x) = E_D[(f̂(x) − E_D[f̂(x)])²]

Variance is the expected squared deviation of a single trained model from
the average model across all training sets. It measures how much the
model's prediction at x fluctuates as you retrain on different data.

A high-variance model is hypersensitive to the particular training set:
swap a handful of data points and the predictions change dramatically.

    High-variance scenario (overfitting):
    Model class:   degree-15 polynomial fitted to n=20 noisy points
    Training on D₁:  smooth fit through the data
    Training on D₂:  completely different wild oscillations
    The model memorises whichever noise happens to be in the training set.

    Low-variance scenario:
    Model class:   linear model, heavily regularised neural network
    Changing a few training points barely nudges the fitted line.
    The model is stable — but may be systematically wrong (biased).


──────────────────────────────────────────────────────────────────────────────
### The Shooting Analogy

Imagine firing arrows at a target where the bullseye is the true f(x):

    Low Bias / Low Variance      Low Bias / High Variance
    ┌────────────────────┐       ┌─────────────────────┐
    │        ╬           │       │  ×     ×            │
    │       ╬╬╬          │       │     ×    ×          │
    │        ╬           │       │  ×   ×              │
    │ (tight cluster     │       │ (spread all around  │
    │  at bullseye)      │       │  the bullseye)      │
    └────────────────────┘       └─────────────────────┘
    Ideal model                  High-capacity unfettered model

    High Bias / Low Variance     High Bias / High Variance
    ┌────────────────────┐       ┌─────────────────────┐
    │                    │       │ ×                   │
    │      ╬╬╬           │       │      ×    ×         │
    │       ╬            │       │  ×           ×      │
    │ (tight cluster but │       │ (spread AND         │
    │  off to the side)  │       │  off-target)        │
    └────────────────────┘       └─────────────────────┘
    Underfit model               Worst of both worlds

    Each arrow = f̂ trained on one particular dataset D.
    Bullseye    = the true function f(x) we are trying to learn.



### PART 3 — THE TRADEOFF: WHY YOU CANNOT MINIMISE BOTH

### Model Complexity as the Dial

Think of model complexity as a single dial from "simple" to "complex":

    COMPLEXITY DIAL ────────────────────────────────────────────►
                   Simple                              Complex
                   ●────────────────────────────────────────●
                   |                                        |
            Linear regression                    Degree-15 polynomial
            Depth-1 decision tree                Deep neural network
            k-NN with k=n                        k-NN with k=1

As you turn the dial from simple to complex:

    Bias²  decreases monotonically  — a more complex model can represent
           more functions, so it can get closer to f(x) on average.

    Variance increases monotonically — a more complex model has more
           parameters to fit, so it is more sensitive to training noise.

    Diagram — the classic U-shaped test error curve:

    Error
      ↑
      │  ╲                                 ╱   Total error
      │   ╲                               ╱
      │    ╲             ╭───────────────╯    ← Variance
      │     ╲           ╱
      │      ╲─────────╱                      ← Bias²
      │       ╲       ╱
      │        ╲     ╱                        ← Irreducible noise (floor)
      │         ╲   ╱
      │          ╲ ╱
      │           ●  ← Sweet spot (minimum total error)
      └───────────────────────────────────── Complexity

    Left of sweet spot  → underfitting: bias dominates
    Right of sweet spot → overfitting:  variance dominates
    At the sweet spot   → bias² ≈ variance (loosely; not always exact)


──────────────────────────────────────────────────────────────────────────────
### Why the Tradeoff is Structural, Not Incidental

The tradeoff is not a practical limitation that better algorithms can
eliminate — it is a mathematical inevitability given a finite training set.

Intuition: to have low variance, a model must ignore fine-grained details
of the training set (otherwise those details — including noise — drive
predictions). But ignoring fine-grained details means the model cannot
adapt to the subtleties of f(x) → bias is introduced.

Conversely, to have low bias a model must faithfully track the local
structure of the training data. With finite n, that local structure
includes noise, which the model will then memorise → variance rises.

The only escape:
    1.  More data (n → ∞ reduces variance without changing bias)
    2.  Better inductive biases (priors that match f(x) let you have
        low bias AND low variance simultaneously — but you must be right
        about the prior, which requires domain knowledge)
    3.  Ensembles (average multiple high-variance models to cancel noise)



### PART 4 — DIAGNOSING BIAS VS VARIANCE FROM LEARNING CURVES

### Learning Curves as Diagnostic Tools

A learning curve plots training error and validation error against n
(the number of training examples). The shape of this curve tells you
exactly which problem you are facing.

**Case 1 — High Bias (Underfitting)**

    Error
      ↑
      │  ─────────────────────────────── ← Validation error
      │                                    plateaus HIGH
      │  ─────────────────────────────── ← Training error
      │                                    converges to SAME HIGH level
      │                                    (model is wrong everywhere)
      └──────────────────────────────── n

    Signature:
    • Both train and val errors are high
    • The GAP between them is small (both are stuck)
    • Adding more data barely helps

    Remedy: increase model capacity, add features, reduce regularisation

**Case 2 — High Variance (Overfitting)**

    Error
      ↑
      │  ─────────────────────────────── ← Validation error
      │                  ╲ converges     stays HIGH, converges slowly
      │                   ╲ slowly
      │                                  ← Training error
      │  ________________               stays LOW
      └──────────────────────────────── n

    Signature:
    • Train error is low, val error is much higher
    • The GAP between them is large
    • More data steadily narrows the gap

    Remedy: more data, regularisation, reduce model capacity, dropout,
            early stopping, feature selection

**Case 3 — Well Balanced**

    Error
      ↑
      │  ─────────────── ← Validation error converges to acceptable level
      │         ╲ small gap
      │  ────────────── ← Training error: slightly lower
      └──────────────────────────────── n

    Signature:
    • Small, stable gap between train and val
    • Both curves have flattened — the model is saturating the signal

──────────────────────────────────────────────────────────────────────────────
### The Practical Decision Tree

Given observed training and validation errors, diagnose and act:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Train error HIGH?                                              │
    │  ──────────────────────────────── YES ──► HIGH BIAS             │
    │       ↓ NO                                  ↓                   │
    │  Val error >> Train error?               Add capacity           │
    │  ─────────────────────────── YES ──► HIGH VARIANCE              │
    │       ↓ NO                                  ↓                   │
    │  Both acceptable ──────────────────► DONE ✓  Regularise/prune   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Critical mistake: applying the wrong fix.
    Adding more data to a high-bias model wastes resources — the model
    cannot use extra data because its hypothesis class is too small.
    Regularising a high-bias model makes performance WORSE — you are
    restricting an already under-powered model further.



### PART 5 — REGULARISATION AS BIAS-VARIANCE NEGOTIATION

### What Regularisation Really Does

Regularisation adds a penalty to the loss function that discourages
the model from using its full complexity:

    L1 (Lasso):   L(θ) = MSE(θ) + λ · Σ|θⱼ|
    L2 (Ridge):   L(θ) = MSE(θ) + λ · Σθⱼ²
    ElasticNet:   L(θ) = MSE(θ) + λ₁·Σ|θⱼ| + λ₂·Σθⱼ²

The regularisation strength λ is a complexity dial in reverse:
    λ = 0    → unregularised, low bias, high variance
    λ → ∞   → all weights → 0 (predict mean), high bias, zero variance

    Effect of λ on bias and variance:

    λ small ──────────────────────────────────────── λ large
                                                (towards predict-mean)
    Bias²     ────────────────────────────────────► increases
    Variance  ◄────────────────────────────────────  decreases
    Total MSE        ╲       ●       ╱               U-shaped in λ
                      ╲─────────────╱

Every regulariser is therefore a deliberate bias-injection device:
by restricting the model's freedom to fit the training data, we accept
some extra bias in exchange for a larger variance reduction.

**Why L2 shrinks weights but keeps them non-zero:**
    The L2 penalty creates a circular constraint region ||θ||₂ ≤ C.
    Its minimum rarely falls exactly on a coordinate axis.
    Solution: all weights pulled toward zero, but none exactly zero.

**Why L1 produces sparsity:**
    The L1 penalty creates a diamond-shaped constraint region ||θ||₁ ≤ C.
    Its corners lie on coordinate axes. Optimisation tends to land
    on corners → exactly zero weights → automatic feature selection.

    Diagram — L1 vs L2 constraint regions (2D parameter space):

         θ₂
          ↑
          │    ╭─────╮          ╱╲
          │   ╭       ╮        ╱  ╲
          │   │   ●   │       ╱  ● ╲    ← MSE contours (ellipses)
          │   ╰       ╯      ╱      ╲   touch constraint at:
          │    ╰─────╯      ╱    ●   ╲
          │   L2 (circle)  ╱──────────╲  L1 (diamond)
          └──────────────────────────────── θ₁
          L2: solution at ●           L1: solution likely at corner
          both weights non-zero       one weight exactly zero



### PART 6 — ENSEMBLES: VARIANCE REDUCTION WITHOUT BIAS COST

### Why Averaging Reduces Variance

If f̂₁, f̂₂, …, f̂_B are B independently trained (identical architecture)
models, each with variance σ² and pairwise correlation ρ, then the
ensemble f̂_avg = (1/B) Σ f̂ᵦ has:

    Variance(f̂_avg) = ρσ² + (1-ρ)σ²/B

    As B → ∞:  Variance(f̂_avg) → ρσ²

Key insights:
    • If models are perfectly independent (ρ=0): variance → 0 as B → ∞
    • If models are perfectly correlated (ρ=1): no reduction at all
    • Averaging does NOT change bias (average of biased models is biased)
    • Real gains require DIVERSE models — the correlation ρ must be low

This is why Random Forests deliberately de-correlate their trees:
    • Bootstrap sampling: each tree sees a different random subset of data
    • Feature subsampling: each split considers only √p features
    Both mechanisms reduce ρ, which is the quantity that limits variance gain.

    Bias-Variance breakdown by ensemble strategy:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Method         │ What it reduces │ How                           │
    ├────────────────┼─────────────────┼───────────────────────────────┤
    │ Bagging        │ Variance        │ Average many low-bias models  │
    │ Random Forest  │ Variance        │ Bagging + feature subsampling │
    │ Boosting       │ Bias            │ Sequentially correct residuals│
    │ Stacking       │ Both            │ Meta-model on diverse preds   │
    └────────────────┴─────────────────┴───────────────────────────────┘

**Boosting — bias reduction via sequential correction:**

    Unlike bagging which trains models in parallel, boosting trains
    models SEQUENTIALLY. Each new model focuses on the examples the
    previous ensemble got wrong.

    Iteration 1: f̂₁ fits data — some systematic errors remain
    Iteration 2: f̂₂ fits the residuals of f̂₁
    Iteration 3: f̂₃ fits the residuals of f̂₁ + f̂₂
    …
    Final: F̂ = Σ αₜ f̂ₜ   (a weighted sum of weak learners)

    Each weak learner is deliberately low-variance (shallow trees).
    The sequential sum progressively reduces bias by approximating
    f(x) with increasing resolution — like a Taylor series expansion
    that adds corrective terms one by one.

    Overfitting risk: if you boost for too many rounds, each new
    model starts fitting noise in the residuals → variance creeps up.
    Early stopping is the main guard against this.



### PART 7 — DOUBLE DESCENT: WHEN THE CLASSIC PICTURE BREAKS DOWN

### The Modern Surprise: Overparameterised Models Can Generalise

Classical theory predicts that error increases monotonically beyond the
sweet spot. This is true for classical model families (polynomials, trees,
kernel methods). But overparameterised neural networks and random features
violate this picture — they show DOUBLE DESCENT.

    Error
      ↑
      │  ╲                  │                  ╲
      │   ╲                 │                   ╲────────────
      │    ╲           ╱    │                    classical
      │     ╲─────────╱     │
      │      classical      │   ← interpolation threshold
      │      regime         │       (parameters ≈ training points)
      │                     │
      │                     │╲
      │                     │ ╲
      │                     │  ╲──────────────── modern regime
      │                     │   (overparameterised)
      └───────────────────────────────────────── Model size / Complexity

The "interpolation threshold" is the point where the model has just
enough capacity to fit the training data exactly (zero training loss).
At this exact point, test error spikes — the model is forced into the
worst possible interpolation. Beyond this point:

    • The model has more parameters than data points
    • There are infinitely many functions that fit the training data
    • Gradient descent with standard initialisation finds the minimum-
      norm solution — the SMOOTHEST interpolation
    • This smooth interpolation generalises surprisingly well

Why does this matter?
    • Modern neural networks routinely operate in this overparameterised
      regime (GPT-4 has ~1.8 trillion parameters; ImageNet has ~1.3M images)
    • The classical "regularise to fight overfitting" prescription does
      not straightforwardly apply — scale itself acts as a regulariser
    • Implicit regularisation from SGD, architecture, and initialisation
      do the work that explicit λ-penalisation does classically

The key practical takeaway: the U-shaped curve is correct for the
classical regime. When you scale far beyond it (as in deep learning),
a second descent can rescue you — but this requires the right
optimiser, architecture, and enough data to be safe.

    Summary of regimes:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Regime            │ Parameters vs n │ Generalisation mechanism   │
    ├───────────────────┼─────────────────┼────────────────────────────┤
    │ Underparameterised│ p << n          │ Fit quality (reduce bias)  │
    │ Overparameterised │ p >> n          │ Implicit regularisation    │
    │ Interpolation     │ p ≈ n           │ Worst of both worlds       │
    │ threshold         │                 │ (avoid this zone)          │
    └───────────────────┴─────────────────┴────────────────────────────┘



### PART 8 — SOURCES OF BIAS AND VARIANCE IN COMMON MODELS

### Model-Specific Intuitions

Understanding WHERE bias and variance come from in each model family
lets you make targeted improvements rather than random hyperparameter
searches.

**Linear / Logistic Regression**
    Bias source:     Assumes f(x) is linear — wrong for most real problems
    Variance source: Low — very few parameters relative to data
    Fixes for bias:  Polynomial features, interaction terms, kernel trick
    Fixes for variance: Rarely needed; already low

**k-Nearest Neighbours (k-NN)**
    k=1 (maximum complexity):
        Bias:     Near zero — uses only the nearest point, no assumptions
        Variance: Very high — single noisy neighbour dominates prediction
    k=n (predict mean):
        Bias:     Very high — same prediction everywhere regardless of x
        Variance: Zero — always predicts the training mean

    This is one of the clearest illustrations of the tradeoff:
    the single hyperparameter k continuously trades bias for variance.

    As n → ∞, with k/n → 0:  optimal k ∝ n^(4/(d+4)) where d = dimension

**Decision Trees**
    Bias source:     Axis-aligned splits — can't represent diagonal boundaries
                     precisely; each leaf approximates f(x) as a constant
    Variance source: Small perturbations to training data change the
                     split structure drastically
    Fix for bias:    Grow deeper (more leaves)
    Fix for variance: Prune, set min_samples_leaf, or use Random Forest

**Support Vector Machines (SVM)**
    Bias source:     Kernel choice; RBF with large γ → tight fit → low bias
    Variance source: Large γ → very sensitive to each training point
    C (margin cost): Large C → low bias, high variance
                     Small C → high bias, low variance (wider margin)

**Neural Networks**
    Bias source:     Architecture too small; wrong activation functions;
                     under-training (too few epochs)
    Variance source: Architecture too large; too many epochs; too large lr;
                     insufficient regularisation or dropout
    Unique property: Both can be tuned independently to a degree:
                     • Width/depth → mainly affects bias
                     • Dropout, weight decay, batch norm → mainly reduce variance

    Full comparison:
    ┌────────────────────┬──────────────┬──────────────┬──────────────────┐
    │ Model              │ Default Bias │ Default Var  │ Complexity knob  │
    ├────────────────────┼──────────────┼──────────────┼──────────────────┤
    │ Linear Regression  │ High         │ Very Low     │ Polynomial degree│
    │ k-NN               │ k-dependent  │ k-dependent  │ k (inverse)      │
    │ Decision Tree      │ Low (deep)   │ High (deep)  │ max_depth        │
    │ Random Forest      │ Low          │ Medium→Low   │ n_estimators     │
    │ SVM (RBF)          │ γ-dependent  │ γ-dependent  │ C, γ             │
    │ Neural Network     │ Arch-dep.    │ Arch-dep.    │ width × depth    │
    └────────────────────┴──────────────┴──────────────┴──────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Direct Bias-Variance Decomposition from First Principles": {
        "description": (
            "Empirically compute bias² and variance by training a model on "
            "hundreds of independent datasets drawn from the same distribution. "
            "Verify the exact identity: MSE = Bias² + Variance + Noise. "
            "Compare across five polynomial degrees to see the tradeoff live."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt
from itertools import combinations_with_replacement as _cwr

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LinearRegression:
    def __init__(self,fit_intercept=True): self.fit_intercept=fit_intercept
    def fit(self,X,y):
        Xa=np.column_stack([np.ones(len(X)),X]) if self.fit_intercept else X
        w,_,_,_=np.linalg.lstsq(Xa,y,rcond=None)
        if self.fit_intercept: self.intercept_=w[0]; self.coef_=w[1:]
        else: self.intercept_=0.; self.coef_=w
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def score(self,X,y):
        p=self.predict(X); ss_res=np.sum((y-p)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True): return {"fit_intercept":self.fit_intercept}

class Ridge:
    def __init__(self,alpha=1.0,fit_intercept=True):
        self.alpha=alpha; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape
        self.coef_=np.linalg.solve(Xc.T@Xc+self.alpha*np.eye(d),Xc.T@yc)
        self.intercept_=y.mean()-mu@self.coef_ if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"fit_intercept":self.fit_intercept}

class Lasso:
    def __init__(self,alpha=1.0,max_iter=500,fit_intercept=True):
        self.alpha=alpha; self.max_iter=max_iter; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=yc-Xc@w+Xc[:,j]*w[j]; rho=Xc[:,j]@r; z=np.sum(Xc[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w
        self.intercept_=y.mean()-mu@w if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"max_iter":self.max_iter}

class StandardScaler:
    def fit(self,X): self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {}

class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True): self.degree=degree; self.include_bias=include_bias
    def _combos(self,n):
        c=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: c.append(())
            else: c.extend(_cwr(range(n),d))
        return c
    def fit(self,X,y=None): self._c=self._combos(X.shape[1]); self.n_output_features_=len(self._c); return self
    def transform(self,X):
        cols=[]
        for combo in self._c:
            if not combo: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {"degree":self.degree,"include_bias":self.include_bias}

class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,s in self.steps[:-1]: Xt=s.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def score(self,X,y):
        yp=self.predict(X); ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True):
        p={}
        for nm,s in self.steps: p.update({nm+"__"+k:v for k,v in s.get_params().items()})
        return p

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

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        return np.array([self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int).ravel().tolist()
                         and [np.bincount(self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int)).argmax()]
                         for xi in X]).ravel()
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

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

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        if scoring=="accuracy": scores.append((est.predict(X[te])==y[te]).mean())
        elif scoring=="neg_mean_squared_error": p=est.predict(X[te]); scores.append(-np.mean((y[te]-p)**2))
        else: scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

def learning_curve(estimator,X,y,train_sizes=None,cv=5,scoring="neg_mean_squared_error",n_jobs=None,random_state=None):
    if train_sizes is None: train_sizes=np.linspace(0.1,1.0,5)
    n_tot=len(X); abs_sizes=np.clip(np.array([int(s) if s>1 else int(s*n_tot) for s in train_sizes]),2,n_tot)
    if isinstance(cv,int):
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    else: splits=list(cv.split(X,y))
    tr_sc=np.zeros((len(abs_sizes),len(splits))); val_sc=np.zeros((len(abs_sizes),len(splits)))
    for si,(tr,te) in enumerate(splits):
        for ti,sz in enumerate(abs_sizes):
            sub=tr[:sz] if sz<=len(tr) else tr; est=_copy.deepcopy(estimator); est.fit(X[sub],y[sub])
            if scoring=="neg_mean_squared_error":
                tr_sc[ti,si]=-np.mean((y[sub]-est.predict(X[sub]))**2)
                val_sc[ti,si]=-np.mean((y[te]-est.predict(X[te]))**2)
            else:
                tr_sc[ti,si]=(est.predict(X[sub])==y[sub]).mean()
                val_sc[ti,si]=(est.predict(X[te])==y[te]).mean()
    return abs_sizes,tr_sc,val_sc

class DecisionTreeRegressor:
    def __init__(self,max_depth=None,min_samples_split=2,random_state=None):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split; self.random_state=random_state
    def _mse(self,y): return np.var(y)*len(y) if len(y)>0 else 0
    def _split(self,X,y):
        best=None; best_g=self._mse(y); n=len(y)
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2; thrs=all_t[np.linspace(0,len(all_t)-1,min(8,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples_split or r.sum()<self.min_samples_split: continue
                g=self._mse(y[l])+self._mse(y[r])
                if g<best_g: best_g=g; best=(f,t)
        return best
    def _build(self,X,y,depth):
        if (self.max_depth is not None and depth>=self.max_depth) or len(y)<=self.min_samples_split:
            return {"leaf":True,"val":y.mean()}
        sp=self._split(X,y)
        if sp is None: return {"leaf":True,"val":y.mean()}
        f,t=sp; l=X[:,f]<=t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[l],y[l],depth+1),"r":self._build(X[~l],y[~l],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd["leaf"]: return nd["val"]
        return self._p1(x,nd["l"] if x[nd["f"]]<=nd["t"] else nd["r"])
    def predict(self,X): return np.array([self._p1(x,self.tree_) for x in X])
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":self.random_state}

class RandomForestRegressor:
    def __init__(self,n_estimators=10,max_depth=None,max_features=0.7,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(X.shape[1]*self.max_features)); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X[np.ix_(idx,fi)],y[idx]); self.estimators_.append((t,fi))
        return self
    def predict(self,X): return np.mean([t.predict(X[:,fi]) for t,fi in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,"max_features":self.max_features,"random_state":self.random_state}

class GradientBoostingRegressor:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); self.init_=y.mean()
        F=np.full(len(y),self.init_); self.estimators_=[]
        for i in range(self.n_estimators):
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X,y-F); self.estimators_.append(t); F+=self.learning_rate*t.predict(X)
        return self
    def predict(self,X):
        F=np.full(len(X),self.init_)
        for t in self.estimators_: F+=self.learning_rate*t.predict(X)
        return F
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"learning_rate":self.learning_rate,"max_depth":self.max_depth,"random_state":self.random_state}

class BaggingRegressor:
    def __init__(self,estimator=None,n_estimators=10,random_state=None):
        self.estimator=estimator; self.n_estimators=n_estimators; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        base=self.estimator or DecisionTreeRegressor()
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_copy.deepcopy(base); t.fit(X[idx],y[idx]); self.estimators_.append(t)
        return self
    def predict(self,X): return np.mean([t.predict(X) for t in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators}


rng = np.random.default_rng(42)

# ── True data-generating process ─────────────────────────────────────────
def f_true(x):
    """True function: a gentle sinusoid — non-linear but smooth."""
    return np.sin(2 * np.pi * x)

NOISE_STD  = 0.3        # irreducible noise level
N_TRAIN    = 25         # small training set → clear overfitting
N_DATASETS = 500        # number of independent training sets to simulate
X_TEST     = np.linspace(0, 1, 200).reshape(-1, 1)
Y_TRUE     = f_true(X_TEST.ravel())
DEGREES    = [1, 3, 5, 9, 15]

print("=" * 72)
print("  BIAS-VARIANCE DECOMPOSITION — DIRECT EMPIRICAL MEASUREMENT")
print("=" * 72)
print(f"  True f(x) = sin(2πx),  noise σ = {NOISE_STD},  "
      f"n_train = {N_TRAIN},  datasets = {N_DATASETS}")
print()
print(f"  Verification: MSE ≈ Bias² + Variance + Noise (= {NOISE_STD**2:.4f})")
print()
print(f"  {'Degree':>8} │ {'Bias²':>10} │ {'Variance':>10} │ "
      f"{'Noise':>10} │ {'Sum':>10} │ {'Direct MSE':>12} │ {'Match?':>7}")
print(f"  {'─'*76}")

all_preds  = {}   # store for plotting
results    = []

for degree in DEGREES:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("lr",   LinearRegression()),
    ])

    preds_matrix = np.zeros((N_DATASETS, len(X_TEST)))  # shape: (B, n_test)

    for b in range(N_DATASETS):
        X_tr = rng.uniform(0, 1, N_TRAIN).reshape(-1, 1)
        y_tr = f_true(X_tr.ravel()) + rng.normal(0, NOISE_STD, N_TRAIN)
        model.fit(X_tr, y_tr)
        preds_matrix[b] = model.predict(X_TEST).ravel()

    all_preds[degree] = preds_matrix

    # ── Compute decomposition ─────────────────────────────────────────────
    mean_pred   = preds_matrix.mean(axis=0)            # E_D[f̂(x)]
    bias_sq     = np.mean((mean_pred - Y_TRUE) ** 2)   # E_x[Bias²(x)]
    variance    = np.mean(preds_matrix.var(axis=0))    # E_x[Var(x)]
    noise       = NOISE_STD ** 2                        # σ²_ε

    # Direct MSE: E_D,ε[(y - f̂)²] approximated with test noise
    y_test_noisy = Y_TRUE + rng.normal(0, NOISE_STD, len(Y_TRUE))
    direct_mse   = np.mean((y_test_noisy - mean_pred) ** 2)
    decomp_sum   = bias_sq + variance + noise
    match        = "✓" if abs(decomp_sum - direct_mse) < 0.02 else "≈"

    results.append((degree, bias_sq, variance, noise, decomp_sum, direct_mse))

    print(f"  {'deg=' + str(degree):>8} │ {bias_sq:>10.4f} │ {variance:>10.4f} │ "
          f"{noise:>10.4f} │ {decomp_sum:>10.4f} │ {direct_mse:>12.4f} │ {match:>7}")

print()
print("  INTERPRETATION:")
print("  • Degree 1: high bias² (line can't fit sinusoid), low variance")
print("  • Degree 3: best balance — sweet spot for this problem")
print("  • Degree 9+: bias² → 0 but variance explodes")
print("  • Sum ≈ Direct MSE confirms the decomposition identity holds")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(DEGREES), figsize=(18, 8))
fig.suptitle(
    "Bias-Variance Decomposition — Each Column is One Polynomial Degree\\n"
    "Top: Individual model fits (grey) + mean model (red) + truth (black)\\n"
    "Bottom: Pointwise Bias² and Variance across test points",
    fontsize=9, fontweight="bold"
)

for col, degree in enumerate(DEGREES):
    preds = all_preds[degree]
    mean_pred = preds.mean(axis=0)
    bias_sq_pointwise = (mean_pred - Y_TRUE) ** 2
    variance_pointwise = preds.var(axis=0)

    # Top: individual fits
    ax_top = axes[0, col]
    for b in range(min(40, N_DATASETS)):  # show 40 of the 500 fits
        ax_top.plot(X_TEST.ravel(), preds[b], color="silver",
                    alpha=0.2, lw=0.6)
    ax_top.plot(X_TEST.ravel(), Y_TRUE,    "k-",  lw=2,   label="f(x) true")
    ax_top.plot(X_TEST.ravel(), mean_pred, "r-",  lw=1.8, label="E[f̂(x)]")
    ax_top.set_ylim(-2.5, 2.5)
    ax_top.set_title(f"Degree {degree}\\nB²={results[col][1]:.3f}  "
                     f"V={results[col][2]:.3f}", fontsize=8)
    if col == 0:
        ax_top.legend(fontsize=6)
        ax_top.set_ylabel("y")

    # Bottom: pointwise decomposition
    ax_bot = axes[1, col]
    ax_bot.fill_between(X_TEST.ravel(), bias_sq_pointwise,
                        color="tomato", alpha=0.5, label="Bias²")
    ax_bot.fill_between(X_TEST.ravel(), variance_pointwise,
                        color="steelblue", alpha=0.5, label="Variance")
    ax_bot.set_ylim(0, 1.2)
    ax_bot.set_xlabel("x")
    if col == 0:
        ax_bot.set_ylabel("Error component")
        ax_bot.legend(fontsize=6)

plt.tight_layout()
plt.savefig("bv_decomposition.png", dpi=120)
print("\\n  Plot saved → bv_decomposition.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · The Classic U-Curve — Model Complexity vs Train/Test Error": {
        "description": (
            "Sweep model complexity across 15 polynomial degrees. "
            "Plot training error, validation error, bias², and variance on the "
            "same axes to produce the classic U-shaped test error curve and "
            "visually identify the sweet spot."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt
from itertools import combinations_with_replacement as _cwr

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LinearRegression:
    def __init__(self,fit_intercept=True): self.fit_intercept=fit_intercept
    def fit(self,X,y):
        Xa=np.column_stack([np.ones(len(X)),X]) if self.fit_intercept else X
        w,_,_,_=np.linalg.lstsq(Xa,y,rcond=None)
        if self.fit_intercept: self.intercept_=w[0]; self.coef_=w[1:]
        else: self.intercept_=0.; self.coef_=w
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def score(self,X,y):
        p=self.predict(X); ss_res=np.sum((y-p)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True): return {"fit_intercept":self.fit_intercept}

class Ridge:
    def __init__(self,alpha=1.0,fit_intercept=True):
        self.alpha=alpha; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape
        self.coef_=np.linalg.solve(Xc.T@Xc+self.alpha*np.eye(d),Xc.T@yc)
        self.intercept_=y.mean()-mu@self.coef_ if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"fit_intercept":self.fit_intercept}

class Lasso:
    def __init__(self,alpha=1.0,max_iter=500,fit_intercept=True):
        self.alpha=alpha; self.max_iter=max_iter; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=yc-Xc@w+Xc[:,j]*w[j]; rho=Xc[:,j]@r; z=np.sum(Xc[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w
        self.intercept_=y.mean()-mu@w if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"max_iter":self.max_iter}

class StandardScaler:
    def fit(self,X): self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {}

class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True): self.degree=degree; self.include_bias=include_bias
    def _combos(self,n):
        c=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: c.append(())
            else: c.extend(_cwr(range(n),d))
        return c
    def fit(self,X,y=None): self._c=self._combos(X.shape[1]); self.n_output_features_=len(self._c); return self
    def transform(self,X):
        cols=[]
        for combo in self._c:
            if not combo: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {"degree":self.degree,"include_bias":self.include_bias}

class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,s in self.steps[:-1]: Xt=s.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def score(self,X,y):
        yp=self.predict(X); ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True):
        p={}
        for nm,s in self.steps: p.update({nm+"__"+k:v for k,v in s.get_params().items()})
        return p

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

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        return np.array([self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int).ravel().tolist()
                         and [np.bincount(self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int)).argmax()]
                         for xi in X]).ravel()
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

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

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        if scoring=="accuracy": scores.append((est.predict(X[te])==y[te]).mean())
        elif scoring=="neg_mean_squared_error": p=est.predict(X[te]); scores.append(-np.mean((y[te]-p)**2))
        else: scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

def learning_curve(estimator,X,y,train_sizes=None,cv=5,scoring="neg_mean_squared_error",n_jobs=None,random_state=None):
    if train_sizes is None: train_sizes=np.linspace(0.1,1.0,5)
    n_tot=len(X); abs_sizes=np.clip(np.array([int(s) if s>1 else int(s*n_tot) for s in train_sizes]),2,n_tot)
    if isinstance(cv,int):
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    else: splits=list(cv.split(X,y))
    tr_sc=np.zeros((len(abs_sizes),len(splits))); val_sc=np.zeros((len(abs_sizes),len(splits)))
    for si,(tr,te) in enumerate(splits):
        for ti,sz in enumerate(abs_sizes):
            sub=tr[:sz] if sz<=len(tr) else tr; est=_copy.deepcopy(estimator); est.fit(X[sub],y[sub])
            if scoring=="neg_mean_squared_error":
                tr_sc[ti,si]=-np.mean((y[sub]-est.predict(X[sub]))**2)
                val_sc[ti,si]=-np.mean((y[te]-est.predict(X[te]))**2)
            else:
                tr_sc[ti,si]=(est.predict(X[sub])==y[sub]).mean()
                val_sc[ti,si]=(est.predict(X[te])==y[te]).mean()
    return abs_sizes,tr_sc,val_sc

class DecisionTreeRegressor:
    def __init__(self,max_depth=None,min_samples_split=2,random_state=None):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split; self.random_state=random_state
    def _mse(self,y): return np.var(y)*len(y) if len(y)>0 else 0
    def _split(self,X,y):
        best=None; best_g=self._mse(y); n=len(y)
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2; thrs=all_t[np.linspace(0,len(all_t)-1,min(8,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples_split or r.sum()<self.min_samples_split: continue
                g=self._mse(y[l])+self._mse(y[r])
                if g<best_g: best_g=g; best=(f,t)
        return best
    def _build(self,X,y,depth):
        if (self.max_depth is not None and depth>=self.max_depth) or len(y)<=self.min_samples_split:
            return {"leaf":True,"val":y.mean()}
        sp=self._split(X,y)
        if sp is None: return {"leaf":True,"val":y.mean()}
        f,t=sp; l=X[:,f]<=t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[l],y[l],depth+1),"r":self._build(X[~l],y[~l],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd["leaf"]: return nd["val"]
        return self._p1(x,nd["l"] if x[nd["f"]]<=nd["t"] else nd["r"])
    def predict(self,X): return np.array([self._p1(x,self.tree_) for x in X])
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":self.random_state}

class RandomForestRegressor:
    def __init__(self,n_estimators=10,max_depth=None,max_features=0.7,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(X.shape[1]*self.max_features)); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X[np.ix_(idx,fi)],y[idx]); self.estimators_.append((t,fi))
        return self
    def predict(self,X): return np.mean([t.predict(X[:,fi]) for t,fi in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,"max_features":self.max_features,"random_state":self.random_state}

class GradientBoostingRegressor:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); self.init_=y.mean()
        F=np.full(len(y),self.init_); self.estimators_=[]
        for i in range(self.n_estimators):
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X,y-F); self.estimators_.append(t); F+=self.learning_rate*t.predict(X)
        return self
    def predict(self,X):
        F=np.full(len(X),self.init_)
        for t in self.estimators_: F+=self.learning_rate*t.predict(X)
        return F
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"learning_rate":self.learning_rate,"max_depth":self.max_depth,"random_state":self.random_state}

class BaggingRegressor:
    def __init__(self,estimator=None,n_estimators=10,random_state=None):
        self.estimator=estimator; self.n_estimators=n_estimators; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        base=self.estimator or DecisionTreeRegressor()
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_copy.deepcopy(base); t.fit(X[idx],y[idx]); self.estimators_.append(t)
        return self
    def predict(self,X): return np.mean([t.predict(X) for t in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators}

# train_test_split defined above

rng = np.random.default_rng(7)

def f_true(x): return 0.5 * x**3 - x**2 + 0.3 * np.sin(5*x) + 1.2

N        = 80
NOISE    = 0.4
DEGREES  = list(range(1, 16))
N_BOOT   = 300   # datasets for bias/variance decomposition

# ── Generate a fixed test set (large, low noise) ──────────────────────────
X_all  = rng.uniform(-2, 2, N)
y_all  = f_true(X_all) + rng.normal(0, NOISE, N)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.35, random_state=0)

X_test = np.linspace(-2, 2, 300)
y_test_true = f_true(X_test)

print("=" * 65)
print("  U-CURVE: MODEL COMPLEXITY vs TRAIN / VALIDATION ERROR")
print("=" * 65)
print(f"  f(x) = 0.5x³ − x² + 0.3·sin(5x) + 1.2   (unknown to model)")
print(f"  Noise σ = {NOISE},  n_train = {len(X_tr)},  n_val = {len(X_val)}")
print()
print(f"  {'Degree':>7} │ {'Train MSE':>10} │ {'Val MSE':>10} │ "
      f"{'Bias²':>8} │ {'Variance':>10} │ {'Region'}")
print(f"  {'─'*65}")

train_errs = []
val_errs   = []
biases_sq  = []
variances  = []

for deg in DEGREES:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(deg, include_bias=True)),
        ("lr",   LinearRegression()),
    ])
    pipe.fit(X_tr.reshape(-1,1), y_tr)

    # Train and validation MSE
    tr_mse  = np.mean((pipe.predict(X_tr.reshape(-1,1)) - y_tr)**2)
    val_mse = np.mean((pipe.predict(X_val.reshape(-1,1)) - y_val)**2)
    train_errs.append(tr_mse)
    val_errs.append(val_mse)

    # Bias and variance via bootstrap on training data
    preds_boot = np.zeros((N_BOOT, len(X_test)))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(X_tr), len(X_tr))
        p = Pipeline([("poly", PolynomialFeatures(deg, include_bias=True)),
                      ("lr", LinearRegression())])
        p.fit(X_tr[idx].reshape(-1,1), y_tr[idx])
        preds_boot[b] = p.predict(X_test.reshape(-1,1))

    mean_pred = preds_boot.mean(axis=0)
    bias_sq   = np.mean((mean_pred - y_test_true)**2)
    variance  = np.mean(preds_boot.var(axis=0))
    biases_sq.append(bias_sq)
    variances.append(variance)

    if val_mse == min(val_errs):
        region = "← SWEET SPOT so far"
    elif tr_mse < 0.05 and val_mse > 1.5 * tr_mse:
        region = "← overfitting"
    elif tr_mse > 0.3:
        region = "← underfitting"
    else:
        region = ""

    print(f"  {deg:>7} │ {tr_mse:>10.4f} │ {val_mse:>10.4f} │ "
          f"{bias_sq:>8.4f} │ {variance:>10.4f} │ {region}")

best_deg = DEGREES[np.argmin(val_errs)]
print(f"\\n  Best degree by validation MSE: {best_deg}")
print(f"  At degree {best_deg}: Bias²={biases_sq[best_deg-1]:.4f},"
      f" Variance={variances[best_deg-1]:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("The U-Shaped Generalisation Error Curve", fontsize=11,
             fontweight="bold")

# Left: train vs validation
ax1 = axes[0]
ax1.plot(DEGREES, train_errs, "steelblue",  lw=2,  marker="o", ms=5,
         label="Training error")
ax1.plot(DEGREES, val_errs,   "tomato",     lw=2,  marker="s", ms=5,
         label="Validation error")
ax1.axvline(best_deg, color="gray", linestyle="--", lw=1.2,
            label=f"Sweet spot (deg={best_deg})")
ax1.set_xlabel("Polynomial degree (complexity)")
ax1.set_ylabel("MSE")
ax1.set_title("Training vs Validation Error", fontsize=9)
ax1.legend(fontsize=8)
ax1.set_ylim(0, min(2.5, max(val_errs) * 1.2))
ax1.grid(alpha=0.3)

# Right: bias-variance decomposition
ax2 = axes[1]
noise_floor = np.full(len(DEGREES), NOISE**2)
b2 = np.array(biases_sq)
v  = np.array(variances)
ax2.stackplot(DEGREES,
              noise_floor,
              b2,
              v,
              labels=["Irreducible noise", "Bias²", "Variance"],
              colors=["#d4e6f1", "#f1948a", "#85c1e9"],
              alpha=0.85)
ax2.plot(DEGREES, noise_floor + b2 + v, "k-", lw=2, label="Total (B²+V+N)")
ax2.axvline(best_deg, color="gray", linestyle="--", lw=1.2)
ax2.set_xlabel("Polynomial degree (complexity)")
ax2.set_ylabel("Error component")
ax2.set_title("Bias-Variance Decomposition", fontsize=9)
ax2.legend(fontsize=7, loc="upper center")
ax2.set_ylim(0, min(3, (b2+v+noise_floor).max() * 1.3))
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("u_curve.png", dpi=120)
print("\\n  Plot saved → u_curve.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Learning Curves — Diagnosing Bias vs Variance": {
        "description": (
            "Generate the canonical learning curve signatures for three scenarios: "
            "high bias (underfitting), high variance (overfitting), and well-balanced. "
            "Show how the train/val gap and plateau height diagnose each problem."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt
from itertools import combinations_with_replacement as _cwr

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LinearRegression:
    def __init__(self,fit_intercept=True): self.fit_intercept=fit_intercept
    def fit(self,X,y):
        Xa=np.column_stack([np.ones(len(X)),X]) if self.fit_intercept else X
        w,_,_,_=np.linalg.lstsq(Xa,y,rcond=None)
        if self.fit_intercept: self.intercept_=w[0]; self.coef_=w[1:]
        else: self.intercept_=0.; self.coef_=w
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def score(self,X,y):
        p=self.predict(X); ss_res=np.sum((y-p)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True): return {"fit_intercept":self.fit_intercept}

class Ridge:
    def __init__(self,alpha=1.0,fit_intercept=True):
        self.alpha=alpha; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape
        self.coef_=np.linalg.solve(Xc.T@Xc+self.alpha*np.eye(d),Xc.T@yc)
        self.intercept_=y.mean()-mu@self.coef_ if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"fit_intercept":self.fit_intercept}

class Lasso:
    def __init__(self,alpha=1.0,max_iter=500,fit_intercept=True):
        self.alpha=alpha; self.max_iter=max_iter; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=yc-Xc@w+Xc[:,j]*w[j]; rho=Xc[:,j]@r; z=np.sum(Xc[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w
        self.intercept_=y.mean()-mu@w if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"max_iter":self.max_iter}

class StandardScaler:
    def fit(self,X): self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {}

class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True): self.degree=degree; self.include_bias=include_bias
    def _combos(self,n):
        c=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: c.append(())
            else: c.extend(_cwr(range(n),d))
        return c
    def fit(self,X,y=None): self._c=self._combos(X.shape[1]); self.n_output_features_=len(self._c); return self
    def transform(self,X):
        cols=[]
        for combo in self._c:
            if not combo: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {"degree":self.degree,"include_bias":self.include_bias}

class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,s in self.steps[:-1]: Xt=s.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def score(self,X,y):
        yp=self.predict(X); ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True):
        p={}
        for nm,s in self.steps: p.update({nm+"__"+k:v for k,v in s.get_params().items()})
        return p

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

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        return np.array([self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int).ravel().tolist()
                         and [np.bincount(self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int)).argmax()]
                         for xi in X]).ravel()
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

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

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        if scoring=="accuracy": scores.append((est.predict(X[te])==y[te]).mean())
        elif scoring=="neg_mean_squared_error": p=est.predict(X[te]); scores.append(-np.mean((y[te]-p)**2))
        else: scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

def learning_curve(estimator,X,y,train_sizes=None,cv=5,scoring="neg_mean_squared_error",n_jobs=None,random_state=None):
    if train_sizes is None: train_sizes=np.linspace(0.1,1.0,5)
    n_tot=len(X); abs_sizes=np.clip(np.array([int(s) if s>1 else int(s*n_tot) for s in train_sizes]),2,n_tot)
    if isinstance(cv,int):
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    else: splits=list(cv.split(X,y))
    tr_sc=np.zeros((len(abs_sizes),len(splits))); val_sc=np.zeros((len(abs_sizes),len(splits)))
    for si,(tr,te) in enumerate(splits):
        for ti,sz in enumerate(abs_sizes):
            sub=tr[:sz] if sz<=len(tr) else tr; est=_copy.deepcopy(estimator); est.fit(X[sub],y[sub])
            if scoring=="neg_mean_squared_error":
                tr_sc[ti,si]=-np.mean((y[sub]-est.predict(X[sub]))**2)
                val_sc[ti,si]=-np.mean((y[te]-est.predict(X[te]))**2)
            else:
                tr_sc[ti,si]=(est.predict(X[sub])==y[sub]).mean()
                val_sc[ti,si]=(est.predict(X[te])==y[te]).mean()
    return abs_sizes,tr_sc,val_sc

class DecisionTreeRegressor:
    def __init__(self,max_depth=None,min_samples_split=2,random_state=None):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split; self.random_state=random_state
    def _mse(self,y): return np.var(y)*len(y) if len(y)>0 else 0
    def _split(self,X,y):
        best=None; best_g=self._mse(y); n=len(y)
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2; thrs=all_t[np.linspace(0,len(all_t)-1,min(8,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples_split or r.sum()<self.min_samples_split: continue
                g=self._mse(y[l])+self._mse(y[r])
                if g<best_g: best_g=g; best=(f,t)
        return best
    def _build(self,X,y,depth):
        if (self.max_depth is not None and depth>=self.max_depth) or len(y)<=self.min_samples_split:
            return {"leaf":True,"val":y.mean()}
        sp=self._split(X,y)
        if sp is None: return {"leaf":True,"val":y.mean()}
        f,t=sp; l=X[:,f]<=t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[l],y[l],depth+1),"r":self._build(X[~l],y[~l],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd["leaf"]: return nd["val"]
        return self._p1(x,nd["l"] if x[nd["f"]]<=nd["t"] else nd["r"])
    def predict(self,X): return np.array([self._p1(x,self.tree_) for x in X])
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":self.random_state}

class RandomForestRegressor:
    def __init__(self,n_estimators=10,max_depth=None,max_features=0.7,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(X.shape[1]*self.max_features)); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X[np.ix_(idx,fi)],y[idx]); self.estimators_.append((t,fi))
        return self
    def predict(self,X): return np.mean([t.predict(X[:,fi]) for t,fi in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,"max_features":self.max_features,"random_state":self.random_state}

class GradientBoostingRegressor:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); self.init_=y.mean()
        F=np.full(len(y),self.init_); self.estimators_=[]
        for i in range(self.n_estimators):
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X,y-F); self.estimators_.append(t); F+=self.learning_rate*t.predict(X)
        return self
    def predict(self,X):
        F=np.full(len(X),self.init_)
        for t in self.estimators_: F+=self.learning_rate*t.predict(X)
        return F
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"learning_rate":self.learning_rate,"max_depth":self.max_depth,"random_state":self.random_state}

class BaggingRegressor:
    def __init__(self,estimator=None,n_estimators=10,random_state=None):
        self.estimator=estimator; self.n_estimators=n_estimators; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        base=self.estimator or DecisionTreeRegressor()
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_copy.deepcopy(base); t.fit(X[idx],y[idx]); self.estimators_.append(t)
        return self
    def predict(self,X): return np.mean([t.predict(X) for t in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators}


rng = np.random.default_rng(0)

def f_true(x): return np.sin(2 * np.pi * x) + 0.3 * x**2

NOISE = 0.35

# ── Build a large pool then draw learning curves ──────────────────────────
N_POOL = 1000
X_pool = rng.uniform(0, 1, N_POOL).reshape(-1, 1)
y_pool = f_true(X_pool.ravel()) + rng.normal(0, NOISE, N_POOL)

TRAIN_SIZES = np.unique(np.concatenate([
    np.linspace(10, 50, 10, dtype=int),
    np.linspace(50, 400, 12, dtype=int),
]))

scenarios = {
    "High Bias — Underfitting\\n(degree=1 linear model)": Pipeline([
        ("poly", PolynomialFeatures(1)),
        ("lr",   LinearRegression()),
    ]),
    "High Variance — Overfitting\\n(degree=12, no regularisation)": Pipeline([
        ("poly", PolynomialFeatures(12)),
        ("lr",   LinearRegression()),
    ]),
    "Well Balanced\\n(degree=4, Ridge regularisation)": Pipeline([
        ("poly", PolynomialFeatures(4)),
        ("lr",   Ridge(alpha=0.01)),
    ]),
}

print("=" * 68)
print("  LEARNING CURVES — DIAGNOSING BIAS vs VARIANCE")
print("=" * 68)
print()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle(
    "Learning Curve Signatures — How Train/Val Gap Reveals the Problem",
    fontsize=10, fontweight="bold"
)

for ax, (title, model) in zip(axes, scenarios.items()):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_pool, y_pool,
        train_sizes=TRAIN_SIZES,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=1,
    )
    # neg MSE → MSE
    tr_mean  = -train_scores.mean(axis=1)
    tr_std   = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std  = val_scores.std(axis=1)

    ax.plot(train_sizes_abs, tr_mean,  "steelblue", lw=2, label="Train MSE")
    ax.plot(train_sizes_abs, val_mean, "tomato",    lw=2, label="Val MSE")
    ax.fill_between(train_sizes_abs,
                    tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color="steelblue")
    ax.fill_between(train_sizes_abs,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="tomato")

    final_gap = val_mean[-1] - tr_mean[-1]
    ax.axhline(NOISE**2, color="gray", linestyle=":", lw=1.2,
               label=f"Noise floor ({NOISE**2:.3f})")

    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("Training set size (n)")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    name = title.split("\\n")[0]
    print(f"  {name}")
    print(f"    Final train MSE:  {tr_mean[-1]:.4f}")
    print(f"    Final val MSE:    {val_mean[-1]:.4f}")
    print(f"    Gap (val-train):  {final_gap:.4f}")
    print(f"    Noise floor:      {NOISE**2:.4f}")
    if tr_mean[-1] > 3 * NOISE**2 and final_gap < 0.05:
        diag = "HIGH BIAS  — both curves plateau HIGH, small gap"
        fix  = "Increase model complexity or add features"
    elif final_gap > 0.15:
        diag = "HIGH VARIANCE — large gap, train stays low"
        fix  = "More data, regularise, reduce complexity"
    else:
        diag = "BALANCED  — small gap, both near noise floor"
        fix  = "Looks good. Monitor with more data."
    print(f"    Diagnosis: {diag}")
    print(f"    Fix:       {fix}")
    print()

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=120)
print("  Plot saved → learning_curves.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Regularisation as Bias-Variance Dial — Ridge vs Lasso": {
        "description": (
            "Sweep the regularisation strength λ across 6 orders of magnitude "
            "for both Ridge and Lasso. Plot the resulting bias², variance, "
            "test MSE, and the number of non-zero coefficients — showing how "
            "λ is a continuous bias-variance tradeoff control."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt
from itertools import combinations_with_replacement as _cwr

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LinearRegression:
    def __init__(self,fit_intercept=True): self.fit_intercept=fit_intercept
    def fit(self,X,y):
        Xa=np.column_stack([np.ones(len(X)),X]) if self.fit_intercept else X
        w,_,_,_=np.linalg.lstsq(Xa,y,rcond=None)
        if self.fit_intercept: self.intercept_=w[0]; self.coef_=w[1:]
        else: self.intercept_=0.; self.coef_=w
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def score(self,X,y):
        p=self.predict(X); ss_res=np.sum((y-p)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True): return {"fit_intercept":self.fit_intercept}

class Ridge:
    def __init__(self,alpha=1.0,fit_intercept=True):
        self.alpha=alpha; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape
        self.coef_=np.linalg.solve(Xc.T@Xc+self.alpha*np.eye(d),Xc.T@yc)
        self.intercept_=y.mean()-mu@self.coef_ if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"fit_intercept":self.fit_intercept}

class Lasso:
    def __init__(self,alpha=1.0,max_iter=500,fit_intercept=True):
        self.alpha=alpha; self.max_iter=max_iter; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=yc-Xc@w+Xc[:,j]*w[j]; rho=Xc[:,j]@r; z=np.sum(Xc[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w
        self.intercept_=y.mean()-mu@w if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"max_iter":self.max_iter}

class StandardScaler:
    def fit(self,X): self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {}

class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True): self.degree=degree; self.include_bias=include_bias
    def _combos(self,n):
        c=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: c.append(())
            else: c.extend(_cwr(range(n),d))
        return c
    def fit(self,X,y=None): self._c=self._combos(X.shape[1]); self.n_output_features_=len(self._c); return self
    def transform(self,X):
        cols=[]
        for combo in self._c:
            if not combo: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {"degree":self.degree,"include_bias":self.include_bias}

class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,s in self.steps[:-1]: Xt=s.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def score(self,X,y):
        yp=self.predict(X); ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True):
        p={}
        for nm,s in self.steps: p.update({nm+"__"+k:v for k,v in s.get_params().items()})
        return p

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

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        return np.array([self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int).ravel().tolist()
                         and [np.bincount(self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int)).argmax()]
                         for xi in X]).ravel()
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

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

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        if scoring=="accuracy": scores.append((est.predict(X[te])==y[te]).mean())
        elif scoring=="neg_mean_squared_error": p=est.predict(X[te]); scores.append(-np.mean((y[te]-p)**2))
        else: scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

def learning_curve(estimator,X,y,train_sizes=None,cv=5,scoring="neg_mean_squared_error",n_jobs=None,random_state=None):
    if train_sizes is None: train_sizes=np.linspace(0.1,1.0,5)
    n_tot=len(X); abs_sizes=np.clip(np.array([int(s) if s>1 else int(s*n_tot) for s in train_sizes]),2,n_tot)
    if isinstance(cv,int):
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    else: splits=list(cv.split(X,y))
    tr_sc=np.zeros((len(abs_sizes),len(splits))); val_sc=np.zeros((len(abs_sizes),len(splits)))
    for si,(tr,te) in enumerate(splits):
        for ti,sz in enumerate(abs_sizes):
            sub=tr[:sz] if sz<=len(tr) else tr; est=_copy.deepcopy(estimator); est.fit(X[sub],y[sub])
            if scoring=="neg_mean_squared_error":
                tr_sc[ti,si]=-np.mean((y[sub]-est.predict(X[sub]))**2)
                val_sc[ti,si]=-np.mean((y[te]-est.predict(X[te]))**2)
            else:
                tr_sc[ti,si]=(est.predict(X[sub])==y[sub]).mean()
                val_sc[ti,si]=(est.predict(X[te])==y[te]).mean()
    return abs_sizes,tr_sc,val_sc

class DecisionTreeRegressor:
    def __init__(self,max_depth=None,min_samples_split=2,random_state=None):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split; self.random_state=random_state
    def _mse(self,y): return np.var(y)*len(y) if len(y)>0 else 0
    def _split(self,X,y):
        best=None; best_g=self._mse(y); n=len(y)
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2; thrs=all_t[np.linspace(0,len(all_t)-1,min(8,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples_split or r.sum()<self.min_samples_split: continue
                g=self._mse(y[l])+self._mse(y[r])
                if g<best_g: best_g=g; best=(f,t)
        return best
    def _build(self,X,y,depth):
        if (self.max_depth is not None and depth>=self.max_depth) or len(y)<=self.min_samples_split:
            return {"leaf":True,"val":y.mean()}
        sp=self._split(X,y)
        if sp is None: return {"leaf":True,"val":y.mean()}
        f,t=sp; l=X[:,f]<=t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[l],y[l],depth+1),"r":self._build(X[~l],y[~l],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd["leaf"]: return nd["val"]
        return self._p1(x,nd["l"] if x[nd["f"]]<=nd["t"] else nd["r"])
    def predict(self,X): return np.array([self._p1(x,self.tree_) for x in X])
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":self.random_state}

class RandomForestRegressor:
    def __init__(self,n_estimators=10,max_depth=None,max_features=0.7,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(X.shape[1]*self.max_features)); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X[np.ix_(idx,fi)],y[idx]); self.estimators_.append((t,fi))
        return self
    def predict(self,X): return np.mean([t.predict(X[:,fi]) for t,fi in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,"max_features":self.max_features,"random_state":self.random_state}

class GradientBoostingRegressor:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); self.init_=y.mean()
        F=np.full(len(y),self.init_); self.estimators_=[]
        for i in range(self.n_estimators):
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X,y-F); self.estimators_.append(t); F+=self.learning_rate*t.predict(X)
        return self
    def predict(self,X):
        F=np.full(len(X),self.init_)
        for t in self.estimators_: F+=self.learning_rate*t.predict(X)
        return F
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"learning_rate":self.learning_rate,"max_depth":self.max_depth,"random_state":self.random_state}

class BaggingRegressor:
    def __init__(self,estimator=None,n_estimators=10,random_state=None):
        self.estimator=estimator; self.n_estimators=n_estimators; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        base=self.estimator or DecisionTreeRegressor()
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_copy.deepcopy(base); t.fit(X[idx],y[idx]); self.estimators_.append(t)
        return self
    def predict(self,X): return np.mean([t.predict(X) for t in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators}


rng = np.random.default_rng(13)

def f_true(x): return np.sin(2 * np.pi * x)

NOISE    = 0.3
N_TRAIN  = 30
DEGREE   = 9      # overparameterised basis — regularisation does the work
N_BOOT   = 60
X_TEST   = np.linspace(0, 1, 200)
Y_TRUE   = f_true(X_TEST)

# ── Fixed training set and bootstrap engine ───────────────────────────────
X_tr_base = rng.uniform(0, 1, N_TRAIN)
y_tr_base = f_true(X_tr_base) + rng.normal(0, NOISE, N_TRAIN)

LAMBDAS  = np.logspace(-4, 2, 20)

def bv_sweep(model_cls, lambdas):
    """Return arrays of bias², variance, mse for each lambda."""
    b2_arr, v_arr, mse_arr, nnz_arr = [], [], [], []
    for lam in lambdas:
        preds_boot = np.zeros((N_BOOT, len(X_TEST)))
        coef_boot  = []
        for b in range(N_BOOT):
            idx = rng.integers(0, N_TRAIN, N_TRAIN)
            poly = PolynomialFeatures(DEGREE, include_bias=False)
            sc   = StandardScaler()
            reg  = model_cls(alpha=max(lam, 1e-9))
            Xb   = sc.fit_transform(poly.fit_transform(
                       X_tr_base[idx].reshape(-1,1)))
            yb   = y_tr_base[idx]
            reg.fit(Xb, yb)
            Xt   = sc.transform(poly.transform(X_TEST.reshape(-1,1)))
            preds_boot[b] = reg.predict(Xt)
            coef_boot.append(np.sum(np.abs(reg.coef_) > 1e-6))

        mean_p = preds_boot.mean(axis=0)
        b2_arr.append(np.mean((mean_p - Y_TRUE)**2))
        v_arr.append(np.mean(preds_boot.var(axis=0)))
        mse_arr.append(b2_arr[-1] + v_arr[-1] + NOISE**2)
        nnz_arr.append(np.mean(coef_boot))

    return (np.array(b2_arr), np.array(v_arr),
            np.array(mse_arr), np.array(nnz_arr))

print("=" * 70)
print("  REGULARISATION SWEEP — BIAS-VARIANCE TRADEOFF vs λ")
print("=" * 70)
print(f"  Model: degree-{DEGREE} polynomial,  n={N_TRAIN},  noise σ={NOISE}")
print(f"  λ sweeps from {LAMBDAS[0]:.1e} (unregularised) to "
      f"{LAMBDAS[-1]:.1e} (over-regularised)")
print()

b2_ridge, v_ridge, mse_ridge, _ = bv_sweep(Ridge, LAMBDAS)
b2_lasso, v_lasso, mse_lasso, nnz_lasso = bv_sweep(Lasso, LAMBDAS)

# Print key lambda checkpoints
checkpoints = [0, 4, 9, 14, 19]
print(f"  RIDGE:   {'λ':>10} │ {'Bias²':>8} │ {'Var':>8} │ {'Total MSE':>10}")
print(f"  {'─'*50}")
for i in checkpoints:
    flag = " ← optimal" if mse_ridge[i] == mse_ridge.min() else ""
    print(f"           {LAMBDAS[i]:>10.4f} │ {b2_ridge[i]:>8.4f} │ "
          f"{v_ridge[i]:>8.4f} │ {mse_ridge[i]:>10.4f}{flag}")

best_lam_ridge = LAMBDAS[np.argmin(mse_ridge)]
best_lam_lasso = LAMBDAS[np.argmin(mse_lasso)]
print(f"\\n  Best λ (Ridge): {best_lam_ridge:.4f} → "
      f"MSE = {mse_ridge.min():.4f}")
print(f"  Best λ (Lasso): {best_lam_lasso:.4f} → "
      f"MSE = {mse_lasso.min():.4f}")
print()
print("  KEY INSIGHT:")
print("  • Small λ: near-zero bias, high variance (fits training noise)")
print("  • Large λ: all coefficients → 0, high bias, near-zero variance")
print("  • Sweet spot: optimal λ minimises total test MSE")
print("  • Lasso produces sparsity — nnz_coefficients drops sharply with λ")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Regularisation λ as a Continuous Bias-Variance Dial",
    fontsize=10, fontweight="bold"
)

for ax, b2, v, mse, nnz, name, colour in [
    (axes[0], b2_ridge, v_ridge, mse_ridge, None, "Ridge", "steelblue"),
    (axes[1], b2_lasso, v_lasso, mse_lasso, nnz_lasso, "Lasso", "tomato"),
]:
    ax.semilogx(LAMBDAS, b2,  colour,   lw=2, label="Bias²",    linestyle="-")
    ax.semilogx(LAMBDAS, v,   "gray",   lw=2, label="Variance", linestyle="--")
    ax.semilogx(LAMBDAS, mse, "black",  lw=2.5, label="Total MSE")
    ax.axhline(NOISE**2, color="lightgray", linestyle=":",
               lw=1.2, label=f"Noise floor")
    best_idx = np.argmin(mse)
    ax.axvline(LAMBDAS[best_idx], color=colour, linestyle=":",
               lw=1.5, label=f"Optimal λ={LAMBDAS[best_idx]:.3f}")
    ax.set_xlabel("λ (regularisation strength, log scale)")
    ax.set_ylabel("Error")
    ax.set_title(f"{name}: Bias²-Variance vs λ", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.8)

# Right: Lasso sparsity
ax3 = axes[2]
ax3.semilogx(LAMBDAS, nnz_lasso, "tomato", lw=2.5, marker="o", ms=3)
ax3.set_xlabel("λ (regularisation strength, log scale)")
ax3.set_ylabel("Mean non-zero coefficients")
ax3.set_title("Lasso Sparsity vs λ\\n(L1 drives coefficients to exactly 0)",
              fontsize=9)
ax3.axvline(best_lam_lasso, color="gray", linestyle="--", lw=1.2)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("regularisation_bv.png", dpi=120)
print("\\n  Plot saved → regularisation_bv.png")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · k-NN — Sharpest Illustration of the Tradeoff": {
        "description": (
            "k-NN exposes the tradeoff through a single integer k. "
            "Sweep k from 1 to n, decompose bias and variance at each step, "
            "plot decision boundaries at k=1, k_optimal, and k=n. "
            "This is the cleanest possible illustration of the tradeoff."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt
from itertools import combinations_with_replacement as _cwr

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LinearRegression:
    def __init__(self,fit_intercept=True): self.fit_intercept=fit_intercept
    def fit(self,X,y):
        Xa=np.column_stack([np.ones(len(X)),X]) if self.fit_intercept else X
        w,_,_,_=np.linalg.lstsq(Xa,y,rcond=None)
        if self.fit_intercept: self.intercept_=w[0]; self.coef_=w[1:]
        else: self.intercept_=0.; self.coef_=w
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def score(self,X,y):
        p=self.predict(X); ss_res=np.sum((y-p)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True): return {"fit_intercept":self.fit_intercept}

class Ridge:
    def __init__(self,alpha=1.0,fit_intercept=True):
        self.alpha=alpha; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape
        self.coef_=np.linalg.solve(Xc.T@Xc+self.alpha*np.eye(d),Xc.T@yc)
        self.intercept_=y.mean()-mu@self.coef_ if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"fit_intercept":self.fit_intercept}

class Lasso:
    def __init__(self,alpha=1.0,max_iter=500,fit_intercept=True):
        self.alpha=alpha; self.max_iter=max_iter; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=yc-Xc@w+Xc[:,j]*w[j]; rho=Xc[:,j]@r; z=np.sum(Xc[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w
        self.intercept_=y.mean()-mu@w if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"max_iter":self.max_iter}

class StandardScaler:
    def fit(self,X): self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {}

class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True): self.degree=degree; self.include_bias=include_bias
    def _combos(self,n):
        c=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: c.append(())
            else: c.extend(_cwr(range(n),d))
        return c
    def fit(self,X,y=None): self._c=self._combos(X.shape[1]); self.n_output_features_=len(self._c); return self
    def transform(self,X):
        cols=[]
        for combo in self._c:
            if not combo: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {"degree":self.degree,"include_bias":self.include_bias}

class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,s in self.steps[:-1]: Xt=s.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def score(self,X,y):
        yp=self.predict(X); ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True):
        p={}
        for nm,s in self.steps: p.update({nm+"__"+k:v for k,v in s.get_params().items()})
        return p

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

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        return np.array([self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int).ravel().tolist()
                         and [np.bincount(self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int)).argmax()]
                         for xi in X]).ravel()
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

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

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        if scoring=="accuracy": scores.append((est.predict(X[te])==y[te]).mean())
        elif scoring=="neg_mean_squared_error": p=est.predict(X[te]); scores.append(-np.mean((y[te]-p)**2))
        else: scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

def learning_curve(estimator,X,y,train_sizes=None,cv=5,scoring="neg_mean_squared_error",n_jobs=None,random_state=None):
    if train_sizes is None: train_sizes=np.linspace(0.1,1.0,5)
    n_tot=len(X); abs_sizes=np.clip(np.array([int(s) if s>1 else int(s*n_tot) for s in train_sizes]),2,n_tot)
    if isinstance(cv,int):
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    else: splits=list(cv.split(X,y))
    tr_sc=np.zeros((len(abs_sizes),len(splits))); val_sc=np.zeros((len(abs_sizes),len(splits)))
    for si,(tr,te) in enumerate(splits):
        for ti,sz in enumerate(abs_sizes):
            sub=tr[:sz] if sz<=len(tr) else tr; est=_copy.deepcopy(estimator); est.fit(X[sub],y[sub])
            if scoring=="neg_mean_squared_error":
                tr_sc[ti,si]=-np.mean((y[sub]-est.predict(X[sub]))**2)
                val_sc[ti,si]=-np.mean((y[te]-est.predict(X[te]))**2)
            else:
                tr_sc[ti,si]=(est.predict(X[sub])==y[sub]).mean()
                val_sc[ti,si]=(est.predict(X[te])==y[te]).mean()
    return abs_sizes,tr_sc,val_sc

class DecisionTreeRegressor:
    def __init__(self,max_depth=None,min_samples_split=2,random_state=None):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split; self.random_state=random_state
    def _mse(self,y): return np.var(y)*len(y) if len(y)>0 else 0
    def _split(self,X,y):
        best=None; best_g=self._mse(y); n=len(y)
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2; thrs=all_t[np.linspace(0,len(all_t)-1,min(8,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples_split or r.sum()<self.min_samples_split: continue
                g=self._mse(y[l])+self._mse(y[r])
                if g<best_g: best_g=g; best=(f,t)
        return best
    def _build(self,X,y,depth):
        if (self.max_depth is not None and depth>=self.max_depth) or len(y)<=self.min_samples_split:
            return {"leaf":True,"val":y.mean()}
        sp=self._split(X,y)
        if sp is None: return {"leaf":True,"val":y.mean()}
        f,t=sp; l=X[:,f]<=t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[l],y[l],depth+1),"r":self._build(X[~l],y[~l],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd["leaf"]: return nd["val"]
        return self._p1(x,nd["l"] if x[nd["f"]]<=nd["t"] else nd["r"])
    def predict(self,X): return np.array([self._p1(x,self.tree_) for x in X])
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":self.random_state}

class RandomForestRegressor:
    def __init__(self,n_estimators=10,max_depth=None,max_features=0.7,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(X.shape[1]*self.max_features)); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X[np.ix_(idx,fi)],y[idx]); self.estimators_.append((t,fi))
        return self
    def predict(self,X): return np.mean([t.predict(X[:,fi]) for t,fi in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,"max_features":self.max_features,"random_state":self.random_state}

class GradientBoostingRegressor:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); self.init_=y.mean()
        F=np.full(len(y),self.init_); self.estimators_=[]
        for i in range(self.n_estimators):
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X,y-F); self.estimators_.append(t); F+=self.learning_rate*t.predict(X)
        return self
    def predict(self,X):
        F=np.full(len(X),self.init_)
        for t in self.estimators_: F+=self.learning_rate*t.predict(X)
        return F
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"learning_rate":self.learning_rate,"max_depth":self.max_depth,"random_state":self.random_state}

class BaggingRegressor:
    def __init__(self,estimator=None,n_estimators=10,random_state=None):
        self.estimator=estimator; self.n_estimators=n_estimators; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        base=self.estimator or DecisionTreeRegressor()
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_copy.deepcopy(base); t.fit(X[idx],y[idx]); self.estimators_.append(t)
        return self
    def predict(self,X): return np.mean([t.predict(X) for t in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators}


rng = np.random.default_rng(3)

# ── Dataset ───────────────────────────────────────────────────────────────
N = 200
X, y = make_moons(n_samples=N, noise=0.25, random_state=42)
X    = StandardScaler().fit_transform(X)

K_VALUES = list(range(1, N + 1, 2))   # odd values 1, 3, 5, … N

print("=" * 60)
print("  k-NN: THE CLEAREST BIAS-VARIANCE ILLUSTRATION")
print("=" * 60)
print(f"  Dataset: make_moons  n={N},  noise=0.25")
print()

# ── Cross-validated accuracy at each k ───────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_accs = []
val_accs   = []

for k in K_VALUES:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Training accuracy
    knn.fit(X, y)
    train_accs.append(knn.score(X, y))
    # Val accuracy via CV
    val_accs.append(cross_val_score(knn, X, y, cv=cv,
                                    scoring="accuracy").mean())

# Convert to error (= 1 - accuracy) for U-curve
train_errs = 1 - np.array(train_accs)
val_errs   = 1 - np.array(val_accs)
best_k     = K_VALUES[np.argmin(val_errs)]

print(f"  {'k':>5} │ {'Train err':>10} │ {'Val err':>10} │ Note")
print(f"  {'─'*48}")
highlight_ks = [1, 3, best_k, N//4, N//2, N]
for k, te, ve in zip(K_VALUES, train_errs, val_errs):
    if k in highlight_ks or k == best_k:
        note = ""
        if k == 1:       note = "← max complexity: memorises every point"
        if k == best_k:  note = "← OPTIMAL k (min val error)"
        if k == N:       note = "← predict majority class everywhere"
        print(f"  {k:>5} │ {te:>10.4f} │ {ve:>10.4f} │ {note}")

print()
print(f"  Optimal k = {best_k}")
print(f"  At k=1:  train=0.000 (perfect), val error is high → high variance")
print(f"  At k=N:  train≈val≈majority error → high bias")

# ── Decision boundary plots ───────────────────────────────────────────────
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                     np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
grid   = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    f"k-NN: Bias-Variance Tradeoff Through a Single Hyperparameter k\\n"
    f"(n={N}, optimal k={best_k})",
    fontsize=10, fontweight="bold"
)

# Top row: decision boundaries
ks_to_plot = [1, best_k, N]
labels_plot = [
    f"k=1\\n(max variance, ~0 bias)",
    f"k={best_k}\\n(optimal balance)",
    f"k={N}\\n(max bias, 0 variance)",
]
for ax, k, label in zip(axes[0], ks_to_plot, labels_plot):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    Z = knn.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="RdBu")
    ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu",
               edgecolors="k", s=18, linewidths=0.4)
    tr_err = 1 - knn.score(X, y)
    ax.set_title(f"{label}\\nTrain err={tr_err:.3f}", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

# Bottom-left: error curves vs 1/k (increasing complexity →)
ax4 = axes[1, 0]
inv_k = [1/k for k in K_VALUES]
ax4.plot(inv_k, train_errs, "steelblue", lw=2, label="Training error")
ax4.plot(inv_k, val_errs,   "tomato",    lw=2, label="Validation error")
ax4.axvline(1/best_k, color="gray", linestyle="--",
            label=f"1/k = 1/{best_k}")
ax4.set_xlabel("1/k  (→ higher = more complex)")
ax4.set_ylabel("Classification Error")
ax4.set_title("U-Curve: Error vs Complexity (1/k)", fontsize=9)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Bottom-middle: same curve vs k (so direction is intuitive)
ax5 = axes[1, 1]
ax5.semilogx(K_VALUES, train_errs, "steelblue", lw=2, label="Training error")
ax5.semilogx(K_VALUES, val_errs,   "tomato",    lw=2, label="Validation error")
ax5.axvline(best_k, color="gray", linestyle="--", label=f"Optimal k={best_k}")
ax5.set_xlabel("k (log scale) — higher k = more regularised")
ax5.set_ylabel("Classification Error")
ax5.set_title("Error vs k (log scale)", fontsize=9)
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# Bottom-right: bias-variance intuition text
ax6 = axes[1, 2]
ax6.axis("off")
ax6.text(0.05, 0.95,
    "Bias-Variance in k-NN\\n\\n"
    "k = 1\\n"
    "  Bias ≈ 0  (no smoothing)\\n"
    "  Variance = HIGH\\n"
    "  Memorises every training point\\n\\n"
    "k = optimal\\n"
    "  Bias²  ≈ Variance\\n"
    "  Best generalisation\\n\\n"
    "k = N (predict majority)\\n"
    "  Bias = HIGH (ignores x)\\n"
    "  Variance = 0\\n\\n"
    "The Lesson:\\n"
    "k is a pure bias-variance dial.\\n"
    "Every ML hyperparameter is secretly\\n"
    "doing the same negotiation.",
    transform=ax6.transAxes,
    va="top", ha="left",
    fontsize=9, fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
)

plt.tight_layout()
plt.savefig("knn_bv.png", dpi=120)
print("\\n  Plot saved → knn_bv.png")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · Bagging vs Boosting — Variance and Bias Reduction Compared": {
        "description": (
            "Train a single deep tree (high variance), a Random Forest (bagging), "
            "and a Gradient Boosting machine on the same dataset. "
            "Decompose the bias and variance of each to show that bagging "
            "attacks variance and boosting attacks bias."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt
from itertools import combinations_with_replacement as _cwr

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LinearRegression:
    def __init__(self,fit_intercept=True): self.fit_intercept=fit_intercept
    def fit(self,X,y):
        Xa=np.column_stack([np.ones(len(X)),X]) if self.fit_intercept else X
        w,_,_,_=np.linalg.lstsq(Xa,y,rcond=None)
        if self.fit_intercept: self.intercept_=w[0]; self.coef_=w[1:]
        else: self.intercept_=0.; self.coef_=w
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def score(self,X,y):
        p=self.predict(X); ss_res=np.sum((y-p)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True): return {"fit_intercept":self.fit_intercept}

class Ridge:
    def __init__(self,alpha=1.0,fit_intercept=True):
        self.alpha=alpha; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape
        self.coef_=np.linalg.solve(Xc.T@Xc+self.alpha*np.eye(d),Xc.T@yc)
        self.intercept_=y.mean()-mu@self.coef_ if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"fit_intercept":self.fit_intercept}

class Lasso:
    def __init__(self,alpha=1.0,max_iter=500,fit_intercept=True):
        self.alpha=alpha; self.max_iter=max_iter; self.fit_intercept=fit_intercept
    def fit(self,X,y):
        if self.fit_intercept: mu=X.mean(0); Xc=X-mu; yc=y-y.mean()
        else: Xc=X; yc=y; mu=np.zeros(X.shape[1])
        n,d=Xc.shape; w=np.zeros(d); lam=self.alpha*n
        for _ in range(self.max_iter):
            w_old=w.copy()
            for j in range(d):
                r=yc-Xc@w+Xc[:,j]*w[j]; rho=Xc[:,j]@r; z=np.sum(Xc[:,j]**2)
                w[j]=0. if z<1e-10 else np.sign(rho)*max(abs(rho)-lam,0)/z
            if np.max(np.abs(w-w_old))<1e-6: break
        self.coef_=w
        self.intercept_=y.mean()-mu@w if self.fit_intercept else 0.
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha,"max_iter":self.max_iter}

class StandardScaler:
    def fit(self,X): self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {}

class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True): self.degree=degree; self.include_bias=include_bias
    def _combos(self,n):
        c=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: c.append(())
            else: c.extend(_cwr(range(n),d))
        return c
    def fit(self,X,y=None): self._c=self._combos(X.shape[1]); self.n_output_features_=len(self._c); return self
    def transform(self,X):
        cols=[]
        for combo in self._c:
            if not combo: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {"degree":self.degree,"include_bias":self.include_bias}

class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,s in self.steps[:-1]: Xt=s.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def score(self,X,y):
        yp=self.predict(X); ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-y.mean())**2)
        return 1-ss_res/ss_tot if ss_tot>0 else 0.
    def get_params(self,deep=True):
        p={}
        for nm,s in self.steps: p.update({nm+"__"+k:v for k,v in s.get_params().items()})
        return p

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

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        return np.array([self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int).ravel().tolist()
                         and [np.bincount(self._y[np.argsort(np.sqrt(np.sum((self._X-xi)**2,axis=1)))[:self.n_neighbors]].astype(int)).argmax()]
                         for xi in X]).ravel()
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

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

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        if scoring=="accuracy": scores.append((est.predict(X[te])==y[te]).mean())
        elif scoring=="neg_mean_squared_error": p=est.predict(X[te]); scores.append(-np.mean((y[te]-p)**2))
        else: scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

def learning_curve(estimator,X,y,train_sizes=None,cv=5,scoring="neg_mean_squared_error",n_jobs=None,random_state=None):
    if train_sizes is None: train_sizes=np.linspace(0.1,1.0,5)
    n_tot=len(X); abs_sizes=np.clip(np.array([int(s) if s>1 else int(s*n_tot) for s in train_sizes]),2,n_tot)
    if isinstance(cv,int):
        n=len(y); fs=n//cv; splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    else: splits=list(cv.split(X,y))
    tr_sc=np.zeros((len(abs_sizes),len(splits))); val_sc=np.zeros((len(abs_sizes),len(splits)))
    for si,(tr,te) in enumerate(splits):
        for ti,sz in enumerate(abs_sizes):
            sub=tr[:sz] if sz<=len(tr) else tr; est=_copy.deepcopy(estimator); est.fit(X[sub],y[sub])
            if scoring=="neg_mean_squared_error":
                tr_sc[ti,si]=-np.mean((y[sub]-est.predict(X[sub]))**2)
                val_sc[ti,si]=-np.mean((y[te]-est.predict(X[te]))**2)
            else:
                tr_sc[ti,si]=(est.predict(X[sub])==y[sub]).mean()
                val_sc[ti,si]=(est.predict(X[te])==y[te]).mean()
    return abs_sizes,tr_sc,val_sc

class DecisionTreeRegressor:
    def __init__(self,max_depth=None,min_samples_split=2,random_state=None):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split; self.random_state=random_state
    def _mse(self,y): return np.var(y)*len(y) if len(y)>0 else 0
    def _split(self,X,y):
        best=None; best_g=self._mse(y); n=len(y)
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2; thrs=all_t[np.linspace(0,len(all_t)-1,min(8,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples_split or r.sum()<self.min_samples_split: continue
                g=self._mse(y[l])+self._mse(y[r])
                if g<best_g: best_g=g; best=(f,t)
        return best
    def _build(self,X,y,depth):
        if (self.max_depth is not None and depth>=self.max_depth) or len(y)<=self.min_samples_split:
            return {"leaf":True,"val":y.mean()}
        sp=self._split(X,y)
        if sp is None: return {"leaf":True,"val":y.mean()}
        f,t=sp; l=X[:,f]<=t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[l],y[l],depth+1),"r":self._build(X[~l],y[~l],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd["leaf"]: return nd["val"]
        return self._p1(x,nd["l"] if x[nd["f"]]<=nd["t"] else nd["r"])
    def predict(self,X): return np.array([self._p1(x,self.tree_) for x in X])
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":self.random_state}

class RandomForestRegressor:
    def __init__(self,n_estimators=10,max_depth=None,max_features=0.7,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(X.shape[1]*self.max_features)); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X[np.ix_(idx,fi)],y[idx]); self.estimators_.append((t,fi))
        return self
    def predict(self,X): return np.mean([t.predict(X[:,fi]) for t,fi in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,"max_features":self.max_features,"random_state":self.random_state}

class GradientBoostingRegressor:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth; self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); self.init_=y.mean()
        F=np.full(len(y),self.init_); self.estimators_=[]
        for i in range(self.n_estimators):
            t=DecisionTreeRegressor(max_depth=self.max_depth,random_state=int(rng.integers(0,10**9)))
            t.fit(X,y-F); self.estimators_.append(t); F+=self.learning_rate*t.predict(X)
        return self
    def predict(self,X):
        F=np.full(len(X),self.init_)
        for t in self.estimators_: F+=self.learning_rate*t.predict(X)
        return F
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators,"learning_rate":self.learning_rate,"max_depth":self.max_depth,"random_state":self.random_state}

class BaggingRegressor:
    def __init__(self,estimator=None,n_estimators=10,random_state=None):
        self.estimator=estimator; self.n_estimators=n_estimators; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self.estimators_=[]
        base=self.estimator or DecisionTreeRegressor()
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_copy.deepcopy(base); t.fit(X[idx],y[idx]); self.estimators_.append(t)
        return self
    def predict(self,X): return np.mean([t.predict(X) for t in self.estimators_],axis=0)
    def get_params(self,deep=True): return {"n_estimators":self.n_estimators}


rng = np.random.default_rng(77)

def f_true(x):
    return np.sin(3 * np.pi * x) + 0.5 * x**2 - 0.2

NOISE   = 0.35
N_TRAIN = 60
N_BOOT  = 80
X_TEST  = np.linspace(0, 1, 300)
Y_TRUE  = f_true(X_TEST)

print("=" * 68)
print("  BAGGING vs BOOSTING — BIAS AND VARIANCE COMPARISON")
print("=" * 68)
print(f"  f(x) = sin(3πx) + 0.5x² − 0.2,  noise={NOISE},  n={N_TRAIN}")
print()

# ── Models to compare ─────────────────────────────────────────────────────
models = {
    "Single Deep Tree\\n(max_depth=None)":
        DecisionTreeRegressor(max_depth=None, random_state=0),
    "Random Forest\\n(bagging, 100 trees)":
        RandomForestRegressor(n_estimators=20, max_features=0.7,
                              random_state=0),
    "Gradient Boosting\\n(boosting, 100 rounds)":
        GradientBoostingRegressor(n_estimators=25, learning_rate=0.1,
                                  max_depth=2, random_state=0),
    "Single Shallow Tree\\n(max_depth=2, high bias)":
        DecisionTreeRegressor(max_depth=2, random_state=0),
}

preds_all = {name: np.zeros((N_BOOT, len(X_TEST)))
             for name in models}

for b in range(N_BOOT):
    X_tr = rng.uniform(0, 1, N_TRAIN)
    y_tr = f_true(X_tr) + rng.normal(0, NOISE, N_TRAIN)
    for name, model in models.items():
        m = type(model)(**model.get_params())  # fresh copy
        m.fit(X_tr.reshape(-1, 1), y_tr)
        preds_all[name][b] = m.predict(X_TEST.reshape(-1, 1))

print(f"  {'Model':35s}  {'Bias²':>8}  {'Variance':>10}  "
      f"{'Total':>8}  {'Ratio V/B²':>10}")
print(f"  {'─'*76}")

results = {}
for name, preds in preds_all.items():
    mean_p  = preds.mean(axis=0)
    bias_sq = np.mean((mean_p - Y_TRUE)**2)
    var     = np.mean(preds.var(axis=0))
    total   = bias_sq + var + NOISE**2
    ratio   = var / (bias_sq + 1e-9)
    short   = name.split("\\n")[0]
    print(f"  {short:35s}  {bias_sq:>8.4f}  {var:>10.4f}  "
          f"{total:>8.4f}  {ratio:>10.2f}x")
    results[name] = (bias_sq, var, total, preds.mean(axis=0))

print()
print("  KEY FINDINGS:")
print("  • Single deep tree: very high variance (memorises noise)")
print("  • Random Forest: nearly same bias as single tree, much lower variance")
print("    → bagging targets VARIANCE, leaves bias unchanged")
print("  • Gradient Boosting: low bias (sequential bias correction)")
print("    lower variance than single deep tree via shallow base learners")
print("  • Shallow single tree: low variance but high bias")
print("    → confirms the tradeoff: can't have both with one model")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Bagging vs Boosting — Bias-Variance Profile Comparison\\n"
    "(grey lines = individual bootstrap models, red = ensemble mean)",
    fontsize=10, fontweight="bold"
)
colours = ["#5b9bd5", "#70ad47", "#ed7d31", "#ffc000"]

for ax, (name, preds_mat), colour in zip(
        axes.ravel(), preds_all.items(), colours):
    mean_pred = preds_mat.mean(axis=0)
    # Plot a random subset of bootstrap predictions
    for b in range(min(40, N_BOOT)):
        ax.plot(X_TEST, preds_mat[b], color="silver",
                alpha=0.15, lw=0.7)
    ax.plot(X_TEST, Y_TRUE,    "k-",   lw=2.5, label="True f(x)")
    ax.plot(X_TEST, mean_pred, color=colour, lw=2.2,
            linestyle="--", label="Mean prediction E[f̂]")
    bias_sq, var, total, _ = results[name]
    ax.set_title(f"{name}\\n"
                 f"Bias²={bias_sq:.3f}  Var={var:.3f}  Total={total:.3f}",
                 fontsize=8.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-3, 3)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ensemble_bv.png", dpi=120)
print("\\n  Plot saved → ensemble_bv.png")
''',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


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