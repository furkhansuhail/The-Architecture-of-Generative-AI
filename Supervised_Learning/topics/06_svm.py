"""
Support Vector Machines — Maximum-Margin Classification
========================================================

SVMs approach classification from a fundamentally different angle than logistic
regression. Rather than asking "what is the probability of class 1?", SVMs ask:
"what is the widest possible street we can draw between the two classes?"

This geometric intuition — maximising the margin — leads to a powerful algorithm
that finds the unique optimal decision boundary, not just any boundary that works.

"""

import base64
import os
import re
import textwrap

DISPLAY_NAME = "06 · Support Vector Machines"
ICON         = "⚔️"
SUBTITLE     = "Maximum-margin classifiers — finding the widest street between classes"
THEORY = """

## 03 · Support Vector Machines

SVMs find the **hyperplane** that maximises the margin between two classes. Instead of just
finding any separating boundary, SVM finds the one with the largest possible gap.

### What is a Support Vector Machine?

A Support Vector Machine (SVM) is a binary classification algorithm that finds the
hyperplane separating two classes with the maximum possible margin — the widest "street"
that fits between the two classes with no data points inside it.

This might sound similar to logistic regression, which also finds a linear decision
boundary. But the objectives are fundamentally different:

    Logistic Regression: find the boundary that maximises the likelihood of the data
                         (equivalent to minimising cross-entropy loss)

    SVM:                 find the boundary that maximises the margin — the distance
                         between the boundary and the closest points of each class

The result is that SVMs find a unique, optimal boundary determined entirely by a
small subset of the training data (the support vectors), while logistic regression
is influenced by every training point.

Think of it as building a road between two cities. Logistic regression draws a road
that best fits all the data points on both sides. SVM finds the widest possible road —
it maximises the empty space between the two classes, ignoring distant points and
focusing only on the ones right at the edge.

    Things that exist inside the model (learnable parameters):
        - Weight vector w  — orientation of the decision hyperplane
        - Bias b           — offset of the hyperplane from the origin
        - Support vectors  — the subset of training points that define the boundary

    Things that exist only at setup time (hyperparameters):
        - C               — regularisation (margin width vs. misclassification tradeoff)
        - Kernel type     — linear, RBF, polynomial, sigmoid
        - Kernel params   — γ for RBF, degree d for polynomial

### SVM as Empirical Risk Minimisation (ERM)

SVM fits cleanly into the ERM framework, but with a different loss function than
anything we've seen so far:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Hypothesis class:  H = { f(x) = sign(w·x + b) | w ∈ ℝᵖ }   │
    │                     (sign of a linear function)             │
    │                                                             │
    │  Loss function:     L(y, f(x)) = max(0, 1 − y·(w·x + b))    │
    │                     (hinge loss — zero when correctly       │
    │                      classified with margin ≥ 1)            │
    │                                                             │
    │  Training objective (soft-margin SVM):                      │
    │      min_w  (1/2)||w||²  +  C · Σᵢ max(0, 1 − yᵢ(w·xᵢ+b))   │
    │              ↑ margin term    ↑ hinge loss sum              │
    │                                                             │
    │  Optimiser:  Quadratic Programming (exact) or               │
    │              SGD with hinge loss (approximate)              │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Comparing the ERM formulations across the series:
    Linear Regression:    MSE loss,            no structural penalty
    Logistic Regression:  BCE loss,            L2 penalty (Ridge) optional
    SVM:                  Hinge loss,          L2 penalty (1/2||w||²) built-in

The L2 penalty in SVM is not optional regularisation added on top — it is the
primary objective. Maximising the margin IS minimising ||w||².


### The Inductive Bias of SVMs

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  SVMs encode the belief that:                               │
    │                                                             │
    │  1. MAXIMUM MARGIN — the best boundary is the one farthest  │
    │     from all training points. Wider margin = better         │
    │     generalisation (structural risk minimisation).          │
    │                                                             │
    │  2. SPARSITY — only the support vectors (closest points)    │
    │     matter. All other training points are irrelevant to     │
    │     the solution. The model is defined by a minority of     │
    │     the training data.                                      │
    │                                                             │
    │  3. KERNEL SMOOTHNESS — the RBF kernel encodes the belief   │
    │     that nearby points in input space should have similar   │
    │     labels (localised similarity structure).                │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The maximum-margin inductive bias has a theoretical justification through
Structural Risk Minimisation (Vapnik, 1990s): maximising the margin minimises
an upper bound on the generalisation error, even without assumptions about the
data distribution. This is different from logistic regression, whose guarantees
rely on the Bernoulli likelihood assumption being correct.

---

### Part 1: Hard-Margin SVM — The Geometry

For linearly separable data, find `w, b` such that:
- `wᵀxᵢ + b ≥ +1` for class +1
- `wᵀxᵢ + b ≤ -1` for class -1

The margin width is `2/||w||`. To **maximise** margin, we **minimise** `||w||²`.

Only the points closest to the boundary — the **support vectors** — determine the hyperplane.
All other training points can be removed without changing the solution.


For binary classification with labels y ∈ {−1, +1}, the SVM decision function is:

                        f(x) = sign(w·x + b)

A point is predicted class +1 if w·x + b > 0, and class −1 if w·x + b < 0.

The decision boundary is the hyperplane w·x + b = 0. The margin is the total width
of the gap between the two classes measured perpendicularly to this hyperplane.

For a correctly classified point xᵢ with label yᵢ ∈ {−1, +1}, the functional margin
is yᵢ(w·xᵢ + b). The hard-margin SVM requires this to be at least 1 for all points:

                    yᵢ(w·xᵢ + b) ≥ 1      for all i

Points where yᵢ(w·xᵢ + b) = 1 exactly — those sitting right on the margin edges —
are the support vectors. Everything else is further away and doesn't affect the solution.


### The Margin Width:

The geometric margin (actual perpendicular distance from a point to the hyperplane) is:

                        geometric margin = yᵢ(w·xᵢ + b) / ||w||

For points on the margin edges (functional margin = 1), the geometric distance to the
decision boundary is exactly 1/||w||. The two class margins are on opposite sides, so
the total margin width is:

                            margin = 2/||w||

Maximising the margin means maximising 2/||w||, which is equivalent to minimising ||w||².


    # ======================================================================================= # 
    **Diagram 1 — The Hard-Margin SVM Geometry:**

    HARD-MARGIN SVM: Maximum Margin Hyperplane
    ══════════════════════════════════════════════════════════════

    x₂
    ↑
    │         ●  ●
    │       ●  ●              ● = Class +1
    │      ●                  ○ = Class -1
    │     [●]──────────────── ← positive margin edge (w·x+b = +1)
    │          ← 1/||w||      [·] = support vectors
    │      ────────────────── ← decision boundary (w·x+b = 0)
    │          ← 1/||w||
    │      ────────────────── ← negative margin edge (w·x+b = -1)
    │     [○]
    │   ○   ○  ○
    │  ○  ○
    └──────────────────────────────────────────────→ x₁

    Total margin = 2/||w||   ← we want this as LARGE as possible
    Equivalently: minimise (1/2)||w||²  ← the SVM objective

    KEY INSIGHT: Only the support vectors [●] and [○] matter.
    Remove any other point → the solution doesn't change.
    This is fundamentally different from logistic regression, where
    every training point influences the boundary.

    ──────────────────────────────────────────────────────────────
    Constraint                     Meaning
    ──────────────────────────────────────────────────────────────
    y(w·x + b) ≥ +1                All points correctly classified
                                   with at least unit functional margin
    y(w·x + b) = +1  exactly       Support vectors (on the margin)
    y(w·x + b) > +1                Interior points (irrelevant)
    ──────────────────────────────────────────────────────────────
    # ======================================================================================= # 


### The Hard-Margin Optimisation Problem:

Formally, hard-margin SVM solves a constrained quadratic program:

    Primal:
        minimise    (1/2)||w||²
        subject to  yᵢ(w·xᵢ + b) ≥ 1    for all i = 1,...,n

This is a convex quadratic objective with linear constraints — there is exactly one
global minimum and it can be found efficiently even for millions of dimensions.

**Why (1/2)||w||² instead of ||w||?**

The factor of 1/2 is a calculus convenience: the derivative of (1/2)||w||² is exactly
w (no factor of 2). The square makes the function strictly convex and differentiable
everywhere. The 1/2 factor doesn't change the solution.

**What changes if data is not linearly separable?**

The hard-margin problem becomes infeasible — the constraints cannot all be satisfied
simultaneously. Soft-margin SVM (Part 2) handles this by allowing violations.


### The Dual Problem and Why It Matters:

The primal hard-margin SVM can be rewritten in its Lagrangian dual form:

    Dual:
        maximise    Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ·xⱼ)
        subject to  αᵢ ≥ 0    and    Σᵢ αᵢ yᵢ = 0

Where αᵢ are Lagrange multipliers — one per training example. The KKT conditions
(Karush-Kuhn-Tucker) tell us that:
    - αᵢ = 0      for non-support-vector points (correctly classified with margin > 1)
    - αᵢ > 0      for support vectors (on the margin exactly)

The weight vector is recovered as:

                        w = Σᵢ αᵢ yᵢ xᵢ

Only support vectors (αᵢ > 0) contribute to w. This is the sparsity property.

**The crucial observation:** the dual objective only involves dot products xᵢ·xⱼ
between training examples. This is the doorway to the kernel trick (Part 3).

---
### Part 2: Soft-Margin SVM (C parameter) — Handling Real Data

Real data is never perfectly separable. Introduce slack variables `ξᵢ ≥ 0`:

Minimise: (1/2)||w||² + C Σ ξᵢ

- **Large C** → narrow margin, fewer violations (overfitting risk)
- **Small C** → wider margin, more violations (underfitting risk)


### Why Hard Margins Fail:

Real data is almost never perfectly linearly separable. Even if the classes are
fundamentally separable, noise and outliers will produce points on the wrong side.
Hard-margin SVM refuses to find a solution in this case.

Soft-margin SVM relaxes the constraints by introducing slack variables ξᵢ ≥ 0:

                    yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ       for all i

Where ξᵢ measures how far a point violates the margin:
    ξᵢ = 0:         point correctly classified, outside or on the margin
    0 < ξᵢ ≤ 1:    point inside the margin but still on the correct side
    ξᵢ > 1:         point on the wrong side of the decision boundary (misclassified)


### The Soft-Margin Objective:

    minimise    (1/2)||w||²  +  C · Σᵢ ξᵢ
    subject to  yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ,    ξᵢ ≥ 0    for all i

The parameter C controls the tradeoff between margin width and constraint violations:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Large C:  violations are very costly.                      │
    │            → model prefers narrow margin, fewer violations  │
    │            → fits training data tightly (risk: overfit)     │
    │                                                             │
    │  Small C:  violations are cheap.                            │
    │            → model prefers wide margin, tolerates errors    │
    │            → smoother, simpler boundary (risk: underfit)    │
    │                                                             │
    │  C = ∞:    hard-margin SVM (no violations allowed)          │
    │  C → 0:    maximum margin regardless of errors              │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


    # ======================================================================================= # 
    **Diagram 2 — Effect of C on the Decision Boundary:**

    SMALL C (wide margin, tolerates violations)   LARGE C (narrow margin, few violations)

    x₂ ↑                                          x₂ ↑
       │  ●  ●  ●                                    │  ●  ●  ●
       │    ●                                        │    ●
       │       ●  ← violator (allowed)               │────●──────── tight boundary
       │                                             │
       │ ═══════════════ wide margin                 │        ○
       │                                             │      ○  ○  ●  ← near-violator
       │  ○  ○                                       │  ○  ○
       │    ○                                        │    ○
       └──────────────────→ x₁                       └──────────────────→ x₁

    Larger margin → better generalisation (if data is noisy)
    Smaller margin → better fit (if data is truly separable)
    Optimal C is found by cross-validation.

    SUPPORT VECTOR COUNT:
    Small C → many support vectors (wider boundary influenced by more points)
    Large C → few support vectors (tight boundary, only closest points matter)
    # ======================================================================================= # 


### The Hinge Loss Connection:

The soft-margin SVM objective can be rewritten without slack variables by substituting
ξᵢ = max(0, 1 − yᵢ(w·xᵢ + b)):

    minimise    (1/2)||w||²  +  C · Σᵢ max(0, 1 − yᵢ(w·xᵢ + b))
                ↑                    ↑
                L2 regularisation    sum of hinge losses

This is exactly the ERM formulation: L2 regularisation + hinge loss.


    # ======================================================================================= # 
    **Diagram 3 — Hinge Loss vs Other Losses:**

    LOSS as a function of the functional margin  m = y(w·x + b)

    Loss
    ↑
    4│●
     │ ╲  ← Hinge loss: max(0, 1-m)
    3│  ╲         ← BCE (logistic loss)
     │   ╲      ╲
    2│    ╲    ╲
     │     ╲  ╲
    1│      ╲╲
     │       ●╲        ╲
    0│─────────────────────→ m = y(w·x+b)
            -1  0  1  2  3
                ↑  ↑
                0  1 = margin boundary

    HINGE LOSS: max(0, 1−m)
        m < 0   : wrong side, penalty = 1-m > 1 (grows linearly)
        0 < m < 1: inside margin, penalty = 1-m (still penalised)
        m > 1   : outside margin, penalty = 0  (ignored completely!)

    LOGISTIC LOSS: log(1 + exp(-m))
        Never reaches exactly zero — every point gets some gradient
        Smooth and differentiable everywhere

    KEY DIFFERENCE:
    Hinge loss: points outside the margin contribute ZERO to training
                → only support vectors matter → sparse solution
    Logistic:   every point contributes gradient → dense solution
    # ======================================================================================= # 


### Concrete Soft-Margin Walkthrough:

Let's verify the margin constraints and hinge losses for a simple 1D example.

    Dataset (1D, y ∈ {-1, +1}):
    ──────────────────────────────────────────────────────────────
    x       y       w·x + b     y(w·x+b)    ξ = max(0,1-m)   Type
    ──────────────────────────────────────────────────────────────
    x=3     +1      2.5         2.5          0.0              safe interior
    x=2     +1      1.5         1.5          0.0              safe interior
    x=1     +1      0.5         0.5          0.5              in-margin violator
    x=-2    -1      -1.5        1.5          0.0              safe interior
    x=-3    -1      -2.5        2.5          0.0              safe interior
    x=0     -1      -0.5        0.5          0.5              in-margin violator
    ──────────────────────────────────────────────────────────────
    (Using w = 1.0, b = -0.5 as current weights)

    SVM loss = (1/2)(1.0)² + C × (0 + 0 + 0.5 + 0 + 0 + 0.5)
             = 0.5 + C × 1.0

    With C = 1: total loss = 1.5
    The two in-margin violators (x=1 and x=0) are the only ones contributing
    to the hinge loss term. All other points contribute zero.

---

### Part 3: The Kernel Trick — Non-Linear SVM

SVM computes dot products `xᵢᵀxⱼ`. Replace with a **kernel function** `K(xᵢ, xⱼ)`:

| Kernel       | Formula           | Use Case                   |
|--------------|-------------------|----------------------------|
| Linear       | `xᵀz`             | Already linearly separable |
| RBF/Gaussian | `exp(-γ||x-z||²)` | Most general-purpose       |
| Polynomial   | `(xᵀz + c)^d`     | Image classification       |

The kernel implicitly maps to a higher-dimensional space without computing it explicitly.

Linear SVM can only find linear decision boundaries. Many real classification problems
require curved boundaries. Rather than redesigning the algorithm, the kernel trick
achieves non-linearity through a conceptually elegant manoeuvre:

    Map the data to a higher-dimensional feature space where the classes
    ARE linearly separable, then find the maximum-margin hyperplane there.

The remarkable insight: we never need to compute this mapping explicitly.


### Feature Maps:

A feature map φ: ℝᵖ → ℝᵈ (where d ≫ p or even d = ∞) transforms each input into
a higher-dimensional space. The linear SVM in that space corresponds to a non-linear
boundary in the original space.

Example: 2D input → polynomial feature map:
    φ([x₁, x₂]) = [1, x₁, x₂, x₁², x₁x₂, x₂²]    (degree-2 polynomial features)

After mapping, a linear boundary in 6D corresponds to a quadratic curve in 2D.

This is exactly the feature engineering approach from logistic regression (Part 4) —
but instead of hand-crafting features, the kernel trick makes it automatic and
computationally tractable even when d is infinite.


### The Kernel Function:

The SVM dual objective only involves dot products between training examples:

                Σᵢ αᵢ − (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ · xⱼ)

After feature mapping, the dot products become:

                (xᵢ · xⱼ)  →  (φ(xᵢ) · φ(xⱼ))

A kernel function K(xᵢ, xⱼ) computes this dot product in the feature space directly,
without explicitly computing φ(xᵢ):

                K(xᵢ, xⱼ) = φ(xᵢ) · φ(xⱼ)

This is the kernel trick: replace every dot product in the SVM algorithm with a
kernel evaluation. We get the full power of the high-dimensional feature space at
the computational cost of operating in the original space.


    # ======================================================================================= # 
    **Diagram 4 — The Kernel Trick: 2D → High-D → Linear Boundary:**

    ORIGINAL 2D SPACE:              AFTER FEATURE MAP φ:
    (not linearly separable)        (linearly separable!)

    x₂ ↑                            φ₃ = x₁x₂ ↑
       │  ○  ○                               │           ●●
       │ ○  ○                                │         ●  ●
       │     ●  ●                            │   ─────────────  ← linear boundary!
       │   ●  ●                              │  ○ ○
       └───────────→ x₁                      └──────────────────→ φ₂ = x₁²
    Classes mixed in 2D            Separation emerges in feature space

    THE TRICK:
    We never actually compute φ(x) = [x₁, x₂, x₁², x₁x₂, x₂², ...]
    We only compute K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ) directly

    Example: Polynomial kernel of degree 2:
    K(x, z) = (x·z + 1)²
             = (x₁z₁ + x₂z₂ + 1)²
             = ... expanded ...
             = φ(x)·φ(z)    where φ maps to 6D
    Cost of K(x,z): O(p)    ← just 2 dot products in original space
    Cost of φ(x)·φ(z): O(p²) ← must compute all cross terms
    # ======================================================================================= # 


### Standard Kernels:

**Linear Kernel:**
    K(x, z) = x·z
    Equivalent to standard linear SVM. No feature expansion.
    Use when: data is already linearly separable or p is very large (text).

**Polynomial Kernel (degree d):**
    K(x, z) = (x·z + c)^d
    Implicitly computes all polynomial features up to degree d.
    Use when: features have multiplicative interactions.
    Hyperparameters: d (degree), c (bias term, usually 0 or 1).

**Radial Basis Function / Gaussian Kernel:**
    K(x, z) = exp(−γ||x − z||²)
    Implicitly maps to INFINITE-dimensional feature space.
    Effectively: similarity measure based on distance.
    When γ is small → wide similarity → smooth, simple boundary.
    When γ is large → narrow similarity → complex, jagged boundary.
    Use when: no prior knowledge about feature structure. Most general-purpose.


    # ======================================================================================= # 
    **Diagram 5 — RBF Kernel: The "Infinite Feature Space" Intuition:**

    RBF KERNEL: K(x, z) = exp(−γ||x − z||²)
    ══════════════════════════════════════════════════════════════

    The RBF kernel measures SIMILARITY between two points:
        x and z very close     → ||x-z||² ≈ 0   → K ≈ 1 (similar)
        x and z far apart      → ||x-z||² large → K ≈ 0 (different)

    EFFECT OF γ:
    ──────────────────────────────────────────────────────────────
    Small γ (e.g. 0.01):            Large γ (e.g. 100):
    Wide "similarity bell curve"    Narrow "similarity bell curve"
    K(x,z) > 0 even for distant    K(x,z) ≈ 0 for all but
    points                          very close points

    Decision boundary:              Decision boundary:
    Smooth, global                  Jagged, local — wraps tightly
                                    around each training point
    ← tends to underfit             tends to overfit →

    INTUITIVE PICTURE (RBF boundary in 2D):
    Small γ:                        Large γ:
    x₂ ↑                            x₂ ↑
       │   ●●                          │  (●)(●)
       │ ●● ●●                         │ (●●)(●●)
       │   ──── smooth curve           │    jagged islands
       │  ○○○○                         │ (○)(○)(○)
       └──────→ x₁                     └──────→ x₁

    Correct γ found by cross-validation (grid search or randomised search).
    # ======================================================================================= # 


### Mercer's Theorem — What Makes a Valid Kernel?

Not every function K(x, z) is a valid kernel (i.e., corresponds to a dot product
in some feature space). A function is a valid kernel if and only if the kernel matrix
K (where Kᵢⱼ = K(xᵢ, xⱼ)) is positive semi-definite for any training set.

This is Mercer's theorem. It guarantees:
    1. A valid feature space φ exists (even if infinite-dimensional)
    2. The SVM dual problem remains convex (global minimum exists)
    3. The model is still interpretable as a linear classifier in feature space

All three standard kernels (linear, polynomial, RBF) are valid by Mercer's theorem.
The sigmoid kernel K(x,z) = tanh(αx·z + c) is NOT always valid — it violates Mercer's
condition for some parameter values.

---

### Part 4: Support Vectors — The Defining Points


### What Are Support Vectors?

Support vectors are the training points that lie exactly on the margin boundaries
(or violate them, in soft-margin SVM). They are the only points that matter for
defining the decision boundary.

From the dual formulation: w = Σᵢ αᵢ yᵢ xᵢ, where αᵢ > 0 only for support vectors.

This means:
    - You can discard all non-support-vector training points
    - The solution remains identical
    - Predictions on new points depend only on their similarity to support vectors:
        f(x) = sign(Σᵢ αᵢ yᵢ K(xᵢ, x) + b)


    # ======================================================================================= # 
    **Diagram 6 — Support Vectors Define the Boundary:**

    x₂ ↑
       │     ●  ●  ●                 Support vectors: the points with boxes
       │   ●  ●                      All other points: irrelevant to the solution
       │  ●
       │ [●]──────────────── ← positive margin (w·x+b = +1)
       │      ← MARGIN →
       │ ─────────────────── ← decision boundary (w·x+b = 0)
       │      ← MARGIN →
       │ [○]──────────────── ← negative margin (w·x+b = -1)
       │   ○
       │  ○   ○  ○
       └─────────────────────────────────────────────→ x₁

    [●] and [○] are support vectors. αᵢ > 0 for these points.
    All ● and ○ without brackets: αᵢ = 0, contribute nothing to w.

    ROBUSTNESS PROPERTY:
    Moving a non-support-vector farther away → solution unchanged
    Moving a support vector → boundary shifts

    COMPARE WITH LOGISTIC REGRESSION:
    Logistic: every point contributes gradient → boundary influenced by all points
    SVM:      only support vectors matter → robust to most outliers
    # =======================================================================================# 


### Support Vectors and Generalisation:

The number of support vectors is a measure of model complexity:
    Few SVs:   wide margin, simple boundary, good generalisation
    Many SVs:  narrow margin, complex boundary, risk of overfitting

In kernel SVM, each support vector is a "basis function" in the decision function.
With RBF kernel, many SVs = many local bumps in the decision boundary = overfitting.

The number of support vectors grows with:
    - Smaller C (allows more violations, so more points fall inside the margin)
    - Larger γ (narrower similarity, more points needed to define the boundary)
    - More noise in the data (more violations → more support vectors)

---

### Part 5: SVM vs Logistic Regression — A Deep Comparison

- SVM cares only about support vectors (robust to most outliers)
- Logistic Regression uses all points; outputs calibrated probabilities
- Both are linear in their feature space; kernel SVM is non-linear

The two algorithms solve similar problems but from different perspectives.

    ────────────────────────────────────────────────────────────────────────
    Property               Logistic Regression    SVM
    ────────────────────────────────────────────────────────────────────────
    Objective              Maximise likelihood     Maximise margin
    Loss function          Cross-entropy (BCE)     Hinge loss
    Solution determined by All training points     Support vectors only
    Output                 Calibrated probability  Decision score (uncalib.)
    Probabilistic          Yes (MLE / Bernoulli)   No (geometric)
    Regularisation         L1 or L2 (optional)     L2 built into objective
    Non-linear extension   Feature engineering     Kernel trick
    Optimiser              Gradient descent        Quadratic programming
    Scalability            Very large n, p         Medium n (kernel: slow)
    Sparse solution        No (dense weights)      Yes (sparse support vectors)
    ────────────────────────────────────────────────────────────────────────


### When to Use SVM vs Logistic Regression:

    ┌──────────────────────────────────────────────────────────────┐
    │  Choose SVM when:                                            │
    │    • High-dimensional data with few features (text, genes)   │
    │    • n < ~100k samples (kernel SVM doesn't scale to large n) │
    │    • Non-linear boundaries needed (use RBF kernel)           │
    │    • Calibrated probabilities NOT required                   │
    │    • Robustness to outliers matters                          │
    │                                                              │
    │  Choose Logistic Regression when:                            │
    │    • Calibrated output probabilities needed                  │
    │    • n is very large (SGD scales well)                       │
    │    • Online learning (streaming data)                        │
    │    • Interpretability of weights is important                │
    │    • Feature selection needed (L1 logistic)                  │
    └──────────────────────────────────────────────────────────────┘


### Hinge Loss vs Logistic Loss — A Probabilistic Connection:

Both losses are upper bounds on the 0-1 loss (the true classification error):

    0-1 loss:      L₀₁(m) = 1[m < 0]   (1 if wrong, 0 if right)
    Hinge loss:    L_H(m)  = max(0, 1−m) ≥ L₀₁(m)   always
    Logistic loss: L_L(m)  = log(1+e⁻ᵐ) ≥ L₀₁(m)   always

Minimising either bound drives the 0-1 loss down. The choice determines which
points receive gradient:
    Hinge: zero gradient for m > 1 (correctly classified with margin) → sparse
    Logistic: non-zero gradient everywhere → dense, all points matter

---

### Part 6: From SVM to Deep Learning


### The Historical Arc:

SVMs dominated machine learning from the mid-1990s to around 2012. They had strong
theoretical guarantees (Structural Risk Minimisation), worked well on small/medium
datasets, and with the kernel trick, could learn highly non-linear boundaries.

The 2012 ImageNet competition changed everything: deep neural networks outperformed
kernel SVMs by a massive margin on image recognition. The reason was scale: neural
networks benefited from more data and more compute in a way that kernel SVMs could not.

**Why kernel SVM doesn't scale:**
    - The kernel matrix K is n×n → storing it costs O(n²) memory
    - Training requires solving an n×n quadratic program → O(n³) worst case
    - With n = 1,000,000, this is completely infeasible
    - Neural networks scale as O(n·p·layers) — far more manageable

**What SVM contributed to deep learning:**
    - The hinge loss is used in neural networks (e.g., multi-class SVM loss)
    - The kernel intuition motivated network architecture design
    - RBF networks (radial basis function networks) are direct neural network
      implementations of kernel machines
    - Support vector regression (SVR) introduced ε-insensitive loss, used in
      modern robust regression


### The Connection to Neural Networks:

A neural network can be seen as learning its own feature map φ, while an SVM fixes
φ via the kernel choice and finds the optimal linear boundary in that feature space.

    SVM (kernel):  fix φ (via kernel), optimise linear boundary
    Neural Net:    learn φ (via hidden layers), learn linear boundary at output

Neural networks generalise SVM by making the feature representation itself learnable.
The last layer of any classification neural network is logistic regression (or softmax)
applied to the learned representation — which is equivalent to a linear SVM in that
learned feature space.

The full progression:

    Perceptron (1958)              ← linear, hard 0/1, no probability
          │
          │  (soft boundary, probabilities, MLE)
          ↓
    Logistic Regression (1958+)    ← linear, probabilistic, BCE loss
          │
          │  (maximum margin objective, hinge loss, structural risk minimisation)
          ↓
    SVM (1992–1995)                ← linear or kernel, geometric, margin maximisation
          │
          │  (learn φ rather than fixing it; scale to large data)
          ↓
    Neural Networks (2012+)        ← learn non-linear φ, scale to billions of examples

"""


VISUAL_HTML = ""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Runnable code demonstrations
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Linear SVM from Scratch": {
        "description": "Hinge loss + subgradient descent — building SVM without any library",
        "runnable": True,
        "code": '''
"""
================================================================================
LINEAR SVM FROM SCRATCH — HINGE LOSS + SUBGRADIENT DESCENT
================================================================================

We implement the primal soft-margin SVM objective directly:

    L(w, b) = (1/2)||w||²  +  C · Σᵢ max(0, 1 − yᵢ(w·xᵢ + b))
               ↑ margin term         ↑ hinge loss sum

The hinge loss max(0, 1−m) is not differentiable at m = 1 (the kink).
We use the subgradient:
    ∂/∂w  hinge(m) = 0         if m > 1   (correct, outside margin)
                   = -yᵢxᵢ    if m ≤ 1   (inside margin or wrong side)

This is a simplified SGD-based SVM — the same approach used by liblinear
for large-scale linear SVMs.

================================================================================
"""

import numpy as np


# =============================================================================
# HINGE LOSS FUNCTIONS
# =============================================================================

def hinge_loss(y_true, scores):
    """
    Hinge loss for a batch: max(0, 1 - y * score)

    Args:
        y_true:  labels in {-1, +1}, shape (n,)
        scores:  w·x + b for each example, shape (n,)

    Returns:
        mean hinge loss over the batch
    """
    margins = y_true * scores                     # functional margin mᵢ = yᵢ(w·xᵢ+b)
    losses  = np.maximum(0, 1 - margins)          # hinge: max(0, 1 - m)
    return losses.mean()


def svm_objective(w, b, X, y, C):
    """
    Full SVM primal objective: (1/2)||w||² + C * Σ hinge(yᵢ(w·xᵢ+b))
    """
    scores = X @ w + b
    return 0.5 * np.dot(w, w) + C * hinge_loss(y, scores)


# =============================================================================
# LINEAR SVM CLASS (Primal, Subgradient Descent)
# =============================================================================

class LinearSVM:
    """
    Linear SVM trained with primal subgradient descent.

    Objective:  min_w  (1/2)||w||²  +  C Σ max(0, 1 - yᵢ(w·xᵢ + b))

    Notes:
        - Labels must be in {-1, +1} (NOT {0, 1})
        - This is a soft-margin SVM (C controls the tradeoff)
        - For production, prefer sklearn's LinearSVC (uses liblinear)
          which is more numerically stable and faster
    """

    def __init__(self, C=1.0, learning_rate=0.01, n_iterations=1000):
        """
        Args:
            C (float): Regularisation parameter.
                       Large C → narrow margin, fewer violations
                       Small C → wide margin, tolerates violations
            learning_rate (float): Step size for gradient descent.
            n_iterations (int): Number of gradient steps.
        """
        self.C      = C
        self.lr     = learning_rate
        self.n_iter = n_iterations
        self.w      = None
        self.b      = 0.0
        self.loss_history = []

    def fit(self, X, y, verbose=True):
        """
        Train with subgradient descent on the primal SVM objective.

        The subgradient of the hinge loss at example i:
            if yᵢ(w·xᵢ+b) < 1:  ∂L/∂w += -C * yᵢxᵢ     (in-margin or wrong)
            else:                 ∂L/∂w += 0              (outside margin)

        Plus the L2 term:  ∂L/∂w += w
        """
        n, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0

        print(f"  Initial SVM objective: {svm_objective(self.w, self.b, X, y, self.C):.4f}")

        for iteration in range(self.n_iter):

            # ── Compute functional margins ────────────────────────────────────
            scores  = X @ self.w + self.b              # (n,)
            margins = y * scores                       # functional margin yᵢ(w·xᵢ+b)

            # ── Subgradient of hinge loss ─────────────────────────────────────
            # Mask: which examples are in the margin or misclassified?
            # These are the only ones contributing gradient
            violators = margins < 1                    # shape (n,), boolean

            # Gradient from L2 regularisation term: w
            grad_w = self.w.copy()
            grad_b = 0.0

            # Gradient from hinge loss (only violators contribute)
            if violators.any():
                grad_w -= self.C * (y[violators, None] * X[violators]).mean(axis=0)
                grad_b -= self.C * y[violators].mean()

            # ── Update ────────────────────────────────────────────────────────
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            # ── Track objective ───────────────────────────────────────────────
            obj = svm_objective(self.w, self.b, X, y, self.C)
            self.loss_history.append(obj)

            if verbose and iteration % 200 == 0:
                preds = self.predict(X)
                acc   = np.mean(preds == y)
                n_sv  = violators.sum()
                print(f"  Iter {iteration:>4} | Obj: {obj:.4f} | Acc: {acc:.4f} | "
                      f"Violators: {n_sv}/{n}")

    def decision_function(self, X):
        """Raw score w·x + b (not thresholded)."""
        return X @ self.w + self.b

    def predict(self, X):
        """Predict labels in {-1, +1}."""
        return np.sign(self.decision_function(X)).astype(int)

    def margin_width(self):
        """Geometric margin = 2 / ||w||"""
        norm_w = np.linalg.norm(self.w)
        return 2.0 / norm_w if norm_w > 0 else float("inf")


# =============================================================================
# DATASET: Linearly separable, labels in {-1, +1}
# =============================================================================

np.random.seed(42)
n = 200
# Two well-separated Gaussian clusters
X_pos = np.random.randn(n // 2, 2) + np.array([2.0,  2.0])
X_neg = np.random.randn(n // 2, 2) + np.array([-2.0, -2.0])
X     = np.vstack([X_pos, X_neg])
y     = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])

print("=" * 65)
print("  LINEAR SVM FROM SCRATCH — SUBGRADIENT DESCENT")
print("=" * 65)
print(f"\\n  Dataset: {n} points, 2 features, labels {{-1, +1}}")
print(f"  Two Gaussian clusters: class +1 centred (2,2), class -1 centred (-2,-2)")
print()

model = LinearSVM(C=1.0, learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

print(f"\\n  Final Results:")
print(f"    Weights:      w = [{model.w[0]:.4f}, {model.w[1]:.4f}]")
print(f"    Bias:         b = {model.b:.4f}")
print(f"    ||w||:        {np.linalg.norm(model.w):.4f}")
print(f"    Margin width: {model.margin_width():.4f}  (= 2/||w||)")
print(f"    Train accuracy: {np.mean(model.predict(X) == y):.4f}")


# =============================================================================
# STEP-BY-STEP HINGE LOSS COMPUTATION
# =============================================================================

print("\\n" + "=" * 65)
print("  STEP-BY-STEP: HINGE LOSS AND GRADIENT")
print("=" * 65)

X_tiny = np.array([[2.0,  1.0],
                    [1.5,  0.5],
                    [0.5, -0.5],   # in-margin, correct side
                    [-1.5,-1.0],
                    [-2.0,-2.0]])
y_tiny = np.array([1., 1., 1., -1., -1.])

# Use a simple w, b for illustration
w_demo = np.array([0.8, 0.6])
b_demo = 0.0

scores  = X_tiny @ w_demo + b_demo
margins = y_tiny * scores
losses  = np.maximum(0, 1 - margins)

print(f"""
  Weights: w = {w_demo},  b = {b_demo}
  (||w|| = {np.linalg.norm(w_demo):.4f})

  {'x':>15} {'y':>4} {'score':>8} {'margin':>8} {'hinge loss':>12} {'type':>20}
  {'-'*15} {'-'*4} {'-'*8} {'-'*8} {'-'*12} {'-'*20}""")

for i in range(len(X_tiny)):
    if margins[i] > 1:
        pt_type = "outside margin ✓"
    elif margins[i] > 0:
        pt_type = "inside margin ⚠"
    else:
        pt_type = "WRONG SIDE ✗"
    print(f"  {str(X_tiny[i].tolist()):>15} {int(y_tiny[i]):>4} {scores[i]:>8.3f} "
          f"{margins[i]:>8.3f} {losses[i]:>12.3f}  {pt_type}")

obj = 0.5 * np.dot(w_demo, w_demo) + 1.0 * losses.mean()
print(f"""
  SVM objective = (1/2)||w||²  +  C × mean(hinge losses)
               = (1/2)({np.dot(w_demo,w_demo):.3f}) + 1.0 × {losses.mean():.4f}
               = {0.5*np.dot(w_demo,w_demo):.4f} + {losses.mean():.4f}
               = {obj:.4f}

  Only the in-margin/wrong-side points contribute gradient.
  The 2 outer points (rows 1, 4, 5 by margin > 1) contribute zero gradient.
  This is the sparsity of the hinge loss — support vectors only.
""")


# =============================================================================
# EFFECT OF C ON MARGIN AND SUPPORT VECTORS
# =============================================================================

print("=" * 65)
print("  EFFECT OF C ON MARGIN WIDTH AND SUPPORT VECTORS")
print("=" * 65)
print()
print(f"  {'C':>8} | {'Train Acc':>10} | {'Margin 2/||w||':>16} | {'||w||':>8} | Violators")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*16}-+-{'-'*8}-+-{'-'*12}")

for C_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
    m = LinearSVM(C=C_val, learning_rate=0.005, n_iterations=2000)
    m.fit(X, y, verbose=False)
    acc = np.mean(m.predict(X) == y)
    margin = m.margin_width()
    norm_w = np.linalg.norm(m.w)
    scores = X @ m.w + m.b
    n_violators = np.sum(y * scores < 1)
    print(f"  {C_val:>8.2f} | {acc:>10.4f} | {margin:>16.4f} | {norm_w:>8.4f} | {n_violators:>5}/{n}")

print(f"""
  As C increases:
    → ||w|| grows (narrower margin)
    → margin 2/||w|| shrinks
    → fewer violators (model fits training data more tightly)
    → risk of overfitting increases

  As C decreases:
    → ||w|| shrinks (wider margin)
    → more violators tolerated
    → model is more regularised, better generalisation on noisy data
""")
''',
    },

    "Kernel SVM and the Kernel Trick": {
        "description": "Manual kernel computation, RBF kernel visualisation, and comparison across kernels",
        "runnable": True,
        "code": '''
"""
================================================================================
KERNEL SVM — THE KERNEL TRICK IN ACTION
================================================================================

The kernel trick replaces every dot product xᵢ·xⱼ with K(xᵢ, xⱼ),
implicitly mapping to a higher-dimensional feature space.

Prediction in kernel SVM:
    f(x) = sign( Σᵢ αᵢ yᵢ K(xᵢ, x) + b )

Only support vectors (αᵢ > 0) contribute to the sum.

This script demonstrates:
    1. Manual kernel matrix computation
    2. Why XOR is solvable with RBF but not linear
    3. Kernel comparison across datasets
    4. Effect of γ on RBF decision boundary complexity

================================================================================
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


np.random.seed(42)

# =============================================================================
# PART 1: MANUAL KERNEL COMPUTATIONS
# =============================================================================

print("=" * 65)
print("  PART 1: MANUAL KERNEL COMPUTATIONS")
print("=" * 65)

def linear_kernel(x, z):
    return np.dot(x, z)

def polynomial_kernel(x, z, degree=2, c=1):
    return (np.dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = x - z
    return np.exp(-gamma * np.dot(diff, diff))

# Show kernel values for some point pairs
x1 = np.array([2.0, 1.0])
x2 = np.array([1.0, 2.0])
x3 = np.array([-2.0, -1.0])   # opposite direction

print(f"""
  Points:
    x₁ = {x1}  (class +1)
    x₂ = {x2}  (class +1, similar to x₁)
    x₃ = {x3}  (class -1, opposite)

  {'Kernel':>25} | {'K(x₁,x₂)':>12} | {'K(x₁,x₃)':>12} | Interpretation
  {'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*30}""")

pairs = [
    ("Linear",                   linear_kernel(x1,x2),            linear_kernel(x1,x3)),
    ("Polynomial (d=2)",         polynomial_kernel(x1,x2,2),      polynomial_kernel(x1,x3,2)),
    ("Polynomial (d=3)",         polynomial_kernel(x1,x2,3),      polynomial_kernel(x1,x3,3)),
    ("RBF (γ=0.1)",              rbf_kernel(x1,x2,0.1),           rbf_kernel(x1,x3,0.1)),
    ("RBF (γ=1.0)",              rbf_kernel(x1,x2,1.0),           rbf_kernel(x1,x3,1.0)),
    ("RBF (γ=10.0)",             rbf_kernel(x1,x2,10.0),          rbf_kernel(x1,x3,10.0)),
]

for name, k12, k13 in pairs:
    interp = "same-class > diff-class ✓" if k12 > k13 else "inverted ✗"
    print(f"  {name:>25} | {k12:>12.4f} | {k13:>12.4f} | {interp}")

print(f"""
  Key observation: For RBF with large γ, K(x₁,x₂) ≈ 0 even though they're
  in the same class — they're not close enough in Euclidean distance.
  This is why large γ creates local, complex boundaries.
""")


# =============================================================================
# PART 2: XOR PROBLEM — LINEAR FAILS, RBF SUCCEEDS
# =============================================================================

print("=" * 65)
print("  PART 2: XOR — WHY KERNEL MATTERS")
print("=" * 65)

# XOR dataset: class 1 if exactly one of x₁,x₂ is positive
n_xor = 400
X_xor = np.random.randn(n_xor, 2)
y_xor = ((X_xor[:, 0] > 0) ^ (X_xor[:, 1] > 0)).astype(int)
y_xor_svm = 2 * y_xor - 1   # convert to {-1, +1}

X_tr, X_te, y_tr, y_te = train_test_split(X_xor, y_xor_svm, test_size=0.25)

print(f"""
  XOR classification: class determined by sign(x₁) XOR sign(x₂)
    Class +1: x₁>0,x₂<0 OR x₁<0,x₂>0  (different signs)
    Class -1: x₁>0,x₂>0 OR x₁<0,x₂<0  (same signs)

  No linear boundary separates these classes.
""")

print(f"  {'Kernel':>20} | {'C':>6} | {'γ':>8} | {'Test Acc':>10} | {'# SVs':>8}")
print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")

configs = [
    ("linear",  1.0,  None,   "N/A"),
    ("rbf",     1.0,  0.1,    "0.1"),
    ("rbf",     1.0,  1.0,    "1.0"),
    ("rbf",     1.0,  10.0,   "10.0"),
    ("poly",    1.0,  None,   "N/A"),
]

for kernel, C, gamma, gamma_str in configs:
    clf = SVC(kernel=kernel, C=C, gamma=gamma if gamma else "scale")
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    n_sv = clf.n_support_.sum()
    print(f"  {kernel:>20} | {C:>6.1f} | {gamma_str:>8} | {acc:>10.4f} | {n_sv:>8}")

print(f"""
  Linear SVM: ~50% accuracy (random) — cannot separate XOR linearly.
  RBF SVM:    high accuracy — kernel maps to space where XOR is separable.

  The RBF kernel implicitly maps each point xᵢ to an infinite-dimensional
  feature space. In that space, the XOR boundary becomes a hyperplane.
""")


# =============================================================================
# PART 3: KERNEL COMPARISON ACROSS DATASETS
# =============================================================================

print("=" * 65)
print("  PART 3: KERNEL COMPARISON — THREE DATASETS")
print("=" * 65)

datasets = {
    "Circles (make_circles)":  make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42),
    "Moons (make_moons)":      make_moons(n_samples=300, noise=0.15, random_state=42),
    "XOR":                     (X_xor, y_xor),
}

kernels = {
    "linear":  SVC(kernel="linear",   C=1.0),
    "rbf":     SVC(kernel="rbf",      C=1.0, gamma="scale"),
    "poly(2)": SVC(kernel="poly",     C=1.0, degree=2),
    "poly(3)": SVC(kernel="poly",     C=1.0, degree=3),
}

print()
for ds_name, (X_ds, y_ds) in datasets.items():
    # Normalise labels to {-1, +1} if needed
    if set(np.unique(y_ds)) == {0, 1}:
        y_ds_svm = 2 * y_ds - 1
    else:
        y_ds_svm = y_ds

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ds)
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_ds_svm, test_size=0.25, random_state=42)

    print(f"  Dataset: {ds_name}")
    print(f"  {'Kernel':>12} | {'Train Acc':>10} | {'Test Acc':>10} | {'# SVs':>8} | {'% SVs':>8}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for k_name, clf in kernels.items():
        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(Xtr, ytr)
        train_acc = accuracy_score(ytr, clf_copy.predict(Xtr))
        test_acc  = accuracy_score(yte, clf_copy.predict(Xte))
        n_sv      = clf_copy.n_support_.sum()
        pct_sv    = 100 * n_sv / len(Xtr)
        print(f"  {k_name:>12} | {train_acc:>10.4f} | {test_acc:>10.4f} | {n_sv:>8} | {pct_sv:>7.1f}%")
    print()


# =============================================================================
# PART 4: KERNEL MATRIX (GRAM MATRIX) — WHAT THE SVM ACTUALLY USES
# =============================================================================

print("=" * 65)
print("  PART 4: THE KERNEL MATRIX (GRAM MATRIX)")
print("=" * 65)

print("""
  The SVM dual problem operates entirely on the n×n kernel matrix K:
      Kᵢⱼ = K(xᵢ, xⱼ)

  K encodes the pairwise similarities between ALL training examples.
  A high Kᵢⱼ means xᵢ and xⱼ are "similar" in the kernel's feature space.

  Below: kernel matrix for 5 toy points.
""")

X5 = np.array([[2.0, 0.0],
               [1.5, 0.5],
               [0.0, 0.0],   # origin
               [-1.5,-0.5],
               [-2.0, 0.0]])
y5 = np.array([1, 1, 0, -1, -1])

print("  Points:", X5.tolist())
print()

for kernel_name, gamma in [("Linear", None), ("RBF (γ=0.5)", 0.5), ("RBF (γ=5.0)", 5.0)]:
    print(f"  {kernel_name} Kernel Matrix:")
    n5 = len(X5)
    K = np.zeros((n5, n5))
    for i in range(n5):
        for j in range(n5):
            if gamma is None:
                K[i,j] = np.dot(X5[i], X5[j])
            else:
                diff = X5[i] - X5[j]
                K[i,j] = np.exp(-gamma * np.dot(diff, diff))
    print("  " + " ".join([f"{'x'+str(i+1):>8}" for i in range(n5)]))
    for i in range(n5):
        row = "  " + " ".join([f"{K[i,j]:>8.3f}" for j in range(n5)])
        print(row)
    print()
''',
    },

    "Support Vectors and Margin Analysis": {
        "description": "Identifying support vectors, margin width, and the role of C",
        "runnable": True,
        "code": '''
"""
================================================================================
SUPPORT VECTORS — IDENTIFICATION, ROLE, AND C SENSITIVITY
================================================================================

This script demonstrates:
    1. How to identify and inspect support vectors
    2. Why only support vectors determine the boundary
    3. How adding/removing non-SVs doesn't change the solution
    4. How C controls the number of support vectors and margin

================================================================================
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)


# =============================================================================
# DATASET
# =============================================================================

X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.5, random_state=42
)
y_svm = 2 * y - 1   # convert to {-1, +1}

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("=" * 65)
print("  SUPPORT VECTOR ANALYSIS")
print("=" * 65)


# =============================================================================
# PART 1: INSPECT SUPPORT VECTORS
# =============================================================================

print("\\n  PART 1: IDENTIFYING SUPPORT VECTORS")
print("  " + "-" * 50)

clf = SVC(kernel="linear", C=1.0)
clf.fit(X, y_svm)

n_sv_total = clf.n_support_.sum()
sv_indices = clf.support_             # indices into training data
svs        = clf.support_vectors_     # the actual support vector coordinates
dual_coefs = clf.dual_coef_[0]        # αᵢ * yᵢ for each SV

print(f"""
  Total training points: {len(X)}
  Support vectors:       {n_sv_total}  ({100*n_sv_total/len(X):.1f}% of training data)
  SVs per class:         {clf.n_support_}  (class -1: {clf.n_support_[0]}, class +1: {clf.n_support_[1]})

  Learned weight vector: w = [{clf.coef_[0,0]:.4f}, {clf.coef_[0,1]:.4f}]
  Bias:                  b = {clf.intercept_[0]:.4f}
  Margin width:          2/||w|| = {2/np.linalg.norm(clf.coef_[0]):.4f}

  First 5 support vectors:
  {'Index':>8} | {'x₁':>8} | {'x₂':>8} | {'α·y':>10} | {'y(w·x+b)':>12} | Label
  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}""")

for i in range(min(5, n_sv_total)):
    sv    = svs[i]
    score = np.dot(clf.coef_[0], sv) + clf.intercept_[0]
    label = np.sign(dual_coefs[i])
    margin = label * score
    print(f"  {sv_indices[i]:>8} | {sv[0]:>8.4f} | {sv[1]:>8.4f} | "
          f"{dual_coefs[i]:>10.4f} | {margin:>12.4f} | {int(label):>8}")

print(f"""
  Note: y(w·x+b) ≈ 1 for all support vectors (they sit on the margin).
  Non-support vectors would have y(w·x+b) > 1 (farther from the boundary).
""")


# =============================================================================
# PART 2: REMOVING NON-SVs DOESN'T CHANGE THE SOLUTION
# =============================================================================

print("  PART 2: REMOVING NON-SVs — SOLUTION IS UNCHANGED")
print("  " + "-" * 50)

# Train on full data
clf_full = SVC(kernel="linear", C=1.0)
clf_full.fit(X, y_svm)

# Train only on support vectors
X_sv_only = clf_full.support_vectors_
y_sv_only = (np.sign(clf_full.dual_coef_[0])).astype(int)

clf_sv = SVC(kernel="linear", C=1.0)
clf_sv.fit(X_sv_only, y_sv_only)

print(f"""
  Full training set ({len(X)} points):
    w = [{clf_full.coef_[0,0]:.4f}, {clf_full.coef_[0,1]:.4f}]
    b = {clf_full.intercept_[0]:.4f}
    Accuracy on full data: {(clf_full.predict(X) == y_svm).mean():.4f}

  Only support vectors ({n_sv_total} points):
    w = [{clf_sv.coef_[0,0]:.4f}, {clf_sv.coef_[0,1]:.4f}]
    b = {clf_sv.intercept_[0]:.4f}
    Accuracy on full data: {(clf_sv.predict(X) == y_svm).mean():.4f}

  The two solutions are nearly identical — the {len(X) - n_sv_total} non-support
  vectors carry zero information about the decision boundary.
  This is the sparsity property of SVMs.
""")


# =============================================================================
# PART 3: C vs NUMBER OF SUPPORT VECTORS vs GENERALISATION
# =============================================================================

print("  PART 3: C, SUPPORT VECTORS, AND GENERALISATION GAP")
print("  " + "-" * 50)

X_tr, X_te, y_tr, y_te = train_test_split(X, y_svm, test_size=0.25, random_state=42)

print(f"""
  {'C':>8} | {'# SVs':>8} | {'% SVs':>8} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gen Gap':>10}
  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}""")

for C_val in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    clf_c = SVC(kernel="linear", C=C_val)
    clf_c.fit(X_tr, y_tr)
    train_acc = (clf_c.predict(X_tr) == y_tr).mean()
    test_acc  = (clf_c.predict(X_te) == y_te).mean()
    n_sv      = clf_c.n_support_.sum()
    pct_sv    = 100 * n_sv / len(X_tr)
    gen_gap   = train_acc - test_acc
    print(f"  {C_val:>8.3f} | {n_sv:>8} | {pct_sv:>7.1f}% | {train_acc:>10.4f} | "
          f"{test_acc:>10.4f} | {gen_gap:>10.4f}")

print(f"""
  As C increases:
    → Fewer SVs (tighter fit, model ignores more points)
    → Higher training accuracy (fits training data harder)
    → Test accuracy peaks then drops (overfitting)
    → Generalisation gap grows

  The optimal C is found by cross-validation (not by inspecting training accuracy).
  A wider margin (small C) usually generalises better when data is noisy.
""")


# =============================================================================
# PART 4: ADDING AN OUTLIER — SVM vs LOGISTIC REGRESSION ROBUSTNESS
# =============================================================================

print("  PART 4: OUTLIER ROBUSTNESS — SVM vs LOGISTIC REGRESSION")
print("  " + "-" * 50)

from sklearn.linear_model import LogisticRegression

# Clean dataset
X_clean = np.vstack([
    np.random.randn(50, 2) + [2, 2],
    np.random.randn(50, 2) + [-2, -2]
])
y_clean = np.concatenate([np.ones(50), -np.ones(50)])
y_lr    = np.concatenate([np.ones(50), np.zeros(50)])

# Add a single extreme outlier to the positive class
outlier = np.array([[10.0, 10.0]])

X_with_outlier     = np.vstack([X_clean, outlier])
y_with_outlier_svm = np.append(y_clean, 1.0)
y_with_outlier_lr  = np.append(y_lr, 1.0)

# Fit SVM (C=1) and LR on clean and outlier data
svm_clean   = SVC(kernel="linear", C=1.0).fit(X_clean, y_clean)
svm_outlier = SVC(kernel="linear", C=1.0).fit(X_with_outlier, y_with_outlier_svm)
lr_clean    = LogisticRegression(C=1.0).fit(X_clean, y_lr)
lr_outlier  = LogisticRegression(C=1.0).fit(X_with_outlier, y_with_outlier_lr)

print(f"""
  Adding a single extreme outlier at [10, 10]:

  SVM (C=1.0):
    Clean data:         w=[{svm_clean.coef_[0,0]:.4f}, {svm_clean.coef_[0,1]:.4f}],  b={svm_clean.intercept_[0]:.4f}
    With outlier:       w=[{svm_outlier.coef_[0,0]:.4f}, {svm_outlier.coef_[0,1]:.4f}],  b={svm_outlier.intercept_[0]:.4f}
    Weight change ||Δw||: {np.linalg.norm(svm_outlier.coef_[0]-svm_clean.coef_[0]):.4f}

  Logistic Regression (C=1.0):
    Clean data:         w=[{lr_clean.coef_[0,0]:.4f}, {lr_clean.coef_[0,1]:.4f}],  b={lr_clean.intercept_[0]:.4f}
    With outlier:       w=[{lr_outlier.coef_[0,0]:.4f}, {lr_outlier.coef_[0,1]:.4f}],  b={lr_outlier.intercept_[0]:.4f}
    Weight change ||Δw||: {np.linalg.norm(lr_outlier.coef_[0]-lr_clean.coef_[0]):.4f}

  The outlier at [10,10] is far from the margin boundary — it's already correctly
  classified with a large margin. SVM ignores it (αᵢ = 0 for this point).
  Logistic regression assigns it a non-zero gradient weight, shifting the boundary.

  SVM is robust to outliers that are far from the decision boundary.
  Logistic regression is influenced by every point, including extreme ones.
""")
''',
    },

    "SVM vs Logistic Regression — Full Comparison": {
        "description": "Side-by-side training, probability calibration, scalability, and when to use each",
        "runnable": True,
        "code": '''
"""
================================================================================
SVM vs LOGISTIC REGRESSION — COMPREHENSIVE COMPARISON
================================================================================

This script systematically compares SVM and Logistic Regression across:
    1. Accuracy and decision boundary on various datasets
    2. Probability calibration (LR gives calibrated probs, SVM does not)
    3. Scalability with n and p
    4. Sensitivity to C (LR) vs C (SVM) — the parameter means different things!

================================================================================
"""

import numpy as np
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

np.random.seed(42)


# =============================================================================
# PART 1: ACCURACY COMPARISON ACROSS DATASET TYPES
# =============================================================================

print("=" * 70)
print("  PART 1: ACCURACY ACROSS DATASET TYPES")
print("=" * 70)

datasets = {
    "Linear (easy)":      make_classification(200, 2, n_informative=2, n_redundant=0, class_sep=2.0, random_state=42),
    "Linear (noisy)":     make_classification(200, 2, n_informative=2, n_redundant=0, class_sep=0.8, random_state=42),
    "Circles (nonlinear)": make_circles(200, noise=0.1, factor=0.5, random_state=42),
    "High-dim (p=50)":    make_classification(200, 50, n_informative=10, random_state=42),
}

lr   = LogisticRegression(C=1.0, max_iter=1000)
svm_l = SVC(kernel="linear", C=1.0, probability=False)
svm_r = SVC(kernel="rbf",    C=1.0, gamma="scale", probability=False)

print(f"\\n  {'Dataset':>22} | {'LR Acc':>8} | {'SVM-Lin':>8} | {'SVM-RBF':>8}")
print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

for ds_name, (X, y) in datasets.items():
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_sc, y, test_size=0.25, random_state=42)

    lr_acc   = cross_val_score(lr,    Xtr, ytr, cv=5).mean()
    sml_acc  = cross_val_score(svm_l, Xtr, ytr, cv=5).mean()
    smr_acc  = cross_val_score(svm_r, Xtr, ytr, cv=5).mean()

    print(f"  {ds_name:>22} | {lr_acc:>8.4f} | {sml_acc:>8.4f} | {smr_acc:>8.4f}")

print(f"""
  Key takeaways:
    Linear data:   LR ≈ SVM-Linear (both are linear classifiers)
    Noisy data:    SVM-RBF can be worse (overfits noise); LR more robust with C
    Non-linear:    SVM-RBF wins (kernel maps to separable space)
    High-dim:      LR often competitive (fast, scales well, good with L1/L2)
""")


# =============================================================================
# PART 2: PROBABILITY CALIBRATION
# =============================================================================

print("=" * 70)
print("  PART 2: PROBABILITY CALIBRATION — LR vs SVM")
print("=" * 70)

print("""
  Logistic Regression directly outputs P(y=1|x) via the sigmoid.
  It is inherently calibrated: P=0.8 means the model is right 80% of the time.

  SVM does NOT output probabilities by default. sklearn's SVC(probability=True)
  uses Platt scaling (fitting a sigmoid to the decision function values) — this
  is a post-hoc calibration, not inherent to the SVM objective.
""")

X_cal, y_cal = make_classification(500, 2, n_informative=2, n_redundant=0,
                                   class_sep=1.0, random_state=42)
scaler = StandardScaler()
X_cal  = scaler.fit_transform(X_cal)
Xtr, Xte, ytr, yte = train_test_split(X_cal, y_cal, test_size=0.3)

lr_cal  = LogisticRegression(C=1.0).fit(Xtr, ytr)
svm_cal = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True).fit(Xtr, ytr)

lr_probs  = lr_cal.predict_proba(Xte)[:, 1]
svm_probs = svm_cal.predict_proba(Xte)[:, 1]

print(f"  Log-Loss (lower = better calibrated):")
print(f"    Logistic Regression: {log_loss(yte, lr_probs):.4f}")
print(f"    SVM (Platt scaling): {log_loss(yte, svm_probs):.4f}")

# Show calibration per bin
print(f"\\n  Calibration check — predicted prob bin vs actual accuracy:")
print(f"  {'Bin':>12} | {'LR: pred prob':>14} | {'LR: actual acc':>14} | "
      f"{'SVM: pred prob':>15} | {'SVM: actual acc':>15}")
print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*15}-+-{'-'*15}")

bins = [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]
for lo, hi in bins:
    lr_mask  = (lr_probs  >= lo) & (lr_probs  < hi)
    svm_mask = (svm_probs >= lo) & (svm_probs < hi)
    if lr_mask.sum() > 0 and svm_mask.sum() > 0:
        lr_pred_p   = lr_probs[lr_mask].mean()
        lr_actual   = yte[lr_mask].mean()
        svm_pred_p  = svm_probs[svm_mask].mean()
        svm_actual  = yte[svm_mask].mean()
        print(f"  [{lo:.1f} – {hi:.1f})    | {lr_pred_p:>14.3f} | {lr_actual:>14.3f} | "
              f"{svm_pred_p:>15.3f} | {svm_actual:>15.3f}")

print(f"""
  LR: predicted probability ≈ actual accuracy (well-calibrated by design)
  SVM: calibration depends on Platt scaling quality (less reliable)

  When you NEED probabilities (medical risk scores, uncertainty estimates):
    → Use Logistic Regression
  When you only need the class label:
    → Either works; SVM may be more robust
""")


# =============================================================================
# PART 3: SCALABILITY — n and p
# =============================================================================

print("=" * 70)
print("  PART 3: SCALABILITY — TIMING COMPARISON")
print("=" * 70)

print(f"""
  Linear SVM scales as O(n·p) with liblinear/SGD solvers.
  Kernel SVM scales as O(n²–n³) — the kernel matrix is n×n.
  Logistic Regression scales as O(n·p) with gradient descent.

  Timing comparison (LinearSVC vs LogisticRegression):
""")

print(f"  {'n (samples)':>14} | {'p (features)':>14} | {'LinearSVC (ms)':>16} | {'LogReg (ms)':>14}")
print(f"  {'-'*14}-+-{'-'*14}-+-{'-'*16}-+-{'-'*14}")

for n_samples, n_features in [(100, 10), (1000, 10), (10000, 10), (1000, 100), (1000, 1000)]:
    X_time = np.random.randn(n_samples, n_features)
    y_time = (X_time[:, 0] + X_time[:, 1] > 0).astype(int)

    t0 = time.perf_counter()
    LinearSVC(C=1.0, max_iter=1000).fit(X_time, y_time)
    t_svm = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    LogisticRegression(C=1.0, max_iter=1000, solver="saga").fit(X_time, y_time)
    t_lr = (time.perf_counter() - t0) * 1000

    print(f"  {n_samples:>14} | {n_features:>14} | {t_svm:>14.1f}ms | {t_lr:>12.1f}ms")

print(f"""
  Both scale similarly for linear classifiers.
  The real difference appears with kernel SVM (not shown — kernel matrix O(n²)):
    n=100:    Kernel SVM fast
    n=10,000: Kernel SVM very slow
    n=1,000,000: Kernel SVM infeasible; use neural networks instead
""")


# =============================================================================
# PART 4: THE C PARAMETER — SAME NAME, DIFFERENT MEANING
# =============================================================================

print("=" * 70)
print("  PART 4: C PARAMETER — SAME NAME, OPPOSITE MEANING!")
print("=" * 70)

print(f"""
  IMPORTANT: C means different things in SVM vs Logistic Regression.

  SVM (sklearn):
    Objective: (1/2)||w||²  + C × hinge_loss
    C controls: how much to penalise VIOLATIONS
    Large C → narrow margin, penalise violations heavily (may overfit)
    Small C → wide margin, tolerate violations (regularised)
    C ↑ = LESS regularisation

  Logistic Regression (sklearn):
    Objective: C × BCE  + (1/2)||w||²    (sklearn uses C = 1/λ convention)
    Large C → less L2 penalty, fits data tightly (may overfit)
    Small C → more L2 penalty, regularised
    C ↑ = LESS regularisation   (same direction as SVM)

  BUT the loss scale is different, so optimal C values are not comparable
  between the two algorithms. Always tune C separately for each.

  Summary:
  {'C value':>12} | {'SVM interpretation':>30} | {'LR interpretation':>30}
  {'-'*12}-+-{'-'*30}-+-{'-'*30}
  {'C=0.001':>12} | {'very wide margin (heavy reg.)':>30} | {'heavy L2 regularisation':>30}
  {'C=1.0':>12} | {'balanced (sklearn default)':>30} | {'balanced (default)':>30}
  {'C=1000':>12} | {'narrow margin, hard constraints':>30} | {'almost no regularisation':>30}

  In both cases: larger C = less regularisation, smaller C = more regularisation.
""")
''',
    },

    # ── Trigger: Logistic Regression ─────────────────────────────────
    "▶ Run: SVM Classification": {
        "description": (
            "Runs 02_SVM_Classification.py from the Implementation folder. "
            "SVM for Classification "
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
    / "02_SVM_Classification.py"
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
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Standalone demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 65)
    print("  SUPPORT VECTOR MACHINES: MAXIMUM-MARGIN CLASSIFICATION")
    print("=" * 65)
    print("""
  This script demonstrates SVM from the ground up.

  Key Concepts:
    • SVM = ERM with hinge loss + L2 regularisation (margin maximisation)
    • Hard margin: perfect separation, minimise ||w||²
    • Soft margin: slack variables + C parameter for tradeoff
    • Hinge loss: zero for correctly classified points outside margin
    • Kernel trick: implicit high-dim feature map via K(xᵢ, xⱼ)
    • Support vectors: the only training points that matter
    """)

    np.random.seed(42)

    # ─── Dataset: two well-separated classes ─────────────────────────────────
    n_per_class = 30
    X_pos = np.random.randn(n_per_class, 2) + np.array([2.0, 2.0])
    X_neg = np.random.randn(n_per_class, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y_svm = np.concatenate([np.ones(n_per_class), -np.ones(n_per_class)])
    y_bin = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    print("=" * 65)
    print("  DATASET: Two Gaussian clusters")
    print("=" * 65)
    print(f"  n={len(X)} points, 2 features, labels {{-1, +1}}")
    print(f"  Class +1 centred at (2, 2),  Class -1 centred at (-2, -2)")

    # ─── Fit linear SVM ───────────────────────────────────────────────────────
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_sc, y_svm)

    w = clf.coef_[0]
    b = clf.intercept_[0]
    norm_w = np.linalg.norm(w)
    margin = 2.0 / norm_w

    print(f"\n  Linear SVM solution:")
    print(f"    w = [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"    b = {b:.4f}")
    print(f"    ||w|| = {norm_w:.4f}")
    print(f"    Margin = 2/||w|| = {margin:.4f}")
    print(f"    Train accuracy: {(clf.predict(X_sc) == y_svm).mean():.4f}")

    # ─── Support vector details ───────────────────────────────────────────────
    n_sv = clf.n_support_.sum()
    print(f"\n  Support vectors: {n_sv} out of {len(X)} training points")
    print(f"  (class -1: {clf.n_support_[0]},  class +1: {clf.n_support_[1]})")

    print(f"\n  Verifying support vectors sit on the margin (y(w·x+b) ≈ 1):")
    for i, sv in enumerate(clf.support_vectors_[:4]):
        score = np.dot(w, sv) + b
        label = np.sign(clf.dual_coef_[0][i])
        margin_val = label * score
        print(f"    SV {i + 1}: score={score:.4f},  label={int(label):+},  y·score={margin_val:.4f} ≈ 1")

    # ─── Hinge loss calculation ───────────────────────────────────────────────
    scores_all = X_sc @ w + b
    margins_all = y_svm * scores_all
    hinge_losses = np.maximum(0, 1 - margins_all)
    svm_obj = 0.5 * np.dot(w, w) + 1.0 * hinge_losses.sum()

    n_sv_active = (hinge_losses > 0).sum()
    n_zero_grad = (hinge_losses == 0).sum()

    print(f"\n  Hinge loss analysis:")
    print(f"    Points with hinge loss > 0: {n_sv_active}  (contribute gradient)")
    print(f"    Points with hinge loss = 0: {n_zero_grad}  (no gradient — outside margin)")
    print(f"    SVM objective (C=1): {svm_obj:.4f}")

    # ─── Kernel comparison ────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  KERNEL COMPARISON ON THIS DATASET")
    print(f"{'=' * 65}")
    print(f"\n  {'Kernel':>16} | {'Accuracy':>10} | {'# SVs':>8} | {'Margin (linear)':>16}")
    print(f"  {'-' * 16}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 16}")

    for kernel_name, kwargs in [
        ("linear", {"kernel": "linear", "C": 1.0}),
        ("rbf", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
        ("poly(d=2)", {"kernel": "poly", "C": 1.0, "degree": 2}),
        ("poly(d=3)", {"kernel": "poly", "C": 1.0, "degree": 3}),
    ]:
        clf_k = SVC(**kwargs).fit(X_sc, y_svm)
        acc = (clf_k.predict(X_sc) == y_svm).mean()
        n_sv = clf_k.n_support_.sum()
        if kernel_name == "linear":
            m = f"{2 / np.linalg.norm(clf_k.coef_[0]):.4f}"
        else:
            m = "(not applicable)"
        print(f"  {kernel_name:>16} | {acc:>10.4f} | {n_sv:>8} | {m:>16}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1. SVM = ERM with hinge loss + L2 regularisation (margin maximisation)
  2. Hard-margin: perfect separation, minimise ||w||²
  3. Soft-margin: add C × Σξᵢ; large C = narrow margin, small C = wide margin
  4. Hinge loss is zero outside the margin — only support vectors matter
  5. Kernel trick: K(xᵢ, xⱼ) implicitly maps to high-dimensional space
  6. RBF kernel = infinite-dimensional feature space (most general-purpose)
  7. γ controls RBF bandwidth: large γ = complex/local, small γ = smooth/global
  8. SVM is robust to outliers far from the boundary (αᵢ = 0 for those points)
  9. SVM doesn't scale to large n with kernels (O(n²–n³)); use neural nets instead
  10. Every neuron in a neural net is logistic regression in a learned feature space
      — the same idea as kernel SVM but with a learned rather than fixed φ
    """)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    return {
        "theory": THEORY,
        "theory_raw": THEORY,
        "operations": OPERATIONS,
        "interactive_components": [],
    }


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
        from Supervised_Learning.visuals.svm_visual import (   # ← match your exact folder casing
            SVM_VISUAL_HTML,
            SVM_VISUAL_HEIGHT,
        )
        visual_html   = SVM_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = SVM_VISUAL_HEIGHT
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
        "complexity":    None, # COMPLEXITY,
        "operations":    OPERATIONS,
    }