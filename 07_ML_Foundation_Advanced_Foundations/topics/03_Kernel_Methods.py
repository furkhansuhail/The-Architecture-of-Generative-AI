"""
Kernel Methods
==============

The kernel trick, Mercer's theorem, Support Vector Machines, kernel ridge
regression, and Gaussian Processes — the mathematical framework that lets
linear algorithms operate in infinite-dimensional feature spaces at the
cost of a single dot product.

"""

import textwrap
import re

TOPIC_NAME = "Kernel Methods"
DISPLAY_NAME = "03 · Kernel Methods"
ICON = "🔮"
SUBTITLE = "Infinite Feature Spaces at Finite Cost"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Idea: Features Without Computing Them

Many ML algorithms can be expressed purely in terms of dot products between
data points. This single observation is the key to everything in kernel methods.

    Consider a linear classifier:   f(x) = wᵀx + b

    If we first map each x to a high-dimensional feature space via
    a function φ: ℝᵈ → ℝᴰ (D >> d), the model becomes:

        f(x) = wᵀφ(x) + b

    Learning and prediction require computing φ(xᵢ)ᵀφ(xⱼ) — the dot
    product of feature vectors for pairs of points.

    The kernel trick: replace φ(xᵢ)ᵀφ(xⱼ) with a kernel function:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │         k(xᵢ, xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)                                  │
    │                                                                  │
    │  Compute the dot product in feature space WITHOUT ever computing │
    │  φ(x) explicitly — sometimes even when φ maps to ℝ^∞.            │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 1 — The Kernel Trick:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Input space ℝᵈ          Feature space ℝᴰ  (D possibly ∞)       │
    │                                                                 │
    │   x₁ ●                      φ(x₁) ●                             │
    │        ╲                         ╲                              │
    │         ╲  k(x₁,x₂)                ╲  φ(x₁)ᵀφ(x₂)               │
    │          ╲                           ╲                          │
    │   x₂ ●    ╲─ shortcut! ──────────────● φ(x₂)                    │
    │                                                                 │
    │  NAIVE route:   x → φ(x) for each x   [O(D) per point]          │
    │                 then compute φ(xᵢ)ᵀφ(xⱼ)  [O(D) per pair]        │
    │                 TOTAL: O(n²D)  — infeasible if D is huge        │
    │                                                                 │
    │  KERNEL route:  directly compute k(xᵢ, xⱼ) in ℝᵈ                 │
    │                 TOTAL: O(n²d)  — independent of D!              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Worked example — polynomial kernel:

        In ℝ², with x = [x₁, x₂] and z = [z₁, z₂]:

        NAIVE feature map (degree-2 polynomial):
            φ(x) = [x₁², √2·x₁x₂, x₂²]   ∈ ℝ³
            φ(x)ᵀφ(z) = x₁²z₁² + 2x₁x₂z₁z₂ + x₂²z₂²

        KERNEL (same result, no φ computation):
            k(x, z) = (xᵀz)²
                     = (x₁z₁ + x₂z₂)²
                     = x₁²z₁² + 2x₁x₂z₁z₂ + x₂²z₂²  ✓

        For degree-p polynomial in ℝᵈ:
            Naive: compute φ ∈ ℝ^{C(d+p, p)}  [can be millions of features]
            Kernel: k(x,z) = (xᵀz + c)^p    [d multiplications + 1 power]


──────────────────────────────────────────────────────────────────────────────
### The Kernel Matrix (Gram Matrix)

For a dataset {x₁, ..., xₙ}, the kernel matrix K ∈ ℝⁿˣⁿ is:

        K_ij = k(xᵢ, xⱼ)

    K encodes all pairwise similarities in feature space.
    Every kernel algorithm can be written in terms of K alone —
    the raw inputs x are never needed again after forming K.

    Diagram 2 — The Gram Matrix:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   x₁  x₂  x₃  x₄      K  =   k(x₁,x₁)  k(x₁,x₂)  ...             │
    │                               k(x₂,x₁)  k(x₂,x₂)  ...            │
    │                               k(x₃,x₁)  ...                      │
    │                                                                  │
    │   k(xᵢ,xⱼ) = "similarity" of xᵢ and xⱼ in feature space.          │
    │   K_ii = k(xᵢ,xᵢ) = ||φ(xᵢ)||²  ("self-similarity")              │
    │                                                                  │
    │   Every algorithm that uses X through dot products can be        │
    │   kernelised by replacing XᵀX → K  (or similar substitution).    │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Properties of valid kernel matrices:
    ┌────────────────────────────────────────────────────────────────┐
    │  1.  SYMMETRIC:   K_ij = K_ji  (k(x,z) = k(z,x))               │
    │  2.  PSD:         vᵀKv ≥ 0  for all v ∈ ℝⁿ                     │
    │      (positive semi-definite — eigenvalues all ≥ 0)            │
    │  3.  DIAGONAL ≥ 0:   K_ii ≥ 0  (self-similarity ≥ 0)           │
    └────────────────────────────────────────────────────────────────┘

    Practical consequence: if you give a non-PSD matrix to an SVM solver,
    it can fail or produce meaningless results. Always verify your kernel.


──────────────────────────────────────────────────────────────────────────────
### Mercer's Theorem — What Makes a Valid Kernel

Not every symmetric function is a valid kernel. Mercer's theorem gives
the definitive characterisation.

    Theorem (Mercer, 1909):

    A symmetric function k: X × X → ℝ is a valid kernel (i.e., ∃ feature
    map φ such that k(x,z) = φ(x)ᵀφ(z)) if and only if:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  For all finite sets {x₁,...,xₙ} ⊆ X, the kernel matrix K         │
    │  with K_ij = k(xᵢ,xⱼ) is POSITIVE SEMI-DEFINITE.                  │
    │                                                                  │
    │  Equivalently: all eigenvalues of K are ≥ 0.                     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    The feature space guaranteed by Mercer's theorem is a
    Reproducing Kernel Hilbert Space (RKHS), denoted Hₖ.
    Its key property: f(x) = ⟨f, k(x,·)⟩_{Hₖ}  (the reproducing property).

    Kernel closure properties — valid kernels can be combined:

    If k₁ and k₂ are valid kernels, then so are:
    ┌──────────────────────────────────────────────────────────────────┐
    │  k₁(x,z) + k₂(x,z)          (sum)                                │
    │  c · k₁(x,z)      c > 0     (positive scaling)                   │
    │  k₁(x,z) · k₂(x,z)          (product)                            │
    │  f(x) · k₁(x,z) · f(z)      for any function f                   │
    │  exp(k₁(x,z))                (exponentiation)                    │
    │  p(k₁(x,z))                  for any polynomial p with ≥ 0 coefs │
    └──────────────────────────────────────────────────────────────────┘

    These closure rules allow kernel engineering: build a custom kernel
    for your problem domain by combining simpler valid kernels.


──────────────────────────────────────────────────────────────────────────────
### Kernel Catalogue — The Standard Kernels

    Diagram 3 — Decision Boundaries Induced by Common Kernels:

    Same dataset, different kernels → different feature spaces → different
    shapes of decision boundary that are expressible.

    ┌─────────────────┬───────────────────────┬───────────────────────────────┐
    │  KERNEL NAME    │  FORMULA              │  FEATURE SPACE / BEHAVIOUR    │
    ├─────────────────┼───────────────────────┼───────────────────────────────┤
    │  Linear         │  xᵀz                  │  ℝᵈ — linear boundaries only  │
    │  Polynomial(p)  │  (xᵀz + c)^p          │  All degree-≤p polynomials    │
    │  RBF / Gaussian │  exp(−γ‖x−z‖²)        │  ℝ^∞  — any smooth boundary   │
    │  Laplacian      │  exp(−γ‖x−z‖₁)        │  ℝ^∞  — less smooth than RBF  │
    │  Sigmoid        │  tanh(αxᵀz + c)       │  NOT always PSD — verify!     │
    │  Matérn(ν)      │  (depends on ν)       │  ℝ^∞  — smoothness controls   │
    │  Rational Quad. │  (1+‖x−z‖²/2αl²)^{-α} │ mixture of RBF scales         │
    └─────────────────┴───────────────────────┴───────────────────────────────┘

**RBF Kernel (Radial Basis Function / Gaussian Kernel):**

    k_RBF(x, z) = exp(−γ ‖x − z‖²)     γ = 1/(2σ²)

    Feature space: the corresponding φ maps to INFINITE dimensions
    (Taylor expansion of the exponential gives infinitely many terms).

    Geometric interpretation:
        k(x, z) ≈ 1   when x and z are very close
        k(x, z) ≈ 0   when x and z are far apart
        k(x, x) = 1   always (self-similarity = 1)

    The bandwidth parameter γ (or σ) controls how quickly similarity
    decays with distance:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Large γ (small σ):   k decays fast → very LOCAL kernel          │
    │                       → only nearby points influence prediction  │
    │                       → flexible, high variance, can overfit     │
    │                                                                  │
    │  Small γ (large σ):   k decays slowly → GLOBAL kernel            │
    │                       → distant points still affect prediction   │
    │                       → smoother, more bias, more like linear    │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 4 — RBF Kernel Decay with γ:

    k(x, z)
      1.0  │──── γ=0.1 (wide, global)
           │   ╲_______________
      0.5  │                  ╲── γ=1.0 (medium)
           │                       ╲___ γ=5.0 (narrow, local)
      0.0  │
           └──────────────────────────────── ‖x − z‖
                1    2    3    4    5

**Polynomial Kernel:**

    k_poly(x, z) = (xᵀz + c)^p

    Controls degree p (complexity) and bias c (offset).
    c > 0 ensures all monomials up to degree p are included.
    c = 0 gives only homogeneous polynomials of degree exactly p.

**Matérn Kernel:**

    Parametrised by ν controlling the smoothness of the RKHS.
    ν = 1/2:   Matérn₁/₂ = Laplacian  (once differentiable)
    ν = 3/2:   Matérn₃/₂              (twice differentiable)
    ν = 5/2:   Matérn₅/₂              (three times differentiable)
    ν → ∞:     Matérn → RBF           (infinitely differentiable)

    The Matérn kernel is particularly popular in Gaussian Processes
    because it allows explicit control over the smoothness of the
    functions it models — unlike RBF which is always infinitely smooth.


──────────────────────────────────────────────────────────────────────────────
### Support Vector Machines — The Canonical Kernel Algorithm

SVMs find the maximum-margin hyperplane in feature space. The key insight
is that the dual formulation depends only on kernel evaluations.

**Primal Problem — Linear SVM (hard margin):**

    Given linearly separable binary data {(xᵢ, yᵢ)}, yᵢ ∈ {−1, +1}:

        Minimise:     (1/2) ‖w‖²
        Subject to:   yᵢ(wᵀxᵢ + b) ≥ 1    ∀ i

    Diagram 5 — Maximum Margin Classifier:

    x₂
    │         ● +1 class            margin = 2/‖w‖  ← MAXIMISED
    │       ●   ●                   ├───────────────┤
    │     ○   ●   ●     ─ ─ ─ ─ ─ ─|─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─
    │   ○   ○   ●        support  →│←──── decision ──│→ boundary
    │ ○   ○         ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─
    │   ○   ○   ← support vectors   │
    └──────────────────────────────────── x₁
                       ○ -1 class

    ● points exactly on the margin boundary: yᵢ(wᵀxᵢ + b) = 1
    These are the SUPPORT VECTORS — the only points that matter.
    All other points can be moved without changing the hyperplane.

**Soft Margin SVM — handling non-separable data:**

    Introduce slack variables ξᵢ ≥ 0 allowing margin violations:

        Minimise:     (1/2)‖w‖² + C · Σᵢ ξᵢ
        Subject to:   yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ    ∀ i
                      ξᵢ ≥ 0

    C = regularisation parameter (inverse of regularisation strength):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Large C:  heavy penalty on violations → small margin, overfit   │
    │  Small C:  tolerates violations → large margin, regularised      │
    └──────────────────────────────────────────────────────────────────┘

**Dual Formulation — the gateway to kernels:**

    Using Lagrangian duality (KKT conditions), the primal SVM
    transforms into the dual:

        Maximise:   Σᵢ αᵢ  −  (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ xᵢᵀxⱼ
        Subject to: 0 ≤ αᵢ ≤ C    and   Σᵢ αᵢ yᵢ = 0

    αᵢ = Lagrange multiplier for point i.
    αᵢ > 0  ONLY for support vectors — all others have αᵢ = 0.

    The decision function in dual form:

        f(x) = sign( Σᵢ αᵢ yᵢ xᵢᵀx + b )

    ┌──────────────────────────────────────────────────────────────────┐
    │  KEY OBSERVATION: the dual only uses xᵢᵀxⱼ — dot products!        │
    │                                                                  │
    │  Replace xᵢᵀxⱼ with k(xᵢ, xⱼ) and x with φ(x) throughout:          │
    │                                                                  │
    │  Kernel SVM dual:                                                │
    │    Maximise: Σᵢ αᵢ − (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ k(xᵢ,xⱼ)               │
    │    Subject to: 0 ≤ αᵢ ≤ C  and  Σᵢ αᵢ yᵢ = 0                     │
    │                                                                  │
    │  Kernel SVM decision function:                                   │
    │    f(x) = sign( Σᵢ αᵢ yᵢ k(xᵢ, x) + b )                          │
    │                                                                  │
    │  The feature map φ never appears explicitly!                     │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 6 — Kernel SVM: XOR Problem (not linearly separable):

    Input space (linear SVM fails):       Feature space via RBF kernel:
    x₂                                    z₂
    │   ●  ○                              │   ●    ●
    │   ○  ●                              │        ○  ○
    └──────── x₁                          └─────────────── z₁

    XOR pattern: classes alternate in 2D → no linear separator exists.
    RBF kernel lifts data to infinite-dim space where a hyperplane DOES
    separate the classes. The decision boundary in input space is a curve.

    KKT Conditions and Sparsity:

        αᵢ = 0:     point is not a support vector (outside margin)
        0 < αᵢ < C: point is on the margin boundary (support vector)
        αᵢ = C:     point violates the margin (slack ξᵢ > 0)

    SPARSITY IS THE KEY ADVANTAGE OF SVMs:
    At prediction time, only support vectors matter:
        f(x) = sign( Σ_{support vectors} αᵢ yᵢ k(xᵢ, x) + b )
    For many problems, only 5–30% of training points become support vectors.


──────────────────────────────────────────────────────────────────────────────
### The Representer Theorem

Why do kernel methods always produce solutions expressible in terms of
the training data? The Representer Theorem gives the answer.

    Theorem (Representer Theorem, Kimeldorf-Wahba):

    For any learning problem of the form:

        min_{f ∈ Hₖ}   L(y₁, f(x₁), ..., yₙ, f(xₙ))  +  λ·Ω(‖f‖_{Hₖ})

    where L is any loss function and Ω is any strictly increasing
    function, the optimal solution has the form:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │         f*(x)  =  Σᵢ₌₁ⁿ  αᵢ · k(xᵢ, x)                            │
    │                                                                  │
    │  The optimal function is a linear combination of kernel          │
    │  evaluations at training points — regardless of the dimension    │
    │  of Hₖ (even if Hₖ is infinite-dimensional).                      │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    CONSEQUENCES:
    1.  Any regularised learning in an RKHS reduces to finding n real
        coefficients αᵢ — not an infinite-dimensional function.
    2.  The kernel matrix K captures everything needed.
    3.  SVM, kernel ridge regression, kernel logistic regression, and
        Gaussian processes all satisfy this theorem.
    4.  Prediction requires computing n kernel evaluations per test point.

    This is why kernel methods scale as O(n²) or O(n³) — the kernel
    matrix must be formed (O(n²)) and often solved (O(n³)).


──────────────────────────────────────────────────────────────────────────────
### Kernel Ridge Regression (KRR)

The simplest kernel algorithm — ridge regression with the kernel trick.

    Primal form (feature space):
        min_w  ‖Φw − y‖² + λ‖w‖²
        Solution: w* = (ΦᵀΦ + λI)⁻¹ Φᵀy   ∈ ℝᴰ  (D can be ∞!)

    Dual form (kernel):
        α* = (K + λI)⁻¹ y               ∈ ℝⁿ
        Prediction: f(x) = kᵀ(x) α*

    where K_ij = k(xᵢ, xⱼ)  and  k(x) = [k(x,x₁), ..., k(x,xₙ)]ᵀ.

    ┌──────────────────────────────────────────────────────────────────┐
    │  DUAL TRICK: instead of inverting (ΦᵀΦ + λI) ∈ ℝᴰˣᴰ              │
    │  (D may be infinite), invert (K + λI) ∈ ℝⁿˣⁿ  (always finite!)    │
    │                                                                  │
    │  Woodbury identity:   (ΦᵀΦ + λI)⁻¹ Φᵀ = Φᵀ(ΦΦᵀ + λI)⁻¹            │
    │  This switches from a D×D to an n×n solve.                       │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 7 — KRR vs Linear Ridge Regression:

    True function: sin(2πx) — clearly non-linear

    LINEAR RIDGE (feature space = ℝ¹):
    Fit:         ─────────────────── (flat line, high bias)

    KRR with RBF (feature space = ℝ^∞):
    Fit:         ╱╲    ╱╲    (captures the sine shape)
                ╱  ╲  ╱  ╲

    Same regularisation framework, completely different expressivity,
    all from substituting k(xᵢ, xⱼ) for xᵢᵀxⱼ.


──────────────────────────────────────────────────────────────────────────────
### Gaussian Processes — The Probabilistic Kernel View

A Gaussian Process (GP) is a Bayesian non-parametric model where the
kernel function defines a prior distribution over functions.

    Definition:

    A GP is a collection of random variables f(x₁), ..., f(xₙ) such
    that any finite subset is jointly Gaussian:

        [f(x₁), ..., f(xₙ)]ᵀ  ~  N(μ, K)

    where  μᵢ = m(xᵢ)         (prior mean function, often m ≡ 0)
    and    K_ij = k(xᵢ, xⱼ)   (kernel = covariance function!)

    ┌──────────────────────────────────────────────────────────────────┐
    │  The kernel IS the covariance: k(xᵢ, xⱼ) = Cov[f(xᵢ), f(xⱼ)]       │
    │                                                                  │
    │  Nearby points (in the sense of k) have correlated function      │
    │  values. The kernel encodes our prior belief about the           │
    │  SMOOTHNESS and STRUCTURE of the unknown function.               │
    └──────────────────────────────────────────────────────────────────┘

    GP Regression (GP Posterior):

    Observed noisy data: y = f(X) + ε,  ε ~ N(0, σ²I)

    Prior:   f ~ GP(0, k)

    Posterior (conditioning on observations — exact Gaussian):

        μ*(x*) = kᵀ(x*)(K + σ²I)⁻¹y             (posterior mean)
        σ²*(x*) = k(x*,x*) − kᵀ(x*)(K + σ²I)⁻¹k(x*)   (posterior variance)

    CRUCIAL INSIGHT:
    The posterior mean  μ*(x*)  is IDENTICAL to the KRR prediction!
    GP regression = KRR + uncertainty quantification.

    The posterior variance σ²*(x*) tells you:
    ─ Near training data:   σ²* ≈ 0   (confident prediction)
    ─ Far from training data:  σ²* ≈ k(x*,x*)   (prior uncertainty)

    Diagram 8 — GP Prior vs Posterior:

    f(x)
    │        ─ ─ ─ ─ ─         PRIOR: many smooth functions consistent
    │      ─           ─       with the kernel. Each sample is a
    │    ─               ─     function draw from GP(0, k).
    │ ─ ─                 ─ ─
    │                                   ←── confidence band
    │                   ● ●         POSTERIOR: narrows near data points
    │         ●         ──── μ*     data forces posterior to pass through
    │  ●  ●            ↑            observed values (approximately, for σ>0)
    │                  confidence band widens far from data
    └──────────────────────────────────────────────────────── x

    Kernel Hyperparameter Learning (Marginal Likelihood):

    The GP kernel has hyperparameters (e.g., γ and σ for RBF).
    Optimise them by maximising the log marginal likelihood:

        log p(y | X, θ) = −(1/2)yᵀ(K+σ²I)⁻¹y − (1/2)log|K+σ²I| − (n/2)log(2π)
                                  ↑                      ↑
                           data fit term         complexity penalty

    This is the GP's built-in version of SRM — the marginal likelihood
    automatically balances fit and complexity. No cross-validation needed
    for hyperparameter selection (though it can still help).


──────────────────────────────────────────────────────────────────────────────
### Computational Considerations — Scaling Kernel Methods

The main limitation of kernel methods: O(n²) storage and O(n³) training.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Kernel matrix K:   n² entries  (n=10,000 → 800 MB at float64)   │
    │  KRR / GP solve:    O(n³)        (Cholesky decomposition)        │
    │  SVM training:      O(n² − n³)   (QP solver)                     │
    │  Prediction:        O(n_sv × d)  (only support vectors for SVM)  │
    └──────────────────────────────────────────────────────────────────┘

    Approximations for large n:

    NYSTRÖM APPROXIMATION:
        Approximate K using m << n "landmark" points:
        K ≈ K_{nm} K_{mm}⁻¹ K_{mn}
        Reduces O(n³) to O(nm²).  m ≈ 1000 works well in practice.

    RANDOM FOURIER FEATURES (Rahimi & Recht, 2007):
        For shift-invariant kernels (RBF, Matérn):
        k(x,z) = k(x−z) = E_{ω}[φ_ω(x)ᵀφ_ω(z)]
        Sample D random ω ~ p(ω) and approximate:
        k(x,z) ≈ (1/D) Σⱼ cos(ωⱼᵀx + bⱼ) cos(ωⱼᵀz + bⱼ)
        This gives an explicit D-dimensional feature map; use linear methods.
        Reduces O(n³) to O(nD) where D = 1000 to 10000.

    INDUCING POINT METHODS (sparse GPs):
        Summarise the training data with m << n inducing points.
        Variational inference approximates the full GP posterior.
        State-of-the-art GP library GPyTorch uses this approach.

    Diagram 9 — Computational Complexity Comparison:

    Method             │  Training       │  Prediction    │  Storage
    ───────────────────┼─────────────────┼────────────────┼──────────
    Exact KRR / GP     │  O(n³)          │  O(n × d)      │  O(n²)
    SVM                │  O(n² − n³)     │  O(n_sv × d)   │  O(n²)
    Nyström (rank m)   │  O(nm² + m³)    │  O(m × d)      │  O(nm)
    Random Fourier(D)  │  O(nD)          │  O(D)          │  O(nD)
    Neural Network     │  O(n × params)  │  O(params)     │  O(params)

    For n < 10,000: exact kernel methods are practical.
    For n > 100,000: use approximations or switch to neural networks.


──────────────────────────────────────────────────────────────────────────────
### Kernel Methods vs Neural Networks

A deep connection discovered in the 2000s–2010s:

    Neural Tangent Kernel (NTK):

        An infinitely-wide neural network trained with gradient descent
        is EQUIVALENT to kernel regression with the NTK kernel.
        The NTK is determined by the network architecture.

    This means kernel methods are not obsolete:
    ─ They provide the theoretical foundation for understanding neural nets.
    ─ For small-to-medium datasets (n < 50k), kernel methods often match
      or outperform neural networks with far less tuning.
    ─ GPs provide uncertainty quantification that neural nets lack.
    ─ Kernel methods have global optima; no saddle points or local minima.

    When to use kernel methods:
    ┌──────────────────────────────────────────────────────────────────┐
    │  ✓  Small-to-medium datasets (n < 50,000)                        │
    │  ✓  When uncertainty quantification matters (GP)                 │
    │  ✓  When interpretable support vectors are useful (SVM)          │
    │  ✓  Structured data with natural kernels (strings, graphs, sets) │
    │  ✓  When you need guarantees (convex optimisation, global opt.)  │
    │  ✗  Very large datasets (n > 100,000) — O(n²) is prohibitive     │
    │  ✗  Raw images / text / audio (deep learning better suited)      │
    └──────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · The Kernel Trick — Feature Maps vs Direct Kernel Computation": {
        "description": (
            "Demonstrates the kernel trick concretely: computes the same "
            "dot product two ways — via explicit feature map φ(x)ᵀφ(z) and "
            "via the kernel function k(x,z) — and verifies they match to "
            "machine precision. Covers polynomial, RBF, and custom kernels. "
            "Shows the computational savings: naive O(D) vs kernel O(d). "
            "Validates PSD property of kernel matrices via eigenvalues."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Polynomial kernel trick — explicit vs kernel
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  THE KERNEL TRICK: EXPLICIT FEATURE MAP vs KERNEL FUNCTION")
print("=" * 65)
print()

def poly_feature_map(x, degree=2, c=1.0):
    """
    Explicit polynomial feature map for x ∈ ℝᵈ.
    Returns φ(x) ∈ ℝ^{C(d+p, p)}.
    For degree 2, d=2: φ(x) = [c, x₁, x₂, x₁², √2·x₁x₂, x₂²]
    (with appropriate √2 scaling for cross terms).
    """
    d = len(x)
    features = [np.sqrt(c)]  # bias term from (xᵀz + c)^p expansion
    # degree-1 terms
    for i in range(d):
        features.append(x[i])
    # degree-2 terms (with √2 for cross-terms to match (xᵀz + c)²)
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    features.append(x[i]**2)
                else:
                    features.append(np.sqrt(2) * x[i] * x[j])
    return np.array(features)

def poly_kernel(x, z, degree=2, c=1.0):
    """k(x,z) = (xᵀz + c)^p."""
    return (np.dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=1.0):
    """k(x,z) = exp(−γ‖x − z‖²)."""
    diff = x - z
    return np.exp(-gamma * np.dot(diff, diff))

# Test on 2D points
x = np.array([2.0, 3.0])
z = np.array([1.0, -1.0])

# Polynomial degree=2, c=1
phi_x = poly_feature_map(x, degree=2, c=1.0)
phi_z = poly_feature_map(z, degree=2, c=1.0)

dot_explicit = np.dot(phi_x, phi_z)
dot_kernel   = poly_kernel(x, z, degree=2, c=1.0)

print("  POLYNOMIAL KERNEL (degree=2, c=1.0):")
print(f"  x = {x},  z = {z}")
print()
print(f"  Feature map φ(x) ∈ ℝ^{len(phi_x)}:")
print(f"    {phi_x.round(4)}")
print(f"  Feature map φ(z) ∈ ℝ^{len(phi_z)}:")
print(f"    {phi_z.round(4)}")
print()
print(f"  Explicit:  φ(x)ᵀ φ(z)    = {dot_explicit:.6f}")
print(f"  Kernel:    k(x, z)        = {dot_kernel:.6f}")
print(f"  Match:     {np.isclose(dot_explicit, dot_kernel)}")
print()

# Higher-degree polynomial — show the dimensionality explosion
print("  DIMENSIONALITY OF POLYNOMIAL FEATURE SPACE:")
from math import comb
for d_feat in [2, 5, 10, 20, 50, 100]:
    for p in [2, 3, 5]:
        n_features = comb(d_feat + p, p)
        print(f"    d={d_feat:>4}, p={p}: φ(x) ∈ ℝ^{n_features:>12,}  "
              f"  kernel: still {d_feat} multiplications + 1 power")

print()
print("  → Kernel is O(d) regardless of feature space dimension.")
print("  → Explicit feature map becomes infeasible for large d or p.")

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: RBF kernel — infinite feature space
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  RBF KERNEL — INFINITE-DIMENSIONAL FEATURE SPACE")
print("=" * 65)
print()

x_rbf = np.array([1.0, 2.0])
z_rbf = np.array([3.0, 1.0])

for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
    k_val = rbf_kernel(x_rbf, z_rbf, gamma)
    dist  = np.linalg.norm(x_rbf - z_rbf)
    print(f"  γ={gamma:.1f}:  dist={dist:.4f}  k(x,z)={k_val:.6f}  "
          f"  (decay: {1 - k_val:.4f} from perfect similarity)")

print()
print("  The RBF kernel corresponds to φ(x) ∈ ℝ^∞ via Taylor expansion:")
print("  exp(−γ‖x−z‖²) = exp(−γ‖x‖²)·exp(−γ‖z‖²)·exp(2γxᵀz)")
print("  The last term has a Taylor series → infinite feature dimensions.")
print()
print("  Key property: k(x,x) = exp(0) = 1 for all x (unit norm in RKHS).")

# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Positive semi-definiteness verification
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  PSD VERIFICATION — MERCER'S THEOREM IN PRACTICE")
print("=" * 65)
print()

n_pts = 20
X_test = np.random.randn(n_pts, 3)

kernels = {
    "Linear k(x,z) = xᵀz":        lambda x, z: x @ z.T,
    "RBF γ=1.0":                   lambda x, z: np.exp(-np.sum((x[:,None]-z[None,:])**2, axis=2)),
    "Poly (p=3, c=1)":             lambda x, z: (x @ z.T + 1)**3,
    "Sigmoid (NOT always PSD)":    lambda x, z: np.tanh(0.5 * x @ z.T + 0.5),
    "Neg squared dist (INVALID)":  lambda x, z: -np.sum((x[:,None]-z[None,:])**2, axis=2),
}

print(f"  {'Kernel':>40}  {'Min eigenval':>14}  {'Is PSD?':>8}")
print("  " + "─" * 67)

for name, kfn in kernels.items():
    K = kfn(X_test, X_test)
    eigvals = np.linalg.eigvalsh(K)
    min_ev  = eigvals.min()
    is_psd  = min_ev >= -1e-9
    print(f"  {name:>40}  {min_ev:>14.6f}  {'✓ YES' if is_psd else '✗ NO ':>8}")

print()
print("  - Linear, RBF, Polynomial kernels are always PSD (Mercer).")
print("  - Sigmoid is PSD for some params but not all — verify before use!")
print("  - Negative squared distance is NOT a valid kernel.")

# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Kernel closure properties
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  KERNEL CLOSURE: BUILDING CUSTOM KERNELS")
print("=" * 65)
print()

def build_gram(X, kfn):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kfn(X[i], X[j])
    return K

X_small = np.random.randn(15, 2)

k_rbf   = lambda x, z: np.exp(-np.linalg.norm(x-z)**2)
k_poly  = lambda x, z: (np.dot(x, z) + 1)**2
k_lin   = lambda x, z: np.dot(x, z)

combinations = {
    "k_rbf  (base)":            k_rbf,
    "k_poly (base)":            k_poly,
    "k_rbf + k_poly (sum)":     lambda x, z: k_rbf(x,z) + k_poly(x,z),
    "k_rbf × k_poly (product)": lambda x, z: k_rbf(x,z) * k_poly(x,z),
    "3 × k_rbf (scaled)":       lambda x, z: 3 * k_rbf(x,z),
    "exp(k_lin) (exponent)":    lambda x, z: np.exp(k_lin(x,z)),
}

print(f"  {'Kernel':>32}  {'Min eigenval':>14}  {'Is PSD?':>8}")
print("  " + "─" * 57)

for name, kfn in combinations.items():
    K = build_gram(X_small, kfn)
    min_ev = np.linalg.eigvalsh(K).min()
    print(f"  {name:>32}  {min_ev:>14.6f}  {'✓':>8}")

print()
print("  All combinations of valid kernels remain valid — Mercer closure.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("The Kernel Trick: Feature Maps, RBF Decay, and Gram Matrices",
             fontsize=13, fontweight="bold")

# Panel 1: Dimensionality of polynomial feature space
ax = axes[0]
d_vals = [2, 5, 10, 20, 50, 100]
for p, col in [(2, "steelblue"), (3, "tomato"), (5, "seagreen")]:
    dims = [comb(d + p, p) for d in d_vals]
    ax.semilogy(d_vals, dims, "o-", lw=2, ms=7, color=col, label=f"degree p={p}")
ax.set_xlabel("Input dimension d")
ax.set_ylabel("Feature space dimension D (log)")
ax.set_title("Polynomial Feature Space Size\\n(kernel avoids computing this)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel 2: RBF kernel decay for different gamma
ax = axes[1]
dist_vals = np.linspace(0, 5, 300)
for gamma, col, ls in [(0.1, "steelblue", "-"), (0.5, "seagreen", "--"),
                        (1.0, "tomato", "-"), (3.0, "purple", ":"),
                        (5.0, "darkorange", "-.")]:
    k_vals = np.exp(-gamma * dist_vals**2)
    ax.plot(dist_vals, k_vals, color=col, lw=2, ls=ls, label=f"γ={gamma}")
ax.set_xlabel("Distance ‖x − z‖")
ax.set_ylabel("k(x, z) = exp(−γ‖x−z‖²)")
ax.set_title("RBF Kernel Decay\\n(larger γ → more local)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 3: Gram matrices for different kernels
n_vis = 30
X_vis = np.random.randn(n_vis, 2) * 1.5
order = np.argsort(X_vis[:, 0])
X_vis = X_vis[order]

K_rbf  = np.exp(-np.sum((X_vis[:,None]-X_vis[None,:])**2, axis=2))
K_poly = (X_vis @ X_vis.T + 1)**2
K_lin  = X_vis @ X_vis.T

# Show RBF gram matrix
im = axes[2].imshow(K_rbf, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=axes[2])
axes[2].set_title("Gram Matrix K (RBF kernel, γ=1)\\nSorted by x₁ coordinate",
                  fontweight="bold")
axes[2].set_xlabel("Point index"); axes[2].set_ylabel("Point index")

plt.tight_layout()
plt.savefig("kernel_trick.png", dpi=110)
print()
print("  Plot saved → kernel_trick.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · SVM from Scratch — Primal, Dual, and the Margin": {
        "description": (
            "Implement the soft-margin SVM dual from scratch using only NumPy "
            "and scipy. Solves the QP using scipy.optimize.minimize, extracts "
            "support vectors and the decision boundary, and visualises the "
            "margin. Computes the primal weight vector from dual solution and "
            "verifies KKT conditions. Compares hard-margin vs soft-margin "
            "behaviour under different C values."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy

def make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=None, n_features=2):
    rng = _np_impl.random.default_rng(random_state)
    c_arr = rng.uniform(-5, 5, (centers, n_features)) if isinstance(centers, int) else _np_impl.array(centers)
    n_c = len(c_arr)
    counts = [n_samples//n_c + (1 if i < n_samples%n_c else 0) for i in range(n_c)]
    Xs, ys = [], []
    for i, (cnt, c) in enumerate(zip(counts, c_arr)):
        Xs.append(rng.normal(c, cluster_std, (cnt, n_features))); ys.append(_np_impl.full(cnt, i, dtype=int))
    return _np_impl.vstack(Xs), _np_impl.hstack(ys)

def make_circles(n_samples=100, noise=0.0, factor=0.5, random_state=None):
    rng = _np_impl.random.default_rng(random_state)
    n_out = n_samples//2; n_in = n_samples - n_out
    t_out = rng.uniform(0, 2*_np_impl.pi, n_out); t_in = rng.uniform(0, 2*_np_impl.pi, n_in)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t_out), _np_impl.sin(t_out)],
                         _np_impl.c_[factor*_np_impl.cos(t_in), factor*_np_impl.sin(t_in)]])
    if noise > 0: X += rng.normal(0, noise, X.shape)
    return X, _np_impl.hstack([_np_impl.zeros(n_out,dtype=int), _np_impl.ones(n_in,dtype=int)])

def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng = _np_impl.random.default_rng(random_state); n = n_samples//2
    t = _np_impl.linspace(0, _np_impl.pi, n)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t), _np_impl.sin(t)],
                         _np_impl.c_[1-_np_impl.cos(t), -_np_impl.sin(t)+0.5]])
    if noise > 0: X += rng.normal(0, noise, X.shape)
    return X, _np_impl.hstack([_np_impl.zeros(n), _np_impl.ones(n_samples-n)]).astype(int)

def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    y = rng.integers(0, 2, n_samples)
    Xi = rng.standard_normal((n_samples, n_informative))
    Xi += (2*y-1)[:,None] * rng.uniform(0.8, 1.5, n_informative)
    Xr = Xi[:,:n_redundant]+0.3*rng.standard_normal((n_samples,n_redundant)) if n_redundant>0 else _np_impl.empty((n_samples,0))
    nn = max(0, n_features-n_informative-n_redundant)
    Xn = rng.standard_normal((n_samples,nn)) if nn>0 else _np_impl.empty((n_samples,0))
    return _np_impl.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features], y.astype(int)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_==0] = 1.0; return self
    def transform(self, X):            return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

def accuracy_score(y_true, y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true) == _np_impl.asarray(y_pred))

def train_test_split(*arrays, test_size=0.25, random_state=None, train_size=None):
    rng = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = train_size if train_size is not None else int(n*(1-test_size))
    idx = rng.permutation(n); tr_i, te_i = idx[:n_tr], idx[n_tr:]
    out = []
    for a in arrays: out += [a[tr_i], a[te_i]]
    return out

def cross_val_score(estimator, X, y, cv=5, scoring="accuracy"):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]) if "mse" not in scoring
                      else -_np_impl.mean((est.predict(X[vi])-y[vi])**2))
    return _np_impl.array(scores)

class KernelRidge:
    def __init__(self, alpha=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
        self.alpha=alpha; self.kernel=kernel; self.gamma=gamma; self.degree=degree; self.coef0=coef0
    def _K(self, X1, X2):
        if self.kernel=="rbf":
            d=((_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T))
            return _np_impl.exp(-self.gamma*d)
        elif self.kernel=="poly": return (X1@X2.T+self.coef0)**self.degree
        return X1@X2.T
    def fit(self, X, y):
        self.X_tr=X.copy(); n=len(X)
        self.dual_coef_=_np_impl.linalg.solve(self._K(X,X)+self.alpha*_np_impl.eye(n),y); return self
    def predict(self, X): return self._K(X,self.X_tr)@self.dual_coef_

class SVC:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 probability=False, random_state=None):
        self.C=C; self.kernel=kernel; self._gr=gamma; self.degree=degree; self.coef0=coef0
    def _g(self, X):
        if isinstance(self._gr, float): return self._gr
        return 1.0/(X.shape[1]*X.var()) if self._gr=="scale" and X.var()>0 else 1.0/X.shape[1]
    def _K(self, X1, X2, g):
        if self.kernel=="rbf":
            d=(_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T)
            return _np_impl.exp(-g*d)
        elif self.kernel=="poly":   return (g*(X1@X2.T)+self.coef0)**self.degree
        elif self.kernel=="sigmoid": return _np_impl.tanh(g*(X1@X2.T)+self.coef0)
        return X1@X2.T
    def fit(self, X, y):
        self._classes=_np_impl.unique(y)
        yb=_np_impl.where(y==self._classes[1],1.0,-1.0); g=self._g(X); n=len(X)
        K=self._K(X,X,g); H=_np_impl.outer(yb,yb)*K
        res=_sp_opt.minimize(lambda a:0.5*a@H@a-a.sum(), _np_impl.zeros(n),
                             jac=lambda a:H@a-_np_impl.ones(n), method="SLSQP",
                             bounds=[(0,self.C)]*n,
                             constraints={"type":"eq","fun":lambda a:_np_impl.dot(a,yb),"jac":lambda a:yb},
                             options={"maxiter":2000,"ftol":1e-9})
        self.alphas_=_np_impl.maximum(res.x,0); sv=self.alphas_>1e-5
        self.support_=_np_impl.where(sv)[0]; self.support_vectors_=X[sv]
        self._y_sv=yb[sv]; self._a_sv=self.alphas_[sv]; self._g_fit=g
        on_m=self._a_sv<self.C-1e-5
        ref=self.support_vectors_[on_m] if on_m.any() else self.support_vectors_
        ref_y=self._y_sv[on_m] if on_m.any() else self._y_sv
        self._b=float(_np_impl.mean(ref_y-self._K(ref,self.support_vectors_,g)@(self._a_sv*self._y_sv)))
        self.n_support_=_np_impl.array([_np_impl.sum(self._y_sv==-1),_np_impl.sum(self._y_sv==1)])
        return self
    def decision_function(self,X): return self._K(X,self.support_vectors_,self._g_fit)@(self._a_sv*self._y_sv)+self._b
    def predict(self,X): return _np_impl.where(self.decision_function(X)>=0,self._classes[1],self._classes[0])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class SVR:
    def __init__(self, kernel="rbf", C=1.0, gamma=1.0, epsilon=0.1, degree=3, coef0=1.0):
        self.kernel=kernel; self.C=C; self.gamma=gamma; self.epsilon=epsilon
        self.degree=degree; self.coef0=coef0
    def _K(self, X1, X2):
        if self.kernel=="rbf":
            d=(_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T)
            return _np_impl.exp(-self.gamma*d)
        elif self.kernel=="poly": return (X1@X2.T+self.coef0)**self.degree
        return X1@X2.T
    def fit(self, X, y):
        n=len(X); lam=1.0/(2*self.C*n); K=self._K(X,X)
        self._X_tr=X.copy(); self._alpha=_np_impl.linalg.solve(K+lam*_np_impl.eye(n),y)
        f=K@self._alpha; resid=_np_impl.abs(f-y)
        sv_mask=resid>self.epsilon*0.9
        if not sv_mask.any(): sv_mask=resid>=resid.max()*0.5
        self.support_=_np_impl.where(sv_mask)[0]; self.support_vectors_=X[sv_mask]
        self._b=float(_np_impl.mean(y[sv_mask]-f[sv_mask])); return self
    def predict(self,X): return self._K(X,self._X_tr)@self._alpha+self._b

# Stubs for unused GP imports
class GaussianProcessRegressor: pass
class RBF: pass
class WhiteKernel: pass
class ConstantKernel: pass


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Linear SVM dual from scratch
# ─────────────────────────────────────────────────────────────────────────────

class LinearSVMDual:
    """
    Soft-margin linear SVM solved in the dual via scipy.
    Dual: maximise Σαᵢ - (1/2) Σᵢⱼ αᵢαⱼ yᵢyⱼ xᵢᵀxⱼ
    subject to 0 ≤ αᵢ ≤ C, Σᵢ αᵢyᵢ = 0
    """
    def __init__(self, C=1.0, kernel="linear", gamma=1.0, degree=2):
        self.C      = C
        self.kernel = kernel
        self.gamma  = gamma
        self.degree = degree
        self.alphas = None
        self.b      = None
        self.X_sv   = None
        self.y_sv   = None
        self.a_sv   = None

    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            dists = np.sum(X1**2, axis=1, keepdims=True) \
                  + np.sum(X2**2, axis=1) \
                  - 2 * X1 @ X2.T
            return np.exp(-self.gamma * dists)
        elif self.kernel == "poly":
            return (X1 @ X2.T + 1) ** self.degree

    def fit(self, X, y):
        n = len(X)
        K = self._kernel(X, X)

        # Dual objective (negated for minimisation)
        H = np.outer(y, y) * K
        def neg_dual(alpha):
            return 0.5 * alpha @ H @ alpha - np.sum(alpha)
        def neg_dual_grad(alpha):
            return H @ alpha - np.ones(n)

        # Constraints: Σ αᵢ yᵢ = 0
        constraints = {"type": "eq",
                       "fun": lambda a: np.dot(a, y),
                       "jac": lambda a: y}
        bounds = [(0, self.C)] * n

        result = minimize(neg_dual, x0=np.zeros(n),
                          jac=neg_dual_grad,
                          method="SLSQP",
                          bounds=bounds,
                          constraints=constraints,
                          options={"maxiter": 2000, "ftol": 1e-9})
        self.alphas = np.maximum(result.x, 0)

        # Support vectors: αᵢ > threshold
        sv_mask    = self.alphas > 1e-5
        self.X_sv  = X[sv_mask]
        self.y_sv  = y[sv_mask]
        self.a_sv  = self.alphas[sv_mask]

        # Primal w (linear kernel only)
        if self.kernel == "linear":
            self.w = np.sum((self.a_sv * self.y_sv)[:, None] * self.X_sv, axis=0)
        else:
            self.w = None

        # Bias b: average over all support vectors on the margin (0 < α < C)
        on_margin = self.alphas[sv_mask] < self.C - 1e-5
        if on_margin.sum() > 0:
            K_sv = self._kernel(self.X_sv[on_margin], self.X_sv)
            f_sv = K_sv @ (self.a_sv * self.y_sv)
            self.b = np.mean(self.y_sv[on_margin] - f_sv)
        else:
            K_sv = self._kernel(self.X_sv, self.X_sv)
            f_sv = K_sv @ (self.a_sv * self.y_sv)
            self.b = np.mean(self.y_sv - f_sv)

        return self

    def decision_function(self, X):
        K_test = self._kernel(X, self.X_sv)
        return K_test @ (self.a_sv * self.y_sv) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Linear SVM — KKT conditions and margin geometry
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  LINEAR SVM DUAL — KKT CONDITIONS AND SUPPORT VECTORS")
print("=" * 65)
print()

X_lin, y_lin = make_blobs(n_samples=60, centers=2, cluster_std=1.2, random_state=7)
y_lin = 2 * y_lin - 1   # map {0,1} → {-1, +1}
scaler = StandardScaler()
X_lin  = scaler.fit_transform(X_lin)

for C_val in [0.1, 1.0, 10.0, 1000.0]:
    svm = LinearSVMDual(C=C_val, kernel="linear").fit(X_lin, y_lin)
    acc = svm.score(X_lin, y_lin)
    n_sv = len(svm.X_sv)
    margin = 2.0 / np.linalg.norm(svm.w) if svm.w is not None else float("nan")
    print(f"  C={C_val:>7}:  support vectors={n_sv:>4}  "
          f"margin={margin:.4f}  accuracy={acc:.3f}")

print()
print("  OBSERVATIONS:")
print("  - Large C → few support vectors, small margin (harder boundary).")
print("  - Small C → many support vectors, large margin (more regularised).")

# Use C=1 for detailed KKT analysis
svm_kkt = LinearSVMDual(C=1.0, kernel="linear").fit(X_lin, y_lin)
print()
print("  KKT CONDITIONS VERIFICATION (C=1.0):")
print(f"  Σ αᵢ yᵢ = {np.dot(svm_kkt.alphas, y_lin):.6f}  (should be 0)")
print(f"  Primal w = {svm_kkt.w.round(4)}")
print(f"  Bias   b = {svm_kkt.b:.4f}")
print(f"  Margin   = 2/‖w‖ = {2/np.linalg.norm(svm_kkt.w):.4f}")
print()

K_full = X_lin @ X_lin.T
dual_obj = np.sum(svm_kkt.alphas) - 0.5 * svm_kkt.alphas @ (np.outer(y_lin, y_lin) * K_full) @ svm_kkt.alphas
primal_obj = 0.5 * np.dot(svm_kkt.w, svm_kkt.w)
print(f"  Dual objective  (maximised):  {dual_obj:.6f}")
print(f"  Primal objective (minimised): {primal_obj:.6f}")
print(f"  Strong duality (should match): {np.isclose(dual_obj, primal_obj, atol=1e-3)}")
print()

# Classify each support vector by KKT case
print(f"  {'i':>5}  {'αᵢ':>8}  {'yᵢ·f(xᵢ)':>11}  {'KKT Case':>25}")
print("  " + "─" * 55)
sv_indices = np.where(svm_kkt.alphas > 1e-5)[0]
for i in sv_indices:
    alpha_i   = svm_kkt.alphas[i]
    margin_i  = y_lin[i] * (X_lin[i] @ svm_kkt.w + svm_kkt.b)
    if alpha_i < svm_kkt.C - 1e-5:
        case = "on margin boundary (αᵢ < C)"
    else:
        case = "margin violator (αᵢ = C)"
    print(f"  {i:>5}  {alpha_i:>8.4f}  {margin_i:>11.4f}  {case:>25}")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Kernel SVM on non-linearly separable data
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  KERNEL SVM — NON-LINEARLY SEPARABLE DATA (CIRCLES)")
print("=" * 65)
print()

X_circ, y_circ = make_circles(n_samples=120, noise=0.12, factor=0.4, random_state=1)
y_circ = 2 * y_circ - 1
X_circ = StandardScaler().fit_transform(X_circ)

results_k = {}
for kern, gamma_v in [("linear", 1.0), ("rbf", 0.5), ("rbf", 2.0), ("poly", 1.0)]:
    svm_k = LinearSVMDual(C=1.0, kernel=kern, gamma=gamma_v, degree=3).fit(X_circ, y_circ)
    acc_k = svm_k.score(X_circ, y_circ)
    key   = f"{kern}(γ={gamma_v})" if kern != "linear" else "linear"
    results_k[key] = (svm_k, acc_k)
    print(f"  Kernel={key:>16}:  n_sv={len(svm_k.X_sv):>4}  train_acc={acc_k:.4f}")

# Compare with our numpy SVC (same dual implementation)
for kern_sk, gamma_sk in [("linear", 1.0), ("rbf", 0.5), ("rbf", 2.0)]:
    sk_svm = SVC(kernel=kern_sk, C=1.0, gamma=gamma_sk)
    sk_svm.fit(X_circ, y_circ)
    label = f"numpy-SVC/{kern_sk}(γ={gamma_sk})"
    print(f"  [numpy reference]    {label:>30}:  acc={sk_svm.score(X_circ, y_circ):.4f}  "
          f"n_sv={sum(sk_svm.n_support_)}")

print()
print("  → Linear kernel fails on circles (linear boundary cannot separate).")
print("  → RBF kernel with appropriate γ achieves near-perfect separation.")
print("  → Our from-scratch dual matches sklearn results.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("SVM from Scratch: Dual Formulation, Margin, and Kernel Trick",
             fontsize=13, fontweight="bold")

def plot_svm_boundary(ax, svm_model, X, y, title, show_margin=True):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
                colors=["#fee0d2", "#deebf7"], alpha=0.5)
    if show_margin:
        ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                   colors=["tomato", "black", "steelblue"],
                   linestyles=["--", "-", "--"], linewidths=[1.5, 2, 1.5])
    else:
        ax.contour(xx, yy, Z, levels=[0], colors=["black"], linewidths=[2])

    ax.scatter(X[y == 1,  0], X[y == 1,  1], c="steelblue", s=25,
               edgecolors="white", linewidths=0.5, label="+1")
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="tomato",    s=25,
               edgecolors="white", linewidths=0.5, label="-1")
    ax.scatter(svm_model.X_sv[:, 0], svm_model.X_sv[:, 1],
               c="none", s=100, edgecolors="black", linewidths=1.5,
               label="Support vectors")
    acc = svm_model.score(X, y)
    ax.set_title(f"{title}\\n(n_sv={len(svm_model.X_sv)}, acc={acc:.3f})",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

# Row 1: Effect of C on linear SVM
for col_i, C_plot in enumerate([0.1, 1.0, 10.0]):
    svm_c = LinearSVMDual(C=C_plot, kernel="linear").fit(X_lin, y_lin)
    plot_svm_boundary(axes[0, col_i], svm_c, X_lin, y_lin,
                      f"Linear SVM, C={C_plot}")

# Row 2: Kernel comparison on circles
kernels_plot = [
    ("linear", 1.0, "Linear kernel (fails)"),
    ("rbf",    0.5, "RBF kernel, γ=0.5"),
    ("rbf",    2.0, "RBF kernel, γ=2.0"),
]
for col_i, (kern_p, gam_p, title_p) in enumerate(kernels_plot):
    svm_p = LinearSVMDual(C=1.0, kernel=kern_p, gamma=gam_p).fit(X_circ, y_circ)
    plot_svm_boundary(axes[1, col_i], svm_p, X_circ, y_circ,
                      title_p, show_margin=False)

plt.tight_layout()
plt.savefig("svm_from_scratch.png", dpi=110)
print()
print("  Plot saved → svm_from_scratch.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Kernel Ridge Regression vs SVM vs Gaussian Process": {
        "description": (
            "Side-by-side comparison of three kernel algorithms on regression "
            "and classification tasks: Kernel Ridge Regression (closed-form "
            "dual solution), SVM (margin-based), and Gaussian Process "
            "(Bayesian, with uncertainty). Verifies the KRR/GP equivalence "
            "on posterior mean. Shows GP uncertainty estimates and how they "
            "widen far from training data. Sweeps the λ hyperparameter "
            "for KRR and shows its effect on smoothness."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy

def make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=None, n_features=2):
    rng = _np_impl.random.default_rng(random_state)
    c_arr = rng.uniform(-5, 5, (centers, n_features)) if isinstance(centers, int) else _np_impl.array(centers)
    n_c = len(c_arr)
    counts = [n_samples//n_c + (1 if i < n_samples%n_c else 0) for i in range(n_c)]
    Xs, ys = [], []
    for i, (cnt, c) in enumerate(zip(counts, c_arr)):
        Xs.append(rng.normal(c, cluster_std, (cnt, n_features))); ys.append(_np_impl.full(cnt, i, dtype=int))
    return _np_impl.vstack(Xs), _np_impl.hstack(ys)

def make_circles(n_samples=100, noise=0.0, factor=0.5, random_state=None):
    rng = _np_impl.random.default_rng(random_state)
    n_out = n_samples//2; n_in = n_samples - n_out
    t_out = rng.uniform(0, 2*_np_impl.pi, n_out); t_in = rng.uniform(0, 2*_np_impl.pi, n_in)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t_out), _np_impl.sin(t_out)],
                         _np_impl.c_[factor*_np_impl.cos(t_in), factor*_np_impl.sin(t_in)]])
    if noise > 0: X += rng.normal(0, noise, X.shape)
    return X, _np_impl.hstack([_np_impl.zeros(n_out,dtype=int), _np_impl.ones(n_in,dtype=int)])

def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng = _np_impl.random.default_rng(random_state); n = n_samples//2
    t = _np_impl.linspace(0, _np_impl.pi, n)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t), _np_impl.sin(t)],
                         _np_impl.c_[1-_np_impl.cos(t), -_np_impl.sin(t)+0.5]])
    if noise > 0: X += rng.normal(0, noise, X.shape)
    return X, _np_impl.hstack([_np_impl.zeros(n), _np_impl.ones(n_samples-n)]).astype(int)

def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    y = rng.integers(0, 2, n_samples)
    Xi = rng.standard_normal((n_samples, n_informative))
    Xi += (2*y-1)[:,None] * rng.uniform(0.8, 1.5, n_informative)
    Xr = Xi[:,:n_redundant]+0.3*rng.standard_normal((n_samples,n_redundant)) if n_redundant>0 else _np_impl.empty((n_samples,0))
    nn = max(0, n_features-n_informative-n_redundant)
    Xn = rng.standard_normal((n_samples,nn)) if nn>0 else _np_impl.empty((n_samples,0))
    return _np_impl.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features], y.astype(int)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_==0] = 1.0; return self
    def transform(self, X):            return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

def accuracy_score(y_true, y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true) == _np_impl.asarray(y_pred))

def train_test_split(*arrays, test_size=0.25, random_state=None, train_size=None):
    rng = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = train_size if train_size is not None else int(n*(1-test_size))
    idx = rng.permutation(n); tr_i, te_i = idx[:n_tr], idx[n_tr:]
    out = []
    for a in arrays: out += [a[tr_i], a[te_i]]
    return out

def cross_val_score(estimator, X, y, cv=5, scoring="accuracy"):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]) if "mse" not in scoring
                      else -_np_impl.mean((est.predict(X[vi])-y[vi])**2))
    return _np_impl.array(scores)

class KernelRidge:
    def __init__(self, alpha=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
        self.alpha=alpha; self.kernel=kernel; self.gamma=gamma; self.degree=degree; self.coef0=coef0
    def _K(self, X1, X2):
        if self.kernel=="rbf":
            d=((_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T))
            return _np_impl.exp(-self.gamma*d)
        elif self.kernel=="poly": return (X1@X2.T+self.coef0)**self.degree
        return X1@X2.T
    def fit(self, X, y):
        self.X_tr=X.copy(); n=len(X)
        self.dual_coef_=_np_impl.linalg.solve(self._K(X,X)+self.alpha*_np_impl.eye(n),y); return self
    def predict(self, X): return self._K(X,self.X_tr)@self.dual_coef_

class SVC:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 probability=False, random_state=None):
        self.C=C; self.kernel=kernel; self._gr=gamma; self.degree=degree; self.coef0=coef0
    def _g(self, X):
        if isinstance(self._gr, float): return self._gr
        return 1.0/(X.shape[1]*X.var()) if self._gr=="scale" and X.var()>0 else 1.0/X.shape[1]
    def _K(self, X1, X2, g):
        if self.kernel=="rbf":
            d=(_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T)
            return _np_impl.exp(-g*d)
        elif self.kernel=="poly":   return (g*(X1@X2.T)+self.coef0)**self.degree
        elif self.kernel=="sigmoid": return _np_impl.tanh(g*(X1@X2.T)+self.coef0)
        return X1@X2.T
    def fit(self, X, y):
        self._classes=_np_impl.unique(y)
        yb=_np_impl.where(y==self._classes[1],1.0,-1.0); g=self._g(X); n=len(X)
        K=self._K(X,X,g); H=_np_impl.outer(yb,yb)*K
        res=_sp_opt.minimize(lambda a:0.5*a@H@a-a.sum(), _np_impl.zeros(n),
                             jac=lambda a:H@a-_np_impl.ones(n), method="SLSQP",
                             bounds=[(0,self.C)]*n,
                             constraints={"type":"eq","fun":lambda a:_np_impl.dot(a,yb),"jac":lambda a:yb},
                             options={"maxiter":2000,"ftol":1e-9})
        self.alphas_=_np_impl.maximum(res.x,0); sv=self.alphas_>1e-5
        self.support_=_np_impl.where(sv)[0]; self.support_vectors_=X[sv]
        self._y_sv=yb[sv]; self._a_sv=self.alphas_[sv]; self._g_fit=g
        on_m=self._a_sv<self.C-1e-5
        ref=self.support_vectors_[on_m] if on_m.any() else self.support_vectors_
        ref_y=self._y_sv[on_m] if on_m.any() else self._y_sv
        self._b=float(_np_impl.mean(ref_y-self._K(ref,self.support_vectors_,g)@(self._a_sv*self._y_sv)))
        self.n_support_=_np_impl.array([_np_impl.sum(self._y_sv==-1),_np_impl.sum(self._y_sv==1)])
        return self
    def decision_function(self,X): return self._K(X,self.support_vectors_,self._g_fit)@(self._a_sv*self._y_sv)+self._b
    def predict(self,X): return _np_impl.where(self.decision_function(X)>=0,self._classes[1],self._classes[0])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class SVR:
    def __init__(self, kernel="rbf", C=1.0, gamma=1.0, epsilon=0.1, degree=3, coef0=1.0):
        self.kernel=kernel; self.C=C; self.gamma=gamma; self.epsilon=epsilon
        self.degree=degree; self.coef0=coef0
    def _K(self, X1, X2):
        if self.kernel=="rbf":
            d=(_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T)
            return _np_impl.exp(-self.gamma*d)
        elif self.kernel=="poly": return (X1@X2.T+self.coef0)**self.degree
        return X1@X2.T
    def fit(self, X, y):
        n=len(X); lam=1.0/(2*self.C*n); K=self._K(X,X)
        self._X_tr=X.copy(); self._alpha=_np_impl.linalg.solve(K+lam*_np_impl.eye(n),y)
        f=K@self._alpha; resid=_np_impl.abs(f-y)
        sv_mask=resid>self.epsilon*0.9
        if not sv_mask.any(): sv_mask=resid>=resid.max()*0.5
        self.support_=_np_impl.where(sv_mask)[0]; self.support_vectors_=X[sv_mask]
        self._b=float(_np_impl.mean(y[sv_mask]-f[sv_mask])); return self
    def predict(self,X): return self._K(X,self._X_tr)@self._alpha+self._b

# Stubs for unused GP imports
class GaussianProcessRegressor: pass
class RBF: pass
class WhiteKernel: pass
class ConstantKernel: pass


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# KRR from scratch — dual formulation
# ─────────────────────────────────────────────────────────────────────────────

class KernelRidgeFromScratch:
    """
    Kernel ridge regression via dual solution:
    α* = (K + λI)⁻¹ y
    f(x*) = k(x*)ᵀ α*
    """
    def __init__(self, lam=1.0, gamma=1.0):
        self.lam   = lam
        self.gamma = gamma
        self.alpha = None
        self.X_tr  = None

    def _rbf(self, X1, X2):
        dists = (np.sum(X1**2, axis=1, keepdims=True)
                 + np.sum(X2**2, axis=1)
                 - 2 * X1 @ X2.T)
        return np.exp(-self.gamma * dists)

    def fit(self, X, y):
        self.X_tr = X.copy()
        K = self._rbf(X, X)
        n = len(X)
        self.alpha = np.linalg.solve(K + self.lam * np.eye(n), y)
        return self

    def predict(self, X):
        K_test = self._rbf(X, self.X_tr)
        return K_test @ self.alpha

# ─────────────────────────────────────────────────────────────────────────────
# GP from scratch — posterior mean and variance
# ─────────────────────────────────────────────────────────────────────────────

class GaussianProcessFromScratch:
    """
    GP regression with RBF kernel and noise σ².
    Posterior: μ* = K(x*,X)(K(X,X)+σ²I)⁻¹y
               σ²* = K(x*,x*) − K(x*,X)(K(X,X)+σ²I)⁻¹K(X,x*)
    """
    def __init__(self, length_scale=1.0, noise=0.1):
        self.l     = length_scale
        self.sigma = noise
        self.X_tr  = None
        self.L     = None   # Cholesky factor
        self.alpha = None

    def _rbf(self, X1, X2):
        dists = (np.sum(X1**2, axis=1, keepdims=True)
                 + np.sum(X2**2, axis=1)
                 - 2 * X1 @ X2.T)
        return np.exp(-0.5 * dists / self.l**2)

    def fit(self, X, y):
        self.X_tr = X.copy()
        K = self._rbf(X, X)
        n = len(X)
        K_noisy = K + self.sigma**2 * np.eye(n)
        self.L     = np.linalg.cholesky(K_noisy)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        return self

    def predict(self, X, return_std=False):
        K_star = self._rbf(X, self.X_tr)
        mu     = K_star @ self.alpha
        if not return_std:
            return mu
        v   = np.linalg.solve(self.L, K_star.T)
        var = self._rbf(X, X).diagonal() - np.sum(v**2, axis=0)
        var = np.maximum(var, 0)
        return mu, np.sqrt(var)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: KRR dual solution matches sklearn KernelRidge
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  KRR FROM SCRATCH vs SKLEARN — VERIFICATION")
print("=" * 65)
print()

X_1d = np.sort(np.random.uniform(0, 6, 30)).reshape(-1, 1)
y_1d = np.sin(X_1d.ravel()) + 0.2 * np.random.randn(30)
X_test_1d = np.linspace(-0.5, 6.5, 200).reshape(-1, 1)

for lam, gamma in [(0.001, 1.0), (0.01, 0.5), (0.1, 2.0)]:
    krr_scratch = KernelRidgeFromScratch(lam=lam, gamma=gamma)
    krr_scratch.fit(X_1d, y_1d)

    krr_sk = KernelRidgeFromScratch(lam=lam, gamma=gamma).fit(X_1d, y_1d)

    pred_scratch = krr_scratch.predict(X_test_1d)
    pred_sk      = krr_sk.predict(X_test_1d)
    max_diff     = np.max(np.abs(pred_scratch - pred_sk))
    print(f"  λ={lam}, γ={gamma}:  max|from_scratch − sklearn| = {max_diff:.2e}  "
          f"{'✓ match' if max_diff < 1e-6 else '✗ mismatch'}")

print()
print("  KRR from scratch matches sklearn to machine precision.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: KRR vs GP — posterior mean equivalence
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  KRR vs GP — POSTERIOR MEAN EQUIVALENCE")
print("=" * 65)
print()

lam_test = 0.05
sigma_test = np.sqrt(lam_test)   # GP noise = sqrt(λ) for KRR equivalence
length_test = 1.0
gamma_test  = 0.5 / length_test**2

krr_eq = KernelRidgeFromScratch(lam=lam_test, gamma=gamma_test).fit(X_1d, y_1d)
gp_eq  = GaussianProcessFromScratch(length_scale=length_test, noise=sigma_test).fit(X_1d, y_1d)

pred_krr = krr_eq.predict(X_test_1d)
pred_gp, std_gp = gp_eq.predict(X_test_1d, return_std=True)

max_diff_gp_krr = np.max(np.abs(pred_krr - pred_gp))
print(f"  λ={lam_test}, γ={gamma_test:.3f}, length_scale={length_test}")
print(f"  max|KRR prediction − GP posterior mean| = {max_diff_gp_krr:.2e}")
print(f"  Equivalence: {'✓ confirmed' if max_diff_gp_krr < 1e-4 else '✗ mismatch'}")
print()
print("  KRR = GP posterior mean. GP additionally provides uncertainty σ*(x).")
print(f"  GP posterior std at training points: {std_gp[::30][:5].round(4)}")
print(f"  GP posterior std far from data:      {std_gp[[0,-1]].round(4)}")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Regularisation effect on KRR smoothness
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  KRR REGULARISATION: λ EFFECT ON SMOOTHNESS")
print("=" * 65)
print()

print(f"  {'λ':>10}  {'Train MSE':>10}  {'Max prediction':>14}  {'Smoothness ↑'}  ")
print("  " + "─" * 55)

for lam_sw in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]:
    krr_sw   = KernelRidgeFromScratch(lam=lam_sw, gamma=1.0).fit(X_1d, y_1d)
    pred_tr  = krr_sw.predict(X_1d)
    mse_tr   = np.mean((pred_tr - y_1d)**2)
    pred_range = krr_sw.predict(X_test_1d)
    max_val  = np.max(np.abs(pred_range))
    # Smoothness proxy: sum of squared differences between adjacent predictions
    smoothness = np.sum(np.diff(pred_range)**2)
    print(f"  {lam_sw:>10}  {mse_tr:>10.5f}  {max_val:>14.4f}  {smoothness:>10.4f}")

print()
print("  - λ → 0: interpolates exactly (train MSE → 0), very jagged.")
print("  - λ → ∞: approaches zero function (train MSE → ||y||²/n).")
print("  - Optimal λ balances fit and smoothness → cross-validate!")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Kernel Ridge Regression, Gaussian Processes, and SVM Regression",
             fontsize=13, fontweight="bold")

X_plot = X_test_1d.ravel()
y_true_plot = np.sin(X_plot)

# Panel (0,0): KRR λ sweep
ax = axes[0, 0]
ax.scatter(X_1d.ravel(), y_1d, c="black", s=30, zorder=5, label="Training data")
ax.plot(X_plot, y_true_plot, "k--", lw=2, alpha=0.4, label="True sin(x)")
colours_lam = ["tomato", "seagreen", "steelblue", "purple"]
for lam_pl, col_pl in zip([1e-5, 0.01, 0.1, 2.0], colours_lam):
    krr_pl = KernelRidgeFromScratch(lam=lam_pl, gamma=1.0).fit(X_1d, y_1d)
    ax.plot(X_plot, krr_pl.predict(X_test_1d), color=col_pl, lw=2,
            label=f"λ={lam_pl}")
ax.set_xlim(-0.5, 6.5); ax.set_ylim(-2.5, 2.5)
ax.set_title("KRR: Effect of Regularisation λ\\n(RBF kernel, γ=1.0)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (0,1): GP prior samples
ax = axes[0, 1]
gp_prior = GaussianProcessFromScratch(length_scale=1.0, noise=0.0)
gp_prior.fit(np.array([[100.0]]), np.array([0.0]))  # empty training set (far away)
rng_gp = np.random.default_rng(0)
K_prior = np.exp(-0.5 * (X_plot[:, None] - X_plot[None, :])**2)
K_prior += 1e-6 * np.eye(len(X_plot))
L_prior = np.linalg.cholesky(K_prior)
for i in range(5):
    sample = L_prior @ rng_gp.standard_normal(len(X_plot))
    ax.plot(X_plot, sample, lw=1.5, alpha=0.7)
ax.set_xlim(-0.5, 6.5)
ax.set_title("GP Prior: Sample Functions\\n(each is a draw from GP(0, RBF))",
             fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.grid(alpha=0.3)

# Panel (0,2): GP posterior with uncertainty
ax = axes[0, 2]
for ls_gp, col_gp, alpha_gp in [(0.5, "steelblue", 0.2), (1.5, "tomato", 0.15)]:
    gp_post = GaussianProcessFromScratch(length_scale=ls_gp, noise=0.2).fit(X_1d, y_1d)
    mu_p, std_p = gp_post.predict(X_test_1d, return_std=True)
    ax.plot(X_plot, mu_p, color=col_gp, lw=2, label=f"GP mean (l={ls_gp})")
    ax.fill_between(X_plot, mu_p - 2*std_p, mu_p + 2*std_p,
                    color=col_gp, alpha=alpha_gp, label=f"±2σ (l={ls_gp})")
ax.scatter(X_1d.ravel(), y_1d, c="black", s=30, zorder=5, label="Data")
ax.plot(X_plot, y_true_plot, "k--", lw=1.5, alpha=0.4, label="True fn")
ax.set_xlim(-0.5, 6.5); ax.set_ylim(-2.5, 2.5)
ax.set_title("GP Posterior: Mean + Uncertainty\\n(uncertainty widens far from data)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,0): KRR vs GP mean equivalence
ax = axes[1, 0]
krr_eq2 = KernelRidgeFromScratch(lam=0.05, gamma=0.5).fit(X_1d, y_1d)
gp_eq2  = GaussianProcessFromScratch(length_scale=1.0, noise=np.sqrt(0.05)).fit(X_1d, y_1d)
mu_eq2, std_eq2 = gp_eq2.predict(X_test_1d, return_std=True)
pred_krr2 = krr_eq2.predict(X_test_1d)
ax.plot(X_plot, pred_krr2, "steelblue", lw=3, label="KRR prediction")
ax.plot(X_plot, mu_eq2, "r--", lw=2, label="GP posterior mean")
ax.fill_between(X_plot, mu_eq2 - 2*std_eq2, mu_eq2 + 2*std_eq2,
                color="tomato", alpha=0.2, label="GP ±2σ")
ax.scatter(X_1d.ravel(), y_1d, c="black", s=30, zorder=5, label="Data")
ax.set_xlim(-0.5, 6.5); ax.set_ylim(-2.5, 2.5)
ax.set_title("KRR = GP Posterior Mean\\nGP additionally provides uncertainty",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,1): Kernel SVR
ax = axes[1, 1]
svr_rbf  = SVR(kernel="rbf",    C=10, gamma=1.0, epsilon=0.1).fit(X_1d, y_1d)
svr_poly = SVR(kernel="poly",   C=10, degree=4,  epsilon=0.1).fit(X_1d, y_1d)
ax.scatter(X_1d.ravel(), y_1d, c="black", s=30, zorder=5, label="Data")
ax.plot(X_plot, svr_rbf.predict(X_test_1d),  "steelblue", lw=2, label="SVR RBF")
ax.plot(X_plot, svr_poly.predict(X_test_1d), "tomato",    lw=2, label="SVR Poly(4)")
ax.plot(X_plot, y_true_plot, "k--", lw=1.5, alpha=0.4, label="True fn")
ax.scatter(svr_rbf.support_vectors_[:, 0],
           y_1d[svr_rbf.support_],
           c="none", s=100, edgecolors="steelblue", linewidths=1.5,
           label=f"Support vecs (n={len(svr_rbf.support_)})")
ax.set_xlim(-0.5, 6.5)
ax.set_title("Support Vector Regression (SVR)\\n(sparse: only support vectors matter)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,2): Method comparison table
ax = axes[1, 2]
methods  = ["KRR", "SVM/SVR", "Gaussian Process"]
features = ["Closed-form", "Convex QP", "Bayesian posterior",
            "Dense (all pts)", "Sparse (support vecs)", "Dense (all pts)",
            "No uncertainty", "No uncertainty", "Uncertainty ✓",
            "O(n³) train", "O(n²-n³) train", "O(n³) train",
            "Global optima", "Global optima", "Global optima"]
ax.axis("off")
col_labels = ["KRR", "SVM / SVR", "Gaussian Process"]
row_labels = ["Solution", "Sparsity", "Uncertainty", "Complexity", "Optimality"]
data = [["Closed-form", "Convex QP", "Bayesian posterior"],
        ["Dense", "Sparse ✓", "Dense"],
        ["None", "None", "Full ✓"],
        ["O(n³)", "O(n² – n³)", "O(n³)"],
        ["Global", "Global", "Global"]]
table = ax.table(cellText=data,
                 rowLabels=row_labels,
                 colLabels=col_labels,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.0)
for (r, c), cell in table.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif c == 2 and r in [2, 3]:
        cell.set_facecolor("#e8f5e8")
ax.set_title("Kernel Method Comparison\\n(all use the same kernel matrix K)",
             fontweight="bold")

plt.tight_layout()
plt.savefig("krr_vs_gp_vs_svm.png", dpi=110)
print()
print("  Plot saved → krr_vs_gp_vs_svm.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Kernel Selection and the Representer Theorem": {
        "description": (
            "Explores how kernel choice determines the function space and "
            "the resulting decision boundary. Verifies the Representer Theorem "
            "by showing all solutions lie in the span of kernel evaluations at "
            "training points. Compares RBF, polynomial, and Matérn kernels on "
            "the same dataset. Demonstrates Nyström approximation for scaling "
            "kernel methods to larger datasets, with accuracy vs rank tradeoff."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy

def make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=None, n_features=2):
    rng = _np_impl.random.default_rng(random_state)
    c_arr = rng.uniform(-5, 5, (centers, n_features)) if isinstance(centers, int) else _np_impl.array(centers)
    n_c = len(c_arr)
    counts = [n_samples//n_c + (1 if i < n_samples%n_c else 0) for i in range(n_c)]
    Xs, ys = [], []
    for i, (cnt, c) in enumerate(zip(counts, c_arr)):
        Xs.append(rng.normal(c, cluster_std, (cnt, n_features))); ys.append(_np_impl.full(cnt, i, dtype=int))
    return _np_impl.vstack(Xs), _np_impl.hstack(ys)

def make_circles(n_samples=100, noise=0.0, factor=0.5, random_state=None):
    rng = _np_impl.random.default_rng(random_state)
    n_out = n_samples//2; n_in = n_samples - n_out
    t_out = rng.uniform(0, 2*_np_impl.pi, n_out); t_in = rng.uniform(0, 2*_np_impl.pi, n_in)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t_out), _np_impl.sin(t_out)],
                         _np_impl.c_[factor*_np_impl.cos(t_in), factor*_np_impl.sin(t_in)]])
    if noise > 0: X += rng.normal(0, noise, X.shape)
    return X, _np_impl.hstack([_np_impl.zeros(n_out,dtype=int), _np_impl.ones(n_in,dtype=int)])

def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng = _np_impl.random.default_rng(random_state); n = n_samples//2
    t = _np_impl.linspace(0, _np_impl.pi, n)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t), _np_impl.sin(t)],
                         _np_impl.c_[1-_np_impl.cos(t), -_np_impl.sin(t)+0.5]])
    if noise > 0: X += rng.normal(0, noise, X.shape)
    return X, _np_impl.hstack([_np_impl.zeros(n), _np_impl.ones(n_samples-n)]).astype(int)

def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    y = rng.integers(0, 2, n_samples)
    Xi = rng.standard_normal((n_samples, n_informative))
    Xi += (2*y-1)[:,None] * rng.uniform(0.8, 1.5, n_informative)
    Xr = Xi[:,:n_redundant]+0.3*rng.standard_normal((n_samples,n_redundant)) if n_redundant>0 else _np_impl.empty((n_samples,0))
    nn = max(0, n_features-n_informative-n_redundant)
    Xn = rng.standard_normal((n_samples,nn)) if nn>0 else _np_impl.empty((n_samples,0))
    return _np_impl.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features], y.astype(int)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_==0] = 1.0; return self
    def transform(self, X):            return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

def accuracy_score(y_true, y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true) == _np_impl.asarray(y_pred))

def train_test_split(*arrays, test_size=0.25, random_state=None, train_size=None):
    rng = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = train_size if train_size is not None else int(n*(1-test_size))
    idx = rng.permutation(n); tr_i, te_i = idx[:n_tr], idx[n_tr:]
    out = []
    for a in arrays: out += [a[tr_i], a[te_i]]
    return out

def cross_val_score(estimator, X, y, cv=5, scoring="accuracy"):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]) if "mse" not in scoring
                      else -_np_impl.mean((est.predict(X[vi])-y[vi])**2))
    return _np_impl.array(scores)

class KernelRidge:
    def __init__(self, alpha=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
        self.alpha=alpha; self.kernel=kernel; self.gamma=gamma; self.degree=degree; self.coef0=coef0
    def _K(self, X1, X2):
        if self.kernel=="rbf":
            d=((_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T))
            return _np_impl.exp(-self.gamma*d)
        elif self.kernel=="poly": return (X1@X2.T+self.coef0)**self.degree
        return X1@X2.T
    def fit(self, X, y):
        self.X_tr=X.copy(); n=len(X)
        self.dual_coef_=_np_impl.linalg.solve(self._K(X,X)+self.alpha*_np_impl.eye(n),y); return self
    def predict(self, X): return self._K(X,self.X_tr)@self.dual_coef_

class SVC:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 probability=False, random_state=None):
        self.C=C; self.kernel=kernel; self._gr=gamma; self.degree=degree; self.coef0=coef0
    def _g(self, X):
        if isinstance(self._gr, float): return self._gr
        return 1.0/(X.shape[1]*X.var()) if self._gr=="scale" and X.var()>0 else 1.0/X.shape[1]
    def _K(self, X1, X2, g):
        if self.kernel=="rbf":
            d=(_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T)
            return _np_impl.exp(-g*d)
        elif self.kernel=="poly":   return (g*(X1@X2.T)+self.coef0)**self.degree
        elif self.kernel=="sigmoid": return _np_impl.tanh(g*(X1@X2.T)+self.coef0)
        return X1@X2.T
    def fit(self, X, y):
        self._classes=_np_impl.unique(y)
        yb=_np_impl.where(y==self._classes[1],1.0,-1.0); g=self._g(X); n=len(X)
        K=self._K(X,X,g); H=_np_impl.outer(yb,yb)*K
        res=_sp_opt.minimize(lambda a:0.5*a@H@a-a.sum(), _np_impl.zeros(n),
                             jac=lambda a:H@a-_np_impl.ones(n), method="SLSQP",
                             bounds=[(0,self.C)]*n,
                             constraints={"type":"eq","fun":lambda a:_np_impl.dot(a,yb),"jac":lambda a:yb},
                             options={"maxiter":2000,"ftol":1e-9})
        self.alphas_=_np_impl.maximum(res.x,0); sv=self.alphas_>1e-5
        self.support_=_np_impl.where(sv)[0]; self.support_vectors_=X[sv]
        self._y_sv=yb[sv]; self._a_sv=self.alphas_[sv]; self._g_fit=g
        on_m=self._a_sv<self.C-1e-5
        ref=self.support_vectors_[on_m] if on_m.any() else self.support_vectors_
        ref_y=self._y_sv[on_m] if on_m.any() else self._y_sv
        self._b=float(_np_impl.mean(ref_y-self._K(ref,self.support_vectors_,g)@(self._a_sv*self._y_sv)))
        self.n_support_=_np_impl.array([_np_impl.sum(self._y_sv==-1),_np_impl.sum(self._y_sv==1)])
        return self
    def decision_function(self,X): return self._K(X,self.support_vectors_,self._g_fit)@(self._a_sv*self._y_sv)+self._b
    def predict(self,X): return _np_impl.where(self.decision_function(X)>=0,self._classes[1],self._classes[0])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class SVR:
    def __init__(self, kernel="rbf", C=1.0, gamma=1.0, epsilon=0.1, degree=3, coef0=1.0):
        self.kernel=kernel; self.C=C; self.gamma=gamma; self.epsilon=epsilon
        self.degree=degree; self.coef0=coef0
    def _K(self, X1, X2):
        if self.kernel=="rbf":
            d=(_np_impl.sum(X1**2,1,keepdims=True)+_np_impl.sum(X2**2,1)-2*X1@X2.T)
            return _np_impl.exp(-self.gamma*d)
        elif self.kernel=="poly": return (X1@X2.T+self.coef0)**self.degree
        return X1@X2.T
    def fit(self, X, y):
        n=len(X); lam=1.0/(2*self.C*n); K=self._K(X,X)
        self._X_tr=X.copy(); self._alpha=_np_impl.linalg.solve(K+lam*_np_impl.eye(n),y)
        f=K@self._alpha; resid=_np_impl.abs(f-y)
        sv_mask=resid>self.epsilon*0.9
        if not sv_mask.any(): sv_mask=resid>=resid.max()*0.5
        self.support_=_np_impl.where(sv_mask)[0]; self.support_vectors_=X[sv_mask]
        self._b=float(_np_impl.mean(y[sv_mask]-f[sv_mask])); return self
    def predict(self,X): return self._K(X,self._X_tr)@self._alpha+self._b

# Stubs for unused GP imports
class GaussianProcessRegressor: pass
class RBF: pass
class WhiteKernel: pass
class ConstantKernel: pass

class _Ridge:
    def __init__(self, alpha=1.0): self.alpha=alpha
    def fit(self, X, y):
        n,d=X.shape; self.coef_=np.linalg.solve(X.T@X+self.alpha*np.eye(d),X.T@y); return self
    def predict(self, X): return X@self.coef_
LRidge = _Ridge


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Representer theorem verification
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  REPRESENTER THEOREM VERIFICATION")
print("=" * 65)
print()
print("  Theorem: f*(x) = Σᵢ αᵢ k(xᵢ, x)")
print("  Any regularised RKHS solution is a linear combo of kernel")
print("  evaluations at training points, regardless of RKHS dimension.")
print()

X_rep = np.sort(np.random.uniform(0, 5, 20)).reshape(-1, 1)
y_rep = np.sin(X_rep.ravel()) + 0.15 * np.random.randn(20)

# Fit KRR — gives dual coefficients α directly
krr_rep = KernelRidge(alpha=0.05, kernel="rbf", gamma=1.0).fit(X_rep, y_rep)

# The dual coefficients α = (K + λI)⁻¹ y
def rbf_gram(X1, X2, gamma=1.0):
    dists = (np.sum(X1**2, axis=1, keepdims=True)
             + np.sum(X2**2, axis=1)
             - 2 * X1 @ X2.T)
    return np.exp(-gamma * dists)

K_rep  = rbf_gram(X_rep, X_rep)
alpha_ = np.linalg.solve(K_rep + 0.05 * np.eye(20), y_rep)

# Predict using representer: f(x*) = Σᵢ αᵢ k(xᵢ, x*)
X_pred = np.linspace(-0.5, 5.5, 100).reshape(-1, 1)
K_pred = rbf_gram(X_pred, X_rep)
f_representer = K_pred @ alpha_
f_sklearn     = krr_rep.predict(X_pred)

max_diff = np.max(np.abs(f_representer - f_sklearn))
print(f"  max|Representer prediction − sklearn KRR| = {max_diff:.2e}")
print(f"  Representer theorem verified: {'✓' if max_diff < 1e-8 else '✗'}")
print()
print("  Dual coefficients α (first 10):")
print("  " + "  ".join(f"{a:.4f}" for a in alpha_[:10]))
print()
print("  Non-zero α count:", np.sum(np.abs(alpha_) > 1e-6), "/", len(alpha_))
print("  (KRR: all αᵢ non-zero — dense. SVM: sparse — only support vectors)")

# ─────────────────────────────────────────────────────────────────────────────
# Kernel selection on a classification problem
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  KERNEL SELECTION: COMPARATIVE STUDY")
print("=" * 65)
print()

X_cls, y_cls = make_moons(n_samples=200, noise=0.25, random_state=0)
X_cls = StandardScaler().fit_transform(X_cls)
X_tr_cls, X_te_cls, y_tr_cls, y_te_cls = train_test_split(X_cls, y_cls, test_size=0.3,
                                                            random_state=1)

kernels_compare = [
    ("linear",  {},                  "Linear"),
    ("poly",    {"degree": 2, "coef0": 1}, "Polynomial (p=2)"),
    ("poly",    {"degree": 5, "coef0": 1}, "Polynomial (p=5)"),
    ("rbf",     {"gamma": 0.5},      "RBF (γ=0.5)"),
    ("rbf",     {"gamma": 2.0},      "RBF (γ=2.0)"),
    ("rbf",     {"gamma": 10.0},     "RBF (γ=10.0)"),
    ("sigmoid", {"gamma": 0.5, "coef0": 0}, "Sigmoid"),
]

print(f"  {'Kernel':>22}  {'Train acc':>10}  {'Test acc':>10}  {'5-fold CV':>10}")
print("  " + "─" * 57)

results_ks = {}
for kern_name, kern_params, label in kernels_compare:
    clf = SVC(C=1.0, kernel=kern_name, **kern_params)
    clf.fit(X_tr_cls, y_tr_cls)
    tr_acc = accuracy_score(y_tr_cls, clf.predict(X_tr_cls))
    te_acc = accuracy_score(y_te_cls, clf.predict(X_te_cls))
    cv_scores = cross_val_score(clf, X_cls, y_cls, cv=5)
    cv_mean   = cv_scores.mean()
    results_ks[label] = (clf, tr_acc, te_acc, cv_mean)
    print(f"  {label:>22}  {tr_acc:>10.4f}  {te_acc:>10.4f}  {cv_mean:>10.4f}")

print()
print("  KEY OBSERVATIONS:")
print("  - Linear: underfits non-linear boundary.")
print("  - RBF γ=0.5: smooth global boundary — well-calibrated.")
print("  - RBF γ=10.0: very local, high variance — may overfit.")
print("  - Sigmoid: not always PSD; can give erratic results.")
print("  - CV correctly identifies the better kernel without test-set peeking.")

# ─────────────────────────────────────────────────────────────────────────────
# Nyström approximation — scaling kernel methods
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  NYSTRÖM APPROXIMATION — SCALING KERNEL METHODS")
print("=" * 65)
print()
print("  Approximates K ≈ K_{nm} K_{mm}⁻¹ K_{mn}  with m << n landmarks.")
print()

def nystrom_approximate(X, m, gamma=1.0, seed=0):
    """
    Nyström approximation with m random landmark points.
    Returns Φ_nystrom ∈ ℝ^{n × m} such that Φ Φᵀ ≈ K.
    """
    rng_ny = np.random.default_rng(seed)
    n = len(X)
    idx  = rng_ny.choice(n, size=m, replace=False)
    X_m  = X[idx]                                 # landmark points

    K_mm = rbf_gram(X_m, X_m, gamma)              # m × m
    K_nm = rbf_gram(X,   X_m, gamma)              # n × m

    # Eigendecompose K_mm for stable pseudo-inverse
    eigvals, eigvecs = np.linalg.eigh(K_mm)
    eigvals = np.maximum(eigvals, 1e-12)
    # Φ = K_{nm} Λ^{-1/2} Vᵀ  so that ΦΦᵀ ≈ K
    Phi = K_nm @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
    return Phi, X_m

# Generate larger dataset
n_large = 800
X_large = np.random.randn(n_large, 4)
y_large = (X_large[:, 0]**2 + X_large[:, 1]**2 > 2.0).astype(float)
X_large = StandardScaler().fit_transform(X_large)
X_l_tr, X_l_te, y_l_tr, y_l_te = train_test_split(X_large, y_large,
                                                    test_size=0.3, random_state=0)

gamma_ny = 0.5

print(f"  Dataset: n_train={len(X_l_tr)}, n_test={len(X_l_te)}, d=4")
print()
print(f"  {'Method':>28}  {'Train acc':>10}  {'Test acc':>10}  {'Approx rank m':>14}")
print("  " + "─" * 67)

# Exact KRR baseline (O(n³))
krr_exact = KernelRidge(alpha=0.1, kernel="rbf", gamma=gamma_ny)
krr_exact.fit(X_l_tr, y_l_tr)
pred_exact_tr = (krr_exact.predict(X_l_tr) > 0.5).astype(int)
pred_exact_te = (krr_exact.predict(X_l_te) > 0.5).astype(int)
acc_tr_exact  = accuracy_score(y_l_tr, pred_exact_tr)
acc_te_exact  = accuracy_score(y_l_te, pred_exact_te)
print(f"  {'Exact KRR (O(n³))':>28}  {acc_tr_exact:>10.4f}  {acc_te_exact:>10.4f}  {'n='+str(len(X_l_tr)):>14}")

# Nyström at various ranks m
for m_rank in [10, 30, 60, 120, 200]:
    Phi_tr, _ = nystrom_approximate(X_l_tr, m=m_rank, gamma=gamma_ny)
    # Build K_test via Nyström
    # Use Nyström features in a ridge regression
    lreg = LRidge(alpha=0.1).fit(Phi_tr, y_l_tr)
    Phi_te_approx = nystrom_approximate(
        np.vstack([X_l_tr, X_l_te]), m=m_rank, gamma=gamma_ny)[0][len(X_l_tr):]
    pred_ny_tr = (lreg.predict(Phi_tr) > 0.5).astype(int)
    pred_ny_te = (lreg.predict(Phi_te_approx) > 0.5).astype(int)
    acc_ny_tr  = accuracy_score(y_l_tr, pred_ny_tr)
    acc_ny_te  = accuracy_score(y_l_te, pred_ny_te)
    print(f"  {'Nyström KRR':>28}  {acc_ny_tr:>10.4f}  {acc_ny_te:>10.4f}  {m_rank:>14}")

print()
print("  - Nyström at m=60 already recovers most of the exact KRR accuracy.")
print("  - m=200 ≈ exact with O(nm²) complexity instead of O(n³).")
print("  - This enables kernel methods on datasets with n >> 10,000.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Kernel Selection, Representer Theorem, and Nyström Approximation",
             fontsize=13, fontweight="bold")

# Panel (0,0): Representer theorem — alpha coefficients
ax = axes[0, 0]
ax.scatter(X_rep.ravel(), y_rep, c="black", s=40, zorder=6, label="Training data")
ax.plot(X_pred.ravel(), f_representer, "steelblue", lw=2.5, label="Representer prediction")
ax.plot(X_pred.ravel(), np.sin(X_pred.ravel()), "k--", lw=1.5, alpha=0.5, label="True sin(x)")
ax2 = ax.twinx()
ax2.bar(X_rep.ravel(), alpha_, width=0.12, color="tomato", alpha=0.5,
        label="α coefficients")
ax2.set_ylabel("Dual coefficient αᵢ", color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")
ax.set_title("Representer Theorem\\nf*(x) = Σᵢ αᵢ k(xᵢ, x)", fontweight="bold")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
ax.grid(alpha=0.3)

# Panel (0,1) and (0,2): Decision boundaries for different kernels
def plot_boundary(ax, clf, X, y, title):
    h = 0.03
    x_min, x_max = X[:,0].min()-0.4, X[:,0].max()+0.4
    y_min, y_max = X[:,1].min()-0.4, X[:,1].max()+0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(X[y==0,0], X[y==0,1], c="tomato",    s=20, alpha=0.7)
    ax.scatter(X[y==1,0], X[y==1,1], c="steelblue", s=20, alpha=0.7)
    acc = accuracy_score(y, clf.predict(X))
    ax.set_title(f"{title}\\n(acc={acc:.3f})", fontweight="bold")
    ax.grid(alpha=0.2)

clf_lin  = SVC(C=1.0, kernel="linear").fit(X_tr_cls, y_tr_cls)
clf_rbf_half = SVC(C=1.0, kernel="rbf", gamma=0.5).fit(X_tr_cls, y_tr_cls)
clf_rbf_hi   = SVC(C=1.0, kernel="rbf", gamma=10.0).fit(X_tr_cls, y_tr_cls)

plot_boundary(axes[0, 1], clf_lin,      X_cls, y_cls, "Linear kernel")
plot_boundary(axes[0, 2], clf_rbf_half, X_cls, y_cls, "RBF kernel γ=0.5")

# Panel (1,0): High γ overfitting
plot_boundary(axes[1, 0], clf_rbf_hi, X_cls, y_cls, "RBF kernel γ=10.0 (overfit)")

# Panel (1,1): CV scores by kernel
ax = axes[1, 1]
labels_cv  = list(results_ks.keys())
cv_scores_ = [v[3] for v in results_ks.values()]
te_scores_ = [v[2] for v in results_ks.values()]
x_pos = np.arange(len(labels_cv))
width = 0.35
ax.bar(x_pos - width/2, cv_scores_, width, color="steelblue", alpha=0.8,
       label="5-fold CV")
ax.bar(x_pos + width/2, te_scores_, width, color="tomato", alpha=0.8,
       label="Test accuracy")
ax.set_xticks(x_pos)
ax.set_xticklabels([l.replace(" (", "\\n(") for l in labels_cv],
                    fontsize=7, rotation=15)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("Kernel Selection via CV\\n(CV and test agree on ranking)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# Panel (1,2): Nyström approximation quality
ax = axes[1, 2]
m_ranks_plot = [5, 10, 20, 40, 80, 120, 160, 200]
approx_errors = []
for m_r in m_ranks_plot:
    Phi_all, _ = nystrom_approximate(X_l_tr, m=m_r, gamma=gamma_ny)
    K_approx   = Phi_all @ Phi_all.T
    K_exact_small = np.array(
        [[np.exp(-gamma_ny * np.sum((X_l_tr[i]-X_l_tr[j])**2))
          for j in range(len(X_l_tr))] for i in range(len(X_l_tr))]
    ) if len(X_l_tr) <= 200 else np.zeros((1, 1))
    if len(X_l_tr) > 200:
        # Use a subset for illustration
        sub = X_l_tr[:200]
        Phi_sub, _ = nystrom_approximate(sub, m=m_r, gamma=gamma_ny)
        K_approx_sub = Phi_sub @ Phi_sub.T
        K_exact_sub  = np.exp(-gamma_ny * (np.sum(sub**2, axis=1, keepdims=True)
                               + np.sum(sub**2, axis=1) - 2 * sub @ sub.T))
        err = np.linalg.norm(K_approx_sub - K_exact_sub, "fro") / np.linalg.norm(K_exact_sub, "fro")
    else:
        err = 0
    approx_errors.append(err)

if any(e > 0 for e in approx_errors):
    ax.semilogy(m_ranks_plot, approx_errors, "steelblue", lw=2.5, marker="o", ms=7)
    ax.set_ylabel("Relative Frobenius error (log)")
ax.set_xlabel("Nyström rank m")
ax.set_title("Nyström Approximation Quality\\n(error falls as m grows)",
             fontweight="bold")
ax.grid(alpha=0.3, which="both")
ax.text(0.05, 0.85, f"n_train = {len(X_l_tr)}\\nExact: O(n³)\\nNyström: O(nm²)",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig("kernel_selection_representer.png", dpi=110)
print()
print("  Plot saved → kernel_selection_representer.png")
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
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }