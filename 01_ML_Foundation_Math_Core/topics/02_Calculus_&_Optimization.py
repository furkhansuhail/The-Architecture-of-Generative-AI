"""
Calculus & Optimization
=======================

The engine of learning. Every time a neural network improves, a gradient
flows backward and parameters shift in the direction of steepest descent.
Every regulariser imposes a geometric constraint. Every convergence proof
rests on continuity and convexity arguments.
This module builds the complete theoretical foundation — from limits and
derivatives to KKT conditions and adaptive optimisers.

"""

import textwrap
import re

TOPIC_NAME   = "Calculus & Optimization"
DISPLAY_NAME = "02 · Calculus & Optimization"
ICON         = "∂"
SUBTITLE     = "Derivatives, Convexity, Gradients — The Engine of Learning"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — LIMITS, CONTINUITY & DIFFERENTIABILITY

### Limits — The Foundation

The limit is the central concept of calculus. Informally:

    lim_{x→a} f(x) = L

means f(x) can be made arbitrarily close to L by taking x sufficiently
close (but not equal) to a.

Formal ε-δ definition:
    ∀ ε > 0, ∃ δ > 0 such that 0 < |x − a| < δ → |f(x) − L| < ε

This definition never refers to f(a) itself — limits are about the
APPROACH, not the value at the point.

Key limit laws (assume lim_{x→a} f(x) = L, lim_{x→a} g(x) = M):
    Sum:       lim [f+g] = L + M
    Product:   lim [fg]  = LM
    Quotient:  lim [f/g] = L/M  (if M ≠ 0)
    Chain:     lim h(f(x)) = h(L)  (if h is continuous at L)

Important limits in ML:
    lim_{n→∞} (1 + x/n)ⁿ = eˣ          (exponential)
    lim_{x→0} sin(x)/x = 1              (sinc at origin)
    lim_{x→−∞} σ(x) = 0,  lim_{x→+∞} σ(x) = 1  (sigmoid saturation)


### Continuity

f is CONTINUOUS at a if:
    1. f(a) is defined
    2. lim_{x→a} f(x) exists
    3. lim_{x→a} f(x) = f(a)

Geometrically: the graph has no holes, jumps, or vertical asymptotes at a.

Continuity classes:
    C⁰: continuous (no jumps)
    C¹: continuously differentiable (smooth curves)
    C²: twice continuously differentiable (Hessian exists and is continuous)
    C∞: infinitely differentiable (smooth) — polynomials, eˣ, sin, cos
    Cᵒ: analytic — has convergent Taylor series

In ML, activation functions vary:
    ReLU:    C⁰ (not differentiable at 0)
    ELU:     C¹ (smooth at 0)
    GELU:    C∞ (uses erf, infinitely smooth)
    sigmoid: C∞

The assumption of smoothness (C² or better) underlies most optimisation
convergence theory. Non-smooth objectives require sub-gradient methods.

Important theorem — Intermediate Value Theorem (IVT):
    If f is continuous on [a,b] and f(a) < y < f(b),
    then ∃ c ∈ (a,b) such that f(c) = y.
    Guarantees existence of zeros, roots, and fixed points.


### Squeeze Theorem

If g(x) ≤ f(x) ≤ h(x) near a, and lim_{x→a} g(x) = lim_{x→a} h(x) = L,
then lim_{x→a} f(x) = L.

The function f is "squeezed" between two functions that converge to the
same limit — so f must converge there too.

    Diagram — Squeeze:

    h(x) ↑  ╮                (h bounds from above)
    f(x) ↑  ╯→ L   at x→a
    g(x) ↑  ╭→ L             (g bounds from below)

Classic examples:
    lim_{x→0} x² sin(1/x) = 0
        Use −x² ≤ x²sin(1/x) ≤ x²; both bounds → 0.
    lim_{x→0} sin(x)/x = 1
        Use cos(x) ≤ sin(x)/x ≤ 1 for x ∈ (0, π/2).

ML relevance — bounding noisy quantities:
    In convergence analysis, gradient variance terms are often bounded
    between zero and a known σ² envelope.  The squeeze theorem formalises
    why "the error is between 0 and εₜ → 0" implies convergence.
    It also underlies the proof that (1 + x/n)ⁿ → eˣ, which appears in
    continuous-time limit derivations of SGD dynamics.


### Differentiability

f is DIFFERENTIABLE at a if the following limit exists:

    f'(a) = lim_{h→0} [f(a+h) − f(a)] / h

The derivative is the slope of the TANGENT LINE — the best linear
approximation to f near a:

    f(a + h) ≈ f(a) + f'(a)·h    for small h

    Diagram 1 — Derivative as Tangent:

    f(x) ↑    ╱ tangent at a  (slope = f'(a))
            ╱
     ─────╱──────────→ x
          a
    The secant [f(a+h)−f(a)]/h converges to f'(a) as h→0.

Differentiability implies continuity, but NOT vice versa:
    |x| is continuous everywhere, not differentiable at x=0.
    f(x) = x²sin(1/x), f(0)=0: differentiable but derivative not continuous.

Differentiability at a point means f locally looks like a straight line.
This linearisability is what makes calculus so powerful:
    ALL of optimisation theory hinges on "locally linear" approximations.


### PART 2 — DIFFERENTIATION RULES & THE CHAIN RULE

### Standard Differentiation Rules

    Constant:      d/dx [c]      = 0
    Power:         d/dx [xⁿ]     = nxⁿ⁻¹
    Sum:           d/dx [f+g]    = f' + g'
    Product:       d/dx [fg]     = f'g + fg'
    Quotient:      d/dx [f/g]    = (f'g − fg') / g²
    Chain rule:    d/dx [f(g(x))] = f'(g(x)) · g'(x)
    Exponential:   d/dx [eˣ]    = eˣ         (the self-referential miracle)
    Log:           d/dx [ln x]  = 1/x
    Trig:          d/dx [sin x] = cos x,   d/dx [cos x] = −sin x

Common derivatives in ML:
    sigmoid σ(x) = 1/(1+e⁻ˣ):    σ'(x) = σ(x)(1−σ(x))
    tanh(x):                       tanh'(x) = 1 − tanh²(x) = sech²(x)
    ReLU max(0,x):                 ReLU'(x) = 𝟙[x>0]   (sub-gradient at 0)
    softmax sⱼ = eˣʲ/Σeˣⁱ:       ∂sⱼ/∂xᵢ = sⱼ(δᵢⱼ − sᵢ)  (Jacobian)
    cross-entropy −log sᵧ:         ∂CE/∂xᵢ = sᵢ − 𝟙[i=y]  (combined)

The sigmoid self-derivative σ'=σ(1−σ) is elegant but causes the
VANISHING GRADIENT problem: σ'(x) ≤ 0.25 for all x, and for deep
networks the product of many σ' values exponentially → 0.


### The Chain Rule — The Heartbeat of Deep Learning

For f(g(x)):    d/dx[f(g(x))] = f'(g(x)) · g'(x)

For a composition of k functions:

    d/dx[f₁(f₂(···fₖ(x)···))] = f₁'(f₂(···)) · f₂'(f₃(···)) ··· fₖ'(x)

This is a PRODUCT of local derivatives along the computation path.

    ┌────────────────────────────────────────────────────────────┐
    │  Chain rule = multiply local derivatives along the path    │
    │  Backpropagation = chain rule applied RIGHT-TO-LEFT        │
    │  (from loss to inputs, accumulating gradients in reverse)  │
    └────────────────────────────────────────────────────────────┘

Neural network forward pass (L layers):

    x⁽⁰⁾ → z⁽¹⁾=W⁽¹⁾x⁽⁰⁾+b⁽¹⁾ → x⁽¹⁾=σ(z⁽¹⁾) → ··· → ℓ(x⁽ᴸ⁾, y)

Gradient at layer ℓ (backpropagation):

    ∂ℓ/∂W⁽ˡ⁾ = δ⁽ˡ⁾(x⁽ˡ⁻¹⁾)ᵀ     where  δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)

The δ⁽ˡ⁾ (error signal) propagates backward; the outer product with the
forward activation gives the weight gradient. This is the chain rule,
vectorised, applied recursively.


### L'Hôpital's Rule

For indeterminate forms 0/0 or ∞/∞:

    lim_{x→a} f(x)/g(x) = lim_{x→a} f'(x)/g'(x)   (when the latter exists)

Useful for analysing activation function behaviour, softmax stability,
and learning rate schedules near convergence.


### Implicit Differentiation

When a curve is defined implicitly by F(x, y) = 0 rather than
explicitly by y = f(x), differentiate both sides with respect to x
treating y as a function of x, then solve for dy/dx.

    Example:  x² + y² = r²    (circle)
        Differentiate: 2x + 2y(dy/dx) = 0
        Solve:          dy/dx = −x/y

    Example:  eˣʸ + x = sin(y)
        Differentiate: eˣʸ(y + x·dy/dx) + 1 = cos(y)·dy/dx
        Collect dy/dx: dy/dx [x·eˣʸ − cos(y)] = −y·eˣʸ − 1
        Solve:          dy/dx = (−y·eˣʸ − 1) / (x·eˣʸ − cos(y))

General rule — implicit function theorem:
    If F(x, y) = 0 and ∂F/∂y ≠ 0, then locally y = f(x) exists and:

        dy/dx = − (∂F/∂x) / (∂F/∂y)

    The multivariable form: if F(x, θ) = 0 defines θ implicitly as a
    function of x, then:

        dθ/dx = − (∂F/∂x) / (∂F/∂θ)    (when ∂F/∂θ is invertible)

ML relevance — implicit gradients:
    Meta-learning (MAML, iMAML) and bi-level optimisation require
    differentiating through an inner optimisation problem.  If θ*(x)
    is defined by the stationarity condition ∇_θ L(θ, x) = 0, then:

        dθ*/dx = − (∂²L/∂θ²)⁻¹ · (∂²L/∂θ∂x)

    This is implicit differentiation applied to the gradient equation
    F(θ, x) = ∇_θ L(θ, x) = 0.  Computing this inverse-Hessian-vector
    product efficiently is the core challenge of meta-learning algorithms.


### Mean Value Theorem (MVT)

If f is continuous on [a,b] and differentiable on (a,b), then:

    ∃ c ∈ (a,b)  such that  f'(c) = [f(b) − f(a)] / (b − a)

The MVT is the backbone of most convergence proofs:
    "The gradient at some intermediate point equals the average rate of change."

Corollary — Descent Lemma (fundamental to optimisation):
    If f has L-Lipschitz gradient: ‖∇f(x) − ∇f(y)‖ ≤ L‖x−y‖, then:

        f(y) ≤ f(x) + ∇f(x)ᵀ(y−x) + (L/2)‖y−x‖²

    This quadratic upper bound is the KEY inequality justifying
    gradient descent with step size η ≤ 1/L.


### PART 3 — MULTIVARIABLE CALCULUS

### Partial Derivatives

For f: ℝⁿ → ℝ, the partial derivative with respect to xᵢ:

    ∂f/∂xᵢ = lim_{h→0} [f(x + h·eᵢ) − f(x)] / h

Differentiate with respect to xᵢ treating all other variables as constants.

Example: f(x,y) = x²y + sin(xy)
    ∂f/∂x = 2xy + y·cos(xy)
    ∂f/∂y = x² + x·cos(xy)


### The Gradient

The GRADIENT of f: ℝⁿ → ℝ is the vector of all partial derivatives:

         ⎡∂f/∂x₁⎤
    ∇f = ⎢∂f/∂x₂⎥  ∈ ℝⁿ
         ⎣∂f/∂xₙ⎦

The gradient has two fundamental properties:

    1. DIRECTION: ∇f(x) points in the direction of STEEPEST ASCENT.
       −∇f(x) points in the direction of steepest DESCENT.

    2. MAGNITUDE: ‖∇f(x)‖ is the rate of change in that direction.

The directional derivative in direction u (‖u‖=1):

    Dᵤf(x) = ∇f(x) · u = ‖∇f(x)‖ cos θ

    Maximum at θ=0 (u parallel to ∇f):  Dᵤf = ‖∇f‖
    Zero at θ=90° (u perpendicular to ∇f)
    Minimum at θ=180° (u anti-parallel):  Dᵤf = −‖∇f‖

    Diagram 2 — Gradient as Level-Curve Normal:

    ↑                    ∇f points perpendicular to level curves f=c.
    │   f=3              Moving along a level curve: no change.
    │  ╭────╮  ↗ ∇f     Moving up the gradient: maximal increase.
    │  │f=2 │
    │  ╰────╯
    └────────────→

The gradient is ZERO at all local minima, maxima, and saddle points.
Finding where ∇f = 0 (the critical points) is the core of optimisation.


### The Jacobian

For f: ℝⁿ → ℝᵐ (vector-valued), the JACOBIAN is the matrix:

    J(x) ∈ ℝ^{m×n}    Jᵢⱼ = ∂fᵢ/∂xⱼ

    ⎡∂f₁/∂x₁  ···  ∂f₁/∂xₙ⎤
J = ⎢    ⋮              ⋮   ⎥
    ⎣∂fₘ/∂x₁  ···  ∂fₘ/∂xₙ⎦

The Jacobian is the best LINEAR APPROXIMATION to f near x:

    f(x + δ) ≈ f(x) + J(x)δ

    This is the multivariable generalisation of f(x+h) ≈ f(x) + f'(x)h.

The DETERMINANT of the Jacobian |det J| measures how much f locally
scales volume — critical for change-of-variables in integration and
for normalising flows in generative modelling.

In a neural network, the Jacobian of the output w.r.t. the input
captures how perturbations in the input affect each output — relevant
to adversarial robustness and sensitivity analysis.


### The Hessian

For f: ℝⁿ → ℝ (scalar-valued), the HESSIAN is the symmetric matrix:

    H(x) ∈ ℝ^{n×n}    Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ

By Schwarz's theorem (if f ∈ C²): Hᵢⱼ = Hⱼᵢ  (symmetric)

The Hessian captures CURVATURE — how the gradient changes direction.

    Diagram 3 — Hessian and Curvature:

     f                    f
     │    ╭─╮             │╲
     │   ╱   ╲            │  ╲
     │  ╱     ╲           │    ╲
     └──────────→ x       └────────→ x
     H > 0 (convex,       H < 0 (concave,
     curves up)           curves down)

The second-order Taylor expansion:

    f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + ½ δᵀH(x)δ

    Linear term: ∇f(x)ᵀδ   — first-order change (gradient)
    Quadratic term: ½δᵀHδ  — second-order correction (curvature)

Second-order test for critical points (∇f(x*) = 0):
    H(x*) positive definite (all eigenvalues > 0) → local MINIMUM
    H(x*) negative definite (all eigenvalues < 0) → local MAXIMUM
    H(x*) indefinite (mixed signs)                → SADDLE POINT

    The eigenvalues of H reveal the curvature in each eigenvector direction.
    The CONDITION NUMBER κ(H) = λ_max/λ_min determines how "elongated"
    the loss landscape is — high κ means slow gradient descent.


### Multivariable Chain Rule

For z = f(x) where x = g(t), x ∈ ℝⁿ, t ∈ ℝᵐ, z ∈ ℝ:

    dz/dtⱼ = Σᵢ (∂f/∂xᵢ)(∂xᵢ/∂tⱼ) = ∇f(x)ᵀ (∂x/∂tⱼ)

In matrix form:

    dz/dt = Jᵀ∇f(x)    where J = ∂x/∂t is the Jacobian

This is the general form that unifies all backpropagation algorithms.
The chain of Jacobian-vector products is exactly what automatic
differentiation computes efficiently (see Part 10).


### PART 4 — TAYLOR SERIES & APPROXIMATIONS

### Taylor's Theorem

If f is (n+1)-times differentiable, the Taylor expansion around a is:

    f(x) = f(a) + f'(a)(x−a) + f''(a)(x−a)²/2! + ··· + f⁽ⁿ⁾(a)(x−a)ⁿ/n!
           + Rₙ(x)

where the remainder:  Rₙ(x) = f⁽ⁿ⁺¹⁾(ξ)(x−a)ⁿ⁺¹/(n+1)!  for some ξ ∈ (a,x)

This is exact — not an approximation. Dropping the remainder gives the
degree-n polynomial approximation, with error bounded by |Rₙ|.

In optimisation, the FIRST-ORDER Taylor expansion:

    f(x + δ) ≈ f(x) + ∇f(x)ᵀδ

explains gradient descent: move δ = −η∇f to decrease f (locally).
The approximation is good when ‖δ‖ is small and f is smooth.

The SECOND-ORDER Taylor expansion:

    f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + ½ δᵀH(x)δ

is a quadratic function in δ. Minimising over δ gives Newton's step:

    δ* = −H(x)⁻¹∇f(x)

This step accounts for curvature and is the basis of Newton's method.


### Common Taylor Series (around x=0, Maclaurin series)

    eˣ  = 1 + x + x²/2! + x³/3! + ···                   (radius: ∞)
    ln(1+x) = x − x²/2 + x³/3 − ···                     (radius: 1)
    sin x = x − x³/3! + x⁵/5! − ···                     (radius: ∞)
    cos x = 1 − x²/2! + x⁴/4! − ···                     (radius: ∞)
    1/(1−x) = 1 + x + x² + x³ + ···                       (|x| < 1)
    (1+x)ᵅ = 1 + αx + α(α−1)x²/2! + ···                   (|x| < 1)

ML approximations:
    σ(x) ≈ ½ + x/4                  (near x=0, first-order)
    log(1 + eˣ) ≈ x  for x >> 0,  ≈ eˣ for x << 0  (softplus)
    softmax numerically: subtract max before exponentiating (avoid overflow)


### Big-O Notation for Approximation Error

    f(x+h) = f(x) + f'(x)h + O(h²)

    O(hᵏ): the error decays at least as fast as hᵏ as h→0.

    First-order (gradient) approximation: error O(h²)   (linear approx)
    Second-order (Newton) approximation:  error O(h³)   (quadratic approx)
    Finite difference for f'(x):
        Forward:  [f(x+h)−f(x)]/h        error O(h)
        Central:  [f(x+h)−f(x−h)]/(2h)  error O(h²)  ← numerically preferred


### PART 5 — CONVEXITY

### Convex Sets

A set C ⊆ ℝⁿ is CONVEX if for any two points x, y ∈ C:

    θx + (1−θ)y ∈ C   for all θ ∈ [0, 1]

The line segment between any two points lies entirely in C.

    Diagram 4 — Convex vs Non-Convex Sets:

    CONVEX         NON-CONVEX
    ╭──────╮      ╭──╮  ╭──╮
    │  ●───────●  │  │  │  │
    │             │  │ ╰──╯ │
    ╰─────────────╯  ╰──────╯
    (segment inside) (segment exits set)

Examples of convex sets:
    ℝⁿ,  any half-space {x: aᵀx ≤ b},  any ball {x: ‖x−c‖ ≤ r}
    Positive orthant {x: xᵢ ≥ 0},  affine subspaces,  polyhedra
    Intersections of convex sets are convex.

Important: the feasible region of most ML constraints is convex (e.g.,
ℓ₁ ball, ℓ₂ ball, simplex, cone). Projecting onto convex sets has
unique solutions.


### Convex Functions — Three Equivalent Definitions

**Definition 1 (geometric — chord lies above curve):**

    f(θx + (1−θ)y) ≤ θf(x) + (1−θ)f(y)   for all θ ∈ [0,1]

    The function value at any convex combination ≤ the convex combination
    of function values. The "chord" connecting two points lies above the graph.

**Definition 2 (first-order condition — tangent lies below curve):**

    f(y) ≥ f(x) + ∇f(x)ᵀ(y−x)   for all x, y

    The tangent hyperplane (first-order approximation) is a GLOBAL LOWER
    BOUND. This is the most useful property for optimisation proofs.

**Definition 3 (second-order condition — non-negative curvature):**

    H(x) ⪰ 0   (Hessian is positive semidefinite) for all x

    f''(x) ≥ 0 in 1D — the function curves upward everywhere.

All three are equivalent for twice-differentiable functions. The first-order
characterisation is the workhorse of convergence proofs.

    ┌────────────────────────────────────────────────────────────┐
    │  KEY THEOREM: For a convex function,                       │
    │  every local minimum is a GLOBAL minimum.                  │
    │  ∇f(x*) = 0 is both necessary AND sufficient for x* to     │
    │  be the global minimiser.                                  │
    └────────────────────────────────────────────────────────────┘


### Degrees of Convexity

**Strictly convex:**  f(θx + (1−θ)y) < θf(x) + (1−θ)f(y)  for x≠y, θ∈(0,1)
    Strictly convex → unique minimiser (if it exists).
    Example: f(x) = x²,  ‖x‖₂² = xᵀx.

**Strongly convex (with parameter μ > 0):**

    f(y) ≥ f(x) + ∇f(x)ᵀ(y−x) + (μ/2)‖y−x‖²

    Equivalently: H(x) ⪰ μI  (Hessian eigenvalues ≥ μ > 0).
    The function curves up at least as fast as a quadratic with curvature μ.

    Strong convexity gives:
    1. Unique minimiser x*
    2. ERROR BOUND: ‖x−x*‖² ≤ (2/μ)[f(x) − f(x*)]
    3. LINEAR CONVERGENCE of gradient descent: error shrinks by factor (1−ημ)

Strong convexity with parameter μ and Lipschitz gradient with constant L:
    κ = L/μ  (the condition number of the loss landscape)
    Gradient descent converges as: f(xₜ) − f* ≤ (1 − μ/L)ᵗ [f(x₀) − f*]


### Convex Functions in ML

    f(x) = xᵀAx + bᵀx + c (A ⪰ 0)   — quadratic form (convex)
    f(x) = ‖Ax − b‖₂²                 — least squares (convex, not strongly)
    f(x) = ‖Ax − b‖₂² + λ‖x‖₂²       — ridge (strongly convex for λ>0)
    f(x) = −Σᵢ yᵢ log pᵢ             — cross-entropy (convex in logits)
    f(W) = loss of neural network      — NON-convex in general

The COMPOSITION RULES for convexity:
    Non-neg weighted sum: Σᵢ wᵢfᵢ (wᵢ≥0)        is convex if each fᵢ convex
    Affine composition:   f(Ax+b)                 is convex if f convex
    Pointwise max:        max(f₁,...,fₖ)           is convex if each fᵢ convex
    Sublevel sets:        {x: f(x) ≤ c}           are convex if f convex

These rules let you check convexity by decomposing complex functions
into simple building blocks.


### Jensen's Inequality

For convex f and a random variable X:

    f(𝔼[X]) ≤ 𝔼[f(X)]

"The function of the mean is ≤ the mean of the function."

This single inequality underpins much of information theory and
probabilistic ML:
    E[X²] ≥ (E[X])²               (variance is non-negative)
    −log 𝔼[p] ≤ 𝔼[−log p]         (KL divergence is non-negative)
    E[eˣ] ≥ e^{E[X]}               (exponential moment ≥ MGF at mean)

The gap f(E[X]) to E[f(X)] measures the "cost of curvature" — the
greater the variance of X and the more curved f, the larger the gap.


### PART 6 — UNCONSTRAINED OPTIMISATION

### First-Order Necessary Conditions

At any local minimum x* of a differentiable f:

    ∇f(x*) = 0    (the gradient vanishes — stationarity)

This is NECESSARY but not sufficient.

All critical points (zeros of ∇f) are:
    Local minima      (H ≻ 0)
    Local maxima      (H ≺ 0)
    Saddle points     (H indefinite — positive and negative eigenvalues)

For non-convex functions (neural networks), the landscape is riddled with
saddle points. Recent theory suggests saddle points — not local minima —
are the main obstacle to convergence in deep networks.


### Second-Order Conditions (Sufficient)

At a critical point ∇f(x*) = 0:

    H(x*) ≻ 0  (all eigenvalues positive) → strict local MINIMUM
    H(x*) ≺ 0  (all eigenvalues negative) → strict local MAXIMUM
    H(x*) indefinite                       → SADDLE POINT

    If H(x*) ⪰ 0 (PSD, not PD): test is inconclusive — higher-order terms decide.

Example: f(x,y) = x² − y²   at origin (0,0)
    ∇f = [2x, −2y] = 0 at origin
    H = diag(2, −2): eigenvalues +2 and −2 → SADDLE


### Gradient Descent — The Workhorse

Algorithm:    xₜ₊₁ = xₜ − η∇f(xₜ)

The negative gradient is the direction of steepest descent.
Step size η (learning rate) controls how far we move.

    ┌────────────────────────────────────────────────────────────┐
    │  Convergence guarantees (L-smooth, μ-strongly convex):     │
    │                                                            │
    │  f(xₜ) − f* ≤ (1 − μ/L)ᵗ [f(x₀) − f*]   (linear rate)       │
    │                                                            │
    │  Optimal step: η = 1/L  (largest safe step under L-bound)  │
    │  Convergence rate: (1 − μ/L)  per iteration                │
    │  Iterations to ε accuracy: O((L/μ) log(1/ε)) = O(κ log 1/ε)│
    └────────────────────────────────────────────────────────────┘

The CONDITION NUMBER κ = L/μ is the fundamental obstacle:
    κ = 1: perfectly round landscape, converges in 1 step
    κ = 1000: very elongated ellipses, takes ~1000 steps per digit of accuracy

Geometric picture:

    Diagram 5 — Gradient Descent on Quadratic Landscape:

    κ ≈ 1 (round)        κ >> 1 (elongated)
    ╭────╮               ╭──────────────╮
    │  ●→●→★             │●             │
    │       │            │ ↘            │
    ╰────╯               │  ↘↗↘         │
    (fast convergence)   │      ★       │
                         ╰──────────────╯
                         (zigzag, slow)

Zigzagging occurs because the gradient points away from the minimum
in high-κ landscapes — each step overshoots in the steep direction.


### Newton's Method — Second-Order Optimisation

Algorithm:    xₜ₊₁ = xₜ − H(xₜ)⁻¹∇f(xₜ)

Minimise the second-order Taylor approximation at each step:

    q(δ) = f(x) + ∇f(x)ᵀδ + ½δᵀH(x)δ

Setting ∇q = 0 gives the Newton step δ = −H⁻¹∇f.

    Convergence: QUADRATIC near the minimum
        ‖xₜ₊₁ − x*‖ ≤ C‖xₜ − x*‖²

    Each step roughly squares the error. Near the optimum:
        t=1: error 10⁻¹ → t=2: error 10⁻² → t=3: error 10⁻⁴ → 10⁻⁸

    Cost: O(n²) to store H, O(n³) to invert — prohibitive for n~10⁷ parameters.
    Workaround: quasi-Newton methods (BFGS, L-BFGS) approximate H⁻¹ cheaply.

The Newton step also PRECONDITIONS the gradient:
    H⁻¹∇f rescales the gradient by the inverse curvature → direction is
    independent of the condition number → κ-independent convergence.
    This is why second-order methods are so powerful in theory.


### Convergence Theory Summary

    ┌──────────────────┬───────────────────────────┬─────────────────────────────┐
    │ Method           │ Convergence Rate          │ Cost per Iteration          │
    ├──────────────────┼───────────────────────────┼─────────────────────────────┤
    │ Gradient Descent │ Linear  O(κ log 1/ε)      │ O(n) (gradient only)        │
    │ Heavy Ball       │ Linear  O(√κ log 1/ε)     │ O(n) + momentum buffer      │
    │ Nesterov (AGD)   │ Optimal O(√κ log 1/ε)     │ O(n) (tight lower bound)    │
    │ Newton's Method  │ Quadratic  O(log log 1/ε) │ O(n³) (Hessian inv.)        │
    │ L-BFGS           │ Super-linear              │ O(n·m) m=memory size        │
    │ Conjugate Grad.  │ O(√κ log 1/ε)             │ O(n) (exact for quadratic)  │
    └──────────────────┴───────────────────────────┴─────────────────────────────┘

    Nesterov's accelerated gradient descent achieves the OPTIMAL rate
    for first-order methods — it cannot be improved without using
    curvature (second-order) information.


### PART 7 — CONSTRAINED OPTIMISATION & DUALITY

### The Constrained Problem

General form:

    min  f(x)
     x
    s.t.  gᵢ(x) ≤ 0    for i = 1, ..., m   (inequality constraints)
          hⱼ(x) = 0    for j = 1, ..., p   (equality constraints)

When f, gᵢ convex and hⱼ affine → CONVEX OPTIMISATION.
Convex problems are tractable; the global minimum is reachable.


### Lagrange Multipliers (Equality Constraints)

For the problem:   min f(x)   s.t.  h(x) = 0

At a constrained minimum x*, the gradient of f must be perpendicular
to the constraint surface — otherwise we could move along the constraint
to decrease f further:

    ∇f(x*) = λ∇h(x*)    for some scalar λ ∈ ℝ

The scalar λ is the LAGRANGE MULTIPLIER. The constrained optimum
satisfies the stationary conditions of the Lagrangian:

    L(x, λ) = f(x) + λh(x)

    ∂L/∂x = 0  →  ∇f = −λ∇h
    ∂L/∂λ = 0  →  h(x) = 0  (constraint satisfied)

Geometric interpretation: at the optimum, level curves of f and the
constraint surface h(x)=0 are tangent — they have the same normal.

Example — Maximum entropy on a simplex:
    max  −Σᵢ pᵢ log pᵢ    s.t.  Σᵢ pᵢ = 1
    Lagrangian: L = −Σpᵢ log pᵢ + λ(Σpᵢ − 1)
    Setting ∂L/∂pᵢ = 0: −log pᵢ − 1 + λ = 0 → pᵢ = e^{λ−1}  (uniform!)
    → The maximum entropy distribution is UNIFORM.


### KKT Conditions (Inequality + Equality Constraints)

For the general constrained problem, the KARUSH-KUHN-TUCKER (KKT)
conditions are NECESSARY (and for convex problems, SUFFICIENT) for x*
to be optimal:

    1. STATIONARITY:         ∇f(x*) + Σᵢ μᵢ∇gᵢ(x*) + Σⱼ λⱼ∇hⱼ(x*) = 0

    2. PRIMAL FEASIBILITY:   gᵢ(x*) ≤ 0  and  hⱼ(x*) = 0

    3. DUAL FEASIBILITY:     μᵢ ≥ 0   (inequality multipliers non-negative)

    4. COMPLEMENTARY SLACKNESS: μᵢ · gᵢ(x*) = 0   for all i

Complementary slackness is the key insight:
    Either the constraint is ACTIVE (gᵢ = 0, μᵢ can be > 0)
    Or the constraint is INACTIVE (gᵢ < 0, so μᵢ = 0)
    An inactive constraint doesn't affect the optimum.

Example — SVM primal:
    min  ½‖w‖²    s.t.  yᵢ(wᵀxᵢ+b) ≥ 1  for all i

    Rewrite as:  gᵢ(w,b) = 1 − yᵢ(wᵀxᵢ+b) ≤ 0
    KKT stationarity: w = Σᵢ μᵢyᵢxᵢ  (weights are a weighted sum of data!)
    Complementary slackness: μᵢ[1 − yᵢ(wᵀxᵢ+b)] = 0
    → μᵢ > 0 only for SUPPORT VECTORS (points on the margin boundary).


### Lagrangian Duality

For the primal:   p* = min_{x} max_{μ≥0,λ} L(x, μ, λ)

The DUAL problem:  d* = max_{μ≥0,λ} min_{x} L(x, μ, λ)

    Weak duality:    d* ≤ p*    (always holds)
    Strong duality:  d* = p*    (holds under constraint qualifications
                                 e.g., Slater's condition for convex problems)

The DUALITY GAP = p* − d* ≥ 0.

Why duality matters:
    1. The dual is often EASIER to solve than the primal (e.g., SVM dual).
    2. The dual variables (multipliers) have economic interpretations
       as "shadow prices" of constraints.
    3. Dual decomposition enables distributed optimisation.
    4. Strong duality means solving the dual gives the primal solution.

SVM dual (derived via KKT + strong duality):
    max_α  Σᵢ αᵢ − ½ Σᵢⱼ αᵢαⱼyᵢyⱼ xᵢᵀxⱼ
    s.t.   0 ≤ αᵢ ≤ C,  Σᵢ αᵢyᵢ = 0

The dual depends only on INNER PRODUCTS xᵢᵀxⱼ — replacing with a
kernel function k(xᵢ, xⱼ) gives the kernel SVM.
This is the KERNEL TRICK: dual formulation enables implicit infinite-
dimensional feature spaces via the kernel function.


### Proximal Operators & Sub-gradients

When f is not differentiable (e.g., f(x) = ‖x‖₁ in Lasso):

**Sub-gradient:** g ∈ ∂f(x) satisfies  f(y) ≥ f(x) + gᵀ(y−x)  for all y
    ∂|x| at x=0: any g ∈ [−1, 1]   (sub-differential is an interval)
    Sub-gradient descent converges but more slowly than gradient descent.

**Proximal operator:**
    prox_{ηf}(v) = argmin_x {f(x) + ‖x−v‖²/(2η)}

    For f(x) = ‖x‖₁:  prox_η(v)ᵢ = sign(vᵢ)·max(|vᵢ|−η, 0)
                                     = SOFT THRESHOLDING

    Proximal gradient descent:  xₜ₊₁ = prox_{η·r}(xₜ − η∇f(xₜ))
    Separates the smooth part (gradient step) from the non-smooth part
    (proximal step). Used in Lasso, Group Lasso, Total Variation.


### PART 8 — STOCHASTIC & ADAPTIVE OPTIMISATION

### Stochastic Gradient Descent (SGD)

Full gradient:  ∇f(x) = (1/n) Σᵢ₌₁ⁿ ∇fᵢ(x)    — expensive for large n

SGD uses a MINIBATCH of size B to estimate the gradient:

    gₜ = (1/B) Σᵢ∈batch ∇fᵢ(xₜ)    (noisy but cheap)
    xₜ₊₁ = xₜ − ηₜ gₜ

    gₜ is an UNBIASED ESTIMATOR: 𝔼[gₜ] = ∇f(xₜ)

    Convergence of SGD (convex case):
        𝔼[f(x̄ₜ) − f*] ≤ R²/(ηt) + ησ²/2

    where R = ‖x₀−x*‖, σ² = variance of stochastic gradient.
    Optimal decay: ηₜ = η₀/√t → convergence O(1/√t) vs O(1/t) for GD.

    ┌────────────────────────────────────────────────────────────┐
    │  FUNDAMENTAL TENSION:                                      │
    │  Large η → fast initial progress but noisy (variance)      │
    │  Small η → slow but precise (bias reduction)               │
    │  Solution: LEARNING RATE SCHEDULE — decrease η over time   │
    └────────────────────────────────────────────────────────────┘

Batch size trade-offs:
    B=1:      maximum noise, parallelism is trivial, 1 step/sample
    B=n:      full gradient, deterministic, slow per step, no noise
    B=32−512: empirical sweet spot — enough noise for generalisation,
              enough signal for stable updates


### Momentum (Heavy Ball Method)

Motivation: SGD zigzags in high-κ landscapes. Add a "momentum buffer"
to smooth the gradient signal:

    mₜ₊₁ = βmₜ + gₜ         (exponential moving average of gradients)
    xₜ₊₁ = xₜ − η mₜ₊₁

    β ∈ [0,1) is the momentum coefficient (β=0.9 typical).

Geometric effect: in consistent directions, momentum accumulates
(speed up). In oscillating directions, positive and negative gradients
cancel (damp oscillations). Converges at rate O(κ) → O(√κ) for quadratics.


### Nesterov Accelerated Gradient (NAG)

    Look-ahead gradient — compute gradient at the "future" position:

    yₜ = xₜ + β(xₜ − xₜ₋₁)         (momentum correction)
    xₜ₊₁ = yₜ − η∇f(yₜ)            (gradient step from look-ahead)

    Achieves the OPTIMAL O(√κ) rate for smooth convex functions.
    In PyTorch: SGD(nesterov=True)


### Adaptive Methods — AdaGrad, RMSProp, Adam

**AdaGrad (Adaptive Gradient):**  Accumulate squared gradients to adapt per-parameter:

    Gₜ = Gₜ₋₁ + gₜ²             (cumulative sum of squared gradients)
    xₜ₊₁ = xₜ − η gₜ / √(Gₜ + ε)

    Intuition: parameters that receive large gradients get a smaller
    effective step; rarely-updated parameters (sparse features) get a
    larger step.

    AdaGrad excels on sparse data (NLP bag-of-words, recommendation
    systems): infrequent features accumulate small Gₜ → large effective η.

    Critical weakness: Gₜ grows monotonically → effective learning rate
    decays to zero over time.  For non-convex deep learning this means
    training stalls before convergence.  RMSProp and Adam fix this by
    using an EXPONENTIAL MOVING AVERAGE of squared gradients instead of
    a running sum.

    AdaGrad convergence (convex, non-smooth):
        Regret ≤ O(√T)   (optimal for online convex optimisation)
        The monotonically shrinking step is a feature here — it enforces
        decreasing updates as the cumulative evidence grows.


**RMSProp:**  Maintain per-parameter estimate of gradient scale:

    vₜ = ρvₜ₋₁ + (1−ρ)gₜ²         (exponential MA of squared gradient)
    xₜ₊₁ = xₜ − η gₜ / √(vₜ + ε)  (rescale by historical scale)

    Adapts learning rate to each parameter individually.
    Large historical gradients → small effective step (don't overshoot).
    Small historical gradients → large effective step (explore sparse dims).


**Adam (Adaptive Moment Estimation):**  Combines momentum + RMSProp:

    mₜ = β₁mₜ₋₁ + (1−β₁)gₜ           (first moment, β₁=0.9)
    vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²          (second moment, β₂=0.999)
    m̂ₜ = mₜ/(1−β₁ᵗ)                  (bias-corrected first moment)
    v̂ₜ = vₜ/(1−β₂ᵗ)                  (bias-corrected second moment)
    xₜ₊₁ = xₜ − η m̂ₜ / (√v̂ₜ + ε)   (update with corrected moments)

    Bias correction: at t=1, m₁=β₁·0+(1−β₁)g₁=(1−β₁)g₁ underestimates
    the true mean; dividing by (1−β₁ᵗ) corrects the initialisation bias.

    Adam is the DEFAULT optimiser for deep learning:
    - Robust to learning rate choice (effective range: 1e-4 to 1e-2)
    - Automatic per-parameter scaling
    - Works well with sparse gradients (NLP, embeddings)

    Theoretical caveats: Adam can fail to converge on convex problems
    in theory; AdamW (Adam + weight decay decoupled from gradient) is
    more principled for regularised training.


### Learning Rate Schedules

The learning rate η is perhaps the most important hyperparameter.
Common schedules:

    Step decay:        ηₜ = η₀ · γ^{⌊t/k⌋}        (multiply by γ every k steps)
    Exponential:       ηₜ = η₀ · e^{−λt}
    Cosine annealing:  ηₜ = ηₘᵢₙ + ½(η₀−ηₘᵢₙ)(1+cos(πt/T))
    Warmup + decay:    linear warmup to η₀, then cosine/linear decay
    Cyclical LR:       oscillates between bounds (helps escape saddle pts)
    OneCycleLR:        single cycle of warmup + decay (fastest in practice)

Warmup intuition: at the start of training, gradient estimates are
unreliable (model is random, batch statistics are noisy). A small initial
lr lets the optimiser orient itself before taking large steps.


### PART 9 — INTEGRATION, MEASURE THEORY & INFORMATION THEORY CONNECTIONS

### The Integral

The Riemann integral of f on [a,b]:

    ∫ₐᵇ f(x) dx = lim_{n→∞} Σᵢ f(xᵢ*)Δx    (area under the curve)

Fundamental Theorem of Calculus:

    d/dx ∫ₐˣ f(t) dt = f(x)     (differentiation undoes integration)
    ∫ₐᵇ f'(x) dx = f(b) − f(a)  (integration over derivative = net change)

Integration by parts (the integration analogue of the product rule):

    ∫ u dv = uv − ∫ v du

Applied in: variational inference, EM algorithm derivations, sampling.


### Multiple Integrals

For f: ℝⁿ → ℝ, integration extends naturally to higher dimensions.

**Double integral:**

    ∬_D f(x, y) dA = ∫_{a}^{b} ∫_{g(x)}^{h(x)} f(x, y) dy dx

    Fubini's theorem: if f is continuous on the rectangle [a,b]×[c,d]:

        ∫_{a}^{b} ∫_{c}^{d} f(x,y) dy dx = ∫_{c}^{d} ∫_{a}^{b} f(x,y) dx dy

    The order of integration can be swapped freely for continuous f.

**Triple and n-fold integrals:**

    ∭_V f(x,y,z) dV   — volume integral over region V ⊆ ℝ³
    ∫_{ℝⁿ} f(x) dx    — n-dimensional integral (Lebesgue sense)

**Change of variables in multiple integrals:**

    For a bijective transformation (x,y) = T(u,v):

        ∬_D f(x,y) dA = ∬_{D'} f(T(u,v)) |det J_T(u,v)| du dv

    The absolute Jacobian determinant |det J| corrects for area distortion.

    Common transforms:
        Polar:      x=r cosθ, y=r sinθ    |J| = r
        Cylindrical: x=r cosθ, y=r sinθ, z=z   |J| = r
        Spherical:  x=ρ sinφ cosθ, ...    |J| = ρ² sinφ

ML relevance — marginalisation and normalisation:
    Marginalising a joint density p(x, z):
        p(x) = ∫ p(x, z) dz    (intractable in general → variational approx)
    Partition function of energy models:
        Z = ∫ e^{−E(x)} dx     (intractable → MCMC or contrastive divergence)
    Gaussian normalisation:
        ∫_{ℝⁿ} exp(−½ xᵀΣ⁻¹x) dx = (2π)^{n/2} |Σ|^{½}
        Proof uses the change-of-variables to principal axes plus the
        1D Gaussian integral ∫e^{−t²}dt = √π.


### Expected Values as Integrals

For a continuous random variable X with density p(x):

    𝔼[f(X)] = ∫ f(x) p(x) dx

    𝔼[X] = ∫ x p(x) dx           (mean)
    Var[X] = 𝔼[X²] − (𝔼[X])²    (variance)
    𝔼[f(g(X))] ≠ f(g(𝔼[X]))  in general (Jensen's inequality)

Monte Carlo estimation:

    𝔼[f(X)] ≈ (1/N) Σᵢ f(xᵢ),  xᵢ ~ p(x)    (unbiased, error O(1/√N))

    The stochastic gradient in SGD is a Monte Carlo estimate of ∇f:
        𝔼[∇fᵢ(x)] = ∇f(x)   (the mini-batch is a random sample)


### Change of Variables

For a bijective differentiable transformation x = g(z):

    p_X(x) = p_Z(g⁻¹(x)) · |det J_{g⁻¹}(x)|

The Jacobian determinant accounts for how g stretches/compresses space.

Normalising flows use this to learn flexible distributions:
    Sample z ~ p_Z (simple base, e.g. Gaussian)
    Transform x = g(z) via a learned bijection g
    Compute the likelihood p_X(x) exactly using the Jacobian formula


### KL Divergence and Calculus of Variation

KL divergence from q to p:

    KL(q ‖ p) = ∫ q(x) log [q(x)/p(x)] dx = 𝔼_q[log q − log p]

Properties:
    KL(q‖p) ≥ 0      (non-negativity — follows from Jensen's inequality)
    KL(q‖p) = 0 iff q = p    (identity of indiscernibles)
    KL is not symmetric: KL(q‖p) ≠ KL(p‖q)

The ELBO (Evidence Lower BOund) in variational inference:

    log p(x) ≥ 𝔼_q[log p(x,z)] − 𝔼_q[log q(z)]
             = 𝔼_q[log p(x|z)] − KL(q(z) ‖ p(z))

Maximising the ELBO simultaneously:
    1. Minimises KL(q‖p): brings approximate posterior q(z|x) close to true p(z|x)
    2. Maximises reconstruction quality: 𝔼[log p(x|z)]


### PART 10 — VECTOR CALCULUS

### The Del Operator ∇

In ℝⁿ, the del (nabla) operator is the vector of partial derivatives:

    ∇ = (∂/∂x₁, ∂/∂x₂, ..., ∂/∂xₙ)

Applied to different objects it produces three fundamental quantities:

    ∇f      = gradient     (scalar field → vector field)
    ∇ · F   = divergence   (vector field → scalar field)
    ∇ × F   = curl         (vector field → vector field, in ℝ³)
    ∇²f     = Laplacian    (scalar field → scalar field)


### Gradient (Recap & Extension)

For f: ℝⁿ → ℝ:

    ∇f(x) = (∂f/∂x₁, ..., ∂f/∂xₙ)

Points in the direction of steepest ascent; magnitude = rate of change.
Already covered in Part 3; listed here to complete the del-operator family.


### Divergence

For a vector field F: ℝⁿ → ℝⁿ,  F = (F₁, ..., Fₙ):

    ∇ · F = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ··· + ∂Fₙ/∂xₙ    (scalar)

Geometric meaning: divergence measures the NET OUTFLOW of F at a point.
    ∇ · F > 0: F is spreading outward (source)
    ∇ · F < 0: F is converging inward (sink)
    ∇ · F = 0: divergence-free / incompressible

    Diagram — Divergence:

    Source (∇·F > 0)       Sink (∇·F < 0)      Uniform (∇·F = 0)
        ↗ ↑ ↖                  ↘ ↓ ↙               → → →
        ← · →                  → · ←               → → →
        ↙ ↓ ↘                  ↗ ↑ ↖               → → →

ML relevance:
    In normalising flows, |det J| = divergence-type measure of volume change.
    Continuous normalising flows (CNFs) use the instantaneous change-of-
    variables formula: d/dt log p(x(t)) = −∇ · f(x(t), t), where f is the
    learned vector field — the trace of the Jacobian equals the divergence.


### Curl

For F = (P, Q, R): ℝ³ → ℝ³:

    ∇ × F = ( ∂R/∂y − ∂Q/∂z,   ∂P/∂z − ∂R/∂x,   ∂Q/∂x − ∂P/∂y )

Geometric meaning: curl measures ROTATIONAL TENDENCY at a point.
    ∇ × F = 0: irrotational (conservative field — has a potential)
    ‖∇ × F‖:   angular speed of infinitesimal fluid rotation

A vector field F is CONSERVATIVE iff ∇ × F = 0 (in simply-connected domain),
equivalently ∃ scalar potential φ such that F = ∇φ.

In 2D, the scalar curl is:  (∂Q/∂x − ∂P/∂y)  (the z-component of ∇×F).


### The Laplacian

For f: ℝⁿ → ℝ:

    ∇²f = ∇ · (∇f) = ∂²f/∂x₁² + ∂²f/∂x₂² + ··· + ∂²f/∂xₙ²    (scalar)

The Laplacian is the SUM OF SECOND DERIVATIVES — the divergence of the
gradient.  It measures how the value of f at a point differs from its
local average: ∇²f > 0 means f is below its neighbours (bowl-shaped);
∇²f < 0 means f is above (hill-shaped).

Harmonic functions: ∇²f = 0  (Laplace equation)
    Solutions are "as smooth as possible" — max/min only on the boundary.

The vector Laplacian: ∇²F = (∇²F₁, ∇²F₂, ∇²F₃)

ML relevance:
    Graph Laplacian L = D − A (D = degree matrix, A = adjacency) is the
    discrete analogue.  The normalised form L̂ = D^{−½}LD^{−½} is the core
    operator in spectral GNNs (GCN, ChebNet).  Its eigenvalues are the graph
    "frequencies"; low eigenvalues = smooth signals over the graph.
    Laplacian regularisation: min ‖y − f‖² + λ fᵀLf
    penalises functions that vary sharply between connected nodes.


### Line Integrals

The line integral of a scalar field f along a curve C:

    ∫_C f ds = ∫_{a}^{b} f(r(t)) ‖r'(t)‖ dt

The line integral of a VECTOR FIELD F along C (work integral):

    ∫_C F · dr = ∫_{a}^{b} F(r(t)) · r'(t) dt

    Physical meaning: total work done by force F along path C.

For a conservative field F = ∇φ:
    ∫_C F · dr = φ(B) − φ(A)   (path-independent — only endpoints matter)
    ∮_C F · dr = 0              (closed-loop integral vanishes)


### Surface Integrals

The surface integral of f over surface S:

    ∬_S f dS = ∬_{D} f(r(u,v)) ‖rᵤ × rᵥ‖ du dv

The flux integral of F through S (flow across a surface):

    ∬_S F · dS = ∬_S F · n̂ dS = ∬_{D} F(r(u,v)) · (rᵤ × rᵥ) du dv

    Physical meaning: net flow of F across surface S per unit time.


### Green's Theorem

Relates a line integral around a CLOSED PLANE CURVE C to a double
integral over the enclosed region D:

    ∮_C (P dx + Q dy) = ∬_D (∂Q/∂x − ∂P/∂y) dA

    Left side: circulation of F = (P, Q) around the boundary.
    Right side: integral of the 2D scalar curl over the interior.

    Special case (area):  A = ½ ∮_C (x dy − y dx)

Green's theorem = 2D special case of Stokes' theorem.

ML relevance: used in proofs about 2D flow models and in deriving
integration-by-parts identities for training objectives.


### Stokes' Theorem

Generalises Green's theorem to a SURFACE S with boundary curve ∂S:

    ∮_{∂S} F · dr = ∬_S (∇ × F) · dS

    Left side: circulation of F around the boundary curve.
    Right side: flux of the curl through the surface.

    ┌────────────────────────────────────────────────────────────┐
    │  Stokes' theorem: boundary integral = interior curl flux   │
    │  "Local rotation (curl) sums to global boundary twist"     │
    └────────────────────────────────────────────────────────────┘

Key corollary: if ∇ × F = 0 everywhere on S, then the circulation
around any closed curve on S is zero — confirming F is conservative.


### Divergence Theorem (Gauss's Theorem)

Relates the flux of F through a CLOSED SURFACE ∂V to the divergence
integral over the enclosed volume V:

    ∯_{∂V} F · dS = ∭_V (∇ · F) dV

    Left side: total flux out through the closed surface.
    Right side: sum of all sources/sinks inside the volume.

    ┌────────────────────────────────────────────────────────────┐
    │  "What flows out = sum of all sources inside"              │
    │  Global flux = integral of local divergence                │
    └────────────────────────────────────────────────────────────┘

ML relevance: the instantaneous change-of-variables in continuous
normalising flows (∂_t log p = −∇ · v_θ) is the infinitesimal form of
the divergence theorem.  Flows preserve probability mass: total
probability that "flows out" of any region equals the divergence inside.


### Unified View — The Generalised Stokes' Theorem

All four integral theorems (FTC, Green's, Stokes', Divergence) are
special cases of a single theorem on manifolds:

    ∫_{∂M} ω = ∫_M dω

    M: oriented manifold with boundary ∂M.
    ω: differential form.
    d: exterior derivative operator.

    FTC:        M = [a,b],    ω = f,    dω = f' dx
    Green's:    M = D ⊆ ℝ², ω = Pdx+Qdy
    Stokes':    M = surface,  ω = 1-form
    Divergence: M = volume,   ω = 2-form, dω = divergence 3-form


### PART 11 — AUTOMATIC DIFFERENTIATION

### Three Ways to Compute Derivatives

**1. Symbolic differentiation:**
    Apply differentiation rules symbolically (like Mathematica).
    Exact, but can cause expression swell — the symbolic form of the
    derivative can be exponentially larger than the original function.

**2. Numerical differentiation (finite differences):**
    f'(x) ≈ [f(x+h) − f(x)] / h    O(h) error
    f'(x) ≈ [f(x+h) − f(x−h)] / (2h)  O(h²) error

    Simple but: requires choosing h carefully (trade-off truncation vs
    rounding error), scales as O(n) function evaluations for gradient
    in ℝⁿ. Used for GRADIENT CHECKING (verify autodiff implementations).

**3. Automatic differentiation (autodiff):**
    Decompose the computation into ELEMENTARY OPERATIONS, apply the
    chain rule to each, accumulate numerically. EXACT (no approximation
    error), scales to millions of parameters, the foundation of PyTorch/JAX.


### Computational Graph

Every differentiable program can be expressed as a directed acyclic
graph (DAG) of elementary operations:

    x → [multiply by 2] → v₁ → [sin] → v₂ → [add with w] → v₃ → y

    Forward pass: compute each node given its inputs (the values)
    Backward pass: propagate gradients backward through the same graph

    Example:  y = sin(2x + w)
    Nodes: v₁ = 2x,  v₂ = v₁ + w,  v₃ = sin(v₂),  y = v₃
    Forward: compute v₁, v₂, v₃, y left to right.
    Backward (chain rule):
        ∂y/∂v₃ = 1
        ∂y/∂v₂ = ∂y/∂v₃ · ∂v₃/∂v₂ = 1 · cos(v₂)
        ∂y/∂v₁ = ∂y/∂v₂ · ∂v₂/∂v₁ = cos(v₂) · 1
        ∂y/∂x  = ∂y/∂v₁ · ∂v₁/∂x  = cos(v₂) · 2 = 2cos(2x+w)
        ∂y/∂w  = ∂y/∂v₂ · ∂v₂/∂w  = cos(v₂) · 1 = cos(2x+w)


### Reverse-Mode AD (Backpropagation)

For f: ℝⁿ → ℝ (loss function):

    1 forward pass:  compute all intermediate values (O(work(f)))
    1 backward pass: compute all n partial derivatives (O(work(f)))

    Total cost: O(work(f)) — INDEPENDENT of n!
    This is why backprop scales to billions of parameters.

The backward pass computes ADJOINT variables (bar notation):

    v̄ᵢ = ∂f/∂vᵢ   (how does the output change with each intermediate node?)

Each node contributes to v̄ of its inputs via the chain rule.
"Gradient flows backward": the adjoint of each node receives from the
adjoint of its outputs, and passes to the adjoint of its inputs.

Reverse-mode is optimal when outputs << inputs (scalar loss, many params).
Forward-mode is optimal when inputs << outputs (few inputs, many outputs).


### Forward-Mode vs Reverse-Mode

**Forward-mode (jvp — Jacobian-vector products):**

    Seed with a tangent vector ṽ (direction in input space).
    Propagates: v̇ᵢ = (∂vᵢ/∂inputs)ṽ   (directional derivative)
    Cost: O(work(f)) per input direction → O(n·work(f)) for full Jacobian.
    Use case: f: ℝⁿ → ℝᵐ with n << m (few inputs, many outputs).

**Reverse-mode (vjp — vector-Jacobian products):**

    Seed with a cotangent ū (direction in output space).
    Propagates: v̄ᵢ = ūᵀ(∂outputs/∂vᵢ)  (output sensitivity)
    Cost: O(work(f)) per output direction → O(m·work(f)) for full Jacobian.
    Use case: f: ℝⁿ → ℝᵐ with m << n (many inputs, few outputs = loss function).

    The magic of backprop: for scalar loss (m=1), reverse-mode computes
    the ENTIRE GRADIENT in one backward pass — this is why it is used.


### Numerical Gradient Checking

Verify an autodiff gradient by comparing to finite differences:

    relative error = ‖∇f_autodiff − ∇f_finitediff‖ / ‖∇f_autodiff + ∇f_finitediff‖

    < 10⁻⁷: implementation almost certainly correct
    10⁻⁵:   probably fine (floating point noise)
    > 10⁻³: likely a bug in the derivative implementation

Use h = 10⁻⁵ for central differences; avoid h too small (catastrophic
cancellation in float64) or too large (Taylor truncation error).


### PART 12 — CALCULUS & OPTIMISATION IN ML: THE UNIFIED VIEW

### Every Major ML Concept Through Optimisation

    ┌─────────────────────────────┬───────────────────────────────────────────┐
    │ Algorithm / Concept         │ Core Calculus / Optimisation              │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Linear regression           │ Convex quadratic: one global min          │
    │                             │ Normal eq. = zero gradient of MSE         │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Logistic regression         │ Convex (log-likelihood is concave)        │
    │                             │ No closed form → gradient descent         │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ L1 regularisation (Lasso)   │ Non-smooth; proximal gradient (soft thr)  │
    │                             │ KKT: sparsity from complementary slack    │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ SVM                         │ Strongly convex QP; KKT gives dual        │
    │                             │ Only support vectors have μᵢ > 0          │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Neural network training     │ Non-convex, many saddle points            │
    │                             │ Backprop = reverse-mode autodiff          │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Batch normalisation         │ Normalise by μ,σ: smoother landscape      │
    │                             │ Reduces effective condition number κ      │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Residual connections        │ Gradient highway: ∂loss/∂xˡ has additive  │
    │                             │ identity path → avoids gradient vanishing │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Dropout                     │ Stochastic regulariser: noisy gradient    │
    │                             │ estimate → implicit Bayesian ensemble     │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Variational autoencoder     │ ELBO = reconstruction − KL divergence     │
    │                             │ Reparameterisation trick enables backprop │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ GAN training                │ Minimax: min_G max_D V(D,G)               │
    │                             │ Non-convex-concave; mode collapse = trap  │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Attention mechanism         │ Softmax = gradient of log-sum-exp         │
    │                             │ Scores are unnormalised; softmax projects │
    │                             │ onto the probability simplex              │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Natural gradient            │ Precondition grad by inverse Fisher info  │
    │                             │ Covariant gradient in parameter space     │
    └─────────────────────────────┴───────────────────────────────────────────┘


### Landscape Theory for Deep Networks

Modern deep networks have loss landscapes with:

    Exponentially many LOCAL MINIMA — but most are nearly equivalent in loss.
    (Dauphin et al., 2014): the error at saddle points is what matters.

    SADDLE POINTS dominate: for a random smooth function in high dimensions,
    most critical points are saddle points, not local minima.
    The probability a critical point is a local min decreases exponentially
    with the number of parameters.

    FLAT REGIONS: large plateaus where ‖∇f‖ ≈ 0 but f >> f*.
    Responsible for the "slow start" in deep learning.

    Sharp vs Flat minima:
        SHARP minima: high curvature (large λ_max of Hessian), poor generalisation.
        FLAT minima: low curvature (small λ_max), better generalisation.
        Large-batch SGD tends toward sharp minima; small-batch finds flat ones.

    Loss of plasticity: over training, network gradients can collapse.
    Periodic re-initialisation or special architectures (LayerNorm, ResNet)
    preserve gradient signal across depth.


### PART 13 — DIFFERENTIAL EQUATIONS FOR ML

### Ordinary Differential Equations (ODEs)

An ODE describes how a state x(t) ∈ ℝⁿ evolves continuously in time:

    dx/dt = f(x(t), t)    (autonomous if f does not depend on t explicitly)

The solution x(t) is a TRAJECTORY through state space.

**Existence & Uniqueness (Picard–Lindelöf):**
    If f is Lipschitz continuous in x, then for any initial condition
    x(t₀) = x₀, a unique solution exists on some interval [t₀, t₀+ε].

**Numerical solvers (discretise continuous dynamics):**

    Euler method (first-order):
        x(t+h) ≈ x(t) + h · f(x(t), t)    (error O(h²) per step)

    Runge-Kutta 4 (fourth-order):
        k₁ = f(x,t),       k₂ = f(x+½hk₁, t+½h)
        k₃ = f(x+½hk₂, t+½h),  k₄ = f(x+hk₃, t+h)
        x(t+h) = x(t) + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)   (error O(h⁵) per step)

    The Euler method = a single gradient descent step with step size h.
    Each forward pass of a ResNet approximates one Euler step of an ODE.


### Neural ODEs

**Chen et al. (2018)** reframed a residual network as a continuous ODE:

    ResNet:    xₗ₊₁ = xₗ + f(xₗ, θₗ)          (discrete residual block)
    NeuralODE: dx/dt = f(x(t), t, θ)            (continuous dynamics)

The forward pass SOLVES the ODE from t=0 to t=T using any black-box solver:

    x(T) = x(0) + ∫₀ᵀ f(x(t), t, θ) dt

    Depth becomes a CONTINUOUS parameter — not a discrete number of layers.
    Memory cost: O(1) with the adjoint method (vs O(L) for backprop through L layers).

**Adjoint method (backpropagation through the ODE solver):**

Define the adjoint a(t) = ∂L/∂x(t)  (how the loss changes with each state).

The adjoint satisfies its own (reverse-time) ODE:

    da/dt = −a(t)ᵀ ∂f/∂x(x(t), t, θ)

Gradients w.r.t. parameters:

    dL/dθ = −∫_T^0 a(t)ᵀ ∂f/∂θ (x(t), t, θ) dt

The adjoint method is implicit differentiation applied to the ODE.
It runs the solver BACKWARDS in time, computing gradients without
storing the forward trajectory:

    ┌────────────────────────────────────────────────────────────┐
    │  Forward:  x(0) → [ODE solver] → x(T)     O(1) memory     │
    │  Backward: a(T) → [adjoint ODE] → a(0)    O(1) memory     │
    │  Compare:  backprop through L Euler steps  O(L) memory     │
    └────────────────────────────────────────────────────────────┘

Adaptive solvers (e.g. Dormand-Prince) automatically choose step sizes
to hit a tolerance target — depth adapts to the complexity of the input.

Applications:
    Continuous normalising flows: learn bijection via ODE dynamics.
    Latent ODEs: model irregular time-series as continuous latent states.
    Second-order Neural ODEs: model Hamiltonian / Lagrangian mechanics.


### Gradient Flow

The GRADIENT FLOW of a function f: ℝⁿ → ℝ is the ODE:

    dx/dt = −∇f(x(t))

This is the CONTINUOUS-TIME LIMIT of gradient descent as step size → 0.

    Gradient descent:  xₜ₊₁ = xₜ − η∇f(xₜ)   (discrete, step size η)
    Gradient flow:     dx/dt = −∇f(x)           (η→0, time becomes continuous)

Properties of gradient flow:
    f decreases monotonically: d/dt f(x(t)) = ∇f · dx/dt = −‖∇f‖² ≤ 0
    Fixed points = critical points: dx/dt = 0 iff ∇f(x) = 0
    For μ-strongly convex f: f(x(t)) − f* ≤ e^{−2μt}[f(x₀)−f*]


### Continuous-Time Optimisation

**Polyak Heavy Ball (continuous ODE form):**

    ẍ + γẋ = −∇f(x)    (damped harmonic oscillator)

    γ > 0: friction / damping coefficient.
    The "ball" rolls down the loss surface with momentum; γ controls damping.
    Optimal γ = 2√μ (critically damped) → convergence O(e^{−√μ t}).

**Nesterov's ODE (Su–Boyd–Candès 2016):**

    ẍ + (3/t)ẋ = −∇f(x)

    The vanishing damping (3/t → 0 as t→∞) is the continuous-time
    explanation for Nesterov's accelerated rate: the system becomes less
    damped over time, allowing it to accelerate toward the minimum.

    ┌────────────────────────────────────────────────────────────┐
    │  Nesterov ODE insight:                                     │
    │  Faster convergence = weaker damping = more oscillation    │
    │  Discretising the ODE recovers the discrete algorithm      │
    └────────────────────────────────────────────────────────────┘

**Langevin Dynamics (stochastic ODE for sampling):**

    dx = −∇f(x) dt + √(2T) dW    (W = Wiener process / Brownian motion)

    The noise term prevents the trajectory from getting stuck at a local
    minimum.  The stationary distribution is the Gibbs distribution:
        π(x) ∝ exp(−f(x)/T)
    Discretised Langevin = Stochastic Gradient Langevin Dynamics (SGLD):
        xₜ₊₁ = xₜ − η∇fᵢ(xₜ) + √(2ηT) ξₜ,   ξₜ ~ N(0,I)
    SGLD simultaneously minimises and samples from the posterior.

**Convergence rate comparison (continuous time):**

    ┌──────────────────────────┬──────────────────────────────────┐
    │ Method                   │ Rate (strongly convex, μ-SC)     │
    ├──────────────────────────┼──────────────────────────────────┤
    │ Gradient flow            │ f − f* ≤ e^{−2μt} [f₀ − f*]     │
    │ Heavy ball (optimal γ)   │ f − f* ≤ e^{−2√μt} [f₀ − f*]    │
    │ Nesterov ODE             │ f − f* = O(1/t²)  (general cvx)  │
    │ Langevin (T→0)           │ converges to global min (non-cvx) │
    └──────────────────────────┴──────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Gradient, Hessian & Critical Points — Geometry of a Loss Surface": {
        "description": (
            "Compute gradients and Hessians analytically and verify with "
            "finite differences. Classify critical points (min, max, saddle) "
            "from Hessian eigenvalues. Visualise the loss landscape of a "
            "2D non-convex function with gradient field and critical points."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


np.set_printoptions(precision=6, suppress=True)

print("=" * 65)
print("  GRADIENT, HESSIAN & CRITICAL POINTS")
print("=" * 65)
print()

# ── Part 1: Gradient and Hessian of a quadratic ───────────────────────────
# f(x,y) = x² + 4y² - 2xy - 4x - 8y  (a convex quadratic)
# ∇f = [2x - 2y - 4, 8y - 2x - 8],  H = [[2, -2], [-2, 8]]
print("  PART 1 — QUADRATIC FUNCTION f(x,y) = x² + 4y² − 2xy − 4x − 8y")
print()

def f_quad(x, y): return x**2 + 4*y**2 - 2*x*y - 4*x - 8*y
def grad_f(x, y): return np.array([2*x - 2*y - 4, 8*y - 2*x - 8])
H_quad = np.array([[2., -2.], [-2., 8.]])

print(f"  H (Hessian) =")
print(f"  {H_quad}")
eigenvalues, eigenvectors = np.linalg.eigh(H_quad)
print(f"  Eigenvalues of H: {eigenvalues.round(4)}")
print(f"  Condition number κ = λ_max/λ_min = {eigenvalues.max()/eigenvalues.min():.4f}")
print()

# Solve ∇f = 0:  [[2,-2],[-2,8]] [x,y]ᵀ = [4,8]ᵀ
b_vec = np.array([4., 8.])
x_star = np.linalg.solve(H_quad, b_vec)
print(f"  Critical point (∇f = 0): x* = {x_star}")
print(f"  All eigenvalues > 0 → this is a LOCAL MINIMUM")
print(f"  f(x*) = {f_quad(*x_star):.4f}")
print()

# Gradient checking
def grad_fd(x, y, h=1e-5):
    df_dx = (f_quad(x+h, y) - f_quad(x-h, y)) / (2*h)
    df_dy = (f_quad(x, y+h) - f_quad(x, y-h)) / (2*h)
    return np.array([df_dx, df_dy])

test_pt = np.array([1.0, 2.0])
g_analytic = grad_f(*test_pt)
g_fd       = grad_fd(*test_pt)
print(f"  Gradient check at {test_pt}:")
print(f"  Analytic:        {g_analytic}")
print(f"  Finite diff:     {g_fd}")
print(f"  Relative error:  {np.linalg.norm(g_analytic - g_fd) / np.linalg.norm(g_analytic):.2e}")
print()

# ── Part 2: Non-convex function with multiple critical points ─────────────
# f(x,y) = sin(x) + cos(y) + 0.1*(x²+y²)  — has local min, max, saddles
print("  PART 2 — NON-CONVEX FUNCTION f(x,y) = sin(x) + cos(y) + 0.1(x²+y²)")
print()

def f_nc(x, y): return np.sin(x) + np.cos(y) + 0.1*(x**2 + y**2)
def grad_nc(x, y): return np.array([np.cos(x) + 0.2*x, -np.sin(y) + 0.2*y])
def hess_nc(x, y):
    return np.array([[-np.sin(x) + 0.2, 0.0],
                     [0.0, -np.cos(y) + 0.2]])

# Find critical points by scanning + Newton polish
from scipy.optimize import fsolve
critical_pts = []
for x0 in np.linspace(-5, 5, 8):
    for y0 in np.linspace(-5, 5, 8):
        sol = fsolve(lambda p: grad_nc(*p), [x0, y0], full_output=True)
        pt  = sol[0]
        if sol[2] == 1:  # converged
            # Deduplicate
            is_new = all(np.linalg.norm(pt - p) > 0.1 for p in critical_pts)
            if is_new and np.linalg.norm(grad_nc(*pt)) < 1e-8:
                critical_pts.append(pt)

print(f"  Found {len(critical_pts)} critical points in [-5,5]²:")
print()
print(f"  {'Point':>22} | {'f(x,y)':>8} | {'λ_min':>8} | {'λ_max':>8} | Type")
print(f"  {'─'*68}")
for pt in sorted(critical_pts, key=lambda p: p[0]):
    H_pt = hess_nc(*pt)
    ev   = np.linalg.eigvalsh(H_pt)
    fval = f_nc(*pt)
    if ev.min() > 1e-10:
        kind = "LOCAL MIN"
    elif ev.max() < -1e-10:
        kind = "LOCAL MAX"
    else:
        kind = "SADDLE"
    print(f"  ({pt[0]:+.4f}, {pt[1]:+.4f})       | {fval:8.4f} | {ev.min():8.4f} | {ev.max():8.4f} | {kind}")
print()

# ── Part 3: Visualisation ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Gradient, Hessian & Critical Points of f(x,y)", fontsize=12, fontweight="bold")

# Plot 1: Convex quadratic with gradient field
xg, yg = np.meshgrid(np.linspace(-1, 5, 60), np.linspace(-1, 5, 60))
Zq = f_quad(xg, yg)
axes[0].contourf(xg, yg, Zq, 25, cmap="coolwarm", alpha=0.75)
axes[0].contour(xg, yg, Zq, 15, colors="white", linewidths=0.5, alpha=0.6)
gx = 2*xg - 2*yg - 4
gy = 8*yg - 2*xg - 8
sk = 5
axes[0].quiver(xg[::sk,::sk], yg[::sk,::sk], -gx[::sk,::sk], -gy[::sk,::sk],
               alpha=0.7, color="white", scale=150)
axes[0].plot(*x_star, "k*", markersize=14, label=f"min ({x_star[0]:.2f},{x_star[1]:.2f})")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_title("Convex Quadratic\\n(−∇f arrows point to min)")
axes[0].legend(fontsize=9); axes[0].set_aspect("equal")

# Plot 2: Non-convex surface with critical points
xg2, yg2 = np.meshgrid(np.linspace(-5, 5, 120), np.linspace(-5, 5, 120))
Znc = f_nc(xg2, yg2)
axes[1].contourf(xg2, yg2, Znc, 30, cmap="RdYlGn", alpha=0.8)
axes[1].contour(xg2, yg2, Znc, 20, colors="k", linewidths=0.4, alpha=0.5)
colors_map = {"LOCAL MIN": "blue", "LOCAL MAX": "red", "SADDLE": "orange"}
marker_map  = {"LOCAL MIN": "v", "LOCAL MAX": "^", "SADDLE": "D"}
plotted = set()
for pt in critical_pts:
    H_pt = hess_nc(*pt)
    ev   = np.linalg.eigvalsh(H_pt)
    if ev.min() > 1e-10:   kind = "LOCAL MIN"
    elif ev.max() < -1e-10: kind = "LOCAL MAX"
    else:                   kind = "SADDLE"
    label = kind if kind not in plotted else "_"
    axes[1].plot(*pt, marker_map[kind], color=colors_map[kind],
                 markersize=10, label=label, markeredgecolor="k", markeredgewidth=0.8)
    plotted.add(kind)
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
axes[1].set_title("Non-Convex Landscape\\n(min▼ max▲ saddle◆)")
axes[1].legend(fontsize=8, loc="upper right"); axes[1].set_aspect("equal")

# Plot 3: Hessian eigenvalue interpretation
angles = np.linspace(0, 2*np.pi, 200)
fig3_pts = [
    (np.array([[ 2., -2.], [-2., 8.]]),   "Quadratic H\\n(Convex: λ>0)",    "steelblue"),
    (np.array([[-2., 0.], [0., -3.]]),     "Negative def H\\n(Concave: λ<0)", "tomato"),
    (np.array([[3., 0.], [0., -1.]]),      "Indefinite H\\n(Saddle)",         "orange"),
]
for ax_idx, (H_ex, title, col) in enumerate(fig3_pts):
    ev, evec = np.linalg.eigh(H_ex)
    # Draw ellipse aligned with eigenvectors, radii = 1/|λ|
    scale = 0.8
    t = np.linspace(0, 2*np.pi, 300)
    rx = scale / (abs(ev[0]) + 1e-2)
    ry = scale / (abs(ev[1]) + 1e-2)
    ellipse_local = np.stack([rx*np.cos(t), ry*np.sin(t)])
    ellipse_world = evec @ ellipse_local
    axes[2].plot(ax_idx*4 + ellipse_world[0], ellipse_world[1], color=col, lw=2, label=title)
    for i, (evc, evl) in enumerate(zip(evec.T, ev)):
        sign = "+" if evl > 0 else "−"
        c = "green" if evl > 0 else "red"
        axes[2].annotate("", xy=(ax_idx*4 + evc[0]*scale*0.9, evc[1]*scale*0.9),
                         xytext=(ax_idx*4, 0),
                         arrowprops=dict(arrowstyle="->", color=c, lw=2))
axes[2].axhline(0, color="k", lw=0.5)
axes[2].set_xlim(-2, 11); axes[2].set_ylim(-2.5, 2.5)
axes[2].set_title("Hessian Eigenvectors & Curvature\\n(green=+, red=−)")
axes[2].legend(fontsize=8, loc="upper right")
axes[2].set_aspect("equal"); axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "critical_points.png", dpi=120)
print("  Plot saved → critical_points.png")
print()
print("  KEY TAKEAWAYS:")
print("  ∇f = 0 is necessary for a minimum, NOT sufficient.")
print("  H ≻ 0 → local min;  H ≺ 0 → local max;  indefinite → saddle.")
print("  Non-convex functions have many saddle points — not just minima.")
print("  κ(H) = λ_max/λ_min determines how elongated the loss landscape is.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Taylor Series — Approximations & Error Bounds": {
        "description": (
            "Build and visualise Taylor polynomial approximations of order 1, 2, 4 "
            "for sin(x) and the sigmoid. Quantify the O(hⁿ) approximation error. "
            "Demonstrate the Descent Lemma: verify f(y) ≤ f(x)+∇fᵀ(y-x)+L/2‖y-x‖² "
            "and show how the Lipschitz constant L gives the safe step size 1/L."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from scipy.special import factorial

print("=" * 65)
print("  TAYLOR SERIES: APPROXIMATIONS & ERROR BOUNDS")
print("=" * 65)
print()

# ── Part 1: Taylor polynomials of sin(x) around x=0 ──────────────────────
print("  PART 1 — TAYLOR POLYNOMIALS OF sin(x) AROUND x = 0")
print()

def sin_taylor(x, order):
    """Sum of Taylor terms for sin up to given order."""
    result = np.zeros_like(x, dtype=float)
    for k in range(order // 2 + 1):
        n = 2*k + 1  # odd powers only
        if n > order:
            break
        sign = (-1)**k
        result += sign * x**n / factorial(n)
    return result

x = np.linspace(-2*np.pi, 2*np.pi, 500)
orders = [1, 3, 5, 9]

print(f"  {'Order':>8} | {'Max |error| on [-π,π]':>25} | {'Max |error| on [-2π,2π]':>25}")
print(f"  {'─'*65}")
x_inner = np.linspace(-np.pi, np.pi, 300)
x_outer = np.linspace(-2*np.pi, 2*np.pi, 300)
for o in orders:
    err_inner = np.max(np.abs(np.sin(x_inner) - sin_taylor(x_inner, o)))
    err_outer = np.max(np.abs(np.sin(x_outer) - sin_taylor(x_outer, o)))
    print(f"  {o:>8} | {err_inner:25.6e} | {err_outer:25.6e}")
print()
print("  Error shrinks rapidly on compact domain as order increases.")
print("  But sin has infinite radius of convergence — error → 0 for all x.")
print()

# ── Part 2: Descent Lemma — L-smooth functions ────────────────────────────
print("  PART 2 — THE DESCENT LEMMA & SAFE STEP SIZE")
print()
# f(x) = (1/2)x² + sin(x)   on ℝ
# f'(x) = x + cos(x)
# f''(x) = 1 - sin(x)   → max |f''| = 2   → L = 2
def f2(x):    return 0.5*x**2 + np.sin(x)
def gradf2(x): return x + np.cos(x)
def hessf2(x): return 1.0 - np.sin(x)

L = 2.0   # max |f''(x)| = 2
print(f"  f(x) = ½x² + sin(x),  f'(x) = x+cos(x),  f''(x) = 1−sin(x)")
print(f"  max |f''(x)| = {L:.1f}  → Lipschitz constant L = {L:.1f}")
print(f"  Safe step size for gradient descent: η ≤ 1/L = {1/L:.3f}")
print()

x0_pts = [-1.5, 0.5, 2.0]
step_sizes = [0.3, 0.5, 1/L, 0.8, 1.5]
print(f"  Descent Lemma verification: f(y) ≤ f(x) + ∇f(x)ᵀ(y−x) + L/2‖y−x‖²")
print()
x0 = 1.0  # test point
gx0 = gradf2(x0)
print(f"  At x₀ = {x0}: f(x₀) = {f2(x0):.4f}, ∇f(x₀) = {gx0:.4f}")
print()
print(f"  {'η (step)':>10} | {'y = x−η∇f':>12} | {'f(y)':>10} | {'Upper bound':>12} | {'Satisfied?':>10}")
print(f"  {'─'*60}")
for eta in step_sizes:
    y   = x0 - eta * gx0
    fy  = f2(y)
    ub  = f2(x0) + gx0*(y - x0) + (L/2)*(y - x0)**2
    sat = "✓" if fy <= ub + 1e-10 else "✗  VIOLATED"
    print(f"  {eta:10.3f} | {y:12.4f} | {fy:10.4f} | {ub:12.4f} | {sat}")
print()
print(f"  For η > 1/L = {1/L:.2f}, the quadratic upper bound may fail → divergence risk.")
print()

# ── Part 3: Finite difference gradient checking ───────────────────────────
print("  PART 3 — FINITE DIFFERENCE GRADIENT CHECKING")
print()
def f_test(v):
    x, y, z = v
    return np.sin(x)*y + np.exp(-z)*x**2 + y*z

def grad_test(v):
    x, y, z = v
    return np.array([np.cos(x)*y + 2*x*np.exp(-z),
                     np.sin(x) + z,
                     -np.exp(-z)*x**2 + y])

v0 = np.array([0.5, -1.2, 0.8])
g_analytic = grad_test(v0)

print(f"  f(x,y,z) = sin(x)y + e^(-z)x² + yz")
print(f"  At v₀ = {v0}")
print()
print(f"  {'h':>12} | {'‖grad_FD − grad_analytic‖':>28} | {'type'}")
print(f"  {'─'*55}")

for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    # Central finite difference
    g_fd = np.zeros(3)
    for i in range(3):
        ei = np.zeros(3); ei[i] = h
        g_fd[i] = (f_test(v0+ei) - f_test(v0-ei)) / (2*h)
    rel_err = np.linalg.norm(g_fd - g_analytic) / (np.linalg.norm(g_analytic)+1e-15)
    kind = "← optimal" if 1e-6 <= h <= 1e-5 else ("← rounding error" if h < 1e-8 else "")
    print(f"  {h:12.0e} | {rel_err:28.2e} | {kind}")
print()
print("  Too large h → truncation error O(h²).  Too small → float cancellation.")
print("  Central differences optimal around h=1e-5 for float64.")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Taylor Series: Approximations & Error Bounds", fontsize=12, fontweight="bold")

# Plot 1: Taylor polynomials of sin
colours = ["tomato", "orange", "steelblue", "seagreen"]
axes[0].plot(x, np.sin(x), "k-", lw=2.5, label="sin(x)")
for o, c in zip(orders, colours):
    axes[0].plot(x, sin_taylor(x, o), "--", color=c, lw=1.5, label=f"Order {o}")
axes[0].set_ylim(-3, 3)
axes[0].axhline(0, color="gray", lw=0.5)
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].set_title("Taylor Approximations of sin(x)\\nAround x = 0")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Plot 2: Descent Lemma illustration
x_range = np.linspace(-0.5, 3.5, 300)
x0_show = 2.0
gx0_show = gradf2(x0_show)
fx0_show = f2(x0_show)
lin  = fx0_show + gx0_show * (x_range - x0_show)
quad = fx0_show + gx0_show * (x_range - x0_show) + (L/2) * (x_range - x0_show)**2
axes[1].plot(x_range, f2(x_range), "k-", lw=2.5, label="f(x)")
axes[1].plot(x_range, lin,  "orange",  lw=1.8, linestyle="--", label="First-order (tangent)")
axes[1].plot(x_range, quad, "steelblue", lw=1.8, linestyle="--",
             label=f"Quadratic upper bound (L={L})")
axes[1].axvline(x0_show, color="gray", lw=0.8, linestyle=":")
axes[1].plot(x0_show, fx0_show, "ko", markersize=8, label=f"x₀={x0_show}")
y_gd = x0_show - (1/L)*gx0_show
axes[1].plot(y_gd, f2(y_gd), "b*", markersize=12, label=f"y=x₀−(1/L)∇f")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].set_title("Descent Lemma\\nf(y) ≤ quadratic upper bound")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
axes[1].set_ylim(-2, 8)

# Plot 3: Gradient check error vs h
hs = np.logspace(-12, 0, 60)
errs = []
for h in hs:
    g_fd = np.zeros(3)
    for i in range(3):
        ei = np.zeros(3); ei[i] = h
        g_fd[i] = (f_test(v0+ei) - f_test(v0-ei)) / (2*h)
    errs.append(np.linalg.norm(g_fd - g_analytic) / (np.linalg.norm(g_analytic)+1e-15))
axes[2].loglog(hs, errs, "steelblue", lw=2)
axes[2].axvline(1e-5, color="tomato", linestyle="--", lw=1.5, label="Optimal h ≈ 1e-5")
axes[2].set_xlabel("Step size h")
axes[2].set_ylabel("Relative gradient error")
axes[2].set_title("Finite Difference Gradient Check\\nError vs Step Size h")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "taylor_descent.png", dpi=120)
print("  Plot saved → taylor_descent.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Convexity — Checking, Visualising & Jensen's Inequality": {
        "description": (
            "Verify convexity of a function via the Hessian PSD condition, "
            "the chord-above-curve definition, and the first-order tangent "
            "lower bound. Visualise convex vs non-convex landscapes and sublevel sets. "
            "Demonstrate Jensen's inequality numerically with different distributions."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


np.random.seed(42)

print("=" * 65)
print("  CONVEXITY: CHECKING, VISUALISING & JENSEN'S INEQUALITY")
print("=" * 65)
print()

# ── Part 1: Three definitions of convexity ────────────────────────────────
print("  PART 1 — THREE EQUIVALENT DEFINITIONS OF CONVEXITY")
print()
# f(x) = x⁴ - 4x² + 5  (non-convex — has two local minima)
# g(x) = x⁴ + x²        (convex — f'' = 12x²+2 > 0 always)
def f_nc(x): return x**4 - 4*x**2 + 5
def f_cx(x): return x**4 + x**2
def g_cx(x): return np.exp(x)     # exp is the prototypical convex function

functions = [
    ("f(x) = x⁴ − 4x² + 5",  f_nc, False),
    ("g(x) = x⁴ + x²",        f_cx, True),
    ("h(x) = eˣ",              g_cx, True),
]

for name, func, expected_convex in functions:
    print(f"  --- {name} ---")
    # Check 1: Chord (definition 1)
    x_vals = np.linspace(-2, 2, 50)
    chord_violations = 0
    for _ in range(2000):
        xa, xb = np.random.uniform(-2, 2, 2)
        theta  = np.random.uniform(0, 1)
        lhs    = func(theta*xa + (1-theta)*xb)
        rhs    = theta*func(xa) + (1-theta)*func(xb)
        if lhs > rhs + 1e-10:
            chord_violations += 1
    print(f"  Chord violations (2000 random pairs): {chord_violations}  →  "
          + ("CONVEX" if chord_violations == 0 else "NON-CONVEX"))

    # Check 2: f''(x) ≥ 0  (second-order condition, numerical)
    h_fd = 1e-5
    x_check = np.linspace(-2, 2, 200)
    f2p = (func(x_check+h_fd) - 2*func(x_check) + func(x_check-h_fd)) / h_fd**2
    print(f"  min f''(x) on [-2,2]: {f2p.min():.4f}  →  "
          + ("f'' ≥ 0 ✓" if f2p.min() >= -1e-6 else "f'' < 0 at some point"))

    # Check 3: Tangent lower bound (first-order condition)
    fx0 = func(np.array(0.5))
    gprime = (func(np.array(0.5+h_fd)) - func(np.array(0.5-h_fd))) / (2*h_fd)
    x_test = np.linspace(-2, 2, 100)
    lb_violations = np.sum(func(x_test) < fx0 + gprime*(x_test - 0.5) - 1e-8)
    print(f"  Tangent lower bound violations: {lb_violations}  →  "
          + ("tangent is global lb ✓" if lb_violations == 0 else "tangent exceeds function"))
    print()

# ── Part 2: Sublevel sets ─────────────────────────────────────────────────
print("  PART 2 — SUBLEVEL SETS  {x : f(x) ≤ c}  (convex iff f convex)")
print()
# For 2D:  f(x,y) = x² + 2y²   (convex, elliptical sublevel sets)
#          g(x,y) = x² - y²    (non-convex saddle, hyperbolic "sublevel sets")
xg, yg = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z_cvx  = xg**2 + 2*yg**2
Z_ncvx = xg**2 - yg**2

for name, Z in [("f(x,y)=x²+2y² (CONVEX)", Z_cvx),
                ("g(x,y)=x²-y² (NON-CONVEX)", Z_ncvx)]:
    levels = [0.5, 1.0, 2.0]
    for c in levels:
        sublevel_area = np.mean(Z <= c)
        print(f"  {name}: sublevel {{z≤{c}}} area fraction ≈ {sublevel_area:.3f}")
print()

# ── Part 3: Jensen's Inequality ───────────────────────────────────────────
print("  PART 3 — JENSEN'S INEQUALITY: f(E[X]) ≤ E[f(X)]  for convex f")
print()
print(f"  {'Distribution':>20} | {'E[X]':>8} | {'f(E[X])':>10} | {'E[f(X)]':>10} | {'Gap':>8}")
print(f"  {'─'*62}")

# f(x) = eˣ  (convex),  several distributions
f_j = np.exp
distributions = {
    "Uniform(-1,1)":   np.random.uniform(-1, 1, 100000),
    "Normal(0,1)":     np.random.normal(0, 1, 100000),
    "Normal(0,0.1)":   np.random.normal(0, 0.1, 100000),
    "Bernoulli±2":     np.random.choice([-2., 2.], 100000),
    "Bernoulli±0.1":   np.random.choice([-0.1, 0.1], 100000),
}
for dist_name, samples in distributions.items():
    ex   = np.mean(samples)
    fex  = f_j(ex)
    efx  = np.mean(f_j(samples))
    gap  = efx - fex
    print(f"  {dist_name:>20} | {ex:8.4f} | {fex:10.4f} | {efx:10.4f} | {gap:8.4f}")

print()
print("  Gap is larger when variance of X is higher (convexity × variance).")
print("  For a Dirac mass (X=constant): gap = 0.  For wider distributions: gap > 0.")
print()
print("  Jensen in ML:  KL(q‖p) = E_q[log q - log p] = E_q[log q/p] ≥ 0")
print("  because -log is convex: −log E[p/q] ≤ E[−log p/q] = KL(q‖p)")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Convexity: Three Definitions & Jensen's Inequality", fontsize=12, fontweight="bold")

# Plot 1: Convex vs Non-convex with chord
x_plot = np.linspace(-2.5, 2.5, 400)
xa_demo, xb_demo = -1.5, 1.8
thetas  = np.linspace(0, 1, 50)
axes[0].plot(x_plot, f_nc(x_plot), "tomato", lw=2.5, label="Non-convex f")
axes[0].plot(x_plot, f_cx(x_plot), "steelblue", lw=2.5, label="Convex g")
# Chord for non-convex
chord_y_nc = f_nc(xa_demo) + (f_nc(xb_demo)-f_nc(xa_demo))/(xb_demo-xa_demo)*(x_plot-xa_demo)
axes[0].plot([xa_demo, xb_demo], [f_nc(xa_demo), f_nc(xb_demo)], "r--", lw=2, alpha=0.7)
# Shade violation region
for t in thetas:
    xm = t*xa_demo + (1-t)*xb_demo
    ym_curve = f_nc(xm)
    ym_chord = t*f_nc(xa_demo) + (1-t)*f_nc(xb_demo)
    if ym_curve > ym_chord:
        axes[0].plot([xm, xm], [ym_curve, ym_chord], "r", alpha=0.15, lw=3)
# Chord for convex (lies above)
axes[0].plot([xa_demo, xb_demo], [f_cx(xa_demo), f_cx(xb_demo)], "b--", lw=2, alpha=0.7)
axes[0].set_title("Chord Above Curve (Convexity)\\nRed shading = violation")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")

# Plot 2: Sublevel sets
cs = axes[1].contourf(xg, yg, Z_cvx, levels=[0, 0.5, 1.0, 2.0, 5.0], cmap="Blues", alpha=0.7)
axes[1].contour(xg, yg, Z_cvx, levels=[0.5, 1.0, 2.0], colors="steelblue", linewidths=2)
plt.colorbar(cs, ax=axes[1], label="f(x,y) = x² + 2y²")
axes[1].set_title("Convex Sublevel Sets\\n{(x,y): f(x,y) ≤ c} are ellipses (convex)")
axes[1].set_aspect("equal"); axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

# Plot 3: Jensen's inequality illustration
x_j = np.linspace(-3, 3, 300)
fj  = np.exp(x_j)
samples_j = np.random.normal(0, 1, 10000)
ex_j  = np.mean(samples_j); fex_j = np.exp(ex_j); efx_j = np.mean(np.exp(samples_j))
axes[2].plot(x_j, fj, "k-", lw=2.5, label="f(x) = eˣ (convex)")
axes[2].hist(samples_j, bins=60, density=True, alpha=0.3, color="steelblue",
             label="X ~ N(0,1)", bottom=0)
axes[2].axvline(ex_j, color="steelblue", lw=2, linestyle="--", label=f"E[X]={ex_j:.2f}")
axes[2].plot(ex_j, fex_j, "bs", markersize=11, label=f"f(E[X])={fex_j:.3f}")
axes[2].plot(ex_j, efx_j, "r^", markersize=11, label=f"E[f(X)]={efx_j:.3f}")
axes[2].annotate("", xy=(ex_j, efx_j), xytext=(ex_j, fex_j),
                 arrowprops=dict(arrowstyle="<->", color="tomato", lw=2))
axes[2].text(ex_j + 0.1, (fex_j+efx_j)/2, f"Gap={efx_j-fex_j:.3f}", color="tomato", fontsize=10)
axes[2].set_xlim(-3, 3); axes[2].set_ylim(-0.5, 8)
axes[2].set_title("Jensen's Inequality\\nf(E[X]) ≤ E[f(X)] for convex f")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
axes[2].set_xlabel("x"); axes[2].set_ylabel("f(x) / density")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "convexity.png", dpi=120)
print("  Plot saved → convexity.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · KKT Conditions — Lagrange Multipliers & Constrained Optimisation": {
        "description": (
            "Solve constrained optimisation problems using the KKT conditions. "
            "Derive and verify Lagrange multipliers for equality constraints. "
            "Demonstrate complementary slackness for inequality constraints. "
            "Solve the SVM primal/dual analytically on a toy 2-class dataset "
            "and verify that only support vectors have non-zero dual variables."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from scipy.optimize import minimize

np.random.seed(7)
np.set_printoptions(precision=5, suppress=True)

print("=" * 65)
print("  KKT CONDITIONS: LAGRANGE MULTIPLIERS & CONSTRAINED OPTIMISATION")
print("=" * 65)
print()

# ── Part 1: Equality constraint — Lagrange multipliers ────────────────────
print("  PART 1 — EQUALITY CONSTRAINT: LAGRANGE MULTIPLIERS")
print()
print("  Problem:  min f(x,y) = x² + 2y²")
print("            s.t. h(x,y) = x + 2y − 3 = 0")
print()
print("  Lagrangian:  L(x,y,λ) = x² + 2y² + λ(x + 2y − 3)")
print()
print("  Stationarity:  ∂L/∂x = 2x + λ = 0  →  x = -λ/2")
print("                 ∂L/∂y = 4y + 2λ = 0 →  y = -λ/2")
print("  Primal feas:   x + 2y = 3  →  -λ/2 + 2(-λ/2) = 3  →  -3λ/2 = 3  →  λ = -2")
print()
lam_star = -2.0
x_star = -lam_star/2
y_star = -lam_star/2
print(f"  Solution:  λ* = {lam_star},  x* = {x_star},  y* = {y_star}")
print(f"  Objective at optimum: f(x*,y*) = {x_star**2 + 2*y_star**2:.4f}")
print(f"  Constraint satisfied: x* + 2y* = {x_star + 2*y_star:.4f} = 3 ✓")
print()

# Verify ∇f = -λ∇h at optimum
grad_f = np.array([2*x_star, 4*y_star])
grad_h = np.array([1., 2.])
print(f"  ∇f(x*) = {grad_f}")
print(f"  λ∇h(x*) = {-lam_star * grad_h}   (stationarity: ∇f + λ∇h = 0 ✓)")
print()

# Geometric check: normals are parallel
cos_angle = np.dot(grad_f, grad_h) / (np.linalg.norm(grad_f) * np.linalg.norm(grad_h))
print(f"  cos(angle between ∇f and ∇h) = {cos_angle:.4f}  (=±1 → they are parallel ✓)")
print()

# ── Part 2: Inequality constraint — KKT & complementary slackness ─────────
print("  PART 2 — INEQUALITY CONSTRAINT: KKT & COMPLEMENTARY SLACKNESS")
print()
print("  Problem:  min f(x) = (x−3)²")
print("            s.t. g(x) = x − 1 ≤ 0  (i.e. x ≤ 1)")
print()
# Unconstrained min at x=3, but g(3)=2>0 violates constraint.
# KKT: x* = 1 (active), μ* > 0
print("  Unconstrained min: x=3, but g(3)=2 > 0 — CONSTRAINT VIOLATED.")
print()
print("  Active constraint case (g(x*) = 0 → x* = 1):")
x_c = 1.0
mu_star = 2*(x_c - 3)*(-1)  # from stationarity: 2(x-3) + μ = 0
print(f"    Stationarity: 2(x*-3) + μ = 0 → μ* = {mu_star:.4f}")
print(f"    Dual feas: μ* = {mu_star:.4f} ≥ 0 ✓")
print(f"    Comp. slack: μ* · g(x*) = {mu_star:.4f} × {x_c - 1:.4f} = 0 ✓")
print(f"    f(x*) = {(x_c-3)**2:.4f}")
print()

print("  Inactive constraint case: min f(x) = (x−3)²  s.t.  x − 5 ≤ 0")
x_c2 = 3.0  # unconstrained min, constraint inactive
mu2  = 0.0
print(f"    Unconstrained min x*=3 satisfies g(x*)=3-5=-2 < 0 — INACTIVE.")
print(f"    μ* = {mu2}  (comp. slack: μ*·g(x*)=0 because μ*=0)")
print(f"    Both cases: μ*·g(x*) = 0  (one of μ*, g(x*) must be zero)")
print()

# ── Part 3: Toy SVM — KKT and support vectors ─────────────────────────────
print("  PART 3 — TOY SVM: KKT CONDITIONS & SUPPORT VECTORS")
print()
# Generate linearly separable data in 2D
n_each = 6
X_pos = np.random.randn(n_each, 2) + np.array([2.0,  1.0])
X_neg = np.random.randn(n_each, 2) + np.array([-2.0, -1.0])
X = np.vstack([X_pos, X_neg])
y = np.array([1.]*n_each + [-1.]*n_each)

# Solve SVM dual: max Σαᵢ - ½ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢxⱼ
#                s.t. 0 ≤ αᵢ ≤ C,  Σαᵢyᵢ = 0
C = 1e6  # hard margin approximation

n = len(y)
K = (X @ X.T) * np.outer(y, y)  # kernel matrix (linear kernel)

from scipy.optimize import minimize as sp_min

def neg_dual(alpha):
    return -(np.sum(alpha) - 0.5 * alpha @ K @ alpha)

def neg_dual_grad(alpha):
    return -(np.ones(n) - K @ alpha)

constraints = [{"type": "eq", "fun": lambda a: np.dot(a, y)}]
bounds = [(0, C)] * n
alpha0 = np.zeros(n)
res = sp_min(neg_dual, alpha0, jac=neg_dual_grad, method="SLSQP",
             bounds=bounds, constraints=constraints,
             options={"ftol": 1e-10, "maxiter": 500})
alpha = res.x

# Recover primal
w = (alpha * y) @ X
# Bias from support vectors
sv_mask = alpha > 1e-4
b_vals  = y[sv_mask] - X[sv_mask] @ w
b       = np.mean(b_vals)

print(f"  SVM solved on {n} points ({n_each} per class)")
print(f"  w* = {w.round(4)}")
print(f"  b* = {b:.4f}")
print()
print(f"  {'i':>4} | {'αᵢ':>10} | {'yᵢ':>4} | {'yᵢ(wᵀxᵢ+b)':>14} | {'Role'}")
print(f"  {'─'*52}")
for i in range(n):
    margin_i = y[i] * (X[i] @ w + b)
    role     = "SUPPORT VECTOR ★" if alpha[i] > 1e-4 else "non-SV"
    print(f"  {i:4d} | {alpha[i]:10.5f} | {int(y[i]):4d} | {margin_i:14.6f} | {role}")
print()
print("  Complementary slackness: αᵢ > 0 ↔ yᵢ(wᵀxᵢ+b) = 1 (on margin boundary)")
print("  Non-support vectors: αᵢ = 0, constraint inactive (margin > 1)")
print()
margin = 2 / np.linalg.norm(w)
print(f"  Margin = 2/‖w‖ = {margin:.4f}")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("KKT Conditions: Lagrange Multipliers & SVM", fontsize=12, fontweight="bold")

# Plot 1: Equality constraint
xp = np.linspace(-1, 4, 300)
yp = np.linspace(-1, 4, 300)
Xmesh, Ymesh = np.meshgrid(xp, yp)
Z_obj = Xmesh**2 + 2*Ymesh**2
axes[0].contourf(Xmesh, Ymesh, Z_obj, 20, cmap="Blues", alpha=0.7)
axes[0].contour(Xmesh, Ymesh, Z_obj, 15, colors="steelblue", linewidths=0.8, alpha=0.6)
axes[0].plot(xp, (3 - xp)/2, "k-", lw=2.5, label="Constraint: x+2y=3")
axes[0].plot(x_star, y_star, "r*", markersize=16, label=f"x*=({x_star:.1f},{y_star:.1f})")
gf_n = grad_f / np.linalg.norm(grad_f) * 0.5
gh_n = grad_h / np.linalg.norm(grad_h) * 0.5
axes[0].annotate("", xy=(x_star+gf_n[0], y_star+gf_n[1]), xytext=(x_star, y_star),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
axes[0].annotate("∇f", (x_star+gf_n[0]+0.05, y_star+gf_n[1]), fontsize=10, color="red")
axes[0].annotate("", xy=(x_star+gh_n[0], y_star+gh_n[1]), xytext=(x_star, y_star),
                 arrowprops=dict(arrowstyle="->", color="green", lw=2))
axes[0].annotate("∇h", (x_star+gh_n[0]+0.05, y_star+gh_n[1]), fontsize=10, color="green")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_title("Equality Constraint\\n∇f ‖ ∇h at optimum")
axes[0].legend(fontsize=9); axes[0].set_aspect("equal")
axes[0].set_xlim(-0.5, 3.5); axes[0].set_ylim(-0.5, 3.5)

# Plot 2: Complementary slackness illustration
ineq_x = np.linspace(-1, 5, 300)
axes[1].plot(ineq_x, (ineq_x-3)**2, "steelblue", lw=2.5, label="f(x)=(x-3)²")
axes[1].axvline(1, color="tomato", lw=2.5, linestyle="--", label="Constraint x≤1")
axes[1].plot(1, (1-3)**2, "r*", markersize=14, label=f"x*=1, f*=4 (active)")
axes[1].plot(3, 0, "go", markersize=12, label="Unconstrained min (infeasible)")
axes[1].fill_betweenx([0, 12], 1, 5, alpha=0.15, color="tomato", label="Infeasible region")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].set_title("Complementary Slackness\\nμ*·g(x*)=0 (active constraint)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
axes[1].set_xlim(-1, 5); axes[1].set_ylim(-0.5, 12)

# Plot 3: SVM
c_p = "steelblue"; c_n = "tomato"
axes[2].scatter(X_pos[:,0], X_pos[:,1], c=c_p, s=60, label="+1", zorder=3, edgecolors="k")
axes[2].scatter(X_neg[:,0], X_neg[:,1], c=c_n, s=60, label="-1", zorder=3, edgecolors="k")
# Highlight support vectors
sv_idx = np.where(sv_mask)[0]
axes[2].scatter(X[sv_idx,0], X[sv_idx,1], s=220, facecolors="none",
                edgecolors="gold", linewidths=2.5, zorder=4, label="Support Vectors")
# Decision boundary and margins
x1_rng = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200)
if abs(w[1]) > 1e-8:
    db   = (-w[0]*x1_rng - b)  / w[1]
    mp   = (-w[0]*x1_rng - b + 1) / w[1]
    mn   = (-w[0]*x1_rng - b - 1) / w[1]
    axes[2].plot(x1_rng, db, "k-",  lw=2.5, label="Decision boundary")
    axes[2].plot(x1_rng, mp, "k--", lw=1.5, alpha=0.7, label="Margin")
    axes[2].plot(x1_rng, mn, "k--", lw=1.5, alpha=0.7)
axes[2].set_xlabel("x₁"); axes[2].set_ylabel("x₂")
axes[2].set_title(f"SVM: KKT & Support Vectors\\nMargin={margin:.3f}")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kkt_svm.png", dpi=120)
print("  Plot saved → kkt_svm.png")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Optimisation Algorithms — GD, Momentum, Adam & Convergence": {
        "description": (
            "Compare gradient descent, momentum, Nesterov, RMSProp, and Adam "
            "on a quadratic (convex) and Rosenbrock (non-convex) objective. "
            "Track loss, gradient norm, and step sizes over iterations. "
            "Visualise the optimisation trajectory on the loss landscape to show "
            "how curvature, momentum, and adaptive scaling affect convergence."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


np.random.seed(0)

print("=" * 65)
print("  OPTIMISATION ALGORITHMS: GD, MOMENTUM, ADAM & CONVERGENCE")
print("=" * 65)
print()

# ── Objective 1: Ill-conditioned quadratic ────────────────────────────────
# f(x) = ½xᵀAx − bᵀx,  A = diag(1, 100)  → κ = 100
A_q   = np.diag([1.0, 100.0])
b_q   = np.array([1.0, 100.0])
x_opt_q = np.linalg.solve(A_q, b_q)   # = [1, 1]

def f_quad(x):    return 0.5 * x @ A_q @ x - b_q @ x
def grad_quad(x): return A_q @ x - b_q

# ── Objective 2: Rosenbrock (non-convex) ──────────────────────────────────
# f(x,y) = (1-x)² + 100(y-x²)²   global min at (1,1)
def f_rosen(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosen(x):
    gx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    gy = 200*(x[1] - x[0]**2)
    return np.array([gx, gy])

# ── Optimiser implementations ─────────────────────────────────────────────
def run_gd(f, grad, x0, lr, n_iters):
    x = x0.copy(); hist = [f(x)]
    for _ in range(n_iters):
        x -= lr * grad(x)
        hist.append(f(x))
    return x, np.array(hist)

def run_momentum(f, grad, x0, lr, beta, n_iters):
    x = x0.copy(); m = np.zeros_like(x); hist = [f(x)]
    for _ in range(n_iters):
        m = beta*m + grad(x)
        x -= lr * m
        hist.append(f(x))
    return x, np.array(hist)

def run_nesterov(f, grad, x0, lr, beta, n_iters):
    x = x0.copy(); v = np.zeros_like(x); hist = [f(x)]
    for _ in range(n_iters):
        x_look = x - beta * v
        v = beta * v + lr * grad(x_look)
        x = x - v
        hist.append(f(x))
    return x, np.array(hist)

def run_adam(f, grad, x0, lr, beta1, beta2, eps, n_iters):
    x  = x0.copy(); m = np.zeros_like(x); v = np.zeros_like(x)
    hist = [f(x)]
    for t in range(1, n_iters+1):
        g = grad(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        mh = m / (1 - beta1**t)
        vh = v / (1 - beta2**t)
        x -= lr * mh / (np.sqrt(vh) + eps)
        hist.append(f(x))
    return x, np.array(hist)

def run_rmsprop(f, grad, x0, lr, rho, eps, n_iters):
    x = x0.copy(); v = np.zeros_like(x); hist = [f(x)]
    for _ in range(n_iters):
        g = grad(x)
        v = rho*v + (1-rho)*g**2
        x -= lr * g / (np.sqrt(v) + eps)
        hist.append(f(x))
    return x, np.array(hist)

# ── Run on Quadratic ──────────────────────────────────────────────────────
print("  OBJECTIVE 1: ILL-CONDITIONED QUADRATIC (κ = 100)")
print(f"  Global min at x* = {x_opt_q},  f(x*) = {f_quad(x_opt_q):.4f}")
print()
x0_q = np.array([0.0, 0.0])
n_iters_q = 300

res_gd_q  = run_gd(f_quad, grad_quad, x0_q, lr=0.018, n_iters=n_iters_q)
res_mom_q = run_momentum(f_quad, grad_quad, x0_q, lr=0.01, beta=0.9, n_iters=n_iters_q)
res_nes_q = run_nesterov(f_quad, grad_quad, x0_q, lr=0.01, beta=0.9, n_iters=n_iters_q)
res_rms_q = run_rmsprop(f_quad, grad_quad, x0_q, lr=0.1, rho=0.9, eps=1e-8, n_iters=n_iters_q)
res_adm_q = run_adam(f_quad, grad_quad, x0_q, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, n_iters=n_iters_q)

opt_methods_q = [
    ("GD",           res_gd_q),
    ("Momentum",     res_mom_q),
    ("Nesterov",     res_nes_q),
    ("RMSProp",      res_rms_q),
    ("Adam",         res_adm_q),
]

f_star_q = f_quad(x_opt_q)
print(f"  {'Method':>12} | {'f(final)-f*':>14} | {'Iters to 0.01':>14}")
print(f"  {'─'*46}")
for name, (x_final, hist) in opt_methods_q:
    gap   = hist[-1] - f_star_q
    diff  = hist - f_star_q
    iters_to = next((i for i, d in enumerate(diff) if d < 0.01), n_iters_q+1)
    print(f"  {name:>12} | {gap:14.6e} | {iters_to:>14}")

print()

# ── Run on Rosenbrock ─────────────────────────────────────────────────────
print("  OBJECTIVE 2: ROSENBROCK (NON-CONVEX),  min at (1,1),  f*=0")
print()
x0_r = np.array([-1.0, 1.0])
n_iters_r = 5000

res_gd_r  = run_gd(f_rosen, grad_rosen, x0_r, lr=1e-3, n_iters=n_iters_r)
res_mom_r = run_momentum(f_rosen, grad_rosen, x0_r, lr=1e-3, beta=0.9, n_iters=n_iters_r)
res_adm_r = run_adam(f_rosen, grad_rosen, x0_r, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, n_iters=n_iters_r)

opt_methods_r = [
    ("GD",       res_gd_r),
    ("Momentum", res_mom_r),
    ("Adam",     res_adm_r),
]

print(f"  {'Method':>12} | {'f(final)':>12} | {'Distance to (1,1)':>18}")
print(f"  {'─'*50}")
xopt_r = np.array([1.0, 1.0])
for name, (x_final, hist) in opt_methods_r:
    print(f"  {name:>12} | {hist[-1]:12.6f} | {np.linalg.norm(x_final - xopt_r):18.6f}")
print()

# ── Collect trajectories for Rosenbrock plot ──────────────────────────────
def run_gd_traj(f, grad, x0, lr, n_iters):
    x = x0.copy(); traj = [x.copy()]
    for _ in range(n_iters):
        x -= lr * grad(x)
        traj.append(x.copy())
    return np.array(traj)

def run_adam_traj(f, grad, x0, lr, beta1, beta2, eps, n_iters):
    x  = x0.copy(); m = np.zeros_like(x); v = np.zeros_like(x)
    traj = [x.copy()]
    for t in range(1, n_iters+1):
        g = grad(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        mh = m / (1 - beta1**t)
        vh = v / (1 - beta2**t)
        x -= lr * mh / (np.sqrt(vh) + eps)
        traj.append(x.copy())
    return np.array(traj)

n_traj = 500
traj_gd   = run_gd_traj(f_rosen, grad_rosen, x0_r, lr=1e-3, n_iters=n_traj)
traj_adam = run_adam_traj(f_rosen, grad_rosen, x0_r, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, n_iters=n_traj)

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Optimisation Algorithms: Convergence on Quadratic & Rosenbrock",
             fontsize=12, fontweight="bold")

# Plot 1: Convergence on quadratic
palette = ["steelblue", "tomato", "seagreen", "purple", "darkorange"]
for (name, (_, hist)), col in zip(opt_methods_q, palette):
    gap = np.maximum(hist - f_star_q, 1e-15)
    axes[0].semilogy(gap, lw=2, label=name, color=col)
axes[0].axhline(0.01, color="gray", linestyle=":", lw=1.2, label="0.01 threshold")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("f(xₜ) − f*")
axes[0].set_title(f"Quadratic (κ=100)\\nAdaptive methods overcome slow GD")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Plot 2: Convergence on Rosenbrock
for (name, (_, hist)), col in zip(opt_methods_r, palette):
    axes[1].semilogy(np.maximum(hist, 1e-10), lw=2, label=name, color=col)
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("f(xₜ)")
axes[1].set_title("Rosenbrock (Non-Convex)\\nNarrow curved valley traps GD")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

# Plot 3: Trajectory on Rosenbrock landscape
xr = np.linspace(-1.5, 1.5, 300)
yr = np.linspace(-0.5, 2.0, 300)
Xr, Yr = np.meshgrid(xr, yr)
Zr = (1 - Xr)**2 + 100*(Yr - Xr**2)**2
axes[2].contourf(Xr, Yr, np.log1p(Zr), 35, cmap="terrain", alpha=0.85)
axes[2].contour(Xr, Yr, np.log1p(Zr), 20, colors="k", linewidths=0.3, alpha=0.4)
axes[2].plot(traj_gd[:,0],   traj_gd[:,1],   "steelblue", lw=1.5,
             alpha=0.9, label=f"GD ({n_traj} iters)")
axes[2].plot(traj_adam[:,0], traj_adam[:,1], "tomato", lw=1.5,
             alpha=0.9, label=f"Adam ({n_traj} iters)")
axes[2].plot(1, 1, "k*", markersize=14, label="Global min (1,1)")
axes[2].plot(*x0_r, "ko", markersize=8, label="Start")
axes[2].set_xlabel("x₁"); axes[2].set_ylabel("x₂")
axes[2].set_title("Trajectory on Rosenbrock Landscape\\n(log-scale colours)")
axes[2].legend(fontsize=8); axes[2].set_aspect("equal")
axes[2].set_xlim(-1.5, 1.5); axes[2].set_ylim(-0.5, 2.0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "optimisers.png", dpi=120)
print("  Plot saved → optimisers.png")
print()
print("  KEY TAKEAWAYS:")
print("  GD on κ=100 quadratic: slow (O(κ) iters) due to zigzagging.")
print("  Momentum/Nesterov: O(√κ) convergence — damp oscillations.")
print("  Adam: per-parameter learning rates overcome curvature anisotropy.")
print("  Rosenbrock: curved banana valley traps gradient methods;")
print("              Adam navigates it faster via adaptive scaling.")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · Vector Calculus — Divergence, Curl, and Integral Theorems": {
        "description": (
            "Compute gradient, divergence, curl and Laplacian on analytic fields. "
            "Numerically verify Green's theorem and the Divergence theorem on a "
            "2D/3D domain.  Visualise field topology and the Laplacian on a "
            "graph (discrete analogue used in spectral GNNs)."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=6, suppress=True)

print("=" * 65)
print("  VECTOR CALCULUS — DIVERGENCE, CURL & INTEGRAL THEOREMS")
print("=" * 65)
print()

# ── PART 1: Analytic gradient, divergence, curl, Laplacian ───────────────
# Field F(x,y) = (sin(x)cos(y),  -cos(x)sin(y))  — known div = 0, curl analytic
print("  PART 1 — ANALYTIC VECTOR FIELD F = (sin x cos y, −cos x sin y)")
print()

def F(x, y):
    return np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y)

def div_F(x, y):
    # ∂F₁/∂x + ∂F₂/∂y = cos(x)cos(y) + cos(x)cos(y) ... wait:
    # F₁ = sin(x)cos(y) → ∂F₁/∂x = cos(x)cos(y)
    # F₂ = −cos(x)sin(y) → ∂F₂/∂y = −cos(x)cos(y)
    # div = cos(x)cos(y) − cos(x)cos(y) = 0  (divergence-free!)
    return np.cos(x)*np.cos(y) - np.cos(x)*np.cos(y)

def curl_F_z(x, y):
    # curl z-component = ∂F₂/∂x − ∂F₁/∂y
    # ∂F₂/∂x = sin(x)sin(y)
    # ∂F₁/∂y = −sin(x)sin(y)
    # curl = sin(x)sin(y) + sin(x)sin(y) = 2sin(x)sin(y)
    return 2*np.sin(x)*np.sin(y)

def laplacian_scalar(f, x, y, h=1e-5):
    return (f(x+h,y) + f(x-h,y) + f(x,y+h) + f(x,y-h) - 4*f(x,y)) / h**2

# Test at a specific point
px, py = np.pi/4, np.pi/3
F1, F2 = F(px, py)
div_val  = div_F(px, py)
curl_val = curl_F_z(px, py)
print(f"  At (π/4, π/3):")
print(f"  F           = ({F1:.6f}, {F2:.6f})")
print(f"  Divergence  = {div_val:.6f}  (analytic = 0.0 — divergence-free field)")
print(f"  Curl (z)    = {curl_val:.6f}  (analytic = 2·sin(π/4)·sin(π/3) = {2*np.sin(px)*np.sin(py):.6f})")
print()

# Numerical verification with finite differences
def div_numerical(x, y, h=1e-5):
    dF1_dx = (F(x+h,y)[0] - F(x-h,y)[0]) / (2*h)
    dF2_dy = (F(x,y+h)[1] - F(x,y-h)[1]) / (2*h)
    return dF1_dx + dF2_dy

def curl_numerical(x, y, h=1e-5):
    dF2_dx = (F(x+h,y)[1] - F(x-h,y)[1]) / (2*h)
    dF1_dy = (F(x,y+h)[0] - F(x,y-h)[0]) / (2*h)
    return dF2_dx - dF1_dy

div_num  = div_numerical(px, py)
curl_num = curl_numerical(px, py)
print(f"  Numerical divergence : {div_num:.2e}  (should be ≈ 0)")
print(f"  Numerical curl       : {curl_num:.6f}  (analytic: {curl_val:.6f})")
print(f"  Curl error           : {abs(curl_num - curl_val):.2e}")
print()

# ── PART 2: Green's Theorem numerical verification ────────────────────────
# ∮_C (P dx + Q dy) = ∬_D (∂Q/∂x − ∂P/∂y) dA
# Use P = −y, Q = x  →  ∂Q/∂x − ∂P/∂y = 1 + 1 = 2
# Rectangle [0,a]×[0,b]:  line integral = 2·a·b = area × 2
print("  PART 2 — GREEN'S THEOREM VERIFICATION")
print("  Field: P = −y, Q = x   →   curl = ∂Q/∂x − ∂P/∂y = 2")
print()

a, b = 2.0, 3.0
N = 10000  # points per side

def line_integral_rect(a, b, N):
    """∮_C P dx + Q dy around rectangle [0,a]×[0,b] CCW"""
    dt = 1.0 / N
    total = 0.0
    # Bottom: y=0, x: 0→a
    xs = np.linspace(0, a, N, endpoint=False)
    ys = np.zeros(N)
    total += np.sum(-ys * a * dt)         # P dx,  dx = a·dt
    # Right: x=a, y: 0→b
    xs2 = np.full(N, a)
    ys2 = np.linspace(0, b, N, endpoint=False)
    total += np.sum(xs2 * b * dt)         # Q dy,  dy = b·dt
    # Top: y=b, x: a→0
    xs3 = np.linspace(a, 0, N, endpoint=False)
    ys3 = np.full(N, b)
    total += np.sum(-ys3 * (-a) * dt)     # P dx,  dx = -a·dt
    # Left: x=0, y: b→0
    xs4 = np.zeros(N)
    ys4 = np.linspace(b, 0, N, endpoint=False)
    total += np.sum(xs4 * (-b) * dt)      # Q dy,  dy = -b·dt
    return total

line_val   = line_integral_rect(a, b, N)
double_val = 2.0 * a * b   # analytic: ∬ 2 dA = 2ab
print(f"  Rectangle [{a}]×[{b}]:")
print(f"  Line integral  ∮ (−y dx + x dy) = {line_val:.6f}")
print(f"  Double integral ∬ 2 dA           = {double_val:.6f}  (analytic)")
print(f"  Relative error                   = {abs(line_val - double_val)/double_val:.2e}")
print()

# ── PART 3: Divergence Theorem in 2D (Green's first identity) ────────────
# For F = (x², xy) on unit disk, ∬ div F dA = ∮ F·n ds
# div F = 2x + x = 3x
# On unit disk: ∬ 3x dA = 0  (by symmetry — integrand is odd)
print("  PART 3 — DIVERGENCE THEOREM (2D) VERIFICATION")
print("  Field: F = (x², xy)   div F = 2x + x = 3x")
print("  Domain: unit disk.  ∬ 3x dA = 0  (odd integrand, symmetric domain)")
print()

# Compute ∬ div F dA numerically on unit disk via Monte Carlo
rng = np.random.default_rng(42)
N_mc = 500_000
pts  = rng.uniform(-1, 1, (N_mc, 2))
inside = (pts[:,0]**2 + pts[:,1]**2) <= 1.0
div_mc = 3.0 * pts[inside, 0]
area   = np.pi  # unit disk
vol_integral = div_mc.mean() * area   # Monte Carlo estimate

# Compute ∮ F·n ds on unit circle (parametric)
N_line = 50000
theta  = np.linspace(0, 2*np.pi, N_line, endpoint=False)
dtheta = 2*np.pi / N_line
cx, cy = np.cos(theta), np.sin(theta)
Fx = cx**2
Fy = cx * cy
# Outward normal n = (cos θ, sin θ), ds = dθ
flux = np.sum((Fx*cx + Fy*cy) * dtheta)

print(f"  Volume integral ∬ div F dA  = {vol_integral:.6f}  (MC, N={N_mc:,}, expected ≈ 0)")
print(f"  Surface flux    ∮ F·n ds    = {flux:.6f}  (parametric, expected ≈ 0)")
print(f"  Absolute error              = {abs(vol_integral - flux):.2e}  (both ≈ 0 by symmetry)")
print()

# ── PART 4: Graph Laplacian (discrete vector calculus) ───────────────────
print("  PART 4 — GRAPH LAPLACIAN (DISCRETE ANALOGUE)")
print()

# Build a simple path graph: 0 - 1 - 2 - 3 - 4
n = 5
A = np.zeros((n, n))
for i in range(n-1):
    A[i, i+1] = A[i+1, i] = 1.0
D = np.diag(A.sum(axis=1))
L = D - A                         # combinatorial Laplacian
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
L_norm = D_inv_sqrt @ L @ D_inv_sqrt  # normalised Laplacian

vals, vecs = eigh(L)
print(f"  Path graph (5 nodes):  L =")
print(f"  {L}")
print()
print(f"  Eigenvalues of L: {vals.round(4)}")
print(f"  (λ₀ = 0 always — constant signal is harmonic)")
print(f"  Eigenvectors (columns) = graph Fourier basis:")
for i, (lam, vec) in enumerate(zip(vals, vecs.T)):
    print(f"    λ_{i} = {lam:.4f}  |  eigenvec = [{', '.join(f'{v:+.3f}' for v in vec)}]")
print()
print(f"  Laplacian regularisation penalty  f^T L f  for f = ones:")
f_sig  = np.ones(n)
pen    = f_sig @ L @ f_sig
print(f"  f = [1,1,1,1,1]  →  f^T L f = {pen:.4f}  (constant → zero penalty)")
f_sig2 = np.array([1,-1,1,-1,1], dtype=float)
pen2   = f_sig2 @ L @ f_sig2
print(f"  f = [1,-1,1,-1,1] →  f^T L f = {pen2:.4f}  (high-freq → large penalty)")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Vector Calculus: Field Topology, Green's Theorem & Graph Laplacian",
             fontsize=12, fontweight="bold")

# Plot 1: Vector field F with divergence=0 and curl overlay
xg, yg = np.meshgrid(np.linspace(-np.pi, np.pi, 30),
                     np.linspace(-np.pi, np.pi, 30))
F1g, F2g = F(xg, yg)
curl_g   = curl_F_z(xg, yg)
axes[0].contourf(xg, yg, curl_g, 25, cmap="RdBu_r", alpha=0.7)
axes[0].quiver(xg, yg, F1g, F2g, alpha=0.8, color="k", scale=25)
axes[0].set_title("F = (sin x cos y, -cos x sin y)\\nColour = curl(F), arrows = F\\n(div F = 0 everywhere)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_aspect("equal")

# Plot 2: Green's theorem — line integral path and integrand
xr = np.array([0, a, a, 0, 0])
yr = np.array([0, 0, b, b, 0])
axes[1].fill(xr, yr, alpha=0.15, color="steelblue", label=f"Domain [{a}]×[{b}]")
axes[1].plot(xr, yr, "steelblue", lw=2, label="Boundary C (CCW)")
xg2, yg2 = np.meshgrid(np.linspace(0, a, 20), np.linspace(0, b, 20))
axes[1].quiver(xg2, yg2, -yg2, xg2, alpha=0.5, color="tomato", scale=40)
axes[1].set_title(f"Green's Theorem: P=−y, Q=x\\n∮=∬2dA={double_val:.2f}  (verified)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
axes[1].legend(fontsize=9); axes[1].set_aspect("equal")

# Plot 3: Graph Laplacian eigenvectors (graph Fourier modes)
palette = ["steelblue", "tomato", "seagreen", "purple", "orange"]
for i, (lam, vec) in enumerate(zip(vals, vecs.T)):
    axes[2].plot(range(n), vec, "o-", lw=2, markersize=8,
                 color=palette[i], label=f"λ={lam:.3f}")
axes[2].axhline(0, color="k", lw=0.5, linestyle="--")
axes[2].set_xticks(range(n))
axes[2].set_xlabel("Node"); axes[2].set_ylabel("Eigenvector component")
axes[2].set_title("Graph Laplacian Eigenvectors\\n(5-node path — graph Fourier basis)")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vector_calculus.png", dpi=120)
print("  Plot saved → vector_calculus.png")
print()
print("  KEY TAKEAWAYS:")
print("  Divergence-free (div F = 0): no sources or sinks — streamlines are closed.")
print("  Curl ≠ 0: field has rotational structure — not conservative.")
print("  Green's theorem numerically verified to <1e-4 relative error.")
print("  Graph Laplacian eigenvectors = graph Fourier modes; λ₀=0 always.")
print("  Low-λ modes are smooth over the graph; high-λ modes oscillate rapidly.")
''',
    },

    # ── 7 ─────────────────────────────────────────────────────────────────────
    "7 · Neural ODEs & Continuous-Time Optimisation": {
        "description": (
            "Implement Euler and RK4 ODE solvers from scratch. "
            "Simulate the Neural ODE forward pass and adjoint-method backward pass. "
            "Compare discrete gradient descent against gradient flow ODE. "
            "Visualise the Nesterov damping ODE and Langevin dynamics trajectories."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(0)

print("=" * 65)
print("  NEURAL ODEs & CONTINUOUS-TIME OPTIMISATION")
print("=" * 65)
print()

# ── ODE Solvers ────────────────────────────────────────────────────────────
def euler(f, x0, t0, T, h):
    """Euler method: x(t+h) = x(t) + h·f(x(t),t)"""
    ts = [t0]; xs = [x0.copy()]
    x, t = x0.copy(), t0
    while t < T - 1e-10:
        h_eff = min(h, T - t)
        x = x + h_eff * f(x, t)
        t += h_eff
        ts.append(t); xs.append(x.copy())
    return np.array(ts), np.array(xs)

def rk4(f, x0, t0, T, h):
    """Runge-Kutta 4:  error O(h^5) per step vs O(h^2) for Euler"""
    ts = [t0]; xs = [x0.copy()]
    x, t = x0.copy(), t0
    while t < T - 1e-10:
        h_eff = min(h, T - t)
        k1 = f(x,          t)
        k2 = f(x + h_eff*k1/2, t + h_eff/2)
        k3 = f(x + h_eff*k2/2, t + h_eff/2)
        k4 = f(x + h_eff*k3,   t + h_eff)
        x = x + (h_eff/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += h_eff
        ts.append(t); xs.append(x.copy())
    return np.array(ts), np.array(xs)

# Test on x' = -2x  (exact: x(t) = x₀ e^{-2t})
print("  ODE SOLVER ACCURACY: x' = −2x,  x(0)=1,  exact: e^{-2t}")
print()
f_decay = lambda x, t: -2.0 * x
x0_d    = np.array([1.0])
T_d     = 2.0
exact_d = np.exp(-2.0 * T_d)
print(f"  {'h':>8} | {'Euler err':>12} | {'RK4 err':>12}")
print(f"  {'─'*40}")
for h in [0.5, 0.1, 0.05, 0.01]:
    _, xs_e = euler(f_decay, x0_d, 0.0, T_d, h)
    _, xs_r = rk4(f_decay, x0_d, 0.0, T_d, h)
    err_e = abs(xs_e[-1, 0] - exact_d)
    err_r = abs(xs_r[-1, 0] - exact_d)
    print(f"  {h:>8.3f} | {err_e:>12.2e} | {err_r:>12.2e}")
print()
print(f"  RK4 error ∝ h⁴ (4th-order), Euler error ∝ h (1st-order).")
print()

# ── Neural ODE forward + adjoint backward ─────────────────────────────────
# Simple 1D Neural ODE: dx/dt = θ·x  (linear, so exact solution exists)
# Loss: L = (x(T) - y*)²
# Adjoint: da/dt = -a·θ,  dL/dθ = -∫ a(t)·x(t) dt
print("  NEURAL ODE: dx/dt = θ·x,  x(0)=1,  T=1,  target y*=0.5")
print()

theta_true = -1.0    # true parameter (gives x(1)=e^{-1}≈0.368)
y_star     = 0.5
theta      = -0.5    # initial guess
T_ode      = 1.0
h_ode      = 0.01

# Forward pass
f_node = lambda x, t: np.array([theta]) * x
ts_f, xs_f = rk4(f_node, np.array([1.0]), 0.0, T_ode, h_ode)
xT = xs_f[-1, 0]
loss = (xT - y_star)**2

# Adjoint backward pass: da/dt = -a(t)·θ  (reversed time)
# dL/dx(T) = 2(x(T)-y*) initialises adjoint
a0  = np.array([2.0 * (xT - y_star)])
f_adj = lambda a, t: -np.array([theta]) * a   # adjoint ODE
ts_b, as_b = rk4(f_adj, a0, 0.0, T_ode, h_ode)
# dL/dθ = -∫_0^T a(T-t) · x(T-t) dt  (numerically integrate)
# We need a(t) and x(t) aligned in FORWARD time; flip adjoint trajectory
a_fwd = as_b[:, 0][::-1]          # a(t) in forward time
x_fwd = xs_f[:, 0]                # x(t) in forward time
n_pts = min(len(a_fwd), len(x_fwd))
dL_dtheta = -np.trapezoid(a_fwd[:n_pts] * x_fwd[:n_pts], ts_f[:n_pts])

# Analytic gradient for verification: x(T)=e^{θT}, dL/dθ = 2(e^{θT}-y*)·T·e^{θT}
dL_dtheta_analytic = 2*(np.exp(theta*T_ode) - y_star) * T_ode * np.exp(theta*T_ode)

print(f"  θ = {theta},  x(T) = {xT:.6f},  target = {y_star},  loss = {loss:.6f}")
print(f"  Adjoint dL/dθ  = {dL_dtheta:.6f}")
print(f"  Analytic dL/dθ = {dL_dtheta_analytic:.6f}")
print(f"  Error          = {abs(dL_dtheta - dL_dtheta_analytic):.2e}")
print()

# Gradient descent on θ to fit target
print("  Fitting θ via gradient descent (10 steps, lr=0.5):")
theta_fit = -0.5
lr_node = 0.5
for step in range(10):
    f_fit = lambda x, t: np.array([theta_fit]) * x
    _, xs_fit = rk4(f_fit, np.array([1.0]), 0.0, T_ode, h_ode)
    xT_fit = xs_fit[-1, 0]
    loss_fit = (xT_fit - y_star)**2
    # analytic grad
    grad_fit = 2*(xT_fit - y_star) * T_ode * xT_fit
    theta_fit -= lr_node * grad_fit
    if step % 2 == 0 or step == 9:
        print(f"  step {step+1:2d}: θ={theta_fit:.6f}, x(T)={xT_fit:.6f}, loss={loss_fit:.6e}")
print()

# ── Gradient flow vs discrete GD ──────────────────────────────────────────
# f(x) = x² + 2x + 1 = (x+1)²,  minimum at x* = -1
print("  GRADIENT FLOW vs DISCRETE GRADIENT DESCENT")
print("  f(x) = (x+1)²,  ∇f = 2(x+1),  min at x*=-1")
print()

f_opt  = lambda x: (x + 1)**2
gf_opt = lambda x: 2*(x + 1)

# Continuous gradient flow: dx/dt = -∇f(x)
f_gflow = lambda x, t: -gf_opt(x)
ts_gf, xs_gf = rk4(f_gflow, np.array([3.0]), 0.0, 5.0, 0.01)

# Discrete GD with various step sizes
x0_gd = 3.0
etas   = [0.1, 0.5, 0.9, 1.1]
n_gd   = 50
hist_gd = {}
for eta in etas:
    x = x0_gd
    h = [x]
    for _ in range(n_gd):
        x = x - eta * gf_opt(x)
        h.append(x)
    hist_gd[eta] = np.array(h)

print(f"  {'η':>6} | {'Final x':>10} | {'Final f(x)':>12} | Status")
print(f"  {'─'*50}")
for eta in etas:
    xf = hist_gd[eta][-1]
    ff = f_opt(xf)
    status = "converged" if abs(xf - (-1)) < 0.01 else ("diverged" if abs(xf) > 100 else "oscillating")
    print(f"  {eta:>6.2f} | {xf:>10.6f} | {ff:>12.6e} | {status}")
print()

# ── Nesterov ODE: ẍ + (3/t)ẋ = -∇f(x)  (converted to 1st-order system) ───
print("  NESTEROV ODE: ẍ + (3/t)ẋ = −∇f(x)  (continuous Nesterov AGD)")
print()

# State: (x, v) where v = ẋ
# dx/dt = v
# dv/dt = -∇f(x) - (3/t)v    (t > 0)
def f_nesterov(state, t):
    x, v = state
    damp = 3.0 / max(t, 0.01)   # avoid division by zero
    dxdt = v
    dvdt = -gf_opt(x) - damp * v
    return np.array([dxdt, dvdt])

ts_nes, xs_nes = rk4(f_nesterov, np.array([3.0, 0.0]), 0.01, 6.0, 0.01)
print(f"  Nesterov ODE final x: {xs_nes[-1,0]:.6f}  (target: -1.0)")
print(f"  f(x_final): {f_opt(xs_nes[-1,0]):.2e}")
print()

# ── Langevin dynamics ──────────────────────────────────────────────────────
print("  LANGEVIN DYNAMICS: dx = -∇f(x) dt + √(2T) dW")
print()
temps = [0.0, 0.1, 0.5]
n_lang = 5000
h_lang = 0.01
lang_trajs = {}
for temp in temps:
    x = np.array([3.0])
    traj = [x[0]]
    for _ in range(n_lang):
        noise = np.sqrt(2*temp*h_lang) * rng.standard_normal(1)
        x = x - h_lang * gf_opt(x) + noise
        traj.append(x[0])
    lang_trajs[temp] = np.array(traj)
    print(f"  T={temp:.1f}: mean x = {lang_trajs[temp][2000:].mean():.4f}, "
          f"std = {lang_trajs[temp][2000:].std():.4f}   "
          f"(theory: mean=-1, std=√T={np.sqrt(temp):.4f})")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Neural ODEs & Continuous-Time Optimisation", fontsize=12, fontweight="bold")

# Plot 1: ODE solver accuracy (Euler vs RK4)
hs = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
err_euler, err_rk4 = [], []
for h in hs:
    _, xe = euler(f_decay, np.array([1.0]), 0.0, T_d, h)
    _, xr = rk4(f_decay, np.array([1.0]), 0.0, T_d, h)
    err_euler.append(abs(xe[-1,0] - exact_d))
    err_rk4.append(abs(xr[-1,0] - exact_d))
axes[0].loglog(hs, err_euler, "o-", color="steelblue", lw=2, label="Euler O(h)")
axes[0].loglog(hs, err_rk4,   "s-", color="tomato",   lw=2, label="RK4 O(h⁴)")
axes[0].loglog(hs, [h**1 * err_euler[0]/hs[0]**1 for h in hs], "k--", lw=1, alpha=0.4)
axes[0].loglog(hs, [h**4 * err_rk4[0]/hs[0]**4 for h in hs],   "k:",  lw=1, alpha=0.4)
axes[0].set_xlabel("Step size h"); axes[0].set_ylabel("Error at T=2")
axes[0].set_title("ODE Solver Accuracy\\nEuler vs RK4")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Plot 2: Gradient flow vs discrete GD
axes[1].plot(ts_gf, xs_gf[:,0], "k-", lw=2.5, label="Gradient flow (ODE)", zorder=5)
colors_gd = ["steelblue", "seagreen", "orange", "tomato"]
for (eta, hist), col in zip(hist_gd.items(), colors_gd):
    axes[1].plot(np.arange(len(hist))*eta, hist, "o-", markersize=3,
                 lw=1.5, color=col, alpha=0.8, label=f"GD η={eta}")
axes[1].axhline(-1, color="gray", linestyle=":", lw=1.2, label="x*=−1")
axes[1].set_xlabel("Time / η·steps"); axes[1].set_ylabel("x(t)")
axes[1].set_title("Gradient Flow vs Discrete GD\\nη=1.1 diverges (step > 1/L=0.5)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
axes[1].set_ylim(-3, 4)

# Plot 3: Nesterov ODE trajectory
axes[2].plot(ts_nes, xs_nes[:,0], color="purple", lw=2, label="Position x(t)")
axes[2].plot(ts_nes, xs_nes[:,1], color="coral",  lw=1.5, alpha=0.7, label="Velocity ẋ(t)")
axes[2].axhline(-1, color="gray", linestyle=":", lw=1.2, label="x*=−1")
axes[2].set_xlabel("t"); axes[2].set_ylabel("x, ẋ")
axes[2].set_title("Nesterov ODE: ẍ + (3/t)ẋ = −∇f\\nDamping → 0 enables acceleration")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

# Plot 4: Langevin dynamics
colors_lang = ["steelblue", "seagreen", "tomato"]
for (temp, traj), col in zip(lang_trajs.items(), colors_lang):
    axes[3].plot(traj[:500], color=col, lw=1, alpha=0.8, label=f"T={temp}")
axes[3].axhline(-1, color="k", linestyle="--", lw=1.2, label="x*=−1")
axes[3].set_xlabel("Step"); axes[3].set_ylabel("x")
axes[3].set_title("Langevin Dynamics\\nHigher T → wider stationary distribution")
axes[3].legend(fontsize=9); axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "neural_odes_continuous_optim.png", dpi=120)
print("  Plot saved → neural_odes_continuous_optim.png")
print()
print("  KEY TAKEAWAYS:")
print("  RK4 is 4th-order accurate — error drops 10000× when h halves.")
print("  Euler step = ResNet residual block: one forward-Euler ODE step.")
print("  Adjoint method gives exact gradients through ODE solver in O(1) memory.")
print("  Gradient flow is the continuous limit of GD; η > 2/L causes divergence.")
print("  Nesterov ODE: vanishing damping (3/t) is why acceleration works.")
print("  Langevin: T controls exploration vs exploitation tradeoff.")
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

# -------------------------------------------------------------------------
# Standalone runner
# Run:  python 02_Calculus_&_Optimization.py
# Generates all plots and saves them to Resultant_Graphs/
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    _run_globals = {"__file__": __file__}
    for _op_name, _op in OPERATIONS.items():
        sep = "=" * 65
        print(f"\n{sep}\n  Running: {_op_name}\n{sep}")
        try:
            exec(_op["code"], _run_globals)
        except Exception as _e:
            print(f"  ERROR in {_op_name}: {_e}")
            traceback.print_exc()
    _out = _run_globals.get("OUTPUT_DIR", "(OUTPUT_DIR not set)")
    print(f"\n  All done. Plots saved to: {_out}")