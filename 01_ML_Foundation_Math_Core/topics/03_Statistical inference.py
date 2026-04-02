"""
Statistical Inference
=====================

The science of learning from data under uncertainty. Every time a model
is fitted, a hypothesis is tested, or a confidence interval is reported,
the machinery of statistical inference is at work. Bayesian and
frequentist perspectives offer complementary views of the same deep
question: what do observations tell us about the world?

This module builds the complete theoretical foundation — from probability
spaces and distributions through estimation, hypothesis testing,
Bayesian inference, and the information-theoretic limits of learning.

"""

import textwrap
import re

TOPIC_NAME   = "Statistical Inference"
DISPLAY_NAME = "03 · Statistical Inference"
ICON         = "𝜎"
SUBTITLE     = "Estimation, Testing, Bayes — The Science of Learning from Data"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — PROBABILITY FOUNDATIONS

### Probability Spaces

A PROBABILITY SPACE is the triple (Ω, ℱ, ℙ):

    Ω  — the SAMPLE SPACE: the set of all possible outcomes
    ℱ  — the σ-ALGEBRA: a collection of subsets of Ω (the "events")
    ℙ  — the PROBABILITY MEASURE: a function ℱ → [0, 1]

The σ-algebra ℱ must satisfy:
    1. Ω ∈ ℱ
    2. If A ∈ ℱ then Aᶜ ∈ ℱ  (closed under complement)
    3. If A₁, A₂, ... ∈ ℱ then ⋃ᵢ Aᵢ ∈ ℱ  (closed under countable union)

The probability measure ℙ satisfies the KOLMOGOROV AXIOMS:
    1. ℙ(A) ≥ 0  for all A ∈ ℱ                  (non-negativity)
    2. ℙ(Ω) = 1                                   (normalisation)
    3. For disjoint A₁, A₂, ...: ℙ(⋃ᵢ Aᵢ) = Σᵢ ℙ(Aᵢ)  (countable additivity)

Why σ-algebras? Not every subset of a continuous space can be assigned a
consistent probability (Banach-Tarski paradox). The σ-algebra restricts
attention to MEASURABLE sets — those for which probability is well-defined.

    Diagram 1 — Anatomy of a Probability Space:

    Ω = [0,1]       (continuous sample space)
    ℱ = Borel σ-algebra on [0,1]  (all open/closed intervals and their
                                    countable unions/intersections)
    ℙ = Lebesgue measure           (ℙ([a,b]) = b − a, i.e., length)

All of statistics lives inside this structure. "Random" means the outcome
is an element of Ω that we do not yet know.


### Conditional Probability & Independence

The CONDITIONAL PROBABILITY of A given B (ℙ(B) > 0):

    ℙ(A | B) = ℙ(A ∩ B) / ℙ(B)

Geometrically: restrict the sample space to B and renormalise.

    ┌─────────────────────────────────────────────────────────┐
    │  ℙ(A | B) is a NEW probability measure on B.            │
    │  All probability axioms hold with Ω replaced by B.      │
    └─────────────────────────────────────────────────────────┘

BAYES' THEOREM — the engine of Bayesian inference:

    ℙ(A | B) = ℙ(B | A) · ℙ(A) / ℙ(B)

    More generally:  ℙ(A | B) = ℙ(B | A)ℙ(A) / [Σⱼ ℙ(B|Aⱼ)ℙ(Aⱼ)]

    (denominator is the TOTAL PROBABILITY / MARGINALISATION)

Two events A, B are INDEPENDENT if:
    ℙ(A ∩ B) = ℙ(A)·ℙ(B)    ⟺    ℙ(A|B) = ℙ(A)

Independence says B carries no information about A — knowing B occurs
does not update the probability of A.

CONDITIONAL INDEPENDENCE: A ⊥⊥ B | C  means:
    ℙ(A ∩ B | C) = ℙ(A|C)·ℙ(B|C)

This is the foundation of graphical models: nodes are conditionally
independent given their Markov blanket.

Pairwise independence does NOT imply mutual independence.
Example: X, Y ~ Bernoulli(½) independent; Z = X ⊕ Y.
  X,Y are independent; X,Z are independent; Y,Z are independent.
  But {X,Y,Z} are not mutually independent: ℙ(X=Y=Z=1) = 0 ≠ ⅛.


### PART 2 — RANDOM VARIABLES & DISTRIBUTIONS

### Random Variables

A RANDOM VARIABLE X is a measurable function X: Ω → ℝ.

It maps outcomes to real numbers. "X = 3" means {ω ∈ Ω : X(ω) = 3}.

For X to be measurable: {ω: X(ω) ≤ x} ∈ ℱ for all x ∈ ℝ.
This ensures ℙ(X ≤ x) is well-defined for every x.

The CUMULATIVE DISTRIBUTION FUNCTION (CDF):

    F(x) = ℙ(X ≤ x)

Properties of any CDF:
    1. Non-decreasing:  x₁ < x₂ → F(x₁) ≤ F(x₂)
    2. Right-continuous: lim_{t↓x} F(t) = F(x)
    3. Limits:          F(−∞) = 0,  F(+∞) = 1

Every valid CDF characterises a unique distribution on ℝ.

Discrete X: PROBABILITY MASS FUNCTION (PMF)  p(x) = ℙ(X = x)
Continuous X: PROBABILITY DENSITY FUNCTION (PDF) f(x) with F(x) = ∫_{-∞}^x f(t)dt

    For continuous X: ℙ(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx
    For discrete X:   ℙ(a ≤ X ≤ b) = Σ_{x=a}^b p(x)

    Diagram 2 — CDF, PDF, and PMF Relationship:

     PDF / PMF ↑          CDF ↑
     (density) │           │          ╭───────
               │  ╭─╮      │      ╭──╯
               │ ╱   ╲     │    ╭╯
               │╱     ╲    │  ╭╯
     ───────────────────→  └───────────────────→
                x                    x
     Area under PDF = 1   CDF = running integral of PDF


### Moments — Summarising a Distribution

The kth MOMENT of X:   𝔼[Xᵏ] = ∫ xᵏ f(x)dx  (continuous)

The kth CENTRAL MOMENT:   𝔼[(X − μ)ᵏ]   where μ = 𝔼[X]

Key moments:
    𝔼[X]           — MEAN (location)
    Var(X) = 𝔼[(X−μ)²]  — VARIANCE (spread)   = 𝔼[X²] − (𝔼[X])²
    SD(X) = √Var(X)    — STANDARD DEVIATION
    𝔼[(X−μ)³]/σ³   — SKEWNESS (asymmetry)
    𝔼[(X−μ)⁴]/σ⁴   — KURTOSIS (tail weight; Gaussian has kurtosis 3)

    ┌──────────────────────────────────────────────────────────┐
    │  Var(aX + b) = a²Var(X)                                  │
    │  Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)                │
    │  Cov(X,Y) = 𝔼[XY] − 𝔼[X]𝔼[Y]                             │
    │  ρ(X,Y) = Cov(X,Y) / [SD(X)·SD(Y)] ∈ [−1, 1]             │
    └──────────────────────────────────────────────────────────┘

If X ⊥⊥ Y: Cov(X,Y) = 0.  Converse is FALSE in general.

MOMENT GENERATING FUNCTION (MGF):   M_X(t) = 𝔼[eᵗˣ]

The MGF uniquely determines the distribution (when it exists).
The kth moment = M_X^(k)(0)  (kth derivative at t=0).

CHARACTERISTIC FUNCTION:   φ_X(t) = 𝔼[eⁱᵗˣ]  (always exists)

The characteristic function is the Fourier transform of the density.
Independence ⟺ φ_{X+Y}(t) = φ_X(t)·φ_Y(t).


### Key Distributions and Their Roles

DISCRETE:
    Bernoulli(p):    P(X=1)=p, P(X=0)=1-p.  Mean=p, Var=p(1-p).
    Binomial(n,p):   X = #{successes in n Bernoulli trials}.
                     PMF: C(n,k)pᵏ(1-p)^{n-k}. Mean=np, Var=np(1-p).
    Poisson(λ):      X = #{events in fixed interval}. PMF: e^{-λ}λᵏ/k!
                     Mean=Var=λ. Limit of Binomial as n→∞, np→λ.
    Geometric(p):    X = #{trials until first success}. Mean=1/p.
    Negative Binomial(r,p): #{trials until rth success}. Sum of r Geometrics.

CONTINUOUS:
    Uniform(a,b):   f(x) = 1/(b-a) on [a,b].  Mean=(a+b)/2, Var=(b-a)²/12.
    Normal(μ,σ²):   f(x) = (1/σ√(2π)) exp(−(x−μ)²/2σ²)
                    The CENTRAL distribution of statistics via CLT.
    Exponential(λ): f(x) = λe^{-λx} x≥0.  Mean=1/λ, Var=1/λ².
                    MEMORYLESS: ℙ(X>s+t|X>s) = ℙ(X>t).
    Gamma(α,β):     Generalises Exponential. Sum of α Exponentials.
    Beta(α,β):      Supported on [0,1]. Natural prior for probabilities.
    Chi-squared(k): Sum of k squared standard Normals. Χ²(k) = Gamma(k/2,2).
    t(ν):           Normal/√(Χ²(ν)/ν). Heavy tails; limit is Normal as ν→∞.
    F(d₁,d₂):      Ratio of two Chi-squared variables. Used in ANOVA.

    Diagram 3 — Relationships Between Distributions:

    Bernoulli(p) ─[sum]→ Binomial(n,p) ─[n→∞,np→λ]→ Poisson(λ)
                                        ─[CLT]───────→ Normal(np, np(1-p))
    Normal(0,1)² ─[sum of k]→ Χ²(k) ─[ratio]→ F(k,m)
    Normal(0,1)/√(Χ²(ν)/ν) → t(ν) ─[ν→∞]→ Normal(0,1)
    Gamma(1,β) = Exponential(1/β)


### PART 3 — THE EXPONENTIAL FAMILY & SUFFICIENT STATISTICS

### The Exponential Family

Most distributions used in statistics belong to the EXPONENTIAL FAMILY:

    p(x | η) = h(x) · exp(ηᵀT(x) − A(η))

    η   — NATURAL PARAMETER (canonical parameterisation)
    T(x) — SUFFICIENT STATISTIC (see below)
    A(η) — LOG-PARTITION FUNCTION (log-normaliser)
    h(x) — BASE MEASURE

A(η) = log ∫ h(x) exp(ηᵀT(x)) dx    (ensures the density integrates to 1)

The family unifies Normal, Bernoulli, Poisson, Exponential, Gamma, Beta,
Dirichlet, and many others under a single mathematical framework.

    Examples:
    Normal(μ,σ²):  η=(μ/σ²,−1/2σ²),  T(x)=(x,x²),  A(η)=−η₁²/4η₂ + ½log(−π/η₂)
    Bernoulli(p):  η=log(p/(1-p)),    T(x)=x,        A(η)=log(1+eᶮ)
    Poisson(λ):    η=log λ,           T(x)=x,        A(η)=eᶮ

The LOG-PARTITION FUNCTION A(η) is CONVEX and generates moments:

    ∇A(η)  = 𝔼[T(X)]         (mean of sufficient statistic)
    ∇²A(η) = Cov(T(X))        (covariance = second derivative of A)

This makes A(η) the generating function for all cumulants of T(X).

    ┌──────────────────────────────────────────────────────────┐
    │  Why the exponential family matters for ML:              │
    │  · GLMs, Naive Bayes, variational inference, HMMs        │
    │  · Conjugate priors always exist                         │
    │  · MLE reduces to moment matching: 𝔼_θ[T(X)] = T̄(xₙ)      │
    │  · Natural gradients are tractable (Fisher = ∇²A)        │
    └──────────────────────────────────────────────────────────┘


### Sufficient Statistics

A statistic T(X₁,...,Xₙ) is SUFFICIENT for parameter θ if:

    The conditional distribution of X₁,...,Xₙ given T(X) does not
    depend on θ — equivalently, T captures ALL information about θ.

FISHER-NEYMAN FACTORISATION THEOREM:
T is sufficient for θ iff the joint density factorises as:

    p(x₁,...,xₙ | θ) = g(T(x₁,...,xₙ), θ) · h(x₁,...,xₙ)

For the exponential family, T(x) = Σᵢ T(xᵢ) is always sufficient.
This is why SAMPLE MEANS and SAMPLE TOTALS are sufficient for most
common distributions — you don't need the raw data, just the summary.

RAO-BLACKWELL THEOREM: If T̂ is an estimator of θ and T is a sufficient
statistic, then the RAO-BLACKWELLISED estimator:

    T̂* = 𝔼[T̂ | T]

has smaller or equal MSE: Var(T̂*) ≤ Var(T̂).

Conditioning on the sufficient statistic never loses information and
often reduces variance — the optimal estimator is always a function
of the sufficient statistic.


### PART 4 — ESTIMATION THEORY

### Properties of Estimators

Given data X₁,...,Xₙ ~ p(x|θ), an ESTIMATOR θ̂ₙ = T(X₁,...,Xₙ) is any
function of the data used to approximate θ.

KEY PROPERTIES:

BIAS:
    Bias(θ̂) = 𝔼_θ[θ̂] − θ
    UNBIASED: 𝔼_θ[θ̂] = θ for all θ.
    Asymptotically unbiased: Bias(θ̂ₙ) → 0 as n → ∞.

VARIANCE: Var_θ(θ̂) — spread of the estimator around its mean.

MEAN SQUARED ERROR:
    MSE(θ̂) = 𝔼_θ[(θ̂ − θ)²] = Bias(θ̂)² + Var(θ̂)

    The BIAS-VARIANCE DECOMPOSITION: MSE = Bias² + Variance.
    Reducing bias often increases variance (and vice versa).
    This is the statistical manifestation of the bias-variance tradeoff.

CONSISTENCY:
    θ̂ₙ is CONSISTENT if θ̂ₙ →ᵖ θ as n→∞.
    i.e.,  ∀ε > 0:  lim_{n→∞} ℙ(|θ̂ₙ − θ| > ε) = 0

    Sufficient condition: Bias(θ̂ₙ) → 0 and Var(θ̂ₙ) → 0.


### Maximum Likelihood Estimation

The LIKELIHOOD of θ given observations x₁,...,xₙ:

    L(θ) = p(x₁,...,xₙ | θ) = Πᵢ p(xᵢ | θ)  (iid case)

The LOG-LIKELIHOOD:   ℓ(θ) = log L(θ) = Σᵢ log p(xᵢ | θ)

The MLE is:   θ̂_MLE = argmax_θ ℓ(θ)

Solving the MLE:   ∂ℓ/∂θ = 0  (SCORE EQUATION — set gradient to zero)

    Example — MLE for Normal(μ, σ²):
        ℓ(μ,σ²) = −(n/2)log(2πσ²) − 1/(2σ²) Σ(xᵢ−μ)²
        ∂ℓ/∂μ = 0  →  μ̂ = (1/n)Σxᵢ  = x̄   (sample mean, unbiased)
        ∂ℓ/∂σ² = 0  →  σ̂² = (1/n)Σ(xᵢ−x̄)²  (biased by factor (n−1)/n)

    Example — MLE for Bernoulli(p):
        ℓ(p) = (Σxᵢ)log p + (n−Σxᵢ)log(1−p)
        ∂ℓ/∂p = 0  →  p̂ = Σxᵢ/n = x̄   (sample proportion)

PROPERTIES OF MLE (under regularity conditions):
    1. CONSISTENCY:  θ̂_MLE →ᵖ θ*
    2. ASYMPTOTIC NORMALITY:  √n(θ̂_MLE − θ*) →ᵈ N(0, I(θ*)⁻¹)
    3. ASYMPTOTIC EFFICIENCY:  achieves the Cramér-Rao lower bound
    4. INVARIANCE:  g(θ̂_MLE) = θ̂_MLE for g(θ)  for any function g

The MLE is the "best" estimator asymptotically — no consistent estimator
can have smaller asymptotic variance.


### The Cramér-Rao Lower Bound

The FISHER INFORMATION:

    I(θ) = 𝔼_θ[(∂ log p(X|θ)/∂θ)²]  =  −𝔼_θ[∂² log p(X|θ)/∂θ²]

The CRAMÉR-RAO LOWER BOUND: For any unbiased estimator θ̂:

    Var(θ̂) ≥ 1 / (n · I(θ))

No unbiased estimator can have variance smaller than 1/(n·I(θ)).
This is the FUNDAMENTAL LIMIT of estimation: the Fisher information
determines how fast we can learn θ from data.

    Intuition: I(θ) measures how "peaked" the likelihood is near θ.
    High I(θ) → sharply peaked → easy to estimate → low variance.
    Low I(θ)  → flat likelihood → hard to estimate → high variance.

An estimator achieving the CRB is called EFFICIENT.
The MLE is asymptotically efficient (achieves 1/(n·I(θ))).

    ┌──────────────────────────────────────────────────────────┐
    │  CRLB in multiple dimensions (θ ∈ ℝᵈ):                   │
    │  I(θ) is now a d×d matrix — the FISHER INFORMATION       │
    │  MATRIX:  Iᵢⱼ(θ) = 𝔼[∂ℓ/∂θᵢ · ∂ℓ/∂θⱼ]                     │
    │  Cov(θ̂) ⪰ I(θ)⁻¹/n    (matrix inequality)                │
    │  The natural gradient uses I(θ)⁻¹ as a preconditioner.   │
    └──────────────────────────────────────────────────────────┘


### Method of Moments

The kth SAMPLE MOMENT: m̂ₖ = (1/n)Σᵢ Xᵢᵏ

The kth POPULATION MOMENT: μₖ(θ) = 𝔼_θ[Xᵏ]

The METHOD OF MOMENTS estimator: set m̂ₖ = μₖ(θ) and solve for θ.
Use as many equations as there are parameters to identify.

MOM is simpler than MLE but generally less efficient (higher variance).
It is useful when the likelihood is intractable or has no closed form.


### PART 5 — SAMPLING DISTRIBUTIONS & THE CENTRAL LIMIT THEOREM

### Laws of Large Numbers

Let X₁,...,Xₙ be iid with mean μ and variance σ².

WEAK LAW OF LARGE NUMBERS (WLLN):
    X̄ₙ →ᵖ μ   (convergence in probability)
    ∀ε>0: ℙ(|X̄ₙ − μ| > ε) → 0

    Proof (Chebyshev): ℙ(|X̄ₙ − μ| > ε) ≤ Var(X̄ₙ)/ε² = σ²/(nε²) → 0

STRONG LAW OF LARGE NUMBERS (SLLN):
    X̄ₙ →ᵃ·ˢ· μ   (convergence almost surely)
    ℙ(lim_{n→∞} X̄ₙ = μ) = 1

    The SLLN makes the frequentist definition of probability rigorous:
    run an experiment infinitely many times; the sample frequency of any
    event converges to its true probability.

    ┌───────────────────────────────────────────────────────────┐
    │  WLLN: any individual long run is approximately μ.        │
    │  SLLN: every sufficiently long run will be near μ.        │
    │  SLLN is strictly stronger: almost sure ⟹ in probability │
    └───────────────────────────────────────────────────────────┘


### The Central Limit Theorem

CLASSICAL CLT: Let X₁,...,Xₙ be iid with mean μ and variance σ² < ∞.

    √n(X̄ₙ − μ) / σ →ᵈ N(0, 1)    as n → ∞

Equivalently:   X̄ₙ ≈ N(μ, σ²/n)  for large n.

The CLT is DISTRIBUTION-FREE — it applies regardless of the shape of
the individual Xᵢ distributions, as long as σ² < ∞.

    Diagram 4 — CLT in Action:

    Xᵢ ~ Uniform(0,1):        n=1: uniform, n=4: peaked, n=30: ≈ Gaussian
    Xᵢ ~ Exponential(1):      n=1: skewed,  n=10: less so, n=50: ≈ Gaussian
    Xᵢ ~ Bernoulli(0.1):      n=1: spiky,   n=100: ≈ Gaussian (rule: npq≥5)

    [─────────────────────────────────────────────────────]
    [  n=1       n=4        n=10        n=30              ]
    [  ████     ▄████▄     ▂▄████▄▂   ▁▂▄▇███▇▄▂▁         ]
    [  ████     ▄████▄     ▂▄████▄▂   ▁▂▄▇███▇▄▂▁         ]
    [─────────────────────────────────────────────────────]
    Distributions of X̄ₙ converge to a bell curve.

Why does the CLT hold? The CHARACTERISTIC FUNCTION of X̄ₙ:

    φ_{X̄ₙ}(t) = [φ_X(t/√n)]ⁿ

    Taylor expanding φ_X around t=0:
        φ_X(t) = 1 − σ²t²/2 + O(t³)

    So: [1 − σ²t²/2n + O(t³/n^{3/2})]ⁿ → e^{−σ²t²/2}

    e^{−σ²t²/2} is the characteristic function of N(0,σ²). ∎

MULTIVARIATE CLT:
    √n(X̄ₙ − μ) →ᵈ N(0, Σ)
    where Σ = Cov(X) is the covariance matrix of a single observation.


### Delta Method

If √n(T̂ₙ − θ) →ᵈ N(0, σ²) and g is differentiable at θ:

    √n(g(T̂ₙ) − g(θ)) →ᵈ N(0, [g'(θ)]² σ²)

The DELTA METHOD propagates asymptotic normality through smooth
transformations. It is the first-order Taylor approximation applied
to the asymptotic distribution.

Example: If X̄ₙ ≈ N(μ, σ²/n), then log(X̄ₙ) ≈ N(log μ, σ²/(nμ²)).

Multivariate: Var(g(T̂ₙ)) ≈ (∇g)ᵀ Σ ∇g / n   (propagation of uncertainty)


### Key Sampling Distributions

Let X₁,...,Xₙ ~ iid N(μ, σ²):

    X̄ₙ ~ N(μ, σ²/n)                               (exact)
    (n−1)S²/σ² ~ Χ²(n−1)    S² = Σ(Xᵢ−X̄)²/(n−1) (exact)
    X̄ₙ ⊥⊥ S²                                       (independence!)
    (X̄ₙ − μ)/(S/√n) ~ t(n−1)                      (exact — use when σ² unknown)

The t distribution has heavier tails than Normal — it accounts for
the extra uncertainty from estimating σ. As n→∞, t(n−1) → N(0,1).

    ┌──────────────────────────────────────────────────────────┐
    │  F distribution: if U~Χ²(d₁), V~Χ²(d₂) independent:      │
    │  F = (U/d₁)/(V/d₂) ~ F(d₁,d₂)                            │
    │  Ratio of two sample variances from Normal populations   │
    │  follows an F distribution. Used in ANOVA, regression.   │
    └──────────────────────────────────────────────────────────┘


### PART 6 — HYPOTHESIS TESTING

### The Hypothesis Testing Framework

A HYPOTHESIS TEST decides between two competing hypotheses:

    H₀: null hypothesis  (default, assumed true unless data contradict it)
    H₁: alternative hypothesis  (what we want to detect)

A TEST is a decision rule δ(X): reject H₀ if the test statistic T(X)
falls in the REJECTION REGION R, else fail to reject.

Two types of ERRORS:

    TYPE I ERROR (false positive):  Reject H₀ when H₀ is true.
    TYPE II ERROR (false negative): Fail to reject H₀ when H₁ is true.

    ┌─────────────────┬────────────────┬────────────────┐
    │                 │  H₀ True       │  H₁ True       │
    ├─────────────────┼────────────────┼────────────────┤
    │  Reject H₀      │  Type I  (α)   │  Correct (1-β) │
    │  Fail to Reject │  Correct (1-α) │  Type II (β)   │
    └─────────────────┴────────────────┴────────────────┘

    SIGNIFICANCE LEVEL:  α = ℙ(reject H₀ | H₀ true)   (Type I rate — controlled)
    POWER:               1−β = ℙ(reject H₀ | H₁ true)  (true positive rate — maximise)

    The fundamental tension: reducing α generally increases β (and vice versa).


### Neyman-Pearson Lemma — Optimal Tests

For a SIMPLE vs SIMPLE test (H₀: θ = θ₀ vs H₁: θ = θ₁):

    NEYMAN-PEARSON LEMMA: The LIKELIHOOD RATIO TEST is the MOST POWERFUL
    test at level α:

        Reject H₀ if  L(θ₁) / L(θ₀) > cα

    where cα is chosen so that ℙ(reject H₀ | H₀) = α.

The likelihood ratio is the SUFFICIENT STATISTIC for the simple vs simple
problem. No other test at the same level can achieve higher power.

For COMPOSITE alternatives (θ ∈ Θ₁), we use:
    UMP (Uniformly Most Powerful) tests — most powerful against all θ∈Θ₁.
    These exist for one-sided tests in exponential families.
    For two-sided tests, UMP tests generally do not exist.


### The p-value — A Precise Definition

The p-VALUE is:
    p = ℙ(T(X) ≥ t_obs | H₀)

The probability, under H₀, of observing a test statistic AT LEAST AS
EXTREME as the one we observed.

    ┌──────────────────────────────────────────────────────────┐
    │  The p-value is NOT:                                     │
    │  · ℙ(H₀ is true)                                         │
    │  · The probability of a false positive                   │
    │  · The probability of the result by chance               │
    │                                                          │
    │  The p-value IS:                                         │
    │  · A random variable (uniform under H₀ if T continuous)  │
    │  · A summary of "how surprising" the data is under H₀    │
    │  · Valid ONLY relative to the pre-specified test         │
    └──────────────────────────────────────────────────────────┘

Under H₀:  p-value ~ Uniform(0,1)  (if T is a continuous test statistic)
Under H₁:  p-value is stochastically smaller (values cluster near 0).

We reject H₀ at level α if  p ≤ α.  This controls Type I error exactly.


### Power Analysis & Effect Size

POWER = ℙ(reject H₀ | H₁) = 1 − β

Power depends on:
    · Effect size δ = (μ₁ − μ₀)/σ  (signal strength)
    · Sample size n
    · Significance level α
    · Test choice

Approximate power for a one-sample z-test (H₀: μ=μ₀ vs H₁: μ=μ₁):

    Power ≈ Φ(|δ|√n − z_{α/2})

    where z_{α/2} = Φ⁻¹(1−α/2) is the critical value.

    Diagram 5 — Power as a Function of Effect Size:

    Power ↑
    1.0 │                              ╭─────────
        │                          ╭──╯
    0.5 │                     ╭────╯
        │               ╭─────╯
    α   │──────────────╯
        └───────────────────────────────────→ |δ|√n
            (small effect)          (large effect)

REQUIRED SAMPLE SIZE to achieve power 1−β at level α:

    n ≈ (z_α + z_β)² / δ²    (one-sample, one-sided)

Doubling the required effect size reduces n by a factor of 4.
This is the fundamental cost of detecting small signals.


### Multiple Testing

If we run m independent tests at level α each:

    FAMILY-WISE ERROR RATE (FWER) = ℙ(at least one false positive)
        = 1 − (1−α)ᵐ  ≈ mα   for small α

For m=100 and α=0.05:  FWER ≈ 99% — almost certain to make an error!

BONFERRONI CORRECTION: Test each hypothesis at level α/m.
    Controls FWER ≤ α.  Conservative (may have high Type II error rate).

BENJAMINI-HOCHBERG (BH) PROCEDURE: Controls the FALSE DISCOVERY RATE:
    FDR = 𝔼[#{false positives} / #{rejections}]

    BH algorithm: Sort p-values p₍₁₎ ≤ p₍₂₎ ≤ ··· ≤ p₍ₘ₎.
    Let k = max{i: p₍ᵢ₎ ≤ i·α/m}.  Reject H₀₍₁₎,...,H₀₍ₖ₎.

BH is less conservative than Bonferroni for large m.
Modern high-throughput experiments (genomics, neuroimaging) use BH.


### PART 7 — CONFIDENCE INTERVALS

### Frequentist Confidence Intervals

A (1−α) CONFIDENCE INTERVAL [L(X), U(X)] satisfies:

    ℙ_θ(L(X) ≤ θ ≤ U(X)) = 1 − α    for all θ

The interval is RANDOM (it depends on X); the parameter θ is FIXED.

    ┌──────────────────────────────────────────────────────────┐
    │  CORRECT INTERPRETATION: In 95% of repeated experiments, │
    │  the computed interval will contain the true θ.          │
    │                                                          │
    │  WRONG: "There is a 95% chance θ lies in [L, U]."        │
    │  (After observing data, θ is a fixed point — either      │
    │   in [L,U] or not. Probability statements about θ        │
    │   require a Bayesian framework.)                         │
    └──────────────────────────────────────────────────────────┘

EXACT CI for Normal mean (σ known):
    [X̄ − z_{α/2}·σ/√n,   X̄ + z_{α/2}·σ/√n]   width = 2z_{α/2}·σ/√n

EXACT CI for Normal mean (σ unknown):
    [X̄ − t_{α/2,n-1}·S/√n,   X̄ + t_{α/2,n-1}·S/√n]

    Uses t-distribution (heavier tails than Normal) to compensate for
    not knowing σ.  As n→∞, t_{α/2,n-1} → z_{α/2}.

ASYMPTOTIC CI (via CLT): For any estimator θ̂ₙ with SE(θ̂ₙ):
    [θ̂ₙ − z_{α/2}·SE(θ̂ₙ),   θ̂ₙ + z_{α/2}·SE(θ̂ₙ)]    (for large n)


### Bootstrap Confidence Intervals

The BOOTSTRAP avoids knowing the distribution of the estimator.

Algorithm (Nonparametric Bootstrap):
    1. Draw B bootstrap samples X*₁,...,X*ₙ with replacement from data.
    2. Compute θ̂*_b for each bootstrap sample b = 1,...,B.
    3. The bootstrap distribution of θ̂*_b approximates the sampling
       distribution of θ̂.

PERCENTILE CI: [θ̂*_{α/2}, θ̂*_{1-α/2}]   (α/2 and 1-α/2 quantiles of θ̂*_b)

BIAS-CORRECTED ACCELERATED (BCa) CI: Corrects for skewness and bias
of the bootstrap distribution — recommended in practice.

    ┌──────────────────────────────────────────────────────────┐
    │  Bootstrap validity: Requires estimator to be smooth     │
    │  and the statistic to be "well-behaved."                 │
    │  Fails for non-smooth statistics like the maximum,       │
    │  or when the empirical distribution is a poor proxy.     │
    └──────────────────────────────────────────────────────────┘


### PART 8 — BAYESIAN INFERENCE

### The Bayesian Framework

In the Bayesian framework, θ is a RANDOM VARIABLE with a prior
distribution π(θ) encoding beliefs before observing data.

BAYES' THEOREM for inference:

    π(θ | x) = p(x | θ) · π(θ) / p(x)

    POSTERIOR  ∝  LIKELIHOOD × PRIOR

    p(x) = ∫ p(x|θ)π(θ)dθ  — the MARGINAL LIKELIHOOD (evidence)

The POSTERIOR DISTRIBUTION π(θ|x) encodes all beliefs about θ
after observing x. It is the complete inferential output.

POSTERIOR PREDICTIVE: Distribution of a new observation x̃:
    p(x̃ | x) = ∫ p(x̃|θ) π(θ|x) dθ

BAYESIAN POINT ESTIMATES:
    · Posterior MEAN:   𝔼[θ | x]         (minimises posterior MSE)
    · Posterior MODE:   argmax π(θ|x)     (MAP = Maximum A Posteriori)
    · Posterior MEDIAN: argmin 𝔼[|θ−a| | x]  (minimises posterior MAE)

CREDIBLE INTERVALS: A (1−α) HIGHEST POSTERIOR DENSITY (HPD) region C:
    ℙ(θ ∈ C | x) = 1 − α   and  C = {θ : π(θ|x) ≥ k}

    Interpretation: θ lies in C with probability 1−α. This is the
    Bayesian analogue of a confidence interval — and has the intuitive
    interpretation that frequentist CIs lack.

    ┌──────────────────────────────────────────────────────────┐
    │  Frequentist CI:   The interval is random; θ is fixed.   │
    │  Bayesian CI:      θ is random (given data); the         │
    │                    interval has a direct probability     │
    │                    statement about θ.                    │
    └──────────────────────────────────────────────────────────┘


### Conjugate Priors

A prior π(θ) is CONJUGATE to the likelihood p(x|θ) if the posterior
is in the same distribution family as the prior.

    Likelihood         Conjugate Prior    Posterior
    ───────────────────────────────────────────────────────
    Bernoulli(θ)       Beta(α,β)          Beta(α+Σxᵢ, β+n−Σxᵢ)
    Poisson(θ)         Gamma(α,β)         Gamma(α+Σxᵢ, β+n)
    Normal(μ, σ²)      Normal(μ₀,τ²)      Normal(μₙ, σₙ²)
      known σ²         (on μ)             weighted combination
    Normal(μ, σ²)      Normal-InvGamma    Normal-InvGamma
      unknown both     (on μ,σ²)          (on μ,σ²)
    Multinomial(p)     Dirichlet(α)       Dirichlet(α+counts)
    ───────────────────────────────────────────────────────

For the Bernoulli-Beta conjugate pair (θ ~ Beta(α,β)):
    α + β = PSEUDO-COUNTS (prior equivalent sample size)
    α/(α+β) = PRIOR MEAN
    After observing k successes in n trials:
        Posterior: Beta(α+k, β+n−k)
        Posterior mean: (α+k)/(α+β+n)  — shrinkage toward prior mean

The posterior mean is a WEIGHTED COMBINATION of prior mean and MLE:
    (α/(α+β)) · w₀ + x̄ · (1−w₀)   where w₀ = (α+β)/(α+β+n)
As n→∞, the data dominate and the prior washes out.


### Prior Choices — Informative vs Non-informative

INFORMATIVE PRIORS: Encode genuine domain knowledge.
    E.g., π(p) = Beta(10, 10) encodes belief that a coin is roughly fair.

WEAKLY INFORMATIVE PRIORS: Regularisation without strong bias.
    E.g., N(0, 10) on regression coefficients. Shrinks toward zero;
    prevents wild parameter values.

JEFFREYS PRIOR:   π(θ) ∝ √(det I(θ))
    INVARIANT under reparameterisation: if you change coordinates,
    the Jeffreys prior transforms consistently.
    For Bernoulli(θ): π(θ) = Beta(½, ½)  (flat on arcsin scale).

MAXIMUM ENTROPY PRIOR: Among all priors consistent with constraints,
    choose the one with maximum entropy H(π) = −∫π(θ)log π(θ)dθ.
    Encodes minimal additional assumptions.

EMPIRICAL BAYES: Estimate hyperparameters from the data:
    π̂ = argmax_{π} p(x | π) = argmax ∫ p(x|θ)π(θ|π)dθ
    Blurs the line between Bayesian and frequentist inference.


### Markov Chain Monte Carlo (MCMC)

When the posterior π(θ|x) is intractable (no closed form), we use
simulation-based methods to draw samples from it.

MCMC constructs a Markov chain θ₀, θ₁, θ₂, ... whose STATIONARY
DISTRIBUTION is the target posterior π(θ|x).

METROPOLIS-HASTINGS:
    1. Propose θ' ~ q(θ' | θᵗ)   (proposal distribution)
    2. Accept with probability:
       a = min(1,  π(θ'|x)q(θᵗ|θ') / [π(θᵗ|x)q(θ'|θᵗ)])
    3. θᵗ⁺¹ = θ' if accepted,  θᵗ otherwise.

The acceptance ratio cancels the normalising constant p(x) —
we only need π(θ|x) ∝ p(x|θ)π(θ). This is the key advantage.

GIBBS SAMPLING: When full conditionals π(θᵢ | θ_{−i}, x) are tractable:
    Cycle through coordinates, sampling each from its full conditional.
    No acceptance step needed — always accepted.

HAMILTONIAN MONTE CARLO (HMC): Uses gradient ∇log π(θ|x) to propose
    distant moves along likelihood contours. Dramatically reduces
    random-walk behaviour. Used in Stan, NumPyro, PyMC.

    ┌──────────────────────────────────────────────────────────┐
    │  MCMC convergence diagnostics:                           │
    │  · R̂ (Gelman-Rubin): compares between-chain to           │
    │    within-chain variance. R̂ ≈ 1.0 → converged.           │
    │  · Effective Sample Size (ESS): accounts for             │
    │    autocorrelation in the chain.                         │
    │  · Trace plots: should look like "hairy caterpillars."   │
    └──────────────────────────────────────────────────────────┘


### PART 9 — INFORMATION THEORY & KL DIVERGENCE

### Shannon Entropy

The ENTROPY of a discrete distribution p over k outcomes:

    H(p) = −Σᵢ p(x) log p(x)   (nats if log base e; bits if log base 2)

Entropy measures UNCERTAINTY or INFORMATION CONTENT.
    H(p) = 0:  deterministic (one outcome has probability 1)
    H(p) = log k:  maximum uncertainty (uniform distribution)

For a continuous density f:  DIFFERENTIAL ENTROPY h(f) = −∫f(x)log f(x)dx

    h(N(μ,σ²)) = ½log(2πeσ²)   (Gaussian maximises differential entropy
                                  among all densities with fixed variance)


### KL Divergence — Information Gain

The KULLBACK-LEIBLER DIVERGENCE from q to p:

    D_KL(p ‖ q) = Σᵢ p(x) log [p(x)/q(x)]  (discrete)
                = ∫ p(x) log [p(x)/q(x)] dx  (continuous)

    D_KL(p ‖ q) ≥ 0   with equality iff p = q  (Gibbs' inequality)

INTERPRETATION:
    · Extra bits needed to encode p using a code optimised for q.
    · D_KL(p‖q) ≠ D_KL(q‖p) — KL is ASYMMETRIC.
    · D_KL(p‖q) = ∞ if q(x) = 0 and p(x) > 0 anywhere.

    ┌──────────────────────────────────────────────────────────┐
    │  FORWARD KL  D_KL(p‖q):  "mean-seeking" — q spreads to   │
    │  cover all of p. Used in variational inference (ELBO).   │
    │  REVERSE KL  D_KL(q‖p):  "mode-seeking" — q concentrates │
    │  on a mode of p. Used in expectation propagation.        │
    └──────────────────────────────────────────────────────────┘

For two Gaussians: D_KL(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²)):
    = log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²)/(2σ₂²) − ½

MUTUAL INFORMATION:
    I(X;Y) = D_KL(p(X,Y) ‖ p(X)p(Y)) = H(X) − H(X|Y) = H(Y) − H(Y|X)

Mutual information measures how much knowing Y reduces uncertainty in X.
I(X;Y) = 0 iff X ⊥⊥ Y.


### Connection: KL Divergence and MLE

MLE = minimising KL divergence from the empirical distribution p̂ₙ to the
model family {p_θ}:

    θ̂_MLE = argmin_θ D_KL(p̂ₙ ‖ p_θ)
           = argmax_θ (1/n)Σᵢ log p_θ(xᵢ)

MLE is the projection of the empirical distribution onto the model family
in the KL geometry. The Fisher information matrix is the Riemannian metric
in this parameter space (information geometry).

CROSS-ENTROPY LOSS in classification:

    H(p, q) = −Σₓ p(x) log q(x) = H(p) + D_KL(p ‖ q)

Since H(p) is constant w.r.t. q, minimising cross-entropy = minimising KL.


### PART 10 — ASYMPTOTICS & CONSISTENCY

### Modes of Convergence

Let Xₙ be a sequence of random variables:

    IN PROBABILITY (→ᵖ):
        Xₙ →ᵖ X  if  ∀ε>0:  ℙ(|Xₙ − X| > ε) → 0

    ALMOST SURELY (→ᵃ·ˢ·):
        Xₙ →ᵃ·ˢ· X  if  ℙ(lim_{n→∞} Xₙ = X) = 1

    IN DISTRIBUTION (→ᵈ):
        Xₙ →ᵈ X  if  Fₙ(x) → F(x) at all continuity points of F

    IN Lᵖ (→^Lᵖ):
        Xₙ →^Lᵖ X  if  𝔼[|Xₙ − X|ᵖ] → 0

    Hierarchy (for convergence to a constant c):
    a.s. → probability → distribution
    L² → probability
    Convergence in distribution does NOT imply a.s. or in probability.

SLUTSKY'S THEOREM: If Xₙ →ᵈ X and Yₙ →ᵖ c (constant):
    Xₙ + Yₙ →ᵈ X + c
    Xₙ · Yₙ →ᵈ c · X
    Xₙ / Yₙ →ᵈ X / c  (c ≠ 0)

This is the workhorse for deriving asymptotic distributions of
estimators — replace unknown nuisance parameters by consistent estimates.


### Asymptotic Theory of MLE

Under regularity conditions (smoothness, identifiability, compact Θ):

    CONSISTENCY:         θ̂_MLE →ᵖ θ₀
    ASYMPTOTIC NORMALITY: √n(θ̂_MLE − θ₀) →ᵈ N(0, I(θ₀)⁻¹)
    ASYMPTOTIC EFFICIENCY: No consistent estimator has smaller
                           asymptotic variance than I(θ₀)⁻¹.

The OBSERVED FISHER INFORMATION:

    Î(θ̂) = −(1/n) ∂²ℓ(θ)/∂θ² |_{θ=θ̂}

Replaces I(θ₀) in practice. Then:  √n(θ̂ − θ₀) ≈ N(0, Î(θ̂)⁻¹)

WILKS' THEOREM: For testing H₀: θ = θ₀ vs H₁: θ ≠ θ₀ using the
    LIKELIHOOD RATIO STATISTIC:

        Λ = 2[ℓ(θ̂_MLE) − ℓ(θ₀)]   →ᵈ   Χ²(d)   under H₀

    where d = dim(θ). This gives the LIKELIHOOD RATIO TEST — one of the
    most powerful tests for composite hypotheses.


### Frequentist vs Bayesian — Conceptual Summary

    ┌──────────────────┬─────────────────────┬─────────────────────┐
    │                  │  FREQUENTIST        │  BAYESIAN           │
    ├──────────────────┼─────────────────────┼─────────────────────┤
    │ Parameters       │ Fixed, unknown      │ Random variables    │
    │ Probability      │ Long-run frequency  │ Degree of belief    │
    │ Inference output │ Point est. + CI     │ Posterior dist.     │
    │ Data quantity    │ Asymptotic valid.   │ Valid for any n     │
    │ Prior knowledge  │ Not incorporated    │ Encoded in prior    │
    │ Small sample     │ Approximate         │ Exact (given prior) │
    │ Computation      │ Often analytic      │ Often requires MCMC │
    │ Regularisation   │ Explicit penalty    │ Informative prior   │
    └──────────────────┴─────────────────────┴─────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │  As n → ∞, Bayesian posterior concentrates at the MLE:       │
    │  The Bernstein-von Mises theorem states that under           │
    │  regularity conditions:                                      │
    │     π(θ | x₁,...,xₙ)  →  N(θ̂_MLE,  I(θ₀)⁻¹/n)                 │
    │  The two frameworks AGREE asymptotically.                    │
    │  Differences matter most with small data or strong priors.   │
    └──────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · MLE, Bias-Variance & the Cramér-Rao Bound": {
        "description": (
            "Compute and compare MLE, method-of-moments, and Bayes estimators "
            "for Normal and Bernoulli data. Empirically verify the Cramér-Rao "
            "lower bound, the bias-variance decomposition, and MLE asymptotic "
            "normality. Visualise sampling distributions and MSE curves."
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

from scipy import stats

np.random.seed(42)

print("=" * 65)
print("  MLE, BIAS-VARIANCE & THE CRAMÉR-RAO BOUND")
print("=" * 65)
print()

# ── Part 1: Sampling Distribution of the MLE for Normal(μ, σ²) ───────────
print("  PART 1 — SAMPLING DISTRIBUTION OF MLE (Normal Data)")
print()

mu_true  = 3.0
sig_true = 2.0
n_sizes  = [5, 20, 100, 500]
n_sims   = 10_000

print(f"  True parameters: μ = {mu_true}, σ² = {sig_true**2}")
print()
print(f"  {'n':>6} | {'E[μ̂]':>10} | {'Bias(μ̂)':>10} | {'Var(μ̂)':>12} | "
      f"{'CRLB':>12} | {'MSE':>12}")
print(f"  {'─'*72}")

results = {}
for n in n_sizes:
    samples   = np.random.normal(mu_true, sig_true, (n_sims, n))
    mu_hat    = samples.mean(axis=1)        # MLE for mean
    var_hat   = samples.var(axis=1)         # MLE for variance (biased)
    s2        = samples.var(axis=1, ddof=1) # unbiased estimator

    bias_mu   = mu_hat.mean() - mu_true
    var_mu    = mu_hat.var()
    crlb_mu   = sig_true**2 / n            # 1 / (n · I(θ))
    mse_mu    = ((mu_hat - mu_true)**2).mean()
    results[n] = {"mu_hat": mu_hat, "var_hat": var_hat, "s2": s2}
    print(f"  {n:>6} | {mu_hat.mean():10.5f} | {bias_mu:10.6f} | "
          f"{var_mu:12.6f} | {crlb_mu:12.6f} | {mse_mu:12.6f}")

print()
print("  Cramér-Rao bound is tight: Var(μ̂) ≈ CRLB = σ²/n at all n.")
print()

# ── Part 2: Biased vs Unbiased Variance Estimator ─────────────────────────
print("  PART 2 — BIASED MLE vs UNBIASED ESTIMATOR FOR VARIANCE")
print()
print(f"  True σ² = {sig_true**2:.4f}")
print()
print(f"  {'n':>6} | {'E[σ̂²_MLE]':>12} | {'E[S²]':>12} | {'Bias(MLE)':>12} | {'MSE(MLE)':>12} | {'MSE(S²)':>12}")
print(f"  {'─'*75}")
for n in n_sizes:
    r = results[n]
    e_var  = r["var_hat"].mean()
    e_s2   = r["s2"].mean()
    bias_v = e_var - sig_true**2
    mse_v  = ((r["var_hat"] - sig_true**2)**2).mean()
    mse_s2 = ((r["s2"] - sig_true**2)**2).mean()
    print(f"  {n:>6} | {e_var:12.5f} | {e_s2:12.5f} | {bias_v:12.5f} | "
          f"{mse_v:12.5f} | {mse_s2:12.5f}")
print()
print("  MLE for σ² is biased low by factor (n-1)/n. For large n, negligible.")
print("  S² is unbiased, but MLE can have LOWER MSE for small n (bias-var trade-off).")
print()

# ── Part 3: MLE for Bernoulli — asymptotic normality check ────────────────
print("  PART 3 — MLE ASYMPTOTIC NORMALITY (Bernoulli Data)")
print()
p_true = 0.3
I_bern = p_true * (1 - p_true)   # Fisher information for Bernoulli

print(f"  Bernoulli(p = {p_true}),  Fisher info I(p) = p(1-p) = {I_bern:.4f}")
print(f"  CRLB for p̂:  1/(n·I(p)) = 1/(n·{I_bern:.4f})")
print()
print(f"  {'n':>6} | {'E[p̂]':>10} | {'Var(p̂)':>12} | {'CRLB':>12} | "
      f"{'W-stat (KS p)':>15}")
print(f"  {'─'*68}")

for n in [10, 50, 200, 1000]:
    samples = np.random.binomial(n, p_true, n_sims) / n
    e_p     = samples.mean()
    var_p   = samples.var()
    crlb_p  = I_bern / n
    # Standardise and test normality
    std_samp = (samples - p_true) / np.sqrt(I_bern / n)
    ks_stat, ks_pval = stats.kstest(std_samp, "norm")
    print(f"  {n:>6} | {e_p:10.5f} | {var_p:12.6f} | {crlb_p:12.6f} | "
          f"p={ks_pval:13.4f}")
print()
print("  KS p-values: large values confirm convergence to Normal (fail to reject).")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("MLE, Bias-Variance & Cramér-Rao Bound",
             fontsize=12, fontweight="bold")

# Plot 1: Sampling distributions of μ̂ for different n
colors = ["tomato", "orange", "steelblue", "seagreen"]
x_plot = np.linspace(1.0, 5.0, 400)
for (n, col) in zip(n_sizes, colors):
    mu_hat = results[n]["mu_hat"]
    kde    = stats.gaussian_kde(mu_hat)
    axes[0].plot(x_plot, kde(x_plot), lw=2, color=col, label=f"n={n}")
    crlb = sig_true**2 / n
    axes[0].plot(x_plot, stats.norm.pdf(x_plot, mu_true, np.sqrt(crlb)),
                 lw=1.2, color=col, linestyle="--", alpha=0.5)
axes[0].axvline(mu_true, color="black", lw=1.5, linestyle=":", label=f"True μ={mu_true}")
axes[0].set_xlabel("μ̂"); axes[0].set_ylabel("Density")
axes[0].set_title("Sampling Distribution of μ̂_MLE\\n(dashed = asymptotic Normal)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Plot 2: MSE vs n for MLE (biased) and S² (unbiased)
n_range = np.arange(3, 201)
n_sims2 = 5000
mse_mle = []; mse_s2 = []
for n in n_range:
    samp   = np.random.normal(mu_true, sig_true, (n_sims2, n))
    v_mle  = samp.var(axis=1)
    v_s2   = samp.var(axis=1, ddof=1)
    mse_mle.append(((v_mle - sig_true**2)**2).mean())
    mse_s2.append(((v_s2 - sig_true**2)**2).mean())
axes[1].semilogy(n_range, mse_mle, "steelblue", lw=2, label="MLE (biased)")
axes[1].semilogy(n_range, mse_s2, "tomato", lw=2, label="S² (unbiased)")
axes[1].set_xlabel("Sample size n"); axes[1].set_ylabel("MSE (log scale)")
axes[1].set_title("MSE of Variance Estimators\\nBias-variance tradeoff")
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

# Plot 3: CRLB vs empirical variance across n
n_range2 = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
emp_vars  = []
crlbs     = []
for n in n_range2:
    samp   = np.random.normal(mu_true, sig_true, (n_sims, n))
    emp_vars.append(samp.mean(axis=1).var())
    crlbs.append(sig_true**2 / n)
axes[2].loglog(n_range2, crlbs, "k--", lw=2, label="CRLB = σ²/n")
axes[2].loglog(n_range2, emp_vars, "steelblue", lw=2, marker="o", label="Empirical Var(μ̂)")
axes[2].set_xlabel("n (log scale)"); axes[2].set_ylabel("Variance (log scale)")
axes[2].set_title("MLE Variance vs Cramér-Rao Bound\\nMLE is asymptotically efficient")
axes[2].legend(fontsize=10); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mle_crlb.png", dpi=120)
print("  Plot saved → mle_crlb.png")
print()
print("  KEY TAKEAWAYS:")
print("  MLE for μ is unbiased with Var = σ²/n = CRLB (efficient).")
print("  MLE for σ² is biased low; S² is unbiased. At small n, MLE")
print("  can have lower MSE despite bias (bias-variance tradeoff).")
print("  √n(p̂ - p) → N(0, p(1-p)) confirmed by KS test convergence.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Hypothesis Testing — Power, p-values & Multiple Testing": {
        "description": (
            "Simulate the distribution of p-values under H₀ and H₁. "
            "Plot power curves as a function of sample size and effect size. "
            "Compare Bonferroni and Benjamini-Hochberg corrections for "
            "multiple testing. Demonstrate that p-values are Uniform(0,1) "
            "under H₀ (exact) and stochastically smaller under H₁."
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

from scipy import stats

np.random.seed(0)

print("=" * 65)
print("  HYPOTHESIS TESTING: POWER, p-VALUES & MULTIPLE TESTING")
print("=" * 65)
print()

# ── Part 1: p-value distribution under H₀ and H₁ ─────────────────────────
print("  PART 1 — p-VALUE DISTRIBUTIONS")
print()

n_sims  = 50_000
n_obs   = 30
alpha   = 0.05

# Under H₀: X ~ N(0,1), test H₀: μ=0, one-sample z-test
pvals_H0 = 2 * stats.norm.sf(np.abs(
    np.random.normal(0, 1, (n_sims, n_obs)).mean(axis=1) * np.sqrt(n_obs)
))

# Under H₁ with varying effect sizes
effect_sizes = [0.2, 0.5, 1.0]
pvals_H1 = {}
for delta in effect_sizes:
    z_stats = (np.random.normal(delta, 1, (n_sims, n_obs)).mean(axis=1)
               * np.sqrt(n_obs))
    pvals_H1[delta] = 2 * stats.norm.sf(np.abs(z_stats))

print(f"  Under H₀ (μ=0, n={n_obs}):  p~Uniform(0,1)")
ks_stat, ks_p = stats.kstest(pvals_H0, "uniform")
print(f"  KS test vs Uniform(0,1): statistic={ks_stat:.4f},  p={ks_p:.4f}")
print(f"  Rejection rate at α={alpha}: {(pvals_H0 < alpha).mean():.4f}  (should ≈ {alpha})")
print()
print(f"  Under H₁ — rejection rates (power) at α={alpha}, n={n_obs}:")
for delta in effect_sizes:
    power = (pvals_H1[delta] < alpha).mean()
    print(f"    δ = {delta:.1f}:  power = {power:.4f}")
print()

# ── Part 2: Power curves ───────────────────────────────────────────────────
print("  PART 2 — POWER CURVES")
print()

n_range = np.arange(5, 201, 5)
alphas  = [0.01, 0.05, 0.10]
delta_p = 0.5   # fixed effect size for power-vs-n plot
n_sims_p = 20_000

print(f"  Effect size δ={delta_p}, varying n:")
print(f"  {'n':>6} | {'α=0.01':>10} | {'α=0.05':>10} | {'α=0.10':>10}")
print(f"  {'─'*45}")
power_curves = {a: [] for a in alphas}
for n in n_range:
    z_stat = (np.random.normal(delta_p, 1, (n_sims_p, n)).mean(axis=1)
              * np.sqrt(n))
    pv = 2 * stats.norm.sf(np.abs(z_stat))
    for a in alphas:
        pwr = (pv < a).mean()
        power_curves[a].append(pwr)
    if n in [10, 30, 50, 100, 200]:
        print(f"  {n:>6} | {power_curves[0.01][-1]:10.4f} | "
              f"{power_curves[0.05][-1]:10.4f} | {power_curves[0.10][-1]:10.4f}")
print()

# Required n formula: n = (z_α + z_β)² / δ²  (one-sided approx)
print("  Required n for 80% power, two-sided test:")
print(f"  {'δ':>6} | {'n (formula)':>14} | {'n (simulation)':>16}")
print(f"  {'─'*42}")
for delta in [0.2, 0.5, 0.8, 1.0, 2.0]:
    n_formula = int(np.ceil((stats.norm.ppf(0.975) + stats.norm.ppf(0.8))**2 / delta**2))
    # empirical
    z_test_stat = (np.random.normal(delta, 1, (n_sims_p, n_formula)).mean(axis=1)
                   * np.sqrt(n_formula))
    pv_emp = 2 * stats.norm.sf(np.abs(z_test_stat))
    emp_power = (pv_emp < 0.05).mean()
    print(f"  {delta:>6.1f} | {n_formula:>14} | {emp_power:>14.3f} (emp. power)")
print()

# ── Part 3: Multiple testing corrections ──────────────────────────────────
print("  PART 3 — MULTIPLE TESTING CORRECTIONS")
print()

m_tests = 200   # total tests
m_true  = 20    # true positives (H₁ true)
delta_m = 0.8   # effect size for true positives

# Simulate p-values: first m_true from H₁, rest from H₀
# Each test uses n=30 observations; .mean(axis=1) gives one z-stat per test
z_h1    = np.random.normal(delta_m, 1, (m_true,        30)).mean(axis=1) * np.sqrt(30)
z_h0    = np.random.normal(0,       1, (m_tests-m_true, 30)).mean(axis=1) * np.sqrt(30)
pvals_m = np.concatenate(
    [2*stats.norm.sf(np.abs(z_h1)), 2*stats.norm.sf(np.abs(z_h0))]
)

# Which are truly null
truly_null = np.array([False]*m_true + [True]*(m_tests - m_true))

def compute_fdr_power(pvals, reject_mask, truly_null):
    rejections = reject_mask.sum()
    if rejections == 0:
        return 0.0, 0.0
    false_pos = (reject_mask & truly_null).sum()
    true_pos  = (reject_mask & ~truly_null).sum()
    fdr   = false_pos / max(rejections, 1)
    power = true_pos  / max((~truly_null).sum(), 1)
    return fdr, power

# Uncorrected
reject_unc = pvals_m < 0.05
fdr_unc, pwr_unc = compute_fdr_power(pvals_m, reject_unc, truly_null)

# Bonferroni
alpha_bonf  = 0.05 / m_tests
reject_bonf = pvals_m < alpha_bonf
fdr_bonf, pwr_bonf = compute_fdr_power(pvals_m, reject_bonf, truly_null)

# Benjamini-Hochberg (BH)
sorted_idx = np.argsort(pvals_m)
bh_thresh  = (np.arange(1, m_tests+1) * 0.05 / m_tests)
k_bh = np.where(pvals_m[sorted_idx] <= bh_thresh)[0]
reject_bh  = np.zeros(m_tests, dtype=bool)
if len(k_bh) > 0:
    reject_bh[sorted_idx[:k_bh[-1]+1]] = True
fdr_bh, pwr_bh = compute_fdr_power(pvals_m, reject_bh, truly_null)

print(f"  m={m_tests} tests, {m_true} true positives, δ={delta_m}, n=30")
print()
print(f"  {'Method':>18} | {'Rejections':>12} | {'FDR':>8} | {'Power':>8}")
print(f"  {'─'*55}")
for name, reject, fdr, pwr in [
    ("Uncorrected (α=0.05)", reject_unc, fdr_unc, pwr_unc),
    ("Bonferroni",           reject_bonf, fdr_bonf, pwr_bonf),
    ("Benjamini-Hochberg",   reject_bh,  fdr_bh,  pwr_bh),
]:
    print(f"  {name:>18} | {reject.sum():>12} | {fdr:8.4f} | {pwr:8.4f}")
print()
print("  Uncorrected: high power but inflated FDR.")
print("  Bonferroni:  controls FWER; FDR near 0 but power suffers.")
print("  BH:          controls FDR at 0.05; keeps power close to uncorrected.")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Hypothesis Testing: Power, p-values & Multiple Testing",
             fontsize=12, fontweight="bold")

# Plot 1: p-value histograms under H₀ and H₁
bins = np.linspace(0, 1, 26)
axes[0].hist(pvals_H0, bins=bins, color="steelblue", alpha=0.6,
             density=True, label="H₀ (μ=0)")
axes[0].hist(pvals_H1[0.5], bins=bins, color="tomato", alpha=0.6,
             density=True, label="H₁ (δ=0.5)")
axes[0].hist(pvals_H1[1.0], bins=bins, color="seagreen", alpha=0.6,
             density=True, label="H₁ (δ=1.0)")
axes[0].axhline(1.0, color="black", linestyle="--", lw=1.5, label="Uniform(0,1)")
axes[0].axvline(0.05, color="gray", linestyle=":", lw=1.5, label="α=0.05")
axes[0].set_xlabel("p-value"); axes[0].set_ylabel("Density")
axes[0].set_title(f"p-value Distributions (n={n_obs})\\nUniform under H₀; small under H₁")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Plot 2: Power curves
palette = ["tomato", "steelblue", "seagreen"]
for a, col in zip(alphas, palette):
    axes[1].plot(n_range, power_curves[a], lw=2, color=col, label=f"α={a}")
axes[1].axhline(0.8, color="gray", linestyle="--", lw=1.5, label="80% power target")
axes[1].set_xlabel("Sample size n"); axes[1].set_ylabel("Power")
axes[1].set_title(f"Power vs Sample Size (δ={delta_p})\\nPower grows with n and α")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.05)

# Plot 3: Sorted p-values and BH threshold line
sorted_pv = np.sort(pvals_m)
rank       = np.arange(1, m_tests + 1)
bh_line    = rank * 0.05 / m_tests
axes[2].plot(rank, sorted_pv, "steelblue", lw=1.5, label="Sorted p-values")
axes[2].plot(rank, bh_line,   "tomato",    lw=2.0, linestyle="--", label="BH threshold")
axes[2].axhline(0.05,         color="gray", linestyle=":", lw=1.5, label="α=0.05")
axes[2].axhline(alpha_bonf,   color="purple", linestyle=":", lw=1.5,
                label=f"Bonferroni (α/m={alpha_bonf:.4f})")
if len(k_bh) > 0:
    axes[2].axvline(k_bh[-1]+1, color="tomato", lw=1, alpha=0.5,
                    linestyle=":", label=f"BH cutoff k={k_bh[-1]+1}")
axes[2].set_xlabel("Rank"); axes[2].set_ylabel("p-value")
axes[2].set_title(f"Sorted p-values vs BH Threshold\\n({m_tests} tests, {m_true} true H₁)")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
axes[2].set_xlim(0, m_tests); axes[2].set_ylim(-0.01, 0.4)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hypothesis_testing.png", dpi=120)
print("  Plot saved → hypothesis_testing.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Bayesian Inference — Conjugate Priors & Posterior Updates": {
        "description": (
            "Visualise sequential Bayesian updating for a Bernoulli model "
            "using Beta conjugate priors. Compare posteriors from informative "
            "vs diffuse priors. Compute MAP and posterior mean estimators. "
            "Demonstrate Bernstein-von Mises: posterior convergence to MLE. "
            "Explore Normal-Normal conjugacy for known and unknown variance."
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

from scipy import stats

np.random.seed(7)

print("=" * 65)
print("  BAYESIAN INFERENCE: CONJUGATE PRIORS & POSTERIOR UPDATES")
print("=" * 65)
print()

# ── Part 1: Beta-Bernoulli Sequential Updating ────────────────────────────
print("  PART 1 — BETA-BERNOULLI CONJUGATE UPDATING")
print()

p_true = 0.35
n_total = 200
data    = (np.random.uniform(0, 1, n_total) < p_true).astype(int)

# Three priors
priors = [
    ("Uniform   Beta(1,1)",    1.0,  1.0,  "steelblue"),
    ("Inform.   Beta(10,20)",  10.0, 20.0, "tomato"),
    ("Weak      Beta(2,2)",    2.0,  2.0,  "seagreen"),
]

print(f"  True p = {p_true},  n = {n_total}")
print()
print(f"  After ALL {n_total} observations:")
print(f"  {'Prior':>22} | {'α_post':>8} | {'β_post':>8} | {'Post. Mean':>12} | "
      f"{'MAP':>10} | {'95% CI':>20}")
print(f"  {'─'*82}")

k = data.sum()   # total successes
for name, a0, b0, col in priors:
    a_post = a0 + k
    b_post = b0 + (n_total - k)
    post_mean = a_post / (a_post + b_post)
    # MAP = (a-1)/(a+b-2)
    map_est   = (a_post - 1) / (a_post + b_post - 2) if (a_post + b_post > 2) else post_mean
    lo, hi    = stats.beta.ppf([0.025, 0.975], a_post, b_post)
    print(f"  {name:>22} | {a_post:8.1f} | {b_post:8.1f} | {post_mean:12.6f} | "
          f"{map_est:10.6f} | [{lo:.4f}, {hi:.4f}]")

print()
print(f"  MLE (sample proportion) = {k/n_total:.6f}")
print(f"  As n increases, posteriors converge to MLE (Bernstein-von Mises).")
print()

# ── Part 2: Posterior convergence to MLE (Bernstein-von Mises) ───────────
print("  PART 2 — BERNSTEIN-VON MISES: POSTERIOR → N(MLE, CRLB) AS n→∞")
print()
a0, b0 = 5.0, 15.0   # informative prior (prior mean = 0.25)
n_steps = [1, 5, 20, 50, 200]
print(f"  Prior: Beta({a0},{b0}), prior mean = {a0/(a0+b0):.3f},  True p = {p_true}")
print()
print(f"  {'n':>6} | {'Post. Mean':>12} | {'MLE':>10} | {'|Post − MLE|':>14} | "
      f"{'Post SD':>10} | {'CRLB SD':>10}")
print(f"  {'─'*72}")
for n in n_steps:
    d      = data[:n]
    k_n    = d.sum()
    mle_n  = k_n / n
    a_n    = a0 + k_n
    b_n    = b0 + (n - k_n)
    pm_n   = a_n / (a_n + b_n)
    psd_n  = np.sqrt(a_n * b_n / ((a_n + b_n)**2 * (a_n + b_n + 1)))
    crlb_n = np.sqrt(p_true * (1 - p_true) / n)
    print(f"  {n:>6} | {pm_n:12.6f} | {mle_n:10.6f} | {abs(pm_n - mle_n):14.6f} | "
          f"{psd_n:10.6f} | {crlb_n:10.6f}")
print()

# ── Part 3: Normal-Normal conjugacy (known variance) ─────────────────────
print("  PART 3 — NORMAL-NORMAL CONJUGACY (σ² known)")
print()
mu_true_n = 5.0
sigma_n   = 2.0

# Prior: N(mu_0, tau_0^2)
mu_0, tau_0 = 0.0, 3.0
print(f"  Prior: N(μ₀={mu_0}, τ₀²={tau_0**2}), known σ²={sigma_n**2}")
print()
print(f"  Posterior formula:")
print(f"    σₙ² = 1 / (1/τ₀² + n/σ²)")
print(f"    μₙ  = σₙ² · (μ₀/τ₀² + n·x̄/σ²)   = shrinkage of x̄ toward μ₀")
print()
print(f"  {'n':>6} | {'x̄':>10} | {'Post. Mean μₙ':>14} | {'Post. SD σₙ':>12} | "
      f"{'Shrinkage weight':>18}")
print(f"  {'─'*68}")
for n in [1, 5, 20, 50, 200]:
    x     = np.random.normal(mu_true_n, sigma_n, n)
    xbar  = x.mean()
    sig_n_sq = 1 / (1/tau_0**2 + n/sigma_n**2)
    mu_n  = sig_n_sq * (mu_0/tau_0**2 + n*xbar/sigma_n**2)
    sig_n = np.sqrt(sig_n_sq)
    w     = (1/tau_0**2) / (1/tau_0**2 + n/sigma_n**2)   # weight on prior
    print(f"  {n:>6} | {xbar:10.4f} | {mu_n:14.6f} | {sig_n:12.6f} | "
          f"prior weight = {w:8.4f}")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Bayesian Inference: Conjugate Priors & Posterior Updates",
             fontsize=12, fontweight="bold")

# Plot 1: Sequential Beta-Bernoulli updating
theta = np.linspace(0.01, 0.99, 400)
update_steps = [0, 3, 10, 30, n_total]
palette = ["#AAAAAA", "#8ab4e8", "#4a86d4", "#1f5fa8", "#0d2e5a"]
a0_s, b0_s = 1.0, 1.0   # uniform prior for plot

for (step, col) in zip(update_steps, palette):
    k_s = data[:step].sum() if step > 0 else 0
    a_s = a0_s + k_s
    b_s = b0_s + (step - k_s)
    label = (f"Prior Beta(1,1)" if step == 0
             else f"n={step}: Beta({a_s:.0f},{b_s:.0f})")
    axes[0].plot(theta, stats.beta.pdf(theta, a_s, b_s), lw=2, color=col, label=label)
axes[0].axvline(p_true, color="red", lw=1.5, linestyle="--", label=f"True p={p_true}")
axes[0].axvline(k/n_total, color="black", lw=1.2, linestyle=":", label=f"MLE={k/n_total:.2f}")
axes[0].set_xlabel("θ (p)"); axes[0].set_ylabel("Posterior density")
axes[0].set_title("Sequential Beta-Bernoulli Updates\\n(Uniform prior, darkens with more data)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Plot 2: Three priors and posteriors
a0_vals = [1, 10, 2]; b0_vals = [1, 20, 2]
prior_labels = ["Uniform Beta(1,1)", "Inform. Beta(10,20)", "Weak Beta(2,2)"]
cols_priors = ["steelblue", "tomato", "seagreen"]
for a0_p, b0_p, name, col in zip(a0_vals, b0_vals, prior_labels, cols_priors):
    a_pr = a0_p + k; b_pr = b0_p + (n_total - k)
    axes[1].plot(theta, stats.beta.pdf(theta, a0_p, b0_p),
                 lw=1.5, color=col, linestyle="--", alpha=0.6)
    axes[1].plot(theta, stats.beta.pdf(theta, a_pr, b_pr),
                 lw=2.5, color=col, label=f"{name}")
axes[1].axvline(p_true, color="red", lw=1.5, linestyle="--", label=f"True p")
axes[1].axvline(k/n_total, color="black", lw=1.2, linestyle=":", label="MLE")
axes[1].set_xlabel("θ"); axes[1].set_ylabel("Density")
axes[1].set_title(f"Prior (dashed) vs Posterior (solid)\\n(n={n_total}, posteriors converge)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# Plot 3: Normal-Normal posterior shrinkage
n_range_nn = np.arange(1, 201)
mu_0_nn, tau_0_nn = 0.0, 3.0
sigma_nn = 2.0
xbar_nn  = 5.2   # fixed x̄ for illustration
mu_posts = []; lb_posts = []; ub_posts = []
for n in n_range_nn:
    sn2   = 1 / (1/tau_0_nn**2 + n/sigma_nn**2)
    mu_n  = sn2 * (mu_0_nn/tau_0_nn**2 + n*xbar_nn/sigma_nn**2)
    mu_posts.append(mu_n)
    lb_posts.append(mu_n - 1.96*np.sqrt(sn2))
    ub_posts.append(mu_n + 1.96*np.sqrt(sn2))
axes[2].plot(n_range_nn, mu_posts, "steelblue", lw=2, label="Posterior mean")
axes[2].fill_between(n_range_nn, lb_posts, ub_posts, alpha=0.15, color="steelblue",
                     label="95% credible interval")
axes[2].axhline(xbar_nn, color="tomato", lw=1.5, linestyle="--", label=f"MLE (x̄={xbar_nn})")
axes[2].axhline(mu_0_nn, color="gray", lw=1.5, linestyle=":", label=f"Prior mean ({mu_0_nn})")
axes[2].set_xlabel("n"); axes[2].set_ylabel("μ")
axes[2].set_title("Normal-Normal Conjugacy\\nPosterior shrinkage as n grows")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bayesian_inference.png", dpi=120)
print("  Plot saved → bayesian_inference.png")
print()
print("  KEY TAKEAWAYS:")
print("  Prior information is overridden by data as n grows.")
print("  Informative priors dominate at small n; all posteriors")
print("  converge to the likelihood (MLE) as n → ∞ (BvM theorem).")
print("  Normal-Normal: posterior mean = weighted average of prior")
print("  mean and MLE, with weights proportional to precision (1/var).")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Bootstrap, CLT & KL Divergence": {
        "description": (
            "Demonstrate the CLT empirically for non-Normal data (Exponential, "
            "Bernoulli). Build bootstrap confidence intervals for the mean, "
            "median, and variance. Compare bootstrap, normal, and t-CIs. "
            "Compute KL divergence between parametric families and visualise "
            "the asymmetry of D_KL(p‖q) vs D_KL(q‖p)."
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

from scipy import stats

np.random.seed(13)

print("=" * 65)
print("  BOOTSTRAP, CLT & KL DIVERGENCE")
print("=" * 65)
print()

# ── Part 1: Central Limit Theorem for non-Normal populations ──────────────
print("  PART 1 — CLT FOR NON-NORMAL POPULATIONS")
print()

populations = {
    "Exponential(1)":  (lambda n: np.random.exponential(1, n), 1.0, 1.0),
    "Bernoulli(0.1)":  (lambda n: np.random.binomial(1, 0.1, n), 0.1, 0.09),
    "Uniform(0,1)":    (lambda n: np.random.uniform(0, 1, n), 0.5, 1/12),
}

n_sims = 20_000

print(f"  Testing H₀: μ=true, using standardised X̄ₙ ~ N(0,1) under CLT")
print()
print(f"  {'Population':>20} | {'n':>5} | {'KS stat':>10} | {'KS p-val':>10} | {'Skew of X̄':>12}")
print(f"  {'─'*65}")
clt_data = {}
for pop_name, (gen, mu, var) in populations.items():
    clt_data[pop_name] = {}
    for n in [5, 30, 100]:
        xbar = np.array([gen(n).mean() for _ in range(n_sims)])
        std_xbar = (xbar - mu) / np.sqrt(var / n)
        ks_stat, ks_p = stats.kstest(std_xbar, "norm")
        skewness = stats.skew(std_xbar)
        clt_data[pop_name][n] = std_xbar
        marker = "← Normal ✓" if ks_p > 0.05 else ""
        print(f"  {pop_name:>20} | {n:>5} | {ks_stat:10.4f} | {ks_p:10.4f} | "
              f"{skewness:12.4f} {marker}")
print()

# ── Part 2: Bootstrap confidence intervals ────────────────────────────────
print("  PART 2 — BOOTSTRAP CONFIDENCE INTERVALS")
print()
n_boot  = 1000
B       = 5000
alpha_b = 0.05

# Exponential(1) data: true mean=1, true median=ln(2)≈0.693
data_boot = np.random.exponential(1, n_boot)
true_mean   = 1.0
true_median = np.log(2)
true_var    = 1.0

# Bootstrap
boot_means   = np.array([np.random.choice(data_boot, n_boot, replace=True).mean()
                         for _ in range(B)])
boot_medians = np.array([np.median(np.random.choice(data_boot, n_boot, replace=True))
                         for _ in range(B)])
boot_vars    = np.array([np.random.choice(data_boot, n_boot, replace=True).var()
                         for _ in range(B)])

def ci_percentile(boots, alpha):
    return np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])

# Normal CI (using SE from CLT)
se_mean   = data_boot.std(ddof=1) / np.sqrt(n_boot)
t_crit    = stats.t.ppf(1 - alpha_b/2, df=n_boot-1)
ci_normal = data_boot.mean() + np.array([-1, 1]) * t_crit * se_mean

stats_tab = [
    ("Mean",   data_boot.mean(),     true_mean,   boot_means,   ci_normal),
    ("Median", np.median(data_boot), true_median, boot_medians, None),
    ("Var",    data_boot.var(),      true_var,    boot_vars,    None),
]

print(f"  Exponential(1) data, n={n_boot}, B={B} bootstrap samples")
print()
print(f"  {'Stat':>8} | {'Estimate':>10} | {'True':>8} | {'Boot CI (95%)':>22} | {'Normal CI (95%)':>22}")
print(f"  {'─'*78}")
for name, est, true_v, boots, norm_ci in stats_tab:
    bci = ci_percentile(boots, alpha_b)
    norm_str = f"[{norm_ci[0]:.4f}, {norm_ci[1]:.4f}]" if norm_ci is not None else "   N/A"
    print(f"  {name:>8} | {est:10.5f} | {true_v:8.5f} | "
          f"[{bci[0]:.4f}, {bci[1]:.4f}] | {norm_str}")
print()
print("  Bootstrap CI for median: no formula needed — works automatically.")
print()

# ── Part 3: KL Divergence ─────────────────────────────────────────────────
print("  PART 3 — KL DIVERGENCE BETWEEN GAUSSIANS")
print()
print("  D_KL(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁²+(μ₁−μ₂)²)/(2σ₂²) − ½")
print()

def kl_gaussian(mu1, s1, mu2, s2):
    return np.log(s2/s1) + (s1**2 + (mu1-mu2)**2) / (2*s2**2) - 0.5

print(f"  {'(μ₁,σ₁) vs (μ₂,σ₂)':>30} | {'D_KL(p‖q)':>12} | {'D_KL(q‖p)':>12} | {'Symmetric?':>12}")
print(f"  {'─'*75}")
cases = [
    (0, 1, 0, 1),     # identical
    (1, 1, 0, 1),     # mean shift
    (0, 2, 0, 1),     # variance ratio
    (2, 1, 0, 2),     # both shifted
    (0, 0.5, 0, 2),   # narrow vs wide
]
for mu1, s1, mu2, s2 in cases:
    kl_fwd = kl_gaussian(mu1, s1, mu2, s2)
    kl_rev = kl_gaussian(mu2, s2, mu1, s1)
    sym = "YES" if abs(kl_fwd - kl_rev) < 1e-8 else "NO"
    label = f"({mu1},{s1}) vs ({mu2},{s2})"
    print(f"  {label:>30} | {kl_fwd:12.5f} | {kl_rev:12.5f} | {sym:>12}")
print()
print("  KL is always non-negative. Asymmetry is fundamental — D_KL(p‖q) ≠ D_KL(q‖p).")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Bootstrap, CLT & KL Divergence", fontsize=12, fontweight="bold")

# Plot 1: CLT convergence for Exponential
x_norm = np.linspace(-4, 4, 400)
palette = ["tomato", "orange", "steelblue"]
for n, col in zip([5, 30, 100], palette):
    axes[0].hist(clt_data["Exponential(1)"][n], bins=60, density=True,
                 alpha=0.35, color=col, label=f"n={n}")
    axes[0].plot(x_norm, stats.norm.pdf(x_norm), "k--", lw=2, alpha=0.6)
axes[0].plot([], [], "k--", lw=2, label="N(0,1)")
axes[0].set_xlabel("Standardised X̄ₙ"); axes[0].set_ylabel("Density")
axes[0].set_title("CLT: Standardised X̄ₙ for Exponential(1)\\nConverges to N(0,1)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_xlim(-4, 4)

# Plot 2: Bootstrap distribution of mean and median
axes[1].hist(boot_means, bins=60, color="steelblue", alpha=0.6, density=True, label="Boot. Mean")
axes[1].hist(boot_medians, bins=60, color="tomato", alpha=0.6, density=True, label="Boot. Median")
axes[1].axvline(true_mean,   color="steelblue", lw=2, linestyle="--", label=f"True mean={true_mean}")
axes[1].axvline(true_median, color="tomato",    lw=2, linestyle="--", label=f"True median={true_median:.3f}")
bci_m = ci_percentile(boot_means, alpha_b)
axes[1].axvspan(*bci_m, alpha=0.08, color="steelblue")
axes[1].set_xlabel("Statistic value"); axes[1].set_ylabel("Density")
axes[1].set_title(f"Bootstrap Distributions (n={n_boot})\\nMean and Median of Exp(1)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# Plot 3: KL divergence asymmetry — forward vs reverse
x_kl = np.linspace(-6, 8, 500)
mu_p, sig_p = 2.0, 0.8   # p = narrow, shifted
mu_q, sig_q = 0.0, 2.0   # q = broad
p_pdf = stats.norm.pdf(x_kl, mu_p, sig_p)
q_pdf = stats.norm.pdf(x_kl, mu_q, sig_q)
axes[2].plot(x_kl, p_pdf, "steelblue", lw=2.5, label=f"p = N({mu_p},{sig_p}²)")
axes[2].plot(x_kl, q_pdf, "tomato",    lw=2.5, label=f"q = N({mu_q},{sig_q}²)")
kl_pq = kl_gaussian(mu_p, sig_p, mu_q, sig_q)
kl_qp = kl_gaussian(mu_q, sig_q, mu_p, sig_p)
axes[2].fill_between(x_kl, 0, np.minimum(p_pdf, q_pdf), alpha=0.12, color="purple",
                     label="Overlap region")
axes[2].set_xlabel("x"); axes[2].set_ylabel("Density")
axes[2].set_title(
    f"KL Divergence Asymmetry\\n"
    f"D_KL(p‖q)={kl_pq:.3f}, D_KL(q‖p)={kl_qp:.3f}"
)
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bootstrap_clt_kl.png", dpi=120)
print("  Plot saved → bootstrap_clt_kl.png")
print()
print("  KEY TAKEAWAYS:")
print("  CLT: X̄ₙ approaches N(0,1) for all finite-variance distributions.")
print("  Bootstrap: provides valid CIs for ANY smooth statistic without")
print("             distributional assumptions on the estimator.")
print("  KL divergence: D_KL(p‖q) ≠ D_KL(q‖p).  Forward KL (used in VI)")
print("  is mean-seeking; reverse KL is mode-seeking.  Both ≥ 0.")
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
# Run:  python 03_Statistical_inference.py
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