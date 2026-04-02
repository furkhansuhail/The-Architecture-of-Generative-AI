"""
Probability Foundations & Distributions
=========================================

The mathematical bedrock of machine learning. Every loss function,
every metric, every probabilistic model rests on these foundations.
Understanding distributions, Bayes' theorem, and MLE/MAP lets you
reason from first principles rather than memorising formulas.

"""

import textwrap
import re

TOPIC_NAME   = "Probability Foundations & Distributions"
DISPLAY_NAME = "00 · Probability & Distributions"
ICON         = "🎲"
SUBTITLE     = "The Mathematical Bedrock of Every ML Algorithm"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — PROBABILITY AXIOMS

### Core Definitions

    Sample space Ω:  the set of ALL possible outcomes
    Event A:         a subset of Ω (a collection of outcomes)
    P(A):            probability of event A — a number in [0, 1]

Kolmogorov's Three Axioms (everything else follows from these):

    1. Non-negativity:   P(A) ≥ 0  for all A
    2. Normalisation:    P(Ω) = 1  (something must happen)
    3. Additivity:       if A ∩ B = ∅, then P(A ∪ B) = P(A) + P(B)

Key rules derived from the axioms:

    Complement:          P(Aᶜ) = 1 − P(A)
    Union (general):     P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
    Conditional:         P(A | B) = P(A ∩ B) / P(B)     [B is the new universe]
    Independence:        P(A ∩ B) = P(A) × P(B)          iff A, B independent
    Law of total prob:   P(A) = Σₖ P(A | Bₖ) × P(Bₖ)   [partition of Ω]


### PART 2 — BAYES' THEOREM

### Derivation

From the definition of conditional probability:
    P(A | B) = P(A ∩ B) / P(B)
    P(B | A) = P(A ∩ B) / P(A)

Equating both expressions for P(A ∩ B):

    P(A | B) × P(B) = P(B | A) × P(A)

    ┌───────────────────────────────────────────────────────────┐
    │                                                           │
    │   P(A | B)  =  P(B | A) × P(A)  /  P(B)                   │
    │                                                           │
    │   Posterior = Likelihood × Prior / Evidence               │
    │                                                           │
    └───────────────────────────────────────────────────────────┘

In ML terms — A = hypothesis (model parameters θ), B = data D:

    P(θ | D) = P(D | θ) × P(θ) / P(D)

    Posterior = Likelihood × Prior / Normalising constant


### The Base Rate Fallacy (classic interview question)

A disease affects 1% of the population. A test has:
    - Sensitivity (true positive rate):   99%  →  P(+ | disease)  = 0.99
    - Specificity (true negative rate):   95%  →  P(- | no disease) = 0.95
    - False positive rate:                5%   →  P(+ | no disease) = 0.05

You test positive. What is the probability you have the disease?

    Most people guess ~95%. The correct answer is ~16.7%.

    Diagram 1 — Bayes Applied to Medical Testing:

    Population of 10,000:

    ┌───────────────────────────────────────────────────────────┐
    │  Disease (100 people = 1%)      No Disease (9,900 = 99%)  │
    │                                                           │
    │  Test+ : 99 people              Test+ : 495 people        │
    │  (true positives, 99%)          (false positives, 5%)     │
    │                                                           │
    │  Test- : 1 person               Test- : 9,405 people      │
    └───────────────────────────────────────────────────────────┘

    Of all Test+ results: 99 + 495 = 594 total
    P(disease | test+) = 99 / 594 ≈ 0.167 = 16.7%

    The low prevalence (1%) OVERWHELMS the high accuracy (99%).
    The huge pool of healthy people generates many more false positives
    than the small diseased pool generates true positives.

    Using Bayes formally:
    P(D+ | T+) = P(T+ | D+) × P(D+) / P(T+)
               = 0.99 × 0.01 / (0.99×0.01 + 0.05×0.99)
               = 0.0099 / (0.0099 + 0.0495)
               ≈ 0.167


### PART 3 — RANDOM VARIABLES & SUMMARY STATISTICS

### Discrete vs Continuous

    Discrete RV:    countable outcomes  (coin flip, word count)
                    Described by PMF (Probability Mass Function)
                    P(X = x) ≥ 0,   Σₓ P(X = x) = 1

    Continuous RV:  uncountable outcomes (height, temperature)
                    Described by PDF (Probability Density Function)
                    f(x) ≥ 0,   ∫ f(x)dx = 1
                    P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx    [area under curve]

    CDF (Cumulative Distribution Function) — works for both:
                    F(x) = P(X ≤ x)
                    F(−∞) = 0,   F(+∞) = 1,   always non-decreasing


### Expected Value and Variance

    E[X] (mean):     Discrete:   Σₓ x · P(X=x)
                     Continuous: ∫ x · f(x)dx

    Var(X):          E[(X − E[X])²] = E[X²] − (E[X])²
                     "Average squared distance from the mean"

    Std(X):          √Var(X)   — same units as X

    Key properties:
        E[aX + b]    = a·E[X] + b              (linearity)
        Var(aX + b)  = a²·Var(X)               (scale only)
        Var(X + Y)   = Var(X) + Var(Y) + 2·Cov(X,Y)


### Covariance and Correlation

    Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)] = E[XY] − E[X]E[Y]

        > 0: X and Y tend to increase together
        < 0: one increases while the other decreases
        = 0: linearly uncorrelated (but could still be dependent!)

    Pearson correlation:
        ρ(X,Y) = Cov(X,Y) / (σₓ · σᵧ)    ∈ [−1, +1]

        ρ = +1: perfect positive linear relationship
        ρ = −1: perfect negative linear relationship
        ρ =  0: no linear relationship (could be non-linear!)

    CRITICAL: correlation ≠ causation, and correlation = 0 ≠ independence
    (Anscombe's quartet: 4 datasets, same mean/variance/correlation, VERY different shapes)


### PART 4 — KEY PROBABILITY DISTRIBUTIONS

### Discrete Distributions

**Bernoulli(p)**
    Models: one coin flip, one binary event
    PMF:    P(X=1) = p,   P(X=0) = 1−p
    Mean:   p      Var: p(1−p)
    Use in ML: binary classification output, each independent label

**Binomial(n, p)**
    Models: number of successes in n independent Bernoulli trials
    PMF:    P(X=k) = C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ
    Mean:   np     Var: np(1−p)
    As n→∞: approaches Normal(np, np(1−p)) [by CLT]

**Poisson(λ)**
    Models: number of rare events in a fixed interval (λ = expected count)
    PMF:    P(X=k) = e⁻λ · λᵏ / k!
    Mean:   λ      Var: λ       (mean = variance — a useful property!)
    Use in ML: modelling rare events, count data regression, NLP word counts
    Note: Poisson(λ) ≈ Binomial(n, λ/n) for large n, small p

**Geometric(p)**
    Models: number of trials until first success
    PMF:    P(X=k) = (1−p)ᵏ⁻¹ · p
    Mean:   1/p    Var: (1−p)/p²
    Memoryless property: P(X>m+n | X>m) = P(X>n) — past failures don't help


### Continuous Distributions

**Uniform(a, b)**
    PDF:    f(x) = 1/(b−a)   for x ∈ [a, b]
    Mean:   (a+b)/2   Var: (b−a)²/12
    Use in ML: random initialisation, random search, sampling

**Normal / Gaussian N(μ, σ²)**
    PDF:    f(x) = (1/σ√2π) · exp(−(x−μ)²/2σ²)
    Mean:   μ     Var: σ²
    Standard Normal: N(0, 1)   [z = (x−μ)/σ is the standardisation]

    The 68-95-99.7 Rule:
    ┌─────────────────────────────┐
    │  P(|X − μ| < 1σ) ≈ 68.27%   │
    │  P(|X − μ| < 2σ) ≈ 95.45%   │
    │  P(|X − μ| < 3σ) ≈ 99.73%   │
    └─────────────────────────────┘

    Use in ML: weight initialisation assumption, noise modelling,
    the MLE of a Gaussian's mean is the sample mean.

**Exponential(λ)**
    PDF:    f(x) = λ·e⁻λˣ   for x ≥ 0
    Mean:   1/λ    Var: 1/λ²
    Memoryless: P(X > s+t | X > s) = P(X > t)
    Use in ML: modelling time-to-event, Lasso prior (Laplace distribution)

**Beta(α, β)**
    PDF:    f(x) ∝ xᵅ⁻¹(1−x)^(β−1)   for x ∈ [0, 1]
    Mean:   α/(α+β)   Mode: (α−1)/(α+β−2)  for α,β > 1
    Use in ML: prior for probabilities (conjugate prior for Bernoulli/Binomial)
    Special cases: Beta(1,1) = Uniform(0,1)
                   α>1, β>1: bell-shaped
                   α<1, β<1: U-shaped (mass at extremes)

    Diagram 2 — Beta Distribution shapes:

    f(x) │
         │  α=0.5,β=0.5   U-shape (mass at 0 and 1)
         │╲               ╱
         │  ╲           ╱
         │   ──────────    α=1,β=1 (Uniform — flat)
         │       ╭──╮      α=5,β=2 (right-skewed bell)
         │      ╱    ╲
         └──────────────── x ∈ [0,1]

**Dirichlet(α₁, ..., αₖ)**
    Generalisation of Beta to K categories (K-dimensional simplex)
    Mean:   αᵢ / Σαⱼ for each category i
    Use in ML: prior over multinomial distributions (topic modelling, LDA)
    Concentration parameter: Σαᵢ controls how peaked vs flat the distribution is


### Distribution Cheat Sheet

    ┌─────────────────┬────────────────┬──────────┬──────────┬────────────────────┐
    │ Distribution    │ Parameters     │ Mean     │ Variance │ ML Use Case        │
    ├─────────────────┼────────────────┼──────────┼──────────┼────────────────────┤
    │ Bernoulli       │ p              │ p        │ p(1-p)   │ Binary output      │
    │ Binomial        │ n, p           │ np       │ np(1-p)  │ Count of successes │
    │ Poisson         │ λ              │ λ        │ λ        │ Count data         │
    │ Geometric       │ p              │ 1/p      │(1-p)/p²  │ Time to first hit  │
    │ Negative Binom  │ r, p           │ r/p      │r(1-p)/p² │ Overdispersed count│
    │ Uniform         │ a, b           │(a+b)/2   │(b-a)²/12 │ Random init        │
    │ Gaussian        │ μ, σ²          │ μ        │ σ²       │ Universal default  │
    │ Exponential     │ λ              │ 1/λ      │ 1/λ²     │ Time-to-event      │
    │ Beta            │ α, β           │α/(α+β)   │ complex  │ Prior for probs    │
    │ Dirichlet       │ α₁...αₖ         │αᵢ/Σα     │ complex  │ Prior for cats     │
    └─────────────────┴────────────────┴──────────┴──────────┴────────────────────┘


### PART 5 — CENTRAL LIMIT THEOREM & LAW OF LARGE NUMBERS

### Law of Large Numbers (LLN)

If X₁, X₂, ..., Xₙ are i.i.d. with mean μ:

    Sample mean X̄ₙ = (X₁ + X₂ + ... + Xₙ) / n  →  μ   as n → ∞

    Weak LLN: X̄ₙ → μ in probability
    Strong LLN: X̄ₙ → μ almost surely

    Implication: with enough data, sample averages converge to
    true expectations. This is why ML works — training on enough
    examples approximates the true population distribution.


### Central Limit Theorem (CLT)

    If X₁, ..., Xₙ are i.i.d. with mean μ and variance σ², then:

    √n · (X̄ₙ − μ) / σ  →  N(0, 1)   as n → ∞

    Equivalently:   X̄ₙ ~ N(μ, σ²/n)   approximately for large n

    This holds regardless of the underlying distribution of Xᵢ!

    Diagram 3 — CLT in Action (rolling a very non-Gaussian die):

    Parent dist (Uniform 1-6):   Sample mean of n=1 rolls:
    █ █ █ █ █ █  (flat)          █ █ █ █ █ █  (still flat)

    Sample mean of n=5 rolls:    Sample mean of n=30 rolls:
       ██ ███ ██                       ╭──────╮
      ████████████                  ╭──╯      ╰──╮  (bell-shaped!)
    █████████████████             ────────────────────

    Why CLT matters for ML:
    1. Justifies treating prediction errors as Gaussian → MSE loss
    2. Enables confidence intervals and hypothesis testing
    3. Explains why SGD noise has beneficial annealing properties
    4. Validates the Gaussian assumption in many probabilistic models


### PART 6 — MLE AND MAP ESTIMATION

### Maximum Likelihood Estimation (MLE)

Given observed data D = {x₁, ..., xₙ} and a model with parameters θ:

    Likelihood:      L(θ) = P(D | θ) = ∏ᵢ P(xᵢ | θ)   [assuming i.i.d.]
    Log-likelihood:  ℓ(θ) = log L(θ) = Σᵢ log P(xᵢ | θ)

    MLE estimate:    θ̂_MLE = argmax_θ ℓ(θ)

    We maximise log-likelihood (not likelihood) because:
    1. log converts products to sums → easier calculus
    2. Avoids numerical underflow from multiplying many small probs
    3. log is monotone → same maximum as original

    Worked Example — MLE for a Gaussian:

        Data: x₁, ..., xₙ ~ N(μ, σ²),   find θ̂ = (μ̂, σ̂²)

        ℓ(μ, σ²) = Σᵢ [−½log(2πσ²) − (xᵢ−μ)²/(2σ²)]

        ∂ℓ/∂μ = 0  →  μ̂ = (1/n)Σxᵢ = x̄    [the sample mean!]
        ∂ℓ/∂σ² = 0 →  σ̂² = (1/n)Σ(xᵢ−μ̂)² [the biased sample variance]

        Result: MLE for Gaussian mean = sample mean. Elegant and intuitive.

    Connection to loss functions:
    • MLE under Gaussian noise   → minimising MSE
    • MLE under Bernoulli model  → minimising BCE
    • MLE under Laplace noise    → minimising MAE


### Maximum A Posteriori (MAP)

MAP incorporates a prior belief P(θ) about the parameters:

    θ̂_MAP = argmax_θ  log P(θ | D)
           = argmax_θ  [log P(D | θ) + log P(θ)]
           = argmax_θ  [ℓ(θ) + log P(θ)]
             ↑              ↑         ↑
           MAP =        MLE term + Prior term (regularisation!)

    Connection to regularisation:
    • Gaussian prior N(0, τ²) on θ:   log P(θ) = −θ²/(2τ²) + const
      → MAP = MLE − λΣwᵢ²   =  L2 regularisation (Ridge)!

    • Laplace prior Laplace(0, b) on θ: log P(θ) = −|θ|/b + const
      → MAP = MLE − λΣ|wᵢ|   =  L1 regularisation (Lasso)!

    ┌───────────────────────────────────────────────────────────┐
    │  Regularisation is Bayesian inference in disguise.        │
    │                                                           │
    │  L2 regularisation  ≡  Gaussian prior on weights          │
    │  L1 regularisation  ≡  Laplace prior on weights           │
    │                                                           │
    │  As n → ∞:  data overwhelms prior → MAP → MLE             │
    │  Small n:   prior dominates → MAP ≠ MLE (regularised)     │
    └───────────────────────────────────────────────────────────┘


### PART 7 — JOINT, MARGINAL & CONDITIONAL DISTRIBUTIONS

### Joint Distribution

The joint distribution P(X, Y) describes the probability of two (or more)
random variables taking specific values simultaneously.

    Discrete:    P(X=x, Y=y) ≥ 0,    ΣₓΣᵧ P(X=x, Y=y) = 1
    Continuous:  f(x, y) ≥ 0,        ∫∫ f(x, y) dx dy = 1

    Joint CDF:   F(x, y) = P(X ≤ x, Y ≤ y)


### Marginal Distribution

Obtained by summing (or integrating) out the other variable — "collapsing"
the joint table along one axis:

    Discrete:    P(X=x) = Σᵧ P(X=x, Y=y)           [sum over all y]
    Continuous:  f_X(x) = ∫ f(x, y) dy             [integrate out y]

    Diagram — Joint Table (discrete example):

          Y=0   Y=1   Y=2   │ P(X)    ← marginal of X
    X=0  0.10  0.05  0.05   │ 0.20
    X=1  0.15  0.20  0.05   │ 0.40
    X=2  0.05  0.15  0.20   │ 0.40
    ─────────────────────────
    P(Y) 0.30  0.40  0.30   ← marginal of Y


### Conditional Distribution

    P(X=x | Y=y) = P(X=x, Y=y) / P(Y=y)     [Bayes denominator revealed]

    For continuous variables:
        f(x | y) = f(x, y) / f_Y(y)

    Independence check:
        X ⊥ Y  ⟺  P(X, Y) = P(X) · P(Y)   for all x, y
                ⟺  f(x | y) = f(x)   for all x, y


### Chain Rule of Probability

Any joint distribution factorises as:

    P(X₁, X₂, ..., Xₙ) = P(X₁) · P(X₂|X₁) · P(X₃|X₁,X₂) · ... · P(Xₙ|X₁,...,Xₙ₋₁)

    Naive Bayes simplification — assume conditional independence given Y:

        P(X₁,...,Xₙ | Y) = ∏ᵢ P(Xᵢ | Y)

    This collapses an exponential joint into n simple conditionals —
    the core assumption behind the Naive Bayes classifier.


### Multivariate Gaussian (the workhorse of ML)

    X ~ N(μ, Σ)   where μ ∈ ℝᵈ (mean vector), Σ ∈ ℝᵈˣᵈ (covariance matrix)

    PDF:   f(x) = (2π)^(-d/2) |Σ|^(-½) exp(-½ (x−μ)ᵀ Σ⁻¹ (x−μ))

    Key properties:
        Σ must be symmetric and positive semi-definite
        Diagonal Σ → independent dimensions
        Σ = σ²I  → spherical (isotropic) Gaussian

    Mahalanobis distance:  (x−μ)ᵀ Σ⁻¹ (x−μ)  — accounts for correlations
    Marginals:             any subset of dims is also Gaussian
    Conditionals:          X₁ | X₂ = x₂  is Gaussian (with updated μ and Σ)

    Diagram 4 — Covariance matrix shapes:

    Σ = [[1, 0],   Σ = [[1, 0.9],  Σ = [[3, 0],
         [0, 1]]        [0.9, 1]]       [0, 0.5]]

       · · · · ·       ·  ·  ·           · · · ·
      · · · · · ·     · ·  · ·         · · · · · ·
     · · · · · · ·   · · ·· · ·       · · · · · · ·
      · · · · · ·     · ·  · ·         · · · · · ·
       · · · · ·       ·  ·  ·           · · · ·
    (sphere)         (tilted ellipse)  (axis-aligned ellipse)

    ML uses: PCA (eigen-decomposition of Σ), Gaussian Processes,
             Linear Discriminant Analysis, Kalman Filters, GMMs.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — INFORMATION THEORY: ENTROPY & KL DIVERGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Shannon Entropy H(X)

Measures the average uncertainty (or information content) of a distribution.
Units: bits (log₂) or nats (lnₑ).

    H(X) = −Σₓ P(X=x) log P(X=x)   [discrete]
    H(X) = −∫ f(x) log f(x) dx      [continuous — called differential entropy]

    Intuition:
        Certain outcome (P=1):        H = 0         [no surprise]
        Fair coin (P=0.5, 0.5):       H = 1 bit     [maximum uncertainty for 2 outcomes]
        Fair 8-sided die:             H = 3 bits     [log₂(8)]
        Uniform over n outcomes:      H = log₂(n)   [maximum entropy]

    Diagram 5 — Entropy of a Bernoulli(p):

         H(p) │    ╭────╮
          1.0 │   ╱      ╲
              │  ╱        ╲
          0.5 │ ╱          ╲
              │╱            ╲
          0.0 ┼──────────────── p
              0     0.5     1.0

    Maximum at p=0.5 (maximum uncertainty), zero at p=0 or p=1 (certainty).

### Cross-Entropy H(P, Q)

Measures the average bits needed to encode samples from P using a code
optimised for Q:

    H(P, Q) = −Σₓ P(x) log Q(x)

    Decomposition:
        H(P, Q) = H(P) + KL(P ‖ Q)

    In ML, when P = true labels (one-hot), H(P) = 0, so:
        Cross-entropy loss = KL divergence from predictions to truth
        Minimising cross-entropy ≡ minimising KL ≡ maximising log-likelihood!

    Binary cross-entropy (BCE):
        H(y, ŷ) = −[y log ŷ + (1−y) log(1−ŷ)]
        This IS the MLE objective for a Bernoulli model.


### KL Divergence KL(P ‖ Q)

Measures how much extra bits are needed when using Q instead of the true P.
Also called "relative entropy".

    KL(P ‖ Q) = Σₓ P(x) log [P(x) / Q(x)]   [discrete]
               = H(P, Q) − H(P)

    Properties:
        KL(P ‖ Q) ≥ 0            always (Gibbs' inequality)
        KL(P ‖ Q) = 0  iff P = Q  [zero only when distributions match]
        KL(P ‖ Q) ≠ KL(Q ‖ P)   asymmetric — NOT a true distance!

    Two modes of failure (KL is asymmetric):
        KL(P ‖ Q): "forward KL" — Q must cover all of P (mean-seeking)
        KL(Q ‖ P): "reverse KL" — Q avoids regions P is zero (mode-seeking)

    ┌───────────────────────────────────────────────────────────┐
    │  KL Divergence in ML:                                     │
    │                                                           │
    │  VAE loss         = Reconstruction loss + KL(q(z|x)‖p(z)) │
    │  Policy gradient  = KL(π_new ‖ π_old) as trust region     │
    │  Cross-entropy    = KL(true ‖ predicted) + H(true)        │
    │  Distillation     = KL(teacher ‖ student)                 │
    └───────────────────────────────────────────────────────────┘


### Mutual Information I(X; Y)

How much knowing Y reduces uncertainty about X:

    I(X; Y) = H(X) − H(X | Y) = H(Y) − H(Y | X)
             = KL(P(X,Y) ‖ P(X)·P(Y))

    Properties:
        I(X; Y) ≥ 0
        I(X; Y) = 0  iff X ⊥ Y  (independence!)
        I(X; Y) = I(Y; X)   [symmetric, unlike KL]

    Use in ML: feature selection (mutual information with the target Y),
    ICA (maximise statistical independence), information bottleneck.


### PART 9 — HYPOTHESIS TESTING & CONFIDENCE INTERVALS

### Framework

    H₀ (null hypothesis):      the boring default (no effect, no difference)
    H₁ (alternative):          what you hope to show

    Test statistic:    T = f(data) — a number computed from your sample
    p-value:           P(seeing data at least this extreme | H₀ is true)
    Significance level α: threshold for rejection (commonly 0.05)

    Decision rule:
        p ≤ α → reject H₀ (result is "statistically significant")
        p > α → fail to reject H₀ (not enough evidence)

    CRITICAL: p-value is NOT the probability H₀ is true!
              p-value is NOT the probability your result is a fluke!
              A low p-value only means the data are unlikely under H₀.


### Type I & Type II Errors

    ┌──────────────────┬─────────────────────┬─────────────────────┐
    │                  │  H₀ True            │  H₀ False (H₁ True) │
    ├──────────────────┼─────────────────────┼─────────────────────┤
    │ Reject H₀        │ Type I error (α)    │ Correct (Power=1−β) │
    │ Fail to reject   │ Correct (1−α)       │ Type II error (β)   │
    └──────────────────┴─────────────────────┴─────────────────────┘

    Power = 1 − β = P(reject H₀ | H₁ is true)
    To increase power: larger n, larger effect size, larger α


### Common Tests

    z-test:   known σ, large n.    T = (x̄ − μ₀) / (σ/√n) ~ N(0,1) under H₀
    t-test:   unknown σ, any n.    T = (x̄ − μ₀) / (s/√n) ~ t(n−1) under H₀
    χ²-test:  categorical data.    T = Σ (O−E)²/E ~ χ²(k−1) under H₀


### Confidence Intervals

A 95% CI does NOT mean "95% probability the true μ is in this interval."
It means: if we repeated the experiment many times, 95% of the constructed
intervals would contain the true μ.

    CI for mean (known σ): x̄ ± z_{α/2} · (σ/√n)
    CI for mean (unknown σ): x̄ ± t_{α/2, n−1} · (s/√n)

    95% CI uses z = 1.96 (from N(0,1):  P(|Z| ≤ 1.96) ≈ 0.95)


### Multiple Testing Problem

If you run 20 tests at α=0.05, you expect ~1 false positive by chance.
This is the "garden of forking paths" or "p-hacking" problem.

    Bonferroni correction: use α/m for m tests (conservative)
    Benjamini-Hochberg:    controls False Discovery Rate (FDR) — preferred in ML


### PART 10 — SAMPLING METHODS

### Why Sampling?

Many posteriors and expectations are intractable analytically.
Sampling lets us approximate them numerically:

    E[f(X)] ≈ (1/N) Σᵢ f(xᵢ)    where xᵢ ~ P(X)    [Monte Carlo estimate]

    Error shrinks at rate 1/√N regardless of dimension — this is the
    great advantage of Monte Carlo over grid-based integration.


### Monte Carlo Integration

    Goal: estimate  I = ∫ f(x) p(x) dx = E_p[f(X)]
    Method:
        1. Draw x₁, ..., xₙ ~ p(x)
        2. Estimate  Î = (1/N) Σᵢ f(xᵢ)
        3. By LLN, Î → I as N → ∞

    Classic use: estimating π by sampling from Uniform(−1,1)² and checking
    if points fall inside the unit circle.


### Importance Sampling

When we cannot sample from p(x) directly, use a proposal q(x):

    E_p[f(X)] = ∫ f(x) p(x)/q(x) · q(x) dx ≈ (1/N) Σᵢ f(xᵢ) · w(xᵢ)

    where w(xᵢ) = p(xᵢ)/q(xᵢ) are importance weights, xᵢ ~ q(x)

    Requirement: q(x) > 0 wherever p(x) > 0


### Rejection Sampling

To sample from p(x) using a proposal q(x) with p(x) ≤ M·q(x):

    1. Draw x ~ q(x)
    2. Draw u ~ Uniform(0, 1)
    3. Accept x if u ≤ p(x) / (M·q(x)),  else reject and repeat

    Acceptance rate = 1/M — works poorly in high dimensions.


### Markov Chain Monte Carlo (MCMC)

Constructs a Markov chain whose stationary distribution is p(x).
Run the chain long enough → samples ≈ draws from p(x).

    Metropolis-Hastings:
        1. Propose x* ~ q(x* | xₜ)
        2. Accept with probability:  α = min(1, p(x*)q(xₜ|x*) / p(xₜ)q(x*|xₜ))
        3. If accepted: xₜ₊₁ = x*,  else: xₜ₊₁ = xₜ

    Only requires p(x) up to a normalising constant (the denominator in Bayes
    is the hard part — MCMC sidesteps it entirely).

    Gibbs Sampling (special MCMC):
        Cycle through each dimension, sampling from its conditional:
        xᵢ ~ p(xᵢ | x₋ᵢ)
        Always accepted (no rejection step).

    Burn-in: early samples before the chain reaches stationarity — discard these.
    Thinning: keep every k-th sample to reduce autocorrelation.

    ┌───────────────────────────────────────────────────────────┐
    │  Sampling Methods Comparison:                             │
    │                                                           │
    │  Monte Carlo:    simple, needs direct sampling from p     │
    │  Importance:     reweights samples from easier q          │
    │  Rejection:      exact but inefficient in high-d          │
    │  MCMC:           scales to complex posteriors; correlated │
    │  HMC/NUTS:       gradient-based MCMC (used in PyMC, Stan) │
    └───────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Distribution Gallery — PDF, PMF & CDF for All Key Distributions": {
        "description": (
            "Plot every major distribution side-by-side: "
            "Bernoulli, Binomial, Poisson, Gaussian, Exponential, Beta. "
            "Show both PDF/PMF and CDF. Print mean, variance, and key properties."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 65)
print("  DISTRIBUTION GALLERY: KEY PROPERTIES")
print("=" * 65)
print()

# ── Define all distributions ──────────────────────────────────────────────
distributions = [
    # (name, scipy_dist, params_dict, x_range, is_discrete, colour)
    ("Bernoulli(p=0.3)",   stats.bernoulli, {"p": 0.3},
     [0, 1], True, "steelblue"),
    ("Binomial(n=10,p=0.3)", stats.binom, {"n": 10, "p": 0.3},
     np.arange(0, 11), True, "seagreen"),
    ("Poisson(λ=3)",       stats.poisson,  {"mu": 3},
     np.arange(0, 13), True, "tomato"),
    ("Gaussian(μ=0,σ=1)",  stats.norm,     {"loc": 0, "scale": 1},
     np.linspace(-4, 4, 300), False, "purple"),
    ("Exponential(λ=1)",   stats.expon,    {"scale": 1},
     np.linspace(0, 6, 300), False, "orange"),
    ("Beta(α=2,β=5)",      stats.beta,     {"a": 2, "b": 5},
     np.linspace(0, 1, 300), False, "crimson"),
]

# ── Print properties table ─────────────────────────────────────────────────
print(f"  {'Distribution':26s} | {'Mean':>8} | {'Variance':>10} | Key property")
print(f"  {'─'*72}")
for name, dist, params, x_range, is_disc, colour in distributions:
    d      = dist(**params)
    mean   = d.mean()
    var    = d.var()
    props  = {
        "Bernoulli(p=0.3)":     "Binary outcome, foundation of BCE",
        "Binomial(n=10,p=0.3)": "Sum of n Bernoullis; → Normal by CLT",
        "Poisson(λ=3)":         "Rare events; mean=variance=λ",
        "Gaussian(μ=0,σ=1)":    "68-95-99.7 rule; MLE→MSE",
        "Exponential(λ=1)":     "Memoryless; Lasso uses Laplace≈2×Exp",
        "Beta(α=2,β=5)":        "Prior for probabilities; conjugate to Bernoulli",
    }
    print(f"  {name:26s} | {mean:8.3f} | {var:10.4f} | {props.get(name,'')}")

print()

# ── Plot PDF/PMF and CDF ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(distributions), figsize=(22, 7))
fig.suptitle("Distribution Gallery: PDF/PMF (top) and CDF (bottom)",
             fontsize=12, fontweight="bold")

for col, (name, dist_cls, params, x_range, is_disc, colour) in enumerate(distributions):
    d    = dist_cls(**params)
    x    = np.array(x_range, dtype=float)
    ax_t = axes[0, col]   # PDF / PMF
    ax_b = axes[1, col]   # CDF

    if is_disc:
        pmf_vals = d.pmf(x)
        ax_t.bar(x, pmf_vals, color=colour, alpha=0.8, width=0.6)
        ax_t.set_ylabel("P(X=k)" if col == 0 else "")
    else:
        pdf_vals = d.pdf(x)
        ax_t.plot(x, pdf_vals, colour, lw=2.5)
        ax_t.fill_between(x, pdf_vals, alpha=0.15, color=colour)
        ax_t.set_ylabel("f(x)" if col == 0 else "")

    cdf_vals = d.cdf(x)
    ax_b.plot(x, cdf_vals, colour, lw=2.5)
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax_b.set_ylabel("F(x) = P(X≤x)" if col == 0 else "")

    ax_t.set_title(name, fontsize=8.5, fontweight="bold")
    ax_t.grid(alpha=0.3)
    ax_b.grid(alpha=0.3)
    ax_b.set_xlabel("x")

    # Annotate mean
    mean_val = d.mean()
    ax_t.axvline(mean_val, color="black", linestyle="--", lw=1.2, alpha=0.7,
                 label=f"mean={mean_val:.2f}")
    ax_t.legend(fontsize=7)

plt.tight_layout()
plt.savefig("distribution_gallery.png", dpi=110)
print("  Plot saved → distribution_gallery.png")
print()
print("  KEY INTERVIEW FACTS:")
print("  Bernoulli: foundation of binary classification (BCE loss)")
print("  Poisson:   unique — its mean equals its variance (λ = λ)")
print("  Gaussian:  MLE of mean = sample mean; justified by CLT")
print("  Exponential: only continuous memoryless distribution")
print("  Beta:      flexible prior for probabilities in [0,1]")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Bayes' Theorem — Medical Test & Base Rate Fallacy": {
        "description": (
            "Solve the classic medical testing problem step-by-step. "
            "Show how the base rate (disease prevalence) dominates the posterior. "
            "Then sweep prevalence from 0.1% to 50% and plot posterior probability."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 65)
print("  BAYES' THEOREM — MEDICAL TEST & BASE RATE FALLACY")
print("=" * 65)
print()

# ── Core function ──────────────────────────────────────────────────────────
def bayes_medical(prevalence, sensitivity, specificity):
    """
    P(disease | positive test)  using Bayes' theorem.

    sensitivity = P(+ | disease)     true positive rate
    specificity = P(- | no disease)  true negative rate
    fpr = 1 - specificity = P(+ | no disease)  false positive rate
    """
    p_disease   = prevalence
    p_no_disease = 1 - prevalence
    fpr          = 1 - specificity

    # Law of total probability for P(+):
    p_pos = sensitivity * p_disease + fpr * p_no_disease

    # Bayes:
    p_disease_given_pos = (sensitivity * p_disease) / p_pos
    return p_disease_given_pos, p_pos

# ── Classic problem ───────────────────────────────────────────────────────
SENS        = 0.99    # P(+ | disease)
SPEC        = 0.95    # P(- | no disease)
PREVALENCE  = 0.01    # 1% of population has the disease

posterior, p_pos = bayes_medical(PREVALENCE, SENS, SPEC)

print("  THE PROBLEM:")
print(f"    Disease prevalence:    {PREVALENCE*100:.1f}%")
print(f"    Test sensitivity:      {SENS*100:.1f}%   [P(+ | disease)]")
print(f"    Test specificity:      {SPEC*100:.1f}%   [P(- | no disease)]")
print(f"    False positive rate:   {(1-SPEC)*100:.1f}%   [P(+ | no disease)]")
print()
print("  STEP-BY-STEP BAYES:")
print()

N = 10_000
n_disease    = int(N * PREVALENCE)
n_no_disease = N - n_disease
tp = round(n_disease    * SENS)
fn = n_disease - tp
fp = round(n_no_disease * (1 - SPEC))
tn = n_no_disease - fp

print(f"  In a population of {N:,} people:")
print(f"    Have disease:          {n_disease:5,}  ({PREVALENCE*100:.1f}%)")
print(f"    Do not have disease:   {n_no_disease:5,}  ({(1-PREVALENCE)*100:.1f}%)")
print()
print(f"  Test results:")
print(f"    True positives  (TP):  {tp:5,}  (disease and test +)")
print(f"    False negatives (FN):  {fn:5,}  (disease but test -)")
print(f"    False positives (FP):  {fp:5,}  (no disease but test +)")
print(f"    True negatives  (TN):  {tn:5,}  (no disease and test -)")
print()
print(f"  Of ALL positive tests ({tp+fp:,} total):")
print(f"    True positives:   {tp:,}  ({tp/(tp+fp)*100:.1f}%)")
print(f"    False positives: {fp:,}  ({fp/(tp+fp)*100:.1f}%)")
print()
print(f"  P(disease | test +) = {tp}/{tp+fp} = {tp/(tp+fp)*100:.1f}%")
print()
print(f"  Bayes formula:  P(D+|T+) = P(T+|D+) x P(D+) / P(T+)")
print(f"    = {SENS} x {PREVALENCE} / {p_pos:.4f}")
print(f"    = {posterior*100:.1f}%")
print()
print(f"  Most people guess ~95%. The correct answer is ~{posterior*100:.0f}%!")
print(f"  The low prevalence ({PREVALENCE*100}%) overwhelms the high accuracy ({SENS*100}%).")
print()

# ── Sweep prevalence ───────────────────────────────────────────────────────
prevalences = np.logspace(-3, np.log10(0.5), 300)
posteriors  = [bayes_medical(p, SENS, SPEC)[0] for p in prevalences]

print("  POSTERIOR vs PREVALENCE:")
print(f"  {'Prevalence':>12} | {'P(disease | +)':>16} | {'Naive guess':>12}")
print(f"  {'─'*47}")
for prev in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
    post, _ = bayes_medical(prev, SENS, SPEC)
    naive   = SENS   # most people think accuracy ≈ posterior
    print(f"  {prev*100:11.1f}% | {post*100:15.1f}% | {naive*100:11.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Bayes' Theorem: Disease Probability After Positive Test",
             fontsize=12, fontweight="bold")

axes[0].semilogx(prevalences * 100, np.array(posteriors) * 100,
                 "steelblue", lw=2.5)
axes[0].axvline(1, color="tomato", linestyle="--", lw=2,
                label="Prevalence = 1%")
axes[0].axhline(posterior * 100, color="tomato", linestyle=":",
                lw=1.5, label=f"Posterior = {posterior*100:.1f}%")
axes[0].axhline(SENS * 100, color="gray", linestyle="--", lw=1.5,
                label=f"Sensitivity = {SENS*100}% (naive guess)")
axes[0].fill_between(prevalences * 100, np.array(posteriors) * 100,
                     alpha=0.1, color="steelblue")
axes[0].set_xlabel("Disease Prevalence (%)")
axes[0].set_ylabel("P(disease | positive test) (%)")
axes[0].set_title("Posterior Probability vs Prevalence")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0.1, 50)

# Confusion matrix visualisation for our scenario
labels = [["True Pos (TP)", "False Neg (FN)"],
          ["False Pos (FP)", "True Neg (TN)"]]
vals   = [[tp, fn], [fp, tn]]
colours_mat = [["#2ecc71", "#e74c3c"], ["#e74c3c", "#2ecc71"]]

for i in range(2):
    for j in range(2):
        axes[1].text(j + 0.5, 1.5 - i, labels[i][j] + " " + str(vals[i][j]),
                     ha="center", va="center", fontsize=10,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor=colours_mat[i][j], alpha=0.5))

axes[1].set_xlim(0, 2)
axes[1].set_ylim(0, 2)
axes[1].set_xticks([0.5, 1.5])
axes[1].set_xticklabels(["Predicted +", "Predicted -"])
axes[1].set_yticks([0.5, 1.5])
axes[1].set_yticklabels(["No Disease", "Disease"])
axes[1].set_title(f"Confusion Matrix (N={N:,}, prevalence=1%)")
axes[1].grid(False)

plt.tight_layout()
plt.savefig("bayes_medical_test.png", dpi=120)
print()
print("  Plot saved → bayes_medical_test.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Central Limit Theorem — Live Demonstration": {
        "description": (
            "Demonstrate the CLT empirically using highly non-Gaussian parent "
            "distributions. Show the sample mean distribution converging "
            "to Gaussian as n grows, regardless of the parent shape."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("=" * 65)
print("  CENTRAL LIMIT THEOREM — EMPIRICAL DEMONSTRATION")
print("=" * 65)
print()
print("  Statement: If X_1,...,X_n are i.i.d. with mean μ and var σ²,")
print("  then X̄_n → N(μ, σ²/n) as n → ∞")
print("  This holds for ANY parent distribution (finite variance).")
print()

# ── Parent distributions to test ───────────────────────────────────────────
N_SAMPLES = 10_000   # number of experiments per n
n_values  = [1, 5, 30, 100]

parents = [
    ("Uniform(0,1)",    lambda s: np.random.uniform(0, 1, s),  0.5,   1/12),
    ("Exponential(1)",  lambda s: np.random.exponential(1, s), 1.0,   1.0),
    ("Bernoulli(0.2)",  lambda s: np.random.binomial(1, 0.2, s), 0.2, 0.2*0.8),
]

fig, axes = plt.subplots(len(parents), len(n_values), figsize=(16, 10))
fig.suptitle("Central Limit Theorem: Sample Mean Distribution Converges to Gaussian",
             fontsize=12, fontweight="bold")

for row, (pname, sampler, true_mean, true_var) in enumerate(parents):
    print(f"  Parent: {pname}  (μ={true_mean}, σ²={true_var:.4f})")

    for col, n in enumerate(n_values):
        # Draw N_SAMPLES independent samples of size n, compute mean of each
        sample_means = np.array([sampler(n).mean() for _ in range(N_SAMPLES)])

        expected_std = np.sqrt(true_var / n)
        obs_mean     = sample_means.mean()
        obs_std      = sample_means.std()

        # Normality test (Shapiro-Wilk on 500-sample subset)
        stat, pval = stats.shapiro(sample_means[:500])
        is_normal  = "Normal ✓" if pval > 0.05 else "Not Normal"

        ax = axes[row, col]
        ax.hist(sample_means, bins=50, density=True, color="steelblue",
                alpha=0.6, label="Sample means")

        # Overlay theoretical Gaussian
        x_range = np.linspace(sample_means.min(), sample_means.max(), 200)
        ax.plot(x_range,
                stats.norm.pdf(x_range, true_mean, expected_std),
                "tomato", lw=2.5, label="N(μ, σ²/n)")

        ax.set_title(f"n={n}  [{is_normal}]", fontsize=9, fontweight="bold")
        if col == 0:
            ax.set_ylabel(pname, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        print(f"    n={n:4d}: obs_mean={obs_mean:.3f} (true:{true_mean:.3f})  "
              f"obs_std={obs_std:.4f} (theory:{expected_std:.4f})  "
              f"Shapiro p={pval:.4f} [{is_normal}]")
    print()

plt.tight_layout()
plt.savefig("central_limit_theorem.png", dpi=110)
print("  Plot saved → central_limit_theorem.png")
print()
print("  KEY OBSERVATIONS:")
print("  - n=1: sample mean = one draw — same shape as parent")
print("  - n=5: starts to look bell-shaped")
print("  - n=30: empirical rule threshold — visually Gaussian")
print("  - n=100: nearly perfect Gaussian for ALL parent shapes")
print()
print("  The convergence speed depends on the parent's skewness:")
print("  Symmetric parents (Uniform) converge faster than skewed (Exponential).")
print()
print("  ML Implications:")
print("  1. Prediction errors tend to be Gaussian → justifies MSE loss")
print("  2. SGD noise is approximately Gaussian → enables analysis")
print("  3. Confidence intervals: X̄ ± z * σ/√n  (z=1.96 for 95%)")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · MLE from Scratch — Gaussian & Bernoulli": {
        "description": (
            "Derive and implement MLE analytically for Gaussian and Bernoulli. "
            "Verify the closed-form solutions match numerical optimisation. "
            "Show the log-likelihood surface for Gaussian MLE."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

np.random.seed(7)

print("=" * 65)
print("  MLE FROM SCRATCH: GAUSSIAN & BERNOULLI")
print("=" * 65)
print()

# ──────────────────────────────────────────────────────────────────────────
# PART 1: MLE for Gaussian
# ──────────────────────────────────────────────────────────────────────────
print("  PART 1 — MLE FOR GAUSSIAN N(μ, σ²)")
print()
print("  Log-likelihood: ℓ(μ,σ²) = -n/2·log(2πσ²) - Σ(xᵢ-μ)²/(2σ²)")
print("  Setting ∂ℓ/∂μ = 0:   μ̂ = (1/n)·Σxᵢ  =  sample mean")
print("  Setting ∂ℓ/∂σ² = 0:  σ̂² = (1/n)·Σ(xᵢ-μ̂)²  =  biased variance")
print()

# Generate data from a known Gaussian
TRUE_MU    = 3.0
TRUE_SIGMA = 1.5
n_samples  = [5, 20, 100, 1000]

print(f"  True parameters: μ={TRUE_MU}, σ={TRUE_SIGMA} (σ²={TRUE_SIGMA**2})")
print()
print(f"  {'n':>6} | {'μ̂ (MLE)':>12} | {'σ̂ (MLE)':>12} | {'μ error':>10} | {'σ error':>10}")
print(f"  {'─'*60}")

for n in n_samples:
    data     = np.random.normal(TRUE_MU, TRUE_SIGMA, n)
    mu_mle   = data.mean()                  # MLE mean = sample mean
    var_mle  = np.mean((data - mu_mle)**2)  # MLE variance (biased, divides by n)
    sigma_mle = np.sqrt(var_mle)

    mu_err    = abs(mu_mle - TRUE_MU)
    sigma_err = abs(sigma_mle - TRUE_SIGMA)
    print(f"  {n:6d} | {mu_mle:12.4f} | {sigma_mle:12.4f} | "
          f"{mu_err:10.4f} | {sigma_err:10.4f}")

print()
print("  NOTE: MLE σ² is BIASED (divides by n, not n-1).")
print("  Unbiased estimator (Bessel's correction): S² = Σ(xᵢ-x̄)²/(n-1)")
print("  For large n the difference is negligible.")
print()

# ── Log-likelihood surface for Gaussian ──────────────────────────────────
data_demo = np.random.normal(TRUE_MU, TRUE_SIGMA, 50)

def gaussian_log_likelihood(mu, sigma, data):
    n  = len(data)
    ll = -n/2 * np.log(2 * np.pi * sigma**2)
    ll -= np.sum((data - mu)**2) / (2 * sigma**2)
    return ll

mu_grid    = np.linspace(1, 5, 100)
sigma_grid = np.linspace(0.5, 3.0, 100)
MU, SIGMA  = np.meshgrid(mu_grid, sigma_grid)
LL         = np.vectorize(lambda m, s: gaussian_log_likelihood(m, s, data_demo))(MU, SIGMA)

# ──────────────────────────────────────────────────────────────────────────
# PART 2: MLE for Bernoulli
# ──────────────────────────────────────────────────────────────────────────
print("  PART 2 — MLE FOR BERNOULLI(p)")
print()
print("  Data: n coin flips, k heads.  Find p̂.")
print()
print("  Log-likelihood: ℓ(p) = k·log(p) + (n-k)·log(1-p)")
print("  Setting dℓ/dp = 0:   p̂ = k/n  =  fraction of heads")
print()

TRUE_P = 0.35
n_list = [10, 50, 200, 1000]

print(f"  True p = {TRUE_P}")
print()
print(f"  {'n':>6} | {'k (heads)':>10} | {'p̂ = k/n':>10} | {'Error':>10}")
print(f"  {'─'*45}")

for n_bern in n_list:
    data_bern  = np.random.binomial(1, TRUE_P, n_bern)
    k          = data_bern.sum()
    p_mle      = k / n_bern
    print(f"  {n_bern:6d} | {k:10d} | {p_mle:10.4f} | {abs(p_mle-TRUE_P):10.4f}")

print()

# ── Visualise log-likelihood curve for Bernoulli ──────────────────────────
data_bern_demo = np.random.binomial(1, TRUE_P, 30)
k_demo = data_bern_demo.sum()
n_demo = len(data_bern_demo)

p_vals = np.linspace(0.01, 0.99, 300)
ll_bern = k_demo * np.log(p_vals) + (n_demo - k_demo) * np.log(1 - p_vals)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("MLE: Log-Likelihood Surfaces and Curves",
             fontsize=12, fontweight="bold")

# Plot 1: Gaussian log-likelihood surface (contour)
contour = axes[0].contourf(MU, SIGMA, LL, levels=30, cmap="viridis")
plt.colorbar(contour, ax=axes[0])
axes[0].plot(data_demo.mean(), np.std(data_demo), "r*", ms=15,
             label=f"MLE: μ̂={data_demo.mean():.2f}, σ̂={np.std(data_demo):.2f}")
axes[0].axvline(TRUE_MU, color="white", linestyle="--", lw=1.5, label=f"True μ={TRUE_MU}")
axes[0].axhline(TRUE_SIGMA, color="cyan", linestyle="--", lw=1.5,
                label=f"True σ={TRUE_SIGMA}")
axes[0].set_xlabel("μ"); axes[0].set_ylabel("σ")
axes[0].set_title("Gaussian LL Surface (n=50)")
axes[0].legend(fontsize=8)

# Plot 2: Bernoulli log-likelihood curve
axes[1].plot(p_vals, ll_bern, "steelblue", lw=2.5)
axes[1].axvline(k_demo/n_demo, color="tomato", linestyle="--", lw=2,
                label=f"MLE: p̂={k_demo/n_demo:.3f} ({k_demo}/{n_demo})")
axes[1].axvline(TRUE_P, color="green", linestyle=":", lw=2,
                label=f"True p={TRUE_P}")
axes[1].set_xlabel("p"); axes[1].set_ylabel("Log-Likelihood ℓ(p)")
axes[1].set_title(f"Bernoulli LL Curve (n={n_demo}, k={k_demo})")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

# Plot 3: MLE convergence (Bernoulli)
ns     = np.arange(1, 500)
errors = []
np.random.seed(0)
data_long = np.random.binomial(1, TRUE_P, 500)
for nn in ns:
    errors.append(abs(data_long[:nn].mean() - TRUE_P))

axes[2].loglog(ns, errors, "seagreen", lw=2, alpha=0.8, label="|p̂ - p_true|")
axes[2].loglog(ns, TRUE_P * np.sqrt(TRUE_P*(1-TRUE_P)/ns), "tomato",
               linestyle="--", lw=2, label="1/√n rate (theory)")
axes[2].set_xlabel("Sample size n"); axes[2].set_ylabel("Estimation error |p̂ - p|")
axes[2].set_title("MLE Convergence Rate (Bernoulli)")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mle_demo.png", dpi=120)
print("  Plot saved → mle_demo.png")
print()
print("  KEY RESULTS:")
print("  Gaussian MLE: μ̂ = sample mean  |  σ̂² = biased sample variance")
print("  Bernoulli MLE: p̂ = k/n = fraction of successes")
print("  MLE error shrinks at rate 1/√n (visible in the log-log plot)")
print("  This 1/√n convergence rate is the Cramér-Rao bound for MLE.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Covariance, Correlation & Multivariate Gaussian": {
        "description": (
            "Compute and visualise covariance matrices. "
            "Show Anscombe's quartet — four datasets with identical "
            "mean/variance/correlation but totally different shapes. "
            "Plot samples from multivariate Gaussians with different covariances."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

print("=" * 65)
print("  COVARIANCE, CORRELATION & MULTIVARIATE GAUSSIAN")
print("=" * 65)
print()

# ── Part 1: Covariance fundamentals ──────────────────────────────────────
print("  PART 1 — COVARIANCE PROPERTIES")
print()

n = 1000
x = np.random.normal(0, 1, n)
y_pos  = 2*x + np.random.normal(0, 0.5, n)     # positive correlation
y_neg  = -2*x + np.random.normal(0, 0.5, n)    # negative correlation
y_none = np.random.normal(0, 1, n)              # no correlation
y_quad = x**2 + np.random.normal(0, 0.3, n)    # non-linear (ρ≈0 but dependent!)

pairs = [
    ("positive corr",  x, y_pos,  "steelblue"),
    ("negative corr",  x, y_neg,  "tomato"),
    ("no corr",        x, y_none, "seagreen"),
    ("non-linear",     x, y_quad, "purple"),
]

print(f"  {'Relationship':18s} | {'Cov(X,Y)':>10} | {'ρ (Pearson)':>12} | "
      f"{'Interpretation':25s}")
print(f"  {'─'*75}")
for name, xi, yi, colour in pairs:
    cov = np.cov(xi, yi)[0, 1]
    rho = np.corrcoef(xi, yi)[0, 1]
    interp = {
        "positive corr": "X up → Y up",
        "negative corr": "X up → Y down",
        "no corr":       "linearly independent",
        "non-linear":    "ρ≈0 but NOT independent!",
    }[name]
    print(f"  {name:18s} | {cov:10.4f} | {rho:12.4f} | {interp}")

print()
print("  WARNING: ρ=0 means NO LINEAR relationship — not independence!")
print("  The non-linear (quadratic) case has ρ≈0 but is strongly dependent.")
print()

# ── Part 2: Anscombe's Quartet ─────────────────────────────────────────────
print("  PART 2 — ANSCOMBE'S QUARTET (identical stats, different shapes)")
print()

anscombe = {
    "I (linear)":     ([10,8,13,9,11,14,6,4,12,7,5],
                       [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]),
    "II (curved)":    ([10,8,13,9,11,14,6,4,12,7,5],
                       [9.14,8.14,8.74,8.77,9.26,8.1,6.13,3.1,9.13,7.26,4.74]),
    "III (outlier)":  ([10,8,13,9,11,14,6,4,12,7,5],
                       [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73]),
    "IV (vertical)":  ([8,8,8,8,8,8,8,19,8,8,8],
                       [6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.5,5.56,7.91,6.89]),
}

print(f"  {'Dataset':16s} | {'mean X':>8} | {'mean Y':>8} | {'var X':>8} | "
      f"{'var Y':>8} | {'ρ':>8} | {'shape'}")
print(f"  {'─'*80}")
for dname, (xi, yi) in anscombe.items():
    xa, ya = np.array(xi, float), np.array(yi, float)
    shape_desc = {"I (linear)": "Line", "II (curved)": "Curve",
                  "III (outlier)": "Line+outlier", "IV (vertical)": "Vertical"}
    print(f"  {dname:16s} | {xa.mean():8.2f} | {ya.mean():8.2f} | "
          f"{xa.var():8.2f} | {ya.var():8.2f} | {np.corrcoef(xa,ya)[0,1]:8.2f} | "
          f"{shape_desc[dname]}")

print()
print("  All four datasets have: mean X≈9, mean Y≈7.5, var X≈11, var Y≈4.1, ρ≈0.816")
print("  Yet they look completely different! ALWAYS visualise your data.")
print()

# ── Part 3: Multivariate Gaussians ────────────────────────────────────────
print("  PART 3 — MULTIVARIATE GAUSSIAN WITH DIFFERENT COVARIANCE MATRICES")
print()

cov_configs = [
    ("Independent", np.array([[1, 0], [0, 1]])),
    ("Pos corr r=0.8", np.array([[1, 0.8], [0.8, 1]])),
    ("Neg corr r=-0.8", np.array([[1, -0.8], [-0.8, 1]])),
    ("Anisotropic", np.array([[3, 0], [0, 0.3]])),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Covariance & Correlation — Multivariate Gaussians",
             fontsize=12, fontweight="bold")

colours_ans = ["steelblue", "tomato", "seagreen", "purple"]

for col, (cname, Sigma) in enumerate(cov_configs):
    samples = np.random.multivariate_normal([0, 0], Sigma, 400)
    rho     = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])

    ax_scatter = axes[0, col]
    ax_scatter.scatter(samples[:, 0], samples[:, 1],
                       alpha=0.3, s=10, color=colours_ans[col])
    ax_scatter.set_title(cname + " rho=" + f"{rho:.1f}",
                         fontsize=8)
    ax_scatter.set_xlim(-4, 4); ax_scatter.set_ylim(-4, 4)
    ax_scatter.set_xlabel("X₁"); ax_scatter.set_ylabel("X₂")
    ax_scatter.grid(alpha=0.3)

for col, (dname, (xi_a, yi_a)) in enumerate(anscombe.items()):
    xa_a, ya_a = np.array(xi_a, float), np.array(yi_a, float)
    axes[1, col].scatter(xa_a, ya_a, color=colours_ans[col], s=60)
    # Fit line
    m, b = np.polyfit(xa_a, ya_a, 1)
    xl   = np.linspace(min(xa_a)-1, max(xa_a)+1, 50)
    axes[1, col].plot(xl, m*xl+b, "black", lw=1.5, linestyle="--")
    axes[1, col].set_title("Anscombe: " + dname, fontsize=9)
    axes[1, col].set_xlabel("X"); axes[1, col].set_ylabel("Y")
    axes[1, col].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("covariance_correlation.png", dpi=110)
print("  Plot saved → covariance_correlation.png")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · MLE vs MAP — The Effect of the Prior": {
        "description": (
            "Show numerically and visually how MAP estimation differs from MLE. "
            "Demonstrate that a Gaussian prior = L2 regularisation and "
            "Laplace prior = L1 regularisation. "
            "Show how the prior dominates with small data and vanishes with large data."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)

print("=" * 65)
print("  MLE vs MAP — EFFECT OF THE PRIOR ON PARAMETER ESTIMATES")
print("=" * 65)
print()
print("  MAP = argmax [ log P(D|θ) + log P(θ) ]")
print("            =    log-likelihood  +  log-prior")
print("            =    MLE objective   +  regularisation term")
print()

# ──────────────────────────────────────────────────────────────────────────
# PART 1: Bernoulli — Beta prior (conjugate)
# ──────────────────────────────────────────────────────────────────────────
print("  PART 1 — BERNOULLI + BETA PRIOR (conjugate pair)")
print()
print("  Model:  X ~ Bernoulli(p)")
print("  Prior:  p ~ Beta(α, β)  →  stronger prior = larger α, β")
print("  Posterior: p | X ~ Beta(α + k, β + n - k)")
print("  MAP estimate: p̂_MAP = (α + k - 1) / (α + β + n - 2)")
print("  MLE estimate: p̂_MLE = k / n")
print()

TRUE_P = 0.7   # true coin bias
ALPHA_PRIOR = 2.0   # prior: slightly biased toward 0.5 (α=β=2)
BETA_PRIOR  = 2.0

ns = [1, 3, 10, 30, 100, 500]

print(f"  Prior: Beta(α={ALPHA_PRIOR}, β={BETA_PRIOR})  "
      f"→ prior mean = {ALPHA_PRIOR/(ALPHA_PRIOR+BETA_PRIOR):.2f}  (neutral)")
print(f"  True p = {TRUE_P}")
print()
print(f"  {'n':>5} | {'k (heads)':>10} | {'MLE k/n':>10} | "
      f"{'MAP':>10} | {'Posterior mean':>16}")
print(f"  {'─'*60}")

np.random.seed(1)
data_all = np.random.binomial(1, TRUE_P, max(ns))
map_estimates, mle_estimates = [], []

for n_bern in ns:
    data_n = data_all[:n_bern]
    k      = data_n.sum()
    p_mle  = k / n_bern
    # MAP: mode of Beta posterior
    alpha_post = ALPHA_PRIOR + k
    beta_post  = BETA_PRIOR  + (n_bern - k)
    p_map      = (alpha_post - 1) / (alpha_post + beta_post - 2)
    p_post_mean = alpha_post / (alpha_post + beta_post)

    mle_estimates.append(p_mle)
    map_estimates.append(p_map)
    print(f"  {n_bern:5d} | {k:10d} | {p_mle:10.4f} | {p_map:10.4f} | {p_post_mean:16.4f}")

print()

# ──────────────────────────────────────────────────────────────────────────
# PART 2: L2 regularisation = Gaussian prior
# ──────────────────────────────────────────────────────────────────────────
print("  PART 2 — GAUSSIAN PRIOR = L2 REGULARISATION (Ridge)")
print()
print("  Prior: w ~ N(0, τ²)  →  log P(w) = -w²/(2τ²) + const")
print("  MAP:   w̃ = argmax [ℓ(w) - w²/(2τ²)]")
print("            = argmin [NLL  + λ·w²]   where λ = 1/(2τ²)")
print("  This is exactly L2 (Ridge) regularisation!")
print()

# sklearn replaced with pure NumPy equivalents


rng = np.random.default_rng(0)
X_reg = rng.standard_normal((30, 20))
true_w_reg = rng.standard_normal(20)
y_reg = X_reg @ true_w_reg + 10 * rng.standard_normal(30)

print(f"  Dataset: n=30 samples, p=20 features (underdetermined!)")
print()
print(f"  {'Method':28s} | {'Train R²':>10} | {'||w||₂':>10} | {'Max |w|':>10}")
print(f"  {'─'*62}")


# OLS via numpy least-squares
ols_w, _, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
ols_w_norm = np.linalg.norm(ols_w)
y_hat_ols = X_reg @ ols_w
ss_res = np.sum((y_reg - y_hat_ols)**2)
ss_tot = np.sum((y_reg - y_reg.mean())**2)
ols_train_r2 = 1 - ss_res / ss_tot
print(f"  {'OLS (no prior, λ=0)':28s} | {ols_train_r2:10.4f} | "
      f"{ols_w_norm:10.4f} | {np.max(np.abs(ols_w)):10.4f}")

def ridge_fit(X, y, lam):
    """Closed-form Ridge: w = (XᵀX + λI)⁻¹ Xᵀy"""
    n_f = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(n_f), X.T @ y)

def r2_score(X, y, w):
    y_p = X @ w
    return 1 - np.sum((y - y_p)**2) / np.sum((y - y.mean())**2)

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    w_r = ridge_fit(X_reg, y_reg, alpha)
    w_norm = np.linalg.norm(w_r)
    train_r2 = r2_score(X_reg, y_reg, w_r)
    prior_str = f"Gaussian prior (λ={alpha})"
    print(f"  {prior_str:28s} | {train_r2:10.4f} | {w_norm:10.4f} | "
          f"{np.max(np.abs(w_r)):10.4f}")

print()
print("  As λ increases: weights shrink toward 0 (prior dominates)")
print("  λ→0: MAP→MLE (prior disappears with infinite data)")
print()

# ── Visualise all results ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("MLE vs MAP: Prior Regularises the Estimate",
             fontsize=12, fontweight="bold")

# Plot 1: MLE vs MAP convergence for Bernoulli
axes[0].semilogx(ns, mle_estimates, "tomato", lw=2.5, marker="o",
                 ms=6, label="MLE (p̂ = k/n)")
axes[0].semilogx(ns, map_estimates, "steelblue", lw=2.5, marker="s",
                 ms=6, label="MAP (Beta prior)")
axes[0].axhline(TRUE_P, color="green", linestyle="--", lw=2,
                label=f"True p = {TRUE_P}")
axes[0].axhline(ALPHA_PRIOR/(ALPHA_PRIOR+BETA_PRIOR), color="gray",
                linestyle=":", lw=1.5, label="Prior mean = 0.5")
axes[0].set_xlabel("Sample size n")
axes[0].set_ylabel("Estimated p")
axes[0].set_title("Bernoulli: MLE vs MAP (Beta prior)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1)

# Plot 2: Beta posterior evolution
p_range = np.linspace(0.01, 0.99, 300)
ax2 = axes[1]
colours_beta = plt.cm.viridis(np.linspace(0, 1, len(ns)))
for n_b, colour_b in zip(ns, colours_beta):
    data_nb = data_all[:n_b]
    k_b = data_nb.sum()
    a_post = ALPHA_PRIOR + k_b
    b_post = BETA_PRIOR  + (n_b - k_b)
    pdf_vals = stats.beta.pdf(p_range, a_post, b_post)
    ax2.plot(p_range, pdf_vals, color=colour_b, lw=2, label=f"n={n_b}")
ax2.axvline(TRUE_P, color="red", linestyle="--", lw=2, label=f"True p={TRUE_P}")
ax2.set_xlabel("p"); ax2.set_ylabel("Posterior density")
ax2.set_title("Beta Posterior Updates with More Data")
ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

# Plot 3: Ridge weight shrinkage
alphas   = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
w_norms  = []
for a_r in alphas:
    w_r2 = ridge_fit(X_reg, y_reg, a_r)
    w_norms.append(np.linalg.norm(w_r2))

axes[2].semilogx(alphas, w_norms, "seagreen", lw=2.5, marker="o", ms=6)
axes[2].axhline(ols_w_norm, color="tomato", linestyle="--", lw=2,
                label=f"OLS ||w||={ols_w_norm:.1f}")
axes[2].set_xlabel("λ (regularisation = 1/prior_variance)")
axes[2].set_ylabel("||w||₂ (weight norm)")
axes[2].set_title("Ridge: Weight Shrinkage vs λ")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
axes[2].annotate("Prior dominates", xy=(100, w_norms[-2]),
                 xytext=(20, w_norms[-2]*1.4), fontsize=8,
                 arrowprops=dict(arrowstyle="->"))
axes[2].annotate("MLE (no prior)", xy=(0.001, ols_w_norm*1.02),
                 xytext=(0.002, ols_w_norm*1.3), fontsize=8,
                 arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig("mle_vs_map.png", dpi=120)
print("  Plot saved → mle_vs_map.png")
print()
print("  SUMMARY TABLE — Prior-Regularisation Equivalences:")
print("  ┌─────────────────────────┬─────────────────────────┐")
print("  │ Prior distribution      │ Equivalent to           │")
print("  ├─────────────────────────┼─────────────────────────┤")
print("  │ Gaussian N(0, τ²)       │ L2 / Ridge              │")
print("  │ Laplace(0, b)           │ L1 / Lasso              │")
print("  │ Beta(α, β) on p         │ Pseudocount smoothing   │")
print("  │ No prior                │ MLE (no regularisation) │")
print("  └─────────────────────────┴─────────────────────────┘")
''',
    },
    # ── 7 ─────────────────────────────────────────────────────────────────────
    "7 · Entropy & KL Divergence — Information Theory for ML": {
        "description": (
            "Compute and visualise Shannon entropy, cross-entropy, and KL divergence. "
            "Show the Bernoulli entropy curve. Demonstrate forward vs reverse KL "
            "mode-seeking vs mean-seeking behaviour. Connect cross-entropy to BCE loss."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import rel_entr

print("=" * 65)
print("  INFORMATION THEORY: ENTROPY, CROSS-ENTROPY & KL DIVERGENCE")
print("=" * 65)
print()

# ── Part 1: Shannon Entropy ───────────────────────────────────────────────
print("  PART 1 — SHANNON ENTROPY H(X) = -Σ p(x) log p(x)")
print()

def entropy(p, base=2):
    """Shannon entropy; p is a probability vector."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p) / np.log(base))

examples = [
    ("Certain (p=[1.0, 0.0])",       [1.0, 0.0]),
    ("Fair coin (p=[0.5, 0.5])",      [0.5, 0.5]),
    ("Biased (p=[0.9, 0.1])",         [0.9, 0.1]),
    ("Fair 4-sided (uniform)",        [0.25]*4),
    ("Fair 8-sided (uniform)",        [1/8]*8),
    ("Non-uniform 4 (p=[.7,.1,.1,.1])",[0.7, 0.1, 0.1, 0.1]),
]

print(f"  {'Distribution':40s} | {'H (bits)':>10} | {'H_max (bits)':>12} | {'H/H_max':>9}")
print(f"  {'─'*78}")
for name, probs in examples:
    h     = entropy(probs)
    h_max = np.log2(len(probs))
    print(f"  {name:40s} | {h:10.4f} | {h_max:12.4f} | {h/h_max:9.3f}")

print()
print("  Key insight: Uniform distribution maximises entropy (maximum uncertainty).")
print("  Peaked distribution → low entropy (high confidence).")
print()

# ── Part 2: Cross-Entropy and KL Divergence ───────────────────────────────
print("  PART 2 — CROSS-ENTROPY H(P,Q) AND KL DIVERGENCE KL(P‖Q)")
print()
print("  H(P,Q) = -Σ P(x) log Q(x)     [average bits to encode P using Q]")
print("  KL(P‖Q) = H(P,Q) - H(P)       [extra bits wasted by using Q not P]")
print()

P = np.array([0.4, 0.35, 0.15, 0.10])   # true distribution
Q_list = [
    ("Q = P (perfect)",      P.copy()),
    ("Q = Uniform",          np.array([0.25, 0.25, 0.25, 0.25])),
    ("Q biased toward x=0",  np.array([0.85, 0.05, 0.05, 0.05])),
    ("Q biased toward x=3",  np.array([0.05, 0.05, 0.05, 0.85])),
]

print(f"  {'Q description':26s} | {'H(P)':>7} | {'H(P,Q)':>9} | {'KL(P‖Q)':>10} | {'KL(Q‖P)':>10}")
print(f"  {'─'*72}")
for qname, Q in Q_list:
    hp   = entropy(P, base=np.e)
    hpq  = -np.sum(P * np.log(np.clip(Q, 1e-12, 1)))
    kl_fwd = np.sum(rel_entr(P, Q))   # KL(P‖Q) in nats
    kl_rev = np.sum(rel_entr(Q, P))   # KL(Q‖P) in nats
    print(f"  {qname:26s} | {hp:7.4f} | {hpq:9.4f} | {kl_fwd:10.4f} | {kl_rev:10.4f}")

print()
print("  KL(P‖Q) ≠ KL(Q‖P) — KL is asymmetric (not a true distance).")
print("  KL = 0 only when P = Q exactly.")
print()

# ── Part 3: BCE = KL + H(P) ──────────────────────────────────────────────
print("  PART 3 — BINARY CROSS-ENTROPY AS MLE")
print()
print("  BCE(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]")
print("  This is the MLE log-loss for a Bernoulli model.")
print()

y_true = np.array([1, 0, 1, 1, 0])
y_preds = {
    "Perfect (ŷ=y)":      np.array([0.99, 0.01, 0.99, 0.99, 0.01]),
    "Good (ŷ≈y)":         np.array([0.80, 0.20, 0.75, 0.85, 0.15]),
    "Random (ŷ=0.5)":     np.array([0.50, 0.50, 0.50, 0.50, 0.50]),
    "Wrong predictions":  np.array([0.10, 0.90, 0.15, 0.10, 0.85]),
}

print(f"  {'Prediction quality':22s} | {'Mean BCE':>10} | {'Accuracy':>10}")
print(f"  {'─'*48}")
for pname, yp in y_preds.items():
    bce = -np.mean(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp))
    acc = np.mean((yp >= 0.5) == y_true)
    print(f"  {pname:22s} | {bce:10.4f} | {acc:10.2%}")

print()

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Information Theory: Entropy, KL Divergence, Cross-Entropy",
             fontsize=12, fontweight="bold")

# Plot 1: Bernoulli entropy curve
p_vals = np.linspace(0.001, 0.999, 300)
h_bern = -p_vals * np.log2(p_vals) - (1 - p_vals) * np.log2(1 - p_vals)
axes[0].plot(p_vals, h_bern, "steelblue", lw=2.5)
axes[0].fill_between(p_vals, h_bern, alpha=0.15, color="steelblue")
axes[0].axvline(0.5, color="tomato", linestyle="--", lw=1.5,
                label="Max at p=0.5")
axes[0].set_xlabel("p (Bernoulli parameter)")
axes[0].set_ylabel("Entropy H(p) [bits]")
axes[0].set_title("Bernoulli Entropy Curve")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Plot 2: KL divergence as Q shifts
p_ref = np.array([0.1, 0.4, 0.35, 0.15])
n_cats = len(p_ref)
alpha_mix = np.linspace(0, 1, 100)
kl_mix = []
p_unif = np.ones(n_cats) / n_cats
for a in alpha_mix:
    Q_mix = (1 - a) * p_ref + a * p_unif
    kl_mix.append(np.sum(rel_entr(p_ref, Q_mix)))

axes[1].plot(alpha_mix, kl_mix, "seagreen", lw=2.5)
axes[1].set_xlabel("α  (0 = Q≡P, 1 = Q=Uniform)")
axes[1].set_ylabel("KL(P ‖ Q) [nats]")
axes[1].set_title("KL Divergence as Q Moves Away from P")
axes[1].axvline(0, color="steelblue", linestyle="--", lw=1.5, label="KL=0 at Q=P")
axes[1].legend(); axes[1].grid(alpha=0.3)

# Plot 3: BCE loss surface
y_true_pt = 1.0
yp_range = np.linspace(0.01, 0.99, 300)
bce_y1 = -(np.log(yp_range))           # BCE when true label = 1
bce_y0 = -(np.log(1 - yp_range))       # BCE when true label = 0
axes[2].plot(yp_range, bce_y1, "tomato", lw=2.5, label="y=1: -log(ŷ)")
axes[2].plot(yp_range, bce_y0, "steelblue", lw=2.5, label="y=0: -log(1-ŷ)")
axes[2].set_xlabel("Predicted probability ŷ")
axes[2].set_ylabel("BCE Loss")
axes[2].set_title("Binary Cross-Entropy Loss Surface")
axes[2].set_ylim(0, 5); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("information_theory.png", dpi=120)
print("  Plot saved → information_theory.png")
print()
print("  SUMMARY:")
print("  H(X)     = average surprise; max for uniform; 0 for certain outcomes")
print("  KL(P‖Q)  ≥ 0, = 0 iff P=Q; asymmetric; measures distributional gap")
print("  H(P,Q)   = H(P) + KL(P‖Q); minimising CE ≡ minimising KL ≡ MLE")
print("  BCE      = MLE under Bernoulli model; penalises confident wrong answers")
''',
    },

    # ── 8 ─────────────────────────────────────────────────────────────────────
    "8 · Joint, Marginal & Conditional Distributions": {
        "description": (
            "Build and visualise a discrete joint distribution. "
            "Derive marginals by summing over rows/columns. "
            "Compute conditional distributions and verify independence. "
            "Then show the multivariate Gaussian: marginals, conditionals, "
            "and the effect of the covariance matrix."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("=" * 65)
print("  JOINT, MARGINAL & CONDITIONAL DISTRIBUTIONS")
print("=" * 65)
print()

# ── Part 1: Discrete joint distribution ──────────────────────────────────
print("  PART 1 — DISCRETE JOINT DISTRIBUTION")
print()

# Weather (X) × Commute delay (Y)
# X: 0=Sunny, 1=Cloudy, 2=Rainy
# Y: 0=No delay, 1=Short, 2=Long
joint = np.array([
    [0.30, 0.05, 0.01],   # Sunny
    [0.10, 0.15, 0.05],   # Cloudy
    [0.03, 0.12, 0.19],   # Rainy
])
x_labels = ["Sunny", "Cloudy", "Rainy"]
y_labels  = ["No delay", "Short delay", "Long delay"]

print("  Joint P(Weather, Delay):")
print(f"  {'':12s}", end="")
for yl in y_labels:
    print(f"  {yl:12s}", end="")
print("  │ P(Weather)")
print(f"  {'─'*65}")

marginal_x = joint.sum(axis=1)   # sum over Y
marginal_y = joint.sum(axis=0)   # sum over X

for i, xl in enumerate(x_labels):
    row = "  " + f"{xl:12s}"
    for j in range(3):
        row += f"  {joint[i,j]:12.3f}"
    row += f"  │ {marginal_x[i]:.3f}"
    print(row)

print(f"  {'─'*65}")
print(f"  {'P(Delay)':12s}", end="")
for j in range(3):
    print(f"  {marginal_y[j]:12.3f}", end="")
print(f"  │ {joint.sum():.3f}")

print()
print("  Note: row sums = P(Weather), column sums = P(Delay)")
print(f"  Total probability = {joint.sum():.3f} ✓")
print()

# ── Part 2: Conditionals ─────────────────────────────────────────────────
print("  PART 2 — CONDITIONAL DISTRIBUTIONS P(Delay | Weather)")
print()
print("  P(Y | X=x) = P(X=x, Y) / P(X=x)  [normalise each row by its marginal]")
print()

conditional_y_given_x = joint / marginal_x[:, np.newaxis]
print(f"  {'':12s}", end="")
for yl in y_labels:
    print(f"  {yl:12s}", end="")
print("  │ Sum")
print(f"  {'─'*65}")
for i, xl in enumerate(x_labels):
    row = f"  {xl:12s}"
    for j in range(3):
        row += f"  {conditional_y_given_x[i,j]:12.3f}"
    row += f"  │ {conditional_y_given_x[i].sum():.3f}"
    print(row)

print()
print("  Interpretation: On a Rainy day, P(Long delay)=0.613 vs 0.033 on Sunny.")
print()

# ── Part 3: Independence check ────────────────────────────────────────────
print("  PART 3 — INDEPENDENCE CHECK")
print()
print("  X ⊥ Y  iff  P(X=x, Y=y) = P(X=x) · P(Y=y) for ALL x, y")
print()

product = np.outer(marginal_x, marginal_y)
max_diff = np.max(np.abs(joint - product))
print(f"  Max |P(X,Y) - P(X)·P(Y)| = {max_diff:.4f}")
if max_diff > 0.001:
    print("  → Weather and Commute Delay are NOT independent (as expected!)")
else:
    print("  → Variables appear independent.")
print()

# ── Part 4: Multivariate Gaussian ─────────────────────────────────────────
print("  PART 4 — MULTIVARIATE GAUSSIAN: MARGINALS AND CONDITIONALS")
print()

# 2D Gaussian: X1 = height, X2 = weight (standardised)
mu = np.array([0.0, 0.0])
rho = 0.75
Sigma = np.array([[1.0, rho],
                  [rho, 1.0]])

print(f"  Distribution: X ~ N(μ, Σ)")
print(f"  μ = {mu},  Σ = [[1.0, {rho}], [{rho}, 1.0]]  (ρ = {rho})")
print()

# Conditional: X1 | X2 = x2_obs
x2_obs = 1.5
mu_cond  = mu[0] + rho * (x2_obs - mu[1])
var_cond = 1 - rho**2
print(f"  Conditional: X₁ | X₂ = {x2_obs}")
print(f"    μ₁|₂ = μ₁ + ρ·(x₂ − μ₂) = {mu_cond:.4f}")
print(f"    σ²₁|₂ = σ²₁·(1 − ρ²)     = {var_cond:.4f}")
print(f"    → X₁ | X₂={x2_obs} ~ N({mu_cond:.3f}, {var_cond:.3f})")
print()

samples = np.random.multivariate_normal(mu, Sigma, 1000)

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Joint, Marginal & Conditional Distributions", fontsize=12,
             fontweight="bold")

# Plot 1: Joint discrete heatmap
im = axes[0, 0].imshow(joint, cmap="Blues", aspect="auto")
axes[0, 0].set_xticks(range(3)); axes[0, 0].set_yticks(range(3))
axes[0, 0].set_xticklabels(y_labels, fontsize=8)
axes[0, 0].set_yticklabels(x_labels, fontsize=8)
axes[0, 0].set_title("Joint P(Weather, Delay)")
plt.colorbar(im, ax=axes[0, 0])
for i in range(3):
    for j in range(3):
        axes[0, 0].text(j, i, f"{joint[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="black")

# Plot 2: Marginal distributions
axes[0, 1].bar(x_labels, marginal_x, color="steelblue", alpha=0.8)
axes[0, 1].set_title("Marginal P(Weather)")
axes[0, 1].set_ylabel("Probability")
axes[0, 1].grid(axis="y", alpha=0.4)
for i, v in enumerate(marginal_x):
    axes[0, 1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

axes[0, 2].bar(y_labels, marginal_y, color="seagreen", alpha=0.8)
axes[0, 2].set_title("Marginal P(Delay)")
axes[0, 2].set_ylabel("Probability")
axes[0, 2].grid(axis="y", alpha=0.4)
for i, v in enumerate(marginal_y):
    axes[0, 2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

# Plot 3: 2D Gaussian scatter + marginals
axes[1, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=8, color="purple")
x_line = np.linspace(-3.5, 3.5, 300)
axes[1, 0].axvline(x2_obs, color="tomato", lw=2, linestyle="--",
                   label=f"X₂={x2_obs} (conditioning)")
axes[1, 0].set_xlabel("X₁"); axes[1, 0].set_ylabel("X₂")
axes[1, 0].set_title(f"MVN Samples (ρ={rho})")
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

# Plot 4: Marginal X1 vs Conditional X1|X2=x2_obs
x_range = np.linspace(-4, 4, 300)
marginal_pdf = stats.norm.pdf(x_range, mu[0], 1.0)
cond_pdf     = stats.norm.pdf(x_range, mu_cond, np.sqrt(var_cond))
axes[1, 1].plot(x_range, marginal_pdf, "steelblue", lw=2.5, label="Marginal X₁ ~ N(0,1)")
axes[1, 1].plot(x_range, cond_pdf, "tomato", lw=2.5,
                label=f"Conditional X₁|X₂={x2_obs} ~ N({mu_cond:.2f},{var_cond:.2f})")
axes[1, 1].set_xlabel("X₁"); axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("Marginal vs Conditional PDF")
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

# Plot 5: Independence check — residuals
diff = joint - product
im2 = axes[1, 2].imshow(diff, cmap="RdBu_r", aspect="auto",
                         vmin=-max_diff, vmax=max_diff)
axes[1, 2].set_xticks(range(3)); axes[1, 2].set_yticks(range(3))
axes[1, 2].set_xticklabels(y_labels, fontsize=8)
axes[1, 2].set_yticklabels(x_labels, fontsize=8)
axes[1, 2].set_title("P(X,Y) − P(X)·P(Y)\\n(0 = independent)")
plt.colorbar(im2, ax=axes[1, 2])
for i in range(3):
    for j in range(3):
        axes[1, 2].text(j, i, f"{diff[i,j]:.3f}", ha="center", va="center",
                        fontsize=9)

plt.tight_layout()
plt.savefig("joint_marginal_conditional.png", dpi=120)
print("  Plot saved → joint_marginal_conditional.png")
print()
print("  KEY TAKEAWAYS:")
print("  Marginal: collapse joint by summing out the other variable")
print("  Conditional: normalise a row/column of the joint table")
print("  Independence: P(X,Y) = P(X)·P(Y) — residual heatmap is all zeros")
print("  MVN conditional: mean shifts linearly with ρ, variance shrinks by (1-ρ²)")
''',
    },

    # ── 9 ─────────────────────────────────────────────────────────────────────
    "9 · Hypothesis Testing & Confidence Intervals": {
        "description": (
            "Run a one-sample t-test step-by-step. "
            "Visualise the null distribution and p-value. "
            "Simulate the correct interpretation of a 95% confidence interval. "
            "Show Type I/II error tradeoffs and the multiple testing problem."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)

print("=" * 65)
print("  HYPOTHESIS TESTING & CONFIDENCE INTERVALS")
print("=" * 65)
print()

# ── Part 1: One-sample t-test ─────────────────────────────────────────────
print("  PART 1 — ONE-SAMPLE t-TEST (step by step)")
print()
print("  Claim: the average height of a group differs from 170 cm")
print()

TRUE_MU   = 173.5    # population mean we will try to detect
MU_0      = 170.0    # null hypothesis mean
N_SAMPLE  = 30

data = np.random.normal(TRUE_MU, 8, N_SAMPLE)
x_bar = data.mean()
s     = data.std(ddof=1)
se    = s / np.sqrt(N_SAMPLE)
t_stat = (x_bar - MU_0) / se
df    = N_SAMPLE - 1
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))    # two-tailed

print(f"  Data: n={N_SAMPLE}, x̄={x_bar:.2f} cm, s={s:.2f} cm")
print()
print(f"  H₀: μ = {MU_0} cm")
print(f"  H₁: μ ≠ {MU_0} cm  (two-tailed)")
print()
print(f"  Standard error: SE = s/√n = {s:.2f}/√{N_SAMPLE} = {se:.3f}")
print(f"  Test statistic: t  = (x̄ − μ₀) / SE = ({x_bar:.2f} − {MU_0}) / {se:.3f} = {t_stat:.3f}")
print(f"  Degrees of freedom: df = n − 1 = {df}")
print(f"  p-value (two-tailed): {p_val:.4f}")
print()
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df)
print(f"  Critical value (α=0.05, two-tailed): ±{t_crit:.3f}")
decision = "REJECT H₀" if p_val < alpha else "FAIL TO REJECT H₀"
print(f"  |t| = {abs(t_stat):.3f} {'>' if abs(t_stat) > t_crit else '<'} {t_crit:.3f}  →  {decision}")
print()
if p_val < alpha:
    print(f"  Conclusion: There is sufficient evidence (p={p_val:.4f} < α={alpha})")
    print(f"  that the mean height differs from {MU_0} cm.")
else:
    print(f"  Conclusion: Insufficient evidence (p={p_val:.4f} ≥ α={alpha})")
    print(f"  to conclude the mean height differs from {MU_0} cm.")
print()

# ── Part 2: Confidence Interval ───────────────────────────────────────────
print("  PART 2 — CONFIDENCE INTERVALS")
print()
ci_lo = x_bar - t_crit * se
ci_hi = x_bar + t_crit * se
print(f"  95% CI:  x̄ ± t_crit · SE  =  {x_bar:.2f} ± {t_crit:.3f} × {se:.3f}")
print(f"         = [{ci_lo:.2f},  {ci_hi:.2f}]")
print()
print("  Correct interpretation:")
print("  'If we repeated this experiment many times, 95% of the constructed")
print("  intervals would contain the true population mean.'")
print()
print("  WRONG interpretations:")
print("  ✗ '95% probability the true mean is in this interval' (frequentist CI!)")
print("  ✗ 'We are 95% sure μ = 173.5' (that would be a Bayesian credible interval)")
print()

# Simulate coverage
n_sims  = 200
n_cover = 0
ci_list = []
for _ in range(n_sims):
    sample = np.random.normal(TRUE_MU, 8, N_SAMPLE)
    xb = sample.mean()
    sb = sample.std(ddof=1)
    lo = xb - t_crit * (sb / np.sqrt(N_SAMPLE))
    hi = xb + t_crit * (sb / np.sqrt(N_SAMPLE))
    covers = lo <= TRUE_MU <= hi
    n_cover += covers
    ci_list.append((lo, hi, covers))

print(f"  Simulation ({n_sims} experiments): {n_cover}/{n_sims} = {n_cover/n_sims:.1%} intervals contain μ={TRUE_MU}")
print(f"  (Expected coverage ≈ 95%)")
print()

# ── Part 3: Type I vs Type II error and power ─────────────────────────────
print("  PART 3 — TYPE I / TYPE II ERRORS AND POWER")
print()
alphas = [0.01, 0.05, 0.10]
effects = [1.0, 2.0, 3.5, 5.0]   # true effect size (μ₁ - μ₀), σ=8

print(f"  Power = P(reject H₀ | H₁ true)   [n={N_SAMPLE}, σ=8, two-tailed]")
print()
print(f"  {'Effect (μ₁-μ₀)':18s}", end="")
for a in alphas:
    print(f"  α={a:.2f}", end="")
print()
print(f"  {'─'*48}")

sigma = 8.0
for eff in effects:
    print(f"  {eff:18.1f}", end="")
    for a in alphas:
        tc = stats.t.ppf(1 - a/2, df)
        ncp = eff / (sigma / np.sqrt(N_SAMPLE))   # non-centrality parameter
        power = 1 - stats.t.cdf(tc, df, loc=ncp) + stats.t.cdf(-tc, df, loc=ncp)
        print(f"  {power:6.3f}", end="")
    print()

print()
print("  Larger effect → higher power. Smaller α → lower power (stricter threshold).")
print()

# ── Part 4: Multiple testing ──────────────────────────────────────────────
print("  PART 4 — MULTIPLE TESTING PROBLEM")
print()
n_tests = 20
print(f"  Running {n_tests} tests at α=0.05, all under H₀ (no true effect).")
p_vals_null = np.random.uniform(0, 1, n_tests)
false_pos = (p_vals_null < 0.05).sum()
print(f"  False positives (p < 0.05): {false_pos}/{n_tests}")
print(f"  Expected: ~{n_tests * 0.05:.0f}  (FWER = 1-(0.95)^{n_tests} = {1-0.95**n_tests:.3f})")
print()
bonferroni_alpha = 0.05 / n_tests
print(f"  Bonferroni correction: use α/{n_tests} = {bonferroni_alpha:.4f} per test")
print(f"  False positives after Bonferroni: {(p_vals_null < bonferroni_alpha).sum()}")
print()

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Hypothesis Testing & Confidence Intervals", fontsize=12,
             fontweight="bold")

# Plot 1: t-distribution with rejection regions
t_range = np.linspace(-5, 5, 500)
t_pdf   = stats.t.pdf(t_range, df)
axes[0].plot(t_range, t_pdf, "steelblue", lw=2.5, label=f"t({df})")
axes[0].fill_between(t_range, t_pdf,
                     where=(t_range >= t_crit) | (t_range <= -t_crit),
                     color="tomato", alpha=0.5, label=f"Rejection region (α={alpha})")
axes[0].axvline(t_stat, color="purple", lw=2.5, linestyle="--",
                label=f"t_obs={t_stat:.2f} (p={p_val:.3f})")
axes[0].axvline(-t_stat, color="purple", lw=2.5, linestyle="--")
axes[0].set_xlabel("t statistic"); axes[0].set_ylabel("Density")
axes[0].set_title("Null Distribution with Observed t")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Plot 2: CI coverage simulation (first 50)
ax = axes[1]
show = 50
for k, (lo, hi, covers) in enumerate(ci_list[:show]):
    color = "steelblue" if covers else "tomato"
    ax.hlines(k, lo, hi, color=color, lw=1.2, alpha=0.8)
ax.axvline(TRUE_MU, color="black", lw=2, linestyle="--", label=f"True μ={TRUE_MU}")
ax.set_xlabel("Mean estimate")
ax.set_ylabel("Simulation #")
ax.set_title(f"95% CI Coverage ({show} experiments)\\nRed = misses true μ")
ax.legend(fontsize=9); ax.grid(alpha=0.2)

# Plot 3: Power curve
effect_range = np.linspace(0, 8, 200)
for a, col in zip([0.01, 0.05, 0.10], ["seagreen", "steelblue", "tomato"]):
    tc_pow = stats.t.ppf(1 - a/2, df)
    powers = []
    for eff in effect_range:
        ncp = eff / (sigma / np.sqrt(N_SAMPLE))
        pw  = 1 - stats.t.cdf(tc_pow, df, loc=ncp) + stats.t.cdf(-tc_pow, df, loc=ncp)
        powers.append(pw)
    axes[2].plot(effect_range, powers, color=col, lw=2.5, label=f"α={a}")
axes[2].axhline(0.8, color="gray", linestyle="--", lw=1.5, label="80% power convention")
axes[2].set_xlabel("True effect size (μ₁ − μ₀)")
axes[2].set_ylabel("Power = P(reject H₀ | H₁)")
axes[2].set_title(f"Power Curve (n={N_SAMPLE}, σ={sigma})")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hypothesis_testing.png", dpi=120)
print("  Plot saved → hypothesis_testing.png")
print()
print("  KEY TAKEAWAYS:")
print("  p-value: P(data this extreme | H₀ true) — NOT P(H₀ true | data)")
print("  CI: long-run coverage frequency — NOT a probability statement per interval")
print("  Power = 1 - β: increases with n, effect size, and α")
print("  Multiple testing: correct with Bonferroni (conservative) or BH-FDR")
''',
    },

    # ── 10 ────────────────────────────────────────────────────────────────────
    "10 · Sampling Methods — Monte Carlo, Rejection & MCMC": {
        "description": (
            "Implement and compare four sampling strategies: "
            "Monte Carlo integration (estimating π), "
            "importance sampling, "
            "rejection sampling (sampling from a non-standard density), "
            "and Metropolis-Hastings MCMC (sampling a bimodal posterior). "
            "Plot traces, histograms, and convergence diagnostics."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("=" * 65)
print("  SAMPLING METHODS: MONTE CARLO, REJECTION & MCMC")
print("=" * 65)
print()

# ── Part 1: Monte Carlo Integration — estimating π ───────────────────────
print("  PART 1 — MONTE CARLO INTEGRATION (estimating π)")
print()
print("  Throw darts uniformly at the unit square [0,1]².")
print("  π/4 = P(dart lands inside unit circle) = E[1(x²+y²≤1)]")
print()

ns_mc = [10, 100, 1000, 10000, 100000]
for n_mc in ns_mc:
    x_mc = np.random.uniform(-1, 1, n_mc)
    y_mc = np.random.uniform(-1, 1, n_mc)
    pi_est = 4 * np.mean(x_mc**2 + y_mc**2 <= 1)
    error  = abs(pi_est - np.pi)
    print(f"  n={n_mc:8,d}: π̂ = {pi_est:.5f}  error = {error:.5f}  "
          f"(theory: 1/√n ≈ {1/np.sqrt(n_mc):.5f})")

print()

# ── Part 2: Importance Sampling ───────────────────────────────────────────
print("  PART 2 — IMPORTANCE SAMPLING")
print()
print("  Goal: E_p[f(X)] where p = N(3, 1) and we can only sample q = N(0, 2)")
print("  True E[X²] under N(3,1) = 3² + 1² = 10.0")
print()

p_dist = stats.norm(loc=3, scale=1)    # target
q_dist = stats.norm(loc=0, scale=2)    # proposal

true_val = 3**2 + 1     # E[X²] = Var(X) + E[X]² = 1 + 9

n_is = 5000
samples_q = q_dist.rvs(n_is)
weights    = p_dist.pdf(samples_q) / q_dist.pdf(samples_q)   # importance weights
f_vals     = samples_q**2                                     # f(x) = x²

naive_mc = np.mean(f_vals[:1000])                  # naive MC from q (wrong!)
IS_est   = np.sum(weights * f_vals) / np.sum(weights)   # self-normalised IS

print(f"  True E[X²] under p=N(3,1):          {true_val:.4f}")
print(f"  Naive MC from q=N(0,2) (wrong):      {naive_mc:.4f}  (biased!)")
print(f"  Importance sampling estimate (n={n_is:,}): {IS_est:.4f}")
print(f"  IS error: {abs(IS_est - true_val):.5f}")
print()

# ── Part 3: Rejection Sampling ────────────────────────────────────────────
print("  PART 3 — REJECTION SAMPLING")
print()
print("  Target: Beta(2.7, 6.3)  [tricky skewed density]")
print("  Proposal: Uniform(0, 1) scaled by M=max(p(x)/q(x))")
print()

target = stats.beta(2.7, 6.3)
M      = target.pdf(np.linspace(0, 1, 1000)).max() + 0.01

n_proposed = 10000
u_prop     = np.random.uniform(0, 1, n_proposed)    # proposed x
accept_u   = np.random.uniform(0, 1, n_proposed)    # acceptance uniform
rej_samples = u_prop[accept_u <= target.pdf(u_prop) / M]
acc_rate    = len(rej_samples) / n_proposed

print(f"  M (envelope constant) = {M:.3f}")
print(f"  Proposed: {n_proposed:,},  Accepted: {len(rej_samples):,}  "
      f"(acceptance rate = {acc_rate:.3f} ≈ 1/M = {1/M:.3f})")
print(f"  Sample mean: {rej_samples.mean():.4f}  (true mean: {target.mean():.4f})")
print()

# ── Part 4: Metropolis-Hastings MCMC ──────────────────────────────────────
print("  PART 4 — METROPOLIS-HASTINGS MCMC (bimodal target)")
print()
print("  Target: 0.4·N(-2, 0.6²) + 0.6·N(2, 0.8²)  [bimodal mixture]")
print("  Proposal: x* ~ N(xₜ, σ_prop²)  [random walk)")
print()

def log_target(x):
    """Log unnormalised target density: mixture of two Gaussians."""
    return np.logaddexp(
        np.log(0.4) + stats.norm.logpdf(x, -2, 0.6),
        np.log(0.6) + stats.norm.logpdf(x,  2, 0.8)
    )

n_mcmc    = 10000
burn_in   = 1000
sigma_prop = 1.5

chain = np.zeros(n_mcmc)
chain[0] = 0.0
n_accept  = 0

for t in range(1, n_mcmc):
    x_prop = chain[t-1] + np.random.normal(0, sigma_prop)
    log_alpha = log_target(x_prop) - log_target(chain[t-1])
    if np.log(np.random.uniform()) < log_alpha:
        chain[t] = x_prop
        n_accept += 1
    else:
        chain[t] = chain[t-1]

acc_rate_mcmc = n_accept / (n_mcmc - 1)
posterior     = chain[burn_in:]

true_mean = 0.4 * (-2) + 0.6 * 2
true_var  = 0.4 * (0.6**2 + (-2 - true_mean)**2) + 0.6 * (0.8**2 + (2 - true_mean)**2)

print(f"  Chain length: {n_mcmc:,}  |  Burn-in: {burn_in:,}  |  Post-burn: {len(posterior):,}")
print(f"  Acceptance rate: {acc_rate_mcmc:.3f}  (ideal range ≈ 0.23–0.44 for 1D)")
print(f"  Posterior mean: {posterior.mean():.4f}  (true: {true_mean:.4f})")
print(f"  Posterior std:  {posterior.std():.4f}  (true: {np.sqrt(true_var):.4f})")
print()

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Sampling Methods: Monte Carlo, Rejection Sampling & MCMC",
             fontsize=12, fontweight="bold")

# Plot 1: MC π estimation
n_vis = 3000
x_v = np.random.uniform(-1, 1, n_vis)
y_v = np.random.uniform(-1, 1, n_vis)
inside = x_v**2 + y_v**2 <= 1
axes[0, 0].scatter(x_v[inside], y_v[inside], s=1, c="steelblue", alpha=0.5)
axes[0, 0].scatter(x_v[~inside], y_v[~inside], s=1, c="tomato", alpha=0.5)
theta = np.linspace(0, 2*np.pi, 300)
axes[0, 0].plot(np.cos(theta), np.sin(theta), "black", lw=2)
axes[0, 0].set_aspect("equal")
axes[0, 0].set_title(f"Monte Carlo π Estimation\\nπ̂ ≈ {4*inside.mean():.4f}, n={n_vis:,}")
axes[0, 0].grid(alpha=0.3)

# Plot 2: IS weight distribution
axes[0, 1].hist(weights, bins=60, color="seagreen", alpha=0.7, density=True)
axes[0, 1].set_xlabel("Importance weight w(x) = p(x)/q(x)")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title("Importance Weights Distribution\\n(ideal: all equal → low variance)")
axes[0, 1].grid(alpha=0.3)

# Plot 3: Rejection sampling
x_rej = np.linspace(0, 1, 300)
axes[0, 2].hist(rej_samples, bins=40, density=True, color="purple", alpha=0.6,
                label="Rejected samples")
axes[0, 2].plot(x_rej, target.pdf(x_rej), "black", lw=2.5, label="True Beta(2.7, 6.3)")
axes[0, 2].plot(x_rej, M * np.ones_like(x_rej), "tomato", linestyle="--", lw=1.5,
                label=f"Envelope M={M:.2f}")
axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("Density")
axes[0, 2].set_title(f"Rejection Sampling\\nacc. rate = {acc_rate:.2f}")
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

# Plot 4: MCMC trace
axes[1, 0].plot(chain[:2000], lw=0.5, color="steelblue", alpha=0.8)
axes[1, 0].axvline(burn_in, color="tomato", lw=2, linestyle="--", label="End of burn-in")
axes[1, 0].set_xlabel("Iteration"); axes[1, 0].set_ylabel("x")
axes[1, 0].set_title("MCMC Chain Trace (first 2,000)")
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.3)

# Plot 5: MCMC posterior vs true density
x_plot = np.linspace(-5, 5, 400)
true_pdf = 0.4 * stats.norm.pdf(x_plot, -2, 0.6) + \
           0.6 * stats.norm.pdf(x_plot,  2, 0.8)
axes[1, 1].hist(posterior, bins=80, density=True, color="steelblue", alpha=0.6,
                label="MCMC samples")
axes[1, 1].plot(x_plot, true_pdf, "tomato", lw=2.5, label="True target density")
axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("MCMC Posterior vs True Bimodal Target")
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)

# Plot 6: Convergence — running mean
running_mean = np.cumsum(chain[burn_in:]) / np.arange(1, len(posterior) + 1)
axes[1, 2].plot(running_mean, "seagreen", lw=1.5, label="MCMC running mean")
axes[1, 2].axhline(true_mean, color="tomato", linestyle="--", lw=2,
                   label=f"True mean = {true_mean:.3f}")
axes[1, 2].set_xlabel("Post-burn-in iteration")
axes[1, 2].set_ylabel("Running mean of chain")
axes[1, 2].set_title("MCMC Convergence: Running Mean")
axes[1, 2].legend(fontsize=9); axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sampling_methods.png", dpi=120)
print("  Plot saved → sampling_methods.png")
print()
print("  KEY TAKEAWAYS:")
print("  Monte Carlo: E[f(X)] ≈ (1/N)Σf(xᵢ), error ~ 1/√N (dimension-free!)")
print("  Importance:  reweights samples from easy proposal; beware weight variance")
print("  Rejection:   exact samples but rate = 1/M — terrible in high dimensions")
print("  MCMC (MH):  samples correlated, needs burn-in; works for any unnorm. density")
print("  Acceptance rate ~0.23-0.44 is the MH sweet spot for 1D random walk")
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