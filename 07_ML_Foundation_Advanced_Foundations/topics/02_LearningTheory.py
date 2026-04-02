"""
Learning Theory
===============

The mathematical foundations of why learning from data works at all —
PAC learning, VC dimension, generalisation bounds, the fundamental theorem
of statistical learning, and the modern double descent phenomenon.

"""

import textwrap
import re

TOPIC_NAME = "Learning Theory"
DISPLAY_NAME = "02 · Learning Theory"
ICON = "📐"
SUBTITLE = "Why Generalisation Works — and When It Doesn't"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Question: What Does It Mean to Learn?

Every ML algorithm takes a finite sample of data and outputs a rule that
we hope works on unseen data. But why should that work at all? We observed
a small slice of the world — why should our rule generalise beyond it?

Learning theory provides the mathematical answer. It asks:

    How many examples do we need to PAC-learn a hypothesis class H?
    What property of H determines whether it is learnable at all?
    How much worse can our hypothesis be, compared to the best in H?
    How does model complexity interact with generalisation error?

These questions were formalised by Valiant (1984) in the PAC framework —
one of the most important theoretical contributions in computer science.

    The setup shared by all of learning theory:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Unknown distribution  D  over  X × Y    (nature's data generator)  │
    │                                                                     │
    │  Training sample       S = {(x₁,y₁), ..., (xₘ,yₘ)} ~ D^m            │
    │  (m i.i.d. draws from D)                                            │
    │                                                                     │
    │  Hypothesis class      H  (the model family we allow)               │
    │                                                                     │
    │  Learning algorithm    A : (X × Y)^m → H                            │
    │  (maps a sample to a hypothesis)                                    │
    │                                                                     │
    │  True risk (goal):     R(h)  = E_{(x,y)~D} [ℓ(h(x), y)]             │
    │  Empirical risk (seen):R̂(h) = (1/m) Σᵢ ℓ(h(xᵢ), yᵢ)                 │
    │                                                                     │
    │  The generalisation gap = R(h) − R̂(h)                               │
    │  We want: small generalisation gap with high probability.           │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    Why is this hard?

    R(h) is the expectation over the infinite population — we can never
    compute it. We only observe R̂(h) on our finite sample. The question
    is: how confident can we be that R̂(h) ≈ R(h)?

    The fundamental tension:
    ┌────────────────────────────────────────────────────────────────────┐
    │  Larger H  → can find h with smaller R̂(h)   (less underfitting)    │
    │            → but R(h) can be much larger than R̂(h) (overfitting)   │
    │                                                                    │
    │  Smaller H → R̂(h) ≈ R(h) more reliably    (good generalisation)    │
    │            → but the best h in H may be mediocre  (underfitting)   │
    └────────────────────────────────────────────────────────────────────┘

    This is the bias-complexity tradeoff — the formal version of the
    bias-variance tradeoff from Module 03.


──────────────────────────────────────────────────────────────────────────────
### PAC Learning — Probably Approximately Correct

Valiant's PAC framework gives the first formal definition of learnability.

    Definition (PAC Learnability):

        A hypothesis class H is PAC-learnable if there exists an algorithm A
        and a polynomial function m_H(ε, δ) such that:

        For any ε > 0  (accuracy parameter)
        For any δ > 0  (confidence parameter)
        For any distribution D over X × Y

        If m ≥ m_H(ε, δ), then with probability ≥ 1 − δ over the draw of
        S ~ D^m:

            R(A(S)) ≤ min_{h ∈ H} R(h) + ε

        The output hypothesis is ε-close to optimal, with confidence 1 − δ.

    Reading the parameters:

        ε  (epsilon) = accuracy.  How close to optimal do we need to be?
                       ε = 0.01 → within 1% of the best possible error.
        δ  (delta)   = confidence.  How often can we fail?
                       δ = 0.05 → at most 5% chance of a "bad" sample.
        m_H(ε, δ)    = sample complexity.  How many examples do we need?
                       This is the key quantity learning theory aims to bound.

    Diagram 1 — PAC Learning Schematic:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │           ALL POSSIBLE HYPOTHESES IN H                              │
    │                                                                     │
    │   ●  ● ●   ●         ← bad hypotheses (high R(h))                   │
    │         ●  ●  ●                                                     │
    │    ●  ●       ●                                                     │
    │         ┌───────────────────────────────────────────────┐           │
    │     ●   │ ε-ball around the optimal h*                  │ ●         │
    │         │                                               │           │
    │   ●     │   ●    h*  ← best in H                  ●     │           │
    │         │          ★ ← h output by algorithm       │    │           │
    │         │              (PAC guarantee: h is here   │    │           │
    │    ●    │               with prob ≥ 1−δ)           │    │           │
    │         └───────────────────────────────────────────────┘           │
    │                                                                     │
    │  PAC guarantee: A(S) lands inside the ε-ball with prob ≥ 1 − δ      │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

**Sample Complexity for Finite Hypothesis Classes:**

    When H is finite (|H| < ∞), the union bound gives a clean result.

    Idea:  A "bad" hypothesis h is one with R(h) > ε but R̂(h) ≈ 0
           (it "looks good" on our sample but fails in the real world).
           The probability that a single bad h fools us:

               P[R̂(h) = 0 | R(h) > ε] ≤ (1 − ε)^m ≤ e^{−mε}

           Applying the union bound over all |H| hypotheses:

               P[∃ bad h that fools us] ≤ |H| · e^{−mε}

    Setting this ≤ δ and solving for m:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │           m ≥  (1/ε) · ln(|H| / δ)                               │
    │                                                                  │
    │   Sample complexity for PAC learning a finite class H.           │
    │   Logarithmic in |H| — surprisingly mild even for huge H.        │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Intuition:  To distinguish among |H| hypotheses, you need only
                log₂(|H|) bits of information — which requires roughly
                log(|H|) samples. PAC theory formalises this intuition.

    Example:
        Boolean conjunctions on d variables:  |H| = 3^d
        For ε = 0.01, δ = 0.05, d = 20:

            m ≥ (1/0.01) · ln(3²⁰ / 0.05)
              = 100 · ln(3.4 × 10⁹ / 0.05)
              = 100 · ln(6.8 × 10¹⁰)
              ≈ 100 · 25.0  ≈ 2500 examples

        Modest! Even though H has ~3.5 billion members.

    Agnostic PAC learning (no realizable assumption):

        The above assumed H contains the true hypothesis (realizable case).
        In practice we never know this. The agnostic version drops the
        assumption and asks: can we compete with the best h ∈ H?

        Agnostic PAC bound (finite H):
            m ≥ (1/2ε²) · ln(2|H| / δ)

        Note the 1/ε² instead of 1/ε — agnostic PAC is harder because
        we must also deal with irreducible noise in the labels.


──────────────────────────────────────────────────────────────────────────────
### VC Dimension — Measuring the Complexity of a Hypothesis Class

For infinite H (lines, polynomials, neural networks), |H| = ∞ and the
finite-H bound is useless. We need a different complexity measure.

    Key concept:  SHATTERING

    A hypothesis class H shatters a set C = {x₁, ..., xₙ} if:
    for EVERY possible binary labelling of C (all 2ⁿ labellings),
    some h ∈ H achieves that labelling perfectly.

    Diagram 2 — Shattering 3 Points with Oriented Lines in ℝ²:

    We need to realise all 2³ = 8 labellings (● = positive, ○ = negative):

    ● ● ●    ● ● ○    ● ○ ●    ○ ● ●    ● ○ ○    ○ ● ○    ○ ○ ●    ○ ○ ○
    (all +)  (++-) ...                                            (all -)

    For 3 points in general position, an oriented hyperplane (halfspace)
    can realise all 8 labellings → these 3 points are shattered by H.

    BUT: no 4 points in ℝ² can be shattered by halfspaces (there is always
    at least one labelling where the positive class is not linearly separable
    from the negative class — e.g. XOR pattern). So VCdim = 3 for ℝ².

    Definition (VC Dimension):

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  VCdim(H) = max { n : ∃ C ⊆ X with |C| = n  s.t.  H shatters C}  │
    │                                                                  │
    │  The size of the LARGEST set that H can shatter.                 │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    VC Dimensions of common hypothesis classes:

    ┌──────────────────────────────────────────┬──────────────────────┐
    │  Hypothesis Class                        │  VCdim               │
    ├──────────────────────────────────────────┼──────────────────────┤
    │  Threshold functions in ℝ¹               │  1                   │
    │  Intervals in ℝ¹                         │  2                   │
    │  Halfspaces (hyperplanes) in ℝᵈ          │  d + 1               │
    │  Axis-aligned rectangles in ℝᵈ           │  2d                  │
    │  k-nearest-neighbour (k=1)               │  ∞                   │
    │  Finite sets of n functions              │  ⌊log₂ n⌋            │
    │  Decision trees with n leaves            │  O(n log n)          │
    │  Neural net (W weights, depth k)         │  O(W k log W)        │
    │  Convex polygons in ℝ²                   │  ∞                   │
    └──────────────────────────────────────────┴──────────────────────┘

    WARNING:  VCdim = ∞ means the class can memorise any dataset of any
              size. This does NOT mean the class is useless — it means
              the PAC framework alone cannot guarantee generalisation.
              Additional structure (implicit regularisation, SGD dynamics,
              early stopping) must provide the missing control.

**The Growth Function — Counting Dichotomies:**

    The growth function is the key link between shattering and bounds.

        Π_H(m) = max_{x₁,...,xₘ ∈ X} | { (h(x₁),...,h(xₘ)) : h ∈ H } |

    Π_H(m) = the maximum number of distinct labellings H can produce
              on any m points. Upper bounded by 2^m (all dichotomies).

    Sauer-Shelah Lemma:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  If VCdim(H) = d, then for all m:                                │
    │                                                                  │
    │         Π_H(m) ≤ Σᵢ₌₀ᵈ C(m, i) ≤ (em/d)^d                         │
    │                                                                  │
    │  For m >> d, this is POLYNOMIAL in m — far smaller than 2^m.     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    This polynomial growth is what makes generalisation possible.
    When Π_H(m) grows as m^d rather than 2^m, the class cannot memorise
    arbitrary noise. Structure is preserved.

    Diagram 3 — Growth Function: Polynomial vs Exponential:

    log₂ Π_H(m)
         │
      m  │ ────────────────────────────── 2^m  (all dichotomies)
         │                              /
         │                           /
         │               d=4 -------
     m·d │         d=2 ---
         │   d=1 --
         │ /
         └────────────────────────────────── m
              d    2d  3d            ← inflection near m = d
                   ↑
         Sauer-Shelah: for m > d, growth becomes polynomial

    For m ≤ d: growth is still exponential (d is the "memorisation zone").
    For m > d: growth is polynomial — generalisation is possible.


──────────────────────────────────────────────────────────────────────────────
### The Fundamental Theorem of Statistical Learning

This theorem — the central result of classical learning theory — gives the
complete characterisation of PAC learnability.

    Theorem (Fundamental Theorem of Statistical Learning):

    The following are equivalent for any hypothesis class H of binary
    classifiers:

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  1.  H is PAC-learnable (agnostic setting).                            │
    │  2.  H has finite VC dimension.                                        │
    │  3.  H has finite Rademacher complexity (see below).                   │
    │  4.  H has the Uniform Convergence Property:                           │
    │      sup_{h ∈ H} |R(h) − R̂(h)| → 0 as m → ∞.                           │
    │                                                                        │
    │  The number of examples needed to PAC-learn H (VCdim = d):             │
    │                                                                        │
    │C₁ · (d + log(1/δ)) / ε² ≤ m_H(ε,δ) ≤ C₂ · (d log(1/ε) + log(1/δ)) / ε² │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    MEANING:  Finite VC dimension is NECESSARY AND SUFFICIENT for
              PAC learnability. This is not just a sufficient condition —
              infinite VC dimension means learning is IMPOSSIBLE in the
              worst case.

**The VC Generalisation Bound:**

    With probability ≥ 1 − δ over the sample S ~ D^m, for all h ∈ H:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   R(h)  ≤  R̂(h)  +  √[ (d · log(2em/d) + log(2/δ)) / (2m) ]      │
    │                                ↑                                 │
    │                     complexity penalty                           │
    │                                                                  │
    │   where d = VCdim(H), m = number of samples.                     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Reading the bound:
    ┌───────────────────────────────────────────────────────────────────┐
    │  True error ≤ Training error + complexity penalty                 │
    │                                                                   │
    │  Complexity penalty grows with:  d (VC dim), 1/δ (confidence)     │
    │  Complexity penalty shrinks with: m (more data)                   │
    │                                                                   │
    │  Penalty → 0 as m → ∞  (consistent learning)                      │
    │  Penalty → ∞ as d → ∞  (infinite VC dim → no bound)               │
    └───────────────────────────────────────────────────────────────────┘

    Diagram 4 — The Bias-Complexity Decomposition:

    Total Expected Error = Approximation Error + Estimation Error

    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │  APPROX. ERROR = min_{h ∈ H} R(h) − R(h*)                         │
    │  (h* = true Bayes optimal)                                        │
    │  → measures how well H can represent the true pattern             │
    │  → DECREASES as H gets larger (more expressive)                   │
    │                                                                   │
    │  ESTIMATION ERROR = R(h_ERM) − min_{h ∈ H} R(h)                   │
    │  (h_ERM = empirical risk minimiser)                               │
    │  → measures how much the finite sample hurts us                   │
    │  → INCREASES as H gets larger (harder to estimate well)           │
    │                                                                   │
    │  Optimum: choose H to balance approximation and estimation error  │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘

    This is the FORMAL basis of the bias-variance tradeoff (Module 03):

        Approximation error  ←→  Bias²
        Estimation error     ←→  Variance (+ complexity penalty)


──────────────────────────────────────────────────────────────────────────────
### Rademacher Complexity — A Tighter Measure

VC dimension is a combinatorial property of H — it does not depend on the
data distribution D. Rademacher complexity is data-dependent and often
gives tighter bounds.

    Definition (Empirical Rademacher Complexity):

        Given sample S = {x₁, ..., xₘ}, and random signs σ₁,...,σₘ ∈ {±1}
        each drawn uniformly (Rademacher variables):

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  R̂_S(H) = E_σ [ sup_{h ∈ H}  (1/m) Σᵢ σᵢ · h(xᵢ) ]               │
    │                                                                  │
    │  The expected correlation of H with random noise on S.           │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Intuition:  σᵢ are random ±1 labels (pure noise, no signal).
                R̂_S(H) measures how well the best h ∈ H can fit
                random noise on S.

                Large R̂_S(H) → H can fit noise → complex class.
                Small R̂_S(H) → H cannot fit noise → simple class.

    Rademacher-based generalisation bound:

        With probability ≥ 1 − δ, for all h ∈ H:

            R(h) ≤ R̂(h)  +  2·R_m(H)  +  √(log(1/δ) / 2m)

        where R_m(H) = E_S[R̂_S(H)] is the Rademacher complexity.

    Advantage over VC bounds: works for real-valued functions, is
    data-distribution-dependent (tighter for specific problems), and
    naturally handles function classes without finite VC dimension.

    Diagram 5 — Rademacher Intuition:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Data points:  x₁   x₂   x₃   x₄   x₅   x₆   x₇   x₈            │
    │  Random signs: +1   -1   +1   +1   -1   +1   -1   -1            │
    │  (pure noise — no real signal)                                  │
    │                                                                 │
    │  SIMPLE H (e.g. linear):  best h correlates with noise ≈ 0.2    │
    │  COMPLEX H (e.g. deep net): best h correlates ≈ 1.0             │
    │                                                                 │
    │  ─── Large correlation with noise ─── Rademacher complexity ─── │
    │  ─── implies larger generalisation gap ─────────────────────────│
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Rademacher Complexities of common function classes:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Linear classifiers (unit ball):   R_m(H) = O(√(1/m))            │
    │  Halfspaces (norm ≤ B, data ≤ r):  R_m(H) = rB/√m                │
    │  Decision trees depth d:           R_m(H) = O(√(d log m / m))    │
    │  k-NN:                             R_m(H) = O(1) — does not → 0  │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Structural Risk Minimisation — Choosing Model Complexity from Theory

Given a nested sequence of hypothesis classes H₁ ⊆ H₂ ⊆ ... ⊆ H_∞ with
increasing VC dimension d₁ < d₂ < ... < ∞:

    ERM inside each class:
        h_n = argmin_{h ∈ Hₙ} R̂(h)   (empirical risk minimiser in class n)

    The generalisation bound says (for each class):
        R(h_n) ≤ R̂(h_n) + ε_n(m, δ)

    where  ε_n = √[ (dₙ · log(2em/dₙ) + log(2n/δ)) / (2m) ]
           (the δ is adjusted by n for the union bound over all classes)

    Structural Risk Minimisation (SRM) selects:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  h_SRM = argmin_n  [ R̂(h_n) + ε_n(m, δ) ]                        │
    │                                                                  │
    │  Minimise TRAINING ERROR + COMPLEXITY PENALTY jointly.           │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 6 — SRM: Training Error, Complexity Penalty, and Total Bound:

    Error
      │
      │           ╲        Total bound = train error + complexity penalty
      │             ╲     /──────────────────────────────────────────────
      │               ╲  /  ← optimal class (minimum total)
      │               ▼/
      │ train error:    ╲_______________
      │                                 ╲_____ (decreasing with complexity)
      │
      │ complexity        ____________________/  (increasing with complexity)
      │ penalty:  _______/
      │
      └──────────────────────────────────────────── model complexity (d)
                                    ↑
                               SRM selects here

    SRM is the theoretical backbone of regularisation (Module 06):

        Tikhonov regularisation:   min R̂(h) + λ·||h||²
        Lasso:                     min R̂(h) + λ·||h||₁
        Early stopping:            implicit complexity control via
                                   number of gradient steps

    The regularisation parameter λ controls which hypothesis class
    we effectively operate in — larger λ → simpler class → more
    regularisation → moves right to left in Diagram 6.


──────────────────────────────────────────────────────────────────────────────
### The Double Descent Phenomenon

Classical learning theory predicts a U-shaped bias-variance curve:
as model complexity increases, test error first decreases (bias↓) then
increases (variance↑). This is the classical view.

Modern over-parameterised models (neural networks, random forests,
interpolating kernel machines) break this picture dramatically.

    Diagram 7 — Classical Bias-Variance vs Double Descent:

    Test
    Error
      │
      │  CLASSICAL:           DOUBLE DESCENT:
      │                        │
      │    ╱╲                  │  classical   │       │
      │   ╱  ╲    U-shape      │  regime      │under- │    over-param
      │  ╱    ╲                │              │param  │    regime
      │ ╱      ╲               │     ╲        │  zone │        ╲___
      │╱        ╲____          │      ╲_______|───────────────────────
      │                        │              │       │
      └──────────────          └──────────────┴───────┴───────────────
        d = 1, 2, ...           d → n   d = n    d > n (interpolation)
                                        ↑
                               interpolation threshold
                               (model just memorises training set)

    What happens at the interpolation threshold (d ≈ n)?

        The model has just enough capacity to fit the training data
        exactly (zero training error). This is the WORST point for
        test error — the model memorises every training example,
        including label noise. Variance is maximally large.

    What happens for d >> n (over-parameterised regime)?

        Counterintuitively, test error DECREASES again. The model
        still fits training data exactly, but gradient descent (or
        the minimum-norm solution) finds a "smooth" interpolating
        function by implicit regularisation.

        The implicit bias of SGD / gradient flow prefers:
        ─ Minimum-norm solutions (for linear models: pseudo-inverse)
        ─ Max-margin solutions (for SVM-like problems)
        ─ Low-rank solutions (via gradient descent on matrices)

    Why does classical theory miss this?

        Classical bounds are WORST-CASE over all hypotheses in H.
        For over-parameterised models, the ACTUAL solution found by
        gradient descent is not the worst-case — it is regularised
        by the optimisation algorithm itself. Theory is catching up.

    The Belkin-Hsu-Ma-Mandal (2019) experiment:

        Polynomial regression with increasing degree p:
        ─ p < n:   classical regime, U-shaped test error
        ─ p ≈ n:   interpolation threshold, test error spikes
        ─ p >> n:  over-parameterised regime, test error falls
                   below classical optimum!

    Diagram 8 — Minimum-Norm Interpolation in 1D:

    n = 5 data points with label noise:   ×₁  ×₂  ×₃  ×₄  ×₅

    Classical optimal (p = 3):             ── smooth cubic curve ──
    Interpolating (p = n = 5):             ──── perfect fit ────── (jagged)
    Over-param (p = 20):                   ── smooth curve through all ──
                                               (wiggly between points
                                               but minimum norm globally)

    Key insight: the over-parameterised model finds a solution that
    is globally smooth even while fitting every noisy point exactly,
    because the pseudo-inverse minimises the L₂ norm of the weights.


──────────────────────────────────────────────────────────────────────────────
### No-Free-Lunch from a Theoretical Perspective

The No-Free-Lunch theorem (Module 03) has a precise statement in the
learning theory framework:

    Theorem (NFL — formal):

        For any learning algorithm A, and any m < |X|/2:
        There exists a distribution D such that:
            ─ ∃ a classifier with R(h) = 0 on D.
            ─ With prob ≥ 1/7:  R(A(S)) ≥ 1/8.

        No algorithm can outperform random guessing in the worst case
        over all possible data distributions.

    The escape from NFL:  INDUCTIVE BIAS

        Every practical ML algorithm implicitly restricts to a hypothesis
        class H (finite VC dimension) OR uses prior knowledge (Bayesian
        priors) OR exploits structure in the distribution (smoothness,
        compositionality).

        Choosing a hypothesis class IS choosing an inductive bias.
        The choice of H is the most important modelling decision.

    ┌──────────────────────────────────────────────────────────────────┐
    │  NFL → without prior knowledge, learning is impossible.          │
    │  PAC → with the right hypothesis class, learning is guaranteed.  │
    │  VC  → the right hypothesis class has finite VC dimension.       │
    │  SRM → choose the right complexity for your sample size.         │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Practical Takeaways from Learning Theory

    The key quantities and what they tell you:

    ┌──────────────────────────────────────────────────────────────────┐
    │  QUANTITY            MEANING              PRACTICAL IMPLICATION  │
    ├──────────────────────────────────────────────────────────────────┤
    │  VCdim(H) = d        Effective complexity  More d → need more m  │
    │  m_H(ε, δ)           Sample complexity     Min data to generalise│
    │  R(h) − R̂(h)        Gen. gap              What val. set measures │
    │  Approx. error       Bias (irreducible)    Choose bigger H       │
    │  Estimation error    Variance (reducible)  Get more data         │
    │  SRM penalty ε_n     Theory regularisation Basis for λ tuning    │
    └──────────────────────────────────────────────────────────────────┘

    Rules of thumb derived from learning theory:

    ┌───────────────────────────────────────────────────────────────────┐
    │  1.  You need at least ~10× VCdim examples to generalise well.    │
    │      (More precisely: m ≥ C·d/ε² for constant-probability bound)  │
    │                                                                   │
    │  2.  Regularisation = implicit SRM. λ controls effective VCdim.   │
    │                                                                   │
    │  3.  Training error is always optimistic. Report validation error.│
    │                                                                   │
    │  4.  Cross-validation approximates the SRM bound empirically —    │
    │      it is model selection by estimating generalisation error.    │
    │                                                                   │
    │  5.  For over-parameterised models, classical theory fails.       │
    │      Implicit regularisation via optimisation dynamics matters.   │
    │                                                                   │
    │  6.  The double descent peak occurs near m ≈ d (interpolation     │
    │      threshold). Avoid this regime or cross it into over-param.   │ 
    └───────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · VC Dimension — Shattering, Growth Function, and Sauer-Shelah": {
        "description": (
            "Visualise the shattering concept for threshold classifiers and "
            "halfspaces. Compute the growth function empirically by counting "
            "distinct labellings producible on m random points. Compare to "
            "the Sauer-Shelah upper bound and 2^m. Demonstrates why finite "
            "VC dimension guarantees polynomial growth and thus learnability."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: count dichotomies produced by a hypothesis class on a random point set
# ─────────────────────────────────────────────────────────────────────────────

def growth_function_empirical(hypothesis_fn, m, n_trials=200, seed=0):
    """
    Estimate Π_H(m) by sampling n_trials sets of m random points
    and counting the distinct label vectors H can produce.
    """
    rng = np.random.default_rng(seed)
    max_dichotomies = 0
    for _ in range(n_trials):
        X = rng.uniform(-3, 3, (m, 2))
        dichotomies = set()
        for params in hypothesis_fn(X):
            labels = params
            dichotomies.add(tuple(labels))
        max_dichotomies = max(max_dichotomies, len(dichotomies))
    return max_dichotomies


def sauer_shelah_bound(m, d):
    """Σᵢ₌₀ᵈ C(m, i)  — Sauer-Shelah lemma."""
    from math import comb
    return min(sum(comb(m, i) for i in range(d + 1)), 2**m)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Class 1: Threshold classifiers on ℝ¹  (VCdim = 1)
#   h_{θ}(x) = 1 if x₁ ≥ θ  else 0
# ─────────────────────────────────────────────────────────────────────────────

def threshold_dichotomies(X):
    """All dichotomies of X[:,0] by threshold classifiers."""
    x1 = X[:, 0]
    m  = len(x1)
    thresholds = np.sort(np.unique(np.concatenate([x1 - 1e-6, x1 + 1e-6])))
    seen = set()
    for θ in thresholds:
        labels = tuple((x1 >= θ).astype(int))
        seen.add(labels)
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Class 2: Halfspaces in ℝ²  (VCdim = 3)
#   h_{w,b}(x) = sign(w₁x₁ + w₂x₂ + b)
# ─────────────────────────────────────────────────────────────────────────────

def halfspace_dichotomies(X, n_directions=800):
    """All dichotomies of X ⊂ ℝ² achievable by linear halfspaces."""
    rng  = np.random.default_rng(7)
    seen = set()
    angles = np.linspace(0, 2 * np.pi, n_directions)
    W = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    scores = X @ W.T                      # (m × n_directions)
    thresholds = np.linspace(scores.min() - 0.1, scores.max() + 0.1, 30)
    for j in range(n_directions):
        for θ in thresholds:
            labels = tuple((scores[:, j] >= θ).astype(int))
            seen.add(labels)
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Class 3: Axis-aligned rectangles in ℝ²  (VCdim = 4)
#   h_{a,b,c,d}(x) = 1 if a ≤ x₁ ≤ b and c ≤ x₂ ≤ d
# ─────────────────────────────────────────────────────────────────────────────

def rectangle_dichotomies(X, n_rects=1200):
    """Dichotomies of X ⊂ ℝ² achievable by axis-aligned rectangles."""
    rng  = np.random.default_rng(7)
    seen = set()
    x1, x2 = X[:, 0], X[:, 1]
    vals1   = np.sort(np.unique(np.r_[x1 - 1e-6, x1 + 1e-6, [-4, 4]]))
    vals2   = np.sort(np.unique(np.r_[x2 - 1e-6, x2 + 1e-6, [-4, 4]]))
    for _ in range(n_rects):
        a, b = sorted(rng.choice(vals1, 2, replace=False))
        c, d = sorted(rng.choice(vals2, 2, replace=False))
        labels = tuple(((x1 >= a) & (x1 <= b) & (x2 >= c) & (x2 <= d)).astype(int))
        seen.add(labels)
    # All-zero dichotomy (empty rectangle)
    seen.add(tuple(np.zeros(len(X), dtype=int)))
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Compute growth functions empirically
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  VC DIMENSION AND GROWTH FUNCTION ANALYSIS")
print("=" * 65)

m_values = list(range(1, 11))
vc_dims  = {"Threshold (d=1)": 1, "Halfspace (d=3)": 3, "Rectangle (d=4)": 4}
growth   = {"Threshold (d=1)": [], "Halfspace (d=3)": [], "Rectangle (d=4)": []}
sauer    = {"Threshold (d=1)": [], "Halfspace (d=3)": [], "Rectangle (d=4)": []}

print()
print(f"  {'m':>3}  {'2^m':>8}  {'Thresh(d=1)':>14}  {'Half(d=3)':>12}  {'Rect(d=4)':>12}")
print("  " + "─" * 57)

rng_fixed = np.random.default_rng(42)
for m in m_values:
    X_test = rng_fixed.uniform(-3, 3, (m, 2))

    g_thresh = len(threshold_dichotomies(X_test))
    g_half   = len(halfspace_dichotomies(X_test))
    g_rect   = len(rectangle_dichotomies(X_test))

    growth["Threshold (d=1)"].append(g_thresh)
    growth["Halfspace (d=3)"].append(g_half)
    growth["Rectangle (d=4)"].append(g_rect)

    s_thresh = sauer_shelah_bound(m, 1)
    s_half   = sauer_shelah_bound(m, 3)
    s_rect   = sauer_shelah_bound(m, 4)

    sauer["Threshold (d=1)"].append(s_thresh)
    sauer["Halfspace (d=3)"].append(s_half)
    sauer["Rectangle (d=4)"].append(s_rect)

    print(f"  {m:>3}  {2**m:>8}  {g_thresh:>6} (≤{s_thresh:>5})  "
          f"{g_half:>5} (≤{s_half:>5})  {g_rect:>5} (≤{s_rect:>5})")

print()
print("  KEY OBSERVATIONS:")
print("  - Threshold  (d=1): growth ≤ 2m (linear in m after m>1)")
print("  - Halfspace  (d=3): growth ≤ m³ (cubic in m after m>3)")
print("  - Rectangle  (d=4): growth ≤ m⁴ (quartic in m after m>4)")
print("  - All are POLYNOMIAL once m > VCdim — this is Sauer-Shelah.")
print("  - Compare to 2^m (exponential) — the gap grows rapidly.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Shattering demo: can 3 points be shattered by halfspaces?
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  SHATTERING CHECK: 3 points, halfspace in ℝ²")
print("=" * 65)
print()

# 3 non-collinear points
three_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
all_labellings = list(product([0, 1], repeat=3))
achieved = halfspace_dichotomies(three_points)

print(f"  3 non-collinear points: {three_points.tolist()}")
print(f"  Total possible labellings: 2³ = 8")
print(f"  Labellings achievable by halfspaces: {len(achieved)}")
print()
for lbl in all_labellings:
    achieved_flag = "✓" if lbl in achieved else "✗"
    print(f"    {list(lbl)}  {achieved_flag}")
print()
if len(achieved) == 8:
    print("  → All 8 labellings achieved → these 3 points are SHATTERED.")
    print("  → VCdim(halfspace in ℝ²) ≥ 3 (confirmed).")
else:
    print(f"  → Only {len(achieved)}/8 achieved → NOT shattered.")
print()

# Check 4 points — should NOT be shatterable
print("=" * 65)
print("  SHATTERING CHECK: 4 points, halfspace in ℝ² (should fail)")
print("=" * 65)
print()
four_points = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])
achieved4   = halfspace_dichotomies(four_points)
all4        = list(product([0,1], repeat=4))
print(f"  4 points (corners of unit square): {four_points.tolist()}")
print(f"  Total possible labellings: 2⁴ = 16")
print(f"  Labellings achievable by halfspaces: {len(achieved4)}")
missing = [lbl for lbl in all4 if lbl not in achieved4]
print(f"  Missing labellings ({len(missing)} not achievable):")
for lbl in missing:
    print(f"    {list(lbl)}  ← cannot be realised by any halfspace")
print()
print("  → Cannot shatter 4 points → VCdim(halfspace in ℝ²) < 4.")
print("  → Combined with d ≥ 3 above: VCdim = 3 for ℝ² (= d+1 = 2+1).")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("VC Dimension: Growth Functions and Sauer-Shelah Bounds",
             fontsize=13, fontweight="bold")

# Panel 1: Growth functions vs 2^m
ax = axes[0]
ax.plot(m_values, [2**m for m in m_values], "k--", lw=2, label="2^m (all dichotomies)")
colors = ["steelblue", "tomato", "seagreen"]
for (name, vals), col in zip(growth.items(), colors):
    ax.plot(m_values, vals, "o-", lw=2, ms=7, color=col,
            label=f"Empirical: {name}")
ax.set_yscale("log")
ax.set_xlabel("m (number of points)"); ax.set_ylabel("Number of dichotomies (log)")
ax.set_title("Growth Function Π_H(m)\\n(empirical vs 2^m)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# Panel 2: Empirical vs Sauer-Shelah bound
ax = axes[1]
for (name, emp), (_, sb), col in zip(growth.items(), sauer.items(), colors):
    d    = vc_dims[name]
    ax.fill_between(m_values, emp, sb, alpha=0.15, color=col)
    ax.plot(m_values, emp, "o-", lw=2, ms=6, color=col, label=f"Empirical: {name}")
    ax.plot(m_values, sb,  "s--", lw=1.5, ms=5, color=col, alpha=0.6,
            label=f"Sauer bound (d={d})")
ax.set_yscale("log")
ax.set_xlabel("m"); ax.set_ylabel("Dichotomies (log)")
ax.set_title("Empirical Growth vs Sauer-Shelah Bound\\n(shaded gap = slack in bound)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

# Panel 3: Shattering visualisation — 3 points + all 8 halfspace labellings
ax = axes[2]
ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.6)
ax.set_aspect("equal")

# The 8 achievable labellings of 3-point set
coords = three_points
colors_lbl = {0: "white", 1: "steelblue"}
for k, lbl in enumerate(all_labellings[:8]):
    ox = (k % 4) * 0.4 - 0.3
    oy = 1.3 - (k // 4) * 0.55
    for i, (pt, lab) in enumerate(zip(coords, lbl)):
        fc = "steelblue" if lab == 1 else "white"
        ax.scatter(ox + pt[0] * 0.3, oy + pt[1] * 0.3,
                   c=fc, s=80, edgecolors="steelblue", linewidths=1.5, zorder=5)
    ax.text(ox + 0.15, oy - 0.12, f"{''.join(map(str,lbl))}", ha="center",
            fontsize=7, color="dimgray")
ax.set_title(f"Shattering: all 8 labellings\\nof 3 non-collinear points by halfspaces",
             fontweight="bold")
ax.axis("off")

fig.text(0.5, 0.01,
         "Sauer-Shelah: Π_H(m) ≤ Σᵢ₌₀ᵈ C(m,i)  —  Polynomial growth once m > VCdim.",
         ha="center", fontsize=10, style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("vc_dimension.png", dpi=110)
print()
print("  Plot saved → vc_dimension.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Generalisation Bounds — Theory vs Empirical Gap": {
        "description": (
            "Empirically verify the VC and agnostic PAC bounds. Train "
            "polynomial classifiers of increasing degree on binary "
            "classification data and compare: training error, test error, "
            "the theoretical VC-based upper bound, and the actual "
            "generalisation gap. Shows how sample size m shrinks the "
            "bound and how VC dimension controls the gap growth."
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
from itertools import combinations_with_replacement as _cwr

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree; self.include_bias = include_bias
    def fit(self, X, y=None):
        _, p = X.shape
        self._combos = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            self._combos.extend(_cwr(range(p), d))
        return self
    def transform(self, X):
        n = X.shape[0]; out = _np_impl.ones((n, len(self._combos)))
        for j, combo in enumerate(self._combos):
            for idx in combo: out[:, j] *= X[:, idx]
        return out
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):           return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class Ridge:
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        n, d = X.shape
        self.coef_ = _np_impl.linalg.solve(X.T @ X + self.alpha * _np_impl.eye(d), X.T @ y)
        self.intercept_ = 0.0; return self
    def predict(self, X): return X @ self.coef_
    def score(self, X, y):
        ss_res = _np_impl.sum((y - self.predict(X))**2)
        ss_tot = _np_impl.sum((y - y.mean())**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _sig(z): return 1.0 / (1.0 + _np_impl.exp(-_np_impl.clip(z, -500, 500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=2000, solver='lbfgs'):
        self.C = C; self.max_iter = max_iter
    def fit(self, X, y):
        n, d = X.shape; lam = 1.0 / (self.C * n)
        def fg(w):
            p = _np_impl.clip(_sig(X @ w[1:] + w[0]), 1e-10, 1-1e-10)
            loss = -_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)) + 0.5*lam*_np_impl.sum(w[1:]**2)
            e = p - y
            return loss, _np_impl.r_[e.mean(), X.T @ e / n + lam * w[1:]]
        res = _sp_opt.minimize(fg, _np_impl.zeros(d+1), method='L-BFGS-B', jac=True,
                               options={'maxiter': self.max_iter})
        self.coef_ = res.x[1:].reshape(1,-1); self.intercept_ = _np_impl.array([res.x[0]]); return self
    def predict_proba(self, X):
        p = _sig(X @ self.coef_.ravel() + self.intercept_[0]); return _np_impl.c_[1-p, p]
    def predict(self, X): return (self.predict_proba(X)[:,1] >= 0.5).astype(int)
    def score(self, X, y): return (self.predict(X) == y).mean()

class Pipeline:
    def __init__(self, steps): self.steps = steps; self.named_steps = dict(steps)
    def _Xt(self, X):
        Xt = X
        for _, s in self.steps[:-1]: Xt = s.transform(Xt)
        return Xt
    def fit(self, X, y=None):
        Xt = X
        for _, s in self.steps[:-1]: Xt = s.fit_transform(Xt, y)
        self.steps[-1][1].fit(Xt, y); return self
    def predict(self, X):    return self.steps[-1][1].predict(self._Xt(X))
    def score(self, X, y):   return self.steps[-1][1].score(self._Xt(X), y)

def train_test_split(*arrays, test_size=0.25, random_state=None, train_size=None):
    rng = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = train_size if train_size is not None else int(n * (1 - test_size))
    idx = rng.permutation(n); tr_i, te_i = idx[:n_tr], idx[n_tr:]
    out = []
    for a in arrays: out += [a[tr_i], a[te_i]]
    return out

def cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_squared_error'):
    n = len(X); idx = _np_impl.arange(n); fs = n // cv; scores = []
    for k in range(cv):
        vi = idx[k*fs:(k+1)*fs]
        ti = _np_impl.concatenate([idx[:k*fs], idx[(k+1)*fs:]])
        est = _copy.deepcopy(estimator); est.fit(X[ti], y[ti])
        scores.append(-_np_impl.mean((est.predict(X[vi]) - y[vi])**2)
                      if scoring == 'neg_mean_squared_error'
                      else est.score(X[vi], y[vi]))
    return _np_impl.array(scores)

def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng = _np_impl.random.default_rng(random_state); n = n_samples // 2
    t = _np_impl.linspace(0, _np_impl.pi, n)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t), _np_impl.sin(t)],
                         _np_impl.c_[1-_np_impl.cos(t), -_np_impl.sin(t)+0.5]])
    X += rng.normal(0, noise, X.shape)
    y = _np_impl.hstack([_np_impl.zeros(n), _np_impl.ones(n_samples-n)]).astype(int)
    return X, y

from math import comb, log, sqrt

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Generalisation bound calculations
# ─────────────────────────────────────────────────────────────────────────────

def vc_bound(train_err, m, vc_dim, delta=0.05):
    """
    Agnostic VC bound:
    R(h) ≤ R̂(h) + sqrt[ (d·log(2em/d) + log(2/δ)) / (2m) ]
    """
    if vc_dim <= 0 or m <= vc_dim:
        return 1.0
    import math
    penalty = sqrt((vc_dim * log(2 * math.e * m / vc_dim) + log(2 / delta)) / (2 * m))
    return min(train_err + penalty, 1.0)


def finite_h_bound(train_err, m, h_size, delta=0.05):
    """
    Agnostic finite-H bound:
    R(h) ≤ R̂(h) + sqrt[ log(2|H|/δ) / (2m) ]
    """
    import math
    penalty = sqrt(log(2 * h_size / delta) / (2 * m))
    return min(train_err + penalty, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Polynomial degree vs generalisation gap
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  GENERALISATION BOUNDS: POLYNOMIAL CLASSIFIERS")
print("=" * 65)

# Generate data
X_raw, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.4, random_state=0)
m = len(X_tr)  # training size

degrees   = list(range(1, 9))
train_errs = []
test_errs  = []
vc_bounds_ = []
gen_gaps   = []

print()
print(f"  Training samples m = {m}, Test samples = {len(X_te)}")
print()
print(f"  {'Deg':>4}  {'VCdim':>7}  {'Train err':>10}  {'Test err':>10}  "
      f"{'VC bound':>10}  {'Gen gap':>9}  {'Bound gap':>10}")
print("  " + "─" * 69)

for deg in degrees:
    # VC dim of polynomial classifier in ℝ² with degree p:
    # number of monomials = C(p+2, 2) features → halfspace in that space
    n_features = comb(deg + 2, 2)  # number of poly features in ℝ²
    vc_d = n_features              # halfspace VC dim = n_features + 1 ≈ n_features

    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg, include_bias=True)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1e4, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(X_tr, y_tr)
    train_err = 1 - pipe.score(X_tr, y_tr)
    test_err  = 1 - pipe.score(X_te, y_te)
    vb        = vc_bound(train_err, m, vc_d)
    gap       = test_err - train_err
    bound_gap = vb - test_err

    train_errs.append(train_err)
    test_errs.append(test_err)
    vc_bounds_.append(vb)
    gen_gaps.append(gap)

    print(f"  {deg:>4}  {vc_d:>7}  {train_err:>10.4f}  {test_err:>10.4f}  "
          f"{vb:>10.4f}  {gap:>+9.4f}  {bound_gap:>10.4f}")

print()
print("  KEY OBSERVATIONS:")
print("  - Training error decreases monotonically with degree (more expressive).")
print("  - Test error has a U-shape: decreases then increases (bias-variance).")
print("  - VC bound is conservative but always valid (test err ≤ VC bound).")
print("  - Generalisation gap grows with degree (higher complexity ↔ higher variance).")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Effect of sample size m on bounds
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  EFFECT OF SAMPLE SIZE ON GENERALISATION BOUND (degree=3)")
print("=" * 65)
print()

# Use degree=3 polynomial classifier (VC dim fixed)
deg_fixed  = 3
n_feats_d3 = comb(deg_fixed + 2, 2)

X_all, y_all = make_moons(n_samples=2000, noise=0.2, random_state=0)
m_vals   = [30, 50, 100, 150, 250, 500, 800, 1200]
train_m  = []
test_m   = []
bound_m  = []

print(f"  Fixed degree = {deg_fixed}  →  VCdim ≈ {n_feats_d3}")
print()
print(f"  {'m':>5}  {'Train err':>10}  {'Test err':>10}  {'VC bound':>10}  "
      f"{'Bound slack':>12}  {'Gen gap':>9}")
print("  " + "─" * 62)

for m_val in m_vals:
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
        X_all, y_all, train_size=m_val, random_state=1)
    pipe_m = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg_fixed)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=100, max_iter=2000)),
    ])
    pipe_m.fit(X_tr_m, y_tr_m)
    te  = 1 - pipe_m.score(X_tr_m, y_tr_m)
    ee  = 1 - pipe_m.score(X_te_m, y_te_m)
    vb  = vc_bound(te, m_val, n_feats_d3)
    train_m.append(te); test_m.append(ee); bound_m.append(vb)
    print(f"  {m_val:>5}  {te:>10.4f}  {ee:>10.4f}  {vb:>10.4f}  "
          f"{vb - ee:>12.4f}  {ee - te:>+9.4f}")

print()
print("  - VC bound tightens as m grows (penalty ∝ 1/√m).")
print("  - Actual generalisation gap also shrinks with more data.")
print("  - At large m, training and test errors converge.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("PAC Generalisation Bounds: Theory vs Empirical Reality",
             fontsize=13, fontweight="bold")

# Panel 1: Error vs polynomial degree
ax = axes[0, 0]
ax.plot(degrees, train_errs, "b-o", lw=2, ms=8, label="Training error")
ax.plot(degrees, test_errs,  "r-o", lw=2, ms=8, label="Test error")
ax.plot(degrees, vc_bounds_, "g--s", lw=2, ms=7, label="VC upper bound")
ax.fill_between(degrees, test_errs, vc_bounds_, alpha=0.12, color="green",
                label="Bound slack")
ax.fill_between(degrees, train_errs, test_errs, alpha=0.12, color="red",
                label="Gen gap")
ax.set_xlabel("Polynomial degree"); ax.set_ylabel("Error rate")
ax.set_title("Error vs Model Complexity\\n(VC bound always above test error)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: Generalisation gap vs degree
ax = axes[0, 1]
vc_penalty = [v - t for v, t in zip(vc_bounds_, train_errs)]
ax.bar(degrees, gen_gaps,   color="tomato", alpha=0.7, label="Actual gen gap (test−train)")
ax.bar(degrees, vc_penalty, color="steelblue", alpha=0.4, label="VC complexity penalty",
       bottom=train_errs)
ax.set_xlabel("Polynomial degree"); ax.set_ylabel("Error contribution")
ax.set_title("Decomposition: Training Error + VC Penalty\\nActual gap must fit inside VC penalty",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: VC bound tightening with m
ax = axes[1, 0]
ax.plot(m_vals, train_m,  "b-o", lw=2, ms=8, label="Training error")
ax.plot(m_vals, test_m,   "r-o", lw=2, ms=8, label="Test error")
ax.plot(m_vals, bound_m,  "g--s", lw=2, ms=7, label="VC bound (degree=3)")
ax.fill_between(m_vals, test_m, bound_m, alpha=0.12, color="green")
ax.set_xlabel("Training set size m")
ax.set_ylabel("Error rate")
ax.set_title("VC Bound Tightens with More Data\\n(penalty ∝ √(d/m))", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 4: Decision boundary for best vs worst degree
ax = axes[1, 1]
xx, yy = np.meshgrid(np.linspace(-2.5, 3.0, 200), np.linspace(-1.5, 2.0, 200))
Xg = np.c_[xx.ravel(), yy.ravel()]

best_deg  = degrees[int(np.argmin(test_errs))]
worst_deg = degrees[-1]

for i_sub, (deg_plot, colour, lbl) in enumerate([
        (best_deg,  "steelblue", f"Optimal deg={best_deg}"),
        (worst_deg, "tomato",    f"Overfit deg={worst_deg}")]):
    pipe_plot = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg_plot)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1e4, max_iter=2000)),
    ])
    pipe_plot.fit(X_tr, y_tr)
    Z = pipe_plot.predict(Xg).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.15 if i_sub == 0 else 0.1,
                colors=[colour], levels=[0.5, 1.5])
    ax.contour(xx, yy, Z, colors=[colour], linewidths=1.5, levels=[0.5])

ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], c="steelblue", s=20, alpha=0.6,
           marker="o", label="Class 0 (train)")
ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], c="tomato",    s=20, alpha=0.6,
           marker="^", label="Class 1 (train)")
ax.set_title(f"Decision Boundaries\\nBlue=deg {best_deg} (optimal)  "
             f"Red=deg {worst_deg} (overfit)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("generalisation_bounds.png", dpi=110)
print()
print("  Plot saved → generalisation_bounds.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Structural Risk Minimisation vs Cross-Validation": {
        "description": (
            "Compare two approaches to model selection: SRM (uses the "
            "theoretical VC-based penalty to pick complexity) and k-fold "
            "cross-validation (uses held-out data empirically). Shows "
            "that both converge to similar choices but via different "
            "mechanisms — theory vs data. Includes full SRM decomposition: "
            "approximation error, estimation error, and the complexity penalty."
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
from itertools import combinations_with_replacement as _cwr

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree; self.include_bias = include_bias
    def fit(self, X, y=None):
        _, p = X.shape
        self._combos = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            self._combos.extend(_cwr(range(p), d))
        return self
    def transform(self, X):
        n = X.shape[0]; out = _np_impl.ones((n, len(self._combos)))
        for j, combo in enumerate(self._combos):
            for idx in combo: out[:, j] *= X[:, idx]
        return out
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):           return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class Ridge:
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        n, d = X.shape
        self.coef_ = _np_impl.linalg.solve(X.T @ X + self.alpha * _np_impl.eye(d), X.T @ y)
        self.intercept_ = 0.0; return self
    def predict(self, X): return X @ self.coef_
    def score(self, X, y):
        ss_res = _np_impl.sum((y - self.predict(X))**2)
        ss_tot = _np_impl.sum((y - y.mean())**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _sig(z): return 1.0 / (1.0 + _np_impl.exp(-_np_impl.clip(z, -500, 500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=2000, solver='lbfgs'):
        self.C = C; self.max_iter = max_iter
    def fit(self, X, y):
        n, d = X.shape; lam = 1.0 / (self.C * n)
        def fg(w):
            p = _np_impl.clip(_sig(X @ w[1:] + w[0]), 1e-10, 1-1e-10)
            loss = -_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)) + 0.5*lam*_np_impl.sum(w[1:]**2)
            e = p - y
            return loss, _np_impl.r_[e.mean(), X.T @ e / n + lam * w[1:]]
        res = _sp_opt.minimize(fg, _np_impl.zeros(d+1), method='L-BFGS-B', jac=True,
                               options={'maxiter': self.max_iter})
        self.coef_ = res.x[1:].reshape(1,-1); self.intercept_ = _np_impl.array([res.x[0]]); return self
    def predict_proba(self, X):
        p = _sig(X @ self.coef_.ravel() + self.intercept_[0]); return _np_impl.c_[1-p, p]
    def predict(self, X): return (self.predict_proba(X)[:,1] >= 0.5).astype(int)
    def score(self, X, y): return (self.predict(X) == y).mean()

class Pipeline:
    def __init__(self, steps): self.steps = steps; self.named_steps = dict(steps)
    def _Xt(self, X):
        Xt = X
        for _, s in self.steps[:-1]: Xt = s.transform(Xt)
        return Xt
    def fit(self, X, y=None):
        Xt = X
        for _, s in self.steps[:-1]: Xt = s.fit_transform(Xt, y)
        self.steps[-1][1].fit(Xt, y); return self
    def predict(self, X):    return self.steps[-1][1].predict(self._Xt(X))
    def score(self, X, y):   return self.steps[-1][1].score(self._Xt(X), y)

def train_test_split(*arrays, test_size=0.25, random_state=None, train_size=None):
    rng = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = train_size if train_size is not None else int(n * (1 - test_size))
    idx = rng.permutation(n); tr_i, te_i = idx[:n_tr], idx[n_tr:]
    out = []
    for a in arrays: out += [a[tr_i], a[te_i]]
    return out

def cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_squared_error'):
    n = len(X); idx = _np_impl.arange(n); fs = n // cv; scores = []
    for k in range(cv):
        vi = idx[k*fs:(k+1)*fs]
        ti = _np_impl.concatenate([idx[:k*fs], idx[(k+1)*fs:]])
        est = _copy.deepcopy(estimator); est.fit(X[ti], y[ti])
        scores.append(-_np_impl.mean((est.predict(X[vi]) - y[vi])**2)
                      if scoring == 'neg_mean_squared_error'
                      else est.score(X[vi], y[vi]))
    return _np_impl.array(scores)

def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng = _np_impl.random.default_rng(random_state); n = n_samples // 2
    t = _np_impl.linspace(0, _np_impl.pi, n)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t), _np_impl.sin(t)],
                         _np_impl.c_[1-_np_impl.cos(t), -_np_impl.sin(t)+0.5]])
    X += rng.normal(0, noise, X.shape)
    y = _np_impl.hstack([_np_impl.zeros(n), _np_impl.ones(n_samples-n)]).astype(int)
    return X, y

from math import comb, log, sqrt

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset: noisy sinusoidal function (true function has low complexity)
# ─────────────────────────────────────────────────────────────────────────────

def true_fn(x):
    return np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)

n_total = 120
X_all   = np.sort(np.random.uniform(0, 1, n_total)).reshape(-1, 1)
y_all   = true_fn(X_all.ravel()) + np.random.normal(0, 0.3, n_total)

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.4, random_state=1)
m = len(X_tr)

# ─────────────────────────────────────────────────────────────────────────────
# VC-based SRM: for regression, use effective degrees of freedom as proxy for d
# SRM bound (regression, squared loss, simplified):
#   bound = train_err + C * sqrt(d * log(m/d) / m)
# ─────────────────────────────────────────────────────────────────────────────

def srm_penalty(m, d, delta=0.05, C_const=2.5):
    """Simplified SRM penalty for regression (heuristic constant C)."""
    if d <= 0 or d >= m:
        return np.inf
    return C_const * sqrt((d * log(m / d) + log(1 / delta)) / m)


print("=" * 65)
print("  STRUCTURAL RISK MINIMISATION vs CROSS-VALIDATION")
print("=" * 65)
print()
print(f"  Data: sin(2πx) + noise.  Train m={m}, Test={len(X_te)}")
print()
print(f"  {'Deg':>4}  {'d (params)':>11}  {'Train MSE':>10}  {'Test MSE':>10}  "
      f"{'SRM bound':>10}  {'CV score':>10}  {'SRM pick':>9}  {'CV pick':>7}")
print("  " + "─" * 80)

degrees   = list(range(1, 14))
results   = {}

for deg in degrees:
    n_params = deg + 1    # polynomial with degree d has d+1 coefficients
    d_vc     = n_params   # VC dim proxy

    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1e-4)),
    ])
    pipe.fit(X_tr, y_tr)

    train_mse = np.mean((pipe.predict(X_tr) - y_tr) ** 2)
    test_mse  = np.mean((pipe.predict(X_te) - y_te) ** 2)
    penalty   = srm_penalty(m, d_vc)
    srm_val   = train_mse + penalty

    # 5-fold CV on training set
    cv_neg_mse = cross_val_score(pipe, X_tr, y_tr, cv=5,
                                  scoring="neg_mean_squared_error")
    cv_mse     = -cv_neg_mse.mean()

    results[deg] = {
        "n_params": n_params, "train": train_mse, "test": test_mse,
        "srm": srm_val, "penalty": penalty, "cv": cv_mse,
    }
    print(f"  {deg:>4}  {n_params:>11}  {train_mse:>10.4f}  {test_mse:>10.4f}  "
          f"{srm_val:>10.4f}  {cv_mse:>10.4f}", end="")
    print()

# Identify selections
train_vals = {d: r["train"] for d, r in results.items()}
srm_vals   = {d: r["srm"]   for d, r in results.items()}
cv_vals    = {d: r["cv"]    for d, r in results.items()}
test_vals  = {d: r["test"]  for d, r in results.items()}

best_srm = min(srm_vals, key=srm_vals.get)
best_cv  = min(cv_vals,  key=cv_vals.get)
best_test = min(test_vals, key=test_vals.get)

print()
print(f"  SRM selects:             degree = {best_srm}  (test MSE = {test_vals[best_srm]:.4f})")
print(f"  CV  selects:             degree = {best_cv}   (test MSE = {test_vals[best_cv]:.4f})")
print(f"  Oracle (true best):      degree = {best_test}  (test MSE = {test_vals[best_test]:.4f})")
print()
print("  INTERPRETATION:")
print("  - SRM and CV typically agree within ±1 degree.")
print("  - SRM is distribution-free (worst case); CV uses data empirically.")
print("  - Both outperform naive train-error minimisation (which over-selects).")

# ─────────────────────────────────────────────────────────────────────────────
# Approximation vs Estimation Error breakdown
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  BIAS-COMPLEXITY DECOMPOSITION (Approximation vs Estimation Error)")
print("=" * 65)
print()

# Best-in-class approximation (lower bound by fitting on huge test set)
X_huge = np.linspace(0, 1, 2000).reshape(-1, 1)
y_huge = true_fn(X_huge.ravel())

approx_errors = {}
for deg in degrees:
    pipe_app = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1e-8)),
    ])
    pipe_app.fit(X_huge, y_huge)
    approx_errors[deg] = np.mean((pipe_app.predict(X_huge) - y_huge) ** 2)

print(f"  {'Deg':>4}  {'Approx err':>12}  {'Estim err':>12}  {'Total test':>12}")
print("  " + "─" * 46)
for deg in degrees:
    approx = approx_errors[deg]
    estim  = max(results[deg]["test"] - approx, 0)
    total  = results[deg]["test"]
    print(f"  {deg:>4}  {approx:>12.5f}  {estim:>12.5f}  {total:>12.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Structural Risk Minimisation vs Cross-Validation",
             fontsize=13, fontweight="bold")

d_list    = list(degrees)
train_arr = [results[d]["train"]   for d in d_list]
test_arr  = [results[d]["test"]    for d in d_list]
srm_arr   = [results[d]["srm"]     for d in d_list]
cv_arr    = [results[d]["cv"]      for d in d_list]
pen_arr   = [results[d]["penalty"] for d in d_list]

# Panel 1: Train, test, SRM bound, CV
ax = axes[0, 0]
ax.plot(d_list, train_arr, "b-o", lw=2, ms=7, label="Training MSE")
ax.plot(d_list, test_arr,  "r-o", lw=2, ms=7, label="Test MSE")
ax.plot(d_list, srm_arr,   "g--s",lw=2, ms=7, label="SRM bound (train + penalty)")
ax.plot(d_list, cv_arr,    "m--^",lw=2, ms=7, label="CV estimate (5-fold)")
ax.axvline(best_srm, color="green", lw=1.5, ls=":", alpha=0.7, label=f"SRM pick (d={best_srm})")
ax.axvline(best_cv,  color="purple",lw=1.5, ls=":", alpha=0.7, label=f"CV pick  (d={best_cv})")
ax.set_xlabel("Polynomial degree"); ax.set_ylabel("MSE")
ax.set_title("SRM vs CV Model Selection\\n(both find near-optimal complexity)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, min(2.5, ax.get_ylim()[1]))

# Panel 2: SRM decomposition: train error + complexity penalty
ax = axes[0, 1]
ax.bar(d_list, train_arr, color="steelblue", alpha=0.8, label="Training MSE")
ax.bar(d_list, pen_arr, bottom=train_arr, color="tomato", alpha=0.7,
       label="SRM complexity penalty")
ax.plot(d_list, test_arr, "k-o", lw=2, ms=6, label="Test MSE (truth)")
ax.axvline(best_srm, color="tomato", lw=2, ls="--", label=f"SRM selects d={best_srm}")
ax.set_xlabel("Polynomial degree"); ax.set_ylabel("MSE")
ax.set_title("SRM Decomposition\\nTrain error + complexity penalty = upper bound",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, min(3.0, ax.get_ylim()[1]))

# Panel 3: Approximation vs Estimation error
ax = axes[1, 0]
approx_arr = [approx_errors[d] for d in d_list]
estim_arr  = [max(results[d]["test"] - approx_errors[d], 0) for d in d_list]
ax.stackplot(d_list, approx_arr, estim_arr, labels=["Approximation error (bias)",
             "Estimation error (variance)"],
             colors=["steelblue", "tomato"], alpha=0.7)
ax.plot(d_list, test_arr, "k-o", lw=2, ms=6, label="Total test MSE")
ax.set_xlabel("Polynomial degree"); ax.set_ylabel("MSE")
ax.set_title("Bias-Complexity Decomposition\\nApprox (↓ with complexity) + Estim (↑ with complexity)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 4: Fitted curves at different degrees
ax = axes[1, 1]
x_plot = np.linspace(0, 1, 300).reshape(-1, 1)
ax.scatter(X_tr, y_tr, c="steelblue", s=20, alpha=0.6, zorder=5, label="Train data")
ax.scatter(X_te, y_te, c="tomato",    s=20, alpha=0.4, zorder=5, marker="^", label="Test data")
ax.plot(x_plot, true_fn(x_plot.ravel()), "k-", lw=2, label="True function", zorder=6)

plot_degrees = [2, best_srm, best_cv, 12]
colours_plot = ["purple", "green", "orange", "red"]
styles       = ["--", "-", "-.", ":"]
for deg_pl, col, sty in zip(plot_degrees, colours_plot, styles):
    pipe_pl = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg_pl)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1e-4)),
    ])
    pipe_pl.fit(X_tr, y_tr)
    ax.plot(x_plot, pipe_pl.predict(x_plot), color=col, lw=2,
            linestyle=sty, label=f"deg={deg_pl}")

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Fitted Functions: SRM and CV Select Near-Optimal\\nLow degree: underfit, high degree: overfit",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig("srm_vs_cv.png", dpi=110)
print()
print("  Plot saved → srm_vs_cv.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Double Descent — The Modern Bias-Variance Breakdown": {
        "description": (
            "Simulate and visualise the double descent phenomenon using "
            "polynomial regression and random feature regression. Shows "
            "the classical U-shaped bias-variance curve, the interpolation "
            "threshold where test error peaks, and the second descent in "
            "the over-parameterised regime. Includes the minimum-norm "
            "interpolation explanation and comparison with classical "
            "regularised models."
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
from itertools import combinations_with_replacement as _cwr

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree; self.include_bias = include_bias
    def fit(self, X, y=None):
        _, p = X.shape
        self._combos = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            self._combos.extend(_cwr(range(p), d))
        return self
    def transform(self, X):
        n = X.shape[0]; out = _np_impl.ones((n, len(self._combos)))
        for j, combo in enumerate(self._combos):
            for idx in combo: out[:, j] *= X[:, idx]
        return out
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):           return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class Ridge:
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        n, d = X.shape
        self.coef_ = _np_impl.linalg.solve(X.T @ X + self.alpha * _np_impl.eye(d), X.T @ y)
        self.intercept_ = 0.0; return self
    def predict(self, X): return X @ self.coef_
    def score(self, X, y):
        ss_res = _np_impl.sum((y - self.predict(X))**2)
        ss_tot = _np_impl.sum((y - y.mean())**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _sig(z): return 1.0 / (1.0 + _np_impl.exp(-_np_impl.clip(z, -500, 500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=2000, solver='lbfgs'):
        self.C = C; self.max_iter = max_iter
    def fit(self, X, y):
        n, d = X.shape; lam = 1.0 / (self.C * n)
        def fg(w):
            p = _np_impl.clip(_sig(X @ w[1:] + w[0]), 1e-10, 1-1e-10)
            loss = -_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)) + 0.5*lam*_np_impl.sum(w[1:]**2)
            e = p - y
            return loss, _np_impl.r_[e.mean(), X.T @ e / n + lam * w[1:]]
        res = _sp_opt.minimize(fg, _np_impl.zeros(d+1), method='L-BFGS-B', jac=True,
                               options={'maxiter': self.max_iter})
        self.coef_ = res.x[1:].reshape(1,-1); self.intercept_ = _np_impl.array([res.x[0]]); return self
    def predict_proba(self, X):
        p = _sig(X @ self.coef_.ravel() + self.intercept_[0]); return _np_impl.c_[1-p, p]
    def predict(self, X): return (self.predict_proba(X)[:,1] >= 0.5).astype(int)
    def score(self, X, y): return (self.predict(X) == y).mean()

class Pipeline:
    def __init__(self, steps): self.steps = steps; self.named_steps = dict(steps)
    def _Xt(self, X):
        Xt = X
        for _, s in self.steps[:-1]: Xt = s.transform(Xt)
        return Xt
    def fit(self, X, y=None):
        Xt = X
        for _, s in self.steps[:-1]: Xt = s.fit_transform(Xt, y)
        self.steps[-1][1].fit(Xt, y); return self
    def predict(self, X):    return self.steps[-1][1].predict(self._Xt(X))
    def score(self, X, y):   return self.steps[-1][1].score(self._Xt(X), y)

def train_test_split(*arrays, test_size=0.25, random_state=None, train_size=None):
    rng = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = train_size if train_size is not None else int(n * (1 - test_size))
    idx = rng.permutation(n); tr_i, te_i = idx[:n_tr], idx[n_tr:]
    out = []
    for a in arrays: out += [a[tr_i], a[te_i]]
    return out

def cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_squared_error'):
    n = len(X); idx = _np_impl.arange(n); fs = n // cv; scores = []
    for k in range(cv):
        vi = idx[k*fs:(k+1)*fs]
        ti = _np_impl.concatenate([idx[:k*fs], idx[(k+1)*fs:]])
        est = _copy.deepcopy(estimator); est.fit(X[ti], y[ti])
        scores.append(-_np_impl.mean((est.predict(X[vi]) - y[vi])**2)
                      if scoring == 'neg_mean_squared_error'
                      else est.score(X[vi], y[vi]))
    return _np_impl.array(scores)

def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng = _np_impl.random.default_rng(random_state); n = n_samples // 2
    t = _np_impl.linspace(0, _np_impl.pi, n)
    X = _np_impl.vstack([_np_impl.c_[_np_impl.cos(t), _np_impl.sin(t)],
                         _np_impl.c_[1-_np_impl.cos(t), -_np_impl.sin(t)+0.5]])
    X += rng.normal(0, noise, X.shape)
    y = _np_impl.hstack([_np_impl.zeros(n), _np_impl.ones(n_samples-n)]).astype(int)
    return X, y


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# True function and noisy data generation
# ─────────────────────────────────────────────────────────────────────────────

def true_fn(x):
    return np.sin(3 * np.pi * x) * np.exp(-0.5 * x)

noise_std = 0.4
n_train   = 40
n_test    = 500

X_tr = np.sort(np.random.uniform(0, 2, n_train))
y_tr = true_fn(X_tr) + np.random.normal(0, noise_std, n_train)

X_te = np.linspace(0, 2, n_test)
y_te = true_fn(X_te) + np.random.normal(0, noise_std, n_test)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Polynomial regression — double descent curve
# ─────────────────────────────────────────────────────────────────────────────

def fit_poly_min_norm(X_tr, y_tr, X_te, y_te, deg):
    """
    Fit a polynomial of given degree.
    For d < n: Ridge (tiny alpha) ≈ OLS.
    For d >= n: minimum norm solution via pseudo-inverse.
    """
    scaler = StandardScaler()
    phi_tr = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(
                 X_tr.reshape(-1, 1))
    phi_te = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(
                 X_te.reshape(-1, 1))
    phi_tr = scaler.fit_transform(phi_tr)
    phi_te = scaler.transform(phi_te)

    n, p = phi_tr.shape
    if p <= n:
        # Under-parameterised: use Ridge with tiny regularisation
        alpha = 1e-8
        A = phi_tr.T @ phi_tr + alpha * np.eye(p)
        w = np.linalg.solve(A, phi_tr.T @ y_tr)
    else:
        # Over-parameterised: minimum-norm interpolation (pseudo-inverse)
        # w = Φᵀ (ΦΦᵀ)⁻¹ y  (minimum L2-norm solution that fits exactly)
        alpha = 1e-8
        A = phi_tr @ phi_tr.T + alpha * np.eye(n)
        w = phi_tr.T @ np.linalg.solve(A, y_tr)

    y_hat_tr = phi_tr @ w
    y_hat_te = phi_te @ w
    train_mse = np.mean((y_hat_tr - y_tr) ** 2)
    test_mse  = np.mean((y_hat_te - y_te) ** 2)
    return train_mse, test_mse, w

print("=" * 65)
print("  DOUBLE DESCENT: POLYNOMIAL REGRESSION")
print(f"  n_train = {n_train}   noise σ = {noise_std}")
print("=" * 65)
print()
print(f"  {'Degree':>7}  {'# params':>9}  {'p/n ratio':>10}  "
      f"{'Train MSE':>10}  {'Test MSE':>10}  {'Regime':>20}")
print("  " + "─" * 72)

degrees_dd = list(range(1, 80))
train_mse_dd = []
test_mse_dd  = []

for deg in degrees_dd:
    n_params = deg + 1
    t_err, e_err, _ = fit_poly_min_norm(X_tr, y_tr, X_te, y_te, deg)
    train_mse_dd.append(t_err)
    test_mse_dd.append(e_err)
    ratio = n_params / n_train
    if n_params < n_train:
        regime = "under-param (classic)"
    elif n_params == n_train:
        regime = "← interpolation"
    elif n_params <= n_train + 3:
        regime = "near-interpolation"
    else:
        regime = "over-param"
    if deg <= 6 or n_params in range(n_train - 2, n_train + 5) or deg % 10 == 0:
        print(f"  {deg:>7}  {n_params:>9}  {ratio:>10.2f}  "
              f"{t_err:>10.4f}  {e_err:>10.4f}  {regime:>20}")

print()
print("  KEY DOUBLE DESCENT OBSERVATIONS:")
interp_idx = n_train - 1  # degree where params = n_train
print(f"  - Interpolation threshold at degree ≈ {interp_idx} (n_params = n_train = {n_train})")
print(f"  - Test MSE at threshold: {test_mse_dd[interp_idx-1]:.4f}  ← PEAK (worst)")
print(f"  - Test MSE at deg=3 (classical optimal): {test_mse_dd[2]:.4f}")
print(f"  - Test MSE at deg=70 (over-param): {test_mse_dd[69]:.4f}")
print()
print("  - Over-parameterised min-norm solution finds a SMOOTHER fit")
print("    than the interpolation-threshold model, even with zero training error.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Effect of noise on double descent
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  EFFECT OF NOISE LEVEL ON DOUBLE DESCENT SEVERITY")
print("=" * 65)
print()

noise_levels = [0.1, 0.3, 0.6, 1.0]
dd_by_noise  = {}

for sigma in noise_levels:
    y_tr_n = true_fn(X_tr) + np.random.normal(0, sigma, n_train)
    y_te_n = true_fn(X_te) + np.random.normal(0, sigma, n_test)
    tmse_n, emse_n = [], []
    for deg in degrees_dd:
        t_, e_, _ = fit_poly_min_norm(X_tr, y_tr_n, X_te_n := X_te, y_te_n, deg)
        tmse_n.append(t_); emse_n.append(e_)
    dd_by_noise[sigma] = (tmse_n, emse_n)
    peak_idx = np.argmax(emse_n[:n_train + 5])
    print(f"  σ = {sigma:.1f}:  peak test MSE = {emse_n[peak_idx]:.4f} at degree {peak_idx+1},  "
          f"over-param (d=70) test MSE = {emse_n[69]:.4f}")

print()
print("  - Higher noise → larger spike at interpolation threshold.")
print("  - In noiseless case (σ→0), interpolation is perfect → no spike.")
print("  - Double descent is a consequence of overfitting NOISE at threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Regularised vs min-norm at interpolation
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  REGULARISATION REMOVES THE DOUBLE DESCENT PEAK")
print("=" * 65)
print()

# At degree = n_train (interpolation), compare min-norm vs Ridge (regularised)
deg_interp = n_train
print(f"  Degree = {deg_interp} (interpolation threshold):  n_params = n_train = {n_train}")
print()

for alpha in [1e-8, 0.001, 0.01, 0.1, 1.0, 10.0]:
    pipe_r = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg_interp)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=alpha)),
    ])
    pipe_r.fit(X_tr.reshape(-1, 1), y_tr)
    test_r = np.mean((pipe_r.predict(X_te.reshape(-1, 1)) - y_te) ** 2)
    print(f"    Ridge alpha={alpha:>8.5f}:  Test MSE = {test_r:.4f}")

print()
print("  - Min-norm (alpha→0) peaks; Ridge (larger alpha) smoothly avoids peak.")
print("  - Regularisation = implicit model selection; avoids interpolation instability.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Double Descent: Beyond the Classical Bias-Variance Tradeoff",
             fontsize=13, fontweight="bold")

n_params_dd = [d + 1 for d in degrees_dd]

# Panel 1: Double descent curve
ax = axes[0, 0]
ax.semilogy(n_params_dd, train_mse_dd, "b-", lw=2, label="Training MSE")
ax.semilogy(n_params_dd, test_mse_dd,  "r-", lw=2, label="Test MSE")
ax.axvline(n_train, color="gray", lw=2, ls="--", label=f"Interpolation threshold (n={n_train})")
ax.fill_betweenx([1e-5, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 100],
                  0, n_train, alpha=0.05, color="steelblue", label="Classical regime")
ax.fill_betweenx([1e-5, 100], n_train, max(n_params_dd), alpha=0.05, color="tomato",
                  label="Over-param regime")
ax.set_xlabel("Number of parameters (d+1)")
ax.set_ylabel("MSE (log scale)")
ax.set_title("Double Descent Curve\\n(min-norm interpolation, polynomial)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# Panel 2: Effect of noise level
ax = axes[0, 1]
noise_colours = ["steelblue", "seagreen", "tomato", "purple"]
for (sigma, (_, emse_n)), col in zip(dd_by_noise.items(), noise_colours):
    ax.semilogy(n_params_dd, emse_n, lw=2, color=col, label=f"σ = {sigma}")
ax.axvline(n_train, color="gray", lw=2, ls="--", alpha=0.7)
ax.set_xlabel("Number of parameters")
ax.set_ylabel("Test MSE (log scale)")
ax.set_title("Double Descent: Effect of Noise Level\\n(higher noise → larger peak at threshold)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel 3: Fitted curves: classical vs interpolation vs over-param
ax = axes[1, 0]
x_plot = np.linspace(0, 2, 500)
ax.scatter(X_tr, y_tr, c="steelblue", s=35, zorder=6, label="Train data", alpha=0.8)
ax.plot(x_plot, true_fn(x_plot), "k-", lw=2.5, label="True function", zorder=7)

show_degrees = [(3, "seagreen", f"deg=3 (classical opt, test={test_mse_dd[2]:.2f})"),
                (n_train, "tomato", f"deg={n_train} (interpolation, test={test_mse_dd[n_train-1]:.2f})"),
                (60, "purple", f"deg=60 (over-param, test={test_mse_dd[59]:.2f})")]

for deg_s, col, lbl in show_degrees:
    _, _, w_s = fit_poly_min_norm(X_tr, y_tr, x_plot, y_te[:500], deg_s)
    scaler_s = StandardScaler()
    phi_plot = PolynomialFeatures(degree=deg_s, include_bias=True).fit_transform(
                   x_plot.reshape(-1, 1))
    phi_tr_s = PolynomialFeatures(degree=deg_s, include_bias=True).fit_transform(
                   X_tr.reshape(-1, 1))
    phi_tr_s = scaler_s.fit_transform(phi_tr_s)
    phi_plot  = scaler_s.transform(phi_plot)
    n_p, p_s = phi_tr_s.shape
    if p_s <= n_p:
        A = phi_tr_s.T @ phi_tr_s + 1e-8 * np.eye(p_s)
        w_s = np.linalg.solve(A, phi_tr_s.T @ y_tr)
    else:
        A = phi_tr_s @ phi_tr_s.T + 1e-8 * np.eye(n_p)
        w_s = phi_tr_s.T @ np.linalg.solve(A, y_tr)
    y_plot_s = phi_plot @ w_s
    ax.plot(x_plot, np.clip(y_plot_s, -3, 3), color=col, lw=1.8, label=lbl)

ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Fitted Functions at Three Regimes\\n(over-param finds smoother fit than threshold!)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel 4: Classical U-shape (with regularisation) vs double descent (no reg)
ax = axes[1, 1]
alphas_compare = [1e-8, 0.01, 0.1, 1.0]
colours_reg    = ["tomato", "seagreen", "steelblue", "purple"]

for alpha_r, col_r in zip(alphas_compare, colours_reg):
    test_by_deg = []
    for deg in degrees_dd:
        n, p = n_train, deg + 1
        phi_tr = PolynomialFeatures(degree=deg).fit_transform(X_tr.reshape(-1, 1))
        phi_te = PolynomialFeatures(degree=deg).fit_transform(X_te.reshape(-1, 1))
        sc = StandardScaler(); phi_tr = sc.fit_transform(phi_tr); phi_te = sc.transform(phi_te)
        if p <= n:
            A = phi_tr.T @ phi_tr + alpha_r * np.eye(p)
            w = np.linalg.solve(A, phi_tr.T @ y_tr)
        else:
            A = phi_tr @ phi_tr.T + alpha_r * np.eye(n)
            w = phi_tr.T @ np.linalg.solve(A, y_tr)
        test_by_deg.append(np.mean((phi_te @ w - y_te) ** 2))
    ax.semilogy(n_params_dd, test_by_deg, lw=2, color=col_r,
                label=f"Ridge α={alpha_r}")

ax.axvline(n_train, color="gray", lw=1.5, ls="--", alpha=0.6)
ax.set_xlabel("Number of parameters")
ax.set_ylabel("Test MSE (log scale)")
ax.set_title("Regularisation Tames Double Descent\\n(larger α → classical U-shape returns)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("double_descent.png", dpi=110)
print()
print("  Plot saved → double_descent.png")
print()
print("  KEY TAKEAWAYS — DOUBLE DESCENT:")
print("  1. Classical regime (d << n): U-shaped test error as d increases.")
print("  2. Interpolation threshold (d ≈ n): test error spikes — WORST POINT.")
print("  3. Over-parameterised regime (d >> n): test error falls again,")
print("     often below classical optimum (min-norm implicit regularisation).")
print("  4. Noise amplifies the spike; more noise → more severe peak.")
print("  5. Ridge regularisation (λ > 0) smooths away the spike at all d.")
print("  6. Modern neural nets live in regime 3 — classical theory fails here.")
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