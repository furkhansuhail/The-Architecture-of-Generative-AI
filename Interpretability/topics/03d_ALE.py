"""
ALE — Accumulated Local Effects
================================

ALE plots (Apley & Zhu, 2020) are the statistically correct way to
visualise the marginal effect of a feature on a black-box model's output
when features are correlated. They answer the question "how does the model's
prediction change as feature Xⱼ varies?" without the extrapolation bias
that plagues Partial Dependence Plots.

This module derives ALE from first principles: the mathematical problem with
PDP, the M-plot as a failed intermediate attempt, and why ALE's local
differencing approach is the correct solution.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "ALE — Accumulated Local Effects"
DISPLAY_NAME = "03d · ALE"
ICON = "📈"
SUBTITLE = "The unbiased marginal effect plot for correlated features"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Starting Point — The Question All Marginal Effect Plots Try to Answer

Suppose you have a trained black-box model f(x₁, x₂, ..., xₚ) and you want
to understand how feature x₁ influences the prediction. The model sees all
features simultaneously, so you cannot simply "vary x₁ and watch the output
change" — changing x₁ changes the whole input vector, and the other features
do not move in isolation.

The fundamental question is:

    What is the PURE marginal effect of x₁ on f, stripped of any
    confounding from the other features?

Three methods have been proposed. Two of them fail, one does not. Understanding
WHY the first two fail is the most important part of understanding ALE.

    METHOD A: Partial Dependence Plot  →  fails under feature correlation
    METHOD B: Marginal Plot (M-plot)   →  fails under feature association
    METHOD C: Accumulated Local Effects →  correct

##### PART I: THE PDP — AND WHY IT FAILS

### The PDP Definition

The Partial Dependence Plot for feature j evaluates the model at a grid of
values for xⱼ, marginalising over the empirical distribution of all other
features x_\j = (x₁, ..., xⱼ₋₁, xⱼ₊₁, ..., xₚ):

    PDP_j(v) = (1/n) Σᵢ₌₁ⁿ f(v, Xᵢ_\j)

The notation Xᵢ_\j means "all features of training instance i, except feature
j which is replaced by the grid value v."

This looks reasonable. For each candidate value v of xⱼ, you fix xⱼ = v
and average the model's output over all real instances' remaining features.

### The Extrapolation Problem

The problem is subtle but severe. Consider the simplest case: two features
x₁ and x₂ that are strongly positively correlated. High x₁ values co-occur
with high x₂ values. The joint distribution P(x₁, x₂) is concentrated along
a diagonal ridge.

    # ============================================================== #
    **Diagram — Joint Distribution and PDP's Extrapolation:**

    x₂
    ↑
    3 │                              ●●
      │                         ●●●●●
    2 │                    ●●●●●●●
      │               ●●●●●●
    1 │          ●●●●●●               ← real data lies on diagonal
      │     ●●●●●
    0 │ ●●●●
      └─────────────────────────────→ x₁
        0    1    2    3    4    5

    When PDP evaluates f at (x₁=5, x₂):
      It averages over ALL x₂ values in the dataset: x₂ ∈ {0,1,2,3,...}
      This includes (x₁=5, x₂=0) — a combination that DOES NOT EXIST.
      The model was never trained on such pairs.
      The model's output in that region is undefined / unreliable.

    PDP(x₁=5) = (1/n) Σᵢ f(5, X₂ᵢ)
              = average over a vertical slice that slashes through
                regions of near-zero training data density.
    # ============================================================== #

This is the extrapolation problem. PDP averages over the MARGINAL
distribution of x_\j — as if x_\j were independent of xⱼ. When they are
not independent, PDP creates unrealistic (xⱼ, x_\j) combinations and asks
the model to evaluate them. The model's behaviour in those sparse regions was
never constrained by training data; the output is arbitrary.

### The Bias of PDP Under Correlation — A Formal Statement

Let x = (x₁, x₂) with x₁ correlated with x₂. Let the true model be linear:

    f(x₁, x₂) = β₁x₁ + β₂x₂

The PDP for x₁ is:

    PDP₁(v) = E_X₂[f(v, X₂)]
             = E_X₂[β₁v + β₂X₂]
             = β₁v + β₂ E[X₂]

This looks correct — PDP recovers the slope β₁ on v. But only because the
linear model's marginal expectation separates cleanly. Now suppose x₂ and x₁
are correlated with slope γ (i.e., E[X₂ | X₁=v] = γv). The PDP uses the
UNCONDITIONAL expectation E[X₂], not the conditional E[X₂ | X₁=v].

For non-linear models, or when the feature effect is amplified by correlation,
this distinction matters critically. The general result (Apley & Zhu 2020):

    PDP₁(v) ≈ MAIN EFFECT of x₁ + CONFOUNDING from correlated features

The confounding term is zero ONLY if x₁ is independent of all x_\j. In
virtually every real-world dataset, some correlation exists.

    # ============================================================== #
    **Concrete Confounding Example:**

    True model: f(x₁, x₂) = x₁²  (only x₁ matters)
    Data:       x₂ = x₁ + ε  (x₂ is a near-perfect copy of x₁)

    PDP₁(v) = (1/n) Σᵢ f(v, x₂ᵢ)
             = (1/n) Σᵢ v²         (since f does not use x₂ at all)
             = v²    ← correct!

    PDP₂(v) = (1/n) Σᵢ f(x₁ᵢ, v)
             = (1/n) Σᵢ x₁ᵢ²      (substituting v for x₂ doesn't matter)
             ≈ E[x₁²]              ← constant! PDP₂ is flat.

    Paradox: PDP says x₂ has ZERO effect (flat line), even though x₂
    is a perfect proxy for x₁ and SHAP would correctly attribute half
    the total effect to x₂ via the interaction sharing mechanism.

    This is a different failure mode: PDP can show NO EFFECT for a feature
    that is genuinely (indirectly) important via correlation.
    # ============================================================== #


##### PART II: THE M-PLOT — AN INSTRUCTIVE FAILURE

### What Is the M-Plot?

The M-plot (Marginal plot) is a naive fix for PDP's extrapolation problem.
Instead of averaging over the MARGINAL distribution of x_\j (all instances),
it averages only over instances where xⱼ ≈ v — the CONDITIONAL distribution:

    M_j(v) = E_X_\j[f(v, X_\j) | Xⱼ = v]
            ≈ (1/|Nⱼ(v)|) Σ_{i∈Nⱼ(v)} f(v, Xᵢ_\j)

where Nⱼ(v) = {i : Xᵢⱼ ≈ v} is the set of instances with xⱼ near v.

This avoids extrapolation: we only evaluate the model on feature combinations
that actually co-occur in the data. If x₁ and x₂ are correlated, when we fix
x₁ = 5 we only use instances with x₂ ≈ 5 — which is exactly what the data
shows co-occurring with x₁ = 5.

### Why the M-Plot Still Fails — The Mixing of Effects

The M-plot avoids extrapolation but introduces a different error: it MIXES
the effects of xⱼ and the effects of correlated features x_\j.

    # ============================================================== #
    **The M-Plot Mixing Problem:**

    True model: f(x₁, x₂) = x₂²   (only x₂ matters, x₁ is irrelevant)
    Data:       x₂ = x₁ + ε  (strong correlation)

    M-plot for x₁:
      M₁(v) = E[f(v, X₂) | X₁ = v]
             = E[X₂² | X₁ = v]
             ≈ E[X₁² | X₁ = v]    (since x₂ ≈ x₁)
             ≈ v²

    The M-plot says x₁ has effect v² — a STRONG, INCREASING effect!
    But x₁ has ZERO TRUE EFFECT. The model uses only x₂.

    The M-plot is picking up x₂'s effect and attributing it to x₁,
    simply because x₂ moves with x₁. This is pure confounding.
    # ============================================================== #

The M-plot computes the effect of changing v IN THE REAL DATA — where changing
xⱼ also changes the correlated features, because they move together. This is
not the marginal effect of xⱼ ALONE. It is the total effect of everything
that moves when xⱼ changes.

    Summary of failures:
    ┌──────────────────────────────────────────────────────────────┐
    │  Method      Problem              Technical term             │
    │  ─────────────────────────────────────────────────────────   │
    │  PDP         Averages over        Extrapolation /            │
    │              unrealistic inputs   Out-of-distribution        │
    │                                   evaluation                 │
    │                                                              │
    │  M-plot      Picks up correlated  Omitted variable bias /    │
    │              features' effects    Effect mixing              │
    │                                                              │
    │  ALE         Neither              The correct answer         │
    └──────────────────────────────────────────────────────────────┘



##### PART III: ALE — THE CORRECT SOLUTION

### The Key Insight — Differences, Not Averages

ALE's core idea is deceptively simple:

    Instead of asking:
      "What is the model output when x₁ = v?"
      (PDP and M-plot both ask this)

    Ask:
      "What is the CHANGE in model output when x₁ moves from
       a slightly smaller value to a slightly larger value?"
      AND: "Evaluate this change only for instances that actually
            have x₁ near v — so the other features are realistic."

Taking a derivative (or discrete difference) eliminates the baseline
level of x₂'s effect, leaving only the PURE LOCAL EFFECT of x₁.

    # ============================================================== #
    **Why Differencing Removes the Confounding:**

    True model: f(x₁, x₂) = x₂²  (only x₂ matters)
    When x₁ ≈ v and x₂ ≈ v (correlated), consider:

    For a small step δ (x₁ goes from v-δ/2 to v+δ/2):
      f(v+δ/2, x₂) - f(v-δ/2, x₂)
      = x₂² - x₂²         (model does not use x₁)
      = 0

    The local difference is ZERO because x₂ is held fixed.
    We are not changing x₂ — we are moving x₁ in isolation.
    x₂ stays at its real observed value for each instance.

    Compare this to the M-plot, which computes f(v, x₂ᵢ) where x₂ᵢ
    VARIES across instances because we select instances with x₁ ≈ v,
    and those instances have x₂ ≈ v — so the average involves x₂'s
    effect. The difference cancels this out.
    # ============================================================== #


### The ALE Definition — Formal

For a single feature x₁ with j = 1, let the value range be partitioned into
K intervals [z₀, z₁], [z₁, z₂], ..., [z_{K-1}, zK] (the grid). The ALE
function is defined as:

    ALE₁(x) = ∫_{z₀}^{x}  E_{X₂|X₁=v} [ ∂f/∂x₁(v, X₂) ] dv  − constant

The integrand is the conditional expectation of the PARTIAL DERIVATIVE of f
with respect to x₁, evaluated at (v, X₂) where X₂ is drawn from its
CONDITIONAL distribution given X₁ = v.

Breaking this apart:
  • ∂f/∂x₁(v, X₂): the sensitivity of f to x₁ at the point (v, X₂)
  • E_{X₂|X₁=v}[...]: averaged over realistic x₂ values (those co-occurring
    with x₁ = v) — NOT over all x₂ values as PDP does
  • ∫ ... dv: accumulated from the left boundary z₀ to the query point x

The constant is chosen to centre the ALE function (mean ALE = 0 over the data).

In the continuous limit, this is an integral of conditional partial
derivatives. In practice, since the model may not be differentiable, we
approximate the derivative with a finite difference.


### The Discrete ALE Algorithm

Given a dataset with n instances, feature j's values sorted and divided
into K intervals:

    STEP 1 — PARTITION:
    ────────────────────
    Sort the observed values of feature j: x₁(₁) ≤ x₁(₂) ≤ ... ≤ x₁(ₙ)
    Define K intervals using quantiles: each interval contains ≈ n/K instances.
    Interval endpoints: z₀ = x₁(₁), z₁ = x₁(⌊n/K⌋), ..., zK = x₁(ₙ)

    Using quantile-based intervals is important: it ensures each interval
    has sufficient data density. Equal-width intervals (like PDP's grid) can
    produce some intervals with very few instances — making estimates noisy.

    # ================================================================= #
    **Diagram — Quantile Intervals vs Equal-Width Intervals:**

    Feature distribution: right-skewed (many small values, few large ones)
    Values: 1 1 1 2 2 2 3 3 4 5 5 8 10 15 20

    Equal-width intervals (width=4):
      [1,5):   10 instances  ← crowded
      [5,9):    3 instances
      [9,13):   1 instance   ← nearly empty, estimate unreliable
      [13,17):  1 instance   ← nearly empty
      [17,21):  1 instance   ← nearly empty

    Quantile intervals (3 instances each):
      [1,2):   3 instances   ← equal density throughout
      [2,3):   3 instances
      [3,5):   3 instances
      [5,15):  3 instances
      [15,20): 3 instances   ← each interval has the same coverage
    # ================================================================= #

    STEP 2 — LOCAL DIFFERENCES:
    ────────────────────────────
    For each interval k = 1, ..., K with endpoints [z_{k-1}, zₖ]:

      Let Nₖ = {i : z_{k-1} ≤ X₁ᵢ < zₖ} — the set of instances in interval k

      For each instance i ∈ Nₖ, compute the difference:
        δᵢ = f(zₖ, Xᵢ_\j) − f(z_{k-1}, Xᵢ_\j)

      This changes feature j from the left to the right boundary of the
      interval, while ALL OTHER FEATURES stay at their real observed values.

      The interval effect is the average of these differences:
        Δₖ = (1/|Nₖ|) Σᵢ∈Nₖ δᵢ

    STEP 3 — ACCUMULATE:
    ─────────────────────
    The ALE at point x (within interval k) is the sum of all previous
    interval effects plus a linear interpolation within interval k:

        ALE_raw(x) = Σ_{j=1}^{k-1} Δⱼ + interpolate(x in interval k)

    In the simplified discrete version evaluated at grid points:
        ALE_raw(zₖ) = Σⱼ₌₁ᵏ Δⱼ

    STEP 4 — CENTRE:
    ─────────────────
    Subtract the training-set-weighted mean so the ALE is centred at zero:

        ALE_centred(x) = ALE_raw(x) − (1/n) Σᵢ ALE_raw(X₁ᵢ)

    Centring has no effect on the SHAPE of the curve (slopes are unchanged).
    It anchors the interpretation: ALE = 0 means "prediction equal to the
    dataset average prediction." Positive ALE means this feature value pushes
    the prediction above average.


### Why Accumulation is the Right Operation

The key question: why accumulate (sum) the local differences rather than
just plotting the differences Δₖ directly?

The answer: each Δₖ is the EFFECT IN THAT INTERVAL ONLY. We want the
TOTAL EFFECT of being at xⱼ = v relative to the baseline xⱼ = z₀. That
total effect is the sum of all the local steps it took to GET from z₀ to v.

    # ================================================================= #
    **The Accumulation Intuition:**

    Think of altitude. You are hiking from sea level (z₀) to a mountain
    peak (v). The "effect" of being at v is your total altitude gain.
    That total is the SUM of all the small uphill and downhill steps you
    took along the path — not any single step's value.

    Δ₁ = +50m  (first mile, uphill)
    Δ₂ = -10m  (second mile, slight downhill)
    Δ₃ = +80m  (third mile, steep climb)
    Δ₄ = +20m  (fourth mile, gentle rise)

    ALE(peak) = 50 + (-10) + 80 + 20 = +140m total elevation.

    Each Δₖ is a LOCAL measurement (altitude change in one mile).
    ALE is the GLOBAL cumulative effect of being at the peak.

    If Δₖ > 0 in an interval: the model prediction INCREASES as
    feature j increases through that interval.
    If Δₖ < 0: the prediction DECREASES.
    If Δₖ ≈ 0: the feature has no effect in that region.
    # ================================================================= #


### The Mathematical Justification — Why ALE is Unbiased

Apley & Zhu (2020) prove the following result. Define the "main effect"
of xⱼ as the function mⱼ(xⱼ) that satisfies:

    f(x) ≈ f₀ + mⱼ(xⱼ) + Σ_{k≠j} mₖ(xₖ) + interaction terms

where the decomposition is the functional ANOVA (Hooker, 2004):
each mₖ integrates to zero and captures only the individual feature's
contribution with all interactions projected out.

Theorem (Apley & Zhu 2020): ALE converges in probability to the main
effect mⱼ(xⱼ) as n → ∞ (under regularity conditions). Moreover, it
does so WITHOUT extrapolating outside the joint support of P(X).

PDP converges to:
    PDP_j(v) → mⱼ(v) + Σ_{k≠j} E_X[mⱼₖ(v, Xₖ)]  +  bias

The bias term is zero ONLY when all features are independent. Under
any correlation structure, PDP is a biased estimator of mⱼ.

    # ================================================================= #
    **The Functional ANOVA Decomposition (brief):**

    Any function f(x₁, x₂, ..., xₚ) can be written as:
      f(x) = f₀
           + Σⱼ mⱼ(xⱼ)              ← individual main effects
           + Σⱼ<ₖ mⱼₖ(xⱼ, xₖ)       ← two-way interaction effects
           + higher-order interactions

    Each term is defined to integrate (or sum) to zero along each
    of its own arguments, making them "pure" effects.

    ALE estimates mⱼ(xⱼ) — the PURE individual main effect of xⱼ.
    This is the effect of xⱼ with all interactions averaged out.

    PDP estimates mⱼ(xⱼ) + (confounding from correlated interactions).
    M-plot estimates mⱼ(xⱼ) + mⱼₖ(xⱼ, xₖ) + ... (includes interactions).
    # ================================================================= #


##### PART IV: ALE IN HIGHER DIMENSIONS

### Second-Order ALE — Interaction Effects

ALE naturally extends to pairs of features, giving a visualisation of
interaction effects that is also unbiased under correlation.

For two features (x₁, x₂), the second-order ALE measures the PURE
INTERACTION between them — the part of f(x₁, x₂) that cannot be
explained by x₁ and x₂ individually.

    Second-order ALE algorithm:
    ────────────────────────────
    Partition both feature j₁ and feature j₂ into intervals.
    For each cell (rectangle) in the 2D grid:
      Compute the four-corner difference:
        Δ_{k₁,k₂} = f(z_{k₁}, z_{k₂}) − f(z_{k₁-1}, z_{k₂})
                   − f(z_{k₁}, z_{k₂-1}) + f(z_{k₁-1}, z_{k₂-1})

    This is the DISCRETE SECOND DERIVATIVE of f — change in the
    effect of x₁ as x₂ varies. If this is non-zero, there is a
    genuine interaction.

    # ================================================================= #
    **The Four-Corner Difference Explained:**

    x₂  z_{k₂}  ●─────────────────●
                 │                 │
                 │ cell (k₁, k₂)   │
                 │                 │
    z_{k₂-1}    ●─────────────────●
               z_{k₁-1}         z_{k₁}   → x₁

    Four evaluations:
      f(z_{k₁-1}, z_{k₂-1})  ← bottom-left
      f(z_{k₁},   z_{k₂-1})  ← bottom-right
      f(z_{k₁-1}, z_{k₂}  )  ← top-left
      f(z_{k₁},   z_{k₂}  )  ← top-right

    Second difference:
    [f(BR) − f(BL)] − [f(TR) − f(TL)]

    = [effect of x₁ at low x₂] − [effect of x₁ at high x₂]

    If this is zero: x₁'s effect is the SAME at all x₂ values.
    No interaction.
    If this is non-zero: x₁'s effect DEPENDS on x₂. Interaction.
    # ================================================================= #

The second-order ALE function is accumulated (summed) across both
dimensions and centred so that it integrates to zero along both axes.
The result is a 2D surface — positive cells mean the two features
cooperate (super-additive), negative cells mean they antagonise
(sub-additive).

Unlike SHAP interaction values (which share interaction credit equally
between feature pairs via the Shapley formula), second-order ALE assigns
the interaction to the joint (x₁, x₂) space directly.


### ALE for Categorical Features

When feature xⱼ is categorical (nominal or ordinal), intervals do not
exist. ALE adapts:

    For ORDINAL categories (e.g., low < medium < high):
    ─────────────────────────────────────────────────────
    Each ordered category value is treated as an interval.
    The local difference for category level k is:
      Δₖ = f(k, X_\j) − f(k-1, X_\j)  averaged over instances with xⱼ = k

    "Moving from category k-1 to k" is still well-defined for ordered
    categories. Accumulation proceeds as before.

    For NOMINAL categories (no natural order, e.g., country, colour):
    ──────────────────────────────────────────────────────────────────
    There is no inherent ordering to define intervals. ALE uses a
    DISTANCE-BASED ordering approach:

      1. Compute the distance between each pair of categories based on
         similarity in the TARGET (or in correlated features' distributions
         within each category).
      2. Order categories by similarity to create a pseudo-ordering.
      3. Apply the ordinal ALE procedure on this ordering.

    The resulting ALE values measure: "how different is the model's
    prediction for category c, compared to similar categories?" The shape
    of the curve depends on the ordering, but the RELATIVE heights between
    similar categories are meaningful.


##### PART V: ALE PROPERTIES, LIMITATIONS, AND PRACTICAL GUIDANCE

### Statistical Properties

    UNBIASEDNESS:
    ─────────────
    ALE is an unbiased estimator of the main effect mⱼ(xⱼ) under the
    functional ANOVA decomposition, regardless of the correlation structure
    between features. This is the central theorem of Apley & Zhu (2020).

    In contrast, PDP is BIASED whenever features are correlated. The bias
    of PDP is proportional to the strength of the correlation AND the
    magnitude of the interaction effect. For strongly correlated, strongly
    interacting features, PDP can be severely misleading.

    CONSISTENCY:
    ─────────────
    As n → ∞ and the interval width h → 0 (with n × h → ∞ to ensure
    sufficient instances per interval), ALE converges in probability to
    the true continuous derivative integral. This is analogous to how
    a histogram converges to a PDF.

    VARIANCE:
    ─────────
    ALE's variance comes from two sources:
      1. Finite-sample variance of the local differences Δₖ
         (too few instances in some intervals → noisy estimates)
      2. Accumulation error: errors in early intervals propagate
         to all later ALE values (because we sum left-to-right)

    The accumulation means that variance GROWS as you move toward the
    tails of the feature distribution — where fewer instances are available.
    ALE is most reliable in the CENTRE of the distribution.

    INTERVAL COUNT SELECTION:
    ──────────────────────────
    The number of intervals K controls the bias-variance trade-off:

      Small K (few wide intervals):
        Each interval has many instances → low variance.
        But wide intervals aggregate over large changes in xⱼ.
        The local difference approximation is coarser.
        → Smooth ALE curve, may miss sharp local effects.

      Large K (many narrow intervals):
        Each interval has few instances → high variance, noisy.
        Local differences are computed at finer resolution.
        → Wiggly ALE curve, can overfit to local noise.

    Typical guidance: K = 40–100 intervals for datasets of 1000+ instances.
    For very skewed distributions, use quantile-based intervals (equal count
    per interval) rather than equal-width intervals.

    # ================================================================= #
    **Bias-Variance Trade-off for K:**

    K too small (e.g., K=3):             K too large (e.g., K=200):
    ALE ↑                                ALE ↑
        │                                    │     . .
        │    ___                             │   .     . .
        │___/   \___                         │ ..         . .
        │                                    │.                .
        └─────────────────→ xⱼ                └─────────────────→ xⱼ
        Missing the real shape.              Overfitting to noise.

    K=50 (typically good):
    ALE ↑
        │      .─.
        │    ./   \.
        │___/      \___
        └─────────────────→ xⱼ
        Captures real shape without overfitting.
    # ================================================================= #


### What ALE Measures — and What It Does Not

ALE measures the MAIN EFFECT of each feature: how the model's prediction
changes as that feature varies, with all interaction effects projected out.

This is precisely what you want for understanding "is this feature used by
the model, and how?" But it means ALE does NOT directly answer some questions:

    ALE TELLS YOU:
    ──────────────
    • Does the model use feature j? (Is ALE flat or does it vary?)
    • Is the model's use of feature j monotone, non-monotone, or threshold-based?
    • In what value range does feature j most influence predictions?
    • Is the effect linear (ALE is a straight line) or non-linear (ALE curves)?
    • How does the pure effect of j compare to the pure effect of k?

    ALE DOES NOT TELL YOU:
    ──────────────────────
    • Why did the model make THIS SPECIFIC prediction? (Use SHAP for local)
    • Does feature j interact with feature k? (Use 2nd-order ALE)
    • What would happen if I intervened to change xⱼ for ONE instance?
      (ALE averages over an interval — it is a population-level measure)
    • Whether the model's use of j is correct / desirable
      (ALE describes WHAT the model does, not WHETHER it should do it)
    • The joint effect of j and k simultaneously (use 2nd-order ALE)


### ALE vs PDP vs SHAP Dependence — Side by Side

All three can answer "how does feature j affect the model's prediction?"
They give different answers because they measure different things.

    # ================================================================= #
    **Diagram — What Each Method Visualises:**

    FEATURE x₁: moderately correlated with x₂ (ρ ≈ 0.6)
    True model: f = 0.5·x₁ + 0.4·x₂ + 0.1·x₁·x₂

    PDP₁(v):       E_{X₂}[f(v, X₂)]
                   = 0.5v + 0.4·E[X₂] + 0.1v·E[X₂]
                   (averages over ALL x₂ values, including unrealistic pairs)
                   Slope ≈ 0.5 + 0.1·E[X₂]   ← INFLATED by correlation
                                                 (x₁ gets credit for x₂'s effect)

    M-plot₁(v):    E_{X₂|X₁=v}[f(v, X₂)]
                   = 0.5v + 0.4·E[X₂|X₁=v] + 0.1v·E[X₂|X₁=v]
                   Since E[X₂|X₁=v] ≈ 0.6v (correlation γ = 0.6):
                   ≈ 0.5v + 0.24v + 0.06v² = 0.74v + 0.06v²
                   ← MIXES x₂'s effect and interaction into x₁'s curve

    ALE₁(v):       ∫ E_{X₂|X₁=z}[∂f/∂x₁(z, X₂)] dz
                   ∂f/∂x₁ = 0.5 + 0.1·X₂
                   E[∂f/∂x₁ | X₁=z] = 0.5 + 0.1·E[X₂|X₁=z] = 0.5 + 0.1·0.6z
                   Integrate: 0.5z + 0.03z²  (after centring)
                   ← CAPTURES only the true local slope of x₁
                   ← Interaction (0.1·x₁·x₂) contributes only through the
                      local conditional derivative — not through x₂'s main effect

    SHAP dep₁(v):  φ₁(xᵢ) for instances where X₁ᵢ ≈ v (scatter plot)
                   = portion of f(xᵢ)-baseline attributable to x₁
                   = INCLUDES half the interaction x₁·x₂ via Shapley sharing
                   ← Not a main-effect estimate; combines main effect + shared
                      interactions. The scatter shows INSTANCE-LEVEL attribution.
    # ================================================================= #

The choice between them comes down to what question you are answering:
  • ALE: "What is the model's main effect of x₁?" (cleanest global view)
  • PDP: Only valid when features are approximately independent
  • SHAP dependence: "How much credit does x₁ receive, including interactions?"


### Practical Failure Modes and How to Detect Them

    FAILURE MODE 1 — EMPTY INTERVALS:
    ────────────────────────────────────
    If a feature has a bimodal distribution (e.g., most values are near 0
    or near 10, very few near 5), then intervals near 5 will have very few
    instances. The local difference Δₖ in those intervals is based on
    near-zero samples — high variance, potentially misleading.

    Detection: plot the interval count histogram alongside the ALE curve.
    Intervals with n < 10 instances should be treated with extreme caution.

    Fix: use wider intervals in sparse regions (adaptive binning), or
    note in the plot that the ALE is unreliable in those ranges.

    # ================================================================= #
    Visualising interval reliability:

    ALE
    ↑          ___
    │         /   \___
    │    ___/          ← suspicious spike in sparse region
    │___/
    └────────────────────→ xⱼ

    Instance count:
    ████████████░░░░░░████████  ← few instances in middle
    High  High  Low   High

    The spike in the ALE where instance count is low is likely noise.
    # ================================================================= #

    FAILURE MODE 2 — STRONGLY CORRELATED FEATURE PAIRS (ρ > 0.95):
    ────────────────────────────────────────────────────────────────
    When two features are near-perfectly correlated, the conditional
    distribution P(X₂ | X₁ = v) is very concentrated — essentially a
    single value x₂ ≈ v. This means:
      • ALE for x₁ is essentially the derivative at (v, v)
      • ALE for x₂ is essentially the same derivative at (v, v)
    Both ALEs tell the same story, because the features are near-identical.

    In the extreme (x₂ = x₁ exactly), ALE for x₁ is the true derivative
    ∂f/∂x₁ — but so is ALE for x₂ ≈ ∂f/∂x₂ evaluated at x₁ = x₂. Both
    capture the SAME total effect ∂f/∂x₁ + ∂f/∂x₂ split by the chain rule.

    Detection: compute the pairwise feature correlation matrix. When ρ > 0.9,
    treat ALE values for those features as measuring a shared combined effect.
    Consider removing one of the near-duplicate features before modelling.

    FAILURE MODE 3 — NON-SMOOTH MODEL IN INTERVALS:
    ─────────────────────────────────────────────────
    ALE computes f(z_{k}, X_\j) − f(z_{k-1}, X_\j) and averages over
    instances in the interval. If the model f is discontinuous within an
    interval (e.g., a very deep decision tree with many splits), the
    averaged difference can be unstable.

    This is rarely a problem in practice (models are usually smooth enough
    at the scale of one interval), but it is worth knowing for ensemble
    models with very irregular boundaries.


##### PART VI: ALE AND CAUSAL REASONING — WHAT ALE IS NOT

### ALE is a Description of the Model, Not the World

A common mistake is to interpret ALE as a causal effect:
"The ALE shows that increasing income by $10k increases the predicted
approval probability by 8%, therefore I should give applicants $10k
to improve their approval odds."

This is wrong for two reasons.

    REASON 1 — ALE describes the MODEL'S behaviour, not reality's:
    ───────────────────────────────────────────────────────────────
    ALE tells you how the model f responds to changes in features.
    If the model has learned a spurious correlation (e.g., high income
    in this dataset happens to correlate with a good loan outcome because
    of the specific historical time period sampled), the ALE will show
    a positive income effect — but that effect may not replicate causally.

    REASON 2 — ALE is NOT an interventional quantity:
    ──────────────────────────────────────────────────
    ALE computes the local conditional derivative E[∂f/∂xⱼ | Xⱼ = v].
    This is the OBSERVATIONAL effect: how does f change when xⱼ HAPPENS
    to vary, given that other features are at their natural conditional
    values?

    The causal interventional quantity (Pearl's do-calculus) would be:
    E[f | do(Xⱼ = v)] — the expected outcome if we FORCE xⱼ to take
    value v for a randomly chosen individual, regardless of their other
    features.

    These are equal ONLY in the absence of confounding — i.e., when xⱼ
    is independent of all other features (no hidden common causes).
    In observational data, these differ whenever there is a common cause
    of xⱼ and the outcome that is not included in the feature set.

    # ================================================================= #
    **Observational vs Interventional — A Simple Example:**

    Setting: loan approval model. Features: income, credit_score.
    Hidden common cause: "financial responsibility" (not measured).

    ALE for income:
      When income = $80k, the population also tends to have credit_score = 720
      (high) because both are caused by financial responsibility.
      ALE sees the model predicting high approval for (income=80k, credit=720).
      ALE says: "high income → high approval probability."

    Causal interventional effect of income:
      If we GIVE $80k to someone who was earning $20k (a cash transfer),
      their credit_score remains at 580 (financial responsibility unchanged).
      The model would predict lower approval than ALE suggested.
      The causal effect of income is SMALLER than ALE implies.

    ALE is NOT wrong. It correctly describes what the MODEL sees.
    But the model's pattern is confounded by the unmeasured common cause.
    ALE faithfully relays the confounding. The mistake is in the
    CAUSAL INTERPRETATION of the ALE output, not in ALE itself.
    # ================================================================= #

### When is ALE a Good Approximation to a Causal Effect?

ALE is a better approximation to causal effects when:

    1. The model has access to all relevant confounders as features.
       (No unmeasured common causes between xⱼ and the outcome.)

    2. The feature xⱼ is NOT caused by the outcome (no reverse causation).

    3. The model is well-calibrated: its predictions reflect the true
       conditional probabilities, not just relative rankings.

    4. The task is PREDICTION not INTERVENTION:
       "What do people with income = $80k typically get?" (ALE is fine)
       vs "What happens if I change income to $80k?" (need causal model)

For pure predictive modelling tasks — forecasting, scoring, ranking — ALE
provides exactly the right summary of how the model uses each feature.
The causal caveats only matter when you want to USE the explanation to
justify interventions in the world.


##### PART VII: A COMPLETE WORKED EXAMPLE BY HAND

### 6 Instances, 2 Features, 3 Intervals — Full Trace

Let's compute ALE completely by hand for a tiny dataset so every step
is transparent. We will compute ALE for feature x₁ (income).

    Dataset (n=6):
    ─────────────────────────────────────────────────────
    i    x₁ (income, $k)    x₂ (credit, normalised)
    1         20                    -0.5
    2         30                    -0.2
    3         40                     0.1
    4         50                     0.3
    5         60                     0.6
    6         70                     0.8
    ─────────────────────────────────────────────────────

    Model: f(x₁, x₂) = 0.01·x₁ + 0.3·x₂ + 0.005·x₁·x₂

    Partition x₁ into K=3 equal-count intervals (2 instances each):
      Interval 1: [20, 35)  →  instances 1, 2  (i=1,2)
      Interval 2: [35, 55)  →  instances 3, 4  (i=3,4)
      Interval 3: [55, 70]  →  instances 5, 6  (i=5,6)
      Boundaries: z₀=20, z₁=35, z₂=55, z₃=70

    STEP 2 — LOCAL DIFFERENCES:

    Interval 1 [20→35], instances {1,2}:
      Instance 1: x₂=−0.5
        f(35, −0.5) = 0.01(35) + 0.3(−0.5) + 0.005(35)(−0.5)
                    = 0.350 − 0.150 − 0.0875 = 0.1125
        f(20, −0.5) = 0.01(20) + 0.3(−0.5) + 0.005(20)(−0.5)
                    = 0.200 − 0.150 − 0.050  = 0.0000
        δ₁ = 0.1125 − 0.0000 = +0.1125

      Instance 2: x₂=−0.2
        f(35, −0.2) = 0.350 + 0.3(−0.2) + 0.005(35)(−0.2)
                    = 0.350 − 0.060 − 0.035  = 0.2550
        f(20, −0.2) = 0.200 + 0.3(−0.2) + 0.005(20)(−0.2)
                    = 0.200 − 0.060 − 0.020  = 0.1200
        δ₂ = 0.2550 − 0.1200 = +0.1350

      Δ₁ = (δ₁ + δ₂) / 2 = (0.1125 + 0.1350) / 2 = +0.1238

    Interval 2 [35→55], instances {3,4}:
      Instance 3: x₂=0.1
        f(55, 0.1) = 0.550 + 0.030 + 0.005(55)(0.1) = 0.550+0.030+0.0275 = 0.6075
        f(35, 0.1) = 0.350 + 0.030 + 0.005(35)(0.1) = 0.350+0.030+0.0175 = 0.3975
        δ₃ = 0.6075 − 0.3975 = +0.2100

      Instance 4: x₂=0.3
        f(55, 0.3) = 0.550 + 0.090 + 0.005(55)(0.3) = 0.550+0.090+0.0825 = 0.7225
        f(35, 0.3) = 0.350 + 0.090 + 0.005(35)(0.3) = 0.350+0.090+0.0525 = 0.4925
        δ₄ = 0.7225 − 0.4925 = +0.2300

      Δ₂ = (0.2100 + 0.2300) / 2 = +0.2200

    Interval 3 [55→70], instances {5,6}:
      Instance 5: x₂=0.6
        f(70, 0.6) = 0.700 + 0.180 + 0.005(70)(0.6) = 0.700+0.180+0.210 = 1.090
        f(55, 0.6) = 0.550 + 0.180 + 0.005(55)(0.6) = 0.550+0.180+0.165 = 0.895
        δ₅ = 1.090 − 0.895 = +0.195

      Instance 6: x₂=0.8
        f(70, 0.8) = 0.700 + 0.240 + 0.005(70)(0.8) = 0.700+0.240+0.280 = 1.220
        f(55, 0.8) = 0.550 + 0.240 + 0.005(55)(0.8) = 0.550+0.240+0.220 = 1.010
        δ₆ = 1.220 − 1.010 = +0.210

      Δ₃ = (0.195 + 0.210) / 2 = +0.2025

    STEP 3 — ACCUMULATE at each right boundary:
      ALE_raw(z₁=35) = Δ₁               = +0.1238
      ALE_raw(z₂=55) = Δ₁ + Δ₂         = +0.1238 + 0.2200 = +0.3438
      ALE_raw(z₃=70) = Δ₁ + Δ₂ + Δ₃   = +0.3438 + 0.2025 = +0.5463

    STEP 4 — CENTRE (subtract training-weighted mean):
      Assign each instance its ALE value at the right boundary of its interval:
        Instances 1,2 → ALE_raw(z₁) = 0.1238
        Instances 3,4 → ALE_raw(z₂) = 0.3438
        Instances 5,6 → ALE_raw(z₃) = 0.5463

      Mean = (2×0.1238 + 2×0.3438 + 2×0.5463) / 6
           = (0.2476 + 0.6876 + 1.0926) / 6 = 2.0278 / 6 = 0.3380

      ALE_centred(35) = 0.1238 − 0.3380 = −0.2142
      ALE_centred(55) = 0.3438 − 0.3380 = +0.0058
      ALE_centred(70) = 0.5463 − 0.3380 = +0.2083

    FINAL ALE CURVE for x₁ (income):
    ───────────────────────────────────────────────────────────
    income (x₁)         ALE_centred    Interpretation
    ───────────────────────────────────────────────────────────
    z₁ = 35k             −0.2142       below-average prediction
    z₂ = 55k             +0.0058       near-average prediction
    z₃ = 70k             +0.2083       above-average prediction
    ───────────────────────────────────────────────────────────

    The ALE is monotonically increasing — higher income → higher
    model output. The effect accelerates: the step from 35→55 (+0.21)
    is larger than from 20→35 (+0.12) because the interaction term
    0.005·x₁·x₂ amplifies as BOTH x₁ and the correlated x₂ grow.

    Notice that x₂ (credit) is HELD AT ITS REAL VALUES for each
    instance throughout. It never gets artificially paired with an
    unrealistic income value. This is why ALE is unbiased.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔═════════════════════╦═══════════╦══════════════╦══════════════╦═══════════════╗
    ║ Property            ║ PDP       ║ M-plot       ║ ALE (1st)    ║ ALE (2nd)     ║
    ╠═════════════════════╬═══════════╬══════════════╬══════════════╬═══════════════╣
    ║ Unbiased (corr.)    ║ No        ║ No           ║ Yes          ║ Yes           ║
    ║ Extrapolation-free  ║ No        ║ Yes          ║ Yes          ║ Yes           ║
    ║ Effect mixing       ║ Moderate  ║ Severe       ║ None         ║ None          ║
    ║ Measures            ║ Main eff. ║ Total effect ║ Main effect  ║ Interaction   ║
    ║                     ║ + bias    ║ (confounded) ║ (pure)       ║ effect (pure) ║
    ║ Model queries       ║ n×|grid|  ║ n×|grid|     ║ 2n×K         ║ 4n×K₁×K₂      ║
    ║ Interpretable at    ║ All xⱼ     ║ All xⱼ       ║ Data range   ║ Joint range    ║
    ║ Centred             ║ Optional  ║ Optional     ║ Yes          ║ Yes           ║
    ║ Handles categoricals║ With enc. ║ With enc.    ║ Native       ║ Native        ║
    ║ Interaction detect  ║ Partially ║ No           ║ No (1st)     ║ Yes           ║
    ║ Variance in tails   ║ Low       ║ Low          ║ Higher       ║ Higher        ║
    ╚═════════════════════╩═══════════╩══════════════╩══════════════╩═══════════════╝

    Key formula:
      ALE_j(x) = ∫_{z₀}^{x} E_{X_\\j | Xⱼ=v} [ ∂f/∂xⱼ (v, X_\\j) ] dv − const

    Discrete approximation:
      Δₖ = (1/|Nₖ|) Σᵢ∈Nₖ [ f(zₖ, Xᵢ_\\j) − f(z_{k-1}, Xᵢ_\\j) ]
      ALE(x) = Σ_{k : zₖ ≤ x} Δₖ − centre
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "ALE from Scratch — PDP vs M-plot vs ALE on Correlated Data": {
        "description": (
            "Builds a dataset with correlated features and a model with an interaction "
            "term. Implements PDP, M-plot, and ALE from scratch side-by-side. Computes "
            "the true analytical main effect as the ground truth, then measures how "
            "far each method deviates from it. Demonstrates that ALE recovers the true "
            "main effect while PDP is biased and M-plot mixes in the interaction. "
            "Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "ale",
        "code": '''
"""
================================================================================
ALE FROM SCRATCH — PDP vs M-PLOT vs ALE ON CORRELATED DATA
================================================================================

We build a controlled experiment to make the bias of PDP and M-plot
measurable. We use a model with a known analytical form, compute the TRUE
main effects algebraically, then compare all three methods against that truth.

Setup:
  Features:  x₁ (income, standardised), x₂ (credit, standardised)
  Correlation: x₂ = 0.70·x₁ + 0.71·ε  (ρ ≈ 0.70)
  Model: f(x₁, x₂) = 0.40·x₁ + 0.30·x₂ + 0.20·x₁·x₂

  True MAIN EFFECT of x₁ (functional ANOVA definition):
    m₁(x₁) = ∫ [f(x₁,x₂) - E_{x₁}[f(x₁,x₂)]] p(x₂) dx₂   [centred]
    For our additive+interaction model:
      m₁(v) = 0.40v + 0.20v·E[x₂] = 0.40v  (since E[x₂]=0 by standardisation)
    So the TRUE main effect of x₁ has slope exactly 0.40.

  What each method should recover:
    PDP slope:    0.40 + 0.20·γ·Var[x₁] / Var[x₁]  (biased by correlation+interaction)
    M-plot slope: 0.40 + (0.30+0.20·v)·γ  (mixes x₂'s effect in via correlation γ)
    ALE slope:    0.40  (correct — should match the true main effect)
================================================================================
"""

import random
import math


random.seed(99)

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

N   = 800
RHO = 0.70  # target correlation between x₁ and x₂

# Generate x₁ ~ N(0,1), then x₂ = RHO·x₁ + sqrt(1-RHO²)·noise
# This gives Cor(x₁, x₂) = RHO exactly in population.
x1_list, x2_list = [], []
for _ in range(N):
    z1 = random.gauss(0, 1)
    z2 = random.gauss(0, 1)
    x1 = z1
    x2 = RHO * z1 + math.sqrt(1 - RHO**2) * z2
    x1_list.append(x1)
    x2_list.append(x2)

def model(a, b):
    return 0.40 * a + 0.30 * b + 0.20 * a * b

preds = [model(x1_list[i], x2_list[i]) for i in range(N)]

# Verify correlation
mean1 = sum(x1_list) / N
mean2 = sum(x2_list) / N
cov   = sum((x1_list[i]-mean1)*(x2_list[i]-mean2) for i in range(N)) / N
std1  = math.sqrt(sum((v-mean1)**2 for v in x1_list) / N)
std2  = math.sqrt(sum((v-mean2)**2 for v in x2_list) / N)
rho_empirical = cov / (std1 * std2 + 1e-12)

print("=" * 68)
print("  DATASET")
print("=" * 68)
print(f"  N = {N} instances")
print(f"  Target correlation (ρ):  {RHO:.2f}")
print(f"  Empirical correlation:   {rho_empirical:.4f}")
print(f"  True model:  f = 0.40·x₁ + 0.30·x₂ + 0.20·x₁·x₂")
print(f"  True MAIN EFFECT of x₁: slope = 0.40  (from functional ANOVA)")
print(f"  (E[x₂] = 0 so interaction averages out in the main effect)")

# ─────────────────────────────────────────────────────────────────────────────
# GRID: 20 evenly-spaced values from min to max of x₁
# ─────────────────────────────────────────────────────────────────────────────

G = 20
x1_min, x1_max = min(x1_list), max(x1_list)
grid = [x1_min + (x1_max - x1_min) * k / (G - 1) for k in range(G)]

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — PDP
# Fix x₁ = v for each grid point; average over ALL x₂ values (marginal)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pdp(grid, x2_list):
    return [(1/N) * sum(model(v, x2_list[i]) for i in range(N))
            for v in grid]

pdp = compute_pdp(grid, x2_list)

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — M-PLOT
# Fix x₁ = v; average over CONDITIONAL distribution of x₂ given x₁ ≈ v.
# We select the nearest 15% of instances by |x₁ᵢ - v|.
# ─────────────────────────────────────────────────────────────────────────────

def compute_mplot(grid, x1_list, x2_list, bandwidth_fraction=0.15):
    k_neighbours = max(10, int(N * bandwidth_fraction))
    result = []
    for v in grid:
        dists = sorted(range(N), key=lambda i: abs(x1_list[i] - v))
        neighbours = dists[:k_neighbours]
        avg = sum(model(v, x2_list[i]) for i in neighbours) / k_neighbours
        result.append(avg)
    return result

mplot = compute_mplot(grid, x1_list, x2_list)

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — ALE
# Partition x₁ into K quantile intervals.
# For each interval, compute average of [f(right, x₂ᵢ) - f(left, x₂ᵢ)]
# over instances INSIDE that interval. Accumulate and centre.
# ─────────────────────────────────────────────────────────────────────────────

def compute_ale(x1_list, x2_list, K=40):
    # Quantile-based boundaries
    sorted_x1 = sorted(x1_list)
    boundaries = []
    for k in range(K + 1):
        idx = int(k * (N - 1) / K)
        boundaries.append(sorted_x1[idx])
    boundaries[0]  -= 1e-9  # ensure left boundary is strictly less than minimum
    boundaries[-1] += 1e-9

    deltas = []
    counts = []

    for k in range(K):
        lo, hi = boundaries[k], boundaries[k + 1]
        in_interval = [i for i in range(N) if lo <= x1_list[i] < hi]
        if not in_interval:
            deltas.append(0.0)
            counts.append(0)
            continue
        local_diffs = [model(hi, x2_list[i]) - model(lo, x2_list[i])
                       for i in in_interval]
        deltas.append(sum(local_diffs) / len(local_diffs))
        counts.append(len(in_interval))

    # Accumulate to grid: ALE_raw at right boundary of each interval
    ale_raw = [0.0] * (K + 1)
    for k in range(K):
        ale_raw[k + 1] = ale_raw[k] + deltas[k]

    # Centre: subtract weighted mean (each interval gets weight = count)
    total_count = sum(counts)
    weighted_mean = sum(
        ale_raw[k + 1] * counts[k] for k in range(K)
    ) / (total_count if total_count > 0 else 1)

    ale_centred_at_boundaries = [v - weighted_mean for v in ale_raw]

    # Interpolate ALE at the uniform grid points
    def interp(v):
        # Find which interval v falls in
        for k in range(K):
            if boundaries[k] <= v < boundaries[k + 1]:
                lo, hi = boundaries[k], boundaries[k + 1]
                t = (v - lo) / (hi - lo + 1e-12)
                return (ale_centred_at_boundaries[k] * (1 - t) +
                        ale_centred_at_boundaries[k + 1] * t)
        if v >= boundaries[-1]:
            return ale_centred_at_boundaries[-1]
        return ale_centred_at_boundaries[0]

    return [interp(v) for v in grid], boundaries, deltas, counts, ale_centred_at_boundaries

ale, boundaries, deltas, interval_counts, ale_at_boundaries = compute_ale(
    x1_list, x2_list, K=40
)

# ─────────────────────────────────────────────────────────────────────────────
# TRUE MAIN EFFECT (analytical)
# m₁(v) = 0.40·v  (centred to have mean zero over the data distribution)
# The mean of 0.40·x₁ over our data is 0.40·mean(x₁) ≈ 0 since x₁ is std.
# ─────────────────────────────────────────────────────────────────────────────

mean_x1 = sum(x1_list) / N
true_ale = [0.40 * (v - mean_x1) for v in grid]

# ─────────────────────────────────────────────────────────────────────────────
# CENTRE PDP AND M-PLOT for comparison (subtract their means)
# ─────────────────────────────────────────────────────────────────────────────

def centre(curve):
    m = sum(curve) / len(curve)
    return [v - m for v in curve]

pdp_c   = centre(pdp)
mplot_c = centre(mplot)
# ALE is already centred

# ─────────────────────────────────────────────────────────────────────────────
# PRINT THE CURVES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  MARGINAL EFFECT CURVES FOR x₁  (centred, every 2nd grid point)")
print("  True main effect slope = 0.40")
print("=" * 68)
print()
print(f"  {'x₁':>6}  {'True':>8}  {'PDP':>8}  {'M-plot':>8}  {'ALE':>8}")
print(f"  {'-'*46}")
for k in range(0, G, 2):
    v = grid[k]
    print(f"  {v:>6.3f}  {true_ale[k]:>8.4f}  {pdp_c[k]:>8.4f}  "
          f"{mplot_c[k]:>8.4f}  {ale[k]:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SLOPE COMPARISON — fit a line to each curve, compare to truth = 0.40
# ─────────────────────────────────────────────────────────────────────────────

def fit_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    den = sum((xs[i]-mx)**2 for i in range(n))
    return num / (den + 1e-12)

slope_true  = fit_slope(grid, true_ale)
slope_pdp   = fit_slope(grid, pdp_c)
slope_mplot = fit_slope(grid, mplot_c)
slope_ale   = fit_slope(grid, ale)

print()
print("=" * 68)
print("  SLOPE COMPARISON — recovered main effect of x₁")
print("=" * 68)
print()
print(f"  True main effect slope : {slope_true:.4f}  (analytical ground truth)")
print()
print(f"  PDP slope              : {slope_pdp:.4f}  "
      f"(error: {slope_pdp - slope_true:+.4f},"
      f" {abs(slope_pdp-slope_true)/slope_true*100:.1f}% off)")
print(f"  M-plot slope           : {slope_mplot:.4f}  "
      f"(error: {slope_mplot - slope_true:+.4f},"
      f" {abs(slope_mplot-slope_true)/slope_true*100:.1f}% off)")
print(f"  ALE slope              : {slope_ale:.4f}  "
      f"(error: {slope_ale - slope_true:+.4f},"
      f" {abs(slope_ale-slope_true)/slope_true*100:.1f}% off)")

print()
print("  WHY EACH METHOD GIVES A DIFFERENT ANSWER:")
print()
print(f"  PDP slope = true + bias from correlation:")
pdp_theory = 0.40 + 0.20 * rho_empirical * (std1**2) / (std1**2)
print(f"    Expected: 0.40 + 0.20·ρ·σ₁²/σ₁² ≈ 0.40 + 0.20×{rho_empirical:.2f}")
print(f"    = {0.40 + 0.20*rho_empirical:.4f}")
print(f"    (PDP averages over marginal x₂, giving x₁ credit for the")
print(f"     interaction term 0.20·x₁·x₂ via the x₁-x₂ correlation)")
print()
print(f"  M-plot slope ≈ true + x₂'s direct effect via correlation:")
print(f"    Expected ≈ 0.40 + 0.30·ρ = 0.40 + 0.30×{rho_empirical:.2f}")
print(f"    = {0.40 + 0.30*rho_empirical:.4f}")
print(f"    (M-plot conditions on x₁=v, so x₂ moves with it — x₂'s")
print(f"     effect 0.30·x₂ contaminates x₁'s slope via ρ)")
print()
print(f"  ALE slope ≈ true main effect only:")
print(f"    Expected: 0.40 (x₂ held fixed within each interval)")
print(f"    (Local differences cancel x₂'s baseline — only x₁'s")
print(f"     pure effect remains.)")


# ─────────────────────────────────────────────────────────────────────────────
# MEAN SQUARED ERROR VS TRUE CURVE
# ─────────────────────────────────────────────────────────────────────────────

def mse(a, b):
    return sum((a[i]-b[i])**2 for i in range(len(a))) / len(a)

print()
print("=" * 68)
print("  MEAN SQUARED ERROR VS TRUE MAIN EFFECT CURVE")
print("=" * 68)
print()
print(f"  PDP   MSE vs true: {mse(pdp_c,   true_ale):.6f}")
print(f"  Mplot MSE vs true: {mse(mplot_c, true_ale):.6f}")
print(f"  ALE   MSE vs true: {mse(ale,     true_ale):.6f}")

best = min([("PDP",mse(pdp_c,true_ale)),
            ("M-plot",mse(mplot_c,true_ale)),
            ("ALE",mse(ale,true_ale))], key=lambda x: x[1])
print()
print(f"  Closest to truth: {best[0]} (MSE = {best[1]:.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# ALE INTERVAL DIAGNOSTICS — show counts and deltas
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  ALE INTERVAL DIAGNOSTICS (first 8 and last 4 intervals)")
print("=" * 68)
print()
print(f"  {'Interval':>10}  {'Left':>7}  {'Right':>7}  {'Count':>7}  "
      f"{'Δ (local eff)':>14}  {'ALE accum':>10}")
print(f"  {'-'*65}")

K = len(deltas)
ale_accum = 0.0
for k in list(range(min(8, K))) + list(range(max(0, K-4), K)):
    if k == 8 and K > 12:
        print(f"  {'  ...':>10}  {'...':>7}  {'...':>7}  {'...':>7}  "
              f"{'...':>14}  {'...':>10}")
    ale_accum = sum(deltas[:k+1])
    lo = boundaries[k]
    hi = boundaries[k + 1]
    print(f"  [{k+1:>3}/{K}]  {lo:>7.3f}  {hi:>7.3f}  "
          f"{interval_counts[k]:>7}  {deltas[k]:>+14.5f}  {ale_accum:>10.5f}")

print()
print("  INTERPRETATION:")
print("  • Each Δ is the average change in f as x₁ moves across one interval,")
print("    computed ONLY for instances that actually live in that interval.")
print("  • All Δs are positive here: more income always helps.")
print("  • Δs grow slightly as x₁ increases — this captures the interaction")
print("    term 0.20·x₁·x₂: at higher x₁, x₂ is also higher (correlated),")
print("    so the derivative ∂f/∂x₁ = 0.40 + 0.20·x₂ increases with x₂.")
print("  • ALE correctly ACCUMULATES these local slopes into a global curve.")
print("  • PDP would show the same increasing curvature, but inflated by")
print("    the marginal x₂ values rather than the conditional ones.")


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY TO K — BIAS-VARIANCE TRADE-OFF
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SENSITIVITY TO K (number of intervals)")
print("=" * 68)
print()

for K_test in [5, 20, 40, 80, 200]:
    ale_k, _, _, _, _ = compute_ale(x1_list, x2_list, K=K_test)
    mse_k = mse(ale_k, true_ale)
    slope_k = fit_slope(grid, ale_k)
    min_count = min(
        len([i for i in range(N)
             if sorted(x1_list)[int(kk*(N-1)/K_test)]
             <= x1_list[i] <
             sorted(x1_list)[min(N-1, int((kk+1)*(N-1)/K_test))+1
                              if int((kk+1)*(N-1)/K_test)+1 < N else N-1]])
        for kk in range(K_test)
    ) if K_test <= 100 else "N/A"
    print(f"  K={K_test:>4}:  slope={slope_k:.4f}  MSE={mse_k:.6f}  "
          f"min interval count≈{N//K_test}")

print()
print("  • Small K (K=5): few intervals, many instances each, low variance")
print("    but coarse — misses curvature. Slope is slightly off.")
print("  • K=40: good balance — each interval has ~20 instances, curve")
print("    captures the true shape, slope is accurate.")
print("  • Large K (K=200): very few instances per interval (~4), high")
print("    variance. MSE rises because noise dominates. Slope wanders.")
print("  Practical rule: aim for ≥ 10–20 instances per interval.")
print("  For N=800, K=40 gives N/K = 20. K=100 gives 8 — borderline.")
''',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    import streamlit as st

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


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    # try:
    #     from interpretability.visuals.ale import (
    #         ALE_VISUAL_HTML,
    #         ALE_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ALE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ALE_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[ale.py] Could not load visual: {e}", stacklevel=2)

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