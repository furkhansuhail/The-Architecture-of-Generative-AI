"""
ICE — Individual Conditional Expectation
=========================================

ICE plots (Goldstein, Kapelner, Bleich & Pitkin, 2015) are the instance-level
microscope for Partial Dependence Plots. Where a PDP shows the average effect
of a feature across all instances, an ICE plot shows one line per instance —
revealing the full distribution of effects and making heterogeneity and
feature interactions directly visible.

This module derives ICE from the PDP failure case, establishes the precise
relationship between ICE and PDP, develops the centred ICE (c-ICE) variant,
and explains the diagnostic workflow for detecting interactions.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "ICE — Individual Conditional Expectation"
DISPLAY_NAME = "03e · ICE Plots"
ICON = "🧵"
SUBTITLE = "Instance-level marginal effect curves and interaction detection"


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

### The Problem ICE Solves

The Partial Dependence Plot (PDP) summarises the relationship between one
feature and the model's output by averaging over all instances. This average
is sometimes exactly what you want. But averaging can destroy information.

Consider two groups of customers in a bank. For Group A (high savings), the
model's approval probability increases strongly with income. For Group B (low
savings), the model barely responds to income at all — creditworthiness is
dominated by savings, not income. A PDP for income averages these two opposing
effects and shows a moderate, blended slope. It is neither wrong nor lying;
it is simply answering a question you did not actually ask.

The question you usually want to ask is:

    "Does the model treat ALL instances the same way as feature j varies,
     or does its response differ from one instance to another?"

ICE answers this directly by refusing to average. Instead of collapsing all
instances into one curve, it shows every instance as its own line. You read
the distribution of effects, not the mean of effects.

    # ================================================================== #
    **The averaging problem — what PDP hides:**

    ICE curves (4 instances):      PDP (mean of all):

    f ↑                            f ↑
    1 │      /  /                  1 │        /
      │     /  /  ← steep for        │       /
      │    /  /    high-savings      │      /
    0 │───/──/                    0 │─────/──── ← average slope
      │  /  ─────  ← flat for       │    /         hides the split
     -1│ /  ─────   low-savings    -1│   /
      └──────────────→ income       └──────────────→ income

    The PDP line sits between the two groups. It correctly describes
    no individual. ICE shows the full truth: two distinct populations
    with completely different income sensitivity.
    # ================================================================== #


##### PART I: FORMAL DEFINITION AND RELATIONSHIP TO PDP

### The ICE Curve for One Instance

Let f be the trained model and let x be a p-dimensional feature vector.
Write xᵢ = (xᵢ_j, xᵢ_\j) where xᵢ_j is the value of the j-th feature for
instance i, and xᵢ_\j is the vector of all other feature values for instance i.

The ICE curve for instance i, varying feature j, is the function:

    ICE_j^(i)(v)  =  f(v, xᵢ_\j)

where v sweeps across a grid of candidate values for feature j, and all
other features are FIXED at instance i's observed values.

In words: take instance i. Replace its j-th feature with each candidate value
v in turn. Record the model's prediction each time. Plot those predictions
against v. That one-instance trajectory is its ICE curve.

For a dataset of n instances you get n ICE curves — one per row of training data.

The ICE curve ICE_j^(i)(v) asks a precise hypothetical:

    "If everything about instance i stayed the same EXCEPT that feature j
     were v instead of its actual value, what would the model predict?"

This is a CETERIS PARIBUS (all else equal) query — one of the cleanest
interpretive quantities in XAI.


### The PDP as the Mean of ICE Curves

The Partial Dependence Plot is exactly the pointwise average of all ICE curves:

    PDP_j(v) = (1/n) Σᵢ ICE_j^(i)(v)

This is not a definition of PDP — it is a theorem that connects the two.
The standard PDP formula substitutes xᵢ_\j values into the model for each
instance i:

    PDP_j(v) = (1/n) Σᵢ f(v, xᵢ_\j)

which is, by definition, the average of the n ICE curves evaluated at v.

This relationship has two important consequences:

    CONSEQUENCE 1 — PDP IS A SUMMARY OF ICE:
    ─────────────────────────────────────────
    Any property visible in the PDP is also present in the ICE curves —
    the PDP just averages it away. If the PDP shows a positive slope,
    at least some (possibly all) ICE curves slope upward.

    CONSEQUENCE 2 — ICE REVEALS WHAT PDP HIDES:
    ─────────────────────────────────────────────
    If ICE curves vary substantially from one another (different slopes,
    different shapes, some crossing others), the PDP average misrepresents
    EVERY individual instance. In the extreme case where half the ICE curves
    slope up and half slope down, the PDP is flat — and flatness is completely
    misleading. There IS a strong effect; it just cancels in the average.


### The ICE Grid — Evaluation Points

In practice, feature j is swept over a finite grid of G values:

    v₁ < v₂ < ... < vG

Common choices:
  • Quantile grid: the G quantiles of the observed xᵢ_j values.
    Ensures grid points are in data-dense regions.
  • Equal-width grid: evenly spaced from min(xᵢ_j) to max(xᵢ_j).
    Simple but wastes resolution in sparse regions.

For each instance i and each grid point vₖ, one model call is made:
    f(vₖ, xᵢ_\j)

Total model calls: n × G. For large n and large G, this is expensive.
A common strategy: subsample n_sub < n instances, compute ICE on the
subsample, and overlay the PDP (which uses all n) for context.

The number of model queries is the primary computational cost of ICE. For
a random forest with n=10,000 and G=50 grid points, ICE requires 500,000
model evaluations. For a neural network, this may take minutes. For a fast
tree model, it is nearly instant.


##### PART II: READING ICE PLOTS — THE FIVE DIAGNOSTIC PATTERNS

### Pattern Recognition in ICE Curves

An ICE plot communicates through the geometry of its bundle of lines. Five
patterns cover the vast majority of real-world cases.

    # ================================================================== #
    PATTERN 1 — PARALLEL LINES (no interaction, homogeneous effect)

    f ↑
      │  /  /  /  /  /
      │ /  /  /  /  /     All curves run in the same direction
      │/  /  /  /  /      with the same slope.
      └────────────────→ xⱼ

    WHAT THIS MEANS:
    The effect of feature j is the SAME for every instance — it does
    not depend on the values of other features. Feature j acts
    independently of all other features. The PDP is a faithful
    summary: it tells you the complete story.

    Statistical test: compute the variance of slopes across ICE curves.
    If it is near zero, lines are parallel.
    # ================================================================== #

    # ================================================================== #
    PATTERN 2 — PARALLEL BUT SPREAD (no interaction, heterogeneous level)

    f ↑
    5 │      /  /  /  /
      │     /  /  /  /    All curves have the same slope, but start
    0 │    /  /  /  /     at very different heights. They never cross.
      │   /  /  /  /
   -5 │  /  /  /  /
      └────────────────→ xⱼ

    WHAT THIS MEANS:
    The EFFECT of feature j (change in prediction as j varies) is the
    same for everyone. But the BASELINE prediction level differs across
    instances — other features shift each curve up or down, while j's
    contribution is uniform.

    The PDP slope is correct, but the PDP does not reveal the spread in
    baseline levels — which tells you that other features (not j) are
    responsible for large differences in predictions.

    This pattern is easier to see in centred ICE (c-ICE) — see Part III.
    # ================================================================== #

    # ================================================================== #
    PATTERN 3 — FAN (interaction: effect gets stronger)

    f ↑
      │              /
      │          /  /
      │      /  /  /      Curves that start bunched at the left diverge
      │  /  /  /  /       as xⱼ increases.
      │ /  /  /  /
      └────────────────→ xⱼ

    WHAT THIS MEANS:
    The effect of feature j is AMPLIFIED for some instances. Typically,
    the instances with steeper slopes have higher values of some OTHER
    feature xₖ that interacts with xⱼ positively.

    Example: if xⱼ = income and xₖ = credit_score, a model might only
    strongly reward income for applicants who already have a decent
    credit score (positive interaction). Low-credit applicants gain
    little from higher income, producing flat ICE lines. High-credit
    applicants gain a lot, producing steep lines. A fan forms.

    The PDP shows an average slope — the truth for no individual.
    # ================================================================== #

    # ================================================================== #
    PATTERN 4 — CROSSING LINES (interaction: effect reverses sign)

    f ↑
      │  \    /
      │   \  /     Some curves slope UP, others slope DOWN.
      │    \/      They cross somewhere in the middle.
      │    /\
      │   /  \
      └────────────────→ xⱼ

    WHAT THIS MEANS:
    Feature j has a POSITIVE effect for some instances and a NEGATIVE
    effect for others. The PDP may show a flat line (if the up and down
    effects roughly cancel) or a misleadingly small slope.

    This is the most alarming pattern: the PDP appears to say "feature j
    has no effect" when in reality it has a large, sign-reversing effect
    depending on some other feature.

    Example: a drug dosage might help patients with a certain genotype
    and harm patients with another. Averaged over genotypes, the PDP
    looks flat. ICE reveals the heterogeneity.

    Action: investigate WHAT drives the crossing. Sort instances by the
    variable you suspect drives the interaction. Do high-xₖ instances
    slope one way and low-xₖ instances slope the other?
    # ================================================================== #

    # ================================================================== #
    PATTERN 5 — NON-LINEAR INDIVIDUAL CURVES (threshold or plateau effects)

    f ↑
      │     ___----       Some ICE curves are not straight lines.
      │___--              They may show kinks, thresholds, or saturation.
      │
      │___----___         A curve that rises and then flattens means:
      └────────────────→  the model learns a diminishing-returns
                          relationship for this instance.

    WHAT THIS MEANS:
    The model's learned function for feature j is non-linear. This might
    or might not be reflected in the PDP, depending on whether the
    non-linearity is consistent across instances (PDP would show it) or
    varies (PDP average might smooth it out).

    A threshold: ICE curves are flat, then jump at a specific value of
    xⱼ. This is common in tree-based models whose splits create sharp
    boundaries. The jump location may vary by instance (different splits
    fire for different instances), making the PDP appear as a gradual
    slope when the truth is a population of sharp steps.
    # ================================================================== #


### Reading the Bundle as a Whole

Beyond individual pattern recognition, the ICE bundle as a whole carries
information:

    WIDTH OF THE BUNDLE:
    How spread apart are the ICE curves? Wide spread means different
    instances get very different absolute predictions — the model is
    strongly stratifying the population. Narrow spread means predictions
    are similar across instances.

    CONSISTENCY OF DIRECTION:
    Do all ICE curves go up, all go down, or do some go each way?
    Consistent direction = the feature has a clear, unambiguous effect.
    Mixed direction = the effect is heterogeneous or reverses.

    LOCATION OF CROSSINGS:
    Where along the xⱼ axis do curves cross? If they all cross near the
    same value, that value is a threshold — instances change their relative
    ordering at that point. If crossings are scattered, the interaction is
    more diffuse.

    DENSITY AT EXTREMES:
    ICE curves at extreme values of xⱼ are extrapolations — the model is
    evaluated on feature values that rarely co-occur with the instance's
    other features. Lines that behave erratically at the extremes may
    indicate model instability in sparse data regions, not a real effect.


##### PART III: CENTRED ICE (c-ICE) — ISOLATING THE EFFECT

### The Problem with Raw ICE for Detecting Interactions

Raw ICE curves have a visual problem: when the baseline spread between
instances is large (Pattern 2), the curves are vertically offset by large
amounts. This makes it hard to compare slopes. Curves that are parallel
in slope look non-parallel because they start at very different heights.

    # ================================================================== #
    **Raw ICE vs c-ICE — the vertical offset problem:**

    RAW ICE:                          c-ICE (subtract left endpoint):

    f ↑                               Δf ↑
    3 │    /  ← high-credit              │  /  /  ← same slope
      │   /     baseline=3              │ /  /
    1 │  /                            0 │/  /
      │ /  ← low-credit                 │    (lines now overlap perfectly)
   -1 │/     baseline=-1
      └────────────→ xⱼ               └────────────→ xⱼ

    In raw ICE: the high-credit curve is above the low-credit curve
    throughout, making them look like they could have different shapes.
    In c-ICE: both curves are anchored to 0 at the left edge.
    Now you can directly compare slopes — and see they are identical.
    # ================================================================== #

### The c-ICE Definition

Centred ICE for instance i, varying feature j, is:

    c-ICE_j^(i)(v)  =  ICE_j^(i)(v) − ICE_j^(i)(v₁)
                     =  f(v, xᵢ_\j) − f(v₁, xᵢ_\j)

where v₁ is the leftmost (minimum) grid point.

Every c-ICE curve starts at 0 (at v = v₁) by construction. What remains
is purely the CHANGE in prediction as xⱼ varies from its minimum to v,
with all other features held fixed for that instance.

The c-PDP is correspondingly:

    c-PDP(v) = PDP(v) − PDP(v₁)
             = (1/n) Σᵢ c-ICE_j^(i)(v)

c-ICE removes the vertical offsets that encode OTHER features' effects,
leaving only the shape of the j-th feature's contribution.

    WHEN TO USE RAW ICE vs c-ICE:
    ─────────────────────────────────────────────────────────────────
    Raw ICE:   When you care about ABSOLUTE prediction levels.
               Shows how different instances are stratified overall.
               Useful for understanding which instances get high vs
               low predictions.

    c-ICE:     When you care about the SHAPE of xⱼ's effect.
               Shows whether the slope (or curvature, or threshold
               location) varies across instances.
               Directly reveals interactions because vertical offsets
               (from other features) are subtracted away.
               Almost always more useful for interaction detection.
    ─────────────────────────────────────────────────────────────────

### What c-ICE Reveals about Interactions

After centring, all curves start at 0. The SPREAD of the curves at any
later grid point v tells you:

    c-ICE_j^(i)(v) − c-ICE_j^(k)(v)
    = [f(v, xᵢ_\j) − f(v₁, xᵢ_\j)] − [f(v, xₖ_\j) − f(v₁, xₖ_\j)]

This difference is the INTERACTION EFFECT: how much the effect of xⱼ
changing from v₁ to v depends on the identity of the instance (i.e., on
its other feature values xᵢ_\j vs xₖ_\j).

If all c-ICE curves coincide (zero spread), then:
    c-ICE_j^(i)(v) = c-ICE_j^(k)(v) for all i, k, v
    ⟹ f(v, xᵢ_\j) − f(v₁, xᵢ_\j) is the same for all i
    ⟹ the effect of xⱼ is identical across instances
    ⟹ NO interaction between xⱼ and any other feature

If c-ICE curves spread apart:
    The effect of xⱼ depends on who the instance is
    ⟹ there IS an interaction between xⱼ and something in x_\j

This makes c-ICE a direct, visual test for the absence of interactions.

    # ================================================================== #
    **Formal interaction condition:**

    For an additive model f(x) = g_j(xⱼ) + h(x_\j):

    ICE_j^(i)(v) = g_j(v) + h(xᵢ_\j)
    c-ICE_j^(i)(v) = g_j(v) − g_j(v₁)

    The h(xᵢ_\j) term VANISHES in c-ICE. All c-ICE curves become
    exactly g_j(v) − g_j(v₁) — identical for every instance i.

    For a model with interaction f(x) = g_j(xⱼ) + h(x_\j) + xⱼ·xₖ:

    ICE_j^(i)(v) = g_j(v) + h(xᵢ_\j) + v·xᵢ_k
    c-ICE_j^(i)(v) = [g_j(v)−g_j(v₁)] + [v−v₁]·xᵢ_k

    The interaction term [v−v₁]·xᵢ_k SURVIVES centring. Its magnitude
    scales with xᵢ_k, so instances with high xₖ get steeper c-ICE curves.
    The spread of c-ICE at any v equals Var[xᵢ_k] × (v−v₁)².
    # ================================================================== #


##### PART IV: ICE AS AN INTERACTION DETECTOR — THE STATISTICS

### The Standard Deviation of c-ICE Curves

The most principled numeric summary of an ICE bundle is the pointwise
standard deviation of c-ICE values across instances:

    SD_j(v) = std_i [ c-ICE_j^(i)(v) ]

Plotting SD_j(v) against v gives a heterogeneity curve. It answers:
"At feature value xⱼ = v, how much disagreement is there among instances
about the effect of having xⱼ = v vs xⱼ = v₁?"

    SD_j(v) ≈ 0 for all v:    No interaction. The PDP is trustworthy.
    SD_j(v) growing with v:   Fan-shaped interaction (Pattern 3).
    SD_j(v) constant and high: Parallel but spread — other features
                                determine the baseline, not xⱼ.
    SD_j(v) non-monotone:      Complex interaction structure.


### The ICE Interaction Index

Greenwell (2018) proposed a scalar interaction index derived from ICE:

    H_j = (1/(G-1)) Σ_{k=2}^{G}  Var_i [ c-ICE_j^(i)(vₖ) ]

This is the average pointwise variance of the c-ICE curves across all
grid points (excluding the anchor point v₁ where all curves start at 0
by construction). A larger H_j indicates stronger interaction between
feature j and some other feature.

H_j can be computed for every feature j and used to RANK features by
interaction strength — a global interaction screening tool that requires
only ICE, not SHAP interaction values.

    # ================================================================== #
    **Computing H_j — the steps:**

    1. Compute c-ICE curves for feature j: n curves, G points each.
    2. At each grid point vₖ (k=2,...,G), compute Var_i[c-ICE^(i)(vₖ)].
    3. Average the G-1 variance values.

    H_j = 0  →  no interaction involving feature j
    H_j > 0  →  feature j interacts with at least one other feature
    Larger H_j → stronger interaction (but not comparable across different
                 feature scales — standardise if comparing across features)
    # ================================================================== #


### What Drives the c-ICE Spread — Identifying the Interacting Feature

Once c-ICE reveals that feature j interacts with SOMETHING, the next
question is: which other feature is responsible? Several strategies:

    STRATEGY 1 — COLOUR-CODE BY A SUSPECTED FEATURE:
    ─────────────────────────────────────────────────
    Colour each ICE line by the value of another feature xₖ (e.g.,
    a blue-to-red gradient from low to high xₖ). If the colouring
    organises the spread — blue lines at the top, red at the bottom,
    or vice versa — then xⱼ and xₖ interact.

    This is the most common practical approach in tools like plotly
    or matplotlib. You hypothesise the interacting feature and colour
    to confirm or deny.

    STRATEGY 2 — RANK BY c-ICE SLOPE, COMPARE TO OTHER FEATURES:
    ──────────────────────────────────────────────────────────────
    Fit a slope to each c-ICE curve. Rank instances by their c-ICE
    slope. Check which features are highest for high-slope instances
    and lowest for low-slope instances. The features that most cleanly
    separate the two groups are the likely interaction partners.

    STRATEGY 3 — SECOND-ORDER ALE / SHAP INTERACTION VALUES:
    ─────────────────────────────────────────────────────────
    ICE detects that an interaction exists and can hint at which feature
    is involved, but it does not QUANTIFY the interaction precisely.
    Once ICE suggests an interaction between xⱼ and xₖ, confirm and
    measure it using second-order ALE or SHAP interaction values.

    STRATEGY 4 — RECURSIVE PARTITIONING ON RESIDUALS:
    ──────────────────────────────────────────────────
    Fit a univariate model to predict the c-ICE slope from all other
    features. A high R² means the slope variation is explainable by
    the other features. The variable importances in that model identify
    the interaction partners.


##### PART V: ICE IN THE THREE-METHOD FAMILY — ICE vs PDP vs ALE

### What Each Method Answers

All three marginal effect methods evaluate the model at hypothetical
feature values. They differ in HOW they handle the remaining features
and HOW they aggregate.

    ICE_j^(i)(v) = f(v, xᵢ_\j)
      Fix all other features for instance i. Vary xⱼ.
      Individual-level. No aggregation. Ceteris paribus for one instance.

    PDP_j(v) = (1/n) Σᵢ f(v, xᵢ_\j)
      Average ICE over all instances using the MARGINAL distribution.
      Population-level. Aggregation by unconditional mean.
      Biased under feature correlation (see ALE module).

    ALE_j(x) = ∫ E_{X_\j | Xⱼ=v} [∂f/∂xⱼ (v, X_\j)] dv
      Conditional local derivative, integrated.
      Population-level. Aggregation by conditional mean of derivatives.
      Unbiased under feature correlation.

The key positioning of ICE:

    ICE is not a GLOBAL method — it makes no claim about the population
    average. Each curve is a statement about one instance only.

    ICE is not a CAUSAL method — it varies one feature while holding
    others fixed at their OBSERVED values, not at values they would take
    under an intervention (which would require a causal model).

    ICE is a DIAGNOSTIC method — its primary value is in revealing
    whether a global summary (PDP or ALE) is trustworthy for your purpose,
    and in signalling whether interactions exist.


### The Recommended Three-Step Workflow

    STEP 1 — PLOT ICE FOR EACH IMPORTANT FEATURE:
    ───────────────────────────────────────────────
    For each feature ranked as important by permutation importance or
    SHAP, plot the c-ICE bundle. Inspect visually for the five patterns.

    STEP 2 — DECIDE WHETHER PDP OR ALE SUMMARISES TRUTHFULLY:
    ───────────────────────────────────────────────────────────
    • Parallel c-ICE → PDP is a faithful summary. Report PDP (but use
      ALE if features are correlated).
    • Non-parallel c-ICE → PDP is misleading. Report ALE for the global
      effect AND keep the ICE plot to show heterogeneity.
    • Crossing c-ICE → NEVER report only PDP. Always show ICE.
      Investigate the interaction.

    STEP 3 — IDENTIFY INTERACTIONS AND HAND OFF TO SHAP / 2nd-ALE:
    ────────────────────────────────────────────────────────────────
    Use the colour-coding and ranking strategies to hypothesise which
    feature xₖ drives the ICE spread. Confirm with second-order ALE or
    SHAP interaction values for a precise, quantified interaction effect.


### The Computational Cost Comparison

    ICE:  n × G model evaluations (n = dataset size, G = grid points)
    PDP:  n × G model evaluations (identical to ICE — PDP is ICE averaged)
    ALE:  2 × n × K model evaluations (K = intervals, each instance
           evaluated at the left AND right boundary of its interval)

ICE and PDP require exactly the same number of model calls. The only
difference is that ICE keeps all n curves; PDP collapses them to one.
Computing ICE costs the same as PDP — you should always compute ICE
first, then average to get PDP, never the other way around.

ALE is typically cheaper than ICE by a factor of G/2 (e.g., G=50 grid
points vs K=25 intervals → ALE needs half the model calls of ICE/PDP).
But ALE and ICE answer different questions — cost alone does not determine
which to use.


##### PART VI: PRACTICAL PITFALLS AND HONEST LIMITATIONS OF ICE

### Pitfall 1 — ICE Inherits PDP's Extrapolation Problem

ICE evaluates f(v, xᵢ_\j) where v ranges across a full grid of feature j
values. For any specific instance i, some values of v may be very far from
xᵢ_j. If feature j is correlated with features in x_\j, then the
combination (v, xᵢ_\j) for extreme v may be unrealistic.

    Example:
    Instance i has xᵢ_credit = 800 (high). When the ICE curve for income
    evaluates at very low income values, it creates the combination
    (income=20k, credit=800) — which barely exists in the real world.
    The model's behaviour at that point was never constrained by data.

    The ICE curve for that instance in the low-income region is effectively
    model extrapolation. The curve may do anything there.

    ALE avoids this by only moving a feature WITHIN its local neighbourhood.
    ICE does not — it moves each feature across its full range, creating
    potentially unrealistic combinations.

Practical guidance: when c-ICE curves behave erratically at the EXTREMES
of the feature range (the leftmost and rightmost grid points), be cautious.
Check how many instances have xᵢ_j near those extreme values. Few instances
→ the model is extrapolating → the curves there are not trustworthy.


### Pitfall 2 — Visual Overplotting

With n = 1000 instances and G = 50 grid points, an ICE plot contains 1000
lines on one figure. This makes individual lines invisible and the plot a
meaningless grey smear.

Common solutions:
  • Subsample: display only 50–200 randomly chosen ICE curves and overlay
    the full-data PDP in bold.
  • Use transparency (alpha blending): set each line to alpha = 0.05–0.15
    so individual lines are faint but the collective density is visible.
  • Cluster first: group ICE curves by shape using k-means or hierarchical
    clustering, then display one representative per cluster.
  • Jitter: add tiny random vertical noise if many curves are identical
    (common in tree-based models whose ICE curves take discrete values).

The goal is to show the DISTRIBUTION of curves while keeping individual
lines visually distinguishable. There is no universal answer — the right
approach depends on n, the diversity of curve shapes, and the output medium.


### Pitfall 3 — Mistaking Vertical Spread for Slope Variation

In RAW ICE (not centred), curves at different heights look like they might
have different slopes even when they are perfectly parallel. The eye tracks
the separation between curves, which is constant for parallel lines, but
the visual impression is of divergence.

    # ================================================================== #
    **Raw ICE mirage — parallel lines look like they spread:**

    RAW ICE:                    c-ICE (corrected):
    f ↑                         Δf ↑
    5 │        /                   │ /  /  /  ← lines overlap
    4 │       / /                  │/  /  /     (actually parallel)
    3 │      / / /               0 └────────→ xⱼ
    2 │     / / / /               (centring reveals the truth)
    1 │    / / / / /
      └──────────────→ xⱼ

    The vertical gap between raw ICE lines looks like it is growing
    from left to right — which would indicate an interaction. But
    after centring, all lines coincide. The apparent divergence was
    purely a baseline offset artefact.
    # ================================================================== #

This is why c-ICE should always be the primary display when assessing
interaction. Raw ICE is secondary, for understanding absolute prediction
levels.


### Pitfall 4 — ICE is Local, Not Causal

ICE computes f(v, xᵢ_\j): what would the model predict if instance i had
feature j equal to v, with everything else unchanged? This is a model
query, not a real intervention.

Two distinct interpretive errors:

    ERROR A — Treating ICE as interventional:
    "The ICE curve shows that if we gave this applicant a $20k raise,
     their approval probability would increase by 15%."
    This claims that changing income in reality produces a 15% increase.
    But ICE only says the MODEL would predict +15% for that feature value.
    If income is correlated with unmeasured features (e.g., job stability),
    the real causal effect of a $20k raise may be very different.

    ERROR B — Treating ICE as a counterfactual for the individual:
    "This specific patient would survive if their blood pressure were lower."
    ICE evaluates f(v=lower_BP, other_features_at_observed). But lowering
    blood pressure in reality also changes other features (medication effects,
    lifestyle), so the true ceteris paribus counterfactual does not exist.

ICE is a description of how the model responds to feature perturbations.
It is a property of f, not of the world. Use it to understand the model;
do not use it to reason about what-would-happen causally.


##### PART VII: A COMPLETE WORKED EXAMPLE BY HAND

### 4 Instances, 2 Features, 5 Grid Points — Full Trace

We compute raw ICE, c-ICE, the PDP, and the c-PDP by hand to make every
step visible.

    Dataset (n=4, 2 features):
    ─────────────────────────────────────────────────
    i    x₁ (income, $k)    x₂ (credit_score/100)
    1         30                    5.0   (low income, low credit)
    2         50                    6.5   (mid income, mid credit)
    3         70                    7.5   (high income, high credit)
    4         90                    8.5   (very high income, top credit)
    ─────────────────────────────────────────────────

    Model with interaction:
    f(x₁, x₂) = 0.01·x₁ + 0.05·x₂ + 0.002·x₁·x₂

    ICE for feature x₁ (income), varying over a 5-point grid:
    Grid: v₁=30, v₂=45, v₃=60, v₄=75, v₅=90  ($k)


    STEP 1 — Compute raw ICE (5 model calls per instance, 20 total):

    Instance 1 (x₂ = 5.0 held fixed):
      f(30, 5.0) = 0.01(30) + 0.05(5) + 0.002(30)(5) = 0.30+0.25+0.30 = 0.85
      f(45, 5.0) = 0.45+0.25+0.45 = 1.15
      f(60, 5.0) = 0.60+0.25+0.60 = 1.45
      f(75, 5.0) = 0.75+0.25+0.75 = 1.75
      f(90, 5.0) = 0.90+0.25+0.90 = 2.05
      ICE¹ = [0.85, 1.15, 1.45, 1.75, 2.05]

    Instance 2 (x₂ = 6.5 held fixed):
      f(30, 6.5) = 0.30+0.325+0.39 = 1.015
      f(45, 6.5) = 0.45+0.325+0.585 = 1.360
      f(60, 6.5) = 0.60+0.325+0.780 = 1.705
      f(75, 6.5) = 0.75+0.325+0.975 = 2.050
      f(90, 6.5) = 0.90+0.325+1.170 = 2.395
      ICE² = [1.015, 1.360, 1.705, 2.050, 2.395]

    Instance 3 (x₂ = 7.5 held fixed):
      f(30, 7.5) = 0.30+0.375+0.45 = 1.125
      f(45, 7.5) = 0.45+0.375+0.675 = 1.500
      f(60, 7.5) = 0.60+0.375+0.900 = 1.875
      f(75, 7.5) = 0.75+0.375+1.125 = 2.250
      f(90, 7.5) = 0.90+0.375+1.350 = 2.625
      ICE³ = [1.125, 1.500, 1.875, 2.250, 2.625]

    Instance 4 (x₂ = 8.5 held fixed):
      f(30, 8.5) = 0.30+0.425+0.51 = 1.235
      f(45, 8.5) = 0.45+0.425+0.765 = 1.640
      f(60, 8.5) = 0.60+0.425+1.020 = 2.045
      f(75, 8.5) = 0.75+0.425+1.275 = 2.450
      f(90, 8.5) = 0.90+0.425+1.530 = 2.855
      ICE⁴ = [1.235, 1.640, 2.045, 2.450, 2.855]


    STEP 2 — Compute PDP (columnwise mean of ICE):

    PDP(30) = (0.850+1.015+1.125+1.235)/4 = 4.225/4 = 1.056
    PDP(45) = (1.150+1.360+1.500+1.640)/4 = 5.650/4 = 1.413
    PDP(60) = (1.450+1.705+1.875+2.045)/4 = 7.075/4 = 1.769
    PDP(75) = (1.750+2.050+2.250+2.450)/4 = 8.500/4 = 2.125
    PDP(90) = (2.050+2.395+2.625+2.855)/4 = 9.925/4 = 2.481
    PDP = [1.056, 1.413, 1.769, 2.125, 2.481]


    STEP 3 — Compute c-ICE (subtract first grid point's value per instance):

    c-ICE¹ = ICE¹ − 0.85 = [0.000, 0.300, 0.600, 0.900, 1.200]
    c-ICE² = ICE² − 1.015 = [0.000, 0.345, 0.690, 1.035, 1.380]
    c-ICE³ = ICE³ − 1.125 = [0.000, 0.375, 0.750, 1.125, 1.500]
    c-ICE⁴ = ICE⁴ − 1.235 = [0.000, 0.405, 0.810, 1.215, 1.620]

    c-PDP = PDP − PDP(v₁) = [0.000, 0.357, 0.713, 1.069, 1.425]


    DIAGNOSIS — are c-ICE curves parallel?

    The c-ICE curves at v₅=90 (the widest point):
      c-ICE¹(90) = 1.200   (instance 1, x₂=5.0)
      c-ICE²(90) = 1.380   (instance 2, x₂=6.5)
      c-ICE³(90) = 1.500   (instance 3, x₂=7.5)
      c-ICE⁴(90) = 1.620   (instance 4, x₂=8.5)

    The curves are NOT equal. Higher credit_score → steeper c-ICE slope.
    Variance of c-ICE values at v₅: Var([1.20, 1.38, 1.50, 1.62])

      Mean = (1.20+1.38+1.50+1.62)/4 = 1.425
      Var  = [(1.20-1.425)²+(1.38-1.425)²+(1.50-1.425)²+(1.62-1.425)²]/4
           = [0.0506+0.0020+0.0056+0.0380]/4 = 0.0241
      SD   = √0.0241 = 0.155

    This variance is non-zero → there IS an interaction between x₁ and x₂.

    WHY THIS HAPPENS:
    The interaction term 0.002·x₁·x₂ means that the derivative
    ∂f/∂x₁ = 0.01 + 0.002·x₂. For instance 1 (x₂=5):  derivative = 0.02
    For instance 4 (x₂=8.5): derivative = 0.027. The slope of the ICE
    curve scales with x₂. Instances with higher credit_score gain more from
    additional income — a synergistic (super-additive) interaction.

    The PDP reports an average slope of (1.425)/(90-30) = 0.024 per $k.
    This is accurate for no individual instance:
      Instance 1 slope: 1.200/60 = 0.020/k
      Instance 4 slope: 1.620/60 = 0.027/k
    The PDP blends these into 0.024 — neither too high nor too low, but
    accurate for no one.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔═══════════════════════╦══════════════════╦══════════════════╦══════════════════╗
    ║ Property              ║ ICE              ║ c-ICE            ║ PDP (for context)║
    ╠═══════════════════════╬══════════════════╬══════════════════╬══════════════════╣
    ║ Scope                 ║ Local (per inst) ║ Local (per inst) ║ Global (average) ║
    ║ Aggregation           ║ None             ║ None             ║ Mean over n      ║
    ║ Interaction detection ║ Yes (visually)   ║ Better (visual)  ║ No               ║
    ║ Baseline offsets      ║ Present          ║ Removed          ║ Averaged away    ║
    ║ Model queries         ║ n × G            ║ n × G            ║ n × G (same)     ║
    ║ Extrapolation risk    ║ Yes              ║ Yes              ║ Yes (same risk)  ║
    ║ Corr. feature bias    ║ Yes (inherited)  ║ Yes (inherited)  ║ Yes (PDP bias)   ║
    ║ Centred at 0          ║ No               ║ Yes (at v₁)      ║ Optional         ║
    ║ Reveals heterogeneity ║ Yes              ║ More clearly     ║ No               ║
    ║ Identifies interaction║ Partially        ║ Yes (spread)     ║ No               ║
    ║ Causal interpretation ║ No               ║ No               ║ No               ║
    ╚═══════════════════════╩══════════════════╩══════════════════╩══════════════════╝

    Relationship: PDP_j(v) = mean_i[ICE_j^(i)(v)]
                  c-ICE_j^(i)(v) = ICE_j^(i)(v) − ICE_j^(i)(v₁)
                  c-PDP(v) = mean_i[c-ICE_j^(i)(v)] = PDP(v) − PDP(v₁)

    Interaction index (Greenwell 2018):
      H_j = mean_{k=2..G} Var_i[c-ICE_j^(i)(vₖ)]
      H_j = 0 ↔ no interaction between xⱼ and any other feature
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "ICE from Scratch — Five Patterns, Interaction Detection, Greenwell H": {
        "description": (
            "Generates five synthetic models — one for each ICE diagnostic pattern "
            "(parallel, spread, fan, crossing, non-linear threshold) — and computes "
            "raw ICE, c-ICE, PDP, and the Greenwell interaction index H_j from "
            "scratch for each. Prints the numeric curves and diagnoses whether PDP "
            "is trustworthy for each case. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "ice",
        "code": '''
"""
================================================================================
ICE FROM SCRATCH — FIVE PATTERNS AND THE GREENWELL INTERACTION INDEX
================================================================================

We build five models, each engineered to produce a specific ICE pattern:

  Pattern 1 — PARALLEL      : no interaction, PDP is trustworthy
  Pattern 2 — SPREAD        : baseline varies, slope is uniform
  Pattern 3 — FAN           : interaction amplifies with xⱼ
  Pattern 4 — CROSSING      : interaction reverses sign of effect
  Pattern 5 — NON-LINEAR    : threshold effect at xⱼ = 0.5

For each pattern we compute:
  • Raw ICE curves (n × G matrix)
  • c-ICE curves (subtract each curve's left endpoint)
  • PDP and c-PDP (mean of ICE and c-ICE)
  • Greenwell H_j (interaction index from c-ICE variance)
  • Diagnosis: is PDP a trustworthy summary?

All from scratch, no external libraries.
================================================================================
"""

import random
import math

random.seed(42)

N  = 120   # instances
G  = 15    # grid points
v_grid = [k / (G - 1) for k in range(G)]   # grid in [0, 1]

# Generate base dataset: x1 is the feature we explain, x2 is the "other" feature
# x2 is a feature that MODULATES x1's effect (the interaction partner)
x2_list = [random.uniform(-1, 1) for _ in range(N)]
x1_obs  = [random.uniform(0, 1)  for _ in range(N)]   # observed x1 (not used in ICE)


# ─────────────────────────────────────────────────────────────────────────────
# CORE ICE MACHINERY
# ─────────────────────────────────────────────────────────────────────────────

def compute_ice(model_fn, x2_list, v_grid):
    """
    Compute raw ICE curves and PDP for feature x1 over v_grid.

    model_fn(x1, x2) → prediction
    Holds x2 fixed per instance; varies x1 across v_grid.

    Returns:
      ice   : n × G matrix of raw ICE values
      pdp   : G-vector of PDP values (mean of ICE columns)
      c_ice : n × G matrix of centred ICE (subtract column 0)
      c_pdp : G-vector of c-PDP
    """
    n = len(x2_list)
    G = len(v_grid)

    ice = []
    for i in range(n):
        row = [model_fn(v, x2_list[i]) for v in v_grid]
        ice.append(row)

    pdp = [sum(ice[i][k] for i in range(n)) / n for k in range(G)]

    c_ice = []
    for i in range(n):
        anchor = ice[i][0]
        c_ice.append([ice[i][k] - anchor for k in range(G)])

    c_pdp = [sum(c_ice[i][k] for i in range(n)) / n for k in range(G)]

    return ice, pdp, c_ice, c_pdp


def greenwell_H(c_ice):
    """
    Compute Greenwell (2018) interaction index H_j.
    H_j = mean over grid points k=1..G-1 of Var_i[c_ice^(i)(vk)].
    k=0 is excluded because c_ice is 0 there by construction.
    """
    n = len(c_ice)
    G = len(c_ice[0])
    variances = []
    for k in range(1, G):
        vals = [c_ice[i][k] for i in range(n)]
        mean_k = sum(vals) / n
        var_k  = sum((v - mean_k)**2 for v in vals) / n
        variances.append(var_k)
    return sum(variances) / len(variances)


def slope_per_instance(c_ice, v_grid):
    """Fit a linear slope to each c-ICE curve via simple OLS."""
    G = len(v_grid)
    mean_v = sum(v_grid) / G
    denom  = sum((v_grid[k] - mean_v)**2 for k in range(G))
    slopes = []
    for row in c_ice:
        mean_r = sum(row) / G
        num    = sum((v_grid[k] - mean_v) * (row[k] - mean_r) for k in range(G))
        slopes.append(num / (denom + 1e-12))
    return slopes


def print_ice_summary(name, model_fn, x2_list, v_grid, show_rows=6):
    """Compute and print a full ICE/c-ICE/PDP summary for one pattern."""
    ice, pdp, c_ice, c_pdp = compute_ice(model_fn, x2_list, v_grid)
    H = greenwell_H(c_ice)
    slopes = slope_per_instance(c_ice, v_grid)
    n = len(x2_list)

    slope_min = min(slopes)
    slope_max = max(slopes)
    slope_std = math.sqrt(sum((s - sum(slopes)/n)**2 for s in slopes) / n)

    print()
    print("=" * 68)
    print(f"  PATTERN: {name}")
    print("=" * 68)
    print(f"  Greenwell H_j  = {H:.6f}   "
          f"({'> 0 → INTERACTION detected' if H > 0.001 else '≈ 0 → no interaction'})")
    print(f"  c-ICE slope range: [{slope_min:.4f}, {slope_max:.4f}]  "
          f"std = {slope_std:.4f}")
    print(f"  PDP trustworthy: "
          f"{'YES — lines are parallel' if slope_std < 0.05 else 'NO  — heterogeneous slopes'}")
    print()

    # Print a selection of c-ICE curves + PDP row
    show_idx = list(range(0, n, n // show_rows))[:show_rows]
    header = f"  {'x2':>7}  " + "  ".join(f"v={v:.2f}" for v in v_grid[::3])
    print(f"  c-ICE curves (selected instances):")
    print(f"  {'x2':>7}  " +
          "  ".join(f"{'v='+str(round(v,2)):>7}" for v in v_grid[::3]))
    print(f"  {'-'*60}")

    for i in show_idx:
        row_vals = "  ".join(f"{c_ice[i][k]:>7.3f}" for k in range(0, G, 3))
        print(f"  {x2_list[i]:>7.3f}  {row_vals}")

    pdp_row = "  ".join(f"{c_pdp[k]:>7.3f}" for k in range(0, G, 3))
    print(f"  {'-'*60}")
    print(f"  {'c-PDP':>7}  {pdp_row}  ← mean of all c-ICE curves")

    # Show pointwise SD to quantify spread
    sd_row = []
    for k in range(0, G, 3):
        vals = [c_ice[i][k] for i in range(n)]
        m = sum(vals)/n
        sd = math.sqrt(sum((v-m)**2 for v in vals)/n)
        sd_row.append(f"{sd:>7.3f}")
    print(f"  {'SD(k)':>7}  {'  '.join(sd_row)}  ← SD across instances at each v")


# ─────────────────────────────────────────────────────────────────────────────
# FIVE MODELS — ONE PER PATTERN
# ─────────────────────────────────────────────────────────────────────────────

# PATTERN 1 — PARALLEL: f = 2·x1 + 0.5·x2  (no interaction term)
# Effect of x1 is always +2 per unit, regardless of x2.
def model_parallel(x1, x2):
    return 2.0 * x1 + 0.5 * x2

# PATTERN 2 — SPREAD: f = 1.5·x1 + 2.0·x2  (no interaction, but x2 large)
# x2 shifts the baseline strongly; x1's slope is still uniform.
def model_spread(x1, x2):
    return 1.5 * x1 + 2.0 * x2

# PATTERN 3 — FAN: f = x1 + x1·x2  (amplifying interaction)
# When x2 is high, x1 has a stronger positive effect.
# When x2 is low (negative), x1 barely matters.
def model_fan(x1, x2):
    return x1 + x1 * x2   # derivative wrt x1 = 1 + x2, grows with x2

# PATTERN 4 — CROSSING: f = x1·x2  (sign-reversing interaction)
# When x2 > 0: increasing x1 raises prediction.
# When x2 < 0: increasing x1 LOWERS prediction.
# Curves cross at x1 = 0 for all instances.
def model_crossing(x1, x2):
    return (x1 - 0.5) * x2   # centred so crossing is visible at x1=0.5

# PATTERN 5 — NON-LINEAR: threshold effect at x1 = 0.5
# The model jumps at x1=0.5; the jump size is modulated by x2.
def model_nonlinear(x1, x2):
    base = 1.0 if x1 >= 0.5 else 0.0      # hard threshold
    return base * (1.0 + 0.3 * x2) + 0.5 * x1  # x2 modulates jump size

# ─────────────────────────────────────────────────────────────────────────────
# PRINT ALL FIVE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

patterns = [
    ("1 — PARALLEL (no interaction)",         model_parallel),
    ("2 — SPREAD (baseline varies)",          model_spread),
    ("3 — FAN (amplifying interaction)",      model_fan),
    ("4 — CROSSING (sign-reversing interac)", model_crossing),
    ("5 — NON-LINEAR THRESHOLD",              model_nonlinear),
]

print("=" * 68)
print("  ICE DIAGNOSTIC PATTERNS — FIVE MODELS")
print("=" * 68)
print(f"  N={N} instances, G={G} grid points, x2 ∈ [-1, 1]")

for name, fn in patterns:
    print_ice_summary(name, fn, x2_list, v_grid, show_rows=5)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION PARTNER IDENTIFICATION — Fan pattern in depth
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  IDENTIFYING THE INTERACTION PARTNER — FAN pattern deep dive")
print("=" * 68)
print()
print("  Once c-ICE reveals an interaction (H_j > 0), the next question")
print("  is: WHICH other feature drives the slope variation?")
print()
print("  Strategy: rank instances by their c-ICE slope. Compare the")
print("  feature values of the steepest-slope group vs flattest group.")
print()

_, _, c_ice_fan, c_pdp_fan = compute_ice(model_fan, x2_list, v_grid)
slopes_fan = slope_per_instance(c_ice_fan, v_grid)

# Sort instances by c-ICE slope
sorted_by_slope = sorted(range(N), key=lambda i: slopes_fan[i])
bottom_q = sorted_by_slope[:N//4]    # lowest 25% slope
top_q    = sorted_by_slope[3*N//4:]  # highest 25% slope

mean_x2_bottom = sum(x2_list[i] for i in bottom_q) / len(bottom_q)
mean_x2_top    = sum(x2_list[i] for i in top_q)    / len(top_q)

print(f"  Instances with FLATTEST c-ICE slopes (bottom 25%):")
print(f"    Mean c-ICE slope = {sum(slopes_fan[i] for i in bottom_q)/len(bottom_q):.4f}")
print(f"    Mean x2          = {mean_x2_bottom:.4f}")
print()
print(f"  Instances with STEEPEST c-ICE slopes (top 25%):")
print(f"    Mean c-ICE slope = {sum(slopes_fan[i] for i in top_q)/len(top_q):.4f}")
print(f"    Mean x2          = {mean_x2_top:.4f}")
print()
print(f"  x2 difference between groups: {mean_x2_top - mean_x2_bottom:+.4f}")
print()
print(f"  FINDING: steep-slope instances have much HIGHER x2 values.")
print(f"  This identifies x2 as the interaction partner for x1.")
print()
print(f"  Theoretical confirmation:")
print(f"    model_fan = x1 + x1·x2  →  ∂f/∂x1 = 1 + x2")
print(f"    The slope of each ICE curve equals 1 + x2 for that instance.")
print(f"    Top-quartile x2 mean = {mean_x2_top:.3f}  → expected slope ≈ {1+mean_x2_top:.3f}")
print(f"    Bot-quartile x2 mean = {mean_x2_bottom:.3f}  → expected slope ≈ {1+mean_x2_bottom:.3f}")
slope_top_actual = sum(slopes_fan[i] for i in top_q) / len(top_q)
slope_bot_actual = sum(slopes_fan[i] for i in bottom_q) / len(bottom_q)
print(f"    Top-quartile actual c-ICE slope = {slope_top_actual:.3f}  ✓")
print(f"    Bot-quartile actual c-ICE slope = {slope_bot_actual:.3f}  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# GREENWELL H vs TRUE INTERACTION STRENGTH — RANKING FEATURES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  GREENWELL H_j AS A FEATURE INTERACTION RANKER")
print("=" * 68)
print()
print("  In practice you compute H_j for every feature and rank them.")
print("  Features with H_j ≈ 0 have no interactions — PDP is safe.")
print("  Features with large H_j need ICE and ALE investigation.")
print()

# Build a richer model: 4 features, varying interaction strengths
# x1: strong interaction with x3 (coefficient 0.5)
# x2: moderate interaction with x3 (coefficient 0.2)
# x3: the modulator feature
# x4: pure additive, no interaction

def rich_model(x1, x2, x3, x4):
    return (0.3*x1 + 0.2*x2 + 0.1*x3 + 0.4*x4
            + 0.5*x1*x3       # strong x1-x3 interaction
            + 0.2*x2*x3)      # moderate x2-x3 interaction

feat_labels = ["x1 (strong interact)", "x2 (mod interact)", "x3 (modulator)", "x4 (additive)"]

x3_list = [random.gauss(0, 1) for _ in range(N)]
x4_list = [random.gauss(0, 1) for _ in range(N)]
x2b_list = [random.gauss(0, 0.5) for _ in range(N)]  # narrower range

datasets = [
    (x2_list,  lambda v, x2: rich_model(v,          x2,          x3_list[i], x4_list[i]) if False else None),  # placeholder
    (x2b_list, None),
    (x3_list,  None),
    (x4_list,  None),
]

# Build proper closures for each feature
def make_model_for_feature(feat_idx, other_data):
    """Return a function model_fn(v, other_feature_i) for ICE."""
    x2s, x3s, x4s = other_data
    if feat_idx == 0:
        return lambda v, i_x2: rich_model(v,   x2s[int(round(i_x2*(N-1)))], x3s[int(round(i_x2*(N-1)))], x4s[int(round(i_x2*(N-1)))])
    return None  # placeholder for simplicity

# Simplified: compute H_j by varying each feature with realistic co-features
H_values = {}
for feat_idx, fname in enumerate(feat_labels):
    # For each feature, the ICE curves vary that feature, hold others at
    # their real observed values (using index i for all other features)
    c_ice_all = []
    for i in range(N):
        # Build the ICE curve for instance i varying feature feat_idx
        row = []
        for v in v_grid:
            # Map grid [0,1] to realistic range for this feature
            if feat_idx == 0:
                v_real = v * 3 - 1.5  # x1 range ~ [-1.5, 1.5]
                pred = rich_model(v_real, x2b_list[i], x3_list[i], x4_list[i])
            elif feat_idx == 1:
                v_real = v * 2 - 1    # x2 range ~ [-1, 1]
                pred = rich_model(x2_list[i], v_real, x3_list[i], x4_list[i])
            elif feat_idx == 2:
                v_real = v * 4 - 2    # x3 range ~ [-2, 2]
                pred = rich_model(x2_list[i], x2b_list[i], v_real, x4_list[i])
            else:
                v_real = v * 4 - 2    # x4 range ~ [-2, 2]
                pred = rich_model(x2_list[i], x2b_list[i], x3_list[i], v_real)
            row.append(pred)
        # Centre the row
        anchor = row[0]
        c_ice_all.append([val - anchor for val in row])

    H = greenwell_H(c_ice_all)
    H_values[fname] = H

print(f"  {'Feature':<25}  {'H_j':>8}  {'Rank'}")
print(f"  {'-'*45}")
ranked = sorted(H_values.items(), key=lambda x: -x[1])
for rank, (fname, H) in enumerate(ranked, 1):
    bar = "█" * int(H * 30 / (max(H_values.values()) + 1e-9))
    print(f"  {fname:<25}  {H:>8.5f}  {rank}   {bar}")

print()
print(f"  x1 and x2 have non-zero H_j → they interact with x3")
print(f"  x4 has H_j ≈ 0 → purely additive, no interaction → PDP safe")
print(f"  x3 has H_j > 0 → x3 interacts with x1 and x2 (it IS the modulator)")
print()
print(f"  ACTION: investigate features with large H_j using c-ICE,")
print(f"  colour-coded by suspected interaction partners, then confirm")
print(f"  with second-order ALE or SHAP interaction values.")
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
    #     from interpretability.visuals.ice import (
    #         ICE_VISUAL_HTML,
    #         ICE_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ICE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ICE_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[ice.py] Could not load visual: {e}", stacklevel=2)

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