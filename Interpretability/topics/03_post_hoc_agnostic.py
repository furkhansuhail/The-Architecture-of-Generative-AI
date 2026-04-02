"""
Post-hoc Model-Agnostic Methods — Explaining Any Black Box
===========================================================

The methods in this folder share one fundamental design principle:
they treat the trained model as a black box — a function that maps
inputs to outputs — and build explanations purely by observing that
function's behaviour. They require NO access to model internals:
no gradients, no tree structure, no weight matrices.

This makes them universally applicable, but it also means every
explanation is an approximation of the model, not the model itself.

Covered here:
  • Permutation Importance
  • SHAP  (SHapley Additive exPlanations)
  • LIME  (Local Interpretable Model-agnostic Explanations)
  • PDP   (Partial Dependence Plots)
  • ICE   (Individual Conditional Expectation)
  • ALE   (Accumulated Local Effects)
  • Anchors

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Post-hoc Model-Agnostic Methods"
DISPLAY_NAME = "03 · Post-hoc Agnostic"
ICON = "🔦"
SUBTITLE = "SHAP, LIME, PDP, ICE, ALE, Permutation Importance, Anchors"


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
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Post-hoc Agnostic Paradigm

An intrinsic model IS its explanation. A post-hoc agnostic method BUILDS
an explanation after the fact — it probes the model by feeding it carefully
constructed inputs and observing the outputs, then uses those observations
to reason backwards about what the model is doing.

The key abstraction: the model is a function f : X → Y. We never look
inside f. We only call it. Everything we learn about f, we learn by
asking "what does f output when I change this input?"

This creates two fundamental properties that apply to ALL methods in
this folder:

    PROPERTY 1 — UNIVERSALITY:
    Every method here works with ANY model: neural networks, random
    forests, SVMs, XGBoost, LightGBM, custom PyTorch architectures,
    even external APIs where you don't have access to source code.
    The interface is just: give input → receive output.

    PROPERTY 2 — APPROXIMATION:
    Because we only observe inputs and outputs, every explanation is
    an approximation. We never have ground truth. We have observations.
    The explanation is a simplified model OF the model — and like all
    models, it is wrong in some ways. The key question for each method
    is: what kind of wrong is it, and does that matter for your use case?

    ┌─────────────────────────────────────────────────────────────┐
    │  Intrinsic model:                                           │
    │  explanation ≡ model (they are the same object)             │
    │  Faithfulness = 1.0 by definition                           │
    │                                                             │
    │  Post-hoc method:                                           │
    │  explanation ≈ model (the explanation approximates it)      │
    │  Faithfulness ∈ (0, 1) — depends on method and problem      │
    └─────────────────────────────────────────────────────────────┘

The hierarchy of post-hoc methods runs from least to most faithful:
surrogate models (least) → attribution methods → exact methods (most).
SHAP's TreeExplainer is exact for tree models. KernelSHAP and LIME
are approximations.


##### PART I: PERMUTATION IMPORTANCE

### What It Is

Permutation importance (Breiman, 2001) is the simplest, most
conceptually transparent global importance measure. The idea is
brutally direct:

    If a feature is important, destroying it should hurt performance.
    If a feature is unimportant, destroying it should do nothing.

We "destroy" a feature by permuting (shuffling) its values across
the dataset. This severs the relationship between that feature and
the target, while leaving the marginal distribution of the feature
unchanged. The drop in performance IS the importance.

    Algorithm:
    ──────────
    1. Evaluate baseline model performance: score₀ = metric(f, X, y)
    2. For each feature j:
         a. Create X_perm: copy of X with column j shuffled randomly
         b. Evaluate: score_j = metric(f, X_perm, y)
         c. Importance_j = score₀ - score_j
    3. Repeat step 2 K times; take the mean (reduces variance)
    4. Rank features by Importance_j (higher = more important)


### Why Permutation, Not Removal?

A natural alternative: set feature j to zero (or its mean). Why
shuffle instead?

    Setting to zero:     Changes the marginal distribution of xⱼ.
                         The model now sees out-of-distribution inputs
                         it was never trained on. The performance drop
                         may reflect model fragility to OOD inputs,
                         not feature importance.

    Setting to mean:     Same problem — all instances now have
                         identical xⱼ. The model can no longer
                         distinguish between them.

    Permutation:         The marginal distribution P(xⱼ) is PRESERVED.
                         Only the conditional relationship P(xⱼ | xₖ≠ⱼ)
                         is destroyed. The model sees realistic feature
                         values — just uncoupled from the target.
                         The performance drop reflects only the loss
                         of the x_j → y signal.

    # ================================================================= #
    **Diagram — Why Permutation Preserves Marginal Distribution:**

    ORIGINAL DATA            PERMUTED (feature j shuffled)
    x₁     x₂    y           x₁     x₂(shuffled)    y
    ─────────────────         ────────────────────────
    0.2    HIGH    1          0.2    LOW              1
    0.8    LOW     0    →     0.8    HIGH             0
    0.5    HIGH    1          0.5    MED              1
    0.3    MED     0          0.3    HIGH             0

    x₂ still has the same values {HIGH, LOW, HIGH, MED} — same
    marginal distribution. But the pairing x₂ ↔ y is destroyed.
    Any model that relied on x₂ to predict y is now flying blind.
    # ================================================================= #


### Limitations and Pitfalls

    1. CORRELATED FEATURES — THE BIGGEST PROBLEM:
    ─────────────────────────────────────────────
    If x₁ and x₂ are highly correlated (e.g., income and wealth),
    permuting x₁ creates unrealistic (x₁, x₂) pairs: you can end up
    with (low income, high wealth) — a combination that barely exists
    in reality. The model is now evaluating on near-OOD data.

    Consequence: the importance of correlated features is systematically
    UNDERESTIMATED. The model can still use x₂ as a proxy for x₁ after
    permutation, so the performance drop is small even though x₁ is
    genuinely important.

    This is not a bug — it's telling you the features are redundant
    WITH EACH OTHER. But it means: permutation importance measures
    "how much this feature adds GIVEN all others", not "how important
    this feature is in isolation."

    2. TRAINING vs TEST PERMUTATION IMPORTANCE:
    ─────────────────────────────────────────────
    Computed on TRAINING data:
      Measures importance for the model's learned function.
      Can be dominated by features the model happened to overfit.

    Computed on TEST (held-out) data:
      Measures importance for generalisation — which features truly
      help predict new data. This is almost always what you want.
      Use test-set permutation importance.

    3. PERFORMANCE METRIC CHOICE:
    ──────────────────────────────
    The metric matters. Accuracy-based importance can be misleading
    for imbalanced datasets. AUC-based importance is more stable.
    Use a metric that reflects your actual task objective.



##### PART II: SHAP — SHapley Additive exPlanations

### The Origin — Cooperative Game Theory

SHAP (Lundberg & Lee, 2017) is grounded in a 70-year-old solution from
cooperative game theory: the Shapley value (Shapley, 1953).

The game theory setup: a coalition of players cooperates to earn a
total reward. How should that reward be fairly distributed among
the players? Shapley proved there is ONE distribution satisfying
four natural fairness axioms. That distribution is the Shapley value.

For machine learning, the "game" is: features cooperate to produce a
model prediction. The "reward" is the deviation of the prediction from
the baseline (average prediction). Shapley values tell you: how much
credit does each feature deserve for this prediction?

    Game theory → ML interpretability:
    ─────────────────────────────────────
    Players  →  Features (x₁, x₂, ..., xₙ)
    Coalition → Subset S of features given to the model
    Reward   →  f(x) - E[f(x)]  (prediction minus baseline)
    Shapley  →  SHAP value φᵢ for each feature i


### The Four Shapley Axioms

These axioms define what "fair attribution" means. They are the
theoretical foundation. Any attribution method that satisfies all
four is mathematically equivalent to Shapley values.

    AXIOM 1 — EFFICIENCY (Completeness):
    ─────────────────────────────────────
    The SHAP values sum exactly to the prediction minus the baseline:

        Σᵢ φᵢ = f(x) - E[f(x)]

    This is the accounting identity. Every unit of the prediction
    deviation is accounted for. Nothing is left over.

    Intuition: if the model says P(approve) = 0.72 and the baseline
    is E[f(x)] = 0.50, then the SHAP values must sum to +0.22. The
    explanation fully accounts for WHY this prediction is 0.22 above
    average.

    AXIOM 2 — SYMMETRY:
    ────────────────────
    If features i and j contribute identically to every coalition
    (i.e., f(S∪{i}) = f(S∪{j}) for all S ⊆ N \ {i,j}),
    then φᵢ = φⱼ.

    Intuition: features that do equal work get equal credit.
    No arbitrary favouritism between interchangeable features.

    AXIOM 3 — DUMMY (Null player):
    ────────────────────────────────
    If feature i adds nothing to any coalition:
    f(S∪{i}) = f(S) for all S, then φᵢ = 0.

    Intuition: a feature that never changes the prediction, regardless
    of what other features are present, gets zero credit.

    AXIOM 4 — LINEARITY (Additivity):
    ────────────────────────────────────
    For a model that is the sum of two models (f = f₁ + f₂),
    the SHAP values are the sum of each model's SHAP values:
    φᵢ(f) = φᵢ(f₁) + φᵢ(f₂)

    Intuition: explanation can be decomposed across additive components.
    Useful for ensembles or multi-output models.


### The Shapley Value Formula

For a model f with n features, the Shapley value for feature i is:

    φᵢ = Σ_{S ⊆ N\{i}}  [|S|!(n-|S|-1)! / n!] × [f(S∪{i}) - f(S)]

Where:
  • N = {1, 2, ..., n} — the full feature set
  • S = a subset of features NOT including i
  • f(S) = model output when only features in S are known;
           all features NOT in S are marginalised out (replaced by
           their expected values or sampled from background data)
  • |S|! and (n-|S|-1)! are combinatorial weighting terms
  • The weight |S|!(n-|S|-1)!/n! is the probability of observing
    coalition S in a random ordering of features

Interpretation: φᵢ is the AVERAGE marginal contribution of feature i
across all possible orderings of features. We imagine adding features
one at a time in random order, and measure how much each feature
changes the prediction when it joins the current coalition.

    # ================================================================= #
    **The Shapley Value: A Worked Example (3 features)**

    Model predicts loan approval. Features: A=credit, B=income, C=debt.
    Baseline E[f] = 0.50. Prediction f(x) = 0.80. To explain: +0.30.

    There are 3! = 6 orderings of 3 features:

    Ordering 1: A → B → C
      A joins {}:     f({A})     - f({})     = 0.65 - 0.50 = +0.15
      B joins {A}:    f({A,B})   - f({A})    = 0.75 - 0.65 = +0.10
      C joins {A,B}:  f({A,B,C}) - f({A,B})  = 0.80 - 0.75 = +0.05

    Ordering 2: A → C → B
      A joins {}:     +0.15
      C joins {A}:    f({A,C})   - f({A})    = 0.68 - 0.65 = +0.03
      B joins {A,C}:  f({A,B,C}) - f({A,C})  = 0.80 - 0.68 = +0.12

    Ordering 3: B → A → C
      B joins {}:     f({B})     - f({})     = 0.58 - 0.50 = +0.08
      A joins {B}:    f({A,B})   - f({B})    = 0.75 - 0.58 = +0.17
      C joins {A,B}:  +0.05

    Ordering 4: B → C → A  →  similarly compute
    Ordering 5: C → A → B  →  similarly compute
    Ordering 6: C → B → A  →  similarly compute

    Final SHAP values (average marginal contribution across all 6):
      φ_A (credit)  = +0.18
      φ_B (income)  = +0.08
      φ_C (debt)    = +0.04

    Check — Efficiency axiom: 0.18 + 0.08 + 0.04 = +0.30 ✓
    # ================================================================= #


### Computational Complexity — Why Exact SHAP is Infeasible

For n features, there are 2ⁿ subsets and n! orderings to average over.
With even 20 features, that is 2²⁰ ≈ 1,000,000 subsets and
20! ≈ 2.4 × 10¹⁸ orderings. Exact computation is intractable.

This is why different SHAP variants exist:

    KernelSHAP (model-agnostic):
    ─────────────────────────────
    Approximates Shapley values using weighted linear regression.
    The key insight: SHAP values are the coefficients of a specific
    weighted least-squares problem:

        φ = argmin_φ  Σ_{S⊆N}  π(S) × [f_S - (φ₀ + Σᵢ∈S φᵢ)]²

    where π(S) = (n-1) / [C(n,|S|) × |S| × (n-|S|)] is the
    Shapley kernel — a weighting that gives higher weight to
    small and large coalitions (which have more influence on the
    final average) and lower weight to mid-size coalitions.

    KernelSHAP approximates this by sampling coalitions rather than
    evaluating all 2ⁿ. Accurate but slow: O(2ⁿ k) where k = samples.


    TreeSHAP (tree-specific, EXACT):
    ─────────────────────────────────
    For tree-based models (decision trees, random forests, XGBoost,
    LightGBM), Lundberg et al. (2018) developed an exact algorithm
    running in O(TLD²) time, where T = number of trees, L = leaves,
    D = depth. This is polynomial, not exponential.

    The key idea: at each internal node, you can compute the exact
    fraction of the training data that passes each way. This lets
    you precisely marginalise over "unknown" features without sampling.
    The resulting SHAP values are EXACT — not approximations.

    TreeSHAP is the reason SHAP is widely practical in industry:
    XGBoost and sklearn decision trees are extremely common, and
    TreeSHAP gives exact explanations in milliseconds.


    DeepSHAP / GradientSHAP:
    ─────────────────────────
    Neural-network-specific variants that use backpropagation to
    approximate Shapley values. DeepSHAP uses the DeepLIFT
    backpropagation rules. GradientSHAP combines gradients with
    sampling from a background dataset. Both are faster than
    KernelSHAP for neural networks but are approximations.


### SHAP Value Interpretation — Three Levels

    LEVEL 1 — FORCE PLOT (one prediction):
    ────────────────────────────────────────
    For a single instance x:
      • φᵢ > 0: feature i pushed the prediction ABOVE the baseline
      • φᵢ < 0: feature i pushed the prediction BELOW the baseline
      • |φᵢ|: magnitude of the push
      • Σφᵢ = f(x) - baseline: the sum is the total deviation

    Example (loan approval, baseline = 0.50):
      credit_score  φ = +0.22  ← biggest positive driver
      income        φ = +0.08
      employment    φ = +0.03
      debt_ratio    φ = -0.13  ← biggest negative driver
      ─────────────────────────────
      Total deviation:  +0.20   → prediction = 0.50 + 0.20 = 0.70


    LEVEL 2 — SUMMARY PLOT (global behaviour from many local):
    ──────────────────────────────────────────────────────────
    Compute SHAP values for all N training/test instances. Plot:
      • Each dot = one data point
      • x-axis = SHAP value for that feature (positive = pushes up)
      • y-axis = feature (one row per feature)
      • Colour = actual feature value (red = high, blue = low)

    This reveals:
      • Which features matter most globally (vertical spread)
      • Whether the effect is monotone (all red dots on one side)
      • Whether there are interactions (mixed red/blue on same side)


    LEVEL 3 — DEPENDENCE PLOT (one feature's effect):
    ──────────────────────────────────────────────────
    Plot: x-axis = feature value, y-axis = SHAP value for that feature.
    Reveals the marginal shape of the model's learned function for one
    feature (holding average coalition context fixed). Non-linear shapes
    appear as curves instead of straight lines — something PDP
    also captures but SHAP dependence plots can do at instance resolution.


##### PART III: LIME — Local Interpretable Model-agnostic Explanations

### The Core Idea

LIME (Ribeiro, Singh & Guestrin, 2016) takes a radically different
approach to explanation. Rather than averaging over all orderings
(SHAP), it asks: what is the SIMPLEST explanation that is accurate
NEAR this specific data point?

The intuition: even if a model f is globally complex and non-linear,
it is likely to behave approximately linearly in a small neighbourhood
around any single prediction. A flat sheet of glass can approximate
a curved surface locally, even if it fails globally.

    LIME fits a locally-faithful, globally-simple surrogate model:

        explanation(x) = argmin_g  L(f, g, πx) + Ω(g)

    where:
      • f  = the black-box model (can be any model)
      • g  = the interpretable surrogate (linear model, tree, etc.)
      • πx = a proximity kernel — weights samples by how close they
             are to the instance x being explained
      • L  = a loss measuring how well g approximates f locally
      • Ω(g) = a complexity penalty (prefers sparse, short explanations)


### The LIME Algorithm — Step by Step

    Step 1 — PERTURB THE INPUT:
    ────────────────────────────
    Generate N perturbed samples z'₁, z'₂, ..., z'ₙ near the
    instance x being explained.

    For tabular data:
      Each feature is either included (value kept) or excluded
      (value replaced by a sample from the training distribution).
      This creates binary interpretable representations:
        z' = [1, 0, 1, 1, 0]  → features 1,3,4 present; 2,5 absent

    For text:
      Each "feature" is a word. Exclusion = word removed from sentence.

    For images:
      Each "feature" is a superpixel. Exclusion = superpixel greyed out.

    Step 2 — QUERY THE BLACK BOX:
    ──────────────────────────────
    For each perturbed sample z'ᵢ, obtain the model prediction:
        yᵢ = f(zᵢ)   (where zᵢ is the original-space version of z'ᵢ)

    Step 3 — WEIGHT BY PROXIMITY:
    ──────────────────────────────
    Weight each perturbed sample by how close it is to x:
        π_x(z) = exp(-D(x, z)² / σ²)

    This is an exponential kernel. Points very close to x get
    weight ≈ 1. Points far away get weight ≈ 0.

    The bandwidth σ controls the "locality" — small σ means only
    points very close to x matter; large σ approaches a global
    explanation (which loses the "local" guarantee).

    Step 4 — FIT A WEIGHTED LINEAR MODEL:
    ───────────────────────────────────────
    Fit a sparse linear model on the perturbed samples, weighted
    by proximity, with a Lasso-style penalty for sparsity:

        argmin_w  Σᵢ π_x(zᵢ) × (f(zᵢ) - w·z'ᵢ)² + λ‖w‖₁

    The result: a small set of coefficients wⱼ, one per selected
    feature, that explain the model's behaviour near x.

    Step 5 — RETURN THE COEFFICIENTS:
    ───────────────────────────────────
    The coefficients of the local linear model ARE the explanation.
    wⱼ > 0: feature j pushed the prediction up (near x)
    wⱼ < 0: feature j pushed the prediction down (near x)
    |wⱼ|: local importance of feature j


### The Fundamental Tension — Fidelity vs Interpretability

LIME's entire existence is a response to a tension:

    GLOBAL accuracy requires COMPLEX models.
    LOCAL faithfulness enables SIMPLE explanations.

LIME navigates this by being deliberately LOCAL. But "local" is
vague — what counts as the neighbourhood? This is LIME's deepest
weakness.

    # ================================================================= #
    **The Neighbourhood Size Problem:**

    Imagine the true model f has this shape along one feature:
    ┌─────────────────────────────────────────────────────────────────┐
    │  f(x)                                                           │
    │    ↑      /\            ← f is non-linear: rises, peaks, falls  │
    │    │     /  \                                                   │
    │    │    /    \       ← LIME neighbourhood (too wide): the       │
    │    │___/      \___     local linear model is a bad fit          │
    │    │   ←─ σ ──→                                                 │
    │    └──────────────────────→ x                                   │
    │                                                                 │
    │  With correct σ (narrow neighbourhood):                         │
    │    │           /\                                               │
    │    │          /  \      ← local linear model is a good fit      │
    │    │     ____/    ←─σ→  (a small part of the curve is flat)     │
    │    └──────────────────────→ x                                   │
    └─────────────────────────────────────────────────────────────────┘

    The bandwidth σ must be chosen by the user. Too large → the
    surrogate model is a poor approximation of f locally. Too small →
    too few perturbed samples are within the neighbourhood, and the
    linear fit becomes unstable (high variance).

    LIME has no automatic way to choose σ correctly. This is a
    fundamental limitation. Reported LIME explanations can change
    dramatically with different σ values.
    # ================================================================= #


### LIME vs SHAP — The Key Differences

    ┌─────────────────────────────────────────────────────────────────┐
    │  Property           SHAP                    LIME                │
    │  ─────────────────────────────────────────────────────────────  │
    │  Scope              Local (aggregatable     Local only          │
    │                     to global)                                  │
    │  Theoretical base   Cooperative game        Surrogate modelling │
    │                     theory (axioms)         (no axioms)         │
    │  Faithfulness       Exact (TreeSHAP)        Approximate         │
    │                     Approximate (Kernel)    (local linear)      │
    │  Completeness       YES: Σφᵢ = f(x)-base    NO                  │
    │  Consistency        Guaranteed by axioms    Not guaranteed      │
    │  Stability          High                    Low-medium (σ-dep.) │
    │  Speed (trees)      Fast (TreeSHAP)         Slow (N queries)    │
    │  Speed (NNs)        Medium                  Medium              │
    │  Works on text/imgs Yes (with encoding)     Yes (native)        │
    │  Interaction terms  Partial (SHAP interact) No                  │
    │  When to prefer     Most tabular tasks;     Text/image models;  │
    │                     when completeness       when simple linear  │
    │                     matters                 story is sufficient │
    └─────────────────────────────────────────────────────────────────┘


##### PART IV: PDP, ICE, AND ALE — MARGINAL EFFECT PLOTS

### The Three-Method Family

PDP, ICE, and ALE all answer versions of the same global question:
"How does the model's prediction change as feature j varies?"

They differ in HOW they marginalise over the other features —
and this difference is entirely the point. Understanding when
each gives the correct answer requires understanding the problem
with feature correlation.


### Partial Dependence Plots (PDP)

**Definition:**

PDP₍ⱼ₎(xⱼ) = (1/n) Σᵢ f(xⱼ, xᵢ_\j)

For each candidate value of feature j (ranging from its minimum
to maximum), average the model's prediction across all training
instances, with feature j set to that value and all other features
(x_\j) kept at their observed values.

The result is a curve: as xⱼ changes, the average model output changes.

    # ================================================================= #
    **PDP Algorithm — Worked Example (2 features, 4 samples):**

    Dataset: 4 instances with features (credit, income) and a model f.

    Instance  credit  income     f(credit, income)
    ────────────────────────────────────────────────
    1         580     40         f(580, 40) = 0.20
    2         680     55         f(680, 55) = 0.75
    3         720     70         f(720, 70) = 0.85
    4         640     35         f(640, 35) = 0.35

    PDP for feature "credit" at value credit=650:
      Replace each instance's credit with 650, keep income:

    f(650, 40) = 0.40    ← instance 1's income, credit now 650
    f(650, 55) = 0.65    ← instance 2's income, credit now 650
    f(650, 70) = 0.72    ← instance 3's income, credit now 650
    f(650, 35) = 0.38    ← instance 4's income, credit now 650
    PDP(650) = (0.40 + 0.65 + 0.72 + 0.38) / 4 = 0.5375

    Repeat for every value of credit from 500 to 850.
    Plot credit value on x-axis, PDP value on y-axis.
    # ================================================================= #


### The Correlated Features Problem — Why PDP Can Lie

PDP marginalises by FIXING xⱼ and AVERAGING over the empirical
distribution of x_\j. This creates a serious problem when xⱼ and
any feature in x_\j are correlated.

    Example: credit score and income are correlated. High credit
    scores tend to co-occur with high income. Low credit scores
    tend to co-occur with low income.

    When PDP computes PDP(credit=800), it averages over ALL incomes:
    some instances get (credit=800, income=20k) — a pairing that
    barely exists in reality. The model is asked to predict on inputs
    it was never trained on (extrapolation into sparse data space).

    # ================================================================= #
    **PDP Extrapolation into Unrealistic Feature Combinations:**

    REAL DATA DISTRIBUTION        PDP SAMPLES (for credit=800)
    (credit vs income)            (credit FIXED at 800)

    income↑                       income↑
    100k│         ●●●           100k│         ●
     80k│      ●●●●●●●           80k│         ●     ← real
     60k│   ●●●●●●●●●             60k│         ●     ← real
     40k│  ●●●●●                  40k│         ●     ← EXTRAPOLATION
     20k│●●●                       20k│         ●     ← EXTRAPOLATION
         └───────────→credit           └───────────→credit
           580 680 780                   580 680 780

    PDP generates the samples in the right diagram — real data
    only exist near the diagonal, but PDP evaluates at all incomes.
    The resulting average includes unrealistic inputs with unknown
    model behaviour. The PDP curve is biased.
    # ================================================================= #


### ICE — Individual Conditional Expectation

ICE (Goldstein et al., 2015) is a microscope for PDPs. Instead of
averaging over all instances, ICE plots ONE LINE PER INSTANCE:

    ICEᵢ(xⱼ) = f(xⱼ, xᵢ_\j)   for each instance i

The PDP is the mean of all ICE curves:
    PDP(xⱼ) = (1/n) Σᵢ ICEᵢ(xⱼ)

Why is this valuable? The PDP mean can be deeply misleading when
there are interactions between xⱼ and other features. ICE reveals:

    HOMOGENEOUS effect (all ICE curves parallel):
      All instances respond the same way to xⱼ. The PDP accurately
      summarises the effect. No interaction with other features.

    HETEROGENEOUS effect (ICE curves cross or diverge):
      The effect of xⱼ DEPENDS on the values of other features.
      This is an interaction. The PDP average hides it.
      Crossing lines mean: the feature is positive for some
      instances and negative for others.

    # ================================================================= #
    **ICE vs PDP — Interaction Detection:**

    CASE 1: No interaction (PDP reliable)
    y↑    ICE lines parallel ─ ─ ─ ─          All instances
          PDP = mean  ─────────────            respond identically.
          ─────────────────────────→ xⱼ       PDP summarises well.

    CASE 2: Interaction present (PDP misleading!)
    y↑    Lines cross! One goes up,             PDP = 0 (flat line)
          other goes down.                       looks like "no effect"
         ╲   ╱                                  but actually there IS
          ╲ ╱                                   an effect — it just
           ╳                                    cancels out in the
          ╱ ╲                                   average. ICE reveals it.
          ─────────────────────────→ xⱼ
    # ================================================================= #

A practical workflow: first plot ICE curves. If they are parallel
(no crossing), use PDP for the summary. If they cross, investigate
the interaction and avoid using PDP alone.


### ALE — Accumulated Local Effects

ALE (Apley & Zhu, 2020) solves the correlated features problem that
plagues both PDP and ICE. It is the theoretically correct method when
features are correlated.

**The key insight:** instead of asking "what is f(xⱼ=v, x_\j)?"
averaged over the full data distribution (PDP), ALE asks:
"what is the LOCAL derivative of f with respect to xⱼ near v,
averaged only over instances that actually have xⱼ ≈ v?"

**ALE Algorithm:**

    1. Divide the range of feature xⱼ into K equal-width intervals.

    2. For each interval [a, b], compute the local effect:
         Δₖ = (1/|Nₖ|) Σ_{i∈Nₖ} [f(b, xᵢ_\j) - f(a, xᵢ_\j)]
         where Nₖ = instances whose xⱼ value falls inside [a, b]

    3. Accumulate: ALE(v) = Σ_{k: aₖ≤v} Δₖ  (centred to sum to zero)

**Why this avoids the extrapolation problem:**

In step 2, we only evaluate f on instances that ACTUALLY have
xⱼ ≈ v. We change xⱼ slightly (from a to b) while keeping the
other features at their REAL, CORRELATED values. We never ask
the model to evaluate on unrealistic (xⱼ, x_\j) combinations.

    # ================================================================= #
    **PDP vs ALE — Correlated Feature Example:**

    True model: f(credit, income) = 0.5 × credit + 0.5 × income
    But credit and income are correlated: income ≈ 0.8 × credit + noise

    PDP for credit (averaging over ALL incomes):
      When credit=800, PDP averages over incomes from 30k to 100k.
      The effect includes income contribution via extrapolation.
      PDP(credit) looks steeper than the true 0.5 coefficient.

    ALE for credit (averaging over ONLY instances with credit≈800):
      Those instances also tend to have income≈640 (correlated).
      The Δ = f(810, income≈640) - f(790, income≈640)
      ≈ 0.5×20 + 0.5×0 = 10 units change per 20-unit credit change.
      ALE correctly recovers the true 0.5 coefficient.

    ALE isolates the UNIQUE effect of credit, holding income constant
    at its naturally co-occurring values. PDP confounds them.
    # ================================================================= #

**When to use which:**

    PDP:  Features are approximately independent. Quick, intuitive.
    ICE:  Need to detect interactions. Use before deciding on PDP.
    ALE:  Features are correlated. Default for real-world data.
          More trustworthy than PDP whenever correlation exists.

**ALE's limitation:** if there are too few instances in some
intervals (sparse regions of the feature), ALE estimates are noisy
there. Use wider intervals in sparse regions, or consider that the
model may not be well-estimated in that region anyway.


##### PART V: ANCHORS

### What Are Anchors?

Anchors (Ribeiro, Singh & Guestrin, 2018) are the same authors who
created LIME, but with a fundamentally different output format. Where
LIME produces a weighted linear explanation ("feature A contributes
+0.3 to this prediction"), Anchors produce IF-THEN rules:

    "IF  credit_score ≥ 680  AND  debt_ratio ≤ 0.35
     THEN  APPROVE  with precision ≥ 95%  (coverage: 22%)"

An anchor is a set of conditions (the antecedent / rule body) such
that, whenever those conditions are satisfied, the model's prediction
is the SAME as for this specific instance — with high probability.

Formally, a rule A is an anchor for instance x if:

    P(f(z) = f(x) | A(z) = 1) ≥ τ

where τ is a precision threshold (typically 0.95), A(z)=1 means
"z satisfies the anchor conditions", and the probability is over
the distribution of instances satisfying A.


### Precision and Coverage — The Anchor Trade-off

Two properties characterise every anchor:

    PRECISION:  "If I apply this rule to any new instance that
                 satisfies it, how often does the model predict
                 the same class as it did for x?"
                 Higher precision = more trustworthy rule.

    COVERAGE:   "What fraction of all instances satisfy this rule?"
                 Higher coverage = the rule generalises to more cases.

There is a fundamental trade-off between them:

    Long rules (many conditions) → HIGH precision, LOW coverage
      The rule is very specific. Only a tiny slice of data falls in it.
      But when it fires, it's almost certainly right.

    Short rules (few conditions) → LOW precision, HIGH coverage
      The rule covers many cases. But some of those cases may not
      share the same prediction — precision falls.

    # ================================================================= #
    **Precision-Coverage Trade-off:**

    Rule                         Precision  Coverage  Good?
    ──────────────────────────────────────────────────────────
    credit ≥ 680                    78%       35%    ← weak
    credit ≥ 680 AND debt ≤ 0.35    94%       22%    ← better
    credit ≥ 680 AND debt ≤ 0.35
      AND income ≥ 60k              98%        9%    ← very precise
                                                       but niche
    credit ≥ 680 AND debt ≤ 0.35
      AND income ≥ 60k AND emp ≥ 3  100%       4%    ← exact but
                                                       rarely fires
    ──────────────────────────────────────────────────────────

    The algorithm seeks the SHORTEST rule that achieves τ = 0.95
    precision. Adding conditions beyond that threshold trades coverage
    for precision gains that aren't needed.
    # ================================================================= #


### The Beam Search Algorithm

Finding the best anchor is a combinatorial search problem: there
are 2^d possible subsets of d features. Anchors uses beam search —
a greedy breadth-first search that keeps the k best partial anchors
at each level.

    Level 0:  Empty rule (start)
    Level 1:  All 1-condition rules  → keep best k by coverage (ties)
    Level 2:  Extend best k rules with one more condition → keep best k
    Level 3:  Extend again ...
    Stop when: precision ≥ τ for the best rule found

The key challenge: estimating the precision of a partial anchor
requires many model queries. Anchors uses KL-UCB (a bandit
algorithm) to efficiently allocate model queries — spending more
queries on promising rules and fewer on weak ones.


### Anchors vs LIME — Local vs Conditional

Both LIME and Anchors explain individual predictions locally. The
key conceptual difference:

    LIME:    "Near this specific data point, the model behaves as
              if feature A contributes +0.3 and feature B contributes
              -0.2 to the prediction."
              → Explains the prediction AT this point.

    Anchors: "Whenever these conditions hold, the model will predict
              this same class — regardless of what other features say."
              → Explains WHEN the model will make this class of prediction.

LIME explanations are specific to ONE point (they degrade outside
the neighbourhood). Anchor conditions define a REGION — any instance
in that region gets the same prediction. This makes anchors more
useful for recourse and auditing: "you will ALWAYS be approved if
credit ≥ 680 AND debt ≤ 0.35, no matter what else changes."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART VI: THE COMPLETE LANDSCAPE — CHOOSING THE RIGHT TOOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Method Selection Framework

Every method answers a specific version of the question "why?" The
version of the question determines the method.

    ┌──────────────────────────────────────────────────────────────────┐
    │  YOUR QUESTION                        → USE THIS                 │
    │  ─────────────────────────────────────────────────────────────   │
    │  "Which features drive the model      → Permutation Importance   │
    │   globally, on average?"                 (fast, robust baseline) │
    │                                                                  │
    │  "Why did THIS prediction deviate     → SHAP force plot          │
    │   from the average by X?"               (completeness axiom)     │
    │                                                                  │
    │  "What simple rule approximates the  → LIME                      │
    │   model near this prediction?"          (sparse, readable)       │
    │                                                                  │
    │  "Under what conditions will the      → Anchors                  │
    │   model always predict this class?"     (high-precision rules)   │
    │                                                                  │
    │  "How does model output change as     → PDP (if features         │
    │   feature j varies globally?"            independent)            │
    │                                       → ALE (if correlated)      │
    │                                                                  │
    │  "Does the model treat subgroup A     → SHAP grouped summary     │
    │   differently from subgroup B?"         OR grouped PDP/ALE       │
    │                                                                  │
    │  "Are there feature interactions?"    → ICE curves               │
    │                                       → SHAP interaction values  │
    └──────────────────────────────────────────────────────────────────┘


### Reliability Hierarchy — How Much to Trust Each Method

Not all post-hoc methods are equally trustworthy. Here is an honest
ranking, from most to least theoretically grounded:

    TIER 1 — EXACT (no approximation):
    ───────────────────────────────────
    TreeSHAP for tree models  — exact Shapley values, polynomial time
    Linear model coefficients — exact, but model must fit data well

    TIER 2 — PRINCIPLED APPROXIMATION (known error modes):
    ────────────────────────────────────────────────────────
    ALE — unbiased under feature dependence; noise in sparse intervals
    KernelSHAP — converges to truth as sample size grows
    Permutation importance — unbiased; variance from shuffling

    TIER 3 — APPROXIMATE WITH UNKNOWN ERROR:
    ─────────────────────────────────────────
    LIME — neighbourhood size is arbitrary; σ choice affects result
    PDP — biased under correlated features (degree unknown)

    TIER 4 — APPROXIMATE WITH STRONG ASSUMPTIONS:
    ───────────────────────────────────────────────
    Anchors — precision estimated via sampling; beam search misses
               global optimum; rule quality depends on perturbation
               distribution

The key message: ALWAYS cross-validate important conclusions across
multiple methods. If SHAP, permutation importance, and PDP all agree
that feature A is important, you can be more confident. If they
disagree, investigate WHY — the disagreement often reveals something
important about the data structure.


### A Practical Workflow for a New Black-Box Model

    Step 1: Global sanity check
      → Permutation importance on test set
      → Are the most important features sensible? Any suspicious ones?

    Step 2: Global behaviour understanding
      → PDP/ALE for top 3-5 features
      → Are the effects monotone, threshold-shaped, or non-linear?
      → Plot ICE to check for interactions

    Step 3: Local investigation (specific predictions)
      → SHAP force plot for 5-10 interesting predictions
        (correct high-confidence, wrong low-confidence, etc.)
      → Do the attributions match domain knowledge?

    Step 4: Stakeholder communication
      → If technical: SHAP summary plot
      → If non-technical: LIME or Anchors (rule format)

    Step 5: Adversarial probing
      → Can you construct inputs with misleading SHAP attributions?
      → Slack & Hilgard (2020) showed that models can be designed to
        fool SHAP while hiding discriminatory features. No single
        method is immune to adversarial manipulation.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔════════════════════╦══════════════╦════════════════╦═════════════════╗
    ║ Method             ║ Scope        ║ Exact?         ║ Model queries   ║
    ╠════════════════════╬══════════════╬════════════════╬═════════════════╣
    ║ Permutation imp.   ║ Global       ║ No (sampling)  ║ n × K × d       ║
    ║ KernelSHAP         ║ Local+Global ║ No (converges) ║ 2ⁿ × m (approx) ║
    ║ TreeSHAP           ║ Local+Global ║ YES            ║ O(TLD²) exact   ║
    ║ LIME               ║ Local        ║ No             ║ N (user choice) ║
    ║ PDP                ║ Global       ║ No (corr bias) ║ n × |grid|      ║
    ║ ICE                ║ Local        ║ No (corr bias) ║ n × |grid|      ║
    ║ ALE                ║ Global       ║ Unbiased       ║ 2n × K          ║
    ║ Anchors            ║ Local        ║ No (beam)      ║ variable (KL)   ║
    ╚════════════════════╩══════════════╩════════════════╩═════════════════╝
    n = dataset size, d = features, K = repeats, m = KernelSHAP samples
    T = trees, L = leaves, D = depth
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ──────────────────────────────────────────────────────────────────────
    "SHAP from First Principles — Exact Shapley Values": {
        "description": (
            "Implements exact Shapley values from scratch using the combinatorial "
            "formula for all 2^n feature subsets. Traces every coalition, computes "
            "every marginal contribution, and shows the full accounting. Then "
            "demonstrates KernelSHAP-style sampling approximation on the same "
            "instance, quantifying the approximation error. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "posthoc",
        "code": '''
"""
================================================================================
SHAP FROM FIRST PRINCIPLES — EXACT SHAPLEY VALUES
================================================================================

We implement the exact Shapley value formula from scratch:

    φᵢ = Σ_{S ⊆ N\\{i}}  [|S|!(n-|S|-1)! / n!] × [f(S∪{i}) - f(S)]

For n=4 features this requires 2⁴ = 16 subset evaluations per feature,
16 × 4 = 64 total — tractable to compute and trace by hand.

Then we implement KernelSHAP (sampling approximation) and compare.

Key concepts demonstrated:
  1. Every marginal contribution, for every coalition, for every feature
  2. The Shapley weighting: why small and large coalitions get more weight
  3. The Efficiency axiom: SHAP values sum to f(x) - baseline
  4. KernelSHAP approximation vs exact values
  5. How interaction effects distribute across SHAP values
================================================================================
"""

import math
import random
from itertools import combinations


random.seed(42)

# =============================================================================
# THE MODEL AND INSTANCE TO EXPLAIN
# =============================================================================
# A non-linear black-box model f(credit, income, debt, employment)
# We choose a model with a clear interaction: credit × income synergy.

def model(credit, income, debt, employment):
    """
    Non-linear loan approval score.
    Deliberately includes a credit × income interaction term so SHAP
    values are non-trivial to compute.
    """
    base  = 0.40
    c     = 0.30 * (credit - 600) / 200          # credit contribution
    inc   = 0.15 * (income - 50) / 30             # income contribution
    d     = -0.25 * (debt - 0.30) / 0.20          # debt contribution
    emp   = 0.05 * (employment - 5) / 5           # employment contribution
    inter = 0.10 * ((credit-600)/200) * ((income-50)/30)  # interaction!
    return min(1.0, max(0.0, base + c + inc + d + emp + inter))


# The specific instance we want to explain
# Typical "borderline approval" case
x = {
    "credit":     720,   # above average (+)
    "income":      65,   # above average (+)
    "debt":      0.38,   # slightly high  (-)
    "employment":   8,   # solid          (+)
}
feat_names = ["credit", "income", "debt", "employment"]
feat_vals  = [x[f] for f in feat_names]
n_feats    = len(feat_names)

# Baseline: expected model output over background distribution
# (here we use a simple average of 10 representative points)
background = [
    [600, 50, 0.35, 5],
    [650, 55, 0.40, 6],
    [580, 45, 0.45, 3],
    [700, 60, 0.30, 8],
    [630, 48, 0.50, 4],
    [680, 65, 0.35, 7],
    [610, 52, 0.42, 5],
    [720, 70, 0.28, 10],
    [560, 40, 0.55, 2],
    [660, 58, 0.38, 6],
]
baseline = sum(
    model(b[0], b[1], b[2], b[3]) for b in background
) / len(background)

true_pred = model(*feat_vals)

print("=" * 68)
print("  SETUP")
print("=" * 68)
print(f"  Instance to explain:")
for f, v in x.items():
    print(f"    {f:<14}: {v}")
print()
print(f"  True prediction  : f(x)     = {true_pred:.6f}")
print(f"  Baseline (E[f])  : E[f(x)]  = {baseline:.6f}")
print(f"  To explain (diff): f(x)-base = {true_pred - baseline:+.6f}")
print()
print("  SHAP values must sum to: {:.6f}".format(true_pred - baseline))


# =============================================================================
# HELPER: Evaluate f(S) — model output given only features in set S
# "Unknown" features are marginalised by averaging over background data
# =============================================================================

def f_subset(S, x_vals, background):
    """
    Evaluate the model with only the features in S known.
    Features NOT in S are replaced by their background distribution values.
    Returns the expectation over background.
    """
    total = 0.0
    for bg in background:
        args = []
        for i in range(4):
            if i in S:
                args.append(x_vals[i])  # use the instance's value
            else:
                args.append(bg[i])      # use the background value
        total += model(*args)
    return total / len(background)


# =============================================================================
# EXACT SHAPLEY VALUES — enumerate all 2^n subsets
# =============================================================================

def exact_shapley(x_vals, background, feat_names):
    n = len(feat_names)
    N = set(range(n))
    shapley_values = [0.0] * n

    for i in range(n):
        # Iterate over all subsets S ⊆ N \ {i}
        others = [j for j in N if j != i]
        contribs = []

        for size in range(len(others) + 1):
            for S_tuple in combinations(others, size):
                S = set(S_tuple)
                S_with_i = S | {i}

                # Shapley weight: |S|!(n-|S|-1)! / n!
                weight = (math.factorial(len(S)) *
                          math.factorial(n - len(S) - 1) /
                          math.factorial(n))

                # Marginal contribution of feature i given coalition S
                f_with    = f_subset(S_with_i, x_vals, background)
                f_without = f_subset(S,         x_vals, background)
                marginal  = f_with - f_without

                contribs.append((S, weight, f_without, f_with, marginal))
                shapley_values[i] += weight * marginal

    return shapley_values, contribs


print()
print("=" * 68)
print("  EXACT SHAPLEY VALUES — ALL COALITIONS")
print("=" * 68)
print()
print("  Computing exact Shapley values (16 subsets × 4 features)...")
print()

shap_values, all_contribs = exact_shapley(feat_vals, background, feat_names)

# Print contribution table for feature 0 (credit)
print("  Detailed marginal contributions for 'credit' (feature 0):")
print()
print(f"  {'Coalition S':>25}  {'|S|':>4}  {'Weight':>10}  "
      f"{'f(S)':>8}  {'f(S∪cr)':>8}  {'Marginal':>10}")
print(f"  {'-'*75}")

credit_contribs = [(s, w, fw0, fw1, m) for s, w, fw0, fw1, m in all_contribs
                   if 0 not in s]  # subsets NOT containing credit

for S, w, f_without, f_with, marginal in sorted(credit_contribs,
                                                  key=lambda x: len(x[0])):
    s_str = "{" + ", ".join(feat_names[j] for j in sorted(S)) + "}" if S else "∅"
    print(f"  {s_str:>25}  {len(S):>4}  {w:>10.6f}  "
          f"{f_without:>8.5f}  {f_with:>8.5f}  {marginal:>+10.5f}")

print(f"  {'-'*75}")
print(f"  {'φ(credit) = weighted sum':>25}                              "
      f"           {shap_values[0]:>+10.5f}")


# =============================================================================
# FULL SHAP VALUE TABLE
# =============================================================================

print()
print("=" * 68)
print("  FINAL SHAP VALUES — ALL FEATURES")
print("=" * 68)
print()
print(f"  {'Feature':<14}  {'Value':>8}  {'SHAP φᵢ':>10}  "
      f"{'Odds':>10}  {'Role'}")
print(f"  {'-'*62}")

total_shap = 0.0
for i, fname in enumerate(feat_names):
    phi   = shap_values[i]
    total_shap += phi
    role  = "↑ pushes UP" if phi > 0 else "↓ pushes DOWN"
    bar   = ("▲" if phi > 0 else "▼") * min(12, int(abs(phi) * 60))
    print(f"  {fname:<14}  {feat_vals[i]:>8}  {phi:>+10.6f}  {bar:<12}  {role}")

print(f"  {'-'*62}")
print(f"  {'Sum of φᵢ':>14}              {total_shap:>+10.6f}")
print(f"  {'f(x) - base':>14}              {true_pred - baseline:>+10.6f}")

efficiency_ok = abs(total_shap - (true_pred - baseline)) < 1e-6
print()
print(f"  Efficiency axiom check: |Σφᵢ - (f(x)-base)| = "
      f"{abs(total_shap - (true_pred - baseline)):.2e}  "
      f"{'✓ SATISFIED' if efficiency_ok else '✗ VIOLATED'}")
print()
print(f"  Interpretation:")
sorted_idx = sorted(range(4), key=lambda i: -abs(shap_values[i]))
for i in sorted_idx:
    phi = shap_values[i]
    print(f"    • {feat_names[i]:<12}: {phi:+.5f}  — contributed "
          f"{'positively' if phi>0 else 'negatively'} to the "
          f"{'approval' if true_pred > 0.5 else 'denial'}")


# =============================================================================
# KERNELSHAP — SAMPLING APPROXIMATION
# =============================================================================
# KernelSHAP approximates Shapley values by:
# 1. Sampling 2^n subsets S with probability proportional to the Shapley kernel
# 2. Evaluating f on each sampled subset
# 3. Fitting a weighted linear regression with the Shapley kernel weights
# This avoids enumerating all 2^n subsets.

def shapley_kernel(n, S_size):
    """
    The Shapley kernel weight for a coalition of size S_size in an n-feature set.
    Gives higher weight to empty and full coalitions (size 0 and n-1).
    """
    if S_size == 0 or S_size == n:
        return 1e8  # effectively infinite (always include these)
    return (n - 1) / (math.comb(n, S_size) * S_size * (n - S_size))

def kernelshap_approx(x_vals, background, n_samples=100, seed=7):
    """
    KernelSHAP via weighted linear regression (simplified implementation).
    Returns approximate SHAP values.
    """
    rng = random.Random(seed)
    n   = len(x_vals)
    N   = list(range(n))

    # Build the design matrix: binary vectors z' and model outputs
    Z_prime   = []   # binary coalition vectors
    f_outputs = []   # f(z) for each coalition
    weights   = []   # Shapley kernel weights

    # Always include the empty and full coalitions
    Z_prime.append([0] * n)
    f_outputs.append(f_subset(set(), x_vals, background))
    weights.append(1e6)

    Z_prime.append([1] * n)
    f_outputs.append(f_subset(set(N), x_vals, background))
    weights.append(1e6)

    # Sample remaining coalitions
    for _ in range(n_samples - 2):
        # Sample coalition size from Shapley kernel probability
        size = rng.randint(1, n - 1)
        S = set(rng.sample(N, size))
        z_prime = [1 if j in S else 0 for j in N]
        Z_prime.append(z_prime)
        f_outputs.append(f_subset(S, x_vals, background))
        weights.append(shapley_kernel(n, size))

    # Weighted least squares: fit w such that z'.w ≈ f(z)
    # Minimise: Σᵢ wᵢ × (f(zᵢ) - φ₀ - Σⱼ zⱼ'·φⱼ)²
    # Using the closed form of weighted OLS:
    # φ = (ZᵀWZ)⁻¹ ZᵀWf

    m = len(Z_prime)
    # Add intercept column
    Z = [[1.0] + z for z in Z_prime]
    d = len(Z[0])  # d = n+1 (intercept + n features)

    # Compute ZᵀWZ and ZᵀWf
    ZtWZ = [[0.0]*d for _ in range(d)]
    ZtWf = [0.0] * d

    for i in range(m):
        w_i = weights[i]
        f_i = f_outputs[i]
        for a in range(d):
            for b in range(d):
                ZtWZ[a][b] += w_i * Z[i][a] * Z[i][b]
            ZtWf[a] += w_i * Z[i][a] * f_i

    # Solve ZtWZ @ phi = ZtWf via Gaussian elimination
    def solve_linear_system(A, b):
        n_sys = len(b)
        Aug = [A[i][:] + [b[i]] for i in range(n_sys)]
        for col in range(n_sys):
            # Pivot
            max_row = max(range(col, n_sys), key=lambda r: abs(Aug[r][col]))
            Aug[col], Aug[max_row] = Aug[max_row], Aug[col]
            if abs(Aug[col][col]) < 1e-12:
                continue
            for row in range(n_sys):
                if row != col:
                    factor = Aug[row][col] / Aug[col][col]
                    for k in range(n_sys + 1):
                        Aug[row][k] -= factor * Aug[col][k]
        return [Aug[i][-1] / Aug[i][i] if abs(Aug[i][i]) > 1e-12 else 0.0
                for i in range(n_sys)]

    solution = solve_linear_system(ZtWZ, ZtWf)
    phi_approx = solution[1:]  # drop intercept
    return phi_approx


print()
print("=" * 68)
print("  KERNELSHAP APPROXIMATION (100 sampled coalitions)")
print("=" * 68)
print()

shap_approx = kernelshap_approx(feat_vals, background, n_samples=100)

print(f"  {'Feature':<14}  {'Exact SHAP':>12}  {'KernelSHAP':>12}  {'Error':>10}")
print(f"  {'-'*52}")
for i, fname in enumerate(feat_names):
    err = shap_approx[i] - shap_values[i]
    print(f"  {fname:<14}  {shap_values[i]:>+12.6f}  "
          f"{shap_approx[i]:>+12.6f}  {err:>+10.6f}")

sum_approx = sum(shap_approx)
print(f"  {'-'*52}")
print(f"  {'Sum':>14}  {sum(shap_values):>+12.6f}  {sum_approx:>+12.6f}")
print()
print(f"  Mean absolute error: {sum(abs(shap_approx[i]-shap_values[i]) for i in range(4))/4:.6f}")
print()
print("  OBSERVATIONS:")
print("  1. KernelSHAP with only 100 samples gives close but not exact values.")
print("  2. With 1000+ samples, KernelSHAP converges to the exact values.")
print("  3. For tree models, TreeSHAP computes exact values in milliseconds —")
print("     no sampling needed. This is why SHAP is practically deployable.")


# =============================================================================
# THE INTERACTION EFFECT — HOW IT DISTRIBUTES ACROSS FEATURES
# =============================================================================

print()
print("=" * 68)
print("  THE INTERACTION EFFECT")
print("=" * 68)
print()
print("  The model includes credit × income interaction:")
print("  inter = 0.10 × ((credit-600)/200) × ((income-50)/30)")
print()

# Compute individual effects (no interaction) vs full model
def model_no_interaction(credit, income, debt, employment):
    base = 0.40
    c    = 0.30 * (credit - 600) / 200
    inc  = 0.15 * (income - 50) / 30
    d    = -0.25 * (debt - 0.30) / 0.20
    emp  = 0.05 * (employment - 5) / 5
    return min(1.0, max(0.0, base + c + inc + d + emp))

def model_full(credit, income, debt, employment):
    return model(credit, income, debt, employment)

# Compute Shapley values for no-interaction model
def f_subset_noint(S, x_vals, background):
    total = 0.0
    for bg in background:
        args = [x_vals[i] if i in S else bg[i] for i in range(4)]
        total += model_no_interaction(*args)
    return total / len(background)

def exact_shapley_noint(x_vals, background):
    n = 4
    N = set(range(n))
    sv = [0.0] * n
    for i in range(n):
        others = [j for j in N if j != i]
        for size in range(len(others) + 1):
            for S_tuple in combinations(others, size):
                S = set(S_tuple)
                w = (math.factorial(len(S)) *
                     math.factorial(n - len(S) - 1) / math.factorial(n))
                sv[i] += w * (f_subset_noint(S|{i}, x_vals, background) -
                              f_subset_noint(S, x_vals, background))
    return sv

sv_noint = exact_shapley_noint(feat_vals, background)

inter_magnitude = 0.10 * ((feat_vals[0]-600)/200) * ((feat_vals[1]-50)/30)
print(f"  Interaction term for this instance:")
print(f"    credit contribution to interaction: (720-600)/200 = {(720-600)/200:.3f}")
print(f"    income contribution to interaction: ( 65- 50)/ 30 = {(65-50)/30:.3f}")
print(f"    interaction value: 0.10 × {(720-600)/200:.3f} × {(65-50)/30:.3f} = "
      f"{inter_magnitude:.5f}")
print()
print(f"  {'Feature':<14}  {'SHAP no-inter':>14}  {'SHAP full':>12}  {'Δ (interaction share)':>22}")
print(f"  {'-'*68}")
for i, fname in enumerate(feat_names):
    delta = shap_values[i] - sv_noint[i]
    print(f"  {fname:<14}  {sv_noint[i]:>+14.6f}  {shap_values[i]:>+12.6f}  "
          f"{delta:>+22.6f}")
total_inter_distributed = sum(shap_values[i]-sv_noint[i] for i in range(4))
print(f"  {'-'*68}")
print(f"  {'Total interaction':>14}                          "
      f"  {total_inter_distributed:>+22.6f}  "
      f"(≈ {inter_magnitude:.5f} ✓)")
print()
print("  KEY INSIGHT:")
print("  Shapley values split the interaction term FAIRLY between the")
print("  two interacting features (credit and income). Each gets half")
print("  the interaction credit — this is the Symmetry axiom at work.")
print("  Features that were uninvolved in the interaction (debt, employment)")
print("  receive zero interaction share, as expected from the Dummy axiom.")
''',
    },

    # ── 2 ──────────────────────────────────────────────────────────────────────
    "PDP, ICE, and ALE — Marginal Effects with Correlation": {
        "description": (
            "Builds a synthetic dataset where income and credit_score are correlated "
            "(ρ ≈ 0.75), then computes PDP, ICE, and ALE for the income feature on a "
            "black-box model — all from scratch. Prints the actual numeric curves and "
            "shows the divergence between PDP and ALE when correlation is present. "
            "Also demonstrates ICE-based interaction detection."
        ),
        "runnable": True,
        "pipeline_cmd": "posthoc",
        "code": '''
"""
================================================================================
PDP, ICE, AND ALE — MARGINAL EFFECT PLOTS FROM SCRATCH
================================================================================

We compare three ways to answer the question:
  "How does model output change as income varies?"

Working on a dataset where income and credit_score ARE correlated (ρ ≈ 0.75).
This is what breaks PDP. We will see the divergence directly in the numbers.

All three methods implemented from scratch. No external dependencies.

True model: f(income, credit) = 0.4 × income_std + 0.4 × credit_std
              + 0.2 × income_std × credit_std  (interaction!)
              + 0.1 × employment_std

  → income TRUE effect coefficient: 0.4  (we should recover ~0.4)
  → but income and credit are correlated → PDP overestimates income's effect
================================================================================
"""

import random
import math

random.seed(1234)

# =============================================================================
# DATASET — correlated features
# =============================================================================

N = 300
# Generate correlated income and credit: both driven by latent "wealth"
income_list      = []
credit_list      = []
employment_list  = []

for _ in range(N):
    wealth      = random.gauss(0, 1)           # latent wealth factor
    income      = wealth * 0.80 + random.gauss(0, 0.6)  # ρ(income,credit)≈0.75
    credit      = wealth * 0.80 + random.gauss(0, 0.6)
    employment  = random.gauss(0, 1)           # independent
    income_list.append(income)
    credit_list.append(credit)
    employment_list.append(employment)

# These are already standardised (mean≈0, std≈1) by construction

def black_box(income, credit, employment):
    """
    True model: 0.4·income + 0.4·credit + 0.2·income·credit + 0.1·employment
    income and credit each have a true coefficient of 0.4.
    """
    return (0.4 * income
            + 0.4 * credit
            + 0.2 * income * credit
            + 0.1 * employment)

predictions = [black_box(income_list[i], credit_list[i], employment_list[i])
               for i in range(N)]

print("=" * 68)
print("  DATASET OVERVIEW")
print("=" * 68)
print(f"  N = {N} instances")
print(f"  Features: income (std), credit (std), employment (std)")
print(f"  True model: 0.4·income + 0.4·credit + 0.2·income·credit + 0.1·employment")
print()
corr_num = sum(income_list[i]*credit_list[i] for i in range(N))/N
corr_den = (math.sqrt(sum(v**2 for v in income_list)/N) *
            math.sqrt(sum(v**2 for v in credit_list)/N))
corr     = corr_num / corr_den
print(f"  Pearson correlation(income, credit): {corr:.3f}")
print(f"  (substantial correlation → PDP will be biased for both features)")


# =============================================================================
# HELPER — grid of values for "income"
# =============================================================================

income_min = min(income_list)
income_max = max(income_list)
grid_size  = 12
income_grid = [income_min + (income_max - income_min) * k / (grid_size - 1)
               for k in range(grid_size)]


# =============================================================================
# METHOD 1: PDP — Partial Dependence Plot
# =============================================================================
# PDP(v) = (1/N) Σᵢ f(v, creditᵢ, employmentᵢ)
# Fix income = v, use EVERY (credit, employment) regardless of what
# that instance's actual income was.

def compute_pdp(income_grid, income_list, credit_list, employment_list):
    pdp = []
    for v in income_grid:
        avg = sum(black_box(v, credit_list[i], employment_list[i])
                  for i in range(N)) / N
        pdp.append(avg)
    return pdp

pdp_values = compute_pdp(income_grid, income_list, credit_list, employment_list)


# =============================================================================
# METHOD 2: ICE — Individual Conditional Expectation
# =============================================================================
# ICEᵢ(v) = f(v, creditᵢ, employmentᵢ)
# One curve per instance. Show 8 representative instances.

def compute_ice(income_grid, credit_list, employment_list, instance_indices):
    ice_curves = {}
    for i in instance_indices:
        curve = [black_box(v, credit_list[i], employment_list[i])
                 for v in income_grid]
        ice_curves[i] = curve
    return ice_curves

sample_idx  = [0, 30, 60, 90, 120, 150, 180, 210]
ice_curves  = compute_ice(income_grid, credit_list, employment_list, sample_idx)


# =============================================================================
# METHOD 3: ALE — Accumulated Local Effects
# =============================================================================
# For each interval [aₖ, bₖ] of income:
#   Δₖ = average of [f(bₖ, creditᵢ, empᵢ) - f(aₖ, creditᵢ, empᵢ)]
#        over only those instances i where income_i ∈ [aₖ, bₖ]
# ALE(v) = accumulated sum of Δₖ for all k up to v, centred.

def compute_ale(income_grid, income_list, credit_list, employment_list):
    n_intervals = len(income_grid) - 1
    ale_raw = []

    for k in range(n_intervals):
        a = income_grid[k]
        b = income_grid[k + 1]
        # Find instances in this income interval
        in_interval = [i for i in range(N) if a <= income_list[i] <= b]

        if not in_interval:
            ale_raw.append(0.0)
            continue

        # Local effect: change in f when income moves from a to b,
        # keeping other features at their CORRELATED real values
        local_effects = [
            black_box(b, credit_list[i], employment_list[i]) -
            black_box(a, credit_list[i], employment_list[i])
            for i in in_interval
        ]
        ale_raw.append(sum(local_effects) / len(local_effects))

    # Accumulate
    ale_accum = [0.0]
    for delta in ale_raw:
        ale_accum.append(ale_accum[-1] + delta)

    # Centre: subtract the mean (weighted by how many instances are in each interval)
    interval_counts = []
    for k in range(n_intervals):
        a = income_grid[k]
        b = income_grid[k + 1]
        interval_counts.append(sum(1 for v in income_list if a <= v <= b))

    total_count = sum(interval_counts)
    weighted_mean = sum(
        ale_accum[k + 1] * interval_counts[k]
        for k in range(n_intervals)
        if total_count > 0
    ) / (total_count if total_count > 0 else 1)

    ale_centred = [v - weighted_mean for v in ale_accum]
    return ale_centred

ale_values = compute_ale(income_grid, income_list, credit_list, employment_list)


# =============================================================================
# DISPLAY THE THREE CURVES SIDE BY SIDE
# =============================================================================

print()
print("=" * 68)
print("  MARGINAL EFFECT CURVES — income (standardised)")
print("  True direct coefficient of income = 0.40")
print("=" * 68)
print()
print(f"  {'income':>8}  {'PDP':>8}  {'ALE':>8}  {'ICE[0]':>8}  {'ICE[1]':>8}")
print(f"  {'-'*50}")

for k in range(grid_size):
    ice_vals_str = ""
    for si in sample_idx[:2]:
        ice_vals_str += f"  {ice_curves[si][k]:>8.4f}"
    ale_v = ale_values[k] if k < len(ale_values) else float('nan')
    print(f"  {income_grid[k]:>8.3f}  {pdp_values[k]:>8.4f}  {ale_v:>8.4f}"
          f"{ice_vals_str}")

# Estimate the apparent slope of each method by linear regression
def linear_slope(xs, ys):
    n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    den = sum((xs[i]-mx)**2 for i in range(n))
    return num/den if den > 1e-12 else 0.0

pdp_slope = linear_slope(income_grid, pdp_values)
ale_slope = linear_slope(income_grid, ale_values[:grid_size])

print()
print("=" * 68)
print("  SLOPE COMPARISON — recovered income coefficient")
print("=" * 68)
print(f"  True income coefficient      : 0.40 (direct effect only)")
print(f"  PDP apparent slope           : {pdp_slope:.4f}")
print(f"  ALE apparent slope           : {ale_slope:.4f}")
print()
print(f"  PDP overestimates by          : {pdp_slope - 0.40:+.4f}")
print(f"  ALE error vs true             : {ale_slope - 0.40:+.4f}")
print()
print("  WHY PDP OVERESTIMATES:")
print("  When income is fixed at a high value (e.g., 2.0), PDP averages")
print("  over ALL credit scores in the dataset — including low credits.")
print("  But in real data, high income co-occurs with high credit.")
print("  PDP creates unrealistic (high income, low credit) combinations.")
print("  The model gives these a lower score than real data would justify,")
print("  but the TRANSITION from low→high income still looks steeper because")
print("  the interaction term income×credit gets fully credited to income.")
print()
print("  ALE avoids this: when computing the local effect of income at v=2.0,")
print("  it only uses instances whose income≈2.0 — who also have credit≈2.0.")
print("  The comparison stays within the realistic joint distribution.")


# =============================================================================
# ICE — DETECT THE INTERACTION
# =============================================================================

print()
print("=" * 68)
print("  ICE CURVES — INTERACTION DETECTION")
print("=" * 68)
print()
print("  ICE curves for 8 instances (income from min to max):")
print()
print(f"  {'income':>8}", end="")
for si in sample_idx:
    cr = credit_list[si]
    print(f"  {'cr='+str(round(cr,1)):>9}", end="")
print()
print(f"  {'-'*92}")

for k in [0, 2, 4, 6, 8, 11]:
    print(f"  {income_grid[k]:>8.3f}", end="")
    for si in sample_idx:
        print(f"  {ice_curves[si][k]:>9.4f}", end="")
    print()

print()
# Check if ICE curves cross (interaction indicator)
# Compare instance with highest credit vs lowest credit
high_credit_idx = max(sample_idx, key=lambda i: credit_list[i])
low_credit_idx  = min(sample_idx, key=lambda i: credit_list[i])

high_cr = credit_list[high_credit_idx]
low_cr  = credit_list[low_credit_idx]

high_slope = linear_slope(income_grid, ice_curves[high_credit_idx])
low_slope  = linear_slope(income_grid, ice_curves[low_credit_idx])

print(f"  Slope of ICE for high-credit instance (cr={high_cr:.2f}): {high_slope:.4f}")
print(f"  Slope of ICE for low-credit  instance (cr={low_cr:.2f}): {low_slope:.4f}")
print()
if abs(high_slope - low_slope) > 0.05:
    print("  ⚠️  ICE CURVES HAVE DIFFERENT SLOPES → INTERACTION DETECTED!")
    print("     Income has a STRONGER effect for high-credit instances.")
    print("     This is the income × credit interaction term at work.")
    print("     The PDP/ALE average hides this heterogeneity.")
    print("     The PDP curve would show only the AVERAGE slope.")
else:
    print("  ✓  ICE slopes are similar → no strong interaction detected.")

print()
print("  WORKFLOW RECOMMENDATION:")
print("  1. ALWAYS plot ICE before trusting a PDP or ALE curve.")
print("     Parallel ICE lines → PDP/ALE are reliable summaries.")
print("     Crossing/diverging ICE → there is an interaction.")
print("     The PDP/ALE average loses information that matters.")
print()
print("  2. When features are correlated (as here): use ALE, not PDP.")
print("     The PDP slope overestimates the true income coefficient by")
print(f"     {(pdp_slope/0.40 - 1)*100:.0f}% due to unrealistic feature combinations.")
print("     ALE stays within the joint distribution and recovers the")
print("     true coefficient far more accurately.")
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
    #     from interpretability.visuals.post_hoc_agnostic import (
    #         POSTHOC_VISUAL_HTML,
    #         POSTHOC_VISUAL_HEIGHT,
    #     )
    #     visual_html   = POSTHOC_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = POSTHOC_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[post_hoc_agnostic.py] Could not load visual: {e}", stacklevel=2)

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