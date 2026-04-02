"""
Anchors — High-Precision Rule-Based Local Explanations
=======================================================

Anchors (Ribeiro, Singh & Guestrin, 2018) produce IF-THEN rules that
describe a sufficient condition for a model's prediction to hold with
high probability. Unlike LIME — which approximates the model locally with
a linear function — an anchor defines a REGION of input space in which
the model's behaviour is predictable and stable.

The core question Anchors answers:

    "Under what conditions will the model always make this prediction —
     regardless of everything else about the input?"

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Anchors — High-Precision Rule-Based Explanations"
DISPLAY_NAME = "03f · Anchors"
ICON = "⚓"
SUBTITLE = "IF-THEN rules that guarantee a model's prediction with high precision"


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

### Why Rules, Not Numbers?

Every interpretability method makes a choice about the FORMAT of its
explanation — coefficients, heatmaps, sorted lists, tree paths. That format
determines who can use the explanation and what they can do with it.

SHAP gives you a number per feature: "income contributed +0.14 to the
predicted approval probability." This is precise, mathematically grounded,
and useful for data scientists. It is not useful for a loan officer deciding
whether to override a denial, or for a patient asking why a treatment was
not recommended, or for a regulator checking whether a rule is being followed.
Numbers require interpretation. Rules do not.

An Anchor explanation reads:

    IF  credit_score ≥ 680  AND  debt_ratio ≤ 0.35
    THEN  APPROVE  with probability ≥ 95%

Any adult can read this. Any auditor can check whether an applicant satisfied
it. Any judge can evaluate whether the condition is discriminatory. Any
engineer can ask: "for how many of our customers does this rule apply?"

This is the design philosophy of Anchors: produce the simplest IF-THEN
condition that makes the model's prediction CERTAIN, not merely probable.
The explanation is a GUARANTEE — a region of the input space in which the
model's output is locked to a specific class.


##### PART I: THE FORMAL DEFINITION

### What is an Anchor?

Let f : X → Y be a trained classifier. Let x be a specific instance we
want to explain. An anchor A for instance x is a decision rule — a
conjunction of feature conditions — such that whenever A is satisfied,
the model predicts the same class as it predicts for x, with high
probability over the distribution of instances that satisfy A.

Formally, A is an anchor for x if:

    E_{z ~ D(·|A)} [ 𝟙{ f(z) = f(x) } ]  ≥  τ

where:
  • D(·|A) is the distribution of instances satisfying A
          (sampled from the neighbourhood of x, with A held fixed)
  • 𝟙{·} is the indicator function (1 if condition holds, 0 otherwise)
  • τ ∈ (0, 1) is the precision threshold, typically set to 0.95

In plain English: if we sample random instances that satisfy the anchor's
conditions, at least τ fraction of them will receive the same model
prediction as x.

The anchor A is written as a conjunction (AND) of conditions:

    A = (featureᵢ ∈ Rᵢ) AND (featureⱼ ∈ Rⱼ) AND ...

For tabular data, each condition Rₖ is typically a range (e.g., income
≥ 50k) or equality (e.g., employment_type = "full-time").

### Precision and Coverage — The Two Properties

Every anchor is characterised by exactly two scalar properties:

    PRECISION:  prec(A) = P(f(z) = f(x) | A(z) = 1)
    COVERAGE:   cov(A)  = P(A(z) = 1)

Precision is how often the anchor's prediction is correct — how trustworthy
the rule is. Coverage is how often the anchor fires — how many instances it
applies to.

These two properties embody the entire design tension of Anchors:

    More conditions → higher precision, lower coverage
    Fewer conditions → lower precision, higher coverage

    # ================================================================== #
    **Precision-Coverage diagram:**

    cov(A)                           cov(A)
    HIGH  ┌───────────────────────┐  HIGH  ┌───────────────────────┐
          │                       │        │  ●  ← credit ≥ 680    │
          │                       │        │    (prec=78%, cov=38%)│
          │                       │        │                       │
          │  ←  all instances     │        │                       │
    LOW   └───────────────────────┘  LOW   └───────────────────────┘
          prec: τ % (guaranteed)           prec: τ % (guaranteed)
          One condition: broad, less sure  Adding "debt ≤ 0.35"
                                           ↓ coverage, ↑ precision

    The algorithm searches for the SHORTEST rule (fewest conditions)
    that achieves precision ≥ τ. Adding conditions beyond τ trades
    coverage for precision gains you do not need.
    # ================================================================== #


### The Distinction From Counterfactuals and LIME

Three local explanation methods — LIME, Counterfactuals, and Anchors —
all explain individual predictions. They answer fundamentally different
questions:

    LIME asks:    "What linear combination of features approximately
                   explains the model's behaviour near this point?"
                   → A weighted list. Specific to one point.

    COUNTERFACTUAL asks: "What is the smallest change to this input that
                          would flip the prediction to a different class?"
                          → A single modified instance. Answers recourse.

    ANCHOR asks:  "For which region of input space will the model always
                   predict this class, with high certainty?"
                   → A rule that applies to many instances.

    # ================================================================== #
    **The three explanations for the same DENIED loan application:**

    LIME:          credit_score contributed -0.22, income contributed +0.11,
                   debt_ratio contributed -0.18  (weights at this point only)

    COUNTERFACTUAL: If credit_score were 672 (up 42 pts) → APPROVE

    ANCHOR:        IF credit_score < 640  →  DENY  (prec: 94%, cov: 31%)
                   (whenever credit_score is below 640, the model denies,
                    regardless of income, debt, employment, or anything else)
    # ================================================================== #

The anchor is the most actionable for regulatory compliance: it tells
you the CONDITION that locks the model into this decision, not just how
much each feature contributed or what the nearest approval looks like.


##### PART II: THE PERTURBATION DISTRIBUTION

### What "Satisfies the Anchor" Means in Practice

The formal definition requires sampling from D(·|A) — the distribution
of instances that satisfy the anchor's conditions. In practice, this
means: take the original instance x, hold the anchor features fixed at
their values in x, and randomly perturb the remaining (non-anchor) features.

This sampling process is fundamental. It determines what "precision" means.

    Example:
    Instance x = (credit=680, income=55k, debt=0.32, employment=8 years)
    Anchor: IF credit ≥ 680 AND debt ≤ 0.35

    To estimate precision:
      Generate samples z where:
        z.credit ≥ 680           (held fixed — anchor condition)
        z.debt ≤ 0.35            (held fixed — anchor condition)
        z.income ~ random        (free — not in anchor)
        z.employment ~ random    (free — not in anchor)
      Run f on each z. Precision = fraction of z where f(z) = f(x) = APPROVE.

The "random" perturbation of non-anchor features follows the training
data distribution — not a uniform distribution. This is important: you
want to evaluate the anchor over realistic input combinations.

For tabular data, the perturbation distribution is typically:
  • Numerical features: sampled from the marginal distribution of that
    feature in the training set (or a Gaussian fit to it)
  • Categorical features: sampled from the empirical frequency of each
    category in the training set


### Why the Perturbation Distribution Matters

Precision is only meaningful relative to a distribution. The same anchor
can have very different precision under different distributions.

    Anchor: IF credit_score ≥ 680

    Under realistic distribution:    prec = 87%   (most real instances
                                                    with high credit also
                                                    have moderate income,
                                                    debt, etc.)
    Under adversarial distribution:  prec = 12%   (if non-anchor features
                                                    are systematically
                                                    perturbed to trigger
                                                    denials)
    Under uniform distribution:      prec = 79%   (all feature values
                                                    equally likely — more
                                                    extreme than realistic)

This is why the distribution choice is a DESIGN DECISION, not a technicality.
Anchors with high precision under the training distribution may have low
precision under a shifted deployment distribution. Calibration to the
deployment context is the practitioner's responsibility.


##### PART III: THE SEARCH PROBLEM — FINDING THE BEST ANCHOR

### The Combinatorial Challenge

For a dataset with d features, each discretised into a finite set of
possible conditions, the number of possible anchors is exponential:
  • For each feature, you can include or exclude it — 2^d rule SHAPES.
  • For each included feature, there may be several threshold choices.
  • In practice, for d=20 features with 5 thresholds each: roughly
    (2×5)^20 ≈ 10²⁶ candidate rules. Exhaustive search is impossible.

The algorithm must find the best anchor efficiently. Two properties guide
the search:

    PROPERTY 1 — MONOTONE PRECISION:
    Adding a condition to a rule can only increase or maintain precision.
    If A ⊂ B (B adds conditions to A), then prec(B) ≥ prec(A).

    Proof: any instance satisfying B also satisfies A. So B's sample is
    a subset of A's sample, filtered by additional constraints. The model
    can only become more consistent within a narrower region.

    This means the search is monotone: once precision ≥ τ, adding
    conditions keeps precision ≥ τ. We search for the SHORTEST such rule.

    PROPERTY 2 — MONOTONE ANTI-COVERAGE:
    Adding a condition can only decrease or maintain coverage.
    cov(B) ≤ cov(A) when A ⊂ B.

    These two monotone properties give the search a direction: go from
    short (high coverage, possibly low precision) to long (lower coverage,
    guaranteed high precision). Stop as soon as precision ≥ τ.

    # ================================================================== #
    **Monotone precision visualised — a lattice of rules:**

    Level 0: {} (empty rule — always fires, precision = base rate)
    Level 1: {credit≥680}    {debt≤0.35}    {income≥50k}    ...
    Level 2: {credit≥680, debt≤0.35}    {credit≥680, income≥50k}  ...
    Level 3: {credit≥680, debt≤0.35, income≥50k}    ...

    prec increases (or stays) as we go DOWN the lattice.
    cov decreases (or stays) as we go DOWN the lattice.
    We stop at the shallowest level where prec ≥ τ.
    # ================================================================== #


### Beam Search — The Algorithm

Anchors uses BEAM SEARCH — a form of greedy best-first search that keeps
the k best candidate rules at each level (the "beam width" k prevents
the exponential blowup of exhaustive search).

    INITIALISATION:
    ─────────────────
    Candidate set = all 1-condition rules over all features.
    Estimate precision of each candidate using the perturbation sampler.
    Keep the top-k by coverage among those with estimated prec ≥ τ.
    If none achieve prec ≥ τ, keep the top-k by estimated precision.

    EXPANSION:
    ──────────
    For each rule A in the current beam:
      For each feature NOT already in A:
        For each threshold condition on that feature:
          Generate A' = A ∪ {new condition}
          Estimate prec(A') and cov(A')

    SELECTION:
    ──────────
    From all expanded candidates A':
      If any A' has prec(A') ≥ τ:
        Among those, select the one with HIGHEST coverage (cov(A')).
        This is the final anchor — return it.
      If none achieve prec ≥ τ:
        Keep the top-k by estimated precision → new beam.
        Expand again.

    STOPPING:
    ─────────
    The search terminates when a sufficiently precise anchor is found,
    or when the maximum rule length (number of conditions) is reached
    (typically 3-4 conditions max), or when no improvement is possible.

    # ================================================================== #
    **Beam search trace (beam width k=2, τ=0.95):**

    Level 0 (all 1-condition candidates):
      {credit≥680}    prec≈79%, cov=38%   ← below τ, keep
      {debt≤0.35}     prec≈71%, cov=42%   ← below τ, keep
      {income≥50k}    prec≈63%, cov=51%   ← not in top-2 beam
      {emp≥5yr}       prec≈58%, cov=49%   ← not in top-2 beam
      Beam = [{credit≥680}, {debt≤0.35}]

    Level 1 (expand the beam):
      {credit≥680, debt≤0.35}  prec≈94%, cov=26%   ← almost!
      {credit≥680, income≥50k} prec≈88%, cov=22%   ← below τ
      {debt≤0.35, income≥50k}  prec≈81%, cov=30%   ← below τ
      {debt≤0.35, emp≥5yr}     prec≈75%, cov=35%   ← below τ
      Beam = [{credit≥680,debt≤0.35}, {credit≥680,income≥50k}]

    Level 2 (expand the beam):
      {credit≥680, debt≤0.35, income≥40k}  prec≈97%, cov=20%  ← prec ≥ τ!
      {credit≥680, debt≤0.35, income≥50k}  prec≈98%, cov=15%  ← prec ≥ τ!
      Select highest coverage among prec≥τ candidates:
      → ANCHOR: IF credit≥680 AND debt≤0.35 AND income≥40k  (prec=97%, cov=20%)
    # ================================================================== #


### KL-UCB — Estimating Precision Efficiently

Each precision estimate requires sampling instances from D(·|A), calling
the model on each, and computing the fraction that match f(x). This is
expensive: a naive estimate for each of the many candidate rules would
require millions of model calls.

The key insight: this is a multi-armed bandit problem. Each candidate
rule A is an "arm." Each model call on a perturbed sample is a Bernoulli
trial — the sample either matches f(x) or does not. We want to identify
which arm has expected reward (precision) ≥ τ, while minimising total
model calls.

Anchors uses KL-UCB (Kullback-Leibler Upper Confidence Bound), a bandit
algorithm that maintains a confidence interval for each arm's true precision
and adaptively allocates samples:

    For each candidate rule A, maintain:
      • n_A  = total samples drawn for A so far
      • s_A  = number of samples where f(z) = f(x)   (successes)
      • p̂_A = s_A / n_A   (estimated precision)

    The KL-UCB upper confidence bound is the largest q such that:
      n_A × KL(p̂_A, q) ≤ log(n_A) + c × log(log(n_A))

    where KL(p, q) = p·log(p/q) + (1-p)·log((1-p)/(1-q)) is the
    Kullback-Leibler divergence between Bernoulli distributions.

    Pull rule A (draw a new sample) if:
      KL-UCB upper bound of A > τ (could still be a winner)
    Stop pulling A if:
      KL-UCB upper bound < τ (this arm cannot be an anchor — eliminate it)

    This gives an ANYTIME algorithm: at any point in the search, the
    best-estimated arm satisfying prec ≥ τ is the current best anchor.
    The algorithm keeps sampling until it is confident (within a
    δ-probability error) about which arms satisfy prec ≥ τ.

The KL-UCB bandit dramatically reduces total model calls by:
  1. Allocating more samples to promising (high-precision) candidates
  2. Quickly eliminating low-precision candidates
  3. Providing probabilistic guarantees on the quality of the result


    # ================================================================== #
    **Why KL divergence, not simpler bounds like Hoeffding?**

    Hoeffding's inequality gives a confidence interval for p̂ that
    is symmetric around the estimate. For Bernoulli experiments near
    p=0 or p=1 (which is the regime of interest for high-precision
    anchors), Hoeffding is conservative — the true CI is much tighter.

    KL divergence naturally handles asymmetry. When p̂_A = 0.93
    (close to τ=0.95), the KL-UCB upper bound is much tighter than
    Hoeffding's, meaning fewer samples are needed to confirm or deny
    that A achieves prec ≥ τ. This is crucial for efficiency when
    many candidate rules hover near the threshold.
    # ================================================================== #


##### PART IV: ANCHORS FOR DIFFERENT DATA TYPES

### Tabular Data

For structured tabular data, the anchor conditions are inequalities or
equalities on individual features. The perturbation strategy is:

    Numerical feature in anchor:   fixed at instance's value (or range)
    Numerical feature NOT in anchor: replaced by a sample from the
                                      empirical marginal distribution
                                      of that feature in the training set

    Categorical in anchor:         fixed at instance's category
    Categorical NOT in anchor:     randomly drawn from empirical
                                    category frequency distribution

The discretisation of numerical features into conditions is a
pre-processing step. Typical approach: fit a quartile grid to each
numerical feature and create conditions like:
    feature ≤ Q1,  Q1 < feature ≤ Q2,  Q2 < feature ≤ Q3,  feature > Q3

The search then considers all combinations of these pre-defined
conditions. This discretisation is lossy — the resulting anchor is
an approximation to the true precision boundary — but it makes the
search tractable and the output human-readable.


### Text Data

For text classification (e.g., sentiment analysis, spam detection),
the "features" are words or n-grams. The anchor conditions are of the form:
    "word W IS present in the document"
    "word W IS NOT present in the document"

The perturbation strategy replaces non-anchor words with words sampled
from the corpus vocabulary (or with UNK tokens). The anchor consists of
the words whose presence / absence is sufficient to lock the classification.

    Example — sentiment classifier, positive review:
    Input: "The acting was excellent and the plot was surprisingly deep"
    Anchor: IF "excellent" is present → POSITIVE (prec=92%, cov=35%)

    The word "excellent" alone is sufficient. Even if all other words
    are randomly replaced, the classifier says POSITIVE with 92%
    probability whenever "excellent" appears.

This is remarkably useful for text models: it reveals which words are
"decision-making words" vs "context words." A model that produces the
same answer regardless of the context around "excellent" is using
that word as a near-sufficient condition — which may be appropriate
(sentiment lexicon) or concerning (brittle pattern matching).


### Image Data

For image classifiers, the "features" are SUPERPIXELS — contiguous
regions of similar colour obtained by segmentation algorithms (e.g.,
SLIC: Simple Linear Iterative Clustering). Each superpixel can be
included (kept at its original value) or excluded (replaced by a
uniform grey or blurred region).

The anchor for an image is the set of superpixels whose presence is
sufficient to lock the classification:

    Example — cat/dog classifier:
    Anchor: IF superpixels {ear region} AND {eye region} are intact
            → CAT  (prec=96%, cov=28%)

    The model needs to see the ears AND the eyes to reliably predict
    "cat." Occluding any other region (body, background, paws) does
    not change the prediction.

This reveals which spatial regions of the image are decision-relevant.
It is analogous to a GradCAM heatmap but expresses the result as a
combinatorial sufficient condition rather than a continuous gradient.

    # ================================================================== #
    **Image anchor — schematic:**

    Original image:        Anchor (shaded = fixed):
    ┌─────────────┐        ┌─────────────┐
    │ background  │        │░░░░░░░░░░░░░│ ← random noise
    │   ┌─ear─┐   │    →   │   ┌─ear─┐   │ ← FIXED (anchor)
    │   │ eye │   │        │   │ eye │   │ ← FIXED (anchor)
    │   └─────┘   │        │░░░└─────┘░░ │ ← random noise
    │   body      │        │░░░░░░░░░░░░ │ ← random noise
    └─────────────┘        └─────────────┘

    Perturb the non-anchor regions. If the model still says "cat"
    in ≥ 95% of perturbations → these two superpixels anchor the
    "cat" prediction.
    # ================================================================== #


##### PART V: PROPERTIES OF ANCHORS — WHAT THE THEORY GUARANTEES


### Sufficiency, Not Necessity

An anchor is a SUFFICIENT condition for the prediction — not a necessary
one. This distinction is critical.

    Anchor: IF credit ≥ 680 AND debt ≤ 0.35 → APPROVE  (prec=96%)

    Sufficient: any instance satisfying this rule gets APPROVE with 96%
    probability. The condition GUARANTEES the outcome.

    NOT necessary: there may be instances that are APPROVED despite NOT
    satisfying the anchor. For example, income ≥ 100k may also guarantee
    approval, even with credit = 600. That is a different anchor.

    # ================================================================== #
    **Sufficient vs Necessary — Venn diagram:**

    ┌─────────────────────────────────────────────────────────┐
    │              All APPROVE instances                      │
    │   ┌─────────────────────┐   ┌────────────────────────┐  │
    │   │  Anchor A satisfied │   │  Anchor B satisfied    │  │
    │   │  credit≥680,        │   │  income≥100k           │  │
    │   │  debt≤0.35          │   │                        │  │
    │   └─────────────────────┘   └────────────────────────┘  │
    │            and other approved instances outside anchors │
    └─────────────────────────────────────────────────────────┘

    Neither anchor covers ALL approved instances.
    Each is sufficient, neither is necessary.
    The union of all anchors approaches a necessary condition,
    but no single anchor claims to be one.
    # ================================================================== #

This has important practical implications:

    1. An anchor explanation should never be read as: "the model approves
       ONLY when credit ≥ 680 AND debt ≤ 0.35." That would be a false claim.

    2. The absence of an anchor for a particular decision does NOT mean
       there is no systematic reason — it may mean the search did not find
       a SHORT sufficient condition (the true sufficient conditions may all
       require many features simultaneously).

    3. Two anchors for two different instances in the same class may be
       DISJOINT — they identify different pathways to the same prediction.


### The Probabilistic Guarantee

The precision guarantee prec(A) ≥ τ is probabilistic, not universal:

    P(f(z) = f(x) | A(z) = 1) ≥ τ

"At least τ fraction of all instances satisfying A get prediction f(x)."

This means (1−τ) of instances satisfying A may get a DIFFERENT prediction.
With τ = 0.95, one in twenty instances satisfying the anchor may be
misclassified by the anchor's claim.

The precision estimate itself is also uncertain: it is estimated from
a finite sample using the KL-UCB bandit. The algorithm provides a
(1−δ)-probability guarantee that the estimated anchor achieves true
precision ≥ τ, where δ is a confidence parameter (typically δ = 0.05).
This means there is a 5% chance the reported anchor has true precision
below τ.

    # ================================================================== #
    **The two layers of uncertainty:**

    Layer 1 — within the anchor's definition:
      prec(A) = 0.96 means 4% of instances satisfying A get wrong class.
      The anchor is not a hard guarantee — it is a probabilistic statement.

    Layer 2 — in the precision estimate:
      The algorithm estimates prec(A) from a finite sample.
      With δ=0.05, there is a 5% chance the true precision is below τ=0.95.
      Increasing sample size reduces Layer 2 uncertainty.
      Layer 1 uncertainty is irreducible (it reflects model behaviour).
    # ================================================================== #


### The Maximality Objective — Why Prefer High Coverage?

Among all rules achieving prec ≥ τ, the algorithm selects the one with
highest coverage. Why?

    A rule with low coverage is valid but fragile. It only applies to a
    tiny slice of the input space. Outside that slice, the same prediction
    may or may not hold. It gives the user very little information about
    the model's general behaviour.

    A rule with high coverage is valid AND informative. It tells you
    that the model's behaviour is ROBUST over a large region. The anchor
    is "anchoring" a lot of predictions — not just the one you asked about.

    HIGH COVERAGE anchor:
      IF credit ≥ 680  →  APPROVE  (prec=94%, cov=38%)
      Applies to 38% of customers. A useful, broad policy statement.

    LOW COVERAGE anchor:
      IF credit = 683 AND income = $55,240 AND age = 34  →  APPROVE
      Applies to 0.001% of customers. Technically valid but useless.

Coverage is maximised subject to the precision constraint — it acts as
a Occam's Razor principle for explanations: prefer the simplest rule
(fewest conditions → highest coverage) that achieves the required precision.

A coverage of cov(A) = 0 would mean the rule never fires (impossible in
practice with sensible conditions). A coverage of cov(A) = 1 would mean
all instances satisfy the anchor — the precision would then be the model's
overall accuracy, which may well be below τ.


##### PART VI: ANCHORS vs LIME — A DEEP COMPARISON

### The Same Authors, Two Philosophies

Both LIME and Anchors were authored by Ribeiro, Singh & Guestrin. They are
not competing methods — they are designed for different audiences with
different needs. The contrast is instructive.

    LIME's philosophy: approximate the model locally with something simple
    (a linear model). Report the approximation's coefficients.

    Anchors' philosophy: find the conditions under which the model's
    output is predictable. Report those conditions as a rule.

The philosophical difference maps directly onto practical tradeoffs:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Property              LIME                 Anchors              │
    │  ──────────────────────────────────────────────────────────────  │
    │  Output format         Weights per feature  IF-THEN rule         │
    │  Theoretical basis     Surrogate model fit  Precision guarantee  │
    │  Locality              One point (σ-radius) A region (the rule)  │
    │  Completeness          No (no sum-to-total) No                   │
    │  Faithfulness          Low-medium (approx.) High (by design)     │
    │  Stability             Low (σ-dependent)    Moderate (search-dep)│
    │  Audience              Data scientists      Domain experts,      │
    │                                             regulators           │
    │  Interpretable to non- No (what is -0.22?)  Yes (read the rule)  │
    │  technical audience?                                             │
    │  Covers many instances No                   YES (cov(A) > 0)     │
    │  Provides recourse?    Partially (signs)    No (it's sufficient, │
    │                                             not necessary)       │
    │  Captures non-linear   Poorly (linear fit)  YES (precision test  │
    │  model behaviour?                            is model-agnostic)  │
    └──────────────────────────────────────────────────────────────────┘

### LIME's Locality vs Anchor's Region

The sharpest conceptual difference:

    LIME explanation: "near THIS POINT, the model behaves approximately
    as if credit_score has weight -0.22 and income has weight +0.11."
    The explanation degrades as you move away from the specific point.

    Anchor explanation: "in THIS REGION (defined by the rule), the model
    RELIABLY predicts DENY with 94% probability."
    The explanation remains valid for ANY instance in the region — not
    just the specific point that generated it.

    # ================================================================== #
    **LIME's point vs Anchor's region:**

    feature j ↑
               │    ×  ←  explained point
               │   /LIME\ ← LIME's local linear approximation
               │  /       \   (valid only near ×)
               │ /         \
               │/            ↘
    ─────────────────────────────────────────────────── feature i →

    Anchor region:
    feature j ↑
               │    ╔══════════════╗
               │    ║  APPROVE     ║ ← anchor region (rule: i≥0.4, j≥0.6)
               │  × ║  (prec=96%)  ║ ← × is inside, but region is bigger
               │    ╚══════════════╝
    ─────────────────────────────────────────────────── feature i →

    The LIME approximation is a local tangent plane.
    The anchor is a box within which the prediction is stable.
    # ================================================================== #

### When to Choose Anchors Over LIME

    Choose Anchors when:
    • The audience is non-technical (decision-makers, judges, patients).
    • The use case requires a POLICY ("what conditions always lead to X?")
      rather than a SCORE ("how much does each feature contribute?").
    • Regulatory compliance demands an auditable, human-readable rule.
    • You need an explanation that generalises to MANY instances sharing
      the same condition, not just one specific prediction.
    • The model is highly non-linear and LIME's linear surrogate is a
      poor fit (the non-linear behaviour violates LIME's assumption
      that the model is approximately linear near x).

    Choose LIME when:
    • The audience is technical and can interpret weights.
    • You need to understand RELATIVE contributions of many features
      simultaneously (LIME reports a weight for every feature; Anchors
      only includes features in the anchor condition).
    • You want an explanation of exactly this one prediction without
      claims about neighbouring instances.
    • Features are continuous and cannot easily be discretised into
      meaningful threshold conditions.


##### PART VII: LIMITATIONS AND HONEST FAILURE MODES

### Limitation 1 — Discretisation Distortion

Anchors requires reducing continuous numerical features to discrete
conditions. The discretisation is done before the search — the algorithm
cannot discover that the real boundary is, say, income ≥ 53,147 if the
discretisation only provides quartile breakpoints at $30k, $50k, $70k, $90k.

The reported anchor condition is always an APPROXIMATION of the true
model boundary. If the true precision boundary for credit_score is at
648, but the discretisation uses 600 and 650 as thresholds, the anchor
will report credit ≥ 650 — which may have precision 91% (below τ), or
credit ≥ 600 — which has coverage 70% but precision only 78%.

    # ================================================================== #
    **Discretisation gap — when the boundary falls between breakpoints:**

    Precision
    ↑ 1.0 │              ←──── true boundary here (credit=648)
          │          ╱
    τ=0.95│         ╱
          │        ╱
          │   ────/─────────── discretisation breakpoints: 600, 650
          │  ╱
          └──────────────────────────────────────→ credit_score
             600              650    700

    The algorithm sees precision at 600 (below τ) and 650 (above τ).
    It reports credit≥650 — but the true sufficient condition is credit≥648.
    The reported anchor is VALID (prec≥τ at credit≥650) but LOSES COVERAGE
    (the 2-point gap between 648 and 650 unnecessarily excludes instances).
    # ================================================================== #

More breakpoints per feature → finer approximation but exponentially
more candidates to search. This is a fundamental tension in the design.


### Limitation 2 — Beam Search Does Not Guarantee Optimality

Beam search is a greedy approximation. It keeps only k candidates at
each level of the search. If the optimal anchor requires two conditions
that individually have low precision (neither rises to the top-k beam at
Level 1), beam search may miss it entirely.

    # ================================================================== #
    **When beam search fails — the "individually bad, jointly good" case:**

    Suppose the true high-precision anchor is:
      IF feature_A = LOW AND feature_B = HIGH  →  DENY (prec=97%)

    But:
      {feature_A = LOW} alone → prec=65%  (not in beam)
      {feature_B = HIGH} alone → prec=62%  (not in beam)

    If beam width k=2 and we keep only the top-2 by precision at Level 1,
    neither feature_A=LOW nor feature_B=HIGH makes it into the beam.
    Their combination — the true anchor — is NEVER EVALUATED.
    Beam search returns a different (worse) anchor.

    Fix: wider beam (higher k). But k=1000 means 1000× more model calls.
    There is no free lunch in combinatorial search.
    # ================================================================== #

In practice, beam widths of k=2 to k=10 work well for most tabular
datasets. Wider beams are needed when there are strong interaction effects
that require multiple features to be specified simultaneously.


### Limitation 3 — Anchors Are Not Unique

There may be many equally valid anchors for the same instance — rules that
all achieve prec ≥ τ with different (feature, condition) combinations.
The algorithm returns ONE of them, chosen by coverage maximisation within
the beam search. But other valid anchors exist and may tell a different
story about the model's behaviour.

    Example:
    For the same APPROVE instance, both of these may be valid anchors:
      Anchor A: IF credit ≥ 680 AND debt ≤ 0.35   (prec=96%, cov=22%)
      Anchor B: IF income ≥ 90k                    (prec=95%, cov=14%)

    Anchor A emphasises credit and debt. Anchor B emphasises income.
    Both are true. The algorithm returns A (higher coverage). But a
    regulator might want to see BOTH — to understand that the model
    has multiple pathways to APPROVE.

    This non-uniqueness is not a bug; it reflects genuine model
    complexity. The model may have genuinely disjoint sufficient conditions.
    But it means a single anchor reported by the algorithm is not a
    COMPLETE description of all the model's reasons.


### Limitation 4 — Precision Means Nothing Outside the Perturbation Distribution

If the perturbation distribution used during search differs from the
actual distribution of instances the model will encounter in deployment,
the precision estimate is meaningless.

    Training set perturbation: anchors are estimated on the training
    data distribution. If deployment data has a different distribution
    (distribution shift), the true precision of the anchor in deployment
    may be much lower.

    Example:
    Anchor trained on 2019 loan applicants: credit ≥ 680 → APPROVE (prec=96%)
    In 2024, economic conditions change. High-credit applicants now have
    much higher debt ratios. The anchor fires on instances with high credit
    AND high debt — which the model was not trained to approve so reliably.
    True precision in 2024: prec=78%  (below τ=0.95).

    Anchors should be RE-ESTIMATED when the input distribution shifts.
    This is not a failure of the concept — it is a reminder that all
    data-driven explanations are valid only under the data distribution
    used to generate them.


##### PART VIII: A COMPLETE WORKED EXAMPLE BY HAND

### Building an Anchor Step by Step

We trace the entire anchor-finding process for one denied loan applicant,
using a tiny dataset where precision can be computed exactly.

    Model: f(credit, income, debt) → {APPROVE, DENY}
    True rule: APPROVE if (credit ≥ 640 AND debt ≤ 0.45) OR income ≥ 80k
               DENY otherwise.

    Instance x = (credit=590, income=55k, debt=0.50)
    Model prediction: f(x) = DENY  (credit<640, debt>0.45, income<80k)

    Goal: find a rule A such that P(f(z)=DENY | A(z)=1) ≥ 0.95

    Available conditions (from training-set quartile discretisation):
      credit: < 580,  580–630,  630–680,  > 680
      income: < 40k,  40k–60k,  60k–80k,  > 80k
      debt:   < 0.30, 0.30–0.45, 0.45–0.60, > 0.60

    LEVEL 1 — Evaluate 1-condition candidates:

    For each condition C, sample 20 perturbed instances z where:
      z satisfies C, all other features are random from the training dist.
      Count how many get DENY (= same as f(x) = DENY).

    {credit < 580}:    16/20 DENY → prec = 80%  (below τ=0.95)
    {credit ∈ 580-630}: 15/20 DENY → prec = 75%  (below τ)
    {debt > 0.45}:     17/20 DENY → prec = 85%  (below τ)
    {income < 40k}:    14/20 DENY → prec = 70%  (below τ)
    {income 40-60k}:   13/20 DENY → prec = 65%  (below τ)

    No 1-condition anchor achieves prec ≥ 0.95.
    Beam = top-2 by prec: [{credit<580}, {debt>0.45}]

    LEVEL 2 — Expand the beam with one more condition:

    From {credit<580}, add each other condition:
      {credit<580, debt>0.45}:     19/20 DENY → prec = 95%  ← prec ≥ τ!
      {credit<580, income 40-60k}: 18/20 DENY → prec = 90%  (below τ)
      {credit<580, income<40k}:    17/20 DENY → prec = 85%  (below τ)

    From {debt>0.45}, add each other condition:
      {debt>0.45, credit<580}:     19/20 DENY → prec = 95%  ← prec ≥ τ!
      {debt>0.45, credit 580-630}: 19/20 DENY → prec = 95%  ← prec ≥ τ!
      {debt>0.45, income<40k}:     18/20 DENY → prec = 90%  (below τ)

    Three candidates achieve prec ≥ τ. Select the one with highest coverage:
      {credit<580, debt>0.45}:      cov = 8%
      {debt>0.45, credit<580}:      cov = 8%   (same rule, different order)
      {debt>0.45, credit 580-630}:  cov = 11%  ← HIGHEST COVERAGE

    FINAL ANCHOR: IF debt > 0.45 AND credit ∈ [580, 630]
                  → DENY  (prec=95%, cov=11%)

    INTERPRETATION:
    For this specific denied applicant (debt=0.50, credit=590):
    The model's DENY prediction holds reliably as long as the debt ratio
    is above 0.45 AND the credit score stays in the 580-630 range.
    Even if income changes dramatically, even if employment changes —
    as long as those two conditions hold, the model says DENY with 95%
    confidence.

    The anchor uses the features that are SUFFICIENT to guarantee the
    denial, not all features that influenced it. Income and employment
    were not needed — they do not add to the certainty of the denial
    once credit and debt are specified.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔══════════════════════════╦═════════════════╦════════════════════════════╗
    ║ Property                 ║ Anchors         ║ Comparison                 ║
    ╠══════════════════════════╬═════════════════╬════════════════════════════╣
    ║ Output format            ║ IF-THEN rule    ║ LIME: weights, SHAP: values║
    ║ Scope                    ║ Local → region  ║ LIME: local point          ║
    ║ Precision guarantee      ║ Yes (τ, prob.)  ║ LIME: no guarantee         ║
    ║ Explanation is non-tech  ║ Yes             ║ LIME: no                   ║
    ║ Covers multiple instances║ Yes (cov > 0)   ║ LIME: no                   ║
    ║ Model queries needed     ║ Variable        ║ LIME: N (user-defined)     ║
    ║                          ║ (KL-UCB driven) ║                            ║
    ║ Optimal solution         ║ Not guaranteed  ║ LIME: not guaranteed       ║
    ║                          ║ (beam search)   ║                            ║
    ║ Handles non-linear model ║ Yes             ║ LIME: poorly               ║
    ║ Unique explanation       ║ No (many valid) ║ LIME: unique (given σ)     ║
    ║ Corr. feature robust     ║ Partial         ║ LIME: No                   ║
    ║ Works on text / images   ║ Yes (native)    ║ LIME: Yes (native)         ║
    ╚══════════════════════════╩═════════════════╩════════════════════════════╝

    Key quantities:
      prec(A) = P(f(z) = f(x) | A(z) = 1)  ← precision (≥ τ by design)
      cov(A)  = P(A(z) = 1)                 ← coverage (maximised)
      H_j     = n_A × KL(p̂_A, q) ≤ log(n_A) + c·log(log(n_A))  ← KL-UCB bound

    Search complexity: O(k × B × C × n_samples_per_candidate)
      k = beam width, B = max rule length, C = candidate conditions per feature
      n_samples = determined adaptively by KL-UCB bandit
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Anchors from Scratch — Beam Search with Precision Estimation": {
        "description": (
            "Implements the full Anchors algorithm from scratch: candidate generation, "
            "precision estimation via perturbation sampling, beam search over rule "
            "lengths, and coverage computation. Runs on a synthetic loan-approval model "
            "with a known non-linear decision boundary. Explains one denied and one "
            "approved instance, prints every beam expansion step, and compares the "
            "final anchor against a LIME-style linear approximation to illustrate the "
            "conceptual difference. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "anchors",
        "code": '''
"""
================================================================================
ANCHORS FROM SCRATCH — BEAM SEARCH AND PRECISION ESTIMATION
================================================================================

We implement the core anchor-finding algorithm from scratch:
  1. Discretise features into candidate conditions
  2. For each candidate rule, estimate precision by perturbation sampling
  3. Use beam search to expand from 1-condition to multi-condition rules
  4. Return the highest-coverage rule that achieves precision ≥ τ

We then run it on two instances (one DENY, one APPROVE) and trace every
step of the beam expansion so the algorithm is fully transparent.

Model (known decision boundary — used as ground truth comparison):
  APPROVE if (credit_score ≥ 640 AND debt_ratio ≤ 0.45) OR income ≥ 80k
  DENY otherwise.
================================================================================
"""

import random
import math
from itertools import combinations


random.seed(7)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL AND TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────

def model(credit, income, debt):
    """True non-linear decision boundary."""
    if income >= 80:
        return 1   # APPROVE
    if credit >= 640 and debt <= 0.45:
        return 1   # APPROVE
    return 0       # DENY

FEAT_NAMES = ["credit", "income", "debt"]

# Training distribution (used for perturbation sampling)
def sample_training():
    return {
        "credit": random.gauss(630, 80),
        "income": random.gauss(58, 25),
        "debt":   random.gauss(0.38, 0.12),
    }

# Build training set
TRAIN = [sample_training() for _ in range(2000)]
for s in TRAIN:
    s["credit"] = max(400, min(850, s["credit"]))
    s["income"] = max(15,  min(150, s["income"]))
    s["debt"]   = max(0.05, min(0.85, s["debt"]))

# ─────────────────────────────────────────────────────────────────────────────
# DISCRETISATION — generate candidate conditions for each feature
# ─────────────────────────────────────────────────────────────────────────────

def make_conditions(train, feat_names, n_quantiles=4):
    """
    For each feature, create n_quantiles threshold conditions.
    Returns a list of (feature_name, operator, threshold) triples.
    """
    conditions = []
    for fname in feat_names:
        vals = sorted(s[fname] for s in train)
        quantiles = [vals[int(i * len(vals) / n_quantiles)]
                     for i in range(1, n_quantiles)]
        for q in quantiles:
            conditions.append((fname, "<=", round(q, 2)))
            conditions.append((fname, ">",  round(q, 2)))
    return conditions

ALL_CONDITIONS = make_conditions(TRAIN, FEAT_NAMES, n_quantiles=4)

def condition_str(cond):
    fname, op, val = cond
    return f"{fname} {op} {val}"

def satisfies(sample, rule):
    """Check whether a sample (dict) satisfies all conditions in a rule."""
    for (fname, op, val) in rule:
        if op == "<=":
            if not (sample[fname] <= val):
                return False
        else:
            if not (sample[fname] > val):
                return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# PRECISION ESTIMATION — perturbation sampling
# ─────────────────────────────────────────────────────────────────────────────

def estimate_precision(rule, x_instance, target_pred, n_samples=500):
    """
    Estimate P(f(z) = target_pred | A(z) = 1) by perturbation sampling.

    For each sample:
      - Keep anchor features at x_instance's values (they satisfy A by design).
      - Randomly perturb non-anchor features from the training distribution.
      - Check if z satisfies A (redundant for anchor features; needed for
        non-anchor features to stay "free").

    Returns (precision_estimate, coverage_estimate, n_satisfying).
    """
    anchor_feats = {fname for (fname, _, _) in rule}
    matches = 0
    satisfying = 0

    for bg in random.choices(TRAIN, k=n_samples):
        # Build perturbed instance z:
        #   anchor features → x_instance's value
        #   non-anchor features → background value
        z = {}
        for fname in FEAT_NAMES:
            if fname in anchor_feats:
                z[fname] = x_instance[fname]
            else:
                z[fname] = bg[fname]

        if not satisfies(z, rule):
            continue  # shouldn't happen for anchor features, but guard

        satisfying += 1
        pred = model(z["credit"], z["income"], z["debt"])
        if pred == target_pred:
            matches += 1

    if satisfying == 0:
        return 0.0, 0.0, 0

    prec = matches / satisfying
    cov  = satisfying / n_samples
    return prec, cov, satisfying

def estimate_coverage(rule, n_samples=1000):
    """Estimate P(A(z)=1) by sampling from the full training distribution."""
    count = sum(1 for bg in random.choices(TRAIN, k=n_samples)
                if satisfies(bg, rule))
    return count / n_samples

# ─────────────────────────────────────────────────────────────────────────────
# BEAM SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def find_anchor(x_instance, target_pred, tau=0.90, beam_width=3,
                max_length=4, n_precision_samples=400, verbose=True):
    """
    Find the highest-coverage rule A such that prec(A) >= tau.

    Returns (best_rule, precision, coverage) or (None, 0, 0) if not found.
    """
    feat_to_conditions = {}
    for cond in ALL_CONDITIONS:
        fname = cond[0]
        if fname not in feat_to_conditions:
            feat_to_conditions[fname] = []
        feat_to_conditions[fname].append(cond)

    # Filter to conditions satisfied by x_instance
    valid_conditions = [c for c in ALL_CONDITIONS if satisfies(x_instance, [c])]

    if verbose:
        print()
        msg = (f"  Instance: credit={x_instance['credit']:.0f}, "
               f"income=${x_instance['income']:.0f}k, "
               f"debt={x_instance['debt']:.2f}")
        print(msg)
        print(f"  Prediction: {'APPROVE' if target_pred==1 else 'DENY'}")
        print(f"  Valid conditions for this instance: {len(valid_conditions)}")
        print(f"  Precision threshold τ = {tau}")

    beam = [[]]  # start with empty rule
    best_anchor = None
    best_prec   = 0.0
    best_cov    = 0.0

    for level in range(1, max_length + 1):
        if verbose:
            print()
            print(f"  ── Level {level} (rules with {level} condition(s)) ──")

        candidates = []
        already_in_beam_feats = {c[0] for rule in beam for c in rule}

        for rule in beam:
            rule_feats = {c[0] for c in rule}
            for cond in valid_conditions:
                if cond[0] in rule_feats:
                    continue  # already using this feature
                if cond in rule:
                    continue
                new_rule = rule + [cond]
                candidates.append(new_rule)

        # Deduplicate by frozenset of conditions
        seen = set()
        unique_candidates = []
        for r in candidates:
            key = frozenset(r)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(r)
        candidates = unique_candidates

        if not candidates:
            break

        # Estimate precision for each candidate
        evaluated = []
        for rule in candidates:
            prec, cov, n_sat = estimate_precision(
                rule, x_instance, target_pred, n_precision_samples)
            real_cov = estimate_coverage(rule, n_samples=500)
            evaluated.append((rule, prec, real_cov, n_sat))

        # Sort by precision descending for display
        evaluated.sort(key=lambda x: -x[1])

        if verbose:
            print(f"  {'Rule':<45} {'Prec':>7} {'Cov':>7}")
            print(f"  {'-'*62}")
            for rule, prec, cov, n_sat in evaluated[:6]:
                rule_str = " AND ".join(condition_str(c) for c in rule)
                marker = " ← ✓ prec ≥ τ" if prec >= tau else ""
                print(f"  {rule_str:<45} {prec:>7.1%} {cov:>7.1%}{marker}")
            if len(evaluated) > 6:
                print(f"  ... ({len(evaluated)-6} more candidates)")

        # Check if any candidate achieves prec >= tau
        passing = [(r, p, c, n) for r, p, c, n in evaluated if p >= tau]
        if passing:
            # Among passing candidates, pick highest coverage
            passing.sort(key=lambda x: -x[2])
            best_rule, best_prec, best_cov, _ = passing[0]
            best_anchor = best_rule
            if verbose:
                print()
                print(f"  ✓ Found anchor at level {level}!")
                print(f"    Selected (highest coverage among prec≥τ):")
                rule_str = " AND ".join(condition_str(c) for c in best_rule)
                print(f"    IF {rule_str}")
                print(f"    Precision: {best_prec:.1%}   Coverage: {best_cov:.1%}")
            break

        # Keep top-k by precision as the new beam
        beam = [r for r, p, c, n in evaluated[:beam_width]]

    return best_anchor, best_prec, best_cov


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — Two instances
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("  ANCHORS DEMO — BEAM SEARCH WITH PRECISION ESTIMATION")
print("=" * 68)
print()
print(f"  Training set size: {len(TRAIN)}")
print(f"  Candidate conditions: {len(ALL_CONDITIONS)}")
print(f"  Features: {FEAT_NAMES}")

# Instance 1 — DENY
x_deny = {"credit": 590.0, "income": 52.0, "debt": 0.51}
pred_deny = model(x_deny["credit"], x_deny["income"], x_deny["debt"])

print()
print("=" * 68)
print("  INSTANCE 1 — DENIED APPLICANT")
print("=" * 68)
anchor_deny, prec_deny, cov_deny = find_anchor(
    x_deny, pred_deny, tau=0.90, beam_width=3, verbose=True)

# Instance 2 — APPROVE
x_approve = {"credit": 710.0, "income": 65.0, "debt": 0.30}
pred_approve = model(x_approve["credit"], x_approve["income"], x_approve["debt"])

print()
print("=" * 68)
print("  INSTANCE 2 — APPROVED APPLICANT")
print("=" * 68)
anchor_approve, prec_approve, cov_approve = find_anchor(
    x_approve, pred_approve, tau=0.90, beam_width=3, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON — Anchor vs LIME-style linear approximation
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  COMPARISON — ANCHOR vs LIME-STYLE APPROXIMATION")
print("=" * 68)

def lime_approx(x_instance, target_pred, n_samples=300):
    """
    Simplified LIME: sample perturbed instances, weight by proximity,
    fit a weighted linear model. Report coefficients.
    """
    anchor_feats = set()   # LIME uses no anchor features — all are free
    feat_names   = FEAT_NAMES

    # Standardisation constants from training set
    means = {f: sum(s[f] for s in TRAIN)/len(TRAIN) for f in feat_names}
    stds  = {f: math.sqrt(sum((s[f]-means[f])**2 for s in TRAIN)/len(TRAIN))
             for f in feat_names}

    # Normalise x_instance
    x_std = {f: (x_instance[f]-means[f])/(stds[f]+1e-9) for f in feat_names}

    samples, targets, weights = [], [], []
    for bg in random.choices(TRAIN, k=n_samples):
        z = {f: bg[f] for f in feat_names}
        # Compute Euclidean distance in standardised space
        dist = math.sqrt(sum(
            ((z[f]-means[f])/(stds[f]+1e-9) - x_std[f])**2
            for f in feat_names))
        # Exponential kernel
        sigma = 0.75
        w = math.exp(-dist**2 / sigma**2)
        pred = model(z["credit"], z["income"], z["debt"])
        label = 1 if pred == target_pred else 0
        z_std = [(z[f]-means[f])/(stds[f]+1e-9) for f in feat_names]
        samples.append(z_std + [1.0])  # +1 for intercept
        targets.append(label)
        weights.append(w)

    # Weighted OLS
    d = len(feat_names) + 1
    XtWX = [[0.0]*d for _ in range(d)]
    XtWy = [0.0]*d
    n_s = len(samples)
    for i in range(n_s):
        w = weights[i]
        y = targets[i]
        for a in range(d):
            for b in range(d):
                XtWX[a][b] += w * samples[i][a] * samples[i][b]
            XtWy[a] += w * samples[i][a] * y

    # Solve XtWX @ coef = XtWy (Gaussian elimination)
    Aug = [XtWX[i][:] + [XtWy[i]] for i in range(d)]
    for col in range(d):
        pivot = max(range(col, d), key=lambda r: abs(Aug[r][col]))
        Aug[col], Aug[pivot] = Aug[pivot], Aug[col]
        if abs(Aug[col][col]) < 1e-10:
            continue
        for row in range(d):
            if row != col:
                f = Aug[row][col] / Aug[col][col]
                for k in range(d+1):
                    Aug[row][k] -= f * Aug[col][k]
    coefs = [Aug[i][-1] / (Aug[i][i] if abs(Aug[i][i]) > 1e-10 else 1e-10)
             for i in range(d)]

    return {feat_names[i]: coefs[i] for i in range(len(feat_names))}

lime_deny = lime_approx(x_deny, pred_deny)
lime_approve = lime_approx(x_approve, pred_approve)

print()
print(f"  FOR DENIED INSTANCE (credit=590, income=$52k, debt=0.51):")
print()
print(f"  ANCHOR explanation:")
if anchor_deny:
    rule_str = " AND ".join(condition_str(c) for c in anchor_deny)
    print(f"    IF {rule_str}")
    print(f"    → DENY  (precision: {prec_deny:.1%}, coverage: {cov_deny:.1%})")
    print(f"    Meaning: whenever these conditions hold — no matter what")
    print(f"    income, employment, or any other feature says — the model")
    print(f"    predicts DENY with {prec_deny:.1%} probability.")
else:
    print(f"    (No anchor found at τ=0.90)")

print()
print(f"  LIME explanation (local linear approximation):")
for fname, coef in sorted(lime_deny.items(), key=lambda x: -abs(x[1])):
    direction = "↑ towards DENY" if coef > 0 else "↓ away from DENY"
    print(f"    {fname:<12}: {coef:>+.4f}  {direction}")
print(f"    Meaning: near this point, credit has the largest weight.")
print(f"    But this linear approximation degrades as you move away.")
print(f"    It says nothing about which conditions lock in the denial.")

print()
print(f"  CONCEPTUAL DIFFERENCE:")
print(f"  • LIME gives a score per feature at THIS POINT ONLY.")
print(f"    (What contributes most to this specific DENY prediction?)")
print(f"  • Anchor gives a REGION that guarantees DENY with high probability.")
print(f"    (Under what conditions does the model always deny?)")
print(f"  • Both are useful. Neither replaces the other.")
print(f"    LIME is for data scientists. Anchor is for everyone else.")


# ─────────────────────────────────────────────────────────────────────────────
# PRECISION SENSITIVITY — what happens when τ changes?
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  PRECISION-COVERAGE TRADE-OFF — varying τ (denied instance)")
print("=" * 68)
print()
print(f"  Higher τ → shorter, narrower anchors → lower coverage")
print(f"  Lower τ → broader anchors → higher coverage, less certain")
print()
print(f"  {'τ':>6}  {'Anchor conditions':^50}  {'prec':>6}  {'cov':>6}")
print(f"  {'-'*75}")

for tau_test in [0.70, 0.80, 0.90, 0.95]:
    a, p, c = find_anchor(x_deny, pred_deny, tau=tau_test,
                          beam_width=3, verbose=False)
    if a:
        rule_str = " AND ".join(condition_str(cond) for cond in a)
    else:
        rule_str = "(no anchor found)"
    conds_count = len(a) if a else 0
    print(f"  {tau_test:>6.2f}  {rule_str:<50}  {p:>6.1%}  {c:>6.1%}")

print()
print("  KEY INSIGHT: as τ increases, the algorithm adds conditions to")
print("  narrow the anchor region, trading coverage for certainty.")
print("  The minimum-length anchor at each τ level reflects the model's")
print("  true decision boundary: at τ=0.95, you need more conditions")
print("  because the model's high-certainty region is more restricted.")
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
    #     from interpretability.visuals.anchors import (
    #         ANCHORS_VISUAL_HTML,
    #         ANCHORS_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ANCHORS_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ANCHORS_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[anchors.py] Could not load visual: {e}", stacklevel=2)

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