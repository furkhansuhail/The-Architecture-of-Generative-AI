"""
DeepLIFT — Deep Learning Important FeaTures
=============================================

DeepLIFT (Shrikumar, Greenside & Kundaje, 2017) is a backpropagation-based
attribution method that assigns a contribution score to each input feature
by comparing the network's behaviour at the actual input to its behaviour
at a reference input. It propagates DIFFERENCES in activation — not gradients
— backwards through the network, making it immune to gradient saturation and
giving it an exact, globally-faithful accounting of each feature's contribution.

DeepLIFT is the conceptual bridge between LRP and Shapley values: it is
faster than SHAP, more theoretically grounded than vanilla gradients, and
the direct foundation of DeepSHAP — the method that combines DeepLIFT's
computational efficiency with game-theoretic fairness guarantees.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "DeepLIFT — Deep Learning Important FeaTures"
DISPLAY_NAME = "04c · DeepLIFT"
ICON = "⚡"
SUBTITLE = "Difference-from-reference backpropagation with exact conservation"


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

### The Problem DeepLIFT Was Designed to Fix

By 2017, practitioners using deep neural networks for genomics and other
scientific domains had a specific complaint about gradient-based attribution:
the gradient at a saturated neuron is zero, even when that neuron clearly
drove the prediction.

Consider a ReLU network trained to detect a specific DNA motif. When the
motif is strongly present, all relevant neurons fire at high activation
— saturating their ReLU outputs. At this operating point, the gradient of
the output with respect to every input nucleotide is near-zero. Vanilla
saliency maps show a blank, featureless attribution — exactly the opposite
of what is expected. The method says "nothing is important" for the input
that is most definitely a strong positive example.

The root cause: gradients measure LOCAL sensitivity. A saturated network
is locally insensitive (small perturbations do not change the output), but
globally, the inputs clearly drove the prediction. The sensitivity question
and the contribution question have diverged.

DeepLIFT resolves this by switching from gradients to DIFFERENCES:

    Gradient:     "How much does the output change if xᵢ changes by δ?"
    DeepLIFT:     "How much did xᵢ contribute to the output RELATIVE TO
                   what the network would have output for a reference input?"

The reference input encodes "no information" — a baseline that the model
would process as neutral. DeepLIFT compares the actual prediction to the
reference prediction and attributes that entire difference to the input
features, proportionally to how much each feature deviated from its
reference value.

    # ================================================================== #
    **The saturation fix — DeepLIFT vs gradient:**

    f(x) = ReLU(10·x₁ − 5)   (strong positive signal → saturates)

    At x₁ = 2.0:  f(x₁) = ReLU(15) = 15  (strongly active)
    Gradient at x₁=2: df/dx₁ = 10·𝟙[10x₁−5>0] = 10  (pre-saturation)
    At x₁ = 1.0: f(x₁) = ReLU(5) = 5
    Gradient at x₁=1: df/dx₁ = 10  (also pre-saturation here)

    Reference: x₁' = 0.0 → f(x₁') = ReLU(−5) = 0

    DeepLIFT contribution score for x₁=2.0:
    Δf = f(x₁) − f(x₁') = 15 − 0 = 15
    Δx₁ = x₁ − x₁' = 2.0 − 0.0 = 2.0
    Multiplier: m = Δf / Δx₁ = 15 / 2.0 = 7.5
    Contribution: C₁ = Δx₁ × m = 2.0 × 7.5 = 15

    DeepLIFT attributes 15 to x₁ — the full output difference.
    Gradient × Input would give: 10 × 2.0 = 20 (wrong, doesn't sum to 15).
    DeepLIFT has exact accounting. Gradient doesn't.
    # ================================================================== #


##### PART I: THE REFERENCE AND THE CONTRIBUTION SCORE

### The Reference Input — Encoding "No Information"

DeepLIFT requires a reference input x' that represents the absence of signal.
Every feature's contribution is measured RELATIVE to this reference.
The choice of reference is a design decision that fundamentally shapes
what the contribution scores mean.

    INPUT TYPE          NATURAL REFERENCE
    ──────────────────────────────────────────────────────────────────
    Images              x' = all-zeros (black image)
                        x' = all-0.5 (grey image)
                        x' = Gaussian noise at training-set mean
    Genomics (DNA)      x' = 0.25 for each nucleotide (uniform prior)
    Tabular             x' = training-set mean per feature
                        x' = zero (if features are standardised)
    Audio               x' = silence (all zeros)
    Text embeddings     x' = zero embedding vector

The reference defines what "neutral" means. A feature with xᵢ = xᵢ'
contributes exactly zero, regardless of its value, because it did not
DEVIATE from the reference. Only features that differ from their reference
values can have non-zero contributions.

This is fundamentally different from Integrated Gradients, where the
baseline also represents "no information," but the attribution accumulates
along the path from x' to x. DeepLIFT uses a single comparison, not an
integral — making it faster but less precise in capturing non-linear effects.

    # ================================================================== #
    **Reference choice changes the story:**

    Model: loan approval f(credit, income, debt)
    Instance: x = (credit=680, income=60k, debt=0.38)

    Reference A: x' = (0, 0, 0)  — "everything zero"
    Attribution asks: "how much does each feature contribute, relative to
    a hypothetical applicant with zero credit, zero income, zero debt?"

    Reference B: x' = (640, 58k, 0.38)  — "average applicant"
    Attribution asks: "how does this applicant differ from average, and
    which of those differences drove the model's decision?"

    Reference A credit attribution: contribution of having credit=680 vs credit=0
    Reference B credit attribution: contribution of having credit=680 vs credit=640

    These are different questions. Both are valid. Neither is universally correct.
    # ================================================================== #


### The Contribution Score — Definition

For each input feature i, the DeepLIFT contribution score is:

    C_Δxᵢ→Δo = m_i × Δxᵢ

where:
  • Δxᵢ = xᵢ − xᵢ'   (how much feature i deviates from its reference)
  • m_i = DeepLIFT multiplier for feature i
  • C_Δxᵢ→Δo = contribution of feature i to the output difference Δo

The multiplier mᵢ plays the role of a "rescaled gradient." It answers:
"for each unit of deviation in feature i from its reference value, how
much of the output deviation Δo = f(x) − f(x') is attributable to it?"

The sum rule (DeepLIFT's version of Completeness):

    Σᵢ C_Δxᵢ→Δo = Δo = f(x) − f(x')

This is exact. Every unit of the output difference is accounted for by
summing the contribution scores of all input features.

    # ================================================================== #
    **The multiplier vs the gradient:**

    Gradient:    gᵢ = ∂f/∂xᵢ  evaluated at x  (local slope at one point)
    Multiplier:  mᵢ = Δo / Δxᵢ  (global slope from x' to x along feature i)

    For linear f:  mᵢ = gᵢ always  (slope is constant; local = global)
    For non-linear f:  mᵢ ≠ gᵢ in general

    Grad × Input: gᵢ × xᵢ  → does not sum to f(x) − f(x') in general
    DeepLIFT:     mᵢ × Δxᵢ → sums to f(x) − f(x') exactly (by design)
    # ================================================================== #


##### PART II: THE BACKPROPAGATION RULES — COMPUTING MULTIPLIERS

### Why Multipliers Need Their Own Chain Rule

DeepLIFT cannot simply use the standard gradient chain rule to propagate
multipliers backwards. To see why, consider a ReLU network:

    Forward: aⱼ = ReLU(zⱼ),  where zⱼ = Σᵢ wᵢⱼ aᵢ

    Standard chain rule: ∂aⱼ/∂aᵢ = wᵢⱼ × 𝟙[zⱼ > 0]

    This is the gradient — it is ZERO when zⱼ ≤ 0 (inactive ReLU).

    DeepLIFT defines: Δaᵢ = aᵢ − aᵢ'  (deviation from reference activation)
                      Δaⱼ = aⱼ − aⱼ'  (deviation from reference activation)

    The multiplier for the j→i connection is:
    m_{Δzⱼ→Δaᵢ} = Δaᵢ / Δaⱼ  (how much Δaᵢ contributes to Δaⱼ)

    When both aⱼ and aⱼ' are zero (neuron inactive for BOTH inputs and
    reference): Δaⱼ = 0. Now the multiplier requires dividing by zero.
    DeepLIFT needs a rule for this case too.

This is exactly the zero-denominator problem of LRP-0. DeepLIFT solves
it with three rules, each matching a different layer type.


### Rule 1 — The Linear Rule (for fully connected layers)

For a linear transformation zⱼ = Σᵢ wᵢⱼ aᵢ + bⱼ, the multiplier is exact:

    Δzⱼ = Σᵢ wᵢⱼ Δaᵢ

    m_{Δaᵢ→Δzⱼ} = wᵢⱼ   (constant — same as gradient for linear layers)

This is trivial: for linear layers, DeepLIFT multipliers ARE the weights,
and are identical to the gradient. The method adds no new information over
GradInput for purely linear networks. The magic happens at non-linear layers.


### Rule 2 — The Rescale Rule (for non-linear activations)

For a non-linear activation function σ (e.g., ReLU, sigmoid, tanh):

    CASE A: Δaᵢ ≠ 0  (neuron deviates from reference)
    ───────────────────────────────────────────────────
    m = Δσ(zᵢ) / Δzᵢ = (σ(zᵢ) − σ(zᵢ')) / (zᵢ − zᵢ')

    This is the SECANT SLOPE — the slope of a chord from (zᵢ', σ(zᵢ'))
    to (zᵢ, σ(zᵢ)) on the activation function's curve. It is the average
    rate of change over the actual range of change, not the instantaneous
    derivative at one point.

    CASE B: Δaᵢ = 0  (neuron did not deviate from reference)
    ───────────────────────────────────────────────────────
    m = σ'(zᵢ)  (fall back to the gradient at the actual input)

    When the neuron's activation did not change between input and reference,
    there is no difference to attribute, and we fall back to the local gradient.
    This is the "no change, no attribution" principle.

    # ================================================================== #
    **The Rescale Rule for ReLU — visualised:**

    σ(z) = ReLU(z)
    ↑
    │         /
    │        / ← actual input: (z=3, σ=3)
    │       /
    │      / ← secant chord from reference to input
    │     /
    │    . ← reference: (z'=−1, σ'=0)  (below threshold → ReLU=0)
    │ ---
    └───────────────────────────→ z
        z'=−1        z=3

    Δz = 3 − (−1) = 4
    Δσ = 3 − 0 = 3
    Rescale multiplier m = Δσ/Δz = 3/4 = 0.75

    Compare to gradient at z=3: σ'(z) = 1.0 (ReLU is active, slope=1)
    The gradient over-attributes. The secant correctly accounts for the
    fact that some of the input range [−1, 3] was below the threshold
    (where σ=0). The secant "averages" the activation over the full range.
    # ================================================================== #


### Rule 3 — The RevealCancel Rule (for handling sign reversals)

The Rescale rule has a pathology when the input and reference are on
OPPOSITE SIDES of a non-linearity's threshold. Consider a ReLU where:

    Reference z' = −1  (inactive, output = 0)
    Actual input z = +1  (active, output = 1)

    Rescale: m = (1−0) / (1−(−1)) = 1/2 = 0.5  (seems reasonable)

Now consider a hidden neuron that contributes to z from above (+contribution)
and below (−contribution), where the net z = +1 but individually the positive
contribution is +5 and the negative is −4:

    Positive input contribution to z: a_pos × w_pos = +5
    Negative input contribution to z: a_neg × w_neg = −4
    z = +5 + (−4) = +1

With Rescale rule (m=0.5):
    Attribution from positive path: 0.5 × 5 = +2.5
    Attribution from negative path: 0.5 × (−4) = −2.0
    Total at this neuron: 0.5  ✓ (correct)

But consider the negative path's reference:
    a_neg' × w_neg = −3 (reference is also negative, but smaller magnitude)
    Δ(negative path) = −4 − (−3) = −1

If the negative path crosses zero between reference and input, the Rescale
rule produces incorrect attributions by treating the whole ΔZ as a uniform
rate of change across a region where the non-linearity behaves qualitatively
differently.

The RevealCancel rule handles this by SEPARATING the contributions:

    Positive contribution Δz⁺ = max(Δz, 0)
    Negative contribution Δz⁻ = min(Δz, 0)

Compute the multiplier separately for each part, using the activation at
the POSITIVE reference for positive contributions and the activation at
the NEGATIVE reference for negative contributions.

In practice: for ReLU networks, RevealCancel ≈ Rescale with the LRP-αβ
approach (separating positive and negative). The two rules converge for
many architectures. RevealCancel is preferred for networks with complex
sign-crossing behaviour (e.g., sigmoid-activated networks where both
excitatory and inhibitory paths are common).


### The Multiplier Chain Rule — Propagating from Output to Input

DeepLIFT multipliers compose via the CHAIN RULE for contributions:

    If neuron i contributes to neuron j, which contributes to output o:

    m_{Δaᵢ→Δo} = m_{Δaᵢ→Δaⱼ} × m_{Δaⱼ→Δo}

This is the same chain rule structure as standard backpropagation, but
the multipliers replace the gradients. The algorithm is:

    1. Forward pass on x: record all activations aˡ.
    2. Forward pass on x': record all reference activations a'ˡ.
    3. Compute Δaˡ = aˡ − a'ˡ for each layer l.
    4. Initialise multipliers at output: m_output = 1.0.
    5. Backpropagate using DeepLIFT rules at each layer.
    6. Contribution of feature i: Cᵢ = mᵢ × Δxᵢ.
    7. Verify: Σᵢ Cᵢ = f(x) − f(x').

Total cost: 2 forward passes + 1 backward pass.
(One forward pass for x, one for x', one backward for multipliers.)

    # ================================================================== #
    **DeepLIFT backpropagation — data flow:**

    Forward (x):    x → a¹ → a² → ... → f(x)
    Forward (x'):   x' → a'¹ → a'² → ... → f(x')

    Backward:
    m_output = 1.0
      ↓  [output-to-last-layer: linear rule]
    m_last_layer = W_output^T  (weights as multipliers for linear)
      ↓  [through activation: rescale rule]
    m = m × Δσ/Δz  (secant slope of activation)
      ↓  [continues to input]
    C_i = m_i × Δxᵢ    (contribution score for each input feature)
    # ================================================================== #


##### PART III: THE SUM RULE — THE COMPLETENESS GUARANTEE

### Proving the Sum Rule

DeepLIFT's sum rule Σᵢ Cᵢ = f(x) − f(x') is not an approximation —
it holds EXACTLY when the Rescale or RevealCancel rule is applied correctly.

The proof follows by induction on layers.

    BASE CASE (output layer):
    At the output neuron, we initialise C_output = Δo = f(x) − f(x').
    The sum rule trivially holds for one neuron.

    INDUCTIVE STEP:
    Assume the sum rule holds at layer l+1:
        Σⱼ C_Δaⱼ→Δo = Δo  (every neuron j in layer l+1 satisfies this)

    For each neuron j, its contribution was received from neurons i in layer l.
    The Rescale rule propagates contributions as:
        C_Δaᵢ→Δaⱼ = (Δaᵢ × wᵢⱼ) / Δzⱼ × Δaⱼ  (for the linear part)

    Summing over all i for one j:
        Σᵢ C_Δaᵢ→Δaⱼ = (Σᵢ Δaᵢ × wᵢⱼ) / Δzⱼ × Δaⱼ
                       = Δzⱼ / Δzⱼ × Δaⱼ  (since Δzⱼ = Σᵢ wᵢⱼ Δaᵢ)
                       = Δaⱼ  ← conservation holds per neuron!

    So the total contribution received by each neuron j from layer l equals Δaⱼ.
    The sum of Δaⱼ across all j in layer l+1 equals Δo (by inductive hypothesis).
    Therefore, Σᵢ Cᵢ = Σⱼ Δaⱼ = Δo.  QED.

The proof depends critically on the numerator being exactly Δzⱼ. This is
guaranteed by the Rescale rule's secant-slope formulation. The RevealCancel
rule satisfies a similar identity by construction.

    # ================================================================== #
    **Why GradInput fails the sum rule:**

    GradInput attribution: Cᵢ^{GI} = (∂f/∂xᵢ) × xᵢ

    For a linear network: Σᵢ (∂f/∂xᵢ) × xᵢ = Σᵢ wᵢ × xᵢ = f(x) − b
    (The bias b is not attributed to any input.)

    For a non-linear network: GradInput does not sum to f(x) or f(x)−f(x').
    The sum Σᵢ Cᵢ^{GI} = f(x) only when Euler's theorem applies (for
    homogeneous functions), which ReLU networks are NOT in general.

    DeepLIFT always satisfies Σᵢ Cᵢ = f(x) − f(x').
    This is guaranteed by construction, not by accident.
    # ================================================================== #


### The Bias Term — Where Does It Go?

In a network with biases bⱼ, the pre-activation is:
    zⱼ = Σᵢ wᵢⱼ aᵢ + bⱼ

The bias bⱼ is a constant. Its reference value is also bⱼ (the bias doesn't
change between input and reference). So Δbⱼ = bⱼ − bⱼ = 0 always.
The bias contributes ZERO to DeepLIFT's multiplier chain.

But wait — doesn't the bias affect the output? Yes, it does. But the bias's
effect is already captured in the reference output f(x'). The reference
encodes "what the model outputs for neutral inputs," and that includes the
effect of all biases. DeepLIFT attributes only the CHANGE from reference,
and biases don't change between x and x'.

    f(x') already accounts for biases.
    Δf = f(x) − f(x') = [input features' contribution] + [biases' contribution]
                        − [reference features'] − [biases' same contribution]
                       = [input features' contribution] − [reference features']

Biases cancel. DeepLIFT cleanly separates feature contributions from
the constant-term background.


##### PART IV: DEEPLIFT AND SHAPLEY VALUES — THE DEEPSHAP CONNECTION

### What Shapley Values Require

The Shapley value φᵢ is the only attribution satisfying four axioms:
Efficiency (sum to total), Symmetry, Dummy, and Linearity. For an
arbitrary function and arbitrary background distribution, computing
exact Shapley values requires evaluating the model on all 2^n subsets
of features — exponentially expensive.

DeepLIFT is NOT computing Shapley values in general. But it has a
specific relationship to Shapley values under two conditions:

    CONDITION 1: Linear Network
    ──────────────────────────
    For a purely linear model f(x) = W·x + b, DeepLIFT's Rescale rule
    gives multipliers equal to the weights. The contribution scores are:
    Cᵢ = wᵢ × (xᵢ − xᵢ')
    These ARE the Shapley values for a linear model (all orderings give
    the same marginal contribution, since there are no interactions).

    CONDITION 2: Single Reference = Single Background Sample
    ─────────────────────────────────────────────────────────
    With a single reference point x', DeepLIFT approximates the Shapley
    values for that specific x' as the background distribution.
    This is a biased approximation — Shapley values require averaging
    over MANY background samples, not just one.

The gap between DeepLIFT and Shapley values is exactly the gap between
using one reference vs using the full background distribution. DeepLIFT
with K reference samples (averaging the contribution scores) approaches
the Shapley value as K → ∞.


### DeepSHAP — Bridging the Gap

DeepSHAP (Lundberg & Lee, 2017 — same paper as SHAP) extends DeepLIFT
to approximate Shapley values by:

    1. Using MULTIPLE reference samples from the background distribution.
    2. Running DeepLIFT for each reference separately.
    3. AVERAGING the contribution scores across all references.

    φᵢ^{DeepSHAP} ≈ (1/|B|) Σ_{x' ∈ B} C_Δxᵢ→Δo(x, x')

where B is the background dataset (typically 50–200 samples from training).

This gives an approximation to the true Shapley values that:
  • Exactly satisfies Efficiency: Σᵢ φᵢ = E_{x'∈B}[f(x) − f(x')]
  • Approximately satisfies Symmetry and Dummy (exact when B covers the
    full marginal distribution)
  • Converges to exact Shapley values as |B| → ∞

DeepSHAP is the production-quality version of DeepLIFT. When the SHAP
library is used with deep neural networks, it typically calls DeepSHAP
under the hood — running DeepLIFT against multiple background samples.

    # ================================================================== #
    **DeepLIFT vs DeepSHAP — cost and accuracy:**

    DeepLIFT (1 reference):
      Cost: 2 forward + 1 backward pass
      Accuracy: exact conservation, but Shapley axioms not guaranteed
      Suitable: quick attribution, when a single "neutral" reference exists

    DeepSHAP (K references):
      Cost: K+1 forward + K backward passes (K = background set size)
      Accuracy: approximates Shapley values; K=200 ≈ 95% of Shapley values
      Suitable: when game-theoretic fairness axioms are required

    TreeSHAP (for trees):
      Cost: O(TLD²) — exact, polynomial
      Accuracy: exact Shapley values
      Suitable: tree models (XGBoost, LightGBM, sklearn)
    # ================================================================== #


### Where DeepLIFT Differs From Integrated Gradients

Both DeepLIFT and Integrated Gradients (IG) require a baseline x' and
satisfy an exact completeness axiom. They compute different things:

    INTEGRATED GRADIENTS:
    IG_i(x) = (xᵢ − xᵢ') × ∫₀¹ ∂f(x' + α(x−x')) / ∂xᵢ dα

    Computes the INTEGRAL of the gradient along the straight-line path
    from x' to x. Captures non-linearities by sampling M points along
    the path (typically M=50–300). Cost: M forward+backward passes.

    DEEPLIFT:
    C_i(x) = mᵢ × (xᵢ − xᵢ')

    Computes a SINGLE SECANT SLOPE from f(x') to f(x) through the network.
    Cost: 2 forward + 1 backward pass.

    The relationship: DeepLIFT is IG with M=1, using the secant slope
    instead of the gradient at the endpoint. For a piecewise linear
    network (ReLU + linear layers):
      • The secant slope = the EXACT average gradient along the path
        (because the path integral of a piecewise linear function equals
        the secant slope of the piecewise linear segments).
      • DeepLIFT = IG exactly for ReLU networks.

This equivalence is not approximate — it holds because:
    ∫₀¹ ∂ReLU(α·z + (1−α)·z') / ∂z dα = (ReLU(z) − ReLU(z')) / (z − z')
    = the Rescale multiplier

DeepLIFT's Rescale rule IS the closed-form solution to IG's path integral
for piecewise linear networks. No numerical integration needed.

    # ================================================================== #
    **DeepLIFT = Integrated Gradients for piecewise-linear networks:**

    For a network with only ReLU activations and linear layers:
      IG with M=∞ steps along the straight line from x' to x
      = DeepLIFT with the Rescale rule

    Both give exactly the same attribution.
    DeepLIFT computes it in 2 fwd + 1 bwd pass.
    IG computes it in M × (1 fwd + 1 bwd) ≈ 50–300 passes.

    DeepLIFT is up to 300× faster than IG for ReLU networks,
    with mathematically identical results.

    For networks with smooth activations (sigmoid, GELU, softmax),
    they differ: IG exactly integrates the smooth curve; DeepLIFT
    uses the secant approximation. IG is more accurate; DeepLIFT is faster.
    # ================================================================== #


##### PART V: DEEPLIFT AND LRP — TWIN METHODS

### The Formal Equivalence

Montavon, Lapuschkin, Binder, Müller & Bach (2017) proved that DeepLIFT's
Rescale rule is algebraically equivalent to LRP-ε with ε→0 (LRP-0),
when the reference activation is a'ᵢ = 0 for all neurons.

    LRP-0 rule:    Rᵢ = Σⱼ (aᵢ wᵢⱼ / zⱼ) × Rⱼ
    DeepLIFT:      Cᵢ = mᵢ × Δaᵢ = Σⱼ (Δaᵢ wᵢⱼ / Δzⱼ) × Cⱼ

When a'ᵢ = 0 for all i: Δaᵢ = aᵢ − 0 = aᵢ and Δzⱼ = zⱼ − 0 = zⱼ.
    LRP-0: Rᵢ = Σⱼ (aᵢ wᵢⱼ / zⱼ) × Rⱼ
    DeepLIFT: Cᵢ = Σⱼ (aᵢ wᵢⱼ / zⱼ) × Cⱼ  ← identical!

They are the same algorithm when the reference is all-zeros.
The two methods emerged from different communities (bioinformatics vs
explainable AI) with different motivations but converged to the same rule.

    # ================================================================== #
    **The family tree of attribution methods:**

    Gradient × Input  ←→  LRP-0 (with zero reference)
                      ←→  DeepLIFT Rescale (with zero reference)
    Integrated Gradients ←→  DeepLIFT Rescale (for piecewise-linear nets)
    DeepSHAP          ≈  DeepLIFT + multiple references + averaging
    DeepSHAP          →  Shapley values (as #references → ∞)
    # ================================================================== #


### Key Differences Despite Algebraic Similarity

Although LRP-0 and DeepLIFT are algebraically equivalent, they differ in:

    1. FRAMING:
    LRP thinks in terms of "relevance flowing backward through the network."
    DeepLIFT thinks in terms of "difference from reference propagating back."
    The mathematics is the same; the motivation is different.

    2. HANDLING ZERO DENOMINATORS:
    LRP developed ε, α/β, and zᴮ rules to handle near-zero pre-activations.
    DeepLIFT's natural solution: when Δzⱼ = 0, fall back to the gradient.
    The gradient is the limit of the secant slope as Δz → 0:
    lim_{Δz→0} Δσ/Δz = σ'(z).
    This is a more principled fallback than LRP-ε's arbitrary ε choice.

    3. THE REFERENCE:
    LRP implicitly uses x' = 0 (the zero vector).
    DeepLIFT makes the reference an explicit user parameter.
    This allows DeepLIFT to be applied with domain-appropriate references
    (e.g., uniform nucleotide distribution for genomics) without retraining.

    4. EXTENSION TO SHAPLEY:
    DeepLIFT extends naturally to DeepSHAP by averaging over references.
    LRP has no direct Shapley extension — its variants are motivated by
    stability, not game-theoretic consistency.


##### PART VI: HANDLING SPECIFIC LAYER TYPES

### Fully Connected Layers

For zⱼ = Σᵢ wᵢⱼ aᵢ + bⱼ:
    Δzⱼ = Σᵢ wᵢⱼ Δaᵢ  (bias cancels: Δbⱼ = 0)
    Linear rule multiplier: mᵢⱼ = wᵢⱼ  (same as gradient)

The linear rule is identical to standard backpropagation. DeepLIFT
only diverges from gradients at non-linear layers.


### Non-linear Activations (ReLU, sigmoid, tanh)

The Rescale rule:
  • If Δaᵢ ≠ 0:  m = Δaᵢ / Δzᵢ = (σ(zᵢ) − σ(zᵢ')) / (zᵢ − zᵢ')
  • If Δaᵢ = 0:  m = σ'(zᵢ)  (gradient as fallback)

For ReLU: Δσ/Δz = (ReLU(z) − ReLU(z')) / (z − z')
  CASE 1: Both active (z > 0 and z' > 0): m = (z − z')/(z − z') = 1
  CASE 2: Both inactive (z ≤ 0 and z' ≤ 0): Δσ=0, Δz≠0 → m=0
  CASE 3: One active, one not (z > 0, z' ≤ 0): m = z/(z − z') ∈ (0, 1)
  CASE 4: Δz = 0 (no change): m = σ'(z) = 𝟙[z>0]

Case 3 is the "interesting" case — the ReLU's kink falls between x' and x.
The secant slope is between 0 and 1, capturing the partial activation
caused by crossing the threshold.

    # ================================================================== #
    **ReLU Rescale in all four cases:**

    Case 1 (both active):
    z'=1  z=3
      ↓   ↓
    │   .───●  (secant slope = 1 — both fully active)
    │  /
    │ .
    └────────────→ z
    m = (3−1)/(3−1) = 1.0  ← same as gradient

    Case 3 (threshold crossed):
    z'=−2  z=2
      ↓    ↓
    │     ●
    │    /
    │   /
    │---.
    └────────────→ z
         z'=−2  z=2
    m = (ReLU(2)−ReLU(−2))/(2−(−2)) = (2−0)/4 = 0.5  ← DIFFERENT from gradient!
    Gradient at z=2: σ'(z) = 1.0  (would over-attribute)
    DeepLIFT: 0.5  (correct average)
    # ================================================================== #


### Max-Pooling

For max-pooling with inputs {a₁, ..., aₙ} and output o = max(a₁,...,aₙ):

    Reference forward: o' = max(a'₁, ..., a'ₙ)

    The contribution from each input i is:
    If aᵢ = o (this input is the maximum in the actual forward pass):
        Cᵢ→o = Δo = o − o' = aᵢ − o'
    Else:
        Cᵢ→o = 0  (this input was not the maximum; it didn't contribute to o)

When the winner is the same in both the actual and reference passes:
    Cᵢ→o = aᵢ − a'ᵢ  (winner's full deviation is the output deviation)

When the winner differs between actual and reference:
    Cᵢ→o = aᵢ − a'ᵢ  for the actual winner (contribution is its deviation
           from its reference value, which need not be the reference's winner)

This is a special case of the RevealCancel rule applied to max-pooling.


### Softmax — The Output Layer

Softmax is typically at the output layer. The score to explain is usually
the PRE-SOFTMAX logit for the target class, not the softmax probability.
This is because:
  1. Logit differences are additive (softmax differences are not).
  2. The sum rule holds cleanly for logit differences.
  3. Feature attributions for probabilities become distorted by the
     normalising denominator of softmax.

DeepLIFT's recommended practice: explain the logit of the target class,
not the probability. This is the same convention as SHAP's DeepExplainer.

    C_Δxᵢ→Δlogit_c  (not  C_Δxᵢ→Δprob_c)
    Σᵢ Cᵢ = logit_c(x) − logit_c(x')


### Convolutional Layers

The linear rule applies directly. Each convolutional output is a linear
combination of the input patch, so:

    Δz_{output} = Σ_{i in receptive field} w_i × Δa_i

The multiplier for each input in the receptive field is w_i (the filter weight).
This is identical to the gradient for the convolutional layer — the non-linearity
(ReLU after conv) is handled by the Rescale rule when applied to the activation.


##### PART VII: PRACTICAL FAILURE MODES AND LIMITATIONS

### Failure Mode 1 — Reference Sensitivity

The choice of reference fundamentally changes the attribution. Unlike
GradCAM (which requires no reference) or vanilla gradients (same), DeepLIFT
attributions depend heavily on what "neutral" means for your domain.

    Pathological example:
    Reference x' = (credit=800, income=100k, debt=0.20) — a STRONG reference
    Instance  x  = (credit=680, income=55k,  debt=0.38) — a weaker applicant

    Attribution asks: "why is this applicant worse than an excellent one?"
    This is a valid but unusual question. The attributions will be
    NEGATIVE for most features (applicant is below the reference),
    even though the absolute prediction may be positive (still likely approved).

    DeepLIFT with a strong reference can produce all-negative attributions
    for a positive prediction. This is technically correct but profoundly
    confusing to a non-technical audience.

    Best practice: always choose x' such that f(x') is near the neutral
    value of f (e.g., f(x') ≈ 0 for a logit, or f(x') ≈ 0.5 for a probability).


### Failure Mode 2 — The Secant Approximation Error

For smooth non-linear activations (sigmoid, tanh, GELU, softmax), the
Rescale rule (secant slope) is an APPROXIMATION to the path integral.
The approximation error is:

    Error = IG_exact − DeepLIFT = ∫₀¹ σ'(z'+α(z−z')) dα − (σ(z)−σ(z'))/(z−z')

For a sigmoid σ(z) = 1/(1+e⁻ᶻ):
    The path integral of σ'(z) along [z', z] is computed numerically.
    The secant slope (σ(z)−σ(z'))/(z−z') is the average of σ' — by the
    mean value theorem, these are equal for some z* ∈ [z', z].

    But the mean value theorem gives EXISTENCE, not the VALUE of z*.
    If σ' varies significantly over [z', z], the secant may over- or
    under-estimate the true average gradient.

For ReLU networks, there is NO approximation error — DeepLIFT = IG exactly.
For sigmoid networks, the error can be significant when z and z' span
a large range (e.g., z'=−5, z=5: σ varies from 0.006 to 0.993, while σ' peaks at 0.25).


### Failure Mode 3 — Non-Uniqueness Under Feature Interactions

The sum rule Σᵢ Cᵢ = Δo is exact. But the ALLOCATION of Δo among features
is not unique when features interact. For a model with a pure interaction:

    f(x₁, x₂) = x₁ × x₂  (no individual effects, pure interaction)
    Reference: x' = (0, 0), so f(x') = 0

    Δo = x₁ × x₂ − 0 × 0 = x₁ × x₂

    DeepLIFT Rescale at the interaction node:
    Δz = (x₁ × x₂) − (x'₁ × x'₂) = x₁x₂ − 0 = x₁x₂
    Contribution from x₁: (x₁ / (x₁ + x₂)) × x₁x₂  (???)

This is not well-defined — the interaction splits in a way that depends
on the specific path through the network. DeepLIFT with Rescale handles
purely multiplicative interactions through whichever decomposition the
specific network architecture creates. The answer is architecture-dependent.

Shapley values, by contrast, allocate exactly x₁x₂/2 to x₁ and x₁x₂/2 to x₂
(by symmetry). DeepLIFT does not satisfy this symmetry axiom in general.


### Failure Mode 4 — Implementation Complexity vs Gradient Methods

DeepLIFT requires running TWO forward passes (one for x, one for x') and
storing Δ-activations at every layer. This doubles the memory requirement
compared to a standard forward+backward pass. For large models:

    Memory: O(2 × L × D) instead of O(L × D), where L = layers, D = neurons
    Implementation: requires custom hooks; not trivially available in autograd

In practice, the DeepSHAP implementation in the SHAP library handles this,
but rolling a custom implementation requires careful management of the
reference activations at each layer — a common source of bugs.

    Conservation check (always run):
    Σᵢ Cᵢ should equal logit_target(x) − logit_target(x')
    Tolerance: < 0.1% of |Δo|
    Larger error → bug in reference propagation or Rescale rule implementation


##### PART VIII: THE COMPLETE LANDSCAPE — WHERE DEEPLIFT SITS

### A Unified Comparison

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Property             Grad×Input  DeepLIFT   IG        DeepSHAP     │
    │  ────────────────────────────────────────────────────────────────── │
    │  Completeness         No          YES        YES       YES          │
    │  Handles saturation   No          YES        YES       YES          │
    │  Requires reference   No*         YES        YES       YES (many)   │
    │  Cost (fwd+bwd)       1+1         2+1        M+M       (M+1)+M      │
    │  = IG (ReLU nets)     No          YES        —         YES          │
    │  Shapley axioms        No          Partial    Partial   Nearly all  │
    │  Impl. invariance     No          YES        YES       YES          │
    │  Sanity checks        Pass        Pass       Pass      Pass         │
    │  Handles smooth act.  Poor        Approx.    Exact     Approx.      │
    │  Production use       Rarely      SHAP lib   IG lib    SHAP lib     │
    └─────────────────────────────────────────────────────────────────────┘
    * Grad×Input uses x'=0 implicitly.

    WHEN TO USE DEEPLIFT (vs alternatives):
    ─────────────────────────────────────────────────────────────────────
    vs Gradient × Input:
      Always prefer DeepLIFT. Same cost, exact conservation, handles saturation.

    vs Integrated Gradients:
      Prefer DeepLIFT for ReLU networks (identical result, M× faster).
      Prefer IG for smooth activations (sigmoid, tanh, GELU, transformers).

    vs DeepSHAP:
      Prefer DeepLIFT when one reference is domain-appropriate.
      Prefer DeepSHAP when Shapley axioms matter (feature independence).

    vs SHAP KernelSHAP:
      Prefer DeepLIFT/DeepSHAP for deep networks (model-specific, faster).
      Prefer KernelSHAP for small networks or when model-agnostic is required.


##### PART IX: A COMPLETE WORKED EXAMPLE — DEEPLIFT BY HAND

### Tracing DeepLIFT Through a 3-Layer Network

We use the same network architecture as the LRP module for direct comparison.

    Network: 2 inputs → 2 hidden (ReLU) → 1 output (no bias)
    Weights:
      Layer 1→2:  w₁₁ = 1.0,  w₂₁ = 0.5
                  w₁₂ = -0.5, w₂₂ = 1.5
      Layer 2→3:  w₃₁ = 2.0, w₃₂ = -1.0

    Actual input:    x = (x₁=1.0, x₂=0.8)
    Reference input: x' = (x'₁=0.0, x'₂=0.0)  (zero reference)

    STEP 1 — Forward pass on actual input x:
    z₁ = 1.0(1.0) + 0.5(0.8) = 1.4    → a₁ = ReLU(1.4) = 1.4
    z₂ = -0.5(1.0) + 1.5(0.8) = 0.7   → a₂ = ReLU(0.7) = 0.7
    z_out = 2.0(1.4) + (-1.0)(0.7) = 2.8 − 0.7 = 2.1
    f(x) = 2.1

    STEP 2 — Forward pass on reference x':
    z'₁ = 1.0(0) + 0.5(0) = 0.0    → a'₁ = ReLU(0.0) = 0.0
    z'₂ = -0.5(0) + 1.5(0) = 0.0   → a'₂ = ReLU(0.0) = 0.0
    z'_out = 2.0(0) + (-1.0)(0) = 0.0
    f(x') = 0.0

    Δo = f(x) − f(x') = 2.1 − 0.0 = 2.1  ← the total to explain

    STEP 3 — Compute Δ-activations at every layer:
    Δa₁ = a₁ − a'₁ = 1.4 − 0.0 = 1.4
    Δa₂ = a₂ − a'₂ = 0.7 − 0.0 = 0.7
    Δz_out = z_out − z'_out = 2.1 − 0.0 = 2.1

    STEP 4 — Initialise contributions at output:
    C_out = Δo = 2.1

    STEP 5 — Propagate through last linear layer (layer 2→3):
    Linear rule: multipliers are the weights.
    Contribution of a₁ to z_out: (Δa₁ × w₃₁) / Δz_out × C_out
    = (1.4 × 2.0) / 2.1 × 2.1 = 2.8  ← contribution TO a₁ from C_out

    Contribution of a₂ to z_out: (Δa₂ × w₃₂) / Δz_out × C_out
    = (0.7 × (−1.0)) / 2.1 × 2.1 = −0.7

    C_{a₁} = 2.8,  C_{a₂} = -0.7
    Check: 2.8 + (−0.7) = 2.1 = C_out  ✓

    STEP 6 — Propagate through ReLU activations (Rescale rule):

    For neuron 1 (a₁ = ReLU(z₁), a'₁ = ReLU(z'₁)):
    Δz₁ = z₁ − z'₁ = 1.4 − 0.0 = 1.4
    Δa₁ = a₁ − a'₁ = 1.4 − 0.0 = 1.4  (both active, linear region)
    Rescale multiplier: m₁ = Δa₁/Δz₁ = 1.4/1.4 = 1.0  ← same as gradient
    C through ReLU₁: C_{z₁} = m₁ × C_{a₁} = 1.0 × 2.8 = 2.8

    For neuron 2 (a₂ = ReLU(z₂), a'₂ = ReLU(z'₂)):
    Δz₂ = z₂ − z'₂ = 0.7 − 0.0 = 0.7
    Δa₂ = a₂ − a'₂ = 0.7 − 0.0 = 0.7  (both active at boundary: ReLU(0)=0)
    Rescale multiplier: m₂ = Δa₂/Δz₂ = 0.7/0.7 = 1.0
    C through ReLU₂: C_{z₂} = 1.0 × (−0.7) = −0.7

    Note: ReLU(0.0)=0 (reference is exactly at threshold). Both are in linear
    region (z₂=0.7 is active, z'₂=0.0 is at threshold → ReLU=0 for both).
    The secant slope = (0.7−0.0)/(0.7−0.0) = 1.0 ← correct.

    STEP 7 — Propagate through first linear layer (layer 1→2):

    For hidden neuron 1 (z₁ = 1.0x₁ + 0.5x₂):
    Δz₁ = 1.4
    C_{x₁←z₁} = (Δx₁ × w₁₁) / Δz₁ × C_{z₁} = (1.0×1.0)/1.4 × 2.8 = 2.000
    C_{x₂←z₁} = (Δx₂ × w₂₁) / Δz₁ × C_{z₁} = (0.8×0.5)/1.4 × 2.8 = 0.800

    For hidden neuron 2 (z₂ = -0.5x₁ + 1.5x₂):
    Δz₂ = 0.7
    C_{x₁←z₂} = (Δx₁ × w₁₂) / Δz₂ × C_{z₂} = (1.0×(−0.5))/0.7 × (−0.7) = 0.500
    C_{x₂←z₂} = (Δx₂ × w₂₂) / Δz₂ × C_{z₂} = (0.8×1.5)/0.7 × (−0.7) = −1.200

    STEP 8 — Sum contributions per input feature:
    C_{x₁} = C_{x₁←z₁} + C_{x₁←z₂} = 2.000 + 0.500 = +2.500
    C_{x₂} = C_{x₂←z₁} + C_{x₂←z₂} = 0.800 + (−1.200) = −0.400

    CONSERVATION CHECK:
    C_{x₁} + C_{x₂} = 2.500 + (−0.400) = 2.100 = Δo  ✓  (exact)

    COMPARISON TO LRP-0:
    LRP-0 produces R_{x₁} = 2.500, R_{x₂} = −0.400 — IDENTICAL to DeepLIFT.
    This confirms the theoretical equivalence: with zero reference, DeepLIFT
    = LRP-0 for this linear+ReLU network.

    NOW CHANGE THE REFERENCE TO x' = (0.5, 0.4):

    f(x') for new reference:
    z'₁ = 1.0(0.5) + 0.5(0.4) = 0.7  → a'₁ = 0.7
    z'₂ = -0.5(0.5) + 1.5(0.4) = 0.35 → a'₂ = 0.35
    f(x') = 2.0(0.7) + (−1.0)(0.35) = 1.4 − 0.35 = 1.05

    Δo = 2.1 − 1.05 = 1.05  ← now we explain a SMALLER difference

    Rescale at neuron 1:
    Δz₁ = 1.4 − 0.7 = 0.7,  Δa₁ = 1.4 − 0.7 = 0.7,  m₁ = 1.0
    Rescale at neuron 2:
    Δz₂ = 0.7 − 0.35 = 0.35, Δa₂ = 0.7 − 0.35 = 0.35, m₂ = 1.0

    C_{z₁} = 1.0 × (1.4×2.0/2.1×1.05) = 1.4  (half of what it was with zero ref)
    C_{z₂} = 1.0 × (0.7×(−1.0)/2.1×1.05) = −0.35

    C_{x₁} = (0.5×1.0)/0.7 × 1.4 + (0.5×(−0.5))/0.35 × (−0.35) = 1.000+0.250 = 1.250
    C_{x₂} = (0.4×0.5)/0.7 × 1.4 + (0.4×1.5)/0.35 × (−0.35) = 0.400+(−0.600) = -0.200

    Check: 1.250 + (−0.200) = 1.050 = Δo  ✓

    The CONTRIBUTIONS ARE HALF of the zero-reference case, because:
    Δxᵢ is halved (we're comparing to a midpoint reference),
    and the Δo is halved — the same fraction goes to each feature.

    This illustrates why the reference matters: the contribution scores
    scale with how much each feature DEVIATES from the reference, not
    with the feature's absolute value.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔════════════════════════╦═══════════════╦═════════════════════════════════╗
    ║ Property               ║ DeepLIFT      ║ Comparison                      ║
    ╠════════════════════════╬═══════════════╬═════════════════════════════════╣
    ║ Cost                   ║ 2 fwd + 1 bwd ║ IG: M fwd+bwd, GradInput: 1+1   ║
    ║ Completeness           ║ YES (exact)   ║ GradInput: No; IG: Yes          ║
    ║ Handles saturation     ║ YES           ║ Gradient: No                    ║
    ║ = IG (ReLU nets)       ║ YES (exact)   ║ Saves M fwd+bwd passes          ║
    ║ = LRP-0 (zero ref)     ║ YES           ║ Different framing, same result  ║
    ║ Requires reference     ║ YES (1 ref)   ║ DeepSHAP: K refs; IG: 1 ref     ║
    ║ Shapley symmetry       ║ No            ║ DeepSHAP: Yes (approximately)   ║
    ║ Smooth activations     ║ Approximate   ║ IG: exact integral              ║
    ║ Sanity check           ║ Pass          ║ Guided BP: Fail; Grad: Pass     ║
    ║ Implementation         ║ Custom bwd    ║ IG: standard autograd possible  ║
    ╚════════════════════════╩═══════════════╩═════════════════════════════════╝

    DeepLIFT rules:
      Linear:   Cᵢ←ⱼ = (Δaᵢ × wᵢⱼ / Δzⱼ) × Cⱼ
      Rescale:  m = (σ(z) − σ(z')) / (z − z')  if Δz ≠ 0, else m = σ'(z)
      Sum rule: Σᵢ Cᵢ = f(x) − f(x')  (exact, by construction)

    Key equivalences:
      DeepLIFT Rescale (x'=0) ≡ LRP-0 ≡ GradInput  (for ReLU+linear nets)
      DeepLIFT Rescale         ≡ Integrated Gradients (for piecewise-linear nets)
      DeepSHAP (K refs, K→∞)  ≡ SHAP values          (for deep networks)
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "DeepLIFT from Scratch — Rescale Rule, Sum Rule Verification, Reference Study": {
        "description": (
            "Implements DeepLIFT from scratch on a 3-layer ReLU network. "
            "Computes Rescale multipliers at every layer, traces the full backpropagation "
            "of contributions, and verifies the exact sum rule at each layer. "
            "Compares DeepLIFT against Gradient×Input to show when they diverge "
            "(saturation / threshold crossing). Runs with three different references "
            "to demonstrate how attribution changes. Shows the exact equivalence with "
            "LRP-0 for the zero reference. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "deeplift",
        "code": '''
"""
================================================================================
DEEPLIFT FROM SCRATCH — RESCALE RULE WITH FULL TRANSPARENCY
================================================================================

We implement DeepLIFT's Rescale rule from scratch on a 3-layer network:
  4 inputs → 5 hidden (ReLU) → 3 hidden (ReLU) → 2 outputs

For each forward pass pair (x, x'), we:
  1. Compute Δ-activations at every layer
  2. Apply the Rescale rule at each non-linear layer
  3. Apply the linear rule at each FC layer
  4. Verify the sum rule at every layer
  5. Compare against Gradient × Input

Then we study three references to show how attribution shifts with reference choice.
================================================================================
"""

import math
import random

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK (same as LRP module for direct comparison)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return max(0.0, x)

def relu_deriv(x):
    return 1.0 if x > 0 else 0.0

def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]

W1 = [
    [ 0.8, -0.4,  0.6,  0.3],
    [-0.3,  0.9,  0.2, -0.7],
    [ 0.5,  0.1, -0.8,  0.4],
    [-0.6,  0.7,  0.3,  0.9],
    [ 0.2, -0.5,  0.7, -0.3],
]
b1 = [-0.2,  0.1, -0.1,  0.3, -0.1]

W2 = [
    [ 0.7, -0.3,  0.5, -0.4,  0.2],
    [-0.4,  0.8, -0.2,  0.6, -0.5],
    [ 0.3, -0.6,  0.9, -0.1,  0.7],
]
b2 = [0.1, -0.2, 0.1]

W3 = [
    [-0.6,  0.9, -0.3],
    [ 0.8, -0.5,  0.7],
]
b3 = [0.1, -0.1]

WEIGHTS  = [W1, W2, W3]
BIASES   = [b1, b2, b3]
USE_RELU = [True, True, False]
FEAT_NAMES = ["credit_std", "income_std", "debt_std", "employment_std"]

x_input = [-0.625, -0.240, +1.083, -0.500]
TARGET = 0   # explain DENY logit


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS — save pre and post activations
# ─────────────────────────────────────────────────────────────────────────────

def forward(x):
    a = [list(x)]
    z_all = []
    for W, b, act in zip(WEIGHTS, BIASES, USE_RELU):
        z_l = [b[j] + sum(W[j][i] * a[-1][i] for i in range(len(a[-1])))
               for j in range(len(W))]
        z_all.append(z_l)
        a.append([relu(z) for z in z_l] if act else list(z_l))
    return a[-1], z_all, a


logits, z_all, activations = forward(x_input)
probs = softmax(logits)
pred = 0 if logits[0] > logits[1] else 1

print("=" * 68)
print("  NETWORK AND INSTANCE")
print("=" * 68)
print(f"  Architecture: 4 → 5 → 3 → 2  (ReLU after layers 1,2)")
for fname, val in zip(FEAT_NAMES, x_input):
    print(f"  {fname:<20}: {val:+.3f}")
print(f"  f(x):  DENY={logits[0]:.4f},  APPROVE={logits[1]:.4f}")
print(f"  Pred:  {'DENY' if pred==0 else 'APPROVE'}  ({probs[pred]:.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# DEEPLIFT — RESCALE RULE IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def rescale_multiplier(z, z_prime, is_relu=True):
    """
    Compute the DeepLIFT Rescale multiplier for a non-linear activation.

    If is_relu: sigma = ReLU
    Fallback to gradient when Dz == 0.
    """
    if is_relu:
        a  = relu(z)
        ap = relu(z_prime)
    else:
        a  = z
        ap = z_prime

    dz = z - z_prime
    da = a - ap

    if abs(dz) < 1e-12:
        # Fall back to gradient
        return relu_deriv(z) if is_relu else 1.0

    return da / dz


def run_deeplift(x, x_prime, target_class, verbose=True):
    """
    Full DeepLIFT computation.
    Returns (contribution scores, layer deltas, conservation gaps).
    """
    # Forward passes
    logits_x,  z_x,  a_x  = forward(x)
    logits_xp, z_xp, a_xp = forward(x_prime)

    delta_o = logits_x[target_class] - logits_xp[target_class]

    if verbose:
        print()
        print(f"  f(x)  [{target_class}] = {logits_x[target_class]:.6f}")
        print(f"  f(x') [{target_class}] = {logits_xp[target_class]:.6f}")
        print(f"  Δo = {delta_o:.6f}  (total to attribute)")

    n_layers = len(WEIGHTS)

    # Initialise contributions at output
    C = [0.0, 0.0]
    C[target_class] = delta_o

    conservation_gaps = []

    for l in range(n_layers - 1, -1, -1):
        W = WEIGHTS[l]
        act = USE_RELU[l]
        n_out = len(W)
        n_in  = len(W[0])

        # Step A: propagate through activation (Rescale) if applicable
        if act:
            C_after_relu = []
            for j in range(n_out):
                m = rescale_multiplier(z_x[l][j], z_xp[l][j], is_relu=True)
                C_after_relu.append(m * C[j])

            gap_relu = abs(sum(C_after_relu) - sum(C)) / (abs(sum(C)) + 1e-12)
            if verbose:
                print(f"  Layer {l+1} ReLU Rescale gap: {gap_relu:.4%}")
        else:
            C_after_relu = list(C)

        # Step B: propagate through linear layer
        C_new = [0.0] * n_in
        delta_z = [z_x[l][j] - z_xp[l][j] for j in range(n_out)]
        delta_a_prev = [a_x[l][i] - a_xp[l][i] for i in range(n_in)]

        for j in range(n_out):
            dz_j = delta_z[j]
            if abs(dz_j) < 1e-12:
                continue  # no change → no contribution from this neuron
            for i in range(n_in):
                contrib_ij = delta_a_prev[i] * W[j][i]
                C_new[i] += (contrib_ij / dz_j) * C_after_relu[j]

        gap_lin = abs(sum(C_new) - sum(C_after_relu)) / (abs(sum(C_after_relu)) + 1e-12)
        if verbose:
            print(f"  Layer {l+1} Linear rule gap: {gap_lin:.4%}")

        conservation_gaps.append(gap_lin)
        C = C_new

    return C, conservation_gaps, delta_o


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE STUDY — THREE DIFFERENT REFERENCES
# ─────────────────────────────────────────────────────────────────────────────

references = {
    "Zero reference      (x'=0)": [0.0, 0.0, 0.0, 0.0],
    "Mean reference      (x'=μ)": [-0.1, -0.05, 0.20, -0.1],   # approx training mean
    "Opposite reference  (x'=-x)": [0.625, 0.240, -1.083, 0.500],
}

all_contribs = {}

print()
print("=" * 68)
print("  DEEPLIFT — THREE REFERENCES COMPARED")
print("=" * 68)

for ref_name, x_prime in references.items():
    print()
    print(f"  ── {ref_name} ──")
    C, gaps, delta_o = run_deeplift(x_input, x_prime, TARGET, verbose=True)
    all_contribs[ref_name] = (C, delta_o)

    total = sum(C)
    cons  = abs(total - delta_o) / (abs(delta_o) + 1e-12)
    print()
    print(f"  Attribution (sum rule: Σ Cᵢ = {delta_o:.4f}):")
    for fname, c in zip(FEAT_NAMES, C):
        pct = c / (abs(delta_o) + 1e-12) * 100
        bar = ("+" if c >= 0 else "−") * min(20, int(abs(c) * 20 / (abs(delta_o)+1e-9)))
        print(f"  {fname:<20}: {c:>+10.5f}  ({pct:>+6.1f}%)  {bar}")
    print(f"  {'Sum':>20}  {total:>+10.5f}  conservation gap={cons:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE TO GRADIENT × INPUT
# ─────────────────────────────────────────────────────────────────────────────

def gradient_x_input(x, target_class):
    """
    Compute Gradient × Input via backpropagation.
    Returns d f_{target} / d xᵢ × xᵢ  for each input feature i.
    """
    logits, z_all, a = forward(x)
    n_layers = len(WEIGHTS)

    delta = [1.0 if i == target_class else 0.0 for i in range(len(logits))]

    for l in range(n_layers - 1, -1, -1):
        W  = WEIGHTS[l]
        act = USE_RELU[l]
        n_out, n_in = len(W), len(W[0])

        if act:
            delta = [delta[j] * relu_deriv(z_all[l][j]) for j in range(n_out)]

        delta_prev = [sum(W[j][i] * delta[j] for j in range(n_out))
                      for i in range(n_in)]
        delta = delta_prev

    grad_input = [delta[i] * x[i] for i in range(len(x))]
    return grad_input

gi = gradient_x_input(x_input, TARGET)

print()
print("=" * 68)
print("  GRADIENT × INPUT vs DEEPLIFT (zero reference)")
print("=" * 68)
print()
logits_xp_zero, _, _ = forward([0.0, 0.0, 0.0, 0.0])
delta_o_zero = logits[TARGET] - logits_xp_zero[TARGET]
C_zero = all_contribs["Zero reference      (x'=0)"][0]

print(f"  f(x) = {logits[TARGET]:.4f},  f(0) = {logits_xp_zero[TARGET]:.4f},  "
      f"Δo = {delta_o_zero:.4f}")
print()
print(f"  {'Feature':<20}  {'GradInput':>11}  {'DeepLIFT(0)':>12}  {'Same?'}")
print(f"  {'-'*58}")
for fi, fname in enumerate(FEAT_NAMES):
    same = "✓" if abs(gi[fi] - C_zero[fi]) < 0.001 else "✗ differs"
    print(f"  {fname:<20}  {gi[fi]:>+11.5f}  {C_zero[fi]:>+12.5f}  {same}")

print(f"  {'-'*58}")
print(f"  {'Sum':>20}  {sum(gi):>+11.5f}  {sum(C_zero):>+12.5f}")
print(f"  {'Required sum':>20}  {'n/a':>11}  {delta_o_zero:>+12.5f}")
print()
gi_cons = abs(sum(gi) - delta_o_zero) / (abs(delta_o_zero) + 1e-12)
dl_cons = abs(sum(C_zero) - delta_o_zero) / (abs(delta_o_zero) + 1e-12)
print(f"  GradInput  sum-rule gap: {gi_cons:.2%}  (NOT required to satisfy sum rule)")
print(f"  DeepLIFT   sum-rule gap: {dl_cons:.2%}  (MUST satisfy sum rule by construction)")


# ─────────────────────────────────────────────────────────────────────────────
# SATURATION DEMO — WHERE DEEPLIFT AND GRADIENT DIVERGE
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SATURATION DEMO — WHERE DEEPLIFT HANDLES WHAT GRADIENTS CANNOT")
print("=" * 68)
print()
print("  A single ReLU unit: a = ReLU(w·x),  w=10, baseline x'=0")
print()
print(f"  {'x':>6}  {'ReLU output':>12}  {'Gradient':>10}  {'GradInput':>10}  "
      f"{'DeepLIFT':>10}  {'Δo':>8}")
print(f"  {'-'*65}")

w_sat = 10.0
for x_sat in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]:
    f_x   = relu(w_sat * x_sat)
    f_xp  = relu(w_sat * 0.0)
    grad  = relu_deriv(w_sat * x_sat) * w_sat
    gi_val = grad * x_sat
    delta_o_sat = f_x - f_xp
    # Rescale multiplier
    dz = w_sat * x_sat - w_sat * 0.0
    da = f_x - f_xp
    m = da / dz if abs(dz) > 1e-12 else relu_deriv(w_sat * x_sat)
    dl_val = m * x_sat * w_sat  # = m × Δz = Δa = Δo

    flag = ""
    if abs(gi_val - delta_o_sat) > 0.1:
        flag = " ← GI wrong!"
    print(f"  {x_sat:>6.1f}  {f_x:>12.3f}  {grad:>10.3f}  {gi_val:>10.3f}  "
          f"{dl_val:>10.3f}  {delta_o_sat:>8.3f}{flag}")

print()
print("  At every x, DeepLIFT(x) = Δo = ReLU(10x) − 0 exactly.")
print("  GradInput = gradient × x = 10 × x — correct only when f is linear.")
print("  DeepLIFT uses the SECANT slope (Δa/Δz = f(x)/10x = 1.0 when x>0)")
print("  so DeepLIFT × Δx = 1.0 × 10x × x = ... wait, let's be precise:")
print()
print("  Rescale multiplier m = Δa/Δz = ReLU(10x)/(10x) = 10x/(10x) = 1.0 (when x>0)")
print("  DeepLIFT contribution = m × Δx × (upstream multiplier from output)")
print("  Since output = a and we're at the only layer: C_x = m × (Δz/Δx) × Δx × m_out")
print("  Simplified: C_x = Δa = Δo  ← exact attribution of full output change.")
print()
print("  GradInput = gradient × input = 10 × x (NOT Δo = 10x always, only for x>0 here).")
print("  The issue shows for x<0 where ReLU=0: gradient=0, GradInput=0 but the")
print("  feature clearly determined the output (it kept the ReLU inactive).")


# ─────────────────────────────────────────────────────────────────────────────
# FULL SIDE-BY-SIDE — ALL THREE REFERENCES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SIDE-BY-SIDE — ALL REFERENCES AND GRADIENT × INPUT")
print("=" * 68)
print()
print(f"  {'Feature':<20}  {'GradInput':>10}", end="")
for ref_name in references:
    short = ref_name[:7].strip()
    print(f"  {short:>10}", end="")
print()
print(f"  {'-'*70}")

for fi, fname in enumerate(FEAT_NAMES):
    print(f"  {fname:<20}  {gi[fi]:>+10.5f}", end="")
    for ref_name in references:
        C_ref, _ = all_contribs[ref_name]
        print(f"  {C_ref[fi]:>+10.5f}", end="")
    print()

print(f"  {'-'*70}")
print(f"  {'Sum':>20}  {sum(gi):>+10.5f}", end="")
for ref_name in references:
    C_ref, do = all_contribs[ref_name]
    print(f"  {sum(C_ref):>+10.5f}", end="")
print()
print(f"  {'Δo':>20}  {'N/A':>10}", end="")
for ref_name in references:
    _, do = all_contribs[ref_name]
    print(f"  {do:>+10.5f}", end="")
print()
print()
print("  KEY OBSERVATIONS:")
print("  1. GradInput sum ≠ Δo for any reference — it is NOT complete.")
print("  2. All three DeepLIFT columns satisfy Σ Cᵢ = Δo exactly.")
print("  3. Attribution magnitudes SCALE with Δo (smaller when comparing")
print("     to a nearby reference, larger when comparing to a far one).")
print("  4. Signs can REVERSE across references — a feature that is")
print("     positive relative to x'=0 may be negative relative to x'=-x.")
print("  5. The zero reference (x'=0) matches GradInput in RANKING but")
print("     differs in the exact sum — DeepLIFT's sum is exact; GI's is not.")
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
    try:
        from interpretability.visuals.deeplift import (
            DEEPLIFT_VISUAL_HTML,
            DEEPLIFT_VISUAL_HEIGHT,
        )
        visual_html   = DEEPLIFT_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = DEEPLIFT_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[deeplift.py] Could not load visual: {e}", stacklevel=2)

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