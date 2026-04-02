"""
LRP — Layer-wise Relevance Propagation
=======================================

LRP (Bach, Binder, Montavon, Müller et al., 2015) is a method for explaining
neural network predictions by propagating the output score BACKWARDS through
the network, redistributing it layer by layer until each input feature receives
a relevance score. The sum of all input relevances equals the original output
score — a conservation law enforced at every layer of the network.

Unlike gradient-based methods that measure local sensitivity, LRP measures
how much each input feature CONTRIBUTED to the final prediction, making it
robust to the saturation and shattered-gradient pathologies that plague vanilla
saliency maps.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "LRP — Layer-wise Relevance Propagation"
DISPLAY_NAME = "04d · LRP"
ICON = "🔁"
SUBTITLE = "Relevance backpropagation with conservation across layers"


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

### The Central Question — Decomposing a Score, Not Measuring Sensitivity

Gradient-based methods (vanilla saliency, Integrated Gradients) answer the
question: "if I perturbed feature i by a tiny amount, how much would the output
change?" This is a sensitivity question. The answer describes the model's LOCAL
behaviour around the input point.

LRP asks a fundamentally different question:

    "The model produced a score of f(x) = 4.3. How much of that 4.3
     is attributable to each input feature?"

This is a DECOMPOSITION question. The answer describes where a specific output
quantity came from. The difference is the same as asking:

    SENSITIVITY: "If this patient had slightly higher blood pressure, how
    much would the risk score change?" — a counterfactual question.

    DECOMPOSITION: "The model assigned risk 4.3. How much of that risk
    came from the patient's blood pressure vs their cholesterol vs their
    age?" — an attribution question.

LRP answers the decomposition question by treating the neural network as a
system of pipes carrying a "relevance fluid." The fluid starts at the output
neuron (carrying the full prediction score) and flows backwards through the
network. At each layer, the fluid is distributed among the neurons of the
previous layer in proportion to how much each neuron contributed to carrying
it. By the time the fluid reaches the input layer, each input pixel or feature
holds a relevance score, and the sum of all scores equals the original output.


##### PART I: THE CONSERVATION PRINCIPLE — THE THEORETICAL FOUNDATION

### Relevance Conservation

LRP is built on a single axiom: relevance is conserved across every layer.
Define Rⱼ as the relevance of neuron j at some layer, and Rᵢ as the relevance
of neuron i at the previous layer. The conservation law states:

    Σⱼ Rⱼ = Σᵢ Rᵢ    (for every adjacent pair of layers)

The total relevance entering any layer from above equals the total relevance
leaving that layer downward. No relevance is created or destroyed — it is
only redistributed. This conservation holds at EVERY layer, not just globally.

Starting condition: at the output layer, R_output = f(x) — the model's score
for the target class. All other output neurons receive R = 0.

Ending condition: at the input layer, each input feature xᵢ holds Rᵢ.
By conservation: Σᵢ Rᵢ = f(x).

This is exactly the Completeness (Efficiency) axiom of SHAP and Integrated
Gradients, restated as a conservation law. But LRP achieves it via a
backwards propagation rule, not by integration along a path.

    # ================================================================== #
    **The conservation law visualised as a flow network:**

    OUTPUT LAYER
    ┌──────────┐
    │   R=4.3  │  ← full output score is the starting relevance
    └────┬─────┘
         │ 4.3 flows down
    HIDDEN LAYER L (3 neurons)
    ┌──────┬──────┬──────┐
    │R=2.1 │R=1.6 │R=0.6 │   2.1 + 1.6 + 0.6 = 4.3 ✓ (conserved)
    └──┬───┴──┬───┴──┬───┘
       │      │      │   relevance redistributes again
    HIDDEN LAYER L-1 (4 neurons)
    ┌──────┬──────┬──────┬──────┐
    │R=1.8 │R=1.2 │R=0.9 │R=0.4 │   1.8+1.2+0.9+0.4 = 4.3 ✓ (conserved)
    └──────┴──────┴──────┴──────┘
         ...continues to input...
    INPUT FEATURES: Σ Rᵢ = 4.3 ✓
    # ================================================================== #


### Why Conservation Matters — The Problem It Avoids

Without the conservation constraint, an attribution method is free to assign
arbitrary scores to features. A method could give every feature a score of
zero (technically correct if the question is "how much does each feature change
the output when perturbed?" and the model is saturated), while the output is
clearly non-zero.

This is not hypothetical — it is the saturation problem of vanilla gradients.
A ReLU network whose output is stuck at its maximum activation value will have
zero gradient everywhere (the gradient of a saturated ReLU is zero), even though
every feature contributed positively to that maximum.

LRP sidesteps this entirely. The total relevance to redistribute is f(x) itself.
Even if the gradient is zero (saturation), f(x) is non-zero, and LRP distributes
that non-zero quantity across features proportionally to their actual activations
— not to their local sensitivity.

    # ================================================================== #
    **Saturation comparison — LRP vs Gradient:**

    Model: f(x₁) = min(x₁, 5)  (output saturates at 5)
    Instance: x₁ = 10  →  f(x₁) = 5

    Vanilla gradient: ∂f/∂x₁ = 0  (gradient is zero above 5)
    Attribution for x₁ = 0.  ← Says "x₁ has no importance!"

    LRP starting relevance: R_output = f(x₁) = 5
    LRP propagates R=5 backwards to x₁.
    Attribution for x₁ = 5.  ← Correctly attributes full score to x₁.

    The gradient is zero because the model is insensitive to CHANGES in x₁
    at this operating point. LRP is non-zero because x₁ CONTRIBUTED to the
    output of 5. These answer different questions.
    # ================================================================== #


##### PART II: THE BASIC LRP RULE — LRP-0

### Deriving the Basic Propagation Rule

Consider a single neuron j in layer l+1 connected to neurons i in layer l.
Neuron j's pre-activation is:

    zⱼ = Σᵢ aᵢ · wᵢⱼ + bⱼ

where aᵢ is the activation of neuron i, wᵢⱼ is the weight from i to j,
and bⱼ is the bias.

Each term aᵢ · wᵢⱼ is the CONTRIBUTION of neuron i to the pre-activation zⱼ.
If we want to redistribute Rⱼ to the neurons in the previous layer, the natural
rule is: give each neuron i a share of Rⱼ proportional to its contribution:

    Rᵢ←ⱼ = (aᵢ · wᵢⱼ / zⱼ) · Rⱼ

The term aᵢ · wᵢⱼ / zⱼ is the FRACTION of zⱼ that came from neuron i.
Multiplying by Rⱼ gives the relevance that neuron i receives from neuron j.

Summing over all neurons j in the next layer (since neuron i may connect to
multiple neurons in layer l+1):

    ┌─────────────────────────────────────────────────────────────────┐
    │  LRP-0 RULE:                                                    │
    │                                                                 │
    │  Rᵢ = Σⱼ  (aᵢ · wᵢⱼ / zⱼ)  · Rⱼ                                    │
    │                                                                 │
    │  where: zⱼ = Σᵢ aᵢ · wᵢⱼ + bⱼ  (pre-activation of neuron j)       │
    └─────────────────────────────────────────────────────────────────┘

This is the Basic Rule (also called LRP-0 or the "naive rule"). It is elegant
and intuitive, but it has a critical numerical problem.

### The Division-by-Zero Problem

The LRP-0 rule divides by zⱼ. What happens when zⱼ ≈ 0?

Near-zero denominators produce extreme relevance values — large positive and
large negative values that cancel each other but are individually enormous.
The conservation law still holds numerically (the sum is preserved), but the
individual attributions are meaningless: a feature might receive R = +10,000
and another R = −9,999 simply because the denominator of one redistribution
step was tiny.

This is not a theoretical edge case. In practice, for many network architectures
and many inputs, some neurons will have near-zero pre-activations. Every LRP
variant beyond LRP-0 is a different strategy for handling this instability.

    # ================================================================== #
    **LRP-0 instability example:**

    Neuron j has: z_j = 0.001 (near-zero pre-activation)
    Neuron j receives: R_j = 0.5

    Two inputs to j:
      Contribution from i=1: a₁·w₁ⱼ = +0.8  (large positive)
      Contribution from i=2: a₂·w₂ⱼ = -0.799  (large negative)
      Sum: 0.8 - 0.799 = 0.001 = zⱼ  ✓

    LRP-0 redistribution:
      R₁ = (0.8 / 0.001) × 0.5 = +400    ← enormous!
      R₂ = (-0.799 / 0.001) × 0.5 = -399.5

    Conservation: 400 + (-399.5) = 0.5 = Rⱼ  ✓ (correct)
    But individual values: +400 and -399.5 are numerically useless.
    # ================================================================== #


##### PART III: THE STABILISED RULES — ε-LRP AND β-LRP

### LRP-ε — The Stabiliser Rule

LRP-ε adds a small constant ε to the denominator:

    ┌─────────────────────────────────────────────────────────────────┐
    │  LRP-ε RULE:                                                    │
    │                                                                 │
    │  Rᵢ = Σⱼ  (aᵢ · wᵢⱼ / (zⱼ + ε · sign(zⱼ)))  · Rⱼ                    │
    │                                                                 │
    │  where ε > 0 is a small stabiliser (typically 0.1 to 1.0)       │
    └─────────────────────────────────────────────────────────────────┘

The `sign(zⱼ)` ensures the stabiliser shifts in the same direction as zⱼ,
preventing the denominator from being pushed toward zero from the other side.
When |zⱼ| is large relative to ε, the rule behaves like LRP-0. When zⱼ ≈ 0,
the denominator is clamped at ε, preventing explosion.

The cost: when ε is non-negligible relative to zⱼ, conservation is violated.
Some relevance is absorbed by the ε term and disappears. The total input
relevance Σᵢ Rᵢ will be slightly less than f(x). This is a deliberate
trade-off: stability over strict conservation.

The intuition: neurons that barely fired (near-zero pre-activation) contributed
little to the network's computation. LRP-ε says: do not assign large relevance
attributions through neurons that contributed nearly nothing. The ε term
absorbs the relevance of near-zero neurons rather than amplifying it.

    Choosing ε:
    ──────────────────────────────────────────────────────────────────
    ε = 0:    Reduces to LRP-0 (numerically unstable).
    ε = 0.1:  Mild stabilisation. Near-conservation, some instability.
    ε = 1.0:  Strong stabilisation. More relevance absorbed, sparser maps.
    ε → ∞:    All relevance absorbed. Attribution everywhere → 0. Useless.

    In practice: ε = 0.1 for most applications. Larger ε for very deep
    networks where many near-zero pre-activations accumulate.


### LRP-β — Separating Positive and Negative Evidence

LRP-β (also written LRP-αβ or LRP-α₁β₀) takes a different approach to
numerical stability. Instead of adding a stabiliser, it separates positive
and negative contributions and treats them independently:

    ┌─────────────────────────────────────────────────────────────────┐
    │  LRP-β RULE:                                                    │
    │                                                                 │
    │  Rᵢ = Σⱼ [ α · (aᵢ · wᵢⱼ)⁺ / zⱼ⁺  +  β · (aᵢ · wᵢⱼ)⁻ / zⱼ⁻ ] · Rⱼ    │
    │                                                                 │
    │  where:                                                         │
    │    (x)⁺ = max(x, 0)  (positive part)                            │
    │    (x)⁻ = min(x, 0)  (negative part)                            │
    │    zⱼ⁺ = Σᵢ (aᵢwᵢⱼ)⁺ + bⱼ⁺   (sum of positive contributions)      │
    │    zⱼ⁻ = Σᵢ (aᵢwᵢⱼ)⁻ + bⱼ⁻   (sum of negative contributions)       │
    │    α, β ≥ 0 with α − β = 1  (to maintain conservation)          │
    └─────────────────────────────────────────────────────────────────┘

The most common choices are α=1, β=0 (called LRP-α₁β₀ or LRP-α1) and
α=2, β=1 (called LRP-α₂β₁).

Why does separating positive and negative help stability?
  • zⱼ⁺ = sum of only positive contributions ≥ 0.
    It is zero only if ALL contributions are negative — an extreme case.
  • zⱼ⁻ = sum of only negative contributions ≤ 0.
    It is zero only if ALL contributions are positive — another extreme.

The two sums never simultaneously equal zero (unless the neuron has zero input,
in which case the whole layer contributes nothing). This eliminates the near-zero
denominator problem for any realistic network input.

    LRP-α₁β₀ (α=1, β=0):
    ──────────────────────
    Rᵢ = Σⱼ [(aᵢ · wᵢⱼ)⁺ / zⱼ⁺] · Rⱼ

    Only positive contributions are propagated. Negative contributions
    (features that SUPPRESSED neuron j) receive zero relevance.
    This produces SPARSE, positive-only attribution maps.
    The trade-off: some relevant features with negative weights are ignored.

    LRP-α₂β₁ (α=2, β=1):
    ──────────────────────
    Rᵢ = Σⱼ [2·(aᵢwᵢⱼ)⁺/zⱼ⁺ − 1·(aᵢwᵢⱼ)⁻/zⱼ⁻] · Rⱼ

    Both positive and negative contributions are included, with the
    negative given half the weight of the positive. Features that
    SUPPRESSED class-relevant neurons receive negative relevance.
    This produces signed attribution maps showing both support and
    counter-evidence.

    Conservation for LRP-β:
    Σᵢ Rᵢ = (α − β) × Rⱼ = 1 × Rⱼ  (since α − β = 1)
    Conservation is EXACT when α − β = 1, regardless of the specific values.


### The Relationship Between LRP-0 and Gradient × Input

LRP-0 is mathematically equivalent to Gradient × Input for networks with
ONLY positive activations (ReLU networks where all activations are ≥ 0).

Proof for a single ReLU layer (aᵢ = ReLU(zᵢ)):
  LRP-0 rule: Rᵢ = aᵢ · (∂zⱼ/∂aᵢ) / zⱼ · Rⱼ
  With ReLU: Gradient of loss w.r.t. aᵢ = (∂zⱼ/∂aᵢ) × ∂output/∂zⱼ
  When zⱼ = output and Rⱼ = 1:
    LRP-0 Rᵢ = aᵢ × (∂output/∂aᵢ)  =  GradInput(aᵢ)

So LRP-0 = GradInput for the first layer of a ReLU network. For deeper layers,
the equivalence breaks because LRP propagates the relevance score as a quantity
(dividing by zⱼ at each step), while GradInput computes a gradient chain.

This relationship means:
  1. Where GradInput is a valid explanation, LRP-0 gives the same answer.
  2. LRP-ε and LRP-β go beyond GradInput by handling negative activations
     and near-zero denominators with explicit design choices, while
     GradInput has no such safeguards.
  3. LRP is strictly more general than GradInput.


##### PART IV: THE z⁺ RULE AND THE INPUT LAYER

### The z⁺ Rule for Input Pixels

At the very first layer (connecting pixel values to the first hidden layer),
a specialised rule is often applied: the z⁺ rule (also called LRP-z or
the "flat" rule in some formulations).

The motivation: pixel values are non-negative (image pixels ∈ [0, 255] or
[0, 1]). At this layer, the activations aᵢ = xᵢ (the raw pixel values).
Using LRP-0 at this layer still works, but the z⁺ rule gives an alternative:

    Rᵢ = Σⱼ  (xᵢ · wᵢⱼ⁺) / (Σₖ xₖ · wₖⱼ⁺)  ·  Rⱼ

where wᵢⱼ⁺ = max(wᵢⱼ, 0) (only positive weights are used at the input layer).

The z⁺ rule ensures that relevance only flows through POSITIVE connections at
the input layer. This is justified by the observation that negative input-layer
weights (which decrease a neuron's activation for a given pixel) represent
"absence of the feature" — they are anti-correlated with the pixel, not caused
by it. Attributing relevance through anti-correlations at the pixel level is
often unintuitive.

### The Bounded Rule (LRP-zᴮ) for the Input Layer

An even more principled rule for the first layer accounts for the known bounds
of pixel values [lᵢ, hᵢ] (e.g., l=0, h=255 for 8-bit images):

    Rᵢ = Σⱼ  xᵢwᵢⱼ − lᵢwᵢⱼ⁺ − hᵢwᵢⱼ⁻
            ──────────────────────────────── · Rⱼ
            zⱼ − (Σₖ lₖwₖⱼ⁺) − (Σₖ hₖwₖⱼ⁻)

This rule uses the lower bound lᵢ and upper bound hᵢ as reference points,
distributing relevance relative to what the network would have computed for
the minimum and maximum possible pixel values. It is more faithful than the
z⁺ rule for bounded input domains and avoids the sign issues that arise when
pixel values are standardised to have negative values.


##### PART V: THE LRP COMPOSITE STRATEGY — MIXING RULES ACROSS LAYERS

### Why No Single Rule Works for All Layers

Different layers in a CNN serve different computational roles, and the
appropriate LRP rule differs accordingly:

    OUTPUT/CLASSIFIER LAYERS:
    Typically fully connected with potentially large, signed weights.
    Instability from near-zero pre-activations is common.
    Recommended rule: LRP-ε (absorbs instability, nearly conserving).

    DEEP HIDDEN LAYERS (far from input):
    ReLU activations ensure non-negative outputs from these layers.
    The network encodes abstract semantic features here.
    Recommended rule: LRP-ε or LRP-α₁β₀.

    LOWER HIDDEN LAYERS (near input):
    Encode local features like edges, textures.
    Need to faithfully trace which local pattern triggered which detection.
    Recommended rule: LRP-α₂β₁ (more faithful, both signs).

    INPUT LAYER:
    Pixel values are bounded and non-negative.
    Recommended rule: LRP-zᴮ (bounded rule).

The "composite" strategy (Montavon et al., 2019) uses different rules at
different depths:

    # ================================================================== #
    **Composite LRP strategy for a VGG-like network:**

    Input        → Conv layers (low-level) : LRP-zᴮ (bounded rule)
    Conv blocks 1–2 (edge/texture detectors): LRP-α₂β₁
    Conv blocks 3–5 (part/object detectors) : LRP-α₁β₀
    Classifier (FC layers)                  : LRP-ε

    Rationale:
    • At the input: use bounded rule to respect pixel bounds [0,1].
    • In early layers: use α₂β₁ to capture both excitatory and
      inhibitory local evidence (both types of edge = relevant).
    • In deep layers: use α₁β₀ to keep explanations positive and sparse.
    • In the classifier: use ε to absorb numerical instability from
      large, signed FC weights.
    # ================================================================== #

This composite approach consistently outperforms any single rule applied
uniformly across all layers, and it is the recommended practice for
LRP in production systems.


### The DEEP TAYLOR DECOMPOSITION — The Unifying Framework

Montavon, Lapuschkin, Binder, Müller & Bach (2017) showed that all LRP
rules can be derived from a single unifying framework: DEEP TAYLOR DECOMPOSITION.

The core idea: LRP-0 can be written as a first-order Taylor expansion of the
neuron's activation around a root point z̃ⱼ where the activation equals zero:

    aⱼ(x) = aⱼ(x̃) + ∇ₓaⱼ(x̃) · (x − x̃) + O(‖x−x̃‖²)

where x̃ is chosen so aⱼ(x̃) = 0 (the "root" of the Taylor expansion).

For a linear unit aⱼ = Σᵢ wᵢⱼxᵢ + bⱼ, the root is any x̃ satisfying
Σᵢ wᵢⱼx̃ᵢ + bⱼ = 0. The Taylor decomposition then assigns to each input xᵢ
the attribution:

    Rᵢ = (xᵢ − x̃ᵢ) × ∂aⱼ/∂xᵢ = (xᵢ − x̃ᵢ) × wᵢⱼ

The LRP rule Rᵢ = aᵢwᵢⱼ/zⱼ · Rⱼ recovers this when x̃ = 0 (zero root) and
Rⱼ = zⱼ (the activation as relevance). Different choices of root x̃ give
different LRP variants:

    x̃ = 0:           → LRP-0 (zero root)
    x̃ = l (lower bd):→ LRP-zᴮ for positive weights
    x̃ = h (upper bd):→ LRP-zᴮ for negative weights

The Deep Taylor Decomposition thus provides a systematic recipe for designing
LRP rules tailored to the constraints of each layer type.

    # ================================================================== #
    **Deep Taylor Decomposition — the geometric interpretation:**

    aⱼ(x) is a function of inputs x. We want to find a point x̃ (the root)
    where aⱼ = 0, and use the linear approximation at that root to distribute
    credit among inputs.

    For a non-negative input (pixel x ∈ [0,1]):
    The natural root is x̃ = 0 (black pixel — no input signal).
    The attribution is the amount of activation caused by "switching on"
    the pixel from 0 to its actual value.

    For a signed, standardised input (x ∈ [−1,1]):
    Two roots exist: x̃ = −1 (minimum) and x̃ = +1 (maximum).
    LRP-zᴮ uses both simultaneously, weighting by sign of the weight.

    This geometric view makes clear why different layers need different rules:
    the natural "reference point" (what the input looks like with no information)
    differs between layers.
    # ================================================================== #


##### PART VI: LRP FOR SPECIAL LAYER TYPES

### LRP Through ReLU Activations

ReLU(z) = max(0, z) is piecewise linear:
  • If z > 0: ReLU(z) = z → the activation passes through unchanged.
  • If z ≤ 0: ReLU(z) = 0 → the activation is blocked.

LRP propagates through ReLU in two ways depending on convention:

    CONVENTION 1 (Propagate to pre-activation):
    Treat the ReLU as part of the preceding linear layer. The relevance
    Rⱼ at the neuron after ReLU is redistributed to the neuron's inputs
    using the PRE-activation zⱼ in the denominator. The ReLU itself
    receives no special treatment.

    CONVENTION 2 (Propagate through activation):
    Apply a relevance gating: if ReLU(zⱼ) = 0 (neuron was inactive),
    R passed through ReLU is 0 (blocked). If ReLU(zⱼ) > 0 (neuron
    active), relevance passes through unchanged.

    Rᵢ^(after ReLU) = Rⱼ × 𝟙[zⱼ > 0]

Convention 2 mirrors Backpropagation and is commonly used. It means:
INACTIVE neurons (those that did not fire for this input) receive
ZERO relevance. Only the active computational pathway carries relevance.

    # ================================================================== #
    **LRP through ReLU — the active pathway principle:**

    For a neuron j with pre-activation zⱼ = 0.73 (active):
    Relevance Rⱼ = 0.4 passes through unchanged → next layer receives 0.4.

    For a neuron j with pre-activation zⱼ = −0.12 (inactive):
    Relevance Rⱼ = 0.4 is BLOCKED → 0.0 reaches the previous layer.

    This is why LRP maps look SPARSE: relevance only flows through the
    neurons that actually fired for the specific input. Inactive neurons
    are transparent — they carry no relevance backward.
    # ================================================================== #


### LRP Through Max-Pooling

Max-pooling selects the MAXIMUM activation within a pooling window and
discards all others. LRP handles this by routing all relevance of the
max-pooling output exclusively to the WINNING LOCATION:

    Rᵢ_winner = R_pool_output  (all relevance to the winning position)
    Rᵢ_others = 0              (zero to all non-winning positions)

This is the "winner-take-all" propagation. It concentrates relevance on
the most active spatial location within each pooling window. The result:
LRP maps through max-pooling layers tend to be sharper and more localised
than through average-pooling layers.

For AVERAGE-POOLING, relevance is distributed equally to all positions in
the window:

    Rᵢ = R_pool_output / window_size  (for each position in the window)


### LRP Through Batch Normalisation

Batch normalisation (BN) applies an affine transformation after standardising:

    BN(z) = γ · (z − μ_B) / √(σ²_B + ε) + β

where μ_B, σ²_B are the batch mean and variance, γ and β are learnable
parameters. During inference, these are replaced by the running statistics.

For LRP, BN is typically treated as a LINEAR LAYER with a fixed scale and
shift. The BN operation can be "folded" into the preceding convolutional
layer's weights and bias before computing LRP, producing a fused layer that
LRP can handle with standard rules.

Fused weights: ŵᵢⱼ = wᵢⱼ × γ / √(σ² + ε)
Fused bias: b̂ⱼ = (bⱼ − μ) × γ / √(σ² + ε) + β


### LRP for Transformers (Attention Mechanisms)

LRP has been extended to transformer architectures. The attention mechanism
computes:

    Attention(Q, K, V) = softmax(QKᵀ/√d) × V

The softmax and the multiplication by V are non-standard layers. The
LRP rule for attention (Chefer et al., 2021) uses a modified relevance
conservation that accounts for the non-linear softmax:

    Rᵢ = Aᵀ · Rⱼ  (relevance flows through the attention matrix)

plus a gradient-based correction term. This "Transformer Interpretability
Beyond Attention Visualisation" approach produces maps that pass sanity
checks and are more faithful than raw attention weights.


##### PART VII: LRP vs OTHER ATTRIBUTION METHODS

### LRP vs Integrated Gradients

Both LRP and Integrated Gradients (IG) satisfy a Completeness axiom:
Σᵢ Rᵢ = f(x) − f(x') (IG with baseline) or Σᵢ Rᵢ = f(x) (LRP).

The difference is conceptual:

    INTEGRATED GRADIENTS:
    Computes the attribution by integrating along a path from baseline x'
    to input x. The baseline represents "no information." Attribution
    measures the contribution relative to the baseline.
    The path is explicit and user-specified. The integral is computed by
    M forward+backward passes (M typically 50–300).

    LRP:
    Computes attribution by distributing f(x) backwards through the network
    using a layer-specific redistribution rule.
    No baseline is needed. Attribution measures absolute contribution to f(x).
    The computation is one forward pass + one backward pass (using modified
    gradient rules, not the standard gradient).

The key practical differences:

    LRP has NO BASELINE CHOICE. The attribution is to the full output f(x),
    not to f(x) − f(x'). This removes a design parameter but also removes
    the ability to ask "contribution relative to absence."

    LRP has MULTIPLE RULES that must be chosen for each layer. This gives
    more flexibility but requires more domain knowledge to apply correctly.

    IG has a SINGLE ALGORITHM with one parameter (number of steps M).
    It is simpler to apply correctly but may fail near saturating units
    (though the path integral naturally handles this better than gradients).

    In empirical comparisons, LRP (with composite rules) and IG produce
    similar quality maps. LRP is faster (no repeated model calls). IG
    has a stronger theoretical basis (axiomatic derivation from game theory
    via Deep SHAP / Shapley values).

    # ================================================================== #
    **Computation cost comparison:**

    Method          Fwd passes   Bwd passes   Notes
    ──────────────────────────────────────────────────────────────────
    Vanilla grad    1            1            1 standard bwd pass
    LRP (any rule)  1            1*           *modified bwd pass, not std grad
    GradCAM         1            1            operates on last conv layer only
    SmoothGrad      N            N            N=50–100 typically
    Integrated Grad M            M            M=50–300 for convergence
    SHAP KernelSHAP 2^n·m        0            m samples, model-agnostic
    SHAP TreeSHAP   —            —            O(TLD²), exact for trees

    * LRP's backward pass uses the relevance propagation rules, not the
      standard chain rule. It cannot be implemented directly in PyTorch's
      autograd — it requires custom backward hooks or a dedicated library.
    # ================================================================== #


### LRP vs GradCAM

Both LRP and GradCAM are model-specific (operate inside the network), and
both produce spatial heatmaps for images. They differ fundamentally in scope:

    GRADCAM:
    Operates only at the LAST CONVOLUTIONAL LAYER. The heatmap resolution
    is limited by the spatial dimensions of that layer (7×7 for VGG16).
    The method is conceptually simple — gradient × activation at one layer.
    Output: coarse spatial heatmap at low resolution.

    LRP:
    Propagates relevance through EVERY LAYER from output to input.
    The attribution is at pixel level — the same resolution as the input.
    The method requires carefully chosen rules at each layer type.
    Output: pixel-level attribution map at full input resolution.

When to use each:
  • GradCAM: need a quick, coarse "where did the network look?" map.
    Interpretable without domain expertise. Works even with poor baseline.
  • LRP: need pixel-level attributions with conservation guarantees.
    More suitable for quantitative analysis, comparison studies, and
    medical/scientific applications where pixel-level accuracy matters.


### LRP vs DeepLIFT

DeepLIFT (Shrikumar et al., 2017) is closely related to LRP-ε. DeepLIFT
defines a reference input x' (like IG's baseline) and propagates the
DIFFERENCE f(x) − f(x') backwards:

    DeepLIFT contribution score for feature i:
    Cᵢ = (aᵢ − a'ᵢ) × (∂f/∂aᵢ evaluated at some intermediate point)

DeepLIFT uses a "multiplier" that is similar to LRP's redistribution factor:

    Multiplier_i = Cᵢ / (aᵢ − a'ᵢ)

The difference from LRP: DeepLIFT attributes the CHANGE in output (f(x) − f(x'))
while LRP attributes the ABSOLUTE output f(x).
Setting the reference a' = 0 makes DeepLIFT's multiplier equal to LRP-ε's
redistribution factor in many cases.

SHAP's DeepExplainer is built on DeepLIFT with the DeepSHAP extension that
uses multiple background references, combining DeepLIFT's computational
efficiency with SHAP's game-theoretic consistency guarantees.


##### PART VIII: PROPERTIES AND FAILURE MODES

### What LRP Guarantees

    CONSERVATION (when ε=0 or α−β=1):
    Σᵢ Rᵢ = f(x). Every unit of the output is accounted for.

    POSITIVITY (LRP-α₁β₀):
    All relevances Rᵢ ≥ 0. Attribution is non-negative everywhere,
    giving a "support only" interpretation.

    SIGNED ATTRIBUTION (LRP-α₂β₁):
    Relevances can be positive (supporting the class) or negative
    (counter-evidence against the class), giving a richer picture.

    SPARSITY:
    Relevance only flows through ACTIVE neurons (fired ReLU units).
    Inactive neurons are transparent. Maps are naturally sparse.

    SANITY CHECK COMPLIANCE:
    LRP passes the Adebayo cascading randomisation tests. Maps change
    meaningfully as network weights are progressively randomised,
    confirming LRP describes the model, not just the input.


### Failure Mode 1 — Rule Selection Sensitivity

The choice of rule (LRP-0, LRP-ε, LRP-α₁β₀, LRP-α₂β₁, LRP-zᴮ) significantly
affects the resulting heatmap. Two practitioners applying LRP to the same model
with different rule choices will get different attributions — both valid but
neither universally correct.

    LRP-0:       Numerically unstable. Produces heatmaps with
                 extreme positive and negative spikes.
    LRP-α₁β₀:    Only positive relevance. Ignores suppressive features.
    LRP-α₂β₁:    Both signs. More faithful but noisier near negative weights.
    LRP-ε:        Most stable. But absorbs some relevance (violates conservation).

There is no single "correct" rule. The best rule for a given task is
empirically determined or theoretically motivated by the Deep Taylor
Decomposition for that layer type.


### Failure Mode 2 — Relevance Concentration

In very deep networks, relevance tends to concentrate on a small number of
highly active neurons during backpropagation. If one neuron in a hidden layer
received 90% of the relevance, the 90% will propagate backwards and concentrate
further, eventually producing a heatmap where 1–2 pixels account for nearly
all the relevance.

This can happen even when the TRUE attribution is distributed across many
features. LRP-ε mitigates this by absorbing relevance at near-zero neurons
(spreading the attribution more evenly), but LRP-0 and LRP-α₁β₀ can suffer
from concentration.

Detecting concentration: compute the entropy of the normalised relevance map.
    H = −Σᵢ (|Rᵢ|/Σ|Rᵢ|) log(|Rᵢ|/Σ|Rᵢ|)
Low entropy → concentrated map. High entropy → diffuse map.
Either can be correct. Cross-validate with SHAP or IG if maps seem pathologically
concentrated.


### Failure Mode 3 — Implementation Dependency

Unlike IG (which requires only standard autograd), LRP requires CUSTOM backward
pass rules. In PyTorch or TensorFlow, this means overriding the standard
gradient computation with hooks or custom modules. Bugs in the hook
implementation can silently break conservation, producing maps that LOOK
reasonable but violate Σᵢ Rᵢ = f(x).

Always verify conservation numerically:
    conservation_check = abs(sum(R_input) − f(x)) / abs(f(x))
    assert conservation_check < 0.01  # ← less than 1% error

A violation exceeding 1% indicates either a bug in the implementation or
an intentional absorption (LRP-ε with large ε). Both should be documented.


##### PART IX: A COMPLETE WORKED EXAMPLE — LRP BY HAND

### Tracing LRP-0, LRP-ε, and LRP-α₁β₀ Through a 3-Layer Network

We use a minimal 3-layer network to trace all three rules side by side,
computing every redistribution step explicitly.

    Network: 2 inputs → 2 hidden → 1 output (no bias for clarity)
    Weights:
      Layer 1→2:  w₁₁ = 1.0, w₁₂ = −0.5
                  w₂₁ = 0.5, w₂₂ =  1.5

      Layer 2→3:  w₃₁ = 2.0, w₃₂ = −1.0

    Input: x = (x₁=1.0, x₂=0.8)

    FORWARD PASS:

    Pre-activations (layer 2):
      z₁ = w₁₁·x₁ + w₂₁·x₂ = 1.0(1.0) + 0.5(0.8) = 1.0 + 0.4 = 1.4
      z₂ = w₁₂·x₁ + w₂₂·x₂ = −0.5(1.0) + 1.5(0.8) = −0.5 + 1.2 = 0.7

    Post-activation (ReLU, both positive → pass through):
      a₁ = ReLU(1.4) = 1.4
      a₂ = ReLU(0.7) = 0.7

    Pre-activation (output):
      z_out = w₃₁·a₁ + w₃₂·a₂ = 2.0(1.4) + (−1.0)(0.7) = 2.8 − 0.7 = 2.1

    Output: f(x) = 2.1


    STEP 1 — Initialise relevance at output:
    R_out = f(x) = 2.1


    STEP 2 — Propagate from output to hidden layer (layer 2):

    Using LRP-0:
    Contribution of a₁ to z_out: a₁·w₃₁ = 1.4 × 2.0 = +2.8
    Contribution of a₂ to z_out: a₂·w₃₂ = 0.7 × (−1.0) = −0.7
    z_out = 2.8 + (−0.7) = 2.1  ✓

    R₁⁽²⁾ = (a₁·w₃₁ / z_out) × R_out = (2.8 / 2.1) × 2.1 = 2.8
    R₂⁽²⁾ = (a₂·w₃₂ / z_out) × R_out = (−0.7 / 2.1) × 2.1 = −0.7

    Check: 2.8 + (−0.7) = 2.1 = R_out  ✓  (LRP-0 conserves exactly)

    Using LRP-ε (ε=0.1):
    denominator = z_out + ε·sign(z_out) = 2.1 + 0.1 = 2.2
    R₁⁽²⁾ = (2.8 / 2.2) × 2.1 = 2.673
    R₂⁽²⁾ = (−0.7 / 2.2) × 2.1 = −0.668

    Check: 2.673 + (−0.668) = 2.005 ≠ 2.1  (LRP-ε absorbs 0.095)

    Using LRP-α₁β₀ (α=1, β=0):
    Positive contributions only: a₁·w₃₁ = +2.8
    z_out⁺ = 2.8  (only positive contributions)
    R₁⁽²⁾ = (2.8 / 2.8) × 2.1 = 2.1  (all relevance to neuron 1)
    R₂⁽²⁾ = 0.0  (negative contribution → zero relevance)

    Check: 2.1 + 0.0 = 2.1 = R_out  ✓  (LRP-α₁β₀ conserves exactly when α−β=1)


    STEP 3 — Propagate from hidden to input (layer 1):

    (Continuing with LRP-0 values: R₁⁽²⁾=2.8, R₂⁽²⁾=−0.7)

    For neuron 1 in hidden layer (R₁⁽²⁾=2.8):
      Contributions from input: x₁·w₁₁=1.0×1.0=1.0, x₂·w₂₁=0.8×0.5=0.4
      z₁ = 1.0 + 0.4 = 1.4

      R_{x₁←1} = (x₁·w₁₁ / z₁) × R₁⁽²⁾ = (1.0/1.4) × 2.8 = 2.000
      R_{x₂←1} = (x₂·w₂₁ / z₁) × R₁⁽²⁾ = (0.4/1.4) × 2.8 = 0.800

    For neuron 2 in hidden layer (R₂⁽²⁾=−0.7):
      Contributions: x₁·w₁₂=1.0×(−0.5)=−0.5, x₂·w₂₂=0.8×1.5=1.2
      z₂ = −0.5 + 1.2 = 0.7

      R_{x₁←2} = (−0.5/0.7) × (−0.7) = 0.500
      R_{x₂←2} = (1.2/0.7) × (−0.7) = −1.200

    Total input relevance (LRP-0):
      R_{x₁} = R_{x₁←1} + R_{x₁←2} = 2.000 + 0.500 = 2.500
      R_{x₂} = R_{x₂←1} + R_{x₂←2} = 0.800 + (−1.200) = −0.400

    Conservation check: 2.500 + (−0.400) = 2.100 = f(x)  ✓

    INTERPRETATION:
    ──────────────────────────────────────────────────────────────────────
    LRP-0 says:
      x₁ contributed +2.500 to the output of 2.1 (119% of total — it
      "over-contributed" because x₂ simultaneously suppressed the output
      via the −0.5 weight, and LRP accounts for this by giving x₂ negative
      relevance −0.400 to cancel the over-attribution of x₁).

    LRP-α₁β₀ says:
      The −1.0 weight from a₂ to z_out blocks relevance flow through
      neuron 2 entirely. The −0.5 weight from x₁ to hidden neuron 2 is
      also blocked. x₂ receives relevance ONLY through the positive pathway.
      This gives a purely positive, sparser attribution.

    The two rules tell genuinely different stories about the same prediction.
    Neither is universally "correct" — they answer different versions of the
    attribution question (all-paths vs positive-only-paths).

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔════════════════════╦══════════════╦══════════════╦══════════════╦══════════════╗
    ║ Property           ║ LRP-0        ║ LRP-ε        ║ LRP-α₁β₀     ║ LRP-α₂β₁     ║
    ╠════════════════════╬══════════════╬══════════════╬══════════════╬══════════════╣
    ║ Conservation       ║ Exact        ║ Approximate  ║ Exact        ║ Exact        ║
    ║ Stability          ║ Poor (÷0)    ║ Good         ║ Good         ║ Good         ║
    ║ Attribution sign   ║ ±signed      ║ ±signed      ║ ≥0 only      ║ ±signed      ║
    ║ Negative evidence  ║ Yes          ║ Yes          ║ No (dropped) ║ Yes          ║
    ║ Recommended for    ║ Never        ║ Classifier   ║ Deep hidden  ║ Lower hidden ║
    ║                    ║              ║ layers       ║ layers       ║ layers       ║
    ║ Sparsity           ║ Low          ║ Medium       ║ High         ║ Medium       ║
    ╚════════════════════╩══════════════╩══════════════╩══════════════╩══════════════╝

    Core LRP rules:
      LRP-0:   Rᵢ = Σⱼ (aᵢwᵢⱼ / zⱼ) · Rⱼ
      LRP-ε:   Rᵢ = Σⱼ (aᵢwᵢⱼ / (zⱼ + ε·sgn(zⱼ))) · Rⱼ
      LRP-αβ:  Rᵢ = Σⱼ [α·(aᵢwᵢⱼ)⁺/zⱼ⁺ + β·(aᵢwᵢⱼ)⁻/zⱼ⁻] · Rⱼ,  α−β=1
      LRP-zᴮ:  Rᵢ = Σⱼ [xᵢwᵢⱼ − lᵢwᵢⱼ⁺ − hᵢwᵢⱼ⁻] / [zⱼ − Σₖlₖwₖⱼ⁺ − Σₖhₖwₖⱼ⁻] · Rⱼ

    Composite strategy (recommended):
      Input layer       → LRP-zᴮ
      Early conv layers → LRP-α₂β₁
      Late conv layers  → LRP-α₁β₀
      Classifier layers → LRP-ε
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "LRP from Scratch — All Rules, Conservation Checks, and Composite Strategy": {
        "description": (
            "Implements LRP-0, LRP-ε, LRP-α₁β₀, LRP-α₂β₁, and the z⁺ rule from "
            "scratch in pure Python on a 3-layer neural network. Traces every "
            "redistribution step for each rule, prints the conservation check after "
            "each layer, compares the final attribution maps side by side, and "
            "demonstrates the composite strategy (different rules at different layers). "
            "Also demonstrates why LRP-0 is numerically unstable for near-zero "
            "pre-activations. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "lrp",
        "code": '''
"""
================================================================================
LRP FROM SCRATCH — ALL RULES WITH CONSERVATION CHECKS
================================================================================

We implement a 3-layer fully-connected network from scratch and compute
LRP attributions using four rules:
  1. LRP-0   (basic rule — exact conservation, numerically unstable)
  2. LRP-ε   (stabilised — approximate conservation, stable)
  3. LRP-α₁β₀ (positive only — exact conservation, sparse)
  4. LRP-α₂β₁ (signed — exact conservation, richer)

Then we demonstrate the COMPOSITE STRATEGY: applying different rules at
different layers, as recommended in the literature.

Network architecture:
  4 inputs → 5 hidden (ReLU) → 3 hidden (ReLU) → 2 outputs
  We explain the prediction for one instance (loan features).
================================================================================
"""

import math
import random

random.seed(77)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK DEFINITION (fixed weights, simulating a trained model)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return max(0.0, x)

def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]

# Layer 1: 4 → 5
W1 = [
    [ 0.8, -0.4,  0.6,  0.3],
    [-0.3,  0.9,  0.2, -0.7],
    [ 0.5,  0.1, -0.8,  0.4],
    [-0.6,  0.7,  0.3,  0.9],
    [ 0.2, -0.5,  0.7, -0.3],
]
b1 = [-0.2,  0.1, -0.1,  0.3, -0.1]

# Layer 2: 5 → 3
W2 = [
    [ 0.7, -0.3,  0.5, -0.4,  0.2],
    [-0.4,  0.8, -0.2,  0.6, -0.5],
    [ 0.3, -0.6,  0.9, -0.1,  0.7],
]
b2 = [0.1, -0.2, 0.1]

# Layer 3 (output): 3 → 2
W3 = [
    [-0.6,  0.9, -0.3],   # DENY logit
    [ 0.8, -0.5,  0.7],   # APPROVE logit
]
b3 = [0.1, -0.1]

WEIGHTS  = [W1, W2, W3]
BIASES   = [b1, b2, b3]
USE_RELU = [True, True, False]  # no activation on output layer

FEAT_NAMES = ["credit_std", "income_std", "debt_std", "employment_std"]

# Input: a denied loan applicant (standardised)
x_input = [-0.625, -0.240, +1.083, -0.500]


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS — save pre- and post-activations for LRP
# ─────────────────────────────────────────────────────────────────────────────

def forward(x, weights, biases, use_relu):
    """
    Returns (outputs, all pre-activations z, all post-activations a).
    z[l] = pre-activation at layer l
    a[l] = post-activation at layer l (a[0] = input x)
    """
    a = [list(x)]
    z_all = []
    for l, (W, b, act) in enumerate(zip(weights, biases, use_relu)):
        z_l = [b[j] + sum(W[j][i] * a[-1][i] for i in range(len(a[-1])))
               for j in range(len(W))]
        z_all.append(z_l)
        if act:
            a.append([relu(z) for z in z_l])
        else:
            a.append(list(z_l))
    return a[-1], z_all, a


logits, z_all, activations = forward(x_input, WEIGHTS, BIASES, USE_RELU)
probs = softmax(logits)
pred = 0 if logits[0] > logits[1] else 1

print("=" * 68)
print("  NETWORK AND INSTANCE")
print("=" * 68)
print(f"  Architecture: 4 → 5 → 3 → 2")
print(f"  Input features (standardised):")
for fname, val in zip(FEAT_NAMES, x_input):
    print(f"    {fname:<18}: {val:+.3f}")
print()
print(f"  Logits:  DENY={logits[0]:.4f},  APPROVE={logits[1]:.4f}")
print(f"  Probs:   DENY={probs[0]:.1%},   APPROVE={probs[1]:.1%}")
print(f"  Prediction: {'DENY' if pred==0 else 'APPROVE'}")
print()
print(f"  Pre-activations at each layer:")
for l, z_l in enumerate(z_all):
    print(f"    Layer {l+1}: [{', '.join(f'{v:.4f}' for v in z_l)}]")


# ─────────────────────────────────────────────────────────────────────────────
# LRP PROPAGATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

TARGET = pred   # explain the predicted class

def lrp_propagate_layer(R_next, a_prev, W, rule, eps=0.1, alpha=1.0, beta=0.0,
                         z_preact=None):
    """
    Propagate relevance R_next (at layer l+1) back to layer l.

    R_next: list of relevance values at layer l+1
    a_prev: list of activations at layer l (inputs to this layer)
    W:      weight matrix for this layer (W[j][i] = weight from i to j)
    rule:   'lrp0', 'lrpe', 'lrp_alpha_beta'
    Returns: list of relevance values for layer l
    """
    n_in  = len(a_prev)
    n_out = len(R_next)
    R_prev = [0.0] * n_in

    for j in range(n_out):
        if abs(R_next[j]) < 1e-12:
            continue  # no relevance to redistribute from this neuron

        if rule == 'lrp0':
            # z_j = Σᵢ aᵢ wᵢⱼ  (bias included in z_preact if provided)
            z_j = z_preact[j] if z_preact is not None else sum(
                a_prev[i] * W[j][i] for i in range(n_in))
            if abs(z_j) < 1e-10:
                continue  # skip near-zero denominator (instability)
            for i in range(n_in):
                R_prev[i] += (a_prev[i] * W[j][i] / z_j) * R_next[j]

        elif rule == 'lrpe':
            z_j = z_preact[j] if z_preact is not None else sum(
                a_prev[i] * W[j][i] for i in range(n_in))
            sign_z = 1.0 if z_j >= 0 else -1.0
            denom = z_j + eps * sign_z
            if abs(denom) < 1e-12:
                continue
            for i in range(n_in):
                R_prev[i] += (a_prev[i] * W[j][i] / denom) * R_next[j]

        elif rule == 'lrp_alpha_beta':
            # Separate positive and negative contributions
            pos_sum = sum(max(a_prev[i] * W[j][i], 0.0) for i in range(n_in))
            neg_sum = sum(min(a_prev[i] * W[j][i], 0.0) for i in range(n_in))
            # Add bias to respective sums
            if z_preact is not None:
                # Use actual z_j, derive bias contribution
                z_j = z_preact[j]
                b_contribution = z_j - sum(a_prev[i]*W[j][i] for i in range(n_in))
                pos_sum += max(b_contribution, 0.0)
                neg_sum += min(b_contribution, 0.0)
            for i in range(n_in):
                pos_c = max(a_prev[i] * W[j][i], 0.0)
                neg_c = min(a_prev[i] * W[j][i], 0.0)
                r_pos = alpha * (pos_c / (pos_sum + 1e-12)) * R_next[j]
                r_neg = beta  * (neg_c / (neg_sum - 1e-12)) * R_next[j]
                R_prev[i] += r_pos + r_neg

    return R_prev


def run_lrp(x_input, activations, z_all, WEIGHTS, BIASES, target_class,
            rule, eps=0.1, alpha=1.0, beta=0.0):
    """
    Run full LRP from output to input.
    Returns (R_input, layer_R_list, conservation_gaps).
    """
    n_layers = len(WEIGHTS)
    # Start: relevance = logit of target class at output
    R = [0.0, 0.0]
    R[target_class] = activations[-1][target_class]

    R_by_layer = [list(R)]
    gaps = []

    for l in range(n_layers - 1, -1, -1):
        R_new = lrp_propagate_layer(
            R, activations[l], WEIGHTS[l],
            rule=rule, eps=eps, alpha=alpha, beta=beta,
            z_preact=z_all[l]
        )
        gap = abs(sum(R_new) - sum(R)) / (abs(sum(R)) + 1e-12)
        gaps.append(gap)
        R = R_new
        R_by_layer.insert(0, list(R))

    return R, R_by_layer, gaps


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL FOUR RULES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  LRP ATTRIBUTIONS — FOUR RULES COMPARED")
print("=" * 68)

rule_configs = [
    ("LRP-0",     "lrp0",         dict(eps=0.0)),
    ("LRP-ε",     "lrpe",         dict(eps=0.25)),
    ("LRP-α₁β₀",  "lrp_alpha_beta", dict(alpha=1.0, beta=0.0)),
    ("LRP-α₂β₁",  "lrp_alpha_beta", dict(alpha=2.0, beta=1.0)),
]

all_results = {}
for name, rule, kw in rule_configs:
    R_in, R_layers, gaps = run_lrp(
        x_input, activations, z_all, WEIGHTS, BIASES, TARGET, rule, **kw)
    all_results[name] = (R_in, R_layers, gaps)

    f_x = activations[-1][TARGET]
    total_R = sum(R_in)
    conservation = abs(total_R - f_x) / (abs(f_x) + 1e-12)

    print()
    print(f"  ── {name} ─────────────────────────────────────────")
    print(f"     f(x) = {f_x:.4f}   ΣRᵢ = {total_R:.4f}   "
          f"conservation gap = {conservation:.2%}")
    print()
    print(f"     {'Feature':<18}  {'Relevance':>10}  {'% of f(x)':>10}  {'Bar'}")
    print(f"     {'-'*55}")
    for fname, r in zip(FEAT_NAMES, R_in):
        pct = r / (abs(f_x) + 1e-12) * 100
        bar_len = int(min(20, abs(r) / (abs(f_x) + 1e-12) * 20))
        bar = ("+" if r >= 0 else "-") * bar_len
        print(f"     {fname:<18}  {r:>+10.5f}  {pct:>+9.1f}%  {bar}")
    print(f"     {'Sum':>18}  {total_R:>+10.5f}")
    print(f"     Layer-wise conservation gaps: "
          f"{', '.join(f'{g:.2%}' for g in reversed(gaps))}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SIDE-BY-SIDE COMPARISON — ALL FOUR RULES")
print("=" * 68)
print()
print(f"  {'Feature':<18}  {'LRP-0':>9}  {'LRP-ε':>9}  "
      f"{'α₁β₀':>9}  {'α₂β₁':>9}")
print(f"  {'-'*60}")
for fi, fname in enumerate(FEAT_NAMES):
    vals = [all_results[n][0][fi] for n, _, _ in rule_configs]
    print(f"  {fname:<18}  " +
          "  ".join(f"{v:>+9.5f}" for v in vals))
print(f"  {'-'*60}")
print(f"  {'Sum':>18}  " +
      "  ".join(f"{sum(all_results[n][0]):>+9.5f}"
                for n, _, _ in rule_configs))

print()
print("  RANKING COMPARISON (by |relevance|):")
for name, _, _ in rule_configs:
    R_in = all_results[name][0]
    ranked = sorted(range(4), key=lambda i: -abs(R_in[i]))
    print(f"  {name:<12}: " +
          " > ".join(FEAT_NAMES[i][:8] for i in ranked))


# ─────────────────────────────────────────────────────────────────────────────
# LRP-0 INSTABILITY DEMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  LRP-0 INSTABILITY — Near-Zero Pre-Activation Demo")
print("=" * 68)
print()
print("  We construct a neuron with near-zero pre-activation z_j ≈ 0.001")
print("  where two inputs nearly cancel each other out.")
print()

# Artificial near-cancellation setup
a_test = [2.0, 1.0]   # activations from prev layer
W_test = [[1.0, -1.999]]  # weights: produce z_j ≈ 0.001
b_test = [0.0]
z_j_test = sum(a_test[i] * W_test[0][i] for i in range(2)) + b_test[0]
R_test = [0.5]  # relevance at this neuron

print(f"  Setup:")
print(f"    a₁ = {a_test[0]},  w₁ⱼ = {W_test[0][0]}")
print(f"    a₂ = {a_test[1]},  w₂ⱼ = {W_test[0][1]}")
print(f"    z_j = {a_test[0]}×{W_test[0][0]} + {a_test[1]}×{W_test[0][1]} = {z_j_test:.4f}  (near-zero!)")
print(f"    R_j = {R_test[0]}")
print()

# LRP-0 on the near-zero neuron
r0_lrp0 = lrp_propagate_layer(R_test, a_test, W_test, 'lrp0', z_preact=z_j_test and [z_j_test])
r0_lrpe = lrp_propagate_layer(R_test, a_test, W_test, 'lrpe', eps=0.1, z_preact=[z_j_test])
r0_ab   = lrp_propagate_layer(R_test, a_test, W_test, 'lrp_alpha_beta', alpha=1.0, beta=0.0,
                                z_preact=[z_j_test])

# Compute manually for clarity
print(f"  LRP-0:   R₁ = ({a_test[0]}×{W_test[0][0]}) / {z_j_test:.4f} × {R_test[0]}")
c1 = a_test[0]*W_test[0][0]
c2 = a_test[1]*W_test[0][1]
if abs(z_j_test) > 1e-10:
    r1_0 = (c1 / z_j_test) * R_test[0]
    r2_0 = (c2 / z_j_test) * R_test[0]
else:
    r1_0 = r2_0 = float('nan')
print(f"           = {c1:.1f} / {z_j_test:.4f} × {R_test[0]} = {r1_0:+.2f}")
print(f"           R₂ = {c2:.3f} / {z_j_test:.4f} × {R_test[0]} = {r2_0:+.2f}")
print(f"           ← ENORMOUS values from near-zero denominator!")
print(f"           Conservation: {r1_0:+.2f} + {r2_0:+.2f} = {r1_0+r2_0:.4f} = {R_test[0]} ✓")
print()
print(f"  LRP-ε (ε=0.1): denominator = {z_j_test:.4f} + 0.1 = {z_j_test+0.1:.4f}")
if abs(z_j_test + 0.1) > 1e-12:
    r1_e = (c1 / (z_j_test+0.1)) * R_test[0]
    r2_e = (c2 / (z_j_test+0.1)) * R_test[0]
    print(f"           R₁ = {c1:.1f}/{z_j_test+0.1:.4f}×{R_test[0]} = {r1_e:+.4f}")
    print(f"           R₂ = {c2:.3f}/{z_j_test+0.1:.4f}×{R_test[0]} = {r2_e:+.4f}")
    print(f"           ← Stable. Conservation absorbed: {abs(r1_e+r2_e - R_test[0]):.4f}")
print()
pos_sum = max(c1, 0.0) + max(c2, 0.0)
print(f"  LRP-α₁β₀: Only positive contributions:")
print(f"    z⁺ = max({c1:.1f},0) + max({c2:.3f},0) = {pos_sum:.4f}")
if pos_sum > 1e-12:
    print(f"    R₁ = max({c1:.1f},0)/{pos_sum:.4f}×{R_test[0]} = {max(c1,0)/pos_sum*R_test[0]:+.4f}")
    print(f"    R₂ = max({c2:.3f},0)/{pos_sum:.4f}×{R_test[0]} = {max(c2,0)/pos_sum*R_test[0]:+.4f}")
    print(f"    ← Stable. Conservation exact (only positive contrib sum used).")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE STRATEGY — DIFFERENT RULES AT DIFFERENT LAYERS
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  COMPOSITE LRP STRATEGY")
print("=" * 68)
print()
print("  Apply LRP-ε at classifier layer, LRP-α₁β₀ at hidden layers.")
print("  (Approximates the recommended composite strategy for real CNNs)")
print()

# Run composite: LRP-ε at last layer, LRP-α₁β₀ at all earlier layers
def run_composite_lrp(x_input, activations, z_all, WEIGHTS, BIASES, target_class):
    n_layers = len(WEIGHTS)
    R = [0.0] * len(activations[-1])
    R[target_class] = activations[-1][target_class]
    f_x = R[target_class]

    for l in range(n_layers - 1, -1, -1):
        if l == n_layers - 1:
            rule_used, kw = 'lrpe', dict(eps=0.25)
            rule_name = "LRP-ε"
        else:
            rule_used, kw = 'lrp_alpha_beta', dict(alpha=1.0, beta=0.0)
            rule_name = "LRP-α₁β₀"

        R_new = lrp_propagate_layer(
            R, activations[l], WEIGHTS[l],
            rule=rule_used, z_preact=z_all[l], **kw)
        gap = abs(sum(R_new) - sum(R)) / (abs(sum(R)) + 1e-12)
        print(f"  Layer {n_layers-l}: {rule_name:<12}  "
              f"ΣR_in={sum(R):.4f} → ΣR_out={sum(R_new):.4f}  "
              f"gap={gap:.2%}")
        R = R_new

    return R

R_composite = run_composite_lrp(
    x_input, activations, z_all, WEIGHTS, BIASES, TARGET)

f_x = activations[-1][TARGET]
print()
print(f"  Final composite attributions (f(x) = {f_x:.4f}):")
print()
print(f"  {'Feature':<18}  {'Composite':>11}  {'LRP-ε alone':>12}  {'α₁β₀ alone':>12}")
print(f"  {'-'*60}")
R_eps   = all_results["LRP-ε"][0]
R_alpha = all_results["LRP-α₁β₀"][0]
for fi, fname in enumerate(FEAT_NAMES):
    print(f"  {fname:<18}  {R_composite[fi]:>+11.5f}  "
          f"{R_eps[fi]:>+12.5f}  {R_alpha[fi]:>+12.5f}")
print(f"  {'Sum':>18}  {sum(R_composite):>+11.5f}  "
      f"{sum(R_eps):>+12.5f}  {sum(R_alpha):>+12.5f}")
print()
print(f"  KEY OBSERVATION:")
print(f"  The composite strategy produces attributions that blend the")
print(f"  stability of LRP-ε (at the classifier) with the sparsity and")
print(f"  exactness of LRP-α₁β₀ (at hidden layers). The conservation gap")
print(f"  is inherited from the LRP-ε step at the top layer only.")
print()
print(f"  In a real CNN, this means:")
print(f"  • Classifier FC weights → LRP-ε (handles large, signed weights)")
print(f"  • Late conv layers → LRP-α₁β₀ (semantic features, positive only)")
print(f"  • Early conv layers → LRP-α₂β₁ (both edges matter)")
print(f"  • Input layer → LRP-zᴮ (pixel bounds respected)")
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
    #     from interpretability.visuals.lrp import (
    #         LRP_VISUAL_HTML,
    #         LRP_VISUAL_HEIGHT,
    #     )
    #     visual_html   = LRP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = LRP_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[lrp.py] Could not load visual: {e}", stacklevel=2)

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