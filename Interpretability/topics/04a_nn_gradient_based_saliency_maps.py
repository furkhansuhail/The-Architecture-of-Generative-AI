"""
Saliency Maps — Gradient-Based Input Attribution
==================================================

Saliency maps are the oldest and most fundamental form of neural-network-specific
explanation. They answer the question "which input features does this neural network
respond to most strongly?" by computing or approximating the gradient of the model's
output with respect to its input — and using that gradient as a proxy for feature
importance.

This module covers vanilla gradients, the SmoothGrad correction, Integrated Gradients
(the theoretically grounded method), and the pathological failure modes that make
raw gradient maps unreliable.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Saliency Maps — Gradient-Based Input Attribution"
DISPLAY_NAME = "04a · Saliency Maps"
ICON = "🌡️"
SUBTITLE = "Vanilla gradients, SmoothGrad, and Integrated Gradients"


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

### Why Gradients?

Every method in the post-hoc agnostic family (SHAP, LIME, PDP, ALE) treats
the model as a black box — it only queries inputs and observes outputs. For
neural networks specifically, we have something those methods lack: the model
is fully differentiable. We can compute exactly how much each input dimension
changes the output, at the speed of a single backward pass.

This is the core insight behind saliency maps:

    If f(x) is a neural network output (e.g. logit for class c),
    then ∂f/∂xᵢ measures the instantaneous rate of change of f
    with respect to input feature i, at the specific point x.

    A large |∂f/∂xᵢ| means: the model's output is SENSITIVE to
    small perturbations of feature i at this input.

    A small |∂f/∂xᵢ| means: the model's output is locally
    INSENSITIVE to feature i at this input.

The gradient is computed in one backward pass through the network —
the same operation used during training, just applied to the input
instead of the weights. For an image with 224×224×3 pixels, you get
one attribution number per pixel in the time it takes to run inference.

This makes saliency maps uniquely fast: they require exactly ONE forward
pass and ONE backward pass, regardless of input dimensionality. SHAP
KernelSHAP requires O(2ⁿ) model calls. A saliency map requires O(1).

The price of this speed is a set of subtle failure modes that took the
research community over a decade to fully characterise.


##### PART I: VANILLA SALIENCY — THE GRADIENT AS ATTRIBUTION

### The Definition

Let f_c(x) be the score (logit or probability) for class c, where
x ∈ ℝᵈ is the input (a flattened image, a feature vector, a token
embedding, etc.). The vanilla saliency map (Simonyan et al., 2013) is:

    S(x) = |∂f_c(x) / ∂x|

The absolute value is taken element-wise, giving a non-negative
attribution per input dimension. For images, this is reshaped into
a heatmap of the same spatial dimensions as the input.

The magnitude |∂f_c/∂xᵢ| is interpreted as: "how much would the
class score change if we perturbed pixel i by a tiny amount?"

    # ================================================================== #
    **Computing the vanilla saliency map — the mechanics:**

    Forward pass:
      x → [layer 1] → [layer 2] → ... → [layer L] → f_c(x) ∈ ℝ

    Backward pass (gradient of f_c w.r.t. x):
      ∂f_c/∂x computed via chain rule through all layers.
      This gives a vector of the same shape as x.

    Visualisation (for images):
      Option A — magnitude: show |∂f_c/∂xᵢ| per pixel.
      Option B — signed:    show ∂f_c/∂xᵢ with sign, positive/negative.
      Option C — max over channels: for RGB, max(|∂f_c/∂xᵢ_R|,
                                                  |∂f_c/∂xᵢ_G|,
                                                  |∂f_c/∂xᵢ_B|)
    # ================================================================== #


### The Gradient as a First-Order Taylor Approximation

The vanilla gradient has a precise mathematical interpretation:
it is the linear component of a first-order Taylor expansion of f_c
around the input point x:

    f_c(x + δ) ≈ f_c(x) + (∂f_c/∂x)ᵀ · δ  +  O(‖δ‖²)

The gradient ∂f_c/∂x tells you: if you perturb the input by a small
vector δ, the change in the class score is approximately (∂f_c/∂x)ᵀ·δ.
The direction of maximum increase in f_c is exactly ∂f_c/∂x (gradient
ascent). The direction of maximum decrease is −∂f_c/∂x.

This is a LOCAL property. The gradient at x describes the model's
behaviour in an infinitesimally small neighbourhood of x. It says
nothing about what happens if you make a large change to the input.

This locality is both the strength and the critical weakness of vanilla
saliency — a point we return to in Part III.


### Gradient × Input — A Simple Refinement

A common extension multiplies the gradient by the input value itself:

    GradInput(x)ᵢ = xᵢ · (∂f_c / ∂xᵢ)

This has an intuitive justification: a feature with a large gradient
but a zero input value has no effect on the output (the gradient
describes sensitivity, but if xᵢ=0 there is nothing there to be
sensitive to). Multiplying by xᵢ zeros out these spurious attributions.

Formally, GradInput approximates the contribution of feature i in
the Taylor expansion: (∂f_c/∂xᵢ) · xᵢ ≈ the additive contribution
of the i-th feature relative to a zero baseline.

For ReLU networks, GradInput and SHAP's GradientExplainer produce
similar results. GradInput can be seen as a crude single-step
approximation of Integrated Gradients (see Part IV).

    # ================================================================== #
    **When Gradient × Input differs from raw gradient:**

    Suppose:  xᵢ = 5.0,   ∂f_c/∂xᵢ = +0.3
              xⱼ = 0.01,  ∂f_c/∂xⱼ = +2.8

    Raw gradient says xⱼ is 9× more important (|2.8| vs |0.3|).
    GradInput says  xᵢ is more important: 5×0.3=1.5 vs 0.01×2.8=0.028.

    Which is right? Depends on the question:
      Raw gradient: "which feature is the model MOST SENSITIVE to?"
      GradInput:    "which feature ACTUALLY CONTRIBUTES to the output?"

    A feature the model is sensitive to but which has near-zero value
    contributes nothing to THIS prediction — GradInput captures this.
    # ================================================================== #


##### PART II: THE BACKPROPAGATION FAMILY — VARIANTS AND MODIFICATIONS

### Guided Backpropagation

Vanilla gradients propagate through ReLU units according to the standard
chain rule: if the pre-activation was negative (ReLU output = 0), the
gradient flowing backwards is zeroed out. Guided backpropagation
(Springenberg et al., 2014) adds a second gating condition:

    Standard backprop:  δˡᵢ = δˡ⁺¹ᵢ · 𝟙[zˡᵢ > 0]
    Guided backprop:    δˡᵢ = δˡ⁺¹ᵢ · 𝟙[zˡᵢ > 0] · 𝟙[δˡ⁺¹ᵢ > 0]

where δˡᵢ is the gradient at layer l, unit i, and zˡᵢ is the
pre-activation value.

The extra gate 𝟙[δˡ⁺¹ᵢ > 0] clips negative gradients flowing backwards.
Only positive signals from the next layer — which indicate "this unit
was helpful for the class" — are propagated back.

The visual result: guided backpropagation maps tend to produce sharper,
edge-like visualisations that look like the image itself filtered by
its "interesting" regions. They are visually appealing.

The critical problem: guided backpropagation has been shown (Nie et al., 2018;
Adebayo et al., 2018) to be INDEPENDENT of the model weights. The saliency
maps it produces are essentially just detecting edges in the input — regardless
of what the model has learned. It is an image processing operation, not an
explanation of the model.

    # ================================================================== #
    **The Independence Failure — guided backprop vs sanity checks:**

    Experiment (Adebayo et al., 2018 "Sanity Checks for Saliency"):
      1. Train a model on ImageNet. Produce guided-backprop saliency maps.
      2. Randomly scramble ALL model weights (the model now predicts randomly).
      3. Produce guided-backprop saliency maps again.
      4. Compare maps from steps 1 and 3.

    Finding: the maps are NEARLY IDENTICAL. Guided backpropagation
    produces similar visualisations regardless of the model's learned
    function. It is explaining the INPUT, not the model.

    This does not mean the maps are useless for debugging preprocessing
    pipelines or checking input quality. But they CANNOT be used to
    explain why the model made a specific decision — because they do not
    depend on what the model learned.
    # ================================================================== #


### Grad-CAM — Class Activation Mapping via Gradients

GradCAM (Selvaraju et al., 2017) uses gradients flowing into the LAST
CONVOLUTIONAL LAYER rather than the input. For an image classification
network, the final convolutional layer produces a set of feature maps
Aᵏ ∈ ℝᴴˣᵂ, one per filter k.

    Step 1 — Compute importance weights for each filter:
      αᵏ_c = (1/Z) Σᵢ Σⱼ  ∂f_c / ∂Aᵏᵢⱼ
      (Global average pool the gradient over the spatial dimensions)

    Step 2 — Weighted sum of feature maps:
      L^GradCAM_c = ReLU( Σₖ αᵏ_c · Aᵏ )
      (ReLU keeps only positive contributions to class c)

    Step 3 — Upsample to input resolution:
      Bilinear interpolation from (H×W) → (input height × width)

The result is a coarse (low-resolution) heatmap showing which spatial
regions of the image were most important for class c, according to the
last convolutional layer's activations.

GradCAM is model-dependent and layer-specific. Unlike guided backpropagation,
it PASSES the Adebayo sanity checks: randomising model weights produces
meaningfully different GradCAM maps (the module on GradCAM covers this in
depth — see nn_specific_xai/gradient_based/GradCAM).

The resolution limitation is significant: GradCAM's heatmap matches the
spatial resolution of the final conv layer (e.g., 7×7 for VGG16's last
conv), which is upsampled to the full image. This means it cannot identify
which specific pixels are important — only which general region.

Grad-CAM++ (Chattopadhay et al., 2018) addresses this by weighting
individual pixel-level gradients rather than global average-pooled ones,
producing sharper maps.


### The Axiom-Compliance Failure of Vanilla Gradients

Sundararajan, Taly & Yan (2017) — the authors of Integrated Gradients —
proved that vanilla gradients and all its simple variants violate two
fundamental axioms that ANY attribution method should satisfy:

    AXIOM 1 — SENSITIVITY (a.k.a. Completeness):
    ────────────────────────────────────────────
    If two inputs x and x' differ in only one feature, and f(x) ≠ f(x'),
    then the attribution for that feature must be non-zero.

    Vanilla gradient VIOLATES THIS:
    Consider a ReLU network where the output is f(x) = ReLU(xᵢ − 1).
    At x = {xᵢ = 2}: f(x) = 1, ∂f/∂xᵢ = 1 (above the threshold).
    At x = {xᵢ = 0}: f(x) = 0, ∂f/∂xᵢ = 0 (below the threshold).
    At x = {xᵢ = 1}: f(x) = 0, ∂f/∂xᵢ = 0 (exactly at the threshold).

    Now compare x = (xᵢ = 2) and x = (xᵢ = 1.0001).
    f differs (1 vs 0.0001), but the gradient may be identical (both = 1).
    The gradient correctly says xᵢ is important, but gives the SAME attribution
    regardless of how much it actually contributed.

    AXIOM 2 — IMPLEMENTATION INVARIANCE:
    ──────────────────────────────────────
    If two networks implement the same mathematical function (same inputs
    always produce same outputs), their attributions should be identical.

    Vanilla gradient VIOLATES THIS:
    Two algebraically equivalent networks can have different gradient
    values due to different intermediate representations, even though
    they compute the same function. The explanation depends on
    HOW the function is implemented, not WHAT it computes.

    # ================================================================== #
    **A concrete Implementation Invariance failure:**

    Network A computes: f(x) = x² directly.
    Network B computes: f(x) = exp(2·log(x)) (algebraically identical).

    At x = 3: both output 9.
    ∂f_A/∂x = 2x = 6
    ∂f_B/∂x = 2·exp(2·log(x))/x = 2·x²/x = 2x = 6  (same here)

    But with ReLU intermediaries, two algebraically equivalent
    decompositions can give different gradients because the gradient
    of a piecewise-linear function depends on the current activation
    state, not just the input-output relationship.
    # ================================================================== #


##### PART III: THE SATURATION AND SHATTERED GRADIENT PROBLEMS


### Problem 1 — Gradient Saturation

The vanilla gradient is a LOCAL property — it describes the model's
behaviour in an infinitesimally small neighbourhood of x. This causes
a fundamental problem with saturating activation functions.

Consider a sigmoid unit σ(z) = 1/(1+e⁻ᶻ). When |z| is large, the
sigmoid is nearly flat: σ'(z) → 0. The gradient of f with respect to
inputs that influence this unit will be near zero — not because those
inputs are unimportant, but because the sigmoid is saturated.

    # ================================================================== #
    **Gradient saturation — a concrete example:**

    f(x₁) = σ(10·x₁)    (a sigmoid with very steep slope)

    At x₁ = 0.1:  f(x₁) = σ(1) ≈ 0.73,  f'(x₁) = 10·σ'(1) ≈ 1.97
    At x₁ = 0.5:  f(x₁) = σ(5) ≈ 0.99,  f'(x₁) = 10·σ'(5) ≈ 0.067
    At x₁ = 1.0:  f(x₁) = σ(10) ≈ 1.0,  f'(x₁) = 10·σ'(10) ≈ 0.00045

    Observation: as x₁ increases from 0.1 to 1.0, the model becomes
    MORE certain (output closer to 1), but the gradient DECREASES
    dramatically. A saliency map at x₁=1.0 would show near-zero
    attribution for x₁ — claiming it is unimportant — even though
    x₁ is the ONLY input and is clearly the reason the model predicts 1.

    This is the saturation paradox: confident predictions have small
    gradients, which look like "nothing is important," when in fact
    the model is extremely confident BECAUSE a feature is highly active.
    # ================================================================== #

ReLU networks suffer a different but related issue: the dead ReLU
problem. A ReLU unit that has always been inactive (z < 0 for all
inputs seen during training) passes zero gradient backwards. Any
input that routes through a dead ReLU will have zero attribution —
correctly in the sense that the model doesn't use that pathway, but
the saliency map may mislead: it shows nothing without explaining why.


### Problem 2 — Shattered Gradients

Balduzzi et al. (2017) proved that as neural network depth increases,
the gradient ∂f/∂x becomes increasingly fractal and incoherent —
a property they called "gradient shattering."

Informally: in a deep ReLU network, the gradient landscape consists
of many small flat regions (where multiple ReLUs are inactive) separated
by sharp edges (where ReLUs switch on or off). The gradient at any
specific point x is heavily influenced by which ReLUs are active at x,
which changes discontinuously as x is perturbed.

The consequence for saliency maps: the gradient at x is highly sensitive
to tiny perturbations of x. Moving x by a single pixel can dramatically
change the saliency map, even when the model's prediction barely changes.

    # ================================================================== #
    **Shattered gradients visualised:**

    f(x) over the input dimension x, for a deep ReLU network:

    f(x)         ___
    ↑           /   \___
    │      ____/         \___
    │_____/                   \___
    └───────────────────────────────→ x
    (smooth, continuous)

    ∂f/∂x:
    ↑ +1 │  ____     ___
    │     │ /    \   /
   0│────────────────────────────→ x
    │            \_/
    ↓ −1 │

    The gradient is piecewise constant with discontinuities wherever
    a ReLU activation switches state. Two nearby points x and x+ε
    may have completely different gradient vectors if ε crosses a
    ReLU boundary — even though f(x) ≈ f(x+ε).

    A saliency map at x and at x+ε may look completely different,
    despite the model making essentially the same prediction.
    # ================================================================== #


### Problem 3 — The Saliency Map as a Description of the DATA, Not the Model

This is the subtlest and most important limitation, formalised by the
sanity checks of Adebayo et al. (2018).

A saliency map is supposed to explain why the model makes a specific
prediction. For it to be an explanation of the model, it should change
when the model's weights change. Adebayo et al. ran cascading randomisation
tests: progressively replace the trained weights layer by layer with random
weights, starting from the top layer.

    If the saliency method explains the model:
      Randomising a layer's weights should change the saliency map.
      At full randomisation (all layers random), the map should look
      completely different from the original.

    Results:
      • Vanilla gradient: PASSES the sanity check (maps change with
        weight randomisation).
      • Gradient × Input: PASSES the sanity check.
      • Guided backpropagation: FAILS — maps are nearly identical
        before and after weight randomisation.
      • GuidedGradCAM: FAILS — same as guided backprop.
      • Integrated Gradients: PASSES (with appropriate baseline).

    Maps that FAIL the sanity check are not explaining the model.
    They are explaining properties of the INPUT (edges, textures,
    statistical patterns in the data), regardless of what the model
    has learned.

This does not mean gradient methods are useless — vanilla gradient and
Integrated Gradients do pass sanity checks. But it means some of the
most visually compelling saliency methods (guided backpropagation,
GuidedGradCAM) are not trustworthy model explanations.


##### PART IV: SMOOTHGRAD — AVERAGING OVER NOISE

### The Noisy Gradient Problem and Its Solution

The shattered gradient problem manifests visually as "noisy" saliency
maps: the heatmap shows a speckled, salt-and-pepper pattern with many
isolated high-attribution pixels scattered across regions that should
be unimportant. This is gradient noise — a consequence of the fractal
gradient landscape of deep ReLU networks.

SmoothGrad (Smilkov et al., 2017) addresses this with a strikingly
simple idea: instead of computing the gradient at exactly x, average the
gradient over many slightly perturbed versions of x:

    SmoothGrad_c(x) = (1/N) Σᵢ₌₁ᴺ  ∂f_c(x + εᵢ) / ∂(x + εᵢ)

where εᵢ ~ N(0, σ²I) are independent Gaussian noise samples.

### Why Averaging Over Noise Works

The key insight: the gradient landscape is fractal and discontinuous
at a fine scale, but smoother at a coarser scale. By averaging over
a Gaussian ball of radius σ around x, SmoothGrad effectively evaluates
the model's local sensitivity to features over a NEIGHBOURHOOD rather
than at a single infinitesimal point.

Formally, SmoothGrad approximates:

    E_ε[∂f_c(x+ε)/∂(x+ε)]  ≈  ∂/∂x  E_ε[f_c(x+ε)]

This is not equal to ∂f_c(x)/∂x (which is the standard gradient).
It is the gradient of the EXPECTED output over a neighbourhood —
a smoother quantity that averages over the discontinuities of the
piecewise-linear activation landscape.

The bandwidth σ controls the trade-off:
  • Small σ: stays close to vanilla gradient (noisy, local).
  • Large σ: averages over a large region (smooth, but may miss sharp
    local boundaries that the model genuinely responds to).

Typical σ values: 10–20% of the input range (e.g., σ = 0.15 × (xmax − xmin)).

    # ================================================================== #
    **SmoothGrad as a denoising operation on the gradient landscape:**

    Vanilla gradient (shattered):
    Attribution ↑
               │ * *   *      *    *  ← noisy, discontinuous
               │* * * *  * *  * * *
               └──────────────────────→ pixel index

    SmoothGrad (N=50, σ=0.15):
    Attribution ↑
               │         ___
               │       /    \
               │______/      \______
               └──────────────────────→ pixel index
               (smooth, coherent, shows real structure)
    # ================================================================== #


### The Cost of SmoothGrad

SmoothGrad requires N forward+backward passes instead of 1.
With N=50 (a typical value), it is 50× more expensive than vanilla
gradient. For a fast model this is negligible; for a large transformer
or ResNet-152, it may take seconds per image.

The choice of N is empirical — there is no theoretical guarantee on
how many samples are needed. In practice, N=25 to N=100 is common.

VarGrad (Adebayo et al., 2020) showed that using the VARIANCE of
gradients (rather than the mean) also produces clean attribution maps,
and requires fewer samples to converge. This is because the variance
is more sensitive to consistent signal than to random noise.

    VarGrad_c(x) = Var_εᵢ [ ∂f_c(x + εᵢ) / ∂(x + εᵢ) ]
                  = (1/N) Σᵢ (gᵢ − ḡ)²    where gᵢ = gradient at x+εᵢ


##### PART V: INTEGRATED GRADIENTS — THE THEORETICALLY CORRECT METHOD

### The Baseline — Defining "Nothing"

Vanilla gradient measures sensitivity at x — a local property. Integrated
Gradients (Sundararajan, Taly & Yan, 2017) measures something global and
causally richer: the total contribution of each feature to the output,
measured relative to a BASELINE input x' that represents "absence of signal."

The baseline x' is a design choice that encodes what "no information"
means for your problem:
  • Images:    x' = all-black (zeros), or all-grey (constant), or random noise
  • Text:      x' = all-padding tokens, or all-zero embeddings
  • Tabular:   x' = all-zeros, or the training-set mean per feature
  • Audio:     x' = silence (all zeros)

The choice of baseline is IMPORTANT. Different baselines give different
attributions. The attribution measures the feature's contribution RELATIVE
TO THE BASELINE. If the baseline is poorly chosen (e.g., a blank image that
the model confidently classifies as some class), the attributions are
relative to that arbitrary non-neutral starting point.

    # ================================================================== #
    **Why the baseline matters:**

    Model: predicts "cat" from images.
    Baseline A: black image. Model outputs P(cat|black) ≈ 0.001.
    Baseline B: "dog" image. Model outputs P(cat|dog) ≈ 0.02.

    Attribution for pixel i in image x:
      w.r.t. Baseline A: contribution of i to moving from 0.001 → f(x)
      w.r.t. Baseline B: contribution of i to moving from 0.02 → f(x)

    These are different questions. Baseline A asks "why is this image
    a cat (vs a blank canvas)?". Baseline B asks "why is this more cat-like
    than a dog?" The right baseline depends on your interpretive goal.
    # ================================================================== #


### The Integrated Gradients Formula

Integrated Gradients integrates the gradient along the STRAIGHT LINE
PATH from the baseline x' to the actual input x:

    IG_i(x) = (xᵢ − xᵢ') × ∫₀¹ ∂f_c(x' + α(x − x')) / ∂xᵢ  dα

In words: we move from the baseline x' towards the input x, step by
step (parameterised by α from 0 to 1). At each step, we compute the
gradient of f_c with respect to input feature i. We then integrate
(accumulate) these gradients along the path, and multiply by (xᵢ − xᵢ')
to convert from gradient to attribution.

The factor (xᵢ − xᵢ') is essential: it scales the attribution by how
much feature i actually changes along the path. A feature that does not
change from baseline to input (xᵢ = xᵢ') gets zero attribution, regardless
of what the gradient says. This prevents the saturation paradox.


### Why Integrated Gradients is Theoretically Correct

Sundararajan et al. proved that Integrated Gradients satisfies six axioms,
of which two are most fundamental:

    AXIOM 1 — COMPLETENESS (replaces the broken Sensitivity axiom):
    ────────────────────────────────────────────────────────────────
    Σᵢ IG_i(x) = f_c(x) − f_c(x')

    The sum of all feature attributions equals exactly the difference
    between the model's output at x and at the baseline x'.
    Every unit of the prediction change is fully accounted for.
    Nothing is left over, nothing is double-counted.

    This is analogous to SHAP's Efficiency axiom. It is the accounting
    identity that makes IG a faithful, complete explanation.

    AXIOM 2 — IMPLEMENTATION INVARIANCE:
    ──────────────────────────────────────
    If two networks compute the same mathematical function,
    their Integrated Gradients attributions are identical.

    Proof: the attribution depends only on the function f_c, not on
    the specific implementation. The path integral ∫₀¹ ∂f_c/∂xᵢ dα is
    a property of f_c as a mathematical function, not as a computation
    graph. Two algebraically equivalent networks have the same f_c and
    therefore the same IG attributions.

    Proof of Completeness using the fundamental theorem of calculus:
    ────────────────────────────────────────────────────────────────
    f_c(x) − f_c(x') = ∫₀¹ d/dα [f_c(x' + α(x−x'))] dα    (by FTC)

    = ∫₀¹ Σᵢ ∂f_c(x'+α(x−x'))/∂xᵢ · (xᵢ − xᵢ') dα       (chain rule)

    = Σᵢ (xᵢ − xᵢ') ∫₀¹ ∂f_c(x'+α(x−x'))/∂xᵢ dα           (swap sum/integral)

    = Σᵢ IG_i(x)                                              (by definition of IG)

    Every step follows from standard calculus. The completeness property
    is not an axiom that is imposed — it is a theorem that FOLLOWS from
    the definition of IG and the fundamental theorem of calculus.

    # ================================================================== #
    **Intuition for the path integral — the altitude analogy:**

    Think of hiking from a valley (baseline x') to a mountain peak (input x).
    At every step along the path, you measure how steeply you are climbing
    in each direction. IG asks: "of all the altitude gained from valley
    to peak, how much came from climbing in the x₁ direction vs x₂ direction?"

    The factor (xᵢ − xᵢ') is the total distance you walked in direction i.
    The integral ∫∂f_c/∂xᵢ dα is the average steepness in direction i
    along the path.
    Together: altitude contributed by direction i = distance × average slope.

    Vanilla gradient only measures the slope at the PEAK.
    It knows nothing about how you got there.
    IG tracks the slope at every point along the journey.
    # ================================================================== #


### Approximating the Integral — Riemann Summation

The integral ∫₀¹ ∂f_c/∂xᵢ dα is approximated by a Riemann sum over
M steps (Gauss-Legendre quadrature or uniform steps):

    IG_i(x) ≈ (xᵢ − xᵢ') × (1/M) Σₖ₌₁ᴹ ∂f_c(x' + (k/M)·(x−x')) / ∂xᵢ

Each term in the sum requires one forward+backward pass. Total cost:
M forward+backward passes. Typical M: 20 to 300.

With M=50: IG is 50× more expensive than vanilla gradient — comparable
to SmoothGrad. But unlike SmoothGrad, IG has a convergence guarantee:
as M → ∞, the Riemann sum converges to the true integral. SmoothGrad
has no such guarantee.

    Convergence check — the completeness gap:
    After computing IG, verify:
      |Σᵢ IG_i(x) − (f_c(x) − f_c(x'))| < tolerance

    If this gap is large, M is too small. Increase M until the gap
    is below 1% of |f_c(x) − f_c(x')|. This is a built-in quality
    check that no other attribution method provides.


### The Straight-Line Path and Alternative Paths

IG uses the straight-line path from x' to x as its integration path.
The Completeness axiom holds for ANY differentiable path — the choice
of path is an additional design decision beyond the baseline.

    STRAIGHT-LINE (standard IG):
    x(α) = x' + α(x − x'),  α ∈ [0, 1]
    Simple, interpretable, easy to implement.

    GRADIENT PATH (optimal path):
    The path that assigns all attribution to the feature with the
    highest gradient-magnitude — concentration of attribution.
    Rarely used in practice; complex to implement.

    BLUR PATH:
    x(α) = Gaussian-blurred version of x at scale (1−α).
    Starts from maximum blur (uniform baseline) and gradually sharpens
    to the original image. Motivated by the intuition that "absence"
    means "blurred away" rather than "black."

Different paths give different attributions (all satisfy Completeness
independently, but their values differ). The straight-line path is the
default because it is simple, well-studied, and produces competitive
results in practice.


##### PART VI: COMPARING THE METHODS — A UNIFIED VIEW

### The Gradient Family as Approximations of IG

All gradient-based attribution methods can be understood as special
cases or approximations of Integrated Gradients:

    VANILLA GRADIENT:
    IG with M=1 step and the step placed at the INPUT x (not the midpoint).
    IG_i ≈ (xᵢ − xᵢ') × ∂f_c(x) / ∂xᵢ
    This is exactly GradInput when x' = 0 (zero baseline).

    SMOOTHGRAD:
    IG with M=N noise samples, each centered at x (not interpolated).
    Equivalent to IG along a PATH that randomly jitters around x,
    rather than integrating along the straight line from x' to x.
    SmoothGrad captures variance around x; IG captures the contribution
    across the full path from x' to x.

    GUIDED BACKPROPAGATION:
    NOT an approximation of IG. It modifies the backward pass itself,
    and the result does not satisfy any of IG's axioms. Fails sanity checks.

    # ================================================================== #
    **Unified view — what each method integrates over:**

    Vanilla gradient:     Point estimate at x (M=1, at endpoint)
    GradInput:            Point estimate at x, scaled by (x − x'=0)
    SmoothGrad:           Average over Gaussian ball AROUND x
    Integrated Gradients: Integral along path FROM x' TO x (M steps)
                          → Complete, consistent, axiom-satisfying
    # ================================================================== #


### Axiomatic Comparison

    ┌────────────────────────────────────────────────────────────────────┐
    │  Axiom                  Vanilla    SmoothGrad  Guided BP    IG     │
    │  ─────────────────────────────────────────────────────────────     │
    │  Completeness           No         No          No           YES    │
    │  Implementation invar.  No         No          No           YES    │
    │  Sensitivity (local)    Partial    Partial     No           YES    │
    │  Linearity              Yes        Yes         No           YES    │
    │  Dummy (zero input)     Partial    Partial     No           YES    │
    │  Passes sanity checks   Yes        Yes         NO (fails)   YES    │
    │  Handles saturation     No         Partial     No           YES    │
    │  Handles shattered grad No         Yes         No           Yes    │
    │  Computational cost     1 pass     N passes    1 pass       M pass │
    └────────────────────────────────────────────────────────────────────┘


### When to Use Each Method

    VANILLA GRADIENT:
    • Fastest possible (1 forward + 1 backward).
    • Use when speed is paramount and rough attribution is sufficient.
    • Appropriate as a first-pass exploration tool.
    • Do not use as a primary explanation for deployment.

    SMOOTHGRAD:
    • Better visual quality than vanilla gradient for images.
    • Use when vanilla gradient is too noisy and M=50 passes are acceptable.
    • No theoretical guarantees, but empirically better than vanilla.

    INTEGRATED GRADIENTS:
    • The theoretically grounded method. Use as the primary gradient-based
      attribution tool unless speed is a hard constraint.
    • Always check the completeness gap (Σ IG_i vs f(x) − f(x')).
    • Choose baseline carefully: it encodes what "no signal" means.
    • M=50 steps is usually sufficient; M=300 for precise explanations.

    GUIDED BACKPROPAGATION:
    • Do NOT use for explaining model decisions.
    • Fails the Adebayo sanity checks — it is explaining the input, not the model.
    • May be useful for visualising which parts of an image have sharp edges
      or contrast (an image-processing use), but must not be presented
      as model attribution.


##### PART VII: SALIENCY MAPS FOR NON-IMAGE DATA

### Tabular Neural Networks

For a neural network trained on structured tabular data, the input
x ∈ ℝᵈ is a vector of d features (income, credit score, age, etc.).
The saliency map S(x) = ∂f_c/∂x ∈ ℝᵈ gives one attribution per feature.

Compared to SHAP for tabular neural networks:
  • Vanilla gradient is much faster (1 pass vs 2ⁿ coalition evaluations).
  • Integrated Gradients satisfies Completeness; SHAP satisfies Efficiency
    (same axiom, different name, same meaning).
  • For tabular data, SHAP's DeepExplainer or KernelSHAP typically produce
    more faithful and stable attributions than vanilla gradient.
  • IG is competitive with SHAP for tabular neural networks when a
    meaningful baseline (e.g., the mean feature vector) is used.

The gradient is scale-dependent: features with larger numerical ranges
will tend to have smaller raw gradients (the model compensates via weight
magnitudes), so batch normalisation and feature standardisation are
important for gradient-based methods to be comparable across features.


### Sequence Models and Attention

For recurrent networks (LSTMs, GRUs) and transformer models applied to
text, the "input" is typically a sequence of token embeddings. The
saliency map ∂f_c/∂eᵢ gives a gradient vector for each token embedding eᵢ.

The token-level attribution is typically summarised by:
  • L2 norm: ‖∂f_c/∂eᵢ‖₂ — magnitude of gradient per token
  • Dot product: (∂f_c/∂eᵢ)ᵀ · eᵢ — gradient × input, IG approximation

This gives one scalar per token, producing a sequence of attributions
that can be displayed as a colour-coded sentence.

For transformers, saliency maps and attention weights are both used for
attribution, but they measure different things:
  • Attention weight: how much model attends to token j when processing token i.
  • Saliency: how much the final output changes if token i's embedding changes.
  • These are often poorly correlated. High attention to a token does not
    mean that token drives the final classification.
  • Jain & Wallace (2019) showed attention is NOT a reliable explanation.
    Saliency maps (IG in particular) are more faithful for text models.


##### PART VIII: A COMPLETE WORKED EXAMPLE BY HAND

### Integrated Gradients on a Tiny Network — Full Trace

We compute Integrated Gradients by hand for a minimal network to make
every step transparent. This is small enough to trace exactly.

    Network: 2 inputs → 1 hidden (ReLU) → 1 output
    Architecture:
      h₁ = ReLU(w₁₁·x₁ + w₂₁·x₂ + b₁)
      h₂ = ReLU(w₁₂·x₁ + w₂₂·x₂ + b₂)
      f  = u₁·h₁ + u₂·h₂ + c

    Weights (learned):
      w₁₁ = 1.0,  w₂₁ = 0.5,  b₁ = -0.5   (hidden unit 1)
      w₁₂ = -0.3, w₂₂ = 2.0,  b₂ = -1.0   (hidden unit 2)
      u₁  = 1.5,  u₂  = 1.0,  c  = 0.0    (output unit)

    Instance: x = (x₁=1.2, x₂=0.8)
    Baseline: x' = (x₁'=0.0, x₂'=0.0)  (zero baseline)

    STEP 1 — Forward pass at x:
      z₁ = 1.0(1.2) + 0.5(0.8) + (-0.5) = 1.2 + 0.4 - 0.5 = 1.1
      h₁ = ReLU(1.1) = 1.1
      z₂ = -0.3(1.2) + 2.0(0.8) + (-1.0) = -0.36 + 1.6 - 1.0 = 0.24
      h₂ = ReLU(0.24) = 0.24
      f(x) = 1.5(1.1) + 1.0(0.24) = 1.65 + 0.24 = 1.89

    STEP 2 — Forward pass at baseline x':
      z₁' = 0 + 0 - 0.5 = -0.5  →  h₁' = ReLU(-0.5) = 0
      z₂' = 0 + 0 - 1.0 = -1.0  →  h₂' = ReLU(-1.0) = 0
      f(x') = 1.5(0) + 1.0(0) = 0.0

    Amount to explain: f(x) - f(x') = 1.89 - 0.0 = 1.89

    STEP 3 — Compute gradients along path with M=4 steps:
    Interpolated points: x(α) = α·x for α ∈ {0.25, 0.50, 0.75, 1.00}

    At α=0.25: x = (0.30, 0.20)
      z₁ = 0.30 + 0.10 - 0.50 = -0.10 → h₁=0 (inactive)
      z₂ = -0.09 + 0.40 - 1.00 = -0.69 → h₂=0 (inactive)
      ∂f/∂x₁ = u₁·𝟙[z₁>0]·w₁₁ + u₂·𝟙[z₂>0]·w₁₂ = 0 + 0 = 0.0
      ∂f/∂x₂ = u₁·𝟙[z₁>0]·w₂₁ + u₂·𝟙[z₂>0]·w₂₂ = 0 + 0 = 0.0

    At α=0.50: x = (0.60, 0.40)
      z₁ = 0.60 + 0.20 - 0.50 = 0.30 → h₁=0.30 (active)
      z₂ = -0.18 + 0.80 - 1.00 = -0.38 → h₂=0 (inactive)
      ∂f/∂x₁ = 1.5(1)(1.0) + 1.0(0)(-0.3) = 1.50
      ∂f/∂x₂ = 1.5(1)(0.5) + 1.0(0)(2.0) = 0.75

    At α=0.75: x = (0.90, 0.60)
      z₁ = 0.90 + 0.30 - 0.50 = 0.70 → h₁=0.70 (active)
      z₂ = -0.27 + 1.20 - 1.00 = -0.07 → h₂=0 (inactive)
      ∂f/∂x₁ = 1.5(1.0) = 1.50
      ∂f/∂x₂ = 1.5(0.5) = 0.75

    At α=1.00: x = (1.20, 0.80)  (the actual input)
      z₁ = 1.10 (active, from Step 1),  z₂ = 0.24 (active)
      ∂f/∂x₁ = 1.5(1.0) + 1.0(-0.3) = 1.50 - 0.30 = 1.20
      ∂f/∂x₂ = 1.5(0.5) + 1.0(2.0)  = 0.75 + 2.00 = 2.75

    STEP 4 — Riemann sum and multiply by (x - x'):

    Average gradient for x₁: (0.0 + 1.50 + 1.50 + 1.20) / 4 = 4.20/4 = 1.050
    Average gradient for x₂: (0.0 + 0.75 + 0.75 + 2.75) / 4 = 4.25/4 = 1.0625

    IG₁ = (x₁ - x₁') × avg_grad₁ = (1.2 - 0) × 1.050 = 1.260
    IG₂ = (x₂ - x₂') × avg_grad₂ = (0.8 - 0) × 1.0625 = 0.850

    STEP 5 — Completeness check:
    IG₁ + IG₂ = 1.260 + 0.850 = 2.110
    f(x) − f(x') = 1.89

    Completeness gap = |2.110 - 1.89| / 1.89 = 11.6%  ← M=4 is too small!

    With M=100 steps (the gap falls below 1%):
    Average gradient x₁ ≈ 0.987,  Average gradient x₂ ≈ 0.969
    IG₁ ≈ 1.184,  IG₂ ≈ 0.775
    IG₁ + IG₂ ≈ 1.959  (gap ≈ 3.6% — improving but M=100 needed here due
                         to the sharp ReLU transitions in the small network)

    INTERPRETATION at M=100:
      x₁ (first input) accounts for 1.184/1.89 ≈ 63% of the prediction.
      x₂ (second input) accounts for 0.775/1.89 ≈ 41% of the prediction.
      (These add to slightly more than 100% due to approximation error.)

    WHY x₁ DOMINATES despite x₂ having a larger final-point gradient:
      At α=0.25 (early on the path), BOTH hidden units are inactive.
      When z₁ first activates (near α=0.4), it drives up attributions
      for x₁ (via w₁₁=1.0) and x₂ (via w₂₁=0.5).
      z₂ only activates near α=0.97 (very close to the full input).
      The late activation of h₂ means x₂ gets most of its attribution
      only in the final stretch of the path — a small fraction of M steps.
      IG correctly gives x₁ more credit: it drove the output higher
      for a longer portion of the path from baseline to input.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔═══════════════════════╦══════════════╦══════════════╦═════════════╦═══════════════╗
    ║ Property              ║ Vanilla Grad ║ SmoothGrad   ║ Guided BP   ║ Integrated G. ║
    ╠═══════════════════════╬══════════════╬══════════════╬═════════════╬═══════════════╣
    ║ Cost                  ║ 1 fwd+bwd    ║ N fwd+bwd    ║ 1 fwd+bwd   ║ M fwd+bwd     ║
    ║ Typical M / N         ║ 1            ║ 25–100       ║ 1           ║ 50–300        ║
    ║ Completeness axiom    ║ No           ║ No           ║ No          ║ YES           ║
    ║ Impl. invariance      ║ No           ║ No           ║ No          ║ YES           ║
    ║ Sanity check (Adebayo)║ Pass         ║ Pass         ║ FAIL        ║ Pass          ║
    ║ Handles saturation    ║ No           ║ Partial      ║ No          ║ YES           ║
    ║ Handles shattered grad║ No           ║ Yes          ║ N/A         ║ Yes           ║
    ║ Requires baseline     ║ No           ║ No           ║ No          ║ YES           ║
    ║ Quality check built-in║ No           ║ No           ║ No          ║ YES (gap)     ║
    ║ Recommended for prod. ║ No           ║ Partial      ║ NO          ║ YES           ║
    ╚═══════════════════════╩══════════════╩══════════════╩═════════════╩═══════════════╝

    Key formulae:
      Vanilla:     S(x) = |∂f_c(x) / ∂x|
      SmoothGrad:  SG(x) = (1/N) Σᵢ ∂f_c(x+εᵢ)/∂(x+εᵢ),  εᵢ ~ N(0,σ²I)
      IG:          IG_i(x) = (xᵢ−xᵢ') × ∫₀¹ ∂f_c(x'+α(x−x'))/∂xᵢ dα
                   Completeness:  Σᵢ IG_i(x) = f_c(x) − f_c(x')
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Vanilla Gradient, SmoothGrad, and Integrated Gradients — All from Scratch": {
        "description": (
            "Implements a small multi-layer neural network with ReLU activations "
            "from scratch, then computes vanilla gradient, SmoothGrad, and Integrated "
            "Gradients on both a tabular instance and a 1D 'image' (a signal vector). "
            "Shows the saturation problem live, demonstrates the completeness gap "
            "as a quality check for IG, and compares all three attribution vectors "
            "side by side. Zero external dependencies — pure Python."
        ),
        "runnable": True,
        "pipeline_cmd": "saliency",
        "code": '''
"""
================================================================================
VANILLA GRADIENT, SMOOTHGRAD, AND INTEGRATED GRADIENTS FROM SCRATCH
================================================================================

We build a small neural network from scratch using only Python and math,
then implement all three attribution methods and compare them.

Network architecture (tabular classifier, 4 inputs → 2 classes):
  Layer 1: Linear(4 → 8) + ReLU
  Layer 2: Linear(8 → 4) + ReLU
  Layer 3: Linear(4 → 2)  → logits for {DENY=0, APPROVE=1}

The network is initialised with fixed weights (for reproducibility).
We explain the prediction for a single loan applicant.
================================================================================
"""

import math
import random


random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK — pure Python, no libraries
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return max(0.0, x)

def relu_grad(x):
    return 1.0 if x > 0 else 0.0

def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def matvec(W, x):
    """W is (out×in), x is (in,) → returns (out,)"""
    return [sum(W[i][j] * x[j] for j in range(len(x)))
            for i in range(len(W))]

def addvec(a, b):
    return [a[i] + b[i] for i in range(len(a))]

# Fixed weights (pre-set for reproducibility — simulating a trained model)
# Layer 1: 4 → 8
W1 = [
    [ 0.52, -0.31,  0.71,  0.18],
    [-0.24,  0.88, -0.45,  0.61],
    [ 0.39,  0.17,  0.55, -0.72],
    [-0.61,  0.43,  0.28,  0.93],
    [ 0.75, -0.52,  0.11,  0.37],
    [-0.18,  0.64, -0.83,  0.22],
    [ 0.44,  0.29,  0.67, -0.55],
    [-0.33,  0.71,  0.14,  0.88],
]
b1 = [-0.20,  0.10, -0.15,  0.25, -0.10,  0.05, -0.30,  0.20]

# Layer 2: 8 → 4
W2 = [
    [ 0.61, -0.45,  0.33,  0.78, -0.22,  0.55, -0.41,  0.19],
    [-0.34,  0.72,  0.15, -0.63,  0.48, -0.29,  0.83, -0.57],
    [ 0.22, -0.18,  0.91,  0.44, -0.67,  0.31, -0.15,  0.76],
    [-0.55,  0.38, -0.24,  0.81,  0.12, -0.72,  0.46, -0.33],
]
b2 = [ 0.15, -0.10,  0.20, -0.05]

# Layer 3: 4 → 2 (logits for DENY=0, APPROVE=1)
W3 = [
    [-0.48,  0.62, -0.35,  0.71],
    [ 0.53, -0.44,  0.67, -0.28],
]
b3 = [0.0, 0.0]

WEIGHTS = [(W1, b1), (W2, b2), (W3, b3)]
ACTIVATIONS = [True, True, False]  # ReLU after layers 1 and 2, not 3


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS — returns output and all pre-activations for backward pass
# ─────────────────────────────────────────────────────────────────────────────

def forward(x, weights=WEIGHTS, activations=ACTIVATIONS):
    """
    Returns: (final output, list of pre-activation vectors per layer)
    Pre-activations are needed for the backward pass (gradient of ReLU).
    """
    z_list = []   # pre-activations
    a = list(x)   # current activations
    for (W, b), use_relu in zip(weights, activations):
        z = addvec(matvec(W, a), b)
        z_list.append(z)
        if use_relu:
            a = [relu(zi) for zi in z]
        else:
            a = z
    return a, z_list


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD PASS — gradient of output[target_class] w.r.t. input x
# ─────────────────────────────────────────────────────────────────────────────

def backward(x, target_class=1, weights=WEIGHTS, activations=ACTIVATIONS):
    """
    Returns the gradient ∂f_{target_class} / ∂x via backpropagation.
    """
    output, z_list = forward(x, weights, activations)
    n_layers = len(weights)

    # Gradient of the target logit w.r.t. the final layer's pre-activation
    # (d_output/d_logit = 1 for the target class, 0 for others)
    delta = [1.0 if i == target_class else 0.0
             for i in range(len(output))]

    # Backpropagate through layers (reverse order)
    for l in range(n_layers - 1, -1, -1):
        W, b = weights[l]
        use_relu = activations[l]

        # Gradient through activation function
        if use_relu:
            delta = [delta[i] * relu_grad(z_list[l][i])
                     for i in range(len(delta))]

        # Gradient w.r.t. previous layer's activations (W^T @ delta)
        n_in = len(W[0])
        delta_prev = [sum(W[i][j] * delta[i] for i in range(len(W)))
                      for j in range(n_in)]
        delta = delta_prev

    return delta  # gradient w.r.t. input x


# ─────────────────────────────────────────────────────────────────────────────
# DATASET AND INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

# Input features (standardised): [credit_std, income_std, debt_std, employment_std]
# credit_std = (credit_score - 640) / 80
# income_std = (income_k - 58) / 25
# debt_std   = (debt_ratio - 0.38) / 0.12
# employment_std = (years - 5) / 4

x_instance = [
    (590 - 640) / 80,     # credit = 590  → -0.625 (below mean)
    (52  -  58) / 25,     # income = 52k  → -0.24  (slightly below)
    (0.51 - 0.38) / 0.12, # debt   = 0.51 → +1.083 (above mean = risky)
    (3   -   5) / 4,      # employment = 3yr → -0.50 (below mean)
]
feat_names = ["credit_score", "income", "debt_ratio", "employment_yrs"]

x_baseline = [0.0, 0.0, 0.0, 0.0]  # all-zero (mean of each standardised feature)

output, _ = forward(x_instance)
probs = softmax(output)

print("=" * 68)
print("  NETWORK AND INSTANCE")
print("=" * 68)
print(f"  Architecture: 4 → 8 → 4 → 2  (ReLU after each hidden layer)")
print(f"  Features (standardised): {[round(v,3) for v in x_instance]}")
print(f"  Raw logits:  DENY={output[0]:.4f},  APPROVE={output[1]:.4f}")
print(f"  Probabilities: DENY={probs[0]:.1%},  APPROVE={probs[1]:.1%}")
print(f"  Prediction: {'DENY' if probs[0]>probs[1] else 'APPROVE'}")
print()
print(f"  Baseline: {x_baseline}  (mean of all standardised features)")
output_base, _ = forward(x_baseline)
probs_base = softmax(output_base)
print(f"  Baseline logits: DENY={output_base[0]:.4f}, APPROVE={output_base[1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — VANILLA GRADIENT
# ─────────────────────────────────────────────────────────────────────────────

# Explain the DENY class (class 0), which is the predicted class
TARGET = 0   # DENY

grad_vanilla = backward(x_instance, target_class=TARGET)

print()
print("=" * 68)
print("  METHOD 1: VANILLA GRADIENT   (∂f_DENY/∂x at instance x)")
print("=" * 68)
print()
print(f"  {'Feature':<20}  {'Gradient':>12}  {'|Gradient|':>12}  {'Rank'}")
print(f"  {'-'*55}")
ranked = sorted(range(4), key=lambda i: -abs(grad_vanilla[i]))
for rank, i in enumerate(ranked, 1):
    g = grad_vanilla[i]
    print(f"  {feat_names[i]:<20}  {g:>+12.6f}  {abs(g):>12.6f}  {rank}")
print()
print(f"  Sum of gradients: {sum(grad_vanilla):+.6f}  (NOT required to equal anything)")
print(f"  Sum |gradients|:  {sum(abs(g) for g in grad_vanilla):.6f}")
print()
print(f"  Interpretation: the gradient at x tells us which features the")
print(f"  model's DENY score is MOST SENSITIVE to, locally at this point.")
print(f"  It does NOT tell us how much each feature CONTRIBUTED to the")
print(f"  score — only how the score would change with a tiny perturbation.")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — SMOOTHGRAD
# ─────────────────────────────────────────────────────────────────────────────

def smoothgrad(x, target_class, n_samples=100, sigma=0.15):
    """
    Average gradient over N noisy perturbations of x.
    sigma is fraction of each feature's range (here, scaled by 1 since std features).
    """
    d = len(x)
    grad_sum = [0.0] * d
    for _ in range(n_samples):
        noise = [random.gauss(0, sigma) for _ in range(d)]
        x_noisy = [x[i] + noise[i] for i in range(d)]
        g = backward(x_noisy, target_class=target_class)
        for i in range(d):
            grad_sum[i] += g[i]
    return [g / n_samples for g in grad_sum]

N_SMOOTH = 100
SIGMA    = 0.15

grad_smooth = smoothgrad(x_instance, target_class=TARGET,
                          n_samples=N_SMOOTH, sigma=SIGMA)

print()
print("=" * 68)
print(f"  METHOD 2: SMOOTHGRAD  (N={N_SMOOTH} samples, σ={SIGMA})")
print("=" * 68)
print()
print(f"  {'Feature':<20}  {'SmoothGrad':>12}  {'Vanilla Grad':>13}  {'Diff':>8}")
print(f"  {'-'*58}")
for i in range(4):
    diff = grad_smooth[i] - grad_vanilla[i]
    print(f"  {feat_names[i]:<20}  {grad_smooth[i]:>+12.6f}  "
          f"{grad_vanilla[i]:>+13.6f}  {diff:>+8.4f}")
print()
print(f"  SmoothGrad averages gradients over a Gaussian ball of radius σ={SIGMA}")
print(f"  around the instance. Values differ from vanilla gradient because:")
print(f"  • Some nearby points activate different ReLU units.")
print(f"  • SmoothGrad averages over these, smoothing the discontinuities.")
print(f"  • The ranking of features may change after smoothing.")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — INTEGRATED GRADIENTS
# ─────────────────────────────────────────────────────────────────────────────

def integrated_gradients(x, x_prime, target_class, m_steps=50):
    """
    Compute Integrated Gradients with M Riemann steps.

    IG_i = (x_i - x'_i) × (1/M) Σ_{k=1}^{M} ∂f/∂x_i at x(k/M)

    Returns (ig_values, completeness_gap_fraction).
    """
    d = len(x)
    grad_sum = [0.0] * d

    # Sum gradients at M equally-spaced interpolation steps
    for k in range(1, m_steps + 1):
        alpha = k / m_steps
        x_interp = [x_prime[i] + alpha * (x[i] - x_prime[i])
                    for i in range(d)]
        g = backward(x_interp, target_class=target_class)
        for i in range(d):
            grad_sum[i] += g[i]

    avg_grads = [g / m_steps for g in grad_sum]

    # IG_i = (x_i - x'_i) × average gradient
    ig = [(x[i] - x_prime[i]) * avg_grads[i] for i in range(d)]

    # Completeness check
    output_x,  _ = forward(x)
    output_xp, _ = forward(x_prime)
    delta_f   = output_x[target_class] - output_xp[target_class]
    sum_ig    = sum(ig)
    gap_abs   = abs(sum_ig - delta_f)
    gap_frac  = gap_abs / (abs(delta_f) + 1e-12)

    return ig, delta_f, sum_ig, gap_frac

M_STEPS = 100

ig, delta_f, sum_ig, gap = integrated_gradients(
    x_instance, x_baseline, target_class=TARGET, m_steps=M_STEPS)

print()
print("=" * 68)
print(f"  METHOD 3: INTEGRATED GRADIENTS  (M={M_STEPS} steps, zero baseline)")
print("=" * 68)
print()
print(f"  Target class: DENY (class 0)")
print(f"  f_DENY(x)     = {forward(x_instance)[0][TARGET]:.6f}")
print(f"  f_DENY(x')    = {forward(x_baseline)[0][TARGET]:.6f}")
print(f"  Δf = f(x) - f(x') = {delta_f:+.6f}   ← amount to explain")
print()
print(f"  {'Feature':<20}  {'IG value':>10}  {'% of Δf':>9}  {'Rank'}")
print(f"  {'-'*52}")
ranked_ig = sorted(range(4), key=lambda i: -abs(ig[i]))
for rank, i in enumerate(ranked_ig, 1):
    pct = ig[i] / (abs(delta_f) + 1e-12) * 100
    print(f"  {feat_names[i]:<20}  {ig[i]:>+10.6f}  {pct:>+8.1f}%  {rank}")
print(f"  {'-'*52}")
print(f"  {'Sum of IG':>20}  {sum_ig:>+10.6f}  {sum_ig/delta_f*100:>+8.1f}%")
print(f"  {'Actual Δf':>20}  {delta_f:>+10.6f}  {'100.0%':>9}")
print(f"  {'Completeness gap':>20}  {abs(sum_ig-delta_f):>10.6f}  "
      f"({gap:.2%} of Δf)")
print()
if gap < 0.02:
    print(f"  ✓ Completeness gap is {gap:.2%} < 2% — M={M_STEPS} is sufficient.")
else:
    print(f"  ⚠️  Completeness gap is {gap:.2%} > 2% — increase M.")
print()
print(f"  INTERPRETATION (faithfully accounting for {delta_f:+.4f} logit change):")
for i in ranked_ig:
    direction = "↑ INCREASES DENY" if ig[i] > 0 else "↓ decreases DENY"
    print(f"  • {feat_names[i]:<18}: {ig[i]:>+.4f}  {direction}")


# ─────────────────────────────────────────────────────────────────────────────
# THREE-WAY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  THREE-WAY COMPARISON — SAME INSTANCE, THREE METHODS")
print("=" * 68)
print()
print(f"  {'Feature':<20}  {'Vanilla Grad':>13}  {'SmoothGrad':>11}  "
      f"{'IG (norm)':>10}  {'Agreement?'}")
print(f"  {'-'*72}")

# Normalise each method to sum-of-absolute-values = 1 for comparison
def normalise(v):
    s = sum(abs(x) for x in v) + 1e-12
    return [x / s for x in v]

n_vanilla = normalise(grad_vanilla)
n_smooth  = normalise(grad_smooth)
n_ig      = normalise(ig)

for i in range(4):
    # Check if all three methods agree on the SIGN
    signs = [1 if v >= 0 else -1 for v in [n_vanilla[i], n_smooth[i], n_ig[i]]]
    agree = "✓ agree" if len(set(signs)) == 1 else "✗ disagree"
    print(f"  {feat_names[i]:<20}  {n_vanilla[i]:>+13.4f}  "
          f"{n_smooth[i]:>+11.4f}  {n_ig[i]:>+10.4f}  {agree}")

print()
print(f"  RANKING COMPARISON:")
rank_v  = sorted(range(4), key=lambda i: -abs(n_vanilla[i]))
rank_s  = sorted(range(4), key=lambda i: -abs(n_smooth[i]))
rank_ig = sorted(range(4), key=lambda i: -abs(n_ig[i]))
print(f"  Vanilla:    {' > '.join(feat_names[i][:8] for i in rank_v)}")
print(f"  SmoothGrad: {' > '.join(feat_names[i][:8] for i in rank_s)}")
print(f"  IG:         {' > '.join(feat_names[i][:8] for i in rank_ig)}")


# ─────────────────────────────────────────────────────────────────────────────
# SATURATION DEMO — where vanilla gradient fails
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SATURATION DEMO — where vanilla gradient misleads")
print("=" * 68)
print()
print("  We create a 'very confident' version of our instance by scaling")
print("  x by a factor of 3 (extreme feature values → model very certain).")
print()

x_extreme = [v * 3.0 for v in x_instance]
out_extreme, _ = forward(x_extreme)
prob_extreme = softmax(out_extreme)
grad_extreme_vanilla = backward(x_extreme, target_class=TARGET)
ig_extreme, delta_f_ex, sum_ig_ex, gap_ex = integrated_gradients(
    x_extreme, x_baseline, target_class=TARGET, m_steps=M_STEPS)

print(f"  Original instance (x):  DENY prob = {probs[0]:.1%}")
print(f"  Extreme instance (3×x): DENY prob = {prob_extreme[0]:.1%}")
print()
print(f"  Feature          Vanilla grad (x)  Vanilla grad (3x)  IG (x)  IG (3x)")
print(f"  {'-'*72}")
for i in range(4):
    g_orig   = grad_vanilla[i]
    g_extrem = grad_extreme_vanilla[i]
    ig_orig  = ig[i]
    ig_extrem = ig_extreme[i]
    print(f"  {feat_names[i]:<16} {g_orig:>+16.4f}  {g_extrem:>+17.4f}  "
          f"{ig_orig:>+7.4f}  {ig_extrem:>+7.4f}")

print()
print(f"  SATURATION EFFECT:")
print(f"  At 3×x the model is very confident (DENY={prob_extreme[0]:.0%}).")

grad_change = [abs(grad_extreme_vanilla[i]) / (abs(grad_vanilla[i])+1e-9)
               for i in range(4)]
print(f"  Average gradient magnitude RATIO (3x vs x): "
      f"{sum(grad_change)/4:.3f}×")
ig_change = [abs(ig_extreme[i]) / (abs(ig[i])+1e-9) for i in range(4)]
print(f"  Average IG magnitude RATIO (3x vs x):       "
      f"{sum(ig_change)/4:.3f}×")
print()
print(f"  Vanilla gradients SHRINK near saturation (model is 'flat' there).")
print(f"  IG values GROW proportionally because the PATH is longer (3x vs 1x).")
print(f"  IG correctly attributes more to features when their contribution")
print(f"  to the output is larger — vanilla gradient does the opposite.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETENESS GAP AS A QUALITY CHECK ACROSS M VALUES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  COMPLETENESS GAP — quality check across M values")
print("=" * 68)
print()
print(f"  The completeness gap |ΣIG_i − Δf| / |Δf| measures")
print(f"  approximation error. It should be < 1–2% for reliable attributions.")
print()
print(f"  {'M steps':>8}  {'ΣIG':>10}  {'Δf (true)':>10}  {'Gap':>8}  {'Reliable?'}")
print(f"  {'-'*52}")

for m in [5, 10, 20, 50, 100, 200]:
    ig_m, df_m, sig_m, gap_m = integrated_gradients(
        x_instance, x_baseline, target_class=TARGET, m_steps=m)
    reliable = "✓ yes" if gap_m < 0.02 else ("⚠️  marginal" if gap_m < 0.05 else "✗ no")
    print(f"  {m:>8}  {sig_m:>+10.5f}  {df_m:>+10.5f}  {gap_m:>7.2%}  {reliable}")

print()
print(f"  This gap converges to 0 as M → ∞. It is the ONLY built-in quality")
print(f"  check available in gradient-based attribution methods.")
print(f"  Always verify the gap is small before reporting IG attributions.")
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
    #     from interpretability.visuals.saliency_maps import (
    #         SALIENCY_VISUAL_HTML,
    #         SALIENCY_VISUAL_HEIGHT,
    #     )
    #     visual_html   = SALIENCY_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SALIENCY_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[saliency_maps.py] Could not load visual: {e}", stacklevel=2)

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