"""
GradCAM — Gradient-weighted Class Activation Mapping
======================================================

GradCAM (Selvaraju, Cogswell, Das, Vedantam, Parikh & Batra, 2017) produces
a coarse spatial heatmap showing WHICH REGIONS of an image drove a convolutional
neural network's classification decision. Unlike pixel-level saliency maps, GradCAM
operates at the semantic level of the last convolutional feature maps — the layer
where the network has already encoded high-level concepts like "ears," "wheels,"
or "tumour margins."

This module derives GradCAM from first principles, proves why the last conv layer
is the right layer to interrogate, traces through the algebra of the gradient
weighting, covers the full family of CAM variants, and establishes what GradCAM
can and cannot tell you about a CNN's reasoning.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "GradCAM — Gradient-weighted Class Activation Mapping"
DISPLAY_NAME = "04b · GradCAM"
ICON = "🗺️"
SUBTITLE = "Spatial heatmaps from the last convolutional layer"


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

### The Problem GradCAM Solves

A trained image classifier predicts "golden retriever" for an input photograph.
You want to know: where in the image did the network look to reach that conclusion?
Was it the dog's face? The fur texture? The background grass? Or a dataset artefact
like a human hand holding a leash?

Pixel-level saliency maps (vanilla gradient, Integrated Gradients) give you one
number per pixel — 224 × 224 × 3 ≈ 150,000 numbers for a standard input. The
signal is there, but it is distributed across every pixel and hard to read as
a spatial "look here" statement.

GradCAM makes a different bet: skip the pixels. Go to the LAST CONVOLUTIONAL
LAYER. By that point in the network, the spatial structure of the image is still
preserved (the feature maps are a compressed, lower-resolution version of the
image's spatial layout), but each channel encodes a learned high-level concept
rather than raw colour values. Interrogating this layer gives you a map that is
spatially meaningful and semantically interpretable — at the cost of resolution.

The result is a low-resolution heatmap (e.g., 7 × 7 for VGG, 14 × 14 for
ResNet-50) that is bilinearly upsampled to the original image size and overlaid
as a colour gradient — cool blue for unimportant regions, warm red for regions
that strongly drove the prediction.


##### PART I: THE ANATOMY OF A CONVOLUTIONAL NETWORK — WHAT EACH LAYER HOLDS

### Feature Maps and the Spatial Hierarchy

A standard image classification CNN (VGG, ResNet, EfficientNet) is a stack
of convolutional blocks followed by a global pooling step and a fully connected
classifier. The convolutional portion produces a sequence of FEATURE MAPS —
3D tensors of shape (C, H, W) where C is the number of channels (filters),
H and W are the spatial height and width.

As you go deeper in the network, two things happen simultaneously:
  • Spatial resolution DECREASES: H and W shrink via pooling or strided conv.
  • Channel count INCREASES: C grows as the network learns more concepts.

    # ================================================================== #
    **Spatial hierarchy in a typical CNN (224×224 input):**

    Layer group     Feature map shape   What channels encode
    ──────────────────────────────────────────────────────────────────
    Input           (3, 224, 224)       Raw RGB pixels
    Conv block 1    (64, 112, 112)      Edges, colour gradients
    Conv block 2    (128, 56, 56)       Textures, corners, junctions
    Conv block 3    (256, 28, 28)       Parts, local patterns
    Conv block 4    (512, 14, 14)       Object parts, semantic regions
    Conv block 5    (512, 7, 7)         High-level concepts, whole objects
    Global avg pool (512, 1, 1)         Per-channel summary (spatial info lost)
    FC + softmax    (num_classes,)      Class scores

    The last conv block's feature maps (e.g., 512 × 7 × 7) are the last
    place in the network where:
      1. Spatial information is still present (7 × 7 positions).
      2. The encoding is class-discriminative (trained by gradients from labels).
    # ================================================================== #

This dual property — spatial + semantic — is why GradCAM targets the last
convolutional layer. Earlier layers have better spatial resolution but poorer
semantics. Later layers (FC) have perfect semantics but zero spatial structure.
The last conv layer is the sweet spot.


### What Each Feature Map Channel Represents

Each of the C channels in the last convolutional layer's output is a filter
that has been learned to respond to a specific high-level pattern. After training
on a large dataset:

    Channel k activates strongly ↔ the network has learned that the receptive
    field of that channel contains a concept relevant to its learned purpose.

    For an ImageNet-trained VGG16:
    Channel 42 might fire for "round, dog-like heads"
    Channel 107 might fire for "golden-brown fur textures"
    Channel 315 might fire for "open-mouth, teeth visible"

These "concepts" are NOT hand-labelled. They emerge from gradient descent.
Some channels encode clean, interpretable concepts (verified by Network
Dissection, Bau et al. 2017). Others encode messy, polysemantic patterns.

GradCAM's key insight: we do not need to know WHAT each channel encodes.
We only need to know HOW MUCH each channel contributed to the final
classification score. The gradient tells us exactly that.


##### PART II: THE ORIGINAL CAM — CLASS ACTIVATION MAPPING

### CAM — The Predecessor to GradCAM

To understand GradCAM, you must first understand the original CAM
(Class Activation Mapping, Zhou et al., 2016), which is the special case
that GradCAM generalises.

CAM requires a specific architecture: a CNN that uses GLOBAL AVERAGE POOLING
(GAP) before the final classification layer, with NO intermediate fully
connected layers. The architecture must be:

    Conv layers → Global Average Pool → Linear(C → num_classes)

Under this architecture, the output score for class c is:

    S_c = Σₖ w_c^k × (1/HW) Σᵢ Σⱼ Aᵏᵢⱼ
        = (1/HW) Σᵢ Σⱼ Σₖ w_c^k × Aᵏᵢⱼ

where:
  • Aᵏᵢⱼ is the activation at spatial position (i,j) of feature map channel k
  • w_c^k is the weight connecting channel k's global-average-pooled value to class c
  • H, W are the spatial dimensions of the last conv feature maps

The CAM for class c is:

    CAM_c(i, j) = Σₖ w_c^k × Aᵏᵢⱼ

This is a linear combination of all feature maps, weighted by their importance
to class c (as given directly by the final FC layer's weights).

    # ================================================================== #
    **Why CAM works under the GAP architecture — the algebra:**

    Score for class c:
    S_c = Σₖ w_c^k × (gap of channel k)
        = Σₖ w_c^k × (1/HW) Σᵢⱼ Aᵏᵢⱼ
        = (1/HW) Σᵢⱼ [Σₖ w_c^k × Aᵏᵢⱼ]
        = (1/HW) Σᵢⱼ CAM_c(i,j)

    The score S_c is literally the spatial average of the CAM heatmap!
    Every spatial position (i,j) contributes to the final score exactly
    in proportion to its CAM value. High CAM value → high contribution.
    The heatmap and the classification score are the same computation,
    just viewed from different angles.
    # ================================================================== #


### The Critical Limitation of CAM

CAM requires the GAP-before-classifier architecture. Standard CNNs (VGG,
AlexNet) use one or more fully connected layers after the last conv block,
not global average pooling. To apply CAM to these networks, you would need
to RETRAIN them with a GAP architecture — which defeats the purpose of
explaining an already-trained model.

GradCAM was created precisely to generalise CAM to ANY architecture,
without retraining.


##### PART III: GRADCAM — THE DERIVATION

### The Core Question

In the CAM formula, the weights w_c^k have an explicit meaning: they are
the FC layer's learned parameters, directly connecting channel k to class c.
GradCAM asks: can we compute EQUIVALENT weights for any network, without
requiring the specific GAP architecture?

The answer is yes — by using GRADIENTS to infer how important each channel
is for each class.

### Step 1 — The Gradient as a Proxy for Importance

For a network with ANY architecture, the gradient of the class score S_c
with respect to the feature maps Aᵏᵢⱼ of the last conv layer tells us:
"how much does a small change in position (i,j) of channel k's feature
map change the score for class c?"

Specifically, the gradient ∂S_c / ∂Aᵏᵢⱼ is large and positive when:
  • Increasing activation at (i,j) in channel k increases the class-c score.
  • Meaning: the network has learned to use that spatial position of that
    channel as evidence for class c.

Conversely, large negative gradients mean "more activation here HURTS class c."

### Step 2 — Summarising the Gradient per Channel

We have a gradient map ∂S_c / ∂Aᵏ of shape (H, W) for each channel k.
This is still an H × W grid of numbers per channel — too granular for a
clean summary of "how important is channel k?"

GradCAM summarises by GLOBAL AVERAGE POOLING the gradient:

    αᵏ_c = (1 / H×W) Σᵢ Σⱼ  ∂S_c / ∂Aᵏᵢⱼ

This scalar αᵏ_c is the IMPORTANCE WEIGHT for channel k with respect to
class c. It is the average sensitivity of the class score to changes in
feature map k across all spatial positions.

### Step 3 — Weighted Combination of Feature Maps

Using the importance weights αᵏ_c, compute the weighted sum of feature maps:

    L^GradCAM_c = ReLU( Σₖ αᵏ_c × Aᵏ )

Two components here:

    THE WEIGHTED SUM Σₖ αᵏ_c × Aᵏ:
    Each feature map Aᵏ (shape H×W) is scaled by its importance αᵏ_c
    and summed across all channels. Positive αᵏ_c amplifies that channel's
    spatial pattern; negative αᵏ_c suppresses it.

    THE ReLU:
    Takes max(0, value) element-wise. This zeroes out locations in the
    combined map that are NEGATIVELY associated with class c.
    Without the ReLU, the heatmap would show BOTH "where class c is
    supported" (positive) AND "where class c is suppressed" (negative),
    producing a confusing mix. The ReLU keeps only the positive class-c
    evidence, giving a pure "look here for class c" heatmap.

### Step 4 — Upsample to Input Resolution

The GradCAM heatmap L^GradCAM_c has shape (H, W) — the spatial dimensions
of the last conv layer (e.g., 7 × 7 for VGG16). This is bilinearly
interpolated to the original image resolution (224 × 224) and overlaid.

    # ================================================================== #
    **Full GradCAM algorithm summary:**

    INPUT: image x, class c, trained CNN f
    OUTPUT: heatmap L ∈ ℝ^{H×W}

    1. Forward pass: compute f(x), save last conv activations A ∈ ℝ^{C×H×W}
    2. Backward pass: compute ∂S_c / ∂Aᵏᵢⱼ for all k, i, j
    3. Global average pool the gradients:
         αᵏ_c = (1/HW) Σᵢ Σⱼ (∂S_c / ∂Aᵏᵢⱼ)   for each k
    4. Weighted sum + ReLU:
         L^GradCAM_c = ReLU(Σₖ αᵏ_c × Aᵏ)
    5. Normalise L to [0, 1]
    6. Upsample L from (H,W) to (input_H, input_W) via bilinear interpolation
    7. Overlay on original image as a heatmap

    Model calls: 1 forward pass + 1 backward pass (same cost as vanilla gradient)
    # ================================================================== #


### Why GradCAM Generalises CAM

Under the GAP architecture, the CAM weights w_c^k are the final FC weights.
GradCAM computes αᵏ_c via gradient averaging. How do these relate?

For the specific case of a GAP + linear classifier architecture:

    S_c = Σₖ w_c^k × (1/HW) Σᵢⱼ Aᵏᵢⱼ

    ∂S_c / ∂Aᵏᵢⱼ = w_c^k / (H×W)

    αᵏ_c = (1/HW) Σᵢⱼ (w_c^k / HW) = w_c^k / HW

    GradCAM heatmap = Σₖ αᵏ_c × Aᵏ = Σₖ (w_c^k / HW) × Aᵏ
                    = (1/HW) × CAM_c

GradCAM recovers the CAM heatmap exactly (up to a constant 1/HW scale factor,
which is normalised away). CAM is the special case of GradCAM for GAP networks.
For all other architectures, GradCAM is the only tractable option.

    # ================================================================== #
    **Generalisation: why the GAP of the gradient equals the weight:**

    ∂S_c / ∂Aᵏᵢⱼ = w_c^k / HW  (constant over all i,j for GAP networks)

    The gradient is spatially UNIFORM within each channel — every position
    in channel k contributes equally to the score (because GAP averages them).
    So the global average of the gradient is trivially equal to the gradient
    itself (times 1/(HW)).

    For non-GAP networks (e.g., VGG with FC layers), the gradient is NOT
    spatially uniform — different positions in the same channel contribute
    differently. GradCAM's averaging still gives a sensible channel-level
    importance, but it is an approximation rather than an exact derivation.
    # ================================================================== #


##### PART IV: THE ROLE OF RELU — THREE REGIMES OF GRADCAM

### Positive, Negative, and Both: Choosing What to Highlight

The final ReLU in GradCAM is a design choice, not a mathematical necessity.
Removing it reveals three distinct use cases:

    CASE 1 — POSITIVE EVIDENCE (standard GradCAM):
    L = ReLU(Σₖ αᵏ_c × Aᵏ)
    Shows spatial regions that positively support class c.
    Use when you want to know: "where does the network SEE class c?"
    Example: for "cat" classification, highlights the cat's face and body.

    CASE 2 — NEGATIVE EVIDENCE:
    L = ReLU(−Σₖ αᵏ_c × Aᵏ)  = ReLU(Σₖ (−αᵏ_c) × Aᵏ)
    Shows spatial regions that SUPPRESS class c — evidence for other classes.
    Use when you want to know: "where does the network see evidence AGAINST class c?"
    Example: for "cat" classification, highlights background objects that
    would push toward "dog" or "chair."

    CASE 3 — ALL EVIDENCE:
    L = |Σₖ αᵏ_c × Aᵏ|  (absolute value, no ReLU)
    Shows all regions that influence the classification, positive or negative.
    Use when debugging: reveals the full set of image regions the network uses.

    # ================================================================== #
    **Why the standard ReLU choice makes interpretive sense:**

    Positive αᵏ_c: channel k's activation HELPS class c score.
    Negative αᵏ_c: channel k's activation HURTS class c score.

    For a clean "where did the model look for THIS class?" explanation,
    you want ONLY the positive support. The ReLU zeroes out channels
    that were actively suppressing the predicted class — keeping only
    the channels that were driving it.

    Without ReLU:
    Suppose a cat image has a prominent background (grass).
    Channels encoding "grass texture" might have negative αᵏ_c for "cat"
    (grass is anti-evidence for cat). Without ReLU, the heatmap shows
    the grass region as IMPORTANT (just in the negative direction).
    This confuses the explanation: the network is not "looking at the
    grass to classify cat" — it is using the lack of cat-features
    there as negative evidence. The ReLU suppresses this.
    # ================================================================== #


### The Importance Weight αᵏ_c — What It Captures and Misses

The global average pooling of the gradient collapses a full H×W gradient
map per channel into a single scalar. This averaging loses spatial information
within a channel.

    Consider a channel k whose feature map Aᵏ activates strongly in
    the TOP-LEFT quadrant (where the dog's face is) and weakly elsewhere.
    If the gradient ∂S_c/∂Aᵏᵢⱼ is also large in the top-left, then
    αᵏ_c = average over ALL positions, including the weak-gradient positions
    in the bottom-right. The averaging DILUTES the importance of the
    top-left region.

    The channel is still marked as "important" (αᵏ_c > 0), but the spatial
    concentration of importance within the channel is lost. This is the
    core resolution limitation of GradCAM.

    Grad-CAM++ addresses this by weighting individual pixel-level gradients:
    αᵏ_c^{++} = Σᵢ Σⱼ  (∂S_c/∂Aᵏᵢⱼ)² / (2(∂S_c/∂Aᵏᵢⱼ)² + Σᵢ'Σⱼ' Aᵏᵢ'ⱼ'(∂³S_c/∂Aᵏᵢⱼ³))

    This more complex weighting gives higher weight to positions where the
    gradient is large, producing sharper, less diluted heatmaps. The cost:
    it requires computing higher-order derivatives, which is more expensive
    and complex to implement.


##### PART V: THE CAM FAMILY — VARIANTS AND THEIR DESIGN CHOICES

### The Design Space of Class Activation Mapping

Every CAM-family method produces a heatmap L_c(i, j) = Σₖ αᵏ_c × Aᵏᵢⱼ.
They differ only in HOW they compute the channel importance weights αᵏ_c.

    METHOD             αᵏ_c COMPUTATION           KEY PROPERTY
    ──────────────────────────────────────────────────────────────────
    CAM (2016)         w_c^k from FC weights      Exact, needs GAP arch.
    GradCAM (2017)     GAP of gradients ∂S_c/∂Aᵏ  Model-agnostic, fast
    GradCAM++ (2018)   Weighted gradient avg      Sharper, handles multi-instance
    ScoreCAM (2020)    Perturbation-based         Gradient-free, slower
    AblationCAM (2020) Gradient-free ablation     Robust, very slow
    EigenCAM (2021)    PCA of feature maps        No class labels needed
    FullGrad (2019)    Gradient of all layers     More faithful to network
    ──────────────────────────────────────────────────────────────────


### Grad-CAM++ — Sharper Maps for Multi-Instance Images

Grad-CAM++ (Chattopadhay et al., 2018) was motivated by two observed
failures of standard GradCAM:

    FAILURE 1 — MULTI-INSTANCE IMAGES:
    If multiple instances of the same class appear in an image (e.g.,
    three cats), GradCAM typically highlights only ONE of them — usually
    the largest or the most discriminative. The other instances are missed
    because the global-average-pooled gradient αᵏ_c is dominated by the
    channels that fire for the dominant instance.

    FAILURE 2 — PARTIAL OBJECT LOCALISATION:
    GradCAM often highlights only part of the object (e.g., just the dog's
    head) rather than the full extent. This is because the gradient is
    large near distinctive features and small near generic/background parts
    of the object.

Grad-CAM++ addresses both by computing a spatially weighted importance:

    αᵏ_c = Σᵢ Σⱼ  wᵏᵢⱼ_c × ReLU(∂S_c / ∂Aᵏᵢⱼ)

where the pixel-level weights wᵏᵢⱼ_c are:

    wᵏᵢⱼ_c = (∂²S_c / ∂(Aᵏᵢⱼ)²) / (2·(∂²S_c/∂(Aᵏᵢⱼ)²) + (Aᵏ·∂³S_c/∂(Aᵏᵢⱼ)³))

These second-order derivative weights give higher influence to positions
where the gradient is CHANGING RAPIDLY (indicating a local maximum of
sensitivity), which tends to correspond to object boundaries and multiple
instance peaks.


### ScoreCAM — Gradient-Free Class Activation Mapping

ScoreCAM (Wang et al., 2020) takes a completely different approach: instead
of using gradients to measure channel importance, it uses PERTURBATION.

For each channel k of the last conv layer:
  1. Upsample the feature map Aᵏ to input resolution and normalise to [0,1].
  2. Use this as a MASK: multiply the original image x by the upsampled Aᵏ.
     The masked image retains the parts of x where channel k activates.
  3. Run the masked image through the network and record the class score S_c.
  4. The improvement over baseline: αᵏ_c = S_c(masked by Aᵏ) − S_c(baseline).

Features:
  • Gradient-free: works even when gradients are unavailable or unreliable.
  • Avoids gradient saturation issues completely.
  • Produces sharper maps than GradCAM in many settings.
  • Cost: C forward passes (one per channel), where C ≈ 512 for VGG.
    This is 512× more expensive than GradCAM for VGG16.

ScoreCAM is the method of choice when:
  • Gradients are unavailable (e.g., discrete operations, non-differentiable layers).
  • The network uses activations prone to gradient saturation.
  • Computation budget allows 512+ forward passes per image.

    # ================================================================== #
    **ScoreCAM vs GradCAM — the speed-quality trade-off:**

    GradCAM:   1 forward + 1 backward pass = 2× inference cost.
               Approximate importance via gradient proxy.
               May miss some information due to gradient saturation.

    ScoreCAM:  C forward passes = 512× inference cost (VGG).
               Direct causal importance via masked-image evaluation.
               No gradient saturation. Sharper, more complete maps.
    # ================================================================== #


### EigenCAM — When Class Labels Are Not Available

EigenCAM (Muhammad & Yeasin, 2021) generates saliency maps WITHOUT class
labels or class scores. It instead uses the PRINCIPAL COMPONENTS of the
last conv layer's feature maps.

Algorithm:
  1. Forward pass to get last conv feature maps A ∈ ℝ^{C×H×W}.
  2. Reshape to a 2D matrix: flatten spatial dimensions → A ∈ ℝ^{C×HW}.
  3. Compute the first principal component (PC1) of A via SVD or power iteration.
  4. The EigenCAM heatmap is the spatial reconstruction of PC1:
       L_EigenCAM = reshape(A^T × v₁) ∈ ℝ^{H×W}
     where v₁ is the first right singular vector of A.

Use cases:
  • Unsupervised spatial attention — no labels, no training required.
  • Anomaly detection: PC1 captures the dominant activation pattern,
    which in anomaly settings corresponds to the anomalous region.
  • Faster than GradCAM (no backward pass, only forward + SVD).
  • The limitation: EigenCAM is not class-discriminative by construction.
    It shows "where the network is most active" rather than "where the
    network is most active FOR CLASS c."


##### PART VI: THE ADEBAYO SANITY CHECKS — DOES GRADCAM EXPLAIN THE MODEL?

### What the Sanity Checks Test

Adebayo et al. (2018) — "Sanity Checks for Saliency Maps" — devised a
systematic test for whether a saliency method explains the MODEL or
merely describes the INPUT. The test:

    CASCADING RANDOMISATION TEST:
    1. Train a CNN on ImageNet. Produce saliency maps. Save them.
    2. Randomise the TOP LAYER's weights (keeping all others trained).
       Produce new saliency maps. Compare to step 1.
    3. Randomise the TOP TWO LAYERS' weights. Compare.
    4. Continue cascading down until ALL layers are randomised.

    If the method explains the model:
      Maps should change progressively as more layers are randomised.
      Fully randomised model → maps should look like random noise,
      bearing no resemblance to the original trained-model maps.

    If the method explains the input:
      Maps remain similar regardless of weight randomisation.
      The method is essentially doing image processing, not model explanation.

### GradCAM's Performance on the Sanity Checks

GradCAM PASSES the cascading randomisation test. As layers are progressively
randomised, the GradCAM heatmaps change in a meaningful, progressive way —
becoming less coherent and eventually resembling noise. This confirms that
GradCAM's maps are genuinely dependent on the model's learned weights.

Contrast with:
  • Guided Backpropagation: FAILS. Maps barely change with randomisation.
  • GuidedGradCAM: FAILS. The guided backpropagation component dominates
    and inherits its failure.
  • Vanilla Gradient: PASSES (with caveats).
  • Integrated Gradients: PASSES.

This is a critical result: GuidedGradCAM, which was presented as an
improvement over GradCAM (combining GradCAM's spatial accuracy with
guided backpropagation's pixel-level sharpness), is actually LESS faithful
to the model than plain GradCAM. The visual improvement in GuidedGradCAM
comes at the cost of explanatory validity.


### The Model Parameter Randomisation Test

A complementary test (also Adebayo et al.): independently randomise a
single model's parameters and check if the saliency map changes. If the
model is unchanged but the saliency map changes, the method has high
variance relative to trivial perturbations — a sign of instability.

GradCAM shows relatively low variance under small parameter perturbations,
consistent with the global-average-pooling operation smoothing out noise
in the gradient signal.

    # ================================================================== #
    **The cascading randomisation test intuition:**

    Analogy: you are examining a forensic fingerprint matching system.
    Does the system actually use fingerprints, or does it just look at
    image quality and lighting?

    Test: replace the fingerprint database with random images (analogous
    to randomising weights). Does the "match" output change?

    If YES → the system uses the fingerprints (model-dependent).
    If NO  → the system ignores the fingerprints (input-dependent).

    Guided backpropagation: the "match" stays the same. It is not actually
    using the learned fingerprint database — it is responding to the
    query image's raw structure, independently of the database.
    GradCAM: the "match" changes correctly. It depends on the database.
    # ================================================================== #


##### PART VII: GRADCAM'S LIMITATIONS — WHAT IT CANNOT TELL YOU

### Limitation 1 — Coarse Resolution

GradCAM's heatmap has the spatial resolution of the last conv layer,
typically 7×7 (VGG), 7×7 (ResNet-50), or 14×14 (ResNet-34). This is
bilinearly upsampled to 224×224 — a 32× or 16× spatial upsampling.

The upsampling introduces spatial imprecision. A small 7×7 activation
blob covers ≈ 32×32 pixels when upsampled. This means GradCAM cannot
tell you WHICH SPECIFIC PIXELS in a 32×32 region are important — only
that the region as a whole matters.

For medical imaging applications where pixel-level localisation matters
(e.g., tumour margin delineation), this resolution may be clinically
unacceptable. Integrated Gradients or Grad-CAM++ provide better resolution.

    # ================================================================== #
    **Resolution comparison on a 224×224 image:**

    Method                  Effective resolution    Upsampling factor
    ─────────────────────────────────────────────────────────────────
    Vanilla Gradient        224×224 (pixel-level)   1× (no upsampling)
    Integrated Gradients    224×224 (pixel-level)   1× (no upsampling)
    GradCAM (VGG)           7×7 → 224×224           32× upsampling
    GradCAM (ResNet-50)     7×7 → 224×224           32× upsampling
    GradCAM (ResNet-34)     14×14 → 224×224         16× upsampling
    Grad-CAM++              7×7 → 224×224           32× (but sharper weights)
    ScoreCAM                7×7 → 224×224           32× (but more precise)
    # ================================================================== #


### Limitation 2 — Class Mixing in the Heatmap

GradCAM highlights regions important for class c, but the feature maps
Aᵏ are shared across all classes. A feature map that fires for "dog fur
texture" might have:
  • Positive αᵏ_c for class "golden retriever"
  • Positive αᵏ_c for class "Labrador"
  • Negative αᵏ_c for class "cat"

When generating a GradCAM heatmap for class "golden retriever," this channel
is weighted positively. When generating for "Labrador," also positive. The
two heatmaps will be similar, because they both share the same underlying
feature maps — only the weighting vector αᵏ_c differs.

For fine-grained classification (distinguishing between dog breeds), the
GradCAM maps for different fine-grained classes may look nearly identical,
failing to reveal what distinguishes breed A from breed B.


### Limitation 3 — No Attribution to Pixel Values

GradCAM says "the SPATIAL REGION around position (i,j) is important."
It does NOT say WHICH ASPECT of that region is important:
  • Is it the colour?
  • The texture?
  • The shape/edge?
  • The relationship between this region and another?

A pixel-level method like Integrated Gradients gives one attribution per
pixel per channel — implicitly encoding which colour channel (R, G, or B)
contributes, not just which spatial position.

GradCAM's upsampled heatmap cannot distinguish between:
  • "The dog's brown fur colour is important at this location"
  • "The dog's ear shape is important at this location"
  Both produce the same hot region in the heatmap.


### Limitation 4 — Sensitivity to the Target Class Definition

GradCAM requires you to specify a class c before computing the heatmap.
The class determines the gradient direction and therefore the αᵏ_c weights.
Different class choices produce different heatmaps FROM THE SAME IMAGE:

    Input: an image containing both a cat and a dog.
    GradCAM for "cat": highlights the cat region.
    GradCAM for "dog": highlights the dog region.
    GradCAM for "sofa" (background object): highlights the sofa.

This is a feature, not a bug — it enables contrastive explanations.
But it means every GradCAM heatmap is relative to a specific class choice,
and presenting a heatmap without specifying the target class is misleading.

In practice, users typically compute GradCAM for the TOP-1 predicted class.
But for understanding a WRONG prediction, computing GradCAM for the true
class (not the predicted class) often reveals WHY the model was confused.


### Limitation 5 — Global Average Pooling Discards Spatial Gradient Structure

The global average pooling of the gradient treats all positions equally:
αᵏ_c = (1/HW) Σᵢⱼ ∂S_c/∂Aᵏᵢⱼ

But the gradient ∂S_c/∂Aᵏᵢⱼ is NOT uniform across spatial positions.
Consider a channel that:
  • Has large positive gradient at the dog's eye position (i₀, j₀)
  • Has large negative gradient at the background position (i₁, j₁)
  • Has zero gradient everywhere else

The global average pools these together: αᵏ_c ≈ 0 (positive and negative
cancel). GradCAM would assign near-zero importance to this channel and
EXCLUDE the dog's eye region from the heatmap — despite the eye being
genuinely important (as the large local gradient indicates).

This cancellation failure was the motivation for Grad-CAM++, which uses
pixel-level weighting to avoid averaging away spatially heterogeneous gradients.


##### PART VIII: PRACTICAL APPLICATIONS AND USE PATTERNS

### Application 1 — Model Debugging

GradCAM is the most commonly used tool for identifying spurious correlations
in CNN classifiers. The canonical failure mode:

    A skin cancer classifier trained on clinical photographs achieves 89%
    AUC on the test set. GradCAM maps for "malignant" predictions consistently
    highlight the RULER at the bottom of the image (placed by clinicians to
    show scale), not the lesion itself. The model learned: "images with rulers
    are more likely to be malignant" — because malignant lesions are more
    likely to be photographed with a scale reference.

    GradCAM makes this spurious correlation immediately visible. Without it,
    the high test AUC would be taken as validation of a clinically useful model.

Other debugging uses:
  • Confirming the model attends to the object and not the background.
  • Checking whether a model ignores important context (e.g., "is this an
    outdoor scene?" might require attending to both foreground and background).
  • Identifying examples where the model attends to correct regions for wrong
    predictions (the model is confused for a different reason than region selection).


### Application 2 — Weakly Supervised Object Localisation

GradCAM heatmaps can be used to generate bounding boxes for objects WITHOUT
providing bounding box labels during training. This is "weakly supervised
localisation" — using class labels only, not location labels.

Procedure:
  1. Train a classification-only CNN (no bounding boxes).
  2. For each image, compute GradCAM for the top predicted class.
  3. Threshold the heatmap: pixels above a threshold (e.g., 20% of max) are
     "object pixels."
  4. Take the bounding box of the connected region with the highest sum.

Performance: GradCAM-based localisation achieves ~50–60% correct localisation
(within 0.5 IoU threshold) on ImageNet, compared to ~75–80% for fully
supervised detection. The gap is the price of not providing location labels.

This is practically valuable in settings where classification labels are
abundant but annotation is expensive (medical imaging, satellite imagery).


### Application 3 — Contrastive Explanations

What makes class A different from class B, according to the model?

Compute GradCAM for class A and GradCAM for class B on the same image.
Subtract: L_A − L_B. Positive regions: support A over B. Negative regions:
support B over A.

    Example: same image, "golden retriever" vs "Labrador"
    L_{golden} − L_{Labrador} highlights the golden-coloured fur and
    the specific shape of the face. This shows WHAT FEATURE the model
    uses to distinguish these breeds — often more informative than
    either individual heatmap.


### Application 4 — Monitoring Prediction Confidence

GradCAM heatmaps can be used as a confidence signal beyond the softmax
probability. If the heatmap for the top class is:
  • Spatially concentrated on one semantically coherent region → HIGH CONFIDENCE
    (the model has a clear spatial reason for its prediction)
  • Diffuse across the whole image or scattered → LOW CONFIDENCE
    (the model may be guessing based on global image statistics)

This "spatial concentration" metric is sometimes called ENTROPY of the
GradCAM heatmap, and it provides a model-agnostic uncertainty signal that
does not require calibration or ensembling.


##### PART IX: A COMPLETE WORKED EXAMPLE — GRADCAM BY HAND

### A 4×4 Input, 2 Channels, 2×2 Feature Maps

We compute GradCAM by hand for a minimal network to make every
arithmetic step explicit and visible.

    SETUP:
    ──────
    Input image x ∈ ℝ^{4×4}  (grayscale for simplicity)
    Last conv layer: 2 feature maps (2 channels), each 2×2 spatial.
    Two classes: 0 (DENY) and 1 (APPROVE).
    We explain the class-1 (APPROVE) prediction.

    Feature maps (output of last conv layer):
      Channel 0 (A⁰): encodes "upper-left activity"
        A⁰ = [[2.5, 0.3],     ← (0,0)=strong, (0,1)=weak
               [0.1, 0.0]]     ← (1,0)=near-zero, (1,1)=zero

      Channel 1 (A¹): encodes "lower-right activity"
        A¹ = [[0.0, 0.1],     ← (0,0)=zero, (0,1)=weak
               [0.2, 3.1]]     ← (1,0)=weak, (1,1)=strong

    Both feature maps have H=2, W=2, so H×W=4.

    Score computation (assume a simple FC after Global Average Pool):
      gap⁰ = (2.5 + 0.3 + 0.1 + 0.0) / 4 = 0.725
      gap¹ = (0.0 + 0.1 + 0.2 + 3.1) / 4 = 0.850

      S₀ (DENY)   = −0.3 × gap⁰ + 0.5 × gap¹ = −0.218 + 0.425 = +0.207
      S₁ (APPROVE)= +0.8 × gap⁰ − 0.2 × gap¹ = +0.580 − 0.170 = +0.410

    Prediction: APPROVE (S₁ > S₀). ✓

    STEP 1 — Gradient of S₁ w.r.t. each feature map position:

    Under the GAP architecture:
    ∂S₁ / ∂A⁰ᵢⱼ = w₁⁰ / H×W = 0.8 / 4 = +0.200  (same for all i,j)
    ∂S₁ / ∂A¹ᵢⱼ = w₁¹ / H×W = −0.2 / 4 = −0.050  (same for all i,j)

    Gradient maps:
      ∂S₁/∂A⁰ = [[+0.200, +0.200],
                  [+0.200, +0.200]]

      ∂S₁/∂A¹ = [[−0.050, −0.050],
                  [−0.050, −0.050]]

    STEP 2 — Global average pool the gradients:

    α⁰₁ = (1/4) Σ ∂S₁/∂A⁰ᵢⱼ = (1/4)(0.200×4) = +0.200
    α¹₁ = (1/4) Σ ∂S₁/∂A¹ᵢⱼ = (1/4)(−0.050×4) = −0.050

    Channel 0 has POSITIVE importance for APPROVE (+0.200).
    Channel 1 has NEGATIVE importance for APPROVE (−0.050).
    (Channel 1 actually HURTS the APPROVE score — it is anti-evidence.)

    STEP 3 — Weighted sum of feature maps:

    Σₖ αᵏ₁ × Aᵏ = α⁰₁ × A⁰ + α¹₁ × A¹

    Position (0,0): 0.200×2.5 + (−0.050)×0.0 = 0.500 + 0.000 = +0.500
    Position (0,1): 0.200×0.3 + (−0.050)×0.1 = 0.060 − 0.005 = +0.055
    Position (1,0): 0.200×0.1 + (−0.050)×0.2 = 0.020 − 0.010 = +0.010
    Position (1,1): 0.200×0.0 + (−0.050)×3.1 = 0.000 − 0.155 = −0.155

    Raw combined map:
      [[+0.500, +0.055],
       [+0.010, −0.155]]

    STEP 4 — Apply ReLU:

    L^GradCAM₁ = ReLU([[+0.500, +0.055],
                        [+0.010, −0.155]])
               = [[0.500, 0.055],
                  [0.010, 0.000]]

    STEP 5 — Normalise to [0,1]:

    Max = 0.500
    L_normalised = [[1.000, 0.110],
                    [0.020, 0.000]]

    STEP 6 — Interpret the heatmap:

    ┌──────────────────────────────────────────────────────────┐
    │  GradCAM for APPROVE class:                              │
    │                                                          │
    │  Position  (0,0): 1.000 ← HOTTEST (top-left region)      │
    │  Position  (0,1): 0.110 ← Warm    (top-right)            │
    │  Position  (1,0): 0.020 ← Cool    (bottom-left)          │
    │  Position  (1,1): 0.000 ← COLD    (bottom-right)         │
    │                                                          │
    │  The model focuses on the TOP-LEFT region of the image   │
    │  when predicting APPROVE. Channel 0 (upper-left activity)│
    │  drives this — it has positive importance (α⁰₁ = +0.200) │
    │  and its strongest activation is at (0,0) = 2.5.         │
    │                                                          │
    │  The bottom-right is cold because:                       │
    │  • Channel 0 is zero there (no activation).              │
    │  • Channel 1 is strong (3.1) but has NEGATIVE αᵏ₁,       │
    │    which after ReLU is zeroed out (not shown as anti-    │
    │    evidence, just excluded from the positive heatmap).   │
    └──────────────────────────────────────────────────────────┘

    WHAT WOULD CHANGE IF WE COMPUTED GRADCAM FOR CLASS 0 (DENY)?

    α⁰₀ = w₀⁰ / H×W = −0.3 / 4 = −0.075
    α¹₀ = w₀¹ / H×W = +0.5 / 4 = +0.125

    Weighted sum at (1,1): (−0.075)×0.0 + 0.125×3.1 = +0.388  ← HOT
    Weighted sum at (0,0): (−0.075)×2.5 + 0.125×0.0 = −0.188  ← negative (ReLU → 0)

    GradCAM for DENY:
    [[0.000, 0.000],
     [0.001, 0.388]]  (normalised: [[0,0],[0,1]])

    For the DENY class, the BOTTOM-RIGHT region is hottest (Channel 1 activity).
    Contrastive observation: APPROVE and DENY use opposite regions of the image.
    The model makes its decision by comparing the upper-left vs lower-right signal.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔══════════════════════╦═══════════════╦══════════════╦═════════════╦══════════════╗
    ║ Property             ║ CAM           ║ GradCAM      ║ GradCAM++   ║ ScoreCAM     ║
    ╠══════════════════════╬═══════════════╬══════════════╬═════════════╬══════════════╣
    ║ Architecture req.    ║ GAP + linear  ║ Any CNN      ║ Any CNN     ║ Any CNN      ║
    ║ Cost (fwd/bwd pass)  ║ 1 fwd         ║ 1 fwd+bwd    ║ 1 fwd+bwd   ║ C fwd        ║
    ║ Gradient-free        ║ Yes           ║ No           ║ No          ║ Yes          ║
    ║ Heatmap resolution   ║ Last conv H×W ║ Last conv HW ║ Last conv   ║ Last conv    ║
    ║ Multi-instance       ║ Partial       ║ Poor         ║ Better      ║ Good         ║
    ║ Adebayo sanity check ║ Pass          ║ Pass         ║ Pass        ║ Pass         ║
    ║ Class-discriminative ║ Yes           ║ Yes          ║ Yes         ║ Yes          ║
    ║ Requires retraining  ║ No (if GAP)   ║ No           ║ No          ║ No           ║
    ╚══════════════════════╩═══════════════╩══════════════╩═════════════╩══════════════╝

    Core GradCAM formula:
      αᵏ_c = (1/HW) Σᵢ Σⱼ ∂S_c / ∂Aᵏᵢⱼ         ← importance weight for channel k
      L^GradCAM_c = ReLU(Σₖ αᵏ_c × Aᵏ)          ← weighted feature maps + ReLU

    GradCAM = CAM when architecture is GAP + linear:
      ∂S_c/∂Aᵏᵢⱼ = w_c^k / HW  →  αᵏ_c = w_c^k / HW  →  L ∝ CAM_c
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "GradCAM from Scratch — Full Pipeline with Contrastive and ScoreCAM": {
        "description": (
            "Implements a small CNN (conv → relu → conv → global avg pool → linear) "
            "from scratch in pure Python. Computes GradCAM for two classes on the same "
            "input, showing the contrastive heatmaps. Then implements a ScoreCAM variant "
            "to compare gradient-based vs perturbation-based importance. Traces every "
            "step: forward pass, gradient backpropagation through conv layers, global "
            "average pooling, weighted feature map combination, ReLU, and normalisation. "
            "Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "gradcam",
        "code": '''
"""
================================================================================
GRADCAM FROM SCRATCH — FULL PIPELINE WITH CONTRASTIVE AND SCORECAM
================================================================================

We build a minimal CNN from scratch and implement:
  1. GradCAM for the predicted class
  2. GradCAM for the second class (contrastive)
  3. ScoreCAM (perturbation-based, gradient-free)
  4. A comparison of all three heatmaps side by side

Network architecture:
  Input: 8×8 "image" (1 channel, grayscale)
  Conv1: 4 filters, 3×3 kernel, padding=1 → (4, 8, 8)
  ReLU
  Conv2 (LAST CONV): 3 filters, 3×3 kernel, stride=2 → (3, 4, 4)
  ReLU
  Global Average Pool: (3, 4, 4) → (3,)
  Linear: (3,) → (2,)   [DENY=0, APPROVE=1]

This network is small enough to trace exactly by hand while being realistic
enough to exhibit genuine GradCAM behaviour.
================================================================================
"""

import math
import random

random.seed(23)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return max(0.0, x)

def relu_deriv(x):
    return 1.0 if x > 0.0 else 0.0

def conv2d(x, weight, bias, stride=1, padding=0):
    """
    x:      (C_in, H, W)
    weight: (C_out, C_in, kH, kW)
    bias:   (C_out,)
    Returns: (C_out, H_out, W_out)
    """
    C_in, H, W = len(x), len(x[0]), len(x[0][0])
    C_out, _, kH, kW = len(weight), len(weight[0]), len(weight[0][0]), len(weight[0][0][0])
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1

    # Pad input
    if padding > 0:
        padded = [[[0.0]*(W + 2*padding) for _ in range(H + 2*padding)]
                  for _ in range(C_in)]
        for c in range(C_in):
            for i in range(H):
                for j in range(W):
                    padded[c][i+padding][j+padding] = x[c][i][j]
    else:
        padded = x

    out = [[[0.0]*W_out for _ in range(H_out)] for _ in range(C_out)]
    for co in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                val = bias[co]
                for ci in range(C_in):
                    for ki in range(kH):
                        for kj in range(kW):
                            val += (weight[co][ci][ki][kj] *
                                    padded[ci][i*stride+ki][j*stride+kj])
                out[co][i][j] = val
    return out

def apply_relu(x):
    """x: (C, H, W) → same shape with ReLU applied."""
    return [[[relu(x[c][i][j]) for j in range(len(x[c][0]))]
             for i in range(len(x[c]))]
            for c in range(len(x))]

def global_avg_pool(x):
    """x: (C, H, W) → (C,) via spatial average."""
    C, H, W = len(x), len(x[0]), len(x[0][0])
    return [sum(x[c][i][j] for i in range(H) for j in range(W)) / (H*W)
            for c in range(C)]

def linear(x, weight, bias):
    """x: (d_in,), weight: (d_out, d_in), bias: (d_out,) → (d_out,)"""
    return [bias[o] + sum(weight[o][i]*x[i] for i in range(len(x)))
            for o in range(len(weight))]

def softmax(logits):
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v/s for v in e]


# ─────────────────────────────────────────────────────────────────────────────
# FIXED WEIGHTS (simulating a trained model)
# ─────────────────────────────────────────────────────────────────────────────

# Conv1: (4 filters, 1 in-channel, 3×3 kernel)
W1 = [[[[ 0.3, -0.2,  0.1],
        [ 0.4,  0.5, -0.3],
        [-0.1,  0.2,  0.4]]],
      [[[-0.2,  0.4,  0.3],
        [ 0.1, -0.5,  0.2],
        [ 0.3, -0.1,  0.4]]],
      [[[ 0.5,  0.1, -0.4],
        [-0.2,  0.3,  0.1],
        [ 0.4,  0.2, -0.3]]],
      [[[-0.1,  0.3,  0.2],
        [ 0.5, -0.4,  0.1],
        [-0.3,  0.4,  0.2]]]]
b1 = [-0.1, 0.1, -0.05, 0.05]

# Conv2 (LAST): (3 filters, 4 in-channels, 3×3 kernel, stride=2)
# This layer is the "last conv layer" GradCAM targets.
W2 = [[[[ 0.6, -0.3,  0.2],
        [ 0.1,  0.5, -0.4],
        [-0.2,  0.3,  0.6]],
       [[-0.3,  0.2,  0.4],
        [ 0.5, -0.1,  0.3],
        [ 0.2, -0.4,  0.1]],
       [[ 0.4,  0.1, -0.3],
        [-0.2,  0.6,  0.2],
        [ 0.3, -0.1,  0.5]],
       [[ 0.2, -0.4,  0.1],
        [ 0.3,  0.2, -0.5],
        [-0.1,  0.4,  0.3]]],

      [[[-0.4,  0.3,  0.1],
        [ 0.2, -0.6,  0.4],
        [ 0.5,  0.1, -0.3]],
       [[ 0.3, -0.2,  0.5],
        [-0.4,  0.1,  0.3],
        [ 0.2,  0.6, -0.1]],
       [[-0.1,  0.4, -0.2],
        [ 0.6,  0.3, -0.4],
        [-0.2,  0.1,  0.5]],
       [[ 0.5, -0.1,  0.3],
        [-0.2,  0.4,  0.1],
        [ 0.3, -0.5,  0.2]]],

      [[[ 0.2,  0.5, -0.1],
        [-0.4,  0.3,  0.6],
        [ 0.1, -0.2,  0.4]],
       [[ 0.4, -0.3,  0.2],
        [ 0.1,  0.5, -0.4],
        [-0.3,  0.2,  0.6]],
       [[-0.2,  0.4,  0.3],
        [ 0.5, -0.1,  0.2],
        [ 0.3, -0.4,  0.1]],
       [[ 0.6,  0.1, -0.3],
        [-0.2,  0.4,  0.5],
        [ 0.1, -0.3,  0.2]]]]
b2 = [0.0, 0.0, 0.0]

# FC: (2 classes, 3 features from GAP)
W_fc = [[-0.7,  0.4,  0.5],   # DENY weights
        [ 0.8, -0.3, -0.6]]   # APPROVE weights
b_fc = [0.1, -0.1]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC INPUT IMAGE (8×8, 1 channel)
# Structured: brighter in upper-left, darker in lower-right
# ─────────────────────────────────────────────────────────────────────────────

def make_image():
    img = [[[0.0]*8 for _ in range(8)]]
    for i in range(8):
        for j in range(8):
            # Gradient: strong signal in upper-left, weak in lower-right
            img[0][i][j] = max(0.0, 1.0 - (i + j) / 10.0 + random.gauss(0, 0.05))
    return img

image = make_image()

print("=" * 68)
print("  INPUT IMAGE (8×8, 1 channel, brighter = upper-left)")
print("=" * 68)
print()
for i in range(8):
    row = "  "
    for j in range(8):
        v = image[0][i][j]
        ch = "█" if v > 0.7 else ("▓" if v > 0.5 else ("░" if v > 0.3 else " "))
        row += ch + ch
    print(row)
print()
print("  (█ = bright, ▓ = medium, ░ = dim,   = dark)")


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS — save intermediate activations for GradCAM
# ─────────────────────────────────────────────────────────────────────────────

def forward_pass(x):
    """Full forward pass. Returns (logits, last_conv_preact, last_conv_act, gap)."""
    z1 = conv2d(x, W1, b1, stride=1, padding=1)  # (4, 8, 8)
    a1 = apply_relu(z1)                            # (4, 8, 8)
    z2 = conv2d(a1, W2, b2, stride=2, padding=1)  # (3, 4, 4) — LAST CONV
    a2 = apply_relu(z2)                            # (3, 4, 4) — LAST CONV ACTS
    gap = global_avg_pool(a2)                      # (3,)
    logits = linear(gap, W_fc, b_fc)               # (2,)
    return logits, z2, a2, gap

logits, z2, A, gap = forward_pass(image)
probs = softmax(logits)
pred_class = 0 if logits[0] > logits[1] else 1

print("=" * 68)
print("  FORWARD PASS RESULTS")
print("=" * 68)
print(f"  Logits:  DENY={logits[0]:.4f},  APPROVE={logits[1]:.4f}")
print(f"  Probabilities: DENY={probs[0]:.1%},  APPROVE={probs[1]:.1%}")
print(f"  Prediction: {'DENY' if pred_class==0 else 'APPROVE'}")
print()

C2, H2, W2_size = len(A), len(A[0]), len(A[0][0])
print(f"  Last conv layer feature maps (channel × spatial):")
for k in range(C2):
    vals = [f"{A[k][i][j]:.2f}" for i in range(H2) for j in range(W2_size)]
    print(f"  Channel {k}: " + "  ".join(vals[:4]) + " / " + "  ".join(vals[4:]))


# ─────────────────────────────────────────────────────────────────────────────
# GRADCAM IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(target_class, A, z2, gap, W_fc):
    """
    Compute GradCAM for target_class.

    Under the GAP architecture:
      ∂S_c / ∂A^k_{ij} = w_c^k / (H×W)
      α^k_c = w_c^k / (H×W)  (uniform gradient → equal to the FC weight / HW)
    """
    C, H, W = len(A), len(A[0]), len(A[0][0])
    HW = H * W

    # Gradient of S_c w.r.t. last-conv pre-activations z2
    # (After GAP and FC, under chain rule via ReLU and GAP:)
    # ∂S_c/∂z2^k_{ij} = w_c^k / HW × relu_deriv(z2^k_{ij})
    # For GradCAM we compute gradient w.r.t. A (post-relu activations)
    # which gives ∂S_c/∂A^k_{ij} = w_c^k / HW (constant across spatial)

    # Compute importance weights α^k_c
    alpha = [W_fc[target_class][k] / HW for k in range(C)]

    print(f"  Importance weights α^k_{target_class}:")
    for k in range(C):
        print(f"    Channel {k}: α = {W_fc[target_class][k]:.3f} / {HW} "
              f"= {alpha[k]:.5f}")

    # Weighted sum of feature maps
    raw = [[0.0]*W for _ in range(H)]
    for k in range(C):
        for i in range(H):
            for j in range(W):
                raw[i][j] += alpha[k] * A[k][i][j]

    print(f"  Weighted sum (before ReLU):")
    for i in range(H):
        row_str = "    " + "  ".join(f"{raw[i][j]:+.4f}" for j in range(W))
        print(row_str)

    # Apply ReLU
    cam = [[max(0.0, raw[i][j]) for j in range(W)] for i in range(H)]

    # Normalise to [0,1]
    max_val = max(cam[i][j] for i in range(H) for j in range(W))
    if max_val > 0:
        cam_norm = [[cam[i][j] / max_val for j in range(W)] for i in range(H)]
    else:
        cam_norm = cam

    return cam_norm, alpha

print()
print("=" * 68)
print("  GRADCAM — APPROVE CLASS (class 1)")
print("=" * 68)
cam_approve, alpha_approve = compute_gradcam(1, A, z2, gap, W_fc)

print(f"  GradCAM heatmap (normalised, 4×4 → will upsample to 8×8):")
for i in range(len(cam_approve)):
    bar_row = "  "
    for j in range(len(cam_approve[0])):
        v = cam_approve[i][j]
        bar = "█" if v > 0.75 else ("▓" if v > 0.50 else ("░" if v > 0.25 else "·"))
        bar_row += f"  {v:.3f}{bar}"
    print(bar_row)

print()
print("=" * 68)
print("  GRADCAM — DENY CLASS (class 0)  [Contrastive]")
print("=" * 68)
cam_deny, alpha_deny = compute_gradcam(0, A, z2, gap, W_fc)

print(f"  GradCAM heatmap (normalised, 4×4):")
for i in range(len(cam_deny)):
    bar_row = "  "
    for j in range(len(cam_deny[0])):
        v = cam_deny[i][j]
        bar = "█" if v > 0.75 else ("▓" if v > 0.50 else ("░" if v > 0.25 else "·"))
        bar_row += f"  {v:.3f}{bar}"
    print(bar_row)


# ─────────────────────────────────────────────────────────────────────────────
# CONTRASTIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  CONTRASTIVE HEATMAP  (APPROVE − DENY)")
print("=" * 68)
H2, W2_sz = len(cam_approve), len(cam_approve[0])
print()
print("  Positive = supports APPROVE over DENY")
print("  Negative = supports DENY over APPROVE")
print()
print(f"  {'Position':<12}  {'APPROVE':>9}  {'DENY':>9}  {'Contrast':>9}  {'Supports'}")
print(f"  {'-'*60}")
for i in range(H2):
    for j in range(W2_sz):
        va = cam_approve[i][j]
        vd = cam_deny[i][j]
        contrast = va - vd
        lbl = "APPROVE ▲" if contrast > 0.1 else ("DENY ▼" if contrast < -0.1 else "neutral")
        print(f"  ({i},{j})       {va:>9.3f}  {vd:>9.3f}  {contrast:>+9.3f}  {lbl}")


# ─────────────────────────────────────────────────────────────────────────────
# SCORECAM — PERTURBATION-BASED (GRADIENT-FREE)
# ─────────────────────────────────────────────────────────────────────────────

def upsample_2x(map_2d, target_H, target_W):
    """
    Simple bilinear upsampling of a (H,W) map to (target_H, target_W).
    """
    H, W = len(map_2d), len(map_2d[0])
    out = [[0.0]*target_W for _ in range(target_H)]
    for ni in range(target_H):
        for nj in range(target_W):
            # Source coordinates (float)
            si = ni * (H - 1) / max(target_H - 1, 1)
            sj = nj * (W - 1) / max(target_W - 1, 1)
            # Bilinear interpolation
            i0, j0 = int(si), int(sj)
            i1 = min(i0 + 1, H - 1)
            j1 = min(j0 + 1, W - 1)
            di, dj = si - i0, sj - j0
            out[ni][nj] = (map_2d[i0][j0] * (1-di) * (1-dj) +
                           map_2d[i0][j1] * (1-di) * dj +
                           map_2d[i1][j0] * di * (1-dj) +
                           map_2d[i1][j1] * di * dj)
    return out

def scorecam(image, A, target_class, W_fc, b_fc):
    """
    ScoreCAM: for each channel k, mask the image by the channel's feature map
    (upsampled to input size), run forward pass, record score improvement.
    α^k_c = S_c(masked) − S_c(baseline)
    """
    C, H_feat, W_feat = len(A), len(A[0]), len(A[0][0])
    H_img, W_img = len(image[0]), len(image[0][0])

    # Baseline: zero image
    baseline = [[[0.0]*W_img for _ in range(H_img)]]
    baseline_logits, _, _, _ = forward_pass(baseline)
    baseline_score = baseline_logits[target_class]

    scores = []
    for k in range(C):
        # Normalise feature map to [0,1]
        feat_k = A[k]
        flat = [feat_k[i][j] for i in range(H_feat) for j in range(W_feat)]
        fmin, fmax = min(flat), max(flat)
        if fmax - fmin > 1e-9:
            norm_k = [[(feat_k[i][j]-fmin)/(fmax-fmin) for j in range(W_feat)]
                      for i in range(H_feat)]
        else:
            norm_k = [[0.0]*W_feat for _ in range(H_feat)]

        # Upsample to image size
        mask = upsample_2x(norm_k, H_img, W_img)

        # Apply mask to image
        masked = [[[image[0][i][j] * mask[i][j] for j in range(W_img)]
                   for i in range(H_img)]]

        # Forward pass on masked image
        masked_logits, _, _, _ = forward_pass(masked)
        masked_score = masked_logits[target_class]

        # ScoreCAM importance: score improvement over baseline
        alpha_k = masked_score - baseline_score
        scores.append(alpha_k)

    return scores

print()
print("=" * 68)
print("  SCORECAM — GRADIENT-FREE PERTURBATION-BASED")
print("=" * 68)
print()
print(f"  For each of the {C2} channels in the last conv layer:")
print(f"  Mask the image by that channel's upsampled feature map.")
print(f"  Run forward pass on masked image.")
print(f"  α^k = S_class(masked) − S_class(baseline=0)")
print()

score_alpha_approve = scorecam(image, A, 1, W_fc, b_fc)
score_alpha_deny    = scorecam(image, A, 0, W_fc, b_fc)

print(f"  {'Channel':>8}  {'Grad α (approve)':>18}  {'Score α (approve)':>18}  {'Agreement?'}")
print(f"  {'-'*62}")
for k in range(C2):
    ga = alpha_approve[k]
    sa = score_alpha_approve[k]
    agree = "✓" if (ga > 0 and sa > 0) or (ga < 0 and sa < 0) or (abs(ga)<0.01 and abs(sa)<0.01) else "✗"
    print(f"  {k:>8}  {ga:>+18.5f}  {sa:>+18.5f}  {agree}")

# Build ScoreCAM heatmap
def build_scorecam_heatmap(A, score_alphas):
    C, H, W = len(A), len(A[0]), len(A[0][0])
    raw = [[0.0]*W for _ in range(H)]
    for k in range(C):
        for i in range(H):
            for j in range(W):
                raw[i][j] += score_alphas[k] * A[k][i][j]
    cam = [[max(0.0, raw[i][j]) for j in range(W)] for i in range(H)]
    max_val = max(cam[i][j] for i in range(H) for j in range(W))
    if max_val > 0:
        return [[cam[i][j]/max_val for j in range(W)] for i in range(H)]
    return cam

scorecam_heatmap = build_scorecam_heatmap(A, score_alpha_approve)

print()
print("  ScoreCAM heatmap for APPROVE (normalised 4×4):")
for i in range(H2):
    bar_row = "  "
    for j in range(W2_sz):
        v = scorecam_heatmap[i][j]
        bar = "█" if v > 0.75 else ("▓" if v > 0.50 else ("░" if v > 0.25 else "·"))
        bar_row += f"  {v:.3f}{bar}"
    print(bar_row)


# ─────────────────────────────────────────────────────────────────────────────
# FINAL THREE-WAY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  THREE-WAY HEATMAP COMPARISON — APPROVE class")
print("=" * 68)
print()
print(f"  {'Position':>10}  {'GradCAM':>9}  {'ScoreCAM':>9}  {'Ranked?'}")
print(f"  {'-'*46}")

pos_vals = [(i, j, cam_approve[i][j], scorecam_heatmap[i][j])
            for i in range(H2) for j in range(W2_sz)]

# Rank positions by GradCAM value
grad_ranks = sorted(range(len(pos_vals)), key=lambda x: -pos_vals[x][2])
score_ranks = sorted(range(len(pos_vals)), key=lambda x: -pos_vals[x][3])

for idx, (i, j, gv, sv) in enumerate(pos_vals):
    gr = grad_ranks.index(idx) + 1
    sr = score_ranks.index(idx) + 1
    agree = f"G#{gr}=S#{sr}" if gr == sr else f"G#{gr} vs S#{sr}"
    print(f"  ({i},{j})       {gv:>9.3f}  {sv:>9.3f}  {agree}")

print()
print(f"  INTERPRETATION:")
print(f"  • GradCAM uses the FC weight × feature activation (fast, 1 fwd+bwd pass)")
print(f"  • ScoreCAM tests each channel by masking the input (slow, {C2} fwd passes)")
print(f"  • Where they AGREE: strong evidence that the region is truly important.")
print(f"  • Where they DISAGREE: gradient may be saturated or misleading.")
print()
print(f"  Both methods focus on the same class-relevant spatial regions when")
print(f"  the gradient is unsaturated — confirming GradCAM's practical validity.")
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
    #     from interpretability.visuals.gradcam import (
    #         GRADCAM_VISUAL_HTML,
    #         GRADCAM_VISUAL_HEIGHT,
    #     )
    #     visual_html   = GRADCAM_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = GRADCAM_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[gradcam.py] Could not load visual: {e}", stacklevel=2)

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