"""
Activation Functions & Loss Functions
======================================

Two of the most fundamental design choices in any neural network.
Activations determine what patterns a network can express.
Loss functions determine what it is actually trying to learn.
Wrong choices in either category silently cripple training.

"""

import textwrap
import re

TOPIC_NAME   = "Activation Functions & Loss Functions"
DISPLAY_NAME = "09 · Activations & Loss Functions"
ICON         = "⚡"
SUBTITLE     = "The Non-Linearity That Gives Networks Power & The Signal That Guides Training"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — ACTIVATION FUNCTIONS

### Why Do We Need Activation Functions?

Without an activation function, a neural network with any number of layers
collapses to a single linear transformation. No matter how many layers you
stack, the composition of linear functions is still linear:

    W₂(W₁x + b₁) + b₂  =  (W₂W₁)x + (W₂b₁ + b₂)  =  Wx + b

    → A 100-layer network with no activations has the same expressive
      power as a single-layer linear regression.

Activation functions introduce NON-LINEARITY, allowing the network to
approximate any continuous function (Universal Approximation Theorem).

Three practical requirements for a good activation function:
    1. Non-linear — enables the network to learn complex mappings
    2. Differentiable (almost everywhere) — enables backpropagation
    3. Non-saturating (ideally) — prevents vanishing gradients


──────────────────────────────────────────────────────────────────────────────
### Sigmoid  σ(z) = 1 / (1 + e⁻ᶻ)

    Range:    (0, 1)
    Output:   smooth, monotonic S-curve
    Gradient: σ'(z) = σ(z) · (1 − σ(z))    max value = 0.25 at z=0

    Diagram 1 — Sigmoid Function and Its Gradient:

    Output │1.0                   ──────────
           │                  ───
           │0.5         ──────
           │        ────
           │0.0 ────
           └─────────────────────────────── z
                -4    -2     0    2     4

    Gradient│0.25
            │         ╭───╮
            │       ╭─╯   ╰─╮
            │0.0 ───╯       ╰───────────
            └─────────────────────────────── z
                -4    -2     0    2     4

    Properties:
    + Output bounded in (0,1) → natural probability interpretation
    + Smooth everywhere → clean gradients

    - SATURATES for |z| > 4: gradient ≈ 0 → vanishing gradients in deep nets
    - Output NOT zero-centred (always positive) → zig-zagging weight updates
    - exp() is computationally expensive vs ReLU

    When to use:
    ✓ Output layer for BINARY CLASSIFICATION (outputs P(y=1|x))
    ✗ Hidden layers of deep networks (use ReLU or variants instead)


──────────────────────────────────────────────────────────────────────────────
### Tanh  tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)

    Range:    (−1, +1)
    Gradient: tanh'(z) = 1 − tanh²(z)    max value = 1.0 at z=0

    Relationship to sigmoid: tanh(z) = 2σ(2z) − 1
    Tanh is a SCALED and SHIFTED sigmoid.

    Properties:
    + Zero-centred output → gradients don't always push in same direction
    + Stronger gradient than sigmoid at z=0 (max 1.0 vs 0.25)

    - Still saturates for |z| > 2 → vanishing gradients remain a problem
    - Still uses exp() → slower than ReLU

    When to use:
    ✓ Hidden layers when zero-centred output matters (e.g., RNNs)
    ✓ Slightly preferred over sigmoid for hidden layers in shallow nets
    ✗ Deep networks (use ReLU or variants)


──────────────────────────────────────────────────────────────────────────────
### ReLU  (Rectified Linear Unit)  f(z) = max(0, z)

    Range:    [0, ∞)
    Gradient: f'(z) = 1 if z > 0,  0 if z ≤ 0

    Introduced by Nair & Hinton (2010). Transformed deep learning.

    Diagram 2 — ReLU and Its Gradient:

    Output │              ╱
           │             ╱
           │            ╱
           │           ╱
           │──────────╱
           └─────────────────────────────── z
                -4    -2     0    2     4

    Gradient│           ████████████████
            │0.0 ────────
            └─────────────────────────────── z
              (gradient is exactly 0 or 1 — no vanishing for z > 0)

    Properties:
    + No saturation for z > 0 → no vanishing gradient in positive region
    + Computationally trivial: just a max() call
    + Sparse activation: ~50% of neurons output 0 → efficient representations
    + Converges much faster in practice than sigmoid/tanh

    - NOT zero-centred output
    - DYING ReLU PROBLEM: if z < 0 for all training examples, the neuron
      is permanently dead (gradient = 0, no updates ever). Common with bad
      init or large learning rates. Can kill 40%+ of neurons.
    - Gradient is NOT smooth at z=0 (technically not differentiable there,
      but convention sets gradient to 0 at z=0)

    Dying ReLU Example:
    ┌─────────────────────────────────────────────────────────────┐
    │  If w is initialised or updated such that z = w·x + b < 0   │
    │  for ALL training examples:                                 │
    │    → output always 0                                        │
    │    → gradient always 0                                      │
    │    → w and b NEVER update                                   │
    │    → neuron is permanently "dead" for the rest of training  │
    │                                                             │
    │  Fix: He initialisation + smaller learning rate + Leaky ReLU│
    └─────────────────────────────────────────────────────────────┘

    When to use:
    ✓ Default for ALL hidden layers in CNNs and MLPs
    ✓ Fast, simple, works well with He init + BatchNorm
    ✗ Output layers (unbounded — use linear/sigmoid/softmax there)


──────────────────────────────────────────────────────────────────────────────
### Leaky ReLU  f(z) = max(αz, z)    (α typically 0.01)

    Gradient: f'(z) = 1 if z > 0,  α if z ≤ 0

    Fixes the dying ReLU problem: neurons always have a small but non-zero
    gradient for z < 0, so they can recover from a dead state.

    PReLU (Parametric ReLU): α is LEARNED, not fixed.
    Each neuron learns its own α during training.


──────────────────────────────────────────────────────────────────────────────
### ELU  (Exponential Linear Unit)

    f(z) = z           if z > 0
    f(z) = α(eᶻ − 1)  if z ≤ 0       (α usually 1.0)

    Properties:
    + Smooth everywhere, including at z=0 (unlike ReLU)
    + Negative outputs → zero-centred-ish → faster convergence
    + Can push mean activations closer to 0 (reduces internal covariate shift)
    - exp() for negative values → slower than ReLU

──────────────────────────────────────────────────────────────────────────────
### GELU  (Gaussian Error Linear Unit)

    f(z) = z · Φ(z)    where Φ(z) = P(X ≤ z) for X~N(0,1)

    Approximation used in practice:
    f(z) ≈ 0.5z · (1 + tanh(√(2/π) · (z + 0.044715z³)))

    Introduced by Hendrycks & Gimpel (2016).
    STANDARD activation in transformer architectures (BERT, GPT, ViT).

    Intuition: GELU stochastically gates inputs.
    For large positive z: outputs z (like ReLU)
    For large negative z: outputs ≈ 0 (like ReLU)
    Around z=0: smooth, curved transition — unlike ReLU's sharp corner

    Diagram 3 — ReLU vs GELU vs Swish near origin:

    Output │
           │               ╱ ← ReLU (sharp corner at 0)
           │             ╱
           │           ╱~~ ← GELU (smooth, slight dip below 0)
           │         ╱
           │────── ╱
           │    ~~╱ ← slight negative bump (around z = -0.17)
           └─────────────────────────────── z
                -3    -1     0    1     3

    The small negative region around z ≈ -0.17 is a key property:
    GELU can "push back" on slightly negative inputs, giving it more
    expressive power than the hard cutoff of ReLU.

    When to use:
    ✓ Transformer models (BERT, GPT-2/3/4, ViT — universal standard)
    ✓ Any task where ReLU's hard boundary is too crude


──────────────────────────────────────────────────────────────────────────────
### Swish / SiLU  f(z) = z · σ(z)    (σ = sigmoid)

    Also written as SiLU (Sigmoid Linear Unit).
    Proposed by Ramachandran et al. (2017) via automated search.
    Used in EfficientNet, many modern vision models.

    Properties:
    + Non-monotonic (like GELU) → small negative output for z ≈ -1 to 0
    + Smooth everywhere
    + Self-gated: the sigmoid of z gates z itself
    + In practice, similar performance to GELU

    Relationship: Swish(z, β=1) = SiLU(z). As β→∞, Swish→ReLU.


──────────────────────────────────────────────────────────────────────────────
### Softmax  f(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ

    Range: (0, 1) per output, and Σ outputs = 1.0
    Used exclusively as an OUTPUT LAYER for multi-class classification.

    Softmax turns a vector of raw scores (logits) into a probability
    distribution over K classes:

    Input logits: [2.1, 0.3, -0.5]
    After softmax: [0.76, 0.14, 0.10]  (sums to 1.0)

    Properties:
    + Outputs are a valid probability distribution
    + Differentiable → works with cross-entropy loss via clean gradients
    + Numerically stable implementation subtracts max(z) first:
        f(zᵢ) = e^(zᵢ - max(z)) / Σⱼ e^(zⱼ - max(z))
      This prevents overflow (exp of large numbers) without changing output.

    Gradient: ∂softmax(zᵢ)/∂zⱼ = softmax(zᵢ)(δᵢⱼ − softmax(zⱼ))
    (where δᵢⱼ is the Kronecker delta — 1 if i=j, else 0)

    When to use:
    ✓ Output layer for multi-class classification (ONLY use case)
    ✗ Hidden layers (use ReLU/GELU)
    ✗ Binary classification (use sigmoid instead)


──────────────────────────────────────────────────────────────────────────────
### Activation Function Comparison Table

    ┌────────────────┬───────────────┬──────────┬──────────┬────────────────┐
    │ Function       │ Range         │ Saturate │ Zero-ctr │ Best used for  │
    ├────────────────┼───────────────┼──────────┼──────────┼────────────────┤
    │ Sigmoid        │ (0, 1)        │ Yes      │ No       │ Binary output  │
    │ Tanh           │ (-1, 1)       │ Yes      │ Yes      │ RNN hidden     │
    │ ReLU           │ [0, ∞)        │ Half     │ No       │ CNN/MLP hidden │
    │ Leaky ReLU     │ (-∞, ∞)       │ No       │ No       │ CNN (no dying) │
    │ ELU            │ (-α, ∞)       │ Half     │ ~Yes     │ Fast conv nets │
    │ GELU           │ (-0.17, ∞)    │ No       │ No       │ Transformers   │
    │ Swish/SiLU     │ (-0.28, ∞)    │ No       │ No       │ EfficientNet   │
    │ Softmax        │ (0,1), sum=1  │ --       │ No       │ Multi-class out│
    └────────────────┴───────────────┴──────────┴──────────┴────────────────┘

    Modern rule of thumb:
    • Hidden layers, CNN/MLP → ReLU (default) or Leaky ReLU (if dying observed)
    • Hidden layers, Transformers → GELU
    • Output, binary classification → Sigmoid
    • Output, multi-class → Softmax
    • Output, regression → Linear (no activation)


### PART 2 — LOSS FUNCTIONS

### The Connection Between Loss Functions and Probability

Loss functions are not arbitrary — they follow directly from the principle
of Maximum Likelihood Estimation (MLE). We assume a probabilistic model for
the outputs and then choose the loss that maximises the likelihood of the
training data given the model parameters.

    MLE objective:     maximise P(y | x, θ)
    Equivalent to:     minimise -log P(y | x, θ)   (negative log-likelihood)

    If outputs are Gaussian → MSE loss
    If outputs are Bernoulli → Binary Cross-Entropy loss
    If outputs are Categorical → Categorical Cross-Entropy loss

This gives every common loss function a rigorous probabilistic justification.

    Diagram 4 — Loss Family Tree:

    Probability Model Assumed
              │
    ┌─────────┼──────────────────┐
    │         │                  │
    Gaussian  Bernoulli       Categorical
    │         │                  │
    MSE      BCE              Cross-Entropy
    │         │                  │
    (regression)  (binary clf)  (multi-class clf)


──────────────────────────────────────────────────────────────────────────────
### Mean Squared Error (MSE / L2 Loss)

    MSE = (1/N) Σᵢ (yᵢ − ŷᵢ)²

    Gradient:  ∂MSE/∂ŷᵢ = −2/N · (yᵢ − ŷᵢ)

    Probabilistic view:  Assumes yᵢ | xᵢ ~ N(ŷᵢ, σ²)
    Minimising MSE = maximising the likelihood of a Gaussian output model.

    Diagram 5 — MSE Loss Shape (for a single prediction):

    Loss │                 ╲         ╱
         │                  ╲       ╱    ← parabolic: penalty grows quadratically
         │                   ╲     ╱
         │                    ╲   ╱
         │                     ╲╱  ← minimum at ŷ = y
         └──────────────────────────── prediction ŷ

    Properties:
    + Differentiable everywhere → clean gradient flow
    + Heavily penalises LARGE errors (quadratic growth)
    + Unique global minimum

    - Sensitive to OUTLIERS: a single prediction error of 10 contributes
      100 to the loss, drowning out 100 predictions with error 0.1
    - Gradient → 0 as prediction improves → can slow late training

    When to use:
    ✓ Regression with roughly Gaussian errors, no significant outliers
    ✗ When outliers are present (use Huber or MAE instead)


──────────────────────────────────────────────────────────────────────────────
### Mean Absolute Error (MAE / L1 Loss)

    MAE = (1/N) Σᵢ |yᵢ − ŷᵢ|

    Gradient:  ∂MAE/∂ŷᵢ = −1/N · sign(yᵢ − ŷᵢ)

    Probabilistic view:  Assumes yᵢ | xᵢ ~ Laplace(ŷᵢ, b)

    Properties:
    + ROBUST to outliers: error grows linearly, not quadratically
    + Gradient is constant magnitude (±1/N) regardless of error size

    - NOT differentiable at zero (gradient undefined there)
      → use subgradient or switch to Huber near zero
    - Constant gradient near the optimum → slow convergence in final stages
    - Does not penalise large errors as aggressively

    When to use:
    ✓ Regression with outliers or heavy-tailed error distributions
    ✓ When you want the MEDIAN prediction (MAE is median regression)
    ✗ When you need smooth gradients everywhere (use Huber)


──────────────────────────────────────────────────────────────────────────────
### Huber Loss  (Smooth L1)

    Huber_δ(y, ŷ) = {  ½(y − ŷ)²            if |y − ŷ| ≤ δ
                     {  δ · (|y − ŷ| − ½δ)   if |y − ŷ| > δ

    The BEST of both worlds:
    • Quadratic (like MSE) for small errors → smooth gradients near minimum
    • Linear (like MAE) for large errors → outlier robustness

    δ is a threshold hyperparameter (commonly 1.0). It controls where the
    transition from quadratic to linear behaviour occurs.

    Diagram 6 — MSE vs MAE vs Huber Loss Shape:

    Loss │
         │   MSE ╲         ╱       ← quadratic everywhere (outlier sensitive)
         │        ╲       ╱
         │         ╲     ╱
         │   Huber  ╲___╱ ← quadratic near 0, linear in tails
         │            ___
         │   MAE  ───╱   ╲───     ← linear everywhere (V-shape)
         │
         └──────────────────────── |y - ŷ|
               0     δ     2δ

    Gradient of Huber:
    • |error| ≤ δ:   gradient = error (proportional, like MSE)
    • |error| > δ:   gradient = ±δ   (clipped, like MAE)
    → No gradient explosion from outliers ✓

    When to use:
    ✓ Regression with potential outliers (most real-world cases)
    ✓ Object detection (SSD, Faster R-CNN use it for bounding box regression)
    ✓ Reinforcement learning (DQN uses Huber/Smooth-L1 for TD error)


──────────────────────────────────────────────────────────────────────────────
### Binary Cross-Entropy (BCE / Log Loss)

    BCE = −(1/N) Σᵢ [yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ)]

    where yᵢ ∈ {0, 1}  (true label)   and   ŷᵢ ∈ (0, 1)  (sigmoid output)

    Probabilistic view:  Assumes yᵢ | xᵢ ~ Bernoulli(ŷᵢ)
    Minimising BCE = maximising the log-likelihood of a Bernoulli model.

    Diagram 7 — BCE Loss for a Positive Example (y=1):

    Loss │  ← ∞ as ŷ→0: model says 0 but truth is 1 → infinite penalty
         │╲
         │  ╲
         │   ╲
         │    ╲
         │     ╲────────╮
         │              ╰─────── ← 0 as ŷ→1: perfect prediction
         └──────────────────────── predicted probability ŷ
              0.0  0.2  0.4  0.6  0.8  1.0

    Key property: logarithmic penalty means the model is punished
    INFINITELY for being completely wrong (log(0) = −∞).
    This creates strong gradient pressure to move away from confident errors.

    Gradient w.r.t. logit z (before sigmoid):
        ∂BCE/∂z = ŷ − y    ← beautifully simple: just the prediction error!
        This is why sigmoid + BCE is the standard binary classification setup.

    When to use:
    ✓ Binary classification (spam/not spam, disease/healthy, etc.)
    ✓ Multi-label classification (apply sigmoid+BCE independently per label)
    ✗ Multi-class (use Categorical Cross-Entropy + Softmax instead)


──────────────────────────────────────────────────────────────────────────────
### Categorical Cross-Entropy (Multi-class Log Loss)

    CE = −(1/N) Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)

    where yᵢₖ is 1 if example i belongs to class k, else 0 (one-hot)
    and ŷᵢₖ = softmax(zᵢ)ₖ

    In practice, since yᵢₖ = 0 for all classes except the true one:
    CE = −(1/N) Σᵢ log(ŷᵢ,yᵢ)    ← only the true class probability matters

    Gradient w.r.t. logit z (before softmax):
        ∂CE/∂zₖ = ŷₖ − yₖ    ← same elegant form as BCE!

    Softmax + Cross-Entropy is the standard multi-class setup.

    Numerical stability trick — LogSumExp:
    Computing softmax then log separately can cause overflow.
    PyTorch's nn.CrossEntropyLoss and F.cross_entropy take RAW LOGITS
    and compute log-softmax + CE in one numerically stable operation.
    → Always pass raw logits to nn.CrossEntropyLoss, not softmax outputs!


──────────────────────────────────────────────────────────────────────────────
### Hinge Loss  (SVM Loss / Margin Loss)

    Hinge = (1/N) Σᵢ max(0, 1 − yᵢ · ŷᵢ)

    where yᵢ ∈ {−1, +1}  (not 0/1!)  and ŷᵢ is the model score (not a prob)

    Diagram 8 — Hinge vs BCE Loss Shape:

    Loss │
         │╲  BCE (log loss)
         │  ╲
         │   ╲────────────────   ← penalises correct predictions too
         │
         │╲  Hinge loss
         │  ╲
         │   ╲___________________  ← zero loss once margin ≥ 1 (correct + confident)
         └──────────────────────────── y · ŷ
              -2  -1   0   1   2
                        ↑
                    margin=1

    Key property: ZERO LOSS once the prediction is correct AND confident
    (margin ≥ 1). BCE continues penalising even correct predictions.
    This is why SVM margins are sharp boundaries, not soft probabilities.

    When to use:
    ✓ SVMs (its defining loss function)
    ✓ Max-margin classification tasks
    ✗ When you need calibrated probabilities (use BCE/CE instead)


──────────────────────────────────────────────────────────────────────────────
### Focal Loss

    FL(pₜ) = −(1 − pₜ)^γ · log(pₜ)

    where pₜ = ŷ if y=1, else (1−ŷ)    and    γ (gamma) ≥ 0

    Introduced by Lin et al. (Facebook AI, 2017) for object detection.
    Addresses the class imbalance problem by DOWN-WEIGHTING easy examples.

    How it works:
    • Easy examples (model confident and correct): pₜ ≈ 1 → (1−pₜ)^γ ≈ 0
      → focal loss ≈ 0 → almost no gradient from easy examples
    • Hard examples (model wrong or uncertain): pₜ small → (1−pₜ)^γ ≈ 1
      → focal loss ≈ BCE → full gradient from hard examples

    Diagram 9 — Focal Loss vs BCE (γ=0 is plain BCE):

    Loss │  BCE (γ=0)
         │╲
         │  ╲
         │   ╲──── γ=0.5
         │    ╲──── γ=1
         │     ╲──── γ=2
         │      ╲──── γ=5  ← very aggressive down-weighting of easy examples
         └──────────────────────────── pₜ (probability of true class)
              0.0   0.2   0.4   0.6   0.8   1.0

    Effect of γ:
    γ=0 → plain BCE
    γ=2 → easy examples (pₜ=0.9) contribute 100× less than hard ones

    Often combined with class weighting factor α:
    FL(pₜ) = −αₜ · (1 − pₜ)^γ · log(pₜ)

    When to use:
    ✓ Severe class imbalance (1:1000 or worse)
    ✓ Object detection (RetinaNet, YOLO variants)
    ✓ Any task where easy negatives dominate training


──────────────────────────────────────────────────────────────────────────────
### KL Divergence Loss

    KL(P ‖ Q) = Σₓ P(x) · log(P(x) / Q(x))
              = Σₓ P(x) · [log P(x) − log Q(x)]
              = H(P, Q) − H(P)

    where H(P,Q) = cross-entropy, H(P) = entropy of P.

    Interpretation: "The extra bits needed to encode samples from P using
    code designed for Q." Always ≥ 0 (Gibbs' inequality). = 0 iff P = Q.

    KL is NOT symmetric: KL(P‖Q) ≠ KL(Q‖P)
        KL(P‖Q): forward KL, mean-seeking → Q covers all modes of P
        KL(Q‖P): reverse KL, mode-seeking → Q concentrates on one mode of P

    Diagram 10 — Forward vs Reverse KL:

    P (bimodal):   ____    ____           (two peaks)
    Forward KL:    Q spreads to cover both peaks (mean-seeking)
    Reverse KL:    Q snaps to one peak only (mode-seeking)

    When to use:
    ✓ Variational Autoencoders (VAE regularisation term: KL(q(z|x) ‖ p(z)))
    ✓ Knowledge Distillation (soft label matching)
    ✓ Language model training (PPO in RLHF uses KL penalty)
    ✓ Distribution matching problems generally


──────────────────────────────────────────────────────────────────────────────
### Loss Function Selection Guide

    ┌──────────────────────────────┬────────────────────────────────────────┐
    │ Task                         │ Recommended loss                       │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Regression, no outliers      │ MSE                                    │
    │ Regression, with outliers    │ Huber (δ=1) or MAE                     │
    │ Binary classification        │ BCE (sigmoid output)                   │
    │ Multi-class classification   │ Cross-Entropy (softmax output)         │
    │ Multi-label classification   │ BCE per label (sigmoid per output)     │
    │ Imbalanced binary clf        │ Focal Loss (γ=2) or weighted BCE       │
    │ SVM / max-margin             │ Hinge Loss                             │
    │ Probability matching (VAE)   │ KL Divergence                          │
    │ Knowledge distillation       │ KL Divergence (soft targets)           │
    │ Ranking / metric learning    │ Triplet Loss / Contrastive Loss        │
    └──────────────────────────────┴────────────────────────────────────────┘

    Universal red flag: if your loss goes to NaN in early training, check:
    1. Gradient clipping (exploding gradients)
    2. Numerical stability (log(0), exp of large values)
    3. Wrong activation on output layer (e.g., ReLU before BCE)

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · All Activation Functions: Shapes, Gradients & Properties": {
        "description": (
            "Plot every activation function alongside its gradient. "
            "Show numerically exactly where saturation occurs and how "
            "GELU / Swish differ from ReLU near the origin."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erf

# ── Activation functions and their gradients ──────────────────────────────
def sigmoid(z):      return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def d_sigmoid(z):    s = sigmoid(z); return s * (1 - s)

def tanh_(z):        return np.tanh(z)
def d_tanh(z):       return 1 - np.tanh(z)**2

def relu(z):         return np.maximum(0, z)
def d_relu(z):       return (z > 0).astype(float)

def leaky_relu(z, a=0.01): return np.where(z > 0, z, a * z)
def d_leaky_relu(z, a=0.01): return np.where(z > 0, 1.0, a)

def elu(z, a=1.0):   return np.where(z > 0, z, a*(np.exp(np.clip(z,-100,0))-1))
def d_elu(z, a=1.0): return np.where(z > 0, 1.0, elu(z,a) + a)

def gelu(z):
    return 0.5*z*(1 + np.tanh(np.sqrt(2/np.pi)*(z + 0.044715*z**3)))
def d_gelu(z):
    cdf  = 0.5*(1 + erf(z/np.sqrt(2)))
    pdf  = np.exp(-0.5*z**2) / np.sqrt(2*np.pi)
    return cdf + z*pdf

def swish(z):        return z * sigmoid(z)
def d_swish(z):      s = sigmoid(z); return s + z*s*(1-s)

def softmax_single(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

activations = [
    ("Sigmoid",      sigmoid,      d_sigmoid,      "steelblue"),
    ("Tanh",         tanh_,        d_tanh,         "seagreen"),
    ("ReLU",         relu,         d_relu,         "tomato"),
    ("Leaky ReLU",   leaky_relu,   d_leaky_relu,   "orange"),
    ("ELU",          elu,          d_elu,           "purple"),
    ("GELU",         gelu,         d_gelu,          "crimson"),
    ("Swish",        swish,        d_swish,         "teal"),
]

z = np.linspace(-4, 4, 500)

print("=" * 65)
print("  ACTIVATION FUNCTIONS: NUMERICAL PROPERTIES")
print("=" * 65)
print()
print(f"  {'Function':12s} | {'Range':>14} | {'Max grad':>10} | "
      f"{'Grad at z=2':>12} | {'Grad at z=-2':>13}")
print(f"  {'─'*70}")
for name, fn, grad, _ in activations:
    out   = fn(z)
    grads = grad(z)
    print(f"  {name:12s} | "
          f"[{out.min():6.3f}, {out.max():6.3f}] | "
          f"{grads.max():10.4f} | "
          f"{grad(np.array([2.0]))[0]:12.4f} | "
          f"{grad(np.array([-2.0]))[0]:13.4f}")

print()
print("  KEY OBSERVATIONS:")
print("  - Sigmoid/Tanh: gradient at z=±2 is tiny → vanishing gradient risk")
print("  - ReLU: gradient is exactly 0 or 1 → no vanishing for z>0")
print("  - GELU/Swish: non-zero gradient for ALL z → no hard dead zones")
print("  - Leaky ReLU: small but non-zero gradient for z<0 → no dying neurons")
print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(3, len(activations), figsize=(20, 9))
fig2.suptitle("Activation Functions: Output, Gradient & Near-Origin Behaviour",
              fontsize=11, fontweight="bold")

for col, (name, fn, grad_fn, colour) in enumerate(activations):
    out_vals  = fn(z)
    grad_vals = grad_fn(z)
    zoom_z    = np.linspace(-1.5, 1.5, 300)
    zoom_out  = fn(zoom_z)

    # Row 0: function output
    axes2[0, col].plot(z, out_vals, colour, lw=2.5)
    axes2[0, col].axhline(0, color="gray", lw=0.6, linestyle="--")
    axes2[0, col].axvline(0, color="gray", lw=0.6, linestyle="--")
    axes2[0, col].set_title(name, fontsize=10, fontweight="bold", color=colour)
    axes2[0, col].set_ylim(-1.8, 1.8)
    axes2[0, col].set_ylabel("f(z)" if col == 0 else "")
    axes2[0, col].grid(alpha=0.3)

    # Row 1: gradient
    axes2[1, col].plot(z, grad_vals, colour, lw=2, linestyle="--", alpha=0.9)
    axes2[1, col].axhline(0, color="gray", lw=0.6, linestyle="--")
    axes2[1, col].set_ylim(-0.1, 1.3)
    axes2[1, col].set_ylabel("f'(z)" if col == 0 else "")
    axes2[1, col].grid(alpha=0.3)

    # Row 2: zoom near origin
    axes2[2, col].plot(zoom_z, zoom_out, colour, lw=2.5)
    axes2[2, col].plot(zoom_z, fn(zoom_z * 0)*0, "k--", lw=0.6)  # zero line
    axes2[2, col].axhline(0, color="gray", lw=0.6, linestyle="--")
    axes2[2, col].set_xlabel("z")
    axes2[2, col].set_ylabel("zoom [-1.5,1.5]" if col == 0 else "")
    axes2[2, col].set_ylim(-0.8, 1.3)
    axes2[2, col].grid(alpha=0.3)

    # Highlight saturation region for sigmoid/tanh
    if name in ("Sigmoid", "Tanh"):
        for ax_ in [axes2[0,col], axes2[1,col]]:
            ax_.axvspan(2.5, 4, alpha=0.1, color="red", label="Saturation")
            ax_.axvspan(-4, -2.5, alpha=0.1, color="red")

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=110)
print("  Plot saved → activation_functions.png")
print()

# ── Softmax demo ──────────────────────────────────────────────────────────
print("  SOFTMAX DEMO (converts logits to probabilities):")
logit_examples = [
    [2.0, 1.0, 0.1],
    [10.0, 1.0, 0.1],   # high confidence
    [0.1, 0.1, 0.1],    # uniform uncertainty
]
for logits in logit_examples:
    probs = softmax_single(np.array(logits, dtype=float))
    print(f"    Logits {str([round(l,1) for l in logits]):22s} → "
          f"Probs {[round(p,3) for p in probs]}  sum={probs.sum():.4f}")
print()
print("  Note: larger logit differences → more confident predictions")
print("  Subtracting max(logits) before exp prevents overflow (same result)")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Dying ReLU Problem — Demo and Fix": {
        "description": (
            "Demonstrate the dying ReLU problem: show neurons becoming permanently "
            "inactive when learning rate is too large or biases go negative. "
            "Prove that Leaky ReLU is immune even under the same bad conditions."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Core functions ─────────────────────────────────────────────────────────
def relu(z):         return np.maximum(0, z)
def leaky_relu(z):   return np.where(z > 0, z, 0.01 * z)
def sigmoid_(z):     return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

print("=" * 65)
print("  DYING ReLU PROBLEM: DEMO AND FIX")
print("=" * 65)
print()

# ── PART 1: Analytical demonstration ──────────────────────────────────────
# Show how initial pre-activation distribution depends on init scale
print("  PART 1 — PRE-ACTIVATION DISTRIBUTION AT LAYER 0 (analytical)")
print()
N_IN   = 256   # neurons in previous layer
BATCH  = 5000  # samples

x_input = np.random.randn(BATCH, N_IN)  # standardised input

print(f"  {'Init strategy':30s} | {'Mean z':>8} | {'Std z':>8} | {'% z < 0':>10} | {'% z < -3':>10}")
print(f"  {'─'*75}")

for label, W_init, bias_init in [
    ("N(0, 0.01)  small weights",  np.random.randn(N_IN)*0.01,      np.zeros(N_IN)),
    ("N(0, 1.0)   large weights",  np.random.randn(N_IN)*1.0,       np.zeros(N_IN)),
    ("He  N(0,√2/n) + b=0",        np.random.randn(N_IN)*np.sqrt(2/N_IN), np.zeros(N_IN)),
    ("He init + bias = -2.0",       np.random.randn(N_IN)*np.sqrt(2/N_IN), np.full(N_IN, -2.0)),
    ("He init + bias = -5.0",       np.random.randn(N_IN)*np.sqrt(2/N_IN), np.full(N_IN, -5.0)),
]:
    # Each output neuron: z_j = sum_i(x_i * w_j) + b_j
    # Using one representative weight vector
    z = x_input @ W_init.reshape(-1, 1) + bias_init[0]
    z = z.ravel()
    neg_pct  = (z < 0).mean()  * 100
    very_neg = (z < -3).mean() * 100
    print(f"  {label:30s} | {z.mean():8.3f} | {z.std():8.3f} | "
          f"{neg_pct:9.1f}% | {very_neg:9.1f}%")

print()
print("  → bias = -5 forces 99%+ of pre-activations negative → ReLU outputs all 0")
print("    → gradient = 0 → weights NEVER update → permanent death")
print()

# ── PART 2: Training simulation showing accumulation of dead neurons ───────
print("  PART 2 — DEAD NEURON ACCUMULATION DURING TRAINING")
print()

N_H     = 200
N_IN_TR = 20
N_TRAIN = 600
N_EPOCHS = 60
LR_BAD   = 0.15   # large lr causes bias to go negative fast

np.random.seed(1)
X = np.random.randn(N_TRAIN, N_IN_TR)
y = (X[:, 0] + X[:, 1] > 0).astype(float)  # simple linear label

def run_training(activation_fn, init_bias, lr, n_epochs):
    """Train and return per-epoch dead neuron fraction."""
    np.random.seed(42)
    W1 = np.random.randn(N_IN_TR, N_H) * np.sqrt(2 / N_IN_TR)
    b1 = np.full(N_H, init_bias, dtype=float)
    W2 = np.random.randn(N_H, 1) * np.sqrt(2 / N_H)
    b2 = np.zeros(1)
    dead_per_epoch = []

    for ep in range(n_epochs):
        # Forward
        z1   = X @ W1 + b1
        a1   = activation_fn(z1)
        logit= (a1 @ W2 + b2).ravel()
        pred = sigmoid_(logit)

        # Dead metric: neurons with output=0 on >80% of training examples
        dead_frac = ((a1 == 0).mean(axis=0) > 0.80).mean()
        dead_per_epoch.append(dead_frac)

        # Backward (binary cross-entropy)
        dlogit = (pred - y) / N_TRAIN
        dW2    = a1.T @ dlogit.reshape(-1, 1)
        da1    = dlogit.reshape(-1, 1) @ W2.T
        # activation gradient
        if activation_fn is relu:
            dz1 = da1 * (z1 > 0)
        else:
            dz1 = da1 * np.where(z1 > 0, 1.0, 0.01)
        dW1 = X.T @ dz1
        db1_grad = dz1.sum(axis=0)

        W1 -= lr * dW1;  b1 -= lr * db1_grad
        W2 -= lr * dW2;  b2 -= lr * dlogit.sum()

    return np.array(dead_per_epoch)

configs = [
    ("ReLU   + He init, b=0  (ok)",     relu,        0.0,  0.005,   "seagreen"),
    ("ReLU   + neg bias b=-3 (DYING)",  relu,       -3.0,  0.005,   "tomato"),
    ("ReLU   + neg bias b=-5 (DYING)",  relu,       -5.0,  0.005,   "crimson"),
    ("Leaky  + neg bias b=-5 (ok)",     leaky_relu, -5.0,  0.005,   "steelblue"),
]

print(f"  {'Configuration':38s} | {'Dead@ep1':>9} | {'Dead@ep30':>10} | {'Dead@ep60':>10}")
print(f"  {'─'*75}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Dying ReLU Problem: Dead Neuron % During Training",
             fontsize=12, fontweight="bold")

epochs = np.arange(1, N_EPOCHS + 1)
all_dead_histories = []

for name, act_fn, bias0, lr, colour in configs:
    dead_hist = run_training(act_fn, bias0, lr, N_EPOCHS)
    all_dead_histories.append((name, dead_hist, colour))

    d1  = dead_hist[0]  * 100
    d30 = dead_hist[29] * 100
    d60 = dead_hist[-1] * 100
    flag = " ← DYING!" if d60 > 30 else " ✓ healthy"
    print(f"  {name:38s} | {d1:8.1f}% | {d30:9.1f}% | {d60:9.1f}%{flag}")

    axes[0].plot(epochs, dead_hist * 100, colour, lw=2, label=name)

print()
print("  CAUSES of dying ReLU:")
print("  1. Large learning rate → weight updates overshoot → biases go very negative")
print("  2. Negative bias initialisation → pre-activations < 0 from the start")
print("  FIXES:")
print("  1. Leaky ReLU: f(z) = 0.01z for z<0 → gradient never exactly 0")
print("  2. He init + small lr: keeps pre-activations balanced around 0")
print("  3. BatchNorm: normalises pre-activations → prevents extreme negative values")

# Bar chart of final dead %
names_short = [n[:22] for n, _, _ in all_dead_histories]
finals      = [d[-1] * 100 for _, d, _ in all_dead_histories]
colours_bar = [c for _, _, c in all_dead_histories]
axes[1].bar(names_short, finals, color=colours_bar, alpha=0.85)
axes[1].axhline(30, color="red", linestyle="--", lw=1.5, label="30% threshold")
axes[1].set_ylabel("Dead neurons (%) at epoch 60")
axes[1].set_title("Final Dead Neuron % by Configuration")
axes[1].set_xticklabels(names_short, rotation=20, ha="right", fontsize=8)
axes[1].set_ylim(0, 100); axes[1].grid(alpha=0.3, axis="y")
axes[1].legend(fontsize=9)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Dead neurons (%)")
axes[0].set_title("Dead Neuron % Over Training")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 100)

plt.tight_layout()
plt.savefig("dying_relu.png", dpi=120)
print()
print("  Plot saved → dying_relu.png")
''',
    },

        # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · ReLU vs GELU vs Swish in Real Training": {
        "description": (
            "Train the same network architecture with ReLU, GELU, and Swish. "
            "Compare convergence speed, final accuracy, and loss curves. "
            "Verify GELU's advantage in deeper networks."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

def _tp_fp_fn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum(); fn=((yp==0)&(yt==1)).sum()
    return tp,fp,fn

def f1_score(yt,yp,zero_division=0):
    tp,fp,fn=_tp_fp_fn(yt,yp); d=2*tp+fp+fn
    return float(2*tp/d) if d>0 else float(zero_division)

def precision_score(yt,yp,zero_division=0):
    tp,fp,fn=_tp_fp_fn(yt,yp)
    return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)

def recall_score(yt,yp,zero_division=0):
    tp,fp,fn=_tp_fp_fn(yt,yp)
    return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)

def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))

def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return float(np.sum(prec*ys2)/pos)
from scipy.special import erf

np.random.seed(42)

# ── Dataset ───────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=3000, n_features=25, n_informative=12,
                            n_redundant=5, random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)

# ── Implement GELU and Swish as custom sklearn-compatible wrappers ─────────
# sklearn MLPClassifier only supports relu/tanh/logistic/identity.
# We implement our own mini-network for GELU and Swish.

def sigmoid_(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
def relu_(x):    return np.maximum(0, x)
def gelu_(x):    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def swish_(x):   return x * sigmoid_(x)

def d_sigmoid_(x): s = sigmoid_(x); return s*(1-s)
def d_relu_(x):    return (x > 0).astype(float)
def d_gelu_(x):
    cdf = 0.5*(1+erf(x/np.sqrt(2)))
    pdf = np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
    return cdf + x*pdf
def d_swish_(x):
    s = sigmoid_(x); return s + x*s*(1-s)

class MiniMLP:
    """Simple 3-hidden-layer MLP with configurable activation."""
    def __init__(self, dims, act_fn, d_act_fn, lr=0.005, seed=0):
        np.random.seed(seed)
        self.act, self.dact = act_fn, d_act_fn
        self.lr = lr
        self.Ws, self.bs = [], []
        for i in range(len(dims)-1):
            scale = np.sqrt(2/dims[i])
            self.Ws.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.bs.append(np.zeros(dims[i+1]))
        self.caches = []

    def forward(self, X):
        self.caches = []
        A = X
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            Z = A @ W + b
            if i < len(self.Ws) - 1:   # hidden layers
                A_new = self.act(Z)
            else:
                A_new = sigmoid_(Z)     # output always sigmoid
            self.caches.append((A, Z))
            A = A_new
        return A.ravel()

    def backward(self, X, y, preds):
        m = len(y)
        grads_W, grads_b = [], []
        dA = (preds - y).reshape(-1,1) / m

        for i in reversed(range(len(self.Ws))):
            A_prev, Z = self.caches[i]
            if i == len(self.Ws) - 1:
                dZ = dA * d_sigmoid_(Z)
            else:
                dZ = dA * self.dact(Z)
            grads_W.insert(0, A_prev.T @ dZ)
            grads_b.insert(0, dZ.sum(axis=0))
            dA = dZ @ self.Ws[i].T

        for i, (gW, gb) in enumerate(zip(grads_W, grads_b)):
            self.Ws[i] -= self.lr * gW
            self.bs[i] -= self.lr * gb

    def train_epoch(self, X, y, batch=64):
        idx = np.random.permutation(len(y))
        total_loss = 0
        for start in range(0, len(y), batch):
            b = idx[start:start+batch]
            p = self.forward(X[b])
            self.backward(X[b], y[b], p)
            pc = np.clip(p, 1e-7, 1-1e-7)
            total_loss += -np.mean(y[b]*np.log(pc)+(1-y[b])*np.log(1-pc))
        return total_loss / (len(y) // batch)

    def score(self, X, y):
        p = self.forward(X)
        return ((p > 0.5).astype(int) == y).mean()

# ── Train all three ────────────────────────────────────────────────────────
dims = [X_tr.shape[1], 128, 128, 64, 1]
N_EPOCHS = 60

act_configs = [
    ("ReLU",  relu_,  d_relu_,  "tomato"),
    ("GELU",  gelu_,  d_gelu_,  "steelblue"),
    ("Swish", swish_, d_swish_, "seagreen"),
]

print("=" * 60)
print("  ReLU vs GELU vs Swish: TRAINING COMPARISON")
print("=" * 60)
print(f"  Architecture: {' → '.join(str(d) for d in dims)}")
print(f"  Dataset: {X_tr.shape[0]} train | {X_te.shape[0]} test")
print(f"  Epochs: {N_EPOCHS}")
print()

histories = {}
for name, act, dact, colour in act_configs:
    net = MiniMLP(dims, act, dact, lr=0.008, seed=0)
    tr_losses, tr_accs, te_accs = [], [], []

    for ep in range(N_EPOCHS):
        loss = net.train_epoch(X_tr, y_tr)
        tr_losses.append(loss)
        tr_accs.append(net.score(X_tr, y_tr))
        te_accs.append(net.score(X_te, y_te))

    histories[name] = {"loss": tr_losses, "tr_acc": tr_accs,
                       "te_acc": te_accs, "colour": colour}

    print(f"  {name:6s}  final train_acc={tr_accs[-1]:.4f}  "
          f"test_acc={te_accs[-1]:.4f}  "
          f"loss={tr_losses[-1]:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────
epochs = np.arange(1, N_EPOCHS+1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("ReLU vs GELU vs Swish — Training Dynamics",
             fontsize=12, fontweight="bold")

for name, data in histories.items():
    colour = data["colour"]
    axes[0].plot(epochs, data["loss"],   colour, lw=2, label=name)
    axes[1].plot(epochs, data["tr_acc"], colour, lw=2)
    axes[2].plot(epochs, data["te_acc"], colour, lw=2)

for ax, title, ylabel in zip(axes,
        ["Training Loss", "Training Accuracy", "Test Accuracy"],
        ["BCE Loss", "Accuracy", "Accuracy"]):
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend(fontsize=9) if title=="Training Loss" else None

plt.tight_layout()
plt.savefig("activation_comparison_training.png", dpi=120)

print()
print("  KEY OBSERVATIONS:")
print("  - GELU and Swish often converge faster than ReLU (smoother gradients)")
print("  - The advantage grows with network depth (transformer scale: huge)")
print("  - For shallow networks: difference is small — ReLU is fine")
print("  - Swish ≈ GELU in practice; GELU is the standard for Transformers")
print()
print("  Plot saved → activation_comparison_training.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Loss Function Landscape & Gradient Analysis": {
        "description": (
            "Visualise the shape of every major loss function and its gradient. "
            "Show concretely how MSE, MAE, Huber, BCE, and Hinge differ "
            "in their sensitivity to outliers."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Loss functions and their gradients ────────────────────────────────────
def mse(err):     return err**2
def d_mse(err):   return 2*err

def mae(err):     return np.abs(err)
def d_mae(err):   return np.sign(err)

def huber(err, delta=1.0):
    return np.where(np.abs(err) <= delta,
                    0.5*err**2,
                    delta*(np.abs(err) - 0.5*delta))
def d_huber(err, delta=1.0):
    return np.where(np.abs(err) <= delta, err, delta*np.sign(err))

def bce_positive(p):
    """BCE loss for y=1 as a function of predicted probability p."""
    p = np.clip(p, 1e-7, 1-1e-7)
    return -np.log(p)
def d_bce_positive(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -1/p

def hinge(margin):  return np.maximum(0, 1 - margin)
def d_hinge(margin):return np.where(margin < 1, -1.0, 0.0)

# ── Plot ──────────────────────────────────────────────────────────────────
err = np.linspace(-4, 4, 500)
p   = np.linspace(0.01, 0.99, 500)   # for BCE
m   = np.linspace(-2, 3, 500)        # for hinge (y*f(x))

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Loss Functions: Shape, Gradient & Outlier Sensitivity",
             fontsize=13, fontweight="bold")

reg_losses = [
    ("MSE",        mse,         d_mse,         "tomato",     err),
    ("MAE",        mae,         d_mae,         "steelblue",  err),
    ("Huber δ=1",  huber,       d_huber,       "seagreen",   err),
    ("Huber δ=0.5",lambda e: huber(e, 0.5),
                   lambda e: d_huber(e, 0.5),  "purple",     err),
]

for col, (name, fn, gfn, colour, x) in enumerate(reg_losses):
    loss_vals = fn(x)
    grad_vals = gfn(x)

    # Row 0: loss shape
    axes[0, col].plot(x, loss_vals, colour, lw=2.5)
    axes[0, col].set_title(name, fontsize=10, fontweight="bold")
    axes[0, col].set_xlabel("Prediction error (ŷ - y)")
    axes[0, col].set_ylabel("Loss" if col == 0 else "")
    axes[0, col].set_ylim(0, 8); axes[0, col].grid(alpha=0.3)

    # Row 1: gradient
    axes[1, col].plot(x, grad_vals, colour, lw=2.5, linestyle="--")
    axes[1, col].axhline(0, color="gray", lw=0.7)
    axes[1, col].set_xlabel("Prediction error")
    axes[1, col].set_ylabel("Gradient" if col == 0 else "")
    axes[1, col].set_ylim(-5, 5); axes[1, col].grid(alpha=0.3)

    # Highlight outlier region
    axes[0, col].axvspan(2, 4, alpha=0.08, color="red", label="Outlier zone")
    axes[0, col].axvspan(-4, -2, alpha=0.08, color="red")

# Row 2: BCE and Hinge
axes[2, 0].plot(p, bce_positive(p), "tomato", lw=2.5, label="BCE y=1")
axes[2, 0].plot(p, -np.log(1-p), "steelblue", lw=2.5, linestyle="--",
                label="BCE y=0")
axes[2, 0].set_xlabel("Predicted probability p"); axes[2, 0].set_ylabel("Loss")
axes[2, 0].set_title("BCE Loss", fontsize=10, fontweight="bold")
axes[2, 0].set_ylim(0, 6); axes[2, 0].legend(fontsize=9); axes[2, 0].grid(alpha=0.3)

axes[2, 1].plot(p, d_bce_positive(p), "tomato", lw=2.5)
axes[2, 1].set_xlabel("Predicted probability p"); axes[2, 1].set_ylabel("Gradient")
axes[2, 1].set_title("BCE Gradient", fontsize=10, fontweight="bold")
axes[2, 1].set_ylim(-20, 0); axes[2, 1].grid(alpha=0.3)

axes[2, 2].plot(m, hinge(m), "seagreen", lw=2.5, label="Hinge")
axes[2, 2].plot(m, bce_positive(np.clip((m+1)/2, 0.01, 0.99)),
                "tomato", lw=2.5, linestyle="--", label="BCE (scaled)")
axes[2, 2].axvline(1, color="gray", lw=1, linestyle=":")
axes[2, 2].set_xlabel("Margin  y·f(x)"); axes[2, 2].set_ylabel("Loss")
axes[2, 2].set_title("Hinge vs BCE Loss", fontsize=10, fontweight="bold")
axes[2, 2].legend(fontsize=9); axes[2, 2].grid(alpha=0.3)
axes[2, 2].annotate("Margin=1: zero loss here", xy=(1, 0),
                      xytext=(1.3, 0.8), fontsize=8,
                      arrowprops=dict(arrowstyle="->"))

axes[2, 3].plot(m, d_hinge(m), "seagreen", lw=2.5)
axes[2, 3].axhline(0, color="gray", lw=0.7)
axes[2, 3].axvline(1, color="gray", lw=1, linestyle=":")
axes[2, 3].set_xlabel("Margin  y·f(x)"); axes[2, 3].set_ylabel("Gradient")
axes[2, 3].set_title("Hinge Gradient", fontsize=10, fontweight="bold")
axes[2, 3].set_ylim(-1.5, 0.5); axes[2, 3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("loss_function_landscapes.png", dpi=110)

print("=" * 65)
print("  LOSS FUNCTION PROPERTIES: NUMERICAL COMPARISON")
print("=" * 65)
print()
errors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
print(f"  {'Error':>8} | {'MSE':>10} | {'MAE':>10} | "
      f"{'Huber':>10} | {'MSE grad':>10} | {'Huber grad':>11}")
print(f"  {'─'*70}")
for e in errors:
    print(f"  {e:8.1f} | {mse(e):10.3f} | {mae(e):10.3f} | "
          f"{huber(e):10.3f} | {d_mse(e):10.3f} | {d_huber(e):11.3f}")

print()
print("  KEY INSIGHT (outlier at error=10):")
print(f"    MSE loss   = {mse(10):.1f}  (100× larger than error=1 which gives 1)")
print(f"    MAE loss   = {mae(10):.1f}  (10× larger — linear growth)")
print(f"    Huber loss = {huber(10):.1f}  (9.5 — linear cap after δ=1)")
print()
print("  MSE gradient at error=10 is {:.0f}× larger than at error=1.".format(
      d_mse(10)/d_mse(1)))
print("  Huber gradient caps at δ=1.0 regardless of outlier magnitude.")
print()
print("  Plot saved → loss_function_landscapes.png")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Focal Loss vs BCE on Severe Class Imbalance": {
        "description": (
            "Implement Focal Loss from scratch. Train BCE vs Focal Loss on "
            "a 1:100 imbalanced dataset. Compare precision, recall, F1, and "
            "show how focal loss reshapes the gradient to focus on hard examples."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

def _tp_fp_fn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum(); fn=((yp==0)&(yt==1)).sum()
    return tp,fp,fn

def f1_score(yt,yp,zero_division=0):
    tp,fp,fn=_tp_fp_fn(yt,yp); d=2*tp+fp+fn
    return float(2*tp/d) if d>0 else float(zero_division)

def precision_score(yt,yp,zero_division=0):
    tp,fp,fn=_tp_fp_fn(yt,yp)
    return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)

def recall_score(yt,yp,zero_division=0):
    tp,fp,fn=_tp_fp_fn(yt,yp)
    return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)

def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))

def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return float(np.sum(prec*ys2)/pos)

np.random.seed(42)

# ── Severely imbalanced dataset: 1% positives ─────────────────────────────
X, y = make_classification(
    n_samples=5000, n_features=15, n_informative=8,
    weights=[0.99, 0.01], flip_y=0.005, random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1)

print("=" * 65)
print("  FOCAL LOSS vs BCE: SEVERE CLASS IMBALANCE (99:1)")
print("=" * 65)
print(f"  Positives in train: {y_tr.sum()} / {len(y_tr)} = {y_tr.mean()*100:.1f}%")
print(f"  Positives in test : {y_te.sum()} / {len(y_te)} = {y_te.mean()*100:.1f}%")
print()

def sigmoid_(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
def relu_(x):    return np.maximum(0, x)

# ── Loss functions ────────────────────────────────────────────────────────
def bce_loss_grad(y, logit):
    """BCE loss value and gradient w.r.t. logit."""
    p    = sigmoid_(logit)
    p_c  = np.clip(p, 1e-7, 1-1e-7)
    loss = -np.mean(y*np.log(p_c) + (1-y)*np.log(1-p_c))
    grad = (p - y) / len(y)    # gradient w.r.t. logit (clean form)
    return loss, grad

def focal_loss_grad(y, logit, gamma=2.0):
    """Focal loss value and gradient w.r.t. logit."""
    p     = sigmoid_(logit)
    p_c   = np.clip(p, 1e-7, 1-1e-7)
    pt    = np.where(y == 1, p_c, 1 - p_c)       # p of true class
    wt    = (1 - pt) ** gamma                      # focusing weight
    loss  = -np.mean(wt * np.log(pt))

    # Gradient: ∂FL/∂logit = (pt-1)·((γ·pt·log(pt)) + pt - 1) for y=1
    # Full gradient derivation:
    d_bce = p - y                                  # base BCE grad (per-sample)
    fl_mod= wt * (1 - gamma * pt * np.log(np.clip(pt, 1e-7, 1)))
    grad  = (fl_mod * d_bce) / len(y)
    return loss, grad

# ── Simple MLP ────────────────────────────────────────────────────────────
class SmallNet:
    def __init__(self, n_in, lr=0.01):
        np.random.seed(0)
        self.W1 = np.random.randn(n_in, 64) * np.sqrt(2/n_in)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 1)   * np.sqrt(2/64)
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X):
        self.a1    = relu_(X @ self.W1 + self.b1)
        self.logit = (self.a1 @ self.W2 + self.b2).ravel()
        return self.logit

    def backward(self, X, grad_logit):
        dW2 = self.a1.T @ grad_logit.reshape(-1, 1)
        db2 = grad_logit.sum()
        da1 = grad_logit.reshape(-1,1) @ self.W2.T
        dz1 = da1 * (self.a1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        for W, dW in [(self.W1, dW1), (self.W2, dW2)]:
            W -= self.lr * dW
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

# ── Train both models ─────────────────────────────────────────────────────
N_EPOCHS  = 60
BATCH     = 128

configs = [
    ("BCE        (γ=0)", bce_loss_grad,                "tomato"),
    ("Focal γ=1",       lambda y, l: focal_loss_grad(y, l, gamma=1), "orange"),
    ("Focal γ=2",       lambda y, l: focal_loss_grad(y, l, gamma=2), "steelblue"),
    ("Focal γ=4",       lambda y, l: focal_loss_grad(y, l, gamma=4), "seagreen"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Focal Loss vs BCE: Severe Class Imbalance (99:1 ratio)",
             fontsize=12, fontweight="bold")

print(f"  {'Loss':20s} | {'Precision':>10} | {'Recall':>8} | "
      f"{'F1':>8} | {'PR-AUC':>8} | {'ROC-AUC':>8}")
print(f"  {'─'*72}")

epoch_arr = np.arange(1, N_EPOCHS+1)

for loss_name, loss_fn, colour in configs:
    net = SmallNet(X_tr.shape[1], lr=0.005)
    tr_losses, f1s = [], []

    for ep in range(N_EPOCHS):
        idx = np.random.permutation(len(y_tr))
        ep_loss = 0
        for start in range(0, len(y_tr), BATCH):
            b = idx[start:start+BATCH]
            logit = net.forward(X_tr[b])
            loss, grad = loss_fn(y_tr[b], logit)
            net.backward(X_tr[b], grad)
            ep_loss += loss
        tr_losses.append(ep_loss / (len(y_tr) // BATCH))

        # Evaluate on test
        logit_te = net.forward(X_te)
        p_te     = sigmoid_(logit_te)
        y_pred   = (p_te > 0.5).astype(int)
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    # Final metrics
    logit_te = net.forward(X_te)
    p_te     = sigmoid_(logit_te)
    y_pred   = (p_te > 0.5).astype(int)

    prec   = precision_score(y_te, y_pred, zero_division=0)
    rec    = recall_score(y_te, y_pred, zero_division=0)
    f1     = f1_score(y_te, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_te, p_te)
    roc    = roc_auc_score(y_te, p_te)

    print(f"  {loss_name:20s} | {prec:10.4f} | {rec:8.4f} | "
          f"{f1:8.4f} | {pr_auc:8.4f} | {roc:8.4f}")

    axes[0].plot(epoch_arr, tr_losses, colour, lw=2, label=loss_name)
    axes[1].plot(epoch_arr, f1s,       colour, lw=2)
    axes[2].bar(loss_name, rec, color=colour, alpha=0.8)

axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].set_title("Test F1 Score"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1"); axes[1].grid(alpha=0.3)
axes[2].set_title("Final Minority Recall"); axes[2].set_ylabel("Recall")
axes[2].set_xticklabels(["BCE", "FL γ=1", "FL γ=2", "FL γ=4"], rotation=15)
axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("focal_loss_comparison.png", dpi=120)

print()
print("  WHY FOCAL LOSS WINS ON IMBALANCED DATA:")
print("  BCE: ~99% examples are easy negatives → gradient dominated by them")
print("  Focal: down-weights easy examples by (1-p_t)^γ")
print("    → γ=2: easy example (p_t=0.9) contributes 100× less gradient")
print("    → model spends its capacity on the hard minority positives")
print()
print("  Plot saved → focal_loss_comparison.png")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · KL Divergence & Cross-Entropy: Information Theory Foundations": {
        "description": (
            "Implement KL Divergence and cross-entropy from scratch. "
            "Show forward vs reverse KL on a bimodal distribution. "
            "Demonstrate how cross-entropy = KL divergence + entropy, and "
            "why minimising cross-entropy is equivalent to minimising KL."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)

# ── Core information-theoretic quantities ─────────────────────────────────
def entropy(p):
    """H(P) = -Σ p log(p)"""
    p = np.clip(p, 1e-10, 1)
    p = p / p.sum()              # ensure normalised
    return -np.sum(p * np.log(p))

def cross_entropy(p, q):
    """H(P, Q) = -Σ p log(q)"""
    p = np.clip(p / p.sum(), 1e-10, 1)
    q = np.clip(q / q.sum(), 1e-10, 1)
    return -np.sum(p * np.log(q))

def kl_divergence(p, q):
    """KL(P || Q) = Σ p log(p/q) = H(P,Q) - H(P)"""
    p = np.clip(p / p.sum(), 1e-10, 1)
    q = np.clip(q / q.sum(), 1e-10, 1)
    return np.sum(p * np.log(p / q))

# ── Verify: H(P,Q) = H(P) + KL(P||Q) ─────────────────────────────────────
print("=" * 65)
print("  KL DIVERGENCE & CROSS-ENTROPY: INFORMATION THEORY")
print("=" * 65)
print()
print("  Fundamental identity:  H(P,Q) = H(P) + KL(P || Q)")
print("  Minimising H(P,Q) w.r.t. Q = minimising KL(P||Q)")
print("  (because H(P) is constant w.r.t. Q)")
print()

# Example distributions over 4 categories
P = np.array([0.4, 0.3, 0.2, 0.1])  # true distribution
Q = np.array([0.25, 0.25, 0.25, 0.25])  # uniform (maximum uncertainty)
R = np.array([0.6, 0.2, 0.1, 0.1])     # similar to P

for name, Q_ in [("Uniform Q", Q), ("Similar R", R)]:
    H_P     = entropy(P)
    H_PQ    = cross_entropy(P, Q_)
    KL_PQ   = kl_divergence(P, Q_)
    verify  = abs(H_PQ - (H_P + KL_PQ))
    print(f"  P = {P}   {name} = {Q_}")
    print(f"    H(P)         = {H_P:.5f}  (entropy of P)")
    print(f"    H(P,Q)       = {H_PQ:.5f}  (cross-entropy)")
    print(f"    KL(P||Q)     = {KL_PQ:.5f}  (extra bits from using Q)")
    print(f"    H(P,Q) - H(P) = {H_PQ - H_P:.5f}  vs KL={KL_PQ:.5f}  "
          f"(match: {'✓' if verify < 1e-10 else '✗'})")
    print()

# ── Forward vs Reverse KL on bimodal distribution ─────────────────────────
x = np.linspace(-6, 6, 1000)
dx = x[1] - x[0]

# True P: bimodal (mixture of two Gaussians)
P_bimodal = (0.5 * norm.pdf(x, -2, 0.8) +
             0.5 * norm.pdf(x,  2, 0.8))
P_bimodal /= P_bimodal.sum() * dx   # normalise to density

# Forward KL: Q ~ N(μ, σ) minimising KL(P || Q) → mean-seeking
# Analytically: μ = E_P[x], σ² = E_P[(x-μ)²]
mu_fwd = np.sum(x * P_bimodal) * dx
var_fwd = np.sum((x - mu_fwd)**2 * P_bimodal) * dx
Q_forward = norm.pdf(x, mu_fwd, np.sqrt(var_fwd))

# Reverse KL: Q minimising KL(Q || P) → mode-seeking
# Q snaps to one mode; we approximate with the first mode
Q_reverse = norm.pdf(x, -2, 0.8)

# Compute KL values
eps = 1e-10
def kl_continuous(p, q):
    p_c = np.maximum(p, eps)
    q_c = np.maximum(q, eps)
    return np.sum(p_c * np.log(p_c / q_c)) * dx

kl_fwd = kl_continuous(P_bimodal, Q_forward)
kl_rev = kl_continuous(Q_reverse, P_bimodal)
ce_fwd = -np.sum(P_bimodal * np.log(np.maximum(Q_forward, eps))) * dx

print(f"  FORWARD vs REVERSE KL on BIMODAL P:")
print(f"    KL(P || Q_forward) = {kl_fwd:.4f}  (Q spreads to cover BOTH modes)")
print(f"    KL(Q || P_reverse) = {kl_rev:.4f}  (Q snaps to ONE mode)")
print(f"    CE(P, Q_forward)   = {ce_fwd:.4f}  (= H(P) + KL(P||Q_forward))")
print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("KL Divergence: Forward (Mean-Seeking) vs Reverse (Mode-Seeking)",
             fontsize=12, fontweight="bold")

axes[0].fill_between(x, P_bimodal, alpha=0.3, color="gray",  label="P (bimodal)")
axes[0].plot(x, P_bimodal,  "k",    lw=2.5, label="P (bimodal)")
axes[0].plot(x, Q_forward,  "tomato", lw=2.5, linestyle="--",
             label=f"Q forward (mean-seeking) KL={kl_fwd:.3f}")
axes[0].set_title("Forward KL(P || Q) — Mean-Seeking")
axes[0].set_xlabel("x"); axes[0].set_ylabel("Density")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].fill_between(x, P_bimodal, alpha=0.3, color="gray",  label="P (bimodal)")
axes[1].plot(x, P_bimodal,  "k",       lw=2.5, label="P (bimodal)")
axes[1].plot(x, Q_reverse,  "steelblue", lw=2.5, linestyle="--",
             label=f"Q reverse (mode-seeking) KL={kl_rev:.3f}")
axes[1].set_title("Reverse KL(Q || P) — Mode-Seeking")
axes[1].set_xlabel("x"); axes[1].set_ylabel("Density")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# Cross-entropy vs KL decomposition bar chart
vals = [entropy(P_bimodal * dx),
        kl_fwd,
        ce_fwd]
labels = ["H(P) entropy", "KL(P||Q) extra bits", "H(P,Q) cross-entropy"]
colours = ["steelblue", "tomato", "seagreen"]
bars = axes[2].bar(labels, vals, color=colours, alpha=0.8, edgecolor="black")
axes[2].set_title("H(P,Q) = H(P) + KL(P||Q) [verified numerically]")
axes[2].set_ylabel("Nats (natural log)")
for bar, val in zip(bars, vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9)
axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("kl_divergence_demo.png", dpi=120)

print("  PRACTICAL CONSEQUENCES:")
print("  - Training CE loss = minimising KL from model Q to data P")
print("  - Forward KL is used in supervised learning (mode-covering)")
print("  - Reverse KL is used in variational inference (mode-seeking)")
print("  - In VAEs: ELBO = reconstruction - KL(q(z|x) || p(z))")
print("  - In RLHF: KL penalty prevents model drifting too far from base")
print()
print("  Plot saved → kl_divergence_demo.png")
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
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }