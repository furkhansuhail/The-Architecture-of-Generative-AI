"""
Gradient Flow Issues
====================

Gradients are the sole mechanism by which a neural network learns.
Every weight update, every representation refined, every decision
boundary shifted — all of it flows backward through the chain rule
as a gradient signal. When that signal pathologically shrinks to
near-zero (vanishing) or explodes toward infinity (exploding), learning
either stalls completely or destabilises catastrophically. Understanding
why these failures occur, how to detect them, and how to prevent them
is not optional knowledge — it is prerequisite to training any deep
network reliably.

"""
import textwrap
import re

TOPIC_NAME   = "Gradient Flow Issues"
DISPLAY_NAME = "08 · Gradient Flow Issues"
ICON         = "⚡"
SUBTITLE     = "Vanishing · Exploding · Dead Neurons · Clipping · Diagnosis"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE CHAIN RULE AND GRADIENT FLOW

### How Gradients Travel Through a Network

In a deep network, the gradient of the loss with respect to a weight
in an early layer is computed by the CHAIN RULE — multiplying together
the local gradients of every layer between that weight and the loss:

    Consider a network with L layers:
    z₁ → z₂ → z₃ → ... → z_L → Loss

    Gradient for layer 1:
    ∂Loss/∂w₁ = ∂Loss/∂z_L · ∂z_L/∂z_{L-1} · ... · ∂z₂/∂z₁ · ∂z₁/∂w₁

    This is a PRODUCT of L terms. Each term is a Jacobian matrix
    (or scalar, in the simplified single-neuron case).

    The catastrophic consequence:
    If each term has magnitude slightly less than 1 → the product
    SHRINKS EXPONENTIALLY with depth.
    If each term has magnitude slightly greater than 1 → the product
    GROWS EXPONENTIALLY with depth.

    Diagram 1 — Gradient Magnitude Through Layers (log scale):

    log|gradient|
          │
     0.0  │────────────────────────────── stable flow (magnitude ≈ 1)
          │
    -5.0  │╲
          │ ╲──────────────────────────── vanishing (magnitude → 0)
   -10.0  │  ╲──────────────────────
          │
    +5.0  │                    ╱───────── exploding (magnitude → ∞)
          │                 ╱──
    +10.0 │              ╱──
          │
          └─────────────────────────────────────────── Layer (1 → L)
          Early layers                          Deep layers

    Gradient magnitude should remain roughly constant across layers
    for healthy training. Significant monotone trends signal a problem.


### Why Early Layers Are Hardest Hit

Both vanishing and exploding gradients affect EARLY LAYERS most severely:
the gradient for layer 1 must travel through ALL L layers of
multiplication, while the gradient for layer L only passes through 1.

    Consequence of vanishing for early layers:
    The first few layers receive nearly zero gradient. Their weights
    barely update. The model effectively reduces to only its later layers
    learning — the representations at the bottom of the network are
    frozen at their random initialisation.

    Consequence of exploding for early layers:
    Massive gradient → massive weight update → weights jump to a
    completely different region of the loss landscape → loss spikes.
    The model "forgets" what it learned and must start over from
    a random-like state.

    Depth amplifies both effects:
    10-layer network:  (0.9)^10 ≈ 0.35   (35% remaining signal — tolerable)
    50-layer network:  (0.9)^50 ≈ 0.005  (0.5% remaining signal — effectively zero)
    100-layer network: (0.9)^100 ≈ 0.00003  (functionally zero)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — VANISHING GRADIENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Sigmoid Saturation Problem

The first deep networks used sigmoid activations:

    σ(x) = 1 / (1 + e^{-x})

    Derivative: σ'(x) = σ(x) · (1 - σ(x))

    Maximum derivative:  σ'(0) = 0.25
    At saturation:       σ'(x) → 0  as x → ±∞

    Diagram 2 — Sigmoid and Its Derivative:

    σ(x)   │            ──────────
           │         ───
           │       ──
      0.5  │─────●                       ← maximum is 0.5
           │   ──
           │──
           └──────────────────── x
          -5   -2   0   2   5

    σ'(x)  │
      0.25 │         ●                   ← maximum is only 0.25!
           │       ╱   ╲
           │     ╱       ╲
      0.0  │────╱─────────╲────          ← gradient → 0 at saturation
           └──────────────────── x
          -5   -2   0   2   5

    Two problems compound each other:
    1. The maximum gradient through one sigmoid is 0.25.
       Through 10 sigmoids: (0.25)^10 ≈ 1 × 10^{-6}. Vanished.
    2. When inputs are large (positive or negative), the neuron SATURATES:
       the output is near 0 or 1, the gradient is near 0, and the neuron
       stops contributing to learning entirely.

    Tanh has the same saturation problem:
    tanh'(x) = 1 - tanh(x)²
    Maximum: tanh'(0) = 1.0  (better than sigmoid)
    At saturation: tanh'(x) → 0 as x → ±∞  (same problem)

    Tanh is ZERO-CENTRED (output in [-1, 1]) unlike sigmoid (output in [0, 1]).
    Zero-centred activations produce zero-mean inputs to subsequent layers,
    which improves conditioning. But saturation vanishing remains.


### Vanishing in RNNs: The Temporal Version

For Recurrent Neural Networks (RNNs), the chain rule unfolds THROUGH TIME:

    h_t = tanh(W_h · h_{t-1} + W_x · x_t)

    Gradient of loss at time T with respect to hidden state at time t:
    ∂L/∂h_t = ∂L/∂h_T · Π_{k=t}^{T-1} ∂h_{k+1}/∂h_k
             = ∂L/∂h_T · Π_{k=t}^{T-1} (W_h^T · diag(tanh'(·)))

    This is a product of T-t matrices. Two things determine whether it
    vanishes or explodes:
    1. The eigenvalues of W_h: if |λ_max| < 1, the product vanishes.
    2. The saturation of tanh: each tanh'(·) ≤ 1, compounding the shrinkage.

    For sequences of length T=100 with even slight gradient shrinkage,
    early-time information is completely erased from the gradient.

    The symptom: the RNN fails to learn LONG-RANGE DEPENDENCIES.
    It can learn patterns over the last 5-10 steps but cannot attribute
    the current loss to events 50 steps ago.

    This is the fundamental motivation for LSTMs and GRUs (gating mechanisms
    that provide a shortcut path for gradient flow), and ultimately for
    the attention mechanism and transformers (which attend to any position
    with a direct gradient path, no multiplication chain through time).


### Diagnosis: How to Detect Vanishing Gradients

    Signal 1 — Gradient norms shrink with depth:
    Log the L2 norm of gradients for each layer's parameters.
    Vanishing: norms near-zero in early layers, nonzero in later layers.

    Signal 2 — Loss does not decrease for early layers:
    Parameters in early layers barely update. Their loss contribution
    is essentially zero. The loss curve may look like it's training
    but only later layers are contributing.

    Signal 3 — Activations saturate:
    Mean activation in each layer approaches 0 or 1 (sigmoid) or
    ±1 (tanh). You can track this with hooks.

    Signal 4 — Hidden representations do not vary:
    The standard deviation of activations in early layers is near zero.
    Every input produces nearly the same hidden representation.

    Code to inspect gradient norms:
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name:40s}  grad_norm={grad_norm:.2e}")

    Healthy output:   grad_norm=1.2e-02 to 1.0e-01  (consistent across layers)
    Vanishing signal: grad_norm=1e-08 in early layers, 1e-02 in later layers


##### PART 3 — EXPLODING GRADIENTS

### Causes of Gradient Explosion

Unlike vanishing (which requires specific activation functions), exploding
gradients can occur with any activation if:

    1. The weight matrices have eigenvalues > 1:
       Each layer multiplies the gradient by W^T. If the largest
       eigenvalue of W is 1.5, after 20 layers: 1.5^20 ≈ 3,325.
       After 50 layers: 1.5^50 ≈ 637,621,500.

    2. Residual networks without proper initialisation:
       In very deep residual networks, accumulated residual contributions
       can cause the effective gradient to grow with depth.

    3. Learning rate too large:
       Not a gradient explosion per se, but large LR amplifies gradient
       updates to the same catastrophic effect: a single step takes the
       weights far outside the valid region of the loss landscape.

    4. RNNs on long sequences:
       If |λ_max(W_h)| > 1, gradients explode through time exactly
       symmetrically to the vanishing case.

    5. Loss scaling bugs:
       Summing (rather than averaging) the loss over a large batch
       produces loss and gradient values that scale with batch size.
       A batch of 512 with sum reduction produces 512× larger gradients
       than a batch of 1. If the learning rate was tuned for batch=1,
       the update is catastrophically large.


### The Loss Cliff: Detecting Explosions in Practice

    Diagram 3 — Loss Cliff (Catastrophic Gradient Explosion):

    Loss
      │
      │              ╭───────────────────
      │             ╱│
      │            ╱ │
      │           ╱  │  ← loss cliff
      │          ╱   │
      │─────────╯    │───────────── (resumes from worse position)
      │
      └──────────────────────────────── Training step

    The loss cliff is the signature of gradient explosion:
    1. Training is progressing normally.
    2. In a single step (or a few steps), the loss spikes dramatically.
    3. If the explosion is moderate, the loss may partially recover.
    4. If the explosion is severe, training does not recover — the weights
       have moved so far that the model is effectively re-initialised randomly.

    Loss cliffs appear regularly in LLM pre-training on large datasets.
    They are a known failure mode. The standard solution is gradient clipping
    (Part 5) plus gradient accumulation to smooth out spike-inducing batches.


### Gradient Explosion in Practice: NaN and Inf

    When gradients become large enough:
    1. Gradient values become Inf (floating-point overflow)
    2. Weight updates become Inf → weights become Inf
    3. Forward pass through an Inf weight produces NaN (0 × Inf)
    4. Loss becomes NaN
    5. All subsequent operations produce NaN
    6. Training has irrecoverably failed

    NaN propagation is SILENT in PyTorch by default:
    torch.tensor(float('nan')) + 5.0 = nan  (no error raised)

    Standard debugging check:
    if not torch.isfinite(loss):
        print(f"Non-finite loss at step {step}: {loss.item()}")
        # Log the batch, examine inputs, check for corrupt data

    Or as a training guard:
    assert torch.isfinite(loss), f"Loss is {loss.item()}"


### Diagnosis: How to Detect Exploding Gradients

    Signal 1 — Gradient norms spike:
    The gradient norm (computed before clipping) should be logged
    every step. Exploding gradients produce norm values 10–1000×
    the typical baseline. A step with norm=1000 when typical is 0.5
    is an explosion event.

    Signal 2 — Loss spike:
    A sudden large increase in training loss (the loss cliff).

    Signal 3 — NaN / Inf in loss or weights:
    Loss becomes nan or inf. torch.isnan(loss) is True.

    Signal 4 — Weight values become very large:
    Inspect weight magnitudes after a suspected explosion:
    for name, param in model.named_parameters():
        print(f"{name}: max={param.data.abs().max():.2e}")
    Values > 1e3 for normalised networks indicate explosion.

    Code to track gradient norms during training:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar('train/grad_norm', total_norm, step)


##### PART 4 — DEAD NEURONS (THE DYING RELU PROBLEM)

### What Is a Dead Neuron?

ReLU activation: f(x) = max(0, x)

    Derivative: f'(x) = 1  if x > 0
                         0  if x ≤ 0

    When a neuron's pre-activation (x = w·input + b) is negative,
    the output is zero AND the gradient through that neuron is ZERO.

    A neuron that receives a large negative weight update may enter
    a state where its pre-activation is negative for EVERY example
    in the training set. In that state:
    - Output is always 0 (never fires)
    - Gradient is always 0 (never receives update)
    - The neuron is permanently "dead" — no gradient can revive it

    This is the DYING RELU PROBLEM (Maas et al., 2013).

    Diagram 4 — A Dead ReLU Neuron:

    Pre-activation x = w·input + b

    Healthy neuron:     Dead neuron:
    x │ ╱               x │      ──────
      │╱                  │
    ──┼───── input      ──┼───────────── input
      │                   │╲
      │                   │ ╲──────       (always negative → always 0 output)

    The neuron never fires. Its gradient is perpetually 0.
    Its weights never update. It contributes nothing to the network.


### Why ReLU Neurons Die

    1. Large learning rate:
       A single large gradient step pushes the bias term to a large
       negative value. All inputs produce negative pre-activations.
       The neuron dies instantly and cannot recover.

    2. Negative bias initialisation:
       Biases initialised to negative values immediately push some
       neurons into the dead zone. With zero-mean weight initialisation,
       approximately half the neurons may die in the first forward pass.

    3. Heavy L2 regularisation:
       Constant weight decay drives weights toward zero. If the bias is
       slightly negative, decaying weights reduce the pre-activation
       further below zero for all inputs.

    4. Large learning rate + large batch loss:
       The product of a large LR and a large gradient (from loss summed
       over a large batch) creates the same large-step problem as (1).


### Scale of the Problem

In a poorly initialised or poorly tuned network, a MAJORITY of neurons
can die within the first few hundred training steps. If 60% of neurons
in a layer are dead, that layer has 60% less effective capacity.
The network behaves as if it were much smaller than designed.

    Tracking dead neurons:
    # Fraction of ReLU neurons that are dead (output=0) on a val batch
    def dead_neuron_fraction(model, val_batch):
        hooks, fractions = [], []
        def hook(module, input, output):
            dead = (output == 0).float().mean()
            fractions.append(dead.item())
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                hooks.append(m.register_forward_hook(hook))
        with torch.no_grad():
            model(val_batch)
        for h in hooks:
            h.remove()
        return fractions   # per-layer fraction of dead neurons

    Healthy: < 5% dead per layer
    Concerning: 20–40% dead per layer
    Critical: > 50% dead per layer


### Solutions to Dying ReLU

    LEAKY RELU:
    f(x) = x       if x > 0
           α·x     if x ≤ 0    (α is a small slope, typically 0.01)

    Derivative: f'(x) = 1   if x > 0
                         α   if x ≤ 0

    The gradient is NEVER zero — inactive neurons still receive a
    small gradient (α) and can recover from the dead zone.
    The slope α is a hyperparameter. α=0.01 is the standard default.

    PARAMETRIC RELU (PReLU):
    Like Leaky ReLU but α is a learnable parameter.
    The network learns the optimal negative slope per channel.
    Used in ResNets and face recognition models.
    Risk: α can be learned to 0 if not regularised — degenerates to ReLU.

    ELU (Exponential Linear Unit):
    f(x) = x              if x > 0
           α·(e^x - 1)   if x ≤ 0    (α typically = 1.0)

    Properties:
    - Smooth at x=0 (unlike ReLU and Leaky ReLU which have a kink)
    - Negative outputs push the mean activation toward zero
      (zero-centering effect similar to tanh but without saturation)
    - Negative saturation at -α (bounded below, unlike Leaky ReLU)

    GELU (Gaussian Error Linear Unit):
    f(x) = x · Φ(x)    where Φ is the standard normal CDF

    Approximation: f(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

    Properties:
    - Smooth everywhere (differentiable at all points)
    - Stochastic interpretation: gate x by the probability that it is
      positive (treating it as drawn from N(0,1))
    - Near-zero gradient for very negative x (some dying possible but
      much less than ReLU)
    - Standard activation in transformers: BERT, GPT-2/3/4, ViT

    SWISH (SiLU):
    f(x) = x · σ(x) = x / (1 + e^{-x})

    Properties:
    - Similar to GELU in shape and performance
    - Smooth, non-monotonic (has a slight negative valley near x=-1)
    - Slightly faster to compute than GELU
    - Used in: EfficientNet, MobileNetV3, some LLM variants

    Diagram 5 — Activation Functions and Their Gradients:

    f(x)   │       GELU/Swish              ReLU
           │    ─────────────────────    ──────────╱
           │ ───                        ─────────╱
      0    │──────────────────────   ────────────
           │                      ↑
           │              slight negative dip (non-monotone)
           └──────────────────────────────── x
           -3   -1   0   1   3

    ReLU gradient at x<0 = 0  (neuron can die)
    GELU gradient at x<0 ≈ small but nonzero  (neuron rarely dies)

    Practical guidance:
    ReLU:    MLPs, CNNs when speed is critical, architectures before ~2018
    GELU:    All transformer models (required for compatibility with pretrained)
    Swish:   EfficientNet family, mobile/edge networks
    Leaky ReLU: when a simple ReLU replacement is needed with no tuning


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — GRADIENT CLIPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Clip Gradients?

Gradient clipping is a direct, surgical intervention to prevent
exploding gradients from corrupting a training run. It operates
AFTER the backward pass and BEFORE the optimiser step:

    standard training step:
    loss.backward()          ← compute gradients
    optimiser.step()         ← update weights with (possibly huge) gradients

    with gradient clipping:
    loss.backward()          ← compute gradients
    clip_gradients()         ← rescale if norm exceeds threshold
    optimiser.step()         ← update weights with safe gradients

    Clipping does not remove the gradient direction — it only limits the
    MAGNITUDE of the update. The weights still move in the correct direction,
    just not as far.


### Clip-by-Norm vs Clip-by-Value

    CLIP-BY-NORM (global norm clipping):
    1. Compute the global gradient norm across all parameters:
       g_norm = sqrt(Σ_p ||∇_p L||²)

    2. If g_norm > max_norm:
       Scale ALL gradients uniformly:
       ∇_p L ← ∇_p L · (max_norm / g_norm)

    Properties:
    - Preserves the DIRECTION of the gradient (all parameters scaled equally)
    - The ratio of gradients across parameters is unchanged
    - Only the magnitude is limited
    - The global norm after clipping is exactly max_norm (or less)

    This is the standard method. Used in PyTorch as:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    CLIP-BY-VALUE (element-wise clipping):
    Clip each gradient component independently:
    ∇ᵢ L ← clamp(∇ᵢ L, -clip_value, +clip_value)

    Properties:
    - Simple to implement
    - DOES NOT preserve gradient direction — different components
      are clipped independently, distorting the gradient vector
    - Less principled than norm clipping
    - Rarely used in practice for deep learning
    - One use case: clipping gradients in RL where value function
      gradients need component-wise bounds

    Diagram 6 — Clip-by-Norm vs Clip-by-Value:

    Gradient vector g = (8.0, 6.0), max_norm = 5.0, clip_value = 5.0

    Original:           (8.0, 6.0),  norm = 10.0
    Clip-by-norm:       (4.0, 3.0),  norm = 5.0    ← direction preserved (0.6, 0.8)
    Clip-by-value:      (5.0, 5.0),  norm = 7.07   ← direction changed (0.71, 0.71)

    Clip-by-norm maintains the RELATIVE SIZE of different gradient components.
    Clip-by-value arbitrarily changes the direction of the update.


### Choosing the Clipping Threshold

    The clipping threshold (max_norm) is one of the most important
    training hyperparameters for large models, yet it is often set
    to a default and forgotten. The correct value depends on:

    1. The typical gradient norm during stable training:
       Set max_norm to ~5× the median gradient norm.
       If your typical gradient norm is 0.2, set max_norm ≈ 1.0.
       This clips only the outlier spikes while leaving normal steps untouched.

    2. The model architecture:
       Transformers are particularly sensitive to gradient explosions.
       Standard max_norm values:
       Transformers (NLP):     max_norm = 1.0  (universal default, GPT/BERT)
       LLM pre-training:       max_norm = 1.0
       RNNs / LSTMs:           max_norm = 1.0 to 5.0
       CNNs:                   max_norm = 1.0 to 10.0
       RL training:            max_norm = 0.5 to 1.0

    3. Checking whether clipping is occurring:
       Log the gradient norm BEFORE clipping. If the norm regularly
       exceeds max_norm, clipping is doing substantial work and you may
       need to reduce the learning rate instead.

       If clipping occurs on < 1% of steps: clipping is a safety net (good).
       If clipping occurs on > 20% of steps: clipping is masking an
       underlying instability (reduce LR, improve initialisation).

    Tracking clip events:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # clip_grad_norm_ returns the norm BEFORE clipping
    was_clipped = (grad_norm > max_norm)
    writer.add_scalar('train/grad_norm_pre_clip', grad_norm, step)
    writer.add_scalar('train/clip_event', float(was_clipped), step)


### Gradient Clipping and Adaptive Optimisers

A subtle interaction: adaptive optimisers (Adam, AdaGrad) divide
gradients by a running estimate of their historical variance. This
means the EFFECTIVE step size is bounded regardless of gradient magnitude —
adaptive optimisers are implicitly less sensitive to gradient explosions
than SGD.

    However, clipping is STILL recommended even with Adam because:
    1. The first few steps have unreliable variance estimates (warmup
       period of the variance accumulator). Explosions during warmup
       can corrupt the variance estimates permanently.
    2. Very large gradients spike the variance estimate itself, which
       can cause subsequent steps to be too large or too small.
    3. The loss cliff phenomenon in LLM training occurs even with Adam
       and is reliably prevented only by gradient clipping.

    Rule: always use gradient clipping with transformers, regardless
    of the optimiser. max_norm = 1.0 is the established standard.


##### PART 6 — ARCHITECTURAL SOLUTIONS TO GRADIENT FLOW

### Residual Connections (Skip Connections)

Introduced in ResNet (He et al., 2016). The most impactful architectural
innovation for gradient flow in deep networks:

    Standard layer:        output = F(x)
    Residual layer:        output = x + F(x)

    The gradient of the residual block:
    ∂Loss/∂x = ∂Loss/∂output · (1 + ∂F/∂x)

    The critical term is the +1. Even if ∂F/∂x ≈ 0 (the layer learns
    a near-zero transformation), the gradient ∂Loss/∂x ≈ ∂Loss/∂output.
    The gradient flows DIRECTLY through the skip connection with no
    multiplication by a Jacobian.

    Diagram 7 — Gradient Flow in Residual Network:

    Without residual:
    Loss ← ∂/∂z_L · ∂z_L/∂z_{L-1} · ... · ∂z_2/∂z_1 ← weights
           ←───────── product chain (can vanish) ──────────

    With residual:
    Loss ← ∂/∂z_L · (1 + ∂F_L/∂z_{L-1}) · (1 + ∂F_{L-1}/∂z_{L-2}) · ...
           ←──── each factor is at least 1 (cannot vanish to 0) ────────
                 (assuming ∂F/∂z remains bounded above 0)

    The product (1+a₁)(1+a₂)...(1+aₙ) cannot shrink to zero as long
    as each aᵢ ≥ -1 (which is guaranteed by the residual structure).

    Consequence: residual networks with 100+ layers train reliably without
    gradient vanishing. ResNets enabled the first deep networks to be
    trained depth-first rather than carefully layer-by-layer.

    In transformers, the residual connection appears everywhere:
    h = h + Attention(LayerNorm(h))
    h = h + FFN(LayerNorm(h))

    The residual stream h flows through the entire network with a
    direct gradient path from loss to input. Every layer adds to h
    rather than transforming it entirely.


### Layer Normalisation

LayerNorm (Ba et al., 2016) normalises activations WITHIN each example
across the feature dimension:

    LayerNorm(x) = γ · (x - μ) / sqrt(σ² + ε)  +  β

    Where μ and σ² are computed over the features of a single example.

    Gradient flow benefit:
    LayerNorm keeps activation scales bounded regardless of depth.
    Even if a layer's weights grow large, the normalisation rescales
    the output back to approximately unit variance. This prevents
    the systematic growth of activation magnitudes that causes
    both vanishing and exploding gradients in un-normalised networks.

    Pre-norm vs Post-norm in transformers:
    Original transformer (Vaswani 2017):  Post-norm
        h = LayerNorm(h + Attention(h))
        Problem: gradient flows through LayerNorm before reaching
        the residual addition — the smoothing benefit is delayed.

    Modern practice:                      Pre-norm
        h = h + Attention(LayerNorm(h))
        Gradient flows directly through h to earlier layers.
        LayerNorm stabilises the input to the attention sub-layer
        without interfering with the residual gradient path.
        Training is significantly more stable with pre-norm.

    Pre-norm is now standard in all major LLMs: GPT-3, GPT-4,
    LLaMA, Mistral, and all transformer variants from ~2020 onward.


### LSTM Gating: Solving the RNN Vanishing Problem

The Long Short-Term Memory (LSTM, Hochreiter & Schmidhuber, 1997)
introduces a CELL STATE with additive (not multiplicative) updates:

    Standard RNN:     h_t = tanh(W · h_{t-1} + ...)
    LSTM cell update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

    Where:
    c_t = cell state (the long-term memory)
    f_t = forget gate  ∈ (0,1): how much of old state to keep
    i_t = input gate   ∈ (0,1): how much new input to add
    g_t = candidate    ∈ (-1,1): the new candidate values

    Key insight for gradient flow:
    ∂c_t/∂c_{t-1} = f_t  (the forget gate)

    If the forget gate is near 1, the gradient flows back through time
    ALMOST UNCHANGED — no vanishing. The forget gate learns to be open
    (near 1) for information that should be remembered long-term.

    This is the CONSTANT ERROR CAROUSEL: the cell state carries the
    gradient signal unchanged (when f≈1) rather than compressing it
    through a tanh (which would squeeze it toward 0).

    Gradient through the LSTM over T steps:
    ∂c_t/∂c_{t-k} = Π_{j=t-k}^{t-1} f_j

    If the forget gate is learned to be 1.0 for certain dimensions:
    ∂c_t/∂c_{t-k} = 1^k = 1  (perfect gradient flow at arbitrary depth)

    LSTMs made RNNs practical for sequences of 100-1000 steps.
    The transformer's attention mechanism generalises this further —
    it provides O(1) gradient distance from any position to any other.


##### PART 7 — WEIGHT INITIALISATION AND ITS ROLE IN GRADIENT FLOW

### Why Initialisation Determines the First Gradient

The first gradient determines whether training even begins successfully.
If activations collapse to near-zero or explode on the very first forward
pass — BEFORE any weight update — the model starts in a bad state that
may be impossible to escape.

    Too-small initialisation:
    If weights are initialised near 0, the output of each layer is near 0.
    tanh(0) ≈ 0, σ(0) = 0.5 (centered but with only max gradient 0.25).
    As depth increases, outputs get compressed toward a single point.
    Backpropagation through near-zero activations: gradient vanishes.

    Too-large initialisation:
    If weights are large, activations immediately saturate (for sigmoid/tanh).
    Gradient is near 0 everywhere.
    Or for ReLU: some activations explode into very large values.
    Subsequent normalisation layers may fail or produce numerical instability.


### The Variance Preservation Principle

Healthy initialisation satisfies the VARIANCE PRESERVATION condition:
the variance of activations should remain roughly constant as signals
propagate forward, and the variance of gradients should remain constant
as they propagate backward.

    For a linear layer y = Wx with n_in inputs:
    Var(yᵢ) = n_in · Var(w) · Var(x)

    To keep Var(y) = Var(x) (preserve signal variance):
    Var(w) = 1 / n_in

    ↔ Standard deviation σ_w = 1 / sqrt(n_in)
    ↔ Initialise weights from N(0, 1/n_in)

    For the BACKWARD pass, preserving gradient variance requires:
    Var(w) = 1 / n_out

    XAVIER / GLOROT INITIALISATION (Glorot & Bengio, 2010):
    Compromise between forward and backward variance:
    σ_w = sqrt(2 / (n_in + n_out))

    Or uniform: w ~ U[-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]

    Xavier is the default for linear layers with sigmoid or tanh activations.

    HE / KAIMING INITIALISATION (He et al., 2015):
    Specifically designed for ReLU activations.
    ReLU zeros out half the inputs, effectively halving the effective fan-in.
    Var(w) = 2 / n_in   (factor of 2 compensates for the ReLU kill)
    σ_w = sqrt(2 / n_in)

    He init is the default for ReLU and Leaky ReLU layers.

    In PyTorch:
    nn.init.xavier_uniform_(layer.weight)     ← for tanh, sigmoid
    nn.init.kaiming_normal_(layer.weight,     ← for ReLU, Leaky ReLU
                            nonlinearity='relu')


### Diagram 8 — Activation Variance Through Layers With Different Initialisations

    Activation variance (relative to layer 1)

    Layer:     1    5    10   20   40   60   80   100

    Too small: 1.0  0.1  0.01 0.00 0.00 0.00 0.00 0.00  ← collapses
    Xavier:    1.0  0.9  0.8  0.7  0.6  0.5  0.5  0.5   ← moderate decay
    He (ReLU): 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0   ← stable
    Too large: 1.0  3.1 12.0 ∞    ∞    ∞    ∞    ∞       ← explodes

    He initialisation + ReLU achieves the flattest activation variance
    profile across depth. This is why it is the standard for deep CNNs.


### Transformer-Specific Initialisation

Large transformer models use modified initialisation to keep the
residual stream stable through many layers:

    SCALED INITIALISATION (Wang et al., 2022, used in GPT-2/3):
    In a residual network with L layers, each residual branch adds
    to the residual stream. If each branch has variance σ², after
    L branches the residual stream has variance ≈ L · σ².
    For L=96 (GPT-3), the variance grows by 96× — a problem.

    Fix: scale DOWN the initialisation of the output projection in
    each residual branch by 1/sqrt(L):
    w ~ N(0, σ²/L)    where L = number of residual blocks

    This keeps the residual stream variance bounded regardless of depth.
    Used by GPT-2, GPT-3, and essentially all modern LLMs.

    In practice:
    for module in transformer.layers:
        nn.init.normal_(module.attn.proj.weight, std=0.02 / sqrt(2 * n_layers))
        nn.init.normal_(module.mlp.proj.weight,  std=0.02 / sqrt(2 * n_layers))


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — MONITORING GRADIENT FLOW IN PRACTICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What to Log Every Training Step

A comprehensive gradient health dashboard logs:

    1. Global gradient norm (pre-clipping):
       Total norm of all gradients concatenated.
       Should be roughly stable; spikes indicate explosions.

    2. Per-layer gradient norms:
       Log every N steps (expensive to compute every step).
       Should be roughly uniform across depth.
       Monotone decrease across depth indicates vanishing.

    3. Gradient-to-weight ratio (update ratio):
       Ratio of |gradient| / |weight| per parameter group.
       This indicates the effective learning rate per layer:
       ratio ≈ η · (grad_norm / weight_norm)
       Target: ~1e-3 to 1e-2. Much smaller = learning stalled.
       Much larger = updates too large relative to weight magnitudes.
       Proposed by Andrej Karpathy as a key diagnostic tool.

    4. Activation statistics (mean and std per layer):
       Detects saturation (std → 0), dead zones (mean → 0),
       or explosion (mean or std → large values).
       Computed via forward hooks.

    5. Gradient sparsity (for ReLU networks):
       Fraction of zero gradients at each layer.
       High sparsity (> 40-50%) suggests dying ReLU issue.


### Full Gradient Health Check: Code Pattern

    def log_gradient_health(model, writer, step, log_full=False):
        total_norm_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.norm(2).item()
            total_norm_sq += grad_norm ** 2
            if log_full and step % 500 == 0:
                writer.add_scalar(f'grad_norm/{name}', grad_norm, step)
                # Gradient-to-weight ratio
                if param.norm() > 0:
                    ratio = grad_norm / param.norm().item()
                    writer.add_scalar(f'grad_weight_ratio/{name}', ratio, step)
        total_norm = total_norm_sq ** 0.5
        writer.add_scalar('train/grad_norm_total', total_norm, step)
        return total_norm

    # In training loop:
    loss.backward()
    grad_norm = log_gradient_health(model, writer, step)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — DECISION FRAMEWORK: DIAGNOSING AND FIXING GRADIENT ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Gradient Problem Diagnosis Table

    ┌────────────────────────────────────────────────────────────────────────┐
    │ Symptom                    │ Most likely cause  │ Primary fix          │
    ├────────────────────────────────────────────────────────────────────────┤
    │ Loss does not decrease     │ Vanishing gradients│ Use ReLU/GELU,       │
    │ despite long training      │ in early layers    │ residual connections,│
    │                            │                    │ He init, LayerNorm   │
    ├────────────────────────────────────────────────────────────────────────┤
    │ Early layer grad norms     │ Vanishing gradients│ Residual connections,│
    │ near zero, late layers ok  │                    │ proper init, reduce  │
    │                            │                    │ depth, use GELU      │
    ├────────────────────────────────────────────────────────────────────────┤
    │ Loss spike / loss cliff    │ Exploding gradients│ Gradient clipping    │
    │                            │                    │ (max_norm=1.0),      │
    │                            │                    │ reduce LR            │
    ├────────────────────────────────────────────────────────────────────────┤
    │ Loss becomes NaN           │ Gradient explosion │ Gradient clipping,   │
    │                            │ → Inf weights      │ reduce LR, check     │
    │                            │                    │ for NaN in data      │
    ├────────────────────────────────────────────────────────────────────────┤
    │ High fraction of zero      │ Dying ReLU         │ Switch to Leaky ReLU │
    │ activations in ReLU layers │                    │ or GELU, reduce LR,  │
    │                            │                    │ He initialisation    │
    ├────────────────────────────────────────────────────────────────────────┤
    │ RNN fails on long          │ Vanishing through  │ Use LSTM or GRU,     │
    │ sequences                  │ time               │ or transformer       │
    ├────────────────────────────────────────────────────────────────────────┤
    │ Gradient-to-weight ratio   │ LR too low OR      │ Increase LR or check │
    │ < 1e-4 consistently        │ vanishing          │ for vanishing        │
    ├────────────────────────────────────────────────────────────────────────┤
    │ Gradient-to-weight ratio   │ LR too high OR     │ Decrease LR or add   │
    │ > 1e-1 consistently        │ explosion starting │ gradient clipping    │
    └────────────────────────────────────────────────────────────────────────┘


### Prevention Checklist for a New Architecture

    Before training begins, verify:

    □ Activation function: ReLU (CNNs), GELU (transformers), Leaky ReLU
      (MLPs). Avoid sigmoid/tanh in hidden layers of deep networks.

    □ Weight initialisation: He init for ReLU/Leaky ReLU, Xavier for
      tanh/sigmoid, scaled init for very deep transformers.

    □ Normalisation: LayerNorm (transformers), BatchNorm (CNNs).
      Place before or after residual addition (pre-norm preferred
      for transformers).

    □ Residual connections: for any network deeper than ~10 layers.

    □ Gradient clipping: always, especially for transformers.
      max_norm = 1.0 is the safe universal default.

    □ Gradient logging: log the global norm before clipping every step.
      Log per-layer norms every 500 steps.

    □ Check the first forward pass: run a single batch through the
      untrained model and verify:
      - Loss is close to the theoretical random-chance baseline
        (e.g., log(n_classes) for cross-entropy)
      - No NaN or Inf in outputs or loss
      - Activation standard deviations are O(1) at every layer

    □ Sanity-check the backward pass: verify that gradients exist
      for all parameters (no disconnected parts of the graph):
      for name, p in model.named_parameters():
          if p.grad is None:
              print(f"WARNING: {name} has no gradient")

"""


# ─────────────────────────────────────────────────────────────────────────────
# No OPERATIONS (theory-only module)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has ~20 leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None # _MAIN_SCRIPT

    scripts_available = main_script.exists()

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
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────


def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 1150
    try:
        from Training_Core.visuals.gradient_flow_visual import (
            GRADIENT_FLOW_VISUAL_HTML,
            GRADIENT_FLOW_VISUAL_HEIGHT,
        )
        visual_html   = GRADIENT_FLOW_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = GRADIENT_FLOW_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    OPERATIONS,
    }