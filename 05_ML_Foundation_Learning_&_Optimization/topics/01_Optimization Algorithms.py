"""
Optimisation Algorithms
========================

How neural networks and machine learning models learn — the mechanics of
navigating loss landscapes, the failure modes that prevent convergence,
and the family of algorithms from vanilla gradient descent to Adam that
power modern deep learning.

"""

import textwrap
import re

TOPIC_NAME = "Optimisation Algorithms"
DISPLAY_NAME = "01 · Optimisation Algorithms"
ICON = "⛰️"
SUBTITLE = "Navigating Loss Landscapes to Train Models"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Problem: Minimising a Loss Function

Training a machine learning model is an optimisation problem.
We have a model with parameters θ (weights and biases), a dataset of
n examples, and a loss function L(θ) that measures how wrong the model's
predictions are. Our goal:

                    θ* = argmin  L(θ)
                            θ

The loss function is a scalar-valued function of potentially millions or
billions of parameters. We cannot enumerate all possible θ — the space is
continuous and astronomically large. We need an iterative algorithm that
starts from some initial θ and takes steps toward lower loss.

The universal template for all gradient-based optimisers is:

                    θ_{t+1}  =  θ_t  +  Δθ_t

Every algorithm in this module is a different way of computing Δθ_t.

    Diagram 1 — The Loss Landscape Metaphor:

    A loss landscape is a surface in (θ, L) space.
    Optimisation = finding the lowest valley.

                    L(θ)
                      │
                 ╭─╮ │
                 │ │ │   ╭──╮
            ╭─╮  │ │ │  ╱    ╲     ╭──
            │ │  │ │ ╰─╯      ╲   ╱
            │ ╰──╯ │            ╲╱
            │      │              ← local minimum
            ╰──────┼──────────────────────────────  θ
                   │↑ global minimum
                   │saddle point ──── flat region ────

    Features of loss landscapes we must navigate:

    •  Global minimum    — the best possible parameter setting
    •  Local minima      — lower than surroundings but not globally optimal
    •  Saddle points     — gradient = 0 but not a minimum (flat in some dims,
                           downward-sloping in others)
    •  Plateaus          — large flat regions where gradient ≈ 0
    •  Ravines           — narrow curved valleys that cause oscillation
    •  Cliffs            — sudden steep gradients (common in RNNs)

    Practical reality in deep learning:
    ┌────────────────────────────────────────────────────────────────────┐
    │  High-dimensional loss landscapes of neural networks are           │
    │  surprisingly well-behaved: most local minima are near-global      │
    │  (Dauphin et al., 2014). The dominant challenge is saddle points   │
    │  and plateaus, not local minima.                                   │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Gradient Descent — The Foundation

The gradient ∇_θ L(θ) points in the direction of steepest ASCENT of the
loss. Moving in the opposite direction — gradient descent — reduces the loss.

    Update rule:

                    θ_{t+1}  =  θ_t  −  η · ∇_θ L(θ_t)

    where η (eta) is the **learning rate** — a hyperparameter controlling
    step size.

    Diagram 2 — Gradient Descent on a 2D Loss Surface:

    Contour lines = iso-loss curves (inner = lower loss)

              ┌────────────────────────────────────┐
              │  (start)                           │
              │    ●                               │
              │     ↘                              │
              │      ●                             │
              │       ↘                            │
              │        ●    ← steps follow         │
              │         ↘     the negative gradient│
              │          ●                         │
              │           ↘                        │
              │            ●  (converged near min) │
              └────────────────────────────────────┘

    The gradient is computed over the ENTIRE training dataset:

                    ∇L(θ)  =  (1/n) ∑_{i=1}^{n} ∇ L_i(θ, x_i, y_i)

    This is called **Batch Gradient Descent** (or Full-Batch GD).

    Properties of Batch GD:
    ┌────────────────────────────────────────────────────────────────────┐
    │  ✓  Stable updates — gradient is exact (uses all data)             │
    │  ✓  Guaranteed to decrease loss at each step (with small η)        │
    │  ✗  Extremely slow — must scan entire dataset per step             │
    │  ✗  Cannot run on datasets larger than memory                      │
    │  ✗  Parallelism is limited by memory, not compute                  │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### The Learning Rate — The Most Critical Hyperparameter

The learning rate η controls how large a step we take in the direction of
the negative gradient. It is the single most important hyperparameter
in any gradient-based optimiser.

    Diagram 3 — Effect of Learning Rate:

    LEARNING RATE TOO SMALL           LEARNING RATE JUST RIGHT
    ────────────────────────          ─────────────────────────

    Loss │╲                           Loss │╲
         │ ╲____                           │ ╲
         │      ╲____                      │  ╲___________
         │           ╲____                 │
         └──────────────────  steps        └───────────────  steps
    Converges but takes many          Converges efficiently.
    thousands of steps.


    LEARNING RATE TOO LARGE           LEARNING RATE CATASTROPHICALLY LARGE
    ─────────────────────────         ─────────────────────────────────────

    Loss │╲  ╱╲  ╱╲  ╱              Loss │
         │ ╲╱  ╲╱  ╲╱  oscillation       │                ╱
         │──────────────────              │          ╱╲  ╱
         └──────────────────  steps       │     ╱╲  ╱  ╲╱
    Oscillates around minimum.            │╲   ╱  ╲╱
    Converges slowly or not at all.       │ ╲ ╱
                                          └──────────────  steps
                                     Diverges — loss INCREASES.


    For a convex quadratic loss with eigenvalue spectrum [λ_min, λ_max]:

        Optimal learning rate:  η* = 2 / (λ_min + λ_max)
        Maximum stable η:       η < 2 / λ_max

    In practice:
    ┌────────────────────────────────────────────────────────────────────┐
    │  Good starting values: 1e-3 (Adam), 1e-2 to 1e-1 (SGD)             │
    │  Always use learning rate scheduling (decay over time)             │
    │  Always use learning rate warmup for very deep networks            │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Stochastic Gradient Descent (SGD)

Instead of computing the gradient over all n samples, SGD computes it
on a SINGLE randomly selected example per step:

                    ∇L̃(θ)  ≈  ∇ L_i(θ, x_i, y_i)     (i drawn randomly)

                    θ_{t+1}  =  θ_t  −  η · ∇ L_i(θ_t)

    The update is NOISY — a single example's gradient is a random variable
    with the true gradient as its expected value:

                    𝔼[∇ L_i(θ)]  =  ∇L(θ)

    Diagram 4 — SGD vs Batch GD Trajectories:

    Contour plot of loss surface (lower = better):

    BATCH GD                             SGD

         ○ start                              ○ start
          ╲                                    ╲↗╲
           ╲                                 ↙    ╲↗
            ╲                              ↙↗      ╲
             ╲                           ╱↙         ↘
              ●  converge                ○ converges (noisily)
                 smoothly

    SGD path oscillates due to noise in each gradient estimate.
    But it arrives at a good solution much faster in wall-clock time.

    Properties of SGD:
    ┌────────────────────────────────────────────────────────────────────┐
    │  ✓  One gradient per step → much faster iteration                  │
    │  ✓  Noise can escape shallow local minima and saddle points        │
    │  ✓  Works on arbitrarily large datasets (online learning)          │
    │  ✓  Implicit regularisation — noise biases toward flat minima      │
    │  ✗  High variance updates → noisy convergence                      │
    │  ✗  Sensitive to learning rate choice                              │
    │  ✗  Never fully converges — keeps oscillating near minimum         │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Mini-Batch Gradient Descent — The Practical Standard

Mini-batch GD is the compromise that dominates modern ML. Compute the
gradient on a small batch B of b randomly selected examples:

                    ∇L̃(θ)  =  (1/b) ∑_{i ∈ B} ∇ L_i(θ)

                    θ_{t+1}  =  θ_t  −  η · ∇L̃(θ_t)

    Typical batch sizes: 32, 64, 128, 256.

    Why this specific range? A balance of three forces:

    ┌────────────────────────────────────────────────────────────────────┐
    │  HARDWARE: GPUs parallelise matrix multiplications. A batch must   │
    │    fill the GPU — too small wastes compute; too large exceeds VRAM │
    │                                                                    │
    │  STATISTICS: Gradient estimate variance ∝ 1/b. Larger b =          │
    │    lower variance = smoother convergence. Diminishing returns.     │
    │                                                                    │
    │  GENERALISATION: Large batches → sharp minima → worse test         │
    │    performance. Small batches → flat minima → better test          │
    │    performance. (Keskar et al., 2016)                              │
    └────────────────────────────────────────────────────────────────────┘

    Batch size comparison:
    ┌────────────┬──────────────┬───────────────┬────────────────────────┐
    │ Batch Size │ Gradient     │ Steps per     │ Generalisation         │
    │            │ Accuracy     │ Epoch         │ Quality                │
    ├────────────┼──────────────┼───────────────┼────────────────────────┤
    │ b = 1      │ Very noisy   │ n (full scan) │ Best (flat minima)     │
    │ b = 32     │ Moderate     │ n/32          │ Good                   │
    │ b = 256    │ Low noise    │ n/256         │ Good                   │
    │ b = n      │ Exact        │ 1             │ Worst (sharp minima)   │
    └────────────┴──────────────┴───────────────┴────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Problem: Ravines, Oscillation, and Slow Progress

Consider a loss landscape shaped like a narrow, elongated valley —
a "ravine." The gradient is steep across the ravine (narrow dimension)
and shallow along it (long dimension).

    Diagram 5 — The Ravine Problem:

    Loss contours (elongated ellipses):

    ┌─────────────────────────────────────────────────┐
    │     ╭──────────────────────────────────────╮    │
    │    ╭╯╭────────────────────────────────────╮╰╮   │
    │   ╭╯ │         ← minimum valley           │ ╰╮  │
    │   │  ╰────────────────────────────────────╯  │  │
    │   ╰╮╭────────────────────────────────────╮╭╯ │  │
    │    ╰╯╰──────────────────────────────────╯╰╯  │  │
    └─────────────────────────────────────────────────┘

    Without momentum, GD steps oscillate ACROSS the ravine
    (gradient is large there) and barely move ALONG it
    (gradient is small there):

    Step: ↗ ↘ ↗ ↘ ↗ ↘ → → → ↗ ↘ ↗ ↘  (very slow zigzag)

    This is the ravine problem. The solution is MOMENTUM.

    Condition Number κ = λ_max / λ_min
    (ratio of curvature across ravine to along ravine)

    κ = 1     → sphere, GD converges in one step direction change
    κ = 10    → mild ravine, GD makes moderate progress
    κ = 1000  → severe ravine, GD takes ~1000 × more steps than needed
    κ → ∞     → GD essentially never converges


──────────────────────────────────────────────────────────────────────────────
### Momentum

Momentum borrows from physics: a ball rolling down a hill accumulates
velocity. Fast-moving directions keep accelerating; oscillating directions
cancel out. Introduces a velocity vector v_t that accumulates gradient history.

    Update rule:

                    v_t      =  β · v_{t-1}  +  ∇L̃(θ_t)
                    θ_{t+1}  =  θ_t  −  η · v_t

    where β ∈ [0, 1) is the momentum coefficient (typically 0.9).

    Unrolling the recursion shows v_t is an exponentially weighted
    moving average (EWMA) of past gradients:

                    v_t  =  ∑_{k=0}^{t}  β^k · ∇L̃(θ_{t-k})

    β = 0.9 means ≈ 10 past gradients contribute meaningfully.
    β = 0.99 means ≈ 100 past gradients contribute.

    Diagram 6 — Momentum vs Plain GD on a Ravine:

    PLAIN GD:                           MOMENTUM:
    ↗ ↘ ↗ ↘ ↗ ↘ → → → ↗ ↘ ↗ ↘        ╱─────────────────→
    Zigzags across ravine.              Oscillations cancel,
    Tiny net progress per step.         builds speed in valley direction.

    Effect on ravine:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Across-ravine gradients ALTERNATE SIGN → cancel in the EWMA      │
    │  Along-ravine gradients KEEP SAME SIGN  → accumulate in the EWMA  │
    │  Result: momentum suppresses oscillation and accelerates progress │
    └───────────────────────────────────────────────────────────────────┘

    Properties of Momentum:
    ✓  Dramatically faster on ravines and elongated loss surfaces
    ✓  Smooths noisy gradient estimates from mini-batches
    ✓  Can escape shallow saddle points by carrying kinetic energy
    ✗  Can overshoot the minimum (momentum carries it past)
    ✗  One more hyperparameter (β) to tune


──────────────────────────────────────────────────────────────────────────────
### Nesterov Accelerated Gradient (NAG)

Nesterov (1983) proposed a modification: instead of computing the gradient
at the CURRENT position θ_t, compute it at the ANTICIPATED future position
θ_t - η·β·v_{t-1} (where momentum would take us).

    Classical Momentum:
        v_t      = β·v_{t-1} + ∇L(θ_t)
        θ_{t+1}  = θ_t − η·v_t

    Nesterov:
        v_t      = β·v_{t-1} + ∇L(θ_t − η·β·v_{t-1})   ← look-ahead
        θ_{t+1}  = θ_t − η·v_t

    Diagram 7 — Classical Momentum vs Nesterov Look-Ahead:

    CLASSICAL MOMENTUM               NESTEROV

    Current position: ●              Current position: ●
    Take momentum step first.        Look ahead first:
          ↓                                ↓ (momentum step)
    Intermediate: ○                  Intermediate: ○
    Add gradient correction AFTER.   Compute gradient HERE.
          ↓                                ↓ (gradient at ○)
    Final: ◆                         Final: ◆

    Nesterov computes gradient at a more informed position.
    It "corrects" before overshooting rather than after.

    Practical result:
    ┌────────────────────────────────────────────────────────────────────┐
    │  NAG converges at O(1/t²) vs O(1/t) for GD — optimal for convex    │
    │  smooth problems. In deep learning, the improvement is visible     │
    │  but less dramatic because loss landscapes are non-convex.         │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### The Adaptive Learning Rate Problem

All methods so far use the SAME learning rate for every parameter.
But parameters can have wildly different gradient scales:

    - Some parameters have large, consistent gradients → need small η
    - Some parameters have tiny, rare gradients → need large η
    - Sparse features (most text tokens) appear infrequently → need
      large updates when they appear, small when they don't

    Diagram 8 — Why Per-Parameter Learning Rates Matter:

    Two parameters θ₁ and θ₂ with very different gradient scales:

    ∂L/∂θ₁  ≈  0.001   (very small gradient — feature appears rarely)
    ∂L/∂θ₂  ≈  10.0    (large gradient — feature appears constantly)

    With a single η = 0.01:
        θ₁ update: 0.01 × 0.001 = 0.00001   → barely moves
        θ₂ update: 0.01 × 10.0  = 0.1       → reasonable step

    We want θ₁ to have a much LARGER effective learning rate than θ₂.
    Adaptive methods solve this automatically.


──────────────────────────────────────────────────────────────────────────────
### AdaGrad (Adaptive Gradient Algorithm)

Duchi et al. (2011). Divides the learning rate by the square root of the
sum of all past squared gradients for each parameter.

    Update rule (per parameter j):

                    G_t,j    =  G_{t-1,j}  +  (∇L_j)²
                    θ_{t+1,j} =  θ_{t,j}  −  (η / √(G_t,j + ε)) · ∇L_j

    where G_t,j accumulates ALL past squared gradients, ε is a small
    constant (≈1e-8) for numerical stability.

    Intuition:

        G large (frequent large gradients) → η/√G is small  → slow update
        G small (rare or small gradients)  → η/√G is large  → fast update

    This is IDEAL for sparse features — words in NLP, items in
    recommender systems.

    Diagram 9 — AdaGrad Effective Learning Rate over Time:

    Parameter with large gradients:      Parameter with small gradients:
    G grows quickly → lr shrinks fast    G grows slowly → lr stays large

    Effective lr │╲                      Effective lr │──────────────
                 │  ╲                                 │         ╲
                 │    ╲─────────                      │           ╲──
                 └──────────── steps                  └──────────── steps

    Fatal flaw:
    ┌────────────────────────────────────────────────────────────────────┐
    │  G_t,j accumulates FOREVER. In long training runs, ALL parameters  │
    │  end up with effectively ZERO learning rate — training stops.      │
    │  AdaGrad is great for convex problems; poor for deep networks.     │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### RMSProp (Root Mean Square Propagation)

Hinton (2012, unpublished lecture notes). Fix AdaGrad's monotonically
decaying learning rate by replacing the sum of squared gradients with
an EXPONENTIALLY WEIGHTED MOVING AVERAGE (EWMA):

    Update rule:

                    v_t      =  ρ · v_{t-1}  +  (1-ρ) · (∇L)²
                    θ_{t+1}  =  θ_t  −  (η / √(v_t + ε)) · ∇L

    where ρ ≈ 0.9 is the decay rate (forget old gradients slowly).

    The EWMA means only RECENT gradient history matters. Old gradients
    are "forgotten" exponentially. This prevents the learning rate from
    shrinking to zero.

    Comparison:
    ┌─────────────────────┬──────────────────────┬────────────────────────┐
    │                     │ AdaGrad              │ RMSProp                │
    ├─────────────────────┼──────────────────────┼────────────────────────┤
    │ Gradient history    │ All past (sum)       │  Recent only (EWMA)    │
    │ Learning rate       │ Monotonically shrinks│  Adapts dynamically    │
    │ Long training       │ Stalls completely    │  Continues to update   │
    │ Best for            │ Convex, sparse data  │  Non-convex, RNNs      │
    └─────────────────────┴──────────────────────┴────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Adam (Adaptive Moment Estimation)

Kingma & Ba (2014). The dominant optimiser in deep learning. Combines:
    1. Momentum    — tracks the first moment (mean) of the gradient
    2. RMSProp     — tracks the second moment (uncentred variance)
    3. Bias correction — corrects for zero-initialisation of moments

    Update rule:

        m_t  =  β₁ · m_{t-1}  +  (1-β₁) · ∇L          ← 1st moment (EWMA of gradient)
        v_t  =  β₂ · v_{t-1}  +  (1-β₂) · (∇L)²       ← 2nd moment (EWMA of sq. gradient)

        m̂_t  =  m_t / (1 - β₁^t)                       ← bias-corrected 1st moment
        v̂_t  =  v_t / (1 - β₂^t)                       ← bias-corrected 2nd moment

        θ_{t+1}  =  θ_t  −  η · m̂_t / (√v̂_t + ε)

    Default hyperparameters (almost always kept at defaults):
        β₁ = 0.9    (1st moment decay — momentum)
        β₂ = 0.999  (2nd moment decay — RMSProp)
        ε  = 1e-8   (numerical stability)
        η  = 1e-3   (initial learning rate)

    Why bias correction matters:
    ┌────────────────────────────────────────────────────────────────────┐
    │  At t=1: m_1 = (1-β₁)·∇L = 0.1·∇L  ← severely underestimates       │
    │  Without correction, early steps are tiny and wasteful.            │
    │  Dividing by (1 - β₁^t) restores the true scale of the moments.    │
    └────────────────────────────────────────────────────────────────────┘

    Diagram 10 — Adam Effective Update Magnitude:

    Adam's update ≈ η · sign(∇L) for large gradients
    (because m̂/√v̂ ≈ mean_gradient / sqrt(mean_sq_gradient) ≈ signal/noise)

    This means Adam adapts not just to gradient magnitude but to the
    SIGNAL-TO-NOISE RATIO of each parameter's gradient stream.
    Parameters with consistent signal get large effective steps.
    Parameters with noisy signal get small effective steps.

    Properties of Adam:
    ✓  Works well with minimal tuning (default β₁, β₂ usually fine)
    ✓  Handles sparse gradients well (like AdaGrad)
    ✓  Handles non-stationary gradients well (like RMSProp)
    ✓  Bias correction prevents bad early steps
    ✓  Effective lr ≈ η regardless of gradient scale — stable
    ✗  Can converge to sharp minima → slightly worse generalisation than SGD
    ✗  Weight decay is broken in Adam (see AdamW below)
    ✗  Does not converge to optimal solution in some convex settings


──────────────────────────────────────────────────────────────────────────────
### AdamW — Fixing Weight Decay in Adam

Loshchilov & Hutter (2017). Discovered that L2 regularisation in Adam
does NOT behave like weight decay — the adaptive learning rate modifies
how the penalty is applied.

    Standard Adam with L2 reg (WRONG way to do weight decay):
        ∇L_reg = ∇L + λ·θ                        ← adds to gradient
        Adam update uses ∇L_reg                    ← adaptive scaling distorts λ

    AdamW (CORRECT weight decay):
        θ_{t+1} = θ_t − η · m̂_t/(√v̂_t + ε)  −  η·λ·θ_t
                                                    ↑
                                        weight decay applied DIRECTLY,
                                        bypassing adaptive scaling

    ┌────────────────────────────────────────────────────────────────────┐
    │  AdamW is now the default for Transformer training (GPT, BERT,     │
    │  ViT, etc.) because proper weight decay dramatically improves      │
    │  generalisation on large models. Use AdamW over Adam by default.   │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Second-Order Methods

All methods above are FIRST-ORDER — they use only gradient information
(first derivative). Second-order methods also use curvature information
(second derivative / Hessian), enabling much more informed steps.

    Newton's Method:

                    θ_{t+1}  =  θ_t  −  H⁻¹ · ∇L

    where H is the Hessian matrix of second derivatives:

                    H_{ij}  =  ∂²L / ∂θ_i ∂θ_j

    The Hessian tells us how curved the loss is in every direction.
    H⁻¹ rescales the gradient so that:
    - Steep dimensions get small steps (already has large curvature)
    - Flat dimensions get large steps (needs to cover more ground)

    This automatically solves the ravine problem without a learning rate!

    Diagram 11 — Newton vs GD on a Ravine:

    Ravine loss surface (elongated ellipse):

    Newton's method:              GD (no momentum):
    Steps are perfectly scaled:   Zigzags:
           →→→→→●                 ↗↘↗↘↗↘↗↘→●
    Converges in ~1 step.         Converges in ~κ steps.

    Why we don't use pure Newton's method in deep learning:
    ┌────────────────────────────────────────────────────────────────────┐
    │  For d parameters, H is d×d. For d=10⁸, storing H requires         │
    │  10^16 bytes = 10 petabytes. Computing H⁻¹ is O(d³). Completely    │
    │  intractable for modern neural networks.                           │
    └────────────────────────────────────────────────────────────────────┘

    L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno):

    Approximates H⁻¹ implicitly from the history of gradient differences,
    using only m recent (θ, ∇L) pairs (typically m=10-20).

    Memory cost: O(m·d) — feasible.

    ┌────────────────────────────────────────────────────────────────────┐
    │  L-BFGS is the standard for small-to-medium batch settings:        │
    │  scikit-learn's LogisticRegression uses L-BFGS by default.         │
    │  It requires the FULL batch gradient — unsuitable for SGD.         │
    │  Dominates for classical ML; rarely used in deep learning.         │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Learning Rate Scheduling

A fixed learning rate is rarely optimal. The standard approach is to
start with a larger rate (fast early progress) and reduce it over time
(fine-grained convergence).

    Schedule 1 — Step Decay:
        Multiply lr by a factor (e.g., 0.1) every N epochs.
        Simple and widely used.

    Schedule 2 — Exponential Decay:
        η_t = η₀ · exp(−k·t)
        Smooth continuous decay.

    Schedule 3 — Cosine Annealing:
        η_t = η_min + ½·(η_max − η_min)·(1 + cos(πt/T))
        Smooth, principled; dominant in modern deep learning.

    Schedule 4 — Warmup + Decay:
        Phase 1 (warmup): Linearly increase η from 0 to η_max
        Phase 2 (decay):  Apply cosine or linear decay

        This is the Transformer training recipe.

    Diagram 12 — Learning Rate Schedules Compared:

    η
    │
    │──────────╮                     ← Step Decay (drops suddenly)
    │          │──────╮
    │                 │──────────
    │
    │╲                               ← Exponential Decay (smooth)
    │ ╲──────────────────────────
    │
    │   ╭──╮                         ← Cosine Annealing (smooth waves)
    │  ╱    ╲      ╭──╮
    │ ╱      ╲    ╱    ╲
    │╱        ╲──╱      ╲──
    │
    │  ╱╲                            ← Warmup + Cosine Decay
    │ ╱  ╲────────────────
    │╱
    └───────────────────────────────  epoch


    Schedule 5 — Cyclical Learning Rates (Smith, 2017):
        Oscillate η between η_min and η_max periodically.
        Counterintuitively helpful: the higher lr phases explore,
        the lower lr phases exploit. Reduces need for precise lr tuning.

    Schedule 6 — 1-Cycle Policy (Super-Convergence):
        One large cycle: warmup to η_max, cosine decay to η_min/10.
        Can converge in 10× fewer epochs than step decay.


──────────────────────────────────────────────────────────────────────────────
### Gradient Clipping — Handling Exploding Gradients

In deep networks and RNNs, gradients can grow exponentially through many
layers (the exploding gradient problem). A single enormous gradient step
can destroy training.

    Gradient clipping caps the gradient norm before applying the update:

    Clip by Norm (most common):
        if ||∇L||₂ > τ:
            ∇L  ←  ∇L · τ / ||∇L||₂
        (scales gradient down to norm τ, preserving direction)

    Clip by Value (simpler):
        ∇L_j  ←  clip(∇L_j, −τ, +τ)   for each component j
        (clips each component independently — distorts direction)

    ┌────────────────────────────────────────────────────────────────────┐
    │  Standard practice for RNNs and Transformers: clip by norm,        │
    │  τ = 1.0 or τ = 5.0. Almost never needed for feedforward nets      │
    │  with batch normalisation or careful weight initialisation.        │
    └────────────────────────────────────────────────────────────────────┘

    Diagram 13 — Exploding Gradient Without Clipping:

    Loss │
         │                                    ╱ (loss explodes)
         │                               ╱╲  ╱
         │                    ╱╲    ╱╲  ╱  ╲╱
         │╲            ╱╲    ╱  ╲  ╱  ╲╱
         │  ╲──────────  ╲──╱    ╲╱              ← Normal training
         └──────────────────────────────────  step
                      ↑
                      Gradient cliff — one step destroys training


──────────────────────────────────────────────────────────────────────────────
### Comparison of All Optimisers

    ┌──────────────┬────────┬──────────┬────────────┬───────────────────────┐
    │ Optimiser    │ Speed  │ Ravines  │ Sparse     │ Best Use Case         │
    │              │        │          │ Gradients  │                       │
    ├──────────────┼────────┼──────────┼────────────┼───────────────────────┤
    │ Batch GD     │ Slow   │ Poor     │ Poor       │ Small convex problems │
    │ SGD          │ Fast   │ Poor     │ Poor       │ Simple models         │
    │ SGD+Momentum │ Fast   │ Good     │ Poor       │ CNNs (with tuning)    │
    │ NAG          │ Fast   │ Better   │ Poor       │ Convex + momentum     │
    │ AdaGrad      │ Fast   │ Good     │ Excellent  │ Sparse, convex        │
    │ RMSProp      │ Fast   │ Good     │ Good       │ RNNs, non-stationary  │
    │ Adam         │ Fast   │ Good     │ Excellent  │ General deep learning │
    │ AdamW        │ Fast   │ Good     │ Excellent  │ Transformers, LLMs    │
    │ L-BFGS       │ Medium │ Optimal  │ Poor       │ Classical ML, convex  │
    └──────────────┴────────┴──────────┴────────────┴───────────────────────┘

    Practical Decision Guide:
    ┌────────────────────────────────────────────────────────────────────┐
    │  Deep learning (any)     → AdamW with cosine decay + warmup        │
    │  Transformer / LLM       → AdamW with warmup, weight decay 0.1     │
    │  CNN (image recognition) → SGD + Momentum (0.9) + cosine           │
    │                            (often beats Adam at convergence)       │
    │  RNN / LSTM              → Adam or RMSProp + gradient clipping     │
    │  Classical ML (sklearn)  → L-BFGS (default in many solvers)        │
    │  Sparse NLP features     → AdaGrad or Adam                         │
    │  Hyperparameter search   → Try Adam first, tune SGD if needed      │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### The SGD vs Adam Generalisation Gap

One of the most studied empirical observations in deep learning:

    SGD + Momentum (with careful tuning) often generalises BETTER
    than Adam, even though Adam converges faster and with less tuning.

    Why?
    1. SGD favours FLAT minima (broader, more forgiving parameter regions)
    2. Adam favours SHARP minima (narrow, precise parameter regions)
    3. Flat minima generalise better — small perturbations in θ don't
       change predictions much, so the model is robust to distribution shift.

    Diagram 14 — Flat vs Sharp Minima:

    Flat minimum:                   Sharp minimum:
         Loss                             Loss
           │                               │
           │    ╭─────────────╮            │        ╭╮
           │   ╱               ╲           │       ╱  ╲
           │──╯                 ╰──        │──────╯    ╰──────
           └────────────────────── θ       └────────────────── θ

    Test loss is similar nearby.    Test loss spikes nearby.
    Robust to test distribution     Fragile to distribution shift.
    shift. Good generalisation.     Poor generalisation.

    ┌────────────────────────────────────────────────────────────────────┐
    │  Practical recommendation: When final accuracy matters more than   │
    │  training speed (production models), switch from Adam to SGD with  │
    │  momentum after initial Adam training, or use AdamW with strong    │
    │  weight decay. When training speed matters (research, iteration),  │
    │  use Adam/AdamW throughout.                                        │
    └────────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Gradient Descent & Learning Rate Effects": {
        "description": (
            "Visualise gradient descent on a 2D convex loss surface (a quadratic "
            "bowl). Run with four different learning rates — too small, just right, "
            "slightly too large, and diverging — and plot the trajectories and "
            "convergence curves side by side. Includes step-by-step update tables."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Loss function: elongated quadratic bowl ───────────────────────────────
# L(theta) = theta[0]^2 / 2 + 10 * theta[1]^2 / 2
# Condition number κ = 10  (mild ravine)
A = np.array([[1.0, 0.0],
              [0.0, 10.0]])

def loss(theta):
    return 0.5 * theta @ A @ theta

def grad(theta):
    return A @ theta

# ── Run gradient descent ──────────────────────────────────────────────────
def run_gd(lr, n_steps=80, theta0=None):
    if theta0 is None:
        theta0 = np.array([3.0, 2.5])
    theta = theta0.copy().astype(float)
    path   = [theta.copy()]
    losses = [loss(theta)]
    for _ in range(n_steps):
        theta = theta - lr * grad(theta)
        path.append(theta.copy())
        losses.append(loss(theta))
    return np.array(path), np.array(losses)

learning_rates = {
    "η = 0.005  (too small)":  0.005,
    "η = 0.09   (just right)": 0.09,
    "η = 0.18   (too large)":  0.18,
    "η = 0.22   (diverging)":  0.22,
}

print("=" * 65)
print("  GRADIENT DESCENT — LEARNING RATE SENSITIVITY")
print("  Loss: L(θ) = 0.5·θ₁² + 5·θ₂²  (κ = 10)")
print("=" * 65)
print()
print(f"  {'Learning Rate':<28}  {'Final Loss':>12}  {'Steps to <0.01':>14}  Status")
print("  " + "─" * 70)

results = {}
for name, lr in learning_rates.items():
    path, losses = run_gd(lr)
    final_loss = losses[-1]
    reached = next((i for i, l in enumerate(losses) if l < 0.01), None)
    status = ("CONVERGED" if final_loss < 0.01
              else "OSCILLATING" if final_loss < 10
              else "DIVERGED")
    reached_str = str(reached) if reached else "—"
    results[name] = (path, losses, lr)
    print(f"  {name:<28}  {final_loss:>12.6f}  {reached_str:>14}  {status}")

print()
print("  Optimal lr for this problem: η < 2/λ_max = 2/10 = 0.20")
print("  Just-right lr converges smoothly; diverging lr explodes past maximum.")

# ── Print step-by-step for just-right lr ─────────────────────────────────
path_good, losses_good = run_gd(0.09, n_steps=20)
print()
print("  Step-by-step: η = 0.09 (just right)")
print(f"  {'Step':>5}  {'θ₁':>10}  {'θ₂':>10}  {'Loss':>12}")
print("  " + "─" * 42)
for step in [0, 1, 2, 3, 5, 10, 15, 20]:
    th = path_good[step]
    ls = losses_good[step]
    print(f"  {step:>5}  {th[0]:>10.4f}  {th[1]:>10.4f}  {ls:>12.6f}")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Gradient Descent: Learning Rate Sensitivity\\n"
             "L(θ) = 0.5θ₁² + 5θ₂²  (condition number κ=10)",
             fontsize=12, fontweight="bold")

theta_vals = np.linspace(-3.5, 3.5, 300)
T1, T2 = np.meshgrid(theta_vals, theta_vals)
Z = 0.5 * T1**2 + 5 * T2**2
levels = np.logspace(-2, 2, 25)

colours = ["steelblue", "seagreen", "darkorange", "crimson"]

for col, ((name, lr), colour) in enumerate(zip(learning_rates.items(), colours)):
    path, losses, _ = results[name]

    # Top row: trajectory on contour plot
    ax_top = axes[0, col]
    ax_top.contour(T1, T2, Z, levels=levels, cmap="Blues", alpha=0.6)
    ax_top.contourf(T1, T2, Z, levels=levels, cmap="Blues", alpha=0.15)
    ax_top.plot(path[:, 0], path[:, 1], "-o", color=colour,
                ms=3, lw=1.5, alpha=0.8)
    ax_top.plot(path[0, 0], path[0, 1], "ko", ms=8, label="Start")
    ax_top.plot(0, 0, "r*", ms=12, label="Minimum")
    ax_top.set_title(name.split(" (")[0], fontsize=10, fontweight="bold")
    ax_top.set_xlabel("θ₁", fontsize=9)
    ax_top.set_ylabel("θ₂", fontsize=9)
    ax_top.set_xlim(-3.5, 3.5)
    ax_top.set_ylim(-3.5, 3.5)
    ax_top.legend(fontsize=7)

    # Bottom row: loss curve
    ax_bot = axes[1, col]
    steps = np.arange(len(losses))
    valid = losses[losses < 1e6]
    ax_bot.semilogy(steps[:len(valid)], valid, color=colour, lw=2)
    ax_bot.axhline(0.01, color="gray", ls="--", lw=1, label="L < 0.01")
    ax_bot.set_title(f"Convergence Curve\\n(log scale)", fontsize=9)
    ax_bot.set_xlabel("Step", fontsize=9)
    ax_bot.set_ylabel("Loss (log)", fontsize=9)
    ax_bot.legend(fontsize=7)
    ax_bot.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gd_learning_rates.png", dpi=110)
print()
print("  Plot saved → gd_learning_rates.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Momentum & Nesterov on a Ravine": {
        "description": (
            "Compare plain SGD, classical momentum, and Nesterov accelerated "
            "gradient on a highly elongated quadratic (condition number κ=100). "
            "Plots trajectories, convergence speed, and prints the exact number "
            "of steps each method needs to reach loss < 0.001."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Loss: SEVERE ravine — condition number κ = 100 ────────────────────────
# L(θ) = 0.5 * θ₁² + 50 * θ₂²
# Eigenvalues: λ_min=1, λ_max=100  → κ=100

def loss(theta):
    return 0.5 * theta[0]**2 + 50 * theta[1]**2

def grad_f(theta):
    return np.array([theta[0], 100 * theta[1]])

THETA0  = np.array([2.5, 1.8])
N_STEPS = 300
TARGET  = 1e-3

# ── Algorithm implementations ─────────────────────────────────────────────
def plain_gd(eta=0.018, n=N_STEPS):
    theta = THETA0.copy().astype(float)
    path, losses = [theta.copy()], [loss(theta)]
    for _ in range(n):
        theta = theta - eta * grad_f(theta)
        path.append(theta.copy()); losses.append(loss(theta))
    return np.array(path), np.array(losses)

def momentum_gd(eta=0.018, beta=0.9, n=N_STEPS):
    theta, vel = THETA0.copy().astype(float), np.zeros(2)
    path, losses = [theta.copy()], [loss(theta)]
    for _ in range(n):
        vel   = beta * vel + grad_f(theta)
        theta = theta - eta * vel
        path.append(theta.copy()); losses.append(loss(theta))
    return np.array(path), np.array(losses)

def nesterov_gd(eta=0.018, beta=0.9, n=N_STEPS):
    theta, vel = THETA0.copy().astype(float), np.zeros(2)
    path, losses = [theta.copy()], [loss(theta)]
    for _ in range(n):
        look_ahead = theta - eta * beta * vel
        vel        = beta * vel + grad_f(look_ahead)
        theta      = theta - eta * vel
        path.append(theta.copy()); losses.append(loss(theta))
    return np.array(path), np.array(losses)

algos = {
    "Plain GD":          plain_gd(),
    "Momentum β=0.9":    momentum_gd(),
    "Nesterov β=0.9":    nesterov_gd(),
}

print("=" * 65)
print("  MOMENTUM ON A RAVINE")
print("  Loss: L(θ) = 0.5·θ₁² + 50·θ₂²  (condition number κ = 100)")
print(f"  η = 0.018 for all methods  |  β = 0.9 for Momentum & NAG")
print("=" * 65)
print()
print(f"  {'Algorithm':<24}  {'Final Loss':>12}  {'Steps to < 0.001':>16}  {'Speedup'}  ")
print("  " + "─" * 65)

gd_steps = None
for name, (path, losses) in algos.items():
    final_loss = losses[-1]
    reached    = next((i for i, l in enumerate(losses) if l < TARGET), None)
    reached_str = str(reached) if reached else f"> {N_STEPS}"
    if name == "Plain GD" and reached:
        gd_steps = reached
    speedup = ""
    if name != "Plain GD" and reached and gd_steps:
        speedup = f"{gd_steps / reached:.1f}×"
    elif name == "Plain GD":
        speedup = "1.0×  (baseline)"
    print(f"  {name:<24}  {final_loss:>12.6f}  {reached_str:>16}  {speedup}")

print()
print("  Observations:")
print("  - Plain GD zigzags across the ravine (large across-valley gradient)")
print("  - Momentum: across-valley gradients CANCEL each other in the EWMA")
print("  - Along-valley gradients ACCUMULATE → momentum accelerates in right dir")
print("  - Nesterov looks ahead before computing gradient → corrects sooner")
print(f"  - On κ=100, momentum is ~100/√100 = 10× faster than GD (theoretical)")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("SGD vs Momentum vs Nesterov on a Severe Ravine (κ=100)\\n"
             "L(θ) = 0.5θ₁² + 50θ₂²",
             fontsize=12, fontweight="bold")

theta_plot = np.linspace(-3, 3, 300)
T1, T2 = np.meshgrid(theta_plot, np.linspace(-2, 2, 300))
Z = 0.5 * T1**2 + 50 * T2**2
levels = np.logspace(-2, 2.5, 30)
colours = {"Plain GD": "tomato", "Momentum β=0.9": "steelblue",
           "Nesterov β=0.9": "seagreen"}

for col, (name, (path, losses)) in enumerate(algos.items()):
    ax = axes[col]
    ax.contourf(T1, T2, Z, levels=levels, cmap="Greys", alpha=0.3)
    ax.contour(T1, T2, Z, levels=levels, colors="gray", alpha=0.4,
               linewidths=0.6)
    plot_steps = min(80, len(path))
    ax.plot(path[:plot_steps, 0], path[:plot_steps, 1],
            "-o", color=colours[name], ms=2.5, lw=1.5, alpha=0.85)
    ax.plot(path[0, 0], path[0, 1], "ko", ms=8, label="Start", zorder=5)
    ax.plot(0, 0, "r*", ms=12, label="Minimum", zorder=5)
    reached = next((i for i, l in enumerate(losses) if l < TARGET), None)
    reached_str = f"{reached} steps" if reached else f"> {N_STEPS}"
    ax.set_title(f"{name}\\n(steps to L<0.001: {reached_str})",
                 fontsize=10, fontweight="bold", color=colours[name])
    ax.set_xlabel("θ₁", fontsize=9)
    ax.set_ylabel("θ₂", fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=8)
    ax.set_aspect("auto")

plt.tight_layout()
plt.savefig("momentum_ravine.png", dpi=110)
print()
print("  Plot saved → momentum_ravine.png")

# ── Loss curves ───────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
for name, (path, losses) in algos.items():
    ax2.semilogy(losses[:100], lw=2, label=name, color=colours[name])
ax2.axhline(TARGET, color="gray", ls="--", lw=1, label=f"Target loss = {TARGET}")
ax2.set_title("Convergence Comparison: GD vs Momentum vs Nesterov\\n(log scale)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss (log scale)")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("momentum_convergence.png", dpi=110)
print("  Plot saved → momentum_convergence.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Adaptive Optimisers Head-to-Head": {
        "description": (
            "Train a small two-layer neural network on a synthetic classification "
            "task using SGD, SGD+Momentum, AdaGrad, RMSProp, and Adam. "
            "Plots training loss, validation accuracy, and effective learning "
            "rate per parameter over training. Demonstrates Adam's robustness "
            "and faster convergence with default hyperparameters."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Synthetic dataset: two interleaved spirals (non-linearly separable) ───
def make_spirals(n=400, noise=0.3):
    n_half = n // 2
    theta0 = np.sqrt(np.random.rand(n_half)) * 3.5 * np.pi
    X0 = np.c_[-np.cos(theta0) * theta0 + np.random.randn(n_half) * noise,
                np.sin(theta0) * theta0 + np.random.randn(n_half) * noise]
    theta1 = np.sqrt(np.random.rand(n_half)) * 3.5 * np.pi
    X1 = np.c_[ np.cos(theta1) * theta1 + np.random.randn(n_half) * noise,
               -np.sin(theta1) * theta1 + np.random.randn(n_half) * noise]
    X = np.vstack([X0, X1])
    y = np.array([0] * n_half + [1] * n_half)
    idx = np.random.permutation(n)
    return X[idx], y[idx]

X, y = make_spirals(n=600)
X = (X - X.mean(0)) / X.std(0)

split = 480
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# ── Tiny 2-layer neural network with numpy (for transparency) ────────────
class TwoLayerNet:
    def __init__(self, n_in=2, n_hid=32, n_out=2, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in,  n_hid)) * np.sqrt(2 / n_in)
        self.b1 = np.zeros(n_hid)
        self.W2 = rng.standard_normal((n_hid, n_out)) * np.sqrt(2 / n_hid)
        self.b2 = np.zeros(n_out)
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        exp_z   = np.exp(self.Z2 - self.Z2.max(1, keepdims=True))
        self.P  = exp_z / exp_z.sum(1, keepdims=True)
        return self.P

    def backward(self, y):
        n    = len(y)
        dZ2  = self.P.copy(); dZ2[np.arange(n), y] -= 1; dZ2 /= n
        dW2  = self.A1.T @ dZ2
        db2  = dZ2.sum(0)
        dA1  = dZ2 @ self.W2.T
        dZ1  = dA1 * (1 - self.A1**2)          # tanh derivative
        dW1  = self.X.T @ dZ1
        db1  = dZ1.sum(0)
        return [dW1, db1, dW2, db2]

    def loss(self, y):
        n = len(y)
        return -np.log(self.P[np.arange(n), y] + 1e-12).mean()

    def accuracy(self, X, y):
        return (self.forward(X).argmax(1) == y).mean() * 100

# ── Optimiser implementations ─────────────────────────────────────────────
class SGDOpt:
    def __init__(self, params, lr=0.05):
        self.lr = lr
    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g

class MomentumOpt:
    def __init__(self, params, lr=0.05, beta=0.9):
        self.lr, self.beta = lr, beta
        self.v = [np.zeros_like(p) for p in params]
    def step(self, params, grads):
        for v, p, g in zip(self.v, params, grads):
            v[:] = self.beta * v + g
            p   -= self.lr * v

class AdaGradOpt:
    def __init__(self, params, lr=0.1, eps=1e-8):
        self.lr, self.eps = lr, eps
        self.G = [np.zeros_like(p) for p in params]
    def step(self, params, grads):
        for G, p, g in zip(self.G, params, grads):
            G[:] += g**2
            p    -= self.lr * g / (np.sqrt(G) + self.eps)

class RMSPropOpt:
    def __init__(self, params, lr=0.01, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps = lr, rho, eps
        self.v = [np.zeros_like(p) for p in params]
    def step(self, params, grads):
        for v, p, g in zip(self.v, params, grads):
            v[:] = self.rho * v + (1 - self.rho) * g**2
            p   -= self.lr * g / (np.sqrt(v) + self.eps)

class AdamOpt:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    def step(self, params, grads):
        self.t += 1
        for m, v, p, g in zip(self.m, self.v, params, grads):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g**2
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p    -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ── Training loop ─────────────────────────────────────────────────────────
EPOCHS     = 200
BATCH_SIZE = 64
N_BATCHES  = len(X_tr) // BATCH_SIZE

optimiser_configs = {
    "SGD  (η=0.05)":       lambda p: SGDOpt(p, lr=0.05),
    "Momentum (η=0.05)":   lambda p: MomentumOpt(p, lr=0.05, beta=0.9),
    "AdaGrad (η=0.1)":     lambda p: AdaGradOpt(p, lr=0.1),
    "RMSProp (η=0.01)":    lambda p: RMSPropOpt(p, lr=0.01),
    "Adam    (η=0.001)":   lambda p: AdamOpt(p, lr=1e-3),
}

all_train_loss = {}
all_val_acc    = {}

print("=" * 65)
print("  ADAPTIVE OPTIMISERS — TRAINING A 2-LAYER NEURAL NET")
print("  Task: spiral classification (non-linear, 2-class)")
print(f"  Architecture: 2 → 32 → 2  (tanh)  |  {EPOCHS} epochs")
print("=" * 65)
print()

colours = {
    "SGD  (η=0.05)":     "tomato",
    "Momentum (η=0.05)": "darkorange",
    "AdaGrad (η=0.1)":   "mediumpurple",
    "RMSProp (η=0.01)":  "steelblue",
    "Adam    (η=0.001)": "seagreen",
}

for opt_name, opt_fn in optimiser_configs.items():
    net = TwoLayerNet(seed=0)
    opt = opt_fn(net.params)
    train_losses, val_accs = [], []

    for epoch in range(EPOCHS):
        idx = np.random.permutation(len(X_tr))
        X_sh, y_sh = X_tr[idx], y_tr[idx]
        epoch_loss = 0
        for b in range(N_BATCHES):
            xb = X_sh[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            yb = y_sh[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            net.forward(xb)
            epoch_loss += net.loss(yb)
            grads = net.backward(yb)
            opt.step(net.params, grads)

        train_losses.append(epoch_loss / N_BATCHES)
        val_accs.append(net.accuracy(X_te, y_te))

    all_train_loss[opt_name] = train_losses
    all_val_acc[opt_name]    = val_accs
    final_loss = train_losses[-1]
    final_acc  = val_accs[-1]
    ep50_acc   = val_accs[49]
    print(f"  {opt_name:<22}  Final loss: {final_loss:.4f}  "
          f"  Val acc: {final_acc:.1f}%  (epoch 50: {ep50_acc:.1f}%)")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Adaptive Optimisers: 2-Layer Net on Spiral Classification",
             fontsize=12, fontweight="bold")

for name in optimiser_configs:
    c = colours[name]
    ax1.semilogy(all_train_loss[name], lw=2, label=name, color=c)
    ax2.plot(all_val_acc[name], lw=2, label=name, color=c)

ax1.set_title("Training Loss (log scale)", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss (log)")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2.set_title("Validation Accuracy", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(40, 102)
ax2.axhline(90, color="gray", ls="--", lw=1, label="90% line")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("optimiser_comparison.png", dpi=110)
print()
print("  Plot saved → optimiser_comparison.png")
print()
print("  Key Takeaways:")
print("  - Adam converges fastest with default hyperparameters")
print("  - SGD and Momentum are competitive but need careful lr tuning")
print("  - AdaGrad stalls on long training runs (lr shrinks to near-zero)")
print("  - RMSProp is robust — good middle ground between AdaGrad and Adam")
print("  - Adaptive methods remove the need to carefully tune lr for each param")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Learning Rate Schedules & Warm Restarts": {
        "description": (
            "Visualise and compare six learning rate schedules — constant, "
            "step decay, exponential decay, cosine annealing, linear warmup "
            "plus cosine, and cyclical (1cycle). Train a model with each "
            "schedule and compare final loss and convergence speed. "
            "Includes a demonstration of super-convergence with 1cycle."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Learning rate schedule functions ──────────────────────────────────────
def sched_constant(epoch, total, lr_max=0.1, **kw):
    return lr_max

def sched_step_decay(epoch, total, lr_max=0.1, drop=0.5, drop_every=30, **kw):
    return lr_max * (drop ** (epoch // drop_every))

def sched_exp_decay(epoch, total, lr_max=0.1, k=0.03, **kw):
    return lr_max * np.exp(-k * epoch)

def sched_cosine(epoch, total, lr_max=0.1, lr_min=1e-5, **kw):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / total))

def sched_warmup_cosine(epoch, total, lr_max=0.1, lr_min=1e-5,
                        warmup_epochs=10, **kw):
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    cos_epochs = total - warmup_epochs
    cos_epoch  = epoch - warmup_epochs
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + np.cos(np.pi * cos_epoch / cos_epochs))

def sched_onecycle(epoch, total, lr_max=0.1, lr_min=1e-4, **kw):
    """One-cycle policy: warmup to lr_max then cosine decay to lr_min/10."""
    warmup = total // 2
    if epoch < warmup:
        return lr_min + (lr_max - lr_min) * epoch / warmup
    else:
        t = (epoch - warmup) / (total - warmup)
        return lr_min / 10 + 0.5 * (lr_max - lr_min / 10) * (1 + np.cos(np.pi * t))

schedules = {
    "Constant":           sched_constant,
    "Step Decay (×0.5 /30ep)": sched_step_decay,
    "Exponential Decay":  sched_exp_decay,
    "Cosine Annealing":   sched_cosine,
    "Warmup + Cosine":    sched_warmup_cosine,
    "1-Cycle Policy":     sched_onecycle,
}

TOTAL_EPOCHS = 120
LR_MAX = 0.1

# ── Simulate training loss under each schedule ────────────────────────────
# Model: loss(lr, step) based on convergence-theory-inspired simulation
# True minimum = 0.05. Loss decreases via SGD-like dynamics with noise.

def simulate_training(schedule, total=TOTAL_EPOCHS, base_lr=LR_MAX, seed=7):
    rng = np.random.default_rng(seed)
    theta = 1.0     # distance from optimum
    loss_history = []
    lr_history   = []

    for epoch in range(total):
        lr = schedule(epoch, total, lr_max=base_lr)
        # Simplified loss dynamics: gradient step + noise
        effective_grad = theta * (1 + rng.standard_normal() * 0.15)
        theta = theta * (1 - lr * 0.9) + lr * 0.15 * rng.standard_normal()
        theta = max(abs(theta), 0)
        loss  = 0.05 + 0.5 * theta**2 + rng.standard_normal() * 0.003
        loss_history.append(max(loss, 0.048))
        lr_history.append(lr)

    return np.array(lr_history), np.array(loss_history)

print("=" * 65)
print("  LEARNING RATE SCHEDULES COMPARISON")
print(f"  Total epochs: {TOTAL_EPOCHS}  |  lr_max = {LR_MAX}")
print("=" * 65)
print()
print(f"  {'Schedule':<30}  {'Final Loss':>10}  {'Min Loss':>10}  {'LR at end':>10}")
print("  " + "─" * 65)

results = {}
for sched_name, sched_fn in schedules.items():
    lr_hist, loss_hist = simulate_training(sched_fn)
    results[sched_name] = (lr_hist, loss_hist)
    final_lr = sched_fn(TOTAL_EPOCHS - 1, TOTAL_EPOCHS, lr_max=LR_MAX)
    print(f"  {sched_name:<30}  {loss_hist[-1]:>10.5f}  "
          f"{loss_hist.min():>10.5f}  {final_lr:>10.6f}")

print()
print("  Key Comparisons:")
print("  - Constant lr oscillates around minimum without fine convergence")
print("  - Step decay converges but has discontinuous jumps in loss")
print("  - Cosine annealing is smooth — no discontinuities, good final acc")
print("  - Warmup + cosine is the Transformer standard: stable early training")
print("  - 1-cycle aggressively explores (high lr) then precisely converges")
print("  - Warmup prevents divergence from bad early large-step updates")

# ── Print step-by-step lr for warmup+cosine and 1cycle ────────────────────
print()
print("  Warmup + Cosine schedule (key epochs):")
print(f"  {'Epoch':>6}  {'LR':>10}")
print("  " + "─" * 20)
for ep in [0, 5, 10, 20, 40, 60, 80, 100, 119]:
    lr = sched_warmup_cosine(ep, TOTAL_EPOCHS, lr_max=LR_MAX)
    phase = "warmup" if ep < 10 else "cosine"
    print(f"  {ep:>6}  {lr:>10.6f}  ({phase})")

# ── Plots ─────────────────────────────────────────────────────────────────
epochs = np.arange(TOTAL_EPOCHS)
colours = ["gray", "tomato", "darkorange", "steelblue", "seagreen", "purple"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Learning Rate Schedules: Shape and Effect on Training Loss",
             fontsize=12, fontweight="bold")

for idx, ((sched_name, (lr_hist, loss_hist)), colour) in enumerate(
        zip(results.items(), colours)):
    row, col = divmod(idx, 3)
    ax = axes[row, col]

    ax2 = ax.twinx()
    ax.plot(epochs, loss_hist, lw=2, color=colour, label="Loss", alpha=0.9)
    ax2.plot(epochs, lr_hist, lw=1.5, color=colour, ls="--",
             label="LR", alpha=0.6)

    ax.set_title(sched_name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9, color=colour)
    ax2.set_ylabel("Learning Rate", fontsize=8, color="gray")
    ax.tick_params(axis="y", labelcolor=colour)
    ax2.tick_params(axis="y", labelcolor="gray")

    final_loss = loss_hist[-1]
    ax.text(0.97, 0.97, f"Final: {final_loss:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour, alpha=0.8))
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("lr_schedules.png", dpi=110)
print()
print("  Plot saved → lr_schedules.png")

# ── Combined LR curves for comparison ─────────────────────────────────────
fig2, ax = plt.subplots(figsize=(13, 5))
for (sched_name, (lr_hist, _)), colour in zip(results.items(), colours):
    ax.plot(epochs, lr_hist, lw=2, label=sched_name, color=colour)
ax.set_title("All Learning Rate Schedules Compared", fontsize=11,
             fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lr_schedules_overlay.png", dpi=110)
print("  Plot saved → lr_schedules_overlay.png")
print()
print("  Final Takeaways:")
print("  - Never leave the learning rate constant in long training runs")
print("  - Cosine annealing is a reliable default for most deep learning")
print("  - Warmup (10–20 epochs) prevents instability at the start of training")
print("  - 1-Cycle is aggressive: can achieve same accuracy in fewer epochs")
print("  - Step decay works but the discontinuities cause loss spikes")
print("  - The schedule interacts with the optimiser — cosine + Adam is dominant")
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
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }