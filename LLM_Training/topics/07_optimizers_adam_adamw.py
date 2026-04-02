"""
Optimisers — Adam and AdamW
============================

The optimiser is the algorithm that translates gradients into parameter
updates. SGD is conceptually simple but converges slowly on the non-convex
loss landscapes of large language models. Adam and its corrected variant
AdamW are the workhorses of modern LLM training. Understanding them deeply —
the momentum buffers, the bias correction, the weight decay coupling — is
essential for debugging training instability and tuning hyperparameters.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Optimisers — Adam and AdamW"
DISPLAY_NAME = "07 · Optimizers: Adam & AdamW"
ICON         = "⚙️"
SUBTITLE     = "Adaptive Moment Estimation and Weight Decay"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Why SGD Is Not Enough

Vanilla stochastic gradient descent updates every parameter by the same
scalar learning rate:

    θ ← θ − lr · ∇L(θ)

This has two critical weaknesses for deep learning at scale:

**1. Every parameter gets the same step size.**
In a language model, some parameters are updated frequently (common token
embeddings, early-layer weights) while others are updated rarely (rare-token
embeddings, highly specific weights). A single learning rate is too large
for frequent parameters (causes oscillation) and too small for infrequent
parameters (causes slow convergence). An adaptive learning rate per parameter
is far more efficient.

**2. The loss landscape is highly curved and anisotropic.**
In some directions the loss is steep (small steps needed), in others it is
flat (large steps needed). SGD applies the same step size in all directions.
Scaling the step size by an estimate of the curvature — the second moment of
the gradient — dramatically improves convergence.

Both problems are solved by Adam.


### SGD with Momentum (a stepping stone)

Before Adam, a key improvement over vanilla SGD was **momentum**:

    v_t  = β · v_{t-1}  +  (1 − β) · g_t      ← exponential moving average of gradients
    θ    ← θ  −  lr · v_t

where g_t is the gradient at step t and β ≈ 0.9 is the momentum coefficient.

Momentum accumulates a velocity vector in directions of consistent gradient
and dampens oscillations in directions where the gradient keeps reversing.
Think of it as a ball rolling down a hill — it builds speed in the descent
direction and doesn't bounce around sharp valleys.

But momentum only tracks the *sign and direction* of gradients — it still
applies the same step size to all parameters.


### Adam — Adaptive Moment Estimation

Adam (Kingma & Ba, 2015) maintains two exponential moving averages per
parameter:

    m_t  = β₁ · m_{t-1}  +  (1 − β₁) · g_t          ← 1st moment (mean of g)
    v_t  = β₂ · v_{t-1}  +  (1 − β₂) · g_t²          ← 2nd moment (mean of g²)

Then applies **bias correction** to account for the fact that m and v are
initialised to zero (they are biased toward zero at the start of training):

    m̂_t  = m_t  / (1 − β₁ᵗ)                          ← bias-corrected 1st moment
    v̂_t  = v_t  / (1 − β₂ᵗ)                          ← bias-corrected 2nd moment

Finally, the parameter update:

    θ  ←  θ  −  lr · m̂_t / (√v̂_t + ε)

**Interpreting the components:**

    m̂_t:        A smoothed estimate of the gradient direction (like momentum)
    √v̂_t:       A smoothed estimate of the RMS of past gradients
    lr / √v̂_t:  An effective, per-parameter adaptive learning rate

Parameters that historically receive large gradients (large √v̂) get a
smaller effective step. Parameters with small gradients (small √v̂) get a
larger step. The learning rate is automatically scaled to be appropriate for
each parameter — this is the "adaptive" in Adam.


    **Diagram 1 — Adam's Three Buffers:**

    ADAM'S STATE PER PARAMETER
    ════════════════════════════════════════════════════════════════

    At each step t, for each parameter θ:

    ┌─────────────────────────────────────────────────────────────┐
    │  g_t  =  ∂L/∂θ                        ← current gradient   │
    │                                                              │
    │  m_t  =  β₁·m_{t-1} + (1-β₁)·g_t     ← 1st moment buffer  │
    │          ↑ typically 0.9                                     │
    │          "smooth gradient direction"                         │
    │                                                              │
    │  v_t  =  β₂·v_{t-1} + (1-β₂)·g_t²    ← 2nd moment buffer  │
    │          ↑ typically 0.999                                   │
    │          "smooth gradient magnitude²"                        │
    │                                                              │
    │  m̂_t  =  m_t  / (1 - β₁ᵗ)            ← bias-corrected m   │
    │  v̂_t  =  v_t  / (1 - β₂ᵗ)            ← bias-corrected v   │
    │                                                              │
    │  θ  ←  θ  -  lr · m̂_t / (√v̂_t + ε)  ← parameter update   │
    └─────────────────────────────────────────────────────────────┘

    Memory cost of Adam: 2 extra tensors (m, v) per parameter
    → 3× the memory of the model itself (params + m + v)
    → For a 7B model in fp32: ~84 GB just for optimiser state


    **Diagram 2 — Bias Correction: Why It Matters at Step 1:**

    BIAS CORRECTION
    ════════════════════════════════════════════════════════════════

    At step t=1, starting from m₀ = v₀ = 0:

    m₁  = β₁ · 0  +  (1-β₁) · g₁  =  (1-β₁) · g₁  =  0.1 · g₁

    Without correction: m̂₁ = m₁ = 0.1 · g₁  ← 10× too small!
    With correction:    m̂₁ = m₁ / (1-β₁¹)  = 0.1·g₁ / 0.1 = g₁  ✓

    The correction factor (1 - β₁ᵗ) → 1 as t → ∞, so it only
    matters in the early steps (the "warm-up" period of the estimators).

    Without bias correction, the first hundreds of steps take
    abnormally tiny steps because m and v are still near zero.


### Typical Adam Hyperparameters for LLM Training

    β₁  = 0.9     (1st moment decay — controls gradient momentum)
    β₂  = 0.95 or 0.999   (2nd moment decay — controls RMS adaptation)
    ε   = 1e-8 or 1e-5    (numerical stability in denominator)
    lr  = 1e-4 to 3e-4    (peak learning rate, scheduler-dependent)

**β₂ = 0.95 vs 0.999:**
Many LLM recipes use β₂ = 0.95 instead of the default 0.999. A lower β₂
makes v_t more responsive to recent gradients, which helps in the early
training phase when the gradient distribution is changing rapidly. GPT-3,
Chinchilla, LLaMA all used β₂ = 0.95.

**ε sensitivity:**
ε prevents division by zero when √v̂ ≈ 0. For mixed-precision training
(fp16/bf16), ε = 1e-5 or larger may be needed because small v̂ values
become zero in low precision. With bf16, a common setting is ε = 1e-8
with the optimiser running in fp32 (master weights).


### The Weight Decay Problem with Adam

L2 regularisation is a standard technique to prevent overfitting. It adds a
penalty λ·‖θ‖² to the loss, which produces a gradient that pulls weights
toward zero:

    ∇L_regularised = ∇L + λ·θ

In SGD, this is equivalent to **weight decay** — decaying weights by a factor
each step:

    θ ← θ · (1 − λ·lr)  −  lr · ∇L

In SGD: L2 regularisation == weight decay. They are mathematically identical.

**In Adam, they are NOT equivalent.** When L2 is applied via the gradient,
Adam adapts the learning rate for the regularisation term too:

    Adam with L2:  update = lr · (m̂ + λ·θ) / (√v̂ + ε)

The effective weight decay becomes lr·λ / (√v̂ + ε), which varies per
parameter. Parameters with large gradients (large √v̂) get LESS regularisation.
This is incorrect — we want a uniform decay, not one that depends on gradient
history.


### AdamW — Decoupled Weight Decay

AdamW (Loshchilov & Hutter, 2019) fixes this by decoupling weight decay from
the gradient update:

    m_t  = β₁ · m_{t-1}  +  (1 − β₁) · g_t          ← same as Adam
    v_t  = β₂ · v_{t-1}  +  (1 − β₂) · g_t²          ← same as Adam

    θ  ←  θ · (1 − lr · λ)  −  lr · m̂_t / (√v̂_t + ε)
              ↑                  ↑
         weight decay         gradient step
         (uniform, direct)    (adaptive)

The weight decay θ · (1 − lr·λ) is applied directly to the parameters,
without going through the adaptive scaling. This restores the correct
regularisation behaviour: every parameter decays at the same rate, regardless
of its gradient history.

**Effect on training:** AdamW consistently outperforms Adam with L2
regularisation on LLM training. The difference matters most for:
    •   Parameters with large gradient variance (attention projections)
    •   Long training runs (the accumulated distortion compounds)
    •   Fine-tuning (where avoiding overfitting is critical)

All modern LLM training uses AdamW, not Adam.


    **Diagram 3 — Adam vs AdamW Weight Decay:**

    ADAM vs ADAMW: HOW WEIGHT DECAY IS APPLIED
    ════════════════════════════════════════════════════════════════

    Adam with L2 regularisation:
        ∇L_reg = ∇L  +  λ·θ                 ← add λθ to gradient
        m_t    = β₁·m_{t-1} + (1-β₁)·∇L_reg ← λθ also goes through EMA!
        v_t    = β₂·v_{t-1} + (1-β₂)·∇L_reg²
        update = m̂_t / (√v̂_t + ε)           ← λθ is adapted → wrong!

    AdamW (decoupled):
        m_t    = β₁·m_{t-1} + (1-β₁)·∇L     ← only true gradient in EMA
        v_t    = β₂·v_{t-1} + (1-β₂)·∇L²
        θ  ←  θ · (1-lr·λ)                   ← direct weight decay (correct!)
              −  lr · m̂_t / (√v̂_t + ε)      ← gradient update (adaptive)

    The two terms are now independent:
        - Weight decay: same effective rate for all parameters ✓
        - Gradient update: still adaptive per parameter ✓


### Memory Cost and Optimiser State

Adam/AdamW stores two extra tensors per parameter (m and v):

    Memory per parameter:
        Parameter:       1 × dtype_bytes
        Gradient:        1 × dtype_bytes
        1st moment (m):  1 × 4 bytes   (kept in fp32 even in mixed precision)
        2nd moment (v):  1 × 4 bytes   (kept in fp32 even in mixed precision)

    Total per parameter: ~12 bytes in mixed-precision training
                         ~16 bytes in pure fp32 training

    For a 7B model:
        fp32:  7B × 16 bytes = 112 GB  (params 28 + grads 28 + m 28 + v 28)
        bf16:  7B × 12 bytes = 84 GB   (params 14 + grads 14 + m 28 + v 28)

This is the primary reason LLMs require 8× GPU memory for training vs
inference, and why ZeRO (Module 24) partitions the optimiser state across GPUs.


### Practical Hyperparameter Guidelines for LLM Training

    Parameter         Typical range          Notes
    ─────────────────────────────────────────────────────────────────────
    lr (peak)         1e-4 … 3e-4           Higher for smaller models
    β₁                0.9                   Almost never changed
    β₂                0.95 … 0.999          0.95 for pre-training, 0.999 fine-tuning
    ε                 1e-8 … 1e-5           Higher for bf16/fp16 stability
    weight_decay      0.01 … 0.1            0.1 is common for pre-training
    grad_clip         0.5 … 1.0             Prevents loss spikes
    ─────────────────────────────────────────────────────────────────────

**Weight decay 0.1:** Aggressive by classical ML standards, but LLMs have
billions of parameters and are heavily over-parameterised. Strong weight
decay acts as a crucial regulariser and is consistently recommended.

**Learning rate and batch size:** The learning rate should scale with the
square root of the batch size (linear scaling rule is less reliable for LLMs).
When doubling the batch size, increase lr by √2.


### Adam Variants Used in Practice

    Variant        Key change                  Used by
    ─────────────────────────────────────────────────────────────
    Adam           Original (2015)             Legacy
    AdamW          Decoupled weight decay      All modern LLMs
    Adafactor      No m/v buffers, rank-1 approx  T5, PaLM (memory saving)
    LAMB           Layer-wise adaptive rates   BERT large-batch training
    Lion           Sign-based update (EMA sign) Some Google models
    Muon           Nesterov + orthogonalisation Some recent small models
    ─────────────────────────────────────────────────────────────

**Adafactor** is notable for not storing full m and v tensors — it uses
a factored low-rank approximation of v, reducing memory from O(P) to O(√P)
for 2D weight matrices. Used when memory is critically constrained.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Optimiser Comparison

| Property               | SGD          | SGD+Momentum | Adam              | AdamW             |
|------------------------|--------------|--------------|-------------------|-------------------|
| Adaptive lr?           | No           | No           | Yes (per-param)   | Yes (per-param)   |
| Momentum?              | No           | Yes          | Yes (m_t)         | Yes (m_t)         |
| Weight decay           | L2 via grad  | L2 via grad  | L2 via grad (wrong)| Decoupled (correct)|
| Extra memory / param   | 0            | 1 buffer     | 2 buffers         | 2 buffers         |
| Convergence speed      | Slow         | Moderate     | Fast              | Fast              |
| Final quality          | OK           | Good         | Good              | Best              |
| Used for LLMs?         | No           | No           | Legacy            | Yes — standard    |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Adam and AdamW — From Scratch": {
        "description": "Implement SGD, Adam, and AdamW from scratch in pure Python/NumPy. Trace every buffer update step-by-step and verify against PyTorch.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ADAM AND ADAMW — IMPLEMENTED FROM SCRATCH
================================================================================

Implements three optimisers with full state tracing:
    1. SGD (baseline)
    2. Adam (with bias correction)
    3. AdamW (decoupled weight decay)

Verifies against PyTorch built-ins and shows per-step buffer evolution.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── From-scratch implementations ──────────────────────────────────────────────

class SGDScratch:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr     = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * p.grad


class AdamScratch:
    """
    Adam with bias correction.
    Uses L2 regularisation via gradient (the WRONG way for weight decay).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        self.params       = list(params)
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        # State: one m and v buffer per parameter
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad
                if self.weight_decay != 0:
                    g = g + self.weight_decay * p   # L2: add λθ to gradient

                # Update biased first and second moment estimates
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Parameter update
                p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class AdamWScratch:
    """
    AdamW — decoupled weight decay.
    Weight decay applied directly to parameters, NOT through the gradient.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params       = list(params)
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad   # pure gradient — NO weight decay here

                # Update moment estimates (only on true gradient)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # ① Direct weight decay (decoupled — the key AdamW change)
                p *= (1 - self.lr * self.weight_decay)

                # ② Gradient-based update (adaptive)
                p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


# ── Verification against PyTorch ──────────────────────────────────────────────

def make_model_and_loss(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 1, bias=False),
    )
    x = torch.randn(32, 8)
    y = torch.randn(32, 1)
    return model, x, y


def run_steps(model, optimiser, x, y, n_steps=5):
    losses = []
    for _ in range(n_steps):
        optimiser.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    LR = 1e-3
    WD = 0.01

    print("=" * 60)
    print("  VERIFICATION: scratch vs PyTorch")
    print("=" * 60)

    # AdamW scratch vs PyTorch
    for name, scratch_cls, pt_cls, kwargs in [
        ("SGD",   SGDScratch,   torch.optim.SGD,
         {"lr": LR}),
        ("Adam",  AdamScratch,  torch.optim.Adam,
         {"lr": LR, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}),
        ("AdamW", AdamWScratch, torch.optim.AdamW,
         {"lr": LR, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": WD}),
    ]:
        model_a, x, y = make_model_and_loss(42)
        model_b, _, _ = make_model_and_loss(42)   # same init

        opt_a = scratch_cls(model_a.parameters(), **kwargs)
        opt_b = pt_cls(model_b.parameters(), **kwargs)

        losses_a = run_steps(model_a, opt_a, x, y, n_steps=10)
        losses_b = run_steps(model_b, opt_b, x, y, n_steps=10)

        # Compare final parameter values
        params_a = torch.cat([p.flatten() for p in model_a.parameters()])
        params_b = torch.cat([p.flatten() for p in model_b.parameters()])
        max_diff  = (params_a - params_b).abs().max().item()

        print(f"  {name:<8}  final loss: {losses_a[-1]:.6f} vs {losses_b[-1]:.6f}  "
              f"max param diff: {max_diff:.2e}  {'✓' if max_diff < 1e-4 else '✗'}")

    # Step-by-step trace for a single parameter
    print()
    print("=" * 60)
    print("  STEP-BY-STEP AdamW STATE TRACE (single scalar parameter)")
    print("=" * 60)
    print()

    theta = torch.tensor([1.0], requires_grad=True)
    β1, β2, lr, ε, λ = 0.9, 0.999, 0.1, 1e-8, 0.01
    m, v, t = 0.0, 0.0, 0

    # Simulate 6 gradient steps with a constant gradient of 0.5
    grads = [0.5, 0.4, 0.6, 0.3, 0.5, 0.4]

    print(f"  {'Step':>4}  {'g_t':>6}  {'m_t':>10}  {'v_t':>12}  "
          f"{'m̂_t':>10}  {'v̂_t':>10}  {'θ':>10}")
    print(f"  {'':─>4}  {'':─>6}  {'':─>10}  {'':─>12}  "
          f"{'':─>10}  {'':─>10}  {'':─>10}")

    theta_val = 1.0
    for g in grads:
        t   += 1
        m    = β1 * m + (1 - β1) * g
        v    = β2 * v + (1 - β2) * g**2
        m_hat = m  / (1 - β1**t)
        v_hat = v  / (1 - β2**t)
        # AdamW update
        theta_val = theta_val * (1 - lr * λ)   # weight decay
        theta_val = theta_val - lr * m_hat / (math.sqrt(v_hat) + ε)   # gradient step

        print(f"  {t:>4}  {g:>6.3f}  {m:>10.6f}  {v:>12.8f}  "
              f"{m_hat:>10.6f}  {v_hat:>10.8f}  {theta_val:>10.6f}")

    print()
    print("  Notice:")
    print("  • m_t starts near 0 (bias) → m̂_t corrects it to ≈ g_1 at step 1")
    print("  • v_t accumulates gradient² → √v̂_t grows → effective lr shrinks")
    print("  • θ decreases both from weight decay AND gradient step")
''',
    },

    "Adam vs AdamW: Weight Decay Comparison": {
        "description": "Empirically show the difference between Adam+L2 and AdamW weight decay — how they diverge for parameters with large vs small gradients.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ADAM vs ADAMW: WEIGHT DECAY COMPARISON
================================================================================

Demonstrates the key difference:
    • Adam + L2:   weight decay is ADAPTIVE (scaled by √v_t)
                   → parameters with large gradients get LESS regularisation
    • AdamW:       weight decay is UNIFORM
                   → all parameters decay at the same rate

This matters for LLM training where different parameters have very different
gradient magnitudes (e.g. embedding weights vs attention projection weights).
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def simulate_weight_decay_experiment():
    """
    Train a simple model with both Adam+L2 and AdamW, tracking the
    weight magnitudes of high-gradient vs low-gradient parameters.
    """
    torch.manual_seed(0)

    # Two-group model: group A has large gradients, group B has small gradients
    class TwoGroupModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Group A: heavily used (large gradients)
            self.group_a = nn.Linear(32, 32, bias=False)
            # Group B: rarely used (small gradients)
            self.group_b = nn.Linear(32, 1, bias=False)
            # Initialise both with same magnitude
            nn.init.normal_(self.group_a.weight, std=1.0)
            nn.init.normal_(self.group_b.weight, std=1.0)

        def forward(self, x, use_b_scale=1.0):
            a_out = self.group_a(x)
            # Group B input is scaled down → small gradients for group B
            b_out = self.group_b(a_out * use_b_scale)
            return b_out

    N_STEPS   = 500
    LR        = 1e-3
    WD        = 0.1
    B1, B2, EPS = 0.9, 0.999, 1e-8

    results = {}
    for variant in ["Adam+L2", "AdamW"]:
        model = TwoGroupModel()
        if variant == "Adam+L2":
            opt = torch.optim.Adam(model.parameters(), lr=LR,
                                   betas=(B1, B2), eps=EPS,
                                   weight_decay=WD)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                    betas=(B1, B2), eps=EPS,
                                    weight_decay=WD)

        history_a, history_b = [], []

        for step in range(N_STEPS):
            x = torch.randn(64, 32)
            y = torch.randn(64, 1)

            opt.zero_grad()
            # group_b gets 0.05× scaled input → very small gradients for group_b
            pred = model(x, use_b_scale=0.05)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()

            if step % 50 == 0:
                a_norm = model.group_a.weight.data.norm().item()
                b_norm = model.group_b.weight.data.norm().item()
                history_a.append(a_norm)
                history_b.append(b_norm)

        results[variant] = (history_a, history_b)

    return results


def print_weight_decay_comparison():
    print("=" * 65)
    print("  ADAM+L2 vs ADAMW: WEIGHT NORM COMPARISON")
    print("=" * 65)
    print()
    print("  Setup:")
    print("    group_a: large gradients (heavily updated)")
    print("    group_b: small gradients (rarely/weakly updated)")
    print("    weight_decay = 0.1")
    print()
    print("  Expectation:")
    print("    Adam+L2: group_a decays LESS (large √v̂ reduces effective λ)")
    print("    AdamW:   both groups decay at the SAME rate (uniform λ·lr)")
    print()

    results = simulate_weight_decay_experiment()

    steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    for variant, (hist_a, hist_b) in results.items():
        print(f"  {variant}")
        print(f"  {'Step':>6}  {'group_a (||W||)':>18}  {'group_b (||W||)':>18}  "
              f"{'ratio b/a':>12}")
        print(f"  {'':─>6}  {'':─>18}  {'':─>18}  {'':─>12}")
        for i, (a, b) in enumerate(zip(hist_a, hist_b)):
            ratio = b / (a + 1e-9)
            print(f"  {steps[i]:>6}  {a:>18.4f}  {b:>18.4f}  {ratio:>12.4f}")
        print()

    print("  Key insight:")
    print("  Adam+L2: group_b (small grads) decays FASTER than group_a")
    print("           because √v̂ is smaller → effective lr·λ/√v̂ is LARGER")
    print("           → MORE weight decay for rarely-updated parameters")
    print()
    print("  AdamW:   both groups decay at lr·λ = constant regardless of √v̂")
    print("           → UNIFORM regularisation as intended ✓")
    print()
    print("  For LLMs this matters: embedding weights (large grads from common tokens)")
    print("  vs weights for rare tokens get asymmetric regularisation with Adam+L2.")


def beta2_sensitivity_demo():
    """Show how different β₂ values affect convergence in early training."""
    print("=" * 65)
    print("  β₂ SENSITIVITY: 0.95 vs 0.999 in EARLY TRAINING")
    print("=" * 65)
    print()

    torch.manual_seed(42)
    model_base = nn.Sequential(nn.Linear(16, 64), nn.ReLU(),
                                nn.Linear(64, 1))

    x = torch.randn(128, 16)
    y = torch.randn(128, 1)

    print(f"  {'Step':>5}", end="")
    configs = [("β₂=0.99",  0.99), ("β₂=0.95",  0.95), ("β₂=0.999", 0.999)]
    for name, _ in configs:
        print(f"  {name:>12}", end="")
    print()
    print(f"  {'':─>5}" + "".join(f"  {'':─>12}" for _ in configs))

    # Run separate models for each β₂
    models = []
    opts   = []
    for name, b2 in configs:
        m = copy.deepcopy(model_base)
        o = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, b2),
                               weight_decay=0.01)
        models.append(m)
        opts.append(o)

    for step in range(1, 201):
        losses = []
        for m, o in zip(models, opts):
            o.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            o.step()
            losses.append(loss.item())

        if step in (1, 5, 10, 25, 50, 100, 200):
            print(f"  {step:>5}", end="")
            for l in losses:
                print(f"  {l:>12.5f}", end="")
            print()

    print()
    print("  Lower β₂ (0.95) is more responsive to recent gradients")
    print("  → faster early convergence but potentially noisier at end of training")
    print("  → recommended for LLM pre-training (GPT-3, Chinchilla, LLaMA-2 all use 0.95)")


if __name__ == "__main__":
    print_weight_decay_comparison()
    beta2_sensitivity_demo()
''',
    },

    "Optimiser Memory and Adafactor": {
        "description": "Quantify Adam's memory cost vs model weights, show how Adafactor reduces optimiser state to near-zero, and compare convergence.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
OPTIMISER MEMORY ANALYSIS AND ADAFACTOR
================================================================================

Adam stores 2 tensors per parameter (m and v). For a 7B model these buffers
alone require ~56 GB in fp32. Adafactor eliminates this overhead almost
entirely by using a factored, low-rank approximation of the second moment.

This demo:
    1. Computes exact memory cost for common model sizes
    2. Implements a simplified Adafactor (scalar + row/col factors)
    3. Compares convergence of AdamW vs Adafactor
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Memory cost analysis ───────────────────────────────────────────────────────

def memory_breakdown(n_params: int, dtype_model: str = "bf16",
                     dtype_optim: str = "fp32") -> dict:
    """
    Compute memory requirements for training a model of n_params parameters.

    Adam/AdamW store:
        - model weights   (dtype_model)
        - gradients       (dtype_model in bf16 training)
        - m buffer        (fp32 master copies)
        - v buffer        (fp32 master copies)

    Adafactor stores:
        - model weights   (dtype_model)
        - gradients       (dtype_model)
        - v_r: row factor per 2D weight  (fp32, shape d_out)
        - v_c: col factor per 2D weight  (fp32, shape d_in)
        → O(d_out + d_in) instead of O(d_out × d_in)
    """
    bytes_model = {"fp32": 4, "bf16": 2, "fp16": 2}[dtype_model]
    bytes_optim = {"fp32": 4, "bf16": 2}[dtype_optim]

    param_bytes = n_params * bytes_model
    grad_bytes  = n_params * bytes_model
    adam_m      = n_params * bytes_optim
    adam_v      = n_params * bytes_optim

    # Adafactor: for a weight matrix of shape (d_out, d_in),
    # stores d_out + d_in scalars instead of d_out × d_in
    # Rough estimate: 2 × sqrt(n_params) for factored 2D weights
    adafactor_v = 2 * int(math.sqrt(n_params)) * bytes_optim

    return {
        "params":         param_bytes,
        "grads":          grad_bytes,
        "adam_m":         adam_m,
        "adam_v":         adam_v,
        "total_adamw":    param_bytes + grad_bytes + adam_m + adam_v,
        "total_adafactor": param_bytes + grad_bytes + adafactor_v,
        "adafactor_v_approx": adafactor_v,
    }


def fmt_gb(b: int) -> str:
    return f"{b / 1e9:.1f} GB"


def print_memory_table():
    print("=" * 70)
    print("  OPTIMISER MEMORY COST BY MODEL SIZE (bf16 weights, fp32 optim)")
    print("=" * 70)
    print()
    print(f"  {'Model':<18} {'Params':>8}  {'Weights':>8}  "
          f"{'AdamW total':>12}  {'Adafactor total':>16}  {'Saving':>8}")
    print(f"  {'':─<18} {'':─>8}  {'':─>8}  {'':─>12}  {'':─>16}  {'':─>8}")

    models = [
        ("GPT-2 117M",   117_000_000),
        ("GPT-2 1.5B",  1_500_000_000),
        ("LLaMA-2 7B",  7_000_000_000),
        ("LLaMA-2 13B",13_000_000_000),
        ("LLaMA-2 70B",70_000_000_000),
    ]

    for name, n in models:
        mem = memory_breakdown(n)
        saving = (mem["total_adamw"] - mem["total_adafactor"]) / mem["total_adamw"] * 100
        print(f"  {name:<18} {n/1e9:>6.1f}B  "
              f"{fmt_gb(mem['params']):>8}  "
              f"{fmt_gb(mem['total_adamw']):>12}  "
              f"{fmt_gb(mem['total_adafactor']):>16}  "
              f"{saving:>6.0f}%")

    print()
    print("  * Adafactor estimate is approximate (assumes all weights are 2D).")
    print("  * In practice Adafactor reduces m+v from 2P to ~O(√P) parameters.")
    print("  * The savings grow with model size — most impactful for 70B+ models.")


# ── Simplified Adafactor implementation ───────────────────────────────────────

class AdafactorScratch:
    """
    Simplified Adafactor for 2D weight matrices.
    Uses factored second moment:  v ≈ v_r ⊗ v_c / v_1
    where v_r and v_c are row and column statistics, v_1 is a scalar.

    Memory: O(d_out + d_in) instead of O(d_out × d_in) for v.
    """

    def __init__(self, params, lr=None, eps=(1e-30, 1e-3),
                 clip_threshold=1.0, decay_rate=-0.8,
                 beta1=0.9, weight_decay=0.0):
        self.params          = list(params)
        self.lr              = lr
        self.eps1, self.eps2 = eps
        self.clip_threshold  = clip_threshold
        self.decay_rate      = decay_rate
        self.beta1           = beta1
        self.weight_decay    = weight_decay
        self.t               = 0
        # State dicts
        self.v_r = {}   # row factors
        self.v_c = {}   # column factors
        self.v_s = {}   # scalar (for 1D tensors)
        self.m   = {}   # first moment (optional, like Adam momentum)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def _get_lr(self):
        if self.lr is not None:
            return self.lr
        # Adafactor default: lr = 1/sqrt(t)  (no manual lr needed)
        return min(1e-2, 1.0 / math.sqrt(self.t))

    def step(self):
        self.t += 1
        rho = min(1 - 1e-8, 1 - self.t ** self.decay_rate)  # decay factor
        lr  = self._get_lr()

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad.float()

                if p.dim() >= 2:
                    # Factored second moment for 2D+ weights
                    # v_r: row-wise mean of g²,  v_c: col-wise mean of g²
                    g2 = g ** 2 + self.eps1

                    if i not in self.v_r:
                        self.v_r[i] = g2.mean(dim=-1)     # (d_out,)
                        self.v_c[i] = g2.mean(dim=-2)     # (d_in,)
                    else:
                        self.v_r[i] = rho * self.v_r[i] + (1 - rho) * g2.mean(dim=-1)
                        self.v_c[i] = rho * self.v_c[i] + (1 - rho) * g2.mean(dim=-2)

                    # Reconstruct v̂: outer product normalised
                    v_hat = torch.outer(self.v_r[i], self.v_c[i])
                    v_hat = v_hat / (v_hat.mean() + self.eps1)
                    rms_g = (g2 / (v_hat + self.eps1)).sqrt()
                else:
                    # Scalar second moment for 1D tensors
                    g2 = g ** 2 + self.eps1
                    if i not in self.v_s:
                        self.v_s[i] = g2.mean()
                    else:
                        self.v_s[i] = rho * self.v_s[i] + (1 - rho) * g2.mean()
                    rms_g = (g2 / (self.v_s[i] + self.eps1)).sqrt()

                # Update (clip by RMS)
                rms  = rms_g.norm() / math.sqrt(rms_g.numel())
                clip = max(1.0, rms / self.clip_threshold)
                u    = g / (rms_g * clip + self.eps1)

                # Weight decay
                p.data *= (1 - lr * self.weight_decay)
                p.data -= lr * u.to(p.dtype)


# ── Convergence comparison ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print_memory_table()

    print()
    print("=" * 70)
    print("  CONVERGENCE COMPARISON: AdamW vs Adafactor")
    print("=" * 70)
    print()

    torch.manual_seed(0)
    # Simple regression model
    model_base = nn.Sequential(
        nn.Linear(32, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    x  = torch.randn(256, 32)
    y  = torch.randn(256, 1)

    configs = [
        ("AdamW",      lambda m: torch.optim.AdamW(
            m.parameters(), lr=1e-3, weight_decay=0.01)),
        ("Adafactor",  lambda m: AdafactorScratch(
            m.parameters(), lr=1e-3, weight_decay=0.01)),
    ]

    import copy
    print(f"  {'Step':>5}", end="")
    for name, _ in configs:
        print(f"  {name:>14}", end="")
    print()
    print(f"  {'':─>5}" + "".join(f"  {'':─>14}" for _ in configs))

    models_opts = [(copy.deepcopy(model_base), factory(copy.deepcopy(model_base)))
                   for _, factory in configs]
    # Fix: create models first, then optimisers
    models_opts = []
    for name, factory in configs:
        m = copy.deepcopy(model_base)
        models_opts.append((m, factory(m)))

    for step in range(1, 301):
        losses = []
        for m, o in models_opts:
            o.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            o.step()
            losses.append(loss.item())

        if step in (1, 10, 25, 50, 100, 200, 300):
            print(f"  {step:>5}", end="")
            for l in losses:
                print(f"  {l:>14.5f}", end="")
            print()

    print()
    print("  Adafactor achieves comparable convergence to AdamW")
    print("  while storing O(√P) optimiser state instead of O(P).")
    print("  This is why T5 and PaLM use Adafactor for their largest models.")
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
    visual_html   = ""
    visual_height = 600
    # try:
    #     from llm_training.visuals.optimizers import (
    #         OPTIM_VISUAL_HTML,
    #         OPTIM_VISUAL_HEIGHT,
    #     )
    #     visual_html   = OPTIM_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = OPTIM_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[07_optimizers_adam_adamw.py] Could not load visual: {e}", stacklevel=2)

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